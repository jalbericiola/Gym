# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, List, Optional

from fastapi import HTTPException, Request, Response
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.episode_checkpoint import (
    EpisodeCheckpoint,
    EpisodeLeaseUnavailable,
    FilesystemCheckpointStore,
    MidEpisodeResumeConfig,
    RestoreSessionRequest,
    SessionCheckpointingCapability,
    SessionSnapshot,
    request_fingerprint,
)
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


class SimpleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None
    max_output_tokens_per_step: Optional[int] = None
    max_total_seq_length: Optional[int] = None
    mid_episode_resume: MidEpisodeResumeConfig = Field(default_factory=MidEpisodeResumeConfig)


class SimpleAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


@dataclass
class ResumeRunContext:
    store: FilesystemCheckpointStore
    episode_id: str
    policy_version: Optional[str]
    request_json: dict[str, Any]
    request_fingerprint: str
    execution_token: str
    checkpoint: Optional[EpisodeCheckpoint] = None
    restart_count: int = 0


@dataclass
class ResumeResponsesContext:
    store: FilesystemCheckpointStore
    checkpoint: EpisodeCheckpoint


class SimpleAgent(SimpleResponsesAPIAgent):
    config: SimpleAgentConfig

    _output_adapter: ClassVar[TypeAdapter] = TypeAdapter(List[NeMoGymResponseOutputItem])

    _RESUME_EPISODE_HEADER: ClassVar[str] = "X-NeMo-Gym-Resume-Episode"
    _RESUME_TOKEN_HEADER: ClassVar[str] = "X-NeMo-Gym-Resume-Token"

    @staticmethod
    def _resume_cookie_values(cookies: Any) -> dict[str, str]:
        if not cookies:
            return {}
        return {key: value.value if hasattr(value, "value") else str(value) for key, value in cookies.items()}

    async def _resume_checkpoint_resources(self, cookies: Any) -> tuple[SessionSnapshot, Any]:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/checkpoint_session",
            json={},
            cookies=cookies,
        )
        await raise_for_status(response)
        return SessionSnapshot.model_validate(await get_response_json(response)), response.cookies

    async def _resume_require_checkpointing_capability(self) -> None:
        response = await self.server_client.get(
            server_name=self.config.resources_server.name,
            url_path="/session_checkpointing",
        )
        await raise_for_status(response)
        capability = SessionCheckpointingCapability.model_validate(await get_response_json(response))
        if not capability.supported:
            raise RuntimeError(
                f"Resources server {self.config.resources_server.name!r} does not support session snapshots"
            )

    @asynccontextmanager
    async def _resume_lease(self, resume: Optional[ResumeRunContext]):
        if resume is None:
            yield
            return
        try:
            async with resume.store.lease(resume.episode_id):
                yield
        except EpisodeLeaseUnavailable as error:
            raise HTTPException(
                status_code=409,
                detail=str(error),
                headers={"Retry-After": "1"},
            ) from error

    async def _resume_restore_resources(self, snapshot: SessionSnapshot) -> Any:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/restore_session",
            json=RestoreSessionRequest(snapshot=snapshot).model_dump(mode="json"),
            cookies=None,
        )
        await raise_for_status(response)
        return response.cookies

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        resume = self._resume_responses_context(request)
        if resume:
            checkpoint = resume.checkpoint
            new_outputs = self._output_adapter.validate_python(checkpoint.outputs)
            usage = NeMoGymResponseUsage.model_validate(checkpoint.usage) if checkpoint.usage else None
            step = checkpoint.step
            model_server_cookies = checkpoint.model_server_cookies or None
            model_response = (
                NeMoGymResponse.model_validate(checkpoint.last_model_response)
                if checkpoint.last_model_response
                else None
            )
        else:
            new_outputs = []
            usage = None
            step = 0
            model_server_cookies = None  # update the cookies on every model response
            model_response = None
        resources_server_cookies = request.cookies  # update the cookies on every resources server response

        while True:
            if resume and resume.checkpoint.phase == "ready_for_verify":
                break

            if not resume or resume.checkpoint.phase == "ready_for_model":
                step += 1
                new_body = body.model_copy(update={"input": body.input + new_outputs})
                # Per-env static cap (max_output_tokens_per_step) takes priority — it
                # exists for multi-turn coherence (force the model to make one decision,
                # see the tool response, then continue). When unset, defer to the
                # inference engine's auto-clamp at dynamic_engine.py:1155-1158, which
                # sets num_tokens_to_generate = max_sequence_length - len(prompt_tokens).
                # max_total_seq_length is the signal to opt into that behavior.
                if self.config.max_output_tokens_per_step is not None:
                    cap = self.config.max_output_tokens_per_step
                    current = new_body.max_output_tokens
                    new_body = new_body.model_copy(update={"max_output_tokens": min(current, cap) if current else cap})
                elif self.config.max_total_seq_length is not None and new_body.max_output_tokens is None:
                    # Only rows WITHOUT their own cap defer to the engine auto-clamp.
                    # This branch used to null unconditionally, which sent capped rows
                    # (e.g. rlhf max_output_tokens=16384) to the full ~131k generation
                    # room; a non-EOSing sample then decodes for ~an hour and the wave
                    # tail stalls (v6 smoke straggler: engine reqs asking 130,9xx
                    # tokens). The engine's block-aware generation-room clamp makes
                    # row caps safe near the boundary, so preserving them is correct.
                    pass

                model_api_response = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path="/v1/responses",
                    json=new_body,
                    cookies=model_server_cookies,
                )
                # We raise for status here since we expect model calls to always work.
                await raise_for_status(model_api_response)
                model_response_json = await get_response_json(model_api_response)
                model_server_cookies = model_api_response.cookies
                try:
                    model_response = NeMoGymResponse.model_validate(model_response_json)
                except ValidationError as e:
                    raise RuntimeError(
                        f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                    ) from e

                output = model_response.output
                new_outputs.extend(output)

                if not usage:
                    usage = model_response.usage
                    model_response.usage = None

                if usage and model_response.usage:
                    usage.input_tokens += model_response.usage.input_tokens
                    usage.output_tokens += model_response.usage.output_tokens
                    usage.total_tokens += model_response.usage.total_tokens

                    # TODO support more advanced token details
                    usage.input_tokens_details.cached_tokens = 0
                    usage.output_tokens_details.reasoning_tokens = 0

                all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
                all_output_messages: List[NeMoGymResponseOutputMessage] = [
                    o for o in output if o.type == "message" and o.role == "assistant"
                ]

                if resume:
                    await self._resume_after_model(
                        resume=resume,
                        new_outputs=new_outputs,
                        usage=usage,
                        step=step,
                        model_response=model_response,
                        model_cookies=model_server_cookies,
                        has_function_calls=bool(all_fn_calls),
                        finished=bool(model_response.incomplete_details or (not all_fn_calls and all_output_messages)),
                    )
            else:
                if resume.checkpoint.phase != "ready_for_tool" or model_response is None:
                    raise RuntimeError(f"Invalid resumable episode phase: {resume.checkpoint.phase}")
                output = model_response.output
                all_fn_calls = [o for o in output if o.type == "function_call"]
                all_output_messages = [o for o in output if o.type == "message" and o.role == "assistant"]

            if model_response.incomplete_details:
                break

            if not all_fn_calls and all_output_messages:
                break

            first_tool_index = resume.checkpoint.next_tool_index if resume else 0
            for tool_index, output_function_call in enumerate(
                all_fn_calls[first_tool_index:],
                start=first_tool_index,
            ):
                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=json.loads(output_function_call.arguments),
                    cookies=resources_server_cookies,
                )
                # We don't raise for status here since it's a valid return for the API to error e.g. if the model outputs an invalid call or something.
                resources_server_cookies = api_response.cookies

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=(await api_response.content.read()).decode(),
                )
                new_outputs.append(tool_response)

                if resume:
                    resources_server_cookies = await self._resume_after_tool(
                        resume=resume,
                        new_outputs=new_outputs,
                        usage=usage,
                        step=step,
                        model_response=model_response,
                        model_cookies=model_server_cookies,
                        resources_cookies=resources_server_cookies,
                        next_tool_index=tool_index + 1,
                        tool_count=len(all_fn_calls),
                    )

            # Check if max steps is not None and if we have exhausted it.
            if self.config.max_steps and step >= self.config.max_steps:
                break

        # Propogate any extra cookies necessary for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)

        model_response.output = new_outputs
        model_response.usage = usage
        return model_response

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        resume = self._resume_prepare(body) if self.config.mid_episode_resume.enabled else None

        async with self._resume_lease(resume):
            if resume:
                self._resume_load_checkpoint(resume)
                cached_result = self._resume_cached_result(resume)
                if cached_result is not None:
                    return cached_result

            cookies = request.cookies

            if resume and resume.checkpoint is not None:
                cookies = await self._resume_restore(resume)
            else:
                if resume:
                    await self._resume_require_checkpointing_capability()
                seed_session_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/seed_session",
                    json=body.model_dump(),
                    cookies=cookies,
                )
                await raise_for_status(seed_session_response)
                cookies = seed_session_response.cookies
                if resume:
                    cookies = await self._resume_after_seed(resume, cookies)

            resume_request_options = {"headers": self._resume_headers(resume)} if resume else {}
            response = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=body.responses_create_params,
                cookies=cookies,
                **resume_request_options,
            )
            await raise_for_status(response)
            cookies = response.cookies

            verify_request = SimpleAgentVerifyRequest.model_validate(
                body.model_dump() | {"response": await get_response_json(response)}
            )

            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            result = SimpleAgentVerifyResponse.model_validate(await get_response_json(verify_response))

            if resume:
                self._resume_complete(resume, result)

            return result

    def _resume_prepare(self, body: SimpleAgentRunRequest) -> ResumeRunContext:
        resume_config = self.config.mid_episode_resume
        if resume_config.checkpoint_store is None:
            raise RuntimeError("mid_episode_resume.checkpoint_store is required when resume is enabled")

        request_json = body.model_dump(mode="json")
        episode_id = request_json.get("_ng_episode_id")
        policy_version = request_json.get("_ng_policy_version")
        if not episode_id:
            raise HTTPException(status_code=422, detail="_ng_episode_id is required for mid-episode resume")
        if resume_config.require_policy_version and not policy_version:
            raise HTTPException(status_code=422, detail="_ng_policy_version is required for mid-episode resume")

        return ResumeRunContext(
            store=FilesystemCheckpointStore(resume_config.checkpoint_store.root_dir),
            episode_id=episode_id,
            policy_version=policy_version,
            request_json=request_json,
            request_fingerprint=request_fingerprint(request_json),
            execution_token=secrets.token_urlsafe(32),
        )

    def _resume_load_checkpoint(self, resume: ResumeRunContext) -> None:
        checkpoint = resume.store.load(resume.episode_id)
        if checkpoint is not None:
            if checkpoint.request_fingerprint != resume.request_fingerprint:
                raise HTTPException(
                    status_code=409,
                    detail="Episode ID was reused with a different run request",
                )
            if checkpoint.policy_version != resume.policy_version:
                resume.restart_count = checkpoint.restart_count + 1
                resume.store.archive(checkpoint, "policy_mismatch")
                checkpoint = None
        resume.checkpoint = checkpoint

    def _resume_cached_result(self, resume: ResumeRunContext) -> Optional[SimpleAgentVerifyResponse]:
        checkpoint = resume.checkpoint
        if checkpoint is None or checkpoint.status != "completed":
            return None
        return SimpleAgentVerifyResponse.model_validate(checkpoint.final_result)

    async def _resume_restore(
        self,
        resume: ResumeRunContext,
    ) -> Any:
        checkpoint = resume.checkpoint
        if checkpoint is None:
            raise RuntimeError("Cannot restore an episode without a checkpoint")
        cookies = await self._resume_restore_resources(checkpoint.resource_snapshot)
        checkpoint.resume_count += 1
        checkpoint.execution_token = resume.execution_token

        resume.store.save(checkpoint)
        return cookies

    async def _resume_after_seed(self, resume: ResumeRunContext, cookies: Any) -> Any:
        snapshot, cookies = await self._resume_checkpoint_resources(cookies)
        checkpoint = EpisodeCheckpoint(
            episode_id=resume.episode_id,
            request_fingerprint=resume.request_fingerprint,
            policy_version=resume.policy_version,
            execution_token=resume.execution_token,
            run_request=resume.request_json,
            phase="ready_for_model",
            resource_snapshot=snapshot,
            restart_count=resume.restart_count,
        )
        resume.checkpoint = checkpoint

        resume.store.save(checkpoint)
        return cookies

    def _resume_headers(self, resume: ResumeRunContext) -> dict[str, str]:
        return {
            self._RESUME_EPISODE_HEADER: resume.episode_id,
            self._RESUME_TOKEN_HEADER: resume.execution_token,
        }

    def _resume_responses_context(self, request: Request) -> Optional[ResumeResponsesContext]:
        episode_id = request.headers.get(self._RESUME_EPISODE_HEADER)
        execution_token = request.headers.get(self._RESUME_TOKEN_HEADER)
        if episode_id is None and execution_token is None:
            return None
        if not episode_id or not execution_token:
            raise HTTPException(status_code=400, detail="Incomplete internal resume headers")

        checkpoint_store = self.config.mid_episode_resume.checkpoint_store
        if not self.config.mid_episode_resume.enabled or checkpoint_store is None:
            raise HTTPException(status_code=400, detail="Mid-episode resume is not enabled")
        store = FilesystemCheckpointStore(checkpoint_store.root_dir)
        checkpoint = store.load(episode_id)
        if checkpoint is None or checkpoint.execution_token != execution_token:
            raise HTTPException(status_code=409, detail="Invalid or stale resume execution token")
        if checkpoint.status != "in_progress":
            raise HTTPException(status_code=409, detail="Episode is not resumable")
        return ResumeResponsesContext(store=store, checkpoint=checkpoint)

    async def _resume_after_model(
        self,
        resume: ResumeResponsesContext,
        new_outputs: list[NeMoGymResponseOutputItem],
        usage: Optional[NeMoGymResponseUsage],
        step: int,
        model_response: NeMoGymResponse,
        model_cookies: Any,
        has_function_calls: bool,
        finished: bool,
    ) -> None:
        checkpoint = resume.checkpoint
        checkpoint.next_tool_index = 0
        if finished:
            checkpoint.phase = "ready_for_verify"
        elif has_function_calls:
            checkpoint.phase = "ready_for_tool"
        elif self.config.max_steps and step >= self.config.max_steps:
            checkpoint.phase = "ready_for_verify"
        else:
            checkpoint.phase = "ready_for_model"
        self._resume_update_checkpoint(
            checkpoint,
            new_outputs,
            usage,
            step,
            model_response,
            model_cookies,
        )
        resume.store.save(checkpoint)

    async def _resume_after_tool(
        self,
        resume: ResumeResponsesContext,
        new_outputs: list[NeMoGymResponseOutputItem],
        usage: Optional[NeMoGymResponseUsage],
        step: int,
        model_response: NeMoGymResponse,
        model_cookies: Any,
        resources_cookies: Any,
        next_tool_index: int,
        tool_count: int,
    ) -> Any:
        snapshot, resources_cookies = await self._resume_checkpoint_resources(resources_cookies)
        checkpoint = resume.checkpoint
        checkpoint.resource_snapshot = snapshot
        checkpoint.next_tool_index = next_tool_index
        if next_tool_index < tool_count:
            checkpoint.phase = "ready_for_tool"
        elif self.config.max_steps and step >= self.config.max_steps:
            checkpoint.phase = "ready_for_verify"
        else:
            checkpoint.phase = "ready_for_model"
        self._resume_update_checkpoint(
            checkpoint,
            new_outputs,
            usage,
            step,
            model_response,
            model_cookies,
        )
        resume.store.save(checkpoint)
        return resources_cookies

    def _resume_complete(
        self,
        resume: ResumeRunContext,
        result: SimpleAgentVerifyResponse,
    ) -> None:
        checkpoint = resume.store.load(resume.episode_id)
        if checkpoint is None or checkpoint.execution_token != resume.execution_token:
            raise RuntimeError("Cannot complete an episode without a checkpoint")
        checkpoint.status = "completed"
        checkpoint.phase = "completed"
        checkpoint.execution_token = None
        checkpoint.final_result = result.model_dump(mode="json")
        checkpoint.expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=self.config.mid_episode_resume.completed_ttl_seconds
        )
        resume.store.save(checkpoint)

    def _resume_update_checkpoint(
        self,
        checkpoint: EpisodeCheckpoint,
        outputs: list[NeMoGymResponseOutputItem],
        usage: Optional[NeMoGymResponseUsage],
        step: int,
        last_model_response: NeMoGymResponse,
        model_cookies: Any,
    ) -> None:
        checkpoint.outputs = [item.model_dump(mode="json") for item in outputs]
        checkpoint.usage = usage.model_dump(mode="json") if usage else None
        checkpoint.step = step
        checkpoint.last_model_response = last_model_response.model_dump(mode="json")
        checkpoint.model_server_cookies = self._resume_cookie_values(model_cookies)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    SimpleAgent.run_webserver()
