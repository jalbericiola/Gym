# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import Request, Response

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.episode_checkpoint import (
    CheckpointRevisionConflict,
    EpisodeCheckpoint,
    EpisodeLeaseUnavailable,
    FilesystemCheckpointStore,
    FilesystemCheckpointStoreConfig,
    MidEpisodeResumeConfig,
    SessionSnapshot,
    request_fingerprint,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig, SimpleAgentRunRequest


class FakeResponse:
    def __init__(self, payload, cookies=None):
        self.payload = payload
        self.cookies = cookies or {}
        self.ok = True
        self.content = self

    async def read(self):
        return json.dumps(self.payload).encode()


def model_response(output):
    return {
        "id": "response-id",
        "created_at": 1.0,
        "model": "policy",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 3,
        },
    }


def config(checkpoint_dir: Path) -> SimpleAgentConfig:
    return SimpleAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        model_server=ModelServerRef(type="responses_api_models", name="model"),
        mid_episode_resume=MidEpisodeResumeConfig(
            enabled=True,
            checkpoint_store=FilesystemCheckpointStoreConfig(root_dir=checkpoint_dir),
        ),
    )


def make_request(headers=None, cookies=None) -> Request:
    raw_headers = [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
    if cookies:
        raw_headers.append((b"cookie", "; ".join(f"{key}={value}" for key, value in cookies.items()).encode()))
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/run",
            "raw_path": b"/run",
            "query_string": b"",
            "headers": raw_headers,
            "client": ("test", 123),
            "server": ("test", 80),
        }
    )


async def test_resume_after_completed_tool_does_not_repeat_tool(tmp_path: Path) -> None:
    calls = []
    fail_next_model = True
    agent = None

    async def post(*, server_name, url_path, json=None, cookies=None, headers=None):
        nonlocal fail_next_model
        calls.append((server_name, url_path))
        if server_name == "agent" and url_path == "/v1/responses":
            api_response = Response()
            result = await agent.responses(
                make_request(headers=headers, cookies=cookies),
                api_response,
                json,
            )
            return FakeResponse(result.model_dump(mode="json"), {"resource-session": "active"})
        if url_path == "/seed_session":
            return FakeResponse({}, {"resource-session": "seeded"})
        if url_path == "/checkpoint_session":
            count = 4 if ("resources", "/increment_counter") in calls else 3
            return FakeResponse(
                {
                    "format_name": "example_session_state_mgmt.counter",
                    "payload": {"count": count},
                },
                {"resource-session": "active"},
            )
        if url_path == "/restore_session":
            assert json["snapshot"]["payload"] == {"count": 4}
            return FakeResponse({}, {"resource-session": "restored"})
        if url_path == "/increment_counter":
            assert json == {"count": 1}
            return FakeResponse({"success": True}, {"resource-session": "active"})
        if server_name == "model" and url_path == "/v1/responses":
            model_call_count = calls.count(("model", "/v1/responses"))
            if model_call_count == 1:
                return FakeResponse(
                    model_response(
                        [
                            {
                                "type": "function_call",
                                "id": "fc-1",
                                "call_id": "call-1",
                                "name": "increment_counter",
                                "arguments": '{"count":1}',
                                "status": "completed",
                            }
                        ]
                    ),
                    {"model-session": "one"},
                )
            if fail_next_model:
                fail_next_model = False
                raise ConnectionError("simulated Agent/model boundary failure")
            assert [item.type for item in json.input] == ["message", "function_call", "function_call_output"]
            return FakeResponse(
                model_response(
                    [
                        {
                            "type": "message",
                            "id": "message-1",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "The count is 4.", "annotations": []}],
                        }
                    ]
                ),
                {"model-session": "two"},
            )
        if url_path == "/verify":
            return FakeResponse(json | {"reward": 1.0})
        raise AssertionError(url_path)

    server_client = MagicMock(spec=ServerClient)
    server_client.post.side_effect = post
    server_client.get.return_value = FakeResponse(
        {
            "supported": True,
            "format_name": "example_session_state_mgmt.counter",
            "format_version": 1,
        }
    )
    agent = SimpleAgent(config=config(tmp_path), server_client=server_client)
    body = SimpleAgentRunRequest.model_validate(
        {
            "_ng_episode_id": "job:row-1:generation-0",
            "_ng_policy_version": "weights-107",
            "initial_count": 3,
            "expected_count": 4,
            "responses_create_params": {"input": [{"role": "user", "content": "add one"}]},
        }
    )

    with pytest.raises(ConnectionError, match="simulated"):
        await agent.run(make_request(), body)

    checkpoint = FilesystemCheckpointStore(tmp_path).load("job:row-1:generation-0")
    assert checkpoint.phase == "ready_for_model"
    assert checkpoint.resource_snapshot.payload == {"count": 4}
    assert [item["type"] for item in checkpoint.outputs] == ["function_call", "function_call_output"]

    result = await agent.run(make_request(), body)
    assert result.reward == 1.0
    assert calls.count(("resources", "/seed_session")) == 1
    assert calls.count(("resources", "/increment_counter")) == 1
    assert calls.count(("resources", "/restore_session")) == 1

    calls_before_cached_retry = len(calls)
    cached = await agent.run(make_request(), body)
    assert cached == result
    assert len(calls) == calls_before_cached_retry


async def test_filesystem_store_round_trip_archive_and_expiry(tmp_path: Path) -> None:
    store = FilesystemCheckpointStore(tmp_path)
    request = {"responses_create_params": {"input": "hello"}}
    checkpoint = EpisodeCheckpoint(
        episode_id="episode/unsafe:path",
        request_fingerprint=request_fingerprint(request),
        run_request=request,
        phase="ready_for_model",
        resource_snapshot=SessionSnapshot(format_name="counter", payload={"count": 3}),
    )
    store.save(checkpoint)
    assert checkpoint.revision == 1
    assert store.load(checkpoint.episode_id) == checkpoint
    assert "/" not in store.checkpoint_path(checkpoint.episode_id).name

    stale = checkpoint.model_copy(deep=True)
    checkpoint.step = 1
    store.save(checkpoint)
    with pytest.raises(CheckpointRevisionConflict):
        store.save(stale)

    async with store.lease(checkpoint.episode_id):
        with pytest.raises(EpisodeLeaseUnavailable):
            async with store.lease(checkpoint.episode_id):
                pass

    archived = store.archive(checkpoint, "policy_mismatch")
    assert archived.exists()
    assert store.load(checkpoint.episode_id) is None

    checkpoint.status = "completed"
    checkpoint.phase = "completed"
    checkpoint.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    store.save(checkpoint)
    assert store.load(checkpoint.episode_id) is None
