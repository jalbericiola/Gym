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
from io import StringIO
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.episode_checkpoint import (
    DiscardSessionResponse,
    RestoreSessionRequest,
    RestoreSessionResponse,
    SessionCheckpointingCapability,
    SessionSnapshot,
)
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.workplace_assistant.utils import get_tools, is_correct


_WORKPLACE_TOOLKITS = (
    "email",
    "calendar",
    "analytics",
    "project_management",
    "customer_relationship_manager",
)

# Tool functions are bound methods, so they cannot be serialized directly.
# Recreate each tool container on restore, then replace only its durable state.
_WORKPLACE_DATAFRAME_STATE = {
    "company_directory": ("_emails",),
    "email": ("_emails",),
    "calendar": ("_calendar_events",),
    "analytics": ("_analytics_data", "_plots_data"),
    "project_management": ("_project_tasks",),
    "customer_relationship_manager": ("_crm_data",),
}
_WORKPLACE_SNAPSHOT_FORMAT = "workplace_assistant.tool_state"
_WORKPLACE_SNAPSHOT_VERSION = 1


def _dataframe_to_snapshot(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Serialize a DataFrame with its schema so string identifiers survive."""

    return json.loads(dataframe.to_json(orient="table", index=False))


def _dataframe_from_snapshot(payload: dict[str, Any]) -> pd.DataFrame:
    return pd.read_json(StringIO(json.dumps(payload)), orient="table")


class WorkbenchResourcesServerConfig(BaseResourcesServerConfig):
    pass


class WorkbenchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class WorkbenchResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class WorkbenchVerifyRequest(BaseVerifyRequest):
    ground_truth: list[Dict[str, str]] | str
    id: int
    category: str
    environment_name: str


class WorkbenchVerifyResponse(BaseVerifyResponse):
    pass


class WorkbenchResourcesServer(SimpleResourcesServer):
    config: WorkbenchResourcesServerConfig
    session_id_to_tool_env: Dict[str, Any] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{path}")(self.route_to_python_function)
        return app

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        # init session once for each sample.
        session_id = request.session[SESSION_ID_KEY]
        self.session_id_to_tool_env[session_id] = get_tools(_WORKPLACE_TOOLKITS)
        return BaseSeedSessionResponse()

    async def checkpoint_session(self, request: Request) -> SessionSnapshot:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self.session_id_to_tool_env:
            raise RuntimeError("Cannot checkpoint an unseeded workplace session")

        containers = self.session_id_to_tool_env[session_id]["containers"]
        dataframes = {
            container_name: {
                field_name: _dataframe_to_snapshot(getattr(containers[container_name], field_name))
                for field_name in field_names
            }
            for container_name, field_names in _WORKPLACE_DATAFRAME_STATE.items()
        }
        return SessionSnapshot(
            format_name=_WORKPLACE_SNAPSHOT_FORMAT,
            format_version=_WORKPLACE_SNAPSHOT_VERSION,
            payload={"dataframes": dataframes},
        )

    async def session_checkpointing(self) -> SessionCheckpointingCapability:
        return SessionCheckpointingCapability(
            supported=True,
            format_name=_WORKPLACE_SNAPSHOT_FORMAT,
            format_version=_WORKPLACE_SNAPSHOT_VERSION,
        )

    async def restore_session(self, request: Request, body: RestoreSessionRequest) -> RestoreSessionResponse:
        snapshot = body.snapshot
        if snapshot.format_name != _WORKPLACE_SNAPSHOT_FORMAT:
            raise ValueError(f"Unsupported workplace snapshot format: {snapshot.format_name}")
        if snapshot.format_version != _WORKPLACE_SNAPSHOT_VERSION:
            raise ValueError(f"Unsupported workplace snapshot version: {snapshot.format_version}")
        if not isinstance(snapshot.payload, dict) or not isinstance(snapshot.payload.get("dataframes"), dict):
            raise ValueError("Workplace snapshot is missing DataFrame state")

        tool_env = get_tools(_WORKPLACE_TOOLKITS)
        containers = tool_env["containers"]
        snapshot_dataframes = snapshot.payload["dataframes"]
        for container_name, field_names in _WORKPLACE_DATAFRAME_STATE.items():
            container_payload = snapshot_dataframes.get(container_name)
            if not isinstance(container_payload, dict):
                raise ValueError(f"Workplace snapshot is missing container {container_name!r}")
            for field_name in field_names:
                dataframe_payload = container_payload.get(field_name)
                if not isinstance(dataframe_payload, dict):
                    raise ValueError(f"Workplace snapshot is missing DataFrame {container_name}.{field_name}")
                setattr(
                    containers[container_name],
                    field_name,
                    _dataframe_from_snapshot(dataframe_payload),
                )

        self.session_id_to_tool_env[request.session[SESSION_ID_KEY]] = tool_env
        return RestoreSessionResponse()

    async def discard_session(self, request: Request) -> DiscardSessionResponse:
        self.session_id_to_tool_env.pop(request.session[SESSION_ID_KEY], None)
        return DiscardSessionResponse()

    async def route_to_python_function(self, path: str, body: WorkbenchRequest, request: Request) -> WorkbenchResponse:
        session_id = request.session[SESSION_ID_KEY]

        # Check if session exists
        if session_id not in self.session_id_to_tool_env:
            raise HTTPException(
                status_code=400,
                detail="Session not initialized. Please call seed_session first.",
            )

        tool_env = self.session_id_to_tool_env[session_id]
        args = {key: value for key, value in body.model_dump(exclude_unset=True).items() if value is not None}

        try:
            function = tool_env["functions"][path]
            result = function(**args)
            return WorkbenchResponse(output=result)
        except Exception as e:
            return WorkbenchResponse(
                output=f"Error executing tool '{path}': {str(e)}"
            )  # return error to model so that it can correct itself

    async def verify(self, body: WorkbenchVerifyRequest) -> WorkbenchVerifyResponse:
        ground_truth = body.ground_truth
        response = body.response.output

        total_score = 0.0

        # Convert list of ResponseFunctionToolCall objects into list of dictionaries
        predicted_function_calls = []

        for message in response:
            if message.type == "function_call":
                predicted_function_calls.append(message.model_dump())

        predicted_chat_content = []

        for message in response:
            if message.type == "output_text":
                predicted_chat_content.append(message.model_dump())

        total_score += is_correct(predicted_function_calls, ground_truth, None) * 1.0
        return WorkbenchVerifyResponse(**body.model_dump(), reward=total_score)


if __name__ == "__main__":
    WorkbenchResourcesServer.run_webserver()
