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
from typing import Dict

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

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


class StatefulCounterResourcesServerConfig(BaseResourcesServerConfig):
    pass


class IncrementCounterRequest(BaseModel):
    count: int


class IncrementCounterResponse(BaseModel):
    success: bool


class GetCounterValueResponse(BaseModel):
    count: int


class StatefulCounterVerifyRequest(BaseVerifyRequest):
    expected_count: int


class BaseVerifyResponse(BaseVerifyRequest):
    reward: float


class StatefulCounterSeedSessionRequest(BaseSeedSessionRequest):
    initial_count: int


class StatefulCounterResourcesServer(SimpleResourcesServer):
    config: StatefulCounterResourcesServerConfig
    session_id_to_counter: Dict[str, int] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/increment_counter")(self.increment_counter)
        app.post("/get_counter_value")(self.get_counter_value)

        return app

    async def seed_session(self, request: Request, body: StatefulCounterSeedSessionRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        self.session_id_to_counter.setdefault(session_id, body.initial_count)
        return BaseSeedSessionResponse()

    async def checkpoint_session(self, request: Request) -> SessionSnapshot:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self.session_id_to_counter:
            raise RuntimeError("Cannot checkpoint an unseeded counter session")
        return SessionSnapshot(
            format_name="example_session_state_mgmt.counter",
            payload={"count": self.session_id_to_counter[session_id]},
        )

    async def session_checkpointing(self) -> SessionCheckpointingCapability:
        return SessionCheckpointingCapability(
            supported=True,
            format_name="example_session_state_mgmt.counter",
            format_version=1,
        )

    async def restore_session(self, request: Request, body: RestoreSessionRequest) -> RestoreSessionResponse:
        if body.snapshot.format_name != "example_session_state_mgmt.counter":
            raise ValueError(f"Unsupported snapshot format: {body.snapshot.format_name}")
        if body.snapshot.format_version != 1:
            raise ValueError(f"Unsupported counter snapshot version: {body.snapshot.format_version}")
        session_id = request.session[SESSION_ID_KEY]
        self.session_id_to_counter[session_id] = int(body.snapshot.payload["count"])
        return RestoreSessionResponse()

    async def discard_session(self, request: Request) -> DiscardSessionResponse:
        self.session_id_to_counter.pop(request.session[SESSION_ID_KEY], None)
        return DiscardSessionResponse()

    async def increment_counter(self, request: Request, body: IncrementCounterRequest) -> IncrementCounterResponse:
        session_id = request.session[SESSION_ID_KEY]
        counter = self.session_id_to_counter.setdefault(session_id, 0)

        counter += body.count

        self.session_id_to_counter[session_id] = counter

        return IncrementCounterResponse(success=True)

    async def get_counter_value(self, request: Request) -> GetCounterValueResponse:
        session_id = request.session[SESSION_ID_KEY]
        counter = self.session_id_to_counter.setdefault(session_id, 0)
        return GetCounterValueResponse(count=counter)

    async def verify(self, request: Request, body: StatefulCounterVerifyRequest) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]

        reward = 0.0
        if session_id in self.session_id_to_counter:
            counter = self.session_id_to_counter[session_id]
            reward = float(body.expected_count == counter)

        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    StatefulCounterResourcesServer.run_webserver()
