# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable state used to resume a multi-turn episode after process failure."""

import asyncio
import fcntl
import hashlib
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class SessionSnapshot(BaseModel):
    """Versioned, resource-server-owned representation of an environment session."""

    protocol_version: int = 1
    format_name: str
    format_version: int = 1
    kind: Literal["inline", "reference"] = "inline"
    payload: Any


class RestoreSessionRequest(BaseModel):
    snapshot: SessionSnapshot


class RestoreSessionResponse(BaseModel):
    pass


class SessionCheckpointingCapability(BaseModel):
    supported: bool
    protocol_version: int = 1
    format_name: Optional[str] = None
    format_version: Optional[int] = None


class DiscardSessionResponse(BaseModel):
    pass


class FilesystemCheckpointStoreConfig(BaseModel):
    type: Literal["filesystem"] = "filesystem"
    root_dir: Path


class MidEpisodeResumeConfig(BaseModel):
    enabled: bool = False
    require_policy_version: bool = True
    checkpoint_store: Optional[FilesystemCheckpointStoreConfig] = None
    completed_ttl_seconds: int = Field(default=7 * 24 * 60 * 60, ge=0)

    @model_validator(mode="after")
    def require_store_when_enabled(self) -> "MidEpisodeResumeConfig":
        if self.enabled and self.checkpoint_store is None:
            raise ValueError("checkpoint_store is required when mid-episode resume is enabled")
        return self


class EpisodeLeaseUnavailable(RuntimeError):
    pass


class CheckpointRevisionConflict(RuntimeError):
    pass


class InvalidCheckpoint(RuntimeError):
    pass


class EpisodeCheckpoint(BaseModel):
    schema_version: int = 1
    revision: int = 0
    status: Literal["in_progress", "completed"] = "in_progress"
    episode_id: str
    request_fingerprint: str
    policy_version: Optional[str] = None
    execution_token: Optional[str] = None
    run_request: dict[str, Any]
    outputs: list[dict[str, Any]] = Field(default_factory=list)
    usage: Optional[dict[str, Any]] = None
    step: int = 0
    phase: Literal["ready_for_model", "ready_for_tool", "ready_for_verify", "completed"]
    next_tool_index: int = 0
    last_model_response: Optional[dict[str, Any]] = None
    model_server_cookies: dict[str, str] = Field(default_factory=dict)
    resource_snapshot: SessionSnapshot
    final_result: Optional[dict[str, Any]] = None
    resume_count: int = 0
    restart_count: int = 0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None


def request_fingerprint(run_request: dict[str, Any]) -> str:
    """Hash request semantics, excluding resume-control metadata."""

    request = {key: value for key, value in run_request.items() if key not in {"_ng_episode_id", "_ng_policy_version"}}
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


class FilesystemCheckpointStore:
    """Atomic JSON persistence and an advisory, per-episode execution lease."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.episodes_dir = root_dir / "episodes"
        self.archive_dir = root_dir / "archive"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(episode_id: str) -> str:
        return hashlib.sha256(episode_id.encode()).hexdigest()

    def checkpoint_path(self, episode_id: str) -> Path:
        key = self._key(episode_id)
        return self.episodes_dir / key[:2] / f"{key}.json"

    def lock_path(self, episode_id: str) -> Path:
        key = self._key(episode_id)
        return self.episodes_dir / key[:2] / f"{key}.lock"

    @asynccontextmanager
    async def lease(self, episode_id: str) -> AsyncIterator[None]:
        lock_path = self.lock_path(episode_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = lock_path.open("a+")
        try:
            try:
                await asyncio.to_thread(fcntl.flock, lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise EpisodeLeaseUnavailable(f"Episode {episode_id!r} is already running") from error
            yield
        finally:
            await asyncio.to_thread(fcntl.flock, lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    def load(self, episode_id: str) -> Optional[EpisodeCheckpoint]:
        path = self.checkpoint_path(episode_id)
        if not path.exists():
            return None
        try:
            checkpoint = EpisodeCheckpoint.model_validate_json(path.read_text())
        except Exception as error:
            raise InvalidCheckpoint(f"Checkpoint {path} is corrupt") from error
        if checkpoint.schema_version != 1:
            raise InvalidCheckpoint(f"Checkpoint {path} uses unsupported schema version {checkpoint.schema_version}")
        if checkpoint.episode_id != episode_id:
            raise InvalidCheckpoint(f"Checkpoint {path} contains the wrong episode ID")
        if (
            checkpoint.status == "completed"
            and checkpoint.expires_at is not None
            and checkpoint.expires_at <= datetime.now(timezone.utc)
        ):
            path.unlink(missing_ok=True)
            return None
        return checkpoint

    def save(self, checkpoint: EpisodeCheckpoint) -> None:
        path = self.checkpoint_path(checkpoint.episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            current = self.load(checkpoint.episode_id)
            if current is not None and current.revision != checkpoint.revision:
                raise CheckpointRevisionConflict(
                    f"Expected revision {checkpoint.revision}, found {current.revision} for {checkpoint.episode_id!r}"
                )
        checkpoint.revision += 1
        checkpoint.updated_at = datetime.now(timezone.utc)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        serialized = checkpoint.model_dump_json(indent=2)
        with temporary.open("w") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def archive(self, checkpoint: EpisodeCheckpoint, reason: str) -> Path:
        source = self.checkpoint_path(checkpoint.episode_id)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = self.archive_dir / f"{self._key(checkpoint.episode_id)}.{timestamp}.{reason}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        return destination
