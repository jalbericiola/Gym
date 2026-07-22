import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping


def _warn(log_prefix: str, message: str) -> None:
    print(f"WARNING: {log_prefix}: {message}", flush=True)


@contextmanager
def _metrics_file_lock(
    metrics_fpath: Path,
    *,
    timeout_seconds: float = 60.0,
    stale_seconds: float = 300.0,
):
    lock_path = metrics_fpath.with_name(f".{metrics_fpath.name}.lockdir")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    acquired = False

    while not acquired:
        try:
            lock_path.mkdir()
            acquired = True
        except FileExistsError:
            try:
                lock_age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if lock_age > stale_seconds:
                shutil.rmtree(lock_path, ignore_errors=True)
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for metrics file lock at {lock_path}")
            time.sleep(0.05)

    try:
        yield
    finally:
        if acquired:
            shutil.rmtree(lock_path, ignore_errors=True)


def read_json_metrics_file(
    metrics_fpath: str | Path,
    *,
    default: Mapping[str, Any] | None = None,
    log_prefix: str = "nemo_gym",
) -> dict[str, Any]:
    metrics_path = Path(metrics_fpath)
    fallback = dict(default or {})

    try:
        with _metrics_file_lock(metrics_path):
            if not metrics_path.exists():
                return fallback
            raw_metrics = metrics_path.read_text()
            if not raw_metrics.strip():
                return fallback
            loaded = json.loads(raw_metrics)
            if not isinstance(loaded, dict):
                _warn(
                    log_prefix,
                    f"read_json_metrics_file: expected object in metrics file "
                    f"(path={metrics_path}), got {type(loaded).__name__}",
                )
                return fallback
            return loaded
    except Exception as e:
        _warn(
            log_prefix,
            f"read_json_metrics_file: error reading metrics file "
            f"(path={metrics_path}): {type(e).__name__} {e}",
        )
        return fallback


def update_json_metrics_file(
    metrics_fpath: str | Path,
    update_dict: Mapping[str, Any] | None = None,
    *,
    increments: Mapping[str, float | int | None] | None = None,
    log_prefix: str = "nemo_gym",
) -> bool:
    metrics_path = Path(metrics_fpath)
    tmp_fpath: Path | None = None

    try:
        with _metrics_file_lock(metrics_path):
            existing_dict: dict[str, Any] = {}
            if metrics_path.exists():
                raw_metrics = metrics_path.read_text()
                if raw_metrics.strip():
                    try:
                        loaded = json.loads(raw_metrics)
                        if isinstance(loaded, dict):
                            existing_dict = loaded
                        else:
                            _warn(
                                log_prefix,
                                f"update_json_metrics_file: expected object in metrics file "
                                f"(path={metrics_path}), got {type(loaded).__name__}; resetting metrics",
                            )
                    except Exception as e:
                        _warn(
                            log_prefix,
                            f"update_json_metrics_file: error reading metrics file "
                            f"(path={metrics_path}): {type(e).__name__} {e}; resetting metrics",
                        )

            existing_dict = {k: v for k, v in existing_dict.items() if v is not None}
            update_dict = {k: v for k, v in (update_dict or {}).items() if v is not None}

            for key, delta in (increments or {}).items():
                if delta is None:
                    continue
                existing_dict[key] = existing_dict.get(key, 0.0) + delta

            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{metrics_path.name}.",
                suffix=f".tmp.{os.getpid()}",
                dir=metrics_path.parent,
                text=True,
            )
            tmp_fpath = Path(tmp_name)
            with os.fdopen(fd, "w") as f:
                json.dump(existing_dict | update_dict, f)
            os.replace(tmp_fpath, metrics_path)
            return True
    except Exception as e:
        _warn(
            log_prefix,
            f"update_json_metrics_file: error updating metrics file "
            f"(path={metrics_path}): {type(e).__name__} {e}",
        )
        if tmp_fpath is not None:
            tmp_fpath.unlink(missing_ok=True)
        return False
