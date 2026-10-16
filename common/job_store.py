import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import fcntl


PROJECT_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = Path(
    os.getenv("JOBS_DIR", str(PROJECT_ROOT / "data" / "jobs"))
).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_path(job_id: str) -> Path:
    # Prevent IDs from being used as arbitrary filesystem paths.
    if not job_id or Path(job_id).name != job_id:
        raise ValueError(f"Invalid job ID: {job_id!r}")

    return JOBS_DIR / f"{job_id}.json"


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def create_job(
    job_id: str,
    *,
    pipeline: str = "inference",
    input_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = _job_path(job_id)
    lock_path = path.with_suffix(".lock")
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

        try:
            if path.exists():
                raise FileExistsError(f"Job already exists: {job_id}")

            now = utc_now()
            job = {
                "job_id": job_id,
                "pipeline": pipeline,
                "status": "queued",
                "input_data": input_data or {},
                "tasks": {},
                "result": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }

            _write_atomic(path, job)
            return job
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def get_job(job_id: str) -> dict[str, Any] | None:
    path = _job_path(job_id)
    lock_path = path.with_suffix(".lock")

    if not path.exists():
        return None

    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)

        try:
            if not path.exists():
                return None

            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def update_task(
    job_id: str,
    task_id: str,
    *,
    task_type: str,
    status: str,
    **fields: Any,
) -> dict[str, Any]:
    allowed_statuses = {
        "queued",
        "processing",
        "completed",
        "failed",
    }
    if status not in allowed_statuses:
        raise ValueError(f"Invalid task status: {status}")

    path = _job_path(job_id)
    lock_path = path.with_suffix(".lock")
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    job = json.load(f)
            else:
                # Compatibility fallback for existing clients that
                # submit tasks without creating a job first.
                now = utc_now()
                job = {
                    "job_id": job_id,
                    "pipeline": "inference",
                    "status": "queued",
                    "input_data": {},
                    "tasks": {},
                    "result": None,
                    "error": None,
                    "created_at": now,
                    "updated_at": now,
                }

            tasks = job.setdefault("tasks", {})
            task = tasks.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "task_type": task_type,
                    "created_at": utc_now(),
                },
            )

            task.update(fields)
            task["task_type"] = task_type
            task["status"] = status
            task["updated_at"] = utc_now()

            # A task completion is not necessarily pipeline completion:
            # ASR or other stages may still need to run.
            if status in {"processing", "completed"}:
                job["status"] = "processing"
                job["error"] = None
            elif status == "failed":
                job["status"] = "failed"
                job["error"] = task.get("error")

            job["updated_at"] = utc_now()
            _write_atomic(path, job)

            return job
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)