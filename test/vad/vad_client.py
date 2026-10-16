
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from celery import Celery


REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "mysecurepassword")

BROKER_URL = (
    f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
)
BACKEND_URL = (
    f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/1"
)

QUEUE_NAME = "vad_queue"

# Must match common.job_store.JOBS_DIR in the consumer environment.
PROJECT_ROOT = Path(
    "/mnt/data/workspace/freelance/voice/queue_stt"
)

JOBS_DIR = Path(
    os.getenv("JOBS_DIR", str(PROJECT_ROOT / "data" / "jobs"))
).resolve()


client = Celery(
    "vad_client",
    broker=BROKER_URL,
    backend=BACKEND_URL,
)

client.conf.update(
    task_default_queue=QUEUE_NAME,
    task_ignore_result=True,
)


def read_job(job_id: str) -> dict | None:
    """Read a job JSON file without importing common.job_store."""
    if not job_id or Path(job_id).name != job_id:
        raise ValueError(f"Invalid job ID: {job_id!r}")

    path = JOBS_DIR / f"{job_id}.json"

    if not path.is_file():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def submit_vad(audio_path: str, job_id: str | None = None) -> str:
    """Submit VAD asynchronously and return its task ID."""
    path = Path(audio_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    task_id = str(uuid4())
    job_id = str(job_id or task_id)

    # For a standalone task, the consumer's update_task() creates
    # the job JSON automatically when the task begins processing.
    # For an existing pipeline, pass its existing job_id.

    client.send_task(
        "detect",
        args=[
            {
                "task_id": task_id,
                "job_id": job_id,
                "audio_path": str(path),
            }
        ],
        task_id=task_id,
        queue=QUEUE_NAME,
        ignore_result=True,
    )

    print(f"Task ID: {task_id}")
    print(f"Job ID:  {job_id}")

    return task_id


def get_status(task_id: str, job_id: str) -> dict:
    """Read the individual task status from the job JSON."""
    job = read_job(job_id)

    if job is None:
        return {
            "job_id": job_id,
            "task_id": task_id,
            "status": "unknown",
        }

    task = job.get("tasks", {}).get(task_id)

    if task is None:
        return {
            "job_id": job_id,
            "task_id": task_id,
            "status": "unknown",
            "job_status": job.get("status"),
        }

    return {
        "job_id": job_id,
        "job_status": job.get("status"),
        **task,
    }


def get_result(task_id: str, job_id: str) -> dict:
    """Return the VAD result stored in the job JSON."""
    task = get_status(task_id, job_id)

    if task.get("status") != "completed":
        raise RuntimeError(
            f"Task {task_id} is not completed: "
            f"{task.get('status', 'unknown')}"
        )

    result = task.get("result")

    if result is None:
        raise KeyError(f"No result stored for task {task_id}")

    return result


def wait_for_result(
    task_id: str,
    job_id: str,
    poll_interval: float = 1.0,
    timeout: float = 600.0,
) -> dict:
    """Poll until the task completes, fails, or times out."""
    deadline = time.monotonic() + timeout

    while True:
        task = get_status(task_id, job_id)
        status = task.get("status", "unknown")

        if status == "completed":
            return get_result(task_id, job_id)

        if status == "failed":
            raise RuntimeError(
                f"VAD failed: {task.get('error', 'Unknown error')}"
            )

        remaining = deadline - time.monotonic()

        if remaining <= 0:
            raise TimeoutError(
                f"Timed out waiting for task {task_id}; "
                f"last status: {status}"
            )

        time.sleep(min(poll_interval, remaining))


if __name__ == "__main__":
    audio_path = (
        "/mnt/data/workspace/freelance/voice/diar/"
        "vad_dataset/audio/blab_0001.wav"
    )

    task_id = submit_vad(audio_path)
    job_id = task_id

    print("Initial status:", get_status(task_id, job_id))

    # Uncomment to wait for the result:
    result = wait_for_result(task_id, job_id)
    print(json.dumps(result, indent=2))
