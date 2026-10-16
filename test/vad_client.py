
import json
import os
from datetime import datetime, timezone
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

# Must match vad.py.
BASE_DIR = Path(
    "/mnt/data/workspace/freelance/voice/queue_stt"
)

STATUS_DIR = Path(
    os.getenv("VAD_STATUS_DIR", str(BASE_DIR / "vad_status"))
).resolve()

RESULTS_DIR = Path(
    os.getenv("VAD_RESULTS_DIR", str(BASE_DIR / "vad_results"))
).resolve()


client = Celery(
    "vad_client",
    broker=BROKER_URL,
    backend=BACKEND_URL,
)

client.conf.task_default_queue = QUEUE_NAME


def write_status(task_id: str, status: str, **extra) -> None:
    """Write the initial client-side status."""
    STATUS_DIR.mkdir(parents=True, exist_ok=True)

    path = STATUS_DIR / f"{task_id}.json"
    temp_path = path.with_name(
        f"{path.stem}.{os.getpid()}.tmp"
    )

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task_id": task_id,
                "status": status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **extra,
            },
            f,
            indent=2,
        )

    temp_path.replace(path)


def submit_vad(audio_path: str) -> str:
    """Submit VAD and return immediately with a task ID."""
    path = Path(audio_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {path}"
        )

    task_id = str(uuid4())

    # Persist queued status before publishing the task.
    write_status(
        task_id,
        "queued",
        audio_path=str(path),
    )

    try:
        client.send_task(
            "detect",
            args=[
                {
                    "task_id": task_id,
                    "audio_path": str(path),
                }
            ],
            task_id=task_id,
            queue=QUEUE_NAME,
            ignore_result=True,
        )
    except Exception as exc:
        write_status(
            task_id,
            "failed",
            audio_path=str(path),
            error=str(exc),
        )
        raise

    print(f"Task submitted: {task_id}")
    print(f"Status file: {STATUS_DIR / f'{task_id}.json'}")

    return task_id


def get_status(task_id: str) -> dict:
    """Read the latest status from the shared status directory."""
    path = STATUS_DIR / f"{task_id}.json"

    if not path.is_file():
        return {
            "task_id": task_id,
            "status": "unknown",
            "status_file": str(path),
        }

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_result(task_id: str) -> dict:
    """Retrieve the completed VAD result."""
    status = get_status(task_id)

    if status["status"] != "completed":
        raise RuntimeError(
            f"Task {task_id} is not completed: "
            f"{status['status']}"
        )

    result_path = Path(status["result_path"])

    if not result_path.is_file():
        raise FileNotFoundError(
            f"Result file not found: {result_path}"
        )

    with result_path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # audio_path = (
    #     "/mnt/data/workspace/freelance/voice/diar/"
    #     "vad_dataset/audio/blab_0001.wav"
    # )

    # task_id = submit_vad(audio_path)

    # # This is expected to return queued or processing initially.
    # print("Initial status:", get_status(task_id))
    # print("Task ID:", task_id)

    # Later, from another script or terminal:
    print(get_status('e666cd48-218e-4313-84c4-c6b6ba3bd2aa'))
    # print(get_result(task_id))
