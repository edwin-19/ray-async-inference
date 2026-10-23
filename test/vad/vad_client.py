
import os
import json
import time
import uuid

from celery import Celery

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "mysecurepassword")

BROKER_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
BACKEND_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/1"

QUEUE_NAME = "vad_queue"
AUDIO_PATH = "/mnt/data/workspace/freelance/voice/diar/vad_dataset/combined/combined_2h.wav"

client = Celery(
    "vad_client",
    broker=BROKER_URL,
    backend=BACKEND_URL,
)


def main():
    if not os.path.isfile(AUDIO_PATH):
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

    job_id = f"vad-{uuid.uuid4().hex[:12]}"

    print(f"Submitting VAD job: {job_id}")
    print(f"Audio: {AUDIO_PATH}")
    start = time.perf_counter()

    result = client.send_task(
        "process",
        args=[{
            "job_id": job_id,
            "audio_path": AUDIO_PATH,
        }],
        queue=QUEUE_NAME,
    )

    print(f"Task ID: {result.id}")
    print("Waiting for inference result...")

    try:
        output = result.get(
            timeout=300,
            propagate=True,
        )
    except Exception:
        print(f"VAD task failed or timed out: {result.id}")
        raise

    elapsed = time.perf_counter() - start

    print(f"\nCompleted in {elapsed:.3f}s")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
