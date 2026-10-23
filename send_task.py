
import os
import sys
import logging
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from celery import Celery
from dotenv import load_dotenv
from sqlmodel import Session

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_task")


def redis_url(db: int) -> str:
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD", "")

    if password:
        return f"redis://:{quote(password, safe='')}@{host}:{port}/{db}"

    return f"redis://{host}:{port}/{db}"


BROKER_URL = redis_url(0)
BACKEND_URL = redis_url(1)
MAIN_QUEUE = os.getenv("MAIN_QUEUE", "main_queue")


def create_job(audio_path: str) -> str:
    # Import at runtime so the script uses the project's DB configuration.
    from db.db import engine
    from db.models import Job, JobStatus, JobStage, utc_now

    job_id = uuid4()
    now = utc_now()

    job = Job(
        id=job_id,
        audio_path=audio_path,
        status=JobStatus.QUEUED,
        stage=JobStage.MAIN,
        created_at=now,
        updated_at=now,
    )

    with Session(engine) as session:
        session.add(job)
        session.commit()

    logger.info("Created job: %s", job_id)
    return str(job_id)


def send_task(job_id: str) -> str:
    celery_app = Celery(
        "speech_pipeline_test",
        broker=BROKER_URL,
        backend=BACKEND_URL,
    )

    try:
        result = celery_app.send_task(
            "process",
            args=[{"job_id": job_id}],
            queue=MAIN_QUEUE,
        )
        logger.info(
            "Published task_id=%s to queue=%s",
            result.id,
            MAIN_QUEUE,
        )
        return result.id
    finally:
        celery_app.close()


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <audio_path>")
        sys.exit(1)

    audio_path = str(Path(sys.argv[1]).expanduser().resolve())

    if not Path(audio_path).is_file():
        print(f"Error: audio file not found: {audio_path}")
        sys.exit(1)

    # 1. Insert job into PostgreSQL.
    job_id = create_job(audio_path)

    # 2. Publish job ID to the main queue.
    try:
        task_id = send_task(job_id)
    except Exception:
        logger.exception(
            "Failed to publish job_id=%s. "
            "The database job remains queued.",
            job_id,
        )
        sys.exit(1)

    print("\nJob submitted successfully")
    print(f"  Job ID:    {job_id}")
    print(f"  Audio:     {audio_path}")
    print(f"  Task ID:   {task_id}")
    print(f"  Queue:     {MAIN_QUEUE}")


if __name__ == "__main__":
    main()
