
import logging
import os
from urllib.parse import quote
from uuid import UUID

from dotenv import load_dotenv
from celery import Celery
from ray import serve
from ray.serve.config import AutoscalingConfig
from ray.serve.schema import CeleryAdapterConfig, TaskProcessorConfig
from ray.serve.task_consumer import task_consumer, task_handler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("main.consumer")


def redis_url(db: int) -> str:
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD", "")

    if password:
        return f"redis://:{quote(password, safe='')}@{host}:{port}/{db}"

    return f"redis://{host}:{port}/{db}"


REDIS_BROKER_URL = redis_url(0)
REDIS_BACKEND_URL = redis_url(1)

MAIN_QUEUE_NAME = os.getenv("MAIN_QUEUE", "main_queue")
VAD_QUEUE_NAME = os.getenv("VAD_QUEUE", "vad_queue")

processor_config = TaskProcessorConfig(
    queue_name=MAIN_QUEUE_NAME,
    adapter_config=CeleryAdapterConfig(
        broker_url=REDIS_BROKER_URL,
        backend_url=REDIS_BACKEND_URL,
    ),
)


@serve.deployment(
    name="main-consumer",
    max_ongoing_requests=8,
    ray_actor_options={"num_cpus": 1},
    autoscaling_config=AutoscalingConfig(
        min_replicas=1,
        max_replicas=2,
        target_ongoing_requests=4,
    ),
)
@task_consumer(task_processor_config=processor_config)
class MainConsumer:

    @task_handler(name="process")
    def process(self, data):
        from sqlalchemy.exc import SQLAlchemyError
        from sqlmodel import Session

        from db.db import engine
        from db.models import Job, JobStage, JobStatus, utc_now

        if not isinstance(data, dict):
            raise TypeError("Task payload must be a dictionary")

        raw_job_id = data.get("job_id")
        if not raw_job_id:
            raise ValueError("Missing job_id")

        try:
            job_id = UUID(str(raw_job_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(
                f"Invalid job_id: {raw_job_id!r}"
            ) from exc

        logger.info("Received job_id=%s", job_id)

        try:
            with Session(engine) as session:
                job = session.get(Job, job_id)

                if job is None:
                    raise ValueError(f"Job not found: {job_id}")

                if job.status == JobStatus.COMPLETED:
                    return {
                        "job_id": str(job_id),
                        "status": "already_completed",
                    }

                if job.status == JobStatus.FAILED:
                    return {
                        "job_id": str(job_id),
                        "status": "already_failed",
                    }

                audio_path = job.audio_path
                if not audio_path:
                    raise ValueError(
                        f"Job has no audio_path: {job_id}"
                    )

                job.status = JobStatus.PROCESSING
                job.stage = JobStage.VAD

                if job.started_at is None:
                    job.started_at = utc_now()

                job.updated_at = utc_now()
                session.add(job)
                session.commit()

            # Create the publisher at runtime, not at module import.
            publisher = Celery(
                "speech_pipeline_publisher",
                broker=REDIS_BROKER_URL,
                backend=REDIS_BACKEND_URL,
            )

            try:
                result = publisher.send_task(
                    "process",
                    args=[{
                        "job_id": str(job_id),
                        "audio_path": audio_path,
                    }],
                    queue=VAD_QUEUE_NAME,
                )
            finally:
                publisher.close()

            logger.info(
                "Dispatched job_id=%s to queue=%s task_id=%s",
                job_id,
                VAD_QUEUE_NAME,
                result.id,
            )

            return {
                "job_id": str(job_id),
                "status": "dispatched",
                "next_queue": VAD_QUEUE_NAME,
                "task_id": result.id,
            }

        except (ValueError, TypeError):
            raise
        except SQLAlchemyError:
            logger.exception("Database error for job_id=%s", job_id)
            raise
        except Exception:
            logger.exception("Failed to dispatch job_id=%s", job_id)
            raise


app = MainConsumer.bind()
