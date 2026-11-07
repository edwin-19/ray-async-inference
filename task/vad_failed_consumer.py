
import logging
import os
from urllib.parse import quote
from uuid import UUID

from dotenv import load_dotenv
from ray import serve
from ray.serve.schema import CeleryAdapterConfig, TaskProcessorConfig
from ray.serve.task_consumer import task_consumer, task_handler

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vad.failed_consumer")


def redis_url(db: int) -> str:
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD", "")

    if password:
        return f"redis://:{quote(password, safe='')}@{host}:{port}/{db}"

    return f"redis://{host}:{port}/{db}"


REDIS_BROKER_URL = redis_url(0)
REDIS_BACKEND_URL = redis_url(1)

VAD_FAILED_QUEUE = os.getenv(
    "VAD_FAILED_QUEUE",
    "vad_failed_queue",
)

VAD_FAILED_MAX_RETRIES = int(
    os.getenv("VAD_FAILED_MAX_RETRIES", "3")
)

processor_config = TaskProcessorConfig(
    queue_name=VAD_FAILED_QUEUE,
    adapter_config=CeleryAdapterConfig(
        broker_url=REDIS_BROKER_URL,
        backend_url=REDIS_BACKEND_URL,
    ),
    max_retries=VAD_FAILED_MAX_RETRIES,
)


@serve.deployment(
    name="vad-failed-consumer",
    max_ongoing_requests=5,
    ray_actor_options={
        "num_cpus": float(os.getenv("VAD_FAILED_NUM_CPUS", "0.2")),
    },
)
@task_consumer(task_processor_config=processor_config)
class VADFailedConsumer:

    def __init__(self):
        logger.info(
            "VAD failed consumer listening on queue=%s",
            VAD_FAILED_QUEUE,
        )

    @task_handler(name="process")
    def process(self, task_id, exception_msg, args, kwargs, einfo):
        from sqlmodel import Session

        from db.db import engine
        from db.models import Job, JobStage, JobStatus, utc_now

        # Original VAD task was dispatched with:
        # args=[{"job_id": "..."}]
        try:
            if not isinstance(args, (list, tuple)) or not args:
                raise ValueError("Missing original task arguments")

            payload = args[0]

            if not isinstance(payload, dict):
                raise ValueError("Invalid original task payload")

            raw_job_id = payload.get("job_id")

            if not raw_job_id:
                raise ValueError("Missing job_id in original task payload")

            job_id = UUID(str(raw_job_id))

        except (ValueError, TypeError, AttributeError):
            logger.exception(
                "Cannot extract job_id from failed VAD task_id=%s args=%r",
                task_id,
                args,
            )
            return {
                "task_id": task_id,
                "status": "invalid_payload",
            }

        error_message = str(
            exception_msg or "VAD task exhausted retries"
        )[:4000]

        logger.error(
            "VAD task exhausted retries job_id=%s task_id=%s error=%s",
            job_id,
            task_id,
            error_message,
        )

        if einfo:
            logger.error("VAD failure traceback: %s", einfo)

        with Session(engine) as session:
            job = session.get(Job, job_id)

            if job is None:
                logger.error("Job not found for failed VAD task: %s", job_id)
                return {
                    "job_id": str(job_id),
                    "status": "not_found",
                }

            if job.status == JobStatus.COMPLETED:
                logger.warning(
                    "Job already completed; ignoring stale VAD failure: %s",
                    job_id,
                )
                return {
                    "job_id": str(job_id),
                    "status": "already_completed",
                }

            job.status = JobStatus.FAILED
            job.stage = JobStage.VAD
            job.error = error_message
            job.updated_at = utc_now()

            session.add(job)
            session.commit()

        logger.info("Marked VAD job as FAILED: %s", job_id)

        return {
            "job_id": str(job_id),
            "status": "failed",
        }


app = VADFailedConsumer.bind()
