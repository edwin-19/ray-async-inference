
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
logger = logging.getLogger("asr.failed_consumer")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ASR_FAILED_QUEUE = os.getenv(
    "ASR_FAILED_QUEUE",
    "asr_failed_queue",
)

ASR_FAILED_MAX_RETRIES = int(
    os.getenv("ASR_FAILED_MAX_RETRIES", "3")
)


def redis_url(db: int) -> str:
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD", "")

    if password:
        return (
            f"redis://:{quote(password, safe='')}"
            f"@{host}:{port}/{db}"
        )

    return f"redis://{host}:{port}/{db}"


REDIS_BROKER_URL = redis_url(0)
REDIS_BACKEND_URL = redis_url(1)


# This consumer reads the ASR dead-letter queue.
#
# It does not configure another failed-task queue.
# If database persistence fails, raise the exception so the
# consumer's own retry mechanism can retry the DB operation.

processor_config = TaskProcessorConfig(
    queue_name=ASR_FAILED_QUEUE,
    adapter_config=CeleryAdapterConfig(
        broker_url=REDIS_BROKER_URL,
        backend_url=REDIS_BACKEND_URL,
    ),
    max_retries=ASR_FAILED_MAX_RETRIES,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def extract_job_id(args, kwargs):
    """
    Extract job_id from the original failed task arguments.

    Original ASR dispatch:

        publisher.send_task(
            "process",
            args=[{"job_id": str(job_id)}],
            queue=ASR_QUEUE,
        )
    """

    payload = None

    # Original positional arguments.
    if isinstance(args, (list, tuple)) and args:
        payload = args[0]

    # Support keyword-based task submissions too.
    if payload is None and isinstance(kwargs, dict):
        payload = kwargs.get("data")

        if payload is None:
            payload = kwargs.get("payload")

        if payload is None and kwargs.get("job_id"):
            return UUID(str(kwargs["job_id"]))

    if not isinstance(payload, dict):
        raise ValueError(
            f"Cannot extract job_id from failed task arguments: "
            f"args={args}, kwargs={kwargs}"
        )

    raw_job_id = payload.get("job_id")

    if not raw_job_id:
        raise ValueError(
            f"Failed task payload has no job_id: {payload}"
        )

    return UUID(str(raw_job_id))


# ---------------------------------------------------------------------
# Failed ASR consumer
# ---------------------------------------------------------------------

@serve.deployment(
    name="asr-failed-consumer",
    max_ongoing_requests=5,
    ray_actor_options={
        "num_cpus": 0.2,
    },
)
@task_consumer(task_processor_config=processor_config)
class ASRFailedConsumer:

    def __init__(self):
        logger.info(
            "ASR failed-task consumer started. queue=%s",
            ASR_FAILED_QUEUE,
        )

    @task_handler(name="process")
    def process(
        self,
        task_id,
        exception_msg,
        args,
        kwargs,
        einfo,
    ):
        from sqlmodel import Session

        from db.db import engine
        from db.models import (
            Job,
            JobStage,
            JobStatus,
            utc_now,
        )

        logger.error(
            "ASR task exhausted retries. task_id=%s error=%s",
            task_id,
            exception_msg,
        )

        # -------------------------------------------------------------
        # 1. Extract original job ID
        # -------------------------------------------------------------

        try:
            job_id = extract_job_id(args, kwargs)

        except Exception:
            logger.exception(
                "Could not extract job_id from failed ASR task. "
                "task_id=%s args=%s kwargs=%s",
                task_id,
                args,
                kwargs,
            )

            # This is an invalid failed-task payload, not a DB error.
            # Return successfully to acknowledge it.
            return {
                "task_id": str(task_id),
                "status": "unprocessable",
                "reason": "Could not extract job_id",
            }

        # -------------------------------------------------------------
        # 2. Build error message
        # -------------------------------------------------------------

        error_message = str(
            exception_msg or "ASR task exhausted retries"
        )

        if einfo:
            logger.error(
                "ASR failure traceback for job=%s:\n%s",
                job_id,
                einfo,
            )

        error_message = error_message[:4000]

        # -------------------------------------------------------------
        # 3. Persist terminal failure
        # -------------------------------------------------------------

        with Session(engine) as session:

            job = session.get(Job, job_id)

            if job is None:
                logger.error(
                    "Cannot mark missing job as failed. job_id=%s",
                    job_id,
                )

                return {
                    "job_id": str(job_id),
                    "task_id": str(task_id),
                    "status": "not_found",
                }

            # Never overwrite an already completed job.
            if job.status == JobStatus.COMPLETED:

                logger.warning(
                    "Ignoring stale ASR failure notification. "
                    "Job already completed: %s",
                    job_id,
                )

                return {
                    "job_id": str(job_id),
                    "task_id": str(task_id),
                    "status": "already_completed",
                }

            job.status = JobStatus.FAILED
            job.stage = JobStage.ASR
            job.error = error_message
            job.updated_at = utc_now()

            session.add(job)

            # If this fails, let the exception propagate so the
            # failed consumer can retry the database operation.
            session.commit()

        logger.error(
            "ASR job permanently marked FAILED. "
            "job_id=%s task_id=%s",
            job_id,
            task_id,
        )

        return {
            "job_id": str(job_id),
            "task_id": str(task_id),
            "status": "failed",
            "error": error_message,
        }


app = ASRFailedConsumer.bind()
