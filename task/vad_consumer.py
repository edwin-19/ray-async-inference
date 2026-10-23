
import logging
import os
from pathlib import Path
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
logger = logging.getLogger("vad.consumer")


# ---------------------------------------------------------------------
# Redis / Celery configuration
# ---------------------------------------------------------------------

def redis_url(db: int) -> str:
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD", "")

    if password:
        return f"redis://:{quote(password, safe='')}@{host}:{port}/{db}"

    return f"redis://{host}:{port}/{db}"


REDIS_BROKER_URL = redis_url(0)
REDIS_BACKEND_URL = redis_url(1)

VAD_QUEUE_NAME = os.getenv("VAD_QUEUE", "vad_queue")
ASR_QUEUE_NAME = os.getenv("ASR_QUEUE", "asr_queue")

processor_config = TaskProcessorConfig(
    queue_name=VAD_QUEUE_NAME,
    adapter_config=CeleryAdapterConfig(
        broker_url=REDIS_BROKER_URL,
        backend_url=REDIS_BACKEND_URL,
    ),
)


# ---------------------------------------------------------------------
# FireRedVAD configuration
# ---------------------------------------------------------------------

VAD_MODEL_PATH = Path(
    os.getenv(
        "VAD_MODEL_PATH",
        "/mnt/data/workspace/freelance/voice/diar/"
        "pretrained_models/FireRedVAD/VAD",
    )
)


@serve.deployment(
    name="vad-consumer",
    max_ongoing_requests=2,
    ray_actor_options={"num_cpus": 2},
    autoscaling_config=AutoscalingConfig(
        min_replicas=1,
        max_replicas=2,
        target_ongoing_requests=1,
    ),
)
@task_consumer(task_processor_config=processor_config)
class VadConsumer:

    def __init__(self):
        # Load the model inside the Ray replica rather than creating
        # a model instance at module scope.
        from fireredvad import FireRedVad, FireRedVadConfig

        if not VAD_MODEL_PATH.is_dir():
            raise FileNotFoundError(
                f"FireRedVAD model directory not found: {VAD_MODEL_PATH}"
            )

        config = FireRedVadConfig(
            use_gpu=True,
            smooth_window_size=5,
            speech_threshold=0.4,
            min_speech_frame=20,
            max_speech_frame=2000,
            min_silence_frame=20,
            merge_silence_frame=0,
            extend_speech_frame=0,
            chunk_max_frame=30000,
        )

        self.vad_model = FireRedVad.from_pretrained(
            str(VAD_MODEL_PATH),
            config,
        )

        logger.info("FireRedVAD loaded from %s", VAD_MODEL_PATH)

    @task_handler(name="process")
    def process(self, data):
        from sqlalchemy.exc import SQLAlchemyError
        from sqlmodel import Session, select

        from db.db import engine
        from db.models import (
            Job,
            JobStage,
            JobStatus,
            TranscriptSegment,
            utc_now,
        )

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

        logger.info("Received VAD job_id=%s", job_id)

        # -------------------------------------------------------------
        # 1. Read and validate the job
        # -------------------------------------------------------------
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

            audio_path = Path(job.audio_path)

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        # -------------------------------------------------------------
        # 2. Run FireRedVAD
        # -------------------------------------------------------------
        logger.info(
            "Running FireRedVAD job_id=%s audio=%s",
            job_id,
            audio_path,
        )

        result, _ = self.vad_model.detect(str(audio_path))

        duration_seconds = float(result["dur"])
        timestamps = result["timestamps"]

        segments = []

        for index, timestamp in enumerate(timestamps):
            start_seconds = float(timestamp[0])
            end_seconds = float(timestamp[1])

            if start_seconds < 0 or end_seconds < start_seconds:
                raise ValueError(
                    f"Invalid VAD timestamps at segment {index}: "
                    f"{start_seconds}-{end_seconds}"
                )

            segments.append(
                {
                    "segment_index": index,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                }
            )

        logger.info(
            "VAD complete job_id=%s duration=%.2fs segments=%d",
            job_id,
            duration_seconds,
            len(segments),
        )

        # -------------------------------------------------------------
        # 3. Persist duration and segments in one DB transaction
        # -------------------------------------------------------------
        with Session(engine) as session:
            job = session.get(Job, job_id)

            if job is None:
                raise ValueError(f"Job disappeared: {job_id}")

            if job.status == JobStatus.COMPLETED:
                return {
                    "job_id": str(job_id),
                    "status": "already_completed",
                }

            # Make retries idempotent: replace the previous VAD output.
            existing_segments = session.exec(
                select(TranscriptSegment).where(
                    TranscriptSegment.job_id == job_id
                )
            ).all()

            for segment in existing_segments:
                session.delete(segment)

            job.duration_seconds = duration_seconds
            job.status = JobStatus.PROCESSING
            job.stage = JobStage.VAD
            job.error = None
            job.updated_at = utc_now()

            session.add(job)

            for segment in segments:
                session.add(
                    TranscriptSegment(
                        job_id=job_id,
                        segment_index=segment["segment_index"],
                        start_seconds=segment["start_seconds"],
                        end_seconds=segment["end_seconds"],
                    )
                )

            # ASR must not be dispatched before this commit succeeds.
            session.commit()

        logger.info(
            "Persisted VAD results job_id=%s segments=%d",
            job_id,
            len(segments),
        )

        # -------------------------------------------------------------
        # 4. Publish to ASR only after the DB transaction has committed
        # -------------------------------------------------------------
        publisher = Celery(
            "speech_pipeline_publisher",
            broker=REDIS_BROKER_URL,
            backend=REDIS_BACKEND_URL,
        )

        try:
            asr_task = publisher.send_task(
                "process",
                args=[{"job_id": str(job_id)}],
                queue=ASR_QUEUE_NAME,
            )
        finally:
            publisher.close()

        logger.info(
            "Dispatched job_id=%s to queue=%s task_id=%s",
            job_id,
            ASR_QUEUE_NAME,
            asr_task.id,
        )

        return {
            "job_id": str(job_id),
            "status": "dispatched",
            "duration_seconds": duration_seconds,
            "segments": len(segments),
            "next_queue": ASR_QUEUE_NAME,
            "task_id": asr_task.id,
        }


app = VadConsumer.bind()
