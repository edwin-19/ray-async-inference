
import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

import boto3
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
# MinIO configuration
# ---------------------------------------------------------------------

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "speech-data")

MINIO_ACCESS_KEY = os.getenv(
    "MINIO_ACCESS_KEY",
    os.getenv("MINIO_ROOT_USER", ""),
)
MINIO_SECRET_KEY = os.getenv(
    "MINIO_SECRET_KEY",
    os.getenv("MINIO_ROOT_PASSWORD", ""),
)

MINIO_SECURE = (
    os.getenv("MINIO_SECURE", "false").lower() == "true"
)


def create_minio_client():
    endpoint = MINIO_ENDPOINT.rstrip("/")

    if not endpoint.startswith(("http://", "https://")):
        scheme = "https" if MINIO_SECURE else "http"
        endpoint = f"{scheme}://{endpoint}"

    if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        raise ValueError(
            "Set MINIO_ACCESS_KEY and MINIO_SECRET_KEY, "
            "or MINIO_ROOT_USER and MINIO_ROOT_PASSWORD."
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name=os.getenv("MINIO_REGION", "us-east-1"),
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

        # Initialize clients within the Ray replica.
        self.minio = create_minio_client()

        logger.info("FireRedVAD loaded from %s", VAD_MODEL_PATH)
        logger.info(
            "MinIO configured: endpoint=%s bucket=%s",
            MINIO_ENDPOINT,
            MINIO_BUCKET,
        )

    def _download_audio(self, object_key: str) -> str:
        """Download an audio object to a local temporary file."""
        suffix = Path(object_key).suffix or ".audio"

        fd, local_path = tempfile.mkstemp(
            prefix="vad-",
            suffix=suffix,
        )
        os.close(fd)

        try:
            self.minio.download_file(
                MINIO_BUCKET,
                object_key,
                local_path,
            )
            logger.info(
                "Downloaded audio from MinIO: bucket=%s key=%s",
                MINIO_BUCKET,
                object_key,
            )
            return local_path
        except Exception:
            Path(local_path).unlink(missing_ok=True)
            logger.exception(
                "MinIO download failed: bucket=%s key=%s",
                MINIO_BUCKET,
                object_key,
            )
            raise

    @task_handler(name="process")
    def process(self, data):
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
        # 1. Read job and resolve audio source
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

            audio_object_key = (
                getattr(job, "audio_object_key", None)
                or data.get("audio_object_key")
            )
            legacy_audio_path = getattr(job, "audio_path", None)

            job.status = JobStatus.PROCESSING
            job.stage = JobStage.VAD

            if job.started_at is None:
                job.started_at = utc_now()

            job.updated_at = utc_now()
            session.add(job)
            session.commit()

        temp_audio_path = None

        try:
            # ---------------------------------------------------------
            # 2. Download audio or use a legacy local file
            # ---------------------------------------------------------
            if audio_object_key:
                temp_audio_path = self._download_audio(
                    str(audio_object_key)
                )
                audio_path = Path(temp_audio_path)
            elif legacy_audio_path:
                audio_path = Path(legacy_audio_path)
            else:
                raise ValueError(
                    f"Job {job_id} has neither audio_object_key "
                    "nor audio_path"
                )

            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Audio file not found: {audio_path}"
                )

            # ---------------------------------------------------------
            # 3. Run FireRedVAD
            # ---------------------------------------------------------
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

                if (
                    start_seconds < 0
                    or end_seconds < start_seconds
                    or end_seconds > duration_seconds + 1.0
                ):
                    raise ValueError(
                        f"Invalid VAD timestamps at segment {index}: "
                        f"{start_seconds}-{end_seconds}; "
                        f"duration={duration_seconds}"
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

            # ---------------------------------------------------------
            # 4. Persist duration and segments atomically
            # ---------------------------------------------------------
            with Session(engine) as session:
                job = session.get(Job, job_id)

                if job is None:
                    raise ValueError(f"Job disappeared: {job_id}")

                if job.status == JobStatus.COMPLETED:
                    return {
                        "job_id": str(job_id),
                        "status": "already_completed",
                    }

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

                session.commit()

            logger.info(
                "Persisted VAD results job_id=%s segments=%d",
                job_id,
                len(segments),
            )

            # ---------------------------------------------------------
            # 5. Dispatch to ASR after the database commit
            # ---------------------------------------------------------
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

        finally:
            if temp_audio_path:
                Path(temp_audio_path).unlink(missing_ok=True)
                logger.info(
                    "Cleaned up temporary audio job_id=%s",
                    job_id,
                )


app = VadConsumer.bind()
