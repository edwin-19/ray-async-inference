
import logging
import os
import tempfile
import time
from collections import Counter
from itertools import islice
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

import boto3
import soundfile as sf
import torch
from dotenv import load_dotenv
from celery import Celery
from ray import serve
from ray.serve.config import AutoscalingConfig
from ray.serve.schema import CeleryAdapterConfig, TaskProcessorConfig
from ray.serve.task_consumer import task_consumer, task_handler

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asr.consumer")


# ---------------------------------------------------------------------
# Queue / model configuration
# ---------------------------------------------------------------------

ASR_QUEUE = os.getenv("ASR_QUEUE", "asr_queue")
ALIGN_QUEUE = os.getenv("ALIGN_QUEUE", "align_queue")

ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "").strip() or None
ASR_BATCH_SIZE = max(1, int(os.getenv("ASR_BATCH_SIZE", "8")))

ASR_NUM_GPUS = float(os.getenv("ASR_NUM_GPUS_PER_REPLICA", "1"))
ASR_NUM_CPUS = int(os.getenv("ASR_NUM_CPUS_PER_REPLICA", "2"))


def redis_url(db: int) -> str:
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD", "")

    if password:
        return f"redis://:{quote(password, safe='')}@{host}:{port}/{db}"

    return f"redis://{host}:{port}/{db}"


REDIS_BROKER_URL = redis_url(0)
REDIS_BACKEND_URL = redis_url(1)

processor_config = TaskProcessorConfig(
    queue_name=ASR_QUEUE,
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
# Helpers
# ---------------------------------------------------------------------

def batched(items, batch_size):
    iterator = iter(items)

    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def result_fields(result):
    if isinstance(result, dict):
        text = result.get("text", "")
        language = result.get("language")
    else:
        text = getattr(result, "text", "")
        language = getattr(result, "language", None)

    text = str(text or "").strip()
    language = str(language).strip() if language else None

    return text, language


# ---------------------------------------------------------------------
# Ray Serve deployment
# ---------------------------------------------------------------------

@serve.deployment(
    name="asr-consumer",
    max_ongoing_requests=1,
    ray_actor_options={
        "num_cpus": ASR_NUM_CPUS,
        "num_gpus": ASR_NUM_GPUS,
    },
    autoscaling_config=AutoscalingConfig(
        min_replicas=1,
        max_replicas=1,
        target_ongoing_requests=1,
    ),
)
@task_consumer(task_processor_config=processor_config)
class ASRConsumer:

    
    def __init__(self):
        from qwen_asr import Qwen3ASRModel

        gpu_memory_utilization = float(
            os.getenv("ASR_GPU_MEMORY_UTILIZATION", "0.85")
        )

        logger.info(
            "Loading ASR model=%s backend=vLLM batch_size=%d "
            "gpu_memory_utilization=%.2f",
            ASR_MODEL_NAME,
            ASR_BATCH_SIZE,
            gpu_memory_utilization,
        )

        self.model = Qwen3ASRModel.LLM(
            model=ASR_MODEL_NAME,
            max_inference_batch_size=ASR_BATCH_SIZE,
            max_model_len=5096,
            max_new_tokens=1028,
            dtype="float16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=ASR_BATCH_SIZE,
        )

        self.minio = create_minio_client()

        logger.info("vLLM ASR model loaded")
        logger.info(
            "MinIO configured: endpoint=%s bucket=%s",
            MINIO_ENDPOINT,
            MINIO_BUCKET,
        )


    def _download_audio(self, object_key: str) -> str:
        """Download an audio object to a local temporary file."""
        suffix = Path(object_key).suffix or ".audio"

        fd, local_path = tempfile.mkstemp(
            prefix="asr-",
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

    def _read_segment_batch(self, source, segments, sample_rate):
        audio_batch = []

        for segment in segments:
            start_frame = max(
                0, int(segment.start_seconds * sample_rate)
            )
            end_frame = min(
                len(source),
                max(
                    start_frame,
                    int(segment.end_seconds * sample_rate),
                ),
            )

            source.seek(start_frame)
            audio = source.read(
                frames=end_frame - start_frame,
                dtype="float32",
                always_2d=False,
            )

            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            audio_batch.append((audio, sample_rate))

        return audio_batch

    def _transcribe_batch(self, audio_batch):
        if not audio_batch:
            return []

        started = time.perf_counter()

        results = self.model.transcribe(
            audio=audio_batch,
            language=ASR_LANGUAGE,
        )

        if not isinstance(results, list):
            raise RuntimeError(
                f"Expected list of ASR results, got "
                f"{type(results).__name__}"
            )

        if len(results) != len(audio_batch):
            raise RuntimeError(
                f"ASR result count mismatch: submitted={len(audio_batch)}, "
                f"received={len(results)}"
            )

        logger.info(
            "ASR inference batch_size=%d elapsed=%.2fs",
            len(audio_batch),
            time.perf_counter() - started,
        )

        return [result_fields(result) for result in results]

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

        if not isinstance(data, dict) or not data.get("job_id"):
            raise ValueError("Expected payload containing job_id")

        job_id = UUID(str(data["job_id"]))
        started = time.perf_counter()

        publisher = Celery(
            "align_dispatch",
            broker=REDIS_BROKER_URL,
            backend=REDIS_BACKEND_URL,
        )

        temp_audio_path = None

        try:
            # ---------------------------------------------------------
            # 1. Resolve the audio source and mark ASR as active
            # ---------------------------------------------------------
            with Session(engine) as session:
                job = session.get(Job, job_id)

                if job is None:
                    raise ValueError(f"Job not found: {job_id}")

                if job.status in (
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                ):
                    logger.info(
                        "Skipping terminal job=%s status=%s",
                        job_id,
                        job.status,
                    )
                    return {
                        "job_id": str(job_id),
                        "status": job.status.value,
                    }

                audio_object_key = (
                    getattr(job, "audio_object_key", None)
                    or data.get("audio_object_key")
                )
                legacy_audio_path = getattr(job, "audio_path", None)

                job.status = JobStatus.PROCESSING
                job.stage = JobStage.ASR
                job.error = None

                if job.started_at is None:
                    job.started_at = utc_now()

                job.updated_at = utc_now()
                session.add(job)
                session.commit()

            # Prefer MinIO. Keep local-path support for older jobs.
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
            # 2. Load VAD segments and identify unfinished work
            # ---------------------------------------------------------
            with Session(engine) as session:
                segments = session.exec(
                    select(TranscriptSegment)
                    .where(TranscriptSegment.job_id == job_id)
                    .order_by(TranscriptSegment.segment_index)
                ).all()

                if not segments:
                    raise RuntimeError(
                        f"No VAD segments found for job {job_id}"
                    )

                total_segments = len(segments)
                pending_indexes = [
                    segment.segment_index
                    for segment in segments
                    if segment.text is None
                ]

                logger.info(
                    "ASR job=%s total=%d pending=%d batch_size=%d",
                    job_id,
                    total_segments,
                    len(pending_indexes),
                    ASR_BATCH_SIZE,
                )

                # -----------------------------------------------------
                # 3. Run ASR in batches, committing each batch
                # -----------------------------------------------------
                with sf.SoundFile(str(audio_path)) as source:
                    sample_rate = source.samplerate

                    for batch_number, batch_indexes in enumerate(
                        batched(pending_indexes, ASR_BATCH_SIZE),
                        start=1,
                    ):
                        batch_started = time.perf_counter()

                        # Re-read current segment rows for this batch.
                        batch_segments = session.exec(
                            select(TranscriptSegment)
                            .where(
                                TranscriptSegment.job_id == job_id,
                                TranscriptSegment.segment_index.in_(
                                    batch_indexes
                                ),
                            )
                            .order_by(TranscriptSegment.segment_index)
                        ).all()

                        # Another retry may have completed some segments.
                        batch_segments = [
                            segment for segment in batch_segments
                            if segment.text is None
                        ]

                        if not batch_segments:
                            continue

                        audio_batch = self._read_segment_batch(
                            source,
                            batch_segments,
                            sample_rate,
                        )
                        results = self._transcribe_batch(audio_batch)

                        for segment, (text, language) in zip(
                            batch_segments, results
                        ):
                            segment.text = text
                            segment.language = language
                            session.add(segment)

                        job = session.get(Job, job_id)
                        if job is None:
                            raise RuntimeError(
                                f"Job disappeared during ASR: {job_id}"
                            )

                        job.updated_at = utc_now()
                        session.add(job)
                        session.commit()

                        completed_count = session.exec(
                            select(TranscriptSegment).where(
                                TranscriptSegment.job_id == job_id,
                                TranscriptSegment.text.is_not(None),
                            )
                        ).all()

                        logger.info(
                            "ASR progress job=%s batch=%d "
                            "completed=%d/%d elapsed=%.2fs",
                            job_id,
                            batch_number,
                            len(completed_count),
                            total_segments,
                            time.perf_counter() - batch_started,
                        )

            # ---------------------------------------------------------
            # 4. Verify all segments and finalize transcript
            # ---------------------------------------------------------
            with Session(engine) as session:
                final_segments = session.exec(
                    select(TranscriptSegment)
                    .where(TranscriptSegment.job_id == job_id)
                    .order_by(TranscriptSegment.segment_index)
                ).all()

                unfinished = [
                    segment.segment_index
                    for segment in final_segments
                    if segment.text is None
                ]

                if unfinished:
                    raise RuntimeError(
                        f"{len(unfinished)} ASR segments remain unfinished; "
                        f"first indexes={unfinished[:10]}"
                    )

                transcript_parts = [
                    segment.text.strip()
                    for segment in final_segments
                    if segment.text and segment.text.strip()
                ]

                languages = Counter(
                    segment.language
                    for segment in final_segments
                    if segment.language
                )

                job = session.get(Job, job_id)

                if job is None:
                    raise RuntimeError(
                        f"Job disappeared during ASR: {job_id}"
                    )

                job.transcript = "\n".join(transcript_parts)
                job.language = (
                    languages.most_common(1)[0][0]
                    if languages
                    else None
                )
                job.status = JobStatus.PROCESSING
                job.stage = JobStage.ALIGN
                job.error = None
                job.updated_at = utc_now()

                session.add(job)
                session.commit()

            # ---------------------------------------------------------
            # 5. Dispatch to alignment after committing ASR results
            # ---------------------------------------------------------
            align_task = publisher.send_task(
                "process",
                args=[{"job_id": str(job_id)}],
                queue=ALIGN_QUEUE,
            )

            logger.info(
                "ASR completed job=%s; dispatched to queue=%s task_id=%s "
                "elapsed=%.2fs",
                job_id,
                ALIGN_QUEUE,
                align_task.id,
                time.perf_counter() - started,
            )

            return {
                "job_id": str(job_id),
                "status": "processing",
                "stage": "align",
                "segments": total_segments,
                "next_queue": ALIGN_QUEUE,
                "task_id": align_task.id,
            }

        except Exception as exc:
            logger.exception("ASR failed job=%s", job_id)

            # Preserve committed segment results for retry.
            try:
                with Session(engine) as session:
                    job = session.get(Job, job_id)

                    if job is not None and job.status != JobStatus.COMPLETED:
                        job.status = JobStatus.PROCESSING
                        job.stage = JobStage.ASR
                        job.error = str(exc)[:4000]
                        job.updated_at = utc_now()
                        session.add(job)
                        session.commit()

            except Exception:
                logger.exception(
                    "Could not persist ASR error for job=%s",
                    job_id,
                )

            raise

        finally:
            publisher.close()

            if temp_audio_path:
                Path(temp_audio_path).unlink(missing_ok=True)
                logger.info(
                    "Cleaned up temporary audio job_id=%s",
                    job_id,
                )


app = ASRConsumer.bind()
