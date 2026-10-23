
import logging
import os
import time
from collections import Counter
from itertools import islice
from urllib.parse import quote
from uuid import UUID

import soundfile as sf
import torch
from dotenv import load_dotenv
from ray import serve
from ray.serve.config import AutoscalingConfig
from ray.serve.schema import CeleryAdapterConfig, TaskProcessorConfig
from ray.serve.task_consumer import task_consumer, task_handler

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asr.consumer")

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
        return (
            f"redis://:{quote(password, safe='')}"
            f"@{host}:{port}/{db}"
        )

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

        if torch.cuda.is_available():
            device = "cuda:0"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        logger.info(
            "Loading ASR model=%s device=%s batch_size=%d",
            ASR_MODEL_NAME,
            device,
            ASR_BATCH_SIZE,
        )

        self.model = Qwen3ASRModel.from_pretrained(
            ASR_MODEL_NAME,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=ASR_BATCH_SIZE,
        )

        logger.info("ASR model loaded")

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

        # Import Celery here to avoid serializing publisher state
        # into the Ray Serve deployment.
        from celery import Celery

        publisher = Celery(
            "align_dispatch",
            broker=REDIS_BROKER_URL,
            backend=REDIS_BACKEND_URL,
        )

        try:
            with Session(engine) as session:
                job = session.exec(
                    select(Job).where(Job.id == job_id)
                ).first()

                if job is None:
                    raise ValueError(f"Job not found: {job_id}")

                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    logger.info(
                        "Skipping terminal job=%s status=%s",
                        job_id,
                        job.status,
                    )
                    return {
                        "job_id": str(job_id),
                        "status": job.status.value,
                    }

                audio_path = job.audio_path

                if not os.path.isfile(audio_path):
                    raise FileNotFoundError(
                        f"Audio file not found: {audio_path}"
                    )

                job.status = JobStatus.PROCESSING
                job.stage = JobStage.ASR
                job.error = None

                if job.started_at is None:
                    job.started_at = utc_now()

                job.updated_at = utc_now()
                session.add(job)
                session.commit()

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
                pending = [
                    segment for segment in segments
                    if segment.text is None
                ]

                logger.info(
                    "ASR job=%s total=%d pending=%d batch_size=%d",
                    job_id,
                    total_segments,
                    len(pending),
                    ASR_BATCH_SIZE,
                )

                with sf.SoundFile(str(audio_path)) as source:
                    sample_rate = source.samplerate

                    for batch_number, batch in enumerate(
                        batched(pending, ASR_BATCH_SIZE),
                        start=1,
                    ):
                        batch_started = time.perf_counter()

                        audio_batch = self._read_segment_batch(
                            source, batch, sample_rate
                        )
                        results = self._transcribe_batch(audio_batch)

                        for segment, (text, language) in zip(
                            batch, results
                        ):
                            segment.text = text
                            segment.language = language
                            session.add(segment)

                        job.updated_at = utc_now()
                        session.add(job)
                        session.commit()

                        completed_count = total_segments - sum(
                            segment.text is None
                            for segment in segments
                        )

                        logger.info(
                            "ASR progress job=%s batch=%d "
                            "completed=%d/%d elapsed=%.2fs",
                            job_id,
                            batch_number,
                            completed_count,
                            total_segments,
                            time.perf_counter() - batch_started,
                        )

                # Reload persisted results before finalization.
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

                job = session.exec(
                    select(Job).where(Job.id == job_id)
                ).first()

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

                # Set the next stage before dispatching alignment.
                job.status = JobStatus.PROCESSING
                job.stage = JobStage.ALIGN
                job.error = None
                job.updated_at = utc_now()

                session.add(job)
                session.commit()

                # Publish only after the ASR results and stage are committed.
                publisher.send_task(
                    "process",
                    args=[{"job_id": str(job_id)}],
                    queue=ALIGN_QUEUE,
                )

                logger.info(
                    "ASR completed job=%s; dispatched to queue=%s "
                    "elapsed=%.2fs",
                    job_id,
                    ALIGN_QUEUE,
                    time.perf_counter() - started,
                )

                return {
                    "job_id": str(job_id),
                    "status": "processing",
                    "stage": "align",
                    "segments": total_segments,
                }

        except Exception as exc:
            logger.exception("ASR failed job=%s", job_id)

            # Preserve committed segment results for a future retry.
            try:
                with Session(engine) as session:
                    job = session.exec(
                        select(Job).where(Job.id == job_id)
                    ).first()

                    if job is not None:
                        job.status = JobStatus.PROCESSING
                        job.stage = JobStage.ASR
                        job.error = str(exc)[:4000]
                        job.updated_at = utc_now()
                        session.add(job)
                        session.commit()

            except Exception:
                logger.exception(
                    "Could not persist ASR error for job=%s", job_id
                )

            raise


app = ASRConsumer.bind()
