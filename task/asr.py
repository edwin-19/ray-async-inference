
import logging
import os
import time
from pathlib import Path
from itertools import islice
from uuid import UUID

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from celery import Celery
from dotenv import load_dotenv

from ray import serve
from ray.serve.config import AutoscalingConfig
from ray.serve.schema import (
    CeleryAdapterConfig,
    TaskProcessorConfig,
)
from ray.serve.task_consumer import (
    task_consumer,
    task_handler,
)

import boto3
from urllib.parse import quote


load_dotenv()

logger = logging.getLogger("align.consumer")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ALIGN_QUEUE = os.getenv("ALIGN_QUEUE", "align_queue")
ALIGN_FAILED_QUEUE = os.getenv(
    "ALIGN_FAILED_QUEUE",
    "align_failed_queue",
)

ALIGN_MODEL_NAME = os.getenv(
    "ALIGN_MODEL_NAME",
    "Qwen/Qwen3-ForcedAligner-0.6B-hf",
)

ALIGN_BATCH_SIZE = max(
    1,
    int(os.getenv("ALIGN_BATCH_SIZE", "4")),
)

ALIGN_MAX_RETRIES = max(
    0,
    int(os.getenv("ALIGN_MAX_RETRIES", "3")),
)

ALIGN_NUM_GPUS = float(
    os.getenv("ALIGN_NUM_GPUS_PER_REPLICA", "1")
)

ALIGN_NUM_CPUS = int(
    os.getenv("ALIGN_NUM_CPUS_PER_REPLICA", "2")
)

ALIGN_DTYPE = os.getenv("ALIGN_DTYPE", "bfloat16")

ALIGN_DEVICE = os.getenv("ALIGN_DEVICE", "cuda")

ALIGN_SAMPLE_RATE = 16000

ALIGN_UNSUPPORTED_POLICY = os.getenv(
    "ALIGN_UNSUPPORTED_POLICY",
    "skip",
).lower()


SUPPORTED_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}


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
    queue_name=ALIGN_QUEUE,
    adapter_config=CeleryAdapterConfig(
        broker_url=REDIS_BROKER_URL,
        backend_url=REDIS_BACKEND_URL,
    ),
    max_retries=ALIGN_MAX_RETRIES,
    failed_task_queue_name=ALIGN_FAILED_QUEUE,
)


# ---------------------------------------------------------------------
# MinIO
# ---------------------------------------------------------------------

MINIO_ENDPOINT = os.getenv(
    "MINIO_ENDPOINT",
    "127.0.0.1:9000",
)

MINIO_BUCKET = os.getenv(
    "MINIO_BUCKET",
    "speech-data",
)

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


def normalize_language(language):
    if not language:
        return None

    language = str(language).strip()

    if not language:
        return None

    # Accept either language code or full language name.
    for code, name in SUPPORTED_LANGUAGES.items():
        if language.lower() in (code, name.lower()):
            return name

    return None


def load_audio_segment(
    source,
    start_seconds,
    end_seconds,
):
    sample_rate = source.samplerate

    start_frame = max(
        0,
        int(start_seconds * sample_rate),
    )

    end_frame = min(
        len(source),
        int(end_seconds * sample_rate),
    )

    if end_frame <= start_frame:
        raise ValueError(
            f"Invalid audio range: "
            f"{start_seconds}-{end_seconds}"
        )

    source.seek(start_frame)

    audio = source.read(
        frames=end_frame - start_frame,
        dtype="float32",
        always_2d=False,
    )

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if len(audio) == 0:
        raise ValueError("Empty audio segment")

    # Qwen3 ForcedAligner expects 16 kHz audio.
    if sample_rate != ALIGN_SAMPLE_RATE:
        from math import gcd

        divisor = gcd(sample_rate, ALIGN_SAMPLE_RATE)

        audio = resample_poly(
            audio,
            ALIGN_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(np.float32)

    return audio


# ---------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------

@serve.deployment(
    name="align-consumer",
    max_ongoing_requests=1,
    ray_actor_options={
        "num_cpus": ALIGN_NUM_CPUS,
        "num_gpus": ALIGN_NUM_GPUS,
    },
    autoscaling_config=AutoscalingConfig(
        min_replicas=int(
            os.getenv("ALIGN_MIN_REPLICAS", "1")
        ),
        max_replicas=int(
            os.getenv("ALIGN_MAX_REPLICAS", "1")
        ),
        target_ongoing_requests=int(
            os.getenv("ALIGN_TARGET_ONGOING_REQUESTS", "1")
        ),
    ),
)
@task_consumer(task_processor_config=processor_config)
class AlignmentConsumer:

    def __init__(self):

        # -------------------------------------------------------------
        # 1. Initialize model
        # -------------------------------------------------------------

        logger.info(
            "Loading forced aligner model=%s batch_size=%d",
            ALIGN_MODEL_NAME,
            ALIGN_BATCH_SIZE,
        )

        from transformers import (
            AutoProcessor,
            AutoModelForTokenClassification,
        )

        self.device = torch.device(
            ALIGN_DEVICE
            if ALIGN_DEVICE == "cpu" or torch.cuda.is_available()
            else "cpu"
        )

        self.dtype = (
            torch.bfloat16
            if ALIGN_DTYPE == "bfloat16"
            else torch.float16
        )

        self.processor = AutoProcessor.from_pretrained(
            ALIGN_MODEL_NAME,
        )

        self.model = (
            AutoModelForTokenClassification.from_pretrained(
                ALIGN_MODEL_NAME,
                dtype=self.dtype if self.device.type == "cuda"
                else torch.float32,
            )
            .to(self.device)
            .eval()
        )

        logger.info(
            "Forced aligner loaded device=%s dtype=%s",
            self.device,
            next(self.model.parameters()).dtype,
        )

        # -------------------------------------------------------------
        # 2. Initialize MinIO
        # -------------------------------------------------------------

        self.minio = create_minio_client()

    # -----------------------------------------------------------------
    # Audio download
    # -----------------------------------------------------------------

    def _download_audio(self, object_key):

        suffix = Path(object_key).suffix or ".audio"

        import tempfile

        fd, local_path = tempfile.mkstemp(
            prefix="align-",
            suffix=suffix,
        )

        os.close(fd)

        try:
            self.minio.download_file(
                MINIO_BUCKET,
                object_key,
                local_path,
            )

            return local_path

        except Exception:
            Path(local_path).unlink(missing_ok=True)
            raise

    # -----------------------------------------------------------------
    # Model inference
    # -----------------------------------------------------------------

    def _align_batch(self, audio_batch, transcript_batch, languages):

        if not audio_batch:
            return []

        started = time.perf_counter()

        # Audio must be 16 kHz mono float32.
        # Unsupported languages are handled by the caller.
        aligner_inputs, word_lists = (
            self.processor.prepare_forced_aligner_inputs(
                audio=audio_batch,
                transcript=transcript_batch,
                language=languages,
            )
        )

        aligner_inputs = aligner_inputs.to(
            self.device,
            dtype=self.dtype if self.device.type == "cuda"
            else torch.float32,
        )

        with torch.inference_mode():
            outputs = self.model(**aligner_inputs)

        timestamps = self.processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=aligner_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=self.model.config.timestamp_token_id,
        )

        if len(timestamps) != len(audio_batch):
            raise RuntimeError(
                "Alignment result count mismatch: "
                f"submitted={len(audio_batch)}, "
                f"received={len(timestamps)}"
            )

        logger.info(
            "Alignment inference batch=%d elapsed=%.2fs",
            len(audio_batch),
            time.perf_counter() - started,
        )

        return timestamps

    # -----------------------------------------------------------------
    # Persistence adapter
    # -----------------------------------------------------------------

    def _persist_word_alignments(
        self,
        session,
        segment,
        timestamps,
        utc_now,
    ):
        """
        Persist word-level alignment results.

        Recommended storage:
            TranscriptWord
                id
                job_id
                segment_id
                word_index
                text
                start_seconds
                end_seconds

        Timestamp values returned by Qwen are relative to
        the supplied audio segment.

        Convert them into absolute job-level timestamps.
        """

        from db.models import TranscriptWord

        # Replace existing results for this segment so that
        # task retries remain idempotent.

        existing = session.exec(
            select(TranscriptWord).where(
                TranscriptWord.segment_id == segment.id
            )
        ).all()

        for item in existing:
            session.delete(item)

        for index, item in enumerate(timestamps):

            word = TranscriptWord(
                job_id=segment.job_id,
                segment_id=segment.id,
                word_index=index,
                text=item["text"],
                start_seconds=(
                    segment.start_seconds + item["start_time"]
                ),
                end_seconds=(
                    segment.start_seconds + item["end_time"]
                ),
            )

            session.add(word)

        # The caller owns the transaction and commits.


    # -----------------------------------------------------------------
    # Main task
    # -----------------------------------------------------------------

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
            raise ValueError(
                "Expected payload containing job_id"
            )

        job_id = UUID(str(data["job_id"]))

        started = time.perf_counter()
        temp_audio_path = None

        try:

            # ---------------------------------------------------------
            # 1. Resolve job
            # ---------------------------------------------------------

            with Session(engine) as session:

                job = session.get(Job, job_id)

                if job is None:
                    raise ValueError(
                        f"Job not found: {job_id}"
                    )

                if job.status in (
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                ):
                    return {
                        "job_id": str(job_id),
                        "status": enum_value(job.status),
                    }

                audio_object_key = (
                    job.audio_object_key
                    or data.get("audio_object_key")
                )

                legacy_audio_path = getattr(
                    job,
                    "audio_path",
                    None,
                )

                job.status = JobStatus.PROCESSING
                job.stage = JobStage.ALIGN
                job.error = None
                job.updated_at = utc_now()

                session.add(job)
                session.commit()

                job_language = (
                    getattr(job, "language", None)
                    or getattr(job, "requested_language", None)
                )

                job_transcript = job.transcript

            # ---------------------------------------------------------
            # 2. Resolve audio
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
                    f"Job {job_id} has no audio source"
                )

            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)

            # ---------------------------------------------------------
            # 3. Load segments
            # ---------------------------------------------------------

            with Session(engine) as session:

                segments = session.exec(
                    select(TranscriptSegment)
                    .where(
                        TranscriptSegment.job_id == job_id
                    )
                    .order_by(
                        TranscriptSegment.segment_index
                    )
                ).all()

                # For jobs without VAD segments, use the full
                # transcript as a single alignment input.
                fallback_to_full_audio = not segments

                if fallback_to_full_audio:
                    if not job_transcript:
                        raise ValueError(
                            "No transcript available for alignment"
                        )

                    alignment_items = [{
                        "segment": None,
                        "text": job_transcript,
                        "start": 0.0,
                        "end": None,
                        "language": job_language,
                    }]

                else:
                    alignment_items = [
                        {
                            "segment": segment,
                            "text": segment.text,
                            "start": segment.start_seconds,
                            "end": segment.end_seconds,
                            "language": (
                                segment.language or job_language
                            ),
                        }
                        for segment in segments
                        if segment.text and segment.text.strip()
                    ]

                logger.info(
                    "Starting alignment job=%s segments=%d "
                    "fallback=%s",
                    job_id,
                    len(alignment_items),
                    fallback_to_full_audio,
                )

                # -----------------------------------------------------
                # 4. Prepare audio and alignment batches
                # -----------------------------------------------------

                with sf.SoundFile(str(audio_path)) as source:

                    prepared_items = []

                    for item in alignment_items:

                        language = normalize_language(
                            item["language"]
                        )

                        if language is None:
                            if item["language"] and (
                                ALIGN_UNSUPPORTED_POLICY == "skip"
                            ):
                                logger.warning(
                                    "Skipping unsupported language "
                                    "job=%s language=%s",
                                    job_id,
                                    item["language"],
                                )

                                continue

                            if item["language"] and (
                                ALIGN_UNSUPPORTED_POLICY == "error"
                            ):
                                raise ValueError(
                                    "Unsupported alignment language: "
                                    f"{item['language']}"
                                )

                        end = (
                            item["end"]
                            if item["end"] is not None
                            else len(source) / source.samplerate
                        )

                        audio = load_audio_segment(
                            source,
                            item["start"],
                            end,
                        )

                        prepared_items.append({
                            **item,
                            "audio": audio,
                            "language": language,
                        })

                    # -------------------------------------------------
                    # 5. Batched forced alignment
                    # -------------------------------------------------

                    for batch in batched(
                        prepared_items,
                        ALIGN_BATCH_SIZE,
                    ):

                        audio_batch = [
                            item["audio"]
                            for item in batch
                        ]

                        transcript_batch = [
                            item["text"]
                            for item in batch
                        ]

                        language_batch = [
                            item["language"]
                            for item in batch
                        ]

                        timestamps_batch = self._align_batch(
                            audio_batch,
                            transcript_batch,
                            language_batch,
                        )

                        # -------------------------------------------------
                        # 6. Persist batch
                        # -------------------------------------------------

                        with Session(engine) as write_session:

                            for item, timestamps in zip(
                                batch,
                                timestamps_batch,
                            ):

                                if item["segment"] is None:
                                    # Full-audio fallback:
                                    # persist against the job-level
                                    # transcript using a separate
                                    # job-level word storage adapter.
                                    self._persist_job_word_alignments(
                                        write_session,
                                        job_id,
                                        timestamps,
                                    )
                                else:
                                    self._persist_word_alignments(
                                        write_session,
                                        item["segment"],
                                        timestamps,
                                        utc_now,
                                    )

                            write_session.commit()

            # ---------------------------------------------------------
            # 7. Finalize job
            # ---------------------------------------------------------

            with Session(engine) as session:

                job = session.get(Job, job_id)

                if job is None:
                    raise RuntimeError(
                        f"Job disappeared: {job_id}"
                    )

                job.status = JobStatus.COMPLETED
                job.stage = JobStage.COMPLETED
                job.error = None
                job.updated_at = utc_now()

                session.add(job)
                session.commit()

            logger.info(
                "Alignment completed job=%s elapsed=%.2fs",
                job_id,
                time.perf_counter() - started,
            )

            return {
                "job_id": str(job_id),
                "status": "completed",
                "stage": "align",
            }

        except Exception as exc:

            logger.exception(
                "Alignment failed job=%s",
                job_id,
            )

            try:
                with Session(engine) as session:

                    job = session.get(Job, job_id)

                    if job is not None:
                        job.status = JobStatus.PROCESSING
                        job.stage = JobStage.ALIGN
                        job.error = str(exc)[:4000]
                        job.updated_at = utc_now()

                        session.add(job)
                        session.commit()

            except Exception:
                logger.exception(
                    "Could not persist alignment error"
                )

            raise

        finally:

            if temp_audio_path:
                Path(temp_audio_path).unlink(missing_ok=True)


def enum_value(value):
    return value.value if hasattr(value, "value") else str(value)


app = AlignmentConsumer.bind()
