
import logging
import os
import tempfile
import time
from itertools import islice
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

import boto3
import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from ray import serve
from ray.serve.config import AutoscalingConfig
from ray.serve.schema import CeleryAdapterConfig, TaskProcessorConfig
from ray.serve.task_consumer import task_consumer, task_handler
from scipy.signal import resample_poly
from sqlalchemy import delete
from sqlmodel import Session, select

from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("align.consumer")


# ============================================================
# CONFIGURATION
# ============================================================

ALIGN_QUEUE = os.getenv(
    "ALIGN_QUEUE",
    "align_queue",
)

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
    int(os.getenv("AUDIO_03_BATCH_SIZE", "4")),
)

ALIGN_MAX_RETRIES = max(
    0,
    int(os.getenv("ALIGN_MAX_RETRIES", "3")),
)

ALIGN_DEVICE = os.getenv(
    "ALIGN_DEVICE",
    "auto",
).lower()

ALIGN_NUM_GPUS = float(
    os.getenv("AUDIO_03_NUM_GPUS_PER_REPLICA", "1")
)

ALIGN_NUM_CPUS = int(
    os.getenv("AUDIO_03_NUM_CPUS_PER_REPLICA", "2")
)

ALIGN_COMPILE = (
    os.getenv("ALIGN_COMPILE", "false").lower()
    in {"1", "true", "yes", "on"}
)

ALIGN_COMPILE_BACKEND = os.getenv(
    "ALIGN_COMPILE_BACKEND",
    "inductor",
)

ALIGN_COMPILE_FULLGRAPH = (
    os.getenv("ALIGN_COMPILE_FULLGRAPH", "false").lower()
    in {"1", "true", "yes", "on"}
)

ALIGN_COMPILE_DYNAMIC = (
    os.getenv("ALIGN_COMPILE_DYNAMIC", "false").lower()
    in {"1", "true", "yes", "on"}
)

ALIGN_SAMPLE_RATE = 16000


# ============================================================
# LANGUAGE CONFIGURATION
# ============================================================

ALIGN_DEFAULT_LANGUAGE = "English"

ALIGN_SUPPORTED_LANGUAGES = {
    "Cantonese",
    "Chinese",
    "English",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
}

ALIGN_LANGUAGE_ALIASES = {
    "yue": "Cantonese",
    "zh": "Chinese",
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}

def aligner_language(language):
    if language is None:
        return ALIGN_DEFAULT_LANGUAGE

    language = str(language).strip()

    if not language:
        return ALIGN_DEFAULT_LANGUAGE

    language_lower = language.lower()

    # Malay, Indonesian and Iban are unsupported
    # by the installed forced aligner.
    if language_lower in {
        "indonesian",
        "id",
        "malay",
        "ms",
        "iban",
    }:
        logger.warning(
            "Language %s is unsupported by Qwen3 Forced Aligner; "
            "falling back to English",
            language,
        )
        return ALIGN_DEFAULT_LANGUAGE

    language = ALIGN_LANGUAGE_ALIASES.get(
        language_lower,
        language,
    )

    for supported in ALIGN_SUPPORTED_LANGUAGES:
        if supported.lower() == language.lower():
            return supported

    logger.warning(
        "Unsupported alignment language=%s; falling back to English",
        language,
    )

    return ALIGN_DEFAULT_LANGUAGE

# Explicit fallbacks for languages not natively supported.
ALIGN_LANGUAGE_FALLBACKS = {
    "indonesian": "Malay",
    "id": "Malay",
}


def normalize_language(language):
    if language is None:
        return None

    language = str(language).strip()

    return language or None

# ============================================================
# REDIS CONFIGURATION
# ============================================================

def redis_url(db: int) -> str:
    host = os.getenv(
        "REDIS_HOST",
        "127.0.0.1",
    )

    port = int(
        os.getenv("REDIS_PORT", "6379")
    )

    password = os.getenv(
        "REDIS_PASSWORD",
        "",
    )

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


# ============================================================
# MINIO CONFIGURATION
# ============================================================

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
    os.getenv("MINIO_SECURE", "false").lower()
    in {"1", "true", "yes", "on"}
)


def create_minio_client():
    endpoint = MINIO_ENDPOINT.rstrip("/")

    if not endpoint.startswith(
        ("http://", "https://")
    ):
        scheme = (
            "https"
            if MINIO_SECURE
            else "http"
        )
        endpoint = f"{scheme}://{endpoint}"

    if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        raise ValueError(
            "Set MINIO_ACCESS_KEY and MINIO_SECRET_KEY "
            "or MINIO_ROOT_USER and MINIO_ROOT_PASSWORD."
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name=os.getenv(
            "MINIO_REGION",
            "us-east-1",
        ),
    )


# ============================================================
# GENERAL HELPERS
# ============================================================

def batched(items, batch_size):
    iterator = iter(items)

    while True:
        batch = list(
            islice(iterator, batch_size)
        )

        if not batch:
            break

        yield batch


def enum_value(value):
    return (
        value.value
        if hasattr(value, "value")
        else str(value)
    )


def resolve_device():
    if ALIGN_DEVICE == "auto":
        return (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

    if ALIGN_DEVICE.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"ALIGN_DEVICE={ALIGN_DEVICE}, "
                "but CUDA is unavailable"
            )

    return ALIGN_DEVICE


def resolve_dtype(device):
    if str(device).startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float16

    return torch.float32


def resample_audio(
    audio,
    source_rate,
    target_rate=ALIGN_SAMPLE_RATE,
):
    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if source_rate <= 0:
        raise ValueError(
            f"Invalid source sample rate: {source_rate}"
        )

    if len(audio) == 0:
        raise ValueError(
            "Cannot resample empty audio"
        )

    if source_rate == target_rate:
        return np.ascontiguousarray(audio)

    divisor = np.gcd(
        source_rate,
        target_rate,
    )

    up = target_rate // divisor
    down = source_rate // divisor

    output = resample_poly(
        audio,
        up,
        down,
    )

    return np.ascontiguousarray(
        output,
        dtype=np.float32,
    )


# ============================================================
# ALIGNMENT RESULT HELPERS
# ============================================================

def result_items(result):
    if isinstance(result, dict):
        for key in (
            "items",
            "words",
            "segments",
            "alignment",
        ):
            value = result.get(key)

            if value is not None:
                return list(value)

        if any(
            key in result
            for key in (
                "start_time",
                "start_seconds",
                "start",
            )
        ):
            return [result]

        return []

    for attribute in (
        "items",
        "words",
        "segments",
        "alignment",
    ):
        value = getattr(
            result,
            attribute,
            None,
        )

        if value is not None:
            return list(value)

    if any(
        hasattr(result, key)
        for key in (
            "start_time",
            "start_seconds",
            "start",
        )
    ):
        return [result]

    return []


def item_field(
    item,
    *names,
    default=None,
):
    for name in names:
        if isinstance(item, dict):
            value = item.get(name)
        else:
            value = getattr(
                item,
                name,
                None,
            )

        if value is not None:
            return value

    return default


# ============================================================
# RAY SERVE DEPLOYMENT
# ============================================================

@serve.deployment(
    name="align-consumer",
    max_ongoing_requests=int(os.getenv("AUDIO_03_TARGET_ONGOING_REQUESTS", "1")),
    ray_actor_options={
        "num_cpus": ALIGN_NUM_CPUS,
        "num_gpus": ALIGN_NUM_GPUS,
    },
    autoscaling_config=AutoscalingConfig(
        min_replicas=int(
            os.getenv("AUDIO_03_MIN_REPLICAS", "1")
        ),
        max_replicas=int(
            os.getenv("AUDIO_03_MAX_REPLICAS", "1")
        ),
        target_ongoing_requests=int(
            os.getenv(
                "AUDIO_03_TARGET_ONGOING_REQUESTS",
                "1",
            )
        ),
    ),
)
@task_consumer(
    task_processor_config=processor_config
)
class AlignConsumer:

    def __init__(self):
        self.device = resolve_device()
        self.dtype = resolve_dtype(self.device)

        logger.info(
            "Loading forced aligner model=%s device=%s dtype=%s",
            ALIGN_MODEL_NAME,
            self.device,
            self.dtype,
        )

        self.aligner = self._load_model()
        self.minio = create_minio_client()

        logger.info(
            "Forced aligner ready model=%s device=%s",
            ALIGN_MODEL_NAME,
            self.device,
        )

    # --------------------------------------------------------
    # MODEL INITIALIZATION
    # --------------------------------------------------------

    def _load_model(self):
        aligner = Qwen3ForcedAligner.from_pretrained(
            ALIGN_MODEL_NAME,
            dtype=self.dtype,
        )

        aligner.model.to(self.device)
        aligner.model.eval()

        aligner.device = torch.device(
            self.device
        )

        if ALIGN_COMPILE:
            logger.info(
                "Compiling forced aligner backend=%s "
                "fullgraph=%s dynamic=%s",
                ALIGN_COMPILE_BACKEND,
                ALIGN_COMPILE_FULLGRAPH,
                ALIGN_COMPILE_DYNAMIC,
            )

            aligner.model.forward = torch.compile(
                aligner.model.forward,
                backend=ALIGN_COMPILE_BACKEND,
                fullgraph=ALIGN_COMPILE_FULLGRAPH,
                dynamic=ALIGN_COMPILE_DYNAMIC,
            )

        return aligner

    # --------------------------------------------------------
    # MINIO AUDIO DOWNLOAD
    # --------------------------------------------------------

    def _download_audio(self, object_key):
        suffix = Path(object_key).suffix or ".audio"

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

            logger.info(
                "Downloaded audio bucket=%s key=%s",
                MINIO_BUCKET,
                object_key,
            )

            return local_path

        except Exception:
            Path(local_path).unlink(
                missing_ok=True
            )

            logger.exception(
                "Audio download failed key=%s",
                object_key,
            )

            raise

    # --------------------------------------------------------
    # AUDIO SEGMENT EXTRACTION
    # --------------------------------------------------------

    def _read_segment_batch(
        self,
        source,
        segments,
        sample_rate,
    ):
        audio_batch = []

        for segment in segments:
            start_frame = max(
                0,
                int(
                    segment.start_seconds * sample_rate
                ),
            )

            end_frame = min(
                len(source),
                max(
                    start_frame,
                    int(
                        segment.end_seconds * sample_rate
                    ),
                ),
            )

            if end_frame <= start_frame:
                raise ValueError(
                    f"Invalid audio range segment="
                    f"{segment.segment_index}"
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
                raise ValueError(
                    f"Empty audio segment="
                    f"{segment.segment_index}"
                )

            audio = resample_audio(
                audio,
                source_rate=sample_rate,
                target_rate=ALIGN_SAMPLE_RATE,
            )

            audio_batch.append(
                (audio, ALIGN_SAMPLE_RATE)
            )

        return audio_batch

    # --------------------------------------------------------
    # ALIGNMENT INFERENCE
    # --------------------------------------------------------

    def _align_batch(
        self,
        audio_batch,
        transcripts,
        languages,
    ):
        if not (
            len(audio_batch)
            == len(transcripts)
            == len(languages)
        ):
            raise ValueError(
                "Audio, transcript and language batch sizes differ"
            )

        normalized_languages = [
            aligner_language(language)
            for language in languages
        ]

        started = time.perf_counter()

        results = self.aligner.align(
            audio=audio_batch,
            text=transcripts,
            language=normalized_languages,
        )

        if not isinstance(
            results,
            (list, tuple),
        ):
            raise RuntimeError(
                f"Unexpected alignment result type: "
                f"{type(results).__name__}"
            )

        if len(results) != len(audio_batch):
            raise RuntimeError(
                f"Alignment result mismatch: "
                f"submitted={len(audio_batch)}, "
                f"received={len(results)}"
            )

        logger.info(
            "Alignment inference batch=%d elapsed=%.2fs",
            len(audio_batch),
            time.perf_counter() - started,
        )

        return results

    # --------------------------------------------------------
    # WORD ROW GENERATION
    # --------------------------------------------------------
    def _word_rows(self, segment, result):
        items = result_items(result)

        if not items:
            raise ValueError(
                f"No word alignment returned for "
                f"segment={segment.segment_index}"
            )

        rows = []

        segment_start = float(segment.start_seconds)
        segment_end = float(segment.end_seconds)
        segment_duration = segment_end - segment_start

        if segment_duration <= 0:
            raise ValueError(
                f"Invalid segment duration: "
                f"segment={segment.segment_index}"
            )

        previous_start = -1.0
        previous_end = -1.0

        # Small numerical differences are acceptable.
        BOUNDARY_TOLERANCE = 0.15

        # Allow a tiny duration for otherwise valid words.
        MIN_WORD_DURATION = 0.005

        for item in items:
            text = item_field(
                item,
                "text",
                "word",
                "token",
                default="",
            )

            text = str(text or "").strip()

            if not text:
                continue

            start_time = item_field(
                item,
                "start_time",
                "start_seconds",
                "start",
            )

            end_time = item_field(
                item,
                "end_time",
                "end_seconds",
                "end",
            )

            if start_time is None or end_time is None:
                logger.warning(
                    "Skipping word without timestamps: "
                    "job=%s segment=%s word=%r",
                    segment.job_id,
                    segment.segment_index,
                    text,
                )
                continue

            start_time = float(start_time)
            end_time = float(end_time)

            if not (
                np.isfinite(start_time)
                and np.isfinite(end_time)
            ):
                logger.warning(
                    "Skipping non-finite timestamps: "
                    "segment=%s word=%r start=%s end=%s",
                    segment.segment_index,
                    text,
                    start_time,
                    end_time,
                )
                continue

            if start_time < 0 or end_time <= start_time:
                logger.warning(
                    "Skipping invalid timestamps: "
                    "segment=%s word=%r start=%.3f end=%.3f",
                    segment.segment_index,
                    text,
                    start_time,
                    end_time,
                )
                continue

            # Qwen3 Forced Aligner timestamps are expected
            # to be relative to the supplied audio segment.
            if start_time > segment_duration + BOUNDARY_TOLERANCE:
                logger.warning(
                    "Skipping word outside segment: "
                    "job=%s segment=%s word=%r "
                    "relative_start=%.3f relative_end=%.3f "
                    "duration=%.3f",
                    segment.job_id,
                    segment.segment_index,
                    text,
                    start_time,
                    end_time,
                    segment_duration,
                )
                continue

            # Convert segment-relative timestamps to absolute.
            absolute_start = segment_start + start_time
            absolute_end = segment_start + end_time

            # Clamp small overruns.
            if absolute_start > segment_end:
                logger.warning(
                    "Clamping word start: segment=%s word=%r "
                    "start=%.3f boundary=%.3f",
                    segment.segment_index,
                    text,
                    absolute_start,
                    segment_end,
                )
                absolute_start = segment_end

            if absolute_end > segment_end:
                logger.warning(
                    "Clamping word end: segment=%s word=%r "
                    "end=%.3f boundary=%.3f",
                    segment.segment_index,
                    text,
                    absolute_end,
                    segment_end,
                )
                absolute_end = segment_end

            # Ensure timestamps remain within segment.
            absolute_start = max(
                segment_start,
                absolute_start,
            )

            absolute_end = min(
                segment_end,
                absolute_end,
            )

            # Preserve monotonic ordering.
            if absolute_start < previous_start:
                logger.warning(
                    "Skipping non-monotonic word: "
                    "segment=%s word=%r",
                    segment.segment_index,
                    text,
                )
                continue

            if absolute_start < previous_end:
                absolute_start = previous_end

            if absolute_end <= absolute_start:
                logger.warning(
                    "Skipping zero-duration word: "
                    "segment=%s word=%r",
                    segment.segment_index,
                    text,
                )
                continue

            if (
                absolute_end - absolute_start
                < MIN_WORD_DURATION
            ):
                logger.warning(
                    "Skipping extremely short word: "
                    "segment=%s word=%r",
                    segment.segment_index,
                    text,
                )
                continue

            rows.append({
                "job_id": segment.job_id,
                "segment_id": segment.id,
                "word_index": len(rows),
                "text": text,
                "start_seconds": absolute_start,
                "end_seconds": absolute_end,
            })

            previous_start = absolute_start
            previous_end = absolute_end

        if not rows:
            raise ValueError(
                f"No valid word timestamps after filtering: "
                f"segment={segment.segment_index}, "
                f"duration={segment_duration:.3f}s"
            )

        return rows

    
    # --------------------------------------------------------
    # DATABASE PERSISTENCE
    # --------------------------------------------------------

    def _persist_segment_result(
        self,
        session,
        segment,
        result,
        TranscriptWord,
    ):
        rows = self._word_rows(
            segment,
            result,
        )

        # Idempotent persistence for task retries.
        session.exec(
            delete(TranscriptWord).where(
                TranscriptWord.segment_id == segment.id
            )
        )

        for row in rows:
            session.add(
                TranscriptWord(**row)
            )

        segment.alignment_status = "completed"

        session.add(segment)

        return len(rows)

    # --------------------------------------------------------
    # MAIN TASK
    # --------------------------------------------------------

    @task_handler(name="process")
    def process(self, data):
        from db.db import engine
        from db.models import (
            Job,
            JobStage,
            JobStatus,
            TranscriptSegment,
            TranscriptWord,
            utc_now,
        )

        if (
            not isinstance(data, dict)
            or not data.get("job_id")
        ):
            raise ValueError(
                "Expected payload containing job_id"
            )

        job_id = UUID(
            str(data["job_id"])
        )

        started = time.perf_counter()

        temp_audio_path = None

        total_segments = 0
        completed_segments = 0
        failed_segments = 0
        total_words = 0
        untranscribed_count = 0

        try:
            # ====================================================
            # 1. LOAD JOB
            # ====================================================

            with Session(engine) as session:
                job = session.get(
                    Job,
                    job_id,
                )

                if job is None:
                    raise ValueError(
                        f"Job not found: {job_id}"
                    )

                if job.status in (
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                ):
                    logger.info(
                        "Skipping terminal job=%s status=%s",
                        job_id,
                        enum_value(job.status),
                    )

                    return {
                        "job_id": str(job_id),
                        "status": enum_value(job.status),
                    }

                audio_object_key = getattr(
                    job,
                    "audio_object_key",
                    None,
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

            logger.info(
                "Starting alignment job=%s",
                job_id,
            )

            # ====================================================
            # 2. RESOLVE AUDIO
            # ====================================================

            if audio_object_key:
                temp_audio_path = self._download_audio(
                    str(audio_object_key)
                )

                audio_path = Path(
                    temp_audio_path
                )

            elif legacy_audio_path:
                audio_path = Path(
                    legacy_audio_path
                )

            else:
                raise ValueError(
                    f"Job {job_id} has no audio object key or path"
                )

            if not audio_path.is_file():
                raise FileNotFoundError(
                    str(audio_path)
                )

            # ====================================================
            # 3. LOAD TRANSCRIPT SEGMENTS
            # ====================================================

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

                total_segments = len(segments)

                pending_indexes = [
                    segment.segment_index
                    for segment in segments
                    if (
                        segment.text
                        and segment.text.strip()
                        and segment.alignment_status != "completed"
                    )
                ]

                logger.info(
                    "Alignment job=%s total=%d pending=%d batch=%d",
                    job_id,
                    total_segments,
                    len(pending_indexes),
                    ALIGN_BATCH_SIZE,
                )

                # ====================================================
                # 4. OPEN AUDIO
                # ====================================================

                with sf.SoundFile(
                    str(audio_path)
                ) as source:

                    sample_rate = source.samplerate

                    if sample_rate <= 0:
                        raise ValueError(
                            f"Invalid sample rate={sample_rate}"
                        )

                    # ====================================================
                    # 5. PROCESS BATCHES
                    # ====================================================

                    for batch_number, batch_indexes in enumerate(
                        batched(
                            pending_indexes,
                            ALIGN_BATCH_SIZE,
                        ),
                        start=1,
                    ):
                        batch_started = time.perf_counter()

                        batch_segments = session.exec(
                            select(TranscriptSegment)
                            .where(
                                TranscriptSegment.job_id == job_id,
                                TranscriptSegment.segment_index.in_(
                                    batch_indexes
                                ),
                            )
                            .order_by(
                                TranscriptSegment.segment_index
                            )
                        ).all()

                        batch_segments = [
                            segment
                            for segment in batch_segments
                            if (
                                segment.text
                                and segment.text.strip()
                                and segment.alignment_status != "completed"
                            )
                        ]

                        if not batch_segments:
                            continue

                        # ------------------------------------------------
                        # 5a. Prepare audio
                        # ------------------------------------------------

                        retry_items = [
                            (
                                segment.id,
                                segment.segment_index,
                            )
                            for segment in batch_segments
                        ]

                        # ------------------------------------------------
                        # 5b. Prepare transcripts and language fallbacks
                        # ------------------------------------------------

                        audio_batch = self._read_segment_batch(
                            source,
                            batch_segments,
                            sample_rate,
                        )

                        transcripts = [
                            segment.text.strip()
                            for segment in batch_segments
                        ]

                        languages = [
                            aligner_language(segment.language)
                            for segment in batch_segments
                        ]

                        logger.info(
                            "Batch language mapping job=%s batch=%d languages=%s",
                            job_id,
                            batch_number,
                            languages,
                        )

                        # ------------------------------------------------
                        # 5c. Batch inference
                        # ------------------------------------------------

                        try:
                            results = self._align_batch(
                                audio_batch=audio_batch,
                                transcripts=transcripts,
                                languages=languages,
                            )

                            batch_word_count = 0

                            for segment, result in zip(
                                batch_segments,
                                results,
                            ):
                                batch_word_count += (
                                    self._persist_segment_result(
                                        session,
                                        segment,
                                        result,
                                        TranscriptWord,
                                    )
                                )

                            job = session.get(
                                Job,
                                job_id,
                            )

                            if job is not None:
                                job.updated_at = utc_now()
                                session.add(job)

                            session.commit()

                            total_words += batch_word_count

                            logger.info(
                                "Batch completed job=%s batch=%d "
                                "segments=%d words=%d elapsed=%.2fs",
                                job_id,
                                batch_number,
                                len(batch_segments),
                                batch_word_count,
                                time.perf_counter() - batch_started,
                            )

                        except Exception:
                            logger.exception(
                                "Batch alignment failed job=%s batch=%d; "
                                "retrying individual segments",
                                job_id,
                                batch_number,
                            )

                            session.rollback()

                            # ------------------------------------------------
                            # 5d. Individual fallback
                            # ------------------------------------------------

                            for segment_id, segment_index in retry_items:

                                segment = session.exec(
                                    select(TranscriptSegment).where(
                                        TranscriptSegment.id == segment_id
                                    )
                                ).first()

                                if segment is None:
                                    continue

                                if (
                                    segment.alignment_status
                                    == "completed"
                                ):
                                    continue

                                language = aligner_language(
                                    segment.language
                                )

                                try:
                                    audio_batch = self._read_segment_batch(
                                        source,
                                        [segment],
                                        sample_rate,
                                    )

                                    results = self._align_batch(
                                        audio_batch=audio_batch,
                                        transcripts=[
                                            segment.text.strip()
                                        ],
                                        languages=[
                                            language
                                        ],
                                    )

                                    word_count = (
                                        self._persist_segment_result(
                                            session,
                                            segment,
                                            results[0],
                                            TranscriptWord,
                                        )
                                    )

                                    job = session.get(
                                        Job,
                                        job_id,
                                    )

                                    if job is not None:
                                        job.updated_at = utc_now()
                                        session.add(job)

                                    session.commit()

                                    total_words += word_count

                                    logger.info(
                                        "Individual alignment succeeded "
                                        "job=%s segment=%s words=%d "
                                        "language=%s",
                                        job_id,
                                        segment_index,
                                        word_count,
                                        language,
                                    )

                                except Exception:
                                    session.rollback()

                                    logger.exception(
                                        "Individual alignment failed "
                                        "job=%s segment=%s language=%s",
                                        job_id,
                                        segment_index,
                                        language,
                                    )

                                    failed_segment = session.exec(
                                        select(TranscriptSegment).where(
                                            TranscriptSegment.id
                                            == segment_id
                                        )
                                    ).first()

                                    if failed_segment is not None:
                                        failed_segment.alignment_status = (
                                            "failed"
                                        )

                                        session.add(
                                            failed_segment
                                        )

                                        session.commit()

                # ====================================================
                # 6. FINAL STATUS
                # ====================================================

                final_segments = session.exec(
                    select(TranscriptSegment)
                    .where(
                        TranscriptSegment.job_id == job_id
                    )
                    .order_by(
                        TranscriptSegment.segment_index
                    )
                ).all()

                completed_segments = sum(
                    1
                    for segment in final_segments
                    if (
                        segment.alignment_status
                        == "completed"
                    )
                )

                failed_segments = sum(
                    1
                    for segment in final_segments
                    if (
                        segment.text
                        and segment.text.strip()
                        and segment.alignment_status != "completed"
                    )
                )

                untranscribed_count = sum(
                    1
                    for segment in final_segments
                    if (
                        not segment.text
                        or not segment.text.strip()
                    )
                )

                job = session.get(
                    Job,
                    job_id,
                )

                if job is None:
                    raise RuntimeError(
                        f"Job disappeared: {job_id}"
                    )

                # The job completes even if some individual
                # alignments fail. Original transcript languages
                # remain unchanged.
                job.status = JobStatus.COMPLETED
                job.stage = JobStage.COMPLETED
                job.updated_at = utc_now()

                if failed_segments:
                    job.error = (
                        f"Alignment completed with "
                        f"{failed_segments} failed segments "
                        f"out of {len(final_segments)}."
                    )
                else:
                    job.error = None

                session.add(job)
                session.commit()

            # ====================================================
            # 7. RESULT
            # ====================================================

            elapsed = time.perf_counter() - started

            logger.info(
                "Alignment completed job=%s "
                "total=%d completed=%d failed=%d "
                "untranscribed=%d words=%d elapsed=%.2fs",
                job_id,
                total_segments,
                completed_segments,
                failed_segments,
                untranscribed_count,
                total_words,
                elapsed,
            )

            return {
                "job_id": str(job_id),
                "status": "completed",
                "stage": "completed",
                "segments": total_segments,
                "completed_segments": completed_segments,
                "failed_segments": failed_segments,
                "untranscribed_segments": untranscribed_count,
                "words": total_words,
                "elapsed_seconds": round(
                    elapsed,
                    2,
                ),
            }

        except Exception as exc:
            logger.exception(
                "Alignment job failed job=%s",
                job_id,
            )

            try:
                with Session(engine) as session:
                    job = session.get(
                        Job,
                        job_id,
                    )

                    if (
                        job is not None
                        and job.status != JobStatus.COMPLETED
                    ):
                        job.status = JobStatus.PROCESSING
                        job.stage = JobStage.ALIGN
                        job.error = str(exc)[:4000]
                        job.updated_at = utc_now()

                        session.add(job)
                        session.commit()

            except Exception:
                logger.exception(
                    "Could not persist alignment error job=%s",
                    job_id,
                )

            raise

        finally:
            if temp_audio_path:
                Path(temp_audio_path).unlink(
                    missing_ok=True
                )

                logger.info(
                    "Cleaned temporary audio job=%s",
                    job_id,
                )


app = AlignConsumer.bind()
