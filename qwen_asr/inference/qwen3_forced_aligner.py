import logging
import math
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
from scipy.signal import resample_poly
from transformers import (
    AutoModelForTokenClassification,
    AutoProcessor,
)

from ray import serve
from ray.serve.config import AutoscalingConfig
from ray.serve.schema import CeleryAdapterConfig, TaskProcessorConfig
from ray.serve.task_consumer import task_consumer, task_handler


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("align.consumer")


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

ALIGN_QUEUE = os.getenv("ALIGN_QUEUE", "align_queue")

ALIGN_MODEL_NAME = os.getenv(
    "ALIGN_MODEL_NAME",
    "Qwen/Qwen3-ForcedAligner-0.6B-hf",
)

ALIGN_BATCH_SIZE = max(
    1,
    int(os.getenv("ALIGN_BATCH_SIZE", "4")),
)

ALIGN_DEVICE = os.getenv("ALIGN_DEVICE", "auto")

ALIGN_NUM_GPUS = float(
    os.getenv("ALIGN_NUM_GPUS_PER_REPLICA", "1")
)

ALIGN_NUM_CPUS = int(
    os.getenv("ALIGN_NUM_CPUS_PER_REPLICA", "2")
)

# torch.compile recompiles for every new input shape. Segments have
# variable lengths, so keep this off unless you bucket/pad lengths.
ALIGN_COMPILE = os.getenv(
    "ALIGN_COMPILE",
    "false",
).lower() in {"1", "true", "yes"}

ALIGN_COMPILE_MODE = os.getenv(
    "ALIGN_COMPILE_MODE",
    "default",
)

# What to do with segments whose language the HF processor rejects
# (for example Malay):
#   fallback - align using ALIGN_FALLBACK_LANGUAGE. In the original
#              qwen_asr implementation the language only selects the
#              word tokenizer (Japanese and Korean have special ones,
#              every other language is split on whitespace), and the
#              model never receives the language. So "English" gives
#              the same tokenization the original uses for Malay.
#              Only suitable for space-delimited languages.
#   skip     - mark the segment "skipped" and leave it unaligned
ALIGN_UNSUPPORTED_POLICY = os.getenv(
    "ALIGN_UNSUPPORTED_POLICY",
    "fallback",
).lower()

ALIGN_FALLBACK_LANGUAGE = os.getenv(
    "ALIGN_FALLBACK_LANGUAGE",
    "English",
)

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

MINIO_SECURE = os.getenv(
    "MINIO_SECURE",
    "false",
).lower() in {"1", "true", "yes"}


# ------------------------------------------------------------------
# Redis
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# MinIO
# ------------------------------------------------------------------

def create_minio_client():
    if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        raise RuntimeError("MinIO credentials are missing")

    return boto3.client(
        "s3",
        endpoint_url=(
            f"http{'s' if MINIO_SECURE else ''}://{MINIO_ENDPOINT}"
        ),
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )


# ------------------------------------------------------------------
# Ray Serve task processor
# ------------------------------------------------------------------

processor_config = TaskProcessorConfig(
    queue_name=ALIGN_QUEUE,
    adapter_config=CeleryAdapterConfig(
        broker_url=redis_url(0),
        backend_url=redis_url(1),
    ),
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def aligner_language(language):
    """
    Map a segment language (code or name) to a language the aligner
    supports. Returns None if it is unsupported and the policy is
    "skip".
    """

    value = str(language or "").strip().lower()

    for code, name in SUPPORTED_LANGUAGES.items():
        if value in (code, name.lower()):
            return name

    if ALIGN_UNSUPPORTED_POLICY == "fallback":
        return ALIGN_FALLBACK_LANGUAGE

    return None


def batched(items, size):
    iterator = iter(items)

    while True:
        batch = list(islice(iterator, size))

        if not batch:
            break

        yield batch


# ------------------------------------------------------------------
# Alignment consumer
# ------------------------------------------------------------------

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
class AlignConsumer:

    # --------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------

    def __init__(self):
        try:
            self._load_model()
        except Exception:
            # Log the real startup error. Otherwise Ray's __del__
            # noise ("no attribute '_adapter'") hides it.
            logger.exception("AlignConsumer failed to initialize")
            raise

    def _load_model(self):
        if ALIGN_DEVICE == "auto":
            device = (
                "cuda:0"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            device = ALIGN_DEVICE

        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"{device} requested but CUDA is unavailable"
                )

            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            dtype = torch.float32

        self.device = device
        self.dtype = dtype

        logger.info(
            "Loading aligner model=%s device=%s dtype=%s",
            ALIGN_MODEL_NAME,
            device,
            dtype,
        )

        # Processor (handles feature extraction and padding).
        self.processor = AutoProcessor.from_pretrained(
            ALIGN_MODEL_NAME,
        )

        self.sample_rate = int(
            getattr(
                self.processor.feature_extractor,
                "sampling_rate",
                16000,
            )
        )

        # Model.
        self.model = AutoModelForTokenClassification.from_pretrained(
            ALIGN_MODEL_NAME,
            dtype=dtype,
        ).to(device)

        self.model.eval()

        if ALIGN_COMPILE and device.startswith("cuda"):
            logger.info(
                "Compiling aligner forward pass mode=%s",
                ALIGN_COMPILE_MODE,
            )

            self.model.forward = torch.compile(
                self.model.forward,
                mode=ALIGN_COMPILE_MODE,
            )

        self.minio = None

        logger.info(
            "Aligner loaded device=%s dtype=%s "
            "sample_rate=%d batch_size=%d compile=%s",
            self.device,
            self.dtype,
            self.sample_rate,
            ALIGN_BATCH_SIZE,
            ALIGN_COMPILE,
        )

    # --------------------------------------------------------------
    # MinIO client
    # --------------------------------------------------------------

    def _get_minio_client(self):
        if self.minio is None:
            self.minio = create_minio_client()

        return self.minio

    # --------------------------------------------------------------
    # Audio download
    # --------------------------------------------------------------

    def _download_audio(self, object_key: str) -> str:
        suffix = Path(object_key).suffix or ".audio"

        with tempfile.NamedTemporaryFile(
            prefix="align_",
            suffix=suffix,
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

        try:
            self._get_minio_client().download_file(
                MINIO_BUCKET,
                object_key,
                temp_path,
            )

            return temp_path

        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

            raise

    # --------------------------------------------------------------
    # Read and prepare audio batch
    # --------------------------------------------------------------

    def _read_audio_batch(
        self,
        source,
        segments,
        sample_rate,
    ):
        if sample_rate <= 0:
            raise ValueError(
                f"Invalid audio sample rate: {sample_rate}"
            )

        audios = []
        total_frames = len(source)

        for segment in segments:
            segment_start = float(segment.start_seconds)
            segment_end = float(segment.end_seconds)

            if not (
                math.isfinite(segment_start)
                and math.isfinite(segment_end)
            ):
                raise ValueError(
                    f"Invalid segment timestamps: "
                    f"{segment_start}, {segment_end}"
                )

            if segment_start < 0 or segment_end <= segment_start:
                raise ValueError(
                    f"Invalid segment boundaries: "
                    f"{segment_start}, {segment_end}"
                )

            start_frame = max(
                0,
                int(segment_start * sample_rate),
            )

            end_frame = min(
                total_frames,
                int(segment_end * sample_rate),
            )

            if end_frame <= start_frame:
                raise ValueError(
                    f"Segment {segment.segment_index} "
                    "has no audio frames"
                )

            source.seek(start_frame)

            audio = source.read(
                frames=end_frame - start_frame,
                dtype="float32",
                always_2d=False,
            )

            if audio.size == 0:
                raise ValueError(
                    f"Segment {segment.segment_index} "
                    "contains empty audio"
                )

            # Convert multichannel audio to mono.
            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            # Reject NaN and infinite samples.
            if not np.isfinite(audio).all():
                raise ValueError(
                    f"Segment {segment.segment_index} "
                    "contains non-finite audio samples"
                )

            # Resample to the model's expected sample rate.
            if sample_rate != self.sample_rate:
                divisor = math.gcd(
                    sample_rate,
                    self.sample_rate,
                )

                audio = resample_poly(
                    audio,
                    self.sample_rate // divisor,
                    sample_rate // divisor,
                )

            audio = np.ascontiguousarray(
                audio,
                dtype=np.float32,
            )

            if audio.size == 0:
                raise ValueError(
                    f"Segment {segment.segment_index} "
                    "became empty after resampling"
                )

            audios.append(audio)

        return audios

    # --------------------------------------------------------------
    # Native Transformers forced alignment
    # --------------------------------------------------------------

    def _align_batch(self, audios, segments):
        if not audios or not segments:
            return []

        if len(audios) != len(segments):
            raise ValueError(
                "Audio and transcript batch sizes do not match"
            )

        transcripts = [
            segment.text
            for segment in segments
        ]

        languages = [
            aligner_language(segment.language)
            for segment in segments
        ]

        if any(language is None for language in languages):
            raise ValueError(
                "Unsupported alignment language in batch: "
                f"{[segment.language for segment in segments]}"
            )

        # Prepare model inputs.
        aligner_inputs, word_lists = (
            self.processor.prepare_forced_aligner_inputs(
                audio=audios,
                transcript=transcripts,
                language=languages,
            )
        )

        # BatchFeature.to casts only floating-point tensors to the
        # model dtype; integer tensors such as input_ids are kept.
        aligner_inputs = aligner_inputs.to(
            self.device,
            self.dtype,
        )

        # Model inference.
        with torch.inference_mode():
            outputs = self.model(**aligner_inputs)

        # Decode word-level timestamps.
        results = self.processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=aligner_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=(
                self.model.config.timestamp_token_id
            ),
        )

        if len(results) != len(segments):
            raise RuntimeError(
                f"Aligner returned {len(results)} results "
                f"for {len(segments)} segments"
            )

        return results

    # --------------------------------------------------------------
    # Convert alignment output into database rows
    # --------------------------------------------------------------

    @staticmethod
    def _word_rows(segment, result):
        items = list(result)

        if not items:
            raise ValueError(
                "Aligner returned no word timestamps"
            )

        rows = []

        segment_start = float(segment.start_seconds)
        segment_end = float(segment.end_seconds)

        for word_index, item in enumerate(items):
            word_text = str(
                item["text"]
            ).strip()

            if not word_text:
                raise ValueError(
                    f"Empty aligned word at index {word_index}"
                )

            relative_start = float(
                item["start_time"]
            )

            relative_end = float(
                item["end_time"]
            )

            if not (
                math.isfinite(relative_start)
                and math.isfinite(relative_end)
            ):
                raise ValueError(
                    f"Non-finite timestamp at word {word_index}"
                )

            if (
                relative_start < 0
                or relative_end < relative_start
            ):
                raise ValueError(
                    f"Invalid timestamps at word {word_index}: "
                    f"{relative_start}, {relative_end}"
                )

            # Convert segment-relative timestamps to absolute
            # timestamps and constrain them to segment boundaries.
            start = min(
                segment_end,
                max(
                    segment_start,
                    segment_start + relative_start,
                ),
            )

            end = min(
                segment_end,
                max(
                    start,
                    segment_start + relative_end,
                ),
            )

            rows.append({
                "word_index": word_index,
                "text": word_text,
                "start_seconds": start,
                "end_seconds": end,
            })

        return rows

    # --------------------------------------------------------------
    # Replace persisted words for a segment
    # --------------------------------------------------------------

    @staticmethod
    def _replace_words(
        session,
        TranscriptWord,
        segment,
        job_id,
        rows,
    ):
        from sqlmodel import delete

        session.exec(
            delete(TranscriptWord).where(
                TranscriptWord.segment_id == segment.id
            )
        )

        for row in rows:
            session.add(
                TranscriptWord(
                    job_id=job_id,
                    segment_id=segment.id,
                    word_index=row["word_index"],
                    text=row["text"],
                    start_seconds=row["start_seconds"],
                    end_seconds=row["end_seconds"],
                )
            )

        segment.alignment_status = "completed"
        session.add(segment)

    # --------------------------------------------------------------
    # Mark segment failed
    # --------------------------------------------------------------

    @staticmethod
    def _mark_segment_failed(
        session,
        TranscriptSegment,
        segment_id,
    ):
        session.rollback()

        segment = session.get(
            TranscriptSegment,
            segment_id,
        )

        if segment is not None:
            segment.alignment_status = "failed"
            session.add(segment)
            session.commit()

    # --------------------------------------------------------------
    # Process one segment independently
    # --------------------------------------------------------------

    def _retry_segment(
        self,
        session,
        TranscriptSegment,
        TranscriptWord,
        job_id,
        segment_id,
        segment_index,
        source,
        sample_rate,
    ):
        try:
            segment = session.get(
                TranscriptSegment,
                segment_id,
            )

            if segment is None:
                raise ValueError(
                    f"Transcript segment disappeared: {segment_id}"
                )

            audio = self._read_audio_batch(
                source,
                [segment],
                sample_rate,
            )

            results = self._align_batch(
                audio,
                [segment],
            )

            if len(results) != 1:
                raise RuntimeError(
                    "Expected exactly one alignment result"
                )

            rows = self._word_rows(
                segment,
                results[0],
            )

            self._replace_words(
                session,
                TranscriptWord,
                segment,
                job_id,
                rows,
            )

            session.commit()

            logger.info(
                "Individual alignment succeeded "
                "job=%s segment=%s",
                job_id,
                segment_index,
            )

            return True

        except Exception as exc:
            session.rollback()

            logger.warning(
                "Individual alignment failed "
                "job=%s segment=%s error=%s",
                job_id,
                segment_index,
                str(exc)[:1000],
                exc_info=True,
            )

            try:
                self._mark_segment_failed(
                    session,
                    TranscriptSegment,
                    segment_id,
                )
            except Exception:
                session.rollback()

                logger.exception(
                    "Could not mark segment failed "
                    "job=%s segment=%s",
                    job_id,
                    segment_index,
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return False

    # --------------------------------------------------------------
    # Main task handler
    # --------------------------------------------------------------

    @task_handler(name="process")
    def process(self, data):
        from sqlmodel import Session, func, select

        from db.db import engine
        from db.models import (
            Job,
            JobStage,
            JobStatus,
            TranscriptSegment,
            TranscriptWord,
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
            with Session(engine) as session:

                # --------------------------------------------------
                # Load job
                # --------------------------------------------------

                job = session.exec(
                    select(Job).where(
                        Job.id == job_id
                    )
                ).first()

                if job is None:
                    raise ValueError(
                        f"Job not found: {job_id}"
                    )

                if job.status == JobStatus.FAILED:
                    return {
                        "job_id": str(job_id),
                        "status": "failed",
                    }

                audio_object_key = getattr(
                    job,
                    "audio_object_key",
                    None,
                )

                audio_path = getattr(
                    job,
                    "audio_path",
                    None,
                )

                # --------------------------------------------------
                # Download audio from MinIO if applicable
                # --------------------------------------------------

                if audio_object_key:
                    temp_audio_path = self._download_audio(
                        audio_object_key
                    )

                    audio_path = temp_audio_path

                if (
                    not audio_path
                    or not os.path.isfile(audio_path)
                ):
                    raise FileNotFoundError(
                        "Audio is unavailable"
                    )

                # --------------------------------------------------
                # Update job state
                # --------------------------------------------------

                job.status = JobStatus.PROCESSING
                job.stage = JobStage.ALIGN
                job.error = None
                job.updated_at = utc_now()

                session.add(job)
                session.commit()

                # --------------------------------------------------
                # Load pending segments
                # --------------------------------------------------

                segments = session.exec(
                    select(TranscriptSegment)
                    .where(
                        TranscriptSegment.job_id == job_id
                    )
                    .where(
                        TranscriptSegment.text.is_not(None)
                    )
                    .where(
                        TranscriptSegment.text != ""
                    )
                    .where(
                        TranscriptSegment.alignment_status
                        != "completed"
                    )
                    .where(
                        TranscriptSegment.alignment_status
                        != "skipped"
                    )
                    .order_by(
                        TranscriptSegment.segment_index
                    )
                ).all()

                logger.info(
                    "Alignment started job=%s "
                    "pending_segments=%d batch_size=%d",
                    job_id,
                    len(segments),
                    ALIGN_BATCH_SIZE,
                )

                # --------------------------------------------------
                # Process audio batches
                # --------------------------------------------------

                with sf.SoundFile(str(audio_path)) as source:
                    sample_rate = int(source.samplerate)

                    for batch_number, batch in enumerate(
                        batched(
                            segments,
                            ALIGN_BATCH_SIZE,
                        ),
                        start=1,
                    ):
                        # Classify segments before touching the model:
                        #   missing language     -> failed
                        #   unsupported language -> skipped
                        valid = []
                        changed = False

                        for segment in batch:
                            if not segment.language:
                                segment.alignment_status = "failed"
                                session.add(segment)
                                changed = True

                                logger.warning(
                                    "Segment=%s failed: "
                                    "language is missing",
                                    segment.segment_index,
                                )

                            elif aligner_language(
                                segment.language
                            ) is None:
                                segment.alignment_status = "skipped"
                                session.add(segment)
                                changed = True

                                logger.warning(
                                    "Segment=%s skipped: language "
                                    "%r is not supported by the "
                                    "aligner",
                                    segment.segment_index,
                                    segment.language,
                                )

                            else:
                                valid.append(segment)

                        if changed:
                            session.commit()

                        if not valid:
                            continue

                        batch_started = time.perf_counter()

                        try:
                            # Read audio.
                            audio_batch = self._read_audio_batch(
                                source,
                                valid,
                                sample_rate,
                            )

                            # Run forced alignment.
                            results = self._align_batch(
                                audio_batch,
                                valid,
                            )

                            # Validate all results before DB writes.
                            prepared = [
                                (
                                    segment,
                                    self._word_rows(
                                        segment,
                                        result,
                                    ),
                                )
                                for segment, result in zip(
                                    valid,
                                    results,
                                )
                            ]

                            # Persist successful batch.
                            for segment, rows in prepared:
                                self._replace_words(
                                    session,
                                    TranscriptWord,
                                    segment,
                                    job_id,
                                    rows,
                                )

                            job.updated_at = utc_now()
                            session.add(job)
                            session.commit()

                            logger.info(
                                "Alignment batch succeeded "
                                "job=%s batch=%d segments=%d "
                                "elapsed=%.2fs",
                                job_id,
                                batch_number,
                                len(valid),
                                time.perf_counter() - batch_started,
                            )

                        except Exception:
                            session.rollback()

                            logger.exception(
                                "Alignment batch failed; "
                                "retrying individually "
                                "job=%s batch=%d",
                                job_id,
                                batch_number,
                            )

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            # Save IDs before retrying because the
                            # transaction rollback expires ORM state.
                            retry_items = [
                                (
                                    segment.id,
                                    segment.segment_index,
                                )
                                for segment in valid
                            ]

                            for segment_id, segment_index in retry_items:
                                self._retry_segment(
                                    session=session,
                                    TranscriptSegment=TranscriptSegment,
                                    TranscriptWord=TranscriptWord,
                                    job_id=job_id,
                                    segment_id=segment_id,
                                    segment_index=segment_index,
                                    source=source,
                                    sample_rate=sample_rate,
                                )

                # --------------------------------------------------
                # Calculate persisted counts
                # --------------------------------------------------

                completed_count = session.exec(
                    select(func.count())
                    .select_from(TranscriptSegment)
                    .where(
                        TranscriptSegment.job_id == job_id
                    )
                    .where(
                        TranscriptSegment.alignment_status
                        == "completed"
                    )
                ).one()

                failed_count = session.exec(
                    select(func.count())
                    .select_from(TranscriptSegment)
                    .where(
                        TranscriptSegment.job_id == job_id
                    )
                    .where(
                        TranscriptSegment.alignment_status
                        == "failed"
                    )
                ).one()

                skipped_count = session.exec(
                    select(func.count())
                    .select_from(TranscriptSegment)
                    .where(
                        TranscriptSegment.job_id == job_id
                    )
                    .where(
                        TranscriptSegment.alignment_status
                        == "skipped"
                    )
                ).one()

                # --------------------------------------------------
                # Finalize job
                # --------------------------------------------------

                job = session.exec(
                    select(Job).where(
                        Job.id == job_id
                    )
                ).first()

                if job is None:
                    raise ValueError(
                        f"Job disappeared: {job_id}"
                    )

                now = utc_now()

                job.status = JobStatus.COMPLETED
                job.stage = JobStage.COMPLETED
                job.updated_at = now

                if job.completed_at is None:
                    job.completed_at = now

                notes = []

                if failed_count:
                    notes.append(f"{failed_count} failed")

                if skipped_count:
                    notes.append(
                        f"{skipped_count} skipped "
                        "(unsupported language)"
                    )

                job.error = (
                    "Alignment completed with "
                    + ", ".join(notes)
                    + " segments."
                    if notes
                    else None
                )

                session.add(job)
                session.commit()

            # ------------------------------------------------------
            # Return result
            # ------------------------------------------------------

            elapsed = time.perf_counter() - started

            logger.info(
                "Alignment finished job=%s "
                "aligned=%d failed=%d elapsed=%.2fs",
                job_id,
                completed_count,
                failed_count,
                elapsed,
            )

            return {
                "job_id": str(job_id),
                "status": "completed",
                "aligned_segments": completed_count,
                "failed_segments": failed_count,
                "elapsed_seconds": round(elapsed, 2),
            }

        except Exception as exc:
            logger.exception(
                "Alignment job failed job=%s",
                job_id,
            )

            # Keep the job in PROCESSING so the task system can
            # retry it if configured to do so. Do not mark it FAILED
            # prematurely when the task may be retried.
            try:
                with Session(engine) as session:
                    job = session.exec(
                        select(Job).where(
                            Job.id == job_id
                        )
                    ).first()

                    if (
                        job is not None
                        and job.status != JobStatus.FAILED
                    ):
                        job.status = JobStatus.PROCESSING
                        job.stage = JobStage.ALIGN
                        job.error = str(exc)[:4000]
                        job.updated_at = utc_now()

                        session.add(job)
                        session.commit()

            except Exception:
                logger.exception(
                    "Could not update job error state job=%s",
                    job_id,
                )

            raise

        finally:
            if temp_audio_path:
                try:
                    os.unlink(temp_audio_path)
                except OSError:
                    logger.warning(
                        "Could not remove temporary audio: %s",
                        temp_audio_path,
                    )


# ------------------------------------------------------------------
# Ray Serve application
# ------------------------------------------------------------------

app = AlignConsumer.bind()