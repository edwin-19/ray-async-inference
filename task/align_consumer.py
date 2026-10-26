
import logging
import math
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
from dotenv import load_dotenv
from ray import serve
from ray.serve.config import AutoscalingConfig
from ray.serve.schema import CeleryAdapterConfig, TaskProcessorConfig
from ray.serve.task_consumer import task_consumer, task_handler

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("align.consumer")

ALIGN_QUEUE = os.getenv("ALIGN_QUEUE", "align_queue")
ALIGN_MODEL_NAME = os.getenv(
    "ALIGN_MODEL_NAME",
    "Qwen/Qwen3-ForcedAligner-0.6B",
)
ALIGN_BATCH_SIZE = max(1, int(os.getenv("ALIGN_BATCH_SIZE", "4")))
ALIGN_NUM_GPUS = float(os.getenv("ALIGN_NUM_GPUS_PER_REPLICA", "1"))
ALIGN_NUM_CPUS = int(os.getenv("ALIGN_NUM_CPUS_PER_REPLICA", "1"))

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
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() in {
    "1", "true", "yes"
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


def create_minio_client():
    if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        raise RuntimeError(
            "MinIO credentials are missing. Configure "
            "MINIO_ACCESS_KEY and MINIO_SECRET_KEY."
        )

    return boto3.client(
        "s3",
        endpoint_url=f"http{'s' if MINIO_SECURE else ''}://{MINIO_ENDPOINT}",
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )


processor_config = TaskProcessorConfig(
    queue_name=ALIGN_QUEUE,
    adapter_config=CeleryAdapterConfig(
        broker_url=redis_url(0),
        backend_url=redis_url(1),
    ),
)


def batched(items, size):
    iterator = iter(items)

    while True:
        batch = list(islice(iterator, size))
        if not batch:
            break
        yield batch


@serve.deployment(
    name="align-consumer",
    max_ongoing_requests=1,
    ray_actor_options={
        "num_cpus": ALIGN_NUM_CPUS,
        "num_gpus": ALIGN_NUM_GPUS,
    },
    autoscaling_config=AutoscalingConfig(
        min_replicas=1,
        max_replicas=1,
        target_ongoing_requests=1,
    ),
)
@task_consumer(task_processor_config=processor_config)
class AlignConsumer:

    def __init__(self):
        import torch
        from qwen_asr import Qwen3ForcedAligner

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        logger.info(
            "Loading aligner model=%s device=%s batch_size=%d",
            ALIGN_MODEL_NAME,
            device,
            ALIGN_BATCH_SIZE,
        )

        self.model = Qwen3ForcedAligner.from_pretrained(
            ALIGN_MODEL_NAME,
            dtype=dtype,
            device_map=device,
        )

        # Lazily create the client so model initialization errors and
        # MinIO configuration errors can be diagnosed independently.
        self.minio = None

        logger.info("Aligner loaded successfully")

    def _get_minio_client(self):
        if self.minio is None:
            self.minio = create_minio_client()
        return self.minio

    def _download_audio(self, object_key: str) -> str:
        """Download a MinIO object to a worker-local temporary file."""
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

    @staticmethod
    def _read_audio_batch(source, segments, sample_rate):
        audios = []

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

            if end_frame <= start_frame:
                raise ValueError(
                    f"Segment {segment.segment_index} has no audio frames"
                )

            source.seek(start_frame)
            audio = source.read(
                frames=end_frame - start_frame,
                dtype="float32",
                always_2d=False,
            )

            if audio.size == 0:
                raise ValueError(
                    f"Segment {segment.segment_index} contains empty audio"
                )

            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            audios.append((audio, sample_rate))

        return audios

    @staticmethod
    def _word_rows(segment, result):
        """
        Validate aligner output and convert relative timestamps into
        absolute audio timestamps, constrained to the segment boundaries.
        """
        items = list(result.items)

        if not items:
            raise ValueError("Aligner returned no word timestamps")

        rows = []
        segment_start = float(segment.start_seconds)
        segment_end = float(segment.end_seconds)

        for word_index, item in enumerate(items):
            word_text = str(item.text).strip()
            if not word_text:
                raise ValueError(
                    f"Aligner returned empty text for word {word_index}"
                )

            relative_start = float(item.start_time)
            relative_end = float(item.end_time)

            if not (
                math.isfinite(relative_start)
                and math.isfinite(relative_end)
            ):
                raise ValueError(
                    f"Non-finite timestamp for word {word_index}"
                )

            if relative_start < 0 or relative_end < relative_start:
                raise ValueError(
                    f"Invalid relative timestamps for word {word_index}: "
                    f"{relative_start}, {relative_end}"
                )

            start = min(
                segment_end,
                max(segment_start, segment_start + relative_start),
            )
            end = min(
                segment_end,
                max(start, segment_start + relative_end),
            )

            rows.append({
                "word_index": word_index,
                "text": word_text,
                "start_seconds": start,
                "end_seconds": end,
            })

        return rows

    @staticmethod
    def _replace_words(session, TranscriptWord, segment, job_id, rows):
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
            raise ValueError("Expected payload containing job_id")

        job_id = UUID(str(data["job_id"]))
        started = time.perf_counter()
        temp_audio_path = None

        try:
            with Session(engine) as session:
                job = session.exec(
                    select(Job).where(Job.id == job_id)
                ).first()

                if job is None:
                    raise ValueError(f"Job not found: {job_id}")

                if job.status == JobStatus.FAILED:
                    return {
                        "job_id": str(job_id),
                        "status": "failed",
                    }

                audio_object_key = getattr(
                    job, "audio_object_key", None
                )
                audio_path = getattr(job, "audio_path", None)

                if audio_object_key:
                    temp_audio_path = self._download_audio(
                        audio_object_key
                    )
                    audio_path = temp_audio_path

                if not audio_path or not os.path.isfile(audio_path):
                    raise FileNotFoundError(
                        "Audio is unavailable. Expected a valid legacy "
                        "audio_path or an audio_object_key in MinIO."
                    )

                job.status = JobStatus.PROCESSING
                job.stage = JobStage.ALIGN
                job.error = None
                job.updated_at = utc_now()
                session.add(job)
                session.commit()

                segments = session.exec(
                    select(TranscriptSegment)
                    .where(TranscriptSegment.job_id == job_id)
                    .where(TranscriptSegment.text.is_not(None))
                    .where(TranscriptSegment.text != "")
                    .where(
                        TranscriptSegment.alignment_status != "completed"
                    )
                    .order_by(TranscriptSegment.segment_index)
                ).all()

                logger.info(
                    "Alignment started job=%s pending_segments=%d "
                    "batch_size=%d",
                    job_id,
                    len(segments),
                    ALIGN_BATCH_SIZE,
                )

                with sf.SoundFile(str(audio_path)) as source:
                    sample_rate = source.samplerate

                    for batch_number, batch in enumerate(
                        batched(segments, ALIGN_BATCH_SIZE),
                        start=1,
                    ):
                        missing_language = [
                            segment for segment in batch
                            if not segment.language
                        ]
                        valid = [
                            segment for segment in batch
                            if segment.language
                        ]

                        for segment in missing_language:
                            segment.alignment_status = "failed"
                            session.add(segment)

                        if missing_language:
                            session.commit()

                            for segment in missing_language:
                                logger.warning(
                                    "Alignment skipped job=%s segment=%s: "
                                    "ASR language is missing",
                                    job_id,
                                    segment.segment_index,
                                )

                        if not valid:
                            continue

                        batch_started = time.perf_counter()

                        try:
                            audio_batch = self._read_audio_batch(
                                source, valid, sample_rate
                            )

                            results = self.model.align(
                                audio=audio_batch,
                                text=[segment.text for segment in valid],
                                language=[
                                    segment.language for segment in valid
                                ],
                            )

                            if len(results) != len(valid):
                                raise RuntimeError(
                                    f"Aligner returned {len(results)} "
                                    f"results for {len(valid)} segments"
                                )

                            # Validate every result before modifying any
                            # word rows, so a malformed result rolls back
                            # the whole batch cleanly.
                            prepared = [
                                (
                                    segment,
                                    self._word_rows(segment, result),
                                )
                                for segment, result in zip(valid, results)
                            ]

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
                                "Alignment batch succeeded job=%s batch=%d "
                                "segments=%d elapsed=%.2fs",
                                job_id,
                                batch_number,
                                len(valid),
                                time.perf_counter() - batch_started,
                            )

                        except Exception:
                            session.rollback()

                            logger.exception(
                                "Alignment batch failed; retrying segments "
                                "individually job=%s batch=%d",
                                job_id,
                                batch_number,
                            )

                            # Retry each segment independently. Reload the
                            # row after rollback to avoid using stale ORM
                            # state from the failed batch.
                            for original in valid:
                                segment_id = original.id
                                segment_index = original.segment_index

                                try:
                                    segment = session.get(
                                        TranscriptSegment, segment_id
                                    )

                                    if segment is None:
                                        raise ValueError(
                                            "Transcript segment disappeared"
                                        )

                                    audio = self._read_audio_batch(
                                        source, [segment], sample_rate
                                    )

                                    results = self.model.align(
                                        audio=audio,
                                        text=[segment.text],
                                        language=[segment.language],
                                    )

                                    if len(results) != 1:
                                        raise RuntimeError(
                                            "Aligner did not return exactly "
                                            "one result"
                                        )

                                    rows = self._word_rows(
                                        segment, results[0]
                                    )

                                    self._replace_words(
                                        session,
                                        TranscriptWord,
                                        segment,
                                        job_id,
                                        rows,
                                    )

                                    session.commit()

                                except Exception as item_exc:
                                    session.rollback()

                                    segment = session.get(
                                        TranscriptSegment, segment_id
                                    )

                                    if segment is not None:
                                        segment.alignment_status = "failed"
                                        session.add(segment)
                                        session.commit()

                                    logger.warning(
                                        "Alignment failed job=%s "
                                        "segment=%s error=%s",
                                        job_id,
                                        segment_index,
                                        str(item_exc)[:1000],
                                    )

                # Compute totals from persisted state, including failures
                # from previous attempts, rather than run-local counters.
                completed_count = session.exec(
                    select(func.count())
                    .select_from(TranscriptSegment)
                    .where(TranscriptSegment.job_id == job_id)
                    .where(
                        TranscriptSegment.alignment_status == "completed"
                    )
                ).one()

                failed_count = session.exec(
                    select(func.count())
                    .select_from(TranscriptSegment)
                    .where(TranscriptSegment.job_id == job_id)
                    .where(
                        TranscriptSegment.alignment_status == "failed"
                    )
                ).one()

                job = session.exec(
                    select(Job).where(Job.id == job_id)
                ).first()

                if job is None:
                    raise ValueError(
                        f"Job disappeared during alignment: {job_id}"
                    )

                now = utc_now()
                job.status = JobStatus.COMPLETED
                job.stage = JobStage.COMPLETED
                job.updated_at = now

                if job.completed_at is None:
                    job.completed_at = now

                if failed_count:
                    job.error = (
                        f"Alignment completed with {failed_count} failed "
                        "segments. Inspect transcript_segments."
                        "alignment_status and requeue the job to retry."
                    )
                else:
                    job.error = None

                session.add(job)
                session.commit()

            elapsed = time.perf_counter() - started

            logger.info(
                "Alignment finished job=%s aligned_segments=%d "
                "failed_segments=%d elapsed=%.2fs",
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
            logger.exception("Alignment job failed job=%s", job_id)

            # Preserve the job as retryable if a global failure occurs,
            # such as an inaccessible MinIO object or unreadable audio.
            try:
                with Session(engine) as session:
                    job = session.exec(
                        select(Job).where(Job.id == job_id)
                    ).first()

                    if job is not None and job.status != JobStatus.FAILED:
                        job.status = JobStatus.PROCESSING
                        job.stage = JobStage.ALIGN
                        job.error = str(exc)[:4000]
                        job.updated_at = utc_now()
                        session.add(job)
                        session.commit()
            except Exception:
                logger.exception(
                    "Could not update job error state job=%s", job_id
                )

            raise

        finally:
            if temp_audio_path:
                try:
                    os.unlink(temp_audio_path)
                except OSError:
                    logger.warning(
                        "Could not remove temporary audio file: %s",
                        temp_audio_path,
                    )


app = AlignConsumer.bind()
