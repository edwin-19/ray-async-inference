
import logging
import os
import time
from collections import Counter
from itertools import islice
from urllib.parse import quote
from uuid import UUID

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

        logger.info("Aligner loaded successfully")

    def _read_audio_batch(self, source, segments, sample_rate):
        audios = []

        for segment in segments:
            start_frame = max(
                0, int(segment.start_seconds * sample_rate)
            )
            end_frame = min(
                len(source),
                max(start_frame, int(segment.end_seconds * sample_rate)),
            )

            source.seek(start_frame)
            audio = source.read(
                frames=end_frame - start_frame,
                dtype="float32",
                always_2d=False,
            )

            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            audios.append((audio, sample_rate))

        return audios

    @task_handler(name="process")
    def process(self, data):
        from sqlmodel import Session, select, delete

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

        with Session(engine) as session:
            job = session.exec(
                select(Job).where(Job.id == job_id)
            ).first()

            if job is None:
                raise ValueError(f"Job not found: {job_id}")

            if job.status == JobStatus.FAILED:
                return {"job_id": str(job_id), "status": "failed"}

            # A completed job may be explicitly requeued to retry
            # segments with alignment_status='failed'.
            job.status = JobStatus.PROCESSING
            job.stage = JobStage.ALIGN
            job.error = None
            job.updated_at = utc_now()
            session.add(job)
            session.commit()

            audio_path = job.audio_path

            segments = session.exec(
                select(TranscriptSegment)
                .where(TranscriptSegment.job_id == job_id)
                .where(TranscriptSegment.text.is_not(None))
                .where(TranscriptSegment.text != "")
                .where(TranscriptSegment.alignment_status != "completed")
                .order_by(TranscriptSegment.segment_index)
            ).all()

            logger.info(
                "Alignment job=%s pending_segments=%d batch_size=%d",
                job_id,
                len(segments),
                ALIGN_BATCH_SIZE,
            )

            failed_count = 0
            completed_count = 0

            with sf.SoundFile(str(audio_path)) as source:
                sample_rate = source.samplerate

                for batch_number, batch in enumerate(
                    batched(segments, ALIGN_BATCH_SIZE),
                    start=1,
                ):
                    # Require the language actually stored by ASR.
                    # Missing language is recorded as an alignment failure.
                    valid = [
                        segment for segment in batch
                        if segment.language
                    ]
                    missing_language = [
                        segment for segment in batch
                        if not segment.language
                    ]

                    for segment in missing_language:
                        segment.alignment_status = "failed"
                        failed_count += 1

                    if missing_language:
                        session.add_all(missing_language)
                        session.commit()

                    if not valid:
                        continue

                    batch_started = time.perf_counter()

                    try:
                        audio_batch = self._read_audio_batch(
                            source, valid, sample_rate
                        )

                        texts = [segment.text for segment in valid]
                        languages = [segment.language for segment in valid]

                        results = self.model.align(
                            audio=audio_batch,
                            text=texts,
                            language=languages,
                        )

                        if len(results) != len(valid):
                            raise RuntimeError(
                                f"Aligner returned {len(results)} results "
                                f"for {len(valid)} segments"
                            )

                        for segment, result in zip(valid, results):
                            # Replace previous word rows if retrying.
                            session.exec(
                                delete(TranscriptWord).where(
                                    TranscriptWord.segment_id == segment.id
                                )
                            )

                            items = list(result.items)

                            for word_index, item in enumerate(items):
                                start = (
                                    segment.start_seconds
                                    + float(item.start_time)
                                )
                                end = (
                                    segment.start_seconds
                                    + float(item.end_time)
                                )

                                # Guard against invalid timestamps.
                                start = max(
                                    segment.start_seconds, start
                                )
                                end = max(start, end)

                                session.add(
                                    TranscriptWord(
                                        job_id=job_id,
                                        segment_id=segment.id,
                                        word_index=word_index,
                                        text=str(item.text),
                                        start_seconds=start,
                                        end_seconds=end,
                                    )
                                )

                            segment.alignment_status = "completed"
                            session.add(segment)
                            completed_count += 1

                        job.updated_at = utc_now()
                        session.add(job)
                        session.commit()

                        logger.info(
                            "Alignment progress job=%s batch=%d "
                            "completed=%d failed=%d elapsed=%.2fs",
                            job_id,
                            batch_number,
                            completed_count,
                            failed_count,
                            time.perf_counter() - batch_started,
                        )

                    except Exception as exc:
                        session.rollback()
                        logger.exception(
                            "Alignment batch failed job=%s batch=%d",
                            job_id,
                            batch_number,
                        )

                        # Retry each item individually to isolate problematic
                        # segments/languages instead of losing the whole batch.
                        for segment in valid:
                            try:
                                audio = self._read_audio_batch(
                                    source, [segment], sample_rate
                                )

                                result = self.model.align(
                                    audio=audio,
                                    text=[segment.text],
                                    language=[segment.language],
                                )[0]

                                session.exec(
                                    delete(TranscriptWord).where(
                                        TranscriptWord.segment_id == segment.id
                                    )
                                )

                                for word_index, item in enumerate(result.items):
                                    start = max(
                                        segment.start_seconds,
                                        segment.start_seconds
                                        + float(item.start_time),
                                    )
                                    end = max(
                                        start,
                                        segment.start_seconds
                                        + float(item.end_time),
                                    )

                                    session.add(
                                        TranscriptWord(
                                            job_id=job_id,
                                            segment_id=segment.id,
                                            word_index=word_index,
                                            text=str(item.text),
                                            start_seconds=start,
                                            end_seconds=end,
                                        )
                                    )

                                segment.alignment_status = "completed"
                                session.add(segment)
                                session.commit()
                                completed_count += 1

                            except Exception as item_exc:
                                session.rollback()

                                segment = session.get(
                                    TranscriptSegment, segment.id
                                )
                                if segment is not None:
                                    segment.alignment_status = "failed"
                                    session.add(segment)
                                    session.commit()

                                failed_count += 1
                                logger.warning(
                                    "Alignment failed job=%s segment=%s "
                                    "language=%r error=%s",
                                    job_id,
                                    segment.segment_index,
                                    segment.language,
                                    str(item_exc)[:1000],
                                )

            # Mark pipeline complete; failed segments remain identifiable
            # and can be retried by republishing this job to align_queue.
            job = session.exec(
                select(Job).where(Job.id == job_id)
            ).first()

            job.status = JobStatus.COMPLETED
            job.stage = JobStage.COMPLETED
            job.updated_at = utc_now()
            job.completed_at = utc_now()

            if failed_count:
                job.error = (
                    f"Alignment finished with {failed_count} failed "
                    "segments; inspect transcript_segments.alignment_status"
                )

            session.add(job)
            session.commit()

        elapsed = time.perf_counter() - started

        logger.info(
            "Alignment finished job=%s completed=%d failed=%d elapsed=%.2fs",
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


app = AlignConsumer.bind()
