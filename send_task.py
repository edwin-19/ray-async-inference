
import logging
import mimetypes
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote
from uuid import UUID, uuid4

import boto3
from celery import Celery
from dotenv import load_dotenv
from sqlmodel import Session, select

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_pipeline")

VAD_QUEUE = os.getenv("VAD_QUEUE", "vad_queue")
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "5"))
TIMEOUT_SECONDS = float(os.getenv("TIMEOUT_SECONDS", "6000"))

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


BROKER_URL = redis_url(0)
BACKEND_URL = redis_url(1)


def create_minio_client():
    if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        raise RuntimeError(
            "MinIO credentials are missing. Set MINIO_ACCESS_KEY and "
            "MINIO_SECRET_KEY, or configure MINIO_ROOT_USER and "
            "MINIO_ROOT_PASSWORD."
        )

    scheme = "https" if MINIO_SECURE else "http"

    return boto3.client(
        "s3",
        endpoint_url=f"{scheme}://{MINIO_ENDPOINT}",
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )


def upload_audio(audio_path: str, job_id: UUID) -> str:
    """Upload audio to MinIO and return its object key."""
    path = Path(audio_path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    suffix = path.suffix or ".audio"
    object_key = f"audio/{job_id}/source{suffix}"

    minio = create_minio_client()

    content_type, _ = mimetypes.guess_type(str(path))

    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type

    logger.info(
        "Uploading audio to MinIO bucket=%s key=%s",
        MINIO_BUCKET,
        object_key,
    )

    minio.upload_file(
        str(path),
        MINIO_BUCKET,
        object_key,
        ExtraArgs=extra_args or None,
    )

    logger.info("Audio uploaded successfully: %s", object_key)
    return object_key


def create_job(job_id: UUID, audio_object_key: str) -> str:
    from db.db import engine
    from db.models import Job, JobStatus, JobStage, utc_now

    now = utc_now()

    job = Job(
        id=job_id,
        audio_path=None,
        audio_object_key=audio_object_key,
        status=JobStatus.QUEUED,
        stage=JobStage.VAD,
        created_at=now,
        updated_at=now,
    )

    with Session(engine) as session:
        session.add(job)
        session.commit()

    logger.info(
        "Created job=%s audio_object_key=%s",
        job_id,
        audio_object_key,
    )
    return str(job_id)


def send_task(job_id: str) -> str:
    celery_app = Celery(
        "speech_pipeline_test",
        broker=BROKER_URL,
        backend=BACKEND_URL,
    )

    try:
        result = celery_app.send_task(
            "process",
            args=[{"job_id": job_id}],
            queue=VAD_QUEUE,
        )

        logger.info(
            "Published task_id=%s queue=%s job_id=%s",
            result.id,
            VAD_QUEUE,
            job_id,
        )
        return result.id
    finally:
        celery_app.close()


def enum_value(value):
    return value.value if hasattr(value, "value") else value


def print_results(job_id_str: str):
    from db.db import engine
    from db.models import Job, TranscriptSegment, TranscriptWord

    job_id = UUID(job_id_str)
    deadline = time.monotonic() + TIMEOUT_SECONDS

    while True:
        with Session(engine) as session:
            job = session.get(Job, job_id)

            if job is None:
                print(f"\nJob not found: {job_id}")
                return

            segments = session.exec(
                select(TranscriptSegment)
                .where(TranscriptSegment.job_id == job_id)
                .order_by(TranscriptSegment.segment_index)
            ).all()

            words = session.exec(
                select(TranscriptWord)
                .where(TranscriptWord.job_id == job_id)
                .order_by(
                    TranscriptWord.segment_id,
                    TranscriptWord.word_index,
                )
            ).all()

            status = enum_value(job.status)
            stage = enum_value(job.stage)

            transcribed = sum(
                segment.text is not None for segment in segments
            )

            aligned_segments = sum(
                getattr(segment, "alignment_status", None) == "completed"
                for segment in segments
            )

            failed_segments = sum(
                getattr(segment, "alignment_status", None) == "failed"
                for segment in segments
            )

            print(
                f"\rStatus={status} | Stage={stage} | "
                f"Segments={len(segments)} | "
                f"Transcribed={transcribed}/{len(segments)} | "
                f"Aligned={aligned_segments}/{len(segments)} | "
                f"Words={len(words)} | "
                f"AlignmentFailed={failed_segments}",
                end="",
                flush=True,
            )

            if status in ("completed", "failed"):
                job_info = {
                    "id": job.id,
                    "audio_path": getattr(job, "audio_path", None),
                    "audio_object_key": getattr(
                        job, "audio_object_key", None
                    ),
                    "status": status,
                    "stage": stage,
                    "duration_seconds": job.duration_seconds,
                    "transcript": job.transcript,
                    "language": job.language,
                    "error": job.error,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "completed_at": job.completed_at,
                    "updated_at": job.updated_at,
                }

                segment_rows = [
                    {
                        "id": segment.id,
                        "segment_index": segment.segment_index,
                        "start_seconds": segment.start_seconds,
                        "end_seconds": segment.end_seconds,
                        "text": segment.text,
                        "language": segment.language,
                        "alignment_status": getattr(
                            segment, "alignment_status", "unknown"
                        ),
                    }
                    for segment in segments
                ]

                word_rows = [
                    {
                        "segment_id": word.segment_id,
                        "word_index": word.word_index,
                        "text": word.text,
                        "start_seconds": word.start_seconds,
                        "end_seconds": word.end_seconds,
                    }
                    for word in words
                ]

                break

        if time.monotonic() >= deadline:
            print("\n\nTimed out waiting for job completion.")
            print("The job may still be processing.")
            print(
                "Check Ray Serve logs for vad-consumer, "
                "asr-consumer, and align-consumer."
            )
            return

        time.sleep(POLL_SECONDS)

    print("\n\n=== JOB ===")
    print(f"Job ID:        {job_info['id']}")
    print(f"Audio path:    {job_info['audio_path']}")
    print(f"MinIO key:     {job_info['audio_object_key']}")
    print(f"Status:        {job_info['status']}")
    print(f"Stage:         {job_info['stage']}")
    print(f"Duration:      {job_info['duration_seconds']} seconds")
    print(f"Language:      {job_info['language']}")
    print(f"Error:         {job_info['error']}")
    print(f"Created:       {job_info['created_at']}")
    print(f"Started:       {job_info['started_at']}")
    print(f"Completed:     {job_info['completed_at']}")
    print(f"Updated:       {job_info['updated_at']}")

    print(f"\n=== TRANSCRIPT SEGMENTS ({len(segment_rows)}) ===")

    transcript_parts = []

    for segment in segment_rows:
        text = segment["text"] or ""
        language = segment["language"] or "unknown"

        print(
            f"[{segment['segment_index']:04d}] "
            f"{segment['start_seconds']:.3f}s - "
            f"{segment['end_seconds']:.3f}s | "
            f"{language} | "
            f"Alignment={segment['alignment_status']} | "
            f"{text}"
        )

        if text.strip():
            transcript_parts.append(text.strip())

    if job_info["duration_seconds"] is not None:
        speech_seconds = sum(
            max(
                0.0,
                segment["end_seconds"] - segment["start_seconds"],
            )
            for segment in segment_rows
        )

        print(f"\nTotal segment duration: {speech_seconds:.2f}s")
        print(
            f"Audio duration:         "
            f"{job_info['duration_seconds']:.2f}s"
        )

    print("\n=== COMBINED TRANSCRIPT ===")
    print(" ".join(transcript_parts) or "(No transcript available)")

    print(f"\n=== WORD-LEVEL ALIGNMENT ({len(word_rows)} words) ===")

    words_by_segment = {}

    for word in word_rows:
        words_by_segment.setdefault(word["segment_id"], []).append(word)

    for segment in segment_rows:
        segment_words = words_by_segment.get(segment["id"], [])

        if not segment_words:
            continue

        print(
            f"\nSegment [{segment['segment_index']:04d}] "
            f"{segment['start_seconds']:.3f}s - "
            f"{segment['end_seconds']:.3f}s"
        )

        for word in segment_words:
            print(
                f"  [{word['word_index']:03d}] "
                f"{word['start_seconds']:.3f}s - "
                f"{word['end_seconds']:.3f}s | "
                f"{word['text']}"
            )

    print("\n=== ALIGNMENT SUMMARY ===")
    print(f"Total segments:     {len(segment_rows)}")
    print(f"Transcribed:        {transcribed}/{len(segment_rows)}")
    print(f"Aligned segments:   {aligned_segments}/{len(segment_rows)}")
    print(f"Failed segments:    {failed_segments}")
    print(f"Words with timings: {len(word_rows)}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <audio_path>")
        sys.exit(1)

    audio_path = str(Path(sys.argv[1]).expanduser().resolve())

    if not Path(audio_path).is_file():
        print(f"Error: audio file not found: {audio_path}")
        sys.exit(1)

    job_id = uuid4()
    audio_object_key = None

    # 1. Upload the source audio to MinIO.
    try:
        audio_object_key = upload_audio(audio_path, job_id)
    except Exception:
        logger.exception("Failed to upload audio to MinIO")
        sys.exit(1)

    # 2. Create the DB job referencing the uploaded object.
    try:
        create_job(job_id, audio_object_key)
    except Exception:
        logger.exception(
            "Failed to create database job. "
            "Uploaded MinIO object: bucket=%s key=%s",
            MINIO_BUCKET,
            audio_object_key,
        )
        print(
            "\nThe audio was uploaded, but job creation failed. "
            "The MinIO object may need to be removed manually."
        )
        sys.exit(1)

    # 3. Publish directly to vad_queue.
    try:
        task_id = send_task(str(job_id))
    except Exception:
        logger.exception(
            "Failed to publish job_id=%s. "
            "The database job and MinIO object remain available.",
            job_id,
        )
        print("\nTask publishing failed; the job remains in PostgreSQL.")
        print(f"Job ID: {job_id}")
        print(f"MinIO key: {audio_object_key}")
        sys.exit(1)

    print("\nJob submitted successfully")
    print(f"  Job ID:      {job_id}")
    print(f"  Local audio: {audio_path}")
    print(f"  MinIO bucket:{MINIO_BUCKET}")
    print(f"  MinIO key:   {audio_object_key}")
    print(f"  Task ID:     {task_id}")
    print(f"  Queue:       {VAD_QUEUE}")

    # 4. Poll PostgreSQL and print final results.
    print(
        f"\nWaiting for pipeline completion "
        f"(timeout={TIMEOUT_SECONDS}s)..."
    )
    print_results(str(job_id))


if __name__ == "__main__":
    main()
