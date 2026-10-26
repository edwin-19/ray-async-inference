
import argparse
import logging
import mimetypes
import os
import time
from pathlib import Path
from urllib.parse import quote
from uuid import UUID, uuid4

import boto3
from celery import Celery
from dotenv import load_dotenv
from sqlmodel import Session, select

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("test_pipeline")


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as a human-readable duration."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes}m {secs}s"


VAD_QUEUE = os.getenv("VAD_QUEUE", "vad_queue")
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "5"))
TIMEOUT_SECONDS = float(os.getenv("TIMEOUT_SECONDS", "6000"))

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "speech-data")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY") or os.getenv(
    "MINIO_ROOT_USER", ""
)
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY") or os.getenv(
    "MINIO_ROOT_PASSWORD", ""
)
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() in {
    "1", "true", "yes"
}


def redis_url(db: int) -> str:
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD", "")

    if password:
        return f"redis://:{quote(password, safe='')}@{host}:{port}/{db}"

    return f"redis://{host}:{port}/{db}"


BROKER_URL = redis_url(0)
BACKEND_URL = redis_url(1)


def create_minio_client():
    if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        raise RuntimeError("MinIO credentials are missing.")

    scheme = "https" if MINIO_SECURE else "http"

    return boto3.client(
        "s3",
        endpoint_url=f"{scheme}://{MINIO_ENDPOINT}",
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )


def upload_audio(audio_path: str, batch_id: UUID) -> str:
    """Upload the input audio once for all jobs in this batch."""
    path = Path(audio_path)
    object_key = f"audio/batches/{batch_id}/source{path.suffix or '.audio'}"

    content_type, _ = mimetypes.guess_type(str(path))
    extra_args = {"ContentType": content_type} if content_type else {}

    client = create_minio_client()

    logger.info(
        "Uploading audio: bucket=%s key=%s",
        MINIO_BUCKET,
        object_key,
    )

    client.upload_file(
        str(path),
        MINIO_BUCKET,
        object_key,
        ExtraArgs=extra_args or None,
    )

    logger.info("Upload complete: %s", object_key)
    return object_key


def create_jobs(job_ids: list[UUID], audio_object_key: str):
    from db.db import engine
    from db.models import Job, JobStatus, JobStage, utc_now

    now = utc_now()

    with Session(engine) as session:
        for job_id in job_ids:
            session.add(
                Job(
                    id=job_id,
                    audio_path=None,
                    audio_object_key=audio_object_key,
                    status=JobStatus.QUEUED,
                    stage=JobStage.VAD,
                    created_at=now,
                    updated_at=now,
                )
            )

        session.commit()

    logger.info("Created %d database jobs", len(job_ids))


def send_tasks(job_ids: list[UUID]) -> dict[UUID, str]:
    """Publish jobs to VAD and return successfully published task IDs."""
    celery_app = Celery(
        "speech_pipeline_test",
        broker=BROKER_URL,
        backend=BACKEND_URL,
    )
    published = {}

    try:
        for job_id in job_ids:
            try:
                result = celery_app.send_task(
                    "process",
                    args=[{"job_id": str(job_id)}],
                    queue=VAD_QUEUE,
                )
                published[job_id] = result.id
                logger.info(
                    "Published job=%s task=%s queue=%s",
                    job_id,
                    result.id,
                    VAD_QUEUE,
                )
            except Exception:
                logger.exception("Failed to publish job=%s", job_id)
    finally:
        celery_app.close()

    return published


def enum_value(value):
    return value.value if hasattr(value, "value") else value


def read_job_summary(job_ids: list[UUID]):
    from db.db import engine
    from db.models import Job, TranscriptSegment

    with Session(engine) as session:
        jobs = session.exec(
            select(Job).where(Job.id.in_(job_ids))
        ).all()

        segments = session.exec(
            select(TranscriptSegment).where(
                TranscriptSegment.job_id.in_(job_ids)
            )
        ).all()

        summary = {
            job.id: {
                "status": enum_value(job.status),
                "stage": enum_value(job.stage),
                "error": job.error,
                "duration_seconds": job.duration_seconds,
                "audio_object_key": getattr(
                    job, "audio_object_key", None
                ),
                "segments": 0,
                "transcribed": 0,
                "aligned": 0,
                "alignment_failed": 0,
            }
            for job in jobs
        }

        for segment in segments:
            row = summary.get(segment.job_id)
            if row is None:
                continue

            row["segments"] += 1

            if segment.text is not None:
                row["transcribed"] += 1

            alignment_status = getattr(
                segment, "alignment_status", "pending"
            )

            if alignment_status == "completed":
                row["aligned"] += 1
            elif alignment_status == "failed":
                row["alignment_failed"] += 1

        return summary


def wait_for_jobs(job_ids: list[UUID]):
    deadline = time.monotonic() + TIMEOUT_SECONDS
    terminal_statuses = {"completed", "failed"}
    last_progress = None

    while True:
        summary = read_job_summary(job_ids)

        completed = sum(
            row["status"] == "completed" for row in summary.values()
        )
        failed = sum(
            row["status"] == "failed" for row in summary.values()
        )
        terminal = completed + failed

        total_segments = sum(row["segments"] for row in summary.values())
        transcribed = sum(
            row["transcribed"] for row in summary.values()
        )
        aligned = sum(row["aligned"] for row in summary.values())
        alignment_failed = sum(
            row["alignment_failed"] for row in summary.values()
        )

        progress = (
            f"Jobs={terminal}/{len(job_ids)} "
            f"(completed={completed}, failed={failed}) | "
            f"Segments={total_segments} | "
            f"Transcribed={transcribed}/{total_segments} | "
            f"Aligned={aligned}/{total_segments} | "
            f"AlignmentFailed={alignment_failed}"
        )

        if progress != last_progress:
            print(f"\r{progress}", end="", flush=True)
            last_progress = progress

        if (
            len(summary) == len(job_ids)
            and all(
                row["status"] in terminal_statuses
                for row in summary.values()
            )
        ):
            print()
            return summary, True

        if time.monotonic() >= deadline:
            print("\nTimed out; some jobs may still be processing.")
            return summary, False

        time.sleep(POLL_SECONDS)


def print_summary(
    job_ids: list[UUID],
    summary: dict,
    published: dict[UUID, str],
    finished: bool,
):
    from db.db import engine
    from db.models import TranscriptWord
    from sqlalchemy import func

    with Session(engine) as session:
        rows = session.exec(
            select(
                TranscriptWord.job_id,
                func.count(TranscriptWord.id),
            )
            .where(TranscriptWord.job_id.in_(job_ids))
            .group_by(TranscriptWord.job_id)
        ).all()

        word_counts = {
            job_id: count for job_id, count in rows
        }

    print("\n=== PIPELINE SUMMARY ===")
    print(f"Jobs requested:       {len(job_ids)}")
    print(f"Tasks published:      {len(published)}")
    print(f"Jobs found in DB:     {len(summary)}")
    print(
        f"Completed:            "
        f"{sum(r['status'] == 'completed' for r in summary.values())}"
    )
    print(
        f"Failed:               "
        f"{sum(r['status'] == 'failed' for r in summary.values())}"
    )
    print(f"Timed out:            {not finished}")
    print(f"MinIO bucket:         {MINIO_BUCKET}")
    print(f"Queue:                {VAD_QUEUE}")

    total_segments = sum(r["segments"] for r in summary.values())
    total_transcribed = sum(r["transcribed"] for r in summary.values())
    total_aligned = sum(r["aligned"] for r in summary.values())
    total_alignment_failed = sum(
        r["alignment_failed"] for r in summary.values()
    )
    total_words = sum(word_counts.values())

    print(f"Total segments:       {total_segments}")
    print(
        f"Transcribed segments: "
        f"{total_transcribed}/{total_segments}"
    )
    print(f"Aligned segments:     {total_aligned}/{total_segments}")
    print(f"Alignment failures:   {total_alignment_failed}")
    print(f"Words with timings:   {total_words}")

    print("\n=== PER-JOB STATUS ===")
    print(
        f"{'JOB ID':<38} {'STATUS':<12} {'STAGE':<10} "
        f"{'SEGMENTS':>8} {'TRANSCRIBED':>12} "
        f"{'ALIGNED':>8} {'WORDS':>8}"
    )

    for job_id in job_ids:
        row = summary.get(job_id)

        if row is None:
            print(f"{str(job_id):<38} {'MISSING':<12}")
            continue

        print(
            f"{str(job_id):<38} "
            f"{str(row['status']):<12} "
            f"{str(row['stage']):<10} "
            f"{row['segments']:>8} "
            f"{row['transcribed']:>12} "
            f"{row['aligned']:>8} "
            f"{word_counts.get(job_id, 0):>8}"
        )

        if row["error"]:
            error = str(row["error"]).replace("\n", " ")
            print(f"  Error: {error[:300]}")

    if not finished:
        print(
            "\nSome jobs are still running. "
            "Check the consumer and Ray Serve logs."
        )


def main():
    global POLL_SECONDS, TIMEOUT_SECONDS

    parser = argparse.ArgumentParser(
        description=(
            "Submit one or more audio jobs to the "
            "MinIO/VAD pipeline."
        )
    )
    parser.add_argument(
        "audio_path",
        help="Path to the input audio file",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of jobs using the same audio (default: 1)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit jobs and exit without polling",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=POLL_SECONDS,
        help=f"Seconds between checks (default: {POLL_SECONDS})",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=TIMEOUT_SECONDS,
        help=f"Maximum wait time (default: {TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--audio-hours-per-job",
        type=float,
        default=2.0,
        help=(
            "Audio duration represented by each job, in hours "
            "(default: 2.0). Used for throughput reporting."
        ),
    )

    args = parser.parse_args()

    audio_path = str(
        Path(args.audio_path).expanduser().resolve()
    )

    if not Path(audio_path).is_file():
        parser.error(f"Audio file not found: {audio_path}")

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("Polling interval and timeout must be positive")

    if args.audio_hours_per_job <= 0:
        parser.error("--audio-hours-per-job must be positive")

    POLL_SECONDS = args.poll_seconds
    TIMEOUT_SECONDS = args.timeout_seconds

    # Start before upload to measure the entire end-to-end run.
    run_started = time.monotonic()

    batch_id = uuid4()
    job_ids = [uuid4() for _ in range(args.jobs)]

    try:
        audio_object_key = upload_audio(audio_path, batch_id)
    except Exception:
        logger.exception("Failed to upload audio to MinIO")
        raise SystemExit(1)

    try:
        create_jobs(job_ids, audio_object_key)
    except Exception:
        logger.exception("Failed to create database jobs")
        print(
            "The audio is uploaded, but database job creation failed. "
            f"MinIO object: {MINIO_BUCKET}/{audio_object_key}"
        )
        raise SystemExit(1)

    published = send_tasks(job_ids)

    print("\n=== SUBMISSION ===")
    print(f"Batch ID:       {batch_id}")
    print(f"Jobs requested: {len(job_ids)}")
    print(f"Tasks published:{len(published)}")
    print(f"MinIO object:   {MINIO_BUCKET}/{audio_object_key}")
    print(f"Queue:          {VAD_QUEUE}")

    if len(published) != len(job_ids):
        print(
            "Warning: some jobs were not published. "
            "Their database rows remain queued."
        )

    if args.no_wait:
        print("\nNot waiting for completion (--no-wait).")
        print("Job IDs:")
        for job_id in job_ids:
            print(job_id)

        elapsed = time.monotonic() - run_started
        print("\n=== TIMING SUMMARY ===")
        print(f"Submission elapsed time: {format_elapsed(elapsed)}")
        print(f"Submission elapsed seconds: {elapsed:.2f}s")
        return

    print("\nWaiting for job completion...")
    summary, finished = wait_for_jobs(job_ids)
    print_summary(job_ids, summary, published, finished)

    # Stop after collecting and printing the pipeline results.
    elapsed = time.monotonic() - run_started

    completed_jobs = sum(
        row["status"] == "completed"
        for row in summary.values()
    )
    completed_audio_hours = (
        completed_jobs * args.audio_hours_per_job
    )

    print("\n=== TIMING SUMMARY ===")
    print(f"Total elapsed time:   {format_elapsed(elapsed)}")
    print(f"Total elapsed seconds: {elapsed:.2f}s")
    print(f"Jobs completed:       {completed_jobs}/{len(job_ids)}")
    print(
        f"Audio hours completed: "
        f"{completed_audio_hours:.2f}"
    )

    if elapsed > 0 and completed_audio_hours > 0:
        throughput = completed_audio_hours / (elapsed / 3600)
        print(
            f"Effective throughput: {throughput:.2f} "
            "audio-hours per wall-clock hour"
        )

        if finished and completed_jobs == len(job_ids):
            print(
                "Batch processing rate: "
                f"{args.jobs / (elapsed / 3600):.2f} jobs/hour"
            )


if __name__ == "__main__":
    main()
