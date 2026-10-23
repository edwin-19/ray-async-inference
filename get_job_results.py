import time
from uuid import UUID

from dotenv import load_dotenv
from sqlmodel import Session, select

load_dotenv()

from db.db import engine
from db.models import Job, TranscriptSegment, TranscriptWord

JOB_ID = "cd107f1b-177a-45a4-af1c-ddcb97a4ef44"
POLL_SECONDS = 5
TIMEOUT_SECONDS = 6000


def enum_value(value):
    return value.value if hasattr(value, "value") else value


def print_results():
    job_id = UUID(JOB_ID)
    deadline = time.monotonic() + TIMEOUT_SECONDS

    while True:
        with Session(engine) as session:
            job = session.get(Job, job_id)

            if job is None:
                print(f"Job not found: {job_id}")
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
                # Copy the ORM values before closing the session.
                job_info = {
                    "id": job.id,
                    "audio_path": job.audio_path,
                    "status": status,
                    "stage": stage,
                    "duration_seconds": job.duration_seconds,
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
                "Check Ray Serve logs for main-consumer, vad-consumer, "
                "asr-consumer, and align-consumer."
            )
            return

        time.sleep(POLL_SECONDS)

    print("\n\n=== JOB ===")
    print(f"Job ID:       {job_info['id']}")
    print(f"Audio:        {job_info['audio_path']}")
    print(f"Status:       {job_info['status']}")
    print(f"Stage:        {job_info['stage']}")
    print(f"Duration:     {job_info['duration_seconds']} seconds")
    print(f"Error:        {job_info['error']}")
    print(f"Created:      {job_info['created_at']}")
    print(f"Started:      {job_info['started_at']}")
    print(f"Completed:    {job_info['completed_at']}")
    print(f"Updated:      {job_info['updated_at']}")

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
        print(f"Audio duration:         {job_info['duration_seconds']:.2f}s")

    print("\n=== COMBINED TRANSCRIPT ===")
    print(" ".join(transcript_parts) or "(No transcript available)")

    print(f"\n=== WORD-LEVEL ALIGNMENT ({len(word_rows)} words) ===")

    words_by_segment = {}
    for word in word_rows:
        words_by_segment.setdefault(word["segment_id"], []).append(word)

    for segment in segment_rows:
        segment_id = segment["id"]
        segment_words = words_by_segment.get(segment_id, [])

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
    print(f"Aligned segments:   {aligned_segments}/{len(segment_rows)}")
    print(f"Failed segments:    {failed_segments}")
    print(f"Words with timings: {len(word_rows)}")


if __name__ == "__main__":
    print(f"Checking job {JOB_ID} ...")
    print_results()