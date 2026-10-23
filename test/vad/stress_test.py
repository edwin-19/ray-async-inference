
import os
import json
import time
import uuid
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from celery import Celery


# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "mysecurepassword")

BROKER_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
BACKEND_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/1"

QUEUE_NAME = "vad_queue"


client = Celery(
    "vad_stress_client",
    broker=BROKER_URL,
    backend=BACKEND_URL,
)


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = min(int((p / 100) * len(values)), len(values) - 1)
    return values[index]


def submit_job(audio_path, index):
    job_id = f"vad-stress-{index}-{uuid.uuid4().hex[:8]}"
    submitted_at = time.perf_counter()

    task = client.send_task(
        "process",
        args=[{
            "job_id": job_id,
            "audio_path": audio_path,
        }],
        queue=QUEUE_NAME,
    )

    return {
        "index": index,
        "job_id": job_id,
        "task_id": task.id,
        "submitted_at": submitted_at,
        "task": task,
    }


def main():
    parser = argparse.ArgumentParser(description="Stress-test Ray Serve FireRedVAD")
    parser.add_argument(
        "--audio",
        default="/mnt/data/workspace/freelance/voice/diar/vad_dataset/combined/combined_2h.wav",
        help="Absolute path to a WAV file accessible by the VAD workers",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=10,
        help="Total number of jobs to submit (default: 100)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent client-side submissions/waiters (default: 20)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per task in seconds (default: 600)",
    )
    args = parser.parse_args()

    if args.jobs < 1 or args.concurrency < 1:
        parser.error("--jobs and --concurrency must be >= 1")

    if not os.path.isfile(args.audio):
        raise FileNotFoundError(f"Client cannot find audio file: {args.audio}")

    print("=" * 72)
    print("FireRedVAD stress test")
    print("=" * 72)
    print(f"Queue:       {QUEUE_NAME}")
    print(f"Audio:       {args.audio}")
    print(f"Jobs:        {args.jobs}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Timeout:     {args.timeout}s")
    print()

    # Submit all jobs concurrently. Workers process them independently.
    start = time.perf_counter()
    submitted = []
    submission_errors = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(submit_job, args.audio, i): i
            for i in range(args.jobs)
        }

        for future in as_completed(futures):
            try:
                submitted.append(future.result())
            except Exception as exc:
                submission_errors.append({
                    "index": futures[future],
                    "error": repr(exc),
                })

    submit_finished = time.perf_counter()
    print(
        f"Submitted {len(submitted)}/{args.jobs} jobs "
        f"in {submit_finished - start:.2f}s"
    )

    successful = []
    failed = []
    completed_latencies = []

    def wait_for_result(job):
        wait_started = time.perf_counter()
        output = job["task"].get(
            timeout=args.timeout,
            propagate=True,
        )
        finished = time.perf_counter()
        return output, finished, wait_started

    # Wait for submitted tasks concurrently to avoid serial result polling.
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        future_to_job = {
            pool.submit(wait_for_result, job): job
            for job in submitted
        }

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                output, finished, _ = future.result()
                latency = finished - job["submitted_at"]

                if not isinstance(output, dict):
                    raise ValueError(f"Unexpected task result: {output!r}")

                successful.append({
                    "job_id": job["job_id"],
                    "task_id": job["task_id"],
                    "latency_seconds": latency,
                    "output": output,
                })
                completed_latencies.append(latency)

                print(
                    f"[OK]   {job['job_id']} | "
                    f"latency={latency:.2f}s | "
                    f"audio={output.get('duration_seconds')}s | "
                    f"segments={output.get('num_segments')}"
                )
            except Exception as exc:
                failed.append({
                    "job_id": job["job_id"],
                    "task_id": job["task_id"],
                    "error": repr(exc),
                })
                print(f"[FAIL] {job['job_id']} | {exc}")

    total_elapsed = time.perf_counter() - start

    audio_seconds = sum(
        float(item["output"].get("duration_seconds", 0) or 0)
        for item in successful
    )
    inference_seconds = sum(
        float(item["output"].get("inference_seconds", 0) or 0)
        for item in successful
    )
    segments = sum(
        int(item["output"].get("num_segments", 0) or 0)
        for item in successful
    )

    summary = {
        "queue": QUEUE_NAME,
        "requested_jobs": args.jobs,
        "submitted_jobs": len(submitted),
        "submission_failures": len(submission_errors),
        "successful_jobs": len(successful),
        "failed_jobs": len(failed),
        "total_wall_seconds": round(total_elapsed, 3),
        "jobs_per_second": round(
            len(successful) / total_elapsed, 3
        ) if total_elapsed > 0 else None,
        "audio_seconds_processed": round(audio_seconds, 3),
        "aggregate_audio_seconds_per_wall_second": round(
            audio_seconds / total_elapsed, 3
        ) if total_elapsed > 0 else None,
        "sum_worker_inference_seconds": round(inference_seconds, 3),
        "aggregate_rtf": round(
            inference_seconds / audio_seconds, 6
        ) if audio_seconds > 0 else None,
        "total_segments": segments,
        "latency_seconds": {
            "min": round(min(completed_latencies), 3)
            if completed_latencies else None,
            "median": round(statistics.median(completed_latencies), 3)
            if completed_latencies else None,
            "p95": round(percentile(completed_latencies, 95), 3)
            if completed_latencies else None,
            "max": round(max(completed_latencies), 3)
            if completed_latencies else None,
        },
    }

    print("\n" + "=" * 72)
    print("STRESS TEST SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))

    report = {
        "summary": summary,
        "submission_errors": submission_errors,
        "failed_jobs": failed,
        "successful_jobs": [
            {
                "job_id": item["job_id"],
                "task_id": item["task_id"],
                "latency_seconds": round(item["latency_seconds"], 3),
                "result": item["output"],
            }
            for item in successful
        ],
    }

    report_path = f"vad_stress_report_{int(time.time())}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
