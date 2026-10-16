
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

from ray import serve
from ray.serve.config import AutoscalingConfig, AutoscalingPolicy
from ray.serve.schema import CeleryAdapterConfig, TaskProcessorConfig
from ray.serve.task_consumer import task_consumer, task_handler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("vad.consumer")


# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "mysecurepassword")

BROKER_URL = (
    f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
)
BACKEND_URL = (
    f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/1"
)

QUEUE_NAME = "vad_queue"

# Absolute defaults: independent of the current working directory.
BASE_DIR = Path(
    "/mnt/data/workspace/freelance/voice/queue_stt"
)

STATUS_DIR = Path(
    os.getenv("VAD_STATUS_DIR", str(BASE_DIR / "vad_status"))
).resolve()

RESULTS_DIR = Path(
    os.getenv("VAD_RESULTS_DIR", str(BASE_DIR / "vad_results"))
).resolve()


def write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON atomically to avoid readers seeing partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_name(
        f"{path.stem}.{os.getpid()}.tmp"
    )

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    temp_path.replace(path)


def update_status(task_id: str, status: str, **extra) -> None:
    """Persist the current task status."""
    status_path = STATUS_DIR / f"{task_id}.json"

    write_json_atomic(
        status_path,
        {
            "task_id": task_id,
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        },
    )


processor_config = TaskProcessorConfig(
    queue_name=QUEUE_NAME,
    adapter_config=CeleryAdapterConfig(
        broker_url=BROKER_URL,
        backend_url=BACKEND_URL,
    ),
)


@serve.deployment(
    max_ongoing_requests=5,
    autoscaling_config=AutoscalingConfig(
        min_replicas=1,
        max_replicas=10,
        target_ongoing_requests=2,
        policy=AutoscalingPolicy(
            policy_function=(
                "ray.serve.async_inference_autoscaling_policy:"
                "AsyncInferenceAutoscalingPolicy"
            ),
            policy_kwargs={
                "broker_url": BROKER_URL,
                "queue_name": QUEUE_NAME,
            },
        ),
    ),
)
@task_consumer(task_processor_config=processor_config)
class VADConsumer:

    def __init__(self):
        self.model = load_silero_vad()

        STATUS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Silero VAD model loaded")
        logger.info("Status directory: %s", STATUS_DIR)
        logger.info("Results directory: %s", RESULTS_DIR)

    @task_handler(name="detect")
    def detect(self, data):
        task_id = str(data["task_id"])
        audio_path = Path(data["audio_path"]).resolve()

        try:
            update_status(
                task_id,
                "processing",
                audio_path=str(audio_path),
            )

            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Audio file not found: {audio_path}"
                )

            audio, sample_rate = sf.read(
                str(audio_path),
                dtype="float32",
                always_2d=False,
            )

            # Convert stereo to mono.
            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            if sample_rate not in (8000, 16000):
                raise ValueError(
                    f"Unsupported sample rate: {sample_rate}. "
                    "Resample to 8000 or 16000 Hz first."
                )

            waveform = torch.as_tensor(
                audio,
                dtype=torch.float32,
            ).flatten()

            if not torch.isfinite(waveform).all():
                raise ValueError(
                    "Audio contains NaN or infinity"
                )

            if waveform.numel() == 0:
                segments = []
            else:
                timestamps = get_speech_timestamps(
                    waveform,
                    self.model,
                    sampling_rate=sample_rate,
                    return_seconds=True,
                )

                segments = [
                    {
                        "start": round(
                            float(segment["start"]), 3
                        ),
                        "end": round(
                            float(segment["end"]), 3
                        ),
                    }
                    for segment in timestamps
                ]

            output = {
                "task_id": task_id,
                "audio_path": str(audio_path),
                "sample_rate": int(sample_rate),
                "duration": round(
                    waveform.numel() / sample_rate, 3
                ),
                "segments": segments,
            }

            output_path = RESULTS_DIR / f"{task_id}.json"

            write_json_atomic(output_path, output)

            # Mark completed only after the result is written.
            update_status(
                task_id,
                "completed",
                result_path=str(output_path),
                segment_count=len(segments),
            )

            logger.info(
                "VAD completed: task=%s segments=%d result=%s",
                task_id,
                len(segments),
                output_path,
            )

            return None

        except Exception as exc:
            logger.exception("VAD failed: task=%s", task_id)

            try:
                update_status(
                    task_id,
                    "failed",
                    audio_path=str(audio_path),
                    error=str(exc),
                )
            except Exception:
                logger.exception(
                    "Could not update failure status: %s",
                    task_id,
                )

            raise


app = VADConsumer.bind()
