import logging
import os
from pathlib import Path

import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from ray import serve
from ray.serve.config import AutoscalingConfig, AutoscalingPolicy
from ray.serve.schema import CeleryAdapterConfig, TaskProcessorConfig
from ray.serve.task_consumer import task_consumer, task_handler

from common.job_store import update_task


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("vad.consumer")


# Redis / Celery configuration
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
        logger.info("Silero VAD model loaded")

    @task_handler(name="detect")
    def detect(self, data):
        task_id = str(data["task_id"])

        # Existing clients can keep sending task_id only.
        # New pipeline clients can provide a separate job_id.
        job_id = str(data.get("job_id", task_id))
        audio_path = Path(data["audio_path"]).resolve()

        try:
            update_task(
                job_id,
                task_id,
                task_type="vad",
                status="processing",
                audio_path=str(audio_path),
                error=None,
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
                raise ValueError("Audio contains NaN or infinity")

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
                        "start": round(float(segment["start"]), 3),
                        "end": round(float(segment["end"]), 3),
                    }
                    for segment in timestamps
                ]

            output = {
                "audio_path": str(audio_path),
                "sample_rate": int(sample_rate),
                "duration": round(
                    waveform.numel() / sample_rate, 3
                ),
                "segments": segments,
            }

            # Persist the result directly in the job JSON.
            update_task(
                job_id,
                task_id,
                task_type="vad",
                status="completed",
                audio_path=str(audio_path),
                sample_rate=int(sample_rate),
                duration=output["duration"],
                segment_count=len(segments),
                result=output,
                error=None,
            )

            logger.info(
                "VAD completed: job=%s task=%s segments=%d",
                job_id,
                task_id,
                len(segments),
            )

            return None

        except Exception as exc:
            logger.exception(
                "VAD failed: job=%s task=%s",
                job_id,
                task_id,
            )

            try:
                update_task(
                    job_id,
                    task_id,
                    task_type="vad",
                    status="failed",
                    audio_path=str(audio_path),
                    error=str(exc),
                )
            except Exception:
                logger.exception(
                    "Could not persist VAD failure: task=%s",
                    task_id,
                )

            raise


app = VADConsumer.bind()