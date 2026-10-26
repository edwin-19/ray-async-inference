
import logging
import os
import time
from pathlib import Path

from ray import serve
from ray.serve.config import AutoscalingConfig, AutoscalingPolicy
from ray.serve.schema import CeleryAdapterConfig, TaskProcessorConfig
from ray.serve.task_consumer import task_consumer, task_handler

from fireredvad import FireRedVad, FireRedVadConfig

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("vad.consumer")


# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "mysecurepassword")

REDIS_BROKER_URL = (
    f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
)
REDIS_BACKEND_URL = (
    f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/1"
)

VAD_QUEUE_NAME = "vad_queue"


# FireRedVAD configuration
VAD_MODEL_PATH = Path(
    os.getenv(
        "FIREDVAD_MODEL_PATH",
        "/mnt/data/workspace/freelance/voice/diar/"
        "pretrained_models/FireRedVAD/VAD",
    )
)

VAD_USE_GPU = os.getenv("VAD_USE_GPU", "true").lower() in (
    "1", "true", "yes"
)

VAD_NUM_GPUS_PER_REPLICA = float(
    os.getenv(
        "VAD_NUM_GPUS_PER_REPLICA",
        "1.0" if VAD_USE_GPU else "0",
    )
)

if VAD_USE_GPU and VAD_NUM_GPUS_PER_REPLICA <= 0:
    raise ValueError(
        "GPU inference requires VAD_NUM_GPUS_PER_REPLICA > 0"
    )


# Celery VAD task consumer configuration
vad_processor_config = TaskProcessorConfig(
    queue_name=VAD_QUEUE_NAME,
    adapter_config=CeleryAdapterConfig(
        broker_url=REDIS_BROKER_URL,
        backend_url=REDIS_BACKEND_URL,
    ),
)


@serve.deployment(
    name="vad-consumer",
    max_ongoing_requests=2,
    ray_actor_options={
        "num_gpus": 1,
        "num_cpus": 2,
    },
    autoscaling_config=AutoscalingConfig(
        min_replicas=1,
        max_replicas=2,
        target_ongoing_requests=2,
        policy=AutoscalingPolicy(
            policy_function=(
                "ray.serve.async_inference_autoscaling_policy:"
                "AsyncInferenceAutoscalingPolicy"
            ),
            policy_kwargs={
                "broker_url": REDIS_BROKER_URL,
                "queue_name": VAD_QUEUE_NAME,
            },
        ),
    ),
)
@task_consumer(task_processor_config=vad_processor_config)
class VadConsumer:

    def __init__(self):
        logger.info(
            "Loading FireRedVAD | model_path=%s | gpu=%s",
            VAD_MODEL_PATH,
            VAD_USE_GPU,
        )

        if not VAD_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"FireRedVAD model directory not found: {VAD_MODEL_PATH}"
            )

        vad_config = FireRedVadConfig(
            use_gpu=VAD_USE_GPU,
            smooth_window_size=5,
            speech_threshold=0.4,
            min_speech_frame=20,
            max_speech_frame=2000,
            min_silence_frame=20,
            merge_silence_frame=0,
            extend_speech_frame=0,
            chunk_max_frame=30000,
        )

        # Load the model once per replica.
        self.vad_model = FireRedVad.from_pretrained(
            str(VAD_MODEL_PATH),
            vad_config,
        )

        logger.info(
            "FireRedVAD loaded | gpu=%s | queue=%s",
            VAD_USE_GPU,
            VAD_QUEUE_NAME,
        )

    @task_handler(name="process")
    def process(self, data):
        """
        Expected task payload:
        {
            "job_id": "vad-001",
            "audio_path": "/path/to/audio.wav"
        }
        """
        if not isinstance(data, dict):
            raise TypeError(
                "VAD task payload must be a dict containing audio_path"
            )

        job_id = data.get("job_id", "unknown")
        audio_path_value = data.get("audio_path")

        if not audio_path_value:
            raise ValueError("Missing required field: audio_path")

        audio_path = Path(audio_path_value)

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"VAD audio file not found: {audio_path}"
            )

        logger.info(
            "Processing VAD task | job_id=%s | audio=%s",
            job_id,
            audio_path,
        )

        start = time.perf_counter()

        result, _ = self.vad_model.detect(str(audio_path))

        inference_seconds = time.perf_counter() - start
        duration_seconds = float(result["dur"])

        segments = [
            {
                "start": float(segment[0]),
                "end": float(segment[1]),
            }
            for segment in result["timestamps"]
        ]

        rtf = (
            inference_seconds / duration_seconds
            if duration_seconds > 0
            else None
        )

        realtime_speedup = (
            duration_seconds / inference_seconds
            if inference_seconds > 0
            else None
        )

        response = {
            "job_id": job_id,
            "status": "completed",
            "audio_path": str(audio_path),
            "duration_seconds": round(duration_seconds, 3),
            "inference_seconds": round(inference_seconds, 4),
            "rtf": round(rtf, 6) if rtf is not None else None,
            "realtime_speedup": (
                round(realtime_speedup, 2)
                if realtime_speedup is not None
                else None
            ),
            "num_segments": len(segments),
            "segments": segments,
            "worker_pid": os.getpid(),
        }

        logger.info(
            "VAD completed | job_id=%s | duration=%.2fs "
            "| inference=%.4fs | segments=%d | RTF=%s",
            job_id,
            duration_seconds,
            inference_seconds,
            len(segments),
            response["rtf"],
        )

        return response


app = VadConsumer.bind()
