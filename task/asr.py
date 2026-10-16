
import logging
import os

from ray import serve
from ray.serve.config import AutoscalingConfig, AutoscalingPolicy
from ray.serve.schema import CeleryAdapterConfig, TaskProcessorConfig
from ray.serve.task_consumer import task_consumer, task_handler


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("asr.consumer")


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

QUEUE_NAME = "my_queue"


# Celery task consumer configuration
processor_config = TaskProcessorConfig(
    queue_name=QUEUE_NAME,
    adapter_config=CeleryAdapterConfig(
        broker_url=REDIS_BROKER_URL,
        backend_url=REDIS_BACKEND_URL,
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
                "broker_url": REDIS_BROKER_URL,
                "queue_name": QUEUE_NAME,
            },
        ),
    ),
)
@task_consumer(task_processor_config=processor_config)
class MyConsumer:
    @task_handler(name="process")
    def process(self, data):
        logger.info("Received task: %s", data)

        result = f"processed: {data}, test something new here"

        logger.info("Task completed: %s", result)
        print("Task completed: %s", result)
        return result


app = MyConsumer.bind()


# if __name__ == "__main__":
#     logger.info("Starting Ray Serve ASR task consumer...")
#     serve.run(app)
