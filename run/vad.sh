
export RABBITMQ_HOST=127.0.0.1
export RABBITMQ_PORT=5672
export RABBITMQ_USER=appuser
export RABBITMQ_PASSWORD='replace_with_another_strong_random_password'
export RABBITMQ_VHOST=/

export RABBITMQ_MANAGEMENT_URL="http://127.0.0.1:15672"

export REDIS_HOST=127.0.0.1
export REDIS_PORT=6379
export REDIS_PASSWORD='your_actual_redis_password'

export QUEUE_NAME=vad_queue

serve run task.vad:app
