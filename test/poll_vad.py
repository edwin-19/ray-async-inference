
import time

from vad_client import submit_vad, get_status, get_result


audio_path = (
    "/mnt/data/workspace/freelance/voice/diar/"
    "vad_dataset/audio/blab_0001.wav"
)

task_id = submit_vad(audio_path)

while True:
    status = get_status(task_id)
    state = status["status"]

    print("Task status:", state)

    if state == "completed":
        result = get_result(task_id)
        print("Audio duration:", result["duration"])
        print("Speech segments:", len(result["segments"]))
        print("First 5 segments:", result["segments"][:5])
        break

    if state == "failed":
        raise RuntimeError(status.get("error", "VAD failed"))

    if state == "unknown":
        raise RuntimeError(
            f"Status file not found: {status}"
        )

    time.sleep(2)
