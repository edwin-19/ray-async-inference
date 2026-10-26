# Long Async Inference with Ray and Redis Celery
- We use 3 stages
- VAD 
- ASR
- Word Alignment

# How to install
```
pip install -r requirements.txt
```

# Start Pipeline worker
```
serve run task.vad_consumer:app

serve run task.asr:app
serve run task.align_consumer:app
```

# Send task over
```bash
python send_task.py /path/audio_path.wav
```

# Send batch job
```bash
python batch_run.py /path/audio_path.wav --jobs 10
```
