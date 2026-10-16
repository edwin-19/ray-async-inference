import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from datasets import load_dataset

ROOT = Path("vad_dataset")
AUDIO_DIR = ROOT / "audio"
ANNOTATION_DIR = ROOT / "annotations"
MANIFEST_DIR = ROOT / "manifests"
COMBINED_DIR = ROOT / "combined"

TARGET_SECONDS = 2 * 60 * 60
SAMPLE_RATE = 16000
MERGE_GAP_SECONDS = 0.30

EXISTING_AUDIO = Path("long_audio.wav")
REUSE_VIDEO_ID = "GDKWU44V8Ys"

for directory in (
    AUDIO_DIR, ANNOTATION_DIR, MANIFEST_DIR, COMBINED_DIR
):
    directory.mkdir(parents=True, exist_ok=True)


def run(cmd):
    subprocess.run(cmd, check=True)


def duration_seconds(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def video_id(url):
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host == "youtu.be":
        return parsed.path.strip("/").split("/")[0]

    if host.endswith("youtube.com"):
        vid = parse_qs(parsed.query).get("v", [None])[0]
        if vid:
            return vid

        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
            return parts[1]

    return hashlib.sha256(url.encode()).hexdigest()[:16]


def download_audio(url, output_path):
    deno = Path.home() / ".deno" / "bin" / "deno"

    run([
        "uv", "run", "yt-dlp",
        "--js-runtimes", f"deno:{deno}",
        "--remote-components", "ejs:npm",
        "-f", "bestaudio",
        "-x", "--audio-format", "wav",
        "-o", str(output_path.with_suffix("")),
        url,
    ])


def normalize_audio(source, destination):
    run([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(source),
        "-vn", "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(destination),
    ])


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def estimate_speech_intervals(words, duration):
    """Merge word timestamps into approximate speech regions."""
    valid = []

    for word in words:
        try:
            start = float(word["start_timestamp"])
            end = float(word["end_timestamp"])
        except (KeyError, TypeError, ValueError):
            continue

        if end <= start or end <= 0 or start >= duration:
            continue

        start = max(0.0, start)
        end = min(duration, end)

        if end > start:
            valid.append([start, end])

    valid.sort(key=lambda x: x[0])

    if not valid:
        return []

    merged = [valid[0]]

    for start, end in valid[1:]:
        previous = merged[-1]

        if start - previous[1] <= MERGE_GAP_SECONDS:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])

    return [
        {
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
        }
        for start, end in merged
    ]


def main():
    ds = load_dataset(
        "oreva/blab_long_audio",
        "word_localization",
        split="word_localization",
        streaming=True,
        trust_remote_code=True,
    )

    records = []
    seen_urls = set()
    total_seconds = 0.0

    for sample in ds:
        if total_seconds >= TARGET_SECONDS:
            break

        url = sample.get("video_url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        uid = video_id(url)
        sample_id = f"blab_{len(records) + 1:04d}"

        raw_path = AUDIO_DIR / f"{sample_id}_raw.wav"
        audio_path = AUDIO_DIR / f"{sample_id}.wav"

        try:
            if uid == REUSE_VIDEO_ID and EXISTING_AUDIO.exists():
                raw_path = EXISTING_AUDIO
            else:
                print(f"Downloading {sample_id}: {url}")
                download_audio(url, raw_path)

            if not raw_path.exists():
                print("Missing audio; skipping.")
                continue

            normalize_audio(raw_path, audio_path)
            duration = duration_seconds(audio_path)

            if duration < 60:
                print("Audio too short; skipping.")
                audio_path.unlink(missing_ok=True)
                continue

            # BLAB word-level annotations.
            groundtruth = sample.get("groundtruth") or []
            if isinstance(groundtruth, str):
                groundtruth = json.loads(groundtruth)

            words = [
                item for item in groundtruth
                if isinstance(item, dict)
                and "start_timestamp" in item
                and "end_timestamp" in item
            ]

            speech_intervals = estimate_speech_intervals(
                words, duration
            )

            annotation = {
                "sample_id": sample_id,
                "source_url": url,
                "audio_duration_seconds": duration,
                "annotation_type": "estimated_vad_from_word_timestamps",
                "merge_gap_seconds": MERGE_GAP_SECONDS,
                "speech_intervals": speech_intervals,
                "word_timestamps": words,
                "limitations": (
                    "Estimated from word timestamps; not human-annotated "
                    "VAD ground truth. Pauses shorter than the merge "
                    "threshold are merged. Breaths, non-lexical speech, "
                    "and omitted words may not be represented."
                ),
            }

            annotation_path = (
                ANNOTATION_DIR / f"{sample_id}.json"
            )
            write_json(annotation_path, annotation)

            record = {
                "sample_id": sample_id,
                "source_dataset": "oreva/blab_long_audio",
                "source_config": "word_localization",
                "source_url": url,
                "audio_path": str(audio_path.relative_to(ROOT)),
                "annotation_path": str(
                    annotation_path.relative_to(ROOT)
                ),
                "duration_seconds": duration,
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "speech_interval_count": len(speech_intervals),
                "word_timestamp_count": len(words),
                "combined_start_seconds": total_seconds,
                "combined_end_seconds": total_seconds + duration,
            }

            records.append(record)
            total_seconds += duration

            write_jsonl(MANIFEST_DIR / "samples.jsonl", records)

            print(
                f"{sample_id}: {duration / 60:.1f} min, "
                f"{len(speech_intervals)} estimated speech regions; "
                f"total {total_seconds / 3600:.2f} h"
            )

        except (
            subprocess.CalledProcessError,
            OSError,
            ValueError,
            TypeError,
        ) as exc:
            print(f"Failed to process {url}: {exc}")
            continue

    if total_seconds < TARGET_SECONDS:
        raise RuntimeError(
            f"Collected only {total_seconds / 3600:.2f} hours. "
            "Fix download failures and rerun."
        )

    # Concatenate audio in manifest order.
    concat_path = COMBINED_DIR / "concat.txt"
    concat_path.write_text(
        "".join(
            f"file '{(ROOT / r['audio_path']).resolve()}'\n"
            for r in records
        ),
        encoding="utf-8",
    )

    combined_path = COMBINED_DIR / "combined_2h.wav"

    run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_path),
        "-c:a", "pcm_s16le",
        str(combined_path),
    ])

    # Shift intervals onto the combined audio timeline.
    combined_intervals = []

    for record in records:
        annotation = json.loads(
            (ROOT / record["annotation_path"]).read_text(
                encoding="utf-8"
            )
        )

        offset = record["combined_start_seconds"]

        for interval in annotation["speech_intervals"]:
            combined_intervals.append({
                "sample_id": record["sample_id"],
                "start": round(offset + interval["start"], 3),
                "end": round(offset + interval["end"], 3),
            })

    combined_intervals.sort(key=lambda x: x["start"])

    write_json(
        ANNOTATION_DIR / "combined_vad_timestamps.json",
        {
            "audio_path": str(combined_path.relative_to(ROOT)),
            "timestamp_unit": "seconds",
            "annotation_type": "estimated_vad_from_word_timestamps",
            "merge_gap_seconds": MERGE_GAP_SECONDS,
            "speech_intervals": combined_intervals,
        },
    )

    write_json(
        MANIFEST_DIR / "dataset_summary.json",
        {
            "dataset_name": "blab_long_audio_vad_estimated",
            "source_dataset": "oreva/blab_long_audio",
            "sample_count": len(records),
            "duration_seconds": total_seconds,
            "duration_hours": total_seconds / 3600,
            "combined_duration_seconds": duration_seconds(combined_path),
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "merge_gap_seconds": MERGE_GAP_SECONDS,
            "manifest": "manifests/samples.jsonl",
            "combined_audio": "combined/combined_2h.wav",
            "combined_vad_annotations": (
                "annotations/combined_vad_timestamps.json"
            ),
            "annotation_quality": "estimated, not human-verified",
        },
    )

    print("\nDataset complete.")
    print(f"Combined WAV: {combined_path}")
    print(
        "VAD timestamps:",
        ANNOTATION_DIR / "combined_vad_timestamps.json",
    )


if __name__ == "__main__":
    main()
