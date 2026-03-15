import os
import numpy as np
from datasets import load_dataset, Audio
import soundfile as sf
import io


DATASET_NAME = "razahtet/crema-d-audio"
SPLIT = "train"                               # or "test", "validation", etc.
SILENCE_THRESHOLD_DB = -50                    # files quieter than this are "silent"
MIN_DURATION_SEC = 0.1                        # files shorter than this are dropped
OUTPUT_DIR = "cleaned_dataset"                # where to save the cleaned dataset


def is_silent(audio_array: np.ndarray, sr: int, threshold_db: float = SILENCE_THRESHOLD_DB) -> bool:
    """Check if an audio signal is effectively silent."""
    if len(audio_array) == 0:
        return True
    # compute RMS energy in dB
    rms = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
    if rms == 0:
        return True
    db = 20 * np.log10(rms)
    return db < threshold_db


def is_too_short(audio_array: np.ndarray, sr: int, min_dur: float = MIN_DURATION_SEC) -> bool:
    """Check if an audio clip is shorter than the minimum duration."""
    duration = len(audio_array) / sr
    return duration < min_dur


def load_and_clean_dataset():
    """Load the HF dataset, filter out silent and corrupted files."""

    print(f"Loading dataset: {DATASET_NAME} (split={SPLIT})...")
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    # If the dataset has an 'audio' column, cast it so HF decodes it
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    total = len(ds)
    kept = []
    removed_silent = 0
    removed_corrupt = 0
    removed_short = 0

    print(f"Processing {total} samples...")

    for i, sample in enumerate(ds):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{total}...")

        try:
            # HF datasets decode audio into {"array": np.array, "sampling_rate": int}
            audio_data = sample["audio"]
            audio_array = np.array(audio_data["array"])
            sr = audio_data["sampling_rate"]

            # Check for corrupted (empty or invalid)
            if audio_array is None or len(audio_array) == 0:
                removed_corrupt += 1
                continue

            # Check for silence
            if is_silent(audio_array, sr):
                removed_silent += 1
                continue

            # Check for very short clips
            if is_too_short(audio_array, sr):
                removed_short += 1
                continue

            # Passed all checks — keep it
            kept.append(i)

        except Exception as e:
            # Any decoding error = corrupted
            print(f"  [CORRUPT] Sample {i}: {e}")
            removed_corrupt += 1
            continue

    # Filter the dataset to only the good indices
    cleaned_ds = ds.select(kept)

    print("\n=== Cleaning Summary ===")
    print(f"Total samples:    {total}")
    print(f"Kept:             {len(kept)}")
    print(f"Removed (silent): {removed_silent}")
    print(f"Removed (corrupt):{removed_corrupt}")
    print(f"Removed (short):  {removed_short}")

    # Save to disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cleaned_ds.save_to_disk(OUTPUT_DIR)
    print(f"\nCleaned dataset saved to: {OUTPUT_DIR}/")

    return cleaned_ds


if __name__ == "__main__":
    cleaned = load_and_clean_dataset()