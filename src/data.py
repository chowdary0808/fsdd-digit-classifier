# src/data.py
from typing import Dict
from pathlib import Path
import io, zipfile, requests
import numpy as np
import soundfile as sf

from datasets import load_dataset, Audio  # HF mirror of FSDD

# Fallback (original FSDD repo)
DATA_DIR = Path("data/fsdd")
RECORDINGS_DIR = DATA_DIR / "recordings"
ZIP_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
ZIP_PREFIX = "free-spoken-digit-dataset-master/recordings/"

def _ensure_fsdd_downloaded() -> None:
    if RECORDINGS_DIR.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading FSDD zip…")
    r = requests.get(ZIP_URL, stream=True, timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    for name in z.namelist():
        if name.startswith(ZIP_PREFIX) and name.endswith(".wav"):
            rel = name[len(ZIP_PREFIX):]
            dest = RECORDINGS_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as fsrc, open(dest, "wb") as fdst:
                fdst.write(fsrc.read())
    print(f"Extracted to: {RECORDINGS_DIR.resolve()}")

def _load_from_github() -> Dict[str, list]:
    _ensure_fsdd_downloaded()
    audio_list, sr_list, labels, speakers = [], [], [], []
    for wav_path in sorted(RECORDINGS_DIR.glob("*.wav")):
        parts = wav_path.stem.split("_")  # e.g., 7_jackson_32.wav
        label = int(parts[0])
        speaker = parts[1] if len(parts) > 1 else "unknown"
        y, sr = sf.read(str(wav_path))
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32)
        audio_list.append(y); sr_list.append(int(sr)); labels.append(label); speakers.append(speaker)
    return {"audio": audio_list, "sr": sr_list, "label": labels, "speaker": speakers}

def load_fsdd() -> Dict[str, list]:
    """
    Preferred: load via Hugging Face (as per challenge).
    Fallback: load the same WAVs from the official FSDD GitHub repo.
    """
    try:
        # Try Hugging Face mirror first (this may require 'torchcodec' on recent versions)
        ds = load_dataset("mteb/free-spoken-digit-dataset")
        d = ds["train"].cast_column("audio", Audio(decode=True))  # decode to numpy
        audio_list, sr_list, labels, speakers = [], [], [], []
        for ex in d:
            arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
            sr = int(ex["audio"]["sampling_rate"])
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            y = int(ex["label"])
            spk = str(ex.get("speaker_id", ex.get("speaker", "unknown")))
            audio_list.append(arr); sr_list.append(sr); labels.append(y); speakers.append(spk)
        return {"audio": audio_list, "sr": sr_list, "label": labels, "speaker": speakers}
    except Exception as e:
        print("Hugging Face audio decode failed, falling back to GitHub WAVs:", repr(e))
        return _load_from_github()
