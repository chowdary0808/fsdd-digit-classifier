import numpy as np
import librosa

TARGET_SR = 8000
MAX_SEC = 1.0  # FSDD clips are short; pad/trim to 1s for consistency

def pad_trim(y, sr, max_sec=MAX_SEC):
    target_len = int(max_sec * sr)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y

def resample_to_target(y, sr, target_sr=TARGET_SR):
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr

def extract_mfcc(y, sr, n_mfcc=20):
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.hstack([m.mean(axis=1), m.std(axis=1)])

def extract_logmel(y, sr, n_mels=64):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logS = librosa.power_to_db(S + 1e-9)
    return np.hstack([logS.mean(axis=1), logS.std(axis=1)])

def waveform_to_features(y, sr, method="mfcc"):
    y, sr = resample_to_target(y, sr, TARGET_SR)
    y = pad_trim(y, sr, MAX_SEC)
    if method == "mfcc":
        return extract_mfcc(y, sr)
    elif method == "logmel":
        return extract_logmel(y, sr)
    else:
        raise ValueError("Unknown feature method")
