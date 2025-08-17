import soundfile as sf
import numpy as np
from .features import waveform_to_features, resample_to_target, pad_trim, TARGET_SR
from .model import load_model

def predict_wav(path, model_path="models/model.joblib", feature_method="mfcc"):
    y, sr = sf.read(path)
    if y.ndim > 1:  # stereo -> mono
        y = np.mean(y, axis=1)
    y, sr = resample_to_target(y, sr, TARGET_SR)
    y = pad_trim(y, sr)
    feat = waveform_to_features(y, sr, feature_method)
    model = load_model(model_path)
    pred = model.predict([feat])[0]
    return int(pred)
