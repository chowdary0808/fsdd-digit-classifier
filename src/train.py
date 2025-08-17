from .data import load_fsdd
from .features import waveform_to_features
from .model import build_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
import os

def prepare_features(method="mfcc"):
    d = load_fsdd()
    X, y = [], []
    for arr, sr, label in tqdm(zip(d["audio"], d["sr"], d["label"]), total=len(d["label"])):
        feat = waveform_to_features(arr, sr, method=method)
        X.append(feat); y.append(label)
    return np.array(X), np.array(y)

def run_train(model_kind="logreg", feature_method="mfcc", out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    X, y = prepare_features(feature_method)

    # Stratified split (train/val/test = 70/15/15)
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val,  y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.1765, random_state=42, stratify=y_tmp)
    # (0.1765 of 0.85 ≈ 0.15 overall)

    model = build_model(model_kind)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("Validation accuracy:", round(val_acc, 4))
    print("Test accuracy:", round(test_acc, 4))
    print("\nClassification report (test):\n", classification_report(y_test, test_pred, digits=4))

    save_model(model, os.path.join(out_dir, "model.joblib"))
    return val_acc, test_acc

if __name__ == "__main__":
    run_train()
