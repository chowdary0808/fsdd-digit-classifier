from .data import load_fsdd
from .features import waveform_to_features
from .model import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def evaluate(model_path="models/model.joblib", feature_method="mfcc"):
    d = load_fsdd()
    X, y = [], []
    for arr, sr, label in zip(d["audio"], d["sr"], d["label"]):
        X.append(waveform_to_features(arr, sr, feature_method)); y.append(label)
    X = np.array(X); y = np.array(y)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    model = load_model(model_path)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=range(10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(values_format='d')
    plt.title("Confusion Matrix - Test")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()
