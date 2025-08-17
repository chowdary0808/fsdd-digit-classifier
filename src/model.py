import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_model(kind="logreg"):
    if kind == "logreg":
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto")
    elif kind == "linsvm":
        clf = LinearSVC()
    else:
        raise ValueError("Unsupported model kind")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    return pipe

def save_model(pipe, path):
    joblib.dump(pipe, path)

def load_model(path):
    return joblib.load(path)
