# LLM Coding Challenge — Digit Classification from Audio (FSDD)

Goal: Build a fast, clean prototype that predicts spoken digits (0–9) from audio using lightweight features and a linear model, with a small Streamlit app.

## Approach
- Dataset: Free Spoken Digit Dataset (Hugging Face: `mteb/free-spoken-digit-dataset`)
- Features: MFCC (20 coeffs) aggregated by mean+std → 40-dim vector per clip
- Model: Logistic Regression (also try Linear SVM). Stratified train/val/test split.
- Latency: Instant (milliseconds) per prediction
- App: Streamlit (upload WAV or optional in-browser recording)

## Quickstart
# Create & activate virtual environment
python -m venv .venv
# Windows
.\.venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Train (downloads dataset automatically)
python -m src.train

# Evaluate (shows confusion matrix)
python -m src.evaluate

# Run demo app
streamlit run app.py


## Results
- Fill after you train:
  - Validation accuracy: 0.934
  - Test accuracy: 0.94- Confusion matrix is produced by --src/evaluate.py


# Summary
Task: classify a single spoken digit (0–9) from a ~1 s WAV clip.
Features: 20 MFCCs extracted with Librosa, aggregated by mean + std → 40-dim vector per clip.
Model: Scikit-learn Pipeline(StandardScaler → LogisticRegression). A LinearSVC variant is available.
Split: stratified train/validation/test ≈ 70/15/15, fixed random seed for stability.
Latency: millisecond-level inference on CPU.
UI: Streamlit page with WAV upload and optional microphone recorder.

# Dataset
Free Spoken Digit Dataset (FSDD) — WAV recordings of digits spoken by multiple English speakers at 8 kHz.
Primary source: Hugging Face dataset mteb/free-spoken-digit-dataset.
Portable fallback: automatic download of the official GitHub ZIP to data/fsdd/recordings/.
This dual path preserves portability across environments (e.g., Windows setups that lack torchcodec).

# Method
** Preprocessing
 Resample to 8 kHz.
 Pad/trim to 1.0 s for consistent frame lengths.
** Feature extraction (src/features.py)
 Compute MFCC (default n_mfcc=20).
 Aggregate across time by mean and std → 40-dim vector.
** Modeling (src/model.py)
 Logistic Regression with max_iter=2000, solver="lbfgs".
 Wrapped in a Pipeline with StandardScaler.
** Training (src/train.py)
 Stratified split into train/val/test.
 Fit on training set; report validation/test metrics; persist model to models/model.joblib.
** Evaluation (src/evaluate.py)
 Rebuild features from the dataset, load the persisted model, produce a confusion matrix.
** Inference (src/infer.py)
 Single-file prediction utility used by the Streamlit app.
** Results
# Validation accuracy: 0.931
# Test accuracy: 0.9407

Confusion matrix (test) indicates strong diagonals; most misclassifications appear among 1/5/6 and a notable block of 6→3. This pattern reflects the loss of temporal dynamics when using mean/std MFCCs. Digits such as 7/8/9/4 achieve the highest recalls (≥ 0.95 in this run).


# Repository Layout
.
├─ app.py                      # Streamlit application (upload + in-browser recording)
├─ requirements.txt
├─ models/
│  └─ model.joblib             # persisted sklearn pipeline (created by training)
├─ data/
│  └─ fsdd/recordings/         # auto-downloaded WAVs (fallback path)
├─ src/
│  ├─ __init__.py
│  ├─ data.py                  # dataset loader (Hugging Face → WAV fallback)
│  ├─ features.py              # resample → pad/trim → MFCC → mean/std
│  ├─ model.py                 # build/save/load pipeline (logreg / linear SVM)
│  ├─ train.py                 # feature prep, split, training, persistence
│  ├─ evaluate.py              # confusion matrix on test set
│  └─ infer.py                 # single-WAV prediction used by the app
├─ PROMPTS.md                  # LLM prompt log (key prompts + outcomes)
└─ README.md

# Reproducibility
The repository includes scripts for environment setup, training, evaluation, and the demo UI. Typical commands:
# virtual environment (Windows)
python -m venv .venv
.\.venv\Scripts\activate
# virtual environment (macOS/Linux)
python -m venv .venv
source .venv/bin/activate
# dependencies
pip install -r requirements.txt
# training (downloads data if needed; saves models/model.joblib)
python -m src.train
# evaluation (displays confusion matrix)
python -m src.evaluate


# Demo Application
The Streamlit app supports both WAV upload and live recording:
streamlit run app.py
Upload any FSDD clip (e.g., data/fsdd/recordings/7_jackson_32.wav).
Or record ~1 s of audio saying a digit; the app saves a temporary WAV and predicts.
The model used by the app is the persisted pipeline in models/model.joblib.


# Design Decisions & Trade-offs
MFCC mean+std keeps the feature vector tiny and robust, offering excellent latency and simplicity at the cost of temporal detail.
Linear models (LogReg/LinearSVM) are compact, interpretable, and fast; they pair well with these features.
Stratified splitting ensures balanced evaluation and stable metrics.
HF→WAV fallback provides reliability on platforms where installing audio decoders is fragile.

# Limitations & Next Steps
Temporal cues are largely discarded. Adding Δ/ΔΔ MFCCs or switching to log-mel frames typically reduces confusions like 6↔3 and boosts 1/5/6.
Augmentation (noise, time-shift, slight speed/pitch) improves microphone robustness.
Speaker-aware evaluation (leave-one-speaker-out) better measures generalization.
A tiny CNN over log-mels tends to improve accuracy while preserving low latency.



# LLM Collaboration
Development was accelerated with an LLM coding assistant.
PROMPTS.md captures representative prompts and outcomes, including:
Selecting MFCC mean+std + Logistic Regression as a minimal strong baseline.
Designing a portable dataset loader with an automatic WAV fallback.
Stabilizing metrics via stratified splits and a fixed random seed.
Producing a concise Streamlit interface for upload and recording.

# Live demo
**Streamlit app:** https://chowdary0808-fsdd-digit-classifier-app-2oesrv.streamlit.app/
 Local URL: http://localhost:8501

# Acknowledgements
Dataset — Free Spoken Digit Dataset (FSDD)
Tooling — NumPy, Librosa, Scikit-learn, Streamlit, Hugging Face Datasets
