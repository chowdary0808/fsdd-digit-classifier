# PROMPTS — LLM Collaboration Log


# Prompt: With a 2–3 hour budget for a spoken-digit prototype + Streamlit demo, propose a minimal module layout and data flow. Prefer small, testable functions.
  Outcome: Adopted src/{data,features,model,train,evaluate,infer}.py + app.py; simple audio→features→model pipeline.

# Prompt: HF dataset (mteb/free-spoken-digit-dataset) may require an audio decoder on Windows. Suggest a portable fallback that yields the same WAV arrays.
  Outcome: Implemented HF-first loader with graceful fallback to the official FSDD ZIP (requests+zipfile+soundfile).

# Prompt: For a tiny, fast baseline, compare MFCC mean+std vs log-mel pooling; recommend one with trade-offs.
  Outcome: Chose MFCC(20) mean+std (~40 dims) + Logistic Regression for instant CPU inference; kept log-mel + Linear SVM as a future variant.

# Prompt: Design a stratified 70/15/15 split with reproducibility; show the two-step split.
  Outcome: Nested train_test_split(..., stratify=y, random_state=42) stabilized metrics.

# Prompt: Create a compact confusion-matrix evaluator and a one-paragraph narration for it.
  Outcome: evaluate.py with ConfusionMatrixDisplay; narration highlights strong diagonal (~92% test), main confusion (6→3) due to mean/std MFCCs.

