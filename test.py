import os

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

print("Model exists:", os.path.exists(MODEL_PATH))
print("Vectorizer exists:", os.path.exists(VECTORIZER_PATH))