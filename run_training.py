# run_training.py
# ─────────────────────────────────────────────────────────────────────────────
# Run this script ONCE to train the ANN and save:
#   • diabetes_model.h5   (the trained Keras model)
#   • scaler.pkl           (the fitted StandardScaler)
#
# Usage:
#   1.  Place diabetes.csv in the same folder as this file.
#   2.  pip install tensorflow scikit-learn pandas numpy
#   3.  python run_training.py
#
# After it finishes, you will see both files appear in the folder.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ── 2. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Scale features ────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 4. Build ANN ─────────────────────────────────────────────────────────────
model = Sequential([
    Dense(12, input_dim=8, activation="relu"),
    Dense(8,  activation="relu"),
    Dense(1,  activation="sigmoid"),
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# ── 5. Train ──────────────────────────────────────────────────────────────────
print("\nTraining model (100 epochs)...")
model.fit(X_train, y_train, epochs=100, batch_size=10,
          validation_split=0.1, verbose=0)
print("Training complete.")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
print("\n── Metrics on test set ──")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.2f}")
print(f"  Precision : {precision_score(y_test, y_pred):.2f}")
print(f"  Recall    : {recall_score(y_test, y_pred):.2f}")
print(f"  F1-score  : {f1_score(y_test, y_pred):.2f}")

# ── 7. Save artifacts ─────────────────────────────────────────────────────────
model.save("diabetes_model.h5")
pickle.dump(scaler, open("scaler.pkl", "wb"))
print("\n✅  Saved: diabetes_model.h5  and  scaler.pkl")
print("You can now run:  streamlit run app.py")
