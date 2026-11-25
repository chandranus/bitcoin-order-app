from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Project directories
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Reproducible synthetic dataset
rng = np.random.default_rng(42)
n_samples = 2000

order_amount = rng.uniform(100, 10000, n_samples)
price_volatility = rng.uniform(0, 1, n_samples)
transaction_frequency = rng.integers(1, 50, n_samples)

X = np.column_stack([
    order_amount,
    price_volatility,
    transaction_frequency
])

# Synthetic risk labels
# Educational rules, not real financial risk criteria
risk_score = (
    (order_amount > 7000).astype(int)
    + (price_volatility > 0.65).astype(int)
    + (transaction_frequency > 35).astype(int)
)

y = (risk_score >= 2).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate on held-out test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save model
joblib.dump(model, ARTIFACTS / "risk_model.joblib")

# Save evaluation metrics
metrics = {
    "dataset": "synthetic",
    "accuracy": float(accuracy),
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "model": "RandomForestClassifier",
    "random_state": 42
}

(ARTIFACTS / "metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8"
)

print("MLOps Model Training Completed")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Model accuracy: {accuracy:.2%}")
print(f"Model saved to: {ARTIFACTS}")
