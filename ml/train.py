from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=500, n_features=5,
    n_informative=4, n_redundant=0, class_sep=1.5,
    random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
accuracy = float(accuracy_score(y_test, model.predict(X_test)))
out = Path("ml/artifacts")
out.mkdir(parents=True, exist_ok=True)
joblib.dump(model, out / "risk_model.joblib")
(out / "metrics.json").write_text(
    json.dumps({"accuracy": accuracy, "dataset": "synthetic"}, indent=2))
print(f"Held-out accuracy: {accuracy:.3f}")
