import json
from pathlib import Path
import joblib

ARTIFACTS = Path("ml/artifacts")

def test_model_and_metrics():
    model = joblib.load(ARTIFACTS / "risk_model.joblib")
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    assert hasattr(model, "predict")
    assert metrics["dataset"] == "synthetic"
    assert metrics["accuracy"] >= 0.80


