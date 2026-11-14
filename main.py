"""Student Course Completion Prediction API.

Run with:
    uvicorn main:app --reload
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier

FEATURE_NAMES = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance\t",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

app = FastAPI(
    title="Student Course Completion Prediction API",
    description="Predicts student dropout or graduation using a trained ML model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).resolve().parent / "student_completion_model.joblib"

bundle: dict | None = None
model = None
class_names: list[str] = []
feature_names: list[str] = []


def _make_training_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(1800):
        row = {
            "Marital status": int(rng.integers(1, 5)),
            "Application mode": int(rng.integers(1, 7)),
            "Application order": int(rng.integers(1, 5)),
            "Course": int(rng.choice([33, 34, 35, 91, 92])),
            "Daytime/evening attendance\t": int(rng.integers(0, 2)),
            "Previous qualification": int(rng.integers(1, 4)),
            "Previous qualification (grade)": float(rng.uniform(120, 200)),
            "Nacionality": int(rng.integers(1, 4)),
            "Mother's qualification": int(rng.integers(1, 22)),
            "Father's qualification": int(rng.integers(1, 22)),
            "Mother's occupation": int(rng.integers(1, 10)),
            "Father's occupation": int(rng.integers(1, 10)),
            "Admission grade": float(rng.uniform(100, 200)),
            "Displaced": int(rng.integers(0, 2)),
            "Educational special needs": int(rng.integers(0, 2)),
            "Debtor": int(rng.integers(0, 2)),
            "Tuition fees up to date": int(rng.integers(0, 2)),
            "Gender": int(rng.integers(0, 2)),
            "Scholarship holder": int(rng.integers(0, 2)),
            "Age at enrollment": int(rng.integers(17, 45)),
            "International": int(rng.integers(0, 2)),
            "Curricular units 1st sem (credited)": int(rng.integers(0, 5)),
            "Curricular units 1st sem (enrolled)": int(rng.integers(2, 8)),
            "Curricular units 1st sem (evaluations)": int(rng.integers(0, 15)),
            "Curricular units 1st sem (approved)": int(rng.integers(0, 8)),
            "Curricular units 1st sem (grade)": float(rng.uniform(0, 20)),
            "Curricular units 1st sem (without evaluations)": int(rng.integers(0, 3)),
            "Curricular units 2nd sem (credited)": int(rng.integers(0, 5)),
            "Curricular units 2nd sem (enrolled)": int(rng.integers(2, 8)),
            "Curricular units 2nd sem (evaluations)": int(rng.integers(0, 15)),
            "Curricular units 2nd sem (approved)": int(rng.integers(0, 8)),
            "Curricular units 2nd sem (grade)": float(rng.uniform(0, 20)),
            "Curricular units 2nd sem (without evaluations)": int(rng.integers(0, 3)),
            "Unemployment rate": float(rng.uniform(3, 20)),
            "Inflation rate": float(rng.uniform(0.2, 10)),
            "GDP": float(rng.uniform(0.5, 10)),
        }

        completion_score = (
            row["Tuition fees up to date"] * 1.2
            + row["Curricular units 1st sem (approved)"] * 1.1
            + row["Curricular units 2nd sem (approved)"] * 1.1
            + row["Admission grade"] / 30
            + (1 if row["Scholarship holder"] else 0)
            - (1 if row["Debtor"] else 0) * 2
            - max(0, 6 - row["Age at enrollment"]) * 0.15
        )
        row["target"] = int(completion_score >= 12)
        rows.append(row)

    frame = pd.DataFrame(rows)
    return frame


def _train_default_model() -> dict:
    frame = _make_training_data()
    feature_data = frame[FEATURE_NAMES]
    labels = frame["target"]

    model = RandomForestClassifier(
        n_estimators=220,
        random_state=42,
        max_depth=10,
        min_samples_leaf=4,
    )
    model.fit(feature_data, labels)

    trained_bundle = {
        "task": "binary-classification",
        "model": model,
        "class_names": ["Dropout", "Graduate"],
        "features": FEATURE_NAMES,
    }
    joblib.dump(trained_bundle, MODEL_PATH)
    return trained_bundle


def _load_model_bundle() -> dict:
    global bundle, model, class_names, feature_names

    if bundle is not None and model is not None:
        return bundle

    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
    else:
        bundle = _train_default_model()

    model = bundle["model"]
    class_names = bundle["class_names"]
    feature_names = bundle["features"]
    return bundle


def _require_model() -> dict:
    try:
        return _load_model_bundle()
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load or train the model: {exc}",
        ) from exc


def _build_prediction(student: dict) -> dict:
    loaded_bundle = _require_model()
    missing = [f for f in feature_names if f not in student]
    extra = [f for f in student if f not in feature_names]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Required features are missing.",
                "missing_features": missing,
                "hint": "Call GET /model-info to see the full feature list.",
            },
        )

    try:
        input_frame = pd.DataFrame(
            [[student[f] for f in feature_names]],
            columns=feature_names,
        )
        prediction = int(model.predict(input_frame)[0])
        probabilities = model.predict_proba(input_frame)[0]

        return {
            "prediction": class_names[prediction],
            "probabilities": {
                class_names[i]: round(float(probabilities[i]), 4)
                for i in range(len(class_names))
            },
            "ignored_extra_fields": extra,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not process student data: {exc}",
        ) from exc


@app.get("/")
def home():
    loaded_bundle = _require_model()
    return {
        "message": "Student Course Completion Prediction API is running.",
        "docs": "/docs",
        "model_task": loaded_bundle.get("task", "unknown"),
    }


@app.get("/health")
def health():
    loaded_bundle = _require_model()
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "task": loaded_bundle.get("task"),
    }


@app.get("/model-info")
def model_info():
    loaded_bundle = _require_model()
    return {
        "task": loaded_bundle.get("task"),
        "classes": class_names,
        "number_of_features": len(feature_names),
        "features": feature_names,
    }


@app.post("/predict")
def predict(student: dict):
    return _build_prediction(student)


@app.post("/predict/batch")
def predict_batch(students: list[dict]):
    if not students:
        raise HTTPException(status_code=422, detail="Provide at least one student.")
    return {
        "count": len(students),
        "results": [_build_prediction(s) for s in students],
    }
