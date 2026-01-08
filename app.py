import json
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, abort
from joblib import load

# ---------- Artifacts ----------
ART_DIR = Path(".")
MODEL_PATH = ART_DIR / "model.joblib"
FEATS_PATH = ART_DIR / "feature_names.json"
LABELS_PATH = ART_DIR / "label_mapping.json"
CATS_PATH  = ART_DIR / "cat_encoders.json"   # string -> int label-encoder mappings

for p in [MODEL_PATH, FEATS_PATH, LABELS_PATH, CATS_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Missing artifact: {p.name}")

model = load(MODEL_PATH)
FEATURE_NAMES = json.loads(FEATS_PATH.read_text())
LABEL_MAP = json.loads(LABELS_PATH.read_text())     # e.g. {"normal":0,"anomaly":1}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
CAT_MAPS = json.loads(CATS_PATH.read_text())        # {"protocol_type": {...}, "service": {...}, "flag": {...}}

# Validate categorical fields are present
CATEGORICALS = list(CAT_MAPS.keys())                # ["protocol_type","service","flag"]
missing = [c for c in CATEGORICALS if c not in FEATURE_NAMES]
if missing:
    raise ValueError(f"Categorical feature(s) {missing} not in feature_names.json")

# Dropdown options (original strings)
CAT_OPTIONS = {col: list(CAT_MAPS[col].keys()) for col in CATEGORICALS}

# ---------- Flask ----------
app = Flask(
    __name__,
    static_url_path="/static",
    static_folder="static",
    template_folder="templates",
)

# cache-bust static
app.config["STATIC_VERSION"] = "3"

@app.context_processor
def inject_static_version():
    return {"static_version": app.config.get("STATIC_VERSION", "1")}

@app.get("/health")
def health():
    return {"status": "ok"}, 200


def parse_form_to_row(form):
    """
    Returns (display_values, numeric_values, errors, threshold)
    - display_values: strings as entered/selected
    - numeric_values: encodings ready for model (floats + label-encoded ints)
    """
    errors = []
    # optional threshold
    th_raw = form.get("threshold", "").strip()
    threshold = None
    if th_raw:
        try:
            threshold = float(th_raw)
            if not (0.0 <= threshold <= 1.0):
                raise ValueError
        except Exception:
            errors.append("Decision threshold must be a number in [0, 1].")

    display_values = {}
    numeric_values = {}

    for feat in FEATURE_NAMES:
        raw = form.get(feat, "").strip()
        if raw == "":
            errors.append(f"'{feat}' is required.")
            continue

        if feat in CATEGORICALS:
            valid = CAT_OPTIONS[feat]
            if raw not in valid:
                errors.append(f"'{feat}' must be one of: {', '.join(valid)}")
            else:
                display_values[feat] = raw
                numeric_values[feat] = CAT_MAPS[feat][raw]  # string -> int
        else:
            try:
                val = float(raw)
                display_values[feat] = raw
                numeric_values[feat] = val
            except Exception:
                errors.append(f"'{feat}' must be a valid number.")

    return display_values, numeric_values, errors, threshold


@app.get("/")
def home():
    return render_template(
        "index.html",
        feature_names=FEATURE_NAMES,
        cat_options=CAT_OPTIONS,
        categoricals=CATEGORICALS,
        form_values={},
        result=None,
        errors=[],
        threshold=None,
    )


@app.post("/predict")
def predict():
    form_values, numeric_row, errors, threshold = parse_form_to_row(request.form)
    result = None

    if not errors:
        try:
            X_inf = pd.DataFrame([[numeric_row[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
        except KeyError as e:
            abort(400, f"Input is missing required feature: {e}")

        # Prediction
        y_pred = model.predict(X_inf)
        label_id = int(y_pred[0])
        label_name = INV_LABEL_MAP.get(label_id, str(label_id))

        # Probability if supported
        anomaly_prob = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_inf)[0]
                pos_idx = 1 if len(proba) > 1 else 0  # assume 1=anomaly
                anomaly_prob = float(proba[pos_idx])
            except Exception:
                anomaly_prob = None

        # Optional threshold override of final label
        overridden = False
        if threshold is not None and anomaly_prob is not None:
            overridden = True
            if anomaly_prob >= threshold:
                label_id = LABEL_MAP.get("anomaly", 1)
                label_name = "anomaly"
            else:
                label_id = LABEL_MAP.get("normal", 0)
                label_name = "normal"

        result = {
            "label": label_name,
            "label_id": label_id,
            "anomaly_prob": anomaly_prob,
            "threshold": threshold,
            "overridden": overridden,
        }

    return render_template(
        "index.html",
        feature_names=FEATURE_NAMES,
        cat_options=CAT_OPTIONS,
        categoricals=CATEGORICALS,
        form_values=form_values,
        result=result,
        errors=errors,
        threshold=threshold,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)