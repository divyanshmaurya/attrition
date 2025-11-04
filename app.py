
import os
import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict, Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Employee Attrition Simulation",
    page_icon="💼",
    layout="wide"
)

# ==========================
# Utilities
# ==========================

def is_pipeline(model_obj) -> bool:
    from sklearn.pipeline import Pipeline as SKPipeline
    return isinstance(model_obj, SKPipeline)

def load_model(pkl_bytes: bytes):
    try:
        return pickle.loads(pkl_bytes)
    except Exception as e:
        st.error(f"Could not load the uploaded model: {e}")
        return None

def try_load_local_best_model(path: str = "best_model.pkl"):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Found {path} but failed to load: {e}")
    return None

def infer_schema_from_csv(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer numeric and categorical columns from a DataFrame."""
    # Drop target if present
    for tgt in ["Attrition", "attrition", "TARGET", "target", "label", "Label"]:
        if tgt in df.columns:
            df = df.drop(columns=[tgt])
            break
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols

def default_hr_schema() -> Tuple[List[str], List[str]]:
    """A sensible default schema for common HR attrition datasets."""
    numeric = [
        "Age",
        "MonthlyIncome",
        "DistanceFromHome",
        "TotalWorkingYears",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
        "NumCompaniesWorked",
        "PercentSalaryHike",
        "TrainingTimesLastYear",
        "JobLevel",
        "Education"
    ]
    categorical = [
        "BusinessTravel",
        "Department",
        "EducationField",
        "Gender",
        "JobRole",
        "MaritalStatus",
        "OverTime"
    ]
    return numeric, categorical

def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor

def make_default_classifier() -> Pipeline:
    # Fallback model so the app runs even if no model is provided.
    numeric, categorical = default_hr_schema()
    clf = Pipeline(steps=[
        ("pre", build_preprocessor(numeric, categorical)),
        ("clf", LogisticRegression(max_iter=200))
    ])
    return clf

def ensure_pipeline(model_obj, numeric: List[str], categorical: List[str]):
    """Wrap model in a Pipeline with preprocessing if it's not already a Pipeline."""
    if is_pipeline(model_obj):
        return model_obj
    pre = build_preprocessor(numeric, categorical)
    pipe = Pipeline(steps=[("pre", pre), ("clf", model_obj)])
    return pipe

def get_proba(model, X: pd.DataFrame) -> float:
    """Return probability of positive class if available, else a calibrated proxy."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        # Assume positive class is class "Yes" or 1; pick the last column if ambiguous
        if p.shape[1] == 2:
            return float(p[:, 1][0])
        else:
            # multi-class; try to find "Yes" or 1
            try:
                classes_ = list(model.classes_)
                if "Yes" in classes_:
                    idx = classes_.index("Yes")
                elif 1 in classes_:
                    idx = classes_.index(1)
                else:
                    idx = np.argmax(p[0])
                return float(p[:, idx][0])
            except Exception:
                return float(np.max(p[0]))
    # Fallback: use decision_function if present
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        # Map score to (0,1) via sigmoid for a pseudo-probability
        return 1.0 / (1.0 + np.exp(-score))
    # Final fallback: 0/1 prediction
    pred = model.predict(X)[0]
    return float(pred) if pred in [0, 1] else (1.0 if str(pred).lower() in ["yes", "true", "leave"] else 0.0)

# ==========================
# Sidebar: Model & Schema
# ==========================
st.sidebar.title("⚙️ Configuration")

# Try to load a local model first
model_obj = try_load_local_best_model("best_model.pkl")

uploaded_model = st.sidebar.file_uploader("Upload trained model (.pkl)", type=["pkl"], help="Upload a pickled scikit-learn model or Pipeline.")
if uploaded_model is not None:
    model_bytes = uploaded_model.read()
    model_obj = load_model(model_bytes)

# Schema: allow user to upload a CSV to infer columns
uploaded_csv = st.sidebar.file_uploader("Upload sample data (CSV) to infer schema", type=["csv"])

if uploaded_csv is not None:
    df_schema = pd.read_csv(uploaded_csv)
    numeric_cols, categorical_cols = infer_schema_from_csv(df_schema)
else:
    numeric_cols, categorical_cols = default_hr_schema()

st.sidebar.markdown("**Numeric features detected:**")
st.sidebar.write(numeric_cols)
st.sidebar.markdown("**Categorical features detected:**")
st.sidebar.write(categorical_cols)

# If we still don't have a model, create a placeholder pipeline (not trained)
if model_obj is None:
    st.sidebar.warning("No model provided. Using a placeholder Logistic Regression pipeline (untrained). Upload your real model for meaningful predictions.")
    model_obj = make_default_classifier()

# Ensure we have a Pipeline with preprocessing
model_pipe = ensure_pipeline(model_obj, numeric_cols, categorical_cols)

# ==========================
# Main UI
# ==========================
st.title("💼 Employee Attrition Simulation")
st.caption("Interactive demo: adjust employee attributes and see attrition predictions in real time.")

colA, colB = st.columns([2, 1])

with colA:
    st.subheader("🔢 Enter Employee Details")

    # Build input widgets dynamically from schema
    user_inputs: Dict[str, Any] = {}

    # Numeric inputs with heuristic ranges (can be adjusted)
    numeric_defaults = {
        "Age": (18, 60, 30),
        "MonthlyIncome": (1000, 25000, 5000),
        "DistanceFromHome": (0, 50, 10),
        "TotalWorkingYears": (0, 40, 8),
        "YearsAtCompany": (0, 40, 5),
        "YearsInCurrentRole": (0, 20, 3),
        "YearsSinceLastPromotion": (0, 20, 1),
        "YearsWithCurrManager": (0, 20, 2),
        "NumCompaniesWorked": (0, 15, 2),
        "PercentSalaryHike": (0, 100, 15),
        "TrainingTimesLastYear": (0, 20, 3),
        "JobLevel": (1, 5, 2),
        "Education": (1, 5, 3)
    }

    # Categorical defaults/options
    cat_defaults = {
        "BusinessTravel": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
        "Department": ["Sales", "Research & Development", "Human Resources"],
        "EducationField": ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
        "Gender": ["Male", "Female"],
        "JobRole": ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"],
        "MaritalStatus": ["Single", "Married", "Divorced"],
        "OverTime": ["No", "Yes"]
    }

    # Build UI in two columns for compactness
    n1, n2 = st.columns(2)
    for i, col in enumerate(numeric_cols):
        rng = numeric_defaults.get(col, (0, 100, 10))
        with (n1 if i % 2 == 0 else n2):
            user_inputs[col] = st.slider(f"{col}", min_value=float(rng[0]), max_value=float(rng[1]), value=float(rng[2]))

    c1, c2 = st.columns(2)
    for i, col in enumerate(categorical_cols):
        opts = cat_defaults.get(col, ["Unknown", "Value1", "Value2"])
        with (c1 if i % 2 == 0 else c2):
            user_inputs[col] = st.selectbox(f"{col}", options=opts)

    input_df = pd.DataFrame([user_inputs])

    predict_btn = st.button("🔮 Run Simulation")

with colB:
    st.subheader("📊 Prediction")
    if predict_btn:
        try:
            proba = get_proba(model_pipe, input_df)
            pred_label = "Will Leave" if proba >= 0.5 else "Will Stay"
            score_pct = f"{proba*100:.1f}%"
            # Metrics-style display
            st.metric(label="Attrition Probability", value=score_pct)
            if pred_label == "Will Leave":
                st.markdown("**Predicted Class:** :red_circle: **Will Leave**")
            else:
                st.markdown("**Predicted Class:** :green_circle: **Will Stay**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Set inputs and click **Run Simulation** to see results.")

st.markdown("---")
st.subheader("🧪 Sensitivity: How does probability change if one feature varies?")

left, right = st.columns([1, 3])

with left:
    sweep_feature = st.selectbox("Feature to vary", options=numeric_cols + categorical_cols)
    num_points = st.slider("Resolution", min_value=5, max_value=50, value=25)

with right:
    try:
        if sweep_feature in numeric_cols:
            # Build a range across the widget min/max if we have it, else use data-driven defaults
            rng = numeric_defaults.get(sweep_feature, (0, 100, 10))
            xs = np.linspace(rng[0], rng[1], num_points)
            probs = []
            for val in xs:
                row = input_df.copy()
                row[sweep_feature] = float(val)
                probs.append(get_proba(model_pipe, row))
            fig = plt.figure()
            plt.plot(xs, probs)
            plt.xlabel(sweep_feature)
            plt.ylabel("Attrition Probability")
            plt.title(f"Probability vs {sweep_feature}")
            st.pyplot(fig)
        else:
            # categorical sweep
            opts = cat_defaults.get(sweep_feature, ["Unknown", "Value1", "Value2"])
            probs = []
            for val in opts:
                row = input_df.copy()
                row[sweep_feature] = val
                probs.append(get_proba(model_pipe, row))
            fig = plt.figure()
            plt.plot(range(len(opts)), probs, marker="o")
            plt.xticks(range(len(opts)), opts, rotation=30)
            plt.ylabel("Attrition Probability")
            plt.title(f"Probability vs {sweep_feature}")
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Sensitivity analysis failed: {e}")

# Footer / Help
with st.expander("ℹ️ Help & Notes"):
    st.markdown("""
- If you trained and saved a model in your notebook, export it as a `.pkl`:
  ```python
  import joblib
  joblib.dump(best_model, "best_model.pkl")
  ```
  Or with pickle:
  ```python
  import pickle
  with open("best_model.pkl", "wb") as f:
      pickle.dump(best_model, f)
  ```
- If your saved model is already a **Pipeline** with preprocessing, the app will use it directly.
- If it's a "bare" estimator, the app wraps it in a preprocessing pipeline using the **schema** shown in the sidebar.
- To match your dataset exactly, upload a small **CSV sample** from your training data—the app will infer numeric vs categorical columns automatically.
- The sensitivity section varies one feature while keeping the others fixed to your current inputs.
    """)
