
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from typing import List, Tuple, Dict, Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Employee Attrition Dashboard", page_icon="📉", layout="wide")

# ==========================
# Utilities
# ==========================

def is_pipeline(model_obj) -> bool:
    from sklearn.pipeline import Pipeline as SKPipeline
    return isinstance(model_obj, SKPipeline)

def load_model_bytes(pkl_bytes: bytes):
    try:
        return pickle.loads(pkl_bytes)
    except Exception as e:
        st.error(f"Could not load the uploaded model: {e}")
        return None

def try_load_local_model(path: str = "best_model.pkl"):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Found {path} but failed to load: {e}")
    return None

def infer_schema_from_df(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    drop_cols = [c for c in ["Attrition", "attrition", "TARGET", "target", "label", "Label"] if c in df.columns]
    df2 = df.drop(columns=drop_cols) if drop_cols else df
    num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df2.columns if c not in num_cols]
    return num_cols, cat_cols

def default_hr_schema() -> Tuple[List[str], List[str]]:
    numeric = [
        "Age","MonthlyIncome","DistanceFromHome","TotalWorkingYears","YearsAtCompany",
        "YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager","NumCompaniesWorked",
        "PercentSalaryHike","TrainingTimesLastYear","JobLevel","Education"
    ]
    categorical = [
        "BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"
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

def ensure_pipeline(model_obj, numeric: List[str], categorical: List[str]):
    """Wrap estimator in a preprocessing pipeline if needed."""
    if is_pipeline(model_obj):
        return model_obj
    pre = build_preprocessor(numeric, categorical)
    return Pipeline(steps=[("pre", pre), ("clf", model_obj)])

def get_positive_proba(model, X: pd.DataFrame) -> float:
    """Return probability of positive class if available; otherwise map decision_function or prediction."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] == 2:
            return float(p[:, 1][0])
        # multiclass: choose max as proxy
        return float(np.max(p[0]))
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        return 1.0 / (1.0 + np.exp(-score))
    pred = model.predict(X)[0]
    return float(pred) if pred in [0, 1] else 0.0

def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

# ==========================
# Sidebar: Inputs & Uploads
# ==========================
st.sidebar.title("⚙️ Configuration")

# Load model (local auto or upload)
model_obj = try_load_local_model("best_model.pkl")
uploaded_model = st.sidebar.file_uploader("Upload trained model (.pkl)", type=["pkl"])
if uploaded_model is not None:
    model_obj = load_model_bytes(uploaded_model.read())

# Load dataset (auto or upload)
data = pd.DataFrame()
if os.path.exists("employee_data.csv"):
    data = safe_read_csv("employee_data.csv")

uploaded_csv = st.sidebar.file_uploader("Upload dataset CSV (optional)", type=["csv"])
if uploaded_csv is not None:
    try:
        data = pd.read_csv(uploaded_csv)
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")

# Optional model scores
scores_df = pd.DataFrame()
if os.path.exists("model_scores.csv"):
    try:
        scores_df = pd.read_csv("model_scores.csv", index_col=0)
    except Exception:
        pass

uploaded_scores = st.sidebar.file_uploader("Upload model_scores.csv (optional)", type=["csv"])
if uploaded_scores is not None:
    try:
        scores_df = pd.read_csv(uploaded_scores)
        # Try to make model names the index if applicable
        if "Model" in scores_df.columns:
            scores_df = scores_df.set_index("Model")
    except Exception as e:
        st.sidebar.error(f"Failed to read model_scores.csv: {e}")

# Feature schema
if not data.empty:
    numeric_cols, categorical_cols = infer_schema_from_df(data)
else:
    numeric_cols, categorical_cols = default_hr_schema()

# Ensure we have a model
if model_obj is None:
    st.sidebar.error("No trained model available. Upload a `.pkl` model or place `best_model.pkl` beside this app.")
else:
    st.sidebar.success("Model loaded.")

st.sidebar.markdown("**Detected numeric features:**")
st.sidebar.write(numeric_cols)
st.sidebar.markdown("**Detected categorical features:**")
st.sidebar.write(categorical_cols)

# Final pipeline (wrap if needed)
if model_obj is not None:
    model_pipe = ensure_pipeline(model_obj, numeric_cols, categorical_cols)
else:
    model_pipe = None

# ==========================
# Main Layout with Tabs
# ==========================
st.title("📉 Employee Attrition — Analytics & Simulation Dashboard")

tab_overview, tab_models, tab_eda, tab_predict = st.tabs(["🏠 Overview", "📊 Model Comparison", "📈 Explore Data", "🔮 Predict & Sensitivity"])

# --------------------------
# Overview Tab
# --------------------------
with tab_overview:
    st.subheader("Dataset Snapshot")
    if data.empty:
        st.info("Upload a dataset CSV (e.g., `employee_data.csv`) to see overview stats and visuals.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        total_rows = len(data)
        attr_col = "Attrition" if "Attrition" in data.columns else ("attrition" if "attrition" in data.columns else None)
        attr_rate = None
        if attr_col:
            try:
                # Accept Yes/No or 1/0
                y = data[attr_col]
                if y.dtype == object:
                    attr_rate = (y.str.lower().isin(["yes", "true", "1"])).mean()
                else:
                    attr_rate = y.mean()
            except Exception:
                attr_rate = None

        with col1:
            st.metric("Total Records", f"{total_rows:,}")
        with col2:
            st.metric("Attrition Rate", f"{(attr_rate*100):.1f}%" if attr_rate is not None else "—")
        with col3:
            if "MonthlyIncome" in data.columns:
                st.metric("Avg Monthly Income", f"${data['MonthlyIncome'].mean():,.0f}")
            else:
                st.metric("Avg Monthly Income", "—")
        with col4:
            if "YearsAtCompany" in data.columns:
                st.metric("Avg Years at Company", f"{data['YearsAtCompany'].mean():.1f}")
            else:
                st.metric("Avg Years at Company", "—")

        st.markdown("### Attrition Distribution")
        if attr_col:
            dist_df = data[attr_col].value_counts(dropna=False).rename_axis("Attrition").reset_index(name="Count")
            chart = alt.Chart(dist_df).mark_bar().encode(
                x=alt.X("Attrition:N", title="Attrition"),
                y=alt.Y("Count:Q", title="Count"),
                tooltip=["Attrition", "Count"]
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No `Attrition` column found.")

# --------------------------
# Model Comparison Tab
# --------------------------
with tab_models:
    st.subheader("Model Performance Overview")
    if scores_df.empty:
        st.info("Upload `model_scores.csv` to visualize model comparison (accuracy, precision, recall, etc.).")
        st.caption("Tip: In your notebook, export a table of model metrics to CSV and upload here.")
    else:
        st.write("Raw scores:")
        st.dataframe(scores_df, use_container_width=True)
        # If there's a column named 'Test' or 'Accuracy', plot it
        metric_col = None
        for candidate in ["Test", "Accuracy", "ROC_AUC", "AUC", "F1", "Precision", "Recall"]:
            if candidate in scores_df.columns:
                metric_col = candidate
                break
        if metric_col:
            chart_df = scores_df.reset_index().rename(columns={"index": "Model"})
            chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("Model:N", sort="-y"),
                y=alt.Y(f"{metric_col}:Q", title=metric_col),
                tooltip=["Model", metric_col]
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No recognized metric column found (looked for Test/Accuracy/ROC_AUC/AUC/F1/Precision/Recall).")

# --------------------------
# EDA Tab
# --------------------------
with tab_eda:
    st.subheader("Explore Data")
    if data.empty:
        st.info("Upload a dataset CSV to explore.")
    else:
        st.markdown("#### Quick Preview")
        st.dataframe(data.head(50), use_container_width=True)

        st.markdown("#### Numeric Feature Distribution")
        num_to_plot = st.selectbox("Pick a numeric feature", options=[c for c in data.columns if c in numeric_cols] or data.select_dtypes(include=[np.number]).columns.tolist())
        if num_to_plot:
            hist = alt.Chart(data).mark_bar().encode(
                x=alt.X(f"{num_to_plot}:Q", bin=alt.Bin(maxbins=30), title=num_to_plot),
                y=alt.Y("count():Q", title="Count"),
                tooltip=[num_to_plot, "count()"]
            )
            st.altair_chart(hist, use_container_width=True)

        st.markdown("#### Attrition by Category")
        if "Attrition" in data.columns:
            cat_to_plot = st.selectbox("Pick a categorical feature", options=categorical_cols)
            if cat_to_plot:
                grp = data.groupby(cat_to_plot)["Attrition"].apply(lambda s: (s.astype(str).str.lower().isin(["yes","true","1"])).mean()).reset_index(name="AttritionRate")
                bar = alt.Chart(grp).mark_bar().encode(
                    x=alt.X(f"{cat_to_plot}:N", sort="-y", title=cat_to_plot),
                    y=alt.Y("AttritionRate:Q", title="Attrition Rate"),
                    tooltip=[cat_to_plot, alt.Tooltip("AttritionRate:Q", format=".2f")]
                )
                st.altair_chart(bar, use_container_width=True)
        else:
            st.info("No `Attrition` column to compute rates.")

# --------------------------
# Predict Tab
# --------------------------
with tab_predict:
    st.subheader("Predict Attrition & Sensitivity")
    if model_pipe is None:
        st.warning("Load a trained model to enable predictions.")
    else:
        # UI for inputs
        user_inputs: Dict[str, Any] = {}

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

        cat_defaults = {
            "BusinessTravel": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
            "Department": ["Sales", "Research & Development", "Human Resources"],
            "EducationField": ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
            "Gender": ["Male", "Female"],
            "JobRole": ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
                        "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"],
            "MaritalStatus": ["Single", "Married", "Divorced"],
            "OverTime": ["No", "Yes"]
        }

        # Build layout
        c1, c2 = st.columns(2)
        for i, col in enumerate(numeric_cols):
            rng = numeric_defaults.get(col, (0, 100, 10))
            with (c1 if i % 2 == 0 else c2):
                user_inputs[col] = st.slider(f"{col}", min_value=float(rng[0]), max_value=float(rng[1]), value=float(rng[2]))

        c3, c4 = st.columns(2)
        for i, col in enumerate(categorical_cols):
            opts = cat_defaults.get(col, ["Unknown", "Value1", "Value2"])
            with (c3 if i % 2 == 0 else c4):
                user_inputs[col] = st.selectbox(f"{col}", options=opts)

        input_df = pd.DataFrame([user_inputs])

        run = st.button("🔮 Predict")
        if run:
            try:
                proba = get_positive_proba(model_pipe, input_df)
                pred_label = "Will Leave" if proba >= 0.5 else "Will Stay"
                st.metric("Attrition Probability", f"{proba*100:.1f}%")
                st.markdown(f"**Predicted Class:** :{'red_circle' if pred_label == 'Will Leave' else 'green_circle'}: **{pred_label}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.markdown("---")
        st.markdown("### Sensitivity Analysis")
        left, right = st.columns([1, 3])
        with left:
            sweep_feature = st.selectbox("Feature to vary", options=numeric_cols + categorical_cols)
            points = st.slider("Resolution", min_value=5, max_value=50, value=25)

        with right:
            try:
                if sweep_feature in numeric_cols:
                    rng = numeric_defaults.get(sweep_feature, (0, 100, 10))
                    xs = np.linspace(rng[0], rng[1], points)
                    probs = []
                    for val in xs:
                        row = input_df.copy()
                        row[sweep_feature] = float(val)
                        probs.append(get_positive_proba(model_pipe, row))
                    sens_df = pd.DataFrame({sweep_feature: xs, "Probability": probs})
                    line = alt.Chart(sens_df).mark_line().encode(
                        x=alt.X(f"{sweep_feature}:Q", title=sweep_feature),
                        y=alt.Y("Probability:Q", title="Attrition Probability"),
                        tooltip=[sweep_feature, alt.Tooltip("Probability:Q", format=".2f")]
                    )
                    st.altair_chart(line, use_container_width=True)
                else:
                    opts = cat_defaults.get(sweep_feature, ["Unknown", "Value1", "Value2"])
                    probs = []
                    for val in opts:
                        row = input_df.copy()
                        row[sweep_feature] = val
                        probs.append(get_positive_proba(model_pipe, row))
                    sens_df = pd.DataFrame({sweep_feature: opts, "Probability": probs})
                    line = alt.Chart(sens_df).mark_line(point=True).encode(
                        x=alt.X(f"{sweep_feature}:N", title=sweep_feature),
                        y=alt.Y("Probability:Q", title="Attrition Probability"),
                        tooltip=[sweep_feature, alt.Tooltip("Probability:Q", format=".2f")]
                    )
                    st.altair_chart(line, use_container_width=True)
            except Exception as e:
                st.error(f"Sensitivity analysis failed: {e}")
