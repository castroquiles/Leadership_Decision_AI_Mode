"""
Leadership Decision AI Model — Streamlit Dashboard
Run locally:  streamlit run app.py
Deploy free:  https://streamlit.io/cloud
"""

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Leadership Decision AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Leadership Decision AI Model")
st.caption("Visualize decision patterns, model performance, and live predictions.")

# ─────────────────────────────────────────────
# DATA + MODEL (cached so it only runs once)
# ─────────────────────────────────────────────
@st.cache_data
def load_data(n_samples=300):
    rng = np.random.RandomState(42)
    urgency        = rng.uniform(0, 10, n_samples)
    stakeholders   = rng.randint(1, 20, n_samples).astype(float)
    data_avail     = rng.uniform(0, 10, n_samples)
    political_risk = rng.uniform(0, 10, n_samples)
    resource_cost  = rng.uniform(0, 10, n_samples)
    impact = ((urgency + data_avail - political_risk - resource_cost * 0.5) > 8).astype(int)
    df = pd.DataFrame({
        "urgency_score":      urgency,
        "stakeholder_count":  stakeholders,
        "data_availability":  data_avail,
        "political_risk":     political_risk,
        "resource_cost":      resource_cost,
        "impact":             impact
    })
    return df

@st.cache_resource
def train_models(df):
    feature_cols = ["urgency_score","stakeholder_count","data_availability","political_risk","resource_cost"]
    X = df[feature_cols].values
    y = df["impact"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    return clf, kmeans, scaler, X_test, y_test, feature_cols

df = load_data()
clf, kmeans, scaler, X_test, y_test, feature_cols = train_models(df)

# ─────────────────────────────────────────────
# SECTION 1 — DATA OVERVIEW
# ─────────────────────────────────────────────
st.header("📊 Data Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Decisions", len(df))
col2.metric("High Impact", int(df["impact"].sum()))
col3.metric("Low Impact", int((df["impact"] == 0).sum()))

with st.expander("View raw data sample"):
    st.dataframe(df.head(20), use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 2 — FEATURE DISTRIBUTIONS
# ─────────────────────────────────────────────
st.header("📈 Feature Distributions")

selected_feature = st.selectbox("Select a feature to explore:", feature_cols)
fig_hist = px.histogram(
    df, x=selected_feature, color=df["impact"].map({0: "Low Impact", 1: "High Impact"}),
    barmode="overlay", nbins=30,
    color_discrete_map={"Low Impact": "#636EFA", "High Impact": "#EF553B"},
    labels={"color": "Impact"}
)
st.plotly_chart(fig_hist, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 3 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────
st.header("🔍 Feature Importance — What Drives Impact?")

importances = clf.feature_importances_
fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
fi_df = fi_df.sort_values("Importance", ascending=True)

fig_fi = px.bar(
    fi_df, x="Importance", y="Feature", orientation="h",
    color="Importance", color_continuous_scale="Blues",
    title="Random Forest Feature Importance"
)
st.plotly_chart(fig_fi, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 4 — CLUSTER PROFILES
# ─────────────────────────────────────────────
st.header("🧩 Decision Profiles (Clusters)")

df["cluster"] = kmeans.labels_
cluster_names = {0: "Profile 1", 1: "Profile 2", 2: "Profile 3"}
df["profile"] = df["cluster"].map(cluster_names)

fig_scatter = px.scatter(
    df, x="urgency_score", y="political_risk",
    color="profile", size="data_availability",
    hover_data=["stakeholder_count", "resource_cost"],
    title="Decision Profiles: Urgency vs Political Risk (bubble size = data availability)",
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig_scatter, use_container_width=True)

centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=feature_cols)
centers_df.index = ["Profile 1", "Profile 2", "Profile 3"]
st.subheader("Cluster Center Values")
st.dataframe(centers_df.round(2), use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 5 — MODEL PERFORMANCE
# ─────────────────────────────────────────────
st.header("✅ Model Performance")

y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=["Low Impact", "High Impact"], output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)
st.dataframe(report_df, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 6 — LIVE PREDICTION
# ─────────────────────────────────────────────
st.header("🎯 Predict a New Decision")
st.write("Adjust the sliders and see the model's prediction in real time.")

col_a, col_b = st.columns(2)
with col_a:
    urgency     = st.slider("Urgency Score",       0.0, 10.0, 5.0, 0.1)
    stakeholders = st.slider("Stakeholder Count",  1,   20,   10)
    data_avail  = st.slider("Data Availability",   0.0, 10.0, 5.0, 0.1)
with col_b:
    pol_risk    = st.slider("Political Risk",       0.0, 10.0, 5.0, 0.1)
    res_cost    = st.slider("Resource Cost",        0.0, 10.0, 5.0, 0.1)

input_data = np.array([[urgency, stakeholders, data_avail, pol_risk, res_cost]])
prediction = clf.predict(input_data)[0]
probability = clf.predict_proba(input_data)[0]
confidence = max(probability) * 100
label = "🔴 High Impact" if prediction == 1 else "🟢 Low Impact"

st.subheader(f"Prediction: {label}")
st.progress(int(confidence))
st.write(f"Confidence: **{confidence:.1f}%**")

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability[1] * 100,
    title={"text": "High Impact Probability (%)"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "#EF553B"},
        "steps": [
            {"range": [0, 40], "color": "#d4edda"},
            {"range": [40, 70], "color": "#fff3cd"},
            {"range": [70, 100], "color": "#f8d7da"},
        ],
    }
))
st.plotly_chart(fig_gauge, use_container_width=True)
