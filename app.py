"""
Leadership Decision AI Model — Streamlit Dashboard
Clean & minimal redesign: no emojis, icons via st.icon / material symbols
"""

import io
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
# THEME INJECTION — clean minimal
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #1a1a1a;
    }
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.25rem;
    }
    .metric-card {
        background: #f7f7f7;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #ebebeb;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a1a;
        font-family: 'DM Mono', monospace;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        margin-top: 0.1rem;
    }
    .prediction-box {
        background: #f7f7f7;
        border-left: 3px solid #1a1a1a;
        border-radius: 4px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    .prediction-box.high {
        border-left-color: #c0392b;
    }
    .prediction-box.low {
        border-left-color: #27ae60;
    }
    .stButton > button {
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.85rem;
        border: 1px solid #ddd;
        background: white;
        color: #1a1a1a;
        padding: 0.4rem 1rem;
    }
    .stButton > button:hover {
        background: #1a1a1a;
        color: white;
        border-color: #1a1a1a;
    }
    [data-testid="stSidebar"] {
        background: #fafafa;
        border-right: 1px solid #ebebeb;
    }
    .stSelectbox label, .stSlider label {
        font-size: 0.82rem;
        font-weight: 500;
        color: #444;
    }
    hr {
        border: none;
        border-top: 1px solid #ebebeb;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────
CREDENTIALS = {
    "admin": "leadership2024",
    "demo":  "demo123",
}

def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True

    st.markdown("<div style='max-width:380px;margin:6rem auto 0'>", unsafe_allow_html=True)
    st.markdown("<p class='section-label'>Leadership Decision AI</p>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:1.8rem;margin-bottom:2rem'>Sign in</h1>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Continue"):
        if username in CREDENTIALS and CREDENTIALS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Incorrect username or password.")

    st.markdown("</div>", unsafe_allow_html=True)
    return False

if not check_login():
    st.stop()

# ─────────────────────────────────────────────
# DATA + MODEL
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
    return pd.DataFrame({
        "urgency_score":      urgency,
        "stakeholder_count":  stakeholders,
        "data_availability":  data_avail,
        "political_risk":     political_risk,
        "resource_cost":      resource_cost,
        "impact":             impact
    })

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
# SIDEBAR NAV
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<p class='section-label' style='margin-top:1rem'>Navigation</p>", unsafe_allow_html=True)
    page = st.radio("", [
        "Overview",
        "Distributions",
        "Feature Importance",
        "Decision Profiles",
        "Model Performance",
        "Predict",
        "Export"
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.78rem;color:#aaa'>Signed in as <strong>{st.session_state.username}</strong></p>", unsafe_allow_html=True)
    if st.button("Sign out"):
        st.session_state.authenticated = False
        st.rerun()

# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────
if page == "Overview":
    st.markdown("<p class='section-label'>Dashboard</p>", unsafe_allow_html=True)
    st.markdown("## Overview")
    st.markdown("Decision data summary — synthetic dataset. Replace with real data when available.")
    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Decisions</div>
            <div class='metric-value'>{len(df)}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>High Impact</div>
            <div class='metric-value'>{int(df['impact'].sum())}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Low Impact</div>
            <div class='metric-value'>{int((df['impact']==0).sum())}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Raw data sample (first 20 rows)"):
        st.dataframe(df.head(20), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: DISTRIBUTIONS
# ─────────────────────────────────────────────
elif page == "Distributions":
    st.markdown("<p class='section-label'>Analysis</p>", unsafe_allow_html=True)
    st.markdown("## Feature Distributions")
    st.markdown("Compare how each factor is distributed across High and Low Impact decisions.")
    st.markdown("<hr>", unsafe_allow_html=True)

    selected = st.selectbox("Select feature", feature_cols)
    fig = px.histogram(
        df, x=selected,
        color=df["impact"].map({0: "Low Impact", 1: "High Impact"}),
        barmode="overlay", nbins=30,
        color_discrete_map={"Low Impact": "#bdc3c7", "High Impact": "#2c3e50"},
        labels={"color": "Impact"}
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Inter", font_color="#444",
        legend_title_text="",
        xaxis=dict(showgrid=False, linecolor="#ebebeb"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", linecolor="#ebebeb"),
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: FEATURE IMPORTANCE
# ─────────────────────────────────────────────
elif page == "Feature Importance":
    st.markdown("<p class='section-label'>Model</p>", unsafe_allow_html=True)
    st.markdown("## What Drives Impact?")
    st.markdown("Factors ranked by their influence on the model's predictions.")
    st.markdown("<hr>", unsafe_allow_html=True)

    fi_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": clf.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=[[0, "#ebebeb"], [1, "#2c3e50"]],
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Inter", font_color="#444",
        coloraxis_showscale=False,
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", linecolor="#ebebeb"),
        yaxis=dict(showgrid=False, linecolor="#ebebeb"),
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: DECISION PROFILES
# ─────────────────────────────────────────────
elif page == "Decision Profiles":
    st.markdown("<p class='section-label'>Clusters</p>", unsafe_allow_html=True)
    st.markdown("## Decision Profiles")
    st.markdown("Three natural groupings found in the decision data.")
    st.markdown("<hr>", unsafe_allow_html=True)

    df["cluster"] = kmeans.labels_
    df["profile"] = df["cluster"].map({0: "Profile A", 1: "Profile B", 2: "Profile C"})

    fig = px.scatter(
        df, x="urgency_score", y="political_risk",
        color="profile", size="data_availability",
        hover_data=["stakeholder_count", "resource_cost"],
        color_discrete_map={
            "Profile A": "#2c3e50",
            "Profile B": "#7f8c8d",
            "Profile C": "#bdc3c7"
        }
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Inter", font_color="#444",
        legend_title_text="",
        xaxis=dict(title="Urgency Score", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(title="Political Risk", showgrid=True, gridcolor="#f0f0f0"),
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=feature_cols)
    centers_df.index = ["Profile A", "Profile B", "Profile C"]
    st.markdown("**Cluster center values**")
    st.dataframe(centers_df.round(2), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif page == "Model Performance":
    st.markdown("<p class='section-label'>Evaluation</p>", unsafe_allow_html=True)
    st.markdown("## Model Performance")
    st.markdown("Classification report on the held-out test set (20% of data).")
    st.markdown("<hr>", unsafe_allow_html=True)

    y_pred = clf.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=["Low Impact", "High Impact"],
        output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose().round(2), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────
elif page == "Predict":
    st.markdown("<p class='section-label'>Live Prediction</p>", unsafe_allow_html=True)
    st.markdown("## Predict a Decision")
    st.markdown("Adjust the inputs and the model will predict impact in real time.")
    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        urgency      = st.slider("Urgency Score",      0.0, 10.0, 5.0, 0.1)
        stakeholders = st.slider("Stakeholder Count",  1,   20,   10)
        data_avail   = st.slider("Data Availability",  0.0, 10.0, 5.0, 0.1)
    with c2:
        pol_risk     = st.slider("Political Risk",      0.0, 10.0, 5.0, 0.1)
        res_cost     = st.slider("Resource Cost",       0.0, 10.0, 5.0, 0.1)

    input_arr   = np.array([[urgency, stakeholders, data_avail, pol_risk, res_cost]])
    pred        = clf.predict(input_arr)[0]
    prob        = clf.predict_proba(input_arr)[0]
    confidence  = max(prob) * 100
    label       = "High Impact" if pred == 1 else "Low Impact"
    box_class   = "high" if pred == 1 else "low"
    accent      = "#c0392b" if pred == 1 else "#27ae60"

    st.markdown(f"""
    <div class='prediction-box {box_class}'>
        <div class='section-label'>Prediction</div>
        <div style='font-size:1.6rem;font-weight:600;color:#1a1a1a;margin:0.3rem 0'>{label}</div>
        <div style='font-size:0.85rem;color:#888'>Confidence: <strong style='color:#1a1a1a'>{confidence:.1f}%</strong></div>
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob[1] * 100,
        number={"suffix": "%", "font": {"family": "DM Mono", "color": "#1a1a1a"}},
        title={"text": "High Impact Probability", "font": {"family": "Inter", "color": "#888", "size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#ccc"},
            "bar": {"color": accent},
            "bgcolor": "#f7f7f7",
            "bordercolor": "#ebebeb",
            "steps": [
                {"range": [0,  40], "color": "#f7f7f7"},
                {"range": [40, 70], "color": "#f0f0f0"},
                {"range": [70,100], "color": "#e8e8e8"},
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor="white", font_family="Inter",
        height=280, margin=dict(l=20, r=20, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: EXPORT
# ─────────────────────────────────────────────
elif page == "Export":
    st.markdown("<p class='section-label'>Export</p>", unsafe_allow_html=True)
    st.markdown("## Export Predictions")
    st.markdown("Log predictions manually or download the full dataset.")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### Prediction Log")
    st.markdown("Go to **Predict**, set your inputs, then come back here to log the result.")

    if "prediction_log" not in st.session_state:
        st.session_state.prediction_log = []

    # Quick re-predict with last known slider state
    c1, c2 = st.columns(2)
    with c1:
        urgency      = st.slider("Urgency Score",      0.0, 10.0, 5.0, 0.1, key="exp_u")
        stakeholders = st.slider("Stakeholder Count",  1,   20,   10,       key="exp_s")
        data_avail   = st.slider("Data Availability",  0.0, 10.0, 5.0, 0.1, key="exp_d")
    with c2:
        pol_risk     = st.slider("Political Risk",      0.0, 10.0, 5.0, 0.1, key="exp_p")
        res_cost     = st.slider("Resource Cost",       0.0, 10.0, 5.0, 0.1, key="exp_r")

    input_arr  = np.array([[urgency, stakeholders, data_avail, pol_risk, res_cost]])
    pred       = clf.predict(input_arr)[0]
    prob       = clf.predict_proba(input_arr)[0]
    confidence = max(prob) * 100
    label      = "High Impact" if pred == 1 else "Low Impact"

    st.markdown(f"**Current prediction:** {label} &nbsp;|&nbsp; Confidence: {confidence:.1f}%")

    if st.button("Add to log"):
        st.session_state.prediction_log.append({
            "urgency_score":     urgency,
            "stakeholder_count": stakeholders,
            "data_availability": data_avail,
            "political_risk":    pol_risk,
            "resource_cost":     res_cost,
            "prediction":        label,
            "confidence_pct":    round(confidence, 1),
        })
        st.success("Added to log.")

    if st.session_state.prediction_log:
        log_df = pd.DataFrame(st.session_state.prediction_log)
        st.dataframe(log_df, use_container_width=True)
        st.download_button(
            label="Download log as CSV",
            data=log_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_log.csv",
            mime="text/csv"
        )
        if st.button("Clear log"):
            st.session_state.prediction_log = []
            st.rerun()
    else:
        st.info("No entries yet. Adjust the sliders and click 'Add to log'.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Full Dataset")
    full_df = df.copy()
    full_df["predicted_impact"] = clf.predict(full_df[feature_cols].values)
    full_df["predicted_impact"] = full_df["predicted_impact"].map({0: "Low Impact", 1: "High Impact"})
    full_df["actual_impact"]    = full_df["impact"].map({0: "Low Impact", 1: "High Impact"})
    st.download_button(
        label="Download full dataset as CSV",
        data=full_df.to_csv(index=False).encode("utf-8"),
        file_name="full_predictions.csv",
        mime="text/csv"
    )
