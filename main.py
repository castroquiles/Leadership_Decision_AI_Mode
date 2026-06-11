"""
Leadership Decision AI Model
main.py — ML skeleton with synthetic data demo

Demonstrates:
  - Supervised model: classify a decision as High/Low impact
  - Unsupervised model: find patterns/clusters in decision data
  - Predict on new data
  - Print insights summary

Replace synthetic data with real data when available.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# Replace this section with real data loading later.
# Example: pd.read_csv("decisions.csv")
# ─────────────────────────────────────────────

def generate_synthetic_data(n_samples=200, random_state=42):
    """
    Generates fake leadership decision data.

    Features (columns):
      - urgency_score     : how time-sensitive the decision is (0–10)
      - stakeholder_count : number of stakeholders involved (1–20)
      - data_availability : how much data exists to support the decision (0–10)
      - political_risk    : estimated political friction (0–10)
      - resource_cost     : estimated cost/effort (0–10)

    Label:
      - impact: 1 = High Impact decision, 0 = Low Impact
    """
    rng = np.random.RandomState(random_state)

    urgency        = rng.uniform(0, 10, n_samples)
    stakeholders   = rng.randint(1, 20, n_samples).astype(float)
    data_avail     = rng.uniform(0, 10, n_samples)
    political_risk = rng.uniform(0, 10, n_samples)
    resource_cost  = rng.uniform(0, 10, n_samples)

    # Simple rule: high impact if urgency + data_availability is high
    # and political_risk is low. This is intentionally naive for demo purposes.
    impact = (
        (urgency + data_avail - political_risk - resource_cost * 0.5) > 8
    ).astype(int)

    X = np.column_stack([urgency, stakeholders, data_avail, political_risk, resource_cost])
    y = impact
    feature_names = [
        "urgency_score",
        "stakeholder_count",
        "data_availability",
        "political_risk",
        "resource_cost",
    ]
    return X, y, feature_names


# ─────────────────────────────────────────────
# 2. SUPERVISED MODEL — train and evaluate
# ─────────────────────────────────────────────

def train_model(X, y):
    """
    Trains a Random Forest classifier to predict decision impact.
    Returns the trained model and the test split for evaluation.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Prints a classification report for the supervised model."""
    y_pred = model.predict(X_test)
    print("\n── Supervised Model: Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=["Low Impact", "High Impact"]))


# ─────────────────────────────────────────────
# 3. UNSUPERVISED MODEL — find patterns
# ─────────────────────────────────────────────

def train_unsupervised_model(X, n_clusters=3):
    """
    Uses KMeans clustering to find natural groupings in decision data.
    n_clusters=3 gives: cautious / moderate / bold decision profiles.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    return kmeans, scaler


def find_patterns(kmeans, scaler, X, feature_names):
    """Prints the cluster centers as human-readable decision profiles."""
    print("\n── Unsupervised Model: Decision Profiles (Clusters) ──")
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    labels = kmeans.labels_
    for i, center in enumerate(centers):
        count = np.sum(labels == i)
        print(f"\n  Profile {i + 1}  ({count} decisions)")
        for name, val in zip(feature_names, center):
            print(f"    {name:<22}: {val:.2f}")


# ─────────────────────────────────────────────
# 4. PREDICT ON NEW DATA
# ─────────────────────────────────────────────

def predict(model, new_data, feature_names):
    """
    Predicts impact for new decision data.

    new_data: list of lists, each inner list is one decision with values for:
      [urgency_score, stakeholder_count, data_availability, political_risk, resource_cost]
    """
    new_data = np.array(new_data)
    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)

    print("\n── Predictions on New Decisions ──")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        label = "High Impact" if pred == 1 else "Low Impact"
        confidence = max(prob) * 100
        print(f"\n  Decision {i + 1}:")
        for name, val in zip(feature_names, new_data[i]):
            print(f"    {name:<22}: {val}")
        print(f"    → Predicted: {label}  (confidence: {confidence:.1f}%)")


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE — challenge assumptions
# ─────────────────────────────────────────────

def show_feature_importance(model, feature_names):
    """Shows which factors drive impact predictions most."""
    print("\n── Feature Importance (What drives impact?) ──")
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for name, score in ranked:
        bar = "█" * int(score * 40)
        print(f"  {name:<22}: {bar} {score:.3f}")


# ─────────────────────────────────────────────
# 6. MAIN — run the full demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Leadership Decision AI Model — Demo Run")
    print("=" * 55)

    # Generate synthetic data (replace with real data later)
    X, y, feature_names = generate_synthetic_data(n_samples=300)

    # Train supervised model
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)

    # Show what drives decisions
    show_feature_importance(model, feature_names)

    # Find patterns with unsupervised model
    kmeans, scaler = train_unsupervised_model(X, n_clusters=3)
    find_patterns(kmeans, scaler, X, feature_names)

    # Predict on two hypothetical new decisions
    new_decisions = [
        [9.0, 15, 8.5, 2.0, 3.0],   # high urgency, lots of data, low risk
        [2.0,  3, 1.0, 9.0, 8.5],   # low urgency, little data, high risk
    ]
    predict(model, new_decisions, feature_names)

    print("\n" + "=" * 55)
    print("  Done. Replace synthetic data with real data to begin.")
    print("=" * 55)
