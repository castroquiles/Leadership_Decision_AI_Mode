"""
Leadership Decision AI Model
main.py — ML skeleton with synthetic data demo
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


def generate_synthetic_data(n_samples=200, random_state=42):
    rng = np.random.RandomState(random_state)
    urgency        = rng.uniform(0, 10, n_samples)
    stakeholders   = rng.randint(1, 20, n_samples).astype(float)
    data_avail     = rng.uniform(0, 10, n_samples)
    political_risk = rng.uniform(0, 10, n_samples)
    resource_cost  = rng.uniform(0, 10, n_samples)
    impact = ((urgency + data_avail - political_risk - resource_cost * 0.5) > 8).astype(int)
    X = np.column_stack([urgency, stakeholders, data_avail, political_risk, resource_cost])
    feature_names = ["urgency_score","stakeholder_count","data_availability","political_risk","resource_cost"]
    return X, impact, feature_names


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\n── Supervised Model: Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=["Low Impact", "High Impact"]))


def train_unsupervised_model(X, n_clusters=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    return kmeans, scaler


def find_patterns(kmeans, scaler, X, feature_names):
    print("\n── Unsupervised Model: Decision Profiles (Clusters) ──")
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    labels = kmeans.labels_
    for i, center in enumerate(centers):
        count = np.sum(labels == i)
        print(f"\n  Profile {i + 1}  ({count} decisions)")
        for name, val in zip(feature_names, center):
            print(f"    {name:<22}: {val:.2f}")


def predict(model, new_data, feature_names):
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


def show_feature_importance(model, feature_names):
    print("\n── Feature Importance (What drives impact?) ──")
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for name, score in ranked:
        bar = "█" * int(score * 40)
        print(f"  {name:<22}: {bar} {score:.3f}")


if __name__ == "__main__":
    print("=" * 55)
    print("  Leadership Decision AI Model — Demo Run")
    print("=" * 55)

    X, y, feature_names = generate_synthetic_data(n_samples=300)
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)
    show_feature_importance(model, feature_names)
    kmeans, scaler = train_unsupervised_model(X, n_clusters=3)
    find_patterns(kmeans, scaler, X, feature_names)

    new_decisions = [
        [9.0, 15, 8.5, 2.0, 3.0],
        [2.0,  3, 1.0, 9.0, 8.5],
    ]
    predict(model, new_decisions, feature_names)

    print("\n" + "=" * 55)
    print("  Done. Replace synthetic data with real data to begin.")
    print("=" * 55)
