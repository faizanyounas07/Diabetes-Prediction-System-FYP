import numpy as np
import pandas as pd
import joblib, os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42),
    "SVM":                 SVC(probability=True, kernel='rbf', random_state=42),
}


def train_all(X_train, y_train) -> dict:
    trained = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def evaluate_all(trained: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, model in trained.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall":    round(recall_score(y_test, y_pred), 4),
            "F1-Score":  round(f1_score(y_test, y_pred), 4),
            "ROC-AUC":   round(roc_auc_score(y_test, y_prob), 4),
        })
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)


def get_confusion(model, X_test, y_test) -> np.ndarray:
    return confusion_matrix(y_test, model.predict(X_test))


def get_roc(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    return fpr, tpr, auc


def get_feature_importance(model, feature_names: list) -> pd.DataFrame | None:
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        imp = np.abs(model.coef_[0])
    else:
        return None
    return pd.DataFrame({"Feature": feature_names, "Importance": imp}).sort_values(
        "Importance", ascending=False
    ).reset_index(drop=True)


def get_crossval(model, X_train, y_train, cv: int = 5) -> dict:
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    return {"mean": round(scores.mean(), 4), "std": round(scores.std(), 4), "scores": scores.tolist()}


def save_model(model, scaler, path: str = "models/best_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)


def load_model(path: str = "models/best_model.pkl"):
    obj = joblib.load(path)
    return obj["model"], obj["scaler"]


def predict_single(model, scaler, input_values: list) -> dict:
    arr = np.array(input_values).reshape(1, -1)
    arr_sc = scaler.transform(arr)
    pred = model.predict(arr_sc)[0]
    prob = model.predict_proba(arr_sc)[0]
    return {
        "prediction": int(pred),
        "label": "Diabetic" if pred == 1 else "Non-Diabetic",
        "probability_diabetic": round(float(prob[1]) * 100, 2),
        "probability_non_diabetic": round(float(prob[0]) * 100, 2),
    }
