"""
Smartovate - Prédiction du risque de REVISION_REQUISE
========================================================
Entraîne et compare 4 modèles de classification (LogReg, RandomForest,
XGBoost, LightGBM) pour prédire si une tâche va nécessiter une révision
du Reviewer, AVANT que le Reviewer ne se prononce.

Usage: python3 train_revision_model.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, make_scorer
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. CHARGEMENT DES DONNEES
# ----------------------------------------------------------------------
df = pd.read_csv("executions_final_corrected.csv")

with open("ml_features.txt") as f:
    raw_lines = [l.strip() for l in f.readlines()]

FEATURES = []
TARGET = "revision_required"
for l in raw_lines:
    if not l:
        continue
    if l.startswith("TARGET"):
        continue
    if l.startswith("DO NOT USE"):
        break  # stop parsing - everything after this is the forbidden list
    FEATURES.append(l)

print(f"Dataset: {df.shape[0]} lignes, {len(FEATURES)} features ML")
print(f"Target distribution: {df[TARGET].value_counts().to_dict()}")

X = df[FEATURES].copy()
y = df[TARGET].copy()

# ----------------------------------------------------------------------
# 2. TRAIN / TEST SPLIT (80/20 stratifié) + CV setup
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain: {X_train.shape[0]} lignes (classe1={y_train.sum()}) | "
      f"Test: {X_test.shape[0]} lignes (classe1={y_test.sum()})")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ----------------------------------------------------------------------
# 3. DEFINITION DES MODELES (pipelines - scaler uniquement pour LogReg)
# ----------------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                    random_state=RANDOM_STATE))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss", random_state=RANDOM_STATE
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        num_leaves=15, min_child_samples=5,
        class_weight="balanced", random_state=RANDOM_STATE, verbose=-1
    ),
}

# ----------------------------------------------------------------------
# 4. CROSS-VALIDATION SUR LE TRAIN
# ----------------------------------------------------------------------
scoring = {
    "accuracy": "accuracy",
    "f1": make_scorer(f1_score, pos_label=1),
    "recall": make_scorer(recall_score, pos_label=1),
    "roc_auc": "roc_auc",
}

cv_results = {}
for name, model in models.items():
    res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    cv_results[name] = {
        "CV Accuracy": (res["test_accuracy"].mean(), res["test_accuracy"].std()),
        "CV F1 (classe1)": (res["test_f1"].mean(), res["test_f1"].std()),
        "CV Recall (classe1)": (res["test_recall"].mean(), res["test_recall"].std()),
        "CV ROC-AUC": (res["test_roc_auc"].mean(), res["test_roc_auc"].std()),
    }

# ----------------------------------------------------------------------
# 5. ENTRAINEMENT SUR TOUT LE TRAIN + EVALUATION SUR LE TEST
# ----------------------------------------------------------------------
test_results = {}
confusions = {}
fitted_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    test_results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (classe1)": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall (classe1)": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1 (classe1)": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    confusions[name] = cm

# ----------------------------------------------------------------------
# 6. TABLEAUX DE COMPARAISON
# ----------------------------------------------------------------------
test_df = pd.DataFrame(test_results).T.round(3)
cv_df = pd.DataFrame({
    name: {k: f"{v[0]:.3f} +/- {v[1]:.3f}" for k, v in d.items()}
    for name, d in cv_results.items()
}).T

print("\n" + "=" * 70)
print("TABLEAU 1 - METRIQUES SUR LE TEST SET (20%)")
print("=" * 70)
print(test_df.to_string())

print("\n" + "=" * 70)
print("TABLEAU 2 - CROSS-VALIDATION (5-fold sur TRAIN)")
print("=" * 70)
print(cv_df.to_string())

print("\n" + "=" * 70)
print("MATRICES DE CONFUSION (Test set, labels: 0=APPROVED, 1=REVISION_REQUISE)")
print("=" * 70)
for name, cm in confusions.items():
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{name}:")
    print(f"           Pred APPROVED   Pred REVISION")
    print(f"Actual APPR      TN={tn:<3}         FP={fp:<3}")
    print(f"Actual REVIS     FN={fn:<3}         TP={tp:<3}")
    print(f"  -> FN (revisions ratees) = {fn} | FP (fausses alertes) = {fp}")

# Save tables
test_df.to_csv("models/test_metrics.csv")
cv_df.to_csv("models/cv_metrics.csv")

# ----------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (RF, XGB, LGBM)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
tree_models = ["Random Forest", "XGBoost", "LightGBM"]

feature_importance_data = {}

for ax, name in zip(axes, tree_models):
    model = fitted_models[name]
    importances = model.feature_importances_
    imp_series = pd.Series(importances, index=FEATURES).sort_values(ascending=False)
    top = imp_series.head(15)
    feature_importance_data[name] = imp_series.to_dict()

    ax.barh(top.index[::-1], top.values[::-1], color="#4C72B0")
    ax.set_title(f"{name} - Top 15 Feature Importances")
    ax.set_xlabel("Importance")

plt.tight_layout()
plt.savefig("figures/feature_importances.png", dpi=120, bbox_inches="tight")
print("\nFeature importance graphique sauvegarde: figures/feature_importances.png")

with open("models/feature_importances.json", "w") as f:
    json.dump(feature_importance_data, f, indent=2)

# ----------------------------------------------------------------------
# 8. CONFUSION MATRICES PLOT
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, (name, cm) in zip(axes, confusions.items()):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["APPROVED", "REVISION"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["APPROVED", "REVISION"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(name)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.tight_layout()
plt.savefig("figures/confusion_matrices.png", dpi=120, bbox_inches="tight")
print("Matrices de confusion sauvegardees: figures/confusion_matrices.png")

# ----------------------------------------------------------------------
# 9. SELECTION DU MEILLEUR MODELE (objectif metier: Recall/F1 classe1)
# ----------------------------------------------------------------------
# Score composite orienté métier : on privilégie recall classe1, puis F1, puis ROC-AUC
ranking = test_df.assign(
    business_score=0.5 * test_df["Recall (classe1)"] + 0.3 * test_df["F1 (classe1)"] + 0.2 * test_df["ROC-AUC"]
).sort_values("business_score", ascending=False)

print("\n" + "=" * 70)
print("CLASSEMENT (score metier = 0.5*Recall + 0.3*F1 + 0.2*ROC-AUC, classe REVISION_REQUISE)")
print("=" * 70)
print(ranking[["Recall (classe1)", "F1 (classe1)", "ROC-AUC", "business_score"]].round(3).to_string())

best_model_name = ranking.index[0]
best_model = fitted_models[best_model_name]
print(f"\n>>> MEILLEUR MODELE (critere metier): {best_model_name}")

# ----------------------------------------------------------------------
# 10. RE-ENTRAINEMENT DU MEILLEUR MODELE SUR TOUT LE TRAIN + SAUVEGARDE
# ----------------------------------------------------------------------
# (deja entraine sur X_train ci-dessus; on le garde tel quel car
#  X_train = 80% du dataset, conforme à l'étape 11 de la consigne)
joblib.dump(best_model, "models/revision_predictor.joblib")

with open("models/feature_names.json", "w") as f:
    json.dump(FEATURES, f, indent=2)

with open("models/best_model_info.json", "w") as f:
    json.dump({
        "best_model_name": best_model_name,
        "test_metrics": test_results[best_model_name],
        "cv_metrics": {k: f"{v[0]:.3f} +/- {v[1]:.3f}" for k, v in cv_results[best_model_name].items()},
    }, f, indent=2)

print(f"\nModele sauvegarde: models/revision_predictor.joblib")
print(f"Features sauvegardees: models/feature_names.json")

print("\n>>> PIPELINE TERMINE <<<")
