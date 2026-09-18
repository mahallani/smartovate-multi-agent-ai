"""
Smartovate - predict_revision_risk() v2
==========================================
Modele final : XGBoost (Config C - top 12 features) + calibration sigmoid
(Platt scaling), seuil de decision 0.25.

IMPORTANT - LIMITE HONNETE :
La Config C retenue utilise 8 features d'execution (planner_*, coder_*,
code_length, code_line_count) qui ne sont PAS calculables a partir du seul
texte de la tache : elles n'existent qu'APRES que le PlannerAgent et le
CodeurAgent aient tourne (mais avant le ReviewerAgent - donc toujours
"avant le verdict", conformement a l'objectif initial du projet).

Cette fonction ne fabrique donc JAMAIS ces valeurs : si elles ne sont pas
fournies, elle refuse de predire plutot que d'inventer un resultat.

Usage typique dans le pipeline Smartovate (future integration) :
    Planner tourne -> Coder tourne -> on a les stats d'execution ->
    predict_revision_risk(task, execution_features) -> avant Reviewer.

Usage: python3 predict_new_task_v2.py (lance les sanity checks des Etapes 11/12)
"""

import json
import re
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "v2" / "models"

MODEL_PATH = MODEL_DIR / "revision_predictor_v2.joblib"
NORM_PARAMS_PATH = MODEL_DIR / "text_feature_norm_params_v2.json"

_bundle = joblib.load(MODEL_PATH)
_model = _bundle["model"]
_calibrator = _bundle["calibrator"]
_threshold = _bundle["threshold"]
_features = _bundle["features"]  # ordre exact attendu par le modele

with open(NORM_PARAMS_PATH) as f:
    _norm = json.load(f)

# Features calculables a partir du SEUL texte de la tache (voir feature_engineering_v2.py)
TEXT_DERIVED_FEATURES = [
    "estimated_task_complexity", "task_complexity_score",
    "technical_keyword_count", "contains_api",
]
# Features necessitant les stats d'execution Planner+Coder (obligatoires en entree)
EXECUTION_REQUIRED_FEATURES = [f for f in _features if f not in TEXT_DERIVED_FEATURES]

TECH_KEYWORDS = [
    "api", "rest", "jwt", "postgresql", "sql", "base de données", "database",
    "csv", "etl", "pipeline", "crud", "multi-agent", "multi agent", "agent",
    "authentification", "authentication", "validation", "visualisation",
    "graphique", "docker",
]
REQUIREMENT_WORDS = [" et ", " avec ", " combinant "]
API_KEYWORDS = ["api", "rest"]  # approximation transparente de contains_api


def _count_tech_keywords(text: str) -> int:
    t = text.lower()
    return sum(t.count(k) for k in TECH_KEYWORDS)


def _count_requirements(text: str) -> int:
    t = text.lower()
    total_commas = t.count(",")
    commas_between_digits = len(re.findall(r"(?<=\d),\s*(?=\d)", t))
    real_commas = max(total_commas - commas_between_digits, 0)
    n_words = sum(t.count(w) for w in REQUIREMENT_WORDS)
    return real_commas + n_words + 1


def _norm_minmax(value, key):
    lo, hi = _norm[key]["min"], _norm[key]["max"]
    return max(0.0, min(1.0, (value - lo) / (hi - lo + 1e-9)))


def _compute_text_features(task: str) -> dict:
    word_count = len(task.split())
    kw_count = _count_tech_keywords(task)
    req_count = _count_requirements(task)

    score = (0.3 * _norm_minmax(word_count, "task_word_count")
             + 0.4 * _norm_minmax(kw_count, "technical_keyword_count")
             + 0.3 * _norm_minmax(req_count, "task_requirement_count"))

    bins = _norm["task_complexity_score_quartile_bins"]
    bucket = 1
    for i in range(1, len(bins)):
        if score <= bins[i] or i == len(bins) - 1:
            bucket = i
            break

    contains_api = int(any(k in task.lower() for k in API_KEYWORDS))

    return {
        "technical_keyword_count": kw_count,
        "task_complexity_score": round(score, 4),
        "estimated_task_complexity": bucket,
        "contains_api": contains_api,
    }


def predict_revision_risk(task: str, execution_features: dict) -> dict:
    """
    Predit le risque qu'une tache necessite une revision du Reviewer,
    a partir du texte de la tache ET des statistiques d'execution
    Planner+Coder deja disponibles a ce stade du pipeline (AVANT Reviewer).

    Parameters
    ----------
    task : str
        Texte brut de la tache.
    execution_features : dict
        Doit contenir les cles suivantes (calculees par Smartovate apres
        l'execution de PlannerAgent puis CodeurAgent, avant ReviewerAgent) :
        code_length, code_line_count, planner_ratio, planner_execution_time,
        planner_completion_tokens, coder_total_tokens, coder_prompt_tokens,
        coder_completion_tokens

    Returns
    -------
    dict avec "task", "probability", "risk_level", "prediction"

    Raises
    ------
    ValueError si une feature d'execution requise est manquante (aucune
    valeur n'est jamais inventee).
    """
    missing = [f for f in EXECUTION_REQUIRED_FEATURES if f not in execution_features]
    if missing:
        raise ValueError(
            f"Features d'execution manquantes: {missing}. "
            f"Cette fonction ne peut PAS predire a partir du seul texte de "
            f"la tache pour la configuration de modele actuelle (Config C) - "
            f"elle a besoin des stats reelles Planner+Coder, pas de valeurs "
            f"inventees."
        )

    text_feats = _compute_text_features(task)
    row = {**text_feats, **{k: execution_features[k] for k in EXECUTION_REQUIRED_FEATURES}}
    X_new = pd.DataFrame([{f: row[f] for f in _features}])

    raw_proba = _model.predict_proba(X_new)[0, 1]
    calibrated_proba = float(_calibrator.predict_proba([[raw_proba]])[0, 1])
    pred = int(calibrated_proba >= _threshold)

    # Niveaux LOW/MEDIUM/HIGH : bases directement sur la probabilite calibree,
    # sans aucune regle metier arbitraire. Le seuil de decision (0.25) separe
    # MEDIUM de HIGH ; un deuxieme repere (0.5x le seuil) separe LOW de MEDIUM.
    # Ce ne sont PAS des regles sur "JWT+API=HIGH" : uniquement la probabilite.
    if calibrated_proba < _threshold / 2:
        risk_level = "LOW"
    elif calibrated_proba < _threshold:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "task": task,
        "probability": round(calibrated_proba, 3),
        "risk_level": risk_level,
        "prediction": pred,  # 0 = APPROVED probable, 1 = REVISION_REQUISE probable
    }


if __name__ == "__main__":
    # Stats d'execution REELLES (moyennes observees par complexity_level dans
    # le dataset d'entrainement), utilisees pour les sanity checks (Etapes
    # 11/12) - dans un vrai appel Smartovate, ces valeurs viendront des
    # vraies executions Planner+Coder de la tache testee, pas d'une moyenne.
    EXEC_SIMPLE = {  # moyenne EASY
        "code_length": 134.7, "code_line_count": 62.3, "planner_ratio": 0.30,
        "planner_execution_time": 13.5, "planner_completion_tokens": 763.8,
        "coder_total_tokens": 6940.1, "coder_prompt_tokens": 4167.1, "coder_completion_tokens": 3661.4,
    }
    EXEC_MEDIUM = {  # moyenne MEDIUM
        "code_length": 134.0, "code_line_count": 80.0, "planner_ratio": 0.30,
        "planner_execution_time": 14.6, "planner_completion_tokens": 994.6,
        "coder_total_tokens": 7895.1, "coder_prompt_tokens": 5229.4, "coder_completion_tokens": 4026.7,
    }
    EXEC_COMPLEX = {  # moyenne COMPLEX
        "code_length": 290.1, "code_line_count": 142.4, "planner_ratio": 0.20,
        "planner_execution_time": 12.6, "planner_completion_tokens": 1324.7,
        "coder_total_tokens": 16021.8, "coder_prompt_tokens": 8614.4, "coder_completion_tokens": 7407.3,
    }
    EXEC_VERY_COMPLEX = {  # moyenne VERY_COMPLEX
        "code_length": 524.8, "code_line_count": 213.9, "planner_ratio": 0.20,
        "planner_execution_time": 11.9, "planner_completion_tokens": 1632.3,
        "coder_total_tokens": 25944.3, "coder_prompt_tokens": 13930.7, "coder_completion_tokens": 12298.2,
    }

    print("=" * 70)
    print("ETAPE 11 - Sanity checks")
    print("=" * 70)
    cases = [
        ("Créer une fonction Python qui calcule la moyenne de trois nombres.", EXEC_SIMPLE),
        ("Créer une fonction qui trie une liste.", EXEC_SIMPLE),
        ("Créer une API REST avec authentification JWT et PostgreSQL.", EXEC_MEDIUM),
        ("Créer un système multi-agent avec authentification, PostgreSQL, ETL, CRUD et Docker.", EXEC_VERY_COMPLEX),
        ("Créer un pipeline ETL complet avec validation et visualisation.", EXEC_COMPLEX),
    ]
    for task, exec_feats in cases:
        result = predict_revision_risk(task, exec_feats)
        print(result)

    print("\n" + "=" * 70)
    print("ETAPE 12 - Tests de coherence / monotonie")
    print("=" * 70)
    monotonic_cases = [
        ("A", "Créer une fonction Python qui calcule une moyenne.", EXEC_SIMPLE),
        ("B", "Créer une API REST avec PostgreSQL.", EXEC_MEDIUM),
        ("C", "Créer une API REST avec JWT, PostgreSQL, CRUD et validation.", EXEC_COMPLEX),
        ("D", "Créer un système multi-agent avec API JWT, PostgreSQL, CRUD, ETL, Docker et visualisation.", EXEC_VERY_COMPLEX),
    ]
    for label, task, exec_feats in monotonic_cases:
        result = predict_revision_risk(task, exec_feats)
        print(f"[{label}] {result}")
