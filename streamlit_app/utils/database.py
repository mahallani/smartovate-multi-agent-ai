# streamlit_app/utils/database.py
"""
Couche d'accès SQLite pour l'historique des exécutions de Smartovate.

Ce module ne contient AUCUNE logique métier, AUCUNE logique Multi-Agent,
AUCUNE logique Docker et AUCUNE logique Data Science. Sa seule
responsabilité est de savoir où se trouve la base SQLite, de créer le
schéma s'il n'existe pas encore, et d'exposer une connexion. C'est
`execution_logger.py` qui sait QUOI écrire ; ce module sait uniquement
COMMENT/OÙ se connecter.

Emplacement de la base :
    <racine_du_projet>/data/smartovate.db

La racine du projet est déduite de l'emplacement de ce fichier
(streamlit_app/utils/database.py -> remonte de 2 niveaux), ce qui évite
de dépendre du répertoire de travail courant (important car Streamlit et
les tests peuvent être lancés depuis des dossiers différents).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# streamlit_app/utils/database.py -> parents[0]=utils, [1]=streamlit_app, [2]=racine projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "smartovate.db"

# Schéma de la table `executions`.
#
# Une exécution complète du pipeline (Planner -> Codeur/Reviewer -> Docker)
# correspond à UNE seule ligne (voir execution_logger.py).
#
# Convention pour les colonnes de tokens : quand l'API Azure OpenAI (via
# AutoGen) fournit un `RequestUsage(prompt_tokens, completion_tokens)` pour
# un appel, on stocke les deux valeurs telles quelles ; `*_total_tokens`
# est leur somme arithmétique (jamais recalculée à partir de caractères).
# Quand aucun appel n'a eu lieu ou que l'information n'était pas
# disponible, la valeur est NULL.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS executions (
    execution_id            TEXT PRIMARY KEY,
    timestamp                TEXT NOT NULL,
    task                     TEXT,

    planner_execution_time   REAL,
    planner_prompt_tokens    INTEGER,
    planner_completion_tokens INTEGER,
    planner_total_tokens     INTEGER,

    coder_execution_time     REAL,
    coder_prompt_tokens      INTEGER,
    coder_completion_tokens  INTEGER,
    coder_total_tokens       INTEGER,

    reviewer_execution_time  REAL,
    reviewer_prompt_tokens   INTEGER,
    reviewer_completion_tokens INTEGER,
    reviewer_total_tokens    INTEGER,
    reviewer_status          TEXT,

    total_execution_time     REAL,
    total_tokens              INTEGER,
    revision_count            INTEGER,

    code_length               INTEGER,
    code_line_count            INTEGER,

    docker_execution_time     REAL,
    docker_status              TEXT,
    docker_exit_code           INTEGER
);
"""


def get_db_path(db_path: str | Path | None = None) -> Path:
    """Retourne le chemin de la base SQLite (créant le dossier parent si besoin)."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """
    Ouvre (et crée si besoin) la connexion SQLite, avec le schéma déjà
    initialisé.

    Chaque appel ouvre une nouvelle connexion courte durée (pattern adapté
    à SQLite + Streamlit : pas de connexion partagée à long terme entre
    threads/reruns).
    """
    path = get_db_path(db_path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Crée la table `executions` si elle n'existe pas déjà (idempotent)."""
    conn.execute(SCHEMA_SQL)
    conn.commit()
