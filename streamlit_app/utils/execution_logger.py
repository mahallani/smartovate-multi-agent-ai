# streamlit_app/utils/execution_logger.py
"""
ExecutionLogger -- historique réel des exécutions du pipeline Smartovate.

Étape 1 de la fonctionnalité Data Science (voir consigne projet) :

    EXÉCUTION SMARTOVATE -> COLLECTE DES MÉTRIQUES -> STOCKAGE -> DATASET

Ce module NE CONTIENT AUCUNE LOGIQUE MACHINE LEARNING. Sa seule
responsabilité, volontairement étroite, est :

    COLLECTER  : recevoir les métriques déjà mesurées par l'appelant
                 (le pipeline, dans streamlit_app/utils/backend.py), sous
                 la forme d'un objet `ExecutionMetrics`.
    NORMALISER : mettre ces valeurs dans un format stable pour SQLite
                 (types corrects, arrondis, None explicite pour toute
                 métrique indisponible).
    SAUVEGARDER: insérer UNE ligne dans la table `executions` (voir
                 database.py pour le schéma).

Ce module ne mesure JAMAIS lui-même un temps d'exécution, ne compte
jamais lui-même des tokens, et n'invente aucune valeur : c'est
`executer_pipeline_complet()` (backend.py), qui a une vue complète et
réelle du pipeline en cours d'exécution, qui mesure et transmet ces
métriques. Voir la docstring de `ExecutionMetrics` pour le détail de
chaque champ et de sa disponibilité réelle.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from streamlit_app.utils.database import get_connection


def new_execution_id() -> str:
    """Identifiant unique d'exécution (UUID4)."""
    return uuid.uuid4().hex


def now_iso() -> str:
    """Horodatage UTC au format ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentUsage:
    """
    Mesures pour UN agent (Planner, Codeur ou Reviewer) sur une exécution
    complète (potentiellement plusieurs appels cumulés pour Codeur/Reviewer
    en cas de révisions).

    `prompt_tokens` / `completion_tokens` doivent provenir directement du
    `RequestUsage` renvoyé par AutoGen/Azure OpenAI (`message.models_usage`)
    -- jamais recalculés à partir du nombre de caractères. Si cette
    information n'était pas disponible pour un appel donné (ex: message
    sans `models_usage`, agent jamais invoqué), laisser `None`.
    """

    execution_time: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        """Somme prompt+completion, ou None si l'une des deux manque."""
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ExecutionMetrics:
    """
    Ensemble des métriques réellement disponibles pour UNE exécution
    complète du pipeline (Planner -> Codeur/Reviewer -> Docker).

    Chaque champ correspond à une colonne de la table `executions`
    (voir database.py). Un champ resté à `None` signifie explicitement
    "non disponible pour cette exécution" (ex: Docker jamais atteint
    car le code n'a pas été approuvé) -- jamais une valeur inventée.

    task: le prompt utilisateur initial tel que transmis au Planner.
        On ne stocke PAS l'historique complet de la conversation
        Codeur<->Reviewer (uniquement ce prompt), conformément à la
        consigne de ne pas sauvegarder plus que nécessaire.
    """

    execution_id: str = field(default_factory=new_execution_id)
    timestamp: str = field(default_factory=now_iso)
    task: str | None = None

    planner: AgentUsage = field(default_factory=AgentUsage)
    coder: AgentUsage = field(default_factory=AgentUsage)
    reviewer: AgentUsage = field(default_factory=AgentUsage)

    reviewer_status: str | None = None  # "APPROVED" | "REVISION_REQUISE" | None

    total_execution_time: float | None = None
    revision_count: int | None = None

    code_length: int | None = None       # nombre de caractères
    code_line_count: int | None = None   # nombre de lignes

    docker_execution_time: float | None = None
    docker_status: str | None = None     # "SUCCESS" | "ERROR" | "SKIPPED" | None
    docker_exit_code: int | None = None


def _round_or_none(value: float | None, ndigits: int = 4) -> float | None:
    return round(value, ndigits) if value is not None else None


def _normalize(metrics: ExecutionMetrics) -> dict:
    """
    Transforme un `ExecutionMetrics` en dict prêt pour l'insertion SQLite :
    - arrondit les durées (secondes) à 4 décimales pour un stockage stable,
    - calcule les totaux dérivés par simple arithmétique (jamais une
      nouvelle mesure),
    - conserve explicitement `None` -> NULL SQLite pour tout ce qui est
      indisponible.
    """
    planner_total = metrics.planner.total_tokens
    coder_total = metrics.coder.total_tokens
    reviewer_total = metrics.reviewer.total_tokens

    # Total tokens pipeline : somme des totaux disponibles. Si AUCUN total
    # n'est disponible, on n'invente pas 0 -- on laisse None.
    available_totals = [
        t for t in (planner_total, coder_total, reviewer_total) if t is not None
    ]
    pipeline_total_tokens = sum(available_totals) if available_totals else None

    return {
        "execution_id": metrics.execution_id,
        "timestamp": metrics.timestamp,
        "task": metrics.task,

        "planner_execution_time": _round_or_none(metrics.planner.execution_time),
        "planner_prompt_tokens": metrics.planner.prompt_tokens,
        "planner_completion_tokens": metrics.planner.completion_tokens,
        "planner_total_tokens": planner_total,

        "coder_execution_time": _round_or_none(metrics.coder.execution_time),
        "coder_prompt_tokens": metrics.coder.prompt_tokens,
        "coder_completion_tokens": metrics.coder.completion_tokens,
        "coder_total_tokens": coder_total,

        "reviewer_execution_time": _round_or_none(metrics.reviewer.execution_time),
        "reviewer_prompt_tokens": metrics.reviewer.prompt_tokens,
        "reviewer_completion_tokens": metrics.reviewer.completion_tokens,
        "reviewer_total_tokens": reviewer_total,
        "reviewer_status": metrics.reviewer_status,

        "total_execution_time": _round_or_none(metrics.total_execution_time),
        "total_tokens": pipeline_total_tokens,
        "revision_count": metrics.revision_count,

        "code_length": metrics.code_length,
        "code_line_count": metrics.code_line_count,

        "docker_execution_time": _round_or_none(metrics.docker_execution_time),
        "docker_status": metrics.docker_status,
        "docker_exit_code": metrics.docker_exit_code,
    }


class ExecutionLogger:
    """
    Composant de persistence pour l'historique des exécutions.

    Volontairement indépendant du reste du pipeline : ne connaît rien
    d'AutoGen, d'Azure OpenAI ou de Docker -- uniquement `ExecutionMetrics`
    (un objet simple) et SQLite (via `database.py`).
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = db_path

    def log_execution(self, metrics: ExecutionMetrics) -> None:
        """
        Enregistre UNE ligne dans `executions` pour cette exécution
        complète. Ne lève pas d'exception vers l'appelant en cas d'échec
        d'écriture (le logging ne doit jamais faire planter le pipeline
        métier) -- l'erreur est propagée uniquement via un `RuntimeError`
        explicite si l'appelant souhaite la traiter.
        """
        row = _normalize(metrics)
        conn: sqlite3.Connection = get_connection(self._db_path)
        try:
            columns = ", ".join(row.keys())
            placeholders = ", ".join(f":{k}" for k in row.keys())
            conn.execute(
                f"INSERT INTO executions ({columns}) VALUES ({placeholders})",
                row,
            )
            conn.commit()
        finally:
            conn.close()

    def fetch_all(self) -> list[dict]:
        """Utilitaire de vérification manuelle (tests, debug) : renvoie toutes les lignes."""
        conn = get_connection(self._db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM executions ORDER BY timestamp")
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()
