# streamlit_app/utils/ml_risk.py
"""
Pont entre le pipeline Smartovate et le modèle ML V2 (XGBoost, Config C,
calibration sigmoid, seuil 0.25) qui prédit le risque de REVISION_REQUISE
AVANT que le ReviewerAgent ne tourne.

------------------------------------------------------------------------
Rôle STRICT de ce module (volontairement étroit)
------------------------------------------------------------------------
- Construire les 12 features du modèle à partir de VALEURS RÉELLEMENT
  DISPONIBLES à ce stade du pipeline (texte de la tâche utilisateur +
  stats Planner + stats Coder accumulées jusqu'à ce tour).
- N'INVENTE JAMAIS une valeur manquante : si une donnée nécessaire n'est
  pas disponible ou invalide (ex: dénominateur nul pour planner_ratio,
  code non extractible du message du Coder), la fonction renvoie `None`
  plutôt que d'estimer/deviner quoi que ce soit.
- Ne lève JAMAIS d'exception vers l'appelant : toute erreur (modèle non
  chargé, feature manquante, bug interne) est capturée et journalisée
  (stderr), et la fonction renvoie `None`. Le pipeline (Planner -> Coder
  -> Reviewer -> Docker) doit continuer à fonctionner à l'identique même
  si le ML échoue intégralement.
- Ne fait AUCUN appel réseau (pas d'Azure OpenAI) : uniquement de
  l'inférence locale sur le modèle déjà entraîné (`revision_predictor_v2.joblib`).

Ce module ne modifie ni ne remplace le ReviewerAgent : il produit
uniquement un signal informatif, consommé par `orchestrator.py` pour
affichage. Le Reviewer continue de tourner et de décider normalement,
quel que soit le résultat (ou l'absence de résultat) de ce module.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# `predict_new_task_v2.py` vit dans ml/ à la racine du repo (livré avec le
# modèle final -- voir FINAL_ML_REPORT.md). On importe paresseusement pour
# ne jamais faire planter l'import de ce module si le modèle n'est pas
# encore déployé dans un environnement donné (ex: CI sans les fichiers
# modèle) -- dans ce cas, `ML_AVAILABLE` reste False et toute prédiction
# renvoie None proprement.
try:
    from ml.predict_new_task_v2 import predict_revision_risk as _predict_revision_risk
    # `_threshold` est la même constante (0.25) déjà utilisée par
    # predict_revision_risk() pour la décision APPROVED/REVISION_REQUISE --
    # on l'importe uniquement pour l'AFFICHER dans la carte UI (voir
    # compute_risk_event ci-dessous), jamais pour la recalculer ou la
    # modifier.
    from ml.predict_new_task_v2 import _threshold as _MODEL_THRESHOLD
    ML_AVAILABLE = True
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - dépend de l'environnement de déploiement
    _predict_revision_risk = None
    _MODEL_THRESHOLD = None
    ML_AVAILABLE = False
    _IMPORT_ERROR = exc


# Import de l'extraction de code déjà validée dans teams/code_review_team.py
# -- on réutilise cette fonction plutôt que de réimplémenter une logique
# d'extraction dupliquée (et potentiellement incohérente) ici.
try:
    from teams.code_review_team import extract_last_code_block as _extract_last_code_block
except Exception:  # pragma: no cover
    _extract_last_code_block = None


@dataclass
class MLRiskEvent:
    """
    Événement de risque ML, façonné pour ressembler (duck-typing) à un
    message d'agent AutoGen standard : `source`, `content`, `models_usage`.

    Pourquoi cette forme précise : le pipeline existant (`orchestrator.py`,
    via `stream.py`) traite déjà tout élément non-`TaskResult` reçu du
    flux comme un "message" générique, en lisant `.source` / `.content` /
    `.models_usage` (voir backend.py, docstring de `PipelineEvent`, et la
    boucle de `orchestrator.py`). En adoptant exactement cette forme, ce
    nouvel événement traverse `stream.py` SANS aucune modification de ce
    fichier, et sans hypothèse supplémentaire sur son fonctionnement
    interne -- seule la boucle de consommation dans `orchestrator.py` a
    besoin d'une branche `elif source == "MLRiskPredictor":` en plus.

    `models_usage` vaut toujours `None` : ce module ne fait aucun appel
    Azure OpenAI, donc aucun token à comptabiliser.
    """

    source: str = "MLRiskPredictor"
    content: dict[str, Any] | None = None
    models_usage: None = None


def _log_ml_error(context: str, exc: Exception) -> None:
    """
    Journalisation best-effort d'une erreur ML, sur stderr uniquement --
    jamais propagée à l'appelant (voir docstring du module).
    """
    print(f"[ml_risk] {context} : {exc!r}", file=sys.stderr)
    traceback.print_exc()


def compute_risk_event(
    *,
    task: str,
    coder_message_content: str,
    planner_execution_time: float | None,
    planner_prompt_tokens: int | None,
    planner_completion_tokens: int | None,
    coder_prompt_tokens_accumulated: int | None,
    coder_completion_tokens_accumulated: int | None,
) -> MLRiskEvent | None:
    """
    Calcule (best-effort) le risque de révision pour le tour de Coder qui
    vient de se terminer, à partir des VRAIES statistiques accumulées
    jusqu'à ce point du pipeline. Ne fabrique jamais de valeur manquante.

    Args:
        task: le prompt utilisateur ORIGINAL (pas le plan du Planner --
            c'est ce champ `task` qui correspond à la colonne `task` du
            modèle entraîné, voir execution_logger.ExecutionMetrics.task).
        coder_message_content: le contenu brut du dernier message du
            CodeurAgent (utilisé pour en extraire le code produit).
        planner_execution_time: `metrics.planner.execution_time` réel.
        planner_prompt_tokens / planner_completion_tokens: tokens RÉELS du
            Planner pour cette exécution (`models_usage`).
        coder_prompt_tokens_accumulated / coder_completion_tokens_accumulated:
            somme RÉELLE des tokens Coder accumulés jusqu'à CE tour inclus
            (pas une estimation -- calculée par l'appelant à partir des
            `models_usage` réels de chaque message Coder déjà reçu).

    Returns:
        Un `MLRiskEvent` avec la prédiction dans `.content`, ou `None` si
        la prédiction n'a pas pu être calculée de façon fiable (donnée
        manquante, modèle indisponible, erreur interne) -- dans tous les
        cas, sans jamais lever d'exception.
    """
    # --- [ML DEBUG] diagnostic temporaire (aucune logique modifiée) ---
    print("[ML DEBUG] compute_risk_event called", file=sys.stderr)
    print(f"[ML DEBUG] ML_AVAILABLE={ML_AVAILABLE!r} import_error={_IMPORT_ERROR!r}", file=sys.stderr)
    print(f"[ML DEBUG] planner_execution_time={planner_execution_time!r}", file=sys.stderr)
    print(f"[ML DEBUG] planner_prompt_tokens={planner_prompt_tokens!r}", file=sys.stderr)
    print(f"[ML DEBUG] planner_completion_tokens={planner_completion_tokens!r}", file=sys.stderr)
    print(f"[ML DEBUG] coder_prompt_tokens_accumulated={coder_prompt_tokens_accumulated!r}", file=sys.stderr)
    print(f"[ML DEBUG] coder_completion_tokens_accumulated={coder_completion_tokens_accumulated!r}", file=sys.stderr)
    # --- fin [ML DEBUG] ---

    if not ML_AVAILABLE:
        # Modèle non chargé (ex: fichiers .joblib absents de ce déploiement).
        # Journalisé une seule fois au niveau de l'import, on ne spam pas
        # les logs à chaque tour.
        print("[ML DEBUG] risk_event=NONE (raison: ML_AVAILABLE=False)", file=sys.stderr)
        return None

    try:
        if _extract_last_code_block is None:
            _log_ml_error("extract_last_code_block indisponible", RuntimeError("import échoué"))
            print("[ML DEBUG] risk_event=NONE (raison: _extract_last_code_block indisponible)", file=sys.stderr)
            return None

        code = _extract_last_code_block(coder_message_content)
        print(f"[ML DEBUG] code_length={len(code) if code else 0}", file=sys.stderr)
        if not code:
            # Pas de bloc de code extractible dans ce message -- on ne
            # fabrique PAS de code_length/code_line_count à partir de rien.
            print("[ML DEBUG] risk_event=NONE (raison: aucun bloc de code extractible)", file=sys.stderr)
            return None

        # --- Features requises, disponibilité stricte ---
        required = {
            "planner_execution_time": planner_execution_time,
            "planner_prompt_tokens": planner_prompt_tokens,
            "planner_completion_tokens": planner_completion_tokens,
            "coder_prompt_tokens": coder_prompt_tokens_accumulated,
            "coder_completion_tokens": coder_completion_tokens_accumulated,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            # Données réellement indisponibles à ce tour (ex: aucun
            # `models_usage` fourni par AutoGen pour cet appel) -- on ne
            # devine rien, on saute simplement cette prédiction.
            print(f"[ML DEBUG] risk_event=NONE (raison: champs manquants={missing!r})", file=sys.stderr)
            return None

        planner_total_tokens = planner_prompt_tokens + planner_completion_tokens
        coder_total_tokens = coder_prompt_tokens_accumulated + coder_completion_tokens_accumulated

        denominator = planner_total_tokens + coder_total_tokens
        if denominator <= 0:
            # planner_ratio non calculable sans dénominateur valide --
            # consigne explicite : ne calculer QUE si le dénominateur est
            # valide.
            print(
                f"[ML DEBUG] risk_event=NONE (raison: denominator<=0, "
                f"planner_total_tokens={planner_total_tokens!r}, coder_total_tokens={coder_total_tokens!r})",
                file=sys.stderr,
            )
            return None
        planner_ratio = planner_total_tokens / denominator

        execution_features = {
            "code_length": len(code),
            "code_line_count": code.count("\n") + 1,
            "planner_ratio": planner_ratio,
            "planner_execution_time": planner_execution_time,
            "planner_completion_tokens": planner_completion_tokens,
            "coder_total_tokens": coder_total_tokens,
            "coder_prompt_tokens": coder_prompt_tokens_accumulated,
            "coder_completion_tokens": coder_completion_tokens_accumulated,
        }

        result = _predict_revision_risk(task, execution_features)

        print(
            f"[ML DEBUG] risk_event=CREATED risk_level={result.get('risk_level')!r} "
            f"probability={result.get('probability')!r}",
            file=sys.stderr,
        )

        return MLRiskEvent(content={
            "task": result["task"],
            "probability": result["probability"],
            "risk_level": result["risk_level"],
            "prediction": result["prediction"],
            "prediction_label": "REVISION_REQUISE" if result["prediction"] == 1 else "APPROVED",
            # --- Transport uniquement (aucune valeur recalculée) : mêmes
            # variables que celles déjà utilisées ci-dessus pour construire
            # `execution_features`, plus le seuil de décision importé tel
            # quel depuis predict_new_task_v2.py -- destinées uniquement à
            # l'affichage dans agent_cards.render_ml_risk_card().
            "code_length": execution_features["code_length"],
            "code_line_count": execution_features["code_line_count"],
            "planner_execution_time": planner_execution_time,
            "planner_prompt_tokens": planner_prompt_tokens,
            "planner_completion_tokens": planner_completion_tokens,
            "coder_prompt_tokens": coder_prompt_tokens_accumulated,
            "coder_completion_tokens": coder_completion_tokens_accumulated,
            "coder_total_tokens": coder_total_tokens,
            "threshold": _MODEL_THRESHOLD,
        })

    except Exception as exc:
        # Filet de sécurité absolu : quelle que soit l'erreur (modèle
        # corrompu, feature inattendue, etc.), le pipeline métier ne doit
        # jamais être impacté par un échec du ML.
        _log_ml_error("échec de compute_risk_event", exc)
        print(f"[ML DEBUG] risk_event=NONE (raison: exception {exc!r})", file=sys.stderr)
        return None