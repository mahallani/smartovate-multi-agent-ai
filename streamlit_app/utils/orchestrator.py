# streamlit_app/utils/orchestrator.py
"""
Orchestre un tour complet de conversation (Planner -> Codeur/Reviewer en
streaming -> Docker), en s'appuyant UNIQUEMENT sur utils/backend.py.

Ce module ne contient aucune logique métier : il coordonne l'affichage
progressif (placeholders Streamlit) pendant que utils/stream.py fait tourner
le backend en arrière-plan. C'est le seul fichier qui orchestre les 3 étapes
dans l'ordre -- les composants (components/*.py) ne font qu'afficher l'état
courant de la session, ils ne déclenchent jamais eux-mêmes le backend.
"""
from __future__ import annotations

import asyncio
import time

import streamlit as st

from utils import backend, state, stream
from utils.execution_logger import ExecutionLogger, ExecutionMetrics
from utils.models import AgentMessage

# ----------------------------------------------------------------------
# DEBUG TEMPORAIRE -- à retirer une fois le diagnostic ML terminé.
# Logge, dans l'ordre RÉEL de réception (stdout / logs Streamlit), chaque
# événement reçu depuis `backend.obtenir_flux_code_review_avec_ml()`.
# Permet de vérifier si un événement `source == "MLRiskPredictor"` est
# effectivement émis en exécution réelle (et pas seulement dans les tests
# mockés) et à quel moment, sans dépendre de l'affichage Streamlit.
# ----------------------------------------------------------------------
_DEBUG_ML_ORDER = True


def executer_pipeline(prompt: str, placeholders: dict) -> None:
    """
    Exécute le pipeline complet pour une demande utilisateur, en mettant à
    jour l'affichage en direct via les placeholders fournis par app.py.

    Args:
        prompt: la demande de l'utilisateur.
        placeholders: dict de st.empty() tenus par app.py, avec les clés
            "workflow", "planner", "codeur", "ml_risk", "reviewer", "docker".

    ------------------------------------------------------------------
    Historique des exécutions (ExecutionLogger)
    ------------------------------------------------------------------
    C'est ICI, et non dans `backend.executer_pipeline_complet()` (jamais
    appelée par l'UI réelle -- voir diagnostic), que se trouve le seul
    endroit qui voit, dans l'ordre, la totalité d'une exécution UI réelle
    (Planner -> Codeur/Reviewer -> Docker). C'est donc ici que
    l'`ExecutionLogger` est branché.

    Contrairement à `executer_pipeline_complet()` (générateur asynchrone
    consommé par un thread séparé, où un `finally` pouvait ne jamais
    s'exécuter si le consommateur arrêtait de tirer les événements),
    `executer_pipeline()` est une fonction SYNCHRONE classique. Toute
    instruction placée avant un `return` s'exécute de façon fiable, garantie
    par le modèle d'exécution standard de Python -- aucun risque de
    générateur abandonné ici.

    Le logging est déclenché via `_log_now()`, appelée explicitement à
    CHACUN des points de sortie de la fonction (succès complet, révision
    sans Docker, erreur pendant le streaming), jamais depuis un `finally`
    seul. Un flag `logged` garantit qu'une seule ligne est écrite par appel
    à `executer_pipeline()`, quel que soit le chemin de sortie emprunté.

    Aucune métrique n'est estimée ou recalculée à partir de caractères :
    - les temps viennent de `time.perf_counter()` autour des étapes
      réellement exécutées ici (Planner, boucle de streaming, Docker) ;
    - les tokens viennent de `models_usage` tel que fourni par AutoGen /
      Azure OpenAI sur chaque message (Planner via
      `backend.generer_plan_avec_usage()`, Codeur/Reviewer directement sur
      les messages reçus depuis `stream.drain_queue()`).
    Toute métrique non disponible pour cette exécution reste `None` (ex:
    champs Docker si le Reviewer a demandé une révision et que Docker n'a
    jamais été atteint).

    Le logging est best-effort : une erreur SQLite est capturée et tracée,
    mais ne doit jamais empêcher Smartovate de terminer son exécution.
    """
    session = state.nouvelle_session(prompt)
    st.session_state["is_running"] = True
    # Dernier résultat ML connu pour CETTE exécution -- stocké dans
    # st.session_state (comme "is_running"/"etape_active" juste au-dessus)
    # plutôt que sur `session`/`AgentMessage` (utils/state.py et
    # utils/models.py ne sont pas modifiés ici). Réinitialisé à chaque
    # nouvelle demande pour ne jamais montrer un résultat ML périmé
    # provenant d'une tâche précédente tant qu'aucun nouveau résultat n'est
    # encore arrivé.
    st.session_state["last_ml_risk_content"] = None
    debut = time.monotonic()
    pipeline_start = time.perf_counter()

    metrics = ExecutionMetrics(task=prompt)
    logger = ExecutionLogger()
    logged = False

    def _log_now() -> None:
        """
        Finalise `metrics` (temps total) et écrit UNE ligne SQLite.
        Idempotente via `logged` : n'écrit qu'une seule fois par appel à
        `executer_pipeline()`, quel que soit le point de sortie emprunté.
        Best-effort : ne relève jamais d'exception vers l'appelant.
        """
        nonlocal logged
        if logged:
            return
        logged = True
        metrics.total_execution_time = time.perf_counter() - pipeline_start
        try:
            logger.log_execution(metrics)
        except Exception as exc:
            print(f"[ExecutionLogger] échec de l'écriture SQLite : {exc!r}")

    # ---------- Étape 1 : PlannerAgent ----------
    st.session_state["etape_active"] = "planner"
    placeholders["workflow"].markdown(_render_workflow("planner"), unsafe_allow_html=True)
    with placeholders["planner"].container():
        with st.spinner("Le PlannerAgent réfléchit au plan…"):
            planner_start = time.perf_counter()
            plan_result = asyncio.run(backend.generer_plan_avec_usage(prompt))
            metrics.planner.execution_time = time.perf_counter() - planner_start

    plan = plan_result.plan
    if plan_result.models_usage is not None:
        metrics.planner.prompt_tokens = plan_result.models_usage.prompt_tokens
        metrics.planner.completion_tokens = plan_result.models_usage.completion_tokens

    session.plan = plan
    session.messages.append(AgentMessage(source="PlannerAgent", content=plan))
    placeholders["planner"].markdown(_render_plan_card(plan), unsafe_allow_html=True)

    # ---------- Étape 2 : CodeReviewTeam (streaming) ----------
    st.session_state["etape_active"] = "codeur_reviewer"
    placeholders["workflow"].markdown(_render_workflow("codeur_reviewer"), unsafe_allow_html=True)

    max_iterations_value = st.session_state.get("max_iterations", 5)
    team = backend.creer_team_code_review(max_iterations=max_iterations_value)
    # Utilise la variante qui insère la prédiction de risque ML entre
    # chaque tour du CodeurAgent et le tour suivant du ReviewerAgent (voir
    # backend.obtenir_flux_code_review_avec_ml -- le Reviewer continue de
    # tourner à l'identique, le ML est purement informatif et ne peut pas
    # le bloquer ni le remplacer).
    q = stream.run_stream_in_thread(
        lambda: backend.obtenir_flux_code_review_avec_ml(
            team,
            plan,
            prompt,
            metrics.planner.execution_time,
            metrics.planner.prompt_tokens,
            metrics.planner.completion_tokens,
        )
    )

    task_result = None
    codeur_placeholder = placeholders["codeur"]
    reviewer_placeholder = placeholders["reviewer"]
    # Placeholder dédié, physiquement positionné par app.py entre la carte
    # Codeur et la carte Reviewer -- le ML a donc son propre emplacement
    # visuel, distinct de celui du Codeur et du Reviewer.
    ml_risk_placeholder = placeholders["ml_risk"]
    codeur_html = ""
    reviewer_html = ""
    # Compte le nombre RÉEL de tours Codeur déjà vus dans cette exécution
    # (incrémenté ci-dessous à chaque message CodeurAgent) -- sert
    # uniquement à afficher "Iteration N / max_iterations" dans la carte
    # ML, à partir d'une donnée réellement observée dans le flux, jamais
    # inventée.
    codeur_iteration_count = 0

    # Chronomètre la boucle de streaming dans son ensemble. Note sur la
    # fiabilité (voir consigne) : `drain_queue()` renvoie des LOTS
    # d'événements (jusqu'à `block_timeout` secondes de vidage non bloquant
    # après le premier), potentiellement plusieurs messages Codeur/Reviewer
    # dans le même lot. Il n'existe donc pas, avec ce mécanisme de
    # streaming par lots, de moyen fiable de séparer le temps consommé par
    # CodeurAgent de celui consommé par ReviewerAgent SANS ajouter de
    # chronométrage dans `stream.py` (hors périmètre autorisé pour cette
    # tâche). On ne l'invente donc PAS : `coder_execution_time` et
    # `reviewer_execution_time` restent `None`. Seuls les TOKENS (attachés
    # individuellement à chaque message via `models_usage`) peuvent être
    # distingués de façon fiable par agent, et sont bien accumulés
    # séparément ci-dessous.
    while True:
        events, finished = stream.drain_queue(q, block_timeout=0.3)
        for event in events:
            if _DEBUG_ML_ORDER:
                print(f"[DEBUG] event.kind={event.kind!r} source={getattr(event.payload, 'source', None)!r}")
            if event.kind == "error":
                st.session_state["is_running"] = False
                session.status = "error"
                # --- MODIFICATION UI UNIQUEMENT ---------------------------
                # Expose au pipeline visuel un état d'échec qui existait
                # déjà côté données (`session.status = "error"` juste
                # au-dessus) mais qui n'était jamais transmis à
                # `etape_active` : le workflow restait donc figé sur
                # "codeur_reviewer / en cours" alors que l'exécution était
                # terminée. Aucune logique métier n'est modifiée ici --
                # ni le flux, ni le résultat, ni le chemin de sortie
                # (`_log_now()` puis `return`, inchangés).
                st.session_state["etape_active"] = "error"
                placeholders["workflow"].markdown(_render_workflow("error"), unsafe_allow_html=True)
                # --- fin MODIFICATION UI ---------------------------------
                placeholders["codeur"].error(f"Erreur pendant la boucle Codeur/Reviewer : {event.payload}")
                _log_now()
                return
            if event.kind == "result":
                task_result = event.payload
                continue
            # event.kind == "message"
            msg = event.payload
            source = getattr(msg, "source", "")
            content = getattr(msg, "content", str(msg))
            usage = getattr(msg, "models_usage", None)
            if source == "CodeurAgent":
                if _DEBUG_ML_ORDER:
                    print("[DEBUG] CodeurAgent")
                codeur_iteration_count += 1
                session.messages.append(AgentMessage(source="CodeurAgent", content=content))
                codeur_html = _render_codeur_card(content)
                codeur_placeholder.markdown(codeur_html, unsafe_allow_html=True)
                if usage is not None:
                    metrics.coder.prompt_tokens = (metrics.coder.prompt_tokens or 0) + usage.prompt_tokens
                    metrics.coder.completion_tokens = (metrics.coder.completion_tokens or 0) + usage.completion_tokens
            elif source == "ReviewerAgent":
                if _DEBUG_ML_ORDER:
                    print("[DEBUG] ReviewerAgent")
                # --- [REVIEWER DEBUG] diagnostic temporaire ---
                print(f"[REVIEWER DEBUG] Reviewer reçu content_len={len(content) if content else 0} content={content!r}")
                # --- fin [REVIEWER DEBUG] ---
                session.messages.append(AgentMessage(source="ReviewerAgent", content=content))
                print(f"[REVIEWER DEBUG] Reviewer sauvegardé dans session.messages (total messages={len(session.messages)})")
                reviewer_html = _render_reviewer_card(content)
                reviewer_placeholder.markdown(reviewer_html, unsafe_allow_html=True)
                if usage is not None:
                    metrics.reviewer.prompt_tokens = (metrics.reviewer.prompt_tokens or 0) + usage.prompt_tokens
                    metrics.reviewer.completion_tokens = (metrics.reviewer.completion_tokens or 0) + usage.completion_tokens
            elif source == "MLRiskPredictor":
                if _DEBUG_ML_ORDER:
                    print(f"[DEBUG] MLRiskPredictor content={content!r}")
                # Signal informatif uniquement : n'affecte ni le Reviewer
                # (déjà passé, voir backend.obtenir_flux_code_review_avec_ml)
                # ni la suite du pipeline. Pas de comptage de tokens (aucun
                # appel Azure OpenAI ici, voir ml_risk.py).
                #
                # `iteration`/`max_iterations` sont ajoutés ICI (pas dans
                # ml_risk.py) car seul orchestrator.py sait combien de
                # tours Codeur ont réellement eu lieu jusqu'ici -- ce ne
                # sont pas des valeurs inventées, juste le compteur réel
                # incrémenté ci-dessus.
                ml_content = content
                if isinstance(content, dict):
                    ml_content = {
                        **content,
                        "iteration": codeur_iteration_count,
                        "max_iterations": max_iterations_value,
                    }
                # Persisté dans st.session_state (voir réinitialisation en
                # tête de fonction) : c'est CE stockage qui manquait -- sans
                # lui, le `st.rerun()` déclenché par prompt_box.py après la
                # fin du pipeline recrée des placeholders["ml_risk"] tout
                # neufs (voir app.py) qu'aucun bloc de ré-affichage
                # d'historique ne remplissait plus jamais, contrairement au
                # Planner/Codeur/Reviewer/Docker qui, eux, sont relus depuis
                # `session.messages` / `session.docker_*`.
                st.session_state["last_ml_risk_content"] = ml_content
                ml_risk_placeholder.markdown(
                    _render_ml_risk_card(ml_content), unsafe_allow_html=True
                )
            elif _DEBUG_ML_ORDER and source:
                # Diagnostic : capte toute source reçue mais non gérée par
                # les 3 branches ci-dessus (ex: le nom réel émis par
                # backend.py diffère de "MLRiskPredictor").
                print(f"[DEBUG] source non gérée : {source!r}")

        if finished:
            break

    if task_result is not None:
        result = backend.construire_resultat(task_result)
        session.final_code = result.final_code
        session.approved = result.approved
        session.iterations_used = result.iterations_used
        st.session_state["code_review_result"] = result

        metrics.revision_count = result.iterations_used
        metrics.reviewer_status = "APPROVED" if result.approved else "REVISION_REQUISE"
        if result.final_code:
            metrics.code_length = len(result.final_code)
            metrics.code_line_count = result.final_code.count("\n") + 1

    # ---------- Étape 3 : Docker ----------
    result = st.session_state.get("code_review_result")

    if result is None or not result.approved or not result.final_code:
        # Révision requise (ou aucun résultat) : Docker jamais atteint.
        # Champs Docker laissés à None -- non une erreur d'infrastructure.
        session.status = "done"
        session.duration_seconds = time.monotonic() - debut
        st.session_state["etape_active"] = "done"
        st.session_state["is_running"] = False
        placeholders["workflow"].markdown(_render_workflow("done"), unsafe_allow_html=True)
        _log_now()
        return

    st.session_state["etape_active"] = "docker"
    placeholders["workflow"].markdown(_render_workflow("docker"), unsafe_allow_html=True)

    with placeholders["docker"].container():
        with st.spinner("Exécution du code dans un conteneur Docker isolé…"):
            docker_start = time.perf_counter()
            outcome = asyncio.run(backend.executer_dans_docker(result))
            metrics.docker_execution_time = time.perf_counter() - docker_start

    if outcome is not None:
        session.docker_error_message = outcome.error_message
        if outcome.result is not None:
            session.docker_success = outcome.result.success
            session.docker_exit_code = outcome.result.exit_code
            session.docker_output = outcome.result.output
            metrics.docker_status = "SUCCESS" if outcome.result.success else "ERROR"
            metrics.docker_exit_code = outcome.result.exit_code
        elif outcome.requires_input:
            metrics.docker_status = "SKIPPED_INTERACTIVE_INPUT"
        else:
            metrics.docker_status = "ERROR"
    placeholders["docker"].markdown(
        _render_docker_card(session), unsafe_allow_html=True
    )

    session.status = "done"
    session.duration_seconds = time.monotonic() - debut
    st.session_state["etape_active"] = "done"
    st.session_state["is_running"] = False
    placeholders["workflow"].markdown(_render_workflow("done"), unsafe_allow_html=True)

    _log_now()


# ----------------------------------------------------------------------
# Petits rendus HTML locaux à l'orchestrateur (le CSS vient de style.css).
# Les rendus "riches" détaillés vivent dans components/agent_cards.py et
# sont réutilisés ici pour rester cohérents.
# ----------------------------------------------------------------------

def _render_workflow(etape: str) -> str:
    from components import workflow
    return workflow.render_html(etape)


def _render_plan_card(plan: str) -> str:
    from components import agent_cards
    return agent_cards.render_planner_card(plan)


def _render_codeur_card(content: str) -> str:
    from components import agent_cards
    return agent_cards.render_codeur_card(content)


def _render_reviewer_card(content: str) -> str:
    from components import agent_cards
    return agent_cards.render_reviewer_card(content)


def _render_docker_card(session) -> str:
    from components import agent_cards
    return agent_cards.render_docker_card(session)


def _render_ml_risk_card(content: dict | str | None) -> str:
    """
    Rendu de la carte ML Risk Predictor, désormais délégué à
    `components/agent_cards.render_ml_risk_card()` -- comme pour le
    Planner, le Codeur, le Reviewer et Docker -- afin de réutiliser
    strictement le même système visuel (mêmes classes CSS, mêmes badges,
    même structure `sv-card sv-agent-card`) plutôt qu'un bloc HTML
    autonome étranger au reste de l'interface.
    """
    from components import agent_cards
    return agent_cards.render_ml_risk_card(content)