# streamlit_app/components/timeline.py
"""
Timeline des étapes du pipeline, avec indicateur de statut animé par étape
(en attente / en cours / terminé / erreur). Signature `render(session)`,
identique à l'appel déjà fait par app.py.

Construit uniquement à partir de champs déjà utilisés ailleurs dans le
frontend (session.plan, session.messages, session.approved,
session.docker_success / docker_error_message / docker_requires_input,
session.status) et, pour la seule ligne ML Risk, de
st.session_state["last_ml_risk_content"] -- le stockage déjà mis en place
par utils/orchestrator.py. Lecture systématiquement défensive (getattr)
pour ne jamais dépendre d'un nom de champ non garanti dans
utils/models.py.

AUCUNE donnée métier n'est créée ici : quand une information n'est pas
disponible, la ligne reste "en attente" plutôt que d'afficher une valeur
inventée.
"""
from __future__ import annotations

import textwrap
from typing import Any

import streamlit as st


def _flat(s: str) -> str:
    dedented = textwrap.dedent(s)
    lines = [line for line in dedented.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def _animations_enabled() -> bool:
    try:
        return bool(st.session_state.get("settings", {}).get("animations", True))
    except Exception:
        return True


def _step(label: str, status: str, detail: str = "") -> str:
    """status: 'waiting' | 'running' | 'done' | 'error'"""
    status_label = {"waiting": "En attente", "running": "En cours", "done": "Terminé", "error": "Erreur"}[status]
    status_class = {
        "waiting": "sv-badge-thinking",
        "running": "sv-badge-typing",
        "done": "sv-badge-approved",
        "error": "sv-badge-revision",
    }[status]
    detail_html = f'<span style="color:var(--color-subtitle);font-size:0.78rem;">{detail}</span>' if detail else ""
    return (
        f'<div class="sv-timeline-row sv-timeline-status-{status}">'
        f'<span class="sv-timeline-dot"></span>'
        f'<span class="sv-timeline-row-label">{label} {detail_html}</span>'
        f'<span class="sv-timeline-row-status {status_class}">{status_label}</span>'
        f"</div>"
    )


def _ml_risk_step(plan: Any, codeur_msgs: list) -> tuple[str, str, str]:
    """
    Ligne ML Risk, dérivée du DERNIER signal réellement reçu et stocké par
    orchestrator.py dans st.session_state["last_ml_risk_content"].

    Aucun recalcul, aucun seuil, aucune probabilité recréée : on se
    contente de refléter le dict tel quel. Si aucun signal n'a été reçu,
    on le dit explicitement plutôt que d'afficher un faux "terminé".
    """
    try:
        content = st.session_state.get("last_ml_risk_content")
    except Exception:
        content = None

    if isinstance(content, dict):
        risk_level = content.get("risk_level")
        probability = content.get("probability")
        parts = []
        if risk_level:
            parts.append(str(risk_level))
        if probability is not None:
            try:
                parts.append(f"{float(probability) * 100:.1f} %")
            except (TypeError, ValueError):
                pass
        detail = " — ".join(parts) if parts else ""
        return ("ML Risk Predictor — signal de risque", "done", detail)

    if not plan:
        return ("ML Risk Predictor — signal de risque", "waiting", "")
    if codeur_msgs:
        return ("ML Risk Predictor — signal de risque", "waiting", "aucun signal reçu pour cette exécution")
    return ("ML Risk Predictor — signal de risque", "waiting", "")


def _derive_steps(session: Any) -> list[tuple[str, str, str]]:
    plan = getattr(session, "plan", None)
    messages = getattr(session, "messages", []) or []
    approved = getattr(session, "approved", None)
    docker_requires_input = getattr(session, "docker_requires_input", False)
    docker_error = getattr(session, "docker_error_message", None)
    docker_success = getattr(session, "docker_success", None)
    global_status = getattr(session, "status", None)

    codeur_msgs = [m for m in messages if getattr(m, "source", None) == "CodeurAgent"]
    reviewer_msgs = [m for m in messages if getattr(m, "source", None) == "ReviewerAgent"]

    steps = []

    if plan:
        steps.append(("Planner — génération du plan", "done", ""))
    elif global_status == "error" and not messages:
        steps.append(("Planner — génération du plan", "error", ""))
    else:
        steps.append(("Planner — génération du plan", "waiting", ""))

    if not plan:
        steps.append(("Codeur — génération du code", "waiting", ""))
    elif codeur_msgs:
        steps.append(("Codeur — génération du code", "done", f"{len(codeur_msgs)} version(s)"))
    elif global_status == "error":
        steps.append(("Codeur — génération du code", "error", ""))
    else:
        steps.append(("Codeur — génération du code", "waiting", ""))

    steps.append(_ml_risk_step(plan, codeur_msgs))

    if not plan:
        steps.append(("Reviewer — revue du code", "waiting", ""))
    elif approved is True:
        steps.append(("Reviewer — revue du code", "done", f"{len(reviewer_msgs)} revue(s) — approuvé"))
    elif approved is False:
        steps.append(("Reviewer — revue du code", "error", "non approuvé après le nombre maximal d'itérations"))
    elif global_status == "error":
        steps.append(("Reviewer — revue du code", "error", "pipeline interrompu"))
    elif reviewer_msgs:
        steps.append(("Reviewer — revue du code", "running", f"{len(reviewer_msgs)} revue(s) en cours"))
    else:
        steps.append(("Reviewer — revue du code", "waiting", ""))

    if approved is not True:
        steps.append(("Docker — exécution isolée", "waiting", ""))
    elif docker_requires_input:
        steps.append(("Docker — exécution isolée", "error", "saisie interactive requise"))
    elif docker_error:
        steps.append(("Docker — exécution isolée", "error", "non exécuté"))
    elif docker_success is True:
        steps.append(("Docker — exécution isolée", "done", "succès"))
    elif docker_success is False:
        steps.append(("Docker — exécution isolée", "error", "échec"))
    else:
        steps.append(("Docker — exécution isolée", "running", ""))

    return steps


def render(session: Any) -> None:
    anim_class = "" if _animations_enabled() else " sv-no-anim"

    if session is None:
        st.markdown(
            _flat(f"""
            <div class="sv-card{anim_class}">
                <div class="sv-empty-state">
                    <span class="sv-empty-state-icon">🕓</span>
                    Aucune conversation en cours — la timeline apparaîtra ici.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )
        return

    steps = _derive_steps(session)
    rows_html = "".join(_step(label, status, detail) for label, status, detail in steps)

    st.markdown(
        _flat(f"""
        <div class="sv-card{anim_class}">
            <h4 style="margin-top:0;">🕓 Timeline d'exécution</h4>
            {rows_html}
        </div>
        """),
        unsafe_allow_html=True,
    )