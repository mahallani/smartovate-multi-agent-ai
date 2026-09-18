# streamlit_app/components/statistics.py
"""
"Execution Overview" : panneau de cartes KPI (durée, itérations, messages,
approbation Reviewer, statut Docker, statut final, modèle). Signature
`render(session)`, identique à l'appel déjà fait par app.py.

Lecture systématiquement défensive (getattr) des champs de session --
aucune donnée non disponible n'affiche d'erreur, seulement un tiret
neutre. AUCUNE donnée n'est calculée, estimée ou inventée ici : toutes
les valeurs proviennent telles quelles de l'objet `session` rempli par
utils/orchestrator.py.
"""
from __future__ import annotations

import os
import textwrap
from typing import Any

import streamlit as st


def _flat(s: str) -> str:
    dedented = textwrap.dedent(s)
    lines = [line for line in dedented.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def _stat(icon: str, label: str, value: str, value_class: str = "") -> str:
    css = f"sv-stat-value {value_class}".strip()
    return (
        f'<div class="sv-stat">'
        f'<span class="sv-stat-icon">{icon}</span>'
        f'<div class="sv-stat-label">{label}</div>'
        f'<div class="{css}">{value}</div>'
        f"</div>"
    )


def render(session: Any) -> None:
    if session is None:
        return

    duration = getattr(session, "duration_seconds", None)
    duration_str = f"{duration:.1f} s" if isinstance(duration, (int, float)) else "—"

    iterations = getattr(session, "iterations_used", None)
    max_it = st.session_state.get("max_iterations")
    if iterations is not None and max_it is not None:
        iterations_str = f"{iterations} / {max_it}"
    elif iterations is not None:
        iterations_str = str(iterations)
    else:
        iterations_str = "—"

    messages = getattr(session, "messages", []) or []
    messages_str = str(len(messages))

    docker_success = getattr(session, "docker_success", None)
    if docker_success is True:
        docker_str, docker_class = "✓ Succès", "sv-stat-value-ok"
    elif docker_success is False:
        docker_str, docker_class = "✗ Échec", "sv-stat-value-err"
    else:
        docker_str, docker_class = "—", ""

    model_name = os.getenv("AZURE_OPENAI_MODEL", "gpt-5")

    approved = getattr(session, "approved", None)
    if approved is True:
        approved_str, approved_class = "APPROVED", "sv-stat-value-ok"
    elif approved is False:
        approved_str, approved_class = "RÉVISION", "sv-stat-value-ko"
    else:
        approved_str, approved_class = "—", ""

    # Statut final : champ `session.status`, posé par orchestrator.py
    # ("done" à chaque sortie normale, "error" sur le chemin d'échec).
    global_status = getattr(session, "status", None)
    if global_status == "error":
        status_str, status_class = "⚠ Erreur", "sv-stat-value-err"
    elif global_status == "done":
        status_str, status_class = "✓ Terminé", "sv-stat-value-ok"
    elif global_status:
        status_str, status_class = str(global_status), ""
    else:
        status_str, status_class = "En cours", ""

    stats_html = "".join([
        _stat("⏱", "Durée totale", duration_str),
        _stat("🔁", "Itérations", iterations_str),
        _stat("💬", "Messages échangés", messages_str),
        _stat("🔍", "Reviewer", approved_str, approved_class),
        _stat("🐳", "Exécution Docker", docker_str, docker_class),
        _stat("🏁", "Statut final", status_str, status_class),
        _stat("🧠", "Modèle", model_name),
    ])

    st.markdown(
        _flat(f"""
        <div class="sv-card">
            <h4 style="margin-top:0;">📊 Execution Overview</h4>
            <div class="sv-stat-grid">{stats_html}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )