# streamlit_app/components/export.py
"""
Boutons d'export (Issue 4) : Python / Markdown / JSON / Logs.

Ce fichier n'existait pas dans les fichiers fournis (les boutons étaient
donc forcément désactivés ou non câblés) -- il est créé ici, from scratch,
à partir des données déjà disponibles sur `session` (telles qu'utilisées
par components/agent_cards.py : `.plan`, `.messages` (source/content),
`.docker_output`, `.docker_success`, `.docker_exit_code`,
`.docker_error_message`, `.final_code`, `.approved`).

Si votre `utils/models.py` (non fourni) utilise des noms de champs
différents, ajustez les `getattr(session, "...", ...)` ci-dessous en
conséquence -- ils sont volontairement défensifs (valeur par défaut si le
champ n'existe pas) pour ne jamais faire planter l'export à cause d'un nom
de champ manquant.

Aucune logique métier ici : uniquement la mise en forme de données déjà
produites par le pipeline (voir utils/backend.py).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import streamlit as st


def _get(session: Any, name: str, default: Any = None) -> Any:
    """Accès défensif à un champ de session, sans jamais lever d'erreur."""
    return getattr(session, name, default)


def _messages_as_dicts(session: Any) -> list[dict[str, str]]:
    messages = _get(session, "messages", []) or []
    result = []
    for msg in messages:
        source = getattr(msg, "source", None) or (msg.get("source") if isinstance(msg, dict) else "inconnu")
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else str(msg))
        result.append({"source": source, "content": content})
    return result


def build_python_export(session: Any) -> str:
    """Code Python final, prêt à exécuter tel quel."""
    code = _get(session, "final_code")
    if not code:
        return (
            "# Aucun code final disponible pour cette conversation.\n"
            "# Le code n'a pas encore été approuvé par le ReviewerAgent, ou "
            "aucune conversation n'a été exécutée.\n"
        )
    header = (
        f"# Généré par Smartovate AI Multi-Agent Studio\n"
        f"# Exporté le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + code


def build_markdown_export(session: Any) -> str:
    """Rapport Markdown complet et lisible de la conversation."""
    prompt = _get(session, "prompt", "")
    plan = _get(session, "plan", "") or "_Aucun plan généré._"
    approved = _get(session, "approved")
    iterations = _get(session, "iterations_used", 0)
    final_code = _get(session, "final_code")
    docker_output = _get(session, "docker_output")
    docker_exit_code = _get(session, "docker_exit_code")
    docker_error = _get(session, "docker_error_message")

    lines: list[str] = []
    lines.append("# Rapport de conversation — Smartovate\n")
    lines.append(f"**Demande initiale :** {prompt}\n")
    lines.append("## Plan (PlannerAgent)\n")
    lines.append(f"```\n{plan}\n```\n")

    lines.append("## Boucle Codeur ↔ Reviewer\n")
    for msg in _messages_as_dicts(session):
        lines.append(f"### {msg['source']}\n")
        lines.append(f"{msg['content']}\n")

    lines.append("## Résultat de la revue\n")
    lines.append(f"- **Approuvé :** {'✅ Oui' if approved else '❌ Non'}")
    lines.append(f"- **Itérations utilisées :** {iterations}\n")

    if final_code:
        lines.append("## Code final\n")
        lines.append(f"```python\n{final_code}\n```\n")

    lines.append("## Exécution Docker\n")
    if docker_error:
        lines.append(f"⚠ {docker_error}\n")
    elif docker_output is not None:
        lines.append(f"- **Exit code :** {docker_exit_code}\n")
        lines.append(f"```\n{docker_output}\n```\n")
    else:
        lines.append("_Non exécuté._\n")

    return "\n".join(lines)


def build_json_export(session: Any) -> str:
    """Export JSON structuré de toute la conversation."""
    data = {
        "id": _get(session, "id"),
        "prompt": _get(session, "prompt"),
        "created_at": str(_get(session, "created_at", "")),
        "plan": _get(session, "plan"),
        "messages": _messages_as_dicts(session),
        "approved": _get(session, "approved"),
        "iterations_used": _get(session, "iterations_used"),
        "final_code": _get(session, "final_code"),
        "docker": {
            "success": _get(session, "docker_success"),
            "exit_code": _get(session, "docker_exit_code"),
            "output": _get(session, "docker_output"),
            "error_message": _get(session, "docker_error_message"),
            "requires_input": _get(session, "docker_requires_input", False),
        },
    }
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def build_logs_export(session: Any) -> str:
    """Logs bruts, format texte simple (horodatage + événement)."""
    lines: list[str] = []
    lines.append(f"[conversation] id={_get(session, 'id')} prompt={_get(session, 'prompt')!r}")
    lines.append(f"[planner] plan_length={len(_get(session, 'plan') or '')}")
    for msg in _messages_as_dicts(session):
        lines.append(f"[{msg['source']}] {msg['content'][:200].replace(chr(10), ' ')}")
    lines.append(
        f"[review] approved={_get(session, 'approved')} "
        f"iterations={_get(session, 'iterations_used')}"
    )
    docker_error = _get(session, "docker_error_message")
    if docker_error:
        lines.append(f"[docker] error={docker_error}")
    else:
        lines.append(
            f"[docker] success={_get(session, 'docker_success')} "
            f"exit_code={_get(session, 'docker_exit_code')}"
        )
        output = _get(session, "docker_output")
        if output:
            for line in output.splitlines():
                lines.append(f"[docker:stdout] {line}")
    return "\n".join(lines)


def render(session: Any) -> None:
    """
    Affiche les 4 boutons d'export. Désactivés uniquement si aucune
    conversation n'a encore été exécutée (session is None) -- sinon
    toujours actifs et fonctionnels, y compris sur un résultat partiel
    (ex: pas encore de code Docker exécuté).
    """
    st.markdown('<div class="sv-card sv-export-card">', unsafe_allow_html=True)
    st.markdown("#### Export")

    disabled = session is None
    session_id = _get(session, "id", "session") if session else "session"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button(
            "⬇ Python",
            data=build_python_export(session) if not disabled else "",
            file_name=f"{session_id}.py",
            mime="text/x-python",
            disabled=disabled,
            use_container_width=True,
            key=f"export_py_{session_id}",
        )
    with col2:
        st.download_button(
            "⬇ Markdown",
            data=build_markdown_export(session) if not disabled else "",
            file_name=f"{session_id}.md",
            mime="text/markdown",
            disabled=disabled,
            use_container_width=True,
            key=f"export_md_{session_id}",
        )
    with col3:
        st.download_button(
            "⬇ JSON",
            data=build_json_export(session) if not disabled else "",
            file_name=f"{session_id}.json",
            mime="application/json",
            disabled=disabled,
            use_container_width=True,
            key=f"export_json_{session_id}",
        )
    with col4:
        st.download_button(
            "⬇ Logs",
            data=build_logs_export(session) if not disabled else "",
            file_name=f"{session_id}.log",
            mime="text/plain",
            disabled=disabled,
            use_container_width=True,
            key=f"export_logs_{session_id}",
        )

    st.markdown("</div>", unsafe_allow_html=True)