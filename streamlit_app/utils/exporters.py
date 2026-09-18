# streamlit_app/utils/exporters.py
"""
Génère le CONTENU des fichiers exportables (Python, Markdown, JSON, Logs).

Pure logique de formatage -- aucun affichage ici. Les boutons de
téléchargement vivent dans components/export.py.
"""
from __future__ import annotations

import json

from utils.models import ConversationSession


def export_python(session: ConversationSession) -> str:
    """Retourne le code final tel qu'approuvé (ou un message si absent)."""
    return session.final_code or "# Aucun code final disponible pour cette conversation.\n"


def export_markdown(session: ConversationSession) -> str:
    """Retourne un rapport Markdown complet de la conversation."""
    lines = [
        "# Smartovate — Rapport de session",
        "",
        f"**Date :** {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Demande :** {session.prompt}",
        "",
        "## Plan",
        "",
        session.plan or "_Aucun plan généré._",
        "",
        "## Conversation entre agents",
        "",
    ]
    for msg in session.messages:
        lines.append(f"### {msg.source} — {msg.timestamp.strftime('%H:%M:%S')}")
        lines.append("")
        lines.append(msg.content)
        lines.append("")

    lines += [
        "## Résultat",
        "",
        f"- **Approuvé par le Reviewer :** {'Oui' if session.approved else 'Non'}",
        f"- **Itérations utilisées :** {session.iterations_used}",
        f"- **Exécution Docker réussie :** {'Oui' if session.docker_success else 'Non'}",
        f"- **Code de sortie :** {session.docker_exit_code}",
        "",
        "## Code final",
        "",
        "```python",
        session.final_code or "# Aucun code",
        "```",
        "",
        "## Logs d'exécution Docker",
        "",
        "```",
        session.docker_output or "Aucun log disponible.",
        "```",
    ]
    return "\n".join(lines)


def export_json(session: ConversationSession) -> str:
    """Retourne un export JSON structuré et complet de la session."""
    data = {
        "id": session.id,
        "prompt": session.prompt,
        "created_at": session.created_at.isoformat(),
        "plan": session.plan,
        "messages": [
            {
                "source": m.source,
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
            }
            for m in session.messages
        ],
        "final_code": session.final_code,
        "approved": session.approved,
        "iterations_used": session.iterations_used,
        "docker": {
            "success": session.docker_success,
            "exit_code": session.docker_exit_code,
            "output": session.docker_output,
            "error_message": session.docker_error_message,
        },
        "duration_seconds": session.duration_seconds,
        "status": session.status,
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def export_logs(session: ConversationSession) -> str:
    """Retourne les logs bruts d'exécution Docker."""
    return session.docker_output or "Aucun log d'exécution disponible."
