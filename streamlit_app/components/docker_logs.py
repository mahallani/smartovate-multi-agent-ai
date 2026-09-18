# streamlit_app/components/docker_logs.py
"""
Terminal dédié à la sortie d'exécution Docker (vue détaillée, distincte
de la carte Docker live d'agent_cards.py). Signature `render(session)`,
identique à l'appel déjà fait par app.py.
"""
from __future__ import annotations

import html
import textwrap
from typing import Any

import streamlit as st


def _flat(s: str) -> str:
    dedented = textwrap.dedent(s)
    lines = [line for line in dedented.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def render(session: Any) -> None:
    if session is None:
        return

    docker_output = getattr(session, "docker_output", None)
    docker_error = getattr(session, "docker_error_message", None)
    docker_requires_input = getattr(session, "docker_requires_input", False)
    exit_code = getattr(session, "docker_exit_code", None)
    success = getattr(session, "docker_success", None)

    if docker_output is None and docker_error is None:
        return  # rien à afficher tant que Docker n'a pas encore été sollicité

    if docker_requires_input:
        exitcode_html = ""
        body = f'<div class="sv-terminal-line sv-terminal-dim">⌨ {html.escape(docker_error or "")}</div>'
    elif docker_error:
        exitcode_html = '<span class="sv-terminal-exitcode sv-terminal-exitcode-fail">non exécuté</span>'
        body = f'<div class="sv-terminal-line sv-terminal-error">{html.escape(docker_error)}</div>'
    else:
        cls = "sv-terminal-exitcode-ok" if success else "sv-terminal-exitcode-fail"
        exitcode_html = f'<span class="sv-terminal-exitcode {cls}">exit code {exit_code}</span>'
        output_html = html.escape(docker_output or "").replace("\n", "<br>")
        body = f'<div class="sv-terminal-line">{output_html}</div>'

    st.markdown(
        _flat(f"""
        <div class="sv-card">
            <h4 style="margin-top:0;">Docker Terminal</h4>
            <div class="sv-terminal">
                <div class="sv-terminal-header">
                    <div class="sv-terminal-dots"><span class="r"></span><span class="y"></span><span class="g"></span></div>
                    {exitcode_html}
                </div>
                <div class="sv-terminal-title">smartovate@docker-executor:~$ python script.py</div>
                {body}
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )