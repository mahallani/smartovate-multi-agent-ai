# streamlit_app/components/history.py
"""
Historique des conversations, affiché dans la sidebar.

Persistance légère : uniquement en mémoire (st.session_state) pour cette
version. Aucune logique backend ici -- ne lit/écrit que l'état frontend.
"""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown('<div class="sv-section-title">Historique</div>', unsafe_allow_html=True)

    sessions = list(st.session_state.get("sessions", {}).values())
    if not sessions:
        st.markdown(
            '<div class="sv-empty">Aucune conversation pour le moment.</div>',
            unsafe_allow_html=True,
        )
        return

    for session in reversed(sessions):
        label = session.prompt[:38] + ("…" if len(session.prompt) > 38 else "")
        is_current = session.id == st.session_state.get("current_session_id")
        css_class = "sv-history-item-active" if is_current else "sv-history-item"
        st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
        if st.button(label, key=f"hist_{session.id}", use_container_width=True):
            st.session_state["current_session_id"] = session.id
            st.session_state["etape_active"] = "done"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
