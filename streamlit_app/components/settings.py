# streamlit_app/components/settings.py
"""
Panneau de réglages : Mode Développeur/Utilisateur, nombre max d'itérations
Codeur/Reviewer (déjà paramétrable côté backend via
build_code_review_team(max_iterations=...)).
"""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown('<div class="sv-section-title">Réglages</div>', unsafe_allow_html=True)

    st.session_state["dev_mode"] = st.toggle(
        "Mode Développeur",
        value=st.session_state.get("dev_mode", False),
        help="Affiche la timeline détaillée, les logs bruts Docker et le JSON complet.",
    )

    st.session_state["max_iterations"] = st.slider(
        "Itérations max Codeur ↔ Reviewer",
        min_value=1,
        max_value=10,
        value=st.session_state.get("max_iterations", 5),
        help="Transmis tel quel à build_code_review_team(max_iterations=...).",
    )
