# streamlit_app/components/prompt_box.py
"""
Zone de saisie principale : textarea, prompts d'exemple cliquables, bouton
Envoyer. Ne contient AUCUNE logique métier -- délègue entièrement
l'exécution du pipeline à utils/orchestrator.py.

Seul l'habillage a changé (carte en verre, libellé, indication). Le bloc
de déclenchement en fin de fonction est STRICTEMENT identique à
l'original : même condition, même appel `orchestrator.executer_pipeline`,
même `st.rerun()`.
"""
from __future__ import annotations

import streamlit as st

from utils import orchestrator

EXEMPLES = [
    "Écris une fonction Python qui additionne deux nombres.",
    "Analyse un fichier CSV de ventes et calcule le chiffre d'affaires mensuel.",
    "Écris un script qui trie une liste de nombres sans utiliser sorted().",
]


def render(placeholders: dict) -> None:
    is_running = st.session_state.get("is_running", False)

    st.markdown('<div class="sv-prompt-wrapper">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sv-prompt-label">✨ Votre tâche</div>'
        '<p class="sv-prompt-hint">Décrivez ce que vous voulez obtenir : '
        "le pipeline Planner → Codeur → ML Risk → Reviewer → Docker s'en charge.</p>",
        unsafe_allow_html=True,
    )

    prompt = st.text_area(
        "Décrivez votre tâche…",
        key="prompt_input",
        label_visibility="collapsed",
        placeholder="Décrivez votre tâche…",
        height=110,
        disabled=is_running,
    )

    st.markdown('<p class="sv-prompt-examples-label">Exemples</p>', unsafe_allow_html=True)

    cols = st.columns([1, 1, 1, 2])
    for i, exemple in enumerate(EXEMPLES):
        with cols[i]:
            if st.button(exemple[:28] + "…", key=f"exemple_{i}", use_container_width=True, disabled=is_running):
                st.session_state["prompt_input"] = exemple
                st.rerun()

    with cols[3]:
        envoyer = st.button(
            "Envoyer ➤",
            type="primary",
            use_container_width=True,
            disabled=is_running,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if envoyer and prompt.strip():
        orchestrator.executer_pipeline(prompt.strip(), placeholders)
        st.rerun()