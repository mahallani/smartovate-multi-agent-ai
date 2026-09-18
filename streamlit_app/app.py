# streamlit_app/app.py
"""
Point d'entrée unique de l'interface Smartovate.

Ne contient AUCUNE logique métier ni CSS en dur : uniquement l'orchestration
de l'affichage. Lancer avec :

    streamlit run streamlit_app/app.py

------------------------------------------------------------------------
Structure de page (§9)
------------------------------------------------------------------------
    HEADER (hero, badges Azure/Docker à état réel)
      -> PIPELINE VISUEL (workflow, 5 noeuds animés)
      -> EXECUTION OVERVIEW (cartes KPI)
      -> ZONE DE SAISIE
      -> CARTE PLANNER / CODEUR / ML RISK / REVIEWER
      -> ZONE DE SUCCÈS (guirlandes, uniquement si session.approved is True)
      -> CARTE DOCKER
      -> TIMELINE
      -> LOGS DOCKER / EXPORT

Le positionnement visuel de la zone de saisie et de l'Execution Overview
est obtenu avec `st.container()` : les conteneurs sont CRÉÉS tôt (donc
placés haut dans la page) mais REMPLIS plus bas dans le code, une fois
`placeholders` construit. C'est indispensable car `prompt_box.render()`
a besoin des placeholders, qui doivent exister avant son appel.

Note d'exécution : `prompt_box.render()` termine par `st.rerun()` lorsqu'un
pipeline vient d'être lancé. Tout ce qui suit cet appel n'est donc PAS
exécuté pendant la passe de lancement -- c'est le rerun qui produit
l'affichage final. C'est ce même rerun qui garantit la persistance des
cartes et des guirlandes, via le bloc de ré-affichage ci-dessous.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Permet d'importer agents/, teams/, config/ (racine du projet) même en
# lançant `streamlit run streamlit_app/app.py` depuis n'importe où.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from components import (
    docker_logs,
    export,
    hero,
    prompt_box,
    sidebar,
    statistics,
    timeline,
    workflow,
)
from components import agent_cards
from utils import state

st.set_page_config(
    page_title="Smartovate — AI Multi-Agent Studio",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_css() -> None:
    css_path = Path(__file__).parent / "styles" / "style.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _section_title(label: str) -> None:
    st.markdown(f'<div class="sv-section-title">{label}</div>', unsafe_allow_html=True)


def main() -> None:
    _load_css()
    state.init_state()

    # sidebar en premier : elle initialise st.session_state["settings"]
    # (toggle Animations, max_iterations) et le cache
    # st.session_state["docker_reachable"] que le hero réutilise ensuite
    # sans refaire de ping.
    sidebar.render()
    hero.render()

    session = state.session_courante()

    st.markdown('<div class="sv-main-container">', unsafe_allow_html=True)

    # ---------- PIPELINE VISUEL ----------
    workflow_ph = st.empty()
    workflow_ph.markdown(
        workflow.render_html(st.session_state.get("etape_active"), session),
        unsafe_allow_html=True,
    )

    # ---------- EXECUTION OVERVIEW (conteneur placé ici, rempli plus bas) ----------
    overview_container = st.container()

    # ---------- ZONE DE SAISIE (conteneur placé ici, rempli plus bas) ----------
    prompt_container = st.container()

    # ---------- CARTES DES AGENTS ----------
    # Les clés de ce dict sont exactement celles attendues par
    # utils/orchestrator.py -- aucune n'a été renommée ni retirée.
    placeholders = {
        "workflow": workflow_ph,
        "planner": st.empty(),
        "codeur": st.empty(),
        "ml_risk": st.empty(),
        "reviewer": st.empty(),
    }

    # Zone de succès : positionnée entre Reviewer et Docker (§16 :
    # Reviewer -> ✨ SUCCESS ✨ -> Docker). Volontairement HORS du dict
    # `placeholders` : orchestrator.py ne la connaît pas et n'a pas besoin
    # de la connaître. Elle est remplie uniquement par le bloc de
    # ré-affichage ci-dessous, à partir de `session.approved`.
    garland_ph = st.empty()

    placeholders["docker"] = st.empty()

    timeline_container = st.container()

    # Déclenche éventuellement le pipeline (et, dans ce cas, st.rerun()).
    with prompt_container:
        prompt_box.render(placeholders)

    # ------------------------------------------------------------------
    # À partir d'ici : uniquement les passes SANS lancement de pipeline
    # (le st.rerun() de prompt_box.py interrompt la passe de lancement).
    # ------------------------------------------------------------------

    with overview_container:
        _section_title("Execution overview")
        statistics.render(session)

    # Ré-affichage des cartes si une session existante est rouverte
    # (historique) plutôt qu'une conversation tout juste exécutée.
    if session is not None and not st.session_state.get("is_running", False):
        if session.plan:
            placeholders["planner"].markdown(
                agent_cards.render_planner_card(session.plan), unsafe_allow_html=True
            )
        last_codeur = next(
            (m.content for m in reversed(session.messages) if m.source == "CodeurAgent"), None
        )
        if last_codeur:
            placeholders["codeur"].markdown(
                agent_cards.render_codeur_card(last_codeur), unsafe_allow_html=True
            )
        # Contrairement à Planner/Codeur/Reviewer/Docker (relus depuis
        # `session`), le dernier résultat ML n'est pas porté par
        # `utils/state.py` -- il est stocké par orchestrator.py dans
        # st.session_state["last_ml_risk_content"] (voir orchestrator.py).
        # Sans ce bloc, la carte ML disparaissait à chaque st.rerun() (ex:
        # juste après la fin du pipeline, dans prompt_box.py) car un
        # placeholder["ml_risk"] flambant neuf est recréé ci-dessus sans
        # jamais être re-rempli.
        last_ml_risk_content = st.session_state.get("last_ml_risk_content")
        if last_ml_risk_content is not None:
            placeholders["ml_risk"].markdown(
                agent_cards.render_ml_risk_card(last_ml_risk_content), unsafe_allow_html=True
            )
        last_reviewer = next(
            (m.content for m in reversed(session.messages) if m.source == "ReviewerAgent"), None
        )
        # CORRECTIF (conservé) : `if last_reviewer:` (test de vérité)
        # traitait un contenu vide ("") comme "aucun Reviewer" et sautait
        # le rendu, alors que le message existait bien dans
        # session.messages. `is not None` restaure la carte dès que le
        # message existe, quel que soit son contenu -- même comportement
        # que pour Docker (`if session.docker_output is not None or ...`)
        # juste en-dessous, et pour ml_risk juste au-dessus.
        if last_reviewer is not None:
            placeholders["reviewer"].markdown(
                # `session` est passée ici (et pas pendant le streaming) :
                # elle apporte approved / iterations_used, donc la carte
                # affiche l'état final réel plutôt qu'un parsing de texte.
                agent_cards.render_reviewer_card(last_reviewer, session),
                unsafe_allow_html=True,
            )

        # ---------- GUIRLANDES DE SUCCÈS (persistantes) ----------
        # `render_success_garland` ne rend RIEN si `session.approved`
        # n'est pas strictement True (donc rien si False, rien si None).
        # Comme ce bloc est ré-exécuté à CHAQUE rerun, les guirlandes
        # restent affichées tant que le résultat est approuvé -- y compris
        # pendant et après l'exécution Docker.
        garland_html = agent_cards.render_success_garland(session)
        if garland_html:
            garland_ph.markdown(garland_html, unsafe_allow_html=True)

        if session.docker_output is not None or session.docker_error_message is not None:
            placeholders["docker"].markdown(
                agent_cards.render_docker_card(session), unsafe_allow_html=True
            )

    with timeline_container:
        _section_title("Timeline")
        timeline.render(session)

    docker_logs.render(session)
    export.render(session)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()