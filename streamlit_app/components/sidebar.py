# streamlit_app/components/sidebar.py
"""
Barre latérale modernisée : logo, statut Azure/Docker, modèle courant,
mini-diagramme d'architecture À ÉTAT RÉEL, historique des conversations,
réglages, mode développeur. Signature `render()` sans argument, identique
à l'appel déjà fait par app.py.

Aucune logique métier :
  - le statut Azure affiché est un indicateur de CONFIGURATION (présence
    des variables d'environnement), pas une vérification réseau active ;
  - le statut Docker est une vérification légère (`client.ping()`), mais
    désormais MISE EN CACHE dans st.session_state["docker_reachable"].
    Auparavant le ping était refait à CHAQUE rerun (et prompt_box.py en
    déclenche un après chaque pipeline), ajoutant jusqu'à ~1 s à chaque
    rechargement de page. Un bouton "Actualiser" permet de le recalculer
    manuellement à la demande.

`utils.state` (session/historique) est utilisé de façon défensive
(try/except) : en son absence ou en cas d'interface différente, la sidebar
reste fonctionnelle avec un historique vide plutôt que de faire planter
l'application.

Le mini-diagramme du pipeline réutilise `workflow.compute_statuses()` :
c'est la MÊME source de vérité que le pipeline principal (aucune logique
d'état dupliquée, aucun état inventé).
"""
from __future__ import annotations

import os
import textwrap

import streamlit as st

from components import workflow

# Ordre et libellés strictement alignés sur workflow._STAGES (5 étapes,
# ML Risk inclus -- il manquait dans l'ancien diagramme).
_ARCH_STEPS = [
    ("planner", "🧭", "Planner"),
    ("codeur", "🧑‍💻", "Codeur"),
    ("ml_risk", "🧠", "ML Risk"),
    ("reviewer", "🔍", "Reviewer"),
    ("docker", "🐳", "Docker"),
]

_ARCH_STATE_LABELS = {
    "waiting": "—",
    "active": "RUN",
    "done": "✓",
    "failed": "✕",
}


def _flat(s: str) -> str:
    dedented = textwrap.dedent(s)
    lines = [line for line in dedented.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def azure_configured() -> bool:
    """Public : réutilisé par components/hero.py pour ses badges d'état."""
    return bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"))


# Alias conservé pour compatibilité avec un éventuel appel existant.
_azure_configured = azure_configured


def _docker_ping() -> bool:
    """Vérification légère (non bloquante) : tente une requête au démon Docker
    avec un délai très court, sans jamais faire échouer le rendu de la sidebar."""
    try:
        import docker
        client = docker.from_env(timeout=1)
        client.ping()
        return True
    except Exception:
        return False


def docker_reachable(force_refresh: bool = False) -> bool:
    """
    Public : statut Docker MIS EN CACHE dans
    st.session_state["docker_reachable"].

    Le ping n'est effectué qu'une seule fois par session (ou sur demande
    explicite via `force_refresh=True`, déclenché par le bouton
    "Actualiser"). Réutilisé par components/hero.py, ce qui garantit que
    le hero et la sidebar affichent EXACTEMENT le même état sans doubler
    le coût réseau.
    """
    if force_refresh or "docker_reachable" not in st.session_state:
        st.session_state["docker_reachable"] = _docker_ping()
    return bool(st.session_state["docker_reachable"])


# Alias conservé pour compatibilité.
_docker_reachable = docker_reachable


def _status_row(label: str, ok: bool) -> str:
    dot_class = "sv-dot-green" if ok else "sv-dot-red"
    text = "Connecté" if ok else "Déconnecté"
    value_class = "sv-status-value-ok" if ok else "sv-status-value-ko"
    return (
        f'<div class="sv-status-row">'
        f'<span><span class="sv-dot {dot_class}"></span>{label}</span>'
        f'<span class="{value_class}">{text}</span>'
        f"</div>"
    )


def _get_history():
    """Retourne la liste des sessions d'historique, ou [] si indisponible."""
    try:
        conversations = st.session_state.get("conversations", {})
        current_id = st.session_state.get("current_id")
        items = []
        for sid, session in (conversations or {}).items():
            title = getattr(session, "title", None) or getattr(session, "prompt", "Conversation")[:32]
            items.append((sid, title, sid == current_id))
        return items
    except Exception:
        return []


def _arch_html() -> str:
    """
    Mini-diagramme du pipeline avec l'ÉTAT RÉEL de chaque étape, dérivé de
    `workflow.compute_statuses(etape_active)` -- la même fonction que
    celle qui pilote le pipeline principal. Aucun état n'est recalculé ni
    inventé ici.
    """
    etape = st.session_state.get("etape_active")
    try:
        session = None
        try:
            from utils import state
            session = state.session_courante()
        except Exception:
            session = None
        statuses = workflow.compute_statuses(etape, session)
    except Exception:
        # Filet de sécurité : la sidebar ne doit jamais casser le rendu.
        statuses = ["waiting"] * len(_ARCH_STEPS)

    parts = []
    for i, (key, icon, label) in enumerate(_ARCH_STEPS):
        status = statuses[i] if i < len(statuses) else "waiting"
        state_label = _ARCH_STATE_LABELS.get(status, "—")
        parts.append(
            f'<div class="sv-arch-step sv-arch-step-{status}">'
            f"<span>{icon}</span><span>{label}</span>"
            f'<span class="sv-arch-state">{state_label}</span>'
            f"</div>"
        )
        if i < len(_ARCH_STEPS) - 1:
            parts.append('<div class="sv-arch-arrow">↓</div>')

    return f'<div class="sv-sidebar-arch">{"".join(parts)}</div>'


def render() -> None:
    with st.sidebar:
        st.markdown(
            _flat(
                """
                <div class="sv-logo">
                    <span class="sv-logo-mark">◆</span>
                    <span>Smartovate</span>
                </div>
                <div class="sv-tagline">AI Multi-Agent Studio</div>
                """
            ),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sv-sidebar-section-title">Statut</div>', unsafe_allow_html=True)
        st.markdown(_status_row("Azure OpenAI", azure_configured()), unsafe_allow_html=True)
        st.markdown(_status_row("Docker", docker_reachable()), unsafe_allow_html=True)
        # Le ping Docker étant désormais mis en cache, on offre un
        # rafraîchissement manuel explicite.
        if st.button("↻ Actualiser le statut Docker", use_container_width=True, key="sidebar_refresh_docker"):
            docker_reachable(force_refresh=True)
            st.rerun()

        st.markdown('<div class="sv-sidebar-section-title">Modèle</div>', unsafe_allow_html=True)
        model_name = os.getenv("AZURE_OPENAI_MODEL", "gpt-5")
        st.markdown(
            _flat(f"""
            <div class="sv-sidebar-model-card">
                <div class="sv-sidebar-model-name">{model_name}</div>
                <div style="font-size:0.75rem;color:var(--color-subtitle);">Azure OpenAI</div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sv-sidebar-section-title">Pipeline</div>', unsafe_allow_html=True)
        st.markdown(_arch_html(), unsafe_allow_html=True)

        st.markdown('<div class="sv-sidebar-section-title">Historique</div>', unsafe_allow_html=True)
        history = _get_history()
        if not history:
            st.markdown(
                '<div class="sv-history-item" style="opacity:0.6;cursor:default;">Aucune conversation pour le moment</div>',
                unsafe_allow_html=True,
            )
        else:
            for sid, title, is_active in reversed(history):
                prefix = "▸" if is_active else "💬"
                if st.button(f"{prefix} {title}", key=f"hist_{sid}", use_container_width=True):
                    st.session_state["current_id"] = sid
                    st.rerun()

        col_new, col_clear = st.columns(2)
        with col_new:
            if st.button("＋ Nouvelle", use_container_width=True, key="sidebar_new_conv"):
                st.session_state["current_id"] = None
                st.session_state["prompt_input"] = ""
                st.rerun()
        with col_clear:
            if st.button("🗑 Effacer", use_container_width=True, key="sidebar_clear_history"):
                st.session_state["conversations"] = {}
                st.session_state["current_id"] = None
                st.rerun()

        st.markdown('<div class="sv-sidebar-section-title">Réglages</div>', unsafe_allow_html=True)
        settings = st.session_state.get("settings", {})
        st.session_state.setdefault("developer_mode", False)
        st.toggle("Mode développeur", key="developer_mode")
        max_it = st.slider("Itérations maximales", min_value=1, max_value=10,
                            value=settings.get("max_iterations", 5), key="sidebar_max_iterations")
        # Ce toggle est désormais RÉELLEMENT appliqué : workflow.py et
        # agent_cards.py le lisent via st.session_state["settings"]
        # ["animations"]. OFF = aucune animation et aucune guirlande
        # animée, mais TOUS les états et informations restent affichés.
        animations = st.toggle(
            "Animations",
            value=settings.get("animations", True),
            key="sidebar_animations",
            help="Désactive les animations du pipeline et les guirlandes de succès. Les états restent affichés.",
        )
        st.session_state["settings"] = {
            **settings,
            "max_iterations": max_it,
            "animations": animations,
        }
        # Compatibilité avec orchestrator.py qui lit directement cette clé
        # (voir utils/orchestrator.py : st.session_state.get("max_iterations", 5))
        st.session_state["max_iterations"] = max_it