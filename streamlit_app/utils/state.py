# streamlit_app/utils/state.py
"""
Point d'entrée UNIQUE pour manipuler st.session_state.

Aucun composant ne doit lire/écrire st.session_state["xyz"] directement avec
des clés en dur -- tout passe par les fonctions de ce module. Cela évite les
bugs classiques de collisions de clés dans les gros projets Streamlit.
"""
from __future__ import annotations

import uuid
from datetime import datetime

import streamlit as st

from utils.models import ConversationSession


def init_state() -> None:
    """Initialise toutes les clés de session_state si elles n'existent pas."""
    defaults = {
        "sessions": {},              # dict[str, ConversationSession]
        "current_session_id": None,
        "etape_active": None,        # "planner" | "codeur_reviewer" | "docker" | "done"
        "is_running": False,
        "dev_mode": False,
        "max_iterations": 5,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def nouvelle_session(prompt: str) -> ConversationSession:
    """Crée et active une nouvelle session de conversation."""
    session_id = str(uuid.uuid4())
    session = ConversationSession(id=session_id, prompt=prompt, created_at=datetime.now())
    st.session_state["sessions"][session_id] = session
    st.session_state["current_session_id"] = session_id
    return session


def session_courante() -> ConversationSession | None:
    """Retourne la session actuellement affichée, ou None."""
    sid = st.session_state.get("current_session_id")
    if sid is None:
        return None
    return st.session_state["sessions"].get(sid)


def demarrer_nouvelle_conversation() -> None:
    """Réinitialise l'écran pour permettre une nouvelle demande."""
    st.session_state["current_session_id"] = None
    st.session_state["etape_active"] = None
    st.session_state["is_running"] = False


def effacer_historique() -> None:
    """Supprime toutes les conversations enregistrées."""
    st.session_state["sessions"] = {}
    st.session_state["current_session_id"] = None
