# streamlit_app/components/hero.py
"""
Hero premium de la page d'accueil : titre, sous-titre, halos animés,
badges de statut. Signature `render()` sans argument, identique à l'appel
déjà fait par app.py -- aucune modification d'app.py nécessaire de ce
côté.

Pur affichage : aucune logique métier. Les badges Azure OpenAI et Docker
affichent leur ÉTAT RÉEL, lu via `components.sidebar` :
  - `sidebar.azure_configured()`  -> présence des variables d'environnement
  - `sidebar.docker_reachable()`  -> ping Docker MIS EN CACHE dans
    st.session_state["docker_reachable"]
Le hero ne déclenche donc jamais de ping supplémentaire : il réutilise le
résultat déjà calculé par la sidebar (rendue avant lui dans app.py), et le
bouton "Actualiser" de la sidebar rafraîchit les deux d'un coup.

L'import de `sidebar` est fait à l'intérieur de `render()` (import tardif)
pour éviter toute dépendance d'ordre d'import entre les deux modules.
"""
from __future__ import annotations

import textwrap

import streamlit as st


def _flat(s: str) -> str:
    """Voir components/agent_cards.py::_flat -- même contrainte CommonMark
    (Streamlit interprète un bloc HTML indenté ou contenant une ligne vide
    comme un bloc de code littéral plutôt que du HTML brut)."""
    dedented = textwrap.dedent(s)
    lines = [line for line in dedented.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def _status_badge(label: str, ok: bool, ok_text: str, ko_text: str) -> str:
    dot_class = "sv-dot-green" if ok else "sv-dot-red"
    badge_class = "sv-hero-badge-on" if ok else "sv-hero-badge-off"
    text = ok_text if ok else ko_text
    return (
        f'<span class="sv-hero-badge {badge_class}">'
        f'<span class="sv-dot {dot_class}"></span>{label} · {text}'
        f"</span>"
    )


def _real_statuses() -> tuple[bool, bool]:
    """
    Retourne (azure_ok, docker_ok). Défensif : en cas d'indisponibilité de
    `components.sidebar`, on retombe sur le cache brut de session_state,
    puis sur False -- jamais d'exception, jamais de ping supplémentaire.
    """
    try:
        from components import sidebar
        return sidebar.azure_configured(), sidebar.docker_reachable()
    except Exception:
        try:
            return False, bool(st.session_state.get("docker_reachable", False))
        except Exception:
            return False, False


def render() -> None:
    azure_ok, docker_ok = _real_statuses()

    azure_badge = _status_badge("⚡ Azure OpenAI", azure_ok, "Connecté", "Non configuré")
    docker_badge = _status_badge("🐳 Docker", docker_ok, "Connecté", "Déconnecté")

    html = _flat(f"""
    <div class="sv-hero">
        <div class="sv-hero-glow sv-hero-glow-a"></div>
        <div class="sv-hero-glow sv-hero-glow-b"></div>
        <div class="sv-hero-glow sv-hero-glow-c"></div>
        <div class="sv-hero-content">
            <div class="sv-hero-eyebrow">◆ AI Multi-Agent Studio</div>
            <h1 class="sv-hero-title">Smartovate Multi-Agent AI Studio</h1>
            <p class="sv-hero-subtitle">
                Ingénierie logicielle multi-agents autonome, propulsée par Azure OpenAI —
                planification, génération, prédiction de risque, revue et exécution isolée
                d'un seul flux.
            </p>
            <div class="sv-hero-badges">
                {azure_badge}
                {docker_badge}
                <span class="sv-hero-badge">🔁 Revue automatique du code</span>
                <span class="sv-hero-badge">🧠 ML Risk Predictor</span>
            </div>
        </div>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)