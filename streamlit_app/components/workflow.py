# streamlit_app/components/workflow.py
"""
Visualisation temps réel du pipeline (Planner -> Codeur -> ML Risk
Predictor -> Reviewer -> Docker -> Terminé), stepper "Étape X/5",
particule lumineuse de flux et bannière de fin de pipeline.

Signature strictement compatible avec l'appel déjà existant dans
utils/orchestrator.py (`workflow.render_html(etape)`, un seul argument
positionnel). Le second paramètre `session` est OPTIONNEL : orchestrator.py
continue d'appeler la fonction avec un seul argument sans aucune
modification, et app.py peut lui passer la session pour affiner l'état.

`etape` (str | None), tel qu'assigné par orchestrator.py à
st.session_state["etape_active"] :
    None                -> rien n'a encore démarré (tout "en attente")
    "planner"           -> le PlannerAgent travaille
    "codeur_reviewer"   -> la boucle Codeur<->Reviewer est en cours
    "docker"            -> exécution Docker en cours
    "done"              -> pipeline terminé (bannière de succès)
    "error"             -> le pipeline s'est interrompu sur une erreur
                           (désormais réellement émis par orchestrator.py,
                           voir son bloc `if event.kind == "error"`)

------------------------------------------------------------------------
Limites connues (honnêtes, car ce composant ne peut pas deviner ce que
l'orchestrateur ne lui transmet pas) :
------------------------------------------------------------------------
1. "codeur_reviewer" ne distingue pas si c'est le tour du Codeur ou du
   Reviewer. On tente un raffinement best-effort via `utils.state` (en
   regardant la source du dernier message reçu) ; si cette info n'est pas
   disponible ou son interface diffère, on affiche simplement les trois
   noeuds du groupe comme actifs simultanément -- ne casse jamais le rendu.
2. En état "error", orchestrator.py ne dit PAS quel agent a échoué : il
   affiche seulement le message sur le placeholder "codeur". On applique
   donc le même raffinement best-effort que ci-dessus pour marquer le
   noeud fautif, avec repli sur "codeur" (le placeholder réellement
   utilisé par l'orchestrateur). Aucun état n'est inventé au-delà de ça :
   les noeuds non atteints restent "waiting".
3. Les descriptions des noeuds (_DESCRIPTIONS) sont des LIBELLÉS
   D'INTERFACE statiques, validés produit. Ils ne proviennent d'aucune
   donnée métier et n'en modifient aucune.
"""
from __future__ import annotations

import html as html_mod
import textwrap

_STAGES = ["planner", "codeur", "ml_risk", "reviewer", "docker"]
_LABELS = {
    "planner": "Planner",
    "codeur": "Codeur",
    "ml_risk": "ML Risk",
    "reviewer": "Reviewer",
    "docker": "Docker",
}
# Libellés d'interface statiques (validés) -- aucune logique métier.
_DESCRIPTIONS = {
    "planner": "Décompose la tâche",
    "codeur": "Génère le code",
    "ml_risk": "Estime le risque de révision",
    "reviewer": "Valide ou demande une révision",
    "docker": "Exécute en environnement isolé",
}
_ICONS = {"planner": "🧭", "codeur": "🧑‍💻", "ml_risk": "🧠", "reviewer": "🔍", "docker": "🐳"}

# Libellé affiché sous chaque noeud, pour les 4 états demandés.
_STATE_LABELS = {
    "waiting": "Waiting",
    "active": "Running",
    "done": "Completed",
    "failed": "Error",
}

# etape (fourni par orchestrator.py) -> index des stages actifs.
# NOTE : orchestrator.py ne distingue toujours, au niveau de sa machine à
# états `etape_active`, que 3 grandes phases (planner / codeur_reviewer /
# docker) -- le calcul ML se produit À L'INTÉRIEUR de la phase
# "codeur_reviewer" (entre un tour du Codeur et le tour suivant du
# Reviewer, voir backend.obtenir_flux_code_review_avec_ml), sans phase
# `etape_active` dédiée. Le noeud "ml_risk" est donc marqué actif en même
# temps que "codeur" et "reviewer" pendant cette phase, comme c'était déjà
# le cas pour ces deux noeuds avant l'ajout du ML.
_ACTIVE_INDEXES = {
    None: [],
    "planner": [0],
    "codeur_reviewer": [1, 2, 3],
    "docker": [4],
    "done": [],
    "error": [],
    "failed": [],
}

# Position horizontale du centre de chaque noeud, en % de la largeur du
# bloc -- alignée sur les bornes de `.sv-trace` (left:9% / right:9%).
# Sert à borner le trajet de la particule lumineuse ET la largeur de la
# barre de progression.
_NODE_CENTERS = [9.0, 29.5, 50.0, 70.5, 91.0]

# Bornes de la ligne de connexion : elle démarre au centre du PREMIER
# noeud (Planner) et se termine au centre du DERNIER (Docker). La barre
# `.sv-trace-progress` étant positionnée en `left:_TRACE_START%`, sa
# largeur doit être exprimée dans ce même repère et ne peut donc jamais
# dépasser `_TRACE_SPAN` -- sans quoi elle déborderait à droite du noeud
# Docker et du conteneur (cas historique : width:100% depuis left:9%,
# soit une fin à 109%).
_TRACE_START = _NODE_CENTERS[0]
_TRACE_END = _NODE_CENTERS[-1]
_TRACE_SPAN = _TRACE_END - _TRACE_START


def _flat(s: str) -> str:
    """Voir components/agent_cards.py::_flat -- même contrainte CommonMark."""
    dedented = textwrap.dedent(s)
    non_empty_lines = [line for line in dedented.splitlines() if line.strip()]
    return "\n".join(non_empty_lines).strip()


def _animations_enabled() -> bool:
    """
    Lit le toggle "Animations" de la sidebar
    (st.session_state["settings"]["animations"], écrit par sidebar.py).
    Défaut : True. Défensif : ne lève jamais, y compris hors contexte
    Streamlit (tests unitaires, import isolé).
    """
    try:
        import streamlit as st
        return bool(st.session_state.get("settings", {}).get("animations", True))
    except Exception:
        return True


def _try_refine_codeur_reviewer_turn(session=None) -> str | None:
    """
    Best-effort : détermine si c'est le tour du Codeur ou du Reviewer en
    regardant le dernier message de la session courante. Ne lève jamais
    d'exception -- retourne None si l'info n'est pas disponible.

    `session` peut être fourni par l'appelant (app.py) ; sinon on retombe
    sur `utils.state.session_courante()` comme avant.
    """
    try:
        if session is None:
            from utils import state
            session = state.session_courante()
        messages = getattr(session, "messages", None) if session else None
        if not messages:
            return None
        last_source = getattr(messages[-1], "source", None)
        if last_source == "CodeurAgent":
            return "reviewer"
        if last_source == "ReviewerAgent":
            return "codeur"
    except Exception:
        return None
    return None


def compute_statuses(etape: str | None, session=None) -> list[str]:
    """
    Retourne le statut des 5 noeuds, dans l'ordre de `_STAGES`, parmi
    'waiting' | 'active' | 'done' | 'failed'.

    Fonction PUBLIQUE : elle est également utilisée par components/sidebar.py
    pour que le mini-diagramme de la barre latérale affiche exactement le
    même état que le pipeline principal (source unique de vérité, aucune
    logique dupliquée).

    Aucun état n'est inventé : tout est dérivé de `etape` (assigné par
    orchestrator.py) et, pour le seul raffinement Codeur/Reviewer, de la
    source du dernier message réellement reçu.
    """
    if etape in ("error", "failed"):
        return _error_statuses(session)

    all_done = etape == "done"
    active_indexes = _ACTIVE_INDEXES.get(etape, [])
    refine = _try_refine_codeur_reviewer_turn(session) if etape == "codeur_reviewer" else None

    statuses: list[str] = []
    for i, key in enumerate(_STAGES):
        if all_done:
            statuses.append("done")
        elif i in active_indexes:
            if refine is not None and key in ("codeur", "reviewer"):
                statuses.append("active" if key == refine else "done")
            else:
                statuses.append("active")
        elif active_indexes and i < min(active_indexes):
            statuses.append("done")
        else:
            statuses.append("waiting")
    return statuses


def _error_statuses(session=None) -> list[str]:
    """
    État ERROR. orchestrator.py n'indique pas QUEL agent a échoué -- il
    signale l'erreur sur le placeholder "codeur". On marque donc en
    'failed' le noeud identifié par le raffinement best-effort
    (dernier message reçu), avec repli sur "codeur". Les noeuds situés
    avant sont 'done' (ils ont réellement produit un message), ceux
    situés après restent 'waiting' : ils n'ont jamais été atteints.
    """
    refine = _try_refine_codeur_reviewer_turn(session)
    failed_key = refine if refine in ("codeur", "reviewer") else "codeur"
    failed_index = _STAGES.index(failed_key)

    plan_present = bool(getattr(session, "plan", None)) if session is not None else True

    statuses: list[str] = []
    for i, _key in enumerate(_STAGES):
        if i == failed_index:
            statuses.append("failed")
        elif i < failed_index:
            # Le Planner n'est marqué 'done' que si un plan existe vraiment.
            if i == 0 and not plan_present:
                statuses.append("failed")
            else:
                statuses.append("done")
        else:
            statuses.append("waiting")
    return statuses


def _node_html(key: str, status: str) -> str:
    """
    status: 'waiting' | 'active' | 'done' | 'failed'.
    Le noeud "done" conserve son icône d'origine (un petit badge vert ✓ est
    superposé en CSS -- voir .sv-node-done .sv-node-dot::after) : la
    structure des 5 étapes (icône + nom + description + état) reste donc
    strictement identique dans tous les états, y compris une fois le
    pipeline terminé, ce qui évite tout changement de disposition.
    """
    icon = _ICONS[key]
    label = _LABELS[key]
    description = _DESCRIPTIONS[key]
    state_label = _STATE_LABELS[status]
    display_icon = icon if status != "failed" else "✕"
    return (
        f'<div class="sv-node sv-node-{status}">'
        f'<span class="sv-node-dot">{display_icon}</span>'
        f'<span class="sv-node-name">{html_mod.escape(label)}</span>'
        f'<span class="sv-node-desc">{html_mod.escape(description)}</span>'
        f'<span class="sv-node-state">{state_label}</span>'
        f"</div>"
    )


def _spark_html(statuses: list[str], animations: bool) -> str:
    """
    Particule lumineuse circulant sur la trace, UNIQUEMENT quand une étape
    est réellement en cours (au moins un noeud 'active') et que les
    animations sont activées. Le trajet est borné au segment réellement
    concerné : du noeud terminé précédent jusqu'au dernier noeud actif.
    """
    if not animations:
        return ""
    active_indexes = [i for i, s in enumerate(statuses) if s == "active"]
    if not active_indexes:
        return ""
    first_active = min(active_indexes)
    last_active = max(active_indexes)
    start = _NODE_CENTERS[first_active - 1] if first_active > 0 else 2.0
    end = _NODE_CENTERS[last_active]
    return (
        f'<span class="sv-flow-spark" style="--sv-spark-from:{start}%;--sv-spark-to:{end}%;">✦</span>'
    )


def render_html(etape: str | None, session=None) -> str:
    """
    `session` est OPTIONNEL -- l'appel historique `render_html(etape)` de
    utils/orchestrator.py reste valide tel quel.
    """
    statuses = compute_statuses(etape, session)
    all_done = etape == "done"
    is_error = etape in ("error", "failed")
    animations = _animations_enabled()

    nodes_html = "".join(_node_html(key, status) for key, status in zip(_STAGES, statuses))
    spark_html = _spark_html(statuses, animations)

    # Progression : dérivée des statuts réellement calculés ci-dessus.
    # La largeur est exprimée dans le repère de la trace (origine
    # _TRACE_START), et non en pourcentage du conteneur : la barre
    # s'arrête donc EXACTEMENT sur le centre du noeud le plus avancé, et
    # au maximum sur celui de Docker (_TRACE_SPAN). Aucun débordement
    # possible à droite, quelle que soit la largeur d'écran.
    advanced = [i for i, s in enumerate(statuses) if s in ("active", "done", "failed")]
    if advanced:
        progress_pct = round(_NODE_CENTERS[max(advanced)] - _TRACE_START, 1)
    else:
        progress_pct = 0

    # Stepper "Étape X/5" (5 noeuds depuis l'ajout de ML Risk Predictor)
    active_indexes = [i for i, s in enumerate(statuses) if s == "active"]
    if all_done:
        step_number = 5
    elif is_error:
        failed = [i for i, s in enumerate(statuses) if s == "failed"]
        step_number = (failed[0] + 1) if failed else 0
    elif active_indexes:
        step_number = min(active_indexes) + 1
    else:
        step_number = 0

    if step_number:
        step_text = f"Étape {step_number}/5 — {_LABELS[_STAGES[step_number - 1]]}"
    else:
        step_text = "En attente d'une demande"

    # Les pills sont purement additives sur la même ligne que le libellé
    # fixe "PIPELINE D'EXÉCUTION" -- elles n'ajoutent aucune hauteur au
    # bloc. Les 5 noeuds restent affichés en permanence, quel que soit
    # `etape`, avec leur ligne de connexion horizontale.
    if all_done:
        status_pill_html = '<span class="sv-workflow-done-pill">✅ Terminé</span>'
    elif is_error:
        status_pill_html = '<span class="sv-workflow-error-pill">✕ Interrompu</span>'
    else:
        status_pill_html = ""

    error_banner_html = (
        '<div class="sv-pipeline-error"><span>⚠</span>'
        "<span>Le pipeline s'est interrompu — voir le détail de l'erreur sur la carte concernée.</span>"
        "</div>"
        if is_error
        else ""
    )

    anim_class = "" if animations else " sv-no-anim"

    return _flat(f"""
    <div class="sv-card sv-workflow-card{anim_class}">
        <div class="sv-workflow-header">
            <span class="sv-workflow-eyebrow">Pipeline d'exécution</span>
            <span class="sv-workflow-step">{html_mod.escape(step_text)}{status_pill_html}</span>
        </div>
        <div class="sv-workflow">
            <div class="sv-trace"></div>
            <div class="sv-trace-progress" style="width:{progress_pct}%"></div>
            {spark_html}
            {nodes_html}
        </div>
        {error_banner_html}
    </div>
    """)