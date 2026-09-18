# streamlit_app/components/agent_cards.py
"""
Cartes visuelles des 5 agents : Planner (cyan), Codeur (violet, coloration
syntaxique + bouton Copier), ML Risk Predictor (violet clair), Reviewer
(rose, statut d'approbation), Docker (cyan, logs / terminal).

Chaque fonction retourne du HTML prêt à afficher via st.markdown(...,
unsafe_allow_html=True) -- aucune logique métier ici, uniquement du rendu à
partir de données déjà produites par utils/orchestrator.py.

------------------------------------------------------------------------
Rappel Issue 1 (cause racine confirmée) -- HTML affiché en texte brut /
entités visibles (&quot;, &#x27;...)
------------------------------------------------------------------------
Streamlit restitue le Markdown via un moteur CommonMark. Une ligne DÉBUTANT
un bloc et indentée de 4 espaces ou plus est interprétée comme un BLOC DE
CODE LITTÉRAL (donc affiché échappé, tel quel) -- et NON comme du HTML brut,
même avec `unsafe_allow_html=True`. Or chaque template ci-dessous est écrit
comme une f-string multi-lignes à l'intérieur d'une fonction indentée : sans
précaution, la première ligne du HTML retourné hérite de cette indentation
Python, et Streamlit l'affiche donc comme du texte échappé plutôt que comme
du HTML.

Vérifié avec un vrai parseur CommonMark (markdown-it-py) : il suffit que la
PREMIÈRE ligne du bloc démarre à la colonne 0 pour que tout le bloc soit
reconnu comme HTML brut -- les lignes internes peuvent rester indentées
sans aucun problème (le bloc HTML, une fois reconnu, continue jusqu'à la
prochaine ligne vide, quelle que soit l'indentation interne).

`_flat()` ci-dessous applique cette règle systématiquement : TOUJOURS
l'utiliser en sortie de chaque fonction de rendu de ce module.
------------------------------------------------------------------------
Rappel Issue "[object Object]" (résolue) -- `_coerce_to_text` garantit
qu'aucune valeur non-string ne peut être interpolée dans le HTML, et
`_render_copy_button` utilise un double échappement JSON+HTML pour le
bouton Copier (jamais de fuite d'objet vers un contexte JS).
------------------------------------------------------------------------
Guirlandes de succès (§5) -- source de vérité
------------------------------------------------------------------------
La décoration de succès est désormais pilotée par `session.approved`
(le booléen produit par `backend.construire_resultat()` et stocké par
orchestrator.py), et NON plus par un parsing du texte du Reviewer.
Voir `render_success_garland()` : elle ne rend RIEN tant que
`session.approved` n'est pas strictement `True`.

Le parsing textuel (`_parse_reviewer_status`) est conservé intact : il
reste utilisé pour le badge de la carte, et comme unique source pendant
le STREAMING, moment où orchestrator.py appelle
`render_reviewer_card(content)` sans objet session (le résultat final
n'existe pas encore à cet instant). C'est exactement le comportement
actuel, préservé.
"""
from __future__ import annotations

import base64
import html
import json
import random
import re
import textwrap
from typing import Any

from utils.models import ConversationSession


def _flat(s: str) -> str:
    """
    Garantit qu'une chaîne HTML multi-lignes ne contient AUCUNE ligne
    vide (voir docstring du module) -- à appliquer sur CHAQUE valeur
    retournée par ce module avant de la passer à
    st.markdown(..., unsafe_allow_html=True).

    Deux causes distinctes de ligne vide ont été identifiées et corrigées
    ici :
      1. La ligne de tête, héritée de l'indentation Python du template
         (corrigée par dedent + strip).
      2. Une ligne interne devenue vide parce qu'une valeur interpolée
         (ex: `{copy_button}` quand il n'y a pas encore de code, ou
         `{badge}` conditionnel) est une chaîne vide -- CommonMark termine
         un bloc HTML à la première ligne vide rencontrée, donc TOUT ce
         qui suit une telle ligne serait alors ré-interprété comme un
         nouveau bloc (et, s'il est indenté, comme un bloc de code
         littéral). On supprime donc systématiquement toute ligne
         entièrement vide, où qu'elle soit dans le template.
    """
    dedented = textwrap.dedent(s)
    non_empty_lines = [line for line in dedented.splitlines() if line.strip()]
    return "\n".join(non_empty_lines).strip()


# ============================================================
# Préférence d'interface : toggle "Animations" de la sidebar
# ============================================================

def _animations_enabled() -> bool:
    """
    Lit st.session_state["settings"]["animations"] (écrit par sidebar.py).
    Défaut : True. Défensif : ne lève jamais, y compris hors contexte
    Streamlit (import isolé, tests unitaires).

    Quand le toggle est sur OFF, on n'enlève AUCUNE information ni AUCUN
    état : seules les animations CSS sont neutralisées (classe
    `sv-no-anim`) et les guirlandes animées ne sont pas rendues. Les
    bannières de statut, badges et coches restent affichés à l'identique.
    """
    try:
        import streamlit as st
        return bool(st.session_state.get("settings", {}).get("animations", True))
    except Exception:
        return True


def _anim_class() -> str:
    """Suffixe de classe à ajouter à l'élément RACINE de chaque carte."""
    return "" if _animations_enabled() else " sv-no-anim"


def _max_iterations() -> int | None:
    """
    Nombre maximal d'itérations RÉELLEMENT utilisé par le pipeline
    (st.session_state["max_iterations"], lu par utils/orchestrator.py au
    moment de créer la team). Retourne None si indisponible -- jamais de
    valeur inventée.
    """
    try:
        import streamlit as st
        value = st.session_state.get("max_iterations")
        return int(value) if value is not None else None
    except Exception:
        return None


# ============================================================
# Normalisation défensive du contenu (voir historique Issue 1)
# ============================================================

def _coerce_to_text(value: Any) -> str:
    """
    Normalise n'importe quelle valeur reçue en texte affichable. Gère une
    chaîne, un objet "message-like" (`.content`), une liste de blocs de
    contenu structurés, un dict, ou toute autre valeur (`str()` en dernier
    recours) -- rempart final contre toute fuite d'objet non-string.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if hasattr(value, "content"):
        return _coerce_to_text(value.content)
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(_coerce_to_text(item))
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        if "text" in value:
            return str(value["text"])
        if "content" in value:
            return _coerce_to_text(value["content"])
        return str(value)
    return str(value)


def _extract_summary_and_code(content: str) -> tuple[str, str | None]:
    """Sépare le court résumé du CodeurAgent et son bloc de code."""
    marker = "```python"
    idx = content.find(marker)
    if idx == -1:
        return content.strip(), None
    summary = content[:idx].strip()
    rest = content[idx + len(marker):]
    end = rest.find("```")
    code = rest[:end].strip() if end != -1 else rest.strip()
    return summary, code


def _parse_reviewer_status(content: str) -> str:
    if "STATUT: APPROVED" in content:
        return "APPROVED"
    if "STATUT: REVISION_REQUISE" in content:
        return "REVISION_REQUISE"
    return "EN_COURS"


# ============================================================
# Sections réutilisables (lisibilité, contenu long repliable)
# ============================================================

def _collapsible(title: str, body_html: str, *, open_by_default: bool = True, extra_class: str = "") -> str:
    """
    Section repliable native (<details>/<summary>, sans JS) -- utilisée
    pour séparer clairement Résumé / Code / Feedback / Logs plutôt qu'un
    unique bloc long, et pour replier automatiquement les sorties longues.
    """
    open_attr = " open" if open_by_default else ""
    return (
        f'<details class="sv-collapsible {extra_class}"{open_attr}>'
        f'<summary>{html.escape(title)}</summary>'
        f'<div class="sv-collapsible-body">{body_html}</div>'
        f"</details>"
    )


def _status_pill(label: str, css_class: str) -> str:
    return f'<span class="sv-badge {css_class}">{html.escape(label)}</span>'


# ============================================================
# PlannerAgent
# ============================================================

def render_planner_card(plan: Any, status: str = "completed") -> str:
    """
    status: "thinking" | "completed" (orchestrator.py n'appelle
    actuellement cette fonction qu'une fois le plan disponible ; le
    paramètre existe pour une future extension sans casser l'appel actuel).
    """
    text = _coerce_to_text(plan)
    is_thinking = status == "thinking" or not text

    if is_thinking:
        badge = _status_pill("Réflexion…", "sv-badge-thinking")
        body = '<div class="sv-shimmer" style="margin-bottom:0.5rem;"></div><div class="sv-shimmer" style="width:70%;"></div>'
    else:
        badge = _status_pill("✓ Plan généré", "sv-badge-approved")
        plan_html = html.escape(text).replace("\n", "<br>")
        body = _collapsible("Plan d'exécution", f'<div class="sv-plan-body">{plan_html}</div>')

    return _flat(f"""
    <div class="sv-card sv-agent-card sv-agent-planner{_anim_class()}">
        <div class="sv-agent-header">
            <span>🧭 PlannerAgent</span>
            {badge}
        </div>
        <p class="sv-summary">Décompose la tâche en étapes exploitables par le CodeurAgent.</p>
        {body}
    </div>
    """)


# ============================================================
# CodeurAgent
# ============================================================

# ============================================================
# Éditeur de code façon VS Code (coloration, numéros de ligne)
# ============================================================

_PY_KEYWORDS = [
    "def", "return", "if", "elif", "else", "for", "while", "import", "from",
    "class", "try", "except", "finally", "with", "as", "pass", "break",
    "continue", "raise", "yield", "lambda", "None", "True", "False", "and",
    "or", "not", "in", "is", "async", "await", "global", "nonlocal", "del",
    "assert", "self",
]

_TOKEN_RE = re.compile(
    r"(?P<comment>#[^\n]*)"
    r"|(?P<string>'''.*?'''|\"\"\".*?\"\"\"|'[^'\n]*'|\"[^\"\n]*\")"
    r"|(?P<number>\b\d+(?:\.\d+)?\b)"
    r"|(?P<keyword>\b(?:" + "|".join(_PY_KEYWORDS) + r")\b)",
    re.S,
)


def _highlight_python(code: str) -> str:
    """
    Coloration syntaxique légère, fondée sur une seule passe d'expressions
    régulières (commentaires, chaînes, nombres, mots-clés) -- volontairement
    PAS un tokenizer complet du langage Python. Suffisant pour la lisibilité
    visuelle demandée (style éditeur), sans la complexité d'un vrai
    analyseur syntaxique. Chaque segment de texte NON reconnu est échappé
    normalement (html.escape) ; seuls les segments reconnus sont enveloppés
    dans un <span> coloré, également échappés.
    """
    out: list[str] = []
    last = 0
    for m in _TOKEN_RE.finditer(code):
        out.append(html.escape(code[last:m.start()]))
        text = html.escape(m.group(0))
        css_class = {"comment": "sv-com", "string": "sv-str", "number": "sv-num", "keyword": "sv-kw"}[m.lastgroup]
        out.append(f'<span class="{css_class}">{text}</span>')
        last = m.end()
    out.append(html.escape(code[last:]))
    return "".join(out)


def _render_code_editor(code: str) -> str:
    """
    Tableau HTML à deux colonnes (numéro de ligne / contenu coloré),
    construit sans aucun retour à la ligne au niveau du HTML généré
    (les lignes du code sont représentées par des <tr> successifs, pas par
    des "\\n" littéraux) -- compatible avec la contrainte `_flat` (voir
    docstring du module) puisqu'aucune ligne vide n'est introduite dans le
    template global.
    """
    lines = code.split("\n")
    highlighted = _highlight_python(code).split("\n")
    if len(highlighted) != len(lines):
        # Filet de sécurité : ne devrait jamais se produire (la coloration
        # ne modifie jamais le nombre de sauts de ligne), mais on préfère
        # un rendu simplement échappé à un décalage numéro/contenu erroné.
        highlighted = [html.escape(l) for l in lines]

    rows = []
    for i, line_html in enumerate(highlighted, start=1):
        rows.append(
            f'<tr><td class="sv-code-line-num">{i}</td>'
            f'<td class="sv-code-line-content">{line_html if line_html else " "}</td></tr>'
        )
    return f'<table class="sv-code-table">{"".join(rows)}</table>'


def _render_download_button(code: str, filename: str = "generated_script.py") -> str:
    """
    Bouton de téléchargement autonome (lien HTML avec URI de données en
    base64) -- ne nécessite aucun script ni aucun aller-retour avec le
    serveur Streamlit, cohérent avec la contrainte "design uniquement".
    """
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    return (
        f'<a class="sv-download-btn" download="{html.escape(filename)}" '
        f'href="data:text/x-python;base64,{encoded}">⬇ Télécharger</a>'
    )


def render_codeur_card(content: Any, status: str = "completed") -> str:
    """
    status: "generating" | "completed" -- déduit automatiquement si aucun
    bloc de code n'est encore présent dans `content` (streaming en cours).
    """
    text = _coerce_to_text(content)
    summary, code = _extract_summary_and_code(text)
    summary_html = html.escape(summary) if summary else ""

    if code:
        line_count = code.count("\n") + 1
        copy_button = _render_copy_button(code)
        download_button = _render_download_button(code)
        badge = _status_pill("✓ Code généré", "sv-badge-approved")
        editor_html = (
            '<div class="sv-code-editor">'
            '<div class="sv-code-editor-topbar">'
            '<span class="sv-code-lang-badge">Python</span>'
            f'<span class="sv-code-lang-badge">{line_count} lignes</span>'
            '</div>'
            f'{_render_code_editor(code)}'
            '</div>'
        )
        code_section = _collapsible(
            f"Code Python ({line_count} lignes)",
            editor_html,
            open_by_default=line_count <= 40,
            extra_class="sv-collapsible-code",
        )
    else:
        copy_button = ""
        download_button = ""
        badge = _status_pill("✍ Génération…", "sv-badge-typing")
        code_section = '<div class="sv-shimmer" style="margin-top:0.6rem;"></div>'

    summary_section = f'<p class="sv-summary sv-typing-cursor">{summary_html}</p>' if summary_html else ""

    return _flat(f"""
    <div class="sv-card sv-agent-card sv-agent-codeur{_anim_class()}">
        <div class="sv-agent-header">
            <span>🧑‍💻 CodeurAgent</span>
            {badge}
            {copy_button}
            {download_button}
        </div>
        {summary_section}
        {code_section}
    </div>
    """)


def _render_copy_button(code: str) -> str:
    """
    Bouton "Copier". `code` (déjà garanti str par `_coerce_to_text`) est
    échappé en DEUX temps : `json.dumps` (littéral de chaîne JS valide,
    jamais un objet) puis `html.escape(..., quote=True)` (insertion sûre
    dans l'attribut `onclick`). Ecrit sur une seule ligne logique (voir
    `_flat`) pour rester un bloc HTML valide.
    """
    js_string_literal = json.dumps(code)
    safe_for_html_attribute = html.escape(js_string_literal, quote=True)
    onclick = (
        f"navigator.clipboard.writeText({safe_for_html_attribute});"
        f"this.innerText='✓ Copié';"
        f"setTimeout(()=>{{this.innerText='Copier';}},1500);"
    )
    return f'<button class="sv-copy-btn" onclick="{onclick}">Copier</button>'


# ============================================================
# ML Risk Predictor
# ============================================================

def render_ml_risk_card(content: Any) -> str:
    """
    Carte du signal ML Risk Predictor (XGBoost, seuil défini dans
    utils/ml_risk.py), affichée entre CodeurAgent et ReviewerAgent.

    Ce signal est STRICTEMENT informatif : il ne doit jamais être présenté
    comme une décision du Reviewer (jamais "Reviewer: HIGH" ni
    "Décision: HIGH" -- voir consigne produit). Le libellé et le
    disclaimer ci-dessous le rappellent explicitement.

    `content` est le dict produit par `utils.ml_risk.MLRiskEvent.content`
    -- AUCUNE valeur n'est recalculée ou transformée ici, uniquement mise
    en forme. Champs garantis (voir predict_revision_risk()) : task,
    probability, risk_level, prediction, prediction_label. Champs
    optionnels, affichés UNIQUEMENT s'ils sont réellement présents (jamais
    inventés, jamais "0" ou "N/A" par défaut si absents -- la ligne
    correspondante est alors simplement omise) : code_length,
    code_line_count, planner_execution_time, planner_prompt_tokens,
    planner_completion_tokens, coder_prompt_tokens, coder_completion_tokens,
    coder_total_tokens, threshold (transportés tels quels par
    ml_risk.compute_risk_event, pas recalculés ici) ; iteration,
    max_iterations (ajoutés par orchestrator.py, qui est seul à savoir
    combien de tours Codeur ont déjà eu lieu).

    Si `content` est `None` ou ne contient pas de prédiction exploitable
    (ML best-effort : le modèle peut échouer sans jamais bloquer le
    Reviewer -- voir ml_risk.py), affiche un état "Prédiction
    indisponible" plutôt que d'inventer un risk_level ou une probability.

    SEUL LE RENDU a été retouché ici (mise en page en grille) : aucune
    valeur, aucun seuil et aucun calcul ne sont modifiés.
    """
    risk_level = content.get("risk_level") if isinstance(content, dict) else None
    probability = content.get("probability") if isinstance(content, dict) else None

    if not risk_level or probability is None:
        badge = _status_pill("Prédiction indisponible", "sv-badge-thinking")
        body = (
            '<p class="sv-summary">Le pipeline continue normalement — '
            "le signal ML n'a pas pu être calculé pour ce tour.</p>"
        )
        return _flat(f"""
        <div class="sv-card sv-agent-card sv-agent-ml-risk{_anim_class()}">
            <div class="sv-agent-header">
                <span>🧠 ML Risk Predictor</span>
                {badge}
            </div>
            {body}
        </div>
        """)

    def _fmt_num(value: Any, suffix: str = "", decimals: int | None = None) -> str | None:
        """Formate une valeur numérique réellement présente, ou None si absente/invalide -- n'invente jamais un 0 ou un N/A à la place d'une valeur manquante."""
        if value is None:
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        text = f"{num:,.{decimals}f}" if decimals is not None else f"{num:,.0f}"
        return f"{text}{suffix}"

    try:
        probability_pct = f"{float(probability) * 100:.1f} %"
    except (TypeError, ValueError):
        probability_pct = html.escape(str(probability))

    risk_level_text = html.escape(str(risk_level))
    risk_badge_class = {
        "LOW": "sv-badge-risk-low",
        "MEDIUM": "sv-badge-risk-medium",
        "HIGH": "sv-badge-risk-high",
    }.get(risk_level_text.upper(), "sv-badge-thinking")

    decision_text = html.escape(str(content.get("prediction_label"))) if content.get("prediction_label") else None

    threshold_pct = _fmt_num(
        float(content["threshold"]) * 100 if content.get("threshold") is not None else None,
        suffix=" %",
        decimals=0,
    )

    badge = _status_pill("Prediction ✓", "sv-badge-approved")

    decision_stat_html = (
        f"""<div class="sv-ml-risk-stat">
            <span class="sv-ml-risk-stat-label">Decision</span>
            <span class="sv-ml-risk-stat-value">{decision_text}</span>
        </div>"""
        if decision_text
        else ""
    )

    threshold_html = (
        f'<p class="sv-ml-risk-threshold">Decision threshold: <strong>{threshold_pct}</strong></p>'
        if threshold_pct
        else ""
    )

    # --- Execution Context : uniquement les champs réellement présents ---
    coder_total = content.get("coder_total_tokens")
    if coder_total is None and content.get("coder_prompt_tokens") is not None and content.get("coder_completion_tokens") is not None:
        coder_total = content["coder_prompt_tokens"] + content["coder_completion_tokens"]
    planner_total = None
    if content.get("planner_prompt_tokens") is not None and content.get("planner_completion_tokens") is not None:
        planner_total = content["planner_prompt_tokens"] + content["planner_completion_tokens"]

    context_rows = [
        ("Code length", _fmt_num(content.get("code_length"))),
        ("Code line count", _fmt_num(content.get("code_line_count"))),
        ("Planner execution time", _fmt_num(content.get("planner_execution_time"), suffix=" s", decimals=2)),
        ("Planner tokens", _fmt_num(planner_total)),
        ("Coder tokens", _fmt_num(coder_total)),
    ]
    iteration = content.get("iteration")
    max_iterations = content.get("max_iterations")
    if iteration is not None and max_iterations is not None:
        context_rows.append(("Iteration", f"{html.escape(str(iteration))} / {html.escape(str(max_iterations))}"))

    context_rows = [(label, value) for label, value in context_rows if value is not None]
    context_html = ""
    if context_rows:
        rows_html = "".join(
            f'<div class="sv-ml-risk-context-row">'
            f'<span class="sv-ml-risk-context-label">{html.escape(label)}</span>'
            f'<span class="sv-ml-risk-context-value">{value}</span>'
            f"</div>"
            for label, value in context_rows
        )
        context_html = f"""
        <div class="sv-ml-risk-context">
            <span class="sv-ml-risk-context-title">📊 Execution Context</span>
            <div class="sv-ml-risk-context-grid">{rows_html}</div>
        </div>
        """

    return _flat(f"""
    <div class="sv-card sv-agent-card sv-agent-ml-risk{_anim_class()}">
        <div class="sv-agent-header">
            <span>🧠 ML Risk Predictor</span>
            {badge}
        </div>
        <div class="sv-ml-risk-grid">
            <div class="sv-ml-risk-stat">
                <span class="sv-ml-risk-stat-label">Risk level</span>
                <span class="sv-badge {risk_badge_class} sv-ml-risk-stat-value">{risk_level_text}</span>
            </div>
            <div class="sv-ml-risk-stat">
                <span class="sv-ml-risk-stat-label">Probability</span>
                <span class="sv-ml-risk-stat-value sv-ml-risk-probability">{probability_pct}</span>
            </div>
            {decision_stat_html}
        </div>
        {threshold_html}
        {context_html}
        <p class="sv-ml-risk-explainer">ℹ️ The model estimates the probability that the generated code will require a review.</p>
        <p class="sv-ml-risk-disclaimer">⚠ Signal d'aide à la décision — ne remplace pas le Reviewer.</p>
    </div>
    """)


# ============================================================
# Célébration / guirlandes de succès
# ============================================================

_CONFETTI_COLORS = ["#6D28D9", "#EC4899", "#06B6D4", "#22C55E", "#8B5CF6"]

# Couleurs des ampoules : violet, rose, cyan, vert (§5).
_GARLAND_COLORS = ["#8B5CF6", "#EC4899", "#06B6D4", "#22C55E", "#6D28D9", "#EC4899", "#06B6D4"]


def _render_garland_row(seed: int) -> str:
    """
    Une guirlande = un fil + des ampoules réparties régulièrement, avec un
    léger décalage d'animation pour un clignotement doux et désynchronisé.
    Rendu DÉTERMINISTE (graine fixe) : la guirlande ne "saute" pas d'un
    st.rerun() à l'autre.
    """
    rng = random.Random(seed)
    bulbs = []
    positions = [4 + i * 8 for i in range(12)]  # 12 ampoules, de 4% à 92%
    for i, left in enumerate(positions):
        color = _GARLAND_COLORS[i % len(_GARLAND_COLORS)]
        delay = round(rng.uniform(0, 2.2), 2)
        duration = round(rng.uniform(2.0, 3.2), 2)
        bulbs.append(
            f'<span class="sv-garland-bulb" style="left:{left}%;--sv-bulb:{color};'
            f'animation-delay:{delay}s;animation-duration:{duration}s;"></span>'
        )
    sparks = []
    for i, left in enumerate([10, 34, 58, 82]):
        delay = round(0.5 * i, 2)
        sparks.append(
            f'<span class="sv-garland-spark" style="left:{left}%;animation-delay:{delay}s;">✨</span>'
        )
    return (
        '<div class="sv-garland">'
        '<span class="sv-garland-wire"></span>'
        f'{"".join(bulbs)}{"".join(sparks)}'
        "</div>"
    )


def render_success_garland(session: Any) -> str:
    """
    Guirlandes lumineuses PERSISTANTES de succès (§5 / §16).

    RÈGLE UNIQUE ET STRICTE : ne rend quelque chose que si
    `session.approved is True`. Dans TOUS les autres cas -- session
    absente, `approved is False`, `approved is None` (pipeline non
    terminé ou aucun résultat) -- retourne une chaîne vide.

    `approved` est le booléen calculé par `backend.construire_resultat()`
    et stocké tel quel par orchestrator.py : aucune donnée métier n'est
    créée ni ré-interprétée ici.

    La persistance après `st.rerun()` est assurée par app.py, qui rappelle
    cette fonction à chaque rendu depuis l'objet `session` -- et par les
    keyframes CSS `infinite` qui ne retombent jamais à opacity:0.

    Si le toggle "Animations" est sur OFF, la zone de succès reste
    affichée avec toutes ses informations, mais sans guirlandes animées.
    """
    if session is None:
        return ""
    if getattr(session, "approved", None) is not True:
        return ""

    animations = _animations_enabled()

    iterations = getattr(session, "iterations_used", None)
    max_it = _max_iterations()
    if iterations is not None and max_it is not None:
        detail = f"Approuvé par le ReviewerAgent après {iterations} itération(s) sur {max_it} autorisée(s)."
    elif iterations is not None:
        detail = f"Approuvé par le ReviewerAgent après {iterations} itération(s)."
    else:
        detail = "Approuvé par le ReviewerAgent."

    if animations:
        top_garland = _render_garland_row(7)
        bottom_garland = _render_garland_row(13)
    else:
        top_garland = ""
        bottom_garland = ""

    anim_class = "" if animations else " sv-no-anim"

    return _flat(f"""
    <div class="sv-success-zone{anim_class}">
        {top_garland}
        <div class="sv-success-core">
            <div class="sv-success-title">✨ CODE APPROVED ✨</div>
            <div class="sv-success-subtitle">{html.escape(detail)}</div>
            <span class="sv-success-arrow">↓ Passage à l'exécution Docker</span>
        </div>
        {bottom_garland}
    </div>
    """)


def _render_celebration() -> str:
    """
    Décoration utilisée DANS la carte Reviewer pendant le streaming,
    lorsque l'objet `session` n'est pas encore disponible (orchestrator.py
    appelle `render_reviewer_card(content)` sans session : le résultat
    final, donc `session.approved`, n'existe pas encore à cet instant).

    Après le `st.rerun()`, c'est `render_success_garland(session)` --
    pilotée par le booléen réel `session.approved` -- qui prend le relais,
    et cette décoration-ci n'est plus rendue : aucune duplication.

    Toutes les keyframes associées sont `infinite` et ne retombent jamais
    à opacity:0 (voir styles/style.css) : l'élément reste donc visible
    tant qu'il est dans le DOM.
    """
    rng = random.Random(42)  # graine fixe : rendu déterministe, reproductible
    pieces = []
    for _ in range(8):  # "quelques petits confettis" -- pas une explosion
        left = rng.randint(4, 94)
        delay = round(rng.uniform(0, 1.4), 2)
        duration = round(rng.uniform(2.4, 3.6), 2)
        color = rng.choice(_CONFETTI_COLORS)
        rotate = rng.choice([0, 15, -15, 30, -30])
        pieces.append(
            f'<span class="sv-confetti-piece" style="left:{left}%;'
            f"background:{color};--sv-rot:{rotate}deg;"
            f'animation-delay:{delay}s;animation-duration:{duration}s;"></span>'
        )
    stars = []
    for i, left in enumerate([8, 22, 38, 58, 74, 90]):  # 6 étoiles ("5 à 8")
        delay = round(0.35 * i, 2)
        stars.append(f'<span class="sv-star-piece" style="left:{left}%;animation-delay:{delay}s;">✨</span>')

    confetti_html = "".join(pieces)
    stars_html = "".join(stars)
    return f'<div class="sv-celebration"><div class="sv-celebration-halo"></div>{confetti_html}{stars_html}</div>'


def _render_approved_banner() -> str:
    return (
        '<div class="sv-approved-banner">'
        '<span class="sv-approved-banner-icon">✅</span>'
        '<span class="sv-approved-banner-title">Code Approved</span>'
        '<span class="sv-approved-banner-badge">STATUT: APPROVED</span>'
        "</div>"
    )


def _render_revision_banner() -> str:
    """
    Contrepartie explicite de la bannière d'approbation : état visuel
    "Révision requise" (§5). Aucune guirlande n'accompagne jamais cet
    état.
    """
    return (
        '<div class="sv-revision-banner">'
        '<span class="sv-revision-banner-icon">↻</span>'
        '<span class="sv-revision-banner-title">Révision requise</span>'
        '<span class="sv-revision-banner-badge">STATUT: REVISION_REQUISE</span>'
        "</div>"
        '<div class="sv-revision-loop">'
        "<span>🔍 Reviewer</span><span>→</span><span>↻ Révision</span><span>→</span>"
        "<span>🧑‍💻 Codeur</span><span>→</span><span>🧠 ML Risk</span><span>→</span><span>🔍 Reviewer</span>"
        "</div>"
    )


# ============================================================
# ReviewerAgent
# ============================================================

def _reviewer_meta_html(session: Any, status: str) -> str:
    """
    Bloc "méta" de la carte Reviewer : numéro d'itération, nombre maximal
    d'itérations, état final.

    STRICTEMENT construit à partir de données déjà présentes :
      - `session.iterations_used`  (posé par orchestrator.py depuis
        `backend.construire_resultat().iterations_used`)
      - `st.session_state["max_iterations"]` (la valeur réellement passée
        à `backend.creer_team_code_review()`)
      - `session.approved`         (le booléen du backend)
    Chaque case est OMISE si sa donnée n'est pas disponible : aucune
    valeur n'est inventée, aucun "0" ni "N/A" de remplissage.
    """
    if session is None:
        return ""

    items: list[str] = []

    iterations = getattr(session, "iterations_used", None)
    max_it = _max_iterations()
    if iterations is not None:
        value = f"{iterations} / {max_it}" if max_it is not None else str(iterations)
        items.append(
            '<div class="sv-reviewer-meta-item">'
            '<span class="sv-reviewer-meta-label">Itération</span>'
            f'<span class="sv-reviewer-meta-value">{html.escape(value)}</span>'
            "</div>"
        )
    elif max_it is not None:
        items.append(
            '<div class="sv-reviewer-meta-item">'
            '<span class="sv-reviewer-meta-label">Itérations max</span>'
            f'<span class="sv-reviewer-meta-value">{max_it}</span>'
            "</div>"
        )

    approved = getattr(session, "approved", None)
    if approved is True:
        items.append(
            '<div class="sv-reviewer-meta-item">'
            '<span class="sv-reviewer-meta-label">État final</span>'
            '<span class="sv-reviewer-meta-value sv-reviewer-meta-value-ok">APPROVED</span>'
            "</div>"
        )
    elif approved is False:
        items.append(
            '<div class="sv-reviewer-meta-item">'
            '<span class="sv-reviewer-meta-label">État final</span>'
            '<span class="sv-reviewer-meta-value sv-reviewer-meta-value-ko">REVISION_REQUISE</span>'
            "</div>"
        )

    # Statut lu dans le message lui-même (source distincte de `approved`,
    # affichée telle quelle sans arbitrage).
    if status in ("APPROVED", "REVISION_REQUISE"):
        css = "sv-reviewer-meta-value-ok" if status == "APPROVED" else "sv-reviewer-meta-value-ko"
        items.append(
            '<div class="sv-reviewer-meta-item">'
            '<span class="sv-reviewer-meta-label">Statut du message</span>'
            f'<span class="sv-reviewer-meta-value {css}">{status}</span>'
            "</div>"
        )

    if not items:
        return ""
    return f'<div class="sv-reviewer-meta">{"".join(items)}</div>'


def render_reviewer_card(content: Any, session: Any = None) -> str:
    """
    `session` est OPTIONNEL -- l'appel historique
    `render_reviewer_card(content)` de utils/orchestrator.py (pendant le
    streaming) reste valide tel quel et conserve exactement son
    comportement d'origine.

    Deux modes :
      - session is None  (streaming) : badge + célébration dérivés du
        parsing textuel, comme aujourd'hui.
      - session fournie  (après st.rerun(), depuis app.py) : ajout du bloc
        méta (itération / max / état final) et de la bannière
        APPROVED / REVISION_REQUISE pilotée par `session.approved`. La
        célébration n'est PAS rendue ici dans ce mode : les guirlandes
        persistantes sont rendues séparément par
        `render_success_garland(session)` (évite tout doublon).
    """
    text = _coerce_to_text(content)
    status = _parse_reviewer_status(text)

    badge_by_status = {
        "APPROVED": ("✓ APPROUVÉ", "sv-badge-approved"),
        "REVISION_REQUISE": ("↻ RÉVISION REQUISE", "sv-badge-revision"),
        "EN_COURS": ("… En cours", "sv-badge-thinking"),
    }
    label, css_class = badge_by_status[status]
    badge = _status_pill(label, css_class)

    approved = getattr(session, "approved", None) if session is not None else None

    if session is None:
        # Mode streaming : comportement historique strictement conservé.
        celebration = _render_celebration() if (status == "APPROVED" and _animations_enabled()) else ""
        banner = _render_approved_banner() if status == "APPROVED" else ""
        if status == "REVISION_REQUISE":
            banner = _render_revision_banner()
        is_approved_visual = status == "APPROVED"
        is_revision_visual = status == "REVISION_REQUISE"
    else:
        # Mode persistant : la source de vérité est le booléen du backend.
        celebration = ""
        if approved is True:
            banner = _render_approved_banner()
        elif approved is False:
            banner = _render_revision_banner()
        else:
            # `approved` inconnu (pipeline interrompu) : on n'invente rien,
            # on retombe sur ce que dit le message lui-même.
            banner = _render_approved_banner() if status == "APPROVED" else ""
            if status == "REVISION_REQUISE":
                banner = _render_revision_banner()
        is_approved_visual = approved is True or (approved is None and status == "APPROVED")
        is_revision_visual = approved is False or (approved is None and status == "REVISION_REQUISE")

    # Pilotée UNIQUEMENT par le statut d'approbation -- jamais liée à
    # Docker ni à une durée.
    if is_approved_visual:
        card_extra_class = " sv-agent-reviewer-approved"
    elif is_revision_visual:
        card_extra_class = " sv-agent-reviewer-revision"
    else:
        card_extra_class = ""

    meta_html = _reviewer_meta_html(session, status)

    body_html = html.escape(text).replace("\n", "<br>")
    line_count = text.count("\n") + 1
    feedback_section = _collapsible(
        "Revue détaillée — résumé et justification",
        f'<div class="sv-reviewer-body">{body_html}</div>',
        open_by_default=line_count <= 12,
    )

    return _flat(f"""
    <div class="sv-card sv-agent-card sv-agent-reviewer{card_extra_class}{_anim_class()}">
        {celebration}
        <div class="sv-agent-header">
            <span>🔍 ReviewerAgent</span>
            {badge}
        </div>
        {banner}
        {meta_html}
        {feedback_section}
    </div>
    """)


# ============================================================
# Docker Executor
# ============================================================

def render_docker_card(session: ConversationSession) -> str:
    requires_input = getattr(session, "docker_requires_input", False)
    error_message = getattr(session, "docker_error_message", None)
    success = getattr(session, "docker_success", None)

    if requires_input:
        badge = _status_pill("⌨ Saisie requise", "sv-badge-revision")
        body = f'<div class="sv-collapsible-body">{html.escape(error_message or "")}</div>'
        return _flat(f"""
        <div class="sv-card sv-agent-card sv-agent-docker{_anim_class()}">
            <div class="sv-agent-header"><span>🐳 Docker Executor</span>{badge}</div>
            {body}
        </div>
        """)

    if error_message:
        badge = _status_pill("⚠ Non exécuté", "sv-badge-revision")
        return _flat(f"""
        <div class="sv-card sv-agent-card sv-agent-docker sv-card-error{_anim_class()}">
            <div class="sv-agent-header"><span>🐳 Docker Executor</span>{badge}</div>
            <div class="sv-collapsible-body">{html.escape(error_message)}</div>
        </div>
        """)

    if success is None:
        badge = _status_pill("⏳ En attente", "sv-badge-thinking")
        return _flat(f"""
        <div class="sv-card sv-agent-card sv-agent-docker{_anim_class()}">
            <div class="sv-agent-header"><span>🐳 Docker Executor</span>{badge}</div>
            <div class="sv-shimmer"></div>
        </div>
        """)

    css_class = "sv-badge-approved" if success else "sv-badge-revision"
    label = "✓ Succès" if success else "✗ Échec"
    badge = _status_pill(label, css_class)
    exit_code = getattr(session, "docker_exit_code", None)
    output = getattr(session, "docker_output", "") or ""
    output_lines = output.split("\n") if output else [""]
    line_count = len(output_lines)

    # Chaque ligne de sortie est un élément distinct avec un délai
    # d'animation croissant (effet d'écriture progressive), et le curseur
    # clignotant (::after en CSS) n'apparaît que sur la DERNIÈRE ligne.
    terminal_lines_html = "".join(
        f'<div class="sv-terminal-line{" sv-terminal-cursor-line" if i == line_count - 1 else ""}" '
        f'style="animation-delay:{min(i * 0.06, 1.2):.2f}s">{html.escape(line) if line else "&nbsp;"}</div>'
        for i, line in enumerate(output_lines)
    )

    success_banner = _render_docker_success_banner() if success else ""

    terminal_section = _collapsible(
        f"Sortie d'exécution — exit code {exit_code}",
        (
            '<div class="sv-terminal">'
            '<div class="sv-terminal-header">'
            '<div class="sv-terminal-dots"><span class="r"></span><span class="y"></span><span class="g"></span></div>'
            f'<span class="sv-terminal-exitcode {"sv-terminal-exitcode-ok" if success else "sv-terminal-exitcode-fail"}">exit {exit_code}</span>'
            "</div>"
            '<div class="sv-terminal-title">smartovate@docker-executor:~$ python script.py</div>'
            f"{terminal_lines_html}"
            "</div>"
        ),
        open_by_default=line_count <= 20,
    )

    return _flat(f"""
    <div class="sv-card sv-agent-card sv-agent-docker{_anim_class()}">
        <div class="sv-agent-header"><span>🐳 Docker Executor</span>{badge}</div>
        {success_banner}
        {terminal_section}
    </div>
    """)


def _render_docker_success_banner() -> str:
    return (
        '<div class="sv-docker-success-banner">'
        '<span class="sv-docker-success-check">✓</span>'
        "<span>🚀 Build successful — 🐳 Docker completed — Safe execution finished</span>"
        "</div>"
    )


def render_placeholder(agent_label: str, dot_class: str) -> str:
    """Carte "en attente" affichée avant que l'agent n'ait produit de contenu."""
    return _flat(f"""
    <div class="sv-card sv-agent-card{_anim_class()}">
        <div class="sv-agent-header">
            <span class="sv-agent-dot {dot_class}"></span>
            <span>{html.escape(agent_label)}</span>
        </div>
        <div class="sv-shimmer"></div>
        <div class="sv-shimmer" style="width:70%;margin-top:0.4rem;"></div>
    </div>
    """)