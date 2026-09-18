# streamlit_app/utils/backend.py
"""
Adaptateur unique entre l'interface Streamlit et le backend Smartovate.

Ce module est le SEUL fichier du frontend autorisé à importer directement
depuis agents/, teams/ ou config/. Aucune logique métier n'est réimplémentée
ici : chaque fonction ne fait qu'appeler une fonction existante du backend et
adapte le résultat (ou les erreurs) à un format facile à consommer par l'UI.

Entry points backend réellement utilisés :
    - agents.planner_agent.create_planner_agent
    - teams.code_review_team.build_code_review_team
    - teams.code_review_team.build_code_review_result
    - teams.code_review_team.execute_approved_code
    - teams.code_review_team.save_final_code_to_workspace
    - teams.docker_executor.DockerNotAvailableError
    - config.settings.get_model_client (pour gérer explicitement le cycle
      de vie du client Azure OpenAI -- voir "Issue 2" ci-dessous)

------------------------------------------------------------------------
Issue 2 -- "RuntimeError: Event loop is closed" (httpx.AsyncClient.aclose)
------------------------------------------------------------------------
Cause racine : `create_planner_agent()` et `build_code_review_team()`
créent chacun, s'ils ne reçoivent pas de `model_client` explicite, un
NOUVEAU `AzureOpenAIChatCompletionClient` via `get_model_client()` --
lequel encapsule un `httpx.AsyncClient`. Si le plan est généré via un
premier `asyncio.run(...)` et la Team streamée via un second (thread
séparé, cf. utils/stream.py), on obtient potentiellement PLUSIEURS
clients, chacun lié à une boucle asyncio différente, et AUCUN n'est
jamais explicitement fermé. Quand Python finit par les garbage-collecter,
leur finaliseur tente un `await ... aclose()` sur une boucle déjà fermée
-> exactement l'erreur observée, typiquement après la fin de l'exécution
(quand le ramasse-miettes passe enfin sur ces objets).

Correction : `executer_pipeline_complet()` crée UN SEUL client, le
transmet explicitement au Planner ET à la Team (tous deux l'acceptent
déjà via leur paramètre `model_client`), exécute tout le pipeline dans
une unique coroutine, puis appelle `await client.close()` dans un bloc
`finally` -- donc toujours DANS la même boucle asyncio, avant qu'elle ne
se ferme. C'est la fonction à utiliser depuis `components/prompt_box.py`
(voir note de migration fournie séparément).

------------------------------------------------------------------------
Issue 3 -- EOFError sur input() dans le code exécuté par Docker
------------------------------------------------------------------------
Docker exécute le script sans stdin interactif. Si le code généré
contient un appel à `input()`, l'exécution échoue avec un `EOFError`
brut et incompréhensible pour l'utilisateur. On détecte ce cas de deux
façons complémentaires :
  1. Analyse statique AVANT exécution (AST) : si `input()` est détecté,
     on n'exécute même pas le conteneur (inutile, gain de temps/coût) et
     on retourne un message clair.
  2. Filet de sécurité APRÈS exécution : si malgré tout la sortie
     contient la signature d'un EOFError (obfuscation, input() généré
     dynamiquement...), on reformate le message de la même façon plutôt
     que d'afficher la stack trace brute.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import time
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from autogen_agentchat.base import TaskResult
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import RequestUsage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from agents.planner_agent import create_planner_agent
from config.settings import get_model_client
from teams.code_review_team import (
    CodeReviewResult,
    build_code_review_result,
    build_code_review_team,
    execute_approved_code,
    save_final_code_to_workspace,
)
from teams.docker_executor import DockerNotAvailableError, ExecutionResult
from streamlit_app.utils.execution_logger import AgentUsage, ExecutionLogger, ExecutionMetrics


# ============================================================
# Issue 3 -- détection d'un input() bloquant
# ============================================================

_INTERACTIVE_INPUT_MESSAGE = (
    "Ce script attend une saisie interactive (input()), ce que "
    "l'environnement Docker non interactif ne peut pas fournir. Le code a "
    "bien été validé par le ReviewerAgent, mais n'a volontairement pas été "
    "exécuté automatiquement pour éviter une erreur incompréhensible. "
    "Téléchargez-le (bouton Export) pour le lancer vous-même dans un "
    "terminal, ou reformulez la demande en précisant les valeurs d'entrée "
    "à l'avance."
)


def _code_requires_interactive_input(code: str) -> bool:
    """
    Détecte, par analyse statique (AST) plutôt que par une recherche
    textuelle fragile, un appel à la fonction native `input()` n'importe
    où dans le script -- y compris à l'intérieur de fonctions imbriquées.

    On utilise l'AST (et non un simple `"input(" in code`) pour éviter les
    faux positifs sur des occurrences dans des chaînes ou des commentaires,
    et les faux négatifs sur des appels reformatés (ex: `input (x)`).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Le ReviewerAgent a déjà validé un code syntaxiquement correct ;
        # si le parsing échoue malgré tout, on ne bloque pas ici pour
        # autant -- ce n'est pas le rôle de cette fonction de re-juger la
        # qualité du code, seulement de détecter un pattern précis.
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "input":
            return True
    return False


def _output_indicates_eof_error(output: str) -> bool:
    """Filet de sécurité : détecte un EOFError dans la sortie d'exécution."""
    return "EOFError" in output


# ============================================================
# Événements du pipeline unifié (Issue 2)
# ============================================================

PipelineEventType = Literal[
    "planner_start",
    "planner_done",
    "agent_message",
    "review_done",
    "docker_start",
    "docker_requires_input",
    "docker_done",
    "docker_error",
    "pipeline_done",
]


@dataclass
class PipelineEvent:
    """
    Événement émis par `executer_pipeline_complet`. `stream.py` traite
    déjà tout élément qui n'est pas un `TaskResult` comme un message
    générique (`kind="message"`) -- un `PipelineEvent` y correspond donc
    sans aucune modification de stream.py : le composant consommateur
    (prompt_box.py) n'a qu'à distinguer `PipelineEvent` des messages bruts
    d'agent via `isinstance(event.payload, PipelineEvent)`.
    """

    type: PipelineEventType
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class DockerExecutionOutcome:
    """
    Résultat de la tentative d'exécution Docker, adapté pour l'UI.

    `requires_input` distingue explicitement le cas "le code attend une
    saisie interactive" (Issue 3, pas une vraie erreur d'infrastructure)
    du cas "Docker est indisponible" (`error_message` seul, sans ce flag)
    -- utile pour que l'UI affiche une icône/un ton différent.
    """

    result: ExecutionResult | None
    error_message: str | None
    requires_input: bool = False


# ============================================================
# Fonctions granulaires (conservées pour compatibilité)
# ============================================================

async def generer_plan(prompt: str) -> str:
    """
    Appelle le PlannerAgent existant et retourne le texte brut du plan.

    Corrigé pour l'Issue 2 : crée son propre client Azure OpenAI et le
    ferme explicitement avant de retourner, dans la MÊME coroutine (donc
    dans la même boucle asyncio) -- évite qu'il ne fuite jusqu'au
    ramasse-miettes.

    Note : si votre pipeline appelle ensuite `creer_team_code_review()`
    séparément, préférez `executer_pipeline_complet()` ci-dessous, qui
    partage un seul client entre le Planner et la Team au lieu d'en créer
    un par étape.
    """
    client = get_model_client()
    try:
        planner = create_planner_agent(model_client=client)
        task_result: TaskResult = await planner.run(task=prompt)
        return task_result.messages[-1].content
    finally:
        await client.close()


@dataclass
class PlanGenerationResult:
    """
    Résultat de `generer_plan_avec_usage()` : le plan tel que retourné par
    `generer_plan()`, plus les tokens réellement consommés pour cet appel
    (`models_usage`, fourni par AutoGen/Azure OpenAI sur le dernier
    message du Planner -- voir `execution_logger.py`, qui interdit
    d'estimer les tokens autrement que via cette valeur).

    `models_usage` vaut `None` si AutoGen ne l'a pas fourni pour cet appel
    (aucune valeur inventée en remplacement).
    """

    plan: str
    models_usage: RequestUsage | None


async def generer_plan_avec_usage(prompt: str) -> PlanGenerationResult:
    """
    Variante de `generer_plan()` qui expose EN PLUS les tokens réellement
    consommés par cet appel Planner (`models_usage`), nécessaires à
    l'historique des exécutions (ExecutionLogger).

    Reprend EXACTEMENT le comportement de `generer_plan()` (même client,
    même agent, même fermeture explicite dans la même coroutine -- voir
    Issue 2 ci-dessus) : `generer_plan()` n'est ni modifiée ni appelée en
    double ici, pour ne pas payer deux fois le coût Azure OpenAI de la
    génération du plan.
    """
    client = get_model_client()
    try:
        planner = create_planner_agent(model_client=client)
        task_result: TaskResult = await planner.run(task=prompt)
        last_message = task_result.messages[-1]
        return PlanGenerationResult(
            plan=last_message.content,
            models_usage=getattr(last_message, "models_usage", None),
        )
    finally:
        await client.close()


def creer_team_code_review(max_iterations: int = 5) -> RoundRobinGroupChat:
    """
    Retourne la Team Codeur/Reviewer, prête pour run_stream().

    ATTENTION (Issue 2) : cette fonction crée un client Azure OpenAI
    (un par agent Codeur/Reviewer, partagé entre eux) qu'AUCUNE fonction
    de ce module ne ferme automatiquement -- car il doit rester ouvert
    pendant toute la durée du streaming, qui se produit ailleurs (dans le
    thread de utils/stream.py). Si vous utilisez cette fonction
    directement, vous êtes responsable de fermer le client vous-même
    après consommation du flux (voir `creer_team_code_review_avec_client`
    ci-dessous). Pour le flux principal, préférez
    `executer_pipeline_complet()`, qui gère ce cycle de vie pour vous.
    """
    return build_code_review_team(max_iterations=max_iterations)


def creer_team_code_review_avec_client(
    max_iterations: int = 5,
) -> tuple[RoundRobinGroupChat, AzureOpenAIChatCompletionClient]:
    """
    Variante de `creer_team_code_review` qui retourne AUSSI le client
    utilisé, pour permettre à l'appelant de le fermer explicitement
    (`await client.close()`) une fois le streaming terminé -- corrige
    l'Issue 2 pour les intégrations qui ne peuvent pas utiliser
    `executer_pipeline_complet()` tel quel.
    """
    client = get_model_client()
    team = build_code_review_team(max_iterations=max_iterations, model_client=client)
    return team, client


async def fermer_client(client: AzureOpenAIChatCompletionClient) -> None:
    """Ferme explicitement un client Azure OpenAI (voir Issue 2)."""
    await client.close()


def obtenir_flux_code_review(team: RoundRobinGroupChat, plan: str) -> AsyncIterator:
    """
    Retourne le générateur asynchrone brut de la Team (non consommé).

    Conservée TELLE QUELLE pour compatibilité (tests existants, éventuels
    autres appelants) -- le pipeline UI réel utilise désormais
    `obtenir_flux_code_review_avec_ml()` ci-dessous, qui ajoute la
    prédiction de risque ML sans changer le comportement Codeur/Reviewer.
    """
    return team.run_stream(task=plan)


async def obtenir_flux_code_review_avec_ml(
    team: RoundRobinGroupChat,
    plan: str,
    prompt: str,
    planner_execution_time: float | None,
    planner_prompt_tokens: int | None,
    planner_completion_tokens: int | None,
) -> AsyncIterator:
    """
    Variante de `obtenir_flux_code_review()` qui insère la prédiction de
    risque ML (utils/ml_risk.py) exactement entre la fin d'un tour du
    CodeurAgent et le début du tour suivant du ReviewerAgent.

    ------------------------------------------------------------------
    Pourquoi ceci garantit réellement "ML calculé AVANT le Reviewer"
    ------------------------------------------------------------------
    `team.run_stream()` est un générateur asynchrone : il calcule le tour
    d'UN participant, le `yield` puis SE SUSPEND jusqu'à ce que son
    consommateur redemande un élément (`__anext__()`). Le tour du
    ReviewerAgent n'est donc calculé (et l'appel Azure OpenAI du Reviewer
    déclenché) que lorsque CE générateur enveloppant redemande l'élément
    suivant au générateur interne -- ce qui, ci-dessous, n'arrive qu'APRÈS
    que la prédiction ML (bloquante mais purement locale, aucun appel
    réseau) soit terminée et son événement émis. Le Reviewer ne peut donc
    matériellement pas démarrer avant que le ML n'ait fini.

    Le Reviewer n'est JAMAIS empêché de s'exécuter par ce wrapper : si le
    ML échoue ou est indisponible, la boucle continue normalement (voir
    `ml_risk.compute_risk_event`, qui ne lève jamais d'exception).

    Args:
        team: la Team Codeur/Reviewer (inchangée, voir teams/code_review_team.py).
        plan: le plan produit par le PlannerAgent (transmis tel quel à la Team).
        prompt: le prompt UTILISATEUR original (pour les features textuelles
            du modèle ML -- distinct de `plan`, voir ml_risk.py).
        planner_execution_time / planner_prompt_tokens / planner_completion_tokens:
            métriques RÉELLES du Planner, déjà mesurées par l'appelant
            (orchestrator.py) avant l'appel à cette fonction.

    Yields:
        Les messages/TaskResult de la Team, INCHANGÉS et dans le même
        ordre qu'avec `obtenir_flux_code_review()`, entrecoupés d'un
        `ml_risk.MLRiskEvent` après chaque message du CodeurAgent pour
        lequel la prédiction a pu être calculée (silencieusement omis
        sinon -- jamais de valeur inventée).
    """
    from utils import ml_risk

    coder_prompt_tokens_acc: int | None = None
    coder_completion_tokens_acc: int | None = None

    async for item in team.run_stream(task=plan):
        # Le message (ou TaskResult) est toujours retransmis EN PREMIER,
        # à l'identique -- le comportement d'affichage existant n'est pas
        # modifié par ce wrapper.
        yield item

        if isinstance(item, TaskResult):
            continue

        source = getattr(item, "source", "")
        if source != "CodeurAgent":
            continue

        # Accumulation LOCALE des tokens Coder (indépendante de celle
        # d'orchestrator.py, qui continue de fonctionner à l'identique sur
        # les événements qu'elle reçoit plus tard) -- garantit que la
        # feature "coder_total_tokens" ne reflète que les tokens
        # RÉELLEMENT consommés jusqu'à CE tour, jamais une estimation.
        usage = getattr(item, "models_usage", None)
        if usage is not None:
            coder_prompt_tokens_acc = (coder_prompt_tokens_acc or 0) + usage.prompt_tokens
            coder_completion_tokens_acc = (coder_completion_tokens_acc or 0) + usage.completion_tokens

        content = getattr(item, "content", None)
        content_text = content if isinstance(content, str) else str(content)

        # Appel BLOQUANT mais purement local (inférence XGBoost déjà
        # entraîné, aucun réseau) -- c'est ce blocage, avant le prochain
        # `async for` (qui redemande l'item suivant à `team.run_stream`),
        # qui retarde matériellement le tour du ReviewerAgent.
        risk_event = ml_risk.compute_risk_event(
            task=prompt,
            coder_message_content=content_text,
            planner_execution_time=planner_execution_time,
            planner_prompt_tokens=planner_prompt_tokens,
            planner_completion_tokens=planner_completion_tokens,
            coder_prompt_tokens_accumulated=coder_prompt_tokens_acc,
            coder_completion_tokens_accumulated=coder_completion_tokens_acc,
        )
        if risk_event is not None:
            yield risk_event


def construire_resultat(task_result: TaskResult) -> CodeReviewResult:
    """
    Construit un CodeReviewResult à partir du TaskResult déjà obtenu --
    ne relance JAMAIS la Team (pas de second coût Azure OpenAI).
    """
    return build_code_review_result(task_result)


async def executer_dans_docker(
    result: CodeReviewResult,
    work_dir: str = "./_docker_workspace",
) -> DockerExecutionOutcome:
    """
    Exécute le code approuvé dans Docker via execute_approved_code().

    Corrigé pour l'Issue 3 : détecte un `input()` bloquant AVANT de lancer
    le conteneur (gain de temps, pas d'EOFError brut), avec un filet de
    sécurité après exécution au cas où la détection statique aurait
    manqué un cas particulier.

    Intercepte toujours :
      - ValueError : code non approuvé par le ReviewerAgent.
      - DockerNotAvailableError : Docker Desktop non démarré/accessible.
    """
    if result.final_code and _code_requires_interactive_input(result.final_code):
        return DockerExecutionOutcome(
            result=None,
            error_message=_INTERACTIVE_INPUT_MESSAGE,
            requires_input=True,
        )

    try:
        exec_result = await execute_approved_code(result, work_dir=work_dir)
    except ValueError as exc:
        return DockerExecutionOutcome(result=None, error_message=str(exc))
    except DockerNotAvailableError as exc:
        return DockerExecutionOutcome(result=None, error_message=str(exc))

    if not exec_result.success and _output_indicates_eof_error(exec_result.output):
        return DockerExecutionOutcome(
            result=exec_result,
            error_message=_INTERACTIVE_INPUT_MESSAGE,
            requires_input=True,
        )

    return DockerExecutionOutcome(result=exec_result, error_message=None)


def sauvegarder_code(
    result: CodeReviewResult,
    filename: str = "generated_script.py",
) -> Path:
    """Sauvegarde le code final via save_final_code_to_workspace() (backend)."""
    return save_final_code_to_workspace(result, filename=filename)


# ============================================================
# Pipeline unifié (recommandé -- corrige l'Issue 2 à la racine)
# ============================================================

async def executer_pipeline_complet(
    prompt: str,
    max_iterations: int = 5,
) -> AsyncIterator[PipelineEvent]:
    """
    Exécute la totalité du pipeline (Planner -> Codeur/Reviewer -> Docker)
    dans UNE SEULE coroutine, avec UN SEUL client Azure OpenAI partagé,
    fermé explicitement en fin de parcours -- corrige l'Issue 2 à la
    racine (plus de client orphelin, plus de boucle fermée prématurément).

    À utiliser ainsi depuis components/prompt_box.py, via
    utils/stream.run_stream_in_thread (AUCUNE modification de stream.py
    n'est nécessaire : chaque `PipelineEvent` yield ici est traité comme
    un message générique par ce module) :

        q = stream.run_stream_in_thread(
            lambda: backend.executer_pipeline_complet(prompt, max_iterations)
        )
        events, finished = stream.drain_queue(q)
        for event in events:
            payload = event.payload
            if isinstance(payload, backend.PipelineEvent):
                ... # traiter selon payload.type / payload.payload

    Chaque message d'agent individuel (Codeur/Reviewer) est également
    émis au fil de l'eau via un événement `"agent_message"`, dont
    `payload["content"]` est TOUJOURS une chaîne de caractères déjà
    extraite (`.content`) -- jamais l'objet message lui-même. C'est ce qui
    ferme l'Issue 1 côté pipeline : si votre `prompt_box.py` consomme ces
    événements plutôt que le flux brut de la Team, il ne peut plus
    recevoir autre chose qu'une chaîne pour le contenu d'un agent.

    Args:
        prompt: la demande initiale de l'utilisateur.
        max_iterations: nombre max d'itérations Codeur<->Reviewer.

    Yields:
        PipelineEvent, dans l'ordre chronologique du pipeline.

    ------------------------------------------------------------------
    Historique des exécutions (ExecutionLogger)
    ------------------------------------------------------------------
    Cette fonction est le SEUL endroit du pipeline qui voit, dans l'ordre,
    la totalité des étapes (Planner -> Codeur/Reviewer -> Docker) : c'est
    donc l'endroit le plus fiable pour mesurer des temps réels
    (`time.perf_counter()`) et collecter les tokens réellement consommés
    (`message.models_usage`, fourni par AutoGen/Azure OpenAI), puis les
    transmettre à `ExecutionLogger.log_execution()` pour qu'ils soient
    persistés (SQLite, voir execution_logger.py) comme UNE ligne de
    dataset par exécution complète.

    Aucune métrique n'est recalculée ou estimée ici : les temps viennent
    de mesures `perf_counter()` autour des étapes réellement exécutées,
    et les tokens viennent tels quels de `models_usage` quand AutoGen les
    fournit. Toute métrique non disponible pour cette exécution (ex:
    Docker jamais atteint car révision requise) reste `None`.

    Le logging est protégé par son propre `try/except` : un échec de
    l'ExecutionLogger (ex: disque plein, base verrouillée) ne doit
    JAMAIS faire échouer le pipeline métier lui-même.

    IMPORTANT (bug corrigé) : l'écriture SQLite est déclenchée juste AVANT
    chaque `yield` terminal (celui qui porte `"pipeline_done"`, ou
    l'événement d'erreur qui le précède immédiatement), PAS dans le bloc
    `finally` de la fonction. Cette fonction est un générateur asynchrone
    consommé depuis un thread dédié avec sa PROPRE boucle asyncio (voir
    `utils/stream.run_stream_in_thread`, citée plus haut). Si ce
    consommateur arrête de tirer les événements dès qu'il voit
    `pipeline_done` (comportement naturel : plus rien d'utile à lire
    après), le générateur reste suspendu indéfiniment sur ce dernier
    `yield`. Sa fermeture (et donc l'exécution de `finally`) ne se
    déclenche alors que via le ramasse-miettes, qui doit reprogrammer un
    `aclose()` sur la boucle d'origine -- or cette boucle est fermée
    (`loop.close()`) dès que la fonction du thread se termine, juste après
    le `break`. Le `finally` ne s'exécute donc jamais, silencieusement (au
    mieux un `RuntimeWarning: Task was destroyed but it is pending!` sur
    stderr, jamais visible dans l'UI Streamlit) : c'est exactement ce qui
    empêchait `logger.log_execution(metrics)` de s'exécuter en usage réel,
    alors qu'un test manuel isolé (sans ce mécanisme thread+queue)
    fonctionnait. Exécuter le log AVANT le yield terminal le place sur le
    chemin normal d'exécution du générateur, garanti atteint tant que le
    consommateur lit au moins jusqu'à cet événement.
    """
    client = get_model_client()
    logger = ExecutionLogger()
    metrics = ExecutionMetrics(task=prompt)
    pipeline_start = time.perf_counter()
    logged = False

    # --- DIAGNOSTIC TEMPORAIRE (à retirer une fois le vrai chemin confirmé) ---
    # Objectif : prouver, depuis les logs du conteneur (`docker logs`), si
    # cette fonction est réellement celle invoquée par le chemin Streamlit
    # réel, et si oui, jusqu'où elle est exécutée.
    from streamlit_app.utils.database import get_db_path
    print(
        f"[DIAG] executer_pipeline_complet() ENTERED "
        f"| module_file={__file__} "
        f"| execution_logger_module={ExecutionLogger.__module__} "
        f"| db_path={get_db_path()} "
        f"| task_preview={prompt[:80]!r}"
    )
    # --- FIN DIAGNOSTIC TEMPORAIRE ---

    def _log_now() -> None:
        """
        Finalise `metrics` et écrit la ligne SQLite. Appelée explicitement
        sur le chemin normal d'exécution (voir note ci-dessus), jamais
        depuis `finally` seul. Idempotente via `logged` : n'écrit qu'une
        seule fois par exécution, quel que soit le nombre de points de
        sortie traversés.
        """
        nonlocal logged
        if logged:
            return
        logged = True
        metrics.total_execution_time = time.perf_counter() - pipeline_start

        # --- DIAGNOSTIC TEMPORAIRE ---
        print(
            f"[DIAG] LOGGING REACHED "
            f"| DB PATH={get_db_path()} "
            f"| execution_id={metrics.execution_id} "
            f"| task={metrics.task!r}"
        )
        # --- FIN DIAGNOSTIC TEMPORAIRE ---

        try:
            logger.log_execution(metrics)
            print(f"[DIAG] log_execution() a retourné sans exception pour execution_id={metrics.execution_id}")
        except Exception:
            # Best-effort : le logging ne doit jamais faire échouer le
            # pipeline métier. On trace quand même l'erreur COMPLÈTE
            # (traceback, pas juste repr) pour qu'un futur problème de
            # logging reste diagnosticable dans les logs du conteneur.
            import traceback
            print(f"[ExecutionLogger] ÉCHEC de l'écriture SQLite pour execution_id={metrics.execution_id} :")
            traceback.print_exc()

    try:
        # --- Étape 1 : Planner ---
        yield PipelineEvent(type="planner_start")
        planner = create_planner_agent(model_client=client)

        planner_start = time.perf_counter()
        plan_task_result: TaskResult = await planner.run(task=prompt)
        metrics.planner.execution_time = time.perf_counter() - planner_start

        plan_message = plan_task_result.messages[-1]
        plan_text = plan_message.content
        planner_usage = getattr(plan_message, "models_usage", None)
        if planner_usage is not None:
            metrics.planner.prompt_tokens = planner_usage.prompt_tokens
            metrics.planner.completion_tokens = planner_usage.completion_tokens

        yield PipelineEvent(type="planner_done", payload={"plan": plan_text})

        # --- Étape 2 : boucle Codeur <-> Reviewer (streaming réel) ---
        team = build_code_review_team(max_iterations=max_iterations, model_client=client)
        task_result: TaskResult | None = None

        # Checkpoint pour attribuer, à chaque message émis par le stream,
        # le temps écoulé depuis le message précédent -- une itération de
        # RoundRobinGroupChat ne produit qu'UN appel modèle (Codeur ou
        # Reviewer) entre deux yields, donc cet écart correspond bien au
        # temps réel de CET appel.
        loop_checkpoint = time.perf_counter()

        async for item in team.run_stream(task=plan_text):
            if isinstance(item, TaskResult):
                task_result = item
                continue

            now = time.perf_counter()
            elapsed = now - loop_checkpoint
            loop_checkpoint = now

            source = getattr(item, "source", "system")
            usage = getattr(item, "models_usage", None)
            if source == "CodeurAgent":
                metrics.coder.execution_time = (metrics.coder.execution_time or 0.0) + elapsed
                if usage is not None:
                    metrics.coder.prompt_tokens = (metrics.coder.prompt_tokens or 0) + usage.prompt_tokens
                    metrics.coder.completion_tokens = (metrics.coder.completion_tokens or 0) + usage.completion_tokens
            elif source == "ReviewerAgent":
                metrics.reviewer.execution_time = (metrics.reviewer.execution_time or 0.0) + elapsed
                if usage is not None:
                    metrics.reviewer.prompt_tokens = (metrics.reviewer.prompt_tokens or 0) + usage.prompt_tokens
                    metrics.reviewer.completion_tokens = (metrics.reviewer.completion_tokens or 0) + usage.completion_tokens

            content = getattr(item, "content", None)
            # Garantit une chaîne de caractères, quel que soit le type
            # réel du message AutoGen reçu -- ferme l'Issue 1 à la source.
            content_text = content if isinstance(content, str) else str(content)

            yield PipelineEvent(
                type="agent_message",
                payload={"source": source, "content": content_text},
            )

        if task_result is None:
            _log_now()
            yield PipelineEvent(type="pipeline_done", payload={"executed": False, "approved": False})
            return

        result = build_code_review_result(task_result)
        metrics.revision_count = result.iterations_used
        metrics.reviewer_status = "APPROVED" if result.approved else "REVISION_REQUISE"
        if result.final_code:
            metrics.code_length = len(result.final_code)
            metrics.code_line_count = result.final_code.count("\n") + 1

        yield PipelineEvent(
            type="review_done",
            payload={
                "approved": result.approved,
                "iterations_used": result.iterations_used,
                "final_code": result.final_code,
            },
        )

        # --- Étape 3 : exécution Docker ---
        if not result.approved or not result.final_code:
            # Docker jamais atteint : pas de métrique Docker disponible
            # (reste None), ce n'est pas une erreur d'infrastructure.
            _log_now()
            yield PipelineEvent(type="pipeline_done", payload={"executed": False, "approved": result.approved})
            return

        if _code_requires_interactive_input(result.final_code):
            metrics.docker_status = "SKIPPED_INTERACTIVE_INPUT"
            _log_now()
            yield PipelineEvent(type="docker_requires_input", payload={"message": _INTERACTIVE_INPUT_MESSAGE})
            yield PipelineEvent(type="pipeline_done", payload={"executed": False, "approved": True})
            return

        yield PipelineEvent(type="docker_start")
        docker_start = time.perf_counter()
        try:
            execution: ExecutionResult = await execute_approved_code(result)
        except DockerNotAvailableError as exc:
            metrics.docker_execution_time = time.perf_counter() - docker_start
            metrics.docker_status = "DOCKER_UNAVAILABLE"
            _log_now()
            yield PipelineEvent(type="docker_error", payload={"error": str(exc)})
            yield PipelineEvent(type="pipeline_done", payload={"executed": False, "approved": True})
            return

        metrics.docker_execution_time = time.perf_counter() - docker_start
        metrics.docker_status = "SUCCESS" if execution.success else "ERROR"
        metrics.docker_exit_code = execution.exit_code

        if not execution.success and _output_indicates_eof_error(execution.output):
            yield PipelineEvent(type="docker_requires_input", payload={"message": _INTERACTIVE_INPUT_MESSAGE})
        else:
            yield PipelineEvent(
                type="docker_done",
                payload={
                    "success": execution.success,
                    "exit_code": execution.exit_code,
                    "output": execution.output,
                },
            )

        _log_now()
        yield PipelineEvent(type="pipeline_done", payload={"executed": True, "approved": True})
    finally:
        # Filet de sécurité : ne s'exécute réellement que sur une
        # exception NON gérée dans le try (propagation synchrone => ce
        # `finally` est alors garanti d'être exécuté, contrairement au cas
        # "consommateur arrête d'itérer" décrit plus haut). Si un des
        # points de sortie normaux a déjà loggé via `_log_now()`, `logged`
        # est déjà `True` et cet appel est un no-op.
        _log_now()

        # Toujours exécuté DANS la même boucle asyncio, avant qu'elle ne se
        # ferme -- c'est la correction de l'Issue 2.
        await client.close()