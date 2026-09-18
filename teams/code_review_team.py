# teams/code_review_team.py
"""
Composition de l'équipe Codeur/Reviewer de Smartovate.

Ce module encapsule la boucle d'amélioration automatique de code :

    CodeurAgent (génère) -> ReviewerAgent (critique)
         ^                         |
         |________ feedback _______|

La boucle s'arrête dès que le ReviewerAgent répond "STATUT: APPROVED",
ou après un nombre maximal d'itérations (garde-fou anti-boucle infinie).

Ce module n'importe volontairement PAS le PlannerAgent : il reçoit le plan
sous forme de texte brut en entrée (paramètre `task` de `run` / `run_stream`).
Ce découplage permet de faire évoluer indépendamment le Planner et la
boucle de génération de code.
"""

from dataclasses import dataclass
from pathlib import Path

from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from agents.codeur_agent import create_codeur_agent
from agents.reviewer_agent import create_reviewer_agent
from teams.docker_executor import ExecutionResult, execute_code_in_docker
from teams.termination import ReviewerApprovalTermination


@dataclass
class CodeReviewResult:
    """
    Résultat structuré de la boucle Codeur/Reviewer, pensé pour être
    consommé facilement par les futurs ExecutorAgent / orchestrateur.
    """
    approved: bool
    iterations_used: int
    final_code: str | None
    task_result: TaskResult


def build_code_review_team(
    max_iterations: int = 5,
    model_client: AzureOpenAIChatCompletionClient | None = None,
) -> RoundRobinGroupChat:
    """
    Construit la Team (GroupChat) responsable de la boucle de génération
    et de revue de code.

    Pourquoi RoundRobinGroupChat :
    Avec seulement 2 participants qui doivent alterner strictement
    (Codeur -> Reviewer -> Codeur -> Reviewer -> ...), un ordre déterministe
    est exactement ce qu'il faut. Un SelectorGroupChat introduirait une
    décision LLM inutile et potentiellement instable, alors que la logique
    métier ici est strictement séquentielle et connue à l'avance.

    Args:
        max_iterations: nombre maximal de cycles Codeur->Reviewer autorisés
            avant l'arrêt forcé de la boucle (garde-fou).
        model_client: client Azure OpenAI partagé. Si None, chaque agent
            crée sa propre instance via la config centralisée.

    Returns:
        Une RoundRobinGroupChat prête à être exécutée avec `.run(task=...)`
        ou `.run_stream(task=...)`.
    """
    codeur = create_codeur_agent(model_client=model_client)
    reviewer = create_reviewer_agent(model_client=model_client)

    # Condition d'arrêt n°1 : le Reviewer a explicitement clos la revue avec
    # le mot-clé de contrôle "TERMINATE". Cette condition est volontairement
    # RESTREINTE au ReviewerAgent (voir teams/termination.py) : si le code
    # généré par le CodeurAgent contient un jour la sous-chaîne "TERMINATE"
    # (variable, commentaire, docstring...), elle est ignorée -- seul un
    # TERMINATE émis par le Reviewer peut arrêter la boucle.
    #
    # Ce signal de contrôle ("TERMINATE") est volontairement DÉCOUPLÉ du
    # signal métier ("STATUT: APPROVED") lu par `run_code_review_loop` :
    # le premier pilote la Team (arrêt du GroupChat), le second qualifie le
    # résultat (le code est-il réellement validé). Les découpler évite
    # qu'une évolution du format métier (ex: renommer "STATUT" en "STATUS")
    # casse silencieusement l'arrêt de la boucle.
    approval_condition = ReviewerApprovalTermination(
        keyword="TERMINATE",
        source="ReviewerAgent",
    )

    # Condition d'arrêt n°2 (garde-fou anti-boucle infinie) : nombre max de
    # messages atteint. C'est l'équivalent, en autogen-agentchat 0.7.x, du
    # `max_consecutive_auto_reply` de pyautogen (qui n'existe plus dans cette
    # API) : ici on borne strictement la conversation au niveau de la Team
    # plutôt qu'au niveau d'un agent individuel.
    # Calcul : 1 message initial (le plan, injecté comme "task") puis
    # 2 messages par itération (1 réponse Codeur + 1 réponse Reviewer).
    max_messages = 1 + (2 * max_iterations)
    safety_condition = MaxMessageTermination(max_messages)

    # Combinaison : la boucle s'arrête au premier des deux événements.
    # -> Si TERMINATE arrive : arrêt propre, coût maîtrisé.
    # -> Si TERMINATE n'arrive jamais : arrêt forcé après max_iterations,
    #    donc consommation de tokens Azure OpenAI toujours bornée.
    termination = approval_condition | safety_condition

    return RoundRobinGroupChat(
        participants=[codeur, reviewer],
        termination_condition=termination,
    )


def extract_last_code_block(text: str) -> str | None:
    """
    Extrait le dernier bloc de code Python (```python ... ```) d'un texte.

    Utilitaire nécessaire car le résultat final utile (le code validé)
    doit être récupéré depuis le dernier message du CodeurAgent, dans un
    format prévisible (imposé par CODEUR_SYSTEM_MESSAGE).

    Args:
        text: le contenu brut du message du CodeurAgent.

    Returns:
        Le code extrait, ou None si aucun bloc n'a été trouvé.
    """
    marker_start = "```python"
    marker_end = "```"

    start_index = text.rfind(marker_start)
    if start_index == -1:
        return None

    start_index += len(marker_start)
    end_index = text.find(marker_end, start_index)
    if end_index == -1:
        return None

    return text[start_index:end_index].strip()


def build_code_review_result(task_result: TaskResult) -> CodeReviewResult:
    """
    Construit un `CodeReviewResult` structuré à partir d'un `TaskResult`
    déjà obtenu (via `team.run(...)` OU `Console(team.run_stream(...))`,
    qui renvoie lui aussi le `TaskResult` final).

    Extrait dans sa propre fonction pour un motif précis : `Console`
    (utilisée pour l'affichage en streaming) retourne déjà le `TaskResult`
    complet à la fin du streaming. Sans cette fonction, on serait tenté de
    relancer `run_code_review_loop` juste après pour obtenir un résultat
    structuré -- ce qui repartirait de zéro et ferait payer une SECONDE
    fois la totalité des appels Azure OpenAI de la conversation déjà
    terminée. Cette fonction permet de réutiliser le `TaskResult` déjà en
    main, sans appel LLM supplémentaire.

    Args:
        task_result: le résultat de la Team déjà exécutée.

    Returns:
        CodeReviewResult contenant le statut d'approbation, le nombre
        d'itérations utilisées, le code final extrait, et le résultat
        brut de la Team (pour debug/logs).
    """
    # Le dernier message du ReviewerAgent indique si le code est approuvé.
    last_reviewer_message = next(
        (
            msg.content
            for msg in reversed(task_result.messages)
            if msg.source == "ReviewerAgent"
        ),
        "",
    )
    approved = "STATUT: APPROVED" in last_reviewer_message

    # Le code final utile est celui du DERNIER message du CodeurAgent
    # (la version la plus récente, potentiellement corrigée).
    last_codeur_message = next(
        (
            msg.content
            for msg in reversed(task_result.messages)
            if msg.source == "CodeurAgent"
        ),
        "",
    )
    final_code = extract_last_code_block(last_codeur_message)

    # Nombre d'itérations réellement utilisées : chaque itération = 2 messages
    # (1 Codeur + 1 Reviewer), hors message de tâche initial.
    iterations_used = (len(task_result.messages) - 1) // 2

    return CodeReviewResult(
        approved=approved,
        iterations_used=iterations_used,
        final_code=final_code,
        task_result=task_result,
    )


async def run_code_review_loop(
    plan: str,
    max_iterations: int = 5,
    model_client: AzureOpenAIChatCompletionClient | None = None,
) -> CodeReviewResult:
    """
    Point d'entrée principal : exécute la boucle Codeur/Reviewer à partir
    d'un plan (texte produit par le PlannerAgent) et retourne un résultat
    structuré et exploitable par le reste du pipeline (futur ExecutorAgent).

    N'utilisez cette fonction QUE si vous n'avez pas déjà exécuté la Team
    par ailleurs (ex: via `Console(team.run_stream(...))`) : dans ce cas,
    utilisez plutôt `build_code_review_result(task_result)` directement
    sur le `TaskResult` déjà obtenu, pour ne pas payer deux fois les
    appels Azure OpenAI de la même conversation (voir docstring de
    `build_code_review_result`).

    Args:
        plan: le plan d'action, généralement produit par PlannerAgent.
        max_iterations: nombre max de cycles Codeur<->Reviewer.
        model_client: client Azure OpenAI partagé (optionnel).

    Returns:
        CodeReviewResult contenant le statut d'approbation, le nombre
        d'itérations utilisées, le code final extrait, et le résultat
        brut de la Team (pour debug/logs).
    """
    team = build_code_review_team(
        max_iterations=max_iterations,
        model_client=model_client,
    )

    task_result: TaskResult = await team.run(task=plan)

    return build_code_review_result(task_result)


def save_final_code_to_workspace(
    result: CodeReviewResult,
    filename: str = "generated_script.py",
) -> Path:
    """
    Sauvegarde le code final approuvé dans le dossier workspace/, prêt à
    être exécuté par le futur ExecutorAgent.

    Args:
        result: le résultat de la boucle Codeur/Reviewer.
        filename: nom du fichier de sortie.

    Returns:
        Le chemin du fichier créé.
    """
    Path("workspace").mkdir(exist_ok=True)
    output_path = Path("workspace") / filename

    if result.final_code:
        output_path.write_text(result.final_code, encoding="utf-8")

    return output_path


async def execute_approved_code(
    result: CodeReviewResult,
    work_dir: str = "./_docker_workspace",
) -> ExecutionResult:
    """
    Exécute, dans un conteneur Docker isolé, le code final de la boucle
    Codeur/Reviewer -- PRÉPARATION du futur ExecutorAgent dans le pipeline
    complet (Planner -> Codeur <-> Reviewer -> Executor).

    Garde-fou (défense en profondeur) : cette fonction refuse d'exécuter
    tout code qui n'a pas été explicitement approuvé par le ReviewerAgent
    (`result.approved is False`) ou dont le code n'a pas pu être extrait
    (`result.final_code is None`). Un code en REVISION_REQUISE ne doit
    JAMAIS être exécuté, même dans un environnement isolé : Docker protège
    la machine hôte, il ne dispense pas de la revue métier.

    Args:
        result: le résultat de `run_code_review_loop`.
        work_dir: dossier de travail local pour l'exécuteur Docker.

    Returns:
        ExecutionResult(success, exit_code, output).

    Raises:
        ValueError: si le code n'a pas été approuvé ou n'a pas pu être
            extrait -- volontairement bloquant, pour qu'aucun appelant
            ne puisse exécuter du code non validé par erreur.
    """
    if not result.approved or not result.final_code:
        raise ValueError(
            "Exécution refusée : le code n'a pas été approuvé par le "
            "ReviewerAgent (STATUT: APPROVED) ou n'a pas pu être extrait."
        )

    return await execute_code_in_docker(code=result.final_code, work_dir=work_dir)