# scripts/test_codeur_reviewer.py
"""
Script de test manuel du pipeline Planner -> Codeur/Reviewer -> Executor.

Objectif : vérifier que
  1. le plan produit par le PlannerAgent est correctement transmis à la
     CodeReviewTeam,
  2. le CodeurAgent génère du code à partir de ce plan,
  3. le ReviewerAgent critique ce code (erreurs, risques, qualité, conformité),
  4. la boucle s'arrête bien sur "TERMINATE" ou sur le nombre max
     d'itérations,
  5. le code approuvé est exécuté UNIQUEMENT dans un conteneur Docker
     isolé -- jamais directement sur la machine hôte.

Ce script utilise le mode streaming (Console + run_stream) pour afficher
chaque message dès qu'il est prêt, afin de voir la progression en temps réel.
"""

import asyncio

from autogen_agentchat.ui import Console

from agents.planner_agent import create_planner_agent
from teams.code_review_team import build_code_review_result, build_code_review_team
from teams.docker_executor import DockerNotAvailableError, execute_code_in_docker


async def main() -> None:
    planner = create_planner_agent()

    demande_utilisateur = (
        "Écris une fonction Python qui additionne deux nombres."
    )

    print("=== Étape 1 : génération du plan ===\n")
    plan_result = await planner.run(task=demande_utilisateur)
    plan_texte = plan_result.messages[-1].content
    print(plan_texte)

    print("\n=== Étape 2 : boucle Codeur <-> Reviewer (streaming) ===\n")
    team = build_code_review_team(max_iterations=5)
    # `Console` retourne le TaskResult final du streaming : on le réutilise
    # directement (voir build_code_review_result) plutôt que de relancer
    # toute la conversation, ce qui doublerait les appels Azure OpenAI.
    task_result = await Console(team.run_stream(task=plan_texte))

    print("\n=== Étape 2bis : résultat structuré (sans nouvel appel LLM) ===\n")
    result = build_code_review_result(task_result)
    print(f"approved={result.approved} | iterations_used={result.iterations_used}")

    print("\n=== Étape 3 : exécution du code approuvé dans Docker ===\n")
    if result.approved and result.final_code:
        try:
            execution = await execute_code_in_docker(result.final_code)
            print(f"success={execution.success} | exit_code={execution.exit_code}")
            print("output:\n", execution.output)
        except DockerNotAvailableError as exc:
            print(f"[Docker indisponible] {exc}")
            print(
                "-> Le code a été validé (STATUT: APPROVED) mais n'a pas pu "
                "être exécuté faute de daemon Docker accessible. Aucun code "
                "n'a été exécuté sur la machine hôte."
            )
    else:
        print("Code non approuvé : aucune exécution effectuée (garde-fou).")

    await demo_dangerous_code_is_sandboxed()


async def demo_dangerous_code_is_sandboxed() -> None:
    """
    Démonstration du scénario de test demandé dans le ticket sécurité :
    le CodeurAgent génère volontairement du code dangereux
    (`os.remove("important.txt")`), et on prouve qu'il ne s'exécute
    JAMAIS sur la machine hôte -- uniquement dans le conteneur Docker
    éphémère, où le fichier ciblé n'existe pas et où toute suppression
    n'affecte que le système de fichiers du conteneur, détruit juste
    après l'exécution.
    """
    print("\n=== Démonstration : code dangereux, exécution sandboxée ===\n")

    dangerous_code = (
        "import os\n"
        "print('Tentative de suppression dans le conteneur...')\n"
        "try:\n"
        "    os.remove('important.txt')\n"
        "    print('Fichier supprimé (dans le conteneur uniquement).')\n"
        "except FileNotFoundError:\n"
        "    print('important.txt introuvable dans le conteneur : '\n"
        "          'la machine hôte Windows n\\'a jamais été exposée.')\n"
    )

    execution = None
    try:
        execution = await execute_code_in_docker(dangerous_code)
    except DockerNotAvailableError as exc:
        print(f"[Docker indisponible] {exc}")
        print(
            "-> Impossible de faire la démonstration sans Docker en cours "
            "d'exécution. Mais notez bien : ce message vient d'un ÉCHEC DE "
            "CONNEXION AU DAEMON, PAS d'une exécution sur la machine hôte. "
            "Aucune tentative de suppression n'a eu lieu sur ce PC Windows, "
            "précisément parce que ce code ne s'exécute JAMAIS ailleurs que "
            "dans le conteneur Docker."
        )
        return

    print(f"success={execution.success} | exit_code={execution.exit_code}")
    print("output:\n", execution.output)
    print(
        "\n-> Le fichier 'important.txt' de votre machine Windows n'a "
        "subi AUCUNE modification : le code s'est exécuté dans le "
        "système de fichiers isolé et éphémère du conteneur Docker, "
        "détruit immédiatement après l'exécution."
    )


if __name__ == "__main__":
    asyncio.run(main())