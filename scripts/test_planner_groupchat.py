# scripts/test_planner_groupchat.py
"""
Script de test manuel pour le PlannerAgent.

Objectif : vérifier que
  1. le PlannerAgent génère bien un plan structuré à partir d'une requête,
  2. ce plan est correctement transmis et visible par les autres participants
     d'un GroupChat (ici, des stubs temporaires).
"""

import asyncio

from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

from agents.planner_agent import create_planner_agent
from agents.stub_agents import create_stub_critic_agent, create_stub_executor_agent


async def main() -> None:
    # 1. Création des agents (un seul modèle client réutilisé pour tous).
    planner = create_planner_agent()
    executor_stub = create_stub_executor_agent()
    critic_stub = create_stub_critic_agent()

    # 2. Condition d'arrêt : "APPROVE" du critic, ou 6 messages max (anti-boucle).
    termination = TextMentionTermination("APPROVE") | MaxMessageTermination(6)

    # 3. Assemblage du GroupChat (Team) en RoundRobin.
    team = RoundRobinGroupChat(
        participants=[planner, executor_stub, critic_stub],
        termination_condition=termination,
    )

    # 4. Requête utilisateur complexe de test.
    task = (
        "Analyse le fichier ventes_2025.csv, calcule le chiffre d'affaires "
        "mensuel, et génère un rapport PDF avec les résultats."
    )

    # 5. Exécution et affichage en direct dans la console.
    await Console(team.run_stream(task=task))
def afficher_plan_lisible(plan_brut: str) -> None:
    """
    Affiche le plan du PlannerAgent de façon claire pour un utilisateur
    non technique, sans passer par une interface graphique.
    """
    print("\n" + "=" * 50)
    print("📋 VOICI LE PLAN PROPOSÉ POUR VOTRE DEMANDE")
    print("=" * 50)

    # On extrait uniquement le contenu entre "PLAN:" et "FIN_PLAN"
    contenu = plan_brut.replace("PLAN:", "").replace("FIN_PLAN", "").strip()
    print(contenu)

    print("=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())