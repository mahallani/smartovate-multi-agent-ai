# tests/test_planner_agent.py
"""
Test unitaire léger pour le PlannerAgent.
Ne fait PAS d'appel réseau réel à Azure OpenAI : on vérifie uniquement
la bonne construction de l'agent (nom, présence du prompt système).
Pour un test end-to-end avec un vrai appel LLM, utiliser
scripts/test_planner_groupchat.py manuellement.
"""

from agents.planner_agent import PLANNER_SYSTEM_MESSAGE, create_planner_agent
from config.settings import get_model_client


def test_planner_agent_creation():
    model_client = get_model_client()
    planner = create_planner_agent(model_client=model_client)

    assert planner.name == "PlannerAgent"
    assert "PLAN:" in PLANNER_SYSTEM_MESSAGE
    assert "FIN_PLAN" in PLANNER_SYSTEM_MESSAGE