# agents/stub_agents.py
"""
Agents temporaires (stubs) utilisés UNIQUEMENT en attendant l'implémentation
réelle de ExecutorAgent, CriticAgent et ReporterAgent.

⚠️ À SUPPRIMER / REMPLACER dès que les tickets correspondants sont terminés.
Ces agents ne font pas de vrai travail : ils se contentent d'acquitter
la réception du plan, afin de permettre de tester le flux complet
PlannerAgent -> GroupChat sans dépendre des futurs agents.
"""

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from config.settings import get_model_client


def create_stub_executor_agent(
    model_client: AzureOpenAIChatCompletionClient | None = None,
) -> AssistantAgent:
    """Stub temporaire simulant l'ExecutorAgent."""
    client = model_client or get_model_client()
    return AssistantAgent(
        name="ExecutorAgent",
        model_client=client,
        system_message=(
            "Tu es un agent temporaire de test (stub). Quand tu reçois un plan, "
            "réponds simplement : 'Plan bien reçu par ExecutorAgent (stub). "
            "En attente de l'implémentation réelle.'"
        ),
        description="Stub temporaire de l'ExecutorAgent (en cours de développement).",
    )


def create_stub_critic_agent(
    model_client: AzureOpenAIChatCompletionClient | None = None,
) -> AssistantAgent:
    """Stub temporaire simulant le CriticAgent."""
    client = model_client or get_model_client()
    return AssistantAgent(
        name="CriticAgent",
        model_client=client,
        system_message=(
            "Tu es un agent temporaire de test (stub). Réponds uniquement : "
            "'APPROVE' pour permettre au flux de test de se terminer."
        ),
        description="Stub temporaire du CriticAgent (en cours de développement).",
    )