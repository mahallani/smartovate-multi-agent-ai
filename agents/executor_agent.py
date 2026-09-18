# agents/executor_agent.py
"""
Module définissant le (futur) ExecutorAgent de Smartovate.

Rôle métier :
L'ExecutorAgent EXÉCUTE le code déjà validé par le ReviewerAgent, dans un
environnement Docker isolé. Il ne génère jamais de code (pas de
`model_client`) et ne juge jamais de sa qualité (rôle du ReviewerAgent) :
son unique responsabilité est l'exécution sécurisée (Single Responsibility
Principle, comme pour CodeurAgent et ReviewerAgent).

Ce module prépare l'intégration future dans le pipeline complet :

    PlannerAgent -> CodeurAgent <-> ReviewerAgent -> ExecutorAgent

Pourquoi CodeExecutorAgent sans model_client :
Utilisé sans `model_client`, `CodeExecutorAgent` ne fait qu'exécuter les
blocs de code trouvés dans les messages reçus (aucun appel LLM, donc
aucun coût Azure OpenAI additionnel, et un comportement déterministe --
pertinent pour un agent dont le rôle est purement mécanique).

Pourquoi `sources=["CodeurAgent"]` :
Cette restriction reproduit le principe déjà appliqué à
`ReviewerApprovalTermination` (voir teams/termination.py) : ne jamais
faire confiance à la provenance d'un message sans la vérifier
explicitement. Sans ce filtre, l'ExecutorAgent exécuterait potentiellement
du texte contenant des blocs ```python``` provenant du ReviewerAgent
(citations de code dans une critique, exemples illustratifs, etc.), ce
qui n'a jamais de sens métier ici.
"""

from autogen_agentchat.agents import CodeExecutorAgent
from autogen_core.code_executor import CodeExecutor


def create_executor_agent(code_executor: CodeExecutor) -> CodeExecutorAgent:
    """
    Fabrique de l'ExecutorAgent.

    Args:
        code_executor: l'exécuteur de code à utiliser. DOIT être un
            exécuteur isolé (ex: DockerCommandLineCodeExecutor). Ce projet
            n'utilise jamais LocalCommandLineCodeExecutor pour du code
            généré par un agent -- voir teams/docker_executor.py.

    Returns:
        Un CodeExecutorAgent nommé "ExecutorAgent", n'exécutant que les
        blocs de code trouvés dans les messages du CodeurAgent.
    """
    return CodeExecutorAgent(
        name="ExecutorAgent",
        code_executor=code_executor,
        sources=["CodeurAgent"],
        description=(
            "Agent chargé d'exécuter, dans un environnement Docker isolé, "
            "le code déjà validé (STATUT: APPROVED) par le ReviewerAgent."
        ),
    )