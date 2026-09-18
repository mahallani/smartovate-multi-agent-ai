# streamlit_app/utils/models.py
"""
Structures de données internes au FRONTEND uniquement.

Ces dataclasses ne sont jamais utilisées par le backend (agents/, teams/) --
elles servent à typer proprement ce qui circule entre le stream, l'état de
session Streamlit et les composants d'affichage, sans manipuler directement
les objets AutoGen bruts (TaskResult, TextMessage...) dans l'UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

AgentName = Literal[
    "PlannerAgent", "CodeurAgent", "ReviewerAgent", "DockerExecutor", "user"
]


@dataclass
class AgentMessage:
    """Un message affichable dans la timeline et les cartes agents."""

    source: AgentName
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationSession:
    """
    Une conversation complète (un prompt + tout son déroulé), conservée pour
    l'historique et l'export.
    """

    id: str
    prompt: str
    created_at: datetime
    messages: list[AgentMessage] = field(default_factory=list)

    plan: str | None = None
    final_code: str | None = None
    approved: bool | None = None
    iterations_used: int = 0

    docker_output: str | None = None
    docker_success: bool | None = None
    docker_exit_code: int | None = None
    docker_error_message: str | None = None

    duration_seconds: float | None = None
    status: Literal["running", "done", "error"] = "running"
