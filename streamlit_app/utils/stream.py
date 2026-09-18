# streamlit_app/utils/stream.py
"""
Pont entre le monde asynchrone du backend (async def / await / générateurs
async) et Streamlit, qui est synchrone.

Principe :
Le générateur asynchrone (ex: team.run_stream(task=...)) est consommé dans
un thread dédié. Chaque événement reçu (message d'agent, ou TaskResult final)
est poussé dans une queue.Queue thread-safe. Le thread principal de
Streamlit lit cette queue en boucle et met à jour l'affichage au fur et à
mesure -- c'est ce qui permet l'affichage "temps réel" exigé par le ticket,
sans jamais bloquer en attendant la fin complète de la conversation.
"""
from __future__ import annotations

import asyncio
import queue
import sys
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from autogen_agentchat.base import TaskResult

_SENTINEL = object()


@dataclass
class StreamEvent:
    """Un événement reçu depuis le générateur asynchrone du backend."""

    kind: str  # "message" | "result" | "error"
    payload: Any


def run_stream_in_thread(
    async_gen_factory: Callable[[], AsyncIterator[Any]],
) -> "queue.Queue[StreamEvent]":
    """
    Lance un générateur asynchrone dans un thread séparé et retourne une
    queue.Queue alimentée en temps réel.

    Args:
        async_gen_factory: fonction sans argument qui, une fois appelée,
            retourne le générateur asynchrone à consommer (ex:
            `lambda: backend.obtenir_flux_code_review(team, plan)`).

    Returns:
        Une queue.Queue thread-safe. Chaque élément est un StreamEvent, ou
        un objet sentinel interne signalant la fin du flux (à consommer via
        `drain_queue`, jamais directement).
    """
    q: "queue.Queue[StreamEvent]" = queue.Queue()

    def _runner() -> None:
        async def _consume() -> None:
            try:
                async for item in async_gen_factory():
                    if isinstance(item, TaskResult):
                        q.put(StreamEvent(kind="result", payload=item))
                    else:
                        # --- [STREAM DEBUG] diagnostic temporaire (aucune
                        # logique modifiée -- même branchement qu'avant) ---
                        if getattr(item, "source", None) == "MLRiskPredictor":
                            print("[STREAM DEBUG] received MLRiskEvent", file=sys.stderr)
                        q.put(StreamEvent(kind="message", payload=item))
                        if getattr(item, "source", None) == "MLRiskPredictor":
                            print("[STREAM DEBUG] forwarding MLRiskEvent", file=sys.stderr)
                        # --- fin [STREAM DEBUG] ---
            except Exception as exc:  # noqa: BLE001 - remonté à l'UI, pas ignoré
                q.put(StreamEvent(kind="error", payload=exc))
            finally:
                q.put(_SENTINEL)  # type: ignore[arg-type]

        asyncio.run(_consume())

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    return q


def drain_queue(
    q: "queue.Queue[StreamEvent]",
    block_timeout: float = 0.3,
) -> tuple[list[StreamEvent], bool]:
    """
    Récupère les événements actuellement disponibles dans la queue.

    Attend au maximum `block_timeout` secondes pour le premier élément (pour
    éviter une boucle d'appel trop agressive côté appelant), puis vide le
    reste de la queue sans attendre.

    Returns:
        (liste des événements reçus, True si le flux est terminé).
    """
    events: list[StreamEvent] = []
    finished = False

    try:
        first = q.get(timeout=block_timeout)
        if first is _SENTINEL:
            return events, True
        events.append(first)  # type: ignore[arg-type]
    except queue.Empty:
        return events, False

    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        if item is _SENTINEL:
            finished = True
            break
        events.append(item)  # type: ignore[arg-type]

    return events, finished