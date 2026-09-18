# teams/termination.py
"""
Conditions de terminaison custom pour les Teams AutoGen de Smartovate.

Pourquoi ce module existe :
AutoGen 0.7.x (autogen-agentchat) ne fournit plus le paramètre historique
`max_consecutive_auto_reply` de pyautogen. L'équivalent moderne est la
composition de `TerminationCondition` (MaxMessageTermination,
TextMentionTermination, etc.), combinables avec les opérateurs `|` (OR)
et `&` (AND) sur une Team.

`TextMentionTermination` native fait cependant un simple test `in` sur le
contenu de TOUS les messages, quel que soit l'agent qui les a émis. Dans
notre boucle Codeur/Reviewer, cela pose un risque réel : le CodeurAgent
génère du code Python librement ; si ce code contient un jour, même par
hasard, la sous-chaîne "TERMINATE" (nom de variable, docstring, commentaire,
chaîne de log...), la conversation s'arrêterait immédiatement après le
message du Codeur -- AVANT que le ReviewerAgent ait pu se prononcer. Le
pipeline validerait alors un code qui n'a jamais été relu. C'est un faux
positif silencieux et dangereux.

`ReviewerApprovalTermination` corrige ce problème en restreignant la
détection du mot-clé à un unique émetteur autorisé (le ReviewerAgent),
en plus d'être combinable avec les conditions de terminaison standard
(ex: `MaxMessageTermination`) via l'opérateur `|`.
"""

from __future__ import annotations

from typing import Sequence

from autogen_agentchat.base import TerminatedException, TerminationCondition
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, StopMessage


class ReviewerApprovalTermination(TerminationCondition):
    """
    Condition d'arrêt déclenchée uniquement quand `keyword` (par défaut
    "TERMINATE") apparaît dans un message émis par `source` (par défaut
    "ReviewerAgent").

    Contrairement à `TextMentionTermination`, cette condition ignore le
    mot-clé s'il provient de n'importe quel autre agent (typiquement le
    CodeurAgent), ce qui élimine le risque de faux positif décrit dans
    la docstring du module.
    """

    def __init__(self, keyword: str = "TERMINATE", source: str = "ReviewerAgent") -> None:
        self._keyword = keyword
        self._source = source
        self._terminated = False

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(
        self,
        messages: Sequence[BaseAgentEvent | BaseChatMessage],
    ) -> StopMessage | None:
        if self._terminated:
            raise TerminatedException(
                "La condition de terminaison a déjà été déclenchée."
            )

        for message in messages:
            if message.source != self._source:
                continue

            content = message.content if isinstance(message.content, str) else str(message.content)
            if self._keyword in content:
                self._terminated = True
                return StopMessage(
                    content=(
                        f"Mot-clé de contrôle '{self._keyword}' détecté dans "
                        f"un message de '{self._source}'."
                    ),
                    source="ReviewerApprovalTermination",
                )

        return None

    async def reset(self) -> None:
        self._terminated = False