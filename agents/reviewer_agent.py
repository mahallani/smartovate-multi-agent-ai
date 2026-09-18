# agents/reviewer_agent.py
"""
Module définissant le ReviewerAgent de Smartovate.

Rôle métier :
Le ReviewerAgent effectue une revue de code exigeante, comme le ferait
un Senior Software Engineer en pull request review. Il détecte les erreurs,
évalue les risques, vérifie la conformité au plan du PlannerAgent, et
propose des améliorations concrètes.

Le ReviewerAgent NE GÉNÈRE JAMAIS de solution de code complète. Il fournit
uniquement des retours ; c'est au CodeurAgent d'implémenter les corrections.

Gestion de la fenêtre de contexte (ticket "Context Window Exceeded") :
Voir la docstring détaillée dans agents/codeur_agent.py -- même mécanisme
et mêmes raisons ici : un TokenLimitedChatCompletionContext borné en tokens
réels (via tiktoken), plutôt que le UnboundedChatCompletionContext par
défaut d'AssistantAgent. Le ReviewerAgent reçoit potentiellement, à chaque
tour, plusieurs versions complètes du code du Codeur (l'historique de la
conversation partagée) : sans cette borne, il serait exposé exactement au
même risque de dépassement que le CodeurAgent.
"""

from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import TokenLimitedChatCompletionContext
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from config.settings import get_context_token_limit, get_model_client


REVIEWER_SYSTEM_MESSAGE = """Tu es le ReviewerAgent du système multi-agents Smartovate.
Tu es un Senior Software Engineer expert en revue de code Python, effectuant
une revue exigeante comme dans un vrai processus de pull request en production.

Ton unique rôle est de CRITIQUER le code produit par le CodeurAgent. Tu ne dois
JAMAIS :
- écrire une nouvelle version complète du code toi-même,
- réécrire des blocs de code entiers en remplacement du Codeur,
- approuver un code par complaisance sans l'avoir réellement analysé.

Pour chaque code reçu, analyse systématiquement les points suivants :

1. Erreurs fonctionnelles (bugs probables, cas non gérés, logique incorrecte).
2. Risques (exceptions non gérées, entrées non validées, effets de bord,
   problèmes de performance sur de gros fichiers).
3. Qualité et bonnes pratiques Python (typage, nommage, structure, PEP8,
   duplication de code, complexité inutile).
4. Conformité avec le plan initial fourni par le PlannerAgent (chaque étape
   du plan est-elle bien couverte par le code ?).

Format de réponse STRICT :

REVUE:
- Erreurs détectées : <liste ou "Aucune">
- Risques identifiés : <liste ou "Aucun">
- Problèmes de qualité / bonnes pratiques : <liste ou "Aucun">
- Conformité avec le plan : <analyse courte>
- Améliorations proposées : <liste concrète et actionnable, ou "Aucune">
STATUT: <APPROVED ou REVISION_REQUISE>
FIN_REVUE

Règles pour le champ STATUT :
- Écris exactement "STATUT: APPROVED" UNIQUEMENT si le code est fonctionnel,
  robuste, conforme au plan, et sans erreur ni risque significatif.
- Écris exactement "STATUT: REVISION_REQUISE" si au moins un point bloquant
  a été identifié. Dans ce cas, les "Améliorations proposées" doivent être
  suffisamment précises pour que le CodeurAgent puisse corriger sans ambiguïté.

Ne mélange jamais les deux mots "APPROVED" et "REVISION_REQUISE" dans la
même réponse en dehors du champ STATUT.

Mot-clé de contrôle TERMINATE :
- "TERMINATE" est un mot-clé RÉSERVÉ qui commande l'arrêt de la conversation
  entre agents. TU ES LE SEUL AGENT AUTORISÉ À L'ÉCRIRE.
- Si, et seulement si, STATUT: APPROVED, ajoute une dernière ligne contenant
  EXACTEMENT et UNIQUEMENT le mot "TERMINATE", juste après "FIN_REVUE".
  Exemple :
      ...
      STATUT: APPROVED
      FIN_REVUE
      TERMINATE
- Si STATUT: REVISION_REQUISE, n'écris JAMAIS le mot "TERMINATE", nulle part
  dans ta réponse (ni dans le corps de la revue, ni en commentaire, ni en
  exemple).
- N'utilise jamais "TERMINATE" pour un autre usage que celui décrit ici.
"""


def create_reviewer_agent(
    model_client: AzureOpenAIChatCompletionClient | None = None,
) -> AssistantAgent:
    """
    Fabrique du ReviewerAgent.

    Args:
        model_client: client Azure OpenAI. Si None, utilise la config centralisée.

    Returns:
        AssistantAgent nommé "ReviewerAgent", avec un contexte de
        conversation borné en tokens (voir docstring du module).
    """
    # Limite de sortie (ticket "Optimisation des coûts et des prompts") :
    # le ReviewerAgent produit une revue textuelle structurée, sans code
    # généré, 4000 tokens de sortie est cohérent avec son format de réponse.
    client = model_client or get_model_client(max_completion_tokens=4000)

    # Aligne la marge réservée pour la réponse (contexte d'entrée) sur la
    # vraie limite de sortie du Reviewer (4000), au lieu du défaut
    # CONTEXT_RESERVED_FOR_COMPLETION (pensé pour le Codeur, 16000) :
    # évite de réserver inutilement de la place côté input pour une sortie
    # qui ne dépassera jamais 4000 tokens.
    model_context = TokenLimitedChatCompletionContext(
        model_client=client,
        token_limit=get_context_token_limit(client, reserved_for_completion=4000),
    )

    return AssistantAgent(
        name="ReviewerAgent",
        model_client=client,
        model_context=model_context,
        system_message=REVIEWER_SYSTEM_MESSAGE,
        description=(
            "Agent chargé de la revue critique du code produit par le "
            "CodeurAgent : erreurs, risques, qualité, conformité au plan."
        ),
    )