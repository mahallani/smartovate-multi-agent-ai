# agents/codeur_agent.py
"""
Module définissant le CodeurAgent de Smartovate.

Rôle métier :
Le CodeurAgent génère du code Python à partir :
  - des instructions du PlannerAgent (première génération),
  - du feedback détaillé du ReviewerAgent (itérations suivantes).

Le CodeurAgent NE VÉRIFIE JAMAIS son propre code. Cette responsabilité
appartient exclusivement au ReviewerAgent (Single Responsibility Principle).

Gestion de la fenêtre de contexte (ticket "Context Window Exceeded") :
Par défaut, AssistantAgent utilise un UnboundedChatCompletionContext (voir
autogen_agentchat.agents._assistant_agent.AssistantAgent.__init__) : TOUT
l'historique de la conversation Codeur<->Reviewer est renvoyé à chaque appel
Azure OpenAI. Comme cette boucle peut aller jusqu'à 10 itérations
(max_iterations réglable côté interface) et que chaque message du Codeur
contient un script complet, ce contexte croît vite et peut dépasser la
fenêtre du modèle bien avant la dernière itération autorisée.

On fournit donc explicitement un TokenLimitedChatCompletionContext :
  - il mesure le nombre RÉEL de tokens (via tiktoken, à travers
    model_client.count_tokens()), pas un nombre de messages -- seule
    approche qui garantit de ne jamais dépasser la fenêtre, même avec des
    messages de taille très variable comme ici,
  - il retire les messages les plus anciens (par le milieu de l'historique)
    un par un, en recalculant à chaque fois, jusqu'à repasser sous le
    budget -- donc il conserve TOUJOURS le plus de messages possible.

Le system_message (les règles ci-dessous) n'est jamais affecté par ce
mécanisme : AssistantAgent le stocke séparément et le réinjecte à chaque
appel, quel que soit l'état du model_context.
"""

from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import TokenLimitedChatCompletionContext
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from config.settings import get_context_token_limit, get_model_client


CODEUR_SYSTEM_MESSAGE = """Tu es le CodeurAgent du système multi-agents Smartovate.
Tu es un développeur Python senior spécialisé en analyse de données et automatisation.

Ton unique rôle est de PRODUIRE du code Python. Tu ne dois jamais :
- valider ou approuver toi-même ton propre code,
- juger de la qualité de ton propre travail,
- ignorer un feedback qui t'est adressé par le ReviewerAgent.

Comportement attendu :

1. Lors de la PREMIÈRE génération, tu reçois un plan d'action détaillé.
   Traduis ce plan en un script Python complet, fonctionnel, et cohérent
   avec chaque étape du plan.

2. Lors des générations SUIVANTES, tu reçois un retour de revue de code
   (ReviewerAgent) contenant des erreurs, risques, ou améliorations à
   apporter. Tu dois alors PRODUIRE UNE NOUVELLE VERSION COMPLÈTE du code,
   en corrigeant réellement chaque point soulevé (ne jamais ignorer un
   point de feedback, ne jamais répondre par une justification sans
   modifier le code).

3. Règles de qualité de code à respecter systématiquement :
   - code Python 3.11 propre, typé (type hints), avec docstrings,
   - gestion des erreurs (try/except) pour les opérations à risque
     (lecture de fichiers, conversions de types, etc.),
   - pas de valeurs en dur non justifiées,
   - respect de PEP8,
   - le script doit être autonome et exécutable tel quel.

4. Format de réponse STRICT :
   - Un court paragraphe (2-3 phrases MAXIMUM) résumant ce que fait le code
     et, si applicable, ce que tu as corrigé depuis la version précédente.
   - Puis le code complet dans UN SEUL bloc ```python ... ```.
   - Ne mets AUCUN texte après le bloc de code.

5. Mot-clé de contrôle réservé :
   - "TERMINATE" est un mot-clé technique réservé au ReviewerAgent : il sert
     UNIQUEMENT à signaler la fin de la conversation entre agents.
   - Tu ne dois JAMAIS écrire le mot "TERMINATE" toi-même : ni dans ton
     paragraphe de résumé, ni dans le code (noms de variables/fonctions/
     classes, commentaires, docstrings, chaînes de caractères, logs), même
     à titre d'exemple ou de test. Utilise un autre terme (ex: "stop",
     "arret", "fin_traitement") si tu as besoin d'exprimer une idée similaire.
"""


def create_codeur_agent(
    model_client: AzureOpenAIChatCompletionClient | None = None,
) -> AssistantAgent:
    """
    Fabrique du CodeurAgent.

    Args:
        model_client: client Azure OpenAI. Si None, utilise la config centralisée.

    Returns:
        AssistantAgent nommé "CodeurAgent", avec un contexte de conversation
        borné en tokens (voir docstring du module).
    """
    # Limite de sortie (ticket "Optimisation des coûts et des prompts") :
    # le CodeurAgent peut produire un script Python complet et long,
    # 16000 tokens de sortie reste cohérent avec CONTEXT_RESERVED_FOR_COMPLETION.
    client = model_client or get_model_client(max_completion_tokens=16000)

    model_context = TokenLimitedChatCompletionContext(
        model_client=client,
        token_limit=get_context_token_limit(client),
    )

    return AssistantAgent(
        name="CodeurAgent",
        model_client=client,
        model_context=model_context,
        system_message=CODEUR_SYSTEM_MESSAGE,
        description=(
            "Agent chargé de générer et corriger du code Python à partir "
            "d'un plan d'action ou d'un feedback de revue de code."
        ),
    )