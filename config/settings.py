import os
from dotenv import load_dotenv
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


def get_model_client(
    max_completion_tokens: int | None = None,
) -> AzureOpenAIChatCompletionClient:
    """
    Construit et retourne un client Azure OpenAI configuré via le .env.

    Args:
        max_completion_tokens: limite stricte de tokens de SORTIE pour ce
            client (ticket "Optimisation des coûts et des prompts",
            critère n°2). Transmis tel quel au constructeur, qui l'inclut
            dans `create_args` et donc dans TOUS les appels `.create()`
            faits avec ce client.

            IMPORTANT : pour ce déploiement Azure OpenAI GPT-5, le
            paramètre accepté est `max_completion_tokens`, PAS `max_tokens`
            (`max_tokens` est explicitement rejeté par l'API -- vérifié en
            conditions réelles). Ne jamais utiliser `max_tokens` ici.

            Si None (valeur par défaut), aucune limite de sortie n'est
            configurée -- comportement identique à avant ce changement.
    """
    extra_args: dict = {}
    if max_completion_tokens is not None:
        extra_args["max_completion_tokens"] = max_completion_tokens

    return AzureOpenAIChatCompletionClient(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        model=os.getenv("AZURE_OPENAI_MODEL", "gpt-5"),
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        timeout=120,
        max_retries=1,  # réessaie automatiquement en cas d'échec réseau
        **extra_args,
    )


def get_context_token_limit(
    model_client: AzureOpenAIChatCompletionClient,
    reserved_for_completion: int | None = None,
) -> int:
    """
    Calcule le budget de tokens à allouer à l'HISTORIQUE de conversation
    transmis à chaque appel Azure OpenAI (paramètre `token_limit` de
    TokenLimitedChatCompletionContext), en réservant explicitement une
    marge pour la réponse du modèle.

    Pourquoi une marge explicite (`reserved_for_completion`) :
    Le CodeurAgent peut produire des scripts Python longs. Si tout le
    contexte disponible était alloué à l'historique, il ne resterait
    presque aucune place pour la réponse elle-même, qui ferait alors
    échouer l'appel Azure OpenAI d'une autre façon (troncature de sortie,
    voire refus si prompt + max_tokens dépasse la fenêtre du modèle).

    Implémentation : utilise UNIQUEMENT l'API publique du client
    (`remaining_tokens([])`, qui retourne la fenêtre totale du modèle
    moins le coût de zéro message), sans dépendre du module interne
    `autogen_ext.models.openai._model_info` (privé, susceptible de changer
    sans préavis entre versions d'autogen-ext).

    Args:
        model_client: le client Azure OpenAI (doit implémenter
            `remaining_tokens()`, ce qui est le cas pour
            AzureOpenAIChatCompletionClient -- vérifié dans cette version).
        reserved_for_completion: nombre de tokens réservés pour la réponse
            du modèle. Si None, lit CONTEXT_RESERVED_FOR_COMPLETION dans le
            .env (défaut : 16000 -- large marge pour un script Python
            complet avec docstrings et gestion d'erreurs).

    Returns:
        Le budget de tokens (entier positif, jamais inférieur à 1000) à
        passer en `token_limit` à TokenLimitedChatCompletionContext.
    """
    if reserved_for_completion is None:
        reserved_for_completion = int(
            os.getenv("CONTEXT_RESERVED_FOR_COMPLETION", "16000")
        )

    total_available = model_client.remaining_tokens([])
    return max(total_available - reserved_for_completion, 1000)