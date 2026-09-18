"""
Script de VALIDATION MANUELLE du ticket "Context Window Exceeded".

Ce script ne modifie AUCUN fichier de production. Il construit les agents
existants tels quels (agents/codeur_agent.py, agents/reviewer_agent.py) et
inspecte leur configuration, puis teste la troncature en isolation avec
un budget de tokens volontairement réduit -- SANS jamais appeler .run()
ni envoyer de requête de complétion à Azure OpenAI.

Emplacement recommandé : tests/test_context_window.py
Lancement (depuis la racine du projet) :
    python tests/test_context_window.py
"""
import asyncio

from autogen_core.model_context import TokenLimitedChatCompletionContext
from autogen_core.models import UserMessage

from agents.codeur_agent import create_codeur_agent
from agents.reviewer_agent import create_reviewer_agent
from config.settings import get_context_token_limit, get_model_client


def test_1_2_3_configuration_des_agents() -> None:
    """Objectifs 1, 2, 3 : vérifie le TYPE de contexte et le budget réel."""
    print("=" * 70)
    print("TEST 1/2/3 — Configuration réelle des agents")
    print("=" * 70)

    client = get_model_client()

    codeur = create_codeur_agent(model_client=client)
    reviewer = create_reviewer_agent(model_client=client)

    codeur_context_type = type(codeur._model_context).__name__
    reviewer_context_type = type(reviewer._model_context).__name__

    print(f"CodeurAgent.model_context   : {codeur_context_type}")
    print(f"ReviewerAgent.model_context : {reviewer_context_type}")

    assert codeur_context_type == "TokenLimitedChatCompletionContext", (
        f"ATTENDU TokenLimitedChatCompletionContext, obtenu {codeur_context_type}"
    )
    assert reviewer_context_type == "TokenLimitedChatCompletionContext", (
        f"ATTENDU TokenLimitedChatCompletionContext, obtenu {reviewer_context_type}"
    )

    codeur_limit = codeur._model_context._token_limit
    reviewer_limit = reviewer._model_context._token_limit
    print(f"Budget de tokens CodeurAgent   : {codeur_limit}")
    print(f"Budget de tokens ReviewerAgent : {reviewer_limit}")

    # Recalcul indépendant, pour confirmer que le budget vient bien de
    # get_context_token_limit() et n'est pas une valeur par défaut cachée.
    budget_attendu = get_context_token_limit(client)
    print(f"Budget recalculé via get_context_token_limit() : {budget_attendu}")

    assert codeur_limit == budget_attendu, "Le budget du CodeurAgent ne correspond pas au calcul attendu"
    assert reviewer_limit == budget_attendu, "Le budget du ReviewerAgent ne correspond pas au calcul attendu"

    print("\n✅ TEST 1/2/3 RÉUSSI\n")


async def test_4_5_6_troncature_isolee() -> None:
    """
    Objectifs 4, 5, 6 : simule un historique dépassant la limite, avec un
    budget ARTIFICIELLEMENT réduit (uniquement dans ce test, ne touche pas
    à la configuration réelle) -- aucun appel réseau de complétion.
    """
    print("=" * 70)
    print("TEST 4/5/6 — Troncature automatique de l'historique")
    print("=" * 70)

    client = get_model_client()

    # Budget volontairement bas pour forcer la troncature rapidement,
    # SANS toucher à CONTEXT_RESERVED_FOR_COMPLETION ni à la config réelle.
    budget_test = 500
    context = TokenLimitedChatCompletionContext(model_client=client, token_limit=budget_test)

    nb_messages_ajoutes = 25
    for i in range(nb_messages_ajoutes):
        contenu = f"Message {i} : " + ("contenu de test répété " * 25)
        await context.add_message(UserMessage(content=contenu, source="test"))

    messages_finaux = await context.get_messages()
    tokens_finaux = client.count_tokens(messages_finaux)

    print(f"Messages ajoutés au total       : {nb_messages_ajoutes}")
    print(f"Messages conservés après troncature : {len(messages_finaux)}")
    print(f"Tokens du contexte final        : {tokens_finaux}")
    print(f"Budget autorisé (test)          : {budget_test}")

    # Objectif 5 : les anciens messages ont bien été supprimés.
    assert len(messages_finaux) < nb_messages_ajoutes, (
        "Aucune troncature détectée : tous les messages sont encore présents"
    )

    # Objectif 6 : le contexte final ne dépasse jamais le budget alloué.
    assert tokens_finaux <= budget_test, (
        f"DÉPASSEMENT : {tokens_finaux} tokens > budget autorisé {budget_test}"
    )

    print("\n✅ TEST 4/5/6 RÉUSSI\n")


def main() -> None:
    test_1_2_3_configuration_des_agents()
    asyncio.run(test_4_5_6_troncature_isolee())
    print("=" * 70)
    print("TOUS LES TESTS ONT RÉUSSI — ticket 'Context Window Exceeded' validé.")
    print("=" * 70)


if __name__ == "__main__":
    main()