import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
)


async def main():
    model_client = AzureOpenAIChatCompletionClient(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        model="gpt-5",
    )

    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="Tu es un assistant IA utile et tu réponds toujours en français.",
    )

    response = await assistant.run(task="Présente-toi en deux phrases.")

    print(response.messages[-1].content)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())