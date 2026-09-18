import asyncio

from autogen_core.models import UserMessage

from config.settings import get_model_client


async def main() -> None:
    client = get_model_client(max_completion_tokens=5)

    try:
        result = await client.create(
            messages=[
                UserMessage(
                    content="Dis juste 'ok'.",
                    source="user",
                )
            ]
        )

        print("SUCCÈS — max_tokens accepté par Azure")
        print("content :", result.content)
        print("usage   :", result.usage)

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())