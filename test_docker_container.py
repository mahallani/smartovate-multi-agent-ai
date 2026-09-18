import asyncio

from teams.docker_executor import execute_code_in_docker


async def main():
    result = await execute_code_in_docker(
        'print("OK depuis le conteneur")'
    )

    print("success=", result.success)
    print("exit_code=", result.exit_code)
    print("output=", result.output)


asyncio.run(main())