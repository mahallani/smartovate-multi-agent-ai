# teams/docker_executor.py

"""
Exécution sécurisée du code généré par le CodeurAgent, dans un conteneur
Docker isolé.

Pourquoi ce module :
Le ticket historique demandait `UserProxyAgent(use_docker=True)`. Cette
classe appartient à `pyautogen` (AutoGen < 0.4), que ce projet n'utilise
pas et ne doit pas utiliser.

Depuis la réécriture d'AutoGen (autogen-core / autogen-agentchat /
autogen-ext), l'exécution de code n'est plus une option booléenne d'un
agent -- c'est un COMPOSANT indépendant et interchangeable : un
`CodeExecutor` (protocole défini dans `autogen_core.code_executor`).

- `LocalCommandLineCodeExecutor`
  -> exécute le code directement sur la machine hôte. INTERDIT
  dans ce projet pour du code généré par un agent.

- `DockerCommandLineCodeExecutor`
  -> exécute le code DANS un conteneur Docker isolé et éphémère.
  C'est l'équivalent moderne et recommandé de
  `UserProxyAgent(use_docker=True)`.

Le protocole `CodeExecutor` étant commun aux deux implémentations, ce
module ne contient volontairement AUCUNE référence à
`LocalCommandLineCodeExecutor`.

En environnement conteneurisé, `DOCKER_HOST_WORKSPACE_PATH` permet de
fournir au Docker daemon le chemin du workspace sur l'hôte Docker Desktop.
Cela est nécessaire lorsque le processus Python tourne lui-même dans
un conteneur et communique avec Docker via /var/run/docker.sock.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from docker.errors import DockerException


class DockerNotAvailableError(RuntimeError):
    """
    Levée quand le daemon Docker n'est pas accessible depuis la machine
    hôte ou depuis le conteneur applicatif.
    """


@dataclass
class ExecutionResult:
    """Résultat structuré d'une exécution de code dans Docker."""

    success: bool
    exit_code: int | None
    output: str


def create_docker_code_executor(
    work_dir: str | Path = "./_docker_workspace",
    image: str = "python:3.11-slim",
    timeout: int = 60,
) -> DockerCommandLineCodeExecutor:
    """
    Fabrique de l'exécuteur Docker.

    `work_dir` correspond au répertoire utilisé par le processus Python
    pour écrire les fichiers temporaires.

    Lorsque l'application tourne dans un conteneur, le Docker daemon
    extérieur doit cependant recevoir le chemin du workspace sur
    l'hôte. Ce chemin est fourni par la variable :

        DOCKER_HOST_WORKSPACE_PATH

    Exemple dans Docker Desktop :

        /run/desktop/mnt/host/c/Users/ASUS/Downloads/smartovate/_docker_workspace

    Hors conteneur, si cette variable n'est pas définie, bind_dir reste
    à None et le comportement existant est conservé.
    """

    bind_dir_env = os.getenv("DOCKER_HOST_WORKSPACE_PATH")

    bind_dir = Path(bind_dir_env) if bind_dir_env else None

    return DockerCommandLineCodeExecutor(
        image=image,
        work_dir=work_dir,
        bind_dir=bind_dir,
        timeout=timeout,
        auto_remove=True,
        stop_container=True,
    )


async def execute_code_in_docker(
    code: str,
    work_dir: str | Path = "./_docker_workspace",
    image: str = "python:3.11-slim",
    timeout: int = 60,
) -> ExecutionResult:
    """
    Exécute `code` dans un conteneur Docker isolé et éphémère.

    Returns:
        ExecutionResult(success, exit_code, output).

    Raises:
        DockerNotAvailableError:
            Si le daemon Docker n'est pas accessible.
    """

    executor = create_docker_code_executor(
        work_dir=work_dir,
        image=image,
        timeout=timeout,
    )

    try:
        await executor.start()

    except (DockerException, RuntimeError) as exc:
        raise DockerNotAvailableError(
            "Impossible de se connecter au daemon Docker.\n\n"
            "Vérifiez, dans l'ordre :\n"
            "  1. Docker Desktop est installé et démarré.\n"
            "  2. Le backend WSL2 est activé.\n"
            "  3. `docker info` fonctionne correctement.\n\n"
            f"Erreur d'origine : {exc}"
        ) from exc

    try:
        code_block = CodeBlock(
            code=code,
            language="python",
        )

        result = await executor.execute_code_blocks(
            [code_block],
            cancellation_token=CancellationToken(),
        )

        return ExecutionResult(
            success=(result.exit_code == 0),
            exit_code=result.exit_code,
            output=result.output,
        )

    finally:
        await executor.stop()