# config/logging_config.py
"""Configuration centralisée du logging pour tout le projet Smartovate."""
import logging
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/smartovate.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger nommé, à utiliser dans chaque module/agent."""
    return logging.getLogger(name)