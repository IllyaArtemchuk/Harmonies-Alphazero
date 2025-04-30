import logging
import os


def setup_logger(name, log_file, level=logging.INFO):

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def game_debug_enabled():
    return os.environ.get("GAME_DEBUG") == "1"
