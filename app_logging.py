import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_FILE = Path(os.getenv("LOG_FILE", str(LOG_DIR / "app.log")))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

_HANDLER_MARKER = "_diarization_system_handler"
_CONFIGURED = False


def _log_level():
    return getattr(logging, LOG_LEVEL, logging.INFO)


def setup_logging():
    global _CONFIGURED

    root_logger = logging.getLogger()
    root_logger.setLevel(_log_level())

    if not any(getattr(handler, _HANDLER_MARKER, False) for handler in root_logger.handlers):
        LOG_FILE.parent.mkdir(exist_ok=True, parents=True)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(_log_level())
        setattr(file_handler, _HANDLER_MARKER, True)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(_log_level())
        setattr(console_handler, _HANDLER_MARKER, True)

        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    logging.captureWarnings(True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _CONFIGURED = True
    return logging.getLogger("diarization_system")


def get_logger(name):
    setup_logging()
    return logging.getLogger(name)
