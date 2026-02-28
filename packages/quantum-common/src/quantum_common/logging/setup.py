"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging
import sys

import structlog

from quantum_common.config.schema import LoggingConfig


def setup_logging(
    level: str = "INFO",
    log_format: str = "console",
    log_file: str | None = None,
    config: LoggingConfig | None = None,
) -> None:
    """Configure structured logging for the entire project.

    Can be called with individual parameters or a LoggingConfig object.
    Uses structlog for structured event logging. JSON output for
    production/CI, colored console output for development.
    """
    if config is None:
        config = LoggingConfig(level=level, format=log_format, log_file=log_file)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if config.include_timestamps:
        shared_processors.insert(2, structlog.processors.TimeStamper(fmt="iso"))

    if config.format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, config.level))

    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
