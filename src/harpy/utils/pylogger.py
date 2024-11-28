import logging
import os


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""
    logger = logging.getLogger(name)
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logger.setLevel(LOGLEVEL)

    logger.propagate = False

    # Create a stream handler to output to console
    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)

    # Add the stream handler to the logger
    if not logger.hasHandlers():
        logger.addHandler(ch)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, getattr(logger, level))
    return logger
