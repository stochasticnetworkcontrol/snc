import logging
from pythonjsonlogger import jsonlogger


def set_up_json_logging():
    logger = logging.getLogger()
    log_handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter()
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
