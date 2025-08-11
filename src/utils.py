import os
import time
import logging


LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


def setup_logger(log_filename: str, log_level=logging.INFO):
    logger = logging.getLogger(__name__)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s[%(lineno)d] ----- %(message)s')
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, log_filename), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def time_monitor(log_file="log.txt"):
    def out_wrapper(func):
        def wrapper(*args, **kwargs):
            logger = setup_logger(log_file)
            logger.info("function [%s] start processing" % func.__name__)
            start_time = time.time()
            res = func(*args, **kwargs)
            used_time = time.time() - start_time
            logger.info("function [%s] cost time %.2f seconds" % (func.__name__, used_time))
            logger.info(f"input args {args} kwargs {kwargs} output {res}\n")
            logger.info(f"output {res}\n\n")
            return res
        return wrapper
    return out_wrapper