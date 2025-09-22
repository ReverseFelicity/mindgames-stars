import os
import re
import sys
import tempfile
import time
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError


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

def my_logger(log_file="log.txt"):
    def out_wrapper(func):
        def wrapper(*args, **kwargs):
            logger = setup_logger(log_file)
            logger.info("function [%s] start processing" % func.__name__)
            start_time = time.time()
            res = func(*args, **kwargs)
            used_time = time.time() - start_time
            logger.info("function [%s] cost time %.2f seconds" % (func.__name__, used_time))
            logger.info(f"input kwargs {kwargs}")
            logger.info(f"output {res}\n\n")
            return res
        return wrapper
    return out_wrapper


def time_monitor():
    def out_wrapper(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            used_time = time.time() - start_time
            print("function [%s] cost time %.2f seconds" % (func.__name__, used_time))
            return res
        return wrapper
    return out_wrapper


def timeout(seconds=60):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timeout=seconds)
                    return result
                except TimeoutError:
                    raise
        return wrapper
    return decorator


PY_BLOCK_RE = re.compile(f'```python(.*?)```', re.DOTALL)


def extract_python_blocks(text: str):
    return [m.group(1).strip() for m in PY_BLOCK_RE.finditer(text)]

def run_python_blocks(blocks, timeout_s=20):
    full_source = '\n\n'.join(blocks)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_source)
        f.flush()
        try:
            completed = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as e:
            return -1, "", f"Execution Timeout, over {timeout_s} seconds"
    return completed.returncode, completed.stdout, completed.stderr