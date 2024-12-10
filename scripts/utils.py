import logging
import os
import psutil


def setup_logging(log_file="tokenizer_training.log"):
    """Configure logging settings."""
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s %(levelname)s:%(message)s",
        level=logging.INFO,
    )
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def log_memory_usage():
    """Log the current memory usage of the process."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # in MB
    logging.info(f"Current memory usage: {mem:.2f} MB")
