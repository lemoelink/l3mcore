import logging
import os

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

def setup_logger(name, log_file, level=logging.INFO):
    """Configures a logger with file and console outputs."""
    handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

app_logger = setup_logger('app', 'logs/app.log')
