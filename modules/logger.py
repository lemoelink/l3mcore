import logging
import os

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

class ColorFormatter(logging.Formatter):
    COLOR_MAP = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[1;31m' # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelname, '')
        orig_levelname = record.levelname
        if color:
            record.levelname = f"{color}{orig_levelname}{self.RESET}"
        res = super().format(record)
        record.levelname = orig_levelname
        return res

def setup_logger(name, log_file, level=logging.INFO):
    """Configures a logger with file and console outputs."""
    handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console = logging.StreamHandler()
    console.setFormatter(ColorFormatter('%(levelname)s: %(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

app_logger = setup_logger('app', 'logs/app.log')
