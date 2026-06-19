import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(log_dir: Path = Path("logs")):
    log_dir.mkdir(exist_ok=True)
    
    # Настройка логгера
    logger = logging.getLogger("license_plate_api")
    logger.setLevel(logging.INFO)
    
    # Файловый логгер с ротацией
    file_handler = RotatingFileHandler(
        log_dir / "api.log",
        maxBytes=10_000_000,
        backupCount=5
    )
    file_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    
    # Консольный логгер
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger