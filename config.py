# config.py
import os
from typing import Dict

def create_dir_if_not_exists(directory: str) -> None:
    """Create a directory if it does not already exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

# API Credentials
# It's a good practice to load these from environment variables.
API_KEY: str = os.environ.get("API_KEY", "ab80e273924516b066061ebda000e0c4")
SECRET_KEY: str = os.environ.get("SECRET_KEY", "7ee54cc01dcfaef1086c36c7439e47e4")

# HTTP Headers for API calls
PUBLIC_HEADERS: Dict[str, str] = {"Content-Type": "application/json"}

# Trading and Indicator Parameters
MIN_ATR_RATIO: float = 0.01         # 1% of current price
TRADING_WINDOW: int = 15            # live trading timeframe in minutes
BASE_TRAINING_TIMEFRAME: int = 1    # training timeframe in minutes

# Directories for caching and models
HIST_CACHE_DIR: str = "cache"
MODEL_DIR = r"F:\Project Archerus\v3\saved_models"


# Ensure required directories exist
create_dir_if_not_exists(HIST_CACHE_DIR)
create_dir_if_not_exists(MODEL_DIR)