"""
config.py
Centralized configuration management for ReviewShield.
Loads database credentials from environment variables and sets up project paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Project Paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RAW_DATA_PATH = DATA_DIR / "raw_reviews.csv"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_reviews.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Database Configuration
DB_BACKEND = os.getenv("DB_BACKEND", "sqlite").strip().lower()
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "reviewshield.sqlite3")))
if not DB_PATH.is_absolute():
    DB_PATH = BASE_DIR / DB_PATH

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "reviewshield")

if DB_BACKEND == "sqlite":
    CONN_STRING = f"sqlite:///{DB_PATH.as_posix()}"
    CONNECT_ARGS = {"timeout": 5}
elif DB_BACKEND == "postgresql":
    CONN_STRING = URL.create(
        "postgresql+psycopg2",
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=int(DB_PORT),
        database=DB_NAME,
    )
    CONNECT_ARGS = {"connect_timeout": 5}
else:
    raise ValueError("DB_BACKEND must be either 'sqlite' or 'postgresql'.")


def get_db_engine():
    """Returns a SQLAlchemy engine instance."""
    return create_engine(CONN_STRING, connect_args=CONNECT_ARGS)
