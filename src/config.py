"""
config.py
Centralized configuration management for ReviewShield.
Loads database credentials from environment variables and sets up project paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Database Configuration
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "reviewshield")

CONN_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Project Paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RAW_DATA_PATH = DATA_DIR / "raw_reviews.csv"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_reviews.csv"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_db_engine():
    """Returns a SQLAlchemy engine instance."""
    return create_engine(CONN_STRING)
