from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

CLEANED_DATA_DIR = PROJECT_ROOT / "cleaned_data"

VECTORSTORES_DIR = PROJECT_ROOT / "vectorstores"
