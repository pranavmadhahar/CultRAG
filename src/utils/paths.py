from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ASSETS_DIR = PROJECT_ROOT / "assets"

DATA_DIR = ASSETS_DIR / "data"
CLEANED_DATA_DIR = ASSETS_DIR / "cleaned_data"
VECTORSTORES_DIR = ASSETS_DIR / "vectorstores"