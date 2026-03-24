from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DEFAULT_MODEL_PATH = MODELS_DIR / "model_artifacts.pkl"
DEFAULT_ENCOUNTERS_PATH = DATA_DIR / "merged_encounters.csv"
DEFAULT_PACKAGES_PATH = DATA_DIR / "navavej_packages.csv"


def resolve_path(env_name: str, default_path: Path) -> Path:
    raw_value = os.getenv(env_name)
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return default_path.resolve()
