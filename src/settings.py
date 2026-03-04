from pathlib import Path

# Replace this if your local data directory is elsewhere.
DEFAULT_DATA_DIR = Path("data")
DEFAULT_TRAIN_FILE = "round1_ijcai_18_train_20180301.txt"
DEFAULT_TEST_A_FILE = "round1_ijcai_18_test_a_20180301.txt"
DEFAULT_TEST_B_FILE = "round1_ijcai_18_test_b_20180418.txt"
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache"


def resolve_path(path_like: str | Path, project_root: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()
