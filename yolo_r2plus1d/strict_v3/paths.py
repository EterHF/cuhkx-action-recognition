"""Repository paths shared by strict-v3 entry points."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "checkpoints/strict_v3"
RESULT_DIR = REPO_ROOT / "results/strict_v3"
DATA_DIR = REPO_ROOT / "data"
