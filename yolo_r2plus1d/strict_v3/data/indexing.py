"""Deterministic dataset discovery for the CUHK-X directory layout."""

from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_test_ids(csv_path: Path) -> list[str]:
    frame = pd.read_csv(csv_path)
    column = "id" if "id" in frame.columns else "path" if "path" in frame.columns else frame.columns[0]
    return [Path(value.rstrip("/")).name for value in frame[column].astype(str)]


def discover_train(root: Path) -> list[tuple[str, int, int, dict[str, Path]]]:
    """Return clips from ``<modality>/<class>/<user>/<trial>`` in stable order."""
    samples: list[tuple[str, int, int, dict[str, Path]]] = []
    modality_names = tuple(
        path.name for path in sorted(root.iterdir()) if path.is_dir()
    ) if root.is_dir() else ()
    relative_trials: set[Path] = set()
    for modality in modality_names:
        modality_root = root / modality
        for class_dir in (path for path in modality_root.iterdir() if path.is_dir()):
            for user_dir in (path for path in class_dir.iterdir() if path.is_dir()):
                relative_trials.update(
                    trial.relative_to(modality_root)
                    for trial in user_dir.iterdir()
                    if trial.is_dir()
                )
    for relative in sorted(relative_trials, key=str):
        try:
            label = int(relative.parts[0].split("_", 1)[0])
            user = int(relative.parts[1].removeprefix("user"))
        except (IndexError, ValueError):
            continue
        modalities = {name: root / name / relative for name in modality_names}
        samples.append((str(relative), label, user, modalities))
    return samples
