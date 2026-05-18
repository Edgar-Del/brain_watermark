from pathlib import Path
import json
from typing import List


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def list_image_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []

    return sorted([
        file_path for file_path in directory.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS
    ])


def count_image_files(directory: Path) -> int:
    return len(list_image_files(directory))


def save_json(data: dict, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_text(text: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)