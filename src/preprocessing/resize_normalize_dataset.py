from pathlib import Path
import cv2
import numpy as np

from src.config import (
    MULTICLASS_TRAIN_DIR,
    MULTICLASS_TEST_DIR,
    IMAGE_SIZE,
)
from src.utils.io_utils import list_image_files, ensure_dir
from src.utils.image_utils import load_grayscale_image, save_image


def process_and_save_image(input_path: Path, output_path: Path) -> None:
    image = load_grayscale_image(input_path, target_size=IMAGE_SIZE)

    # Normalização opcional (mantemos simples nesta fase)
    image = image.astype(np.uint8)

    save_image(output_path, image)


def process_split(source_root: Path, target_root: Path) -> None:
    for class_dir in source_root.iterdir():
        if not class_dir.is_dir():
            continue

        target_class_dir = target_root / class_dir.name
        ensure_dir(target_class_dir)

        files = list_image_files(class_dir)

        for file_path in files:
            output_path = target_class_dir / file_path.name
            process_and_save_image(file_path, output_path)

        print(f"{class_dir.name}: {len(files)} imagens processadas")


def main() -> None:
    print("=== REDIMENSIONAMENTO E NORMALIZAÇÃO ===\n")

    processed_root = Path("data/processed/resized")

    train_target = processed_root / "train"
    test_target = processed_root / "test"

    ensure_dir(train_target)
    ensure_dir(test_target)

    print("Processando treino...")
    process_split(MULTICLASS_TRAIN_DIR, train_target)

    print("\nProcessando teste...")
    process_split(MULTICLASS_TEST_DIR, test_target)

    print("\nRedimensionamento concluído.")


if __name__ == "__main__":
    main()