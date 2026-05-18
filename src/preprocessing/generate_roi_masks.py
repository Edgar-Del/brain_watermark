from pathlib import Path

from src.config import (
    MULTICLASS_TRAIN_DIR,
    MULTICLASS_TEST_DIR,
    ROI_MASKS_TRAIN_DIR,
    ROI_MASKS_TEST_DIR,
    IMAGE_SIZE,
)
from src.utils.io_utils import list_image_files, ensure_dir
from src.utils.image_utils import load_grayscale_image, save_image
from src.utils.mask_utils import generate_roi_mask


def generate_masks_for_split(source_root: Path, target_root: Path) -> None:
    for class_dir in source_root.iterdir():
        if not class_dir.is_dir():
            continue

        target_class_dir = target_root / class_dir.name
        ensure_dir(target_class_dir)

        files = list_image_files(class_dir)

        for file_path in files:
            image = load_grayscale_image(file_path, target_size=IMAGE_SIZE)

            mask = generate_roi_mask(image)

            output_path = target_class_dir / file_path.name
            save_image(output_path, mask * 255)

        print(f"{class_dir.name}: {len(files)} máscaras geradas")


def main() -> None:
    print("=== GERAÇÃO DE MÁSCARAS ROI ===\n")

    print("Treino...")
    generate_masks_for_split(MULTICLASS_TRAIN_DIR, ROI_MASKS_TRAIN_DIR)

    print("\nTeste...")
    generate_masks_for_split(MULTICLASS_TEST_DIR, ROI_MASKS_TEST_DIR)

    print("\nMáscaras ROI geradas com sucesso.")


if __name__ == "__main__":
    main()