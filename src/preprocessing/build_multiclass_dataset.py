from pathlib import Path

from src.config import (
    MULTICLASS_TRAIN_DIR,
    MULTICLASS_TEST_DIR,
    RAW_CLASSES,
)


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def count_images_in_dir(directory: Path) -> int:
    if not directory.exists():
        return 0

    return sum(
        1 for file_path in directory.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS
    )


def summarize_multiclass_dataset() -> None:
    print("=== RESUMO DO DATASET MULTICLASSES ===\n")

    total_train = 0
    total_test = 0

    print("Treino:")
    for class_name in RAW_CLASSES:
        class_dir = MULTICLASS_TRAIN_DIR / class_name
        count = count_images_in_dir(class_dir)
        total_train += count
        print(f"  - {class_name}: {count}")

    print(f"Total treino: {total_train}\n")

    print("Teste:")
    for class_name in RAW_CLASSES:
        class_dir = MULTICLASS_TEST_DIR / class_name
        count = count_images_in_dir(class_dir)
        total_test += count
        print(f"  - {class_name}: {count}")

    print(f"Total teste: {total_test}\n")
    print("Resumo multiclasses concluído.")


def main() -> None:
    summarize_multiclass_dataset()


if __name__ == "__main__":
    main()