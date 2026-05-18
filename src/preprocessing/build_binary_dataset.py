from pathlib import Path
import shutil

from src.config import (
    MULTICLASS_TRAIN_DIR,
    MULTICLASS_TEST_DIR,
    BINARY_TRAIN_DIR,
    BINARY_TEST_DIR,
    TUMOR_CLASSES,
    ensure_directories,
)


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_valid_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def copy_images_to_binary_class(source_dir: Path, target_dir: Path, prefix: str) -> int:
    """
    Copia imagens de uma classe origem para uma classe binária destino.
    Acrescenta prefixo ao nome para evitar colisões.
    """
    count = 0
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_path in sorted(source_dir.iterdir()):
        if not is_valid_image_file(file_path):
            continue

        new_name = f"{prefix}_{file_path.name}"
        destination = target_dir / new_name
        shutil.copy2(file_path, destination)
        count += 1

    return count


def build_binary_split(split_name: str, source_root: Path, target_root: Path) -> None:
    """
    Constrói o split binário ('train' ou 'test').
    """
    print(f"\nA construir split binário: {split_name}")

    tumor_target = target_root / "tumor"
    notumor_target = target_root / "notumor"

    tumor_count = 0
    notumor_count = 0

    for class_name in TUMOR_CLASSES:
        source_class_dir = source_root / class_name
        if not source_class_dir.exists():
            print(f"[AVISO] Classe não encontrada: {source_class_dir}")
            continue

        copied = copy_images_to_binary_class(
            source_dir=source_class_dir,
            target_dir=tumor_target,
            prefix=class_name
        )
        tumor_count += copied
        print(f"{class_name} -> tumor: {copied} imagens")

    source_notumor_dir = source_root / "notumor"
    if not source_notumor_dir.exists():
        print(f"[AVISO] Classe não encontrada: {source_notumor_dir}")
    else:
        notumor_count = copy_images_to_binary_class(
            source_dir=source_notumor_dir,
            target_dir=notumor_target,
            prefix="notumor"
        )
        print(f"notumor -> notumor: {notumor_count} imagens")

    print(f"Total tumor ({split_name})   : {tumor_count}")
    print(f"Total notumor ({split_name}) : {notumor_count}")


def main() -> None:
    ensure_directories()

    build_binary_split(
        split_name="train",
        source_root=MULTICLASS_TRAIN_DIR,
        target_root=BINARY_TRAIN_DIR
    )

    build_binary_split(
        split_name="test",
        source_root=MULTICLASS_TEST_DIR,
        target_root=BINARY_TEST_DIR
    )

    print("\nConversão para dataset binário concluída com sucesso.")


if __name__ == "__main__":
    main()