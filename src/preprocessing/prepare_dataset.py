from pathlib import Path
import shutil

from src.config import (
    RAW_TRAIN_DIR,
    RAW_TEST_DIR,
    RAW_CLASSES,
    MULTICLASS_TRAIN_DIR,
    MULTICLASS_TEST_DIR,
    ensure_directories,
)


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_valid_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def copy_class_images(source_dir: Path, target_dir: Path) -> int:
    """
    Copia imagens válidas de uma pasta de classe para outra.
    Retorna o número de imagens copiadas.
    """
    count = 0
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_path in sorted(source_dir.iterdir()):
        if not is_valid_image_file(file_path):
            continue

        destination = target_dir / file_path.name
        shutil.copy2(file_path, destination)
        count += 1

    return count


def build_multiclass_structure() -> None:
    """
    Copia os dados brutos para a estrutura multiclasses processada.
    """
    print("A preparar estrutura multiclasses...")

    total_train = 0
    total_test = 0

    for class_name in RAW_CLASSES:
        src_train_class = RAW_TRAIN_DIR / class_name
        dst_train_class = MULTICLASS_TRAIN_DIR / class_name

        src_test_class = RAW_TEST_DIR / class_name
        dst_test_class = MULTICLASS_TEST_DIR / class_name

        if not src_train_class.exists():
            print(f"[AVISO] Pasta de treino não encontrada: {src_train_class}")
        else:
            copied = copy_class_images(src_train_class, dst_train_class)
            total_train += copied
            print(f"Treino - {class_name}: {copied} imagens copiadas")

        if not src_test_class.exists():
            print(f"[AVISO] Pasta de teste não encontrada: {src_test_class}")
        else:
            copied = copy_class_images(src_test_class, dst_test_class)
            total_test += copied
            print(f"Teste - {class_name}: {copied} imagens copiadas")

    print(f"\nTotal multiclass train: {total_train}")
    print(f"Total multiclass test : {total_test}")


def main() -> None:
    ensure_directories()
    build_multiclass_structure()
    print("\nPreparação multiclasses concluída com sucesso.")


if __name__ == "__main__":
    main()