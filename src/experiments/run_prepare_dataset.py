from src.config import ensure_directories
from src.preprocessing.prepare_dataset import build_multiclass_structure
from src.preprocessing.build_binary_dataset import main as build_binary_main
from src.preprocessing.build_multiclass_dataset import summarize_multiclass_dataset


def main() -> None:
    print("=== INÍCIO DA PREPARAÇÃO DO DATASET ===\n")

    ensure_directories()

    print("1. Construção da estrutura multiclasses")
    build_multiclass_structure()

    print("\n2. Conversão para dataset binário")
    build_binary_main()

    print("\n3. Resumo final do dataset multiclasses")
    summarize_multiclass_dataset()

    print("\n=== PREPARAÇÃO DO DATASET CONCLUÍDA ===")


if __name__ == "__main__":
    main()