from pathlib import Path
import csv

from src.config import (
    MULTICLASS_TRAIN_DIR,
    MULTICLASS_TEST_DIR,
    WATERMARK_DIR,
    WATERMARKED_DIR,
    METRICS_DIR,
    IMAGE_SIZE,
    WATERMARK_SIZE,
    DEFAULT_WAVELET,
    DEFAULT_SUBBAND,
    V1_ALPHA,
    ensure_directories,
)
from src.utils.io_utils import list_image_files, ensure_dir
from src.utils.image_utils import (
    load_grayscale_image,
    save_image,
    binarize_image,
)
from src.evaluation.metrics_image import mse, psnr, compute_ssim
from src.watermarking.watermark_dwt_svd import DWTSVDWatermarker


def process_split(split_name: str, source_root: Path, output_root: Path, csv_writer) -> None:
    watermark_path = WATERMARK_DIR / "watermark.png"
    watermark = load_grayscale_image(watermark_path, target_size=WATERMARK_SIZE)
    watermark_bin = binarize_image(watermark).astype(float)

    watermarker = DWTSVDWatermarker(
        wavelet=DEFAULT_WAVELET,
        alpha=V1_ALPHA,
        subband=DEFAULT_SUBBAND
    )

    for class_dir in sorted(source_root.iterdir()):
        if not class_dir.is_dir():
            continue

        output_class_dir = output_root / split_name / class_dir.name
        ensure_dir(output_class_dir)

        for image_path in list_image_files(class_dir):
            image = load_grayscale_image(image_path, target_size=IMAGE_SIZE)

            result = watermarker.embed(image, watermark_bin)
            watermarked = result.watermarked_image

            output_path = output_class_dir / image_path.name
            save_image(output_path, watermarked)

            row = {
                "version": "v1_uniform",
                "split": split_name,
                "class_name": class_dir.name,
                "image_name": image_path.name,
                "mse": mse(image, watermarked),
                "psnr": psnr(image, watermarked),
                "ssim": compute_ssim(image, watermarked),
            }
            csv_writer.writerow(row)


def main() -> None:
    ensure_directories()

    output_root = WATERMARKED_DIR / "v1_uniform"
    ensure_dir(output_root)
    ensure_dir(METRICS_DIR)

    csv_path = METRICS_DIR / "image_quality_results_v1.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["version", "split", "class_name", "image_name", "mse", "psnr", "ssim"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print("Aplicando watermark V1 em treino...")
        process_split("train", MULTICLASS_TRAIN_DIR, output_root, writer)

        print("Aplicando watermark V1 em teste...")
        process_split("test", MULTICLASS_TEST_DIR, output_root, writer)

    print(f"\nConcluído. Métricas guardadas em: {csv_path}")


if __name__ == "__main__":
    main()