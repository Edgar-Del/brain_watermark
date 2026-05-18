from pathlib import Path
import csv
import numpy as np

from src.config import (
    MULTICLASS_TRAIN_DIR,
    MULTICLASS_TEST_DIR,
    ROI_MASKS_TRAIN_DIR,
    ROI_MASKS_TEST_DIR,
    WATERMARK_DIR,
    WATERMARKED_DIR,
    METRICS_DIR,
    IMAGE_SIZE,
    WATERMARK_SIZE,
    DEFAULT_WAVELET,
    DEFAULT_SUBBAND,
    V2_ALPHA_BASE,
    V2_BLOCK_SIZE,
    V2_ROI_ALPHA_FACTOR,
    V2_ROI_THRESHOLD,
    ensure_directories,
)
from src.utils.io_utils import list_image_files, ensure_dir
from src.utils.image_utils import (
    load_grayscale_image,
    save_image,
    binarize_image,
)
from src.watermarking.watermark_dwt_svd_adaptive import AdaptiveDWTSVDWatermarker
from src.evaluation.metrics_image import mse, psnr, compute_ssim


def load_mask_or_empty(mask_path: Path) -> np.ndarray:
    if mask_path.exists():
        mask = load_grayscale_image(mask_path, target_size=IMAGE_SIZE)
        return (mask > 127).astype(np.uint8)
    return np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint8)


def process_split(
    split_name: str,
    source_root: Path,
    mask_root: Path,
    output_root: Path,
    csv_writer
) -> None:
    watermark_path = WATERMARK_DIR / "watermark.png"
    watermark = load_grayscale_image(watermark_path, target_size=WATERMARK_SIZE)
    watermark_bin = binarize_image(watermark).astype(float)

    watermarker = AdaptiveDWTSVDWatermarker(
        wavelet=DEFAULT_WAVELET,
        subband=DEFAULT_SUBBAND,
        block_size=V2_BLOCK_SIZE,
        alpha_base=V2_ALPHA_BASE,
        roi_alpha_factor=V2_ROI_ALPHA_FACTOR,
        roi_threshold=V2_ROI_THRESHOLD
    )

    for class_dir in sorted(source_root.iterdir()):
        if not class_dir.is_dir():
            continue

        output_class_dir = output_root / split_name / class_dir.name
        ensure_dir(output_class_dir)

        for image_path in list_image_files(class_dir):
            image = load_grayscale_image(image_path, target_size=IMAGE_SIZE)

            mask_path = mask_root / class_dir.name / image_path.name
            roi_mask = load_mask_or_empty(mask_path)

            result = watermarker.embed(
                host_image=image,
                watermark=watermark_bin,
                roi_mask=roi_mask
            )
            watermarked = result.watermarked_image

            output_path = output_class_dir / image_path.name
            save_image(output_path, watermarked)

            row = {
                "version": "v2_adaptive",
                "split": split_name,
                "class_name": class_dir.name,
                "image_name": image_path.name,
                "mse": mse(image, watermarked),
                "psnr": psnr(image, watermarked),
                "ssim": compute_ssim(image, watermarked),
                "used_blocks": len(result.used_blocks),
            }
            csv_writer.writerow(row)


def main() -> None:
    ensure_directories()

    output_root = WATERMARKED_DIR / "v2_adaptive"
    ensure_dir(output_root)
    ensure_dir(METRICS_DIR)

    csv_path = METRICS_DIR / "image_quality_results_v2.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["version", "split", "class_name", "image_name", "mse", "psnr", "ssim", "used_blocks"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print("Aplicando watermark V2 em treino...")
        process_split("train", MULTICLASS_TRAIN_DIR, ROI_MASKS_TRAIN_DIR, output_root, writer)

        print("Aplicando watermark V2 em teste...")
        process_split("test", MULTICLASS_TEST_DIR, ROI_MASKS_TEST_DIR, output_root, writer)

    print(f"\nConcluído. Métricas guardadas em: {csv_path}")


if __name__ == "__main__":
    main()