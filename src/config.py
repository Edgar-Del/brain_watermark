from pathlib import Path


# =========================
# CAMINHOS PRINCIPAIS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "Brain_Tumor_MRI_Dataset"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
WATERMARK_DIR = DATA_DIR / "watermark"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
WATERMARKED_DIR = OUTPUTS_DIR / "watermarked"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"


# =========================
# DADOS BRUTOS
# =========================
RAW_TRAIN_DIR = RAW_DATA_DIR / "Training"
RAW_TEST_DIR = RAW_DATA_DIR / "Testing"


# =========================
# DADOS PROCESSADOS
# =========================
BINARY_DIR = PROCESSED_DATA_DIR / "binary"
BINARY_TRAIN_DIR = BINARY_DIR / "train"
BINARY_TEST_DIR = BINARY_DIR / "test"

MULTICLASS_DIR = PROCESSED_DATA_DIR / "multiclass"
MULTICLASS_TRAIN_DIR = MULTICLASS_DIR / "train"
MULTICLASS_TEST_DIR = MULTICLASS_DIR / "test"

ROI_MASKS_DIR = PROCESSED_DATA_DIR / "roi_masks"
ROI_MASKS_TRAIN_DIR = ROI_MASKS_DIR / "train"
ROI_MASKS_TEST_DIR = ROI_MASKS_DIR / "test"


# =========================
# CLASSES
# =========================
RAW_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
TUMOR_CLASSES = ["glioma", "meningioma", "pituitary"]
BINARY_CLASSES = ["tumor", "notumor"]


# =========================
# PARÂMETROS DE IMAGEM
# =========================
IMAGE_SIZE = (256, 256)   # (width, height)
WATERMARK_SIZE = (64, 64)


# =========================
# PARÂMETROS DE WATERMARK
# =========================
DEFAULT_WAVELET = "haar"
DEFAULT_SUBBAND = "HL"

V1_ALPHA = 0.08

V2_ALPHA_BASE = 0.12
V2_BLOCK_SIZE = 32
V2_ROI_ALPHA_FACTOR = 0.0
V2_ROI_THRESHOLD = 0.05


# =========================
# PARÂMETROS CAD
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.30


def ensure_directories() -> None:
    """
    Garante que as principais pastas do projecto existem.
    """
    directories = [
        DATA_DIR,
        PROCESSED_DATA_DIR,
        WATERMARK_DIR,
        OUTPUTS_DIR,
        WATERMARKED_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        BINARY_TRAIN_DIR / "tumor",
        BINARY_TRAIN_DIR / "notumor",
        BINARY_TEST_DIR / "tumor",
        BINARY_TEST_DIR / "notumor",
        MULTICLASS_TRAIN_DIR / "glioma",
        MULTICLASS_TRAIN_DIR / "meningioma",
        MULTICLASS_TRAIN_DIR / "notumor",
        MULTICLASS_TRAIN_DIR / "pituitary",
        MULTICLASS_TEST_DIR / "glioma",
        MULTICLASS_TEST_DIR / "meningioma",
        MULTICLASS_TEST_DIR / "notumor",
        MULTICLASS_TEST_DIR / "pituitary",
        ROI_MASKS_TRAIN_DIR / "glioma",
        ROI_MASKS_TRAIN_DIR / "meningioma",
        ROI_MASKS_TRAIN_DIR / "notumor",
        ROI_MASKS_TRAIN_DIR / "pituitary",
        ROI_MASKS_TEST_DIR / "glioma",
        ROI_MASKS_TEST_DIR / "meningioma",
        ROI_MASKS_TEST_DIR / "notumor",
        ROI_MASKS_TEST_DIR / "pituitary",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)