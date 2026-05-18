from pathlib import Path
import cv2
import numpy as np


def load_grayscale_image(path: str | Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {path}")

    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    return image


def load_color_image(path: str | Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {path}")

    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(image, 0, 255).astype(np.uint8))


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    min_val = image.min()
    max_val = image.max()

    if max_val - min_val < 1e-12:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = 255 * (image - min_val) / (max_val - min_val)
    return normalized.astype(np.uint8)


def binarize_image(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    return (image > threshold).astype(np.uint8)


def resize_image(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def image_shape(path: str | Path) -> tuple[int, int]:
    image = load_grayscale_image(path)
    return image.shape