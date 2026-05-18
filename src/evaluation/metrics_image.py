from __future__ import annotations

from typing import Any, cast
import numpy as np
from skimage.metrics import structural_similarity as ssim


def _validate_image_pair(image_a: np.ndarray, image_b: np.ndarray) -> None:
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"image_a e image_b devem ter o mesmo shape, mas receberam "
            f"{image_a.shape} e {image_b.shape}."
        )


def mse(image_a: np.ndarray, image_b: np.ndarray) -> float:
    _validate_image_pair(image_a, image_b)
    image_a = image_a.astype(np.float64)
    image_b = image_b.astype(np.float64)
    return float(np.mean((image_a - image_b) ** 2))


def psnr(image_a: np.ndarray, image_b: np.ndarray, data_range: float = 255.0) -> float:
    _validate_image_pair(image_a, image_b)
    err = mse(image_a, image_b)
    if err == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / err))


def compute_ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    _validate_image_pair(image_a, image_b)

    image_a = image_a.astype(np.float64)
    image_b = image_b.astype(np.float64)

    data_min = float(min(float(image_a.min()), float(image_b.min())))
    data_max = float(max(float(image_a.max()), float(image_b.max())))
    data_range = data_max - data_min

    if data_range == 0:
        return 1.0

    if image_a.ndim < 2:
        raise ValueError(f"SSIM requer pelo menos 2 dimensões. Recebido shape {image_a.shape}.")

    min_spatial_dim = min(image_a.shape[0], image_a.shape[1])
    if min_spatial_dim < 3:
        raise ValueError(
            "SSIM requer dimensões espaciais >= 3. "
            f"Recebido shape {image_a.shape}."
        )

    win_size = min(7, min_spatial_dim)
    if win_size % 2 == 0:
        win_size -= 1

    channel_axis = -1 if image_a.ndim == 3 else None

    ssim_result: Any = ssim(
        image_a,
        image_b,
        data_range=data_range,
        win_size=win_size,
        channel_axis=channel_axis,
        full=False,
        gradient=False,
    )

    # O runtime aqui deve devolver um escalar.
    # Este bloco torna isso explícito para o Pylance e evita o erro de tipagem.
    if isinstance(ssim_result, tuple):
        ssim_scalar = ssim_result[0]
    else:
        ssim_scalar = ssim_result

    return float(cast(float, ssim_scalar))