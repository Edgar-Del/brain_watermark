import cv2
import numpy as np


def threshold_segmentation(image: np.ndarray, threshold: int = 0) -> np.ndarray:
    """
    Segmentação simples baseada em limiar (Otsu se threshold=0).
    """
    if threshold == 0:
        _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return (mask > 0).astype(np.uint8)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Mantém apenas a maior componente conectada.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)


def refine_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Refina a máscara com operações morfológicas.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask.astype(np.uint8)


def generate_roi_mask(image: np.ndarray) -> np.ndarray:
    """
    Pipeline completo para gerar uma máscara ROI aproximada.
    """
    mask = threshold_segmentation(image)
    mask = largest_connected_component(mask)
    mask = refine_mask(mask)

    return mask


def invert_mask(mask: np.ndarray) -> np.ndarray:
    return 1 - mask


def mask_ratio(mask: np.ndarray) -> float:
    return float(np.mean(mask > 0))