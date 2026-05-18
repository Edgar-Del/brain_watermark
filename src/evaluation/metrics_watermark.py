import numpy as np


def normalized_correlation(w1: np.ndarray, w2: np.ndarray) -> float:
    w1 = w1.astype(np.float64).flatten()
    w2 = w2.astype(np.float64).flatten()

    numerator = np.sum(w1 * w2)
    denominator = np.sqrt(np.sum(w1 ** 2) * np.sum(w2 ** 2))

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def bit_error_rate(w1: np.ndarray, w2: np.ndarray) -> float:
    w1_bin = (w1 > 0.5).astype(np.uint8).flatten()
    w2_bin = (w2 > 0.5).astype(np.uint8).flatten()

    if w1_bin.shape != w2_bin.shape:
        raise ValueError("As watermarks devem ter o mesmo tamanho para BER.")

    errors = np.sum(w1_bin != w2_bin)
    return float(errors / len(w1_bin))