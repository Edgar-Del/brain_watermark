import numpy as np
import pywt
import cv2
from dataclasses import dataclass


@dataclass
class WatermarkResult:
    watermarked_image: np.ndarray


class DWTSVDWatermarker:

    def __init__(self, wavelet="haar", alpha=0.1, subband="HL"):
        self.wavelet = wavelet
        self.alpha = alpha
        self.subband = subband

    def embed(self, host_image: np.ndarray, watermark: np.ndarray) -> WatermarkResult:
        host = host_image.astype(np.float64)

        coeffs = pywt.dwt2(host, self.wavelet)
        LL, (LH, HL, HH) = coeffs

        if self.subband == "HL":
            target = HL
        elif self.subband == "LH":
            target = LH
        else:
            target = HH

        wm_resized = cv2.resize(watermark, (target.shape[1], target.shape[0]))
        wm_resized = wm_resized.astype(np.float64)

        U, S, Vt = np.linalg.svd(target, full_matrices=False)
        Uw, Sw, Vtw = np.linalg.svd(wm_resized, full_matrices=False)

        S_mod = S + self.alpha * Sw

        target_mod = U @ np.diag(S_mod) @ Vt

        if self.subband == "HL":
            HL = target_mod
        elif self.subband == "LH":
            LH = target_mod
        else:
            HH = target_mod

        watermarked = pywt.idwt2((LL, (LH, HL, HH)), self.wavelet)

        return WatermarkResult(
            watermarked_image=np.clip(watermarked, 0, 255).astype(np.uint8)
        )