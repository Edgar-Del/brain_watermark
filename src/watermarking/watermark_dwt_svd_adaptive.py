from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pywt


@dataclass
class AdaptiveWatermarkResult:
    watermarked_image: np.ndarray
    original_singular_values_by_block: list[np.ndarray]
    alpha_by_block: list[float]
    used_blocks: list[tuple[int, int, int]]
    watermark_shape: tuple[int, int]
    wavelet: str
    subband: str
    block_size: int
    alpha_base: float


class AdaptiveDWTSVDWatermarker:
    """
    Watermarking DWT-SVD adaptativo com máscara ROI/RONI e processamento por blocos.

    Ideia:
    - aplica DWT;
    - selecciona uma sub-banda;
    - divide a sub-banda em blocos;
    - calcula alpha por bloco com base em:
        * energia local;
        * intersecção com ROI;
    - modifica apenas blocos permitidos.

    Esquema de extracção: semi-blind.
    """

    def __init__(
        self,
        wavelet: str = "haar",
        subband: str = "HL",
        block_size: int = 32,
        alpha_base: float = 0.12,
        roi_alpha_factor: float = 0.0,
        roi_threshold: float = 0.05,
    ) -> None:
        self.wavelet = wavelet
        self.subband = subband.upper()
        self.block_size = int(block_size)
        self.alpha_base = float(alpha_base)
        self.roi_alpha_factor = float(roi_alpha_factor)
        self.roi_threshold = float(roi_threshold)

        valid_subbands = {"LL", "LH", "HL", "HH"}
        if self.subband not in valid_subbands:
            raise ValueError(f"Sub-banda inválida. Escolha entre {valid_subbands}.")

        if self.block_size <= 0:
            raise ValueError("block_size deve ser maior que zero.")

        if self.alpha_base <= 0:
            raise ValueError("alpha_base deve ser maior que zero.")

        if not (0.0 <= self.roi_alpha_factor <= 1.0):
            raise ValueError("roi_alpha_factor deve estar no intervalo [0, 1].")

        if not (0.0 <= self.roi_threshold <= 1.0):
            raise ValueError("roi_threshold deve estar no intervalo [0, 1].")

    def _select_subband(
        self,
        coeffs: tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        ll, (lh, hl, hh) = coeffs
        if self.subband == "LL":
            return ll, (lh, hl, hh)
        if self.subband == "LH":
            return lh, (ll, hl, hh)
        if self.subband == "HL":
            return hl, (ll, lh, hh)
        return hh, (ll, lh, hl)

    def _replace_subband(
        self,
        modified_subband: np.ndarray,
        others: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self.subband == "LL":
            lh, hl, hh = others
            return modified_subband, (lh, hl, hh)
        if self.subband == "LH":
            ll, hl, hh = others
            return ll, (modified_subband, hl, hh)
        if self.subband == "HL":
            ll, lh, hh = others
            return ll, (lh, modified_subband, hh)

        ll, lh, hl = others
        return ll, (lh, hl, modified_subband)

    @staticmethod
    def _split_into_blocks(image: np.ndarray, block_size: int) -> list[tuple[int, int, np.ndarray]]:
        h, w = image.shape
        blocks: list[tuple[int, int, np.ndarray]] = []

        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = image[i:i + block_size, j:j + block_size]
                if block.shape == (block_size, block_size):
                    blocks.append((i, j, block))

        return blocks

    @staticmethod
    def _place_block(canvas: np.ndarray, block: np.ndarray, i: int, j: int) -> None:
        h, w = block.shape
        canvas[i:i + h, j:j + w] = block

    @staticmethod
    def _local_energy(block: np.ndarray) -> float:
        block = block.astype(np.float64)
        return float(np.sum(block ** 2) / block.size)

    def _block_intersects_roi(self, mask_block: np.ndarray) -> bool:
        roi_ratio = float(np.mean(mask_block > 0))
        return roi_ratio >= self.roi_threshold

    def _compute_block_alpha(
        self,
        block: np.ndarray,
        mask_block: np.ndarray,
        max_energy: float,
    ) -> float:
        energy = self._local_energy(block)
        psi = energy / (max_energy + 1e-12)

        if self._block_intersects_roi(mask_block):
            phi = self.roi_alpha_factor
        else:
            phi = 1.0

        return float(self.alpha_base * phi * psi)

    def _resize_mask_to_subband(self, roi_mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """
        Reduz a máscara para a resolução da sub-banda sem depender de OpenCV.
        """
        mask = roi_mask.astype(np.float64)
        coeffs_mask = pywt.dwt2(mask, self.wavelet)
        ll_mask, _ = coeffs_mask

        resized = ll_mask
        resized = resized[: target_shape[0], : target_shape[1]]

        # Se ainda houver diferença, ajusta por corte/padding simples
        out = np.zeros(target_shape, dtype=np.uint8)
        h = min(target_shape[0], resized.shape[0])
        w = min(target_shape[1], resized.shape[1])
        out[:h, :w] = (resized[:h, :w] > np.mean(resized)).astype(np.uint8)

        return out

    def embed(
        self,
        host_image: np.ndarray,
        watermark: np.ndarray,
        roi_mask: np.ndarray,
    ) -> AdaptiveWatermarkResult:
        if host_image.ndim != 2:
            raise ValueError(f"host_image deve ser 2D. Recebido shape {host_image.shape}.")

        if watermark.ndim != 2:
            raise ValueError(f"watermark deve ser 2D. Recebido shape {watermark.shape}.")

        if roi_mask.ndim != 2:
            raise ValueError(f"roi_mask deve ser 2D. Recebido shape {roi_mask.shape}.")

        host = host_image.astype(np.float64)
        wm = watermark.astype(np.float64)
        mask = (roi_mask > 0).astype(np.uint8)

        coeffs = pywt.dwt2(host, self.wavelet)
        target_subband, others = self._select_subband(coeffs)

        target_h, target_w = target_subband.shape
        mask_resized = self._resize_mask_to_subband(mask, (target_h, target_w))

        image_blocks = self._split_into_blocks(target_subband, self.block_size)
        mask_blocks = self._split_into_blocks(mask_resized, self.block_size)
        mask_dict = {(i, j): block for i, j, block in mask_blocks}

        energies = [self._local_energy(block) for _, _, block in image_blocks]
        max_energy = max(energies) if energies else 1.0

        modified_subband = target_subband.copy()

        wm_flat = wm.flatten()
        wm_length = len(wm_flat)
        wm_pointer = 0

        original_singular_values_by_block: list[np.ndarray] = []
        alpha_by_block: list[float] = []
        used_blocks: list[tuple[int, int, int]] = []

        for i, j, block in image_blocks:
            if wm_pointer >= wm_length:
                break

            mask_block = mask_dict.get((i, j))
            if mask_block is None:
                continue

            alpha_k = self._compute_block_alpha(block, mask_block, max_energy)

            if alpha_k <= 1e-12:
                continue

            u, s, vh = np.linalg.svd(block, full_matrices=False)

            usable_length = min(len(s), wm_length - wm_pointer)
            wm_segment = wm_flat[wm_pointer:wm_pointer + usable_length]

            s_original = s.copy()
            s_modified = s.copy()
            s_modified[:usable_length] += alpha_k * wm_segment

            modified_block = u @ np.diag(s_modified) @ vh
            self._place_block(modified_subband, modified_block, i, j)

            original_singular_values_by_block.append(s_original)
            alpha_by_block.append(alpha_k)
            used_blocks.append((i, j, usable_length))

            wm_pointer += usable_length

        modified_coeffs = self._replace_subband(modified_subband, others)
        watermarked = pywt.idwt2(modified_coeffs, self.wavelet)
        watermarked = watermarked[: host.shape[0], : host.shape[1]]

        return AdaptiveWatermarkResult(
            watermarked_image=np.clip(watermarked, 0, 255).astype(np.uint8),
            original_singular_values_by_block=original_singular_values_by_block,
            alpha_by_block=alpha_by_block,
            used_blocks=used_blocks,
            watermark_shape=wm.shape,
            wavelet=self.wavelet,
            subband=self.subband,
            block_size=self.block_size,
            alpha_base=self.alpha_base,
        )

    def extract(
        self,
        watermarked_image: np.ndarray,
        result: AdaptiveWatermarkResult,
    ) -> np.ndarray:
        if watermarked_image.ndim != 2:
            raise ValueError(f"watermarked_image deve ser 2D. Recebido shape {watermarked_image.shape}.")

        watermarked = watermarked_image.astype(np.float64)

        coeffs_w = pywt.dwt2(watermarked, self.wavelet)
        target_subband_w, _ = self._select_subband(coeffs_w)

        extracted_flat = np.zeros(result.watermark_shape[0] * result.watermark_shape[1], dtype=np.float64)
        pointer = 0

        for idx, (i, j, usable_length) in enumerate(result.used_blocks):
            block = target_subband_w[i:i + self.block_size, j:j + self.block_size]

            if block.shape != (self.block_size, self.block_size):
                continue

            _, s_w, _ = np.linalg.svd(block, full_matrices=False)

            s_orig = result.original_singular_values_by_block[idx]
            alpha_k = result.alpha_by_block[idx]

            if alpha_k <= 1e-12:
                continue

            estimated = (s_w[:usable_length] - s_orig[:usable_length]) / alpha_k
            extracted_flat[pointer:pointer + usable_length] = estimated
            pointer += usable_length

        extracted = extracted_flat.reshape(result.watermark_shape)

        min_val = float(extracted.min())
        max_val = float(extracted.max())

        if max_val - min_val > 1e-12:
            extracted = (extracted - min_val) / (max_val - min_val)
        else:
            extracted = np.zeros_like(extracted)

        return extracted