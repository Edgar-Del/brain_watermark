from __future__ import annotations

import cv2
import numpy as np

from src.config import WATERMARK_DIR, WATERMARK_SIZE, ensure_directories


def main() -> None:
    ensure_directories()
    WATERMARK_DIR.mkdir(parents=True, exist_ok=True)

    width, height = WATERMARK_SIZE
    watermark = np.zeros((height, width), dtype=np.uint8)

    cv2.putText(
        watermark,
        "WM",
        (5, height // 2 + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        255,
        2,
        cv2.LINE_AA,
    )

    output_path = WATERMARK_DIR / "watermark.png"
    cv2.imwrite(str(output_path), watermark)

    print(f"Watermark criada com sucesso em: {output_path}")


if __name__ == "__main__":
    main()