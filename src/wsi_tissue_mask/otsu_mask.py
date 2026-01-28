"""Otsu binarization tissue masking (fallback for non-H&E stains)."""

from __future__ import annotations

import cv2
import numpy as np


def otsu_tissue_mask(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Binary tissue mask via Otsu threshold on grayscale.

    Args:
        image:       RGB thumbnail (H, W, 3) uint8.
        kernel_size: Morphological cleanup kernel.

    Returns:
        Binary mask (H, W) bool.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    return cleaned > 0
