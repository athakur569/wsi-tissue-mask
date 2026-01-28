"""HSV saturation-based tissue masking."""

from __future__ import annotations

import cv2
import numpy as np


def hsv_tissue_mask(
    image: np.ndarray,
    sat_threshold: int = 20,
    kernel_size: int = 7,
) -> np.ndarray:
    """Generate a binary tissue mask using HSV saturation thresholding.

    Args:
        image:         RGB thumbnail (H, W, 3) uint8.
        sat_threshold: Saturation threshold (0–255). Pixels above = tissue.
        kernel_size:   Morphological cleanup kernel size.

    Returns:
        Binary mask (H, W) bool, True = tissue.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    _, binary = cv2.threshold(sat, sat_threshold, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    return cleaned > 0
