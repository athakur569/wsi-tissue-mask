"""WSITissueMasker: slide-level tissue segmentation from WSI thumbnails."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import openslide
from PIL import Image

from .hsv_mask import hsv_tissue_mask
from .otsu_mask import otsu_tissue_mask


class WSITissueMasker:
    """Generate tissue masks from WSI thumbnails.

    Args:
        downsample:   Thumbnail downsample factor from level 0.
        mode:         Masking method: 'hsv' (default) or 'otsu'.
        sat_threshold: HSV saturation threshold (mode='hsv' only).
    """

    def __init__(
        self,
        downsample: int = 32,
        mode: str = "hsv",
        sat_threshold: int = 20,
    ) -> None:
        if mode not in ("hsv", "otsu"):
            raise ValueError(f"mode must be 'hsv' or 'otsu', got '{mode}'")
        self.downsample = downsample
        self.mode = mode
        self.sat_threshold = sat_threshold

    def get_mask(self, slide_path: str | Path) -> np.ndarray:
        """Return a boolean tissue mask at downsampled resolution.

        Args:
            slide_path: Path to .svs/.tif WSI file.

        Returns:
            Boolean ndarray (H//downsample, W//downsample).
        """
        thumbnail = self._load_thumbnail(slide_path)
        img = np.array(thumbnail)
        if self.mode == "hsv":
            return hsv_tissue_mask(img, sat_threshold=self.sat_threshold)
        return otsu_tissue_mask(img)

    def tissue_fraction(self, slide_path: str | Path) -> float:
        """Return fraction of the slide covered by tissue."""
        mask = self.get_mask(slide_path)
        return float(mask.sum()) / float(mask.size)

    def save_overlay(
        self,
        slide_path: str | Path,
        output_path: str | Path,
        alpha: float = 0.4,
    ) -> None:
        """Save a thumbnail with tissue mask overlay for visual inspection."""
        thumbnail = self._load_thumbnail(slide_path)
        img = np.array(thumbnail)
        mask = self.get_mask(slide_path).astype(np.uint8) * 255
        overlay = img.copy()
        green = np.zeros_like(img)
        green[:, :, 1] = mask
        blended = cv2.addWeighted(overlay, 1.0, green, alpha, 0)
        Image.fromarray(blended).save(output_path)

    def _load_thumbnail(self, slide_path: str | Path) -> Image.Image:
        slide = openslide.OpenSlide(str(slide_path))
        w, h = slide.dimensions
        thumb_size = (w // self.downsample, h // self.downsample)
        thumbnail = slide.get_thumbnail(thumb_size)
        slide.close()
        return thumbnail.convert("RGB")
