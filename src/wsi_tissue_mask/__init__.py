"""wsi-tissue-mask: slide-level tissue segmentation for WSIs."""
from .masker import WSITissueMasker
from .hsv_mask import hsv_tissue_mask
from .otsu_mask import otsu_tissue_mask

__version__ = "0.1.0"
__all__ = ["WSITissueMasker", "hsv_tissue_mask", "otsu_tissue_mask"]
