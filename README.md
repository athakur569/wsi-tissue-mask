# wsi-tissue-mask

Fast tissue segmentation for whole-slide images: HSV saturation masking and Otsu binarization with morphological cleanup. Generates downsampled tissue masks for patch extraction grid planning.

## Installation
```bash
pip install wsi-tissue-mask
```

## Quick start
```python
from wsi_tissue_mask import WSITissueMasker

masker = WSITissueMasker(downsample=32)
mask = masker.get_mask("slide.svs")  # numpy bool array at 1/32 resolution
masker.save_overlay("slide.svs", "overlay.png")
```

## License
MIT
