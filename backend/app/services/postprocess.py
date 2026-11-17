"""Segmentation post-processing: removes speckle and fills holes."""
import numpy as np
from scipy import ndimage as ndi


def postprocess_mask(prob: np.ndarray, threshold: float = 0.5, min_area_px: int = 20,
                     keep_ratio: float = 0.2, fill_holes: bool = True) -> np.ndarray:
    """prob: (H,W) tumour probability -> uint8 {0,1} mask.

    * thresholds the probability map
    * drops connected components smaller than `min_area_px` OR smaller than `keep_ratio` x the largest
      component (so a genuinely multi-focal lesion is kept, but isolated speckle is removed)
    * fills internal holes (whole-tumour masks are solid)
    """
    binary = prob > threshold
    if not binary.any():
        return np.zeros(prob.shape, dtype=np.uint8)
    labels, n = ndi.label(binary)
    if n == 0:
        return np.zeros(prob.shape, dtype=np.uint8)
    sizes = ndi.sum(binary, labels, index=np.arange(1, n + 1))
    cutoff = max(min_area_px, keep_ratio * sizes.max())
    keep_ids = [i + 1 for i, s in enumerate(sizes) if s >= cutoff]
    out = np.isin(labels, keep_ids)
    if fill_holes:
        out = ndi.binary_fill_holes(out)
    return out.astype(np.uint8)
