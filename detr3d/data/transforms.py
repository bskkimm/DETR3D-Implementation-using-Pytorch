"""Image augmentations used by the official DETR3D training pipeline."""

from __future__ import annotations

import cv2
import numpy as np


def photometric_distort_bgr(
    image: np.ndarray,
    brightness_delta: float = 32.0,
    contrast_range: tuple[float, float] = (0.5, 1.5),
    saturation_range: tuple[float, float] = (0.5, 1.5),
    hue_delta: float = 18.0,
) -> np.ndarray:
    """Match the upstream per-view BGR photometric distortion sequence."""
    image = image.astype(np.float32, copy=True)
    if np.random.randint(2):
        image += np.random.uniform(-brightness_delta, brightness_delta)

    mode = np.random.randint(2)
    if mode == 1 and np.random.randint(2):
        image *= np.random.uniform(*contrast_range)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if np.random.randint(2):
        image[..., 1] *= np.random.uniform(*saturation_range)
    if np.random.randint(2):
        image[..., 0] += np.random.uniform(-hue_delta, hue_delta)
        image[..., 0][image[..., 0] > 360] -= 360
        image[..., 0][image[..., 0] < 0] += 360
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    if mode == 0 and np.random.randint(2):
        image *= np.random.uniform(*contrast_range)
    if np.random.randint(2):
        image = image[..., np.random.permutation(3)]
    return image


__all__ = ["photometric_distort_bgr"]
