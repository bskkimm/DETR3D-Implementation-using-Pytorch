import numpy as np
import torch
from PIL import Image

from detr3d.data.nuscenes_dataset import resize_and_normalize_official_image
from detr3d.data.transforms import photometric_distort_bgr


def test_official_image_preprocessing_uses_bgr_mean_and_divisor_padding():
    rgb = np.array([[[123, 116, 104]]], dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB")

    result = resize_and_normalize_official_image(image, image_size=(1, 1), size_divisor=32)

    assert result.shape == (3, 32, 32)
    assert torch.allclose(result[:, 0, 0], torch.tensor([0.47, -0.28, -0.675]), atol=1e-5)
    assert torch.count_nonzero(result[:, 1:, :]) == 0


def test_photometric_distortion_is_repeatable_for_a_seed():
    image = np.full((4, 4, 3), 100, dtype=np.float32)
    np.random.seed(11)
    first = photometric_distort_bgr(image)
    np.random.seed(11)
    second = photometric_distort_bgr(image)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, image)
