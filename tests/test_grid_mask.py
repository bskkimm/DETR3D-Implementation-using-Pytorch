import numpy as np
import torch

from detr3d.models.grid_mask import GridMask


def test_grid_mask_is_noop_during_evaluation():
    augmentation = GridMask(probability=1.0).eval()
    images = torch.randn(3, 2, 16, 16)

    assert torch.equal(augmentation(images), images)


def test_grid_mask_is_seeded_and_shared_across_images_and_channels():
    augmentation = GridMask(probability=1.0).train()
    images = torch.ones(3, 2, 16, 16)
    np.random.seed(7)
    first = augmentation(images)
    np.random.seed(7)
    second = augmentation(images)

    assert torch.equal(first, second)
    assert torch.equal(first[0, 0], first[2, 1])
    assert torch.count_nonzero(first) < first.numel()
