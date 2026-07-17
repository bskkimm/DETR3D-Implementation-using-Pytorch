import numpy as np
import torch
from torch.utils.data import Dataset

from detr3d.data.nuscenes_dataset import NUSCENES_CLASSES
from detr3d.data.sampler import OFFICIAL_CBGS_CLASSES, CBGSDataset


class MetadataDataset(Dataset):
    def __init__(self, categories, runtime_empty=()):
        self.categories = categories
        self.runtime_empty = set(runtime_empty)

    def __len__(self):
        return len(self.categories)

    def get_cat_ids(self, index):
        return self.categories[index]

    def __getitem__(self, index):
        labels = [] if index in self.runtime_empty else self.categories[index]
        return {"source_index": index, "gt_labels": torch.tensor(labels)}


def balanced_metadata_dataset(**kwargs):
    name_to_id = {name: index for index, name in enumerate(NUSCENES_CLASSES)}
    categories = [[name_to_id[name]] for name in OFFICIAL_CBGS_CLASSES]
    categories += [[name_to_id["car"]], [name_to_id["car"]], []]
    return MetadataDataset(categories, **kwargs)


def test_cbgs_uses_official_equal_quotas_with_replacement():
    dataset = balanced_metadata_dataset()
    car_id = NUSCENES_CLASSES.index("car")
    dataset.categories.extend([[car_id]] * 8)
    wrapper = CBGSDataset(dataset, seed=7)

    # Twenty class incidences produce two draws per pool. Singleton pools must
    # therefore be sampled with replacement.
    assert wrapper.stats["duplicated_sample_count"] == 20
    assert wrapper.stats["per_class_quota"] == 2
    assert len(wrapper) == 20
    for offset, name in enumerate(OFFICIAL_CBGS_CLASSES):
        local_id = NUSCENES_CLASSES.index(name)
        class_draws = wrapper.source_indices[offset * 2 : (offset + 1) * 2]
        assert all(local_id in dataset.get_cat_ids(index) for index in class_draws)
    assert wrapper.source_indices[2:4] == (1, 1)


def test_cbgs_is_deterministic_and_does_not_touch_global_numpy_rng():
    dataset = balanced_metadata_dataset()
    np.random.seed(123)
    expected = np.random.random(4)
    np.random.seed(123)

    first = CBGSDataset(dataset, seed=19)
    observed = np.random.random(4)
    second = CBGSDataset(dataset, seed=19)
    different = CBGSDataset(dataset, seed=20)

    assert np.array_equal(observed, expected)
    assert first.source_indices == second.source_indices
    assert first.fingerprint == second.fingerprint
    assert first.source_indices != different.source_indices


def test_cbgs_excludes_metadata_empty_samples_and_falls_back_at_runtime():
    dataset = balanced_metadata_dataset()
    wrapper = CBGSDataset(dataset, seed=1)
    dataset.runtime_empty.add(wrapper.source_indices[0])

    assert wrapper.stats["empty_source_count"] == 1
    assert len(wrapper[0]["gt_labels"]) > 0


def test_cbgs_rejects_an_empty_class_pool():
    categories = [[NUSCENES_CLASSES.index(name)] for name in OFFICIAL_CBGS_CLASSES[:-1]]

    try:
        CBGSDataset(MetadataDataset(categories))
    except ValueError as error:
        assert "traffic_cone" in str(error)
    else:
        raise AssertionError("expected empty class pool to fail validation")
