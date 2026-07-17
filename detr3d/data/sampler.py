"""Deterministic dataset sampling strategies."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from torch.utils.data import Dataset

from .nuscenes_dataset import NUSCENES_CLASSES

# This is the class iteration order used by the official DETR3D nuScenes config.
# It intentionally differs from this project's model-label order.
OFFICIAL_CBGS_CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)


class CBGSDataset(Dataset):
    """One-time deterministic class-balanced grouping and sampling wrapper.

    Empty metadata samples are excluded before sampling. As a defensive
    approximation to pipeline-time resampling, an unexpectedly empty fetched
    item is replaced by the next prevalidated wrapper entry in cyclic order.
    """

    def __init__(
        self,
        dataset: Dataset,
        seed: int = 0,
        class_names: Sequence[str] = NUSCENES_CLASSES,
    ) -> None:
        if not hasattr(dataset, "get_cat_ids"):
            raise TypeError("CBGSDataset requires dataset.get_cat_ids(index)")

        self.dataset = dataset
        self.class_names = tuple(class_names)
        local_id_by_name = {name: index for index, name in enumerate(self.class_names)}
        missing = [
            name for name in OFFICIAL_CBGS_CLASSES if name not in local_id_by_name
        ]
        if missing:
            raise ValueError(
                f"CBGS class names missing from local label order: {missing}"
            )

        pools: dict[str, list[int]] = defaultdict(list)
        empty_source_indices = []
        for source_index in range(len(dataset)):
            cat_ids = set(dataset.get_cat_ids(source_index))
            names = {
                self.class_names[cat_id]
                for cat_id in cat_ids
                if 0 <= cat_id < len(self.class_names)
            }
            matched = False
            for name in OFFICIAL_CBGS_CLASSES:
                if name in names:
                    pools[name].append(source_index)
                    matched = True
            if not matched:
                empty_source_indices.append(source_index)

        missing_pools = [name for name in OFFICIAL_CBGS_CLASSES if not pools[name]]
        if missing_pools:
            raise ValueError(f"CBGS cannot sample empty class pools: {missing_pools}")

        duplicated_sample_count = sum(
            len(pools[name]) for name in OFFICIAL_CBGS_CLASSES
        )
        quota = int(duplicated_sample_count / len(OFFICIAL_CBGS_CLASSES))
        rng = np.random.default_rng(seed)
        source_indices = []
        for name in OFFICIAL_CBGS_CLASSES:
            source_indices.extend(
                rng.choice(pools[name], size=quota, replace=True).tolist()
            )

        self.source_indices = tuple(int(index) for index in source_indices)
        if hasattr(dataset, "has_nonempty_gt"):
            nonempty_by_index = [
                bool(dataset.has_nonempty_gt(index)) for index in range(len(dataset))
            ]
            nonempty_indices = [
                index for index, is_nonempty in enumerate(nonempty_by_index) if is_nonempty
            ]
            if not nonempty_indices:
                raise ValueError("CBGS source dataset has no nonempty training samples")
            effective_indices = [
                source_index
                if nonempty_by_index[source_index]
                else int(rng.choice(nonempty_indices))
                for source_index in self.source_indices
            ]
        else:
            effective_indices = list(self.source_indices)
        self.effective_source_indices = tuple(effective_indices)
        self.sample_indices = self.source_indices
        self.fingerprint = hashlib.sha256(
            json.dumps(
                [self.source_indices, self.effective_source_indices],
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        self.stats = {
            "source_size": len(dataset),
            "sampled_size": len(self.source_indices),
            "duplicated_sample_count": duplicated_sample_count,
            "per_class_quota": quota,
            "class_pool_sizes": {
                name: len(pools[name]) for name in OFFICIAL_CBGS_CLASSES
            },
            "empty_source_count": len(empty_source_indices),
            "post_filter_replacement_count": sum(
                source != effective
                for source, effective in zip(
                    self.source_indices, self.effective_source_indices
                )
            ),
        }

    def __len__(self) -> int:
        return len(self.source_indices)

    @staticmethod
    def _has_gt(item) -> bool:
        if not isinstance(item, dict):
            return True
        for key in ("gt_labels", "gt_boxes_lidar", "gt_boxes_ego"):
            if key in item:
                return len(item[key]) > 0
        return True

    def __getitem__(self, index: int):
        if not self.source_indices:
            raise IndexError("CBGSDataset is empty")
        index %= len(self.source_indices)
        for offset in range(len(self.effective_source_indices)):
            source_index = self.effective_source_indices[
                (index + offset) % len(self.source_indices)
            ]
            item = self.dataset[source_index]
            if self._has_gt(item):
                return item
        raise RuntimeError("All prevalidated CBGS samples produced empty ground truth")


__all__ = ["CBGSDataset", "OFFICIAL_CBGS_CLASSES"]
