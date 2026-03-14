"""Validate a local nuScenes extraction before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_META_FILES = [
    "attribute.json",
    "calibrated_sensor.json",
    "category.json",
    "ego_pose.json",
    "instance.json",
    "log.json",
    "map.json",
    "sample.json",
    "sample_annotation.json",
    "sample_data.json",
    "scene.json",
    "sensor.json",
    "visibility.json",
]

REQUIRED_CAMERA_DIRS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an extracted nuScenes dataset tree.")
    parser.add_argument(
        "--dataroot",
        type=Path,
        required=True,
        help="Path to the extracted nuScenes root directory.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        help="Metadata split directory to validate.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sample_data records to resolve against image files.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        fail(f"Missing {description}: {path}")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_meta(dataroot: Path, version: str) -> Path:
    meta_root = dataroot / version
    require_path(meta_root, f"metadata directory for {version}")
    for filename in REQUIRED_META_FILES:
        require_path(meta_root / filename, f"metadata file {filename}")
    return meta_root


def validate_camera_dirs(dataroot: Path) -> None:
    samples_root = dataroot / "samples"
    sweeps_root = dataroot / "sweeps"
    require_path(samples_root, "samples directory")
    require_path(sweeps_root, "sweeps directory")
    for camera_name in REQUIRED_CAMERA_DIRS:
        require_path(samples_root / camera_name, f"samples camera directory {camera_name}")
        require_path(sweeps_root / camera_name, f"sweeps camera directory {camera_name}")


def validate_sample_records(dataroot: Path, meta_root: Path, num_samples: int) -> None:
    sample_data = load_json(meta_root / "sample_data.json")
    image_records = [record for record in sample_data if record.get("fileformat") == "jpg"]
    if not image_records:
        fail("No jpg camera records found in sample_data.json")

    checked = 0
    for record in image_records:
        filename = record.get("filename")
        if not filename:
            continue
        require_path(dataroot / filename, f"sample file referenced by metadata ({filename})")
        checked += 1
        if checked >= num_samples:
            break

    if checked == 0:
        fail("No image records could be validated against files on disk")


def main() -> None:
    args = parse_args()
    dataroot = args.dataroot.expanduser().resolve()
    require_path(dataroot, "nuScenes dataroot")

    meta_root = validate_meta(dataroot, args.version)
    validate_camera_dirs(dataroot)
    validate_sample_records(dataroot, meta_root, args.num_samples)

    print("[OK] nuScenes dataset layout looks valid")
    print(f"[INFO] dataroot: {dataroot}")
    print(f"[INFO] version: {args.version}")
    print(f"[INFO] checked camera dirs: {len(REQUIRED_CAMERA_DIRS)}")
    print(f"[INFO] checked sample files: {args.num_samples}")


if __name__ == "__main__":
    main()
