"""Training entry point aligned with the notebook data contract."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from detr3d.data import NuScenesDetr3DDataset, detr3d_collate
from detr3d.engine.trainer import train_one_epoch
from detr3d.models import Detr3DModel
from detr3d.models.backbone import MultiViewImageBackbone
from detr3d.models.heads import Detr3DHead
from detr3d.models.losses import Detr3DLoss
from detr3d.models.neck import ImageFPN
from detr3d.models.transformer import Detr3DTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pure PyTorch DETR3D baseline.")
    parser.add_argument("--dataroot", type=str, default="/home/user/datasets/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=448)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model(num_queries: int) -> Detr3DModel:
    return Detr3DModel(
        backbone=MultiViewImageBackbone(),
        neck=ImageFPN(),
        transformer=Detr3DTransformer(num_queries=num_queries),
        head=Detr3DHead(),
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset = NuScenesDetr3DDataset(
        dataroot=args.dataroot,
        version=args.version,
        image_size=(args.image_height, args.image_width),
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detr3d_collate,
    )

    model = build_model(num_queries=args.num_queries).to(device)
    criterion = Detr3DLoss(num_classes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        metrics = train_one_epoch(model, criterion, dataloader, optimizer, device)
        print(f"epoch={epoch} loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
