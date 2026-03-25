"""Top-level DETR3D composition module."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class Detr3DModel(nn.Module):
    """Wires backbone, neck, transformer, and head into one module."""

    def __init__(self, backbone: nn.Module, neck: nn.Module, transformer: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.transformer = transformer
        self.head = head

    def forward(self, images: torch.Tensor, img_metas: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        # `images`: [B, N_cam, 3, H, W]
        features = self.backbone(images)
        # Multi-scale per-camera pyramid, e.g. p3..p6 with shape [B, N_cam, C, H_l, W_l].
        pyramid = self.neck(features)
        if img_metas is None:
            raise ValueError("img_metas is required for Detr3DModel.")

        mlvl_feats = list(pyramid.values())
        batch_size = mlvl_feats[0].shape[0]
        # Learned object queries are the state propagated through the 6 decoder layers.
        query, query_pos = self.transformer.init_decoder_state(batch_size, mlvl_feats[0].device)
        reference_points = self.head.init_reference_points(query_pos)
        init_reference = reference_points

        cls_scores = []
        bbox_preds = []
        inter_references = []
        hidden = query
        for layer_idx, layer in enumerate(self.transformer.layers):
            inter_references.append(reference_points)

            # Each layer samples image features using the current normalized 3D references.
            hidden = layer(
                query=hidden,
                mlvl_feats=mlvl_feats,
                query_pos=query_pos,
                reference_points=reference_points,
                img_metas=img_metas,
            )

            reg_output = self.head.regress_boxes(layer_idx, hidden)
            cls_score = self.head.classify(layer_idx, hidden)
            bbox_pred = self.head._encode_box_predictions(reg_output, reference_points)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

            if layer_idx < len(self.transformer.layers) - 1:
                reference_points = self.head.refine_reference_points_from_reg_output(
                    reg_output=reg_output,
                    reference_points=reference_points,
                ).detach()

        return {
            "cls_scores": torch.stack(cls_scores),
            "bbox_preds": torch.stack(bbox_preds),
            "init_reference": init_reference,
            "inter_references": torch.stack(inter_references),
        }
