from typing import Dict, List, Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import roi_align


class CNNEncoder(nn.Module):
    """CNN feature extractor with ROI pooling for object-grounded features."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        train_backbone: bool = False,
        pool_size: Tuple[int, int] = (7, 7),
        feature_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.pool_size = pool_size
        self.backbone_name = backbone

        if backbone == "resnet50":
            backbone_model = models.resnet50(pretrained=pretrained)
            modules = list(backbone_model.children())[:-2]
            self.backbone = nn.Sequential(*modules)
            self.feature_dim = backbone_model.fc.in_features
        elif backbone == "resnet101":
            backbone_model = models.resnet101(pretrained=pretrained)
            modules = list(backbone_model.children())[:-2]
            self.backbone = nn.Sequential(*modules)
            self.feature_dim = backbone_model.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        for param in self.backbone.parameters():
            param.requires_grad = train_backbone

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feature_dim, self.feature_dim)
        self.roi_fc = nn.Sequential(
            nn.Linear(self.feature_dim * self.pool_size[0] * self.pool_size[1], self.feature_dim),
            nn.ReLU(inplace=True),
        )
        self.object_norm = nn.LayerNorm(self.feature_dim)
        self.temporal_input_dim = self.feature_dim * 2

    def forward(
        self,
        frames: torch.Tensor,
        detections: Optional[List[List[Dict[str, torch.Tensor]]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract global and object-level features."""
        batch_size, time_steps, c, h, w = frames.size()
        frames = frames.view(batch_size * time_steps, c, h, w)

        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.backbone.parameters())):
            features = self.backbone(frames)
        _, feat_c, feat_h, feat_w = features.size()

        pooled = self.avgpool(features).view(batch_size, time_steps, feat_c)
        pooled = self.fc(pooled)

        frame_object_feats = torch.zeros(batch_size, time_steps, self.feature_dim, device=frames.device)
        frame_object_counts = torch.zeros(batch_size, time_steps, device=frames.device)
        roi_embeddings: List[List[List[torch.Tensor]]] = [[[] for _ in range(time_steps)] for _ in range(batch_size)]

        if detections is not None:
            rois: List[torch.Tensor] = []
            frame_map: List[Tuple[int, int]] = []

            for b in range(batch_size):
                for t in range(time_steps):
                    frame_index = b * time_steps + t
                    frame_detections = (
                        detections[b][t]
                        if b < len(detections) and t < len(detections[b])
                        else []
                    )
                    for det in frame_detections:
                        x1, y1, x2, y2 = det.get("bbox", [0.0, 0.0, float(w), float(h)])
                        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                        # Clamp to frame boundaries
                        x1 = max(0.0, min(x1, float(w - 1)))
                        y1 = max(0.0, min(y1, float(h - 1)))
                        x2 = max(x1 + 1.0, min(x2, float(w)))
                        y2 = max(y1 + 1.0, min(y2, float(h)))
                        roi = torch.tensor(
                            [float(frame_index), x1, y1, x2, y2],
                            device=features.device,
                            dtype=features.dtype,
                        )
                        rois.append(roi)
                        frame_map.append((b, t))

            if rois:
                rois_tensor = torch.stack(rois)
                spatial_scale = feat_w / float(w)
                pooled_rois = roi_align(
                    features,
                    rois_tensor,
                    output_size=self.pool_size,
                    spatial_scale=spatial_scale,
                    sampling_ratio=-1,
                    aligned=True,
                )
                pooled_rois = pooled_rois.view(pooled_rois.size(0), -1)
                roi_embeds = self.roi_fc(pooled_rois)

                for idx, (b, t) in enumerate(frame_map):
                    emb = roi_embeds[idx]
                    frame_object_feats[b, t] += emb
                    frame_object_counts[b, t] += 1.0
                    roi_embeddings[b][t].append(emb.detach())

        # Normalize aggregated object features
        counts = frame_object_counts.unsqueeze(-1).clamp_min(1.0)
        frame_object_feats = frame_object_feats / counts
        zero_mask = frame_object_counts.unsqueeze(-1) == 0
        frame_object_feats = frame_object_feats.masked_fill(zero_mask, 0.0)
        frame_object_feats = self.object_norm(frame_object_feats)

        combined = torch.cat([pooled, frame_object_feats], dim=-1)

        return {
            "frame_features": pooled,
            "frame_object_features": frame_object_feats,
            "combined_features": combined,
            "roi_embeddings": roi_embeddings,
        }
