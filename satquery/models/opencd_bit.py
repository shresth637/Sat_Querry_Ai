import math
from pathlib import Path
import time
from typing import Any, Optional, Union
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import Affine
import torch
import torch.nn as nn
import torch.nn.functional as F

from satquery.domain.schemas import ModelResult
from satquery.models.base import ModelAdapter


# ---------------------------------------------------------------------------
# Pure PyTorch BIT Architecture (Chen et al. / Open-CD ResNetV1c + BITHead)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetStem(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )


class ResNetV1cBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = ResNetStem()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),
        )
        down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2, downsample=down2),
            BasicBlock(128, 128, stride=1),
        )
        down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=1, downsample=down3),
            BasicBlock(256, 256, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer3(self.layer2(self.layer1(self.maxpool(self.stem(x)))))


class CrossAttention(nn.Module):
    def __init__(self, in_dims: int, embed_dims: int, num_heads: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = in_dims ** -0.5
        self.to_q = nn.Linear(in_dims, embed_dims, bias=False)
        self.to_k = nn.Linear(in_dims, embed_dims, bias=False)
        self.to_v = nn.Linear(in_dims, embed_dims, bias=False)
        self.fc_out = nn.Sequential(
            nn.Linear(embed_dims, in_dims),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        b, n = x.shape[:2]
        h = self.num_heads
        q = self.to_q(x).reshape((b, n, h, -1)).permute((0, 2, 1, 3))
        k = self.to_k(ref).reshape((b, ref.shape[1], h, -1)).permute((0, 2, 1, 3))
        v = self.to_v(ref).reshape((b, ref.shape[1], h, -1)).permute((0, 2, 1, 3))
        mult = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        mult = F.softmax(mult, dim=-1)
        out = torch.matmul(mult, v).permute((0, 2, 1, 3)).flatten(2)
        return self.fc_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, in_dims: int = 32, embed_dims: int = 64, num_heads: int = 8, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.attn = CrossAttention(in_dims, embed_dims, num_heads, dropout_rate=drop_rate)
        self.ff = nn.Sequential(
            nn.Linear(in_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dims, in_dims),
            nn.Dropout(drop_rate),
        )
        self.norm1 = nn.LayerNorm(in_dims)
        self.norm2 = nn.LayerNorm(in_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.attn(self.norm1(x), self.norm1(x)) + x
        return self.ff(self.norm2(x_)) + x_


class TransformerDecoder(nn.Module):
    def __init__(self, in_dims: int = 32, embed_dims: int = 64, num_heads: int = 8, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.attn = CrossAttention(in_dims, embed_dims, num_heads, dropout_rate=drop_rate)
        self.ff = nn.Sequential(
            nn.Linear(in_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dims, in_dims),
            nn.Dropout(drop_rate),
        )
        self.norm1 = nn.LayerNorm(in_dims)
        self.norm1_ = nn.LayerNorm(in_dims)
        self.norm2 = nn.LayerNorm(in_dims)

    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        x_ = self.attn(self.norm1(x), self.norm1_(ref)) + x
        return self.ff(self.norm2(x_)) + x_


class BITNet(nn.Module):
    """Bitemporal Image Transformer network."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResNetV1cBackbone()
        self.token_len = 4
        self.channels = 32
        self.conv_att = nn.Module()
        self.conv_att.conv = nn.Conv2d(32, 4, 1)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, 8, 32))
        self.pre_process = nn.Sequential(
            nn.Identity(),
            nn.Module(),
        )
        self.pre_process[1].conv = nn.Conv2d(256, 32, 3, padding=1)
        self.encoder = nn.ModuleList([TransformerEncoder(32, 64, 8)])
        self.decoder = nn.ModuleList([TransformerDecoder(32, 64, 8) for _ in range(8)])
        self.conv_seg = nn.Conv2d(32, 2, kernel_size=1)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        att_map = self.conv_att.conv(x).reshape((b, self.token_len, 1, -1))
        att_map = F.softmax(att_map, dim=-1)
        x_flat = x.reshape((b, 1, c, -1))
        return (x_flat * att_map).sum(-1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Backbone feature extraction
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        # Preprocess
        f1 = F.interpolate(f1, scale_factor=2, mode='bilinear', align_corners=False)
        f2 = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=False)
        x1_p = F.relu(self.pre_process[1].conv(f1))
        x2_p = F.relu(self.pre_process[1].conv(f2))
        # Context Tokenization
        t1 = self.forward_tokens(x1_p)
        t2 = self.forward_tokens(x2_p)
        t = torch.cat([t1, t2], dim=1) + self.enc_pos_embedding
        for enc in self.encoder:
            t = enc(t)
        t1, t2 = torch.chunk(t, 2, dim=1)
        # Decoder
        b, c, h, w = x1_p.shape
        x1_flat = x1_p.permute(0, 2, 3, 1).flatten(1, 2)
        x2_flat = x2_p.permute(0, 2, 3, 1).flatten(1, 2)
        for dec in self.decoder:
            x1_flat = dec(x1_flat, t1)
            x2_flat = dec(x2_flat, t2)
        x1_out = x1_flat.transpose(1, 2).reshape(b, c, h, w)
        x2_out = x2_flat.transpose(1, 2).reshape(b, c, h, w)
        # Differencing and classifier projection
        diff = torch.abs(x1_out - x2_out)
        diff_up = F.interpolate(diff, scale_factor=4, mode='bilinear', align_corners=False)
        logits = self.conv_seg(diff_up)
        return logits


# ---------------------------------------------------------------------------
# Open-CD BIT Model Adapter
# ---------------------------------------------------------------------------

class OpenCDBITModel(ModelAdapter):
    """Model adapter for Open-CD BIT change detection."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            name="Open-CD BIT ResNet-18",
            version="r18-levir",
            capabilities=["change_detect"],
            status="unloaded",
        )
        self.weights_path = Path(weights_path) if weights_path else Path("models/checkpoints/bit_r18_256x256_40k_levircd.pth")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[BITNet] = None
        self.output_dir = Path("outputs/change_maps")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Attempt to load checkpoint if on disk
        if self.weights_path.exists():
            self.load()
        else:
            self.status = "not_configured"

    def load(self) -> None:
        """Load pretrained BIT weights from disk."""
        if not self.weights_path.exists():
            self.status = "not_configured"
            return

        try:
            self.model = BITNet()
            ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            raw_state = ckpt.get("state_dict", ckpt)

            # Map mmengine prefix keys to pure BITNet
            mapped_state = {}
            for k, v in raw_state.items():
                if k.startswith("backbone."):
                    mapped_state[k] = v
                elif k.startswith("decode_head."):
                    mapped_state[k[len("decode_head."):]] = v

            self.model.load_state_dict(mapped_state, strict=True)
            self.model.to(self.device)
            self.model.eval()
            self.status = "ready"
        except Exception as exc:
            self.status = "error"
            self.model = None

    def _preprocess_raster(self, path: Union[str, Path], target_size: tuple[int, int] = (256, 256)) -> tuple[torch.Tensor, dict]:
        """Read raster, handle nodata, extract 3 RGB channels, normalize, and return tensor."""
        with rasterio.open(path) as src:
            meta = {
                "crs": src.crs.to_string() if src.crs else None,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "nodata": src.nodata,
                "res": src.res,
            }
            # Read first 3 channels (or repeat if 1 channel)
            if src.count >= 3:
                data = src.read((1, 2, 3), out_shape=(3, target_size[1], target_size[0])).astype(np.float32)
            else:
                band1 = src.read(1, out_shape=(target_size[1], target_size[0])).astype(np.float32)
                data = np.stack([band1, band1, band1], axis=0)

        # Clean nodata / nan
        if meta["nodata"] is not None:
            data[data == meta["nodata"]] = 0.0
        data[~np.isfinite(data)] = 0.0

        # Percentile scale to [0, 255] if data is uint16 or float
        for c in range(3):
            band = data[c]
            p2, p98 = np.percentile(band, (2, 98))
            if p98 > p2:
                data[c] = np.clip((band - p2) / (p98 - p2) * 255.0, 0.0, 255.0)

        # ImageNet standardization: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)
        norm_data = (data - mean) / std

        tensor = torch.from_numpy(norm_data).unsqueeze(0).to(self.device)
        return tensor, meta

    def predict(self, inputs: dict[str, Any], **kwargs: Any) -> ModelResult:
        """Run bi-temporal change detection inference and generate real geospatial artifacts."""
        if self.status != "ready" or self.model is None:
            return ModelResult(
                registry_id="opencd_bit_change",
                capability="change_detect",
                status=self.status,
                text=None,
                artifacts=[],
                raw_scores=None,
                error=f"Model not ready (status: {self.status}). Weights path: {self.weights_path}",
            )

        t0_path = inputs.get("t0")
        t1_path = inputs.get("t1")

        if not t0_path or not t1_path:
            return ModelResult(
                registry_id="opencd_bit_change",
                capability="change_detect",
                status="error",
                error="Inputs must contain 't0' and 't1' raster paths.",
            )

        try:
            # 1. Preprocess both rasters
            tensor0, meta0 = self._preprocess_raster(t0_path, target_size=(256, 256))
            tensor1, meta1 = self._preprocess_raster(t1_path, target_size=(256, 256))

            # 2. Run real model inference under no_grad
            with torch.inference_mode():
                logits = self.model(tensor0, tensor1)
                probs = F.softmax(logits, dim=1)
                change_prob_256 = probs[0, 1].cpu().numpy()  # (256, 256) in [0, 1]

            # 3. Resample probability map back to original spatial dimensions
            orig_w, orig_h = meta0["width"], meta0["height"]
            pil_prob = Image.fromarray((change_prob_256 * 255.0).astype(np.uint8))
            pil_prob_orig = pil_prob.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
            change_prob_orig = np.array(pil_prob_orig).astype(np.float32) / 255.0

            # 4. Generate binary change mask (threshold = 0.5)
            binary_mask = (change_prob_orig >= 0.5).astype(np.uint8)

            # 5. Save output GeoTIFF preserving CRS, transform, and original dimensions
            run_id = f"cd_{int(time.time())}"
            mask_geotiff_path = self.output_dir / f"{run_id}_change_mask.tif"

            crs_val = meta0["crs"]
            transform_val = meta0["transform"]

            with rasterio.open(
                mask_geotiff_path,
                "w",
                driver="GTiff",
                height=orig_h,
                width=orig_w,
                count=1,
                dtype="uint8",
                crs=crs_val,
                transform=transform_val,
            ) as dst:
                dst.write(binary_mask, 1)

            # 6. Generate visual preview of the change map (color-coded overlay)
            vis_png_path = self.output_dir / f"{run_id}_change_vis.png"
            vis_rgb = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            # Red color for changed pixels, dark gray for unchanged
            vis_rgb[binary_mask == 1] = [255, 40, 40]
            vis_rgb[binary_mask == 0] = [30, 30, 35]
            Image.fromarray(vis_rgb).save(vis_png_path)

            # 7. Compute real change statistics
            total_px = int(orig_w * orig_h)
            changed_px = int(np.sum(binary_mask == 1))
            unchanged_px = total_px - changed_px
            pct_changed = round(float(changed_px / total_px) * 100.0, 2) if total_px > 0 else 0.0
            pct_unchanged = round(100.0 - pct_changed, 2)

            stats = {
                "total_pixels": total_px,
                "changed_pixels": changed_px,
                "unchanged_pixels": unchanged_px,
                "percentage_changed": pct_changed,
                "percentage_unchanged": pct_unchanged,
            }

            # Area calculations if resolution is available
            res = meta0.get("res")
            if res and crs_val and "epsg" in crs_val.lower() and not ("4326" in crs_val):
                # Projected CRS (resolution in meters)
                pixel_area_m2 = abs(float(res[0]) * float(res[1]))
                area_m2 = round(changed_px * pixel_area_m2, 1)
                area_ha = round(area_m2 / 10000.0, 2)
                stats["changed_area_sq_m"] = area_m2
                stats["changed_area_ha"] = area_ha
                area_text = f" Covering approximately {area_m2:,.1f} m² ({area_ha:,.2f} hectares)."
            else:
                stats["changed_area_sq_m"] = None
                stats["changed_area_ha"] = None
                area_text = " (Physical area calculation unavailable: geographic CRS or resolution unprojected)."

            # 8. Derived confidence from winning softmax probability
            winning_probs = np.maximum(change_prob_orig, 1.0 - change_prob_orig)
            mean_conf = round(float(np.mean(winning_probs)), 4)
            max_conf = round(float(np.max(winning_probs)), 4)

            summary_text = (
                f"Bi-temporal change detection completed successfully via Open-CD BIT.\n\n"
                f"- **Changed Area:** {changed_px:,} pixels ({pct_changed}% of the scene).\n"
                f"- **Unchanged Area:** {unchanged_px:,} pixels ({pct_unchanged}% of the scene).\n"
                f"- **Geospatial Reference:** CRS {crs_val or 'Not available'}.{area_text}"
            )

            artifacts = [
                {
                    "type": "change_mask_geotiff",
                    "path": str(mask_geotiff_path),
                    "crs": crs_val,
                    "dimensions": [orig_w, orig_h],
                },
                {
                    "type": "change_visualization",
                    "path": str(vis_png_path),
                },
                {
                    "type": "change_statistics",
                    "data": stats,
                },
            ]

            raw_scores = {
                "confidence_score": mean_conf,
                "max_confidence": max_conf,
                "percentage_changed": pct_changed,
                "changed_pixels": changed_px,
                "total_pixels": total_px,
            }

            return ModelResult(
                registry_id="opencd_bit_change",
                capability="change_detect",
                status="success",
                text=summary_text,
                artifacts=artifacts,
                raw_scores=raw_scores,
                error=None,
            )

        except Exception as exc:
            return ModelResult(
                registry_id="opencd_bit_change",
                capability="change_detect",
                status="error",
                error=f"Open-CD BIT inference error: {str(exc)}",
            )
