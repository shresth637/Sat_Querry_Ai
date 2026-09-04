import json
import logging
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
from satquery.models.geospatial import calculate_raster_area, validate_bitemporal_rasters
from satquery.models.tiling import TiledInferenceEngine

logger = logging.getLogger(__name__)


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
        f1 = F.interpolate(f1, scale_factor=2, mode="bilinear", align_corners=False)
        f2 = F.interpolate(f2, scale_factor=2, mode="bilinear", align_corners=False)
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
        diff_up = F.interpolate(diff, scale_factor=4, mode="bilinear", align_corners=False)
        logits = self.conv_seg(diff_up)
        return logits


# ---------------------------------------------------------------------------
# Device Detection & Status Helper
# ---------------------------------------------------------------------------

def get_device_info(requested_device: Optional[str] = None) -> dict[str, Any]:
    """Inspect PyTorch installation and host hardware to determine execution device."""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None

    # Handle requested device
    if requested_device in ["cuda", "gpu"]:
        resolved_device = "cuda" if cuda_available else "cpu"
    elif requested_device == "cpu":
        resolved_device = "cpu"
    else:  # "auto" or None
        resolved_device = "cuda" if cuda_available else "cpu"

    return {
        "device": resolved_device,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "pytorch_version": torch.__version__,
        "requested_device": requested_device or "auto",
    }


# ---------------------------------------------------------------------------
# Open-CD BIT Model Adapter (Production-Grade Phase 3B)
# ---------------------------------------------------------------------------

class OpenCDBITModel(ModelAdapter):
    """Production-grade model adapter for Open-CD BIT change detection.

    Supports:
    - Sliding-window tiled inference for arbitrary resolution satellite scenes.
    - Seamless overlapping tile reconstruction using cosine blend weighting.
    - Configurable change detection thresholding.
    - Multi-artifact outputs (Binary Mask GeoTIFF, Probability GeoTIFF, Visualization PNG, Stats JSON).
    - Robust projected & geographic WGS84 area calculations.
    - Granular confidence and uncertainty reporting.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        tile_size: int = 256,
        tile_overlap: float = 0.25,
        change_threshold: float = 0.5,
        batch_size: int = 4,
    ) -> None:
        super().__init__(
            name="Open-CD BIT ResNet-18",
            version="r18-levir",
            capabilities=["change_detect"],
            status="unloaded",
        )
        self.weights_path = Path(weights_path) if weights_path else Path("models/checkpoints/bit_r18_256x256_40k_levircd.pth")

        # Resolve device
        dev_info = get_device_info(device)
        self.device_str = dev_info["device"]
        self.device = self.device_str
        self.torch_device = torch.device(self.device_str)
        self.cuda_available = dev_info["cuda_available"]
        self.gpu_name = dev_info["gpu_name"]
        self.pytorch_version = dev_info["pytorch_version"]

        # Inference configuration
        self.tile_size = int(tile_size)
        self.tile_overlap = float(tile_overlap)
        self.change_threshold = float(change_threshold)
        self.batch_size = int(batch_size)

        self.model: Optional[BITNet] = None
        self.output_dir = Path("outputs/change_maps")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized OpenCDBITModel [device: {self.device_str}, "
            f"cuda: {self.cuda_available}, pytorch: {self.pytorch_version}]"
        )

        if self.weights_path.exists():
            self.load()
        else:
            self.status = "not_configured"

    def load(self) -> None:
        """Load pretrained BIT weights from disk into memory."""
        if not self.weights_path.exists():
            self.status = "not_configured"
            return

        try:
            self.model = BITNet()
            ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            raw_state = ckpt.get("state_dict", ckpt)

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
            logger.info(f"Open-CD BIT loaded successfully on {self.device_str} from {self.weights_path}")
        except Exception as exc:
            logger.error(f"Failed to load Open-CD BIT checkpoint: {exc}", exc_info=True)
            self.status = "error"
            self.model = None

    def _read_and_preprocess_scene(self, path: Union[str, Path]) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
        """Read full-resolution raster, handle nodata, normalize, and return (3, H, W) array and raw RGB preview.

        Returns:
            norm_chw: Normalized float32 array of shape (3, H, W).
            meta: Raster metadata dictionary.
            raw_rgb: Visual RGB uint8 array of shape (H, W, 3).
        """
        with rasterio.open(path) as src:
            meta = {
                "crs": src.crs.to_string() if src.crs else None,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "nodata": src.nodata,
                "res": src.res,
                "bounds": src.bounds,
                "count": src.count,
            }
            if src.count >= 3:
                data = src.read((1, 2, 3)).astype(np.float32)
            else:
                band1 = src.read(1).astype(np.float32)
                data = np.stack([band1, band1, band1], axis=0)

        # Clean nodata / nan / inf
        if meta["nodata"] is not None:
            data[data == meta["nodata"]] = 0.0
        data[~np.isfinite(data)] = 0.0

        # Percentile scale to [0, 255]
        visual_rgb_chw = np.zeros_like(data, dtype=np.uint8)
        scaled_data = np.copy(data)
        for c in range(3):
            band = data[c]
            p2, p98 = np.percentile(band, (2, 98))
            if p98 > p2:
                scaled = np.clip((band - p2) / (p98 - p2) * 255.0, 0.0, 255.0)
            else:
                scaled = np.clip(band, 0.0, 255.0)
            scaled_data[c] = scaled
            visual_rgb_chw[c] = scaled.astype(np.uint8)

        # ImageNet normalization: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)
        norm_chw = (scaled_data - mean) / std

        raw_rgb = np.transpose(visual_rgb_chw, (1, 2, 0))  # (H, W, 3)
        return norm_chw, meta, raw_rgb

    def _preprocess_raster(self, path: Union[str, Path], target_size: tuple[int, int] = (256, 256)) -> tuple[torch.Tensor, dict]:
        """Backward-compatible preprocessing helper returning resampled tensor and metadata."""
        norm_chw, meta, _ = self._read_and_preprocess_scene(path)
        tensor = torch.from_numpy(norm_chw).unsqueeze(0).to(self.device)
        if (meta["width"], meta["height"]) != target_size:
            tensor = F.interpolate(tensor, size=(target_size[1], target_size[0]), mode="bilinear", align_corners=False)
        return tensor, meta

    def predict(self, inputs: dict[str, Any], **kwargs: Any) -> ModelResult:
        """Run production bi-temporal change detection using tiled sliding-window inference."""
        t_start = time.perf_counter()

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

        # 1. Configurable execution parameters with validation
        threshold = float(kwargs.get("change_threshold", self.change_threshold))
        if not (0.0 <= threshold <= 1.0):
            return ModelResult(
                registry_id="opencd_bit_change",
                capability="change_detect",
                status="error",
                error=f"Invalid change_threshold: {threshold}. Must be between 0.0 and 1.0.",
            )

        tile_sz = int(kwargs.get("tile_size", self.tile_size))
        overlap = float(kwargs.get("tile_overlap", self.tile_overlap))
        batch_sz = int(kwargs.get("batch_size", self.batch_size))

        # 2. Strict Geospatial & Input Validation
        try:
            validate_bitemporal_rasters(t0_path, t1_path)
        except (ValueError, FileNotFoundError) as val_err:
            return ModelResult(
                registry_id="opencd_bit_change",
                capability="change_detect",
                status="error",
                error=f"Open-CD BIT inference error: Raster validation failed: {str(val_err)}",
            )

        try:
            # 3. Read and Preprocess Scenes (Full Resolution)
            t_prep_start = time.perf_counter()
            t0_norm, meta0, rgb0 = self._read_and_preprocess_scene(t0_path)
            t1_norm, meta1, rgb1 = self._read_and_preprocess_scene(t1_path)
            t_prep = time.perf_counter() - t_prep_start

            orig_w, orig_h = meta0["width"], meta0["height"]

            # 4. Tiled Sliding-Window Inference
            t_infer_start = time.perf_counter()
            engine = TiledInferenceEngine(
                tile_size=tile_sz,
                overlap_ratio=overlap,
                batch_size=batch_sz,
                device=self.device,
            )
            prob_map, num_tiles = engine.predict_tiled(self.model, t0_norm, t1_norm)
            t_infer = time.perf_counter() - t_infer_start

            # 5. Postprocessing & Artifact Generation
            t_post_start = time.perf_counter()

            # Binary change mask derived from configured threshold
            binary_mask = (prob_map >= threshold).astype(np.uint8)

            run_id = f"cd_{int(time.time())}"
            crs_val = meta0["crs"]
            transform_val = meta0["transform"]

            # A. Binary Change Mask GeoTIFF
            mask_geotiff_path = self.output_dir / f"{run_id}_change_mask.tif"
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

            # B. Change Probability GeoTIFF
            prob_geotiff_path = self.output_dir / f"{run_id}_change_prob.tif"
            with rasterio.open(
                prob_geotiff_path,
                "w",
                driver="GTiff",
                height=orig_h,
                width=orig_w,
                count=1,
                dtype="float32",
                crs=crs_val,
                transform=transform_val,
            ) as dst:
                dst.write(prob_map.astype(np.float32), 1)

            # C. Visual Change Overlay PNG (Red highlight on T1 imagery)
            vis_png_path = self.output_dir / f"{run_id}_change_vis.png"
            vis_img = np.copy(rgb1)
            # Alpha blend changed pixels with bright red (255, 30, 30)
            changed_mask_bool = binary_mask == 1
            if np.any(changed_mask_bool):
                alpha = 0.55
                vis_img[changed_mask_bool, 0] = np.clip(vis_img[changed_mask_bool, 0] * (1 - alpha) + 255 * alpha, 0, 255).astype(np.uint8)
                vis_img[changed_mask_bool, 1] = np.clip(vis_img[changed_mask_bool, 1] * (1 - alpha) + 30 * alpha, 0, 255).astype(np.uint8)
                vis_img[changed_mask_bool, 2] = np.clip(vis_img[changed_mask_bool, 2] * (1 - alpha) + 30 * alpha, 0, 255).astype(np.uint8)
            Image.fromarray(vis_img).save(vis_png_path)

            # 6. Physical Area Calculation
            total_px = int(orig_w * orig_h)
            changed_px = int(np.sum(binary_mask == 1))
            unchanged_px = total_px - changed_px
            pct_changed = round(float(changed_px / total_px) * 100.0, 2) if total_px > 0 else 0.0
            pct_unchanged = round(100.0 - pct_changed, 2)

            area_info = calculate_raster_area(meta0, changed_px)

            # 7. Comprehensive Confidence Metrics
            winning_probs = np.maximum(prob_map, 1.0 - prob_map)
            mean_conf = round(float(np.mean(winning_probs)), 4)
            max_conf = round(float(np.max(winning_probs)), 4)

            changed_conf = (
                round(float(np.mean(winning_probs[binary_mask == 1])), 4)
                if changed_px > 0
                else None
            )
            unchanged_conf = (
                round(float(np.mean(winning_probs[binary_mask == 0])), 4)
                if unchanged_px > 0
                else None
            )
            low_conf_pct = round(float(np.mean(winning_probs < 0.60) * 100.0), 2)

            t_post = time.perf_counter() - t_post_start
            t_total = time.perf_counter() - t_start

            # D. Comprehensive Statistics Dictionary & JSON
            stats = {
                "total_pixels": total_px,
                "changed_pixels": changed_px,
                "unchanged_pixels": unchanged_px,
                "percentage_changed": pct_changed,
                "percentage_unchanged": pct_unchanged,
                "mean_change_probability": round(float(np.mean(prob_map)), 4),
                "mean_prediction_confidence": mean_conf,
                "maximum_confidence": max_conf,
                "changed_pixel_confidence": changed_conf,
                "unchanged_pixel_confidence": unchanged_conf,
                "low_confidence_pixel_percentage": low_conf_pct,
                "raster_dimensions": {"width": orig_w, "height": orig_h},
                "crs": crs_val,
                "resolution": list(meta0["res"]) if meta0.get("res") else None,
                "area_calculation_method": area_info["area_calculation_method"],
                "changed_area_m2": area_info["changed_area_m2"],
                "changed_area_hectares": area_info["changed_area_hectares"],
                "changed_area_km2": area_info["changed_area_km2"],
                "pixel_area_m2": area_info["pixel_area_m2"],
                "tile_size": tile_sz,
                "tile_overlap": overlap,
                "total_tiles_evaluated": num_tiles,
                "change_threshold": threshold,
                "inference_device": self.device_str,
                "timings_seconds": {
                    "preprocessing": round(t_prep, 3),
                    "inference": round(t_infer, 3),
                    "postprocessing": round(t_post, 3),
                    "total": round(t_total, 3),
                },
            }

            stats_json_path = self.output_dir / f"{run_id}_stats.json"
            with open(stats_json_path, "w", encoding="utf-8") as f_json:
                json.dump(stats, f_json, indent=2)

            # Area summary text
            if area_info["changed_area_m2"] is not None:
                area_text = (
                    f" Physical area changed: approximately {area_info['changed_area_m2']:,.1f} m² "
                    f"({area_info['changed_area_hectares']:,.2f} ha / {area_info['changed_area_km2']:.4f} km²) "
                    f"via {area_info['area_calculation_method']}."
                )
            else:
                area_text = " (Physical metric area calculation unavailable: missing CRS or resolution)."

            summary_text = (
                f"Bi-temporal change detection completed via Open-CD BIT (tiled sliding-window inference).\n\n"
                f"- **Changed Area:** {changed_px:,} pixels ({pct_changed}% of the scene).\n"
                f"- **Unchanged Area:** {unchanged_px:,} pixels ({pct_unchanged}% of the scene).\n"
                f"- **Scene Dimensions:** {orig_w}x{orig_h} ({num_tiles} tiles evaluated at {tile_sz}x{tile_sz}).\n"
                f"- **Selected Threshold:** {threshold:.2f} | **Device:** {self.device_str}.\n"
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
                    "type": "change_prob_geotiff",
                    "path": str(prob_geotiff_path),
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
                    "path": str(stats_json_path),
                },
            ]

            raw_scores = {
                "confidence_score": mean_conf,
                "mean_prediction_confidence": mean_conf,
                "max_confidence": max_conf,
                "changed_pixel_confidence": changed_conf,
                "unchanged_pixel_confidence": unchanged_conf,
                "low_confidence_pixel_percentage": low_conf_pct,
                "percentage_changed": pct_changed,
                "changed_pixels": changed_px,
                "total_pixels": total_px,
                "threshold": threshold,
                "num_tiles": num_tiles,
                "device": self.device_str,
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
            logger.error(f"Open-CD BIT inference error: {exc}", exc_info=True)
            return ModelResult(
                registry_id="opencd_bit_change",
                capability="change_detect",
                status="error",
                error=f"Open-CD BIT inference error: {str(exc)}",
            )
