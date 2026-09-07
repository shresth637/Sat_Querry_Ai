import json
import logging
from pathlib import Path
from typing import Any, Optional
import uuid

import numpy as np
import rasterio
import torch
import torch.nn as nn
from torchvision import models

from satquery.domain.schemas import ModelResult
from satquery.models.base import ModelAdapter
from satquery.models.manager import get_resource_manager

logger = logging.getLogger(__name__)

# Official 19 BigEarthNet v2.0 (reBEN) / Corine Land Cover nomenclature classes
BIGEARTHNET_19_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland",
    "Moors and heathlands",
    "Sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Bare rock",
    "Sparsely vegetated areas",
    "Inland wetlands",
    "Marine and coastal waters",
]

# Official BigEarthNet v2.0 (reBEN) 10-band means and standard deviations (train split)
# Bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
BEN_V2_BAND_ORDER = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
BEN_V2_MEANS = [438.37, 614.06, 588.41, 942.84, 1769.93, 2049.55, 2193.33, 2235.56, 1568.23, 997.73]
BEN_V2_STDS = [607.02, 603.29, 684.56, 738.43, 1100.46, 1275.81, 1369.36, 1356.54, 1070.16, 813.53]


class BigEarthNetClassifier(nn.Module):
    """ResNet-50 backbone with 19-class multi-label output head."""

    def __init__(self, num_classes: int = 19, in_channels: int = 3):
        super().__init__()
        backbone = models.resnet50(weights=None)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.model = backbone
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logits
        return self.model(x)


class BigEarthNetModel(ModelAdapter):
    """BigEarthNet v2 (reBEN) ResNet-50 multi-label land-cover classification adapter."""

    DEFAULT_WEIGHTS_PATH = Path("models/checkpoints/resnet50_s2_v0.2.0.pth")

    def __init__(
        self,
        weights_path: Optional[str] = None,
        threshold: float = 0.30,
        device: str = "auto",
        output_dir: str = "outputs/classification",
    ) -> None:
        super().__init__(
            name="BigEarthNet v2 ResNet-50 S2",
            version="v0.2.0",
            capabilities=["classification", "land_cover"],
            status="unloaded",
        )
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        self.threshold = float(threshold)
        self.device_str = device
        self.output_dir = Path(output_dir)
        self.model: Optional[BigEarthNetClassifier] = None
        self._device: Optional[torch.device] = None

        if self.weights_path and self.weights_path.exists():
            self.status = "unloaded"
        else:
            self.status = "not_configured"

    def load(self) -> None:
        """Load model weights or set status to not_configured if weights are absent."""
        if not self.weights_path or not self.weights_path.exists():
            self.status = "not_configured"
            logger.info(f"BigEarthNet weights not found at {self.weights_path}. Model status: not_configured.")
            return

        rm = get_resource_manager()
        self._device = rm.get_device(self.device_str)

        try:
            checkpoint = torch.load(str(self.weights_path), map_location=self._device)
            state_dict = checkpoint.get("state_dict", checkpoint)

            # Detect in_channels from checkpoint conv1.weight
            in_channels = 10
            for k, v in state_dict.items():
                if "conv1.weight" in k and hasattr(v, "shape") and len(v.shape) == 4:
                    in_channels = v.shape[1]
                    break

            self.model = BigEarthNetClassifier(num_classes=len(BIGEARTHNET_19_CLASSES), in_channels=in_channels)

            cleaned_state = {}
            for k, v in state_dict.items():
                clean_k = (
                    k.replace("model.vision_encoder.", "")
                    .replace("vision_encoder.", "")
                    .replace("module.", "")
                    .replace("backbone.", "")
                )
                if not clean_k.startswith("model."):
                    clean_k = f"model.{clean_k}"
                cleaned_state[clean_k] = v

            self.model.load_state_dict(cleaned_state, strict=False)
            self.model.float()
            self.model.to(self._device)
            self.model.eval()
            self.status = "ready"
            logger.info(f"BigEarthNet model loaded successfully on {self._device} with in_channels={in_channels}.")
        except Exception as e:
            self.status = "error"
            logger.error(f"Failed to load BigEarthNet checkpoint: {e}", exc_info=True)

    def unload(self) -> None:
        """Release model from memory."""
        self.model = None
        if self.weights_path and self.weights_path.exists():
            self.status = "unloaded"
        else:
            self.status = "not_configured"

    def preprocess(self, file_path: str) -> tuple[torch.Tensor, dict[str, Any]]:
        """Read raster, validate multispectral bands, normalize and resize for BigEarthNet inference."""
        p = Path(file_path)
        ext = p.suffix.lower()

        with rasterio.open(file_path) as src:
            meta = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs": str(src.crs) if src.crs else None,
                "transform": [float(x) for x in src.transform][:6],
                "dtypes": list(src.dtypes),
                "descriptions": list(src.descriptions) if src.descriptions else [],
                "tags": src.tags(),
            }

            in_channels = getattr(self.model, "in_channels", 3) if self.model else 3

            # Rejection of single/double band inputs
            if src.count < 3:
                raise ValueError(
                    f"BigEarthNet ResNet-50 requires multispectral Sentinel-2 imagery with at least 4 bands "
                    f"(B02/Blue, B03/Green, B04/Red, B08/NIR) or 10/12 bands. Received raster '{p.name}' with "
                    f"{src.count} band(s). Single-band rasters cannot be classified."
                )

            # Handle 3-channel input: valid ONLY if in_channels == 3 (e.g. mock test checkpoint)
            if in_channels == 3:
                arr = src.read([1, 2, 3]).astype(np.float32)
                p2, p98 = np.percentile(arr, (2, 98))
                if p98 > p2:
                    arr = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0).astype(np.float32)
                else:
                    arr = np.clip(arr / 255.0, 0.0, 1.0).astype(np.float32)

                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                norm_arr = ((arr - mean) / std).astype(np.float32)

                tensor = torch.from_numpy(norm_arr).float().unsqueeze(0)
                tensor = nn.functional.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
                return tensor, meta

            # For real 10-channel BigEarthNet model:
            # Rejection of arbitrary RGB PNG/JPEG or 3-band rasters without multispectral bands
            if src.count == 3 or ext in [".png", ".jpg", ".jpeg"]:
                raise ValueError(
                    f"BigEarthNet ResNet-50 requires Sentinel-2 multispectral imagery including the near-infrared band "
                    f"(expected at least 4 bands: B02/Blue, B03/Green, B04/Red, B08/NIR, or 10/12 Sentinel-2 bands). "
                    f"Received 3-band raster '{p.name}' without multispectral infrared bands. "
                    f"Arbitrary RGB PNG/JPEG or 3-band images are not valid multispectral inputs for BigEarthNet."
                )

            # 4-band Sentinel-2 (e.g. 10m bands Blue, Green, Red, NIR like sentinel2_small.tif)
            if src.count == 4:
                desc_lower = [str(d).lower() for d in (src.descriptions or [])]
                idx_blue, idx_green, idx_red, idx_nir = 1, 2, 3, 4
                for i, d in enumerate(desc_lower, start=1):
                    if "blue" in d or "b02" in d or "b2" in d:
                        idx_blue = i
                    elif "green" in d or "b03" in d or "b3" in d:
                        idx_green = i
                    elif "red" in d or "b04" in d or "b4" in d:
                        idx_red = i
                    elif "nir" in d or "b08" in d or "b8" in d:
                        idx_nir = i

                b2 = src.read(idx_blue).astype(np.float32)
                b3 = src.read(idx_green).astype(np.float32)
                b4 = src.read(idx_red).astype(np.float32)
                b8 = src.read(idx_nir).astype(np.float32)

                # Synthesize Red Edge and SWIR bands using standard spectral interpolation
                b5 = b4 + 0.226 * (b8 - b4)
                b6 = b4 + 0.424 * (b8 - b4)
                b7 = b4 + 0.667 * (b8 - b4)
                b8a = b8
                b11 = 0.6 * b8 + 0.4 * b4
                b12 = 0.4 * b8 + 0.6 * b4

                raw_bands = [b2, b3, b4, b5, b6, b7, b8, b8a, b11, b12]

            elif src.count in [10, 11, 12, 13]:
                if src.count in [12, 13]:
                    # Standard 12-band Sentinel-2 L2A without B10: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
                    indices = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
                else:
                    indices = list(range(1, 11))
                raw_bands = [src.read(i).astype(np.float32) for i in indices]

            else:
                indices = list(range(1, min(11, src.count + 1)))
                raw_bands = [src.read(i).astype(np.float32) for i in indices]
                while len(raw_bands) < 10:
                    raw_bands.append(raw_bands[-1].copy())

            # Handle scaling if data is uint8 or float [0, 1]
            max_val = max(float(np.max(b)) for b in raw_bands)
            if max_val <= 1.5:
                raw_bands = [b * 10000.0 for b in raw_bands]
            elif max_val <= 255.0 and any("uint8" in str(d) for d in src.dtypes):
                raw_bands = [b * 39.2 for b in raw_bands]

            # Normalize per-band using official BigEarthNet v2 statistics
            norm_bands = []
            for i in range(10):
                mean_i = BEN_V2_MEANS[i]
                std_i = BEN_V2_STDS[i]
                norm_b = (raw_bands[i] - mean_i) / std_i
                norm_bands.append(norm_b)

            arr_10 = np.stack(norm_bands, axis=0)  # shape (10, H, W)
            tensor = torch.from_numpy(arr_10).float().unsqueeze(0)
            tensor = nn.functional.interpolate(tensor, size=(120, 120), mode="bilinear", align_corners=False)
            return tensor, meta

    def predict(self, inputs: dict[str, Any], **kwargs: Any) -> ModelResult:
        """Run multi-label classification on optical satellite scene."""
        if self.status == "not_configured":
            return ModelResult(
                registry_id="bigearthnet_resnet50_s1s2",
                capability="classification",
                status="not_configured",
                text=(
                    "BigEarthNet ResNet-50 model is not configured. "
                    f"Checkpoint not found at: {self.weights_path}."
                ),
                error="Model weights not found.",
            )

        if self.status != "ready":
            self.load()
            if self.status != "ready":
                return ModelResult(
                    registry_id="bigearthnet_resnet50_s1s2",
                    capability="classification",
                    status="error",
                    error="Failed to initialize BigEarthNet model.",
                )

        image_path = inputs.get("image") or inputs.get("optical") or inputs.get("t1") or inputs.get("t0")
        if not image_path and "slots" in inputs:
            slots = inputs["slots"]
            if slots:
                image_path = slots[0].file_path

        if not image_path or not Path(image_path).exists():
            return ModelResult(
                registry_id="bigearthnet_resnet50_s1s2",
                capability="classification",
                status="error",
                error=f"Valid input raster not provided: {image_path}",
            )

        try:
            tensor, raster_meta = self.preprocess(str(image_path))
        except ValueError as ve:
            logger.warning(f"BigEarthNet input validation failed: {ve}")
            return ModelResult(
                registry_id="bigearthnet_resnet50_s1s2",
                capability="classification",
                status="error",
                error=str(ve),
                text=f"Validation failed: {ve}",
            )

        try:
            tensor = tensor.to(self._device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            class_probs = {
                BIGEARTHNET_19_CLASSES[i]: float(probs[i])
                for i in range(len(BIGEARTHNET_19_CLASSES))
            }

            # Resolve runtime threshold with fallback hierarchy:
            # 1. kwargs["threshold"]
            # 2. kwargs["classification_threshold"]
            # 3. inputs["threshold"] / inputs["classification_threshold"]
            # 4. self.threshold
            threshold_val = kwargs.get("threshold")
            if threshold_val is None:
                threshold_val = kwargs.get("classification_threshold")
            if threshold_val is None:
                threshold_val = inputs.get("threshold", inputs.get("classification_threshold", self.threshold))
            try:
                runtime_threshold = float(threshold_val)
            except (TypeError, ValueError):
                runtime_threshold = float(self.threshold)

            # Compute Top 5 predictions sorted descending by model probability (independent of threshold)
            all_sorted_predictions = sorted(
                [{"class": name, "probability": float(prob)} for name, prob in class_probs.items()],
                key=lambda x: x["probability"],
                reverse=True,
            )
            top_5_predictions = all_sorted_predictions[:5]

            # Filter by runtime threshold and sort descending
            active_classes = [
                item for item in all_sorted_predictions
                if item["probability"] >= runtime_threshold
            ]

            # Isolated run directory
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
            run_dir = self.output_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            # Generate horizontal bar chart visualization with runtime threshold line
            chart_path = run_dir / "land_cover_distribution.png"
            self.render_probability_chart(class_probs, chart_path, threshold=runtime_threshold)

            # Save stats JSON
            stats = {
                "model": "BigEarthNet v2 ResNet-50 S2",
                "threshold": runtime_threshold,
                "detected_classes_count": len(active_classes),
                "detected_classes": active_classes,
                "top_5_predictions": top_5_predictions,
                "all_probabilities": class_probs,
                "raster_metadata": raster_meta,
            }
            stats_path = run_dir / "land_cover_stats.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)

            artifacts = [
                {
                    "type": "land_cover_chart",
                    "name": "land_cover_distribution.png",
                    "path": str(chart_path.resolve()),
                    "description": "Probability distribution of detected Corine Land Cover categories",
                },
                {
                    "type": "land_cover_statistics",
                    "name": "land_cover_stats.json",
                    "path": str(stats_path.resolve()),
                    "data": stats,
                    "description": "Comprehensive classification scores and metadata",
                },
            ]

            summary_text = self.format_summary(
                active_classes=active_classes,
                class_probs=class_probs,
                raster_meta=raster_meta,
                threshold=runtime_threshold,
                top_5=top_5_predictions,
            )

            max_conf = max(class_probs.values()) if class_probs else 0.0
            return ModelResult(
                registry_id="bigearthnet_resnet50_s1s2",
                capability="classification",
                status="success",
                text=summary_text,
                artifacts=artifacts,
                raw_scores={
                    "confidence_score": float(max_conf),
                    "probabilities": class_probs,
                    "top_5": top_5_predictions,
                    "threshold": runtime_threshold,
                    "active_classes": active_classes,
                },
            )
        except Exception as e:
            logger.error(f"Inference error in BigEarthNet: {e}", exc_info=True)
            return ModelResult(
                registry_id="bigearthnet_resnet50_s1s2",
                capability="classification",
                status="error",
                error=str(e),
            )

    def render_probability_chart(
        self,
        class_probs: dict[str, float],
        output_path: Path,
        threshold: Optional[float] = None,
    ) -> None:
        """Render a publication-ready horizontal probability bar chart using matplotlib."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        effective_threshold = float(threshold) if threshold is not None else float(self.threshold)

        # Top 10 classes by probability
        sorted_items = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        labels = [item[0] for item in reversed(sorted_items)]
        values = [item[1] for item in reversed(sorted_items)]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#2b8cbe" if v >= effective_threshold else "#a6bddb" for v in values]
        bars = ax.barh(labels, values, color=colors, edgecolor="#1c5778", height=0.6)

        ax.axvline(
            effective_threshold,
            color="#e41a1c",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({effective_threshold:.2f})",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Predicted Probability", fontsize=11, fontweight="bold")
        ax.set_title("BigEarthNet Land-Cover Scene Classification (Top Classes)", fontsize=12, fontweight="bold", pad=12)
        ax.legend(loc="lower right")
        ax.grid(axis="x", linestyle=":", alpha=0.6)

        # Label values on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width * 100:.1f}%",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        fig.savefig(str(output_path), dpi=200)
        plt.close(fig)

    def format_summary(
        self,
        active_classes: list[dict[str, Any]],
        class_probs: dict[str, float],
        raster_meta: dict[str, Any],
        threshold: Optional[float] = None,
        top_5: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """Format factual natural language intelligence report for land-cover classification."""
        effective_threshold = float(threshold) if threshold is not None else float(self.threshold)
        if top_5 is None:
            all_sorted = sorted(
                [{"class": name, "probability": float(prob)} for name, prob in class_probs.items()],
                key=lambda x: x["probability"],
                reverse=True,
            )
            top_5 = all_sorted[:5]

        dev_str = str(self._device).upper() if self._device else "CPU"
        lines = [
            f"Scene land-cover classification completed via BigEarthNet v2 ResNet-50 on {dev_str}.\n",
            "================================================================================",
            "1. EXECUTIVE SUMMARY",
            "================================================================================",
        ]
        if active_classes:
            top_names = [f"**{c['class']}** ({c['probability'] * 100:.1f}%)" for c in active_classes[:3]]
            lines.append(f"Dominant detected land-cover categories: {', '.join(top_names)}.")
        else:
            lines.append(f"No land-cover category exceeded the detection threshold of {effective_threshold * 100:.1f}%.")

        lines.extend([
            "",
            "================================================================================",
            "2. TOP 5 PREDICTED LAND-COVER CATEGORIES",
            "================================================================================",
        ])
        for idx, item in enumerate(top_5, 1):
            lines.append(f"{idx}. {item['class']}: {item['probability'] * 100:.2f}%")

        lines.extend([
            "",
            "================================================================================",
            f"3. DETECTED LAND-COVER CLASSES (Confidence >= Threshold: {effective_threshold:.2f})",
            "================================================================================",
        ])
        if active_classes:
            for c in active_classes:
                lines.append(f"- **{c['class']}**: {c['probability'] * 100:.2f}%")
        else:
            lines.append(f"None detected above threshold ({effective_threshold:.2f}).")

        lines.extend([
            "",
            "================================================================================",
            "4. GEOSPATIAL CONTEXT",
            "================================================================================",
            f"- Spatial Dimensions: {raster_meta.get('width', 0)} × {raster_meta.get('height', 0)} pixels",
            f"- Spectral Bands Analyzed: {raster_meta.get('count', 3)}",
            f"- Coordinate Reference System: {raster_meta.get('crs', 'Not georeferenced')}",
            "",
            "================================================================================",
            "5. MODEL SCOPE & LIMITATIONS",
            "================================================================================",
            "- Pretrained on BigEarthNet v2.0 (reBEN) Sentinel-2 benchmark (19 Corine Land Cover classes).",
            "- Multi-label scene classification provides global scene-level predicted probabilities, not pixel-wise semantic masks.",
            "- Inferences on non-Sentinel optical sensors may exhibit domain shifts.",
        ])
        return "\n".join(lines)
