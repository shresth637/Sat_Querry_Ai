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


class BigEarthNetClassifier(nn.Module):
    """ResNet-50 backbone with 19-class multi-label output head."""

    def __init__(self, num_classes: int = 19):
        super().__init__()
        backbone = models.resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logits
        return self.model(x)


class BigEarthNetModel(ModelAdapter):
    """BigEarthNet v2 (reBEN) ResNet-50 multi-label land-cover classification adapter."""

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
            capabilities=["classification"],
            status="unloaded",
        )
        self.weights_path = Path(weights_path) if weights_path else None
        self.threshold = float(threshold)
        self.device_str = device
        self.output_dir = Path(output_dir)
        self.model: Optional[nn.Module] = None
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
            self.model = BigEarthNetClassifier(num_classes=len(BIGEARTHNET_19_CLASSES))
            checkpoint = torch.load(str(self.weights_path), map_location=self._device)

            state_dict = checkpoint.get("state_dict", checkpoint)
            # Remove any prefix like 'module.' or 'model.' if present
            cleaned_state = {}
            for k, v in state_dict.items():
                clean_k = k.replace("module.", "").replace("backbone.", "")
                if not clean_k.startswith("model."):
                    clean_k = f"model.{clean_k}"
                cleaned_state[clean_k] = v

            self.model.load_state_dict(cleaned_state, strict=False)
            self.model.float()
            self.model.to(self._device)
            self.model.eval()
            self.status = "ready"
            logger.info(f"BigEarthNet model loaded successfully on {self._device}.")
        except Exception as e:
            self.status = "error"
            logger.error(f"Failed to load BigEarthNet checkpoint: {e}")

    def unload(self) -> None:
        """Release model from memory."""
        self.model = None
        if self.weights_path and self.weights_path.exists():
            self.status = "unloaded"
        else:
            self.status = "not_configured"

    def preprocess(self, file_path: str) -> tuple[torch.Tensor, dict[str, Any]]:
        """Read raster, normalize to RGB channels and remote sensing standard scale."""
        with rasterio.open(file_path) as src:
            meta = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs": str(src.crs) if src.crs else None,
                "transform": [float(x) for x in src.transform][:6],
            }
            if src.count >= 3:
                # Read 3 channels
                arr = src.read([1, 2, 3]).astype(np.float32)
            else:
                # Replicate single band to 3 channels
                band = src.read(1).astype(np.float32)
                arr = np.stack([band, band, band], axis=0)

        # Robust min-max normalization to [0, 1]
        p2, p98 = np.percentile(arr, (2, 98))
        if p98 > p2:
            arr = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0).astype(np.float32)
        else:
            arr = np.clip(arr / 255.0, 0.0, 1.0).astype(np.float32)

        # Standard ImageNet / visual normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        norm_arr = ((arr - mean) / std).astype(np.float32)

        # Resize to 224x224 using torch interpolate
        tensor = torch.from_numpy(norm_arr).float().unsqueeze(0)
        tensor = nn.functional.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        return tensor, meta

    def predict(self, inputs: dict[str, Any], **kwargs: Any) -> ModelResult:
        """Run multi-label classification on optical satellite scene."""
        if self.status == "not_configured":
            return ModelResult(
                registry_id="bigearthnet_resnet50_s2",
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
                    registry_id="bigearthnet_resnet50_s2",
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
                registry_id="bigearthnet_resnet50_s2",
                capability="classification",
                status="error",
                error=f"Valid input raster not provided: {image_path}",
            )

        try:
            tensor, raster_meta = self.preprocess(str(image_path))
            tensor = tensor.to(self._device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            class_probs = {
                BIGEARTHNET_19_CLASSES[i]: float(probs[i])
                for i in range(len(BIGEARTHNET_19_CLASSES))
            }

            # Filter by threshold and sort descending
            active_classes = [
                {"class": name, "probability": prob}
                for name, prob in class_probs.items()
                if prob >= self.threshold
            ]
            active_classes.sort(key=lambda x: x["probability"], reverse=True)

            # Isolated run directory
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
            run_dir = self.output_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            # Generate horizontal bar chart visualization
            chart_path = run_dir / "land_cover_distribution.png"
            self.render_probability_chart(class_probs, chart_path)

            # Save stats JSON
            stats = {
                "model": "BigEarthNet v2 ResNet-50 S2",
                "threshold": self.threshold,
                "detected_classes_count": len(active_classes),
                "detected_classes": active_classes,
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

            summary_text = self.format_summary(active_classes, class_probs, raster_meta)

            max_conf = max(class_probs.values()) if class_probs else 0.0
            return ModelResult(
                registry_id="bigearthnet_resnet50_s2",
                capability="classification",
                status="success",
                text=summary_text,
                artifacts=artifacts,
                raw_scores={"confidence_score": float(max_conf), "probabilities": class_probs},
            )
        except Exception as e:
            logger.error(f"Inference error in BigEarthNet: {e}", exc_info=True)
            return ModelResult(
                registry_id="bigearthnet_resnet50_s2",
                capability="classification",
                status="error",
                error=str(e),
            )

    def render_probability_chart(self, class_probs: dict[str, float], output_path: Path) -> None:
        """Render a publication-ready horizontal probability bar chart using matplotlib."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Top 10 classes by probability
        sorted_items = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        labels = [item[0] for item in reversed(sorted_items)]
        values = [item[1] for item in reversed(sorted_items)]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#2b8cbe" if v >= self.threshold else "#a6bddb" for v in values]
        bars = ax.barh(labels, values, color=colors, edgecolor="#1c5778", height=0.6)

        ax.axvline(self.threshold, color="#e41a1c", linestyle="--", linewidth=1.5, label=f"Threshold ({self.threshold})")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Probability", fontsize=11, fontweight="bold")
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
    ) -> str:
        """Format factual natural language intelligence report for land-cover classification."""
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
            lines.append(f"No land-cover category exceeded the detection threshold of {self.threshold * 100:.0f}%.")

        lines.extend([
            "",
            "================================================================================",
            "2. DETECTED LAND-COVER CLASSES (Confidence >= Threshold)",
            "================================================================================",
        ])
        if active_classes:
            for c in active_classes:
                lines.append(f"- **{c['class']}**: {c['probability'] * 100:.2f}%")
        else:
            lines.append("None detected above threshold.")

        lines.extend([
            "",
            "================================================================================",
            "3. GEOSPATIAL CONTEXT",
            "================================================================================",
            f"- Spatial Dimensions: {raster_meta.get('width', 0)} × {raster_meta.get('height', 0)} pixels",
            f"- Spectral Bands Analyzed: {raster_meta.get('count', 3)}",
            f"- Coordinate Reference System: {raster_meta.get('crs', 'Not georeferenced')}",
            "",
            "================================================================================",
            "4. MODEL SCOPE & LIMITATIONS",
            "================================================================================",
            "- Pretrained on BigEarthNet v2.0 (reBEN) Sentinel-2 benchmark (19 Corine Land Cover classes).",
            "- Multi-label scene classification provides global scene-level probabilities, not pixel-wise semantic masks.",
            "- Inferences on non-Sentinel optical sensors may exhibit domain shifts.",
        ])
        return "\n".join(lines)
