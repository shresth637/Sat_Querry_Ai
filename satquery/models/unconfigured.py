from typing import Any
from satquery.domain.schemas import ModelResult
from satquery.models.base import ModelAdapter


class UnconfiguredSpecialistAdapter(ModelAdapter):
    """Placeholder adapter for unconfigured or disabled specialist models.

    Never downloads weights or fakes model outputs.
    """

    def __init__(
        self,
        name: str,
        version: str,
        capabilities: list[str],
        status: str = "not_configured",
        notes: str = "Model weights not present on disk",
    ):
        super().__init__(name=name, version=version, capabilities=capabilities, status=status)
        self.notes = notes

    def load(self) -> None:
        # Intentionally no-op to prevent auto-downloading large weights at startup
        self.status = "not_configured"

    def predict(self, inputs: dict[str, Any], **kwargs: Any) -> ModelResult:
        primary_cap = self.capabilities[0] if self.capabilities else "unknown"
        return ModelResult(
            registry_id=self.name,
            capability=primary_cap,
            status="not_configured",
            text=None,
            artifacts=[],
            raw_scores=None,
            error=f"Model '{self.name}' is not configured: {self.notes}",
        )


class UnconfiguredVQAModel(UnconfiguredSpecialistAdapter):
    def __init__(self, **kwargs: Any):
        super().__init__(
            name="GeoChat-7B",
            version="7B",
            capabilities=["vqa"],
            notes="VQA weights not downloaded. Deferred to Phase 3.",
        )


class UnconfiguredCaptioningModel(UnconfiguredSpecialistAdapter):
    def __init__(self, **kwargs: Any):
        super().__init__(
            name="GeoChat-7B",
            version="7B",
            capabilities=["caption"],
            notes="Captioning weights not downloaded. Deferred to Phase 3.",
        )


class UnconfiguredGroundingModel(UnconfiguredSpecialistAdapter):
    def __init__(self, **kwargs: Any):
        super().__init__(
            name="GeoChat-7B",
            version="7B",
            capabilities=["grounding"],
            notes="Grounding weights not downloaded. Deferred to Phase 3.",
        )


class UnconfiguredChangeDetectionModel(UnconfiguredSpecialistAdapter):
    def __init__(self, **kwargs: Any):
        super().__init__(
            name="Open-CD BIT ResNet-18",
            version="r18-levir",
            capabilities=["change_detect"],
            notes="Change detection checkpoint not downloaded. Deferred to Phase 3.",
        )


class UnconfiguredChangeVQAModel(UnconfiguredSpecialistAdapter):
    def __init__(self, **kwargs: Any):
        super().__init__(
            name="CDVQA baseline",
            version="paper-2022",
            capabilities=["change_vqa"],
            notes="Change VQA weights not present. Deferred to Phase 3.",
        )


class UnconfiguredOpticalSARModel(UnconfiguredSpecialistAdapter):
    def __init__(self, **kwargs: Any):
        super().__init__(
            name="BigEarthNet v2 ResNet-50 S1+S2",
            version="v0.2.0",
            capabilities=["optical_sar_fusion", "land_cover"],
            notes="BigEarthNet weights not present. Deferred to Phase 3.",
        )
