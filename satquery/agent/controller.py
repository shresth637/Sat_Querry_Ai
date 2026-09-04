from pathlib import Path
from typing import Any, Optional

from satquery.agent.router import route_query
from satquery.confidence.estimator import build_confidence_report
from satquery.domain.schemas import (
    AnalysisResult,
    Evidence,
    InputMode,
    RasterMeta,
    SlotAssignment,
    ValidationStatus,
)
from satquery.evidence.builder import (
    create_metadata_evidence,
    create_preview_evidence,
    create_source_image_evidence,
)
from satquery.preprocess.raster import (
    assess_image_quality,
    inspect_raster,
)
from satquery.registry.models import ModelRegistry
from satquery.registry.tools import ToolRegistry
from satquery.trace.tracer import Tracer
from satquery.validation.validator import validate_input


class AgentController:
    """The central agentic controller for SatQuery AI.

    Orchestrates:
    Input Inspection -> Quality -> Validation -> Query Routing ->
    Specialist Selection -> Tool Execution -> Evidence -> Confidence -> Output.
    """

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.model_registry = model_registry or ModelRegistry.from_yaml("config/models.yaml")
        self.tool_registry = tool_registry or ToolRegistry.default()

    def analyze(
        self,
        query: str,
        slots: list[SlotAssignment],
        input_mode: InputMode,
        allow_png_jpeg: bool = False,
    ) -> AnalysisResult:
        tracer = Tracer()
        uncertainties: list[str] = []
        evidence_items: list[Evidence] = []
        metas: dict[str, RasterMeta] = {}
        qualities: dict = {}

        # 1. Inspect inputs
        with tracer.span(
            step="inspect_inputs",
            component="preprocessor",
            parameters={"slots_count": len(slots), "input_mode": input_mode.value},
        ):
            for slot in slots:
                meta = inspect_raster(slot.file_path)
                metas[slot.slot_id] = meta
                if not meta.crs:
                    uncertainties.append(f"Slot '{slot.slot_id}' lacks CRS georeferencing.")
                if meta.is_valid:
                    evidence_items.append(
                        create_source_image_evidence(
                            file_path=slot.file_path,
                            title=f"Source Raster ({slot.slot_id})",
                            crs=meta.crs,
                        )
                    )
                    evidence_items.append(
                        create_metadata_evidence(
                            meta=meta,
                            title=f"Metadata ({slot.slot_id})",
                        )
                    )

        # 2. Quality assessment
        with tracer.span(step="assess_quality", component="quality_analyzer"):
            for slot in slots:
                q = assess_image_quality(slot.file_path, metas.get(slot.slot_id))
                qualities[slot.slot_id] = q
                if q.level.value in ["LOW", "INSUFFICIENT"]:
                    uncertainties.append(
                        f"Slot '{slot.slot_id}' has {q.level.value} quality: {'; '.join(q.reasons)}"
                    )

        # 3. Input validation
        with tracer.span(step="validate_compatibility", component="validator"):
            validation = validate_input(
                slots=slots,
                metas=metas,
                input_mode=input_mode,
                allow_png_jpeg_benchmark=allow_png_jpeg,
            )
            if validation.spatial_compatibility and validation.spatial_compatibility.value != "CONFIRMED":
                uncertainties.append(f"Spatial co-registration: {validation.spatial_compatibility.value}")

        # 4. Query interpretation and task selection
        with tracer.span(step="interpret_query", component="query_router", parameters={"query": query}):
            plan = route_query(
                query=query,
                slots=slots,
                metas=metas,
                input_mode=input_mode,
                validation=validation,
            )

        # 5. Model & tool resolution
        with tracer.span(step="select_specialists", component="registry"):
            selected_adapters = []
            for mid in plan.selected_models:
                adapter = self.model_registry.get(mid)
                if adapter:
                    selected_adapters.append(adapter)
                else:
                    uncertainties.append(f"Model '{mid}' not found in registry.")

        # 6. Execution
        result_text = ""
        with tracer.span(step="execute_specialists", component="executor"):
            if plan.blocked:
                result_text = (
                    f"Agent planning stopped: {plan.block_reason}\n\n"
                    f"Please check the required image slots for this task."
                )
            elif validation.status == ValidationStatus.FAIL:
                blocking_msgs = [c.message for c in validation.checks if c.is_blocking]
                result_text = (
                    f"Validation failed:\n"
                    + "\n".join(f"- {msg}" for msg in blocking_msgs)
                )
            else:
                # Specialist models are currently unconfigured in Phase 2
                unconfigured_models = [a.name for a in selected_adapters if a.status == "not_configured"]
                if unconfigured_models:
                    result_text = (
                        "Specialist model not configured.\n\n"
                        "The SatQuery agent successfully identified the required specialist workflow, "
                        "but the specialist model is not currently configured."
                    )
                    uncertainties.append(
                        f"Specialist model(s) {unconfigured_models} not configured on local system."
                    )
                else:
                    # If models are configured in later phases, predict here
                    result_text = "Analysis executed."

        # 7. Integrate evidence
        with tracer.span(step="integrate_evidence", component="evidence_integrator"):
            pass

        # 8. Confidence estimation
        with tracer.span(step="estimate_confidence", component="confidence_estimator"):
            confidence = build_confidence_report(
                score=None,
                source="not_available",
                reason="The selected specialist does not currently provide a calibrated confidence score.",
            )

        return AnalysisResult(
            query=query,
            plan=plan,
            result_text=result_text,
            confidence=confidence,
            evidence=evidence_items,
            trace=tracer.get_events(),
            validation=validation,
            metas=metas,
            qualities=qualities,
            uncertainties=uncertainties,
        )
