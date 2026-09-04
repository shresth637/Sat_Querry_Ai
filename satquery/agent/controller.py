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
        **kwargs: Any,
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
        model_result: Optional[Any] = None

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
                # Find available ready model adapters for this plan
                ready_adapters = [a for a in selected_adapters if a.status == "ready"]
                unconfigured_models = [a.name for a in selected_adapters if a.status == "not_configured"]

                if ready_adapters:
                    # Execute first ready specialist (e.g. Open-CD BIT)
                    active_adapter = ready_adapters[0]
                    t0_slot = next((s for s in slots if s.slot_id == "t0"), slots[0] if len(slots) > 0 else None)
                    t1_slot = next((s for s in slots if s.slot_id == "t1"), slots[1] if len(slots) > 1 else None)

                    inputs_dict = {
                        "t0": t0_slot.file_path if t0_slot else None,
                        "t1": t1_slot.file_path if t1_slot else None,
                        "slots": slots,
                        "metas": metas,
                    }

                    model_result = active_adapter.predict(inputs_dict, **kwargs)
                    if model_result.status == "success":
                        result_text = model_result.text or "Change detection analysis completed."
                    else:
                        result_text = f"Specialist inference failed: {model_result.error}"
                        uncertainties.append(f"Inference error: {model_result.error}")
                elif unconfigured_models:
                    result_text = (
                        "Specialist model not configured.\n\n"
                        "The SatQuery agent successfully identified the required specialist workflow, "
                        "but the specialist model is not currently configured."
                    )
                    uncertainties.append(
                        f"Specialist model(s) {unconfigured_models} not configured on local system."
                    )
                else:
                    result_text = "No compatible specialist model could be resolved."

        # 7. Integrate evidence
        with tracer.span(step="integrate_evidence", component="evidence_integrator"):
            if model_result and model_result.status == "success" and model_result.artifacts:
                from satquery.evidence.builder import (
                    create_change_map_evidence,
                    create_statistics_evidence,
                )
                for artifact in model_result.artifacts:
                    art_type = artifact.get("type")
                    if art_type == "change_mask_geotiff":
                        evidence_items.append(
                            create_change_map_evidence(
                                title="Binary Change Mask (GeoTIFF)",
                                file_path=artifact.get("path"),
                                crs=artifact.get("crs"),
                                description="Georeferenced binary change map preserving original raster bounds and CRS",
                            )
                        )
                    elif art_type == "change_prob_geotiff":
                        evidence_items.append(
                            create_change_map_evidence(
                                title="Change Probability Map (GeoTIFF)",
                                file_path=artifact.get("path"),
                                crs=artifact.get("crs"),
                                description="Continuous float32 change probability map in [0, 1] at native raster resolution",
                            )
                        )
                    elif art_type == "change_visualization":
                        evidence_items.append(
                            create_preview_evidence(
                                file_path=artifact.get("path"),
                                title="Change Map Visual Overlay",
                                description="Visual representation of detected change (red = changed)",
                            )
                        )
                    elif art_type == "change_regions_visualization":
                        evidence_items.append(
                            create_preview_evidence(
                                file_path=artifact.get("path"),
                                title="Change Regions with Bounding Boxes",
                                description="Visual overlay identifying individual contiguous change regions and ranking badges",
                            )
                        )
                    elif art_type == "change_regions_geojson":
                        from satquery.domain.schemas import Evidence, EvidenceType
                        evidence_items.append(
                            Evidence(
                                evidence_type=EvidenceType.CHANGE_MAP,
                                title="Change Regions Vector Map (GeoJSON)",
                                file_path=artifact.get("path"),
                                data=artifact.get("data"),
                                crs=artifact.get("crs"),
                                description="RFC 7946 GeoJSON FeatureCollection of significant connected change regions with bounding boxes and metrics",
                            )
                        )
                    elif art_type == "change_statistics":
                        evidence_items.append(
                            create_statistics_evidence(
                                stats=artifact.get("data", {}),
                                title="Bi-Temporal Change Statistics",
                            )
                        )

        # 8. Confidence estimation
        with tracer.span(step="estimate_confidence", component="confidence_estimator"):
            if model_result and model_result.status == "success" and model_result.raw_scores and model_result.raw_scores.get("confidence_score") is not None:
                confidence = build_confidence_report(
                    score=model_result.raw_scores["confidence_score"],
                    source="model",
                    method="bit_softmax_mean_confidence",
                    signals_used=["softmax_probabilities", "prediction_margin", "sliding_window_probabilities"],
                    reason="Mean prediction confidence derived from Open-CD BIT output layer across all sliding window tiles",
                )
            else:
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
