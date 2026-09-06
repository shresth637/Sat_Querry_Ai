from pathlib import Path
from typing import Any, Optional

from satquery.agent.router import QueryPlanner, route_query
from satquery.confidence.estimator import build_confidence_report
from satquery.domain.schemas import (
    AgentPlan,
    AnalysisResult,
    Evidence,
    InputMode,
    RasterMeta,
    SlotAssignment,
    TaskType,
    ValidationReport,
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
        planner: Optional[QueryPlanner] = None,
    ) -> None:
        self.model_registry = model_registry or ModelRegistry.from_yaml("config/models.yaml")
        self.tool_registry = tool_registry or ToolRegistry.default()
        self.planner = planner or QueryPlanner()

    def plan(
        self,
        query: str,
        slots: list[SlotAssignment],
        input_mode: InputMode,
        metas: Optional[dict[str, RasterMeta]] = None,
        validation: Optional[ValidationReport] = None,
    ) -> AgentPlan:
        """Route natural language query and input slots to an AgentPlan."""
        return self.planner.plan(
            query=query,
            slots=slots,
            input_mode=input_mode,
            metas=metas,
            validation=validation,
        )


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
            plan = self.planner.plan(
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
                # Find available ready or unloaded model adapters matching required capability for this plan
                def _adapter_matches_task(adapter, task):
                    if task == TaskType.OPTICAL_SAR_ANALYSIS.value:
                        return "optical_sar_fusion" in adapter.capabilities
                    if task == TaskType.BI_TEMPORAL_CHANGE.value:
                        return "change_detect" in adapter.capabilities
                    if task == TaskType.CLASSIFICATION.value:
                        return "classification" in adapter.capabilities
                    if task == TaskType.CHANGE_VQA.value:
                        return "change_vqa" in adapter.capabilities
                    return True

                matching_adapters = [a for a in selected_adapters if _adapter_matches_task(a, plan.task)]
                available_adapters = [a for a in matching_adapters if a.status in ["ready", "unloaded"]]
                unconfigured_models = [
                    a.name for a in selected_adapters
                    if a.status == "not_configured" or not _adapter_matches_task(a, plan.task)
                ]

                active_adapter = None
                if available_adapters:
                    from satquery.models.manager import get_resource_manager
                    rm = get_resource_manager()
                    model_id = plan.selected_models[0] if plan.selected_models else "specialist"
                    target_adapter = available_adapters[0]
                    active_adapter = rm.acquire_model(
                        model_id=model_id,
                        adapter=target_adapter,
                        is_large=(target_adapter.version == "7B"),
                    )

                    t0_slot = next((s for s in slots if s.slot_id == "t0"), slots[0] if len(slots) > 0 else None)
                    t1_slot = next((s for s in slots if s.slot_id == "t1"), slots[1] if len(slots) > 1 else None)
                    primary_image = slots[0].file_path if slots else None

                    inputs_dict = {
                        "image": primary_image,
                        "query": query,
                        "t0": t0_slot.file_path if t0_slot else None,
                        "t1": t1_slot.file_path if t1_slot else None,
                        "slots": slots,
                        "metas": metas,
                    }

                    model_result = active_adapter.predict(inputs_dict, model_id=model_id, **kwargs)
                    if model_result.status == "success":
                        result_text = model_result.text or "Specialist analysis completed."
                    elif model_result.status == "not_configured":
                        result_text = (
                            "Specialist model not configured.\n\n"
                            "The SatQuery agent successfully identified the required specialist workflow, "
                            "but the specialist model is not currently configured."
                        )
                        uncertainties.append(f"Specialist model {active_adapter.name} not configured.")
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
                    elif art_type in [
                        "change_visualization",
                        "change_regions_visualization",
                        "land_cover_chart",
                        "grounding_visualization",
                    ]:
                        title_map = {
                            "change_visualization": "Change Map Visual Overlay",
                            "change_regions_visualization": "Change Regions with Bounding Boxes",
                            "land_cover_chart": "Land-Cover Probability Distribution",
                            "grounding_visualization": "Spatial Grounding Visual Overlay",
                        }
                        evidence_items.append(
                            create_preview_evidence(
                                file_path=artifact.get("path"),
                                title=title_map.get(art_type, "Visual Output"),
                                description=artifact.get("description", "Generated visual artifact"),
                            )
                        )
                    elif art_type in ["change_regions_geojson", "grounding_geojson"]:
                        from satquery.domain.schemas import Evidence, EvidenceType
                        title_map = {
                            "change_regions_geojson": "Change Regions Vector Map (GeoJSON)",
                            "grounding_geojson": "Spatial Grounding Vector Map (GeoJSON)",
                        }
                        evidence_items.append(
                            Evidence(
                                evidence_type=EvidenceType.CHANGE_MAP if "change" in art_type else EvidenceType.BOUNDING_BOX,
                                title=title_map.get(art_type, "Vector Map (GeoJSON)"),
                                file_path=artifact.get("path"),
                                data=artifact.get("data"),
                                crs=artifact.get("crs"),
                                description=artifact.get("description", "RFC 7946 GeoJSON FeatureCollection"),
                            )
                        )
                    elif art_type in ["change_statistics", "land_cover_statistics"]:
                        evidence_items.append(
                            create_statistics_evidence(
                                stats=artifact.get("data", {}),
                                title="Bi-Temporal Change Statistics" if "change" in art_type else "Land-Cover Classification Statistics",
                            )
                        )

        # 8. Confidence estimation
        with tracer.span(step="estimate_confidence", component="confidence_estimator"):
            if model_result and model_result.status == "success" and model_result.raw_scores and model_result.raw_scores.get("confidence_score") is not None:
                adapter_name_lower = str(getattr(active_adapter, "name", "")).lower()
                if "bit" in adapter_name_lower or "change" in adapter_name_lower:
                    method = "bit_softmax_mean_confidence"
                    signals = ["softmax_probabilities", "prediction_margin", "sliding_window_probabilities"]
                    reason = "Mean prediction confidence derived from Open-CD BIT output layer across all sliding window tiles"
                elif "bigearthnet" in adapter_name_lower:
                    method = "bigearthnet_sigmoid_max_confidence"
                    signals = ["sigmoid_probabilities", "class_threshold_margin"]
                    reason = "Highest class confidence score derived from BigEarthNet v2 ResNet-50 output layer"
                else:
                    method = "model_confidence_score"
                    signals = ["raw_model_confidence"]
                    reason = f"Confidence score derived from {active_adapter.name}"

                confidence = build_confidence_report(
                    score=model_result.raw_scores["confidence_score"],
                    source="model",
                    method=method,
                    signals_used=signals,
                    reason=reason,
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
