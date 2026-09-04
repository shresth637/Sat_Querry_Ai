import re
from typing import Optional

from satquery.domain.schemas import (
    AgentPlan,
    InputMode,
    Modality,
    QueryTheme,
    RasterMeta,
    SlotAssignment,
    TaskType,
    ValidationReport,
)


THEME_KEYWORDS: dict[QueryTheme, list[str]] = {
    QueryTheme.DISASTER: [
        "flood", "fire", "landslide", "damage", "disaster", "earthquake",
        "cyclone", "hurricane", "storm", "wildfire", "tsunami", "destruction",
    ],
    QueryTheme.AGRICULTURE: [
        "crop", "agriculture", "farmland", "farm", "field", "harvest",
        "plantation", "paddy", "cultivation", "agri",
    ],
    QueryTheme.INFRASTRUCTURE: [
        "bridge", "railway", "infrastructure", "highway", "airport",
        "runway", "port", "road", "rail", "dam",
    ],
    QueryTheme.URBAN: [
        "building", "built-up", "built up", "city", "expansion", "urban",
        "settlement", "residential", "downtown", "suburb", "houses",
    ],
    QueryTheme.ENVIRONMENT: [
        "forest", "water", "lake", "river", "deforestation", "wetland",
        "mangrove", "tree", "ocean", "vegetation", "canopy",
    ],
}


def classify_theme(query: str) -> QueryTheme:
    """Classify the query into a high-level thematic domain.

    Used for routing/context metadata only. Does not generate answers.
    """
    q_lower = query.lower()

    # Score each theme by matching keywords
    for theme, keywords in THEME_KEYWORDS.items():
        for kw in keywords:
            # Word boundary or exact phrase match
            if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
                return theme

    return QueryTheme.GENERAL


def route_query(
    query: str,
    slots: list[SlotAssignment],
    metas: dict[str, RasterMeta],
    input_mode: InputMode,
    validation: Optional[ValidationReport] = None,
) -> AgentPlan:
    """Deterministically route natural language query and input configuration to an AgentPlan.

    Never routes temporal queries to BI_TEMPORAL_CHANGE if only 1 image is provided.
    Never routes optical-SAR queries without both modalities.
    """
    q_lower = query.lower().strip()
    theme = classify_theme(query)
    num_slots = len(slots)

    # 1. Optical-SAR joint analysis patterns
    optical_sar_phrases = [
        "optical and sar", "sar and optical", "optical & sar",
        "optical and radar", "radar and optical",
        "together to identify built-up and water",
        "use the optical and sar", "optical + sar",
    ]
    is_optical_sar_query = any(phrase in q_lower for phrase in optical_sar_phrases)

    # 2. Change VQA patterns (interrogative question about change/increase/decrease)
    change_vqa_phrases = [
        "has the built-up area increased", "increased, decreased",
        "increase or decrease", "increased or decreased",
        "growth rate", "did the water increase", "has the area changed",
        "has built-up increased", "did change occur",
    ]
    is_change_vqa_query = any(phrase in q_lower for phrase in change_vqa_phrases)

    # 3. Bi-temporal change detection patterns
    change_detect_phrases = [
        "what changed", "what changed between", "where did the change occur", "where did change occur",
        "detect change", "change between these two", "changed between",
        "differences between", "difference between these two",
        "find changes", "map changes", "change occurred",
        "where are the changed buildings", "where are the changed", "changed buildings",
        "show me areas where construction happened", "construction happened", "areas where construction",
        "how much of the area changed", "how much changed",
        "compare these two satellite images", "compare these two", "compare the images",
        "compare images", "bitemporal change", "temporal change",
    ]
    is_change_detect_query = any(phrase in q_lower for phrase in change_detect_phrases)

    # If bitemporal input mode is selected, check for general change/difference/comparison concepts
    if input_mode == InputMode.I4_BITEMPORAL_PAIR and not is_change_detect_query and not is_change_vqa_query:
        bitemporal_keywords = [
            "change", "changed", "construction", "difference", "differences",
            "compare", "growth", "expansion", "before and after", "new buildings",
            "development", "modification",
        ]
        if any(re.search(r"\b" + re.escape(kw) + r"\b", q_lower) for kw in bitemporal_keywords):
            is_change_detect_query = True

    # 4. Grounding patterns
    grounding_phrases = [
        "highlight", "locate", "detect the", "bounding box", "find the",
        "pinpoint", "where is the", "outline", "box the",
    ]
    # Do not treat "where are the changed buildings" as grounding when change detect is matched
    is_grounding_query = any(phrase in q_lower for phrase in grounding_phrases) and not is_change_detect_query

    # 5. Captioning patterns
    caption_phrases = [
        "describe", "caption", "overview of", "summary of the scene",
        "what does this image show", "describe the land-cover",
        "describe the scene", "generate a caption",
        "what is visible in this satellite image", "what is visible in this",
        "what is visible",
    ]
    is_caption_query = any(phrase in q_lower for phrase in caption_phrases)

    warnings: list[str] = []
    blocked = False
    block_reason = None

    # Routing decision logic
    if is_optical_sar_query or input_mode == InputMode.I3_OPTICAL_SAR_PAIR:
        task = TaskType.OPTICAL_SAR_ANALYSIS
        required_inputs = ["optical", "sar"]

        # Check if both modalities are actually present
        detected_mods = []
        for s in slots:
            m = metas.get(s.slot_id)
            if s.declared_modality:
                detected_mods.append(s.declared_modality)
            elif m:
                from satquery.preprocess.raster import detect_modality
                detected_mods.append(detect_modality(m))

        has_optical = any(m in [Modality.OPTICAL, Modality.MULTISPECTRAL] for m in detected_mods)
        has_sar = any(m == Modality.SAR for m in detected_mods)

        if num_slots < 2 or not has_optical or not has_sar:
            blocked = True
            block_reason = (
                f"Optical-SAR analysis requires both an optical and a SAR image, but received "
                f"{num_slots} image(s) with detected modalities: {[m.value for m in detected_mods]}."
            )

        selected_models = ["bigearthnet_resnet50_s1s2"]
        selected_tools = [
            "raster_inspection", "metadata_extraction", "validation",
            "visualization", "spatial_statistics",
        ]

    elif is_change_vqa_query:
        if num_slots < 2:
            # Temporal query with 1 image MUST NOT be routed to temporal tasks
            task = TaskType.SINGLE_VQA
            blocked = True
            block_reason = (
                "Change VQA query requires two images from different dates (Before and After), "
                "but only 1 image was provided."
            )
            required_inputs = ["image"]
            selected_models = ["geochat_vqa"]
            selected_tools = ["raster_inspection", "metadata_extraction", "image_quality"]
        else:
            task = TaskType.CHANGE_VQA
            required_inputs = ["t0", "t1"]
            selected_models = ["cdvqa_baseline", "opencd_bit_change"]
            selected_tools = [
                "raster_inspection", "metadata_extraction", "validation",
                "change_map", "spatial_statistics",
            ]

    elif is_change_detect_query:
        if num_slots < 2:
            # Temporal query with 1 image MUST NOT be routed to BI_TEMPORAL_CHANGE
            task = TaskType.SINGLE_VQA
            blocked = True
            block_reason = (
                "Temporal change query requires two images from different dates (Before and After), "
                "but only 1 image was provided. Cannot route to BI_TEMPORAL_CHANGE."
            )
            required_inputs = ["image"]
            selected_models = ["geochat_vqa"]
            selected_tools = ["raster_inspection", "metadata_extraction", "image_quality"]
        else:
            task = TaskType.BI_TEMPORAL_CHANGE
            required_inputs = ["t0", "t1"]
            selected_models = ["opencd_bit_change"]
            selected_tools = [
                "raster_inspection", "metadata_extraction", "validation",
                "change_map", "spatial_statistics",
            ]

    elif is_grounding_query:
        task = TaskType.SINGLE_GROUNDING
        required_inputs = ["image"]
        selected_models = ["geochat_grounding"]
        selected_tools = ["raster_inspection", "metadata_extraction", "image_quality"]

    elif is_caption_query:
        task = TaskType.SINGLE_CAPTION
        required_inputs = ["image"]
        selected_models = ["geochat_caption"]
        selected_tools = ["raster_inspection", "metadata_extraction", "image_quality"]

    else:
        # Default single-image VQA
        task = TaskType.SINGLE_VQA
        required_inputs = ["image"]
        selected_models = ["geochat_vqa"]
        selected_tools = ["raster_inspection", "metadata_extraction", "image_quality"]

    # Parameter extraction
    params = {
        "query_length": len(query),
        "theme": theme.value,
        "input_count": num_slots,
    }

    return AgentPlan(
        task=task.value,
        required_inputs=required_inputs,
        input_mode=input_mode,
        selected_models=selected_models,
        selected_tools=selected_tools,
        permitted_parameters=params,
        warnings=warnings,
        blocked=blocked,
        block_reason=block_reason,
        theme=theme,
    )
