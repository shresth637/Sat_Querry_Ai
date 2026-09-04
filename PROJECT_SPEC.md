# SatQuery AI — Project Specification

| Field | Value |
| --- | --- |
| Product name | SatQuery AI |
| Problem statement ID | 26167 |
| Title | SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries |
| Document type | Engineering specification |
| Status | Spec only — implementation deferred |
| Audience | Engineering, evaluation, and reviewers |

This document is the source of truth for scope, architecture, inputs, capabilities, GUI, evaluation, and non-negotiable engineering constraints. It does not prescribe a particular model vendor, training recipe, or UI framework.

---

## 1. Problem and product goal

SatQuery AI is an **interactive GUI/web application** with an **agentic remote-sensing AI backend**. Users upload one or more remote-sensing images and ask natural-language questions. The system must interpret the query, validate inputs, route work to **specialized remote-sensing models and tools**, fuse results, estimate confidence from real model/tool outputs, and return text plus **visual evidence**.

The architecture **must not** rely on a generic large language model (LLM) or vision-language model (VLM) alone. Generic language components, if used, are limited to orchestration, query parsing, or report wording. Perception, change detection, optical–SAR fusion, and remote-sensing semantics must come from **replaceable specialist models/tools** behind a common interface.

---

## 2. In-scope / out-of-scope

### 2.1 In scope

- Interactive GUI or web application covering the mandatory UI surfaces in §8.
- Agentic orchestration covering the steps in §7.
- Mandatory input modes in §3.
- Mandatory capabilities in §5.
- Remote-sensing adaptation using an open-source remote-sensing training resource (minimum: BigEarthNet or equivalent) as in §5.1.
- Support for evaluation datasets listed in §9.
- Modular, configuration-driven, typed, tested, logged, and reproducible engineering as in §10.

### 2.2 Out of scope (this document)

- Implementation, scaffolding, model training, dataset download, or UI code.
- Claiming benchmark scores, metadata, or confidence values that were not produced by the running system.
- Hard-coding answers, traces, or maps for representative queries.

---

## 3. Mandatory inputs

The system must accept and process the following **input modes**. A session may provide a subset; the agent must validate that the provided set is compatible with the interpreted task.

| ID | Input mode | Description |
| --- | --- | --- |
| I1 | Single optical / multispectral image | One optical or multispectral scene. |
| I2 | Single SAR image | One synthetic aperture radar scene. |
| I3 | Co-registered optical + SAR pair | Optical/multispectral and SAR images that are spatially aligned (or declared co-registered by the user/pipeline after validation). |
| I4 | Bi-temporal image pair | Two corresponding images from different dates (same or compatible modality as required by the change-analysis tools). |

### 3.1 Supported file formats

| Format | Policy |
| --- | --- |
| GeoTIFF | Required. |
| TIFF | Required. |
| PNG / JPEG | Permitted **only** where allowed for benchmark datasets (for example dataset-provided RGB previews). Not a substitute for geospatial rasters when georeferencing, CRS, or pixel-aligned change maps are required. |

The pipeline must inspect format, raster properties, and (when present) geospatial metadata. Missing georeferencing must be reported as validation status, not invented.

---

## 4. System context

```
User (GUI)
  → Query + image slot(s)
  → Agent (interpret, inspect, validate, plan, fuse, score, explain)
       → Specialist models / tools (common interface, config-selected)
       → Evidence artifacts (overlays, change maps, grounding boxes/masks)
  → GUI (preview, metadata, validation, results, trace, downloadable report)
```

**Layer separation (mandatory):**

| Layer | Responsibility |
| --- | --- |
| UI | Upload, preview, query, display of results and evidence; no model logic. |
| Agent | Query interpretation, input inspection, compatibility validation, task determination, tool/model selection, execution order, fusion, confidence estimation, execution trace. |
| Models | Learnable specialists (VQA, captioning, change detection, optical–SAR, land-cover, etc.). |
| Tools | Deterministic or algorithmic operations (I/O, registration checks, raster math, overlay generation, report packaging). |
| Configuration | Model/tool IDs, weights paths, thresholds, device, dataset roots — not hardcoded in UI or agent control flow. |

---

## 5. Mandatory functionality

### 5.1 Remote-sensing adaptation (M1)

At least one **visual or vision-language** component must be **fine-tuned or otherwise adapted** using `BigEarthNet.txt` or **another open-source remote-sensing training dataset**.

Requirements:

- Adaptation must be real (training, continued pretraining, adapters, or equivalent documented procedure).
- Training data source, split, and procedure must be recorded for reproducibility.
- The adapted component must be used in at least one production path (not a disconnected unused checkpoint).

### 5.2 Single-image visual question answering (M2)

Given a single remote-sensing image and a natural-language question, the system must produce an answer grounded in that image (and in specialist outputs, not generic caption hallucination).

Representative query: *“Describe the land-cover and major objects visible in this image.”* (may overlap captioning; VQA must still handle interrogative queries.)

### 5.3 Additional single-image capability (M3)

Implement **at least one** of:

- Captioning / scene description
- Text-guided region grounding

If feasible, implement both. Grounding, when present, must produce **spatial evidence** (boxes, polygons, or masks) displayable in the GUI.

Representative query: *“Highlight the water body referred to in the query.”*

### 5.4 Bi-temporal change analysis (M4)

Given two corresponding images from different dates, the system must:

1. Detect changes.
2. Describe changes in natural language.
3. Answer natural-language questions about changes.
4. Generate a **spatial change map** where the data and tools allow (pixel- or region-level). If a map cannot be produced, the system must state why (validation or tool limitation), not fabricate a map.

Representative queries:

- *“What changed between these two dates, and where did the change occur?”*
- *“Has the built-up area increased, decreased, or remained unchanged?”*

### 5.5 Optical + SAR analysis (M5)

Given co-registered optical/multispectral and SAR images, the system must:

- Extract complementary information from each modality.
- Perform joint analysis.
- Identify built-up, water, and other relevant regions as supported by the selected specialists.
- Provide **evidence-grounded** results (overlays, masks, or equivalent tied to the rasters).

Representative query: *“Use the optical and SAR images together to identify built-up and water-covered regions.”*

### 5.6 Agentic orchestration (M6)

See §7. This is a first-class capability, not a UI-only “chat wrapper.”

---

## 6. Agent, models, and tools — common interface

Specialists must be invocable through a **shared contract** so models are replaceable without changing the GUI.

Minimum contract expectations (to be refined at implementation):

- Typed input/output schemas (image handles, query text, task type, artifacts).
- Explicit capability tags (e.g. `vqa`, `caption`, `grounding`, `change_detect`, `change_vqa`, `optical_sar_fusion`).
- Structured artifacts: text, confidence **derived from the run**, paths to visual evidence.
- Failure modes that propagate as errors or low-confidence, documented outcomes — never silent fake success.

The agent selects specialists using **configuration** (enabled models, priority, resource constraints) plus input/task compatibility — not hardcoded model names in UI code.

---

## 7. Agentic orchestration requirements

For each user request, the agent must perform the following steps in an **observable** way (execution trace visible in the GUI):

| Step | Requirement |
| --- | --- |
| Interpret query | Parse intent (VQA, caption, grounding, change, optical–SAR, mixed). |
| Inspect inputs | Count, modality, format, size, georeference if present, temporal labels if present. |
| Validate compatibility | Reject or warn when slots, modalities, or co-registration/temporal pairing do not match the task. |
| Determine task | Map to one or more specialist capabilities. |
| Select specialists | Choose models/tools via configuration and capability matching. |
| Execute | Run selected specialists; record actual outputs and errors. |
| Combine outputs | Fuse text and spatial artifacts into a single response. |
| Confidence | Calculate or estimate confidence from model scores, agreement, or other **run-derived** signals. No hardcoded fake scores. |
| Visual evidence | Return overlays, change maps, and/or grounding regions when produced. |
| Execution trace | Expose the steps above to the user (ordered, inspectable). |

---

## 8. GUI requirements

The GUI must support:

| Surface | Requirement |
| --- | --- |
| Image upload | Single-file upload. |
| Multiple image upload | Multiple files for pairs (bi-temporal and optical + SAR). |
| Query input | Natural-language query. |
| Image preview | Preview of uploaded raster(s) (or a defined preview derivative). |
| Metadata display | Real file/raster metadata only (format, size, bands, CRS/transform if present). No fake metadata. |
| Validation status | Pass / warn / fail from inspection (format, pairing, co-registration checks as implemented). |
| Analysis result | Primary textual/structured answer. |
| Confidence | Display of run-derived confidence. |
| Visual evidence | Overlays and related artifacts. |
| Change maps | Display when M4 produces a map. |
| Grounding regions | Display when grounding is produced. |
| Execution trace | Observable agent steps. |
| Downloadable report | Export of query, inputs summary, results, confidence, evidence references, and trace. |

The GUI must not embed specialist logic; it calls the backend/agent API.

---

## 9. Evaluation datasets

The system and evaluation plan must be able to use (as data availability and licenses allow):

| Dataset | Role (typical) |
| --- | --- |
| BigEarthNet | Remote-sensing adaptation / land-cover related training and eval. |
| VRSBench | Vision-language remote-sensing benchmark. |
| RSVQA | Remote-sensing VQA. |
| CDVQA | Change-detection VQA. |
| ISRO/SAC evaluation dataset | Official / challenge evaluation as provided. |

PNG/JPEG usage must follow §3.1 (benchmark-permitted only). Evaluation reports must use **actual** metrics computed on held-out or official splits — never fabricated scores.

---

## 10. Engineering principles (non-negotiable)

| Principle | Requirement |
| --- | --- |
| Modular architecture | Agent, models, tools, and UI are separate packages/modules. |
| Replaceable models | Specialists swap via interface + configuration. |
| Configuration-driven selection | Model choice is not hardcoded in the UI or scattered through the agent. |
| Type hints | Public APIs and core pipelines are typed. |
| Unit tests | Tests for validation, routing, fusion, I/O, and non-model logic; model tests as feasible with fixtures. |
| Logging | Structured logs for inspect, validate, select, execute, fuse. |
| Error handling | Failures surface to GUI and trace; no silent fake answers. |
| Reproducibility | Config, seeds, checkpoint paths, and data versions recorded. |
| Honest outputs | No fake AI outputs, fake confidence, fake metadata, or fake benchmark results. |

---

## 11. Representative queries (acceptance examples)

These queries are **acceptance scenarios**, not canned answers:

1. *Describe the land-cover and major objects visible in this image.*
2. *Highlight the water body referred to in the query.*
3. *What changed between these two dates, and where did the change occur?*
4. *Use the optical and SAR images together to identify built-up and water-covered regions.*
5. *Has the built-up area increased, decreased, or remained unchanged?*

Passing means the live pipeline handles the matching input mode and returns real specialist outputs plus evidence/trace as applicable.

---

## 12. Constraints and anti-patterns

**Must:**

- Use specialized remote-sensing models and tools behind a common interface.
- Keep agent, models, tools, and UI clearly separated.
- Derive confidence and metadata from actual runs and files.

**Must not:**

- Depend on a generic LLM/VLM as the sole perception stack.
- Implement dummy models that return plausible but invented remote-sensing answers.
- Hardcode confidence, EXIF/GeoTIFF fields, change maps, or leaderboard numbers.
- Collapse all capabilities into a single untyped “ask the model” call without routing and validation.

---

## 13. Deliverables (when implementation is authorized)

Not started in this document’s scope. Expected later:

- Application (GUI + backend agent).
- Specialist adapters and configuration.
- Documented RS adaptation (dataset, method, checkpoint).
- Tests, logging, and a report export path.
- Honest evaluation notes against the datasets in §9 as access allows.

---

## 14. Open decisions (deferred)

To be decided at implementation time, not in this spec:

- UI stack (e.g. web framework).
- Exact specialist model catalog and licenses.
- Training hardware and adaptation recipe beyond the M1 requirement.
- Co-registration: user-provided only vs. optional registration tool.
- Report format (PDF vs. HTML vs. ZIP of artifacts).

---

## Document control

| Version | Date | Notes |
| --- | --- | --- |
| 0.1 | 2026-09-04 | Initial engineering spec from problem statement 26167. Implementation not started. |
