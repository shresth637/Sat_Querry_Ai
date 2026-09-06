# SatQuery AI — Multi-Modal Geospatial Agentic AI Platform

SatQuery AI is an agentic Earth observation and satellite imagery platform designed for interactive geospatial query interpretation, multi-modal raster validation, spatial change intelligence, and specialist neural network inference.

---

## Phase 4: Multi-Model Satellite Intelligence

Phase 4 evolves SatQuery AI into a comprehensive multi-model satellite intelligence platform. Beyond bi-temporal change detection, the platform now integrates multi-label land-cover scene classification, vision-language intelligence, spatial bounding box grounding, and dynamic VRAM/device resource management.

### Key Capabilities

1. **Multi-Model Architecture & Catalog**
   - **Open-CD BIT ResNet-18 (`opencd_bit_change`)**: **READY**. Production bi-temporal change detection with 256×256 tiled sliding-window inference, 2D cosine blending, spatial change region extraction, and RFC 7946 GeoJSON export.
   - **BigEarthNet v2 (reBEN) ResNet-50 (`bigearthnet_resnet50_s1s2`)**: **INTEGRATED**. 19 Corine Land Cover multi-label scene classifier on Sentinel-2 optical imagery. Generates class probability distributions, publication-ready horizontal bar charts (`land_cover_distribution.png`), and structured factual summaries.
   - **GeoChat-7B VLM (`geochat_vqa`, `geochat_caption`, `geochat_grounding`)**: **INTEGRATED ADAPTER**. Vision-language model supporting single-image visual QA, detailed scene descriptions, and referring expression grounding. Includes 4-bit `bitsandbytes` quantization for 6 GB GPUs, CPU fallback, and strict honesty reporting (`status="not_configured"` if 14 GB checkpoint is absent).

2. **Resource & Device Governance (`satquery/models/manager.py`)**
   - **Hardware Detection**: Automatically detects host CUDA capabilities and VRAM constraints (e.g. host NVIDIA GeForce RTX 4050 6 GB GPU) with safe CPU fallback.
   - **Lazy Loading**: Specialist models are loaded into memory/device only when requested by an active plan.
   - **Single-Model Residency**: To protect the 6 GB VRAM budget, only one large model resides in VRAM at any given time.
   - **Cache Cleanup**: Automatically triggers Python garbage collection and `torch.cuda.empty_cache()` whenever models are swapped.

3. **Spatial Grounding & Bounding Box Intelligence (`satquery/models/grounding.py`)**
   - **Standardized Coordinates**: Converts normalized $[ymin, xmin, ymax, xmax]$ predictions to discrete raster pixel coordinates $[xmin, ymin, xmax, ymax]$ and native CRS geographic/projected coordinates.
   - **RFC 7946 GeoJSON**: Serializes bounding boxes to `grounding_boxes.geojson` with native EPSG CRS URN metadata.
   - **Visual Overlays**: Renders high-contrast bounding boxes, ranking badges (`#1 water_body (0.95)`), and semi-transparent fills (`grounding_overlay.png`).

4. **Multi-Modal Semantic Query Router (`QueryIntentParser`)**
   - Deterministically extracts:
     - **Task Type**: `CLASSIFICATION`, `SINGLE_VQA`, `SINGLE_CAPTION`, `SINGLE_GROUNDING`, `BI_TEMPORAL_CHANGE`, `CHANGE_VQA`, `OPTICAL_SAR_ANALYSIS`
     - **Temporal Mode**: `SINGLE_IMAGE`, `BITEMPORAL`, `OPTICAL_SAR`
     - **Spatial Requirement**: `WHOLE_SCENE`, `REGIONAL_CLUSTERS`, `BOUNDING_BOX_GROUNDING`
   - Intelligently routes land-cover, crop, forest, and scene classification queries to BigEarthNet, questions to VQA, descriptions to Captioning, and differences/construction to Change Detection.

5. **Adaptive Production Streamlit UX (`app.py`)**
   - **Multi-Workflow Switcher**: Seamlessly toggles between **Single Satellite Image**, **Bi-Temporal Pair (Change Detection)**, and **Optical + SAR Pair**.
   - **Dynamic Controls**: Displays relevant parameters based on active mode (e.g., Land-Cover Confidence Threshold slider for Single Image; Change Threshold and Min Region Size sliders for Bi-Temporal).
   - **Real-Time Stage Progression**: Multi-stage `st.status` widget tracking raster inspection, intent parsing, neural inference, and artifact generation.
   - **Dynamic Evidence Tabs**: Automatically presents relevant tabs (Source Scene, Land-Cover Classes, Change Probability, Binary Mask, Change Overlay, Change Regions, Grounded Bounding Boxes).
   - **One-Click Artifact Downloads**: Direct UI buttons to download GeoTIFFs, PNG visualizations, GeoJSON vectors, and machine-readable JSON statistics.

---

## Model Architecture & Checkpoint Information

* **Open-CD BIT ResNet-18**:
  * Architecture: ResNetV1c stem + Spatial-Temporal Tokenizer + Transformer Encoder/Decoder.
  * Checkpoint: `models/checkpoints/bit_r18_256x256_40k_levircd.pth` (40.5 MB).
  * Status: **READY**.
* **BigEarthNet v2 ResNet-50**:
  * Architecture: ResNet-50 backbone with 19-class linear sigmoid projection.
  * Checkpoint: `models/checkpoints/resnet50_s2_v0.2.0.pth` (~95 MB).
  * 19 Classes: Urban fabric, Industrial/commercial, Arable land, Permanent crops, Pastures, Complex cultivation, Agriculture, Broad-leaved forest, Coniferous forest, Mixed forest, Natural grassland, Moors/heathlands, Sclerophyllous vegetation, Transitional woodland, Beaches/dunes/sands, Bare rock, Sparsely vegetated, Inland wetlands, Marine/coastal waters.
* **GeoChat-7B**:
  * Architecture: CLIP-ViT-L/14-336 + Vicuna-7B v1.5 / LLaMA-2 backbone.
  * Size: 14.2 GB.
  * Quantization: 4-bit NF4 via `bitsandbytes` when CUDA is active; status reports `not_configured` when weights are not downloaded.

---

## Configuration

Model parameters can be customized in [`config/models.yaml`](file:///C:/Users/HP/Projects/satquery-ai/config/models.yaml):

```yaml
models:
  - id: opencd_bit_change
    name: Open-CD BIT ResNet-18
    version: r18-levir
    capabilities: [change_detect]
    adapter: satquery.models.opencd_bit.OpenCDBITModel
    enabled: true
    weights_path: models/checkpoints/bit_r18_256x256_40k_levircd.pth
    tile_size: 256
    tile_overlap: 0.25
    change_threshold: 0.5
    min_change_region_pixels: 20
    max_regions_visualized: 20
    device: auto

  - id: bigearthnet_resnet50_s1s2
    name: BigEarthNet v2 ResNet-50 S1+S2
    version: v0.2.0
    capabilities: [optical_sar_fusion, land_cover, classification]
    adapter: satquery.models.bigearthnet.BigEarthNetModel
    enabled: false
    weights_path: models/checkpoints/resnet50_s2_v0.2.0.pth
    threshold: 0.3
    device: auto

  - id: geochat_vqa
    name: GeoChat-7B
    version: "7B"
    capabilities: [vqa]
    adapter: satquery.models.geochat.GeoChatVLMModel
    capability: image_vqa
    enabled: false
    weights_path: null
```

---

## Benchmark Performance

Measured on the local environment (`torch==2.14.0+cpu`):

| Scene Dimensions | Pixel Count | Tiles Evaluated | Preprocess Time | Inference Time | Postprocess Time | Total Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **256 × 256** | 65,536 | 1 | 0.016 s | 0.104 s | 0.083 s | **0.259 s** |
| **512 × 512** | 262,144 | 9 | 0.059 s | 0.889 s | 0.112 s | **1.090 s** |
| **1024 × 1024** | 1,048,576 | 25 | 0.252 s | 2.557 s | 0.413 s | **3.253 s** |

To run the benchmark utility:
```powershell
.\.venv\Scripts\python.exe -m satquery.models.benchmarks
```

---

## Running the Applications

### 1. Launch Cinematic 3D Web Experience (Interactive Orbital Command)
```powershell
.\.venv\Scripts\python.exe server.py
```
Open [http://localhost:8000](http://localhost:8000) in any browser to access the 3D Web Experience featuring:
- **Interactive 3D Orbital Earth & Sensor Simulation**: Procedural WebGL globe with real-time continent graticules, polar satellite orbit, volumetric nadir sensor scanning beam, and smooth spherical camera targeting.
- **Interactive Optical Split Curtain Scanner**: High-precision draggable before/after comparison with dynamic difference overlays and hover coordinate/likelihood inspector.
- **3D Isometric Layer Decomposer**: Exploded 3D spatial stack showing T0 Baseline, Neural Probability Mesh, and T1 Resurvey with customizable explosion separation and pitch/yaw rotation.
- **10-Stage Agent Execution Pipeline HUD**: Real-time visual tracking of all 10 AgentController execution phases with millisecond latency telemetry.
- **Authentic Recon Missions**: Pre-loaded with Dubai Urban Expansion (0.5m GSD, EPSG:32640), Danube Hydrological Inundation (10m GSD, EPSG:4326), and Castile Agricultural Land-Cover (10m GSD).
- **Custom Upload & Production Artifact Vault**: Upload arbitrary GeoTIFF/PNG rasters and download georeferenced binary masks, continuous probability maps, and RFC 7946 GeoJSON vectors.

### 2. Launch Streamlit Web UI
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

### 3. Run Test Suite
```powershell
.\.venv\Scripts\pytest.exe -v
```
All **91 unit and integration tests** pass covering:
- Phase 1: Foundation, configuration, geospatial validators, evidence schemas (17 tests)
- Phase 2: Agent controller, natural language routing, execution traces (12 tests)
- Phase 3A: Open-CD BIT ResNet-18 model integration and weights verification (17 tests)
- Phase 3B: Tiled sliding-window inference, geospatial preservation, area calculation (18 tests)
- Phase 3C: Spatial change region extraction, GeoJSON vectorization, ranked visualizer (12 tests)
- Phase 4: ResourceManager, BigEarthNet classifier, GeoChat VLM adapter, grounding utilities, multi-model router (15 tests)

---

## Example Workflows

### 1. Single-Image Land-Cover Scene Understanding
1. Select **Single Satellite Image** in the Streamlit UI.
2. Upload a satellite scene GeoTIFF.
3. Adjust the **Land-Cover Confidence Threshold** (default: `0.30`).
4. Enter: `"Classify the land-cover categories in this scene"`.
5. Click **🚀 Run Agent Analysis**.
6. View executive summary, top detected Corine classes, geospatial context, probability bar chart (`land_cover_distribution.png`), and download statistics JSON.

### 2. Bi-Temporal Change Detection & Spatial Intelligence
1. Select **Bi-Temporal Pair (Change Detection)** in the Streamlit UI.
2. Upload the Date T0 ("Before") GeoTIFF and Date T1 ("After") GeoTIFF.
3. Adjust the **Change Threshold** (default: `0.50`) and **Min Region Size** (default: `20 px`).
4. Enter: `"What changed between these two dates?"`.
5. Click **🚀 Run Agent Analysis**.
6. View 8-section factual intelligence report, change statistics metrics, tabbed visual evidence (Probability, Binary Mask, Red Overlay, Ranked Labeled Regions), and download GeoTIFFs, PNGs, and RFC 7946 GeoJSON.
