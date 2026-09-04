# SatQuery AI — Multi-Modal Geospatial Agentic AI Platform

SatQuery AI is an agentic Earth observation and satellite imagery platform designed for interactive geospatial query interpretation, multi-modal raster validation, spatial change intelligence, and specialist neural network inference.

---

## Phase 3C: Spatial Intelligence & Production UX

Phase 3C elevates SatQuery AI into a production-grade geospatial intelligence system. Beyond pixel-level change detection, the platform now extracts connected spatial change regions, polygonizes them into standard GeoJSON vectors, visualizes ranked bounding boxes, and generates structured, factual natural language intelligence reports.

### Key Capabilities

1. **Spatial Change Region Extraction & Vectorization**
   - **GDAL/Rasterio Polygonization**: Employs `rasterio.features.shapes` to convert binary change masks into vector polygons directly in native Coordinate Reference System (CRS) coordinates.
   - **Connected Component Attributes**: Calculates pixel count, physical surface area ($m^2$ and hectares), geographic/projected centroid, and bounding box for every discrete change cluster.
   - **Configurable Noise Filtering**: Eliminates speckle noise via `min_change_region_pixels` (default: 20 pixels), filtering out isolated pixel noise while preserving significant regional changes.
   - **RFC 7946 GeoJSON Export**: Automatically serializes extracted regions to `change_regions.geojson` containing full vector geometries and metadata.

2. **Ranked Change Regions Visualization**
   - Renders a publication-ready visualization (`change_regions_vis.png`) on top of Date T1 satellite imagery.
   - Highlights the top detected regions (up to 20 by area) with high-contrast bounding boxes, numbered ranking badges (`#1 (1,240px)`), and semi-transparent fills.

3. **Structured Factual Natural Language Interpretation**
   - Replaces generic text with an 8-section technical intelligence report:
     - **SUMMARY**: Executive assessment of detected changes and scene footprint.
     - **WHAT CHANGED**: Factual descriptions based strictly on model scope (explicit LEVIR-CD building/structure change focus; no fabricated semantics).
     - **HOW MUCH CHANGED**: Metric change statistics (hectares, $m^2$, change %, pixel count).
     - **SPATIAL DISTRIBUTION**: Count of significant regions, noise filtered, top 5 largest clusters with area and centroids.
     - **CONFIDENCE**: Mean prediction probability, changed/unchanged confidence, and low-confidence percentage.
     - **GEOSPATIAL INFORMATION**: CRS, affine transform, dimensions, spatial resolution, and ellipsoidal/planar area calculation method.
     - **ARTIFACTS**: Canonical absolute paths to all 6 generated outputs.
     - **LIMITATIONS**: Explicit model domain boundaries, resolution constraints, and verification recommendations.

4. **Production Run Isolation & Artifact Pipeline**
   - Every inference execution is isolated in its own run directory:
     `outputs/change_maps/run_<YYYYMMDD_HHMMSS>_<uuid8>/`
   - Complete 6-artifact delivery per run:
     - `change_mask.tif`: Discrete `uint8` GeoTIFF (`0 = unchanged`, `1 = changed`).
     - `change_prob.tif`: Continuous `float32` GeoTIFF ($[0.0, 1.0]$ probability map).
     - `change_overlay.png`: High-contrast red overlay on Date T1 imagery.
     - `change_regions_vis.png`: Ranked bounding box overlay with ranking badges.
     - `change_regions.geojson`: Vector polygons with geometry and physical metrics.
     - `stats.json`: Machine-readable execution statistics, confidence, and region metadata.

5. **Production Streamlit UX**
   - **6-Step Guided Workflow**: (1) Data Input & Mode Selection, (2) Spatial & Model Parameters, (3) Natural Language Query, (4) Execution Engine, (5) Spatial Intelligence Results, (6) Artifact Downloads.
   - **Real-Time Stage Progress Indicators**: Live `st.status` widget showing raster validation, tiled sliding-window inference, spatial region extraction, and report formatting.
   - **Dedicated Change Regions View**: Tabbed visual evidence featuring T0, T1, Probability, Binary Mask, Change Overlay, and Change Regions.
   - **Direct Downloads**: UI buttons to download GeoTIFFs, PNGs, stats JSON, and GeoJSON.

6. **Large-Image Tiled Sliding-Window Inference**
   - High-resolution satellite scenes (512×512, 1024×1024, and larger) are divided into overlapping 256×256 tiles with configurable overlap (default 25% / 64 px).
   - Reconstructed seamlessly at full native spatial resolution using a 2D cosine blend window, eliminating edge seams and tile boundary artifacts.

7. **Geodesically Rigorous Physical Area Calculation**
   - **Projected CRS (e.g. UTM)**: Direct planar metric calculation ($m^2$, hectares, $km^2$).
   - **Geographic CRS (e.g. EPSG:4326)**: WGS84 ellipsoidal surface area calculation scaling latitude and longitude degree deltas by local radii of curvature ($M$ and $N$).

8. **Enhanced Natural Language Query Router**
   - Intelligently recognizes queries including "construction", "differences", "compare", "changed buildings", "urban expansion", and "surface differences", routing seamlessly to the change detection pipeline.

---

## Model Architecture & Checkpoint Information

* **Architecture**: Bitemporal Image Transformer (BIT) (Chen et al., 2021; Open-CD).
  * **Backbone**: ResNetV1c (stem with three 3×3 convolutions, stages 1–3 feature extraction).
  * **Decoder**: Spatial-Temporal Tokenizer, Transformer Encoder, and Transformer Decoder (`BITHead`) with spatial differencing.
* **Checkpoint**: `models/checkpoints/bit_r18_256x256_40k_levircd.pth` (40.5 MB).
* **Weights Source**: Likyoo Open-CD Model Zoo (`likyoo/Open-CD_Model_Zoo`).
* **Training Dataset**: LEVIR-CD building change detection benchmark.
* **Domain Note**: The model is pretrained on the LEVIR-CD benchmark (primarily building and urban change detection). Detection characteristics on arbitrary natural surfaces, agricultural plots, or SAR imagery may differ.

---

## Input Requirements & Geospatial Validation

Before neural execution, the agent performs strict geospatial validation:
* Both T0 and T1 files must exist on disk and be readable by Rasterio.
* Both rasters must have matching dimensions (`width` and `height`).
* Rasters must share the same Coordinate Reference System (CRS).
* Rasters must share the same affine transform (spatial bounds and pixel resolution).
* Misaligned or incompatible rasters are rejected with explicit diagnostic messages.

---

## Configuration

Model parameters can be customized in [`config/models.yaml`](file:///C:/Users/HP/Projects/satquery-ai/config/models.yaml):

```yaml
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
    batch_size: 4
    device: auto
```

---

## Benchmark Performance

Measured on the local environment (`torch==2.14.0+cpu`):

| Scene Dimensions | Pixel Count | Tiles Evaluated | Preprocess Time | Inference Time | Postprocess Time | Total Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **256 × 256** | 65,536 | 1 | 0.015 s | 0.101 s | 0.069 s | **0.242 s** |
| **512 × 512** | 262,144 | 9 | 0.055 s | 0.786 s | 0.054 s | **1.021 s** |
| **1024 × 1024** | 1,048,576 | 25 | 0.231 s | 2.334 s | 0.172 s | **2.977 s** |

To run the benchmark utility:
```powershell
.\.venv\Scripts\python.exe -m satquery.models.benchmarks
```

---

## Running the Application

### 1. Launch Streamlit Web UI
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

### 2. Run Test Suite
```powershell
.\.venv\Scripts\pytest.exe -v
```
All 76 unit and integration tests pass covering:
- Phase 1: Foundation, configuration, geospatial validators, evidence schemas
- Phase 2: Agent controller, natural language routing, execution traces
- Phase 3A: Open-CD BIT ResNet-18 model integration and weights verification
- Phase 3B: Tiled sliding-window inference, geospatial preservation, area calculation
- Phase 3C: Spatial change region extraction, GeoJSON vectorization, ranked visualizer, factual interpretation, UI workflow

---

## Example Workflow

1. Navigate to `http://localhost:8501/`.
2. Under **1. Data Input & Mode**, select **Bi-Temporal Pair**.
3. Upload the Date T0 ("Before") GeoTIFF and Date T1 ("After") GeoTIFF.
4. Under **2. Spatial & Model Parameters**, adjust the **Change Threshold** (default: `0.50`) and **Min Region Size** (default: `20 px`) sliders.
5. Under **3. Natural Language Query**, enter:
   `"Identify any building or structural changes between T0 and T1"` or click an example query button.
6. Click **🚀 Run Agent Analysis**.
7. Observe the real-time stage progress indicator tracking validation, tiled inference, region vectorization, and report generation.
8. View:
   - Factual 8-section technical intelligence report.
   - Key statistics metrics (Total Changed Area, Change %, Regions Detected, Confidence, Total Latency).
   - Tabbed visual evidence: Before (T0), After (T1), Probability Map, Binary Mask, Change Overlay, and **Change Regions** with ranked bounding boxes.
   - Download generated binary mask GeoTIFF, probability GeoTIFF, GeoJSON vector polygons, statistics JSON, and visualizations directly from the UI.
