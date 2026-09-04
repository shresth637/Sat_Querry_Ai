# SatQuery AI — Multi-Modal Geospatial Agentic AI Platform

SatQuery AI is an agentic Earth observation and satellite imagery platform designed for interactive geospatial query interpretation, multi-modal raster validation, and specialist neural network inference.

---

## Phase 3B: Production-Grade Bi-Temporal Change Detection

Phase 3B delivers full production-grade bi-temporal change detection powered by the **Open-CD Bitemporal Image Transformer (BIT)** with sliding-window tiled inference.

### Key Capabilities

1. **Large-Image Tiled Sliding-Window Inference**
   - Eliminates whole-scene downsampling. High-resolution satellite scenes (512×512, 1024×1024, and larger) are divided into overlapping 256×256 tiles.
   - Evaluated using configurable tile overlap (default 25% / 64 px).
   - Reconstructed seamlessly at full native spatial resolution using a 2D cosine (tapered) blend window, eliminating edge seams and tile boundaries.

2. **Multi-Artifact Geospatial Output Pipeline**
   - **Binary Change Mask (`*_change_mask.tif`)**: Discrete `uint8` GeoTIFF (`0 = unchanged`, `1 = changed`) preserving original raster width, height, CRS, and affine transform.
   - **Change Probability Map (`*_change_prob.tif`)**: Continuous `float32` GeoTIFF containing change probabilities in $[0.0, 1.0]$ at native spatial resolution.
   - **Visualization Overlay (`*_change_vis.png`)**: RGB composite highlighting changed regions in high-contrast red overlaid on date T1 satellite imagery.
   - **Comprehensive Statistics (`*_stats.json`)**: Detailed report covering total pixels, changed pixels, change %, confidence distribution, timing breakdowns, and physical area.

3. **Geodesically Rigorous Physical Area Calculation**
   - **Projected CRS (e.g. UTM)**: Direct planar metric calculation ($m^2$, hectares, $km^2$).
   - **Geographic CRS (e.g. EPSG:4326)**: Rigorous WGS84 ellipsoidal surface area calculation scaling latitude and longitude degree deltas by local radii of curvature ($M$ and $N$). Degrees are never incorrectly treated as meters.
   - Calculation method is explicitly recorded in output statistics (`area_calculation_method`).

4. **Hardware Acceleration & CPU Fallback**
   - Automatically detects whether CUDA acceleration is available on the host.
   - Uses `torch.device("cuda")` when supported with peak VRAM tracking.
   - Provides deterministic CPU fallback without degradation of inference quality.

5. **Configurable Sensitivity Threshold**
   - Change classification threshold is fully configurable (default: `0.50`, range: `[0.05, 0.95]`).
   - Dynamically adjustable via UI slider or configuration file.

6. **Transparent Confidence & Uncertainty Metrics**
   - Discloses accurate prediction metrics: Mean Prediction Confidence, Changed-Pixel Confidence, Unchanged-Pixel Confidence, and Low-Confidence Pixel Percentage.

---

## Model Architecture & Checkpoint Information

* **Architecture**: Bitemporal Image Transformer (BIT) (Chen et al., 2021; Open-CD).
  * **Backbone**: ResNetV1c (stem with three 3×3 convolutions, stages 1–3 feature extraction).
  * **Decoder**: Spatial-Temporal Tokenizer, Transformer Encoder, and Transformer Decoder (`BITHead`) with spatial differencing.
* **Checkpoint**: `bit_r18_256x256_40k_levircd.pth` (40.5 MB).
* **Weights Source**: Likyoo Open-CD Model Zoo (`likyoo/Open-CD_Model_Zoo`).
* **Training Dataset**: LEVIR-CD building change detection benchmark.
* **Domain Note**: The model is pretrained on the LEVIR-CD benchmark (primarily building and urban change detection). Detection characteristics on arbitrary natural surfaces, agricultural plots, or SAR imagery may differ.

---

## Input Requirements & Validation

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
    batch_size: 4
    device: auto
```

---

## Benchmark Performance

Measured on the local environment (`torch==2.14.0+cpu`):

| Scene Dimensions | Pixel Count | Tiles Evaluated | Preprocess Time | Inference Time | Postprocess Time | Total Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **256 × 256** | 65,536 | 1 | 0.015 s | 0.101 s | 0.069 s | **0.213 s** |
| **512 × 512** | 262,144 | 9 | 0.055 s | 0.786 s | 0.054 s | **0.923 s** |
| **1024 × 1024** | 1,048,576 | 25 | 0.231 s | 2.334 s | 0.172 s | **2.768 s** |

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
All 64 unit and integration tests pass covering Phase 1 (foundation), Phase 2 (agent controller & routing), Phase 3A (Open-CD model integration), and Phase 3B (tiled inference, geospatial validation, and artifact pipeline).

---

## Example Workflow

1. Navigate to `http://localhost:8501/`.
2. Under **1. Data Input**, select **Bi-Temporal Pair**.
3. Upload the Date T0 ("Before") GeoTIFF and Date T1 ("After") GeoTIFF.
4. Adjust the **Change Threshold** slider if desired (default `0.50`).
5. Under **3. Natural Language Query**, enter:
   `"What changed between these two dates?"` or click the example query button.
6. Click **🚀 Run Agent Analysis**.
7. View:
   - Model specifications card (device, tile size, overlap, threshold).
   - Change statistics metrics (Changed Area in hectares/$m^2$, Change %, Changed Pixels, Confidence, Processing Time).
   - Tabbed visual evidence: T0 image, T1 image, continuous Change Probability, discrete Binary Change Map, and high-contrast Change Overlay.
   - Download generated binary mask GeoTIFF, probability GeoTIFF, statistics JSON, and overlay PNG directly from the UI.
