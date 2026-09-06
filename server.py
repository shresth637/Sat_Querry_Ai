import os
import sys
import time
import json
import shutil
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import JSONResponse, FileResponse, HTMLResponse
from starlette.staticfiles import StaticFiles
from PIL import Image
import numpy as np

# Ensure satquery is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from satquery.agent.controller import AgentController
from satquery.domain.schemas import SlotAssignment, InputMode, Modality
from satquery.registry.models import ModelRegistry
from satquery.models.manager import get_resource_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satquery.server")

# Paths
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLES_DIR = PROJECT_ROOT / "samples"
STATIC_DIR = PROJECT_ROOT / "static"

for p in [UPLOAD_DIR, OUTPUT_DIR, SAMPLES_DIR, STATIC_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Initialize Agent Controller
agent_controller = AgentController()


def serialize_obj(obj: Any) -> Any:
    """Recursively convert Pydantic models, enums, numpy arrays, and dicts to JSON-serializable primitives."""
    if hasattr(obj, "model_dump"):
        return serialize_obj(obj.model_dump())
    if hasattr(obj, "value") and not hasattr(obj, "__dict__"):  # Enum
        return obj.value
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [serialize_obj(i) for i in obj]
    if isinstance(obj, dict):
        return {str(k): serialize_obj(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {str(k): serialize_obj(v) for k, v in obj.__dict__.items() if not str(k).startswith("_")}
    return obj


def ensure_preview_image(file_path: str) -> Optional[str]:
    """Generate a high-contrast web PNG preview if the image is a GeoTIFF or lacks preview."""
    try:
        p = Path(file_path)
        if not p.exists():
            return None
        
        # If it's already a PNG/JPEG, return web relative path
        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            return f"/{p.relative_to(PROJECT_ROOT).as_posix()}"
        
        preview_png = p.with_suffix(".png")
        if preview_png.exists():
            return f"/{preview_png.relative_to(PROJECT_ROOT).as_posix()}"
        
        # Render preview from raster
        import rasterio
        with rasterio.open(str(p)) as src:
            count = src.count
            if count >= 3:
                r = src.read(1)
                g = src.read(2)
                b = src.read(3)
                arr = np.dstack([r, g, b])
            else:
                band = src.read(1)
                arr = np.dstack([band, band, band])
            
            if arr.dtype != np.uint8:
                valid = arr[np.isfinite(arr)]
                if len(valid) > 0:
                    v_min, v_max = np.percentile(valid, 2), np.percentile(valid, 98)
                    if v_max > v_min:
                        arr = np.clip((arr - v_min) / (v_max - v_min) * 255.0, 0, 255).astype(np.uint8)
                    else:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
            
            img = Image.fromarray(arr)
            img.save(str(preview_png), format="PNG")
            return f"/{preview_png.relative_to(PROJECT_ROOT).as_posix()}"
    except Exception as e:
        logger.warning(f"Could not generate preview for {file_path}: {e}")
        return None


# =====================================================================
# API Endpoints
# =====================================================================

async def index(request: Request):
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>SatQuery AI Engine Online</h1><p>Static index.html not yet initialized.</p>")


async def get_health(request: Request):
    rm = get_resource_manager()
    vram_info = rm.get_vram_info()
    model_entries = agent_controller.model_registry.list_models()
    active_models = list(getattr(rm, "_resident_models", {}).keys())
    active_name = active_models[0] if active_models else "None (Standby)"
    
    return JSONResponse({
        "status": "online",
        "system": "SatQuery AI Remote Sensing Intelligence",
        "device": "cuda" if vram_info.get("cuda_available") else "cpu",
        "device_name": vram_info.get("device_name", "CPU"),
        "vram_allocated_mb": round(vram_info.get("allocated_mb", 0.0), 2),
        "vram_total_mb": round(vram_info.get("total_mb", 0.0), 2),
        "models_registered": len(model_entries),
        "active_model": active_name,
        "models": [
            {
                "id": mid,
                "name": getattr(adapter, "name", mid),
                "status": getattr(adapter, "status", "unloaded"),
                "device": str(getattr(adapter, "device", "cpu")),
                "capabilities": getattr(adapter, "capabilities", [])
            }
            for mid, adapter in model_entries
        ]
    })



async def get_models(request: Request):
    model_entries = agent_controller.model_registry.list_models()
    return JSONResponse({
        "models": [
            {
                "id": mid,
                "name": getattr(adapter, "name", mid),
                "status": getattr(adapter, "status", "unloaded"),
                "device": str(getattr(adapter, "device", "cpu")),
                "capabilities": getattr(adapter, "capabilities", [])
            }
            for mid, adapter in model_entries
        ]
    })


async def get_samples(request: Request):
    samples = [
        {
            "id": "dubai_expansion",
            "name": "Dubai Metropolis Urban Expansion",
            "region": "Dubai, United Arab Emirates",
            "coordinates": {"lat": 25.2048, "lng": 55.2708},
            "sensor": "WorldView-3 / QuickBird Optical",
            "resolution": "0.5m GSD",
            "crs": "EPSG:32640 (UTM Zone 40N)",
            "task_type": "BI_TEMPORAL_CHANGE",
            "input_mode": "I4_BITEMPORAL_PAIR",
            "t0_path": "samples/dubai_t0.tif",
            "t1_path": "samples/dubai_t1.tif",
            "t0_preview": "/samples/dubai_t0.png",
            "t1_preview": "/samples/dubai_t1.png",
            "recommended_query": "Detect new building footprints, urban infrastructure, and commercial expansion between T0 and T1",
            "description": "High-resolution bi-temporal optical scene capturing massive urban infrastructure construction, foundations, and building footprints."
        },
        {
            "id": "danube_flood",
            "name": "Danube River Hydrological Inundation",
            "region": "Danube River Basin, Europe",
            "coordinates": {"lat": 45.2671, "lng": 19.8335},
            "sensor": "Sentinel-2 MSI Multi-spectral",
            "resolution": "10.0m GSD",
            "crs": "EPSG:4326 (WGS 84)",
            "task_type": "BI_TEMPORAL_CHANGE",
            "input_mode": "I4_BITEMPORAL_PAIR",
            "t0_path": "samples/danube_flood_t0.tif",
            "t1_path": "samples/danube_flood_t1.tif",
            "t0_preview": "/samples/danube_flood_t0.png",
            "t1_preview": "/samples/danube_flood_t1.png",
            "recommended_query": "Identify flood extent, submerged agricultural parcels, and surface water inundation",
            "description": "Bi-temporal monitoring pair tracking severe hydrological flooding, embankment breaches, and agricultural inundation."
        },
        {
            "id": "spain_agriculture",
            "name": "Castile Agricultural Land-Cover",
            "region": "Castile and León, Spain",
            "coordinates": {"lat": 41.6523, "lng": -4.7245},
            "sensor": "Sentinel-2 Multi-spectral",
            "resolution": "10.0m GSD",
            "crs": "EPSG:32640 (UTM Zone 30N)",
            "task_type": "CLASSIFICATION",
            "input_mode": "I1_SINGLE_OPTICAL",
            "t0_path": "samples/agricultural_scene.tif",
            "t1_path": None,
            "t0_preview": "/samples/agricultural_scene.png",
            "t1_preview": None,
            "recommended_query": "Classify agricultural land-use, crop distributions, and arable parcels across multi-spectral bands",
            "description": "Multi-spectral agricultural parcel classification utilizing BigEarthNet deep convolutional representations."
        }
    ]
    return JSONResponse({"samples": samples})


async def upload_files(request: Request):
    try:
        form = await request.form()
        uploaded_items = {}
        timestamp = int(time.time() * 1000)

        for key in ["t0_file", "t1_file", "single_file", "file"]:
            upload_item = form.get(key)
            if upload_item and hasattr(upload_item, "filename") and upload_item.filename:
                orig_name = Path(upload_item.filename).name
                safe_name = f"{timestamp}_{orig_name}"
                dest_path = UPLOAD_DIR / safe_name
                
                content = await upload_item.read()
                with open(dest_path, "wb") as f:
                    f.write(content)
                
                rel_path = dest_path.relative_to(PROJECT_ROOT).as_posix()
                preview_url = ensure_preview_image(str(dest_path))
                
                uploaded_items[key] = {
                    "filename": orig_name,
                    "server_path": rel_path,
                    "preview_url": preview_url or f"/{rel_path}",
                    "size_bytes": len(content)
                }

        if not uploaded_items:
            return JSONResponse({"error": "No valid files received."}, status_code=400)

        return JSONResponse({
            "status": "success",
            "uploaded": uploaded_items
        })
    except Exception as e:
        logger.exception("Upload error")
        return JSONResponse({"error": str(e)}, status_code=500)


async def analyze(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    query = data.get("query", "Detect changes")
    task_type = data.get("task_type", "BI_TEMPORAL_CHANGE")
    threshold = float(data.get("threshold", 0.50))
    t0_path = data.get("t0_path")
    t1_path = data.get("t1_path")
    single_image_path = data.get("image_path")

    # Resolve input mode and slots
    slots: List[SlotAssignment] = []
    
    if t0_path and t1_path:
        input_mode = InputMode.I4_BITEMPORAL_PAIR
        slots.append(SlotAssignment(slot_id="t0", file_path=t0_path, declared_modality=Modality.OPTICAL))
        slots.append(SlotAssignment(slot_id="t1", file_path=t1_path, declared_modality=Modality.OPTICAL))
    elif single_image_path or t0_path:
        input_mode = InputMode.I1_SINGLE_OPTICAL
        target_path = single_image_path or t0_path
        slots.append(SlotAssignment(slot_id="primary", file_path=target_path, declared_modality=Modality.OPTICAL))
    else:
        return JSONResponse({"error": "At least one valid image path must be provided."}, status_code=400)

    logger.info(f"Executing analyze with query='{query}', mode={input_mode.value}, threshold={threshold}")

    try:
        # Run agent analysis
        result = agent_controller.analyze(
            query=query,
            slots=slots,
            input_mode=input_mode,
            threshold=threshold,
            allow_png_jpeg=True
        )

        serialized = serialize_obj(result)

        # Enhance evidence items with web accessible URLs and format artifacts
        artifacts_summary = {
            "change_mask_geotiff": None,
            "change_prob_geotiff": None,
            "change_visualization": None,
            "change_regions_visualization": None,
            "geojson_data": None,
            "geojson_url": None,
            "statistics": {},
            "raster_meta_t0": serialized.get("metas", {}).get("t0"),
            "raster_meta_t1": serialized.get("metas", {}).get("t1"),
        }

        for ev in serialized.get("evidence", []):
            fp = ev.get("file_path")
            if fp:
                try:
                    p = Path(fp)
                    if p.is_absolute():
                        try:
                            rel = p.relative_to(PROJECT_ROOT).as_posix()
                        except ValueError:
                            rel = p.name
                    else:
                        rel = p.as_posix()
                    ev["web_url"] = f"/{rel}"
                except Exception:
                    ev["web_url"] = fp

            title = ev.get("title", "")
            ev_data = ev.get("data")

            if "Binary Change Mask (GeoTIFF)" in title:
                artifacts_summary["change_mask_geotiff"] = ev.get("web_url")
            elif "Change Probability Map (GeoTIFF)" in title:
                artifacts_summary["change_prob_geotiff"] = ev.get("web_url")
            elif "Change Map Visual Overlay" in title:
                artifacts_summary["change_visualization"] = ev.get("web_url")
            elif "Change Regions with Bounding Boxes" in title:
                artifacts_summary["change_regions_visualization"] = ev.get("web_url")
            elif "Vector Map (GeoJSON)" in title or "change_regions_geojson" in str(fp):
                artifacts_summary["geojson_url"] = ev.get("web_url")
                if ev_data:
                    artifacts_summary["geojson_data"] = ev_data
                elif fp and Path(fp).exists():
                    try:
                        with open(fp, "r", encoding="utf-8") as gf:
                            artifacts_summary["geojson_data"] = json.load(gf)
                    except Exception as ge:
                        logger.warning(f"Could not load GeoJSON from {fp}: {ge}")
            elif "Statistics" in title and ev_data:
                artifacts_summary["statistics"] = ev_data

        serialized["artifacts_summary"] = artifacts_summary

        return JSONResponse(serialized)

    except Exception as e:
        logger.exception("Inference execution failed")
        return JSONResponse({"error": str(e)}, status_code=500)


# =====================================================================
# Starlette Application Setup
# =====================================================================

routes = [
    Route("/", endpoint=index, methods=["GET"]),
    Route("/api/health", endpoint=get_health, methods=["GET"]),
    Route("/api/models", endpoint=get_models, methods=["GET"]),
    Route("/api/samples", endpoint=get_samples, methods=["GET"]),
    Route("/api/upload", endpoint=upload_files, methods=["POST"]),
    Route("/api/analyze", endpoint=analyze, methods=["POST"]),
    Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
    Mount("/samples", app=StaticFiles(directory=str(SAMPLES_DIR)), name="samples"),
    Mount("/outputs", app=StaticFiles(directory=str(OUTPUT_DIR)), name="outputs"),
    Mount("/uploads", app=StaticFiles(directory=str(UPLOAD_DIR)), name="uploads"),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(routes=routes, middleware=middleware)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
