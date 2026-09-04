import os
from pathlib import Path
import tempfile
import time
from typing import Any, Optional
import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from satquery.models.opencd_bit import OpenCDBITModel


def create_synthetic_geotiff(
    path: Path,
    width: int,
    height: int,
    seed: int = 42,
    crs: str = "EPSG:32633",
) -> None:
    """Generate a realistic synthetic 3-band GeoTIFF test raster."""
    np.random.seed(seed)
    # Generate structured random imagery
    base = np.random.randint(40, 200, size=(3, height, width), dtype=np.uint8)
    transform = from_origin(500000.0, 4649760.0, 10.0, 10.0)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(base)


def run_bit_benchmark(
    model: Optional[OpenCDBITModel] = None,
    sizes: Optional[list[tuple[int, int]]] = None,
) -> dict[str, Any]:
    """Benchmark Open-CD BIT change detection across various scene dimensions.

    Measures:
    - Preprocessing time
    - Inference time
    - Postprocessing time
    - Total processing time
    - Number of tiles
    - Peak GPU memory (if CUDA available)
    """
    if sizes is None:
        sizes = [(256, 256), (512, 512), (1024, 1024)]

    if model is None:
        model = OpenCDBITModel()

    results: dict[str, Any] = {
        "device": model.device_str,
        "cuda_available": model.cuda_available,
        "pytorch_version": model.pytorch_version,
        "benchmarks": [],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for w, h in sizes:
            t0_p = tmp_path / f"t0_{w}x{h}.tif"
            t1_p = tmp_path / f"t1_{w}x{h}.tif"

            create_synthetic_geotiff(t0_p, width=w, height=h, seed=10)
            create_synthetic_geotiff(t1_p, width=w, height=h, seed=20)

            # Measure peak GPU memory if CUDA
            if model.cuda_available:
                torch.cuda.reset_peak_memory_stats()

            t_start = time.perf_counter()
            res = model.predict({"t0": str(t0_p), "t1": str(t1_p)})
            total_wall_s = time.perf_counter() - t_start

            if res.status != "success":
                results["benchmarks"].append({
                    "dimensions": f"{w}x{h}",
                    "status": "error",
                    "error": res.error,
                })
                continue

            # Extract statistics artifact
            stats_art = next((a for a in res.artifacts if a["type"] == "change_statistics"), None)
            stats = stats_art["data"] if stats_art else {}
            timings = stats.get("timings_seconds", {})

            peak_vram_mb = None
            if model.cuda_available:
                peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

            benchmark_entry = {
                "dimensions": f"{w}x{h}",
                "total_pixels": w * h,
                "tiles_evaluated": stats.get("total_tiles_evaluated", 0),
                "preprocessing_time_s": timings.get("preprocessing", 0.0),
                "inference_time_s": timings.get("inference", 0.0),
                "postprocessing_time_s": timings.get("postprocessing", 0.0),
                "total_time_s": timings.get("total", round(total_wall_s, 3)),
                "peak_gpu_memory_mb": peak_vram_mb,
            }
            results["benchmarks"].append(benchmark_entry)

    return results


if __name__ == "__main__":
    import pprint
    print("Running SatQuery AI Phase 3B Performance Benchmark...")
    res = run_bit_benchmark()
    pprint.pprint(res)
