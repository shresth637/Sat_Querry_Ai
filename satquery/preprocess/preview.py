from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import rasterio


def generate_preview_image(
    path: Union[str, Path],
    max_side: int = 512,
) -> Tuple[Optional[Image.Image], str]:
    """Generate a safe, derived display representation of a raster for visualization.

    Does not modify the original raster.
    Applies 2-98% percentile stretch for contrast.
    Labels derived representation explicitly.
    """
    file_path = Path(path)
    if not file_path.exists():
        return None, "File not found"

    try:
        with rasterio.open(file_path) as src:
            w, h = src.width, src.height
            if w <= 0 or h <= 0:
                return None, "Invalid image dimensions"

            # Calculate downscaled dimensions to preserve aspect ratio
            scale = min(max_side / max(w, h), 1.0)
            out_w = max(1, int(w * scale))
            out_h = max(1, int(h * scale))
            out_shape = (out_h, out_w)

            count = src.count

            if count >= 3:
                # Multi-band: read first 3 bands
                bands = []
                for b_idx in (1, 2, 3):
                    arr = src.read(b_idx, out_shape=out_shape).astype(np.float32)
                    # Mask nodata / nan
                    valid = arr[np.isfinite(arr)]
                    if src.nodata is not None:
                        valid = valid[valid != src.nodata]
                    if valid.size > 0:
                        p2, p98 = np.percentile(valid, (2, 98))
                        if p98 > p2:
                            arr = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
                        else:
                            arr = np.zeros_like(arr)
                    else:
                        arr = np.zeros_like(arr)
                    bands.append((arr * 255.0).astype(np.uint8))

                rgb_array = np.stack(bands, axis=-1)
                img = Image.fromarray(rgb_array, mode="RGB")
                label = "Derived RGB display representation (percentile-stretched bands 1-3)"
                return img, label

            elif count == 1:
                # Single-band or SAR
                arr = src.read(1, out_shape=out_shape).astype(np.float32)
                valid = arr[np.isfinite(arr)]
                if src.nodata is not None:
                    valid = valid[valid != src.nodata]
                if valid.size > 0:
                    p2, p98 = np.percentile(valid, (2, 98))
                    if p98 > p2:
                        arr = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
                    else:
                        arr = np.zeros_like(arr)
                else:
                    arr = np.zeros_like(arr)

                gray_array = (arr * 255.0).astype(np.uint8)
                img = Image.fromarray(gray_array, mode="L")
                label = "Derived grayscale/SAR intensity display representation (percentile-stretched)"
                return img, label

            else:
                # 2-band raster
                arr = src.read(1, out_shape=out_shape).astype(np.float32)
                valid = arr[np.isfinite(arr)]
                if valid.size > 0:
                    p2, p98 = np.percentile(valid, (2, 98))
                    if p98 > p2:
                        arr = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
                    else:
                        arr = np.zeros_like(arr)
                else:
                    arr = np.zeros_like(arr)
                img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
                label = "Derived 2-band visualization (band 1 stretched)"
                return img, label

    except Exception as exc:
        return None, f"Could not generate visual preview: {str(exc)}"
