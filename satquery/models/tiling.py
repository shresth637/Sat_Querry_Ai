import math
from typing import Callable, Generator, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_tapered_window(tile_size: int, taper_ratio: float = 0.25) -> np.ndarray:
    """Create a 2D smooth blend window (raised cosine / Hann taper) to eliminate tile seams.

    The window has a flat region in the center and smoothly tapers down to 0 at the boundaries.
    """
    if tile_size <= 1:
        return np.ones((tile_size, tile_size), dtype=np.float32)

    taper_len = max(2, int(tile_size * taper_ratio))
    w_1d = np.ones(tile_size, dtype=np.float32)

    # Cosine ramp up
    ramp_up = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, taper_len)))
    w_1d[:taper_len] = ramp_up
    # Cosine ramp down
    w_1d[-taper_len:] = ramp_up[::-1]

    # 2D outer product
    w_2d = np.outer(w_1d, w_1d).astype(np.float32)
    # Ensure min weight is not strictly 0 to avoid zero-division near corners
    w_2d = np.maximum(w_2d, 1e-4)
    return w_2d


def generate_tile_windows(
    height: int,
    width: int,
    tile_size: int = 256,
    overlap_ratio: float = 0.25,
) -> list[tuple[int, int, int, int]]:
    """Generate a grid of sliding window coordinates (y, x, tile_h, tile_w) covering (height, width).

    Guarantees complete coverage of edges by placing boundary-aligned tiles when dimensions
    do not divide evenly by the step size.
    """
    if height <= tile_size and width <= tile_size:
        return [(0, 0, height, width)]

    step_y = max(1, int(tile_size * (1.0 - overlap_ratio)))
    step_x = max(1, int(tile_size * (1.0 - overlap_ratio)))

    # Compute y offsets
    y_offsets = list(range(0, max(1, height - tile_size + 1), step_y))
    if not y_offsets or y_offsets[-1] + tile_size < height:
        y_offsets.append(max(0, height - tile_size))
    # Remove duplicates and sort
    y_offsets = sorted(list(set(y_offsets)))

    # Compute x offsets
    x_offsets = list(range(0, max(1, width - tile_size + 1), step_x))
    if not x_offsets or x_offsets[-1] + tile_size < width:
        x_offsets.append(max(0, width - tile_size))
    x_offsets = sorted(list(set(x_offsets)))

    windows = []
    for y in y_offsets:
        for x in x_offsets:
            th = min(tile_size, height - y)
            tw = min(tile_size, width - x)
            windows.append((y, x, th, tw))

    return windows


def extract_tile_with_padding(
    image: np.ndarray,
    y: int,
    x: int,
    th: int,
    tw: int,
    target_size: int = 256,
) -> np.ndarray:
    """Extract a (C, th, tw) slice from image (C, H, W) and pad to (C, target_size, target_size) if needed."""
    tile = image[:, y : y + th, x : x + tw]
    if th == target_size and tw == target_size:
        return tile

    # Pad with edge replication
    pad_h = target_size - th
    pad_w = target_size - tw
    padded = np.pad(
        tile,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode="edge",
    )
    return padded


class TiledInferenceEngine:
    """Memory-safe tiled sliding-window inference engine for bi-temporal change detection."""

    def __init__(
        self,
        tile_size: int = 256,
        overlap_ratio: float = 0.25,
        batch_size: int = 4,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.batch_size = max(1, batch_size)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.blend_window = create_tapered_window(tile_size, taper_ratio=overlap_ratio)

    def predict_tiled(
        self,
        model: nn.Module,
        t0_chw: np.ndarray,
        t1_chw: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Execute sliding-window inference over full-resolution T0 and T1 scenes.

        Args:
            model: PyTorch BIT model.
            t0_chw: Preprocessed normalized T0 image of shape (3, H, W).
            t1_chw: Preprocessed normalized T1 image of shape (3, H, W).

        Returns:
            reconstructed_prob_map: Seamless 2D float32 change probability map in [0, 1] of shape (H, W).
            num_tiles: Total number of evaluated sliding-window tiles.
        """
        _, h, w = t0_chw.shape
        windows = generate_tile_windows(
            height=h,
            width=w,
            tile_size=self.tile_size,
            overlap_ratio=self.overlap_ratio,
        )
        total_tiles = len(windows)

        # Accumulator maps at original scene resolution
        prob_accum = np.zeros((h, w), dtype=np.float32)
        weight_accum = np.zeros((h, w), dtype=np.float32)

        # Process in batches
        for batch_start in range(0, total_tiles, self.batch_size):
            batch_windows = windows[batch_start : batch_start + self.batch_size]
            b_t0 = []
            b_t1 = []

            for y, x, th, tw in batch_windows:
                p0 = extract_tile_with_padding(t0_chw, y, x, th, tw, target_size=self.tile_size)
                p1 = extract_tile_with_padding(t1_chw, y, x, th, tw, target_size=self.tile_size)
                b_t0.append(p0)
                b_t1.append(p1)

            t0_tensor = torch.from_numpy(np.stack(b_t0)).to(self.device)
            t1_tensor = torch.from_numpy(np.stack(b_t1)).to(self.device)

            with torch.inference_mode():
                logits = model(t0_tensor, t1_tensor)
                probs = F.softmax(logits, dim=1)
                # Class 1 is 'change'
                change_probs = probs[:, 1].cpu().numpy()  # (B, tile_size, tile_size)

            # Reconstruct into scene
            for idx, (y, x, th, tw) in enumerate(batch_windows):
                tile_p = change_probs[idx, :th, :tw]
                tile_w = self.blend_window[:th, :tw]

                prob_accum[y : y + th, x : x + tw] += tile_p * tile_w
                weight_accum[y : y + th, x : x + tw] += tile_w

            # Cleanup batch tensors
            del t0_tensor, t1_tensor, logits, probs, change_probs
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Normalize accumulated weighted probabilities
        reconstructed = prob_accum / np.maximum(weight_accum, 1e-6)
        reconstructed = np.clip(reconstructed, 0.0, 1.0)
        return reconstructed, total_tiles
