# stephanie/analysis/vpm_differential_analyzer.py
from __future__ import annotations

import os
from pathlib import Path
from tkinter import Image

import matplotlib
import numpy as np
from zeromodel.vpm.logic import vpm_add, vpm_and, vpm_subtract, vpm_xor

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


class VPMDifferentialAnalyzer:
    def __init__(self, output_dir="vpm_analysis"):
        Path(output_dir).mkdir(exist_ok=True)
        self.output_dir = output_dir

    def analyze(self, vpm_good, vpm_bad, prefix="diff"):
        diff = vpm_subtract(vpm_good, vpm_bad)
        overlap = vpm_and(vpm_good, vpm_bad)
        contrast = vpm_xor(vpm_good, vpm_bad)
        enriched = vpm_add(diff, overlap * 0.25)
        self.save_vpm_image(diff, "Unique (Good - Bad)", f"{self.output_dir}/{prefix}_unique.png")
        self.save_vpm_image(contrast, "Contrast (Good XOR Bad)", f"{self.output_dir}/{prefix}_contrast.png")
        self.save_vpm_image(enriched, "Enriched Knowledge", f"{self.output_dir}/{prefix}_enriched.png")
        return {"diff": diff, "contrast": contrast, "enriched": enriched}

    def save_vpm_image(self, vpm, title: str, filename: str):
        """Save VPM as image with proper handling of both array and PIL Image types."""
        # Convert to normalized array for consistent processing
        arr = _to_normalized_array(vpm)
        
        # Handle 3D arrays (RGB) by converting to grayscale if needed
        if arr.ndim == 3:
            if arr.shape[2] == 3:
                # Convert RGB to grayscale
                arr = 0.2989 * arr[:,:,0] + 0.5870 * arr[:,:,1] + 0.1140 * arr[:,:,2]
            else:
                # Take first channel
                arr = arr[:,:,0]
        
        # Create and save image
        plt.figure(figsize=(6, 6))
        plt.imshow(arr, cmap='gray', vmin=0, vmax=1)
        plt.title(title)
        plt.colorbar(label='Normalized Score')
        plt.xlabel('Metrics (sorted)')
        plt.ylabel('Documents (sorted)')

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved VPM image: {filepath}")

def _to_normalized_array(obj):
    """Convert PIL Image or numpy array to normalized float32 array in [0,1] range."""
    if isinstance(obj, Image.Image):
        # Convert PIL Image to numpy array
        arr = np.array(obj.convert("RGB"))
        # Normalize to [0,1] range
        if arr.dtype != np.float32:
            if np.issubdtype(arr.dtype, np.integer):
                max_val = np.iinfo(arr.dtype).max
                arr = arr.astype(np.float32) / max_val
            else:
                arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
        return arr
    elif isinstance(obj, np.ndarray):
        return np.clip(obj.astype(np.float32), 0.0, 1.0)
    else:
        raise TypeError(f"Expected PIL.Image or numpy.ndarray, got {type(obj)}")
