"""Utilities for computing and applying weight deltas between models"""

import hashlib
import json
import os
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch


def compute_weight_delta(
    sd_after: OrderedDict, 
    sd_before: OrderedDict,
    output_dir: str,
    id_prefix: str = "skill"
) -> Dict[str, Any]:
    """
    Compute parameter-space delta vector v = θ_after - θ_before
    
    Args:
        sd_after: State dict after training
        sd_before: State dict before training
        output_dir: Directory to save delta
        id_prefix: Prefix for skill ID
        
    Returns:
        Dict with metadata about the skill filter
    """
    # Compute delta
    v = OrderedDict()
    for k, w in sd_after.items():
        if k not in sd_before: 
            continue
        if w.shape != sd_before[k].shape:
            continue
        v[k] = (w - sd_before[k]).cpu()
    
    # Generate skill ID (hash of key properties)
    param_hash = hashlib.sha256(str(list(v.keys())[:10]).encode()).hexdigest()[:8]
    size_mb = sum(p.numel() * p.element_size() for p in v.values()) / (1024 * 1024)
    skill_id = f"{id_prefix}_{param_hash}_{size_mb:.1f}mb"
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    delta_path = f"{output_dir}/{skill_id}.pt"
    torch.save(v, delta_path)
    
    return {
        "id": skill_id,
        "weight_delta_path": delta_path,
        "weight_size_mb": size_mb,
        "num_parameters": len(v)
    }

def apply_weight_delta(
    model,
    weight_delta_path: str,
    intensity: float = 1.0,
    safety_threshold: float = 0.15
) -> Any:
    """
    Apply weight delta to a model with safety checks
    
    Args:
        model: Base model to enhance
        weight_delta_path: Path to .pt file with weight delta
        intensity: Strength of application (0.0-1.0)
        safety_threshold: Max relative parameter change allowed
        
    Returns:
        Enhanced model (same type as input)
    """
    # Create deep copy to avoid modifying original
    import copy
    enhanced = copy.deepcopy(model)
    
    # Load delta
    v = torch.load(weight_delta_path, map_location="cpu")
    
    # Apply delta to parameters
    params = dict(enhanced.named_parameters())
    unstable_layers = []
    
    for name, delta in v.items():
        if name not in params:
            continue
            
        # Convert delta to model's device and dtype
        delta_tensor = delta.to(params[name].device).to(params[name].dtype)
        
        # Safety check: prevent excessive changes
        abs_w = torch.abs(params[name].data)
        max_change = safety_threshold * (abs_w.mean().item() + 1e-8)
        actual_change = torch.abs(delta_tensor).mean().item()
        
        if actual_change > max_change:
            # Scale down delta to safe level
            safe_intensity = safety_threshold * abs_w.mean() / (torch.abs(delta_tensor).mean() + 1e-8)
            delta_tensor = safe_intensity * delta_tensor
            unstable_layers.append((name, actual_change/max_change))
        
        # Apply with intensity scaling
        params[name].data += intensity * delta_tensor
    
    # Log warnings for unstable layers
    if unstable_layers:
        max_ratio = max(ratio for _, ratio in unstable_layers)
        print(
            f"⚠️ WARNING: Applied skill with safety scaling: "
            f"{len(unstable_layers)}/{len(v)} layers adjusted "
            f"(max={max_ratio:.2f}x threshold)"
        )
    
    return enhanced