
import os
import torch
import json
from datetime import datetime

def get_model_version(session, model_type: str, target_type: str, dimension: str):
    return "v1"

def get_model_path(model_path, model_type: str, target_type: str, dimension: str, version: str = "v1"):
    return f"{model_path}/{model_type}/{target_type}/{dimension}/{version}/"

def discover_saved_dimensions(model_type: str, target_type: str, model_dir: str = "models") -> list:
    """
    Discover saved dimensions for a given model and target type.
    Filters out scalers and metadata artifacts.
    """
    path = os.path.join(model_dir, model_type, target_type)
    if not os.path.exists(path):
        print(f"[discover_saved_dimensions] Path {path} does not exist.")
        return []

    dimension_names = set()

    for filename in os.listdir(path):
        # Ignore scalers, tuners, and meta
        if any(ex in filename for ex in ["_scaler", ".tuner", ".meta", ".json"]):
            continue

        # Match patterns for EBT, MRQ, SVM
        if filename.endswith(".pt") or filename.endswith(".joblib"):
            # Extract base name (e.g., alignment_v1.pt -> alignment)
            base = filename.split("_v")[0]  # remove _v1 suffix
            base = base.replace(".joblib", "").replace(".pt", "")
            dimension_names.add(base)

    return sorted(dimension_names)

def get_svm_file_paths(model_path, model_type, target_type, dim, model_version="v1"):
    base = get_model_path(model_path, model_type, target_type, dim, model_version)
    return {
        "model": base + f"{dim}.joblib",
        "scaler": base + f"{dim}_scaler.joblib",
        "tuner": base + f"{dim}.tuner.json",
        "meta": base + f"{dim}.meta.json"
    }

def save_model_with_version(
    model_state: dict, 
    model_type: str, 
    target_type: str, 
    dimension: str,
    version: str
):
    """Save a model with versioned metadata"""
    version_path = get_model_path(model_type, target_type, dimension, version)
    os.makedirs(version_path, exist_ok=True)
    
    # Save model state
    torch.save(model_state, os.path.join(version_path, "model.pt"))
    
    # Save metadata
    metadata = {
        "model_type": model_type,
        "target_type": target_type,
        "dimension": dimension,
        "version": version,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    with open(os.path.join(version_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
        
    return version_path