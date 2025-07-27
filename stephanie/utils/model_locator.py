# stephanie/utils/model_locator.py
import json
import os
from pathlib import Path
from typing import Dict

import torch
from joblib import load

from stephanie.models.incontext_q_model import InContextQModel
from stephanie.scoring.model.ebt_model import EBTModel
from stephanie.scoring.model.text_encoder import TextEncoder
from stephanie.scoring.mrq.model import MRQModel
from stephanie.scoring.mrq.value_predictor import ValuePredictor


class ModelLocator:
    def __init__(
        self,
        root_dir: str = "models",
        embedding_type: str = "default",
        model_type: str = "mrq",
        target_type: str = "document",
        dimension: str = "alignment",
        version: str = "v1",
        variant: str = None  # e.g., "sicql"
    ):
        self.root_dir = root_dir
        self.embedding_type = embedding_type
        self.model_type = model_type
        self.target_type = target_type
        self.dimension = dimension
        self.version = version
        self.variant = variant
        self._scaler = None
        self._tuner = None

    @property
    def base_path(self) -> str:
        """Build hierarchical path: models/embedding_type/model_type/target_type/dimension/version"""
        path = os.path.join(
            self.root_dir,
            self.embedding_type,
            self.model_type,
            self.target_type,
            self.dimension,
            self.version
        )
        return os.path.join(path, self.variant) if self.variant else path

    # Model-specific paths
    def model_file(self, suffix: str = ".pt") -> str:
        return os.path.join(self.base_path, f"{self.dimension}{suffix}")

    def encoder_file(self) -> str:
        return os.path.join(self.base_path, f"{self.dimension}_encoder.pt")

    def q_head_file(self) -> str:
        return os.path.join(self.base_path, f"{self.dimension}_q.pt")

    def v_head_file(self) -> str:
        return os.path.join(self.base_path, f"{self.dimension}_v.pt")

    def pi_head_file(self) -> str:
        return os.path.join(self.base_path, f"{self.dimension}_pi.pt")

    def meta_file(self) -> str:
        return os.path.join(self.base_path, f"{self.dimension}.meta.json")

    def tuner_file(self) -> str:
        return os.path.join(self.base_path, f"{self.dimension}.tuner.json")

    def scaler_file(self) -> str:
        return os.path.join(self.base_path, f"{self.dimension}_scaler.joblib")

    # Component paths
    def all_files(self) -> Dict[str, str]:
        """Get paths for all model components"""
        return {
            "encoder": self.encoder_file(),
            "q_head": self.get_q_head_path(),
            "v_head": self.get_v_head_path(),
            "pi_head": self.get_pi_head_path(),
            "meta": self.meta_file(),
            "tuner": self.tuner_file(),
            "scaler": self.scaler_file() if self.model_type == "svm" else None
        }

    def ensure_dirs(self):
        """Ensure all directories exist for this model"""
        Path(self.base_path).mkdir(parents=True, exist_ok=True)
        return self.base_path

    # Model loading
    def load_sicql_model(self, device="cpu"):
        """Load SICQL model with Q/V/π heads"""
        return InContextQModel.load_from_path(
            self.base_path, 
            self.dimension,
            device=device
        )

    def load_ebt_model(self, device="cpu"):
        """Load EBT model with Q/V/π heads"""
        try:
            # Load meta first
            meta_path = self.meta_file()
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Meta file not found: {meta_path}")
                
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                
            # Build model
            model = EBTModel(
                embedding_dim=meta["dim"],
                hidden_dim=meta["hdim"],
                num_actions=meta.get("num_actions", 3),
                device=device
            ).to(device)
            
            # Load weights
            model.encoder.load_state_dict(
                torch.load(self.encoder_file(), map_location=device)
            )
            model.q_head.load_state_dict(
                torch.load(self.q_head_file(), map_location=device)
            )
            model.v_head.load_state_dict(
                torch.load(self.v_head_file(), map_location=device)
            )
            model.pi_head.load_state_dict(
                torch.load(self.pi_head_file(), map_location=device)
            )
            
            model.eval()
            return model, meta
            
        except Exception as e:
            raise RuntimeError(f"Failed to load EBT model: {e}")

    def load_tuner(self):
        """Load regression tuner if available"""
        tuner_path = self.tuner_file()
        if os.path.exists(tuner_path):
            with open(tuner_path, "r") as f:
                self._tuner = json.load(f)
        return self._tuner

    def load_scaler(self):
        """Load feature scaler for SVM models"""
        scaler_path = self.scaler_file()
        if os.path.exists(scaler_path):
            self._scaler = load(scaler_path)
        return self._scaler

    # Model discovery
    @staticmethod
    def list_available_models(root_dir="models") -> list:
        """List all available models with full path components"""
        available = []
        for embedding in os.listdir(root_dir):
            embedding_path = os.path.join(root_dir, embedding)
            if not os.path.isdir(embedding_path):
                continue
                
            for model_type in os.listdir(embedding_path):
                type_path = os.path.join(embedding_path, model_type)
                for target_type in os.listdir(type_path):
                    target_path = os.path.join(type_path, target_type)
                    for dimension in os.listdir(target_path):
                        dim_path = os.path.join(target_path, dimension)
                        for version in os.listdir(dim_path):
                            if not version.startswith("v"):
                                continue
                            model_path = os.path.join(dim_path, version)
                            available.append({
                                "embedding_type": embedding,
                                "model_type": model_type,
                                "target_type": target_type,
                                "dimension": dimension,
                                "version": version,
                                "path": model_path,
                                "has_components": ModelLocator._check_model_components(model_path, model_type)
                            })
        return available

    @staticmethod
    def _check_model_components(model_path, model_type):
        """Check if all required model components exist"""
        required_files = {
            "mrq": ["encoder.pt", "predictor.pt", "meta.json"],
            "ebt": ["encoder.pt", "q_head.pt", "v_head.pt", "pi_head.pt", "meta.json"],
            "sicql": ["encoder.pt", "q.pt", "v.pt", "pi.pt", "meta.json"],
            "svm": ["scaler.joblib", "model.joblib", "meta.json"]
        }
        
        if model_type not in required_files:
            return False
            
        return all(
            os.path.exists(os.path.join(model_path, f))
            for f in required_files[model_type]
        )

    @staticmethod
    def discover_dimensions(root_dir, embedding_type, model_type, target_type):
        """Discover dimensions with complete model files"""
        base = os.path.join(root_dir, embedding_type, model_type, target_type)
        if not os.path.exists(base):
            return []
        
        return [
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d)) and ModelLocator._has_trained_models(
                os.path.join(base, d)
            )
        ]

    @staticmethod
    def _has_trained_models(path: str) -> bool:
        """Check if dimension has trained models"""
        versions = [v for v in os.listdir(path) if os.path.isdir(os.path.join(path, v))]
        return any(
            ModelLocator._check_model_components(
                os.path.join(path, v), 
                model_type="ebt"
            ) for v in versions
        )

    @staticmethod
    def find_best_model_per_dimension(root_dir="models") -> dict:
        """Find latest version for each dimension"""
        model_paths = ModelLocator.list_available_models(root_dir)
        best = {}
        
        for info in model_paths:
            key = f"{info['model_type']}/{info['dimension']}"
            if key not in best or info["version"] > best[key]["version"]:
                best[key] = info
                
        return {f"{info['dimension']}/{info['model_type']}": info["path"] for info in best.values()}
    


    def load_mrq_model(self, device="cpu"):
        """Load MRQ model components and build MRQModel"""
        try:
            # Check for required files
            if not self._check_required_files(["encoder", "predictor"]):
                raise FileNotFoundError(f"Missing required model files in {self.base_path}")
            
            # Load metadata
            meta_path = self.meta_file()
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            else:
                meta = {
                    "dim": self.memory.embedding.dim,
                    "hdim": self.memory.embedding.hdim,
                    "version": self.version
                }
            
            # Build model components
            encoder = TextEncoder(dim=meta["dim"], hdim=meta["hdim"])
            predictor = ValuePredictor(zsa_dim=meta["dim"], hdim=meta["hdim"])
            
            # Load weights
            encoder.load_state_dict(
                torch.load(self.encoder_file(), map_location=device)
            )


            # Load state_dict
            state_dict = torch.load(self.model_file(), map_location=device)
            
            # Remap keys from v_head to value_net
            remapped_dict = {
                k.replace("v_head.net", "value_net"): v
                for k, v in state_dict.items()
                if k.startswith("v_head.net")
            }
            
            if not remapped_dict:
                raise ValueError("No relevant keys found for ValuePredictor")
            
            # Build ValuePredictor
            predictor = ValuePredictor(zsa_dim=meta["dim"], hdim=meta["hdim"]).to(device)
            predictor.load_state_dict(remapped_dict)
            predictor.eval()
        
            
            # Build MRQModel
            mrq_model = MRQModel(encoder, predictor, self.memory.embedding, device=device)
            mrq_model.eval()
            
            # Log successful load
            self.logger.log("MRQModelLoaded", {
                "dimension": self.dimension,
                "embedding_type": self.embedding_type,
                "model_path": self.base_path
            })
            
            return mrq_model, meta
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MRQ model: {e}")

    def _check_required_files(self, components):
        """Validate required model files exist"""
        required = {
            "encoder": self.encoder_file(),
            "predictor": self.model_file()
        }
        
        missing = [
            name for name, path in required.items() 
            if not os.path.exists(path)
        ]
        
        if missing:
            self.logger.log("MissingModelFiles", {
                "dimension": self.dimension,
                "missing_files": missing
            })
            return False
        return True
