"""
jaf.py
======
Jitter Artifact Format v0 - standard format for legacy preservation and reproduction.

This implementation:
- Defines a standardized artifact format for Jitter state
- Handles serialization/deserialization
- Includes versioning for forward compatibility
- Provides methods for ingestion by offspring
- Supports structured data for analysis and reproduction
"""

import json
import time
import random
import torch
import torch.nn as nn

import numpy as np
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Dict, Any, List, Optional, Union
from enum import Enum

class JAFVersion(str, Enum):
    """Supported JAF versions for backward compatibility"""
    V0 = "jaf/0"
    V1 = "jaf/1"

@dataclass
class CognitiveSnapshot:
    """Snapshot of cognitive state for JAF"""
    reptilian: float
    mammalian: float
    primate: float
    integrated: float
    cognitive_energy: float
    attention_weights: Dict[str, float]
    layer_veto: str
    threat_level: float
    emotional_valence: float
    reasoning_depth: int
    latency_ms: float

@dataclass
class EnergySnapshot:
    """Snapshot of energy state for JAF"""
    cognitive: float
    metabolic: float
    reserve: float
    cognitive_in: float
    to_metabolic: float
    maintenance: float
    to_reserve: float

@dataclass
class BoundarySnapshot:
    """Snapshot of boundary state for JAF"""
    integrity: float
    thickness: float
    stress: float
    permeability: float

@dataclass
class HealthMetrics:
    """Health metrics for JAF"""
    stability: float
    efficiency: float
    balance: float
    veto_frequency: Dict[str, float]
    crisis_count: int
    last_crisis: Optional[float] = None

@dataclass
class JitterArtifactV0:
    """Jitter Artifact Format v0 - standardized legacy artifact"""
    spec: str = JAFVersion.V0
    artifact_id: str = ""
    organism_id: str = ""
    parent_id: str = ""
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    tick: int = 0
    cause: str = ""  # For apoptosis: reason for termination
    
    # Core state snapshots
    final_vitals: Dict[str, Any] = field(default_factory=dict)
    cognitive_snapshot: Optional[CognitiveSnapshot] = None
    energy_snapshot: Optional[EnergySnapshot] = None
    boundary_snapshot: Optional[BoundarySnapshot] = None
    health_metrics: Optional[HealthMetrics] = None
    
    # Lineage and reproduction data
    lineage: Dict[str, Any] = field(default_factory=dict)
    reproduction_opportunity: bool = False
    variation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Cognitive history for context
    cognition_trace: List[Dict[str, Any]] = field(default_factory=list)
    recent_vitals: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        def serialize(obj):
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(item) for item in obj]
            elif is_dataclass(obj) and not isinstance(obj, type):
                return {k: serialize(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, (np.ndarray, torch.Tensor)):
                # Convert numpy/tensor to list
                if isinstance(obj, torch.Tensor):
                    obj = obj.detach().cpu().numpy()
                return obj.tolist()
            else:
                return str(obj)
                
        return serialize(asdict(self))
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for transmission/storage"""
        return json.dumps(self.to_dict()).encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'JitterArtifactV0':
        """Deserialize from bytes"""
        try:
            return cls.from_dict(json.loads(data.decode('utf-8')))
        except Exception as e:
            raise ValueError(f"Failed to parse JAF artifact: {str(e)}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JitterArtifactV0':
        """Create from dictionary with validation"""
        # Validate spec version
        if data.get('spec', '') != JAFVersion.V0:
            raise ValueError(f"Unsupported JAF version: {data.get('spec', 'unknown')}")
        
        # Convert nested structures
        if 'cognitive_snapshot' in data and data['cognitive_snapshot']:
            data['cognitive_snapshot'] = CognitiveSnapshot(**data['cognitive_snapshot'])
        if 'energy_snapshot' in data and data['energy_snapshot']:
            data['energy_snapshot'] = EnergySnapshot(**data['energy_snapshot'])
        if 'boundary_snapshot' in data and data['boundary_snapshot']:
            data['boundary_snapshot'] = BoundarySnapshot(**data['boundary_snapshot'])
        if 'health_metrics' in data and data['health_metrics']:
            data['health_metrics'] = HealthMetrics(**data['health_metrics'])
        
        return cls(**data)
    
    def validate(self) -> bool:
        """Validate artifact integrity"""
        # Basic structural validation
        if self.spec != JAFVersion.V0:
            return False
            
        # Validate core metrics are in expected ranges
        if self.cognitive_snapshot:
            if not (0 <= self.cognitive_snapshot.integrated <= 1):
                return False
            if self.cognitive_snapshot.cognitive_energy < 0:
                return False
                
        if self.energy_snapshot:
            for pool in ['cognitive', 'metabolic', 'reserve']:
                if getattr(self.energy_snapshot, pool, -1) < 0:
                    return False
                    
        if self.boundary_snapshot:
            if not (0 <= self.boundary_snapshot.integrity <= 1):
                return False
            if not (0 <= self.boundary_snapshot.thickness <= 1):
                return False
                
        return True

def create_artifact_from_state(
    core,
    triune,
    homeostasis,
    telemetry,
    cause: str = "",
    reproduction_opportunity: bool = False
) -> JitterArtifactV0:
    """Create a JAF artifact from current system state"""
    # Get most recent vital signs
    latest_vitals = telemetry.history[-1] if telemetry.history else None
    
    # Get most recent cognitive state
    cognitive_state = triune.state_history[-1] if triune.state_history else None
    
    # Create cognitive snapshot
    cognitive_snapshot = None
    if cognitive_state:
        cognitive_snapshot = CognitiveSnapshot(
            reptilian=cognitive_state.reptilian,
            mammalian=cognitive_state.mammalian,
            primate=cognitive_state.primate,
            integrated=cognitive_state.integrated,
            cognitive_energy=cognitive_state.cognitive_energy,
            attention_weights=cognitive_state.attention_weights,
            layer_veto=cognitive_state.layer_veto,
            threat_level=cognitive_state.threat_level,
            emotional_valence=cognitive_state.emotional_valence,
            reasoning_depth=cognitive_state.reasoning_depth,
            latency_ms=cognitive_state.latency_ms
        )
    
    # Create energy snapshot
    energy_snapshot = EnergySnapshot(
        cognitive=core.energy.level("cognitive"),
        metabolic=core.energy.level("metabolic"),
        reserve=core.energy.level("reserve"),
        cognitive_in=cognitive_state.cognitive_energy_in if cognitive_state else 0.0,
        to_metabolic=cognitive_state.cognitive_energy_to_metabolic if cognitive_state else 0.0,
        maintenance=cognitive_state.maintenance_cost if cognitive_state else 0.0,
        to_reserve=cognitive_state.reserve_transfer if cognitive_state else 0.0
    )
    
    # Create boundary snapshot
    boundary_snapshot = BoundarySnapshot(
        integrity=core.membrane.integrity,
        thickness=core.membrane.thickness,
        stress=core.membrane.last_stress if hasattr(core.membrane, 'last_stress') else 0.0,
        permeability=core.membrane.permeability if hasattr(core.membrane, 'permeability') else 0.5
    )
    
    # Get health metrics
    health_metrics = triune.get_health_metrics()
    crisis_count = sum(1 for v in telemetry.history if v.crisis_level > 0.5)
    
    # Create cognition trace (last 5 states)
    cognition_trace = []
    for state in triune.state_history[-5:]:
        cognition_trace.append({
            "tick": state.tick,
            "integrated": state.integrated,
            "layer_veto": state.layer_veto,
            "threat_level": state.threat_level,
            "emotional_valence": state.emotional_valence
        })
    
    # Create recent vitals (last 10)
    recent_vitals = []
    for vital in telemetry.history[-10:]:
        recent_vitals.append({
            "tick": vital.tick,
            "health_score": vital.health_score,
            "boundary_integrity": vital.boundary_integrity,
            "energy_cognitive": vital.energy_cognitive,
            "energy_metabolic": vital.energy_metabolic
        })
    
    return JitterArtifactV0(
        organism_id=getattr(core, 'id', f"jas_{int(time.time())}"),
        parent_id=getattr(core, 'parent_id', ''),
        generation=getattr(core, 'generation', 0),
        tick=getattr(core, 'tick', 0),
        cause=cause,
        cognitive_snapshot=cognitive_snapshot,
        energy_snapshot=energy_snapshot,
        boundary_snapshot=boundary_snapshot,
        health_metrics=HealthMetrics(
            stability=health_metrics["stability"],
            efficiency=health_metrics["efficiency"],
            balance=health_metrics["balance"],
            veto_frequency=health_metrics["veto_frequency"],
            crisis_count=crisis_count
        ),
        reproduction_opportunity=reproduction_opportunity,
        cognition_trace=cognition_trace,
        recent_vitals=recent_vitals
    )

def ingest_artifact(artifact: JitterArtifactV0, core, triune, homeostasis):
    """Ingest a JAF artifact to initialize or tune a new Jitter instance"""
    if not artifact.validate():
        raise ValueError("Invalid JAF artifact")
    
    # Initialize core state from artifact
    if artifact.energy_snapshot:
        core.energy.energy_pools = {
            "cognitive": artifact.energy_snapshot.cognitive,
            "metabolic": artifact.energy_snapshot.metabolic,
            "reserve": artifact.energy_snapshot.reserve
        }
    
    if artifact.boundary_snapshot:
        core.membrane.integrity = artifact.boundary_snapshot.integrity
        core.membrane.thickness = artifact.boundary_snapshot.thickness
    
    # Initialize triune cognition
    if artifact.cognitive_snapshot:
        # Apply variation based on reproduction parameters
        variation_rate = artifact.variation_params.get("variation_rate", 0.05)
        
        # Slightly vary attention weights
        attention_weights = [
            artifact.cognitive_snapshot.attention_weights["reptilian"],
            artifact.cognitive_snapshot.attention_weights["mammalian"],
            artifact.cognitive_snapshot.attention_weights["primate"]
        ]
        
        # Apply variation
        for i in range(3):
            attention_weights[i] *= (1 + random.uniform(-variation_rate, variation_rate))
        
        # Normalize
        total = sum(attention_weights)
        for i in range(3):
            attention_weights[i] /= total
            
        # Update triune attention
        triune.attention_weights = nn.Parameter(torch.tensor(attention_weights))
    
    # Initialize homeostasis setpoints
    if artifact.health_metrics:
        # Update adaptive setpoints based on legacy
        homeostasis.adaptive_setpoints = {
            "energy_balance": homeostasis.adaptive_setpoints["energy_balance"] * 0.7 + 
                             (1.0 / (1.0 + artifact.health_metrics["veto_frequency"].get("reptilian", 0.3))) * 0.3,
            "boundary_integrity": max(0.2, min(0.95, artifact.boundary_snapshot.integrity * 0.9)),
            "cognitive_flow": homeostasis.adaptive_setpoints["cognitive_flow"] * 0.8 + 
                             artifact.health_metrics["efficiency"] * 0.2,
            "vpm_diversity": max(0.2, min(0.95, artifact.health_metrics["balance"]))
        }
        
        # Adjust PID parameters based on legacy performance
        if artifact.health_metrics["stability"] > 0.7:
            # Good stability - reduce gains to avoid overcorrection
            homeostasis.controllers["energy_balance"].kp *= 0.9
            homeostasis.controllers["boundary_integrity"].kp *= 0.9
        elif artifact.health_metrics["stability"] < 0.3:
            # Poor stability - increase gains
            homeostasis.controllers["energy_balance"].kp *= 1.1
            homeostasis.controllers["boundary_integrity"].kp *= 1.1
    
    # Initialize VPM store from cognition trace
    if artifact.cognition_trace and core.vpm_manager:
        for trace in artifact.cognition_trace:
            # Convert trace to VPM content
            content = f"Legacy trace from generation {artifact.generation}: {trace['layer_veto']} layer active with threat {trace['threat_level']:.2f}"
            core.vpm_manager.create(content, source="legacy")