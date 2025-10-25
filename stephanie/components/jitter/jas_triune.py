"""
TriuneCognition
===============
The cognitive nervous system of the Jitter organism, implementing the three-layer
biological model where each layer has veto power over the others.

Integration Points:
- Reptilian Core: Uses EBT for boundary threat assessment
- Mammalian Layer: Leverages SVM for pattern recognition
- Primate Cortex: Employs MRQ for abstract reasoning

The system implements true biological veto power: lower layers can override
higher layers when survival is at stake.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

log = logging.getLogger("stephanie.jas.triune")

@dataclass
class CognitiveState:
    """Complete cognitive state snapshot for telemetry and reproduction"""
    reptilian: float  # Boundary integrity assessment (0-1)
    mammalian: float  # Pattern recognition confidence (0-1)
    primate: float    # Abstract reasoning quality (0-1)
    integrated: float # Final cognitive output (0-1)
    cognitive_energy: float  # Energy extracted from cognitive process
    attention_weights: Dict[str, float]  # Current attention allocation
    layer_veto: str   # Which layer has veto power (if any)
    latency_ms: float # Processing time
    threat_level: float  # Current boundary threat (0-1)
    emotional_valence: float  # Mammalian layer emotional signal (-1 to 1)
    reasoning_depth: int    # Primate cortex reasoning steps taken

class TriuneCognition(nn.Module):
    """
    The complete triune cognitive architecture with biological veto power cascade.
    
    Key Features:
    - Layer-specific processing with different neural architectures
    - Dynamic attention allocation based on context
    - Biological veto power (lower layers override higher ones)
    - Energy-based cognitive efficiency measurement
    - Continuous state history for reproduction system
    """
    
    def __init__(self, cfg: Dict[str, Any], container, memory, logger):
        super().__init__()
        self.cfg = cfg
        self.container = container
        self.memory = memory
        self.logger = logger or log


        # Layer thresholds for veto power
        self.veto_thresholds = {
            "reptilian": cfg.get("reptilian_veto_threshold", 0.7),
            "mammalian": cfg.get("mammalian_veto_threshold", 0.6)
        }
        
        # Dynamic attention weights (learned during operation)
        self.attention_weights = nn.Parameter(torch.tensor([
            cfg.get("reptilian_weight", 0.3),
            cfg.get("mammalian_weight", 0.3),
            cfg.get("primate_weight", 0.4)
        ]))
        
        # Energy extraction network (converts cognitive output to metabolic energy)
        self.energy_extractor = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()
        )
        
        # Layer-specific processing networks
        self.reptilian_net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Tanh()
        )
        
        self.mammalian_net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Tanh()
        )
        
        self.primate_net = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=3
        )
        
        # State history for reproduction and learning
        self.state_history: List[CognitiveState] = []
        self.max_history = cfg.get("max_state_history", 1000)
        
        log.info("TriuneCognition initialized with biological veto power cascade")

    def forward(self, input_emb: torch.Tensor) -> CognitiveState:
        """ No no keep it over keep it over OK OK input through all three cognitive layers with veto power cascade"""
        start_time = torch.tensor(time.time())
        
        try:
            # 1. Reptilian Core: Boundary threat assessment
            reptilian_out = self._process_reptilian(input_emb)
            threat_level = reptilian_out.item()
            
            # Check for reptilian veto (immediate boundary threats)
            if threat_level > self.veto_thresholds["reptilian"]:
                cognitive_state = self._create_veto_state(
                    "reptilian", threat_level, input_emb, start_time
                )
                self._record_state(cognitive_state)
                return cognitive_state
            
            # 2. Mammalian Layer: Pattern recognition and emotional valence
            mammalian_out = self._process_mammalian(input_emb)
            pattern_confidence, emotional_valence = mammalian_out
            
            # Check for mammalian veto (emotional valence threshold)
            if emotional_valence < -self.veto_thresholds["mammalian"]:
                cognitive_state = self._create_veto_state(
                    "mammalian", pattern_confidence, input_emb, start_time,
                    emotional_valence=emotional_valence
                )
                self._record_state(cognitive_state)
                return cognitive_state
            
            # 3. Primate Cortex: Abstract reasoning
            primate_out, reasoning_depth = self._process_primate(input_emb)
            
            # No veto - integrate all layers
            integrated = self._integrate_layers(
                threat_level, pattern_confidence, primate_out
            )
            
            # Extract cognitive energy
            cognitive_energy = self._extract_energy(
                threat_level, pattern_confidence, primate_out
            )
            
            # Create final cognitive state
            cognitive_state = CognitiveState(
                reptilian=threat_level,
                mammalian=pattern_confidence,
                primate=primate_out,
                integrated=integrated,
                cognitive_energy=cognitive_energy,
                attention_weights=self._get_attention_dict(),
                layer_veto="none",
                latency_ms=(time.time() - start_time.item()) * 1000,
                threat_level=threat_level,
                emotional_valence=emotional_valence,
                reasoning_depth=reasoning_depth
            )
            
            self._record_state(cognitive_state)
            return cognitive_state
            
        except Exception as e:
            log.error(f"TriuneCognition error: {str(e)}", exc_info=True)
            # Return safe fallback state
            return CognitiveState(
                reptilian=0.5, mammalian=0.5, primate=0.5, integrated=0.5,
                cognitive_energy=0.0, attention_weights=self._get_attention_dict(),
                layer_veto="error", latency_ms=0.0, threat_level=0.5,
                emotional_valence=0.0, reasoning_depth=0
            )

    def _process_reptilian(self, input_emb: torch.Tensor) -> torch.Tensor:
        """Process input through Reptilian Core (boundary threat assessment)"""
        try:
            # Use EBT to measure compatibility with core identity
            # Lower energy = better compatibility = lower threat
            threat_score = self.ebt.score("core_identity", input_emb)
            
            # Process through reptilian network
            processed = self.reptilian_net(threat_score)
            
            # Normalize to 0-1 (1 = high threat)
            return torch.sigmoid(processed)
            
        except Exception as e:
            log.warning(f"Reptilian processing error: {str(e)}")
            # Fallback: random threat assessment
            return torch.tensor([0.5])

    def _process_mammalian(self, input_emb: torch.Tensor) -> Tuple[float, float]:
        """Process input through Mammalian Layer (pattern recognition)"""
        try:
            # Use SVM for pattern recognition confidence
            pattern_confidence = self.svm.score(input_emb).item()
            
            # Determine emotional valence (negative = bad pattern)
            emotional_valence = self._determine_emotional_valence(input_emb)
            
            return pattern_confidence, emotional_valence
            
        except Exception as e:
            log.warning(f"Mammalian processing error: {str(e)}")
            return 0.5, 0.0

    def _determine_emotional_valence(self, input_emb: torch.Tensor) -> float:
        """Determine emotional valence of pattern (negative = bad)"""
        try:
            # Check against negative VPMs (learned bad patterns)
            negative_vpms = self.vpm_manager.get_negative_vpms()
            if not negative_vpms:
                return 0.0
                
            # Calculate similarity to negative patterns
            negative_similarity = self._calculate_similarity(
                input_emb, negative_vpms
            )
            
            # Check against positive VPMs
            positive_vpms = self.vpm_manager.get_positive_vpms()
            positive_similarity = self._calculate_similarity(
                input_emb, positive_vpms
            ) if positive_vpms else 0.0
            
            # Emotional valence = positive - negative
            return float(positive_similarity - negative_similarity)
            
        except Exception as e:
            log.warning(f"Emotional valence error: {str(e)}")
            return 0.0

    def _process_primate(self, input_emb: torch.Tensor) -> Tuple[float, int]:
        """Process input through Primate Cortex (abstract reasoning)"""
        try:
            # Get relevant VPMs for reasoning
            relevant_vpms = self.vpm_manager.get_relevant_vpms(
                input_emb, top_k=self.cfg.get("primate_top_k", 5)
            )
            
            if not relevant_vpms:
                return 0.5, 0
                
            # Convert to tensor for transformer
            vpm_tensors = torch.stack([v.embedding for v in relevant_vpms])
            
            # Process through transformer
            reasoning_output = self.primate_net(vpm_tensors)
            
            # Get reasoning depth (how many VPMs were chained)
            reasoning_depth = len(relevant_vpms)
            
            # Calculate reasoning quality
            reasoning_quality = self._calculate_reasoning_quality(
                reasoning_output, input_emb
            )
            
            return reasoning_quality, reasoning_depth
            
        except Exception as e:
            log.warning(f"Primate processing error: {str(e)}")
            return 0.5, 0

    def _integrate_layers(
        self, 
        reptilian: float, 
        mammalian: float, 
        primate: float
    ) -> float:
        """Integrate outputs from all three layers with attention weighting"""
        weights = torch.softmax(self.attention_weights, dim=0)
        return float(
            reptilian * weights[0] + 
            mammalian * weights[1] + 
            primate * weights[2]
        )

    def _extract_energy(
        self, 
        reptilian: float, 
        mammalian: float, 
        primate: float
    ) -> float:
        """Extract metabolic energy from cognitive processing"""
        # Stack layer outputs for energy network
        inputs = torch.tensor([[reptilian, mammalian, primate]], dtype=torch.float32)
        # Get energy value (0-1 scale)
        energy = self.energy_extractor(inputs).item()
        # Scale by configuration parameter
        return energy * self.cfg.get("energy_gain_factor", 1.0)

    def _create_veto_state(
        self,
        veto_layer: str,
        primary_value: float,
        input_emb: torch.Tensor,
        start_time: torch.Tensor,
        emotional_valence: float = 0.0
    ) -> CognitiveState:
        """Create cognitive state when a layer has veto power"""
        # Set all values to primary_value for veto layer
        values = {
            "reptilian": primary_value if veto_layer == "reptilian" else 0.0,
            "mammalian": primary_value if veto_layer == "mammalian" else 0.0,
            "primate": 0.0  # Primate never has veto power
        }
        
        # For reptilian veto, threat_level = primary_value
        # For mammalian veto, emotional_valence = primary_value
        threat_level = primary_value if veto_layer == "reptilian" else 0.5
        emotional_valence = emotional_valence if veto_layer == "mammalian" else 0.0
        
        return CognitiveState(
            reptilian=values["reptilian"],
            mammalian=values["mammalian"],
            primate=values["primate"],
            integrated=primary_value,
            cognitive_energy=0.0,  # No energy gain during veto
            attention_weights=self._get_attention_dict(),
            layer_veto=veto_layer,
            latency_ms=(time.time() - start_time.item()) * 1000,
            threat_level=threat_level,
            emotional_valence=emotional_valence,
            reasoning_depth=0
        )

    def _record_state(self, state: CognitiveState):
        """Record cognitive state for history and reproduction"""
        self.state_history.append(state)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)

    def _get_attention_dict(self) -> Dict[str, float]:
        """Get attention weights as dictionary"""
        weights = torch.softmax(self.attention_weights, dim=0)
        return {
            "reptilian": weights[0].item(),
            "mammalian": weights[1].item(),
            "primate": weights[2].item()
        }

    def _calculate_similarity(
        self, 
        query_emb: torch.Tensor, 
        vpm_list: List
    ) -> float:
        """Calculate average similarity to a list of VPMs"""
        if not vpm_list:
            return 0.0
            
        similarities = []
        for vpm in vpm_list:
            # Cosine similarity
            sim = torch.nn.functional.cosine_similarity(
                query_emb, vpm.embedding.unsqueeze(0)
            ).item()
            similarities.append(sim)
            
        return float(np.mean(similarities))

    def _calculate_reasoning_quality(
        self, 
        reasoning_output: torch.Tensor, 
        input_emb: torch.Tensor
    ) -> float:
        """Calculate quality of reasoning output"""
        try:
            # Compare reasoning output to input for coherence
            coherence = torch.nn.functional.cosine_similarity(
                reasoning_output.mean(dim=0), input_emb
            ).item()
            
            # Apply sigmoid to get 0-1 range
            return float(torch.sigmoid(torch.tensor(coherence)).item())
            
        except Exception as e:
            log.warning(f"Reasoning quality error: {str(e)}")
            return 0.5

    def get_recent_states(self, n: int = 10) -> List[CognitiveState]:
        """Get recent cognitive states for reproduction system"""
        return self.state_history[-n:] if self.state_history else []

    def update_attention(self, reward_signal: float):
        """Update attention weights based on reward signal"""
        with torch.no_grad():
            # Simple reinforcement: increase weights for layers that contributed
            # to positive outcomes
            self.attention_weights += reward_signal * 0.01
            # Ensure weights sum to 1
            self.attention_weights = torch.softmax(self.attention_weights, dim=0)

    def get_health_metrics(self) -> Dict[str, float]:
        """Get health metrics from cognitive state history"""
        if not self.state_history:
            return {
                "stability": 0.5,
                "efficiency": 0.5,
                "balance": 0.5,
                "veto_frequency": {"reptilian": 0.0, "mammalian": 0.0}
            }
        
        # Get last 50 states for metrics
        recent = self.state_history[-50:]
        
        # Stability: variance in integrated cognitive output
        integrated_values = [s.integrated for s in recent]
        stability = 1.0 / (1.0 + np.var(integrated_values))
        
        # Efficiency: average cognitive energy per tick
        energy_values = [s.cognitive_energy for s in recent]
        efficiency = float(np.mean(energy_values)) if energy_values else 0.5
        
        # Balance: how evenly attention is distributed
        attention_values = [s.attention_weights for s in recent]
        avg_attention = {
            k: np.mean([a[k] for a in attention_values]) 
            for k in attention_values[0].keys()
        }
        balance = 1.0 - np.std(list(avg_attention.values())) * 3
        
        # Veto frequency
        veto_counts = {"reptilian": 0, "mammalian": 0}
        for s in recent:
            if s.layer_veto == "reptilian":
                veto_counts["reptilian"] += 1
            elif s.layer_veto == "mammalian":
                veto_counts["mammalian"] += 1
                
        veto_freq = {
            k: v / len(recent) for k, v in veto_counts.items()
        }
        
        return {
            "stability": float(stability),
            "efficiency": efficiency,
            "balance": float(balance),
            "veto_frequency": veto_freq
        }
