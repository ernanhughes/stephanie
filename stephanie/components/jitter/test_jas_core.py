# tests/components/jitter/test_jas_core.py
"""
Test suite for Jitter Autopoietic System core components
"""

import pytest
import torch
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from stephanie.components.jitter.jas_core import (
    AutopoieticCore,
    Membrane,
    EnergyMetabolism,
    ReproductionSystem,
    HomeostasisController
)
from stephanie.memory.vpm_manager import VPMManager

from stephanie.components.jitter.apoptosis import ApoptosisSystem
from stephanie.components.jitter.jas_lifecycle_agent import JASLifecycleAgent 
from stephanie.components.jitter.jas_triune import TriuneCognition
from stephanie.components.jitter.telemetry.jas_telemetry import JASTelemetry, VitalSigns
from stephanie.components.jitter.jas_homeostasis import EnhancedHomeostasis

@pytest.fixture
def mock_cfg():
    """Basic configuration for testing"""
    return {
        "membrane": {
            "thickness": 1.0,
            "initial_integrity": 0.5
        },
        "energy_metabolism": {
            "initial_cognitive": 50.0,
            "initial_metabolic": 50.0,
            "initial_reserve": 10.0
        },
        "reproduction": {
            "ready_threshold": 80.0,
            "variation_rate": 0.1
        },
        "apoptosis": {
            "boundary_threshold": 0.1,
            "energy_threshold": 1.0,
            "max_crisis": 10
        },
        "homeostasis": {
            "energy_balance": 1.0,
            "vpm_diversity": 0.6,
            "boundary_integrity": 0.8,
            "pid_kp": 1.0,
            "pid_ki": 0.1,
            "pid_kd": 0.05
        }
    }

@pytest.fixture
def mock_ebt():
    """Mock EBT model for testing"""
    mock = MagicMock()
    mock.score = MagicMock(return_value=0.5)
    return mock

@pytest.fixture
def mock_vpm_manager():
    """Mock VPM manager for testing"""
    mock = MagicMock(spec=VPMManager)
    mock.diversity_score = MagicMock(return_value=0.6)
    mock.count = MagicMock(return_value=100)
    mock.get_relevant_vpms = MagicMock(return_value=[])
    mock.get_high_value_vpms = MagicMock(return_value=[])
    return mock

def test_membrane_initialization(mock_cfg):
    """Test membrane initialization"""
    membrane = Membrane(mock_cfg["membrane"])
    assert membrane.thickness == 1.0
    assert 0.0 <= membrane.integrity <= 1.0
    assert isinstance(membrane.permeability_nn, torch.nn.Module)

def test_membrane_update(mock_cfg, mock_ebt):
    """Test membrane update functionality"""
    membrane = Membrane(mock_cfg["membrane"])
    membrane.ebt_model = mock_ebt
    
    # Create test embedding
    emb = torch.randn(1, 1024)
    
    # Initial integrity
    initial_integrity = membrane.integrity
    
    # Update membrane
    stress = membrane.update(emb, metabolic_energy=50.0)
    
    # Check integrity changed appropriately
    assert 0.0 <= membrane.integrity <= 1.0
    assert membrane.integrity != initial_integrity
    
    # Check stress calculation
    assert 0.0 <= stress <= 1.0
    mock_ebt.score.assert_called_once()

def test_membrane_permeability(mock_cfg, mock_ebt):
    """Test membrane permeability logic"""
    membrane = Membrane(mock_cfg["membrane"])
    membrane.ebt_model = mock_ebt
    
    # Create test embedding
    emb = torch.randn(1, 1024)
    
    # Test permeability when boundary is strong
    membrane.integrity = 0.8
    assert membrane.is_permeable(emb) == True
    
    # Test permeability when boundary is weak
    membrane.integrity = 0.2
    assert membrane.is_permeable(emb) == True  # Should still be permeable but with higher stress
    
    # Test permeability when boundary is critically weak
    membrane.integrity = 0.1
    membrane.thickness = 0.1
    assert membrane.is_permeable(emb) == False  # Boundary too weak to maintain identity

def test_energy_metabolism_initialization(mock_cfg):
    """Test energy metabolism initialization"""
    energy = EnergyMetabolism(mock_cfg["energy_metabolism"])
    
    # Check energy pools initialized correctly
    assert energy.level("cognitive") == 50.0
    assert energy.level("metabolic") == 50.0
    assert energy.level("reserve") == 10.0
    
    # Check pathway networks are proper modules
    assert isinstance(energy.pathways["cognitive_to_metabolic"], torch.nn.Module)
    assert isinstance(energy.pathways["metabolic_to_reserve"], torch.nn.Module)

def test_energy_processing(mock_cfg, mock_ebt, mock_vpm_manager):
    """Test energy processing workflow"""
    energy = EnergyMetabolism(mock_cfg["energy_metabolism"])
    membrane = Membrane(mock_cfg["membrane"])
    
    # Create test embedding
    emb = torch.randn(1, 1024)
    
    # Process energy
    energy_value = energy.process(emb, membrane)
    
    # Check energy value is reasonable
    assert 0.0 <= energy_value <= 2.0
    
    # Check energy pools updated
    assert energy.level("cognitive") >= 50.0
    assert energy.level("metabolic") <= 50.0
    assert energy.level("reserve") >= 10.0
    
    # Check maintenance costs applied
    energy._run_maintenance(membrane)
    assert energy.level("cognitive") < 50.0 + energy_value
    assert energy.level("metabolic") < 50.0

def test_energy_aliveness(mock_cfg):
    """Test energy aliveness detection"""
    energy = EnergyMetabolism(mock_cfg["energy_metabolism"])
    assert energy.alive() == True
    
    # Deplete cognitive energy
    energy.energy_pools["cognitive"] = 0.0
    assert energy.alive() == False
    
    # Reset and deplete metabolic energy
    energy.energy_pools["cognitive"] = 50.0
    energy.energy_pools["metabolic"] = 0.0
    assert energy.alive() == False

def test_autopoietic_core_cycle(mock_cfg, mock_ebt, mock_vpm_manager):
    """Test complete autopoietic cycle"""
    core = AutopoieticCore(mock_cfg, mock_ebt, mock_vpm_manager)
    
    # Create test embedding
    emb = torch.randn(1, 1024)
    
    # Run cycle
    result = core.cycle(emb)
    
    # Check result structure
    assert "status" in result
    assert "tick" in result
    assert "integrity" in result
    assert "energy_pools" in result
    
    # Check status
    assert result["status"] == "alive"
    
    # Check tick incremented
    assert result["tick"] == 1
    
    # Run multiple cycles
    for _ in range(10):
        result = core.cycle(emb)
    
    # Check tick incremented properly
    assert result["tick"] == 11
    
    # Check energy decreasing over time (simulating metabolic cost)
    initial_energy = sum(mock_cfg["energy_metabolism"].values())
    final_energy = sum(result["energy_pools"].values())
    assert final_energy < initial_energy * 1.5  # Allow for some energy gain from processing

def test_reproduction_system(mock_cfg, mock_vpm_manager):
    """Test reproduction system functionality"""
    reproduction = ReproductionSystem(mock_cfg["reproduction"])
    
    # Mock core system
    class MockCore:
        def __init__(self):
            self.energy = MagicMock()
            self.energy.level = lambda x: 90.0 if x == "reserve" else 50.0
            self.membrane = MagicMock()
            self.membrane.integrity = 0.8
            self.generation = 1
            self.id = "mock_core"
    
    core = MockCore()
    
    # Test reproduction readiness
    assert reproduction.ready(core) == True
    
    # Test offspring creation
    offspring = reproduction.create_offspring(core)
    assert offspring is not None
    assert "vpm_store" in offspring
    assert "energy_state" in offspring
    assert "membrane_state" in offspring
    assert "generation" in offspring
    assert offspring["generation"] == 2
    
    # Test variation application
    state = {
        "vpm_store": [{"content": "test"}],
        "membrane_state": {"thickness": 1.0},
        "homeostasis_params": {"param": 1.0}
    }
    mutated = reproduction._apply_variation(state)
    assert mutated["vpm_store"][0]["content"] == "test"  # Content should remain
    assert mutated["membrane_state"]["thickness"] != 1.0  # Thickness should vary slightly
    assert mutated["homeostasis_params"]["param"] != 1.0  # Params should vary slightly

def test_apoptosis_system(mock_cfg):
    """Test apoptosis system functionality"""
    apoptosis = ApoptosisSystem(mock_cfg["apoptosis"])
    
    # Mock core system
    class MockCore:
        def __init__(self, metabolic=50.0, cognitive=50.0):
            self.energy = MagicMock()
            self.energy.level = lambda x: metabolic if x == "metabolic" else cognitive
            self.membrane = MagicMock()
            self.membrane.integrity = 0.5
    
    # Test normal conditions
    core = MockCore()
    assert apoptosis.check_viability(core) == False
    
    # Test boundary failure
    core.membrane.integrity = 0.05
    assert apoptosis.check_viability(core) == True
    
    # Test energy depletion
    core = MockCore(metabolic=0.5, cognitive=0.5)
    assert apoptosis.check_viability(core) == True
    
    # Test crisis counter
    core = MockCore(metabolic=5.0, cognitive=5.0)
    core.membrane.integrity = 0.3
    
    # Should not trigger immediately
    for _ in range(mock_cfg["apoptosis"]["max_crisis"] - 1):
        assert apoptosis.check_viability(core) == False
    
    # Should trigger on next call
    assert apoptosis.check_viability(core) == True

def test_homeostasis_controller(mock_cfg):
    """Test homeostasis controller functionality"""
    homeostasis = HomeostasisController(mock_cfg["homeostasis"])
    
    # Mock core system
    class MockCore:
        def __init__(self):
            self.energy = MagicMock()
            self.energy.level = lambda x: 50.0
            self.vpm_manager = MagicMock()
            self.vpm_manager.diversity_score = lambda: 0.6
            self.membrane = MagicMock()
            self.membrane.integrity = 0.8
    
    core = MockCore()
    
    # Test regulation
    corrections = homeostasis.regulate(core)
    
    # Check corrections structure
    assert "energy_balance" in corrections
    assert "vpm_diversity" in corrections
    assert "boundary_integrity" in corrections
    
    # Check correction values are within expected range
    for dim, value in corrections.items():
        assert -0.5 <= value <= 0.5

@pytest.mark.asyncio
async def test_lifecycle_agent(mock_cfg, mock_ebt, mock_vpm_manager):
    """Test complete lifecycle agent operation"""
    # Mock memory system
    memory = MagicMock()
    memory.get_model = MagicMock(return_value=mock_ebt)
    memory.vpm_manager = mock_vpm_manager
    
    # Create lifecycle agent
    agent = JASLifecycleAgent(mock_cfg, memory)
    
    # Test initialization
    assert await agent.initialize() == True
    
    # Mock the tick function
    agent._tick = AsyncMock()
    
    # Run agent for a few ticks
    task = asyncio.create_task(agent.run({}))
    await asyncio.sleep(0.1)
    agent.running = False
    await task
    
    # Check tick was called
    assert agent._tick.called
    
    # Test shutdown
    await agent._shutdown()

@pytest.mark.asyncio
async def test_reproduction_integration(mock_cfg, mock_ebt, mock_vpm_manager):
    """Test reproduction integration with lifecycle agent"""
    # Mock memory system
    memory = MagicMock()
    memory.get_model = MagicMock(return_value=mock_ebt)
    memory.vpm_manager = mock_vpm_manager
    memory.store_reproduction_data = MagicMock()
    memory.log_event = MagicMock()
    
    # Create lifecycle agent
    agent = JASLifecycleAgent(mock_cfg, memory)
    assert await agent.initialize() == True
    
    # Mock core energy to trigger reproduction
    agent.core.energy.energy_pools["reserve"] = 90.0
    
    # Run enough ticks to trigger reproduction
    agent.tick = agent.reproduction_interval
    await agent._tick()
    
    # Check reproduction was triggered
    assert agent.reproduction_ready == True
    assert memory.store_reproduction_data.called
    assert memory.log_event.called

def test_triune_cognition(mock_cfg, mock_ebt, mock_vpm_manager):
    """Test triune cognition system"""
    # Mock models
    class MockModel:
        def score(self, emb):
            return torch.tensor([0.5])
        def energy(self, emb):
            return torch.tensor([0.5])
        def q_value(self, emb):
            return torch.tensor([0.5])
    
    mrq = MockModel()
    svm = MockModel()
    
    # Create triune cognition
    triune = TriuneCognition(mock_cfg["triune"], mrq, mock_ebt, svm, mock_vpm_manager)
    
    # Create test embedding
    emb = torch.randn(1, 1024)
    
    # Process through triune cognition
    state = triune(emb)
    
    # Check state structure
    assert hasattr(state, "reptilian")
    assert hasattr(state, "mammalian")
    assert hasattr(state, "primate")
    assert hasattr(state, "integrated")
    assert hasattr(state, "cognitive_energy")
    
    # Check values are in expected range
    assert 0.0 <= state.reptilian <= 1.0
    assert 0.0 <= state.mammalian <= 1.0
    assert 0.0 <= state.primate <= 1.0
    assert 0.0 <= state.integrated <= 1.0
    assert state.cognitive_energy >= 0.0
    
    # Test veto power
    # Force high reptilian threat
    with patch.object(triune, '_process_reptilian', return_value=torch.tensor([0.8])):
        state = triune(emb)
        assert state.layer_veto == "reptilian"
        assert state.integrated == 0.8
    
    # Force negative emotional valence
    with patch.object(triune, '_process_mammalian', return_value=(0.5, -0.7)):
        state = triune(emb)
        assert state.layer_veto == "mammalian"
        assert state.integrated == 0.5

def test_homeostasis_telemetry(mock_cfg):
    """Test homeostasis telemetry system"""
    homeostasis = EnhancedHomeostasis(mock_cfg["homeostasis"])
    
    # Mock core system
    class MockCore:
        def __init__(self):
            self.energy = MagicMock()
            self.energy.level = lambda x: 50.0
            self.vpm_manager = MagicMock()
            self.vpm_manager.diversity_score = lambda: 0.6
            self.membrane = MagicMock()
            self.membrane.integrity = 0.8
    
    core = MockCore()
    
    # Run regulation to generate telemetry
    homeostasis.regulate(core)
    
    # Get telemetry
    telemetry = homeostasis.get_telemetry()
    
    # Check telemetry structure
    assert "health" in telemetry
    assert "crisis_level" in telemetry
    assert "stability" in telemetry
    assert "trend" in telemetry
    assert "setpoints" in telemetry
    assert "regulatory_actions" in telemetry
    assert "measurements" in telemetry
    
    # Check health is in expected range
    assert 0.0 <= telemetry["health"] <= 1.0

def test_telemetry_system(mock_cfg):
    """Test telemetry system functionality"""
    telemetry = JASTelemetry(mock_cfg["telemetry"])
    
    # Mock JAS components
    class MockCore:
        def __init__(self):
            self.membrane = MagicMock()
            self.membrane.integrity = 0.8
            self.energy = MagicMock()
            self.energy.level = lambda x: 50.0
            self.tick = 10
            self.vpm_manager = MagicMock()
            self.vpm_manager.count = lambda: 100
            self.vpm_manager.diversity_score = lambda: 0.6
    
    class MockHomeostasis:
        def get_telemetry(self):
            return {
                "health": 0.8,
                "crisis_level": 0.2,
                "homeostatic_error": 0.1,
                "regulatory_actions": {"energy_balance": 0.1},
                "setpoints": {"energy_balance": 1.0}
            }
    
    class MockTriune:
        state_history = [
            MagicMock(
                reptilian=0.3, mammalian=0.4, primate=0.7, integrated=0.5,
                cognitive_energy=0.2, attention_weights={"reptilian": 0.3, "mammalian": 0.3, "primate": 0.4},
                layer_veto="none", threat_level=0.3, emotional_valence=0.1, reasoning_depth=3
            )
        ]
    
    core = MockCore()
    homeostasis = MockHomeostasis()
    triune = MockTriune()
    
    # Collect vital signs
    vital_signs = telemetry.collect(core, homeostasis, triune)
    
    # Check vital signs structure
    assert vital_signs.boundary_integrity == 0.8
    assert vital_signs.energy_cognitive == 50.0
    assert vital_signs.vpm_count == 100
    assert vital_signs.health_score == 0.8
    assert vital_signs.crisis_level == 0.2
    
    # Check alert detection
    alerts = telemetry._detect_alerts(core, homeostasis, triune, homeostasis.get_telemetry())
    assert len(alerts) == 0
    
    # Force alert conditions
    core.energy.level = lambda x: 2.0 if x == "metabolic" else 50.0
    alerts = telemetry._detect_alerts(core, homeostasis, triune, homeostasis.get_telemetry())
    assert "metabolic_low" in alerts
    
    # Test health trend
    health_trend = telemetry.get_health_trend()
    assert "current_health" in health_trend
    assert "trend" in health_trend
    assert "stability" in health_trend

@pytest.mark.asyncio
async def test_lifecycle_agent_full(mock_cfg, mock_ebt, mock_vpm_manager):
    """Test full lifecycle agent operation with all components"""
    # Mock memory system
    memory = MagicMock()
    memory.get_model = MagicMock(return_value=mock_ebt)
    memory.vpm_manager = mock_vpm_manager
    memory.store_reproduction_data = MagicMock()
    memory.log_event = MagicMock()
    
    # Create lifecycle agent
    agent = JASLifecycleAgent(mock_cfg, memory)
    
    # Test initialization
    assert await agent.initialize() == True
    
    # Check components initialized
    assert agent.core is not None
    assert agent.triune is not None
    assert agent.homeostasis is not None
    assert agent.telemetry is not None
    
    # Mock the sensory input
    agent._get_sensory_input = AsyncMock(return_value=torch.randn(1, 1024))
    
    # Run agent for a few ticks
    agent.running = True
    await agent._tick()
    
    # Check telemetry collected
    assert len(agent.telemetry.history) == 1
    
    # Check reproduction not triggered yet
    assert agent.reproduction_ready == False
    
    # Mock energy to trigger reproduction
    agent.core.energy.energy_pools["reserve"] = 90.0
    agent.tick = agent.reproduction_interval
    
    # Run another tick
    await agent._tick()
    
    # Check reproduction triggered
    assert agent.reproduction_ready == True
    assert memory.store_reproduction_data.called
    
    # Test shutdown
    await agent._shutdown()
