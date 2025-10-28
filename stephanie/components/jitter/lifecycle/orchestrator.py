# stephanie/components/jitter/lifecycle/orchestrator.py
"""
orchestrator.py
===============
Thin, GAP-style orchestrator for Jitter.

This orchestrator:
- Registers required services (best-effort)
- Loads/merges JAS config
- Constructs and runs the lifecycle engine
- Ensures VPM plugins are properly registered
- Exposes start/stop and single entrypoint `execute(context)`

Key Features:
- Clean pipeline integration with start/stop controls
- Graceful shutdown with legacy preservation
- Configuration-driven operation
- Integration with Stephanie's agent ecosystem
- Comprehensive telemetry and monitoring
- Service registration and dependency management
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List
import logging
import time
import asyncio
import uuid
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from enum import Enum

from stephanie.components.jitter.boundary import Membrane, BoundaryMaintenance
from stephanie.components.jitter.cognition.attention_manager import AttentionManager
from stephanie.components.jitter.cognition.sense_making import SenseMakingEngine
from stephanie.components.jitter.cognition.triune import TriuneCognition
from stephanie.components.jitter.core.energy import EnergyPools
from stephanie.components.jitter.lifecycle.lifecycle_agent import JASLifecycleAgent
from stephanie.components.jitter.metabolism import MetabolicPathways, EnergyOptimizer
from stephanie.components.jitter.production.closed_production import ProductionNetwork
from stephanie.components.jitter.regulation.homeostasis.controller import PIDController
from stephanie.components.jitter.regulation.homeostasis.adaptive_setpoints import AdaptiveSetpoints
from stephanie.components.jitter.regulation.homeostasis.crisis_detector import CrisisDetector
from stephanie.components.jitter.regulation.apoptosis import ApoptosisSystem
from stephanie.components.jitter.regulation.reproduction.reproduction_system import ReproductionSystem
from stephanie.components.jitter.regulation.reproduction.quality_control import QualityControlledReproduction
from stephanie.components.jitter.regulation.reproduction.heritage_manager import HeritageManager

log = logging.getLogger("stephanie.jitter.lifecycle.orchestrator")

class OrchestratorConfig(BaseModel):
    """Validated configuration for JASOrchestrator"""
    name: str = Field("jitter_orchestrator", description="Name of the orchestrator")
    tick_interval: float = Field(1.0, ge=0.1, le=10.0, description="Time between ticks in seconds")
    enable_reproduction: bool = Field(True, description="Whether reproduction is enabled")
    reproduction_energy_threshold: float = Field(80.0, ge=0.0, le=100.0, description="Energy threshold for reproduction")
    max_history: int = Field(1000, ge=100, le=10000, description="Maximum history length")
    
    @validator('reproduction_energy_threshold')
    def validate_energy_threshold(cls, v):
        if v < 0 or v > 100:
            raise ValueError('reproduction_energy_threshold must be between 0 and 100')
        return v

class ServiceStatus(str, Enum):
    """Status of service registration"""
    REGISTERED = "registered"
    MISSING = "missing"
    FAILED = "failed"

@dataclass
class ServiceRegistration:
    """Service registration status"""
    name: str
    status: ServiceStatus
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

class JASOrchestrator:
    """
    Thin, GAP-style orchestrator for Jitter.
    
    This orchestrator:
    - Registers required services (best-effort)
    - Loads/merges JAS config
    - Constructs and runs the lifecycle engine
    - Ensures VPM plugins are properly registered
    - Exposes start/stop and single entrypoint `execute(context)`
    
    Key Features:
    - Clean pipeline integration with start/stop controls
    - Graceful shutdown with legacy preservation
    - Configuration-driven operation
    - Integration with Stephanie's agent ecosystem
    - Comprehensive telemetry and monitoring
    - Service registration and dependency management
    """
    
    def __init__(self, cfg: Dict[str, Any], container, memory=None, logger=None):
        self.cfg = cfg
        self.container = container
        self.memory = memory
        self.logger = logger or logging.getLogger("stephanie.jitter.lifecycle.orchestrator")
        self.lifecycle_agent = None
        self._running = False
        self.service_registry: Dict[str, ServiceRegistration] = {}
        self._initialized = False
        
        # Validate configuration
        try:
            self.config = OrchestratorConfig(**cfg)
            self.logger.info("JASOrchestrator configuration validated successfully")
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            self.config = OrchestratorConfig()
        
        log.info("JASOrchestrator initialized")
    
    def setup_services(self):
        """Ensure all required services are properly registered"""
        self.logger.info("Setting up JAS services...")
        
        # 1. Register core components with the container
        self._register_core_components()
        
        # 2. Register JAS-specific services
        self._register_jas_services()
        
        # 3. Verify all services are available
        self._verify_services()
        
        self._initialized = True
        self.logger.info("JAS services setup complete")
    
    def _register_core_components(self):
        """Register core system components"""
        # Register memory manager if available
        if self.memory and hasattr(self.memory, 'vpm_manager'):
            self.container.register(
                name="vpm_manager",
                factory=lambda: self.memory.vpm_manager,
                dependencies=[],
                init_args={}
            )
            self.service_registry["vpm_manager"] = ServiceRegistration(
                name="vpm_manager",
                status=ServiceStatus.REGISTERED
            )
        
        # Register memory if available
        if self.memory:
            self.container.register(
                name="memory",
                factory=lambda: self.memory,
                dependencies=[],
                init_args={}
            )
            self.service_registry["memory"] = ServiceRegistration(
                name="memory",
                status=ServiceStatus.REGISTERED
            )
    
    def _register_jas_services(self):
        """Register JAS-specific services with the container"""
        # 1. Production network
        self.container.register(
            name="jas_production",
            factory=lambda: ProductionNetwork(cfg=self.cfg.get("production", {})),
            dependencies=[],
            init_args={}
        )
        self.service_registry["jas_production"] = ServiceRegistration(
            name="jas_production",
            status=ServiceStatus.REGISTERED
        )
        
        # 2. Sense-making engine
        self.container.register(
            name="jas_sense_making",
            factory=lambda: SenseMakingEngine(cfg=self.cfg.get("sense_making", {})),
            dependencies=[],
            init_args={}
        )
        self.service_registry["jas_sense_making"] = ServiceRegistration(
            name="jas_sense_making",
            status=ServiceStatus.REGISTERED
        )
        
        # 3. Triune cognition
        self.container.register(
            name="jas_triune",
            factory=lambda: TriuneCognition(
                cfg=self.cfg.get("triune", {}),
                sense_making=self.container.get("jas_sense_making"),
                production=self.container.get("jas_production"),
                coupling=self.container.get("jas_coupling", None) or None
            ),
            dependencies=["jas_sense_making", "jas_production"],
            init_args={}
        )
        self.service_registry["jas_triune"] = ServiceRegistration(
            name="jas_triune",
            status=ServiceStatus.REGISTERED
        )
        
        # 4. Attention manager
        self.container.register(
            name="jas_attention",
            factory=lambda: AttentionManager(
                cfg=self.cfg.get("attention", {}),
                initial_weights={
                    "reptilian": 0.3,
                    "mammalian": 0.3,
                    "primate": 0.4
                }
            ),
            dependencies=[],
            init_args={}
        )
        self.service_registry["jas_attention"] = ServiceRegistration(
            name="jas_attention",
            status=ServiceStatus.REGISTERED
        )
        
        # 5. Membrane system
        self.container.register(
            name="jas_membrane",
            factory=lambda: Membrane(cfg=self.cfg.get("membrane", {})),
            dependencies=[],
            init_args={}
        )
        self.service_registry["jas_membrane"] = ServiceRegistration(
            name="jas_membrane",
            status=ServiceStatus.REGISTERED
        )
        
        # 6. Boundary maintenance
        self.container.register(
            name="jas_boundary_maintenance",
            factory=lambda: BoundaryMaintenance(
                cfg=self.cfg.get("boundary_maintenance", {}),
                membrane=self.container.get("jas_membrane")
            ),
            dependencies=["jas_membrane"],
            init_args={}
        )
        self.service_registry["jas_boundary_maintenance"] = ServiceRegistration(
            name="jas_boundary_maintenance",
            status=ServiceStatus.REGISTERED
        )
        
        # 7. Energy pools
        self.container.register(
            name="jas_energy",
            factory=lambda: EnergyPools(cfg=self.cfg.get("energy", {})),
            dependencies=[],
            init_args={}
        )
        self.service_registry["jas_energy"] = ServiceRegistration(
            name="jas_energy",
            status=ServiceStatus.REGISTERED
        )
        
        # 8. Metabolic pathways
        self.container.register(
            name="jas_metabolic_pathways",
            factory=lambda: MetabolicPathways(
                cfg=self.cfg.get("metabolic_pathways", {}),
                energy_pools=self.container.get("jas_energy")
            ),
            dependencies=["jas_energy"],
            init_args={}
        )
        self.service_registry["jas_metabolic_pathways"] = ServiceRegistration(
            name="jas_metabolic_pathways",
            status=ServiceStatus.REGISTERED
        )
        
        # 9. Energy optimizer
        self.container.register(
            name="jas_energy_optimizer",
            factory=lambda: EnergyOptimizer(
                cfg=self.cfg.get("energy_optimizer", {}),
                energy_pools=self.container.get("jas_energy"),
                metabolic_pathways=self.container.get("jas_metabolic_pathways")
            ),
            dependencies=["jas_energy", "jas_metabolic_pathways"],
            init_args={}
        )
        self.service_registry["jas_energy_optimizer"] = ServiceRegistration(
            name="jas_energy_optimizer",
            status=ServiceStatus.REGISTERED
        )
        
        # 10. Homeostasis controllers
        self.container.register(
            name="jas_pid_controller",
            factory=lambda: PIDController(cfg=self.cfg.get("homeostasis", {}).get("pid", {})),
            dependencies=[],
            init_args={}
        )
        self.service_registry["jas_pid_controller"] = ServiceRegistration(
            name="jas_pid_controller",
            status=ServiceStatus.REGISTERED
        )
        
        # 11. Adaptive setpoints
        self.container.register(
            name="jas_adaptive_setpoints",
            factory=lambda: AdaptiveSetpoints(cfg=self.cfg.get("homeostasis", {}).get("adaptive", {})),
            dependencies=[],
            init_args={}
        )
        self.service_registry["jas_adaptive_setpoints"] = ServiceRegistration(
            name="jas_adaptive_setpoints",
            status=ServiceStatus.REGISTERED
        )
        
        # 12. Crisis detector
        self.container.register(
            name="jas_crisis_detector",
            factory=lambda: CrisisDetector(cfg=self.cfg.get("homeostasis", {}).get("crisis", {})),
            dependencies=[],
            init_args={}
        )
        self.service_registry["jas_crisis_detector"] = ServiceRegistration(
            name="jas_crisis_detector",
            status=ServiceStatus.REGISTERED
        )
        
        # 13. Apoptosis system
        self.container.register(
            name="jas_apoptosis",
            factory=lambda: ApoptosisSystem(cfg=self.cfg.get("apoptosis", {})),
            dependencies=[],
            init_args={}
        )
        self.service_registry["jas_apoptosis"] = ServiceRegistration(
            name="jas_apoptosis",
            status=ServiceStatus.REGISTERED
        )
        
        # 14. Reproduction system
        self.container.register(
            name="jas_reproduction",
            factory=lambda: ReproductionSystem(cfg=self.cfg.get("reproduction", {})),
            dependencies=[],
            init_args={}
        )
        self.service_registry["jas_reproduction"] = ServiceRegistration(
            name="jas_reproduction",
            status=ServiceStatus.REGISTERED
        )
        
        # 15. Quality-controlled reproduction
        self.container.register(
            name="jas_quality_control",
            factory=lambda: QualityControlledReproduction(cfg=self.cfg.get("reproduction", {}).get("quality", {})),
            dependencies=[],
            init_args={}
        )
        self.service_registry["jas_quality_control"] = ServiceRegistration(
            name="jas_quality_control",
            status=ServiceStatus.REGISTERED
        )
        
        # 16. Heritage manager
        self.container.register(
            name="jas_heritage",
            factory=lambda: HeritageManager(cfg=self.cfg.get("reproduction", {}).get("heritage", {})),
            dependencies=[],
            init_args={}
        )
        self.service_registry["jas_heritage"] = ServiceRegistration(
            name="jas_heritage",
            status=ServiceStatus.REGISTERED
        )
    
    def _verify_services(self):
        """Verify that all required services are available"""
        # Check that all services are registered
        required_services = [
            "jas_production",
            "jas_sense_making",
            "jas_triune",
            "jas_attention",
            "jas_membrane",
            "jas_boundary_maintenance",
            "jas_energy",
            "jas_metabolic_pathways",
            "jas_energy_optimizer",
            "jas_pid_controller",
            "jas_adaptive_setpoints",
            "jas_crisis_detector",
            "jas_apoptosis",
            "jas_reproduction",
            "jas_quality_control",
            "jas_heritage"
        ]
        
        for service in required_services:
            if service not in self.container._services:
                self.service_registry[service] = ServiceRegistration(
                    name=service,
                    status=ServiceStatus.MISSING,
                    error=f"Service {service} not registered"
                )
                self.logger.warning(f"Service {service} not found in container")
            else:
                self.service_registry[service] = ServiceRegistration(
                    name=service,
                    status=ServiceStatus.REGISTERED
                )
    
    def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the Jitter lifecycle with proper service integration.
        
        This method:
        - Sets up services including plugin registration
        - Creates lifecycle agent with proper container reference
        - Initializes and runs the JAS lifecycle
        - Handles shutdown and cleanup
        
        Args:
            context: Context dictionary for pipeline integration
            
        Returns:
            Dictionary with execution results and metrics
        """
        try:
            # Setup services including plugin registration
            self.setup_services()
            
            # Create lifecycle agent with proper container reference
            self.lifecycle_agent = JASLifecycleAgent(
                cfg=self.cfg,
                container=self.container,
                memory=self.memory,
                logger=self.logger
            )
            
            # Initialize and run
            if not self.lifecycle_agent.initialize():
                return {"status": "error", "message": "Initialization failed"}
                
            self._running = True
            return self.lifecycle_agent.run(context or {})
            
        except Exception as e:
            self.logger.error(f"JAS execution failed: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    def stop(self):
        """Stop the JAS lifecycle gracefully"""
        if self._running and self.lifecycle_agent:
            self.lifecycle_agent.running = False
            self._running = False
            self.logger.info("JAS lifecycle stopped")
    
    def get_service_status(self) -> Dict[str, ServiceRegistration]:
        """Get current service registration status"""
        return self.service_registry.copy()
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics from all services"""
        metrics = {}
        
        # Get metrics from each service if available
        for service_name, registration in self.service_registry.items():
            if registration.status == ServiceStatus.REGISTERED:
                try:
                    service = self.container.get(service_name)
                    if hasattr(service, 'get_metrics'):
                        service_metrics = service.get_metrics()
                        metrics[service_name] = service_metrics
                    elif hasattr(service, 'get_ssp_integration_metrics'):
                        integration_metrics = service.get_ssp_integration_metrics()
                        metrics[f"{service_name}_integration"] = integration_metrics
                except Exception as e:
                    self.logger.warning(f"Failed to get metrics from {service_name}: {str(e)}")
        
        return metrics

# Utility function for service