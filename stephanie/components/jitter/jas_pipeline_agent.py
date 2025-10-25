"""
JASPipelineAgent - The entry point for running the Jitter Autopoietic System as part of a pipeline.

This agent provides:
- Clean pipeline integration with start/stop controls
- Graceful shutdown with legacy preservation
- Configuration-driven operation
- Integration with Stephanie's agent ecosystem
- Comprehensive telemetry and monitoring
- Support for both standalone and orchestrated execution

Usage:
    # As part of a pipeline
    from stephanie.agents.jas_pipeline_agent import JASPipelineAgent
    
    jas_agent = JASPipelineAgent(config_path="conf/agent/jas_pipeline.yaml")
    await jas_agent.start()
    
    # Run for a specific duration
    await asyncio.sleep(3600)  # Run for 1 hour
    
    # Or until a condition is met
    while not some_condition:
        await asyncio.sleep(1)
    
    await jas_agent.stop()

    # As a CLI command
    python -m stephanie.agents.jas_pipeline_agent --config conf/agent/jas_pipeline.yaml --duration 3600
"""

import asyncio
import logging
import time
import signal
import sys
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.jitter.jas_lifecycle_agent import JASLifecycleAgent
from stephanie.components.jitter.jaf import JitterArtifactV0

log = logging.getLogger("stephanie.agents.jas_pipeline")

@dataclass
class JASPipelineConfig:
    """Configuration for the JAS Pipeline Agent"""
    # Core operation parameters
    name: str = field(default="jas_pipeline_agent", metadata={"description": "Name of the JAS Pipeline Agent"})
    description: str = field(default="JAS Pipeline Agent for running Jitter Autopoietic System", metadata={"description": "Description of the agent"})
    duration: Optional[int] = field(default=None, metadata={"description": "Run duration in seconds (None = run until stopped)"})
    auto_start: bool = field(default=True, metadata={"description": "Start automatically when initialized"})
    graceful_shutdown_timeout: float = field(default=30.0, metadata={"description": "Max time for graceful shutdown"})
    
    # JAS configuration
    jas_config_path: str = field(default="conf/agent/jitter_v1.yaml", metadata={"description": "Path to JAS configuration"})
    jas_reproduction_enabled: bool = field(default=True, metadata={"description": "Enable reproduction system"})
    jas_max_runtime: Optional[int] = field(default=None, metadata={"description": "Maximum runtime in seconds"})
    
    # Integration parameters
    
    # Monitoring and telemetry
    telemetry_interval: float = field(default=1.0, metadata={"description": "Telemetry publish interval in seconds"})
    health_check_interval: float = field(default=5.0, metadata={"description": "Health check interval in seconds"})
    
    # Reproduction and legacy
    max_offspring: int = field(default=5, metadata={"description": "Maximum number of offspring to create"})
    legacy_preservation: bool = field(default=True, metadata={"description": "Preserve legacy on shutdown"})

class JASPipelineAgent(BaseAgent):
    """
    Pipeline-friendly agent for running the Jitter Autopoietic System.
    
    This agent serves as the primary entry point for integrating JAS into Stephanie's ecosystem.
    It handles initialization, execution, monitoring, and shutdown of the JAS lifecycle.
    
    Key Features:
    - Clean pipeline integration with start/stop controls
    - Graceful shutdown with legacy preservation
    - Configuration-driven operation
    - Integration with Stephanie's agent ecosystem
    - Comprehensive telemetry and monitoring
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        """
        Initialize the JAS Pipeline Agent.
        
        Args:
            cfg: Configuration dictionary or path to config file
            memory: Memory system for integration with Stephanie
            logger: Custom logger (optional)
        """
        super().__init__(cfg, memory, container, logger)
        
        # Parse configuration
        self.config = self._parse_config(cfg)
        
        # Initialize components
        self.jas_agent = None
        self.running = False
        self.start_time = None
        self.offspring_count = 0
        self.legacy_artifacts = []
        self.health_task = None
        self.telemetry_task = None
        
        log.info(f"JASPipelineAgent initialized with config: {self._config_summary()}")
    
    def _parse_config(self, cfg: Dict[str, Any]) -> JASPipelineConfig:
        """Parse and validate configuration"""
        if isinstance(cfg, dict):
            return JASPipelineConfig(**cfg)
        elif isinstance(cfg, str):
            # Load from file if string is provided
            try:
                import yaml
                with open(cfg, 'r') as f:
                    config_data = yaml.safe_load(f)
                return JASPipelineConfig(**config_data.get('jas_pipeline', {}))
            except Exception as e:
                log.error(f"Failed to load config from {cfg}: {str(e)}")
                return JASPipelineConfig()
        else:
            return JASPipelineConfig()
    
    def _config_summary(self) -> str:
        """Create a summary of the configuration for logging"""
        return (
            f"duration={self.config.duration}s, "
            f"auto_start={self.config.auto_start}, "
            f"max_offspring={self.config.max_offspring}, "
            f"reproduction={'enabled' if self.config.jas_reproduction_enabled else 'disabled'}"
        )
    
    async def initialize(self) -> bool:
        """Initialize the JAS system and required components"""
        try:
            log.info("Initializing JAS system components...")
            
            # Load JAS configuration
            jas_cfg = self._load_jas_config()
            
            # Configure reproduction
            jas_cfg.setdefault("reproduction", {})["enable_reproduction"] = self.config.jas_reproduction_enabled
            
            
            # Create JAS lifecycle agent
            self.jas_agent = JASLifecycleAgent(
                cfg=jas_cfg,
                container=self.container,
                memory=self.memory,
                logger=self.logger
            )
            
            # Initialize JAS
            if not await self.jas_agent.initialize():
                log.error("Failed to initialize JAS lifecycle agent")
                return False
                
            log.info("JAS system initialized successfully")
            return True
            
        except Exception as e:
            log.error(f"JAS initialization failed: {str(e)}", exc_info=True)
            return False
    
    def _load_jas_config(self) -> Dict[str, Any]:
        """Load and merge JAS configuration"""
        try:
            import yaml
            with open(self.config.jas_config_path, 'r') as f:
                jas_cfg = yaml.safe_load(f)
            return jas_cfg
        except Exception as e:
            log.error(f"Failed to load JAS config from {self.config.jas_config_path}: {str(e)}")
            # Return minimal config
            return {
                "core": {},
                "triune": {},
                "homeostasis": {},
                "reproduction": {"enable_reproduction": self.config.jas_reproduction_enabled},
                "apoptosis": {},
                "telemetry": {
                    "interval": self.config.telemetry_interval
                }
            }
    
    async def run(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the JAS Pipeline Agent.
        
        This method:
        - Initializes the JAS system
        - Starts the lifecycle agent
        - Monitors health and performance
        - Handles shutdown conditions
        - Preserves legacy on termination
        
        Args:
            context: Optional context dictionary for pipeline integration
            
        Returns:
            Dict[str, Any]: Final status and metrics
        """
        log.info("Starting JAS Pipeline Agent execution")
        
        # Initialize JAS system
        if not await self.initialize():
            return {"status": "error", "message": "Initialization failed"}
        
        # Start the agent
        await self.start()
        
        # Wait for completion
        try:
            if self.config.duration:
                log.info(f"Running JAS for {self.config.duration} seconds")
                await asyncio.sleep(self.config.duration)
            else:
                log.info("Running JAS until explicitly stopped")
                while self.running:
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            log.info("JAS execution cancelled")
        finally:
            await self.stop()
        
        return self._get_final_status()
    
    async def start(self):
        """Start the JAS lifecycle agent"""
        if self.running:
            log.warning("JAS is already running")
            return
            
        self.running = True
        self.start_time = time.time()
        
        # Start JAS lifecycle
        self.jas_task = asyncio.create_task(self.jas_agent.run({}))
        
        # Start health monitoring
        if self.config.health_check_interval > 0:
            self.health_task = asyncio.create_task(self._health_monitor())
        
        # Start telemetry
        if self.config.telemetry_interval > 0:
            self.telemetry_task = asyncio.create_task(self._telemetry_monitor())
        
        log.info("JAS lifecycle started")
    
    async def stop(self):
        """Stop the JAS lifecycle agent gracefully"""
        if not self.running:
            return
            
        self.running = False
        log.info("Initiating graceful shutdown of JAS")
        
        # Cancel monitoring tasks
        if self.health_task:
            self.health_task.cancel()
            try:
                await self.health_task
            except asyncio.CancelledError:
                pass
        
        if self.telemetry_task:
            self.telemetry_task.cancel()
            try:
                await self.telemetry_task
            except asyncio.CancelledError:
                pass
        
        # Stop JAS lifecycle
        if self.jas_agent:
            try:
                # Create a timeout for graceful shutdown
                stop_task = asyncio.create_task(self.jas_agent.stop())
                await asyncio.wait_for(
                    stop_task,
                    timeout=self.config.graceful_shutdown_timeout
                )
                log.info("JAS stopped gracefully")
            except asyncio.TimeoutError:
                log.warning(f"JAS shutdown timed out after {self.config.graceful_shutdown_timeout}s")
                # Force stop
                if hasattr(self.jas_agent, 'running'):
                    self.jas_agent.running = False
            except Exception as e:
                log.error(f"Error during JAS shutdown: {str(e)}")
        
        # Preserve legacy if configured
        if self.config.legacy_preservation:
            await self._preserve_legacy()
        
        log.info("JAS Pipeline Agent shutdown complete")
    
    async def _health_monitor(self):
        """Monitor JAS health and respond to issues"""
        while self.running:
            try:
                if not self.jas_agent or not hasattr(self.jas_agent, 'telemetry'):
                    await asyncio.sleep(self.config.health_check_interval)
                    continue
                
                # Get latest telemetry
                if self.jas_agent.telemetry and self.jas_agent.telemetry.history:
                    latest = self.jas_agent.telemetry.history[-1]
                    health = latest.get('health_score', 0.5)
                    
                    # Check for reproduction
                    if latest.get('reproduction_ready', False):
                        self.offspring_count += 1
                        log.info(f"JAS reproduction opportunity detected (count: {self.offspring_count})")
                        
                        # Check max offspring limit
                        if self.offspring_count >= self.config.max_offspring:
                            log.info(f"Maximum offspring count ({self.config.max_offspring}) reached - disabling reproduction")
                            if self.jas_agent.triune:
                                self.jas_agent.triune.cfg.enable_reproduction = False
                    
                    # Check for critical health issues
                    if health < 0.3:
                        log.warning(f"JAS health critical: {health:.2f}")
                        
                        # Check for prolonged crisis
                        if hasattr(self.jas_agent, 'apoptosis_system') and self.jas_agent.apoptosis_system:
                            crisis_level = self.jas_agent.apoptosis_system.crisis_counter
                            if crisis_level > 20:
                                log.warning("Prolonged crisis detected - considering intervention")
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Health monitoring error: {str(e)}")
                await asyncio.sleep(5.0)
    
    async def _telemetry_monitor(self):
        """Monitor and process telemetry data"""
        while self.running:
            try:
                if not self.jas_agent or not hasattr(self.jas_agent, 'telemetry'):
                    await asyncio.sleep(self.config.telemetry_interval)
                    continue
                
                # Process telemetry if available
                if self.jas_agent.telemetry and self.jas_agent.telemetry.history:
                    latest = self.jas_agent.telemetry.history[-1]
                    
                    # Log key metrics
                    log.debug(
                        f"JAS Telemetry | Health: {latest.get('health_score', 0.5):.2f} | "
                        f"Energy: C={latest.get('energy_cognitive', 0):.1f}/"
                        f"M={latest.get('energy_metabolic', 0):.1f}/"
                        f"R={latest.get('energy_reserve', 0):.1f} | "
                        f"Boundary: {latest.get('boundary_integrity', 0.5):.2f}"
                    )
                
                await asyncio.sleep(self.config.telemetry_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Telemetry monitoring error: {str(e)}")
                await asyncio.sleep(5.0)
    
    async def _preserve_legacy(self):
        """Preserve legacy artifacts before shutdown"""
        if not self.jas_agent:
            return
            
        try:
            log.info("Preserving JAS legacy artifacts...")
            
            # Create final artifact
            artifact = self._create_final_artifact()
            if artifact:
                self.legacy_artifacts.append(artifact)
                
                # Store in memory
                legacy_id = f"jas_legacy_{int(time.time())}"
                self.memory.store_legacy_data(legacy_id, artifact.to_dict())
                log.info(f"Legacy preserved with ID: {legacy_id}")
                
                # Log key metrics
                log.info(
                    f"Final legacy: Health={artifact.health_metrics.stability:.2f}, "
                    f"Energy={artifact.energy_snapshot.reserve:.1f}, "
                    f"VPMs={len(artifact.cognition_trace)}"
                )
            
            # Store all artifacts
            if self.legacy_artifacts:
                all_artifacts_id = f"jas_all_legacy_{int(time.time())}"
                artifacts_data = [a.to_dict() for a in self.legacy_artifacts]
                self.memory.store_legacy_data(all_artifacts_id, artifacts_data)
                log.info(f"All legacy artifacts stored with ID: {all_artifacts_id}")
                
        except Exception as e:
            log.error(f"Legacy preservation failed: {str(e)}", exc_info=True)
    
    def _create_final_artifact(self) -> Optional[JitterArtifactV0]:
        """Create a final artifact for legacy preservation"""
        if not self.jas_agent or not self.jas_agent.telemetry or not self.jas_agent.telemetry.history:
            return None
            
        try:
            # Get latest vital signs
            latest = self.jas_agent.telemetry.history[-1]
            
            # Create artifact
            return JitterArtifactV0(
                organism_id=getattr(self.jas_agent.core, 'id', f"jas_{int(time.time())}"),
                parent_id=getattr(self.jas_agent.core, 'parent_id', ''),
                generation=getattr(self.jas_agent.core, 'generation', 0),
                timestamp=time.time(),
                tick=getattr(self.jas_agent, 'tick', 0),
                cause="pipeline_shutdown",
                final_vitals=latest,
                cognition_trace=[
                    {"tick": vs.tick, "integrated": vs.integrated, "veto": vs.layer_veto}
                    for vs in self.jas_agent.telemetry.history[-10:]
                ],
                recent_vitals=self.jas_agent.telemetry.history[-20:]
            )
            
        except Exception as e:
            log.error(f"Failed to create final artifact: {str(e)}")
            return None
    
    def _get_final_status(self) -> Dict[str, Any]:
        """Get final status and metrics for the pipeline"""
        status = {
            "status": "stopped",
            "duration": time.time() - self.start_time if self.start_time else 0,
            "offspring_count": self.offspring_count,
            "legacy_artifacts": len(self.legacy_artifacts),
            "final_health": 0.5
        }
        
        # Add final telemetry if available
        if self.jas_agent and self.jas_agent.telemetry and self.jas_agent.telemetry.history:
            latest = self.jas_agent.telemetry.history[-1]
            status.update({
                "final_health": latest.get('health_score', 0.5),
                "final_energy": {
                    "cognitive": latest.get('energy_cognitive', 0),
                    "metabolic": latest.get('energy_metabolic', 0),
                    "reserve": latest.get('energy_reserve', 0)
                },
                "final_boundary": latest.get('boundary_integrity', 0.5)
            })
        
        return status
    
    async def get_health(self) -> Dict[str, Any]:
        """Get current health status for monitoring systems"""
        base_health = {
            "status": "running" if self.running else "stopped",
            "component": "jas_pipeline",
            "timestamp": time.time()
        }
        
        if not self.running:
            return {**base_health, "health_score": 0.0}
        
        try:
            # Get JAS health if available
            if self.jas_agent and self.jas_agent.telemetry and self.jas_agent.telemetry.history:
                latest = self.jas_agent.telemetry.history[-1]
                return {
                    **base_health,
                    "health_score": latest.get('health_score', 0.5),
                    "details": {
                        "energy": {
                            "cognitive": latest.get('energy_cognitive', 0),
                            "metabolic": latest.get('energy_metabolic', 0),
                            "reserve": latest.get('energy_reserve', 0)
                        },
                        "boundary": latest.get('boundary_integrity', 0.5),
                        "vpm_count": latest.get('vpm_count', 0),
                        "offspring_count": self.offspring_count
                    }
                }
            
            return {**base_health, "health_score": 0.5}
            
        except Exception as e:
            log.error(f"Health check error: {str(e)}")
            return {
                **base_health,
                "health_score": 0.0,
                "error": str(e)
            }

async def main():
    """CLI entry point for the JAS Pipeline Agent"""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="JAS Pipeline Agent - Run the Jitter Autopoietic System")
    parser.add_argument("--config", type=str, default="conf/agent/jas_pipeline.yaml",
                        help="Path to configuration file")
    parser.add_argument("--duration", type=int, default=None,
                        help="Run duration in seconds (overrides config)")
    parser.add_argument("--no-reproduction", action="store_true",
                        help="Disable reproduction system")
    parser.add_argument("--max-offspring", type=int, default=None,
                        help="Maximum number of offspring to create")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        log.error(f"Failed to load config from {args.config}: {str(e)}")
        sys.exit(1)
    
    # Override with CLI arguments
    if args.duration is not None:
        config.setdefault("jas_pipeline", {})["duration"] = args.duration
    if args.no_reproduction:
        config.setdefault("jas_pipeline", {})["jas_reproduction_enabled"] = False
    if args.max_offspring is not None:
        config.setdefault("jas_pipeline", {})["max_offspring"] = args.max_offspring
    
    # Create and run agent
    agent = JASPipelineAgent(config)
    
    # Set up signal handlers for graceful shutdown
    stop = asyncio.Future()
    
    def signal_handler():
        if not stop.done():
            stop.set_result(None)
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(sig, signal_handler)
    
    try:
        await agent.run()
    except Exception as e:
        log.error(f"Agent execution failed: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        await agent.stop()
        log.info("JAS Pipeline Agent terminated")

if __name__ == "__main__":
    asyncio.run(main())