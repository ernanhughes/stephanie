# stephanie/components/jitter/core/config.py
from __future__ import annotations

import logging
from typing import Any, Dict

from pydantic import BaseModel, Field, root_validator, validator

log = logging.getLogger(__name__)

class JitterConfig(BaseModel):
    """Validated configuration for the entire Jitter system"""
    
    # Core system parameters
    tick_interval: float = Field(1.0, gt=0.1, description="Time between ticks in seconds")
    enable_reproduction: bool = Field(True, description="Whether reproduction is enabled")
    reproduction_interval: int = Field(1000, gt=100, description="Minimum ticks between reproduction")
    
    # Component configurations
    core: Dict[str, Any] = Field(default_factory=dict)
    triune: Dict[str, Any] = Field(default_factory=dict)
    homeostasis: Dict[str, Any] = Field(default_factory=dict)
    production: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('tick_interval')
    def validate_tick_interval(cls, v):
        if v > 10.0:
            raise ValueError('tick_interval should be less than 10 seconds for responsiveness')
        return v
    
    @root_validator
    def validate_component_configs(cls, values):
        # Validate production config if present
        if 'production' in values:
            from .production.closed_production import ProductionConfig
            try:
                ProductionConfig(**values['production'])
            except Exception as e:
                raise ValueError(f"Invalid production configuration: {str(e)}")
        
        # Add similar validation for other component configs
        return values

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize Jitter configuration"""
    try:
        validated = JitterConfig(**config)
        return validated.dict()
    except Exception as e:
        log.error(f"Configuration validation failed: {str(e)}")
        # Return safe defaults
        return {
            "tick_interval": 1.0,
            "enable_reproduction": True,
            "reproduction_interval": 1000,
            "core": {},
            "triune": {},
            "homeostasis": {},
            "production": {}
        }