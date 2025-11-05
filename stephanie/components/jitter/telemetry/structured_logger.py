# stephanie/components/jitter/telemetry/structured_logger.py
"""
structured_logger.py
====================
Enhanced logging system for the Jitter Autopoietic System.

This module implements structured logging that integrates with the telemetry system
to provide detailed insights into Jitter's behavior while maintaining performance.

Key Features:
- Structured logging with contextual information
- Integration with telemetry system for cross-system observability
- Performance monitoring and metrics collection
- Configuration validation with Pydantic
- Circuit breaker pattern for resilience
- Comprehensive telemetry and monitoring
- SSP integration hooks
- Performance optimizations
"""
from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, validator

from stephanie.components.jitter.telemetry.telemetry import JASTelemetry

log = logging.getLogger("stephanie.jitter.structured_logger")

class LogLevel(str, Enum):
    """Log levels for structured logging"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class LoggerConfig(BaseModel):
    """Validated configuration for StructuredLogger"""
    log_level: str = Field("INFO", description="Minimum log level to capture")
    enable_telemetry: bool = Field(True, description="Whether to send logs to telemetry")
    max_history: int = Field(1000, ge=100, le=10000, description="Maximum log history")
    flush_interval: float = Field(5.0, ge=1.0, le=30.0, description="Flush interval in seconds")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in levels:
            raise ValueError(f'log_level must be one of {levels}')
        return v.upper()

class CircuitBreakerState:
    """States for circuit breaker pattern"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """
    Circuit breaker pattern implementation for service resilience.
    
    Example usage:
    
    @CircuitBreaker(failure_threshold=3, recovery_timeout=30)
    def log(message, level):
        # Logging logic here
        pass
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_attempts: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_attempts = half_open_attempts
        
        self.state = CircuitBreakerState.CLOSED
        self.failures = 0
        self.last_failure_time = 0.0
        self.half_open_successes = 0
        self.logger = logging.getLogger(f"{__name__}.circuit_breaker")
    
    def __call__(self, func: callable) -> callable:
        """Decorator implementation"""
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN state")
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_successes = 0
                else:
                    log.warning("Circuit breaker is OPEN - skipping call")
                    return None
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Reset failures if successful
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.half_open_successes += 1
                    if self.half_open_successes >= self.half_open_attempts:
                        self.logger.info("Circuit breaker transitioning to CLOSED state")
                        self.state = CircuitBreakerState.CLOSED
                        self.failures = 0
                        self.half_open_successes = 0
                
                return result
                
            except Exception as e:
                # Record failure
                self.failures += 1
                self.last_failure_time = time.time()
                self.logger.error(f"Service failure: {str(e)}, failures: {self.failures}")
                
                # Transition to OPEN state if threshold reached
                if self.failures >= self.failure_threshold:
                    log.warning("Circuit breaker transitioning to OPEN state")
                    self.state = CircuitBreakerState.OPEN
                
                raise
        
        return wrapper
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state for monitoring"""
        return {
            "state": self.state,
            "failures": self.failures,
            "last_failure_time": self.last_failure_time,
            "half_open_successes": self.half_open_successes
        }

@dataclass
class LogEntry:
    """Structured log entry for detailed monitoring"""
    timestamp: float = field(default_factory=time.time)
    level: str = "info"
    message: str = ""
    module: str = ""
    function: str = ""
    line_number: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    thread_id: str = ""
    process_id: int = 0

@dataclass
class LoggerMetrics:
    """Metrics for structured logger performance"""
    log_rate: float = 0.0
    error_rate: float = 0.0
    context_coverage: float = 0.0
    processing_time_ms: float = 0.0
    history_size: int = 0

class StructuredLogger:
    """
    Enhanced logging system for the Jitter Autopoietic System.
    
    Key Features:
    - Structured logging with contextual information
    - Integration with telemetry system for cross-system observability
    - Performance monitoring and metrics collection
    - Configuration validation with Pydantic
    - Circuit breaker pattern for resilience
    - Comprehensive telemetry and monitoring
    - SSP integration hooks
    - Performance optimizations
    """
    
    def __init__(self, cfg: Dict[str, Any], telemetry: Optional[JASTelemetry] = None):
        try:
            # Validate configuration
            self.config = LoggerConfig(**cfg)
            log.info("StructuredLogger configuration validated successfully")
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self.config = LoggerConfig()
        
        # Initialize logger
        self.logger = logging.getLogger("stephanie.jitter")
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Initialize telemetry integration
        self.telemetry = telemetry
        
        # Initialize history
        self.log_history: List[LogEntry] = []
        self.context_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Initialize metrics
        self.metrics = LoggerMetrics()
        
        # Initialize performance tracking
        self.processing_times = []
        self.max_processing_history = 100
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # Initialize thread-local storage for context
        self._local = threading.local()
        
        log.info("StructuredLogger initialized with enhanced logging capabilities")
    
    @CircuitBreaker()
    def log(
        self,
        message: str,
        level: str = "info",
        module: Optional[str] = None,
        function: Optional[str] = None,
        line_number: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """
        Log a structured message with contextual information.
        
        Args:
            message: The log message
            level: Log level ("debug", "info", "warning", "error", "critical")
            module: Module name (optional)
            function: Function name (optional)
            line_number: Line number (optional)
            context: Additional context data (optional)
            exception: Exception object (optional)
        """
        start_time = time.time()
        
        try:
            # Get caller info if not provided
            if module is None or function is None or line_number is None:
                import inspect
                frame = inspect.currentframe()
                try:
                    caller_frame = frame.f_back
                    if caller_frame:
                        if module is None:
                            module = caller_frame.f_globals.get('__name__', 'unknown')
                        if function is None:
                            function = caller_frame.f_code.co_name
                        if line_number is None:
                            line_number = caller_frame.f_lineno
                finally:
                    del frame
            
            # Create log entry
            log_entry = LogEntry(
                level=level.lower(),
                message=message,
                module=module or "unknown",
                function=function or "unknown",
                line_number=line_number or 0,
                context=context or {},
                thread_id=threading.current_thread().name,
                process_id=os.getpid()
            )
            
            # Add exception info if provided
            if exception:
                log_entry.exception = str(exception)
                log_entry.stack_trace = traceback.format_exc()
            
            # Add to history
            self.log_history.append(log_entry)
            if len(self.log_history) > self.config.max_history:
                self.log_history.pop(0)
            
            # Update context history
            if context:
                context_key = f"{module}.{function}"
                self.context_history[context_key].append(context)
                if len(self.context_history[context_key]) > 100:
                    self.context_history[context_key].pop(0)
            
            # Update metrics
            self._update_metrics()
            
            # Log to standard logger
            standard_level = getattr(logging, level.upper(), logging.INFO)
            self.logger.log(standard_level, message)
            
            # Send to telemetry if enabled
            if self.config.enable_telemetry and self.telemetry:
                self._send_to_telemetry(log_entry)
            
            # Log processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_history:
                self.processing_times.pop(0)
                
            log.debug(f"Logged message (level={level}, module={module})")
            
        except Exception as e:
            log.error(f"Error in structured logging: {str(e)}", exc_info=True)
    
    def debug(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log a debug message"""
        self.log(message, "debug", context=context, **kwargs)
    
    def info(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log an info message"""
        self.log(message, "info", context=context, **kwargs)
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log a warning message"""
        self.log(message, "warning", context=context, **kwargs)
    
    def error(self, message: str, context: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None, **kwargs):
        """Log an error message"""
        self.log(message, "error", context=context, exception=exception, **kwargs)
    
    def critical(self, message: str, context: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None, **kwargs):
        """Log a critical message"""
        self.log(message, "critical", context=context, exception=exception, **kwargs)
    
    def _send_to_telemetry(self, log_entry: LogEntry):
        """Send log entry to telemetry system"""
        try:
            # Convert to telemetry-friendly format
            telemetry_data = {
                "type": "structured_log",
                "level": log_entry.level,
                "message": log_entry.message,
                "module": log_entry.module,
                "function": log_entry.function,
                "line_number": log_entry.line_number,
                "timestamp": log_entry.timestamp,
                "context": log_entry.context,
                "thread_id": log_entry.thread_id,
                "process_id": log_entry.process_id
            }
            
            # Add exception info if present
            if log_entry.exception:
                telemetry_data["exception"] = log_entry.exception
                telemetry_data["stack_trace"] = log_entry.stack_trace
            
            # Send to telemetry (this would be implemented based on system needs)
            # For now, just log that we tried to send
            log.debug(f"Attempting to send log to telemetry: {log_entry.message}")
            
        except Exception as e:
            log.warning(f"Failed to send log to telemetry: {str(e)}")
    
    def _update_metrics(self):
        """Update logger metrics based on recent logs"""
        # Update log rate
        if len(self.log_history) > 1:
            time_diff = self.log_history[-1].timestamp - self.log_history[0].timestamp
            if time_diff > 0:
                self.metrics.log_rate = len(self.log_history) / time_diff
        
        # Update error rate
        if len(self.log_history) > 0:
            error_count = sum(1 for l in self.log_history[-100:] if l.level in ['error', 'critical'])
            self.metrics.error_rate = error_count / max(1, len(self.log_history[-100:]))
        
        # Update processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        self.metrics.processing_time_ms = avg_processing_time * 1000
        self.metrics.history_size = len(self.log_history)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current logger metrics for monitoring and adaptation"""
        return {
            "log_rate": self.metrics.log_rate,
            "error_rate": self.metrics.error_rate,
            "context_coverage": self.metrics.context_coverage,
            "processing_time_ms": self.metrics.processing_time_ms,
            "history_size": self.metrics.history_size,
            "circuit_breaker_state": self.circuit_breaker.get_state(),
            "enable_telemetry": self.config.enable_telemetry
        }
    
    def get_recent_logs(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent log entries for analysis or reporting"""
        return [
            {
                "timestamp": l.timestamp,
                "level": l.level,
                "message": l.message,
                "module": l.module,
                "function": l.function,
                "line_number": l.line_number,
                "context": l.context,
                "exception": l.exception,
                "thread_id": l.thread_id
            }
            for l in self.log_history[-n:]
        ]
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get statistics about logged context data"""
        if not self.context_history:
            return {"total_context_entries": 0, "unique_context_keys": 0}
        
        total_entries = sum(len(v) for v in self.context_history.values())
        unique_keys = len(self.context_history)
        
        return {
            "total_context_entries": total_entries,
            "unique_context_keys": unique_keys,
            "context_keys": list(self.context_history.keys())
        }
    
    def get_ssp_integration_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for SSP integration.
        
        Returns logger metrics in a format SSP can use for reward shaping.
        """
        metrics = self.get_metrics()
        
        return {
            "log_rate": metrics["log_rate"],
            "error_rate": metrics["error_rate"],
            "processing_efficiency": 1.0 / (1.0 + metrics["processing_time_ms"]),
            "context_coverage": metrics["context_coverage"]
        }
    
    def reset(self):
        """Reset logger history and metrics"""
        self.log_history.clear()
        self.context_history.clear()
        self.metrics = LoggerMetrics()
        self.processing_times.clear()
        log.info("StructuredLogger reset")
    
    def set_context(self, key: str, value: Any):
        """Set context for current thread"""
        if not hasattr(self._local, 'context'):
            self._local.context = {}
        self._local.context[key] = value
    
    def get_context(self) -> Dict[str, Any]:
        """Get context for current thread"""
        if hasattr(self._local, 'context'):
            return self._local.context.copy()
        return {}
    
    def clear_context(self):
        """Clear context for current thread"""
        if hasattr(self._local, 'context'):
            self._local.context.clear()