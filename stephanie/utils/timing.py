# stephanie/utils/timing.py
import functools
import inspect
import time
from typing import Optional, Callable, Any, Dict, TypeVar

T = TypeVar('T')

def format_timing_string(
    class_name: str, func_name: str, duration_ms: float, timestamp: str, 
    status: str = "completed", error_type: Optional[str] = None
) -> str:
    """Format timing information as a readable string"""
    status_emoji = "✅" if status == "completed" else "❌"
    status_text = f" {status_emoji} {status}"
    if error_type:
        status_text += f" ({error_type})"
    
    return f"⏱️ {class_name}.{func_name}{status_text}: {round(duration_ms, 2)}ms [{timestamp}]"


def time_function(logger=None, threshold: Optional[float] = None, name: Optional[str] = None):
    """
    Decorator to time functions with configurable logging.
    
    Args:
        logger: Optional logger to use for logging timing information
        threshold: Optional threshold in milliseconds; only log if duration exceeds threshold
        name: Optional custom name for the timed operation (defaults to function name)
    
    Returns:
        Decorated function that logs its execution time
    
    Example:
        @time_function(logger=my_logger, threshold=100)  # Only log if > 100ms
        async def my_async_function():
            # ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        func_name = name or func.__name__
        
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                start = time.perf_counter()
                class_name = "Function"
                obj = args[0] if args and hasattr(args[0], "__class__") else None
                if obj:
                    class_name = obj.__class__.__name__
                
                try:
                    result = await func(*args, **kwargs)
                    duration = time.perf_counter() - start
                    duration_ms = duration * 1000
                    
                    # Only log if no threshold or duration exceeds threshold
                    if threshold is None or duration_ms > threshold:
                        log_data = {
                            "function": func_name,
                            "class": class_name,
                            "duration_ms": round(duration_ms, 2),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "completed"
                        }
                        
                        # Add pipeline context if available
                        if obj and hasattr(obj, "current_plan_trace") and obj.current_plan_trace:
                            log_data["pipeline_run_id"] = obj.current_plan_trace.trace_id
                        
                        if obj and hasattr(obj, "trace"):
                            log_data["trace_length"] = len(getattr(obj, "trace", []))
                        
                        if logger:
                            logger.log("FunctionTiming", log_data)
                        else:
                            print(format_timing_string(
                                class_name, 
                                func_name, 
                                duration_ms, 
                                log_data["timestamp"]
                            ))
                    
                    return result
                    
                except Exception as e:
                    duration = time.perf_counter() - start
                    duration_ms = duration * 1000
                    
                    # Always log exceptions, regardless of threshold
                    error_log_data = {
                        "function": func_name,
                        "class": class_name,
                        "duration_ms": round(duration_ms, 2),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "failed",
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                    
                    # Add pipeline context if available
                    if obj and hasattr(obj, "current_plan_trace") and obj.current_plan_trace:
                        error_log_data["pipeline_run_id"] = obj.current_plan_trace.trace_id
                    
                    if obj and hasattr(obj, "trace"):
                        error_log_data["trace_length"] = len(getattr(obj, "trace", []))
                    
                    if logger:
                        logger.log("FunctionTiming", error_log_data)
                    else:
                        print(format_timing_string(
                            class_name, 
                            func_name, 
                            duration_ms, 
                            error_log_data["timestamp"],
                            "failed",
                            type(e).__name__
                        ))
                    
                    # Re-raise the exception
                    raise
                
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                start = time.perf_counter()
                class_name = "Function"
                obj = args[0] if args and hasattr(args[0], "__class__") else None
                if obj:
                    class_name = obj.__class__.__name__
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.perf_counter() - start
                    duration_ms = duration * 1000
                    
                    # Only log if no threshold or duration exceeds threshold
                    if threshold is None or duration_ms > threshold:
                        log_data = {
                            "function": func_name,
                            "class": class_name,
                            "duration_ms": round(duration_ms, 2),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "completed"
                        }
                        
                        # Add pipeline context if available
                        if obj and hasattr(obj, "current_plan_trace") and obj.current_plan_trace:
                            log_data["pipeline_run_id"] = obj.current_plan_trace.trace_id
                        
                        if obj and hasattr(obj, "trace"):
                            log_data["trace_length"] = len(getattr(obj, "trace", []))
                        
                        if logger:
                            logger.log("FunctionTiming", log_data)
                        else:
                            print(format_timing_string(
                                class_name, 
                                func_name, 
                                duration_ms, 
                                log_data["timestamp"]
                            ))
                    
                    return result
                    
                except Exception as e:
                    duration = time.perf_counter() - start
                    duration_ms = duration * 1000
                    
                    # Always log exceptions, regardless of threshold
                    error_log_data = {
                        "function": func_name,
                        "class": class_name,
                        "duration_ms": round(duration_ms, 2),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "failed",
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                    
                    # Add pipeline context if available
                    if obj and hasattr(obj, "current_plan_trace") and obj.current_plan_trace:
                        error_log_data["pipeline_run_id"] = obj.current_plan_trace.trace_id
                    
                    if obj and hasattr(obj, "trace"):
                        error_log_data["trace_length"] = len(getattr(obj, "trace", []))
                    
                    if logger:
                        logger.log("FunctionTiming", error_log_data)
                    else:
                        print(format_timing_string(
                            class_name, 
                            func_name, 
                            duration_ms, 
                            error_log_data["timestamp"],
                            "failed",
                            type(e).__name__
                        ))
                    
                    # Re-raise the exception
                    raise
                
            return sync_wrapper

    return decorator


class TimingAnalyzer:
    def __init__(self, logger):
        self.logger = logger

    def analyze(self, event_type="FunctionTiming", min_duration_ms: float = 0):
        """
        Analyze timing logs with filtering options
        
        Args:
            event_type: Type of timing events to analyze
            min_duration_ms: Only include functions that took at least this long
            
        Returns:
            Dictionary with timing analysis results
        """
        logs = self.logger.get_logs_by_type(event_type)
        if not logs:
            return {
                "avg_times": {},
                "total_calls": {},
                "max_times": {},
                "min_times": {},
                "call_counts": {}
            }

        # Group by function
        from collections import defaultdict
        function_times = defaultdict(list)
        
        for log in logs:
            data = log["data"]
            if data.get("duration_ms", 0) >= min_duration_ms:
                key = f"{data.get('class', '')}.{data.get('function', '')}"
                function_times[key].append(data["duration_ms"])

        return {
            "avg_times": {k: sum(v) / len(v) for k, v in function_times.items()},
            "total_calls": {k: len(v) for k, v in function_times.items()},
            "max_times": {k: max(v) for k, v in function_times.items()},
            "min_times": {k: min(v) for k, v in function_times.items()},
            "call_counts": {k: len(v) for k, v in function_times.items()}
        }
    
    def analyze_by_pipeline(self, pipeline_run_id: str, min_duration_ms: float = 0):
        """
        Analyze timing logs for a specific pipeline run
        
        Args:
            pipeline_run_id: ID of the pipeline to analyze
            min_duration_ms: Only include functions that took at least this long
            
        Returns:
            Dictionary with timing analysis results for the pipeline
        """
        logs = self.logger.get_logs_by_type("FunctionTiming")
        pipeline_logs = [log for log in logs if log["data"].get("pipeline_run_id") == pipeline_run_id]
        
        if not pipeline_logs:
            return {
                "avg_times": {},
                "total_calls": {},
                "max_times": {},
                "min_times": {},
                "call_counts": {}
            }

        # Group by function
        from collections import defaultdict
        function_times = defaultdict(list)
        
        for log in pipeline_logs:
            data = log["data"]
            if data.get("duration_ms", 0) >= min_duration_ms:
                key = f"{data.get('class', '')}.{data.get('function', '')}"
                function_times[key].append(data["duration_ms"])

        return {
            "avg_times": {k: sum(v) / len(v) for k, v in function_times.items()},
            "total_calls": {k: len(v) for k, v in function_times.items()},
            "max_times": {k: max(v) for k, v in function_times.items()},
            "min_times": {k: min(v) for k, v in function_times.items()},
            "call_counts": {k: len(v) for k, v in function_times.items()}
        }