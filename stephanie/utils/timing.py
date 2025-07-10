import time
from functools import wraps
from typing import Any, Callable
import inspect
import functools
import time
import asyncio

def time_function(logger=None):
    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                duration = time.perf_counter() - start

                obj = args[0] if args and hasattr(args[0], '__class__') else None
                class_name = obj.__class__.__name__ if obj else "Function"

                log_data = {
                    "function": func.__name__,
                    "class": class_name,
                    "duration_ms": round(duration * 1000, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                if obj and hasattr(obj, 'trace'):
                    log_data["trace_length"] = len(getattr(obj, 'trace', []))

                if logger:
                    logger.log("FunctionTiming", log_data)
                else:
                    print(f"⏱️ {class_name}.{func.__name__}: {log_data['duration_ms']}ms [{log_data['timestamp']}]")

                return result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start

                obj = args[0] if args and hasattr(args[0], '__class__') else None
                class_name = obj.__class__.__name__ if obj else "Function"

                log_data = {
                    "function": func.__name__,
                    "class": class_name,
                    "duration_ms": round(duration * 1000, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                if obj and hasattr(obj, 'trace'):
                    log_data["trace_length"] = len(getattr(obj, 'trace', []))

                if logger:
                    logger.log("FunctionTiming", log_data)
                else:
                    print(f"⏱️ {class_name}.{func.__name__}: {log_data['duration_ms']}ms [{log_data['timestamp']}]")

                return result
            return sync_wrapper
    return decorator


class TimingAnalyzer:
    def __init__(self, logger):
        self.logger = logger
    
    def analyze(self, event_type="FunctionTiming"):
        logs = self.logger.get_logs_by_type(event_type)
        
        # Group by function
        from collections import defaultdict
        function_times = defaultdict(list)
        for log in logs:
            data = log["data"]
            key = f"{data.get('class', '')}.{data.get('function', '')}"
            function_times[key].append(data["duration_ms"])
        
        return {
            "avg_times": {k: sum(v)/len(v) for k, v in function_times.items()},
            "total_calls": {k: len(v) for k, v in function_times.items()},
            "max_times": {k: max(v) for k, v in function_times.items()}
        }