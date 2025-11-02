from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import psutil  # pip install psutil
except Exception:
    psutil = None

@dataclass
class GPUMetric:
    index: int
    util: float       # 0..1
    mem_frac: float   # 0..1
    mem_used_mb: float
    mem_total_mb: float
    name: str = "GPU"

@dataclass
class OSSnapshot:
    ts: float
    cpu_util: float         # 0..1
    mem_frac: float         # 0..1
    disk_frac: float        # 0..1
    gpus: List[GPUMetric]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "cpu_util": self.cpu_util,
            "mem_frac": self.mem_frac,
            "disk_frac": self.disk_frac,
            "gpus": [g.__dict__ for g in self.gpus],
        }

class OSSensor:
    """
    Cross-platform OS sensor.
    - CPU, RAM, Disk via psutil if present, else conservative fallbacks.
    - GPU via nvidia-smi if present; otherwise empty list.

    Usage:
        sensor = OSSensor(disk_path="/")
        snap = sensor.sample()
        print(snap.as_dict())
    """
    def __init__(self, disk_path: str = "/"):
        self.disk_path = disk_path

    def _cpu_util(self) -> float:
        if psutil:
            # Non-blocking: use last interval, falls back gracefully
            try:
                v = psutil.cpu_percent(interval=None) / 100.0
                return max(0.0, min(1.0, v))
            except Exception:
                pass
        return 0.0

    def _mem_frac(self) -> float:
        if psutil:
            try:
                vm = psutil.virtual_memory()
                return max(0.0, min(1.0, vm.percent / 100.0))
            except Exception:
                pass
        return 0.0

    def _disk_frac(self) -> float:
        try:
            usage = shutil.disk_usage(self.disk_path)
            frac = (usage.used / max(1, usage.total))
            return max(0.0, min(1.0, frac))
        except Exception:
            return 0.0

    def _gpu_metrics(self) -> List[GPUMetric]:
        # Prefer nvidia-smi if available
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=0.5,
            ).decode("utf-8", errors="ignore")
            gpus: List[GPUMetric] = []
            for line in out.strip().splitlines():
                # index, name, util%, mem_used(MB), mem_total(MB)
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5: 
                    continue
                idx = int(parts[0])
                name = parts[1]
                util = max(0.0, min(1.0, float(parts[2]) / 100.0))
                mem_used = float(parts[3])
                mem_total = max(1.0, float(parts[4]))
                mem_frac = max(0.0, min(1.0, mem_used / mem_total))
                gpus.append(GPUMetric(index=idx, name=name, util=util,
                                      mem_frac=mem_frac,
                                      mem_used_mb=mem_used, mem_total_mb=mem_total))
            return gpus
        except Exception:
            return []

    def sample(self) -> OSSnapshot:
        return OSSnapshot(
            ts=time.time(),
            cpu_util=self._cpu_util(),
            mem_frac=self._mem_frac(),
            disk_frac=self._disk_frac(),
            gpus=self._gpu_metrics(),
        )
