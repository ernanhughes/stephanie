# stephanie/services/verification_queue.py
"""
VerificationQueue
-----------------
Work-queue atop your HybridKnowledgeBus (NATS + KV).
Avoids MaxPayloadError by storing big context in KV and sending only a pointer.

Subjects:
  - verify.section (jobs)
  - verify.section.done (completions)

KV buckets:
  - vsec.jobs
  - vsec.results
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, Union

MAX_INLINE = 128_000  # stay well below default NATS max payload
RESULT_TTL_SEC = 3600

def _sha(x: Union[str, bytes]) -> str:
    """Generate SHA-256 hash of input."""
    if isinstance(x, str):
        x = x.encode("utf-8")
    return hashlib.sha256(x).hexdigest()

class VerificationQueue:
    """
    Work-queue for distributed verification jobs.
    
    Usage:
        queue = VerificationQueue(memory, logger)
        job_id = queue.enqueue(section_ctx)
        # Wait for result via polling or subscription to "verify.section.done"
        result = queue.get_result(job_id)
    """
    
    def __init__(self, memory, logger, bucket_prefix="vsec"):
        self.memory = memory
        self.logger = logger
        self.bucket_prefix = bucket_prefix
        self.kv_jobs = None
        self.kv_results = None
        
        # Try to initialize KV stores
        self._init_kv()
    
    def _init_kv(self):
        """Initialize KV stores if bus supports it."""
        try:
            if hasattr(self.memory, "bus") and self.memory.bus:
                self.kv_jobs = self._kv(f"{self.bucket_prefix}.jobs")
                self.kv_results = self._kv(f"{self.bucket_prefix}.results")
        except Exception as e:
            self.logger.log("VerificationKVInitError", {
                "bucket_prefix": self.bucket_prefix,
                "error": str(e)
            })
    
    def _kv(self, bucket: str):
        """Get KV bucket with error handling."""
        try:
            return self.memory.bus.get_kv(
                bucket=bucket,
                description=f"{bucket} (verification queue)",
                max_age_seconds=RESULT_TTL_SEC
            )
        except Exception as e:
            self.logger.log("VerificationKVInitError", {
                "bucket": bucket,
                "error": str(e)
            })
            return None
    
    def enqueue(self, section_ctx: Dict[str, Any]) -> str:
        """
        Enqueue a verification job.
        
        Args:
            section_ctx: Context for section verification
            
        Returns:
            job_id: Unique ID for the job
        """
        # Create payload and job ID
        payload = json.dumps(section_ctx).encode("utf-8")
        job_id = _sha(payload)[:16]
        
        # Store body in KV (pointer pattern to avoid MaxPayload)
        if self.kv_jobs:
            try:
                self.kv_jobs.put(job_id, payload)
            except Exception as e:
                self.logger.log("VerificationJobStoreError", {
                    "job_id": job_id,
                    "error": str(e)
                })
        
        # Publish small job envelope
        envelope = json.dumps({"job_id": job_id}).encode("utf-8")
        try:
            self.memory.bus.publish("verify.section", envelope)
            self.logger.log("VerificationEnqueued", {
                "job_id": job_id, 
                "bytes": len(payload),
                "section": section_ctx.get("paper_section", {}).get("section_name", "unknown")
            })
        except Exception as e:
            self.logger.log("VerificationEnqueueError", {
                "job_id": job_id,
                "error": str(e)
            })
        
        return job_id
    
    def put_result(self, job_id: str, result: Dict[str, Any]) -> None:
        """Store verification result in KV."""
        if not self.kv_results:
            return
            
        try:
            buf = json.dumps(result).encode("utf-8")
            self.kv_results.put(job_id, buf)
        except Exception as e:
            self.logger.log("VerificationResultStoreError", {
                "job_id": job_id,
                "error": str(e)
            })
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job context from KV."""
        if not self.kv_jobs:
            return None
            
        try:
            raw = self.kv_jobs.get(job_id)
            return json.loads(raw.decode("utf-8")) if raw else None
        except Exception as e:
            self.logger.log("VerificationJobFetchError", {
                "job_id": job_id,
                "error": str(e)
            })
            return None
    
    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve verification result from KV."""
        if not self.kv_results:
            return None
            
        try:
            raw = self.kv_results.get(job_id)
            return json.loads(raw.decode("utf-8")) if raw else None
        except Exception as e:
            self.logger.log("VerificationResultFetchError", {
                "job_id": job_id,
                "error": str(e)
            })
            return None
    
    def subscribe(self, handler: callable) -> None:
        """
        Subscribe to verification job completion events.
        
        Args:
            handler: Function to call with job_id when result is ready
        """
        def wrapped_handler(msg):
            try:
                envelope = json.loads(msg.data.decode("utf-8"))
                job_id = envelope["job_id"]
                handler(job_id)
            except Exception as e:
                self.logger.log("VerificationSubscriptionError", {
                    "error": str(e)
                })
        
        if hasattr(self.memory, "bus") and self.memory.bus:
            self.memory.bus.subscribe("verify.section.done", wrapped_handler)
        else:
            self.logger.log("VerificationSubscriptionFailed", {
                "reason": "bus_not_available"
            }) 