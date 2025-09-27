# stephanie/workers/verified_section_worker.py
"""
VerifiedSectionWorker
---------------------
Processes verification jobs from the queue and runs the VerifiedSectionGenerator.
"""
from __future__ import annotations

import json
import logging
import traceback

from stephanie.agents.knowledge.verified_section_generator import \
    VerifiedSectionGeneratorAgent
from stephanie.services.verification_queue import VerificationQueue


class VerifiedSectionWorker:
    """
    Worker that processes verification jobs from the queue.
    
    Usage:
        worker = VerifiedSectionWorker(cfg, memory, container, logger)
        worker.start()
    """
    
    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.queue = VerificationQueue(memory, logger)
        self.agent = VerifiedSectionGeneratorAgent(cfg, memory, container, logger)
        self.logger = logger or logging.getLogger(__name__)
    
    async def handle_job(self, msg_bytes: bytes):
        """
        Handle a verification job message.
        
        Args:
            msg_bytes: Raw message bytes from the bus
        """
        try:
            envelope = json.loads(msg_bytes.decode("utf-8"))
            job_id = envelope["job_id"]
            self.logger.log("VerificationJobReceived", {"job_id": job_id})
            
            # Get job context
            ctx = self.queue.get_job(job_id)
            if not ctx:
                self.logger.log("VerificationJobNotFound", {"job_id": job_id})
                return
            
            # Run verification
            result_ctx = await self.agent.run(ctx)
            
            # Store result
            result = {
                "verified_section": result_ctx.get("verified_section"),
                "quality_confidence": result_ctx.get("quality_confidence"),
                "verification_trace": result_ctx.get("verification_trace"),
                "timestamp": result_ctx.get("timestamp", time.time())
            }
            self.queue.put_result(job_id, result)
            
            # Publish completion signal
            self.memory.bus.publish("verify.section.done", json.dumps({"job_id": job_id}).encode("utf-8"))
            
            self.logger.log("VerificationJobDone", {
                "job_id": job_id, 
                "score": result.get("quality_confidence"),
                "section": ctx.get("paper_section", {}).get("section_name", "unknown")
            })
        except Exception as e:
            self.logger.log("VerificationJobError", {
                "error": str(e), 
                "trace": traceback.format_exc(),
                "job_id": envelope.get("job_id", "unknown") if 'envelope' in locals() else "unknown"
            })
    
    def start(self):
        """
        Start listening for verification jobs.
        
        Uses a queue group so multiple workers can share load.
        """
        if hasattr(self.memory, "bus") and self.memory.bus:
            self.memory.bus.subscribe(
                "verify.section", 
                queue_group="vsec", 
                handler=self.handle_job
            )
            self.logger.log("VerifiedSectionWorkerStarted", {
                "subject": "verify.section",
                "queue_group": "vsec"
            })
        else:
            self.logger.log("VerifiedSectionWorkerStartFailed", {
                "reason": "bus_not_available"
            })