# stephanie/components/gap/risk/api.py
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from .orchestrator import RiskOrchestrator

# ------------------------- Pydantic Models ----------------------------------

class AnalyzeReq(BaseModel):
    goal: str = Field(..., description="Original user goal/question")
    reply: str = Field(..., description="Model's finalized reply text")
    model_alias: str = Field("chat", description="Alias for the chat model")
    monitor_alias: str = Field("tiny", description="Alias for the monitor model (e.g., tiny|hrm)")
    run_id: Optional[str] = Field(None, description="Optional external run id")
    sparkline: Optional[list] = Field(None, description="Optional token-entropy trend in [0,1]")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context (baseline metrics, evidence, etc.)")
    policy_profile: Optional[str] = Field(None, description="Override policy profile for this call (e.g., chat.standard)")


class AnalyzeResp(BaseModel):
    run_id: str
    model_alias: str
    monitor_alias: str
    metrics: Dict[str, float]
    decision: str
    thresholds: Dict[str, float]
    reasons: Dict[str, float]
    badge_svg: str


# --------------------------- Router Factory ---------------------------------

def create_router(container: Any, *, default_profile: str = "chat.standard") -> APIRouter:
    """
    Creates an APIRouter bound to a container and a default policy profile.
    Mount this under e.g. '/v1/gap/risk' in your FastAPI app:

        app.include_router(create_router(container), prefix="/v1/gap/risk", tags=["gap-risk"])
    """
    router = APIRouter()
    orch = RiskOrchestrator(container, policy_profile=default_profile)

    @router.get("/health")
    async def health():
        return {"ok": True, "profile": default_profile}

    @router.post("/analyze", response_model=AnalyzeResp)
    async def analyze(req: AnalyzeReq) -> AnalyzeResp:
        # Switch profile per-request if provided (cheap: spin a throwaway orchestrator with override)
        local_orch = orch if not req.policy_profile else RiskOrchestrator(
            container, policy_profile=req.policy_profile
        )

        rec = await local_orch.evaluate(
            goal=req.goal,
            reply=req.reply,
            model_alias=req.model_alias,
            monitor_alias=req.monitor_alias,
            sparkline=req.sparkline,
            run_id=req.run_id,
            context=req.context,
        )
        return AnalyzeResp(**rec)  # rec shape matches AnalyzeResp

    return router

