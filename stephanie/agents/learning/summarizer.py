# stephanie/agents/learning/summarizer.py
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List

from stephanie.utils.json_sanitize import dumps_safe

_logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, cfg, memory, container, logger, strategy, scoring, prompt_loader, call_llm):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.strategy = strategy
        self.scoring = scoring
        self.prompt_loader = prompt_loader
        self.call_llm = call_llm

        # Configure concurrency
        self._max_concurrent = int(cfg.get("max_concurrent_improvements", 8))
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._in_flight: Dict[str, float] = {}  # track outstanding prompt_id
        self._cleanup_task = None

    async def baseline(self, paper: Dict[str, Any], section: Dict[str, Any],
                 critical_msgs: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        merged = {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "focus_section": section.get("section_name"),
            "focus_text": section.get("section_text", "")[:5000],
            # prefer assistant_text but fall back to text
            "hints": "\n".join((m.get("assistant_text") or m.get("text") or "")
                               for m in (critical_msgs[:6] if critical_msgs else [])),
            **context,
        }
        prompt = self.prompt_loader.from_file("baseline_summary", self.cfg, merged)
        prompt_service = self.container.get("prompt")
        return await prompt_service.run_prompt(prompt, merged)

    async def improve_once(self, paper: Dict[str, Any], section: Dict[str, Any],
                    current_summary: str, context: Dict[str, Any],
                    return_attribution: bool = False):

        metrics = self.scoring.score_summary(current_summary, paper, section, context)
        merged_context = {
            "title": paper.get("title", ""),
            "section_name": section.get("section_name"),
            "section_text": section.get("section_text", "")[:6000],
            "current_summary": current_summary,
            "skeptic_weight": self.strategy.state.skeptic_weight,
            "editor_weight": self.strategy.state.editor_weight,
            "risk_weight": self.strategy.state.risk_weight,
            "weaknesses": json.dumps(metrics.get("weaknesses", []), ensure_ascii=False),
            **context,
        }
        prompt = self.prompt_loader.from_file("improve_summary", self.cfg, merged_context)

        prompt_service = self.container.get("prompt")
        # 🚀 call the PromptService instead of LLM directly
        improved = await prompt_service.run_prompt(prompt, merged_context)

        # --- if no attribution needed ---
        if not return_attribution:
            return improved 

        # --- attribution (AR/AKL) ---
        claims = self._extract_claim_sentences(improved)

        # Normalize retrieval & arena pools
        rpool = (context.get("retrieval_items") or []) + (context.get("arena_initial_pool") or [])
        norm_sources = self._normalize_sources(rpool)

        th = float(self.cfg.get("applied_knowledge", {}).get("attr_sim_threshold", 0.75))
        matches = self._attribute_claims(claims, norm_sources, th)

        # store a compact scorable if case exists
        try:
            case_id = context.get("case_id")
            if case_id and matches:
                payload = {"claims": matches, "threshold": th, "timestamp": time.time()}
                self.memory.casebooks.add_scorable(
                    case_id=case_id, role="improve_attribution",
                    text=dumps_safe(payload),
                    pipeline_run_id=context.get("pipeline_run_id"),
                    meta={"iteration": context.get("iteration")}
                )
        except Exception as e:
            _logger.error(f"Failed to save attribution: {str(e)}", exc_info=True)

        return {"text": improved, "attribution": matches}
    

    async def _run_prompt_service(self, prompt: str, context: dict) -> Dict[str, Any]:
        """Fixed implementation with proper subscription handling"""
        prompt_id = str(uuid.uuid4())
        fut = asyncio.get_event_loop().create_future()

        async def _handler(msg: dict):
            """Handler for prompt results - properly cleans up after itself"""
            if msg.get("prompt_id") == prompt_id:
                try:
                    if "response" in msg:
                        fut.set_result({
                            "prompt_service": msg["response"],
                            "prompt_id": prompt_id,
                            "meta": msg.get("meta", {})
                        })
                    else:
                        error = msg.get("error", "unknown error")
                        error_type = msg.get("error_type", "RuntimeError")
                        fut.set_exception(
                            RuntimeError(f"{error_type}: {error}")
                        )
                finally:
                    # Always clean up subscription
                    try:
                        await self.memory.bus.unsubscribe("prompts.run.result", _handler)
                    except Exception as e:
                        _logger.debug(f"Error unsubscribing: {str(e)}")
                    self._in_flight.pop(prompt_id, None)
        
        # Add to in-flight tracking
        self._in_flight[prompt_id] = time.time()
        
        # Subscribe before publishing to avoid race condition
        await self.memory.bus.subscribe("prompts.run.result", _handler)
        
        # Publish request with proper timeout handling
        async with self._semaphore:
            try:
                await self.memory.bus.publish("prompts.run.request", {
                    "prompt_id": prompt_id,
                    "text": prompt,
                    "meta": {
                        "context": {k: str(v)[:200] for k,v in (context or {}).items()}
                    },
                    "timeout": self.cfg.get("prompt_timeout", 300)
                })
                
                # Add timeout guard
                return await asyncio.wait_for(
                    fut, 
                    timeout=self.cfg.get("prompt_timeout", 300)
                )
            except asyncio.TimeoutError:
                # Clean up on timeout
                try:
                    await self.memory.bus.unsubscribe("prompts.run.result", _handler)
                    self._in_flight.pop(prompt_id, None)
                except Exception:
                    pass
                raise RuntimeError(f"Prompt {prompt_id} timed out after {self.cfg.get('prompt_timeout', 300)}s")
            except Exception:
                # Clean up on error
                try:
                    await self.memory.bus.unsubscribe("prompts.run.result", _handler)
                    self._in_flight.pop(prompt_id, None)
                except Exception:
                    pass
                raise

    def _normalize_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map mixed corpus/arena items to {text, origin, variant}."""
        out: List[Dict[str, Any]] = []
        for s in sources or []:
            # corpus items often store text under "assistant_text"
            txt = (s.get("text") or s.get("assistant_text") or "").strip()
            if not txt:
                continue
            origin = s.get("origin") or (s.get("meta") or {}).get("source")
            out.append({
                "text": txt,
                "origin": origin,
                "variant": s.get("variant")
            })
        return out

    def _cos_sim(self, a, b) -> float:
        num = sum(x*y for x,y in zip(a,b))
        na = (sum(x*x for x in a))**0.5 or 1.0
        nb = (sum(x*x for x in b))**0.5 or 1.0
        return max(0.0, min(1.0, num/(na*nb)))

    def _attribute_claims(self, claims: List[str], sources: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        emb = getattr(self.memory, "embedding", None) if self.cfg.get("applied_knowledge",{}).get("use_embeddings", True) else None
        out = []

        if emb:
            # pre-embed sources (cap for speed)
            S = [{"meta": s, "v": emb.get_or_create(s["text"][:2000])} for s in (sources[:50] if sources else [])]
        else:
            S = sources[:50] if sources else []

        import re
        for c in claims:
            if not c:
                continue
            best, best_sim = None, 0.0
            if emb and S:
                cv = emb.get_or_create(c)
                for s in S:
                    sim = self._cos_sim(cv, s["v"])
                    if sim > best_sim:
                        best_sim, best = sim, s["meta"]
            else:
                # fallback: token overlap
                ctoks = set(re.findall(r"\b\w+\b", c.lower()))
                for s in S:
                    stoks = set(re.findall(r"\b\w+\b", s["text"].lower()))
                    denom = float(max(1, max(len(ctoks), len(stoks))))  # ← fix
                    sim = len(ctoks & stoks) / denom
                    if sim > best_sim:
                        best_sim, best = sim, s

            if best and best_sim >= threshold:
                out.append({
                    "claim": c,
                    "support": {
                        "text": (best.get("text") or "")[:220],
                        "origin": best.get("origin"),
                        "variant": best.get("variant")
                    },
                    "similarity": round(best_sim, 3)
                })
        return out

    async def verify_and_improve(self, baseline: str, paper: Dict[str, Any], section: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Fixed implementation with proper async handling"""
        current = baseline
        iters: List[Dict[str, Any]] = []
        max_iter = int(self.cfg.get("max_iterations", 3))
        last_attr_supported = False
        first_score_with_knowledge = None
        metrics = {"overall": 0.0}  # default

        for i in range(1, max_iter + 1):
            metrics = self.scoring.score_summary(current, paper, section, context)
            # annotate whether this iteration's text was influenced by knowledge (from prior improvement step)
            it = {"iteration": i, "score": metrics["overall"], "metrics": metrics, "knowledge_applied": bool(last_attr_supported)}
            iters.append(it)

            if last_attr_supported and first_score_with_knowledge is None:
                first_score_with_knowledge = metrics["overall"]

            if metrics["overall"] >= self.strategy.state.verification_threshold:
                break

            # run one improvement with attribution
            improved = await self.improve_once(paper, section, current, {**context, "iteration": i}, return_attribution=True)
            if isinstance(improved, dict):
                current = improved["text"]
                last_attr_supported = bool(improved.get("attribution"))
            else:
                current = improved
                last_attr_supported = False

        # rollups that Persistence can store
        k_applied_iters = sum(1 for it in iters if it.get("knowledge_applied"))
        k_applied_lift = (metrics["overall"] - first_score_with_knowledge) if first_score_with_knowledge is not None else 0.0

        self.strategy.evolve(iters, context)
        return {"summary": current, "metrics": {**metrics, "knowledge_applied_iters": k_applied_iters, "knowledge_applied_lift": k_applied_lift}, "iterations": iters}

    def health_status(self) -> Dict[str, Any]:
        """Expose summarizer health for dashboards / SIS"""
        now = time.time()
        return {
            "max_concurrent": self._max_concurrent,
            "in_flight": len(self._in_flight),
            "oldest_task_age": (
                round(now - min(self._in_flight.values()), 1) if self._in_flight else 0
            ),
            "queue_backlog": self._semaphore._value < self._max_concurrent,
            "timeout": self.cfg.get("prompt_timeout", 300)
        }
    
    def _extract_claim_sentences(self, text: str) -> List[str]:
        import re
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', (text or '').strip()) if s.strip()]
        triggers = ("show", "demonstrate", "evidence", "result", "increase", "decrease", "improve", "we propose", "we find")
        return [s for s in sents if any(t in s.lower() for t in triggers)][: int(self.cfg.get("applied_knowledge",{}).get("max_claims", 8))]

    def _attribute_claims(self, claims: List[str], sources: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        emb = getattr(self.memory, "embedding", None) if self.cfg.get("applied_knowledge",{}).get("use_embeddings", True) else None
        out = []
        # pre-embed sources (cap for speed)
        S = [{"meta": s, "v": (emb.get_or_create((s.get("text") or "")[:2000]) if emb else None)} for s in (sources[:50] if sources else [])]
        for c in claims:
            if not c: 
                continue
            best, best_sim = None, 0.0
            if emb and S:
                cv = emb.get_or_create(c)
                for s in S:
                    sim = self._cos_sim(cv, s["v"])
                    if sim > best_sim:
                        best_sim, best = sim, s["meta"]
            else:
                # fallback: token overlap
                import re
                ctoks = set(re.findall(r"\b\w+\b", c.lower()))
                for s in S or sources:
                    stoks = set(re.findall(r"\b\w+\b", (s.get("text","").lower())))
                    sim = len(ctoks & stoks) / float(max(1, len(ctoks)|len(stoks)))
                    if sim > best_sim:
                        best_sim, best = sim, s
            if best and best_sim >= threshold:
                out.append({"claim": c, "support": {"text": best.get("text","")[:220], "origin": best.get("origin"), "variant": best.get("variant")}, "similarity": round(best_sim, 3)})
        return out
