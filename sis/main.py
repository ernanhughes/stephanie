# sis/main.py
from __future__ import annotations

from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from stephanie.core.logging.json_logger import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from sis.routes import arena, db, nexus, pipelines, models, logs, plan_traces, documents, mars, learning
from sis.routes import casebooks as casebooks_routes
from sis.routes import chats, cards
from sis.routes import explore_ui
from sis.routes import overnight_ui
from sis.routes import ssp
from sis.utils.date_utls import short_dt, datetimeformat
from stephanie.services.service_container import ServiceContainer
from stephanie.services.registry_loader import load_services_profile
from hydra import initialize, compose

def load_sis_cfg(overrides: list[str] | None = None):
    with initialize(version_base=None, config_path="../config"):  # path containing sis.yaml
        cfg = compose(config_name="sis", overrides=overrides or [])
    return cfg

app = FastAPI(title="Stephanie Insight System (SIS)")

app.state.cfg = load_sis_cfg()
app.state.logger = JSONLogger("logs/sis.jsonl")
app.state.memory = MemoryTool(cfg=app.state.cfg, logger=app.state.logger)
app.state.container = ServiceContainer(cfg=app.state.cfg, logger=app.state.logger)

# Put into app.state so routers can access without importing main
app.state.templates = Jinja2Templates(directory="sis/templates")
app.state.templates.env.globals["now"] = datetime.now
app.state.templates.env.filters["short_dt"] = short_dt
# after you set up templates
app.state.templates.env.filters["datetimeformat"] = datetimeformat

try:
    load_services_profile(
        cfg=app.state.cfg, memory=app.state.memory, container=app.state.container, logger=app.state.logger,
        profile_path="config/services/default.yaml", supervisor=None
    )
except Exception as e:
    app.state.logger.info({"msg":"Service profile not loaded in SIS", "error": str(e)})


app.mount("/static", StaticFiles(directory="sis/static"), name="static")

# Add custom headers for JS modules
@app.middleware("http")
async def add_js_module_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Add module header for JS files
    if request.url.path.endswith(".js"):
        response.headers["Content-Type"] = "application/javascript"
        
    return response

# Include routers
app.include_router(pipelines.router)
app.include_router(models.router)
app.include_router(db.router)
app.include_router(logs.router)
app.include_router(plan_traces.router)
app.include_router(documents.router)
app.include_router(mars.router)
app.include_router(casebooks_routes.router)
app.include_router(chats.router)
app.include_router(arena.router)
app.include_router(cards.router)
app.include_router(learning.router)
app.include_router(explore_ui.router)
app.include_router(overnight_ui.router)
app.include_router(nexus.router)

# # After app = FastAPI(...)
# class SISContainer:
#     """Minimal DI bridge so GAP Risk can discover services already in SIS.
#     Add attributes here as your stack grows (metrics_service, tiny, hrm, event bus, etc.).
#     """
#     def __init__(self, app):
#         self.app = app
#         # Preferred unified scorer if present (contract: score_text_pair)
#         self.metrics_service = app.state.container.get_or_none("scoring")

#         # Tiny & HRM monitors (any attribute name you use; best-effort discovery)
#         self.tiny_scorer = (
#             getattr(app.state, "tiny_scorer", None)
#             or getattr(app.state, "monitor_tiny", None)
#             or getattr(app.state, "tiny", None)
#         )
#         self.hrm_scorer = getattr(app.state, "hrm_scorer", None) or getattr(app.state, "hrm", None)

#         # Optional event publisher for arena/overlay; method: publish(topic, payload)
#         self.event_publisher = getattr(app.state, "event_publisher", None)

# Mount JSON API under /v1/gap/risk
# app.include_router(create_gap_risk_router(app.state.container), prefix="/v1/gap/risk", tags=["gap-risk"])

# Mount SIS UI for risk
# Yeah app.include_router(risk_ui.router)

# try:
#     from stephanie.components.ssp.component import SSPComponent
#     from stephanie.utils.trace_logger import attach_to_app
#     app.state.ssp = SSPComponent(app.state.cfg.components.ssp, app.state.memory, app.state.container, app.state.logger)
#     attach_to_app(app, jsonl_path="logs/plan_traces.jsonl", enable_stdout=False)
# except Exception as e:
#     # Don't crash SIS if SSP isn't ready; you can still turn it on later
#     app.state.ssp = None
#     logger.info({"msg": "SSP not initialized at boot", "error": str(e)})
app.include_router(ssp.router, prefix="/ssp")   # ← add prefix
