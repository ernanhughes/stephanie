# sis/main.py
from __future__ import annotations

from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from stephanie.logging.json_logger import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from sis.routes import arena, db, pipelines, models, logs, plan_traces, documents, mars, learning
from sis.routes import casebooks as casebooks_routes
from sis.routes import chats, cards
from zoneinfo import ZoneInfo
from stephanie.components.gap.risk.api import create_router as create_gap_risk_router
from sis.routes import risk_ui # (new router below)
from sis.routes import explore_ui  # add with other route imports
from sis.routes import overnight_ui
from sis.routes import ssp as ssp_routes
from sis.routes import ssp_ui

import yaml

def datetimeformat(value, fmt="%Y-%m-%d %H:%M:%S"):
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value).strftime(fmt)
    if isinstance(value, datetime):
        return value.strftime(fmt)
    return str(value)



def load_config(path="sis/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def short_dt(value, fmt="%Y-%m-%d %H:%M", tz="Europe/Dublin"):
    if value is None:
        return ""
    # accept datetime or ISO string
    if isinstance(value, str):
        try:
            # minimal ISO parse
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return value[:16]  # last resort trim
    try:
        if value.tzinfo is None:
            value = value.replace(tzinfo=ZoneInfo("UTC"))
        value = value.astimezone(ZoneInfo(tz))
        return value.strftime(fmt)
    except Exception:
        return str(value)

cfg = load_config()
logger = JSONLogger("logs/sis.jsonl")
memory = MemoryTool(cfg=cfg, logger=logger)

app = FastAPI(title="Stephanie Insight System (SIS)")

# Put into app.state so routers can access without importing main
app.state.memory = memory
app.state.templates = Jinja2Templates(directory="sis/templates")
app.state.templates.env.globals["now"] = datetime.now
app.state.templates.env.filters["short_dt"] = short_dt
# after you set up templates
app.state.templates.env.filters["datetimeformat"] = datetimeformat
# set config
app.state.config = cfg



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

# After app = FastAPI(...)
class SISContainer:
    """Minimal DI bridge so GAP Risk can discover services already in SIS.
    Add attributes here as your stack grows (metrics_service, tiny, hrm, event bus, etc.).
    """
    def __init__(self, app):
        self.app = app
        # Preferred unified scorer if present (contract: score_text_pair)
        self.metrics_service = getattr(app.state, "metrics_service", None)

        # Tiny & HRM monitors (any attribute name you use; best-effort discovery)
        self.tiny_scorer = (
            getattr(app.state, "tiny_scorer", None)
            or getattr(app.state, "monitor_tiny", None)
            or getattr(app.state, "tiny", None)
        )
        self.hrm_scorer = getattr(app.state, "hrm_scorer", None) or getattr(app.state, "hrm", None)

        # Optional event publisher for arena/overlay; method: publish(topic, payload)
        self.event_publisher = getattr(app.state, "event_publisher", None)

# Bind container for GAP Risk orchestrator/router
app.state.container = SISContainer(app)

# Mount JSON API under /v1/gap/risk
app.include_router(create_gap_risk_router(app.state.container), prefix="/v1/gap/risk", tags=["gap-risk"])

# Mount SIS UI for risk
app.include_router(risk_ui.router)

try:
    from stephanie.components.ssp.substrate import SspComponent
    from stephanie.components.ssp.config import ensure_cfg
    from stephanie.utils.trace_logger import attach_to_app
    app.state.ssp = SspComponent(ensure_cfg(app.state.config))
    attach_to_app(app, jsonl_path="logs/plan_traces.jsonl", enable_stdout=False)
except Exception as e:
    # Don't crash SIS if SSP isn't ready; you can still turn it on later
    app.state.ssp = None
    logger.info({"msg": "SSP not initialized at boot", "error": str(e)})
app.include_router(ssp_ui.router)

app.include_router(ssp_routes.router)