# sis/main.py
from __future__ import annotations

from datetime import datetime
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from stephanie.logging.json_logger import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from sis.routes import db, pipelines, models, logs, plan_traces, documents, mars
from sis.routes import casebooks as casebooks_routes
from sis.routes import chats, arena
from datetime import datetime
from zoneinfo import ZoneInfo

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






# Static assets
app.mount("/static", StaticFiles(directory="sis/static"), name="static")

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

