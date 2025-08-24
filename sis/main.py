# sis/main.py
from datetime import datetime
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from stephanie.logging.json_logger import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from sis.routes import db, pipelines, models
import yaml

def load_config(path="sis/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()
logger = JSONLogger("sis.jsonl")
memory = MemoryTool(cfg=cfg, logger=logger)

app = FastAPI(title="Stephanie Insight System (SIS)")

# Put into app.state so routers can access without importing main
app.state.memory = memory
app.state.templates = Jinja2Templates(directory="sis/templates")
app.state.templates.env.globals["now"] = datetime.now

# Static assets
app.mount("/static", StaticFiles(directory="sis/static"), name="static")

# Include routers
app.include_router(pipelines.router)
app.include_router(models.router)
app.include_router(db.router)