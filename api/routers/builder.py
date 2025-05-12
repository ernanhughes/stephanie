
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="api/templates")

@router.get("/builder", response_class=HTMLResponse)
async def get_builder_ui(request: Request):
    return templates.TemplateResponse("builder.html", {"request": request})
