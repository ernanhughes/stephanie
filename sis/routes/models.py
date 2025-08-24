from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sis.utils.model_tree import get_model_tree

router = APIRouter()

@router.get("/models", response_class=HTMLResponse)
def view_models(request: Request):
    templates = request.app.state.templates
    model_tree = "\n".join(get_model_tree("models"))
    return templates.TemplateResponse(
        "models.html",
        {"request": request, "model_tree": model_tree, "active_page": "models"}
    )
