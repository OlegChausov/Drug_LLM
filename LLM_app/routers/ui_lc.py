# LLM_app/routers/ui_lc.py

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from LLM_app.config import templates

router = APIRouter()


@router.get("/ui_lc", response_class=HTMLResponse)
def ui_lc(request: Request):
    return templates.TemplateResponse(request, "index_lc.html")


router1 = APIRouter()

@router1.get("/widget", response_class=HTMLResponse)
def widget(request: Request):
    return templates.TemplateResponse(request, "widget.html")