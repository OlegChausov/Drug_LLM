
# LLM_app/main.py

from fastapi import FastAPI
from .routers import all_routers
from fastapi.staticfiles import StaticFiles
import os


app = FastAPI()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


for r in all_routers:
    app.include_router(r)

