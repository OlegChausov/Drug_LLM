#config.py
from pathlib import Path
from starlette.templating import Jinja2Templates


import starlette

import fastapi, starlette




BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


