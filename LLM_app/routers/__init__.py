# LLM_app/routers/__init__.py


from .ws_lc import router as ws_lc_router
from .ui_lc import router as ui_lc_rourer
from .ui_lc import router1 as widget_rourer


all_routers = [
    ws_lc_router,
    ui_lc_rourer,
    widget_rourer,
    

]