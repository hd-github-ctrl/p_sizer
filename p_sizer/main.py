import os
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware


from nicegui import app, ui
from p_sizer import main_page, svg, fly

on_fly = fly.setup()
app.add_static_files('/favicon', str(Path(__file__).parent / 'favicon'))
app.add_static_files('/fonts', str(Path(__file__).parent / 'fonts'))
app.add_static_files('/static', str(Path(__file__).parent / 'static'))
app.add_static_file(local_file=svg.PATH / 'logo.png', url_path='/logo.png')
app.add_static_file(local_file=svg.PATH / 'logo_square.png', url_path='/logo_square.png')


app.add_middleware(SessionMiddleware, secret_key=os.environ.get('NICEGUI_SECRET_KEY', ''))

@ui.page('/')
def _main_page() -> None:
    main_page.create()
    
ui.run(uvicorn_reload_includes='*.py, *.css, *.html', reload=not on_fly, reconnect_timeout=10.0)
