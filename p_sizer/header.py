from pathlib import Path
from typing import Optional

from nicegui import app, ui

from . import svg


HEADER_HTML = (Path(__file__).parent / "static" / "header.html").read_text()
STYLE_CSS = (Path(__file__).parent / "static" / "style.css").read_text()


def add_head_html() -> None:
    """Add the code from header.html and reference style.css."""
    ui.add_head_html(HEADER_HTML + f"<style>{STYLE_CSS}</style>")


def add_header(menu: Optional[ui.left_drawer] = None) -> None:
    """Create the page header."""
    menu_items = {
        "Quick": "/#Quick",
        "Position": "/#Position",
        "Chart": "/#Chart",
        "Settings": "/Settings",
        "Template": "/#Template",
  
    }
    dark_mode = ui.dark_mode(
        value=app.storage.browser.get("dark_mode"),
        on_change=lambda e: ui.run_javascript(f"""
        fetch('/dark_mode', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{value: {e.value}}}),
        }});
    """),
    )
    with ui.header().classes("items-center duration-200 p-0 px-4 no-wrap").style(
        "box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1)"
    ):
        if menu:
            ui.button(on_click=menu.toggle, icon="menu").props(
                "flat color=white round"
            ).classes("lg:hidden")
        with ui.link(target="/").classes("row gap-4 items-center no-wrap mr-auto"):
            svg.face().classes("w-8 stroke-white stroke-2 max-[610px]:hidden")
            ui.label("Position Sizer").classes("text-white text-2xl")

        with ui.row().classes("max-[1050px]:hidden"):
            for title_, target in menu_items.items():
                ui.link(title_, target).classes(replace="text-lg text-white")

        #search = Search()
        #search.create_button()

        with ui.element().classes("max-[420px]:hidden").tooltip(
            "Cycle theme mode through dark, light, and system/auto."
        ):
            ui.button(
                icon="dark_mode", on_click=lambda: dark_mode.set_value(None)
            ).props("flat fab-mini color=white").bind_visibility_from(
                dark_mode, "value", value=True
            )
            ui.button(
                icon="light_mode", on_click=lambda: dark_mode.set_value(True)
            ).props("flat fab-mini color=white").bind_visibility_from(
                dark_mode, "value", value=False
            )
            ui.button(
                icon="brightness_auto", on_click=lambda: dark_mode.set_value(False)
            ).props("flat fab-mini color=white").bind_visibility_from(
                dark_mode, "value", lambda mode: mode is None
            )
        server_connected = True
        icon = ui.icon("wifi_off" if not server_connected else "wifi").props(
            f'color={ "red" if not server_connected else "green"}'
        ).classes("fill-white scale-125 m-1")


        #add_star().classes("max-[550px]:hidden")

        with ui.row().classes("min-[1051px]:hidden"):
            with ui.button(icon="more_vert").props("flat color=white round"):
                with ui.menu().classes("bg-primary text-white text-lg"):
                    for title_, target in menu_items.items():
                        ui.menu_item(
                            title_,
                            on_click=lambda target=target: ui.navigate.to(target),
                        )
