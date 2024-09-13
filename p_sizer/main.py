from nicegui import ui


def show_quick_operation_page():
    ui.open("/quick_operation_page")


def show_trading_chart_page():
    ui.open("/trading_chart_page")


def show_position_page():
    ui.open("/position_page")


def show_settings_page():
    ui.open("/settings_page")


def show_template_page():
    ui.open("/template_page")


server_connected = True
# 创建菜单栏
with ui.header().classes("bg-blue-500") as header:
    with ui.row():
        ui.button("Menu", on_click=lambda: menu.open()).props("flat color=white")
        with ui.menu() as menu:
            ui.menu_item("Quick", on_click=show_quick_operation_page)
            ui.menu_item("Trading", on_click=show_trading_chart_page)
            ui.menu_item("Positions", on_click=show_position_page)
            ui.menu_item("Template", on_click=show_template_page)
            ui.menu_item("Settings", on_click=show_settings_page)
        # 根据服务器连接状态显示不同图标
        icon = ui.icon("wifi_off" if not server_connected else "wifi").props(
            f'color={ "red" if not server_connected else "green"}'
        )

soption='''


'''
echart = ui.echart()


ui.run()
