from nicegui import ui
from p_sizer.header import add_head_html, add_header
from p_sizer.style import features, heading, link_target, section_heading, subtitle, title
from p_sizer.svg import face

import pandas as pd
import FinanceDataReader as fdr  # pip install   // Open Source Financial data reader

def create() -> None:
    """Create the content of the main page."""
    ui.context.client.content.classes('p-0 gap-0')
    add_head_html()
    add_header()
    ui.add_body_html('''
    <link rel="stylesheet" type="text/css" href="https://code.highcharts.com/css/stocktools/gui.css">
    <link rel="stylesheet" type="text/css" href="https://code.highcharts.com/css/annotations/popup.css">
    <script src="https://code.highcharts.com/stock/highstock.js"></script>
    <script src="https://code.highcharts.com/stock/indicators/indicators-all.js"></script>
    <script src="https://code.highcharts.com/stock/modules/drag-panes.js"></script>
    <script src="https://code.highcharts.com/modules/annotations-advanced.js"></script>
    <script src="https://code.highcharts.com/modules/price-indicator.js"></script>
    <script src="https://code.highcharts.com/modules/full-screen.js"></script>
    <script src="https://code.highcharts.com/modules/stock-tools.js"></script>
    <script src="https://code.highcharts.com/stock/modules/heikinashi.js"></script>
    <script src="https://code.highcharts.com/stock/modules/hollowcandlestick.js"></script>
    <script src="https://code.highcharts.com/modules/accessibility.js"></script>
    ''')

    with ui.row().classes('w-full h-screen items-center gap-8 pr-4 no-wrap into-section'):
        #face(half=True).classes('stroke-black dark:stroke-white w-[200px] md:w-[230px] lg:w-[300px]')
        getPriceChart('005930','삼성전자')
def getPriceChart(corpCode,corpName):
    # get Price Data // corpCode == stock code
    df = fdr.DataReader(corpCode,start='20200101')
    df = df.reset_index()
    df = df.round(2)

    # highchart series
    series_data = []
    df['Timestamp'] = pd.to_datetime(df['Date']).apply(lambda x: int(x.timestamp() * 1000))
    candlestick_series = {
        'type': 'candlestick',
        'name': 'Price',
        'id': 'main-series',
        'data': df[['Timestamp','Open', 'High', 'Low', 'Close']].values.tolist(),
    }
    series_data.append(candlestick_series)
    volume_series = {
        'type': 'column',
        'name': 'volume',
        'id': 'volume',
        'data': df[['Timestamp','Volume']].values.tolist(),
        'yAxis': 1,
        'pointWidth': 2
    }
    ohlc = df[['Timestamp','Open', 'High', 'Low', 'Close']].values.tolist()
    volume = df[['Timestamp','Volume']].values.tolist()
    series_data.append(volume_series)
    container = ui.row().classes('w-full h-screen')
    ui.run_javascript(f'''
        (async () => {{
            const ohlc = {ohlc}
            const volume = {volume}
            Highcharts.stockChart('c{container.id}', {{
                yAxis: [{{
                    labels: {{
                        align: 'left'
                    }},
                    height: '80%',
                    resize: {{
                        enabled: true
                    }}
                }}, {{
                    labels: {{
                        align: 'left'
                    }},
                    top: '80%',
                    height: '20%',
                    offset: 0
                }}],
                tooltip: {{
                    shape: 'square',
                    headerShape: 'callout',
                    borderWidth: 0,
                    shadow: false,
                    positioner: function (width, height, point) {{
                        const chart = this.chart;
                        let position;

                        if (point.isHeader) {{
                            position = {{
                                x: Math.max(
                                    // Left side limit
                                    chart.plotLeft,
                                    Math.min(
                                        point.plotX + chart.plotLeft - width / 2,
                                        // Right side limit
                                        chart.chartWidth - width - chart.marginRight
                                    )
                                ),
                                y: point.plotY
                            }};
                        }} else {{
                            position = {{
                                x: point.series.chart.plotLeft,
                                y: point.series.yAxis.top - chart.plotTop
                            }};
                        }}

                        return position;
                    }}
                }},
                series: [{{
                    type: 'candlestick',
                    id: 'price',
                    name: 'Price',
                    data: ohlc
                }}, {{
                    type: 'column',
                    id: 'volume',
                    name: 'volume',
                    data: volume,
                    yAxis: 1
                }}],
                responsive: {{
                    rules: [{{
                        condition: {{
                            maxWidth: 1024
                        }},
                        chartOptions: {{
                            rangeSelector: {{
                                inputEnabled: false
                            }}
                        }}
                    }}]
                }}
            }});
        }})();
    ''')



