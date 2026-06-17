import pandas as pd
import requests
import numpy as np
import concurrent.futures
import warnings
import gradio as gr

warnings.filterwarnings("ignore")

BASE_URL = "https://data-api.binance.vision/api/v3"
QUOTE_ASSET = 'USDT'

def get_high_volume_pairs(min_volume_threshold):
    url = f"{BASE_URL}/ticker/24hr"
    try:
        data = requests.get(url).json()
    except Exception:
        return []

    valid_coins = []
    for item in data:
        symbol = item['symbol']
        if symbol.endswith(QUOTE_ASSET) and 'UPUSDT' not in symbol and 'DOWNUSDT' not in symbol:
            try:
                vol_usdt = float(item['quoteVolume'])
                if vol_usdt >= min_volume_threshold:
                    valid_coins.append({
                        'coin': symbol,
                        'volume': vol_usdt,
                        'change': float(item['priceChangePercent']),
                        'price': float(item['lastPrice'])
                    })
            except ValueError:
                continue

    valid_coins.sort(key=lambda x: x['volume'], reverse=True)
    return valid_coins

def fetch_data(symbol, interval):
    url = f"{BASE_URL}/klines?symbol={symbol}&interval={interval}&limit=1000"
    res = requests.get(url).json()
    if isinstance(res, dict) and 'code' in res:
        return None
    df = pd.DataFrame(res, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'])
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    return df

def calculate_macd(series, fast, slow, signal):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def check_strategy(df, tf_string, lookback_candles):
    close = df['close']
    hlc3 = df['hlc3']
    ema21 = close.ewm(span=21, adjust=False).mean()
    hist1 = calculate_macd(hlc3, 8, 21, 5)
    hist2 = calculate_macd(close, 50, 200, 10)

    bothGreen = (hist1 > 0) & (hist2 > 0)
    longCondition1 = bothGreen & (~bothGreen.shift(1, fill_value=False))
    longCondition2 = (hist1 > 0) & (hist1.shift(1) <= 0) & (hist2 > 0)
    longCondition3 = (close > ema21) & (close.shift(1) <= ema21.shift(1)) & (hist1 > 0) & (hist2 > 0)
    earlyEntry = (hist1 < 0) & (hist1 > hist1.shift(1)) & (hist1.shift(1) > hist1.shift(2)) & (hist2 > 0)

    raw_buy_trigger = (longCondition1 | longCondition2 | longCondition3 | earlyEntry) & (close > ema21)
    exit_trigger = close < ema21

    buy_arr = raw_buy_trigger.to_numpy()
    exit_arr = exit_trigger.to_numpy()

    in_trade = False
    bars_since_entry = -1

    for i in range(len(buy_arr)):
        if not in_trade:
            if buy_arr[i]:
                in_trade = True       
                bars_since_entry = 0  
        else:
            bars_since_entry += 1
            if exit_arr[i]:           
                in_trade = False      
                bars_since_entry = -1

    if in_trade and (0 <= bars_since_entry < lookback_candles):
        return True
    return False

def run_web_screener(timeframe, min_volume, lookback):
    filtered_coins = get_high_volume_pairs(min_volume)
    
    if len(filtered_coins) == 0:
        return "<div style='color:#ef4444; padding:20px; text-align:center; font-weight:bold;'>❌ No coins match that minimum volume. Try lowering the number.</div>"
    
    buy_signals = []

    def process_coin(coin_data):
        try:
            df = fetch_data(coin_data['coin'], timeframe)
            if df is not None and len(df) > 250:
                is_buy = check_strategy(df, timeframe, lookback)
                if is_buy:
                    return coin_data 
        except Exception:
            pass
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        results = executor.map(process_coin, filtered_coins)
        for result in results:
            if result:
                buy_signals.append(result)

    buy_signals.sort(key=lambda x: x['volume'], reverse=True)

    table_rows = ""
    if len(buy_signals) > 0:
        for rank, item in enumerate(buy_signals, start=1):
            change_color = "#10b981" if item['change'] >= 0 else "#ef4444"
            change_sign = "+" if item['change'] >= 0 else ""
            table_rows += f"""
            <tr class="report-row">
                <td class="col-rank">{rank:02d}</td>
                <td class="col-coin">{item['coin']}</td>
                <td class="col-price">${item['price']:,.4f}</td>
                <td class="col-change" style="color: {change_color}; font-weight: 600;">{change_sign}{item['change']:.2f}%</td>
                <td class="col-vol">${item['volume']:,.0f}</td>
            </tr>
            """
    else:
        table_rows = """<tr><td colspan="5" class="no-results">❌ No coins passed both the volume filter and the strategy trigger right now.</td></tr>"""

    html_dashboard = f"""
    <style>
        .market-report-container {{ --bg-main: #ffffff; --bg-card: #f8f9fa; --text-main: #1f2937; --text-muted: #6b7280; --border-color: #e5e7eb; --header-bg: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); --header-text: #ffffff; --accent-color: #10b981; --row-hover: #f3f4f6; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; width: 100%; max-width: 650px; margin: 10px auto; border: 1px solid var(--border-color); border-radius: 12px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1); overflow: hidden; background-color: var(--bg-main); color: var(--text-main); box-sizing: border-box; }}
        @media (prefers-color-scheme: dark) {{ .market-report-container {{ --bg-main: #111827; --bg-card: #1f2937; --text-main: #f9fafb; --text-muted: #9ca3af; --border-color: #374151; --header-bg: linear-gradient(135deg, #2563eb 0%, #1e40af 100%); --row-hover: #374151; --accent-color: #34d399; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5); }} }}
        .report-header {{ background: var(--header-bg); color: var(--header-text); padding: 18px; text-align: center; }}
        .report-header h2 {{ margin: 0; font-size: 20px; font-weight: 700; }}
        .report-meta {{ padding: 16px 20px; background-color: var(--bg-card); border-bottom: 1px solid var(--border-color); font-size: 14px; display: flex; flex-direction: column; gap: 8px; }}
        .meta-row {{ display: flex; justify-content: space-between; }}
        .meta-label {{ color: var(--text-muted); }}
        .meta-value {{ font-weight: 600; color: var(--text-main); }}
        .meta-accent {{ font-weight: 700; color: var(--accent-color); font-size: 15px; }}
        .table-wrapper {{ width: 100%; overflow-x: auto; }}
        .report-table {{ width: 100%; border-collapse: collapse; font-size: 14px; text-align: left; }}
        .report-table th {{ padding: 12px 15px; background-color: var(--bg-card); color: var(--text-muted); text-transform: uppercase; font-size: 11px; border-bottom: 1px solid var(--border-color); white-space: nowrap; }}
        .report-row {{ border-bottom: 1px solid var(--border-color); transition: background-color 0.2s ease; }}
        .report-row:hover {{ background-color: var(--row-hover); }}
        .report-table td {{ padding: 12px 15px; white-space: nowrap; }}
        .col-rank {{ text-align: center; color: var(--text-muted); width: 40px; }}
        .col-coin {{ font-weight: 600; color: var(--text-main); }}
        .col-price, .col-change, .col-vol {{ text-align: right; }}
        .col-vol {{ color: var(--text-muted); font-weight: 500; }}
        .no-results {{ text-align: center; padding: 24px; color: #ef4444; font-weight: 500; white-space: normal !important; }}
    </style>
    <div class="market-report-container">
        <div class="report-header"><h2>🎯Screener Results🎯</h2></div>
        <div class="report-meta">
            <div class="meta-row"><span class="meta-label">Timeframe:</span><span class="meta-value">{timeframe}</span></div>
            <div class="meta-row"><span class="meta-label">Max Window:</span><span class="meta-value">{lookback} candle(s)</span></div>
            <div class="meta-row"><span class="meta-label">Min Volume:</span><span class="meta-value">${min_volume:,.0f} USDT</span></div>
            <div class="meta-row"><span class="meta-label">Total Active Setups:</span><span class="meta-accent">{len(buy_signals)}</span></div>
        </div>
        <div class="table-wrapper">
            <table class="report-table">
                <thead>
                    <tr><th style="text-align: center;">#</th><th>Asset</th><th style="text-align: right;">Price</th><th style="text-align: right;">24h %</th><th style="text-align: right;">24H Vol (USDT)</th></tr>
                </thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>
    </div>
    """
    return html_dashboard

custom_css = """
/* ---- MOBILE ONLY STYLES ---- */
@media screen and (max-width: 768px) {
    input, select, textarea, button,
    .gradio-container input,
    .gradio-container select,
    .gradio-container textarea,
    div[role="combobox"],
    div[role="listbox"],
    .wrap-inner * {
        font-size: 16px !important;
        touch-action: manipulation !important;
    }
    
    .app-header-container h1 {
        font-size: 20px !important; 
        text-align: center !important;
        white-space: nowrap !important;
    }
    
    .app-header-container p {
        text-align: center !important;
    }
}

/* ---- GENERAL STYLES ---- */
.app-header-container {
    margin-top: 60px !important;
}
.app-header-container h1 {
    margin-bottom: 0px !important;
}
.app-header-container p {
    color: #9ca3af !important; 
    font-size: 0.95em !important; 
    font-style: italic !important;
    margin-top: 5px !important;
}
"""

anti_pinch_zoom_head = """
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
<script>
    document.addEventListener('gesturestart', function (e) {
        e.preventDefault();
    });
    document.addEventListener('touchstart', function(e) {
        if (e.touches.length > 1) {
            e.preventDefault();
        }
    }, { passive: false });
    document.addEventListener('touchmove', function(e) {
        if (e.touches.length > 1) {
            e.preventDefault();
        }
    }, { passive: false });
</script>
"""

with gr.Blocks(title="Crypto Breakout Scanner", theme=gr.themes.Soft(), css=custom_css, head=anti_pinch_zoom_head) as app:
    
    gr.HTML("""
    <div class="app-header-container">
        <h1>📈Crypto Trend Screener📈</h1>
        <p>Adjust your filters below and click <strong>Scan Market</strong> to fetch live signals.</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            tf_input = gr.Dropdown(choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"], value="4h", label="Timeframe", filterable=False)
            vol_input = gr.Number(value=1000000, label="Minimum 24H Volume (USDT)")
            look_input = gr.Slider(minimum=1, maximum=20, step=1, value=2, label="Lookback Window (Candles)")
            scan_btn = gr.Button("🚀 Scan Market", variant="primary")
        with gr.Column(scale=2):
            output_html = gr.HTML(value="<div style='text-align:center; padding:40px; color:#888; border: 2px dashed #ddd; border-radius: 10px;'>Click 'Scan Market' to generate the report. It takes about 10-15 seconds.</div>")
            
    scan_btn.click(fn=run_web_screener, inputs=[tf_input, vol_input, look_input], outputs=output_html)

app.launch()
