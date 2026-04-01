import os
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from tabulate import tabulate
from vnstock import Quote

# =====================
# CONFIG & PARAMETERS
# =====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

try:
    from vn100_list import VN100
except ImportError:
    VN100 = [
        "HPG", "SSI", "VND", "DIG", "DGC", "VNM", "FPT", "MWG",
        "TCB", "VPB", "MBB", "STB", "ACB", "KBC", "IDC"
    ]

PORTFOLIO = VN100
LOOKBACK_DAYS = 300
TR_RANGE = 60
ATR_WIN = 14
AVG_VOL_WIN = 20
RSI_WIN = 14
SLEEP = 0.6
SOURCES = ("VCI", "TCBS")

# --- Risk & Filter Rules ---
MIN_RR_EARLY = 2.0
MIN_RR_SWING = 2.2
MIN_AVG_VOL = 50_000
MIN_VAL_BN = 2.0
MAX_EXTENDED_PCT = 0.12
RSI_MAX_BUY = 75

# --- Event / candle rules ---
BREAKOUT_VOL_RATIO = 1.35
BREAKOUT_CNH = 0.65
LPS_PULLBACK_ATR = 1.2
LPS_VOL_RATIO_MAX = 1.0

# --- SSI request headers ---
SSI_HEADERS = {
    "sec-ch-ua-platform": '"Windows"',
    "Referer": "https://iboard.ssi.com.vn/",
    "Accept-Language": "vi",
    "sec-ch-ua": '"Chromium";v="146", "Not-A.Brand";v="24", "Microsoft Edge";v="146"',
    "sec-ch-ua-mobile": "?0",
    "Device-ID": "EE10EA63-E21F-4C05-AFE9-09F0CD0F6405",
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36 Edg/146.0.0.0",
}


# =====================
# 1. HELPER FUNCTIONS
# =====================
def send_telegram_msg(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Thiếu TELEGRAM_TOKEN hoặc TELEGRAM_CHAT_ID. Bỏ qua gửi Telegram.")
        return None

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, data=data, timeout=20)
        print(f"📨 Telegram status: {response.status_code}")
        return response.json()
    except Exception as e:
        print(f"❌ Lỗi gửi Telegram: {e}")
        return None


def hose_tick(price: float) -> float:
    if price < 10:
        return 0.01
    if price < 50:
        return 0.05
    return 0.1


def round_hose(price: float) -> float:
    if price is None or not np.isfinite(price):
        return price
    t = hose_tick(float(price))
    return round(float(price) / t) * t


def fmt(price, dec=2):
    if price is None or not np.isfinite(price):
        return ""
    return f"{round_hose(float(price)):,.{dec}f}"


def get_val_bn(avg_vol: float, close: float) -> float:
    if avg_vol is None or close is None:
        return 0.0
    return (avg_vol * (close * 1000)) / 1e9


def calc_rr(entry, stop, target):
    if None in (entry, stop, target):
        return None
    risk = float(entry) - float(stop)
    reward = float(target) - float(entry)
    return round(reward / risk, 2) if risk > 0 and reward > 0 else None


# =====================
# 2. INDICATORS & ANALYSIS
# =====================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    pc = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - pc).abs()
    tr3 = (df["low"] - pc).abs()
    df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))
    df["atr"] = df["tr"].ewm(alpha=1 / ATR_WIN, adjust=False).mean()

    df["spread"] = df["high"] - df["low"]
    df["close_near_high"] = np.where(
        df["spread"] > 0,
        1 - (df["high"] - df["close"]) / df["spread"],
        0.0
    )

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / RSI_WIN, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / RSI_WIN, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()

    return df


def mid_trend(last) -> str:
    if last["close"] > last["ema20"] > last["ema50"]:
        return "UP"
    if last["close"] < last["ema20"] < last["ema50"]:
        return "DOWN"
    return "SIDE"


def check_structure_tightness(df: pd.DataFrame, box_h: float, box_l: float) -> str:
    avg_atr = df["atr"].tail(20).mean()
    if not np.isfinite(avg_atr) or avg_atr <= 0:
        return "Unknown"
    ratio = (box_h - box_l) / avg_atr
    if ratio < 3.5:
        return "TIGHT"
    if ratio > 6.5:
        return "LOOSE"
    return "NORMAL"


def check_macd_divergence(df: pd.DataFrame, lookback=60) -> str:
    if len(df) < lookback + 10:
        return "No"

    sub = df.iloc[-lookback:].copy()
    pivots = []
    arr = sub["low"].values

    for i in range(3, len(arr) - 3):
        if arr[i] == np.min(arr[i - 3:i + 4]):
            pivots.append(sub.index[i])

    if len(pivots) < 2:
        return "No"

    p1, p2 = pivots[-2], pivots[-1]
    if df.loc[p2, "low"] < df.loc[p1, "low"] and df.loc[p2, "macd"] > df.loc[p1, "macd"]:
        return "YES"
    return "No"


def detect_event(df: pd.DataFrame, box_h: float, box_l: float, vol_ratio: float) -> str:
    last = df.iloc[-1]
    close = last["close"]
    ema20 = last["ema20"]
    ema50 = last["ema50"]
    atr = last["atr"]
    cnh = last["close_near_high"]

    if close > box_h and vol_ratio >= BREAKOUT_VOL_RATIO and cnh >= BREAKOUT_CNH:
        return "Breakout"
    if close >= box_h * 0.995 and cnh >= 0.65 and vol_ratio >= 1.1:
        return "SOS"

    dist = min(abs(close - ema20), abs(close - ema50))
    if dist <= LPS_PULLBACK_ATR * atr and close > box_l and vol_ratio <= LPS_VOL_RATIO_MAX:
        return "LPS"

    return "None"


def classify_signal(df, box_h, box_l, trend, event, vol_ratio, tightness, macd_div, nn_net_buy):
    last = df.iloc[-1]
    atr = last["atr"]
    close = last["close"]
    ema20 = last["ema20"]
    ema50 = last["ema50"]
    rsi = last["rsi"]
    cnh = last["close_near_high"]

    tgt_breakout = round_hose(box_h + (box_h - box_l))
    tgt_range_top = round_hose(box_h)

    notes = []
    if pd.notna(nn_net_buy) and nn_net_buy > 0:
        notes.append(f"Tây gom {int(nn_net_buy / 1000)}k")
    if macd_div == "YES":
        notes.append("MACD Div(+)")
    if rsi < 30:
        notes.append("Oversold")

    def _sanity(e, s, t):
        if None in (e, s, t):
            return None, None, None, None
        s = e - 0.8 * atr if s >= e else s
        t = e + 1.8 * atr if t <= e else t
        e, s, t = round_hose(e), round_hose(s), round_hose(t)
        return e, s, t, calc_rr(e, s, t)

    if trend in ("DOWN", "SIDE") and ((pd.notna(nn_net_buy) and nn_net_buy > 100000) or macd_div == "YES") and cnh >= 0.6:
        e, s, t, rr = _sanity(close, last["low"] - 0.6 * atr, ema50)
        return ("BOTTOM_FISHING" if rr and rr >= 2.0 else "WATCH_BOTTOM"), e, s, t, rr, " + ".join(notes)

    if vol_ratio >= 1.5 and cnh >= 0.7 and close > ema20 and rsi < RSI_MAX_BUY and (close - ema20) / ema20 <= MAX_EXTENDED_PCT:
        e, s, t, rr = _sanity(close, min(ema20 - 0.3 * atr, last["low"] - 0.7 * atr), close + 2.0 * atr)
        return "MOMENTUM_WEEK", e, s, t, rr, f"Vol burst{' (Loose)' if tightness == 'LOOSE' else ''}"

    if trend == "UP" and event in ("LPS", "SOS", "Breakout") and rsi >= 45:
        s = (
            last["low"] - 0.7 * atr if event == "LPS"
            else box_h - 0.7 * atr if event == "SOS"
            else box_h - 0.8 * atr
        )
        tar = tgt_breakout if event == "Breakout" else tgt_range_top
        e, s, t, rr = _sanity(close, s, tar)
        return ("SWING_CONFIRMED" if rr and rr >= MIN_RR_SWING else "WATCH"), e, s, t, rr, f"{event} Uptrend"

    if close > ema20 and close > ema50 and close < box_h and tightness != "LOOSE" and (vol_ratio >= 1.2 or cnh >= 0.65):
        e, s, t, rr = _sanity(close, min(ema50 - 0.4 * atr, last["low"] - 0.6 * atr), tgt_range_top)
        if rr and rr >= MIN_RR_EARLY:
            return "EARLY_SWING", e, s, t, rr, "Reclaim MA"

    return "NONE", None, None, None, None, ""


# =====================
# 3. DATA FETCHING
# =====================
def fetch_history(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    for src in SOURCES:
        try:
            df = Quote(symbol=symbol, source=src).history(
                start=start,
                end=end,
                interval="1D"
            )
            if df is None or df.empty:
                continue

            df.columns = [c.lower() for c in df.columns]
            if "time" in df.columns:
                df = df.sort_values("time")
            df = df.reset_index(drop=True)

            if {"open", "high", "low", "close", "volume"}.issubset(set(df.columns)):
                return df
        except Exception as e:
            print(f"⚠️ Lỗi lịch sử {symbol} từ {src}: {e}")
            continue

    return None


def fetch_ssi_foreign_data():
    print("🔄 Đang lấy dữ liệu khối ngoại trực tuyến từ SSI...")

    exchanges = ["hose", "hnx", "upcom"]
    all_data = []

    for ex in exchanges:
        response = None
        last_error = None
        url = f"https://iboard-query.ssi.com.vn/stock/exchange/{ex}?boardId=MAIN"

        for attempt in range(3):
            try:
                print(f"\n🌐 Fetching {ex.upper()}... attempt {attempt + 1}/3")
                response = requests.get(url, headers=SSI_HEADERS, timeout=20)
                print(f"➡️ HTTP status: {response.status_code}")
                break
            except Exception as e:
                last_error = e
                print(f"❌ {ex.upper()} attempt {attempt + 1} failed: {e}")
                time.sleep(2)

        if response is None:
            print(f"🚨 {ex.upper()}: Không gọi được API. Last error: {last_error}")
            continue

        if response.status_code != 200:
            print(f"⚠️ {ex.upper()}: HTTP {response.status_code}")
            continue

        try:
            content = response.json()
        except Exception as e:
            print(f"❌ {ex.upper()}: JSON parse lỗi: {e}")
            print(response.text[:500])
            continue

        if isinstance(content, list):
            items = content
        elif isinstance(content, dict):
            if isinstance(content.get("data"), list):
                items = content["data"]
            elif isinstance(content.get("items"), list):
                items = content["items"]
            else:
                items = [content]
        else:
            items = []

        print(f"📊 {ex.upper()}: {len(items)} records")

        ok_rows = 0
        for item in items:
            if not isinstance(item, dict):
                continue

            symbol = str(item.get("stockSymbol", "")).strip().upper()
            buy_vol = pd.to_numeric(item.get("buyForeignQtty"), errors="coerce")
            sell_vol = pd.to_numeric(item.get("sellForeignQtty"), errors="coerce")

            if not symbol or pd.isna(buy_vol) or pd.isna(sell_vol):
                continue

            all_data.append({
                "Mã": symbol,
                "Net_Buy": float(buy_vol) - float(sell_vol)
            })
            ok_rows += 1

        print(f"✅ {ex.upper()}: parsed {ok_rows} rows")

    if not all_data:
        print("🚨 Không có dữ liệu khối ngoại nào được lấy!")
        return pd.DataFrame(columns=["Mã", "Net_Buy"])

    df = pd.DataFrame(all_data)
    df["Mã"] = df["Mã"].astype(str).str.strip().str.upper()
    df = df.groupby("Mã", as_index=False)["Net_Buy"].sum()

    print("\n📊 SAMPLE DF_NN:")
    print(df.head())

    return df


# =====================
# 4. SCANNER RUNNER
# =====================
def analyze_symbol(symbol, df_nn):
    symbol = str(symbol).strip().upper()

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    df = fetch_history(symbol, start, end)
    if df is None or len(df) < 80:
        return {"Mã": symbol, "Signal": "AVOID"}

    df = add_indicators(df)

    last = df.iloc[-1]
    avg_vol = float(df["volume"].tail(AVG_VOL_WIN).mean())
    recent = df.tail(TR_RANGE)
    box_h = float(recent["high"].max())
    box_l = float(recent["low"].min())
    vol_ratio = float(last["volume"] / avg_vol) if avg_vol > 0 else 1.0

    nn_row = df_nn[df_nn["Mã"].astype(str).str.strip().str.upper() == symbol]
    nn_buy = float(nn_row["Net_Buy"].values[0]) if not nn_row.empty else np.nan

    macd_div = check_macd_divergence(df)
    event = detect_event(df, box_h, box_l, vol_ratio)
    trend = mid_trend(last)
    tightness = check_structure_tightness(df, box_h, box_l)

    sig, e, s, t, rr, note = classify_signal(
        df, box_h, box_l, trend, event, vol_ratio, tightness, macd_div, nn_buy
    )

    return {
        "Mã": symbol,
        "Giá": fmt(last["close"]),
        "RSI": f"{last['rsi']:.0f}",
        "GTGD_BN": f"{get_val_bn(avg_vol, last['close']):.1f}",
        "NN_MuaRong": f"{int(nn_buy / 1000)}k" if pd.notna(nn_buy) else "N/A",
        "MACD_Div": macd_div,
        "Event": event,
        "Trend": trend,
        "Signal": sig,
        "Entry": fmt(e) if e else "",
        "Stop": fmt(s) if s else "",
        "Target": fmt(t) if t else "",
        "RR": f"{rr:.2f}" if rr else "",
        "Note": note,
    }


def build_telegram_message(df_show: pd.DataFrame) -> str:
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    msg = f"🚀 *SMART SCAN PRO*\\n_{now_str}_\\n"

    if df_show.empty:
        msg += "\nKhông có tín hiệu phù hợp hôm nay."
        return msg

    for cat, name in [
        ("BOTTOM", "🎯 BOTTOM CATCH"),
        ("MOMENTUM", "🔥 MOMENTUM"),
        (["SWING_CONFIRMED", "EARLY_SWING"], "📈 SWING"),
    ]:
        sub = (
            df_show[df_show["Signal"].str.contains(cat, na=False)]
            if isinstance(cat, str)
            else df_show[df_show["Signal"].isin(cat)]
        )

        if not sub.empty:
            msg += f"\n*{name}:*\n"
            for _, r in sub.head(10).iterrows():
                nn_text = r["NN_MuaRong"] if str(r["NN_MuaRong"]).strip() else "N/A"
                note_text = r["Note"] if str(r["Note"]).strip() else "-"
                rr_text = r["RR"] if str(r["RR"]).strip() else "-"
                msg += (
                    f"• *{r['Mã']}* | Giá: {r['Giá']} | RR: {rr_text}\n"
                    f"  NN: {nn_text} | {note_text}\n"
                )

    return msg


def run_scanner():
    print(f"\n🔔 [SYSTEM] Bắt đầu Smart Scan Pro: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    df_nn = fetch_ssi_foreign_data()

    rows = []
    for i, s in enumerate(PORTFOLIO):
        print(f"[{i + 1}/{len(PORTFOLIO)}] Quét mã {s}...", end="\r")
        rows.append(analyze_symbol(s, df_nn))
        time.sleep(SLEEP)

    print()
    df_all = pd.DataFrame(rows)

    df_show = df_all[df_all["Signal"].isin([
        "MOMENTUM_WEEK",
        "EARLY_SWING",
        "SWING_CONFIRMED",
        "BOTTOM_FISHING",
        "WATCH_BOTTOM"
    ])].copy()

    if not df_show.empty:
        print("\n" + "=" * 110 + "\n")
        print(tabulate(df_show, headers="keys", tablefmt="grid"))
    else:
        print("ℹ️ Không có tín hiệu phù hợp.")

    msg = build_telegram_message(df_show)
    send_telegram_msg(msg)


if __name__ == "__main__":
    run_scanner()
