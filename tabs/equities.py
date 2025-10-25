# tabs/equities.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone, date
from typing import Dict, List, Tuple, Optional, Any
import time

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_searchbox import st_searchbox
import plotly.graph_objects as go

from tabs.yf_store import preload_hist_5y_daily, get_hist_sliced_years, get_hist_5y, get_meta

# yfinance es requerido para métricas (solo usado para fallback de sector si lo quisieras)
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

TRADING_DAYS = 252

# ============================================================================
# 🆕 FUNCIONES PARA LEER ACTIVOS DEL LEDGER
# ============================================================================

def _get_rv_from_ledger(ss) -> Dict[str, Dict[str, str]]:
    """
    Extrae activos de RV (RV_MX y RV_EXT) del ledger que están actualmente en el portafolio.
    Retorna dict con formato: {ticker: {"name": name, "index": index, "clase": clase}}
    """
    is_operating = ss.get("ops_operating", False)
    if not is_operating:
        return {}
    
    try:
        from tabs import ledger as ledger_module
        from tabs import runtime
        
        # Obtener fecha actual de operación
        act_as_of = runtime.get_act_as_of() or datetime.now().date()
        if isinstance(act_as_of, datetime):
            act_as_of = act_as_of.date()
        
        # Obtener ledger hasta la fecha
        led = ledger_module.get_ledger_until_act_as_of(ss)
        
        if led.empty:
            return {}
        
        # Reconstruir holdings
        holdings = {}
        for _, row in led.iterrows():
            kind = str(row.get("kind", "")).upper()
            symbol = str(row.get("symbol", "")).strip()
            qty = float(row.get("qty", 0.0))
            name = str(row.get("name", symbol))
            
            if kind == "BUY":
                if symbol not in holdings:
                    holdings[symbol] = {"qty": 0.0, "name": name, "clase": ""}
                holdings[symbol]["qty"] += qty
                holdings[symbol]["name"] = name
            elif kind == "SELL":
                if symbol in holdings:
                    holdings[symbol]["qty"] -= qty
        
        # Enriquecer con clases
        holdings = _enrich_holdings_with_classes(ss, holdings)
        
        # Filtrar solo RV (RV_MX, RV_EXT y FIBRA_MX) con qty > 0
        rv_holdings = {}
        for symbol, data in holdings.items():
            if data.get("qty", 0) > 1e-6:
                clase = data.get("clase", "")
                if clase in ["RV_MX", "RV_EXT", "FIBRA_MX"]:  # ← Incluye FIBRAs
                    # Inferir índice desde la clase
                    if clase == "FIBRA_MX":
                        index_name = "IPC"  # FIBRAs usan IPC como benchmark
                    elif clase == "RV_MX":
                        index_name = "IPC"
                    else:  # RV_EXT
                        index_name = "SP500"
                    
                    rv_holdings[symbol] = {
                        "name": data.get("name", symbol),
                        "index": index_name,
                        "clase": clase
                    }
        
        return rv_holdings
        
    except Exception as e:
        # Si hay error, no romper la app, solo retornar vacío
        return {}


def _enrich_holdings_with_classes(ss, holdings: Dict) -> Dict:
    """Enriquece holdings con clases de activo."""
    synthetic_prices = ss.get("synthetic_prices", {})
    opt_table = ss.get("optimization_table", pd.DataFrame())
    equity_selection = ss.get("equity_selection", {})
    rv_selection = ss.get("rv_selection", {})
    
    for symbol, data in holdings.items():
        if not data.get("clase"):
            # 1. Buscar en synthetic_prices
            if symbol in synthetic_prices:
                data["clase"] = synthetic_prices[symbol].get("clase", "")
                if not data.get("name"):
                    data["name"] = synthetic_prices[symbol].get("activo", symbol)
                continue
            
            # 2. Buscar en optimization_table
            if not opt_table.empty and "ticker" in opt_table.columns:
                mask = opt_table["ticker"] == symbol
                if mask.any():
                    data["clase"] = opt_table[mask].iloc[0].get("clase", "")
                    if not data.get("name"):
                        data["name"] = opt_table[mask].iloc[0].get("nombre", symbol)
                    continue
            
            # 3. Buscar en equity_selection
            if symbol in equity_selection:
                data["clase"] = "RV_MX"  # Asumimos que equity_selection es RV_MX
                if not data.get("name"):
                    eq_data = equity_selection[symbol]
                    if isinstance(eq_data, dict):
                        data["name"] = eq_data.get("name", symbol)
                continue
            
            # 4. Buscar en rv_selection
            if symbol in rv_selection:
                rv_data = rv_selection[symbol]
                if isinstance(rv_data, dict):
                    data["clase"] = rv_data.get("clase", "RV_MX")
                    if not data.get("name"):
                        data["name"] = rv_data.get("name", symbol)
                continue
            
            # 5. Inferir por sufijo del ticker
            symbol_upper = symbol.upper()
            if ".MX" in symbol_upper:
                data["clase"] = "RV_MX"
            elif any(ext in symbol_upper for ext in [".US", ".NYSE", ".NASDAQ", "^"]):
                data["clase"] = "RV_EXT"
            else:
                # Default: si no sabemos, asumir RV_MX
                data["clase"] = "RV_MX"
    
    return holdings

# ============================================================================
# FIN DE FUNCIONES PARA LEDGER
# ============================================================================

# ---------- Carga de universo ----------
@st.cache_data(show_spinner=False)
def load_universe() -> pd.DataFrame:
    base = Path(".").resolve()
    data_dir = base / "data"
    candidates = [
        data_dir / "ConstituentsWithYahoo.csv",
        data_dir / "constituents_with_yahoo.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "No encontré data/ConstituentsWithYahoo.csv ni data/constituents_with_yahoo.csv"
        )

    last_err = None
    df = None
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError as e:
            last_err = e
    if df is None:
        raise RuntimeError(f"Error cargando universo (encoding): {last_err}")

    needed = {"index", "yahoo", "name", "status"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    df = df.astype({"index":"string","yahoo":"string","name":"string","status":"string"}).copy()
    df["status"] = df["status"].str.lower().str.strip()

    df_ok = df[df["status"].isin(["ok", "found", "active"])].dropna(subset=["yahoo","name","index"])
    df_ok = df_ok.drop_duplicates(subset=["yahoo"], keep="first")
    return df_ok

# ---------- Mapeos índice/país y benchmarks ----------
INDEX2COUNTRY = {
    "SP500":"USA","S&P500":"USA","S&P 500":"USA","SPX":"USA","DJIA":"USA","NASDAQ":"USA","NDX":"USA","USA":"USA",
    "IPC":"MEX","S&P/BMV IPC":"MEX","BMV":"MEX","MEX":"MEX",
    "IBOV":"BRA","IBOVESPA":"BRA","B3":"BRA","BRA":"BRA",
    "ARG":"ARG",
    "TSX":"CAN","S&P/TSX":"CAN","TSX COMPOSITE":"CAN","CAN":"CAN",
    "FTSE100":"GBR","FTSE 100":"GBR","LSE":"GBR","GBR":"GBR","UK":"GBR",
    "DAX":"DEU","XETRA":"DEU","FWB":"DEU","DEU":"DEU",
    "CAC40":"FRA","CAC 40":"FRA","EURONEXT PARIS":"FRA","FRA":"FRA",
    "IBEX35":"ESP","IBEX 35":"ESP","BME":"ESP","ESP":"ESP",
    "FTSEMIB":"ITA","FTSE MIB":"ITA","BORSA ITALIANA":"ITA","ITA":"ITA",
    "SMI":"CHE","SIX":"CHE","CHE":"CHE",
    "OMXC25":"DNK","CSE":"DNK","DNK":"DNK",
    "OSEBX":"NOR","OSE":"NOR","NOR":"NOR",
    "EURONEXT":"NLD","NLD":"NLD",
    "NIKKEI225":"JPN","NIKKEI 225":"JPN","TSE":"JPN","JPN":"JPN",
    "SSE":"CHN","SZSE":"CHN","HANG SENG":"CHN","HKEX":"CHN","CHN":"CHN",
    "KOSPI":"KOR","KRX":"KOR","KOR":"KOR",
    "IND": "IND", "NSE": "IND", "BSE SENSEX": "IND",
    "TAIEX":"TWN","TWN":"TWN",
    "STI":"SGP","SGP":"SGP",
    "ASX200":"AUS","ASX 200":"AUS","AUS":"AUS",
}
COUNTRY_BENCH = {
    "USA": "^GSPC",
    "MEX": "^MXX",
    "BRA": "^BVSP",
    "CAN": "^GSPTSE",
    "GBR": "^FTSE",
    "DEU": "^GDAXI",
    "FRA": "^FCHI",
    "ESP": "^IBEX",
    "ITA": "FTSEMIB.MI",
    "CHE": "^SSMI",
    "DNK": "OMXC25.CO",
    "NOR": "OBX.OL",
    "NLD": "^AEX",
    "JPN": "^N225",
    "CHN": "^HSI",
    "KOR": "^KS11",
    "IND": "^BSESN",
    "TWN": "^TWII",
    "SGP": "^STI",
    "AUS": "^AXJO",
}
def _country_from_index(idx: str, ticker: str) -> str:
    v = (idx or "").strip().upper()
    if v in INDEX2COUNTRY:
        return INDEX2COUNTRY[v]
    t = (ticker or "").upper()
    if t.endswith(".MX"):
        return "MEX"
    if t.endswith(".SA"):
        return "BRA"
    return "USA"
def _benchmark_for_index(idx: str, ticker: str) -> str:
    country = _country_from_index(idx, ticker)
    return COUNTRY_BENCH.get(country, "^GSPC")

# ---------- Históricos (lee del store) ----------
@st.cache_data(show_spinner=True, ttl=1800)
def fetch_hist_yahoo(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    years_map = {"1y": 1.0, "3y": 3.0, "5y": 5.0}
    y = years_map.get(str(period).lower().strip(), 5.0)

    try:
        df = get_hist_5y(ticker) if y >= 5.0 else get_hist_sliced_years(ticker, y)
    except Exception:
        df = None

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    try:
        out.index = pd.to_datetime(out.index).tz_localize(None)
    except Exception:
        pass

    if "Adj Close" not in out.columns:
        if "Close" in out.columns:
            out["Adj Close"] = out["Close"]
        else:
            out["Adj Close"] = np.nan
    return out

def _currency_fallback(ticker: str) -> str:
    t = str(ticker).upper()
    if t.endswith(".MX"):
        return "MXN"
    if t.endswith(".SA"):
        return "BRL"
    return "USD"

# ---------- Sector (opcional, no crítico) ----------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_sector(ticker: str) -> str:
    if not YF_OK:
        return "—"
    try:
        tk = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        sector = info.get("sector") or info.get("industry") or "—"
        if isinstance(sector, str):
            sector = sector.strip()
        return sector or "—"
    except Exception:
        return "—"

# ---------- Métricas helpers ----------
def _annual_vol(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.size else np.nan

def _period_return_from_price(px: pd.Series) -> float:
    px = px.dropna()
    if px.size < 2:
        return np.nan
    return float(px.iloc[-1] / px.iloc[0] - 1.0)

def _avg_annual_return_from_daily(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.size == 0:
        return np.nan
    return float(r.mean() * TRADING_DAYS)

def _ytd_return_from_price(px: pd.Series) -> float:
    s = px.dropna()
    if s.size < 2:
        return np.nan
    jan1 = pd.Timestamp(year=pd.Timestamp.today().year, month=1, day=1)
    s_year = s[s.index >= jan1]
    if s_year.size < 2:
        return np.nan
    first = s_year.iloc[0]
    last = s_year.iloc[-1]
    if not np.isfinite(first) or first == 0:
        return np.nan
    return float(last / first - 1.0)

def _beta_vs(ra: pd.Series, rm: pd.Series) -> float:
    a = ra.dropna(); m = rm.dropna()
    if a.empty or m.empty:
        return np.nan
    aligned = pd.concat([a, m], axis=1, join="inner").dropna()
    if aligned.shape[0] < 60:
        return np.nan
    ra_al = aligned.iloc[:, 0]; rm_al = aligned.iloc[:, 1]
    var_m = rm_al.var(ddof=1)
    if var_m <= 0 or np.isnan(var_m):
        return np.nan
    cov_am = ra_al.cov(rm_al)
    return float(cov_am / var_m)

def _max_drawdown(px: pd.Series) -> float:
    s = px.dropna()
    if s.size < 2:
        return np.nan
    roll_max = s.cummax()
    dd = s / roll_max - 1.0
    return float(dd.min())

def _cagr_from_price(px: pd.Series) -> float:
    s = px.dropna()
    if s.size < 2:
        return np.nan
    dt_years = (s.index[-1] - s.index[0]).days / 365.25
    if dt_years <= 0:
        return np.nan
    return float((s.iloc[-1] / s.iloc[0]) ** (1.0 / dt_years) - 1.0)

def _sharpe_from_daily_returns(rets: pd.Series, rf_annual: float = 0.07) -> float:
    r = rets.dropna()
    if r.size == 0:
        return np.nan
    daily_rf = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0
    excess = r - daily_rf
    mu_excess = excess.mean() * TRADING_DAYS
    sig = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    if sig <= 0 or np.isnan(sig):
        return np.nan
    return float(mu_excess / sig)

def _fmt_pct(val: float, decimals: int = 2) -> str:
    if pd.isna(val):
        return "—"
    return f"{val*100:.{decimals}f}%"

def _fmt_num(val: float, d: int = 2) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:.{d}f}"

def _resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample OHLC data to a different frequency."""
    if df.empty:
        return df
    resampled = df.resample(freq).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()
    return resampled

def _tipo_from_index_or_ticker(nm: str, tkr: str, idx: str) -> str:
    tkr_u = (tkr or "").upper()
    idx_u = (idx or "").upper()
    nm_u = (nm or "").upper()
    if tkr_u.endswith(".MX") or "IPC" in idx_u or "BMV" in idx_u or "MEX" in idx_u:
        return "RV_MX"
    return "RV_EXT"

# ---------- Sugerencias búsqueda ----------
def _search_suggestions_factory(universe: pd.DataFrame):
    def fn(searchterm: str) -> List[Tuple[str, str]]:
        if not searchterm:
            return []
        s = searchterm.lower()
        results = []
        for _, r in universe.iterrows():
            nm = str(r["name"]).lower()
            tk = str(r["yahoo"]).lower()
            disp = str(r["display"]).lower()
            if s in nm or s in tk or s in disp:
                results.append((r["display"], r["yahoo"]))
        return results[:50]
    return fn

# ---------- Métricas de activo ----------
def compute_metrics_for_asset(
    ticker: str,
    idx_name: str,
    px5: Optional[pd.DataFrame],
    currency_hint: Optional[str] = None,
) -> Dict[str, Any]:
    if px5 is None or px5.empty or "Adj Close" not in px5.columns:
        return {
            "price": np.nan,
            "ytd": np.nan,
            "ret_1y": np.nan,
            "ret_prom_5y": np.nan,
            "vol_5y": np.nan,
            "beta_5y": np.nan,
            "maxdd_1y": np.nan,
            "ret_total_5y": np.nan,
            "cagr_5y": np.nan,
            "sharpe_5y": np.nan,
            "currency": currency_hint or _currency_fallback(ticker),
        }

    adj = px5["Adj Close"].dropna()
    if adj.empty:
        return {
            "price": np.nan,
            "ytd": np.nan,
            "ret_1y": np.nan,
            "ret_prom_5y": np.nan,
            "vol_5y": np.nan,
            "beta_5y": np.nan,
            "maxdd_1y": np.nan,
            "ret_total_5y": np.nan,
            "cagr_5y": np.nan,
            "sharpe_5y": np.nan,
            "currency": currency_hint or _currency_fallback(ticker),
        }

    curr_price = float(adj.iloc[-1])
    ret_full = adj.pct_change().dropna()

    ret_total_5y = _period_return_from_price(adj)
    cagr_5y = _cagr_from_price(adj)
    ret_prom_5y = _avg_annual_return_from_daily(ret_full)
    vol_5y = _annual_vol(ret_full)
    ytd_val = _ytd_return_from_price(adj)

    now = pd.Timestamp.today().tz_localize(None)
    cutoff_1y = now - pd.DateOffset(years=1)
    px_1y = adj[adj.index >= cutoff_1y]
    ret_1y_val = _period_return_from_price(px_1y)
    maxdd_1y = _max_drawdown(px_1y)

    bench_ticker = _benchmark_for_index(idx_name, ticker)
    df_bench = get_hist_5y(bench_ticker)
    beta_5y = np.nan
    if df_bench is not None and not df_bench.empty:
        adj_bench = df_bench.get("Adj Close", pd.Series(dtype=float))
        if not adj_bench.empty:
            ret_bench = adj_bench.pct_change().dropna()
            beta_5y = _beta_vs(ret_full, ret_bench)

    sharpe_val = _sharpe_from_daily_returns(ret_full, rf_annual=0.07)

    return {
        "price": curr_price,
        "ytd": ytd_val,
        "ret_1y": ret_1y_val,
        "ret_prom_5y": ret_prom_5y,
        "vol_5y": vol_5y,
        "beta_5y": beta_5y,
        "maxdd_1y": maxdd_1y,
        "ret_total_5y": ret_total_5y,
        "cagr_5y": cagr_5y,
        "sharpe_5y": sharpe_val,
        "currency": currency_hint or _currency_fallback(ticker),
    }

# ---------- Panel de detalle ----------
def _render_detail_panel(ticker: str, name: str, idx_name: str):
    st.markdown("---")
    
    # Obtener metadata y métricas
    try:
        meta = get_meta(ticker) or {}
    except Exception:
        meta = {}
    
    try:
        px5 = get_hist_5y(ticker)
    except Exception:
        px5 = None
    
    currency_md = meta.get("currency")
    
    try:
        m = compute_metrics_for_asset(
            ticker=ticker,
            idx_name=idx_name,
            px5=px5,
            currency_hint=currency_md,
        )
    except Exception:
        m = {"price": np.nan, "ytd": np.nan, "ret_1y": np.nan, "ret_prom_5y": np.nan,
             "vol_5y": np.nan, "beta_5y": np.nan, "maxdd_1y": np.nan,
             "ret_total_5y": np.nan, "cagr_5y": np.nan,
             "sharpe_5y": np.nan, "currency": currency_md or "—"}
    
    currency = currency_md or m.get("currency") or "—"
    sector = meta.get("sector") or "—"
    industry = meta.get("industry") or "—"
    exchange = meta.get("exchange") or "—"
    
    # UI
    st.markdown(f"### {name} ({ticker})")
    
    chip_cols = st.columns(9)
    price_val = m.get("price", np.nan)
    chip_cols[0].metric("Precio", "—" if (price_val is None or not np.isfinite(price_val)) else f"{price_val:,.2f}")
    chip_cols[1].metric("YTD", _fmt_pct(m.get("ytd", np.nan)))
    chip_cols[2].metric("Rendimiento 1Y", _fmt_pct(m.get("ret_1y", np.nan)))
    chip_cols[3].metric("Rendimiento 5Y", _fmt_pct(m.get("ret_prom_5y", np.nan)))
    chip_cols[4].metric("Volatilidad", _fmt_pct(m.get("vol_5y", np.nan)))
    chip_cols[5].metric("Máximo Draw Down", _fmt_pct(m.get("maxdd_1y", np.nan)))
    chip_cols[6].metric("Beta", _fmt_num(m.get("beta_5y", np.nan), d=3))
    chip_cols[7].metric("Sharpe", _fmt_num(m.get("sharpe_5y", np.nan), d=2))
    chip_cols[8].metric("Moneda", currency)

    left, right = st.columns([1.8, 2.2], gap="large")
    with left:
        st.caption(" • ".join([f"**Sector:** {sector}", f"**Industria:** {industry}", f"**Exchange:** {exchange}"]))
        summary = (meta.get("longBusinessSummary") or "").strip() if meta else ""
        city    = (meta.get("city") or "") if meta else ""
        state   = (meta.get("state") or "") if meta else ""
        country_i = (meta.get("country") or "") if meta else ""
        hq = ", ".join([p for p in [city, state, country_i] if p])
        employees = meta.get("fullTimeEmployees")
        website   = meta.get("website")
        founded   = meta.get("founded")
        ipo_year  = meta.get("ipo_year")

        if summary:
            max_chars = 900
            if len(summary) > max_chars:
                summary = summary[:max_chars].rsplit(" ", 1)[0] + "…"
            st.write(summary)

        bullets = []
        if hq: bullets.append(f"- **Sede:** {hq}")
        if founded or ipo_year: bullets.append(f"- **Fundación / IPO:** {founded or ipo_year}")
        if isinstance(employees, (int, float)) and np.isfinite(employees): bullets.append(f"- **Empleados:** {int(employees):,}")
        if website: bullets.append(f"- **Sitio:** {website}")
        if bullets: st.markdown("\n".join(bullets))
        else: st.caption("Sin detalles adicionales disponibles.")

    with right:
        try:
            px3 = get_hist_sliced_years(ticker, 3.0)
        except Exception:
            px3 = None

        df_plot = None
        if px3 is not None and not px3.empty and {"Open","High","Low","Close"}.issubset(px3.columns):
            df_plot = _resample_ohlc(px3[["Open","High","Low","Close"]], "W-FRI")

        if df_plot is not None and not df_plot.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=df_plot.index,
                open=df_plot["Open"], high=df_plot["High"],
                low=df_plot["Low"], close=df_plot["Close"],
                increasing_line_color="green", decreasing_line_color="red",
                name="Precio"
            )])
            fig.update_layout(
                xaxis_rangeslider_visible=False,
                height=420,
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"candle_{ticker}")
        else:
            st.info("No hay datos suficientes para graficar velas semanales de 3 años.")


# ---------- Callback multiselect ----------
def handle_multiselect_change(display2ticker: Dict[str, str], ticker2meta: Dict[str, Tuple[str,str]]):
    current_displays = st.session_state.eq_picks_raw
    current_tickers: List[str] = []
    for d in current_displays:
        t = display2ticker.get(d)
        if t and t not in current_tickers:
            current_tickers.append(t)

    new_selection: Dict[str, Dict[str, str]] = {}
    for t in current_tickers:
        nm, ix = ticker2meta.get(t, (t, ""))
        new_selection[t] = {"name": nm, "index": ix}

    st.session_state.equity_selection = new_selection

    if st.session_state.eq_detail_ticker not in new_selection:
        st.session_state.eq_detail_ticker = next(iter(new_selection.keys()), None)

    st.session_state.eq_picks = current_displays

# ---------- Render principal ----------
def render():

    # -------- Estado --------
    ss = st.session_state
    if "equity_selection" not in ss:
        ss.equity_selection = {}
    if "eq_picks" not in ss:
        ss.eq_picks = []
    if "eq_picks_raw" not in ss:
        ss.eq_picks_raw = []
    if "eq_detail_ticker" not in ss:
        ss.eq_detail_ticker = None
    if "add_ticker_consumed" not in ss:
        ss.add_ticker_consumed = None

    # -------- Universo --------
    try:
        universe = load_universe()
    except Exception as e:
        st.error(f"Error cargando universo: {e}")
        return

    # ============================================================================
    # 🆕 INTEGRACIÓN CON LEDGER: Agregar activos de RV del ledger
    # ============================================================================
    ledger_rv = _get_rv_from_ledger(ss)
    
    # Agregar activos del ledger a equity_selection si no están ya
    for ticker, meta in ledger_rv.items():
        if ticker not in ss.equity_selection:
            ss.equity_selection[ticker] = {
                "name": meta["name"],
                "index": meta["index"]
            }
    
    # ============================================================================
    # FIN DE INTEGRACIÓN CON LEDGER
    # ============================================================================

    # -------- Mapeos --------
    universe = universe.copy()
    universe["display"] = universe["name"].str.strip() + " (" + universe["yahoo"].str.strip() + ")"
    display2ticker = dict(zip(universe["display"], universe["yahoo"]))
    ticker2meta: Dict[str, Tuple[str, str]] = dict(zip(universe["yahoo"], zip(universe["name"], universe["index"])))
    ticker2display = {t: d for d, t in display2ticker.items()}
    
    # 🧩 Asegurar displays para tickers ya seleccionados, aunque no estén en el universo ni en el ledger
    for t, meta_sel in ss.equity_selection.items():
        if t not in ticker2display:
            disp = f"{meta_sel.get('name', t)} ({t})"
            ticker2display[t] = disp
            display2ticker[disp] = t
        if t not in ticker2meta:
            ticker2meta[t] = (meta_sel.get("name", t), meta_sel.get("index", ""))

    # 🧩 Agregar activos del ledger a mapeos (no-op si ops_operating=False)
    for ticker, meta in ledger_rv.items():
        if ticker not in ticker2meta:
            ticker2meta[ticker] = (meta["name"], meta["index"])
        if ticker not in ticker2display:
            disp = f"{meta['name']} ({ticker})"
            ticker2display[ticker] = disp
            display2ticker[disp] = ticker

    # ✅ Las opciones deben incluir universo + lo que ya está seleccionado + (opcional) ledger
    all_display_options = list(
        set(universe["display"].tolist())
        | set(ticker2display.values())
        | set(ss.get("eq_picks", []))
    )
    
    # -------- Sincronización inicial --------
    if ss.equity_selection:
        current_tickers_in_order = list(ss.equity_selection.keys())
        
        # 🔧 Agregar nuevos activos que están en equity_selection pero no en eq_picks
        new_displays = []
        for t in current_tickers_in_order:
            if t in ticker2display:
                display = ticker2display[t]
                if display not in ss.eq_picks:
                    new_displays.append(display)
        
        # Agregar los nuevos a eq_picks
        if new_displays:
            ss.eq_picks.extend(new_displays)

        # Asegura que lo seleccionado exista en options (por si algo cambió)
        ss.eq_picks = [d for d in ss.eq_picks if d in all_display_options]
        ss.eq_picks_raw = ss.eq_picks.copy()
    else:
        ss.eq_picks = []
        ss.eq_picks_raw = []

    # -------- Searchbox --------
    col_search, _ = st.columns([2, 6])
    with col_search:
        st.caption("Buscador de Acciones")
        selected_ticker = st_searchbox(
            search_function=_search_suggestions_factory(universe),
            placeholder="Escribe la empresa o el ticker",
            key="add_ticker_search",
        )

    if selected_ticker:
        if selected_ticker not in ss.equity_selection:
            nm, ix = ticker2meta.get(selected_ticker, (selected_ticker, ""))
            ss.equity_selection[selected_ticker] = {"name": nm, "index": ix}
            display_to_add = ticker2display.get(selected_ticker)
            if display_to_add:
                ss.eq_picks.append(display_to_add)
                ss.eq_picks_raw = ss.eq_picks.copy()
            ss.eq_detail_ticker = selected_ticker
            st.rerun()

    # -------- Multiselect --------
    sel = ss.equity_selection
    if not sel:
        st.info("Utiliza el buscador para seleccionar las emisoras que deseas analizar.")
        return
    
    st.multiselect(
        "Activos Seleccionados",
        options=all_display_options,
        key="eq_picks_raw",
        on_change=handle_multiselect_change,
        args=(display2ticker, ticker2meta),
    )

    if ss.eq_detail_ticker and ss.eq_detail_ticker not in sel:
        ss.eq_detail_ticker = None

    # -------- Orden de la tabla --------
    tickers_in_order: List[str] = []
    for disp in ss.eq_picks:
        t = display2ticker.get(disp)
        if t and t in ss.equity_selection:
            tickers_in_order.append(t)

    # -------- Preload 5y (emisoras + benchmarks) --------
    try:
        benchs = []
        for t in tickers_in_order:
            idx_name = ss.equity_selection[t].get("index", "")
            benchs.append(_benchmark_for_index(idx_name, t))
        preload_hist_5y_daily(list({*tickers_in_order, *benchs}))
    except Exception:
        pass

    # -------- Cache local 5y tras preload --------
    hist5_cache: Dict[str, pd.DataFrame] = {}
    for t in tickers_in_order:
        try:
            hist5_cache[t] = get_hist_5y(t)
        except Exception:
            hist5_cache[t] = pd.DataFrame()

    # -------- Pre-carga meta --------
    metas: Dict[str, dict] = {}
    for tkr in tickers_in_order:
        try:
            metas[tkr] = get_meta(tkr) or {}
        except Exception:
            metas[tkr] = {}

    # -------- Construcción de filas --------
    rows: List[Dict[str, Any]] = []
    for tkr in tickers_in_order:
        meta_row = sel[tkr]
        nm = meta_row["name"]
        idx_name = meta_row["index"]

        md = metas.get(tkr, {})
        sector = md.get("sector") or "—"
        currency_md = md.get("currency")

        try:
            m = compute_metrics_for_asset(
                ticker=tkr,
                idx_name=idx_name,
                px5=hist5_cache.get(tkr),
                currency_hint=currency_md,
            )
        except Exception:
            m = {"price": np.nan, "ytd": np.nan, "ret_1y": np.nan, "ret_prom_5y": np.nan,
                 "vol_5y": np.nan, "beta_5y": np.nan, "maxdd_1y": np.nan,
                 "ret_total_5y": np.nan, "cagr_5y": np.nan,
                 "sharpe_5y": np.nan, "currency": currency_md or "—"}

        row_currency = currency_md or m.get("currency") or "—"
        tipo = _tipo_from_index_or_ticker(nm, tkr, idx_name)

        rows.append({
            "Empresa": nm,
            "Ticker": tkr,
            "Tipo": tipo,
            "Sector": sector,
            "Precio": "—" if (m.get("price") is None or not np.isfinite(m.get("price"))) else f"{m['price']:,.2f}",
            "YTM": _fmt_pct(m.get("ytd", np.nan)),
            "Rendimiento 1Y": _fmt_pct(m.get("ret_1y", np.nan)),
            "Rendimiento 5Y": _fmt_pct(m.get("ret_prom_5y", np.nan)),
            "Volatilidad": _fmt_pct(m.get("vol_5y", np.nan)),
            "Máximo Draw Down": _fmt_pct(m.get("maxdd_1y", np.nan)),
            "Beta": "—" if pd.isna(m.get("beta_5y")) else f"{m['beta_5y']:.3f}",
            "Sharpe": "—" if pd.isna(m.get("sharpe_5y")) else f"{m['sharpe_5y']:.2f}",
            "Moneda": row_currency,
        })

    # -------- Render tabla --------
    st.caption(f"{len(rows)} activos seleccionados.")
    out = pd.DataFrame(rows)[[
        "Empresa", "Ticker", "Tipo", "Sector",
        "Precio", "YTM", "Rendimiento 1Y", "Rendimiento 5Y",
        "Volatilidad", "Máximo Draw Down", "Beta", "Sharpe", "Moneda"
    ]]

    event = st.dataframe(
        out,
        key="eq_table",
        width="stretch",
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
    )

    # -------- Selección de fila --------
    if isinstance(event, dict):
        sel_rows = event.get("selection", {}).get("rows", [])
    else:
        sel_rows = getattr(getattr(event, "selection", None), "rows", []) or []

    if sel_rows:
        i = sel_rows[0]
        try:
            tkr_clicked = str(out.iloc[i]["Ticker"]).strip()
        except Exception:
            tkr_clicked = ""
        if tkr_clicked and tkr_clicked in ss.equity_selection:
            ss.eq_detail_ticker = tkr_clicked
    else:
        if not ss.eq_detail_ticker and sel:
            ss.eq_detail_ticker = next(iter(sel.keys()), None)

    if ss.eq_detail_ticker and ss.eq_detail_ticker in sel:
        tkr_det = ss.eq_detail_ticker
        meta_det_row = sel[tkr_det]
        md_det = metas.get(tkr_det) or get_meta(tkr_det) or {}
        _render_detail_panel(
            tkr_det,
            meta_det_row.get("name", tkr_det),
            meta_det_row.get("index", ""),
        )
