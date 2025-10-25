# tabs/fx.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Set
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_searchbox import st_searchbox
from pathlib import Path
import os
import datetime as dt

from tabs.yf_store import preload_hist_5y_daily, get_hist_sliced_years, get_hist_5y

TRADING_DAYS = 252

# Nombre del archivo que vamos a buscar
_FX_FILE_NAME = "fx.csv"

# ============================================================================
# 🆕 FUNCIONES PARA LEER ACTIVOS DE FX DEL LEDGER
# ============================================================================

def _get_fx_from_ledger(ss) -> Dict[str, Dict[str, str]]:
    """
    Extrae activos de FX del ledger que están actualmente en el portafolio.
    Retorna dict con formato: {pair_code: {"name": name, "ticker": ticker, "inverse": bool}}
    """
    is_operating = ss.get("ops_operating", False)
    if not is_operating:
        return {}
    
    try:
        from tabs import ledger as ledger_module
        from tabs import runtime
        
        # Obtener fecha actual de operación
        act_as_of = runtime.get_act_as_of() or dt.datetime.now().date()
        if isinstance(act_as_of, dt.datetime):
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
        holdings = _enrich_fx_holdings_with_classes(ss, holdings)
        
        # Filtrar solo FX con qty > 0
        fx_holdings = {}
        for symbol, data in holdings.items():
            if data.get("qty", 0) > 1e-6:
                clase = data.get("clase", "")
                if clase == "FX":
                    # Extraer el par de divisas del símbolo
                    pair_code = _extract_pair_from_symbol(symbol)
                    
                    # Determinar si necesita inversión
                    inverse = _needs_inversion(symbol)
                    
                    fx_holdings[pair_code] = {
                        "name": data.get("name", pair_code),
                        "ticker": symbol,
                        "inverse": inverse
                    }
        
        return fx_holdings
        
    except Exception as e:
        # Si hay error, no romper la app, solo retornar vacío
        return {}


def _enrich_fx_holdings_with_classes(ss, holdings: Dict) -> Dict:
    """Enriquece holdings con clases de activo."""
    synthetic_prices = ss.get("synthetic_prices", {})
    opt_table = ss.get("optimization_table", pd.DataFrame())
    fx_selection = ss.get("fx_selection", {})
    
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
            
            # 3. Buscar en fx_selection
            pair_code = _extract_pair_from_symbol(symbol)
            if pair_code in fx_selection:
                data["clase"] = "FX"
                if not data.get("name"):
                    fx_data = fx_selection[pair_code]
                    if isinstance(fx_data, dict):
                        data["name"] = fx_data.get("name", pair_code)
                continue
            
            # 4. Inferir por patrón del ticker (típicos de FX en Yahoo)
            symbol_upper = symbol.upper()
            if symbol_upper.endswith("=X") or len(symbol_upper.replace("=X", "")) == 6:
                data["clase"] = "FX"
            else:
                # Default: si no sabemos, no asumir que es FX
                data["clase"] = ""
    
    return holdings


def _extract_pair_from_symbol(symbol: str) -> str:
    """
    Extrae el par de divisas de un símbolo de Yahoo Finance.
    Ej: 'MXNUSD=X' -> 'MXN/USD', 'EURUSD=X' -> 'EUR/USD'
    """
    symbol = str(symbol).strip().upper()
    # Remover '=X' si existe
    core = symbol.replace("=X", "")
    
    # Si tiene 6 caracteres alfabéticos, asumir formato AAABBB
    if len(core) == 6 and core.isalpha():
        return f"{core[:3]}/{core[3:]}"
    
    # Si ya tiene formato AAA/BBB o AAA-BBB
    if "/" in core or "-" in core:
        core = core.replace("-", "/")
        return core
    
    # Fallback: retornar como está
    return symbol


def _needs_inversion(symbol: str) -> bool:
    """
    Determina si un símbolo de Yahoo necesita inversión para mostrarse correctamente.
    Esto depende de cómo Yahoo cotiza cada par.
    Por ejemplo, USDMXN=X cotiza dólares por peso, pero queremos pesos por dólar.
    """
    symbol = str(symbol).strip().upper()
    core = symbol.replace("=X", "")
    
    # Casos comunes que necesitan inversión
    # (Ajustar según la convención de tu aplicación)
    needs_inv = [
        "USDMXN",  # Yahoo cotiza USD/MXN pero queremos MXN/USD
        "USDBRL",  # Yahoo cotiza USD/BRL pero queremos BRL/USD
    ]
    
    if core in needs_inv:
        return True
    
    return False

# ============================================================================
# FIN DE FUNCIONES PARA LEDGER
# ============================================================================

# ===== Nombres bonitos por divisa =====
CCY_FULLNAME = {
    "USD": "Dólar Estadounidense", "MXN": "Peso Mexicano", "EUR": "Euro", "CAD": "Dólar Canadiense",
    "GBP": "Libra Esterlina", "JPY": "Yen Japonés", "CNY": "Yuan Chino", "BRL": "Real Brasileño",
    "CHF": "Franco Suizo", "AUD": "Dólar Australiano", "SEK": "Corona Sueca", "NOK": "Corona Noruega",
    "ZAR": "Rand Sudafricano", "CLP": "Peso Chileno", "COP": "Peso Colombiano", "PEN": "Sol Peruano",
}
def _fullname(ccy: str) -> str:
    return CCY_FULLNAME.get(ccy, ccy)

def pretty_fx_label(pair_code: str) -> str:
    """
    'BRL/AUD' -> 'Real Brasileño/Dólar Australiano (BRL-AUD)'
    'MXN/USD' -> 'Peso Mexicano/Dólar Estadounidense (MXN-USD)'
    """
    s = str(pair_code or "").strip().upper()
    m = re.match(r"^([A-Z]{3})/([A-Z]{3})$", s)
    if not m:
        return pair_code
    a, b = m.group(1), m.group(2)
    fa = CCY_FULLNAME.get(a, a)
    fb = CCY_FULLNAME.get(b, b)
    return f"{fa}/{fb} ({a}-{b})"

def extract_pair_codes(s: str) -> str:
    """
    Extrae 'AAA/BBB' desde:
      - 'AAA/BBB'
      - 'Nombre Bonito (AAA/BBB)'
      - 'Nombre Bonito (AAA-BBB)'
    """
    t = str(s or "").strip().upper()
    m = re.search(r"\(([A-Z]{3})[-/]([A-Z]{3})\)\s*$", t)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    m2 = re.match(r"^([A-Z]{3})/([A-Z]{3})$", t)
    if m2:
        return f"{m2.group(1)}/{m2.group(2)}"
    return t  # fallback

# ==========================
# 1. UTILIDADES Y CÁLCULOS
# ==========================
def _annual_vol(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.size else np.nan

def _avg_annual_return(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() * TRADING_DAYS) if r.size else np.nan

def _series_from_yahoo(ticker: str, period: str = "5y", *, inverse: bool = False) -> Tuple[pd.Series, pd.DataFrame]:
    # Mapeo de period -> años y lectura desde el pool 5y/1d del yf_store
    period = str(period or "5y").lower().strip()
    years_map = {"5y": 5.0, "3y": 3.0, "1y": 1.0}
    years = years_map.get(period, 5.0)

    try:
        df = get_hist_5y(ticker) if years >= 5.0 else get_hist_sliced_years(ticker, years)
    except Exception:
        df = None

    if df is None or df.empty or "Adj Close" not in df.columns:
        return pd.Series(dtype=float), pd.DataFrame()

    px = df["Adj Close"].astype(float)
    if inverse:
        px = 1.0 / px

    r = px.pct_change(); r.name = "ret"
    out_df = pd.DataFrame({"Adj Close": px}); out_df.index = df.index
    return r, out_df


def _mu_sigma_fx_5y(ticker: str, *, inverse: bool = False) -> Tuple[float, float]:
    for period in ("5y", "3y", "1y"):
        r, _ = _series_from_yahoo(ticker, period=period, inverse=inverse)
        if r.empty:
            continue
        mu = _avg_annual_return(r); sigma = _annual_vol(r)
        if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
            return float(mu), float(sigma)
    return 0.0, 0.12


def _last_price(tkr: str, *, inverse: bool = False) -> float:
    _, df = _series_from_yahoo(tkr, period="1y", inverse=inverse)
    if df is None or df.empty or "Adj Close" not in df.columns:
        return np.nan
    return float(df["Adj Close"].dropna().iloc[-1])

def _max_drawdown_1y(tkr: str, *, inverse: bool = False) -> float:
    _, df = _series_from_yahoo(tkr, period="1y", inverse=inverse)
    if df is None or df.empty or "Adj Close" not in df.columns:
        return np.nan
    px = df["Adj Close"].dropna(); roll_max = px.cummax()
    dd_series = px / roll_max - 1.0
    return float(dd_series.min())

def _ret_1y(tkr: str, *, inverse: bool = False) -> float:
    _, df = _series_from_yahoo(tkr, period="1y", inverse=inverse)
    if df is None or df.empty or "Adj Close" not in df.columns:
        return np.nan
    px = df["Adj Close"].dropna()
    if len(px) < 2:
        return np.nan
    return float(px.iloc[-1] / px.iloc[0] - 1.0)

def _beta_vs_usdmxn(ticker: str, period: str = "5y", *, inverse: bool = False) -> float:
    bench_tkr = "MXN=X"; r_a, _ = _series_from_yahoo(ticker, period=period, inverse=inverse)
    r_b, _ = _series_from_yahoo(bench_tkr, period=period, inverse=False)
    if r_a.empty or r_b.empty:
        return np.nan
    df = pd.concat([r_a, r_b], axis=1, join="inner").dropna()
    df.columns = ["ra", "rb"]; varb = float(df["rb"].var(ddof=1))
    if varb <= 0:
        return np.nan
    cov = float(np.cov(df["ra"], df["rb"], ddof=1)[0, 1]); return cov / varb

def _ytd_return_from_price(px: pd.Series) -> float:
    s = px.dropna()
    if s.size < 2:
        return np.nan
    jan1 = pd.Timestamp(year=pd.Timestamp.today().year, month=1, day=1)
    s_year = s[s.index >= jan1]
    if s_year.size < 2:
        return np.nan
    first = s_year.iloc[0]; last = s_year.iloc[-1]
    if not np.isfinite(first) or first == 0:
        return np.nan
    return float(last / first - 1.0)

def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
        return pd.DataFrame()
    o = df["Open"].resample(rule).first()
    h = df["High"].resample(rule).max()
    l = df["Low"].resample(rule).min()
    c = df["Close"].resample(rule).last()
    out = pd.concat([o, h, l, c], axis=1)
    out.columns = ["Open", "High", "Low", "Close"]
    return out.dropna(how="any")

def _invert_ohlc_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ["Open", "High", "Low", "Close"]:
        if col in out.columns:
            out[col] = 1.0 / out[col].astype(float)
    if {"High", "Low"}.issubset(out.columns):
        hh = out[["Open","High","Low","Close"]].max(axis=1); ll = out[["Open","High","Low","Close"]].min(axis=1)
        out["High"] = hh; out["Low"]  = ll
    return out

def _fmt_pct_eq(x, d=2):
    try:
        return f"{x*100:.{d}f}%"
    except Exception:
        return "—"

def _is_code_pair(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{3}/[A-Z]{3}", str(s or "")))

def _make_pair_label(code_pair: str) -> str:
    return f"{code_pair} ({code_pair})"

# ----------------------------------------------------
# 2. CARGA DE CSV
# ----------------------------------------------------
def _resolve_fx_path(name: str) -> Path | None:
    try:
        base = Path(".").resolve(); data_dir = base / "data"
        candidates = [data_dir / name, data_dir / name.lower(), data_dir / name.upper()]
        return next((p for p in candidates if p.exists()), None)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _load_fx_csv() -> pd.DataFrame:
    path = _resolve_fx_path(_FX_FILE_NAME)
    if path is None:
        raise FileNotFoundError(f"No encontré {_FX_FILE_NAME} en la carpeta data/")
    df = pd.read_csv(path)
    cols = {"base", "quote", "pair", "yahoo_symbol", "method"}
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")

    # Mantén todo en MAYÚSCULAS excepto 'method' en minúsculas
    for c in ["base", "quote", "pair", "yahoo_symbol"]:
        df[c] = df[c].astype(str).str.strip().str.upper()
    df["method"] = df["method"].astype(str).str.strip().str.lower()

    return df

def _catalog_from_csv() -> List[Dict[str, object]]:
    """
    Devuelve opciones únicas por par mostrado ('AAA/BBB' y 'BBB/AAA'),
    evitando duplicados cuando el CSV ya trae ambas direcciones.
    Preferimos el símbolo nativo (inverse=False) y solo usamos inversa si no hay directo.
    """
    try:
        df = _load_fx_csv()
    except (FileNotFoundError, ValueError):
        return []
    except Exception:
        return []

    # Candidatos crudos (dos por fila: directo y su invertido)
    raw: List[Dict[str, object]] = []
    rank = 0
    for _, row in df.iterrows():
        a, b = row["base"], row["quote"]
        sym = row["yahoo_symbol"]
        method = row["method"]  # 'direct' o 'inverse'
        rank += 1

        # Entrada 1: como viene en el CSV (a/b)
        raw.append({
            "ticker": sym,
            "name": f"{a}/{b}",
            "inverse": (method == "inverse"),
            "rank": rank
        })

        # Entrada 2: la contraria (b/a) usando el mismo símbolo
        # Si el símbolo del CSV es directo para a/b, entonces para b/a hay que invertir.
        # Si el símbolo del CSV es inverse para a/b, entonces para b/a no hay que invertir.
        raw.append({
            "ticker": sym,
            "name": f"{b}/{a}",
            "inverse": (method == "direct"),
            "rank": rank
        })

    # Deduplicado por 'name', preferimos inverse=False (símbolo directo nativo)
    best_by_name: Dict[str, Dict[str, object]] = {}
    for it in raw:
        name = it["name"]
        cur = best_by_name.get(name)
        if cur is None:
            best_by_name[name] = it
        else:
            # Si ya existe, preferimos el que NO requiere inverse
            if cur.get("inverse", True) and not it.get("inverse", True):
                best_by_name[name] = it
            # Si ambos requieren/no requieren inverse, mantenemos el actual

    # Devuelve estable por rank y luego alfabético
    out = list(best_by_name.values())
    out.sort(key=lambda d: (d.get("rank", 0), d.get("name", "")))
    return out

# ----------------------------------------------------
# 3. PERSISTENCIA Y SINCRONIZACIÓN
# ----------------------------------------------------
def _ensure_fx_selection() -> None:
    """
    Garantiza que st.session_state['fx_selection'] se derive del basket, sin limpiar por accidente.
    - Si no hay basket pero sí picks => intenta reconstruir basket y reintenta.
    - Solo sobrescribe fx_selection cuando haya algo válido (evita flicker/lag).
    - Mantiene un backup fx_selection_last_good por si un render intermedio trae vacío.
    """
    st.session_state.setdefault("fx_picks", [])
    st.session_state.setdefault("fx_basket", [])
    st.session_state.setdefault("fx_selection", {})
    st.session_state.setdefault("fx_selection_last_good", {})

    basket = st.session_state.get("fx_basket", [])
    picks = st.session_state.get("fx_picks", [])

    if (not basket) and picks:
        try:
            _init_fx_state_from_picks()
            basket = st.session_state.get("fx_basket", [])
        except Exception:
            pass

    if not basket and not picks:
        st.session_state["fx_selection"] = {}
        st.session_state["fx_selection_last_good"] = {}
        return

    fx_sel_new: Dict[str, Dict[str, object]] = {}
    if basket:
        for it in basket:
            ticker = str(it.get("ticker", "")).strip()
            if not ticker:
                continue
            inverse = bool(it.get("inverse", False))
            pretty_name = str(it.get("name", "")).strip()
            if not pretty_name:
                m = re.search(r"\(([A-Z]{3}/[A-Z]{3})\)\s*$", str(it.get("name", "")))
                pretty_name = (m.group(1) if m else ticker)

            mu5, sig5 = _mu_sigma_fx_5y(ticker, inverse=inverse)
            mu = float(mu5) if np.isfinite(mu5) else 0.0
            sigma = float(sig5) if (np.isfinite(sig5) and sig5 > 0) else 0.12

            # Deriva el par estándar AAA/BBB
            pair = ""
            if ticker:
                # Ejemplos comunes de Yahoo: MXNBRL=X, EURUSD=X, USDMXN=X
                core = ticker.replace("=X", "")
                if len(core) == 6 and core.isalpha():
                    pair = f"{core[:3]}/{core[3:]}"
            if not pair:
                # Fallback: intenta extraer AAA/BBB del pretty_name
                m2 = re.search(r"([A-Z]{3})/([A-Z]{3})", pretty_name.upper())
                if m2:
                    pair = f"{m2.group(1)}/{m2.group(2)}"
                else:
                    pair = pretty_name  # Último recurso

            fx_sel_new[pretty_name] = {
                "name": pretty_name,
                "pair": pair,              # ← Nuevo campo normalizado (MXN/BRL)
                "mu": mu,
                "sigma": sigma,
                "yahoo_ticker": ticker,
                "inverse": inverse
        }


    if not fx_sel_new:
        fx_last = st.session_state.get("fx_selection_last_good", {})
        if fx_last:
            st.session_state["fx_selection"] = fx_last
        else:
            st.session_state["fx_selection"] = {}
        return

    fx_cur = st.session_state.get("fx_selection", {})
    if fx_cur != fx_sel_new:
        st.session_state["fx_selection"] = fx_sel_new
        st.session_state["fx_selection_last_good"] = fx_sel_new.copy()

def _get_fx_catalog_and_mappings() -> Tuple[List[str], Dict[str, str], Dict[str, Dict[str, object]]]:
    """Genera el catálogo de divisas y los mapeos necesarios."""
    options = _catalog_from_csv()
    def _key_for(opt: Dict[str, object]) -> str:
        return f"{opt['ticker']}|inv={int(bool(opt.get('inverse', False)))}"
    keys = [_key_for(o) for o in options]
    codes_list = [o["name"] for o in options]  # Lista de códigos simples (MXN/USD)
    key2opt = {_key_for(o): o for o in options}
    codes2key = dict(zip(codes_list, keys))    # Mapeo de código simple a key
    return codes_list, codes2key, key2opt

def _init_fx_state_from_picks() -> None:
    """Reconstruye fx_basket a partir de fx_picks (si fx_basket está vacío)."""
    st.session_state.setdefault("fx_picks", [])
    st.session_state.setdefault("fx_basket", [])

    v = st.session_state.get("fx_picks")
    if v is None:
        st.session_state["fx_picks"] = []
    elif not isinstance(v, list):
        st.session_state["fx_picks"] = list(v) if v else []

    if st.session_state["fx_basket"] or not st.session_state["fx_picks"]:
        return

    try:
        codes_list, codes2key, key2opt = _get_fx_catalog_and_mappings()
    except Exception:
        return

    st.session_state["fx_basket"] = []
    seen_keys: Set[str] = set()

    for code_label in st.session_state["fx_picks"]:
        if isinstance(code_label, str):
            m = re.search(r"\(([A-Z]{3}/[A-Z]{3})\)\s*$", code_label.strip())
            code_simple = m.group(1) if m else code_label.strip()
        else:
            continue

        k = codes2key.get(code_simple)
        if not k or k in seen_keys:
            continue
        seen_keys.add(k)

        opt = key2opt.get(k)
        if not opt:
            continue

        # PERSISTE nombre bonito
        st.session_state["fx_basket"].append({
            "key": k,
            "ticker": opt["ticker"],
            "name": pretty_fx_label(code_simple),
            "inverse": bool(opt.get("inverse", False)),
        })

# ---------- CALLBACK UI (multiselect FX) ----------
def _handle_fx_multiselect_change(codes2key: Dict[str, str], key2opt: Dict[str, Dict[str, object]]):
    """
    Sincroniza el multiselect (fx_picks_raw_widget) con la fuente de verdad (fx_picks),
    y reconstruye fx_basket acorde. Mantiene orden, evita reapariciones.
    """
    ss = st.session_state
    ss.setdefault("fx_picks", [])
    ss.setdefault("fx_picks_raw", [])
    ss.setdefault("fx_basket", [])

    raw = ss.get("fx_picks_raw_widget") or []
    picks_new: List[str] = []
    for code in raw:
        code = str(code).strip()
        if code and code not in picks_new:
            picks_new.append(code)

    ss["fx_picks"] = picks_new
    ss["fx_picks_raw"] = picks_new

    new_basket: List[Dict[str, object]] = []
    seen = set()
    for code in picks_new:
        k = codes2key.get(code)
        if not k or k in seen:
            continue
        seen.add(k)
        opt = key2opt.get(k)
        if not opt:
            continue
        new_basket.append({
            "key": k,
            "ticker": opt["ticker"],
            "name": pretty_fx_label(code),  # <-- siempre bonito
            "inverse": bool(opt.get("inverse", False)),
        })

    ss["fx_basket"] = new_basket

    # Ajusta detalle si quedó fuera
    if ss.get("fx_detail_key") and all(it["key"] != ss["fx_detail_key"] for it in new_basket):
        ss["fx_detail_key"] = (new_basket[0]["key"] if new_basket else None)

# ----------------------------------------------------
# 4. RENDER PRINCIPAL Y LÓGICA DE DETALLE
# ----------------------------------------------------
def _fx_metrics_row(tkr: str, display_name: str, pair_codes: str, *, inverse: bool) -> Dict[str, float | str]:
    mu5, sig5 = _mu_sigma_fx_5y(tkr, inverse=inverse); px_last = _last_price(tkr, inverse=inverse)
    ret1y = _ret_1y(tkr, inverse=inverse); mdd1y = _max_drawdown_1y(tkr, inverse=inverse)
    sharpe5 = (mu5 / sig5) if (np.isfinite(mu5) and np.isfinite(sig5) and sig5 > 0) else np.nan
    return {
        "Divisa": display_name,
        "Precio actual": round(px_last, 6) if np.isfinite(px_last) else np.nan,
        "Rendimiento 1Y": round((ret1y or np.nan) * 100, 2) if ret1y is not None else np.nan,
        "Rendimiento 5Y": round(mu5 * 100, 2) if np.isfinite(mu5) else np.nan,
        "Volatilidad 5Y": round(sig5 * 100, 2) if np.isfinite(sig5) else np.nan,
        "Sharpe": round(sharpe5, 2) if np.isfinite(sharpe5) else np.nan,
        "Max Drawdown 1Y": round((mdd1y or np.nan) * 100, 2) if mdd1y is not None else np.nan,
        "Ticker base": tkr, "Codes": pair_codes, "inv_flag": int(inverse),
        "key": f"{tkr}|inv={int(inverse)}"
    }

def _build_table(basket: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for it in basket:
        inverse = bool(it.get("inverse", False))
        label = str(it.get("name", ""))    # puede venir legacy
        pair_codes = extract_pair_codes(label)  # asegura 'AAA/BBB'
        display_name = pretty_fx_label(pair_codes)
        rows.append(_fx_metrics_row(it["ticker"], display_name, pair_codes, inverse=inverse))
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def _render_fx_detail(tkr: str, label_bonito: str, *, inverse: bool):
    code_pair = extract_pair_codes(label_bonito) or "—"
    r5, df5 = _series_from_yahoo(tkr, period="5y", inverse=inverse)
    _, df1 = _series_from_yahoo(tkr, period="1y", inverse=inverse)
    price_val = _last_price(tkr, inverse=inverse)

    ytd = np.nan
    if df1 is not None and "Adj Close" in df1.columns:
        ytd = _ytd_return_from_price(df1["Adj Close"])

    ret_1y = _ret_1y(tkr, inverse=inverse)
    mu5 = _avg_annual_return(r5) if r5 is not None and not r5.empty else np.nan
    sig5 = _annual_vol(r5) if r5 is not None and not r5.empty else np.nan
    mdd1y = _max_drawdown_1y(tkr, inverse=inverse)

    st.markdown(f"### {pretty_fx_label(code_pair)}")
    cols = st.columns(7)
    cols[0].metric("Precio", "—" if (price_val is None or not np.isfinite(price_val)) else f"{price_val:,.6f}")
    cols[1].metric("YTM", _fmt_pct_eq(ytd))
    cols[2].metric("Rendimiento 1Y", _fmt_pct_eq(ret_1y))
    cols[3].metric("Rendimiento 5Y", _fmt_pct_eq(mu5))
    cols[4].metric("Volatilidad", _fmt_pct_eq(sig5))
    cols[5].metric("Máximo Draw Down", _fmt_pct_eq(mdd1y))
    cols[6].metric("Moneda", code_pair)

    try:
        px3 = get_hist_sliced_years(tkr, 3.0)
    except Exception:
        px3 = None

    df_plot = None
    if px3 is not None and not px3.empty and {"Open","High","Low","Close"}.issubset(px3.columns):
        df_ohlc = px3[["Open","High","Low","Close"]].copy()
        if inverse:
            df_ohlc = _invert_ohlc_df(df_ohlc)
        df_plot = _resample_ohlc(df_ohlc, "W-FRI")
    if df_plot is not None and not df_plot.empty:
        fig = go.Figure(
            data=[go.Candlestick(
                x=df_plot.index, open=df_plot["Open"], high=df_plot["High"],
                low=df_plot["Low"], close=df_plot["Close"],
                increasing_line_color="green", decreasing_line_color="red", name="Precio"
            )]
        )
        fig.update_layout(xaxis_rangeslider_visible=False, height=420, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True, key=f"fx_candle_{tkr}_{int(inverse)}")
    else:
        st.info("No hay datos suficientes para graficar velas semanales de 3 años.")

# ==========================
# 5. RENDER PRINCIPAL
# ==========================
def render():

    st.caption("Buscador de Divisas")
    
    # ============================================================================
    # 🆕 INTEGRACIÓN CON LEDGER: Agregar activos de FX del ledger
    # ============================================================================
    ss = st.session_state
    ledger_fx = _get_fx_from_ledger(ss)
    
    # Catálogo desde CSV
    try:
        options = _catalog_from_csv()
    except FileNotFoundError:
        st.error(f"No se pudo encontrar '{_FX_FILE_NAME}' en la carpeta 'data/'. Asegúrate de que el archivo exista en la raíz de la app.")
        _ensure_fx_selection()
        return
    except Exception as e:
        st.error(f"Error al cargar/procesar el catálogo de divisas: {e}")
        _ensure_fx_selection()
        return

    if not options:
        st.warning(f"No se pudieron cargar pares de divisas. El archivo '{_FX_FILE_NAME}' está vacío o incompleto.")
        _ensure_fx_selection()
        return

    # --- Mapeos y estado ---
    def _label_for(opt: Dict[str, object]) -> str:
        codes = str(opt.get("name", "")).strip().upper()  # ej. 'BRL/AUD'
        return pretty_fx_label(codes)

    def _key_for(opt: Dict[str, object]) -> str:
        return f"{opt['ticker']}|inv={int(bool(opt.get('inverse', False)))}"

    keys = [_key_for(o) for o in options]
    codes_list = [o["name"] for o in options]
    key2opt = {_key_for(o): o for o in options}
    key2codes = dict(zip(keys, codes_list))
    codes2key = dict(zip(codes_list, keys))

    ss.setdefault("fx_basket", [])
    ss.setdefault("fx_picks", [])
    ss.setdefault("fx_picks_raw", [])
    ss.setdefault("_fx_synced", False)
    ss.setdefault("fx_detail_key", None)

    # ============================================================================
    # 🆕 Agregar activos del ledger a fx_basket si no están ya
    # ============================================================================
    added_pairs = []
    for pair_code, meta in ledger_fx.items():
        # Buscar si ya existe en basket
        ticker = meta["ticker"]
        inverse = meta["inverse"]
        key = f"{ticker}|inv={int(inverse)}"
        
        # Agregar el par a codes_list si no está (para que sea opción válida en multiselect)
        if pair_code not in codes_list:
            codes_list.append(pair_code)
            codes2key[pair_code] = key
            key2codes[key] = pair_code
            # Agregar a key2opt para que funcione el sistema de mapeo
            if key not in key2opt:
                key2opt[key] = {
                    "ticker": ticker,
                    "name": pair_code,
                    "inverse": inverse,
                    "rank": 9999  # Al final de la lista
                }
            added_pairs.append(pair_code)
        
        # Si no está en basket, agregarlo
        if not any(it.get("key") == key for it in ss.fx_basket):
            ss.fx_basket.append({
                "key": key,
                "ticker": ticker,
                "name": pretty_fx_label(pair_code),
                "inverse": inverse,
            })
        
        # Agregar también a picks si no está (para el multiselect)
        if pair_code not in ss.fx_picks:
            ss.fx_picks.append(pair_code)
        
        # Agregar a picks_raw para que aparezca en el multiselect inmediatamente
        if pair_code not in ss.fx_picks_raw:
            ss.fx_picks_raw.append(pair_code)
    
    # ============================================================================
    # FIN DE INTEGRACIÓN CON LEDGER
    # ============================================================================

    # --- One-shot sync (restauración desde persistencia) ---
    if not ss._fx_synced:
        # Migración: si en basket hay 'AAA/BBB' plano, convierte a bonito definitivo
        if ss.fx_basket:
            migrated = False
            for it in ss.fx_basket:
                nm = it.get("name", "")
                if _is_code_pair(nm):
                    it["name"] = pretty_fx_label(nm); migrated = True
            if migrated:
                pass

        # Restaurar basket desde picks (persistidos)
        if not ss.fx_basket and ss.fx_picks:
            ss.fx_basket = []
            for code_label in ss.fx_picks:
                match = re.search(r"\(([A-Z]{3}/[A-Z]{3})\)$", code_label)
                code_simple = match.group(1) if match else code_label
                k = codes2key.get(code_simple)
                if k:
                    opt = key2opt.get(k)
                    if opt:
                        ss.fx_basket.append({
                            "key": k,
                            "ticker": opt["ticker"],
                            "name": pretty_fx_label(code_simple),
                            "inverse": bool(opt["inverse"]),
                        })
        
        # Restaurar picks desde basket (MEJORADO: extend en vez de replace)
        if ss.fx_basket:
            # Encontrar los que faltan en fx_picks
            new_codes = []
            for it in ss.fx_basket:
                k = it.get("key")
                codes = key2codes.get(k)
                if not codes:
                    nm = it.get("name", "")
                    m = re.search(r"\(([A-Z]{3}/[A-Z]{3})\)\s*$", nm)
                    codes = m.group(1) if m else None
                if codes and codes in codes_list and codes not in ss.fx_picks:
                    new_codes.append(codes)
            
            # Agregar los nuevos (no reemplazar)
            if new_codes:
                ss.fx_picks.extend(new_codes)
                ss.fx_picks_raw = ss.fx_picks.copy()

        ss._fx_synced = True

    # --- Normaliza picks a "AAA/BBB" SIN encoger por renders intermedios ---
    fx_prev = list(ss.fx_picks or [])
    new_codes: List[str] = []
    for v in fx_prev:
        if _is_code_pair(v):
            if (v in codes_list) and (v not in new_codes):
                new_codes.append(v)
            else:
                if v not in new_codes:
                    new_codes.append(v)
        else:
            m = re.search(r"\(([A-Z]{3}/[A-Z]{3})\)\s*$", str(v))
            c = m.group(1) if m else None
            if c and (c in codes_list):
                if c not in new_codes:
                    new_codes.append(c)
            else:
                if v not in new_codes:
                    new_codes.append(v)
    prev_set = set(fx_prev)
    new_set  = set(new_codes)
    if prev_set.issubset(new_set):
        ss.fx_picks = new_codes

    # --- Searchbox y Multiselect ---
    disp2key = {}
    for k in keys:
        codes = key2codes[k]
        try:
            a, b = codes.split("/", 1)
        except ValueError:
            a, b = codes, ""
        pretty_left = f"{_fullname(a)}/{_fullname(b)}"
        display = f"{pretty_left} — {codes}"
        disp2key[display] = k

    def _fx_search_fn(q: str):
        q = (q or "").strip().lower()
        pool = list(disp2key.keys())
        if q:
            pool = [d for d in pool if q in d.lower()]
        return [(d, disp2key[d]) for d in pool[:10]]

    sel_key_from_search = st_searchbox(
        placeholder="Busca una divisa… (ej. MXN/USD o Euro/Dólar)",
        key="fx_add_searchbox",
        search_function=_fx_search_fn
    )

    # Alta desde searchbox (sin lag)
    if sel_key_from_search:
        if all(it["key"] != sel_key_from_search for it in ss.fx_basket):
            opt = key2opt.get(sel_key_from_search)
            if opt:
                ss.fx_basket.append({
                    "key": sel_key_from_search,
                    "ticker": opt["ticker"],
                    "name": _label_for(opt),  # ya viene bonito
                    "inverse": bool(opt["inverse"]),
                })
                code = key2codes.get(sel_key_from_search)
                if code and code not in ss.fx_picks:
                    ss.fx_picks.append(code)
                if code and code not in ss.fx_picks_raw:
                    ss.fx_picks_raw.append(code)
                ss.fx_detail_key = sel_key_from_search
                ss["fx_add_search"] = ""
                st.rerun()

    # Asegura que el widget arranque con lo que hay en fx_picks
    # FORZAR actualización si hay nuevos del ledger
    if ledger_fx:
        ss.fx_picks_raw = list(ss.fx_picks)
        # CRÍTICO: Forzar que el widget use el nuevo valor
        ss.fx_picks_raw_widget = list(ss.fx_picks)
    elif not ss.fx_picks_raw:
        ss.fx_picks_raw = list(ss.fx_picks)
    
    # Sincronizar widget key con fx_picks_raw si hay desincronización
    if ss.get("fx_picks_raw_widget") != ss.fx_picks_raw:
        ss.fx_picks_raw_widget = list(ss.fx_picks_raw)

    # Multiselect controlado (callback = fuente de verdad)
    selected = st.multiselect(
        "Divisas Seleccionadas",
        options=codes_list,
        default=ss.fx_picks_raw,
        key="fx_picks_raw_widget",
        on_change=_handle_fx_multiselect_change,
        args=(codes2key, key2opt),
    )
    
    # Actualizar fx_picks_raw con lo seleccionado
    if selected != ss.fx_picks_raw:
        ss.fx_picks_raw = selected

    # Bootstrap: si hay picks pero basket vacío
    if ss.fx_picks and not ss.fx_basket:
        try:
            _init_fx_state_from_picks()
        except Exception:
            pass

    basket = ss.fx_basket
    if not basket:
        _ensure_fx_selection()
        st.info("No has agregado ninguna divisa. Usa el buscador para añadir.")
        return

    # 📦 Precarga en lote 5y/1d para todos los tickers de FX (cache 24h + pool de sesión)
    try:
        tickers_to_load = [str(it.get("ticker","")).strip() for it in basket if str(it.get("ticker","")).strip()]
        preload_hist_5y_daily(tickers_to_load)
    except Exception:
        pass  # fallback: cada _series_from_yahoo hará lectura/precarga por ticker si faltara

    # Garantiza que fx_selection esté listo para el optimizador
    _ensure_fx_selection()

    # --- Render Tabla y Detalle ---
    df = _build_table(basket)

    df_full = df.copy()
    visible_cols = [
        "Divisa", "Precio actual", "Rendimiento 1Y", "Rendimiento 5Y",
        "Volatilidad 5Y", "Sharpe", "Max Drawdown 1Y"
    ]
    df_view = df_full[visible_cols].copy()

    def _fmt_pct_str(v, d=2):
        try:
            if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
                return "—"
            return f"{float(v):.{d}f}%"
        except Exception:
            return "—"

    def _fmt_num_str(v, d=6):
        try:
            if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
                return "—"
            return f"{float(v):.{d}f}"
        except Exception:
            return "—"

    pct_cols = ["Rendimiento 1Y", "Rendimiento 5Y", "Volatilidad 5Y", "Max Drawdown 1Y"]
    for c in pct_cols:
        if c in df_view.columns:
            df_view[c] = df_view[c].apply(_fmt_pct_str)
    if "Precio actual" in df_view.columns:
        df_view["Precio actual"] = df_view["Precio actual"].apply(_fmt_num_str)
    if "Sharpe" in df_view.columns:
        df_view["Sharpe"] = df_view["Sharpe"].apply(lambda v: "—" if pd.isna(v) else f"{float(v):.2f}")

    if not ss.get("fx_detail_key") and not df_full.empty:
        ss["fx_detail_key"] = str(df_full.iloc[0]["key"])

    if not df_full.empty:
        st.caption(f"{len(df_full)} Divisas Seleccionadas")

    event = st.dataframe(
        df_view,
        key="fx_table_view",
        width="stretch",
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun"
    )
    sel_rows = event.get("selection", {}).get("rows", []) if isinstance(event, dict) else getattr(getattr(event, "selection", None), "rows", []) or []

    if sel_rows:
        i = sel_rows[0]
        try:
            new_key = str(df_full.iloc[i]["key"])
        except Exception:
            new_key = None
        if new_key and new_key != ss.get("fx_detail_key"):
            ss["fx_detail_key"] = new_key

    detail_key = ss.get("fx_detail_key")
    if detail_key:
        item = next((it for it in ss.fx_basket if it["key"] == detail_key), None)
        if item:
            _render_fx_detail(
                tkr=item["ticker"],
                label_bonito=item["name"],
                inverse=bool(item.get("inverse", False))
            )