# tabs/banxico_client.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import math

import pandas as pd
import streamlit as st
import requests

# =========================
# Constantes Banxico
# =========================
BANXICO_TOKEN_DEFAULT = "6da7d3340a11467efc40a85449a0f5260f46260082044b96e8c7637cf8c1e423"
BANXICO_BASE = "https://www.banxico.org.mx/SieAPIRest/service/v1"

# Series relevantes
INPC_SERIE = "SP30577"      # INPC variación mensual (%)
EXP_T  = "SR14139"
EXP_T1 = "SR14146"
EXP_T2 = "SR14153"
EXP_T3 = "SR14160"

TIIE_28D_DAILY = "SF43783"
FONDEO_Q = [
    "SR14659","SR14666","SR14673","SR14680","SR14687",
    "SR14694","SR14701","SR14708","SR14715","SR16841"
]

# =========================
# Helpers básicos
# =========================
def _parse_fecha_mx(fecha: str) -> Optional[datetime]:
    for fmt in ("%d/%m/%Y","%Y-%m-%d"):
        try:
            return datetime.strptime(fecha, fmt)
        except Exception:
            pass
    return None

def _parse_iso(d: str) -> datetime:
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(d, fmt)
        except Exception:
            pass
    return datetime.min

def _safe_json(resp: requests.Response):
    ctype = resp.headers.get("Content-Type","")
    if "application/json" not in ctype:
        preview = resp.text[:200].replace("\n"," ").strip()
        raise RuntimeError(f"Banxico no devolvió JSON. HTTP={resp.status_code} CT={ctype}. {preview}")
    return resp.json()

def _check_dates_str(start: str, end: str) -> Tuple[str, str, datetime, datetime]:
    def parse(d: str) -> datetime: return datetime.strptime(d, "%Y-%m-%d")
    s, e = parse(start), parse(end)
    if e < s: s, e = e, s
    return s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"), s, e

def _month_end(dt: datetime) -> datetime:
    p = pd.Period(dt.date(), freq="M")
    return p.end_time.replace(hour=0, minute=0, second=0, microsecond=0)

def _closest_month_end(dt: datetime) -> datetime:
    return _month_end(dt)

# =========================
# Llamadas Banxico (con caché)
# =========================
@st.cache_data(show_spinner=False, ttl=60*60*6)  # 6 horas
def fetch_range(series_id: str, start: str, end: str, token: str) -> List[Dict[str, str]]:
    """Envuelve la llamada de rango a Banxico y cachea el resultado."""
    url = f"{BANXICO_BASE}/series/{series_id}/datos/{start}/{end}"
    headers = {"Bmx-Token": token or BANXICO_TOKEN_DEFAULT}
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} {series_id}: {r.text[:160]}")
    data = _safe_json(r)
    series = data.get("bmx",{}).get("series",[])
    puntos = series[0].get("datos", []) if series else []
    puntos = [d for d in puntos if d.get("dato") not in (None, "", "N/E")]
    return puntos

@st.cache_data(show_spinner=False, ttl=60*60*6)
def latest_on_or_before(series_id: str, end_dt: datetime, token: str
) -> Tuple[Optional[str], Optional[float], Optional[int]]:
    """Último dato <= end_dt con degradado de ventanas; cacheado."""
    lookbacks = [30, 180, 365, 365*5, 365*10]
    for days in lookbacks:
        start_dt = end_dt - timedelta(days=days)
        s, e, *_ = _check_dates_str(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        try:
            puntos = fetch_range(series_id, s, e, token)
            if puntos:
                puntos = sorted(puntos, key=lambda d: _parse_fecha_mx(d["fecha"]) or datetime.min)
                f = puntos[-1]["fecha"]; v = float(puntos[-1]["dato"])
                f_dt = _parse_fecha_mx(f)
                dd = (end_dt - f_dt).days if f_dt else None
                return f, v, dd
        except Exception:
            continue
    return None, None, None

# =========================
# Indicadores de inflación
# =========================
@st.cache_data(show_spinner=False, ttl=60*60*6)
def get_inflation_12m(
    end_date: Optional[datetime.date] = None, 
    series_id: str = INPC_SERIE
) -> Tuple[Optional[float], Optional[str]]:
    """
    Retorna inflación compuesta 12m y label de rango; cacheado.
    MODIFICADO: Acepta un 'end_date' para cálculos históricos.
    """
    try:
        token = st.session_state.get("banxico_token") or BANXICO_TOKEN_DEFAULT
    except Exception:
        token = BANXICO_TOKEN_DEFAULT

    # --- LÓGICA MODIFICADA PARA USAR end_date ---
    if end_date is None:
        end_dt = datetime.now()
    else:
        # Aseguramos que sea un objeto datetime para la lógica siguiente
        end_dt = datetime.combine(end_date, datetime.min.time())
    
    # Buscamos en una ventana de 450 días hacia atrás desde la fecha de fin
    start_dt = end_dt - timedelta(days=450)
    # --- FIN DE LA MODIFICACIÓN ---

    s, e, *_ = _check_dates_str(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    
    puntos = fetch_range(series_id, s, e, token)
    if not puntos: return None, None
    
    puntos = sorted(puntos, key=lambda d: _parse_iso(d.get("fecha", "")))
    
    # Filtramos puntos futuros si end_date fue especificada
    puntos = [p for p in puntos if _parse_iso(p.get("fecha", "")) <= end_dt]

    vals = [float(p["dato"]) for p in puntos if p.get("dato") not in (None, "", "N/E")]
    
    if len(vals) < 12: return None, None
    
    comp = 1.0
    for v in vals[-12:]:
        comp *= (1.0 + v/100.0)
    
    infl_12m = comp - 1.0
    label = f"INPC (variación compuesta) {puntos[-12]['fecha']}→{puntos[-1]['fecha']}"
    
    return infl_12m, label

@st.cache_data(show_spinner=False, ttl=60*60*6)
def realized_inflation_factor(liq_dt: datetime, val_dt: datetime, token: str) -> float:
    """Factor de inflación realizado (compuesto) entre fin de mes de liq→val; cacheado."""
    s, e, *_ = _check_dates_str((liq_dt - timedelta(days=450)).strftime("%Y-%m-%d"),
                                (val_dt + timedelta(days=60)).strftime("%Y-%m-%d"))
    puntos = fetch_range(INPC_SERIE, s, e, token or BANXICO_TOKEN_DEFAULT)
    if not puntos:
        return 1.0
    df = (pd.DataFrame([( _parse_iso(p["fecha"]), float(p["dato"]) ) for p in puntos if p.get("dato") not in (None,"","N/E")],
                       columns=["fecha","pct"])
            .dropna().sort_values("fecha").set_index("fecha"))
    df.index = df.index.map(_month_end)
    liq_me = df.index[df.index.get_indexer([_closest_month_end(liq_dt)], method="nearest")[0]]
    val_me = df.index[df.index.get_indexer([_closest_month_end(val_dt)], method="nearest")[0]]
    window = df.loc[(df.index > liq_me) & (df.index <= val_me), "pct"]
    if window.empty: return 1.0
    comp = 1.0
    for v in window.values:
        comp *= (1.0 + float(v)/100.0)
    return max(1.0, comp)

@st.cache_data(show_spinner=False, ttl=60*60*6)
def inflation_expectations_path(years_needed: float, token: str
) -> Tuple[float, Dict[str, Optional[float]]]:
    """
    Promedio anual equivalente de inflación esperada usando EXP_T..T+3.
    Devuelve (pi_avg_decimal, breakdown_en_%).
    """
    if years_needed <= 0:
        return 0.0, {"t": None, "t+1": None, "t+2": None, "t+3": None}

    def _latest(sid: str) -> Optional[float]:
        end_dt = datetime.now()
        s, e, *_ = _check_dates_str((end_dt - timedelta(days=240)).strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        pts = fetch_range(sid, s, e, token or BANXICO_TOKEN_DEFAULT)
        if not pts: return None
        pts = sorted(pts, key=lambda d: _parse_iso(d.get("fecha","")))
        return float(pts[-1]["dato"])

    t  = _latest(EXP_T)
    t1 = _latest(EXP_T1)
    t2 = _latest(EXP_T2)
    t3 = _latest(EXP_T3)

    t_d  = t/100.0  if t  is not None else None
    t1_d = t1/100.0 if t1 is not None else None
    t2_d = t2/100.0 if t2 is not None else None
    t3_d = t3/100.0 if t3 is not None else None

    pool = [x for x in [t_d, t1_d, t2_d, t3_d] if x is not None]
    if not pool: return 0.0, {"t": t, "t+1": t1, "t+2": t2, "t+3": t3}
    avg_pool = sum(pool)/len(pool)

    n_full = int(math.floor(years_needed))
    frac = years_needed - n_full
    path: List[float] = []
    if n_full >= 1: path.append(t_d  if t_d  is not None else avg_pool)
    if n_full >= 2: path.append(t1_d if t1_d is not None else avg_pool)
    if n_full >= 3: path.append(t2_d if t2_d is not None else avg_pool)
    if n_full >= 4: path.append(t3_d if t3_d is not None else avg_pool)
    if n_full > 4:  path.extend([avg_pool]*(n_full-4))
    if frac > 1e-9:
        next_year_pi = (
            t_d if n_full == 0 and t_d is not None else
            t1_d if n_full == 1 and t1_d is not None else
            t2_d if n_full == 2 and t2_d is not None else
            t3_d if n_full == 3 and t3_d is not None else
            avg_pool
        )
        path.append(next_year_pi * frac)

    comp = 1.0; years_eff = 0.0
    for i, p in enumerate(path):
        comp *= (1.0 + p)
        years_eff += 1.0 if (i < len(path)-1 or frac < 1e-9) else frac
    total_pi = comp - 1.0
    pi_avg = ((1.0 + total_pi) ** (1.0/years_eff) - 1.0) if years_eff > 1e-9 else 0.0
    return pi_avg, {"t": t, "t+1": t1, "t+2": t2, "t+3": t3}

# =========================
# TIIE / Fondeo (BONDES)
# =========================
@st.cache_data(show_spinner=False, ttl=60*60*6)
def tiie_realized_avg(liq: datetime, val: datetime, token: str) -> float:
    """Promedio aritmético anual de TIIE 28d diaria SF43783 entre liq→val (incluye extremos)."""
    if val <= liq: return 0.0
    s, e, *_ = _check_dates_str(liq.strftime("%Y-%m-%d"), val.strftime("%Y-%m-%d"))
    pts = fetch_range(TIIE_28D_DAILY, s, e, token or BANXICO_TOKEN_DEFAULT)
    if not pts: return 0.0
    vals = [float(p["dato"]) for p in pts if p.get("dato") not in (None, "", "N/E")]
    if not vals: return 0.0
    return sum(vals)/len(vals)

@st.cache_data(show_spinner=False, ttl=60*60*6)
def fondeo_expectations_quarter_vector(token: str) -> List[Optional[float]]:
    """Lista [t, t+1, ..., t+9] de fondeo trimestral en % anual; cacheada."""
    out: List[Optional[float]] = []
    for sid in FONDEO_Q:
        end_dt = datetime.now()
        s, e, *_ = _check_dates_str((end_dt - timedelta(days=240)).strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        pts = fetch_range(sid, s, e, token or BANXICO_TOKEN_DEFAULT)
        if not pts:
            out.append(None); continue
        pts = sorted(pts, key=lambda d: _parse_iso(d.get("fecha","")))
        out.append(float(pts[-1]["dato"]))
    return out

@st.cache_data(show_spinner=False, ttl=60*60*6)
def tiie_expected_avg(years_needed: float, token: str) -> float:
    """Promedio anual esperado de TIIE usando encuestas trimestrales; cacheado."""
    if years_needed <= 0: return 0.0
    q_needed = max(1, int(round(years_needed * 4)))
    vec = fondeo_expectations_quarter_vector(token)  # % anual
    clean = [v for v in vec if v is not None]
    if not clean:
        return 0.0
    avg_all = sum(clean)/len(clean)
    seq: List[float] = []
    for i in range(q_needed):
        v = vec[i] if i < len(vec) and vec[i] is not None else avg_all
        seq.append(float(v))
    return sum(seq)/len(seq)

# =========================
# BONDES: spread por precio de colocación
# =========================
@st.cache_data(show_spinner=False, ttl=60*60*6)
def bondes_spread_auto_by_sid(series_id: str, liq_dt: datetime, token: str) -> Tuple[float, str]:
    """
    Spread BONDES ≈ 100 - precio ponderado de colocación.
    Recibe directamente el series_id del BONDES; cacheado.
    """
    if not series_id:
        return 0.0, ""
    f_px, v_px, _ = latest_on_or_before(series_id, liq_dt, token or BANXICO_TOKEN_DEFAULT)
    if v_px is None:
        return 0.0, "Precio (sin dato)"
    try:
        price = float(v_px)
    except Exception:
        return 0.0, f"Precio (dato inválido) Banxico {f_px}"
    spread_pp = 100.0 - price
    return float(spread_pp), f"Precio ponderado de colocación {price:.4f} — Banxico {f_px}"
