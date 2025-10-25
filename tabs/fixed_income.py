# tabs/fixed_income.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from datetime import date, datetime, timedelta
import calendar

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# yfinance para benchmark automático (EE.UU.)
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

# ===== Banxico client (todas las llamadas cacheadas) =====
from tabs.banxico_client import (
    BANXICO_TOKEN_DEFAULT,
    latest_on_or_before as banxico_latest_on_or_before,
    inflation_expectations_path as bmx_inflation_expectations_path,
    realized_inflation_factor as bmx_realized_inflation_factor,
    tiie_realized_avg as bmx_tiie_realized_avg,
    tiie_expected_avg as bmx_tiie_expected_avg,
    bondes_spread_auto_by_sid,
)

# =========================
# Utilidades generales
# =========================
def _today() -> date:
    return datetime.now().date()

FREQ_MAP = {
    "Cero cupón": 0,
    "Mensual": 12,
    "Trimestral": 4,
    "Semestral": 2,
    "Anual": 1,
}

BASES = ["ACT/360", "30/360"]
PAISES = ["México", "EE.UU."]

TIPOS_BONO = [
    "Bono gubernamental",
    "Bono corporativo",
    "Bono temático (verde/social/sustentable)",
]

MX_RATINGS = [
    "mxAAA","mxAA+","mxAA","mxAA-","mxA+","mxA","mxA-",
    "mxBBB+","mxBBB","mxBBB-","mxBB+","mxBB","mxBB-",
    "mxB+","mxB","mxB-","mxCCC","mxCC","mxC","D"
]
US_RATINGS = [
    "AAA","AA+","AA","AA-","A+","A","A-",
    "BBB+","BBB","BBB-","BB+","BB","BB-",
    "B+","B","B-","CCC","CC","C","D"
]

# =========================
# Day count & fechas
# =========================
def _days_30_360(d1: date, d2: date) -> int:
    d1_day = min(d1.day, 30)
    d2_day = d2.day
    if d1_day == 30:
        d2_day = min(d2_day, 30)
    return (d2.year - d1.year) * 360 + (d2.month - d1.month) * 30 + (d2_day - d1_day)

def _dcf(d1: date, d2: date, base: str) -> float:
    if d2 <= d1:
        return 0.0
    if base.upper() == "30/360":
        return _days_30_360(d1, d2) / 360.0
    return (d2 - d1).days / 360.0

# ✅ NO pandas aquí: evita Timestamps en toda la cadena
def _add_months(dt: date, months: int) -> date:
    y = dt.year + (dt.month - 1 + months) // 12
    m = (dt.month - 1 + months) % 12 + 1
    last_day = calendar.monthrange(y, m)[1]
    d = min(dt.day, last_day)
    return date(y, m, d)

def _add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(month=2, day=28, year=d.year + years)

# ======= Sugerencia de referencia MX por tenor (robusta) =======
def _auto_pick_mx_ref_name(liq: date, mat: date) -> str:
    def _to_date(x):
        if x is None: return None
        if isinstance(x, pd.Timestamp): return x.date()
        if isinstance(x, datetime):     return x.date()
        return x
    liq = _to_date(liq); mat = _to_date(mat)
    if not isinstance(liq, date) or not isinstance(mat, date):
        return "MBono 5y"
    if mat < liq: liq, mat = mat, liq
    tenor_days = max(1, int((mat - liq).days))
    if tenor_days <= 45:  return "CETES 28d"
    if tenor_days <= 120: return "CETES 91d"
    if tenor_days <= 210: return "CETES 182d"
    if tenor_days <= 540: return "CETES 364d"
    if tenor_days <= 820: return "CETES 728d"
    years = tenor_days / 365.25
    if years <= 3:  return "MBono 3y"
    if years <= 5:  return "MBono 5y"
    if years <= 10: return "MBono 10y"
    if years <= 20: return "MBono 20y"
    return "MBono 30y"

# ======= Cupones anclados a LIQUIDACIÓN =======
def _prev_next_from_anchor(
    anchor: date,
    maturity: date,
    valuation: date,
    freq_per_year: int,
) -> Tuple[Optional[date], date, float, bool]:
    # Convierte todo a date puro
    def _ensure_date(d):
        if isinstance(d, datetime):
            return d.date()
        return d
    
    anchor = _ensure_date(anchor)
    maturity = _ensure_date(maturity)
    valuation = _ensure_date(valuation)
    
    if freq_per_year <= 0:
        return None, maturity, max(0.0, (maturity - anchor).days) / 365.0, True
    months = 12 // freq_per_year
    first_cp = _ensure_date(_add_months(anchor, months))
    if valuation < first_cp:
        nxt = first_cp if first_cp <= maturity else maturity
        return None, nxt, 1.0 / freq_per_year, True
    prev_cp: Optional[date] = None
    d_iter = first_cp
    nxt = first_cp
    while d_iter <= maturity and d_iter <= valuation:
        prev_cp = d_iter
        d_iter = _ensure_date(_add_months(d_iter, months))
        nxt = d_iter
    if nxt > maturity:
        nxt = maturity
    return prev_cp, nxt, 1.0 / freq_per_year, False

# =========================
# Pricing (Street)
# =========================
def price_bond_dirty_clean(
    valuation: date,
    maturity: date,
    coupon_annual_pct: float,
    ytm_annual_pct: float,
    freq_per_year: int,
    nominal: float,
    base: str,
    anchor: Optional[date] = None,
) -> Tuple[float, float, float, Optional[date], date, float, float]:
    coupon_annual = coupon_annual_pct / 100.0
    ytm_annual = ytm_annual_pct / 100.0

    if freq_per_year == 0:
        T_years = max(0.0, (maturity - valuation).days) / 365.0
        dirty = nominal / ((1.0 + ytm_annual) ** T_years)
        accrued = 0.0
        clean = dirty
        return dirty, clean, accrued, None, maturity, 0.0, T_years * 360.0

    if anchor is None:
        anchor = valuation

    last_cp, next_cp, _, no_coupon_yet = _prev_next_from_anchor(anchor, maturity, valuation, freq_per_year)

    if no_coupon_yet or last_cp is None:
        a = 0.0; w = 1.0; accrued = 0.0
        days_accrued = 0.0
        days_period = _days_30_360(anchor, next_cp) if base.upper() == "30/360" else (next_cp - anchor).days
    else:
        denom = _dcf(last_cp, next_cp, base)
        numer = _dcf(last_cp, valuation, base)
        a = 0.0 if denom == 0 else max(0.0, min(1.0, numer / denom))
        w = 1.0 - a
        c = nominal * (coupon_annual / freq_per_year)
        accrued = c * a
        days_accrued = _days_30_360(last_cp, valuation) if base.upper() == "30/360" else (valuation - last_cp).days
        days_period = _days_30_360(last_cp, next_cp) if base.upper() == "30/360" else (next_cp - last_cp).days

    months = 12 // freq_per_year
    N = 0
    d_iter = next_cp
    while d_iter <= maturity:
        N += 1
        d_iter = _add_months(d_iter, months)

    y_p = ytm_annual / freq_per_year
    c = nominal * (coupon_annual / freq_per_year)

    if abs(y_p) < 1e-12:
        pv_leg = c * N
        pv_prin = nominal
        dirty = (pv_leg + pv_prin) * (1.0 + y_p) ** (-w)
    else:
        pv_leg = c * (1.0 - (1.0 + y_p) ** (-N)) / y_p
        pv_prin = nominal * (1.0 + y_p) ** (-N)
        dirty = (pv_leg + pv_prin) * (1.0 + y_p) ** (-w)

    clean = dirty - accrued
    return float(dirty), float(clean), float(accrued), last_cp, next_cp, float(days_accrued), float(days_period)

# =========================
# Benchmarks EE.UU. (automático)
# =========================
US_BENCH_CHOICES = ["T-Bill 3M (IRX)", "UST 5Y (FVX)", "UST 10Y (TNX)", "UST 30Y (TYX)"]

def _fetch_us_benchmark(choice: str) -> Optional[float]:
    if not YF_OK: return None
    symbol, divisor = None, 1.0
    if "IRX" in choice: symbol="^IRX"; divisor=1.0
    elif "FVX" in choice: symbol="^FVX"; divisor=10.0
    elif "TNX" in choice: symbol="^TNX"; divisor=10.0
    elif "TYX" in choice: symbol="^TYX"; divisor=10.0
    else: return None
    try:
        df = yf.download(symbol, period="5d", interval="1d", progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            last = float(df["Close"].dropna().iloc[-1])
            return last / divisor
    except Exception:
        return None
    return None

# =========================
# Instrumentos gubernamentales (UI)
# =========================
GOB_INSTRUMENTOS = {
    # CETES
    "CETES 28d":  {"tipo": "CETES",  "dias": 28,  "serie": "SF43936", "nominal": 10.0},
    "CETES 91d":  {"tipo": "CETES",  "dias": 91,  "serie": "SF43939", "nominal": 10.0},
    "CETES 182d": {"tipo": "CETES",  "dias": 182, "serie": "SF43942", "nominal": 10.0},
    "CETES 364d": {"tipo": "CETES",  "dias": 364, "serie": "SF43945", "nominal": 10.0},
    "CETES 728d": {"tipo": "CETES",  "dias": 728, "serie": "SF349785", "nominal": 10.0},
    # MBONOS
    "MBono 3y":   {"tipo": "MBONO",  "anios": 3,  "serie": "SF43883", "nominal": 100.0},
    "MBono 5y":   {"tipo": "MBONO",  "anios": 5,  "serie": "SF43886", "nominal": 100.0},
    "MBono 10y":  {"tipo": "MBONO",  "anios": 10, "serie": "SF44071", "nominal": 100.0},
    "MBono 20y":  {"tipo": "MBONO",  "anios": 20, "serie": "SF45384", "nominal": 100.0},
    "MBono 30y":  {"tipo": "MBONO",  "anios": 30, "serie": "SF60696", "nominal": 100.0},
    # UDIBONOS (5y REMOVIDO)
    "Udibono 3y":  {"tipo": "UDIBONO", "anios": 3,  "serie": "SF61592", "nominal": 100.0},
    "Udibono 10y": {"tipo": "UDIBONO", "anios": 10, "serie": "SF43924", "nominal": 100.0},
    "Udibono 20y": {"tipo": "UDIBONO", "anios": 20, "serie": "SF46958", "nominal": 100.0},
    "Udibono 30y": {"tipo": "UDIBONO", "anios": 30, "serie": "SF46961", "nominal": 100.0},
    # BONDES D (agregados)
    "Bondes D 1y": {"tipo": "BONDESD", "anios": 1, "serie": "SF60650",  "nominal": 100.0},
    "Bondes D 2y": {"tipo": "BONDESD", "anios": 2, "serie": "SF339745","nominal": 100.0},
    "Bondes D 3y": {"tipo": "BONDESD", "anios": 3, "serie": "SF60651",  "nominal": 100.0},
    "Bondes D 5y": {"tipo": "BONDESD", "anios": 5, "serie": "SF60652",  "nominal": 100.0},
    # BONDES F (solo 1,2,3,5,7,10)
    "Bondes F 1y":  {"tipo": "BONDESF", "anios": 1,  "serie": "SF341526","nominal": 100.0},
    "Bondes F 2y":  {"tipo": "BONDESF", "anios": 2,  "serie": "SF341527","nominal": 100.0},
    "Bondes F 3y":  {"tipo": "BONDESF", "anios": 3,  "serie": "SF341528","nominal": 100.0},
    "Bondes F 5y":  {"tipo": "BONDESF", "anios": 5,  "serie": "SF341529","nominal": 100.0},
    "Bondes F 7y":  {"tipo": "BONDESF", "anios": 7,  "serie": "SF345527","nominal": 100.0},
    "Bondes F 10y": {"tipo": "BONDESF", "anios": 10, "serie": "SF345528","nominal": 100.0},
}

MX_REF_CHOICES = {
    "CETES 28d": "SF43936",
    "CETES 91d": "SF43939",
    "CETES 182d":"SF43942",
    "CETES 364d":"SF43945",
    "CETES 728d":"SF349785",
    "MBono 3y":  "SF43883",
    "MBono 5y":  "SF43886",
    "MBono 10y": "SF44071",
    "MBono 20y": "SF45384",
    "MBono 30y": "SF60696",
}

# =========================
# UDI / BONDES helpers de alto nivel
# =========================
def udibono_nominal_ytm(
    liq: date,
    val: date,
    maturity: date,
    udibono_real_ytm_pct: float,
    token: str,
) -> Tuple[float, float, float, float, Dict[str, float]]:
    liq_dt = datetime(liq.year, liq.month, liq.day)
    val_dt = datetime(val.year, val.month, val.day)
    mat_dt = datetime(maturity.year, maturity.month, maturity.day)

    years_left = max(0.0, (mat_dt - val_dt).days / 365.0)
    try:
        realized_factor = bmx_realized_inflation_factor(liq_dt, val_dt, token)
    except Exception:
        realized_factor = 1.0
    realized_years = max(0.0, (val_dt - liq_dt).days / 365.0)
    pi_realized_avg = (realized_factor ** (1.0/realized_years) - 1.0) if realized_years > 1e-9 else 0.0
    pi_exp_avg, breakdown = bmx_inflation_expectations_path(years_left, token)
    total_years = realized_years + years_left
    if total_years <= 1e-9:
        pi_blend_avg = 0.0
    else:
        comp_total = ((1.0 + pi_realized_avg) ** realized_years) * ((1.0 + pi_exp_avg) ** years_left)
        pi_blend_avg = comp_total ** (1.0/total_years) - 1.0
    r_real = udibono_real_ytm_pct / 100.0
    r_nom = (1.0 + r_real) * (1.0 + pi_blend_avg) - 1.0
    pi_real_acum_pct = (realized_factor - 1.0) * 100.0 
    return r_nom * 100.0, pi_real_acum_pct, pi_exp_avg * 100.0, pi_blend_avg * 100.0, breakdown

def bondes_nominal_ytm(
    liq: date,
    val: date,
    maturity: date,
    spread_pp: float,
    token: str,
) -> Tuple[float, float, float, float]:
    years_left = max(0.0, (datetime(maturity.year, maturity.month, maturity.day) - datetime(val.year, val.month, val.day)).days / 365.0)
    years_real = max(0.0, (datetime(val.year, val.month, val.day) - datetime(liq.year, liq.month, liq.day)).days / 365.0)
    tiie_real = bmx_tiie_realized_avg(datetime(liq.year, liq.month, liq.day), datetime(val.year, val.month, val.day), token)  # %
    tiie_exp  = bmx_tiie_expected_avg(years_left, token)  # %
    total_years = years_real + years_left
    if total_years <= 1e-9:
        tiie_blend = 0.0
    else:
        tiie_blend = (tiie_real * years_real + tiie_exp * years_left) / total_years
    y_nom = tiie_blend + float(spread_pp)
    return y_nom, tiie_real, tiie_exp, tiie_blend

def _mbono_series_for_tenor(years: int) -> Tuple[str, str]:
    if years <= 3:  return "MBono 3y",  "SF43883"
    if years <= 5:  return "MBono 5y",  "SF43886"
    if years <= 10: return "MBono 10y", "SF44071"
    if years <= 20: return "MBono 20y", "SF45384"
    return "MBono 30y", "SF60696"

# =========================
# Categorías UI
# =========================
CAT_LABELS = {
    "CETES": "CETES",
    "MBONO": "MBonos",
    "UDIBONO": "UDIBONOS",
    "BONDESD": "BONDES D",
    "BONDESF": "BONDES F",
}
REV_CAT_LABELS = {v: k for k, v in CAT_LABELS.items()}

def _category_from_instrument(instr_name: str) -> str:
    meta = GOB_INSTRUMENTOS.get(instr_name, {})
    tipo = meta.get("tipo")
    return CAT_LABELS.get(tipo, "CETES")

def _instruments_for_category(cat_label: str) -> List[str]:
    tipo = REV_CAT_LABELS.get(cat_label, "CETES")
    kv = [(k, v.get("anios", v.get("dias", 0))) for k, v in GOB_INSTRUMENTOS.items() if v.get("tipo") == tipo]
    kv.sort(key=lambda x: x[1])
    return [k for k,_ in kv]

# =========================
# UI state y helpers
# =========================
def _init_state():
    if "fi_rows" not in st.session_state:
        st.session_state.fi_rows = []
    defaults = {
        "fi_tipo_bono": "Bono gubernamental",
        "fi_pais": "México",
        "fi_base": "ACT/360",
        "fi_freq": "Semestral",
        "fi_ytm": 0.0,
        "fi_maturity": _today(),
        "fi_nominal": 100.0,
        "fi_calif": "",
        "fi_liq": _today(),
        "fi_val": _today(),
        "fi_nombre": "",
        "fi_bench_label": "",
        "fi_bench_rate": 0.0,
        "fi_bench_us_choice": "UST 10Y (TNX)",
        "fi_gob_categoria": "CETES",
        "fi_gob_instrumento": "CETES 28d",
        "fi_maturity_mode": "Fecha",
        "fi_days_to_maturity": 28,
        "fi_ref_choice_version": 0,
        "_prev_tenor_days": None,
        "banxico_token": BANXICO_TOKEN_DEFAULT,
        "_prev_tipo_bono": None,
        "_prev_pais": None,
        "_do_clear_after_add": False,
        "fi_next_row_id": 1,
        "fi_ytm_view_version": 0,
        "_prev_gob_sel": None,
        "_prev_gob_cat": None,
        # UDI fields
        "fi_udi_pi_exp_pct": None,
        "fi_udi_real_ytm_at_liq_pct": None,
        "fi_udi_caption": "",
        # BONDES fields
        "fi_bondes_spread_pp": 0.00,
        "fi_bondes_price_label": "",
        "fi_bondes_caption": "",
        # UI helpers
        "fi_coupon_view_version": 0,
        "fi_instr_label": "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def _reset_capture():
    d = st.session_state
    d["fi_base"] = "ACT/360"
    d["fi_freq"] = "Semestral"
    d["fi_ytm"] = 0.0
    d["fi_maturity"] = _today()
    d["fi_nominal"] = 100.0
    d["fi_calif"] = ""
    d["fi_nombre"] = ""
    d["fi_bench_label"] = ""
    d["fi_bench_rate"] = 0.0
    d["fi_maturity_mode"] = "Fecha"
    d["fi_days_to_maturity"] = 28
    d["_prev_tenor_days"] = None
    d["fi_ref_choice_version"] = 0
    d["fi_val"] = _today()
    d["fi_udi_pi_exp_pct"] = None
    d["fi_udi_real_ytm_at_liq_pct"] = None
    d["fi_udi_caption"] = ""
    d["fi_bondes_spread_pp"] = 0.00
    d["fi_bondes_price_label"] = ""
    d["fi_bondes_caption"] = ""
    d["fi_instr_label"] = ""

def _update_mx_benchmark_by_series(series_name: str, liq: date, prefix: str = "fi_"):
    d = st.session_state
    token = d.get("banxico_token") or ""
    sid = MX_REF_CHOICES.get(series_name)
    
    key_bench_rate = f"{prefix}bench_rate"
    key_bench_label = f"{prefix}bench_label"

    if not (token and sid and liq):
        d[key_bench_rate] = 0.0
        d[key_bench_label] = series_name
        return
        
    f, v, _ = banxico_latest_on_or_before(sid, datetime(liq.year, liq.month, liq.day), token)
    
    if v is None:
        d[key_bench_rate] = 0.0
        d[key_bench_label] = f"{series_name} (sin dato)"
    else:
        d[key_bench_rate] = float(v)
        d[key_bench_label] = f"{series_name} (Banxico {f})"

def _update_instr_caption(selection: str, liq: date, prefix: str = "fi_"):
    """
    Genera el caption con información de Banxico para el instrumento seleccionado.
    Retorna el texto del caption (no solo lo guarda en el estado).
    """
    d = st.session_state
    token = d.get("banxico_token") or ""
    meta = GOB_INSTRUMENTOS.get(selection)
    
    key_instr_label = f"{prefix}instr_label"
    
    if not meta or not token or not isinstance(liq, date):
        d[key_instr_label] = ""
        return ""

    sid = meta.get("serie")
    liq_dt = datetime(liq.year, liq.month, liq.day)

    f, v, _ = banxico_latest_on_or_before(sid, liq_dt, token)

    tipo = meta.get("tipo")
    caption_text = ""
    
    if tipo in ("BONDESD", "BONDESF"):
        if v is not None and f:
            caption_text = f"Precio Ponderado de Colocación: **${float(v):,.2f}** (Banxico {f})"
        else:
            caption_text = "Precio ponderado de colocación — (sin dato)"
    else:
        if v is not None and f:
            caption_text = f"Rendimiento: **{float(v):.2f}%** (Banxico {f})"
        else:
            caption_text = "Rendimiento — (sin dato)"
    
    d[key_instr_label] = caption_text
    return caption_text

# tabs/fixed_income.py

def _prefill_from_gob(selection: str, liq: date, prefix: str = "fi_"):
    d = st.session_state
    meta = GOB_INSTRUMENTOS.get(selection, {})
    if not meta:
        return

    # --- Define dinámicamente TODAS las claves de estado necesarias ---
    key_nominal = f"{prefix}nominal"
    key_maturity = f"{prefix}maturity"
    key_freq = f"{prefix}freq"
    key_ytm = f"{prefix}ytm"
    key_ytm_pct = f"{prefix}ytm_pct" # Clave usada en el componente compacto
    key_cupon_edit = f"{prefix}cupon_edit" # Clave usada en la pestaña principal
    key_cupon_pct = f"{prefix}cupon_pct"   # Clave usada en el componente compacto
    key_bench_rate = f"{prefix}bench_rate"
    key_bench_label = f"{prefix}bench_label"
    key_maturity_mode = f"{prefix}maturity_mode"
    key_days_to_maturity = f"{prefix}days_to_maturity"
    key_ytm_view_version = f"{prefix}ytm_view_version"
    key_coupon_view_version = f"{prefix}coupon_view_version"
    # Claves para BONDES
    key_bondes_spread = f"{prefix}bondes_spread_pp"
    key_bondes_label = f"{prefix}bondes_price_label"
    # Claves para UDIBONOS
    key_udi_real_ytm = f"{prefix}udi_real_ytm_at_liq_pct"
    key_udi_pi_exp = f"{prefix}udi_pi_exp_pct"


    token = d.get("banxico_token") or ""
    tipo_instr = meta.get("tipo", "")

    # --- Lógica para UDIBONOS ---
    if tipo_instr == "UDIBONO":
        d[key_freq] = "Semestral"
        udi_sid = meta["serie"]
        f_real, v_real, _ = banxico_latest_on_or_before(udi_sid, datetime(liq.year, liq.month, liq.day), token)
        v_real = float(v_real) if v_real is not None else 0.0
        d[key_udi_real_ytm] = v_real
        d[key_maturity] = _add_years(liq, meta["anios"])
        pi_exp_avg, _ = bmx_inflation_expectations_path(meta["anios"], token)
        d[key_udi_pi_exp] = float(pi_exp_avg * 100.0)
        r_nom_est_pct = ((1.0 + v_real/100.0) * (1.0 + pi_exp_avg) - 1.0) * 100.0
        
        # Escribe en todas las posibles claves de cupón y YTM para ser compatible
        d[key_cupon_edit] = float(r_nom_est_pct)
        d[key_cupon_pct] = float(r_nom_est_pct)
        d[key_ytm] = float(r_nom_est_pct)
        d[key_ytm_pct] = float(r_nom_est_pct)

        d[key_coupon_view_version] = d.get(key_coupon_view_version, 0) + 1
        d[key_ytm_view_version] = d.get(key_ytm_view_version, 0) + 1
        
        ref_name, ref_sid = _mbono_series_for_tenor(meta["anios"])
        f_bm, v_bm, _ = banxico_latest_on_or_before(ref_sid, datetime(liq.year, liq.month, liq.day), token)
        d[key_bench_rate] = float(v_bm) if v_bm is not None else 0.0
        d[key_bench_label] = f"{ref_name} (Banxico {f_bm})" if v_bm is not None else f"{ref_name} (sin dato)"
        return

    # --- Lógica para BONDES ---
    if tipo_instr in ("BONDESD", "BONDESF"):
        d[key_freq] = "Mensual"
        d[key_maturity] = _add_years(liq, meta["anios"])
        spread_pp, px_label = bondes_spread_auto_by_sid(meta["serie"], datetime(liq.year, liq.month, liq.day), token)
        d[key_bondes_spread] = float(spread_pp)
        d[key_bondes_label] = px_label
        
        # Se necesita la fecha de valuación para el cálculo
        val_date = d.get(f"{prefix}val", liq)
        y_nom, _, _, _ = bondes_nominal_ytm(liq, val_date, d[key_maturity], float(spread_pp), token)

        # Escribe en todas las posibles claves de cupón y YTM
        d[key_cupon_edit] = float(y_nom)
        d[key_cupon_pct] = float(y_nom)
        d[key_ytm] = float(y_nom)
        d[key_ytm_pct] = float(y_nom)
        
        ref_name, ref_sid = _mbono_series_for_tenor(meta["anios"])
        f_bm, v_bm, _ = banxico_latest_on_or_before(ref_sid, datetime(liq.year, liq.month, liq.day), token)
        d[key_bench_rate] = float(v_bm) if v_bm is not None else 0.0
        d[key_bench_label] = f"{ref_name} (Banxico {f_bm})" if v_bm is not None else f"{ref_name} (sin dato)"
        return

    # --- LÓGICA CORREGIDA Y CENTRALIZADA PARA CETES Y MBONOS ---
    
    # 1. Obtiene la tasa de Banxico (asumiendo que _update_mx_benchmark_by_series está corregida)
    _update_mx_benchmark_by_series(selection, liq, prefix=prefix)
    
    # 2. Lee la tasa recién guardada del estado
    rate = d.get(key_bench_rate, 0.0)
    
    # 3. Asigna la misma tasa a TODOS los campos relevantes de YTM y Cupón
    d[key_ytm] = rate
    d[key_ytm_pct] = rate
    d[key_cupon_edit] = rate
    d[key_cupon_pct] = rate
    
    # 4. Asigna la frecuencia correcta
    if tipo_instr == "CETES":
        d[key_freq] = "Cero cupón"
    elif tipo_instr == "MBONO":
        d[key_freq] = "Semestral"

    # 5. Incrementa las "versiones" para forzar la actualización de la UI
    d[key_ytm_view_version] = d.get(key_ytm_view_version, 0) + 1
    d[key_coupon_view_version] = d.get(key_coupon_view_version, 0) + 1

    # 6. Actualiza la fecha de vencimiento
    if d.get(key_maturity_mode, "Fecha") == "Días":
        dias = d.get(key_days_to_maturity, 28)
        d[key_maturity] = liq + timedelta(days=int(dias))
    else:
        if "dias" in meta:
            d[key_maturity] = liq + timedelta(days=meta["dias"])
        elif "anios" in meta:
            d[key_maturity] = _add_years(liq, meta["anios"])

def render():
    _init_state()
    d = st.session_state

    # --- Seed robusto de fi_next_row_id basado en lo que ya hay ---
    if d.get("fi_rows"):
        try:
            max_id = max(int(r.get("row_id", 0)) for r in d.fi_rows)
        except Exception:
            max_id = 0
        # Solo sube el next si es menor o igual al máximo actual
        if int(d.get("fi_next_row_id", 1)) <= max_id:
            d["fi_next_row_id"] = max_id + 1
    else:
        # Si no hay filas, respeta el next si ya existe; si no, arranca en 1
        d.setdefault("fi_next_row_id", 1)


    if d.get("_do_clear_after_add"):
        _reset_capture(); d["_do_clear_after_add"] = False

    with st.expander(" Agregar Bono", expanded=False):
        # 1) Tipo & País
        col_top1, col_top2 = st.columns(2)
        tipo_bono = col_top1.selectbox("Tipo de bono", options=TIPOS_BONO, key="fi_tipo_bono")
        pais = col_top2.selectbox("País", options=PAISES, key="fi_pais")

        if d.get("_prev_tipo_bono") is not None and tipo_bono != d.get("_prev_tipo_bono"):
            _reset_capture()
        if d.get("_prev_pais") is not None and pais != d.get("_prev_pais"):
            _reset_capture()

        # 2) Fechas & base
        col_dates1, col_dates2, col_dates3 = st.columns(3)
        liquidacion = col_dates1.date_input("Fecha de liquidación / compra", key="fi_liq")
        base = col_dates2.selectbox("Base de días", options=BASES, key="fi_base")
        valuacion = col_dates3.date_input("Fecha de valuación", key="fi_val", value=_today())

        col_mode1, col_mode2 = st.columns([0.45, 0.55])
        maturity_mode = col_mode1.radio("Modo de vencimiento", options=["Fecha","Días"], key="fi_maturity_mode", horizontal=True)
        if maturity_mode == "Días":
            days_val = col_mode2.number_input("Días al vencimiento", min_value=1, step=1, key="fi_days_to_maturity")
            if liquidacion:
                d["fi_maturity"] = liquidacion + timedelta(days=int(days_val))

        # 3) Gob MX (con Categoría)
        is_gob_mx = (tipo_bono == "Bono gubernamental" and d.get("fi_pais") == "México")
        if is_gob_mx:
            if d.get("_prev_gob_cat") is None and d.get("fi_gob_instrumento"):
                d["fi_gob_categoria"] = _category_from_instrument(d["fi_gob_instrumento"])

            col_cat, col_inst = st.columns([0.42, 0.58])
            cat_label = col_cat.selectbox("Categoría", options=list(CAT_LABELS.values()), key="fi_gob_categoria")

            cat_options = _instruments_for_category(cat_label)
            current_instr = d.get("fi_gob_instrumento")
            if current_instr not in cat_options and cat_options:
                d["fi_gob_instrumento"] = cat_options[0]

            default_index = 0
            if d.get("fi_gob_instrumento") in cat_options:
                default_index = cat_options.index(d["fi_gob_instrumento"])

            gob_sel = col_inst.selectbox("Selecciona instrumento", options=cat_options, index=default_index, key="fi_gob_instrumento")

            if liquidacion:
                _prefill_from_gob(gob_sel, liquidacion)
                _update_instr_caption(gob_sel, liquidacion)
                if d.get("fi_instr_label"):
                    st.caption(d.get("fi_instr_label"))

            # Detecta cambios de selección/categoría
            if liquidacion:
                if d.get("_prev_gob_sel") != gob_sel or d.get("_prev_gob_cat") != cat_label:
                    if gob_sel.startswith("MBono"):
                        d["fi_cupon_edit"] = float(d.get("fi_bench_rate", 0.0)); d["fi_ytm"] = d["fi_cupon_edit"]
                    elif gob_sel.startswith("CETES"):
                        d["fi_ytm"] = float(d.get("fi_bench_rate", 0.0))
                    meta = GOB_INSTRUMENTOS.get(gob_sel, {})
                    if meta.get("tipo") in ("BONDESD","BONDESF"):
                        token = d.get("banxico_token") or BANXICO_TOKEN_DEFAULT
                        spread_pp, px_label = bondes_spread_auto_by_sid(
                            meta["serie"],
                            datetime(d["fi_liq"].year, d["fi_liq"].month, d["fi_liq"].day),
                            token
                        )
                        d["fi_bondes_spread_pp"] = float(spread_pp)
                        d["fi_bondes_price_label"] = px_label
                    d["fi_ytm_view_version"] = int(d.get("fi_ytm_view_version", 0)) + 1
                    d["fi_coupon_view_version"] = int(d.get("fi_coupon_view_version", 0)) + 1
                    d["_prev_gob_sel"] = gob_sel
                    d["_prev_gob_cat"] = cat_label

        # 4) Datos del bono
        if not is_gob_mx:
            st.text_input("Nombre del Bono", key="fi_nombre")

        col_row1, col_row2, col_row3 = st.columns(3)

        # Frecuencia (bloqueada para gob MX)
        freq_disabled = is_gob_mx
        freq_label = col_row1.selectbox(
            "Frecuencia",
            options=list(FREQ_MAP.keys()),
            key="fi_freq",
            disabled=freq_disabled
        )

        zero_coupon = (freq_label == "Cero cupón")
        gob_sel = d.get("fi_gob_instrumento", "")
        meta = GOB_INSTRUMENTOS.get(gob_sel, {}) if is_gob_mx else {}
        t_tipo = meta.get("tipo", "")
        is_mbono = is_gob_mx and t_tipo == "MBONO"
        is_udi   = is_gob_mx and t_tipo == "UDIBONO"
        is_bondes = is_gob_mx and t_tipo in ("BONDESD","BONDESF")

        # CUPÓN
        if zero_coupon:
            col_row2.number_input("Tasa Cupón (%)", value=0.00, step=0.01, disabled=True, key="fi_cupon_view")
            cupon_pct = 0.0
        else:
            if is_udi or is_bondes:
                cupon_pct = float(d.get("fi_cupon_edit", 0.0))
                cvk = f"fi_cupon_view_locked_{int(d.get('fi_coupon_view_version',0))}"
                label_locked = "Tasa Cupón (%) — nominal estimada"
                col_row2.number_input(label_locked, value=cupon_pct, step=0.01, disabled=True, key=cvk)
            else:
                cupon_pct = col_row2.number_input("Tasa Cupón (%)", step=0.01, min_value=0.0, key="fi_cupon_edit")

        # Vencimiento (bloqueado para gob MX)
        maturity_disabled = is_gob_mx
        maturity = col_row3.date_input("Fecha de Vencimiento", key="fi_maturity", disabled=maturity_disabled)

        if is_mbono:
            d["fi_ytm"] = float(cupon_pct)

        # --- Fila: Nominal | YTM | Calificación ---
        col_row4, col_row5, col_row6 = st.columns(3)

        # Nominal / Valor facial (bloqueado para gob MX)
        nominal_disabled = is_gob_mx
        nominal = col_row4.number_input(
            "Nominal / Valor facial",
            step=1.0,
            min_value=0.01,
            key="fi_nominal",
            disabled=nominal_disabled
        )

        # YTM (centro) — bloqueado para gob MX
        if is_gob_mx:
            ytm_val_locked = float(d.get("fi_ytm", 0.0))
            ytm_view_key = f"fi_ytm_view_{d.get('fi_ytm_view_version', 0)}"
            col_row5.number_input(
                "Rendimiento (YTM %) — anual",
                value=ytm_val_locked,
                step=0.01,
                disabled=True,
                key=ytm_view_key
            )
            ytm_pct = ytm_val_locked
        else:
            ytm_pct = col_row5.number_input(
                "Rendimiento (YTM %) — anual",
                step=0.01,
                min_value=0.0,
                key="fi_ytm"
            )

        # Calificación (derecha)
        if is_gob_mx:
            calif_value = "Soberano MX" if d.get("fi_pais") == "México" else "Soberano US"
            d["fi_calif"] = calif_value
            col_row6.selectbox(
                "Calificación (auto)",
                options=[calif_value],
                index=0,
                key="fi_calif_auto_view",
                disabled=True
            )
        else:
            rating_opts = MX_RATINGS if d.get("fi_pais") == "México" else US_RATINGS
            d["fi_calif"] = col_row6.selectbox(
                "Calificación (Fitch/S&P/Moody's)",
                options=rating_opts,
                index=0,
                key="fi_calif_select"
            )

        # --- UDI y BONDES: captions informativos ---
        if is_udi:
            try:
                token = d.get("banxico_token") or BANXICO_TOKEN_DEFAULT
                udi_sid = meta["serie"]
                _, real_at_liq, _ = banxico_latest_on_or_before(
                    udi_sid,
                    datetime(d["fi_liq"].year, d["fi_liq"].month, d["fi_liq"].day),
                    token
                )
                real_at_liq = float(real_at_liq) if real_at_liq is not None else (d.get("fi_udi_real_ytm_at_liq_pct") or 0.0)

                y_nom_calc, pi_real_acum_pct, pi_exp_avg_pct, pi_blend_avg_pct, _ = udibono_nominal_ytm(
                    d["fi_liq"], d["fi_val"], d["fi_maturity"], real_at_liq, token
                )

                years_real = max(
                    0.0,
                    (datetime(d["fi_val"].year, d["fi_val"].month, d["fi_val"].day)
                     - datetime(d["fi_liq"].year, d["fi_liq"].month, d["fi_liq"].day)
                    ).days / 365.0
                )
                if years_real > 1e-9 and pi_real_acum_pct is not None:
                    pi_real_ann_pct = ((1.0 + pi_real_acum_pct/100.0)**(1.0/years_real) - 1.0) * 100.0
                else:
                    pi_real_ann_pct = 0.0

                caption_parts = [f"**YTM real** ({real_at_liq:.2f}%)"]
                if d.get("fi_liq") < d.get("fi_val"):
                    caption_parts.append(f"**inflación realizada anualizada** ({pi_real_ann_pct:.2f}%)")
                caption_parts.append(f"**inflación esperada promedio** ({pi_exp_avg_pct:.2f}%)")
                caption_string = " + ".join(caption_parts)
                d["fi_udi_caption"] = f"{caption_string} ≈ **nominal** ({d['fi_ytm']:.2f}%)"
                st.caption(d["fi_udi_caption"])

                d["fi_cupon_edit"] = float(d.get("fi_ytm", y_nom_calc))
                d["fi_ytm"] = float(d.get("fi_ytm", y_nom_calc))
                d["fi_coupon_view_version"] += 1
                d["fi_ytm_view_version"] += 1
            except Exception:
                pass

        if is_bondes:
            try:
                token = d.get("banxico_token") or BANXICO_TOKEN_DEFAULT
                spread_pp, px_label = bondes_spread_auto_by_sid(
                    meta["serie"],
                    datetime(d["fi_liq"].year, d["fi_liq"].month, d["fi_liq"].day),
                    token
                )
                d["fi_bondes_spread_pp"] = float(spread_pp)
                d["fi_bondes_price_label"] = px_label
                y_nom, tiie_real, tiie_exp, tiie_blend = bondes_nominal_ytm(
                    d["fi_liq"], d["fi_val"], d["fi_maturity"],
                    float(d["fi_bondes_spread_pp"]), token
                )
                if d.get("fi_liq") == d["fi_val"]:
                    d["fi_bondes_caption"] = (
                        f"**TIIE esperada promedio** ({tiie_exp:.2f}%) "
                        f"+ **spread** ({float(d['fi_bondes_spread_pp']):.2f} pp) ≈ "
                        f"**nominal** ({y_nom:.2f}%)"
                    )
                else:
                    d["fi_bondes_caption"] = (
                        f"**TIIE realizada** ({tiie_real:.2f}%) + "
                        f"**TIIE esperada promedio** ({tiie_exp:.2f}%) + "
                        f"**spread** ({float(d['fi_bondes_spread_pp']):.2f} pp) ≈ "
                        f"**nominal** ({y_nom:.2f}%)"
                    )
                st.caption(d["fi_bondes_caption"])
                d["fi_cupon_edit"] = float(y_nom); d["fi_ytm"] = float(y_nom)
                d["fi_coupon_view_version"] += 1; d["fi_ytm_view_version"] += 1
            except Exception:
                pass

        # 5) Benchmarks
        if d.get("fi_pais") == "EE.UU.":
            colb1, colb2 = st.columns([0.5, 0.5])
            bench_choice = colb1.selectbox("Referencia EE.UU.", options=US_BENCH_CHOICES, key="fi_bench_us_choice")
            bench_val = _fetch_us_benchmark(bench_choice) if YF_OK else None
            if bench_val is None:
                colb2.metric("Benchmark (%)", "—"); d["fi_bench_rate"] = 0.0; d["fi_bench_label"] = bench_choice + " (sin dato)"
            else:
                colb2.metric("Benchmark (%)", f"{bench_val:.2f}"); d["fi_bench_rate"] = float(bench_val); d["fi_bench_label"] = bench_choice
        else:
            if is_gob_mx:
                st.metric(
                    "Benchmark (%)",
                    f"{d.get('fi_bench_rate',0.0):.2f}" if d.get("fi_bench_rate",0.0) else "—"
                )
                st.caption(d.get("fi_bench_label") or "—")
            else:
                colm1, _ = st.columns([0.5, 0.5])
                tenor_days = (maturity - liquidacion).days if (liquidacion and maturity) else None
                if tenor_days is not None and d.get("_prev_tenor_days") != tenor_days:
                    d["_prev_tenor_days"] = tenor_days; d["fi_ref_choice_version"] += 1
                suggested = _auto_pick_mx_ref_name(liquidacion, maturity) if (liquidacion and maturity) else "MBono 5y"
                options = list(MX_REF_CHOICES.keys()); default_index = options.index(suggested) if suggested in options else 0
                ref_key = f"fi_ref_choice_{d.get('fi_ref_choice_version',0)}"
                ref_name = colm1.selectbox("Referencia México", options=options, index=default_index, key=ref_key)
                if liquidacion: _update_mx_benchmark_by_series(ref_name, liquidacion)
                st.metric(
                    "Benchmark (%)",
                    f"{d.get('fi_bench_rate',0.0):.2f}" if d.get("fi_bench_rate",0.0) else "—"
                )
                st.caption(d.get("fi_bench_label") or "—")

        # 6) Calcular y agregar
        add_btn = st.button("Calcular y agregar", type="primary")
        if add_btn:
            try:
                use_benchmark = float(d.get("fi_bench_rate", 0.0))
                freq = FREQ_MAP[freq_label]

                if is_udi:
                    token = d.get("banxico_token") or BANXICO_TOKEN_DEFAULT
                    udi_sid = meta["serie"]
                    _, real_at_liq, _ = banxico_latest_on_or_before(
                        udi_sid, datetime(d["fi_liq"].year, d["fi_liq"].month, d["fi_liq"].day), token
                    )
                    real_ytm_pct = float(real_at_liq) if real_at_liq is not None else float(d.get("fi_udi_real_ytm_at_liq_pct") or 0.0)
                    _ = udibono_nominal_ytm(
                        liq=d["fi_liq"], val=d["fi_val"], maturity=maturity,
                        udibono_real_ytm_pct=real_ytm_pct, token=token
                    )
                    ytm_for_pricing = float(d.get("fi_ytm"))

                elif is_bondes:
                    token = d.get("banxico_token") or BANXICO_TOKEN_DEFAULT
                    spread_pp, _ = bondes_spread_auto_by_sid(
                        meta["serie"], datetime(d["fi_liq"].year, d["fi_liq"].month, d["fi_liq"].day), token
                    )
                    d["fi_bondes_spread_pp"] = float(spread_pp)
                    ytm_nominal_equiv_pct, _, _, _ = bondes_nominal_ytm(
                        liq=d["fi_liq"], val=d["fi_val"], maturity=maturity,
                        spread_pp=float(d["fi_bondes_spread_pp"]), token=token
                    )
                    ytm_for_pricing = ytm_nominal_equiv_pct
                else:
                    ytm_for_pricing = float(d.get("fi_ytm"))

                dirty, clean, accrued, last_cp, next_cp, _, _ = price_bond_dirty_clean(
                    valuation=d.get("fi_val"),
                    maturity=maturity,
                    coupon_annual_pct=float(cupon_pct),
                    ytm_annual_pct=float(ytm_for_pricing),
                    freq_per_year=freq,
                    nominal=float(nominal),
                    base=base,
                    anchor=d.get("fi_liq"),
                )

                spread_pp_val = float(ytm_for_pricing) - use_benchmark
                nombre = d.get("fi_nombre", "") if not is_gob_mx else gob_sel

                gov_subtype = None
                if is_gob_mx:
                    if t_tipo == "CETES":    gov_subtype = "CETES"
                    elif t_tipo == "MBONO":  gov_subtype = "MBONO"
                    elif t_tipo == "UDIBONO":gov_subtype = "UDIBONO"
                    elif t_tipo == "BONDESD":gov_subtype = "BONDES D"
                    elif t_tipo == "BONDESF":gov_subtype = "BONDES F"

                # Asegura un ID único aunque el next esté desfasado
                existing_ids = {int(r.get("row_id", 0)) for r in d.fi_rows} if d.get("fi_rows") else set()
                next_id = int(d.get("fi_next_row_id", 1))
                if next_id in existing_ids:
                    next_id = (max(existing_ids) + 1) if existing_ids else 1


                fila = {
                    "row_id": next_id,
                    "Bono": nombre,
                    "País": d.get("fi_pais"),
                    "Tipo de bono": "Bono gubernamental" if is_gob_mx else tipo_bono,
                    **({"Subtipo (Gob)": gov_subtype} if gov_subtype else {}),
                    "COUPON (%)": round(float(cupon_pct), 4),
                    "Frecuencia": freq_label,
                    "Liquidación": d.get("fi_liq").strftime("%Y-%m-%d"),
                    "Maturity": maturity.strftime("%Y-%m-%d"),
                    "Fitch": "Soberano MX" if is_gob_mx else d.get("fi_calif",""),
                    "Nominal": float(nominal),
                    "Rend (%)": round(float(ytm_for_pricing), 4),
                    "Benchmark (%)": round(use_benchmark, 4),
                    "Spread (pp)": round(spread_pp_val, 4),
                    "Último Cupón": ("—" if last_cp is None else last_cp.strftime("%Y-%m-%d")),
                    "Siguiente Cupón": next_cp.strftime("%Y-%m-%d"),
                    "Intereses D": round(float(accrued), 6),
                    "Precio Sucio": round(float(dirty), 6),
                    "Precio Limpio": round(float(clean), 6),
                    "_benchmark_name": d.get("fi_bench_label",""),
                }

                d.fi_rows.append(fila)
                d["fi_next_row_id"] = next_id + 1
                d["_do_clear_after_add"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Error al valuar: {e}")

    d["_prev_tipo_bono"] = d.get("fi_tipo_bono")
    d["_prev_pais"] = d.get("fi_pais")

    # =========================
    # Multiselect para eliminar bonos por row_id
    # =========================
    if st.session_state.fi_rows:
        # Mantén IDs internamente; muestra solo el nombre en la UI
        id2name = {r["row_id"]: r.get("Bono", "(sin nombre)") for r in st.session_state.fi_rows}
        options_ids = list(id2name.keys())

        # Estado inicial: todos seleccionados (guardamos IDS, no textos)
        if "fi_ms_keep_ids" not in st.session_state:
            st.session_state.fi_ms_keep_ids = options_ids.copy()

        # Si llegaron nuevos bonos desde la última vez, agrégalos por defecto
        nuevos_ids = [rid for rid in options_ids if rid not in st.session_state.fi_ms_keep_ids]
        if nuevos_ids:
            st.session_state.fi_ms_keep_ids.extend(nuevos_ids)

        # Multiselect trabajando con IDs pero mostrando solo el nombre
        current_sel_ids = st.multiselect(
            "Bonos Capturados",
            options=options_ids,
            default=st.session_state.fi_ms_keep_ids,
            key="fi_ms_widget",
            format_func=lambda rid: id2name.get(rid, f"(#{rid})")
        )

        # Detecta desmarcados por ID y elimina del estado/tabla
        removed_ids = set(st.session_state.fi_ms_keep_ids) - set(current_sel_ids)
        if removed_ids:
            st.session_state.fi_rows = [
                r for r in st.session_state.fi_rows if r.get("row_id") not in removed_ids
            ]
            st.session_state.fi_ms_keep_ids = list(current_sel_ids)
            st.rerun()
        else:
            # Sin cambios: sincroniza selección
            st.session_state.fi_ms_keep_ids = list(current_sel_ids)
    else:
        st.info("Agrega un bono para poder usar el multiselect de eliminación.")


    # =========================
    # Tabla de resultados con selección de fila
    # =========================
    rows: List[Dict] = st.session_state.fi_rows
    if not rows:
        st.info("Agrega un bono para ver la tabla de valuación.")
        return

    st.caption(f"{len(rows)} bonos capturados.")
    base_df = pd.DataFrame(rows)

    visible_cols = [
        "Bono", "País",
        "Tipo de bono",
        "COUPON (%)", "Frecuencia",
        "Maturity",
        "Fitch", "Rend (%)",
        "Benchmark (%)", "Spread (pp)",
        "Intereses D", "Precio Sucio", "Precio Limpio",
    ]
    visible_cols = [c for c in visible_cols if c in base_df.columns]

    df_full = base_df[["row_id"] + visible_cols + ["Liquidación", "Nominal"]].copy()
    df_view = df_full[visible_cols].copy()

    event = st.dataframe(
        df_view,
        key="fi_table_view",
        width="stretch",
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun"
    )

    sel_rows = event.get("selection", {}).get("rows", []) if isinstance(event, dict) else getattr(getattr(event, "selection", None), "rows", []) or []

    # =========================
    # Métricas + calendario + gráfica del bono seleccionado
    # =========================
    if sel_rows:
        i = sel_rows[0]
        try:
            row_full = df_full.iloc[i]
        except Exception:
            row_full = None

        if row_full is not None:
            try:
                # --- Fechas y parámetros (usa date puro) ---
                liq_str = str(base_df.loc[base_df["row_id"] == row_full["row_id"], "Liquidación"].iloc[0])
                mat_str = str(row_full["Maturity"])

                liq = datetime.strptime(liq_str, "%Y-%m-%d").date()
                mat = datetime.strptime(mat_str, "%Y-%m-%d").date()

                nominal = float(base_df.loc[base_df["row_id"] == row_full["row_id"], "Nominal"].iloc[0])
                cupon_pct = float(row_full["COUPON (%)"])
                ytm_pct   = float(row_full["Rend (%)"])
                freq_lbl  = str(row_full["Frecuencia"])
                freq      = FREQ_MAP.get(freq_lbl, 2)
                base_dc   = "ACT/360"

                if mat < liq:
                    st.error("La fecha de vencimiento es menor que la de liquidación.")
                    return

                # Valuación hoy (si hoy está en vida del bono)
                hoy = date.today()
                val_today = hoy if (liq <= hoy <= mat) else liq

                dirty_hoy, clean_hoy, accrued_hoy, last_cp_hoy, next_cp_hoy, _, _ = price_bond_dirty_clean(
                    valuation=val_today,
                    maturity=mat,
                    coupon_annual_pct=cupon_pct,
                    ytm_annual_pct=ytm_pct,
                    freq_per_year=freq,
                    nominal=nominal,
                    base=base_dc,
                    anchor=liq,
                )

                # Días al vencimiento
                days_to_mat = max(0, (mat - hoy).days)

                # Duración modificada aprox (central diff sobre precio limpio)
                def _clean_at_y(ytm_pct_local: float) -> float:
                    _, cpx, _, _, _, _, _ = price_bond_dirty_clean(
                        valuation=val_today,
                        maturity=mat,
                        coupon_annual_pct=cupon_pct,
                        ytm_annual_pct=ytm_pct_local,
                        freq_per_year=freq,
                        nominal=nominal,
                        base=base_dc,
                        anchor=liq,
                    )
                    return float(cpx)

                y0 = ytm_pct / 100.0
                dy = 0.0001  # 1 bp
                p0 = float(clean_hoy)
                p_up = _clean_at_y((y0 + dy) * 100.0)
                p_dn = _clean_at_y((y0 - dy) * 100.0)
                mod_duration = -((p_up - p_dn) / (2.0 * dy)) / p0 if p0 > 0 else 0.0

                # ======= MÉTRICAS (una sola fila) =======
                c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
                with c1:
                    st.metric("YTM (anual)", f"{ytm_pct:.2f}%")
                with c2:
                    st.metric("Cupón", f"{cupon_pct:.2f}%")
                with c3:
                    st.metric("Frecuencia", freq_lbl)
                with c4:
                    st.metric("Precio Limpio", f"${clean_hoy:,.2f}")
                with c5:
                    st.metric("Precio Sucio", f"${dirty_hoy:,.2f}")
                with c6:
                    st.metric("Días al venc.", f"{days_to_mat:,}")
                with c7:
                    st.metric("Próx. Cupón", next_cp_hoy.strftime("%Y-%m-%d") if isinstance(next_cp_hoy, date) else "—")


                # ======= CALENDARIO (izq) + GRÁFICA INTERÉS COMPUESTO (der) =======
                left, right = st.columns(2)

                # --- Calendario de pagos (lado izquierdo) ---
                pagos_data = []
                if freq > 0:
                    months_per_period = 12 // freq
                    fecha_cupon = liq
                    cupon_pago = nominal * (cupon_pct / 100.0) / freq
                    num_pago = 0
                    while fecha_cupon <= mat:
                        fecha_cupon = _add_months(fecha_cupon, months_per_period)
                        if liq < fecha_cupon <= mat:
                            num_pago += 1
                            es_final = (fecha_cupon == mat or _add_months(fecha_cupon, months_per_period) > mat)
                            total_pago = cupon_pago + (nominal if es_final else 0.0)
                            pagos_data.append({
                                "#": num_pago,
                                "Fecha": fecha_cupon.strftime("%Y-%m-%d"),
                                "Cupón ($)": cupon_pago,
                                "Principal ($)": (nominal if es_final else 0.0),
                                "Total ($)": total_pago,
                                "Días desde hoy": (fecha_cupon - hoy).days if fecha_cupon >= hoy else 0
                            })
                else:
                    pagos_data.append({
                        "#": 1,
                        "Fecha": mat.strftime("%Y-%m-%d"),
                        "Cupón ($)": 0.0,
                        "Principal ($)": nominal,
                        "Total ($)": nominal,
                        "Días desde hoy": (mat - hoy).days if mat >= hoy else 0
                    })

                pagos_futuros = [p for p in pagos_data if p["Días desde hoy"] > 0]

                with left:
                    if pagos_futuros:
                        df_pagos_futuros = pd.DataFrame(pagos_futuros)
                        for coln in ["Cupón ($)", "Principal ($)", "Total ($)"]:
                            df_pagos_futuros[coln] = df_pagos_futuros[coln].map(lambda x: f"${x:,.2f}")
                        st.dataframe(df_pagos_futuros, use_container_width=True, hide_index=True)

                    else:
                        st.info("No hay pagos futuros pendientes (bono vencido o con todos los pagos realizados).")

                # --- Gráfica de interés compuesto (lado derecho) ---
                with right:

                    # Genera fechas como date puro
                    dates: List[date] = []
                    cur = liq
                    while cur <= mat:
                        dates.append(cur)
                        cur = (datetime.combine(cur, datetime.min.time()) + timedelta(days=1)).date()

                    # Precalcula precio sucio en liquidación una sola vez
                    dirty_at_liq, _, _, _, _, _, _ = price_bond_dirty_clean(
                        valuation=liq,
                        maturity=mat,
                        coupon_annual_pct=cupon_pct,
                        ytm_annual_pct=ytm_pct,
                        freq_per_year=freq,
                        nominal=nominal,
                        base=base_dc,
                        anchor=liq,
                    )

                    # Fechas de cupón para marcar en la gráfica
                    fechas_cupones: List[date] = []
                    if freq > 0:
                        months_per_period = 12 // freq
                        fecha_cupon = liq
                        while fecha_cupon <= mat:
                            fecha_cupon = _add_months(fecha_cupon, months_per_period)
                            if liq < fecha_cupon <= mat:
                                fechas_cupones.append(fecha_cupon)

                    # Serie de valores acumulados
                    valores_acumulados = []
                    valores_cupones = []
                    for dte in dates:
                        years_elapsed = max(0.0, (dte - liq).days / 365.25)
                        valor_acum = dirty_at_liq * ((1 + ytm_pct/100.0) ** years_elapsed)
                        valores_acumulados.append(valor_acum)
                        if dte in fechas_cupones:
                            valores_cupones.append(valor_acum)

                    # Convertimos solo para graficar
                    fechas_dt = [pd.Timestamp(d.year, d.month, d.day) for d in dates]
                    fechas_cupones_dt = [pd.Timestamp(d.year, d.month, d.day) for d in fechas_cupones]

                    plot_df = pd.DataFrame({
                        "Fecha": fechas_dt,
                        "Valor Acumulado": valores_acumulados
                    })

                    fig = go.Figure()

                    # Curva principal
                    fig.add_trace(go.Scatter(
                        x=plot_df["Fecha"],
                        y=plot_df["Valor Acumulado"],
                        mode="lines",
                        name="Acumulación de interés",
                        line=dict(color="#1f77b4", width=2.5)
                    ))

                    # Líneas verticales de cupones
                    for i, fecha_cupon in enumerate(fechas_cupones_dt):
                        if i < len(valores_cupones):
                            fig.add_shape(
                                type="line",
                                x0=fecha_cupon,
                                x1=fecha_cupon,
                                y0=0,
                                y1=valores_cupones[i],
                                line=dict(color="green", width=1, dash="dot")
                            )
                            if i % max(1, len(fechas_cupones_dt) // 5) == 0:
                                fig.add_annotation(
                                    x=fecha_cupon,
                                    y=valores_cupones[i],
                                    text="Cupón",
                                    showarrow=False,
                                    yshift=15,
                                    font=dict(size=9, color="green")
                                )

                    # Línea de "Hoy"
                    hoy_ts = pd.Timestamp(date.today())
                    if fechas_dt and fechas_dt[0] <= hoy_ts <= fechas_dt[-1]:
                        fig.add_shape(
                            type="line",
                            x0=hoy_ts,
                            x1=hoy_ts,
                            y0=0,
                            y1=1,
                            yref="paper",
                            line=dict(color="red", width=2, dash="dash")
                        )
                        fig.add_annotation(
                            x=hoy_ts,
                            y=1,
                            yref="paper",
                            text="Hoy",
                            showarrow=False,
                            yshift=10
                        )

                    fig.update_layout(
                        xaxis_title="Fecha",
                        yaxis_title="Valor Acumulado ($)",
                        margin=dict(l=10, r=10, t=50, b=10),
                        height=380,
                        showlegend=False,
                    )

                    # Anotación al vencimiento
                    if fechas_dt:
                        fig.add_annotation(
                            x=fechas_dt[-1],
                            y=float(plot_df["Valor Acumulado"].iloc[-1]),
                            text=f"Vencimiento: ${plot_df['Valor Acumulado'].iloc[-1]:,.2f}",
                            showarrow=True,
                            arrowhead=2,
                            ax=40, ay=-40
                        )

                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"💥 ERROR: {e}")
                import traceback
                st.code(traceback.format_exc())