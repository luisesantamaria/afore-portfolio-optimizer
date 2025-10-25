from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Set
from pathlib import Path
import datetime as dt
from datetime import date, datetime
import json

import numpy as np
import pandas as pd
import streamlit as st

from tabs import consar_limits
from tabs import fx  # <-- Correcto
from tabs.equities import _country_from_index  # mantenemos esto
from tabs.yf_store import preload_hist_5y_daily, get_hist_5y, preload_meta
from tabs import fixed_income as fi  # <-- para CETES 28d (Banxico)


# ===== Constantes públicas que usan otras tabs =====
TRADING_DAYS = 252
Z_95 = 1.645
CVaR_OVER_VaR_95 = 1.25
OP_BUFFER = 0.03
RISK_FREE_RATE = 0.02  # fallback si Banxico no responde


# ===== Persistencia simple (sin cambios) =====
_STATE_PATH = Path("./afore_state.json")
_PERSIST_KEYS = [
    "equity_selection",
    "fi_rows",
    "fx_selection",
    "fx_basket",
    "fx_picks",
    "siefore_selected",
    "afore_selected",
    "structured_notes",
]

def _snapshot_state() -> dict:
    snap = {}
    keys_to_persist = set(_PERSIST_KEYS) | {"consar_limits_selected", "consar_limits_current", "eq_picks", "_eq_last_picks"}
    for k in keys_to_persist:
        if k in st.session_state:
            try:
                def _to_jsonable_dummy(obj): return str(obj)
                snap[k] = _to_jsonable_dummy(st.session_state[k])
            except Exception:
                pass
    for k in ("fx_picks", "fx_basket"):
        v = snap.get(k, [])
        if v is None:
            snap[k] = []
        elif not isinstance(v, list):
            snap[k] = list(v) if v else []
    snap["_meta"] = {"saved_at": dt.datetime.now().isoformat()}
    return snap

def _restore_state(data: dict) -> None:
    for k, v in data.items():
        if k in st.session_state:
             st.session_state[k] = v

def _auto_load_once():
    if st.session_state.get("_autosave_loaded", False):
        return
    if _STATE_PATH.exists():
        try:
            data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
            _restore_state(data)
            try:
                for k in ("fx_picks", "fx_basket"):
                    v = st.session_state.get(k, [])
                    if v is None:
                        st.session_state[k] = []
                    elif not isinstance(v, list):
                        st.session_state[k] = list(v) if v else []
            except Exception:
                st.session_state.setdefault("fx_picks", [])
                st.session_state.setdefault("fx_basket", [])
            st.session_state["_autosave_loaded"] = True
            st.toast("Configuración cargada automáticamente.", icon="✅")
        except Exception as e:
            st.session_state["_autosave_loaded"] = True
            st.toast(f"No se pudo cargar configuración: {e}", icon="⚠️")

def _ensure_sticky_siefore_afore() -> None:
    """
    Mantiene 'siefore_selected' y 'afore_selected' pegajosos (sticky) durante la sesión.
    - Si hay valor actual -> lo copia a _sticky_*
    - Si no hay valor actual pero sí sticky -> restaura desde _sticky_*
    No guarda nada en disco; todo es session_state.
    """
    for key in ("siefore_selected", "afore_selected"):
        sticky_key = f"_sticky_{key}"
        val = st.session_state.get(key, None)
        if val not in (None, "", "N/A", "—"):
            # Tenemos selección viva: actualiza el sticky
            st.session_state[sticky_key] = val
        else:
            # Si está vacío pero tenemos sticky, restaura
            sticky_val = st.session_state.get(sticky_key, None)
            if sticky_val not in (None, "", "N/A", "—"):
                st.session_state[key] = sticky_val


# ===== Utilidades de series =====
def _annual_vol(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.size else np.nan

def _avg_annual_return(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() * TRADING_DAYS) if r.size else np.nan


# ===== Risk-free dinámico (CETES 28d Banxico) =====
def _get_risk_free_ann() -> float:
    """
    Devuelve tasa libre de riesgo anual en tanto.
    Prioridad:
      1) st.session_state["risk_free_rate"] (override manual)
      2) Banxico CETES 28d (serie SF43936)
      3) Fallback: RISK_FREE_RATE (0.02)
    """
    rf_manual = st.session_state.get("risk_free_rate")
    if isinstance(rf_manual, (int, float)) and np.isfinite(rf_manual) and rf_manual >= 0:
        st.session_state["risk_free_rate_source"] = "manual"
        return float(rf_manual)

    try:
        token = st.session_state.get("banxico_token") or fi.BANXICO_TOKEN_DEFAULT
        series_id = fi.MX_REF_CHOICES.get("CETES 28d", "SF43936")
        today_dt = datetime.now()
        f, v, _ = fi.banxico_latest_on_or_before(series_id, today_dt, token)
        if v is not None and np.isfinite(v):
            rf = float(v) / 100.0
            st.session_state["risk_free_rate"] = rf
            st.session_state["risk_free_rate_source"] = f"CETES 28d Banxico ({f})"
            return rf
    except Exception:
        pass

    st.session_state["risk_free_rate_source"] = "fallback_const"
    return float(RISK_FREE_RATE)


# ===== Helpers de series para métricas (solo visualización) =====
def _daily_pct_from_hist(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    ser = df.get("Adj Close", None)
    if ser is None:
        return None
    if isinstance(ser, pd.DataFrame):
        ser = ser.iloc[:, 0]
    r = ser.pct_change(fill_method=None)
    if r is None or r.dropna().empty:
        return None
    return r

def _portfolio_equity_returns_series(weights_by_ticker: Dict[str, float]) -> Optional[pd.Series]:
    """
    Serie diaria de la sleeve de RV del portafolio (RV_MX, RV_EXT, FIBRA_MX),
    ponderada por el peso TOTAL del portafolio (no solo dentro de RV).
    """
    pieces: List[pd.Series] = []
    for tkr, w in weights_by_ticker.items():
        if w <= 0:
            continue
        df_hist = get_hist_5y(tkr)
        r = _daily_pct_from_hist(df_hist)
        if r is None:
            continue
        pieces.append(r * float(w))
    if not pieces:
        return None
    out = None
    for s in pieces:
        out = s if out is None else out.add(s, fill_value=0.0)
    return out

def _beta_from_two_series(r_port: Optional[pd.Series], r_index: Optional[pd.Series]) -> float:
    if r_port is None or r_index is None:
        return float("nan")
    df = pd.concat([r_port, r_index], axis=1, join="inner").dropna()
    if df.shape[0] < 30:
        return float("nan")
    rp = df.iloc[:, 0]
    ri = df.iloc[:, 1]
    var_i = float(np.var(ri, ddof=1)) if ri.size else np.nan
    if not np.isfinite(var_i) or var_i <= 0:
        return float("nan")
    cov = float(np.cov(rp, ri, ddof=1)[0, 1])
    return cov / var_i

def _downside_dev_annual_from_excess_series(excess_daily: Optional[pd.Series]) -> float:
    """
    Desviación a la baja anualizada a partir de una serie diaria de excesos.
    """
    if excess_daily is None or excess_daily.dropna().empty:
        return float("nan")
    neg = excess_daily[excess_daily < 0.0]
    if neg.empty:
        return float("nan")
    dd_daily = float(neg.std(ddof=1))
    return dd_daily * np.sqrt(TRADING_DAYS)

def _sortino_vs_rf(r_port: Optional[pd.Series], rf_ann: float) -> float:
    """
    Sortino clásico vs tasa libre de riesgo.
      Numerador: mu_port_a - rf_ann
      Denominador: downside deviation anual de (r_port_daily - rf_daily)
    """
    if r_port is None or r_port.dropna().shape[0] < 30:
        return float("nan")
    # mu anual del portafolio (sleeve de RV ponderada por peso total)
    mu_p_a = float(r_port.mean()) * TRADING_DAYS
    # aproximación de tasa diaria libre de riesgo (compuesto)
    try:
        rf_daily = (1.0 + float(rf_ann)) ** (1.0 / TRADING_DAYS) - 1.0
    except Exception:
        rf_daily = float(rf_ann) / TRADING_DAYS
    excess_daily = r_port - rf_daily
    dd_a = _downside_dev_annual_from_excess_series(excess_daily)
    if not np.isfinite(dd_a) or dd_a <= 0:
        return float("nan")
    return (mu_p_a - float(rf_ann)) / dd_a

def _beta_portfolio_vs_ipc_using_selection_and_weights(df_with_weights: pd.DataFrame) -> Tuple[float, Optional[pd.Series], Optional[pd.Series]]:
    """
    Calcula beta vs IPC y devuelve (beta, r_port_equity, r_ipc) para otras métricas.
    """
    sel = st.session_state.get("equity_selection", {}) or {}
    if not sel:
        return float("nan"), None, None

    # nombre -> peso total en df
    pesos_por_nombre: Dict[str, float] = {}
    for _, row in df_with_weights.iterrows():
        pesos_por_nombre.setdefault(row["nombre"], 0.0)
        pesos_por_nombre[row["nombre"]] += float(row["w"])

    # ticker -> peso (solo RV_MX, RV_EXT, FIBRA_MX)
    weights_by_ticker: Dict[str, float] = {}
    for ticker, meta in sel.items():
        nm = str(meta.get("name", ticker))
        mask = (df_with_weights["nombre"].eq(nm) &
                df_with_weights["clase"].isin(["RV_MX", "RV_EXT", "FIBRA_MX"]))
        if mask.any():
            w = float(df_with_weights.loc[mask, "w"].sum())
            if w > 0:
                weights_by_ticker[ticker] = w
    if not weights_by_ticker:
        return float("nan"), None, None

    # Serie del portafolio (equity sleeve)
    r_port = _portfolio_equity_returns_series(weights_by_ticker)

    # Serie del IPC
    df_ipc = get_hist_5y("^MXX")
    if df_ipc is None or df_ipc.empty:
        df_ipc = get_hist_5y("^MEXBOL")
    r_ipc = _daily_pct_from_hist(df_ipc)

    beta = _beta_from_two_series(r_port, r_ipc)
    return beta, r_port, r_ipc


# ===== Otras utilidades =====
def _calculate_max_drawdown(r_daily: pd.Series) -> float:
    if r_daily.empty:
        return 0.0
    cumulative_returns = (1 + r_daily).cumprod()
    cumulative_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / cumulative_max) - 1
    return float(drawdown.min())

def _calculate_beta(df: pd.DataFrame, clases: List[str], sig_annual: np.ndarray, index_class: str = "RV_MX") -> float:
    index_mask = np.array(clases) == index_class
    index_idx = np.where(index_mask)[0]
    if index_idx.size == 0 or df.loc[index_mask, "w"].sum() < 1e-9:
        return 0.0
    sigma_index_a = df.loc[index_mask, "sigma"].mean()
    n = len(sig_annual)
    C = np.eye(n)
    for i in range(n):
        for j in range(i, n):
            r = _rho(clases[i], clases[j])
            C[i, j] = C[j, i] = r
    betas = []
    for i in range(n):
        rho_i_index = C[i, index_idx[0]]
        sigma_i = sig_annual[i]
        beta_i = rho_i_index * (sigma_i / sigma_index_a) if sigma_index_a > 1e-9 else 0.0
        betas.append(beta_i)
    w = df["w"].to_numpy()
    return float(np.sum(w * np.array(betas)))

def _ensure_limits_loaded():
    if (
        "consar_limits_catalog" not in st.session_state
        and "siefore_selected" not in st.session_state
        and "afore_selected" not in st.session_state
    ):
        try:
            consar_limits._init_state()
        except Exception:
            pass

def _caps_from_consar() -> tuple[dict, dict, str, str]:
    _ensure_limits_loaded()
    siefore_name = st.session_state.get("siefore_selected", "Básica 95-99")
    afore_name   = st.session_state.get("afore_selected", "N/A")

    cur = consar_limits.get_current_limits_all(siefore_name=siefore_name) or {}
    classes = cur.get("classes", {})
    risk = cur.get("risk", {})

    def _max_pct(key: str, default: float) -> float:
        node = classes.get(key, {})
        val = node.get("max_pct", default)
        return float(val) / 100.0

    def _max_pct_multi(keys: list[str], default: float) -> float:
        for k in keys:
            if k in classes:
                node = classes.get(k, {})
                val = node.get("max_pct", default)
                return float(val) / 100.0
        norm = lambda s: ''.join(ch for ch in s.lower() if ch.isalnum())
        classes_norm = {norm(k): k for k in classes.keys()}
        for k in keys:
            nk = norm(k)
            if nk in classes_norm:
                node = classes.get(classes_norm[nk], {})
                val = node.get("max_pct", default)
                return float(val) / 100.0
        return default / 100.0

    caps = {
        "RV_TOTAL_MAX": _max_pct("Renta Variable", 60.0),
        "EXT_MAX": _max_pct("Valores Extranjeros (tope)", 20.0),
        "FIBRA_MX_MAX": _max_pct("FIBRAS / REITs", 10.0),
        "VAR_DAILY95_MAX": float(risk.get("VaR (%)", 1.30)) / 100.0,
        "EXT_SINGLE_ISSUER_MAX": 0.05,
        "FX_MAX": _max_pct_multi(["Instrumentos en Divisas","Divisas","FX","Instrumentos en divisas"], 10.0),
        "SN_MAX": _max_pct("Instrumentos Estructurados", 30.0),
        "MERCANCÍAS_MAX": _max_pct("Mercancías", 5.0),
    }
    risk_limits = {
        "VaR": float(risk.get("VaR (%)", 1.30)) / 100.0,
        "CVaR": float(risk.get("CVaR / Valor en Riesgo Condicional (%)", 1.30)) / 100.0,
        "LIQ": float(risk.get("Coeficiente de liquidez (%)", 80.0)) / 100.0,
        "TE": float(risk.get("Error de seguimiento / tracking error (%)", 5.0)) / 100.0,
    }
    return caps, risk_limits, siefore_name, afore_name


    caps = {
        "RV_TOTAL_MAX": _max_pct("Renta Variable", 60.0),
        "EXT_MAX": _max_pct("Valores Extranjeros (tope)", 20.0),
        "FIBRA_MX_MAX": _max_pct("FIBRAS / REITs", 10.0),
        "VAR_DAILY95_MAX": float(risk.get("VaR (%)", 1.30)) / 100.0,
        "EXT_SINGLE_ISSUER_MAX": 0.05,
        "FX_MAX": _max_pct_multi(["Instrumentos en Divisas", "Divisas", "FX", "Instrumentos en divisas"], 10.0),
        "SN_MAX": _max_pct("Instrumentos Estructurados", 30.0),
        "MERCANCÍAS_MAX": _max_pct("Mercancías", 5.0),
    }
    risk_limits = {
        "VaR": float(risk.get("VaR (%)", 1.30)) / 100.0,
        "CVaR": float(risk.get("CVaR / Valor en Riesgo Condicional (%)", 1.30)) / 100.0,
        "LIQ": float(risk.get("Coeficiente de liquidez (%)", 80.0)) / 100.0,
        "TE": float(risk.get("Error de seguimiento / tracking error (%)", 5.0)) / 100.0,
    }
    return caps, risk_limits, siefore_name, afore_name

def _caps_operativos(caps: Dict[str, float], buffer: float = OP_BUFFER) -> Dict[str, float]:
    m = 1.0 - float(buffer)
    keys_to_buffer = ["RV_TOTAL_MAX", "EXT_MAX", "FIBRA_MX_MAX", "FX_MAX", "EXT_SINGLE_ISSUER_MAX"]
    out = caps.copy()
    for k in keys_to_buffer:
        if k in out and out[k] is not None:
            try:
                out[k] = max(0.0, float(out[k]) * m)
            except Exception:
                pass
    return out


# ===== Datos de activos (sin cambios) =====
@dataclass
class AssetRow:
    nombre: str
    clase: str   # RV_MX, RV_EXT, FIBRA_MX, FI_GOB_MX, FI_CORP_MX, FI_EXT, FX, SN
    mu: float    # anual (tanto)
    sigma: float # anual (tanto)
    limite_txt: str

def _mu_sigma_equity_5y(ticker: str) -> Tuple[float, float]:
    df5 = get_hist_5y(ticker)
    if df5 is None or df5.empty or "Adj Close" not in df5.columns:
        return 0.0, 0.25
    ser = df5["Adj Close"]
    if isinstance(ser, pd.DataFrame):
        ser = ser.iloc[:, 0]
    r = ser.pct_change(fill_method=None)
    mu = _avg_annual_return(r)
    sigma = _annual_vol(r)
    if not np.isfinite(mu):
        mu = 0.0
    if (not np.isfinite(sigma)) or sigma <= 0:
        sigma = 0.25
    return float(mu), float(sigma)

def _clase_equity(index_name: str, ticker: str, nombre: str) -> Tuple[str, str]:
    nm = (nombre or "").lower()
    if ("fibra" in nm) or ("fibr" in nm):
        return "FIBRA_MX", "FIBRAS ≤ tope CONSAR"
    country = _country_from_index(index_name or "", ticker or "")
    if country in {"USA","DEU","FRA","GBR","ESP","ITA","CHE","DNK","NOR","NLD","JPN","CHN","KOR","IND","TWN","SGP","AUS","ARG","BRA","CAN"}:
        return "RV_EXT", "RV ≤ tope CONSAR, Extranjeros ≤ tope CONSAR"
    return "RV_MX", "RV ≤ tope CONSAR"

def _cosechar_activos() -> List[AssetRow]:
    rows: List[AssetRow] = []
    if st.session_state.get("fx_picks") is None:
        st.session_state["fx_picks"] = []
    if st.session_state.get("fx_basket") is None:
        st.session_state["fx_basket"] = []
    try:
        if st.session_state["fx_picks"] and not st.session_state["fx_basket"]:
            if hasattr(fx, "_init_fx_state_from_picks"):
                fx._init_fx_state_from_picks()
    except Exception:
        pass

    sel = st.session_state.get("equity_selection", {}) or {}
    try:
        preload_hist_5y_daily(list(sel.keys()))
    except Exception:
        pass
    try:
        preload_meta(list(sel.keys()))
    except Exception:
        pass

    for tkr, meta in sel.items():
        nm = str(meta.get("name", tkr))
        idx = str(meta.get("index", ""))
        mu, sig = _mu_sigma_equity_5y(tkr)
        clase, limtxt = _clase_equity(idx, tkr, nm)
        rows.append(AssetRow(nm, clase, mu, sig, limtxt))

    fi_rows = st.session_state.get("fi_rows", []) or []
    for r in fi_rows:
        nm = str(r.get("Bono"))
        tipo = str(r.get("Tipo de bono"))
        pais = str(r.get("País"))
        ytm_pct = float(r.get("Rend (%)", 0.0))
        mu = ytm_pct / 100.0
        if (tipo == "Bono gubernamental") and (pais == "México"):
            clase = "FI_GOB_MX"; txt = "—"
        else:
            if pais == "México":
                clase = "FI_CORP_MX"; txt = "—"
            else:
                clase = "FI_EXT"; txt = "Extranjeros ≤ tope CONSAR"
        sigma = _sigma_bond_from_row(r)
        rows.append(AssetRow(nm, clase, mu, sigma, txt))

    try:
        fx._ensure_fx_selection()
    except Exception as e:
        st.error(f"FX SYNC FAILED: {e}")

    fx_sel = st.session_state.get("fx_selection", {}) or {}
    for pair_label, meta in fx_sel.items():
        nm = str(meta.get("name", pair_label))
        mu = float(meta.get("mu", 0.0))
        sig = float(meta.get("sigma", 0.10))
        rows.append(AssetRow(nm, "FX", mu, sig, "Divisas ≤ tope CONSAR"))

    sn_list = st.session_state.get("structured_notes", []) or []
    for n in sn_list:
        try:
            nm = str(n.get("subyacente", "Nota"))
            mu = float(n.get("mu_ann_pct", 0.0)) / 100.0
            sig = float(n.get("vol_ann_pct", 0.0)) / 100.0
            if not np.isfinite(mu): mu = 0.0
            if not np.isfinite(sig) or sig <= 0: sig = 0.05
            rows.append(AssetRow(nm, "SN", mu, sig, "Estructurados ≤ tope CONSAR"))
        except Exception:
            continue

    return rows


# ===== Modelo de volatilidad de bonos (sin cambios) =====
_US_ORDER = ["AAA","AA+","AA","AA-","A+","A","A-","BBB+","BBB","BBB-","BB+","BB","BB-","B+","B","B-","CCC","CC","C","D"]
_MX_ORDER = ["mxAAA","mxAA+","mxAA","mxAA-","mxA+","mxA","mxA-","mxBBB+","mxBBB","mxBBB-","mxBB+","mxBB","mxBB-","mxB+","mxB","mxB-","mxCCC","mxCC","mxC","D"]

def _parse_date_any(s: str | date | datetime | None) -> Optional[date]:
    if s is None: return None
    if isinstance(s, date) and not isinstance(s, datetime): return s
    if isinstance(s, datetime): return s.date()
    s = str(s)
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y"):
        try: return datetime.strptime(s, fmt).date()
        except Exception: pass
    return None

def _years_between(a: Optional[date], b: Optional[date]) -> float:
    if not a or not b: return 1.0
    return max(0.05, (b - a).days / 365.0)

def _rating_multiplier(rating: str, pais: str, tipo: str) -> float:
    r = (rating or "").strip()
    if pais == "México" and r.lower().startswith("mx"):
        order = _MX_ORDER
        multipliers = [1.00,1.05,1.08,1.10, 1.12,1.15,1.18, 1.25,1.35,1.45,1.60,1.80,2.00, 2.30,2.60,3.00,3.50,4.00,5.00,6.00]
    else:
        order = _US_ORDER
        multipliers = [1.00,1.04,1.06,1.08, 1.10,1.12,1.15, 1.25,1.35,1.45,1.65,1.85,2.10, 2.40,2.80,3.20,3.80,4.50,5.50,6.50]
    try: idx = order.index(r)
    except ValueError: idx = 5
    mult = multipliers[idx]
    if tipo == "Bono gubernamental":
        mult = 1.0 + (mult - 1.0) * 0.5
    return float(mult)

def _base_sigma_by_country_type(pais: str, tipo: str) -> float:
    if pais == "México" and tipo == "Bono gubernamental": return 0.025
    if pais == "México" and tipo != "Bono gubernamental": return 0.05
    if pais == "EE.UU." and tipo == "Bono gubernamental": return 0.012
    if pais == "EE.UU." and tipo != "Bono gubernamental": return 0.03
    return 0.04

def _tenor_multiplier(years: float) -> float:
    return float(np.clip(np.sqrt(max(years, 0.05)), 0.5, 3.5))

def _sigma_bond_from_row(row: Dict) -> float:
    pais = str(row.get("País","")).strip() or "México"
    tipo = str(row.get("Tipo de bono","")).strip() or "Bono corporativo"
    rating = str(row.get("Fitch","")).strip()
    liq = _parse_date_any(row.get("Liquidación"))
    mat = _parse_date_any(row.get("Maturity"))
    yrs = _years_between(liq, mat)
    base = _base_sigma_by_country_type(pais, tipo)
    mult_tenor = _tenor_multiplier(yrs)
    mult_rating = _rating_multiplier(rating, pais, tipo)
    sigma = base * mult_tenor * mult_rating
    return float(np.clip(sigma, 0.005, 0.25))


# ===== Correlaciones/covarianzas y métricas de riesgo (sin cambios) =====
CORR_BLOCKS = { "RV_MX": 0.35, "RV_EXT": 0.45, "FIBRA_MX": 0.40, "FI_GOB_MX": 0.05, "FI_CORP_MX": 0.20, "FI_EXT": 0.25, "FX": 0.20, "SN": 0.08 }
CORR_BETWEEN = {
    ("RV_MX","RV_EXT"): 0.40, ("RV_MX","FIBRA_MX"): 0.45, ("RV_EXT","FIBRA_MX"): 0.50,
    ("RV_MX","FI_GOB_MX"): 0.05, ("RV_EXT","FI_GOB_MX"): 0.05, ("FIBRA_MX","FI_GOB_MX"): 0.05,
    ("RV_MX","FI_CORP_MX"): 0.15, ("RV_EXT","FI_CORP_MX"): 0.20, ("FIBRA_MX","FI_CORP_MX"): 0.20,
    ("FI_EXT","RV_MX"): 0.20, ("FI_EXT","RV_EXT"): 0.25, ("FI_EXT","FIBRA_MX"): 0.20,
    ("FI_EXT","FI_GOB_MX"): 0.10, ("FI_EXT","FI_CORP_MX"): 0.25,
    ("SN","FI_GOB_MX"): 0.05, ("SN","FI_CORP_MX"): 0.10, ("SN","FI_EXT"): 0.10,
    ("SN","RV_MX"): 0.10, ("SN","RV_EXT"): 0.12, ("SN","FIBRA_MX"): 0.12, ("SN","FX"): 0.05,
}
def _rho(a: str, b: str) -> float:
    if a == b: return CORR_BLOCKS.get(a, 0.2)
    return CORR_BETWEEN.get((a,b), CORR_BETWEEN.get((b,a), 0.10))

def _cov_from_sigmas(sig_annual: np.ndarray, clases: List[str], *, daily: bool) -> np.ndarray:
    sig = sig_annual / (np.sqrt(TRADING_DAYS) if daily else 1.0)
    n = len(sig)
    C = np.eye(n)
    for i in range(n):
        for j in range(i, n):
            r = _rho(clases[i], clases[j])
            C[i, j] = C[j, i] = r
    D = np.diag(sig)
    return D @ C @ D

def _var95_daily(w: np.ndarray, sig_annual: np.ndarray, clases: List[str]) -> float:
    cov_d = _cov_from_sigmas(sig_annual, clases, daily=True)
    sigma_p_d = float(np.sqrt(w @ cov_d @ w))
    return Z_95 * sigma_p_d

# CORRECCIÓN DE TE: Recibe rv_limit y usa ese valor para el benchmark
def _te_annual(w: np.ndarray, clases: List[str], sig_annual: np.ndarray, rv_limit: float) -> float:
    cov_a = _cov_from_sigmas(sig_annual, clases, daily=False)
    w_bench = np.zeros_like(w)
    BENCH_RV_PROP = rv_limit
    BENCH_FI_PROP = 1.0 - BENCH_RV_PROP

    rv_classes = {"RV_MX", "RV_EXT", "FIBRA_MX"}
    rv_mask = np.isin(clases, list(rv_classes))
    rv_idx = np.where(rv_mask)[0]
    w_rv_active = w[rv_idx].sum()
    if w_rv_active > 1e-9:
        rv_prop = w[rv_idx] / w_rv_active
        w_bench[rv_idx] = BENCH_RV_PROP * rv_prop

    fi_gov_class = "FI_GOB_MX"
    gov_idx = np.where(np.array(clases) == fi_gov_class)[0]
    if gov_idx.size:
        w_bench[gov_idx] += BENCH_FI_PROP
    else:
        fi_mask = ~rv_mask
        fi_idx = np.where(fi_mask)[0]
        w_fi_active_total = w[fi_idx].sum()
        if w_fi_active_total > 1e-9:
            fi_prop = w[fi_idx] / w_fi_active_total
            w_bench[fi_idx] += BENCH_FI_PROP * fi_prop

    w_bench = _normalize(w_bench)
    active = w - w_bench
    return float(np.sqrt(active @ cov_a @ active))


# ===== Helpers de pesos y optimizador principal (sin cambios) =====
def _normalize(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else w

def _initial_by_sharpe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    shp = (df["mu"] / df["sigma"].replace(0, np.nan)).fillna(0.0)
    shp = np.maximum(shp, 0.0)
    df["w"] = (shp / shp.sum()) if shp.sum() > 0 else 0.0
    return df

def _realloc(df: pd.DataFrame, from_mask: np.ndarray, to_mask: np.ndarray, amount: float) -> None:
    if amount <= 0: return
    w = df["w"].to_numpy()
    take_base = w[from_mask].sum()
    if take_base <= 1e-12: return
    w[from_mask] *= (1 - amount / take_base)
    add_base = w[to_mask].sum()
    if add_base <= 1e-12:
        add_weights = np.zeros_like(w)
        add_weights[to_mask] = 1.0 / max(to_mask.sum(), 1)
        w += add_weights * amount
    else:
        w[to_mask] += w[to_mask] / add_base * amount
    df["w"] = _normalize(w)

def _apply_caps(df: pd.DataFrame, caps: Dict[str, float]) -> None:
    df["w"] = _normalize(df["w"].to_numpy())

    # FIBRAS
    m_fibras = df["clase"].eq("FIBRA_MX").to_numpy()
    exceso = max(0.0, df.loc[m_fibras, "w"].sum() - caps["FIBRA_MX_MAX"])
    if exceso > 1e-9:
        _realloc(df, m_fibras, df["clase"].eq("FI_GOB_MX").to_numpy(), exceso)

    # RV total
    m_rv_total = df["clase"].isin(["RV_MX","RV_EXT","FIBRA_MX"]).to_numpy()
    exceso = max(0.0, df.loc[m_rv_total, "w"].sum() - caps["RV_TOTAL_MAX"])
    if exceso > 1e-9:
        _realloc(df, m_rv_total, df["clase"].eq("FI_GOB_MX").to_numpy(), exceso)

    # Extranjeros
    m_ext = df["clase"].isin(["RV_EXT","FI_EXT"]).to_numpy()
    exceso = max(0.0, df.loc[m_ext, "w"].sum() - caps["EXT_MAX"])
    if exceso > 1e-9:
        if df["clase"].eq("RV_MX").any():
            _realloc(df, m_ext, df["clase"].eq("RV_MX").to_numpy(), exceso)
        else:
            _realloc(df, m_ext, df["clase"].eq("FI_GOB_MX").to_numpy(), exceso)

    # Tope por activo extranjero
    fallback_mask = df["clase"].isin(["RV_MX","FI_GOB_MX"]).to_numpy()
    for i, row in df.reset_index().iterrows():
        if row["clase"] in ("RV_EXT","FI_EXT"):
            cap = caps.get("EXT_SINGLE_ISSUER_MAX", 0.05)
            w = df["w"].to_numpy()
            if w[i] > cap + 1e-12:
                exceso = w[i] - cap
                w[i] = cap
                add_base = w[fallback_mask].sum()
                if add_base <= 1e-12:
                    add_weights = np.zeros_like(w)
                    add_weights[fallback_mask] = 1.0 / max(fallback_mask.sum(), 1)
                    w += add_weights * exceso
                else:
                    w[fallback_mask] += w[fallback_mask] / add_base * exceso
                df["w"] = _normalize(w)

    # FX
    if "FX_MAX" in caps:
        m_fx = df["clase"].eq("FX").to_numpy()
        exceso = max(0.0, df.loc[m_fx, "w"].sum() - caps["FX_MAX"])
        if exceso > 1e-9:
            if df["clase"].eq("FI_GOB_MX").any():
                destino = df["clase"].eq("FI_GOB_MX").to_numpy()
            elif df["clase"].eq("RV_MX").any():
                destino = df["clase"].eq("RV_MX").to_numpy()
            else:
                destino = ~m_fx
            _realloc(df, m_fx, destino, exceso)

    # SN
    if "SN_MAX" in caps:
        m_sn = df["clase"].eq("SN").to_numpy()
        exceso = max(0.0, df.loc[m_sn, "w"].sum() - caps["SN_MAX"])
        if exceso > 1e-9:
            if df["clase"].eq("FI_GOB_MX").any():
                destino = df["clase"].eq("FI_GOB_MX").to_numpy()
            else:
                destino = ~m_sn
            _realloc(df, m_sn, destino, exceso)

    df["w"] = _normalize(df["w"].to_numpy())

def _te_annual_from_cov(w: np.ndarray, clases: List[str], sig_annual: np.ndarray, rv_limit: float) -> float:
    return _te_annual(w, clases, sig_annual, rv_limit)

def _enforce_all_constraints(df: pd.DataFrame, caps: Dict[str, float], rlim: Dict[str, float]) -> None:
    df["w"] = _normalize(df["w"].to_numpy())
    order = ["RV_EXT", "RV_MX", "FIBRA_MX", "FX", "FI_CORP_MX", "FI_EXT"]
    rv_limit = caps.get("RV_TOTAL_MAX", 0.60)
    for _ in range(300):
        _apply_caps(df, caps)
        clases = df["clase"].tolist()
        sig_a = df["sigma"].to_numpy()
        w = df["w"].to_numpy()
        var_d = _var95_daily(w, sig_a, clases)
        cvar_d = CVaR_OVER_VaR_95 * var_d
        liq_core = float(df.loc[df["clase"].isin(["RV_MX","RV_EXT","FI_GOB_MX", "SN"]), "w"].sum())
        liq_ext  = float(df.loc[df["clase"].eq("FI_EXT"), "w"].sum())
        liq = liq_core + 0.60 * liq_ext
        te = _te_annual(w, clases, sig_a, rv_limit=rv_limit)
        ok_var = var_d <= rlim["VaR"] + 1e-9
        ok_cvar = cvar_d <= rlim["CVaR"] + 1e-9
        ok_liq = liq >= rlim["LIQ"] - 1e-9
        ok_te = te <= rlim["TE"] + 1e-9
        if ok_var and ok_cvar and ok_liq and ok_te:
            break
        step = 0.01
        if (not ok_var) or (not ok_cvar) or (not ok_te):
            moved = False
            for cl in order:
                mask_from = df["clase"].eq(cl).to_numpy()
                if df.loc[mask_from, "w"].sum() > 1e-6:
                    _realloc(df, mask_from, df["clase"].eq("FI_GOB_MX").to_numpy(), step)
                    moved = True
                    break
            if moved:
                continue
        if not ok_liq:
            for source in ["FI_CORP_MX", "FIBRA_MX", "RV_EXT", "RV_MX", "FX", "FI_EXT", "SN"]:
                mask_from = df["clase"].eq(source).to_numpy()
                if df.loc[mask_from, "w"].sum() > 1e-6:
                    dest = df["clase"].isin(["RV_MX","RV_EXT","FI_GOB_MX"]).to_numpy()
                    _realloc(df, mask_from, dest, step)
                    break
    df["w"] = _normalize(df["w"].to_numpy())


# tabs/optimization.py

# ... (código previo)

# ===== Optimizador público (COMPLETO Y MODIFICADO) =====
def _optimizar(rows: List[AssetRow], caps: Dict[str, float], rlim: Dict[str, float],
               stable_base_weights: Optional[Dict[str, float]] = None,
               base_asset_names: Optional[Set[str]] = None
               ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, float]]:

    if not rows:
        return pd.DataFrame(), pd.DataFrame(), {}, {}

    df = pd.DataFrame([{"nombre": r.nombre, "clase": r.clase, "mu": r.mu, "sigma": r.sigma, "lim_txt": r.limite_txt} for r in rows])

    # ------------------- 💡 PARCHE 2 (OPTIMIZACIÓN): USAR PESOS ACTUALES SI ESTÁ OPERANDO 💡 -------------------
    current_asset_names = set(df["nombre"].tolist())
    
    if st.session_state.get("ops_operating", False):
        current_weights = st.session_state.get("current_ops_weights", {})
        current_ops_asset_names = st.session_state.get("current_ops_asset_names", set())

        # Verifica si los activos de la optimización son los mismos que los activos operando
        if current_ops_asset_names and current_asset_names == current_ops_asset_names and current_weights:
            
            # Asigna los pesos actuales (en decimal, guardados desde operations.py)
            df["w"] = df["nombre"].apply(lambda n: current_weights.get(n, 0.0))
            df["w"] = _normalize(df["w"].to_numpy())
            
            # Si, por alguna razón, todos los pesos fueran 0 o la suma no es ~1, recurre a la optimización normal.
            if df["w"].sum() < 0.99 or (df["w"] == 0).all():
                df = _initial_by_sharpe(df)
            
            # Omitimos el resto del bloque de inicialización de pesos y pasamos a aplicar restricciones
        
        else:
             # Si el usuario cambió los activos mientras estaba 'operando', recurre a optimización normal.
             df = _initial_by_sharpe(df)
             
    # ------------------- CONTINUACIÓN DEL CÓDIGO ORIGINAL (MODO DISEÑO) -------------------
    else: # Modo Diseño: Aplica lógica de base estable o Sharpe inicial
        use_stable_base = False
        if stable_base_weights and base_asset_names:
            if current_asset_names == base_asset_names:
                use_stable_base = True

        if use_stable_base:
            df["w"] = df["nombre"].apply(lambda n: stable_base_weights.get(n, 0.0))
            df["w"] = _normalize(df["w"].to_numpy())
            if df["w"].sum() < 0.99 or (df["w"] == 0).all():
                 df = _initial_by_sharpe(df)
        else:
            df = _initial_by_sharpe(df)

    # Semilla FX si quedó en 0
    m_fx = df["clase"].eq("FX").to_numpy()
    if m_fx.any() and df.loc[m_fx, "w"].sum() < 1e-9:
        fx_cap = caps.get("FX_MAX", 0.10)
        seed = min(0.005, max(0.0, fx_cap - 0.0))
        if seed > 0:
            from_mask = df["clase"].eq("FI_GOB_MX").to_numpy()
            if df.loc[from_mask, "w"].sum() < seed:
                from_mask = ~m_fx
            _realloc(df, from_mask, m_fx, seed)

    _enforce_all_constraints(df, caps, rlim)

    # ===== Salidas de composición =====
    out = df.copy()
    out["Esperado (%)"] = (out["mu"] * 100).round(2)
    out["Vol (%)"] = (out["sigma"] * 100).round(2)
    out["Sharpe (aprox)"] = (out["mu"] / out["sigma"]).replace([np.inf,-np.inf], np.nan).fillna(0.0).round(2)
    out["Peso (%)"] = (out["w"] * 100).round(2)
    out["Límite aplicable"] = out["lim_txt"]
    out = out[["nombre","clase","Esperado (%)","Vol (%)","Sharpe (aprox)","Peso (%)","Límite aplicable"]]

    tot = df.groupby("clase")["w"].sum().reset_index()
    tot["Peso (%)"] = (tot["w"] * 100).round(2)
    tot = tot[["clase","Peso (%)"]]

    clases = df["clase"].tolist()
    sig_a = df["sigma"].to_numpy()
    w = df["w"].to_numpy()

    # ===== Métricas de riesgo (no afectan optimización) =====
    var_d = _var95_daily(w, sig_a, clases)
    cvar_d = CVaR_OVER_VaR_95 * var_d
    liq_core = float(df.loc[df["clase"].isin(["RV_MX","RV_EXT","FI_GOB_MX", "SN"]), "w"].sum())
    liq_ext  = float(df.loc[df["clase"].eq("FI_EXT"), "w"].sum())
    liq = liq_core + 0.60 * liq_ext
    te = _te_annual(w, clases, sig_a, rv_limit=caps.get("RV_TOTAL_MAX", 0.60))

    cov_a = _cov_from_sigmas(sig_a, clases, daily=False)
    sigma_p_a = float(np.sqrt(w @ cov_a @ w))

    mu_port = float((df["w"] * df["mu"]).sum())

    # Sharpe con CETES 28d (Banxico) como R_f
    rf = _get_risk_free_ann()
    sharpe_ratio = (mu_port - rf) / sigma_p_a if sigma_p_a > 1e-9 else 0.0

    # (1) Proxy anterior (referencia)
    beta_proxy_rv_mx = _calculate_beta(df, clases, sig_a, index_class="RV_MX")

    # (2) Beta vs IPC (visualización) + series de RV para Sortino clásico contra Rf
    beta_vs_ipc, r_port_equity, _r_ipc_unused = _beta_portfolio_vs_ipc_using_selection_and_weights(df)

    # ===== NUEVO: Sortino clásico vs Rf (CETES 28d) =====
    sortino_rf = _sortino_vs_rf(r_port_equity, rf)

    # MaxDD simulado (por falta de serie de todo el portafolio)
    mu_daily = mu_port / TRADING_DAYS
    sigma_daily = sigma_p_a / np.sqrt(TRADING_DAYS)
    rng = np.random.default_rng(20251015)
    r_daily_sim = pd.Series(rng.normal(loc=mu_daily, scale=sigma_daily, size=TRADING_DAYS * 3))
    max_dd = _calculate_max_drawdown(r_daily_sim)

    risk_metrics = {
        "mu_port": mu_port,
        "var_daily": var_d,
        "sigma_annual": sigma_p_a,
        "sharpe_ratio": sharpe_ratio,
        "risk_free_rate_used": rf,                      # para mostrar qué Rf se usó
        "risk_free_rate_source": st.session_state.get("risk_free_rate_source", ""),
        "max_drawdown": max_dd,
        "beta_portfolio": beta_vs_ipc,                  # Beta vs IPC (^MXX / ^MEXBOL)
        "beta_vs_rv_mx_proxy": beta_proxy_rv_mx,        # referencia opcional
        "te_annual": te,
        "cvar_daily": cvar_d,
        "liq_coef": liq,
        "sortino_rf": sortino_rf,                       # <-- AHORA: Sortino contra Rf (CETES 28d)
    }

    final_weights = dict(zip(df["nombre"], df["w"].tolist()))
    return out, tot, risk_metrics, final_weights

