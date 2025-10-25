# tabs/aportaciones.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Tuple, Optional, Any

import math
import numpy as np
import pandas as pd
import streamlit as st
# (opcional) from streamlit_searchbox import st_searchbox

from tabs.yf_store import get_hist_5y

# ======================================================================
# Config y helpers
# ======================================================================

GROWTH_CLASSES = {"RV_MX", "RV_EXT", "FIBRA_MX"}
DEFENSIVE_CLASSES = {"FI_GOB_MX", "FI_EXT"}
FX_CLASSES = {"FX"}
LEVEL0_CLASSES = {"FI_CORP_MX", "Nota estructurada"}  # sin señal granular

_DISPLAY_NAME = {
    "FI_GOB_MX": "Deuda Gubernamental",
    "FI_CORP_MX": "Deuda Privada Nacional",
    "FI_EXT": "Deuda Internacional",
    "RV_MX": "Renta Variable Nacional",
    "RV_EXT": "Renta Variable Internacional",
    "FIBRA_MX": "FIBRAS",
    "FX": "Divisas",
    "Nota estructurada": "Notas Estructuradas",
    "SN": "Notas Estructuradas",
}

_REVERSE_DISPLAY_NAME = {
    "deuda gubernamental": "FI_GOB_MX",
    "deuda privada nacional": "FI_CORP_MX",
    "deuda internacional": "FI_EXT",
    "renta variable nacional": "RV_MX",
    "renta variable internacional": "RV_EXT",
    "fibras": "FIBRA_MX",
    "fibras (reits)": "FIBRA_MX",  # Alias por si acaso
    "divisas": "FX",
    "notas estructuradas": "Nota estructurada",
    "sn": "Nota estructurada",
}

def _norm_key(s: str) -> str:
    s = str(s or "").strip().lower()
    for a,b in (("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ñ","n")):
        s = s.replace(a,b)
    return " ".join(s.split())

def _canon_class(x: Optional[str]) -> Optional[str]:
    """Convierte un nombre (de display o alias) a su nombre canónico."""
    if x is None:
        return None
    s = str(x).strip()
    k = _norm_key(s)
    return _REVERSE_DISPLAY_NAME.get(k, s)

def _display_name(code: str) -> str:
    return _DISPLAY_NAME.get(code, str(code))

CLASS_PROXY = {
    "RV_MX": "^MXX",
    "RV_EXT": "^GSPC",
    "FIBRA_MX": "^MXX",
    "FI_GOB_MX": None,
    "FI_EXT": None,
    "FX": "MXN=X",
}

def _today() -> date:
    return datetime.now().date()

def _safe_adj_close_series(df: Optional[pd.DataFrame]) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    ser = df.get("Adj Close", None)
    if ser is None:
        return pd.Series(dtype=float)
    if isinstance(ser, pd.DataFrame):
        ser = ser.iloc[:, 0]
    try:
        return ser.astype(float).dropna()
    except Exception:
        return pd.Series(dtype=float)

def _pct(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def _normalize_weights(d: Dict[str, float]) -> Dict[str, float]:
    # Preliminar: sanitizar valores
    clean = {k: max(0.0, _pct(val)) for k, val in d.items()}
    s = float(sum(clean.values()))
    if s <= 0:
        return {k: 0.0 for k in clean.keys()}
    # Normalizar
    return {k: val / s for k, val in clean.items()}


# ======================================================================
# Lotes y buffer de comisiones (para compras con enteros)
# ======================================================================

# Tamaño de lote por clase (None = permitir fraccional)
LOT_SIZE_BY_CLASS = {
    "RV_MX": 1,
    "RV_EXT": 1,
    "FIBRA_MX": 1,
    "FI_GOB_MX": 1,
    "FI_CORP_MX": 1,
    "FI_EXT": 1,
    "Nota estructurada": 1,
    "SN": 1,  # alias por si llegara como SN
    "FX": None,  # divisas se dejan fraccionales por defecto
}

def _commission_buffer_pct_ss(ss) -> float:
    """Porcentaje a reservar para comisiones (default 0.10%, cap 5%)."""
    try:
        val = float(ss.get("commission_buffer_pct", 0.001))
        return max(0.0, min(0.05, val))
    except Exception:
        return 0.001

def _lot_size_for(ss, symbol: str, clase_display_or_code: str) -> Optional[int]:
    """Determina tamaño de lote por símbolo o por clase (override > clase > default)."""
    try:
        # override por símbolo
        ov_sym = (ss.get("lot_size_overrides", {}) or {}).get(symbol)
        if ov_sym is not None:
            return int(ov_sym) if ov_sym is not None else None

        # override por clase canónica
        cls_code = _canon_class(clase_display_or_code)
        ov_cls = (ss.get("lot_size_overrides_by_class", {}) or {}).get(cls_code)
        if ov_cls is not None:
            return int(ov_cls) if ov_cls is not None else None

        # default
        return LOT_SIZE_BY_CLASS.get(cls_code, 1)
    except Exception:
        return 1

# ======================================================================
# Reconstrucción del portafolio hasta una fecha (solo por clase)
# ======================================================================

@dataclass
class Snapshot:
    effective_date: date
    total_value: float
    value_by_class: Dict[str, float]
    weight_by_class: Dict[str, float]
    class_list: List[str]

def _pick_total_col(ts_df: pd.DataFrame) -> Optional[str]:
    """Detecta la columna de valor total en el timeseries."""
    if ts_df is None or ts_df.empty:
        return None
    candidates = ["total_value", "valor_portafolio", "total_portfolio_value", "V_t", "valor_total"]
    for c in candidates:
        if c in ts_df.columns:
            return c
    # fallback: primera columna numérica
    for c in ts_df.columns:
        try:
            if pd.to_numeric(ts_df[c], errors="coerce").notna().any():
                return c
        except Exception:
            continue
    return None

def _rebuild_until_date(
    start_dt: date,
    end_dt: date,
    seed_capital_pesos: float,
    fee_pp: float,
    df_opt: pd.DataFrame,
    fi_rows: List[Dict],
    eq_sel: Dict[str, Dict],
    fx_sel: Dict[str, Dict],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Si necesitas reconstruir, usa build_initial_portfolio desde tabs.ledger.
    OJO: en la práctica tomamos positions/timeseries ya existentes desde session_state,
    así que esta función suele no ser llamada. Se deja por compatibilidad.
    """
    from tabs.ledger import build_initial_portfolio as _build_init

    pos_df, ts_df_full = _build_init(
        start_dt=start_dt,
        seed_capital_pesos=seed_capital_pesos,
        fee_pp=fee_pp,
        df_opt=df_opt,
        fi_rows=fi_rows,
        eq_sel=eq_sel,
        fx_sel=fx_sel,
        end_dt=end_dt,
    )
    if ts_df_full is None or ts_df_full.empty:
        return pos_df, ts_df_full

    end_ts = pd.Timestamp(end_dt)
    last_available = ts_df_full.index.max()
    cut = ts_df_full.copy().loc[: end_ts] if end_ts <= last_available else ts_df_full.copy()
    if cut.empty:
        if not ts_df_full.empty:
            first_idx = ts_df_full.index.min()
            cut = ts_df_full.loc[[first_idx]]
        else:
            return pos_df, pd.DataFrame()
    return pos_df, cut

def _aggregate_by_class(pos_df_on_start: pd.DataFrame, ts_df_until_t: pd.DataFrame) -> Snapshot:
    if ts_df_until_t is None or ts_df_until_t.empty:
        return Snapshot(effective_date=_today(), total_value=0.0, value_by_class={}, weight_by_class={}, class_list=[])

    t_idx = ts_df_until_t.index.max()
    col_total = _pick_total_col(ts_df_until_t)
    if not col_total:
        return Snapshot(effective_date=_today(), total_value=0.0, value_by_class={}, weight_by_class={}, class_list=[])

    V_t = float(pd.to_numeric(ts_df_until_t.loc[t_idx, col_total], errors="coerce"))

    if pos_df_on_start is None or pos_df_on_start.empty:
        return Snapshot(effective_date=t_idx.date(), total_value=V_t, value_by_class={}, weight_by_class={}, class_list=[])

    base = pos_df_on_start.copy()
    if "costo_ini" in base.columns:
        base["_base_val"] = pd.to_numeric(base["costo_ini"], errors="coerce").fillna(0.0)
    else:
        base["_base_val"] = pd.to_numeric(base.get("valor_actual", 0.0), errors="coerce").fillna(0.0)

    base["clase_canon"] = base["clase"].map(_canon_class)

    byc = base.groupby("clase_canon")["_base_val"].sum().to_dict()
    s = sum(byc.values()) or 1.0
    prop_init = {k: (v / s) for k, v in byc.items()}
    value_by_class = {k: float(prop_init[k] * V_t) for k in prop_init.keys()}

    total = sum(value_by_class.values()) or 1.0
    weight_by_class = {k: value_by_class[k] / total for k in value_by_class.keys()}
    class_list = sorted(value_by_class.keys())

    return Snapshot(
        effective_date=t_idx.date(),
        total_value=V_t,
        value_by_class=value_by_class,
        weight_by_class=weight_by_class,
        class_list=class_list
    )

def _class_targets_from_opt(df_opt: pd.DataFrame) -> Dict[str, float]:
    if df_opt is None or df_opt.empty or "clase" not in df_opt.columns or "Peso (%)" not in df_opt.columns:
        return {}
    tmp = df_opt.copy()
    tmp["clase_canon"] = tmp["clase"].map(_canon_class)
    g = tmp.groupby("clase_canon")["Peso (%)"].sum().to_dict()
    g01 = {k: max(0.0, float(v) / 100.0) for k, v in g.items()}
    return _normalize_weights(g01)

def _compute_proxy_sharpe(ticker: str, end_dt: date, lookback_days: int = 60) -> Optional[float]:
    if not ticker:
        return None
    end = pd.Timestamp(end_dt)
    df = get_hist_5y(ticker)
    s = _safe_adj_close_series(df)
    if s.empty:
        return None
    s = s.loc[:end].iloc[-lookback_days-2:]
    if s.size < 10:
        return None
    rets = s.pct_change().dropna()
    if rets.empty:
        return None
    mu = rets.mean() * 252.0
    sig = rets.std() * np.sqrt(252.0)
    if sig <= 0:
        return None
    return float(mu / sig)

def _tilt_targets_with_sharpe(base_targets: Dict[str, float], end_dt: date, alpha: float = 0.25) -> Dict[str, float]:
    if not base_targets:
        return {}
    base = base_targets.copy()
    growth_keys = [k for k in base if k in GROWTH_CLASSES]

    def softmax(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        arr = np.array(list(scores.values()), dtype=float)
        arr = arr - np.nanmax(arr)
        w = np.exp(arr)
        w = w / (np.sum(w) if np.isfinite(np.sum(w)) and np.sum(w) > 0 else 1.0)
        return {k: float(w[i]) for i, k in enumerate(scores.keys())}

    if growth_keys:
        scores = {}
        for k in growth_keys:
            proxy = CLASS_PROXY.get(k)
            sh = _compute_proxy_sharpe(proxy, end_dt=end_dt, lookback_days=60) if proxy else None
            scores[k] = sh if (sh is not None and np.isfinite(sh)) else 0.0
        sm = softmax(scores)
        S = sum(base[k] for k in growth_keys)
        tilted = {k: S * sm.get(k, 0.0) for k in growth_keys}
        for k in growth_keys:
            base[k] = (1.0 - alpha) * base[k] + alpha * tilted.get(k, 0.0)

    return _normalize_weights(base)

# ======================================================================
# Asignación sin ventas
# ======================================================================

@dataclass
class ContributionResult:
    effective_date: date
    contribution_amount: float
    df_allocation: pd.DataFrame
    leftover: float
    total_after: float
    final_weights: Dict[str, float]

def _allocate_toward_targets_no_sells(
    value_by_class: Dict[str, float],
    total_value_t: float,
    contrib: float,
    target_weights: Dict[str, float],
    overflow_policy: str = "pro_rata_targets",
) -> ContributionResult:
    value_by_class = {_canon_class(k): float(v) for k, v in value_by_class.items()}
    target_weights = {_canon_class(k): float(v) for k, v in target_weights.items()}

    classes = sorted(set(list(value_by_class.keys()) + list(target_weights.keys())))
    V = float(total_value_t)
    C = max(0.0, float(contrib))
    T = _normalize_weights(target_weights)

    needs = {}
    for k in classes:
        target_value = T.get(k, 0.0) * (V + C)
        current_value = float(value_by_class.get(k, 0.0))
        need = target_value - current_value
        needs[k] = max(0.0, need)
    sum_needs = float(sum(needs.values()))

    alloc = {k: 0.0 for k in classes}
    leftover = 0.0

    if sum_needs <= 0.0:
        leftover = C
    elif sum_needs >= C - 1e-9:
        for k in classes:
            alloc[k] = (needs[k] / sum_needs) * C if sum_needs > 0 else 0.0
    else:
        for k in classes:
            alloc[k] = needs[k]
        leftover = C - sum_needs
        if leftover > 0:
            if overflow_policy == "pro_rata_targets":
                for k in classes:
                    alloc[k] += leftover * T.get(k, 0.0)
                leftover = 0.0
            elif overflow_policy == "growth_first":
                gsum = sum(T.get(k, 0.0) for k in classes if k in GROWTH_CLASSES)
                if gsum > 0:
                    for k in classes:
                        if k in GROWTH_CLASSES:
                            alloc[k] += leftover * (T.get(k, 0.0) / gsum)
                    leftover = 0.0

    V_after = V + C
    final_vals = {k: float(value_by_class.get(k, 0.0)) + float(alloc.get(k, 0.0)) for k in classes}
    w_final = {k: (final_vals[k] / V_after if V_after > 0 else 0.0) for k in classes}

    w_inicial = {k: (float(value_by_class.get(k, 0.0)) / V if V > 0 else 0.0) for k in classes}
    w_aport = {k: (float(alloc.get(k, 0.0)) / C if C > 0 else 0.0) for k in classes}
    monto_inicial_dict = {k: float(value_by_class.get(k, 0.0)) for k in classes}

    df = pd.DataFrame({
        "clase": [_display_name(k) for k in classes],
        "w_inicial_%": [100.0 * w_inicial.get(k, 0.0) for k in classes],
        "monto_inicial_$": [monto_inicial_dict.get(k, 0.0) for k in classes],
        "w_aportacion_%": [100.0 * w_aport.get(k, 0.0) for k in classes],
        "aportacion_$": [float(alloc.get(k, 0.0)) for k in classes],
        "w_final_%": [100.0 * w_final.get(k, 0.0) for k in classes],
        "valor_final_$": [float(final_vals.get(k, 0.0)) for k in classes],
    }).sort_values("clase").reset_index(drop=True)

    return ContributionResult(
        effective_date=_today(),
        contribution_amount=C,
        df_allocation=df,
        leftover=float(leftover),
        total_after=V_after,
        final_weights=w_final,
    )

# ======================================================================
# API pública
# ======================================================================

def compute_contribution_recommendation(
    *,
    start_dt: date,
    effective_dt: date,
    seed_capital_pesos: float,
    fee_pp: float,
    df_opt: pd.DataFrame,
    fi_rows: List[Dict],
    eq_sel: Dict[str, Dict],
    fx_sel: Dict[str, Dict],
    contribution_amount: float,
    use_tilt: bool = True,
    tilt_alpha: float = 0.25,
    freeze_level0_to_current: bool = True,
    overflow_policy: str = "pro_rata_targets",
) -> ContributionResult:

    existing_pos = st.session_state.get("ops_positions")
    existing_ts = st.session_state.get("ops_timeseries")

    if existing_pos is None or existing_pos.empty or existing_ts is None or existing_ts.empty:
        raise ValueError("No hay datos del portafolio. Debes iniciar operación primero.")

    pos_df_init = existing_pos.copy()
    end_ts = pd.Timestamp(effective_dt)
    ts_df_t = existing_ts.loc[:end_ts].copy()

    if ts_df_t.empty:
        ts_df_t = existing_ts.iloc[[0]].copy()

    snap = _aggregate_by_class(pos_df_init, ts_df_t)
    V_t, by_class, w_t = snap.total_value, snap.value_by_class, snap.weight_by_class

    targets = _class_targets_from_opt(df_opt)

    if freeze_level0_to_current and w_t:
        for k in LEVEL0_CLASSES:
            if k in w_t:
                targets[k] = w_t[k]
        targets = _normalize_weights(targets)

    if use_tilt:
        targets = _tilt_targets_with_sharpe(targets, end_dt=effective_dt, alpha=tilt_alpha)

    result = _allocate_toward_targets_no_sells(
        value_by_class=by_class,
        total_value_t=V_t,
        contrib=contribution_amount,
        target_weights=targets,
        overflow_policy=overflow_policy,
    )
    result.effective_date = effective_dt
    return result

def apply_user_contribution_weights(
    *,
    start_dt: date,
    effective_dt: date,
    seed_capital_pesos: float,
    fee_pp: float,
    df_opt: pd.DataFrame,
    fi_rows: List[Dict],
    eq_sel: Dict[str, Dict],
    fx_sel: Dict[str, Dict],
    contribution_amount: float,
    contrib_weights_by_class_pct: Dict[str, float],
) -> ContributionResult:

    existing_pos = st.session_state.get("ops_positions")
    existing_ts = st.session_state.get("ops_timeseries")

    if existing_pos is None or existing_pos.empty or existing_ts is None or existing_ts.empty:
        raise ValueError("No hay datos del portafolio. Debes iniciar operación primero.")

    pos_df_init = existing_pos.copy()
    end_ts = pd.Timestamp(effective_dt)
    ts_df_t = existing_ts.loc[:end_ts].copy()

    if ts_df_t.empty:
        ts_df_t = existing_ts.iloc[[0]].copy()

    snap = _aggregate_by_class(pos_df_init, ts_df_t)
    V_t, by_class, _ = snap.total_value, snap.value_by_class, snap.weight_by_class

    uw_raw = {_canon_class(k): max(0.0, float(v)) for k, v in contrib_weights_by_class_pct.items()}
    if max(uw_raw.values() or [0]) > 1.0:
        uw_raw = {k: v / 100.0 for k, v in uw_raw.items()}
    uw = _normalize_weights(uw_raw)

    classes = sorted(set(list(by_class.keys()) + list(uw.keys())))
    C = max(0.0, float(contribution_amount))
    V_after = V_t + C

    alloc = {k: C * uw.get(k, 0.0) for k in classes}
    final_vals = {k: float(by_class.get(k, 0.0)) + alloc.get(k, 0.0) for k in classes}
    w_final = {k: (final_vals[k] / V_after if V_after > 0 else 0.0) for k in classes}

    w_inicial = {k: (float(by_class.get(k, 0.0)) / V_t if V_t > 0 else 0.0) for k in classes}
    monto_inicial_dict = {k: float(by_class.get(k, 0.0)) for k in classes}

    df = pd.DataFrame({
        "clase": [_display_name(k) for k in classes],
        "w_inicial_%": [100.0 * w_inicial.get(k, 0.0) for k in classes],
        "monto_inicial_$": [monto_inicial_dict.get(k, 0.0) for k in classes],
        "w_aportacion_%": [100.0 * uw.get(k, 0.0) for k in classes],
        "aportacion_$": [float(alloc.get(k, 0.0)) for k in classes],
        "w_final_%": [100.0 * w_final.get(k, 0.0) for k in classes],
        "valor_final_$": [float(final_vals.get(k, 0.0)) for k in classes],
    }).sort_values("clase").reset_index(drop=True)

    return ContributionResult(
        effective_date=effective_dt,
        contribution_amount=C,
        df_allocation=df,
        leftover=0.0,
        total_after=V_after,
        final_weights=w_final,
    )

# ======================================================================
# FUNCIONES AUXILIARES PARA RECOMENDACIONES
# ======================================================================

def _calcular_rendimiento_historico(
    ticker: str,
    fecha_inicio: pd.Timestamp,
    fecha_fin: pd.Timestamp
) -> Optional[float]:
    """Calcula el rendimiento histórico entre dos fechas."""
    try:
        df = get_hist_5y(ticker)
        if df is None or df.empty or "Adj Close" not in df.columns:
            return None
        adj = df["Adj Close"]
        if isinstance(adj, pd.DataFrame):
            adj = adj.iloc[:, 0]
        adj = adj.dropna()
        if adj.empty:
            return None
        s = adj.loc[fecha_inicio:fecha_fin]
        if s.size < 2:
            return None
        ret = (s.iloc[-1] / s.iloc[0]) - 1.0
        return float(ret) if np.isfinite(ret) else None
    except Exception:
        return None

def _calcular_score_value_investing(ticker: str) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Calcula score de Value Investing usando cache de fundamentales.
    Respeta (si existe) el lookback dinámico en st.session_state["_recs_lookback_days"].
    """
    try:
        from tabs.yf_store import get_hist_5y, get_fundamentals_cached

        fundamentals = get_fundamentals_cached(ticker, max_age_hours=168.0)  # 7 días
        if not fundamentals:
            return None, {}

        df = get_hist_5y(ticker)
        if df is None or df.empty or "Adj Close" not in df.columns:
            return None, {}

        adj = df["Adj Close"]
        if isinstance(adj, pd.DataFrame):
            adj = adj.iloc[:, 0]
        adj = adj.dropna()

        # 🔹 Ventana dinámica (si está configurada)
        lb = st.session_state.get("_recs_lookback_days")
        if lb and isinstance(lb, int) and lb > 0:
            adj = adj.iloc[-(lb+2):]

        if adj.empty or len(adj) < 252:
            return None, {}

        metricas = {}
        score_total = 0.0

        sector = fundamentals.get("sector")
        metricas["sector"] = sector

        # --- I. VALORACIÓN (35%) ---
        pe = fundamentals.get("forwardPE") or fundamentals.get("trailingPE")
        if pe and np.isfinite(pe) and pe > 0:
            metricas["pe_ratio"] = float(pe)
            if pe < 10:
                score_total += 20.0
            elif pe < 15:
                score_total += 17.0
            elif pe < 20:
                score_total += 12.0
            elif pe < 25:
                score_total += 7.0
            elif pe < 30:
                score_total += 3.0
            else:
                score_total -= 5.0
        else:
            metricas["pe_ratio"] = None

        pb = fundamentals.get("priceToBook")
        if pb and np.isfinite(pb) and pb > 0:
            metricas["price_to_book"] = float(pb)
            if pb < 1.0:
                score_total += 15.0
            elif pb < 1.5:
                score_total += 12.0
            elif pb < 3.0:
                score_total += 8.0
            elif pb < 5.0:
                score_total += 4.0
            else:
                score_total -= 3.0
        else:
            metricas["price_to_book"] = None

        # --- II. CALIDAD (30%) ---
        roe = fundamentals.get("returnOnEquity")
        if roe and np.isfinite(roe):
            roe_pct = float(roe * 100)
            metricas["roe"] = roe_pct
            if roe_pct > 25:
                score_total += 15.0
            elif roe_pct > 20:
                score_total += 12.0
            elif roe_pct > 15:
                score_total += 9.0
            elif roe_pct > 10:
                score_total += 5.0
            else:
                score_total -= 2.0
        else:
            metricas["roe"] = None

        eps_growth = fundamentals.get("earningsQuarterlyGrowth") or fundamentals.get("earningsGrowth")
        if eps_growth and np.isfinite(eps_growth):
            eps_pct = float(eps_growth * 100)
            eps_pct = max(-100.0, min(300.0, eps_pct))
            metricas["eps_growth"] = eps_pct
            if eps_pct > 20:
                score_total += 15.0
            elif eps_pct > 15:
                score_total += 12.0
            elif eps_pct > 10:
                score_total += 9.0
            elif eps_pct > 5:
                score_total += 5.0
            elif eps_pct > 0:
                score_total += 2.0
            else:
                score_total -= 3.0
        else:
            metricas["eps_growth"] = None

        # --- III. SALUD FINANCIERA (20%) ---
        de = fundamentals.get("debtToEquity")
        if de is not None and np.isfinite(de) and de >= 0:
            metricas["debt_to_equity"] = float(de)
            if de < 0.3:
                score_total += 10.0
            elif de < 0.5:
                score_total += 8.0
            elif de < 1.0:
                score_total += 5.0
            elif de < 2.0:
                score_total += 2.0
            else:
                score_total -= 5.0
        else:
            metricas["debt_to_equity"] = None

        fcf = fundamentals.get("freeCashflow")
        market_cap = fundamentals.get("marketCap")
        fcf_yield = None
        if fcf and market_cap and np.isfinite(fcf) and np.isfinite(market_cap) and market_cap > 0:
            fcf_yield = (float(fcf) / float(market_cap)) * 100
            fcf_yield = max(-50.0, min(100.0, fcf_yield))
        if fcf_yield is not None and np.isfinite(fcf_yield):
            metricas["fcf_yield"] = fcf_yield
            if fcf_yield > 12:
                score_total += 10.0
            elif fcf_yield > 8:
                score_total += 8.0
            elif fcf_yield > 6:
                score_total += 6.0
            elif fcf_yield > 3:
                score_total += 3.0
            elif fcf_yield > 0:
                score_total += 1.0
            else:
                score_total -= 8.0
        else:
            metricas["fcf_yield"] = None

        # --- IV. MOMENTUM (10%) ---
        n = len(adj)
        ret_3m = (adj.iloc[-1] / adj.iloc[-63]) - 1.0 if n >= 63 else np.nan
        if np.isfinite(ret_3m):
            metricas["ret_3m"] = float(ret_3m)
            if ret_3m > 0.10:
                score_total += 5.0
            elif ret_3m > 0.05:
                score_total += 3.5
            elif ret_3m > 0:
                score_total += 2.0
        else:
            metricas["ret_3m"] = None

        ret_12m = (adj.iloc[-1] / adj.iloc[-252]) - 1.0 if n >= 252 else np.nan
        if np.isfinite(ret_12m):
            metricas["ret_12m"] = float(ret_12m)
            if ret_12m > 0.20:
                score_total += 5.0
            elif ret_12m > 0.10:
                score_total += 3.5
            elif ret_12m > 0:
                score_total += 2.0
        else:
            metricas["ret_12m"] = None

        # --- V. RIESGO - Volatilidad (5%) ---
        volatilidad = adj.pct_change().std() * np.sqrt(252)
        if np.isfinite(volatilidad):
            metricas["volatilidad"] = float(volatilidad)
            if volatilidad < 0.15:
                score_total += 5.0
            elif volatilidad < 0.25:
                score_total += 3.0
            elif volatilidad < 0.35:
                score_total += 1.0
            elif volatilidad < 0.50:
                score_total += 0.0
            else:
                score_total -= 3.0
        else:
            metricas["volatilidad"] = None

        return float(score_total) if score_total > 0 else 0.0, metricas

    except Exception:
        return None, {}

def _limpiar_outliers(recomendaciones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Winsoriza outliers a límites razonables."""
    if not recomendaciones:
        return recomendaciones
    df = pd.DataFrame(recomendaciones)
    limites_metricas = {
        "pe_ratio": (0, 100),
        "price_to_book": (0, 20),
        "roe": (-50, 100),
        "eps_growth": (-100, 300),
        "debt_to_equity": (0, 10),
        "fcf_yield": (-50, 100),
        "ret_3m": (-0.5, 2.0),
        "ret_12m": (-0.8, 5.0),
        "volatilidad": (0, 2.0),
    }
    for metrica, (min_val, max_val) in limites_metricas.items():
        if metrica in df.columns:
            df[metrica] = df[metrica].clip(lower=min_val, upper=max_val)
    return df.to_dict('records')

def _imputar_metricas_faltantes(recomendaciones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Imputa NaNs con promedio por sector o promedio global."""
    if not recomendaciones:
        return recomendaciones
    df = pd.DataFrame(recomendaciones)
    metricas_a_imputar = [
        "pe_ratio", "price_to_book", "roe", "eps_growth",
        "debt_to_equity", "fcf_yield", "ret_3m", "ret_12m", "volatilidad"
    ]
    for metrica in metricas_a_imputar:
        if metrica not in df.columns:
            continue
        mask_nan = df[metrica].isna()
        if not mask_nan.any():
            continue
        promedios_sector = df.groupby("sector")[metrica].mean()
        promedio_general = df[metrica].mean()
        for idx in df[mask_nan].index:
            sector = df.loc[idx, "sector"]
            if sector in promedios_sector.index:
                val = promedios_sector[sector]
                if pd.notna(val):
                    df.loc[idx, metrica] = val
                    continue
            if pd.notna(promedio_general):
                df.loc[idx, metrica] = promedio_general
    return df.to_dict('records')

def recomendar_rv(
    categoria: str,  # "RV_EXT", "RV_MX", o "FIBRA_MX"
    fecha_efectiva: date,
    peso_asignado: float,
    top_n: int = 10,
    progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Recomienda acciones según categoría (RV Internacional, Nacional o FIBRAs).
    - Si fecha_efectiva < hoy: ranking por rendimiento histórico (Parquet)
    - Si fecha_efectiva >= hoy: score predictivo (momentum + PE cacheado)
    Emite 'pe_ratio' (no 'pe') para ser compatible con operations.py.
    """

    # --- CACHE por (categoria, fecha, top_n) para evitar recomputar ---
    cache_key = f"rec_cache::{categoria}::{fecha_efectiva.isoformat()}::{top_n}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    from datetime import datetime as _dt
    from tabs.equities import load_universe
    from tabs.yf_store import (
        preload_hist_5y_daily,
        get_hist_5y,
        get_meta_cached_light,
        get_fundamentals_cached,
    )

    # helpers locales
    def _safe_adj_close_series(df: Optional[pd.DataFrame]) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype=float)
        ser = df.get("Adj Close", None)
        if ser is None:
            return pd.Series(dtype=float)
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
        try:
            return ser.astype(float).dropna()
        except Exception:
            return pd.Series(dtype=float)

    def _ret_between(adj: pd.Series, a: pd.Timestamp, b: pd.Timestamp) -> Optional[float]:
        s = adj.loc[a:b]
        if s.size < 2:
            return None
        r = float(s.iloc[-1] / s.iloc[0] - 1.0)
        return r if np.isfinite(r) else None

    def _momentum_score(adj: pd.Series) -> float:
        n = adj.size
        if n < 63:  # ~3m
            return 0.0
        one_m = float(adj.iloc[-1] / adj.iloc[-21] - 1.0) if n >= 21 else 0.0
        three_m = float(adj.iloc[-1] / adj.iloc[-63] - 1.0) if n >= 63 else 0.0
        twelve_m = float(adj.iloc[-1] / adj.iloc[-252] - 1.0) if n >= 252 else 0.0
        score = 1.0 * one_m + 1.5 * three_m + 2.0 * twelve_m
        return float(score) if np.isfinite(score) else 0.0

    try:
        universe = load_universe()
    except Exception:
        return []

    if progress_callback:
        progress_callback(0.10, f"Filtrando universo de {categoria}…")

    # Filtrado por categoría
    U = universe.copy()
    idx_upper = U["index"].str.upper().str.strip()
    name_upper = U["name"].str.upper().str.strip()

    if categoria == "RV_EXT":
        mask = ~idx_upper.isin(["MEX", "MEXICO", "MX", "BMV", "IPC"])
        filtered = U[mask].copy()
    elif categoria == "RV_MX":
        mask = idx_upper.isin(["MEX", "MEXICO", "MX", "BMV", "IPC"]) & ~name_upper.str.startswith("FIBRA")
        filtered = U[mask].copy()
    elif categoria == "FIBRA_MX":
        mask = idx_upper.isin(["MEX", "MEXICO", "MX", "BMV", "IPC"]) & name_upper.str.startswith("FIBRA")
        filtered = U[mask].copy()
    else:
        return []

    if filtered.empty:
        return []

    tickers = filtered["yahoo"].astype(str).tolist()
    total_tickers = len(tickers)

    # --- Horizonte dinámico (cap en 5 años ≈ 1260 días hábiles) ---
    hoy = _dt.now().date()
    es_backdating = fecha_efectiva < hoy
    if es_backdating:
        a = pd.Timestamp(fecha_efectiva)
        b = pd.Timestamp(hoy)
        dias_naturales = max(1, (b - a).days)
    else:
        dias_naturales = 365  # por defecto 1Y si es prospectivo
    lookback_days = min(1260, max(252, int(dias_naturales * 252 / 365)))
    # Exponer lookback al helper (para momentum/vol)
    st.session_state["_recs_lookback_days"] = lookback_days

    # ---------- Warm-up del caché: rápido, paralelo y una sola vez por (categoria, ventana) ----------
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _warm_hist_cache_once(_tickers: List[str], _categoria: str, _lookback_days: int, _budget: int = 160):
        key = f"_hist_warm::{_categoria}::{_lookback_days}"
        warmed: set = st.session_state.get(key, set())
        if warmed:
            return

        # (opcional) priorizar por market cap si tu universo lo trae
        try:
            from tabs.equities import load_universe
            U_all = load_universe()
            mcap_map = {}
            if "yahoo" in U_all.columns and "market_cap" in U_all.columns:
                mcap_map = dict(zip(U_all["yahoo"].astype(str), pd.to_numeric(U_all["market_cap"], errors="coerce")))
                _tickers = sorted(_tickers, key=lambda t: -(mcap_map.get(t, 0) or 0))
        except Exception:
            pass

        subset = _tickers[: min(_budget, len(_tickers))]

        def _load_one(tk: str):
            try:
                _ = get_hist_5y(tk)
                return True
            except Exception:
                return False

        if progress_callback:
            progress_callback(0.25, f"Templando caché de {len(subset)}/{len(_tickers)} tickers…")

        ok = 0
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = {ex.submit(_load_one, tk): tk for tk in subset}
            for i, fut in enumerate(as_completed(futs)):
                _ = fut.result()
                ok += 1
                if progress_callback and len(subset):
                    frac = 0.25 + 0.10 * (ok / len(subset))  # 25%->35%
                    progress_callback(frac, f"Templando caché… {ok}/{len(subset)}")

        st.session_state[key] = set(subset)

    _warm_hist_cache_once(tickers, categoria, lookback_days, _budget=160)

    recomendaciones: List[Dict[str, Any]] = []

    if progress_callback:
        progress_callback(0.40, "Calculando métricas…")

    if es_backdating:
        a = pd.Timestamp(fecha_efectiva)
        b = pd.Timestamp(hoy)

        for idx, tk in enumerate(tickers):
            df = get_hist_5y(tk)
            adj = _safe_adj_close_series(df)
            if adj.empty:
                continue
            adj = adj.iloc[-(lookback_days+2):]
            r = _ret_between(adj, a, b)
            if r is None:
                continue

            from tabs.yf_store import get_meta_cached_light, get_fundamentals_cached
            meta_l = get_meta_cached_light(tk)
            fundamentals = get_fundamentals_cached(tk, max_age_hours=168.0, force_refresh=False)
            pe_val = fundamentals.get("forwardPE") or fundamentals.get("trailingPE")
            pe_ratio = float(pe_val) if (pe_val is not None and np.isfinite(pe_val) and pe_val > 0) else None

            recomendaciones.append({
                "ticker": tk,
                "nombre": meta_l.get("longName", tk),
                "sector": meta_l.get("sector", "—"),
                "score": float(r),
                "rendimiento": float(r),
                "pe_ratio": pe_ratio,
            })

            if progress_callback and total_tickers:
                progress_callback(0.40 + 0.55 * (idx + 1) / total_tickers, f"Analizando {idx+1}/{total_tickers}…")

        recomendaciones.sort(key=lambda x: x["score"], reverse=True)

    else:
        for idx, tk in enumerate(tickers):
            score, metricas = _calcular_score_value_investing(tk)
            if score is None or score <= 0:
                if progress_callback and total_tickers:
                    progress_callback(0.40 + 0.55 * (idx + 1) / total_tickers, f"Analizando {idx+1}/{total_tickers}…")
                continue

            # Obtener nombre desde el universo (CSV)
            nombre = tk
            try:
                match = filtered[filtered["yahoo"].str.upper() == tk.upper()]
                if not match.empty:
                    nombre = str(match.iloc[0]["name"])
            except Exception:
                from tabs.yf_store import get_meta_cached_light
                meta_l = get_meta_cached_light(tk)
                nombre = meta_l.get("longName", tk)

            sector = metricas.get("sector", "—")
            if categoria == "FIBRA_MX":
                sector = None
            elif not sector or str(sector).strip() == "":
                sector = "—"

            recomendaciones.append({
                "ticker": tk,
                "nombre": nombre,
                "sector": sector,
                "score": float(score),
                "pe_ratio": metricas.get("pe_ratio"),
                "price_to_book": metricas.get("price_to_book"),
                "roe": metricas.get("roe"),
                "eps_growth": metricas.get("eps_growth"),
                "fcf_yield": metricas.get("fcf_yield"),
                "debt_to_equity": metricas.get("debt_to_equity"),
                "ret_3m": metricas.get("ret_3m"),
                "ret_12m": metricas.get("ret_12m"),
                "volatilidad": metricas.get("volatilidad"),
            })

            if progress_callback and total_tickers:
                progress_callback(0.40 + 0.55 * (idx + 1) / total_tickers, f"Analizando {idx+1}/{total_tickers}…")

        recomendaciones = _limpiar_outliers(recomendaciones)
        recomendaciones = _imputar_metricas_faltantes(recomendaciones)
        recomendaciones.sort(key=lambda x: x["score"], reverse=True)

    if progress_callback:
        progress_callback(1.0, "¡Completado!")

    out = recomendaciones[:max(1, int(top_n))]
    st.session_state[cache_key] = out
    return out

# ======================================================================
# CONFIRMAR APORTACIÓN EN LEDGER (por clase) — Compra con títulos enteros
# ======================================================================

def confirmar_aportacion_en_ledger(ss, effective_date: date, debug: bool = True) -> dict:
    """
    Registra la aportación SIN tocar el depósito inicial:
      1) Inserta un DEPOSIT (nota: 'APORTACIÓN') por el total.
      2) Inserta BUY por ACTIVO, repartiendo cada monto de CLASE a sus activos
         según su peso por valor actual dentro de la clase (uniforme si todo es 0).
      3) COMPRA EN ENTEROS según tamaño de lote (por clase/símbolo) y deja buffer
         de comisión (commission_buffer_pct). Lo que no alcance por redondeo queda en efectivo.

    Requiere:
      - ss['last_contrib_result'] con ['clase','aportacion_$']
      - ss['ops_positions'] con ['ticker','activo','clase','qty','px_inicial' (y opcional 'px_actual')]
      - ss['contrib_amount_last'] > 0
      - ss['synthetic_prices'] opcional (para precio a la fecha)
    """
    import pandas as pd
    import numpy as np
    from tabs.ledger import add_transaction

    if debug: st.write("▶️ confirmar_aportacion_en_ledger()")
    
    # ⭐ Inicializar debug log
    if "aportaciones_debug_log" not in ss:
        ss["aportaciones_debug_log"] = []
    ss["aportaciones_debug_log"].clear()
    
    def log_debug(msg):
        ss["aportaciones_debug_log"].append(msg)
        if debug:
            st.write(msg)

    df_alloc = ss.get("last_contrib_result", pd.DataFrame())
    if df_alloc is None or df_alloc.empty:
        raise ValueError("No hay resultado de aportación para confirmar (last_contrib_result vacío).")

    contrib_total = float(ss.get("contrib_amount_last") or 0.0)
    if contrib_total <= 0:
        raise ValueError("El monto de aportación debe ser positivo.")

    pos_df = ss.get("ops_positions", pd.DataFrame())
    if pos_df is None or pos_df.empty:
        raise ValueError("No hay posiciones cargadas (ops_positions). Inicia operación primero.")

    # normaliza numéricos
    df_alloc = df_alloc.copy()
    df_alloc["aportacion_$"] = pd.to_numeric(df_alloc.get("aportacion_$", 0.0), errors="coerce").fillna(0.0)

    if debug:
        st.write("• effective_date:", effective_date)
        st.write("• contrib_total:", contrib_total)
        st.write("• df_alloc (head):")
        st.dataframe(df_alloc.head())

    # 1) Fila separadora: APORTACIÓN como DEPOSIT
    add_transaction(
        ss,
        ts=effective_date,
        kind="DEPOSIT",
        cash=float(contrib_total),
        note="APORTACIÓN"
    )
    if debug: st.write("✓ Insertado DEPOSIT con nota 'APORTACIÓN'")

def _price_at_date_mxn(ss, ticker: str, clase_raw: str, fallback_px: float, effective_date: date) -> float:
    """
    Precio del activo a la fecha efectiva, expresado en MXN.
    - Si hay serie sintética (ya en MXN), usa esa.
    - Si es FX, aplica la misma lógica que ledger (invertir USD/XXX y multiplicar por USD→MXN).
    - Si no es FX, convierte según la moneda reportada por Yahoo.
    """
    import numpy as np
    import pandas as pd
    import re
    from tabs.yf_store import get_hist_5y, get_meta_cached_light

    ts = pd.Timestamp(effective_date)

    # 0) Preferir serie sintética ya en MXN
    try:
        sp = ss.get("synthetic_prices", {}) or {}
        meta_syn = sp.get(ticker, {})
        ser = meta_syn.get("price_series")
        if isinstance(ser, pd.Series) and not ser.empty:
            px = ser.reindex(pd.DatetimeIndex([ts]), method="pad").iloc[0]
            if np.isfinite(px) and px > 0:
                return float(px)
    except Exception:
        pass

    # Helper: último Adj Close <= ts
    def _adj_close_last_on_or_before(sym: str) -> float:
        try:
            df = get_hist_5y(sym)
            s = df["Adj Close"] if (df is not None and not df.empty and "Adj Close" in df.columns) else None
            if s is None or s.empty:
                return float("nan")
            s = s.dropna()
            s_cut = s[s.index <= ts]
            if s_cut.empty:
                return float("nan")
            return float(s_cut.iloc[-1])
        except Exception:
            return float("nan")

    # 1) Precio nativo (en moneda del ticker)
    px_native = _adj_close_last_on_or_before(ticker)
    if not np.isfinite(px_native) or px_native <= 0:
        px_native = float(fallback_px or 1.0)

    tkr_upper = str(ticker).upper()

    # 2) Caso FX: misma regla que en ledger
    if tkr_upper.endswith("=X"):
        # USD/MXN para convertir cuando el par no es ya contra MXN
        usd_mxn = _adj_close_last_on_or_before("MXN=X")

        # ¿Es un USD/XXX de 3 letras? (ej. BRL=X, JPY=X) → invertir
        inverse = bool(re.fullmatch(r"^[A-Z]{3}=X$", tkr_upper)) and tkr_upper not in {"MXN=X", "USD=X"}

        # Respetar bandera del UI si existe
        fx_sel = ss.get("fx_selection", {}) or {}
        for _, meta in (fx_sel or {}).items():
            fx_tkr = str(meta.get("yahoo_ticker", "")).upper()
            if fx_tkr == tkr_upper:
                inverse = bool(meta.get("inverse", inverse))
                break

        px_series = (1.0 / px_native) if inverse else px_native

        # Si ya es XXX/MXN (o MXN=X), no convertir más
        if ("MXN" in tkr_upper) or (tkr_upper == "MXN=X"):
            return float(px_series)

        # Si no es contra MXN, multiplicar por USD→MXN
        if np.isfinite(usd_mxn) and usd_mxn > 0:
            return float(px_series * usd_mxn)
        return float(px_series)

    # 3) No-FX: convertir según moneda reportada por Yahoo
    try:
        meta_l = (get_meta_cached_light(ticker) or {})
        ccy = str(meta_l.get("currency") or "").upper().strip()
    except Exception:
        ccy = ""

    if not ccy:
        if tkr_upper.endswith(".MX"):
            ccy = "MXN"
        else:
            ccy = "USD"  # fallback razonable

    if ccy in ("", "MXN"):
        return float(px_native)

    if ccy == "USD":
        usd_mxn = _adj_close_last_on_or_before("MXN=X")
        if np.isfinite(usd_mxn) and usd_mxn > 0:
            return float(px_native * usd_mxn)
        return float(px_native)

    # Otra moneda: intentar par directo XXXMXN=X; si no, XXXUSD=X * USD→MXN
    k_direct = _adj_close_last_on_or_before(f"{ccy}MXN=X")
    if np.isfinite(k_direct) and k_direct > 0:
        return float(px_native * k_direct)

    k1 = _adj_close_last_on_or_before(f"{ccy}USD=X")
    k2 = _adj_close_last_on_or_before("MXN=X")
    if np.isfinite(k1) and k1 > 0 and np.isfinite(k2) and k2 > 0:
        return float(px_native * k1 * k2)

    return float(px_native)



    # Valor actual por activo (para pesos internos de la clase)
    work = pos_df.copy()
    px_now = pd.to_numeric(work.get("px_actual", work.get("px_inicial", 0.0)), errors="coerce").fillna(0.0)
    qty_now = pd.to_numeric(work.get("qty", 0.0), errors="coerce").fillna(0.0)
    work["_val_actual"] = (qty_now * px_now).astype(float)

    buys_count = 0
    leftover_total = 0.0

    def _eq(a, b):
        return str(a).strip().lower() == str(b).strip().lower()

    buffer_pct = _commission_buffer_pct_ss(ss)

    # 2) Reparto por clase -> activos
    # ⭐ DEBUG: Ver columnas de df_alloc
    log_debug(f"🔍 Columnas en df_alloc: {list(df_alloc.columns)}")
    log_debug(f"🔍 Primera fila de df_alloc: {df_alloc.iloc[0].to_dict() if not df_alloc.empty else 'VACÍO'}")
    
    for _, row in df_alloc.iterrows():
        clase_display = str(row.get("clase", "")).strip()
        class_amount = float(row.get("aportacion_$", 0.0))
        if class_amount <= 1e-9:
            continue
        
        # ⭐ Convertir nombre display a código canónico
        clase_canon = _canon_class(clase_display)
        
        # ⭐ DEBUG
        log_debug(f"🔍 Procesando clase: display='{clase_display}' → canon='{clase_canon}'")

        cols_keep = [c for c in ["ticker", "activo", "clase", "_val_actual", "px_inicial"] if c in work.columns]
        activos_clase = work[cols_keep].copy()
        
        # ⭐ Comparar con código canónico
        activos_clase = activos_clase[activos_clase["clase"].apply(lambda c: _canon_class(str(c)) == clase_canon)]

        if activos_clase.empty:
            if debug: st.write(f"• Clase sin activos existentes: {clase_display} (se omite)")
            continue

        vals = pd.to_numeric(activos_clase["_val_actual"], errors="coerce").fillna(0.0)
        if float(vals.sum()) > 0:
            w_in_class = (vals / float(vals.sum())).values
        else:
            n = len(activos_clase)
            w_in_class = np.ones(n) / n

        if debug:
            log_debug(f"• Clase '{clase_display}': monto={class_amount:,.2f}, activos={len(activos_clase)} (buffer {buffer_pct*100:.2f}%)")
            log_debug(f"🔍 Activos en clase {clase_display}:")
            for idx, row in activos_clase.iterrows():
                log_debug(f"   - Ticker: {row.get('ticker')} | Clase: {row.get('clase')}")

        for (i, r_act) in activos_clase.reset_index(drop=True).iterrows():
            tkr = str(r_act.get("ticker", "")).strip()
            nm  = str(r_act.get("activo", "")).strip()
            px0 = float(r_act.get("px_inicial", 1.0) or 1.0)

            # Monto a invertir en este activo (con buffer para comisiones)
            alloc_i = class_amount * float(w_in_class[i])
            alloc_net = max(0.0, alloc_i * (1.0 - buffer_pct))

            # Precio en la fecha efectiva
            # Precio en la fecha efectiva
            px = _price_at_date_mxn(ss, tkr, r_act.get("clase"), px0, effective_date)
            
            # ⭐ DEBUG: Ver precio calculado
            log_debug(f"🔍 {tkr} | clase={r_act.get('clase')} | px_mxn={px:,.2f} | px0_fallback={px0:,.2f}")
            
            if px <= 0:
                continue

            # Tamaño de lote por clase/símbolo
            lot = _lot_size_for(ss, tkr, r_act.get("clase"))

            # Cantidad a comprar:
            # - Si FX (lot=None): permitir fraccional
            # - Si no, redondear hacia abajo a múltiplos de 'lot'
            if lot is None or lot <= 0:
                qty = alloc_net / px
            else:
                qty_raw = alloc_net / px
                qty = math.floor(qty_raw / float(lot)) * float(lot)

            if qty <= 0:
                leftover_total += alloc_i  # todo ese monto no se invirtió
                continue

            spent = qty * px
            leftover_total += max(0.0, alloc_i - spent)  # lo no invertido queda en efectivo

            add_transaction(
                ss,
                ts=effective_date,
                kind="BUY",
                symbol=tkr,
                qty=qty,
                px=px,
                name=nm,
                note=f"Aportación {effective_date.isoformat()} (lot={lot}, buffer={buffer_pct:.4f})"
            )
            
            # ⭐ Guardar metadata de clase
            # ⭐ Guardar metadata de clase
            # ⭐ Guardar metadata de clase COMPLETA
            if "synthetic_prices" not in ss:
                ss["synthetic_prices"] = {}
            if tkr not in ss["synthetic_prices"]:
                ss["synthetic_prices"][tkr] = {}

            # Guardar toda la metadata necesaria
            ss["synthetic_prices"][tkr]["clase"] = str(r_act.get("clase", "")).strip()
            ss["synthetic_prices"][tkr]["activo"] = nm
            ss["synthetic_prices"][tkr]["px_inicial"] = px0
            ss["synthetic_prices"][tkr]["start_date"] = effective_date

            # 🔧 CREAR price_series para activos sintéticos
            if "price_series" not in ss["synthetic_prices"][tkr]:
                price_dict = {pd.Timestamp(effective_date): px0}
                ss["synthetic_prices"][tkr]["price_series"] = price_dict
                log_debug(f"   ✅ Creado price_series para {tkr}: {px0} en {effective_date}")
            else:
                existing_series = ss["synthetic_prices"][tkr]["price_series"]
                if isinstance(existing_series, dict):
                    existing_series[pd.Timestamp(effective_date)] = px0
                else:
                    price_dict = existing_series.to_dict() if hasattr(existing_series, 'to_dict') else {}
                    price_dict[pd.Timestamp(effective_date)] = px0
                    ss["synthetic_prices"][tkr]["price_series"] = price_dict
                log_debug(f"   ✅ Actualizado price_series para {tkr}: {px0} en {effective_date}")

            # ⭐ DEBUG: Confirmar guardado
            log_debug(f"✅ Guardado synthetic_prices['{tkr}']:")
            log_debug(f"   clase='{ss['synthetic_prices'][tkr].get('clase')}'")
            log_debug(f"   activo='{ss['synthetic_prices'][tkr].get('activo')}'")
            
            # ⭐ DEBUG: Confirmar guardado
            log_debug(f"✅ Guardado synthetic_prices['{tkr}']['clase'] = '{ss['synthetic_prices'][tkr].get('clase')}'")
            
            buys_count += 1
            if debug:
                log_debug(f"   - BUY {tkr}: alloc={alloc_i:,.2f} net={alloc_net:,.2f} @ {px:,.4f} → qty={qty:,.6f} spent={spent:,.2f}")

    if debug:
        st.info(f"Leftover total no invertido (queda en efectivo): ${leftover_total:,.2f}")

    return {
        "effective_date": str(effective_date),
        "deposit": float(contrib_total),
        "lines": int(buys_count),
        "leftover_cash": float(leftover_total),
    }

