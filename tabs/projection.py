# tabs/projection.py
from __future__ import annotations
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Set

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from tabs.yf_store import preload_hist_5y_daily, get_hist_sliced_years
from tabs.optimization import TRADING_DAYS
import tabs.optimization as optmod
# IMPORTAMOS el módulo de límites y la función de formato
from tabs import consar_limits 
from tabs.consar_limits import _si_fmt_number 


# ===== Config =====
HIST_YEARS = 3
FORECAST_YEARS = 1
STEPS_FWD = FORECAST_YEARS * TRADING_DAYS
MILLION = 1_000_000
BILLON = 1_000_000_000_000 # Un millón de millones

# Parámetros para la proyección a 40 años (cifras en Pesos)
PROJECTION_40Y_PARAMS = {
    "PROFUTURO": {
        "2026_anual": 21_754_000_000,
    },
    "SURA": {
        "2026_anual": 13_193_000_000,
    }
}


# ---------- helpers ----------
def _ensure_series(x):
    return x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x

def _fallback_caps_and_limits() -> Tuple[Dict[str, float], Dict[str, float], str, str]:
    caps = {"RV_TOTAL_MAX": 1.0,"EXT_MAX": 1.0, "FIBRA_MX_MAX": 1.0,"VAR_DAILY95_MAX": 0.013,"EXT_SINGLE_ISSUER_MAX": 1.0,"FX_MAX": 1.0,"SN_MAX": 1.0}
    risk_limits = {"VaR": 0.013,"CVaR": 0.013, "LIQ": 0.80,"TE": 0.05}
    return caps, risk_limits, "Básica Inicial (Fallback)", "N/A (Fallback)" 

def _pull_latest_optimization(force_refresh: bool = False) -> tuple[pd.DataFrame | None, dict]:
    if not force_refresh:
        tbl = st.session_state.get("optimization_table")
        risk = st.session_state.get("optimization_risk")
        if tbl is not None and risk is not None: return tbl, risk
    caps, risk_limits, siefore_name, afore_name = _fallback_caps_and_limits()
    try:
        if hasattr(optmod, "_auto_load_once"): optmod._auto_load_once()
        try:
            caps, risk_limits, siefore_name, afore_name = optmod._caps_from_consar()
        except Exception: pass
        rows = optmod._cosechar_activos()
        if not rows: return None, {}
        op_caps = caps
        if hasattr(optmod, "_caps_operativos"):
            buf = getattr(optmod, "OP_BUFFER", 0.0)
            op_caps = optmod._caps_operativos(caps, buffer=buf)
        stable_weights = st.session_state.get("stable_base_weights")
        base_names = st.session_state.get("stable_base_asset_names")
        opt_ret = optmod._optimizar(rows, op_caps, risk_limits, stable_base_weights=stable_weights, base_asset_names=base_names)
        if isinstance(opt_ret, tuple) and len(opt_ret) >= 1:
            tbl = opt_ret[0]
            risk = opt_ret[2] if len(opt_ret) >= 3 else {} 
        else: tbl, risk = None, {}
        if tbl is None or getattr(tbl, "empty", True): return None, {}
        final_weights = opt_ret[3] if len(opt_ret) >= 4 else {}
        if "stable_base_weights" not in st.session_state and final_weights:
            st.session_state["stable_base_weights"] = final_weights
            st.session_state["stable_base_asset_names"] = set(final_weights.keys())
        st.session_state["optimization_table"] = tbl
        st.session_state["optimization_risk"] = risk if isinstance(risk, dict) else {}
        return tbl, (risk if isinstance(risk, dict) else {})
    except Exception as e:
        tbl = st.session_state.get("optimization_table")
        risk = st.session_state.get("optimization_risk", {})
        return (tbl if tbl is not None else None), (risk if isinstance(risk, dict) else {})

def _ensure_optimization_table() -> pd.DataFrame | None:
    tbl, _ = _pull_latest_optimization(force_refresh=True)
    return tbl

def _ensure_mu_sigma(opt: pd.DataFrame) -> pd.DataFrame:
    out = opt.copy()
    if "mu" not in out.columns and "Esperado (%)" in out.columns: out["mu"] = pd.to_numeric(out["Esperado (%)"], errors="coerce") / 100.0
    if "sigma" not in out.columns and "Vol (%)" in out.columns: out["sigma"] = pd.to_numeric(out["Vol (%)"], errors="coerce") / 100.0
    if "Peso (%)" not in out.columns and "w" in out.columns: out["Peso (%)"] = pd.to_numeric(out["w"], errors="coerce") * 100.0
    need = {"nombre","clase","Peso (%)","mu"}
    miss = need - set(out.columns)
    if miss: raise ValueError(f"Faltan columnas: {miss}")
    if "sigma" not in out.columns: out["sigma"] = 0.0
    return out

def _name_to_ticker_map() -> dict:
    eq_sel = st.session_state.get("equity_selection", {}) or {}
    return {str(v.get("name", k)).strip(): str(k).strip() for k, v in eq_sel.items()}

def _fetch_daily_returns_for_assets(opt: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.DatetimeIndex]:
    n = HIST_YEARS * TRADING_DAYS
    idx = pd.bdate_range(end=datetime.today().date(), periods=n)
    name2ticker = _name_to_ticker_map()
    out = {}
    for _, row in opt.iterrows():
        nombre, clase, mu_a = str(row["nombre"]).strip(), str(row["clase"]).strip(), float(row["mu"])
        mu_d = mu_a / TRADING_DAYS
        if clase.startswith("RV") or clase == "FIBRA_MX":
            tkr = name2ticker.get(nombre)
            if not tkr:
                m = re.search(r"\(([A-Za-z0-9\.\-\^_=:]+)\)\s*$", nombre)
                if m: tkr = m.group(1)
            hist = None
            if tkr:
                try: hist = get_hist_sliced_years(tkr, HIST_YEARS)
                except Exception: hist = None
            if hist is not None and not hist.empty and "Adj Close" in hist.columns:
                r = _ensure_series(hist["Adj Close"]).pct_change(fill_method=None).reindex(idx).fillna(0.0)
                out[nombre] = r
            else:
                np.random.seed(abs(hash(f"SIM_RV_{nombre}")) % (2**32))
                sim = np.random.normal(mu_d, 0.001, len(idx))
                out[nombre] = pd.Series(sim, index=idx)
        else:
            out[nombre] = pd.Series(np.full(len(idx), mu_d), index=idx)
    return out, idx

def _portfolio_daily(r_by_name: dict[str, pd.Series], w_by_name: dict[str, float]) -> pd.Series:
    df = pd.DataFrame(r_by_name).fillna(0.0)
    for col in df.columns: df[col] = df[col] * float(w_by_name.get(col, 0.0))
    return df.sum(axis=1)

def _asset_annual_by_calendar_year(r: pd.Series) -> dict[int, float]:
    by_year = {}
    if r.empty: return by_year
    df = r.to_frame("r")
    df["year"] = df.index.year
    for y, g in df.groupby("year"): by_year[int(y)] = float((1.0 + g["r"]).prod() - 1.0)
    return by_year

def _get_mu_port_expected_exact(opt: pd.DataFrame, weights: dict[str, float]) -> float:
    _, risk = _pull_latest_optimization(force_refresh=True)
    if isinstance(risk, dict) and ("mu_port" in risk):
        try: return float(risk["mu_port"])
        except Exception: pass
    name_to_mu = {str(r["nombre"]).strip(): float(r["mu"]) for _, r in opt.iterrows()}
    return float(sum(weights.get(n, 0.0) * name_to_mu.get(n, 0.0) for n in weights.keys()))

def _modelA_table_and_targets(opt: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    opt = _ensure_mu_sigma(opt)
    r_by_name, idx_hist = _fetch_daily_returns_for_assets(opt)
    weights = {str(r["nombre"]).strip(): float(r["Peso (%)"])/100.0 for _, r in opt.iterrows()}
    r_port = _portfolio_daily(r_by_name, weights)
    dfp = r_port.to_frame("r"); dfp["year"] = dfp.index.year
    port_by_year = dfp.groupby("year", group_keys=False, sort=False).apply(lambda g: float((1.0 + g["r"]).prod() - 1.0), include_groups=False).to_dict()
    if not port_by_year: return pd.DataFrame(), {"P": 0.0, "N": 0.0, "O": 0.0}
    yr_pess = min(port_by_year, key=lambda y: port_by_year[y])
    yr_opt  = max(port_by_year, key=lambda y: port_by_year[y])
    rows = []
    for _, row in opt.iterrows():
        nombre, clase, wi, mu_a = str(row["nombre"]).strip(), str(row["clase"]).strip(), float(row["Peso (%)"])/100.0, float(row["mu"])
        r_i = r_by_name.get(nombre, pd.Series(0.0, index=idx_hist))
        if clase.startswith("RV") or clase == "FIBRA_MX":
            asset_by_year = _asset_annual_by_calendar_year(r_i)
            r_p, r_o, r_n = float(asset_by_year.get(yr_pess, mu_a)), float(asset_by_year.get(yr_opt, mu_a)), mu_a
        else: r_p = r_n = r_o = mu_a
        rows.append({"Activo": nombre,"Clase": clase,"Peso (%)": wi*100.0,"Rend. Pesimista (%)": r_p*100.0,"Rend. Neutral (%)": r_n*100.0,"Rend. Optimista (%)": r_o*100.0,"Rend×Peso (P)": wi*r_p*100.0,"Rend×Peso (N)": wi*r_n*100.0,"Rend×Peso (O)": wi*r_o*100.0})
    tbl = pd.DataFrame(rows)
    if tbl.empty: return tbl, {"P": 0.0, "N": 0.0, "O": 0.0}
    total_P = float(tbl["Rend×Peso (P)"].sum())/100.0
    total_O = float(tbl["Rend×Peso (O)"].sum())/100.0
    mu_port_exact = _get_mu_port_expected_exact(opt, {str(r["nombre"]).strip(): float(r["Peso (%)"])/100.0 for _, r in opt.iterrows()})
    total_N = float(mu_port_exact)
    total = {"Activo": "Total (ponderado)","Clase": "","Peso (%)": float(tbl["Peso (%)"].sum()),"Rend. Pesimista (%)": None,"Rend. Neutral (%)": None,"Rend. Optimista (%)": None,"Rend×Peso (P)": total_P*100.0,"Rend×Peso (N)": total_N*100.0,"Rend×Peso (O)": total_O*100.0}
    row_total = {c: (np.nan if pd.api.types.is_numeric_dtype(tbl[c].dtype) else None) for c in tbl.columns}
    row_total.update(total)
    df_total = pd.DataFrame([row_total]).reindex(columns=tbl.columns)
    num_cols = [c for c in tbl.columns if pd.api.types.is_numeric_dtype(tbl[c].dtype)]
    if num_cols: df_total[num_cols] = df_total[num_cols].apply(pd.to_numeric, errors="coerce")
    tbl = pd.concat([tbl, df_total], ignore_index=True)
    targets = {"P": total_P, "N": total_N, "O": total_O}
    return tbl, targets

def _portfolio_history(opt: pd.DataFrame):
    opt = _ensure_mu_sigma(opt)
    r_by_name, idx_hist = _fetch_daily_returns_for_assets(opt)
    weights = {str(r["nombre"]).strip(): float(r["Peso (%)"])/100.0 for _, r in opt.iterrows()}
    r_port = _portfolio_daily(r_by_name, weights)
    return r_by_name, r_port, idx_hist, weights

def _simulate_paths(r_by_name: dict[str, pd.Series], idx_hist: pd.DatetimeIndex, weights: dict[str, float], targets: dict, steps: int = STEPS_FWD, start_date: pd.Timestamp | None = None) -> tuple[pd.DatetimeIndex, dict]:
    opt = _ensure_mu_sigma(st.session_state["optimization_table"])
    name2class = {str(r["nombre"]).strip(): str(r["clase"]).strip() for _, r in opt.iterrows()}
    df_hist = pd.DataFrame({k: v for k, v in r_by_name.items()}).fillna(0.0)
    rv_cols = [n for n in df_hist.columns if name2class.get(n, "").startswith("RV") or name2class.get(n, "") == "FIBRA_MX"]
    nr_cols = [n for n in df_hist.columns if n not in rv_cols]
    mu_daily_nr = {n: (float(opt.loc[opt["nombre"]==n, "mu"].iloc[0]) / TRADING_DAYS) if (opt["nombre"]==n).any() else 0.0 for n in nr_cols}
    if start_date is None: future_dates = pd.bdate_range(start=idx_hist[-1] + timedelta(days=1), periods=steps)
    else: future_dates = pd.bdate_range(start=start_date, periods=steps)
    paths, rng = {}, np.random.default_rng(20251012)
    for scen_key, targ_ann in {"Pesimista": targets["P"], "Neutral": targets["N"], "Optimista": targets["O"]}.items():
        if rv_cols:
            sample_idx = rng.integers(low=0, high=len(df_hist), size=steps, endpoint=False)
            rv_block = df_hist[rv_cols].to_numpy()[sample_idx, :]
        else: rv_block = np.zeros((steps, 0))
        total_sim = np.zeros(steps, dtype=float)
        for j, col in enumerate(rv_cols): total_sim += float(weights.get(col, 0.0)) * rv_block[:, j]
        for col in nr_cols: total_sim += float(weights.get(col, 0.0)) * (mu_daily_nr.get(col, 0.0))
        targ_daily = (1.0 + float(targ_ann)) ** (1.0 / TRADING_DAYS) - 1.0
        total_sim += targ_daily - float(np.mean(total_sim))
        growth, target_growth = (np.cumprod(1.0 + total_sim)[-1] if steps > 0 else 1.0), 1.0 + float(targ_ann)
        f = 1.0 if growth <= 0 else (target_growth / growth) ** (1.0 / steps)
        total_sim = (1.0 + total_sim) * f - 1.0
        paths[scen_key] = (1.0 + pd.Series(total_sim, index=future_dates)).cumprod()
    return future_dates, paths

def _densify_monthly_to_bdays(monthly_pesos: pd.Series, r_port_hist: pd.Series | None = None, seed: int = 20251012) -> pd.Series:
    if len(monthly_pesos) < 2: return monthly_pesos.copy()
    rng = np.random.default_rng(seed)
    if r_port_hist is not None and len(r_port_hist) > 50:
        base_eps = r_port_hist.dropna().to_numpy()
        base_eps = base_eps[np.isfinite(base_eps)]
        if base_eps.size:
            p1, p99 = np.percentile(base_eps, [1, 99])
            base_eps = np.clip(base_eps, p1, p99)
        sigma_fallback = np.std(base_eps) if base_eps.size else 0.0005
    else: base_eps, sigma_fallback = np.array([]), 0.0005
    pieces, idx_months = [], monthly_pesos.index.to_list()
    for i in range(len(idx_months) - 1):
        t0, t1, v0, v1 = idx_months[i], idx_months[i + 1], float(monthly_pesos.iloc[i]), float(monthly_pesos.iloc[i + 1])
        try: bdays = pd.bdate_range(start=t0, end=t1, closed="right")
        except Exception: bdays = pd.bdate_range(start=t0, end=t1)[1:]
        if len(bdays) == 0:
            pieces.append(pd.Series([v1], index=[t1]))
            continue
        G_target = v1 / max(v0, 1e-12)
        if base_eps.size: eps = rng.choice(base_eps, size=len(bdays), replace=True)
        else: eps = rng.normal(loc=0.0, scale=sigma_fallback, size=len(bdays))
        P = float(np.prod(1.0 + eps))
        if P <= 0: eps, P = np.full(len(bdays), 0.0), 1.0
        q_day = (G_target / P) ** (1.0 / len(bdays))
        vals = v0 * np.cumprod((1.0 + eps) * q_day)
        vals[-1] = v1
        pieces.append(pd.Series(vals, index=bdays))
    hist_daily = pd.concat(pieces).sort_index()
    first_month = monthly_pesos.index[0]
    if first_month not in hist_daily.index: hist_daily = pd.concat([pd.Series([float(monthly_pesos.iloc[0])], index=[first_month]), hist_daily])
    return hist_daily

def _calculate_40_year_projection(start_amount_pesos: float, annual_return: float, afore: str, start_year: int = 2026, num_years: int = 40) -> pd.DataFrame:
    """Calcula la proyección con crecimiento de aportaciones lineal sobre una base."""
    afore_upper = afore.upper()
    if afore_upper not in PROJECTION_40Y_PARAMS:
        return pd.DataFrame()

    params = PROJECTION_40Y_PARAMS[afore_upper]
    projection_data = []
    
    total_value = start_amount_pesos
    base_collection = params.get(f"{start_year}_anual", 0)
    current_collection = base_collection

    for i in range(num_years):
        year = start_year + i
        
        # Calcular rendimiento del año
        rendimiento_anual = total_value * annual_return
        
        # Aplicar rendimiento y aportación
        total_value = (total_value * (1 + annual_return)) + current_collection
        
        # Determinar la tasa de crecimiento para el año *actual*
        if year <= 2031:
            growth_rate = 0.37
        elif 2032 <= year <= 2044:
            growth_rate = 0.24
        else:
            growth_rate = 0.06
        
        projection_data.append({
            "Año": year, 
            "Monto Total": total_value,
            "Aportación": current_collection,
            "Rendimiento": rendimiento_anual,
            "Tasa Crec. Aportación": growth_rate,
        })
        
        # CORRECCIÓN: Calcular la aportación para el siguiente año ANTES de terminar el ciclo
        # Si no es el último año, calcula el siguiente incremento.
        if i < num_years - 1:
            next_year = year + 1
            if next_year <= 2031:
                next_growth_rate = 0.37
            elif 2032 <= next_year <= 2044:
                next_growth_rate = 0.24
            else:
                next_growth_rate = 0.06
            
            incremento = base_collection * next_growth_rate
            current_collection += incremento
        
    return pd.DataFrame(projection_data)


# ---------------- UI ----------------
def render():
    siefore_name = st.session_state.get("siefore_selected", "Básica Inicial")
    afore_name = st.session_state.get("afore_selected", "—")
    opt_caps, risk_limits, siefore_name_opt, afore_name_opt = _fallback_caps_and_limits()
    try:
        opt_caps, risk_limits, siefore_name_opt, afore_name_opt = optmod._caps_from_consar()
        siefore_name, afore_name = siefore_name_opt, afore_name_opt
    except Exception: pass
    opt_tbl, risk = _pull_latest_optimization(force_refresh=True)
    if opt_tbl is None or len(opt_tbl) == 0:
        st.warning("No encontré 'optimization_table'. Abre Portafolio/Optimization y selecciona activos.")
        return
    try:
        name2ticker = _name_to_ticker_map()
        tickers_to_load = [name2ticker.get(str(n).strip()) for n in opt_tbl["nombre"].tolist() if name2ticker.get(str(n).strip())]
        preload_hist_5y_daily(tickers_to_load)
    except Exception: pass
    try:
        tbl_A, targets = _modelA_table_and_targets(opt_tbl)
    except Exception as e:
        st.error(f"Error al construir tabla del Modelo A: {e}"); return
    if tbl_A.empty:
        st.info("No se pudo construir la tabla (faltan datos)."); return
    
    net_asset_value, latest_date = consar_limits.get_latest_net_asset_value(siefore_name, afore_name)
    if net_asset_value is None:
        st.error(f"No se encontraron Activos Netos (MDP) para **{afore_name}** en la SIEFORE **{siefore_name}**."); return
    
    TODAY_AMOUNT_LOCAL_PESOS = net_asset_value 

    try:
        df_a, _ = consar_limits._si_load_data() 
        cohort = siefore_name.split()[-1] if siefore_name else None
        col_name = f"{afore_name} {cohort}"
        hist_values_pesos = df_a.set_index("Fecha")[col_name].dropna().sort_index()
    except Exception:
        st.error(f"Error al cargar el histórico de activos netos para {afore_name} / {siefore_name}."); return
    if hist_values_pesos.empty or len(hist_values_pesos) < 2: cagr_hist = sigma_annual = max_dd = 0.0
    else:
        hist_values_mdp = hist_values_pesos / MILLION
        hist_monthly = hist_values_mdp.resample('MS').first().dropna()
        years = (len(hist_monthly) - 1) / 12.0 if len(hist_monthly) > 1 else 0.0
        cagr_hist = ((hist_monthly.iloc[-1] / hist_monthly.iloc[0]) ** (1.0 / max(years, 1e-9))) - 1.0 if years > 0 else 0.0
        r_m = hist_monthly.pct_change().dropna()
        sigma_annual = float(r_m.std(ddof=1)) * np.sqrt(12.0) if len(r_m) > 1 else 0.0
        drawdown = hist_values_mdp / hist_values_mdp.cummax() - 1.0
        max_dd = float(drawdown.min())
    
    r_by_name, r_port_hist, idx_hist, weights = _portfolio_history(opt_tbl)
    hist_monthly_for_densify = hist_values_pesos.resample('MS').first().dropna()
    hist_values_daily_pesos = _densify_monthly_to_bdays(hist_monthly_for_densify, r_port_hist=r_port_hist)
    
    if hist_values_daily_pesos is None or len(hist_values_daily_pesos) == 0:
        st.info("Aún no hay histórico densificado para graficar."); return
    
    hist_last_date = pd.to_datetime(hist_values_daily_pesos.index.max())
    future_dates, paths = _simulate_paths(r_by_name, idx_hist, weights, targets, steps=STEPS_FWD, start_date=hist_last_date)
    
    paths_pesos = {k: TODAY_AMOUNT_LOCAL_PESOS * v for k, v in paths.items()}

    st.caption(f"Proyección basada en la **SIEFORE {siefore_name}** de **{afore_name}** — Activos Netos: **{_si_fmt_number(TODAY_AMOUNT_LOCAL_PESOS)}**")
    
    mu_port_exact = float(targets["N"])
    ret_P = (paths_pesos["Pesimista"].iloc[-1] / TODAY_AMOUNT_LOCAL_PESOS) - 1.0 if "Pesimista" in paths_pesos and TODAY_AMOUNT_LOCAL_PESOS > 0 else np.nan
    ret_O = (paths_pesos["Optimista"].iloc[-1] / TODAY_AMOUNT_LOCAL_PESOS) - 1.0 if "Optimista" in paths_pesos and TODAY_AMOUNT_LOCAL_PESOS > 0 else np.nan
    
    # Todas las métricas en una sola fila
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Escenario Pesimista", "—" if np.isnan(ret_P) else f"{ret_P*100:,.2f}%")
    c2.metric("Escenario Neutral", f"{mu_port_exact*100:,.2f}%")
    c3.metric("Escenario Optimista", "—" if np.isnan(ret_O) else f"{ret_O*100:,.2f}%")
    c4.metric("Salario Mínimo", "6.5%")
    c5.metric("Inflación", "3.7%")
    c6.metric("Cuotas", "12.6%", delta="2027-2031", delta_color="off")
    c7.metric("Trayectoria Salarial", "18%", delta="2027-2044", delta_color="off")

    df_40y = pd.DataFrame()
    afore_key_upper = afore_name.upper()
    if afore_key_upper in PROJECTION_40Y_PARAMS:
        df_40y = _calculate_40_year_projection(
            start_amount_pesos=TODAY_AMOUNT_LOCAL_PESOS,
            annual_return=targets["N"],  # Usar el rendimiento neutral
            afore=afore_name,
            start_year=2026
        )

    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        st.subheader("Proyección a 40 años")
        if not df_40y.empty:
            # NO incluir 2025, iniciar desde 2026
            summary_data = []
            
            # Filtrar solo años múltiplos de 5 desde 2030, PERO también incluir 2026
            filtered_df = df_40y[((df_40y['Año'] >= 2030) & (df_40y['Año'] % 5 == 0)) | (df_40y['Año'] == 2026)]
            
            for _, row in filtered_df.iterrows():
                monto_mdp = row['Monto Total'] / MILLION
                aportacion_mdp = row['Aportación'] / MILLION
                rendimiento_mdp = row['Rendimiento'] / MILLION
                summary_data.append({
                    'Año': row['Año'], 
                    'Monto (MDP)': f"${monto_mdp:,.2f}",
                    'Aportación del Año (MDP)': f"${aportacion_mdp:,.2f}",
                    'Rendimiento del Año (MDP)': f"${rendimiento_mdp:,.2f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            st.dataframe(
                summary_df,
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info(f"No hay datos de proyección a 40 años para '{afore_name}'.")

    with right:
        view_choice = st.segmented_control(
            label="Selecciona la vista de proyección:",
            options=["Proyección a 1 Año", "Proyección a 40 Años"],
        )

        if view_choice == "Proyección a 1 Año":
            
            df_hist = pd.DataFrame({"fecha": pd.to_datetime(hist_values_daily_pesos.index), "valor_pesos": hist_values_daily_pesos.values, "serie": "Histórico"})
            dfs = [df_hist] + [pd.DataFrame({"fecha": pd.to_datetime(future_dates), "valor_pesos": np.asarray(s, dtype=float), "serie": scen}) for scen, s in paths_pesos.items() if s is not None and len(s) > 0]
            df_all = pd.concat(dfs, ignore_index=True).dropna(subset=["valor_pesos"])
            if not df_all.empty:
                df_all["valor_mdp"] = df_all["valor_pesos"] / MILLION
                base = alt.Chart(df_all).mark_line().encode(x=alt.X("fecha:T", title="Fecha"), y=alt.Y("valor_mdp:Q", title="Valor del portafolio (MDP)", axis=alt.Axis(format="~s")), color=alt.Color("serie:N", title="", sort=["Histórico", "Pesimista", "Neutral", "Optimista"], legend=alt.Legend(orient="bottom-right")), tooltip=[alt.Tooltip("fecha:T", title="Fecha"), alt.Tooltip("valor_mdp:Q", title="Valor (MDP)", format=",.2f")])
                rule = alt.Chart(pd.DataFrame({"fecha":[hist_last_date]})).mark_rule(strokeDash=[4,4], opacity=0.6).encode(x="fecha:T")
                st.altair_chart(alt.layer(base, rule).properties(height=440).configure_view(fill="transparent", strokeOpacity=0).configure_axis(grid=True, gridColor="#FFFFFF", gridOpacity=0.12, domain=False, labelColor="#FFFFFF", titleColor="#FFFFFF"), use_container_width=True)
        
        else: # Proyección a 40 Años
            mu_port_exact = float(targets["N"])
            if not df_40y.empty:
                df_40y['Monto (BDP)'] = df_40y['Monto Total'] / BILLON
                df_40y['Aportación (BDP)'] = df_40y['Aportación'] / BILLON
                
                chart_40y = alt.Chart(df_40y).mark_area(
                    line={'color':'#2ca02c'},
                    color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color='#2ca02c', offset=1)], x1=1, x2=1, y1=1, y2=0),
                    opacity=0.7
                ).encode(
                    x=alt.X('Año:O', title='Año', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Monto (BDP):Q', title='Monto del Fondo (Billones de Pesos)', axis=alt.Axis(format='~s')),
                    tooltip=[
                        alt.Tooltip('Año:O'),
                        alt.Tooltip('Monto (BDP):Q', title='Monto Total (BDP)', format=',.2f'),
                        alt.Tooltip('Aportación (BDP):Q', title='Aportación (BDP)', format=',.2f'),
                        alt.Tooltip('Tasa Crec. Aportación:Q', title='Crec. Aportación Anual', format='.2%')
                    ]
                ).properties(
                    height=440, title=f"Basado en Rendimiento Neutral ({mu_port_exact:.2%})"
                ).configure_title(fontSize=14, anchor='middle', dy=-10).configure_view(fill="transparent", strokeOpacity=0)
                
                st.altair_chart(chart_40y, use_container_width=True)
            else:
                st.info(f"No hay datos de proyección a 40 años para '{afore_name}'.")