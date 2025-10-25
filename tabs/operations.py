# tabs/operations.py
from __future__ import annotations

# ==== Standard lib ====
from datetime import date, datetime
from typing import Dict, Tuple
import math

# ==== Third-party ====
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt

# ==== Local modules ====
from tabs.ledger import init_state, build_initial_portfolio
from tabs.aportaciones import (
    compute_contribution_recommendation,
    apply_user_contribution_weights,
    _price_at_date_mxn as _px_mxn_at,
)
from tabs import runtime
from tabs import ledger

from tabs.recommendations import (
    mostrar_recomendaciones_rv,
    mostrar_structured_note_compacto,
    mostrar_fixed_income_compacto,
)


# =========================
# Helpers generales de UI / formato
# =========================
def _today() -> date:
    return datetime.now().date()

def _fmt_mdp(x: float | None, d: int = 2) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        return f"{float(x)/1_000_000:.{d}f}"
    except Exception:
        return "—"

def _fmt_mdp_money(x: float | None, d: int = 2) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        return f"${float(x)/1_000_000:,.{d}f}"
    except Exception:
        return "—"

def _fmt_pct_pp(x: float | None, d: int = 2) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        return f"{float(x):.{d}f}%"
    except Exception:
        return "—"

def _parse_money_input(s: str) -> float:
    if s is None:
        return 0.0
    s = str(s).strip().replace(",", "")
    try:
        return max(0.0, float(s) if s else 0.0)
    except Exception:
        return 0.0

def _fmt_date_only(ts) -> str:
    try:
        return pd.to_datetime(ts).date().strftime("%d/%m/%Y")
    except Exception:
        return "—"

def _fmt_money_abs(x: float | int | None) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "—"

def _fmt_int(x) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "—"

# === Cache de recomendaciones (helper) ===
def _recos_cache_key(categoria_code: str, fecha_efectiva: date) -> tuple[str, str]:
    return (str(categoria_code).strip().upper(), str(fecha_efectiva))

# =========================
# Tabla transpuesta “bonita” (Aportaciones)
# =========================
def _render_transposed_contrib(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty or "clase" not in df_in.columns:
        return df_in
    df = df_in.copy()

    for c in ["w_inicial_%", "w_aportacion_%", "w_final_%"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["monto_inicial_$", "aportacion_$", "valor_final_$", "V_t"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "monto_inicial_$" not in df.columns:
        if "V_t" in df.columns:
            df["monto_inicial_$"] = df["V_t"]
        elif {"valor_final_$", "aportacion_$"}.issubset(df.columns):
            df["monto_inicial_$"] = (
                pd.to_numeric(df["valor_final_$"], errors="coerce").fillna(0.0)
                - pd.to_numeric(df["aportacion_$"], errors="coerce").fillna(0.0)
            )

    must_have = {"w_inicial_%","monto_inicial_$","w_aportacion_%","aportacion_$","w_final_%","valor_final_$"}
    if not must_have.issubset(df.columns):
        return df_in

    expected_order = [
        "w_inicial_%","monto_inicial_$","w_aportacion_%","aportacion_$","w_final_%","valor_final_$",
    ]
    cols_present = ["clase"] + [c for c in expected_order if c in df.columns]
    wide = df[cols_present].set_index("clase").T

    pretty = {
        "w_inicial_%": "Peso inicial al aportar",
        "monto_inicial_$": "Monto inicial (antes de aportar)",
        "w_aportacion_%": "Composición de la aportación",
        "aportacion_$": "Monto de la aportación",
        "w_final_%": "Peso final",
        "valor_final_$": "Valor final",
    }
    wide.index = [pretty.get(i, i) for i in wide.index]

    perc_rows = {pretty["w_inicial_%"], pretty["w_aportacion_%"], pretty["w_final_%"]}
    money_rows = {pretty["monto_inicial_$"], pretty["aportacion_$"], pretty["valor_final_$"]}

    totals_num = []
    for idx in wide.index:
        is_money = idx in money_rows
        col_name_raw = [k for k,v in pretty.items() if v == idx][0]
        if is_money and col_name_raw in df_in.columns:
            total_val = float(pd.to_numeric(df_in[col_name_raw], errors="coerce").sum())
        else:
            total_val = float(pd.to_numeric(wide.loc[idx], errors="coerce").sum())
        totals_num.append(total_val)
    wide["Total"] = totals_num

    wide.insert(0, "", wide.index)

    def fmt_pct(x):
        try: return f"{float(x):,.2f}%"
        except Exception: return "—"

    def fmt_money(x):
        try: return f"${float(x):,.2f}"
        except Exception: return "—"

    for idx in list(wide.index):
        if idx in perc_rows:
            row = wide.loc[idx].copy()
            row.iloc[1:-1] = row.iloc[1:-1].map(fmt_pct)
            row.iloc[-1]   = fmt_pct(100.0)
            wide.loc[idx] = row
        elif idx in money_rows:
            row = wide.loc[idx].copy()
            row.iloc[1:] = row.iloc[1:].map(fmt_money)
            wide.loc[idx] = row
        else:
            row = wide.loc[idx].copy()
            row.iloc[1:] = row.iloc[1:].map(fmt_money)
            wide.loc[idx] = row

    return wide.reset_index(drop=True)


# =========================
# Helpers de rango/serie (gráfica y performance)
# =========================
def _start_for_chart(ss, fallback: date) -> date:
    """Elige un start_date robusto para la gráfica (mínimo entre fallback y primera fecha del ledger)."""
    try:
        led = ledger.get_ledger(ss)
        if isinstance(led, pd.DataFrame) and not led.empty and "ts" in led.columns:
            t0 = pd.to_datetime(led["ts"]).min().date()
            return min(fallback, t0)
    except Exception:
        pass
    return fallback

def _period_bounds_for(d: date, kind: str) -> Tuple[date | None, date]:
    if kind == "MTD":
        start = d.replace(day=1)
    elif kind == "QTD":
        q_start_month = ((d.month - 1)//3)*3 + 1
        start = date(d.year, q_start_month, 1)
    elif kind == "YTD":
        start = date(d.year, 1, 1)
    else:  # ITD
        start = None
    return start, d

def _ret_over(ts: pd.Series, start_d: date | None, end_d: date) -> float:
    try:
        s = ts.dropna()
        if start_d:
            s = s.loc[(s.index >= pd.Timestamp(start_d)) & (s.index <= pd.Timestamp(end_d))]
        else:
            s = s.loc[s.index <= pd.Timestamp(end_d)]
        if s.size >= 2 and s.iloc[0] > 0:
            return float(s.iloc[-1] / s.iloc[0] - 1.0)
    except Exception:
        pass
    return math.nan

def _net_contributions_from_ledger(ss) -> float:
    """Depósitos - Retiros - Fees (si fees salen como negativos también cuadra)."""
    try:
        led = ledger.get_ledger(ss)
        if isinstance(led, pd.DataFrame) and not led.empty:
            cash = pd.to_numeric(led.get("cash", 0.0), errors="coerce").fillna(0.0)
            return float(cash.sum())
    except Exception:
        pass
    return 0.0

def _ticker_to_name_map(ss) -> dict[str, str]:
    pos_df = ss.get("ops_positions", pd.DataFrame())
    if not pos_df.empty and {"ticker", "activo"} <= set(pos_df.columns):
        return pos_df.dropna(subset=["ticker"]).set_index("ticker")["activo"].to_dict()
    return {}

def _norm_class_for_drift(c: str) -> str:
    c = str(c)
    if c in ("Nota estructurada", "SN"):
        return "SN"
    return c


# =========================
# Render principal
# =========================
def render():
    runtime.ensure_ops_keys()
    init_state(st.session_state)
    ss = st.session_state

    CLASS_MAP = {
        "RV_EXT": "Renta Variable Internacional",
        "RV_MX": "Renta Variable Nacional",
        "FIBRA_MX": "FIBRAS (REITs)",
        "FI_CORP_MX": "Deuda Privada Nacional",
        "FI_GOB_MX": "Deuda Gubernamental",
        "FI_EXT": "Deuda Internacional",
        "FX": "Divisas",
        "Nota estructurada": "Notas Estructuradas",
        "SN": "Notas Estructuradas",
        "FI": "Renta Fija (General)",
    }

    df_opt: pd.DataFrame = ss.get("optimization_table", pd.DataFrame())
    fi_rows = ss.get("fi_rows", []) or []
    eq_sel  = ss.get("equity_selection", {}) or {}
    fx_sel  = ss.get("fx_selection", {}) or {}

    siefore = ss.get("siefore_selected")
    afore   = ss.get("afore_selected")
    start   = runtime.get_ops_started_at() or ss.get("ops_start_date") or _today()
    cap     = ss.get("ops_seed_capital") or 0.0
    fee_pp  = ss.get("ops_mgmt_fee_pp")
    act_as_of = runtime.get_act_as_of() or _today()   # 👈 FECHA DE CORTE GLOBAL

    # Cabecera fija
    top_cols = st.columns([1, 1, 1, 1, 1, 1])
    top_cols[0].metric("SIEFORE", siefore or "—")
    top_cols[1].metric("AFORE", afore or "—")
    top_cols[2].metric("Capital Inicial", _fmt_mdp_money(cap, 2))
    top_cols[3].metric("Comisión de la Afore", _fmt_pct_pp(fee_pp, 2))
    top_cols[4].metric("Inicio", (start.strftime("%d/%m/%Y") if hasattr(start, "strftime") else str(start)))
    top_cols[5].metric("Modo", "Operando" if ss.get("ops_operating") else "Diseño")

    # =========================
    # DISEÑO
    # =========================
    if not ss.get("ops_operating"):
        c1, c2, c3 = st.columns([1.0, 1.0, 1.0])

        c1.date_input("Fecha de inicio de operación", key="ops_start_date",
                      value=ss.get("ops_start_date") or _today())

        cap_str = c2.text_input("Capital Inicial", value=ss.get("ops_seed_capital_str") or "")
        cap_num = _parse_money_input(cap_str)
        ss["ops_seed_capital"] = cap_num
        ss["ops_seed_capital_str"] = f"{cap_num:,.2f}"

        c3.number_input("Comisión de la AFORE",
                        value=float(ss.get("ops_mgmt_fee_pp") or 0.0),
                        min_value=0.0, step=0.01, format="%.4f", key="ops_mgmt_fee_pp")

        if df_opt.empty:
            st.warning("⚠️ La tabla de optimización está vacía.")
        else:
            df_opt_view = df_opt.copy()
            if not df_opt_view.empty and "clase" in df_opt_view.columns:
                df_opt_view["Clase"] = df_opt_view["clase"].astype(str).replace(CLASS_MAP)
                mask_fx = df_opt_view["clase"].astype(str).eq("FX")
                name_to_pair_map = {}
                for key, meta in (fx_sel or {}).items():
                    short_pair = str(meta.get("pair") or key).strip()
                    long_name_or_ticker = str(meta.get("name") or key).strip()
                    matching_rows = df_opt_view[
                        mask_fx & df_opt_view['nombre'].str.contains(long_name_or_ticker, case=False, na=False, regex=False)
                    ]
                    if not matching_rows.empty:
                        for original_name in matching_rows['nombre'].unique():
                            name_to_pair_map[original_name] = short_pair
                if name_to_pair_map:
                    df_opt_view.loc[mask_fx, 'nombre'] = df_opt_view.loc[mask_fx, 'nombre'].map(name_to_pair_map).fillna(df_opt_view.loc[mask_fx, 'nombre'])

            cols_to_show = ["nombre", "Clase", "Peso (%)", "mu_pct", "vol_pct", "sharpe"]
            df_display = df_opt_view[[c for c in cols_to_show if c in df_opt_view.columns]].copy()
            df_display.rename(columns={
                "Peso (%)": "Composición Inicial", "mu_pct": "μ anual (%)",
                "vol_pct": "σ anual (%)", "sharpe": "Sharpe", "nombre": "Activo"
            }, inplace=True)

            if "Composición Inicial" in df_display.columns:
                df_display["Composición Inicial"] = df_display["Composición Inicial"].apply(lambda x: f"{x:,.2f}%" if pd.notna(x) else "—")
            for col in ["μ anual (%)", "σ anual (%)"]:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:,.2f}%" if pd.notna(x) else "—")
            if "Sharpe" in df_display.columns:
                df_display["Sharpe"] = df_display["Sharpe"].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "—")

            donut_data = df_opt_view.groupby("Clase")["Peso (%)"].sum().reset_index()
            donut_data.columns = ["Clase", "Peso (%)"]

            try:
                fig = px.pie(donut_data, values='Peso (%)', names='Clase', hole=0.5)
                fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=0.5)))
                fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=450)

                colL_viz, colR_viz = st.columns([0.5, 0.5])
                with colL_viz:
                    st.subheader("Detalle de Composición por Instrumento")
                    st.dataframe(df_display, hide_index=True, use_container_width=True)
                with colR_viz:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.dataframe(df_display, hide_index=True, use_container_width=True)

        btn_col_design, _ = st.columns([0.5, 0.5])
        if btn_col_design.button("🚀 Iniciar operación", type="primary", use_container_width=True):
            cap_num_check = _parse_money_input(ss.get("ops_seed_capital_str"))
            if cap_num_check <= 0:
                st.error("Necesitas un capital inicial positivo.")
            else:
                from tabs.ledger import ensure_ledger_initialized, add_transaction, _bizdays_index, _safe_adj_close_series_init, _note_series_bs
                ensure_ledger_initialized(ss)
                start_dt = ss.get("ops_start_date") or _today()

                pos_df, ts_df = build_initial_portfolio(
                    start_dt=start_dt,
                    seed_capital_pesos=ss.get("ops_seed_capital") or 0.0,
                    fee_pp=float(ss.get("ops_mgmt_fee_pp") or 0.0),
                    df_opt=df_opt, fi_rows=fi_rows, eq_sel=eq_sel, fx_sel=fx_sel,
                    end_dt=act_as_of,
                )

                add_transaction(ss, ts=start_dt, kind="DEPOSIT",
                                cash=float(ss.get("ops_seed_capital") or 0.0), note="Capital inicial")

                if "synthetic_prices" not in ss:
                    ss["synthetic_prices"] = {}

                bdays = _bizdays_index(start_dt, act_as_of)
                from tabs.yf_store import get_hist_5y
                usd_mxn_df = get_hist_5y("MXN=X")
                usd_mxn_series = _safe_adj_close_series_init(usd_mxn_df).reindex(bdays, method="pad").dropna()

                synthetic_counter = 1
                if not pos_df.empty:
                    for _, r in pos_df.iterrows():
                        activo = str(r.get("activo", ""))
                        clase = str(r.get("clase", ""))
                        tkr = str(r.get("ticker", "")).strip()
                        qty = float(r["qty"])
                        px_inicial = float(r["px_inicial"])

                        if tkr and not tkr.startswith("SYNTH_"):
                            try:
                                df = get_hist_5y(tkr)
                                s = _safe_adj_close_series_init(df).reindex(bdays, method="pad").dropna()
                                if not s.empty:
                                    if clase == "RV_EXT" and not usd_mxn_series.empty:
                                        s = s.multiply(usd_mxn_series, fill_value=np.nan).dropna()
                                    if clase == "FX":
                                        for key, meta in (fx_sel or {}).items():
                                            fx_tkr = str(meta.get("yahoo_ticker", "")).strip()
                                            if tkr == fx_tkr:
                                                if meta.get("inverse", False):
                                                    s = 1.0 / s
                                                tkr_upper = tkr.upper().replace("=X", "")
                                                if "MXN" not in tkr_upper and tkr_upper != "MXN" and not usd_mxn_series.empty:
                                                    s = s.multiply(usd_mxn_series, fill_value=np.nan).dropna()
                                                break
                                    ss["synthetic_prices"][tkr] = {
                                        "activo": activo, "clase": clase, "px_inicial": px_inicial,
                                        "start_date": start_dt, "price_series": s
                                    }
                            except Exception:
                                pass
                        else:
                            if not tkr:
                                if clase in ["FI_GOB_MX", "FI_CORP_MX", "FI_EXT"]:
                                    tkr = f"SYNTH_FI_{synthetic_counter:03d}"
                                elif clase == "Nota estructurada":
                                    tkr = f"SYNTH_SN_{synthetic_counter:03d}"
                                else:
                                    tkr = f"SYNTH_{synthetic_counter:03d}"
                                synthetic_counter += 1

                            if clase in ["FI_GOB_MX", "FI_CORP_MX", "FI_EXT"]:
                                mu_pct = 0.0
                                for fi_row in (fi_rows or []):
                                    bono_name = str(fi_row.get("Bono", "")).strip()
                                    if activo and bono_name and (activo.lower() in bono_name.lower() or bono_name.lower() in activo.lower()):
                                        try:
                                            mu_pct = float(fi_row.get("Rend (%)", 0.0))
                                        except Exception:
                                            mu_pct = 0.0
                                        break
                                mu = (mu_pct / 100.0) if mu_pct else 0.0
                                r_daily = (1.0 + mu) ** (1.0 / 252.0) - 1.0
                                days_count = len(bdays)
                                grow = (1.0 + r_daily) ** np.arange(days_count)
                                price_series = pd.Series(px_inicial * grow, index=bdays)
                            elif clase == "Nota estructurada":
                                sn_meta = None
                                for sn in (ss.get("structured_notes", []) or []):
                                    sn_name = str(sn.get("nombre") or sn.get("name") or "").strip()
                                    if sn_name.lower() == str(activo).lower():
                                        sn_meta = sn; break

                                if sn_meta:
                                    try:
                                        underlying_tkr = str(sn_meta.get("yahoo_ticker") or sn_meta.get("ticker") or sn_meta.get("underlying_ticker") or "").strip()
                                        if underlying_tkr:
                                            s_u = _safe_adj_close_series_init(get_hist_5y(underlying_tkr)).reindex(bdays, method="pad").dropna()
                                        else:
                                            s_u = pd.Series(dtype=float)
                                        if not s_u.empty:
                                            S0 = float(sn_meta.get("spot", sn_meta.get("S0", px_inicial)) or px_inicial)
                                            K1 = float(sn_meta.get("K1", 0.0) or 0.0)
                                            K2 = float(sn_meta.get("K2", 0.0) or 0.0)
                                            days_total = int(sn_meta.get("plazo_d", sn_meta.get("tenor_d", 360)) or 360)
                                            r_pct = float(sn_meta.get("r_pct", 7.70) or 7.70)
                                            sigma_pct = float(sn_meta.get("vol_pct", 25.0) or 25.0)
                                            from tabs.ledger import _note_series_bs  # local import
                                            price_series = _note_series_bs(
                                                S_hist=s_u, S0=S0, K1=K1, K2=K2, r_pct=r_pct,
                                                sigma_pct=sigma_pct, days_total=days_total, bdays=bdays, price_unit=10.0
                                            )
                                            try:
                                                first_price_calculated = float(price_series.iloc[0])
                                                if abs(first_price_calculated - px_inicial) > 1.0:
                                                    scale_factor = px_inicial / first_price_calculated
                                                    price_series = price_series * scale_factor
                                            except Exception:
                                                pass
                                        else:
                                            price_series = pd.Series(px_inicial, index=bdays)
                                    except Exception:
                                        price_series = pd.Series(px_inicial, index=bdays)
                                else:
                                    price_series = pd.Series(px_inicial, index=bdays)
                            else:
                                price_series = pd.Series(px_inicial, index=bdays)

                            ss["synthetic_prices"][tkr] = {
                                "activo": activo, "clase": clase, "px_inicial": px_inicial,
                                "start_date": start_dt, "price_series": price_series
                            }

                        ledger.add_transaction(ss, ts=start_dt, kind="BUY", symbol=tkr, qty=qty, px=px_inicial, name=activo, note=f"Compra inicial: {activo}")

                ss["ops_positions"] = pos_df
                ss["ops_timeseries"] = ts_df
                ss["ops_operating"] = True
                runtime.set_ops_started_at(ss.get("ops_start_date") or _today())
                runtime.set_act_as_of(act_as_of)
                ss["ops_snapshot"] = {
                    "start_date": (ss.get("ops_start_date") or _today()).isoformat(),
                    "seed_capital": float(ss.get("ops_seed_capital") or 0.0),
                    "mgmt_fee_pp": float(ss.get("ops_mgmt_fee_pp") or 0.0),
                    "seed_source_date": (
                        ss.get("ops_seed_source_date").isoformat()
                        if hasattr(ss.get("ops_seed_source_date"), "isoformat")
                        else (ss.get("ops_seed_source_date") or None)
                    ),
                    "siefore": siefore, "afore": afore,
                }
                st.rerun()

        return  # no renderizar lo de abajo mientras está en DISEÑO

    # =========================
    # OPERANDO
    # =========================
    last_act_as_of = ss.get("ops_last_act_as_of")
    need_rebuild = (last_act_as_of is None or str(last_act_as_of) != str(act_as_of) or
                    ss.get("ops_positions", pd.DataFrame()).empty or ss.get("ops_timeseries", pd.DataFrame()).empty)

    pos_df = ss.get("ops_positions", pd.DataFrame()).copy()
    ts_df  = ss.get("ops_timeseries", pd.DataFrame()).copy()
    if need_rebuild:
        try:
            snapshot = ss.get("ops_snapshot", {})
            start_dt_iso = snapshot.get("start_date")
            start_dt_runtime = runtime.get_ops_started_at()
            start_dt = (
                start_dt_runtime
                or (datetime.fromisoformat(start_dt_iso).date() if start_dt_iso else None)
                or ss.get("ops_start_date")
                or _today()
            )
            if "ops_start_date" not in ss or ss.get("ops_start_date") is None:
                ss["ops_start_date"] = start_dt
            pos_df, ts_df = build_initial_portfolio(
                start_dt=start_dt,
                seed_capital_pesos=snapshot.get("seed_capital") or (ss.get("ops_seed_capital") or 0.0),
                fee_pp=float(snapshot.get("mgmt_fee_pp") or (ss.get("ops_mgmt_fee_pp") or 0.0)),
                df_opt=ss.get("optimization_table", pd.DataFrame()),
                fi_rows=ss.get("fi_rows", []) or [],
                eq_sel=ss.get("equity_selection", {}) or {},
                fx_sel=ss.get("fx_selection", {}) or {},
                end_dt=act_as_of,
            )
            ss["ops_positions"] = pos_df
            ss["ops_timeseries"] = ts_df
            ss["ops_last_act_as_of"] = str(act_as_of)
        except Exception as e:
            st.error(f"❌ Error al reconstruir portafolio: {e}")

# ===== Performance y Transacciones (sin tabs) =====
    try:
        from tabs.ledger import rebuild_portfolio_from_ledger

        act_as_of_perf = runtime.get_act_as_of() or _today()
        start_dt_perf  = runtime.get_ops_started_at() or ss.get("ops_start_date") or _today()
        start_for_chart = _start_for_chart(ss, start_dt_perf) if "_start_for_chart" in globals() else start_dt_perf

        # ========================
        # Serie de valor del portafolio
        # ========================
        _, ts_led = rebuild_portfolio_from_ledger(ss, start_date=start_for_chart, end_date=act_as_of_perf)
        if not ts_led.empty and "total_value" in ts_led.columns:
            port = ts_led["total_value"].dropna().copy().sort_index()
            if port.empty or port.size < 2:
                st.info("Serie insuficiente para evaluar performance.")
                st.stop()

            # ========================
            # Flujos EXTERNOS por día (DEPOSIT/WITHDRAW/FEES)
            # ========================
            flows = pd.Series(0.0, index=port.index)
            try:
                led_df = ledger.get_ledger(ss)
                if isinstance(led_df, pd.DataFrame) and not led_df.empty and {"ts","kind","cash"} <= set(led_df.columns):
                    cf = led_df.copy()
                    cf["ts"]   = pd.to_datetime(cf["ts"], errors="coerce")
                    cf["cash"] = pd.to_numeric(cf.get("cash", 0.0), errors="coerce").fillna(0.0)

                    EXTERNAL_KINDS = {"DEPOSIT", "WITHDRAW", "FEE", "MGMT_FEE", "ADMIN_FEE"}
                    cf = cf[cf["kind"].astype(str).str.upper().isin(EXTERNAL_KINDS)]

                    cf_daily = (
                        cf.assign(day=cf["ts"].dt.tz_localize(None).dt.normalize())
                        .groupby("day")["cash"]
                        .sum(min_count=1)
                    )

                    pidx_norm = pd.to_datetime(port.index).tz_localize(None).normalize()
                    flows = cf_daily.reindex(pidx_norm, fill_value=0.0)
                    flows.index = port.index
            except Exception:
                pass

            # Ignorar flujo del primer día (seed capital) en el TWR
            try:
                first_idx = port.index[0]
                if first_idx in flows.index:
                    flows.loc[first_idx] = 0.0
            except Exception:
                pass

            # ========================
            # Retornos diarios TWR (convención fin de día)
            # r_t = (V_t - CF_t) / V_{t-1} - 1
            # ========================
            rets_twr = ((port - flows) / port.shift(1) - 1.0)
            rets_twr = rets_twr.replace([np.inf, -np.inf], np.nan).dropna()

            # TWR total y anualizado (252 hábiles)
            if rets_twr.size > 0:
                twr_total = float(np.prod(1.0 + rets_twr) - 1.0)
                n = float(rets_twr.size)
                twr_annual = float((1.0 + twr_total) ** (252.0 / n) - 1.0)
            else:
                twr_total = np.nan
                twr_annual = np.nan

            # Vol, mejor/peor día con TWR
            vol_ann   = (rets_twr.std() * np.sqrt(252.0)) if rets_twr.size > 1 else np.nan
            best_day  = float(rets_twr.max()) if rets_twr.size else np.nan
            worst_day = float(rets_twr.min()) if rets_twr.size else np.nan

            # Drawdown sobre índice "sin flujos" (encadena r_t)
            try:
                base = float(port.iloc[0])
                perf_idx = base * (1.0 + rets_twr).cumprod()
                perf_idx = pd.concat([pd.Series({rets_twr.index[0]: base}), perf_idx])
                perf_idx = perf_idx[~perf_idx.index.duplicated(keep="first")]
                roll_max = perf_idx.cummax()
                dd = (perf_idx / roll_max) - 1.0
                max_dd = float(dd.min()) if not dd.empty else np.nan
            except Exception:
                max_dd = np.nan

            st.markdown("---")

            # ========================
            # Métricas
            # ========================
            mcols = st.columns(7)
            mcols[0].metric("Rendimiento Total (TWR)", _fmt_pct_pp(twr_total * 100.0, 2))
            mcols[1].metric("Rendimiento anualizado", _fmt_pct_pp(twr_annual * 100.0, 2))
            mcols[2].metric("Volatilidad Anual", _fmt_pct_pp(vol_ann * 100.0, 2))
            mcols[3].metric("Valor Actual", _fmt_mdp_money(float(port.iloc[-1]), 2))
            mcols[4].metric("Mejor Día", _fmt_pct_pp(best_day * 100.0, 2))
            mcols[5].metric("Peor día", _fmt_pct_pp(worst_day * 100.0, 2))
            mcols[6].metric("Máx. Drawdown", _fmt_pct_pp(max_dd * 100.0, 2))

            

            # ========================
            # LAYOUT: Tabla izquierda + Gráfica derecha
            # ========================
            left_col, right_col = st.columns([1, 1], gap="large")

            # ========================
            # COLUMNA IZQUIERDA: TRANSACCIONES
            # ========================
            with left_col:
                st.subheader("Transacciones")
                
                # Recalcular si hay pendiente
                ledger.ensure_ledger_initialized(ss)
                act_as_of = runtime.get_act_as_of() or _today()
                pending_from = ss.get("_fee_recalc_needed_from")

                if pending_from:
                    try:
                        result = ledger.apply_daily_fees_and_autosell(
                            ss, start_date=pending_from, end_date=act_as_of, debug=False
                        )
                        ss["_fee_recalc_needed_from"] = None
                        if result.get("fees_added", 0) > 0:
                            st.info(
                                f"✅ Se calcularon {result['fees_added']} comisiones diarias "
                                f"(Total: ${result['total_fee']:,.2f})"
                            )
                    except Exception as e:
                        st.warning(f"⚠️ No se pudieron aplicar comisiones: {e}")
                        ss["_fee_recalc_needed_from"] = None

                # Tabla de transacciones
                try:
                    led = ledger.get_ledger(ss)
                    if isinstance(led, pd.DataFrame) and not led.empty:
                        show = led.copy()

                        # Ordenar: fecha asc + tipo (DEPOSIT->BUY->SELL->WITHDRAW)
                        kind_order = {"DEPOSIT": 1, "BUY": 2, "SELL": 3, "WITHDRAW": 4, "FEE": 5, "MGMT_FEE": 6, "ADMIN_FEE": 7}
                        show["_kind_order"] = show["kind"].astype(str).str.upper().map(kind_order).fillna(99)
                        show = show.sort_values(["ts", "_kind_order"], ascending=[True, True])

                        # Campos base para mostrar
                        show["Fecha"] = show["ts"].map(_fmt_date_only)

                        sy2nm = _ticker_to_name_map(ss)
                        symbol_col = "symbol" if "symbol" in show.columns else None

                        name_series = pd.Series("", index=show.index)
                        if "name" in show.columns:
                            name_series = show["name"].astype(str)
                        elif "activo" in show.columns:
                            name_series = show["activo"].astype(str)
                        elif symbol_col:
                            name_series = show[symbol_col].map(lambda s: sy2nm.get(str(s), str(s)))
                        show["Nombre"] = name_series.fillna("")

                        qty_series = pd.to_numeric(show.get("qty", 0), errors="coerce").fillna(0)
                        dep_mask = show.get("kind", "").astype(str).str.upper().isin(["DEPOSIT", "APORT"])
                        show["Cantidad"] = qty_series.where(~dep_mask, np.nan).map(lambda v: "" if pd.isna(v) else _fmt_int(v))

                        px_series = pd.to_numeric(show.get("px", 0.0), errors="coerce").fillna(0.0)
                        show["Precio"] = px_series.map(_fmt_money_abs)

                        # Importe mostrado
                        if "cash" in show.columns:
                            amt = pd.to_numeric(show["cash"], errors="coerce")
                        elif "amount" in show.columns:
                            amt = pd.to_numeric(show["amount"], errors="coerce")
                        else:
                            amt = pd.Series(np.nan, index=show.index)
                        fallback_amt = qty_series.fillna(0.0) * px_series.fillna(0.0)
                        amt = amt.where(amt.notna(), fallback_amt)

                        show.loc[dep_mask, "Nombre"] = ""

                        # WITHDRAW con comisión → "Comisión"
                        withdraw_mask = show.get("kind", "").astype(str).str.upper() == "WITHDRAW"
                        comision_mask = show.get("note", "").astype(str).str.contains(
                            "COMISI[OÓ]N", case=False, na=False, regex=True
                        )
                        show.loc[withdraw_mask & comision_mask, "Nombre"] = "Comisión"

                        # SELL → nombre del activo vendido
                        sell_mask = show.get("kind", "").astype(str).str.upper() == "SELL"
                        for idx in show[sell_mask].index:
                            if not show.loc[idx, "Nombre"]:
                                sym = str(show.loc[idx, "symbol"]) if "symbol" in show.columns else ""
                                if sym:
                                    nombre_activo = sy2nm.get(sym, sym)
                                    if nombre_activo == sym and "name" in show.columns:
                                        nombre_activo = show.loc[idx, "name"]
                                    show.loc[idx, "Nombre"] = nombre_activo

                        # Efectivo disponible
                        kind_u = show.get("kind", pd.Series("", index=show.index)).astype(str).str.upper()
                        cash_col = pd.to_numeric(show.get("cash", np.nan), errors="coerce")

                        fee_candidates = ["fee", "fees", "commission", "commission_mxn", "comision_mxn", "commission_flat", "comm_flat"]
                        fee_mxn = None
                        for c in fee_candidates:
                            if c in show.columns:
                                fee_mxn = pd.to_numeric(show[c], errors="coerce").fillna(0.0)
                                break
                        if fee_mxn is None:
                            fee_mxn = pd.Series(0.0, index=show.index)

                        delta = pd.Series(0.0, index=show.index)
                        external_kinds = {"DEPOSIT", "WITHDRAW", "APORT", "FEE", "MGMT_FEE", "ADMIN_FEE"}

                        is_external = kind_u.isin(external_kinds)
                        if "cash" in show.columns:
                            delta.loc[is_external] = cash_col.loc[is_external].fillna(0.0)
                        else:
                            sign_map = {"DEPOSIT": 1.0, "APORT": 1.0, "WITHDRAW": -1.0, "FEE": -1.0, "MGMT_FEE": -1.0, "ADMIN_FEE": -1.0}
                            signs = kind_u.map(lambda k: sign_map.get(k, 0.0))
                            delta.loc[is_external] = signs.loc[is_external] * amt.loc[is_external].abs().fillna(0.0)

                        is_buy  = kind_u.eq("BUY")
                        is_sell = kind_u.eq("SELL")
                        if "cash" in show.columns:
                            delta.loc[is_buy | is_sell] = cash_col.loc[is_buy | is_sell].fillna(0.0)
                        else:
                            delta.loc[is_buy]  = -(qty_series.loc[is_buy]  * px_series.loc[is_buy]).fillna(0.0) - fee_mxn.loc[is_buy]
                            delta.loc[is_sell] =  +(qty_series.loc[is_sell] * px_series.loc[is_sell]).fillna(0.0) - fee_mxn.loc[is_sell]

                        show["Efectivo disponible"] = delta.cumsum().map(_fmt_money_abs)

                        # Importe final
                        show["Importe"] = amt.map(_fmt_money_abs)

                        # Tipo
                        if "Tipo" not in show.columns:
                            if "kind" in show.columns:
                                show["Tipo"] = show["kind"].astype(str)
                            else:
                                show["Tipo"] = ""

                        # Columnas requeridas
                        required_cols = ["Fecha", "Tipo", "Nombre", "Cantidad", "Precio", "Importe", "Efectivo disponible"]
                        for c in required_cols:
                            if c not in show.columns:
                                show[c] = ""

                        show_final = show[required_cols]
                        st.dataframe(show_final, hide_index=True, use_container_width=True, height=400)

                    else:
                        st.info("Sin actividad registrada en el ledger.")
                except Exception as e:
                    st.error(f"No se pudo cargar la actividad: {e}")

            # ========================
            # COLUMNA DERECHA: GRÁFICA
            # ========================
            with right_col:
                st.subheader("Performance del Portafolio")
                
                # Gráfica (AUM en MDP) - últimos 6 meses
                df_show = port.copy()
                six_months_ago = pd.Timestamp(act_as_of_perf) - pd.DateOffset(months=6)
                if df_show.index.min() < six_months_ago:
                    df_show = df_show.loc[df_show.index >= six_months_ago]

                dfp = df_show.reset_index().rename(columns={"index": "Fecha"})
                dfp["Fecha"] = pd.to_datetime(dfp["Fecha"]).dt.tz_localize(None)
                dfp["MDP"] = dfp.iloc[:, 1] / 1_000_000.0

                chart = (
                    alt.Chart(dfp).mark_line(strokeWidth=2.5, color='#2ca02c')
                    .encode(
                        x=alt.X("Fecha:T", title=""),
                        y=alt.Y("MDP:Q", title="Portafolio (MDP)", scale=alt.Scale(zero=False)),
                        tooltip=[
                            alt.Tooltip("Fecha:T", format="%d/%m/%Y"),
                            alt.Tooltip("MDP:Q", title="Valor (MDP)", format=",.2f"),
                        ],
                    )
                    .properties(height=400)
                )
                st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No hay datos de portafolio para mostrar.")
    except Exception as e:
        st.error(f"Error al calcular performance: {e}")
        import traceback
        st.code(traceback.format_exc())

    # =========================
    # APORTACIONES
    # =========================
    with st.expander("➕ Aportaciones (recomendación por clase)", expanded=True):
        cA, cB = st.columns([1.0, 1.0])
        fecha_aport = cA.date_input(
            "Fecha efectiva de la aportación",
            key="contrib_effective_date",
            value=ss.get("contrib_effective_date") or _today(),
            max_value=_today()
        )
        monto_str = cB.text_input(
            "Monto de aportación",
            key="contrib_amount_str",
            value=f"{float(ss.get('contrib_amount_last', 0.0)):,.2f}" if ss.get("contrib_amount_last") else "",
            help="Ingresa el monto en pesos. Ej: 1,000,000"
        )

        # Defaults (venían del expander avanzado eliminado)
        use_tilt = True
        tilt_alpha = 0.25
        overflow_policy = "pro_rata_targets"

        # ---- Lógica de auto-recalculo + ocultamiento del botón ----
        prev_inputs = ss.get("contrib_prev_inputs", {})
        prev_fecha = prev_inputs.get("fecha")
        prev_monto = prev_inputs.get("monto")
        curr_fecha = str(fecha_aport)
        curr_monto = monto_str

        df_alloc: pd.DataFrame = ss.get("last_contrib_result", pd.DataFrame())
        calc_done = ss.get("contrib_calc_done", False)

        def _do_calculate():
            monto = _parse_money_input(curr_monto)
            ss["contrib_amount_last"] = monto
            ss["last_contrib_result"] = pd.DataFrame()
            ss["last_contrib_meta"] = {}
            ss["snapshot_before_edit"] = {}
            ss["last_applied_constraints"] = {}
            ss["contrib_recalc_counter"] = ss.get("contrib_recalc_counter", 0) + 1
            ss["recs_cache"] = {}
            if monto <= 0:
                st.error("Ingresa un monto positivo.")
                return False
            if df_opt.empty:
                st.error("No hay tabla de optimización.")
                return False
            try:
                res = compute_contribution_recommendation(
                    start_dt=ss.get("ops_start_date") or _today(),
                    effective_dt=fecha_aport,
                    seed_capital_pesos=ss.get("ops_seed_capital") or 0.0,
                    fee_pp=float(ss.get("ops_mgmt_fee_pp") or 0.0),
                    df_opt=df_opt, fi_rows=fi_rows, eq_sel=eq_sel, fx_sel=fx_sel,
                    contribution_amount=monto,
                    use_tilt=use_tilt,
                    tilt_alpha=float(tilt_alpha or 0.0),
                    freeze_level0_to_current=True,
                    overflow_policy=overflow_policy if overflow_policy != "dejar_en_efectivo" else "none",
                )
                ss["last_contrib_result"] = res.df_allocation
                ss["last_contrib_meta"] = {
                    "leftover": float(res.leftover),
                    "total_after": float(res.total_after),
                    "effective_date": str(res.effective_date),
                }
                ss["contrib_prev_inputs"] = {"fecha": curr_fecha, "monto": curr_monto}
                ss["contrib_calc_done"] = True
                st.rerun()
                return True  # (no se alcanza por el rerun)
            except Exception as e:
                st.error(f"❌ No se pudo calcular la aportación: {e}")
                return False

        # Si ya hay cálculo y cambiaron inputs, recalcula automáticamente (sin mostrar botón)
        inputs_changed = (calc_done and (prev_fecha != curr_fecha or prev_monto != curr_monto))
        if inputs_changed:
            _do_calculate()

        # Mostrar botón solo si aún no se ha calculado nunca (o no hay df_alloc)
        if not calc_done or df_alloc.empty:
            btn_cols = st.columns([0.8, 0.2])
            with btn_cols[1]:
                # botón pequeño (sin ocupar todo el ancho)
                if st.button("Calcular distribución recomendada", type="primary"):
                    _do_calculate()

        # =======================================
        #  UI POST-CÁLCULO (editor + resumen + recs)
        # =======================================
        df_alloc = ss.get("last_contrib_result", pd.DataFrame())
        if not df_alloc.empty:
            # Editor directo (sin títulos/captions)
            df_edit = df_alloc[['clase', 'w_aportacion_%']].copy()
            df_edit.rename(columns={'w_aportacion_%': 'Peso (%)'}, inplace=True)
            editor_key = f"aport_editor_{ss.get('contrib_recalc_counter', 0)}"

            edited_df = st.data_editor(
                df_edit,
                column_config={
                    "clase": st.column_config.TextColumn("Clase de Activo", disabled=True),
                    "Peso (%)": st.column_config.NumberColumn("Peso (%)", min_value=0.0, max_value=100.0, step=0.01, format="%.2f%%")
                },
                hide_index=True, use_container_width=True, key=editor_key
            )

            original_weights = df_alloc.set_index('clase')['w_aportacion_%'].round(4)
            edited_weights = edited_df.set_index('clase')['Peso (%)'].round(4)

            if not original_weights.equals(edited_weights):
                changed_mask = original_weights != edited_weights
                fixed_sum = edited_weights[changed_mask].sum()
                final_weights_pct = edited_weights.copy()

                if fixed_sum > 100.0:
                    st.warning("La suma de los valores modificados excede 100%. Se normalizarán.")
                    final_weights_pct = pd.Series(0.0, index=edited_weights.index)
                    norm_fixed = (edited_weights[changed_mask] / fixed_sum) * 100.0
                    for cls, weight in norm_fixed.items():
                        final_weights_pct[cls] = weight
                else:
                    remaining_weight = 100.0 - fixed_sum
                    unchanged_mask = ~changed_mask
                    original_unchanged_weights = original_weights[unchanged_mask]
                    original_unchanged_sum = original_unchanged_weights.sum()
                    if original_unchanged_sum > 0:
                        for cls, original_weight in original_unchanged_weights.items():
                            final_weights_pct[cls] = (original_weight / original_unchanged_sum) * remaining_weight
                    elif remaining_weight > 0 and len(original_unchanged_weights) > 0:
                        equal_share = remaining_weight / len(original_unchanged_weights)
                        for cls in original_unchanged_weights.index:
                            final_weights_pct[cls] = equal_share

                CLASS_MAP_REVERSE = {v: k for k, v in CLASS_MAP.items()}
                contrib_weights_canonical = {CLASS_MAP_REVERSE.get(idx, idx): val for idx, val in final_weights_pct.to_dict().items()}

                try:
                    res_recalculated = apply_user_contribution_weights(
                        start_dt=ss.get("ops_start_date") or _today(),
                        effective_dt=ss.get("contrib_effective_date") or _today(),
                        seed_capital_pesos=ss.get("ops_seed_capital") or 0.0,
                        fee_pp=float(ss.get("ops_mgmt_fee_pp") or 0.0),
                        df_opt=df_opt, fi_rows=fi_rows, eq_sel=eq_sel, fx_sel=fx_sel,
                        contribution_amount=float(ss.get("contrib_amount_last", 0.0)),
                        contrib_weights_by_class_pct=contrib_weights_canonical,
                    )
                    ss['last_contrib_result'] = res_recalculated.df_allocation
                    ss['last_contrib_meta']['leftover'] = res_recalculated.leftover
                    ss['last_contrib_meta']['total_after'] = res_recalculated.total_after
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error al recalcular con tus pesos: {e}")

            # Resumen: tabla transpuesta
            try:
                df_to_render = ss.get("last_contrib_result", pd.DataFrame())
                if not df_to_render.empty:
                    tbl = _render_transposed_contrib(df_to_render)
                    st.dataframe(tbl, hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Error al renderizar la tabla: {e}")
                st.dataframe(ss.get("last_contrib_result", pd.DataFrame()), use_container_width=True)

            # Segmented control + recomendaciones (sin botón Confirmar)
            clases_con_peso, pesos_por_clase = [], {}
            for _, row in df_alloc.iterrows():
                clase = row["clase"]; peso = float(row["w_aportacion_%"])
                if peso > 0:
                    clases_con_peso.append(clase)
                    pesos_por_clase[clase] = peso

            if clases_con_peso:
                selected_categoria = st.segmented_control(
                    "",  # sin texto
                    options=clases_con_peso,
                    default=clases_con_peso[0],
                    key="categoria_detalle_selector",
                    label_visibility="collapsed",
                )

                if selected_categoria:
                    peso_seleccionado = pesos_por_clase.get(selected_categoria, 0.0)
                    st.caption(f"📊 Peso asignado a **{selected_categoria}**: {peso_seleccionado:.2f}%")

                    cat = (selected_categoria or "").strip().lower()

                    if cat in ("renta variable internacional",):
                        mostrar_recomendaciones_rv(
                            categoria_code="RV_EXT",
                            titulo="Acciones internacionales",
                            selected_categoria=selected_categoria,
                            pesos_por_clase=pesos_por_clase,
                        )
                    elif cat in ("renta variable nacional",):
                        mostrar_recomendaciones_rv(
                            categoria_code="RV_MX",
                            titulo="Acciones mexicanas",
                            selected_categoria=selected_categoria,
                            pesos_por_clase=pesos_por_clase,
                        )
                    elif cat in ("fibras", "fibras (reits)"):
                        mostrar_recomendaciones_rv(
                            categoria_code="FIBRA_MX",
                            titulo="FIBRAs",
                            selected_categoria=selected_categoria,
                            pesos_por_clase=pesos_por_clase,
                        )
                    elif cat in ("divisas", "fx", "monedas", "foreign exchange"):
                        mostrar_recomendaciones_rv(
                            categoria_code="FX",
                            titulo="Divisas",
                            selected_categoria=selected_categoria,
                            pesos_por_clase=pesos_por_clase,
                        )
                    elif cat in ("nota estructurada", "notas estructuradas", "estructurados"):
                        mostrar_structured_note_compacto()
                    elif cat in ("deuda gubernamental",):
                        ss["fic_tipo_bono"] = "Deuda Gubernamental"
                        mostrar_fixed_income_compacto()
                    elif cat in ("deuda privada nacional",):
                        ss["fic_tipo_bono"] = "Deuda Privada Nacional"
                        mostrar_fixed_income_compacto()
                    elif cat in ("deuda internacional",):
                        ss["fic_tipo_bono"] = "Deuda Internacional"
                        mostrar_fixed_income_compacto()
                    else:
                        st.info(f"Recomendaciones para **{selected_categoria}** no implementadas aún.")


                def _confirmar_aportacion_y_registrar(ss, effective_date: date) -> dict:
                    """
                    Registra la aportación en el ledger consultando:
                    1. Activos seleccionados por el usuario en la UI (recs_selected_*)
                    2. Pesos asignados por el usuario (recs_weights_*)
                    3. Bonos agregados (fic_bonds_list)
                    4. Notas estructuradas (snc_notes_list)
                    """
                    monto_aportado = float(ss.get("contrib_amount_last") or 0.0)
                    df_alloc = ss.get("last_contrib_result", pd.DataFrame()).copy()
                    
                    if df_alloc.empty or monto_aportado <= 0:
                        raise ValueError("No hay asignación calculada o el monto es 0.")

                    # 1) DEPOSIT
                    ledger.add_transaction(
                        ss, ts=effective_date, kind="DEPOSIT",
                        cash=monto_aportado, note="Aportación de capital"
                    )

                    # Mapeo COMPLETO de clases (incluyendo variaciones)
                    CLASS_MAP_REVERSE = {
                        "Renta Variable Internacional": "RV_EXT",
                        "Renta Variable Nacional": "RV_MX",
                        "FIBRAS (REITs)": "FIBRA_MX",
                        "FIBRAS": "FIBRA_MX",  # ← Alias adicional
                        "Divisas": "FX",
                        "Deuda Gubernamental": "FI_GOB_MX",
                        "Deuda Privada Nacional": "FI_CORP_MX",
                        "Deuda Internacional": "FI_EXT",
                        "Notas Estructuradas": "Nota estructurada",
                    }

                    buy_lines = 0
                    debug_rows = []

                    def _get_price_at_date(ticker: str, fallback: float, clase_code: str = "") -> float:
                        """
                        Devuelve SIEMPRE precio en MXN a la fecha efectiva, usando exactamente
                        la misma lógica que aportaciones/ledger (incluye FX: invertir USD/XXX
                        y multiplicar por USD→MXN cuando aplica).
                        """
                        import streamlit as st
                        ss_local = st.session_state
                        return float(_px_mxn_at(ss_local, ticker, clase_code, fallback, effective_date))


                    # 2) Procesar compras por clase
                    for _, row in df_alloc.iterrows():
                        monto_clase = float(row.get("aportacion_$", 0.0))
                        if monto_clase <= 0:
                            continue

                        clase_pretty = str(row["clase"]).strip()
                        cat_code = CLASS_MAP_REVERSE.get(clase_pretty)
                        
                        if not cat_code:
                            st.warning(f"⚠️ Clase '{clase_pretty}' no reconocida. Omitiendo.")
                            continue

                        # ========== RENTA VARIABLE / FX / FIBRAS ==========
                        if cat_code in ("RV_EXT", "RV_MX", "FIBRA_MX", "FX"):
                            # Leer selección del usuario desde la UI
                            key_selected = f"recs_selected_{cat_code}"
                            key_weights = f"recs_weights_{cat_code}"
                            
                            selected_tickers = ss.get(key_selected, {})  # {ticker: {nombre, score}}
                            user_weights = ss.get(key_weights, {})       # {ticker: peso_pct}
                            
                            if not selected_tickers:
                                st.warning(f"⚠️ No hay activos seleccionados para {clase_pretty} (key: {key_selected})")
                                continue
                            
                            # Normalizar pesos
                            total_weight = sum(user_weights.values())
                            if total_weight <= 0:
                                st.warning(f"⚠️ Suma de pesos en {clase_pretty} es 0. Distribuyendo equitativamente.")
                                # Fallback: distribuir equitativamente
                                n = len(selected_tickers)
                                user_weights = {tk: 100.0/n for tk in selected_tickers.keys()}
                                total_weight = 100.0
                            
                            for ticker, meta in selected_tickers.items():
                                peso_pct = user_weights.get(ticker, 0.0)
                                if peso_pct <= 0:
                                    continue
                                
                                # Monto proporcional
                                monto_ticker = monto_clase * (peso_pct / total_weight)
                                
                                # Precio
                                px_inicial_meta = ss.get("ops_positions", pd.DataFrame())
                                px_fallback = 1.0
                                if not px_inicial_meta.empty:
                                    match = px_inicial_meta[px_inicial_meta["ticker"] == ticker]
                                    if not match.empty:
                                        px_fallback = match.iloc[0].get("px_inicial", 1.0)
                                
                                px = _get_price_at_date(ticker, px_fallback, cat_code)
                                qty = monto_ticker / px if px > 0 else 0
                                
                                if qty <= 0:
                                    continue
                                
                                nombre = meta.get("nombre", ticker)
                                
                                # REGISTRO EN LEDGER
                                ledger.add_transaction(
                                    ss, ts=effective_date, kind="BUY",
                                    symbol=ticker, qty=qty, px=px, name=nombre,
                                    note=f"Aportación {clase_pretty}"
                                )
                                
                                # ⭐ GUARDAR/ACTUALIZAR METADATA EN SYNTHETIC_PRICES (se mantiene tu lógica)
                                if "synthetic_prices" not in ss:
                                    ss["synthetic_prices"] = {}
                                if ticker not in ss["synthetic_prices"]:
                                    ss["synthetic_prices"][ticker] = {}
                                
                                ss["synthetic_prices"][ticker]["clase"] = cat_code  # RV_EXT, RV_MX, etc.
                                ss["synthetic_prices"][ticker]["activo"] = nombre
                                ss["synthetic_prices"][ticker]["px_inicial"] = px
                                ss["synthetic_prices"][ticker]["start_date"] = effective_date

                                # ✅ NUEVO: asegurar price_series[timestamp(effective_date)] = px
                                ts = pd.Timestamp(effective_date)
                                ps = ss["synthetic_prices"][ticker].get("price_series")
                                if isinstance(ps, dict):
                                    ps[ts] = float(px)
                                elif hasattr(ps, "to_dict"):  # pandas Series/DataFrame
                                    d = ps.to_dict()
                                    d[ts] = float(px)
                                    ss["synthetic_prices"][ticker]["price_series"] = d
                                else:
                                    ss["synthetic_prices"][ticker]["price_series"] = {ts: float(px)}
                                
                                buy_lines += 1
                                debug_rows.append({
                                    "clase": clase_pretty,
                                    "symbol": ticker,
                                    "nombre": nombre,
                                    "px": px,
                                    "monto": monto_ticker,
                                    "qty": qty
                                })

                        # ========== RENTA FIJA ==========
                        elif cat_code in ("FI_GOB_MX", "FI_CORP_MX", "FI_EXT"):
                            bonds_list = ss.get("fic_bonds_list", [])
                            bonds_weights = ss.get("fic_weights", {})
                            
                            if not bonds_list:
                                st.warning(f"⚠️ No hay bonos agregados para {clase_pretty}.")
                                continue
                            
                            total_weight = sum(bonds_weights.values())
                            if total_weight <= 0:
                                n = len(bonds_list)
                                bonds_weights = {b["bond_id"]: 100.0/n for b in bonds_list}
                                total_weight = 100.0
                            
                            for bond in bonds_list:
                                bond_id = bond["bond_id"]
                                peso_pct = bonds_weights.get(bond_id, 0.0)
                                if peso_pct <= 0:
                                    continue
                                
                                monto_bond = monto_clase * (peso_pct / total_weight)
                                
                                # Precio limpio como px_inicial
                                px = float(bond.get("precio_limpio", 100.0))
                                qty = monto_bond / px if px > 0 else 0
                                
                                if qty <= 0:
                                    continue
                                
                                # Generar ticker sintético si no existe
                                ticker = bond.get("ticker", f"SYNTH_FI_{bond_id[:8]}")
                                nombre = bond.get("nombre", ticker)
                                
                                ledger.add_transaction(
                                    ss, ts=effective_date, kind="BUY",
                                    symbol=ticker, qty=qty, px=px, name=nombre,
                                    note=f"Aportación {clase_pretty}"
                                )

                                # ✅ NUEVO: metadata + price_series para FI
                                if "synthetic_prices" not in ss:
                                    ss["synthetic_prices"] = {}
                                if ticker not in ss["synthetic_prices"]:
                                    ss["synthetic_prices"][ticker] = {}
                                ss["synthetic_prices"][ticker]["clase"] = cat_code
                                ss["synthetic_prices"][ticker]["activo"] = nombre
                                ss["synthetic_prices"][ticker]["px_inicial"] = px
                                ss["synthetic_prices"][ticker]["start_date"] = effective_date
                                ts = pd.Timestamp(effective_date)
                                ps = ss["synthetic_prices"][ticker].get("price_series")
                                if isinstance(ps, dict):
                                    ps[ts] = float(px)
                                elif hasattr(ps, "to_dict"):
                                    d = ps.to_dict()
                                    d[ts] = float(px)
                                    ss["synthetic_prices"][ticker]["price_series"] = d
                                else:
                                    ss["synthetic_prices"][ticker]["price_series"] = {ts: float(px)}

                                buy_lines += 1
                                debug_rows.append({
                                    "clase": clase_pretty,
                                    "symbol": ticker,
                                    "nombre": nombre,
                                    "px": px,
                                    "monto": monto_bond,
                                    "qty": qty
                                })

                        # ========== NOTAS ESTRUCTURADAS ==========
                        elif cat_code == "Nota estructurada":
                            notes_list = ss.get("snc_notes_list", [])
                            notes_weights = ss.get("snc_weights", {})
                            
                            if not notes_list:
                                st.warning(f"⚠️ No hay notas estructuradas agregadas.")
                                continue
                            
                            total_weight = sum(notes_weights.values())
                            if total_weight <= 0:
                                n = len(notes_list)
                                notes_weights = {nt["note_id"]: 100.0/n for nt in notes_list}
                                total_weight = 100.0
                            
                            for note in notes_list:
                                note_id = note["note_id"]
                                peso_pct = notes_weights.get(note_id, 0.0)
                                if peso_pct <= 0:
                                    continue
                                
                                monto_note = monto_clase * (peso_pct / total_weight)
                                
                                # Precio: usar spot como px_inicial (base $10)
                                px = float(note.get("spot", 10.0))
                                qty = monto_note / px if px > 0 else 0
                                
                                if qty <= 0:
                                    continue
                                
                                ticker = f"SYNTH_SN_{note_id[:8]}"
                                nombre = note.get("name", ticker)
                                
                                ledger.add_transaction(
                                    ss, ts=effective_date, kind="BUY",
                                    symbol=ticker, qty=qty, px=px, name=nombre,
                                    note=f"Aportación Nota Estructurada"
                                )

                                # ✅ NUEVO: metadata + price_series para Notas
                                if "synthetic_prices" not in ss:
                                    ss["synthetic_prices"] = {}
                                if ticker not in ss["synthetic_prices"]:
                                    ss["synthetic_prices"][ticker] = {}
                                ss["synthetic_prices"][ticker]["clase"] = cat_code
                                ss["synthetic_prices"][ticker]["activo"] = nombre
                                ss["synthetic_prices"][ticker]["px_inicial"] = px
                                ss["synthetic_prices"][ticker]["start_date"] = effective_date
                                ts = pd.Timestamp(effective_date)
                                ps = ss["synthetic_prices"][ticker].get("price_series")
                                if isinstance(ps, dict):
                                    ps[ts] = float(px)
                                elif hasattr(ps, "to_dict"):
                                    d = ps.to_dict()
                                    d[ts] = float(px)
                                    ss["synthetic_prices"][ticker]["price_series"] = d
                                else:
                                    ss["synthetic_prices"][ticker]["price_series"] = {ts: float(px)}

                                buy_lines += 1
                                debug_rows.append({
                                    "clase": "Notas Estructuradas",
                                    "symbol": ticker,
                                    "nombre": nombre,
                                    "px": px,
                                    "monto": monto_note,
                                    "qty": qty
                                })

                    # Guardar debug
                    ss["last_contrib_debug_rows"] = debug_rows

                    return {
                        "deposit": monto_aportado,
                        "lines": buy_lines,
                        "effective_date": effective_date.strftime("%d/%m/%Y")
                    }



                # ================================
                # BOTÓN CONFIRMAR (SIEMPRE VISIBLE)
                # ================================
                st.divider()

                df_confirm = ss.get("last_contrib_result", pd.DataFrame())
                sum_weights = float(pd.to_numeric(df_confirm.get("w_aportacion_%"), errors="coerce").fillna(0).sum()) if not df_confirm.empty else 0.0
                sum_ok = abs(sum_weights - 100.0) <= 0.01
                monto_ok = (float(ss.get("contrib_amount_last") or 0.0) > 0)

                if not df_confirm.empty and not sum_ok:
                    st.warning(f"⚠️ La suma de 'Peso (%)' es **{sum_weights:.2f}%** y debe ser **100.00%**.")
                if not monto_ok:
                    st.warning("⚠️ Debes capturar un **monto de aportación** positivo.")

                # ✅ MOSTRAR RESULTADO SI YA SE PROCESÓ
                if ss.get("_contrib_just_confirmed"):
                    res = ss.get("_contrib_last_result", {})
                    st.success(
                        f"✅ Aportación registrada: Depósito de ${res.get('deposit', 0):,.2f} "
                        f"y {int(res.get('lines', 0))} compras — Fecha: {res.get('effective_date', '')}."
                    )

                    # 👉 Renderiza DEBUG
                    dbg = ss.get("last_contrib_debug_rows", [])
                    if dbg:
                        with st.expander("🐞 DEBUG aportación (líneas generadas)", expanded=True):
                            st.dataframe(pd.DataFrame(dbg), use_container_width=True)
                    
                    # ✅ Botón para limpiar mensaje y volver
                    if st.button("✅ Entendido, cerrar mensaje", type="secondary"):
                        ss["_contrib_just_confirmed"] = False
                        st.rerun()

                else:
                    # ✅ MOSTRAR BOTÓN CONFIRMAR SOLO SI NO SE HA PROCESADO
                    col_btn_spacer, col_btn = st.columns([0.7, 0.3])
                    with col_btn:
                        confirmar = st.button(
                            "💾 Confirmar aportación y registrar",
                            type="primary",
                            use_container_width=True,
                            disabled=not (sum_ok and monto_ok and not df_confirm.empty)
                        )

                    if confirmar:
                        try:
                            # ✅ PROCESAR APORTACIÓN
                            res = _confirmar_aportacion_y_registrar(ss, effective_date=fecha_aport)
                            
                            # ✅ RECONSTRUIR PORTFOLIO
                            from tabs.ledger import rebuild_portfolio_from_ledger
                            start_dt = runtime.get_ops_started_at() or ss.get("ops_start_date") or _today()
                            end_dt = runtime.get_act_as_of() or _today()
                            
                            pos_new, ts_new = rebuild_portfolio_from_ledger(ss, start_date=start_dt, end_date=end_dt)
                            ss["ops_positions"] = pos_new
                            ss["ops_timeseries"] = ts_new
                            
                            # ✅ GUARDAR FLAG Y RESULTADO
                            ss["_contrib_just_confirmed"] = True
                            ss["_contrib_last_result"] = res
                            
                            # 🔄 LIMPIAR ESTADO DE APORTACIÓN PARA REINICIAR INTERFAZ
                            ss["last_contrib_result"] = pd.DataFrame()
                            ss["contrib_amount_last"] = None
                            if "contrib_user_weights" in ss:
                                del ss["contrib_user_weights"]
                            
                            # ✅ RERUN PARA MOSTRAR RESULTADO
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ No se pudo confirmar la aportación: {e}")
                            import traceback
                            st.code(traceback.format_exc())