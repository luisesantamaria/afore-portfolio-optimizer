# tabs/portfolio.py - VERSIÓN COMPLETA DINÁMICA (ALINEADA A OPERACIONES)
from __future__ import annotations
from typing import Dict, Any, Tuple
from pathlib import Path
from datetime import date, datetime

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# 👉 Core de optimización y meta
from tabs import optimization as core
from tabs import consar_limits
from tabs import fixed_income as fi
from tabs.banxico_client import get_inflation_12m, latest_on_or_before
from tabs.yf_store import get_hist_5y

# 👉 Ledger/Runtime (para alinear con Operaciones)
from tabs import runtime, ledger
from tabs.ledger import rebuild_portfolio_from_ledger

_STATE_PATH = getattr(core, "_STATE_PATH", Path("./afore_state.json"))

# ============================================================================
# Helpers internos opcionales (mantengo por compatibilidad)
# ============================================================================

def _peek(obj, n=5):
    """Pequeño helper para mostrar lista/dict de manera compacta."""
    try:
        if isinstance(obj, dict):
            keys = list(obj.keys())[:n]
            return ", ".join(map(str, keys)) + (" ..." if len(obj) > n else "")
        if isinstance(obj, (list, tuple)):
            vals = [str(x) for x in obj[:n]]
            return ", ".join(vals) + (" ..." if len(obj) > n else "")
        return str(obj)
    except Exception:
        return str(obj)

# ============================================================================
# RENDER
# ============================================================================

def render():
    # Asegura defaults y autosave del core
    core._ensure_sticky_siefore_afore()
    core._auto_load_once()

    # =======================
    # DATOS BASE DEL CORE
    # =======================
    try:
        caps, risk_limits, siefore_name, afore_name = core._caps_from_consar()
    except Exception as e:
        st.error(f"Fallo _caps_from_consar(): {e}")
        return

    try:
        rows = core._cosechar_activos()  # <- fuente principal de instrumentos (modo diseño)
    except Exception as e:
        st.error(f"Fallo _cosechar_activos(): {e}")
        rows = []

    try:
        caps_op = core._caps_operativos(caps, buffer=core.OP_BUFFER)
    except Exception as e:
        st.error(f"Fallo _caps_operativos(): {e}")
        caps_op = caps

    # =======================
    # 📊 CAPTION DINÁMICO (ARRIBA)
    # =======================
    afore_display = afore_name if afore_name and afore_name != "N/A" and "Fallback" not in afore_name else "—"
    is_operating = st.session_state.get("ops_operating", False)

    # ⬆️ Placeholder para que el caption quede siempre arriba
    caption_ph = st.empty()

    # Fecha de corte global (idéntica a Operaciones)
    act_as_of = runtime.get_act_as_of() or datetime.now().date()
    if isinstance(act_as_of, datetime):
        act_as_of = act_as_of.date()

    # Si hay fees pendientes (p. ej. después de una aportación), postéalos aquí también
    pending_from = st.session_state.get("_fee_recalc_needed_from")
    if pending_from:
        try:
            ledger.apply_daily_fees_and_autosell(
                st.session_state, start_date=pending_from, end_date=act_as_of, debug=False
            )
            st.session_state["_fee_recalc_needed_from"] = None
        except Exception:
            pass

    # =======================
    # SIN ACTIVOS -> DIAGNÓSTICO ÚTIL
    # =======================
    if not rows and not is_operating:
        st.warning(
            "No hay activos para optimizar. "
            "Revisa que `equity_selection`/`rv_selection`/`fi_rows` estén poblados. "
            "Arriba, en *Debug Portfolio*, puedes forzar la relectura del autosave."
        )
        if hasattr(core, "_peek_cosecha_fuentes"):
            try:
                fuentes = core._peek_cosecha_fuentes()
                st.caption("Fuentes detectadas por _cosechar_activos (peek):")
                st.json(fuentes, expanded=False)
            except Exception:
                pass
        return

    # =======================
    # LÓGICA DE BASE ESTABLE + OPTIMIZACIÓN (se usa en modo diseño)
    # =======================
    stable_weights = st.session_state.get("stable_base_weights")
    base_names = st.session_state.get("stable_base_asset_names")

    try:
        tabla_opt, totales_opt_raw, risk, final_weights = core._optimizar(
            rows,
            caps_op,
            risk_limits,
            stable_base_weights=stable_weights,
            base_asset_names=base_names
        )
    except Exception as e:
        st.error(f"Error en _optimizar(): {e}")
        st.stop()

    if "stable_base_weights" not in st.session_state and final_weights:
        st.session_state["stable_base_weights"] = final_weights
        st.session_state["stable_base_asset_names"] = set(final_weights.keys())

    st.session_state["optimization_table"] = tabla_opt
    st.session_state["optimization_risk"] = risk

    # =======================
    # KPI SECTION (modo diseño)
    # =======================
    if not is_operating:
        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10, gap="small")
        with col1:  st.metric("Rend. Esperado", f"{risk['mu_port']*100:,.2f}%")
        with col2:  st.metric("Volatilidad", f"{risk['sigma_annual']*100:,.2f}%")
        with col3:  st.metric("Beta (IPC)", f"{risk['beta_portfolio']:,.2f}")
        with col4:  st.metric("Sharpe", f"{risk['sharpe_ratio']:,.2f}")
        with col5:  st.metric("Sortino (Rf)", f"{risk.get('sortino_rf', float('nan')):,.2f}")
        with col6:  st.metric("Máx. Drawdown", f"{risk['max_drawdown']*100:,.2f}%")
        with col7:  st.metric("VaR 95%", f"{risk['var_daily']*100:,.2f}%")
        with col8:  st.metric("Tracking Error", f"{risk['te_annual']*100:,.2f}%")
        with col9:
            rf_val = risk.get("risk_free_rate_used", float('nan'))
            st.metric("CETES 28 días", f"{rf_val*100:,.2f}%")
        with col10:
            infl_12m, _ = get_inflation_12m()
            st.metric("Inflación 12M", f"{infl_12m*100:,.2f}%")

        st.markdown("<div style='margin-top: 70px;'></div>", unsafe_allow_html=True)

    # =======================
    # MODO DISEÑO vs MODO OPERANDO
    # =======================
    PRETTY = {
        "FIBRA_MX": "FIBRA",
        "FI_EXT": "Deuda Internacional",
        "FI_GOB_MX": "Deuda Gubernamental",
        "RV_EXT": "Renta Variable Internacional",
        "RV_MX": "Renta Variable Nacional",
        "FI_CORP_MX": "Deuda Privada Nacional",
        "FX": "Divisa",
        "SN": "Estructurados",
        "CASH": "Efectivo",
    }

    if not is_operating:
        # ---------- MODO DISEÑO (usa optimizador) ----------
        caption_ph.caption(f"SIEFORE: **{siefore_name}** | AFORE: **{afore_display}** | Activos en portafolio: **{len(rows)}**")

        tabla_para_visualizar = tabla_opt.copy()
        totales_raw = totales_opt_raw.copy()

        left, right = st.columns([1.1, 1.0], gap="large")

        # ===== Izquierda: tabla CONSAR (diseño) =====
        with left:
            pesos_por_clase = {row["clase"]: float(row["Peso (%)"]) for _, row in totales_raw.iterrows()}
            def w(cl): return pesos_por_clase.get(cl, 0.0)

            df_comp_data = consar_limits.get_siefore_composition_data(siefore_name, afore_name)
            consar_rows = [
                {"Agrupación": "Renta Variable",         "Peso (%)": round(w("RV_MX") + w("RV_EXT"), 2), "Límite CONSAR (%)": caps["RV_TOTAL_MAX"]*100},
                {"Agrupación": "Valores Extranjeros",    "Peso (%)": round(w("RV_EXT") + w("FI_EXT"), 2), "Límite CONSAR (%)": caps["EXT_MAX"]*100},
                {"Agrupación": "FIBRAS",                 "Peso (%)": round(w("FIBRA_MX"), 2),             "Límite CONSAR (%)": caps["FIBRA_MX_MAX"]*100},
                {"Agrupación": "Deuda Gubernamental",    "Peso (%)": round(w("FI_GOB_MX"), 2),            "Límite CONSAR (%)": caps.get("FI_GOB_MX_MAX", 1.0)*100},
                {"Agrupación": "Deuda Privada Nacional", "Peso (%)": round(w("FI_CORP_MX"), 2),           "Límite CONSAR (%)": caps.get("FI_CORP_MX_MAX", 1.0)*100},
                {"Agrupación": "Divisas",                "Peso (%)": round(w("FX"), 2),                   "Límite CONSAR (%)": caps.get("FX_MAX", 0.10)*100},
                {"Agrupación": "Estructurados",          "Peso (%)": round(w("SN"), 2),                   "Límite CONSAR (%)": caps.get("SN_MAX", 0.05)*100},
                {"Agrupación": "Mercancías",             "Peso (%)": 0.0,                                 "Límite CONSAR (%)": caps.get("MERCANCÍAS_MAX", 0.0)*100},
            ]
            consar_df = pd.DataFrame(consar_rows)

            if not df_comp_data.empty:
                consar_df = pd.merge(consar_df, df_comp_data, on="Agrupación", how="left").fillna(0.0)
                col_especifica_label = f"{siefore_name} {afore_name if afore_name and afore_name!='N/A' and 'Fallback' not in (afore_name or '') else '—'}"
                col_promedio_label = f"{siefore_name} Promedio"
                consar_df.rename(columns={
                    "Agrupación": "Categoría",
                    "Peso (%)": "Portafolio (%)",
                    "Promedio": col_promedio_label,
                    "Especifica": col_especifica_label,
                    "Límite CONSAR (%)": "Límites CONSAR TEMP",
                }, inplace=True)
                consar_df["Límites CONSAR"] = consar_df["Límites CONSAR TEMP"]
                COLS_FINALES = ["Categoría", "Portafolio (%)", col_especifica_label, col_promedio_label, "Límites CONSAR"]
                consar_df = consar_df[COLS_FINALES]

                st.dataframe(
                    consar_df, hide_index=True, use_container_width=True,
                    column_config={
                        "Portafolio (%)": st.column_config.NumberColumn(label="Portafolio", format="%.2f%%"),
                        "Límites CONSAR": st.column_config.NumberColumn(label="Límites CONSAR", format="%.2f%%"),
                        col_promedio_label: st.column_config.NumberColumn(label=col_promedio_label, format="%.2f%%"),
                        col_especifica_label: st.column_config.NumberColumn(label=col_especifica_label, format="%.2f%%"),
                    },
                )
            else:
                st.dataframe(
                    consar_df.rename(columns={"Agrupación": "Categoría", "Peso (%)": "Portafolio (%)"}),
                    hide_index=True, use_container_width=True,
                    column_config={
                        "Portafolio (%)": st.column_config.NumberColumn(label="Portafolio", format="%.2f%%"),
                        "Límite CONSAR (%)": st.column_config.NumberColumn(label="Límites CONSAR", format="%.2f%%"),
                    },
                )

        # ===== Derecha: Donut (diseño) =====
        with right:
            if not tabla_para_visualizar.empty:
                slices = []
                for _, row in tabla_para_visualizar.iterrows():
                    clase = row["clase"]; peso = float(row["Peso (%)"])
                    label = PRETTY.get(clase, clase)
                    slices.append({"Clase": label, "Peso": peso})
                pie_df = pd.DataFrame(slices).groupby("Clase", as_index=False)["Peso"].sum()
                fig = px.pie(pie_df, names="Clase", values="Peso", hole=0.35)
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

        # ===== Detalle por categoría (diseño) =====
        st.subheader("Detalle por categoría")

        fi_meta = st.session_state.get("fi_rows", []) or []
        bono_to_tipo = {str(r.get("Bono")): str(r.get("Tipo de bono", "")) for r in fi_meta}

        tabla_detalle = tabla_para_visualizar.copy()
        tabla_detalle["_is_theme"] = False

        def has(mask):
            return bool(tabla_detalle.loc[mask, "nombre"].size) if "nombre" in tabla_detalle.columns else False

        cats = []
        if has(tabla_detalle["clase"].eq("RV_MX")):      cats.append(("Renta Variable Nacional",   ("RV_MX", None)))
        if has(tabla_detalle["clase"].eq("RV_EXT")):     cats.append(("Renta Variable Internacional", ("RV_EXT", None)))
        if has(tabla_detalle["clase"].eq("FIBRA_MX")):   cats.append(("FIBRA",                ("FIBRA_MX", None)))
        if has(tabla_detalle["clase"].eq("FI_GOB_MX")):  cats.append(("Deuda Gubernamental",  ("FI_GOB_MX", None)))
        if has(tabla_detalle["clase"].eq("FI_CORP_MX")): cats.append(("Deuda Privada Nacional", ("FI_CORP_MX", None)))
        if has(tabla_detalle["clase"].eq("SN")):         cats.append(("Estructurados",        ("SN", None)))
        if has(tabla_detalle["clase"].eq("FI_EXT")):     cats.append(("Deuda Internacional",  ("FI_EXT", None)))
        if has(tabla_detalle["clase"].eq("FX")):         cats.append(("Divisas",              ("FX", None)))

        cats_all = [("Todas", (None, None))] + cats
        opciones = [lbl for (lbl, _) in cats_all]
        label_to_key = {lbl: key for (lbl, key) in cats_all}

        selected_pretty = st.segmented_control(
            "Selecciona una categoría para ver su detalle:",
            opciones,
            default=("Todas" if "Todas" in opciones else (opciones[0] if opciones else None))
        ) if opciones else None

        if selected_pretty:
            sel_class, subflag = label_to_key[selected_pretty]
            if sel_class is None:
                detalle = tabla_detalle.copy()
                def _tipo_label(clase: str) -> str:
                    return {
                        "FI_EXT": "Deuda Internacional",
                        "FI_GOB_MX": "Deuda Gubernamental",
                        "FI_CORP_MX": "Deuda Privada Nacional",
                        "FIBRA_MX": "FIBRA",
                        "RV_MX": "Renta Variable Nacional",
                        "RV_EXT": "Renta Variable Internacional",
                        "FX": "Divisa",
                        "SN": "Estructurado",
                    }.get(clase, clase)
                if "nombre" in detalle.columns:
                    detalle["Tipo"] = detalle["clase"].map(_tipo_label)
            else:
                detalle = tabla_detalle[tabla_detalle["clase"].eq(sel_class)].copy()

            if detalle.empty:
                st.info("No hay instrumentos en esta categoría actualmente.")
            else:
                if "nombre" in detalle.columns:
                    bar_df = detalle[["nombre", "Peso (%)"]].sort_values("Peso (%)", ascending=False)
                    fig_bar = px.bar(bar_df, x="nombre", y="Peso (%)")
                    fig_bar.update_layout(xaxis_title=None, yaxis_title="Peso (%)", margin=dict(l=10, r=10, t=10, b=10), height=380)
                    fig_bar.update_xaxes(tickangle=-35, automargin=True)
                    st.plotly_chart(fig_bar, use_container_width=True)

                detalle_vista = detalle.drop(columns=["clase"], errors="ignore").copy().rename(columns={
                    "nombre": "Empresa",
                    "Peso (%)": "Peso",
                })
                ordered_cols = [c for c in ["Empresa", "Tipo", "Peso"] if c in detalle_vista.columns]
                st.dataframe(
                    detalle_vista[ordered_cols],
                    hide_index=True,
                    width='stretch',
                    column_config={
                        "Peso": st.column_config.NumberColumn(format="%.2f%%"),
                    },
                )

        return  # fin modo diseño

    # ---------- MODO OPERANDO (ALINEADO A OPERACIONES) ----------
    # Construir SIEMPRE desde el mismo ledger/serie que Operaciones
    start_dt = runtime.get_ops_started_at() or st.session_state.get("ops_start_date") or act_as_of
    if isinstance(start_dt, datetime):
        start_dt = start_dt.date()

    pos_df, ts_df = rebuild_portfolio_from_ledger(st.session_state, start_date=start_dt, end_date=act_as_of)

    # Valor total idéntico al KPI “Valor Actual” de Operaciones
    total_value = float(ts_df["total_value"].dropna().iloc[-1]) if (
        isinstance(ts_df, pd.DataFrame) and not ts_df.empty and "total_value" in ts_df.columns
    ) else 0.0

    # Tabla para visualización (nombre/clase/peso)
    if isinstance(pos_df, pd.DataFrame) and not pos_df.empty:
        # Esperamos columnas: activo, clase, peso_actual_% (usadas ya en Drift/Operaciones)
        tabla_para_visualizar = pd.DataFrame({
            "nombre": pos_df.get("activo", pos_df.get("name", "")),
            "clase": pos_df.get("clase", ""),
            "Peso (%)": pd.to_numeric(pos_df.get("peso_actual_%", np.nan), errors="coerce"),
        }).dropna(subset=["Peso (%)"])
        # Totales por clase
        totales_raw = tabla_para_visualizar.groupby("clase")["Peso (%)"].sum().reset_index()
        # Nº de activos (excluye efectivo explícito si existiera)
        num_activos = int((tabla_para_visualizar["clase"] != "CASH").sum())
        caption_ph.caption(f"SIEFORE: **{siefore_name}** | AFORE: **{afore_display}** | Activos en portafolio: **{num_activos}**")
    else:
        tabla_para_visualizar = pd.DataFrame(columns=["nombre", "clase", "Peso (%)"])
        totales_raw = pd.DataFrame(columns=["clase", "Peso (%)"])
        caption_ph.caption(f"SIEFORE: **{siefore_name}** | AFORE: **{afore_display}** | Activos en portafolio: **0**")

    # --- Métricas REALIZADAS (reemplazan solo “Rend. Esperado” por TWR y TWR anual) ---
    try:

        token = st.session_state.get("banxico_token")
        cetes_data_date = "N/A" # Variable para guardar la fecha de la subasta

        # 1. Obtener CETES 28d a la fecha de corte (act_as_of)
        if token:
            # CORRECCIÓN: Convertimos 'act_as_of' (date) a un 'datetime'
            act_as_of_dt = datetime.combine(act_as_of, datetime.min.time())
            
            # MEJORA: Capturamos la fecha (f_cetes) además del valor (v_cetes)
            f_cetes, v_cetes, _ = latest_on_or_before("SF43936", act_as_of_dt, token)
            
            if v_cetes is not None:
                rf_annual = float(v_cetes) / 100.0
                cetes_data_date = f_cetes # Guardamos la fecha de la subasta
            else:
                rf_annual = float('nan')
        else:
            rf_annual = float('nan')

        # 2. Obtener Inflación 12M (código sin cambios)
        infl_12m, _ = get_inflation_12m(end_date=act_as_of)
        infl_12m = infl_12m if infl_12m is not None else float('nan')
        
        rf_daily = rf_annual / 252.0 if np.isfinite(rf_annual) else 0.0


        if isinstance(ts_df, pd.DataFrame) and not ts_df.empty and "total_value" in ts_df.columns:
            port = ts_df["total_value"].dropna().copy().sort_index()

            # Flujos externos por día (DEPOSIT/WITHDRAW/FEE/MGMT_FEE/ADMIN_FEE)
            flows = pd.Series(0.0, index=port.index)
            led_df = ledger.get_ledger(st.session_state)
            if isinstance(led_df, pd.DataFrame) and not led_df.empty and {"ts", "kind", "cash"} <= set(led_df.columns):
                cf = led_df.copy()
                cf["ts"] = pd.to_datetime(cf["ts"], errors="coerce")
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

                # Ignorar flujo del primer día (seed capital) para el TWR
                first_idx = port.index[0]
                if first_idx in flows.index:
                    flows.loc[first_idx] = 0.0

            # Retornos diarios TWR
            rets = ((port - flows) / port.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan).dropna()

            # TWR total y anualizado
            if rets.size > 0:
                twr_total = float(np.prod(1.0 + rets) - 1.0)
                n = float(rets.size)
                twr_annual = float((1.0 + twr_total) ** (252.0 / n) - 1.0)
            else:
                twr_total = float("nan")
                twr_annual = float("nan")

            # Vol anual
            vol_ann = (rets.std() * np.sqrt(252.0)) if rets.size > 1 else float("nan")

            # Beta y Tracking Error vs IPC (Yahoo '^MXX')
            beta_ipc = float("nan")
            te_annual = float("nan")
            try:
                bench = get_hist_5y("^MXX")
                if bench is not None and not bench.empty and "Adj Close" in bench:
                    rb = bench["Adj Close"].pct_change().dropna()
                    # Alinear índices
                    df_align = pd.concat([
                        rets.rename("rp"),
                        rb.rename("rb")
                    ], axis=1).dropna()
                    if len(df_align) > 2:
                        cov = np.cov(df_align["rp"], df_align["rb"])[0, 1]
                        var_b = np.var(df_align["rb"])
                        if var_b > 0:
                            beta_ipc = cov / var_b
                        te_annual = (df_align["rp"] - df_align["rb"]).std() * np.sqrt(252.0)
            except Exception:
                pass

            # Máx drawdown (sobre índice de rendimiento)
            try:
                base = float(port.iloc[0])
                perf_idx = base * (1.0 + rets).cumprod()
                perf_idx = pd.concat([pd.Series({rets.index[0]: base}), perf_idx])
                perf_idx = perf_idx[~perf_idx.index.duplicated(keep="first")]
                roll_max = perf_idx.cummax()
                dd = (perf_idx / roll_max) - 1.0
                max_dd = float(dd.min()) if not dd.empty else float("nan")
            except Exception:
                max_dd = float("nan")

            # Sharpe (anual)
            try:
                mean_excess_daily = (rets - rf_daily).mean()
                sharpe = (mean_excess_daily / rets.std()) * np.sqrt(252.0) if rets.std() > 0 else float("nan")
            except Exception:
                sharpe = float("nan")

            # Sortino (anual)
            try:
                downside = np.where((rets - rf_daily) < 0.0, (rets - rf_daily), 0.0)
                downside_std = np.sqrt(np.mean(downside ** 2))
                sortino = ((rets.mean() - rf_daily) / downside_std) * np.sqrt(252.0) if downside_std > 0 else float("nan")
            except Exception:
                sortino = float("nan")

            # VaR 95% (histórico, diario)
            try:
                var95 = -np.nanpercentile(rets, 5)  # magnitud positiva
            except Exception:
                var95 = float("nan")

            # KPIs: dos de TWR + el resto dinámicos (mismos nombres que en diseño)
            cols = st.columns(10, gap="small")
            with cols[0]: st.metric("Rend. Total (TWR)", f"{twr_total*100:,.2f}%")
            with cols[1]: st.metric("Rend. anualizado",  f"{twr_annual*100:,.2f}%")
            with cols[2]: st.metric("Volatilidad",       f"{vol_ann*100:,.2f}%")
            with cols[3]: st.metric("Beta (IPC)",        f"{beta_ipc:,.2f}")
            with cols[4]: st.metric("Sharpe",            f"{sharpe:,.2f}")
            with cols[5]: st.metric("Sortino (Rf)",      f"{sortino:,.2f}")
            with cols[6]: st.metric("Máx. Drawdown",     f"{max_dd*100:,.2f}%")
            with cols[7]: st.metric("VaR 95%",           f"{var95*100:,.2f}%")
            with cols[8]: st.metric("CETES 28 días",     f"{(rf_annual if np.isfinite(rf_annual) else float('nan'))*100:,.2f}%")
            with cols[9]:
                st.metric("Inflación 12M", f"{infl_12m*100:,.2f}%")

            st.markdown("<div style='margin-top: 70px;'></div>", unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"No se pudieron calcular métricas realizadas: {e}")

    # --- Pretty names/renombres ---
    totales = totales_raw.copy()
    totales["Clase"] = totales["clase"].map(PRETTY).fillna(totales["clase"])
    totales = totales[["Clase", "Peso (%)"]]

    left, right = st.columns([1.1, 1.0], gap="large")

    # ===== Izquierda: tabla CONSAR (operando) =====
    with left:
        pesos_por_clase = {row["clase"]: float(row["Peso (%)"]) for _, row in totales_raw.iterrows()}
        def w(cl): return pesos_por_clase.get(cl, 0.0)

        df_comp_data = consar_limits.get_siefore_composition_data(siefore_name, afore_name)
        consar_rows = [
            {"Agrupación": "Renta Variable",         "Peso (%)": round(w("RV_MX") + w("RV_EXT"), 2), "Límite CONSAR (%)": caps["RV_TOTAL_MAX"]*100},
            {"Agrupación": "Valores Extranjeros",    "Peso (%)": round(w("RV_EXT") + w("FI_EXT"), 2), "Límite CONSAR (%)": caps["EXT_MAX"]*100},
            {"Agrupación": "FIBRAS",                 "Peso (%)": round(w("FIBRA_MX"), 2),             "Límite CONSAR (%)": caps["FIBRA_MX_MAX"]*100},
            {"Agrupación": "Deuda Gubernamental",    "Peso (%)": round(w("FI_GOB_MX"), 2),            "Límite CONSAR (%)": caps.get("FI_GOB_MX_MAX", 1.0)*100},
            {"Agrupación": "Deuda Privada Nacional", "Peso (%)": round(w("FI_CORP_MX"), 2),           "Límite CONSAR (%)": caps.get("FI_CORP_MX_MAX", 1.0)*100},
            {"Agrupación": "Divisas",                "Peso (%)": round(w("FX"), 2),                   "Límite CONSAR (%)": caps.get("FX_MAX", 0.10)*100},
            {"Agrupación": "Estructurados",          "Peso (%)": round(w("SN"), 2),                   "Límite CONSAR (%)": caps.get("SN_MAX", 0.05)*100},
            {"Agrupación": "Mercancías",             "Peso (%)": 0.0,                                 "Límite CONSAR (%)": caps.get("MERCANCÍAS_MAX", 0.0)*100},
        ]

        consar_df = pd.DataFrame(consar_rows)
        if not df_comp_data.empty:
            consar_df = pd.merge(consar_df, df_comp_data, on="Agrupación", how="left").fillna(0.0)
            col_especifica_label = f"{siefore_name} {afore_name if afore_name and afore_name!='N/A' and 'Fallback' not in (afore_name or '') else '—'}"
            col_promedio_label = f"{siefore_name} Promedio"
            consar_df.rename(columns={
                "Agrupación": "Categoría",
                "Peso (%)": "Portafolio (%)",
                "Promedio": col_promedio_label,
                "Especifica": col_especifica_label,
                "Límite CONSAR (%)": "Límites CONSAR TEMP",
            }, inplace=True)
            consar_df["Límites CONSAR"] = consar_df["Límites CONSAR TEMP"]
            COLS_FINALES = ["Categoría", "Portafolio (%)", col_especifica_label, col_promedio_label, "Límites CONSAR"]
            consar_df = consar_df[COLS_FINALES]

            st.dataframe(
                consar_df, hide_index=True, use_container_width=True,
                column_config={
                    "Portafolio (%)": st.column_config.NumberColumn(label="Portafolio", format="%.2f%%"),
                    "Límites CONSAR": st.column_config.NumberColumn(label="Límites CONSAR", format="%.2f%%"),
                    col_promedio_label: st.column_config.NumberColumn(label=col_promedio_label, format="%.2f%%"),
                    col_especifica_label: st.column_config.NumberColumn(label=col_especifica_label, format="%.2f%%"),
                },
            )
        else:
            st.dataframe(
                consar_df.rename(columns={"Agrupación": "Categoría", "Peso (%)": "Portafolio (%)"}),
                hide_index=True, use_container_width=True,
                column_config={
                    "Portafolio (%)": st.column_config.NumberColumn(label="Portafolio", format="%.2f%%"),
                    "Límite CONSAR (%)": st.column_config.NumberColumn(label="Límites CONSAR", format="%.2f%%"),
                },
            )

    # ===== Derecha: Donut + Valor total (operando) =====
    with right:
        if not tabla_para_visualizar.empty:
            # Mostrar métrica de valor total (mismo origen que Operaciones)
            st.metric("Valor Total del Portafolio", f"${total_value:,.2f}")

            slices = []
            for _, row in tabla_para_visualizar.iterrows():
                clase = row["clase"]; peso = float(row["Peso (%)"])
                label = PRETTY.get(clase, clase)
                slices.append({"Clase": label, "Peso": peso})
            pie_df = pd.DataFrame(slices).groupby("Clase", as_index=False)["Peso"].sum()
            fig = px.pie(pie_df, names="Clase", values="Peso", hole=0.35)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay posiciones activas en el portafolio.")

    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

    # ===== Detalle por categoría (operando) =====
    st.subheader("Detalle por categoría")

    tabla_detalle = tabla_para_visualizar.copy()
    tabla_detalle["_is_theme"] = False  # placeholder para compatibilidad

    def has(mask):
        return bool(tabla_detalle.loc[mask, "nombre"].size) if "nombre" in tabla_detalle.columns else False

    cats = []
    if has(tabla_detalle["clase"].eq("RV_MX")):      cats.append(("Renta Variable Nacional",   ("RV_MX", None)))
    if has(tabla_detalle["clase"].eq("RV_EXT")):     cats.append(("Renta Variable Internacional", ("RV_EXT", None)))
    if has(tabla_detalle["clase"].eq("FIBRA_MX")):   cats.append(("FIBRA",                ("FIBRA_MX", None)))
    if has(tabla_detalle["clase"].eq("FI_GOB_MX")):  cats.append(("Deuda Gubernamental",  ("FI_GOB_MX", None)))
    if has(tabla_detalle["clase"].eq("FI_CORP_MX")): cats.append(("Deuda Privada Nacional", ("FI_CORP_MX", None)))
    if has(tabla_detalle["clase"].eq("SN")):         cats.append(("Estructurados",        ("SN", None)))
    if has(tabla_detalle["clase"].eq("FI_EXT")):     cats.append(("Deuda Internacional",  ("FI_EXT", None)))
    if has(tabla_detalle["clase"].eq("FX")):         cats.append(("Divisas",              ("FX", None)))

    cats_all = [("Todas", (None, None))] + cats
    opciones = [lbl for (lbl, _) in cats_all]
    label_to_key = {lbl: key for (lbl, key) in cats_all}

    selected_pretty = st.segmented_control(
        "Selecciona una categoría para ver su detalle:",
        opciones,
        default=("Todas" if "Todas" in opciones else (opciones[0] if opciones else None))
    ) if opciones else None

    if selected_pretty:
        sel_class, _ = label_to_key[selected_pretty]
        if sel_class is None:
            detalle = tabla_detalle.drop(columns=["_is_theme"], errors="ignore").copy()

            def _tipo_label(clase: str) -> str:
                return {
                    "FI_EXT": "Deuda Internacional",
                    "FI_GOB_MX": "Deuda Gubernamental",
                    "FI_CORP_MX": "Deuda Privada Nacional",
                    "FIBRA_MX": "FIBRA",
                    "RV_MX": "Renta Variable Nacional",
                    "RV_EXT": "Renta Variable Internacional",
                    "FX": "Divisa",
                    "SN": "Estructurado",
                }.get(clase, clase)

            if "nombre" in detalle.columns:
                detalle["Tipo"] = detalle["clase"].map(_tipo_label)
        else:
            detalle = tabla_detalle[tabla_detalle["clase"].eq(sel_class)].drop(columns=["_is_theme"], errors="ignore").copy()

        if detalle.empty:
            st.info("No hay instrumentos en esta categoría actualmente.")
        else:
            if "nombre" in detalle.columns:
                bar_df = detalle[["nombre", "Peso (%)"]].sort_values("Peso (%)", ascending=False)
                fig_bar = px.bar(bar_df, x="nombre", y="Peso (%)")
                fig_bar.update_layout(xaxis_title=None, yaxis_title="Peso (%)", margin=dict(l=10, r=10, t=10, b=10), height=380)
                fig_bar.update_xaxes(tickangle=-35, automargin=True)
                st.plotly_chart(fig_bar, use_container_width=True)

            detalle_vista = detalle.drop(columns=["clase"], errors="ignore").copy().rename(columns={
                "nombre": "Empresa",
                "Peso (%)": "Peso",
            })
            ordered_cols = [c for c in ["Empresa", "Tipo", "Peso"] if c in detalle_vista.columns]
            st.dataframe(
                detalle_vista[ordered_cols],
                hide_index=True,
                width='stretch',
                column_config={
                    "Peso": st.column_config.NumberColumn(format="%.2f%%"),
                },
            )

    else:
        st.info("No hay categorías para mostrar.")
