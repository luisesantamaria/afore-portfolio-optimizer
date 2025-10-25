# tabs/structured_note.py
from __future__ import annotations
import math
import uuid
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Nota: Asumo que los módulos tabs.yf_store existen y están accesibles.
try:
    from tabs.yf_store import preload_hist_5y_daily, get_hist_sliced_years
except ImportError:
    # Implementación dummy para evitar errores de importación si no tiene yf_store
    def preload_hist_5y_daily(tickers):
        pass
    def get_hist_sliced_years(ticker, years):
        return pd.DataFrame()

# =========================
# 📌 Utilidades auxiliares
# =========================
def _min_safe(val, minimum):
    """Asegura que el valor no sea menor al mínimo permitido (para evitar StreamlitValueBelowMinError)."""
    try:
        v = float(val)
    except Exception:
        v = minimum
    if not np.isfinite(v):
        v = minimum
    return max(minimum, v)

TRADING_DAYS = 252

def _hist_1y_from_store(ticker: str) -> pd.DataFrame:
    """
    Devuelve un DF con al menos 'Adj Close' para ~1y, leyendo del pool 5y/1d.
    Si no hay datos, devuelve DataFrame vacío.
    """
    try:
        df = get_hist_sliced_years(ticker, 1.0)
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if "Adj Close" not in out.columns:
            if "Close" in out.columns:
                out["Adj Close"] = out["Close"]
            else:
                out["Adj Close"] = np.nan
        return out
    except Exception:
        return pd.DataFrame()

# ========= Utilidades =========
def _phi(x: float) -> float:
    """Normal CDF usando erf (sin SciPy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _bs_call(S, K, r, sigma, T):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _phi(d1) - K * math.exp(-r * T) * _phi(d2)

def _bs_put(S, K, r, sigma, T):
    c = _bs_call(S, K, r, sigma, T)
    # Paridad put-call
    return c - S + K * math.exp(-r * T)

# === Risk-free: leer de lo ya cargado por Renta Fija (sin llamar API) ===
def _mx_risk_free_pct_default() -> float:
    """
    Prioridad:
      1) Buscar en fi_rows un CETES 364d y tomar Rend (%) o Benchmark (%)
      2) Si la etiqueta actual de benchmark menciona CETES 364d, usar fi_bench_rate
      3) Si hay mx_risk_free_pct en sesión, usarlo
      4) Fallback 7.70
    """
    try:
        rows = st.session_state.get("fi_rows", []) or []
        for r in rows:
            bono = str(r.get("Bono", "")).upper()
            tipo = str(r.get("Tipo de bono", "")).upper()
            if "CETES 364D" in bono or (tipo == "BONO GUBERNAMENTAL" and "364" in bono):
                v = r.get("Benchmark (%)")
                if v is None: v = r.get("Rend (%)")
                if v is not None:
                    v = float(v)
                    if np.isfinite(v) and v >= 0: return v
    except Exception:
        pass

    try:
        lbl = str(st.session_state.get("fi_bench_label", "") or "")
        rate = st.session_state.get("fi_bench_rate", None)
        if rate is not None and ("CETES 364" in lbl.upper()):
            v = float(rate)
            if np.isfinite(v) and v >= 0: return v
    except Exception:
        pass

    try:
        v = float(st.session_state.get("mx_risk_free_pct", 7.70))
        if np.isfinite(v) and v >= 0: return v
    except Exception:
        pass

    return 7.70

def _load_universe_from_csv() -> pd.DataFrame:
    """
    Carga universo desde data/constituents_with_yahoo.csv.
    Devuelve columnas normalizadas: ['ticker','name','index_yahoo']
    """
    try:
        df = pd.read_csv("data/constituents_with_yahoo.csv")
        df.columns = [c.strip().lower() for c in df.columns]

        if "name" not in df.columns: raise ValueError("El CSV debe incluir columna 'name'.")
        ticker_col = None
        for cand in ["symbo", "symbol", "ticker"]:
            if cand in df.columns:
                ticker_col = cand
                break
        if not ticker_col: raise ValueError("El CSV debe incluir columna de ticker.")

        out = df[[ticker_col, "name"]].copy()
        out.rename(columns={ticker_col: "ticker"}, inplace=True)
        if "index_yahoo" in df.columns:
            out["index_yahoo"] = df["index_yahoo"]
        elif "index" in df.columns:
            out["index_yahoo"] = df["index"]
        else:
            out["index_yahoo"] = ""

        out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
        out["name"] = out["name"].astype(str).str.strip()
        out["index_yahoo"] = out["index_yahoo"].astype(str).str.strip().str.upper()
        is_mex = out["index_yahoo"].isin(["MEX", "MEX.", "IPC"])
        out.loc[is_mex, "ticker"] = out.loc[is_mex, "ticker"].astype(str) + ".MX"
        out = out.dropna(subset=["ticker", "name"]).drop_duplicates(subset=["ticker"])
        return out[["ticker", "name", "index_yahoo"]]
    except Exception:
        return pd.DataFrame([
            {"ticker": "TSLA", "name": "Tesla", "index_yahoo": ""},
            {"ticker": "CEMEX.MX", "name": "Cemex", "index_yahoo": "MEX"}
        ])

def _parse_ticker_from_label(label: str) -> Tuple[str, str]:
    """De 'Nombre (TICKER)' devuelve (TICKER, Nombre)."""
    s = str(label)
    lpar, rpar = s.rfind("("), s.rfind(")")
    if lpar != -1 and rpar != -1 and rpar > lpar + 1:
        name = s[:lpar].strip()
        tkr = s[lpar + 1 : rpar].strip()
        return tkr, name
    return s.strip(), s.strip()

def _latest_spot_and_vol(ticker: str) -> tuple[float, float]:
    df = _hist_1y_from_store(ticker)
    if df is None or df.empty or "Adj Close" not in df.columns:
        if ticker == "TSLA": return 148.50, 0.40
        if ticker == "CEMEX.MX": return 16.53, 0.58
        return 100.0, 0.25
    s = df["Adj Close"].dropna().astype(float)
    if s.empty: return 100.0, 0.25
    spot = float(s.iloc[-1])
    rets = s.pct_change().dropna()
    vol_ann = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(rets) else 0.25
    if not np.isfinite(vol_ann) or vol_ann <= 0: vol_ann = 0.25
    return spot, vol_ann

def _calculate_and_render_note_details(
    S0: float, K1: float, K2: float, sigma_pct: float, r_pct: float, days: int, title: str,
    render_add_button: bool = False, note_name: str = "",
    mu_ann_pct_saved: Optional[float] = None, vol_ann_pct_saved: Optional[float] = None, sharpe_saved: Optional[float] = None
):
    """Calcula y renderiza las métricas, tabla y gráfica de payoff."""
    sigma = float(sigma_pct) / 100.0
    r = float(r_pct) / 100.0
    T = float(days) / 365.0

    # Cálculos Black-Scholes
    c1 = _bs_call(S0, K1, r, sigma, T)
    p1 = _bs_put(S0, K1, r, sigma, T)
    c2 = _bs_call(S0, K2, r, sigma, T)
    p2 = _bs_put(S0, K2, r, sigma, T)

    # Lógica de cálculo o uso de valores guardados
    if mu_ann_pct_saved is not None and vol_ann_pct_saved is not None and sharpe_saved is not None:
        mu_ann_pct = mu_ann_pct_saved
        vol_ann_pct = vol_ann_pct_saved
        sharpe = sharpe_saved
    else:
        DF = math.exp(-r * T)
        B = max(0.0, 1.0 - DF)
        spread_cost = max(c1 - c2, 0.0)
        n_spreads = (B / spread_cost) if spread_cost > 1e-12 else 0.0

        mu_view = r
        N = 20000
        z = np.random.normal(size=N)
        ST = S0 * np.exp((mu_view - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)

        spread_payoff = np.clip(ST - K1, 0, K2 - K1)
        payoff_note_T = 1.0 + n_spreads * spread_payoff
        ret_T = payoff_note_T - 1.0
        ret_ann = (np.maximum(payoff_note_T, 1e-12)) ** (1.0 / max(T, 1e-9)) - 1.0

        mu_ann_pct = float(np.mean(ret_ann)) * 100.0
        vol_T = float(np.std(ret_T, ddof=1))
        vol_ann_pct = (vol_T / math.sqrt(max(T, 1e-9))) * 100.0
        sharpe = (mu_ann_pct / vol_ann_pct) if vol_ann_pct > 1e-12 else 0.0

    st.subheader(title)
    sA, sB, sC, sD, sE, sF, sG = st.columns(7, gap="small")
    with sA: st.metric("Call(K1) — Prima", f"${c1:,.4f}")
    with sB: st.metric("Put(K1) — Prima",  f"${p1:,.4f}")
    with sC: st.metric("Call(K2) — Prima", f"${c2:,.4f}")
    with sD: st.metric("Put(K2) — Prima",  f"${p2:,.4f}")
    with sE: st.metric("Rendimiento anual (μ)", f"{mu_ann_pct:,.2f}%")
    with sF: st.metric("Volatilidad anual (σ)", f"{vol_ann_pct:,.2f}%")
    with sG: st.metric("Sharpe (μ/σ)", f"{sharpe:,.2f}")

    st.markdown("##### Perfiles de Payoff")
    left, right = st.columns([1.0, 1.1], gap="large")

    start = math.floor(K1 * 2) / 2.0
    end_raw = K2 + 7.0
    end = math.floor(end_raw * 2) / 2.0
    grid_ST = np.arange(start, end + 1e-9, 0.5)

    call_long   = np.maximum(grid_ST - K1, 0.0) - c1
    call_short  = c2 - np.maximum(grid_ST - K2, 0.0)
    call_spread = call_long + call_short

    min_pay, max_pay = float(np.min(call_spread)), float(np.max(call_spread))
    eps = 1e-9
    idx_flat_low  = np.where(np.isclose(call_spread, min_pay,  atol=eps))[0]
    idx_flat_high = np.where(np.isclose(call_spread, max_pay,  atol=eps))[0]
    keep = set(range(len(grid_ST)))
    if len(idx_flat_low) > 3:
        for i in idx_flat_low[3:]: keep.discard(i)
    if len(idx_flat_high) > 3:
        for i in idx_flat_high[:-3]: keep.discard(i)
    mask = np.array([i in keep for i in range(len(grid_ST))])

    with left:
        df_pay = pd.DataFrame({
            "Subyacente (ST)": [f"${x:,.2f}" for x in grid_ST[mask]],
            "Call largo (K1)": [f"${x:,.2f}" for x in call_long[mask]],
            "Call corto (K2)": [f"${x:,.2f}" for x in call_short[mask]],
            "Call spread":     [f"${x:,.2f}" for x in call_spread[mask]],
        })
        st.dataframe(df_pay, hide_index=True, width='stretch')

    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid_ST, y=call_long,   mode="lines", name="Call largo (K1)"))
        fig.add_trace(go.Scatter(x=grid_ST, y=call_short,  mode="lines", name="Call corto (K2)"))
        fig.add_trace(go.Scatter(x=grid_ST, y=call_spread, mode="lines", name="Call spread", line=dict(width=3, dash='solid')))
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Subyacente (ST)",
            yaxis_title="Payoff por unidad",
            height=360, margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    return {
        "mu_ann_pct": mu_ann_pct,
        "vol_ann_pct": vol_ann_pct,
        "sharpe": sharpe
    }

# ========= Render principal (FINAL) =========
def render():

    # ---------- Estado de apertura del expander ----------
    # Arranca cerrado SOLO una vez por sesión / refresh.
    if "sn_expander_open" not in st.session_state:
        st.session_state["sn_expander_open"] = False

    # Universo de activos
    df_uni = _load_universe_from_csv()
    labels = [f"{row['name']} ({row['ticker']})" for _, row in df_uni.iterrows()]
    placeholder = "— Selecciona subyacente —"
    options = [placeholder] + labels

    current_sel_label = st.session_state.get("sn_last_under", placeholder)

    # Notas guardadas
    notes: List[Dict[str, Any]] = st.session_state.get("structured_notes", [])

    # --- Lógica de Selección y Valores Predeterminados ---
    show_label = current_sel_label
    tkr_to_show, name_to_show = _parse_ticker_from_label(show_label) if show_label != placeholder else ("", "")
    ready = show_label != placeholder

    # --- Actualización de Session State (Solo si hay un activo READY) ---
    current_session_ticker = st.session_state.get("sn_under_ticker")
    force_reload = (ready and (not current_session_ticker or
                    (tkr_to_show != "" and current_session_ticker != tkr_to_show))
                   )

    if ready and force_reload:
        try:
            preload_hist_5y_daily([tkr_to_show])
        except Exception:
            pass

        spot_auto, vol_auto = _latest_spot_and_vol(tkr_to_show)

        # ***** LÓGICA DE INICIALIZACIÓN DE PARÁMETROS *****
        st.session_state["sn_under_ticker"] = tkr_to_show
        st.session_state["sn_under_name"] = name_to_show
        st.session_state["sn_last_under"] = show_label

        st.session_state["sn_spot"] = float(_min_safe(spot_auto, 0.01))
        st.session_state["sn_sigma_pct"] = float(_min_safe(vol_auto * 100.0, 0.01))
        st.session_state["sn_k1"] = round(0.97 * spot_auto, 4)
        st.session_state["sn_k2"] = round(1.25 * spot_auto, 4)
        st.session_state["sn_days"] = 360
        st.session_state["sn_r_pct"] = float(_mx_risk_free_pct_default())

        # ⚠️ NO abrir aquí para no forzar apertura en cache miss.
        # st.session_state["sn_expander_open"] = True

    sel_label = show_label
    current_idx = options.index(show_label) if show_label in options else 0

    # Variables para capturar las métricas del análisis actual antes de agregarlas
    current_metrics = {}

    # --- Expander para Inputs y Análisis Actual ---
    with st.expander("Agregar Nota Estructurada", expanded=st.session_state.get("sn_expander_open", False)):

        # ---- Fila 1: Subyacente | Spot | K1 | K2 ----
        r1c1, r1c2, r1c3, r1c4 = st.columns(4, gap="small")
        with r1c1:
            sel_label = st.selectbox(
                "Subyacente",
                options=options,
                index=current_idx,
                key="sn_under_sel",
            )
            # Si cambia, ACTUALIZA y PRESERVA el estado de apertura (no abre si estaba cerrado)
            if sel_label != st.session_state.get("sn_last_under"):
                prev_open = st.session_state.get("sn_expander_open", False)
                st.session_state["sn_last_under"] = sel_label
                st.session_state["sn_expander_open"] = prev_open  # preserva
                st.rerun()

        with r1c2:
            S0 = st.number_input(
                "Spot (S₀)", min_value=0.01, format="%.4f", key="sn_spot", disabled=not ready,
                value=_min_safe(st.session_state.get("sn_spot", 0.0), 0.01) if ready else 0.01
            )

        with r1c3:
            K1 = st.number_input(
                "K1 (call largo)", min_value=0.01, format="%.4f", key="sn_k1", disabled=not ready,
                value=_min_safe(st.session_state.get("sn_k1", 0.0), 0.01) if ready else 0.01
            )

        with r1c4:
            K2 = st.number_input(
                "K2 (call corto)", min_value=0.01, format="%.4f", key="sn_k2", disabled=not ready,
                value=_min_safe(st.session_state.get("sn_k2", 0.0), 0.01) if ready else 0.01
            )

        # ---- Fila 2: Vol | Tasa | Plazo | Botón Agregar Nota ----
        r2c1, r2c2, r2c3, r2c4 = st.columns([1, 1, 0.8, 0.4], gap="small")

        with r2c1:
            sigma_pct = st.number_input(
                "Volatilidad anual (%)", min_value=0.01, format="%.2f", key="sn_sigma_pct", disabled=not ready,
                value=_min_safe(st.session_state.get("sn_sigma_pct", 0.0), 0.01) if ready else 0.01
            )

        with r2c2:
            r_pct = st.number_input(
                "Tasa libre de riesgo anual (%)", min_value=0.00, format="%.2f", key="sn_r_pct", disabled=not ready,
                value=_min_safe(st.session_state.get("sn_r_pct", _mx_risk_free_pct_default()), 0.00) if ready else 0.00
            )

        with r2c3:
            _days_default = int(_min_safe(st.session_state.get("sn_days", 360), 7)) if ready else 7
            days = st.number_input(
                "Plazo (días)", min_value=7, step=1, key="sn_days", disabled=not ready,
                value=_days_default
            )

        with r2c4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Agregar nota", key="add_note_btn_expander", type="primary", disabled=not ready, use_container_width=True):
                st.session_state.setdefault("structured_notes", [])
                name_current = st.session_state.get("sn_under_name", "ACTIVO DESCONOCIDO")
                ticker_current = st.session_state.get("sn_under_ticker", tkr_to_show)

                # <<< NUEVO: name = "Nota " + subyacente >>>
                note_name_composed = f"Nota {name_current}" if name_current else f"Nota {ticker_current or 'SIN_SUBYACENTE'}"

                note = {
                    "note_id": uuid.uuid4().hex,
                    "name": note_name_composed,                         # <<< nuevo campo
                    "subyacente": name_current,
                    "ticker": ticker_current,                           # ← persistimos ticker
                    "spot": float(S0),
                    "K1": float(K1),
                    "K2": float(K2),
                    "plazo_d": int(days),
                    "r_pct": float(r_pct),
                    "vol_pct": float(sigma_pct),
                    "mu_ann_pct": current_metrics.get("mu_ann_pct", 0.0),
                    "vol_ann_pct": current_metrics.get("vol_ann_pct", 0.0),
                    "sharpe": current_metrics.get("sharpe", 0.0),
                    "Borrar": False,
                }
                st.session_state["structured_notes"].append(note)
                # Si quieres que NO se abra automáticamente, comenta la línea siguiente:
                st.session_state["sn_expander_open"] = True
                st.toast("Nota agregada a la tabla (no guardada aún).", icon="✅")

        # ========= Cálculos y UI dentro del expander (ANÁLISIS ACTUAL) =========
        if ready:
            S0_current = st.session_state.get("sn_spot", S0)
            K1_current = st.session_state.get("sn_k1", K1)
            K2_current = st.session_state.get("sn_k2", K2)
            sigma_pct_current = st.session_state.get("sn_sigma_pct", sigma_pct)
            r_pct_current = st.session_state.get("sn_r_pct", r_pct)
            days_current = st.session_state.get("sn_days", days)
            name_current = st.session_state.get("sn_under_name", "ACTIVO DESCONOCIDO")

            current_metrics = _calculate_and_render_note_details(
                S0=S0_current, K1=K1_current, K2=K2_current,
                sigma_pct=sigma_pct_current, r_pct=r_pct_current, days=days_current,
                title=f"Primas y Métricas para **{name_current}** (Modelo Black-Scholes)",
                render_add_button=False,
                note_name=name_current
            )
        else:
            st.info("Selecciona un subyacente para ver el análisis en curso.")

    # ========= Tabla de notas =========
    notes = st.session_state.get("structured_notes", [])
    num_notas = len(notes)
    st.caption(f"{num_notas} Nota(s) Seleccionada(s)")

    if not notes:
        st.info("No hay notas en la tabla. Agrega una con el botón 'Agregar nota' en el análisis actual.")
        return
    
    # Backfill de tickers y name para notas viejas (una sola vez basta)
    try:
        uni = _load_universe_from_csv()
        name2tkr = {str(r["name"]).strip(): str(r["ticker"]).strip() for _, r in uni.iterrows()}
        changed = False
        for n in st.session_state.get("structured_notes", []):
            # ticker faltante
            if not n.get("ticker"):
                tkr_guess = name2tkr.get(str(n.get("subyacente","")).strip())
                if tkr_guess:
                    n["ticker"] = tkr_guess
                    changed = True
            # name faltante -> "Nota " + subyacente (o ticker)
            if not n.get("name"):
                base = str(n.get("subyacente") or n.get("ticker") or "").strip()
                if base:
                    n["name"] = f"Nota {base}"
                    changed = True
        if changed:
            st.toast("Se completaron campos faltantes (ticker/name) en notas existentes.", icon="🔁")
            st.rerun()
    except Exception:
        pass

    df_notes = pd.DataFrame(notes).copy()
    if "note_id" not in df_notes.columns:
        df_notes["note_id"] = [uuid.uuid4().hex for _ in range(len(df_notes))]
    df_notes = df_notes.set_index("note_id", drop=False)

    if "ticker" not in df_notes.columns:
        df_notes["ticker"] = ""
    if "name" not in df_notes.columns:
        df_notes["name"] = ""  # aseguramos la columna aunque no la mostremos

    display_cols = [
        "ticker",
        "subyacente", "spot", "vol_pct", "K1", "K2", "plazo_d", "r_pct",
        "mu_ann_pct", "vol_ann_pct", "sharpe", "Borrar"
        # 'name' se guarda en session_state pero no es necesario mostrarlo aquí.
    ]
    for c in display_cols:
        if c not in df_notes.columns: df_notes[c] = np.nan if c != "Borrar" else False

    edited = st.data_editor(
        df_notes[display_cols],
        key="structured_notes_editor",
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "ticker":     st.column_config.TextColumn("Ticker (Yahoo)"),
            "subyacente": st.column_config.TextColumn("Subyacente"),
            "spot":       st.column_config.NumberColumn("Spot", format="%.4f"),
            "vol_pct":    st.column_config.NumberColumn("Volatilidad (%)", format="%.2f"),
            "K1":         st.column_config.NumberColumn("K1", format="%.4f"),
            "K2":         st.column_config.NumberColumn("K2", format="%.4f"),
            "plazo_d":    st.column_config.NumberColumn("Plazo (días)", format="%d"),
            "r_pct":      st.column_config.NumberColumn("Tasa libre (%)", format="%.2f"),
            "mu_ann_pct": st.column_config.NumberColumn("Rend. anual (%)", format="%.2f"),
            "vol_ann_pct":st.column_config.NumberColumn("Vol. anual (%)", format="%.2f"),
            "sharpe":     st.column_config.NumberColumn("Sharpe (μ/σ)", format="%.2f"),
            "Borrar":     st.column_config.CheckboxColumn("Eliminar", default=False),
        },
    )
    edited.index = df_notes.index

    to_delete_ids = edited.index[edited["Borrar"] == True].tolist()
    if to_delete_ids:
        st.session_state["structured_notes"] = [
            n for n in st.session_state["structured_notes"]
            if n.get("note_id") not in to_delete_ids
        ]
        st.toast(f"Se eliminó {len(to_delete_ids)} nota(s).", icon="🗑️")
        st.rerun()

    # Sincronizamos los cambios editados (no editamos 'name' desde la tabla)
    edited_no_del = edited.drop(columns=["Borrar"], errors="ignore")
    merged = edited_no_del.reset_index().to_dict(orient="records")
    new_list = []
    for row in merged:
        nid = row.get("note_id")
        orig = next((n for n in st.session_state["structured_notes"] if n.get("note_id") == nid), None)
        if orig is None:
            continue
        for k, v in row.items():
            if k in ["note_id"]:
                continue
            # mantenemos 'name' tal como está en session_state (no editable desde la tabla)
            orig[k] = v
        # backstop: si por algún motivo 'name' está vacío aquí, lo regeneramos
        if not orig.get("name"):
            base = str(orig.get("subyacente") or orig.get("ticker") or "").strip()
            if base:
                orig["name"] = f"Nota {base}"
        new_list.append(orig)

    # conservar notas que no estaban en el editor
    existing_ids = {x["note_id"] for x in new_list}
    for n in st.session_state["structured_notes"]:
        if n.get("note_id") not in existing_ids:
            # idem: asegurar 'name'
            if not n.get("name"):
                base = str(n.get("subyacente") or n.get("ticker") or "").strip()
                if base:
                    n["name"] = f"Nota {base}"
            new_list.append(n)

    st.session_state["structured_notes"] = new_list

    # ========= Análisis de la Primera Nota Guardada =========
    first_note = st.session_state["structured_notes"][0]
    _calculate_and_render_note_details(
        S0=first_note["spot"],
        K1=first_note["K1"],
        K2=first_note["K2"],
        sigma_pct=first_note["vol_pct"],
        r_pct=first_note["r_pct"],
        days=first_note["plazo_d"],
        title=f"Primas y Métricas de la Nota Estructurada de **{first_note['subyacente']}** (Modelo Black-Scholes)",
        render_add_button=False,
        mu_ann_pct_saved=first_note["mu_ann_pct"],
        vol_ann_pct_saved=first_note["vol_ann_pct"],
        sharpe_saved=first_note["sharpe"]
    )
