# tabs/ledger.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Dict, Any, List
from datetime import date, datetime
import math
import re

import numpy as np
import pandas as pd
import streamlit as st

from tabs import runtime  # get_act_as_of(), get_ops_started_at()
from tabs.consar_limits import get_latest_net_asset_value, get_commission_value
from tabs.yf_store import get_hist_5y

# =========================
# Constantes / tipos
# =========================
KIND = Literal["BUY", "SELL", "DEPOSIT", "WITHDRAW"]
LEDGER_COLUMNS = ["ts", "kind", "symbol", "qty", "px", "cash", "note", "name"]

@dataclass
class Tx:
    ts: date | datetime
    kind: KIND
    symbol: str = ""
    qty: float = 0.0
    px: float = 0.0
    cash: float = 0.0
    note: str = ""
    name: str = ""   # nombre legible del instrumento (bono/activo)

# =========================
# Helpers internos
# =========================
def _today() -> date:
    return datetime.now().date()

def _fmt_money(x: Any) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "—"

def _sort_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena el ledger por fecha, tipo y símbolo."""
    if df.empty:
        return df

    kind_order = {"DEPOSIT": 1, "BUY": 2, "SELL": 3, "WITHDRAW": 4}
    df = df.copy()
    df["_kind_order"] = df["kind"].map(kind_order).fillna(99)
    df["_ts_sort"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["_ts_sort", "_kind_order", "symbol"], ascending=[True, True, True])
    df = df.drop(columns=["_kind_order", "_ts_sort"])
    return df.reset_index(drop=True)

def _safe_adj_close_series(df: Optional[pd.DataFrame]) -> pd.Series:
    """Serie de Adj Close como float, sin NaNs."""
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

# =========================
# API pública: estado
# =========================
def ensure_ledger_initialized(ss) -> None:
    if "ops_ledger" not in ss or not isinstance(ss.get("ops_ledger"), pd.DataFrame):
        ss["ops_ledger"] = pd.DataFrame(columns=LEDGER_COLUMNS)

def reset_ledger_seed_flag(ss) -> None:
    ss["ops_ledger_seeded"] = False

# =========================
# API pública: lectura y agregado
# =========================
def get_ledger(ss) -> pd.DataFrame:
    ensure_ledger_initialized(ss)
    df = ss["ops_ledger"].copy()
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df[[c for c in LEDGER_COLUMNS if c in df.columns]]
    return df

def get_ledger_until_act_as_of(ss) -> pd.DataFrame:
    """
    Obtiene el ledger hasta la fecha Act As Of, ordenado por:
    1. Fecha (ascendente)
    2. Tipo (DEPOSIT, BUY, SELL, WITHDRAW)
    3. Símbolo (alfabético)
    """
    df = get_ledger(ss)
    if df.empty:
        return df

    act_as_of = runtime.get_act_as_of() or _today()
    df["ts"] = pd.to_datetime(df["ts"]).dt.date
    df = df.loc[df["ts"] <= act_as_of].copy()
    return _sort_ledger(df)

def add_transaction(
    ss,
    ts: date | datetime,
    kind: KIND,
    symbol: str = "",
    qty: float = 0.0,
    px: float = 0.0,
    cash: float = 0.0,
    note: str = "",
    name: str = "",
) -> None:
    ensure_ledger_initialized(ss)

    kind = str(kind).upper().strip()
    symbol = str(symbol or "").strip()
    qty = float(qty or 0.0)
    px = float(px or 0.0)
    cash = float(cash or 0.0)
    note = str(note or "")

    if kind in ("BUY", "SELL"):
        if qty <= 0 or px <= 0:
            raise ValueError("Para BUY/SELL, 'qty' y 'px' deben ser > 0.")
        if abs(cash) < 1e-9:
            cash = (-qty * px) if kind == "BUY" else (qty * px)
        if not symbol:
            raise ValueError("Para BUY/SELL, 'symbol' no puede estar vacío.")
    elif kind == "DEPOSIT":
        if cash <= 0:
            raise ValueError("Para DEPOSIT, 'cash' debe ser positivo.")
        qty = 0.0
        px = 0.0
        symbol = ""
    elif kind == "WITHDRAW":
        if cash >= 0:
            raise ValueError("Para WITHDRAW, 'cash' debe ser negativo.")
        qty = 0.0
        px = 0.0
        symbol = ""
    else:
        raise ValueError("Tipo de transacción inválido.")

    new_row = {
        "ts": pd.to_datetime(ts),
        "kind": kind,
        "symbol": symbol,
        "qty": qty,
        "px": px,
        "cash": cash,
        "note": note.strip(),
        "name": str(name or "").strip(),
    }
    ss["ops_ledger"] = pd.concat([ss["ops_ledger"], pd.DataFrame([new_row])], ignore_index=True)
    ss["ops_ledger"] = _sort_ledger(ss["ops_ledger"])

    # ---------- AUTO: marcar recálculo de comisiones ----------
    # Evitar recursión cuando nosotros mismos agregamos comisiones/auto-ventas
    is_fee_tx = (kind == "WITHDRAW") and ("COMISION" in note.upper())
    if not ss.get("_suppress_fee_autorecalc", False) and not is_fee_tx:
        tsd = pd.to_datetime(ts).date()
        prev = ss.get("_fee_recalc_needed_from")
        ss["_fee_recalc_needed_from"] = min(prev, tsd) if prev else tsd

# =========================
# Render: expander ligero
# =========================
def render_expander(ss, *, expanded: bool = False) -> None:
    ensure_ledger_initialized(ss)
    act_as_of = runtime.get_act_as_of() or _today()

    # -------- AUTO: si hay un recálculo pendiente de comisiones, ejecútalo --------
    pending_from = ss.get("_fee_recalc_needed_from")
    if pending_from:
        try:
            _ = apply_daily_fees_and_autosell(
                ss,
                start_date=pending_from,
                end_date=act_as_of,
                debug=False,
            )
            ss["_fee_recalc_needed_from"] = None
            st.rerun()
        except Exception as e:
            st.warning(f"No se pudieron aplicar comisiones automáticas: {e}")
            ss["_fee_recalc_needed_from"] = None

    with st.expander("📘 Ledger (transacciones hasta Act As Of)", expanded=expanded):
        ledger_view = get_ledger_until_act_as_of(ss)

        st.markdown("**Histórico hasta Act As Of**")
        if not ledger_view.empty:
            show = ledger_view.copy()
            show.rename(
                columns={
                    "ts": "Fecha",
                    "kind": "Tipo",
                    "name": "Nombre",
                    "symbol": "Símbolo",
                    "qty": "Cantidad",
                    "px": "Precio",
                    "cash": "Efectivo",
                    "note": "Nota",
                },
                inplace=True,
            )

            def _monto(row):
                if row["Tipo"] in ("BUY", "SELL"):
                    try:
                        return float(row["Cantidad"]) * float(row["Precio"])
                    except Exception:
                        return np.nan
                return np.nan

            show["Monto (qty*px)"] = show.apply(_monto, axis=1)
            if "Precio" in show.columns:
                show["Precio"] = show["Precio"].apply(_fmt_money)
            if "Efectivo" in show.columns:
                show["Efectivo"] = show["Efectivo"].apply(_fmt_money)
            if "Monto (qty*px)" in show.columns:
                show["Monto (qty*px)"] = show["Monto (qty*px)"].apply(
                    lambda v: _fmt_money(v) if pd.notna(v) else "—"
                )

            st.dataframe(show, hide_index=True, use_container_width=True)
        else:
            st.info("Sin transacciones registradas hasta la fecha seleccionada.")

        st.markdown("---")
        st.markdown("**Agregar transacción**")
        with st.form("add_tx_form", clear_on_submit=True):
            c1, c2, c3 = st.columns([1, 1, 1])
            ts_new = c1.date_input("Fecha", value=act_as_of, max_value=_today(), key="ledger_ts_input")
            kind_new = c2.selectbox("Tipo", options=["BUY", "SELL", "DEPOSIT", "WITHDRAW"], index=0, key="ledger_kind_input")
            symbol_new = c3.text_input("Símbolo (ticker)", value="", key="ledger_symbol_input")

            name_new = st.text_input("Nombre (opcional)", value="", key="ledger_name_input")

            c4, c5, c6 = st.columns([1, 1, 1])
            qty_new = c4.number_input("Cantidad (qty)", min_value=0.0, step=1.0, value=0.0, key="ledger_qty_input")
            px_new = c5.number_input("Precio (px)", min_value=0.0, step=0.01, value=0.0, key="ledger_px_input")
            cash_new = c6.number_input(
                "Efectivo (cash)",
                step=0.01,
                value=0.0,
                key="ledger_cash_input",
                help="Cero = se infiere para BUY/SELL; para DEPOSIT/WITHDRAW especifica signo.",
            )
            note_new = st.text_input("Nota (opcional)", value="", key="ledger_note_input")

            if st.form_submit_button("➕ Agregar"):
                try:
                    add_transaction(
                        ss,
                        ts=ts_new,
                        kind=kind_new,
                        symbol=symbol_new,
                        qty=qty_new,
                        px=px_new,
                        cash=cash_new,
                        note=note_new,
                        name=name_new,
                    )
                    st.success("Transacción agregada.")
                    st.rerun()
                except Exception as e:
                    st.error(f"No se pudo agregar la transacción: {e}")

# =========================
# Reconstrucción del portafolio desde el ledger
# =========================
def rebuild_portfolio_from_ledger(
    ss,
    start_date: date,
    end_date: date,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def _safe_adj_close_series_local(df: Optional[pd.DataFrame]) -> pd.Series:
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

    ensure_ledger_initialized(ss)
    ledger_df = get_ledger_until_act_as_of(ss)
    if ledger_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    bdays = pd.bdate_range(start_date, end_date, freq="B")
    if len(bdays) == 0:
        return pd.DataFrame(), pd.DataFrame()

    tickers_needed = ledger_df[ledger_df["kind"].isin(["BUY", "SELL"])]["symbol"].unique()
    tickers_needed = [t for t in tickers_needed if t and str(t).strip()]

    if not tickers_needed:
        ledger_df = ledger_df.sort_values("ts")
        cash_balance = 0.0
        values = []
        for current_date in bdays:
            txs_today = ledger_df[ledger_df["ts"] == pd.Timestamp(current_date).date()]
            for _, tx in txs_today.iterrows():
                cash_balance += float(tx["cash"])
            values.append({"cash": cash_balance, "market_value": 0.0, "total_value": cash_balance})
        ts_df = pd.DataFrame(values, index=bdays)
        return pd.DataFrame(), ts_df

    prices: Dict[str, pd.Series] = {}
    synthetic_prices_cache = ss.get("synthetic_prices", {})

    # ⭐ Cargar tipo de cambio USD/MXN
    usd_mxn_series = pd.Series(index=bdays, dtype=float)
    try:
        usd_mxn_df = get_hist_5y("MXN=X")
        usd_mxn_adj = _safe_adj_close_series_local(usd_mxn_df)
        if not usd_mxn_adj.empty:
            usd_mxn_series = usd_mxn_adj.reindex(bdays, method="pad")
    except Exception:
        pass

    with st.spinner(f"📊 Cargando precios de {len(tickers_needed)} instrumentos..."):
        for ticker in tickers_needed:
            try:
                # ⭐ Obtener clase del activo (si está en synthetic cache)
                ticker_metadata = synthetic_prices_cache.get(ticker, {})
                clase = str(ticker_metadata.get("clase", "")).strip()

                if ticker in synthetic_prices_cache and "price_series" in synthetic_prices_cache[ticker]:
                    price_series = synthetic_prices_cache[ticker]["price_series"]
                    if isinstance(price_series, pd.Series) and not price_series.empty:
                        prices[ticker] = price_series.reindex(bdays, method="pad")
                        continue

                if not ticker.startswith("SYNTH_"):
                    df = get_hist_5y(ticker)
                    adj = _safe_adj_close_series_local(df)
                    if not adj.empty:
                        adj_reindexed = adj.reindex(bdays, method="pad")

                        # ⭐ Convertir RV_EXT a MXN
                        if clase == "RV_EXT" and not usd_mxn_series.empty:
                            adj_reindexed = adj_reindexed.multiply(usd_mxn_series, fill_value=np.nan).dropna()

                        # ⭐ Convertir FX a MXN si no es MXN=X
                        elif clase == "FX":
                            # Serie base del par tal como viene de Yahoo (adj_reindexed)
                            tkr_upper = str(ticker).upper()

                            # 1) Determinar si hay que invertir el par (heurística + selección del UI)
                            inverse = False
                            # Heurística: tickers de 3 letras + '=X' suelen ser USD/XXX (ej. 'BRL=X'), y hay que invertirlos
                            is_usd_cross = bool(re.fullmatch(r"^[A-Z]{3}=X$", tkr_upper)) and tkr_upper not in {"MXN=X", "USD=X"}
                            if is_usd_cross:
                                inverse = True

                            # Preferir bandera del UI si existe
                            fx_sel = ss.get("fx_selection", {}) or {}
                            for _, meta in fx_sel.items():
                                fx_tkr = str(meta.get("yahoo_ticker", "")).upper()
                                if tkr_upper == fx_tkr:
                                    inverse = bool(meta.get("inverse", inverse))
                                    break

                            series = adj_reindexed.copy()
                            if inverse:
                                series = 1.0 / series

                            # 2) Si el par ya está contra MXN (p. ej., 'EURMXN=X' o 'MXN=X') no convertir más
                            if ("MXN" in tkr_upper) or (tkr_upper == "MXN=X"):
                                adj_reindexed = series.dropna()
                            else:
                                # 3) Si el par está contra USD (p. ej., 'EURUSD=X' o invertido 'USD por BRL'),
                                #    multiplicar por USD/MXN para llevar a MXN
                                if usd_mxn_series is not None and not usd_mxn_series.empty:
                                    adj_reindexed = series.multiply(usd_mxn_series, fill_value=np.nan).dropna()
                                else:
                                    adj_reindexed = series.dropna()


                        prices[ticker] = adj_reindexed
                    else:
                        prices[ticker] = pd.Series(index=bdays, dtype=float)
                    continue

                px_inicial_fallback = synthetic_prices_cache.get(ticker, {}).get("px_inicial", 100.0)
                prices[ticker] = pd.Series(px_inicial_fallback, index=bdays)
            except Exception:
                prices[ticker] = pd.Series(index=bdays, dtype=float)

    # Ordenar igual que get_ledger_until_act_as_of
    kind_order = {"DEPOSIT": 1, "BUY": 2, "SELL": 3, "WITHDRAW": 4}
    ledger_df["_kind_order"] = ledger_df["kind"].map(kind_order).fillna(99)
    ledger_df = ledger_df.sort_values(["ts", "_kind_order", "symbol"], ascending=[True, True, True]).copy()
    ledger_df = ledger_df.drop(columns=["_kind_order"])
    ledger_df["ts"] = pd.to_datetime(ledger_df["ts"]).dt.date

    holdings: Dict[str, float] = {}
    cash = 0.0
    portfolio_values: List[Dict[str, float]] = []

    for current_date in bdays:
        current_date_py = current_date.date()
        txs_today = ledger_df[ledger_df["ts"] == current_date_py]

        for _, tx in txs_today.iterrows():
            kind = str(tx["kind"]).strip().upper()
            if kind == "BUY":
                symbol = str(tx["symbol"]).strip()
                qty = float(tx["qty"])
                holdings[symbol] = holdings.get(symbol, 0.0) + qty
                cash += float(tx["cash"])
            elif kind == "SELL":
                symbol = str(tx["symbol"]).strip()
                qty = float(tx["qty"])
                holdings[symbol] = holdings.get(symbol, 0.0) - qty
                cash += float(tx["cash"])
            elif kind == "DEPOSIT":
                cash += float(tx["cash"])
            elif kind == "WITHDRAW":
                cash += float(tx["cash"])

        market_value = 0.0
        for ticker, qty in holdings.items():
            if qty > 0 and ticker in prices:
                try:
                    px = prices[ticker].loc[current_date]
                    if pd.notna(px) and px > 0:
                        market_value += qty * px
                except Exception:
                    pass

        total_value = cash + market_value
        portfolio_values.append({"cash": cash, "market_value": market_value, "total_value": total_value})

    ts_df = pd.DataFrame(portfolio_values, index=bdays)

    # ⭐ DEBUG: Log para rebuild
    if "rebuild_debug_log" not in ss:
        ss["rebuild_debug_log"] = []
    ss["rebuild_debug_log"].clear()

    positions = []
    for ticker, qty in holdings.items():
        if qty > 1e-6 and ticker in prices:
            try:
                px_current = prices[ticker].iloc[-1]
                if pd.notna(px_current) and px_current > 0:
                    ticker_meta = synthetic_prices_cache.get(ticker, {})
                    clase = ticker_meta.get("clase", "")
                    activo = ticker_meta.get("activo", "")
                    ss["rebuild_debug_log"].append(
                        f"Ticker: {ticker} | En cache: {ticker in synthetic_prices_cache} | "
                        f"Clase: '{clase}' | Activo: '{activo}'"
                    )
                    positions.append({
                        "ticker": ticker,
                        "qty": qty,
                        "px_actual": px_current,
                        "valor_actual": qty * px_current,
                        "clase": clase,
                        "activo": activo
                    })
            except Exception:
                pass

    pos_df = pd.DataFrame(positions)
    if not pos_df.empty and ts_df["total_value"].iloc[-1] > 0:
        total_val = ts_df["total_value"].iloc[-1]
        pos_df["peso_actual_%"] = pos_df["valor_actual"] / total_val * 100.0
    return pos_df, ts_df

# =========================
# Helpers compartidos y construcción inicial
# =========================
def _safe_adj_close_series_init(df: Optional[pd.DataFrame]) -> pd.Series:
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

def _bizdays_index(start: date, end: date) -> pd.DatetimeIndex:
    if not start or not end:
        return pd.DatetimeIndex([])
    start_safe = min(start, end)
    end_safe = max(start, end)
    return pd.bdate_range(pd.Timestamp(start_safe), pd.Timestamp(end_safe), freq="B")

def _canon(s: str) -> str:
    if not s:
        return ""
    x = s.strip().lower()
    x = x.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    x = re.sub(r"\s+", " ", x)
    return x

def _find_weight(df_opt: pd.DataFrame, clase: Optional[str], target: str) -> Optional[float]:
    if "Peso (%)" not in df_opt.columns or "nombre" not in df_opt.columns:
        return None
    tgt = _canon(target)

    def _scan(df):
        for _, r in df.iterrows():
            nm = _canon(str(r.get("nombre", "")))
            if nm == tgt or (nm and (nm in tgt or tgt in nm)):
                val = float(pd.to_numeric(r.get("Peso (%)", np.nan), errors="coerce"))
                return val
        return None

    if clase is not None and "clase" in df_opt.columns:
        cand = _scan(df_opt[df_opt["clase"].eq(clase)])
        if cand is not None and np.isfinite(cand):
            return cand
    cand = _scan(df_opt)
    if cand is not None and np.isfinite(cand):
        return cand
    return None

def _phi_std(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _bs_call_price(S: float, K: float, r: float, sigma: float, T_yrs: float) -> float:
    if S <= 0 or K <= 0 or sigma <= 0 or T_yrs <= 0:
        return max(S - K, 0.0)
    rt = math.sqrt(T_yrs)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T_yrs) / (sigma * rt)
    d2 = d1 - sigma * rt
    return S * _phi_std(d1) - K * math.exp(-r * T_yrs) * _phi_std(d2)

def _note_series_bs(
    S_hist: pd.Series,
    S0: float,
    K1: float,
    K2: float,
    r_pct: float,
    sigma_pct: float,
    days_total: int,
    bdays: pd.DatetimeIndex,
    price_unit: float = 10.0,
) -> pd.Series:
    """
    Modelo de nota estructurada idéntico al Excel:
    - Valor nominal CETES: price_unit (ej: $10)
    - Número de spreads: fraccional
    - Precio inicial ≈ price_unit (con ligera variación por valuación de opciones)
    """
    r = max(float(r_pct), 0.0) / 100.0
    sigma = max(float(sigma_pct), 1e-8) / 100.0
    T_years_total = max(int(days_total), 1) / 365.0

    DF = math.exp(-r * T_years_total)
    precio_cetes_hoy = price_unit * DF
    presupuesto_opciones = price_unit - precio_cetes_hoy

    costo_spread_unitario = max(
        _bs_call_price(S0, K1, r, sigma, T_years_total) -
        _bs_call_price(S0, K2, r, sigma, T_years_total),
        0.0,
    )

    if costo_spread_unitario <= 1e-12:
        t_days = np.arange(len(bdays))
        factor_crecimiento = ((1.0 + r) ** (t_days / 360.0))
        Z_t = precio_cetes_hoy * factor_crecimiento
        return pd.Series(Z_t, index=bdays, dtype=float)

    n_spreads = presupuesto_opciones / costo_spread_unitario

    t_days = np.arange(len(bdays))
    factor_crecimiento = ((1.0 + r) ** (t_days / 360.0))
    Z_t = pd.Series(precio_cetes_hoy * factor_crecimiento, index=bdays, dtype=float)

    opt_vals = np.zeros(len(bdays), dtype=float)
    for i in range(len(bdays)):
        S_t = float(S_hist.iloc[i])
        T_rem_days = max(days_total - i, 0)
        T_rem = T_rem_days / 365.0
        if T_rem > 0:
            c1 = _bs_call_price(S_t, K1, r, sigma, T_rem)
            c2 = _bs_call_price(S_t, K2, r, sigma, T_rem)
            opt_vals[i] = n_spreads * (c1 - c2)
        else:
            payoff_spread = max(S_t - K1, 0.0) - max(S_t - K2, 0.0)
            opt_vals[i] = n_spreads * payoff_spread

    OPT_t = pd.Series(opt_vals, index=bdays, dtype=float)
    V_t = Z_t + OPT_t
    return V_t.fillna(method="ffill").fillna(method="bfill")

def _unit_cost_from_dirty_nominal(dirty: float, nominal: float) -> float:
    try:
        d = float(dirty)
        n = float(nominal)
    except Exception:
        return np.nan
    if not np.isfinite(d) or not np.isfinite(n) or n <= 0:
        return np.nan
    if (n <= 20 and d <= n * 1.5) or d <= 15:
        return d
    return (d / 100.0) * n

# =========================
# Inicialización y build inicial
# =========================
def init_state(ss) -> None:
    ss.setdefault("ops_operating", False)
    ss.setdefault("ops_start_date", _today())
    ss.setdefault("ops_seed_capital", None)
    ss.setdefault("ops_seed_capital_str", None)
    ss.setdefault("ops_seed_source_date", None)
    ss.setdefault("ops_mgmt_fee_pp", None)
    ss.setdefault("ops_snapshot", {})
    ss.setdefault("ops_positions", pd.DataFrame())
    ss.setdefault("ops_timeseries", pd.DataFrame())
    ss.setdefault("structured_notes", [])

    siefore = ss.get("siefore_selected")
    afore = ss.get("afore_selected")

    if ss.get("ops_seed_capital") is None and siefore and afore:
        try:
            val_raw, fecha = get_latest_net_asset_value(siefore, afore)
            if val_raw is not None:
                vr = float(val_raw)
                pesos = vr if vr > 1e7 else vr * 1_000_000.0
                ss["ops_seed_capital"] = pesos
                ss["ops_seed_capital_str"] = f"{pesos:,.2f}"
            if fecha is not None:
                ss["ops_seed_source_date"] = fecha.date() if hasattr(fecha, "date") else None
        except Exception:
            pass

    if ss.get("ops_mgmt_fee_pp") is None and siefore and afore:
        try:
            fee_val = get_commission_value(siefore, afore)
            if fee_val is not None:
                x = float(fee_val)
                if x > 10:
                    x = x / 100.0
                elif 0 <= x < 0.01:
                    x = x * 100.0
                ss["ops_mgmt_fee_pp"] = float(x)
        except Exception:
            pass

    if ss.get("ops_seed_capital_str") is None:
        cap = ss.get("ops_seed_capital") or 0.0
        ss["ops_seed_capital_str"] = f"{float(cap):,.2f}"

def build_initial_portfolio(
    start_dt: date,
    seed_capital_pesos: float,
    fee_pp: float,
    df_opt: pd.DataFrame,
    fi_rows: List[Dict],
    eq_sel: Dict[str, Dict],
    fx_sel: Dict[str, Dict],
    end_dt: date | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    end_dt = end_dt or _today()
    bdays = _bizdays_index(start_dt, end_dt)
    if bdays.size == 0:
        return pd.DataFrame(), pd.DataFrame()

    usd_mxn_df = get_hist_5y("MXN=X")
    usd_mxn_series = _safe_adj_close_series_init(usd_mxn_df).reindex(bdays, method="pad").dropna()

    name_to_ticker: Dict[str, str] = {}
    for tkr, meta in (eq_sel or {}).items():
        nm = str(meta.get("name", "") or "").strip()
        if nm:
            name_to_ticker[nm] = str(tkr).strip()

    fx_items: List[Tuple[str, bool, str]] = []
    for k, meta in (fx_sel or {}).items():
        tkr = str(meta.get("yahoo_ticker", "")).strip()
        inv = bool(meta.get("inverse", False))
        disp = str(meta.get("name") or meta.get("pair") or k)
        if tkr:
            fx_items.append((tkr, inv, disp))

    positions: List[Dict] = []
    series_parts: List[pd.Series] = []

    if "Peso (%)" not in df_opt.columns:
        return pd.DataFrame(), pd.DataFrame()

    # -------- RV / FIBRA --------
    mask_eq = df_opt["clase"].isin(["RV_MX", "RV_EXT", "FIBRA_MX"]) if "clase" in df_opt.columns else pd.Series(False, index=df_opt.index)
    for _, row in df_opt.loc[mask_eq].iterrows():
        nm = str(row.get("nombre", ""))
        pct = float(pd.to_numeric(row.get("Peso (%)", 0.0), errors="coerce"))
        tkr = name_to_ticker.get(nm, "").strip()
        clase = str(row.get("clase", ""))

        if not tkr or pct <= 0:
            continue

        df = get_hist_5y(tkr)
        s = _safe_adj_close_series_init(df).reindex(bdays, method="pad").dropna()
        if s.empty:
            continue

        if clase == "RV_EXT":
            if usd_mxn_series.empty or s.size == 0:
                continue
            s_mxn = s.multiply(usd_mxn_series, fill_value=np.nan).dropna()
            if s_mxn.empty:
                continue
            s = s_mxn

        px0 = float(s.iloc[0])
        alloc_cash = seed_capital_pesos * (pct / 100.0)
        qty = 0 if px0 <= 0 else int(math.floor(alloc_cash / px0))
        if qty <= 0:
            continue

        cost = qty * px0
        series_parts.append(s * qty)
        positions.append(
            {
                "activo": nm,
                "clase": clase,
                "ticker": tkr,
                "peso_%": pct,
                "qty": float(qty),
                "px_inicial": px0,
                "costo_ini": cost,
                "px_actual": float(s.iloc[-1]),
                "valor_actual": float(s.iloc[-1]) * qty,
            }
        )

    # -------- FX --------
    for tkr, inv, disp in fx_items:
        pct = _find_weight(df_opt, "FX", disp) or _find_weight(df_opt, "FX", tkr) or 0.0
        if pct <= 0:
            continue

        s = _safe_adj_close_series_init(get_hist_5y(tkr)).reindex(bdays, method="pad").dropna()
        if s.empty:
            continue

        if inv:
            s = 1.0 / s

        tkr_upper = tkr.upper().replace("=X", "")
        if "MXN" not in tkr_upper and tkr_upper != "MXN":
            if usd_mxn_series.empty or s.size == 0:
                continue
            s_mxn = s.multiply(usd_mxn_series, fill_value=np.nan).dropna()
            if s_mxn.empty:
                continue
            s = s_mxn
            disp_mod = disp + " (cruzada a MXN)"
        else:
            disp_mod = disp

        px0 = float(s.iloc[0])
        alloc_cash = seed_capital_pesos * (pct / 100.0)
        qty = 0 if px0 <= 0 else int(math.floor(alloc_cash / px0))
        if qty <= 0:
            continue

        cost = qty * px0
        series_parts.append(s * qty)
        positions.append(
            {
                "activo": f"FX {disp_mod}" + (" (inv)" if inv else ""),
                "clase": "FX",
                "ticker": tkr,
                "peso_%": pct,
                "qty": float(qty),
                "px_inicial": px0,
                "costo_ini": cost,
                "px_actual": float(s.iloc[-1]),
                "valor_actual": float(s.iloc[-1]) * qty,
            }
        )

    # -------- Renta fija --------
    fi_df = pd.DataFrame(fi_rows or [])
    if not fi_df.empty:
        mask_rf = df_opt["clase"].isin(["FI_GOB_MX", "FI_CORP_MX", "FI_EXT"]) if "clase" in df_opt.columns else pd.Series(False, index=df_opt.index)
        for _, r_opt in df_opt[mask_rf].iterrows():
            nombre = str(r_opt.get("nombre", ""))
            pct = float(pd.to_numeric(r_opt.get("Peso (%)", 0.0), errors="coerce"))
            clase_dfopt = str(r_opt.get("clase", ""))
            if pct <= 0:
                continue

            cand = None
            cn = _canon(nombre)
            for _, rb in fi_df.iterrows():
                bn = _canon(str(rb.get("Bono", "")))
                if bn == cn or (bn and (bn in cn or cn in bn)):
                    cand = rb
                    break
            if cand is None:
                continue

            try:
                nominal = float(cand.get("Nominal", 100))
            except Exception:
                nominal = 100.0

            dirty = cand.get("Precio Sucio", cand.get("Precio Sucio ", np.nan))
            try:
                dirty = float(dirty)
            except Exception:
                dirty = np.nan
            if not np.isfinite(dirty) or dirty <= 0:
                try:
                    clean = float(cand.get("Precio Limpio", np.nan))
                except Exception:
                    clean = np.nan
                dirty = clean if (np.isfinite(clean) and clean > 0) else 100.0

            unit_cost = _unit_cost_from_dirty_nominal(dirty, nominal)

            try:
                mu_pct = float(cand.get("Rend (%)", np.nan))
            except Exception:
                mu_pct = np.nan
            mu = (mu_pct / 100.0) if np.isfinite(mu_pct) else 0.0
            r_daily = (1.0 + mu) ** (1.0 / 252.0) - 1.0

            s_val = pd.Series(index=bdays, dtype=float)
            s_val.iloc[0] = unit_cost
            if s_val.size > 1:
                grow = (1.0 + r_daily) ** np.arange(s_val.size)
                s_val = s_val.iloc[0] * pd.Series(grow, index=bdays)

            alloc_cash = seed_capital_pesos * (pct / 100.0)
            qty = 0 if unit_cost <= 0 else int(math.floor(alloc_cash / unit_cost))
            if qty <= 0:
                continue

            series_parts.append(s_val * qty)
            positions.append(
                {
                    "activo": str(cand.get("Bono", nombre)),
                    "clase": clase_dfopt,
                    "ticker": "",
                    "peso_%": pct,
                    "qty": float(qty),
                    "px_inicial": float(s_val.iloc[0]),
                    "costo_ini": float(qty * s_val.iloc[0]),
                    "px_actual": float(s_val.iloc[-1]),
                    "valor_actual": float(s_val.iloc[-1]) * float(qty),
                }
            )

    # -------- Notas estructuradas --------
    sn_rows_all = st.session_state.get("structured_notes", []) or []
    sn_rows = [sn for sn in sn_rows_all if sn.get("yahoo_ticker") or sn.get("ticker") or sn.get("underlying_ticker")]
    if sn_rows:
        sn_counter = 1
        for sn in sn_rows:
            nombre = str(sn.get("nombre") or sn.get("name") or "").strip()
            tkr_underlying = str(sn.get("yahoo_ticker") or sn.get("ticker") or sn.get("underlying_ticker") or "").strip()
            if not nombre or not tkr_underlying:
                continue
            pct = _find_weight(df_opt, None, nombre)
            if not pct or pct <= 0:
                continue

            synthetic_note_ticker = f"SYNTH_SN_{sn_counter:03d}"
            sn_counter += 1

            s_u = _safe_adj_close_series_init(get_hist_5y(tkr_underlying)).reindex(bdays, method="pad").dropna()
            if s_u.empty:
                continue

            try:
                S0 = float(sn.get("spot"))
                K1 = float(sn.get("K1"))
                K2 = float(sn.get("K2"))
                days_total = int(sn.get("plazo_d", 360))
            except Exception:
                continue

            r_pct = float(sn.get("r_pct", 7.70))
            sigma_pct = float(sn.get("vol_pct", 25.0))

            try:
                s_note = _note_series_bs(
                    S_hist=s_u,
                    S0=S0,
                    K1=K1,
                    K2=K2,
                    r_pct=r_pct,
                    sigma_pct=sigma_pct,
                    days_total=days_total,
                    bdays=bdays,
                    price_unit=10.0,
                )
            except Exception:
                continue

            px0 = float(s_note.iloc[0])
            alloc_cash = seed_capital_pesos * (pct / 100.0)
            qty = 0 if px0 <= 0 else int(math.floor(alloc_cash / px0))
            if qty <= 0:
                continue

            series_parts.append(s_note * qty)
            positions.append(
                {
                    "activo": nombre,
                    "clase": "SN",
                    "ticker": synthetic_note_ticker,
                    "peso_%": float(pct),
                    "qty": float(qty),
                    "px_inicial": float(s_note.iloc[0]),
                    "costo_ini": float(qty * s_note.iloc[0]),
                    "px_actual": float(s_note.iloc[-1]),
                    "valor_actual": float(s_note.iloc[-1]) * float(qty),
                }
            )

    if not series_parts:
        return pd.DataFrame(positions), pd.DataFrame()

    port = None
    for s in series_parts:
        port = s if port is None else port.add(s, fill_value=0.0)

    fee = float(fee_pp or 0.0) / 100.0
    if fee > 0:
        days = np.arange(port.size)
        fee_factor = (1.0 - fee) ** (days / 252.0)
        port = port * pd.Series(fee_factor, index=port.index)

    ts_df = pd.DataFrame({"valor_portafolio": port})

    pos_df = pd.DataFrame(positions)
    if not pos_df.empty:
        pos_df["pnl"] = pos_df["valor_actual"] - pos_df["costo_ini"]
        pos_df["ret_%"] = np.where(pos_df["costo_ini"] > 0, pos_df["pnl"] / pos_df["costo_ini"] * 100.0, np.nan)
        total_val = float(pos_df["valor_actual"].sum())
        pos_df["peso_actual_%"] = np.where(total_val > 0.0, pos_df["valor_actual"] / total_val * 100.0, np.nan)
        cols_order = [
            "activo",
            "clase",
            "ticker",
            "peso_%",
            "peso_actual_%",
            "qty",
            "px_inicial",
            "costo_ini",
            "px_actual",
            "valor_actual",
            "pnl",
            "ret_%",
        ]
        pos_df = pos_df[[c for c in cols_order if c in pos_df.columns]]
    return pos_df, ts_df

def fix_structured_note_price_unit(ss, from_unit: float = 100.0, to_unit: float = 10.0):
    """
    Migra transacciones de notas estructuradas de base 100 a base 10.
    Ajusta precio y cantidad proporcionalmente.
    """
    ensure_ledger_initialized(ss)
    led = ss["ops_ledger"].copy()

    if led.empty:
        return

    # Identificar notas estructuradas (símbolos sintéticos o clase)
    mask_sn = (
        led["symbol"].str.startswith("SYNTH_SN_", na=False) |
        led["name"].str.contains("nota", case=False, na=False)
    )

    if not mask_sn.any():
        return

    scale_factor = from_unit / to_unit  # 100/10 = 10

    for idx in led[mask_sn].index:
        old_px = float(led.loc[idx, "px"])
        old_qty = float(led.loc[idx, "qty"])

        new_px = old_px / scale_factor
        new_qty = old_qty * scale_factor

        led.loc[idx, "px"] = new_px
        led.loc[idx, "qty"] = new_qty

        if led.loc[idx, "kind"] in ("BUY", "SELL"):
            led.loc[idx, "cash"] = -new_qty * new_px if led.loc[idx, "kind"] == "BUY" else new_qty * new_px

    ss["ops_ledger"] = led
    st.success(f"✅ Migradas {mask_sn.sum()} transacciones de notas: base {from_unit} → {to_unit}")

# ================================================================
# COMISION DIARIA AUTOMATICA + AUTO-VENTA DEL MEJOR RENDIMIENTO
# ================================================================
def _annual_fee_frac(ss) -> float:
    """
    Devuelve la comisión anual como fracción (p.ej. 0.0085 = 0.85%).
    Lee ss['ops_mgmt_fee_pp'] que fue seteado en init_state().
    """
    try:
        x_pp = float(ss.get("ops_mgmt_fee_pp") or 0.0)  # porcentaje anual
        return max(0.0, x_pp) / 100.0
    except Exception:
        return 0.0

def _daily_fee_from_annual(annual_frac: float) -> float:
    """
    Convierte comisión anual a fracción diaria consistente con 252 días hábiles:
    fee_daily = 1 - (1 - fee_annual)^(1/252)
    """
    annual = max(0.0, float(annual_frac))
    return 1.0 - (1.0 - annual) ** (1.0 / 252.0)

def _price_mxn_on_date(ss, ticker: str, dt: date) -> float:
    """
    Precio en MXN del ticker en la fecha dt:
      - Si existe synthetic_prices[ticker]['price_series'] (ya en MXN), usa eso.
      - Si no, toma Adj Close de Yahoo y:
          * si es FX: respeta 'inverse' y, si no es MXN la cotizada, cruza con USD/MXN
          * si parece RV_EXT (si en synthetic_prices figura clase RV_EXT), multiplica por USD/MXN
          * en otro caso, asume que ya está en MXN
    """
    import numpy as np
    t = pd.Timestamp(dt)

    # 1) synthetic_prices directo
    sp = ss.get("synthetic_prices", {}) or {}
    meta = sp.get(ticker, {}) or {}
    ser = meta.get("price_series")
    if ser is not None:
        try:
            # Convertir a Series
            if isinstance(ser, dict):
                s = pd.Series(ser)
            else:
                s = pd.Series(ser)
            
            s = s.sort_index()
            
            if not s.empty:
                # Precio en la fecha o anterior
                s_until = s[s.index <= t]
                if not s_until.empty:
                    last_price = s_until.iloc[-1]
                    if np.isfinite(last_price) and last_price > 0:
                        return float(last_price)
                
                # Si no hay precio <= t, usar el primero
                first_price = s.iloc[0]
                if np.isfinite(first_price) and first_price > 0:
                    return float(first_price)
        except Exception:
            pass
            
            # Si no hay precio <= t, usar el primer precio disponible
            first_price = s.iloc[0]
            if np.isfinite(first_price) and first_price > 0:
                return float(first_price)
    # Helper: último Adj Close <= t
    def _adj_close_last_on_or_before(sym: str) -> float:
        try:
            df = get_hist_5y(sym)
            s = _safe_adj_close_series(df)
            s = s[s.index <= t]
            if s.empty:
                return float("nan")
            return float(s.iloc[-1])
        except Exception:
            return float("nan")

    px = _adj_close_last_on_or_before(ticker)
    if not np.isfinite(px) or px <= 0:
        return float("nan")

    # USD/MXN
    usd_mxn = _adj_close_last_on_or_before("MXN=X")
    if not np.isfinite(usd_mxn) or usd_mxn <= 0:
        usd_mxn = 1.0

    # 2) Clasificación si la tenemos
    clase = str(meta.get("clase", "")).strip().upper()

    # FX
    if ticker.upper().endswith("=X") or clase == "FX":
        fx_sel = ss.get("fx_selection", {}) or {}
        fx_meta = None
        for m in fx_sel.values():
            if str(m.get("yahoo_ticker", "")).strip().upper() == ticker.upper():
                fx_meta = m
                break
        if fx_meta and fx_meta.get("inverse", False) and px > 0:
            px = 1.0 / px
        tkr_upper = ticker.upper().replace("=X", "")
        if "MXN" not in tkr_upper and tkr_upper != "MXN":
            px = px * usd_mxn
        return float(px)

    # RV_EXT: convertir a MXN si sabemos su clase
    if clase == "RV_EXT":
        px = px * usd_mxn

    return float(px)

def _avg_cost_until(ss, symbol: str, up_to_date: date) -> float:
    """
    Costeo promedio (promedio ponderado) hasta la fecha.
    """
    df = get_ledger(ss)
    if df.empty:
        return float("nan")
    df["ts"] = pd.to_datetime(df["ts"]).dt.date
    df = df.loc[df["ts"] <= up_to_date].copy()
    df = df[df["symbol"].str.strip() == str(symbol).strip()]
    if df.empty:
        return float("nan")
    df = df.sort_values("ts")
    qty = 0.0
    cost = 0.0
    for _, r in df.iterrows():
        k = str(r["kind"]).upper()
        q = float(r["qty"])
        p = float(r["px"])
        if k == "BUY":
            cost += q * p
            qty += q
        elif k == "SELL":
            if qty > 0:
                avg = cost / qty
                qty -= q
                cost -= avg * q
            else:
                qty -= q
    return (cost / qty) if qty > 1e-12 else float("nan")

def _pick_winner_to_sell(ss, on_date: date, shortfall: float) -> Tuple[Optional[str], float, float]:
    """
    Elige el símbolo con mejor rendimiento desde su compra hasta on_date.
    Si ningún activo tiene rendimiento positivo, elige el de "mejor" rendimiento
    (el menos negativo o cero).
    Devuelve (symbol, qty_to_sell, price_mxn) o (None, 0, nan) si no hay.
    """
    import numpy as np
    start_dt = ss.get("ops_start_date") or on_date
    
    # Usar rebuild_base para asegurar que reconstruimos desde el inicio real
    ops_start_base = start_dt
    led_temp = get_ledger(ss)
    if not led_temp.empty:
        ops_start_base = min(start_dt, pd.to_datetime(led_temp["ts"]).min().date())
        
    pos_df, _ = rebuild_portfolio_from_ledger(ss, ops_start_base, on_date)
    if pos_df is None or pos_df.empty:
        return None, 0.0, float("nan")

    best_sym = None
    best_ret = -float('inf') # Usar infinito negativo para asegurar que cualquier número finito sea mayor
    best_px = float("nan")
    best_avail = 0.0

    for _, r in pos_df.iterrows():
        sym = str(r.get("ticker", "") or "").strip()
        if not sym:
            continue
        
        avail = float(r.get("qty", 0.0) or 0.0)
        if avail <= 1e-9: # Usar una tolerancia pequeña
            continue
            
        px = _price_mxn_on_date(ss, sym, on_date)
        if not np.isfinite(px) or px <= 0:
            continue
            
        avg = _avg_cost_until(ss, sym, on_date)
        
        # Lógica de rendimiento mejorada
        ret = -float('inf') # Default a un rendimiento muy bajo
        if np.isfinite(avg) and avg > 0:
            ret = (px / avg) - 1.0
        elif np.isfinite(px):
            # Si no podemos calcular el costo promedio pero tenemos un precio,
            # asumimos un rendimiento de 0.0 para que sea elegible para la venta
            # si no hay otras opciones mejores.
            ret = 0.0

        if ret > best_ret:
            best_ret = ret
            best_sym = sym
            best_px = px
            best_avail = avail

    if not best_sym or not np.isfinite(best_px) or best_px <= 0:
        return None, 0.0, float("nan")

    # Determinar cantidad a vender
    needed_cash = max(0.0, shortfall)
    qty_to_sell = needed_cash / best_px

    # Para acciones, redondear hacia arriba para asegurar que cubrimos el monto.
    # Para FX o activos fraccionarios, no es necesario.
    is_fx = best_sym.upper().endswith("=X")
    clase = str(pos_df[pos_df["ticker"] == best_sym]["clase"].iloc[0])
    if not is_fx and "FX" not in clase:
        qty_to_sell = math.ceil(qty_to_sell)

    # No vender más de lo que tenemos
    qty_to_sell = min(float(best_avail), float(qty_to_sell))
    
    if qty_to_sell <= 1e-9:
        return None, 0.0, float("nan")

    return best_sym, float(qty_to_sell), float(best_px)

# ============================================================
# REEMPLAZO COMPLETO: apply_daily_fees_and_autosell con DEBUG
# ============================================================
# Reemplaza toda la función en ledger.py (línea ~1201)
# ============================================================

def apply_daily_fees_and_autosell(
    ss,
    start_date: date,
    end_date: date,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    1) Elimina comisiones y auto-ventas previas en [start_date, end_date].
    2) Para cada día hábil:
        - Calcula V_t (valor del portafolio) y 'cash' hasta ese día.
        - Fee diario = V_t * fee_diaria
        - Si no hay efectivo suficiente, vende el activo con mejor rendimiento
          (solo lo necesario) al precio de ese día.
        - Inserta WITHDRAW (nota 'COMISION DIARIA') por el fee del día.
    """
    ensure_ledger_initialized(ss)
    
    if debug:
        st.write("=" * 80)
        st.write("🐛 DEBUG: apply_daily_fees_and_autosell")
        st.write("=" * 80)
    
    if not start_date or not end_date:
        if debug:
            st.error("❌ start_date o end_date faltante")
        return {"fees_added": 0, "sells_added": 0, "total_fee": 0.0}

    a = pd.to_datetime(start_date).date()
    b = pd.to_datetime(end_date).date()
    if b < a:
        a, b = b, a

    if debug:
        st.info(f"📅 Rango de recálculo: **{a}** hasta **{b}**")

    # Eliminar comisiones y auto-ventas anteriores en el rango
    led = get_ledger(ss)
    if not led.empty:
        led["ts"] = pd.to_datetime(led["ts"]).dt.date
        mask_rng = (led["ts"] >= a) & (led["ts"] <= b)
        mask_fee = (led["kind"].str.upper() == "WITHDRAW") & led["note"].str.contains(r"comisi[oó]n", case=False, na=False, regex=True)
        mask_autosell = (led["kind"].str.upper() == "SELL") & led["note"].str.contains(r"auto[-\s]?venta.*comisi[oó]n", case=False, na=False, regex=True)
        
        elim_fee = (mask_rng & mask_fee).sum()
        elim_sell = (mask_rng & mask_autosell).sum()
        
        if debug:
            st.write(f"🗑️ Eliminando: {elim_fee} comisiones, {elim_sell} auto-ventas")
        
        led = led.loc[~(mask_rng & (mask_fee | mask_autosell))].copy()
        ss["ops_ledger"] = _sort_ledger(led)

    bdays = pd.bdate_range(a, b, freq="B")
    if len(bdays) == 0:
        if debug:
            st.warning("⚠️ No hay días hábiles en el rango")
        return {"fees_added": 0, "sells_added": 0, "total_fee": 0.0}
    
    if debug:
        st.info(f"📊 Días hábiles a procesar: **{len(bdays)}**")
        fechas_str = ", ".join([d.strftime("%m/%d") for d in bdays[:10]])
        if len(bdays) > 10:
            fechas_str += f"... (+{len(bdays)-10} más)"
        st.write(f"   {fechas_str}")

    fee_ann = _annual_fee_frac(ss)
    fee_day = _daily_fee_from_annual(fee_ann)
    
    if debug:
        st.write(f"💰 Fee anual: **{fee_ann*100:.4f}%**")
        st.write(f"💰 Fee diario: **{fee_day*100:.6f}%**")
    
    if fee_day <= 0:
        if debug:
            st.error("❌ Fee diario es 0 o negativo")
        return {"fees_added": 0, "sells_added": 0, "total_fee": 0.0}

    fees_added = 0
    sells_added = 0
    total_fee = 0.0

    prev_flag = ss.get("_suppress_fee_autorecalc", False)
    ss["_suppress_fee_autorecalc"] = True

    # 🔧 Obtener ops_start correcto
    ops_start = ss.get("ops_start_date")
    if not ops_start:
        led_temp = get_ledger(ss)
        if not led_temp.empty:
            ops_start = pd.to_datetime(led_temp["ts"]).min().date()
        else:
            ops_start = a

    if isinstance(ops_start, pd.Timestamp):
        ops_start = ops_start.date()
    rebuild_base = min(ops_start, a)
        
    if debug:
        st.info(f"🎯 ops_start_date (inicio real): **{ops_start}**")
        st.write("")

    try:
        for i, d in enumerate(bdays):
            dt = d.date()
            
            if debug:
                st.write(f"### 📆 Día {i+1}/{len(bdays)}: {dt.strftime('%Y-%m-%d (%A)')}")
            
            # Usar ops_start en lugar de 'a'
            try:
                _, ts_df = rebuild_portfolio_from_ledger(ss, rebuild_base, dt)
            except Exception as e:
                if debug:
                    st.error(f"   ❌ Error en rebuild_portfolio: {e}")
                continue
            
            if ts_df is None or ts_df.empty:
                if debug:
                    st.warning(f"   ⚠️ ts_df vacío o None")
                continue

            last_row = ts_df.iloc[-1]
            V_t = float(last_row.get("total_value", 0.0) or 0.0)
            cash_t = float(last_row.get("cash", 0.0) or 0.0)
            
            if debug:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Valor portafolio (V_t)", f"${V_t:,.2f}")
                with col2:
                    st.metric("Efectivo", f"${cash_t:,.2f}")
            
            if V_t <= 0:
                if debug:
                    st.warning(f"   ⚠️ V_t <= 0, saltando día")
                continue

            fee_amt = V_t * fee_day
            if fee_amt <= 1e-9:
                if debug:
                    st.warning(f"   ⚠️ fee_amt muy pequeño")
                continue
            
            if debug:
                st.write(f"   💸 **Comisión calculada:** ${fee_amt:,.2f}")

            # ¿Alcanza el efectivo?
            shortfall = max(0.0, fee_amt - max(0.0, cash_t))
            
            if debug:
                if shortfall > 1e-6:
                    st.error(f"   🔴 **Falta efectivo:** ${shortfall:,.2f}")
                else:
                    st.success(f"   ✅ Efectivo suficiente")

            safety_counter = 0
            while shortfall > 1e-6 and safety_counter < 10:
                if debug:
                    st.write(f"   🔄 Auto-venta intento #{safety_counter+1}")
                
                sym, qty_sell, px_sell = _pick_winner_to_sell(ss, dt, shortfall)
                
                if not sym or qty_sell <= 0 or not np.isfinite(px_sell) or px_sell <= 0:
                    if debug:
                        st.error(f"   ❌ No se pudo encontrar activo para vender")
                        st.write(f"      sym={sym}, qty={qty_sell}, px={px_sell}")
                    break
                
                if debug:
                    st.success(f"   💰 Vendiendo **{qty_sell:,.2f}** de **{sym}** @ ${px_sell:,.2f}")
                
                add_transaction(
                    ss,
                    ts=dt,
                    kind="SELL",
                    symbol=sym,
                    qty=qty_sell,
                    px=px_sell,
                    note="Auto-venta comisión",
                    name="",
                )
                sells_added += 1
                
                # Recalcular cash tras la venta
                _, ts_df2 = rebuild_portfolio_from_ledger(ss, rebuild_base, dt)
                if ts_df2 is None or ts_df2.empty:
                    break
                cash_t = float(ts_df2.iloc[-1].get("cash", 0.0) or 0.0)
                shortfall = max(0.0, fee_amt - max(0.0, cash_t))
                safety_counter += 1

            # Registrar WITHDRAW de comisión del día
            add_transaction(
                ss,
                ts=dt,
                kind="WITHDRAW",
                cash=-fee_amt,
                note="COMISION DIARIA",
                name="",
            )
            fees_added += 1
            total_fee += fee_amt
            
            if debug:
                st.success(f"   ✅ **WITHDRAW registrado:** ${fee_amt:,.2f}")
                st.write("")

    except Exception as e:
        if debug:
            st.error(f"💥 ERROR FATAL: {e}")
            import traceback
            st.code(traceback.format_exc())
        else:
            # Si no está en debug, al menos registrar el error
            import traceback
            print(f"Error in apply_daily_fees_and_autosell: {e}")
            print(traceback.format_exc())
    finally:
        ss["_suppress_fee_autorecalc"] = prev_flag

    if debug:
        st.write("=" * 80)
        st.success(f"✅ **COMPLETADO**")
        st.write(f"   📊 Comisiones agregadas: **{fees_added}**")
        st.write(f"   💰 Auto-ventas: **{sells_added}**")
        st.write(f"   💵 Total comisiones: **${total_fee:,.2f}**")
        st.write("=" * 80)
    
    return {"fees_added": fees_added, "sells_added": sells_added, "total_fee": float(total_fee)}
