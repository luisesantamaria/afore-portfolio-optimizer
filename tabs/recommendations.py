# tabs/recommendations.py
from __future__ import annotations

from datetime import datetime, date
from typing import Optional, Dict, List, Any

# ✅ Imports de terceros arriba (orden/estilo)
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_searchbox import st_searchbox


__all__ = [
    "mostrar_recomendaciones_rv",
    "mostrar_structured_note_compacto",
    "mostrar_fixed_income_compacto",
]

# =========================
# Helpers locales livianos
# =========================
def _today() -> date:
    return datetime.now().date()

def _recos_cache_key(categoria_code: str, eff_date: date) -> str:
    try:
        d = eff_date.isoformat()
    except Exception:
        d = str(eff_date)
    return f"{str(categoria_code).upper()}__{d}"


# =====================================================================================
# 1) RENTA VARIABLE / DIVISAS: UI de recomendaciones + buscador + pesos (Sharpe máx)
# =====================================================================================
def mostrar_recomendaciones_rv(
    categoria_code: str,
    titulo: str,
    selected_categoria: Optional[str] = None,
    pesos_por_clase: Optional[Dict[str, float]] = None,
):
    """
    Renderiza la UI de selección de activos para RV / FX:
      - Obtiene recomendaciones (caché por categoría y fecha efectiva)
      - Buscador + multiselect y sincronización con session_state
      - Optimización de pesos (Sharpe máx) + editor
      - Tabla con checkboxes para agregar/quitar picks
    """
    # 🔹 Imports internos (evita dependencias circulares)
    from tabs.aportaciones import recomendar_rv
    from tabs import runtime
    from tabs.equities import load_universe, _search_suggestions_factory
    from tabs.fx import (
        _catalog_from_csv as fx_catalog,
        _mu_sigma_fx_5y as fx_mu_sigma,
        _ret_1y as fx_ret_1y,
        _series_from_yahoo as fx_series,
        pretty_fx_label as fx_pretty,
        extract_pair_codes as fx_extract,
        _max_drawdown_1y as fx_mdd1y,
        _beta_vs_usdmxn as fx_beta,
    )

    ss = st.session_state

    # --- Fallbacks (compatibilidad): usa session_state solo si no se pasan explícitos ---
    if selected_categoria is None:
        selected_categoria = ss.get("__selected_categoria_actual__")
    if pesos_por_clase is None:
        pesos_por_clase = ss.get("__pesos_por_clase_actual__", {})

    # Peso de la categoría actual (si disponible)
    if selected_categoria and pesos_por_clase:
        peso_seleccionado = float(pesos_por_clase.get(selected_categoria, 0.0))
    else:
        peso_seleccionado = 0.0

    is_fx = str(categoria_code or "").upper() in {"FX", "DIVISAS"}

    # Fechas y caché
    eff_date = ss.get("contrib_effective_date") or _today()
    es_backdating = (eff_date < _today())
    cache_key = _recos_cache_key(categoria_code, eff_date)
    cache = ss.setdefault("recs_cache", {})
    recomendaciones = cache.get(cache_key)
    from_cache = recomendaciones is not None

    # =================== RECOMENDACIONES (RV vs FX) ===================
    if not from_cache:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(value, text):
            try:
                progress_bar.progress(min(max(float(value), 0.0), 1.0))
            except Exception:
                pass
            status_text.text(str(text))

        try:
            if not is_fx:
                # === Camino RV existente ===
                recomendaciones = recomendar_rv(
                    categoria=categoria_code,
                    fecha_efectiva=eff_date,
                    peso_asignado=peso_seleccionado,
                    top_n=20,
                    progress_callback=update_progress
                )
            else:
                # === Camino FX: score compuesto multi-señal (mismas series que fx.py) ===
                try:
                    options = fx_catalog() or []
                except Exception:
                    options = []

                rows: List[Dict[str, Any]] = []
                n = max(1, len(options))
                for i, opt in enumerate(options, 1):
                    update_progress(i / n, f"Calculando FX ({i}/{n})…")

                    tkr = str(opt.get("ticker", "")).strip()
                    inv = bool(opt.get("inverse", False))
                    if not tkr:
                        continue

                    # Serie y retornos (vía fx.py → yf_store)
                    r5, df5 = fx_series(tkr, period="5y", inverse=inv)
                    r = r5.dropna() if r5 is not None else pd.Series(dtype=float)

                    s = pd.Series(dtype=float)
                    if df5 is not None and "Adj Close" in df5.columns:
                        try:
                            s = df5["Adj Close"].astype(float).dropna()
                        except Exception:
                            s = pd.Series(dtype=float)

                    # Métricas base (mismas que en FX tab)
                    mu, sig = fx_mu_sigma(tkr, inverse=inv)            # anualizadas
                    ret12   = fx_ret_1y(tkr, inverse=inv)              # 12M
                    mdd1y   = fx_mdd1y(tkr, inverse=inv)               # drawdown 1Y (negativo)
                    beta_mxn = fx_beta(tkr, period="5y", inverse=inv)  # beta vs USD/MXN

                    # Sortino (solo downside)
                    sortino = np.nan
                    try:
                        dside = r[r < 0].std(ddof=1) * np.sqrt(252) if not r.empty else np.nan
                        if np.isfinite(mu) and np.isfinite(dside) and dside > 0:
                            sortino = float(mu) / float(dside)
                    except Exception:
                        pass

                    # Momentum multi-horizonte
                    def _ret_n_days(px, n):
                        try:
                            if px is None or px.size < (n + 1) or float(px.iloc[-n]) == 0.0:
                                return np.nan
                            return float(px.iloc[-1] / px.iloc[-n] - 1.0)
                        except Exception:
                            return np.nan

                    mom_1m  = _ret_n_days(s, 21)
                    mom_3m  = _ret_n_days(s, 63)
                    mom_6m  = _ret_n_days(s, 126)
                    mom_12m = _ret_n_days(s, 252)

                    # Tendencia (SMA200) + hit-rate 3M
                    trend_above200 = np.nan
                    hit_3m = np.nan
                    try:
                        if not s.empty:
                            ma200 = s.rolling(200).mean()
                            if ma200.dropna().size:
                                trend_above200 = float(s.iloc[-1] > ma200.iloc[-1])  # 1.0/0.0
                        if not r.empty:
                            hit_3m = float((r.tail(63) > 0).mean())  # % días +
                    except Exception:
                        pass

                    # Régimen de volatilidad (spike 30d vs σ 5y)
                    vol_ratio = np.nan
                    try:
                        vol30 = r.tail(30).std(ddof=1) * np.sqrt(252) if not r.empty else np.nan
                        if np.isfinite(vol30) and np.isfinite(sig) and sig > 0:
                            vol_ratio = float(vol30 / sig)
                    except Exception:
                        pass

                    # Rendimiento desde eff_date (para backdating)
                    rend = np.nan
                    try:
                        if not s.empty:
                            start = min(pd.Timestamp(eff_date), s.index.max())
                            bdays = pd.bdate_range(start, s.index.max())
                            s2 = s.reindex(bdays, method="pad").dropna()
                            if s2.size >= 2 and float(s2.iloc[0]) != 0.0:
                                rend = float(s2.iloc[-1] / s2.iloc[0] - 1.0)
                    except Exception:
                        pass

                    pair   = fx_extract(opt.get("name", ""))
                    nombre = fx_pretty(pair)

                    rows.append({
                        "ticker": tkr,
                        "nombre": nombre,
                        "rendimiento": rend,          # usado si backdating
                        # features crudos
                        "mu": mu, "sigma": sig, "sortino": sortino,
                        "mom_1m": mom_1m, "mom_3m": mom_3m, "mom_6m": mom_6m, "mom_12m": mom_12m,
                        "trend_above200": trend_above200, "hit_3m": hit_3m,
                        "mdd_1y": mdd1y, "beta_mxn": beta_mxn, "vol_ratio": vol_ratio,
                        # para UI y compat
                        "ret_3m": mom_3m, "ret_12m": ret12, "volatilidad": sig,
                        # placeholders de RV (no se mostrarán en FX)
                        "pe_ratio": np.nan, "price_to_book": np.nan, "roe": np.nan,
                        "eps_growth": np.nan, "debt_to_equity": np.nan, "fcf_yield": np.nan,
                    })

                df_fx = pd.DataFrame(rows)

                # Rank-percentile helper
                def _pct_rank(s: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
                    if s is None or s.empty or s.notna().sum() == 0:
                        return pd.Series([0.5] * len(df_fx), index=df_fx.index)
                    x = s.replace([np.inf, -np.inf], np.nan)
                    rp = x.rank(pct=True, method="average")
                    rp = rp.fillna(0.5)
                    return rp if higher_is_better else (1.0 - rp)

                # Sub-scores
                z_mom = (
                    0.25 * _pct_rank(df_fx["mom_1m"]) +
                    0.35 * _pct_rank(df_fx["mom_3m"]) +
                    0.25 * _pct_rank(df_fx["mom_6m"]) +
                    0.15 * _pct_rank(df_fx["mom_12m"])
                )
                z_sortino   = _pct_rank(df_fx["sortino"])
                z_trend     = 0.6 * _pct_rank(df_fx["trend_above200"]) + 0.4 * _pct_rank(df_fx["hit_3m"])
                z_drawdown  = _pct_rank(df_fx["mdd_1y"].abs(), higher_is_better=False)      # menor DD mejor
                z_beta_div  = _pct_rank(df_fx["beta_mxn"].abs(), higher_is_better=False)    # menor |beta| mejor
                z_volregime = _pct_rank(df_fx["vol_ratio"], higher_is_better=False)         # menor spike mejor

                # Ponderaciones
                W_MOM, W_SORTINO, W_TREND, W_DD, W_BETA, W_VOL = 0.40, 0.25, 0.10, 0.15, 0.05, 0.05
                score_comp = (
                    W_MOM * z_mom +
                    W_SORTINO * z_sortino +
                    W_TREND * z_trend +
                    W_DD * z_drawdown +
                    W_BETA * z_beta_div +
                    W_VOL * z_volregime
                )

                df_fx["score"] = (score_comp * 100.0).astype(float)  # 0..100
                recomendaciones = df_fx.sort_values("score", ascending=False).to_dict("records")

            # 🔽 Backdating: filtra NaN en 'rendimiento' y ordena desc
            if es_backdating and recomendaciones:
                recomendaciones = [r for r in recomendaciones if pd.notna(r.get("rendimiento"))]
                recomendaciones.sort(key=lambda d: d.get("rendimiento", -np.inf), reverse=True)

            cache[cache_key] = recomendaciones
        finally:
            progress_bar.empty()
            status_text.empty()

    if not recomendaciones:
        st.warning("No se encontraron recomendaciones.")
        return

    # =================== UNIVERSO/MAPEOS (RV vs FX) ===================
    try:
        if not is_fx:
            universe = load_universe()
            idx_upper = universe["index"].str.upper().str.strip()
            name_upper = universe["name"].str.upper().str.strip()
            if categoria_code == "RV_EXT":
                mask = ~idx_upper.isin(["MEX","MEXICO","MX","BMV","IPC"])
            elif categoria_code == "RV_MX":
                mask = idx_upper.isin(["MEX","MEXICO","MX","BMV","IPC"]) & ~name_upper.str.startswith("FIBRA")
            elif categoria_code == "FIBRA_MX":
                mask = idx_upper.isin(["MEX","MEXICO","MX","BMV","IPC"]) & name_upper.str.startswith("FIBRA")
            else:
                mask = pd.Series([True]*len(universe))
            universe_filtrado = universe[mask].copy()
            universe_filtrado["display"] = universe_filtrado["name"].str.strip() + " (" + universe_filtrado["yahoo"].str.strip() + ")"
        else:
            # Universo FX desde fx.csv
            opts = fx_catalog() or []
            data = []
            for o in opts:
                tkr  = str(o.get("ticker","")).strip()
                pair = fx_extract(o.get("name",""))
                nm   = fx_pretty(pair)
                data.append({"index": "FX", "name": nm, "yahoo": tkr})
            universe_filtrado = pd.DataFrame(data)
            if not universe_filtrado.empty:
                universe_filtrado["display"] = universe_filtrado["name"].str.strip() + " (" + universe_filtrado["yahoo"].str.strip() + ")"

        display2ticker = dict(zip(universe_filtrado["display"], universe_filtrado["yahoo"])) if not universe_filtrado.empty else {}
        ticker2display = {t: d for d, t in display2ticker.items()}
        all_display_options = universe_filtrado["display"].tolist() if not universe_filtrado.empty else []
    except Exception:
        universe_filtrado = pd.DataFrame()
        display2ticker, ticker2display, all_display_options = {}, {}, []

    df_rec = pd.DataFrame(recomendaciones)

    col_left, col_right = st.columns([0.4, 0.6], gap="large")

    # ============================
    # LEFT: Buscador + Multiselect + Pesos
    # ============================
    with col_left:
        st.markdown("**Activos seleccionados**")

        key_selected   = f"recs_selected_{categoria_code}"      # dict {ticker: {nombre, score}}
        key_picks      = f"recs_picks_{categoria_code}"         # [display,...]
        key_picks_raw  = f"recs_picks_raw_{categoria_code}"     # usado SOLO para rev key
        key_w_sig      = f"recs_weights_sig_{categoria_code}"
        key_weights    = f"recs_weights_{categoria_code}"       # dict {ticker: pct (0-100)}
        multi_rev_var  = f"{key_picks_raw}__rev"                # contador para recrear multiselect

        if key_selected not in ss: ss[key_selected] = {}
        if key_picks not in ss: ss[key_picks] = []
        if key_weights not in ss: ss[key_weights] = {}
        if multi_rev_var not in ss: ss[multi_rev_var] = 0

        if not universe_filtrado.empty:
            selected_ticker = st_searchbox(
                search_function=_search_suggestions_factory(universe_filtrado),
                placeholder=f"Buscar en {titulo}",
                key=f"search_{categoria_code}",
            )
            if selected_ticker and selected_ticker not in ss[key_selected]:
                match = df_rec[df_rec["ticker"].str.upper() == selected_ticker.upper()]
                if not match.empty:
                    rec_row = match.iloc[0]
                    ss[key_selected][selected_ticker] = {"nombre": rec_row["nombre"], "score": rec_row.get("score", 0.0)}
                else:
                    try:
                        u_match = universe_filtrado[universe_filtrado["yahoo"].str.upper() == selected_ticker.upper()]
                        nm = u_match.iloc[0]["name"] if not u_match.empty else selected_ticker
                    except Exception:
                        nm = selected_ticker
                    ss[key_selected][selected_ticker] = {"nombre": nm, "score": 0.0}
                disp = ticker2display.get(selected_ticker) or f"{ss[key_selected][selected_ticker]['nombre']} ({selected_ticker})"
                if disp not in ss[key_picks]:
                    ss[key_picks].append(disp)
                ss[multi_rev_var] += 1

        # Reconstruir displays
        current_displays = []
        for tk in ss[key_selected].keys():
            disp = ticker2display.get(tk) or f"{ss[key_selected][tk]['nombre']} ({tk})"
            current_displays.append(disp)
        if current_displays != ss[key_picks]:
            ss[key_picks] = current_displays

        # Options extendidas
        options_list = list(all_display_options)
        for d in current_displays:
            if d not in options_list:
                options_list.append(d)

        # Mapeo extendido display→ticker
        display2ticker_ext = dict(display2ticker)
        for d in current_displays:
            if d not in display2ticker_ext and "(" in d and d.endswith(")"):
                try:
                    t = d.split("(")[-1].strip(")")
                    if t: display2ticker_ext[d] = t
                except Exception:
                    pass

        # on_change del multiselect
        def handle_multiselect_change():
            current_displays_now = ss.get(multi_key, [])
            current_tickers = []
            for d in current_displays_now:
                t = display2ticker_ext.get(d)
                if not t:
                    try: t = d.split("(")[-1].strip(")")
                    except Exception: t = None
                if t and t not in current_tickers:
                    current_tickers.append(t)

            new_selection = {}
            for t in current_tickers:
                if t in ss[key_selected]:
                    new_selection[t] = ss[key_selected][t]
                else:
                    match = df_rec[df_rec["ticker"].str.upper() == t.upper()]
                    if not match.empty:
                        rec_row = match.iloc[0]
                        new_selection[t] = {"nombre": rec_row["nombre"], "score": rec_row.get("score", 0.0)}
                    else:
                        try:
                            u_match = universe_filtrado[universe_filtrado["yahoo"].str.upper() == t.upper()]
                            nm = u_match.iloc[0]["name"] if not u_match.empty else t
                        except Exception:
                            nm = t
                        new_selection[t] = {"nombre": nm, "score": 0.0}
            ss[key_selected] = new_selection

            rebuilt_displays = []
            for t, meta in ss[key_selected].items():
                disp = ticker2display.get(t) or f"{meta['nombre']} ({t})"
                if disp not in rebuilt_displays:
                    rebuilt_displays.append(disp)
            ss[key_picks] = rebuilt_displays

            ss[multi_rev_var] += 1

        # multiselect con key dinámico
        multi_key = f"{key_picks_raw}__v{ss[multi_rev_var]}"
        st.multiselect(
            "Activos en tu cartera",
            options=options_list,
            default=ss[key_picks],
            key=multi_key,
            on_change=handle_multiselect_change,
            label_visibility="collapsed"
        )
        st.caption(f"**{len(ss[key_selected])}** activos seleccionados")

        # =========================
        # 🧮 PESOS (Sharpe máx) + Editor
        # =========================
        selected_tickers = list(ss[key_selected].keys())
        sig = tuple(sorted(selected_tickers))
        if ss.get(key_w_sig) != sig:
            def _optimize_weights_max_sharpe(tickers, categoria):
                from tabs.yf_store import get_hist_5y
                from tabs.ledger import _safe_adj_close_series_init
                act_of = runtime.get_act_as_of() or _today()
                start_ts = pd.Timestamp(act_of) - pd.DateOffset(years=3)
                bdays = pd.date_range(start_ts, pd.Timestamp(act_of), freq="B")
                try:
                    usd_mxn = _safe_adj_close_series_init(get_hist_5y("MXN=X")).reindex(bdays, method="pad").dropna()
                except Exception:
                    usd_mxn = pd.Series(dtype=float)
                px = {}
                for t in tickers:
                    try:
                        s = _safe_adj_close_series_init(get_hist_5y(t)).reindex(bdays, method="pad")
                        if categoria == "RV_EXT" and not str(t).upper().endswith(".MX") and not usd_mxn.empty:
                            s = (s * usd_mxn).dropna()
                        px[t] = s.dropna()
                    except Exception:
                        continue
                if not px:
                    n = max(1, len(tickers)); return {t: 100.0/n for t in tickers}
                prices = pd.DataFrame(px).dropna(how="any")
                if prices.shape[0] < 60 or prices.shape[1] < 1:
                    n = max(1, len(tickers)); return {t: 100.0/n for t in tickers}
                rets = prices.pct_change().dropna(how="any")
                mu  = rets.mean().values
                cov = rets.cov().values + np.eye(rets.shape[1]) * 1e-6
                rf_daily = 0.0
                try:
                    from datetime import datetime as _dt
                    from tabs import banxico_client as bx
                    CETES_28_SERIE = "SF43936"
                    token = getattr(st.session_state, "banxico_token", bx.BANXICO_TOKEN_DEFAULT)
                    _f, v, _lag = bx.latest_on_or_before(CETES_28_SERIE, _dt.combine(act_of, _dt.min.time()), token)
                    rf_daily = float(v)/100.0/252.0 if v is not None else 0.0
                except Exception:
                    pass
                mu_ex = mu - rf_daily
                n = len(mu_ex)
                active = np.ones(n, dtype=bool)
                if np.all(mu_ex <= 0):
                    w = np.ones(n)/n
                else:
                    for _ in range(n):
                        Sigma = cov[np.ix_(active, active)]
                        mu_a  = mu_ex[active]
                        try:
                            inv = np.linalg.pinv(Sigma); w_a = inv @ mu_a
                        except Exception:
                            w_a = np.ones(mu_a.shape[0])
                        if (w_a >= -1e-12).all():
                            w = np.zeros(n); w[active] = w_a; break
                        neg_local = np.argmin(w_a)
                        idxs = np.where(active)[0]; active[idxs[neg_local]] = False
                    else:
                        w = np.ones(n)/n
                w = np.clip(w, 0, None); w = w / (w.sum() if w.sum()>0 else 1.0)
                weights_pct = {t: float(w[i])*100.0 for i, t in enumerate(rets.columns)}
                for t in tickers: weights_pct.setdefault(t, 0.0)
                return weights_pct

            ss[key_weights] = _optimize_weights_max_sharpe(selected_tickers, categoria_code)
            ss[key_w_sig] = sig

        if selected_tickers:
            rows = [{"Ticker": t, "Nombre": ss[key_selected][t]["nombre"],
                    "Peso (%)": round(float(ss[key_weights].get(t, 0.0)), 2)} for t in selected_tickers]
            df_w = pd.DataFrame(rows)

            st.markdown("**Pesos por activo (Sharpe máx. inicial, ajustables)**")
            st.caption(f"Los pesos deben sumar **100%** de la categoría seleccionada (equivale al {peso_seleccionado:.2f}% de la aportación).")

            ed_key = f"weights_editor_{categoria_code}_{len(selected_tickers)}"
            df_w_edit = st.data_editor(
                df_w,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
                    "Nombre": st.column_config.TextColumn("Nombre", disabled=True),
                    "Peso (%)": st.column_config.NumberColumn("Peso (%)", min_value=0.0, max_value=100.0, step=0.01, format="%.2f%%"),
                },
                hide_index=True,
                use_container_width=True,
                key=ed_key
            )

            orig = pd.Series({r["Ticker"]: r["Peso (%)"] for r in rows})
            newv = pd.Series({r["Ticker"]: r["Peso (%)"] for _, r in df_w_edit.iterrows()})
            if not orig.round(4).equals(newv.round(4)):
                changed = (orig.round(4) != newv.round(4))
                fixed_sum = float(newv[changed].sum())
                final = newv.copy()
                if fixed_sum > 100.0:
                    final[:] = 0.0
                    norm_fixed = (newv[changed] / fixed_sum) * 100.0 if fixed_sum > 0 else 0.0
                    for tk, val in norm_fixed.items(): final[tk] = val
                else:
                    remaining = 100.0 - fixed_sum
                    unchanged = ~changed
                    orig_unch = orig[unchanged]; s_ = float(orig_unch.sum())
                    if s_ > 0:
                        for tk, w0 in orig_unch.items(): final[tk] = (w0 / s_) * remaining
                    elif remaining > 0 and unchanged.sum() > 0:
                        eq = remaining / int(unchanged.sum())
                        for tk in orig_unch.index: final[tk] = eq
                ss[key_weights] = {tk: float(max(0.0, min(100.0, val))) for tk, val in final.items()}

            suma = float(sum(ss[key_weights].values()))
            st.metric("Suma (debe ser 100%)", f"{suma:,.2f}%")

    # ============================
    # RIGHT: Tabla de recomendaciones con checkboxes
    # ============================
    with col_right:
        st.markdown("**Recomendaciones sugeridas**")
        if es_backdating:
            st.caption(f"📈 {titulo} con **mejor rendimiento histórico** desde la fecha de aportación." + (" (caché)" if from_cache else ""))

            if "rendimiento" in df_rec.columns:
                df_rec = df_rec[pd.notna(df_rec["rendimiento"])].sort_values("rendimiento", ascending=False)
            df_rec["Rendimiento"] = df_rec["rendimiento"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")

            if is_fx:
                df_rec_display = df_rec[["ticker","nombre","Rendimiento"]].rename(columns={"ticker":"Ticker","nombre":"Divisa"})
            elif categoria_code == "FIBRA_MX":
                df_rec_display = df_rec[["ticker","nombre","Rendimiento"]].rename(columns={"ticker":"Ticker","nombre":"FIBRA"})
            else:
                df_rec_display = df_rec[["ticker","nombre","Rendimiento"]].rename(columns={"ticker":"Ticker","nombre":"Empresa"})

        else:
            st.caption(
                ("🔮 " + titulo + " con **score multi-señal (momentum/tendencia/riesgo)**." if is_fx
                 else "🔮 " + titulo + " con **potencial de crecimiento** (valuación + momentum).")
                + (" (caché)" if from_cache else "")
            )

            if is_fx:
                raw_cols_for_viz = ["score", "sortino", "mom_1m", "mom_3m", "mom_12m", "volatilidad", "mdd_1y", "beta_mxn"]
                df_fx_viz = df_rec.copy()
                mask_ok = df_fx_viz[raw_cols_for_viz].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
                df_fx_viz = df_fx_viz[mask_ok].copy()

                df_fx_viz["Score"]      = df_fx_viz["score"].apply(lambda x: f"{x:.1f}")
                df_fx_viz["Sortino"]    = df_fx_viz["sortino"].apply(lambda x: f"{x:.2f}")
                df_fx_viz["Mom 1M"]     = df_fx_viz["mom_1m"].apply(lambda x: f"{x*100:.1f}%")
                df_fx_viz["Mom 3M"]     = df_fx_viz["mom_3m"].apply(lambda x: f"{x*100:.1f}%")
                df_fx_viz["Mom 12M"]    = df_fx_viz["mom_12m"].apply(lambda x: f"{x*100:.1f}%")
                df_fx_viz["Vol 5Y"]     = df_fx_viz["volatilidad"].apply(lambda x: f"{x*100:.1f}%")
                df_fx_viz["MDD 1Y"]     = df_fx_viz["mdd_1y"].apply(lambda x: f"{x*100:.1f}%")

                df_rec_display = df_fx_viz[
                    ["ticker","nombre","Score","Sortino","Mom 1M","Mom 3M","Mom 12M","Vol 5Y","MDD 1Y"]
                ].rename(columns={"ticker":"Ticker","nombre":"Divisa"})

            else:
                df_rec["Score"]   = df_rec["score"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
                df_rec["P/E"]     = df_rec.get("pe_ratio", np.nan).apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
                df_rec["P/B"]     = df_rec.get("price_to_book", np.nan).apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
                df_rec["ROE"]     = df_rec.get("roe", np.nan).apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
                df_rec["EPS Gr"]  = df_rec.get("eps_growth", np.nan).apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
                df_rec["D/E"]     = df_rec.get("debt_to_equity", np.nan).apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
                df_rec["FCF Yld"] = df_rec.get("fcf_yield", np.nan).apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
                df_rec["Mom 3M"]  = df_rec.get("ret_3m", np.nan).apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
                df_rec["Mom 12M"] = df_rec.get("ret_12m", np.nan).apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
                df_rec["Vol"]     = df_rec.get("volatilidad", np.nan).apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")

                if categoria_code == "FIBRA_MX":
                    df_rec_display = df_rec[[
                        "ticker","nombre","Score","P/E","P/B","ROE","EPS Gr","D/E","FCF Yld","Mom 3M","Mom 12M","Vol"
                    ]].rename(columns={"ticker":"Ticker","nombre":"FIBRA"})
                else:
                    df_rec_display = df_rec[[
                        "ticker","nombre","Score","P/E","P/B","ROE","EPS Gr","D/E","FCF Yld","Mom 3M","Mom 12M","Vol"
                    ]].rename(columns={"ticker":"Ticker","nombre":"Empresa"})

        # Editor con columna de selección (Ticker oculto SOLO en visual)
        sel_set = set(ss[key_selected].keys())
        df_editor = df_rec_display.copy()
        df_editor.insert(0, "Sel", df_editor["Ticker"].map(lambda t: t in sel_set))

        # === Column order para ocultar Ticker en TODA renta variable ===
        editor_kwargs = {}
        visible_cols = [c for c in df_editor.columns if c != "Ticker"]
        editor_kwargs["column_order"] = visible_cols

        table_rev_key = f"_table_rev_{categoria_code}"
        last_cat_key  = f"_last_cat_{categoria_code}"
        if ss.get(last_cat_key) != (selected_categoria or titulo):
            ss[last_cat_key] = (selected_categoria or titulo)
            ss[table_rev_key] = ss.get(table_rev_key, 0) + 1
        editor_key = f"recs_editor_{categoria_code}_{ss.get(table_rev_key, 0)}"

        edited = st.data_editor(
            df_editor,
            key=editor_key,
            hide_index=True,
            use_container_width=True,
            column_config={"Sel": st.column_config.CheckboxColumn("", help="Seleccionar activo")},
            **editor_kwargs,  # ← Oculta Ticker
        )

        # Sincroniza cambios del editor → estado canónico
        # Si Ticker no viene en 'edited' por estar oculto, lo recuperamos desde df_editor
        try:
            if "Ticker" in edited.columns:
                new_checked = set(edited.loc[edited["Sel"], "Ticker"].astype(str).tolist())
            else:
                idx_sel = edited.index[edited["Sel"]].tolist()
                new_checked = set(df_editor.loc[idx_sel, "Ticker"].astype(str).tolist())
        except Exception:
            new_checked = set()

        current_checked = set(sel_set)
        tickers_in_grid = set(df_rec_display["Ticker"].astype(str).tolist())

        to_add = sorted((new_checked - current_checked) & tickers_in_grid)
        to_remove = sorted((current_checked - new_checked) & tickers_in_grid)

        changed = False
        if to_add:
            for tk in to_add:
                match = df_rec[df_rec["ticker"].str.upper() == tk.upper()]
                if not match.empty:
                    rec_row = match.iloc[0]
                    ss[key_selected][tk] = {"nombre": rec_row["nombre"], "score": rec_row.get("score", 0.0)}
                else:
                    ss[key_selected][tk] = {"nombre": tk, "score": 0.0}
                disp = ticker2display.get(tk) or f"{ss[key_selected][tk]['nombre']} ({tk})"
                if disp not in ss[key_picks]:
                    ss[key_picks].append(disp)
            changed = True

        if to_remove:
            for tk in to_remove:
                ss[key_selected].pop(tk, None)
                try:
                    ss[key_picks] = [d for d in ss[key_picks] if not d.endswith(f"({tk})")]
                except Exception:
                    pass
            changed = True

        if changed:
            ss[multi_rev_var] += 1
            st.rerun()


# =====================================================================================
# 2) NOTA ESTRUCTURADA (compacta)
# =====================================================================================
def mostrar_structured_note_compacto():
    """
    Interfaz compacta para notas estructuradas (call spread financiado con CETES):
    - Selector de subyacente (con volatilidad y spot auto)
    - Primas BS, métricas por Monte Carlo, payoff y tabla de notas con pesos
    """
    from tabs.structured_note import (
        _load_universe_from_csv, _parse_ticker_from_label, _latest_spot_and_vol,
        _mx_risk_free_pct_default, _bs_call, _bs_put, _min_safe
    )
    from tabs.yf_store import preload_hist_5y_daily

    ss = st.session_state

    base_key = "snc"
    k_sel   = f"{base_key}_sel_label"
    k_tkr   = f"{base_key}_ticker"
    k_name  = f"{base_key}_name"
    k_spot  = f"{base_key}_spot"
    k_k1    = f"{base_key}_k1"
    k_k2    = f"{base_key}_k2"
    k_sigma = f"{base_key}_sigma_pct"
    k_r     = f"{base_key}_r_pct"
    k_days  = f"{base_key}_days"
    k_notes = f"{base_key}_notes_list"
    k_weights = f"{base_key}_weights"

    # Inicializar lista/pesos
    ss.setdefault(k_notes, [])
    ss.setdefault(k_weights, {})

    # Universo y selector
    uni = _load_universe_from_csv()
    labels = [f"{row['name']} ({row['ticker']})" for _, row in uni.iterrows()]
    placeholder = "— Selecciona subyacente —"
    options = [placeholder] + labels
    ss.setdefault(k_sel, placeholder)

    sel_label = ss.get(k_sel, placeholder)
    ready = (sel_label != placeholder)

    # Función auxiliar (idéntica en comportamiento a ledger.py)
    def _calcular_metricas_mc(S0, K1, K2, sigma, r, T):
        # CETES (base $10 nominal)
        DF = np.exp(-r * T)
        precio_cetes_hoy = 10.0 * DF
        presupuesto_opciones = 10.0 - precio_cetes_hoy

        spread_cost = max(_bs_call(S0, K1, r, sigma, T) - _bs_call(S0, K2, r, sigma, T), 0.0)
        if spread_cost <= 1e-12:
            return float(r * 100.0), 0.0, 0.0

        n_spreads = presupuesto_opciones / spread_cost

        mu_view = r
        N = 20000
        np.random.seed(42)
        z = np.random.normal(size=N)
        ST = S0 * np.exp((mu_view - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)

        spread_payoff = np.clip(ST - K1, 0, K2 - K1)

        payoff_note_T = 10.0 + n_spreads * spread_payoff

        precio_inicial = precio_cetes_hoy + (n_spreads * spread_cost)
        ret_T = (payoff_note_T / precio_inicial) - 1.0
        ret_ann = np.maximum(payoff_note_T / precio_inicial, 1e-12) ** (1.0 / max(T, 1e-9)) - 1.0

        mu_ann_pct = float(np.mean(ret_ann)) * 100.0
        vol_T = float(np.std(ret_T, ddof=1))
        vol_ann_pct = (vol_T / np.sqrt(max(T, 1e-9))) * 100.0
        sharpe = (mu_ann_pct / vol_ann_pct) if vol_ann_pct > 1e-12 else 0.0
        return mu_ann_pct, vol_ann_pct, sharpe

    # Métricas si hay subyacente
    if ready:
        S0 = float(ss.get(k_spot, 0.01))
        K1 = float(ss.get(k_k1, 0.01))
        K2 = float(ss.get(k_k2, 0.01))
        sigma_pct = float(ss.get(k_sigma, 0.01))
        r_pct = float(ss.get(k_r, 0.0))
        days = int(ss.get(k_days, 360))

        sigma = sigma_pct / 100.0
        r = r_pct / 100.0
        T = days / 365.0

        call_k1_prima = _bs_call(S0, K1, r, sigma, T)
        put_k1_prima = _bs_put(S0, K1, r, sigma, T)
        call_k2_prima = _bs_call(S0, K2, r, sigma, T)
        put_k2_prima = _bs_put(S0, K2, r, sigma, T)

        rend_anual, vol_anual, sharpe = _calcular_metricas_mc(S0, K1, K2, sigma, r, T)

        DF = np.exp(-r * T)
        precio_cetes_hoy = 10.0 * DF
        spread_cost = call_k1_prima - call_k2_prima
        n_spreads = (10.0 - precio_cetes_hoy) / spread_cost if spread_cost > 1e-12 else 0.0
        precio_nota = precio_cetes_hoy + (n_spreads * spread_cost)

        st.markdown("##### Primas y Métricas de la Nota Estructurada (Modelo Black-Scholes)")
        metrics_cols = st.columns(8)
        metrics_cols[0].metric("Call(K1) — Prima", f"${call_k1_prima:,.4f}")
        metrics_cols[1].metric("Put(K1) — Prima", f"${put_k1_prima:,.4f}")
        metrics_cols[2].metric("Call(K2) — Prima", f"${call_k2_prima:,.4f}")
        metrics_cols[3].metric("Put(K2) — Prima", f"${put_k2_prima:,.4f}")
        metrics_cols[4].metric("Rendimiento anual (μ)", f"{rend_anual:.2f}%")
        metrics_cols[5].metric("Volatilidad anual (σ)", f"{vol_anual:.2f}%")
        metrics_cols[6].metric("Sharpe (μ/σ)", f"{sharpe:.2f}")
        metrics_cols[7].metric("Precio Nota", f"${precio_nota:,.4f}")
        st.markdown("---")

    left, right = st.columns([0.45, 0.55], gap="large")

    # --- Lado izquierdo: inputs
    with left:
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            sel_label = st.selectbox(
                "Subyacente",
                options=options,
                index=options.index(ss[k_sel]) if ss[k_sel] in options else 0,
                key=k_sel
            )

        ready = (sel_label != placeholder)

        # Inicialización al cambiar subyacente
        if ready and (ss.get(k_tkr) is None or ss.get(k_sel) != sel_label):
            tkr, name = _parse_ticker_from_label(sel_label)
            try:
                preload_hist_5y_daily([tkr])
            except Exception:
                pass
            spot_auto, vol_auto = _latest_spot_and_vol(tkr)
            ss[k_tkr] = tkr
            ss[k_name] = name
            ss[k_spot] = float(_min_safe(spot_auto, 0.01))
            ss[k_sigma] = float(_min_safe(vol_auto * 100.0, 0.01))
            ss[k_k1] = round(0.97 * spot_auto, 4)
            ss[k_k2] = round(1.25 * spot_auto, 4)
            ss[k_days] = 360
            ss[k_r] = float(_mx_risk_free_pct_default())
            st.rerun()

        with col1_2:
            S0 = st.number_input("Spot (S₀)", min_value=0.01, format="%.4f",
                                 value=float(ss.get(k_spot, 0.01)) if ready else 0.01,
                                 key=k_spot, disabled=not ready)

        col2_1, col2_2 = st.columns(2)
        with col2_1:
            K1 = st.number_input("K1 (call largo)", min_value=0.01, format="%.4f",
                                 value=float(ss.get(k_k1, 0.01)) if ready else 0.01,
                                 key=k_k1, disabled=not ready)
        with col2_2:
            K2 = st.number_input("K2 (call corto)", min_value=0.01, format="%.4f",
                                 value=float(ss.get(k_k2, 0.01)) if ready else 0.01,
                                 key=k_k2, disabled=not ready)

        col3_1, col3_2 = st.columns(2)
        with col3_1:
            sigma_pct = st.number_input("Volatilidad anual (%)", min_value=0.01, format="%.2f",
                                        value=float(ss.get(k_sigma, 0.01)) if ready else 0.01,
                                        key=k_sigma, disabled=not ready)
        with col3_2:
            r_pct = st.number_input("Tasa libre de riesgo anual (%)", min_value=0.00, format="%.2f",
                                    value=float(ss.get(k_r, 0.0)) if ready else 0.00,
                                    key=k_r, disabled=not ready)

        col4_1, col4_2 = st.columns(2)
        with col4_1:
            days = st.number_input("Plazo (días)", min_value=7, step=1,
                                   value=int(ss.get(k_days, 360)) if ready else 7,
                                   key=k_days, disabled=not ready)
        with col4_2:
            st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
            if st.button("Agregar nota", type="primary", use_container_width=True, disabled=not ready, key=f"{base_key}_add"):
                note_id = __import__("uuid").uuid4().hex[:8]
                sigma = float(sigma_pct) / 100.0
                r = float(r_pct) / 100.0
                T = float(days) / 365.0
                rend_anual, vol_anual, sharpe = _calcular_metricas_mc(S0, K1, K2, sigma, r, T)
                note = {
                    "note_id": note_id,
                    "name": f"Nota {ss.get(k_name) or ss.get(k_tkr) or 'SIN_SUBYACENTE'}",
                    "subyacente": ss.get(k_name, ""),
                    "ticker": ss.get(k_tkr, ""),
                    "spot": float(S0),
                    "volatilidad_pct": float(sigma_pct),
                    "K1": float(K1),
                    "K2": float(K2),
                    "plazo_d": int(days),
                    "tasa_libre_pct": float(r_pct),
                    "rend_anual_pct": float(rend_anual),
                    "vol_anual_pct": float(vol_anual),
                    "sharpe": float(sharpe),
                }
                ss[k_notes].append(note)
                # Inicializar peso equitativo
                n_notes = len(ss[k_notes])
                equal_weight = 100.0 / n_notes
                ss[k_weights] = {n["note_id"]: equal_weight for n in ss[k_notes]}
                st.toast("Nota agregada", icon="✅")
                st.rerun()

    # --- Lado derecho: payoff
    with right:
        st.markdown("**Payoff del call spread**")
        if not ready:
            st.info("Selecciona un subyacente para ver la gráfica.")
        else:
            sigma = float(sigma_pct) / 100.0
            r = float(r_pct) / 100.0
            T = float(days) / 365.0
            from tabs.structured_note import _bs_call  # evitar ciclo

            c1 = _bs_call(S0, K1, r, sigma, T)
            c2 = _bs_call(S0, K2, r, sigma, T)

            start = float(np.floor(K1 * 2) / 2.0)
            end_raw = K2 + 7.0
            end = float(np.floor(end_raw * 2) / 2.0)
            grid_ST = np.arange(start, end + 1e-9, 0.5)

            call_long   = np.maximum(grid_ST - K1, 0.0) - c1
            call_short  = c2 - np.maximum(grid_ST - K2, 0.0)
            call_spread = call_long + call_short

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=grid_ST, y=call_long,   mode="lines", name="Call largo (K1)"))
            fig.add_trace(go.Scatter(x=grid_ST, y=call_short,  mode="lines", name="Call corto (K2)"))
            fig.add_trace(go.Scatter(x=grid_ST, y=call_spread, mode="lines", name="Call spread", line=dict(width=3)))
            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Subyacente (ST)",
                yaxis_title="Payoff por unidad",
                height=360, margin=dict(l=10, r=10, t=10, b=10),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Tabla de notas agregadas
    if ss[k_notes]:
        st.markdown("---")
        st.markdown("#### Notas estructuradas agregadas")

        rows = []
        id_to_idx = {}
        for idx, note in enumerate(ss[k_notes]):
            note_id = note["note_id"]
            id_to_idx[idx] = note_id
            rows.append({
                "_idx": idx,
                "Subyacente": note["subyacente"],
                "Spot": note["spot"],
                "Volatilidad (%)": note["volatilidad_pct"],
                "K1": note["K1"],
                "K2": note["K2"],
                "Plazo (días)": note["plazo_d"],
                "Tasa libre (%)": note["tasa_libre_pct"],
                "Rend. anual (%)": note["rend_anual_pct"],
                "Vol. anual (%)": note["vol_anual_pct"],
                "Sharpe (μ/σ)": note["sharpe"],
                "Peso (%)": round(ss[k_weights].get(note_id, 0.0), 2),
                "Eliminar": False
            })

        df_notes = pd.DataFrame(rows)
        edited_df = st.data_editor(
            df_notes,
            column_config={
                "_idx": None,
                "Subyacente": st.column_config.TextColumn("Subyacente", disabled=True),
                "Spot": st.column_config.NumberColumn("Spot", disabled=True, format="%.4f"),
                "Volatilidad (%)": st.column_config.NumberColumn("Volatilidad (%)", disabled=True, format="%.2f"),
                "K1": st.column_config.NumberColumn("K1", disabled=True, format="%.4f"),
                "K2": st.column_config.NumberColumn("K2", disabled=True, format="%.4f"),
                "Plazo (días)": st.column_config.NumberColumn("Plazo (días)", disabled=True),
                "Tasa libre (%)": st.column_config.NumberColumn("Tasa libre (%)", disabled=True, format="%.2f"),
                "Rend. anual (%)": st.column_config.NumberColumn("Rend. anual (%)", disabled=True, format="%.2f"),
                "Vol. anual (%)": st.column_config.NumberColumn("Vol. anual (%)", disabled=True, format="%.2f"),
                "Sharpe (μ/σ)": st.column_config.NumberColumn("Sharpe (μ/σ)", disabled=True, format="%.2f"),
                "Peso (%)": st.column_config.NumberColumn("Peso (%)", min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
                "Eliminar": st.column_config.CheckboxColumn("Eliminar", help="Marca para eliminar")
            },
            hide_index=True,
            use_container_width=True,
            key=f"{base_key}_notes_editor"
        )

        total_weight = edited_df["Peso (%)"].sum()
        col_sum, col_msg = st.columns([0.3, 0.7])
        col_sum.metric("Suma de pesos", f"{total_weight:.2f}%")
        if abs(total_weight - 100.0) > 0.5:
            col_msg.caption("⚠️ **Los pesos deben sumar 100%**")

        # Actualizar pesos en session state
        for _, row in edited_df.iterrows():
            idx = int(row["_idx"])
            note_id = id_to_idx[idx]
            ss[k_weights][note_id] = float(row["Peso (%)"])

        # Eliminar notas marcadas
        indices_to_delete = edited_df[edited_df["Eliminar"]]["_idx"].tolist()
        if indices_to_delete:
            note_ids_to_delete = [id_to_idx[idx] for idx in indices_to_delete]
            ss[k_notes] = [n for n in ss[k_notes] if n["note_id"] not in note_ids_to_delete]
            for note_id in note_ids_to_delete:
                ss[k_weights].pop(note_id, None)
            st.rerun()


def mostrar_fixed_income_compacto():
    """
    Interfaz compacta para bonos en el contexto de aportaciones:
    - Gubernamental: auto-completado (CETES/MBONO/UDIBONO/BONDES)
    - Privada Nac. / Internacional: captura manual
    """
    from tabs.fixed_income import (
        GOB_INSTRUMENTOS, _prefill_from_gob, _update_instr_caption,
        CAT_LABELS, _instruments_for_category,
        price_bond_dirty_clean, FREQ_MAP, _add_years
    )
    from datetime import timedelta

    ss = st.session_state

    base_key = "fic_"  # Incluye el guión bajo para que coincida con las claves en fixed_income.py
    fecha_aportacion = ss.get("contrib_effective_date") or _today()

    # Keys de estado (usando los mismos nombres que en fixed_income.py)
    k_tipo = f"{base_key}tipo_bono"
    k_cat = f"{base_key}categoria"
    k_instr = f"{base_key}instrumento"
    k_nombre = f"{base_key}nombre"
    k_liq = f"{base_key}liquidacion"
    k_val = f"{base_key}val"  # Cambiado de valuacion a val
    k_mat = f"{base_key}maturity"
    k_freq = f"{base_key}freq"  # Cambiado de frecuencia a freq
    k_cupon = f"{base_key}cupon_pct"
    k_ytm = f"{base_key}ytm_pct"
    k_nominal = f"{base_key}nominal"
    k_calif = f"{base_key}calificacion"
    k_bonds_list = f"{base_key}bonds_list"
    k_weights = f"{base_key}weights"
    k_prev_instr = f"{base_key}prev_instr"
    k_prev_liq = f"{base_key}prev_liq"

    # Estado inicial
    ss.setdefault(k_tipo, ss.get("fic_tipo_bono", "Deuda Gubernamental"))
    ss.setdefault(k_cat, "CETES")
    ss.setdefault(k_instr, "CETES 2d")
    ss.setdefault(k_liq, fecha_aportacion)
    ss.setdefault(k_val, fecha_aportacion)
    ss.setdefault(k_mat, fecha_aportacion + timedelta(days=28))
    ss.setdefault(k_freq, "Semestral")
    ss.setdefault(k_cupon, 0.0)
    ss.setdefault(k_ytm, 0.0)
    ss.setdefault(k_nominal, 100.0)
    ss.setdefault(k_calif, "Soberano MX")
    ss.setdefault(k_bonds_list, [])
    ss.setdefault(k_weights, {})
    
    tipo_selected = ss[k_tipo]
    is_gob = (tipo_selected == "Deuda Gubernamental")

    if is_gob:
        # === BONOS GUBERNAMENTALES ===
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            st.selectbox("Tipo de Bono", ["Deuda Gubernamental", "Deuda Privada Nacional", "Deuda Internacional"],
                         index=0, key=k_tipo, disabled=True,
                         help="El tipo se determina por la categoría seleccionada en aportaciones")
        with col1_2:
            st.text_input("Calificación", value="Soberano MX", disabled=True)

        col2_1, col2_2 = st.columns(2)
        with col2_1:
            cat_label = st.selectbox("Categoría", options=list(CAT_LABELS.values()), key=k_cat)
        with col2_2:
            cat_options = _instruments_for_category(cat_label)
            if ss.get(k_instr) not in cat_options and cat_options:
                ss[k_instr] = cat_options[0]
            default_idx = cat_options.index(ss[k_instr]) if ss.get(k_instr) in cat_options else 0
            instr_sel = st.selectbox("Instrumento", options=cat_options, index=default_idx, key=k_instr)

        # Llama a la función de ayuda para pre-llenar todos los datos cuando algo cambie
        if ss.get(k_prev_instr) != instr_sel or ss.get(k_prev_liq) != ss.get(k_liq):
            _prefill_from_gob(instr_sel, ss[k_liq], prefix=base_key)
            ss[k_prev_instr] = instr_sel
            ss[k_prev_liq] = ss[k_liq]

        col3_1, col3_2, col3_3 = st.columns(3)
        with col3_1:
            st.date_input("Liquidación", key=k_liq)
        with col3_2:
            st.date_input("Valuación", key=k_val)
        with col3_3:
            st.date_input("Vencimiento", key=k_mat)

        caption_text = _update_instr_caption(ss[k_instr], ss[k_liq], prefix=base_key)
        if caption_text:
            st.caption(caption_text)

        col4_1, col4_2, col4_3 = st.columns(3)
        with col4_1:
            # Usamos una key dinámica para forzar la actualización del valor mostrado
            cupon_view_key = f"{k_cupon}_view_{ss.get(k_cupon, 0.0)}"
            st.number_input("Cupón (%)", value=float(ss.get(k_cupon, 0.0)), step=0.01, disabled=True, key=cupon_view_key)
        with col4_2:
            st.selectbox("Frecuencia", options=list(FREQ_MAP.keys()), key=k_freq, disabled=True)
        with col4_3:
            st.number_input("Nominal", value=float(ss.get(k_nominal, 100.0)), min_value=0.01, step=1.0, key=k_nominal, disabled=True)

    else:
        # === CAPTURA MANUAL (Sin cambios) ===
        st.selectbox("Tipo de Bono", ["Deuda Gubernamental", "Deuda Privada Nacional", "Deuda Internacional"],
                     index=["Deuda Gubernamental", "Deuda Privada Nacional", "Deuda Internacional"].index(tipo_selected),
                     key=k_tipo, disabled=True)
        st.text_input("Nombre del Bono", key=k_nombre)

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.date_input("Liquidación", value=ss.get(k_liq, fecha_aportacion), key=k_liq)
        with col_f2:
            st.date_input("Valuación", value=ss.get(k_val, fecha_aportacion), key=k_val)
        with col_f3:
            st.date_input("Vencimiento", value=ss.get(k_mat, fecha_aportacion), key=k_mat)

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.number_input("Cupón (%)", min_value=0.0, step=0.01, key=k_cupon)
        with col_c2:
            st.selectbox("Frecuencia", options=list(FREQ_MAP.keys()), key=k_freq)
        with col_c3:
            st.number_input("Nominal", min_value=0.01, step=1.0, key=k_nominal)

        col_y1, col_y2 = st.columns(2)
        with col_y1:
            st.number_input("YTM (%)", min_value=0.0, step=0.01, key=k_ytm)
        with col_y2:
            from tabs.fixed_income import MX_RATINGS, US_RATINGS
            rating_opts = US_RATINGS if tipo_selected == "Deuda Internacional" else MX_RATINGS
            st.selectbox("Calificación", options=rating_opts, key=k_calif)

    # === BOTÓN AGREGAR COMÚN ===
    if st.button("Agregar bono", type="primary", use_container_width=True, key=f"{base_key}_add"):
        try:
            from tabs.fixed_income import FREQ_MAP as _FREQ_MAP_LOCAL
            freq_val = _FREQ_MAP_LOCAL[ss[k_freq]]

            dirty, clean, accrued, last_cp, next_cp, _, _ = price_bond_dirty_clean(
                valuation=ss[k_val],
                maturity=ss[k_mat],
                coupon_annual_pct=float(ss[k_cupon]),
                ytm_annual_pct=float(ss[k_ytm]),
                freq_per_year=freq_val,
                nominal=float(ss[k_nominal]),
                base="ACT/360",
                anchor=ss[k_liq]
            )

            bond_id = __import__("uuid").uuid4().hex[:8]
            nombre = ss[k_instr] if is_gob else ss.get(k_nombre, "Bono sin nombre")
            calif_final = "Soberano MX" if is_gob else ss.get(k_calif, "")

            bond = {
                "bond_id": bond_id,
                "nombre": nombre,
                "tipo": ss[k_tipo],
                "liquidacion": ss[k_liq].strftime("%Y-%m-%d"),
                "maturity": ss[k_mat].strftime("%Y-%m-%d"),
                "cupon_pct": float(ss[k_cupon]),
                "frecuencia": ss[k_freq],
                "ytm_pct": float(ss[k_ytm]),
                "nominal": float(ss[k_nominal]),
                "calificacion": calif_final,
                "precio_sucio": float(dirty),
                "precio_limpio": float(clean),
            }

            ss[k_bonds_list].append(bond)

            # Pesos equitativos
            n_bonds = len(ss[k_bonds_list])
            equal_weight = 100.0 / n_bonds
            ss[k_weights] = {b["bond_id"]: equal_weight for b in ss[k_bonds_list]}

            st.toast("Bono agregado", icon="✅")
            st.rerun()

        except Exception as e:
            st.error(f"Error al agregar bono: {e}")

    # === TABLA DE BONOS AGREGADOS ===
    if ss[k_bonds_list]:
        st.markdown("---")
        st.markdown("#### Bonos agregados")

        rows = []
        id_to_idx = {}
        for idx, bond in enumerate(ss[k_bonds_list]):
            bond_id = bond["bond_id"]
            id_to_idx[idx] = bond_id
            rows.append({
                "_idx": idx,
                "Nombre": bond["nombre"],
                "Tipo": bond["tipo"],
                "Maturity": bond["maturity"],
                "Cupón (%)": bond["cupon_pct"],
                "YTM (%)": bond["ytm_pct"],
                "Precio Limpio": bond["precio_limpio"],
                "Peso (%)": round(ss[k_weights].get(bond_id, 0.0), 2),
                "Eliminar": False
            })

        df_bonds = pd.DataFrame(rows)
        edited_df = st.data_editor(
            df_bonds,
            column_config={
                "_idx": None,
                "Nombre": st.column_config.TextColumn("Nombre", disabled=True),
                "Tipo": st.column_config.TextColumn("Tipo", disabled=True),
                "Maturity": st.column_config.TextColumn("Vencimiento", disabled=True),
                "Cupón (%)": st.column_config.NumberColumn("Cupón (%)", disabled=True, format="%.2f"),
                "YTM (%)": st.column_config.NumberColumn("YTM (%)", disabled=True, format="%.2f"),
                "Precio Limpio": st.column_config.NumberColumn("Precio Limpio", disabled=True, format="%.2f"),
                "Peso (%)": st.column_config.NumberColumn("Peso (%)", min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
                "Eliminar": st.column_config.CheckboxColumn("Eliminar")
            },
            hide_index=True,
            use_container_width=True,
            key=f"{base_key}_bonds_editor"
        )

        total_weight = edited_df["Peso (%)"].sum()
        col_sum, col_msg = st.columns([0.3, 0.7])
        col_sum.metric("Suma de pesos", f"{total_weight:.2f}%")
        if abs(total_weight - 100.0) > 0.5:
            col_msg.caption("⚠️ **Los pesos deben sumar 100%**")

        # Actualizar pesos
        for _, row in edited_df.iterrows():
            idx = int(row["_idx"])
            bond_id = id_to_idx[idx]
            ss[k_weights][bond_id] = float(row["Peso (%)"])

        # Eliminar marcados
        indices_to_delete = edited_df[edited_df["Eliminar"]]["_idx"].tolist()
        if indices_to_delete:
            bond_ids_to_delete = [id_to_idx[idx] for idx in indices_to_delete]
            ss[k_bonds_list] = [b for b in ss[k_bonds_list] if b["bond_id"] not in bond_ids_to_delete]
            for bond_id in bond_ids_to_delete:
                ss[k_weights].pop(bond_id, None)
            st.rerun()