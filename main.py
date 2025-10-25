# main.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, date
import time
import threading

import numpy as np
import pandas as pd
import streamlit as st

# ==================== CONFIG ====================
SHOW_BACKGROUND_UPDATER_EXPANDER: bool = False # ✅ NUEVO: Controla si se muestra el expander
PRELOAD_IN_MEMORY: bool = False   # no bloquee splash
SHOW_SAVE_BUTTON: bool = True
AUTO_REFRESH_SECS: float = 2.0    # <- intervalo de actualización
AUTO_UPDATE_INTERVAL_HOURS = 24  # Actualización cada 24 horas

# ==================== IMPORTS TABS ==================
from tabs import equities, fixed_income, consar_limits, fx, portfolio, projection, structured_note, cerpis_ckds, operations
from tabs.set_page_config import CircleStatus
from tabs.yf_store import preload_hist_5y_daily, preload_meta, ensure_parquet_incremental_verbose, PRICES_DIR
from tabs.equities import _benchmark_for_index

# 👇 NUEVO: runtime para Act As Of y ops_started_at
from tabs import runtime

# ==================== PAGE CONFIG ======================
st.set_page_config(page_title="AFORE Portafolio", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem !important; }
      h1 { margin-bottom: 0rem !important; }
      div[data-baseweb="tab-list"] { margin-top: -0.3rem !important; }
      [data-testid="stStatusWidget"] { display: none !important; }
      [data-testid="stSpinner"] { display: none !important; }
      .stSpinner { display: none !important; }
      div[role="status"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

_STATE_PATH = Path("./afore_state.json")
_FIXED_KEYS = {"siefore_selected", "afore_selected", "structured_notes"}

# ==================== JSON HELPERS =====================
def _to_jsonable(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)): return obj
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, (datetime, date, pd.Timestamp)): return obj.isoformat()
    if isinstance(obj, (list, tuple, set)): return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict): return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return {"__kind__": "DataFrame","data": obj.to_dict(orient="records"),"columns": list(obj.columns)}
    if isinstance(obj, pd.Series): return {"__kind__": "Series","data": obj.to_list(),"name": obj.name}
    return str(obj)

def _from_jsonable(val):
    if isinstance(val, dict) and "__kind__" in val:
        kind = val["__kind__"]
        if kind == "DataFrame":
            try: return pd.DataFrame(val.get("data", []), columns=val.get("columns"))
            except Exception: return val
        if kind == "Series":
            try: return pd.Series(val.get("data", []), name=val.get("name"))
            except Exception: return val
    if isinstance(val, list): return [_from_jsonable(x) for x in val]
    if isinstance(val, dict): return {k: _from_jsonable(v) for k, v in val.items()}
    return val

# ==================== AUTOSAVE =========================
def _keys_to_persist():
    keys = set()
    for k in st.session_state.keys():
        if k in _FIXED_KEYS: keys.add(k)
        elif k.endswith("_selection") or k.endswith("_rows") or k.endswith("_picks"): keys.add(k)
        elif k == "structured_notes": keys.add(k)
    return sorted(keys)

def _snapshot_state():
    snap = {}
    for k in _keys_to_persist():
        try: snap[k] = _to_jsonable(st.session_state[k])
        except Exception: pass
    return snap

def _restore_state(data: dict):
    for k, v in (data or {}).items():
        st.session_state[k] = _from_jsonable(v)

def _auto_load_once():
    if st.session_state.get("_autosave_loaded", False): return
    if _STATE_PATH.exists():
        try:
            data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
            _restore_state(data)
        except Exception as e:
            st.toast(f"No se pudo cargar la configuración: {e}", icon="⚠️")
    st.session_state["_autosave_loaded"] = True

def _save_config():
    try:
        payload = _snapshot_state()
        _STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        st.toast("Configuración guardada (acciones, bonos, divisas, límites y notas)")
    except Exception as e:
        st.error(f"No se pudo guardar: {e}")

# ==================== CSV UNIVERSO =====================
def _read_universe_csv(path: str = "data/constituents_with_yahoo.csv") -> list[str]:
    p = Path(path)
    if not p.exists(): return []
    try: df = pd.read_csv(p)
    except Exception: return []
    if df.empty: return []
    for c in ["yahoo","ticker","Ticker","symbol","Symbol"]:
        if c in df.columns: col = c; break
    else: col = df.columns[0]
    vals = (df[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan}).dropna().tolist())
    return list(dict.fromkeys(vals))

# ==================== BACKGROUND UPDATE ==================
_BG_STATUS_PATH = Path("data/bg_status.json")
_BG_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)

def _bg_status_read() -> dict:
    try:
        if _BG_STATUS_PATH.exists():
            return json.loads(_BG_STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _bg_status_write(payload: dict) -> None:
    try:
        _BG_STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def _bg_update_worker(universe: list[str], *, force_refresh: bool, max_age_hours: float, batch_size: int = 25):
    """
    Worker que actualiza precios Y fundamentales en segundo plano.
    """
    total = len(universe)
    now_timestamp = time.time()
    status = {
        "state": "running",
        "total": total,
        "processed": 0,
        "started_at": now_timestamp,
        "last_update_date": datetime.fromtimestamp(now_timestamp).strftime("%Y-%m-%d %H:%M:%S"),
        "last": None,
        "bootstrap_rows": 0,
        "incremental_rows": 0,
        "fundamentals_updated": 0,
        "errors": 0,
        "finished_at": None,
        "hb": now_timestamp,
        "has_changes": False
    }
    _bg_status_write(status)

    _hb_running = True
    def _hb_loop():
        while _hb_running:
            status["hb"] = time.time(); _bg_status_write(status); time.sleep(2.0)
    hb_thread = threading.Thread(target=_hb_loop, daemon=True); hb_thread.start()

    def _on_log(ticker: str, added_rows: int, phase: str, took_secs: float):
        if phase.startswith("bootstrap"):
            status["bootstrap_rows"] += max(0, added_rows)
            status["processed"] = min(total, status["processed"] + 1)
            if added_rows > 0:
                status["has_changes"] = True
        elif phase.startswith("incremental"):
            status["incremental_rows"] += max(0, added_rows)
            status["processed"] = min(total, status["processed"] + 1)
            if added_rows > 0:
                status["has_changes"] = True
        elif phase == "fundamentals":
            status["fundamentals_updated"] += max(0, added_rows)
            if added_rows > 0:
                status["has_changes"] = True
        elif phase.endswith("_error"):
            status["errors"] += 1
        status["last"] = {"ticker": ticker,"phase": phase,"added_rows": added_rows,"took_secs": round(took_secs, 1),"ts": time.time()}
        status["hb"] = time.time()
        _bg_status_write(status)

    try:
        # Actualizar precios Y fundamentales
        ensure_parquet_incremental_verbose(
            universe,
            years=5,
            margin_days=3,
            batch_new=batch_size,
            sleep_between_calls=0.05,
            max_age_hours=max_age_hours,
            force_refresh=force_refresh,
            fetch_pe_along=True,  # ✅ Activar descarga de fundamentales
            pe_max_age_hours=168.0,  # 7 días de frescura para fundamentales
            on_log=_on_log
        )
    except Exception:
        status["errors"] += 1
    finally:
        finish_timestamp = time.time()
        status["state"] = "done"
        status["finished_at"] = finish_timestamp
        status["last_update_date"] = datetime.fromtimestamp(finish_timestamp).strftime("%Y-%m-%d %H:%M:%S")
        status["hb"] = finish_timestamp
        _bg_status_write(status)
        _hb_running = False
        try: hb_thread.join(timeout=0.2)
        except Exception: pass

def _bg_is_running() -> bool:
    t = st.session_state.get("_bg_thread")
    return bool(t and t.is_alive())

def _start_background_update(universe: list[str], *, force_refresh: bool = False, max_age_hours: float = 0.1, batch_size: int = 25):
    if _bg_is_running(): return
    thread = threading.Thread(
        target=_bg_update_worker,
        args=(universe,),
        kwargs={"force_refresh": force_refresh, "max_age_hours": max_age_hours, "batch_size": batch_size},
        daemon=True
    )
    thread.start()
    st.session_state["_bg_thread"] = thread
    st.session_state["_bg_started"] = True

def _should_run_auto_update(interval_hours: float = 24.0) -> bool:
    status = _bg_status_read()
    last_update_date = status.get("last_update_date")
    
    if not last_update_date:
        return True
    
    try:
        last_update_dt = datetime.strptime(last_update_date, "%Y-%m-%d %H:%M:%S")
        now_dt = datetime.now()
        elapsed_hours = (now_dt - last_update_dt).total_seconds() / 3600
        return elapsed_hours >= interval_hours
    except Exception:
        return True

def _ensure_background_update_on_boot(universe: list[str], batch_size: int = 25, interval_hours: float = 24.0):
    if not universe: return
    if _bg_is_running(): return
    
    status = _bg_status_read()
    
    if status.get("state") != "running" and not st.session_state.get("_bg_started", False):
        if _should_run_auto_update(interval_hours):
            _start_background_update(universe, force_refresh=False, max_age_hours=0.1, batch_size=batch_size)

# ==================== PREWARM (no bloqueante) ===========
def _prewarm_everything(cs: CircleStatus):
    cs.begin("Preparando entorno…")
    cs.step("Cargando configuración guardada")

    cs.step("Iniciando actualización de histórico (segundo plano)")
    universe = _read_universe_csv("data/constituents_with_yahoo.csv")
    _ensure_background_update_on_boot(universe, batch_size=25, interval_hours=AUTO_UPDATE_INTERVAL_HOURS)

    cs.step("Leyendo selección de acciones del usuario")
    sel = st.session_state.get("equity_selection", {}) or {}
    tickers_sel = list(sel.keys())

    benches = []
    for t in tickers_sel[:50]:
        try:
            idx_name = (sel.get(t) or {}).get("index", "")
            benches.append(_benchmark_for_index(idx_name, t))
        except Exception:
            continue
    all_to_preload = list({*tickers_sel[:50], *benches})

    if PRELOAD_IN_MEMORY and all_to_preload:
        cs.step("Precargando 5y/1d (memoria) para selección")
        try: preload_hist_5y_daily(all_to_preload[:20])
        except Exception: pass
    else:
        cs.step("Saltando precarga en memoria")

    if PRELOAD_IN_MEMORY and tickers_sel:
        cs.step("Precargando metadatos (selección)")
        try: preload_meta(tickers_sel[:40])
        except Exception: pass
    else:
        cs.step("Saltando precarga de metadatos")

    cs.step("Sincronizando divisas (FX)")
    try:
        if hasattr(fx, "_ensure_fx_selection"): fx._ensure_fx_selection()
    except Exception: pass

    cs.step("Verificando límites CONSAR")
    try:
        if "consar_limits_current" not in st.session_state and hasattr(consar_limits, "_init_state"):
            consar_limits._init_state()
    except Exception: pass

    cs.done("Todo listo")

# ==================== INIT =============================
_auto_load_once()

# ==================== HEADER (TÍTULO + DATEBOX + GUARDAR) ==========================
def _today() -> date:
    return datetime.now().date()

# asegura llaves runtime
runtime.ensure_ops_keys()
is_operating = st.session_state.get("ops_operating", False)

# si ya hay operación, asegúrate de tener ops_started_at
if is_operating and runtime.get_ops_started_at() is None:
    start_candidate = st.session_state.get("ops_start_date") or _today()
    runtime.set_ops_started_at(start_candidate)

ops_start = runtime.get_ops_started_at() or st.session_state.get("ops_start_date") or _today()
act_value = runtime.get_act_as_of() or _today()

# CSS fino para alinear
st.markdown(
    """
    <style>
      .top-row { margin-top: -6px; }
      .compact-btn > div > button { margin-top: 2px !important; }
      div[data-testid="stDateInput"] > div { margin-top: 0px !important; }
      div[data-testid="stDateInput"] { margin-top: 0px !important; padding-top: 0px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

c1, c2 = st.columns([0.80, 0.20])
with c1:
    st.markdown("<div class='top-row'></div>", unsafe_allow_html=True)
    st.title("AFORE Dashboard")


with c2:
    if is_operating:
        st.markdown("<br>", unsafe_allow_html=True)
        min_dt = ops_start
        max_dt = _today()
        new_dt = st.date_input(
            "Fecha de operación",
            value=act_value,
            min_value=min_dt,
            max_value=max_dt,
            key="__act_as_of_main",
        )
        if new_dt != act_value:
            runtime.set_act_as_of(new_dt)
            st.toast(f"Fecha efectiva: {new_dt:%d/%m/%Y}", icon="🗓️")
            st.rerun()

# ==================== HELPERS PARA INFO ADICIONAL =================
def _get_latest_price_date() -> str:
    try:
        latest_date = None
        parquet_files = list(PRICES_DIR.glob("*.parquet"))[:20]
        
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                if not df.empty and hasattr(df.index, 'max'):
                    file_max = df.index.max()
                    if latest_date is None or file_max > latest_date:
                        latest_date = file_max
            except Exception:
                continue
        
        if latest_date:
            return pd.to_datetime(latest_date).strftime("%Y-%m-%d")
        return "N/A"
    except Exception:
        return "N/A"

def _calculate_next_auto_update(interval_hours: float = 24.0) -> tuple[int, str]:
    status = _bg_status_read()
    last_update_date = status.get("last_update_date")
    
    if not last_update_date:
        return (0, "pendiente (nunca ejecutada)")
    
    try:
        last_update_dt = datetime.strptime(last_update_date, "%Y-%m-%d %H:%M:%S")
        next_update_dt = last_update_dt + pd.Timedelta(hours=interval_hours)
        now_dt = datetime.now()
        
        seconds_remaining = int((next_update_dt - now_dt).total_seconds())
        
        if seconds_remaining <= 0:
            return (0, "debería iniciarse pronto")
        
        if seconds_remaining < 60:
            return (seconds_remaining, f"{seconds_remaining}s")
        elif seconds_remaining < 3600:
            mins = seconds_remaining // 60
            return (seconds_remaining, f"{mins}m")
        else:
            hours = seconds_remaining // 3600
            mins = (seconds_remaining % 3600) // 60
            if mins > 0:
                return (seconds_remaining, f"{hours}h {mins}m")
            else:
                return (seconds_remaining, f"{hours}h")
    except Exception:
        return (0, "error al calcular")

# ==================== PANEL BACKGROUND =================
universe = _read_universe_csv("data/constituents_with_yahoo.csv")
seconds_to_next, _ = _calculate_next_auto_update(AUTO_UPDATE_INTERVAL_HOURS)
is_currently_running = _bg_is_running() or (_bg_status_read().get("state") == "running")

if seconds_to_next <= 0 and not is_currently_running and not st.session_state.get("_auto_update_triggered", False):
    st.session_state["_auto_update_triggered"] = True
    _bg_status_write({})
    st.session_state["_bg_started"] = False
    _start_background_update(universe, force_refresh=True, max_age_hours=0.1, batch_size=25)
    st.toast("⏰ Iniciando actualización automática (precios + fundamentales)...", icon="🔄")
    time.sleep(0.5)
    st.rerun()

if is_currently_running:
    st.session_state["_auto_update_triggered"] = False

# ==================== MODIFICACIÓN AL EXPANDER DE ACTUALIZACIÓN ====================
if SHOW_BACKGROUND_UPDATER_EXPANDER:
    with st.expander("🧵 Actualización de precios y fundamentales (segundo plano)", expanded=False):
        _ensure_background_update_on_boot(universe, batch_size=25, interval_hours=AUTO_UPDATE_INTERVAL_HOURS)

        status = _bg_status_read()
        state = status.get("state")
        processed = status.get("processed", 0)
        total = max(1, status.get("total", 1))
        last = status.get("last") or {}
        
        yfinance_ok = False
        try:
            import yfinance as yf
            yfinance_ok = True
        except ImportError:
            yfinance_ok = False
        
        next_ui_refresh_in = None
        is_actively_running = (state == "running") or _bg_is_running()
        
        if is_actively_running:
            last_rerun = st.session_state.get("_last_rerun_time", time.time())
            elapsed = time.time() - last_rerun
            next_ui_refresh_in = max(0, int(AUTO_REFRESH_SECS - elapsed))
        
        latest_price_date = _get_latest_price_date()
        seconds_to_next, next_update_text = _calculate_next_auto_update(AUTO_UPDATE_INTERVAL_HOURS)
        
        if state == "done":
            if not yfinance_ok:
                st.error(
                    "❌ **ERROR CRÍTICO: yfinance no está instalado**\n\n"
                    "La actualización automática no puede funcionar sin yfinance.\n\n"
                    "**Solución:**\n"
                    "```bash\n"
                    "pip install --upgrade yfinance\n"
                    "```\n\n"
                    "Luego reinicia la aplicación."
                )
            elif processed == 0 and status.get('bootstrap_rows', 0) == 0 and status.get('incremental_rows', 0) == 0:
                st.warning(
                    "⚠️ **Actualización completada sin cambios**\n\n"
                    "No se procesaron archivos. Esto puede indicar:\n"
                    "- Todos los archivos ya están actualizados\n"
                    "- Hay un problema de conectividad\n"
                    "- Hay un error en la configuración\n\n"
                    "Usa el panel de diagnóstico abajo para más detalles."
                )
            else:
                has_changes = status.get("has_changes", False)
                st.success(
                    f"✅ Actualización finalizada {'con cambios' if has_changes else 'sin cambios'} · "
                    f"precios: +{status.get('bootstrap_rows',0)} bootstrap, +{status.get('incremental_rows',0)} incremental · "
                    f"fundamentales: {status.get('fundamentals_updated',0)} actualizados"
                )
            
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"📅 Última fecha de precios: **{latest_price_date}**")
            with col2:
                st.caption(f"⏰ Próxima actualización automática en: **{next_update_text}**")
        
        elif is_actively_running:
            if not yfinance_ok:
                st.error(
                    "❌ **ERROR: yfinance no está instalado**\n\n"
                    "El proceso de actualización va a fallar. Instala yfinance:\n\n"
                    "```bash\n"
                    "pip install --upgrade yfinance\n"
                    "```"
                )
            
            st.progress(processed / total if total > 0 else 0.0)
            st.write(f"Procesado {processed}/{total} · precios: +{status.get('bootstrap_rows',0)} bootstrap, +{status.get('incremental_rows',0)} incremental · fundamentales: {status.get('fundamentals_updated',0)}")
            if last:
                since_last = int(time.time() - float(last.get("ts", time.time())))
                phase_display = last.get('phase', '')
                if phase_display == 'fundamentals':
                    phase_display = '📊 fundamentales'
                elif phase_display == 'skipped_error':
                    phase_display = '❌ error'
                st.caption(f"Último: `{last.get('ticker')}` · fase **{phase_display}** · +{last.get('added_rows')} · {last.get('took_secs')}s · hace {since_last}s")
            
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"📅 Última fecha de precios: **{latest_price_date}**")
            with col2:
                if next_ui_refresh_in is not None:
                    st.caption(f"🔄 Actualizando automáticamente (próximo refresco en {next_ui_refresh_in}s)")
        
        else:
            if not yfinance_ok:
                st.error(
                    "❌ **ERROR CRÍTICO: yfinance no está instalado**\n\n"
                    "La actualización automática no funcionará hasta que instales yfinance.\n\n"
                    "**Solución:**\n"
                    "```bash\n"
                    "pip install --upgrade yfinance\n"
                    "```\n\n"
                    "Luego reinicia la aplicación con: `streamlit run main.py`"
                )
            else:
                st.info("La actualización se inicia automáticamente al abrir la app.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"📅 Última fecha de precios: **{latest_price_date}**")
            with col2:
                st.caption(f"⏰ Próxima actualización automática en: **{next_update_text}**")

# ==================== SPLASH ===========================
if "_splash_done" not in st.session_state: st.session_state["_splash_done"] = False
if not st.session_state["_splash_done"]:
    cs = CircleStatus("Preparando y optimizando", total_steps=7)
    _prewarm_everything(cs)
    cs.close()
    st.session_state["_splash_done"] = True

# ==================== TABS =============================
tabs = st.tabs([
    "Portafolio","AFORE","Renta Variable","Renta Fija","Divisas","Nota Estructurada","Operaciones", "Proyección"
])
with tabs[0]: portfolio.render()
with tabs[1]: consar_limits.render()
with tabs[2]: equities.render()
with tabs[3]: fixed_income.render()
with tabs[4]: fx.render()
with tabs[5]: structured_note.render()
with tabs[6]: operations.render()
with tabs[7]: projection.render()