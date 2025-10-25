# tabs/yf_store.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Callable
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# yfinance puede fallar silenciosamente; maneja con cuidado
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False


# ================== Helpers internos ==================
def _normalize_yf_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Devuelve un DataFrame con columnas estándar y una columna de fecha explícita.
    No asume que el índice sea DatetimeIndex (lo vuelve columna).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Si el índice parece ser de fechas, lo llevamos a columna 'Date'
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={out.index.name or "index": "Date"})
    else:
        if "Date" not in out.columns:
            first_col = out.columns[0]
            try:
                pd.to_datetime(out[first_col], errors="raise")
                out = out.rename(columns={first_col: "Date"})
            except Exception:
                return pd.DataFrame()

    # Aplanar MultiIndex de columnas si aparece
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["|".join([str(c) for c in col if str(c) != ""]).strip("|") for col in out.columns]

    # Crear 'Adj Close' si falta
    if "Adj Close" not in out.columns:
        if "Adj close" in out.columns:
            out["Adj Close"] = out["Adj close"]
        elif "AdjClose" in out.columns:
            out["Adj Close"] = out["AdjClose"]
        elif "Close" in out.columns:
            out["Adj Close"] = out["Close"]

    cols_keep = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    out = out[[c for c in cols_keep if c in out.columns]].copy()

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])
    out["Date"] = out["Date"].dt.tz_localize(None)

    out["ticker"] = ticker
    return out


def _hist_pool() -> Dict[Tuple[str, str, str], pd.DataFrame]:
    """Pool en sesión para históricos {(ticker,'5y','1d'): df}."""
    return st.session_state.setdefault("yf_hist_pool", {})


# ================== Meta unificada (moneda + info) ==================
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_meta(ticker: str) -> dict:
    meta = {
        "ticker": ticker,
        "currency": None,
        "longName": None,
        "shortName": None,
        "sector": None,
        "industry": None,
        "exchange": None,
        "marketCap": None,
        "longBusinessSummary": None,
        "city": None,
        "state": None,
        "country": None,
        "fullTimeEmployees": None,
        "website": None,
        "founded": None,
        "ipo_year": None,
        "trailingPE": None,
        "forwardPE": None,
    }
    if not YF_OK:
        t = (ticker or "").upper()
        if t.endswith(".MX"):
            meta["currency"] = "MXN"
        elif t.endswith(".SA"):
            meta["currency"] = "BRL"
        else:
            meta["currency"] = "USD"
        return meta

    try:
        tk = yf.Ticker(ticker)

        try:
            fi = getattr(tk, "fast_info", None)
            if isinstance(fi, dict):
                if fi.get("currency"):
                    meta["currency"] = str(fi.get("currency")).upper()
                if fi.get("market_cap") is not None:
                    meta["marketCap"] = fi.get("market_cap")
                if fi.get("exchange"):
                    meta["exchange"] = fi.get("exchange")
        except Exception:
            pass

        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        if not meta["currency"]:
            c = info.get("currency")
            if c:
                meta["currency"] = str(c).upper()
        meta["longName"] = info.get("longName") or info.get("shortName") or ticker
        meta["shortName"] = info.get("shortName") or info.get("symbol") or ticker
        meta["sector"] = info.get("sector")
        meta["industry"] = info.get("industry")
        meta["exchange"] = meta["exchange"] or info.get("exchange")
        if meta["marketCap"] is None:
            meta["marketCap"] = info.get("marketCap")
        meta["longBusinessSummary"] = info.get("longBusinessSummary")
        meta["city"] = info.get("city")
        meta["state"] = info.get("state")
        meta["country"] = info.get("country")
        meta["fullTimeEmployees"] = info.get("fullTimeEmployees")
        meta["website"] = info.get("website")
        meta["founded"] = info.get("founded")
        meta["trailingPE"] = info.get("trailingPE")
        meta["forwardPE"] = info.get("forwardPE")

        ipo_epoch = info.get("firstTradeDateEpochUtc")
        try:
            meta["ipo_year"] = int(pd.to_datetime(ipo_epoch, unit="s").year) if ipo_epoch else None
        except Exception:
            meta["ipo_year"] = None

    except Exception:
        pass

    if not meta["currency"]:
        t = (ticker or "").upper()
        if t.endswith(".MX"):
            meta["currency"] = "MXN"
        elif t.endswith(".SA"):
            meta["currency"] = "BRL"
        elif t.endswith(".TO"):
            meta["currency"] = "CAD"
        elif t.endswith(".L"):
            meta["currency"] = "GBP"
        elif t.endswith(".PA") or t.endswith(".DE") or t.endswith(".F"):
            meta["currency"] = "EUR"
        else:
            meta["currency"] = "USD"

    return meta


def _safe_get_meta(t: str) -> dict:
    try:
        return get_meta(t) or {}
    except Exception:
        return {}


def preload_meta(tickers: List[str], max_workers: int = 6, jitter: bool = True) -> Dict[str, dict]:
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))

    metas_cache: Dict[str, dict] = st.session_state.setdefault("metas_cache", {})
    to_fetch = [t for t in tickers if t not in metas_cache]

    if to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_safe_get_meta, t): t for t in to_fetch}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    metas_cache[t] = fut.result() or {}
                except Exception:
                    metas_cache[t] = {}
                if jitter:
                    time.sleep(0.01 + 0.02 * random.random())

    return {t: metas_cache.get(t, {}) for t in tickers}


# ================== Descarga en lote 5y/1d ==================
@st.cache_data(show_spinner=True, ttl=24 * 3600)
def _download_batch_5y_daily(tickers: Tuple[str, ...]) -> Dict[str, pd.DataFrame]:
    if not YF_OK or not tickers:
        return {}
    raw = yf.download(
        " ".join(tickers),
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    out: Dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for tk in tickers:
            try:
                sub = raw.xs(tk, axis=1, level=-1)
            except Exception:
                try:
                    cols = [c for c in raw.columns if isinstance(c, tuple) and c[-1] == tk]
                    sub = raw.loc[:, cols]
                    sub.columns = [c[0] for c in cols]
                except Exception:
                    sub = pd.DataFrame()
            out[tk] = _normalize_yf_df(sub, tk)
    else:
        tk = tickers[0]
        out[tk] = _normalize_yf_df(raw, tk)
    return out


def preload_hist_5y_daily(all_tickers: List[str], batch_size: int = 20) -> None:
    pool = _hist_pool()
    need: List[str] = []
    for tk in {str(t).strip() for t in all_tickers if t}:
        key = (tk, "5y", "1d")
        if key not in pool:
            need.append(tk)
    if not need:
        return
    for i in range(0, len(need), batch_size):
        chunk = tuple(need[i : i + batch_size])
        data = _download_batch_5y_daily(chunk)
        for tk, df in data.items():
            pool[(tk, "5y", "1d")] = df


# ================== Lecturas rápidas (Parquet primero) ==================
_PREFER_PARQUET = True

def _to_equities_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    rename_map = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "adj_close": "Adj Close", "volume": "Volume",
    }
    if any(col in d.columns for col in rename_map.keys()):
        d = d.rename(columns=rename_map)
    cols = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in d.columns]
    d = d[cols].copy() if cols else d
    try:
        d.index = pd.to_datetime(d.index).tz_localize(None)
    except Exception:
        pass
    return d

def get_hist_5y(ticker: str) -> pd.DataFrame:
    try:
        if _PREFER_PARQUET:
            df_parq = get_history_parquet(ticker, years_if_missing=5)
            if df_parq is not None and not df_parq.empty:
                return _to_equities_schema(df_parq)
    except Exception:
        pass

    pool = _hist_pool()
    key = (ticker, "5y", "1d")
    if key not in pool:
        preload_hist_5y_daily([ticker])
    return pool.get(key, pd.DataFrame())

def get_hist_sliced_years(ticker: str, years: float) -> pd.DataFrame:
    try:
        if _PREFER_PARQUET:
            need_years = max(5, int(round(float(years))))
            full = get_history_parquet(ticker, years_if_missing=need_years)
            if full is not None and not full.empty:
                full = _to_equities_schema(full)
                end = full.index.max()
                if isinstance(end, pd.Timestamp):
                    start = end - pd.DateOffset(years=float(years))
                    return full[full.index >= start].copy()
                return full
    except Exception:
        pass

    df5 = get_hist_5y(ticker)
    if df5 is None or df5.empty:
        return df5
    end = df5.index.max()
    if not isinstance(end, pd.Timestamp):
        return df5
    start = end - pd.DateOffset(years=float(years))
    return df5[df5.index >= start].copy()


# ================== Helper opcional: precarga combinada ==================
def preload_hist_and_meta(tickers: List[str], benchs: Optional[List[str]] = None) -> None:
    all_hist = list({*(tickers or []), *(benchs or [])})
    if all_hist:
        preload_hist_5y_daily(all_hist)
    if tickers:
        preload_meta(tickers)


# ================== Persistencia incremental en Parquet ==================
DATA_DIR = Path("data")
PRICES_DIR = DATA_DIR / "prices"
PRICES_DIR.mkdir(parents=True, exist_ok=True)

try:
    import pyarrow  # asegura que pyarrow esté instalado
    _HAS_PA = True
except Exception:
    _HAS_PA = False


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_index()
    df.to_parquet(path, index=True)

def _as_date(x) -> date:
    return pd.to_datetime(x).date()

def _today_utc() -> date:
    return pd.Timestamp.utcnow().date()

def _download_range_daily(tickers: List[str], start: date, end: date) -> Dict[str, pd.DataFrame]:
    if not YF_OK or not tickers:
        return {}
    raw = yf.download(
        " ".join(tickers),
        start=pd.to_datetime(start).strftime("%Y-%m-%d"),
        end=pd.to_datetime(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    out: Dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for tk in tickers:
            try:
                sub = raw.xs(tk, axis=1, level=-1)
            except Exception:
                try:
                    cols = [c for c in raw.columns if isinstance(c, tuple) and c[-1] == tk]
                    sub = raw.loc[:, cols]
                    sub.columns = [c[0] for c in cols]
                except Exception:
                    sub = pd.DataFrame()
            out[tk] = _normalize_yf_df(sub, tk)
    else:
        if tickers:
            tk = tickers[0]
            out[tk] = _normalize_yf_df(raw, tk)
    return out

def _canon_cols(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Adj close": "adj_close",
        "AdjClose": "adj_close",
        "Volume": "volume",
        "ticker": "ticker",
    }
    rename_map.update({k.lower(): v for k, v in rename_map.items()})
    d.columns = [c if c in rename_map else c for c in d.columns]
    d = d.rename(columns=rename_map)

    if "date" not in d.columns:
        first_col = d.columns[0]
        d = d.rename(columns={first_col: "date"})

    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    d["date"] = d["date"].dt.tz_localize(None)
    d["ticker"] = ticker

    keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    d = d[[c for c in keep if c in d.columns]].copy()

    if d.empty:
        return pd.DataFrame()

    d = d.drop_duplicates(subset=["date"]).sort_values("date").set_index("date")
    return d


# ====== Cache de fundamentales (PE) ======
META_DIR = DATA_DIR / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = META_DIR / "fundamentals.parquet"

def _read_meta_parquet() -> pd.DataFrame:
    if not META_PATH.exists():
        # Schema completo con TODOS los fundamentales
        columns = [
            "ticker",
            # Valoración
            "trailingPE", "forwardPE", "priceToBook",
            # Calidad
            "returnOnEquity", "earningsGrowth", "earningsQuarterlyGrowth",
            # Salud Financiera
            "debtToEquity", "totalDebt", "stockholdersEquity", 
            "totalEquityGrossMinority", "freeCashflow", "marketCap",
            # Metadata
            "sector", "longName", "currency",
            # Control
            "updated_at"
        ]
        df = pd.DataFrame(columns=columns)
        return df.set_index("ticker")
    
    try:
        df = pd.read_parquet(META_PATH)
        if "ticker" in df.columns:
            df = df.set_index("ticker")
        return df
    except Exception:
        return pd.DataFrame(columns=["sector", "longName", "currency", "updated_at"])

def _write_meta_parquet(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    out = df.copy()
    
    # 🔧 FIX: Solo hacer set_index si "ticker" está en columnas (no ya es índice)
    if "ticker" in out.columns:
        out = out.set_index("ticker")
    elif out.index.name != "ticker":
        # Si no tiene ticker ni como columna ni como índice, no podemos guardarlo
        return
    
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(META_PATH)

def _fetch_fundamentals_yf(ticker: str) -> dict:
    """
    Descarga TODOS los fundamentales necesarios para Value Investing.
    Incluye balance sheet y cash flow statement.
    """
    if not YF_OK:
        return {}
    
    try:
        tk = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        
        # Estados financieros
        balance = None
        cash_flow = None
        try:
            if hasattr(tk, 'balance_sheet') and not tk.balance_sheet.empty:
                balance = tk.balance_sheet.iloc[:, 0]
            if hasattr(tk, 'cashflow') and not tk.cashflow.empty:
                cash_flow = tk.cashflow.iloc[:, 0]
        except Exception:
            pass
        
        # Debt-to-Equity manual
        debt_to_equity = None
        total_debt = None
        stockholders_equity = None
        total_equity_gross = None
        
        if balance is not None:
            try:
                total_debt = balance.get('Total Debt')
                stockholders_equity = balance.get('Stockholders Equity')
                total_equity_gross = balance.get('Total Equity Gross Minority Interest')
                
                # Calcular D/E
                patrimonio = stockholders_equity if pd.notna(stockholders_equity) else total_equity_gross
                if pd.notna(total_debt) and pd.notna(patrimonio) and patrimonio != 0:
                    debt_to_equity = float(total_debt) / float(patrimonio)
            except Exception:
                pass
        
        # Free Cash Flow
        free_cashflow = None
        if cash_flow is not None:
            try:
                free_cashflow = cash_flow.get('Free Cash Flow')
            except Exception:
                pass
        
        out = {
            # Valoración
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "priceToBook": info.get("priceToBook"),
            
            # Calidad
            "returnOnEquity": info.get("returnOnEquity"),
            "earningsGrowth": info.get("earningsGrowth"),
            "earningsQuarterlyGrowth": info.get("earningsQuarterlyGrowth"),
            
            # Salud Financiera
            "debtToEquity": debt_to_equity,
            "totalDebt": float(total_debt) if pd.notna(total_debt) else None,
            "stockholdersEquity": float(stockholders_equity) if pd.notna(stockholders_equity) else None,
            "totalEquityGrossMinority": float(total_equity_gross) if pd.notna(total_equity_gross) else None,
            "freeCashflow": float(free_cashflow) if pd.notna(free_cashflow) else None,
            "marketCap": info.get("marketCap"),
            
            # Metadata
            "sector": info.get("sector"),
            "longName": info.get("longName") or info.get("shortName") or ticker,
            "currency": info.get("currency"),
        }
        return out
    except Exception:
        return {}
    
def get_fundamentals_cached(
    ticker: str,
    *,
    max_age_hours: Optional[float] = 168.0,  # 7 días por defecto
    force_refresh: bool = False
) -> dict:
    """
    Lee fundamentales completos del cache.
    Si no existen o están viejos, los descarga.
    
    Args:
        ticker: Símbolo del ticker
        max_age_hours: Máxima edad del cache en horas (default 7 días)
        force_refresh: Forzar descarga aunque estén frescos
    
    Returns:
        Dict con fundamentales o {} si fallan
    """
    df = _read_meta_parquet()
    now = pd.Timestamp.utcnow()
    
    # Verificar si existe y está fresco
    if ticker in df.index and not force_refresh:
        row = df.loc[ticker]
        ts = pd.to_datetime(row.get("updated_at")) if "updated_at" in row and pd.notna(row.get("updated_at")) else None
        
        fresh = False
        if ts is not None and max_age_hours is not None:
            fresh = (now - ts) <= pd.Timedelta(hours=float(max_age_hours))
        elif ts is not None:
            fresh = True
        
        if fresh:
            # Convertir row a dict y limpiar NaNs
            fundamentals = row.to_dict()
            fundamentals = {k: (v if pd.notna(v) else None) for k, v in fundamentals.items()}
            return fundamentals
    
    # Descargar fundamentales frescos
    fundamentals = _fetch_fundamentals_yf(ticker)
    if fundamentals:
        # Actualizar cache
        for key, value in fundamentals.items():
            df.loc[ticker, key] = value
        df.loc[ticker, "updated_at"] = now.isoformat()
        _write_meta_parquet(df)
    
    return fundamentals

def get_meta_cached_light(ticker: str) -> dict:
    """
    Lee nombre/sector/moneda del cache rápido (sin red). Fallback mínimo.
    """
    df = _read_meta_parquet()
    if ticker in df.index:
        row = df.loc[ticker]
        return {
            "longName": row.get("longName") or ticker,
            "sector": row.get("sector") or "—",
            "currency": row.get("currency"),
        }
    return {"longName": ticker, "sector": "—", "currency": None}


# ====== Actualización incremental (con hooks de progreso y PE paralelo) ======
def ensure_parquet_incremental(
    tickers: List[str],
    years: int = 5,
    margin_days: int = 3,
    chunk_size: int = 20,
    sleep_between_chunks: float = 0.4,
) -> None:
    """
    Garantiza que cada ticker tenga histórico diario hasta hoy en Parquet.
    - Si no existe parquet: baja ~years atrás hasta hoy.
    - Si existe: baja desde (última_fecha - margin_days) hasta hoy y mergea.
    """
    if not _HAS_PA:
        return

    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers or not YF_OK:
        return

    today = _today_utc()
    earliest_needed = (pd.Timestamp(today) - pd.DateOffset(years=years)).date()

    plan = []
    for tk in tickers:
        ppath = PRICES_DIR / f"{tk}.parquet"
        cur = _read_parquet(ppath)
        if cur.empty:
            plan.append((tk, earliest_needed, today))
        else:
            last = cur.index.max()
            try:
                last_d = _as_date(last)
            except Exception:
                last_d = earliest_needed
            start_dl = max(earliest_needed, last_d - timedelta(days=margin_days))
            plan.append((tk, start_dl, today))

    for i in range(0, len(plan), chunk_size):
        chunk = plan[i : i + chunk_size]
        tickers_chunk = [t for (t, _, __) in chunk]
        start_min = min(s for (_, s, __) in chunk)
        data = _download_range_daily(tickers_chunk, start=start_min, end=today)

        for (tk, _, __) in chunk:
            new_raw = data.get(tk, pd.DataFrame())
            new_df = _canon_cols(new_raw, tk)
            ppath = PRICES_DIR / f"{tk}.parquet"
            old_df = _read_parquet(ppath)
            if old_df.empty:
                merged = new_df
            else:
                merged = pd.concat([old_df, new_df], axis=0)
                merged = merged[~merged.index.duplicated(keep="last")]
            if not merged.empty:
                cols = ["ticker", "open", "high", "low", "close", "adj_close", "volume"]
                merged = merged[[c for c in cols if c in merged.columns]]
                _write_parquet(merged, ppath)
        time.sleep(sleep_between_chunks)


def ensure_parquet_incremental_verbose(
    tickers: List[str],
    years: int = 5,
    margin_days: int = 3,
    # aliases que usa main.py:
    batch_new: int = 20,
    sleep_between_calls: float = 0.4,
    # controles de frescura/forzado:
    max_age_hours: Optional[float] = None,
    force_refresh: bool = False,
    # NUEVO: bajar PE en paralelo al histórico
    fetch_pe_along: bool = True,
    pe_max_age_hours: float = 48.0,
    # callbacks
    on_log: Optional[Callable[[str, int, str, float], None]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    # nombres nativos (por compatibilidad si alguien llama con estos):
    chunk_size: Optional[int] = None,
    sleep_between_chunks: Optional[float] = None,
) -> Dict[str, int]:
    """
    Igual que ensure_parquet_incremental pero:
      - Acepta alias (batch_new, sleep_between_calls, on_log).
      - Soporta max_age_hours y force_refresh.
      - Emite on_log(ticker, added_rows, phase, took_secs) por ticker/PE.
      - Emite progress_cb(done, total) por chunk.
      - Devuelve {"processed": total, "bootstrap": B, "incremental": I}.
    """
    if not _HAS_PA or not YF_OK:
        if on_log: on_log("—", 0, "skipped_error", 0.0)
        return {"processed": 0, "bootstrap": 0, "incremental": 0}

    # Normaliza parámetros
    chunk_size = int(chunk_size or batch_new or 20)
    sleep_between_chunks = float(sleep_between_chunks or sleep_between_calls or 0.4)

    # Limpia/normaliza tickers
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))
    total = len(tickers)
    if total == 0:
        return {"processed": 0, "bootstrap": 0, "incremental": 0}

    # Si NO es forzado y hay max_age_hours, valida frescura global
    if (max_age_hours is not None) and (not force_refresh):
        now = time.time()
        fresh_all = True
        max_age_sec = float(max_age_hours) * 3600.0
        for tk in tickers:
            p = PRICES_DIR / f"{tk}.parquet"
            if not p.exists():
                fresh_all = False
                break
            try:
                age = now - p.stat().st_mtime
            except Exception:
                fresh_all = False
                break
            if age > max_age_sec:
                fresh_all = False
                break
        if fresh_all:
            # Todo fresco; nada que hacer
            if on_log:
                on_log("—", 0, "fresh_skip", 0.0)
            return {"processed": 0, "bootstrap": 0, "incremental": 0}

    today = _today_utc()
    earliest_needed = (pd.Timestamp(today) - pd.DateOffset(years=years)).date()

    # Plan por ticker: bootstrap (no hay parquet) o incremental (sí hay)
    plan: List[tuple[str, date, date, str]] = []
    bootstrap = incremental = 0
    for tk in tickers:
        ppath = PRICES_DIR / f"{tk}.parquet"
        cur = _read_parquet(ppath)
        if cur.empty:
            plan.append((tk, earliest_needed, today, "bootstrap"))
            bootstrap += 1
        else:
            last = cur.index.max()
            try:
                last_d = _as_date(last)
            except Exception:
                last_d = earliest_needed
            start_dl = max(earliest_needed, last_d - timedelta(days=margin_days))
            plan.append((tk, start_dl, today, "incremental"))
            incremental += 1

    done = 0
    for i in range(0, len(plan), chunk_size):
        chunk = plan[i : i + chunk_size]
        tickers_chunk = [t for (t, _, __, ___) in chunk]
        start_min = min(s for (_, s, __, ___) in chunk)

        t0_chunk = time.time()
        data = _download_range_daily(tickers_chunk, start=start_min, end=today)

        # Dentro del loop, después de guardar precios:
        for (tk, s, e, phase) in chunk:
            t0 = time.time()
            new_raw = data.get(tk, pd.DataFrame())
            new_df = _canon_cols(new_raw, tk)

            ppath = PRICES_DIR / f"{tk}.parquet"
            old_df = _read_parquet(ppath)
            before_rows = 0 if old_df.empty else int(old_df.shape[0])

            if old_df.empty:
                merged = new_df
            else:
                merged = pd.concat([old_df, new_df], axis=0)
                merged = merged[~merged.index.duplicated(keep="last")]

            added_rows = 0
            if not merged.empty:
                cols = ["ticker", "open", "high", "low", "close", "adj_close", "volume"]
                merged = merged[[c for c in cols if c in merged.columns]]
                _write_parquet(merged, ppath)
                after_rows = int(merged.shape[0])
                added_rows = max(0, after_rows - before_rows)

            took = time.time() - t0
            if on_log:
                on_log(tk, added_rows, phase, took)

            # ✅ NUEVO: Actualizar fundamentales en paralelo
            if fetch_pe_along:  # Reutilizar el flag existente
                t1 = time.time()
                fundamentals = get_fundamentals_cached(
                    tk,
                    max_age_hours=pe_max_age_hours,  # Usar el mismo parámetro
                    force_refresh=False
                )
                # Dentro del loop de tickers individuales
                if on_log:
                    has_data = 1 if fundamentals else 0
                    on_log(tk, has_data, "fundamentals", time.time() - t1)

                # ✅ FUERA del loop de tickers, DESPUÉS de procesar el chunk completo
                done += len(chunk)
                if progress_cb:
                    progress_cb(done, total)
        if progress_cb:
            try:
                progress_cb(done, total)
            except Exception:
                pass

        # descanso ligero entre chunks para no saturar
        time.sleep(sleep_between_chunks)

    return {"processed": total, "bootstrap": bootstrap, "incremental": incremental}


def get_history_parquet(
    ticker: str,
    start: Optional[str | pd.Timestamp | date] = None,
    end: Optional[str | pd.Timestamp | date] = None,
    years_if_missing: int = 5,
) -> pd.DataFrame:
    if not _HAS_PA:
        return get_hist_5y(ticker)

    ppath = PRICES_DIR / f"{ticker}.parquet"
    if not ppath.exists():
        ensure_parquet_incremental([ticker], years=years_if_missing)

    df = _read_parquet(ppath)
    if df.empty:
        ensure_parquet_incremental([ticker], years=years_if_missing)
        df = _read_parquet(ppath)

    if df.empty:
        return df

    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
    return df
