from __future__ import annotations
import uuid
from typing import List, Dict, Any, Optional
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Helpers
# =========================
def _mx_risk_free_pct_default() -> float:
    """
    Toma la tasa libre (CETES 364d) desde lo ya cargado en la tab de Renta Fija,
    sin volver a llamar a Banxico.
    Prioridad:
      1) Buscar en fi_rows un CETES 364d (Benchmark o Rend %)
      2) Si el benchmark activo es CETES 364d, usar fi_bench_rate
      3) mx_risk_free_pct en sesión
      4) Fallback 7.70
    """
    try:
        rows = st.session_state.get("fi_rows", []) or []
        for r in rows:
            bono = str(r.get("Bono", "")).upper()
            tipo = str(r.get("Tipo de bono", "")).upper()
            if "CETES 364D" in bono or (tipo == "BONO GUBERNAMENTAL" and "364" in bono):
                v = r.get("Benchmark (%)")
                if v is None:
                    v = r.get("Rend (%)")
                if v is not None:
                    v = float(v)
                    if np.isfinite(v) and v >= 0:
                        return v
    except Exception:
        pass

    try:
        lbl = str(st.session_state.get("fi_bench_label", "") or "")
        rate = st.session_state.get("fi_bench_rate", None)
        if rate is not None and ("CETES 364" in lbl.upper()):
            v = float(rate)
            if np.isfinite(v) and v >= 0:
                return v
    except Exception:
        pass

    try:
        v = float(st.session_state.get("mx_risk_free_pct", 7.70))
        if np.isfinite(v) and v >= 0:
            return v
    except Exception:
        pass

    return 7.70

def _min_safe_float(x, default):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default

# Heurística suave de volatilidad si el usuario no la cambia:
# - CERPI: típicamente proyectos/PE → algo mayor
# - CKD: depende, pero sin sector/etapa, usamos un valor prudente menor
DEFAULT_SIGMA = {
    "CERPI": 0.14,  # 14% anual (editable)
    "CKD":   0.10,  # 10% anual (editable)
}

CALL_FREQ = ["A demanda", "Anual", "Semestral", "Trimestral"]

# =========================
# Render principal
# =========================
def render():

    # --------- Captura mínima ----------
    top1, top2, top3, top4 = st.columns(4, gap="small")
    with top1:
        tipo = st.selectbox("Tipo", options=["CERPI", "CKD"], key="cer_ckd_tipo")
    with top2:
        vehiculo = st.text_input("Nombre del vehículo", key="cer_ckd_nombre", placeholder="p.ej., Fondo Infra I")
    with top3:
        plazo_anios = st.number_input("Plazo (años)", min_value=1, value=8, step=1, key="cer_ckd_plazo")
    with top4:
        freq = st.selectbox("Frecuencia de llamadas", options=CALL_FREQ, index=1, key="cer_ckd_freq")
        # Nota: No pedimos montos de llamadas; mantenemos metadato ligero.

    mid1, mid2, mid3 = st.columns(3, gap="small")
    with mid1:
        # Fechas: opcionales; si las dejan vacías, se almacenan como cadena vacía
        emision = st.date_input("Fecha de emisión", key="cer_ckd_emision", value=None, format="YYYY-MM-DD")
    with mid2:
        venc = st.date_input("Fecha de vencimiento", key="cer_ckd_venc", value=None, format="YYYY-MM-DD")
    with mid3:
        amort = st.selectbox("Amortización", options=["Bullet", "Amortizable"], index=0, key="cer_ckd_amort")

    low1, low2, low3, low4 = st.columns(4, gap="small")
    with low1:
        tir_pct = st.number_input("TIR esperada (%)", min_value=0.0, value=12.0, step=0.10, format="%.2f", key="cer_ckd_tir")
    with low2:
        # Volatilidad editable; default por tipo
        sigma_default = DEFAULT_SIGMA.get(tipo, 0.12)*100.0
        vol_pct = st.number_input("Volatilidad estimada (%)", min_value=0.01, value=float(sigma_default),
                                  step=0.10, format="%.2f", key="cer_ckd_vol")
    with low3:
        r_mx = _mx_risk_free_pct_default()  # solo para referencia en UI; Sharpe = μ/σ (como tu optimizador)
        st.metric("Tasa libre (ref.)", f"{r_mx:.2f}%")
    with low4:
        # Sharpe ≈ μ/σ (consistente con el optimizador)
        mu = _min_safe_float(tir_pct, 0.0) / 100.0
        sigma = max(1e-6, _min_safe_float(vol_pct, sigma_default) / 100.0)
        sharpe = mu / sigma if sigma > 0 else 0.0
        st.metric("Sharpe (aprox)", f"{sharpe:.2f}")

    st.caption("Nota: trabajamos **por porcentaje**. El optimizador asignará el peso óptimo; no se piden montos.")

    # --------- Agregar a la tabla en memoria (sin guardar archivo) ----------
    add_cols = st.columns([1, 4])
    with add_cols[0]:
        if st.button("➕ Agregar vehículo", type="primary", key="cer_ckd_add"):
            st.session_state.setdefault("cerpi_ckd_rows", [])
            row = {
                "row_id": uuid.uuid4().hex,
                "Vehículo": vehiculo.strip() if vehiculo else "",
                "Tipo": tipo,
                "TIR (%)": round(float(tir_pct), 4),
                "Vol (%)": round(float(vol_pct), 4),
                "Sharpe": round(float(sharpe), 4),
                "Plazo (años)": int(plazo_anios),
                "Frecuencia llamadas": freq,
                "Amortización": amort,
                "Emisión": (emision.strftime("%Y-%m-%d") if isinstance(emision, date) else ""),
                "Vencimiento": (venc.strftime("%Y-%m-%d") if isinstance(venc, date) else ""),
                "❌": False,
            }
            st.session_state["cerpi_ckd_rows"].append(row)
            st.toast("Vehículo agregado (no guardado aún).", icon="✅")

    st.divider()

    # --------- Tabla editable + eliminación por checkbox ----------
    rows: List[Dict[str, Any]] = st.session_state.get("cerpi_ckd_rows", [])
    st.caption(f"{len(rows)} vehículo(s)")

    if not rows:
        st.info("No hay CERPI/CKD capturados. Agrega uno con el botón de arriba.")
        return

    df = pd.DataFrame(rows)
    # Orden y configuración de columnas visibles
    visible_cols = [
        "Vehículo", "Tipo", "TIR (%)", "Vol (%)", "Sharpe",
        "Plazo (años)", "Frecuencia llamadas", "Amortización",
        "Emisión", "Vencimiento", "❌",
    ]
    for c in visible_cols:
        if c not in df.columns:
            df[c] = "" if c not in ("❌",) else False

    edited = st.data_editor(
        df[visible_cols],
        hide_index=True,
        width='stretch',
        height=int(42 + 34 * max(1, len(df)) + 10),
        column_config={
            "TIR (%)": st.column_config.NumberColumn(format="%.2f"),
            "Vol (%)": st.column_config.NumberColumn(format="%.2f"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Plazo (años)": st.column_config.NumberColumn(format="%d"),
            "❌": st.column_config.CheckboxColumn(help="Marcar para eliminar esta fila", default=False, width="small"),
        },
        key="cer_ckd_editor",
    )

    # Eliminar marcados
    to_delete_idx = edited.index[edited["❌"] == True].tolist()
    if to_delete_idx:
        # Mapear por contenido (Vehículo + Tipo + Emisión + Vencimiento) para robustez
        def _row_sig(r):
            return (r.get("Vehículo",""), r.get("Tipo",""), r.get("Emisión",""), r.get("Vencimiento",""))
        del_sigs = {_row_sig(edited.loc[i].to_dict()) for i in to_delete_idx}
        st.session_state["cerpi_ckd_rows"] = [r for r in st.session_state["cerpi_ckd_rows"] if _row_sig(r) not in del_sigs]
        st.toast(f"Eliminado(s) {len(to_delete_idx)} vehículo(s).", icon="🗑️")
        st.rerun()

    # Aplicar ediciones (en memoria)
    new_rows = edited.to_dict(orient="records")
    # Re-sincronizar con session_state (buscando por firma flexible)
    def _sig(x): return (x.get("Vehículo",""), x.get("Tipo",""), x.get("Emisión",""), x.get("Vencimiento",""))
    old_map = {_sig(r): r for r in st.session_state["cerpi_ckd_rows"]}
    updated_list = []
    for r in new_rows:
        sig = _sig(r)
        base = old_map.get(sig, None)
        if base is None:
            # fila nueva desde editor (poco probable)
            r["row_id"] = uuid.uuid4().hex
            r["❌"] = False
            updated_list.append(r)
        else:
            # conserva row_id y ❌
            r2 = {**base, **r}
            r2["row_id"] = base.get("row_id", uuid.uuid4().hex)
            updated_list.append(r2)
    st.session_state["cerpi_ckd_rows"] = updated_list
