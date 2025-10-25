# tabs/consar_limits.py
from __future__ import annotations
import json
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import streamlit as st
import pandas as pd
import re
import altair as alt
import locale
import plotly.graph_objects as go


# Configurar locale para el formato de fecha en español (manteniendo solo LC_TIME)
try:
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
except locale.Error:
    try:
        # Alternativa para sistemas que no soportan 'es_ES.UTF-8'
        locale.setlocale(locale.LC_TIME, 'es_MX.UTF-8')
    except locale.Error:
        # Fallback si no se puede configurar el español
        pass


# =========================
# Helper general
# =========================
def _clone(obj: Any) -> Any:
    """Hace una copia profunda usando JSON (para evitar referencias mutables)."""
    return json.loads(json.dumps(obj))


# =========================
# Carga y parsing del CSV (Límites CONSAR)
# =========================
@st.cache_data(show_spinner=False)
def _load_limits_from_csv() -> pd.DataFrame:
    """
    Lee data/limites_consar.csv y devuelve un DataFrame limpio.
    """
    base = Path(".").resolve()
    csv_path = base / "data" / "limites_consar.csv"
    df = pd.read_csv(csv_path)

    # Limpieza básica de encabezados y celdas
    df.columns = [str(c).strip() for c in df.columns]
    if "75-59" in df.columns:
        # Tolerancia al posible typo
        df.rename(columns={"75-59": "75-79"}, inplace=True)

    df["Categoría"] = df["Categoría"].astype(str).str.strip()

    # Normalizar valores numéricos (pueden venir con espacios)
    num_cols = [c for c in df.columns if c != "Categoría"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")

    return df

SELECTED_COLOR = "#ff4b4b"

# =========================
# Mapeos y catálogo
# =========================
SIEFORE_COL_TO_FULL = {
    "95-99": "Básica 95-99",
    "90-94": "Básica 90-94",
    "85-89": "Básica 85-89",
    "80-84": "Básica 80-84",
    "75-79": "Básica 75-79",
    "70-74": "Básica 70-74",
    "65-69": "Básica 65-69",
    "60-64": "Básica 60-64",
}

# Grupos para separar secciones
CLASS_CATS = {
    "Valores Extranjeros",
    "Renta Variable",
    "Instrumentos en Divisas",
    "Instrumentos Bursatilizados",
    "Instrumentos Estructurados",
    "FIBRAS / REITs",
    "Mercancías",
}

CONCENTRATION_CATS = {
    "Gobierno Federal (soberano MX)",
    "Deuda EPE (mxBBB a mxAAA)",
    "Deuda corporativa (mxBBB a mxAAA)",
    "Deuda subordinada",
    "Deuda híbridos",
    "Fibras / inmobiliarios",
    "Estructurados (CKD/CERPIs)",
    "Un solo emisor extranjero (BBB- a AAA)",
}

RISK_CATS = {
    "VaR",
    "Diferencial del Valor en Riesgo Condicional",
    "Coeficiente de liquidez",
    "Error de Seguimiento / Tracking Error",
}

def _pct(x: float | None) -> float:
    """Convierte decimal (p.ej. 0.5977) a porcentaje (59.77)."""
    if pd.isna(x):
        return float("nan")
    return float(x) * 100.0

def _display_name_for_risk(raw: str) -> str:
    """Ajustes de rótulos solo para Riesgo."""
    mapping = {
        "VaR": "VaR (%)",
        "Diferencial del Valor en Riesgo Condicional": "CVaR / Valor en Riesgo Condicional (%)",
        "Coeficiente de liquidez": "Coeficiente de liquidez (%)",
        "Error de Seguimiento / Tracking Error": "Error de seguimiento / tracking error (%)",
    }
    return mapping.get(raw, raw)


@st.cache_data(show_spinner=False)
def _build_catalog_from_csv() -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Construye un catálogo de límites a partir del CSV.
    """
    df = _load_limits_from_csv()

    # Columnas SIEFORE en el MISMO orden del CSV
    siefore_cols_present: List[str] = [c for c in df.columns if c in SIEFORE_COL_TO_FULL]
    siefores_full: List[str] = [SIEFORE_COL_TO_FULL[c] for c in siefore_cols_present]

    catalog = {
        "siefores_full": siefores_full,
        "classes": {},
        "concentration": {},
        "risk": {},
    }

    # Construimos por SIEFORE manteniendo ORDEN DE FILAS del CSV
    for col in siefore_cols_present:
        siefore_name = SIEFORE_COL_TO_FULL[col]
        classes_map: Dict[str, Dict[str, float]] = {}
        concentration_map: Dict[str, Dict[str, float]] = {}
        risk_map: Dict[str, float] = {}

        for _, row in df.iterrows():
            cat = row["Categoría"]
            val = row[col]

            if cat in CLASS_CATS:
                classes_map[cat] = {"max_pct": _pct(val)}
            elif cat in CONCENTRATION_CATS:
                concentration_map[cat] = {"max_pct": _pct(val)}
            elif cat in RISK_CATS:
                risk_map[_display_name_for_risk(cat)] = _pct(val)
            # Categorías no mapeadas se ignoran silenciosamente

        catalog["classes"][siefore_name] = classes_map
        catalog["concentration"][siefore_name] = concentration_map
        catalog["risk"][siefore_name] = risk_map

    return catalog


# =========================
# Inicialización de estado (solo selecciones)
# =========================
def _init_state():
    catalog = _build_catalog_from_csv()
    
    # El catálogo de límites base se guarda una vez
    st.session_state.setdefault("consar_limits_catalog", _clone(catalog))

    siefores_full: List[str] = st.session_state["consar_limits_catalog"]["siefores_full"]
    default_sel = siefores_full[0] if siefores_full else "Básica 95-99"
    
    # 1. Persistir la SIEFORE seleccionada (siefore_selected)
    st.session_state.setdefault("siefore_selected", default_sel)
    
    # 2. Persistir la AFORE seleccionada (afore_selected)
    st.session_state.setdefault("afore_selected", None)
    
    # 3. Mantenemos consar_limits_current solo como un placeholder, no persiste
    st.session_state.setdefault("consar_limits_current", {})


# =========================
# Helpers para DataFrames
# =========================
def _classes_to_df(d: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for cat, cfg in d.items():
        if isinstance(cfg, dict) and "max_pct" in cfg:
            val = cfg["max_pct"]
        else:
            val = float(cfg) if pd.notna(cfg) else float("nan")
        rows.append({"Categoría": cat, "Límite (%)": val})
    return pd.DataFrame(rows)

def _concentration_to_df(d: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for cat, cfg in d.items():
        if isinstance(cfg, dict) and "max_pct" in cfg:
            val = cfg["max_pct"]
        else:
            val = float(cfg) if pd.notna(cfg) else float("nan")
        rows.append({"Categoría": cat, "Límite (%)": val})
    return pd.DataFrame(rows)

def _risk_to_df(d: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{"Límite": k, "Valor (%)": v} for k, v in d.items()])


# =========================
# Función Principal para obtener límites (ÚTIL PARA OTROS TABS)
# =========================
def get_current_limits_all(siefore_name: str | None = None) -> Dict[str, Any]:
    """
    Obtiene los límites de la CONSAR para la SIEFORE seleccionada.
    """
    catalog = st.session_state.get("consar_limits_catalog")
    if not catalog:
        return {"classes": {}, "concentration": {}, "risk": {}}
    
    if siefore_name is None:
        # Lee la SIEFORE seleccionada del estado de sesión
        siefore_name = st.session_state.get("siefore_selected") 
        if siefore_name is None:
            return {"classes": {}, "concentration": {}, "risk": {}}
            
    # Devuelve los límites clonados (genera los límites dinámicamente)
    limits = {
        "classes": _clone(catalog["classes"].get(siefore_name, {})),
        "concentration": _clone(catalog["concentration"].get(siefore_name, {})),
        "risk": _clone(catalog["risk"].get(siefore_name, {})),
    }
    return limits


# =================================================================
# NUEVAS FUNCIONES PARA COMPOSICIÓN DE SIEFORES (LLAMADAS DESDE OTRAS TABS)
# =================================================================
@st.cache_data(show_spinner=False)
def load_siefore_composition() -> pd.DataFrame:
    """
    Carga el archivo de composición de SIEFORES.
    """
    base = Path(".").resolve()
    comp_path = base / "data" / "composicion_siefores.csv"
    
    try:
        df_comp = pd.read_csv(comp_path)
        
        # Lógica de limpieza y pre-procesamiento básica:
        df_comp.columns = [str(c).strip() for c in df_comp.columns]
        if df_comp.columns.size > 0:
             # Asumimos que la primera columna es la categoría y es de tipo string
             df_comp.iloc[:, 0] = df_comp.iloc[:, 0].astype(str).str.strip()
        
        return df_comp
    except FileNotFoundError:
        st.error(f"Error: Archivo de composición no encontrado en {comp_path}")
        return pd.DataFrame() # Devuelve DataFrame vacío si hay error.


# Mapeo de Agrupaciones de la tabla CONSAR (izquierda) a Categorías del CSV de composición (derecha).
COMPOSITION_MAPPING = {
    "Renta Variable": ["Renta Variable Nacional", "Renta Variable Internacional"],
    "Valores Extranjeros": ["Renta Variable Internacional", "Deuda Internacional"],
    "FIBRAS": ["FIBRAS"],
    "Deuda Gubernamental": ["Deuda Gubernamental"],
    "Deuda Privada Nacional": ["Deuda Privada Nacional"],
    "Divisas": ["Otros Activos"],
    "Estructurados": ["Estructurados"],
    "Mercancías": ["Mercancias"],
}


def get_siefore_composition_data(siefore_name: str, afore_name: str) -> pd.DataFrame:
    """
    Carga el DataFrame de composición y extrae las columnas de promedio y AFORE/SIEFORE
    para la SIEFORE actual, aplicando el COMPOSITION_MAPPING.
    """
    df_comp = load_siefore_composition()
    if df_comp.empty or "Fallback" in siefore_name:
        return pd.DataFrame()

    cohort = siefore_name.split()[-1]
    col_promedio = f"Promedio {cohort}"
    
    afore_search_name = afore_name.strip()
    col_afore = None
    
    for col in df_comp.columns:
        if col.strip().startswith(afore_search_name) and col.strip().endswith(cohort):
            col_afore = col.strip()
            break
    
    if col_promedio not in df_comp.columns or col_afore is None:
        return pd.DataFrame()
    
    df_comp.set_index(df_comp.columns[0], inplace=True)
    
    rows_out = []
    
    def safe_numeric_value(df: pd.DataFrame, cat: str, col: str) -> float:
        try:
            val = df.loc[cat, col]
            num_val = pd.to_numeric(val, errors='coerce')
            return num_val if pd.notna(num_val) else 0.0
        except KeyError:
            return 0.0
        except Exception:
            return 0.0
    
    for agg_name, comp_cats in COMPOSITION_MAPPING.items():
        promedio_sum = 0.0
        afore_sum = 0.0
        for cat in comp_cats:
            promedio_sum += safe_numeric_value(df_comp, cat, col_promedio)
            afore_sum += safe_numeric_value(df_comp, cat, col_afore)
                
        rows_out.append({
            "Agrupación": agg_name,
            "Promedio": float(promedio_sum),
            "Especifica": float(afore_sum)
        })

    return pd.DataFrame(rows_out)

def get_commission_value(siefore_name: str, afore_name: str) -> Optional[float]:
    """
    Lee la fila 'Comisiones' del CSV de composición y obtiene la comisión (%) para la columna
    de la AFORE y cohorte seleccionados. Si el valor está en [0,1], lo convierte a %.
    """
    df_comp = load_siefore_composition()
    if df_comp.empty or not siefore_name or not afore_name:
        return None

    cohort = siefore_name.split()[-1].strip()

    cat_col = df_comp.columns[0]
    df_comp[cat_col] = df_comp[cat_col].astype(str).str.strip()

    target_col = None
    for col in df_comp.columns:
        col_str = str(col).strip()
        if col_str.startswith(afore_name.strip()) and col_str.endswith(cohort):
            target_col = col
            break
    if target_col is None:
        exact = f"{afore_name} {cohort}"
        if exact in df_comp.columns:
            target_col = exact
        else:
            return None

    df_idx = df_comp.set_index(cat_col)
    index_map = {str(i).strip().lower(): i for i in df_idx.index}
    key = "comisiones"
    if key not in index_map:
        return None

    raw_val = pd.to_numeric(df_idx.loc[index_map[key], target_col], errors="coerce")
    if pd.isna(raw_val):
        return None

    val = float(raw_val)
    return val



# =================================================================
# NUEVA FUNCIÓN: Obtener Activos Netos del CSV centralizado
# =================================================================
def get_latest_net_asset_value(siefore_name: str, afore_name: str) -> tuple[float | None, pd.Timestamp | None]:
    """
    Busca el último valor de Activos Netos **en pesos** y la fecha para una AFORE/SIEFORE dada.
    (El CSV ya viene en pesos; no se hace conversión a MDP).
    """  # CHANGED: docstring actualizado
    try:
        df_a, _ = _si_load_data()
        if df_a.empty:
            return None, None
    except Exception:
        return None, None

    cohort = siefore_name.split()[-1] if siefore_name else None
    if not cohort or not afore_name:
        return None, None
        
    col_name = f"{afore_name} {cohort}"
    if col_name not in df_a.columns:
        return None, None

    s = df_a[["Fecha", col_name]].dropna(subset=[col_name]).sort_values("Fecha", ascending=True)
    if s.empty:
        return None, None
        
    latest_row = s.iloc[-1]
    valor_pesos = float(latest_row[col_name])   # CHANGED: ya está en pesos
    fecha = pd.to_datetime(latest_row["Fecha"])

    return valor_pesos, fecha


# =========================
# ------- INSPECTOR SIEFORE/AFORE (Datos externos) -------
# =========================
@st.cache_data(show_spinner=False)
def _si_load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(".").resolve()
    activos_path = base / "data" / "activos_netos_siefores.csv"
    rend_path = base / "data" / "rendimientos_siefores.csv"

    # Activos
    df_a = pd.read_csv(activos_path)
    df_a["Fecha"] = pd.to_datetime(df_a["Fecha"], dayfirst=True, errors="coerce")

    # Rendimientos
    df_r = pd.read_csv(rend_path)
    df_r["Fecha"] = pd.to_datetime(df_r["Fecha"], dayfirst=True, errors="coerce")
    return df_a, df_r

def _si_cohorts_from_columns(cols: List[str]) -> List[str]:
    pat = re.compile(r"\s(\d{2}-\d{2})$")
    cohorts = set()
    for c in cols:
        m = pat.search(str(c))
        if m:
            cohorts.add(m.group(1))
    def keyfn(x: str) -> int:
        try:
            return int(x.split("-")[0])
        except Exception:
            return 0
    return sorted(cohorts, key=keyfn)

def _si_afores_for_cohort(cols: List[str], cohort: str) -> List[str]:
    suffix = f" {cohort}"
    afores = []
    for c in cols:
        c = str(c)
        if c.endswith(suffix):
            base = c[: -len(suffix)].strip()
            if base not in ("Total", "Promedio"):
                afores.append(base)
    return sorted(set(afores))

def _si_years_available_for_series(df: pd.DataFrame, col: str) -> List[int]:
    s = df.set_index("Fecha")[col].dropna()
    return sorted(s.index.year.unique().tolist())

# =================================================================
# FORMATO DE PESOS
# =================================================================
def _si_fmt_number(x):
    """
    Formatea un número grande a formato de moneda estándar:
    $9,238,882,310.00
    """
    if pd.isna(x):
        return x
    formatted_number = "{:,.2f}".format(float(x))
    return f"${formatted_number}"
# =================================================================


def render():
    import plotly.express as px  # sólo por si lo usas en otro lado
    _init_state()

    catalog = st.session_state["consar_limits_catalog"]
    siefores_full: List[str] = catalog["siefores_full"]

    # ======== LAYOUT PRINCIPAL: 2 COLUMNAS ========
    left, right = st.columns([1.0, 1.0], gap="large")

    # =======================
    # SELECTORES (ARRIBA IZQUIERDA)
    # =======================
    with left:
        sel_cols = st.columns([1, 1])
        # --- SIEFORE ---
        selected_siefore_name = st.session_state["siefore_selected"]
        idx_default = siefores_full.index(selected_siefore_name) if selected_siefore_name in siefores_full else 0
        sel_cols[0].selectbox(
            "SIEFORE",
            options=siefores_full,
            index=idx_default,
            key="siefore_selected",
        )
        sel_name_from_state = st.session_state["siefore_selected"]   # p.ej. "Básica 95-99"
        cohort = sel_name_from_state.split()[-1] if sel_name_from_state else "95-99"

        # --- Datos externos para AFORE ---
        try:
            df_a, df_r = _si_load_data()
        except Exception as e:
            st.warning(f"No pude cargar los CSV de SIEFORES en ./data. Detalle: {e}")
            df_a = df_r = None

        # AFORES disponibles (intersección de ambos CSV para el cohort)
        afores = []
        if df_a is not None and df_r is not None:
            afores_a = _si_afores_for_cohort(df_a.columns.tolist(), cohort)
            afores_r = _si_afores_for_cohort(df_r.columns.tolist(), cohort)
            afores = [a for a in afores_r if a in afores_a] or sorted(set(afores_a + afores_r))

        if not afores:
            sel_cols[1].info("Carga los CSV de activos y rendimientos.")
            selected_afore = None
        else:
            default_afore = st.session_state.get("afore_selected")
            if default_afore not in afores:
                default_afore = afores[0]
                st.session_state["afore_selected"] = default_afore
            selected_afore = sel_cols[1].selectbox(
                "AFORE",
                afores,
                index=afores.index(default_afore),
                key="afore_selected",
            )
        
        if selected_afore:
            commission = get_commission_value(sel_name_from_state, selected_afore)
            if commission is not None:
                st.caption(
                    f"Comisión de la SIEFORE {sel_name_from_state} de {selected_afore}: **{commission:.2f}%**"
                )
            else:
                st.caption("**Comisión**: dato no disponible para esta combinación.")

    # =======================
    # CONTENIDO: IZQUIERDA (TABLA)
    # =======================
    if df_a is not None and df_r is not None and selected_afore:
        # --- Tabla combinada ---
        col_act = f"{selected_afore} {cohort}"
        # CHANGED: el CSV ya viene en **pesos**, NO multiplicamos por 1e6
        df_act_raw = (
            df_a.loc[:, ["Fecha", col_act]]
              .dropna()
              .sort_values("Fecha", ascending=True)
              .reset_index(drop=True)
              .rename(columns={col_act: "Activos (Pesos)"})   # CHANGED: etiqueta en Pesos
        )

        # Rendimientos
        col_rend = f"{selected_afore} {cohort}"
        df_r_sel = (
            df_r.loc[:, ["Fecha", col_rend]]
                .dropna()
                .sort_values("Fecha", ascending=True)
                .reset_index(drop=True)
                .rename(columns={col_rend: "Rendimiento (%)"})
        )

        # Sólo fechas con ambos datos
        df_merge = pd.merge(df_r_sel, df_act_raw, on="Fecha", how="inner")
        # Fecha bonita (Mes Capitalizado)
        df_merge["FechaStr"] = df_merge["Fecha"].dt.strftime("%B %Y").str.capitalize()

        # Renombrar columnas para la tabla
        df_tabla = (
            df_merge[["FechaStr", "Rendimiento (%)", "Activos (Pesos)"]]
            .rename(columns={
                "FechaStr": "Fecha",
                "Rendimiento (%)": "Rendimiento",
                "Activos (Pesos)": "Activos Netos"
            })
        )

        # Formato de columnas
        df_tabla_fmt = df_tabla.copy()
        df_tabla_fmt["Rendimiento"] = df_tabla_fmt["Rendimiento"].map(lambda x: f"{x:,.2f}%")
        df_tabla_fmt["Activos Netos"] = df_tabla_fmt["Activos Netos"].map(lambda x: f"${x:,.2f}")

        with left:
            st.markdown(f"### Performance SIEFORE {sel_name_from_state} de {selected_afore}")
            st.data_editor(
                df_tabla_fmt,
                hide_index=True,
                use_container_width=True,
                disabled=True,
                height=528,
                column_config={
                    "Fecha": st.column_config.TextColumn(),
                    "Rendimiento (%)": st.column_config.TextColumn(),
                    "Activos (Pesos)": st.column_config.TextColumn(),
                },
            )

        # =======================
        # CONTENIDO: DERECHA (GRÁFICAS)
        # =======================
        with right:
            # ----- Línea de rendimientos (Altair) con título -----
            cols_r = [f"{a} {cohort}" for a in afores if f"{a} {cohort}" in df_r.columns]
            df_r_all = (
                df_r.loc[:, ["Fecha"] + cols_r]
                    .sort_values("Fecha")
                    .reset_index(drop=True)
            )
            df_long = df_r_all.melt("Fecha", var_name="Serie", value_name="Rendimiento").dropna()
            df_long["AFORE"] = df_long["Serie"].str.replace(r"\s\d{2}-\d{2}$", "", regex=True)

            afores_list = df_long["AFORE"].unique().tolist()
            afores_others = [a for a in afores_list if a != selected_afore]
            color_domain = [selected_afore] + afores_others
            other_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            color_range = [SELECTED_COLOR] + other_colors[:len(afores_others)]

            LINE_HEIGHT = 330
            BAR_HEIGHT  = 330

            line_chart = alt.Chart(df_long).mark_line(point=False).encode(
                x=alt.X("Fecha:T", title=None),
                y=alt.Y("Rendimiento:Q", title="Rendimiento (%)", scale=alt.Scale(zero=False)),
                color=alt.Color("AFORE:N", title="AFORE", scale=alt.Scale(domain=color_domain, range=color_range)),
                opacity=alt.condition(alt.datum.AFORE == selected_afore, alt.value(1.0), alt.value(0.35)),
                strokeWidth=alt.condition(alt.datum.AFORE == selected_afore, alt.value(3), alt.value(1)),
                tooltip=[
                    alt.Tooltip("Fecha:T", title=None),
                    alt.Tooltip("AFORE:N"),
                    alt.Tooltip("Rendimiento:Q", title="Rendimiento (%)", format=".2f"),
                ],
            ).properties(
                height=LINE_HEIGHT,
                title=f"Rendimientos de las AFOREs en la SIEFORE {sel_name_from_state}"
            ).configure_title(fontSize=16, anchor="start")

            st.altair_chart(line_chart, use_container_width=True)

            # --- Barras de activos (último periodo) con la AFORE seleccionada resaltada ---
            df_last = (
                df_a[["Fecha"] + [f"{a} {cohort}" for a in afores if f"{a} {cohort}" in df_a.columns]]
                .dropna(how="all", subset=[f"{a} {cohort}" for a in afores if f"{a} {cohort}" in df_a.columns])
            )

            if not df_last.empty:
                last_date = df_last["Fecha"].max()
                row_last = df_last[df_last["Fecha"] == last_date].iloc[0]
                barras = []
                for a in afores:
                    col = f"{a} {cohort}"
                    if col in row_last.index and pd.notna(row_last[col]):
                        # ⚠️ convertir a MDP dividiendo entre 1,000,000
                        barras.append({"AFORE": a, "Activos_MDP": float(row_last[col]) / 1_000_000.0})
                df_barras = pd.DataFrame(barras).sort_values("Activos_MDP", ascending=False)

                fig_bar = go.Figure()

                # Traza base
                fig_bar.add_bar(
                    x=df_barras["AFORE"].tolist(),
                    y=df_barras["Activos_MDP"].tolist(),
                    hovertemplate="$%{y:,.2f} MDP",   # tooltip en MDP con 2 decimales
                    name="Activos",
                    showlegend=False,
                )

                # Resaltado AFORE seleccionada
                if selected_afore in df_barras["AFORE"].values:
                    mask = df_barras["AFORE"] == selected_afore
                    fig_bar.add_bar(
                        x=df_barras.loc[mask, "AFORE"].tolist(),
                        y=df_barras.loc[mask, "Activos_MDP"].tolist(),
                        hovertemplate="$%{y:,.2f} MDP",
                        marker_color=SELECTED_COLOR,
                        name=selected_afore,
                        showlegend=False,
                    )

                fig_bar.update_layout(
                    barmode="overlay",
                    title=f"Activos de las AFOREs en la SIEFORE {sel_name_from_state}",
                    height=BAR_HEIGHT,
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis=dict(title=None),
                    yaxis=dict(
                        title="Activos (MDP)",      # eje a MDP
                        tickformat=",.0f",          # 2 decimales
                        tickprefix="$"              # prefijo $
                    ),
                    showlegend=False,
                )

                st.plotly_chart(fig_bar, use_container_width=True)

