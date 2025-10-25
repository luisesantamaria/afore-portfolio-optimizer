# tabs/runtime.py
from __future__ import annotations
from datetime import date, datetime
import streamlit as st

def _today() -> date:
    return datetime.now().date()

# ---- Claves canónicas en session_state ----
OPS_STARTED_AT = "ops_started_at"     # fecha real de inicio de operación
ACT_AS_OF      = "act_as_of"          # fecha Act as of (vista/simulación)
OPS_MODE       = "ops_operating"      # ya la usas, pero la centralizamos aquí también
AAS_TOKEN      = "act_as_of_changed_token"

def ensure_ops_keys() -> None:
    """Asegura claves mínimas. No fija valores si ya existen."""
    ss = st.session_state
    ss.setdefault(OPS_MODE, False)
    # No fijamos ops_started_at hasta que inicies operación
    if ACT_AS_OF not in ss:
        ss[ACT_AS_OF] = _today()
    ss.setdefault(AAS_TOKEN, 0)

def ops_is_active() -> bool:
    return bool(st.session_state.get(OPS_MODE, False))

def get_ops_started_at() -> date | None:
    return st.session_state.get(OPS_STARTED_AT, None)

def set_ops_started_at(d: date) -> None:
    st.session_state[OPS_STARTED_AT] = d

def get_act_as_of() -> date:
    """Devuelve la fecha Act as of vigente (si no hay, devuelve today)."""
    return st.session_state.get(ACT_AS_OF) or _today()

def set_act_as_of(d: date) -> None:
    """Actualiza la fecha Act as of y sube un token para invalidar cachés."""
    st.session_state[ACT_AS_OF] = d
    st.session_state[AAS_TOKEN] = int(st.session_state.get(AAS_TOKEN, 0)) + 1
