# set_page_config.py
from __future__ import annotations
from typing import Optional, List
import time
import streamlit as st

# ╔═════════════════════════════════════════════════════════════════════════╗
# ║               UI: Spinner circular bonito, compartible                  ║
# ╚═════════════════════════════════════════════════════════════════════════╝
class CircleStatus:
    """
    Anillo de progreso con título/subtítulo y lista de eventos.
    — Comparte una sola instancia entre pasos: crea fuera y pásala como 'cs'.
    — Llama a cs.close() al final para removerlo del DOM.

    Ejemplo:
        cs = CircleStatus("Preparando y optimizando", total_steps=11)
        cs.begin("Cargando…")
        rows = _cosechar_activos(cs=cs)
        out, tot, risk, w = _optimizar(rows, caps, rlim, cs=cs)
        cs.done("Listo"); cs.close()
    """
    def __init__(self, title: str, total_steps: int = 8, mount: Optional[st.delta_generator.DeltaGenerator]=None):
        self.title = title
        self.total = max(int(total_steps), 1)
        self.step_i = 0
        self.t0 = time.perf_counter()
        self._host = mount or st.container()
        self._block = self._host.empty()
        self._log: List[str] = []
        self._state: str = "running"
        self._render(subtitle="Inicializando…")

    def add_steps(self, n: int):
        self.total = max(1, self.total + int(n))

    def _pct(self) -> int:
        return min(100, int(self.step_i * 100 / self.total))

    def _render(self, subtitle: str = ""):
        pct = self._pct()
        ms = int((time.perf_counter() - self.t0) * 1000)
        state = self._state
        if state == "running":
            ring = f"conic-gradient(var(--pri) {pct}%, #20242A {pct}%)"
            center = "●"; color = "var(--pri)"
        elif state == "complete":
            ring = "conic-gradient(var(--ok) 100%, var(--ok) 0)"
            center = "✓"; color = "var(--ok)"
        else:
            ring = "conic-gradient(var(--err) 100%, var(--err) 0)"
            center = "!"; color = "var(--err)"

        html = f"""
<style>
.cs-wrap {{
  --pri:#6aa6ff; --ok:#2ecc71; --err:#ff5c5c;
  display:flex; flex-direction:column; align-items:center; gap:14px;
  padding:14px 10px 6px; margin:8px auto 2px; width:min(640px, 95%);
}}
.cs-title {{ font-size: 1.25rem; font-weight:700; text-align:center; letter-spacing:.2px; }}
.cs-sub   {{ font-size:.95rem; color:#aeb6c2; text-align:center; margin-top:-6px; }}
.cs-time  {{ font-size:.8rem;  color:#8d97a5; text-align:center; margin-top:-6px; }}
.cs-ring {{
  width: 86px; height: 86px; border-radius:50%;
  background: {ring};
  display:flex; align-items:center; justify-content:center;
  box-shadow: 0 0 0 6px rgba(255,255,255,.03) inset, 0 6px 24px rgba(0,0,0,.25);
}}
.cs-center {{
  width:68px; height:68px; border-radius:50%; background:#0f1116;
  display:flex; align-items:center; justify-content:center;
  font-size: 28px; color:{color}; font-weight:800;
}}
.cs-log {{
  width:100%; background:#0f1116; border:1px solid #1d232f;
  border-radius:12px; padding:10px 12px; margin-top:4px; max-height:220px; overflow:auto;
}}
.cs-log ul {{ margin:0; padding-left:18px; }}
.cs-log li {{ margin:2px 0; font-size:.9rem; }}
.cs-foot {{ font-size:.8rem; color:#8d97a5; text-align:center; margin-top:4px; }}
</style>
<div class="cs-wrap">
  <div class="cs-ring"><div class="cs-center">{center}</div></div>
  <div class="cs-title">{self.title}</div>
  <div class="cs-sub">{subtitle}</div>
  <div class="cs-time">{pct}% • {ms} ms</div>
  <div class="cs-log"><ul>{"".join(f"<li>{x}</li>" for x in self._log[-8:])}</ul></div>
  <div class="cs-foot">Consejo: no cierres esta pestaña mientras procesamos.</div>
</div>
"""
        self._block.markdown(html, unsafe_allow_html=True)

    def begin(self, subtitle: str = "Iniciando…"):
        self._state = "running"; self._render(subtitle)

    def step(self, text: str):
        self.step_i = min(self.step_i + 1, self.total)
        self._log.append("✅ " + text)
        self._render(subtitle=text)

    def warn(self, text: str):
        self._log.append("⚠️ " + text)
        self._render(subtitle=text)

    def done(self, final_text: str = "Listo"):
        self.step_i = self.total; self._state = "complete"
        self._log.append("🎉 " + final_text)
        self._render(subtitle=final_text)

    def fail(self, text: str):
        self._state = "error"; self._log.append("❌ " + text)
        self._render(subtitle=text)

    def close(self):
        """Elimina el spinner del layout (deja limpio para tus métricas/gráficas)."""
        try:
            self._block.empty()
            self._host.empty()
        except Exception:
            pass



