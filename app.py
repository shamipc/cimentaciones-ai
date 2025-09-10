# app.py
# ==========================================================
# Optimización de Cimentaciones — versión UI mejorada
# ==========================================================
import math
from io import BytesIO
import base64

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

# ---------- CONFIG BÁSICA ----------
st.set_page_config(
    page_title="Optimización de Cimentaciones",
    layout="wide",
    page_icon="🧱",
)

# ---- THEME & GLOBAL CSS ---------------------------------------
PALETTE = {
    "bg": "#F7FAFC",
    "card": "#FFFFFF",
    "accent": "#12B886",     # verde suave (OK)
    "muted": "#6B7280",
    "text": "#1F2937",
    "bar1": "#8DA2FB",       # pastel azul
    "bar2": "#5ED1B5",       # pastel verde
}

st.markdown("""
<style>
/* Fondo suave */
.main, .block-container { background: #F7FAFC; }

/* Titular */
h1, h2, h3, h4 { color: #1F2937; }

/* Tarjetas tipo “metric” más compactas */
.metric-card {
  background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
  padding: 18px 20px; box-shadow: 0 6px 20px rgba(2, 6, 23, .06);
}
.metric-label { font-size: 0.95rem; color: #6B7280; margin-bottom: 6px; }
.metric-value { font-size: 2.1rem; font-weight: 700; color: #1F2937; line-height: 1; }

/* Pills / badges */
.badge-ok{
  display:inline-flex; align-items:center; gap:10px;
  background:#12B886; color:white; padding:10px 16px;
  border-radius:999px; font-weight:600;
}

/* Caja resumen “card” */
.panel {
  background:#FFFFFF; border:1px solid #E5E7EB; border-radius:14px;
  padding:18px 18px; box-shadow: 0 6px 20px rgba(2, 6, 23, .06);
}

/* Notas */
.note {
  background:#E6F4F1; border:1px solid #BEE3DB; color:#135D4A;
  padding:12px 14px; border-radius:12px;
}

/* Barra separadora elegante */
.hr { height:12px; background:#0B1222; border-radius:10px; opacity:.92; }
</style>
""", unsafe_allow_html=True)

# ---------- FUNCIONES DE CÁLCULO ----------
def bearing_capacity_factors(phi_deg: float):
    """Factores de capacidad portante clásicos (Meyerhof/Prandtl)."""
    phi = math.radians(phi_deg)
    if phi_deg == 0:
        Nq = 1.0
    else:
        Nq = math.e**(math.pi * math.tan(phi)) * (math.tan(math.pi/4 + phi/2))**2
    Nc = (Nq - 1) / math.tan(phi) if phi > 1e-6 else 5.7  # ~5.7 si phi=0
    Ngamma = 2*(Nq+1)*math.tan(phi)
    return Nc, Nq, Ngamma

def qu_terzaghi(B, D, c, gamma, phi):
    """qu para zapata corrida (simplificado)."""
    Nc, Nq, Ngamma = bearing_capacity_factors(phi)
    return c*Nc + gamma*D*Nq + 0.5*gamma*B*Ngamma

def qu_meyerhof(B, D, c, gamma, phi):
    # Muy similar para propósitos didácticos; se mantiene forma
    Nc, Nq, Ngamma = bearing_capacity_factors(phi)
    sc = 1 + 0.2*(B/D) if D > 0 else 1
    sq = 1 + 0.1*(B/D) if D > 0 else 1
    sg = 1 - 0.3*(B/D) if D > 0 else 1
    return c*Nc*sc + gamma*D*Nq*sq + 0.5*gamma*B*Ngamma*sg

def qu_vesic(B, D, c, gamma, phi):
    Nc, Nq, Ngamma = bearing_capacity_factors(phi)
    dc = 1 + 0.2*(D/B) if B > 0 else 1
    dq = 1 + 0.1*(D/B) if B > 0 else 1
    dg = 1
    return c*Nc*dc + gamma*D*Nq*dq + 0.5*gamma*B*Ngamma*dg

def qu_by_model(modelo, B, D, c, gamma, phi):
    if modelo == "Terzaghi (recomendado)":
        return qu_terzaghi(B, D, c, gamma, phi)
    elif modelo == "Meyerhof":
        return qu_meyerhof(B, D, c, gamma, phi)
    elif modelo == "Vesic":
        return qu_vesic(B, D, c, gamma, phi)
    else:
        return qu_terzaghi(B, D, c, gamma, phi)

# ---------- SIDEBAR ----------
st.sidebar.header("Parámetros de entrada")

modelo = st.sidebar.selectbox(
    "Modelo de capacidad",
    ["Terzaghi (recomendado)", "Meyerhof", "Vesic"],
    index=0
)

gamma = st.sidebar.number_input("Peso unitario γ (kN/m³)", 10.0, 25.0, 18.0, 0.1)
c = st.sidebar.number_input("Cohesión c (kPa)", 0.0, 200.0, 20.0, 0.5)
phi = st.sidebar.number_input("Ángulo de fricción φ (°)", 0.0, 45.0, 35.0, 0.5)
D = st.sidebar.number_input("Profundidad D (m)", 0.5, 3.0, 1.5, 0.1)
N = st.sidebar.number_input("Carga N (kN)", 200.0, 10000.0, 1000.0, 10.0)
FS = st.sidebar.number_input("Factor de seguridad", 2.0, 5.0, 2.5, 0.1)

st.sidebar.subheader("Costos")
costo_concreto = st.sidebar.number_input("Concreto (S/ por m³)", 300, 3000, 650, 10)
costo_acero = st.sidebar.number_input("Acero (S/ por kg)", 1.0, 20.0, 5.5, 0.1)
porc_acero = st.sidebar.slider("Acero estimado (% del peso del concreto)", 0.0, 5.0, 1.0, 0.1)

st.sidebar.subheader("Rangos de diseño")
B_min, B_max = st.sidebar.slider("Base B (m)", 1.0, 4.0, (1.2, 3.8), 0.1)
L_min, L_max = st.sidebar.slider("Largo L (m)", 1.0, 4.0, (1.2, 3.8), 0.1)
h_min, h_max = st.sidebar.slider("Altura h (m)", 0.4, 1.2, (0.6, 1.2), 0.05)
paso = 0.1

st.sidebar.subheader("Adjuntos")
uploaded_file = st.sidebar.file_uploader("Sube un croquis / perfil de suelo (PNG/JPG)", type=["png", "jpg", "jpeg"])

st.sidebar.markdown("---")
run = st.sidebar.button("🔎 Analizar y optimizar", use_container_width=True)

# ---------- CONTENIDO ----------
st.title("Optimización de Cimentaciones")
st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta")

if not run:
    st.markdown('<div class="note">Configura los parámetros a la izquierda y pulsa <b>Analizar y optimizar</b>.</div>', unsafe_allow_html=True)
    st.stop()

# ---------- BÚSQUEDA DEL ÓPTIMO ----------
resultados = []
for B in np.arange(B_min, B_max + 1e-6, paso):
    for L in np.arange(L_min, L_max + 1e-6, paso):
        for h in np.arange(h_min, h_max + 1e-6, paso):
            area = B * L
            qu = qu_by_model(modelo, B, D, c, gamma, phi)          # kPa
            qadm = qu / FS
            qreq = N / area                                        # kPa (kN/m²)
            cumple = qadm > qreq

            volumen = B * L * h                                    # m³
            costo = volumen * costo_concreto
            # acero: peso del concreto (densidad aprox 2.4 t/m³ --> 2400 kg/m³)
            peso_concreto_kg = volumen * 2400.0
            costo += peso_concreto_kg * (porc_acero/100.0) * costo_acero

            resultados.append([B, L, h, qadm, qreq, cumple, costo])

df = pd.DataFrame(
    resultados,
    columns=["B (m)", "L (m)", "h (m)", "q_adm (kPa)", "q_req (kPa)", "Cumple", "Costo (S/)"]
)
df_validos = df[df["Cumple"] == True].copy()

if df_validos.empty:
    st.warning("No se encontraron diseños que cumplan con la capacidad admisible. Prueba con B y L mayores, φ o c más altos, FS menor o una carga N menor.")
    st.stop()

# Óptimo = menor costo entre los válidos
idx = df_validos["Costo (S/)"].idxmin()
mejor = df_validos.loc[idx]

B_opt = float(mejor["B (m)"])
L_opt = float(mejor["L (m)"])
h_opt = float(mejor["h (m)"])
qadm_opt = float(mejor["q_adm (kPa)"])
qreq_opt = float(mejor["q_req (kPa)"])
costo_opt = float(mejor["Costo (S/)"])

# ---------- TARJETAS MÉTRICAS ----------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-label">B (m)</div>'
                f'<div class="metric-value">{B_opt:.2f}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-label">L (m)</div>'
                f'<div class="metric-value">{L_opt:.2f}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-label">h (m)</div>'
                f'<div class="metric-value">{h_opt:.2f}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-label">Costo (S/)</div>'
                f'<div class="metric-value">{costo_opt:.0f}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="badge-ok">✅ Diseño óptimo encontrado</div>', unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ---------- GRÁFICOS (PLOTLY, PASTEL) ----------
# Barras pastel q_req vs q_adm
fig1 = go.Figure()
fig1.add_bar(name="q_req", x=["q_req"], y=[qreq_opt], marker_color=PALETTE["bar1"])
fig1.add_bar(name="q_adm", x=["q_adm"], y=[qadm_opt], marker_color=PALETTE["bar2"])
fig1.update_traces(texttemplate="%{y:.1f}", textposition="outside")
fig1.update_layout(
    barmode="group",
    height=380,
    margin=dict(l=40, r=20, t=40, b=60),
    showlegend=False,
    yaxis_title="kPa",
    title="q_req vs q_adm (óptimo)",
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="rgba(0,0,0,0)",
)

# Dispersión candidatos válidos (color=costo, tamaño=h)
fig2 = px.scatter(
    df_validos, x="B (m)", y="L (m)", color="Costo (S/)", size="h (m)",
    color_continuous_scale=px.colors.sequential.Tealgrn,
    height=420, title="Candidatos válidos (color = Costo, tamaño = h)"
)
fig2.update_layout(
    margin=dict(l=40, r=20, t=50, b=40),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="rgba(0,0,0,0)",
)

g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(fig1, use_container_width=True)
with g2:
    st.plotly_chart(fig2, use_container_width=True)

# ---------- RESUMEN Y RECOMENDACIONES ----------
st.subheader("Resumen")
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.json({
    "Modelo": modelo,
    "B (m)": round(B_opt, 2),
    "L (m)": round(L_opt, 2),
    "h (m)": round(h_opt, 2),
    "q_adm (kPa)": round(qadm_opt, 1),
    "q_req (kPa)": round(qreq_opt, 1),
    "Costo (S/)": round(costo_opt, 0)
})
st.markdown('</div>', unsafe_allow_html=True)

st.subheader("Recomendaciones")
st.markdown('<div class="panel">', unsafe_allow_html=True)
ok_margin = qadm_opt >= qreq_opt*1.05
if ok_margin:
    st.markdown("✅ **Buen diseño**: margen suficiente entre capacidad y demanda.")
else:
    st.markdown("⚠️ **Margen ajustado**: considera aumentar **B** o **L**, "
                "o revisar parámetros de suelo (c, φ, γ).")
st.markdown(f"💡 **Óptimo actual**: S/ **{costo_opt:.0f}**. "
            "Evalúa alternativas con *h* ligeramente mayor si buscas rigidez.")
st.markdown("📚 **Referencias clave**: Terzaghi & Peck; Meyerhof; Vesic (capacidad portante clásica).")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- IMAGEN ADJUNTA (solo si sube) ----------
if uploaded_file is not None:
    st.subheader("Imagen adjunta")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.image(uploaded_file, caption="Croquis / perfil estratigráfico", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- HISTORIAL (sesión) ----------
if "runs" not in st.session_state:
    st.session_state["runs"] = []
st.session_state["runs"].append({
    "B": round(B_opt,2), "L": round(L_opt,2), "h": round(h_opt,2),
    "q_adm": round(qadm_opt,1), "q_req": round(qreq_opt,1),
    "costo": round(costo_opt,0)
})
with st.expander("🗂 Historial de resultados (sesión actual)"):
    st.dataframe(st.session_state["runs"], use_container_width=True)

# ---------- REPORTE HTML DESCARGABLE ----------
def fig_to_base64(fig):
    """Devuelve la imagen PNG (base64) de una figura plotly usando kaleido."""
    try:
        buf = BytesIO()
        fig.write_image(buf, format="png", scale=2)  # requiere kaleido
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None

img1 = fig_to_base64(fig1)
img2 = fig_to_base64(fig2)

html_report = f"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Reporte de Cimentaciones</title>
<style>
body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:#111827; }}
h1,h2 {{ color:#111827; }}
.card {{ border:1px solid #E5E7EB; border-radius:12px; padding:16px; margin:14px 0; }}
.small {{ color:#6B7280; }}
</style>
</head>
<body>
<h1>Optimización de Cimentaciones</h1>
<p class="small">Reporte generado automáticamente.</p>
<div class="card">
  <h2>Óptimo</h2>
  <ul>
    <li><b>Modelo:</b> {modelo}</li>
    <li><b>B (m):</b> {B_opt:.2f} &nbsp;&nbsp; <b>L (m):</b> {L_opt:.2f} &nbsp;&nbsp; <b>h (m):</b> {h_opt:.2f}</li>
    <li><b>q_adm (kPa):</b> {qadm_opt:.1f} &nbsp;&nbsp; <b>q_req (kPa):</b> {qreq_opt:.1f}</li>
    <li><b>Costo:</b> S/ {costo_opt:.0f}</li>
  </ul>
</div>
<div class="card">
  <h2>Gráficos</h2>
  {"<img src='"+img1+"' style='max-width:720px;display:block;margin-bottom:10px;' />" if img1 else "<p class='small'>*Las imágenes no se pudieron incrustar (falta kaleido).*</p>"}
  {"<img src='"+img2+"' style='max-width:720px;display:block;' />" if img2 else ""}
</div>
<div class="card">
  <h2>Recomendaciones</h2>
  <ul>
    <li>{'Buen margen entre q_adm y q_req.' if qadm_opt >= qreq_opt*1.05 else 'Margen ajustado: aumentar B/L o revisar parámetros.'}</li>
    <li>Referencias: Terzaghi & Peck; Meyerhof; Vesic.</li>
  </ul>
</div>
</body>
</html>
"""

st.download_button(
    "⬇️ Descargar reporte (HTML)",
    data=html_report.encode("utf-8"),
    file_name="reporte_cimentaciones.html",
    mime="text/html"
)

