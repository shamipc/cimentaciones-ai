
# -------------------------------
# Optimización de Cimentaciones
# vista compacta + gráficos elegantes
# -------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# ---------- CONFIG ----------
st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")

# ---------- CSS suave ----------
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    h1, h2, h3 { font-weight: 700; letter-spacing: .2px; }
    .small-muted { color: #6b7280; font-size: .95rem; }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 8px 16px rgba(17, 24, 39, .05);
    }
    .kpi-title { font-size: .90rem; color: #6b7280; margin-bottom: .15rem; }
    .kpi-value { font-size: 2.0rem; font-weight: 700; color: #111827; }
    .ok-chip { display:inline-block; background:#10b981; color:#fff; padding:.35rem .6rem; border-radius:999px; font-weight:600; }
    .warn-chip { display:inline-block; background:#f59e0b; color:#111827; padding:.35rem .6rem; border-radius:999px; font-weight:700; }
    .bad-chip { display:inline-block; background:#ef4444; color:#fff; padding:.35rem .6rem; border-radius:999px; font-weight:600; }
    .panel {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: 0 10px 20px rgba(17,24,39, .04);
    }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def bearing_capacity_terzaghi(gamma, c, phi_deg, B, D):
    """Terzaghi (c + q + γ) para zapata corrida (simplificado y estable)."""
    phi = np.radians(phi_deg)
    q = gamma * D

    # Factores N (robustos para φ=0)
    if phi_deg <= 1e-6:
        Nq = 1.0
        Nc = 5.7
        Nγ = 0.0
    else:
        Nq = np.exp(np.pi * np.tan(phi)) * (np.tan(np.pi/4 + phi/2))**2
        Nc = (Nq - 1.0) / np.tan(phi)
        # Aproximación estable para Nγ
        Nγ = 2 * (Nq + 1) * np.tan(phi)

    qu = c * Nc + q * Nq + 0.5 * gamma * B * Nγ
    return qu

def costo_aprox(B, L, h, costo_concreto, costo_acero):
    # Concreto: volumen * S/, Acero: 1% del peso del concreto
    volumen = B * L * h              # m3
    peso_concreto = volumen * 2.4e3  # kg (≈2.4 t/m3)
    costo = volumen * costo_concreto + (0.01 * peso_concreto) * costo_acero
    return costo

def recomendaciones(q_req, q_adm, h, FS, mejor):
    tips = []
    margen = (q_adm - q_req) / q_adm if q_adm > 0 else -1

    if q_adm <= q_req:
        tips.append("⚠️ La capacidad admisible **no supera** la requerida. "
                    "Aumenta B o L, reduce N, o emplea un FS menor (si está técnicamente justificado).")
    elif margen < 0.15:
        tips.append("✅ Cumple, pero con **margen ajustado** (<15%). "
                    "Puedes subir ligeramente B o L, o aumentar h para mayor rigidez.")
    else:
        tips.append("✅ Buen diseño: margen suficiente entre capacidad y demanda.")

    if h < 0.6:
        tips.append("⬆️ Considera **h ≥ 0.6 m** para mejorar rigidez y facilitar armado.")
    if FS > 3.5:
        tips.append("ℹ️ El **FS** es alto. Si el proyecto lo permite, un FS más moderado puede disminuir costos.")
    if mejor["Costo (S/)"] > 0:
        tips.append(f"💡 Tu **óptimo** actual cuesta **S/ {mejor['Costo (S/)']:.0f}**; "
                    "revisa alternativas con B y L ligeramente mayores si necesitas más margen.")
    return tips

# ---------- Sidebar (Inputs) ----------
st.sidebar.header("Parámetros de entrada")

modelo = st.sidebar.selectbox("Modelo de capacidad", ["Terzaghi (recomendado)", "Modelo simple"])

gamma = st.sidebar.number_input("Peso unitario γ (kN/m³)", 10.0, 25.0, 18.0, step=0.1)
c     = st.sidebar.number_input("Cohesión c (kPa)", 0.0, 200.0, 20.0, step=0.5)
phi   = st.sidebar.number_input("Ángulo de fricción φ (°)", 0.0, 45.0, 35.0, step=0.5)
D     = st.sidebar.number_input("Profundidad D (m)", 0.5, 3.0, 1.5, step=0.1)
N     = st.sidebar.number_input("Carga N (kN)", 100.0, 6000.0, 1000.0, step=10.0)
FS    = st.sidebar.number_input("Factor de seguridad", 1.5, 5.0, 2.5, step=0.1)

st.sidebar.subheader("Costos")
costo_concreto = st.sidebar.number_input("Concreto (S/ por m³)", 300, 1500, 650, step=10)
costo_acero    = st.sidebar.number_input("Acero (S/ por kg)", 3.0, 12.0, 5.50, step=0.1)

st.sidebar.subheader("Rangos de diseño")
B_min, B_max = st.sidebar.slider("Base B (m)", 1.0, 3.8, (1.2, 3.8), step=0.1)
L_min, L_max = st.sidebar.slider("Largo L (m)", 1.0, 3.8, (1.2, 3.8), step=0.1)
h_min, h_max = st.sidebar.slider("Altura h (m)", 0.4, 1.2, (0.6, 1.2), step=0.05)
paso = 0.1

st.sidebar.subheader("Adjuntos")
up = st.sidebar.file_uploader("Sube croquis o perfil estratigráfico (PNG/JPG)", type=["png","jpg","jpeg"])

analizar = st.sidebar.button("🔎 Analizar y optimizar", use_container_width=True)

# ---------- Encabezado ----------
st.title("Optimización de Cimentaciones")
st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — **vista compacta**")

if not analizar:
    st.info("🛠️ Configura los parámetros a la izquierda y pulsa **Analizar y optimizar**.", icon="🧰")
    st.stop()

# ---------- Cálculo de combinaciones ----------
Bs = np.arange(B_min, B_max + 1e-9, paso)
Ls = np.arange(L_min, L_max + 1e-9, paso)
hs = np.arange(h_min, h_max + 1e-9, paso)

rows = []
for B in Bs:
    for L in Ls:
        for h in hs:
            area = B * L
            q_req = N / area  # kPa (kN/m2)
            if modelo.startswith("Terzaghi"):
                qu = bearing_capacity_terzaghi(gamma, c, phi, B, D)
            else:
                # Modelo simple (conservador)
                qu = 1.3 * c * (1 + 0.2 * (B / L)) + (gamma * D * np.tan(np.radians(phi)))
            q_adm = qu / FS
            cumple = q_adm > q_req
            costo = costo_aprox(B, L, h, costo_concreto, costo_acero)
            rows.append([B, L, h, q_adm, q_req, cumple, costo])

df = pd.DataFrame(rows, columns=["B (m)", "L (m)", "h (m)", "q_adm (kPa)", "q_req (kPa)", "Cumple", "Costo (S/)"])
df_validos = df[df["Cumple"] == True].copy()

# ---------- Si no hay válidos ----------
if df_validos.empty:
    st.error("No se encontraron diseños que cumplan con la capacidad admisible. "
             "Prueba con B/L mayores, φ o c más altos, FS menor o N menor.", icon="⚠️")
    st.stop()

# ---------- Óptimo ----------
mejor = df_validos.loc[df_validos["Costo (S/)"].idxmin()]

# ---------- Tarjetas KPI ----------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="kpi-card"><div class="kpi-title">B (m)</div>'
                f'<div class="kpi-value">{mejor["B (m)"]:.2f}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="kpi-card"><div class="kpi-title">L (m)</div>'
                f'<div class="kpi-value">{mejor["L (m)"]:.2f}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="kpi-card"><div class="kpi-title">h (m)</div>'
                f'<div class="kpi-value">{mejor["h (m)"]:.2f}</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="kpi-card"><div class="kpi-title">Costo (S/)</div>'
                f'<div class="kpi-value">{mejor["Costo (S/)"]:.0f}</div></div>', unsafe_allow_html=True)

# ---------- Etiqueta de cumplimiento ----------
chip = '<span class="ok-chip">Diseño óptimo encontrado</span>'
st.markdown(chip, unsafe_allow_html=True)

# ---------- Gráficos (Plotly, colores suaves) ----------
pal = ["#4f46e5", "#10b981", "#22c55e", "#93c5fd", "#a78bfa"]

# Barra q_req vs q_adm
bar_df = pd.DataFrame({
    "Tipo": ["q_req", "q_adm"],
    "kPa":  [mejor["q_req (kPa)"], mejor["q_adm (kPa)"]]
})
bar = px.bar(
    bar_df, x="Tipo", y="kPa",
    color="Tipo",
    color_discrete_sequence=[pal[0], pal[1]],
    text="kPa",
    title="q_req vs q_adm (óptimo)"
)
bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
bar.update_layout(yaxis_title="kPa", xaxis_title="")
# Dispersión candidatos válidos
scatter = px.scatter(
    df_validos, x="B (m)", y="L (m)",
    size="h (m)", color="Costo (S/)",
    color_continuous_scale="Tealgrn",
    hover_data={"q_adm (kPa)":":.1f","q_req (kPa)":":.1f","h (m)":True,"Costo (S/)":":.0f"},
    title="Candidatos válidos (color = Costo, tamaño = h)"
)
scatter.update_layout(legend_title_text="")

c1, c2 = st.columns([1,1])
with c1:
    st.plotly_chart(bar, use_container_width=True)
with c2:
    st.plotly_chart(scatter, use_container_width=True)

# ---------- Resumen + recomendaciones ----------
st.subheader("Resumen")
res = {
    "Modelo": modelo.split(" (")[0],
    "B (m)": float(mejor["B (m)"]),
    "L (m)": float(mejor["L (m)"]),
    "h (m)": float(mejor["h (m)"]),
    "q_adm (kPa)": float(mejor["q_adm (kPa)"]),
    "q_req (kPa)": float(mejor["q_req (kPa)"]),
    "Costo (S/)": float(mejor["Costo (S/)"])
}
st.json(res)

st.subheader("Recomendaciones")
for tip in recomendaciones(mejor["q_req (kPa)"], mejor["q_adm (kPa)"], mejor["h (m)"], FS, mejor):
    st.markdown(f"- {tip}")

# ---------- Tabla de candidatos (Top 10 por costo) ----------
st.subheader("Top 10 diseños válidos por costo")
top10 = df_validos.sort_values("Costo (S/)").head(10)
st.dataframe(top10.style.format({
    "B (m)": "{:.2f}", "L (m)": "{:.2f}", "h (m)": "{:.2f}",
    "q_adm (kPa)": "{:.1f}", "q_req (kPa)": "{:.1f}", "Costo (S/)": "{:.0f}"
}), use_container_width=True, height=360)

# ---------- Imagen adjunta ----------
st.subheader("Imagen adjunta")
if up is not None:
    st.image(up, caption="Croquis / Perfil estratigráfico", use_container_width=True)
else:
    st.info("Puedes subir un croquis o perfil estratigráfico desde la barra lateral para adjuntarlo en el informe.")
