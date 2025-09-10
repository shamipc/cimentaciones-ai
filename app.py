# app.py
import io
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")

# ------------------------------ ESTILO ---------------------------------
PALETTE = {
    "primary": "#2563eb",   # azul suave
    "green":   "#10b981",
    "red":     "#ef4444",
    "ink":     "#0f172a",
    "muted":   "#64748b",
}
px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = px.colors.sequential.Tealgrn

CSS = f"""
<style>
h1,h2,h3,h4 {{ color:{PALETTE['ink']}; }}
div[data-testid="stMetricValue"] {{ font-size: 28px; }}
div[data-testid="stMetricDelta"] {{ font-size: 14px; }}
.block-note {{ background:#f1f5f9; padding:12px 16px; border-radius:14px; }}
.card {{ border:1px solid #e5e7eb; border-radius:16px; padding:20px; }}
hr {{ border:0; height:1px; background:#e5e7eb; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------- FUNCIONES --------------------------------
def qu_terzaghi(c, gamma, phi_deg, D, B):
    """Capacidad portante última muy simplificada (fines didácticos)."""
    phi = np.radians(phi_deg)
    Nc = (np.tan(np.radians(45) + phi/2))**2
    Nq = Nc * np.e**(np.pi*np.tan(phi))
    Ny = 2*(Nq+1)*np.tan(phi)
    # término sobrecarga ~ gamma*D*Nq y término peso propio ~ 0.5*gamma*B*Ny
    qu = c*Nc + gamma*D*Nq + 0.5*gamma*B*Ny
    return qu

def evaluar_grid(B_rng, L_rng, h_rng, paso, params):
    gamma, c, phi, D, N, FS, costo_conc, costo_acero = params
    rows = []
    for B in np.arange(B_rng[0], B_rng[1] + 1e-9, paso):
        for L in np.arange(L_rng[0], L_rng[1] + 1e-9, paso):
            for h in np.arange(h_rng[0], h_rng[1] + 1e-9, paso):
                area = B*L
                qu = qu_terzaghi(c, gamma, phi, D, B)
                qadm = qu/FS
                qreq = N/area
                cumple = qadm >= qreq

                vol = B*L*h
                peso_conc = vol*2.4*1000  # kg aprox
                costo = vol*costo_conc + peso_conc*0.01*costo_acero

                rows.append([B, L, h, area, qadm, qreq, cumple, costo])

    df = pd.DataFrame(rows, columns=["B","L","h","area","q_adm","q_req","cumple","costo"])
    return df

def esquema_optimo(B, L, h):
    """Dibujo simple de la zapata óptima."""
    fig, ax = plt.subplots(figsize=(4,3))
    ax.add_patch(plt.Rectangle((0,0), B, L, color="#dbeafe"))
    ax.add_patch(plt.Rectangle((B*0.2, L*0.2), B*0.6, L*0.6, fill=False, lw=2, ec="#2563eb"))
    ax.set_xlim(0, B*1.1); ax.set_ylim(0, L*1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("B (m)"); ax.set_ylabel("L (m)")
    ax.set_title(f"Esquema (h = {h:.2f} m)")
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def df_to_excel_bytes(df):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="resultados", index=False)
    bio.seek(0)
    return bio

def generar_pdf(params, df_validos, best, imagen_user=None):
    """Reporte rápido en PDF (1–2 páginas)."""
    gamma, c, phi, D, N, FS, costo_conc, costo_acero, modelo = params
    buf = io.BytesIO()
    can = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    def t(x,y,s,sz=11): 
        can.setFont("Helvetica", sz); can.drawString(x, y, s)

    y = H - 2*cm
    t(2*cm, y, "Optimización de Cimentaciones – Reporte", 14); y -= 0.8*cm
    t(2*cm, y, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}"); y -= 0.6*cm
    t(2*cm, y, "Parámetros de entrada:", 12); y -= 0.6*cm
    for line in [
        f"Modelo: {modelo}",
        f"γ={gamma} kN/m³, c={c} kPa, φ={phi}°, D={D} m",
        f"N={N} kN, FS={FS}, Concreto=S/ {costo_conc}/m³, Acero=S/ {costo_acero}/kg",
    ]:
        t(2.2*cm, y, "• "+line); y -= 0.6*cm

    y -= 0.2*cm
    t(2*cm, y, "Óptimo:", 12); y -= 0.6*cm
    t(2.2*cm, y, f"B={best.B:.2f} m, L={best.L:.2f} m, h={best.h:.2f} m, Costo=S/ {best.costo:.0f}")
    y -= 0.8*cm

    # Top 10 tabla
    t(2*cm, y, "Top 10 soluciones por costo:", 12); y -= 0.6*cm
    top = df_validos.nsmallest(10, "costo")[["B","L","h","q_adm","q_req","costo"]]
    y0 = y
    for i, row in top.iterrows():
        line = f"{row.B:.2f}\t{row.L:.2f}\t{row.h:.2f}\t{row.q_adm:.1f}\t{row.q_req:.1f}\t{row.costo:.0f}"
        t(2.2*cm, y, line); y -= 0.5*cm
        if y < 3*cm:  # nueva página si se acaba el espacio
            can.showPage(); y = H - 2*cm

    # esquema
    try:
        esqu_buf = esquema_optimo(best.B, best.L, best.h)
        can.drawImage(esqu_buf, 12.5*cm, H-8*cm, width=5.5*cm, height=4*cm, preserveAspectRatio=True)
    except:
        pass

    # imagen de usuario (si hay)
    if imagen_user is not None:
        can.showPage()
        t(2*cm, H-2*cm, "Imagen adjunta del usuario:", 12)
        can.drawImage(imagen_user, 2*cm, H-18*cm, width=16*cm, height=14*cm, preserveAspectRatio=True)

    can.save()
    buf.seek(0)
    return buf

# -------------------------- SIDEBAR / ENTRADA ---------------------------
st.sidebar.header("Parámetros de entrada")

# Presets de suelo
presets = {
    "Terzaghi (recomendado)": dict(gamma=18.0, c=20.0, phi=35.0),
    "Arena densa":            dict(gamma=19.5, c=0.0,  phi=38.0),
    "Arcilla media":          dict(gamma=17.5, c=25.0, phi=22.0),
    "Personalizado":          dict(gamma=18.0, c=10.0, phi=30.0),
}
modelo = st.sidebar.selectbox("Modelo de capacidad", list(presets.keys()))
gamma = st.sidebar.number_input("Peso unitario γ (kN/m³)", 10.0, 25.0, presets[modelo]["gamma"], step=0.1)
c     = st.sidebar.number_input("Cohesión c (kPa)",       0.0, 200.0, presets[modelo]["c"], step=0.5)
phi   = st.sidebar.number_input("Ángulo de fricción φ (°)", 10.0, 45.0, presets[modelo]["phi"], step=0.5)
D     = st.sidebar.number_input("Profundidad D (m)", 0.5, 5.0, 1.5, step=0.1)
N     = st.sidebar.number_input("Carga N (kN)",     100.0, 6000.0, 1000.0, step=10.0)
FS    = st.sidebar.number_input("Factor de seguridad", 1.5, 5.0, 2.5, step=0.1)

st.sidebar.subheader("Costos")
costo_conc  = st.sidebar.number_input("Concreto (S/ m³)", 300, 1500, 650)
costo_acero = st.sidebar.number_input("Acero (S/ kg)",     2.0, 10.0, 5.5, step=0.1)

st.sidebar.subheader("Rangos de diseño")
B_rng = st.sidebar.slider("Base B (m)", 1.2, 3.8, (1.2, 3.8), step=0.1)
L_rng = st.sidebar.slider("Largo L (m)", 1.2, 3.8, (1.2, 3.8), step=0.1)
fix_ratio = st.sidebar.checkbox("Fijar relación L/B")
ratio = st.sidebar.number_input("L/B", 1.0, 3.0, 1.0, step=0.05, disabled=not fix_ratio)
h_rng = st.sidebar.slider("Altura h (m)", 0.4, 1.2, (0.6, 1.0), step=0.05)
paso  = 0.1

st.sidebar.subheader("Adjuntos")
up_img = st.sidebar.file_uploader("Sube un croquis / perfil de suelo (PNG/JPG)", type=["png","jpg","jpeg"])

# ------------------------------- TÍTULO ---------------------------------
st.title("Optimización de Cimentaciones")
st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta")

# ----------------------------- CÁLCULO ----------------------------------
params_core = (gamma, c, phi, D, N, FS, costo_conc, costo_acero)

if fix_ratio:
    # si se fija L/B, reinterpreta el rango de L en función del de B
    L_rng = (B_rng[0]*ratio, B_rng[1]*ratio)

if st.sidebar.button("🔎 Analizar y optimizar", use_container_width=True):
    st.session_state["run"] = True

if "run" not in st.session_state:
    st.info("🔧 Configura los parámetros a la izquierda y pulsa **Analizar y optimizar**.")
    st.stop()

df = evaluar_grid(B_rng, L_rng, h_rng, paso, params_core)
df_validos = df[df["cumple"]].copy()

if df_validos.empty:
    st.warning("No se encontraron diseños que cumplan la capacidad admisible. Prueba con B y L mayores, φ o c más altos, FS menor o una carga N menor.")
    st.stop()

best = df_validos.loc[df_validos["costo"].idxmin()]

# Métricas
mcol = st.columns(4)
mcol[0].metric("B (m)", f"{best.B:.2f}")
mcol[1].metric("L (m)", f"{best.L:.2f}")
mcol[2].metric("h (m)", f"{best.h:.2f}")
mcol[3].metric("Costo (S/)", f"{best.costo:.0f}")
st.success("✅ Diseño óptimo encontrado")

st.markdown("<hr/>", unsafe_allow_html=True)

# ---------------------------- GRÁFICOS ----------------------------------
c1, c2 = st.columns(2)

with c1:
    fig_bar = px.bar(
        x=["q_req","q_adm"], 
        y=[best.q_req, best.q_adm],
        text=[f"{best.q_req:.1f}", f"{best.q_adm:.1f}"],
        labels={"x":"", "y":"kPa"},
        color=["q_req","q_adm"],
        color_discrete_map={"q_req":"#7c83fd","q_adm":"#10b981"},
        title="q_req vs q_adm (óptimo)"
    )
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    fig_sc = px.scatter(
        df_validos, x="B", y="L", size="h", color="costo",
        labels={"B":"B (m)", "L":"L (m)", "costo":"Costo (S/)"},
        title="Candidatos válidos (color = costo, tamaño = h)"
    )
    st.plotly_chart(fig_sc, use_container_width=True)

exp = st.expander("Más gráficos (3D y costo vs área)", expanded=False)
with exp:
    c3, c4 = st.columns(2)
    with c3:
        fig3d = px.scatter_3d(
            df_validos.sample(min(2000, len(df_validos))), x="B", y="L", z="h",
            color="costo", labels={"B":"B","L":"L","h":"h","costo":"S/"},
            title="Exploración 3D (B-L-h, color=costo)",
            opacity=0.7
        )
        st.plotly_chart(fig3d, use_container_width=True)
    with c4:
        area_df = df_validos.groupby("area", as_index=False)["costo"].min().sort_values("area")
        fig_line = px.line(area_df, x="area", y="costo", markers=True,
                           labels={"area":"Área (m²)","costo":"Costo (S/)"},
                           title="Costo mínimo vs Área")
        st.plotly_chart(fig_line, use_container_width=True)

# ---------------------------- RESUMEN -----------------------------------
st.subheader("Resumen")
st.code(json.dumps({
    "Modelo": modelo,
    "B (m)": round(best.B,2),
    "L (m)": round(best.L,2),
    "h (m)": round(best.h,2),
    "q_adm (kPa)": round(best.q_adm,1),
    "q_req (kPa)": round(best.q_req,1),
    "Costo (S/)": round(best.costo),
}, indent=2), language="json")

st.subheader("Recomendaciones")
st.markdown(
    f"""
<div class="block-note">
✅ <b>Buen diseño:</b> margen suficiente entre capacidad y demanda.<br/>
💡 <b>Óptimo actual:</b> S/ {best.costo:.0f}. Evalúa alternativas con <i>h</i> ligeramente mayor si buscas rigidez.<br/>
📚 <b>Referencias clave:</b> Terzaghi & Peck; Meyerhof; Vesic (capacidad portante clásica).
</div>
""", unsafe_allow_html=True)

# --------------------------- TABLA / DESCARGAS --------------------------
st.subheader("Tabla de resultados válidos")
df_show = df_validos[["B","L","h","area","q_adm","q_req","costo"]].sort_values("costo")
st.dataframe(df_show, use_container_width=True, height=300)

c_dl1, c_dl2, c_dl3 = st.columns(3)
csv_bytes = df_show.to_csv(index=False).encode("utf-8")
c_dl1.download_button("⬇️ Descargar CSV", csv_bytes, "resultados.csv", mime="text/csv", use_container_width=True)
xlsx_bytes = df_to_excel_bytes(df_show).getvalue()
c_dl2.download_button("⬇️ Descargar Excel", xlsx_bytes, "resultados.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

# esquema óptimo
esquema_png = esquema_optimo(best.B, best.L, best.h)
st.image(esquema_png, caption="Esquema de la zapata óptima", use_column_width=False)

# imagen de usuario
if up_img is not None:
    st.subheader("Imagen adjunta")
    st.image(up_img, use_column_width=True)

# reporte PDF
params_pdf = (gamma, c, phi, D, N, FS, costo_conc, costo_acero, modelo)
img_for_pdf = up_img if up_img is not None else None
pdf_buf = generar_pdf(params_pdf, df_show, best, img_for_pdf)
c_dl3.download_button("🧾 Descargar reporte (PDF)", pdf_buf.getvalue(), "reporte_cimentaciones.pdf", mime="application/pdf", use_container_width=True)

# ------------------------------ HISTORIAL -------------------------------
st.subheader("Historial de resultados (sesión actual)")
hist = st.session_state.get("hist", pd.DataFrame(columns=df_show.columns))
hist = pd.concat([hist, df_show.head(1)], ignore_index=True)
st.session_state["hist"] = hist
st.dataframe(hist, use_container_width=True, height=180)
