# ------------------------------------------------------------
# Optimización de Cimentaciones - versión pro
# ------------------------------------------------------------
import io
import math
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# ---------------- Utils ----------------
def pretty(x, nd=2):
    try:
        return float(np.round(x, nd))
    except Exception:
        return x

def to_excel_bytes(df_dict):
    """df_dict: {'Hoja': dataframe, ...} -> bytes de Excel"""
    from pandas import ExcelWriter
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in df_dict.items():
            df.to_excel(writer, sheet_name=name, index=False)
    buf.seek(0)
    return buf.getvalue()

def pdf_bytes(resumen_texto, soluciones_top, file_name="reporte.pdf"):
    """Crea PDF simple con ReportLab."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setTitle("Reporte de Optimización de Cimentaciones")

    # Título
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, h-2*cm, "Optimización de Cimentaciones - Reporte")

    # Resumen
    textobject = c.beginText(2*cm, h-3.2*cm)
    textobject.setFont("Helvetica", 10)
    for line in resumen_texto.split("\n"):
        textobject.textLine(line)
    c.drawText(textobject)

    # Tabla simple de top soluciones
    y = h-9.5*cm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2*cm, y, "Top soluciones:")
    y -= 0.6*cm

    cols = ["B (m)", "L (m)", "h (m)", "q req (kPa)", "q adm (kPa)", "Costo (S/)"]
    c.setFont("Helvetica", 9)
    c.drawString(2*cm, y, " | ".join(cols))
    y -= 0.4*cm

    for _, r in soluciones_top.head(10).iterrows():
        row_txt = " | ".join(str(pretty(r[c])) for c in cols)
        c.drawString(2*cm, y, row_txt[:95])  # acotar
        y -= 0.35*cm
        if y < 2*cm:
            c.showPage()
            y = h-2.5*cm

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()

# ---------------- Modelos de capacidad (simplificados) ----------------
def Nc(phi_deg):
    phi = np.radians(phi_deg)
    return np.exp(np.pi * np.tan(phi)) * (np.tan(np.pi/4 + phi/2))**2

def Nq(phi_deg):
    phi = np.radians(phi_deg)
    return np.exp(np.pi * np.tan(phi)) * (np.tan(np.pi/4 + phi/2))**2

def Ngamma(phi_deg):
    phi = np.radians(phi_deg)
    return 2 * (Nq(phi_deg) + 1) * np.tan(phi)

def q_admisible_terzaghi(c, phi, gamma, B, D, FS):
    # Terzaghi (simplificado, zapata corrida / rectangular con factores 1)
    q_ult = c*Nc(phi) + gamma*D*Nq(phi) + 0.5*gamma*B*Ngamma(phi)
    return q_ult / max(FS, 1e-6)

def q_admisible_simple(c, phi, gamma, B, D, FS):
    # Modelo “rápido” y conservador
    q_ult = 5*c + 0.5*gamma*B + gamma*D
    return q_ult / max(FS, 1e-6)

# ---------------- Sugerencias cuando no hay soluciones ----------------
def sugerencias_sin_solucion(df_todos, N, FS,
                             B_min, B_max, L_min, L_max, h_min, h_max,
                             c, phi, gamma, D, modelo):
    st.warning(
        "No se encontraron soluciones que cumplan la capacidad admisible. "
        "Ajusta uno o más parámetros usando estas recomendaciones."
    )

    if {"q adm (kPa)", "q req (kPa)"}.issubset(df_todos.columns):
        tmp = df_todos.copy()
        tmp["ratio"] = tmp["q adm (kPa)"] / (tmp["q req (kPa)"] + 1e-9)
        best = tmp.loc[tmp["ratio"].idxmax()]
        ratio = float(best["ratio"])
        Bb, Ll, hh = best["B (m)"], best["L (m)"], best["h (m)"]
        qadm, qreq = best["q adm (kPa)"], best["q req (kPa)"]
    else:
        ratio, Bb, Ll, hh, qadm, qreq = 0.0, B_min, L_min, h_min, 0.0, N/(B_min*L_min)

    st.info(
        f"➕ Mejor intento (NO cumple): B={pretty(Bb)} m, L={pretty(Ll)} m, h={pretty(hh)} m • "
        f"q_req={pretty(qreq)} kPa vs q_adm={pretty(qadm)} kPa (ratio={pretty(ratio,3)})"
    )

    sugerencias = []

    if ratio < 1.0:
        factor_area = 1.0 / max(ratio, 1e-9)
        factor_lineal = np.sqrt(factor_area)
        B_nec = min(B_max, Bb * factor_lineal)
        L_nec = min(L_max, Ll * factor_lineal)
        sugerencias.append(
            f"• Aumenta B y L. Escala lineal ≈×{pretty(factor_lineal,2)} "
            f"(prueba B≈{pretty(B_nec)} m y L≈{pretty(L_nec)} m dentro de tus rangos)."
        )

    sugerencias.append(
        f"• Reduce la carga N o baja el FS (p.ej. {pretty(max(1.8, FS-0.5),2)}–{pretty(max(2.0, FS-0.3),2)}) si es admisible."
    )

    if c < 25 or phi < 20:
        sugerencias.append("• Mejorar el terreno o estrato (↑ c y/o ↑ φ).")

    if D < 2.0:
        sugerencias.append("• Aumenta la profundidad D (↑ confinamiento).")

    if (B_max < 5.0) or (L_max < 5.0):
        sugerencias.append("• Amplía los rangos B y L (hasta 5–6 m).")

    sugerencias.append("• Prueba el modelo 'Simple' para un chequeo rápido (más conservador).")

    st.markdown("### Recomendaciones")
    for s in sugerencias:
        st.markdown(s)

# ---------------- Estado de ejecución ----------------
if "run" not in st.session_state:
    st.session_state.run = False

st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")

# ---------------- Sidebar: entradas ----------------
st.sidebar.header("Parámetros de entrada")

modelo = st.sidebar.selectbox(
    "Modelo de capacidad",
    ["Terzaghi (recomendado)", "Simple (rápido)"]
)

# Presets rápidos (opcional)
preset = st.sidebar.selectbox(
    "Preset de suelo (rápido)",
    [
        "Arcilla blanda (γ=17, c=18, φ=0)",
        "Arena densa (γ=19, c=0, φ=35)",
        "Intermedio (γ=18, c=10, φ=25)",
        "Personalizado"
    ],
    index=0
)

# Valores por defecto del preset
p_gamma, p_c, p_phi = 17.0, 18.0, 0.0
if preset == "Arena densa (γ=19, c=0, φ=35)":
    p_gamma, p_c, p_phi = 19.0, 0.0, 35.0
elif preset == "Intermedio (γ=18, c=10, φ=25)":
    p_gamma, p_c, p_phi = 18.0, 10.0, 25.0

gamma = st.sidebar.number_input("Peso unitario γ (kN/m³)", 15.0, 24.0, value=float(p_gamma), step=0.5)
c     = st.sidebar.number_input("Cohesión c (kPa)", 0.0, 200.0, value=float(p_c), step=1.0)
phi   = st.sidebar.number_input("Ángulo de fricción φ (°)", 0.0, 45.0, value=float(p_phi), step=1.0)
D     = st.sidebar.number_input("Profundidad D (m)", 0.0, 5.0, value=1.5, step=0.1)
N     = st.sidebar.number_input("Carga N (kN)", 100.0, 6000.0, value=1000.0, step=50.0)
FS    = st.sidebar.number_input("Factor de seguridad", 1.5, 4.0, value=2.5, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Rangos de búsqueda")
B_min, B_max = st.sidebar.slider("Base B (m)", 1.0, 6.0, (1.2, 4.0), step=0.1)
L_min, L_max = st.sidebar.slider("Largo L (m)", 1.0, 6.0, (1.2, 4.0), step=0.1)
h_min, h_max = st.sidebar.slider("Altura h (m)", 0.4, 1.5, (0.6, 1.0), step=0.05)

# Upload opcional
st.sidebar.markdown("---")
st.sidebar.subheader("Adjuntos")
img_file = st.sidebar.file_uploader("Sube croquis/perfil (PNG/JPG)", type=["png","jpg","jpeg"])

st.sidebar.markdown("---")
if st.sidebar.button("🔎 Analizar y optimizar"):
    st.session_state.run = True
    st.experimental_rerun()

# ---------------- Cabecera ----------------
st.title("Optimización de Cimentaciones")
st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta")

if not st.session_state.run:
    st.info("Configura los parámetros a la izquierda y pulsa **Analizar y optimizar**.")
    st.stop()

# ---------------- Cálculo ----------------
with st.spinner("Calculando candidatos..."):
    # Mallas
    B_vals = np.round(np.linspace(B_min, B_max, 16), 2)
    L_vals = np.round(np.linspace(L_min, L_max, 16), 2)
    h_vals = np.round(np.linspace(h_min, h_max, 9), 2)

    rows = []
    for B in B_vals:
        for L in L_vals:
            for h in h_vals:
                area = B*L
                q_req = N / max(area, 1e-9)  # kPa (asumiendo kN/m2 ~ kPa)
                if modelo.startswith("Terzaghi"):
                    q_adm = q_admisible_terzaghi(c, phi, gamma, B, D, FS)
                else:
                    q_adm = q_admisible_simple(c, phi, gamma, B, D, FS)
                cumple = q_adm >= q_req
                # costo simplificado = volumen hormigón + penalización acero ~ h*B*L + 30*h
                costo = 650*B*L*h + 30*h
                rows.append([B, L, h, q_req, q_adm, costo, cumple])

    df = pd.DataFrame(rows, columns=["B (m)","L (m)","h (m)","q req (kPa)","q adm (kPa)","Costo (S/)","Cumple"])
    df_validos = df[df["Cumple"]==True].copy()

# ---------------- Si no hay soluciones -> sugerencias y auto-ajuste ----------------
if df_validos.empty:
    sugerencias_sin_solucion(
        df_todos=df, N=N, FS=FS,
        B_min=B_min, B_max=B_max, L_min=L_min, L_max=L_max, h_min=h_min, h_max=h_max,
        c=c, phi=phi, gamma=gamma, D=D, modelo=modelo
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("🪄 Auto-ajustar y reintentar"):
            # Relaja FS y amplía 20% los rangos si es posible
            st.session_state.run = True
            st.session_state.relax = True
            st.experimental_rerun()
    with colB:
        st.info("Consejo: reduce FS en 0.25 y/o aumenta máximos de B y L un 20% y vuelve a intentar.")
    st.stop()

# Auto-ajuste si fue solicitado
if st.session_state.get("relax", False):
    st.session_state.relax = False
    FS = max(1.5, FS-0.25)
    B_max = min(6.0, B_max*1.2)
    L_max = min(6.0, L_max*1.2)
    st.info(f"Se aplicó auto-ajuste: FS={pretty(FS)}, B_max={pretty(B_max)}, L_max={pretty(L_max)}. Pulsa de nuevo **Analizar y optimizar**.")
    st.stop()

# ---------------- Óptimo y KPIs ----------------
df_validos["Score"] = df_validos["Costo (S/)"]  # minimizar costo
opt = df_validos.loc[df_validos["Score"].idxmin()].copy()

kpi_cols = st.columns(4)
kpi_cols[0].metric("B (m)", pretty(opt["B (m)"]))
kpi_cols[1].metric("L (m)", pretty(opt["L (m)"]))
kpi_cols[2].metric("h (m)", pretty(opt["h (m)"]))
kpi_cols[3].metric("Costo (S/)", int(round(opt["Costo (S/)"])))

st.success("✅ Diseño óptimo encontrado")

# ---------------- Gráficos principales ----------------
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(
        pd.DataFrame({
            "Tipo":["q_req","q_adm"],
            "kPa":[opt["q req (kPa)"], opt["q adm (kPa)"]]
        }),
        x="Tipo", y="kPa", title="q_req vs q_adm (óptimo)",
        color="Tipo", color_discrete_sequence=px.colors.sequential.Tealgrn
    )
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig2 = px.scatter(
        df_validos, x="B (m)", y="L (m)",
        size="h (m)", color="Costo (S/)",
        color_continuous_scale=px.colors.sequential.Teal,
        title="Candidatos válidos (color = Costo, tamaño = h)"
    )
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- Tabla de soluciones y export ----------------
st.subheader("Soluciones válidas")
df_show = df_validos.sort_values("Costo (S/)").head(200).copy()
st.dataframe(df_show, use_container_width=True, hide_index=True)

colx, coly = st.columns(2)

with colx:
    xls = to_excel_bytes({"Top": df_show, "Óptimo": pd.DataFrame([opt])})
    st.download_button(
        "📥 Descargar Excel",
        data=xls, file_name="cimentaciones_resultados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with coly:
    resumen = (
        f"Modelo: {modelo}\n"
        f"γ={pretty(gamma)} kN/m³, c={pretty(c)} kPa, φ={pretty(phi)}°, D={pretty(D)} m, N={pretty(N)} kN, FS={pretty(FS)}\n"
        f"Óptimo: B={pretty(opt['B (m)'])} m, L={pretty(opt['L (m)'])} m, h={pretty(opt['h (m)'])} m, "
        f"q_req={pretty(opt['q req (kPa)'])} kPa, q_adm={pretty(opt['q adm (kPa)'])} kPa, "
        f"Costo=S/ {int(round(opt['Costo (S/)']))}"
    )
    pdfbin = pdf_bytes(resumen, df_show, "reporte.pdf")
    st.download_button("🧾 Descargar reporte (PDF)", data=pdfbin, file_name="reporte_cimentaciones.pdf", mime="application/pdf")

# ---------------- Esquema óptimo ----------------
st.subheader(f"Esquema (h = {pretty(opt['h (m)'])} m)")
Bopt, Lopt, hopt = float(opt["B (m)"]), float(opt["L (m)"]), float(opt["h (m)"])
fig3 = px.imshow(
    np.ones((int(max(1, hopt*100)), int(max(1, Bopt*100))))*0.9,
    color_continuous_scale=[[0,"#e9f5f2"],[1,"#9fd6cc"]],
    origin="lower"
)
fig3.update_layout(coloraxis_showscale=False, margin=dict(l=10,r=10,t=10,b=10), height=250)
fig3.update_xaxes(showticklabels=False)
fig3.update_yaxes(showticklabels=False)
st.plotly_chart(fig3, use_container_width=True)

# Imagen adjunta (opcional)
if img_file is not None:
    st.subheader("Imagen adjunta")
    st.image(img_file, use_container_width=True, caption="Adjunto del usuario")

# ---------------- Recomendaciones finales ----------------
st.subheader("Recomendaciones")
st.markdown(
    f"✅ **Buen diseño:** margen suficiente entre capacidad y demanda.\n\n"
    f"💡 **Óptimo actual:** S/ {int(round(opt['Costo (S/)']))}. "
    f"Si necesitas más rigidez, evalúa alternativas con **h** ligeramente mayor.\n\n"
    "📚 **Referencias clave:** Terzaghi & Peck; Meyerhof; Vesic (capacidad portante clásica).\n"
)


     
