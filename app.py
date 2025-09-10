import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

# =========================
# CONFIG & THEME
# =========================
st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")

PALETTE = {
    "primary": "#2E86AB",
    "accent": "#20bf6b",
    "muted": "#96a5b4",
    "bg_pill": "#ECF4F9",
    "ok": "#2ecc71",
    "warn": "#f39c12",
    "danger": "#e74c3c",
    "blue": "#6C8AE4",
    "green": "#2DBE88",
    "teal": "#1abc9c",
}

PLOT_TEMPLATE = "plotly_white"
BASE_FONT = 13  # texto general en gráficos

# =========================
# FUNCIONES DE CÁLCULO
# =========================
def bearing_capacity_terzaghi(c_kpa, phi_deg, gamma_kN_m3, D_m):
    """Nc, Nq, Ny aproximados (suelo con cohesión y fricción)."""
    phi = np.radians(phi_deg)
    Nq = np.exp(np.pi * np.tan(phi)) * (np.tan(np.radians(45) + phi / 2)) ** 2
    Nc = (Nq - 1) / np.tan(phi) if phi_deg > 0 else 5.7  # manejo φ≈0
    Ny = 2 * (Nq + 1) * np.tan(phi)
    qu = c_kpa * Nc + gamma_kN_m3 * D_m * Nq + 0.5 * gamma_kN_m3 * Ny
    return qu, Nc, Nq, Ny

def bearing_capacity_simple(c_kpa, phi_deg, gamma_kN_m3, D_m):
    qu = 5 * c_kpa + gamma_kN_m3 * D_m + 0.4 * gamma_kN_m3
    return qu, np.nan, np.nan, np.nan

def capacity(modelo, c_kpa, phi_deg, gamma_kN_m3, D_m):
    if modelo.startswith("Terzaghi"):
        return bearing_capacity_terzaghi(c_kpa, phi_deg, gamma_kN_m3, D_m)
    return bearing_capacity_simple(c_kpa, phi_deg, gamma_kN_m3, D_m)

def costo_est(B, L, h, s_concreto=650):
    # costo simple proporcional a volumen (puedes refinar si quieres)
    vol = B * L * h
    return vol * s_concreto

def grid_search(modelo, c, phi, gamma, D, FS, N, 
                B_range, L_range, h_range, step=0.05,
                fix_ratio=False, ratio=1.0):
    data = []
    if fix_ratio:
        # barrido solo en B; L = ratio * B (si cae dentro del rango)
        for Bb in np.round(np.arange(B_range[0], B_range[1] + 1e-9, step), 2):
            Ll = round(float(ratio * Bb), 2)
            if Ll < L_range[0] or Ll > L_range[1]:
                continue
            for hh in np.round(np.arange(h_range[0], h_range[1] + 1e-9, step), 2):
                A = Bb * Ll
                qu, *_ = capacity(modelo, c, phi, gamma, D)
                qadm = qu / FS
                qreq = (N * 1000) / A  # kN -> kPa
                ok = qadm >= qreq
                cost = costo_est(Bb, Ll, hh)
                data.append([Bb, Ll, hh, A, qadm, qreq, ok, cost])
    else:
        for Bb in np.round(np.arange(B_range[0], B_range[1] + 1e-9, step), 2):
            for Ll in np.round(np.arange(L_range[0], L_range[1] + 1e-9, step), 2):
                for hh in np.round(np.arange(h_range[0], h_range[1] + 1e-9, step), 2):
                    A = Bb * Ll
                    qu, *_ = capacity(modelo, c, phi, gamma, D)
                    qadm = qu / FS
                    qreq = (N * 1000) / A
                    ok = qadm >= qreq
                    cost = costo_est(Bb, Ll, hh)
                    data.append([Bb, Ll, hh, A, qadm, qreq, ok, cost])
    df = pd.DataFrame(data, columns=["B", "L", "h", "Area", "q_adm", "q_req", "ok", "costo"])
    return df

# =========================
# SIDEBAR – ENTRADAS
# =========================
st.sidebar.header("Parámetros de entrada")
modelo = st.sidebar.selectbox("Modelo de capacidad", ["Terzaghi (recomendado)", "Simple"])

# Presets de suelo
preset = st.sidebar.selectbox(
    "Preset de suelo (rápido)",
    ["—", "Arena densa (γ=19, φ=38°, c≈0)", "Arena media (γ=18, φ=34°, c≈0)",
     "Arcilla blanda (γ=17, φ=0°, c=18)", "Arcilla media (γ=18, φ=0°, c=30)"]
)

# Valores por defecto
gamma_def, c_def, phi_def = 18.0, 20.0, 35.0

# Inputs con estado (para que el preset pueda sobrescribir)
if "gamma" not in st.session_state: st.session_state.gamma = gamma_def
if "c" not in st.session_state: st.session_state.c = c_def
if "phi" not in st.session_state: st.session_state.phi = phi_def

# Aplica preset si se selecciona uno distinto
if preset != "—":
    if "Arena densa" in preset:
        st.session_state.gamma, st.session_state.c, st.session_state.phi = 19.0, 0.0, 38.0
    elif "Arena media" in preset:
        st.session_state.gamma, st.session_state.c, st.session_state.phi = 18.0, 0.0, 34.0
    elif "Arcilla blanda" in preset:
        st.session_state.gamma, st.session_state.c, st.session_state.phi = 17.0, 18.0, 0.0
    elif "Arcilla media" in preset:
        st.session_state.gamma, st.session_state.c, st.session_state.phi = 18.0, 30.0, 0.0

gamma = st.sidebar.number_input("Peso unitario γ (kN/m³)", 10.0, 25.0, st.session_state.gamma)
c = st.sidebar.number_input("Cohesión c (kPa)", 0.0, 200.0, st.session_state.c)
phi = st.sidebar.number_input("Ángulo de fricción φ (°)", 0.0, 45.0, st.session_state.phi)
D = st.sidebar.number_input("Profundidad D (m)", 0.5, 3.0, 1.5, step=0.1)
N = st.sidebar.number_input("Carga N (kN)", 100.0, 6000.0, 1000.0, step=50.0)
FS = st.sidebar.number_input("Factor de seguridad", 1.5, 5.0, 2.5, step=0.1)

st.sidebar.subheader("Rangos de diseño")
B_min, B_max = 1.2, 3.8
L_min, L_max = 1.2, 3.8
h_min, h_max = 0.6, 1.2
B = st.sidebar.slider("Base B (m)", B_min, B_max, (1.2, 1.2), 0.02)
L = st.sidebar.slider("Largo L (m)", L_min, L_max, (1.2, 1.2), 0.02)
h = st.sidebar.slider("Altura h (m)", h_min, h_max, (0.6, 0.6), 0.02)

st.sidebar.subheader("Opciones avanzadas")
fix_ratio = st.sidebar.checkbox("Fijar relación L/B")
ratio = st.sidebar.number_input("Razón R = L/B", 0.5, 3.0, 1.50, 0.05, disabled=not fix_ratio)

st.sidebar.subheader("Adjuntos")
uploaded = st.sidebar.file_uploader(
    "Sube un croquis / perfil de suelo (PNG/JPG)",
    type=["png", "jpg", "jpeg"]
)

st.sidebar.markdown("---")
run = st.sidebar.button("🔎 Analizar y optimizar", use_container_width=True)

# =========================
# TÍTULO
# =========================
st.title("Optimización de Cimentaciones")
st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta")

# =========================
# CÁLCULO / OPTIMIZACIÓN
# =========================
if run:
    with st.spinner("Calculando combinaciones y graficando…"):
        df = grid_search(
            modelo, c, phi, gamma, D, FS, N,
            B_range=B, L_range=L, h_range=h,
            step=0.1, fix_ratio=fix_ratio, ratio=ratio
        )
        valid = df[df["ok"]].copy()

        if valid.empty:
            st.warning(
                "No se encontraron soluciones que cumplan la capacidad admisible. "
                "Prueba con B y L mayores, φ o c más altos, FS menor o carga menor."
            )
            st.stop()

        opt = valid.sort_values("costo", ascending=True).iloc[0]

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("B (m)", f"{opt.B:.2f}")
        with col2:
            st.metric("L (m)", f"{opt.L:.2f}")
        with col3:
            st.metric("h (m)", f"{opt.h:.2f}")
        with col4:
            st.metric("Costo (S/)", f"{opt.costo:.0f}")

        st.success("✅ Diseño óptimo encontrado")

        # =========================
        # GRÁFICOS PRINCIPALES
        # =========================
        # 1) Barra q_req vs q_adm
        qdf = pd.DataFrame({"Tipo": ["q_req", "q_adm"], "kPa": [opt.q_req, opt.q_adm]})
        fig_bar = px.bar(
            qdf, x="Tipo", y="kPa",
            color="Tipo",
            color_discrete_map={"q_req": PALETTE["blue"], "q_adm": PALETTE["green"]},
            text=qdf["kPa"].round(1),
            template=PLOT_TEMPLATE
        )
        fig_bar.update_layout(
            title="q_req vs q_adm (óptimo)",
            title_font_size=BASE_FONT+1,
            font_size=BASE_FONT-1,
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        fig_bar.update_traces(textposition="outside")

        # 2) Dispersión B–L coloreado por costo (tamaño = h)
        fig_sc = px.scatter(
            valid, x="B", y="L",
            color="costo", size="h",
            color_continuous_scale=px.colors.sequential.Tealgrn,
            template=PLOT_TEMPLATE,
        )
        fig_sc.update_layout(
            title="Candidatos válidos (color = Costo, tamaño = h)",
            title_font_size=BASE_FONT+1,
            font_size=BASE_FONT-1,
            margin=dict(l=10, r=10, t=40, b=10),
            coloraxis_colorbar=dict(title="Costo (S/)")
        )

        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_bar, use_container_width=True)
        c2.plotly_chart(fig_sc, use_container_width=True)

        # 3) Costo vs Área (2D)
        fig_area = px.scatter(
            valid.sort_values("Area"),
            x="Area", y="costo",
            color="h",
            color_continuous_scale=px.colors.sequential.Blues,
            template=PLOT_TEMPLATE,
            labels={"Area": "Área (m²)", "costo": "Costo (S/)", "h": "h (m)"}
        )
        fig_area.update_layout(
            title="Costo vs. Área (m²) — color por h",
            title_font_size=BASE_FONT+1,
            font_size=BASE_FONT-1,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_area, use_container_width=True)

        # 4) 3D Scatter opcional
        with st.expander("Visualización 3D (B–L–h, color por costo)"):
            fig_3d = px.scatter_3d(
                valid, x="B", y="L", z="h", color="costo",
                color_continuous_scale=px.colors.sequential.Viridis,
                template=PLOT_TEMPLATE,
            )
            fig_3d.update_layout(
                scene=dict(xaxis_title="B (m)", yaxis_title="L (m)", zaxis_title="h (m)"),
                margin=dict(l=0, r=0, t=0, b=0),
                font_size=BASE_FONT-2,
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        # =========================
        # RESUMEN + RECOMENDACIONES
        # =========================
        st.subheader("Resumen")
        resumen = {
            "Modelo": "Terzaghi" if modelo.startswith("Terzaghi") else "Simple",
            "B (m)": round(float(opt.B), 2),
            "L (m)": round(float(opt.L), 2),
            "h (m)": round(float(opt.h), 2),
            "Área (m²)": round(float(opt.Area), 2),
            "q_adm (kPa)": round(float(opt.q_adm), 1),
            "q_req (kPa)": round(float(opt.q_req), 1),
            "Utilización q_req/q_adm": round(float(opt.q_req/opt.q_adm), 3),
            "Costo (S/)": round(float(opt.costo), 0),
        }
        st.code(json.dumps(resumen, indent=2), language="json")

        st.subheader("Recomendaciones")
        util = opt.q_req/opt.q_adm
        tips = []
        if util < 0.7:
            tips.append("El diseño tiene margen alto; podrías reducir B o L para bajar costo si lo permite la rigidez.")
        elif util > 0.95:
            tips.append("Demasiado ajustado; considera aumentar B o L ligeramente para margen de construcción.")
        if opt.h < (h_min + h_max)/2:
            tips.append("h relativamente pequeño; si necesitas menor deformación, aumenta h un 10–20%.")
        if fix_ratio:
            tips.append(f"Se usó razón fija L/B = {ratio:.2f}. Verifica dicho criterio con tu caso de carga.")
        tips_text = "\n".join([f"• {t}" for t in tips]) if tips else "• El diseño está balanceado en capacidad y costo."

        st.info(
            "✅ **Buen diseño**: cumple capacidad con FS.\n\n"
            f"🔎 **Utilización** q_req/q_adm = **{util:.3f}**.\n\n"
            f"{tips_text}\n\n"
            "📚 **Referencias**: Terzaghi & Peck; Meyerhof; Vesic (capacidad portante clásica)."
        )

        # =========================
        # TABLA TOP 20
        # =========================
        st.markdown("**Top 20 soluciones válidas (por costo):**")
        top = valid.sort_values("costo", ascending=True).head(20).copy()
        top_display = top[["B", "L", "h", "Area", "q_adm", "q_req", "costo"]]
        st.dataframe(top_display, use_container_width=True)

        # =========================
        # ESQUEMA DE ZAPATA (matplotlib)
        # =========================
        st.subheader(f"Esquema (h = {opt.h:.2f} m)")
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
        # Planta
        axs[0].set_title("Planta (B × L)", fontsize=11)
        axs[0].add_patch(plt.Rectangle((0, 0), opt.B, opt.L, color="#cfe9ff"))
        axs[0].set_aspect("equal")
        axs[0].set_xlim(0, max(opt.B*1.15, 1.0))
        axs[0].set_ylim(0, max(opt.L*1.15, 1.0))
        axs[0].set_xlabel("B (m)")
        axs[0].set_ylabel("L (m)")
        axs[0].grid(alpha=.3)

        # Corte
        axs[1].set_title("Corte (h)", fontsize=11)
        axs[1].add_patch(plt.Rectangle((0, 0), opt.B, opt.h, color="#d7f5ea"))
        axs[1].set_aspect("auto")
        axs[1].set_xlim(0, max(opt.B*1.15, 1.0))
        axs[1].set_ylim(0, max(opt.h*1.6, 1.0))
        axs[1].set_xlabel("B (m)")
        axs[1].set_ylabel("h (m)")
        axs[1].grid(alpha=.3)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # =========================
        # DESCARGAS (EXCEL & PDF)
        # =========================
        colx1, colx2 = st.columns(2)

        # Excel
        with io.BytesIO() as buffer:
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                top_display.to_excel(writer, index=False, sheet_name="Top20")
                df.to_excel(writer, index=False, sheet_name="Grid")
                pd.DataFrame([resumen]).to_excel(writer, index=False, sheet_name="Resumen")
            excel_bytes = buffer.getvalue()
        colx1.download_button(
            "📥 Descargar Excel",
            data=excel_bytes,
            file_name="cimentaciones_optimizacion.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        # PDF
        def pdf_bytes_from_summary(resumen_dict, recomendaciones_str):
            buf = io.BytesIO()
            cpdf = canvas.Canvas(buf, pagesize=A4)
            w, hpdf = A4
            x, y = 2*cm, hpdf - 2*cm

            cpdf.setFont("Helvetica-Bold", 14)
            cpdf.drawString(x, y, "Reporte de Optimización de Cimentaciones")
            y -= 1.2*cm

            cpdf.setFont("Helvetica", 10)
            cpdf.drawString(x, y, f"Modelo: {resumen_dict['Modelo']}")
            y -= 0.6*cm
            cpdf.drawString(x, y, f"B (m): {resumen_dict['B (m)']}   L (m): {resumen_dict['L (m)']}   h (m): {resumen_dict['h (m)']}")
            y -= 0.6*cm
            cpdf.drawString(x, y, f"Área (m²): {resumen_dict['Área (m²)']}")
            y -= 0.6*cm
            cpdf.drawString(x, y, f"q_adm (kPa): {resumen_dict['q_adm (kPa)']}   q_req (kPa): {resumen_dict['q_req (kPa)']}   Utilización: {resumen_dict['Utilización q_req/q_adm']}")
            y -= 0.6*cm
            cpdf.drawString(x, y, f"Costo (S/): {resumen_dict['Costo (S/)']}")
            y -= 1.0*cm

            cpdf.setFont("Helvetica-Bold", 12)
            cpdf.drawString(x, y, "Recomendaciones")
            y -= 0.8*cm
            cpdf.setFont("Helvetica", 10)

            for line in recomendaciones_str.split("\n"):
                cpdf.drawString(x, y, line)
                y -= 0.5*cm
                if y < 2*cm:
                    cpdf.showPage()
                    y = hpdf - 2*cm
                    cpdf.setFont("Helvetica", 10)

            cpdf.showPage()
            cpdf.save()
            return buf.getvalue()

        rec_text = (
            f"• Utilización q_req/q_adm = {util:.3f}\n"
            f"{tips_text}\n"
            "• Referencias: Terzaghi & Peck; Meyerhof; Vesic."
        )
        pdf_bytes = pdf_bytes_from_summary(resumen, rec_text)
        colx2.download_button(
            "📄 Descargar reporte (PDF)",
            data=pdf_bytes,
            file_name="reporte_cimentaciones.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        # =========================
        # IMAGEN ADJUNTA (solo si hay)
        # =========================
        if uploaded is not None:
            st.subheader("Imagen adjunta")
            st.image(uploaded, caption="Croquis / Perfil estratigráfico", use_container_width=True)

else:
    st.info("Configura los parámetros a la izquierda y pulsa **Analizar y optimizar**.")
