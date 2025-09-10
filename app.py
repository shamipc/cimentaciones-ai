
import io
import json
from math import tan, pi, exp
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# -------------------------- CONFIG & STYLE --------------------------
st.set_page_config(
    page_title="Optimización de Cimentaciones",
    page_icon="🧱",
    layout="wide",
)

PASTEL = {
    "bg": "#f7fafc",
    "card": "#ffffff",
    "primary": "#2563eb",        # azul
    "ok": "#16a34a",
    "warn": "#c084fc",           # lila suave
    "bar1": "#8eb5ff",
    "bar2": "#51d0b1",
}

st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 1.2rem;}}
      .stMetric > div > div > span {{font-size: 18px!important;}}
      .stMetric > div > div {{font-size: 36px!important;}}
      .metric-card {{
        background: {PASTEL['card']};
        border: 1px solid #e5e7eb; border-radius:16px; padding: 18px 22px;
        box-shadow: 0px 5px 12px rgba(0,0,0,0.04);
      }}
      .pill {{
        display:inline-block; padding:8px 14px; border-radius:999px; font-weight:600;
      }}
      .pill-ok {{ background:{PASTEL['ok']}; color:white; }}
      .pill-warn {{ background:{PASTEL['warn']}; color:white; }}
      .callout {{
        background:#eef2ff; border-radius:12px; padding:12px 14px;
        border:1px dashed #94a3b8;
      }}
      .title-accent {{ color:{PASTEL['primary']};}}
      .kpi-title {{color:#475569; font-size:14px; margin-bottom:6px;}}
      .kpi-value {{font-size:32px; font-weight:800; color:#0f172a;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------- HELPERS --------------------------

def deg2rad(x):
    return np.deg2rad(x)

def bearing_capacity_factors(phi_deg):
    """Nc, Nq, Nγ - factores (aprox. Terzaghi/Meyerhof)"""
    phi = deg2rad(phi_deg)
    if phi_deg < 0: phi = 0
    Nq = np.exp(np.pi * np.tan(phi)) * np.tan(np.radians(45) + phi/2) ** 2
    Nc = (Nq - 1) / np.tan(phi) if phi_deg > 0 else 5.7  # valor típico para φ≈0
    Ngamma = 2 * (Nq + 1) * np.tan(phi)
    return Nc, Nq, Ngamma

def qu_terzaghi(gamma, c, phi_deg, B, D):
    """Capacidad última (kPa). q = γD (sobrecarga)."""
    Nc, Nq, Ng = bearing_capacity_factors(phi_deg)
    q = gamma * D
    return c*Nc + q*Nq + 0.5*gamma*B*Ng

def optimize(gamma, c, phi_deg, D, N_kN, FS, B_range, L_range, h_range,
             model="Terzaghi", concrete=650, steel=5.5):
    """
    Devuelve:
      df_valid: soluciones válidas (DataFrame)
      best: fila con mejor costo (Series) o None
    """
    # grid
    B = np.linspace(*B_range)
    L = np.linspace(*L_range)
    H = np.linspace(*h_range)

    rows = []
    for b in B:
        for l in L:
            A = b*l                         # área (m2)
            q_req = (N_kN*1000)/A           # kPa
            qu = qu_terzaghi(gamma, c, phi_deg, b, D)
            q_adm = qu / FS

            for h in H:
                vol = b*l*h
                cost = vol*concrete + (b+l)*2*h*steel  # sencillo
                ok = q_adm >= q_req
                rows.append((b, l, h, q_adm, q_req, cost, ok))

    df = pd.DataFrame(rows, columns=["B","L","h","q_adm","q_req","costo","ok"])
    df_valid = df[df["ok"]].copy().sort_values("costo", ascending=True)
    best = df_valid.iloc[0] if len(df_valid) else None
    return df_valid, best

def render_kpi(col, title, value):
    with col:
        st.markdown(f"<div class='metric-card'><div class='kpi-title'>{title}</div>"
                    f"<div class='kpi-value'>{value}</div></div>", unsafe_allow_html=True)

def make_excel(df_valid, best_row, meta):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([meta]).T.to_excel(writer, index=True, header=False, sheet_name="Parámetros")
        if best_row is not None:
            pd.DataFrame([best_row]).to_excel(writer, sheet_name="Óptimo", index=False)
        df_valid.to_excel(writer, sheet_name="Soluciones", index=False)
    return output.getvalue()

def make_pdf(best_row, meta):
    buffer = io.BytesIO()
    cpdf = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4

    cpdf.setFont("Helvetica-Bold", 14)
    cpdf.drawString(40, h-60, "Optimización de Cimentaciones - Reporte")

    cpdf.setFont("Helvetica", 11)
    y = h-90
    for k, v in meta.items():
        cpdf.drawString(40, y, f"{k}: {v}")
        y -= 16

    if best_row is not None:
        y -= 10
        cpdf.setFont("Helvetica-Bold", 12)
        cpdf.drawString(40, y, "Diseño óptimo:")
        y -= 18
        cpdf.setFont("Helvetica", 11)
        cpdf.drawString(40, y, f"B = {best_row['B']:.2f} m, L = {best_row['L']:.2f} m, h = {best_row['h']:.2f} m")
        y -= 16
        cpdf.drawString(40, y, f"q_req = {best_row['q_req']:.1f} kPa, q_adm = {best_row['q_adm']:.1f} kPa")
        y -= 16
        cpdf.drawString(40, y, f"Costo = S/ {best_row['costo']:.0f}")

    cpdf.showPage()
    cpdf.save()
    buffer.seek(0)
    return buffer.read()

def sketch(b, l, h):
    fig, ax = plt.subplots(figsize=(4, 2.2), dpi=130)
    ax.add_patch(plt.Rectangle((0, 0), b, l, edgecolor="#0f172a", facecolor="#c7d2fe", lw=2))
    ax.set_xlim(0, max(b*1.15, 0.5))
    ax.set_ylim(0, max(l*1.15, 0.5))
    ax.set_aspect("equal")
    ax.set_xlabel("B (m)")
    ax.set_ylabel("L (m)")
    ax.set_title(f"Esquema (h={h:.2f} m)")
    ax.grid(alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# -------------------------- SIDEBAR --------------------------

st.sidebar.header("Parámetros de entrada")

model = st.sidebar.selectbox(
    "Modelo de capacidad",
    ["Terzaghi (recomendado)"],
    index=0
)

# Presets de suelo
SOILS = {
    "Arcilla blanda (γ=17, c=18, φ=0)": (17.0, 18.0, 0.0),
    "Arena densa (γ=20, c=0, φ=35)": (20.0, 0.0, 35.0),
    "Arcilla media (γ=18.5, c=30, φ=5)": (18.5, 30.0, 5.0),
}
preset = st.sidebar.selectbox("Preset de suelo (rápido)", list(SOILS.keys()), index=0)
gamma_default, c_default, phi_default = SOILS[preset]

gamma = st.sidebar.number_input("Peso unitario γ (kN/m³)", 10.0, 25.0, gamma_default, 0.25)
c = st.sidebar.number_input("Cohesión c (kPa)", 0.0, 300.0, c_default, 1.0)
phi = st.sidebar.number_input("Ángulo de fricción φ (°)", 0.0, 45.0, phi_default, 0.5)
D = st.sidebar.number_input("Profundidad D (m)", 0.0, 5.0, 1.50, 0.05)
N_kN = st.sidebar.number_input("Carga N (kN)", 50.0, 10000.0, 1000.0, 10.0)
FS = st.sidebar.number_input("Factor de seguridad", 1.5, 5.0, 2.5, 0.1)

st.sidebar.subheader("Costos")
concrete = st.sidebar.number_input("Concreto (S/ por m³)", 300.0, 2000.0, 650.0, 10.0)
steel = st.sidebar.number_input("Acero (S/ por kg)", 3.0, 12.0, 5.50, 0.1)

st.sidebar.subheader("Rangos de diseño")
B_range = (1.0, 3.8, 25)   # min, max, puntos
L_range = (1.0, 3.8, 25)
h_range = (0.5, 1.2, 8)

st.sidebar.subheader("Adjuntos")
user_img = st.sidebar.file_uploader("Sube una imagen (perfil del suelo, croquis)", type=["png", "jpg", "jpeg"])

# Botón de correr
if st.sidebar.button("🔎 Analizar y optimizar"):
    st.session_state.run = True
    st.rerun()  # <- reemplazo correcto

# -------------------------- TÍTULO --------------------------

st.markdown(
    "<h1 class='title-accent'>Optimización de Cimentaciones</h1>",
    unsafe_allow_html=True,
)
st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta")

# -------------------------- LÓGICA --------------------------

run = st.session_state.get("run", False)
relax = st.session_state.get("relax", False)

if run:
    df_valid, best = optimize(
        gamma, c, phi, D, N_kN, FS,
        (B_range[0], B_range[1], B_range[2]),
        (L_range[0], L_range[1], L_range[2]),
        (h_range[0], h_range[1], h_range[2]),
        model="Terzaghi", concrete=concrete, steel=steel
    )

    if best is None:
        st.info(
            "No se encontraron soluciones que cumplan la capacidad admisible. "
            "Prueba con **B y L mayores**, **φ o c más altos**, **FS menor** o **carga menor**."
        )
        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("🪄 Auto-ajustar y reintentar"):
                st.session_state.run = True
                st.session_state.relax = True
                st.rerun()  # <- reemplazo correcto
        with colB:
            st.markdown(
                "<div class='callout'>Sugerencia rápida: prueba con B y L en el rango 1.4–4.0 m.</div>",
                unsafe_allow_html=True,
            )

    else:
        # Métricas principales
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        render_kpi(c1, "B (m)", f"{best['B']:.2f}")
        render_kpi(c2, "L (m)", f"{best['L']:.2f}")
        render_kpi(c3, "h (m)", f"{best['h']:.2f}")
        render_kpi(c4, "Costo (S/)", f"{best['costo']:.0f}")

        st.markdown("<span class='pill pill-ok'>Diseño óptimo encontrado</span>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        # ----------------- Gráficos (Plotly con colores suaves) -----------------
        left, right = st.columns([1, 1])

        with left:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=["q_req"], y=[best["q_req"]],
                name="q_req", marker_color=PASTEL["bar1"], text=[f"{best['q_req']:.1f}"],
                textposition="outside"
            ))
            fig_bar.add_trace(go.Bar(
                x=["q_adm"], y=[best["q_adm"]],
                name="q_adm", marker_color=PASTEL["bar2"], text=[f"{best['q_adm']:.1f}"],
                textposition="outside"
            ))
            fig_bar.update_layout(
                title="q_req vs q_adm (óptimo)",
                yaxis_title="kPa", bargap=0.35, template="plotly_white",
                legend_title_text="", height=420
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with right:
            df_show = df_valid.copy()
            fig_sc = px.scatter(
                df_show, x="B", y="L", color="costo", size="h",
                title="Candidatos válidos (color = Costo, tamaño = h)",
                template="plotly_white", color_continuous_scale=px.colors.sequential.Tealgrn,
                height=420,
            )
            fig_sc.update_traces(marker=dict(line=dict(width=0)))
            st.plotly_chart(fig_sc, use_container_width=True)

        # ----------------- Resumen JSON -----------------
        st.subheader("Resumen")
        summary = {
            "Modelo": "Terzaghi",
            "B (m)": float(best["B"]),
            "L (m)": float(best["L"]),
            "h (m)": float(best["h"]),
            "q_adm (kPa)": float(best["q_adm"]),
            "q_req (kPa)": float(best["q_req"]),
            "Costo (S/)": float(best["costo"]),
        }
        st.code(json.dumps(summary, indent=2, ensure_ascii=False))

        # ----------------- Recomendaciones -----------------
        st.subheader("Recomendaciones")
        st.markdown(
            f"""
            - ✅ **Buen diseño**: margen suficiente entre capacidad y demanda.
            - 💡 **Óptimo actual**: S/ **{best['costo']:.0f}**. Evalúa alternativas con **h** ligeramente mayor si buscas rigidez.
            - 📐 Verifica asentamientos y punzonamiento si aplica.
            """,
        )
        with st.expander("Referencias clave"):
            st.markdown(
                "- Terzaghi & Peck (1997) – Capacidad portante clásica.  \n"
                "- Meyerhof (1963); Vesic (1973) – Factores N y correcciones.  \n"
                "- Normas locales para factores de reducción y FS."
            )

        # ----------------- Tabla de soluciones & descargas -----------------
        st.subheader("Soluciones válidas (Top 200 por costo)")
        df_top = df_valid.sort_values("costo").head(200).reset_index(drop=True)
        st.dataframe(df_top, use_container_width=True, hide_index=True)

        meta = {
            "γ (kN/m³)": gamma,
            "c (kPa)": c,
            "φ (°)": phi,
            "D (m)": D,
            "N (kN)": N_kN,
            "FS": FS,
            "Concreto": concrete,
            "Acero": steel,
        }

        colx, coly = st.columns([1, 1])
        with colx:
            xls_bytes = make_excel(df_top, best, meta)
            st.download_button(
                "📥 Descargar Excel",
                data=xls_bytes,
                file_name="reporte_cimentacion.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with coly:
            pdf_bytes = make_pdf(best, meta)
            st.download_button(
                "📄 Descargar reporte (PDF)",
                data=pdf_bytes,
                file_name="reporte_cimentacion.pdf",
                mime="application/pdf",
            )

        # ----------------- Esquema & Adjuntos -----------------
        st.subheader("Esquema del óptimo")
        sketch(float(best["B"]), float(best["L"]), float(best["h"]))

        if user_img is not None:
            st.subheader("Imagen adjunta")
            st.image(user_img, caption="Adjunto del usuario", use_container_width=True)

else:
    st.markdown(
        "<div class='callout'>Configura los parámetros a la izquierda y pulsa "
        "<b>Analizar y optimizar</b>.</div>",
        unsafe_allow_html=True,
    )

     

