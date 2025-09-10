import io
import json
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
st.set_page_config(page_title="Optimización de Cimentaciones", page_icon="🧱", layout="wide")

PASTEL = {
    "bg": "#f7fafc",
    "card": "#ffffff",
    "primary": "#2563eb",
    "ok": "#16a34a",
    "warn": "#c084fc",
    "bar1": "#8eb5ff",
    "bar2": "#51d0b1",
}
st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 1.1rem;}}
      .metric-card {{
        background: {PASTEL['card']};
        border: 1px solid #e5e7eb; border-radius:16px; padding: 18px 22px;
        box-shadow: 0px 5px 12px rgba(0,0,0,0.04);
      }}
      .kpi-title {{color:#475569; font-size:14px; margin-bottom:6px;}}
      .kpi-value {{font-size:32px; font-weight:800; color:#0f172a;}}
      .pill {{ display:inline-block; padding:8px 14px; border-radius:999px; font-weight:600; }}
      .pill-ok {{ background:{PASTEL['ok']}; color:white; }}
      .callout {{ background:#eef2ff; border-radius:12px; padding:12px 14px; border:1px dashed #94a3b8; }}
      .title-accent {{ color:{PASTEL['primary']};}}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------- HELPERS --------------------------
def deg2rad(x): return np.deg2rad(x)

def bearing_capacity_factors(phi_deg):
    """Nc, Nq, Nγ (aprox. Terzaghi/Meyerhof)"""
    phi = deg2rad(max(phi_deg, 0))
    if phi_deg <= 0:
        Nq = 1.0
        Nc = 5.7
        Ngamma = 0.0
    else:
        Nq = np.exp(np.pi * np.tan(phi)) * np.tan(np.radians(45) + phi/2) ** 2
        Nc = (Nq - 1) / np.tan(phi)
        Ngamma = 2 * (Nq + 1) * np.tan(phi)
    return Nc, Nq, Ngamma

def qu_terzaghi(gamma, c, phi_deg, B, D):
    q = gamma * D
    Nc, Nq, Ng = bearing_capacity_factors(phi_deg)
    return c*Nc + q*Nq + 0.5*gamma*B*Ng

def optimize(gamma, c, phi_deg, D, N_kN, FS,
             B_range, L_range, H_range,
             locked_LB=False, ratio_LB=1.0,
             concrete=650, steel=5.5):
    """
    Rango: (min, max, puntos)
    locked_LB=True  -> L = ratio_LB * B (se respeta L_range: si no cae, se descarta)
    """
    B = np.linspace(*B_range)
    if not locked_LB:
        L = np.linspace(*L_range)
    H = np.linspace(*H_range)

    rows = []
    for b in B:
        if locked_LB:
            l_list = [b * ratio_LB]
        else:
            l_list = np.linspace(*L_range)

        for l in l_list:
            if not (L_range[0] <= l <= L_range[1]):
                continue
            A = b * l
            q_req = (N_kN * 1000) / max(A, 1e-9)
            qu = qu_terzaghi(gamma, c, phi_deg, b, D)
            q_adm = qu / FS

            for h in H:
                vol = b * l * h
                cost = vol * concrete + (b + l) * 2 * h * steel  # costo simple
                rows.append((b, l, h, q_adm, q_req, cost, q_adm >= q_req))

    df = pd.DataFrame(rows, columns=["B", "L", "h", "q_adm", "q_req", "costo", "ok"])
    df_valid = df[df["ok"]].copy().sort_values("costo", ascending=True)
    best = df_valid.iloc[0] if len(df_valid) else None
    return df_valid, best

def render_kpi(col, title, value):
    with col:
        st.markdown(
            f"<div class='metric-card'><div class='kpi-title'>{title}</div>"
            f"<div class='kpi-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )

def make_excel(df_valid, best_row, meta):
    from openpyxl import Workbook  # aseguramos import disponible
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([meta]).T.to_excel(writer, index=True, header=False, sheet_name="Parámetros")
        if best_row is not None:
            pd.DataFrame([best_row]).to_excel(writer, sheet_name="Óptimo", index=False)
        df_valid.to_excel(writer, sheet_name="Soluciones", index=False)
    return output.getvalue()

def make_pdf(best_row, meta):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, h - 60, "Optimización de Cimentaciones - Reporte")

    pdf.setFont("Helvetica", 11)
    y = h - 90
    for k, v in meta.items():
        pdf.drawString(40, y, f"{k}: {v}")
        y -= 16

    if best_row is not None:
        y -= 8
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(40, y, "Óptimo:")
        y -= 18
        pdf.setFont("Helvetica", 11)
        pdf.drawString(40, y, f"B={best_row['B']:.2f} m  L={best_row['L']:.2f} m  h={best_row['h']:.2f} m")
        y -= 16
        pdf.drawString(40, y, f"q_req={best_row['q_req']:.1f} kPa  q_adm={best_row['q_adm']:.1f} kPa")
        y -= 16
        pdf.drawString(40, y, f"Costo=S/ {best_row['costo']:.0f}")

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.read()

def sketch(b, l, h):
    fig, ax = plt.subplots(figsize=(4, 2.2), dpi=130)
    ax.add_patch(plt.Rectangle((0, 0), b, l, edgecolor="#0f172a", facecolor="#c7d2fe", lw=2))
    ax.set_xlim(0, max(b * 1.15, 0.5))
    ax.set_ylim(0, max(l * 1.15, 0.5))
    ax.set_aspect("equal")
    ax.set_xlabel("B (m)")
    ax.set_ylabel("L (m)")
    ax.set_title(f"Esquema (h={h:.2f} m)")
    ax.grid(alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def expand_ranges(B_range, L_range, H_range):
    B2 = (B_range[0], min(B_range[1] * 1.25, 5.0), min(B_range[2] + 6, 60))
    L2 = (L_range[0], min(L_range[1] * 1.25, 5.0), min(L_range[2] + 6, 60))
    H2 = (H_range[0], min(H_range[1] * 1.15, 2.0), min(H_range[2] + 2, 30))
    return B2, L2, H2

# -------------------------- SIDEBAR --------------------------
st.sidebar.header("Parámetros de entrada")

model = st.sidebar.selectbox("Modelo de capacidad", ["Terzaghi (recomendado)"], index=0)

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
# sliders de rango + resolución
B_minmax = st.sidebar.slider("Base B (m)", 0.8, 5.0, (1.2, 3.8), 0.05)
n_B = st.sidebar.number_input("Resolución B (puntos)", 5, 60, 25, 1)
L_minmax = st.sidebar.slider("Largo L (m)", 0.8, 5.0, (1.2, 3.8), 0.05)
n_L = st.sidebar.number_input("Resolución L (puntos)", 5, 60, 25, 1)
H_minmax = st.sidebar.slider("Altura h (m)", 0.4, 2.0, (0.6, 1.2), 0.02)
n_H = st.sidebar.number_input("Resolución h (puntos)", 3, 30, 8, 1)

# Opción para fijar L/B
locked_LB = st.sidebar.checkbox("Fijar relación L/B", value=False)
ratio_LB = st.sidebar.number_input("Relación L/B", 0.5, 4.0, 1.0, 0.05, disabled=not locked_LB)

# Guardamos rangos actuales en sesión (sirve para auto-ajuste)
st.session_state.setdefault("B_range", (B_minmax[0], B_minmax[1], int(n_B)))
st.session_state.setdefault("L_range", (L_minmax[0], L_minmax[1], int(n_L)))
st.session_state.setdefault("H_range", (H_minmax[0], H_minmax[1], int(n_H)))

# Actualizamos si el usuario mueve los sliders
st.session_state["B_range"] = (B_minmax[0], B_minmax[1], int(n_B))
st.session_state["L_range"] = (L_minmax[0], L_minmax[1], int(n_L))
st.session_state["H_range"] = (H_minmax[0], H_minmax[1], int(n_H))

st.sidebar.subheader("Adjuntos")
user_img = st.sidebar.file_uploader("Sube una imagen (perfil del suelo, croquis)", type=["png", "jpg", "jpeg"])

run_clicked = st.sidebar.button("🔎 Analizar y optimizar")

# -------------------------- TÍTULO --------------------------
st.markdown("<h1 class='title-accent'>Optimización de Cimentaciones</h1>", unsafe_allow_html=True)
st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta")

# -------------------------- LÓGICA --------------------------
if run_clicked:
    st.session_state["run"] = True
    st.session_state["auto"] = False
    st.rerun()

run = st.session_state.get("run", False)
auto = st.session_state.get("auto", False)

if run:
    B_range = st.session_state["B_range"]
    L_range = st.session_state["L_range"]
    H_range = st.session_state["H_range"]

    df_valid, best = optimize(
        gamma, c, phi, D, N_kN, FS,
        B_range, L_range, H_range,
        locked_LB=locked_LB, ratio_LB=ratio_LB,
        concrete=concrete, steel=steel
    )

    if (best is None) and (not auto):
        # pantalla de "no hay soluciones" + botón autoajuste
        st.info(
            "No se encontraron soluciones que cumplan la capacidad admisible. "
            "Prueba con **B y L mayores**, **φ o c más altos**, **FS menor** o **carga menor**."
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("🪄 Auto-ajustar y reintentar"):
                st.session_state["auto"] = True
                st.rerun()
        with c2:
            st.markdown(
                "<div class='callout'>Sugerencia rápida: prueba con B y L en el rango 1.4–4.0 m.</div>",
                unsafe_allow_html=True,
            )

    else:
        # Si sigue sin soluciones y auto=True, expandimos rangos temporalmente
        if (best is None) and auto:
            Bx, Lx, Hx = expand_ranges(B_range, L_range, H_range)
            df_valid, best = optimize(
                gamma, c, phi, D, N_kN, FS, Bx, Lx, Hx,
                locked_LB=locked_LB, ratio_LB=ratio_LB,
                concrete=concrete, steel=steel
            )
            if best is None:
                st.warning("Con el auto-ajuste tampoco se hallaron soluciones. Amplía rangos o reduce FS/carga.")
                st.stop()
            else:
                st.caption("✅ Se aplicó auto-ajuste de rangos para encontrar una solución factible.")

        # ---------- MÉTRICAS ----------
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        render_kpi(c1, "B (m)", f"{best['B']:.2f}")
        render_kpi(c2, "L (m)", f"{best['L']:.2f}")
        render_kpi(c3, "h (m)", f"{best['h']:.2f}")
        render_kpi(c4, "Costo (S/)", f"{best['costo']:.0f}")
        st.markdown("<span class='pill pill-ok'>Diseño óptimo encontrado</span>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        # ---------- GRÁFICOS ----------
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
                title="q_req vs q_adm (óptimo)", yaxis_title="kPa",
                bargap=0.35, template="plotly_white", legend_title_text="", height=420
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

        # ---------- RESUMEN ----------
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

        # ---------- RECOMENDACIONES ----------
        st.subheader("Recomendaciones")
        st.markdown(
            f"""
            - ✅ **Buen diseño**: margen suficiente entre capacidad y demanda.
            - 💡 **Óptimo actual**: S/ **{best['costo']:.0f}**. Evalúa **h** un poco mayor si buscas más rigidez.
            - 📐 Revisa asentamientos y punzonamiento según norma local.
            """
        )
        with st.expander("Referencias clave"):
            st.markdown(
                "- Terzaghi & Peck (1997) – Capacidad portante clásica.  \n"
                "- Meyerhof (1963); Vesic (1973) – Factores N y correcciones.  \n"
                "- Normativas locales para FS y coeficientes de reducción."
            )

        # ---------- TABLA + DESCARGAS ----------
        st.subheader("Soluciones válidas (Top 200 por costo)")
        df_top = df_valid.sort_values("costo").head(200).reset_index(drop=True)
        st.dataframe(df_top, use_container_width=True, hide_index=True)

        meta = {
            "γ (kN/m³)": gamma, "c (kPa)": c, "φ (°)": phi, "D (m)": D, "N (kN)": N_kN,
            "FS": FS, "Concreto": concrete, "Acero": steel,
            "B rango": B_range, "L rango": L_range, "h rango": H_range,
            "Relación L/B": f"{ratio_LB:.2f}" if locked_LB else "Libre",
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

        # ---------- ESQUEMA + ADJUNTO ----------
        st.subheader("Esquema del óptimo")
        sketch(float(best["B"]), float(best["L"]), float(best["h"]))

        if user_img is not None:
            st.subheader("Imagen adjunta")
            st.image(user_img, caption="Adjunto del usuario", use_container_width=True)

else:
    st.markdown(
        "<div class='callout'>Configura los rangos de <b>B, L y h</b> a la izquierda "
        "y pulsa <b>Analizar y optimizar</b>.</div>",
        unsafe_allow_html=True,
    )


