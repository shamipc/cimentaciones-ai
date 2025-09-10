# app.py
# -----------------------------------------------------------------------------
# Optimización de Cimentaciones (vista compacta) – con barrido y GA
# - Modelos: Terzaghi (clásico) y Meyerhof (opcional simple)
# - Fix GA: ok -> bool(...) y protección NaN/Inf en fitness
# - Exports: Excel + PDF
# - Plotly pastel + esquema simple de zapata
# -----------------------------------------------------------------------------

import io
import math
import json
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px

# --- Exports
import xlsxwriter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

st.set_page_config(
    page_title="Optimización de Cimentaciones",
    page_icon="🧱",
    layout="wide"
)

# ===========================
# ==== UTILIDADES UI =========
# ===========================
PASTEL = px.colors.qualitative.Pastel

def pastel_layout(fig, title=None):
    fig.update_layout(
        template="plotly_white",
        title=title or "",
        title_font=dict(size=18),
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig

def badge_ok(text="Diseño óptimo encontrado"):
    st.markdown(
        f"""
        <div style="
            background:#e8fff2;border:1px solid #b6f0cf;border-radius:12px;
            padding:.35rem 0.8rem;display:inline-flex;gap:.5rem;align-items:center;">
            <span style="background:#14a44d;width:.6rem;height:.6rem;border-radius:50%;display:inline-block"></span>
            <b>{text}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===========================
# ==== MODELOS GEOTÉCNICOS ===
# ===========================
def nc_terzaghi(phi_deg):
    """Coeficientes Terzaghi (aprox)"""
    phi = np.radians(phi_deg)
    Nq = np.exp(np.pi*np.tan(phi)) * (np.tan(np.radians(45)+phi/2))**2
    if phi_deg == 0:
        Nc = 5.14
    else:
        Nc = (Nq - 1)/np.tan(phi)
    Ng = 2*(Nq+1)*np.tan(phi)
    return Nc, Nq, Ng

def q_admisible(modelo, c, phi, gamma, D, B, FS):
    """Capacidad admisible simple por Terzaghi o Meyerhof (simplificado)."""
    Nc, Nq, Ng = nc_terzaghi(phi)
    q_ult = c*Nc + gamma*D*Nq + 0.5*gamma*B*Ng

    if modelo.startswith("Meyerhof"):
        # Ajuste muy básico (ligeramente menor para B/L > 1)
        q_ult *= (0.95 if B > 1.5 else 1.0)

    q_adm = q_ult/FS
    return float(q_adm)

def q_requerida(N, B, L):
    """Presión requerida por carga N [kN] sobre base BxL [m] (en kPa)."""
    area = B*L
    if area <= 0:
        return np.inf
    return float(1000.0*N/area)  # kPa

def factibilidad(modelo, c, phi, gamma, D, FS, N, B, L, h):
    qadm = q_admisible(modelo, c, phi, gamma, D, B, FS)
    qreq = q_requerida(N, B, L)

    # Fuerza booleano de Python (evita ambigüedad numpy)
    try:
        ok = bool(qreq <= qadm)
    except Exception:
        ok = False

    # Evita NaN/Inf propagándose
    if not np.isfinite(qadm): qadm = 0.0
    if not np.isfinite(qreq): qreq = np.inf

    return ok, float(qadm), float(qreq)

def costo_aprox(concreto_Sm3, acero_Skg, B, L, h):
    """Costo muy simple: concreto (B·L·h) + acero (10 kg/m3, p. ej.)."""
    vol = B*L*h            # m3
    kg  = vol*10.0         # kg de acero (sup. mínima)
    return float(concreto_Sm3*vol + acero_Skg*kg)

# ===========================
# ==== OPTIMIZACIÓN ==========
# ===========================
def barrido(modelo, gamma, c, phi, D, FS, N, Bmin, Bmax, nB, Lmin, Lmax, nL, hmin, hmax, nh, concreto_Sm3, acero_Skg):
    Bs = np.linspace(Bmin, Bmax, nB)
    Ls = np.linspace(Lmin, Lmax, nL)
    hs = np.linspace(hmin, hmax, nh)

    rows = []
    for B in Bs:
        for L in Ls:
            for h in hs:
                ok, qadm, qreq = factibilidad(modelo, c, phi, gamma, D, FS, N, B, L, h)
                cost = costo_aprox(concreto_Sm3, acero_Skg, B, L, h)
                if ok:
                    rows.append((B, L, h, qadm, qreq, cost))
    df = pd.DataFrame(rows, columns=["B","L","h","q_adm","q_req","costo"])
    return df

def ga_optimizar(
    modelo, c, phi, gamma, D, FS, N,
    Bmin, Bmax, Lmin, Lmax, hmin, hmax,
    concreto_Sm3, acero_Skg,
    pop=60, gens=60, pm=0.15, seed=42
):
    rng = np.random.default_rng(seed)
    def init_ind():
        return np.array([
            rng.uniform(Bmin, Bmax),
            rng.uniform(Lmin, Lmax),
            rng.uniform(hmin, hmax),
        ], dtype=float)

    def mutate(ind):
        i = rng.integers(3)
        if i == 0: ind[0] = np.clip(ind[0] + rng.normal(0, (Bmax-Bmin)/10), Bmin, Bmax)
        elif i == 1: ind[1] = np.clip(ind[1] + rng.normal(0, (Lmax-Lmin)/10), Lmin, Lmax)
        else: ind[2] = np.clip(ind[2] + rng.normal(0, (hmax-hmin)/10), hmin, hmax)

    def crossover(a, b):
        w = rng.random(3)
        return w*a + (1-w)*b

    def fitness(ind):
        B, L, h = float(ind[0]), float(ind[1]), float(ind[2])
        ok, qadm, qreq = factibilidad(modelo, c, phi, gamma, D, FS, N, B, L, h)
        cost = float(costo_aprox(concreto_Sm3, acero_Skg, B, L, h))
        # penalización si no cumple o hay NaN
        if (not ok) or (not np.isfinite(cost)):
            gap = max(0.0, float(qreq - qadm)) if (np.isfinite(qreq) and np.isfinite(qadm)) else 1e3
            penalty = 1e6 + 1e4*gap
            return float(penalty), float(qadm), float(qreq), float(cost)
        return float(cost), float(qadm), float(qreq), float(cost)

    popu = np.array([init_ind() for _ in range(pop)], dtype=float)
    hist = []

    for _ in range(gens):
        # torneo (usa mismo rng para reproducibilidad)
        a, b = popu[rng.integers(pop)], popu[rng.integers(pop)]
        c_, d_ = popu[rng.integers(pop)], popu[rng.integers(pop)]
        fa = fitness(a)[0]; fb = fitness(b)[0]
        fc = fitness(c_)[0]; fd = fitness(d_)[0]
        p1 = a if fa < fb else b
        p2 = c_ if fc < fd else d_
        child = crossover(p1, p2)
        if rng.random() < pm: mutate(child)
        # reemplaza el peor
        costs = np.array([fitness(x)[0] for x in popu])
        worst = np.argmax(costs)
        popu[worst] = child
        best = popu[np.argmin(costs)]
        hist.append([*best, *fitness(best)])

    # devuelve el mejor
    costs = np.array([fitness(x)[0] for x in popu])
    best = popu[np.argmin(costs)]
    fa, qadm, qreq, cost = fitness(best)
    Bopt, Lopt, hopt = best
    df_hist = pd.DataFrame(hist, columns=["B","L","h","fa","qadm","qreq","cost"])
    df_hist["gen"] = np.arange(1, len(df_hist)+1)
    return float(Bopt), float(Lopt), float(hopt), float(cost), float(qreq), float(qadm), df_hist

# ===========================
# ==== EXPORTS ==============
# ===========================
def to_excel_bytes(df_sorted):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df_sorted.to_excel(writer, sheet_name="Candidatos", index=False)
    return bio.getvalue()

def to_pdf_bytes(datos, optimo, recomendaciones):
    bio = io.BytesIO()
    c = canvas.Canvas(bio, pagesize=A4)
    w, h = A4
    c.setTitle("Reporte de cimentación")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, h-2.5*cm, "Optimización de Cimentaciones (reporte)")

    c.setFont("Helvetica", 11)
    y = h-4.0*cm
    c.drawString(2*cm, y, "Datos de entrada:"); y -= 0.6*cm
    for k, v in datos.items():
        c.drawString(2.5*cm, y, f"• {k}: {v}")
        y -= 0.5*cm

    y -= 0.3*cm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(2*cm, y, "Óptimo:"); y -= 0.6*cm
    c.setFont("Helvetica", 11)
    for k, v in optimo.items():
        c.drawString(2.5*cm, y, f"• {k}: {v}")
        y -= 0.5*cm

    y -= 0.3*cm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(2*cm, y, "Recomendaciones:"); y -= 0.6*cm
    c.setFont("Helvetica", 11)
    for r in recomendaciones:
        c.drawString(2.5*cm, y, f"• {r}")
        y -= 0.5*cm

    c.showPage()
    c.save()
    return bio.getvalue()

# ===========================
# ==== UI LATERAL ===========
# ===========================
with st.sidebar:
    st.header("Parámetros de entrada")

    modelo = st.selectbox(
        "Modelo de capacidad",
        ["Terzaghi (recomendado)", "Meyerhof (simple)"],
        index=0
    )

    preset = st.selectbox(
        "Preset de suelo (rápido)",
        [
            "Arcilla blanda (γ=17, c=18, φ=0)",
            "Arena densa (γ=19, c=0, φ=35)",
            "Grava densa (γ=20, c=0, φ=40)",
            "Personalizado",
        ],
        index=0
    )

    # defaults coherentes según preset
    if preset.startswith("Arcilla"):
        gamma_d, c_d, phi_d = 17.0, 18.0, 0.0
    elif preset.startswith("Arena"):
        gamma_d, c_d, phi_d = 19.0, 0.0, 35.0
    elif preset.startswith("Grava"):
        gamma_d, c_d, phi_d = 20.0, 0.0, 40.0
    else:
        gamma_d, c_d, phi_d = 18.0, 10.0, 25.0

    gamma = st.number_input("Peso unitario γ (kN/m³)", value=gamma_d, step=0.5, format="%.2f")
    c = st.number_input("Cohesión c (kPa)", value=c_d, step=1.0, format="%.2f")
    phi = st.number_input("Ángulo de fricción φ (°)", value=phi_d, step=1.0, format="%.2f")
    D = st.number_input("Profundidad D (m)", value=1.50, step=0.1, format="%.2f")
    N = st.number_input("Carga N (kN)", value=1000.0, step=10.0, format="%.2f")
    FS = st.number_input("Factor de seguridad", value=2.50, min_value=1.1, step=0.1, format="%.2f")

    st.markdown("---")
    st.subheader("Costos")
    concreto_Sm3 = st.number_input("Concreto (S/ por m³)", value=650.0, step=10.0, format="%.2f")
    acero_Skg = st.number_input("Acero (S/ por kg)", value=5.50, step=0.1, format="%.2f")

    st.markdown("---")
    st.subheader("Rangos de diseño")
    Bmin, Bmax = st.slider("Base B (m)", 1.2, 4.0, (1.4, 3.0), step=0.1)
    nB = st.number_input("Resolución B (puntos)", value=25, min_value=5, step=1)

    Lmin, Lmax = st.slider("Largo L (m)", 1.2, 4.0, (1.6, 3.5), step=0.1)
    nL = st.number_input("Resolución L (puntos)", value=25, min_value=5, step=1)

    hmin, hmax = st.slider("Altura h (m)", 0.40, 1.20, (0.50, 1.00), step=0.05)
    nh = st.number_input("Resolución h (puntos)", value=8, min_value=3, step=1)

    st.markdown("---")
    sub_img = st.file_uploader("Sube un croquis / perfil del suelo (PNG/JPG)", type=["png","jpg","jpeg"])

    col_btn = st.columns(2)
    with col_btn[0]:
        run = st.button("🔎 Analizar y optimizar", use_container_width=True)
    with col_btn[1]:
        run_ga = st.button("🧬 Optimizar con GA", use_container_width=True)

# ===========================
# ==== CONTENIDO =============
# ===========================
st.title("Optimización de Cimentaciones")
st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta")

# Mensaje inicial corto
st.info("Ingresa **los datos** en la parte izquierda y pulsa **Analizar y optimizar** o **Optimizar con GA**.")

# Ejecutar
df = pd.DataFrame()
opt_txt = None

if run:
    with st.spinner("Calculando barrido de soluciones…"):
        df = barrido(
            modelo, gamma, c, phi, D, FS, N,
            Bmin, Bmax, nB, Lmin, Lmax, nL, hmin, hmax, nh,
            concreto_Sm3, acero_Skg
        )

    if df.empty:
        st.warning(
            "No se encontraron soluciones que cumplan la capacidad admisible. "
            "Prueba con **B y L mayores**, **φ** o **c** más altos, **FS** menor o **carga** menor."
        )
    else:
        df_sorted = df.sort_values("costo", ascending=True).reset_index(drop=True)
        best = df_sorted.iloc[0]
        badge_ok()
        st.subheader("Resultados (óptimo del barrido)")
        cols = st.columns(4)
        cols[0].metric("B (m)", f"{best.B:.2f}")
        cols[1].metric("L (m)", f"{best.L:.2f}")
        cols[2].metric("h (m)", f"{best.h:.2f}")
        cols[3].metric("Costo (S/)", f"{best.costo:,.0f}")

        # Comparación q
        fig1 = go.Figure()
        fig1.add_bar(x=["q_req"], y=[best.q_req], name="q_req", marker_color=PASTEL[2])
        fig1.add_bar(x=["q_adm"], y=[best.q_adm], name="q_adm", marker_color=PASTEL[3])
        pastel_layout(fig1, "q_req vs q_adm (óptimo)")
        st.plotly_chart(fig1, use_container_width=True)

        # Mapa candidatos (color=costo)
        fig2 = px.scatter(
            df_sorted, x="B", y="L",
            color="costo", color_continuous_scale="Tealgrn",
            size="h", opacity=0.75, height=420
        )
        pastel_layout(fig2, "Candidatos válidos (color=costo, tamaño=h)")
        st.plotly_chart(fig2, use_container_width=True)

        # Resumen y recomendaciones
        with st.expander("Resumen (JSON)"):
            st.code(
                json.dumps({
                    "Modelo": modelo.split(" ")[0],
                    "B (m)": round(float(best.B), 2),
                    "L (m)": round(float(best.L), 2),
                    "h (m)": round(float(best.h), 2),
                    "q_adm (kPa)": round(float(best.q_adm), 1),
                    "q_req (kPa)": round(float(best.q_req), 1),
                    "Costo (S/)": round(float(best.costo), 0),
                }, indent=2, ensure_ascii=False),
                language="json"
            )

        st.subheader("Recomendaciones")
        st.success("Buen diseño: margen suficiente entre capacidad y demanda.")
        st.info(f"Óptimo actual: S/ {best.costo:,.0f}. Evalúa **h ligeramente mayor** si buscas rigidez.")

        # Historial (barrido ordenado, las 10 mejores)
        st.subheader("Top 10 candidatos")
        st.dataframe(df_sorted.head(10), use_container_width=True)

        # Descargas
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Descargar Excel",
                data=to_excel_bytes(df_sorted),
                file_name="cimentaciones_candidatos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        with c2:
            datos = {
                "Modelo": modelo,
                "γ (kN/m³)": gamma, "c (kPa)": c, "φ (°)": phi, "D (m)": D,
                "FS": FS, "N (kN)": N
            }
            optimo = {
                "B (m)": round(float(best.B),2),
                "L (m)": round(float(best.L),2),
                "h (m)": round(float(best.h),2),
                "q_adm (kPa)": round(float(best.q_adm),1),
                "q_req (kPa)": round(float(best.q_req),1),
                "Costo (S/)": round(float(best.costo),0)
            }
            recs = [
                "Si necesitas menor asentamiento, incrementa h o considera mayor B·L.",
                "Verifica capacidad por punzonamiento y asentamientos con tu método habitual."
            ]
            st.download_button(
                "📄 Descargar reporte (PDF)",
                data=to_pdf_bytes(datos, optimo, recs),
                file_name="reporte_cimentacion.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        # Esquema simple
        st.subheader(f"Esquema (h = {best.h:.2f} m)")
        fig_s = go.Figure()
        fig_s.add_shape(type="rect", x0=0, x1=best.B, y0=0, y1=best.h, line=dict(color="#4F6D7A"), fillcolor="#C8E6C9")
        fig_s.update_yaxes(range=[0, max(best.h*2, 1.5)])
        fig_s.update_xaxes(range=[-0.1, best.B+0.1])
        pastel_layout(fig_s, "")
        st.plotly_chart(fig_s, use_container_width=True)

elif run_ga:
    with st.spinner("Buscando óptimo con Algoritmo Genético…"):
        Bopt, Lopt, hopt, costopt, qreq, qadm, df_hist = ga_optimizar(
            modelo, c, phi, gamma, D, FS, N,
            Bmin, Bmax, Lmin, Lmax, hmin, hmax,
            concreto_Sm3, acero_Skg,
            pop=80, gens=80, pm=0.15, seed=42
        )
    ok, _, _ = factibilidad(modelo, c, phi, gamma, D, FS, N, Bopt, Lopt, hopt)
    if not ok:
        st.warning("El GA no encontró un diseño factible. Aumenta rangos o reduce FS/carga y vuelve a intentar.")
    else:
        badge_ok("Óptimo (Algoritmo Genético)")
        cols = st.columns(4)
        cols[0].metric("B (m)", f"{Bopt:.2f}")
        cols[1].metric("L (m)", f"{Lopt:.2f}")
        cols[2].metric("h (m)", f"{hopt:.2f}")
        cols[3].metric("Costo (S/)", f"{costopt:,.0f}")

        # progreso GA
        figg = px.line(df_hist, x="gen", y="cost", color_discrete_sequence=[PASTEL[6]], height=380)
        pastel_layout(figg, "Convergencia del GA (costo por generación)")
        st.plotly_chart(figg, use_container_width=True)

