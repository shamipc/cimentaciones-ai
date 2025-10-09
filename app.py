   # app.py
# ------------------------------------------------------------
# Optimización de Cimentaciones Superficiales – UI + nombres claros
# ------------------------------------------------------------
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ===================== Config & Estilo ======================
st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")
st.markdown(
    """
<style>
h1 {font-size: 28px !important; margin-bottom: .4rem;}
h2 {font-size: 22px !important; margin-bottom: .4rem;}
h3 {font-size: 18px !important; margin-bottom: .3rem;}
section.main .block-container {padding-top: 1rem; padding-bottom: 1rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ===================== Defaults =============================
DEFAULTS = dict(
    # --- Suelo (kN, m, kPa) ---
    gamma=18.0, c=25.0, phi=30.0, Es=15000.0, nu=0.30,
    nivel_freatico=2.0,
    # --- Cargas (kN, kN·m) ---
    N=1000.0, Mx=10.0, My=10.0,
    # --- Materiales/costos ---
    fc=21.0, fy=420.0, recubrimiento=0.05,
    concreto_Sm3=650.0, acero_Skg=5.50, excav_Sm3=80.0,
    # --- Reglas de diseño ---
    D=1.5, FS=2.5, asent_max=0.025,
    # --- Rango de búsqueda ---
    B_min=1.0, B_max=4.0, L_min=1.0, L_max=4.0, h_min=0.5, h_max=1.5,
    nB=30, nL=30, nh=12,
    # --- Modelo capacidad ---
    modelo="Meyerhof",
    # --- Acero (kg/m3) para costo ---
    acero_kg_por_m3=60.0,
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===================== Funciones geotécnicas ================
def bearing_capacity_factors(phi_deg: float):
    """Nc, Nq, Ngamma (Meyerhof) con phi en grados."""
    phi = math.radians(phi_deg)
    if phi < 1e-6:
        Nq = 1.0
        Nc = 5.14
        Ng = 0.0
    else:
        Nq = math.e ** (math.pi * math.tan(phi)) * (math.tan(math.pi / 4.0 + phi / 2.0)) ** 2
        Nc = (Nq - 1.0) / math.tan(phi)
        Ng = 2.0 * (Nq + 1.0) * math.tan(phi)
    return Nc, Nq, Ng

def shape_factors(modelo: str):
    if "Terzaghi" in modelo:  return 1.3, 1.2, 0.8
    if "Meyerhof" in modelo:  return 1.3, 1.2, 1.0
    return 1.0, 1.0, 1.0  # Hansen/Genérico

def qult(modelo: str, gamma: float, c: float, phi_deg: float, B: float, D: float):
    """Capacidad última a cortante general: cNc + qNq + 0.5*γ*B*Ng."""
    Nc, Nq, Ng = bearing_capacity_factors(phi_deg)
    sc, sq, sg = shape_factors(modelo)
    q = gamma * D
    return c * Nc * sc + q * Nq * sq + 0.5 * gamma * B * Ng * sg

def qult_corr(modelo, gamma, c, phi, B, D, nivel_freatico):
    """Corrección simple: si base por debajo del N.F., reduce γ efectivo."""
    gamma_eff = gamma if D >= nivel_freatico else 0.5 * gamma
    return qult(modelo, gamma_eff, c, phi, B, D)

def q_required(N: float, Mx: float, My: float, B: float, L: float):
    """
    Presión requerida con excentricidades (área efectiva).
    e_x = Mx/N, e_y = My/N.  B' = B - 2|e_x| ; L' = L - 2|e_y|
    """
    if N <= 0:
        return float("inf"), 0.0, 0.0, 0.0, 0.0
    e_x, e_y = (Mx / N, My / N)
    B_eff = B - 2.0 * abs(e_x)
    L_eff = L - 2.0 * abs(e_y)
    if (B_eff <= 0.0) or (L_eff <= 0.0):
        return float("inf"), e_x, e_y, B_eff, L_eff
    return N / (B_eff * L_eff), e_x, e_y, B_eff, L_eff

def costo_total(B, L, h, concreto_Sm3, acero_Skg, excav_Sm3, D, acero_kg_por_m3):
    vol = B * L * h
    acero_kg = acero_kg_por_m3 * vol
    excav = B * L * D
    return (vol * concreto_Sm3) + (acero_kg * acero_Skg) + (excav * excav_Sm3)

def evaluar(B, L, h, ss):
    q_req, ex, ey, B_eff, L_eff = q_required(ss.N, ss.Mx, ss.My, B, L)
    q_ult = qult_corr(ss.modelo, ss.gamma, ss.c, ss.phi, B, ss.D, ss.nivel_freatico)
    q_adm = q_ult / ss.FS
    margen = (q_adm - q_req) if np.isfinite(q_req) else -1e9
    cumple = (q_adm >= q_req) and np.isfinite(q_req)
    costo = costo_total(B, L, h, ss.concreto_Sm3, ss.acero_Skg, ss.excav_Sm3, ss.D, ss.acero_kg_por_m3)
    return {
        "B": B, "L": L, "h": h, "B_eff": B_eff, "L_eff": L_eff,
        "q_req": q_req, "q_adm": q_adm, "q_ult": q_ult, "margen": margen,
        "ex": ex, "ey": ey, "cumple": cumple, "costo": costo
    }

# ===================== UI – Entradas =========================
st.title("Optimización de Cimentaciones Superficiales")

st.markdown("### 🧪 Propiedades de suelo")
c1, c2, c3 = st.columns(3)
with c1:
    st.session_state.gamma = st.number_input("γ (kN/m³)", 10.0, 25.0, st.session_state.gamma, 0.5)
    st.session_state.c = st.number_input("Cohesión c (kPa)", 0.0, 500.0, st.session_state.c, 1.0)
with c2:
    st.session_state.phi = st.number_input("φ (°)", 0.0, 45.0, st.session_state.phi, 1.0)
    st.session_state.Es = st.number_input("Módulo Eₛ (kPa)", 1_000.0, 200_000.0, st.session_state.Es, 1000.0)
with c3:
    st.session_state.nu = st.number_input("ν (Poisson)", 0.1, 0.5, st.session_state.nu, 0.05)
    st.session_state.nivel_freatico = st.number_input("Nivel freático (m)", 0.0, 20.0, st.session_state.nivel_freatico, 0.5)

st.markdown("### 🏗️ Cargas")
cc1, cc2, cc3 = st.columns(3)
with cc1:
    st.session_state.N = st.number_input("N (kN)", 1.0, 100000.0, st.session_state.N, 10.0)
with cc2:
    st.session_state.Mx = st.number_input("Mx (kN·m)", 0.0, 1e6, st.session_state.Mx, 5.0)
with cc3:
    st.session_state.My = st.number_input("My (kN·m)", 0.0, 1e6, st.session_state.My, 5.0)

st.markdown("### 🧮 Parámetros de diseño y costos")
d1, d2, d3 = st.columns(3)
with d1:
    st.session_state.D = st.number_input("Profundidad D (m)", 0.5, 6.0, st.session_state.D, 0.1)
    st.session_state.FS = st.number_input("FS capacidad", 1.5, 4.0, st.session_state.FS, 0.1)
with d2:
    st.session_state.concreto_Sm3 = st.number_input("Concreto (S/ m³)", 100.0, 2000.0, st.session_state.concreto_Sm3, 10.0)
    st.session_state.acero_Skg = st.number_input("Acero (S/ kg)", 1.0, 30.0, st.session_state.acero_Skg, 0.1)
with d3:
    st.session_state.excav_Sm3 = st.number_input("Excavación (S/ m³)", 10.0, 500.0, st.session_state.excav_Sm3, 5.0)
    st.session_state.modelo = st.selectbox(
        "Modelo de capacidad", ["Meyerhof", "Terzaghi", "Hansen"],
        index=["Meyerhof", "Terzaghi", "Hansen"].index(st.session_state.modelo)
    )

st.markdown("### 🔎 Rangos de búsqueda")
r1, r2, r3 = st.columns(3)
with r1:
    st.session_state.B_min, st.session_state.B_max = st.slider(
        "Base B (m)", 0.5, 8.0, (float(st.session_state.B_min), float(st.session_state.B_max))
    )
with r2:
    st.session_state.L_min, st.session_state.L_max = st.slider(
        "Largo L (m)", 0.5, 8.0, (float(st.session_state.L_min), float(st.session_state.L_max))
    )
with r3:
    st.session_state.h_min, st.session_state.h_max = st.slider(
        "Espesor h (m)", 0.3, 2.5, (float(st.session_state.h_min), float(st.session_state.h_max))
    )

g1, g2, g3 = st.columns(3)
with g1:
    st.session_state.nB = st.number_input("N° puntos en B", 5, 60, int(st.session_state.nB), 1)
with g2:
    st.session_state.nL = st.number_input("N° puntos en L", 5, 60, int(st.session_state.nL), 1)
with g3:
    st.session_state.nh = st.number_input("N° puntos en h", 3, 40, int(st.session_state.nh), 1)

# ===================== Cálculo / Búsqueda ===================
if st.button("🚀 Analizar soluciones", use_container_width=True):
    Bs = np.linspace(st.session_state.B_min, st.session_state.B_max, int(st.session_state.nB))
    Ls = np.linspace(st.session_state.L_min, st.session_state.L_max, int(st.session_state.nL))
    hs = np.linspace(st.session_state.h_min, st.session_state.h_max, int(st.session_state.nh))

    rows = []
    for B in Bs:
        for L in Ls:
            for h in hs:
                rows.append(evaluar(B, L, h, st.session_state))
    df = pd.DataFrame(rows)
    df_ok = df[df["cumple"]].copy()

    if df_ok.empty:
        st.error("⚠️ No se encontraron soluciones que cumplan capacidad portante. Ajusta FS, N, φ, c o los rangos B–L–h.")
        st.stop()

    # ---------- Nombres legibles ----------
    nice = {
        "B": "Base B (m)",
        "L": "Largo L (m)",
        "h": "Espesor h (m)",
        "B_eff": "Base efectiva B′ (m)",
        "L_eff": "Largo efectivo L′ (m)",
        "q_req": "Presión de contacto requerida (kPa)",
        "q_adm": "Capacidad admisible del suelo (kPa)",
        "q_ult": "Capacidad última del suelo (kPa)",
        "margen": "Margen de seguridad (kPa)",
        "ex": "Excentricidad eₓ (m)",
        "ey": "Excentricidad eᵧ (m)",
        "cumple": "Cumple capacidad portante",
        "costo": "Costo estimado (S/)"
    }
    df_view = df_ok.rename(columns=nice)

    # ===================== KPIs ============================
    mejor_idx = df_view["Costo estimado (S/)"].idxmin()
    mejor = df_view.loc[mejor_idx]
    p25 = df_view["Costo estimado (S/)"].quantile(0.25)
    df_rob = df_view[df_view["Costo estimado (S/)"] <= p25]
    robusta = df_rob.sort_values(["Margen de seguridad (kPa)", "Costo estimado (S/)"],
                                 ascending=[False, True]).iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Soluciones viables", f"{len(df_view):,}")
    k2.metric("Costo mínimo (S/)", f"{mejor['Costo estimado (S/)']:.0f}")
    k3.metric("Margen (mín. costo)", f"{mejor['Margen de seguridad (kPa)']:.1f} kPa")
    k4.metric("Margen (opción robusta)", f"{robusta['Margen de seguridad (kPa)']:.1f} kPa")

    # ===================== Top 10 ==========================
    st.subheader("Top 10 soluciones por menor costo")
    st.dataframe(df_view.sort_values("Costo estimado (S/)").head(10), use_container_width=True)

    # ===================== Gráficos ========================
    st.subheader("Visualizaciones")
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.scatter(
            df_view,
            x="Base B (m)", y="Largo L (m)",
            color="Costo estimado (S/)", size="Espesor h (m)",
            hover_data=[
                "Presión de contacto requerida (kPa)",
                "Capacidad admisible del suelo (kPa)",
                "Margen de seguridad (kPa)",
                "Base efectiva B′ (m)", "Largo efectivo L′ (m)"
            ],
            title="Soluciones viables (color = costo, tamaño = h)"
        )
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.density_heatmap(
            df_view,
            x="Base B (m)", y="Largo L (m)", z="Margen de seguridad (kPa)",
            nbinsx=30, nbinsy=30, histfunc="avg",
            title="Mapa de calor del margen de seguridad (kPa)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.histogram(df_view, x="Costo estimado (S/)", nbins=30,
                            title="Distribución de costos de soluciones viables")
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        fig4 = px.box(df_view, y="Costo estimado (S/)", title="Resumen estadístico de costos")
        st.plotly_chart(fig4, use_container_width=True)

    # ===================== Estadística =====================
    st.subheader("📈 Estadística descriptiva (soluciones viables)")
    stats_cols = [
        "Costo estimado (S/)", "Presión de contacto requerida (kPa)",
        "Capacidad admisible del suelo (kPa)", "Margen de seguridad (kPa)",
        "Base B (m)", "Largo L (m)", "Espesor h (m)"
    ]
    st.dataframe(df_view[stats_cols].describe().T, use_container_width=True)

    # ===================== Recomendación ===================
    st.markdown("## ✅ Recomendación automática")
    texto = (
        "**Opción de mínimo costo**  \n"
        f"- B = **{mejor['Base B (m)']:.2f} m**, L = **{mejor['Largo L (m)']:.2f} m**, "
        f"h = **{mejor['Espesor h (m)']:.2f} m**  \n"
        f"- Presión requerida = **{mejor['Presión de contacto requerida (kPa)']:.1f} kPa**  \n"
        f"- Capacidad admisible del suelo = **{mejor['Capacidad admisible del suelo (kPa)']:.1f} kPa**  \n"
        f"- Margen de seguridad = **{mejor['Margen de seguridad (kPa)']:.1f} kPa**  \n"
        f"- Costo estimado = **S/ {mejor['Costo estimado (S/)']:.2f}**  \n\n"
        "**Opción robusta (≤ P25 de costo y mayor margen)**  \n"
        f"- B = **{robusta['Base B (m)']:.2f} m**, L = **{robusta['Largo L (m)']:.2f} m**, "
        f"h = **{robusta['Espesor h (m)']:.2f} m**  \n"
        f"- Presión requerida = **{robusta['Presión de contacto requerida (kPa)']:.1f} kPa**  \n"
        f"- Capacidad admisible del suelo = **{robusta['Capacidad admisible del suelo (kPa)']:.1f} kPa**  \n"
        f"- Margen de seguridad = **{robusta['Margen de seguridad (kPa)']:.1f} kPa**  \n"
        f"- Costo estimado = **S/ {robusta['Costo estimado (S/)']:.2f}**  \n\n"
        f"**Criterio:** verificación por **área efectiva** (B′ = B − 2|eₓ|, L′ = L − 2|eᵧ|), "
        f"modelo **{st.session_state.modelo}**, FS = **{st.session_state.FS:.2f}**, "
        f"D = **{st.session_state.D:.2f} m**."
    )
    st.markdown(texto)

    # ===== Definiciones breves =====
    with st.expander("ℹ️ Definiciones rápidas"):
        st.markdown(
            "- **Presión de contacto requerida**: presión media necesaria en la base, "
            "considerando excentricidades (área efectiva B′·L′).  \n"
            "- **Capacidad admisible del suelo**: capacidad última reducida por el FS.  \n"
            "- **Margen de seguridad**: *Capacidad admisible − Presión requerida* (kPa)."
        )

    # ===================== Exportación =====================
    st.subheader("📥 Exportar soluciones")
    csv_sol = df_view.sort_values("Costo estimado (S/)").to_csv(index=False)
    st.download_button(
        "Descargar CSV de soluciones viables",
        data=csv_sol,
        file_name="soluciones_cimentacion.csv",
        mime="text/csv",
        use_container_width=True,
    )


