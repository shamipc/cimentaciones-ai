# app.py
# ------------------------------------------------------------
# Optimización de Cimentaciones Superficiales – versión robusta
# Autor: Adrian + ChatGPT
# ------------------------------------------------------------
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ===================== Config & Estilo ======================
st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")
st.markdown("""
<style>
h1 {font-size: 28px !important; margin-bottom: .4rem;}
h2 {font-size: 22px !important; margin-bottom: .4rem;}
h3 {font-size: 18px !important; margin-bottom: .3rem;}
section.main .block-container {padding-top: 1rem; padding-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

# ===================== Defaults =============================
DEFAULTS = dict(
    # --- Suelo (kN, m, kPa) ---
    gamma=18.0, c=25.0, phi=30.0, Es=15000.0, nu=0.30,
    nivel_freatico=2.0, q_adm=150.0, q_ult=375.0,
    # --- Cargas (kN, kN·m) ---
    N=1000.0, Mx=0.0, My=0.0,
    # --- Materiales/costos ---
    fc=21.0, fy=420.0, recubrimiento=0.05,
    concreto_Sm3=650.0, acero_Skg=5.50, excav_Sm3=80.0, relleno_Sm3=50.0,
    # --- Reglas de diseño ---
    D=1.5, FS=2.5, asent_max=0.025,
    # --- Rango de búsqueda ---
    B_min=1.0, B_max=4.0, L_min=1.0, L_max=4.0, h_min=0.5, h_max=1.5,
    nB=30, nL=30, nh=12,
    # --- Modelo capacidad ---
    modelo="Meyerhof"
)

for k,v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===================== Funciones base =======================
def deg2rad(d): 
    return math.radians(d)

def bearing_capacity_factors(phi_deg: float):
    """
    Factores clásicos (phi en grados) en FORMATO CONSISTENTE (radianes internos).
    Nq = exp(pi*tan(phi)) * tan^2(45+phi/2)
    Nc = (Nq-1)/tan(phi)  (si phi>0)  |  Nc=5.14 para phi≈0
    Ngamma adopta forma de Meyerhof: 2*(Nq+1)*tan(phi)
    """
    phi = deg2rad(phi_deg)
    if phi < 1e-6:
        Nq = 1.0
        Nc = 5.14
        Ng = 0.0
    else:
        Nq = math.e**(math.pi*math.tan(phi)) * (math.tan(math.pi/4 + phi/2))**2
        Nc = (Nq - 1.0)/math.tan(phi)
        Ng = 2.0*(Nq + 1.0)*math.tan(phi)
    return Nc, Nq, Ng

def shape_factors(modelo: str, B: float, L: float, phi_deg: float):
    # Ajustes simples por forma (Terzaghi/Meyerhof). Mantener 1.0 si se desconoce.
    if "Terzaghi" in modelo:
        sc, sq, sg = 1.3, 1.2, 0.8
    elif "Meyerhof" in modelo:
        sc, sq, sg = 1.3, 1.2, 1.0
    else:  # Hansen/Genérico
        sc, sq, sg = 1.0, 1.0, 1.0
    return sc, sq, sg

def qult(modelo: str, gamma: float, c: float, phi_deg: float, B: float, D: float):
    """
    Capacidad última a cortante general (superficial) sin inclinación, sin carga inclinada.
    q = gamma*D (sobrecarga)
    qult = c*Nc*sc + q*Nq*sq + 0.5*gamma*B*Ng*sg
    """
    Nc, Nq, Ng = bearing_capacity_factors(phi_deg)
    sc, sq, sg = shape_factors(modelo, B, B, phi_deg)
    q = gamma * D
    return c*Nc*sc + q*Nq*sq + 0.5*gamma*B*Ng*sg

def qult_corr(modelo, gamma, c, phi, B, D, nivel_freatico):
    """
    Corrección hídrica muy simple: si la zapata está por debajo del N.F.,
    reducir gamma efectivo (aprox). Para estudio serio usar correcciones
    por peso sumergido y posición de N.F. respecto a base.
    """
    gamma_eff = gamma if D >= nivel_freatico else 0.5*gamma
    return qult(modelo, gamma_eff, c, phi, B, D)

def q_required(N: float, Mx: float, My: float, B: float, L: float):
    """
    Presión requerida con excentricidades. Se usa el método del área efectiva:
    e_x = Mx/N, e_y = My/N
    B' = B - 2*|e_x| ; L' = L - 2*|e_y|
    q_req = N / (B' * L'), si B',L' > 0; si no, no cumple (retorna inf).
    """
    if N <= 0:
        return float("inf"), 0, 0, 0, 0
    e_x, e_y = (Mx/N, My/N)
    B_eff = B - 2.0*abs(e_x)
    L_eff = L - 2.0*abs(e_y)
    if (B_eff <= 0) or (L_eff <= 0):
        return float("inf"), e_x, e_y, B_eff, L_eff
    qreq = N/(B_eff*L_eff)
    return qreq, e_x, e_y, B_eff, L_eff

def total_cost(B,L,h,concreto_Sm3,acero_Skg,excav_Sm3,D):
    vol = B*L*h
    # acero_kg: supuesto de 60 kg/m3 (ajustable por UI)
    acero_kg = 60.0*vol
    excav = B*L*D
    return (vol*concreto_Sm3) + (acero_kg*acero_Skg) + (excav*excav_Sm3)

def evaluar_candidato(B,L,h,ss):
    q_req, ex, ey, B_eff, L_eff = q_required(ss.N, ss.Mx, ss.My, B, L)
    q_ult = qult_corr(ss.modelo, ss.gamma, ss.c, ss.phi, B, ss.D, ss.nivel_freatico)
    q_adm = q_ult/ss.FS
    margen = (q_adm - q_req) if np.isfinite(q_req) else -1e9
    cumple = (q_adm >= q_req) and np.isfinite(q_req)
    costo = total_cost(B,L,h, ss.concreto_Sm3, ss.acero_Skg, ss.excav_Sm3, ss.D)
    return {
        "B":B, "L":L, "h":h, "B_eff":B_eff, "L_eff":L_eff,
        "q_req":q_req, "q_adm":q_adm, "q_ult":q_ult, "margen":margen,
        "ex":ex, "ey":ey, "cumple":cumple, "costo":costo
    }

# ===================== UI – Entradas =========================
st.title("Optimización de Cimentaciones Superficiales")

st.markdown("### 🧪 Propiedades de suelo")
c1,c2,c3 = st.columns(3)
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
cc1,cc2,cc3 = st.columns(3)
with cc1: st.session_state.N  = st.number_input("N (kN)", 1.0, 100000.0, st.session_state.N, 10.0)
with cc2: st.session_state.Mx = st.number_input("Mx (kN·m)", 0.0, 1e6, st.session_state.Mx, 5.0)
with cc3: st.session_state.My = st.number_input("My (kN·m)", 0.0, 1e6, st.session_state.My, 5.0)

st.markdown("### 📐 Parámetros de diseño y costos")
d1,d2,d3 = st.columns(3)
with d1:
    st.session_state.D  = st.number_input("Profundidad D (m)", 0.5, 6.0, st.session_state.D, 0.1)
    st.session_state.FS = st.number_input("FS capacidad", 1.5, 4.0, st.session_state.FS, 0.1)
with d2:
    st.session_state.concreto_Sm3 = st.number_input("Concreto (S/ m³)", 100.0, 2000.0, st.session_state.concreto_Sm3, 10.0)
    st.session_state.acero_Skg    = st.number_input("Acero (S/ kg)", 1.0, 30.0, st.session_state.acero_Skg, 0.1)
with d3:
    st.session_state.excav_Sm3    = st.number_input("Excavación (S/ m³)", 10.0, 500.0, st.session_state.excav_Sm3, 5.0)
    st.session_state.modelo       = st.selectbox("Modelo de capacidad", ["Meyerhof","Terzaghi","Hansen"], index=["Meyerhof","Terzaghi","Hansen"].index(st.session_state.modelo))

st.markdown("### 🔎 Rangos de búsqueda")
r1,r2,r3 = st.columns(3)
with r1:
    st.session_state.B_min, st.session_state.B_max = st.slider("Base B (m)", 0.5, 8.0, (st.session_state.B_min, st.session_state.B_max))
with r2:
    st.session_state.L_min, st.session_state.L_max = st.slider("Largo L (m)", 0.5, 8.0, (st.session_state.L_min, st.session_state.L_max))
with r3:
    st.session_state.h_min, st.session_state.h_max = st.slider("Altura h (m)", 0.3, 2.5, (st.session_state.h_min, st.session_state.h_max))

g1,g2,g3 = st.columns(3)
with g1: st.session_state.nB = st.number_input("N° puntos en B", 5, 60, st.session_state.nB, 1)
with g2: st.session_state.nL = st.number_input("N° puntos en L", 5, 60, st.session_state.nL, 1)
with g3: st.session_state.nh = st.number_input("N° puntos en h", 3, 40, st.session_state.nh, 1)

# ===================== BÚSQUEDA =============================
if st.button("🚀 Analizar soluciones", use_container_width=True):
    Bs = np.linspace(st.session_state.B_min, st.session_state.B_max, st.session_state.nB)
    Ls = np.linspace(st.session_state.L_min, st.session_state.L_max, st.session_state.nL)
    hs = np.linspace(st.session_state.h_min, st.session_state.h_max, st.session_state.nh)

    rows = []
    for B in Bs:
        for L in Ls:
            for h in hs:
                rows.append(evaluar_candidato(B,L,h, st.session_state))
    df = pd.DataFrame(rows)
    df_ok = df[df["cumple"]].copy()

    if df_ok.empty:
        st.error("⚠️ No se encontraron soluciones que cumplan capacidad portante (revisa FS, N, φ, c o rangos B–L–h).")
        st.stop()

    # ===================== KPIs ============================
    mejor = df_ok.sort_values("costo", ascending=True).iloc[0]
    # Opción robusta: 25º percentil de costo pero con mayor margen
    p25 = df_ok["costo"].quantile(0.25)
    df_rob = df_ok[df_ok["costo"] <= p25]
    robusta = df_rob.sort_values(["margen","costo"], ascending=[False,True]).iloc[0]

    cA,cB,cC,cD = st.columns(4)
    cA.metric("Soluciones viables", f"{len(df_ok):,}")
    cB.metric("Costo mínimo (S/)", f"{mejor['costo']:.0f}")
    cC.metric("Margen (q_adm - q_req) mincost (kPa)", f"{mejor['margen']:.1f}")
    cD.metric("Robusta: margen (kPa)", f"{robusta['margen']:.1f}")

    # ===================== Tablas ==========================
    st.subheader("Top 10 soluciones por menor costo")
    st.dataframe(df_ok.sort_values("costo").head(10), use_container_width=True)

    # ===================== Gráficos ========================
    st.subheader("Visualizaciones")

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.scatter(
            df_ok, x="B", y="L", color="costo", size="h",
            hover_data=["q_req","q_adm","margen","B_eff","L_eff"],
            title="Soluciones viables (color=costo, tamaño=h)"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.density_heatmap(
            df_ok, x="B", y="L", z="margen", nbinsx=30, nbinsy=30, histfunc="avg",
            title="Mapa de calor del margen de seguridad (kPa)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.histogram(df_ok, x="costo", nbins=30, title="Distribución de costos de soluciones viables")
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        fig4 = px.box(df_ok, y="costo", title="Resumen estadístico de costos")
        st.plotly_chart(fig4, use_container_width=True)

    # ===================== Estadística =====================
    st.subheader("📈 Estadística descriptiva (soluciones viables)")
    stats_cols = ["costo","q_req","q_adm","margen","B","L","h"]
    st.dataframe(df_ok[stats_cols].describe().T, use_container_width=True)

    # ===================== Recomendación ===================
    st.markdown("## ✅ Recomendación automática")
    st.write(
        f"""
**Opción de mínimo costo**  
- B = **{mejor['B']:.2f} m**, L = **{mejor['L']:.2f} m**, h = **{mejor['h']:.2f} m**  
- q_req = **{mejor['q_req']:.1f} kPa**, q_adm = **{mejor['q_adm']:.1f} kPa**, margen = **{mejor['margen']:.1f} kPa**  
- Costo estimado = **S/ {mejor['costo']:.2f}**

**Opción robusta (≤ P25 de costo, mayor margen)**  
- B = **{robusta['B']:.2f} m**, L = **{robusta['L']:.2f} m**, h = **{robusta['h']:.2f} m**  
- q_req = **{robusta['q_req']:.1f} kPa**, q_adm = **{robusta['q_adm']:.1f} kPa**, margen = **{robusta['margen']:.1f} kPa**  
- Costo estimado = **S/ {robusta['costo']:.2f}**

**Criterio:** Se verifica capacidad





