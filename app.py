# app.py
# ------------------------------------------------------------
# Optimización de Cimentaciones Superficiales – perfil estratificado
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

# ===================== Perfil de Suelo (Proyecto 01) ========
# H [m], tipo, c[kPa], φ[°], γ[kN/m3]
SOIL_PROFILE = [
    {"z_from": 0.0,  "z_to": 1.1,  "tipo": "ML",  "c": 5.0,  "phi": 28.0, "gamma": 17.0},
    {"z_from": 1.1,  "z_to": 4.0,  "tipo": "GP1", "c": 32.0, "phi": 40.0, "gamma": 21.0},
    {"z_from": 4.0,  "z_to": 7.0,  "tipo": "GP2", "c": 37.0, "phi": 39.0, "gamma": 22.0},
    {"z_from": 7.0,  "z_to": 11.15,"tipo": "GP3", "c": 42.0, "phi": 39.0, "gamma": 22.0},
]

GAMMA_W = 9.81  # kN/m3
def gamma_efectivo(gamma, sumergido: bool) -> float:
    return gamma - GAMMA_W if sumergido else gamma

def estrato_en(z):
    for layer in SOIL_PROFILE:
        if layer["z_from"] <= z <= layer["z_to"]:
            return layer
    return SOIL_PROFILE[-1]  # si D > perfil, usar último

def sobrecarga_efectiva(D, nivel_freatico):
    """
    σ'v = ∑ gamma(efectivo) * espesor_por_encima_de_D
    Considera cambio a γ' bajo el nivel freático.
    """
    sigma = 0.0
    for L in SOIL_PROFILE:
        top = L["z_from"]; bot = L["z_to"]
        if top >= D:   # estratos por debajo de la base no aportan sobrecarga
            break
        tramo = min(D, bot) - top
        if tramo > 1e-9:
            # el tramo puede estar parcialmente sumergido
            z_mid = top + tramo/2
            sumerg = (z_mid >= nivel_freatico)
            gamma_eff = gamma_efectivo(L["gamma"], sumerg)
            sigma += gamma_eff * tramo
    return sigma  # kPa (kN/m2)

def params_en_base(D, nivel_freatico):
    """c, φ y γ (efectivo para término γ·B) del estrato que recibe la zapata."""
    L = estrato_en(D)
    sumerg = (D >= nivel_freatico)
    return L["c"], L["phi"], gamma_efectivo(L["gamma"], sumerg), L["tipo"]

# ===================== Defaults =============================
DEFAULTS = dict(
    # --- Cargas (kN, kN·m) ---
    N=1000.0, Mx=10.0, My=10.0,
    # --- Reglas de diseño / costos ---
    D=1.50, FS=2.5,
    concreto_Sm3=650.0, acero_Skg=5.50, excav_Sm3=80.0,
    acero_kg_por_m3=60.0,
    # --- Rango de búsqueda ---
    B_min=1.0, B_max=4.0, L_min=1.0, L_max=4.0, h_min=0.5, h_max=1.5,
    nB=30, nL=30, nh=12,
    modelo="Meyerhof",
    nivel_freatico=100.0,  # por defecto "muy abajo" (sin efecto); cambia esto si hay agua
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===================== Capacidad portante ====================
def bearing_capacity_factors(phi_deg: float):
    phi = math.radians(phi_deg)
    if phi < 1e-6:
        Nq = 1.0
        Nc = 5.14
        Ng = 0.0
    else:
        Nq = math.e ** (math.pi * math.tan(phi)) * (math.tan(math.pi/4 + phi/2)) ** 2
        Nc = (Nq - 1.0) / math.tan(phi)
        Ng = 2.0 * (Nq + 1.0) * math.tan(phi)
    return Nc, Nq, Ng

def shape_factors(modelo: str):
    if "Terzaghi" in modelo:  return 1.3, 1.2, 0.8
    if "Meyerhof" in modelo:  return 1.3, 1.2, 1.0
    return 1.0, 1.0, 1.0

def qult_estratificado(modelo, D, B, nivel_freatico):
    """
    c y φ del estrato de fundación.
    q = σ'v(D) (sobrecarga efectiva acumulada).
    γ para término 0.5*γ*B*Nγ es el γ(efectivo) del estrato de base.
    """
    c, phi, gamma_base_eff, _ = params_en_base(D, nivel_freatico)
    q_eff = sobrecarga_efectiva(D, nivel_freatico)
    Nc, Nq, Ng = bearing_capacity_factors(phi)
    sc, sq, sg = shape_factors(modelo)
    qult = c*Nc*sc + q_eff*Nq*sq + 0.5*gamma_base_eff*B*Ng*sg
    return qult, q_eff, c, phi, gamma_base_eff

def q_required(N: float, Mx: float, My: float, B: float, L: float):
    if N <= 0:  # inválido
        return float("inf"), 0.0, 0.0, 0.0, 0.0
    ex, ey = Mx/N, My/N
    B_eff = B - 2.0*abs(ex)
    L_eff = L - 2.0*abs(ey)
    if (B_eff <= 0.0) or (L_eff <= 0.0):
        return float("inf"), ex, ey, B_eff, L_eff
    return N/(B_eff*L_eff), ex, ey, B_eff, L_eff

def costo_total(B, L, h, concreto_Sm3, acero_Skg, excav_Sm3, D, acero_kg_por_m3):
    vol = B*L*h
    acero_kg = acero_kg_por_m3 * vol
    excav = B*L*D
    return vol*concreto_Sm3 + acero_kg*acero_Skg + excav*excav_Sm3

def evaluar(B, L, h, ss):
    q_req, ex, ey, B_eff, L_eff = q_required(ss.N, ss.Mx, ss.My, B, L)
    qult, q_overburden, c_base, phi_base, gamma_base_eff = qult_estratificado(
        ss.modelo, ss.D, B, ss.nivel_freatico
    )
    q_adm = qult / ss.FS
    margen = (q_adm - q_req) if np.isfinite(q_req) else -1e9
    cumple = (q_adm >= q_req) and np.isfinite(q_req)
    costo = costo_total(B, L, h, ss.concreto_Sm3, ss.acero_Skg, ss.excav_Sm3, ss.D, ss.acero_kg_por_m3)
    tipo_base = estrato_en(ss.D)["tipo"]
    return {
        "B": B, "L": L, "h": h, "B_eff": B_eff, "L_eff": L_eff,
        "q_req": q_req, "q_adm": q_adm, "q_ult": qult, "margen": margen,
        "ex": ex, "ey": ey, "cumple": cumple, "costo": costo,
        "q_overburden": q_overburden, "c_base": c_base, "phi_base": phi_base,
        "gamma_base_eff": gamma_base_eff, "estrato_base": tipo_base
    }

# ===================== UI – Entradas =========================
st.title("Optimización de Cimentaciones Superficiales")

with st.expander("📋 Perfil de suelo (Proyecto 01) – usado automáticamente"):
    df_perfil = pd.DataFrame(SOIL_PROFILE)[["z_from","z_to","tipo","c","phi","gamma"]]
    df_perfil.columns = ["Desde z (m)","Hasta z (m)","Tipo SUCS","c (kPa)","φ (°)","γ (kN/m³)"]
    st.dataframe(df_perfil, use_container_width=True, hide_index=True)

st.markdown("### 🌊 Condiciones hidráulicas y profundidad")
c0, c1, c2 = st.columns(3)
with c0:
    st.session_state.D = st.number_input("Profundidad de base D (m)", 0.5, 10.0, st.session_state.D, 0.1)
with c1:
    st.session_state.nivel_freatico = st.number_input("Nivel freático (m)", 0.0, 100.0, float(st.session_state.nivel_freatico), 0.1)
with c2:
    st.session_state.FS = st.number_input("FS capacidad", 1.5, 4.0, st.session_state.FS, 0.1)

# Mostrar parámetros derivados del perfil a esa D
cD, phiD, gD, qD = params_en_base(st.session_state.D, st.session_state.nivel_freatico)[:4]
tipoD = estrato_en(st.session_state.D)["tipo"]
q_over = sobrecarga_efectiva(st.session_state.D, st.session_state.nivel_freatico)
st.info(
    f"**En D = {st.session_state.D:.2f} m (estrato {tipoD})** → "
    f"c = **{cD:.1f} kPa**, φ = **{phiD:.1f}°**, γₑₑf = **{gD:.2f} kN/m³**, "
    f"σ′v(D) = **{q_over:.1f} kPa**"
)

st.markdown("### 🏗️ Cargas")
cc1, cc2, cc3 = st.columns(3)
with cc1:
    st.session_state.N = st.number_input("Carga axial N (kN)", 1.0, 100000.0, st.session_state.N, 10.0)
with cc2:
    st.session_state.Mx = st.number_input("Momento Mx (kN·m)", 0.0, 1e6, st.session_state.Mx, 5.0)
with cc3:
    st.session_state.My = st.number_input("Momento My (kN·m)", 0.0, 1e6, st.session_state.My, 5.0)

st.markdown("### 💰 Costos")
d1, d2, d3 = st.columns(3)
with d1:
    st.session_state.concreto_Sm3 = st.number_input("Concreto (S/ m³)", 100.0, 2000.0, st.session_state.concreto_Sm3, 10.0)
with d2:
    st.session_state.acero_Skg = st.number_input("Acero (S/ kg)", 1.0, 30.0, st.session_state.acero_Skg, 0.1)
with d3:
    st.session_state.excav_Sm3 = st.number_input("Excavación (S/ m³)", 10.0, 500.0, st.session_state.excav_Sm3, 5.0)

st.markdown("### ⚙️ Modelo y búsqueda")
m1, m2, m3 = st.columns(3)
with m1:
    st.session_state.modelo = st.selectbox("Modelo de capacidad", ["Meyerhof", "Terzaghi", "Hansen"],
                                           index=["Meyerhof", "Terzaghi", "Hansen"].index(st.session_state.modelo))
with m2:
    st.session_state.acero_kg_por_m3 = st.number_input("Acero asumido (kg/m³)", 10.0, 150.0, float(st.session_state.acero_kg_por_m3), 5.0)
with m3:
    st.caption("El perfil estratificado fija c, φ y γ automáticamente en D.")

st.markdown("### 🔎 Rangos de búsqueda de geometría")
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
        st.error("⚠️ No hay soluciones que cumplan. Sube B/L/D o baja Mx/My/FS.")
        st.stop()

    # ----- nombres legibles -----
    nice = {
        "B":"Base B (m)","L":"Largo L (m)","h":"Espesor h (m)",
        "B_eff":"Base efectiva B′ (m)","L_eff":"Largo efectivo L′ (m)",
        "q_req":"Presión de contacto requerida (kPa)",
        "q_adm":"Capacidad admisible del suelo (kPa)",
        "q_ult":"Capacidad última del suelo (kPa)",
        "margen":"Margen de seguridad (kPa)","costo":"Costo estimado (S/)",
        "ex":"Excentricidad eₓ (m)","ey":"Excentricidad eᵧ (m)",
        "q_overburden":"Sobrecarga efectiva σ′v(D) (kPa)",
        "c_base":"c del estrato de base (kPa)","phi_base":"φ del estrato de base (°)",
        "gamma_base_eff":"γ efectivo en base (kN/m³)","estrato_base":"Estrato en base"
    }
    df_view = df_ok.rename(columns=nice)

    # ===================== KPIs ============================
    mejor_idx = df_view["Costo estimado (S/)"].idxmin()
    mejor = df_view.loc[mejor_idx]
    p25 = df_view["Costo estimado (S/)"].quantile(0.25)
    df_rob = df_view[df_view["Costo estimado (S/)"] <= p25]
    robusta = df_rob.sort_values(["Margen de seguridad (kPa)","Costo estimado (S/)"],
                                 ascending=[False,True]).iloc[0]

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Soluciones viables", f"{len(df_view):,}")
    k2.metric("Costo mínimo (S/)", f"{mejor['Costo estimado (S/)']:.0f}")
    k3.metric("Margen (mín. costo)", f"{mejor['Margen de seguridad (kPa)']:.1f} kPa")
    k4.metric("Margen (opción robusta)", f"{robusta['Margen de seguridad (kPa)']:.1f} kPa")

    # ===================== Top 10 ==========================
    st.subheader("Top 10 soluciones por menor costo")
    st.dataframe(
        df_view.sort_values("Costo estimado (S/)").head(10),
        use_container_width=True
    )

    # ===================== Gráficos ========================
    st.subheader("Visualizaciones")
    c1,c2 = st.columns(2)
    with c1:
        fig1 = px.scatter(
            df_view, x="Base B (m)", y="Largo L (m)",
            color="Costo estimado (S/)", size="Espesor h (m)",
            hover_data=[
                "Presión de contacto requerida (kPa)",
                "Capacidad admisible del suelo (kPa)",
                "Margen de seguridad (kPa)",
                "Base efectiva B′ (m)","Largo efectivo L′ (m)",
                "Estrato en base","Sobrecarga efectiva σ′v(D) (kPa)"
            ],
            title="Soluciones viables (color = costo, tamaño = h)"
        )
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.density_heatmap(
            df_view, x="Base B (m)", y="Largo L (m)",
            z="Margen de seguridad (kPa)", nbinsx=30, nbinsy=30, histfunc="avg",
            title="Mapa de calor del margen de seguridad (kPa)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ===================== Estadística en español ==========
    st.subheader("📈 Estadística descriptiva (soluciones viables)")
    columnas_resumen = {
        "Costo estimado (S/)":"Costo (S/)",
        "Presión de contacto requerida (kPa)":"Presión requerida (kPa)",
        "Capacidad admisible del suelo (kPa)":"Capacidad admisible (kPa)",
        "Margen de seguridad (kPa)":"Margen de seguridad (kPa)",
        "Base B (m)":"Base B (m)","Largo L (m)":"Largo L (m)","Espesor h (m)":"Espesor h (m)",
    }
    def resumen(serie):
        s = pd.to_numeric(serie, errors="coerce").dropna()
        if s.empty:
            return [0,None,None,None,None,None,None,None]
        return [
            int(s.count()), float(s.mean()),
            float(s.std(ddof=1)) if s.count()>1 else 0.0,
            float(s.min()), float(np.percentile(s,25)),
            float(np.percentile(s,50)), float(np.percentile(s,75)),
            float(s.max())
        ]
    filas=[]
    for col, nom in columnas_resumen.items():
        vals = resumen(df_view[col])
        filas.append([nom]+vals)
    df_stats = pd.DataFrame(
        filas,
        columns=["Variable","Cantidad de soluciones","Promedio","Desviación estándar",
                 "Mínimo","Percentil 25 (Q1)","Mediana (Q2)","Percentil 75 (Q3)","Máximo"]
    )
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

    # ===================== Recomendación ===================
    st.markdown("## ✅ Recomendación automática (perfil estratificado)")
    texto = (
        "**Opción de mínimo costo**  \n"
        f"- B = **{mejor['Base B (m)']:.2f} m**, L = **{mejor['Largo L (m)']:.2f} m**, "
        f"h = **{mejor['Espesor h (m)']:.2f} m**  \n"
        f"- Estrato de apoyo: **{mejor['Estrato en base']}**, c = **{mejor['c del estrato de base (kPa)']:.1f} kPa**, "
        f"φ = **{mejor['φ del estrato de base (°)']:.1f}°**, γₑₑf = **{mejor['γ efectivo en base (kN/m³)']:.2f}**  \n"
        f"- Sobrecarga efectiva en base σ′v(D) = **{mejor['Sobrecarga efectiva σ′v(D) (kPa)']:.1f} kPa**  \n"
        f"- Presión requerida = **{mejor['Presión de contacto requerida (kPa)']:.1f} kPa**  \n"
        f"- Capacidad admisible = **{mejor['Capacidad admisible del suelo (kPa)']:.1f} kPa**  \n"
        f"- Margen de seguridad = **{mejor['Margen de seguridad (kPa)']:.1f} kPa**  \n"
        f"- Costo estimado = **S/ {mejor['Costo estimado (S/)']:.2f}**  \n\n"
        "**Opción robusta (≤ P25 de costo y mayor margen)**  \n"
        f"- B = **{robusta['Base B (m)']:.2f} m**, L = **{robusta['Largo L (m)']:.2f} m**, "
        f"h = **{robusta['Espesor h (m)']:.2f} m**  \n"
        f"- Estrato de apoyo: **{robusta['Estrato en base']}**, c = **{robusta['c del estrato de base (kPa)']:.1f} kPa**, "
        f"φ = **{robusta['φ del estrato de base (°)']:.1f}°**  \n"
        f"- Presión requerida = **{robusta['Presión de contacto requerida (kPa)']:.1f} kPa**  \n"
        f"- Capacidad admisible = **{robusta['Capacidad admisible del suelo (kPa)']:.1f} kPa**  \n"
        f"- Margen de seguridad = **{robusta['Margen de seguridad (kPa)']:.1f} kPa**  \n"
        f"- Costo estimado = **S/ {robusta['Costo estimado (S/)']:.2f}**  \n\n"
        f"**Criterio:** Para suelos estratificados se adoptó c, φ y γ del **estrato de apoyo**, "
        f"y la **sobrecarga efectiva σ′v(D)** como suma de γ o γ′ por espesor por encima de la base; "
        f"nivel freático a **{st.session_state.nivel_freatico:.2f} m**. Modelo **{st.session_state.modelo}**, "
        f"FS = **{st.session_state.FS:.2f}**."
    )
    st.markdown(texto)

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

