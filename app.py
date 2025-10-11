# app.py
# ------------------------------------------------------------
# Optimización de Cimentaciones Superficiales
# Suelo estratificado + Cargas por NORMA
# Costeo: Unitario S/, Ajuste por suelo (Leyton 2025), Benchmark m² (Calderón 2015)
# Clasificación geométrica (D/min(B,L))
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

# ===================== CARGAS VIVAS POR NORMA (kPa) =========
NORM_LL = {
    "Viviendas – áreas comunes/corredores": 2.0,
    "Viviendas – cuartos": 2.0,
    "Tiendas/Comercios": 5.0,
    "Oficinas (excepto archivo/computación)": 2.5,
    "Oficinas – salas de archivo": 5.0,
    "Oficinas – corredores y escaleras": 4.0,
    "Bibliotecas – salas de lectura": 3.0,
    "Almacenaje con estantes fijos (no apilables)": 7.5,
    "Centros de educación – aulas": 2.5,
    "Centros de educación – laboratorios": 3.0,
    "Centros de educación – corredores/escaleras": 4.0,
    "Talleres": 3.5,
    "Garajes (vehículos < 2.40 m altura)": 2.5,
    "Hospitales – salas de operación/laboratorios": 3.0,
    "Hospitales – cuartos": 2.0,
    "Hospitales – corredores/escaleras": 4.0,
    "Hoteles – cuartos": 2.0,
    "Lugares de asamblea – asientos fijos": 3.0,
    "Lugares de asamblea – asientos movibles": 4.0,
    "Lugares de asamblea – graderías/tribunas": 5.0,
    "Teatros – vestidores": 2.0,
    "Teatros – cuarto de proyección": 3.0,
}

# ===================== Benchmarks de costo ==================
# Calderón (2015): USD/m²
CALDERON_USD_M2 = {
    "Zapata aislada": 141.77,
    "Zapata corrida": 166.85,
    "Losa de cimentación": 268.96,
}

# Leyton et al. (2025) – multiplicadores aproximados vs. friccional
# (a partir de la tabla de volúmenes y costos totales)
SOIL_COST_MULT = {
    "Friccional": 1.00,
    "Cohesivo": 1.20,       # ~10200/8500
    "Mixto (c-φ)": 1.094,   # ~9300/8500
}

# ===================== Utilidades suelos ====================
GAMMA_W = 9.81  # kN/m3

def gamma_efectivo(gamma, sumergido: bool) -> float:
    return gamma - GAMMA_W if sumergido else gamma

def estrato_en(z):
    for layer in SOIL_PROFILE:
        if layer["z_from"] <= z <= layer["z_to"]:
            return layer
    return SOIL_PROFILE[-1]

def sobrecarga_efectiva(D, nivel_freatico):
    sigma = 0.0
    for L in SOIL_PROFILE:
        top = L["z_from"]; bot = L["z_to"]
        if top >= D:
            break
        tramo = min(D, bot) - top
        if tramo > 1e-9:
            z_mid = top + tramo/2
            sumerg = (z_mid >= nivel_freatico)
            gamma_eff = gamma_efectivo(L["gamma"], sumerg)
            sigma += gamma_eff * tramo
    return sigma  # kPa

def params_en_base(D, nivel_freatico):
    L = estrato_en(D)
    sumerg = (D >= nivel_freatico)
    return L["c"], L["phi"], gamma_efectivo(L["gamma"], sumerg), L["tipo"]

def soil_category(c, phi):
    # regla simple para clasificar tipo de suelo para costos
    if c >= 25 and phi <= 30:
        return "Cohesivo"
    elif c <= 10 and phi >= 34:
        return "Friccional"
    else:
        return "Mixto (c-φ)"

# ===================== Defaults UI ==========================
DEFAULTS = dict(
    # Cargas base
    N=1000.0, Mx=10.0, My=10.0,
    # Diseño / costos
    D=1.50, FS=2.5,
    concreto_Sm3=650.0, acero_Skg=5.50, excav_Sm3=80.0,
    acero_kg_por_m3=60.0,
    tcambio=3.80,   # S/ por USD (puedes cambiar)
    # Búsqueda
    B_min=1.0, B_max=4.0, L_min=1.0, L_max=4.0, h_min=0.5, h_max=1.5,
    nB=30, nL=30, nh=12,
    modelo="Meyerhof",
    nivel_freatico=100.0,  # sin agua por defecto
    # Costeo
    modo_costo="Unitario detallado (S/)",
    tipo_benchmark="Zapata aislada",
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===================== Capacidad portante ====================
def bearing_capacity_factors(phi_deg: float):
    phi = math.radians(phi_deg)
    if phi < 1e-6:
        Nq = 1.0; Nc = 5.14; Ng = 0.0
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
    c, phi, gamma_base_eff, _ = params_en_base(D, nivel_freatico)
    q_eff = sobrecarga_efectiva(D, nivel_freatico)
    Nc, Nq, Ng = bearing_capacity_factors(phi)
    sc, sq, sg = shape_factors(modelo)
    qult = c*Nc*sc + q_eff*Nq*sq + 0.5*gamma_base_eff*B*Ng*sg
    return qult, q_eff, c, phi, gamma_base_eff

def q_required(N: float, Mx: float, My: float, B: float, L: float):
    if N <= 0:
        return float("inf"), 0.0, 0.0, 0.0, 0.0
    ex, ey = Mx/N, My/N
    B_eff = B - 2.0*abs(ex)
    L_eff = L - 2.0*abs(ey)
    if (B_eff <= 0.0) or (L_eff <= 0.0):
        return float("inf"), ex, ey, B_eff, L_eff
    return N/(B_eff*L_eff), ex, ey, B_eff, L_eff

# ===================== Costos ================================
def costo_modelo(B, L, h, D, ss, mult_suelo=1.0):
    """Costo del modelo unitario en S/ (con posible multiplicador por tipo de suelo)."""
    vol = B*L*h
    acero_kg = ss.acero_kg_por_m3 * vol
    excav = B*L*D
    costo = (vol*ss.concreto_Sm3 + acero_kg*ss.acero_Skg + excav*ss.excav_Sm3) * mult_suelo
    return costo, vol, acero_kg, excav

def costo_benchmark_usd(B, L, tcambio, tipo_benchmark):
    """Costo benchmark (Calderón 2015) convertido a S/; usa área en planta del elemento."""
    area_m2 = B*L
    usd_m2 = CALDERON_USD_M2[tipo_benchmark]
    return area_m2 * usd_m2 * tcambio  # S/

# ===================== Evaluación de una tripleta ===========
def evaluar(B, L, h, ss):
    # esfuerzos
    q_req, ex, ey, B_eff, L_eff = q_required(ss.N, ss.Mx, ss.My, B, L)
    qult, q_overburden, c_base, phi_base, gamma_base_eff = qult_estratificado(
        ss.modelo, ss.D, B, ss.nivel_freatico
    )
    q_adm = qult / ss.FS
    margen = (q_adm - q_req) if np.isfinite(q_req) else -1e9
    cumple = (q_adm >= q_req) and np.isfinite(q_req)

    # tipo de suelo (Leyton)
    tipo_suelo_cost = soil_category(c_base, phi_base)
    mult = SOIL_COST_MULT[tipo_suelo_cost] if ss.modo_costo == "Unitario + ajuste por suelo (Leyton 2025)" else 1.0

    # costos
    costo_S, vol, acero_kg, excav = costo_modelo(B, L, h, ss.D, ss, mult_suelo=mult)
    costo_bench_S = costo_benchmark_usd(B, L, ss.tcambio, ss.tipo_benchmark)

    tipo_base = estrato_en(ss.D)["tipo"]
    return {
        "B": B, "L": L, "h": h, "B_eff": B_eff, "L_eff": L_eff,
        "q_req": q_req, "q_adm": q_adm, "q_ult": qult, "margen": margen,
        "ex": ex, "ey": ey, "cumple": cumple,
        "costo": costo_S, "costo_benchmark": costo_bench_S,
        "vol_concreto": vol, "acero_kg": acero_kg, "excav_m3": excav,
        "q_overburden": q_overburden, "c_base": c_base, "phi_base": phi_base,
        "gamma_base_eff": gamma_base_eff, "estrato_base": tipo_base,
        "tipo_suelo_cost": tipo_suelo_cost
    }

# ===================== UI – Suelo ===========================
st.title("Optimización de Cimentaciones Superficiales")

with st.expander("📋 Perfil de suelo (Proyecto 01) – usado automáticamente"):
    df_perfil = pd.DataFrame(SOIL_PROFILE)[["z_from","z_to","tipo","c","phi","gamma"]]
    df_perfil.columns = ["Desde z (m)","Hasta z (m)","Tipo SUCS","c (kPa)","φ (°)","γ (kN/m³)"]
    st.dataframe(df_perfil, use_container_width=True, hide_index=True)

st.markdown("### 🌊 Condiciones hidráulicas y profundidad de fundación")
c0, c1, c2 = st.columns(3)
with c0:
    st.session_state.D = st.number_input("Profundidad de base D (m)", 0.5, 10.0, st.session_state.D, 0.1)
with c1:
    st.session_state.nivel_freatico = st.number_input("Nivel freático (m)", 0.0, 100.0, float(st.session_state.nivel_freatico), 0.1)
with c2:
    st.session_state.FS = st.number_input("FS capacidad", 1.5, 4.0, st.session_state.FS, 0.1)

cD, phiD, gD, _tipoD = params_en_base(st.session_state.D, st.session_state.nivel_freatico)
q_over = sobrecarga_efectiva(st.session_state.D, st.session_state.nivel_freatico)
st.info(
    f"**En D = {st.session_state.D:.2f} m (estrato {_tipoD})** → "
    f"c = **{cD:.1f} kPa**, φ = **{phiD:.1f}°**, γₑₑf = **{gD:.2f} kN/m³**, "
    f"σ′v(D) = **{q_over:.1f} kPa**"
)

# ===================== UI – Cargas por NORMA ================
st.markdown("### 📦 Cargas por **Norma** (opcional; rellena N, Mx, My automáticamente)")
coln1, coln2, coln3 = st.columns(3)
with coln1:
    uso = st.selectbox("Ocupación/uso", list(NORM_LL.keys()),
                       index=list(NORM_LL.keys()).index("Viviendas – áreas comunes/corredores"))
    q_LL = NORM_LL[uso]
with coln2:
    q_DL = st.number_input("Carga muerta DL (kPa)", 0.5, 15.0, 5.0, 0.1)
    niveles = st.number_input("N° de niveles tributarios", 1, 50, 1, 1)
with coln3:
    area = st.number_input("Área tributaria (m²)", 0.5, 1000.0, 20.0, 0.5)
    carga_extra = st.number_input("Cargas adicionales (kN)", 0.0, 1e6, 0.0, 1.0)

colm1, colm2, colm3 = st.columns(3)
with colm1:
    usar_norma = st.button("⬅️ Usar estas cargas")
with colm2:
    ex_in = st.number_input("Excentricidad eₓ (m) (opcional)", 0.0, 2.0, 0.0, 0.01)
with colm3:
    ey_in = st.number_input("Excentricidad eᵧ (m) (opcional)", 0.0, 2.0, 0.0, 0.01)

if usar_norma:
    N_calc = (q_DL + q_LL) * area * niveles + carga_extra
    st.session_state.N = float(N_calc)
    st.session_state.Mx = float(N_calc * ex_in)
    st.session_state.My = float(N_calc * ey_in)
    st.success(
        f"Se asignó **N = {st.session_state.N:.1f} kN**, "
        f"**Mx = {st.session_state.Mx:.1f} kN·m**, **My = {st.session_state.My:.1f} kN·m** "
        f"(uso: {uso}, LL = {q_LL:.2f} kPa, DL = {q_DL:.2f} kPa, área = {area:.2f} m², niveles = {niveles})."
    )

# ===================== UI – Cargas manuales ==================
st.markdown("### 🏗️ Cargas manuales (si prefieres)")
cc1, cc2, cc3 = st.columns(3)
with cc1:
    st.session_state.N = st.number_input("Carga axial N (kN)", 1.0, 100000.0, float(st.session_state.N), 10.0)
with cc2:
    st.session_state.Mx = st.number_input("Momento Mx (kN·m)", 0.0, 1e6, float(st.session_state.Mx), 5.0)
with cc3:
    st.session_state.My = st.number_input("Momento My (kN·m)", 0.0, 1e6, float(st.session_state.My), 5.0)

# ===================== UI – Costos y búsqueda ===============
st.markdown("### 💰 Costos / Benchmarks")
d1, d2, d3 = st.columns(3)
with d1:
    st.session_state.concreto_Sm3 = st.number_input("Concreto (S/ m³)", 100.0, 2000.0, st.session_state.concreto_Sm3, 10.0)
with d2:
    st.session_state.acero_Skg = st.number_input("Acero (S/ kg)", 1.0, 30.0, st.session_state.acero_Skg, 0.1)
with d3:
    st.session_state.excav_Sm3 = st.number_input("Excavación (S/ m³)", 10.0, 500.0, st.session_state.excav_Sm3, 5.0)

e1, e2, e3 = st.columns(3)
with e1:
    st.session_state.modo_costo = st.selectbox(
        "Modo de costo que gobierna la optimización",
        ["Unitario detallado (S/)", "Unitario + ajuste por suelo (Leyton 2025)"],
        index=["Unitario detallado (S/)", "Unitario + ajuste por suelo (Leyton 2025)"].index(st.session_state.modo_costo)
    )
with e2:
    st.session_state.tcambio = st.number_input("Tipo de cambio (S/ por USD)", 2.5, 6.0, float(st.session_state.tcambio), 0.01)
with e3:
    st.session_state.tipo_benchmark = st.selectbox(
        "Benchmark (Calderón 2015) para comparar",
        list(CALDERON_USD_M2.keys()),
        index=list(CALDERON_USD_M2.keys()).index(st.session_state.tipo_benchmark)
    )
st.caption("La optimización usa el **modo de costo** seleccionado; el **benchmark Calderón 2015** se muestra a modo comparativo.")

st.markdown("### 🔎 Rangos de B, L, h")
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
        "margen":"Margen de seguridad (kPa)",
        "costo":"Costo estimado (S/)",
        "costo_benchmark":"Costo benchmark Calderón (S/)",
        "ex":"Excentricidad eₓ (m)","ey":"Excentricidad eᵧ (m)",
        "q_overburden":"Sobrecarga efectiva σ′v(D) (kPa)",
        "c_base":"c del estrato de base (kPa)","phi_base":"φ del estrato de base (°)",
        "gamma_base_eff":"γ efectivo en base (kN/m³)",
        "estrato_base":"Estrato en base","tipo_suelo_cost":"Tipo de suelo (costos)",
        "vol_concreto":"Volumen de concreto (m³)","acero_kg":"Acero (kg)","excav_m3":"Excavación (m³)"
    }
    df_view = df_ok.rename(columns=nice)

    # ===================== KPIs ============================
    # Gobernará el costo del modo seleccionado (columna "Costo estimado (S/)")
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
    k4.metric("Benchmark m² (misma B×L)", f"{mejor['Costo benchmark Calderón (S/)']:.0f} S/")

    # ===================== Clasificación superficial/profunda ===
    B_char = min(float(mejor['Base B (m)']), float(mejor['Largo L (m)']))
    eta = float(st.session_state.D) / B_char
    if eta <= 1.0:
        clase_cimentacion = "CIMENTACIÓN SUPERFICIAL (zapata)"
    elif eta <= 3.0:
        clase_cimentacion = "CIMENTACIÓN SUPERFICIAL (semiprofunda)"
    elif eta <= 4.0:
        clase_cimentacion = "ZONA DE TRANSICIÓN (revisar opción profunda)"
    else:
        clase_cimentacion = "CIMENTACIÓN PROFUNDA (pilotes/pozos)"
    st.metric("Clasificación (D / min(B,L))", f"{clase_cimentacion}", f"{eta:.2f}")

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
                "Tipo de suelo (costos)",
                "Volumen de concreto (m³)","Acero (kg)","Excavación (m³)",
                "Costo benchmark Calderón (S/)"
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

    # ===================== Estadística (en español) =========
    st.subheader("📈 Estadística descriptiva (soluciones viables)")
    columnas_resumen = {
        "Costo estimado (S/)":"Costo modelo (S/)",
        "Costo benchmark Calderón (S/)":"Costo benchmark Calderón (S/)",
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
        vals = resumen(df_view[col]); filas.append([nom]+vals)
    df_stats = pd.DataFrame(
        filas,
        columns=["Variable","Cantidad de soluciones","Promedio","Desviación estándar",
                 "Mínimo","Percentil 25 (Q1)","Mediana (Q2)","Percentil 75 (Q3)","Máximo"]
    )
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ Definiciones rápidas"):
        st.markdown(
            "- **Presión requerida**: N /(B′·L′), usando excentricidades (área efectiva).\n"
            "- **Capacidad admisible**: qult/FS con c, φ y γ del estrato de apoyo y σ′v(D) acumulada.\n"
            "- **Margen de seguridad**: Capacidad admisible − Presión requerida.\n"
            "- **Costos**:\n"
            "  - *Modelo unitario*: concreto + acero + excavación (S/).\n"
            "  - *Leyton 2025*: aplica multiplicador por tipo de suelo (Friccional=1.00, Cohesivo≈1.20, Mixto≈1.094).\n"
            "  - *Benchmark Calderón 2015*: USD/m² → S/ con tu tipo de cambio, para comparar.\n"
            "- **Clasificación**: D/min(B,L) ≤ 1 superficial; 1–3 semiprofunda; 3–4 transición; >4 profunda."
        )

    # ===================== Recomendación ===================
    st.markdown("## ✅ Recomendación automática")
    texto = (
        f"**Modo de costo usado en la optimización:** **{st.session_state.modo_costo}**  \n"
        f"**Benchmark de comparación:** **{st.session_state.tipo_benchmark}** (Calderón 2015), "
        f"**TC = {st.session_state.tcambio:.2f} S//USD**.  \n\n"
        "**Opción de mínimo costo (modelo seleccionado)**  \n"
        f"- B = **{mejor['Base B (m)']:.2f} m**, L = **{mejor['Largo L (m)']:.2f} m**, "
        f"h = **{mejor['Espesor h (m)']:.2f} m**  \n"
        f"- Tipo de suelo para costos: **{mejor['Tipo de suelo (costos)']}**  \n"
        f"- Presión requerida = **{mejor['Presión de contacto requerida (kPa)']:.1f} kPa**  \n"
        f"- Capacidad admisible = **{mejor['Capacidad admisible del suelo (kPa)']:.1f} kPa**  \n"
        f"- Margen de seguridad = **{mejor['Margen de seguridad (kPa)']:.1f} kPa**  \n"
        f"- **Costo estimado (modelo)** = **S/ {mejor['Costo estimado (S/)']:.2f}**  \n"
        f"- **Costo benchmark (Calderón)** = **S/ {mejor['Costo benchmark Calderón (S/)']:.2f}**  \n\n"
        "**Opción robusta (≤ P25 de costo y mayor margen)**  \n"
        f"- B = **{robusta['Base B (m)']:.2f} m**, L = **{robusta['Largo L (m)']:.2f} m**, "
        f"h = **{robusta['Espesor h (m)']:.2f} m**  \n"
        f"- Presión requerida = **{robusta['Presión de contacto requerida (kPa)']:.1f} kPa**, "
        f"Capacidad admisible = **{robusta['Capacidad admisible del suelo (kPa)']:.1f} kPa**  \n"
        f"- Margen de seguridad = **{robusta['Margen de seguridad (kPa)']:.1f} kPa**  \n"
        f"- **Costo estimado (modelo)** = **S/ {robusta['Costo estimado (S/)']:.2f}**  \n\n"
        f"**Clasificación geométrica** (con la solución de mínimo costo): "
        f"D/min(B,L) = **{eta:.2f}** → **{clase_cimentacion}**.  \n"
        f"*(Confirma con asentamientos y punzonamiento según normas.)*"
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

