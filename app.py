import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")

# ====== Estilos ======
st.markdown("""
<style>
h1 {font-size: 28px !important; margin-bottom: 0.4rem;}
h2 {font-size: 22px !important; margin-bottom: 0.4rem;}
h3 {font-size: 18px !important; margin-bottom: 0.3rem;}
section.main .block-container {padding-top: 1rem; padding-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

# ---------- Parámetros iniciales ----------
DEFAULTS = dict(
    # Suelo
    gamma=18.0, c=25.0, phi=30.0, Es=15000.0, nu=0.30, nivel_freatico=2.0,
    # Cargas
    N=1000.0, Mx=0.0, My=0.0,
    # Materiales cimentación
    fc=21.0, fy=420.0, recubrimiento=0.05,
    # Factores de seguridad y límites
    FS=2.5, asent_max=0.025,
    # Costos
    concreto_Sm3=650.0, acero_Skg=5.50, excav_Sm3=80.0, relleno_Sm3=50.0,
    # Geometría
    D=1.5, B_min=1.0, B_max=4.0, L_min=1.0, L_max=4.0, h_min=0.5, h_max=1.5,
    nB=20, nL=20, nh=10,
    modelo="Terzaghi"
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- Factores capacidad portante ----------
def Nq(phi):  return math.e ** (math.pi * math.tan(phi)) * math.tan(math.radians(45)+phi/2) ** 2
def Nc(phi):  return (Nq(phi) - 1) / math.tan(phi) if phi > 1e-6 else 5.14
def Ng(phi):  return 2 * (Nq(phi) + 1) * math.tan(phi)

def factors(modelo, phi_deg):
    phi = math.radians(phi_deg)
    if "Terzaghi" in modelo:   sc, sq, sγ = 1.3, 1.2, 0.8
    elif "Meyerhof" in modelo: sc, sq, sγ = 1.3, 1.2, 1.0
    elif "Hansen" in modelo:   sc, sq, sγ = 1.0, 1.0, 1.0
    else:                      sc, sq, sγ = 1.0, 1.0, 1.0
    return Nc(phi), Nq(phi), Ng(phi), sc, sq, sγ

def qult(modelo, gamma, c, phi, B, D):
    Nc_, Nq_, Ng_, sc, sq, sγ = factors(modelo, phi)
    q = gamma * D
    return c*Nc_*sc + q*Nq_*sq + 0.5*gamma*B*Ng_*sγ

def qult_corr(modelo, gamma, c, phi, B, D, nivel_freatico):
    if D < nivel_freatico:
        gamma_eff = gamma * 0.5
    else:
        gamma_eff = gamma
    return qult(modelo, gamma_eff, c, phi, B, D)

# ---------- Funciones de cálculo ----------
def qreq(N, Mx, My, B, L):
    e_x, e_y = (Mx/N if N>0 else 0), (My/N if N>0 else 0)
    A = B*L
    sigma_avg = N/A
    sigma_max = sigma_avg*(1 + 6*e_x/B + 6*e_y/L)
    return sigma_max

def costo_total(B,L,h,concreto_Sm3,acero_Skg,excav_Sm3,D):
    vol = B*L*h
    acero_kg = 60*vol
    excav = B*L*D
    return (vol*concreto_Sm3) + (acero_kg*acero_Skg) + (excav*excav_Sm3)

def eval_tripleta(B,L,h):
    q_req = qreq(st.session_state.N, st.session_state.Mx, st.session_state.My, B, L)
    q_adm = qult_corr(st.session_state.modelo, st.session_state.gamma, st.session_state.c,
                      st.session_state.phi, B, st.session_state.D, st.session_state.nivel_freatico) / st.session_state.FS
    cumple = q_adm >= q_req
    costo = costo_total(B,L,h, st.session_state.concreto_Sm3, st.session_state.acero_Skg,
                        st.session_state.excav_Sm3, st.session_state.D)
    return q_req, q_adm, cumple, costo

# ---------- Entradas ----------
st.title("Optimización de Cimentaciones Superficiales")

st.markdown("## Propiedades del suelo")
colS1, colS2, colS3 = st.columns(3)
with colS1:
    st.session_state.gamma = st.number_input("γ (kN/m³)", value=st.session_state.gamma)
    st.session_state.c = st.number_input("Cohesión c (kPa)", value=st.session_state.c)
with colS2:
    st.session_state.phi = st.number_input("φ (°)", value=st.session_state.phi)
    st.session_state.Es = st.number_input("Módulo Eₛ (kPa)", value=st.session_state.Es)
with colS3:
    st.session_state.nu = st.number_input("ν (Poisson)", value=st.session_state.nu, step=0.05)
    st.session_state.nivel_freatico = st.number_input("Nivel freático (m)", value=st.session_state.nivel_freatico)

st.markdown("## Cargas")
colC1, colC2, colC3 = st.columns(3)
with colC1: st.session_state.N = st.number_input("Carga axial N (kN)", value=st.session_state.N)
with colC2: st.session_state.Mx = st.number_input("Momento Mx (kN·m)", value=st.session_state.Mx)
with colC3: st.session_state.My = st.number_input("Momento My (kN·m)", value=st.session_state.My)

st.markdown("## Materiales cimentación")
colM1, colM2, colM3 = st.columns(3)
with colM1: st.session_state.fc = st.number_input("f'c (MPa)", value=st.session_state.fc)
with colM2: st.session_state.fy = st.number_input("fy (MPa)", value=st.session_state.fy)
with colM3: st.session_state.recubrimiento = st.number_input("Recubrimiento (m)", value=st.session_state.recubrimiento)

st.markdown("## Factores y límites")
colF1, colF2 = st.columns(2)
with colF1: st.session_state.FS = st.number_input("FS (capacidad)", value=st.session_state.FS)
with colF2: st.session_state.asent_max = st.number_input("Asentamiento máximo (m)", value=st.session_state.asent_max)

st.markdown("## Costos unitarios")
colCU1, colCU2, colCU3 = st.columns(3)
with colCU1: st.session_state.concreto_Sm3 = st.number_input("Concreto S/ m³", value=st.session_state.concreto_Sm3)
with colCU2: st.session_state.acero_Skg = st.number_input("Acero S/ kg", value=st.session_state.acero_Skg)
with colCU3: st.session_state.excav_Sm3 = st.number_input("Excavación S/ m³", value=st.session_state.excav_Sm3)

# ---------- Rango diseño ----------
st.markdown("## Rangos de diseño")
r1, r2, r3 = st.columns(3)
with r1: st.session_state.B_min, st.session_state.B_max = st.slider("Base B (m)", 0.5, 6.0, (st.session_state.B_min, st.session_state.B_max))
with r2: st.session_state.L_min, st.session_state.L_max = st.slider("Largo L (m)", 0.5, 6.0, (st.session_state.L_min, st.session_state.L_max))
with r3: st.session_state.h_min, st.session_state.h_max = st.slider("Altura h (m)", 0.3, 2.0, (st.session_state.h_min, st.session_state.h_max))

# ---------- Botón ----------
if st.button("🔎 Analizar soluciones"):
    Bs = np.linspace(st.session_state.B_min, st.session_state.B_max, st.session_state.nB)
    Ls = np.linspace(st.session_state.L_min, st.session_state.L_max, st.session_state.nL)
    hs = np.linspace(st.session_state.h_min, st.session_state.h_max, st.session_state.nh)
    rows=[]
    for B in Bs:
        for L in Ls:
            for h in hs:
                q_req, q_adm, ok, costo = eval_tripleta(B,L,h)
                if ok:
                    rows.append([B,L,h,q_req,q_adm,costo])
    if not rows:
        st.error("⚠️ No se encontraron soluciones que cumplan capacidad portante.")
    else:
        df = pd.DataFrame(rows, columns=["B","L","h","q_req","q_adm","costo"]).sort_values("costo")
        st.dataframe(df.head(10), use_container_width=True)
        mejor = df.iloc[0]
        st.success(f"Mejor: B={mejor.B:.2f} m, L={mejor.L:.2f} m, h={mejor.h:.2f} m, Costo S/ {mejor.costo:.2f}")
        fig = px.scatter(df, x="B", y="L", color="costo", size="h", title="Soluciones válidas")
        st.plotly_chart(fig, use_container_width=True)
