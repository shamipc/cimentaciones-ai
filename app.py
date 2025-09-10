# app.py
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")

# ---------- Utils de estado (evita errores al reset) ----------
DEFAULTS = dict(
    modelo="Terzaghi (recomendado)",
    preset="Arcilla blanda (γ=17, c=25, φ=0)",
    gamma=17.0, c=25.0, phi=0.0,  # kN/m3, kPa, º
    D=1.50, N=1000.0, FS=2.50,    # m, kN, -
    concreto_Sm3=650.0, acero_Skg=5.50,
    B_min=1.20, B_max=3.20, L_min=1.60, L_max=4.20, h_min=0.50, h_max=1.10,
    nB=30, nL=30, nh=10,
)

def ensure_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

def set_demo_soft():
    st.session_state.update(DEFAULTS)

def reset_all():
    st.session_state.clear()
    ensure_state()
    st.rerun()

ensure_state()

# ---------- Capacidad portante (4 modelos clásicos) ----------
# Factores de capacidad (Nc, Nq, Nγ) para φ en radianes
def Nq(phi):  return math.e ** (math.pi * math.tan(phi)) * math.tan(math.radians(45)+phi/2) ** 2
def Nc(phi):  return (Nq(phi) - 1) / math.tan(phi) if phi > 1e-6 else 5.14
def Ng(phi):  return 2 * (Nq(phi) + 1) * math.tan(phi)

def factors(modelo, phi_deg):
    phi = math.radians(phi_deg)
    # Coeficientes por modelo (k, m, n) para componentes c, q, γ
    # Referencias clásicas aproximadas
    if "Terzaghi" in modelo:
        sc, sq, sγ = 1.3, 1.2, 0.8
    elif "Meyerhof" in modelo:
        sc, sq, sγ = 1.3, 1.2, 1.0
    elif "Hansen" in modelo:
        sc, sq, sγ = 1.0, 1.0, 1.0
    else:  # Vesic
        sc, sq, sγ = 1.0, 1.0, 1.0
    return Nc(phi), Nq(phi), Ng(phi), sc, sq, sγ

def qult(modelo, gamma, c, phi, B, D):
    Nc_, Nq_, Ng_, sc, sq, sγ = factors(modelo, phi)
    # Formulación general: q_ult = c*Nc*sc + q*Nq*sq + 0.5*γ*B*Nγ*sγ
    q = gamma * D  # presión de sobrecarga
    return c*Nc_*sc + q*Nq_*sq + 0.5*gamma*B*Ng_*sγ  # kPa

def qadm(modelo, gamma, c, phi, B, D, FS):
    return qult(modelo, gamma, c, phi, B, D) / FS

# Costo aproximado (volumen * costo concreto + % acero)
def costo(B, L, h, concreto_Sm3, acero_Skg):
    vol = B * L * h            # m3
    acero_kg = 60 * vol        # 60 kg/m3 típico: ajustable
    return vol * concreto_Sm3 + acero_kg * acero_Skg

# ---------- Encabezado ----------
st.title("Optimización de Cimentaciones")
st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta")
st.info("Ingresa **los datos** en la parte inferior y pulsa **Analizar y optimizar** u **Optimizar con GA**.")

# ---------- BLOQUE 1: Parámetros de entrada ----------
st.markdown("## Parámetros de entrada")
colM, colG, colD = st.columns([1.2, 1, 1])

with colM:
    st.session_state.modelo = st.selectbox(
        "Modelo de capacidad",
        ["Terzaghi (recomendado)", "Meyerhof", "Hansen", "Vesic"],
        index=["Terzaghi (recomendado)", "Meyerhof", "Hansen", "Vesic"].index(st.session_state.modelo),
        help="Usa Terzaghi como base; prueba otros para sensibilidad.",
    )
    st.session_state.preset = st.selectbox(
        "Preset de suelo (rápido)",
        ["Arcilla blanda (γ=17, c=25, φ=0)",
         "Arena densa (γ=18, c=0, φ=35)",
         "Grava/cemento (γ=20, c=15, φ=30)"],
        index= ["Arcilla blanda (γ=17, c=25, φ=0)",
                "Arena densa (γ=18, c=0, φ=35)",
                "Grava/cemento (γ=20, c=15, φ=30)"].index(st.session_state.preset)
    )
    # aplica preset sin sobreescribir manual si cambió
    if st.session_state.preset.startswith("Arcilla"):
        st.session_state.gamma, st.session_state.c, st.session_state.phi = 17.0, 25.0, 0.0
    elif st.session_state.preset.startswith("Arena"):
        st.session_state.gamma, st.session_state.c, st.session_state.phi = 18.0, 0.0, 35.0
    else:
        st.session_state.gamma, st.session_state.c, st.session_state.phi = 20.0, 15.0, 30.0

with colG:
    st.session_state.gamma = st.number_input("Peso unitario γ (kN/m³)", value=float(st.session_state.gamma), step=0.5)
    st.session_state.c      = st.number_input("Cohesión c (kPa)", value=float(st.session_state.c), step=1.0)
    st.session_state.phi    = st.number_input("Ángulo de fricción φ (°)", value=float(st.session_state.phi), step=0.5)

with colD:
    st.session_state.D  = st.number_input("Profundidad D (m)", value=float(st.session_state.D), step=0.1)
    st.session_state.N  = st.number_input("Carga N (kN)", value=float(st.session_state.N), step=10.0)
    st.session_state.FS = st.number_input("Factor de seguridad FS", value=float(st.session_state.FS), step=0.1, min_value=1.1)

# ---------- BLOQUE 2: Costos y rangos ----------
st.markdown("## Costos")
colC, colA = st.columns(2)
with colC:
    st.session_state.concreto_Sm3 = st.number_input("Concreto (S/ por m³)", value=float(st.session_state.concreto_Sm3), step=10.0)
with colA:
    st.session_state.acero_Skg    = st.number_input("Acero (S/ por kg)", value=float(st.session_state.acero_Skg), step=0.1)

st.markdown("## Rangos de diseño (B, L, h) ↪")
r1, r2, r3 = st.columns(3)
with r1:
    st.session_state.B_min, st.session_state.B_max = st.slider("Base B (m)", 0.6, 5.0,
        (float(st.session_state.B_min), float(st.session_state.B_max)), step=0.05)
    st.session_state.nB = st.number_input("Resolución B", value=int(st.session_state.nB), min_value=5, step=1)
with r2:
    st.session_state.L_min, st.session_state.L_max = st.slider("Largo L (m)", 0.6, 6.0,
        (float(st.session_state.L_min), float(st.session_state.L_max)), step=0.05)
    st.session_state.nL = st.number_input("Resolución L", value=int(st.session_state.nL), min_value=5, step=1)
with r3:
    st.session_state.h_min, st.session_state.h_max = st.slider("Altura h (m)", 0.30, 2.00,
        (float(st.session_state.h_min), float(st.session_state.h_max)), step=0.02)
    st.session_state.nh = st.number_input("Resolución h", value=int(st.session_state.nh), min_value=3, step=1)

# ---------- Motor: malla + GA ----------
def eval_tripleta(B, L, h):
    q_req = st.session_state.N / (B*L)        # kPa (=kN/m2)
    q_adm = qadm(st.session_state.modelo, st.session_state.gamma, st.session_state.c,
                 st.session_state.phi, B, st.session_state.D, st.session_state.FS)
    ok = q_adm >= q_req
    return q_req, q_adm, ok

def grid_search():
    Bs = np.linspace(st.session_state.B_min, st.session_state.B_max, st.session_state.nB)
    Ls = np.linspace(st.session_state.L_min, st.session_state.L_max, st.session_state.nL)
    hs = np.linspace(st.session_state.h_min, st.session_state.h_max, st.session_state.nh)

    rows = []
    for B in Bs:
        for L in Ls:
            for h in hs:
                q_req, q_adm, ok = eval_tripleta(B, L, h)
                rows.append([B, L, h, q_adm, q_req,
                             costo(B,L,h, st.session_state.concreto_Sm3, st.session_state.acero_Skg),
                             ok])
    df = pd.DataFrame(rows, columns=["B","L","h","q_adm","q_req","costo","ok"])
    return df[df["ok"]].sort_values("costo", ascending=True), df

def ga_simple(max_it=80, pop=60, mut=0.25):
    # Población inicial uniforme en los rangos
    rng = np.random.default_rng(42)
    def rand_ind():
        B = rng.uniform(st.session_state.B_min, st.session_state.B_max)
        L = rng.uniform(st.session_state.L_min, st.session_state.L_max)
        h = rng.uniform(st.session_state.h_min, st.session_state.h_max)
        return np.array([B,L,h])

    P = np.array([rand_ind() for _ in range(pop)])
    best = None
    for _ in range(max_it):
        scores = []
        for ind in P:
            B,L,h = ind
            q_req, q_adm, ok = eval_tripleta(B,L,h)
            if not ok:
                penalty = 1e6 + (q_req - q_adm)**2
            else:
                penalty = costo(B,L,h, st.session_state.concreto_Sm3, st.session_state.acero_Skg)
            scores.append(penalty)
        scores = np.array(scores)
        idx = np.argsort(scores)
        P = P[idx]
        if best is None or scores[idx[0]] < best[0]:
            best = (scores[idx[0]], P[0].copy())

        # selección y reproducción (elitismo 20%)
        elite = P[:max(4, pop//5)]
        # cruza simple
        childs = []
        while len(childs) < pop - len(elite):
            a, b = elite[rng.integers(0,len(elite))], elite[rng.integers(0,len(elite))]
            alpha = rng.random()
            c = alpha*a + (1-alpha)*b
            # mutación
            if rng.random() < mut:
                c += rng.normal(scale=[0.1,0.1,0.05])
            # límites
            c[0] = np.clip(c[0], st.session_state.B_min, st.session_state.B_max)
            c[1] = np.clip(c[1], st.session_state.L_min, st.session_state.L_max)
            c[2] = np.clip(c[2], st.session_state.h_min, st.session_state.h_max)
            childs.append(c)
        P = np.vstack([elite] + childs)

    B,L,h = best[1]
    q_req, q_adm, ok = eval_tripleta(B,L,h)
    return dict(B=B,L=L,h=h,q_req=q_req,q_adm=q_adm,ok=ok,
                costo=costo(B,L,h, st.session_state.concreto_Sm3, st.session_state.acero_Skg))

# ---------- Botones principales ----------
c1, c2, c3 = st.columns([1,1,1])
with c1:
    run_grid = st.button("🔎 Analizar y optimizar", use_container_width=True)
with c2:
    run_ga   = st.button("🧬 Optimizar con GA (simple)", use_container_width=True)
with c3:
    if st.button("↺ Restablecer", use_container_width=True, type="secondary"):
        reset_all()

# ---------- Ejecución + presentación ----------
def kpi_row(col, label, value, suffix=""):
    col.metric(label, f"{value:.2f}{suffix}")

def pinta_resultado_mejor(mejor):
    st.markdown("### Mejor solución")
    k1,k2,k3,k4 = st.columns(4)
    kpi_row(k1, "B (m)", mejor["B"])
    kpi_row(k2, "L (m)", mejor["L"])
    kpi_row(k3, "h (m)", mejor["h"])
    kpi_row(k4, "Costo (S/)", mejor["costo"])

    fig = px.bar(x=["q_req","q_adm"], y=[mejor["q_req"], mejor["q_adm"]],
                 labels={"x":"Tipo","y":"kPa"}, text=[f"{mejor['q_req']:.1f}", f"{mejor['q_adm']:.1f}"],
                 title="q_req vs q_adm")
    st.plotly_chart(fig, use_container_width=True)

if run_grid:
    validos, df_all = grid_search()
    st.markdown("## Resultado (búsqueda en grilla)")
    if validos.empty:
        st.warning("No se encontraron soluciones que cumplan la **capacidad admisible**. "
                   "Prueba con **B y L mayores**, **φ o c** más altos, **FS menor** o **carga menor**.")
    else:
        mejor = validos.iloc[0].to_dict()
        pinta_resultado_mejor(mejor)

        # nube B-L coloreada por costo (h cercano al óptimo)
        fig2 = px.scatter(validos, x="B", y="L", color="costo",
                          labels={"costo":"Costo (S/)"},
                          title="Candidatos válidos (color=costo)")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Top 10 por costo")
        st.dataframe(validos[["B","L","h","q_req","q_adm","costo"]].head(10), use_container_width=True)

        # recomendaciones
        st.markdown("#### Recomendaciones")
        ok_margen = "Buen margen" if mejor["q_adm"] > 1.1*mejor["q_req"] else "Margen justo"
        st.write(f"• {ok_margen} entre capacidad y demanda.")
        st.write("• Si buscas **menor costo**, explora h ligeramente menor y ajusta B/L manteniendo q_adm ≥ q_req.")
        st.write("• Contrasta con **otro modelo** (Meyerhof / Hansen / Vesic) para análisis de sensibilidad.")

if run_ga:
    st.markdown("## Resultado GA (simple)")
    out = ga_simple()
    if not out["ok"]:
        st.warning("El **GA simple** no encontró un candidato que cumpla. "
                   "Amplía rangos o disminuye FS ligeramente.")
    else:
        pinta_resultado_mejor(out)

# ---------- Barra de acciones secundaria ----------
st.divider()
b1, b2 = st.columns([1,1])
with b1:
    if st.button("🧪 Cargar demo segura", type="secondary"):
        set_demo_soft()
        st.rerun()
with b2:
    st.caption("Versión estable: rango amplio + GA de respaldo para evitar ‘sin resultados’.")

