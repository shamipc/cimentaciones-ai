# app.py
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")

# ====== Estilos (títulos pequeños, espacios compactos) ======
st.markdown("""
<style>
/* Títulos más pequeños */
h1 {font-size: 30px !important; margin-bottom: 0.4rem;}
h2 {font-size: 22px !important; margin-top: 0.8rem; margin-bottom: 0.4rem;}
h3 {font-size: 18px !important; margin-top: 0.6rem; margin-bottom: 0.3rem;}
/* Texto general más compacto */
section.main .block-container {padding-top: 1rem; padding-bottom: 1rem;}
/* Métricas compactas */
.css-1ht1j8u, .stMetric {gap: .15rem;}
.stMetric label {font-size: 12px !important;}
.stMetric [data-testid="stMetricValue"] {font-size: 20px !important;}
/* Inputs compactos */
[data-baseweb="input"] input {font-size: 14px !important;}
/* Plotly leyendas tipografía chica */
.js-plotly-plot .legend text {font-size: 12px !important;}
</style>
""", unsafe_allow_html=True)

# ---------- Utils de estado ----------
DEFAULTS = dict(
    modelo="Terzaghi (recomendado)",
    preset="Arcilla blanda (γ=17, c=25, φ=0)",
    gamma=17.0, c=25.0, phi=0.0,
    D=1.50, N=1000.0, FS=2.50,
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

# ---------- Capacidad portante (4 modelos) ----------
def Nq(phi):  return math.e ** (math.pi * math.tan(phi)) * math.tan(math.radians(45)+phi/2) ** 2
def Nc(phi):  return (Nq(phi) - 1) / math.tan(phi) if phi > 1e-6 else 5.14
def Ng(phi):  return 2 * (Nq(phi) + 1) * math.tan(phi)
def factors(modelo, phi_deg):
    phi = math.radians(phi_deg)
    if "Terzaghi" in modelo:   sc, sq, sγ = 1.3, 1.2, 0.8
    elif "Meyerhof" in modelo: sc, sq, sγ = 1.3, 1.2, 1.0
    elif "Hansen" in modelo:   sc, sq, sγ = 1.0, 1.0, 1.0
    else:                      sc, sq, sγ = 1.0, 1.0, 1.0  # Vesic
    return Nc(phi), Nq(phi), Ng(phi), sc, sq, sγ
def qult(modelo, gamma, c, phi, B, D):
    Nc_, Nq_, Ng_, sc, sq, sγ = factors(modelo, phi)
    q = gamma * D
    return c*Nc_*sc + q*Nq_*sq + 0.5*gamma*B*Ng_*sγ
def qadm(modelo, gamma, c, phi, B, D, FS):
    return qult(modelo, gamma, c, phi, B, D) / FS

# Costo y despiece
def costo(B, L, h, concreto_Sm3, acero_Skg):
    vol = B * L * h
    acero_kg = 60 * vol
    return vol * concreto_Sm3 + acero_kg * acero_Skg
def desglose_costos(B,L,h,concreto_Sm3,acero_Skg):
    vol = B*L*h
    conc = vol*concreto_Sm3
    acero = 60*vol*acero_Skg
    return vol, conc, acero, conc+acero

# ---------- Encabezado ----------
st.title("Optimización de Cimentaciones")
st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta")
st.info("Ingresa **los datos** en la parte inferior y pulsa **Analizar y optimizar** u **Optimizar con GA**.")

# ---------- Entradas ----------
st.markdown("## Parámetros de entrada")
colM, colG, colD = st.columns([1.2, 1, 1])
with colM:
    st.session_state.modelo = st.selectbox(
        "Modelo de capacidad",
        ["Terzaghi (recomendado)", "Meyerhof", "Hansen", "Vesic"],
        index=["Terzaghi (recomendado)", "Meyerhof", "Hansen", "Vesic"].index(st.session_state.modelo),
    )
    st.session_state.preset = st.selectbox(
        "Preset de suelo (rápido)",
        ["Arcilla blanda (γ=17, c=25, φ=0)",
         "Arena densa (γ=18, c=0, φ=35)",
         "Grava/cemento (γ=20, c=15, φ=30)"],
        index=["Arcilla blanda (γ=17, c=25, φ=0)",
               "Arena densa (γ=18, c=0, φ=35)",
               "Grava/cemento (γ=20, c=15, φ=30)"].index(st.session_state.preset),
    )
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

# ---------- Costos y rangos ----------
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
    q_req = st.session_state.N / (B*L)
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
    rng = np.random.default_rng(42)
    def rand_ind():
        return np.array([
            rng.uniform(st.session_state.B_min, st.session_state.B_max),
            rng.uniform(st.session_state.L_min, st.session_state.L_max),
            rng.uniform(st.session_state.h_min, st.session_state.h_max),
        ])
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
        idx = np.argsort(scores); P = P[idx]
        if best is None or scores[idx[0]] < best[0]: best = (scores[idx[0]], P[0].copy())
        elite = P[:max(4, pop//5)]
        childs = []
        while len(childs) < pop - len(elite):
            a, b = elite[rng.integers(0,len(elite))], elite[rng.integers(0,len(elite))]
            alpha = rng.random(); c = alpha*a + (1-alpha)*b
            if rng.random() < mut: c += rng.normal(scale=[0.1,0.1,0.05])
            c[0] = np.clip(c[0], st.session_state.B_min, st.session_state.B_max)
            c[1] = np.clip(c[1], st.session_state.L_min, st.session_state.L_max)
            c[2] = np.clip(c[2], st.session_state.h_min, st.session_state.h_max)
            childs.append(c)
        P = np.vstack([elite] + childs)
    B,L,h = best[1]
    q_req, q_adm, ok = eval_tripleta(B,L,h)
    return dict(B=B,L=L,h=h,q_req=q_req,q_adm=q_adm,ok=ok,
                costo=costo(B,L,h, st.session_state.concreto_Sm3, st.session_state.acero_Skg))

# ---------- Botones ----------
c1, c2, c3 = st.columns([1,1,1])
with c1: run_grid = st.button("🔎 Analizar y optimizar", use_container_width=True)
with c2: run_ga   = st.button("🧬 Optimizar con GA (simple)", use_container_width=True)
with c3:
    if st.button("↺ Restablecer", use_container_width=True, type="secondary"):
        reset_all()

# ---------- Render compacto ----------
def kpi(col, label, value, suf=""):
    col.metric(label, f"{value:.2f}{suf}")

def pinta_resultado_mejor(mejor, titulo="Mejor solución"):
    st.markdown(f"### {titulo}")
    vol, conc, acero, total = desglose_costos(
        mejor["B"], mejor["L"], mejor["h"],
        st.session_state.concreto_Sm3, st.session_state.acero_Skg
    )
    margen = mejor["q_adm"]/mejor["q_req"] if mejor["q_req"]>0 else np.nan

    r1,r2,r3,r4,r5,r6 = st.columns(6)
    kpi(r1,"B (m)", mejor["B"]);  kpi(r2,"L (m)", mejor["L"]);  kpi(r3,"h (m)", mejor["h"])
    kpi(r4,"Costo (S/)", mejor["costo"]); kpi(r5,"Área (m²)", mejor["B"]*mejor["L"]); kpi(r6,"FS real q_adm/q_req", margen)

    # q_req vs q_adm (compacto)
    fig = px.bar(x=["q_req","q_adm"], y=[mejor["q_req"], mejor["q_adm"]],
                 labels={"x":"Tipo","y":"kPa"}, text=[f"{mejor['q_req']:.1f}", f"{mejor['q_adm']:.1f}"],
                 title="q_req vs q_adm", template="simple_white")
    fig.update_layout(height=280, font_size=12, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Pastel de costos
    figc = px.pie(values=[conc, acero], names=["Concreto","Acero"], hole=.35,
                  title="Desglose de costos", template="simple_white")
    figc.update_layout(height=280, font_size=12, margin=dict(l=20,r=20,t=30,b=10))
    st.plotly_chart(figc, use_container_width=True)

def conclusiones(mejor, origen="Grilla"):
    vol, conc, acero, total = desglose_costos(
        mejor["B"], mejor["L"], mejor["h"],
        st.session_state.concreto_Sm3, st.session_state.acero_Skg
    )
    margen = mejor["q_adm"]/mejor["q_req"] if mejor["q_req"]>0 else np.nan
    st.markdown("#### Recomendaciones")
    if margen >= 1.15:
        st.write("• **Buen margen** entre capacidad y demanda (q_adm/q_req ≥ 1.15). Puedes explorar pequeños recortes en h para reducir costo.")
    elif margen >= 1.05:
        st.write("• **Margen justo**. Evita recortes. Si se espera variabilidad, considera +5–10% en B o L.")
    else:
        st.write("• **Margen insuficiente**. Aumenta B o L, o reduce FS/carga según criterio del proyecto.")

    st.write("• Revisa sensibilidad con **Meyerhof/Hansen/Vesic** para validar robustez del diseño.")
    st.write("• Verifica normativas locales y consideraciones constructivas (recubrimientos, asentamientos).")

    st.markdown("#### Conclusión breve (para informe)")
    texto = (
        f"Con el modelo **{st.session_state.modelo}**, la solución óptima ({origen}) "
        f"resultó en B={mejor['B']:.2f} m, L={mejor['L']:.2f} m, h={mejor['h']:.2f} m, "
        f"con costo estimado **S/ {mejor['costo']:.0f}**. La relación **q_adm/q_req = {mejor['q_adm']/mejor['q_req']:.2f}** "
        f"indica que el diseño cumple la capacidad admisible con un margen {'adecuado' if margen>=1.1 else 'ajustado'}. "
        f"El costo se compone de ~S/ {conc:.0f} (concreto) y ~S/ {acero:.0f} (acero). "
        f"Se recomienda contrastar el resultado con **otro modelo de capacidad** y considerar ajustes menores "
        f"si se desea mayor robustez o economía."
    )
    st.success(texto)

# ---------- Resultados ----------
if run_grid:
    validos, df_all = grid_search()
    st.markdown("## Resultado (búsqueda en grilla)")
    if validos.empty:
        st.warning("No se encontraron soluciones que cumplan la **capacidad admisible**. "
                   "Prueba con **B y L mayores**, **φ o c** más altos, **FS menor** o **carga menor**.")
    else:
        mejor = validos.iloc[0].to_dict()
        pinta_resultado_mejor(mejor, "Mejor solución (grilla)")
        # Dispersión B-L
        fig2 = px.scatter(validos, x="B", y="L", color="costo", labels={"costo":"Costo (S/)"},
                          title="Candidatos válidos (color=costo)", template="simple_white")
        fig2.update_layout(height=380, font_size=12, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("#### Top 10 por costo")
        st.dataframe(validos[["B","L","h","q_req","q_adm","costo"]].head(10), use_container_width=True)
        conclusiones(mejor, "grilla")

if run_ga:
    st.markdown("## Resultado GA (simple)")
    out = ga_simple()
    if not out["ok"]:
        st.warning("El **GA simple** no encontró un candidato que cumpla. "
                   "Amplía rangos o disminuye FS ligeramente.")
    else:
        pinta_resultado_mejor(out, "Mejor solución (GA)")
        conclusiones(out, "GA")

# ---------- Barra secundaria ----------
st.divider()
b1, b2 = st.columns([1,1])
with b1:
    if st.button("🧪 Cargar demo segura", type="secondary"):
        set_demo_soft(); st.rerun()
with b2:
    st.caption("Resultados compactos con conclusiones para tu informe.")

