# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------
# Utilidades
# --------------------------
def rad(deg):
    return deg * math.pi / 180.0

def capacidad_factores(phi_deg):
    """ Factores Nq, Nc, Nγ en función de φ (deg). Aproximaciones estándar. """
    if phi_deg <= 0.0:
        # Caso cohesivo puro
        Nq = 1.0
        Nc = 5.14
        N_gamma = 0.0
        return Nq, Nc, N_gamma

    phi = rad(phi_deg)
    Nq = math.e ** (math.pi * math.tan(phi)) * (math.tan(math.pi / 4.0 + phi / 2.0) ** 2)
    Nc = (Nq - 1.0) / math.tan(phi)
    # Aproximación habitual para Nγ (existen varias; esta funciona bien para 0°–40°)
    N_gamma = 2.0 * (Nq + 1.0) * math.tan(phi)
    return Nq, Nc, N_gamma

def factores_forma_y_profundidad(modelo, B, L, D, phi_deg):
    """ Factores de forma y profundidad simplificados por modelo clásico. """
    B_L = max(B / L, 1e-6)
    D_B = max(D / B, 0.0)

    # Defaults conservadores
    sc = 1.0; sq = 1.0; sγ = 1.0
    dc = 1.0; dq = 1.0; dγ = 1.0

    if modelo in ["Meyerhof", "Vesic", "Hansen", "Terzaghi"]:
        # Meyerhof: (aprox. muy usada; Terzaghi no contempla todos)
        sc = 1.0 + 0.2 * B_L
        sq = 1.0 + 0.1 * B_L
        sγ = max(1.0 - 0.4 * B_L, 0.6)

        dc = 1.0 + 0.2 * D_B
        dq = 1.0 + 0.1 * D_B
        dγ = 1.0  # Poco efecto en aprox simple
    # Para personalizado no aplicamos nada aquí (se mete abajo vía caja de texto si quisieras)

    return sc, sq, sγ, dc, dq, dγ

def q_admisible(modelo, gamma, c, phi, D, B, L, FS,
                personalizados=None):
    """
    q_adm = q_ult/FS  con:
    q_ult = c*Nc*sc*dc + 0.5*γ*B*Nγ*sγ*dγ + γ*D*Nq*sq*dq
    """
    # Factores base
    if modelo == "Personalizado" and personalizados:
        Nq = personalizados.get("Nq", 1.0)
        Nc = personalizados.get("Nc", 5.14)
        Ng = personalizados.get("Nγ", 0.0)
        sc = personalizados.get("sc", 1.0)
        sq = personalizados.get("sq", 1.0)
        sg = personalizados.get("sγ", 1.0)
        dc = personalizados.get("dc", 1.0)
        dq = personalizados.get("dq", 1.0)
        dg = personalizados.get("dγ", 1.0)
    else:
        Nq, Nc, Ng = capacidad_factores(phi)
        sc, sq, sg, dc, dq, dg = factores_forma_y_profundidad(modelo, B, L, D, phi)

    q_ult = c * Nc * sc * dc + 0.5 * gamma * B * Ng * sg * dg + gamma * D * Nq * sq * dq
    return q_ult / max(FS, 1e-6)

def costo_cimentacion(B, L, h, costo_concreto_Sm3=650.0, costo_acero_Skg=5.5):
    """
    Costo sencillo: concreto + acero (aprox. 90 kg/m3 de acero)
    Puedes ajustar si deseas.
    """
    vol = B * L * h
    acero_kg_m3 = 90.0
    costo = vol * (costo_concreto_Sm3 + acero_kg_m3 * costo_acero_Skg)
    return costo

def evaluar_diseno(modelo, gamma, c, phi, D, FS, N, B, L, h,
                   costo_concreto, costo_acero, personalizados=None):
    area = B * L
    q_req = N / max(area, 1e-6)
    q_adm = q_admisible(modelo, gamma, c, phi, D, B, L, FS, personalizados)
    ok = q_req <= q_adm
    costo = costo_cimentacion(B, L, h, costo_concreto, costo_acero)
    return ok, q_req, q_adm, costo

def barrido(modelo, gamma, c, phi, D, FS, N,
            Brange, Lrange, hrange, resB, resL, resh,
            costo_concreto, costo_acero, personalizados=None):
    Bs = np.linspace(Brange[0], Brange[1], resB)
    Ls = np.linspace(Lrange[0], Lrange[1], resL)
    hs = np.linspace(hrange[0], hrange[1], resh)

    registros = []
    for B in Bs:
        for L in Ls:
            for h in hs:
                ok, q_req, q_adm, costo = evaluar_diseno(
                    modelo, gamma, c, phi, D, FS, N, B, L, h,
                    costo_concreto, costo_acero, personalizados
                )
                if ok:
                    registros.append({
                        "B": round(B, 3),
                        "L": round(L, 3),
                        "h": round(h, 3),
                        "q_adm": round(q_adm, 1),
                        "q_req": round(q_req, 1),
                        "costo": round(costo, 0)
                    })
    df = pd.DataFrame(registros)
    if df.empty:
        return None, None, None, None

    # Mejor por costo
    idx = df["costo"].idxmin()
    Bopt, Lopt, hopt = df.loc[idx, ["B", "L", "h"]].values
    qadm_opt, qreq_opt, cost_opt = df.loc[idx, ["q_adm", "q_req", "costo"]].values
    return (Bopt, Lopt, hopt), qreq_opt, qadm_opt, df.sort_values("costo").reset_index(drop=True)

# --------------------------
# GA muy simple y rápido
# --------------------------
def ga_optimizar(modelo, gamma, c, phi, D, FS, N,
                 Brange, Lrange, hrange, costo_concreto, costo_acero,
                 personalizados=None,
                 pop=30, gens=40, mut=0.15):
    rng = np.random.default_rng(42)
    def random_ind():
        return np.array([
            rng.uniform(Brange[0], Brange[1]),
            rng.uniform(Lrange[0], Lrange[1]),
            rng.uniform(hrange[0], hrange[1])
        ])

    def fitness(ind):
        B, L, h = ind
        ok, q_req, q_adm, costo = evaluar_diseno(
            modelo, gamma, c, phi, D, FS, N, B, L, h,
            costo_concreto, costo_acero, personalizados
        )
        if not ok:
            return 1e12  # penalización
        return costo

    P = np.array([random_ind() for _ in range(pop)])
    F = np.array([fitness(ind) for ind in P])

    for _ in range(gens):
        # torneo
        idx1 = rng.integers(0, pop, pop)
        idx2 = rng.integers(0, pop, pop)
        winners = np.where(F[idx1] < F[idx2], P[idx1], P[idx2])
        # cruza simple
        children = winners.copy()
        cut = rng.integers(1, 3)
        children[1::2, :cut] = winners[::2, :cut]
        children[::2, cut:] = winners[1::2, cut:]
        # mutación
        mmask = rng.random(children.shape) < mut
        noise = rng.normal(0, 0.05, children.shape)
        children = children + mmask * noise
        # límites
        children[:, 0] = np.clip(children[:, 0], Brange[0], Brange[1])
        children[:, 1] = np.clip(children[:, 1], Lrange[0], Lrange[1])
        children[:, 2] = np.clip(children[:, 2], hrange[0], hrange[1])

        P2 = np.vstack([P, children])
        F2 = np.array([fitness(ind) for ind in P2])
        idx = np.argsort(F2)[:pop]
        P, F = P2[idx], F2[idx]

    best = P[0]
    Bopt, Lopt, hopt = best.tolist()
    ok, q_req, q_adm, cost = evaluar_diseno(
        modelo, gamma, c, phi, D, FS, N, Bopt, Lopt, hopt, costo_concreto, costo_acero, personalizados
    )
    if not ok:
        return None, None, None, None
    df = pd.DataFrame([{
        "B": round(Bopt, 3), "L": round(Lopt, 3), "h": round(hopt, 3),
        "q_adm": round(q_adm, 1), "q_req": round(q_req, 1), "costo": round(cost, 0)
    }])
    return (Bopt, Lopt, hopt), q_req, q_adm, df

# --------------------------
# UI
# --------------------------
st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide", page_icon="🧱")
st.title("Optimización de Cimentaciones  ")

st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta")

st.info("Ingresa los **datos** en la parte izquierda y pulsa **Analizar y optimizar** o **Optimizar con GA**.")

# --------------------------
# Estado y valores por defecto (válidos)
# --------------------------
defaults = dict(
    modelo="Terzaghi (recomendado)",
    preset="Arcilla blanda (γ=17, c=25, φ=0)",
    gamma=18.0,
    c=20.0,
    phi=30.0,
    D=1.5,
    N=800.0,
    FS=2.0,
    costo_concreto=650.0,
    costo_acero=5.5,
    Brange=(1.2, 3.2),
    Lrange=(1.6, 4.2),
    hrange=(0.50, 1.10),
    resB=30, resL=30, resh=10
)

for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# --------------------------
# Panel de entrada en 3 columnas
# --------------------------
with st.container():
    cols = st.columns(3)

    # Modelo de capacidad
    modelos = [
        "Terzaghi (recomendado)",
        "Meyerhof",
        "Vesic",
        "Hansen",
        "Personalizado"
    ]
    with cols[0]:
        st.subheader("Parámetros de entrada", divider=False)
        modelo = st.selectbox("Modelo de capacidad", modelos,
                              index=modelos.index(st.session_state["modelo"])
                              if st.session_state["modelo"] in modelos else 0)

    # Presets rápidos de suelo
    presets = {
        "Arcilla blanda (γ=17, c=25, φ=0)": (17.0, 25.0, 0.0),
        "Arena densa (γ=19, c=0, φ=35)": (19.0, 0.0, 35.0),
        "Grava densa (γ=20, c=0, φ=40)": (20.0, 0.0, 40.0),
        "Arcilla arenosa (γ=18, c=30, φ=20)": (18.0, 30.0, 20.0)
    }
    with cols[0]:
        preset = st.selectbox("Preset de suelo (rápido)", list(presets.keys()),
                              index=list(presets.keys()).index(st.session_state["preset"])
                              if st.session_state["preset"] in presets else 0)
        if st.session_state.get("preset") != preset:
            st.session_state["gamma"], st.session_state["c"], st.session_state["phi"] = presets[preset]
            st.session_state["preset"] = preset

    with cols[1]:
        gamma = st.number_input("Peso unitario γ (kN/m³)", min_value=10.0, max_value=25.0, step=0.1,
                                value=float(st.session_state["gamma"]))
        c = st.number_input("Cohesión c (kPa)", min_value=0.0, max_value=400.0, step=1.0,
                            value=float(st.session_state["c"]))
        phi = st.number_input("Ángulo de fricción φ (°)", min_value=0.0, max_value=45.0, step=0.5,
                              value=float(st.session_state["phi"]))

    with cols[2]:
        D = st.number_input("Profundidad D (m)", min_value=0.0, max_value=5.0, step=0.1,
                            value=float(st.session_state["D"]))
        N = st.number_input("Carga N (kN)", min_value=10.0, max_value=5000.0, step=10.0,
                            value=float(st.session_state["N"]))
        FS = st.number_input("Factor de seguridad", min_value=1.1, max_value=4.0, step=0.1,
                             value=float(st.session_state["FS"]))

# Costos & Rangos
st.markdown("### Costos y rangos de diseño")
c1, c2 = st.columns([1, 2], gap="large")

with c1:
    costo_concreto = st.number_input("Concreto (S/ por m³)", min_value=200.0, max_value=1500.0, step=10.0,
                                     value=float(st.session_state["costo_concreto"]))
    costo_acero = st.number_input("Acero (S/ por kg)", min_value=2.0, max_value=20.0, step=0.1,
                                  value=float(st.session_state["costo_acero"]))

with c2:
    Brange = list(st.slider("Base B (m)", 0.6, 5.0,
                            (float(st.session_state["Brange"][0]), float(st.session_state["Brange"][1])), 0.01))
    Lrange = list(st.slider("Largo L (m)", 0.6, 6.0,
                            (float(st.session_state["Lrange"][0]), float(st.session_state["Lrange"][1])), 0.01))
    hrange = list(st.slider("Altura h (m)", 0.30, 2.00,
                            (float(st.session_state["hrange"][0]), float(st.session_state["hrange"][1])), 0.01))

    colr = st.columns(3)
    with colr[0]:
        resB = st.number_input("Resolución B", 10, 60, int(st.session_state["resB"]))
    with colr[1]:
        resL = st.number_input("Resolución L", 10, 60, int(st.session_state["resL"]))
    with colr[2]:
        resh = st.number_input("Resolución h", 5, 30, int(st.session_state["resh"]))

# Personalizado (opcional)
personalizados = None
if modelo == "Personalizado":
    with st.expander("Factores personalizados (opcional)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            Nq = st.number_input("Nq", 0.0, 500.0, 1.0, 0.1)
            Nc = st.number_input("Nc", 0.0, 500.0, 5.14, 0.1)
            Ng = st.number_input("Nγ", 0.0, 500.0, 0.0, 0.1)
        with c2:
            sc = st.number_input("sc", 0.1, 2.0, 1.0, 0.05)
            sq = st.number_input("sq", 0.1, 2.0, 1.0, 0.05)
            sg = st.number_input("sγ", 0.1, 2.0, 1.0, 0.05)
        with c3:
            dc = st.number_input("dc", 0.1, 2.0, 1.0, 0.05)
            dq = st.number_input("dq", 0.1, 2.0, 1.0, 0.05)
            dg = st.number_input("dγ", 0.1, 2.0, 1.0, 0.05)
        personalizados = dict(Nq=Nq, Nc=Nc, Nγ=Ng, sc=sc, sq=sq, sγ=sg, dc=dc, dq=dq, dγ=dg)

# --------------------------
# Botonera (Restablecer junto a ejecutar)
# --------------------------
bc1, bc2, bc3 = st.columns([1, 1, 1], gap="small")
with bc1:
    run_grid = st.button("🔎 Analizar y optimizar", use_container_width=True)
with bc2:
    run_ga = st.button("🧬 Optimizar con GA", use_container_width=True)
with bc3:
    if st.button("↺ Restablecer", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

# --------------------------
# Ejecutar
# --------------------------
df = None
resultado = None

if run_grid or run_ga:
    if run_grid:
        resultado = barrido(
            modelo.split(" ")[0] if modelo != "Personalizado" else "Personalizado",
            gamma, c, phi, D, FS, N,
            Brange, Lrange, hrange, int(resB), int(resL), int(resh),
            costo_concreto, costo_acero, personalizados
        )
    else:
        resultado = ga_optimizar(
            modelo.split(" ")[0] if modelo != "Personalizado" else "Personalizado",
            gamma, c, phi, D, FS, N,
            Brange, Lrange, hrange,
            costo_concreto, costo_acero, personalizados
        )

    if resultado[0] is None:
        st.warning("No se encontraron soluciones que cumplan la capacidad admisible. "
                   "Prueba con **B y L mayores**, **φ o c más altos**, **FS menor** o **carga menor**.")
    else:
        (Bopt, Lopt, hopt), qreq_opt, qadm_opt, df = resultado

        st.success("✅ Diseño óptimo encontrado")
        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1:
            st.metric("B (m)", f"{Bopt:.2f}")
        with kc2:
            st.metric("L (m)", f"{Lopt:.2f}")
        with kc3:
            st.metric("h (m)", f"{hopt:.2f}")
        with kc4:
            cost_opt = costo_cimentacion(Bopt, Lopt, hopt, costo_concreto, costo_acero)
            st.metric("Costo (S/)", f"{cost_opt:,.0f}")

        # Gráfico q_req vs q_adm
        fig = px.bar(pd.DataFrame({
            "Magnitud": ["q_req", "q_adm"], "kPa": [qreq_opt, qadm_opt]
        }), x="Magnitud", y="kPa", text="kPa",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(title="q_req vs q_adm (óptimo)", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Dispersión (candidatos válidos)
        if df is not None and len(df) > 1:
            st.markdown("#### Candidatos válidos (color = Costo, tamaño = h)")
            fig2 = px.scatter(
                df, x="B", y="L", color="costo", size="h",
                color_continuous_scale="Tealgrn"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Tabla de top resultados
        if df is not None:
            st.markdown("#### Top 30 soluciones por costo")
            st.dataframe(df.head(30), use_container_width=True)

# Nota útil si aún no se ha corrido
if not (run_grid or run_ga):
    st.info("Configura parámetros y pulsa **Analizar y optimizar**. "
            "Con los valores iniciales deberías ver un óptimo sin problema.")

