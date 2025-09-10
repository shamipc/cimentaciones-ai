import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------- Config general -------------------------
st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")
st.title("Optimización de Cimentaciones")
st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta")

# ------------------------- Utilidades -------------------------
def deg2rad(x):
    return np.deg2rad(x)

def capacidad_factores(phi_rad, modelo):
    """
    Devuelve (Nq, Nc, Ng) para el ángulo phi (rad) según modelo.
    Fórmulas clásicas (aprox). Si phi=0 => Nc ~ 5.14 (Terzaghi) y Nq=Ng=0.
    """
    if phi_rad <= 1e-6:
        # Cohesivo puro
        Nq = 1.0
        Nc = 5.14  # Terzaghi stripping
        Ng = 0.0
        return Nq, Nc, Ng

    # Comunes
    Nq = math.e ** (math.pi * math.tan(phi_rad)) * (math.tan(math.pi / 4 + phi_rad / 2) ** 2)
    Nc = (Nq - 1.0) / math.tan(phi_rad)

    if modelo == "Terzaghi (recomendado)":
        Ng = 2.0 * (Nq + 1.0) * math.tan(phi_rad)
    elif modelo == "Meyerhof":
        Ng = (Nq - 1.0) * math.tan(1.4 * phi_rad)
    elif modelo == "Vesic":
        Ng = 2.0 * (Nq + 1.0) * math.tan(phi_rad)  # simple y estable
    else:
        # Personalizado: Ng por defecto (se ignora aquí, se tomará de UI)
        Ng = 2.0 * (Nq + 1.0) * math.tan(phi_rad)
    return Nq, Nc, Ng


def factores_forma_y_profundidad(modelo, B, L, D, phi_rad):
    """
    Factores de forma y profundidad básicos (aprox) válidos para todos los modelos.
    """
    # Forma (rectangular)
    sc = 1.0 + 0.2 * (B / L)
    sq = 1.0 + 0.1 * (B / L)
    sg = 1.0 - 0.4 * (B / L)
    sg = max(sg, 0.6)  # limitar inferiormente

    # Profundidad (aprox)
    dc = 1.0 + 0.2 * (D / B)
    dq = 1.0 + 0.1 * (D / B)
    dg = 1.0

    return sc, sq, sg, dc, dq, dg


def q_admisible(modelo, c, phi_deg, gamma, B, L, D, FS, personalizados=None):
    """
    Calcula q_adm (kPa) para una zapata rectangular (B x L).
    q_adm = q_ult/FS - gamma*D (neto).
    """
    phi_rad = deg2rad(phi_deg)
    if modelo == "Personalizado" and personalizados:
        # Leer factores personalizados (ASCII)
        Nq = float(personalizados.get("Nq", 1.0))
        Nc = float(personalizados.get("Nc", 5.14))
        Ng = float(personalizados.get("Ng", 0.0))
        sc = float(personalizados.get("sc", 1.0))
        sq = float(personalizados.get("sq", 1.0))
        sg = float(personalizados.get("sg", 1.0))
        dc = float(personalizados.get("dc", 1.0))
        dq = float(personalizados.get("dq", 1.0))
        dg = float(personalizados.get("dg", 1.0))
    else:
        Nq, Nc, Ng = capacidad_factores(phi_rad, modelo)
        sc, sq, sg, dc, dq, dg = factores_forma_y_profundidad(modelo, B, L, D, phi_rad)

    # q_ult clásica de capacidad portante (rectangular)
    q_ult = c * Nc * sc * dc + 0.5 * gamma * B * Ng * sg * dg + gamma * D * Nq * sq * dq

    # q_adm neta
    q_adm = q_ult / FS - gamma * D
    return max(q_adm, 0.0)


def costo_concreto(B, L, h, precio_concreto):
    # Costo simple: volumen * precio
    return (B * L * h) * precio_concreto


def grid_search(modelo, gamma, c, phi_deg, D, N, FS,
                Bmin, Bmax, nB,
                Lmin, Lmax, nL,
                hmin, hmax, nh,
                precio_concreto,
                personalizados=None):
    """
    Explora una malla de B, L, h y devuelve:
    - mejor (dict con B,L,h,costo,q_req,q_adm)
    - DF de candidatos válidos
    """
    Bs = np.linspace(Bmin, Bmax, nB)
    Ls = np.linspace(Lmin, Lmax, nL)
    hs = np.linspace(hmin, hmax, nh)

    registros = []
    mejor = None

    for B in Bs:
        for L in Ls:
            area = B * L
            q_req = N / area  # presión requerida (kPa si N en kN y dimensiones en m)
            q_adm = q_admisible(modelo, c, phi_deg, gamma, B, L, D, FS, personalizados=personalizados)

            if q_adm >= q_req:
                for h in hs:
                    costo = costo_concreto(B, L, h, precio_concreto)
                    registros.append(dict(B=B, L=L, h=h, q_req=q_req, q_adm=q_adm, costo=costo))
                    if (mejor is None) or (costo < mejor["costo"]):
                        mejor = dict(B=B, L=L, h=h, q_req=q_req, q_adm=q_adm, costo=costo)

    df = pd.DataFrame(registros) if registros else pd.DataFrame(columns=["B", "L", "h", "q_req", "q_adm", "costo"])
    return mejor, df


def suelos_presets():
    return {
        "Arcilla blanda (γ=17, c=25, φ=0)": dict(gamma=17.0, c=25.0, phi=0.0),
        "Arena densa (γ=18.5, c=0, φ=35)": dict(gamma=18.5, c=0.0, phi=35.0),
        "Limo arenoso (γ=18, c=10, φ=25)": dict(gamma=18.0, c=10.0, phi=25.0),
    }

# ------------------------- Barra de ayuda -------------------------
with st.container():
    st.info("Ingresa **los datos** en la parte izquierda y pulsa **Analizar y optimizar** o **Optimizar con GA**.")

st.divider()

# ------------------------- Panel de entrada (2 filas) -------------------------
colA, colB, colC = st.columns([1.2, 1.0, 1.0])
colD, colE, colF = st.columns([1.2, 1.0, 1.0])

with colA:
    modelo = st.selectbox("Modelo de capacidad",
                          ["Terzaghi (recomendado)", "Meyerhof", "Vesic", "Personalizado"],
                          index=0)

with colB:
    preset_name = st.selectbox("Preset de suelo (rápido)", list(suelos_presets().keys()), index=0)
    preset = suelos_presets()[preset_name]

with colC:
    gamma = st.number_input("Peso unitario γ (kN/m³)", value=float(preset["gamma"]), step=0.1, format="%.2f")

with colD:
    c = st.number_input("Cohesión c (kPa)", value=float(preset["c"]), step=0.5, format="%.2f")

with colE:
    phi = st.number_input("Ángulo de fricción φ (°)", value=float(preset["phi"]), step=0.5, format="%.2f")

with colF:
    D = st.number_input("Profundidad D (m)", value=1.50, step=0.05, format="%.2f")

colG, colH, colI = st.columns([1.0, 1.0, 1.0])
with colG:
    N = st.number_input("Carga N (kN)", value=1000.0, step=5.0, format="%.2f")
with colH:
    FS = st.number_input("Factor de seguridad", value=2.50, step=0.1, format="%.2f")
with colI:
    precio_concreto = st.number_input("Concreto (S/ por m³)", value=650.0, step=10.0, format="%.2f")

# Rangos
st.subheader("Rangos de diseño (B, L, h)")
c1, c2, c3 = st.columns(3)
with c1:
    Bmin, Bmax = st.slider("Base B (m)", min_value=0.6, max_value=5.0, value=(1.20, 3.20), step=0.05)
    nB = st.number_input("Resolución B", value=30, step=1, min_value=5)
with c2:
    Lmin, Lmax = st.slider("Largo L (m)", min_value=0.6, max_value=6.0, value=(1.60, 4.20), step=0.05)
    nL = st.number_input("Resolución L", value=30, step=1, min_value=5)
with c3:
    hmin, hmax = st.slider("Altura h (m)", min_value=0.20, max_value=2.00, value=(0.50, 1.10), step=0.02)
    nh = st.number_input("Resolución h", value=10, step=1, min_value=3)

# Personalizado (si aplica)
personalizados = None
if modelo == "Personalizado":
    with st.expander("Factores personalizados (ASCII)"):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            Nc = st.number_input("Nc", value=5.14, step=0.1)
            sc = st.number_input("sc", value=1.0, step=0.05)
            dc = st.number_input("dc", value=1.0, step=0.05)
        with pc2:
            Nq = st.number_input("Nq", value=1.0, step=0.1)
            sq = st.number_input("sq", value=1.0, step=0.05)
            dq = st.number_input("dq", value=1.0, step=0.05)
        with pc3:
            Ng = st.number_input("Ng", value=0.0, step=0.1)
            sg = st.number_input("sg", value=1.0, step=0.05)
            dg = st.number_input("dg", value=1.0, step=0.05)
        personalizados = dict(Nc=Nc, Nq=Nq, Ng=Ng, sc=sc, sq=sq, sg=sg, dc=dc, dq=dq, dg=dg)

st.divider()

# ------------------------- Botonera -------------------------
bc1, bc2, bc3 = st.columns([1.0, 1.0, 1.0])
with bc1:
    run_grid = st.button("🔎 Analizar y optimizar", use_container_width=True)
with bc2:
    run_ga = st.button("🧬 Optimizar con GA (simple)", use_container_width=True)
with bc3:
    reset = st.button("↺ Restablecer", type="secondary", use_container_width=True)

if reset:
    st.experimental_rerun()

# ------------------------- Ejecución: GRID -------------------------
mejor = None
df_all = pd.DataFrame()

if run_grid:
    with st.spinner("Buscando mejores combinaciones..."):
        mejor, df_all = grid_search(modelo, gamma, c, phi, D, N, FS,
                                    Bmin, Bmax, nB,
                                    Lmin, Lmax, nL,
                                    hmin, hmax, nh,
                                    precio_concreto,
                                    personalizados=personalizados)

    if (df_all is None) or df_all.empty:
        st.warning("No se encontraron soluciones que cumplan la **capacidad admisible**. "
                   "Prueba con **B y L mayores**, **φ o c** más altos, **FS menor** o **carga menor**.")
    else:
        # Encabezado resumen
        st.subheader("Resultados")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("B óptimo (m)", f"{mejor['B']:.2f}")
        with kpi2:
            st.metric("L óptimo (m)", f"{mejor['L']:.2f}")
        with kpi3:
            st.metric("h óptimo (m)", f"{mejor['h']:.2f}")
        with kpi4:
            st.metric("Costo (S/)", f"{mejor['costo']:.0f}")

        # Gráfico q_req vs q_adm
        fig_bar = go.Figure()
        fig_bar.add_bar(name="q_req", x=["q_req"], y=[mejor["q_req"]])
        fig_bar.add_bar(name="q_adm", x=["q_adm"], y=[mejor["q_adm"]])
        fig_bar.update_layout(barmode="group", title="q_req vs q_adm (óptimo)", yaxis_title="kPa")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Dispersión B vs L color costo
        st.subheader("Candidatos válidos (color = Costo; tamaño = h)")
        fig_sc = go.Figure(
            data=go.Scatter(
                x=df_all["B"], y=df_all["L"], mode="markers",
                marker=dict(
                    size=8 + 20 * (df_all["h"] - df_all["h"].min()) / (df_all["h"].max() - df_all["h"].min() + 1e-9),
                    color=df_all["costo"], colorscale="Tealgrn", showscale=True, colorbar=dict(title="Costo (S/)")
                ),
                text=[f"h={h:.2f} m<br>Costo={c:.0f}" for h, c in zip(df_all["h"], df_all["costo"])]
            )
        )
        fig_sc.update_layout(xaxis_title="B (m)", yaxis_title="L (m)")
        st.plotly_chart(fig_sc, use_container_width=True)

        with st.expander("Tabla de candidatos"):
            st.dataframe(df_all.sort_values("costo").reset_index(drop=True), use_container_width=True)

# ------------------------- Ejecución: GA (muy simple) -------------------------
if run_ga:
    # Pequeño GA aleatorio (solo para mostrar flujo; usa la misma verificación)
    rng = np.random.default_rng(42)
    pobl = 120
    generaciones = 30

    def random_ind():
        return dict(
            B=float(rng.uniform(Bmin, Bmax)),
            L=float(rng.uniform(Lmin, Lmax)),
            h=float(rng.uniform(hmin, hmax))
        )

    def fitness(ind):
        area = ind["B"] * ind["L"]
        q_req = N / area
        q_adm = q_admisible(modelo, c, phi, gamma, ind["B"], ind["L"], D, FS, personalizados=personalizados)
        ok = q_adm >= q_req
        costo = costo_concreto(ind["B"], ind["L"], ind["h"], precio_concreto)
        # Penaliza si no cumple
        return costo if ok else 1e12

    poblacion = [random_ind() for _ in range(pobl)]
    for _ in range(generaciones):
        scores = np.array([fitness(ind) for ind in poblacion])
        # selección (los mejores 40%)
        idx = np.argsort(scores)[: int(0.4 * pobl)]
        elite = [poblacion[i] for i in idx]
        # reproducción + mutación
        hijos = []
        while len(elite) + len(hijos) < pobl:
            a, b = rng.choice(elite, 2, replace=True)
            child = dict(
                B=max(Bmin, min(Bmax, (a["B"] + b["B"]) / 2 + rng.normal(0, 0.05))),
                L=max(Lmin, min(Lmax, (a["L"] + b["L"]) / 2 + rng.normal(0, 0.05))),
                h=max(hmin, min(hmax, (a["h"] + b["h"]) / 2 + rng.normal(0, 0.02))),
            )
            hijos.append(child)
        poblacion = elite + hijos

    # Mejor individuo
    scores = np.array([fitness(ind) for ind in poblacion])
    best = poblacion[int(np.argmin(scores))]
    area = best["B"] * best["L"]
    best_qreq = N / area
    best_qadm = q_admisible(modelo, c, phi, gamma, best["B"], best["L"], D, FS, personalizados=personalizados)
    best_cost = costo_concreto(best["B"], best["L"], best["h"], precio_concreto)

    st.subheader("Resultado GA (simple)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("B (m)", f"{best['B']:.2f}")
    k2.metric("L (m)", f"{best['L']:.2f}")
    k3.metric("h (m)", f"{best['h']:.2f}")
    k4.metric("Costo (S/)", f"{best_cost:.0f}")

    fig_bar2 = go.Figure()
    fig_bar2.add_bar(name="q_req", x=["q_req"], y=[best_qreq])
    fig_bar2.add_bar(name="q_adm", x=["q_adm"], y=[best_qadm])
    fig_bar2.update_layout(barmode="group", title="q_req vs q_adm (GA)", yaxis_title="kPa")
    st.plotly_chart(fig_bar2, use_container_width=True)

