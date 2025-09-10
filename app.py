# app.py
import math
import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------
# Configuración general UI
# ------------------------------
st.set_page_config(
    page_title="Optimización de Cimentaciones",
    layout="wide",
    page_icon=":building_construction:",
)

PALETTE = ["#6C8AE4", "#64C5B1", "#FFD166", "#EF476F", "#118AB2"]

def h2(txt: str):
    st.markdown(f"### {txt}")

def tip(txt: str):
    st.info(txt, icon="🛠️")

def warn(txt: str):
    st.warning(txt, icon="⚠️")

# ------------------------------
# Presets de suelo (rápido)
# ------------------------------
SOIL_PRESETS = {
    "Arcilla blanda (γ=17, c=25, φ=0)": dict(gamma=17.0, c=25.0, phi=0.0),
    "Arena densa (γ=19, c=0, φ=35)": dict(gamma=19.0, c=0.0, phi=35.0),
    "Arcilla media (γ=18, c=35, φ=8)": dict(gamma=18.0, c=35.0, phi=8.0),
    "Relleno granular (γ=18.5, c=0, φ=32)": dict(gamma=18.5, c=0.0, phi=32.0),
}

# ------------------------------
# Factores de capacidad (Nγ, Nq, Nc)
# Fórmulas clásicas en función de φ (rad)
# ------------------------------
def bearing_factors(phi_deg: float) -> Tuple[float, float, float]:
    """Devuelve (Nγ, Nq, Nc) para φ en grados (Hansen/Mehraf)."""
    phi = math.radians(max(phi_deg, 0.0))
    if phi == 0:
        Nq = 1.0
        Nc = 5.7  # Terzaghi para φ=0
        Nγ = 0.0
        return Nγ, Nq, Nc
    Nq = math.e ** (math.pi * math.tan(phi)) * (math.tan(math.radians(45) + phi / 2)) ** 2
    Nc = (Nq - 1) / math.tan(phi)
    Nγ = 2 * (Nq + 1) * math.tan(phi)
    return Nγ, Nq, Nc

# ------------------------------
# Modelos de capacidad admisible (simplificados)
# ------------------------------
def q_admisible(modelo: str, c: float, phi: float, gamma: float, B: float, D: float, FS: float) -> float:
    """
    Calcula capacidad admisible (kPa) con modelos simplificados.
    Los factores de forma/profundidad se dejan básicos para mantener estabilidad.
    """
    Nγ, Nq, Nc = bearing_factors(phi)
    # Factores sencillos de forma y profundidad (opcionalmente puedes detallarlos)
    sγ = 1.0; sq = 1.0; sc = 1.0
    dγ = 1.0; dq = 1.0; dc = 1.0
    # Presión efectiva a la base
    q0 = gamma * D

    if modelo == "Terzaghi (recomendado)":
        # Terzaghi clásica 2D (para zapata corrida equivalente) – simplificación razonable para curso
        qult = c * Nc * sc * dc + q0 * Nq * sq * dq + 0.5 * gamma * B * Nγ * sγ * dγ
    elif modelo == "Meyerhof":
        qult = c * Nc * sc * dc + q0 * Nq * sq * dq + 0.5 * gamma * B * Nγ * sγ * dγ
        qult *= 1.05  # ligera corrección
    elif modelo == "Hansen":
        qult = c * Nc * sc * dc + q0 * Nq * sq * dq + 0.5 * gamma * B * Nγ * sγ * dγ
        qult *= 1.08
    elif modelo == "Vesic":
        qult = c * Nc * sc * dc + q0 * Nq * sq * dq + 0.5 * gamma * B * Nγ * sγ * dγ
        qult *= 1.12
    else:
        qult = c * Nc + q0 * Nq + 0.5 * gamma * B * Nγ

    qadm = qult / max(FS, 1e-6)
    return float(qadm)

# ------------------------------
# Costo simple (concreto + acero)
# ------------------------------
def costo(B: float, L: float, h: float, Sm3: float, Skg: float) -> float:
    vol = B * L * h  # m3
    # acero muy básico – 60 kg/m3 como placeholder (puedes cambiar a una entrada adicional)
    kg_acero = 60.0 * vol
    return vol * Sm3 + kg_acero * Skg

# ------------------------------
# Búsqueda exhaustiva en grilla
# ------------------------------
def grid_solve(
    modelo: str, c: float, phi: float, gamma: float, D: float, N: float, FS: float,
    Sm3: float, Skg: float,
    Brange: Tuple[float, float, int], Lrange: Tuple[float, float, int], hrange: Tuple[float, float, int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Bmin, Bmax, Bn = Brange
    Lmin, Lmax, Ln = Lrange
    hmin, hmax, hn = hrange

    Bs = np.linspace(Bmin, Bmax, Bn)
    Ls = np.linspace(Lmin, Lmax, Ln)
    hs = np.linspace(hmin, hmax, hn)

    rows = []
    for B in Bs:
        for L in Ls:
            for h in hs:
                area = B * L  # m2
                q_req = (N * 1e3) / max(area, 1e-6)  # kPa (kN/m2)
                q_adm = q_admisible(modelo, c, phi, gamma, B, D, FS)
                ok = q_adm >= q_req
                rows.append(
                    dict(B=round(B, 3), L=round(L, 3), h=round(h, 3),
                         q_req=round(q_req, 2), q_adm=round(q_adm, 2),
                         ok=ok, costo=round(costo(B, L, h, Sm3, Skg), 2))
                )

    df = pd.DataFrame(rows)
    df_valid = df[df["ok"]].sort_values("costo", ascending=True).reset_index(drop=True)
    return df, df_valid

# ------------------------------
# "GA" simple (búsqueda aleatoria guiada)
# ------------------------------
def ga_simple(
    modelo: str, c: float, phi: float, gamma: float, D: float, N: float, FS: float,
    Sm3: float, Skg: float,
    Brange: Tuple[float, float], Lrange: Tuple[float, float], hrange: Tuple[float, float],
    iters: int = 1500
) -> Tuple[dict, pd.DataFrame]:
    Bmin, Bmax = Brange
    Lmin, Lmax = Lrange
    hmin, hmax = hrange

    best = None
    traj = []

    for _ in range(iters):
        # muestreo con ligera preferencia por valores medianos
        B = np.random.beta(2, 2) * (Bmax - Bmin) + Bmin
        L = np.random.beta(2, 2) * (Lmax - Lmin) + Lmin
        h = np.random.beta(2, 2) * (hmax - hmin) + hmin

        area = B * L
        q_req = (N * 1e3) / max(area, 1e-6)
        q_adm = q_admisible(modelo, c, phi, gamma, B, D, FS)

        if q_adm >= q_req:
            cost = costo(B, L, h, Sm3, Skg)
            cand = dict(B=B, L=L, h=h, q_req=q_req, q_adm=q_adm, costo=cost)
            if (best is None) or (cost < best["costo"]):
                best = cand
        traj.append(dict(B=B, L=L, h=h, q_req=q_req, q_adm=q_adm, costo=costo(B, L, h, Sm3, Skg)))

    df_traj = pd.DataFrame(traj)
    return best, df_traj

# ------------------------------
# Estado inicial seguro
# ------------------------------
if "init" not in st.session_state:
    st.session_state.update(
        modelo="Terzaghi (recomendado)",
        preset=list(SOIL_PRESETS.keys())[0],
        gamma=17.0, c=25.0, phi=0.0,
        D=1.5, N=1000.0, FS=2.5,
        concreto=650.0, acero=5.5,
        Bmin=1.2, Bmax=3.2, Bn=30,
        Lmin=1.6, Lmax=4.2, Ln=30,
        hmin=0.5, hmax=1.1, hn=10,
        init=True
    )

# ------------------------------
# Cabecera
# ------------------------------
st.title("Optimización de Cimentaciones")
st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta")

st.info(
    "Ingresa **los datos** en la parte inferior y pulsa **Analizar y optimizar** "
    "u **Optimizar con GA**.", icon="ℹ️"
)

# ------------------------------
# Parámetros de entrada (grilla 3 columnas)
# ------------------------------
h2("Parámetros de entrada")

cols = st.columns(3)
with cols[0]:
    modelo = st.selectbox("Modelo de capacidad",
                          ["Terzaghi (recomendado)", "Meyerhof", "Hansen", "Vesic"],
                          index=["Terzaghi (recomendado)", "Meyerhof", "Hansen", "Vesic"].index(st.session_state.modelo),
                          key="modelo")

    preset = st.selectbox("Preset de suelo (rápido)", list(SOIL_PRESETS.keys()), index=list(SOIL_PRESETS.keys()).index(st.session_state.preset), key="preset")
    if st.session_state.preset:
        st.session_state.gamma = SOIL_PRESETS[preset]["gamma"]
        st.session_state.c = SOIL_PRESETS[preset]["c"]
        st.session_state.phi = SOIL_PRESETS[preset]["phi"]

with cols[1]:
    gamma = st.number_input("Peso unitario γ (kN/m³)", value=float(st.session_state.gamma), step=0.1, min_value=10.0, max_value=25.0, key="gamma")
    c = st.number_input("Cohesión c (kPa)", value=float(st.session_state.get("c", 25.0)), step=1.0, min_value=0.0, max_value=400.0, key="c")
    phi = st.number_input("Ángulo de fricción φ (°)", value=float(st.session_state.get("phi", 0.0)), step=1.0, min_value=0.0, max_value=45.0, key="phi")

with cols[2]:
    D = st.number_input("Profundidad D (m)", value=float(st.session_state.D), step=0.1, min_value=0.0, max_value=10.0, key="D")
    N = st.number_input("Carga N (kN)", value=float(st.session_state.N), step=10.0, min_value=1.0, max_value=1e5, key="N")
    FS = st.number_input("Factor de seguridad", value=float(st.session_state.FS), step=0.1, min_value=1.5, max_value=4.0, key="FS")

# Costos
h2("Costos")
colsC = st.columns(2)
with colsC[0]:
    concreto_Sm3 = st.number_input("Concreto (S/ por m³)", value=float(st.session_state.concreto), step=10.0, min_value=0.0, key="concreto")
with colsC[1]:
    acero_Skg = st.number_input("Acero (S/ por kg)", value=float(st.session_state.acero), step=0.1, min_value=0.0, key="acero")

# Rangos
h2("Rangos de diseño (B, L, h)")
colsR = st.columns(3)
with colsR[0]:
    st.write("Base B (m)")
    Bmin, Bmax = st.slider(" ", min_value=0.8, max_value=5.0, value=(float(st.session_state.Bmin), float(st.session_state.Bmax)), step=0.1, label_visibility="collapsed")
    Bn = st.number_input("Resolución B", value=int(st.session_state.Bn), step=1, min_value=5, max_value=60, key="Bn")
with colsR[1]:
    st.write("Largo L (m)")
    Lmin, Lmax = st.slider("  ", min_value=0.8, max_value=6.0, value=(float(st.session_state.Lmin), float(st.session_state.Lmax)), step=0.1, label_visibility="collapsed")
    Ln = st.number_input("Resolución L", value=int(st.session_state.Ln), step=1, min_value=5, max_value=60, key="Ln")
with colsR[2]:
    st.write("Altura h (m)")
    hmin, hmax = st.slider("   ", min_value=0.30, max_value=2.0, value=(float(st.session_state.hmin), float(st.session_state.hmax)), step=0.05, label_visibility="collapsed")
    hn = st.number_input("Resolución h", value=int(st.session_state.hn), step=1, min_value=3, max_value=40, key="hn")

st.divider()

# Botonera (en una fila)
bcols = st.columns([1, 1, 1])
run_grid = bcols[0].button("🔎 Analizar y optimizar", use_container_width=True)
run_ga   = bcols[1].button("🧬 Optimizar con GA (simple)", use_container_width=True)
reset    = bcols[2].button("↺ Restablecer", use_container_width=True)

if reset:
    for k in ["modelo","preset","gamma","c","phi","D","N","FS","concreto","acero",
              "Bmin","Bmax","Bn","Lmin","Lmax","Ln","hmin","hmax","hn"]:
        if k in st.session_state: del st.session_state[k]
    st.rerun()

# ------------------------------
# Lógica de cálculo / Resultados
# ------------------------------
def render_results(df_all: pd.DataFrame, df_valid: pd.DataFrame, title: str):
    st.subheader(title)

    if df_valid.empty:
        warn(
            "No se encontraron soluciones que cumplan la **capacidad admisible**. "
            "Prueba con **B y L mayores**, **φ o c más altos**, **FS menor** o **carga menor**."
        )
        if st.button("✨ Auto-ajustar y reintentar"):
            # Ampliamos rangos y re-ejecutamos
            st.session_state.Bmax = float(min(5.0, Bmax + 0.6))
            st.session_state.Lmax = float(min(6.0, Lmax + 0.8))
            st.session_state.hmax = float(min(2.0, hmax + 0.15))
            st.session_state.Bn = int(min(60, Bn + 10))
            st.session_state.Ln = int(min(60, Ln + 10))
            st.session_state.hn = int(min(40, hn + 4))
            st.rerun()
        return

    best = df_valid.iloc[0].to_dict()

    # Tarjetas del óptimo
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("B (m)", f"{best['B']:.2f}")
    c2.metric("L (m)", f"{best['L']:.2f}")
    c3.metric("h (m)", f"{best['h']:.2f}")
    c4.metric("Costo (S/)", f"{best['costo']:.0f}")

    # Gráfico barras q_req vs q_adm (óptimo)
    fig_bar = go.Figure()
    fig_bar.add_bar(name="q_req", x=["q_req"], y=[best["q_req"]], marker_color=PALETTE[0])
    fig_bar.add_bar(name="q_adm", x=["q_adm"], y=[best["q_adm"]], marker_color=PALETTE[1])
    fig_bar.update_layout(
        barmode="group", yaxis_title="kPa",
        title="q_req vs q_adm (óptimo)",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", x=0.5, xanchor="center")
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Dispersión B vs L (color = costo, tamaño = h)
    fig_sc = px.scatter(
        df_valid, x="B", y="L", color="costo", size="h",
        color_continuous_scale="Tealgrn", title="Candidatos válidos (color=costo, tamaño=h)"
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # Top-10
    st.write("**Top 10 por costo**")
    st.dataframe(df_valid.head(10), use_container_width=True)

# Ejecutar búsquedas
if run_grid or run_ga:
    # Entradas reunidas
    Br = (Bmin, Bmax, int(Bn))
    Lr = (Lmin, Lmax, int(Ln))
    hr = (hmin, hmax, int(hn))
    Br2 = (Bmin, Bmax)
    Lr2 = (Lmin, Lmax)
    hr2 = (hmin, hmax)

    if run_grid:
        df_all, df_ok = grid_solve(
            modelo, c, phi, gamma, D, N, FS, concreto_Sm3, acero_Skg,
            Br, Lr, hr
        )
        render_results(df_all, df_ok, "Resultado (búsqueda en grilla)")

    if run_ga:
        best, traj = ga_simple(
            modelo, c, phi, gamma, D, N, FS, concreto_Sm3, acero_Skg,
            Br2, Lr2, hr2, iters=1600
        )
        if best is None:
            warn("El GA simple no encontró un candidato que cumpla. Prueba ampliar los rangos o disminuir FS.")
        else:
            dfx = pd.DataFrame([best])
            st.subheader("Resultado GA (simple)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("B (m)", f"{best['B']:.2f}")
            c2.metric("L (m)", f"{best['L']:.2f}")
            c3.metric("h (m)", f"{best['h']:.2f}")
            c4.metric("Costo (S/)", f"{best['costo']:.0f}")

            fig_bar = go.Figure()
            fig_bar.add_bar(name="q_req", x=["q_req"], y=[best["q_req"]], marker_color=PALETTE[0])
            fig_bar.add_bar(name="q_adm", x=["q_adm"], y=[best["q_adm"]], marker_color=PALETTE[1])
            fig_bar.update_layout(barmode="group", yaxis_title="kPa",
                                  title="q_req vs q_adm (GA)",
                                  margin=dict(l=10, r=10, t=40, b=10),
                                  legend=dict(orientation="h", x=0.5, xanchor="center"))
            st.plotly_chart(fig_bar, use_container_width=True)

            # trayectorias (puntos factibles)
            traj_ok = traj[(traj["q_adm"] >= traj["q_req"])]
            if not traj_ok.empty:
                fig_sc = px.scatter(
                    traj_ok, x="B", y="L", color="costo", size="h",
                    color_continuous_scale="Tealgrn", title="Candidatos válidos muestreados por GA"
                )
                st.plotly_chart(fig_sc, use_container_width=True)
            else:
                tip("El GA no generó puntos factibles; considera ampliar rangos o reducir FS.")

