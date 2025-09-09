import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")
st.title("Optimización de Cimentaciones")
st.sidebar.header("Parámetros de entrada")

# ------------------ ENTRADAS ------------------
modelo = st.sidebar.selectbox("Modelo de capacidad", ["Terzaghi (recomendado)", "Simple"])
gamma = st.sidebar.number_input("Peso unitario γ (kN/m³)", 10.0, 25.0, 18.0)
c = st.sidebar.number_input("Cohesión c (kPa)", 0.0, 200.0, 20.0)
phi = st.sidebar.number_input("Ángulo φ (°)", 20.0, 45.0, 35.0)
D = st.sidebar.number_input("Profundidad D (m)", 0.5, 3.0, 1.5)
N = st.sidebar.number_input("Carga N (kN)", 100.0, 6000.0, 1000.0)
FS = st.sidebar.number_input("Factor de seguridad", 1.5, 5.0, 2.5, step=0.1)

st.sidebar.subheader("Costos")
costo_concreto = st.sidebar.number_input("Concreto (S/ m³)", 300, 1500, 650)
costo_acero = st.sidebar.number_input("Acero (S/ kg)", 3.0, 15.0, 5.5)

st.sidebar.subheader("Rangos de diseño")
B_min, B_max = st.sidebar.slider("Base B (m)", 0.8, 4.0, (1.2, 3.8))
L_min, L_max = st.sidebar.slider("Largo L (m)", 0.8, 4.0, (1.2, 3.8))
h_min, h_max = st.sidebar.slider("Altura h (m)", 0.4, 1.5, (0.6, 1.2))
paso = 0.2

btn = st.sidebar.button("🔍 Analizar y optimizar")
st.subheader("Resultados de diseño")

# ------------------ FUNCIONES ------------------
def frange(a, b, s):
    n = int(np.floor((b - a) / s + 0.5)) + 1
    return [round(a + i * s, 10) for i in range(n)]

def capacidad_simple(c, phi_deg, gamma, D, B, L):
    # (modelo académico simple usado antes)
    return (1.3 * c * (1 + 0.2 * (B / max(L, 1e-6)))) + (gamma * D * np.tan(np.radians(phi_deg)))

def N_factors(phi_rad):
    Nq = np.exp(np.pi * np.tan(phi_rad)) * (np.tan(np.pi/4 + phi_rad/2) ** 2)
    Nc = (Nq - 1.0) / np.tan(phi_rad) if phi_rad > 1e-6 else 5.7  # ~φ→0
    Ng = 2.0 * (Nq + 1.0) * np.tan(phi_rad)  # aproximación usual
    return Nc, Nq, Ng

def capacidad_terzaghi(c, phi_deg, gamma, D, B, L):
    # Capacidad última Terzaghi rectangular (factores de forma simples)
    phi_rad = np.radians(phi_deg)
    Nc, Nq, Ng = N_factors(phi_rad)
    q = gamma * D  # sobrecarga
    r = min(B / max(L, 1e-6), 1.0)  # B<=L asumido

    sc = 1.0 + 0.2 * r
    sq = 1.0 + 0.1 * r
    sg = max(0.6, 1.0 - 0.3 * r)   # limitado a >=0.6

    qu = c * Nc * sc + q * Nq * sq + 0.5 * gamma * B * Ng * sg
    return qu  # kPa

# ------------------ CÁLCULOS ------------------
if not btn:
    st.info("Configura los parámetros y pulsa **Analizar y optimizar**.")
else:
    resultados = []
    for B in frange(B_min, B_max, paso):
        for L in frange(L_min, L_max, paso):
            for h in frange(h_min, h_max, paso):
                area = B * L  # m²

                if modelo.startswith("Terzaghi"):
                    qu = capacidad_terzaghi(c, phi, gamma, D, B, L)
                else:
                    qu = capacidad_simple(c, phi, gamma, D, B, L)

                qadm = qu / FS                      # kPa
                qreq = N / area                     # kPa (1 kN/m² = 1 kPa)
                cumple = qadm > qreq

                volumen = B * L * h                 # m³
                peso_concreto = volumen * 2400.0    # kg
                costo = volumen * costo_concreto + (peso_concreto * 0.01) * costo_acero

                resultados.append([B, L, h, qadm, qreq, cumple, costo])

    df = pd.DataFrame(resultados, columns=[
        "B (m)", "L (m)", "h (m)", "q adm (kPa)", "q req (kPa)", "Cumple", "Costo (S/)"
    ])

    df_validos = df[df["Cumple"] == True].sort_values("Costo (S/)")

    if df_validos.empty:
        st.warning("No se encontraron diseños que cumplan. "
                   "Prueba con B y L mayores, φ o c más altos, FS menor o N menor. "
                   "También puedes usar el **modelo Terzaghi** (arriba).")
    else:
        mejor = df_validos.iloc[0]
        st.success("✅ Diseño óptimo encontrado")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("B (m)", f"{mejor['B (m)']:.2f}")
            st.metric("L (m)", f"{mejor['L (m)']:.2f}")
            st.metric("h (m)", f"{mejor['h (m)']:.2f}")
        with c2:
            st.metric("q adm (kPa)", f"{mejor['q adm (kPa)']:.1f}")
            st.metric("q req (kPa)", f"{mejor['q req (kPa)']:.1f}")
            st.metric("Costo (S/)", f"{mejor['Costo (S/)']:.2f}")

        st.divider()
        st.caption("🔎 Soluciones que cumplen")
        st.dataframe(df_validos, use_container_width=True)

        # ---------- Gráfico B-L coloreado por costo ----------
        st.subheader("Mapa de costo (puntos válidos)")
        fig, ax = plt.subplots()
        sc = ax.scatter(df_validos["B (m)"], df_validos["L (m)"],
                        c=df_validos["Costo (S/)"], s=90, cmap="viridis")
        ax.set_xlabel("B (m)")
        ax.set_ylabel("L (m)")
        cb = plt.colorbar(sc)
        cb.set_label("Costo (S/)")
        st.pyplot(fig)

        # ---------- Esquema de la zapata óptima ----------
        st.subheader("Esquema de la zapata óptima (planta)")
        fig2, ax2 = plt.subplots()
        ax2.add_patch(plt.Rectangle((0, 0), mejor["B (m)"], mejor["L (m)"],
                                    color="lightblue", alpha=0.7))
        ax2.set_aspect("equal")
        ax2.set_xlim(0, mejor["B (m)"] * 1.4)
        ax2.set_ylim(0, mejor["L (m)"] * 1.4)
        ax2.set_xlabel("B (m)")
        ax2.set_ylabel("L (m)")
        ax2.set_title(f"{mejor['B (m)']:.2f} m × {mejor['L (m)']:.2f} m, h = {mejor['h (m)']:.2f} m")
        st.pyplot(fig2)

        # ---------- Descarga ----------
        st.download_button("⬇️ Descargar CSV", df_validos.to_csv(index=False).encode("utf-8"),
                           file_name="soluciones_validas.csv", mime="text/csv")
