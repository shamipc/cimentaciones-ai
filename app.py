import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")

# Toquecito de estilo
st.markdown("""
<style>
    .stApp {background: radial-gradient(1200px 600px at 10% 10%, #151a22 0%, #0f1217 40%, #0b0d11 100%);}
    .stMarkdown, .stText, .stSelectbox, .stNumberInput, .stSlider, .stButton, .stDataFrame, .stAlert, .stTabs {color: #e7eef7 !important;}
    h1, h2, h3, h4 { color: #e7eef7 !important; }
    .css-1cpxqw2, .css-12ttj6m, .st-emotion-cache-1wivap2 { color: #e7eef7 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Optimización de Cimentaciones")

# ------------------ FÓRMULAS BÁSICAS ------------------
def bearing_capacity_factors(phi_deg):
    """Devuelve Nq, Nc, Nγ (clásicos) a partir de phi en grados."""
    phi = np.radians(phi_deg)
    if phi_deg == 0:
        Nq = 1.0
        Nc = 5.7
        Nγ = 0.0
    else:
        Nq = np.exp(np.pi * np.tan(phi)) * (np.tan(np.pi/4 + phi/2) ** 2)
        Nc = (Nq - 1.0) / np.tan(phi)
        Nγ = 2 * (Nq + 1) * np.tan(phi)
    return Nq, Nc, Nγ

def qu_terzaghi(c, phi_deg, gamma, D, B):
    """Capacidad última Terzaghi (cimentación corrida, simplificada)."""
    Nq, Nc, Nγ = bearing_capacity_factors(phi_deg)
    return c * Nc + gamma * D * Nq + 0.5 * gamma * B * Nγ

def qu_meyerhof(c, phi_deg, gamma, D, B):
    """
    Meyerhof con factores de profundidad simplificados (muy compactado).
    Aquí usamos la misma base que Terzaghi para mantenerlo simple,
    con un ligero incremento por profundidad.
    """
    Nq, Nc, Nγ = bearing_capacity_factors(phi_deg)
    sD = 1.0 + 0.2 * (D / (B + 1e-9))  # factor de profundidad muy simple
    return (c * Nc + gamma * D * Nq + 0.5 * gamma * B * Nγ) * sD

def qu_hansen(c, phi_deg, gamma, D, B):
    """
    Hansen (simplificado). Para no inflar con todos los factores (s, d, i),
    damos un incremento moderado respecto a Terzaghi.
    """
    base = qu_terzaghi(c, phi_deg, gamma, D, B)
    return base * 1.10  # +10% aprox, modo demo

def capacidad_ultima(modelo, c, phi, gamma, D, B):
    if modelo.startswith("Terzaghi"):
        return qu_terzaghi(c, phi, gamma, D, B)
    elif modelo.startswith("Meyerhof"):
        return qu_meyerhof(c, phi, gamma, D, B)
    else:
        return qu_hansen(c, phi, gamma, D, B)

# ------------------ SIDEBAR ------------------
st.sidebar.header("Parámetros de entrada")

modelo = st.sidebar.selectbox(
    "Modelo de capacidad",
    ["Terzaghi (recomendada)", "Meyerhof", "Hansen (simplificado)"]
)

gamma = st.sidebar.number_input("Peso unitario del suelo γ (kN/m³)", 10.0, 25.0, 18.0, 0.1)
c = st.sidebar.number_input("Cohesión c (kPa)", 0.0, 100.0, 20.0, 0.5)
phi = st.sidebar.number_input("Ángulo de fricción φ (°)", 0.0, 45.0, 35.0, 0.5)
D = st.sidebar.number_input("Profundidad de cimentación D (m)", 0.5, 5.0, 1.5, 0.1)

st.sidebar.subheader("Cargas")
N = st.sidebar.number_input("Carga aplicada N (kN)", 100.0, 10000.0, 1000.0, 10.0)

st.sidebar.subheader("Seguridad y Costos")
FS = st.sidebar.number_input("Factor de seguridad mínimo", 1.5, 5.0, 2.5, 0.1)
costo_concreto = st.sidebar.number_input("Concreto (S/ por m³)", 300, 1500, 650, 10)
costo_acero = st.sidebar.number_input("Acero (S/ por kg)", 3.0, 12.0, 5.5, 0.1)

st.sidebar.subheader("Rangos de diseño")
B_min, B_max = st.sidebar.slider("Base B (m)", 0.8, 4.0, (1.2, 3.8), 0.01)
L_min, L_max = st.sidebar.slider("Largo L (m)", 0.8, 4.0, (1.2, 3.8), 0.01)
h_min, h_max = st.sidebar.slider("Altura h (m)", 0.4, 1.5, (0.6, 1.2), 0.01)
paso = 0.1

st.sidebar.subheader("Adjuntos")
img_file = st.sidebar.file_uploader("Sube una imagen (perfil del suelo, croquis)", type=["png", "jpg", "jpeg"])

# ------------------ LAYOUT PRINCIPAL ------------------
tab_res, tab_cand, tab_adj = st.tabs(["Resultados", "Candidatos", "Adjuntos / Ayuda"])

with tab_adj:
    st.markdown("### Imagen adjunta")
    if img_file is not None:
        try:
            img = Image.open(img_file)
            st.image(img, caption="Imagen cargada por el usuario", use_column_width=True)
        except Exception as e:
            st.warning(f"No se pudo abrir la imagen: {e}")
    else:
        st.info("Puedes subir una imagen (PNG/JPG) desde la barra lateral.")

    st.markdown("""
**Ayuda rápida**
- Ajusta los parámetros de suelo y la carga.
- Define los **rangos de B, L y h** (m).
- Pulsa **Analizar y optimizar**; la app explorará combinaciones y te mostrará el **diseño óptimo** por costo que cumpla con la capacidad admisible.
- Cambia de modelo entre **Terzaghi, Meyerhof y Hansen (simplificado)** para comparar.
""")

# ------------------ BOTÓN ------------------
run = st.sidebar.button("🔎 Analizar y optimizar")

# ------------------ CÁLCULOS ------------------
if run:
    # Validaciones básicas
    if B_min <= 0 or L_min <= 0 or h_min <= 0:
        st.error("Los rangos deben ser positivos.")
        st.stop()

    resultados = []
    Bs = np.arange(B_min, B_max + 1e-9, paso)
    Ls = np.arange(L_min, L_max + 1e-9, paso)
    hs = np.arange(h_min, h_max + 1e-9, paso)

    for B in Bs:
        for L in Ls:
            for h in hs:
                area = B * L  # m²
                qu = capacidad_ultima(modelo, c, phi, gamma, D, B)  # kPa ~ kN/m²
                qadm = qu / FS
                qreq = (N * 1.0) / area  # kN/m²
                cumple = qadm >= qreq

                volumen = B * L * h  # m³
                peso_concreto = volumen * 2.4 * 1000  # kg (γ_conc ~ 24 kN/m³ ~ 2.4 t/m³)
                costo = volumen * costo_concreto + (peso_concreto * 0.01) * costo_acero  # factor 0.01 para ponderar acero

                resultados.append([B, L, h, qu, qadm, qreq, cumple, costo])

    df = pd.DataFrame(
        resultados,
        columns=["B (m)", "L (m)", "h (m)", "q_ult (kPa)", "q_adm (kPa)", "q_req (kPa)", "Cumple", "Costo (S/)"]
    )

    df_validos = df[df["Cumple"] == True].copy().reset_index(drop=True)

    with tab_res:
        st.subheader("Resultados de diseño")
        if not df_validos.empty:
            mejor = df_validos.loc[df_validos["Costo (S/)"].idxmin()]
            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.success("✅ **Diseño óptimo encontrado**")
                st.json({
                    "Modelo": modelo,
                    "B (m)": round(mejor["B (m)"], 2),
                    "L (m)": round(mejor["L (m)"], 2),
                    "h (m)": round(mejor["h (m)"], 2),
                    "q_adm (kPa)": round(mejor["q_adm (kPa)"], 1),
                    "q_req (kPa)": round(mejor["q_req (kPa)"], 1),
                    "Costo (S/)": round(mejor["Costo (S/)"], 2)
                })
            with col2:
                st.markdown("**Comparación de presiones (óptimo):**")
                st.progress(min(1.0, float(mejor["q_req (kPa)"]/max(mejor["q_adm (kPa)"], 1e-6))))
                st.caption(f"q_req / q_adm = {mejor['q_req (kPa)']:.1f} / {mejor['q_adm (kPa)']:.1f}")

        else:
            st.warning(
                "No se encontraron diseños que cumplan con la capacidad admisible. "
                "Prueba con B y L mayores, φ o c más altos, FS menor o una carga N menor."
            )

    with tab_cand:
        st.subheader("Candidatos que cumplen")
        if not df_validos.empty:
            st.dataframe(df_validos.sort_values("Costo (S/)").head(100), use_container_width=True)
            csv = df_validos.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar resultados (CSV)", csv, file_name="cimentaciones_candidatos.csv", mime="text/csv")
        else:
            st.info("Ajusta los parámetros y vuelve a intentar.")

else:
    st.info("Configura los parámetros y pulsa **Analizar y optimizar**.")

