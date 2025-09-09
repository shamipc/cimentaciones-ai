import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- CONFIG BÁSICA ----------------------
st.set_page_config(
    page_title="Optimización de Cimentaciones",
    page_icon="📐",
    layout="wide"
)

# ---------------------- ESTILOS (CSS) ----------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    .app-title {
        font-weight: 800;
        font-size: 2.1rem;
        letter-spacing: 0.3px;
        margin-bottom: 0.25rem;
    }
    .app-subtitle{
        color: #94a3b8;
        margin-top: -6px;
        margin-bottom: 1.3rem;
        font-size: 0.95rem;
    }
    .card {
        background: #0b1220;
        border: 1px solid rgba(148,163,184,0.15);
        padding: 1.1rem 1.1rem 0.9rem 1.1rem;
        border-radius: 14px;
        box-shadow: 0 8px 30px rgba(2,12,27,0.35);
    }
    .good-badge{
        display:inline-block;
        background: linear-gradient(90deg,#22c55e,#10b981);
        color: white;
        font-weight: 600;
        padding: 6px 10px;
        border-radius: 30px;
        font-size: 0.85rem;
        letter-spacing: .2px;
    }
    .warn-badge{
        display:inline-block;
        background: linear-gradient(90deg,#f59e0b,#ef4444);
        color: white;
        font-weight: 600;
        padding: 6px 10px;
        border-radius: 30px;
        font-size: 0.85rem;
        letter-spacing: .2px;
    }
    .metric-grid{
        display: grid;
        grid-template-columns: repeat(4, minmax(0,1fr));
        gap: 10px;
        margin-top: .75rem;
        margin-bottom: .6rem;
    }
    .mini{font-size: .85rem; color:#94a3b8; margin-bottom: .4rem}
    .vsep{height:10px}
</style>
""", unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("Parámetros de entrada")

modelo = st.sidebar.selectbox(
    "Modelo de capacidad",
    ["Terzaghi (recomendado)","Meyerhof (simple)","Vesic (simple)"],
    index=0
)

gamma = st.sidebar.number_input("Peso unitario γ (kN/m³)", 10.0, 25.0, 18.0, step=0.1)
c = st.sidebar.number_input("Cohesión c (kPa)", 0.0, 200.0, 20.0, step=0.5)
phi = st.sidebar.number_input("Ángulo de fricción φ (°)", 0.0, 45.0, 35.0, step=0.5)
D = st.sidebar.number_input("Profundidad D (m)", 0.5, 3.0, 1.5, step=0.1)
N = st.sidebar.number_input("Carga N (kN)", 100.0, 6000.0, 1000.0, step=10.0)
FS = st.sidebar.number_input("Factor de seguridad", 1.5, 5.0, 2.5, step=0.1)

st.sidebar.subheader("Costos")
costo_concreto = st.sidebar.number_input("Concreto (S/ por m³)", 300, 1500, 650, step=10)
costo_acero = st.sidebar.number_input("Acero (S/ por kg)", 3.0, 10.0, 5.5, step=0.1)

st.sidebar.subheader("Rangos de diseño")
B_min, B_max = st.sidebar.slider("Base B (m)", 1.2, 3.8, (1.2, 3.8), step=0.1)
L_min, L_max = st.sidebar.slider("Largo L (m)", 1.2, 3.8, (1.2, 3.8), step=0.1)
h_min, h_max = st.sidebar.slider("Altura h (m)", 0.6, 1.2, (0.6, 1.2), step=0.05)
paso = 0.1

st.sidebar.subheader("Adjuntos")
up = st.sidebar.file_uploader("Sube una imagen (perfil del suelo, croquis)", type=["png","jpg","jpeg"])

run = st.sidebar.button("🔍 Analizar y optimizar", use_container_width=True)

# ---------------------- FUNCIONES DE CÁLCULO ----------------------
def bearing_capacity_factors(phi_deg: float):
    """Factores de capacidad (aprox. clásicas)."""
    phi = np.radians(phi_deg)
    if phi_deg == 0:
        Nq = 1.0
        Nc = 5.7
        Ny = 0.0
        return Nc, Nq, Ny

    Nq = np.exp(np.pi*np.tan(phi)) * (np.tan(np.pi/4 + phi/2))**2
    Nc = (Nq - 1.0)/np.tan(phi)
    Ny = 2.0*(Nq + 1.0) * np.tan(phi)
    return Nc, Nq, Ny

def capacidad_portante_qu(modelo, c, gamma, D, B, phi_deg):
    """qu (kPa) sin FS, con fórmulas simples."""
    Nc, Nq, Ny = bearing_capacity_factors(phi_deg)

    # Coef. sencillos por modelo (afinación ligera de Ny)
    if "Terzaghi" in modelo:
        s_c, s_q, s_y = 1.0, 1.0, 1.0
    elif "Meyerhof" in modelo:
        s_c, s_q, s_y = 1.1, 1.05, 0.9
    else:  # Vesic simple
        s_c, s_q, s_y = 1.15, 1.1, 1.0

    qu = c*Nc*s_c + gamma*D*Nq*s_q + 0.5*gamma*B*Ny*s_y  # kPa
    return qu

def costo_total(B,L,h, costo_conc, costo_ac):
    volumen = B*L*h
    peso_concreto = volumen * 2.4 * 1000  # kg (2.4 t/m³)
    costo = volumen*costo_conc + (peso_concreto*0.01)*costo_ac  # acero como 1% del peso
    return costo

# ---------------------- ENCABEZADO ----------------------
st.markdown('<div class="app-title">Optimización de Cimentaciones</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta</div>', unsafe_allow_html=True)

# ---------------------- LÓGICA PRINCIPAL ----------------------
if not run:
    st.info("Configura los parámetros a la izquierda y pulsa **Analizar y optimizar**.", icon="🛠️")
    st.stop()

# Generar combinaciones y evaluar
resultados = []
for B in np.arange(B_min, B_max + 1e-9, paso):
    for L in np.arange(L_min, L_max + 1e-9, paso):
        for h in np.arange(h_min, h_max + 1e-9, paso):
            area = B * L
            qu = capacidad_portante_qu(modelo, c, gamma, D, B, phi)
            qadm = qu / FS
            qreq = N / area
            cumple = qadm > qreq
            costo = costo_total(B, L, h, costo_concreto, costo_acero)
            resultados.append([B, L, h, qadm, qreq, cumple, costo])

df = pd.DataFrame(
    resultados,
    columns=["B (m)","L (m)","h (m)","q_adm (kPa)","q_req (kPa)","Cumple","Costo (S/)"]
)

df_validos = df[df["Cumple"] == True].copy()

colL, colR = st.columns([1.15, 1])

# ---------------------- IZQUIERDA: TARJETAS RESUMEN ----------------------
with colL:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if df_validos.empty:
        st.markdown('<span class="warn-badge">Sin soluciones que cumplan</span>', unsafe_allow_html=True)
        st.write("Prueba con **B y L mayores**, **φ o c más altos**, **FS menor** o una **carga N** más pequeña.")
    else:
        mejor = df_validos.loc[df_validos["Costo (S/)"].idxmin()]

        st.markdown('<span class="good-badge">Diseño óptimo encontrado</span>', unsafe_allow_html=True)

        # Métricas
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("B (m)", f"{mejor['B (m)']:.2f}")
        c2.metric("L (m)", f"{mejor['L (m)']:.2f}")
        c3.metric("h (m)", f"{mejor['h (m)']:.2f}")
        c4.metric("Costo (S/)", f"{mejor['Costo (S/)']:.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Texto JSON elegante
        st.markdown('<div class="mini">Resumen</div>', unsafe_allow_html=True)
        st.json({
            "Modelo": modelo,
            "B (m)": round(float(mejor["B (m)"]),2),
            "L (m)": round(float(mejor["L (m)"]),2),
            "h (m)": round(float(mejor["h (m)"]),2),
            "q_adm (kPa)": round(float(mejor["q_adm (kPa)"]),1),
            "q_req (kPa)": round(float(mejor["q_req (kPa)"]),1),
            "Costo (S/)": round(float(mejor["Costo (S/)"]),2)
        })

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="vsep"></div>', unsafe_allow_html=True)

    # Candidatos (Top 12)
    if not df_validos.empty:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="mini">Mejores candidatos por costo (Top 12)</div>', unsafe_allow_html=True)
        top = df_validos.sort_values("Costo (S/)").head(12).reset_index(drop=True)
        st.dataframe(
            top.style.format({
                "B (m)": "{:.2f}",
                "L (m)": "{:.2f}",
                "h (m)": "{:.2f}",
                "q_adm (kPa)": "{:.1f}",
                "q_req (kPa)": "{:.1f}",
                "Costo (S/)": "{:.0f}"
            }),
            use_container_width=True,
            height=340
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- DERECHA: GRÁFICO + IMAGEN ----------------------
with colR:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="mini">Comparación de presiones (óptimo)</div>', unsafe_allow_html=True)

    if df_validos.empty:
        st.warning("No hay gráfico porque no se encontró un diseño que cumpla.", icon="⚠️")
    else:
        qadm = mejor["q_adm (kPa)"]
        qreq = mejor["q_req (kPa)"]

        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        bars = ax.bar(
            ["q_req", "q_adm"],
            [qreq, qadm],
            width=0.55,
            color=["#1e40af", "#10b981"]
        )
        ax.set_ylabel("kPa")
        ax.set_title("q_req vs q_adm")
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, f"{b.get_height():.1f}",
                    ha='center', va='bottom', fontsize=9)
        ax.grid(axis='y', alpha=0.25)
        st.pyplot(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="vsep"></div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="mini">Imagen adjunta</div>', unsafe_allow_html=True)
    if up is not None:
        st.image(up, caption="Adjunto del usuario", use_container_width=True)
    else:
        st.info("Puedes subir un croquis o perfil estratigráfico desde la barra lateral.", icon="🖼️")
    st.markdown('</div>', unsafe_allow_html=True)

