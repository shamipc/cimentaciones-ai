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

# ---------- Propiedades de Suelos ----------
# Configuración de la página
st.set_page_config(page_title="Parámetros Geotécnicos", layout="wide")

# ---------- Propiedades de Suelos Mejoradas ----------
# Configuración de la página
st.set_page_config(page_title="Parámetros Geotécnicos Completos", layout="wide")

# Inicialización de session_state con parámetros completos y válidos
def inicializar_parametros():
    parametros_base = {
        # === PROPIEDADES BÁSICAS DEL SUELO ===
        'gamma': 18.0,                    # Peso unitario natural (kN/m³)
        'gamma_sat': 20.0,                # Peso unitario saturado (kN/m³)
        'gamma_d': 16.0,                  # Peso unitario seco (kN/m³)
        'c': 10.0,                        # Cohesión drenada (kPa)
        'phi': 30.0,                      # Ángulo de fricción (°)
        'cu': 25.0,                       # Cohesión no drenada (kPa)
        'su': 20.0,                       # Resistencia al corte no drenado (kPa)
        
        # === PROPIEDADES DE DEFORMABILIDAD ===
        'Es': 30000.0,                    # Módulo de elasticidad del suelo (kPa)
        'Eu': 15000.0,                    # Módulo no drenado (kPa)
        'nu': 0.35,                       # Coeficiente de Poisson
        'mv': 0.0001,                     # Coeficiente de compresibilidad (m²/kN)
        'Cv': 5.0,                        # Coeficiente de consolidación (m²/año)
        
        # === PROPIEDADES FÍSICAS E ÍNDICE ===
        'w': 18.0,                        # Contenido de humedad natural (%)
        'w_sat': 22.0,                    # Contenido de humedad en saturación (%)
        'Gs': 2.65,                       # Gravedad específica de sólidos
        'e': 0.65,                        # Relación de vacíos
        'n': 0.39,                        # Porosidad
        'Sr': 75.0,                       # Grado de saturación (%)
        
        # === LÍMITES DE ATTERBERG ===
        'LL': 45.0,                       # Límite líquido (%)
        'LP': 25.0,                       # Límite plástico (%)
        'IP': 20.0,                       # Índice plástico (%)
        'LI': 0.0,                        # Índice de liquidez
        'IC': 1.0,                        # Índice de consistencia
        
        # === ENSAYOS DE CAMPO ===
        'N_SPT': 15,                      # Valor N del ensayo SPT (golpes/pie)
        'qc_CPT': 5000.0,                 # Resistencia de cono CPT (kPa)
        'fs_CPT': 100.0,                  # Resistencia por fricción CPT (kPa)
        'Vs': 200.0,                      # Velocidad de ondas de corte (m/s)
        'Vp': 500.0,                      # Velocidad de ondas primarias (m/s)
        
        # === PROPIEDADES HIDRÁULICAS ===
        'k': 1e-5,                        # Coeficiente de permeabilidad (m/s)
        'k_h': 1e-5,                      # Permeabilidad horizontal (m/s)
        'k_v': 1e-5,                      # Permeabilidad vertical (m/s)
        'nivel_freatico': 2.0,            # Profundidad nivel freático (m)
        'presion_poros': 0.0,             # Presión de poros (kPa)
        
        # === COMPRESIBILIDAD Y CONSOLIDACIÓN ===
        'Cc': 0.25,                       # Índice de compresión
        'Cs': 0.05,                       # Índice de expansión
        'Cr': 0.03,                       # Índice de recompresión
        'sigma_p': 100.0,                 # Presión de preconsolidación (kPa)
        'OCR': 1.5,                       # Razón de sobreconsolidación
        't90': 2.0,                       # Tiempo para 90% consolidación (años)
        
        # === CLASIFICACIÓN SUCS ===
        'sucs_grupo': 'SM',               # Grupo SUCS
        'sucs_subgrupo': 'Sandy SILT',    # Subgrupo SUCS
        'aashto_grupo': 'A-2-4',          # Grupo AASHTO
        'aashto_gi': 4,                   # Índice de grupo AASHTO
        
        # === CAPACIDAD PORTANTE Y ASENTAMIENTOS ===
        'q_adm': 150.0,                   # Capacidad portante admisible (kPa)
        'q_ult': 375.0,                   # Capacidad portante última (kPa)
        'asent_elastico': 0.015,          # Asentamiento elástico (m)
        'asent_consolidacion': 0.035,     # Asentamiento por consolidación (m)
        'asent_total': 0.050,             # Asentamiento total (m)
        
        # === PARÁMETROS SÍSMICOS ===
        'Gmax': 50000.0,                  # Módulo de corte máximo (kPa)
        'D50': 0.2,                       # Tamaño efectivo de partícula (mm)
        'FC': 15.0,                       # Contenido de finos (%)
        'rd': 0.95,                       # Factor de reducción de esfuerzos
        'MSF': 1.0,                       # Factor de escala de magnitud
        
        # === PROPIEDADES QUÍMICAS ===
        'pH': 7.0,                        # pH del suelo
        'sulfatos': 0.1,                  # Contenido de sulfatos (%)
        'materia_organica': 1.5,          # Contenido de materia orgánica (%)
        'carbonatos': 5.0,                # Contenido de carbonatos (%)
        
        # === PARÁMETROS DE COMPACTACIÓN ===
        'w_optima': 16.0,                 # Contenido de humedad óptimo (%)
        'gamma_max': 19.5,                # Peso unitario máximo (kN/m³)
        'RC': 95.0,                       # Grado de compactación (%)
    }
    
    for key, value in parametros_base.items():
        if key not in st.session_state:
            st.session_state[key] = value

inicializar_parametros()

# FUNCIONES DE CLASIFICACIÓN MEJORADAS
def clasificar_suelo_spt(N):
    """Clasificación de densidad/consistencia basada en N-SPT"""
    if N < 2: return "Muy suelto / Muy blando"
    elif N < 4: return "Suelto / Blando"
    elif N < 8: return "Medio suelto / Medio blando"
    elif N < 15: return "Medio / Firme"
    elif N < 30: return "Medio denso / Muy firme"
    elif N < 50: return "Denso / Rígido"
    else: return "Muy denso / Muy rígido"

def clasificar_por_permeabilidad(k):
    """Clasificación según coeficiente de permeabilidad"""
    if k > 1e-1: return "Alta - Gravas limpias"
    elif k > 1e-3: return "Media-Alta - Arenas gruesas"
    elif k > 1e-5: return "Media - Arenas finas, limos"
    elif k > 1e-7: return "Baja - Limos arcillosos"
    elif k > 1e-9: return "Muy baja - Arcillas"
    else: return "Impermeable - Arcillas muy plásticas"

def clasificar_por_compresibilidad(Cc):
    """Clasificación según índice de compresión"""
    if Cc < 0.1: return "Baja compresibilidad"
    elif Cc < 0.3: return "Media compresibilidad"
    elif Cc < 0.5: return "Alta compresibilidad"
    else: return "Muy alta compresibilidad"

def estimar_phi_por_spt(N):
    """Estimar ángulo de fricción a partir de N-SPT (para arenas)"""
    if N < 4: return 28
    elif N < 10: return 30
    elif N < 30: return 32
    elif N < 50: return 35
    else: return 38

def estimar_cu_por_spt(N):
    """Estimar cohesión no drenada a partir de N-SPT (para arcillas)"""
    return N * 5.0  # Aproximación común: cu (kPa) ≈ 5 * N

def clasificar_por_atterberg(LL, IP):
    """Clasificación según carta de plasticidad"""
    if IP < 4: return "No plástico"
    elif IP < 7: return "Baja plasticidad"
    elif IP < 15: return "Media plasticidad"
    elif IP < 30: return "Alta plasticidad"
    else: return "Muy alta plasticidad"

# INTERFAZ PRINCIPAL MEJORADA
st.title("🏗️ PARÁMETROS GEOTÉCNICOS COMPLETOS")
st.markdown("### Sistema integrado de parámetros para análisis geotécnico avanzado")

# PESTAÑAS PRINCIPALES MEJORADAS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧱 PROPIEDADES BÁSICAS", 
    "📊 ENSAYOS Y CLASIFICACIÓN", 
    "💧 HIDRÁULICA Y CONSOLIDACIÓN",
    "⚡ PROPIEDADES AVANZADAS",
    "📋 RESUMEN Y EXPORTACIÓN"
])

with tab1:
    st.header("Propiedades Fundamentales del Suelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔹 Resistencia al Corte")
        st.session_state.c = st.number_input("Cohesión drenada c (kPa)", 
                                           value=st.session_state.c, 
                                           min_value=0.0, max_value=500.0, step=5.0,
                                           help="Cohesión efectiva para condiciones drenadas")
        
        st.session_state.phi = st.number_input("Ángulo de fricción φ (°)", 
                                             value=st.session_state.phi, 
                                             min_value=0.0, max_value=45.0, step=1.0,
                                             help="Ángulo de fricción interna")
        
        st.session_state.cu = st.number_input("Resistencia no drenada c_u (kPa)", 
                                            value=st.session_state.cu, 
                                            min_value=0.0, max_value=300.0, step=5.0,
                                            help="Cohesión no drenada para análisis a corto plazo")
        
        st.session_state.su = st.number_input("Resistencia corte no drenado s_u (kPa)", 
                                            value=st.session_state.su, 
                                            min_value=0.0, max_value=500.0, step=10.0)
        
        st.subheader("🔹 Pesos Unitarios")
        st.session_state.gamma = st.number_input("Peso unitario natural γ (kN/m³)", 
                                               value=st.session_state.gamma, 
                                               min_value=10.0, max_value=25.0, step=0.5,
                                               help="Peso unitario en condiciones naturales")
        
        st.session_state.gamma_sat = st.number_input("Peso unitario saturado γ_sat (kN/m³)", 
                                                   value=st.session_state.gamma_sat, 
                                                   min_value=15.0, max_value=23.0, step=0.5)
        
        st.session_state.gamma_d = st.number_input("Peso unitario seco γ_d (kN/m³)", 
                                                 value=st.session_state.gamma_d, 
                                                 min_value=12.0, max_value=22.0, step=0.5)
    
    with col2:
        st.subheader("🔹 Deformabilidad")
        st.session_state.Es = st.number_input("Módulo elasticidad E_s (kPa)", 
                                            value=st.session_state.Es, 
                                            min_value=1000.0, max_value=200000.0, step=1000.0,
                                            help="Módulo de Young del suelo para condiciones drenadas")
        
        st.session_state.Eu = st.number_input("Módulo no drenado E_u (kPa)", 
                                            value=st.session_state.Eu, 
                                            min_value=500.0, max_value=100000.0, step=1000.0)
        
        st.session_state.nu = st.number_input("Coeficiente de Poisson ν", 
                                            value=st.session_state.nu, 
                                            min_value=0.1, max_value=0.5, step=0.05,
                                            help="Rango típico: 0.2-0.35 para arenas, 0.4-0.5 para arcillas saturadas")
        
        st.session_state.Gmax = st.number_input("Módulo corte máximo G_max (kPa)", 
                                              value=st.session_state.Gmax, 
                                              min_value=1000.0, max_value=300000.0, step=5000.0)
        
        st.subheader("🔹 Propiedades Índice")
        st.session_state.Gs = st.number_input("Gravedad específica G_s", 
                                            value=st.session_state.Gs, 
                                            min_value=2.4, max_value=2.8, step=0.01,
                                            help="Rango típico: 2.65-2.67 para arenas, 2.70-2.75 para arcillas")
        
        st.session_state.e = st.number_input("Relación de vacíos e", 
                                           value=st.session_state.e, 
                                           min_value=0.3, max_value=2.0, step=0.05)
        
        st.session_state.n = st.number_input("Porosidad n", 
                                           value=st.session_state.n, 
                                           min_value=0.2, max_value=0.8, step=0.01)

with tab2:
    st.header("Ensayos de Campo y Clasificación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔹 Ensayos de Penetración")
        st.session_state.N_SPT = st.number_input("Valor N-SPT (golpes/pie)", 
                                               value=st.session_state.N_SPT, 
                                               min_value=0, max_value=100, step=1,
                                               help="Standard Penetration Test - Golpes para 30 cm de penetración")
        
        # Mostrar clasificación automática
        clasif_spt = clasificar_suelo_spt(st.session_state.N_SPT)
        st.metric("Clasificación por N-SPT", clasif_spt)
        
        # Estimaciones automáticas
        if st.checkbox("Estimar parámetros desde N-SPT"):
            phi_estimado = estimar_phi_por_spt(st.session_state.N_SPT)
            cu_estimado = estimar_cu_por_spt(st.session_state.N_SPT)
            st.info(f"φ estimado: {phi_estimado}° | c_u estimado: {cu_estimado} kPa")
        
        st.session_state.qc_CPT = st.number_input("Resistencia de cono q_c (kPa) - CPT", 
                                                value=st.session_state.qc_CPT, 
                                                min_value=0.0, max_value=30000.0, step=500.0)
        
        st.session_state.fs_CPT = st.number_input("Resistencia por fricción f_s (kPa) - CPT", 
                                                value=st.session_state.fs_CPT, 
                                                min_value=0.0, max_value=300.0, step=10.0)
        
        st.session_state.Vs = st.number_input("Velocidad onda de corte V_s (m/s)", 
                                            value=st.session_state.Vs, 
                                            min_value=50.0, max_value=1000.0, step=25.0)
        
        st.session_state.Vp = st.number_input("Velocidad onda primaria V_p (m/s)", 
                                            value=st.session_state.Vp, 
                                            min_value=100.0, max_value=2000.0, step=50.0)
        
    with col2:
        st.subheader("🔹 Límites de Atterberg")
        st.session_state.LL = st.number_input("Límite líquido LL (%)", 
                                            value=st.session_state.LL, 
                                            min_value=0.0, max_value=100.0, step=5.0)
        
        st.session_state.LP = st.number_input("Límite plástico LP (%)", 
                                            value=st.session_state.LP, 
                                            min_value=0.0, max_value=80.0, step=5.0)
        
        st.session_state.IP = st.number_input("Índice plástico IP (%)", 
                                            value=st.session_state.IP, 
                                            min_value=0.0, max_value=60.0, step=5.0)
        
        # Clasificación por Atterberg
        if st.session_state.IP > 0:
            clasif_plasticidad = clasificar_por_atterberg(st.session_state.LL, st.session_state.IP)
            st.metric("Clasificación por plasticidad", clasif_plasticidad)
        
        st.session_state.LI = st.number_input("Índice de liquidez LI", 
                                            value=st.session_state.LI, 
                                            min_value=-1.0, max_value=2.0, step=0.1)
        
        st.session_state.IC = st.number_input("Índice de consistencia IC", 
                                            value=st.session_state.IC, 
                                            min_value=0.0, max_value=2.0, step=0.1)
        
        st.subheader("🔹 Clasificación SUCS/AASHTO")
        sucs_grupos = ['GW', 'GP', 'GM', 'GC', 'SW', 'SP', 'SM', 'SC', 'ML', 'CL', 'OL', 'MH', 'CH', 'OH', 'PT']
        st.session_state.sucs_grupo = st.selectbox("Grupo SUCS", sucs_grupos, 
                                                 index=sucs_grupos.index(st.session_state.sucs_grupo))
        
        aashto_grupos = ['A-1-a', 'A-1-b', 'A-2-4', 'A-2-5', 'A-2-6', 'A-2-7', 'A-3', 'A-4', 'A-5', 'A-6', 'A-7-5', 'A-7-6']
        st.session_state.aashto_grupo = st.selectbox("Grupo AASHTO", aashto_grupos, 
                                                   index=aashto_grupos.index(st.session_state.aashto_grupo))

with tab3:
    st.header("Hidráulica y Consolidación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔹 Propiedades Hidráulicas")
        st.session_state.k = st.number_input("Coeficiente de permeabilidad k (m/s)", 
                                           value=st.session_state.k, 
                                           format="%.2e", min_value=1e-9, max_value=1e-1, step=1e-6)
        
        # Clasificación por permeabilidad
        clasif_k = clasificar_por_permeabilidad(st.session_state.k)
        st.metric("Clasificación permeabilidad", clasif_k)
        
        st.session_state.k_h = st.number_input("Permeabilidad horizontal k_h (m/s)", 
                                             value=st.session_state.k_h, 
                                             format="%.2e", min_value=1e-9, max_value=1e-1, step=1e-6)
        
        st.session_state.k_v = st.number_input("Permeabilidad vertical k_v (m/s)", 
                                             value=st.session_state.k_v, 
                                             format="%.2e", min_value=1e-9, max_value=1e-1, step=1e-6)
        
        st.session_state.nivel_freatico = st.number_input("Nivel freático (m)", 
                                                        value=st.session_state.nivel_freatico, 
                                                        min_value=0.0, max_value=20.0, step=0.5)
        
        st.session_state.presion_poros = st.number_input("Presión de poros u (kPa)", 
                                                       value=st.session_state.presion_poros, 
                                                       min_value=0.0, max_value=500.0, step=10.0)
        
    with col2:
        st.subheader("🔹 Compresibilidad y Consolidación")
        st.session_state.Cc = st.number_input("Índice de compresión C_c", 
                                            value=st.session_state.Cc, 
                                            min_value=0.01, max_value=2.0, step=0.05)
        
        # Clasificación por compresibilidad
        clasif_comp = clasificar_por_compresibilidad(st.session_state.Cc)
        st.metric("Clasificación compresibilidad", clasif_comp)
        
        st.session_state.Cs = st.number_input("Índice de expansión C_s", 
                                            value=st.session_state.Cs, 
                                            min_value=0.001, max_value=0.5, step=0.01)
        
        st.session_state.Cr = st.number_input("Índice de recompresión C_r", 
                                            value=st.session_state.Cr, 
                                            min_value=0.001, max_value=0.3, step=0.01)
        
        st.session_state.sigma_p = st.number_input("Presión preconsolidación σ'_p (kPa)", 
                                                 value=st.session_state.sigma_p, 
                                                 min_value=10.0, max_value=1000.0, step=25.0)
        
        # Calcular OCR automáticamente
        sigma_vo = st.session_state.gamma * 5.0  # Suponiendo 5m de profundidad
        OCR = st.session_state.sigma_p / sigma_vo if sigma_vo > 0 else 1.0
        st.session_state.OCR = st.number_input("OCR (Razón sobreconsolidación)", 
                                             value=OCR, min_value=1.0, max_value=10.0, step=0.1)
        
        st.session_state.Cv = st.number_input("Coeficiente consolidación C_v (m²/año)", 
                                            value=st.session_state.Cv, 
                                            min_value=0.1, max_value=100.0, step=1.0)

with tab4:
    st.header("Propiedades Avanzadas y Especiales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔹 Propiedades Sísmicas")
        st.session_state.rd = st.number_input("Factor reducción esfuerzos r_d", 
                                            value=st.session_state.rd, 
                                            min_value=0.5, max_value=1.2, step=0.05)
        
        st.session_state.MSF = st.number_input("Factor escala magnitud MSF", 
                                            value=st.session_state.MSF, 
                                            min_value=0.5, max_value=2.0, step=0.1)
        
        st.session_state.D50 = st.number_input("Tamaño efectivo D₅₀ (mm)", 
                                            value=st.session_state.D50, 
                                            min_value=0.001, max_value=10.0, step=0.1)
        
        st.session_state.FC = st.number_input("Contenido de finos FC (%)", 
                                           value=st.session_state.FC, 
                                           min_value=0.0, max_value=100.0, step=5.0)
        
        st.subheader("🔹 Propiedades de Compactación")
        st.session_state.w_optima = st.number_input("Humedad óptima w_opt (%)", 
                                                  value=st.session_state.w_optima, 
                                                  min_value=5.0, max_value=30.0, step=1.0)
        
        st.session_state.gamma_max = st.number_input("Peso unitario máximo γ_max (kN/m³)", 
                                                   value=st.session_state.gamma_max, 
                                                   min_value=15.0, max_value=23.0, step=0.5)
        
        st.session_state.RC = st.number_input("Grado compactación RC (%)", 
                                           value=st.session_state.RC, 
                                           min_value=80.0, max_value=105.0, step=1.0)
    
    with col2:
        st.subheader("🔹 Propiedades Químicas")
        st.session_state.pH = st.number_input("pH del suelo", 
                                            value=st.session_state.pH, 
                                            min_value=3.0, max_value=10.0, step=0.1)
        
        st.session_state.sulfatos = st.number_input("Contenido de sulfatos (%)", 
                                                  value=st.session_state.sulfatos, 
                                                  min_value=0.0, max_value=5.0, step=0.1)
        
        st.session_state.materia_organica = st.number_input("Materia orgánica (%)", 
                                                          value=st.session_state.materia_organica, 
                                                          min_value=0.0, max_value=10.0, step=0.5)
        
        st.session_state.carbonatos = st.number_input("Carbonatos (%)", 
                                                    value=st.session_state.carbonatos, 
                                                    min_value=0.0, max_value=20.0, step=1.0)
        
        st.subheader("🔹 Capacidad Portante y Asentamientos")
        st.session_state.q_adm = st.number_input("Capacidad portante admisible q_adm (kPa)", 
                                               value=st.session_state.q_adm, 
                                               min_value=50.0, max_value=1000.0, step=25.0)
        
        st.session_state.q_ult = st.number_input("Capacidad portante última q_ult (kPa)", 
                                               value=st.session_state.q_ult, 
                                               min_value=100.0, max_value=3000.0, step=50.0)
        
        st.session_state.asent_total = st.number_input("Asentamiento total esperado (m)", 
                                                     value=st.session_state.asent_total, 
                                                     min_value=0.001, max_value=0.200, step=0.005)

with tab5:
    st.header("Resumen y Exportación de Parámetros")
    
    # Crear DataFrame de resumen completo
    categorias = {
        'RESISTENCIA': ['c', 'phi', 'cu', 'su'],
        'DEFORMABILIDAD': ['Es', 'Eu', 'nu', 'Gmax'],
        'PESOS UNITARIOS': ['gamma', 'gamma_sat', 'gamma_d'],
        'PROPIEDADES ÍNDICE': ['Gs', 'e', 'n', 'w', 'Sr'],
        'ATTERBERG': ['LL', 'LP', 'IP', 'LI', 'IC'],
        'ENSAYOS CAMPO': ['N_SPT', 'qc_CPT', 'fs_CPT', 'Vs', 'Vp'],
        'HIDRÁULICA': ['k', 'k_h', 'k_v', 'nivel_freatico'],
        'CONSOLIDACIÓN': ['Cc', 'Cs', 'Cr', 'sigma_p', 'OCR', 'Cv'],
        'CLASIFICACIÓN': ['sucs_grupo', 'aashto_grupo'],
        'CAPACIDAD PORTANTE': ['q_adm', 'q_ult'],
        'SÍSMICA': ['D50', 'FC', 'rd', 'MSF'],
        'QUÍMICAS': ['pH', 'sulfatos', 'materia_organica', 'carbonatos']
    }
    
    # Mostrar resumen por categorías
    for categoria, parametros in categorias.items():
        with st.expander(f"📁 {categoria}"):
            datos_categoria = []
            for param in parametros:
                if param in st.session_state:
                    valor = st.session_state[param]
                    unidad = obtener_unidad(param)
                    datos_categoria.append([param, valor, unidad])
            
            if datos_categoria:
                df_cat = pd.DataFrame(datos_categoria, columns=['Parámetro', 'Valor', 'Unidad'])
                st.dataframe(df_cat, use_container_width=True, hide_index=True)
    
    # Botones de exportación
    st.subheader("📤 Exportar Datos")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("💾 Guardar Parámetros Actuales", use_container_width=True):
            # Crear DataFrame completo
            todos_parametros = {key: st.session_state[key] for key in st.session_state.keys() 
                               if not key.startswith('_') and not callable(st.session_state[key])}
            df_completo = pd.DataFrame.from_dict(todos_parametros, orient='index', columns=['Valor'])
            df_completo['Unidad'] = df_completo.index.map(obtener_unidad)
            df_completo.index.name = 'Parámetro'
            
            st.success("Parámetros guardados en sesión")

def obtener_unidad(parametro):
    """Función auxiliar para obtener unidades de cada parámetro"""
    unidades = {
        'gamma': 'kN/m³', 'gamma_sat': 'kN/m³', 'gamma_d': 'kN/m³', 'gamma_max': 'kN/m³',
        'c': 'kPa', 'phi': '°', 'cu': 'kPa', 'su': 'kPa',
        'Es': 'kPa', 'Eu': 'kPa', 'Gmax': 'kPa', 'q_adm': 'kPa', 'q_ult': 'kPa',
        'sigma_p': 'kPa', 'presion_poros': 'kPa',
        'w': '%', 'w_sat': '%', 'w_optima': '%', 'Sr': '%', 'RC': '%',
        'LL': '%', 'LP': '%', 'IP': '%',
        'qc_CPT': 'kPa', 'fs_CPT': 'kPa',
        'Vs': 'm/s', 'Vp': 'm/s',
        'k': 'm/s', 'k_h': 'm/s', 'k_v': 'm/s',
        'Cv': 'm²/año', 't90': 'años',
        'D50': 'mm', 'FC': '%',
        'sulfatos': '%', 'materia_organica': '%', 'carbonatos': '%',
        'asent_total': 'm', 'asent_elastico': 'm', 'asent_consolidacion': 'm'
    }
    return unidades.get(parametro, '-')

# INFORMACIÓN DE REFERENCIA MEJORADA
with st.expander("📚 RANGOS TÍPICOS DE REFERENCIA"):
    st.markdown("""
    ### **Arenas:**
    - φ: 28°-42° | c: 0-10 kPa | γ: 16-20 kN/m³ | E_s: 10,000-80,000 kPa
    - N-SPT: 10-50 | Permeabilidad: 10⁻¹-10⁻³ m/s
    
    ### **Arcillas:**
    - φ: 0°-25° | c: 5-200 kPa | γ: 16-21 kN/m³ | E_s: 2,000-50,000 kPa  
    - N-SPT: 2-15 | Permeabilidad: 10⁻⁷-10⁻¹¹ m/s
    
    ### **Limos:**
    - φ: 25°-35° | c: 0-25 kPa | γ: 17-20 kN/m³ | E_s: 5,000-30,000 kPa
    - N-SPT: 5-20 | Permeabilidad: 10⁻⁵-10⁻⁸ m/s
    
    ### **Clasificación N-SPT:**
    - 0-4: Muy suelto/blando
    - 4-10: Suelto/blando  
    - 10-30: Medio/firme
    - 30-50: Denso/rígido
    - >50: Muy denso/muy rígido
    """)

st.success("✅ Sistema de parámetros geotécnicos completos configurado correctamente")
# INTERFAZ PRINCIPAL
st.title("🏗️ PARÁMETROS GEOTÉCNICOS")
st.markdown("### Parámetros esenciales para análisis geotécnico")

# FUNCIONES AUXILIARES
def clasificar_suelo_spt(N):
    if N < 4: return "Muy suelto/Blando"
    elif N < 10: return "Suelto"
    elif N < 30: return "Medio"
    elif N < 50: return "Denso"
    else: return "Muy denso"

def clasificar_permeabilidad(k):
    if k > 1e-1: return "Alta - Gravas"
    elif k > 1e-3: return "Media - Arenas"
    elif k > 1e-5: return "Baja - Limos"
    else: return "Muy baja - Arcillas"

# PESTAÑAS PRINCIPALES
tab1, tab2, tab3 = st.tabs(["🧱 PROPIEDADES BÁSICAS", "📊 ENSAYOS Y CLASIFICACIÓN", "⚡ PROPIEDADES AVANZADAS"])

with tab1:
    st.header("Propiedades Fundamentales del Suelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Resistencia al Corte")
        st.session_state.c = st.number_input("Cohesión c (kPa)", 
                                           value=st.session_state.c, min_value=0.0, max_value=500.0, step=5.0)
        st.session_state.phi = st.number_input("Ángulo de fricción φ (°)", 
                                             value=st.session_state.phi, min_value=0.0, max_value=45.0, step=1.0)
        st.session_state.cu = st.number_input("Resistencia no drenada c_u (kPa)", 
                                            value=st.session_state.cu, min_value=0.0, max_value=300.0, step=5.0)
        
        st.subheader("Pesos Unitarios")
        st.session_state.gamma = st.number_input("Peso unitario γ (kN/m³)", 
                                               value=st.session_state.gamma, min_value=10.0, max_value=25.0, step=0.5)
        st.session_state.gamma_sat = st.number_input("Peso unitario saturado γ_sat (kN/m³)", 
                                                   value=st.session_state.gamma_sat, min_value=15.0, max_value=23.0, step=0.5)
    
    with col2:
        st.subheader("Deformabilidad")
        st.session_state.Es = st.number_input("Módulo elasticidad E_s (kPa)", 
                                            value=st.session_state.Es, min_value=1000.0, max_value=200000.0, step=1000.0)
        st.session_state.nu = st.number_input("Coeficiente de Poisson ν", 
                                            value=st.session_state.nu, min_value=0.1, max_value=0.5, step=0.05)
        st.session_state.q_adm = st.number_input("Capacidad portante admisible q_adm (kPa)", 
                                               value=st.session_state.q_adm, min_value=50.0, max_value=1000.0, step=25.0)
        
        st.subheader("Hidrogeología")
        st.session_state.nivel_freatico = st.number_input("Nivel freático (m)", 
                                                        value=st.session_state.nivel_freatico, min_value=0.0, max_value=20.0, step=0.5)

with tab2:
    st.header("Ensayos de Campo y Clasificación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ensayos de Penetración")
        st.session_state.N_SPT = st.number_input("Valor N-SPT (golpes/pie)", 
                                               value=st.session_state.N_SPT, min_value=0, max_value=100, step=1)
        st.session_state.qc_CPT = st.number_input("Resistencia cono q_c (kPa) - CPT", 
                                                value=st.session_state.qc_CPT, min_value=0.0, max_value=30000.0, step=500.0)
        st.session_state.Vs = st.number_input("Velocidad onda de corte V_s (m/s)", 
                                            value=st.session_state.Vs, min_value=50.0, max_value=1000.0, step=25.0)
        
        # Mostrar clasificaciones
        st.metric("Clasificación por N-SPT", clasificar_suelo_spt(st.session_state.N_SPT))
        
    with col2:
        st.subheader("Propiedades Índice")
        st.session_state.w = st.number_input("Contenido humedad w (%)", 
                                           value=st.session_state.w, min_value=0.0, max_value=100.0, step=1.0)
        st.session_state.Gs = st.number_input("Gravedad específica G_s", 
                                            value=st.session_state.Gs, min_value=2.4, max_value=2.8, step=0.01)
        st.session_state.e = st.number_input("Relación de vacíos e", 
                                           value=st.session_state.e, min_value=0.3, max_value=1.5, step=0.05)
        
        st.subheader("Límites de Atterberg")
        st.session_state.LL = st.number_input("Límite líquido LL (%)", 
                                            value=st.session_state.LL, min_value=0.0, max_value=100.0, step=5.0)
        st.session_state.LP = st.number_input("Límite plástico LP (%)", 
                                            value=st.session_state.LP, min_value=0.0, max_value=80.0, step=5.0)
        st.session_state.IP = st.number_input("Índice plástico IP (%)", 
                                            value=st.session_state.IP, min_value=0.0, max_value=60.0, step=5.0)

with tab3:
    st.header("Propiedades Avanzadas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Compresibilidad")
        st.session_state.Cc = st.number_input("Índice compresión C_c", 
                                            value=st.session_state.Cc, min_value=0.01, max_value=1.0, step=0.05)
        st.session_state.sigma_p = st.number_input("Presión preconsolidación σ'p (kPa)", 
                                                 value=st.session_state.sigma_p, min_value=10.0, max_value=1000.0, step=25.0)
        
        # Calcular OCR
        sigma_vo = st.session_state.gamma * 5.0  # Suponiendo 5m de profundidad
        OCR = st.session_state.sigma_p / sigma_vo if sigma_vo > 0 else 1.0
        st.metric("OCR (Razón sobreconsolidación)", f"{OCR:.2f}")
        
        st.subheader("Clasificación SUCS")
        sucs_grupos = ['GW', 'GP', 'GM', 'GC', 'SW', 'SP', 'SM', 'SC', 'ML', 'CL', 'OL', 'MH', 'CH', 'OH', 'PT']
        st.session_state.sucs_grupo = st.selectbox("Grupo SUCS", sucs_grupos, index=sucs_grupos.index(st.session_state.sucs_grupo))
    
    with col2:
        st.subheader("Resistencia Avanzada")
        st.session_state.su = st.number_input("Resistencia corte no drenado s_u (kPa)", 
                                            value=st.session_state.su, min_value=0.0, max_value=500.0, step=10.0)
        
        st.subheader("Permeabilidad")
        k_value = st.number_input("Permeabilidad k (cm/s)", value=1e-5, format="%.2e", min_value=1e-9, max_value=1e-1, step=1e-6)
        st.metric("Clasificación permeabilidad", clasificar_permeabilidad(k_value))

# RESUMEN Y EXPORTACIÓN
st.markdown("---")
st.header("📋 RESUMEN DE PARÁMETROS")

# Crear DataFrame de resumen
resumen_data = {
    'Parámetro': [
        'Cohesión c (kPa)', 'Ángulo fricción φ (°)', 'Peso unitario γ (kN/m³)',
        'Módulo elasticidad E_s (kPa)', 'Capacidad portante q_adm (kPa)',
        'N-SPT (golpes/pie)', 'Resistencia cono q_c (kPa)', 'Velocidad V_s (m/s)',
        'Contenido humedad w (%)', 'Índice plástico IP (%)', 'Grupo SUCS'
    ],
    'Valor': [
        st.session_state.c, st.session_state.phi, st.session_state.gamma,
        st.session_state.Es, st.session_state.q_adm,
        st.session_state.N_SPT, st.session_state.qc_CPT, st.session_state.Vs,
        st.session_state.w, st.session_state.IP, st.session_state.sucs_grupo
    ]
}

df_resumen = pd.DataFrame(resumen_data)
st.dataframe(df_resumen, use_container_width=True, hide_index=True)

# Botones de acción
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("💾 Guardar Parámetros", use_container_width=True):
        # Crear DataFrame completo
        todos_parametros = {key: st.session_state[key] for key in st.session_state.keys() 
                           if not key.startswith('_') and not callable(st.session_state[key])}
        df_completo = pd.DataFrame.from_dict(todos_parametros, orient='index', columns=['Valor'])
        df_completo.index.name = 'Parámetro'
        
        # Convertir a CSV
        csv = df_completo.to_csv()
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name="parametros_geotecnicos.csv",
            mime="text/csv",
            use_container_width=True
        )

with col_btn2:
    if st.button("🔄 Restablecer Valores", use_container_width=True):
        for key in st.session_state.keys():
            if not key.startswith('_') and not callable(st.session_state[key]):
                del st.session_state[key]
        st.rerun()

# INFORMACIÓN DE REFERENCIA
with st.expander("📚 Rangos Típicos de Referencia"):
    st.markdown("""
    **Arenas:**
    - φ: 28°-42° | c: 0 kPa | γ: 16-20 kN/m³ | E_s: 10,000-80,000 kPa
    
    **Arcillas:**
    - φ: 0°-25° | c: 5-200 kPa | γ: 16-21 kN/m³ | E_s: 2,000-50,000 kPa
    
    **Clasificación N-SPT:**
    - 0-4: Muy suelto/blando
    - 4-10: Suelto  
    - 10-30: Medio
    - 30-50: Denso
    - >50: Muy denso
    """)

st.success("✅ Parámetros geotécnicos configurados correctamente")

#--------------------------------------------------
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




