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
st.set_page_config(page_title="Parámetros Geotécnicos Completos", layout="wide")

# Inicialización de session_state con TODOS los parámetros
def inicializar_parametros():
    parametros_base = {
        # PROPIEDADES BÁSICAS DEL SUELO
        'gamma': 18.0,
        'c': 10.0,
        'phi': 30.0,
        
        # PROPIEDADES DE DEFORMABILIDAD
        'Es': 30000.0,
        'nu': 0.35,
        
        # CAPACIDAD PORTANTE
        'q_adm': 150.0,
        
        # ENSAYOS DE CAMPO
        'N_SPT': 15,
        'qc_CPT': 5000.0,
        'fs_CPT': 100.0,
        
        # PROPIEDADES FÍSICAS ADICIONALES
        'w': 18.0,           # Contenido de humedad %
        'Gs': 2.65,          # Gravedad específica
        'e': 0.65,           # Relación de vacíos
        'Sr': 75.0,          # Grado de saturación %
        'LL': 45.0,          # Límite líquido %
        'LP': 25.0,          # Límite plástico %
        'IP': 20.0,          # Índice plástico %
        
        # CLASIFICACIÓN SUCS
        'sucs_grupo': 'SM',  # Grupo SUCS
        'sucs_subgrupo': 'Poorly graded sand with silt',
        
        # HIDROGEOLOGÍA
        'nivel_freatico': 2.0,
        'k': 1e-5,           # Permeabilidad cm/s
        'gamma_sat': 20.0,   # Peso unitario saturado
        
        # PARÁMETROS DE COMPRESIBILIDAD
        'Cc': 0.25,          # Índice de compresión
        'Cs': 0.05,          # Índice de hinchamiento
        'eo': 0.70,          # Relación de vacíos inicial
        'sigma_p': 100.0,    # Presión de preconsolidación kPa
        
        # PARÁMETROS SÍSMICOS
        'Vs': 200.0,         # Velocidad de onda de corte m/s
        'amax': 0.3,         # Aceleración máxima g
        'M': 6.0,            # Magnitud sísmica
        
        # PARÁMETROS DE RESISTENCIA AVANZADOS
        'cu': 25.0,          # Resistencia no drenada kPa
        'su': 20.0,          # Resistencia al corte no drenado kPa
        'phi_pico': 32.0,    # Ángulo de fricción pico
        'phi_residual': 28.0,# Ángulo de fricción residual
        
        # PROPIEDADES QUÍMICAS
        'pH': 7.0,
        'sulfatos': 0.1,     # Contenido de sulfatos %
        'materia_organica': 1.0, # Materia orgánica %
        
        # CARACTERÍSTICAS ESTRATIGRÁFICAS
        'profundidad_estrato': 5.0,
        'espesor_estrato': 3.0,
        'numero_estratos': 3,
        
        # PARÁMETROS DE EXPANSIVIDAD
        'indice_expansion': 2.0, # % de expansión
        'presion_hinchamiento': 50.0, # kPa
        
        # PROPIEDADES DINÁMICAS
        'Gmax': 50000.0,     # Módulo de corte máximo kPa
        'D': 5.0,            # Amortiguamiento %
        
        # CARGAS ESTRUCTURALES
        'N': 500.0,
        'Mx': 50.0,
        'My': 30.0,
        
        # MATERIALES CIMENTACIÓN
        'fc': 21.0,
        'fy': 420.0,
        'recubrimiento': 0.075,
        
        # FACTORES DE SEGURIDAD
        'FS': 3.0,
        'asent_max': 0.025,
        
        # COSTOS UNITARIOS
        'concreto_Sm3': 250.0,
        'acero_Skg': 3.5,
        'excav_Sm3': 50.0,
        
        # RANGOS DE DISEÑO
        'B_min': 0.5, 'B_max': 3.0,
        'L_min': 0.5, 'L_max': 3.0,
        'h_min': 0.3, 'h_max': 1.0,
        
        # CONFIGURACIÓN ANÁLISIS
        'nB': 10, 'nL': 10, 'nh': 5
    }
    
    for key, value in parametros_base.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Llamar a la inicialización
inicializar_parametros()

# INTERFAZ PRINCIPAL
st.title("🏗️ PARÁMETROS GEOTÉCNICOS COMPLETOS")
st.markdown("### Base de datos completa de propiedades del suelo para proyectos geotécnicos")

# CREAR PESTAÑAS PARA ORGANIZAR TODOS LOS PARÁMETROS
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🧱 PROPIEDADES BÁSICAS", 
    "📊 ENSAYOS DE CAMPO", 
    "💧 PROPIEDADES FÍSICAS", 
    "⚡ PROPIEDADES MECÁNICAS",
    "🌋 PARÁMETROS SÍSMICOS",
    "🧪 PROPIEDADES QUÍMICAS",
    "🏗️ DISEÑO ESTRUCTURAL"
])

with tab1:
    st.header("🧱 PROPIEDADES BÁSICAS DEL SUELO")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Resistencia al Corte")
        st.session_state.c = st.number_input("Cohesión c (kPa)", 
                                           value=st.session_state.c, 
                                           min_value=0.0, max_value=500.0, step=5.0,
                                           help="Cohesión del suelo obtenida de ensayos triaxiales o directos")
        
        st.session_state.phi = st.number_input("Ángulo de fricción φ (°)", 
                                             value=st.session_state.phi, 
                                             min_value=0.0, max_value=45.0, step=1.0,
                                             help="Ángulo de fricción interna del suelo")
        
        st.session_state.cu = st.number_input("Resistencia no drenada c_u (kPa)", 
                                            value=st.session_state.cu, 
                                            min_value=0.0, max_value=300.0, step=5.0)
    
    with col2:
        st.subheader("Pesos Unitarios")
        st.session_state.gamma = st.number_input("Peso unitario γ (kN/m³)", 
                                               value=st.session_state.gamma, 
                                               min_value=10.0, max_value=25.0, step=0.5,
                                               help="Peso unitario natural del suelo")
        
        st.session_state.gamma_sat = st.number_input("Peso unitario saturado γ_sat (kN/m³)", 
                                                   value=st.session_state.gamma_sat, 
                                                   min_value=15.0, max_value=23.0, step=0.5)
        
        st.session_state.nivel_freatico = st.number_input("Nivel freático (m)", 
                                                        value=st.session_state.nivel_freatico, 
                                                        min_value=0.0, max_value=20.0, step=0.5)
    
    with col3:
        st.subheader("Deformabilidad")
        st.session_state.Es = st.number_input("Módulo elasticidad E_s (kPa)", 
                                            value=st.session_state.Es, 
                                            min_value=1000.0, max_value=200000.0, step=1000.0,
                                            help="Módulo de elasticidad del suelo para cálculo de asentamientos")
        
        st.session_state.nu = st.number_input("Coeficiente de Poisson ν", 
                                            value=st.session_state.nu, 
                                            min_value=0.1, max_value=0.5, step=0.05,
                                            help="Relación de Poisson para análisis de deformaciones")
        
        st.session_state.q_adm = st.number_input("Capacidad portante admisible q_adm (kPa)", 
                                               value=st.session_state.q_adm, 
                                               min_value=50.0, max_value=1000.0, step=25.0)

with tab2:
    st.header("📊 ENSAYOS DE CAMPO Y CLASIFICACIÓN")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ensayos de Penetración")
        st.session_state.N_SPT = st.number_input("Valor N-SPT (golpes/pie)", 
                                               value=st.session_state.N_SPT, 
                                               min_value=0, max_value=100, step=1,
                                               help="Resistencia a la penetración estándar")
        
        st.session_state.qc_CPT = st.number_input("Resistencia cono q_c (kPa) - CPT", 
                                                value=st.session_state.qc_CPT, 
                                                min_value=0.0, max_value=30000.0, step=500.0)
        
        st.session_state.fs_CPT = st.number_input("Resistencia friccional f_s (kPa) - CPT", 
                                                value=st.session_state.fs_CPT, 
                                                min_value=0.0, max_value=500.0, step=10.0)
        
        st.session_state.Vs = st.number_input("Velocidad onda de corte V_s (m/s)", 
                                            value=st.session_state.Vs, 
                                            min_value=50.0, max_value=1000.0, step=25.0)
    
    with col2:
        st.subheader("Clasificación SUCS")
        sucs_grupos = ['GW', 'GP', 'GM', 'GC', 'SW', 'SP', 'SM', 'SC', 'ML', 'CL', 'OL', 'MH', 'CH', 'OH', 'PT']
        st.session_state.sucs_grupo = st.selectbox("Grupo SUCS", 
                                                 sucs_grupos, 
                                                 index=sucs_grupos.index(st.session_state.sucs_grupo))
        
        st.session_state.sucs_subgrupo = st.text_input("Descripción SUCS", 
                                                     value=st.session_state.sucs_subgrupo,
                                                     help="Descripción completa del suelo según SUCS")

with tab3:
    st.header("💧 PROPIEDADES FÍSICAS E HIDROLÓGICAS")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Propiedades Físicas")
        st.session_state.w = st.number_input("Contenido humedad w (%)", 
                                           value=st.session_state.w, 
                                           min_value=0.0, max_value=100.0, step=1.0)
        
        st.session_state.Gs = st.number_input("Gravedad específica G_s", 
                                            value=st.session_state.Gs, 
                                            min_value=2.4, max_value=2.8, step=0.01)
        
        st.session_state.e = st.number_input("Relación de vacíos e", 
                                           value=st.session_state.e, 
                                           min_value=0.3, max_value=1.5, step=0.05)
        
        st.session_state.Sr = st.number_input("Grado de saturación S_r (%)", 
                                            value=st.session_state.Sr, 
                                            min_value=0.0, max_value=100.0, step=5.0)
    
    with col2:
        st.subheader("Límites de Consistencia")
        st.session_state.LL = st.number_input("Límite líquido LL (%)", 
                                            value=st.session_state.LL, 
                                            min_value=0.0, max_value=100.0, step=5.0)
        
        st.session_state.LP = st.number_input("Límite plástico LP (%)", 
                                            value=st.session_state.LP, 
                                            min_value=0.0, max_value=80.0, step=5.0)
        
        st.session_state.IP = st.number_input("Índice plástico IP (%)", 
                                            value=st.session_state.IP, 
                                            min_value=0.0, max_value=60.0, step=5.0)
        
        # Clasificación visual
        consistencia = st.selectbox("Consistencia (arcillas)", 
                                  ["Muy blanda", "Blanda", "Media", "Rígida", "Muy rígida", "Dura"])
    
    with col3:
        st.subheader("Propiedades Hidráulicas")
        st.session_state.k = st.number_input("Permeabilidad k (cm/s)", 
                                           value=st.session_state.k, 
                                           format="%.2e",
                                           help="Coeficiente de permeabilidad")
        
        # Clasificación de permeabilidad basada en valor
        k_value = st.session_state.k
        if k_value > 1e-1:
            perm_clasif = "Alta - Gravas"
        elif k_value > 1e-3:
            perm_clasif = "Media - Arenas"
        elif k_value > 1e-5:
            perm_clasif = "Baja - Limos"
        else:
            perm_clasif = "Muy baja - Arcillas"
        
        st.metric("Clasificación permeabilidad", perm_clasif)

with tab4:
    st.header("⚡ PROPIEDADES MECÁNICAS AVANZADAS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Compresibilidad")
        st.session_state.Cc = st.number_input("Índice compresión C_c", 
                                            value=st.session_state.Cc, 
                                            min_value=0.01, max_value=1.0, step=0.05)
        
        st.session_state.Cs = st.number_input("Índice hinchamiento C_s", 
                                            value=st.session_state.Cs, 
                                            min_value=0.01, max_value=0.5, step=0.01)
        
        st.session_state.eo = st.number_input("Relación vacíos inicial e₀", 
                                            value=st.session_state.eo, 
                                            min_value=0.3, max_value=1.5, step=0.05)
        
        st.session_state.sigma_p = st.number_input("Presión preconsolidación σ'p (kPa)", 
                                                 value=st.session_state.sigma_p, 
                                                 min_value=10.0, max_value=1000.0, step=25.0)
        
        # Calcular OCR
        sigma_vo = st.session_state.gamma * 5.0  # Suponiendo 5m de profundidad
        OCR = st.session_state.sigma_p / sigma_vo if sigma_vo > 0 else 1.0
        st.metric("OCR (Razón sobreconsolidación)", f"{OCR:.2f}")
    
    with col2:
        st.subheader("Resistencia Avanzada")
        st.session_state.phi_pico = st.number_input("Ángulo fricción pico φ_pico (°)", 
                                                  value=st.session_state.phi_pico, 
                                                  min_value=20.0, max_value=45.0, step=1.0)
        
        st.session_state.phi_residual = st.number_input("Ángulo fricción residual φ_residual (°)", 
                                                      value=st.session_state.phi_residual, 
                                                      min_value=15.0, max_value=40.0, step=1.0)
        
        st.session_state.su = st.number_input("Resistencia corte no drenado s_u (kPa)", 
                                            value=st.session_state.su, 
                                            min_value=0.0, max_value=500.0, step=10.0)
        
        st.session_state.Gmax = st.number_input("Módulo corte máximo G_max (kPa)", 
                                              value=st.session_state.Gmax, 
                                              min_value=1000.0, max_value=200000.0, step=5000.0)

with tab5:
    st.header("🌋 PARÁMETROS SÍSMICOS Y DINÁMICOS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Propiedades Dinámicas")
        st.session_state.Vs = st.number_input("Velocidad onda corte V_s (m/s)", 
                                            value=st.session_state.Vs, 
                                            min_value=100.0, max_value=800.0, step=25.0)
        
        st.session_state.Gmax = st.number_input("Módulo corte máximo G_max (kPa)", 
                                              value=st.session_state.Gmax, 
                                              min_value=10000.0, max_value=500000.0, step=10000.0)
        
        st.session_state.D = st.number_input("Amortiguamiento D (%)", 
                                           value=st.session_state.D, 
                                           min_value=1.0, max_value=10.0, step=0.5)
        
        # Clasificación sísmica basada en Vs
        vs_value = st.session_state.Vs
        if vs_value > 360:
            sitio_clasif = "Tipo A - Roca dura"
        elif vs_value > 180:
            sitio_clasif = "Tipo B - Roca"
        elif vs_value > 90:
            sitio_clasif = "Tipo C - Suelo muy denso"
        else:
            sitio_clasif = "Tipo D - Suelo blando"
        
        st.metric("Clasificación sitio sísmico", sitio_clasif)
    
    with col2:
        st.subheader("Cargas Sísmicas")
        st.session_state.amax = st.number_input("Aceleración máxima a_max (g)", 
                                              value=st.session_state.amax, 
                                              min_value=0.1, max_value=1.0, step=0.05)
        
        st.session_state.M = st.number_input("Magnitud sísmica M", 
                                           value=st.session_state.M, 
                                           min_value=5.0, max_value=8.5, step=0.1)

with tab6:
    st.header("🧪 PROPIEDADES QUÍMICAS Y ESPECIALES")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Propiedades Químicas")
        st.session_state.pH = st.number_input("pH del suelo", 
                                            value=st.session_state.pH, 
                                            min_value=2.0, max_value=12.0, step=0.1)
        
        st.session_state.sulfatos = st.number_input("Contenido sulfatos SO₄ (%)", 
                                                  value=st.session_state.sulfatos, 
                                                  min_value=0.0, max_value=5.0, step=0.1)
        
        st.session_state.materia_organica = st.number_input("Materia orgánica (%)", 
                                                          value=st.session_state.materia_organica, 
                                                          min_value=0.0, max_value=10.0, step=0.5)
    
    with col2:
        st.subheader("Propiedades Expansivas")
        st.session_state.indice_expansion = st.number_input("Índice de expansión (%)", 
                                                          value=st.session_state.indice_expansion, 
                                                          min_value=0.0, max_value=20.0, step=0.5)
        
        st.session_state.presion_hinchamiento = st.number_input("Presión hinchamiento (kPa)", 
                                                              value=st.session_state.presion_hinchamiento, 
                                                              min_value=0.0, max_value=200.0, step=10.0)
        
        # Clasificación de expansividad
        expansion = st.session_state.indice_expansion
        if expansion > 10:
            exp_clasif = "Muy alto"
        elif expansion > 5:
            exp_clasif = "Alto"
        elif expansion > 2:
            exp_clasif = "Medio"
        else:
            exp_clasif = "Bajo"
        
        st.metric("Potencial expansivo", exp_clasif)

with tab7:
    st.header("🏗️ PARÁMETROS DE DISEÑO ESTRUCTURAL")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cargas Estructurales")
        st.session_state.N = st.number_input("Carga axial N (kN)", 
                                           value=st.session_state.N, 
                                           min_value=100.0, max_value=10000.0, step=100.0)
        
        st.session_state.Mx = st.number_input("Momento Mx (kN·m)", 
                                            value=st.session_state.Mx, 
                                            min_value=0.0, max_value=1000.0, step=25.0)
        
        st.session_state.My = st.number_input("Momento My (kN·m)", 
                                            value=st.session_state.My, 
                                            min_value=0.0, max_value=1000.0, step=25.0)
    
    with col2:
        st.subheader("Materiales y Factores")
        st.session_state.fc = st.number_input("Resistencia concreto f'c (MPa)", 
                                            value=st.session_state.fc, 
                                            min_value=17.5, max_value=35.0, step=3.5)
        
        st.session_state.fy = st.number_input("Fluencia acero fy (MPa)", 
                                            value=st.session_state.fy, 
                                            min_value=280.0, max_value=500.0, step=20.0)
        
        st.session_state.FS = st.number_input("Factor seguridad FS", 
                                            value=st.session_state.FS, 
                                            min_value=2.0, max_value=5.0, step=0.5)
        
        st.session_state.asent_max = st.number_input("Asentamiento máximo (m)", 
                                                   value=st.session_state.asent_max, 
                                                   min_value=0.01, max_value=0.10, step=0.005)

# SECCIÓN DE RESUMEN Y EXPORTACIÓN
st.markdown("---")
st.header("📋 RESUMEN DE PARÁMETROS GEOTÉCNICOS")

# Crear resumen en columnas
col_res1, col_res2, col_res3 = st.columns(3)

with col_res1:
    st.subheader("🧱 Propiedades Básicas")
    st.write(f"**Cohesión (c):** {st.session_state.c} kPa")
    st.write(f"**Ángulo fricción (φ):** {st.session_state.phi}°")
    st.write(f"**Peso unitario (γ):** {st.session_state.gamma} kN/m³")
    st.write(f"**Módulo elasticidad (E_s):** {st.session_state.Es} kPa")

with col_res2:
    st.subheader("📊 Ensayos Campo")
    st.write(f"**N-SPT:** {st.session_state.N_SPT} golpes/pie")
    st.write(f"**Resistencia cono (q_c):** {st.session_state.qc_CPT} kPa")
    st.write(f"**Velocidad onda (V_s):** {st.session_state.Vs} m/s")
    st.write(f"**Grupo SUCS:** {st.session_state.sucs_grupo}")

with col_res3:
    st.subheader("⚡ Propiedades Mecánicas")
    st.write(f"**Capacidad portante:** {st.session_state.q_adm} kPa")
    st.write(f"**Índice compresión (C_c):** {st.session_state.Cc}")
    st.write(f"**Presión preconsolidación:** {st.session_state.sigma_p} kPa")
    st.write(f"**Resistencia no drenada:** {st.session_state.cu} kPa")

# Botón para exportar datos
if st.button("📤 Exportar Parámetros a CSV"):
    # Crear DataFrame con todos los parámetros
    parametros_df = pd.DataFrame.from_dict(st.session_state, orient='index', columns=['Valor'])
    parametros_df.index.name = 'Parámetro'
    
    # Descargar CSV
    csv = parametros_df.to_csv()
    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name="parametros_geotecnicos.csv",
        mime="text/csv"
    )
    
    st.success("✅ Parámetros listos para exportar")

# VISUALIZACIÓN DE RELACIONES
st.markdown("---")
st.header("📊 RELACIONES ENTRE PARÁMETROS")

# Gráfico interactivo de relaciones
col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    # Seleccionar parámetros para gráfico
    param_x = st.selectbox("Eje X", 
                          ['c', 'phi', 'gamma', 'Es', 'N_SPT', 'qc_CPT', 'Vs'])
    param_y = st.selectbox("Eje Y", 
                          ['q_adm', 'cu', 'Gmax', 'Cc', 'IP'])

with col_viz2:
    # Crear datos de ejemplo para visualización
    datos_ejemplo = pd.DataFrame({
        'c': [5, 10, 15, 20, 25, 30],
        'phi': [25, 28, 30, 32, 34, 36],
        'gamma': [16, 17, 18, 19, 20, 21],
        'Es': [20000, 30000, 40000, 50000, 60000, 70000],
        'N_SPT': [10, 15, 20, 25, 30, 35],
        'qc_CPT': [2000, 4000, 6000, 8000, 10000, 12000],
        'Vs': [150, 200, 250, 300, 350, 400],
        'q_adm': [100, 150, 200, 250, 300, 350],
        'cu': [20, 30, 40, 50, 60, 70],
        'Gmax': [30000, 40000, 50000, 60000, 70000, 80000],
        'Cc': [0.3, 0.25, 0.2, 0.15, 0.1, 0.05],
        'IP': [15, 20, 25, 30, 35, 40]
    })
    
    fig = px.scatter(datos_ejemplo, x=param_x, y=param_y, 
                    title=f"Relación {param_x} vs {param_y}",
                    trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

# INFORMACIÓN ADICIONAL
with st.expander("📚 Información Técnica de Referencia"):
    st.markdown("""
    ### Rangos Típicos de Parámetros Geotécnicos
    
    **Arenas:**
    - φ: 28°-42°
    - c: 0 kPa
    - γ: 16-20 kN/m³
    - E_s: 10,000-80,000 kPa
    
    **Arcillas:**
    - φ: 0°-25°
    - c: 5-200 kPa
    - γ: 16-21 kN/m³
    - E_s: 2,000-50,000 kPa
    
    **Clasificación por N-SPT:**
    - 0-4: Muy suelto/blando
    - 4-10: Suelto
    - 10-30: Medio
    - 30-50: Denso
    - >50: Muy denso
    """)

st.success("🎯 Todos los parámetros geotécnicos han sido configurados correctamente")

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

