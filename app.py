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

# Inicialización de session_state
def inicializar_parametros():
    parametros_base = {
        # PROPIEDADES BÁSICAS DEL SUELO
        'gamma': 18.0,
        'c': 10.0,
        'phi': 30.0,
        'Es': 30000.0,
        'nu': 0.35,
        'q_adm': 150.0,
        
        # ENSAYOS DE CAMPO
        'N_SPT': 15,
        'qc_CPT': 5000.0,
        'Vs': 200.0,
        
        # PROPIEDADES FÍSICAS
        'w': 18.0,
        'Gs': 2.65,
        'e': 0.65,
        'LL': 45.0,
        'LP': 25.0,
        'IP': 20.0,
        
        # CLASIFICACIÓN
        'sucs_grupo': 'SM',
        
        # HIDROGEOLOGÍA
        'nivel_freatico': 2.0,
        'gamma_sat': 20.0,
        
        # COMPRESIBILIDAD
        'Cc': 0.25,
        'sigma_p': 100.0,
        
        # RESISTENCIA AVANZADA
        'cu': 25.0,
        'su': 20.0,
    }
    
    for key, value in parametros_base.items():
        if key not in st.session_state:
            st.session_state[key] = value

inicializar_parametros()

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

