# app.py — Optimización de Cimentaciones (mínima+) con ML para qu y Asentamiento
# - ML qu (opcional): GradientBoosting + 10-fold CV (métricas R2, RMSE, MAE, nRMSE)
# - ML S (opcional): GradientBoosting + 10-fold CV siguiendo la idea del paper (GB, 10-fold) [Ngo & Tran 2024]
# - NF: gamma_eff automático (γ_boyante) y ajuste de Es por NF alto según tipo de suelo
# - Verificaciones de servicio: q_serv ≤ q_adm, q_max ≤ q_adm, s ≤ s_adm
# - Recomendador "IA" de tipo de cimiento (motor experto explicable)

import math
import numpy as np
import pandas as pd
import streamlit as st

# ====== ML opcional (importaciones defensivas) ======
try:
    from sklearn.model_selection import train_test_split, KFold, cross_val_predict
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False
    train_test_split = KFold = cross_val_predict = GradientBoostingRegressor = None
    mean_squared_error = mean_absolute_error = r2_score = None

st.set_page_config(page_title="Optimización de Cimentaciones — Minimal+", layout="wide")
st.title("Optimización de Cimentaciones (mínima+)")
st.caption("2 FO (costo/asentamiento) + ML opcional para capacidad y asentamiento + NF + recomendación de tipo de cimiento.")

# ======================== Entradas principales ========================
c0, c00 = st.columns(2)
with c0:
    N = st.number_input("Carga axial N (kN)", min_value=100.0, max_value=2e5, value=800.0, step=10.0)
    phi = st.number_input("ϕ (°)", min_value=0.0, max_value=45.0, value=32.0, step=0.5)
    # DATO de densidades para NF
    gamma_dry = st.number_input("γ (kN/m³) por encima del NF", min_value=10.0, max_value=24.0, value=18.0, step=0.1)
    gamma_sat = st.number_input("γ_sat (kN/m³) por debajo del NF", min_value=14.0, max_value=24.0, value=20.0, step=0.1)
    soil_type = st.selectbox("Tipo de suelo predominante para asentamiento", ["Arena", "Arcilla", "Mixto"], index=0)
with c00:
    D = st.number_input("Profundidad de base D (m)", min_value=0.5, max_value=6.0, value=1.5, step=0.1)
    FS = st.number_input("FS (capacidad)", min_value=1.5, max_value=4.0, value=2.5, step=0.1)
    L_over_B = st.number_input("Relación L/B (constante)", min_value=0.8, max_value=2.0, value=1.0, step=0.05)
    s_adm_mm = st.number_input("Asentamiento admisible sₐ (mm)", min_value=5.0, max_value=50.0, value=25.0, step=1.0)

# Nivel freático
st.markdown("**Nivel freático**")
cNF1, cNF2, cNF3 = st.columns(3)
with cNF1:
    z_ground = st.number_input("Cota del terreno (m, referencia)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
with cNF2:
    z_base = st.number_input("Cota de la base (m)", min_value=-100.0, max_value=100.0, value=-D, step=0.1)
with cNF3:
    z_NF = st.number_input("Cota del NF (m)", min_value=-100.0, max_value=100.0, value=-0.5, step=0.1)

gamma_w = 9.81  # kN/m³ aproximado

def gamma_effective_at_depth(gamma_dry, gamma_sat, z, z_nf):
    """Devuelve gamma_eff en la cota 'z' considerando NF en z_nf.
    Por encima del NF: gamma_dry; por debajo: gamma_sat - gamma_w (boyante)."""
    if z >= z_nf:
        return gamma_dry
    return max(gamma_sat - gamma_w, 0.0)

# ======================== Rangos de búsqueda ==========================
st.markdown("**Rangos de búsqueda (B y h)**")
c3, c4, c5 = st.columns(3)
with c3:
    Bmin = st.number_input("B min (m)", min_value=0.8, max_value=6.0, value=1.0, step=0.1)
    Bmax = st.number_input("B max (m)", min_value=0.8, max_value=6.0, value=3.0, step=0.1)
with c4:
    hmin = st.number_input("h min (m)", min_value=0.3, max_value=2.0, value=0.5, step=0.05)
    hmax = st.number_input("h max (m)", min_value=0.3, max_value=2.0, value=1.2, step=0.05)
with c5:
    nB = st.number_input("Puntos B", min_value=5, max_value=80, value=30, step=1)
    nh = st.number_input("Puntos h", min_value=3, max_value=60, value=12, step=1)

st.markdown("---")

# ======================== ML qu (opcional) ============================
st.subheader("ML opcional (capacidad última qu)")
if not SKLEARN_OK:
    st.warning("Para usar ML, agrega `scikit-learn>=1.3,<1.5`. Si no, se usará Meyerhof.")

# Estados
if "QU_MODEL" not in st.session_state:
    st.session_state.QU_MODEL = None
if "QU_METRICS" not in st.session_state:
    st.session_state.QU_METRICS = {}

up_qu = st.file_uploader("CSV qu (gamma_eff,B,D,phi,L_over_B,qu)", type=["csv"], key="up_qu")
use_ml_qu = st.toggle("Usar ML para qu si está entrenado", value=False, key="tog_qu")

def train_qu(csvfile):
    df = pd.read_csv(csvfile)
    req = ["gamma_eff", "B", "D", "phi", "L_over_B", "qu"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan columnas en CSV de qu: {miss}")

    X = df[["gamma_eff","B","D","phi","L_over_B"]].astype(float)
    y = df["qu"].astype(float)

    # 10-fold CV (como en el paper de asentamiento, adoptamos enfoque similar)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    gbr = GradientBoostingRegressor(random_state=42)

    yhat_cv = cross_val_predict(gbr, X, y, cv=kf)
    # Métricas
    rmse = float(mean_squared_error(y, yhat_cv, squared=False))
    mae  = float(mean_absolute_error(y, yhat_cv))
    r2   = float(r2_score(y, yhat_cv))
    med  = float(np.median(y)) if len(y) else float("nan")
    nrmse_med = float(rmse / med * 100) if np.isfinite(med) and med != 0 else float("nan")
    bias = float(np.mean(yhat_cv - y))
    st.session_state.QU_METRICS = {"rmse": rmse, "mae": mae, "r2": r2, "nrmse_med": nrmse_med, "bias": bias, "n": int(len(y))}

    # Entrena final con todo el dataset
    gbr.fit(X, y)
    return gbr

cbtn1, cbtn2 = st.columns(2)
with cbtn1:
    if st.button("Entrenar modelo de qu", use_container_width=True):
        if not SKLEARN_OK:
            st.error("scikit-learn no está disponible en el entorno.")
        elif up_qu is None:
            st.warning("Sube el CSV de qu.")
        else:
            try:
                model = train_qu(up_qu)
                st.session_state.QU_MODEL = model
                m = st.session_state.QU_METRICS
                st.success(f"Modelo qu entrenado | R²={m['r2']:.3f} · RMSE≈{m['rmse']:.2f} kPa · MAE≈{m['mae']:.2f} kPa · nRMSE≈{m['nrmse_med']:.1f}% · Sesgo={m['bias']:+.2f} kPa")
            except Exception as e:
                st.error(f"Error entrenando qu: {e}")

if st.session_state.QU_METRICS:
    m = st.session_state.QU_METRICS
    st.caption(f"Validación (10-fold, todo el dataset): n={m['n']} | R²={m['r2']:.3f} · RMSE={m['rmse']:.2f} kPa · MAE={m['mae']:.2f} kPa · nRMSE={m['nrmse_med']:.1f}% · Sesgo={m['bias']:+.2f} kPa")

st.markdown("---")

# ======================== ML Asentamiento (opcional) ==================
st.subheader("ML opcional (asentamiento S) — GB + 10-fold (paper)")

if "S_MODEL" not in st.session_state:
    st.session_state.S_MODEL = None
if "S_METRICS" not in st.session_state:
    st.session_state.S_METRICS = {}

up_s = st.file_uploader("CSV asentamiento (B,L_over_B,D_over_B,q,SPT,d,S_mm)", type=["csv"], key="up_s")
use_ml_s = st.toggle("Usar ML para S si está entrenado", value=False, key="tog_s")

def train_s(csvfile):
    df = pd.read_csv(csvfile)
    req = ["B","L_over_B","D_over_B","q","SPT","d","S_mm"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan columnas en CSV de asentamiento: {miss}")

    X = df[["B","L_over_B","D_over_B","q","SPT","d"]].astype(float)
    y = df["S_mm"].astype(float)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    gbr = GradientBoostingRegressor(random_state=42)
    yhat_cv = cross_val_predict(gbr, X, y, cv=kf)

    rmse = float(mean_squared_error(y, yhat_cv, squared=False))
    mae  = float(mean_absolute_error(y, yhat_cv))
    r2   = float(r2_score(y, yhat_cv))
    med  = float(np.median(y)) if len(y) else float("nan")
    nrmse_med = float(rmse / med * 100) if np.isfinite(med) and med != 0 else float("nan")
    bias = float(np.mean(yhat_cv - y))
    st.session_state.S_METRICS = {"rmse": rmse, "mae": mae, "r2": r2, "nrmse_med": nrmse_med, "bias": bias, "n": int(len(y))}

    gbr.fit(X, y)
    return gbr

with cbtn2:
    if st.button("Entrenar modelo de asentamiento", use_container_width=True):
        if not SKLEARN_OK:
            st.error("scikit-learn no está disponible en el entorno.")
        elif up_s is None:
            st.warning("Sube el CSV de asentamiento.")
        else:
            try:
                model = train_s(up_s)
                st.session_state.S_MODEL = model
                m = st.session_state.S_METRICS
                st.success(f"Modelo S entrenado | R²={m['r2']:.3f} · RMSE≈{m['rmse']:.2f} mm · MAE≈{m['mae']:.2f} mm · nRMSE≈{m['nrmse_med']:.1f}% · Sesgo={m['bias']:+.2f} mm")
            except Exception as e:
                st.error(f"Error entrenando S: {e}")

if st.session_state.S_METRICS:
    m = st.session_state.S_METRICS
    st.caption(f"Validación (10-fold, todo el dataset): n={m['n']} | R²={m['r2']:.3f} · RMSE={m['rmse']:.2f} mm · MAE={m['mae']:.2f} mm · nRMSE={m['nrmse_med']:.1f}% · Sesgo={m['bias']:+.2f} mm")
st.markdown("---")

# ======================== Modelos clásicos ============================
def bearing_factors(phi_deg: float):
    phi_rad = math.radians(phi_deg)
    if phi_rad < 1e-6:
        Nq, Nc, Ng = 1.0, 5.14, 0.0
    else:
        Nq = math.e ** (math.pi * math.tan(phi_rad)) * (math.tan(math.pi/4 + phi_rad/2))**2
        Nc = (Nq - 1.0) / math.tan(phi_rad)
        Ng = 2.0 * (Nq + 1.0) * math.tan(phi_rad)
    return Nc, Nq, Ng

def qult_meyerhof(B, D, phi, gamma_eff_at_base):
    Nc, Nq, Ng = bearing_factors(phi)
    sc, sq, sg = 1.3, 1.2, 1.0  # forma rectangular
    q_eff = gamma_eff_at_base * D  # sobrecarga efectiva a la cota de base
    return q_eff * Nq * sq + 0.5 * gamma_eff_at_base * B * Ng * sg

def qult_pred(gamma_eff_base, B, D, phi_val, L_over_B_val):
    if use_ml_qu and (st.session_state.QU_MODEL is not None) and SKLEARN_OK:
        X = pd.DataFrame([{
            "gamma_eff": gamma_eff_base, "B": B, "D": D, "phi": phi_val, "L_over_B": L_over_B_val
        }])
        return float(st.session_state.QU_MODEL.predict(X)[0])
    return qult_meyerhof(B, D, phi_val, gamma_eff_base)

def contact_pressures(N, B, L, ex=0.0, ey=0.0):
    qavg = N / (B * L)
    in_kern = (abs(ex) <= B/6) and (abs(ey) <= L/6)
    if in_kern:
        qmax = qavg * (1 + 6*abs(ex)/B + 6*abs(ey)/L)
        qserv = qavg
    else:
        Beff, Leff = B - 2*abs(ex), L - 2*abs(ey)
        if Beff <= 0 or Leff <= 0:
            return np.inf, np.inf
        qserv = N / (Beff * Leff)
        qmax  = 2 * N / (Beff * Leff)
    return qserv, qmax

def settlement_elastic_mm(qserv_kpa, B_m, Es_kpa, nu=0.30):
    return 1000.0 * (qserv_kpa * B_m * (1 - nu**2) / Es_kpa)

def Es_adjusted_for_NF(Es_kpa, soil_type, D, B, z_nf, z_base):
    """Reducción simplificada por NF alto:
       - Si NF está a menos de 0.5B por encima de la base: arenas 0.75, arcillas 0.65, mixto 0.7
       - Si NF a nivel de base o más arriba: arenas 0.7, arcillas 0.6, mixto 0.65
       - Caso contrario: 1.0
    """
    dz = z_nf - z_base  # positivo si NF está por encima de la base
    if dz >= 0:  # NF por encima de la base (peor escenario)
        f = {"Arena":0.70, "Arcilla":0.60, "Mixto":0.65}[soil_type]
    elif (-dz) <= 0.5*B:  # NF cercano por debajo (≤0.5B)
        f = {"Arena":0.75, "Arcilla":0.65, "Mixto":0.70}[soil_type]
    else:
        f = 1.0
    return max(Es_kpa * f, 1.0)

def settlement_pred_mm(B, L_over_B, D, qserv_kpa, SPT, d_wt, soil_type, Es_user_kpa):
    """Predicción de S: si hay modelo ML entrenado se usa; si no, se usa elástico con ajuste por NF."""
    if use_ml_s and (st.session_state.S_MODEL is not None) and SKLEARN_OK:
        X = pd.DataFrame([{
            "B": B,
            "L_over_B": L_over_B,
            "D_over_B": D / B,
            "q": qserv_kpa,   # neto aplicado (kPa)
            "SPT": SPT,
            "d": d_wt        # profundidad/ cota de NF (m) según tu base
        }])
        return float(st.session_state.S_MODEL.predict(X)[0])

    # Alternativa elástica con ajuste por NF
    Es_eff = Es_adjusted_for_NF(Es_user_kpa, soil_type, D, B, z_NF, z_base)
    return float(settlement_elastic_mm(qserv_kpa, B, Es_eff))

# ======================== Costeo ================================
def cost_S(B, L, h, c_conc=650.0, c_acero_kg=5.5, acero_kg_m3=60.0, c_exc=80.0, D=1.5):
    vol = B * L * h
    acero_kg = acero_kg_m3 * vol
    exc = B * L * D
    return vol * c_conc + acero_kg * c_acero_kg + exc * c_exc

# ======================== Recomendador de tipo de cimiento ======
def recommend_foundation_type(N_kN, qserv, qadm, s_mm, B, L, SPT, z_nf, explanations):
    """Motor experto simple y explicable."""
    rec = "Zapata aislada"
    why = []

    if qserv > 0.9*qadm or s_mm > 0.9*s_adm_mm:
        why.append("Demanda cercana al límite de servicio.")
    if SPT <= 10:
        why.append("SPT bajo (suelo blando a medio).")
    if z_nf >= z_base: # NF por encima o al nivel de base
        why.append("NF alto en la base o por encima.")

    area = B*L
    presion = N_kN / area

    # Reglas
    if (qserv <= 0.6*qadm) and (s_mm <= 0.6*s_adm_mm) and (SPT >= 20):
        rec = "Zapata aislada"
    elif (L_over_B >= 2.0) or (presion > 250):  # cargas lineales/muros o presión alta
        rec = "Zapata corrida"
    if (presion > 300 and qserv > 0.7*qadm) or (s_mm > 0.8*s_adm_mm and SPT < 15):
        rec = "Zapata combinada o Losa"
    if (SPT < 10 and (qserv > 0.7*qadm or s_mm > 0.8*s_adm_mm)) or (z_nf >= z_base and presion > 250):
        rec = "Pilotes (o micropilotes) + viga de coronación"

    # Ajustes por NF muy alto
    if z_nf >= z_base and rec == "Zapata aislada":
        rec = "Zapata combinada o Losa"
        why.append("NF alto desfavorece zapatas aisladas en suelos blandos.")

    explanations.extend(why)
    return rec

# ======================== Optimización ==========================
st.markdown("### Parámetros adicionales para asentamiento elástico")
Es_user = st.number_input("Eₛ de referencia (kPa) — solo si NO usas ML de S", min_value=3000.0, max_value=150000.0, value=25000.0, step=500.0)
SPT_user = st.number_input("SPT promedio (golpes/0.3 m) — usado por ML S o para reglas", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

if st.button("🚀 Optimizar (FO1 & FO2)", use_container_width=True):
    Bs = np.linspace(Bmin, Bmax, int(nB))
    hs = np.linspace(hmin, hmax, int(nh))

    rows = []
    for B in Bs:
        L = L_over_B * B
        # gamma_eff en la cota de base:
        gamma_eff_base = gamma_effective_at_depth(gamma_dry, gamma_sat, z_base, z_NF)

        qu = qult_pred(gamma_eff_base, B, D, phi, L_over_B)
        qadm = qu / FS
        qserv, qmax = contact_pressures(N, B, L)
        if not (qserv <= qadm and qmax <= qadm):
            continue

        # d_wt para el modelo de S: siguiendo el paper, usan "d" (profundidad/cota NF). Aquí pasamos z_NF (m)
        s_mm = settlement_pred_mm(B, L_over_B, D, qserv, SPT_user, z_NF, soil_type, Es_user)
        if s_mm > s_adm_mm:
            continue

        costo = cost_S(B, L, h, D=D)
        rows.append([B, L, h, qu, qadm, qserv, qmax, s_mm, costo])

    if not rows:
        st.error("Sin soluciones factibles con los parámetros dados.")
        st.stop()

    df = pd.DataFrame(rows, columns=["B","L","h","qu","qadm","qserv","qmax","s_mm","costo"])

    # FO1: costo mínimo
    fo1 = df.loc[df["costo"].idxmin()]
    # FO2: asentamiento mínimo
    fo2 = df.loc[df["s_mm"].idxmin()]

    # Banners de modelos usados
    if use_ml_qu and (st.session_state.QU_MODEL is not None) and SKLEARN_OK:
        mq = st.session_state.QU_METRICS
        st.success(f"Capacidad (qu) por **ML (GB, 10-fold)** | R²={mq['r2']:.3f} · RMSE≈{mq['rmse']:.2f} kPa · MAE≈{mq['mae']:.2f} kPa")
    else:
        st.success("Capacidad (qu) por **Meyerhof (clásico)**")

    if use_ml_s and (st.session_state.S_MODEL is not None) and SKLEARN_OK:
        ms = st.session_state.S_METRICS
        st.success(f"Asentamiento por **ML (GB, 10-fold)** | R²={ms['r2']:.3f} · RMSE≈{ms['rmse']:.2f} mm · MAE≈{ms['mae']:.2f} mm  (modelo inspirado en el paper).")
    else:
        st.info("Asentamiento por **modelo elástico** con ajuste por NF y tipo de suelo.")

    cA, cB = st.columns(2)
    with cA:
        st.subheader("FO1 · Mínimo costo")
        st.table(fo1[["B","L","h","qserv","qadm","qmax","s_mm","costo"]])
    with cB:
        st.subheader("FO2 · Mínimo asentamiento")
        st.table(fo2[["B","L","h","qserv","qadm","qmax","s_mm","costo"]])

    # Recomendación automática entre FO1/FO2
    def elegir(fo1, fo2):
        s1, s2 = fo1.s_mm, fo2.s_mm
        c1, c2 = fo1.costo, fo2.costo
        if (s1 - s2 >= 5.0) and (c2 <= 1.05*c1):
            return fo2, "FO2 (mínimo asentamiento)", f"Menor s (−{s1 - s2:.1f} mm) con costo ≤ +5%."
        return fo1, "FO1 (mínimo costo)", "El ahorro de costo domina y las verificaciones se cumplen."
    chosen, tag, why = elegir(fo1, fo2)

    st.markdown("## ✅ Recomendación geométrica")
    st.write(
        f"**{tag}**  \n"
        f"B = **{chosen.B:.2f} m**, L = **{chosen.L:.2f} m**, h = **{chosen.h:.2f} m**  \n"
        f"q_serv = **{chosen.qserv:.1f} kPa** ≤ q_adm = **{chosen.qadm:.1f} kPa**; "
        f"q_max = **{chosen.qmax:.1f} kPa** ≤ q_adm  \n"
        f"s = **{chosen.s_mm:.1f} mm** ≤ s_adm = **{s_adm_mm:.0f} mm**  \n"
        f"Costo ≈ **S/ {chosen.costo:,.2f}**  \n"
        f"**Motivo:** {why}"
    )

    # Recomendación de tipo de cimiento (motor experto “IA”)
    expl = []
    tipo = recommend_foundation_type(N, chosen.qserv, chosen.qadm, chosen.s_mm, chosen.B, chosen.L, SPT_user, z_NF, expl)
    st.markdown("## 🧠 Recomendación de tipo de cimiento (IA explicable)")
    st.write(f"**Sugerido:** **{tipo}**")
    if expl:
        st.caption("Razones: " + " · ".join(expl))

    st.download_button(
        "Descargar soluciones (CSV)",
        df.to_csv(index=False),
        "soluciones_minimal.csv",
        "text/csv",
        use_container_width=True,
    )




