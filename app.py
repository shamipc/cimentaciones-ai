# app.py — Optimización de Cimentaciones (mínima) con ML opcional y consideraciones NF/SPT
# FO1: minimizar costo; FO2: minimizar asentamiento
# Verifica: q_serv ≤ q_adm, q_max ≤ q_adm y s ≤ s_adm

import math
import numpy as np
import pandas as pd
import streamlit as st

# ========= ML opcional (importación defensiva) =========
try:
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False
    train_test_split = GridSearchCV = GradientBoostingRegressor = None
    mean_squared_error = mean_absolute_error = r2_score = None

st.set_page_config(page_title="Optimización de Cimentaciones — Minimal", layout="centered")
st.title("Optimización de Cimentaciones (mínima)")
st.caption("Entradas mínimas + 2 funciones objetivo (FO1 costo, FO2 asentamiento) con verificación de servicio. "
           "ML opcional para predecir qu (paper-like).")

# ======================== Entradas ========================
c1, c2 = st.columns(2)
with c1:
    N = st.number_input("Carga axial N (kN)", 100.0, 2e5, 800.0, 10.0)
    phi = st.number_input("ϕ (°)", 0.0, 45.0, 32.0, 0.5)
    gamma_input = st.number_input("γ natural (kN/m³)", 10.0, 24.0, 18.0, 0.1)
    Es = st.number_input("Eₛ (kPa) para s (aprox.)", 3000.0, 100000.0, 25000.0, 500.0)

with c2:
    D = st.number_input("Profundidad de base D (m)", 0.5, 6.0, 1.5, 0.1)
    FS = st.number_input("FS (capacidad)", 1.5, 4.0, 2.5, 0.1)
    L_over_B = st.number_input("Relación L/B (constante)", 0.8, 2.0, 1.0, 0.05)
    s_adm_mm = st.number_input("Asentamiento admisible sₐ (mm)", 5.0, 50.0, 25.0, 1.0)

st.markdown("**Condiciones hidrogeotécnicas (opcionales)**")
c3, c4, c5 = st.columns(3)
with c3:
    z_NF = st.number_input("Profundidad del nivel freático z_NF (m)  (0 = en terreno)", 0.0, 10.0, 2.0, 0.1)
with c4:
    suelo_tipo = st.selectbox("Tipo de suelo predominante", ["Arena (no cohesiva)", "Arcilla (cohesiva)"])
with c5:
    N60 = st.number_input("SPT N₆₀ (solo ajusta si es arena; 0 = ignorar)", 0.0, 60.0, 0.0, 1.0)

# γ efectivo en la base (efecto del NF simple)
gamma_eff = gamma_input if z_NF > D else max(gamma_input - 9.81, 5.0)

st.markdown("**Rangos de búsqueda (B y h)**")
c6, c7, c8 = st.columns(3)
with c6:
    Bmin = st.number_input("B min (m)", 0.8, 6.0, 1.0, 0.1)
    Bmax = st.number_input("B max (m)", 0.8, 6.0, 3.0, 0.1)
with c7:
    hmin = st.number_input("h min (m)", 0.3, 2.0, 0.5, 0.05)
    hmax = st.number_input("h max (m)", 0.3, 2.0, 1.2, 0.05)
with c8:
    nB = st.number_input("Puntos B", 5, 80, 30, 1)
    nh = st.number_input("Puntos h", 3, 60, 12, 1)

st.markdown("---")

# ======================== ML opcional (paper) ========================
st.subheader("ML opcional (paper) — auto-ML")
if not SKLEARN_OK:
    st.warning("Para usar ML, agrega `scikit-learn>=1.3,<1.5` en requirements.txt. "
               "Si no, se usará el método clásico (Meyerhof).")

# Estados globales
if "ML_MODEL" not in st.session_state:
    st.session_state.ML_MODEL = None
if "RMSE_ML" not in st.session_state:
    st.session_state.RMSE_ML = None
if "ML_METRICS" not in st.session_state:
    st.session_state.ML_METRICS = {}

up = st.file_uploader("CSV entrenamiento (gamma_eff,B,D,phi,L_over_B,qu)  — también acepta 'gamma' en vez de 'gamma_eff'", type=["csv"])
use_ml = st.toggle("Usar modelo ML si está entrenado", value=False)

def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    dfn = df.copy()
    dfn.columns = [c.strip().lower() for c in dfn.columns]
    return dfn

def train_ml(csv):
    df_raw = pd.read_csv(csv)
    df = _standardize_cols(df_raw)

    # Acepta gamma_eff o gamma
    has_gamma_eff = "gamma_eff" in df.columns
    has_gamma     = "gamma"     in df.columns

    req_base = ["b", "d", "phi", "l_over_b", "qu"]
    missing = [c for c in req_base if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}. Presentes: {list(df.columns)}")
    if not (has_gamma_eff or has_gamma):
        raise ValueError("Falta 'gamma_eff' o 'gamma' en el CSV de entrenamiento.")
    if not has_gamma_eff and has_gamma:
        df["gamma_eff"] = df["gamma"]

    X = df[["gamma_eff", "b", "d", "phi", "l_over_b"]].astype(float)
    y = df["qu"].astype(float)

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.20, random_state=42)

    gbr = GradientBoostingRegressor(random_state=42)
    grid = GridSearchCV(
        gbr,
        {
            "n_estimators": [150, 300, 500],
            "max_depth": [2, 3],
            "learning_rate": [0.05, 0.1],
            "min_samples_leaf": [3, 5],
        },
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_

    # ===== Métricas de validación (compatibles) =====
    yhat = best.predict(Xva)

    # RMSE sin usar 'squared=False' (para evitar errores en versiones antiguas)
    try:
        mse  = float(mean_squared_error(yva, yhat))
        rmse = float(np.sqrt(mse))
    except Exception:
        rmse = float(np.sqrt(np.mean((yva - yhat) ** 2)))

    try:
        mae = float(mean_absolute_error(yva, yhat))
    except Exception:
        mae = float(np.mean(np.abs(yva - yhat)))

    try:
        r2 = float(r2_score(yva, yhat))
    except Exception:
        r2 = float("nan")

    med = float(np.median(yva)) if len(yva) else float("nan")
    nrmse_med = float(rmse / med * 100) if np.isfinite(med) and med != 0 else float("nan")
    bias = float(np.mean(yhat - yva))

    st.session_state.ML_METRICS = {
        "rmse": rmse, "mae": mae, "r2": r2,
        "nrmse_med": nrmse_med, "bias": bias, "n_val": int(len(yva)),
    }
    return best, rmse

if st.button("Entrenar modelo", use_container_width=True):
    if not SKLEARN_OK:
        st.error("scikit-learn no está disponible en el entorno.")
    elif up is None:
        st.warning("Sube un CSV válido con columnas: gamma_eff,B,D,phi,L_over_B,qu (o usa 'gamma').")
    else:
        try:
            model, rmse = train_ml(up)
            st.session_state.ML_MODEL = model
            st.session_state.RMSE_ML = rmse
            st.success(f"Modelo entrenado. RMSE≈ {rmse:,.2f} kPa")
        except Exception as e:
            st.error(f"Error entrenando: {e}")

# Panel de métricas tras entrenar
if st.session_state.ML_METRICS:
    m = st.session_state.ML_METRICS
    st.markdown("### Métricas del modelo (validación)")
    cA, cB, cC, cD, cE = st.columns(5)
    cA.metric("R²", f"{m['r2']:.3f}" if np.isfinite(m['r2']) else "—")
    cB.metric("RMSE", f"{m['rmse']:.2f} kPa")
    cC.metric("MAE", f"{m['mae']:.2f} kPa")
    cD.metric("nRMSE/Mediana", f"{m['nrmse_med']:.1f}%" if np.isfinite(m['nrmse_med']) else "—")
    cE.metric("Sesgo", f"{m['bias']:+.2f} kPa")
    st.caption(f"Conjunto de validación: n = {m['n_val']}")

st.markdown("---")

# ======================== Capacidad última ========================
def bearing_factors(phi_deg: float):
    phi_rad = math.radians(phi_deg)
    if phi_rad < 1e-6:
        Nq, Nc, Ng = 1.0, 5.14, 0.0
    else:
        Nq = math.e ** (math.pi * math.tan(phi_rad)) * (math.tan(math.pi / 4 + phi_rad / 2)) ** 2
        Nc = (Nq - 1.0) / math.tan(phi_rad)
        Ng = 2.0 * (Nq + 1.0) * math.tan(phi_rad)
    return Nc, Nq, Ng

def qult_meyerhof(B, D, phi, gamma_eff_local):
    # c≈0 (friccionantes); coef. de forma rectangulares
    Nc, Nq, Ng = bearing_factors(phi)
    sc, sq, sg = 1.3, 1.2, 1.0
    q_eff = gamma_eff_local * D
    return q_eff * Nq * sq + 0.5 * gamma_eff_local * B * Ng * sg

def spt_factor_arena(N60_val: float) -> float:
    """Factor suave por SPT en arenas (opcional). 0 → 1.0 (sin ajuste)."""
    if N60_val <= 0:
        return 1.0
    # Escala moderada: 0.6 a 1.6 aprox. (cap en 6–50)
    return float(np.clip(N60_val / 25.0, 0.6, 1.6))

def qult_pred(gamma_eff_local, B, D, phi_val, L_over_B_val):
    """Predicción de qu: ML (si está activo) o Meyerhof."""
    if use_ml and (st.session_state.ML_MODEL is not None) and SKLEARN_OK:
        X = pd.DataFrame([{
            "gamma_eff": gamma_eff_local, "B": B, "D": D, "phi": phi_val, "L_over_B": L_over_B_val
        }])
        return float(st.session_state.ML_MODEL.predict(X)[0])
    # Clásico
    qu = qult_meyerhof(B, D, phi_val, gamma_eff_local)
    # Ajuste SPT sólo si tipo de suelo es arena
    if suelo_tipo.startswith("Arena"):
        qu *= spt_factor_arena(N60)
    return qu

# ======================== Servicio y asentamiento ================
def contact_pressures(NkN, B, L, ex=0.0, ey=0.0):
    qavg = NkN / (B * L)
    in_kern = (abs(ex) <= B/6) and (abs(ey) <= L/6)
    if in_kern:
        qmax = qavg * (1 + 6*abs(ex)/B + 6*abs(ey)/L)
        qserv = qavg
    else:
        Beff, Leff = B - 2*abs(ex), L - 2*abs(ey)
        if Beff <= 0 or Leff <= 0:
            return np.inf, np.inf
        qserv = NkN / (Beff * Leff)
        qmax  = 2 * NkN / (Beff * Leff)
    return qserv, qmax

def settlement_mm(qserv_kpa, B_m, Es_kpa, nu=0.30, z_nf=10.0, D_base=1.5):
    """Modelo elástico simple: s ≈ q·B·(1-ν²)/Es en mm.
       Penaliza saturación si NF cerca o por debajo de la base."""
    s = 1000.0 * (qserv_kpa * B_m * (1 - nu**2) / Es_kpa)
    # Si NF está en la base o por encima (z_NF ≤ D), aumentamos asentamiento (20%)
    if z_nf <= D_base:
        s *= 1.20
    return s

# ======================== Costo ================================
def cost_S(B, L, h, c_conc=650.0, c_acero_kg=5.5, acero_kg_m3=60.0, c_exc=80.0, D_base=1.5):
    """Costo simple (S/): concreto + acero + excavación."""
    vol = B * L * h
    acero_kg = acero_kg_m3 * vol
    exc = B * L * D_base
    return vol * c_conc + acero_kg * c_acero_kg + exc * c_exc

# ======================== Corrida principal ====================
if st.button("🚀 Optimizar (FO1 & FO2)"):
    Bs = np.linspace(Bmin, Bmax, int(nB))
    hs = np.linspace(hmin, hmax, int(nh))

    rows = []
    for B in Bs:
        L = L_over_B * B
        for h in hs:
            qu = qult_pred(gamma_eff, B, D, phi, L_over_B)
            qadm = qu / FS
            qserv, qmax = contact_pressures(N, B, L)
            if not (qserv <= qadm and qmax <= qadm):
                continue
            s = settlement_mm(qserv, B, Es, z_nf=z_NF, D_base=D)
            if s > s_adm_mm:
                continue
            costo = cost_S(B, L, h, D_base=D)
            rows.append([B, L, h, qu, qadm, qserv, qmax, s, costo])

    if not rows:
        st.error("Sin soluciones factibles con los parámetros dados.")
        st.stop()

    df = pd.DataFrame(rows, columns=["B","L","h","qu","qadm","qserv","qmax","s_mm","costo"])

    # FO1: costo mínimo
    fo1 = df.loc[df["costo"].idxmin()]
    # FO2: asentamiento mínimo
    fo2 = df.loc[df["s_mm"].idxmin()]

    # Banner del modelo usado
    if use_ml and (st.session_state.ML_MODEL is not None) and SKLEARN_OK:
        if st.session_state.ML_METRICS:
            m = st.session_state.ML_METRICS
            st.success(
                "Modelo de capacidad usado: **ML (paper)** — "
                f"RMSE≈ {m['rmse']:.2f} kPa | MAE≈ {m['mae']:.2f} kPa | "
                f"R²≈ {m['r2']:.3f} | nRMSE≈ {m['nrmse_med']:.1f}% | "
                f"Sesgo≈ {m['bias']:+.2f} kPa"
            )
        else:
            st.success(f"Modelo de capacidad usado: **ML (paper)** — RMSE≈ {st.session_state.RMSE_ML:,.2f} kPa")
    else:
        st.success("Modelo de capacidad usado: **Meyerhof (clásico)**")

    cA, cB = st.columns(2)
    with cA:
        st.subheader("FO1 · Mínimo costo")
        st.table(fo1[["B","L","h","qserv","qadm","qmax","s_mm","costo"]])
    with cB:
        st.subheader("FO2 · Mínimo asentamiento")
        st.table(fo2[["B","L","h","qserv","qadm","qmax","s_mm","costo"]])

    # Recomendación
    def recomendar(fo1, fo2):
        s1, s2 = fo1.s_mm, fo2.s_mm
        c1, c2 = fo1.costo, fo2.costo
        if (s1 - s2 >= 5.0) and (c2 <= 1.05 * c1):
            tag = "FO2 (mínimo asentamiento)"
            return fo2, tag, f"Se elige FO2 por menor asentamiento (−{s1 - s2:.1f} mm) con costo ≤ +5%."
        tag = "FO1 (mínimo costo)"
        return fo1, tag, "Se elige FO1: el ahorro de costo domina y las verificaciones se cumplen."
    chosen, tag, why = recomendar(fo1, fo2)

    st.markdown("## ✅ Recomendación")
    st.write(
        f"**{tag}**  \n"
        f"B = **{chosen.B:.2f} m**, L = **{chosen.L:.2f} m**, h = **{chosen.h:.2f} m**  \n"
        f"q_serv = **{chosen.qserv:.1f} kPa** ≤ q_adm = **{chosen.qadm:.1f} kPa**; "
        f"q_max = **{chosen.qmax:.1f} kPa** ≤ q_adm  \n"
        f"s = **{chosen.s_mm:.1f} mm** ≤ s_adm = **{s_adm_mm:.0f} mm**  \n"
        f"Costo ≈ **S/ {chosen.costo:,.2f}**  \n"
        f"**Motivo:** {why}"
    )

    st.download_button(
        "Descargar soluciones (CSV)",
        df.to_csv(index=False),
        "soluciones_minimal.csv",
        "text/csv",
        use_container_width=True,
    )
