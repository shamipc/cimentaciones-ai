# ===============================================================
# app.py — Optimización de Cimentaciones con ML (qu + asentamiento)
# ===============================================================

import math
import numpy as np
import pandas as pd
import streamlit as st

# ============================= ML defensivo =============================
try:
    from sklearn.model_selection import train_test_split, GridSearchCV, KFold
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


st.set_page_config(page_title="Optimización de Cimentaciones con ML", layout="centered")
st.title("Optimización de Cimentaciones (ML para q₍ult₎ y asentamiento)")
st.caption("Incluye predicción ML optimizada para capacidad última y asentamiento basado en el paper 2024.")

# ========================================================================
# ESTADOS GLOBALES
# ========================================================================
if "ML_MODEL" not in st.session_state:
    st.session_state.ML_MODEL = None
if "R2_ML" not in st.session_state:
    st.session_state.R2_ML = None

if "ML_S_MODEL" not in st.session_state:
    st.session_state.ML_S_MODEL = None
if "R2_S" not in st.session_state:
    st.session_state.R2_S = None


# ========================================================================
# ENTRADAS
# ========================================================================
c1, c2 = st.columns(2)
with c1:
    N = st.number_input("Carga axial N (kN)", 100.0, 200000.0, 800.0, 10.0)
    phi = st.number_input("ϕ (°)", 0.0, 45.0, 32.0, 0.5)
    gamma = st.number_input("γ (kN/m³)", 10.0, 24.0, 18.0, 0.1)
    Es = st.number_input("Eₛ (kPa) si NO se usa ML", 3000.0, 100000.0, 25000.0, 500.0)

with c2:
    D = st.number_input("Profundidad D (m)", 0.5, 6.0, 1.5, 0.1)
    FS = st.number_input("Factor de seguridad (FS)", 1.5, 4.0, 2.5, 0.1)
    L_over_B = st.number_input("Relación L/B", 0.8, 2.0, 1.0, 0.05)
    s_adm_mm = st.number_input("Asentamiento admisible (mm)", 5.0, 50.0, 25.0, 1.0)

st.markdown("### Parámetro adicional para asentamiento ML")
SPT = st.number_input("SPT promedio (N)", 1, 80, 25)

st.markdown("---")
st.markdown("### Rango de búsqueda B y h")

c3, c4, c5 = st.columns(3)
with c3:
    Bmin = st.number_input("B min (m)", 0.5, 6.0, 1.0, 0.1)
    Bmax = st.number_input("B max (m)", 0.5, 6.0, 3.0, 0.1)

with c4:
    hmin = st.number_input("h min (m)", 0.3, 2.0, 0.5, 0.05)
    hmax = st.number_input("h max (m)", 0.3, 2.0, 1.2, 0.05)

with c5:
    nB = st.number_input("Puntos B", 5, 80, 30, 1)
    nh = st.number_input("Puntos h", 3, 60, 12, 1)


# ========================================================================
# =========================== ML PARA q_ULTIMO ===========================
# ========================================================================
st.subheader("Modelo ML optimizado para q₍ult₎")

up = st.file_uploader("CSV (gamma,B,D,phi,L_over_B,qu)", type=["csv"])
use_ml = st.toggle("Usar ML para q₍ult₎", value=False)


def train_ml_qu(csv):
    df = pd.read_csv(csv)
    req = ["gamma", "B", "D", "phi", "L_over_B", "qu"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan columnas: {miss}")

    X = df[["gamma", "B", "D", "phi", "L_over_B"]]
    y = df["qu"].astype(float)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor())
    ])

    param_grid = {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__min_samples_leaf": [3, 5, 8]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        n_jobs=-1,
        scoring="r2"
    )

    grid.fit(X, y)

    return grid.best_estimator_, grid.best_score_


if st.button("Entrenar modelo q₍ult₎"):
    if not SKLEARN_OK:
        st.error("scikit-learn no disponible.")
    elif up is None:
        st.warning("Sube un archivo CSV válido.")
    else:
        try:
            model, r2 = train_ml_qu(up)
            st.session_state.ML_MODEL = model
            st.session_state.R2_ML = r2
            st.success(f"Modelo q₍ult₎ entrenado correctamente. R² = {r2:.3f}")
        except Exception as e:
            st.error(str(e))


# ========================================================================
# ====================== ML PARA ASENTAMIENTO ===========================
# ========================================================================
st.subheader("Modelo ML para asentamiento (según paper 2024)")

up_s = st.file_uploader("CSV (SPT,B,D,q,s_mm)", type=["csv"])
use_ml_s = st.toggle("Usar ML para asentamiento", value=False)


def train_ml_settlement(csv):
    df = pd.read_csv(csv)
    req = ["SPT", "B", "D", "q", "s_mm"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan columnas: {miss}")

    df["Df_over_B"] = df["D"] / df["B"]

    X = df[["SPT", "B", "Df_over_B", "q"]]
    y = df["s_mm"].astype(float)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor())
    ])

    param_grid = {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__min_samples_leaf": [3, 5, 8]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        n_jobs=-1,
        scoring="r2"
    )

    grid.fit(X, y)

    return grid.best_estimator_, grid.best_score_


if st.button("Entrenar modelo asentamiento"):
    if not SKLEARN_OK:
        st.error("scikit-learn no disponible.")
    elif up_s is None:
        st.warning("Sube un archivo CSV válido.")
    else:
        try:
            model, r2 = train_ml_settlement(up_s)
            st.session_state.ML_S_MODEL = model
            st.session_state.R2_S = r2
            st.success(f"Modelo asentamiento entrenado. R² = {r2:.3f}")
        except Exception as e:
            st.error(str(e))


# ========================================================================
# =============== MODELOS CLÁSICOS COMO RESPALDO ========================
# ========================================================================

def bearing_factors(phi_deg):
    phi_rad = math.radians(phi_deg)
    if phi_rad < 1e-6:
        return 5.14, 1.0, 0.0
    Nq = math.e ** (math.pi * math.tan(phi_rad)) * (math.tan(math.pi/4 + phi_rad/2))**2
    Nc = (Nq - 1.0) / math.tan(phi_rad)
    Ng = 2.0 * (Nq + 1.0) * math.tan(phi_rad)
    return Nc, Nq, Ng


def qult_meyerhof(B, D, phi, gamma):
    Nc, Nq, Ng = bearing_factors(phi)
    q_eff = gamma * D
    return q_eff * Nq + 0.5 * gamma * B * Ng


def qult_pred(gamma, B, D, phi, L_over_B):
    if use_ml and st.session_state.ML_MODEL is not None:
        X = pd.DataFrame([{
            "gamma": gamma,
            "B": B,
            "D": D,
            "phi": phi,
            "L_over_B": L_over_B
        }])
        return float(st.session_state.ML_MODEL.predict(X)[0])
    return qult_meyerhof(B, D, phi, gamma)


def settlement_elastic(qserv, B, Es, nu=0.30):
    return 1000 * (qserv * B * (1 - nu**2) / Es)


def settlement_pred(SPT, B, D, qserv):
    if use_ml_s and st.session_state.ML_S_MODEL is not None:
        X = pd.DataFrame([{
            "SPT": SPT,
            "B": B,
            "Df_over_B": D / B,
            "q": qserv
        }])
        return float(st.session_state.ML_S_MODEL.predict(X)[0])
    return settlement_elastic(qserv, B, Es)


def contact_pressures(N, B, L):
    qavg = N / (B * L)
    return qavg, qavg


# ========================================================================
# =========================== OPTIMIZACIÓN ===============================
# ========================================================================
if st.button("🚀 Optimizar"):

    Bs = np.linspace(Bmin, Bmax, int(nB))
    hs = np.linspace(hmin, hmax, int(nh))

    rows = []

    for B in Bs:
        L = L_over_B * B
        for h in hs:

            # Capacidad última
            qu = qult_pred(gamma, B, D, phi, L_over_B)
            qadm = qu / FS

            qserv, qmax = contact_pressures(N, B, L)

            if qserv > qadm or qmax > qadm:
                continue

            # Asentamiento
            s = settlement_pred(SPT, B, D, qserv)
            if s > s_adm_mm:
                continue

            # Costo simple
            vol = B * L * h
            costo = (
                vol * 650 +
                vol * 60 * 5.5 +
                (B * L * D) * 80
            )

            rows.append([B, L, h, qu, qadm, qserv, qmax, s, costo])

    if not rows:
        st.error("No se encontraron soluciones factibles.")
        st.stop()

    df = pd.DataFrame(rows, columns=["B","L","h","qu","qadm","qserv","qmax","s_mm","costo"])

    fo1 = df.loc[df["costo"].idxmin()]
    fo2 = df.loc[df["s_mm"].idxmin()]

    st.success("Optimización realizada correctamente.")

    cA, cB = st.columns(2)

    with cA:
        st.subheader("FO1 — Mínimo costo")
        st.table(fo1[["B","L","h","qserv","qadm","qmax","s_mm","costo"]])

    with cB:
        st.subheader("FO2 — Mínimo asentamiento")
        st.table(fo2[["B","L","h","qserv","qadm","qmax","s_mm","costo"]])

    def recomendar(fo1, fo2):
        if (fo1.s_mm - fo2.s_mm >= 5) and (fo2.costo <= 1.05 * fo1.costo):
            return fo2, "FO2 (mínimo asentamiento)", "Menor asentamiento con costo razonable"
        return fo1, "FO1 (mínimo costo)", "El menor costo domina y cumple verificaciones"

    chosen, tag, why = recomendar(fo1, fo2)

    st.markdown(f"""
    ## ✅ Recomendación de Diseño  
    **{tag}**  
    - B = **{chosen.B:.2f} m**  
    - L = **{chosen.L:.2f} m**  
    - h = **{chosen.h:.2f} m**  
    - Asentamiento = **{chosen.s_mm:.2f} mm**  
    - Costo = **S/ {chosen.costo:,.2f}**  

    **Motivo:** {why}
    """)

    st.download_button(
        "Descargar soluciones (CSV)",
        df.to_csv(index=False),
        "soluciones.csv",
        "text/csv"
    )
