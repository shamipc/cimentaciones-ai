# ===============================================================
# app.py — Optimización de Cimentaciones con ML (qu + asentamiento)
# Versión: asentamiento igual al paper (SPT, B, Df/B, q) + mejoras de predicción
# Sin q_max (carga centrada → presión uniforme)
# + Gráficas Measured vs Calculated (ML) con Streamlit
# ===============================================================

import math
import numpy as np
import pandas as pd
import streamlit as st

# ============================= ML defensivo =============================
try:
    from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


st.set_page_config(page_title="Optimización de Cimentaciones con ML", layout="centered")
st.title("Optimización de Cimentaciones (ML para q₍ult₎ y asentamiento)")
st.caption("q₍ult₎ con ML + asentamiento según paper 2024 (SPT, B, Df/B, q).")

# ========================================================================
# ESTADOS GLOBALES
# ========================================================================
if "ML_MODEL" not in st.session_state:
    st.session_state.ML_MODEL = None
if "R2_ML" not in st.session_state:
    st.session_state.R2_ML = None
if "DF_QU" not in st.session_state:
    st.session_state.DF_QU = None

if "ML_S_MODEL" not in st.session_state:
    st.session_state.ML_S_MODEL = None
if "R2_S" not in st.session_state:
    st.session_state.R2_S = None
if "DF_S" not in st.session_state:
    st.session_state.DF_S = None

# ========================================================================
# ENTRADAS PRINCIPALES
# ========================================================================
c1, c2 = st.columns(2)
with c1:
    N = st.number_input("Carga axial N (kN)", 100.0, 200000.0, 800.0, 10.0)
    phi = st.number_input("ϕ (°)", 0.0, 45.0, 32.0, 0.5)
    gamma = st.number_input("γ (kN/m³)", 10.0, 24.0, 18.0, 0.1)
    Es = st.number_input("Eₛ (kPa) si NO se usa ML (asentamiento clásico)", 3000.0, 100000.0, 25000.0, 500.0)

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
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn no disponible.")
    df = pd.read_csv(csv)
    req = ["gamma", "B", "D", "phi", "L_over_B", "qu"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan columnas: {miss}")

    X = df[["gamma", "B", "D", "phi", "L_over_B"]]
    y = df["qu"].astype(float)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [300, 600, 900],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__min_samples_leaf": [1, 3, 5],
        "model__subsample": [0.7, 0.9, 1.0],
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

    return grid.best_estimator_, grid.best_score_, df


if st.button("Entrenar modelo q₍ult₎"):
    if not SKLEARN_OK:
        st.error("scikit-learn no disponible.")
    elif up is None:
        st.warning("Sube un archivo CSV válido.")
    else:
        try:
            model, r2, df_qu = train_ml_qu(up)
            st.session_state.ML_MODEL = model
            st.session_state.R2_ML = r2
            st.session_state.DF_QU = df_qu  # guardar datos para las gráficas
            st.success(f"Modelo q₍ult₎ entrenado. R² (CV) = {r2:.3f}")
        except Exception as e:
            st.error(str(e))

# ========================================================================
# ====================== ML PARA ASENTAMIENTO (PAPER) ====================
# ========================================================================
st.subheader("Modelo ML para asentamiento (según paper 2024)")

# CSV con columnas exactamente: SPT,B,Df_over_B,q,s_mm
up_s = st.file_uploader("CSV asentamiento (SPT,B,Df_over_B,q,s_mm)", type=["csv"])
use_ml_s = st.toggle("Usar ML para asentamiento", value=False)


def train_ml_settlement(csv):
    """
    Entrena el modelo de asentamiento como en el paper:
    variables: SPT, B, Df/B, q → objetivo: s_mm (trabajado en log1p).
    Devuelve el modelo y el R² en la escala original de s_mm.
    """
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn no disponible.")
    df = pd.read_csv(csv)

    req = ["SPT", "B", "Df_over_B", "q", "s_mm"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan columnas: {miss}")

    X = df[["SPT", "B", "Df_over_B", "q"]]
    y = df["s_mm"].astype(float)
    y_log = np.log1p(y)  # objetivo en log(1 + s)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [300, 600, 900],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__min_samples_leaf": [1, 3, 5],
        "model__subsample": [0.7, 0.9, 1.0],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        n_jobs=-1,
        scoring="r2"  # R² en espacio log
    )

    grid.fit(X, y_log)

    best = grid.best_estimator_

    # R² en escala original usando predicción cruzada
    y_log_cv = cross_val_predict(best, X, y_log, cv=cv)
    y_cv = np.expm1(y_log_cv)
    r2_orig = r2_score(y, y_cv)

    return best, r2_orig, df


if st.button("Entrenar modelo asentamiento"):
    if not SKLEARN_OK:
        st.error("scikit-learn no disponible.")
    elif up_s is None:
        st.warning("Sube un archivo CSV con columnas: SPT,B,Df_over_B,q,s_mm.")
    else:
        try:
            model, r2, df_s = train_ml_settlement(up_s)
            st.session_state.ML_S_MODEL = model
            st.session_state.R2_S = r2
            st.session_state.DF_S = df_s  # guardar datos para las gráficas
            st.success(f"Modelo asentamiento entrenado. R² (escala original) = {r2:.3f}")
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
    if use_ml and st.session_state.ML_MODEL is not None and SKLEARN_OK:
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
    """Asentamiento elástico clásico (backup)."""
    return 1000 * (qserv * B * (1 - nu**2) / Es)


def settlement_pred(SPT_val, B, D, qserv):
    """
    Predicción de asentamiento:
    - Si ML está activo: usa SPT, B, Df_over_B=D/B, q → igual que el paper.
    - Si no: usa el modelo elástico clásico.
    """
    if use_ml_s and st.session_state.ML_S_MODEL is not None and SKLEARN_OK:
        df_over_B = D / B
        X = pd.DataFrame([{
            "SPT": SPT_val,
            "B": B,
            "Df_over_B": df_over_B,
            "q": qserv
        }])
        y_log_pred = st.session_state.ML_S_MODEL.predict(X)[0]
        return float(np.expm1(y_log_pred))  # volver a mm

    return settlement_elastic(qserv, B, Es)


def contact_pressures(N, B, L):
    """Presión de servicio promedio (carga centrada, sin momentos)."""
    return N / (B * L)

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

            # Presión de servicio (media)
            qserv = contact_pressures(N, B, L)

            # Verificación de capacidad portante
            if qserv > qadm:
                continue

            # Asentamiento (ML o clásico)
            s = settlement_pred(SPT, B, D, qserv)
            if s > s_adm_mm:
                continue

            # Costo simple (valores aproximados basados en Revista Costos)
            vol = B * L * h
            costo = (
                vol * 379 +          # concreto ≈ S/ 379 / m³
                vol * 60 * 5.5 +     # acero (60 kg/m³, ≈ S/ 5.5 / kg)
                (B * L * D) * 80     # excavación ≈ S/ 80 / m³
            )

            rows.append([B, L, h, qu, qadm, qserv, s, costo])

    if not rows:
        st.error("No se encontraron soluciones factibles.")
        st.stop()

    df = pd.DataFrame(rows, columns=["B", "L", "h", "qu", "qadm", "qserv", "s_mm", "costo"])

    fo1 = df.loc[df["costo"].idxmin()]
    fo2 = df.loc[df["s_mm"].idxmin()]

    st.success("Optimización realizada correctamente.")

    cA, cB = st.columns(2)

    with cA:
        st.subheader("FO1 — Mínimo costo")
        st.table(fo1[["B", "L", "h", "qserv", "qadm", "s_mm", "costo"]])

    with cB:
        st.subheader("FO2 — Mínimo asentamiento")
        st.table(fo2[["B", "L", "h", "qserv", "qadm", "s_mm", "costo"]])

    def recomendar(fo1, fo2):
        """
        Mini sistema experto:
        - Si FO2 reduce el asentamiento ≥ 5 mm y su costo es ≤ 5 % más caro → prioriza FO2.
        - En caso contrario → prioriza FO1 (más económico).
        """
        if (fo1.s_mm - fo2.s_mm >= 5.0) and (fo2.costo <= 1.05 * fo1.costo):
            tag = "FO2 (mínimo asentamiento)"
            motivo = (
                f"Se prioriza un menor asentamiento (reducción de {fo1.s_mm - fo2.s_mm:.1f} mm) "
                "con un incremento de costo aceptable (≤ 5 %)."
            )
            return fo2, tag, motivo
        else:
            tag = "FO1 (mínimo costo)"
            motivo = (
                "Se prioriza la solución más económica, dado que la reducción adicional de asentamiento "
                "no justifica el incremento de costo."
            )
            return fo1, tag, motivo

    chosen, tag, why = recomendar(fo1, fo2)

    texto_md = (
        "## ✅ Recomendación de Diseño  \n\n"
        f"**Solución seleccionada:** {tag}  \n\n"
        f"- B = **{chosen.B:.2f} m**  \n"
        f"- L = **{chosen.L:.2f} m**  \n"
        f"- h = **{chosen.h:.2f} m**  \n"
        f"- q_serv = **{chosen.qserv:.1f} kPa** ≤ q_adm = **{chosen.qadm:.1f} kPa**  \n"
        f"- Asentamiento estimado = **{chosen.s_mm:.2f} mm** ≤ {s_adm_mm:.1f} mm  \n"
        f"- Costo aproximado ≈ **S/ {chosen.costo:,.2f}**  \n\n"
        f"**Criterio de elección entre FO1 y FO2:**  \n"
        f"{why}"
    )

    st.markdown(texto_md)

    st.download_button(
        "Descargar soluciones (CSV)",
        df.to_csv(index=False),
        "soluciones.csv",
        "text/csv"
    )

# ========================================================================
# ==================== GRÁFICAS MEASURED vs CALCULATED ===================
# ========================================================================
st.markdown("---")
st.header("Análisis gráfico del desempeño de los modelos ML")

col_qu, col_s = st.columns(2)

# ---------- Gráfica para q_ult (solo ML) ----------
with col_qu:
    st.subheader("q₍ult₎ · Predecido ML vs Calculado (Meyerhof)")

    if st.session_state.ML_MODEL is None or st.session_state.DF_QU is None:
        st.info("Entrena primero el modelo de q₍ult₎ para ver esta gráfica.")
    else:
        df_qu = st.session_state.DF_QU.copy()
        X_qu = df_qu[["gamma", "B", "D", "phi", "L_over_B"]]
        y_qu = df_qu["qu"].astype(float)

        # Predicción ML
        y_pred_ml = st.session_state.ML_MODEL.predict(X_qu)

        st.caption("Modelo ML (Gradient Boosting)")
        df_plot_ml = pd.DataFrame({
            "Measured_qu": y_qu,
            "Calculated_qu_ML": y_pred_ml
        })
        st.scatter_chart(df_plot_ml, x="Measured_qu", y="Calculated_qu_ML")

# ---------- Gráfica para asentamiento ----------
with col_s:
    st.subheader("Asentamiento · Predecido ML vs Calculado (Meyerhof) ")

    if st.session_state.ML_S_MODEL is None or st.session_state.DF_S is None:
        st.info("Entrena primero el modelo de asentamiento para ver esta gráfica.")
    else:
        df_s = st.session_state.DF_S.copy()
        X_s = df_s[["SPT", "B", "Df_over_B", "q"]]
        y_s = df_s["s_mm"].astype(float)

        # Predicción ML (recordar que entrenamos en log, pero aquí queremos s_mm)
        y_log_pred = st.session_state.ML_S_MODEL.predict(X_s)
        y_pred_s = np.expm1(y_log_pred)

        df_plot_s = pd.DataFrame({
            "Measured_s": y_s,
            "Calculated_s_ML": y_pred_s
        })
        st.scatter_chart(df_plot_s, x="Measured_s", y="Calculated_s_ML")


