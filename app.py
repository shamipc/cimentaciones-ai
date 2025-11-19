# app.py — Optimización de Cimentaciones con ML opcional (qu y asentamiento)
# Corrige error 'squared' en mean_squared_error con helper _rmse (compatible con scikit-learn antiguas)

import math
import numpy as np
import pandas as pd
import streamlit as st

# ================= Utilidades métricas (robusto a versiones antiguas) =================
def _rmse(y_true, y_pred):
    """RMSE robusto a versiones antiguas de scikit-learn (sin argumento 'squared')."""
    try:
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_true, y_pred)  # sin 'squared'
        return float(np.sqrt(mse))
    except Exception:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mae(y_true, y_pred):
    try:
        from sklearn.metrics import mean_absolute_error
        return float(mean_absolute_error(y_true, y_pred))
    except Exception:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean(np.abs(y_true - y_pred)))

def _r2(y_true, y_pred):
    try:
        from sklearn.metrics import r2_score
        return float(r2_score(y_true, y_pred))
    except Exception:
        # fallback de R² manual
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

# ================= ML opcional: imports defensivos =================
try:
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import GradientBoostingRegressor
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False
    train_test_split = GridSearchCV = GradientBoostingRegressor = None

# ================= Streamlit config =================
st.set_page_config(page_title="Cimentaciones • Optimización + ML", layout="wide")
st.title("Optimización de Cimentaciones con ML opcional")
st.caption("Dos funciones objetivo (FO1 costo, FO2 asentamiento). ML opcional para predecir **qu** y **s**.")

# ================= Entradas básicas =================
c1, c2 = st.columns(2)
with c1:
    N = st.number_input("Carga axial N (kN)", 100.0, 2e5, 800.0, 10.0)
    phi = st.number_input("ϕ (°)", 0.0, 45.0, 32.0, 0.5)
    gamma = st.number_input("γ efectivo en base (kN/m³)", 10.0, 24.0, 18.0, 0.1)
    Es = st.number_input("Eₛ (kPa) para s (aprox.)", 3000.0, 100000.0, 25000.0, 500.0)
with c2:
    D = st.number_input("Profundidad de base D (m)", 0.5, 6.0, 1.5, 0.1)
    FS = st.number_input("FS (capacidad)", 1.5, 4.0, 2.5, 0.1)
    L_over_B = st.number_input("Relación L/B (constante)", 0.8, 2.0, 1.0, 0.05)
    s_adm_mm = st.number_input("Asentamiento admisible sₐ (mm)", 5.0, 50.0, 25.0, 1.0)

st.markdown("**Rangos de búsqueda (B y h)**")
c3, c4, c5 = st.columns(3)
with c3:
    Bmin = st.number_input("B min (m)", 0.8, 6.0, 1.0, 0.1)
    Bmax = st.number_input("B max (m)", 0.8, 6.0, 3.0, 0.1)
with c4:
    hmin = st.number_input("h min (m)", 0.3, 2.0, 0.5, 0.05)
    hmax = st.number_input("h max (m)", 0.3, 2.0, 1.2, 0.05)
with c5:
    nB = st.number_input("Puntos B", 5, 80, 30, 1)
    nh = st.number_input("Puntos h", 3, 60, 12, 1)

st.markdown("---")

# =============================================================================
#                    ML opcional — qu  y  asentamiento
# =============================================================================
if not SKLEARN_OK:
    st.warning("Para usar ML, agrega `scikit-learn>=1.3,<1.5` a requirements.txt. "
               "Sin ML, se usa Meyerhof para qu y modelo elástico simple para s.")

# Estado global
for key in ["ML_MODEL_QU", "ML_MODEL_S", "ML_METRICS_QU", "ML_METRICS_S"]:
    if key not in st.session_state:
        st.session_state[key] = None if "MODEL" in key else {}

c_ml1, c_ml2 = st.columns(2)

# -------------------- Columna izquierda: modelo qu --------------------
with c_ml1:
    st.subheader("CSV qu (gamma,B,D,phi,L_over_B,qu)")
    up_qu = st.file_uploader("Sube CSV para **qu**", type=["csv"], key="up_qu")
    use_ml_qu = st.toggle("Usar ML para qu si está entrenado", value=False, key="switch_qu")

    def train_qu(csv_file):
        df = pd.read_csv(csv_file)
        req = ["gamma", "B", "D", "phi", "L_over_B", "qu"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise ValueError(f"Faltan columnas: {miss}")

        X = df[["gamma", "B", "D", "phi", "L_over_B"]]
        y = df["qu"].astype(float)

        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)

        gbr = GradientBoostingRegressor(random_state=42)
        grid = GridSearchCV(
            gbr,
            {"n_estimators": [150, 300, 500],
             "max_depth": [2, 3],
             "learning_rate": [0.05, 0.1],
             "min_samples_leaf": [3, 5]},
            cv=5,
            scoring="neg_mean_squared_error",  # MSE; luego hacemos sqrt para RMSE
            n_jobs=-1
        )
        grid.fit(Xtr, ytr)
        best = grid.best_estimator_

        yhat = best.predict(Xva)
        rmse = _rmse(yva, yhat)
        mae = _mae(yva, yhat)
        r2 = _r2(yva, yhat)
        med = float(np.median(yva)) if len(yva) else float("nan")
        nrmse_med = float(rmse / med * 100) if np.isfinite(med) and med != 0 else float("nan")
        bias = float(np.mean(yhat - yva))
        return best, {"rmse": rmse, "mae": mae, "r2": r2, "nrmse_med": nrmse_med, "bias": bias, "n_val": int(len(yva))}

    if st.button("Entrenar modelo de qu", use_container_width=True):
        if not SKLEARN_OK:
            st.error("scikit-learn no está disponible en el entorno.")
        elif up_qu is None:
            st.warning("Sube un CSV con columnas: gamma,B,D,phi,L_over_B,qu.")
        else:
            try:
                model, metrics = train_qu(up_qu)
                st.session_state.ML_MODEL_QU = model
                st.session_state.ML_METRICS_QU = metrics
                st.success(f"Modelo qu entrenado. RMSE≈ {metrics['rmse']:.2f} kPa")
            except Exception as e:
                st.error(f"Error entrenando qu: {e}")

    if st.session_state.ML_METRICS_QU:
        m = st.session_state.ML_METRICS_QU
        st.caption("Métricas (validación qu)")
        a,b,c,d,e = st.columns(5)
        a.metric("R²", f"{m['r2']:.3f}" if np.isfinite(m['r2']) else "—")
        b.metric("RMSE", f"{m['rmse']:.2f} kPa")
        c.metric("MAE", f"{m['mae']:.2f} kPa")
        d.metric("nRMSE/med", f"{m['nrmse_med']:.1f}%"
                 if np.isfinite(m['nrmse_med']) else "—")
        e.metric("Sesgo", f"{m['bias']:+.2f} kPa")

# -------------------- Columna derecha: modelo asentamiento --------------------
with c_ml2:
    st.subheader("CSV asentamiento (gamma,B,D,phi,L_over_B,qserv,Es,s_mm)")
    up_s = st.file_uploader("Sube CSV para **s**", type=["csv"], key="up_s")
    use_ml_s = st.toggle("Usar ML para s si está entrenado", value=False, key="switch_s")

    def train_s(csv_file):
        df = pd.read_csv(csv_file)
        req = ["gamma", "B", "D", "phi", "L_over_B", "qserv", "Es", "s_mm"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise ValueError(f"Faltan columnas: {miss}")

        X = df[["gamma", "B", "D", "phi", "L_over_B", "qserv", "Es"]]
        y = df["s_mm"].astype(float)

        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)

        gbr = GradientBoostingRegressor(random_state=42)
        grid = GridSearchCV(
            gbr,
            {"n_estimators": [150, 300, 500],
             "max_depth": [2, 3],
             "learning_rate": [0.05, 0.1],
             "min_samples_leaf": [3, 5]},
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        grid.fit(Xtr, ytr)
        best = grid.best_estimator_

        yhat = best.predict(Xva)
        rmse = _rmse(yva, yhat)
        mae = _mae(yva, yhat)
        r2 = _r2(yva, yhat)
        med = float(np.median(yva)) if len(yva) else float("nan")
        nrmse_med = float(rmse / med * 100) if np.isfinite(med) and med != 0 else float("nan")
        bias = float(np.mean(yhat - yva))
        return best, {"rmse": rmse, "mae": mae, "r2": r2, "nrmse_med": nrmse_med, "bias": bias, "n_val": int(len(yva))}

    if st.button("Entrenar modelo de asentamiento", use_container_width=True):
        if not SKLEARN_OK:
            st.error("scikit-learn no está disponible en el entorno.")
        elif up_s is None:
            st.warning("Sube un CSV con columnas: gamma,B,D,phi,L_over_B,qserv,Es,s_mm.")
        else:
            try:
                model, metrics = train_s(up_s)
                st.session_state.ML_MODEL_S = model
                st.session_state.ML_METRICS_S = metrics
                st.success(f"Modelo s entrenado. RMSE≈ {metrics['rmse']:.2f} mm")
            except Exception as e:
                st.error(f"Error entrenando s: {e}")

    if st.session_state.ML_METRICS_S:
        m = st.session_state.ML_METRICS_S
        st.caption("Métricas (validación s)")
        a,b,c,d,e = st.columns(5)
        a.metric("R²", f"{m['r2']:.3f}" if np.isfinite(m['r2']) else "—")
        b.metric("RMSE", f"{m['rmse']:.2f} mm")
        c.metric("MAE", f"{m['mae']:.2f} mm")
        d.metric("nRMSE/med", f"{m['nrmse_med']:.1f}%"
                 if np.isfinite(m['nrmse_med']) else "—")
        e.metric("Sesgo", f"{m['bias']:+.2f} mm")

st.markdown("---")

# =============================================================================
#                         Cálculos geotécnicos base
# =============================================================================
def bearing_factors(phi_deg: float):
    phi_rad = math.radians(phi_deg)
    if phi_rad < 1e-6:
        Nq, Nc, Ng = 1.0, 5.14, 0.0
    else:
        Nq = math.e ** (math.pi * math.tan(phi_rad)) * (math.tan(math.pi / 4 + phi_rad / 2)) ** 2
        Nc = (Nq - 1.0) / math.tan(phi_rad)
        Ng = 2.0 * (Nq + 1.0) * math.tan(phi_rad)
    return Nc, Nq, Ng

def qult_meyerhof(B, D, phi, gamma):
    Nc, Nq, Ng = bearing_factors(phi)
    sc, sq, sg = 1.3, 1.2, 1.0  # forma rectangular
    q_eff = gamma * D
    return q_eff * Nq * sq + 0.5 * gamma * B * Ng * sg

def contact_pressures(N, B, L, ex=0.0, ey=0.0):
    qavg = N / (B * L)
    in_kern = (abs(ex) <= B / 6) and (abs(ey) <= L / 6)
    if in_kern:
        qmax = qavg * (1 + 6 * abs(ex) / B + 6 * abs(ey) / L)
        qserv = qavg
    else:
        Beff, Leff = B - 2 * abs(ex), L - 2 * abs(ey)
        if Beff <= 0 or Leff <= 0:
            return np.inf, np.inf
        qserv = N / (Beff * Leff)
        qmax = 2 * N / (Beff * Leff)
    return qserv, qmax

def settlement_elastic_mm(qserv_kpa, B_m, Es_kpa, nu=0.30):
    return 1000.0 * (qserv_kpa * B_m * (1 - nu ** 2) / Es_kpa)

# Predictores unificados con ML opcional
def qult_pred(gamma_val, B, D, phi_val, L_over_B_val):
    if SKLEARN_OK and st.session_state.ML_MODEL_QU is not None and st.session_state.get("switch_qu", False):
        X = pd.DataFrame([{"gamma": gamma_val, "B": B, "D": D, "phi": phi_val, "L_over_B": L_over_B_val}])
        return float(st.session_state.ML_MODEL_QU.predict(X)[0])
    return qult_meyerhof(B, D, phi_val, gamma_val)

def s_pred(gamma_val, B, D, phi_val, L_over_B_val, qserv_kpa, Es_kpa):
    if SKLEARN_OK and st.session_state.ML_MODEL_S is not None and st.session_state.get("switch_s", False):
        X = pd.DataFrame([{
            "gamma": gamma_val, "B": B, "D": D, "phi": phi_val, "L_over_B": L_over_B_val,
            "qserv": qserv_kpa, "Es": Es_kpa
        }])
        return float(st.session_state.ML_MODEL_S.predict(X)[0])
    return settlement_elastic_mm(qserv_kpa, B, Es_kpa)

# Costo simple
def cost_S(B, L, h, c_conc=650.0, c_acero_kg=5.5, acero_kg_m3=60.0, c_exc=80.0, D=1.5):
    vol = B * L * h
    acero_kg = acero_kg_m3 * vol
    exc = B * L * D
    return vol * c_conc + acero_kg * c_acero_kg + exc * c_exc

# =============================================================================
#                                 Optimización
# =============================================================================
if st.button("🚀 Optimizar (FO1 & FO2)", use_container_width=True):
    Bs = np.linspace(Bmin, Bmax, int(nB))
    hs = np.linspace(hmin, hmax, int(nh))

    rows = []
    for B in Bs:
        L = L_over_B * B
        for h in hs:
            qu = qult_pred(gamma, B, D, phi, L_over_B)
            qadm = qu / FS
            qserv, qmax = contact_pressures(N, B, L)
            if not (qserv <= qadm and qmax <= qadm):
                continue
            s = s_pred(gamma, B, D, phi, L_over_B, qserv, Es)
            if s > s_adm_mm:
                continue
            costo = cost_S(B, L, h, D=D)
            rows.append([B, L, h, qu, qadm, qserv, qmax, s, costo])

    if not rows:
        st.error("Sin soluciones factibles con los parámetros dados.")
        st.stop()

    df = pd.DataFrame(rows, columns=["B","L","h","qu","qadm","qserv","qmax","s_mm","costo"])

    # FO1: costo mínimo | FO2: asentamiento mínimo
    fo1 = df.loc[df["costo"].idxmin()]
    fo2 = df.loc[df["s_mm"].idxmin()]

    # Banner del modelo usado
    banner = []
    if st.session_state.get("switch_qu", False) and st.session_state.ML_METRICS_QU:
        m = st.session_state.ML_METRICS_QU
        banner.append(f"qu-ML RMSE≈{m['rmse']:.1f} kPa R²≈{m['r2']:.2f}")
    else:
        banner.append("qu: Meyerhof")
    if st.session_state.get("switch_s", False) and st.session_state.ML_METRICS_S:
        m2 = st.session_state.ML_METRICS_S
        banner.append(f"s-ML RMSE≈{m2['rmse']:.1f} mm R²≈{m2['r2']:.2f}")
    else:
        banner.append("s: elástico simple")

    st.success("Modelo de capacidad/servicio usado → " + " | ".join(banner))

    cA, cB = st.columns(2)
    with cA:
        st.subheader("FO1 · Mínimo costo")
        st.table(fo1[["B","L","h","qserv","qadm","qmax","s_mm","costo"]])
    with cB:
        st.subheader("FO2 · Mínimo asentamiento")
        st.table(fo2[["B","L","h","qserv","qadm","qmax","s_mm","costo"]])

    # Recomendación (simple)
    def recomendar(fo1, fo2):
        s1, s2 = fo1.s_mm, fo2.s_mm
        c1, c2 = fo1.costo, fo2.costo
        if (s1 - s2 >= 5.0) and (c2 <= 1.05 * c1):
            tag = "FO2 (mínimo asentamiento)"
            why = f"Menor s (−{s1 - s2:.1f} mm) con costo ≤ +5%."
            return fo2, tag, why
        tag = "FO1 (mínimo costo)"
        why = "El ahorro de costo domina y se cumplen verificaciones."
        return fo1, tag, why

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

    st.download_button("Descargar soluciones (CSV)",
                       df.to_csv(index=False), "soluciones.csv", "text/csv",
                       use_container_width=True)


