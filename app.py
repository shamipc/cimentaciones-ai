# app.py — versión mínima con 2 funciones objetivo (FO1 costo, FO2 asentamiento)
# Verifica: q_serv ≤ q_adm, q_max ≤ q_adm y s ≤ s_adm
# ML opcional (paper) con APRENDIZAJE DE RESIDUO: qu = qu_meyerhof + ML(gamma,B,D,phi,L/B,D/B)

import math
import numpy as np
import pandas as pd
import streamlit as st

# ====== ML opcional (importación defensiva) ======
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False
    train_test_split = GradientBoostingRegressor = mean_squared_error = None

st.set_page_config(page_title="Optimización de Cimentaciones — Minimal", layout="centered")
st.title("Optimización de Cimentaciones (mínima)")
st.caption("Entradas mínimas (paper) + 2 funciones objetivo (FO1 costo, FO2 asentamiento) con verificación de servicio.")

# ======================== Entradas mínimas ========================
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

# ======================== Estado global ML ========================
if "ML_MODEL" not in st.session_state:
    st.session_state.ML_MODEL = None
if "RMSE_ML" not in st.session_state:
    st.session_state.RMSE_ML = None

# ======================== ML opcional (paper) =====================
st.subheader("ML opcional (paper)")
if not SKLEARN_OK:
    st.warning("Para usar ML, agrega `scikit-learn>=1.3,<1.5` en requirements.txt. "
               "Si no, se usará el método clásico (Meyerhof).")

up = st.file_uploader("CSV entrenamiento (gamma,B,D,phi,L_over_B,qu)", type=["csv"])
use_ml = st.toggle("Usar modelo ML si está entrenado", value=False)

# ====== Capacidad clásica (Meyerhof) ======
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
    # rectangular; suelo friccionante (c≈0)
    Nc, Nq, Ng = bearing_factors(phi)
    sc, sq, sg = 1.3, 1.2, 1.0
    q_eff = gamma * D
    return q_eff * Nq * sq + 0.5 * gamma * B * Ng * sg

# ====== ENTRENAMIENTO: aprendizaje de residuo ======
def train_ml(csv):
    """
    Entrena ML para predecir qu vía residuo:
      residuo y = qu_real - qu_meyerhof
    En inferencia: qu_pred = qu_meyerhof + y_hat
    """
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn no disponible en el entorno.")

    df = pd.read_csv(csv).copy()
    req = ["gamma", "B", "D", "phi", "L_over_B", "qu"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan columnas: {miss}")

    # feature adicional adimensional
    df["D_over_B"] = df["D"] / np.clip(df["B"], 1e-9, None)

    # base (Meyerhof) para residuo
    qu_m = df.apply(lambda r: qult_meyerhof(r["B"], r["D"], r["phi"], r["gamma"]), axis=1)

    # objetivo: residuo
    y = (df["qu"] - qu_m).astype(float).values
    feat_cols = ["gamma", "B", "D", "phi", "L_over_B", "D_over_B"]
    X = df[feat_cols].astype(float).values

    # estratifica levemente por phi para estabilidad
    phi_bins = pd.cut(df["phi"], bins=[0, 25, 30, 35, 40, 50], labels=False, include_lowest=True)
    Xtr, Xva, ytr, yva, qm_tr, qm_va = train_test_split(
        X, y, qu_m.values, test_size=0.20, random_state=42, stratify=phi_bins
    )

    # Gradient Boosting robusto (Huber)
    gbr = GradientBoostingRegressor(
        loss="huber",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
    )
    gbr.fit(Xtr, ytr)

    # reconstruimos qu en validación
    yhat_res = gbr.predict(Xva)
    qu_pred = qm_va + yhat_res
    qu_true = qm_va + yva
    rmse = float(np.sqrt(mean_squared_error(qu_true, qu_pred)))

    # adjunta metadatos para inferencia
    gbr._feat_cols = feat_cols
    gbr._use_residual = True
    return gbr, rmse

if st.button("Entrenar modelo", use_container_width=True):
    if up is None:
        st.warning("Sube un CSV con columnas: gamma,B,D,phi,L_over_B,qu.")
    else:
        try:
            mdl, rmse = train_ml(up)
            st.session_state.ML_MODEL = mdl
            st.session_state.RMSE_ML = rmse
            st.success(f"Modelo entrenado. RMSE≈ {rmse:,.2f} kPa")
        except Exception as e:
            st.error(f"Error entrenando: {e}")

# ====== Predicción de qu (usa ML si procede) ======
def qult_pred(gamma_val, B, D, phi_val, L_over_B_val):
    """qu: ML (si entrenado y switch activo) o Meyerhof."""
    if use_ml and (st.session_state.ML_MODEL is not None) and SKLEARN_OK:
        mdl = st.session_state.ML_MODEL
        # vector de features (incluye D/B)
        X = pd.DataFrame([{
            "gamma": gamma_val, "B": B, "D": D, "phi": phi_val,
            "L_over_B": L_over_B_val, "D_over_B": D / max(B, 1e-9)
        }])
        # si es residual: qu = qu_meyerhof + res_pred
        if getattr(mdl, "_use_residual", False):
            base = qult_meyerhof(B, D, phi_val, gamma_val)
            res = float(mdl.predict(X[mdl._feat_cols])[0])
            return base + res
        # modo directo (por compatibilidad)
        cols = mdl._feat_cols if hasattr(mdl, "_feat_cols") else X.columns
        return float(mdl.predict(X[cols])[0])
    # clásico
    return qult_meyerhof(B, D, phi_val, gamma_val)

# ======================== Servicio & asentamiento ==============
def contact_pressures(N, B, L, ex=0.0, ey=0.0):
    """q_serv y q_max con núcleo/área efectiva (sin momentos por defecto)."""
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

def settlement_mm(qserv_kpa, B_m, Es_kpa, nu=0.30):
    """Modelo elástico simple: s ≈ q·B·(1-ν²)/Es → mm."""
    return 1000.0 * (qserv_kpa * B_m * (1 - nu ** 2) / Es_kpa)

# ======================== Costo ================================
def cost_S(B, L, h, c_conc=650.0, c_acero_kg=5.5, acero_kg_m3=60.0, c_exc=80.0, D=1.5):
    """Costo simple (S/): concreto + acero + excavación."""
    vol = B * L * h
    acero_kg = acero_kg_m3 * vol
    exc = B * L * D
    return vol * c_conc + acero_kg * c_acero_kg + exc * c_exc

# ======================== Optimización =========================
if st.button("🚀 Optimizar (FO1 & FO2)"):
    Bs = np.linspace(Bmin, Bmax, int(nB))
    hs = np.linspace(hmin, hmax, int(nh))

    rows = []
    for B in Bs:
        L = L_over_B * B
        for h in hs:
            qu = qult_pred(gamma, B, D, phi, L_over_B)
            qadm = qu / FS
            qserv, qmax = contact_pressures(N, B, L)  # sin excentricidad
            if not (qserv <= qadm and qmax <= qadm):
                continue
            s = settlement_mm(qserv, B, Es)
            if s > s_adm_mm:
                continue
            costo = cost_S(B, L, h, D=D)
            rows.append([B, L, h, qu, qadm, qserv, qmax, s, costo])

    if not rows:
        st.error("Sin soluciones factibles con los parámetros dados.")
        st.stop()

    df = pd.DataFrame(rows, columns=["B", "L", "h", "qu", "qadm", "qserv", "qmax", "s_mm", "costo"])

    # FO1: costo mínimo
    fo1 = df.loc[df["costo"].idxmin()]
    # FO2: asentamiento mínimo
    fo2 = df.loc[df["s_mm"].idxmin()]

    # Mensaje modelo usado
    if use_ml and (st.session_state.ML_MODEL is not None) and SKLEARN_OK:
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

    # Recomendación: FO2 solo si reduce s ≥5 mm y su costo ≤ +5% del FO1
    def recomendar(fo1, fo2):
        s1, s2 = fo1.s_mm, fo2.s_mm
        c1, c2 = fo1.costo, fo2.costo
        if (s1 - s2 >= 5.0) and (c2 <= 1.05 * c1):
            return fo2, "FO2 (mínimo asentamiento)", f"Menor s (−{s1 - s2:.1f} mm) con costo ≤ +5%."
        return fo1, "FO1 (mínimo costo)", "El ahorro de costo domina y se cumplen verificaciones."

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

