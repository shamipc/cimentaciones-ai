# app.py — versión mínima con 2 funciones objetivo (FO1 costo, FO2 asentamiento)
# Verifica: q_serv ≤ q_adm, q_max ≤ q_adm y s ≤ s_adm
# ML opcional (paper) con APRENDIZAJE DE RESIDUO:
#     qu = qu_meyerhof + residuo_ML(gamma,B,D,phi,L/B,D/B,q_eff)

import math
import numpy as np
import pandas as pd
import streamlit as st

# ====== ML opcional (con importación defensiva) ======
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
st.caption("Entradas mínimas (paper) + 2 funciones objetivo (FO1 costo, FO2 asentamiento) con verificación de servicio y ML opcional por residuo vs Meyerhof.")

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
if "REL_RMSE_ML" not in st.session_state:
    st.session_state.REL_RMSE_ML = None

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
    # c≈0 (arenas/friccionantes); coef. de forma rectangulares
    Nc, Nq, Ng = bearing_factors(phi)
    sc, sq, sg = 1.3, 1.2, 1.0
    q_eff = gamma * D
    return q_eff * Nq * sq + 0.5 * gamma * B * Ng * sg

# ====== ENTRENAMIENTO: aprendizaje de residuo (recomendado) ======
def train_ml(csv):
    """
    Entrena ML para predecir el RESIDUO respecto a Meyerhof:
        residuo y = qu_real - qu_meyerhof
    En inferencia: qu_pred = qu_meyerhof + y_hat
    Devuelve: modelo, RMSE(kPa) y RMSE relativo (%) vs mediana(qu) de validación.
    """
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn no disponible en el entorno.")

    df = pd.read_csv(csv).copy()
    req = ["gamma", "B", "D", "phi", "L_over_B", "qu"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan columnas: {miss}")

    # Features físicos adicionales
    df["D_over_B"] = df["D"] / np.clip(df["B"], 1e-9, None)
    df["q_eff"]    = df["gamma"] * df["D"]

    # Base clásica por fila
    def qu_meyerhof_row(r):
        Nc, Nq, Ng = bearing_factors(r["phi"])
        sc, sq, sg = 1.3, 1.2, 1.0
        q_eff = r["gamma"] * r["D"]
        return q_eff * Nq * sq + 0.5 * r["gamma"] * r["B"] * Ng * sg

    qu_m = df.apply(qu_meyerhof_row, axis=1)

    # Objetivo: residuo
    y = (df["qu"].astype(float).values - qu_m.values)
    feat_cols = ["gamma", "B", "D", "phi", "L_over_B", "D_over_B", "q_eff"]
    X = df[feat_cols].astype(float).values

    # Partición 80/20 (estratificada levemente por phi)
    phi_bins = pd.cut(df["phi"], bins=[0, 25, 30, 35, 40, 50], labels=False, include_lowest=True)
    Xtr, Xva, ytr, yva, qm_tr, qm_va = train_test_split(
        X, y, qu_m.values, test_size=0.20, random_state=42, stratify=phi_bins
    )

    # Modelo robusto a outliers
    gbr = GradientBoostingRegressor(
        loss="huber",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
    )
    gbr.fit(Xtr, ytr)

    # Métricas en validación (RMSE en kPa y relativo %)
    yhat_res = gbr.predict(Xva)
    qu_pred  = qm_va + yhat_res
    qu_true  = qm_va + yva
    rmse     = float(np.sqrt(mean_squared_error(qu_true, qu_pred)))
    med_qu   = float(np.median(qu_true))
    rel_rmse = float(100.0 * rmse / max(1.0, med_qu))

    # Guardar metadatos mínimos para inferencia
    gbr._feat_cols = feat_cols
    gbr._use_residual = True
    return gbr, rmse, rel_rmse

if st.button("Entrenar modelo", use_container_width=True):
    if not SKLEARN_OK:
        st.error("scikit-learn no está disponible en el entorno.")
    elif up is None:
        st.warning("Sube un CSV válido con columnas: gamma,B,D,phi,L_over_B,qu.")
    else:
        try:
            model, rmse, rel_rmse = train_ml(up)
            st.session_state.ML_MODEL = model
            st.session_state.RMSE_ML = rmse
            st.session_state.REL_RMSE_ML = rel_rmse
            st.success(f"Modelo entrenado. RMSE≈ {rmse:,.2f} kPa (≈ {rel_rmse:.1f}% de la mediana de qu)")
        except Exception as e:
            st.error(f"Error entrenando: {e}")

# ====== Predicción de qu (usa ML residual si procede) ======
def qult_pred(gamma_val, B, D, phi_val, L_over_B_val):
    """
    qu: ML residual (si entrenado y switch activo) o Meyerhof (clásico).
    En residual: qu = qu_meyerhof + residuo_ML(...)
    """
    # Siempre calcula la base clásica (ancla física)
    Nc, Nq, Ng = bearing_factors(phi_val)
    sc, sq, sg = 1.3, 1.2, 1.0
    q_eff = gamma_val * D
    qu_base = q_eff * Nq * sq + 0.5 * gamma_val * B * Ng * sg

    if use_ml and (st.session_state.ML_MODEL is not None) and SKLEARN_OK:
        mdl = st.session_state.ML_MODEL
        if getattr(mdl, "_use_residual", False):
            X = pd.DataFrame([{
                "gamma": gamma_val, "B": B, "D": D, "phi": phi_val,
                "L_over_B": L_over_B_val,
                "D_over_B": D / max(B, 1e-9),
                "q_eff": q_eff
            }])
            res = float(mdl.predict(X[mdl._feat_cols])[0])
            return max(0.0, qu_base + res)
        else:
            # (fallback por si se entrenó un modelo directo accidentalmente)
            X = pd.DataFrame([{
                "gamma": gamma_val, "B": B, "D": D, "phi": phi_val, "L_over_B": L_over_B_val
            }])
            return float(mdl.predict(X)[0])

    # Clásico
    return qu_base

# ======================== Servicio y asentamiento =============
def contact_pressures(N, B, L, ex=0.0, ey=0.0):
    """q_serv y q_max considerando núcleo/área efectiva (sin momentos por defecto)."""
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

# ======================== Corrida principal ====================
if st.button("🚀 Optimizar (FO1 & FO2)"):
    Bs = np.linspace(Bmin, Bmax, int(nB))
    hs = np.linspace(hmin, hmax, int(nh))

    rows = []
    for B in Bs:
        L = L_over_B * B
        for h in hs:
            qu = qult_pred(gamma, B, D, phi, L_over_B)
            qadm = qu / FS
            qserv, qmax = contact_pressures(N, B, L)  # sin excentricidad por simplicidad
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

    # ===== Banner del modelo usado =====
    if use_ml and (st.session_state.ML_MODEL is not None) and SKLEARN_OK:
        txt = f"Modelo de capacidad usado: **ML (residuo vs. Meyerhof)** — RMSE≈ {st.session_state.RMSE_ML:,.2f} kPa"
        if st.session_state.REL_RMSE_ML is not None:
            txt += f" (≈ {st.session_state.REL_RMSE_ML:.1f}% de la mediana de qu)"
        st.success(txt)
    else:
        st.success("Modelo de capacidad usado: **Meyerhof (clásico)**")

    # ===== Mostrar FO1 y FO2 =====
    cA, cB = st.columns(2)
    with cA:
        st.subheader("FO1 · Mínimo costo")
        st.table(fo1[["B","L","h","qserv","qadm","qmax","s_mm","costo"]])
    with cB:
        st.subheader("FO2 · Mínimo asentamiento")
        st.table(fo2[["B","L","h","qserv","qadm","qmax","s_mm","costo"]])

    # ===== Recomendación (elige FO1 o FO2) =====
    # Criterio: si FO2 baja s en ≥5 mm respecto a FO1 y su costo ≤ +5% del FO1, toma FO2; si no, FO1.
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


