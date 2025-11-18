# app.py — 2 funciones objetivo: costo (FO1) y asentamiento (FO2)
# Verificaciones: q_serv ≤ q_adm, q_max ≤ q_adm y s ≤ s_adm
# Incluye ML opcional (paper) con persistencia en session_state

import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Optimización de Cimentaciones — Minimal", layout="centered")
st.title("Optimización de Cimentaciones (mínima)")
st.caption("FO1: minimizar costo · FO2: minimizar asentamiento · con verificación de servicio.")

# ===================== Estado persistente =====================
if "ml_model" not in st.session_state:
    st.session_state.ml_model = None
if "ml_rmse" not in st.session_state:
    st.session_state.ml_rmse = None
if "use_ml_flag" not in st.session_state:
    st.session_state.use_ml_flag = False

# ========= Diagnóstico e importación de ML (paper) =========
SKLEARN_OK = False
try:
    import sklearn
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import GradientBoostingRegressor
    SKLEARN_OK = True
    st.info(f"✅ scikit-learn cargado: versión {sklearn.__version__}")
except Exception as _e:
    st.warning(f"⚠️ scikit-learn NO cargó. Motivo: {type(_e).__name__}: {_e}")
    st.caption("Se usará el método clásico (Meyerhof). Para activar ML, revisa tu requirements.txt.")

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

# ======================== ML opcional (paper) ========================
st.subheader("ML opcional (paper)")
if SKLEARN_OK:
    up = st.file_uploader("CSV entrenamiento (gamma,B,D,phi,L_over_B,qu)", type=["csv"])

    # toggle persistente
    st.session_state.use_ml_flag = st.toggle(
        "Usar modelo ML si está entrenado",
        value=st.session_state.use_ml_flag
    )

    def train_ml(csv_file):
        df = pd.read_csv(csv_file)
        req = ["gamma", "B", "D", "phi", "L_over_B", "qu"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise ValueError(f"Faltan columnas: {miss}")
        X = df[["gamma", "B", "D", "phi", "L_over_B"]].astype(float)
        y = df["qu"].astype(float)
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.15, random_state=42)
        gbr = GradientBoostingRegressor(random_state=42)
        grid = GridSearchCV(
            gbr,
            {"n_estimators": [100, 200], "max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1]},
            cv=5,
            scoring="neg_root_mean_squared_error",
        )
        grid.fit(Xtr, ytr)
        return grid.best_estimator_, -grid.best_score_

    if st.button("Entrenar modelo", use_container_width=True):
        if up is None:
            st.warning("Sube un CSV válido con columnas: gamma,B,D,phi,L_over_B,qu.")
        else:
            try:
                model, rmse = train_ml(up)
                st.session_state.ml_model = model
                st.session_state.ml_rmse = rmse
                st.session_state.use_ml_flag = True  # activar automáticamente tras entrenar
                st.success(f"Modelo entrenado. RMSE≈ {rmse:,.2f} kPa (ML activado)")
            except Exception as e:
                st.error(f"Error entrenando: {e}")
else:
    st.caption("Para usar ML, instala scikit-learn (ver requirements.txt).")

# ======================== Capacidad última (q_u) ====================
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
    sc, sq, sg = 1.3, 1.2, 1.0  # rectangular
    q_eff = gamma * D
    # c≈0 (friccional). Si hay cohesión: qu += c*Nc*sc
    return q_eff * Nq * sq + 0.5 * gamma * B * Ng * sg

def qult_pred(gamma, B, D, phi, L_over_B):
    """Predice q_u con ML si está activo; si no, usa Meyerhof."""
    if (SKLEARN_OK
        and st.session_state.use_ml_flag
        and st.session_state.ml_model is not None):
        X = pd.DataFrame([{
            "gamma": float(gamma), "B": float(B), "D": float(D),
            "phi": float(phi), "L_over_B": float(L_over_B)
        }])
        return float(st.session_state.ml_model.predict(X)[0])
    return qult_meyerhof(B, D, phi, gamma)

# ======================== Servicio y asentamiento =============
def contact_pressures(N, B, L, ex=0.0, ey=0.0):
    """q_serv y q_max con núcleo/área efectiva (sin momentos aquí)."""
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
    vol = B * L * h
    acero_kg = acero_kg_m3 * vol
    exc = B * L * D
    return vol * c_conc + acero_kg * c_acero_kg + exc * c_exc

# ======================== Corrida principal ====================
if st.button("🚀 Optimizar (FO1 & FO2)"):
    # Banner del modelo usado
    st.success(
        "Modelo de capacidad usado: "
        + ("**ML** (paper)" + (f" — RMSE≈ {st.session_state.ml_rmse:,.2f} kPa"
           if st.session_state.ml_rmse else "")
           if (SKLEARN_OK and st.session_state.use_ml_flag and st.session_state.ml_model is not None)
           else "**Meyerhof** (clásico)")
    )

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
            margen = qadm - qserv
            rows.append([B, L, h, qu, qadm, qserv, qmax, s, costo, margen])

    if not rows:
        st.error("Sin soluciones factibles con los parámetros dados.")
        st.stop()

    df = pd.DataFrame(rows, columns=["B", "L", "h", "qu", "qadm", "qserv", "qmax", "s_mm", "costo", "margen"])

    # -------- FO1: mínimo costo --------
    fo1 = df.loc[df["costo"].idxmin()]

    # -------- FO2: mínimo asentamiento (desempate por costo) --------
    fo2 = df.sort_values(["s_mm", "costo"]).iloc[0]

    # ===== Regla de decisión =====
    def decidir(fo1, fo2):
        c1, c2 = float(fo1.costo), float(fo2.costo)
        s1, s2 = float(fo1.s_mm), float(fo2.s_mm)
        m1, m2 = float(fo1.margen), float(fo2.margen)
        mejora_s = s1 - s2
        sobrecosto = (c2 / c1) - 1.0
        degrado_margen = m1 - m2  # positivo si FO2 tiene menor margen
        if (mejora_s >= 5.0) and (sobrecosto <= 0.06) and (degrado_margen <= 50.0):
            why = (f"Se elige **FO2 (asentamiento)**: mejora s en {mejora_s:.1f} mm "
                   f"con ΔCosto {100*sobrecosto:.1f}% ≤ 6% y margen ≈ {m2:.1f} kPa (no se degrada >50 kPa).")
            return fo2, "FO2 (mínimo asentamiento)", why
        return fo1, "FO1 (mínimo costo)", "Se elige **FO1**: el ahorro de costo domina; la mejora de s no justifica el sobrecosto."

    chosen, tag, why = decidir(fo1, fo2)

    cA, cB = st.columns(2)
    with cA:
        st.subheader("FO1 · Mínimo costo")
        st.write(fo1[["B", "L", "h", "qserv", "qadm", "qmax", "s_mm", "costo", "margen"]])
    with cB:
        st.subheader("FO2 · Mínimo asentamiento")
        st.write(fo2[["B", "L", "h", "qserv", "qadm", "qmax", "s_mm", "costo", "margen"]])

    st.markdown("### ✅ Recomendación")
    st.write(
        f"**{tag}**  \n"
        f"B = **{float(chosen.B):.2f} m**, L = **{float(chosen.L):.2f} m**, h = **{float(chosen.h):.2f} m**  \n"
        f"q_serv = **{float(chosen.qserv):.1f} kPa** ≤ q_adm = **{float(chosen.qadm):.1f} kPa**; "
        f"q_max = **{float(chosen.qmax):.1f} kPa** ≤ q_adm  \n"
        f"s = **{float(chosen.s_mm):.1f} mm** ≤ s_adm = **{s_adm_mm:.0f} mm**  \n"
        f"Costo ≈ **S/ {float(chosen.costo):,.2f}**; Margen = **{float(chosen.margen):.1f} kPa**  \n"
        f"**Motivo:** {why}"
    )

    st.download_button(
        "Descargar soluciones (CSV)",
        df.to_csv(index=False),
        "soluciones_minimal.csv",
        "text/csv",
        use_container_width=True,
    )


