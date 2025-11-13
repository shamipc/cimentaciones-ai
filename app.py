# app.py — versión mínima centrada en funciones objetivo
# Cumple: q_serv ≤ q_adm, q_max ≤ q_adm y s ≤ s_adm
# FO1 = Min. costo | FO2 = Min. Utilización (q_serv/q_adm) | Recomendación automática

import math
import numpy as np
import pandas as pd
import streamlit as st

# ========= ML (paper-like) con importación defensiva =========
try:
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import GradientBoostingRegressor
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False
    train_test_split = GridSearchCV = GradientBoostingRegressor = None

st.set_page_config(page_title="Optimización de Cimentaciones — Minimal", layout="centered")
st.title("Optimización de Cimentaciones (mínima)")
st.caption("Entradas mínimas (paper) + 2 funciones objetivo (FO1 costo, FO2 utilización) con verificación de servicio.")

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
if not SKLEARN_OK:
    st.warning("Para usar ML, agrega `scikit-learn>=1.3,<1.5` en requirements.txt. "
               "Si no, se usará el método clásico (Meyerhof).")
use_ml = False
ML_MODEL = None
RMSE = None

if SKLEARN_OK:
    up = st.file_uploader("CSV entrenamiento (gamma,B,D,phi,L_over_B,qu)", type=["csv"])
    use_ml = st.toggle("Usar modelo ML si está entrenado", value=False)

    def train_ml(csv):
        df = pd.read_csv(csv)
        req = ["gamma", "B", "D", "phi", "L_over_B", "qu"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise ValueError(f"Faltan columnas: {miss}")
        X = df[["gamma", "B", "D", "phi", "L_over_B"]]
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
                ML_MODEL, RMSE = train_ml(up)
                st.success(f"Modelo entrenado. RMSE≈ {RMSE:,.2f} kPa")
            except Exception as e:
                st.error(f"Error entrenando: {e}")

# ======================== Capacidad última (q_u) ====================
# Si no hay ML, CALCULAMOS q_u con Meyerhof (validado en literatura).
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
    # c≈0 (suelo friccional). Si necesitas cohesión, añade: + c*Nc*sc
    return q_eff * Nq * sq + 0.5 * gamma * B * Ng * sg

def qult_pred(gamma, B, D, phi, L_over_B):
    if use_ml and (ML_MODEL is not None) and SKLEARN_OK:
        X = pd.DataFrame([{"gamma": gamma, "B": B, "D": D, "phi": phi, "L_over_B": L_over_B}])
        return float(ML_MODEL.predict(X)[0])
    return qult_meyerhof(B, D, phi, gamma)

# ======================== Servicio y asentamiento =============
def contact_pressures(N, B, L, ex=0.0, ey=0.0):
    """q_serv y q_max con núcleo/área efectiva (aquí sin momentos)."""
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
            margen = qadm - qserv
            U = qserv / qadm  # ← Utilización (Demanda/Capacidad)
            rows.append([B, L, h, qu, qadm, qserv, qmax, s, costo, margen, U])

    if not rows:
        st.error("Sin soluciones factibles con los parámetros dados.")
        st.stop()

    df = pd.DataFrame(rows, columns=[
        "B", "L", "h", "qu", "qadm", "qserv", "qmax", "s_mm", "costo", "margen", "U"
    ])

    # -------- FO1: mínimo costo --------
    fo1 = df.loc[df["costo"].idxmin()]

    # -------- FO2: Utilización mínima (criterio clásico) --------
    fo2 = df.sort_values(["U", "costo"]).iloc[0]

    # ===== Regla de decisión acordada =====
    # Elige FO2 si: mejora relativa de U ≥ 12 %, mejora de s ≥ 5 mm y ΔCosto ≤ 6 %
    def decidir(fo1, fo2):
        c1, c2 = fo1.costo, fo2.costo
        s1, s2 = fo1.s_mm, fo2.s_mm
        U1, U2 = fo1.U, fo2.U
        mejora_rel_U = (U1 - U2) / U1 if U1 > 0 else 0.0
        if (mejora_rel_U >= 0.12) and (s1 - s2 >= 5.0) and (c2 <= 1.06 * c1):
            why = (f"Se elige **FO2 (Utilización)**: reduce U en {100*mejora_rel_U:.1f}% "
                   f"y s en {s1 - s2:.1f} mm, con ΔCosto {100*(c2/c1-1):.1f}% ≤ 6%.")
            return fo2, "FO2 (Utilización mínima)", why
        return fo1, "FO1 (Costo mínimo)", "Se elige **FO1**: el ahorro de costo domina; mejoras de U/s no justifican el sobrecosto."

    chosen, tag, why = decidir(fo1, fo2)

    # Mensaje del modelo usado
    st.success(
        "Modelo de capacidad usado: "
        + ("**ML** (paper)" + (f" — RMSE≈ {RMSE:,.2f} kPa" if RMSE else "")
           if (use_ml and (ML_MODEL is not None) and SKLEARN_OK) else "**Meyerhof** (clásico)")
    )

    cA, cB = st.columns(2)
    with cA:
        st.subheader("FO1 · Mínimo costo")
        st.write(fo1[["B", "L", "h", "qserv", "qadm", "U", "qmax", "s_mm", "costo", "margen"]])
    with cB:
        st.subheader("FO2 · Utilización mínima")
        st.write(fo2[["B", "L", "h", "qserv", "qadm", "U", "qmax", "s_mm", "costo", "margen"]])

    st.markdown("### ✅ Recomendación")
    st.write(
        f"**{tag}**  \n"
        f"B = **{chosen.B:.2f} m**, L = **{chosen.L:.2f} m**, h = **{chosen.h:.2f} m**  \n"
        f"q_serv = **{chosen.qserv:.1f} kPa** ≤ q_adm = **{chosen.qadm:.1f} kPa**; "
        f"q_max = **{chosen.qmax:.1f} kPa** ≤ q_adm  \n"
        f"s = **{chosen.s_mm:.1f} mm** ≤ s_adm = **{s_adm_mm:.0f} mm**  \n"
        f"U = **{chosen.U:.3f}**; Costo ≈ **S/ {chosen.costo:,.2f}**; Margen = **{chosen.margen:.1f} kPa**  \n"
        f"**Motivo:** {why}"
    )

    st.download_button(
        "Descargar soluciones (CSV)",
        df.to_csv(index=False),
        "soluciones_minimal.csv",
        "text/csv",
        use_container_width=True,
    )

