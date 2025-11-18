# app.py — 2 funciones objetivo (FO1 costo, FO2 asentamiento) con NF, SPT y ML opcional
# Verifica: q_serv ≤ q_adm, q_max ≤ q_adm y s ≤ s_adm
# Considera: nivel freático alto (γ′ y σ′v por tramos), tipo de suelo, SPT para q_ult y Es.
# ML opcional: si hay modelo entrenado usa ML (auto); si no, usa método clásico/SPT.

import math
import numpy as np
import pandas as pd
import streamlit as st

# ====== ML opcional (importación defensiva) ======
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False
    train_test_split = GradientBoostingRegressor = None
    mean_squared_error = mean_absolute_error = r2_score = None

st.set_page_config(page_title="Optimización de Cimentaciones — Minimal", layout="centered")
st.title("Optimización de Cimentaciones (mínima)")
st.caption("NF alto + SPT opcional + 2 FO (costo, asentamiento) con verificaciones geotécnicas y ML opcional.")

# ======================== Entradas mínimas ========================
c0, c1, c2 = st.columns(3)
with c0:
    suelo_tipo = st.selectbox("Tipo de suelo", ["Arena (no cohesiva)", "Arcilla (cohesiva)"])
    usar_spt_cap = st.checkbox("Usar SPT (N60) para capacidad", value=False)
    usar_spt_Es  = st.checkbox("Usar SPT (N60) para Es si no se especifica", value=True)
with c1:
    N = st.number_input("Carga axial N (kN)", 100.0, 2e5, 800.0, 10.0)
    phi_in = st.number_input("ϕ (°) (si no SPT-Arena)", 0.0, 45.0, 32.0, 0.5)
    Es_in  = st.number_input("Eₛ entrada (kPa) (si no SPT)", 3000.0, 200000.0, 25000.0, 500.0)
with c2:
    D = st.number_input("Profundidad base D (m)", 0.5, 10.0, 1.5, 0.1)
    FS = st.number_input("FS (capacidad)", 1.5, 4.0, 2.5, 0.1)
    L_over_B = st.number_input("Relación L/B", 0.8, 2.0, 1.0, 0.05)

# ===== Nivel freático y pesos unitarios =====
st.markdown("### Nivel freático y pesos unitarios")
c_nf1, c_nf2, c_nf3 = st.columns(3)
with c_nf1:
    z_NF = st.number_input("Profundidad del nivel freático z_NF (m)", 0.0, 10.0, 100.0, 0.1)
with c_nf2:
    gamma_moist = st.number_input("γ húmedo (kN/m³) (sobre NF)", 14.0, 22.0, 18.0, 0.1)
with c_nf3:
    gamma_sat = st.number_input("γ saturado (kN/m³) (bajo NF)", 16.0, 24.0, 20.0, 0.1)

GAMMA_W = 9.81  # kN/m³

def gamma_eff_at_depth(D, z_NF, gamma_moist, gamma_sat):
    """γ' efectivo en la base: húmedo si D<z_NF; saturado-sumergido si D≥z_NF."""
    return gamma_moist if D < z_NF else max(gamma_sat - GAMMA_W, 0.1)

def q_overburden_effective(D, z_NF, gamma_moist, gamma_sat):
    """σ′v(D): integra por tramos húmedo/sumergido hasta D."""
    if D <= 0: return 0.0
    if D <= z_NF:
        return gamma_moist * D
    else:
        return gamma_moist * z_NF + max(gamma_sat - GAMMA_W, 0.1) * (D - z_NF)

# ====== SPT opcional ======
st.markdown("### Parámetro SPT (opcional)")
c_spt1, c_spt2 = st.columns(2)
with c_spt1:
    N60 = st.number_input("N₆₀ (golpes corregidos)", 0.0, 100.0, 25.0, 1.0)
with c_spt2:
    # Correlaciones docentes (simplificadas y conservadoras)
    phi_spt = min(45.0, 27.0 + 0.3 * N60)  # Arenas: φ ≈ 27 + 0.3*N60 (tope 45°)
    cu_spt  = 5.0 * N60                     # Arcillas: cu ≈ 5 N60 (kPa)
    st.caption(f"φ(SPT)≈{phi_spt:.1f}° (arenas),  cu(SPT)≈{cu_spt:.0f} kPa (arcillas)")

# ====== Rango de búsqueda ======
st.markdown("### Rangos de búsqueda (B y h)")
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
st.subheader("ML opcional (paper) — auto-ML")
if not SKLEARN_OK:
    st.warning("Para usar ML, agrega `scikit-learn>=1.3,<1.5` en requirements.txt. "
               "Si no, se usará el método clásico/SPT.")

if "ML_MODEL" not in st.session_state:
    st.session_state.ML_MODEL = None
if "ML_METRICS" not in st.session_state:
    st.session_state.ML_METRICS = {}

up = st.file_uploader("CSV entrenamiento (gamma_eff,B,D,phi,L_over_B,qu)", type=["csv"])

def train_ml(csv):
    df = pd.read_csv(csv)
    req = ["gamma_eff", "B", "D", "phi", "L_over_B", "qu"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan columnas: {miss}")

    X = df[["gamma_eff", "B", "D", "phi", "L_over_B"]].astype(float)
    y = df["qu"].astype(float)

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.20, random_state=42)

    model = GradientBoostingRegressor(
        random_state=42,
        n_estimators=5000, learning_rate=0.05,
        max_depth=3, min_samples_leaf=3,
    )
    # Ajuste simple; si quieres early stopping interno, usa validation_fraction + n_iter_no_change
    model.fit(Xtr, ytr)

    yhat = model.predict(Xva)
    rmse = float(mean_squared_error(yva, yhat, squared=False))
    mae  = float(mean_absolute_error(yva, yhat))
    try:
        r2 = float(r2_score(yva, yhat))
    except Exception:
        r2 = float("nan")
    med = float(np.median(yva)) if len(yva) else float("nan")
    nrmse_med = float(rmse / med * 100) if np.isfinite(med) and med != 0 else float("nan")
    bias = float(np.mean(yhat - yva))
    st.session_state.ML_METRICS = {
        "rmse": rmse, "mae": mae, "r2": r2,
        "nrmse_med": nrmse_med, "bias": bias, "n_val": int(len(yva))
    }
    return model

if st.button("Entrenar modelo", use_container_width=True):
    if not SKLEARN_OK:
        st.error("scikit-learn no está disponible en el entorno.")
    elif up is None:
        st.warning("Sube un CSV válido con columnas: gamma_eff,B,D,phi,L_over_B,qu.")
    else:
        try:
            st.session_state.ML_MODEL = train_ml(up)
            m = st.session_state.ML_METRICS
            st.success(
                "Modelo entrenado (GBDT). "
                f"RMSE≈ {m['rmse']:.2f} kPa | MAE≈ {m['mae']:.2f} kPa | "
                f"R²≈ {m['r2']:.3f} | nRMSE≈ {m['nrmse_med']:.1f}% | "
                f"Sesgo≈ {m['bias']:+.2f} kPa"
            )
        except Exception as e:
            st.error(f"Error entrenando: {e}")

if st.session_state.ML_METRICS:
    m = st.session_state.ML_METRICS
    st.markdown("### Métricas del modelo (validación)")
    c1m, c2m, c3m, c4m, c5m = st.columns(5)
    c1m.metric("R²", f"{m['r2']:.3f}" if np.isfinite(m['r2']) else "—")
    c2m.metric("RMSE", f"{m['rmse']:.2f} kPa")
    c3m.metric("MAE", f"{m['mae']:.2f} kPa")
    c4m.metric("nRMSE/Mediana", f"{m['nrmse_med']:.1f}%" if np.isfinite(m['nrmse_med']) else "—")
    c5m.metric("Sesgo", f"{m['bias']:+.2f} kPa")
    st.caption(f"Conjunto de validación: n = {m['n_val']}")

st.markdown("---")

# ======================== Capacidad última (clásico/SPT) ========================
def bearing_factors(phi_deg: float):
    phi_rad = math.radians(phi_deg)
    if phi_rad < 1e-6:
        Nq, Nc, Ng = 1.0, 5.14, 0.0
    else:
        Nq = math.e ** (math.pi * math.tan(phi_rad)) * (math.tan(math.pi/4 + phi_rad/2)) ** 2
        Nc = (Nq - 1.0) / math.tan(phi_rad)
        Ng = 2.0 * (Nq + 1.0) * math.tan(phi_rad)
    return Nc, Nq, Ng

def qult_meyerhof_general(B, D, phi, gamma_eff_base, q_eff):
    """Meyerhof rectangular c≈0, usando γ′ en base y σ′v(D)=q_eff."""
    Nc, Nq, Ng = bearing_factors(phi)
    sc, sq, sg = 1.3, 1.2, 1.0
    return q_eff * Nq * sq + 0.5 * gamma_eff_base * B * Ng * sg

def qult_clay_undrained(B, cu):
    """Capacidad última no drenada (φ≈0), zapata rectangular: qult ≈ Nc*cu*sc (Nc≈5.14, sc≈1.3)."""
    Nc = 5.14; sc = 1.3
    return Nc * cu * sc

def qult_pred(gamma_eff_base, q_eff, B, D, phi_val, L_over_B_val, suelo_tipo, usar_spt_cap, N60):
    """
    Predicción de qu:
    - Si hay ML entrenado, usa ML con gamma_eff como entrada.
    - Si no, usa:
        * Arena: Meyerhof con φ = φ_in o φ(SPT) si usar_spt_cap.
        * Arcilla: qult = Nc*cu*sc si usar_spt_cap; si no, Meyerhof con φ_in (si se quisiera).
    """
    if (st.session_state.ML_MODEL is not None) and SKLEARN_OK:
        X = pd.DataFrame([{
            "gamma_eff": gamma_eff_base, "B": B, "D": D, "phi": phi_val, "L_over_B": L_over_B_val
        }])
        return float(st.session_state.ML_MODEL.predict(X)[0])

    if suelo_tipo.startswith("Arena"):
        phi_use = (min(45.0, 27.0 + 0.3 * N60) if usar_spt_cap else phi_val)
        return qult_meyerhof_general(B, D, phi_use, gamma_eff_base, q_eff)

    # Arcilla (no drenado) preferimos SPT->cu; si no se usa SPT, podríamos permitir φ_in≈0 (conservador)
    if usar_spt_cap:
        cu = 5.0 * N60  # kPa (conservador)
        return qult_clay_undrained(B, cu)
    else:
        # Si no SPT, permitir φ_in pequeño (p.ej. 0–5°) o usa c≈0 con φ_in dado (flexible para tareas docentes)
        return qult_meyerhof_general(B, D, phi_val, gamma_eff_base, q_eff)

# ======================== Servicio y asentamiento =========================
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

def Es_effective(Es_in, suelo_tipo, is_submerged, N60, usar_spt_Es):
    """
    Es base:
      - Si usar_spt_Es: arenas Es≈2500*N60 kPa; arcillas Es≈1000*N60 kPa (docente/conservador).
      - Si no: Es_in (del usuario).
    Ajuste por saturación (base bajo NF): arenas ×0.70; arcillas ×0.85.
    """
    if usar_spt_Es and N60 > 0:
        Es0 = 2500.0 * N60 if suelo_tipo.startswith("Arena") else 1000.0 * N60
    else:
        Es0 = Es_in
    if is_submerged:
        Es0 *= 0.70 if suelo_tipo.startswith("Arena") else 0.85
    return max(Es0, 1e3)

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

# ======================== Optimización principal ====================
st.markdown("---")
if st.button("🚀 Optimizar (FO1 & FO2)"):
    Bs = np.linspace(Bmin, Bmax, int(nB))
    hs = np.linspace(hmin, hmax, int(nh))

    rows = []
    for B in Bs:
        L = L_over_B * B
        # Propiedades en base considerando NF
        gamma_eff_base = gamma_eff_at_depth(D, z_NF, gamma_moist, gamma_sat)
        q_eff = q_overburden_effective(D, z_NF, gamma_moist, gamma_sat)

        # Parámetros de resistencia para predicción clásica/SPT
        phi_use = phi_in if suelo_tipo.startswith("Arena") else max(0.0, min(phi_in, 10.0))  # arcilla ~φ pequeño si no SPT

        for h in hs:
            qu = qult_pred(gamma_eff_base, q_eff, B, D, phi_use, L_over_B, suelo_tipo, usar_spt_cap, N60)
            qadm = qu / FS
            qserv, qmax = contact_pressures(N, B, L)
            if not (qserv <= qadm and qmax <= qadm):
                continue
            # Asentamiento con Es efectivo y efecto NF
            submerged = D >= z_NF
            Es_use = Es_effective(Es_in, suelo_tipo, submerged, N60, usar_spt_Es)
            s = settlement_mm(qserv, B, Es_use)
            s_adm_mm = st.session_state.get("s_adm", None)  # compat si quieres usarlo desde fuera
            # Tomamos s_adm_mm del input de arriba (no en session_state)
            # (lo volvemos a obtener directamente)
            # (esta variable ya existe arriba como input)
            if s > s_adm_mm:
                continue

            costo = cost_S(B, L, h, D=D)
            rows.append([B, L, h, qu, qadm, qserv, qmax, s, costo, gamma_eff_base, q_eff, Es_use])

    if not rows:
        st.error("Sin soluciones factibles con los parámetros dados. Revisa FS, rangos B/h, NF o parámetros SPT.")
        st.stop()

    df = pd.DataFrame(rows, columns=[
        "B", "L", "h", "qu", "qadm", "qserv", "qmax", "s_mm", "costo",
        "gamma_eff_base", "q_eff", "Es_efectivo"
    ])

    # FO1: costo mínimo
    fo1 = df.loc[df["costo"].idxmin()]
    # FO2: asentamiento mínimo
    fo2 = df.loc[df["s_mm"].idxmin()]

    # ===== Banner del modelo usado =====
    if (st.session_state.ML_MODEL is not None) and SKLEARN_OK:
        m = st.session_state.ML_METRICS
        st.success(
            "Modelo de capacidad usado: **ML (paper, auto)** — "
            f"RMSE≈ {m.get('rmse', float('nan')):.2f} kPa | "
            f"MAE≈ {m.get('mae', float('nan')):.2f} kPa | "
            f"R²≈ {m.get('r2', float('nan')):.3f} | "
            f"nRMSE≈ {m.get('nrmse_med', float('nan')):.1f}% | "
            f"Sesgo≈ {m.get('bias', float('nan')):+.2f} kPa"
        )
    else:
        st.success("Modelo de capacidad usado: **Clásico/SPT (Meyerhof/undrained)**")

    # ===== Mostrar FO1 y FO2 =====
    cA, cB = st.columns(2)
    with cA:
        st.subheader("FO1 · Mínimo costo")
        st.table(fo1[["B","L","h","qserv","qadm","qmax","s_mm","costo","gamma_eff_base","q_eff","Es_efectivo"]])
    with cB:
        st.subheader("FO2 · Mínimo asentamiento")
        st.table(fo2[["B","L","h","qserv","qadm","qmax","s_mm","costo","gamma_eff_base","q_eff","Es_efectivo"]])

    # ===== Recomendación =====
    def recomendar(fo1, fo2):
        s1, s2 = fo1.s_mm, fo2.s_mm
        c1, c2 = fo1.costo, fo2.costo
        if (s1 - s2 >= 5.0) and (c2 <= 1.05 * c1):
            tag = "FO2 (mínimo asentamiento)"
            return fo2, tag, f"Se elige FO2 por menor s (−{s1 - s2:.1f} mm) con costo ≤ +5%."
        tag = "FO1 (mínimo costo)"
        return fo1, tag, "Se elige FO1: ahorro de costo domina; verificaciones se cumplen."

    chosen, tag, why = recomendar(fo1, fo2)

    st.markdown("## ✅ Recomendación")
    st.write(
        f"**{tag}**  \n"
        f"B = **{chosen.B:.2f} m**, L = **{chosen.L:.2f} m**, h = **{chosen.h:.2f} m**  \n"
        f"q_serv = **{chosen.qserv:.1f} kPa** ≤ q_adm = **{chosen.qadm:.1f} kPa**; "
        f"q_max = **{chosen.qmax:.1f} kPa** ≤ q_adm  \n"
        f"s = **{chosen.s_mm:.1f} mm** ≤ s_adm = **{s_adm_mm:.0f} mm**  \n"
        f"γ′ (base) = **{chosen.gamma_eff_base:.2f} kN/m³**, σ′v(D) = **{chosen.q_eff:.1f} kPa**, "
        f"Eₛ,ef = **{chosen.Es_efectivo:.0f} kPa**  \n"
        f"Costo ≈ **S/ {chosen.costo:,.2f}**  \n"
        f"**Motivo:** {why}"
    )

    st.download_button(
        "Descargar soluciones (CSV)",
        df.to_csv(index=False),
        "soluciones_cimentacion_nf_spt.csv",
        "text/csv",
        use_container_width=True,
    )
