
# ----------------------------
# Optimización de Cimentaciones (vista compacta)
# ----------------------------
import math
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")

# ---------- UTILIDADES ----------

PALETTE = px.colors.sequential.Tealgrn
INFO = "Ingresa los datos en la parte izquierda y pulsa Analizar y optimizar o Optimizar con GA."

DEFAULTS = {
    # suelo demo segura (arcilla blanda)
    "preset": "Arcilla blanda (γ=17, c=25, φ=0)",
    "gamma": 17.0,
    "c": 25.0,
    "phi": 0.0,
    "D": 1.5,
    "N": 1000.0,
    "FS": 2.5,
    "concreto": 650.0,
    "acero": 5.5,
    # rangos demo seguros
    "Bmin": 1.2, "Bmax": 3.2, "nB": 30,
    "Lmin": 1.6, "Lmax": 4.2, "nL": 30,
    "hmin": 0.50,"hmax": 1.10,"nh": 10,
}

PRESETS = {
    "Arcilla blanda (γ=17, c=25, φ=0)": dict(gamma=17.0, c=25.0, phi=0.0),
    "Arena densa (γ=19, c=0, φ=35)":    dict(gamma=19.0, c=0.0,  phi=35.0),
    "Limo (γ=18, c=15, φ=25)":          dict(gamma=18.0, c=15.0, phi=25.0),
}

def set_demo_defaults():
    st.session_state.update(DEFAULTS)
    st.rerun()

def auto_adjust_ranges():
    """Si no hay soluciones, ampliar un poco rangos y bajar FS para mostrar algo."""
    st.session_state["Bmax"] = st.session_state.get("Bmax", DEFAULTS["Bmax"]) + 0.5
    st.session_state["Lmax"] = st.session_state.get("Lmax", DEFAULTS["Lmax"]) + 0.5
    st.session_state["hmax"] = min(1.60, st.session_state.get("hmax", DEFAULTS["hmax"]) + 0.20)
    st.session_state["FS"] = max(1.8, st.session_state.get("FS", DEFAULTS["FS"]) - 0.2)
    st.rerun()

@st.cache_data(show_spinner=False)
def bearing_capacity_factors(phi_deg: float):
    # Aproximaciones clásicas (OK para comparaciones)
    phi = math.radians(phi_deg)
    if phi_deg <= 0.0:
        return 5.7, 1.0, 0.0
    Nq = math.e**(math.pi*math.tan(phi)) * (math.tan(math.radians(45.0)+phi/2.0))**2
    Nc = (Nq - 1.0)/math.tan(phi)
    Ngamma = 2.0 * (Nq+1.0) * math.tan(phi)
    return Nc, Nq, Ngamma

def q_adm_Terzaghi(c, phi, gamma, D, B, FS):
    Nc, Nq, Ngamma = bearing_capacity_factors(phi)
    qu = c*Nc + gamma*D*Nq + 0.5*gamma*B*Ngamma
    return qu/FS

def costo(B, L, h, concreto_Sm3, acero_Skg):
    vol = B*L*h
    acero_kg = 35.0 * B * L  # regla simple (kg/m2), ajusta si quieres
    return vol*concreto_Sm3 + acero_kg*acero_Skg

def grid_optimize(modelo, c, phi, gamma, D, FS, N,
                  Bmin, Bmax, nB, Lmin, Lmax, nL, hmin, hmax, nh,
                  concreto_Sm3, acero_Skg):
    Bs = np.linspace(Bmin, Bmax, nB)
    Ls = np.linspace(Lmin, Lmax, nL)
    hs = np.linspace(hmin, hmax, nh)

    rows = []
    best = None
    for B in Bs:
        for L in Ls:
            qadm = q_adm_Terzaghi(c, phi, gamma, D, B, FS)
            qreq = N/(B*L)
            ok = (qadm >= qreq)
            for h in hs:
                cost = costo(B, L, h, concreto_Sm3, acero_Skg)
                rows.append((B, L, h, qadm, qreq, cost, ok))
                if ok:
                    if (best is None) or (cost < best["cost"]):
                        best = dict(B=B, L=L, h=h, qadm=qadm, qreq=qreq, cost=cost)

    df = pd.DataFrame(rows, columns=["B","L","h","q_adm","q_req","costo","ok"])
    df_ok = df[df["ok"]].copy()
    if best is None:
        return None, df, df_ok
    return best, df, df_ok

def df_to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="resultados")
    return output.getvalue()

def quick_pdf(best, params):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Reporte de Cimentación - Resumen")
    y -= 30
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Modelo: {params['modelo']}")
    y -= 16
    c.drawString(50, y, f"Suelo: γ={params['gamma']} kN/m³, c={params['c']} kPa, φ={params['phi']}°  |  D={params['D']} m, FS={params['FS']}")
    y -= 24
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Óptimo:")
    y -= 18
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"B={best['B']:.2f} m, L={best['L']:.2f} m, h={best['h']:.2f} m")
    y -= 16
    c.drawString(50, y, f"q_req={best['qreq']:.1f} kPa,  q_adm={best['qadm']:.1f} kPa")
    y -= 16
    c.drawString(50, y, f"Costo ≈ S/ {best['cost']:.0f}")
    c.showPage()
    c.save()
    return buf.getvalue()

# ---------- UI SUPERIOR ----------
st.title("Optimización de Cimentaciones")
st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta")
st.info(INFO)

# Botón DEMO arriba (antes de crear widgets)
top_cols = st.columns([1,1,4,1])
with top_cols[0]:
    if st.button("Cargar demo segura", use_container_width=True):
        set_demo_defaults()

with top_cols[1]:
    if st.button("Restablecer", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

st.markdown("### Parámetros de entrada")
c1, c2, c3 = st.columns(3)

# ---- COLUMNA 1: modelo, preset, suelo
with c1:
    modelo = st.selectbox(
        "Modelo de capacidad",
        ["Terzaghi (recomendado)"],
        index=0,
        key="modelo"
    )
    preset = st.selectbox(
        "Preset de suelo (rápido)",
        list(PRESETS.keys()),
        index=list(PRESETS.keys()).index(st.session_state.get("preset", DEFAULTS["preset"]))
        if "preset" in st.session_state else 0,
        key="preset"
    )
    # si cambia preset, actualiza γ, c, φ:
    p = PRESETS[preset]
    if ("gamma" not in st.session_state) or (st.session_state.get("preset_changed_once") is True):
        pass
    # Widgets para γ, c, φ en C2 pero set defaults acá:
    st.session_state.setdefault("gamma", p["gamma"])
    st.session_state.setdefault("c", p["c"])
    st.session_state.setdefault("phi", p["phi"])

# ---- COLUMNA 2: γ, c, φ
with c2:
    gamma = st.number_input("Peso unitario γ (kN/m³)", 12.0, 23.0,
                            value=float(st.session_state.get("gamma", DEFAULTS["gamma"])), step=0.5, key="gamma")
    c = st.number_input("Cohesión c (kPa)", 0.0, 300.0,
                        value=float(st.session_state.get("c", DEFAULTS["c"])), step=1.0, key="c")
    phi = st.number_input("Ángulo de fricción φ (°)", 0.0, 45.0,
                          value=float(st.session_state.get("phi", DEFAULTS["phi"])), step=1.0, key="phi")

# ---- COLUMNA 3: D, N, FS
with c3:
    D = st.number_input("Profundidad D (m)", 0.5, 5.0,
                        value=float(st.session_state.get("D", DEFAULTS["D"])), step=0.1, key="D")
    N = st.number_input("Carga N (kN)", 100.0, 10000.0,
                        value=float(st.session_state.get("N", DEFAULTS["N"])), step=50.0, key="N")
    FS = st.number_input("Factor de seguridad", 1.2, 4.0,
                         value=float(st.session_state.get("FS", DEFAULTS["FS"])), step=0.1, key="FS")

st.markdown("---")

# COSTOS y RANGOS en dos columnas
cc1, cc2 = st.columns(2)

with cc1:
    st.subheader("Costos")
    concreto_Sm3 = st.number_input("Concreto (S/ por m³)", 300.0, 1500.0,
                                   value=float(st.session_state.get("concreto", DEFAULTS["concreto"])),
                                   step=10.0, key="concreto")
    acero_Skg = st.number_input("Acero (S/ por kg)", 2.0, 20.0,
                                value=float(st.session_state.get("acero", DEFAULTS["acero"])),
                                step=0.1, key="acero")

with cc2:
    st.subheader("Rangos de diseño (B, L, h)")
    Bmin = st.slider("Base B (m)", 0.8, 4.0,
                     (float(st.session_state.get("Bmin", DEFAULTS["Bmin"])),
                      float(st.session_state.get("Bmax", DEFAULTS["Bmax"]))),
                     0.05, key="B_range")
    Lmin = st.slider("Largo L (m)", 1.0, 5.0,
                     (float(st.session_state.get("Lmin", DEFAULTS["Lmin"])),
                      float(st.session_state.get("Lmax", DEFAULTS["Lmax"]))),
                     0.05, key="L_range")
    hmin = st.slider("Altura h (m)", 0.30, 1.60,
                     (float(st.session_state.get("hmin", DEFAULTS["hmin"])),
                      float(st.session_state.get("hmax", DEFAULTS["hmax"]))),
                     0.02, key="h_range")

    # Resoluciones
    col_res = st.columns(3)
    with col_res[0]:
        nB = st.number_input("Resolución B", 5, 60, value=int(st.session_state.get("nB", DEFAULTS["nB"])), key="nB")
    with col_res[1]:
        nL = st.number_input("Resolución L", 5, 60, value=int(st.session_state.get("nL", DEFAULTS["nL"])), key="nL")
    with col_res[2]:
        nh = st.number_input("Resolución h", 3, 40, value=int(st.session_state.get("nh", DEFAULTS["nh"])), key="nh")

# Guardamos en session (para demo/auto-ajuste)
st.session_state.update({
    "Bmin": Bmin[0], "Bmax": Bmin[1],
    "Lmin": Lmin[0], "Lmax": Lmin[1],
    "hmin": hmin[0], "hmax": hmin[1],
})

# ---------- BOTONES DE EJECUCIÓN ----------
btns = st.columns([1,1,3,1])
run_bruteforce = btns[0].button("🔍 Analizar y optimizar", use_container_width=True)
run_ga         = btns[1].button("🧬 Optimizar con GA", use_container_width=True)

# ---------- OPTIMIZACIÓN ----------
best = None
df = None
df_ok = None

if run_bruteforce or run_ga:
    with st.spinner("Buscando soluciones..."):
        best, df, df_ok = grid_optimize(
            modelo, c, phi, gamma, D, FS, N,
            st.session_state["Bmin"], st.session_state["Bmax"], nB,
            st.session_state["Lmin"], st.session_state["Lmax"], nL,
            st.session_state["hmin"], st.session_state["hmax"], nh,
            concreto_Sm3, acero_Skg
        )

    if best is None:
        st.warning("No se encontraron soluciones que cumplan la capacidad admisible.")
        sugg = f"Prueba con B y L mayores, φ o c más altos, FS menor o carga menor."
        st.write(sugg)
        if st.button("✨ Auto-ajustar y reintentar"):
            auto_adjust_ranges()
    else:
        # ------- Encabezado de resultados -------
        st.success("Diseño óptimo encontrado")
        met = st.columns(4)
        met[0].metric("B (m)", f"{best['B']:.2f}")
        met[1].metric("L (m)", f"{best['L']:.2f}")
        met[2].metric("h (m)", f"{best['h']:.2f}")
        met[3].metric("Costo (S/)", f"{best['cost']:.0f}")

        # ------- Gráficos -------
        colg1, colg2 = st.columns(2, gap="large")

        with colg1:
            fig_bar = px.bar(pd.DataFrame({
                "Tipo":["q_req","q_adm"],
                "kPa":[best["qreq"], best["qadm"]]
            }), x="Tipo", y="kPa", text="kPa", color="Tipo",
            color_discrete_sequence=[PALETTE[4], PALETTE[7]])
            fig_bar.update_layout(margin=dict(l=10,r=10,t=40,b=10), showlegend=False)
            fig_bar.update_traces(texttemplate="%{y:.1f}", textposition="outside")
            st.subheader("q_req vs q_adm (óptimo)")
            st.plotly_chart(fig_bar, use_container_width=True)

        with colg2:
            df_ok_plot = df_ok.copy()
            df_ok_plot["Costo (S/)"] = df_ok_plot["costo"]
            fig_sc = px.scatter(
                df_ok_plot, x="B", y="L", size="h", color="Costo (S/)",
                color_continuous_scale=PALETTE, title="Candidatos válidos (color=costo, tamaño=h)"
            )
            fig_sc.update_layout(margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("### Resumen (JSON)")
        st.code(pd.Series({
            "Modelo":"Terzaghi (recomendado)",
            "B (m)":round(best["B"],2),
            "L (m)":round(best["L"],2),
            "h (m)":round(best["h"],2),
            "q_adm (kPa)":round(best["qadm"],1),
            "q_req (kPa)":round(best["qreq"],1),
            "Costo (S/)":round(best["cost"])
        }).to_json(indent=2, force_ascii=False), language="json")

        st.markdown("### Recomendaciones")
        st.write("✅ Buen diseño: margen suficiente entre capacidad y demanda.")
        st.write(f"💡 Óptimo actual: **S/ {best['cost']:.0f}**. Si buscas más rigidez, evalúa **h** ligeramente mayor.")
        st.write("📚 Referencias: Terzaghi & Peck; Meyerhof; Vesic (capacidad portante clásica).")

        st.markdown("### Tabla de candidatos válidos")
        st.dataframe(df_ok.sort_values("costo").head(200), use_container_width=True)

        # ---------- DESCARGAS ----------
        st.markdown("### Descargas")
        col_dwn = st.columns(3)
        with col_dwn[0]:
            xls = df_to_excel_bytes(df_ok.sort_values("costo"))
            st.download_button("📥 Descargar Excel", xls, "resultados.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

        with col_dwn[1]:
            pdf = quick_pdf(best, dict(modelo="Terzaghi", gamma=gamma, c=c, phi=phi, D=D, FS=FS))
            st.download_button("📄 Descargar reporte (PDF)", pdf, "reporte.pdf",
                               "application/pdf", use_container_width=True)

# Mensaje inicial si no se ha ejecutado nada
if (best is None) and not (run_bruteforce or run_ga):
    st.info("Configura los parámetros y pulsa **Analizar y optimizar**.")
