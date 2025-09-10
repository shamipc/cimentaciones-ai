
  
          # -*- coding: utf-8 -*-
import math
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# -----------------------------
# Utilidades comunes
# -----------------------------
PALETTE = ["#7AA6C2", "#6BC1A3", "#C0D6DF", "#9AD1D4", "#85C7DE", "#A6E3E9"]

def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Resultados") -> bytes | None:
    """Exporta DataFrame a Excel intentando xlsxwriter y luego openpyxl, sin romper si no existen."""
    engine = None
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            engine = "openpyxl"
        except Exception:
            engine = None

    if engine is None:
        return None

    out = BytesIO()
    with pd.ExcelWriter(out, engine=engine) as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        # Auto-anchos
        for i, col in enumerate(df.columns):
            width = max(10, min(35, df[col].astype(str).map(len).max() + 3))
            try:
                ws.set_column(i, i, width)           # xlsxwriter
            except Exception:
                try:
                    ws.column_dimensions[chr(65+i)].width = width  # openpyxl
                except Exception:
                    pass
    out.seek(0)
    return out.getvalue()


def generar_pdf_resumen(param, optimo, top10_df):
    """PDF simple con parámetros, óptimo y top 10."""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    y = H - 2*cm

    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, y, "Reporte de Optimización de Cimentaciones")
    y -= 0.8*cm

    c.setFont("Helvetica", 9)
    c.drawString(2*cm, y, f"Fecha: {datetime.now():%Y-%m-%d %H:%M}")
    y -= 0.5*cm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(2*cm, y, "Parámetros de entrada")
    y -= 0.5*cm
    c.setFont("Helvetica", 9)
    lines = [
        f"Modelo: {param['modelo']}",
        f"γ = {param['gamma']} kN/m³   c = {param['c']} kPa   φ = {param['phi']}°",
        f"D = {param['D']} m   N = {param['N']} kN   FS = {param['FS']}",
        f"C. concreto = S/ {param['c_concreto']} m³   C. acero = S/ {param['c_acero']} kg",
        f"Rangos  B: {param['Bmin']}–{param['Bmax']} m | L: {param['Lmin']}–{param['Lmax']} m | h: {param['hmin']}–{param['hmax']} m",
    ]
    for t in lines:
        c.drawString(2*cm, y, t); y -= 0.45*cm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(2*cm, y, "Diseño óptimo")
    y -= 0.5*cm
    c.setFont("Helvetica", 9)
    if optimo is None:
        c.drawString(2*cm, y, "No se encontraron soluciones válidas.")
        y -= 0.4*cm
    else:
        c.drawString(2*cm, y, f"B={optimo['B']} m  L={optimo['L']} m  h={optimo['h']} m")
        y -= 0.4*cm
        c.drawString(2*cm, y, f"q_adm={optimo['q_adm']} kPa  q_req={optimo['q_req']} kPa  Costo=S/ {optimo['costo']}")
        y -= 0.6*cm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(2*cm, y, "Top 10 soluciones")
    y -= 0.5*cm
    c.setFont("Helvetica", 8)
    if top10_df is None or top10_df.empty:
        c.drawString(2*cm, y, "— (vacío) —")
    else:
        for i in range(min(10, len(top10_df))):
            r = top10_df.iloc[i]
            c.drawString(2*cm, y,
                f"{i+1:>2}. B={r['B']:.2f}  L={r['L']:.2f}  h={r['h']:.2f}  "
                f"q_adm={r['q_adm']:.1f}  q_req={r['q_req']:.1f}  Costo=S/ {r['costo']:.0f}"
            )
            y -= 0.38*cm

    c.showPage(); c.save()
    buf.seek(0)
    return buf.getvalue()


# -----------------------------
# Modelos de capacidad (simple)
# -----------------------------
def _factors_from_phi(phi):
    """Factores N aproximados a partir de φ (rad)."""
    phir = math.radians(phi)
    if phi == 0:
        return 5.14, 1.0, 0.0  # Nc, Nq, Nγ
    Nq = math.e ** (math.pi * math.tan(phir)) * (math.tan(math.pi/4 + phir/2) ** 2)
    Nc = (Nq - 1) / math.tan(phir)
    Ngamma = 2 * (Nq + 1) * math.tan(phir)
    return Nc, Nq, Ngamma

def qult_terzaghi(c, q, gamma, B, phi):
    Nc, Nq, Ng = _factors_from_phi(phi)
    return c*Nc + q*Nq + 0.5*gamma*B*Ng

def qult_meyerhof(c, q, gamma, B, phi):
    # Versión compacta: mismos N pero algo más conservador en Nγ
    Nc, Nq, Ng = _factors_from_phi(phi)
    Ng_m = 0.8 * Ng    # algo más conservador
    return c*Nc + q*Nq + 0.5*gamma*B*Ng_m


def costo_zapata(B, L, h, c_conc, c_acero):
    vol = B * L * h
    costo_conc = vol * c_conc
    peso_acero = vol * 80.0      # 80 kg/m³
    costo_ac = peso_acero * c_acero
    return costo_conc + costo_ac


def evaluar_grilla(params):
    """Barre B-L-h y devuelve soluciones válidas (q_req ≤ q_adm/FS)."""
    modelo = params["modelo"]
    gamma = params["gamma"]; c = params["c"]; phi = params["phi"]
    D = params["D"]; N = params["N"]; FS = params["FS"]
    c_conc = params["c_concreto"]; c_acero = params["c_acero"]

    Bvec = np.linspace(params["Bmin"], params["Bmax"], params["Bres"])
    Lvec = np.linspace(params["Lmin"], params["Lmax"], params["Lres"])
    hvec = np.linspace(params["hmin"], params["hmax"], params["hres"])

    q = gamma * D
    datos = []
    for B in Bvec:
        for L in Lvec:
            area = B * L
            if area <= 0:
                continue
            q_req = (N * 1000.0) / (area * 1000.0)  # kPa (kN/m²)
            for h in hvec:
                if modelo.startswith("Meyerhof"):
                    qult = qult_meyerhof(c, q, gamma, B, phi)
                else:
                    qult = qult_terzaghi(c, q, gamma, B, phi)
                q_adm = qult / FS
                if q_req <= q_adm:
                    costo = costo_zapata(B, L, h, c_conc, c_acero)
                    datos.append([B, L, h, area, q_adm, q_req, costo])

    df = pd.DataFrame(datos, columns=["B", "L", "h", "area", "q_adm", "q_req", "costo"])
    df = df.sort_values(["costo", "q_req"], ascending=[True, True]).reset_index(drop=True)
    return df


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")
st.title("Optimización de Cimentaciones")
st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta")

with st.sidebar:
    st.header("Parámetros de entrada")

    modelo = st.selectbox(
        "Modelo de capacidad",
        ["Terzaghi (recomendado)", "Meyerhof (φ>0)"],
        index=0,
    )

    # Presets (dejo 'Intermedio' por defecto para que veas resultados al primer clic)
    presets = {
        "Intermedio (γ=18, c=20, φ=28)": {"gamma": 18.0, "c": 20.0, "phi": 28.0, "idx": 0},
        "Arcilla blanda (γ=17, c=18, φ=0)": {"gamma": 17.0, "c": 18.0, "phi": 0.0, "idx": 1},
        "Arena densa (γ=19, c=0, φ=36)": {"gamma": 19.0, "c": 0.0, "phi": 36.0, "idx": 2},
    }
    preset_name = st.selectbox("Preset de suelo (rápido)", list(presets.keys()),
                               index=presets["Intermedio (γ=18, c=20, φ=28)"]["idx"])
    preset = presets[preset_name]

    gamma = st.number_input("Peso unitario γ (kN/m³)", 10.0, 24.0, preset["gamma"], step=0.5)
    c = st.number_input("Cohesión c (kPa)", 0.0, 200.0, preset["c"], step=1.0)
    phi = st.number_input("Ángulo de fricción φ (°)", 0.0, 45.0, preset["phi"], step=1.0)
    D = st.number_input("Profundidad D (m)", 0.0, 5.0, 1.50, step=0.1)
    N = st.number_input("Carga N (kN)", 10.0, 10000.0, 1000.0, step=10.0)
    FS = st.number_input("Factor de seguridad", 1.0, 5.0, 2.00, step=0.1)

    st.subheader("Costos")
    c_concreto = st.number_input("Concreto (S/ por m³)", 200.0, 2000.0, 650.0, step=10.0)
    c_acero = st.number_input("Acero (S/ por kg)", 1.0, 20.0, 5.50, step=0.1)

    st.subheader("Rangos de diseño")
    Bmin, Bmax = st.slider("Base B (m)", 1.20, 4.00, (1.40, 4.00), step=0.05)
    Bres = st.number_input("Resolución B (puntos)", 3, 120, 25, step=1)
    Lmin, Lmax = st.slider("Largo L (m)", 1.20, 4.00, (1.40, 4.00), step=0.05)
    Lres = st.number_input("Resolución L (puntos)", 3, 120, 25, step=1)
    hmin, hmax = st.slider("Altura h (m)", 0.50, 1.50, (0.60, 1.20), step=0.02)
    hres = st.number_input("Resolución h (puntos)", 3, 60, 8, step=1)

st.info("Ingresa los **datos** en la parte izquierda y pulsa **Analizar y optimizar**.")

params = dict(
    modelo=modelo, gamma=gamma, c=c, phi=phi, D=D, N=N, FS=FS,
    c_concreto=c_concreto, c_acero=c_acero,
    Bmin=Bmin, Bmax=Bmax, Bres=int(Bres),
    Lmin=Lmin, Lmax=Lmax, Lres=int(Lres),
    hmin=hmin, hmax=hmax, hres=int(hres),
)

run = st.sidebar.button("🔎 Analizar y optimizar", use_container_width=True)

if run:
    with st.spinner("Evaluando diseño..."):
        df_valid = evaluar_grilla(params)

    if df_valid.empty:
        st.warning(
            "No se encontraron soluciones que cumplan la capacidad admisible. "
            "Prueba con **B y L mayores**, **φ o c más altos**, **FS menor** o **carga menor**."
        )

        # ------- Auto-ajustar (sin tocar tus widgets) -------
        if st.button("✨ Auto-ajustar y reintentar"):
            sug = params.copy()
            sug["Bmax"] = min(5.0, params["Bmax"] + 0.5)
            sug["Lmax"] = min(5.0, params["Lmax"] + 0.5)
            sug["FS"] = max(1.5, params["FS"] - 0.25)
            with st.spinner("Buscando con rangos expandidos..."):
                df_valid = evaluar_grilla(sug)

            if df_valid.empty:
                st.info("Aún sin resultados. Amplía más los rangos o revisa la carga/FS/suelo.")
            else:
                st.success("Se encontraron candidatos con los rangos expandidos (demo).")
        else:
            st.caption("Sugerencia: prueba con B y L en 1.4–4.0 m y FS≈2.0 para una exploración rápida.")
    # -------------------------------------------------------
    if not df_valid.empty:
        # Óptimo
        opt = df_valid.iloc[0]
        optimo = dict(B=round(opt["B"], 2), L=round(opt["L"], 2), h=round(opt["h"], 2),
                      q_adm=round(opt["q_adm"], 1), q_req=round(opt["q_req"], 1),
                      costo=round(opt["costo"], 0))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("B (m)", f"{optimo['B']:.2f}")
        c2.metric("L (m)", f"{optimo['L']:.2f}")
        c3.metric("h (m)", f"{optimo['h']:.2f}")
        c4.metric("Costo (S/)", f"{optimo['costo']:.0f}")
        st.success("✅ Diseño óptimo encontrado")

        st.markdown("---")
        g1, g2 = st.columns(2)

        df_bar = pd.DataFrame({"Tipo": ["q_req", "q_adm"], "kPa": [optimo["q_req"], optimo["q_adm"]]})
        fig_bar = px.bar(df_bar, x="Tipo", y="kPa", color="Tipo",
                         color_discrete_sequence=PALETTE[:2], text="kPa")
        fig_bar.update_traces(texttemplate="%{y:.1f}", textposition="outside")
        fig_bar.update_layout(height=420, showlegend=False, margin=dict(l=20, r=10, t=50, b=10),
                              title="q_req vs q_adm (óptimo)")
        g1.plotly_chart(fig_bar, use_container_width=True)

        fig_sc = px.scatter(df_valid, x="B", y="L", color="costo", size="h",
                            color_continuous_scale="YlGnBu",
                            labels={"costo": "Costo (S/)"},
                            title="Candidatos válidos (color = Costo, tamaño = h)")
        fig_sc.update_layout(height=420, margin=dict(l=20, r=10, t=50, b=10))
        g2.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("### Resumen")
        st.json({
            "Modelo": modelo,
            "B (m)": optimo["B"], "L (m)": optimo["L"], "h (m)": optimo["h"],
            "q_adm (kPa)": optimo["q_adm"], "q_req (kPa)": optimo["q_req"],
            "Costo (S/)": optimo["costo"]
        }, expanded=False)

        st.markdown("### Recomendaciones")
        st.info(
            "✅ **Buen diseño**: margen entre q_adm y q_req.\n\n"
            "💡 **Sugerencia**: si necesitas mayor rigidez, incrementa **h**; "
            "si buscas menor costo, ajusta B y L manteniendo el margen de seguridad.\n\n"
            "📚 **Referencias**: Terzaghi & Peck; Meyerhof; Vesic."
        )

        st.markdown("### Tabla de soluciones válidas (top 300 por costo)")
        df_show = df_valid.head(300).copy()
        st.dataframe(df_show.style.format({
            "B": "{:.2f}", "L": "{:.2f}", "h": "{:.2f}",
            "area": "{:.2f}", "q_adm": "{:.1f}", "q_req": "{:.1f}", "costo": "{:.0f}"
        }), use_container_width=True)

        st.markdown("### Descargas")
        d1, d2 = st.columns(2)

        xls = df_to_excel_bytes(df_show, sheet_name="Resultados")
        if xls:
            d1.download_button(
                "📥 Descargar Excel",
                data=xls,
                file_name="resultados_cimentaciones.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            d1.info("Instala *xlsxwriter* u *openpyxl* para habilitar la descarga en Excel.", icon="ℹ️")

        pdf = generar_pdf_resumen(
            {
                "modelo": modelo, "gamma": gamma, "c": c, "phi": phi, "D": D, "N": N, "FS": FS,
                "c_concreto": c_concreto, "c_acero": c_acero,
                "Bmin": params["Bmin"], "Bmax": params["Bmax"],
                "Lmin": params["Lmin"], "Lmax": params["Lmax"],
                "hmin": params["hmin"], "hmax": params["hmax"],
            },
            optimo,
            df_show.head(10)
        )
        d2.download_button(
            "📄 Descargar reporte (PDF)",
            data=pdf,
            file_name="reporte_cimentaciones.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
