# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- utilidades geotécnicas ----------
def nc_nq_ngamma(phi_deg: float):
    """Coeficientes clásicos de capacidad portante (Terzaghi)
    Fórmulas estándar en función de φ (en grados).
    """
    phi = np.radians(phi_deg)
    if phi_deg == 0:
        Nq = 1.0
    else:
        Nq = np.exp(np.pi * np.tan(phi)) * (np.tan(np.radians(45.0) + phi / 2.0) ** 2)
    Nc = (Nq - 1.0) / np.tan(phi) if phi_deg != 0 else 5.7  # valor típico para φ=0
    Nγ = 2.0 * (Nq + 1.0) * np.tan(phi)
    return Nc, Nq, Nγ


def q_admisible(c_kpa, phi_deg, gamma, B, D, FS):
    """Capacidad admisible por Terzaghi (kPa).
       q_adm = (c*Nc + γ*D*Nq + 0.5*γ*B*Nγ) / FS
    """
    Nc, Nq, Nγ = nc_nq_ngamma(phi_deg)
    q_ult = c_kpa * Nc + gamma * D * Nq + 0.5 * gamma * B * Nγ
    return q_ult / FS


def costo_aprox(B, L, h, precio_conc_s_m3):
    """Costo muy simplificado: volumen * precio del concreto."""
    vol = B * L * h  # m3
    return vol * precio_conc_s_m3


# ========= UI =========
st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")

st.title("Optimización de Cimentaciones")
st.info(
    "Ingresa los **datos del suelo, la carga y los rangos de B, L y h** "
    "en el panel izquierdo y pulsa **Analizar y optimizar**."
)

with st.sidebar:
    st.header("Parámetros de entrada")

    modelo = st.selectbox(
        "Modelo de capacidad",
        ["Terzaghi (recomendado)"],
        index=0,
        help="Usamos la fórmula clásica de Terzaghi."
    )

    # Presets de suelo (rellenan gamma, c, phi)
    presets = {
        "Arcilla blanda (γ=17, c=25, φ=0)": {"gamma": 17.0, "c": 25.0, "phi": 0.0},
        "Arcilla firme (γ=18, c=40, φ=0)": {"gamma": 18.0, "c": 40.0, "phi": 0.0},
        "Arena densa (γ=19, c=0, φ=35)": {"gamma": 19.0, "c": 0.0, "phi": 35.0},
        "SUELO DEMO (γ=18, c=20, φ=35) ✅": {"gamma": 18.0, "c": 20.0, "phi": 35.0},
    }
    preset_name = st.selectbox(
        "Preset de suelo (rápido)",
        list(presets.keys()),
        index=3,
    )
    preset = presets[preset_name]

    gamma = st.number_input("Peso unitario γ (kN/m³)", 10.0, 25.0, value=preset["gamma"], step=0.5)
    c_kpa = st.number_input("Cohesión c (kPa)", 0.0, 300.0, value=preset["c"], step=1.0)
    phi = st.number_input("Ángulo de fricción φ (°)", 0.0, 45.0, value=preset["phi"], step=1.0)
    D = st.number_input("Profundidad D (m)", 0.0, 5.0, value=1.5, step=0.1)
    N_kN = st.number_input("Carga N (kN)", 10.0, 50000.0, value=1000.0, step=50.0)
    FS = st.number_input("Factor de seguridad", 1.0, 4.0, value=2.0, step=0.1)

    st.subheader("Costos")
    precio_concreto = st.number_input("Concreto (S/ por m³)", 200.0, 2500.0, value=650.0, step=10.0)

    st.subheader("Rangos de diseño")
    B_min, B_max = st.slider("Base B (m)", 1.2, 4.0, (1.2, 3.8), step=0.1)
    B_pts = st.number_input("Resolución B (puntos)", 5, 80, value=25, step=1)

    L_min, L_max = st.slider("Largo L (m)", 1.2, 4.0, (1.2, 3.8), step=0.1)
    L_pts = st.number_input("Resolución L (puntos)", 5, 80, value=25, step=1)

    h_min, h_max = st.slider("Altura h (m)", 0.4, 2.0, (0.6, 1.2), step=0.05)
    h_pts = st.number_input("Resolución h (puntos)", 2, 40, value=8, step=1)

    st.markdown("---")
    run = st.button("🔍 Analizar y optimizar", use_container_width=True)

# ========= OPTIMIZACIÓN =========
if run:
    B_grid = np.linspace(B_min, B_max, B_pts)
    L_grid = np.linspace(L_min, L_max, L_pts)
    h_grid = np.linspace(h_min, h_max, h_pts)

    rows = []
    for B in B_grid:
        for L in L_grid:
            area = B * L
            q_req = N_kN / area  # kPa (1 kN/m2 = 1 kPa)
            q_adm = q_admisible(c_kpa, phi, gamma, B, D, FS)

            if q_adm >= q_req:
                for h in h_grid:
                    costo = costo_aprox(B, L, h, precio_concreto)
                    rows.append(
                        dict(B=round(B, 2), L=round(L, 2), h=round(h, 2),
                             q_adm=round(q_adm, 1), q_req=round(q_req, 1),
                             costo=round(costo, 0))
                    )

    if not rows:
        st.warning(
            "No se encontraron soluciones que cumplan la **capacidad admisible**. "
            "Prueba con **B y L mayores**, **φ o c más altos**, **FS menor** o **carga menor**."
        )
    else:
        df = pd.DataFrame(rows)
        df_sorted = df.sort_values(["costo", "B", "L", "h"], ascending=[True, True, True, True]).reset_index(drop=True)
        opt = df_sorted.iloc[0].to_dict()

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("B (m)", f'{opt["B"]:.2f}')
        c2.metric("L (m)", f'{opt["L"]:.2f}')
        c3.metric("h (m)", f'{opt["h"]:.2f}')
        c4.metric("Costo (S/)", f'{int(opt["costo"])}')

        st.markdown("### Comparaciones y candidatos")

        colA, colB = st.columns([1, 1])
        with colA:
            fig = px.bar(
                pd.DataFrame(
                    {"tipo": ["q_req", "q_adm"], "kPa": [opt["q_req"], opt["q_adm"]]}
                ),
                x="tipo", y="kPa",
                color="tipo",
                color_discrete_sequence=["#8fa8ff", "#39b69a"],
                text="kPa",
                title="q_req vs q_adm (óptimo)"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with colB:
            fig2 = px.scatter(
                df_sorted,
                x="B", y="L",
                size="h",
                color="costo",
                color_continuous_scale="Tealgrn",
                title="Candidatos válidos (color=costo, tamaño=h)",
                hover_data=["q_adm", "q_req", "h", "costo"],
            )
            fig2.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Resumen")
        resumen = {
            "Modelo": "Terzaghi",
            "B (m)": opt["B"], "L (m)": opt["L"], "h (m)": opt["h"],
            "q_adm (kPa)": opt["q_adm"], "q_req (kPa)": opt["q_req"],
            "Costo (S/)": opt["costo"],
        }
        st.code(json.dumps(resumen, indent=2, ensure_ascii=False))

        st.markdown("### Recomendaciones")
        ok_margin = "✅ Buen diseño: **q_adm ≥ q_req**."
        tip = f"💡 Óptimo actual: **S/ {int(opt['costo'])}**. Prueba variaciones con **h** ligeramente mayor si necesitas rigidez."
        st.success(ok_margin)
        st.info(tip)

        st.markdown("### Tabla de candidatos (top 150 por costo)")
        top = df_sorted.head(150)
        st.dataframe(top, use_container_width=True, height=300)

        # Descargar Excel
        @st.cache_data
        def _to_excel_bytes(_df):
            from io import BytesIO
            with pd.ExcelWriter(BytesIO(), engine="xlsxwriter") as writer:
                _df.to_excel(writer, sheet_name="Candidatos", index=False)
                writer.save()
                data = writer.handles.handle.getvalue()
            return data

        xls = _to_excel_bytes(df_sorted)
        st.download_button(
            "⬇️ Descargar Excel (todos los candidatos)",
            data=xls,
            file_name="cimentaciones_candidatos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

else:
    # Estado inicial (sin ejecutar)
    st.caption("Cuando estés lista/ listo, ajusta los rangos y pulsa **Analizar y optimizar**.")
