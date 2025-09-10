import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from io import BytesIO
import xlsxwriter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# ==========================
# Configuración general UI
# ==========================
st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")
st.title("Optimización de Cimentaciones")

st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta")

st.info("Ingresa los **datos** en la parte izquierda y pulsa **Analizar y optimizar** o **Optimizar con GA**.", icon="🛠️")

# =========================================
# Utilidades de geotecnia (modelo clásico)
# =========================================
def terzaghi_Nf(phi_deg: float):
    """Coeficientes de capacidad portante (Terzaghi) simplificados en función de φ (°)."""
    phi = np.radians(phi_deg)
    if phi_deg <= 0:
        Nq = 1.0
        Nc = 5.7
        Ny = 0.0
    else:
        Nq = np.exp(np.pi * np.tan(phi)) * (np.tan(np.radians(45) + phi/2))**2
        Nc = (Nq - 1) / np.tan(phi)
        Ny = 2 * (Nq + 1) * np.tan(phi)
    return Nc, Nq, Ny

def q_admisible(modelo: str, c_kPa: float, phi_deg: float, gamma: float, D: float, B: float, FS: float):
    """
    Capacidad admisible simplificada (kPa) con correcciones básicas.
    gamma en kN/m3, B y D en m.
    """
    Nc, Nq, Ny = terzaghi_Nf(phi_deg)

    # Capacidad última (Terzaghi) sin factores de forma/profundidad para simpleza
    qu = c_kPa * Nc + gamma * D * Nq + 0.5 * gamma * B * Ny

    # modelos alternativos muy simples (variaciones +/-10%)
    if modelo == "Meyerhof":
        qu *= 1.05
    elif modelo == "Vesic":
        qu *= 1.1
    # Terzaghi recomendado: sin cambio

    qadm = qu / FS
    return qadm

def q_requerida(N_kN: float, B: float, L: float):
    area = B * L
    if area <= 0:
        return np.inf
    return N_kN / area  # kPa si N en kN y area en m2 (kN/m2 = kPa)

def costo_aprox(concreto_Sm3: float, acero_Skg: float, B: float, L: float, h: float):
    """
    Estimación de costo simple:
    - Concreto ~ volumen (B*L*h)
    - Acero ~ 80 kg/m3 de concreto (aprox) -> se multiplica por costo S/kg
    """
    vol = B * L * h  # m3
    masa_acero = max(0.0, 80.0 * vol)  # kg aprox
    return vol * concreto_Sm3 + masa_acero * acero_Skg

def factibilidad(modelo, c, phi, gamma, D, FS, N, B, L, h):
    qadm = q_admisible(modelo, c, phi, gamma, D, B, FS)
    qreq = q_requerida(N, B, L)
    return qreq <= qadm, qadm, qreq

# =================================
# Algoritmo Genético (minimizar $)
# =================================
def ga_optimizar(
    modelo, c, phi, gamma, D, FS, N,
    B_range, L_range, h_range,
    pop=30, gens=35, pmut=0.15, pcross=0.8, seed=42,
    concreto_Sm3=650, acero_Skg=5.5
):
    rng = np.random.default_rng(seed)

    def random_ind():
        B = rng.uniform(*B_range)
        L = rng.uniform(*L_range)
        h = rng.uniform(*h_range)
        return np.array([B, L, h], dtype=float)

    def mutate(ind):
        child = ind.copy()
        # pequeña perturbación relativa
        child[0] = np.clip(child[0] * rng.normal(1, 0.05), *B_range)
        child[1] = np.clip(child[1] * rng.normal(1, 0.05), *L_range)
        child[2] = np.clip(child[2] * rng.normal(1, 0.05), *h_range)
        return child

    def crossover(a, b):
        alpha = rng.uniform(0, 1, size=3)
        c1 = alpha * a + (1 - alpha) * b
        c2 = (1 - alpha) * a + alpha * b
        c1[0] = np.clip(c1[0], *B_range)
        c1[1] = np.clip(c1[1], *L_range)
        c1[2] = np.clip(c1[2], *h_range)
        c2[0] = np.clip(c2[0], *B_range)
        c2[1] = np.clip(c2[1], *L_range)
        c2[2] = np.clip(c2[2], *h_range)
        return c1, c2

    def fitness(ind):
        B, L, h = ind
        ok, qadm, qreq = factibilidad(modelo, c, phi, gamma, D, FS, N, B, L, h)
        cost = costo_aprox(concreto_Sm3, acero_Skg, B, L, h)
        if not ok:
            # penalización si no cumple
            penalty = 1e6 + (qreq - qadm) * 1e5
            return penalty, qadm, qreq, cost
        return cost, qadm, qreq, cost

    # población inicial
    popu = [random_ind() for _ in range(pop)]
    best_hist = []
    elite = None
    elite_fit = np.inf
    all_candidates = []

    for g in range(gens):
        fits = [fitness(ind)[0] for ind in popu]

        # Hallar elite
        idx_best = int(np.argmin(fits))
        if fits[idx_best] < elite_fit:
            elite_fit = fits[idx_best]
            elite = popu[idx_best].copy()

        best_hist.append(elite_fit)

        # Selección por torneo simple
        new_pop = [elite.copy()]  # conservar elite
        while len(new_pop) < pop:
            a, b = popu[np.random.randint(pop)], popu[np.random.randint(pop)]
            c, d = popu[np.random.randint(pop)], popu[np.random.randint(pop)]
            fa = fitness(a)[0]; fb = fitness(b)[0]
            fc = fitness(c)[0]; fd = fitness(d)[0]
            parent1 = a if fa < fb else b
            parent2 = c if fc < fd else d

            if rng.random() < pcross:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if rng.random() < pmut: child1 = mutate(child1)
            if rng.random() < pmut: child2 = mutate(child2)
            new_pop.extend([child1, child2])

        popu = new_pop[:pop]

        # Log de candidatos por generación (para scatter)
        for ind in popu:
            B, L, h = ind
            f, qadm, qreq, cost = fitness(ind)
            all_candidates.append([g, B, L, h, cost, qadm, qreq, f])

    # Mejor individuo final
    fbest, qadm, qreq, cost = fitness(elite)
    B, L, H = elite
    df_all = pd.DataFrame(all_candidates, columns=["gen", "B", "L", "h", "costo", "q_adm", "q_req", "fitness"])
    return (B, L, H, cost, qreq, qadm, best_hist, df_all)

# =====================
# Sidebar: Parámetros
# =====================
with st.sidebar:
    st.header("Parámetros de entrada")

    modelo = st.selectbox("Modelo de capacidad", ["Terzaghi (recomendado)", "Meyerhof", "Vesic"])

    preset = st.selectbox(
        "Preset de suelo (rápido)",
        [
            "Arcilla blanda (γ=17, c=18, φ=0)",
            "Arena densa (γ=19, c=0, φ=35)",
            "Arcilla media (γ=18, c=25, φ=5)"
        ],
        index=0
    )

    # Defaults según preset
    if preset == "Arcilla blanda (γ=17, c=18, φ=0)":
        gamma0, c0, phi0 = 17.0, 18.0, 0.0
    elif preset == "Arena densa (γ=19, c=0, φ=35)":
        gamma0, c0, phi0 = 19.0, 0.0, 35.0
    else:
        gamma0, c0, phi0 = 18.0, 25.0, 5.0

    gamma = st.number_input("Peso unitario γ (kN/m³)", value=gamma0, step=0.5, min_value=10.0, max_value=25.0)
    c = st.number_input("Cohesión c (kPa)", value=c0, step=1.0, min_value=0.0, max_value=300.0)
    phi = st.number_input("Ángulo de fricción φ (°)", value=phi0, step=1.0, min_value=0.0, max_value=45.0)
    D = st.number_input("Profundidad D (m)", value=1.5, step=0.1, min_value=0.0, max_value=10.0)
    N = st.number_input("Carga N (kN)", value=1000.0, step=50.0, min_value=10.0)
    FS = st.number_input("Factor de seguridad", value=2.5, step=0.1, min_value=1.2, max_value=4.0)

    st.subheader("Costos")
    concreto_Sm3 = st.number_input("Concreto (S/ por m³)", value=650.0, step=10.0, min_value=0.0)
    acero_Skg = st.number_input("Acero (S/ por kg)", value=5.5, step=0.1, min_value=0.0)

    st.subheader("Rangos de diseño")
    B_min, B_max = st.slider("Base B (m)", 1.2, 3.8, value=(1.4, 3.0), step=0.05)
    L_min, L_max = st.slider("Largo L (m)", 1.2, 3.8, value=(1.6, 3.5), step=0.05)
    h_min, h_max = st.slider("Altura h (m)", 0.40, 1.50, value=(0.50, 1.0), step=0.05)
    B_range = (B_min, B_max)
    L_range = (L_min, L_max)
    h_range = (h_min, h_max)

    st.markdown("---")
    colb, colg = st.columns(2)
    with colb:
        run_grid = st.button("🔎 Analizar y optimizar")
    with colg:
        run_ga = st.button("🧬 Optimizar con GA")

# ===========================================
# Barrido clásico (tu botón Analizar y optim)
# ===========================================
def barrido(modelo, c, phi, gamma, D, FS, N, concreto_Sm3, acero_Skg, B_range, L_range, h_range, nB=25, nL=25, nh=8):
    Bs = np.linspace(*B_range, nB)
    Ls = np.linspace(*L_range, nL)
    hs = np.linspace(*h_range, nh)
    rows = []
    for B in Bs:
        for L in Ls:
            for h in hs:
                ok, qadm, qreq = factibilidad(modelo, c, phi, gamma, D, FS, N, B, L, h)
                cost = costo_aprox(concreto_Sm3, acero_Skg, B, L, h)
                if ok:
                    rows.append([B, L, h, cost, qreq, qadm])
    df = pd.DataFrame(rows, columns=["B","L","h","costo","q_req","q_adm"])
    return df

# =========================
# Panel de resultados
# =========================
def kpi_cards(B, L, h, costo):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("B (m)", f"{B:.2f}")
    col2.metric("L (m)", f"{L:.2f}")
    col3.metric("h (m)", f"{h:.2f}")
    col4.metric("Costo (S/)", f"{costo:,.0f}")

def chart_qreq_qadm(qreq, qadm):
    dfb = pd.DataFrame({
        "Tipo": ["q_req", "q_adm"],
        "kPa": [qreq, qadm]
    })
    fig = px.bar(dfb, x="Tipo", y="kPa",
                 text="kPa",
                 color="Tipo",
                 color_discrete_sequence=["#6377F1","#18B495"])
    fig.update_traces(texttemplate="%{y:.1f}", textposition="outside")
    fig.update_layout(height=340, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

def chart_candidates(df):
    if df.empty: 
        return
    fig = px.scatter(
        df, x="B", y="L", color="costo", size="h",
        color_continuous_scale="Tealgrn",
        title="Candidatos válidos (color = Costo, tamaño = h)"
    )
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

def sensibilidad(modelo, c, phi, gamma, D, FS, N, best, concreto_Sm3, acero_Skg):
    """Barras de sensibilidad ±10%."""
    B, L, h = best
    base_cost = costo_aprox(concreto_Sm3, acero_Skg, B, L, h)
    base_ok, base_qadm, base_qreq = factibilidad(modelo, c, phi, gamma, D, FS, N, B, L, h)

    def delta_cost(var, factor):
        cc, pphi, gga, ffs = c, phi, gamma, FS
        if var=="c": cc = c*factor
        if var=="phi": pphi = phi*factor
        if var=="gamma": gga = gamma*factor
        if var=="FS": ffs = FS*factor
        ok, qadm, qreq = factibilidad(modelo, cc, pphi, gga, D, ffs, N, B, L, h)
        # si deja de ser factible, marcamos costo alto (penalizamos)
        if not ok:
            return base_cost * 1.2
        return costo_aprox(concreto_Sm3, acero_Skg, B, L, h)

    labels = ["c", "φ", "γ", "FS"]
    plus = []
    minus = []
    for var in labels:
        plus.append(delta_cost(var, 1.10))
        minus.append(delta_cost(var, 0.90))

    df = pd.DataFrame({
        "Variable": labels*2,
        "Tipo": ["+10%"]*4 + ["-10%"]*4,
        "Costo": plus + minus
    })
    fig = px.bar(df, x="Variable", y="Costo", color="Tipo", barmode="group",
                 color_discrete_sequence=["#4CB2A3","#8898F7"],
                 title="Sensibilidad de costo ±10% (manteniendo B,L,h)")
    fig.update_layout(height=380, margin=dict(l=10,r=10,t=40,b=20))
    st.plotly_chart(fig, use_container_width=True)

# ======================
# Exportaciones
# ======================
def export_excel(df, best_row):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        if df is not None and not df.empty:
            df.to_excel(writer, index=False, sheet_name="Candidatos")
        pd.DataFrame([best_row]).to_excel(writer, index=False, sheet_name="Optimo")
        wb = writer.book
        ws = writer.sheets["Optimo"]
        ws.write(0, 6, "Nota: cálculos simplificados (capacidad admisible y costo).")
    buffer.seek(0)
    return buffer

def export_pdf(best_row, modelo, FS):
    buffer = BytesIO()
    cpdf = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4

    cpdf.setFont("Helvetica-Bold", 14)
    cpdf.drawString(2*cm, h-2*cm, "Optimización de Cimentaciones (Resumen)")

    cpdf.setFont("Helvetica", 11)
    y = h-3.2*cm
    lines = [
        f"Modelo de capacidad: {modelo}",
        f"FS = {FS:.2f}",
        f"B = {best_row['B']:.2f} m, L = {best_row['L']:.2f} m, h = {best_row['h']:.2f} m",
        f"q_req = {best_row['q_req']:.1f} kPa, q_adm = {best_row['q_adm']:.1f} kPa",
        f"Costo estimado = S/ {best_row['costo']:,.0f}",
        "",
        "Notas:",
        "- Método clásico simplificado (Terzaghi/Meyerhof/Vesic).",
        "- Costo aproximado: vol. concreto + 80 kg/m³ de acero.",
        "- Use valores representativos del proyecto para decisión final."
    ]
    for ln in lines:
        cpdf.drawString(2*cm, y, ln); y -= 0.7*cm

    cpdf.showPage()
    cpdf.save()
    buffer.seek(0)
    return buffer

# ==========================
# Ejecución según botones
# ==========================
if run_grid or run_ga:

    # 1) Optimización por barrido (candidatos válidos)
    with st.spinner("Evaluando candidatos (barrido)…"):
        df = barrido(modelo.split()[0], c, phi, gamma, D, FS, N,
                     concreto_Sm3, acero_Skg, B_range, L_range, h_range,
                     nB=25, nL=25, nh=8)

    if df.empty and not run_ga:
        st.warning("No se encontraron soluciones factibles con el barrido. Ajusta los rangos o usa **Optimizar con GA**.", icon="⚠️")
    else:
        if run_ga:
            # 2) GA
            with st.spinner("Buscando óptimo con Algoritmo Genético…"):
                Bopt, Lopt, hopt, costopt, qreq, qadm, hist, df_all = ga_optimizar(
                    modelo.split()[0], c, phi, gamma, D, FS, N,
                    B_range, L_range, h_range,
                    pop=36, gens=40, pmut=0.15, pcross=0.85,
                    concreto_Sm3=concreto_Sm3, acero_Skg=acero_Skg
                )

            st.subheader("Resultado (GA)")
            kpi_cards(Bopt, Lopt, hopt, costopt)

            colA, colB = st.columns([1,1])
            with colA:
                chart_qreq_qadm(qreq, qadm)
            with colB:
                # Curva de convergencia
                dfhist = pd.DataFrame({"Generación": np.arange(len(hist)), "Costo": hist})
                fig = px.line(dfhist, x="Generación", y="Costo",
                              title="Convergencia GA (mejor costo por generación)",
                              markers=True)
                fig.update_layout(height=340, margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Candidatos visitados (GA)")
            # mostrar solo factibles en scatter
            dff = df_all[df_all["q_req"] <= df_all["q_adm"]].copy()
            chart_candidates(dff)

            st.subheader("Análisis de sensibilidad")
            sensibilidad(modelo.split()[0], c, phi, gamma, D, FS, N, (Bopt, Lopt, hopt), concreto_Sm3, acero_Skg)

            # exportaciones
            best_row = {"B":Bopt,"L":Lopt,"h":hopt,"costo":costopt,"q_req":qreq,"q_adm":qadm}
            colx, coly = st.columns(2)
            with colx:
                xbuf = export_excel(dff, best_row)
                st.download_button("📄 Descargar Excel", data=xbuf, file_name="optimizacion_cimentaciones.xlsx")
            with coly:
                pbuf = export_pdf(best_row, modelo.split()[0], FS)
                st.download_button("🧾 Descargar reporte (PDF)", data=pbuf, file_name="reporte_cimentaciones.pdf")

        else:
            # 3) Solo Barrido: mostrar óptimo del barrido
            st.subheader("Resultado (Barrido clásico)")
            best = df.sort_values("costo").iloc[0]
            kpi_cards(best["B"], best["L"], best["h"], best["costo"])

            colA, colB = st.columns([1,1])
            with colA:
                chart_qreq_qadm(best["q_req"], best["q_adm"])
            with colB:
                chart_candidates(df)

            st.subheader("Análisis de sensibilidad")
            sensibilidad(modelo.split()[0], c, phi, gamma, D, FS, N, (best["B"], best["L"], best["h"]),
                         concreto_Sm3, acero_Skg)

            best_row = best.to_dict()
            colx, coly = st.columns(2)
            with colx:
                xbuf = export_excel(df, best_row)
                st.download_button("📄 Descargar Excel", data=xbuf, file_name="optimizacion_cimentaciones.xlsx")
            with coly:
                pbuf = export_pdf(best_row, modelo.split()[0], FS)
                st.download_button("🧾 Descargar reporte (PDF)", data=pbuf, file_name="reporte_cimentaciones.pdf")

else:
    st.stop()


