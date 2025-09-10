# ------------------------------
# Optimización de Cimentaciones
# ------------------------------
import math
from io import BytesIO
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ============ CONFIG ============
st.set_page_config(page_title="Optimización de Cimentaciones", layout="wide")

# Estado inicial para autorun
if "has_run_once" not in st.session_state:
    st.session_state.has_run_once = False
if "autorun_now" not in st.session_state:
    st.session_state.autorun_now = False

# ============ UTILIDADES ============
def deg2rad(phi_deg: float) -> float:
    return phi_deg * math.pi / 180.0

def N_factors(phi_deg: float):
    """Coeficientes de capacidad portante clásicos (Terzaghi)."""
    phi = deg2rad(phi_deg)
    if phi_deg <= 0:
        # Para arcillas puras se acostumbra usar Nq=1, Nγ≈0, Nc≈5.7~5.14
        Nq = 1.0
        Ng = 0.0
        Nc = 5.7
        return Nc, Nq, Ng
    Nq = math.e ** (math.pi * math.tan(phi)) * (math.tan(math.pi / 4 + phi / 2)) ** 2
    Nc = (Nq - 1) / math.tan(phi)
    Nγ = 2 * (Nq + 1) * math.tan(phi)
    return Nc, Nq, Nγ

def q_admisible_Terzaghi(c, phi, gamma, D, B, FS):
    """q_adm = (q_ult / FS) - γD."""
    Nc, Nq, Nγ = N_factors(phi)
    q_ult = c * Nc + gamma * D * Nq + 0.5 * gamma * B * Nγ
    q_adm = (q_ult / FS) - gamma * D
    return max(q_adm, 0.0)

def costo_simple(concreto_Sm3, acero_Skg, B, L, h):
    """Costo simple = Concreto (m3). Si quieres, agrega acero y otros ítems."""
    vol = B * L * h  # m3
    # si quieres agregar acero, deja algo simple como 20 kg/m3:
    acero_kg = vol * 20.0
    return concreto_Sm3 * vol + acero_Skg * acero_kg

def grid_optimize(modelo:str, c, phi, gamma, D, N, FS,
                  Bmin, Bmax, nB,
                  Lmin, Lmax, nL,
                  hmin, hmax, nh,
                  concreto_Sm3, acero_Skg):
    """Búsqueda en malla: retorna mejor diseño y dataframe con candidatos válidos."""
    Bs = np.linspace(Bmin, Bmax, nB)
    Ls = np.linspace(Lmin, Lmax, nL)
    hs = np.linspace(hmin, hmax, nh)
    rows = []
    for B in Bs:
        for L in Ls:
            for h in hs:
                area = B * L
                if area <= 0:
                    continue
                q_req = (N * 1000.0) / area  # N en kN, pasamos a kPa si área m2 (1 kN/m2 = 1 kPa)
                if modelo.startswith("Terzaghi"):
                    q_adm = q_admisible_Terzaghi(c, phi, gamma, D, B, FS)
                else:
                    # por ahora igual que Terzaghi
                    q_adm = q_admisible_Terzaghi(c, phi, gamma, D, B, FS)
                ok = q_adm >= q_req
                cost = costo_simple(concreto_Sm3, acero_Skg, B, L, h)
                rows.append([B, L, h, q_adm, q_req, cost, ok])
    df = pd.DataFrame(rows, columns=["B","L","h","q_adm","q_req","costo","ok"])
    validos = df[df["ok"]].copy()
    if validos.empty:
        return None, df, validos
    best = validos.sort_values("costo", ascending=True).iloc[0]
    return best, df, validos

# --- GA sencillo opcional ---
@dataclass
class GAConf:
    pop: int = 40
    gens: int = 25
    px: float = 0.8
    pm: float = 0.15

def ga_optimizar(modelo, c, phi, gamma, D, N, FS,
                 Bmin, Bmax, Lmin, Lmax, hmin, hmax,
                 concreto_Sm3, acero_Skg, conf:GAConf):
    """Pequeño GA continuo. Devuelve mejor diseño y dataframe con historial."""
    rng = np.random.default_rng(42)

    def fitness(B,L,h):
        area = B*L
        if area <= 0: return np.inf, False, 0.0, 0.0
        q_req = (N*1000.0)/area
        q_adm = q_admisible_Terzaghi(c,phi,gamma,D,B,FS)
        ok = q_adm >= q_req
        cost = costo_simple(concreto_Sm3, acero_Skg, B,L,h)
        return (cost if ok else 1e12+cost), ok, q_adm, q_req

    def clip(x, lo, hi): return max(lo, min(hi, x))

    # Población
    P = np.column_stack([
        rng.uniform(Bmin,Bmax,conf.pop),
        rng.uniform(Lmin,Lmax,conf.pop),
        rng.uniform(hmin,hmax,conf.pop)
    ])

    hist = []
    for g in range(conf.gens):
        fits, oks, qadms, qreqs = [],[],[],[]
        for i in range(conf.pop):
            f, ok, qa, qr = fitness(P[i,0], P[i,1], P[i,2])
            fits.append(f); oks.append(ok); qadms.append(qa); qreqs.append(qr)
        fits = np.array(fits)
        idx = np.argsort(fits)
        P = P[idx]
        fits = fits[idx]
        hist.append([g, P[0,0],P[0,1],P[0,2], fits[0]])

        # selección elitista + torneo
        elite = P[:max(2, conf.pop//5)].copy()
        newP = [elite[0], elite[1]]
        while len(newP)<conf.pop:
            i,j = rng.integers(0,elite.shape[0],2)
            if rng.random() < conf.px:
                alpha = rng.random()
                child = alpha*elite[i] + (1-alpha)*elite[j]
            else:
                child = elite[i].copy()
            # mutación
            if rng.random() < conf.pm:
                child += rng.normal(scale=[0.1*(Bmax-Bmin),0.1*(Lmax-Lmin),0.1*(hmax-hmin)], size=3)
            child[0] = clip(child[0],Bmin,Bmax)
            child[1] = clip(child[1],Lmin,Lmax)
            child[2] = clip(child[2],hmin,hmax)
            newP.append(child)
        P = np.array(newP)

    # Mejor final
    bestF, bestB, bestL, besth = np.inf, None,None,None
    for i in range(conf.pop):
        f, ok, qa, qr = fitness(P[i,0],P[i,1],P[i,2])
        if ok and f < bestF:
            bestF = f; bestB, bestL, besth = P[i,0],P[i,1],P[i,2]
    if bestB is None:
        # no factibles -> devuelve None
        return None, pd.DataFrame(hist, columns=["gen","B","L","h","fitness"])
    # arma df
    # recalcular q_adm y q_req:
    area = bestB*bestL
    q_req = (N*1000)/area
    q_adm = q_admisible_Terzaghi(c,phi,gamma,D,bestB,FS)
    best = pd.Series({"B":bestB,"L":bestL,"h":besth,"q_adm":q_adm,"q_req":q_req,"costo":bestF})
    hist_df = pd.DataFrame(hist, columns=["gen","B","L","h","fitness"])
    return best, hist_df

# Excel bytes
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Resultados")
    return output.getvalue()

# PDF simple
def pdf_bytes(resumen_texto:str) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Informe de Optimización de Cimentaciones")
    y -= 30
    c.setFont("Helvetica", 10)
    for line in resumen_texto.split("\n"):
        c.drawString(40, y, line[:100])
        y -= 14
        if y < 60:
            c.showPage(); y = h - 40
    c.showPage(); c.save()
    buf.seek(0)
    return buf.getvalue()

# ============ UI ============
st.title("Optimización de Cimentaciones")
st.caption("Diseño óptimo por costo cumpliendo capacidad admisible — vista compacta")
st.info("Ingresa los **datos** en la parte izquierda y pulsa **Analizar y optimizar** o **Optimizar con GA**.", icon="🛠️")

colL, colR = st.columns([0.30, 0.70])

with colL:
    # ---- Sidebar/inputs principales ----
    st.subheader("Parámetros de entrada")
    modelo = st.selectbox(
        "Modelo de capacidad",
        ["Terzaghi (recomendado)","Meyerhof (simple)"],
        index=0
    )

    preset = st.selectbox(
        "Preset de suelo (rápido)",
        ["Arcilla blanda (γ=17, c=25, φ=0)","Arena media (γ=19.5, c=1, φ=32)"]
    )
    if preset.startswith("Arcilla"):
        gamma0, c0, phi0 = 17.0, 25.0, 0.0
    else:
        gamma0, c0, phi0 = 19.5, 1.0, 32.0

    gamma = st.number_input("Peso unitario γ (kN/m³)", value=gamma0, step=0.5, min_value=10.0, max_value=25.0, key="gamma")
    c = st.number_input("Cohesión c (kPa)", value=c0, step=1.0, min_value=0.0, max_value=200.0, key="c")
    phi = st.number_input("Ángulo de fricción φ (°)", value=phi0, step=0.5, min_value=0.0, max_value=45.0, key="phi")
    D = st.number_input("Profundidad D (m)", value=1.5, step=0.1, min_value=0.0, max_value=5.0, key="D")
    N = st.number_input("Carga N (kN)", value=800.0, step=10.0, min_value=100.0, max_value=5000.0, key="N")
    FS = st.number_input("Factor de seguridad", value=2.5, step=0.1, min_value=1.5, max_value=4.0, key="FS")

    st.markdown("### Costos")
    concreto_Sm3 = st.number_input("Concreto (S/ por m³)", value=650.0, step=10.0, min_value=300.0, max_value=2000.0, key="ccon")
    acero_Skg = st.number_input("Acero (S/ por kg)", value=5.5, step=0.1, min_value=2.0, max_value=20.0, key="cacr")

    st.markdown("### Rangos de diseño")
    Bmin, Bmax = st.slider("Base B (m)", 1.0, 5.0, (1.40, 3.00))
    nB = st.number_input("Resolución B (puntos)", 5, 100, 25)

    Lmin, Lmax = st.slider("Largo L (m)", 1.0, 6.0, (1.80, 4.00))
    nL = st.number_input("Resolución L (puntos)", 5, 100, 25)

    hmin, hmax = st.slider("Altura h (m)", 0.30, 1.50, (0.50, 1.00))
    nh = st.number_input("Resolución h (puntos)", 3, 30, 8)

    st.markdown("### Adjuntos")
    img = st.file_uploader("Sube un croquis / perfil del suelo (PNG/JPG)", type=["png","jpg","jpeg"])

    st.markdown("### Acciones")
    # DEMO button
    if st.button("🧪 Cargar demo segura"):
        # carga valores DEMO seguros en session_state
        st.session_state.gamma = 17.0
        st.session_state.c = 25.0
        st.session_state.phi = 0.0
        st.session_state.D = 1.5
        st.session_state.N = 800.0
        st.session_state.FS = 2.5
        st.session_state.ccon = 650.0
        st.session_state.cacr = 5.5
        # rangos
        st.session_state["Base B (m)"] = (1.4, 3.0)  # no afecta, pero por si usas .slider(label=.., key=..)
        st.session_state.autorun_now = True
        st.rerun()

    btn_grid = st.button("🔍 Analizar y optimizar")
    btn_ga = st.button("🧬 Optimizar con GA")

with colR:
    # Aquí pintaremos resultados
    cont_resultados = st.container()

# ---- Función pipeline ----
def ejecutar_pipeline_principal(modo:str):
    """Calcula y pinta resultados con el modo 'grid' o 'ga'."""
    with cont_resultados:
        if modo == "grid":
            best, df, validos = grid_optimize(
                modelo, c, phi, gamma, D, N, FS,
                Bmin, Bmax, nB,
                Lmin, Lmax, nL,
                hmin, hmax, nh,
                concreto_Sm3, acero_Skg
            )
        else:
            conf = GAConf(pop=40, gens=25, px=0.8, pm=0.15)
            best, hist = ga_optimizar(
                modelo, c, phi, gamma, D, N, FS,
                Bmin, Bmax, Lmin, Lmax, hmin, hmax,
                concreto_Sm3, acero_Skg, conf
            )
            if best is None:
                best, df, validos = None, pd.DataFrame(), pd.DataFrame()
            else:
                # construimos df "validos" con el punto GA
                df = pd.DataFrame([best])
                validos = df.copy()
                df["ok"] = True

        if (best is None) or validos.empty:
            # Solo muestra warning si ya corrimos
            if st.session_state.get("has_run_once"):
                st.warning(
                    "No se encontraron soluciones que cumplan la capacidad admisible. "
                    "Prueba con **B y L mayores**, **φ o c** más altos, **FS menor** o **carga** menor.",
                    icon="⚠️"
                )
            return

        # ---- KPIs
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("B (m)", f"{best['B']:.2f}")
        k2.metric("L (m)", f"{best['L']:.2f}")
        k3.metric("h (m)", f"{best['h']:.2f}")
        k4.metric("Costo (S/)", f"{best['costo']:.0f}")

        # ---- Gráficos
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                x=["q_req","q_adm"], y=[best["q_req"], best["q_adm"]],
                labels={"x":"Tipo","y":"kPa"},
                title="q_req vs q_adm (óptimo)",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if not validos.empty:
                sc = px.scatter(
                    validos, x="B", y="L", size="h", color="costo",
                    color_continuous_scale="Tealgrn", title="Candidatos válidos (color=costo, tamaño=h)"
                )
                sc.update_layout(margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(sc, use_container_width=True)

        # ---- Resumen JSON-like
        st.subheader("Resumen")
        resumen_str = (
            f"Modelo: {modelo}\n"
            f"B (m): {best['B']:.2f}\n"
            f"L (m): {best['L']:.2f}\n"
            f"h (m): {best['h']:.2f}\n"
            f"q_adm (kPa): {best['q_adm']:.1f}\n"
            f"q_req (kPa): {best['q_req']:.1f}\n"
            f"Costo (S/): {best['costo']:.0f}\n"
        )
        st.code(
            "{\n"
            f'  "Modelo": "{modelo}",\n'
            f'  "B (m)": {best["B"]:.2f},\n'
            f'  "L (m)": {best["L"]:.2f},\n'
            f'  "h (m)": {best["h"]:.2f},\n'
            f'  "q_adm (kPa)": {best["q_adm"]:.1f},\n'
            f'  "q_req (kPa)": {best["q_req"]:.1f},\n'
            f'  "Costo (S/)": {best["costo"]:.0f}\n'
            "}"
        )

        # ---- Recomendaciones
        st.subheader("Recomendaciones")
        tips = []
        margen = best["q_adm"] - best["q_req"]
        if margen > 150:
            tips.append("✅ Buen margen entre capacidad y demanda.")
        else:
            tips.append("🟡 Margen justo. Considera h un poco mayor si buscas rigidez.")
        tips.append(f"💡 Óptimo actual: **S/ {best['costo']:.0f}**.")
        st.success("\n".join(tips))

        # ---- Exportar (Excel + PDF)
        st.markdown("### Descargas")
        df_export = validos.copy()
        if not df_export.empty:
            xls = to_excel_bytes(df_export)
            st.download_button("⬇️ Descargar Excel", xls, "resultados.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        pdf = pdf_bytes(resumen_str)
        st.download_button("⬇️ Descargar Reporte (PDF)", pdf, "informe.pdf", "application/pdf")

        # ---- Imagen adjunta (solo si subieron)
        if img is not None:
            st.subheader("Imagen adjunta")
            st.image(img, use_container_width=True)

# ---- Eventos de botones ----
if btn_grid:
    ejecutar_pipeline_principal("grid")
    st.session_state.has_run_once = True

if btn_ga:
    ejecutar_pipeline_principal("ga")
    st.session_state.has_run_once = True

# ---- AUTORUN: demo o primera carga ----
def _autorun_si_corresponde():
    if st.session_state.autorun_now or not st.session_state.has_run_once:
        ejecutar_pipeline_principal("grid")
        st.session_state.has_run_once = True
        st.session_state.autorun_now = False

_autorun_si_corresponde()
