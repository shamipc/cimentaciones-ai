    df = pd.DataFrame(rows)
    df_ok = df[df["cumple"]].copy()

    if df_ok.empty:
        st.error("⚠️ No se encontraron soluciones que cumplan capacidad portante. Ajusta FS, N, φ, c o los rangos B–L–h.")
        st.stop()

    # ---------- Nombres legibles ----------
    nice = {
        "B": "Base B (m)",
        "L": "Largo L (m)",
        "h": "Espesor h (m)",
        "B_eff": "Base efectiva B′ (m)",
        "L_eff": "Largo efectivo L′ (m)",
        "q_req": "Presión de contacto requerida (kPa)",
        "q_adm": "Capacidad admisible del suelo (kPa)",
        "q_ult": "Capacidad última del suelo (kPa)",
        "margen": "Margen de seguridad (kPa)",
        "ex": "Excentricidad eₓ (m)",
        "ey": "Excentricidad eᵧ (m)",
        "cumple": "Cumple capacidad portante",
        "costo": "Costo estimado (S/)"
    }
    df_view = df_ok.rename(columns=nice)

    # ===================== KPIs ============================
    mejor_idx = df_view["Costo estimado (S/)"].idxmin()
    mejor = df_view.loc[mejor_idx]

    p25 = df_view["Costo estimado (S/)"].quantile(0.25)
    df_rob = df_view[df_view["Costo estimado (S/)"] <= p25]
    robusta = df_rob.sort_values(["Margen de seguridad (kPa)", "Costo estimado (S/)"],
                                 ascending=[False, True]).iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Soluciones viables", f"{len(df_view):,}")
    k2.metric("Costo mínimo (S/)", f"{mejor['Costo estimado (S/)']:.0f}")
    k3.metric("Margen (mín. costo)", f"{mejor['Margen de seguridad (kPa)']:.1f} kPa")
    k4.metric("Margen (opción robusta)", f"{robusta['Margen de seguridad (kPa)']:.1f} kPa")

    # ===================== Top 10 ==========================
    st.subheader("Top 10 soluciones por menor costo")
    st.dataframe(df_view.sort_values("Costo estimado (S/)").head(10), use_container_width=True)

    # ===================== Gráficos ========================
    st.subheader("Visualizaciones")
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.scatter(
            df_view,
            x="Base B (m)", y="Largo L (m)",
            color="Costo estimado (S/)", size="Espesor h (m)",
            hover_data=[
                "Presión de contacto requerida (kPa)",
                "Capacidad admisible del suelo (kPa)",
                "Margen de seguridad (kPa)",
                "Base efectiva B′ (m)", "Largo efectivo L′ (m)"
            ],
            title="Soluciones viables (color = costo, tamaño = h)"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.density_heatmap(
            df_view,
            x="Base B (m)", y="Largo L (m)", z="Margen de seguridad (kPa)",
            nbinsx=30, nbinsy=30, histfunc="avg",
            title="Mapa de calor del margen de seguridad (kPa)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.histogram(df_view, x="Costo estimado (S/)", nbins=30,
                            title="Distribución de costos de soluciones viables")
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        fig4 = px.box(df_view, y="Costo estimado (S/)", title="Resumen estadístico de costos")
        st.plotly_chart(fig4, use_container_width=True)

    # ===================== Estadística =====================
    st.subheader("📈 Estadística descriptiva (soluciones viables)")
    stats_cols = [
        "Costo estimado (S/)", "Presión de contacto requerida (kPa)",
        "Capacidad admisible del suelo (kPa)", "Margen de seguridad (kPa)",
        "Base B (m)", "Largo L (m)", "Espesor h (m)"
    ]
    st.dataframe(df_view[stats_cols].describe().T, use_container_width=True)

    # ===================== Recomendación ===================
    st.markdown("## ✅ Recomendación automática")
    texto = (
        "**Opción de mínimo costo**  \n"
        f"- B = **{mejor['Base B (m)']:.2f} m**, L = **{mejor['Largo L (m)']:.2f} m**, "
        f"h = **{mejor['Espesor h (m)']:.2f} m**  \n"
        f"- Presión requerida = **{mejor['Presión de contacto requerida (kPa)']:.1f} kPa**  \n"
        f"- Capacidad admisible del suelo = **{mejor['Capacidad admisible del suelo (kPa)']:.1f} kPa**  \n"
        f"- Margen de seguridad = **{mejor['Margen de seguridad (kPa)']:.1f} kPa**  \n"
        f"- Costo estimado = **S/ {mejor['Costo estimado (S/)']:.2f}**  \n\n"
        "**Opción robusta (≤ P25 de costo y mayor margen)**  \n"
        f"- B = **{robusta['Base B (m)']:.2f} m**, L = **{robusta['Largo L (m)']:.2f} m**, "
        f"h = **{robusta['Espesor h (m)']:.2f} m**  \n"
        f"- Presión requerida = **{robusta['Presión de contacto requerida (kPa)']:.1f} kPa**  \n"
        f"- Capacidad admisible del suelo = **{robusta['Capacidad admisible del suelo (kPa)']:.1f} kPa**  \n"
        f"- Margen de seguridad = **{robusta['Margen de seguridad (kPa)']:.1f} kPa**  \n"
        f"- Costo estimado = **S/ {robusta['Costo estimado (S/)']:.2f}**  \n\n"
        f"**Criterio:** verificación por **área efectiva** (B′ = B − 2|eₓ|, L′ = L − 2|eᵧ|), "
        f"modelo **{st.session_state.modelo}**, FS = **{st.session_state.FS:.2f}**, "
        f"D = **{st.session_state.D:.2f} m**."
    )
    st.markdown(texto)

    # ===== Definiciones breves para el informe =====
    with st.expander("ℹ️ Definiciones rápidas"):
        st.markdown(
            "- **Presión de contacto requerida**: presión media necesaria en la base de la zapata "
            "considerando excentricidades de los momentos (área efectiva B′·L′).  \n"
            "- **Capacidad admisible del suelo**: capacidad última reducida por el **FS** especificado.  \n"
            "- **Margen de seguridad**: diferencia *Capacidad admisible − Presión requerida* (kPa)."
        )

    # ===================== Exportación =====================
    st.subheader("📥 Exportar soluciones")
    csv_sol = df_view.sort_values("Costo estimado (S/)").to_csv(index=False)
    st.download_button(
        "Descargar CSV de soluciones viables",
        data=csv_sol,
        file_name="soluciones_cimentacion.csv",
        mime="text/csv",
        use_container_width=True,
    )
