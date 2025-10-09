    # ===================== Estadística =====================
    st.subheader("📈 Estadística descriptiva (soluciones viables)")

    # Qué columnas queremos resumir (en lenguaje humano)
    columnas_resumen = {
        "Costo estimado (S/)": "Costo (S/)",
        "Presión de contacto requerida (kPa)": "Presión requerida (kPa)",
        "Capacidad admisible del suelo (kPa)": "Capacidad admisible (kPa)",
        "Margen de seguridad (kPa)": "Margen de seguridad (kPa)",
        "Base B (m)": "Base B (m)",
        "Largo L (m)": "Largo L (m)",
        "Espesor h (m)": "Espesor h (m)",
    }

    # Función para armar una fila de estadísticos en español
    def resumen_columna(serie: pd.Series):
        s = pd.to_numeric(serie, errors="coerce").dropna()
        if s.empty:
            return {
                "Cantidad de soluciones": 0,
                "Promedio": None,
                "Desviación estándar": None,
                "Mínimo": None,
                "Percentil 25 (Q1)": None,
                "Mediana (Q2)": None,
                "Percentil 75 (Q3)": None,
                "Máximo": None,
            }
        return {
            "Cantidad de soluciones": int(s.count()),
            "Promedio": float(s.mean()),
            "Desviación estándar": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
            "Mínimo": float(s.min()),
            "Percentil 25 (Q1)": float(np.percentile(s, 25)),
            "Mediana (Q2)": float(np.percentile(s, 50)),
            "Percentil 75 (Q3)": float(np.percentile(s, 75)),
            "Máximo": float(s.max()),
        }

    filas = []
    for col, etiqueta in columnas_resumen.items():
        stats = resumen_columna(df_view[col])
        stats = {"Variable": etiqueta, **stats}
        filas.append(stats)

    df_stats = pd.DataFrame(filas)[
        [
            "Variable",
            "Cantidad de soluciones",
            "Promedio",
            "Desviación estándar",
            "Mínimo",
            "Percentil 25 (Q1)",
            "Mediana (Q2)",
            "Percentil 75 (Q3)",
            "Máximo",
        ]
    ]

    st.dataframe(df_stats, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ ¿Qué significa cada estadístico?"):
        st.markdown(
            "- **Cantidad de soluciones**: número de alternativas viables analizadas.\n"
            "- **Promedio**: valor medio de la variable.\n"
            "- **Desviación estándar**: qué tanto se dispersan los valores alrededor del promedio.\n"
            "- **Mínimo / Máximo**: extremos observados.\n"
            "- **Percentil 25 (Q1)**: el 25% de los valores está por **debajo** de este número.\n"
            "- **Mediana (Q2)**: el 50% de los valores está por debajo (mitad de los datos).\n"
            "- **Percentil 75 (Q3)**: el 75% de los valores está por debajo (cuartil superior)."
        )



