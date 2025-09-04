import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import itertools
import numpy as np
import os

# =========================
# Configuración de la página
# =========================
st.set_page_config(
    page_title="Dashboard de Análisis de Riesgo Psicosocial",
    page_icon="🧠",
    layout="wide"
)

# =========================
# Función para cargar y procesar los datos (con cache)
# =========================
@st.cache_data
def cargar_datos():
    # =========================
    # 1. Leer y limpiar datos
    # =========================
    ruta_archivo = os.path.join(os.path.dirname(__file__), "transpuesto.parquet")
    df = pd.read_parquet(
        ruta_archivo,
        engine='pyarrow'
    )

    # Limpieza de "Nivel de riesgo"
    df["Nivel de riesgo"] = df["Nivel de riesgo"].replace({
        "Anulado": "No Referido",
        "Incompleta": "No Referido",
        "No Aplica": "No Referido",
        "No Responde": "No Referido",
        "No Respondio": "No Referido",
        "No Respondió": "No Referido",
        "Sin Responder": "No Referido",
        "Vacio": "No Referido"
    })

    # Limpieza de "Seleccione tipo de cargo que mas se parece"
    df["Seleccione tipo de cargo que mas se parece"] = df["Seleccione tipo de cargo que mas se parece"].replace({
        "Auxiliar, asistente administrativo, asistente técnico.": "Auxiliar, asistente administrativo, asistente técnico",
        "Profesional; analista; técnico; tecnólogo": "Profesional, analista, técnico, tecnólogo"
    })

    # Limpieza de "Estrato según servicios Públicos"
    df["Estrato según servicios Públicos"] = df["Estrato según servicios Públicos"].replace({
        "No sé": "No referido",
        "Finca": "No referido",
        "": "No referido",
        "(1,No referido)": "No referido",
        "(3,No referido)": "No referido",
        "FINCA": "No referido",
        "finca": "No referido",
        "No Refiere": "No referido",
        "No refiere": "No referido",
        "NO SE": "No referido",
        "0": "No referido",
        "(3,Finca)": "No referido",
        "(1,No sé)": "No referido",
    })

    return df

@st.cache_data
def procesar_datos_base(df):
    """Procesa los datos básicos sin predicciones"""
    # =========================
    # Codificación
    # =========================
    codificacion_riesgo = {
        'No Referido': 0,
        'Sin Riesgo O Con Riesgo Despreciable': 1,
        'Riesgo Bajo': 2,
        'Riesgo Medio': 3,
        'Riesgo Alto': 4,
        'Riesgo Muy Alto': 5
    }

    df['Nivel de riesgo codificado'] = df['Nivel de riesgo'].map(codificacion_riesgo)

    # Variables demográficas
    variables_demograficas = [
        'Sexo', 
        'Generación', 
        'Rango de Edad', 
        'Tipo de servicio', 
        'Seleccione tipo de cargo que mas se parece', 
        'Estrato según servicios Públicos',
        'Empresa',
        'Factor a Evaluar'
    ]

    # =========================
    # Agrupación por niveles básicos
    # =========================
    agrupaciones_base = {
        'Factor': df.groupby(['Año', 'Factor'])['Nivel de riesgo codificado'].mean().reset_index(),
        'Dominio': df.groupby(['Año', 'Dominio'])['Nivel de riesgo codificado'].mean().reset_index(),
        'Dimension': df.groupby(['Año', 'Dimension'])['Nivel de riesgo codificado'].mean().reset_index()
    }

    return df, agrupaciones_base, variables_demograficas

@st.cache_data
def generar_todas_predicciones(_df, _agrupaciones_base, _variables_demograficas):
    """Genera todas las predicciones posibles de una sola vez"""
    
    predicciones = {}
    datos_con_predicciones = {}
    
    # =============================================
    # 1. PREDICCIONES GENERALES (sin demografía)
    # =============================================
    for nivel, df_nivel in _agrupaciones_base.items():
        predicciones[f'{nivel}_general'] = {}
        df_con_pred = df_nivel.copy()
        
        df_nivel_clean = df_nivel.dropna(subset=[nivel])
        
        for valor in df_nivel_clean[nivel].unique():
            df_filtrado = df_nivel_clean[df_nivel_clean[nivel] == valor]
            if len(df_filtrado) >= 2:
                X = df_filtrado[['Año']].values.reshape(-1, 1)
                y = df_filtrado['Nivel de riesgo codificado'].values
                
                try:
                    # Usar modelo más simple y rápido
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    pred = model.predict([[2026]])[0]
                    predicciones[f'{nivel}_general'][valor] = round(pred, 2)
                    
                    # Agregar predicción al DataFrame
                    nueva_fila = pd.DataFrame({
                        'Año': [2026],
                        nivel: [valor],
                        'Nivel de riesgo codificado': [round(pred, 2)]
                    })
                    df_con_pred = pd.concat([df_con_pred, nueva_fila], ignore_index=True)
                except Exception as e:
                    print(f"Error en predicción general para {nivel} - {valor}: {e}")
                    continue
        
        datos_con_predicciones[nivel] = df_con_pred

    # =============================================
    # 2. PREDICCIONES DEMOGRÁFICAS
    # =============================================
    agrupaciones_demograficas = {}
    
    for nivel in ['Factor', 'Dominio', 'Dimension']:
        predicciones[f'{nivel}_demografico'] = {}
        agrupaciones_demograficas[nivel] = {}
        
        for var_demo in _variables_demograficas:
            predicciones[f'{nivel}_demografico'][var_demo] = {}
            
            # Agrupar datos demográficos
            columnas_grupo = ['Año', nivel, var_demo]
            df_agrupado = _df.groupby(columnas_grupo)['Nivel de riesgo codificado'].mean().reset_index()
            df_agrupado = df_agrupado.dropna(subset=[nivel, var_demo])
            
            # Hacer una copia para agregar predicciones
            df_demo_con_pred = df_agrupado.copy()
            
            # Para cada combinación única de nivel y variable demográfica
            combinaciones = df_agrupado[[nivel, var_demo]].drop_duplicates()
            
            for _, row in combinaciones.iterrows():
                valor_nivel = row[nivel]
                valor_demo = row[var_demo]
                
                # Filtrar datos para esta combinación específica
                df_filtrado = df_agrupado[
                    (df_agrupado[nivel] == valor_nivel) & 
                    (df_agrupado[var_demo] == valor_demo)
                ]
                
                if len(df_filtrado) >= 2:
                    X = df_filtrado[['Año']].values.reshape(-1, 1)
                    y = df_filtrado['Nivel de riesgo codificado'].values
                    
                    try:
                        model = LinearRegression()
                        model.fit(X, y)
                        pred = model.predict([[2026]])[0]
                        
                        # Guardar predicción
                        clave = f"{valor_nivel}|{valor_demo}"
                        predicciones[f'{nivel}_demografico'][var_demo][clave] = round(pred, 2)
                        
                        # Agregar predicción al DataFrame demográfico
                        nueva_fila = pd.DataFrame({
                            'Año': [2026],
                            nivel: [valor_nivel],
                            var_demo: [valor_demo],
                            'Nivel de riesgo codificado': [round(pred, 2)]
                        })
                        df_demo_con_pred = pd.concat([df_demo_con_pred, nueva_fila], ignore_index=True)
                        
                    except Exception as e:
                        print(f"Error en predicción demográfica para {nivel}-{var_demo}: {e}")
                        continue
            
            # Guardar agrupación demográfica con predicciones
            agrupaciones_demograficas[nivel][var_demo] = df_demo_con_pred

    return predicciones, datos_con_predicciones, agrupaciones_demograficas

@st.cache_data
def obtener_datos_filtrados(_datos_con_predicciones, _agrupaciones_demograficas, _domain_to_factor, _dimension_to_factor,
                          factor_sel, dominio_sel, dimension_sel, var_demo, valor_demo, comparar_demo):
    """Obtiene los datos filtrados con predicciones pre-computadas"""
    
    dfs = []
    
    if var_demo == 'Sin filtro demográfico':
        # =============================================
        # DATOS GENERALES (sin variables demográficas)
        # =============================================
        
        if factor_sel == 'Todos':
            # Mostrar todos los niveles
            for nivel in ['Factor', 'Dominio', 'Dimension']:
                df_nivel = _datos_con_predicciones[nivel].copy()
                df_nivel['Grupo'] = df_nivel[nivel]
                dfs.append(df_nivel[['Año', 'Grupo', 'Nivel de riesgo codificado']])
        
        else:
            # Factor específico seleccionado
            # 1. Mostrar el factor seleccionado
            df_f = _datos_con_predicciones['Factor'][
                _datos_con_predicciones['Factor']['Factor'] == factor_sel
            ].copy()
            df_f['Grupo'] = df_f['Factor']
            dfs.append(df_f[['Año', 'Grupo', 'Nivel de riesgo codificado']])

            # 2. Mostrar dominios relacionados
            if dominio_sel == 'Todos':
                dominios = [d for d, f in _domain_to_factor.items() if f == factor_sel]
            else:
                dominios = [dominio_sel]

            df_d = _datos_con_predicciones['Dominio'][
                _datos_con_predicciones['Dominio']['Dominio'].isin(dominios)
            ].copy()
            df_d['Grupo'] = df_d['Dominio']
            dfs.append(df_d[['Año', 'Grupo', 'Nivel de riesgo codificado']])

            # 3. Mostrar dimensiones relacionadas
            if dimension_sel == 'Todos':
                dimensiones = [d for d, f in _dimension_to_factor.items() if f == factor_sel]
            else:
                dimensiones = [dimension_sel]

            df_dim = _datos_con_predicciones['Dimension'][
                _datos_con_predicciones['Dimension']['Dimension'].isin(dimensiones)
            ].copy()
            df_dim['Grupo'] = df_dim['Dimension']
            dfs.append(df_dim[['Año', 'Grupo', 'Nivel de riesgo codificado']])
    
    else:
        # =============================================
        # DATOS CON VARIABLES DEMOGRÁFICAS
        # =============================================
        
        # Determinar qué niveles mostrar
        niveles_a_mostrar = []
        if factor_sel != 'Todos':
            niveles_a_mostrar.append(('Factor', factor_sel))
        if dominio_sel != 'Todos':
            niveles_a_mostrar.append(('Dominio', dominio_sel))
        if dimension_sel != 'Todos':
            niveles_a_mostrar.append(('Dimension', dimension_sel))
        
        if not niveles_a_mostrar:
            niveles_a_mostrar = [('Factor', 'Todos'), ('Dominio', 'Todos'), ('Dimension', 'Todos')]
        
        for nivel_tipo, nivel_valor in niveles_a_mostrar:
            if nivel_tipo in _agrupaciones_demograficas and var_demo in _agrupaciones_demograficas[nivel_tipo]:
                df_demo = _agrupaciones_demograficas[nivel_tipo][var_demo].copy()
                
                # Aplicar filtros de nivel
                if nivel_valor != 'Todos':
                    df_demo = df_demo[df_demo[nivel_tipo] == nivel_valor]
                
                # Aplicar filtros demográficos
                if valor_demo != 'Todos' and not comparar_demo:
                    df_demo = df_demo[df_demo[var_demo] == valor_demo]
                
                # Crear grupos para el gráfico
                if comparar_demo:
                    df_demo['Grupo'] = df_demo[nivel_tipo].astype(str) + ' - ' + df_demo[var_demo].astype(str)
                else:
                    df_demo['Grupo'] = df_demo[nivel_tipo]
                
                dfs.append(df_demo[['Año', 'Grupo', 'Nivel de riesgo codificado']])
    
    # Combinar todos los DataFrames
    if dfs:
        df_combinado = pd.concat(dfs, ignore_index=True)
        df_combinado = df_combinado.drop_duplicates()
        return df_combinado
    else:
        return pd.DataFrame(columns=['Año', 'Grupo', 'Nivel de riesgo codificado'])

# =========================
# Funciones auxiliares
# =========================
def obtener_color_riesgo(valor):
    if valor < 0.5:
        return "lightgray"
    elif valor < 1.5:
        return "#149e11"
    elif valor < 2.5:
        return "#92d050"
    elif valor < 3.5:
        return "#ffff00"
    elif valor < 4.5:
        return "red"
    else:
        return "darkred"

# =========================
# Diccionario de segmentación
# =========================
segmentacion = {
    # Factores globales
    'Resultado General Cuestionario Intralaboral': ('Factor Intralaboral', None, None),
    'Puntaje Total Extralaboral': ('Factor Extralaboral', None, None),
    'Estrés': ('Percepción del Estrés', None, None),
    'Intralaboral + Extralaboral': ('General Intralaboral + Extralaboral', None, None),

    # Intralaboral > Liderazgo y relaciones sociales en el trabajo
    'Caracteristicas de Liderazgo': ('Factor Intralaboral', 'Liderazgo y relaciones sociales en el trabajo', 'Características de Liderazgo'),
    'Relaciones Sociales en el Trabajo': ('Factor Intralaboral', 'Liderazgo y relaciones sociales en el trabajo', 'Relaciones Sociales en el Trabajo'),
    'Retroalimentación del desempeño': ('Factor Intralaboral', 'Liderazgo y relaciones sociales en el trabajo', 'Retroalimentación del desempeño'),
    'Relación con los Colaboradores': ('Factor Intralaboral', 'Liderazgo y relaciones sociales en el trabajo', 'Relación con los Colaboradores'),

    # Intralaboral > Control sobre el trabajo
    'Claridad del Rol': ('Factor Intralaboral', 'Control sobre el trabajo', 'Claridad del Rol'),
    'Capacitación': ('Factor Intralaboral', 'Control sobre el trabajo', 'Capacitación'),
    'Participación y Manejo del Cambio': ('Factor Intralaboral', 'Control sobre el trabajo', 'Participación y Manejo del Cambio'),
    'Oport para el uso y desarrollo de habilidades y conocimientos Riesgo': ('Factor Intralaboral', 'Control sobre el trabajo', 'Oportunidades para el uso y desarrollo de habilidades y conocimientos Riesgo'),
    'Control y Automia Sobre Trabajo': ('Factor Intralaboral', 'Control sobre el trabajo', 'Control y Autonomía Sobre Trabajo'),

    # Intralaboral > Demandas del trabajo
    'Demandas ambientales y de esfuerzo fisico': ('Factor Intralaboral', 'Demandas del trabajo', 'Demandas ambientales y de esfuerzo físico'),
    'Demandas emocionales': ('Factor Intralaboral', 'Demandas del trabajo', 'Demandas emocionales'),
    'Demandas Cuentitativas': ('Factor Intralaboral', 'Demandas del trabajo', 'Demandas Cuantitativas'),
    'Influencia del trabajo sobre el entorno extralaboral': ('Factor Intralaboral', 'Demandas del trabajo', 'Influencia del trabajo sobre el entorno extralaboral'),
    'Exigencias de responsabilidad del cargo': ('Factor Intralaboral', 'Demandas del trabajo', 'Exigencias de responsabilidad del cargo'),
    'Demandas de carga mental': ('Factor Intralaboral', 'Demandas del trabajo', 'Demandas de carga mental'),
    'Consistencia del Rol': ('Factor Intralaboral', 'Demandas del trabajo', 'Consistencia del Rol'),
    'Demandas de la jornada de trabajo': ('Factor Intralaboral', 'Demandas del trabajo', 'Demandas de la jornada de trabajo'),

    # Intralaboral > Recompensas
    'Recompensas Derivadas de la pertenencia a la organización y del trabajo que se realiza': ('Factor Intralaboral', 'Recompensas', 'Recompensas Derivadas de la pertenencia a la organización y del trabajo que se realiza'),
    'Reconociemiento y comensación': ('Factor Intralaboral', 'Recompensas', 'Reconocimiento y compensación'),

    # Factor extralaboral
    'Tiempo fuera del trabajo': ('Factor Extralaboral', None, 'Tiempo fuera del trabajo'),
    'Relaciones Familiares': ('Factor Extralaboral', None, 'Relaciones Familiares'),
    'Comunicación y Relaciones Interpersonales': ('Factor Extralaboral', None, 'Comunicación y Relaciones Interpersonales'),
    'Situación Economica del grupo Familiar': ('Factor Extralaboral', None, 'Situación Económica del grupo Familiar'),
    'Caracteristicas de la vivienda y su entorno': ('Factor Extralaboral', None, 'Características de la vivienda y su entorno'),
    'Influencia del entonor extralaboral sobre el trabajo': ('Factor Extralaboral', None, 'Influencia del entorno extralaboral sobre el trabajo'),
    'Desplazamiento Vivienda Trabajo vivienda': ('Factor Extralaboral', None, 'Desplazamiento Vivienda-Trabajo-Vivienda'),
}

# Crear mapeos para dominio y dimensión que asocien al factor
domain_to_factor = {}
dimension_to_factor = {}

for key, (f, d, dim) in segmentacion.items():
    if d is not None:
        domain_to_factor[d] = f
    if dim is not None:
        dimension_to_factor[dim] = f

# =========================
# Interfaz de Streamlit
# =========================

def main():
    st.title("🧠 Dashboard de Análisis de Riesgo Psicosocial")
    st.subheader("📈 Tendencias y Predicciones para 2026")
    
    # =========================
    # CARGA DE DATOS (OPTIMIZADA)
    # =========================
    
    # Paso 1: Cargar datos base
    with st.spinner("Cargando datos base..."):
        df = cargar_datos()
        df, agrupaciones_base, variables_demograficas = procesar_datos_base(df)
    
    # Paso 2: Generar todas las predicciones una sola vez
    with st.spinner("Generando predicciones para 2026... (esto toma unos segundos)"):
        predicciones, datos_con_predicciones, agrupaciones_demograficas = generar_todas_predicciones(
            df, agrupaciones_base, variables_demograficas
        )
    
    st.success("¡Datos y predicciones cargados exitosamente! 🎉")
    
    # =========================
    # SIDEBAR PARA FILTROS
    # =========================
    
    st.sidebar.header("🎯 Filtros de Niveles")
    
    # Filtro de Factor
    factores = ['Todos'] + sorted([f for f in datos_con_predicciones['Factor']['Factor'].dropna().unique()])
    factor_seleccionado = st.sidebar.selectbox("🔎 Selecciona el Factor:", factores)
    
    # Filtro de Dominio (dinámico basado en factor)
    if factor_seleccionado == 'Todos':
        dominios_disponibles = sorted(datos_con_predicciones['Dominio']['Dominio'].dropna().unique())
    else:
        dominios_disponibles = sorted([d for d, f in domain_to_factor.items() if f == factor_seleccionado])
    
    dominios = ['Todos'] + dominios_disponibles
    dominio_seleccionado = st.sidebar.selectbox("🗂 Selecciona el Dominio:", dominios)
    
    # Filtro de Dimensión (dinámico basado en dominio/factor)
    if dominio_seleccionado == 'Todos':
        if factor_seleccionado == 'Todos':
            dimensiones_disponibles = sorted(datos_con_predicciones['Dimension']['Dimension'].dropna().unique())
        else:
            dimensiones_disponibles = sorted([dim for dim, f in dimension_to_factor.items() if f == factor_seleccionado])
    else:
        dimensiones_disponibles = sorted([
            dim for dim, f in dimension_to_factor.items()
            if f == factor_seleccionado and segmentacion.get(dim, (None, None, None))[1] == dominio_seleccionado
        ])
    
    dimensiones = ['Todos'] + dimensiones_disponibles
    dimension_seleccionada = st.sidebar.selectbox("📊 Selecciona la Dimensión:", dimensiones)
    
    # Filtros demográficos
    st.sidebar.header("👥 Filtros Demográficos")
    
    variables_demo_opciones = ['Sin filtro demográfico'] + variables_demograficas
    variable_demografica = st.sidebar.selectbox("🧬 Variable Demográfica:", variables_demo_opciones)
    
    if variable_demografica != 'Sin filtro demográfico':
        valores_demo = ['Todos'] + sorted(df[variable_demografica].dropna().unique())
        valor_demografico = st.sidebar.selectbox("💡 Valor de la Variable:", valores_demo)
        
        comparar_demograficos = st.sidebar.checkbox("📊 Comparar todos los valores de la variable seleccionada")
    else:
        valor_demografico = 'Todos'
        comparar_demograficos = False
    
    # =========================
    # GENERAR GRÁFICO (SUPER RÁPIDO AHORA)
    # =========================
    
    try:
        # Obtener datos filtrados (ya con predicciones incluidas)
        df_combinado = obtener_datos_filtrados(
            datos_con_predicciones, agrupaciones_demograficas, domain_to_factor, dimension_to_factor,
            factor_seleccionado, dominio_seleccionado, dimension_seleccionada, 
            variable_demografica, valor_demografico, comparar_demograficos
        )
        
        # Crear título del gráfico
        titulo = 'Tendencia del Nivel de Riesgo'
        if variable_demografica != 'Sin filtro demográfico':
            titulo += f' - {variable_demografica}'
            if valor_demografico != 'Todos' and not comparar_demograficos:
                titulo += f': {valor_demografico}'
        
        # Crear gráfico
        if df_combinado.empty:
            st.warning("No hay datos disponibles para los filtros seleccionados")
        else:
            fig = px.line(
                df_combinado,
                x='Año',
                y='Nivel de riesgo codificado',
                color='Grupo',
                markers=True,
                title=titulo
            )

            # Añadir franjas de riesgo en el fondo
            fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=-0.5, y1=0.5, 
                          fillcolor="#E0E0E0", opacity=0.2, layer="below", line_width=0)
            fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=0.5, y1=1.5, 
                          fillcolor="#149e11", opacity=0.2, layer="below", line_width=0)
            fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=1.5, y1=2.5, 
                          fillcolor="#92d050", opacity=0.2, layer="below", line_width=0)
            fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=2.5, y1=3.5, 
                          fillcolor="#ffff00", opacity=0.2, layer="below", line_width=0)
            fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=3.5, y1=4.5, 
                          fillcolor="red", opacity=0.2, layer="below", line_width=0)
            fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=4.5, y1=5.5, 
                          fillcolor="darkred", opacity=0.2, layer="below", line_width=0)

            # Añadir anotaciones para las franjas
            fig.add_annotation(x=0.02, y=0, text="No Referido", showarrow=False, xref="paper", yref="y")
            fig.add_annotation(x=0.02, y=1, text="Sin Riesgo", showarrow=False, xref="paper", yref="y")
            fig.add_annotation(x=0.02, y=2, text="Riesgo Bajo", showarrow=False, xref="paper", yref="y")
            fig.add_annotation(x=0.02, y=3, text="Riesgo Medio", showarrow=False, xref="paper", yref="y")
            fig.add_annotation(x=0.02, y=4, text="Riesgo Alto", showarrow=False, xref="paper", yref="y")
            fig.add_annotation(x=0.02, y=5, text="Riesgo Muy Alto", showarrow=False, xref="paper", yref="y")

            fig.update_layout(
                template='plotly_white', 
                title_x=0.5, 
                yaxis_title="Nivel de riesgo codificado",
                xaxis_title="Año",
                hovermode='x unified',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # =========================
            # ESTADÍSTICAS ADICIONALES
            # =========================
            
            if not df_combinado.empty:
                st.header("📊 Estadísticas Adicionales")
                
                # Predicciones para 2026
                df_2026 = df_combinado[df_combinado['Año'] == 2026]
                
                if not df_2026.empty:
                    st.subheader("🔮 Predicciones para 2026")
                    
                    # Crear tabla de predicciones
                    tabla_datos = []
                    for grupo in df_2026['Grupo'].unique():
                        valor = df_2026[df_2026['Grupo'] == grupo]['Nivel de riesgo codificado'].iloc[0]
                        
                        # Determinar el nivel de riesgo textual
                        if valor < 0.5:
                            nivel_texto = "No Referido"
                        elif valor < 1.5:
                            nivel_texto = "Sin Riesgo"
                        elif valor < 2.5:
                            nivel_texto = "Riesgo Bajo"
                        elif valor < 3.5:
                            nivel_texto = "Riesgo Medio"
                        elif valor < 4.5:
                            nivel_texto = "Riesgo Alto"
                        else:
                            nivel_texto = "Riesgo Muy Alto"
                        
                        tabla_datos.append({
                            'Grupo': grupo,
                            'Valor Predicho': f"{valor:.2f}",
                            'Nivel de Riesgo': nivel_texto
                        })
                    
                    df_tabla = pd.DataFrame(tabla_datos)
                    st.dataframe(df_tabla, use_container_width=True)
                    
                    # Análisis de tendencias
                    st.subheader("📈 Análisis de Tendencias")
                    
                    tendencias = []
                    for grupo in df_combinado['Grupo'].unique():
                        df_grupo = df_combinado[df_combinado['Grupo'] == grupo]
                        if len(df_grupo) >= 2:
                            años = df_grupo['Año'].values
                            valores = df_grupo['Nivel de riesgo codificado'].values
                            
                            if len(años) > 1:
                                # Ordenar por año para calcular tendencia correctamente
                                orden = np.argsort(años)
                                años_ord = años[orden]
                                valores_ord = valores[orden]
                                
                                pendiente = (valores_ord[-1] - valores_ord[0]) / (años_ord[-1] - años_ord[0])
                                
                                if pendiente > 0.1:
                                    tendencia = "↑ Aumentando"
                                    color = "🔴"
                                elif pendiente < -0.1:
                                    tendencia = "↓ Disminuyendo"
                                    color = "🟢"
                                else:
                                    tendencia = "→ Estable"
                                    color = "🟡"
                                
                                tendencias.append(f"{color} **{grupo}**: {tendencia} ({pendiente:.3f} puntos/año)")
                    
                    if tendencias:
                        for tendencia in tendencias:
                            st.write(f"{tendencia}")
                    
                    # Información de rendimiento
                    st.sidebar.info(f"🚀 **Optimización activa**\n\nPredicciones pre-calculadas para máxima velocidad.\n\n📊 Datos mostrados: {len(df_combinado)} registros")
    
    except Exception as e:
        st.error(f"Error al generar el gráfico: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
