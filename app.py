"""
Taller individual: ACP y K-Means
Perfiles de clientes de tarjeta de crédito

Todo el análisis se calcula aquí mismo cada vez que se abre la aplicación:
  1. carga y limpieza de datos        -> cargar_datos(), preparar()
  2. estandarización y ACP            -> ajustar_acp()
  3. comparación de K y K-Means       -> comparar_k(), ajustar_kmeans()
Las pantallas (al final del archivo) solo muestran esos resultados.

Ejecutar en el computador:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler, normalize

# --------------------------------------------------------------------------
# Configuración general
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Perfiles de clientes de tarjeta de crédito | ACP y K-Means",
    page_icon="💳",
    layout="wide",
)

RUTA_DATOS = Path(__file__).parent / "data" / "cc_general.csv"
SEMILLA = 42          # fija los resultados para que sean reproducibles
K_MIN, K_MAX = 2, 8   # rango de K que se compara

# Diccionario de variables: nombre en la base -> (nombre corto, descripción)
VARIABLES = {
    "BALANCE": ("Saldo", "Saldo promedio que el cliente debe en la tarjeta (USD)"),
    "BALANCE_FREQUENCY": ("Frec. saldo", "Qué tan seguido se actualiza el saldo (0 a 1)"),
    "PURCHASES": ("Compras", "Monto total de compras (USD)"),
    "ONEOFF_PURCHASES": ("Compras de contado", "Monto de compras en un solo pago (USD)"),
    "INSTALLMENTS_PURCHASES": ("Compras a cuotas", "Monto de compras diferidas a cuotas (USD)"),
    "CASH_ADVANCE": ("Avances", "Monto retirado como avance en efectivo (USD)"),
    "PURCHASES_FREQUENCY": ("Frec. compras", "Qué tan seguido compra (0 a 1)"),
    "ONEOFF_PURCHASES_FREQUENCY": ("Frec. contado", "Qué tan seguido compra de contado (0 a 1)"),
    "PURCHASES_INSTALLMENTS_FREQUENCY": ("Frec. cuotas", "Qué tan seguido compra a cuotas (0 a 1)"),
    "CASH_ADVANCE_FREQUENCY": ("Frec. avances", "Qué tan seguido pide avances (0 a 1)"),
    "CASH_ADVANCE_TRX": ("N.º avances", "Número de transacciones de avance"),
    "PURCHASES_TRX": ("N.º compras", "Número de transacciones de compra"),
    "CREDIT_LIMIT": ("Cupo", "Cupo de la tarjeta (USD)"),
    "PAYMENTS": ("Pagos", "Monto total pagado por el cliente (USD)"),
    "MINIMUM_PAYMENTS": ("Pagos mínimos", "Monto de pagos mínimos realizados (USD)"),
    "PRC_FULL_PAYMENT": ("% pago total", "Proporción de meses en que pagó el total (0 a 1)"),
    "TENURE": ("Antigüedad", "Meses de permanencia con la tarjeta"),
}
NOMBRE = {k: v[0] for k, v in VARIABLES.items()}
TODAS = list(VARIABLES)

# Variables de monto o conteo: muy asimétricas, candidatas a logaritmo
ASIMETRICAS = [
    "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE", "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT",
    "PAYMENTS", "MINIMUM_PAYMENTS",
]

# Colores: un color fijo por grupo (siempre el mismo para la misma letra)
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
LETRAS = list("ABCDEFGH")
COLOR_GRUPO = {f"Grupo {l}": c for l, c in zip(LETRAS, COLORES)}
DIVERGENTE = [[0.0, "#1c5cab"], [0.5, "#f0efec"], [1.0, "#d03b3b"]]
AZUL, GRIS, TINTA = "#2a78d6", "#898781", "#0b0b0b"


def estilo(fig, titulo, x=None, y=None, alto=420):
    """Aplica el mismo estilo a todos los gráficos."""
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=16, color=TINTA)),
        xaxis_title=x, yaxis_title=y, height=alto,
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        font=dict(color="#52514e", size=13),
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    fig.update_xaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", zeroline=False)
    fig.update_yaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", zeroline=False)
    return fig


def mostrar(fig):
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------
# 1. Datos y preparación
# --------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    return pd.read_csv(RUTA_DATOS)


@st.cache_data
def preparar(variables, faltantes, usar_log):
    """Limpia la base y devuelve:
    - original: variables en sus unidades reales (para describir los grupos)
    - modelo:   variables transformadas que entran al ACP
    - notas:    conteos de lo que se corrigió
    """
    df = cargar_datos()
    notas = {"filas_iniciales": len(df)}

    # a) duplicados
    notas["duplicados"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # b) el identificador no entra al modelo
    datos = df[list(variables)].copy()

    # c) inconsistencia: una frecuencia no puede ser mayor que 1
    if "CASH_ADVANCE_FREQUENCY" in datos:
        fuera = datos["CASH_ADVANCE_FREQUENCY"] > 1
        notas["frecuencias_corregidas"] = int(fuera.sum())
        datos.loc[fuera, "CASH_ADVANCE_FREQUENCY"] = 1.0
    else:
        notas["frecuencias_corregidas"] = 0

    # d) valores faltantes
    notas["celdas_faltantes"] = int(datos.isna().sum().sum())
    notas["filas_con_faltantes"] = int(datos.isna().any(axis=1).sum())
    if faltantes == "Imputar con la mediana":
        datos = datos.fillna(datos.median())
    elif faltantes == "Reemplazar por cero":
        datos = datos.fillna(0)
    else:
        datos = datos.dropna()
    notas["filas_finales"] = len(datos)

    # e) logaritmo para reducir asimetría (log(1 + x) admite ceros)
    modelo = datos.copy()
    if usar_log:
        cols = [c for c in ASIMETRICAS if c in modelo]
        modelo[cols] = np.log1p(modelo[cols])
    return datos, modelo, notas


# --------------------------------------------------------------------------
# 2. Estandarización y ACP
# --------------------------------------------------------------------------
@st.cache_data
def ajustar_acp(modelo):
    """Estandariza (media 0, desviación 1) y calcula todos los componentes."""
    z = StandardScaler().fit_transform(modelo)
    acp = PCA(random_state=SEMILLA).fit(z)
    etiquetas = [f"CP{i + 1}" for i in range(acp.n_components_)]
    puntajes = pd.DataFrame(acp.transform(z), columns=etiquetas, index=modelo.index)
    cargas = pd.DataFrame(acp.components_.T, columns=etiquetas,
                          index=[NOMBRE[c] for c in modelo.columns])
    varianza = pd.DataFrame({
        "Componente": etiquetas,
        "Autovalor": acp.explained_variance_,
        "Varianza explicada (%)": acp.explained_variance_ratio_ * 100,
        "Acumulada (%)": acp.explained_variance_ratio_.cumsum() * 100,
    })
    z = pd.DataFrame(z, columns=modelo.columns, index=modelo.index)
    return z, puntajes, cargas, varianza


def componentes_segun(varianza, criterio, umbral, manual):
    if criterio == "Varianza acumulada":
        return int((varianza["Acumulada (%)"] >= umbral).idxmax()) + 1
    if criterio == "Kaiser (autovalor > 1)":
        return max(2, int((varianza["Autovalor"] > 1).sum()))
    return manual


# --------------------------------------------------------------------------
# 3. K-Means
# --------------------------------------------------------------------------
@st.cache_data
def comparar_k(espacio):
    """Inercia (codo) y silueta para cada K del rango."""
    filas = []
    muestra = min(3000, len(espacio))  # la silueta se estima sobre una muestra
    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(espacio)
        sil = silhouette_score(espacio, km.labels_, sample_size=muestra,
                               random_state=SEMILLA)
        filas.append({"K": k, "Inercia": km.inertia_, "Silueta": sil})
    return pd.DataFrame(filas)


@st.cache_data
def ajustar_kmeans(espacio, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(espacio)
    # Los grupos se nombran con letras, de mayor a menor tamaño: la letra
    # es solo un nombre, no un orden ni una calificación.
    orden = pd.Series(km.labels_).value_counts().index
    letra = {viejo: f"Grupo {LETRAS[i]}" for i, viejo in enumerate(orden)}
    grupos = pd.Series([letra[x] for x in km.labels_], index=espacio.index, name="Grupo")
    sil = pd.Series(silhouette_samples(espacio, km.labels_), index=espacio.index)
    return grupos, sil


# --------------------------------------------------------------------------
# 4. Método trabajado en clase (para comparar)
# --------------------------------------------------------------------------
@st.cache_data
def metodo_clase():
    """Reproduce las decisiones del cuaderno de la Clase 13 sobre la misma base:
    faltantes en cero, estandarización + normalización por fila,
    2 componentes y K = 3."""
    datos = cargar_datos().drop(columns="CUST_ID").fillna(0)
    x = normalize(StandardScaler().fit_transform(datos))
    acp = PCA(random_state=SEMILLA).fit(x)
    var_2 = acp.explained_variance_ratio_[:2].sum() * 100
    plano = pd.DataFrame(acp.transform(x)[:, :2], columns=["CP1", "CP2"], index=datos.index)
    km = KMeans(n_clusters=3, n_init=10, random_state=SEMILLA).fit(plano)
    sil = silhouette_score(plano, km.labels_, sample_size=min(3000, len(plano)),
                           random_state=SEMILLA)
    orden = pd.Series(km.labels_).value_counts().index
    nombre = {viejo: f"Clase {i + 1}" for i, viejo in enumerate(orden)}
    grupos = pd.Series([nombre[v] for v in km.labels_], index=datos.index, name="Método de clase")
    return datos, plano, grupos, var_2, sil


# --------------------------------------------------------------------------
# Barra lateral: navegación y decisiones de análisis
# --------------------------------------------------------------------------
st.sidebar.title("💳 Perfiles de clientes")
st.sidebar.caption("Taller individual · ACP y K-Means")
seccion = st.sidebar.radio("Sección", [
    "1. Contexto", "2. Exploración", "3. Preparación",
    "4. ACP", "5. Segmentación K-Means", "6. Comparación con la clase",
    "7. Conclusiones",
])

st.sidebar.divider()
st.sidebar.subheader("Decisiones de análisis")
st.sidebar.caption("Cambia cualquier opción y todo el análisis se recalcula.")

with st.sidebar.expander("Variables del modelo"):
    elegidas = st.multiselect("Variables", TODAS, default=TODAS,
                              format_func=lambda c: NOMBRE[c])
if len(elegidas) < 4:
    st.sidebar.error("Elige al menos 4 variables.")
    st.stop()
variables = tuple(c for c in TODAS if c in elegidas)

faltantes = st.sidebar.radio("Valores faltantes",
                             ["Imputar con la mediana", "Reemplazar por cero",
                              "Eliminar filas"])
usar_log = st.sidebar.checkbox("Aplicar logaritmo a montos y conteos", value=True)

criterio = st.sidebar.selectbox("Cómo elegir los componentes",
                                ["Varianza acumulada", "Kaiser (autovalor > 1)", "Manual"])
umbral, manual = 80, 2
if criterio == "Varianza acumulada":
    umbral = st.sidebar.slider("Varianza acumulada mínima (%)", 60, 95, 80, step=5)
elif criterio == "Manual":
    manual = st.sidebar.slider("Número de componentes", 2, len(variables), 2)

k = st.sidebar.slider("Número de grupos (K)", K_MIN, K_MAX, 5)

# --------------------------------------------------------------------------
# Cálculo (se ejecuta siempre, con las decisiones de la barra lateral)
# --------------------------------------------------------------------------
crudo = cargar_datos()
original, modelo, notas = preparar(variables, faltantes, usar_log)
z, puntajes, cargas, varianza = ajustar_acp(modelo)
n_comp = componentes_segun(varianza, criterio, umbral, manual)
espacio = puntajes.iloc[:, :n_comp]
var_retenida = varianza["Acumulada (%)"].iloc[n_comp - 1]
var_2d = varianza["Acumulada (%)"].iloc[1]
comparacion = comparar_k(espacio)
grupos, sil_ind = ajustar_kmeans(espacio, k)
sil_k = float(comparacion.loc[comparacion["K"] == k, "Silueta"].iloc[0])
k_mejor = int(comparacion.loc[comparacion["Silueta"].idxmax(), "K"])
orden_grupos = [f"Grupo {l}" for l in LETRAS[:k]]
centros = espacio.groupby(grupos).mean().loc[orden_grupos]  # centroides de K-Means


def perfil_estandarizado():
    """Promedio por grupo de las variables estandarizadas (0 = promedio general)."""
    p = z.groupby(grupos).mean().loc[orden_grupos]
    p.columns = [NOMBRE[c] for c in p.columns]
    return p


def eje(cp):
    pct = varianza.loc[varianza["Componente"] == cp, "Varianza explicada (%)"].iloc[0]
    return f"{cp} ({pct:.1f} %)"


def elegir_ejes(clave):
    """Selector de los dos componentes que se grafican (entre los conservados)."""
    opciones = list(espacio.columns)
    a, b = st.columns(2)
    ex = a.selectbox("Eje horizontal", opciones, index=0, key=f"x_{clave}")
    ey = b.selectbox("Eje vertical", opciones, index=1, key=f"y_{clave}")
    pct = varianza.set_index("Componente").loc[list({ex, ey}), "Varianza explicada (%)"].sum()
    return ex, ey, pct


def rasgos(fila, n=3):
    alto = fila.sort_values(ascending=False).head(n)
    bajo = fila.sort_values().head(n)
    return ([i for i, v in alto.items() if v > 0.25],
            [i for i, v in bajo.items() if v < -0.25])


# --------------------------------------------------------------------------
# Pantallas
# --------------------------------------------------------------------------
if seccion == "1. Contexto":
    st.title("Perfiles de clientes de tarjeta de crédito")
    st.caption("Machine Learning aplicado a Economía y Finanzas · "
               "Pontificia Universidad Javeriana Cali")
    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes", f"{len(crudo):,}".replace(",", "."))
    c2.metric("Variables cuantitativas", len(TODAS))
    c3.metric("Periodo observado", "6 meses")

    st.subheader("El problema")
    st.markdown("""
Un banco gana dinero con una tarjeta de crédito de formas muy distintas según el
cliente: con unos gana por los intereses de quien financia su saldo, con otros por
las comisiones de los avances en efectivo y con otros casi nada, porque pagan el
total cada mes. Tratar a todos igual significa ofrecer el producto equivocado a
la mayoría.

**Pregunta que orienta el ejercicio:** ¿existen perfiles de uso de la tarjeta de
crédito que se puedan distinguir a partir del comportamiento de compras, avances
y pagos, y qué caracteriza a cada uno?
""")

    st.subheader("Los datos")
    st.markdown("""
- **Fuente:** *Credit Card Dataset for Clustering*, publicado en Kaggle por Arjun
  Bhasin (`kaggle.com/datasets/arjunbhasin2013/ccdata`).
- **Qué es cada fila:** un cliente con tarjeta de crédito activa.
- **Cobertura:** 8.950 clientes, con su comportamiento resumido durante seis meses.
- **Variables:** 17 cuantitativas (saldos, compras, avances, pagos, cupo y
  antigüedad) más un identificador que no entra al modelo.
""")
    st.dataframe(pd.DataFrame(
        [(k_, v[0], v[1]) for k_, v in VARIABLES.items()],
        columns=["Variable en la base", "Nombre en la app", "Qué mide"]),
        hide_index=True, width="stretch")

    st.subheader("Limitaciones de la fuente")
    st.markdown("""
- La base no indica el banco, el país ni el año, así que los resultados no se
  pueden generalizar a un mercado específico como el colombiano.
- Cada cliente aparece ya resumido: no hay transacciones individuales ni
  evolución en el tiempo.
- No incluye datos del cliente (ingreso, edad, ocupación) ni si entró en mora.
- Los datos son anónimos y de acceso público.
""")

    st.subheader("Por qué ACP y K-Means")
    st.markdown("""
- **ACP:** muchas de las 17 variables miden lo mismo desde ángulos distintos (por
  ejemplo, monto, frecuencia y número de compras). El ACP resume esa redundancia
  en pocas dimensiones y evita que un mismo comportamiento cuente varias veces al
  medir distancias entre clientes.
- **K-Means:** la base no trae ninguna etiqueta de "tipo de cliente". K-Means
  permite explorar si los clientes se agrupan de forma natural y describir cada
  grupo, sin suponer de antemano cuántos perfiles hay.
""")

elif seccion == "2. Exploración":
    st.title("Exploración de los datos")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(crudo):,}".replace(",", "."))
    c2.metric("Columnas", crudo.shape[1])
    c3.metric("Celdas faltantes", int(crudo.isna().sum().sum()))
    c4.metric("Filas duplicadas", int(crudo.duplicated().sum()))

    st.subheader("Primeras filas")
    st.dataframe(crudo.head(10), width="stretch")

    st.subheader("Tipos de variable y faltantes")
    st.dataframe(pd.DataFrame({
        "Tipo": crudo.dtypes.astype(str),
        "Faltantes": crudo.isna().sum(),
        "% faltantes": (crudo.isna().mean() * 100).round(2),
        "Valores distintos": crudo.nunique(),
    }), width="stretch")
    st.info("`CUST_ID` es un identificador: tiene un valor distinto por fila y no "
            "describe comportamiento, por eso no entra al modelo. Los faltantes "
            "están en `MINIMUM_PAYMENTS` (313) y `CREDIT_LIMIT` (1).")

    st.subheader("Estadísticas descriptivas")
    st.dataframe(crudo[TODAS].describe().T.round(2), width="stretch")
    st.markdown("En los montos la media es muy superior a la mediana (50%) y el "
                "máximo está muy lejos del percentil 75: pocos clientes con valores "
                "enormes y muchos con valores bajos o en cero.")

    st.subheader("Distribución de una variable")
    var = st.selectbox("Variable", TODAS, format_func=lambda c: f"{NOMBRE[c]} ({c})")
    fig = px.histogram(crudo, x=var, nbins=50, color_discrete_sequence=[AZUL])
    fig.update_traces(marker_line_color="#fcfcfb", marker_line_width=1)
    mostrar(estilo(fig, f"Distribución de {NOMBRE[var]}", VARIABLES[var][1],
                   "Número de clientes", 380))
    st.caption(f"Asimetría: {crudo[var].skew():.2f}. Valores por encima de 1 indican "
               "una cola larga hacia la derecha.")

    st.subheader("Matriz de correlaciones")
    corr = crudo[TODAS].corr().round(2)
    corr.index = corr.columns = [NOMBRE[c] for c in TODAS]
    fig = px.imshow(corr, text_auto=True, color_continuous_scale=DIVERGENTE,
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_traces(textfont_size=10)
    mostrar(estilo(fig, "Correlación de Pearson entre variables (-1 a 1)", alto=640))
    st.markdown("""
Aparecen bloques de variables que se mueven juntas: el bloque de **compras**
(monto, frecuencia y número de compras) y el bloque de **avances** (monto,
frecuencia y número de avances). Esa redundancia es la razón para usar ACP antes
de segmentar.
""")

elif seccion == "3. Preparación":
    st.title("Preparación de los datos")
    st.markdown("Cada decisión se puede cambiar en la barra lateral para ver su efecto.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas iniciales", f"{notas['filas_iniciales']:,}".replace(",", "."))
    c2.metric("Filas que entran al modelo", f"{notas['filas_finales']:,}".replace(",", "."))
    c3.metric("Variables que entran al modelo", len(variables))

    st.subheader("Decisiones tomadas")
    st.markdown(f"""
1. **Identificador.** Se excluye `CUST_ID`, porque no mide comportamiento.
2. **Duplicados.** Se encontraron **{notas['duplicados']}** filas duplicadas.
3. **Inconsistencias.** `CASH_ADVANCE_FREQUENCY` es una proporción y no puede
   superar 1; **{notas['frecuencias_corregidas']}** clientes tenían valores mayores
   y se ajustaron a 1.
4. **Faltantes.** Había **{notas['celdas_faltantes']}** celdas vacías en
   **{notas['filas_con_faltantes']}** filas. Opción aplicada: **{faltantes.lower()}**.
   La mediana se prefiere a la media porque los montos son muy asimétricos, a
   eliminar filas porque así no se pierde ningún cliente, y al cero (la opción
   usada en clase) porque el cero crea un valor que no existe en el resto de la base.
5. **Logaritmo.** {"Se aplicó" if usar_log else "No se aplicó"} `log(1 + x)` a los
   montos y conteos. Sin esta transformación, unos pocos clientes con valores
   extremos dominan las distancias y K-Means termina formando grupos diminutos de
   casos atípicos.
6. **Estandarización.** Todas las variables se llevan a media 0 y desviación 1.
   Es indispensable: el cupo está en miles de dólares y las frecuencias van de 0 a
   1, y tanto ACP como K-Means dependen de la escala.
""")

    st.subheader("Efecto de la transformación")
    cand = [c for c in variables]
    var = st.selectbox("Variable", cand, format_func=lambda c: NOMBRE[c])
    a, b = st.columns(2)
    fig = px.histogram(original, x=var, nbins=50, color_discrete_sequence=[GRIS])
    fig.update_traces(marker_line_color="#fcfcfb", marker_line_width=1)
    a.plotly_chart(estilo(fig, "Antes: unidades originales", NOMBRE[var],
                          "Número de clientes", 340), width="stretch")
    fig = px.histogram(z, x=var, nbins=50, color_discrete_sequence=[AZUL])
    fig.update_traces(marker_line_color="#fcfcfb", marker_line_width=1)
    b.plotly_chart(estilo(fig, "Después: transformada y estandarizada",
                          f"{NOMBRE[var]} (desviaciones estándar)",
                          "Número de clientes", 340), width="stretch")

    st.subheader("Comprobación de la estandarización")
    chequeo = pd.DataFrame({"Media": z.mean().round(3), "Desviación": z.std(ddof=0).round(3),
                            "Asimetría antes": original.skew().round(2),
                            "Asimetría después": z.skew().round(2)})
    chequeo.index = [NOMBRE[c] for c in chequeo.index]
    st.dataframe(chequeo, width="stretch")

elif seccion == "4. ACP":
    st.title("Análisis de Componentes Principales")
    c1, c2, c3 = st.columns(3)
    c1.metric("Componentes conservados", n_comp)
    c2.metric("Varianza conservada", f"{var_retenida:.1f} %")
    c3.metric("Autovalores mayores que 1", int((varianza["Autovalor"] > 1).sum()))

    st.subheader("Varianza explicada")
    fig = go.Figure()
    fig.add_bar(x=varianza["Componente"], y=varianza["Varianza explicada (%)"],
                name="Por componente", marker_color=AZUL)
    fig.add_scatter(x=varianza["Componente"], y=varianza["Acumulada (%)"],
                    name="Acumulada", mode="lines+markers",
                    line=dict(color="#eb6834", width=2), marker=dict(size=8))
    fig.add_vline(x=n_comp - 0.5, line_dash="dot", line_color=GRIS,
                  annotation_text=f"Se conservan {n_comp}", annotation_position="top left")
    mostrar(estilo(fig, "Gráfico de sedimentación: varianza por componente y acumulada",
                   "Componente principal", "Varianza explicada (%)"))
    st.dataframe(varianza.round(2), hide_index=True, width="stretch")

    st.subheader("Cuántos componentes conservar")
    kaiser = int((varianza["Autovalor"] > 1).sum())
    st.markdown(f"""
- **Criterio aplicado:** {criterio.lower()}{f" de al menos {umbral} %" if criterio == "Varianza acumulada" else ""}.
  Resultado: **{n_comp} componentes**, que conservan **{var_retenida:.1f} %** de la
  variación de las {len(variables)} variables.
- **Criterio de Kaiser:** {kaiser} componentes tienen autovalor mayor que 1, es
  decir, aportan más información que una variable original por sí sola.
- **Sedimentación:** se busca el punto donde la curva deja de caer con fuerza;
  a partir de ahí cada componente adicional aporta poco.
- **Lectura:** los dos primeros componentes resumen **{var_2d:.1f} %** de la
  variación. Sirven para graficar, pero K-Means se aplica sobre los {n_comp}
  componentes conservados.
""")

    st.subheader("Cargas: qué significa cada componente")
    ver = cargas.iloc[:, :n_comp].round(2)
    fig = px.imshow(ver, text_auto=True, color_continuous_scale=DIVERGENTE,
                    zmin=-0.6, zmax=0.6, aspect="auto")
    mostrar(estilo(fig, "Cargas de cada variable en los componentes conservados",
                   "Componente", "Variable", 620))
    st.caption("Rojo: la variable empuja el componente hacia arriba. Azul: hacia abajo. "
               "Cerca de cero: la variable casi no participa.")

    comp = st.selectbox("Ver un componente en detalle", list(ver.columns))
    serie = cargas[comp].sort_values()
    fig = go.Figure(go.Bar(x=serie.values, y=serie.index, orientation="h",
                           marker_color=["#d03b3b" if v > 0 else "#1c5cab" for v in serie]))
    mostrar(estilo(fig, f"Cargas del {comp}", "Carga", None, 520))
    pos = ", ".join(serie.sort_values(ascending=False).head(4).index)
    neg = [i for i, v in serie.head(3).items() if v < -0.15]
    st.markdown(f"**{comp}** sube sobre todo con: {pos}."
                + (f" Baja con: {', '.join(neg)}." if neg else
                   " No tiene cargas negativas importantes."))

    st.subheader("Los clientes en el espacio de componentes")
    ex, ey, pct_plano = elegir_ejes("acp")
    fig = go.Figure(go.Scattergl(
        x=puntajes[ex], y=puntajes[ey], mode="markers",
        marker=dict(size=5, color=AZUL, opacity=0.35),
        hovertemplate=ex + ": %{x:.2f}<br>" + ey + ": %{y:.2f}<extra></extra>"))
    mostrar(estilo(fig, "Cada punto es un cliente", eje(ex), eje(ey), 520))
    st.warning(f"Este gráfico muestra solo {pct_plano:.1f} % de la información. El "
               f"{100 - pct_plano:.1f} % restante está en los demás componentes, de modo "
               "que dos clientes cercanos aquí pueden ser distintos en otras dimensiones.")

elif seccion == "5. Segmentación K-Means":
    st.title("Segmentación con K-Means")
    st.markdown(f"K-Means se aplica sobre los **{n_comp} componentes conservados** "
                f"({var_retenida:.1f} % de la variación). Así cada comportamiento pesa "
                "una sola vez y se descarta el ruido de los últimos componentes.")

    st.subheader("Comparación de valores de K")
    a, b = st.columns(2)
    for col, medida, titulo, nombre_eje in [
        (a, "Inercia", "Método del codo", "Inercia (suma de distancias al centro)"),
        (b, "Silueta", "Coeficiente de silueta", "Silueta promedio (-1 a 1)"),
    ]:
        fig = go.Figure(go.Scatter(x=comparacion["K"], y=comparacion[medida],
                                   mode="lines+markers", line=dict(color=AZUL, width=2),
                                   marker=dict(size=9)))
        fig.add_vline(x=k, line_dash="dot", line_color=GRIS, annotation_text=f"K = {k}")
        col.plotly_chart(estilo(fig, titulo, "Número de grupos (K)", nombre_eje, 360),
                         width="stretch")
    st.dataframe(comparacion.round(3), hide_index=True, width="stretch")

    resto = comparacion[comparacion["K"] > 2]
    k_alt = int(resto.loc[resto["Silueta"].idxmax(), "K"])
    st.markdown(f"""
- **Codo:** la inercia siempre baja al aumentar K. Lo que se busca es el punto
  donde agregar un grupo más deja de reducirla de forma apreciable.
- **Silueta:** el valor más alto está en **K = {k_mejor}**
  ({comparacion['Silueta'].max():.3f}). Entre las opciones con más de dos grupos, la
  mejor es **K = {k_alt}** ({resto['Silueta'].max():.3f}).
- **Elección actual:** K = {k}, con silueta de **{sil_k:.3f}**.
- **Lectura:** {"todas las siluetas son bajas (por debajo de 0,5). Los grupos existen como tendencias, pero se solapan: no son segmentos nítidos." if comparacion["Silueta"].max() < 0.5 else "hay valores de K con silueta superior a 0,5, lo que indica grupos razonablemente separados."}
""")
    st.caption("La silueta de esta tabla se estima sobre una muestra de 3.000 clientes "
               "para que la aplicación responda rápido.")

    st.subheader(f"Los {k} grupos")
    tam = grupos.value_counts().loc[orden_grupos]
    fig = go.Figure(go.Bar(x=tam.index, y=tam.values, text=tam.values,
                           textposition="outside",
                           marker_color=[COLOR_GRUPO[g] for g in tam.index]))
    mostrar(estilo(fig, "Tamaño de cada grupo", None, "Número de clientes", 340))
    st.caption("Las letras son solo nombres, asignados de mayor a menor tamaño. "
               "No indican orden, calidad ni riesgo.")

    ex, ey, pct_plano = elegir_ejes("km")
    fig = go.Figure()
    for g in orden_grupos:
        m = grupos == g
        fig.add_trace(go.Scattergl(
            x=puntajes.loc[m, ex], y=puntajes.loc[m, ey], mode="markers", name=g,
            marker=dict(size=5, color=COLOR_GRUPO[g], opacity=0.4),
            hovertemplate=g + "<br>" + ex + ": %{x:.2f}<br>" + ey + ": %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=centros[ex], y=centros[ey], mode="markers+text", name="Centroides",
        text=[g.replace("Grupo ", "") for g in centros.index], textposition="top center",
        textfont=dict(size=15, color=TINTA),
        marker=dict(symbol="x", size=14, color=TINTA, line=dict(width=2, color="#fcfcfb")),
        hovertemplate="Centroide %{text}<extra></extra>"))
    mostrar(estilo(fig, "Grupos y centroides en el espacio de componentes",
                   eje(ex), eje(ey), 560))
    st.caption(f"El plano muestra {pct_plano:.1f} % de la información; parte del solapamiento "
               "que se ve aquí se resuelve en los otros componentes. Cada X negra es el "
               "centroide: el cliente promedio de ese grupo.")

    st.subheader("Qué caracteriza a cada grupo")
    perfil = perfil_estandarizado()
    fig = px.imshow(perfil.T.round(2), text_auto=True, color_continuous_scale=DIVERGENTE,
                    zmin=-1.5, zmax=1.5, aspect="auto")
    mostrar(estilo(fig, "Perfil de cada grupo frente al promedio general",
                   None, None, 620))
    st.caption("Valores en desviaciones estándar: 0 es el promedio de todos los clientes. "
               "Rojo: el grupo está por encima. Azul: por debajo.")

    st.markdown("**Promedios en unidades originales**")
    medias = original.groupby(grupos).mean().loc[orden_grupos]
    medias.columns = [NOMBRE[c] for c in medias.columns]
    medias.insert(0, "Clientes", tam.values)
    medias.loc["Todos"] = [len(original)] + list(original.mean().values)
    st.dataframe(medias.T.round(2), width="stretch")

    st.markdown("**Rasgos que más distinguen a cada grupo**")
    for g in orden_grupos:
        alto, bajo = rasgos(perfil.loc[g])
        st.markdown(
            f"- **{g}** ({tam[g]} clientes, {tam[g] / len(grupos):.0%}): "
            + (f"por encima del promedio en {', '.join(alto)}" if alto else "sin rasgos altos marcados")
            + (f"; por debajo en {', '.join(bajo)}." if bajo else "."))

    st.subheader("Qué tan bien separado queda cada grupo")
    sil_g = sil_ind.groupby(grupos).mean().loc[orden_grupos]
    mal = (sil_ind < 0).groupby(grupos).mean().loc[orden_grupos] * 100
    fig = go.Figure(go.Bar(x=sil_g.index, y=sil_g.values,
                           text=[f"{v:.2f}" for v in sil_g], textposition="outside",
                           marker_color=[COLOR_GRUPO[g] for g in sil_g.index]))
    mostrar(estilo(fig, "Silueta promedio por grupo", None, "Silueta (-1 a 1)", 340))
    st.markdown(f"**{(sil_ind < 0).mean():.1%}** de los clientes tiene silueta negativa: "
                "están más cerca de otro grupo que del suyo y son los casos difíciles de "
                "clasificar. Por grupo: "
                + ", ".join(f"{g} {mal[g]:.0f} %" for g in orden_grupos) + ".")

elif seccion == "6. Comparación con la clase":
    st.title("Comparación con el ejercicio de clase")
    st.markdown("""
Esta base es la misma que se trabajó en la Clase 13. Para que el ejercicio no sea
una repetición, aquí se reproduce el método de clase y se contrasta con las
decisiones de este proyecto. La pregunta es: **¿cuánto dependen los perfiles de
clientes de las decisiones de preparación?**
""")
    datos_c, plano_c, grupos_c, var_c, sil_c = metodo_clase()

    st.subheader("Decisiones de cada método")
    st.dataframe(pd.DataFrame({
        "Decisión": ["Valores faltantes", "Tratamiento de escala", "Componentes",
                     "Varianza conservada", "Número de grupos", "Silueta"],
        "Método de clase": ["Reemplazar por cero", "Estandarizar y normalizar cada fila",
                            "2 (fijados)", f"{var_c:.1f} %", "3", f"{sil_c:.3f}"],
        "Este proyecto": [faltantes, ("Logaritmo y estandarizar" if usar_log else "Estandarizar"),
                          f"{n_comp} ({criterio.lower()})", f"{var_retenida:.1f} %",
                          str(k), f"{sil_k:.3f}"],
    }), hide_index=True, width="stretch")
    st.markdown(f"""
- **Faltantes.** Poner cero en `MINIMUM_PAYMENTS` supone que esos clientes no
  pagaron nada. Pero el pago mínimo más bajo observado es mayor que cero, así que
  el cero crea un valor que no existe en el resto de la base. La mediana no
  inventa un comportamiento extremo.
- **Escala.** Normalizar cada fila deja a todos los clientes con la misma
  "longitud": conserva la *forma* del comportamiento (en qué se concentra el
  cliente) y borra la *intensidad* (cuánto usa la tarjeta). El logaritmo conserva
  las dos cosas y reduce el peso de los valores extremos.
- **Componentes.** Con 2 componentes el método de clase conserva {var_c:.1f} % de la
  variación de sus datos; aquí se conservan {n_comp} con {var_retenida:.1f} %.
- **Silueta.** Las dos siluetas **no son comparables entre sí**: se calculan en
  espacios distintos. En 2 dimensiones es más fácil obtener grupos compactos
  porque se descartó el resto de la información.
""")

    st.subheader("Los grupos del método de clase")
    paleta_c = {"Clase 1": "#52514e", "Clase 2": "#898781", "Clase 3": "#c3c2b7"}
    fig = go.Figure()
    for g in ["Clase 1", "Clase 2", "Clase 3"]:
        m = grupos_c == g
        fig.add_trace(go.Scattergl(x=plano_c.loc[m, "CP1"], y=plano_c.loc[m, "CP2"],
                                   mode="markers", name=g,
                                   marker=dict(size=5, color=paleta_c[g], opacity=0.5)))
    cen = plano_c.groupby(grupos_c).mean()
    fig.add_trace(go.Scatter(x=cen["CP1"], y=cen["CP2"], mode="markers+text",
                             name="Centroides", text=[g[-1] for g in cen.index],
                             textposition="top center", textfont=dict(size=15, color=TINTA),
                             marker=dict(symbol="x", size=14, color=TINTA)))
    mostrar(estilo(fig, "Método de clase: 3 grupos sobre 2 componentes",
                   "CP1 (datos normalizados)", "CP2 (datos normalizados)", 480))
    medias_c = datos_c.groupby(grupos_c).mean()
    medias_c.columns = [NOMBRE[c] for c in medias_c.columns]
    medias_c.insert(0, "Clientes", grupos_c.value_counts().loc[medias_c.index].values)
    st.markdown("**Promedios en unidades originales**")
    st.dataframe(medias_c.T.round(2), width="stretch")

    st.subheader("¿Los dos métodos agrupan igual a los clientes?")
    comunes = grupos.index.intersection(grupos_c.index)
    cruce = pd.crosstab(grupos.loc[comunes], grupos_c.loc[comunes])
    cruce_pct = cruce.div(cruce.sum(axis=1), axis=0) * 100
    fig = px.imshow(cruce_pct.round(0), text_auto=True, aspect="auto",
                    color_continuous_scale=[[0, "#f0efec"], [1, "#1c5cab"]], zmin=0, zmax=100)
    mostrar(estilo(fig, "A qué grupo de clase va cada grupo de este proyecto (% de la fila)",
                   "Grupo según el método de clase", "Grupo según este proyecto", 380))
    principal = cruce_pct.idxmax(axis=1)
    st.markdown("Cada fila suma 100 %. Si una fila se concentra en una sola columna, "
                "ese grupo existe en los dos métodos; si se reparte, es un perfil que "
                "el método de clase no separa.")
    for g in cruce_pct.index:
        st.markdown(f"- **{g}**: {cruce_pct.loc[g, principal[g]]:.0f} % cae en "
                    f"{principal[g]}.")
    repetidos = principal.value_counts()
    fusionados = [c for c, n in repetidos.items() if n > 1]
    if fusionados:
        st.info("Varios grupos de este proyecto caen sobre todo en el mismo grupo de clase ("
                + ", ".join(fusionados) + "): con 3 grupos y 2 componentes esos perfiles "
                "quedan fusionados.")
    st.markdown("""
**Qué se aprende de la comparación.** Las grandes divisiones aparecen con los dos
métodos, lo que indica que son rasgos de los datos y no un efecto de la
preparación. Las subdivisiones más finas sí dependen de las decisiones tomadas, y
por eso deben leerse como una propuesta de segmentación, no como la única posible.
""")

else:  # 7. Conclusiones
    st.title("Conclusiones")
    perfil = perfil_estandarizado()
    tam = grupos.value_counts().loc[orden_grupos]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Componentes", n_comp)
    c2.metric("Varianza conservada", f"{var_retenida:.1f} %")
    c3.metric("Grupos (K)", k)
    c4.metric("Silueta", f"{sil_k:.3f}")

    st.subheader("Qué define los componentes principales")
    for comp in cargas.columns[:min(3, n_comp)]:
        s = cargas[comp]
        pos = ", ".join(s.sort_values(ascending=False).head(3).index)
        neg = [i for i, v in s.sort_values().head(3).items() if v < -0.15]
        pct = varianza.loc[varianza["Componente"] == comp, "Varianza explicada (%)"].iloc[0]
        st.markdown(f"- **{comp}** ({pct:.1f} %): sube con {pos}"
                    + (f" y baja con {', '.join(neg)}." if neg else "."))
    st.markdown(f"Se conservaron **{n_comp} de {len(variables)}** dimensiones con "
                f"**{var_retenida:.1f} %** de la variación original. Los dos primeros "
                f"componentes reúnen {var_2d:.1f} %: el comportamiento de estos clientes no "
                "se reduce a dos ejes.")

    st.subheader("Qué diferencia a los grupos")
    for g in orden_grupos:
        alto, bajo = rasgos(perfil.loc[g])
        st.markdown(
            f"- **{g}** ({tam[g] / len(grupos):.0%} de los clientes): "
            + (f"alto en {', '.join(alto)}" if alto else "cercano al promedio")
            + (f"; bajo en {', '.join(bajo)}." if bajo else "."))

    st.subheader("Qué tan distinguibles son")
    st.markdown(f"""
La silueta es de **{sil_k:.3f}** y **{(sil_ind < 0).mean():.1%}** de los clientes
queda más cerca de otro grupo que del suyo. {"Los grupos describen tendencias reales de uso, pero con fronteras difusas: hay muchos clientes intermedios. Los perfiles sirven para orientar una estrategia, no para clasificar a un cliente individual con certeza." if sil_k < 0.5 else "Con esta configuración los grupos quedan razonablemente separados, aunque siguen existiendo clientes intermedios."}
""")

    st.subheader("Lectura del caso")
    st.markdown("""
La pregunta era si existen perfiles de uso de la tarjeta. La respuesta es que sí,
con matices. Las dos dimensiones que más separan a los clientes son **cuánto
compran** con la tarjeta y **cuánto la usan como fuente de efectivo y deuda**
(saldo, avances y pagos mínimos). Los grupos resultan de combinar esas dos
dimensiones con la forma de compra, de contado o a cuotas.

Para un banco, esto sugiere tratamientos distintos: a quienes compran mucho y
pagan, beneficios por uso; a quienes dependen de avances y mantienen saldo alto,
seguimiento de riesgo y alternativas de financiación más baratas; a quienes casi
no usan la tarjeta, incentivos de activación.
""")

    st.subheader("Limitaciones")
    st.markdown("""
- **Estabilidad.** La asignación de muchos clientes cambia si se modifica K, las
  variables o la transformación; con siluetas bajas los grupos se solapan.
- **Sensibilidad a las decisiones.** Sin logaritmo, K-Means forma grupos muy
  pequeños de clientes extremos; el resultado depende de esa elección.
- **Supuestos de K-Means.** Busca grupos compactos y de tamaño parecido; si los
  perfiles reales tienen otra forma, no los captura bien.
- **Fuente.** No se conoce el banco, el país ni el año, y no hay datos de mora ni
  de ingresos, por lo que no se puede afirmar qué grupo es más rentable o riesgoso.
- **Foto fija.** Son seis meses resumidos: no se observa si un cliente cambia de
  perfil con el tiempo.
""")
