# Perfiles de clientes de tarjeta de crédito: ACP y K-Means

Taller individual de aprendizaje no supervisado.
Curso: Machine Learning aplicado a Economía y Finanzas, Pontificia Universidad Javeriana Cali.

## Enlaces de entrega

- **Repositorio:** https://github.com/ladydayanarestrepoloaiza-ui/acp-kmeans-tarjetas-credito
- **Aplicación desplegada:** _pendiente: pegar aquí el enlace de Streamlit Community Cloud_
- **Video de sustentación:** _pendiente: pegar aquí el enlace del video_

## Contexto

Un banco obtiene ingresos de una tarjeta de crédito de formas distintas según el cliente: intereses de quien financia su saldo, comisiones de quien pide avances en efectivo, o muy poco de quien paga el total cada mes. Conocer esos perfiles permite ofrecer a cada cliente el producto y el seguimiento adecuados.

**Pregunta:** ¿existen perfiles de uso de la tarjeta de crédito que se puedan distinguir por el comportamiento de compras, avances y pagos, y qué caracteriza a cada uno?

## Datos

- **Fuente:** [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata), publicado en Kaggle por Arjun Bhasin.
- **Archivo:** `data/cc_general.csv` (incluido en este repositorio; en Kaggle se llama `CC GENERAL.csv`).
- **Cada fila:** un cliente con tarjeta de crédito activa.
- **Cobertura:** 8.950 clientes, con su comportamiento resumido durante seis meses.
- **Variables:** 17 cuantitativas y un identificador (`CUST_ID`) que no entra al modelo.

| Variable | Qué mide |
|---|---|
| `BALANCE` | Saldo promedio que debe el cliente (USD) |
| `BALANCE_FREQUENCY` | Qué tan seguido se actualiza el saldo (0 a 1) |
| `PURCHASES` | Monto total de compras (USD) |
| `ONEOFF_PURCHASES` | Monto de compras de contado (USD) |
| `INSTALLMENTS_PURCHASES` | Monto de compras a cuotas (USD) |
| `CASH_ADVANCE` | Monto de avances en efectivo (USD) |
| `PURCHASES_FREQUENCY` | Qué tan seguido compra (0 a 1) |
| `ONEOFF_PURCHASES_FREQUENCY` | Qué tan seguido compra de contado (0 a 1) |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | Qué tan seguido compra a cuotas (0 a 1) |
| `CASH_ADVANCE_FREQUENCY` | Qué tan seguido pide avances (0 a 1) |
| `CASH_ADVANCE_TRX` | Número de avances |
| `PURCHASES_TRX` | Número de compras |
| `CREDIT_LIMIT` | Cupo de la tarjeta (USD) |
| `PAYMENTS` | Monto total pagado (USD) |
| `MINIMUM_PAYMENTS` | Monto de pagos mínimos (USD) |
| `PRC_FULL_PAYMENT` | Proporción de meses con pago total (0 a 1) |
| `TENURE` | Meses de antigüedad con la tarjeta |

**Limitaciones de la fuente:** no indica el banco, el país ni el año; cada cliente aparece ya resumido, sin transacciones ni evolución en el tiempo; no incluye ingreso, edad ni mora.

## Por qué ACP y K-Means

- **ACP:** varias variables miden el mismo comportamiento (monto, frecuencia y número de compras; monto, frecuencia y número de avances). El ACP resume esa redundancia y evita que un mismo comportamiento pese varias veces en las distancias.
- **K-Means:** la base no trae una etiqueta de tipo de cliente. K-Means permite explorar si existen grupos y describirlos sin fijar de antemano cuántos hay.

## Metodología

Todo el procesamiento está en `app.py` y se recalcula cada vez que se abre la aplicación.

1. **Exploración:** estructura, tipos, estadísticas descriptivas, faltantes, duplicados, distribuciones y matriz de correlaciones.
2. **Preparación** (función `preparar`):
   - Se excluye el identificador `CUST_ID`.
   - No hay filas duplicadas.
   - `CASH_ADVANCE_FREQUENCY` es una proporción; 8 clientes tenían valores mayores que 1 y se ajustaron a 1.
   - Faltantes: 313 en `MINIMUM_PAYMENTS` y 1 en `CREDIT_LIMIT`, imputados con la mediana.
   - Logaritmo `log(1 + x)` en montos y conteos para reducir la asimetría.
3. **Estandarización y ACP** (función `ajustar_acp`): `StandardScaler` y `PCA` de scikit-learn. Se presentan varianza por componente, acumulada, sedimentación y cargas.
4. **K-Means** (funciones `comparar_k` y `ajustar_kmeans`): se aplica sobre los componentes conservados. Se comparan K de 2 a 8 con el método del codo y el coeficiente de silueta. `n_init=10` y `random_state=42`.
5. **Descripción de los grupos:** promedios de las variables originales por grupo y silueta por grupo.

Las decisiones (variables, faltantes, logaritmo, criterio de componentes y K) se pueden cambiar en la barra lateral de la aplicación.

## Resultados principales

Con la configuración por defecto:

- **ACP:** se conservan **6 componentes** con **84,2 %** de la variación. Cinco tienen autovalor mayor que 1. Los dos primeros reúnen 55,9 %.
  - CP1 (33,9 %): intensidad de compras frente a uso de avances.
  - CP2 (22,0 %): nivel de saldo, pagos mínimos y pagos; es decir, endeudamiento.
  - CP3 (9,4 %): compras de contado frente a compras a cuotas.
- **K-Means:** la silueta más alta está en K = 2 (0,292) y, entre las opciones con más de dos grupos, en **K = 5** (0,276), que es la elección por defecto porque separa perfiles más útiles.

| Grupo | Clientes | Rasgos principales |
|---|---|---|
| A | 2.419 (27 %) | Casi no compra; usa la tarjeta para avances en efectivo y mantiene saldo |
| B | 1.919 (21 %) | Compra mucho, sobre todo de contado; cupo alto y casi sin avances |
| C | 1.872 (21 %) | Compra con frecuencia a cuotas, montos moderados, saldo bajo |
| D | 1.451 (16 %) | Uso bajo de la tarjeta: poco saldo, pocas compras y pocos pagos |
| E | 1.289 (14 %) | Compra y además pide avances; el saldo más alto y casi nunca paga el total |

Las letras son solo nombres asignados por tamaño; no indican orden ni calidad.

- **Separación:** las siluetas son bajas (menos de 0,5) y 5,1 % de los clientes queda más cerca de otro grupo que del suyo. El grupo D es el más difuso. Los grupos describen tendencias, no segmentos nítidos.

## Limitaciones

- Los grupos se solapan y la asignación cambia con K, las variables o la transformación.
- Sin logaritmo, K-Means forma grupos muy pequeños de clientes extremos.
- K-Means supone grupos compactos y de tamaño parecido.
- Sin datos de mora ni de ingresos no se puede afirmar qué grupo es más rentable o riesgoso.
- Son seis meses resumidos: no se observa si un cliente cambia de perfil.

## Cómo ejecutar el proyecto

Se necesita Python 3.10 o superior.

```bash
git clone https://github.com/ladydayanarestrepoloaiza-ui/acp-kmeans-tarjetas-credito.git
cd acp-kmeans-tarjetas-credito
pip install -r requirements.txt
streamlit run app.py
```

La aplicación se abre en `http://localhost:8501`.

## Estructura del repositorio

```
├── app.py               # aplicación: datos, ACP, K-Means y pantallas
├── requirements.txt     # librerías necesarias
├── README.md
└── data/
    └── cc_general.csv   # base de datos
```
