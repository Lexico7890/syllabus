# Medidas de Dispersión

Las **medidas de dispersión** indican qué tan dispersos o alejados están los datos de un conjunto respecto a un valor central (como la media). Son esenciales para entender la variabilidad y consistencia de los datos en estadística y ciencia de datos.

---

## 1. Rango
El **rango** es la diferencia entre el valor máximo y el valor mínimo de un conjunto de datos.

**Fórmula:**
\[
R = X_{\text{max}} - X_{\text{min}}
\]

**Ejemplo:**
Datos: 5, 8, 12, 15  
\[
R = 15 - 5 = 10
\]

**Ventajas:**
- Fácil de calcular.
- Útil para una visión rápida de la dispersión.

**Desventajas:**
- Solo considera los valores extremos.
- Sensible a outliers.

---

## 2. Varianza
La **varianza** mide el promedio de las diferencias al cuadrado entre cada dato y la media.

**Fórmula (poblacional):**
\[
\sigma^2 = \frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}
\]

**Fórmula (muestral):**
\[
s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1}
\]

**Ejemplo:**
Datos: 2, 4, 6  
Media = 4  
Varianza poblacional:
\[
\sigma^2 = \frac{(2-4)^2 + (4-4)^2 + (6-4)^2}{3} = \frac{4 + 0 + 4}{3} = 2.67
\]

---

## 3. Desviación estándar
La **desviación estándar** es la raíz cuadrada de la varianza y expresa la dispersión en las mismas unidades que los datos originales.

**Fórmula:**
\[
\sigma = \sqrt{\sigma^2}
\]
(poblacional) o  
\[
s = \sqrt{s^2}
\]
(muestral)

**Ejemplo:**
Siguiendo el ejemplo anterior:
\[
\sigma = \sqrt{2.67} \approx 1.63
\]

**Ventajas:**
- Más interpretable que la varianza.
- Usada ampliamente en estadística y machine learning.

---

## 4. Rango intercuartílico (IQR)
El **IQR** mide la dispersión de la mitad central de los datos. Se calcula como la diferencia entre el tercer cuartil (Q3) y el primer cuartil (Q1).

**Fórmula:**
\[
IQR = Q3 - Q1
\]

**Ejemplo:**
Datos ordenados: 1, 3, 5, 7, 9, 11, 13  
Q1 = 3, Q3 = 11  
\[
IQR = 11 - 3 = 8
\]

**Ventajas:**
- No se ve afectado por outliers.
- Útil para distribuciones asimétricas.

---

## Comparación Rápida

| Medida              | Sensibilidad a outliers | Uso principal                             |
|---------------------|-------------------------|-------------------------------------------|
| Rango               | Alta                    | Visión rápida de dispersión               |
| Varianza            | Alta                    | Análisis estadístico profundo             |
| Desviación estándar | Alta                    | Comparar variabilidad en la misma escala  |
| IQR                 | Baja                    | Análisis robusto contra valores extremos  |

---

## Aplicaciones en Data Science
- Identificar consistencia o variabilidad en los datos.
- Detectar anomalías o outliers.
- Evaluar el riesgo en modelos predictivos.
- Selección de características (features) en machine learning.

---
**Nota:** Una baja dispersión indica datos más consistentes, mientras que una alta dispersión muestra gran variabilidad.

