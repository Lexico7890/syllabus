# Medidas de Tendencia Central

Las **medidas de tendencia central** son valores estadísticos que resumen o representan un conjunto de datos, indicando el punto alrededor del cual se agrupan la mayoría de los valores. Son esenciales para describir y analizar datos en estadística y ciencia de datos.

## 1. Media (Promedio)
La **media aritmética** se obtiene sumando todos los valores de un conjunto de datos y dividiendo entre el número total de observaciones.

**Fórmula:**
\[
\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}
\]

**Ejemplo:**
Datos: 4, 6, 8  
\[
\bar{x} = \frac{4 + 6 + 8}{3} = \frac{18}{3} = 6
\]

**Ventajas:**
- Fácil de calcular y entender.
- Utiliza toda la información disponible.

**Desventajas:**
- Muy sensible a valores extremos (**outliers**).

---

## 2. Mediana
La **mediana** es el valor que se encuentra en la posición central cuando los datos están ordenados. Divide el conjunto en dos partes iguales.

**Cálculo:**
1. Ordenar los datos.
2. Si el número de observaciones es impar → la mediana es el valor central.
3. Si es par → la mediana es el promedio de los dos valores centrales.

**Ejemplo:**
Datos: 2, 5, 7, 9, 10  
Mediana = 7 (valor central).

**Ventajas:**
- No se ve afectada por valores extremos.

---

## 3. Moda
La **moda** es el valor que aparece con mayor frecuencia en un conjunto de datos.

**Ejemplo:**
Datos: 3, 4, 4, 5, 6  
Moda = 4.

**Tipos:**
- **Unimodal:** Una sola moda.
- **Bimodal:** Dos modas.
- **Multimodal:** Más de dos modas.

**Ventajas:**
- Útil para variables cualitativas o categóricas.

---

## Comparación Rápida

| Medida  | Uso Principal                                  | Sensibilidad a outliers |
|---------|-----------------------------------------------|-------------------------|
| Media   | Datos numéricos, análisis general             | Alta                    |
| Mediana | Datos con valores extremos                    | Baja                    |
| Moda    | Datos categóricos o más frecuente en numéricos| Ninguna                 |

---

## Aplicaciones en Data Science
- Resumir grandes volúmenes de datos.
- Detectar sesgos en la distribución.
- Servir como referencia para análisis comparativos.
- Preprocesamiento de datos faltantes (imputación).

---
**Nota:** En distribuciones simétricas, la media, mediana y moda tienden a ser iguales. En distribuciones sesgadas, estas medidas difieren.

