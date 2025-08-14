# Ejercicios Prácticos

Estos ejercicios te ayudarán a reforzar los conceptos de **medidas de tendencia central** y **medidas de dispersión**.  

---

## 1. Cálculo de media, mediana y moda
Dado el siguiente conjunto de datos:


**Tareas:**
1. Calcula la **media**.
2. Calcula la **mediana**.
3. Calcula la **moda**.

**Pistas:**
- Ordena los datos antes de calcular la mediana.
- La moda es el valor que más se repite.

---

## 2. Varianza y desviación estándar
Usando los mismos datos del ejercicio anterior:

**Tareas:**
1. Calcula la **varianza poblacional**.
2. Calcula la **desviación estándar poblacional**.
3. Interpreta los resultados: ¿Los datos están muy dispersos o son consistentes?

**Pistas:**
- Varianza poblacional:  
\[
\sigma^2 = \frac{\sum (x_i - \mu)^2}{N}
\]
- Desviación estándar:  
\[
\sigma = \sqrt{\sigma^2}
\]

---

## 3. Rango e IQR
Con los mismos datos:

**Tareas:**
1. Encuentra el **valor mínimo** y el **valor máximo**.
2. Calcula el **rango**.
3. Calcula el **rango intercuartílico (IQR)**.

**Pistas:**
- IQR = Q3 − Q1.
- Q1 es el valor que divide el 25% inferior de los datos.
- Q3 es el valor que divide el 25% superior de los datos.

---

## 4. Interpretación de resultados
Con los cálculos anteriores, responde:
- ¿La media y la mediana son similares? ¿Qué indica esto sobre la simetría de los datos?
- Si existen diferencias importantes entre media y mediana, ¿puede ser por la presencia de outliers?
- ¿El IQR confirma la dispersión observada en la desviación estándar?

---

## 5. Ejercicio extra (programación en Python)
Escribe un código en Python que, dado un conjunto de datos, calcule:

- Media
- Mediana
- Moda
- Rango
- Varianza
- Desviación estándar
- IQR

```python
import statistics as stats
import numpy as np

datos = [8, 12, 15, 10, 8, 14, 20, 8]

media = stats.mean(datos)
mediana = stats.median(datos)
moda = stats.mode(datos)
rango = max(datos) - min(datos)
varianza = np.var(datos)
desviacion = np.std(datos)
q1 = np.percentile(datos, 25)
q3 = np.percentile(datos, 75)
iqr = q3 - q1

print(f"Media: {media}")
print(f"Mediana: {mediana}")
print(f"Moda: {moda}")
print(f"Rango: {rango}")
print(f"Varianza: {varianza}")
print(f"Desviación estándar: {desviacion}")
print(f"IQR: {iqr}")
