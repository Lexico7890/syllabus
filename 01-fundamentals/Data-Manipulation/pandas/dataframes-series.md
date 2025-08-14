# DataFrames y Series en Pandas

## Introducción

Pandas es la librería fundamental para manipulación y análisis de datos en Python. Sus dos estructuras de datos principales son **Series** y **DataFrame**, que proporcionan herramientas poderosas para trabajar con datos estructurados.

## Series

### ¿Qué es una Serie?

Una **Series** es una estructura de datos unidimensional que puede contener cualquier tipo de dato (enteros, strings, floats, objetos Python, etc.). Es similar a una columna en una tabla o un array unidimensional con etiquetas.

### Creación de Series

```python
import pandas as pd
import numpy as np

# Crear una Serie desde una lista
serie_numeros = pd.Series([1, 3, 5, 7, 9])
print(serie_numeros)

# Serie con índice personalizado
serie_temperaturas = pd.Series([25, 28, 32, 19], 
                              index=['Lunes', 'Martes', 'Miércoles', 'Jueves'])
print(serie_temperaturas)

# Serie desde un diccionario
datos_poblacion = {'Madrid': 6600000, 'Barcelona': 1600000, 'Valencia': 800000}
serie_poblacion = pd.Series(datos_poblacion)
print(serie_poblacion)
```

### Propiedades y Métodos Importantes de Series

```python
serie = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])

# Propiedades básicas
print(f"Valores: {serie.values}")
print(f"Índice: {serie.index}")
print(f"Forma: {serie.shape}")
print(f"Tamaño: {serie.size}")
print(f"Tipo de datos: {serie.dtype}")

# Estadísticas descriptivas
print(f"Media: {serie.mean()}")
print(f"Mediana: {serie.median()}")
print(f"Desviación estándar: {serie.std()}")
print(f"Valor mínimo: {serie.min()}")
print(f"Valor máximo: {serie.max()}")
```

### Acceso a Datos en Series

```python
serie = pd.Series([100, 200, 300, 400, 500], 
                  index=['A', 'B', 'C', 'D', 'E'])

# Acceso por posición
print(serie[0])        # Primer elemento
print(serie[-1])       # Último elemento

# Acceso por etiqueta
print(serie['A'])      # Elemento con etiqueta 'A'

# Slicing
print(serie[1:4])      # Elementos del índice 1 al 3
print(serie['B':'D'])  # Elementos de 'B' a 'D' (inclusivo)

# Acceso múltiple
print(serie[['A', 'C', 'E']])  # Elementos específicos
```

## DataFrames

### ¿Qué es un DataFrame?

Un **DataFrame** es una estructura de datos bidimensional con etiquetas que puede contener columnas de diferentes tipos de datos. Es similar a una tabla SQL, una hoja de cálculo de Excel o un data.frame de R.

### Creación de DataFrames

```python
# Desde un diccionario
datos = {
    'Nombre': ['Ana', 'Luis', 'María', 'Carlos'],
    'Edad': [25, 30, 35, 28],
    'Ciudad': ['Madrid', 'Barcelona', 'Valencia', 'Sevilla'],
    'Salario': [35000, 42000, 38000, 40000]
}
df = pd.DataFrame(datos)
print(df)

# Desde una lista de diccionarios
empleados = [
    {'Nombre': 'Ana', 'Edad': 25, 'Departamento': 'IT'},
    {'Nombre': 'Luis', 'Edad': 30, 'Departamento': 'Ventas'},
    {'Nombre': 'María', 'Edad': 35, 'Departamento': 'HR'}
]
df_empleados = pd.DataFrame(empleados)
print(df_empleados)

# Desde arrays de NumPy
datos_numericos = np.random.randn(4, 3)
df_numerico = pd.DataFrame(datos_numericos, 
                          columns=['Col_A', 'Col_B', 'Col_C'],
                          index=['Fila_1', 'Fila_2', 'Fila_3', 'Fila_4'])
print(df_numerico)
```

### Propiedades y Métodos Importantes de DataFrames

```python
# Creamos un DataFrame de ejemplo
data = {
    'Producto': ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Tablet'],
    'Precio': [800, 25, 60, 300, 400],
    'Stock': [15, 100, 45, 8, 20],
    'Categoria': ['Electrónicos', 'Accesorios', 'Accesorios', 'Electrónicos', 'Electrónicos']
}
df = pd.DataFrame(data)

# Información básica
print(f"Forma del DataFrame: {df.shape}")
print(f"Columnas: {df.columns.tolist()}")
print(f"Tipos de datos:\n{df.dtypes}")
print(f"Información general:")
df.info()

# Primeras y últimas filas
print("Primeras 3 filas:")
print(df.head(3))
print("\nÚltimas 2 filas:")
print(df.tail(2))

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df.describe())
```

### Acceso a Datos en DataFrames

```python
# Selección de columnas
print("Una columna (retorna Serie):")
print(df['Producto'])

print("\nUna columna (retorna DataFrame):")
print(df[['Producto']])

print("\nVarias columnas:")
print(df[['Producto', 'Precio']])

# Selección de filas
print("\nPrimeras 3 filas:")
print(df[0:3])

# Acceso con .loc (por etiquetas)
print("\nFilas específicas con .loc:")
print(df.loc[1:3, ['Producto', 'Precio']])

# Acceso con .iloc (por posición)
print("\nFilas y columnas por posición con .iloc:")
print(df.iloc[1:4, 0:2])

# Acceso a una celda específica
print(f"\nPrecio del producto en fila 2: {df.loc[2, 'Precio']}")
print(f"Mismo valor con .iloc: {df.iloc[2, 1]}")
```

### Operaciones Básicas con DataFrames

```python
# Agregar nuevas columnas
df['Precio_con_IVA'] = df['Precio'] * 1.21
df['Valor_Total'] = df['Precio'] * df['Stock']

# Modificar valores
df.loc[df['Stock'] < 10, 'Estado'] = 'Stock Bajo'
df.loc[df['Stock'] >= 10, 'Estado'] = 'Stock Normal'

# Eliminar columnas
df_sin_columna = df.drop('Estado', axis=1)  # axis=1 para columnas
print(df_sin_columna)

# Eliminar filas
df_sin_fila = df.drop(0, axis=0)  # axis=0 para filas
print(df_sin_fila)

# Renombrar columnas
df_renombrado = df.rename(columns={'Producto': 'Nombre_Producto', 'Precio': 'Costo'})
print(df_renombrado.columns)
```

## Trabajando con Índices

### Configuración de Índices

```python
# Usar una columna como índice
df_indexed = df.set_index('Producto')
print(df_indexed)

# Resetear el índice
df_reset = df_indexed.reset_index()
print(df_reset)

# Índice múltiple
df_multi = df.set_index(['Categoria', 'Producto'])
print(df_multi)
```

### Ordenamiento

```python
# Ordenar por columna
df_ordenado = df.sort_values('Precio', ascending=False)
print("Ordenado por precio (descendente):")
print(df_ordenado)

# Ordenar por múltiples columnas
df_multi_ordenado = df.sort_values(['Categoria', 'Precio'], ascending=[True, False])
print("\nOrdenado por categoría y precio:")
print(df_multi_ordenado)

# Ordenar por índice
df_por_indice = df.sort_index()
print("\nOrdenado por índice:")
print(df_por_indice)
```

## Manejo de Valores Faltantes

```python
# Crear DataFrame con valores faltantes
datos_faltantes = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, np.nan],
    'C': [1, 2, 3, 4, 5]
}
df_nan = pd.DataFrame(datos_faltantes)

# Detectar valores faltantes
print("Valores nulos:")
print(df_nan.isnull())
print(f"\nCantidad de valores nulos por columna:")
print(df_nan.isnull().sum())

# Eliminar filas con valores faltantes
df_sin_nan = df_nan.dropna()
print(f"\nDataFrame sin NaN (filas eliminadas):")
print(df_sin_nan)

# Rellenar valores faltantes
df_rellenado = df_nan.fillna(0)
print(f"\nDataFrame con NaN rellenados con 0:")
print(df_rellenado)

# Rellenar con la media
df_media = df_nan.fillna(df_nan.mean())
print(f"\nDataFrame con NaN rellenados con la media:")
print(df_media)
```

## Casos de Uso Prácticos

### Análisis de Ventas

```python
# Crear datos de ventas
ventas_data = {
    'Fecha': pd.date_range('2024-01-01', periods=100, freq='D'),
    'Producto': np.random.choice(['Laptop', 'Mouse', 'Teclado', 'Monitor'], 100),
    'Cantidad': np.random.randint(1, 10, 100),
    'Precio_Unitario': np.random.uniform(20, 800, 100),
    'Vendedor': np.random.choice(['Ana', 'Luis', 'María', 'Carlos'], 100)
}
df_ventas = pd.DataFrame(ventas_data)
df_ventas['Ingresos'] = df_ventas['Cantidad'] * df_ventas['Precio_Unitario']

print("Resumen de ventas:")
print(df_ventas.head())
print(f"\nIngresos totales: ${df_ventas['Ingresos'].sum():.2f}")
print(f"Venta promedio: ${df_ventas['Ingresos'].mean():.2f}")
print(f"Producto más vendido: {df_ventas['Producto'].mode().iloc[0]}")
```

## Consejos y Mejores Prácticas

### 1. Exploración Inicial de Datos
```python
def explorar_datos(df):
    print(f"Forma del dataset: {df.shape}")
    print(f"\nTipos de datos:\n{df.dtypes}")
    print(f"\nValores faltantes:\n{df.isnull().sum()}")
    print(f"\nPrimeras 5 filas:")
    print(df.head())
    return df.describe()

# Usar la función
estadisticas = explorar_datos(df_ventas)
print(f"\nEstadísticas:\n{estadisticas}")
```

### 2. Optimización de Memoria
```python
# Verificar uso de memoria
print(f"Uso de memoria: {df_ventas.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Optimizar tipos de datos
df_optimizado = df_ventas.copy()
df_optimizado['Cantidad'] = df_optimizado['Cantidad'].astype('int8')
df_optimizado['Producto'] = df_optimizado['Producto'].astype('category')
df_optimizado['Vendedor'] = df_optimizado['Vendedor'].astype('category')

print(f"Uso de memoria optimizado: {df_optimizado.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

### 3. Cadenas de Métodos (Method Chaining)
```python
# Análisis con method chaining
resultado = (df_ventas
             .groupby('Producto')['Ingresos']
             .sum()
             .sort_values(ascending=False)
             .head(3))

print("Top 3 productos por ingresos:")
print(resultado)
```

## Recursos Adicionales

- **Documentación oficial**: https://pandas.pydata.org/docs/
- **Cheat Sheet**: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
- **Tutorials**: https://pandas.pydata.org/docs/getting_started/tutorials.html

## Ejercicios Sugeridos

1. Crea un DataFrame con datos de estudiantes (nombre, edad, calificaciones en diferentes materias)
2. Calcula el promedio de calificaciones por estudiante
3. Encuentra al estudiante con la calificación más alta en cada materia
4. Crea una nueva columna que indique si el estudiante aprobó (promedio >= 6.0)
5. Exporta los resultados a un archivo CSV