# Filtrado y Selección de Datos en Pandas

## Introducción

El filtrado y selección de datos son operaciones fundamentales en el análisis de datos. Pandas ofrece múltiples formas de seleccionar subconjuntos específicos de datos basados en criterios diversos, desde condiciones simples hasta consultas complejas.

## Tipos de Selección en Pandas

### 1. Selección por Etiquetas (.loc)
### 2. Selección por Posición (.iloc)
### 3. Selección por Condiciones Booleanas
### 4. Selección con Query
### 5. Selección Avanzada

## Dataset de Ejemplo

```python
import pandas as pd
import numpy as np

# Crear un dataset de ejemplo - Tienda Online
np.random.seed(42)
n_records = 1000

data = {
    'producto_id': range(1, n_records + 1),
    'nombre_producto': np.random.choice(['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Tablet', 
                                        'Smartphone', 'Auriculares', 'Cámara', 'Impresora'], n_records),
    'categoria': np.random.choice(['Electrónicos', 'Accesorios', 'Gaming', 'Oficina'], n_records),
    'precio': np.random.uniform(10, 2000, n_records).round(2),
    'stock': np.random.randint(0, 100, n_records),
    'calificacion': np.random.uniform(1, 5, n_records).round(1),
    'vendedor': np.random.choice(['TechStore', 'ElectroMax', 'GamerZone', 'OfficeSupply'], n_records),
    'fecha_ingreso': pd.date_range('2023-01-01', periods=n_records, freq='H'),
    'activo': np.random.choice([True, False], n_records, p=[0.8, 0.2]),
    'descuento': np.random.uniform(0, 0.5, n_records).round(2)
}

df = pd.DataFrame(data)
df['precio_final'] = df['precio'] * (1 - df['descuento'])

print("Dataset de ejemplo:")
print(df.head())
print(f"\nForma del dataset: {df.shape}")
```

## 1. Selección por Etiquetas (.loc)

### Selección Básica con .loc

```python
# Seleccionar filas específicas por índice
print("Filas 5 a 10:")
print(df.loc[5:10])

# Seleccionar filas y columnas específicas
print("\nProductos y precios (filas 0-4):")
print(df.loc[0:4, ['nombre_producto', 'precio']])

# Seleccionar todas las filas, columnas específicas
print("\nSolo columnas de producto y precio:")
print(df.loc[:, ['nombre_producto', 'precio']].head())

# Usar slice de columnas
print("\nColumnas desde 'producto_id' hasta 'precio':")
print(df.loc[:5, 'producto_id':'precio'])
```

### Selección Condicional con .loc

```python
# Productos con precio mayor a 1000
productos_caros = df.loc[df['precio'] > 1000, ['nombre_producto', 'precio', 'categoria']]
print("Productos con precio > 1000:")
print(productos_caros.head())

# Múltiples condiciones
productos_premium = df.loc[
    (df['precio'] > 500) & (df['calificacion'] > 4.0) & (df['stock'] > 10),
    ['nombre_producto', 'precio', 'calificacion', 'stock']
]
print(f"\nProductos premium (precio>500, calificación>4.0, stock>10):")
print(productos_premium.head())

# Usar isin() para múltiples valores
laptops_tablets = df.loc[
    df['nombre_producto'].isin(['Laptop', 'Tablet']),
    ['nombre_producto', 'precio', 'stock']
]
print(f"\nSolo Laptops y Tablets:")
print(laptops_tablets.head())
```

## 2. Selección por Posición (.iloc)

### Selección Básica con .iloc

```python
# Primeras 5 filas, primeras 3 columnas
print("Primeras 5 filas, 3 columnas:")
print(df.iloc[:5, :3])

# Filas específicas, columnas específicas
print("\nFilas 10, 20, 30 - columnas 1, 3, 5:")
print(df.iloc[[10, 20, 30], [1, 3, 5]])

# Última fila, todas las columnas
print("\nÚltima fila:")
print(df.iloc[-1:])

# Cada décima fila
print("\nCada décima fila (primeras 5 columnas):")
print(df.iloc[::10, :5])
```

### Selección Aleatoria

```python
# Muestra aleatoria de filas
muestra_aleatoria = df.iloc[np.random.choice(df.index, 5, replace=False)]
print("Muestra aleatoria de 5 productos:")
print(muestra_aleatoria[['nombre_producto', 'precio', 'categoria']])
```

## 3. Filtrado con Condiciones Booleanas

### Condiciones Simples

```python
# Productos en stock
en_stock = df[df['stock'] > 0]
print(f"Productos en stock: {len(en_stock)} de {len(df)}")

# Productos con descuento
con_descuento = df[df['descuento'] > 0]
print(f"Productos con descuento: {len(con_descuento)}")

# Productos activos
productos_activos = df[df['activo'] == True]
print(f"Productos activos: {len(productos_activos)}")
```

### Condiciones Múltiples

```python
# AND lógico con &
gaming_premium = df[
    (df['categoria'] == 'Gaming') & 
    (df['precio'] > 200) & 
    (df['calificacion'] >= 4.0)
]
print(f"Productos Gaming premium: {len(gaming_premium)}")
print(gaming_premium[['nombre_producto', 'precio', 'calificacion']].head())

# OR lógico con |
electronicos_o_gaming = df[
    (df['categoria'] == 'Electrónicos') | (df['categoria'] == 'Gaming')
]
print(f"\nProductos Electrónicos o Gaming: {len(electronicos_o_gaming)}")

# NOT lógico con ~
no_accesorios = df[~(df['categoria'] == 'Accesorios')]
print(f"Productos que NO son Accesorios: {len(no_accesorios)}")
```

### Filtros con Strings

```python
# Productos que contienen "Laptop" en el nombre
laptops = df[df['nombre_producto'].str.contains('Laptop', case=False)]
print("Productos que contienen 'Laptop':")
print(laptops[['nombre_producto', 'precio']].head())

# Productos que empiezan con 'M'
productos_m = df[df['nombre_producto'].str.startswith('M')]
print(f"\nProductos que empiezan con 'M': {len(productos_m)}")

# Productos con nombres de cierta longitud
nombres_largos = df[df['nombre_producto'].str.len() > 7]
print(f"Productos con nombres largos (>7 chars): {len(nombres_largos)}")
```

### Filtros con Fechas

```python
# Productos ingresados en enero 2023
enero_2023 = df[
    (df['fecha_ingreso'] >= '2023-01-01') & 
    (df['fecha_ingreso'] < '2023-02-01')
]
print(f"Productos ingresados en enero 2023: {len(enero_2023)}")

# Productos ingresados en los últimos 7 días (del dataset)
ultimos_dias = df[df['fecha_ingreso'] >= df['fecha_ingreso'].max() - pd.Timedelta(days=7)]
print(f"Productos de los últimos 7 días: {len(ultimos_dias)}")
```

## 4. Método Query

### Query Básico

```python
# Sintaxis SQL-like más legible
productos_caros_query = df.query('precio > 1000')
print(f"Productos caros (query): {len(productos_caros_query)}")

# Múltiples condiciones
gaming_query = df.query('categoria == "Gaming" and precio > 100 and calificacion >= 4')
print(f"Gaming products (query): {len(gaming_query)}")

# Usar variables externas
precio_minimo = 500
productos_query = df.query('precio > @precio_minimo')
print(f"Productos > {precio_minimo}: {len(productos_query)}")
```

### Query Avanzado

```python
# Operadores in y not in
categorias_interes = ['Gaming', 'Electrónicos']
productos_interes = df.query('categoria in @categorias_interes')
print(f"Productos en categorías de interés: {len(productos_interes)}")

# Expresiones complejas
descuento_significativo = df.query(
    'descuento > 0.3 and precio_final < 200 and stock > 5'
)
print(f"Productos con descuento significativo: {len(descuento_significativo)}")
print(descuento_significativo[['nombre_producto', 'precio', 'descuento', 'precio_final']].head())
```

## 5. Selección Avanzada

### Top N y Bottom N

```python
# Top 10 productos más caros
top_caros = df.nlargest(10, 'precio')[['nombre_producto', 'precio', 'categoria']]
print("Top 10 productos más caros:")
print(top_caros)

# Top 5 mejor calificados con stock
mejor_calificados = df[df['stock'] > 0].nlargest(5, 'calificacion')[
    ['nombre_producto', 'calificacion', 'precio', 'stock']
]
print("\nTop 5 mejor calificados (con stock):")
print(mejor_calificados)

# Productos más baratos por categoría
mas_baratos_por_categoria = df.loc[df.groupby('categoria')['precio'].idxmin()]
print("\nProducto más barato por categoría:")
print(mas_baratos_por_categoria[['categoria', 'nombre_producto', 'precio']])
```

### Muestreo

```python
# Muestra aleatoria estratificada
muestra_estratificada = df.groupby('categoria', group_keys=False).apply(
    lambda x: x.sample(min(5, len(x)))
)
print(f"Muestra estratificada por categoría:")
print(muestra_estratificada.groupby('categoria').size())

# Muestra por fracción
muestra_10_porciento = df.sample(frac=0.1)
print(f"Muestra del 10%: {len(muestra_10_porciento)} registros")
```

## 6. Filtros con Múltiples Criterios

### Sistema de Recomendación Simple

```python
def recomendar_productos(df, precio_max=1000, calificacion_min=4.0, 
                        categoria=None, con_stock=True):
    """
    Sistema simple de recomendación de productos
    """
    filtros = df.copy()
    
    # Aplicar filtros
    filtros = filtros[filtros['precio'] <= precio_max]
    filtros = filtros[filtros['calificacion'] >= calificacion_min]
    
    if categoria:
        filtros = filtros[filtros['categoria'] == categoria]
    
    if con_stock:
        filtros = filtros[filtros['stock'] > 0]
    
    # Ordenar por calificación y precio
    recomendaciones = filtros.sort_values(['calificacion', 'precio'], 
                                        ascending=[False, True])
    
    return recomendaciones[['nombre_producto', 'categoria', 'precio', 
                          'calificacion', 'stock']].head(10)

# Usar el sistema de recomendación
print("Recomendaciones generales:")
print(recomendar_productos(df))

print("\nRecomendaciones Gaming:")
print(recomendar_productos(df, precio_max=800, categoria='Gaming'))
```

### Análisis de Segmentos

```python
# Definir segmentos de precio
def categorizar_precio(precio):
    if precio < 100:
        return 'Económico'
    elif precio < 500:
        return 'Medio'
    elif precio < 1000:
        return 'Premium'
    else:
        return 'Lujo'

df['segmento_precio'] = df['precio'].apply(categorizar_precio)

# Análisis por segmento
print("Distribución por segmento de precio:")
print(df['segmento_precio'].value_counts())

# Productos premium con alta calificación
premium_quality = df[
    (df['segmento_precio'] == 'Premium') & 
    (df['calificacion'] >= 4.5) &
    (df['activo'] == True)
]
print(f"\nProductos Premium de alta calidad: {len(premium_quality)}")
print(premium_quality[['nombre_producto', 'precio', 'calificacion']].head())
```

## 7. Filtros Dinámicos y Interactivos

### Clase para Filtros Dinámicos

```python
class FiltroProductos:
    def __init__(self, df):
        self.df_original = df.copy()
        self.df_filtrado = df.copy()
        self.filtros_aplicados = []
    
    def filtrar_precio(self, min_precio=None, max_precio=None):
        """Filtrar por rango de precios"""
        if min_precio is not None:
            self.df_filtrado = self.df_filtrado[self.df_filtrado['precio'] >= min_precio]
            self.filtros_aplicados.append(f'precio >= {min_precio}')
        
        if max_precio is not None:
            self.df_filtrado = self.df_filtrado[self.df_filtrado['precio'] <= max_precio]
            self.filtros_aplicados.append(f'precio <= {max_precio}')
        
        return self
    
    def filtrar_categoria(self, categorias):
        """Filtrar por categorías específicas"""
        if isinstance(categorias, str):
            categorias = [categorias]
        
        self.df_filtrado = self.df_filtrado[self.df_filtrado['categoria'].isin(categorias)]
        self.filtros_aplicados.append(f'categoria in {categorias}')
        return self
    
    def filtrar_calificacion(self, min_calificacion):
        """Filtrar por calificación mínima"""
        self.df_filtrado = self.df_filtrado[self.df_filtrado['calificacion'] >= min_calificacion]
        self.filtros_aplicados.append(f'calificación >= {min_calificacion}')
        return self
    
    def solo_con_stock(self):
        """Mostrar solo productos con stock"""
        self.df_filtrado = self.df_filtrado[self.df_filtrado['stock'] > 0]
        self.filtros_aplicados.append('con stock')
        return self
    
    def reset_filtros(self):
        """Resetear todos los filtros"""
        self.df_filtrado = self.df_original.copy()
        self.filtros_aplicados = []
        return self
    
    def obtener_resultados(self):
        """Obtener DataFrame filtrado"""
        return self.df_filtrado
    
    def resumen(self):
        """Mostrar resumen de filtros y resultados"""
        print(f"Filtros aplicados: {', '.join(self.filtros_aplicados) if self.filtros_aplicados else 'Ninguno'}")
        print(f"Registros encontrados: {len(self.df_filtrado)} de {len(self.df_original)}")
        print(f"Porcentaje: {len(self.df_filtrado)/len(self.df_original)*100:.1f}%")

# Ejemplo de uso de la clase FiltroProductos
filtro = FiltroProductos(df)

# Encadenar filtros
productos_gaming_premium = (filtro
                           .filtrar_categoria('Gaming')
                           .filtrar_precio(min_precio=200)
                           .filtrar_calificacion(4.0)
                           .solo_con_stock())

productos_gaming_premium.resumen()
print("\nPrimeros 10 resultados:")
print(productos_gaming_premium.obtener_resultados()[['nombre_producto', 'precio', 'calificacion', 'stock']].head(10))
```

## 8. Filtros con Expresiones Regulares

### Búsquedas Avanzadas de Texto

```python
# Productos que contienen patrones específicos
patron_tech = df[df['nombre_producto'].str.contains(r'(Laptop|Tablet|Phone)', case=False, regex=True)]
print(f"Productos tech (Laptop|Tablet|Phone): {len(patron_tech)}")

# Vendedores que empiezan con vocal
vendedores_vocal = df[df['vendedor'].str.contains(r'^[AEIOUaeiou]', regex=True)]
print(f"Productos de vendedores que empiezan con vocal: {len(vendedores_vocal)}")
print(vendedores_vocal['vendedor'].unique())

# Productos con nombres que contienen números
nombres_con_numeros = df[df['nombre_producto'].str.contains(r'\d', regex=True)]
print(f"Productos con números en el nombre: {len(nombres_con_numeros)}")
```

## 9. Filtros con Percentiles y Rangos Estadísticos

### Análisis por Percentiles

```python
# Calcular percentiles de precio
percentiles = df['precio'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])
print("Percentiles de precio:")
print(percentiles)

# Productos en el top 10% por precio
top_10_percent = df[df['precio'] >= df['precio'].quantile(0.9)]
print(f"\nProductos en top 10% por precio: {len(top_10_percent)}")

# Productos entre percentil 25 y 75 (rango intercuartílico)
rango_medio = df[
    (df['precio'] >= df['precio'].quantile(0.25)) & 
    (df['precio'] <= df['precio'].quantile(0.75))
]
print(f"Productos en rango medio (P25-P75): {len(rango_medio)}")

# Outliers por precio (fuera de 1.5 * IQR)
Q1 = df['precio'].quantile(0.25)
Q3 = df['precio'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[
    (df['precio'] < Q1 - 1.5 * IQR) | (df['precio'] > Q3 + 1.5 * IQR)
]
print(f"Outliers por precio: {len(outliers)}")
print(outliers[['nombre_producto', 'precio']].head())
```

## 10. Filtros Temporales Avanzados

### Análisis por Período de Tiempo

```python
# Productos por día de la semana
df['dia_semana'] = df['fecha_ingreso'].dt.day_name()
productos_lunes = df[df['dia_semana'] == 'Monday']
print(f"Productos ingresados los lunes: {len(productos_lunes)}")

# Productos por hora del día
df['hora'] = df['fecha_ingreso'].dt.hour
productos_mañana = df[(df['hora'] >= 6) & (df['hora'] < 12)]
print(f"Productos ingresados en la mañana (6-12h): {len(productos_mañana)}")

# Productos del último mes del dataset
ultimo_mes = df['fecha_ingreso'].max() - pd.DateOffset(months=1)
productos_recientes = df[df['fecha_ingreso'] >= ultimo_mes]
print(f"Productos del último mes: {len(productos_recientes)}")
```

## 11. Combinación de Filtros Complejos

### Análisis de Oportunidades de Negocio

```python
def analizar_oportunidades(df):
    """
    Encontrar productos con oportunidades de negocio
    """
    oportunidades = {}
    
    # 1. Productos con alta demanda (bajo stock, alta calificación)
    alta_demanda = df[
        (df['stock'] < 10) & 
        (df['calificacion'] >= 4.0) & 
        (df['activo'] == True)
    ]
    oportunidades['Reabastecer'] = len(alta_demanda)
    
    # 2. Productos sobrevalorados (alto precio, baja calificación)
    sobrevalorados = df[
        (df['precio'] > df['precio'].quantile(0.75)) & 
        (df['calificacion'] < 3.0)
    ]
    oportunidades['Reducir precio'] = len(sobrevalorados)
    
    # 3. Productos con exceso de inventario (mucho stock, baja calificación)
    exceso_inventario = df[
        (df['stock'] > df['stock'].quantile(0.9)) & 
        (df['calificacion'] < 3.5)
    ]
    oportunidades['Liquidar'] = len(exceso_inventario)
    
    # 4. Productos estrella (alta calificación, precio competitivo, buen stock)
    estrellas = df[
        (df['calificacion'] >= 4.5) & 
        (df['precio'] <= df['precio'].quantile(0.6)) & 
        (df['stock'] > 5)
    ]
    oportunidades['Promocionar'] = len(estrellas)
    
    return oportunidades

oportunidades = analizar_oportunidades(df)
print("Análisis de oportunidades:")
for categoria, cantidad in oportunidades.items():
    print(f"{categoria}: {cantidad} productos")
```

### Segmentación de Clientes por Producto

```python
# Crear segmentos basados en múltiples criterios
def segmentar_productos(df):
    condiciones = [
        (df['precio'] <= 100) & (df['calificacion'] >= 4.0),  # Valor por dinero
        (df['precio'] > 100) & (df['precio'] <= 500) & (df['calificacion'] >= 4.0),  # Gama media
        (df['precio'] > 500) & (df['calificacion'] >= 4.0),  # Premium
        (df['calificacion'] < 3.0),  # Baja calidad
    ]
    
    valores = ['Valor por dinero', 'Gama media', 'Premium', 'Baja calidad']
    
    df['segmento'] = np.select(condiciones, valores, default='Estándar')
    return df

df_segmentado = segmentar_productos(df.copy())
print("Distribución por segmentos:")
print(df_segmentado['segmento'].value_counts())

# Análisis por segmento
for segmento in df_segmentado['segmento'].unique():
    datos_segmento = df_segmentado[df_segmentado['segmento'] == segmento]
    print(f"\n{segmento}:")
    print(f"  Productos: {len(datos_segmento)}")
    print(f"  Precio promedio: ${datos_segmento['precio'].mean():.2f}")
    print(f"  Calificación promedio: {datos_segmento['calificacion'].mean():.1f}")
```

## 12. Técnicas de Filtrado Eficiente

### Optimización de Performance

```python
# Comparar métodos de filtrado
import time

def comparar_metodos_filtrado(df, n_iteraciones=1000):
    """Comparar performance de diferentes métodos de filtrado"""
    
    # Método 1: Boolean indexing
    start = time.time()
    for _ in range(n_iteraciones):
        resultado1 = df[(df['precio'] > 500) & (df['calificacion'] > 4.0)]
    tiempo1 = time.time() - start
    
    # Método 2: Query
    start = time.time()
    for _ in range(n_iteraciones):
        resultado2 = df.query('precio > 500 and calificacion > 4.0')
    tiempo2 = time.time() - start
    
    # Método 3: loc con condición
    start = time.time()
    for _ in range(n_iteraciones):
        resultado3 = df.loc[(df['precio'] > 500) & (df['calificacion'] > 4.0)]
    tiempo3 = time.time() - start
    
    print(f"Boolean indexing: {tiempo1:.4f}s")
    print(f"Query: {tiempo2:.4f}s")  
    print(f"Loc con condición: {tiempo3:.4f}s")
    
    return resultado1

resultado = comparar_metodos_filtrado(df)
print(f"\nResultados encontrados: {len(resultado)}")
```

### Índices para Filtrado Rápido

```python
# Crear índice para búsquedas rápidas por categoría
df_indexed = df.set_index('categoria')

# Filtrado rápido por categoría
gaming_products = df_indexed.loc['Gaming']
print(f"Productos Gaming (con índice): {len(gaming_products)}")

# Múltiples niveles de índice para consultas complejas
df_multi_index = df.set_index(['categoria', 'vendedor'])
gaming_techstore = df_multi_index.loc[('Gaming', 'TechStore')]
print(f"Productos Gaming de TechStore: {len(gaming_techstore)}")
```

## 13. Casos de Uso Prácticos

### Dashboard de Productos

```python
def crear_dashboard_productos(df):
    """Crear un resumen tipo dashboard"""
    
    print("=== DASHBOARD DE PRODUCTOS ===\n")
    
    # Métricas generales
    print("MÉTRICAS GENERALES:")
    print(f"Total productos: {len(df):,}")
    print(f"Productos activos: {len(df[df['activo']])} ({len(df[df['activo']])/len(df)*100:.1f}%)")
    print(f"Productos con stock: {len(df[df['stock'] > 0])} ({len(df[df['stock'] > 0])/len(df)*100:.1f}%)")
    print(f"Valor total inventario: ${df['precio_final'].sum():,.2f}")
    
    print("\nTOP CATEGORÍAS:")
    top_categorias = df['categoria'].value_counts()
    for cat, count in top_categorias.items():
        pct = count/len(df)*100
        precio_promedio = df[df['categoria'] == cat]['precio'].mean()
        print(f"  {cat}: {count} productos ({pct:.1f}%) - Precio promedio: ${precio_promedio:.2f}")
    
    print("\nALERTAS:")
    # Productos sin stock
    sin_stock = df[df['stock'] == 0]
    print(f"  ⚠️  {len(sin_stock)} productos sin stock")
    
    # Productos con baja calificación
    baja_calificacion = df[df['calificacion'] < 3.0]
    print(f"  ⚠️  {len(baja_calificacion)} productos con calificación < 3.0")
    
    # Productos inactivos con stock
    inactivos_con_stock = df[(df['activo'] == False) & (df['stock'] > 0)]
    print(f"  ⚠️  {len(inactivos_con_stock)} productos inactivos con stock")
    
    print("\nOPORTUNidades:")
    # Productos con descuento alto
    descuento_alto = df[df['descuento'] > 0.3]
    print(f"  💰 {len(descuento_alto)} productos con descuento > 30%")
    
    # Productos mejor valorados
    top_rated = df[df['calificacion'] >= 4.5]
    print(f"  ⭐ {len(top_rated)} productos con calificación ≥ 4.5")

crear_dashboard_productos(df)
```

### Sistema de Alertas Automáticas

```python
def generar_alertas(df):
    """Sistema de alertas automáticas"""
    
    alertas = []
    
    # Stock crítico
    stock_critico = df[(df['stock'] <= 5) & (df['activo'] == True)]
    if len(stock_critico) > 0:
        alertas.append({
            'tipo': 'STOCK_CRITICO',
            'mensaje': f'{len(stock_critico)} productos con stock crítico (≤5)',
            'productos': stock_critico['nombre_producto'].tolist()[:5]
        })
    
    # Precios anómalos
    precio_medio = df['precio'].median()
    precios_altos = df[df['precio'] > precio_medio * 3]
    if len(precios_altos) > 0:
        alertas.append({
            'tipo': 'PRECIO_ANOMALO',
            'mensaje': f'{len(precios_altos)} productos con precios anómalamente altos',
            'productos': precios_altos['nombre_producto'].tolist()[:5]
        })
    
    # Productos sin movimiento (baja calificación + alto stock)
    sin_movimiento = df[(df['calificacion'] < 3.0) & (df['stock'] > 20)]
    if len(sin_movimiento) > 0:
        alertas.append({
            'tipo': 'SIN_MOVIMIENTO',
            'mensaje': f'{len(sin_movimiento)} productos probablemente sin movimiento',
            'productos': sin_movimiento['nombre_producto'].tolist()[:5]
        })
    
    return alertas

alertas = generar_alertas(df)
print("SISTEMA DE ALERTAS:")
for alerta in alertas:
    print(f"\n🚨 {alerta['tipo']}")
    print(f"   {alerta['mensaje']}")
    print(f"   Ejemplos: {', '.join(alerta['productos'])}")
```

## Mejores Prácticas

### 1. Rendimiento
- Usar índices para filtros frecuentes
- Evitar loops cuando sea posible
- Usar `query()` para condiciones complejas legibles
- Filtrar datos lo antes posible en el pipeline

### 2. Legibilidad
- Usar nombres descriptivos para filtros
- Documentar condiciones complejas
- Usar paréntesis para clarificar precedencia de operadores
- Preferir múltiples filtros simples a uno muy complejo

### 3. Mantenibilidad
- Crear funciones para filtros reutilizables
- Usar constantes para valores de filtro frecuentes
- Validar datos antes de filtrar
- Registrar criterios de filtrado aplicados

### 4. Debugging
```python
# Función helper para debug de filtros
def debug_filtro(df_original, df_filtrado, nombre_filtro):
    """Debug información de filtros aplicados"""
    print(f"Filtro: {nombre_filtro}")
    print(f"Registros originales: {len(df_original):,}")
    print(f"Registros filtrados: {len(df_filtrado):,}")
    print(f"Registros eliminados: {len(df_original) - len(df_filtrado):,}")
    print(f"Porcentaje retenido: {len(df_filtrado)/len(df_original)*100:.2f}%")
    print("-" * 50)

# Ejemplo de uso
df_temp = df[df['precio'] > 500]
debug_filtro(df, df_temp, "Precio > 500")
```

## Recursos y Referencias

- **Documentación oficial de indexing**: https://pandas.pydata.org/docs/user_guide/indexing.html
- **Boolean indexing**: https://pandas.pydata.org/docs/user_guide/indexing.html#boolean-indexing
- **Query method**: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html
- **Performance tips**: https://pandas.pydata.org/docs/user_guide/enhancingperf.html

## Ejercicios Sugeridos

1. **Análisis de Ventas**: Filtra productos por diferentes criterios y analiza patrones
2. **Sistema de Recomendación**: Crea filtros para recomendar productos según preferencias
3. **Detección de Anomalías**: Identifica productos con características inusuales
4. **Segmentación de Mercado**: Agrupa productos en segmentos usando múltiples filtros
5. **Dashboard Interactivo**: Combina diferentes filtros para crear vistas personalizadas