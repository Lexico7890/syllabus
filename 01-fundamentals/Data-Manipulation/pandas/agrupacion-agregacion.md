# Agrupación y Agregación en Pandas

## Introducción

La agrupación y agregación son operaciones fundamentales en el análisis de datos que nos permiten resumir, transformar y analizar datos por grupos. En pandas, estas operaciones se realizan principalmente con `groupby()`, que implementa la metodología "split-apply-combine":

1. **Split**: Dividir los datos en grupos basados en criterios
2. **Apply**: Aplicar una función a cada grupo independientemente  
3. **Combine**: Combinar los resultados en una estructura de datos

## Dataset de Ejemplo

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Crear dataset de ventas para ejemplos
np.random.seed(42)
n_records = 2000

# Generar datos de ventas
data = {
    'fecha': pd.date_range('2023-01-01', periods=n_records, freq='H'),
    'vendedor': np.random.choice(['Ana García', 'Luis Martín', 'María López', 
                                 'Carlos Ruiz', 'Elena Torres'], n_records),
    'producto': np.random.choice(['Laptop Dell', 'iPhone 14', 'Samsung TV', 
                                 'Auriculares Sony', 'iPad Air', 'Gaming Chair',
                                 'Mechanical Keyboard', 'Wireless Mouse'], n_records),
    'categoria': np.random.choice(['Electrónicos', 'Tecnología', 'Gaming', 'Accesorios'], n_records),
    'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste', 'Centro'], n_records),
    'cantidad': np.random.randint(1, 10, n_records),
    'precio_unitario': np.random.uniform(50, 2000, n_records).round(2),
    'descuento': np.random.choice([0, 0.05, 0.10, 0.15, 0.20], n_records),
    'cliente_tipo': np.random.choice(['Nuevo', 'Recurrente', 'VIP'], n_records, p=[0.3, 0.5, 0.2]),
    'canal': np.random.choice(['Online', 'Tienda', 'Teléfono'], n_records, p=[0.6, 0.3, 0.1])
}

df = pd.DataFrame(data)
df['ingresos'] = df['cantidad'] * df['precio_unitario'] * (1 - df['descuento'])
df['mes'] = df['fecha'].dt.to_period('M')
df['dia_semana'] = df['fecha'].dt.day_name()
df['trimestre'] = df['fecha'].dt.to_period('Q')

print("Dataset de ventas:")
print(df.head())
print(f"\nForma del dataset: {df.shape}")
print(f"Período: {df['fecha'].min()} a {df['fecha'].max()}")
```

## 1. GroupBy Básico

### Agrupación Simple

```python
# Agrupar por una columna
ventas_por_vendedor = df.groupby('vendedor')
print("Tipo de objeto GroupBy:", type(ventas_por_vendedor))

# Ver los grupos creados
print("\nNúmero de grupos:", ventas_por_vendedor.ngroups)
print("Tamaños de grupos:")
print(ventas_por_vendedor.size())

# Obtener un grupo específico
grupo_ana = ventas_por_vendedor.get_group('Ana García')
print(f"\nVentas de Ana García: {len(grupo_ana)} registros")
print(grupo_ana[['producto', 'ingresos']].head())
```

### Múltiples Columnas de Agrupación

```python
# Agrupar por vendedor y región
ventas_vendedor_region = df.groupby(['vendedor', 'region'])
print("Agrupación por vendedor y región:")
print(ventas_vendedor_region.size().head(10))

# Agrupar por vendedor, región y tipo de cliente
ventas_multiples = df.groupby(['vendedor', 'region', 'cliente_tipo'])
print(f"\nAgrupación múltiple - Grupos: {ventas_multiples.ngroups}")
print("Primeros 10 grupos:")
print(ventas_multiples.size().head(10))
```

## 2. Funciones de Agregación

### Agregaciones Básicas

```python
# Funciones básicas de agregación
resumen_basico = df.groupby('vendedor').agg({
    'ingresos': ['sum', 'mean', 'count', 'std'],
    'cantidad': ['sum', 'mean'],
    'precio_unitario': ['mean', 'min', 'max']
})

print("Resumen básico por vendedor:")
print(resumen_basico)

# Aplanar columnas multi-nivel
resumen_basico.columns = ['_'.join(col).strip() for col in resumen_basico.columns.values]
print("\nColumnas aplanadas:")
print(resumen_basico.head())
```

### Agregaciones con Métodos Directos

```python
# Métodos directos más comunes
print("INGRESOS POR VENDEDOR:")
print("Suma total:", df.groupby('vendedor')['ingresos'].sum().sort_values(ascending=False))
print("\nPromedio:", df.groupby('vendedor')['ingresos'].mean().sort_values(ascending=False))
print("\nConteo:", df.groupby('vendedor')['ingresos'].count())

print("\nVENTAS POR REGIÓN:")
print("Ingresos totales por región:")
print(df.groupby('region')['ingresos'].sum().sort_values(ascending=False))

print("\nProductos más vendidos:")
print(df.groupby('producto')['cantidad'].sum().sort_values(ascending=False))
```

### Funciones de Agregación Personalizadas

```python
# Definir funciones personalizadas
def rango_valores(serie):
    """Calcular el rango (max - min)"""
    return serie.max() - serie.min()

def coeficiente_variacion(serie):
    """Calcular coeficiente de variación"""
    return serie.std() / serie.mean() if serie.mean() != 0 else 0

# Aplicar funciones personalizadas
estadisticas_personalizadas = df.groupby('vendedor')['ingresos'].agg([
    'sum', 'mean', 'std', 'min', 'max', rango_valores, coeficiente_variacion
])

print("Estadísticas personalizadas por vendedor:")
print(estadisticas_personalizadas)
```

## 3. Agregaciones Múltiples y Complejas

### Diferentes Agregaciones por Columna

```python
# Agregaciones específicas para cada columna
agregaciones_complejas = df.groupby('region').agg({
    'ingresos': ['sum', 'mean', 'count'],
    'cantidad': ['sum', 'mean'],
    'descuento': ['mean', 'max'],
    'precio_unitario': ['mean', 'median'],
    'vendedor': 'nunique',  # Número de vendedores únicos
    'producto': 'nunique'   # Número de productos únicos
})

print("Agregaciones complejas por región:")
print(agregaciones_complejas)
```

### Named Aggregations (Agregaciones con Nombres)

```python
# Usar pd.NamedAgg para nombres más claros (pandas 0.25+)
resumen_claro = df.groupby('categoria').agg(
    ingresos_totales=pd.NamedAgg(column='ingresos', aggfunc='sum'),
    ingreso_promedio=pd.NamedAgg(column='ingresos', aggfunc='mean'),
    ventas_totales=pd.NamedAgg(column='cantidad', aggfunc='sum'),
    productos_unicos=pd.NamedAgg(column='producto', aggfunc='nunique'),
    vendedores_activos=pd.NamedAgg(column='vendedor', aggfunc='nunique'),
    descuento_promedio=pd.NamedAgg(column='descuento', aggfunc='mean')
)

print("Resumen con nombres claros por categoría:")
print(resumen_claro)
```

## 4. Transform vs Apply

### Transform - Mantener la Forma Original

```python
# Transform mantiene el número de filas original
df['ingresos_promedio_vendedor'] = df.groupby('vendedor')['ingresos'].transform('mean')
df['ranking_vendedor'] = df.groupby('vendedor')['ingresos'].transform('rank', method='dense')

# Calcular desviación respecto a la media del grupo
df['desviacion_grupo'] = df['ingresos'] - df['ingresos_promedio_vendedor']

print("Primeras filas con transformaciones:")
print(df[['vendedor', 'ingresos', 'ingresos_promedio_vendedor', 'desviacion_grupo']].head(10))

# Estadísticas de normalización
df['ingresos_normalizados'] = df.groupby('vendedor')['ingresos'].transform(
    lambda x: (x - x.mean()) / x.std()
)

print("\nIngresos normalizados por vendedor (z-score):")
print(df[['vendedor', 'ingresos', 'ingresos_normalizados']].head(10))
```

### Apply - Flexibilidad Máxima

```python
# Apply puede cambiar la forma y estructura
def analisis_vendedor(grupo):
    """Función personalizada para análisis por vendedor"""
    return pd.Series({
        'ventas_totales': grupo['ingresos'].sum(),
        'numero_transacciones': len(grupo),
        'ticket_promedio': grupo['ingresos'].mean(),
        'mejor_mes': grupo.groupby('mes')['ingresos'].sum().idxmax(),
        'producto_estrella': grupo.groupby('producto')['ingresos'].sum().idxmax(),
        'region_principal': grupo['region'].mode().iloc[0],
        'consistencia': grupo['ingresos'].std() / grupo['ingresos'].mean()
    })

resumen_vendedores = df.groupby('vendedor').apply(analisis_vendedor)
print("Análisis completo por vendedor:")
print(resumen_vendedores)
```

## 5. Agrupación por Tiempo

### Análisis Temporal

```python
# Agregar columnas temporales adicionales
df['año'] = df['fecha'].dt.year
df['mes_nombre'] = df['fecha'].dt.month_name()
df['semana'] = df['fecha'].dt.isocalendar().week

# Ventas por período
ventas_mensuales = df.groupby('mes')['ingresos'].sum()
print("Ventas mensuales:")
print(ventas_mensuales)

# Tendencia por día de la semana
ventas_dia_semana = df.groupby('dia_semana')['ingresos'].agg(['sum', 'mean', 'count'])
# Reordenar días de la semana
orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ventas_dia_semana = ventas_dia_semana.reindex(orden_dias)
print("\nVentas por día de la semana:")
print(ventas_dia_semana)

# Análisis por trimestre y vendedor
ventas_trimestre_vendedor = df.groupby(['trimestre', 'vendedor'])['ingresos'].sum().unstack()
print("\nVentas por trimestre y vendedor:")
print(ventas_trimestre_vendedor)
```

### Resample para Series Temporales

```python
# Configurar fecha como índice para resample
df_temporal = df.set_index('fecha')

# Ventas diarias
ventas_diarias = df_temporal['ingresos'].resample('D').sum()
print("Ventas diarias (primeros 10 días):")
print(ventas_diarias.head(10))

# Ventas semanales con múltiples métricas
ventas_semanales = df_temporal.resample('W').agg({
    'ingresos': ['sum', 'mean'],
    'cantidad': 'sum',
    'vendedor': 'nunique'
})
print("\nVentas semanales:")
print(ventas_semanales.head())

# Media móvil
ventas_diarias_ma = ventas_diarias.rolling(window=7).mean()
print("\nMedia móvil 7 días (últimos 10 valores):")
print(ventas_diarias_ma.tail(10))
```

## 6. Filtros en GroupBy

### Having (Filtrar Grupos)

```python
# Filtrar grupos después de la agregación
# Vendedores con más de 100 ventas
vendedores_activos = df.groupby('vendedor').filter(lambda x: len(x) > 100)
print(f"Vendedores con más de 100 ventas: {vendedores_activos['vendedor'].nunique()}")

# Productos con ingresos totales > 50000
productos_exitosos = df.groupby('producto').filter(lambda x: x['ingresos'].sum() > 50000)
print(f"Productos exitosos (ingresos > 50000): {productos_exitosos['producto'].nunique()}")

# Regiones con ventas consistentes (desviación estándar baja)
regiones_consistentes = df.groupby('region').filter(
    lambda x: x['ingresos'].std() / x['ingresos'].mean() < 0.8
)
print(f"Regiones con ventas consistentes: {regiones_consistentes['region'].nunique()}")
```

### Combinación de GroupBy con Query

```python
# Filtrar antes y después de agrupar
# Primero filtrar por clientes VIP, luego agrupar
ventas_vip = df.query('cliente_tipo == "VIP"').groupby('vendedor')['ingresos'].sum()
print("Ventas a clientes VIP por vendedor:")
print(ventas_vip.sort_values(ascending=False))

# Productos con alta rotación en ventas online
alta_rotacion_online = (df.query('canal == "Online"')
                       .groupby('producto')['cantidad']
                       .sum()
                       .sort_values(ascending=False))
print("\nProductos con alta rotación en canal online:")
print(alta_rotacion_online.head())
```

## 7. GroupBy Avanzado

### Agrupación por Múltiples Niveles

```python
# Índices jerárquicos con groupby
ventas_jerarquico = df.groupby(['region', 'vendedor', 'categoria'])['ingresos'].sum()
print("Ventas con índice jerárquico:")
print(ventas_jerarquico.head(15))

# Trabajar con índices multi-nivel
print("\nVentas en la región Norte:")
print(ventas_jerarquico.loc['Norte'])

# Agregar totales por nivel
ventas_con_totales = ventas_jerarquico.unstack(level='categoria', fill_value=0)
ventas_con_totales['Total'] = ventas_con_totales.sum(axis=1)
print("\nVentas por región y vendedor con totales:")
print(ventas_con_totales.head())
```

### Pivot Tables

```python
# Crear pivot table - similar a Excel
pivot_ventas = pd.pivot_table(df, 
                             values='ingresos',
                             index='vendedor',
                             columns='region',
                             aggfunc='sum',
                             fill_value=0,
                             margins=True)
print("Pivot table - Ventas por vendedor y región:")
print(pivot_ventas)

# Pivot table con múltiples métricas
pivot_completo = pd.pivot_table(df,
                               values=['ingresos', 'cantidad'],
                               index='producto',
                               columns='cliente_tipo',
                               aggfunc={'ingresos': 'sum', 'cantidad': 'mean'},
                               fill_value=0)
print("\nPivot table con múltiples métricas:")
print(pivot_completo.head())
```

### Crosstab para Análisis de Frecuencias

```python
# Análisis de frecuencias entre categorías
crosstab_canal_cliente = pd.crosstab(df['canal'], df['cliente_tipo'], margins=True)
print("Crosstab: Canal vs Tipo de Cliente")
print(crosstab_canal_cliente)

# Crosstab con porcentajes
crosstab_pct = pd.crosstab(df['canal'], df['cliente_tipo'], normalize='index') * 100
print("\nCrosstab en porcentajes por fila:")
print(crosstab_pct.round(1))

# Crosstab con valores agregados
crosstab_ingresos = pd.crosstab(df['region'], df['categoria'], 
                               values=df['ingresos'], aggfunc='sum')
print("\nCrosstab: Ingresos por región y categoría:")
print(crosstab_ingresos.round(2))
```

## 8. Análisis Estadístico Avanzado

### Percentiles y Distribuciones

```python
# Calcular percentiles por grupo
percentiles_vendedor = df.groupby('vendedor')['ingresos'].quantile([0.25, 0.5, 0.75, 0.9])
print("Percentiles de ingresos por vendedor:")
print(percentiles_vendedor.unstack())

# Descripción estadística completa por grupo
estadisticas_completas = df.groupby('categoria')[['ingresos', 'cantidad', 'precio_unitario']].describe()
print("\nEstadísticas completas por categoría:")
print(estadisticas_completas)

# Análisis de variabilidad
def analisis_variabilidad(grupo):
    return pd.Series({
        'media': grupo.mean(),
        'mediana': grupo.median(),
        'desv_std': grupo.std(),
        'coef_var': grupo.std() / grupo.mean() if grupo.mean() != 0 else 0,
        'asimetria': grupo.skew(),
        'curtosis': grupo.kurtosis()
    })

variabilidad = df.groupby('region')['ingresos'].apply(analisis_variabilidad)
print("\nAnálisis de variabilidad por región:")
print(variabilidad)
```

### Detección de Outliers por Grupo

```python
def detectar_outliers_iqr(grupo):
    """Detectar outliers usando método IQR por grupo"""
    Q1 = grupo.quantile(0.25)
    Q3 = grupo.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    return pd.Series({
        'outliers_bajos': (grupo < limite_inferior).sum(),
        'outliers_altos': (grupo > limite_superior).sum(),
        'total_outliers': ((grupo < limite_inferior) | (grupo > limite_superior)).sum(),
        'porcentaje_outliers': ((grupo < limite_inferior) | (grupo > limite_superior)).mean() * 100
    })

outliers_por_vendedor = df.groupby('vendedor')['ingresos'].apply(detectar_outliers_iqr)
print("Análisis de outliers por vendedor:")
print(outliers_por_vendedor)
```

## 9. Comparaciones y Rankings

### Rankings Dentro de Grupos

```python
# Ranking de productos por vendedor
df['ranking_producto_vendedor'] = df.groupby('vendedor')['ingresos'].rank(method='dense', ascending=False)

# Top 3 productos por vendedor
top_productos_vendedor = (df.groupby('vendedor')
                         .apply(lambda x: x.nlargest(3, 'ingresos')[['producto', 'ingresos']])
                         .reset_index(level=1, drop=True))
print("Top 3 productos por vendedor:")
print(top_productos_vendedor)

# Contribución porcentual al total del grupo
df['contrib_pct_vendedor'] = df.groupby('vendedor')['ingresos'].transform(
    lambda x: x / x.sum() * 100
)

print("\nContribución porcentual por transacción:")
print(df[['vendedor', 'producto', 'ingresos', 'contrib_pct_vendedor']].head())
```

### Comparaciones Temporales

```python
# Crecimiento mes a mes
ingresos_mensuales = df.groupby('mes')['ingresos'].sum()
crecimiento_mensual = ingresos_mensuales.pct_change() * 100

print("Crecimiento mensual (%):")
print(crecimiento_mensual.dropna())

# Comparación año anterior (simulado)
# En un caso real, tendrías datos de múltiples años
comparacion_trimestral = df.groupby(['trimestre', 'vendedor'])['ingresos'].sum()
print("\nVentas por trimestre y vendedor:")
print(comparacion_trimestral.unstack().fillna(0))
```

## 10. Casos de Uso Prácticos

### Dashboard de Ventas

```python
def crear_dashboard_ventas(df):
    """Crear dashboard completo de ventas"""
    
    print("="*60)
    print("           DASHBOARD DE VENTAS")
    print("="*60)
    
    # KPIs principales
    ingresos_totales = df['ingresos'].sum()
    ventas_totales = df['cantidad'].sum()
    transacciones = len(df)
    ticket_promedio = ingresos_totales / transacciones
    
    print(f"\n📊 KPIs PRINCIPALES:")
    print(f"   💰 Ingresos Totales: ${ingresos_totales:,.2f}")
    print(f"   📦 Unidades Vendidas: {ventas_totales:,}")
    print(f"   🧾 Transacciones: {transacciones:,}")
    print(f"   🎫 Ticket Promedio: ${ticket_promedio:.2f}")
    
    # Top performers
    print(f"\n🏆 TOP PERFORMERS:")
    top_vendedores = df.groupby('vendedor')['ingresos'].sum().sort_values(ascending=False)
    print("   Vendedores:")
    for i, (vendedor, ingresos) in enumerate(top_vendedores.head(3).items(), 1):
        print(f"   {i}. {vendedor}: ${ingresos:,.2f}")
    
    top_productos = df.groupby('producto')['ingresos'].sum().sort_values(ascending=False)
    print("   Productos:")
    for i, (producto, ingresos) in enumerate(top_productos.head(3).items(), 1):
        print(f"   {i}. {producto}: ${ingresos:,.2f}")
    
    # Análisis por segmentos
    print(f"\n📈 ANÁLISIS POR SEGMENTOS:")
    
    # Por región
    ingresos_region = df.groupby('region')['ingresos'].sum().sort_values(ascending=False)
    print("   Por región:")
    for region, ingresos in ingresos_region.items():
        pct = (ingresos / ingresos_totales) * 100
        print(f"   • {region}: ${ingresos:,.2f} ({pct:.1f}%)")
    
    # Por canal
    ingresos_canal = df.groupby('canal')['ingresos'].sum().sort_values(ascending=False)
    print("   Por canal:")
    for canal, ingresos in ingresos_canal.items():
        pct = (ingresos / ingresos_totales) * 100
        print(f"   • {canal}: ${ingresos:,.2f} ({pct:.1f}%)")
    
    # Por tipo de cliente
    ingresos_cliente = df.groupby('cliente_tipo')['ingresos'].sum().sort_values(ascending=False)
    print("   Por tipo de cliente:")
    for tipo, ingresos in ingresos_cliente.items():
        pct = (ingresos / ingresos_totales) * 100
        ticket_tipo = df[df['cliente_tipo'] == tipo]['ingresos'].mean()
        print(f"   • {tipo}: ${ingresos:,.2f} ({pct:.1f}%) - Ticket: ${ticket_tipo:.2f}")

crear_dashboard_ventas(df)
```

### Análisis de Cohorts Simplificado

```python
def analisis_cohorts_mensual(df):
    """Análisis básico de cohorts por mes"""
    
    # Primer mes de cada vendedor
    primer_mes_vendedor = df.groupby('vendedor')['mes'].min().reset_index()
    primer_mes_vendedor.columns = ['vendedor', 'mes_inicio']
    
    # Merge con datos originales
    df_cohort = df.merge(primer_mes_vendedor, on='vendedor')
    
    # Calcular período relativo
    df_cohort['periodo'] = (df_cohort['mes'] - df_cohort['mes_inicio']).apply(lambda x: x.n)
    
    # Tabla de cohorts
    cohort_table = df_cohort.groupby(['mes_inicio', 'periodo'])['vendedor'].nunique().unstack(fill_value=0)
    
    print("Tabla de Cohorts (Vendedores activos por período):")
    print(cohort_table)
    
    # Tasas de retención
    cohort_sizes = cohort_table.iloc[:, 0]
    retention_table = cohort_table.divide(cohort_sizes, axis=0) * 100
    
    print("\nTasas de retención (%):")
    print(retention_table.round(1))

# Solo ejecutar si hay suficientes datos temporales
if df['mes'].nunique() > 1:
    analisis_cohorts_mensual(df)
```

### Sistema de Alertas de Rendimiento

```python
def sistema_alertas_rendimiento(df):
    """Sistema de alertas basado en análisis de grupos"""
    
    alertas = []
    
    # Alertas de vendedores
    ventas_vendedor = df.groupby('vendedor')['ingresos'].agg(['sum', 'count', 'mean'])
    media_ingresos = ventas_vendedor['sum'].mean()
    
    # Vendedores por debajo del promedio
    bajo_rendimiento = ventas_vendedor[ventas_vendedor['sum'] < media_ingresos * 0.7]
    if len(bajo_rendimiento) > 0:
        alertas.append({
            'tipo': 'RENDIMIENTO_VENDEDOR',
            'mensaje': f'{len(bajo_rendimiento)} vendedores con rendimiento bajo',
            'detalle': bajo_rendimiento.index.tolist()
        })
    
    # Alertas de productos
    ventas_producto = df.groupby('producto')['cantidad'].sum()
    productos_baja_rotacion = ventas_producto[ventas_producto < ventas_producto.quantile(0.25)]
    
    if len(productos_baja_rotacion) > 0:
        alertas.append({
            'tipo': 'BAJA_ROTACION',
            'mensaje': f'{len(productos_baja_rotacion)} productos con baja rotación',
            'detalle': productos_baja_rotacion.index.tolist()
        })
    
    # Alertas regionales
    ingresos_region = df.groupby('region')['ingresos'].sum()
    region_baja = ingresos_region[ingresos_region < ingresos_region.mean() * 0.8]
    
    if len(region_baja) > 0:
        alertas.append({
            'tipo': 'REGION_BAJO_RENDIMIENTO',
            'mensaje': f'{len(region_baja)} regiones con bajo rendimiento',
            'detalle': region_baja.index.tolist()
        })
    
    # Mostrar alertas
    print("🚨 SISTEMA DE ALERTAS DE RENDIMIENTO:")
    if not alertas:
        print("✅ No se detectaron problemas de rendimiento")
    else:
        for alerta in alertas:
            print(f"\n⚠️  {alerta['tipo']}")
            print(f"    {alerta['mensaje']}")
            print(f"    Afectados: {', '.join(alerta['detalle'][:3])}{'...' if len(alerta['detalle']) > 3 else ''}")

sistema_alertas_rendimiento(df)
```

## 11. Optimización y Performance

### Técnicas de Optimización

```python
import time

def comparar_metodos_agregacion(df):
    """Comparar performance de diferentes métodos"""
    
    # Método 1: groupby con agg
    start = time.time()
    resultado1 = df.groupby('vendedor')['ingresos'].agg(['sum', 'mean', 'count'])
    tiempo1 = time.time() - start
    
    # Método 2: múltiples operaciones separadas
    start = time.time()
    suma = df.groupby('vendedor')['ingresos'].sum()
    promedio = df.groupby('vendedor')['ingresos'].mean()
    conteo = df.groupby('vendedor')['ingresos'].count()
    resultado2 = pd.concat([suma, promedio, conteo], axis=1)
    tiempo2 = time.time() - start
    
    # Método 3: pivot_table
    start = time.time()
    resultado3 = pd.pivot_table(df, values='ingresos', index='vendedor', 
                               aggfunc=['sum', 'mean', 'count'])
    tiempo3 = time.time() - start
    
    print(f"Método 1 (groupby.agg): {tiempo1:.4f}s")
    print(f"Método 2 (operaciones separadas): {tiempo2:.4f}s")
    print(f"Método 3 (pivot_table): {tiempo3:.4f}s")
    
    return resultado1

resultado = comparar_metodos_agregacion(df)
print("\nResultado de agregación:")
print(resultado)
```

### Manejo de Memoria en Grupos Grandes

```python
# Para datasets muy grandes, usar chunking
def procesar_por_chunks(df, chunk_size=1000):
    """Procesar grandes datasets por chunks"""
    
    resultados = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        resultado_chunk = chunk.groupby('vendedor')['ingresos'].sum()
        resultados.append(resultado_chunk)
    
    # Combinar resultados
    resultado_final = pd.concat(resultados).groupby(level=0).sum()
    return resultado_final

# Ejemplo con dataset completo
resultado_chunks = procesar_por_chunks(df)
print("Procesamiento por chunks:")
print(resultado_chunks.head())
```

## 12. Visualización de Resultados Agrupados

### Gráficos Básicos

```python
# Preparar datos para visualización
import matplotlib.pyplot as plt

# Gráfico de barras - Ventas por vendedor
ventas_vendedor = df.groupby('vendedor')['ingresos'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
ventas_vendedor.plot(kind='bar', color='skyblue')
plt.title('Ingresos Totales por Vendedor')
plt.xlabel('Vendedor')
plt.ylabel('Ingresos ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Gráfico de líneas - Tendencia temporal
ventas_mensuales = df.groupby('mes')['ingresos'].sum()
plt.figure(figsize=(12, 6))
ventas_mensuales.plot(kind='line', marker='o', linewidth=2, markersize=8)
plt.title('Tendencia de Ventas Mensuales')
plt.xlabel('Mes')
plt.ylabel('Ingresos ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Mejores Prácticas

### 1. Performance
- Usar `agg()` para múltiples agregaciones en una sola operación
- Aplicar filtros antes de agrupar cuando sea posible
- Considerar `transform()` vs `apply()` según el caso de uso
- Para datasets grandes, considerar usar Dask o procesamiento por chunks

### 2. Legibilidad
- Usar nombres descriptivos con `pd.NamedAgg`
- Documentar funciones personalizadas de agregación
- Aplanar columnas multi-nivel cuando sea apropiado
- Usar `reset_index()` cuando necesites el índice como columna

### 3. Análisis
- Combinar múltiples niveles de agrupación para insights más profundos
- Usar `describe()` para obtener estadísticas completas rápidamente
- Considerar percentiles además de media y mediana
- Validar resultados con totales y subtotales

### 4. Debugging
```python
def debug_groupby(df, columna_grupo, columna_valor):
    """Función helper para debugging de operaciones groupby"""
    grupo = df.groupby(columna_grupo)[columna_valor]
    
    print(f"Análisis de groupby: {columna_grupo} -> {columna_valor}")
    print(f"Número de grupos: {grupo.ngroups}")
    print(f"Tamaños de grupos:")
    print(grupo.size().describe())
    print(f"Grupos vacíos: {(grupo.size() == 0).sum()}")
    
    return grupo

# Ejemplo de uso
debug_info = debug_groupby(df, 'vendedor', 'ingresos')
```

## Recursos y Referencias

- **Documentación oficial de GroupBy**: https://pandas.pydata.org/docs/user_guide/groupby.html
- **Split-Apply-Combine**: https://pandas.pydata.org/docs/user_guide/groupby.html#splitting-an-object-into-groups
- **Aggregation functions**: https://pandas.pydata.org/docs/reference/groupby.html
- **Time series grouping**: https://pandas.pydata.org/docs/user_guide/timeseries.html

## Ejercicios Sugeridos

1. **Análisis de Productividad**: Agrupa por vendedor y mes, calcula métricas de productividad
2. **Segmentación de Clientes**: Usa múltiples dimensiones para segmentar tipos de cliente
3. **Análisis de Estacionalidad**: Agrupa por períodos temporales y busca patrones
4. **Detección de Anomalías**: Usa estadísticas grupales para identificar valores atípicos
5. **Dashboard Ejecutivo**: Combina múltiples agregaciones para crear un resumen ejecutivo