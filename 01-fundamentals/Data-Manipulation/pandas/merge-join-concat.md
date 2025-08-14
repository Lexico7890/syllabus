# Merge, Join y Concat en Pandas

## Introducción

La combinación de datasets es una operación fundamental en el análisis de datos. Pandas ofrece varias funciones para unir DataFrames:

- **`concat()`**: Concatenar DataFrames a lo largo de un eje
- **`merge()`**: Combinar DataFrames basado en columnas o índices comunes (similar a SQL JOIN)
- **`.join()`**: Combinar DataFrames basado en índices (método del DataFrame)
- **`append()`**: Agregar filas (deprecado en favor de concat)

## Datasets de Ejemplo

```python
import pandas as pd
import numpy as np

# Dataset 1: Información de empleados
empleados = pd.DataFrame({
    'empleado_id': [101, 102, 103, 104, 105, 106, 107],
    'nombre': ['Ana García', 'Luis Martín', 'María López', 'Carlos Ruiz', 
               'Elena Torres', 'Jorge Silva', 'Patricia Vega'],
    'departamento_id': [1, 2, 1, 3, 2, 1, 3],
    'fecha_ingreso': pd.to_datetime(['2020-01-15', '2019-03-22', '2021-06-10', 
                                   '2018-11-05', '2020-09-18', '2021-02-28', '2019-07-12']),
    'salario': [45000, 52000, 48000, 60000, 55000, 43000, 58000]
})

# Dataset 2: Información de departamentos
departamentos = pd.DataFrame({
    'departamento_id': [1, 2, 3, 4],
    'nombre_depto': ['Tecnología', 'Ventas', 'Marketing', 'Recursos Humanos'],
    'presupuesto': [150000, 120000, 80000, 90000],
    'gerente': ['Ana García', 'Elena Torres', 'Carlos Ruiz', 'Sin asignar']
})

# Dataset 3: Evaluaciones de desempeño
evaluaciones = pd.DataFrame({
    'empleado_id': [101, 102, 103, 104, 105, 108, 109],  # Incluye IDs que no existen en empleados
    'año': [2023, 2023, 2023, 2023, 2023, 2023, 2023],
    'calificacion': [4.5, 4.2, 4.8, 4.0, 4.6, 3.8, 4.1],
    'comentarios': ['Excelente trabajo', 'Muy bueno', 'Excepcional', 
                   'Buen desempeño', 'Sobresaliente', 'Satisfactorio', 'Muy bueno']
})

# Dataset 4: Proyectos
proyectos = pd.DataFrame({
    'proyecto_id': ['P001', 'P002', 'P003', 'P004'],
    'nombre_proyecto': ['Sistema CRM', 'App Mobile', 'Website Redesign', 'Data Analytics'],
    'departamento_id': [1, 1, 2, 1],
    'presupuesto_proyecto': [80000, 45000, 30000, 60000],
    'estado': ['Activo', 'Completado', 'Activo', 'En Planificación']
})

# Dataset 5: Asignaciones empleado-proyecto
asignaciones = pd.DataFrame({
    'empleado_id': [101, 101, 102, 103, 104, 105, 106],
    'proyecto_id': ['P001', 'P004', 'P002', 'P003', 'P001', 'P003', 'P002'],
    'horas_asignadas': [40, 20, 35, 30, 45, 25, 40],
    'rol_proyecto': ['Lead', 'Analyst', 'Developer', 'Designer', 
                    'Manager', 'Developer', 'Tester']
})

print("Datasets creados:")
print(f"Empleados: {empleados.shape}")
print(f"Departamentos: {departamentos.shape}")
print(f"Evaluaciones: {evaluaciones.shape}")
print(f"Proyectos: {proyectos.shape}")
print(f"Asignaciones: {asignaciones.shape}")

# Mostrar algunos datos
print("\nEmpleados:")
print(empleados)
print("\nDepartamentos:")
print(departamentos)
```

## 1. Concatenación (concat)

### Concatenación Básica

```python
# Crear DataFrames adicionales para ejemplos de concat
nuevos_empleados_2023 = pd.DataFrame({
    'empleado_id': [108, 109, 110],
    'nombre': ['Roberto Díaz', 'Carmen Morales', 'Diego Fernández'],
    'departamento_id': [2, 3, 1],
    'fecha_ingreso': pd.to_datetime(['2023-01-10', '2023-02-15', '2023-03-01']),
    'salario': [47000, 51000, 44000]
})

nuevos_empleados_2024 = pd.DataFrame({
    'empleado_id': [111, 112],
    'nombre': ['Isabel Herrera', 'Miguel Santos'],
    'departamento_id': [1, 2],
    'fecha_ingreso': pd.to_datetime(['2024-01-08', '2024-01-22']),
    'salario': [49000, 53000]
})

# Concatenación vertical (agregar filas)
todos_empleados = pd.concat([empleados, nuevos_empleados_2023, nuevos_empleados_2024], 
                           ignore_index=True)
print("Concatenación vertical - Todos los empleados:")
print(todos_empleados)
print(f"Shape original: {empleados.shape}, Shape final: {todos_empleados.shape}")
```

### Concatenación con Etiquetas

```python
# Concatenación con etiquetas para identificar origen
empleados_con_etiquetas = pd.concat([
    empleados,
    nuevos_empleados_2023,
    nuevos_empleados_2024
], keys=['2022_anterior', '2023', '2024'], ignore_index=False)

print("Concatenación con etiquetas (índice jerárquico):")
print(empleados_con_etiquetas.head(10))

# Acceder a datos por etiqueta
print("\nEmpleados contratados en 2023:")
print(empleados_con_etiquetas.loc['2023'])
```

### Concatenación Horizontal

```python
# Crear DataFrames con información adicional (mismos índices)
info_personal = pd.DataFrame({
    'telefono': ['555-0101', '555-0102', '555-0103', '555-0104', '555-0105'],
    'email': ['ana.garcia@empresa.com', 'luis.martin@empresa.com', 
              'maria.lopez@empresa.com', 'carlos.ruiz@empresa.com', 
              'elena.torres@empresa.com']
}, index=[101, 102, 103, 104, 105])

beneficios = pd.DataFrame({
    'seguro_medico': [True, True, False, True, True],
    'dias_vacaciones': [20, 25, 15, 30, 22]
}, index=[101, 102, 103, 104, 105])

# Concatenación horizontal (agregar columnas)
empleados_completo = pd.concat([empleados.set_index('empleado_id'), 
                               info_personal, beneficios], axis=1)
print("Concatenación horizontal:")
print(empleados_completo)
```

### Concatenación con Manejo de Valores Faltantes

```python
# DataFrames con columnas diferentes
df1 = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

df2 = pd.DataFrame({
    'B': [10, 11, 12],
    'C': [13, 14, 15],
    'D': [16, 17, 18]
})

# Concatenación con columnas diferentes
concat_diferentes = pd.concat([df1, df2], ignore_index=True)
print("Concatenación con columnas diferentes:")
print(concat_diferentes)

# Concatenación solo con intersección de columnas
concat_interseccion = pd.concat([df1, df2], ignore_index=True, join='inner')
print("\nConcatenación solo columnas comunes:")
print(concat_interseccion)

# Concatenación con columnas específicas
concat_especificas = pd.concat([df1[['A', 'B']], df2[['B', 'D']]], 
                              ignore_index=True, sort=False)
print("\nConcatenación con columnas específicas:")
print(concat_especificas)
```

## 2. Merge - Combinaciones Tipo SQL

### Inner Join (Intersección)

```python
# Inner join - solo registros que existen en ambos DataFrames
empleados_departamentos = pd.merge(empleados, departamentos, on='departamento_id', how='inner')
print("Inner Join - Empleados con Departamentos:")
print(empleados_departamentos[['nombre', 'nombre_depto', 'salario', 'presupuesto']])

# Verificar que solo incluye departamentos que tienen empleados
print(f"\nDepartamentos únicos en empleados: {empleados['departamento_id'].unique()}")
print(f"Departamentos únicos en resultado: {empleados_departamentos['departamento_id'].unique()}")
```

### Left Join (Preservar tabla izquierda)

```python
# Left join - todos los empleados, incluso si no tienen departamento válido
empleados_con_evaluaciones = pd.merge(empleados, evaluaciones, 
                                     on='empleado_id', how='left')
print("Left Join - Todos los empleados con sus evaluaciones (si existen):")
print(empleados_con_evaluaciones[['nombre', 'calificacion', 'comentarios']])

# Identificar empleados sin evaluación
sin_evaluacion = empleados_con_evaluaciones[empleados_con_evaluaciones['calificacion'].isnull()]
print(f"\nEmpleados sin evaluación: {len(sin_evaluacion)}")
print(sin_evaluacion[['nombre', 'departamento_id']])
```

### Right Join (Preservar tabla derecha)

```python
# Right join - todas las evaluaciones, incluso si el empleado no existe
evaluaciones_con_empleados = pd.merge(empleados, evaluaciones, 
                                     on='empleado_id', how='right')
print("Right Join - Todas las evaluaciones con información de empleados:")
print(evaluaciones_con_empleados[['empleado_id', 'nombre', 'calificacion']])

# Identificar evaluaciones de empleados que no están en la tabla empleados
evaluaciones_huerfanas = evaluaciones_con_empleados[evaluaciones_con_empleados['nombre'].isnull()]
print(f"\nEvaluaciones sin empleado correspondiente: {len(evaluaciones_huerfanas)}")
print(evaluaciones_huerfanas[['empleado_id', 'calificacion']])
```

### Outer Join (Unión completa)

```python
# Outer join - todos los registros de ambas tablas
union_completa = pd.merge(empleados, evaluaciones, on='empleado_id', how='outer')
print("Outer Join - Empleados y evaluaciones (todos los registros):")
print(union_completa[['empleado_id', 'nombre', 'calificacion']].fillna('N/A'))

# Análisis de la unión completa
print(f"\nTotal registros: {len(union_completa)}")
print(f"Con información de empleado: {union_completa['nombre'].notna().sum()}")
print(f"Con información de evaluación: {union_completa['calificacion'].notna().sum()}")
print(f"Con ambas informaciones: {(union_completa['nombre'].notna() & union_completa['calificacion'].notna()).sum()}")
```

### Merge con Múltiples Columnas

```python
# Crear dataset con clave compuesta
ventas_mensuales = pd.DataFrame({
    'empleado_id': [101, 101, 102, 102, 103, 103],
    'mes': [1, 2, 1, 2, 1, 2],
    'ventas': [15000, 18000, 12000, 14000, 20000, 22000]
})

objetivos_mensuales = pd.DataFrame({
    'empleado_id': [101, 101, 102, 102, 103, 104],
    'mes': [1, 2, 1, 2, 1, 1],
    'objetivo': [16000, 17000, 13000, 15000, 19000, 25000]
})

# Merge con múltiples columnas
ventas_vs_objetivos = pd.merge(ventas_mensuales, objetivos_mensuales, 
                              on=['empleado_id', 'mes'], how='outer')
ventas_vs_objetivos['cumplimiento'] = (ventas_vs_objetivos['ventas'] / 
                                      ventas_vs_objetivos['objetivo'] * 100).round(1)

print("Merge con múltiples columnas - Ventas vs Objetivos:")
print(ventas_vs_objetivos)
```

### Merge con Columnas de Nombres Diferentes

```python
# Simular tablas con nombres de columnas diferentes
empleados_alt = empleados.rename(columns={'empleado_id': 'id_empleado'})

# Merge especificando columnas diferentes
merge_nombres_diferentes = pd.merge(empleados_alt, evaluaciones, 
                                   left_on='id_empleado', right_on='empleado_id')
print("Merge con nombres de columnas diferentes:")
print(merge_nombres_diferentes[['id_empleado', 'empleado_id', 'nombre', 'calificacion']].head())

# Eliminar columna duplicada
merge_limpio = merge_nombres_diferentes.drop('empleado_id', axis=1)
print("\nResultado limpio:")
print(merge_limpio[['id_empleado', 'nombre', 'calificacion']].head())
```

## 3. Join - Combinación por Índice

### Join Básico

```python
# Preparar DataFrames con índice
empleados_indexed = empleados.set_index('empleado_id')
evaluaciones_indexed = evaluaciones.set_index('empleado_id')

# Join por índice (equivale a merge con índices)
empleados_evaluaciones_join = empleados_indexed.join(evaluaciones_indexed, how='left')
print("Join por índice - Empleados con evaluaciones:")
print(empleados_evaluaciones_join[['nombre', 'salario', 'calificacion']])
```

### Join con Sufijos

```python
# Crear conflicto de nombres de columnas
evaluaciones_con_nombre = evaluaciones.copy()
evaluaciones_con_nombre['nombre'] = 'Evaluación ' + evaluaciones_con_nombre['empleado_id'].astype(str)

empleados_indexed = empleados.set_index('empleado_id')
evaluaciones_con_nombre_indexed = evaluaciones_con_nombre.set_index('empleado_id')

# Join con sufijos para manejar nombres duplicados
join_con_sufijos = empleados_indexed.join(evaluaciones_con_nombre_indexed, 
                                         how='inner', rsuffix='_eval')
print("Join con sufijos para nombres duplicados:")
print(join_con_sufijos[['nombre', 'nombre_eval', 'calificacion']])
```

## 4. Operaciones Complejas de Combinación

### Merge en Cadena

```python
# Combinar múltiples DataFrames en secuencia
resultado_cadena = (empleados
                   .merge(departamentos, on='departamento_id', how='left')
                   .merge(evaluaciones, on='empleado_id', how='left')
                   .merge(asignaciones, on='empleado_id', how='left')
                   .merge(proyectos, on='proyecto_id', how='left'))

print("Merge en cadena - Vista completa:")
columnas_importantes = ['nombre', 'nombre_depto', 'calificacion', 
                       'nombre_proyecto', 'rol_proyecto', 'horas_asignadas']
print(resultado_cadena[columnas_importantes].head(10))

print(f"\nShape final: {resultado_cadena.shape}")
print(f"Columnas: {list(resultado_cadena.columns)}")
```

### Merge con Agregación Previa

```python
# Calcular estadísticas por departamento antes de merge
estadisticas_depto = empleados.groupby('departamento_id').agg({
    'salario': ['mean', 'count', 'sum'],
    'empleado_id': 'count'
}).round(2)

# Aplanar columnas multi-nivel
estadisticas_depto.columns = ['_'.join(col).strip() for col in estadisticas_depto.columns.values]
estadisticas_depto = estadisticas_depto.rename(columns={'empleado_id_count': 'num_empleados'})
estadisticas_depto = estadisticas_depto.reset_index()

# Merge con estadísticas
empleados_con_stats = pd.merge(empleados, estadisticas_depto, on='departamento_id')
empleados_con_stats['salario_vs_promedio'] = (empleados_con_stats['salario'] / 
                                              empleados_con_stats['salario_mean']).round(2)

print("Empleados con estadísticas departamentales:")
print(empleados_con_stats[['nombre', 'salario', 'salario_mean', 
                          'salario_vs_promedio', 'num_empleados']])
```

### Merge con Validación

```python
# Merge con validación de relaciones
try:
    # Validar que es relación uno-a-uno
    merge_validado = pd.merge(empleados, departamentos, on='departamento_id', 
                             validate='many_to_one')
    print("Validación exitosa: relación many-to-one")
except pd.errors.MergeError as e:
    print(f"Error de validación: {e}")

# Verificar duplicados antes de merge
print(f"\nDuplicados en empleados por departamento_id: {empleados.duplicated('departamento_id').sum()}")
print(f"Duplicados en departamentos por departamento_id: {departamentos.duplicated('departamento_id').sum()}")
```

## 5. Casos de Uso Prácticos

### Dashboard de Empleados Completo

```python
def crear_dashboard_empleados():
    """Crear dashboard completo combinando todas las fuentes de datos"""
    
    # Combinar todas las fuentes
    dashboard = (empleados
                .merge(departamentos, on='departamento_id', how='left')
                .merge(evaluaciones, on='empleado_id', how='left'))
    
    # Agregar información de proyectos
    proyectos_por_empleado = (asignaciones
                             .merge(proyectos, on='proyecto_id', how='left')
                             .groupby('empleado_id')
                             .agg({
                                 'proyecto_id': 'count',
                                 'horas_asignadas': 'sum',
                                 'nombre_proyecto': lambda x: ', '.join(x.astype(str))
                             })
                             .rename(columns={
                                 'proyecto_id': 'num_proyectos',
                                 'horas_asignadas': 'total_horas_proyectos',
                                 'nombre_proyecto': 'proyectos_asignados'
                             }))
    
    # Combinar con información de proyectos
    dashboard_completo = dashboard.merge(proyectos_por_empleado, 
                                       left_on='empleado_id', 
                                       right_index=True, 
                                       how='left')
    
    # Llenar valores nulos
    dashboard_completo['num_proyectos'] = dashboard_completo['num_proyectos'].fillna(0).astype(int)
    dashboard_completo['total_horas_proyectos'] = dashboard_completo['total_horas_proyectos'].fillna(0)
    dashboard_completo['proyectos_asignados'] = dashboard_completo['proyectos_asignados'].fillna('Ninguno')
    
    return dashboard_completo

dashboard_final = crear_dashboard_empleados()
print("Dashboard completo de empleados:")
columnas_dashboard = ['nombre', 'nombre_depto', 'salario', 'calificacion', 
                     'num_proyectos', 'total_horas_proyectos']
print(dashboard_final[columnas_dashboard])

# Estadísticas del dashboard
print(f"\nEstadísticas del dashboard:")
print(f"Total empleados: {len(dashboard_final)}")
print(f"Empleados con evaluación: {dashboard_final['calificacion'].notna().sum()}")
print(f"Empleados con proyectos: {(dashboard_final['num_proyectos'] > 0).sum()}")
print(f"Promedio de proyectos por empleado: {dashboard_final['num_proyectos'].mean():.1f}")
```

### Análisis de Asignación de Recursos

```python
def analizar_asignacion_recursos():
    """Análisis completo de asignación de recursos por departamento y proyecto"""
    
    # Combinar datos de asignaciones, empleados, proyectos y departamentos
    analisis = (asignaciones
               .merge(empleados[['empleado_id', 'nombre', 'departamento_id', 'salario']], 
                      on='empleado_id')
               .merge(proyectos[['proyecto_id', 'nombre_proyecto', 'presupuesto_proyecto']], 
                      on='proyecto_id')
               .merge(departamentos[['departamento_id', 'nombre_depto']], 
                      on='departamento_id'))
    
    # Calcular costo por hora (aproximado)
    analisis['costo_por_hora'] = analisis['salario'] / (52 * 40)  # Asumiendo 40h/semana
    analisis['costo_asignacion'] = analisis['horas_asignadas'] * analisis['costo_por_hora']
    
    # Resumen por proyecto
    resumen_proyectos = analisis.groupby(['proyecto_id', 'nombre_proyecto']).agg({
        'horas_asignadas': 'sum',
        'costo_asignacion': 'sum',
        'empleado_id': 'count',
        'presupuesto_proyecto': 'first'
    }).rename(columns={'empleado_id': 'num_empleados_asignados'})
    
    resumen_proyectos['porcentaje_presupuesto_usado'] = (
        resumen_proyectos['costo_asignacion'] / 
        resumen_proyectos['presupuesto_proyecto'] * 100
    ).round(1)
    
    print("Análisis de asignación de recursos por proyecto:")
    print(resumen_proyectos)
    
    # Identificar proyectos con problemas
    sobre_presupuesto = resumen_proyectos[
        resumen_proyectos['porcentaje_presupuesto_usado'] > 80
    ]
    
    if len(sobre_presupuesto) > 0:
        print(f"\n⚠️  Proyectos cerca o sobre presupuesto:")
        print(sobre_presupuesto[['nombre_proyecto', 'porcentaje_presupuesto_usado']])
    
    return analisis, resumen_proyectos

analisis_recursos, resumen_proyectos = analizar_asignacion_recursos()
```

### Sistema de Recomendaciones

```python
def recomendar_asignaciones():
    """Sistema de recomendación para nuevas asignaciones"""
    
    # Obtener capacidad disponible por empleado
    capacidad_empleados = empleados[['empleado_id', 'nombre', 'departamento_id']].copy()
    
    # Calcular horas ya asignadas
    horas_actuales = asignaciones.groupby('empleado_id')['horas_asignadas'].sum()
    capacidad_empleados = capacidad_empleados.merge(horas_actuales, 
                                                   left_on='empleado_id', 
                                                   right_index=True, 
                                                   how='left')
    capacidad_empleados['horas_asignadas'] = capacidad_empleados['horas_asignadas'].fillna(0)
    
    # Asumir capacidad máxima de 40 horas
    capacidad_empleados['capacidad_disponible'] = 40 - capacidad_empleados['horas_asignadas']
    
    # Agregar información de evaluaciones
    capacidad_empleados = capacidad_empleados.merge(evaluaciones[['empleado_id', 'calificacion']], 
                                                   on='empleado_id', how='left')
    
    # Agregar información de departamentos
    capacidad_empleados = capacidad_empleados.merge(departamentos[['departamento_id', 'nombre_depto']], 
                                                   on='departamento_id')
    
    # Filtrar empleados disponibles con buena evaluación
    empleados_recomendados = capacidad_empleados[
        (capacidad_empleados['capacidad_disponible'] > 10) &
        (capacidad_empleados['calificacion'].fillna(4.0) >= 4.0)
    ].sort_values(['calificacion', 'capacidad_disponible'], ascending=[False, False])
    
    print("Empleados recomendados para nuevas asignaciones:")
    columnas_rec = ['nombre', 'nombre_depto', 'capacidad_disponible', 'calificacion']
    print(empleados_recomendados[columnas_rec])
    
    return empleados_recomendados

recomendaciones = recomendar_asignaciones()
```

## 6. Técnicas Avanzadas

### Merge Asof (Merge por Proximidad)

```python
# Crear datos de series temporales para ejemplo
fechas_transacciones = pd.DataFrame({
    'timestamp': pd.to_datetime(['2023-01-01 09:00', '2023-01-01 11:00', 
                                '2023-01-01 14:00', '2023-01-01 16:00']),
    'transaccion_id': [1001, 1002, 1003, 1004],
    'monto': [150, 200, 75, 300]
})

fechas_tipos_cambio = pd.DataFrame({
    'timestamp': pd.to_datetime(['2023-01-01 08:00', '2023-01-01 12:00', 
                                '2023-01-01 15:00']),
    'tipo_cambio_usd': [0.95, 0.96, 0.94]
})

# Merge asof - encuentra el valor más cercano hacia atrás en el tiempo
transacciones_con_cambio = pd.merge_asof(fechas_transacciones.sort_values('timestamp'),
                                        fechas_tipos_cambio.sort_values('timestamp'),
                                        on='timestamp',
                                        direction='backward')

transacciones_con_cambio['monto_usd'] = (transacciones_con_cambio['monto'] * 
                                        transacciones_con_cambio['tipo_cambio_usd']).round(2)

print("Merge Asof - Transacciones con tipo de cambio:")
print(transacciones_con_cambio)
```

### Cross Join (Producto Cartesiano)

```python
# Crear todas las combinaciones posibles
empleados_mini = empleados[['empleado_id', 'nombre']].head(3)
proyectos_mini = proyectos[['proyecto_id', 'nombre_proyecto']].head(2)

# Cross join
cross_join = empleados_mini.merge(proyectos_mini, how='cross')
print("Cross Join - Todas las combinaciones empleado-proyecto:")
print(cross_join)

# Útil para crear matrices de planificación
print(f"\nTotal combinaciones: {len(cross_join)}")
```

### Merge con Transformación de Datos

```python
def merge_con_ranking():
    """Merge incluyendo rankings y transformaciones"""
    
    # Crear ranking de empleados por salario dentro de cada departamento
    empleados_con_ranking = empleados.copy()
    empleados_con_ranking['ranking_salario_depto'] = (
        empleados_con_ranking.groupby('departamento_id')['salario']
        .rank(method='dense', ascending=False)
    )
    
    # Merge con información adicional
    resultado = (empleados_con_ranking
                .merge(departamentos, on='departamento_id')
                .merge(evaluaciones, on='empleado_id', how='left'))
    
    # Crear categorías de rendimiento
    resultado['categoria_rendimiento'] = pd.cut(
        resultado['calificacion'].fillna(3.5),
        bins=[0, 3.5, 4.2, 5.0],
        labels=['Básico', 'Bueno', 'Excelente']
    )
    
    return resultado

empleados_completo = merge_con_ranking()
print("Empleados con ranking y categorización:")
columnas_importantes = ['nombre', 'nombre_depto', 'ranking_salario_depto', 
                       'calificacion', 'categoria_rendimiento']
print(empleados_completo[columnas_importantes])
```

## 7. Optimización y Performance

### Comparación de Métodos

```python
import time

def comparar_metodos_join(df1, df2):
    """Comparar performance de diferentes métodos de join"""
    
    # Preparar datos
    df1_indexed = df1.set_index('empleado_id')
    df2_indexed = df2.set_index('empleado_id')
    
    # Método 1: merge
    start = time.time()
    resultado1 = pd.merge(df1, df2, on='empleado_id', how='left')
    tiempo1 = time.time() - start
    
    # Método 2: join
    start = time.time()
    resultado2 = df1_indexed.join(df2_indexed, how='left')
    tiempo2 = time.time() - start
    
    # Método 3: concat con keys
    start = time.time()
    resultado3 = pd.concat([df1_indexed, df2_indexed], axis=1, join='outer')
    tiempo3 = time.time() - start
    
    print(f"Merge: {tiempo1:.6f}s")
    print(f"Join: {tiempo2:.6f}s")
    print(f"Concat: {tiempo3:.6f}s")
    
    return resultado1

resultado_benchmark = comparar_metodos_join(empleados, evaluaciones)
print("Resultado del benchmark:")
print(resultado_benchmark[['nombre', 'calificacion']].head())
```

### Mejores Prácticas para Performance

```python
def optimizar_merges():
    """Técnicas para optimizar operaciones de merge"""
    
    # 1. Usar índices cuando sea apropiado
    print("1. Optimización con índices:")
    empleados_idx = empleados.set_index('departamento_id')
    departamentos_idx = departamentos.set_index('departamento_id')
    
    start = time.time()
    resultado_optimizado = empleados_idx.join(departamentos_idx, how='left')
    tiempo_optimizado = time.time() - start
    
    start = time.time()
    resultado_normal = pd.merge(empleados, departamentos, on='departamento_id', how='left')
    tiempo_normal = time.time() - start
    
    print(f"   Con índices: {tiempo_optimizado:.6f}s")
    print(f"   Sin índices: {tiempo_normal:.6f}s")
    
    # 2. Filtrar datos antes de merge cuando sea posible
    print("\n2. Filtrado previo:")
    empleados_filtrados = empleados[empleados['salario'] > 50000]
    resultado_filtrado = pd.merge(empleados_filtrados, departamentos, on='departamento_id')
    print(f"   Registros antes del filtro: {len(empleados)}")
    print(f"   Registros después del merge filtrado: {len(resultado_filtrado)}")
    
    # 3. Usar categorías para strings repetitivos
    print("\n3. Optimización de memoria con categorías:")
    departamentos_cat = departamentos.copy()
    departamentos_cat['nombre_depto'] = departamentos_cat['nombre_depto'].astype('category')
    
    print(f"   Memoria antes: {departamentos['nombre_depto'].memory_usage(deep=True)} bytes")
    print(f"   Memoria después: {departamentos_cat['nombre_depto'].memory_usage(deep=True)} bytes")

optimizar_merges()
```

## 8. Manejo de Errores y Validación

### Validación de Datos Antes de Merge

```python
def validar_antes_merge(df1, df2, key_column):
    """Validar datos antes de realizar merge"""
    
    print(f"Validación para merge en columna: {key_column}")
    
    # Verificar valores nulos en la clave
    nulos_df1 = df1[key_column].isnull().sum()
    nulos_df2 = df2[key_column].isnull().sum()
    
    print(f"Valores nulos en df1[{key_column}]: {nulos_df1}")
    print(f"Valores nulos en df2[{key_column}]: {nulos_df2}")
    
    # Verificar tipos de datos
    tipo_df1 = df1[key_column].dtype
    tipo_df2 = df2[key_column].dtype
    
    print(f"Tipo de dato en df1: {tipo_df1}")
    print(f"Tipo de dato en df2: {tipo_df2}")
    
    if tipo_df1 != tipo_df2:
        print("⚠️  ADVERTENCIA: Los tipos de datos no coinciden")
    
    # Verificar duplicados
    duplicados_df1 = df1[key_column].duplicated().sum()
    duplicados_df2 = df2[key_column].duplicated().sum()
    
    print(f"Duplicados en df1: {duplicados_df1}")
    print(f"Duplicados en df2: {duplicados_df2}")
    
    # Verificar intersección
    comunes = set(df1[key_column]) & set(df2[key_column])
    solo_df1 = set(df1[key_column]) - set(df2[key_column])
    solo_df2 = set(df2[key_column]) - set(df1[key_column])
    
    print(f"Valores comunes: {len(comunes)}")
    print(f"Solo en df1: {len(solo_df1)}")
    print(f"Solo en df2: {len(solo_df2)}")
    
    if solo_df1:
        print(f"Valores solo en df1: {list(solo_df1)[:5]}...")
    if solo_df2:
        print(f"Valores solo en df2: {list(solo_df2)[:5]}...")

# Ejecutar validación
validar_antes_merge(empleados, evaluaciones, 'empleado_id')
```

### Limpieza Post-Merge

```python
def limpiar_resultado_merge(df):
    """Limpiar y validar resultado de merge"""
    
    print("Limpieza post-merge:")
    print(f"Shape inicial: {df.shape}")
    
    # Identificar duplicados completos
    duplicados_completos = df.duplicated().sum()
    print(f"Filas completamente duplicadas: {duplicados_completos}")
    
    if duplicados_completos > 0:
        df = df.drop_duplicates()
        print(f"Shape después de eliminar duplicados: {df.shape}")
    
    # Identificar columnas con muchos nulos
    nulos_por_columna = df.isnull().sum()
    columnas_problematicas = nulos_por_columna[nulos_por_columna > len(df) * 0.5]
    
    if len(columnas_problematicas) > 0:
        print(f"Columnas con >50% nulos: {list(columnas_problematicas.index)}")
    
    # Estadísticas finales
    print(f"Registros finales: {len(df)}")
    print(f"Columnas finales: {len(df.columns)}")
    print(f"Memoria total: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

# Ejemplo de limpieza
resultado_sucio = pd.merge(empleados, evaluaciones, on='empleado_id', how='outer')
resultado_limpio = limpiar_resultado_merge(resultado_sucio)
```

## Mejores Prácticas

### 1. Selección del Tipo de Join
- **Inner**: Cuando solo necesites registros que existen en ambas tablas
- **Left**: Para preservar todos los registros de la tabla principal
- **Right**: Raramente usado, prefiere reordenar y usar left
- **Outer**: Para análisis de completitud de datos

### 2. Performance
- Usar índices para joins frecuentes
- Filtrar datos antes de merge cuando sea posible
- Considerar el orden de las operaciones en merges múltiples
- Usar `validate` parameter para verificar tipos de relación

### 3. Manejo de Memoria
- Eliminar columnas innecesarias antes del merge
- Usar tipos de datos apropiados (categorías para strings)
- Procesar en chunks para datasets muy grandes

### 4. Debugging
```python
def debug_merge(df1, df2, **merge_kwargs):
    """Helper para debugear operaciones merge"""
    print(f"Merge Debug:")
    print(f"df1 shape: {df1.shape}")
    print(f"df2 shape: {df2.shape}")
    
    resultado = pd.merge(df1, df2, **merge_kwargs)
    print(f"Resultado shape: {resultado.shape}")
    
    if 'on' in merge_kwargs:
        key = merge_kwargs['on']
        print(f"Registros únicos en clave '{key}':")
        print(f"  df1: {df1[key].nunique()}")
        print(f"  df2: {df2[key].nunique()}")
        print(f"  resultado: {resultado[key].nunique()}")
    
    return resultado

# Ejemplo de uso del debug
resultado_debug = debug_merge(empleados, evaluaciones, on='empleado_id', how='left')
```

## Recursos y Referencias

- **Documentación oficial de Merge**: https://pandas.pydata.org/docs/user_guide/merging.html
- **Database-style joins**: https://pandas.pydata.org/docs/user_guide/merging.html#database-style-dataframe-or-named-series-joining-merging
- **Concatenation**: https://pandas.pydata.org/docs/user_guide/merging.html#concatenating-objects
- **Performance tips**: https://pandas.pydata.org/docs/user_guide/enhancingperf.html

## Ejercicios Sugeridos

1. **Sistema HR Completo**: Combina todas las tablas para crear un sistema de gestión de recursos humanos
2. **Análisis de Brechas**: Identifica empleados sin evaluaciones, departamentos sin empleados, etc.
3. **Dashboard Ejecutivo**: Crea métricas agregadas combinando múltiples fuentes
4. **Sistema de Alertas**: Detecta inconsistencias entre diferentes tablas
5. **Optimización de Asignaciones**: Usa combinaciones para optimizar asignación de recursos
6. **Análisis Temporal**: Combina datos históricos para análisis de tendencias
7. **Reporte de Anomalías**: Identifica registros huérfanos o inconsistencias en relaciones

## Casos de Uso Avanzados

### 1. Data Warehouse Simulation
```python
def simular_data_warehouse():
    """Simular operaciones típicas de data warehouse"""
    
    # Tabla de hechos (fact table)
    fact_ventas = pd.DataFrame({
        'venta_id': range(1, 101),
        'empleado_id': np.random.choice(empleados['empleado_id'], 100),
        'producto_id': np.random.choice(range(1, 21), 100),
        'fecha_venta': pd.date_range('2023-01-01', periods=100, freq='D'),
        'cantidad': np.random.randint(1, 5, 100),
        'monto': np.random.uniform(100, 1000, 100).round(2)
    })
    
    # Tabla de dimensión productos
    dim_productos = pd.DataFrame({
        'producto_id': range(1, 21),
        'nombre_producto': [f'Producto {i}' for i in range(1, 21)],
        'categoria_producto': np.random.choice(['A', 'B', 'C'], 20),
        'precio_base': np.random.uniform(50, 500, 20).round(2)
    })
    
    # Star schema join
    ventas_completas = (fact_ventas
                       .merge(empleados[['empleado_id', 'nombre']], on='empleado_id')
                       .merge(dim_productos, on='producto_id')
                       .merge(departamentos, on='departamento_id'))
    
    print("Simulación Data Warehouse - Star Schema:")
    print(ventas_completas.groupby(['nombre_depto', 'categoria_producto'])['monto'].sum().unstack(fill_value=0))

simular_data_warehouse()
```

### 2. CDC (Change Data Capture) Simulation
```python
def simular_cdc():
    """Simular captura de cambios de datos"""
    
    # Estado actual
    empleados_actual = empleados.copy()
    
    # Cambios (actualizaciones)
    cambios = pd.DataFrame({
        'empleado_id': [101, 103, 105],
        'salario': [47000, 50000, 57000],
        'fecha_cambio': pd.Timestamp('2024-01-01')
    })
    
    # Nuevos empleados
    nuevos = pd.DataFrame({
        'empleado_id': [108, 109],
        'nombre': ['Nuevo Empleado 1', 'Nuevo Empleado 2'],
        'departamento_id': [1, 2],
        'fecha_ingreso': pd.Timestamp('2024-01-01'),
        'salario': [45000, 52000]
    })
    
    # Aplicar cambios
    empleados_actualizado = empleados_actual.merge(
        cambios[['empleado_id', 'salario']], 
        on='empleado_id', 
        how='left',
        suffixes=('', '_nuevo')
    )
    
    # Actualizar salarios donde hay cambios
    mask = empleados_actualizado['salario_nuevo'].notna()
    empleados_actualizado.loc[mask, 'salario'] = empleados_actualizado.loc[mask, 'salario_nuevo']
    empleados_actualizado = empleados_actualizado.drop('salario_nuevo', axis=1)
    
    # Agregar nuevos empleados
    empleados_final = pd.concat([empleados_actualizado, nuevos], ignore_index=True)
    
    print("Simulación CDC:")
    print(f"Empleados originales: {len(empleados)}")
    print(f"Empleados actualizados: {len(empleados_final)}")
    print(f"Cambios de salario aplicados: {mask.sum()}")

simular_cdc()
```

Esta guía completa cubre todas las operaciones fundamentales de combinación de datos en pandas, desde conceptos básicos hasta casos de uso avanzados. Cada sección incluye ejemplos prácticos y casos reales que encontrarás en proyectos de data science.