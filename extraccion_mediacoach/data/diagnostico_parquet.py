import pandas as pd
from pathlib import Path

# Prueba con las dos rutas posibles (ajusta si es necesario)
rutas = [
    Path('estadisticas_equipo.parquet'),
    Path('estadisticas_partido.parquet')
]

for r in rutas:
    print(f"\n--- REVISANDO ARCHIVO: {r} ---")
    if r.exists():
        try:
            df = pd.read_parquet(r)
            print("✅ Archivo encontrado.")
            print(f"📊 Columnas reales: {df.columns.tolist()}")
            print(f"📝 Primeras 3 filas:\n", df.head(3))
            
            # Revisar una jornada y un partido concreto
            col_j = next((c for c in df.columns if c.lower() == 'jornada'), None)
            col_p = next((c for c in df.columns if c.lower() == 'partido'), None)
            
            if col_j: print(f"📍 Valores únicos en {col_j}: {df[col_j].unique()[:5]}")
            if col_p: print(f"⚽ Ejemplo de {col_p}: {df[col_p].iloc[0]}")
        except Exception as e:
            print(f"❌ Error al leer: {e}")
    else:
        print("f❌ El archivo NO existe en esa ruta.")