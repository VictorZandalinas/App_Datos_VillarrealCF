import pandas as pd
import os

# Lista de archivos que genera tu script original
files_to_check = [
    'match_events.parquet',
    'player_stats.parquet',
    'team_stats.parquet',
    'xg_events.parquet',
    'player_xg_stats.parquet',
    'team_officials.parquet'
]

print("🔍 ANALIZANDO ESTRUCTURA DE ARCHIVOS LOCALES (TARGET)")
print("=" * 60)

for filename in files_to_check:
    if os.path.exists(filename):
        print(f"\n📂 ARCHIVO: {filename}")
        try:
            df = pd.read_parquet(filename)
            print(f"   📊 Dimensiones: {df.shape}")
            print("   📋 Columnas y Ejemplo (Primera fila):")
            
            # Mostramos tipo de dato y valor de la primera fila
            sample = df.iloc[0] if not df.empty else pd.Series()
            for col in df.columns:
                val = sample[col] if not df.empty else "VACÍO"
                dtype = df[col].dtype
                print(f"      - {col} ({dtype}): {val}")
                
        except Exception as e:
            print(f"   ❌ Error leyendo archivo: {e}")
    else:
        print(f"\n⚠️ ARCHIVO NO ENCONTRADO: {filename}")

print("\n" + "=" * 60)