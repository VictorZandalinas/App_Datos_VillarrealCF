import pandas as pd

# Cargar el parquet
df = pd.read_parquet("corners_tracking.parquet")

print("="*60)
print("📊 ESTRUCTURA DEL ARCHIVO PARQUET")
print("="*60)

print(f"\n📦 Total de filas: {len(df)}")
print(f"📋 Total de columnas: {len(df.columns)}")

print("\n📝 NOMBRES DE LAS COLUMNAS:")
print("-"*60)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\n🔍 INFORMACIÓN DETALLADA:")
print("-"*60)
print(df.info())

print("\n📊 TIPOS DE DATOS:")
print("-"*60)
print(df.dtypes)

print("\n👀 PRIMERAS 5 FILAS:")
print("-"*60)
print(df.head())

print("\n📈 ESTADÍSTICAS DESCRIPTIVAS (columnas numéricas):")
print("-"*60)
print(df.describe())