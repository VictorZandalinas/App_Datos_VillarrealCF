import pandas as pd
import os

# Definir los cambios de equipos
cambios_equipos = {
    'Real Oviedo': [7409, 29117],
    'RC Celta': [25768],
    'Otro Equipo': [31588, 1512, 835, 20, 2132, 1646, 821],
    'Villarreal CF': [31595, 22097, 24797]
}

# Lista de archivos a procesar
archivos = [
    'estadisticas_jugador.parquet',
    'eventos_partido.parquet',
    'maxima_exigencia.parquet',
    'rendimiento_5.parquet',
    'rendimiento_10.parquet',
    'rendimiento_15.parquet',
    'rendimiento_fisico.parquet'
]

def modificar_equipos(archivo):
    """Modifica los equipos en un archivo parquet específico"""
    try:
        print(f"Procesando {archivo}...")
        
        # Leer el archivo parquet
        df = pd.read_parquet(archivo)
        print(f"  📊 Filas en el archivo: {len(df)}")
        
        # Mostrar columnas disponibles
        print(f"  📋 Columnas disponibles: {list(df.columns)}")
        
        # Buscar la columna de equipo
        columna_equipo = None
        posibles_equipos = ['Equipo', 'equipo', 'team', 'Team']
        for col in posibles_equipos:
            if col in df.columns:
                columna_equipo = col
                break
        
        if columna_equipo is None:
            print(f"  ⚠️  No se encontró columna de equipo en {archivo}")
            print(f"  🔍 Columnas que contienen 'equipo': {[col for col in df.columns if 'equipo' in col.lower()]}")
            return
        
        print(f"  ✅ Columna de equipo encontrada: '{columna_equipo}'")
        
        # Buscar columna de ID jugador
        columna_id = None
        posibles_ids = ['Id Jugador', 'ID Jugador', 'id_jugador', 'Id_Jugador', 'jugador_id', 'ID', 'Id']
        for col in posibles_ids:
            if col in df.columns:
                columna_id = col
                break
        
        if columna_id is None:
            print(f"  ⚠️  No se encontró columna de ID jugador en {archivo}")
            print(f"  🔍 Columnas que contienen 'id' o 'jugador': {[col for col in df.columns if 'id' in col.lower() or 'jugador' in col.lower()]}")
            return
        
        print(f"  ✅ Columna de ID encontrada: '{columna_id}'")
        print(f"  🔍 Tipo de datos en {columna_id}: {df[columna_id].dtype}")
        print(f"  📊 Primeros 5 IDs: {df[columna_id].head().tolist()}")
        
        # Verificar tipos de datos y convertir si es necesario
        if df[columna_id].dtype == 'object':
            try:
                df[columna_id] = pd.to_numeric(df[columna_id], errors='coerce')
                print("  🔧 Convertidos IDs a numérico")
            except:
                print("  ⚠️  No se pudo convertir IDs a numérico")
        
        # Verificar qué IDs existen
        ids_existentes = set(df[columna_id].dropna().astype(int))
        print(f"  📈 Total de IDs únicos en el archivo: {len(ids_existentes)}")
        
        # Aplicar los cambios
        cambios_realizados = 0
        for nuevo_equipo, ids_jugadores in cambios_equipos.items():
            # Verificar qué IDs se encuentran
            ids_encontrados = set(ids_jugadores).intersection(ids_existentes)
            ids_no_encontrados = set(ids_jugadores) - ids_existentes
            
            print(f"  🎯 {nuevo_equipo}: {len(ids_encontrados)}/{len(ids_jugadores)} IDs encontrados")
            if ids_no_encontrados:
                print(f"    ❌ IDs no encontrados: {sorted(list(ids_no_encontrados))}")
            if ids_encontrados:
                print(f"    ✅ IDs encontrados: {sorted(list(ids_encontrados))}")
            
            # Aplicar cambios solo a los IDs encontrados
            if ids_encontrados:
                mask = df[columna_id].isin(list(ids_encontrados))
                equipos_anteriores = df.loc[mask, columna_equipo].unique()
                print(f"    🔄 Equipos anteriores: {list(equipos_anteriores)}")
                
                df.loc[mask, columna_equipo] = nuevo_equipo
                cambios_count = mask.sum()
                cambios_realizados += cambios_count
                print(f"    ✅ Cambiados {cambios_count} registros a {nuevo_equipo}")
        
        if cambios_realizados > 0:
            # Crear backup antes de guardar
            backup_name = f"{archivo}.backup"
            if not os.path.exists(backup_name):
                df_original = pd.read_parquet(archivo)
                df_original.to_parquet(backup_name, index=False)
                print(f"  💾 Backup creado: {backup_name}")
            
            # Guardar el archivo modificado
            df.to_parquet(archivo, index=False)
            print(f"  💾 {archivo} guardado con {cambios_realizados} cambios")
        else:
            print(f"  ℹ️  No se realizaron cambios en {archivo}")
        
    except Exception as e:
        print(f"  ❌ Error procesando {archivo}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Función principal"""
    print("🚀 Iniciando modificación de equipos en archivos parquet...")
    print(f"📁 Directorio actual: {os.getcwd()}")
    print()
    
    # Mostrar resumen de cambios a aplicar
    total_ids = sum(len(ids) for ids in cambios_equipos.values())
    print(f"📋 Cambios a aplicar ({total_ids} jugadores total):")
    for equipo, ids in cambios_equipos.items():
        print(f"  • {equipo}: {len(ids)} jugadores {ids}")
    print()
    
    # Verificar que los archivos existen
    archivos_existentes = []
    for archivo in archivos:
        if os.path.exists(archivo):
            archivos_existentes.append(archivo)
            size_mb = os.path.getsize(archivo) / 1024 / 1024
            print(f"✅ {archivo} ({size_mb:.1f} MB)")
        else:
            print(f"❌ Archivo no encontrado: {archivo}")
    
    if not archivos_existentes:
        print("❌ No se encontraron archivos parquet para procesar")
        return
    
    print(f"\n📊 Procesando {len(archivos_existentes)} archivos...")
    print("=" * 50)
    
    # Procesar cada archivo
    for i, archivo in enumerate(archivos_existentes, 1):
        print(f"\n[{i}/{len(archivos_existentes)}] {archivo}")
        print("-" * 40)
        modificar_equipos(archivo)
    
    print("\n" + "=" * 50)
    print("✅ Proceso completado!")

if __name__ == "__main__":
    main()