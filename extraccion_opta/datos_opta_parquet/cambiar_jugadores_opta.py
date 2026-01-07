import pandas as pd
import os

# Definir los cambios de equipos y dorsales
cambios_datos = {
    '5j6v5fqvmhuwq5u8mamm2sv10': {'equipo': 'Sin equipo', 'dorsal': None},
    '6wpnjdlg2p8f19y3gjpz1su6t': {'equipo': 'Getafe', 'dorsal': 17.0},
    '9jw0h4escxh3eqz8oh860r3yt': {'equipo': 'Sin equipo', 'dorsal': None},
    '5zoz3mjooujpv0wdcphyrztd6': {'equipo': 'Atlético de Madrid', 'dorsal': 10.0},
    '1wileoy8mxtd2t1b106uhwzro': {'equipo': 'Sin equipo', 'dorsal': None},
    '4rktv6j9sioe0gmu7jt77fsh5': {'equipo': 'Valencia', 'dorsal': 7.0},
    'cuqawmfekh86mgo758udxw7v9': {'equipo': 'Barcelona', 'dorsal': 40.0},
    '8xd59g0c36012q6mg5f2fl239': {'equipo': 'Sin equipo', 'dorsal': None},
    '1abvcjwaxlhc22zoxdqu0t7th': {'equipo': 'Sin equipo', 'dorsal': None},
    '59234as8h70fejyfm7492np3p': {'equipo': 'Girona', 'dorsal': 20.0},
    'ejcyvgclry4g5pc2rtptifphx': {'equipo': 'Girona', 'dorsal': 11.0},
    '1qkneebzln8byl9td2aswznmy': {'equipo': 'Villarreal', 'dorsal': 20.0},
    'albt55j4oir1veynmos532psk': {'equipo': 'Villarreal', 'dorsal': 15.0},
    '9g4ojrys88awchyj6zgmd6oes': {'equipo': 'Sin equipo', 'dorsal': None},
    '7lo14fnq32yyyqql6jws5w3xl': {'equipo': 'Sin equipo', 'dorsal': None},
    '8k88isvd2hwr5e4gn9oxzr2t': {'equipo': 'Sin equipo', 'dorsal': None},
    'bzv32rpgvs111w8e2gch452xg': {'equipo': 'Celta de Vigo', 'dorsal': 15.0},
    '4eezya3lwcwur6c8s6rl0rzvu': {'equipo': 'Sin equipo', 'dorsal': None},
}

def modificar_archivo(archivo):
    """Modifica equipos y dorsales en un archivo parquet específico"""
    try:
        print(f"🚀 Procesando {archivo}...")
        
        # Verificar que el archivo existe
        if not os.path.exists(archivo):
            print(f"❌ Archivo no encontrado: {archivo}")
            return
        
        # Leer el archivo parquet
        df = pd.read_parquet(archivo)
        
        # Buscar las columnas necesarias
        columnas_player_id = ['Player ID', 'player_id', 'PlayerID', 'playerId', 'Id Jugador', 'ID Jugador', 'id_jugador']
        columnas_team = ['Team Name', 'team_name', 'TeamName', 'Team', 'equipo', 'Equipo']
        columnas_dorsal = ['Shirt Number', 'shirt_number', 'ShirtNumber', 'Dorsal', 'dorsal', 'numero', 'Numero']
        
        # Encontrar columna Player ID
        columna_player_id = None
        for col in columnas_player_id:
            if col in df.columns:
                columna_player_id = col
                break
        
        # Encontrar columna Team
        columna_team = None
        for col in columnas_team:
            if col in df.columns:
                columna_team = col
                break
        
        # Encontrar columna Dorsal
        columna_dorsal = None
        for col in columnas_dorsal:
            if col in df.columns:
                columna_dorsal = col
                break
        
        # Verificar que se encontró al menos Player ID
        if columna_player_id is None:
            print(f"❌ No se encontró columna Player ID en {archivo}")
            return
        
        print(f"✅ Columnas encontradas en {archivo}:")
        print(f"   Player ID: {columna_player_id}")
        print(f"   Team: {columna_team if columna_team else 'No encontrada'}")
        print(f"   Dorsal: {columna_dorsal if columna_dorsal else 'No encontrada'}")
        print()
        
        # Aplicar los cambios
        cambios_realizados = 0
        
        for player_id, datos in cambios_datos.items():
            mask = df[columna_player_id] == player_id
            
            if mask.any():
                cambios_jugador = []
                
                # Cambiar equipo si existe la columna
                if columna_team is not None:
                    df.loc[mask, columna_team] = datos['equipo']
                    cambios_jugador.append(f"Equipo: {datos['equipo']}")
                
                # Cambiar dorsal si existe la columna y no es None
                if columna_dorsal is not None and datos['dorsal'] is not None:
                    df.loc[mask, columna_dorsal] = datos['dorsal']
                    cambios_jugador.append(f"Dorsal: {datos['dorsal']}")
                
                if cambios_jugador:
                    print(f"✅ Jugador {player_id}: {' - '.join(cambios_jugador)}")
                    cambios_realizados += 1
            else:
                print(f"⚠️  Jugador {player_id} no encontrado en {archivo}")
        
        if cambios_realizados > 0:
            # Guardar el archivo modificado
            df.to_parquet(archivo, index=False)
            print()
            print(f"💾 {archivo} guardado con {cambios_realizados} cambios")
        else:
            print(f"ℹ️  No se realizaron cambios en {archivo}")
            
    except Exception as e:
        print(f"❌ Error procesando {archivo}: {str(e)}")

def main():
    """Función principal"""
    print("🚀 Iniciando modificación de archivos parquet...")
    print(f"📁 Directorio actual: {os.getcwd()}")
    print()
    
    # Lista de archivos a procesar
    archivos = ['player_stats.parquet', 'abp_events.parquet']
    
    # Verificar archivos existentes
    archivos_existentes = []
    for archivo in archivos:
        if os.path.exists(archivo):
            archivos_existentes.append(archivo)
        else:
            print(f"⚠️  Archivo no encontrado: {archivo}")
    
    if not archivos_existentes:
        print("❌ No se encontraron archivos parquet para procesar")
        return
    
    print(f"📊 Procesando {len(archivos_existentes)} archivos...")
    print()
    
    # Procesar cada archivo
    for archivo in archivos_existentes:
        modificar_archivo(archivo)
        print()
    
    print("✅ Proceso completado!")
    print()
    print("📋 Resumen de cambios aplicados:")
    for player_id, datos in cambios_datos.items():
        if datos['dorsal'] is not None:
            print(f"  • {player_id}: {datos['equipo']} - Dorsal {datos['dorsal']}")
        else:
            print(f"  • {player_id}: {datos['equipo']}")

if __name__ == "__main__":
    main()