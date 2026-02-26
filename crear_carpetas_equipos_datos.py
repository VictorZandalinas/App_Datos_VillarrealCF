import pandas as pd
import os
import shutil  # Añadimos shutil para copiar archivos directamente

# ==========================================
# 1. CONFIGURACIÓN Y DICCIONARIO DE EQUIPOS
# ==========================================
equipos_info = {
    "Alavés": {"mc_id": 32, "opta_id": "4dtdjgnpdq9uw4sdutti0vaar", "sportian_name": "Alavés"},
    "Athletic Club": {"mc_id": 33, "opta_id": "3czravw89omgc9o4s0w3l1bg5", "sportian_name": "Athletic Club"},
    "Atlético de Madrid": {"mc_id": 34, "opta_id": "4ku8o6uf87yd8iecdalipo6wd", "sportian_name": "Atlético de Madrid"},
    "Barcelona": {"mc_id": 37, "opta_id": "agh9ifb2mw3ivjusgedj7c3fe", "sportian_name": "Barcelona"},
    "Celta de Vigo": {"mc_id": 35, "opta_id": "6f27yvbqcngegwsg2ozxxdj4", "sportian_name": "Celta de Vigo"},
    "Elche": {"mc_id": 80, "opta_id": "4yg9ttzw0m51048doksv8uq5r", "sportian_name": "Elche"},
    "Espanyol": {"mc_id": 36, "opta_id": "c8llrezkm3b3op4afrou6b487", "sportian_name": "Espanyol"},
    "Getafe": {"mc_id": 89, "opta_id": "1n1j0wsl763lq7ee1k0c11c02", "sportian_name": "Getafe"},
    "Girona": {"mc_id": 110, "opta_id": "7h7eg7q7dbwvzww78h9d5eh0h", "sportian_name": "Girona"},
    "Levante": {"mc_id": 76, "opta_id": "4grc9qgcvusllap8h5j6gc5h5", "sportian_name": "Levante"},
    "Mallorca": {"mc_id": 40, "opta_id": "50x1m4u58lffhq6v6ga1hbxmy", "sportian_name": "Mallorca"},
    "Osasuna": {"mc_id": 65, "opta_id": "2l0ldgiwsgb8d6y3z0sfgjzyj", "sportian_name": "Osasuna"},
    "Rayo Vallecano": {"mc_id": 43, "opta_id": "3budh3j9xivsid3ptm8ptpy4k", "sportian_name": "Rayo Vallecano"},
    "Real Betis": {"mc_id": 44, "opta_id": "ah8dala7suqqkj04n2l8xz4zd", "sportian_name": "Real Betis"},
    "Real Madrid": {"mc_id": 45, "opta_id": "3kq9cckrnlogidldtdie2fkbl", "sportian_name": "Real Madrid"},
    "Real Oviedo": {"mc_id": 46, "opta_id": "clo928vcczjp0mczavs5o7p5k", "sportian_name": "Real Oviedo"},
    "Real Sociedad": {"mc_id": 47, "opta_id": "63f5h8t5e9qm1fqmvfkb23ghh", "sportian_name": "Real Sociedad"},
    "Sevilla": {"mc_id": 38, "opta_id": "10eyb18v5puw4ez03ocaug09m", "sportian_name": "Sevilla"},
    "Valencia": {"mc_id": 50, "opta_id": "ba5e91hjacvma2sjvixn00pjo", "sportian_name": "Valencia"},
    "Villarreal": {"mc_id": 64, "opta_id": "74mcjsm72vr3l9pw2i4qfjchj", "sportian_name": "Villarreal"}
}

carpetas_config = {
    "mediacoach": {
        "ruta": "extraccion_mediacoach/data",
        "col_partido": "ID PARTIDO",
        "claves_equipo":["ID EQUIPO", "id_equipo"]
    },
    "opta": {
        "ruta": "extraccion_opta/datos_opta_parquet",
        "col_partido": "Match ID",
        "claves_equipo": ["Team ID", "team_id"]
    },
    "sportian": {
        "ruta": "extraccion_sportian",
        "col_partido": "ID_Partido",
        "claves_equipo": ["NombreEquipoJugador", "equipo"]
    }
}

# Aquí definimos los archivos que se deben copiar enteros sin retocar
archivos_copia_directa =['estadisticas_abp.parquet', 'estadisticas_abp_liga.parquet']
carpeta_destino_base = "data_por_equipos"

def buscar_columna_equipo(columnas, claves):
    """Busca dinámicamente la columna del equipo en el parquet"""
    for col in columnas:
        for clave in claves:
            if clave.lower() in col.lower():
                return col
    return None

# ==========================================
# 2. PASO 1: DESCUBRIR PARTIDOS EN LOS DATOS ORIGEN
# ==========================================
print("Paso 1: Identificando los partidos de cada equipo en los datos de origen...")
partidos_por_equipo = {eq: {'mediacoach': set(), 'opta': set(), 'sportian': set()} for eq in equipos_info}

for proveedor, config in carpetas_config.items():
    if not os.path.exists(config["ruta"]):
        continue
    
    for archivo in os.listdir(config["ruta"]):
        # Ignoramos los de copia directa en este paso porque no sirven para buscar partidos
        if archivo.endswith(".parquet") and archivo not in archivos_copia_directa:
            ruta_archivo = os.path.join(config["ruta"], archivo)
            try:
                df = pd.read_parquet(ruta_archivo)
                if config["col_partido"] in df.columns:
                    col_eq = buscar_columna_equipo(df.columns, config["claves_equipo"])
                    
                    if col_eq:
                        for equipo, ids in equipos_info.items():
                            id_buscar = ids["mc_id"] if proveedor == "mediacoach" else ids["opta_id"] if proveedor == "opta" else ids["sportian_name"]
                            # Guardamos todos los IDs de partidos donde aparece este equipo
                            partidos = df[df[col_eq] == id_buscar][config["col_partido"]].dropna().unique()
                            partidos_por_equipo[equipo][proveedor].update(partidos)
            except Exception as e:
                print(f"Error escaneando {archivo}: {e}")

# ==========================================
# 3. PASO 2: FILTRAR Y AÑADIR INCREMENTALMENTE
# ==========================================
print("\nPaso 2: Actualizando carpetas de equipos de forma incremental...")

for equipo in equipos_info.keys():
    print(f"\n--- Procesando: {equipo} ---")
    
    for proveedor, config in carpetas_config.items():
        if not os.path.exists(config["ruta"]):
            continue
            
        ruta_origen = config["ruta"]
        carpeta_proveedor = os.path.basename(ruta_origen) if os.path.basename(ruta_origen) != "data" else os.path.basename(os.path.dirname(ruta_origen)) + "/data"
        ruta_destino = os.path.join(carpeta_destino_base, equipo, carpeta_proveedor)
        
        os.makedirs(ruta_destino, exist_ok=True)
        partidos_validos_origen = partidos_por_equipo[equipo][proveedor]
        
        for archivo in os.listdir(ruta_origen):
            if not archivo.endswith(".parquet"):
                continue
                
            ruta_archivo_origen = os.path.join(ruta_origen, archivo)
            ruta_archivo_destino = os.path.join(ruta_destino, archivo)
            
            # NUEVO: Si es un archivo de copia directa, lo copiamos y pasamos al siguiente
            if archivo in archivos_copia_directa:
                shutil.copy2(ruta_archivo_origen, ruta_archivo_destino)
                # print(f"  [>] {archivo}: Copiado directamente.") # (Opcional, descomentar si quieres que te avise)
                continue
            
            try:
                df_origen = pd.read_parquet(ruta_archivo_origen)
                
                # Si es un archivo de eventos por partido
                if config["col_partido"] in df_origen.columns:
                    
                    # Filtramos el origen para tener solo los partidos que nos interesan
                    df_origen_equipo = df_origen[df_origen[config["col_partido"]].isin(partidos_validos_origen)]
                    
                    if os.path.exists(ruta_archivo_destino):
                        df_destino = pd.read_parquet(ruta_archivo_destino)
                        partidos_existentes = set(df_destino[config["col_partido"]].unique())
                        partidos_nuevos = set(df_origen_equipo[config["col_partido"]].unique()) - partidos_existentes
                        
                        if partidos_nuevos:
                            df_a_añadir = df_origen_equipo[df_origen_equipo[config["col_partido"]].isin(partidos_nuevos)]
                            df_final = pd.concat([df_destino, df_a_añadir], ignore_index=True)
                            df_final.to_parquet(ruta_archivo_destino, index=False)
                            print(f"  [+] {archivo}: Añadidos {len(partidos_nuevos)} partidos nuevos.")
                            
                    else:
                        if not df_origen_equipo.empty:
                            df_origen_equipo.to_parquet(ruta_archivo_destino, index=False)
                            print(f"  [C] {archivo}: Creado por primera vez con {len(df_origen_equipo[config['col_partido']].unique())} partidos.")
                        else:
                            df_origen.head(0).to_parquet(ruta_archivo_destino, index=False)
                
                # Si es un archivo GENERAL que NO está en la lista de copia directa, lo sobreescribimos
                else:
                    df_origen.to_parquet(ruta_archivo_destino, index=False)
                    
            except Exception as e:
                print(f"Error procesando {archivo} para {equipo}: {e}")

print("\n¡Proceso finalizado! Los archivos han sido actualizados incrementalmente y copiados tal cual.")