import pandas as pd
import os
import shutil
import argparse
import sys

# ==========================================
# 0. ARGUMENTOS DE LÍNEA DE COMANDOS
# ==========================================
parser = argparse.ArgumentParser(description='Actualizar carpetas de datos por equipo')
parser.add_argument(
    '--delta_path', default=None,
    help='Carpeta temporal con solo los datos nuevos de esta sesión (modo delta rápido). '
         'Estructura: delta/opta/, delta/mediacoach/, delta/sportian/'
)
args = parser.parse_args()

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

# carpeta_destino: subcarpeta real que se usa dentro de data_por_equipos/<equipo>/
# Se separa de "ruta" para que en modo delta la ruta cambie pero la carpeta destino no.
carpetas_config = {
    "mediacoach": {
        "ruta": "extraccion_mediacoach/data",
        "col_partido": "ID PARTIDO",
        "claves_equipo": ["ID EQUIPO", "id_equipo"],
        "carpeta_destino": "extraccion_mediacoach/data"
    },
    "opta": {
        "ruta": "extraccion_opta/datos_opta_parquet",
        "col_partido": "Match ID",
        "claves_equipo": ["Team ID", "team_id"],
        "carpeta_destino": "datos_opta_parquet"
    },
    "sportian": {
        "ruta": "extraccion_sportian",
        "col_partido": "ID_Partido",
        "claves_equipo": ["NombreEquipoJugador", "equipo"],
        "carpeta_destino": "extraccion_sportian"
    }
}

# Archivos que se copian enteros sin filtrar por equipo
archivos_copia_directa = ['estadisticas_abp.parquet', 'estadisticas_abp_liga.parquet']
carpeta_destino_base = "data_por_equipos"

def buscar_columna_equipo(columnas, claves):
    """Busca dinámicamente la columna del equipo en el parquet"""
    for col in columnas:
        for clave in claves:
            if clave.lower() in col.lower():
                return col
    return None

# ==========================================
# 2. MODO DELTA vs. MODO COMPLETO
# En modo delta, se procesan solo las subcarpetas de args.delta_path
# que contengan archivos .parquet nuevos, en lugar de las rutas completas.
# ==========================================
DELTA_MODE = args.delta_path is not None

if DELTA_MODE:
    print(f"⚡ MODO DELTA: procesando solo datos nuevos desde '{args.delta_path}'")
    carpetas_config_efectivas = {}
    for proveedor, config in carpetas_config.items():
        delta_ruta = os.path.join(args.delta_path, proveedor)
        if os.path.exists(delta_ruta) and any(
            f.endswith('.parquet') for f in os.listdir(delta_ruta)
        ):
            carpetas_config_efectivas[proveedor] = {**config, "ruta": delta_ruta}

    if not carpetas_config_efectivas:
        print("⚠️  Delta vacío: no hay archivos .parquet nuevos. Saliendo.")
        sys.exit(0)
    proveedores_activos = list(carpetas_config_efectivas.keys())
    print(f"   Proveedores con datos nuevos: {', '.join(proveedores_activos)}")
else:
    carpetas_config_efectivas = carpetas_config

# ==========================================
# 3. PASO 1: DESCUBRIR PARTIDOS EN LOS DATOS ORIGEN
# ==========================================
print("Paso 1: Identificando los partidos de cada equipo en los datos de origen...")
partidos_por_equipo = {eq: {'mediacoach': set(), 'opta': set(), 'sportian': set()} for eq in equipos_info}

for proveedor, config in carpetas_config_efectivas.items():
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
                            id_buscar = (
                                ids["mc_id"] if proveedor == "mediacoach"
                                else ids["opta_id"] if proveedor == "opta"
                                else ids["sportian_name"]
                            )
                            partidos = df[df[col_eq] == id_buscar][config["col_partido"]].dropna().unique()
                            partidos_por_equipo[equipo][proveedor].update(partidos)
            except Exception as e:
                print(f"Error escaneando {archivo}: {e}")

# ==========================================
# 4. PASO 2: FILTRAR Y AÑADIR INCREMENTALMENTE
# Optimización: bucle por proveedor→archivo→equipo para leer cada archivo UNA SOLA VEZ.
# En modo delta los archivos origen son pequeños (solo lo nuevo), por lo que es muy rápido.
# ==========================================
print("\nPaso 2: Actualizando carpetas de equipos de forma incremental...")

for proveedor, config in carpetas_config_efectivas.items():
    if not os.path.exists(config["ruta"]):
        continue

    ruta_origen = config["ruta"]
    # Usar carpeta_destino fija del config para que el modo delta no cambie la estructura
    carpeta_proveedor = config["carpeta_destino"]

    # Pre-crear todas las carpetas destino de este proveedor
    for equipo in equipos_info.keys():
        os.makedirs(os.path.join(carpeta_destino_base, equipo, carpeta_proveedor), exist_ok=True)

    for archivo in os.listdir(ruta_origen):
        if not archivo.endswith(".parquet"):
            continue

        ruta_archivo_origen = os.path.join(ruta_origen, archivo)

        # Archivos de copia directa: se copian tal cual solo en modo completo.
        # En modo delta estos archivos no existen (son generados, no descargados).
        if archivo in archivos_copia_directa:
            if not DELTA_MODE:
                for equipo in equipos_info.keys():
                    ruta_destino = os.path.join(carpeta_destino_base, equipo, carpeta_proveedor)
                    shutil.copy2(ruta_archivo_origen, os.path.join(ruta_destino, archivo))
            continue

        try:
            # LECTURA ÚNICA: leemos el archivo origen una sola vez y lo distribuimos a los 20 equipos
            df_origen = pd.read_parquet(ruta_archivo_origen)
            tiene_col_partido = config["col_partido"] in df_origen.columns

            for equipo in equipos_info.keys():
                ruta_destino = os.path.join(carpeta_destino_base, equipo, carpeta_proveedor)
                ruta_archivo_destino = os.path.join(ruta_destino, archivo)

                if tiene_col_partido:
                    partidos_validos = partidos_por_equipo[equipo][proveedor]
                    df_origen_equipo = df_origen[df_origen[config["col_partido"]].isin(partidos_validos)]

                    if os.path.exists(ruta_archivo_destino):
                        df_destino = pd.read_parquet(ruta_archivo_destino)
                        partidos_existentes = set(df_destino[config["col_partido"]].unique())
                        partidos_nuevos = set(df_origen_equipo[config["col_partido"]].unique()) - partidos_existentes

                        if partidos_nuevos:
                            df_a_añadir = df_origen_equipo[df_origen_equipo[config["col_partido"]].isin(partidos_nuevos)]
                            df_final = pd.concat([df_destino, df_a_añadir], ignore_index=True)
                            df_final.to_parquet(ruta_archivo_destino, index=False)
                            print(f"  [+] {equipo} / {archivo}: Añadidos {len(partidos_nuevos)} partidos nuevos.")

                    else:
                        if not df_origen_equipo.empty:
                            df_origen_equipo.to_parquet(ruta_archivo_destino, index=False)
                            print(f"  [C] {equipo} / {archivo}: Creado por primera vez con {len(df_origen_equipo[config['col_partido']].unique())} partidos.")
                        else:
                            df_origen.head(0).to_parquet(ruta_archivo_destino, index=False)

                else:
                    # Archivo general sin columna de partido: sobreescribir para cada equipo
                    df_origen.to_parquet(ruta_archivo_destino, index=False)

        except Exception as e:
            print(f"Error procesando {archivo}: {e}")

modo_str = "delta (solo datos nuevos)" if DELTA_MODE else "completo"
print(f"\n¡Proceso finalizado en modo {modo_str}! Las carpetas de equipos han sido actualizadas incrementalmente.")
