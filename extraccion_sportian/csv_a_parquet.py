import pandas as pd
import os
import numpy as np
import glob
import gc
import logging
import sys
from datetime import datetime

# ==========================================
# CONFIGURACIÓN DE LOGGING
# ==========================================

def setup_logging():
    """Configura logging dual: consola + archivo con timestamp."""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'sportian_etl_{timestamp}.log')

    # Formato detallado
    fmt = '%(asctime)s | %(levelname)-8s | %(message)s'

    # Configurar logger
    logger = logging.getLogger('sportian_etl')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Handler consola (INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    # Handler archivo (DEBUG - todo detalle)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    return logger, log_file

logger, log_file = setup_logging()

# ==========================================
# 1. FUNCIONES DE LÓGICA DE NEGOCIO (ETL)
# ==========================================

def clasificar_zona_vectorizada(df, y_balon_start):
    """
    Versión vectorizada de clasificar_zona_simple.
    Calcula la zona táctica para TODO un dataframe de una vez usando numpy.
    """
    x = df['X_Jugador'].values
    y = df['Y_Jugador'].values
    es_izq = y_balon_start > 50

    # Inicializar con valor por defecto
    zonas = np.full(len(df), 'OTRO', dtype=object)

    # VIGILANCIA: x < 70
    mask_vigilancia = x < 70
    zonas[mask_vigilancia] = 'VIGILANCIA'

    # CORTO: 75 <= x <= 100 Y (es_izq and y >= 81) or (not es_izq and y <= 25)
    mask_corto_base = (x >= 75) & (x <= 100)
    if es_izq:
        mask_corto = mask_corto_base & (y >= 81)
    else:
        mask_corto = mask_corto_base & (y <= 25)
    zonas[mask_corto] = 'CORTO'

    # Area Pequeña (x > 94.2)
    mask_ap = x > 94.2
    mask_ap_centro = mask_ap & (y >= 45) & (y <= 55)
    zonas[mask_ap_centro] = 'AP CENTRO'

    if es_izq:
        mask_ap_1er = mask_ap & (y > 55)
        mask_ap_2do = mask_ap & (y < 45)
    else:
        mask_ap_1er = mask_ap & (y < 45)
        mask_ap_2do = mask_ap & (y > 55)

    zonas[mask_ap_1er] = 'AP 1ER PALO'
    zonas[mask_ap_2do] = 'AP 2DO PALO'

    # Area Grande / Penalti (83 <= x <= 94.2)
    mask_ag = (x >= 83) & (x <= 94.2)
    mask_penalti = mask_ag & (y >= 45) & (y <= 55) & (x < 88)
    mask_centro_area = mask_ag & (y >= 45) & (y <= 55) & (x >= 88)
    zonas[mask_penalti] = 'PENALTI'
    zonas[mask_centro_area] = 'CENTRO AREA'

    # 1ER/2DO PALO en area grande
    if es_izq:
        mask_1er = mask_ag & (y > 55)
        mask_2do = mask_ag & (y < 45)
    else:
        mask_1er = mask_ag & (y < 45)
        mask_2do = mask_ag & (y > 55)

    zonas[mask_1er] = '1ER PALO'
    zonas[mask_2do] = '2DO PALO'

    # RECHACE: 70 <= x < 83
    mask_rechace = (x >= 70) & (x < 83)
    zonas[mask_rechace] = 'RECHACE'

    return zonas


def clasificar_zona_simple(x, y, y_balon_start):
    """Calcula la zona táctica basándose en coordenadas normalizadas (para compatibilidad)."""
    es_izq = y_balon_start > 50

    if x < 70: return "VIGILANCIA"
    if 75 <= x <= 100:
        if (es_izq and y >= 81) or (not es_izq and y <= 25): return "CORTO"

    if x > 94.2: # Area Pequeña
        if 45 <= y <= 55: return "AP CENTRO"
        return "AP 1ER PALO" if ((es_izq and y > 55) or (not es_izq and y < 45)) else "AP 2DO PALO"

    if 83 <= x <= 94.2: # Area Grande / Penalti
        if 45 <= y <= 55: return "PENALTI" if x < 88 else "CENTRO AREA"
        return "1ER PALO" if ((es_izq and y > 55) or (not es_izq and y < 45)) else "2DO PALO"

    if 70 <= x < 83: return "RECHACE"
    return "OTRO"


# ==========================================
# 2. FUNCIÓN DE ACTUALIZACIÓN INCREMENTAL
# ==========================================

def actualizar_parquets_especializados(df_maestro, id_col, csv_filename=None):
    """
    Compara el DF Maestro con los Parquets especializados existentes.
    Procesa SOLO los IDs nuevos y los añade (Append).
    OPTIMIZADO: Usa vectorización y libera memoria explícitamente.
    """
    logger.info("=" * 60)
    logger.info("🏭 VERIFICANDO ACTUALIZACIONES EN PARQUETS ESPECIALIZADOS")
    logger.info("=" * 60)

    # Nombres de archivo
    file_off = "corners_ofensivo_enrich.parquet"
    file_def = "corners_defensivo_snapshot.parquet"

    # --- 1. CARGAR EXISTENTES Y DETECTAR FALTANTES ---
    ids_procesados = set()
    df_old_off = pd.DataFrame()
    df_old_def = pd.DataFrame()

    # Intentamos leer el Ofensivo para ver qué IDs ya tenemos
    if os.path.exists(file_off):
        try:
            logger.info(f"   📂 Leyendo {file_off}...")
            df_old_off = pd.read_parquet(file_off)
            if id_col in df_old_off.columns:
                ids_procesados = set(df_old_off[id_col].unique())
            logger.info(f"      → {len(df_old_off)} filas ({len(ids_procesados)} eventos únicos)")
        except Exception as e:
            logger.warning(f"      ⚠️ Error leyendo {file_off}: {e}")

    # Identificar IDs nuevos en el maestro
    logger.info(f"   🔍 Buscando IDs nuevos en el maestro...")
    ids_maestro = set(df_maestro[id_col].unique())
    ids_nuevos = ids_maestro - ids_procesados

    if not ids_nuevos:
        logger.info("   ✅ Todos los datos están al día. No se requiere procesamiento ETL.")
        return

    logger.info(f"   ⚡ Detectados {len(ids_nuevos)} eventos nuevos. Procesando...")

    # --- 2. FILTRAR SOLO DATOS NUEVOS (liberar memoria pronto) ---
    logger.info(f"   📉 Filtrando dataframe para procesar solo {len(ids_nuevos)} eventos nuevos...")
    df_delta = df_maestro[df_maestro[id_col].isin(ids_nuevos)].copy()
    logger.info(f"      → DataFrame filtrado: {len(df_delta)} filas")

    # Forzar GC después de filtrar
    gc.collect()

    new_data_off = []
    new_data_def = []
    eventos_procesados = 0
    eventos_con_error = 0

    # Agrupar por evento
    logger.info(f"   🔄 Procesando eventos individualmente...")
    total_eventos = df_delta[id_col].nunique()

    for idx, (evento_id, df_ev) in enumerate(df_delta.groupby(id_col)):
        eventos_procesados += 1

        # Logging de progreso cada 100 eventos
        if eventos_procesados % 100 == 0:
            logger.info(f"      → {eventos_procesados}/{total_eventos} eventos procesados...")

        try:
            row_0 = df_ev.iloc[0]

            # Detectar lado (según Y balón al inicio)
            mask_saque = df_ev['Segundos_Desde_Saque'].between(-0.1, 0.5)
            y_saque = df_ev.loc[mask_saque, 'Y_Balon'].mean()
            if pd.isna(y_saque):
                y_saque = df_ev.iloc[0]['Y_Balon']

            # Detectar tipo lanzamiento
            tipo_raw = str(row_0.get('Tipo_Lanzamiento', ''))
            if 'Out-swinger' in tipo_raw:
                tipo_lanz = 'Abierto'
            elif 'In-swinger' in tipo_raw:
                tipo_lanz = 'Cerrado'
            else:
                tipo_lanz = 'Neutro'

            # --- PROCESO OFENSIVO (Secuencia) ---
            mask_off = df_ev['Segundos_Desde_Saque'].between(-0.5, 6.0)
            df_off_temp = df_ev.loc[mask_off].copy()

            if len(df_off_temp) > 0:
                # Enriquecemos - usando vectorización
                df_off_temp['Tipo_Lanzamiento_Calc'] = tipo_lanz
                df_off_temp['Zona_Precalc'] = clasificar_zona_vectorizada(df_off_temp, y_saque)
                new_data_off.append(df_off_temp)

            # --- PROCESO DEFENSIVO (Snapshot) ---
            # Buscamos frame más cercano a -3.0
            idx_def = (df_ev['Segundos_Desde_Saque'] - (-3.0)).abs().idxmin()
            t_def = df_ev.loc[idx_def, 'Segundos_Desde_Saque']

            if abs(t_def - (-3.0)) < 0.5:  # Si existe el frame
                df_snap = df_ev[df_ev['Segundos_Desde_Saque'] == t_def].copy()

                # Enriquecemos - usando vectorización
                df_snap['Zona_Precalc'] = clasificar_zona_vectorizada(df_snap, y_saque)
                new_data_def.append(df_snap)

        except Exception as e:
            eventos_con_error += 1
            logger.warning(f"      ⚠️ Error procesando evento {evento_id}: {e}")
            continue

    logger.info(f"      → Procesamiento completado: {eventos_procesados} eventos, {eventos_con_error} errores")

    # --- 3. CONCATENAR Y GUARDAR ---
    logger.info(f"   💾 Guardando archivos actualizados...")

    # Actualizar Ofensivo
    if new_data_off:
        logger.info(f"      → Concatenando {len(new_data_off)} bloques ofensivos...")
        df_new_off = pd.concat(new_data_off, ignore_index=True)
        logger.info(f"         → {len(df_new_off)} filas nuevas")

        if not df_old_off.empty:
            df_final_off = pd.concat([df_old_off, df_new_off], ignore_index=True)
        else:
            df_final_off = df_new_off

        df_final_off.to_parquet(file_off, index=False)
        logger.info(f"      ✅ Actualizado {file_off}: +{len(df_new_off)} filas (Total: {len(df_final_off)})")

        # Liberar memoria
        del df_new_off, df_final_off
        gc.collect()

    # Actualizar Defensivo
    if new_data_def:
        # Cargar defensivo existente si no se cargó antes
        if os.path.exists(file_def) and df_old_def.empty:
            try:
                logger.info(f"      → Leyendo {file_def}...")
                df_old_def = pd.read_parquet(file_def)
            except:
                pass

        logger.info(f"      → Concatenando {len(new_data_def)} bloques defensivos...")
        df_new_def = pd.concat(new_data_def, ignore_index=True)
        logger.info(f"         → {len(df_new_def)} filas nuevas")

        if not df_old_def.empty:
            df_final_def = pd.concat([df_old_def, df_new_def], ignore_index=True)
        else:
            df_final_def = df_new_def

        df_final_def.to_parquet(file_def, index=False)
        logger.info(f"      ✅ Actualizado {file_def}: +{len(df_new_def)} filas (Total: {len(df_final_def)})")

        # Liberar memoria
        del df_new_def, df_final_def
        gc.collect()

    # Limpieza final
    del df_delta
    gc.collect()

    logger.info(f"   🧹 Memoria liberada. Proceso especializado completado.")


# ==========================================
# 3. PROCESO PRINCIPAL (CSV -> MAESTRO)
# ==========================================

def procesar_dataset(csv_source, parquet_dest, id_col_name):
    """
    Procesa un CSV de Sportian y lo mergea incrementalmente al parquet maestro.
    OPTIMIZADO: Libera memoria explícitamente y loggea cada paso.
    """
    logger.info("=" * 60)
    logger.info(f"🔵 PROCESANDO: {csv_source}")
    logger.info(f"   📍 Destino: {parquet_dest}")
    logger.info(f"   📝 Columna ID: {id_col_name}")
    logger.info("=" * 60)

    df_new = None
    df_old = None
    df_combined = None

    try:
        # 1. CARGAR CSV NUEVO
        logger.info(f"   📄 Leyendo CSV fuente...")
        if os.path.exists(csv_source):
            df_new = pd.read_csv(csv_source)
            logger.info(f"      → CSV cargado: {len(df_new)} filas, {len(df_new.columns)} columnas")

            # Normalizar nombres de columnas
            if 'Segundos_Relativos_Al_Saque' in df_new.columns:
                df_new.rename(columns={'Segundos_Relativos_Al_Saque': 'Segundos_Desde_Saque'}, inplace=True)
                logger.info(f"      → Columna 'Segundos_Relativos_Al_Saque' renombrada")

            # Validar columna ID
            if id_col_name not in df_new.columns:
                possible_ids = [c for c in df_new.columns if 'ID_Evento' in c]
                if possible_ids:
                    id_col_name = possible_ids[0]
                    logger.info(f"      → ID alternativo detectado: {id_col_name}")
                else:
                    logger.error(f"      ❌ No se encontró columna ID en el CSV")
                    return

            # Log de memoria
            mem_mb = df_new.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(f"      → Memoria en uso: {mem_mb:.2f} MB")
        else:
            logger.error(f"      ❌ Archivo CSV no encontrado: {csv_source}")
            return

        # 2. CARGAR PARQUET MAESTRO EXISTENTE
        if os.path.exists(parquet_dest):
            logger.info(f"   📚 Leyendo parquet maestro existente...")
            df_old = pd.read_parquet(parquet_dest)
            logger.info(f"      → Maestro existente: {len(df_old)} filas")
            mem_mb = df_old.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(f"      → Memoria en uso: {mem_mb:.2f} MB")

        # 3. COMBINAR DATOS (Incremental)
        logger.info(f"   🔀 Combinando datos (modo incremental)...")
        hay_cambios = False

        if df_new is not None and df_old is not None:
            if id_col_name in df_old.columns:
                ids_old = set(df_old[id_col_name].unique())
                logger.info(f"      → {len(ids_old)} IDs existentes en maestro")

                df_new_filtered = df_new[~df_new[id_col_name].isin(ids_old)]
                logger.info(f"      → {len(df_new_filtered)} filas nuevas después de filtrar")

                if not df_new_filtered.empty:
                    logger.info(f"      ✨ Añadiendo {len(df_new_filtered)} filas nuevas al maestro")
                    df_combined = pd.concat([df_old, df_new_filtered], ignore_index=True)
                    hay_cambios = True
                else:
                    logger.info(f"      zzz Sin datos nuevos para el maestro")
                    df_combined = df_old
            else:
                logger.warning(f"      ⚠️ Columna ID no existe en maestro, concatenando todo")
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                hay_cambios = True

        elif df_new is not None and df_old is None:
            logger.info(f"      ✨ Creando archivo maestro nuevo (no existía previo)")
            df_combined = df_new
            hay_cambios = True

        elif df_new is None and df_old is not None:
            df_combined = df_old

        # Liberar df_new si ya no se necesita
        del df_new
        gc.collect()

        # 4. NORMALIZACIÓN (Solo si hubo cambios)
        cols_coords = ['X_Jugador', 'Y_Jugador', 'X_Balon', 'Y_Balon']
        if hay_cambios and all(col in df_combined.columns for col in cols_coords + ['Segundos_Desde_Saque', id_col_name]):
            logger.info(f"   🔄 Aplicando normalización de coordenadas...")

            # Identificar eventos a invertir (Saque en X<50 en T=0)
            saques = df_combined[df_combined['Segundos_Desde_Saque'] == 0.0]
            ids_invertir = saques.loc[saques['X_Balon'] < 50, id_col_name].unique()

            if len(ids_invertir) > 0:
                mask = df_combined[id_col_name].isin(ids_invertir)
                logger.info(f"      → {len(ids_invertir)} eventos requieren inversión de coordenadas")

                for col in cols_coords:
                    df_combined.loc[mask, col] = 100 - df_combined.loc[mask, col]
                logger.info(f"      ✅ Coordenadas invertidas para eventos del lado incorrecto")
            else:
                logger.info(f"      → No se requiere inversión (todos los eventos están orientados correctamente)")

        # 5. GUARDAR MAESTRO
        if hay_cambios:
            logger.info(f"   💾 Guardando maestro actualizado...")
            df_combined.to_parquet(parquet_dest, index=False)

            # Verificar archivo guardado
            if os.path.exists(parquet_dest):
                file_size = os.path.getsize(parquet_dest) / (1024 * 1024)
                logger.info(f"      ✅ MAESTRO GUARDADO: {parquet_dest} ({file_size:.2f} MB)")
            else:
                logger.error(f"      ❌ Error: archivo no se guardó correctamente")
        else:
            logger.info(f"      → Sin cambios, no se guarda maestro")

        # 6. BORRAR CSV
        if hay_cambios or (df_new is not None):
            try:
                os.remove(csv_source)
                logger.info(f"   🗑️ CSV temporal eliminado: {csv_source}")
            except Exception as e:
                logger.warning(f"      ⚠️ No se pudo eliminar CSV: {e}")

        # 7. LLAMADA CRÍTICA: ACTUALIZAR PARQUETS ESPECIALIZADOS
        if "corners" in parquet_dest:
            logger.info(f"   📣 Iniciando actualización de parquets especializados...")
            # Pasar solo df_combined (ya filtrado si era incremental)
            actualizar_parquets_especializados(df_combined, id_col_name, csv_source)
            logger.info(f"   ✅ Parquets especializados actualizados")
        else:
            logger.info(f"   ℹ️ Saltando parquets especializados (no es corners)")

        # Limpieza final
        del df_combined, df_old
        gc.collect()

        logger.info("=" * 60)
        logger.info("🏁 PROCESO COMPLETADO EXITOSAMENTE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ ERROR CRÍTICO: {e}", exc_info=True)
        raise
    finally:
        # Asegurar liberación de memoria
        df_new = None
        df_old = None
        df_combined = None
        gc.collect()
        logger.info(f"   🧹 Limpieza final completada")


def main():
    """Función principal para procesamiento por lotes."""
    logger.info("=" * 60)
    logger.info("🚀 INICIANDO SISTEMA ETL DE MANTENIMIENTO")
    logger.info(f"   📁 Directorio: {os.getcwd()}")
    logger.info(f"   📄 Log file: {log_file}")
    logger.info("=" * 60)

    csvs = glob.glob("*.csv")

    # --- CASO 1: HAY ARCHIVOS CSV NUEVOS ---
    if csvs:
        logger.info(f"   📂 Detectados {len(csvs)} archivos CSV para procesar")

        for csv_file in csvs:
            nombre_lower = csv_file.lower()

            if "corners" in nombre_lower:
                procesar_dataset(csv_file, "corners_tracking.parquet", "ID_Evento_Corner")
            elif "faltas" in nombre_lower:
                procesar_dataset(csv_file, "faltas_tracking.parquet", "ID_Evento_Falta")
            else:
                logger.warning(f"   ⚠️ Omitido (tipo desconocido): {csv_file}")

    # --- CASO 2: NO HAY CSVs (REVISIÓN DE MANTENIMIENTO) ---
    else:
        logger.info(f"   ⚠️ No hay archivos CSV nuevos en el directorio")

        # Comprobamos si existe el maestro de Corners para ver si le falta sincronizar algo
        maestro_corners = "corners_tracking.parquet"

        if os.path.exists(maestro_corners):
            logger.info(f"🔎 Revisando integridad entre maestro y especializados...")
            try:
                df_maestro = pd.read_parquet(maestro_corners)
                actualizar_parquets_especializados(df_maestro, "ID_Evento_Corner")
            except Exception as e:
                logger.error(f"❌ Error leyendo maestro para revisión: {e}", exc_info=True)
        else:
            logger.error(f"❌ No existe el archivo maestro de corners. No hay nada que procesar.")

    logger.info("=" * 60)
    logger.info("🏁 PROCESO COMPLETADO")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
