import pandas as pd
import os
import glob
from pathlib import Path
import logging
import shutil
import zipfile

# Configuración
base_data_path = "VCF_Mediacoach_Data"
output_base_path = "data"

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def identificar_archivos_postpartido(carpeta_partido):
    """
    Identifica TODOS los archivos CSV que empiezan por postpartido
    """
    archivos_postpartido = glob.glob(os.path.join(carpeta_partido, 'postpartido*.csv'))
    archivos_validos = []
    
    for archivo in archivos_postpartido:
        try:
            # Verificar que se puede leer el archivo
            df_test = pd.read_csv(archivo, sep=';', nrows=1)
            archivos_validos.append(archivo)
            logger.info(f"Archivo válido encontrado: {archivo}")
        except Exception as e:
            logger.warning(f"Error al leer {archivo}: {e}")
    
    return archivos_validos

def leer_csv_con_encoding(archivo):
    """
    Intenta leer el CSV con diferentes encodings
    """
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(archivo, sep=';', encoding=encoding)
            logger.info(f"Archivo {archivo} leído exitosamente con encoding {encoding}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error al leer {archivo} con encoding {encoding}: {e}")
            continue
    
    raise Exception(f"No se pudo leer el archivo {archivo} con ningún encoding")

def limpiar_columnas(df):
    """
    Limpia los nombres de las columnas eliminando espacios extra
    """
    # 🔧 CORREGIDO: Solo limpiar espacios, sin eliminar columnas
    df.columns = df.columns.str.strip()
    return df

def separar_datos_equipo_jugador(df, temporada, liga):
    """
    Separa los datos en estadísticas de equipo y jugador
    """
    # Identificar la columna de nombre del jugador (puede tener variaciones)
    columnas_jugador = [col for col in df.columns if 'NOMBRE' in col.upper() and 'JUGADOR' in col.upper()]
    if not columnas_jugador:
        # Buscar otras variaciones comunes
        columnas_jugador = [col for col in df.columns if any(keyword in col.upper() for keyword in ['PLAYER', 'NOMBRE', 'NAME'])]
    
    if not columnas_jugador:
        logger.warning("No se encontró columna de nombre del jugador - procesando todo como datos generales")
        # Procesar todo el DataFrame como datos (sin separar)
        df_limpio = limpiar_columnas(df.copy())
        df_limpio['temporada'] = temporada
        df_limpio['liga'] = liga
        return df_limpio, pd.DataFrame()  # Todo va a equipo, jugador vacío
    
    columna_jugador = columnas_jugador[0]
    logger.info(f"Usando columna de jugador: {columna_jugador}")
    
    # Datos de equipo: filas donde el nombre del jugador está vacío
    mask_equipo = df[columna_jugador].isna() | (df[columna_jugador].astype(str).str.strip() == '') | (df[columna_jugador].astype(str).str.strip() == 'nan')
    datos_equipo = df[mask_equipo].copy()
    
    # Eliminar columnas completamente vacías en datos de equipo
    datos_equipo = datos_equipo.dropna(how='all')
    
    # Datos de jugador: filas donde todas las columnas están completas
    datos_jugador = df[~mask_equipo].copy()
    datos_jugador = datos_jugador.dropna(how='all')
    
    # 🔧 CORREGIDO: Limpiar columnas DESPUÉS de procesar los datos
    if not datos_equipo.empty:
        datos_equipo = limpiar_columnas(datos_equipo)
        datos_equipo['temporada'] = temporada
        datos_equipo['liga'] = liga
    
    if not datos_jugador.empty:
        datos_jugador = limpiar_columnas(datos_jugador)
        datos_jugador['temporada'] = temporada
        datos_jugador['liga'] = liga
    
    logger.info(f"Filas de equipo: {len(datos_equipo)}, Filas de jugador: {len(datos_jugador)}")
    
    return datos_equipo, datos_jugador

def extraer_jornada_partido(nombre_carpeta):
    """
    Extrae jornada y partido del nombre de la carpeta
    Ejemplo: j1_athleticclub1-1getafecf -> jornada='j1', partido='athleticclub1-1getafecf'
    """
    try:
        logger.info(f"🔍 Analizando nombre de carpeta: '{nombre_carpeta}'")
        
        jornada = ''
        partido = ''
        
        # Buscar patrón j + número al inicio
        if nombre_carpeta.lower().startswith('j'):
            # Encontrar dónde termina la jornada (j + números)
            i = 1
            while i < len(nombre_carpeta) and nombre_carpeta[i].isdigit():
                i += 1
            
            jornada = nombre_carpeta[:i]
            
            # El resto es el partido (después del primer _ si existe)
            resto = nombre_carpeta[i:]
            if resto.startswith('_'):
                partido = resto[1:]  # Quitar el _
            else:
                partido = resto
        
        # Si no encontramos jornada, usar toda la carpeta como partido
        if not jornada:
            jornada = ''
            partido = nombre_carpeta
        
        # Limpiar partido
        if not partido:
            partido = nombre_carpeta
        
        logger.info(f"✅ Extraído - Jornada: '{jornada}', Partido: '{partido}'")
        return jornada, partido
        
    except Exception as e:
        logger.warning(f"Error al extraer jornada y partido de '{nombre_carpeta}': {e}")
        return '', nombre_carpeta

def limpiar_carpetas_partidos():
    """
    Comprime y elimina todas las carpetas con nombre 'Partidos' en VCF_Mediacoach_Data
    """
    import zipfile
    
    ruta_base = "VCF_Mediacoach_Data"
    
    if not os.path.exists(ruta_base):
        logger.info("No existe la carpeta VCF_Mediacoach_Data para limpiar")
        return
    
    carpetas_eliminadas = 0
    
    # Buscar recursivamente todas las carpetas llamadas "Partidos"
    carpetas_partidos_encontradas = []
    
    for root, dirs, files in os.walk(ruta_base):
        if "Partidos" in dirs:
            carpeta_partidos = os.path.join(root, "Partidos")
            carpetas_partidos_encontradas.append(carpeta_partidos)
    
    logger.info(f"🗂️ Encontradas {len(carpetas_partidos_encontradas)} carpetas 'Partidos' para comprimir")
    
    for i, carpeta_partidos in enumerate(carpetas_partidos_encontradas, 1):
        try:
            logger.info(f"📦 Comprimiendo carpeta {i}/{len(carpetas_partidos_encontradas)}: {carpeta_partidos}")
            
            # Crear el archivo zip en la carpeta PADRE (un nivel arriba)
            carpeta_padre = os.path.dirname(carpeta_partidos)
            nombre_liga = os.path.basename(carpeta_padre)
            ruta_zip = os.path.join(carpeta_padre, f"{nombre_liga}_Partidos_procesado.zip")
            
            # Contar archivos primero para mostrar progreso
            total_archivos = sum([len(files) for r, d, files in os.walk(carpeta_partidos)])
            logger.info(f"📄 Total archivos a comprimir: {total_archivos}")
            
            archivos_procesados = 0
            
            # Comprimir la carpeta
            with zipfile.ZipFile(ruta_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root_zip, dirs_zip, files_zip in os.walk(carpeta_partidos):
                    for file in files_zip:
                        file_path = os.path.join(root_zip, file)
                        # Mantener la estructura de carpetas en el zip
                        arcname = os.path.relpath(file_path, os.path.dirname(carpeta_partidos))
                        zipf.write(file_path, arcname)
                        
                        archivos_procesados += 1
                        if archivos_procesados % 100 == 0:  # Mostrar progreso cada 100 archivos
                            logger.info(f"📊 Progreso: {archivos_procesados}/{total_archivos} archivos")
            
            # Verificar que el zip se creó correctamente
            if os.path.exists(ruta_zip) and os.path.getsize(ruta_zip) > 0:
                tamaño_zip = os.path.getsize(ruta_zip) / (1024*1024)  # MB
                logger.info(f"✅ ZIP creado exitosamente: {ruta_zip} ({tamaño_zip:.1f} MB)")
                
                # Solo entonces borrar la carpeta original
                shutil.rmtree(carpeta_partidos)
                logger.info(f"🗑️ Carpeta original eliminada: {carpeta_partidos}")
                carpetas_eliminadas += 1
            else:
                logger.error(f"❌ Error: no se pudo crear el zip {ruta_zip}")
                
        except Exception as e:
            logger.error(f"❌ Error al comprimir/eliminar {carpeta_partidos}: {e}")
            # Continuar con la siguiente carpeta
            continue
    
    logger.info(f"🎉 Limpieza completada. {carpetas_eliminadas} carpetas 'Partidos' comprimidas y eliminadas")

def cargar_datos_existentes(archivo_parquet):
    """
    Carga datos existentes desde archivo parquet si existe
    """
    if os.path.exists(archivo_parquet):
        try:
            return pd.read_parquet(archivo_parquet)
        except Exception as e:
            logger.warning(f"Error al leer {archivo_parquet}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def combinar_y_deduplicar(df_nuevo, df_existente):
    """
    Combina datos nuevos con existentes y elimina duplicados
    """
    if df_existente.empty:
        logger.info("No hay datos existentes, usando solo datos nuevos")
        return df_nuevo
    
    if df_nuevo.empty:
        logger.info("No hay datos nuevos, manteniendo datos existentes")
        return df_existente
    
    # 🔧 LOGS DE DEBUG
    logger.info(f"🐛 DEBUG: Datos existentes: {len(df_existente)} filas")
    logger.info(f"🐛 DEBUG: Datos nuevos: {len(df_nuevo)} filas")
    logger.info(f"🐛 DEBUG: Columnas existentes: {len(df_existente.columns)}")
    logger.info(f"🐛 DEBUG: Columnas nuevas: {len(df_nuevo.columns)}")
    
    # 🔧 UNIR TODAS LAS COLUMNAS (no filtrar)
    todas_columnas = list(set(df_existente.columns) | set(df_nuevo.columns))
    
    # Asegurar que ambos DataFrames tengan todas las columnas
    for col in todas_columnas:
        if col not in df_existente.columns:
            df_existente[col] = None
        if col not in df_nuevo.columns:
            df_nuevo[col] = None
    
    # Reordenar columnas para que coincidan
    df_existente = df_existente[todas_columnas]
    df_nuevo = df_nuevo[todas_columnas]
    
    # 🔧 COMBINAR SIN FILTRAR
    df_combinado = pd.concat([df_existente, df_nuevo], ignore_index=True)
    logger.info(f"🐛 DEBUG: Después de combinar: {len(df_combinado)} filas")
    
    # 🔧 DEDUPLICACIÓN INTELIGENTE - Solo columnas clave para identificar duplicados
    columnas_clave = []
    
    # Buscar columnas que realmente identifican registros únicos
    posibles_claves = ['ID EQUIPO', 'ID PARTIDO', 'jornada', 'partido', 'temporada', 'liga', 'ID MÉTRICA', 'PERIODO']
    for col in posibles_claves:
        if col in df_combinado.columns:
            columnas_clave.append(col)
    
    # Si hay columna de jugador, añadirla también
    for col in df_combinado.columns:
        if 'NOMBRE' in col.upper() and 'JUGADOR' in col.upper():
            columnas_clave.append(col)
            break
    
    if columnas_clave:
        logger.info(f"🔑 Usando columnas clave para deduplicar: {columnas_clave}")
        filas_antes = len(df_combinado)
        df_sin_duplicados = df_combinado.drop_duplicates(subset=columnas_clave, keep='first')
        duplicados_eliminados = filas_antes - len(df_sin_duplicados)
        logger.info(f"🚫 Duplicados eliminados: {duplicados_eliminados}")
    else:
        logger.warning("⚠️ No se encontraron columnas clave, usando todas las columnas")
        df_sin_duplicados = df_combinado.drop_duplicates()
    
    # 🔧 CÁLCULO CORRECTO de nuevas filas
    nuevas_filas = len(df_sin_duplicados) - len(df_existente)
    logger.info(f"✅ Filas nuevas añadidas: {nuevas_filas}")
    logger.info(f"✅ Total final: {len(df_sin_duplicados)} filas")
    
    return df_sin_duplicados

def procesar_datos_vcf():
    """
    Función principal para procesar todos los datos
    """
    base_data_path = "VCF_Mediacoach_Data"
    
    if not os.path.exists(base_data_path):
        logger.error(f"La ruta {base_data_path} no existe")
        return
    
    # Crear carpeta data si no existe
    os.makedirs("data", exist_ok=True)
    logger.info("Carpeta 'data' verificada/creada")
    
    # Cargar datos existentes
    datos_equipo_existentes = cargar_datos_existentes("data/estadisticas_equipo.parquet")
    datos_jugador_existentes = cargar_datos_existentes("data/estadisticas_jugador.parquet")
    
    # Inicializar DataFrames para acumular datos
    todos_datos_equipo = pd.DataFrame()
    todos_datos_jugador = pd.DataFrame()
    
    partidos_procesados = 0
    total_carpetas_procesadas = 0
    total_archivos_exitosos = 0
    total_errores = 0
    
    # Recorrer todas las temporadas
    for temporada in os.listdir(base_data_path):
        temporada_path = os.path.join(base_data_path, temporada)
        
        if not os.path.isdir(temporada_path) or not (temporada.startswith('Temporada_') or temporada.startswith('Season_')):
            continue
            
        logger.info(f"=== PROCESANDO TEMPORADA: {temporada} ===")
        
        # Recorrer todas las ligas
        for liga in os.listdir(temporada_path):
            liga_path = os.path.join(temporada_path, liga)
            
            if not os.path.isdir(liga_path):
                continue
                
            partidos_path = os.path.join(liga_path, 'Partidos')
            
            if not os.path.exists(partidos_path):
                continue
                
            logger.info(f"--- Procesando liga: {liga} ---")
            
            # Buscar todas las carpetas de partidos
            carpetas_partidos = [d for d in os.listdir(partidos_path) if os.path.isdir(os.path.join(partidos_path, d))]
            logger.info(f"Encontradas {len(carpetas_partidos)} carpetas de partidos")
            
            carpetas_procesadas = 0
            archivos_exitosos = 0
            errores = 0
            
            for carpeta in carpetas_partidos:
                ruta_carpeta = os.path.join(partidos_path, carpeta)
                logger.info(f"Procesando carpeta: {carpeta}")
                
                # Extraer jornada y partido del nombre de la carpeta
                jornada, partido = extraer_jornada_partido(carpeta)
                
                # Identificar archivos válidos
                archivos_validos = identificar_archivos_postpartido(ruta_carpeta)
                
                if len(archivos_validos) == 0:
                    logger.warning(f"No se encontraron archivos válidos en {carpeta}")
                    errores += 1
                    continue
                
                if len(archivos_validos) < 2:
                    logger.warning(f"Solo se encontraron {len(archivos_validos)} archivos válidos en {carpeta}")
                
                carpetas_procesadas += 1
                archivos_en_esta_carpeta = 0
                
                # Procesar cada archivo válido
                for archivo in archivos_validos:
                    try:
                        # Leer CSV
                        df = leer_csv_con_encoding(archivo)
                        
                        # 🔧 CORREGIDO: Limpiar columnas ANTES de separar datos
                        df = limpiar_columnas(df)
                        
                        # Separar datos de equipo y jugador
                        datos_equipo, datos_jugador = separar_datos_equipo_jugador(df, temporada, liga)
                        
                        # Añadir columnas de jornada y partido
                        if not datos_equipo.empty:
                            datos_equipo['jornada'] = jornada
                            datos_equipo['partido'] = partido
                            
                            if todos_datos_equipo.empty:
                                todos_datos_equipo = datos_equipo.copy()
                            else:
                                todos_datos_equipo = pd.concat([todos_datos_equipo, datos_equipo], ignore_index=True)
                        
                        if not datos_jugador.empty:
                            datos_jugador['jornada'] = jornada
                            datos_jugador['partido'] = partido
                            
                            if todos_datos_jugador.empty:
                                todos_datos_jugador = datos_jugador.copy()
                            else:
                                todos_datos_jugador = pd.concat([todos_datos_jugador, datos_jugador], ignore_index=True)
                        
                        archivos_exitosos += 1
                        archivos_en_esta_carpeta += 1
                        logger.info(f"Archivo procesado exitosamente: {os.path.basename(archivo)}")
                        
                    except Exception as e:
                        errores += 1
                        logger.error(f"Error al procesar {archivo}: {e}")
                
                logger.info(f"Archivos procesados en esta carpeta: {archivos_en_esta_carpeta}/{len(archivos_validos)}")
            
            # Resumen por liga
            logger.info(f"📊 Resumen {liga}:")
            logger.info(f"  📁 Carpetas procesadas: {carpetas_procesadas}")
            logger.info(f"  ✅ Archivos procesados exitosamente: {archivos_exitosos}")
            logger.info(f"  ❌ Archivos con errores: {errores}")
            
            # Acumular totales globales
            total_carpetas_procesadas += carpetas_procesadas
            total_archivos_exitosos += archivos_exitosos
            total_errores += errores
    
    # RESUMEN GLOBAL
    logger.info(f"🎯 RESUMEN GLOBAL DEL PROCESAMIENTO:")
    logger.info(f"  📁 Total carpetas procesadas: {total_carpetas_procesadas}")
    logger.info(f"  ✅ Total archivos procesados exitosamente: {total_archivos_exitosos}")
    logger.info(f"  ❌ Total archivos con errores: {total_errores}")
    logger.info(f"  📊 Filas equipo obtenidas: {len(todos_datos_equipo)}")
    logger.info(f"  👥 Filas jugador obtenidas: {len(todos_datos_jugador)}")
    logger.info(f"  📚 Filas equipo existentes: {len(datos_equipo_existentes)}")
    logger.info(f"  📚 Filas jugador existentes: {len(datos_jugador_existentes)}")
    
    # Mostrar muestras de datos para verificación
    if not todos_datos_equipo.empty:
        logger.info(f"📋 Muestra columnas datos_equipo: {list(todos_datos_equipo.columns)}")
        logger.info(f"📋 Muestra jornadas datos_equipo: {todos_datos_equipo['jornada'].unique()[:5]}")
    
    if not todos_datos_jugador.empty:
        logger.info(f"📋 Muestra columnas datos_jugador: {list(todos_datos_jugador.columns)}")
        logger.info(f"📋 Muestra jornadas datos_jugador: {todos_datos_jugador['jornada'].unique()[:5]}")
    
    # Combinar con datos existentes y eliminar duplicados
    logger.info("Combinando con datos existentes y eliminando duplicados...")
    
    datos_equipo_finales = combinar_y_deduplicar(todos_datos_equipo, datos_equipo_existentes)
    datos_jugador_finales = combinar_y_deduplicar(todos_datos_jugador, datos_jugador_existentes)
    
    # Guardar en archivos parquet
    try:
        archivos_guardados = 0
        
        if not datos_equipo_finales.empty:
            datos_equipo_finales.to_parquet("data/estadisticas_equipo.parquet", index=False)
            logger.info(f"✅ Guardado data/estadisticas_equipo.parquet con {len(datos_equipo_finales)} filas")
            archivos_guardados += 1
        else:
            logger.warning("⚠️ No hay datos de equipo para guardar")
        
        if not datos_jugador_finales.empty:
            datos_jugador_finales.to_parquet("data/estadisticas_jugador.parquet", index=False)
            logger.info(f"✅ Guardado data/estadisticas_jugador.parquet con {len(datos_jugador_finales)} filas")
            archivos_guardados += 1
        else:
            logger.warning("⚠️ No hay datos de jugador para guardar")
        
        logger.info(f"🎉 Procesamiento completado exitosamente!")
        logger.info(f"📊 Estadísticas finales:")
        logger.info(f"  📁 Carpetas procesadas: {total_carpetas_procesadas}")
        logger.info(f"  📄 Archivos CSV procesados: {total_archivos_exitosos}")
        logger.info(f"  💾 Archivos parquet guardados: {archivos_guardados}")
        logger.info(f"  📊 Total filas equipo: {len(datos_equipo_finales)}")
        logger.info(f"  👥 Total filas jugador: {len(datos_jugador_finales)}")
        
        # Limpiar carpetas Partidos después del procesamiento exitoso
        # if archivos_guardados > 0:
        #     logger.info("Iniciando limpieza de carpetas 'Partidos'...")
        #     limpiar_carpetas_partidos()
        # else:
        #     logger.warning("No se guardaron archivos, saltando limpieza de carpetas")
        
    except Exception as e:
        logger.error(f"❌ Error al guardar archivos parquet: {e}")
        raise
    

if __name__ == "__main__":
    try:
        procesar_datos_vcf()
        print("✅ Procesamiento completado exitosamente")
    except Exception as e:
        logger.error(f"Error en el procesamiento: {e}")
        print("❌ Error en el procesamiento. Consulta los logs para más detalles.")