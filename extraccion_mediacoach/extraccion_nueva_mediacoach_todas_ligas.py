import requests
import subprocess
import json
import zipfile
import csv
import os
import io
import mimetypes
from datetime import datetime
from collections import defaultdict
import logging
import re
import pandas as pd
import sys


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mediacoach_download_por_partido.log'),
        logging.StreamHandler()
    ]
)

def detectar_tipo_archivo(content, content_type=None):
    """Detecta el tipo de archivo basándose en magic numbers y content-type"""
    if not content:
        return 'unknown'
    
    # Magic numbers para detectar tipos de archivo
    if content.startswith(b'%PDF'):
        return 'pdf'
    elif content.startswith(b'PK\x03\x04'):  # ZIP-based files (XLSX, DOCX, etc.)
        try:
            with zipfile.ZipFile(io.BytesIO(content), 'r') as zip_file:
                if 'xl/workbook.xml' in zip_file.namelist():
                    return 'xlsx'
                elif 'word/document.xml' in zip_file.namelist():
                    return 'docx'
        except:
            pass
        return 'zip'
    elif content.startswith(b'<?xml') or content.startswith(b'<'):
        return 'xml'
    else:
        # Intentar detectar CSV por contenido
        try:
            content_str = content.decode('utf-8', errors='ignore')[:1000]
            if (';' in content_str or ',' in content_str) and '\n' in content_str:
                lines = content_str.split('\n')[:5]
                if len(lines) > 1:
                    sep_counts = []
                    for line in lines:
                        if line.strip():
                            sep_counts.append(line.count(';') + line.count(','))
                    if len(set(sep_counts)) <= 2 and max(sep_counts) > 0:
                        return 'csv'
        except:
            pass
    
    return 'bin'

def analizar_contenido_xml(content):
    """Analiza el contenido del XML para categorizarlo"""
    try:
        content_str = content.decode('utf-8', errors='ignore')[:2000]
        
        # Buscar patrones específicos
        if 'ALL_INSTANCES' in content_str and 'instance' in content_str:
            return 'eventos_partido'
        elif 'IdGame' in content_str and 'start' in content_str and 'end' in content_str:
            return 'eventos_partido'
        elif 'beyond' in content_str.lower() or 'stats' in content_str.lower():
            return 'beyond_stats'
        elif 'maxima' in content_str.lower() or 'exigencia' in content_str.lower():
            return 'maxima_exigencia'
        else:
            return 'general'
    except:
        return 'general'

def analizar_contenido_csv(content):
    """Analiza el contenido del CSV para categorizarlo"""
    try:
        content_str = content.decode('utf-8', errors='ignore')[:1000]
        
        # Buscar patrones en headers
        if 'equipo' in content_str.lower() or 'team' in content_str.lower():
            return 'equipos'
        elif 'jugador' in content_str.lower() or 'player' in content_str.lower():
            return 'jugadores'
        else:
            # Analizar estructura para determinar tipo
            lines = content_str.split('\n')[:3]
            if len(lines) > 0:
                header = lines[0].lower()
                if any(word in header for word in ['nombre', 'name', 'player', 'jugador']):
                    return 'jugadores'
                elif any(word in header for word in ['equipo', 'team', 'club']):
                    return 'equipos'
            return 'general'
    except:
        return 'general'

def analizar_contenido_xlsx(content, posicion_original=None):
    """Analiza el contenido del XLSX para categorizarlo"""
    try:
        # Usar posición original como hint si está disponible
        if posicion_original is not None:
            if posicion_original in [0, 1]:  # Posiciones típicas de rendimiento
                return 'rendimiento'
            elif posicion_original in [11, 12]:  # Posiciones típicas de máxima exigencia
                return 'maxima_exigencia'
        
        # Si no podemos determinar por posición, usar análisis básico
        return 'general'
    except:
        return 'general'

def descargar_y_categorizar_archivo(file_url, match_id, posicion, timeout=30):
    """Descarga un archivo y lo categoriza según su contenido"""
    try:
        logging.info(f"Descargando archivo {posicion}: {file_url}")
        response = requests.get(file_url, timeout=timeout)
        
        if response.status_code == 200:
            content = response.content
            content_type = response.headers.get('Content-Type', '')
            tipo_archivo = detectar_tipo_archivo(content, content_type)
            
            # Categorizar según tipo y contenido
            categoria = 'general'
            if tipo_archivo == 'xml':
                categoria = analizar_contenido_xml(content)
            elif tipo_archivo == 'csv':
                categoria = analizar_contenido_csv(content)
            elif tipo_archivo == 'xlsx':
                categoria = analizar_contenido_xlsx(content, posicion)
            elif tipo_archivo == 'pdf':
                categoria = 'informe'  # Todos los PDFs los consideramos informes
            
            return {
                'success': True,
                'content': content,
                'tipo': tipo_archivo,
                'categoria': categoria,
                'size': len(content),
                'content_type': content_type,
                'posicion': posicion
            }
        else:
            logging.error(f"Error HTTP {response.status_code} para archivo en posición {posicion}")
            return {'success': False, 'error': f'HTTP {response.status_code}', 'posicion': posicion}
            
    except Exception as e:
        logging.error(f"Error descargando archivo en posición {posicion}: {e}")
        return {'success': False, 'error': str(e), 'posicion': posicion}

def procesar_partido(match_id, asset_data, temporada, competicion, ejecutar_curl_comando):
    """Procesa un partido completo y descarga todos sus archivos organizadamente"""
    
    logging.info(f"\n=== PROCESANDO PARTIDO {match_id} ===")
    
    # DEBUG: Ver qué llega en asset_data
    logging.info(f"DEBUG: asset_data tipo: {type(asset_data)}")
    logging.info(f"DEBUG: asset_data contenido completo: {asset_data}")
    
    if not asset_data or len(asset_data) == 0:
        logging.error(f"No hay asset_data para el partido {match_id}")
        return False
    
    # ✅ ESTRUCTURA CORREGIDA: Temporada/Liga/Partidos/Partido_ID
    carpeta_partido = f'./VCF_Mediacoach_Data/{temporada}/{competicion}/Partidos/Partido_{match_id}'
    os.makedirs(carpeta_partido, exist_ok=True)
    
    # Obtener información del partido si está disponible
    try:
        match_info = asset_data[0].get('metadata', {}) if len(asset_data) > 0 else {}
        jornada = match_info.get('matchDay', 'desconocida')
    except:
        jornada = 'desconocida'
    
    # Estructuras para organizar archivos por tipo
    archivos_descargados = {
        'xml_eventos': [],
        'pdf_informes': [],
        'xlsx_rendimiento': [],
        'xlsx_maxima_exigencia': [],
        'csv_equipos': [],
        'csv_jugadores': [],
        'otros': []
    }
    
    errores = []
    total_descargados = 0
    
    # Descargar todos los archivos disponibles
    for i, asset in enumerate(asset_data):
        if 'url' in asset and asset['url']:
            # ✅ SALTAR VIDEOS MP4 (son muy pesados, 4-10GB cada uno)
            friendly_name = asset.get('friendlyName', '')
            extension = asset.get('qualifiers', {}).get('fileExtension', '')
            
            if extension == 'mp4' or friendly_name.lower().endswith('.mp4'):
                logging.info(f"⏭️ Saltando video MP4: {friendly_name}")
                continue  # Salta al siguiente archivo
            
            resultado = descargar_y_categorizar_archivo(asset['url'], match_id, i)
            
            if resultado['success']:
                # Categorizar el archivo según su tipo y contenido
                tipo = resultado['tipo']
                categoria = resultado['categoria']
                content = resultado['content']
                size = resultado['size']
                
                # Crear nombre de archivo descriptivo
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if tipo == 'xml' and categoria == 'eventos_partido':
                    nombre_archivo = f'eventos_partido_j{jornada}_{match_id}_{timestamp}.xml'
                    archivos_descargados['xml_eventos'].append(nombre_archivo)
                    
                elif tipo == 'pdf':
                    contador_pdf = len(archivos_descargados['pdf_informes']) + 1
                    nombre_archivo = f'informe_{contador_pdf}_j{jornada}_{match_id}_{timestamp}.pdf'
                    archivos_descargados['pdf_informes'].append(nombre_archivo)
                    
                elif tipo == 'xlsx' and categoria == 'rendimiento':
                    contador_xlsx = len(archivos_descargados['xlsx_rendimiento']) + 1
                    nombre_archivo = f'rendimiento_{contador_xlsx}_j{jornada}_{match_id}_{timestamp}.xlsx'
                    archivos_descargados['xlsx_rendimiento'].append(nombre_archivo)
                    
                elif tipo == 'xlsx' and categoria == 'maxima_exigencia':
                    contador_xlsx = len(archivos_descargados['xlsx_maxima_exigencia']) + 1
                    nombre_archivo = f'maxima_exigencia_{contador_xlsx}_j{jornada}_{match_id}_{timestamp}.xlsx'
                    archivos_descargados['xlsx_maxima_exigencia'].append(nombre_archivo)
                    
                elif tipo == 'csv' and categoria == 'equipos':
                    nombre_archivo = f'postpartido_equipos_j{jornada}_{match_id}_{timestamp}.csv'
                    archivos_descargados['csv_equipos'].append(nombre_archivo)
                    
                elif tipo == 'csv' and categoria == 'jugadores':
                    nombre_archivo = f'postpartido_jugadores_j{jornada}_{match_id}_{timestamp}.csv'
                    archivos_descargados['csv_jugadores'].append(nombre_archivo)
                    
                else:
                    # Archivos que no encajan en las categorías principales
                    nombre_archivo = f'otro_{tipo}_{categoria}_pos{i}_j{jornada}_{match_id}_{timestamp}.{tipo}'
                    archivos_descargados['otros'].append(nombre_archivo)
                
                # Guardar el archivo
                ruta_completa = os.path.join(carpeta_partido, nombre_archivo)
                try:
                    with open(ruta_completa, 'wb') as f:
                        f.write(content)
                    
                    total_descargados += 1
                    logging.info(f"  ✅ {nombre_archivo} ({size} bytes) - {tipo}/{categoria}")
                    
                except Exception as e:
                    errores.append(f"Error guardando {nombre_archivo}: {e}")
                    logging.error(f"  ❌ Error guardando {nombre_archivo}: {e}")
            else:
                errores.append(f"Posición {i}: {resultado['error']}")
                logging.error(f"  ❌ Posición {i}: {resultado['error']}")
    
    # Crear resumen del partido
    resumen = {
        'match_id': match_id,
        'jornada': jornada,
        'total_archivos_descargados': total_descargados,
        'archivos_por_tipo': {k: len(v) for k, v in archivos_descargados.items()},
        'archivos_descargados': archivos_descargados,
        'errores': errores,
        'timestamp': datetime.now().isoformat()
    }
    
    # Guardar resumen en JSON
    resumen_path = os.path.join(carpeta_partido, f'resumen_partido_{match_id}.json')
    try:
        with open(resumen_path, 'w', encoding='utf-8') as f:
            json.dump(resumen, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error guardando resumen: {e}")
    
    # Log del resumen
    logging.info(f"RESUMEN PARTIDO {match_id} (Jornada {jornada}):")
    logging.info(f"  - XML eventos: {len(archivos_descargados['xml_eventos'])}")
    logging.info(f"  - PDF informes: {len(archivos_descargados['pdf_informes'])}")
    logging.info(f"  - XLSX rendimiento: {len(archivos_descargados['xlsx_rendimiento'])}")
    logging.info(f"  - XLSX máxima exigencia: {len(archivos_descargados['xlsx_maxima_exigencia'])}")
    logging.info(f"  - CSV equipos: {len(archivos_descargados['csv_equipos'])}")
    logging.info(f"  - CSV jugadores: {len(archivos_descargados['csv_jugadores'])}")
    logging.info(f"  - Otros: {len(archivos_descargados['otros'])}")
    
    if errores:
        logging.warning(f"  - Errores: {len(errores)}")
    
    return total_descargados > 0

def procesar_partidos(ids, temporada, competicion, archivo_ids, credenciales, api_url_base, ejecutar_curl_comando):
    """Procesa múltiples partidos"""
    
    logging.info(f"=== INICIANDO PROCESAMIENTO DE {len(ids)} PARTIDOS ===")
    total = len(ids) 
    
    estadisticas_globales = {
        'partidos_procesados': 0,
        'partidos_con_errores': 0,
        'total_archivos_descargados': 0,
        'archivos_por_tipo': defaultdict(int),
        'errores_globales': []
    }
    
    for i, match_id in enumerate(ids):
        # Calculamos un progreso interno entre 0 y 14%
        progreso_interno = int((i / total) * 14)
        print(f"PROGRESS:{progreso_interno} - Descargando partido {i+1}/{total} (ID: {match_id})")
        sys.stdout.flush() # Obliga a enviar el texto a la web inmediatamente
        # ------------------------------
        logging.info(f"\n--- PARTIDO {i+1}/{len(ids)}: {match_id} ---")
        
        try:
            # Obtener asset data del partido
            asset_data = ejecutar_curl_comando(f"curl --location '{api_url_base}/Assets/{match_id}' {credenciales}")
            
            if not asset_data:
                estadisticas_globales['partidos_con_errores'] += 1
                estadisticas_globales['errores_globales'].append(f"Partido {match_id}: No se pudo obtener asset_data")
                continue
            
            # Procesar el partido
            exito = procesar_partido(match_id, asset_data, temporada, competicion, ejecutar_curl_comando)
            
            if exito:
                estadisticas_globales['partidos_procesados'] += 1
                
            else:
                estadisticas_globales['partidos_con_errores'] += 1
                estadisticas_globales['errores_globales'].append(f"Partido {match_id}: Error en procesamiento")
            
        except Exception as e:
            estadisticas_globales['partidos_con_errores'] += 1
            estadisticas_globales['errores_globales'].append(f"Partido {match_id}: Excepción - {str(e)}")
            logging.error(f"Error procesando partido {match_id}: {e}")
    
    # Resumen final
    logging.info(f"\n=== RESUMEN FINAL ===")
    logging.info(f"Partidos procesados exitosamente: {estadisticas_globales['partidos_procesados']}")
    logging.info(f"Partidos con errores: {estadisticas_globales['partidos_con_errores']}")
    
    if estadisticas_globales['errores_globales']:
        logging.warning(f"\nErrores encontrados:")
        for error in estadisticas_globales['errores_globales'][:10]:
            logging.warning(f"  - {error}")
        if len(estadisticas_globales['errores_globales']) > 10:
            logging.warning(f"  ... y {len(estadisticas_globales['errores_globales'])-10} errores más")
    
    return estadisticas_globales['partidos_procesados']

def guardar_id_en_csv(id, archivo_ids):
    """Guarda ID en CSV"""
    try:
        nombre_archivo = f'./VCF_Mediacoach_Data/{archivo_ids}'
        os.makedirs(os.path.dirname(nombre_archivo), exist_ok=True)
        
        with open(nombre_archivo, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([id])
        logging.debug(f"ID {id} guardado en {archivo_ids}")
    except Exception as e:
        logging.error(f"Error guardando ID {id}: {e}")

def leer_ids_csv(archivo_ids):
    """Lee IDs desde CSV"""
    nombre_archivo = f'./VCF_Mediacoach_Data/{archivo_ids}'
    logging.info(f"Buscando archivo: {nombre_archivo}")
    
    if not os.path.exists(nombre_archivo):
        logging.info(f"Archivo '{nombre_archivo}' no encontrado. Se creará uno nuevo.")
        os.makedirs(os.path.dirname(nombre_archivo), exist_ok=True)
        return []
    
    try:
        with open(nombre_archivo, 'r', newline='') as file:
            reader = csv.reader(file)
            ids_existentes = [row[0] for row in reader if row]
        logging.info(f"Se encontraron {len(ids_existentes)} IDs ya procesados")
        return ids_existentes
    except Exception as e:
        logging.error(f"Error leyendo el archivo {nombre_archivo}: {e}")
        return []

def obtener_ids(jornada_inicial, jornada_final, season_id, temporada, competition_id, competicion, ruta_parquet, credenciales, api_url_base, ejecutar_curl_comando):
    """Obtiene IDs de partidos validando contra el Parquet"""
    logging.info(f"Obteniendo IDs para {temporada} - {competicion}")
    
    # CAMBIO AQUÍ: Usamos la nueva función
    ids_existente = obtener_ids_desde_parquet(ruta_parquet)
    
    try:
        matches = ejecutar_curl_comando(f"""
        curl --location '{api_url_base}/Championships/seasons/{season_id}/competitions/{competition_id}/matches' \\
        {credenciales}
        """)

        if not matches:
            logging.error("No se pudieron obtener los matches")
            return []
            
        ids_filtrados = [item['id'] for item in matches if jornada_inicial <= int(item['matchDayNumber']) <= jornada_final]
        
        # Comparamos contra lo encontrado en el parquet
        ids_nuevos = [id for id in ids_filtrados if id not in ids_existente]
        
        logging.info(f"Matches totales en temporada: {len(matches)}")
        logging.info(f"Matches entre jornadas {jornada_inicial}-{jornada_final}: {len(ids_filtrados)}")
        logging.info(f"Matches ya en Parquet: {len(ids_existente)}")
        logging.info(f"Matches nuevos a procesar: {len(ids_nuevos)}")
        
        return ids_nuevos
        
    except Exception as e:
        logging.error(f"Error obteniendo matches: {e}")
        return []

def obtener_ids_desde_parquet(ruta_parquet):
    """Lee el parquet y extrae los IDs (UUIDs) de la columna archivo_origen"""
    logging.info(f"Buscando historial en: {ruta_parquet}")
    
    if not os.path.exists(ruta_parquet):
        logging.warning(f"No existe el archivo {ruta_parquet}. Se descargarán todos los partidos.")
        return []
    
    try:
        # Leemos solo la columna necesaria
        df = pd.read_parquet(ruta_parquet, columns=['archivo_origen'])
        
        # Filtramos nulos y convertimos a string
        archivos = df['archivo_origen'].dropna().astype(str).tolist()
        
        ids_encontrados = set()
        # Patrón para encontrar el UUID (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
        uuid_pattern = re.compile(r'([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})')
        
        for archivo in archivos:
            match = uuid_pattern.search(archivo)
            if match:
                ids_encontrados.add(match.group(1)) # Añadimos solo el ID
                
        logging.info(f"Se encontraron {len(ids_encontrados)} IDs únicos en el Parquet")
        return list(ids_encontrados)
        
    except Exception as e:
        logging.error(f"Error leyendo el parquet: {e}")
        return []

def obtener_temporadas_disponibles(credenciales, api_url_base, ejecutar_curl_comando):
    """Obtiene todas las temporadas disponibles (PRIMERO)"""
    try:
        logging.info("🔍 Descubriendo temporadas disponibles en la API...")
        
        endpoints_a_probar = [
            "/Championships/seasons",
            "/seasons", 
            "/Championships",
            "/api/seasons"
        ]
        
        for endpoint in endpoints_a_probar:
            try:
                logging.info(f"🔍 Probando endpoint: {endpoint}")
                
                temp_data = ejecutar_curl_comando(f"""
                curl --location '{api_url_base}{endpoint}' \\
                {credenciales}
                """)
                
                logging.info(f"DEBUG: Tipo: {type(temp_data)}, Contenido: {str(temp_data)[:300]}")
                
                if temp_data and isinstance(temp_data, list):
                    temporadas = []
                    for i, temp in enumerate(temp_data):
                        if isinstance(temp, dict):
                            nombre = temp.get('name') or temp.get('Name') or temp.get('seasonName') or f"Temporada {i+1}"
                            id_temp = temp.get('id') or temp.get('Id') or temp.get('seasonId') or ''
                        else:
                            nombre = f"Temporada {i+1}"
                            id_temp = str(temp)
                        
                        temporadas.append({
                            "nombre": nombre,
                            "id": id_temp,
                            "input": i,
                            "competiciones_raw": temp.get('competitions', []) # Guardamos las ligas aquí
                        })
                    
                    logging.info(f"✅ Encontradas {len(temporadas)} temporadas en {endpoint}")
                    return temporadas
                    
            except Exception as e:
                logging.warning(f"❌ Error en endpoint {endpoint}: {e}")
                continue
        
        return None
        
    except Exception as e:
        logging.error(f"Error descubriendo temporadas: {e}")
        return None

def obtener_competiciones_para_temporada(season_id, credenciales, api_url_base, ejecutar_curl_comando):
    """Obtiene competiciones disponibles para una temporada específica (SEGUNDO)"""
    try:
        logging.info(f"🔍 Descubriendo competiciones para temporada {season_id}...")
        
        endpoints_a_probar = [
            f"/Championships/seasons/{season_id}/competitions",
            f"/seasons/{season_id}/competitions",
            f"/Championships/seasons/{season_id}"
        ]
        
        for endpoint in endpoints_a_probar:
            try:
                logging.info(f"🔍 Probando endpoint: {endpoint}")
                
                comp_data = ejecutar_curl_comando(f"""
                curl --location '{api_url_base}{endpoint}' \\
                {credenciales}
                """)
                
                logging.info(f"DEBUG: Tipo: {type(comp_data)}, Contenido: {str(comp_data)[:300]}")
                
                if comp_data and isinstance(comp_data, list):
                    competiciones = []
                    for i, comp in enumerate(comp_data):
                        if isinstance(comp, dict):
                            nombre = comp.get('name') or comp.get('Name') or f"Competición {i+1}"
                            id_comp = comp.get('id') or comp.get('Id') or ''
                        else:
                            nombre = f"Competición {i+1}"
                            id_comp = str(comp)
                        
                        competiciones.append({
                            "nombre": nombre,
                            "id": id_comp,
                            "input": i
                        })
                    
                    logging.info(f"✅ Encontradas {len(competiciones)} competiciones en {endpoint}")
                    return competiciones
                
                # Si es dict, puede que tenga las competiciones en una key
                elif isinstance(comp_data, dict) and 'competitions' in comp_data:
                    logging.info("🔍 Encontradas competiciones en key 'competitions'")
                    return obtener_competiciones_para_temporada_desde_lista(comp_data['competitions'])
                    
            except Exception as e:
                logging.warning(f"❌ Error en endpoint {endpoint}: {e}")
                continue
        
        return None
        
    except Exception as e:
        logging.error(f"Error obteniendo competiciones para temporada: {e}")
        return None

def obtener_competiciones_para_temporada_desde_lista(lista_competiciones):
    """Procesa una lista de competiciones desde la API"""
    competiciones = []
    for i, comp in enumerate(lista_competiciones):
        if isinstance(comp, dict):
            nombre = comp.get('name') or comp.get('Name') or f"Competición {i+1}"
            id_comp = comp.get('id') or comp.get('Id') or ''
        else:
            nombre = f"Competición {i+1}"
            id_comp = str(comp)
        
        competiciones.append({
            "nombre": nombre,
            "id": id_comp,
            "input": i
        })
    
    return competiciones

def main():
    """Función principal corregida para evitar errores 404 y NoneType"""
    logging.info("=== DESCARGADOR MEDIACOACH - MODO AUTOMÁTICO/MANUAL ===")
    
    # 1. VERIFICAR MODO DE EJECUCIÓN (Web/Dash o Manual)
    modo_automatico = len(sys.argv) > 4
    arg_liga = sys.argv[1] if modo_automatico else None
    arg_temporada = sys.argv[2] if modo_automatico else None
    arg_j_inicio = int(sys.argv[3]) if modo_automatico else None
    arg_j_fin = int(sys.argv[4]) if modo_automatico else None

    # Configuración de autenticación
    client_id = '58191b89-cee4-11ed-a09d-ee50c5eb4bb5'
    scope = 'b2bapiclub-api'
    grant_type = 'password'
    username = 'b2bvillarealcf@mediacoach.es'
    password = 'r728-FHj3RE!'
    token_url = 'https://id.mediacoach.es/connect/token'

    data = {'client_id': client_id, 'scope': scope, 'grant_type': grant_type, 'username': username, 'password': password}
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    # Obtener token
    logging.info("Obteniendo token de acceso...")
    response = requests.post(token_url, data=data, headers=headers)
    if response.status_code == 200:
        AccessToken = response.json().get('access_token', '')
    else:
        logging.error(f'Error obteniendo token: {response.status_code}')
        return

    SubscriptionKey = '729f9154234d4ff3bb0a692c6a0510c4'
    api_url_base = "https://club-api.mediacoach.es"
    credenciales = f"--header 'Ocp-Apim-Subscription-Key: {SubscriptionKey}' --header 'Authorization: Bearer {AccessToken}'"

    def ejecutar_curl_comando(comando):
        process = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0: return None
        return json.loads(stdout)

    # 1️⃣ PRIMERO: Obtener temporadas (Usando el endpoint que sabemos que funciona)
    temporadas = obtener_temporadas_disponibles(credenciales, api_url_base, ejecutar_curl_comando)
    if not temporadas:
        logging.error("No se pudieron obtener temporadas de la API.")
        return

    season_id = None
    season_name = None
    input_temporadas = 0

    if modo_automatico:
        # Dash envía "Temporada 24-25", buscamos coincidencia
        for i, t in enumerate(temporadas):
            if arg_temporada.lower() in t["nombre"].lower():
                season_id = t["id"]
                season_name = t["nombre"]
                input_temporadas = i
                break
    else:
        print("\n📅 Lista de temporadas disponibles:")
        for t in temporadas: print(f'{t["nombre"]}, seleccione {t["input"]}')
        input_temporadas = int(input("\nSeleccione una temporada: "))
        season_id = temporadas[input_temporadas]["id"]
        season_name = temporadas[input_temporadas]["nombre"]

    if not season_id:
        logging.error(f"Temporada no encontrada: {arg_temporada}")
        return

    # 2️⃣ SEGUNDO: Obtener competiciones (SIN LLAMAR A LA API OTRA VEZ para evitar 404)
    print(f"\n🔍 Descubriendo competiciones para {season_name}...")
    
    # Extraemos las ligas que YA bajamos en el paso anterior (vienen dentro del objeto temporada)
    raw_competiciones = temporadas[input_temporadas].get('competiciones_raw', [])
    competiciones = []
    
    # Forzamos tus nombres preferidos según el ID real de MediaCoach
    mapping_vcf = {
        "39df9ec8-be91-4be5-1925-4b670a4cbed9": "La Liga",    # Primera
        "39df9ec8-becb-86ea-b5e8-600c1b47968d": "La Liga 2"   # Segunda
    }

    # Aliases para normalizar nombres comerciales a nombres internos
    aliases_competiciones = {
        "laliga ea sports": "La Liga",
        "laliga hypermotion": "La Liga 2",
        "la liga": "La Liga",
        "la liga 2": "La Liga 2",
        "primera": "La Liga",
        "segunda": "La Liga 2",
    }

    for i, c_api in enumerate(raw_competiciones):
        c_id = c_api.get('id')
        nombre_api = c_api.get('name', '')
        nombre_final = mapping_vcf.get(c_id, nombre_api)
        competiciones.append({"nombre": nombre_final, "id": c_id, "input": i})

    # Si la API no devolvió nada, usamos los IDs fijos que proporcionaste como rescate
    if not competiciones:
        logging.warning("⚠️ Lista de ligas vacía en API, usando backup manual...")
        competiciones = [
            {"nombre": "La Liga", "id": "39df9ec8-be91-4be5-1925-4b670a4cbed9", "input": 0},
            {"nombre": "La Liga 2", "id": "39df9ec8-becb-86ea-b5e8-600c1b47968d", "input": 1}
        ]

    competition_id = None
    competition_name = None

    if modo_automatico:
        # Normalizar arg_liga usando aliases (LALIGA EA SPORTS -> La Liga, etc.)
        arg_liga_normalizado = aliases_competiciones.get(arg_liga.strip().lower(), arg_liga)
        logging.info(f"🔍 Buscando liga: '{arg_liga}' -> normalizado: '{arg_liga_normalizado}'")

        # 1. Intentamos primero coincidencia EXACTA (para no confundir Liga con Liga 2)
        for c in competiciones:
            if arg_liga_normalizado.strip().lower() == c["nombre"].strip().lower():
                competition_id = c["id"]
                competition_name = c["nombre"]
                logging.info(f"🎯 Coincidencia exacta encontrada: {competition_name}")
                break

        # 2. Si no hubo exacta, probamos coincidencia parcial (fallback)
        if not competition_id:
            for c in competiciones:
                if arg_liga_normalizado.lower() in c["nombre"].lower():
                    competition_id = c["id"]
                    competition_name = c["nombre"]
                    logging.info(f"⚠️ Coincidencia parcial encontrada: {competition_name}")
                    break
    else:
        print(f"\n📋 Ligas disponibles:")
        for c in competiciones: print(f'  {c["input"]}: {c["nombre"]}')
        input_idx = int(input("\nSeleccione competición: "))
        competition_id = competiciones[input_idx]["id"]
        competition_name = competiciones[input_idx]["nombre"]

    if not competition_id:
        logging.error(f"Competición no identificada: {arg_liga}")
        return

    # 3️⃣ TERCERO: Configurar jornadas y Parquet
    jornada_inicial = arg_j_inicio if modo_automatico else int(input("\nJornada inicial: "))
    jornada_final = arg_j_fin if modo_automatico else int(input("Jornada final: "))
    
    # Ruta donde se verifica qué partidos ya tenemos para no duplicar
    ruta_parquet = './data/rendimiento_fisico.parquet' 
    
    # Normalizar nombres para las carpetas de archivos sucios
    temporada_dir = season_name.replace(" ", "_").replace("/", "_")
    competicion_dir = competition_name.replace(" ", "_")

    # 4️⃣ CUARTO: Obtener IDs y procesar descarga
    print(f"\n🚀 Iniciando descarga de {competicion_dir} ({jornada_inicial}-{jornada_final})...")
    
    ids = obtener_ids(jornada_inicial, jornada_final, season_id, temporada_dir, 
                      competition_id, competicion_dir, ruta_parquet, 
                      credenciales, api_url_base, ejecutar_curl_comando)

    if ids:
        partidos_ok = procesar_partidos(ids, temporada_dir, competicion_dir, None, 
                                        credenciales, api_url_base, ejecutar_curl_comando)
        print(f"\n✅ Finalizado: {partidos_ok} partidos descargados.")
    else:
        print("\nℹ️ No hay partidos nuevos para las jornadas seleccionadas.")

if __name__ == "__main__":
    main()