import os
import xml.etree.ElementTree as ET
import pandas as pd
import glob
from pathlib import Path

def es_xml_valido(xml_path):
    """Verifica si el XML es válido según los criterios especificados"""
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            primera_linea = f.readline().strip()
            segunda_linea = f.readline().strip()
        
        # Queremos el XML que tiene UTF-8 mayúscula Y NO tiene IdCompetition
        xml_header_ok = primera_linea == '<?xml version="1.0" encoding="UTF-8"?>'
        no_tiene_idcompetition = 'IdCompetition' not in segunda_linea
        
        es_valido = xml_header_ok and no_tiene_idcompetition
        
        if not es_valido:
            print(f"      ❌ Rechazado - Header UTF-8: {xml_header_ok}, Sin IdCompetition: {no_tiene_idcompetition}")
            print(f"      📋 L1: {primera_linea}")
            print(f"      📋 L2: {segunda_linea}")
        
        return es_valido
        
    except Exception as e:
        print(f"      ❌ Error leyendo XML: {e}")
        return False

def obtener_xml_para_procesar(carpeta_partido):
    """
    Obtiene un XML para procesar de una carpeta de partido.
    Prioridad:
    1. XMLs que cumplan los criterios de validación
    2. Si no hay válidos, cualquier XML disponible (fallback)
    """
    xml_files = glob.glob(os.path.join(carpeta_partido, "*.xml"))
    
    if not xml_files:
        return None, "no_xml"
    
    # Primero, buscar XMLs válidos (criterios estrictos)
    xmls_validos = []
    for xml_file in xml_files:
        if es_xml_valido(xml_file):
            xmls_validos.append(xml_file)
    
    # Si hay XMLs válidos, devolver el primero
    if xmls_validos:
        return xmls_validos[0], "valido"
    
    # Si no hay XMLs válidos, usar cualquier XML como fallback
    print(f"   ⚠️  No hay XMLs válidos, usando fallback: {os.path.basename(xml_files[0])}")
    return xml_files[0], "fallback"

def extraer_jornada_partido(nombre_carpeta):
    """Extrae jornada y partido del nombre de la carpeta"""
    jornada = None
    partido = None
    
    # Extraer jornada (si empieza por j seguido de número)
    if nombre_carpeta.lower().startswith('j') and len(nombre_carpeta) > 1:
        for i, char in enumerate(nombre_carpeta[1:], 1):
            if not char.isdigit():
                jornada = nombre_carpeta[:i]
                break
        else:
            jornada = nombre_carpeta
    
    # Extraer partido (desde la primera _ hasta el final)
    if '_' in nombre_carpeta:
        partido = nombre_carpeta[nombre_carpeta.index('_')+1:]
    
    return jornada, partido

def parse_xml_file(xml_file_path, jornada, partido, temporada, liga):
    """Parsea un archivo XML y extrae los datos de las instancias"""
    data = []
    
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        file_metadata = {
            'id_game': root.get('IdGame'),
            'co_quality': root.get('CoQuality'),
            'id_competition': root.get('IdCompetition')
        }
        
        instances = root.findall('.//instance')
        
        for instance in instances:
            id_elem = instance.find('ID')
            start_elem = instance.find('start')
            end_elem = instance.find('end')
            code_elem = instance.find('code')
            
            if id_elem is None or start_elem is None or end_elem is None or code_elem is None:
                continue
                
            instance_data = {
                'temporada': temporada,
                'liga': liga,
                'ID': int(id_elem.text),
                'start': float(start_elem.text),
                'end': float(end_elem.text),
                'code': code_elem.text,
                'player_name': None,
                'team': None,
                'player_group': None,
                'jornada': jornada,
                'partido': partido,
                'xml_source': os.path.basename(xml_file_path),
                'id_game': file_metadata['id_game'],
                'co_quality': file_metadata['co_quality'],
                'id_competition': file_metadata['id_competition']
            }
            
            labels = instance.findall('label')
            for label in labels:
                text_elem = label.find('text')
                group_elem = label.find('group')
                
                if text_elem is not None:
                    if group_elem is not None:
                        group_text = group_elem.text
                        if group_text == 'Equipo':
                            instance_data['team'] = text_elem.text
                        elif 'Jugadores' in group_text:
                            instance_data['player_name'] = text_elem.text
                            instance_data['player_group'] = group_text
                        else:
                            instance_data['player_group'] = group_text
                    else:
                        instance_data['player_name'] = text_elem.text
            
            data.append(instance_data)
            
    except Exception as e:
        print(f"Error procesando {xml_file_path}: {e}")
    
    return data

def procesar_partidos():
    """Procesa todos los partidos y crea el archivo parquet"""
    base_data_path = "VCF_Mediacoach_Data"
    output_path = "data/eventos_partido.parquet"
    
    print(f"📁 Directorio actual: {os.getcwd()}")
    print(f"🔍 Buscando en: {base_data_path}")
    
    # Crear carpeta data si no existe
    os.makedirs("data", exist_ok=True)
    
    all_data = []
    
    # Recorrer todas las temporadas
    for temporada in os.listdir(base_data_path):
        temporada_path = os.path.join(base_data_path, temporada)
        
        if not os.path.isdir(temporada_path) or not (temporada.startswith('Temporada_') or temporada.startswith('Season_')):
            continue
            
        print(f"\n=== PROCESANDO TEMPORADA: {temporada} ===")
        
        # Recorrer todas las ligas
        for liga in os.listdir(temporada_path):
            liga_path = os.path.join(temporada_path, liga)
            
            if not os.path.isdir(liga_path):
                continue
                
            partidos_path = os.path.join(liga_path, 'Partidos')
            
            if not os.path.exists(partidos_path):
                continue
                
            print(f"\n--- Procesando liga: {liga} ---")
            
            carpetas = os.listdir(partidos_path)
            # FILTRAR solo carpetas que empiecen por j + número
            carpetas = [c for c in carpetas if os.path.isdir(os.path.join(partidos_path, c)) and 
                       c.lower().startswith('j') and len(c) > 1 and c[1].isdigit()]
            
            print(f"📁 Encontradas {len(carpetas)} carpetas de partidos en {partidos_path}")
    
            carpetas_procesadas = 0
            xmls_encontrados = 0
            xmls_validos_usados = 0
            xmls_fallback_usados = 0
            carpetas_sin_xml = []
    
            for carpeta in carpetas:
                carpeta_path = os.path.join(partidos_path, carpeta)
                
                if not os.path.isdir(carpeta_path):
                    print(f"⏭️  Saltando archivo: {carpeta}")
                    continue
                
                print(f"📂 Procesando carpeta: {carpeta}")
                carpetas_procesadas += 1
                
                # Buscar XMLs en la carpeta
                xml_files = glob.glob(os.path.join(carpeta_path, "*.xml"))
                print(f"   📄 XMLs encontrados: {len(xml_files)}")
                xmls_encontrados += len(xml_files)
                
                if len(xml_files) == 0:
                    print(f"   ❌ CARPETA SIN XMLs: {carpeta}")
                    carpetas_sin_xml.append(f"{carpeta} (sin XMLs)")
                    continue
                
                if len(xml_files) == 1:
                    print(f"   📋 CARPETA CON 1 XML: vamos a ver qué tipo es...")
                elif len(xml_files) > 1:
                    print(f"   ⚠️  CARPETA CON MÚLTIPLES XMLs: {len(xml_files)} archivos")
                
                # Analizar XMLs en detalle para carpetas con 1 solo XML
                if len(xml_files) == 1:
                    xml_file = xml_files[0]
                    print(f"   🔍 Analizando: {os.path.basename(xml_file)}")
                    
                    try:
                        with open(xml_file, 'r', encoding='utf-8') as f:
                            l1 = f.readline().strip()
                            l2 = f.readline().strip()
                        print(f"      📋 L1: {l1}")
                        print(f"      📋 L2: {l2}")
                        
                        tiene_utf8_mayus = l1 == '<?xml version="1.0" encoding="UTF-8"?>'
                        tiene_idcompetition = 'IdCompetition' in l2
                        
                        print(f"      📊 UTF-8 mayúscula: {tiene_utf8_mayus}")
                        print(f"      📊 Tiene IdCompetition: {tiene_idcompetition}")
                        
                        if tiene_utf8_mayus and tiene_idcompetition:
                            print(f"      🔍 Este XML tiene UTF-8 mayús pero SÍ tiene IdCompetition")
                        elif tiene_utf8_mayus and not tiene_idcompetition:
                            print(f"      ✅ Este XML tiene UTF-8 mayús y NO tiene IdCompetition (es el bueno)")
                        elif not tiene_utf8_mayus and tiene_idcompetition:
                            print(f"      ❌ Este XML tiene utf-8 minus y SÍ tiene IdCompetition")
                        else:
                            print(f"      ❓ Caso raro: utf-8 minus sin IdCompetition")
                            
                    except Exception as e:
                        print(f"      ❌ Error leyendo: {e}")
                
                # Obtener XML para procesar (con fallback garantizado)
                xml_a_procesar, tipo_seleccion = obtener_xml_para_procesar(carpeta_path)
                
                if not xml_a_procesar:
                    print(f"   ❌ CARPETA SIN XMLs: {carpeta}")
                    carpetas_sin_xml.append(f"{carpeta} (sin XMLs)")
                    continue
                
                # Contar estadísticas
                if tipo_seleccion == "valido":
                    xmls_validos_usados += 1
                    print(f"   ✅ Usando XML válido: {os.path.basename(xml_a_procesar)}")
                elif tipo_seleccion == "fallback":
                    xmls_fallback_usados += 1
                    print(f"   🔄 Usando XML fallback: {os.path.basename(xml_a_procesar)}")
                    
                jornada, partido = extraer_jornada_partido(carpeta)
                print(f"   📊 Jornada: {jornada}, Partido: {partido}")
                
                file_data = parse_xml_file(xml_a_procesar, jornada, partido, temporada, liga)
                
                if file_data:
                    print(f"   ✅ Extraídas {len(file_data)} instancias")
                    all_data.extend(file_data)
                else:
                    print(f"   ❌ No se extrajeron datos del XML")
            
            # Resumen por liga
            print(f"\n📊 Resumen {liga}:")
            print(f"   Carpetas procesadas: {carpetas_procesadas}")
            print(f"   XMLs encontrados: {xmls_encontrados}")
            print(f"   XMLs válidos usados: {xmls_validos_usados}")
            print(f"   XMLs fallback usados: {xmls_fallback_usados}")
            print(f"   Total XMLs procesados: {xmls_validos_usados + xmls_fallback_usados}")
            
            if carpetas_sin_xml:
                print(f"\n❌ Carpetas sin XMLs ({len(carpetas_sin_xml)}):")
                for carpeta_info in carpetas_sin_xml:
                    print(f"   - {carpeta_info}")
    
    # Resumen final (FUERA de todos los bucles)
    print(f"\n🎯 RESUMEN FINAL GLOBAL:")
    print(f"   Total instancias extraídas: {len(all_data)}")
    
    if not all_data:
        print("❌ No se encontraron datos para procesar")
        return
    
    new_df = pd.DataFrame(all_data)
    
    # Cargar archivo existente si existe
    if os.path.exists(output_path):
        existing_df = pd.read_parquet(output_path)
        
        # Eliminar duplicados basados en combinación única
        if 'temporada' in existing_df.columns and 'liga' in existing_df.columns and 'partido' in existing_df.columns:
            # Crear clave única combinando múltiples campos
            existing_df['clave_unica'] = existing_df['temporada'] + '_' + existing_df['liga'] + '_' + existing_df['partido'] + '_' + existing_df['ID'].astype(str)
            new_df['clave_unica'] = new_df['temporada'] + '_' + new_df['liga'] + '_' + new_df['partido'] + '_' + new_df['ID'].astype(str)
            
            # 🔧 CORREGIDO: Definir existing_keys ANTES de usarla
            existing_keys = set(existing_df['clave_unica'].values)
            
            print(f"\n🔍 DEBUG: Primeras 5 claves nuevas:")
            print(new_df['clave_unica'].head().tolist())
            print(f"🔍 DEBUG: Total claves nuevas únicas: {len(set(new_df['clave_unica']))}")
            print(f"🔍 DEBUG: Total claves existentes: {len(existing_keys)}")
            print(f"🔍 DEBUG: Intersección (duplicados): {len(set(new_df['clave_unica']) & existing_keys)}")
            
            new_df = new_df[~new_df['clave_unica'].isin(existing_keys)]
            
            # Limpiar columnas temporales
            new_df = new_df.drop('clave_unica', axis=1)
        else:
            # Archivo viejo sin nuevas columnas, usar solo ID
            existing_ids = set(existing_df['ID'].values)
            new_df = new_df[~new_df['ID'].isin(existing_ids)]
        
        if len(new_df) > 0:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"✅ Añadidos {len(new_df)} nuevos registros al archivo existente")
        else:
            combined_df = existing_df
            print(f"ℹ️  No hay nuevos registros para añadir")
    else:
        combined_df = new_df
        print(f"✅ Creando nuevo archivo con {len(new_df)} registros")
    
    # Guardar archivo parquet
    combined_df.to_parquet(output_path, index=False)
    print(f"\n🎉 Archivo guardado: {output_path} con {len(combined_df)} registros totales")
    
    # Mostrar distribución por temporada/liga
    if 'temporada' in combined_df.columns and 'liga' in combined_df.columns:
        print(f"\n📊 Distribución de registros:")
        distribucion = combined_df.groupby(['temporada', 'liga']).size().reset_index(name='registros')
        for _, row in distribucion.iterrows():
            print(f"   {row['temporada']} - {row['liga']}: {row['registros']} registros")

if __name__ == "__main__":
    procesar_partidos()