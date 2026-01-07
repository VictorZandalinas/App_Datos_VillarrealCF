import os
import zipfile
import re

def buscar_texto_en_excel(file_path):
    """Busca el texto de LALIGA en un archivo Excel"""
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            for name in z.namelist():
                if name.endswith('.xml'):
                    with z.open(name) as f:
                        try:
                            content = f.read().decode('utf-8', errors='ignore')
                            # Buscar el texto que empiece por LALIGA
                            matches = re.findall(r'>\s*(LALIGA.*?)<', content, re.DOTALL)
                            if matches:
                                # Retorna la primera coincidencia encontrada
                                return matches[0].replace('\n', ' ').strip()
                        except Exception as e:
                            print(f"Error leyendo {name} dentro de {file_path}: {e}")
    except Exception as e:
        print(f"Error abriendo {file_path}: {e}")
    return None

def procesar_carpetas_partidos():
    """Procesa y renombra las carpetas de partidos para todas las temporadas y ligas"""
    base_data_path = 'VCF_Mediacoach_Data'
    
    if not os.path.exists(base_data_path):
        print(f"Error: No existe la carpeta {base_data_path}")
        return
    
    # 🔍 DEBUG: Ver qué hay en la carpeta base
    print(f"🔍 Contenido de {base_data_path}:")
    for item in os.listdir(base_data_path):
        print(f"  - {item}")
    
    total_procesados = 0
    total_errores = 0
    
    # Recorrer todas las temporadas
    for temporada in os.listdir(base_data_path):
        temporada_path = os.path.join(base_data_path, temporada)
        
        # Saltar archivos (como ids_procesados.csv)
        if not os.path.isdir(temporada_path) or temporada == 'ids_procesados.csv':
            continue
            
        if 'temporada' in temporada.lower() or 'season' in temporada.lower():
            print(f"\n=== PROCESANDO TEMPORADA: {temporada} ===")
            
            # 🔍 DEBUG: Ver qué hay en la temporada
            print(f"🔍 Contenido de {temporada}:")
            for item in os.listdir(temporada_path):
                print(f"  - {item}")
            
            # Recorrer todas las ligas en esta temporada
            for liga in os.listdir(temporada_path):
                liga_path = os.path.join(temporada_path, liga)
                
                if os.path.isdir(liga_path):
                    partidos_path = os.path.join(liga_path, 'Partidos')
                    
                    if os.path.exists(partidos_path):
                        print(f"\n--- Procesando liga: {liga} ---")
                        
                        # 🔍 DEBUG: Ver qué carpetas de partidos hay
                        carpetas_partidos = os.listdir(partidos_path)
                        print(f"🔍 Carpetas de partidos encontradas: {len(carpetas_partidos)}")
                        for carpeta in carpetas_partidos[:3]:  # Solo mostrar las primeras 3
                            print(f"  - {carpeta}")
                        
                        # Procesar cada carpeta de partido
                        for folder_name in os.listdir(partidos_path):
                            folder_path = os.path.join(partidos_path, folder_name)
                            
                            if os.path.isdir(folder_path):
                                # 🔍 DEBUG: Ver qué archivos hay en la carpeta
                                archivos = os.listdir(folder_path)
                                archivos_rendimiento = [f for f in archivos if f.startswith('rendimiento_1') and f.endswith('.xlsx')]
                                print(f"🔍 En {folder_name} hay {len(archivos)} archivos, {len(archivos_rendimiento)} de rendimiento")
                                
                                # Buscar archivo de rendimiento
                                for file in os.listdir(folder_path):
                                    if file.startswith('rendimiento_1') and file.endswith('.xlsx'):
                                        print(f"📄 Procesando archivo: {file}")
                                        file_path = os.path.join(folder_path, file)
                                        texto = buscar_texto_en_excel(file_path)
                                        
                                        if texto:
                                            try:
                                                print(f"Texto encontrado en {folder_name}: {texto}")
                                                
                                                # Extraer jornada
                                                jornada_match = re.search(r'\b(J\d+)\b', texto)
                                                # Extraer partido (después de la última '|')
                                                partido_match = re.search(r'\|\s*[^|]*\|\s*(J\d+)\s*\|\s*(.*?)\s*\(', texto)
                                                
                                                if jornada_match and partido_match:
                                                    jornada = jornada_match.group(1).lower()
                                                    partido = partido_match.group(2).strip()
                                                    
                                                    # Formatear el nombre final de carpeta
                                                    nuevo_nombre = f"{jornada}_{partido}"
                                                    nuevo_nombre = nuevo_nombre.lower().replace(' - ', '-').replace(' ', '')
                                                    
                                                    new_folder_path = os.path.join(partidos_path, nuevo_nombre)
                                                    
                                                    # Verificar que no existe ya una carpeta con ese nombre
                                                    if not os.path.exists(new_folder_path):
                                                        print(f'✅ Renombrando: {folder_name} -> {nuevo_nombre}')
                                                        os.rename(folder_path, new_folder_path)
                                                        total_procesados += 1
                                                    else:
                                                        print(f'⚠️ Ya existe carpeta: {nuevo_nombre}')
                                                        total_errores += 1
                                                else:
                                                    print(f"❌ No se pudo extraer jornada o partido de: {texto}")
                                                    total_errores += 1
                                                    
                                            except Exception as e:
                                                print(f"❌ Error procesando {folder_name}: {e}")
                                                total_errores += 1
                                        else:
                                            print(f"⚠️ No se encontró texto LALIGA en: {folder_name}")
                                        
                                        # Solo procesar el primer archivo de rendimiento encontrado
                                        break
                    else:
                        print(f"ℹ️ No existe carpeta Partidos en: {liga}")
        else:
            print(f"⏭️ Saltando carpeta que no es temporada: {temporada}")
    
    # Resumen final
    print(f"\n=== RESUMEN FINAL ===")
    print(f"✅ Carpetas renombradas exitosamente: {total_procesados}")
    print(f"❌ Errores encontrados: {total_errores}")

# Ejecutar el procesamiento
if __name__ == "__main__":
    procesar_carpetas_partidos()