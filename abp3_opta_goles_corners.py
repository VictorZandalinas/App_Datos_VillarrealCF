import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import patheffects
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

class CornersOfensivosReport:
    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/abp_events.parquet"):
        """
        Inicializa la clase para generar informes de córners ofensivos usando datos de eventos.
        """
        self.data_path = data_path
        self.team_stats_path = "extraccion_opta/datos_opta_parquet/team_stats.parquet"
        self.df_events = None
        self.df_teams = None
        self.corner_sequences = None # DataFrame para guardar los resultados de las secuencias
        
        # Cargar los datos de eventos y equipos
        self.load_data()
        
        # Extraer y analizar TODAS las secuencias de córner una sola vez
        if self.df_events is not None:
            self.extract_all_corner_sequences()
    
    def load_villarreal_logo(self):
        """Carga el escudo del Villarreal usando búsqueda por similitud"""
        return self.find_team_logo_by_similarity('Villarreal')

    def debug_metrics(self, equipo_seleccionado):
        """
        Muestra un informe detallado paso a paso para depurar la extracción de métricas.
        """
        print("\n" + "="*25 + " INICIO DEL MODO DEBUG " + "="*25)

        # --- PASO 1: VERIFICACIÓN DE DATOS CRUDOS ---
        print("\n--- PASO 1: Verificación de Datos Crudos ---")
        if self.df is None:
            print("❌ ERROR: El DataFrame (self.df) no se ha cargado. Revisa la ruta del archivo.")
            return

        print(f"✅ DataFrame cargado con {self.df.shape[0]} filas (filtrado por 'Total Partido').")
        
        # Es CRUCIAL ver los nombres exactos de las métricas disponibles en el archivo
        if 'NOMBRE MÉTRICA' in self.df.columns:
            metricas_disponibles = sorted(self.df['NOMBRE MÉTRICA'].unique())
            print(f"\n[INFO] Métricas Disponibles en el Archivo ({len(metricas_disponibles)} en total):")
            for metrica in metricas_disponibles:
                print(f"  - '{metrica}'")
        else:
            print("❌ ERROR: No se encuentra la columna 'NOMBRE MÉTRICA' en los datos.")
            return

        # --- PASO 2: COINCIDENCIA DE NOMBRES DE MÉTRICAS ---
        print("\n--- PASO 2: Comprobando coincidencia de nombres de métricas ---")
        print("Se comparan las métricas definidas en `self.corner_metrics` con las del archivo.")
        
        for metric_key, metric_name in self.corner_metrics.items():
            if metric_name in metricas_disponibles:
                print(f"  ✅ ENCONTRADA: '{metric_name}' (mapeada a '{metric_key}')")
            else:
                print(f"  ❌ NO ENCONTRADA: '{metric_name}'")
                print("     -> Causa probable: El nombre en el código no es idéntico al del archivo (revisa espacios, acentos, etc.).")
        
        # --- PASO 3: EXTRACCIÓN DE DATOS PARA EL EQUIPO SELECCIONADO ---
        print(f"\n--- PASO 3: Extrayendo datos para '{equipo_seleccionado}' ---")
        
        equipo_df = self.df[self.df['EQUIPO'] == equipo_seleccionado]
        if equipo_df.empty:
            print(f"❌ ERROR: No se encontraron datos para el equipo '{equipo_seleccionado}' en la columna 'EQUIPO'.")
            return
            
        print(f"✅ Se encontraron {len(equipo_df)} filas para '{equipo_seleccionado}'.")

        for metric_key, metric_name in self.corner_metrics.items():
            print(f"\n  Buscando métrica: '{metric_name}'...")
            metric_data = equipo_df[equipo_df['NOMBRE MÉTRICA'] == metric_name]
            
            if not metric_data.empty:
                print(f"    ✅ Se encontraron {len(metric_data)} entradas para esta métrica.")
                valores = metric_data['VALOR']
                print(f"    Valores originales (columna VALOR): {list(valores)}")
                
                # Simular la conversión a numérico y la suma
                valores_numericos = pd.to_numeric(valores, errors='coerce').fillna(0)
                suma_total = valores_numericos.sum()
                print(f"    Valores tras conversión numérica: {list(valores_numericos)}")
                print(f"    -> SUMA TOTAL para '{metric_key}': {suma_total}")
            else:
                print(f"    ❌ No se encontraron entradas para esta métrica para '{equipo_seleccionado}'. El valor será 0.")

        # --- PASO 4: VERIFICACIÓN DE CÁLCULOS FINALES ---
        print("\n--- PASO 4: Verificando el DataFrame final de estadísticas ---")
        all_stats, team_stats_final = self.get_team_corner_stats(equipo_seleccionado)
        
        if team_stats_final is not None:
            print(f"✅ Datos finales calculados para '{equipo_seleccionado}':")
            print(team_stats_final)
        else:
            print(f"❌ No se pudieron generar las estadísticas finales para '{equipo_seleccionado}'.")
            
        print("\n" + "="*26 + " FIN DEL MODO DEBUG " + "="*27 + "\n")
        
    def find_team_logo_by_similarity(self, equipo):
        """Busca el escudo del equipo por similitud en la carpeta escudos"""
        if not os.path.exists('assets/escudos'):
            return None
        
        # Obtener todos los archivos .png en la carpeta
        escudos_disponibles = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
        
        if not escudos_disponibles:
            return None
        
        # Limpiar nombre del equipo para comparar
        equipo_clean = equipo.lower().replace(' ', '').replace('cf', '').replace('fc', '').replace('real', '').replace('rcd', '')
        
        best_match = None
        best_similarity = 0
        
        # Buscar por similitud
        for escudo_file in escudos_disponibles:
            escudo_name = escudo_file.replace('.png', '').lower().replace('_', '').replace('cf', '').replace('fc', '').replace('real', '').replace('rcd', '')
            
            similarity = self.similarity(equipo_clean, escudo_name)
            
            if similarity > best_similarity and similarity > 0.4:  # Mínimo 40% similitud
                best_similarity = similarity
                best_match = escudo_file
        
        # Cargar el mejor match
        if best_match:
            try:
                logo_path = f"assets/escudos/{best_match}"
                print(f"Escudo encontrado para {equipo}: {best_match} (similitud: {best_similarity:.2f})")
                
                # CARGAR Y REDIMENSIONAR A TAMAÑO FIJO
                escudo_original = plt.imread(logo_path)
                escudo_redimensionado = self.resize_image_to_fixed_size(escudo_original, target_size=100)
                
                return escudo_redimensionado
            except Exception as e:
                print(f"Error al cargar {best_match}: {e}")
        
        return None

    def resize_image_to_fixed_size(self, image, target_size=100):
        """Redimensiona imagen a un tamaño fijo manteniendo proporción"""
        try:
            from PIL import Image as PILImage
            import numpy as np
            
            # Convertir de matplotlib a PIL
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Crear imagen PIL
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    pil_image = PILImage.fromarray(image, 'RGBA')
                else:  # RGB
                    pil_image = PILImage.fromarray(image, 'RGB')
            else:  # Grayscale
                pil_image = PILImage.fromarray(image, 'L')
            
            # Redimensionar manteniendo proporción
            pil_image.thumbnail((target_size, target_size), PILImage.Resampling.LANCZOS)
            
            # Crear imagen cuadrada con fondo transparente
            square_image = PILImage.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
            
            # Centrar la imagen redimensionada
            x_offset = (target_size - pil_image.width) // 2
            y_offset = (target_size - pil_image.height) // 2
            square_image.paste(pil_image, (x_offset, y_offset))
            
            # Convertir de vuelta a numpy array
            return np.array(square_image) / 255.0
            
        except Exception as e:
            print(f"Error al redimensionar imagen: {e}")
            return image

    def convert_to_grayscale(self, image):
        """Convierte imagen a escala de grises"""
        try:
            # Si la imagen tiene canal alfa, mantenerlo
            if image.shape[2] == 4:  # RGBA
                gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])  # Solo RGB
                # Crear imagen en escala de grises manteniendo alfa
                gray_image = np.zeros_like(image)
                gray_image[..., 0] = gray  # R
                gray_image[..., 1] = gray  # G  
                gray_image[..., 2] = gray  # B
                gray_image[..., 3] = image[..., 3]  # Alpha
                return gray_image
            else:  # RGB
                gray = np.dot(image, [0.2989, 0.5870, 0.1140])
                return np.stack([gray, gray, gray], axis=2)
        except Exception as e:
            print(f"Error al convertir a escala de grises: {e}")
            return image
        
    def load_data(self):
        """Carga los datos de eventos (Opta) y estadísticas de equipos."""
        try:
            self.df_events = pd.read_parquet(self.data_path)
            self.df_teams = pd.read_parquet(self.team_stats_path)

            print(f"Columnas en team_stats.parquet: {self.df_teams.columns.tolist()}")

            
            # Normalizar timestamp para ordenar eventos correctamente
            if 'timeStamp' in self.df_events.columns:
                 self.df_events['timeStamp'] = pd.to_datetime(self.df_events['timeStamp'].str.replace('Z', ''), errors='coerce')

            print(f"✅ Datos de eventos cargados: {self.df_events.shape[0]} filas")
            print(f"✅ Datos de equipos cargados: {self.df_teams.shape[0]} filas")
        except Exception as e:
            print(f"❌ Error al cargar los datos de Opta: {e}")

    
    def extract_all_corner_sequences(self):
            """
            Extrae TODOS los lanzamientos de córner (ambos lados) y analiza la secuencia resultante.
            Esta función adapta la lógica de abp2.1.
            """
            if self.df_events is None:
                print("❌ No hay datos de eventos cargados.")
                return

            print("🔍 Extrayendo y analizando todas las secuencias de córner...")
            
            df_sorted = self.df_events.sort_values(['Match ID', 'periodId', 'timeStamp']).reset_index(drop=True)
            
            lanzamientos_list = []
            
            # Procesar por partido
            for match_id, match_events in df_sorted.groupby('Match ID'):
                
                # --- MODIFICACIÓN CLAVE: Filtro para TODOS los córners ---
                # Ya no filtramos por y > 99, solo por el evento de córner.
                all_corners = match_events[
                    (match_events.get('Corner taken', '') == 'Sí') &
                    (match_events['x'].notna()) & 
                    (match_events['y'].notna())
                ].copy()
                
                for lanz_idx, lanzamiento in all_corners.iterrows():
                    # El índice de la fila en el grupo no es el mismo que en el DF original, hay que buscarlo
                    original_df_idx = lanzamiento.name
                    
                    # Analizar secuencia
                    result_data = self.analyze_lanzamiento_sequence(df_sorted, original_df_idx, lanzamiento)
                    
                    lanzamiento_data = {
                        'Match ID': match_id,
                        'Team ID': lanzamiento['Team ID'],
                        'Team Name': lanzamiento['Team Name'],
                        'playerName': lanzamiento['playerName'],
                    }
                    
                    lanzamiento_data.update(result_data)
                    lanzamientos_list.append(lanzamiento_data)
            
            if lanzamientos_list:
                self.corner_sequences = pd.DataFrame(lanzamientos_list)
                print(f"✅ Total de secuencias de córner analizadas: {len(self.corner_sequences)}")
            else:
                print("❌ No se encontraron lanzamientos de córner.")    

    def analyze_lanzamiento_sequence(self, match_events, lanzamiento_idx, lanzamiento_pass):
        """
        Analiza la secuencia post-lanzamiento con la lógica de tiempo y segunda jugada
        corregida y replicada exactamente desde ABP2.
        """
        start_time_min = lanzamiento_pass['timeMin']
        start_time_sec = lanzamiento_pass['timeSec']
        
        events_found = {
            'Goal': None, 'Post': None, 'Attempt Saved': None,
            'Miss': None, 'Otro contacto': None
        }
        is_second_play = False
        previous_event_coords = None
        result_event_idx = None
        pass_count = 0  # Contador de pases
        
        # ▼▼▼ AÑADE ESTA LÍNEA ▼▼▼
        # Guarda el timeStamp del último pase contado. Empezamos con el del lanzamiento.
        last_pass_timestamp = pd.to_datetime(lanzamiento_pass['timeStamp'])
        
        # Usamos el tiempo del evento previo para calcular la diferencia
        prev_event_time = lanzamiento_pass['timeMin'] * 60 + lanzamiento_pass['timeSec']

        for next_idx in range(lanzamiento_idx + 1, len(match_events)):

            next_event = match_events.iloc[next_idx]
            
            if (next_event['Event Name'] in ['Corner Awarded', 'Foul', 'Offside', 'End Period', 'Out'] or
                next_event.get('Throw in', '') == 'Sí' or
                next_event.get('Free kick taken', '') == 'Sí'):
                break
                
            next_event_time = next_event['timeMin'] * 60 + next_event['timeSec']
            time_diff = next_event_time - prev_event_time
            
            if time_diff > 5:
                break

            prev_event_time = next_event_time
            
            event_name = next_event['Event Name']
            event_team_id = next_event['Team ID']
            lanzamiento_team_id = lanzamiento_pass['Team ID']
            
            # ← NUEVO: Contar pases del mismo equipo
            if event_name == 'Pass' and event_team_id == lanzamiento_team_id:
                next_timestamp = pd.to_datetime(next_event['timeStamp'])
                # Solo contamos el pase si su timestamp es posterior al del último pase contado.
                if next_timestamp > last_pass_timestamp:
                    pass_count += 1
                    last_pass_timestamp = next_timestamp  # Actualizamos el timestamp.
                    
                    # Si llegamos a 5 o más pases, verificar la nueva condición
                    if pass_count >= 5:
                        # Contar pases con x < 70 en los últimos eventos
                        passes_back_field = 0
                        for check_idx in range(lanzamiento_idx + 1, next_idx + 1):
                            check_event = match_events.iloc[check_idx]
                            if (check_event['Event Name'] == 'Pass' and 
                                check_event['Team ID'] == lanzamiento_team_id and
                                float(check_event.get('x', 0)) < 70):
                                passes_back_field += 1
                        
                        # Si hay 2 o más pases con x < 70, cortar la secuencia
                        if passes_back_field >= 2:
                            break
            
            if event_name in ['Goal', 'Post', 'Attempt Saved', 'Miss'] and events_found[event_name] is None:
                if event_team_id == lanzamiento_team_id:
                    # ← NUEVO: Para goles, verificar que no sea el mismo jugador
                    if event_name == 'Goal' and next_event.get('playerName') == lanzamiento_pass.get('playerName'):
                        continue  # Saltar goles del mismo lanzador
                    
                    events_found[event_name] = next_event
                    result_event_idx = next_idx
                    if event_name == 'Goal':
                        break
            
            elif (event_name == 'Pass' and 
                event_team_id == lanzamiento_team_id and
                next_event.get('outcome') == 1 and
                # Añadimos la lógica estricta: debe haber un segundo pase consecutivo
                next_idx + 1 < len(match_events) and 
                match_events.iloc[next_idx + 1]['Event Name'] == 'Pass' and 
                match_events.iloc[next_idx + 1]['Team ID'] == lanzamiento_team_id and
                events_found['Otro contacto'] is None):
                events_found['Otro contacto'] = next_event

        # --- LÓGICA DE SEGUNDA JUGADA (COPIA EXACTA Y CORREGIDA DE ABP2) ---
        if result_event_idx is not None and result_event_idx > lanzamiento_idx + 1:
            previous_event_found = None
            # Buscar hacia atrás desde el remate hasta justo después del córner
            for search_idx in range(result_event_idx - 1, lanzamiento_idx, -1):
                candidate_event = match_events.iloc[search_idx]
                
                # CORRECCIÓN 1: La condición es x > 55, no 60.
                # CORRECCIÓN 2: Se añade la comprobación crucial del timeStamp.
                if (float(candidate_event.get('x', 0)) > 55 and
                        candidate_event['Event Name'] != 'Deleted event' and
                        # Esta línea es la clave: asegura que no es el mismo evento del córner
                        candidate_event['timeStamp'] != lanzamiento_pass['timeStamp']):
                    
                    previous_event_found = candidate_event
                    break # Encontramos el evento relevante, salimos del bucle.
            
            # CORRECCIÓN 3: La lógica es más explícita. Si se encontró un evento, es segunda jugada.
            if previous_event_found is not None:
                is_second_play = True
                prev_x = float(previous_event_found.get('x', 0))
                prev_y = float(previous_event_found.get('y', 0))
                previous_event_coords = (prev_x, prev_y)
            else:
                is_second_play = False
                
        # --- Determinar resultado final (sin cambios en esta parte) ---
        # (El resto de la función para devolver el diccionario de resultados es idéntico y correcto)
        if events_found['Goal'] is not None:
            event = events_found['Goal']
            return {
                'result_type': 'Gol', 'final_x': float(event.get('x', 0)), 'final_y': float(event.get('y', 0)),
                'goal_player': event.get('playerName', ''), 'goal_player_id': event.get('playerId'),
                'is_second_play': is_second_play, 'previous_event_coords': previous_event_coords
            }
        elif events_found['Attempt Saved'] is not None:
            event = events_found['Attempt Saved']
            return {
                'result_type': 'Tiro a puerta', 'final_x': float(event.get('x', 0)), 'final_y': float(event.get('y', 0)),
                'goal_player': event.get('playerName', ''), 'goal_player_id': event.get('playerId'),
                'is_second_play': is_second_play, 'previous_event_coords': previous_event_coords
            }
        elif events_found['Post'] is not None:
            event = events_found['Post']
            return {
                'result_type': 'Tiro al poste', 'final_x': float(event.get('x', 0)), 'final_y': float(event.get('y', 0)),
                'goal_player': event.get('playerName', ''), 'goal_player_id': event.get('playerId'),
                'is_second_play': is_second_play, 'previous_event_coords': previous_event_coords
            }
        elif events_found['Miss'] is not None:
            event = events_found['Miss']
            return {
                'result_type': 'Tiro fuera', 'final_x': float(event.get('x', 0)), 'final_y': float(event.get('y', 0)),
                'goal_player': event.get('playerName', ''), 'goal_player_id': event.get('playerId'),
                'is_second_play': is_second_play, 'previous_event_coords': previous_event_coords
            }
        elif events_found['Otro contacto'] is not None:
            event = events_found['Otro contacto']
            return {
                'result_type': 'Otro contacto', 'final_x': float(event.get('Pass End X', 0)), 'final_y': float(event.get('Pass End Y', 0)),
                'goal_player': None, 'goal_player_id': None,
                'is_second_play': False, 'previous_event_coords': None
            }
        else:
            return {
                'result_type': 'Sin remate', 'final_x': float(lanzamiento_pass.get('Pass End X', 0)), 'final_y': float(lanzamiento_pass.get('Pass End Y', 0)),
                'goal_player': None, 'goal_player_id': None,
                'is_second_play': False, 'previous_event_coords': None
            }
    
    def prepare_corners_metrics(self):
        """Prepara las métricas específicas de córners ofensivos"""
        if self.df is None:
            return
        
        print("Preparando métricas de córners ofensivos...")
        
        # Métricas específicas de córners ofensivos
        self.corner_metrics = {
            'goles_corners': 'Goles a favor B. P. saque de esquina (Nº)', # <--- ¡CAMBIO REALIZADO!
            'num_corners': 'B.P Saque de esquina a favor (Nº)',
            'pct_corners_rematados': 'B.P Saque de esquina a favor (% Rematados)',
            'goles_favor_total': 'Goles a favor (Nº)',
            'remates_totales': 'Remates totales (Nº)'
        }
        
        # Verificar qué métricas están disponibles
        available_metrics = []
        if 'NOMBRE MÉTRICA' in self.df.columns:
            metricas_disponibles = self.df['NOMBRE MÉTRICA'].unique()
            
            for metric_key, metric_name in self.corner_metrics.items():
                if metric_name in metricas_disponibles:
                    available_metrics.append(metric_key)
                    print(f"✅ Métrica encontrada: {metric_name}")
                else:
                    print(f"❌ Métrica no encontrada: {metric_name}")
        
        self.available_metrics = available_metrics
        print(f"Métricas disponibles para análisis: {available_metrics}")
    
    def get_team_corner_stats(self, equipo_seleccionado=None):
        """
        Calcula las estadísticas de córners para todos los equipos a partir de los datos de secuencia de eventos.
        """
        if self.corner_sequences is None:
            print("❌ No se han procesado las secuencias de córner.")
            return None, None
        
        # 1. Obtener la liga del equipo seleccionado para filtrar
        liga_seleccionada = self.df_teams[self.df_teams['Team Name'] == equipo_seleccionado]['Competition Name'].iloc[0]
        equipos_liga = self.df_teams[self.df_teams['Competition Name'] == liga_seleccionada]['Team Name'].unique()
        
        # Filtrar las secuencias para que solo incluyan equipos de esa liga
        df_filtrado = self.corner_sequences[self.corner_sequences['Team Name'].isin(equipos_liga)].copy()
        print(f"Filtrando secuencias de la liga: {liga_seleccionada}")

        # 2. Agrupar por equipo y calcular las métricas base
        stats_agrupadas = df_filtrado.groupby('Team Name').agg(
            # Contar el total de córners lanzados
            num_corners=('Team Name', 'size'),
            
            # Contar los goles resultantes de un córner
            goles_corners=('result_type', lambda x: (x == 'Gol').sum())
        ).reset_index()

        # 3. Calcular el número total de tiros (shots) por equipo
        def count_shots(series):
            shot_types = ['Gol', 'Tiro a puerta', 'Tiro al poste', 'Tiro fuera', 'Attempt Saved', 'Post', 'Miss']
            return series.isin(shot_types).sum()

        tiros_por_equipo = df_filtrado.groupby('Team Name')['result_type'].apply(count_shots).reset_index(name='tiros_corners')
        
        # Unir todo en un solo DataFrame
        team_stats = pd.merge(stats_agrupadas, tiros_por_equipo, on='Team Name', how='left')
        team_stats.fillna(0, inplace=True) # Rellenar con 0 si un equipo no tiene tiros

        # 4. Calcular las métricas derivadas (las que se usan en los gráficos)
        team_stats['goles_por_corner'] = team_stats['goles_corners'] / team_stats['num_corners'].replace(0, 1)
        team_stats['tiros_por_corner'] = team_stats['tiros_corners'] / team_stats['num_corners'].replace(0, 1)

        # Renombrar 'Team Name' a 'EQUIPO' para que coincida con lo que esperan las gráficas
        team_stats.rename(columns={'Team Name': 'EQUIPO'}, inplace=True)
        
        # 5. Devolver los datos en el formato esperado
        if equipo_seleccionado:
            equipo_data = team_stats[team_stats['EQUIPO'] == equipo_seleccionado]
            if not equipo_data.empty:
                return team_stats, equipo_data.iloc[0]
            else:
                return team_stats, None
        
        return team_stats, None
    
    def similarity(self, a, b):
        """Calcula la similitud entre dos strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def load_team_logo(self, equipo):
        """Carga el escudo del equipo - mismo método que el código original"""
        possible_names = [
            equipo,
            equipo.replace(' ', '_'),
            equipo.replace(' ', ''),
            equipo.lower(),
            equipo.lower().replace(' ', '_'),
            equipo.lower().replace(' ', '')
        ]
        
        for name in possible_names:
            logo_path = f"assets/escudos/{name}.png"
            if os.path.exists(logo_path):
                try:
                    return plt.imread(logo_path)
                except Exception as e:
                    continue
        
        return None
    
    def load_ball_image(self):
        """Carga la imagen del balón"""
        ball_path = "assets/balon.png"
        if os.path.exists(ball_path):
            try:
                return plt.imread(ball_path)
            except Exception as e:
                print(f"Error al cargar balón: {e}")
                return None
        return None
    
    def load_background(self):
        """Carga el fondo del informe"""
        bg_path = "assets/fondo_informes.png"
        if os.path.exists(bg_path):
            try:
                return plt.imread(bg_path)
            except Exception as e:
                print(f"Error al cargar fondo: {e}")
                return None
        return None
    
    def create_visualization(self, equipo_seleccionado, figsize=(11.69, 8.27)):
        """Crea la visualización completa del informe de córners ofensivos"""
        
        # Obtener datos de córners
        result = self.get_team_corner_stats(equipo_seleccionado)
        if result is None:
            print("No se pudieron obtener las estadísticas de córners")
            return None
        
        team_stats, equipo_data = result
        if equipo_data is None:
            print(f"No se encontraron datos para {equipo_seleccionado}")
            return None
                
        # Crear figura
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        # Cargar y establecer fondo
        background = self.load_background()
        if background is not None:
            try:
                ax_background = fig.add_axes([0, 0, 1, 1], zorder=-1)
                ax_background.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25, zorder=-1)
                ax_background.axis('off')
            except Exception as e:
                print(f"Error al aplicar fondo: {e}")
        
        # Configurar grid - Layout similar al PNG
        gs = fig.add_gridspec(2, 4, 
                 height_ratios=[0.15, 1], 
                 width_ratios=[1.0, 1, 1.6, 0.05],
                 hspace=0.3, wspace=1.0,
                 left=0.10, right=1., top=0.95, bottom=0.05)
        
        # Área del título
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        # Título principal
        ax_title.text(0.5, 0.6, 'CÓRNERS OFENSIVOS', 
                     fontsize=28, weight='bold', ha='center', va='center',
                     color='#1e3d59', family='serif')
        
        # Logos
        ball = self.load_ball_image()
        if ball is not None:
            try:
                imagebox = OffsetImage(ball, zoom=0.12)
                ab = AnnotationBbox(imagebox, (0.05, 0.5), frameon=False)
                ax_title.add_artist(ab)
            except:
                pass

        # ESCUDOS CON MISMO TAMAÑO GARANTIZADO
        escudo_zoom = 1.0

        # Escudo Villarreal
        villarreal_logo = self.find_team_logo_by_similarity('Villarreal')
        if villarreal_logo is not None:
            try:
                imagebox = OffsetImage(villarreal_logo, zoom=escudo_zoom)
                ab = AnnotationBbox(imagebox, (0.88, 0.5), frameon=False, zorder=2) 
                ax_title.add_artist(ab)
                print(f"✅ Escudo Villarreal: shape={villarreal_logo.shape}, zoom={escudo_zoom}")
            except Exception as e:
                print(f"❌ Error con escudo Villarreal: {e}")

        # Escudo del equipo seleccionado  
        equipo_logo = self.find_team_logo_by_similarity(equipo_seleccionado)
        if equipo_logo is not None:
            try:
                imagebox = OffsetImage(equipo_logo, zoom=escudo_zoom)  # MISMO ZOOM
                ab = AnnotationBbox(imagebox, (0.92, 0.5), frameon=False, zorder=1)
                ax_title.add_artist(ab)
                print(f"✅ Escudo {equipo_seleccionado}: shape={equipo_logo.shape}, zoom={escudo_zoom}")
            except Exception as e:
                print(f"❌ Error con escudo {equipo_seleccionado}: {e}")
        else:
            print(f"❌ No se encontró escudo para {equipo_seleccionado}")
        
        # Gráfico 1: Goles de córner a favor (izquierda)
        ax_goles = fig.add_subplot(gs[1, 0])
        ax_goles.set_facecolor('none')
        ax_goles.set_title('Goles de córner a favor', fontsize=14, weight='bold', 
                          color='#1e3d59', pad=15)
        self.plot_goles_corner(ax_goles, team_stats, equipo_seleccionado)
        
        # Gráfico 2: Goles a favor por córner (centro)
        ax_eficiencia = fig.add_subplot(gs[1, 1])
        ax_eficiencia.set_facecolor('none')
        ax_eficiencia.set_title('Goles a favor por córner', fontsize=14, weight='bold', 
                               color='#1e3d59', pad=15)
        self.plot_eficiencia_corner(ax_eficiencia, team_stats, equipo_seleccionado)  

        
        # Gráfico 3: Scatter plot (derecha)
        ax_scatter = fig.add_subplot(gs[1, 2])
        ax_scatter.set_facecolor('none')
        ax_scatter.set_title('Suma Tiros a favor / Córner', fontsize=14, weight='bold', 
                            color='#1e3d59', pad=15)
        self.plot_corner_scatter(ax_scatter, team_stats, equipo_seleccionado)
        
        return fig
    
    def plot_goles_corner(self, ax, team_stats, equipo_seleccionado):
        """Dibuja el gráfico de goles de córner a favor"""
        # Ordenar por goles de córners
        sorted_data = team_stats.sort_values('goles_corners', ascending=True)
        
        # Colores: destacar equipo seleccionado y Villarreal
        colors = []
        for team in sorted_data['EQUIPO']:
            if team == equipo_seleccionado:
                colors.append('#e74c3c')  # Rojo para equipo seleccionado
            elif 'villarreal' in team.lower():
                colors.append('#f39c12')  # Naranja para Villarreal
            else:
                colors.append('#95a5a6')  # Gris para otros
        
        # Crear gráfico horizontal
        bars = ax.barh(range(len(sorted_data)), sorted_data['goles_corners'], 
                    height=0.6,  # Barras más estrechas
                    color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=0.5)
        
        # Añadir valores para equipos destacados
        for i, (bar, valor, team) in enumerate(zip(bars, sorted_data['goles_corners'], sorted_data['EQUIPO'])):
            if team == equipo_seleccionado or 'villarreal' in team.lower():
                ax.text((valor) + 0.05, i, f'{int(valor)}', va='center', fontsize=9, weight='bold')
        
        # Configurar ejes
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data['EQUIPO'], fontsize=8)
        ax.set_xlabel('Goles de Córner', fontsize=10, weight='bold')
        
        # Línea promedio
        promedio = sorted_data['goles_corners'].mean()
        ax.axvline(promedio, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def plot_eficiencia_corner(self, ax, team_stats, equipo_seleccionado):
        """Dibuja el gráfico de eficiencia (goles por córner)"""
        sorted_data = team_stats.sort_values('goles_por_corner', ascending=True)
        
        colors = []
        for team in sorted_data['EQUIPO']:
            if team == equipo_seleccionado:
                colors.append('#e74c3c')
            elif 'villarreal' in team.lower():
                colors.append('#f39c12')
            else:
                colors.append('#95a5a6')
        
        bars = ax.barh(range(len(sorted_data)), sorted_data['goles_por_corner'], 
                    color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=0.5)
        
        for i, (bar, valor, team) in enumerate(zip(bars, sorted_data['goles_por_corner'], sorted_data['EQUIPO'])):
            if team == equipo_seleccionado or 'villarreal' in team.lower():
                ax.text((valor) + 0.001, i, f'{float(valor):.3f}', va='center', fontsize=9, weight='bold')
        
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data['EQUIPO'], fontsize=8)
        ax.set_xlabel('Goles por Córner', fontsize=10, weight='bold')
        
        promedio = sorted_data['goles_por_corner'].mean()
        ax.axvline(promedio, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def plot_corner_scatter(self, ax, team_stats, equipo_seleccionado):
        """Dibuja el scatter plot: Córners vs Tiros por Córner"""
        x_data = team_stats['num_corners']  # Suma de córners a favor
        y_data = team_stats['tiros_por_corner']  # Tiros por córner
        equipos = team_stats['EQUIPO']
        
        # Calcular promedios para las líneas de la cruz
        x_promedio = x_data.mean()
        y_promedio = y_data.mean()
        
        # Dibujar líneas de la cruz
        ax.axvline(x_promedio, color='#34495e', linestyle='-', linewidth=2, alpha=0.6)
        ax.axhline(y_promedio, color='#34495e', linestyle='-', linewidth=2, alpha=0.6)
        
        # TAMAÑO UNIFORME PARA TODOS LOS ESCUDOS
        uniform_zoom = 0.24
        
        # Scatter plot con escudos
        for i, (equipo, x_val, y_val) in enumerate(zip(equipos, x_data, y_data)):
            is_selected = equipo == equipo_seleccionado
            is_villarreal = 'villarreal' in equipo.lower()
            
            print(f"Procesando {equipo}: x={x_val}, y={y_val}")
            
            # Buscar escudo por similitud
            escudo = self.find_team_logo_by_similarity(equipo)
            
            if escudo is not None:
                print(f"✅ Escudo cargado para {equipo}, shape: {escudo.shape}")
                try:
                    # APLICAR FILTRO DE COLOR Y TAMAÑO
                    if is_selected or is_villarreal:
                        # Mantener colores originales Y MÁS GRANDES
                        zoom_size = 0.45  # 33% más grande que 0.24
                        imagebox = OffsetImage(escudo, zoom=zoom_size, alpha=1.0)
                        print(f"  → Escudo en color y grande para {equipo}")
                    else:
                        # Convertir a blanco y negro Y TAMAÑO NORMAL
                        escudo_bn = self.convert_to_grayscale(escudo)
                        zoom_size = 0.24  # Tamaño normal
                        imagebox = OffsetImage(escudo_bn, zoom=zoom_size, alpha=0.8)
                        print(f"  → Escudo en B&N y normal para {equipo}")
                    
                    ab = AnnotationBbox(imagebox, (x_val, y_val), frameon=False, pad=0)
                    ax.add_artist(ab)
                    print(f"  → Escudo añadido al gráfico para {equipo}")
                    continue
                except Exception as e:
                    print(f"❌ Error al mostrar escudo para {equipo}: {e}")
            else:
                print(f"❌ No se encontró escudo para {equipo}")
            
            # Si no hay escudo, mostrar nombre del equipo
            if is_selected or is_villarreal:
                color, fontsize, weight = '#e74c3c' if is_selected else '#f39c12', 8, 'bold'
            else:
                color, fontsize, weight = '#2c3e50', 7, 'normal'
            
            ax.text(x_val, y_val, equipo, ha='center', va='center', 
                    fontsize=fontsize, weight=weight, color=color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.8))
        
        ax.set_xlabel('Suma de Córners a Favor', fontsize=12, weight='bold')
        ax.set_ylabel('Tiros por Córner', fontsize=12, weight='bold')
        
        # Asegurar que todos los puntos sean visibles
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()

        # Añadir margen del 10%
        x_margin = (x_max - x_min) * 0.1 if x_max > x_min else 1
        y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1

        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        print(f"Límites del gráfico: x=[{x_min - x_margin:.1f}, {x_max + x_margin:.1f}], y=[{y_min - y_margin:.1f}, {y_max + y_margin:.1f}]")

        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def verificar_datos_disponibles(data_path="extraccion_mediacoach/data/estadisticas_equipo.parquet"):
    """Función para verificar qué datos están disponibles"""
    try:
        if not os.path.exists(data_path):
            print(f"❌ Error: No se encontró el archivo en la ruta: {data_path}")
            print("Verifica que la ruta sea correcta y que el archivo exista.")
            print("\nVerificando rutas alternativas...")
            
            # Verificar rutas alternativas comunes
            alternative_paths = [
                "estadisticas_equipo.parquet",
                "data/estadisticas_equipo.parquet", 
                "extraccion_mediacoach/estadisticas_equipo.parquet",
                "./estadisticas_equipo.parquet"
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    print(f"✅ Archivo encontrado en: {alt_path}")
                    data_path = alt_path
                    break
            else:
                return None
        
        df = pd.read_parquet(data_path)
        print(f"=== ANÁLISIS DEL DATASET ===")
        print(f"Archivo: {data_path}")
        print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
        print(f"\nColumnas disponibles:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        print(f"\nPrimeras 3 filas:")
        print(df.head(3))
        
        # Buscar métricas relacionadas con córners
        if 'NOMBRE MÉTRICA' in df.columns:
            corner_metrics = df[df['NOMBRE MÉTRICA'].str.contains('esquina|córner|corner', case=False, na=False)]['NOMBRE MÉTRICA'].unique()
            if len(corner_metrics) > 0:
                print(f"\nMétricas relacionadas con córners encontradas:")
                for metric in corner_metrics:
                    print(f"- {metric}")
            else:
                print(f"\n❌ No se encontraron métricas relacionadas con córners")
                
            # Mostrar algunas métricas como ejemplo
            print(f"\nEjemplo de métricas disponibles:")
            for metric in df['NOMBRE MÉTRICA'].unique()[:10]:
                print(f"- {metric}")
        
        return df
    except Exception as e:
        print(f"❌ Error al verificar datos: {e}")
        return None

def seleccionar_equipo_interactivo(df_teams):
    """
    Función para seleccionar equipo interactivamente desde el DataFrame de equipos de Opta.
    """
    try:
        # La columna de equipos en los datos de Opta se llama 'Team Name'
        equipos = sorted(df_teams['Team Name'].dropna().unique())
        if not equipos: 
            print("No se encontraron equipos.")
            return None
        
        print("\n=== SELECCIÓN DE EQUIPO ===")
        for i, equipo in enumerate(equipos, 1):
            # La columna de competición en los datos de Opta se llama 'Competition Name'
            liga = df_teams[df_teams['Team Name'] == equipo]['Competition Name'].iloc[0]
            print(f"{i}. {equipo} ({liga})")
        
        while True:
            try:
                seleccion = input(f"\nSelecciona un equipo (1-{len(equipos)}): ").strip()
                indice = int(seleccion) - 1
                if 0 <= indice < len(equipos):
                    return equipos[indice]
                else:
                    print(f"Por favor, ingresa un número entre 1 y {len(equipos)}")
            except (ValueError, IndexError):
                print("Por favor, ingresa un número válido.")
    except Exception as e:
        print(f"Error en la selección de equipo: {e}")
        return None

def main():
    """Función principal para ejecutar el reporte con la nueva lógica de Opta."""
    print("=== GENERADOR DE REPORTES DE CÓRNERS OFENSIVOS (DATOS OPTA) ===")
    
    # 1. Crear la instancia. Esto automáticamente carga los datos de Opta (eventos y equipos).
    report_generator = CornersOfensivosReport()
    
    # Si la carga de datos falló, no continuar.
    if report_generator.df_teams is None:
        print("❌ No se pudieron cargar los datos de equipos. Terminando ejecución.")
        return
        
    # 2. Seleccionar el equipo usando el DataFrame de equipos de Opta.
    equipo_seleccionado = seleccionar_equipo_interactivo(report_generator.df_teams)
    if equipo_seleccionado is None:
        return

    print(f"\nGenerando reporte para: {equipo_seleccionado}")
    
    # 3. Crear la visualización (esta parte no cambia)
    fig = report_generator.create_visualization(equipo_seleccionado)
    
    if fig:
        plt.show()
        
        # Guardar como PDF
        output_path = "reporte_corners_ofensivos.pdf"
        from matplotlib.backends.backend_pdf import PdfPages
        fig.set_size_inches(11.69, 8.27) 

        with PdfPages(output_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight', pad_inches=0, dpi=300)
        
        print(f"✅ Reporte guardado como: {output_path}")
    else:
        print("❌ No se pudo generar la visualización")

if __name__ == "__main__":
    main()