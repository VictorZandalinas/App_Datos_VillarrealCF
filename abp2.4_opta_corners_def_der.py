import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import patheffects
import seaborn as sns
import numpy as np
import os
import re
import base64
from io import BytesIO
from PIL import Image
from mplsoccer import VerticalPitch
from difflib import SequenceMatcher
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

class LanzamientosLadoDerecho:
    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/abp_events.parquet", team_filter=None):
        self.data_path = data_path
        self.team_filter = team_filter
        self.df = None
        self.lanzamientos_data = pd.DataFrame()
        self.team_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/team_stats.parquet")
        self.player_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/player_stats.parquet")
        self.load_data(team_filter)
        
        # ← AÑADIR ESTAS LÍNEAS:
        if team_filter:
            self.extract_lanzamientos_derecha(team_filter)
    
    def normalize_timestamp(self, timestamp):
        """Normaliza timestamps quitando la Z final si existe"""
        if pd.isna(timestamp):
            return timestamp
        
        timestamp_str = str(timestamp).strip()
        
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1]
        
        try:
            dt = pd.to_datetime(timestamp_str)
            return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        except:
            return timestamp_str

    def get_player_shirt_number_by_name(self, player_name):
        """Obtiene el dorsal del jugador por nombre"""
        if pd.isna(player_name):
            return None
        
        player_info = self.player_stats[self.player_stats['Match Name'] == player_name]
        if not player_info.empty:
            shirt_number = player_info['Shirt Number'].iloc[0]
            return str(int(shirt_number)) if pd.notna(shirt_number) else None
        return None

    def load_data(self, team_filter=None):
        """Carga los datos necesarios desde los parquets"""
        try:
            columns_needed = ['Match ID', 'periodId', 'Team ID', 'Team Name', 'Event Name', 'outcome', 
                'timeMin', 'timeSec', 'x', 'y', 'Pass End X', 'Pass End Y', 
                'playerName', 'playerId', 'Corner taken', 
                'Throw in', 'Free kick taken', 'timeStamp']
            
            self.df = pd.read_parquet(self.data_path, columns=columns_needed)
            self.df['timeStamp'] = self.df['timeStamp'].apply(self.normalize_timestamp)
            
            # CAMBIO: Filtrar por equipos rivales en los mismos matches
            if team_filter:
                team_matches = self.team_stats[self.team_stats['Team Name'] == team_filter]['Match ID'].unique()
                # Filtrar: mismo Match ID pero equipo diferente
                self.df = self.df[
                    (self.df['Match ID'].isin(team_matches)) & 
                    (self.df['Team Name'] != team_filter)
                ]
            
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")
    
    def extract_lanzamientos_derecha(self, team_filter=None):
        """Extrae lanzamientos del lado derecho con análisis de secuencia"""
        if self.df is None:
            print("❌ No hay datos cargados")
            return
        
        
        # Ordenar datos una sola vez
        df_sorted = self.df.sort_values(['Match ID', 'timeMin', 'timeSec']).reset_index(drop=True)
        
        if team_filter:
            team_matches = set(self.team_stats[
                self.team_stats['Team Name'] == team_filter
            ]['Match ID'].unique())
        
        lanzamientos_list = []
        
        # Procesar por match
        for match_id in df_sorted['Match ID'].unique():
            if team_filter and match_id not in team_matches:
                continue
                
            match_events = df_sorted[df_sorted['Match ID'] == match_id].reset_index(drop=True)
            
            # Filtrar pases del lado derecho (y < 1)
            lanzamientos_der = match_events[
                (match_events.get('Corner taken', '') == 'Sí') &
                (match_events['y'] < 1) &
                (match_events['x'].notna()) & 
                (match_events['y'].notna())
            ]
            
            for lanz_idx, lanzamiento in lanzamientos_der.iterrows():
                # Analizar secuencia
                result_data = self.analyze_lanzamiento_sequence(
                    match_events, lanz_idx, lanzamiento
                )
                
                # Construir datos del lanzamiento
                lanzamiento_data = {
                    'Match ID': match_id,
                    'Team ID': lanzamiento['Team ID'],
                    'Team Name': lanzamiento['Team Name'],
                    'playerName': lanzamiento['playerName'],
                    'playerId': lanzamiento['playerId'],
                    'timeMin': lanzamiento['timeMin'],
                    'timeSec': lanzamiento['timeSec'],
                    'x': lanzamiento['x'],
                    'y': lanzamiento['y'],
                    'Pass End X': lanzamiento['Pass End X'],
                    'Pass End Y': lanzamiento['Pass End Y']
                }
                
                lanzamiento_data.update(result_data)
                lanzamientos_list.append(lanzamiento_data)
        # Debug: Verificar qué hay en los datos
        
        # Verificar valores únicos en Corner taken
        corner_values = df_sorted['Corner taken'].value_counts(dropna=False)
        
        # Verificar si hay corners en general
        corners_encontrados = df_sorted[df_sorted.get('Corner taken', '').notna()]
        
        # Verificar coordenadas Y para eventos con corners
        if not corners_encontrados.empty:
            y_values = corners_encontrados['y'].describe()
            
            # Verificar específicamente si hay y < 1
            lado_der_y = corners_encontrados[corners_encontrados['y'] < 1]

        # Crear DataFrame final
        if lanzamientos_list:
            self.lanzamientos_data = pd.DataFrame(lanzamientos_list)
            
            # AÑADIR ESTA LÍNEA PARA ELIMINAR DUPLICADOS:
            self.lanzamientos_data = self.lanzamientos_data.drop_duplicates(subset=['Match ID', 'timeMin', 'timeSec'], keep='first')
            
            
            
        else:
            print("❌ No se encontraron lanzamientos")
        
            # Test diferentes valores para Corner taken
            for test_value in ['Sí', 'Si', 'YES', 'Yes', 'yes', '1', True]:
                test_corners = df_sorted[df_sorted.get('Corner taken', '') == test_value]
                if not test_corners.empty:
                    pass


    def debug_lanzamientos(self, team_filter=None):
        """Debug detallado de lanzamientos extraídos"""
        if self.lanzamientos_data.empty:
            pass
            return
        
        team_data = self.lanzamientos_data
        
        for idx, row in team_data.iterrows():
            pass
            if row['goal_player']:
                dorsal = self.get_player_shirt_number_by_name(row['goal_player'])
           
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
        pass_count = 0  # ← NUEVO: Contador de pases
        
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
                pass_count += 1
                
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
        
    def determine_corner_side(self, x, y):
        """Determina el lado del lanzamiento"""
        y = float(y)
        return 'derecha' if y < 50 else 'izquierda'
    
    def get_player_shirt_number(self, player_id):
        """Obtiene el número de camiseta del jugador"""
        if pd.isna(player_id):
            return None
        
        player_info = self.player_stats[self.player_stats['Player ID'] == player_id]
        if not player_info.empty:
            shirt_number = player_info['Shirt Number'].iloc[0]
            return str(int(shirt_number)) if pd.notna(shirt_number) else None
        return None
    
    def load_team_logo(self, equipo, target_size=(80, 80)):
        """Carga y redimensiona el logo del equipo"""
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            print("⚠️ PIL no está instalado. Usando método original sin redimensión.")
            return self._load_team_logo_original(equipo)
        
        possible_names = [
            equipo, equipo.replace(' ', '_'), equipo.replace(' ', ''),
            equipo.lower(), equipo.lower().replace(' ', '_'), equipo.lower().replace(' ', '')
        ]
        
        logo_path = None
        # Buscar archivo exacto
        for name in possible_names:
            path = f"assets/escudos/{name}.png"
            if os.path.exists(path): 
                logo_path = path
                break
        
        # Si no encuentra, buscar por similitud
        if not logo_path and os.path.exists('assets/escudos'):
            all_files = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
            best_match, best_score = None, 0
            for filename in all_files:
                name_without_ext = os.path.splitext(filename)[0]
                score = SequenceMatcher(None, equipo.lower(), name_without_ext.lower()).ratio()
                if score > best_score:
                    best_score, best_match = score, filename
            if best_match and best_score > 0.6:
                logo_path = f"assets/escudos/{best_match}"
        
        if logo_path:
            try:
                with Image.open(logo_path) as img:
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    final_img = Image.new('RGBA', target_size, (0, 0, 0, 0))
                    paste_x = (target_size[0] - img.width) // 2
                    paste_y = (target_size[1] - img.height) // 2
                    final_img.paste(img, (paste_x, paste_y), img)
                    return np.array(final_img) / 255.0
            except Exception as e:
                print(f"⚠️ Error procesando {logo_path}: {e}")
                return self._load_team_logo_original(equipo)
        
        return None
    
    def _load_team_logo_original(self, equipo):
        """Método original como fallback"""
        possible_names = [
            equipo, equipo.replace(' ', '_'), equipo.replace(' ', ''),
            equipo.lower(), equipo.lower().replace(' ', '_'), equipo.lower().replace(' ', '')
        ]
        for name in possible_names:
            logo_path = f"assets/escudos/{name}.png"
            if os.path.exists(logo_path): 
                return plt.imread(logo_path)
        
        if os.path.exists('assets/escudos'):
            all_files = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
            best_match, best_score = None, 0
            for filename in all_files:
                name_without_ext = os.path.splitext(filename)[0]
                score = SequenceMatcher(None, equipo.lower(), name_without_ext.lower()).ratio()
                if score > best_score:
                    best_score, best_match = score, filename
            if best_match and best_score > 0.6:
                return plt.imread(f"assets/escudos/{best_match}")
        return None
    
    def load_ball_image(self): 
        return plt.imread("assets/balon.png") if os.path.exists("assets/balon.png") else None
    
    def load_background(self): 
        return plt.imread("assets/fondo_informes.png") if os.path.exists("assets/fondo_informes.png") else None
    
    def get_outcome_marker(self, result_type):
        marker_map = {
            'Sin remate': 'v',
            'Tiro a puerta': 's',
            'Tiro fuera': 'X',
            'Tiro al poste': 'P',  # Nuevo
            'Gol': 'o',
            'Otro contacto': 'D'
        }
        return marker_map.get(result_type, 'o')

    def get_outcome_color(self, result_type):
        color_map = {
            'Sin remate': '#95A5A6',
            'Tiro a puerta': '#3498DB', 
            'Tiro fuera': '#E74C3C',
            'Tiro al poste': '#FF6B35',  # Nuevo naranja
            'Gol': '#F1C40F',
            'Otro contacto': '#9B59B6'
        }
        return color_map.get(result_type, '#95A5A6')
    
    def load_player_photos(self):
        """Carga el JSON con las fotos de jugadores"""
        import json
        try:
            with open('assets/jugadores_optimizados.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ No se encontró el archivo jugadores_optimizados.json")
            return []

    def get_player_photo_without_dorsal(self, player_name, photos_data):
        """Obtiene la foto sin fondo blanco pero SIN dorsal"""
        match = self.match_player_name(player_name, photos_data, self.team_filter)
        if not match:
            return None
        
        try:
            # Decodificar base64 y convertir a imagen
            img_data = base64.b64decode(match['image_base64'])
            img = Image.open(BytesIO(img_data))
            
            # Asegurar que sea RGBA
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            data = np.array(img)
            
            # Verificar dimensiones
            if len(data.shape) != 3 or data.shape[2] != 4:
                return None
            
            height, width = data.shape[:2]
            
            # Flood fill ITERATIVO para evitar recursión infinita
            def flood_fill_iterative(start_points, threshold=235):
                visited = np.zeros((height, width), dtype=bool)
                background_mask = np.zeros((height, width), dtype=bool)
                
                def is_background_color(y, x):
                    if y < 0 or y >= height or x < 0 or x >= width:
                        return False
                    return (data[y, x, 0] >= threshold and 
                            data[y, x, 1] >= threshold and 
                            data[y, x, 2] >= threshold)
                
                # Usar pila en lugar de recursión
                for start_y, start_x in start_points:
                    if visited[start_y, start_x] or not is_background_color(start_y, start_x):
                        continue
                    
                    # Pila para flood fill iterativo
                    stack = [(start_y, start_x)]
                    
                    while stack:
                        y, x = stack.pop()
                        
                        if (y < 0 or y >= height or x < 0 or x >= width or 
                            visited[y, x] or not is_background_color(y, x)):
                            continue
                        
                        visited[y, x] = True
                        background_mask[y, x] = True
                        
                        # Añadir vecinos a la pila (4-connected es más estable)
                        stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
                
                return background_mask
            
            # Puntos de inicio: solo esquinas y algunos puntos de borde
            border_points = [
                (0, 0), (0, width-1), (height-1, 0), (height-1, width-1),  # Esquinas
                (0, width//2), (height-1, width//2),  # Centro superior e inferior
                (height//2, 0), (height//2, width-1)   # Centro izquierda y derecha
            ]
            
            # Aplicar flood fill
            background_mask = flood_fill_iterative(border_points, threshold=230)
            
            # Aplicar la máscara para hacer transparente el fondo
            data[background_mask] = [0, 0, 0, 0]
            
            return data.astype(np.float32) / 255.0
        
        except Exception as e:
            print(f"⚠️ Error procesando foto de {player_name}: {e}")
            return None

    def get_player_photo_with_team_filter(self, player_name, photos_data, team_filter):
        """Wrapper para obtener foto con filtro de equipo - usa el mismo algoritmo mejorado"""
        match = self.match_player_name(player_name, photos_data, team_filter)
        if not match:
            return None
        
        try:
            # Decodificar base64 y convertir a imagen
            img_data = base64.b64decode(match['image_base64'])
            img = Image.open(BytesIO(img_data))
            
            # Asegurar que sea RGBA
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            data = np.array(img)
            
            # Verificar dimensiones
            if len(data.shape) != 3 or data.shape[2] != 4:
                return None
            
            height, width = data.shape[:2]
            
            # Flood fill ITERATIVO (mismo algoritmo que arriba)
            def flood_fill_iterative(start_points, threshold=235):
                visited = np.zeros((height, width), dtype=bool)
                background_mask = np.zeros((height, width), dtype=bool)
                
                def is_background_color(y, x):
                    if y < 0 or y >= height or x < 0 or x >= width:
                        return False
                    return (data[y, x, 0] >= threshold and 
                            data[y, x, 1] >= threshold and 
                            data[y, x, 2] >= threshold)
                
                # Usar pila en lugar de recursión
                for start_y, start_x in start_points:
                    if visited[start_y, start_x] or not is_background_color(start_y, start_x):
                        continue
                    
                    # Pila para flood fill iterativo
                    stack = [(start_y, start_x)]
                    
                    while stack:
                        y, x = stack.pop()
                        
                        if (y < 0 or y >= height or x < 0 or x >= width or 
                            visited[y, x] or not is_background_color(y, x)):
                            continue
                        
                        visited[y, x] = True
                        background_mask[y, x] = True
                        
                        # Añadir vecinos a la pila (4-connected)
                        stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
                
                return background_mask
            
            # Puntos de inicio: solo esquinas y algunos puntos de borde
            border_points = [
                (0, 0), (0, width-1), (height-1, 0), (height-1, width-1),  # Esquinas
                (0, width//2), (height-1, width//2),  # Centro superior e inferior
                (height//2, 0), (height//2, width-1)   # Centro izquierda y derecha
            ]
            
            # Aplicar flood fill
            background_mask = flood_fill_iterative(border_points, threshold=230)
            
            # Aplicar la máscara para hacer transparente el fondo
            data[background_mask] = [0, 0, 0, 0]
            
            return data.astype(np.float32) / 255.0
        
        except Exception as e:
            print(f"⚠️ Error procesando foto de {player_name}: {e}")
            return None
        
    def match_player_name(self, player_name, photos_data, team_filter=None):
        """
        Encuentra el nombre más parecido en los datos de las fotos con una lógica mejorada
        para manejar apodos, iniciales y nombres compuestos, evitando ambigüedades.
        """
        
        def normalize_name(name):
            """Normaliza el nombre eliminando acentos, puntuación y convirtiendo a minúsculas."""
            name = name.lower().strip()
            replacements = {
                'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
                'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
                'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
                'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
                'ñ': 'n', 'ç': 'c'
            }
            for old, new in replacements.items():
                name = name.replace(old, new)
            name = re.sub(r'[^\w\s]', '', name)
            return ' '.join(name.split())

        def extract_names_parts(name):
            """Extrae las partes de un nombre normalizado."""
            normalized = normalize_name(name)
            parts = normalized.split()
            
            if not parts:
                return {'full': '', 'first_name': '', 'last_name': '', 'all_parts': []}
            
            return {
                'full': normalized,
                'first_name': parts[0],
                'last_name': parts[-1] if len(parts) > 1 else parts[0],
                'all_parts': parts
            }

        def calculate_match_score(player_parts, photo_parts):
            """
            Calcula el score de coincidencia con reglas específicas para casos comunes.
            Devuelve una tupla (score, razón_del_match).
            """
            player_full = player_parts['full']
            photo_full = photo_parts['full']

            # 1. MATCH EXACTO COMPLETO - Máxima prioridad
            if player_full == photo_full:
                return (1.0, "MATCH EXACTO COMPLETO")

            # 2. MATCH DE NOMBRE ÚNICO / APODO (ej: 'Gavi' en 'Pablo Martin Paez Gaviria')
            # Si el nombre a buscar es una sola palabra.
            if len(player_parts['all_parts']) == 1:
                if player_full in photo_parts['all_parts']:
                    return (0.95, f"MATCH DE NOMBRE ÚNICO '{player_full}'")

            # 3. MATCH DE INICIAL + APELLIDO (ej: 'G. Lopez' vs 'Gerard Lopez')
            if (len(player_parts['first_name']) == 1 and
                player_parts['last_name'] == photo_parts['last_name'] and
                photo_parts['first_name'].startswith(player_parts['first_name'])):
                return (0.90, "INICIAL + APELLIDO EXACTO")

            # 4. MATCH DE NOMBRE + INICIAL DE APELLIDO (ej: 'Gerard M' vs 'Gerard Martin')
            if (player_parts['first_name'] == photo_parts['first_name'] and
                len(player_parts['last_name']) == 1 and
                photo_parts['last_name'].startswith(player_parts['last_name'][0])):
                return (0.90, "NOMBRE + INICIAL APELLIDO")

            # 5. MATCH DE APELLIDO EXACTO Y NOMBRE SIMILAR
            if player_parts['last_name'] == photo_parts['last_name']:
                first_name_sim = SequenceMatcher(None, player_parts['first_name'], photo_parts['first_name']).ratio()
                if first_name_sim >= 0.8:
                    return (0.85 + (first_name_sim * 0.05), f"APELLIDO EXACTO + NOMBRE SIMILAR ({first_name_sim:.2f})")
            
            # 6. Fallback a similitud general (menos prioritario)
            full_sim = SequenceMatcher(None, player_full, photo_full).ratio()
            if full_sim > 0.8:
                return (full_sim, f"SIMILITUD GENERAL ALTA ({full_sim:.2f})")

            return (0.0, "SIN COINCIDENCIA CLARA")

        # --- Lógica principal de la función ---
        
        player_parts = extract_names_parts(player_name)
        if not player_parts['full']:
            return None

        
        found_matches = []
        
        for photo_entry in photos_data:
            photo_name = photo_entry.get('player_name')
            if not photo_name:
                continue
                
            photo_parts = extract_names_parts(photo_name)
            score, reason = calculate_match_score(player_parts, photo_parts)
            
            # Consideramos un match potencial si el score es suficientemente alto
            if score >= 0.90:
                found_matches.append({
                    "entry": photo_entry,
                    "score": score,
                    "reason": reason
                })
                
        # --- Lógica de desambiguación ---
        if len(found_matches) == 1:
            best_match = found_matches[0]
            return best_match['entry']
        
        elif len(found_matches) > 1:
            # Si hay múltiples matches con score alto, es ambiguo.
            print(f"⚠️  ADVERTENCIA: Se encontraron {len(found_matches)} matches de alta calidad para '{player_name}'. Se descarta por ambigüedad.")
            for match in sorted(found_matches, key=lambda x: x['score'], reverse=True):
                pass
            return None
            
        else:
            print(f"❌ NO SE ENCONTRÓ UN MATCH DE ALTA CONFIANZA para '{player_name}'")
            return None

    def get_aerial_ranking_data(self, team_filter=None):
        """Obtiene datos de ranking de equipos que más remates han creado"""
        from collections import Counter
        
        if self.lanzamientos_data.empty:
            return Counter()
        
        # Ya no filtrar por equipo porque son datos de rivales
        # Filtrar solo lanzamientos que resultaron en remate
        team_data = self.lanzamientos_data[
            self.lanzamientos_data['result_type'].isin(['Gol', 'Tiro a puerta', 'Tiro fuera', 'Tiro al poste'])
        ]
        
        # Contar por EQUIPO en lugar de jugador
        ranking = Counter(team_data['Team Name'].dropna())
        return ranking

    def create_aerial_ranking(self, ax, team_filter):
        """Crea el ranking visual de equipos que más remates han creado"""
        aerial_data = self.get_aerial_ranking_data(team_filter)
        
        if not aerial_data:
            ax.text(0.5, 0.5, 'RANKING EQUIPOS\nCON MÁS REMATES\n\n(Sin datos)', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        ax.set_facecolor('#f8f9fa')
        top_teams = aerial_data.most_common(5)  # Top 5 equipos
        
        # Título
        ax.text(0.5, 0.97, 'RANKING EQUIPOS\nCON MÁS REMATES', fontsize=12, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes, color='#2c3e50')
        
        total_remates = sum(aerial_data.values())
        position_colors = ['#e74c3c', '#f39c12', '#f1c40f', '#95a5a6', '#bdc3c7']
        
        for i, (team_name, remate_count) in enumerate(top_teams):
            y_pos = 0.80 - (i * 0.15)
            
            # Fondo del equipo
            rect_bg = patches.FancyBboxPatch((0.02, y_pos - 0.06), 0.96, 0.12,
                                            boxstyle="round,pad=0.01", 
                                            facecolor='white',
                                            edgecolor=position_colors[i],
                                            linewidth=2, alpha=0.9)
            ax.add_patch(rect_bg)
            
            # Número de posición
            ax.text(0.08, y_pos, f"#{i+1}", fontsize=14, fontweight='bold', 
                    va='center', ha='center', color=position_colors[i])
            
            # CAMBIO: Escudo del equipo en lugar de foto del jugador
            team_logo = self.load_team_logo(team_name)
            if team_logo is not None:
                logo_ax = ax.inset_axes([0.12, y_pos - 0.078, 0.25, 0.16])
                logo_ax.imshow(team_logo, aspect='auto')
                logo_ax.axis('off')
            
            # Nombre del equipo
            display_name = team_name[:12] + '...' if len(team_name) > 12 else team_name
            ax.text(0.45, y_pos + 0.02, display_name, fontsize=8, fontweight='bold', 
                    va='center', ha='left', color='#2c3e50')
            
            # Estadística
            ax.text(0.45, y_pos - 0.02, f'{remate_count} remates', 
                    fontsize=10, va='center', ha='left', color='#7f8c8d')
            
            # Sistema de estrellas
            if total_remates > 0:
                percentage = (remate_count / total_remates) * 100
                stars = min(5, max(1, int(percentage / 10)))
                star_text = '★' * stars
                ax.text(0.45, y_pos - 0.05, star_text, fontsize=14, va='center', ha='left',
                        color='#f39c12')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def create_lanzamientos_visualization(self, figsize=(11.69, 8.27), team_filter=None):  # A4 horizontal
        """Crea la visualización con subplots según especificaciones"""
        
        if self.lanzamientos_data.empty:
            print("❌ No hay datos de lanzamientos para visualizar")
            return None
        
        # Filtrar datos del equipo
        team_data = self.lanzamientos_data
        
        if team_data.empty:
            print(f"❌ No hay datos de lanzamientos para {team_filter}")
            return None
        
        # Obtener top 3 equipos con más centros
        remates_exitosos = team_data[
            team_data['result_type'].isin(['Gol', 'Tiro a puerta', 'Tiro fuera', 'Tiro al poste'])
        ]
        top_equipos_remates = remates_exitosos['Team Name'].value_counts().head(3)
        
        # Crear figura
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        # Fondo
        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25)
            ax_bg.axis('off')
        
        # Título principal
        fig.suptitle('CORNERS DEFENSIVOS - LADO DERECHO', 
                     fontsize=20, fontweight='bold', color='#1e3d59', y=0.95, family='serif')
        
        # Logos superiores
        if (ball := self.load_ball_image()) is not None:
            ax_ball = fig.add_axes([0.05, 0.90, 0.06, 0.06])
            ax_ball.imshow(ball)
            ax_ball.axis('off')
        
        if team_filter and (team_logo := self.load_team_logo(team_filter)) is not None:
            ax_team = fig.add_axes([0.85, 0.88, 0.12, 0.12])
            ax_team.imshow(team_logo, aspect='auto')
            ax_team.axis('off')
        
        # Configurar grid de subplots: 2 filas, 5 columnas para más ancho arriba
        # Fila 1: 1 subplot que ocupa las 5 columnas (más ancho)
        # Fila 2: 3 subplots centrados (columnas 1, 2, 3)
        # Gridspec principal solo para las filas
        gs_main = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.2,
                                left=0.02, right=0.98, top=0.85, bottom=0.05)

        # CAMBIO: Intercambiar posiciones - campo a la izquierda (3.0) y ranking a la derecha (0.8)
        gs_top = gs_main[0].subgridspec(1, 2, width_ratios=[3.0, 0.8], wspace=0.1)

        # Subgridspec para fila inferior: 3 columnas iguales
        gs_bottom = gs_main[1].subgridspec(1, 3, wspace=0.1)  
        # SUBPLOT SUPERIOR: Campo general con todos los lanzamientos del lado derecho
        ax_general = fig.add_subplot(gs_top[0])  # CAMBIO: Ahora es [0] (izquierda)
        pitch_general = VerticalPitch(half=True, pitch_type='opta', pitch_color='#2d5a27', 
                                     line_color='white', linewidth=2)
        pitch_general.draw(ax=ax_general)
        ax_general.set_aspect('auto')

        
        ax_general.set_title(f'LANZAMIENTOS GENERALES - LADO DERECHO ({len(team_data)} total)', 
                           fontsize=14, fontweight='bold', color='#1e3d59', pad=10, family='serif')
        
        # Plotear todos los lanzamientos por tipo de resultado
        for result_type in team_data['result_type'].unique():
            data_tipo = team_data[team_data['result_type'] == result_type]
            if len(data_tipo) > 0:
                if result_type == 'Gol':
                    # Para goles, mostrar escudo del equipo que hizo el gol
                    for _, gol_data in data_tipo.iterrows():
                        if gol_data.get('is_second_play') and gol_data.get('previous_event_coords'):
                            prev_x, prev_y = gol_data['previous_event_coords']
                            ax_general.plot([prev_y, gol_data['final_y']], [prev_x, gol_data['final_x']], 
                            color=self.get_outcome_color(result_type), linestyle='--', linewidth=1.5, alpha=0.8)

                        if gol_data['goal_player']:
                            dorsal = self.get_player_shirt_number_by_name(gol_data['goal_player'])
                            
                            # CAMBIO: Escudo del equipo que hizo el gol
                            equipo_goleador = gol_data['Team Name']
                            team_logo = self.load_team_logo(equipo_goleador)
                            if team_logo is not None:
                                logo_box = OffsetImage(team_logo, zoom=0.28)
                                ab_logo = AnnotationBbox(logo_box, (gol_data['final_y'], gol_data['final_x']), 
                                                    frameon=False, zorder=8)
                                ax_general.add_artist(ab_logo)

                            # Dorsal ENCIMA del escudo
                            if dorsal:
                                ax_general.text(gol_data['final_y'], gol_data['final_x'], dorsal,
                                    fontsize=12, fontweight='bold', color='white',
                                    ha='center', va='center', zorder=15,
                                    path_effects=[patheffects.withStroke(linewidth=4, foreground='black')])
                else:
                    # LÍNEAS DISCONTINUAS PARA NO-GOLES:
                    for _, remate_data in data_tipo.iterrows():
                        if remate_data.get('is_second_play') and remate_data.get('previous_event_coords'):
                            prev_x, prev_y = remate_data['previous_event_coords']
                            ax_general.plot([prev_y, remate_data['final_y']], [prev_x, remate_data['final_x']], 
                            color=self.get_outcome_color(result_type), linestyle='--', linewidth=1.5, alpha=0.8)
                            
                    # Para no-goles, scatter normal
                    ax_general.scatter(data_tipo['final_y'], data_tipo['final_x'], 
                                    c=self.get_outcome_color(result_type), 
                                    marker=self.get_outcome_marker(result_type),
                                    s=50, 
                                    alpha=0.7,
                                    edgecolors='white',
                                    linewidth=1,
                                    label=f'{result_type} ({len(data_tipo)})')

        # Balón en el campograma superior
        if (ball := self.load_ball_image()) is not None:
            ball_box = OffsetImage(ball, zoom=0.06)
            ab_ball = AnnotationBbox(ball_box, (0.5, 99.5), 
                            frameon=False, zorder=5)
            ax_general.add_artist(ab_ball)

        
        # Crear leyenda personalizada
        legend_elements = []

        for result_type in team_data['result_type'].unique():
            data_tipo = team_data[team_data['result_type'] == result_type]
            if len(data_tipo) > 0:
                if result_type == 'Gol':
                    # Para goles, usar un marcador especial que represente el escudo
                    legend_elements.append(plt.Line2D([0], [0], 
                                                    marker='H', 
                                                    color='w', 
                                                    markerfacecolor='#F1C40F',
                                                    markeredgecolor='#E67E22',
                                                    markeredgewidth=2,
                                                    markersize=12,
                                                    label=f'Gol con dorsal ({len(data_tipo)})'))
                else:
                    legend_elements.append(plt.Line2D([0], [0], 
                                                    marker=self.get_outcome_marker(result_type), 
                                                    color='w', 
                                                    markerfacecolor=self.get_outcome_color(result_type),
                                                    markersize=8,
                                                    label=f'{result_type} ({len(data_tipo)})'))

        if legend_elements:
            ax_general.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.25), 
                            framealpha=0.9, facecolor='white', edgecolor='#1e3d59')
            

        # RANKING: CAMBIO - Ahora está a la derecha [1]
        ax_ranking = fig.add_subplot(gs_top[1])
        self.create_aerial_ranking(ax_ranking, team_filter)
        
        # SUBPLOTS INFERIORES: Top 3 jugadores ocupando todo el ancho
        top_3_equipos = top_equipos_remates.index[:3].tolist()  # Top 3 equipos con más remates
        photos_data = self.load_player_photos()  # Ya no se usa, pero déjalo

        for i, equipo in enumerate(top_3_equipos):  # Equipos con más remates exitosos
            if i >= 3:
                break
                
            ax = fig.add_subplot(gs_bottom[i])
            pitch = VerticalPitch(half=True, pitch_type='opta', pitch_color='#2d5a27', 
                                line_color='white', linewidth=2)
            pitch.draw(ax=ax)
            ax.set_aspect('auto')
            
            data_equipo_remates = remates_exitosos[remates_exitosos['Team Name'] == equipo]
            
            # CAMBIO: Contar tipos específicos para el título con desglose detallado
            goles = len(data_equipo_remates[data_equipo_remates['result_type'] == 'Gol'])
            tiros_puerta = len(data_equipo_remates[data_equipo_remates['result_type'] == 'Tiro a puerta'])
            tiros_fuera = len(data_equipo_remates[data_equipo_remates['result_type'] == 'Tiro fuera'])
            tiros_poste = len(data_equipo_remates[data_equipo_remates['result_type'] == 'Tiro al poste'])

            total_remates = goles + tiros_puerta + tiros_fuera + tiros_poste

            ax.set_title(f'{equipo}\nRemates: {total_remates}\nG:{goles} TP:{tiros_puerta} TF:{tiros_fuera} P:{tiros_poste}',
                        fontsize=9, fontweight='bold', color='#1e3d59', 
                        pad=4, family='serif')
            
            # Plotear lanzamientos del jugador
            if len(data_equipo_remates) > 0:
                for result_type in data_equipo_remates['result_type'].unique():
                    data_result = data_equipo_remates[data_equipo_remates['result_type'] == result_type]
                    
                    if result_type == 'Gol':
                        # Para goles, mostrar escudo del equipo que hizo el gol
                        for _, gol_data in data_result.iterrows():
                            if gol_data.get('is_second_play') and gol_data.get('previous_event_coords'):
                                prev_x, prev_y = gol_data['previous_event_coords']
                                ax.plot([prev_y, gol_data['final_y']], [prev_x, gol_data['final_x']], 
                                        color=self.get_outcome_color(result_type), linestyle='--', linewidth=1.5, alpha=0.9)
                            if gol_data['goal_player']:
                                dorsal = self.get_player_shirt_number_by_name(gol_data['goal_player'])
                                
                                # CAMBIO: Escudo del equipo que hizo el gol
                                equipo_goleador = gol_data['Team Name']
                                team_logo = self.load_team_logo(equipo_goleador)
                                if team_logo is not None:
                                    logo_box = OffsetImage(team_logo, zoom=0.35)
                                    ab_logo = AnnotationBbox(logo_box, (gol_data['final_y'], gol_data['final_x']), 
                                                        frameon=False, zorder=8)
                                    ax.add_artist(ab_logo)

                                # Dorsal ENCIMA del escudo
                                if dorsal:
                                    ax.text(gol_data['final_y'], gol_data['final_x'], dorsal, 
                                        fontsize=14, fontweight='bold', color='white',
                                        ha='center', va='center', zorder=15,
                                        path_effects=[patheffects.withStroke(linewidth=4, foreground='black')])
                    else:
                        # LÍNEAS DISCONTINUAS PRIMERO:
                        for _, remate_data in data_result.iterrows():  # ✅ data_result, no data_tipo
                            if remate_data.get('is_second_play') and remate_data.get('previous_event_coords'):
                                prev_x, prev_y = remate_data['previous_event_coords']
                                ax.plot([prev_y, remate_data['final_y']], [prev_x, remate_data['final_x']], 
                                        color=self.get_outcome_color(result_type), linestyle='--', linewidth=1.5, alpha=0.9)
                        
                        # Para no-goles, dibujar scatter normal
                        ax.scatter(data_result['final_y'], data_result['final_x'], 
                                c=self.get_outcome_color(result_type), 
                                marker=self.get_outcome_marker(result_type),
                                s=35, alpha=0.8, edgecolors='white', linewidth=1.5)
                        
            # Balón en las coordenadas especificadas (lado derecho)
            if (ball := self.load_ball_image()) is not None:
                ball_box = OffsetImage(ball, zoom=0.04)
                ab_ball = AnnotationBbox(ball_box, (0.5, 99.5), 
                                frameon=False, zorder=5)
                ax.add_artist(ab_ball)   
            
            
            # Foto del jugador en el campo (más grande)
            team_logo = self.load_team_logo(equipo)
            if team_logo is not None:
                logo_ax = ax.inset_axes([0.12, 0.02, 0.35, 0.35])
                logo_ax.imshow(team_logo, aspect='auto', alpha=0.8)
                logo_ax.axis('off')
            
            # Escudo del equipo 
            if team_filter and (team_logo := self.load_team_logo(team_filter)) is not None:
                logo_ax = ax.inset_axes([0.55, 0.02, 0.40, 0.40])
                logo_ax.imshow(team_logo, alpha=0.15)
                logo_ax.axis('off')

        plt.tight_layout()
        return fig
    
    def print_summary(self, team_filter=None):
        """Imprime resumen de los datos"""
        if self.lanzamientos_data.empty:
            pass
            return
        
        
        if team_filter:
            # CAMBIO: Ya no filtrar por team_filter, usar todos los datos
            team_data = self.lanzamientos_data
            if not team_data.empty:
                pass
                

                # Top jugadores con más lanzamientos

def seleccionar_equipo_interactivo():
    """Función para seleccionar equipo interactivamente"""
    try:
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/abp_events.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        if not equipos: 
            pass
            return None
        
        for i, equipo in enumerate(equipos, 1): 
            pass
        
        while True:
            try:
                indice = int(input(f"\nSelecciona un equipo (1-{len(equipos)}): ").strip()) - 1
                if 0 <= indice < len(equipos): 
                    return equipos[indice]
                else: 
                    pass
            except ValueError: 
                pass
    except Exception as e: 
        pass
        return None

def main():
    """Función principal"""
    try:
        pass
        if (equipo := seleccionar_equipo_interactivo()) is None:
            pass
            return
        
        analyzer = LanzamientosLadoDerecho(team_filter=equipo)
        analyzer.print_summary(team_filter=equipo)
        analyzer.debug_lanzamientos(team_filter=equipo)

        
        if (fig := analyzer.create_lanzamientos_visualization(team_filter=equipo)):
            plt.show()
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"campogramas_lanzamientos_derecha_{equipo_filename}.pdf"
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1, 
                       facecolor='white', dpi=300, orientation='landscape')
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()

def generar_campogramas_personalizado(equipo, mostrar=True, guardar=True):
    """Función para generar campogramas de forma personalizada"""
    try:
        analyzer = LanzamientosLadoDerecho(team_filter=equipo)
        analyzer.print_summary(team_filter=equipo)
        fig = analyzer.create_lanzamientos_visualization(team_filter=equipo)
        
        if fig:
            if mostrar: plt.show()
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"campogramas_lanzamientos_derecha_{equipo_filename}.pdf"
                fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1, 
                           facecolor='white', dpi=300)
            return fig
        else:
            print("❌ No se pudo generar la visualización")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def verificar_assets():
    """Verifica que todos los assets necesarios estén disponibles"""
    os.makedirs('assets/escudos', exist_ok=True)
    files_to_check = [
        'extraccion_opta/datos_opta_parquet/abp_events.parquet',
        'extraccion_opta/datos_opta_parquet/team_stats.parquet',
        'extraccion_opta/datos_opta_parquet/player_stats.parquet',
        'assets/fondo_informes.png', 
        'assets/balon.png'
    ]
    for file_path in files_to_check:
        print(f"✅ Encontrado: {file_path}" if os.path.exists(file_path) else f"❌ Faltante: {file_path}")
    
    if os.path.exists('assets/escudos') and (escudos := [f for f in os.listdir('assets/escudos') if f.endswith('.png')]):
        pass
    else:
        print("⚠️  No hay escudos en el directorio")

if __name__ == "__main__":
    pass
    try:
        verificar_assets()
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/abp_events.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        if equipos:
            pass
    except Exception as e:
        print(f"❌ Error al inicializar: {e}")
    
    main()