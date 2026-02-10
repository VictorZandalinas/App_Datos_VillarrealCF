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
from difflib import SequenceMatcher
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

class AnalizadorPenaltis:
    
    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/match_events.parquet", team_filter=None):
        """
        Inicializa el analizador. 
        NOTA: Por defecto busca en 'match_events.parquet' porque 'abp_events' 
        suele contener solo córners y faltas, no los tiros de penalti.
        """
        self.data_path = data_path
        self.team_filter = team_filter
        
        # Inicializar variables
        self.df = None
        self.penalties_data = pd.DataFrame()
        
        # Cargar archivos auxiliares
        try:
            pass
            self.team_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/team_stats.parquet")
            self.player_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/player_stats.parquet")
        except Exception as e:
            pass
            self.team_stats = pd.DataFrame()
            self.player_stats = pd.DataFrame()

        # Cargar datos principales
        self.load_data(team_filter)
        
        # --- DIAGNÓSTICO AUTOMÁTICO ---
        # Ejecuta el debug aquí para ver qué está pasando antes de filtrar
        if hasattr(self, 'debug_profundo_penaltis'):
            self.debug_profundo_penaltis()
        else:
            pass

        # Extraer penaltis si hay filtro de equipo y datos cargados
        if team_filter and self.df is not None and not self.df.empty:
            self.extract_penalties(team_filter)
    
    def guardar_sin_espacios(self, fig, filename):
        """Guarda sin espacios manteniendo landscape A4"""
        # Define el tamaño exacto de un A4 horizontal en pulgadas
        fig.set_size_inches(11.69, 8.27)
        
        fig.savefig(
            filename,
            dpi=300,
            bbox_inches='tight', # Elimina el borde blanco innecesario
            pad_inches=0,        # Elimina el relleno
            facecolor='white',
            edgecolor='none',
            format='pdf' if filename.endswith('.pdf') else 'png',
            transparent=False,
            orientation='landscape' # Asegura la orientación horizontal
        )
    
    def create_team_penalties_report(self, team_name, figsize=(11.69, 8.27)):
        """
        Método wrapper que maneja el caso de no datos.
        """
        # Intentar generar con filtro de equipo
        fig = self.create_penalties_visualization(figsize=figsize, team_filter=team_name)
        
        # Si no hay datos, crear placeholder informativo
        if fig is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 
                   f'SIN PENALTIS DISPONIBLES\n\n{team_name}\n\nNo hay datos de penaltis\nen las jornadas analizadas',
                   ha='center', va='center', fontsize=16, color='#7f8c8d')
            ax.axis('off')
        
        return fig
    
    def debug_profundo_penaltis(self):
        """
        Ejecuta este método para ver exactamente qué hay en el DataFrame
        y por qué fallan los filtros.
        """

        if self.df is None or self.df.empty:
            print("❌ ERROR CRÍTICO: El DataFrame está vacío o es None.")
            return

        
        # 1. BUSCAR COLUMNAS SOSPECHOSAS
        # Buscamos cualquier columna que tenga "pen" o "qual" en el nombre
        cols_penalti = [c for c in self.df.columns if 'pen' in c.lower()]
        cols_qualifiers = [c for c in self.df.columns if 'qual' in c.lower()]
        
        
        # 2. ANALIZAR VALORES EN ESAS COLUMNAS
        # Si existe la columna 'Penalty', ¿qué valores tiene? ¿'Sí', 'True', '1'?
        if 'Penalty' in self.df.columns:
            unique_vals = self.df['Penalty'].unique()
            print(f"   ⚠️ Tu código busca: 'Sí'. Si ves 'True', 1, o 'Yes', ahí está el fallo.")
        else:
            print(f"\n2️⃣ ❌ NO existe la columna 'Penalty'.")

        # 3. BUSCAR EVENTOS DE TIRO (Goles, Fallos)
        # Vamos a ver si existen los tiros, aunque no estén marcados como penalti
        eventos_tiro = ['Goal', 'Attempt Saved', 'Miss', 'Post', 'Missed']
        tiros = self.df[self.df['Event Name'].isin(eventos_tiro)]
        
        
        if not tiros.empty:
            # Cogemos 5 tiros al azar y mostramos sus datos de penalti
            sample = tiros.head(5)
            cols_interes = ['Event Name', 'playerName'] + cols_penalti
            # Filtramos solo columnas que existen
            cols_interes = [c for c in cols_interes if c in self.df.columns]
            
        else:
            print("   ⚠️ No se encontraron eventos de tiro. ¿Estás leyendo el archivo correcto?")

        # 4. BÚSQUEDA DE TEXTO BRUTA
        # A veces el penalti está escondido en qualifiers numéricos (ej: qualifier_id 9)
        # Vamos a buscar la palabra "Penalty" dentro de los nombres de evento
        eventos_con_nombre_penalty = self.df[self.df['Event Name'].str.contains('enalty', case=False, na=False)]
        if not eventos_con_nombre_penalty.empty:
            pass


    def load_data(self, team_filter=None):
        """Carga los datos necesarios desde los parquets"""
        try:
            # Cargar columnas básicas que sabemos que existen
            columns_needed = ['Match ID', 'Team ID', 'Team Name', 'Event Name', 'outcome', 
                'timeMin', 'timeSec', 'x', 'y', 'playerName', 'playerId', 'timeStamp', 'periodId']
            
            # Agregar columnas de penaltis que existen
            penalty_columns = ['Penalty', 'Low Left', 'High Left', 'Low Centre', 'High Centre', 
                            'Low Right', 'High Right', 'Goalmouth Y Coordinate', 'Goalmouth Z Coordinate',
                            'Right footed', 'Left footed', 'Strong', 'Weak', 'Scored', 'Saved',
                            'Standing', 'Diving']
            
            # Cargar todas las columnas (no filtrar, cargar todo el dataset)
            self.df = pd.read_parquet(self.data_path)
            
            # Normalizar timestamp
            self.df['timeStamp'] = self.df['timeStamp'].apply(self.normalize_timestamp)
            
            # Si hay filtro de equipo, filtrar matches desde el inicio
            if team_filter:
                team_matches = self.team_stats[self.team_stats['Team Name'] == team_filter]['Match ID'].unique()
                self.df = self.df[self.df['Match ID'].isin(team_matches)]
            
        except Exception as e:
            pass

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
            return timestamp  # Devolver original si falla la conversión

    def extract_penalties(self, team_filter=None):
        """Extrae todos los penaltis del equipo seleccionado"""
        if self.df is None:
            pass
            return
        
        
        # Filtrar penaltis - buscar en eventos que contengan 'Penalty' o en columnas de qualifiers
        penalty_events = self.df[
            (self.df['Event Name'].isin(['Goal', 'Attempt Saved', 'Miss', 'Post'])) &
            (
                (self.df.get('Penalty', '') == 'Sí') |
                (self.df['Event Name'].str.contains('penalty', case=False, na=False)) |
                (self.df.get('Penalty taken', '') == 'Sí')
            )
        ].copy()
        
        penalties_list = []
        
        for _, penalty in penalty_events.iterrows():
            # Determinar el resultado del penalti
            result_type = self.determine_penalty_result(penalty)
            
            # Obtener coordenadas de la portería
            goal_coords = self.get_goal_coordinates(penalty)

            # DEBUG: Imprimir coordenadas para cada penalti
            
            # Obtener características del lanzamiento
            penalty_characteristics = self.get_penalty_characteristics(penalty)
            
            penalty_data = {
                'Match ID': penalty['Match ID'],
                'Team ID': penalty['Team ID'],
                'Team Name': penalty['Team Name'],
                'playerName': penalty['playerName'],
                'playerId': penalty['playerId'],
                'timeMin': penalty['timeMin'],
                'timeSec': penalty['timeSec'],
                'x': penalty.get('x', 0),
                'y': penalty.get('y', 0),
                'result_type': result_type,
                'goal_y': goal_coords['y'],
                'goal_z': goal_coords['z'],
                'goal_zone': goal_coords['zone']
            }
            
            penalty_data.update(penalty_characteristics)
            penalties_list.append(penalty_data)
        
        if penalties_list:
            self.penalties_data = pd.DataFrame(penalties_list)
            
            # Eliminar duplicados
            self.penalties_data = self.penalties_data.drop_duplicates(
                subset=['Match ID', 'timeMin', 'timeSec', 'playerName'], 
                keep='first'
            )
            
            
            if team_filter:
                team_penalties = self.penalties_data[self.penalties_data['Team Name'] == team_filter]
        else:
            pass
    
    def debug_penalty_coordinates(self, team_filter=None):
        """Debug específico para analizar coordenadas de penaltis"""
        if self.penalties_data.empty:
            pass
            return
        
        team_data = self.penalties_data[
            self.penalties_data['Team Name'] == team_filter
        ] if team_filter else self.penalties_data
        
        
        
        
        
        for _, penalty in team_data.iterrows():
            pass

    def determine_penalty_result(self, penalty_row):
        """Determina el resultado del penalti basado en el Event Name y qualifiers"""
        event_name = penalty_row['Event Name']
        
        # Verificar qualifiers específicos si están disponibles
        if penalty_row.get('Scored') == 'Sí':
            return 'Gol'
        elif penalty_row.get('Saved') == 'Sí':
            return 'Parado'
        elif penalty_row.get('Missed') == 'Sí':
            return 'Fallado'
        
        # Fallback basado en Event Name
        if event_name == 'Goal':
            return 'Gol'
        elif event_name == 'Attempt Saved':
            return 'Parado'
        elif event_name in ['Miss', 'Post']:
            return 'Fallado'
        else:
            return 'Desconocido'

    def get_goal_coordinates(self, penalty_row):
        """Obtiene las coordenadas usando solo Goalmouth Y y Z Coordinates"""
        
        # Usar directamente las coordenadas de Opta
        goal_y = penalty_row.get('Goalmouth Y Coordinate', 50)  # Horizontal (0-100)
        goal_z = penalty_row.get('Goalmouth Z Coordinate', 19)  # Vertical (0-38, donde 38 es el tope)
        
        return {
            'y': float(goal_y) if pd.notna(goal_y) else 50,
            'z': float(goal_z) if pd.notna(goal_z) else 19,
            'zone': 'Coordenadas'  # Ya no usamos zonas
        }

    def get_penalty_characteristics(self, penalty_row):
        """Extrae características del lanzamiento de penalti"""
        characteristics = {
            'foot_used': 'Desconocido',
            'power': 'Desconocido',
            'technique': 'Normal',
            'goalkeeper_action': 'Desconocido'
        }
        
        # Pie utilizado
        if penalty_row.get('Right footed') == 'Sí':
            characteristics['foot_used'] = 'Derecho'
        elif penalty_row.get('Left footed') == 'Sí':
            characteristics['foot_used'] = 'Izquierdo'
        
        # Potencia del disparo
        if penalty_row.get('Strong') == 'Sí':
            characteristics['power'] = 'Fuerte'
        elif penalty_row.get('Weak') == 'Sí':
            characteristics['power'] = 'Suave'
        
        # Técnica especial
        if penalty_row.get('Panenka') == 'Sí':
            characteristics['technique'] = 'Panenka'
        
        # Acción del portero
        if penalty_row.get('Standing') == 'Sí':
            characteristics['goalkeeper_action'] = 'De pie'
        elif penalty_row.get('Diving') == 'Sí':
            characteristics['goalkeeper_action'] = 'Lanzándose'
        
        return characteristics

    def format_player_name_multiline(self, player_name, max_chars_per_line=12):
        """Divide nombres largos en 2 líneas de forma inteligente"""
        words = player_name.split()
        
        if len(words) == 1:
            if len(player_name) > max_chars_per_line:
                mid = len(player_name) // 2
                return player_name[:mid], player_name[mid:]
            else:
                return player_name, None
        
        line1 = words[0]
        line2 = ' '.join(words[1:])
        
        if len(line1) > max_chars_per_line:
            line1 = line1[:max_chars_per_line-3] + '...'
        
        if len(line2) > max_chars_per_line:
            line2 = line2[:max_chars_per_line-3] + '...'
        
        return line1, line2

    def get_player_shirt_number_by_name(self, player_name):
        """Obtiene el dorsal del jugador por nombre"""
        if pd.isna(player_name):
            return None
        
        player_info = self.player_stats[self.player_stats['Match Name'] == player_name]
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
            return self._load_team_logo_original(equipo)
        
        possible_names = [
            equipo, equipo.replace(' ', '_'), equipo.replace(' ', ''),
            equipo.lower(), equipo.lower().replace(' ', '_'), equipo.lower().replace(' ', '')
        ]
        
        logo_path = None
        for name in possible_names:
            path = f"assets/escudos/{name}.png"
            if os.path.exists(path): 
                logo_path = path
                break
        
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
                pass
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

    def load_player_photos(self):
        """Carga el JSON con las fotos de jugadores"""
        import json
        try:
            with open('assets/jugadores_optimizados.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            pass
            return []

    def get_player_photo_without_dorsal(self, player_name, photos_data):
        """Obtiene la foto sin fondo blanco pero SIN dorsal"""
        match = self.match_player_name(player_name, photos_data, self.team_filter)
        if not match:
            return None
        
        try:
            img_data = base64.b64decode(match['image_base64'])
            img = Image.open(BytesIO(img_data))
            
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            data = np.array(img)
            
            if len(data.shape) != 3 or data.shape[2] != 4:
                return None
            
            height, width = data.shape[:2]
            
            def flood_fill_iterative(start_points, threshold=235):
                visited = np.zeros((height, width), dtype=bool)
                background_mask = np.zeros((height, width), dtype=bool)
                
                def is_background_color(y, x):
                    if y < 0 or y >= height or x < 0 or x >= width:
                        return False
                    return (data[y, x, 0] >= threshold and 
                            data[y, x, 1] >= threshold and 
                            data[y, x, 2] >= threshold)
                
                for start_y, start_x in start_points:
                    if visited[start_y, start_x] or not is_background_color(start_y, start_x):
                        continue
                    
                    stack = [(start_y, start_x)]
                    
                    while stack:
                        y, x = stack.pop()
                        
                        if (y < 0 or y >= height or x < 0 or x >= width or 
                            visited[y, x] or not is_background_color(y, x)):
                            continue
                        
                        visited[y, x] = True
                        background_mask[y, x] = True
                        
                        stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
                
                return background_mask
            
            border_points = [
                (0, 0), (0, width-1), (height-1, 0), (height-1, width-1),
                (0, width//2), (height-1, width//2),
                (height//2, 0), (height//2, width-1)
            ]
            
            background_mask = flood_fill_iterative(border_points, threshold=230)
            data[background_mask] = [0, 0, 0, 0]
            
            return data.astype(np.float32) / 255.0
        
        except Exception as e:
            pass
            return None

    def match_player_name(self, player_name, photos_data, team_filter=None):
        """Encuentra el nombre más parecido en los datos de las fotos"""
        
        def normalize_name(name):
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
            player_full = player_parts['full']
            photo_full = photo_parts['full']

            if player_full == photo_full:
                return (1.0, "MATCH EXACTO COMPLETO")

            if len(player_parts['all_parts']) == 1:
                if player_full in photo_parts['all_parts']:
                    return (0.95, f"MATCH DE NOMBRE ÚNICO '{player_full}'")

            if (len(player_parts['first_name']) == 1 and
                player_parts['last_name'] == photo_parts['last_name'] and
                photo_parts['first_name'].startswith(player_parts['first_name'])):
                return (0.90, "INICIAL + APELLIDO EXACTO")

            if (player_parts['first_name'] == photo_parts['first_name'] and
                len(player_parts['last_name']) == 1 and
                photo_parts['last_name'].startswith(player_parts['last_name'][0])):
                return (0.90, "NOMBRE + INICIAL APELLIDO")

            if player_parts['last_name'] == photo_parts['last_name']:
                first_name_sim = SequenceMatcher(None, player_parts['first_name'], photo_parts['first_name']).ratio()
                if first_name_sim >= 0.8:
                    return (0.85 + (first_name_sim * 0.05), f"APELLIDO EXACTO + NOMBRE SIMILAR ({first_name_sim:.2f})")
            
            full_sim = SequenceMatcher(None, player_full, photo_full).ratio()
            if full_sim > 0.8:
                return (full_sim, f"SIMILITUD GENERAL ALTA ({full_sim:.2f})")

            return (0.0, "SIN COINCIDENCIA CLARA")

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
            
            if score >= 0.90:
                found_matches.append({
                    "entry": photo_entry,
                    "score": score,
                    "reason": reason
                })
                
        if len(found_matches) == 1:
            return found_matches[0]['entry']
        elif len(found_matches) > 1:
            return None
        else:
            return None

    def get_outcome_color(self, result_type):
        """Colores para diferentes resultados de penalti"""
        color_map = {
            'Gol': '#2ECC71',      # Verde
            'Parado': '#3498DB',   # Azul
            'Fallado': '#E74C3C',  # Rojo
            'Desconocido': '#95A5A6'  # Gris
        }
        return color_map.get(result_type, '#95A5A6')

    def get_outcome_marker(self, result_type):
        """Marcadores para diferentes resultados de penalti"""
        marker_map = {
            'Gol': 'o',
            'Parado': 's', 
            'Fallado': 'X',
            'Desconocido': 'D'
        }
        return marker_map.get(result_type, 'o')

    def draw_goal_area(self, ax):
        """Dibuja la portería con las coordenadas exactas de Opta"""
        
        # Coordenadas exactas según Opta
        goal_left = 54.8     
        goal_right = 45.2    
        goal_bottom = 0      
        goal_top = 38        
        
        # Postes y larguero
        left_post_outer = 55.8
        right_post_outer = 44.2
        crossbar_top = 42
        
        # Fondo verde césped
        ax.set_facecolor('#2d5a27')
        
        # --- INICIO DE CAMBIOS ---
        
        # (CAMBIO: Red ampliada) La red ahora usa las coordenadas EXTERIORES de los postes
        # para ocupar todo el fondo de la portería.
        for x in np.linspace(left_post_outer, right_post_outer, 14): # Ampliado de poste a poste
            ax.plot([x, x], [goal_bottom, goal_top], 'white', linewidth=0.5, alpha=0.6, zorder=1)
        for y in np.linspace(goal_bottom, goal_top, 10):
            ax.plot([left_post_outer, right_post_outer], [y, y], 'white', linewidth=0.5, alpha=0.6, zorder=1) # Ampliado de poste a poste

        # Coordenadas para la forma de 'U' (poste izq -> larguero -> poste der)
        goal_frame_x = [right_post_outer, right_post_outer, left_post_outer, left_post_outer]
        goal_frame_y = [goal_bottom, crossbar_top, crossbar_top, goal_bottom]
        
        # Dibujar la 'U' (postes y larguero) de una vez con grosor de 12
        ax.plot(goal_frame_x, goal_frame_y, 'white', linewidth=12, solid_capstyle='butt', zorder=2)
        
        # (CAMBIO: Línea de gol más fina) Se dibuja la línea de gol con un grosor menor (4).
        ax.plot([right_post_outer, left_post_outer], [goal_bottom, goal_bottom], 'white', linewidth=4, zorder=2)

        # --- FIN DE CAMBIOS ---

        # Ajustar límites
        ax.set_xlim(42, 58)      
        ax.set_ylim(-2, 44)      
        ax.set_aspect('auto')
        
        ax.set_xticks([])
        ax.set_yticks([])
        
    def create_penalty_ranking(self, ax, team_filter):
        """Crea el ranking visual de lanzadores de penaltis"""
        if self.penalties_data.empty:
            ax.text(0.5, 0.5, 'RANKING LANZADORES\nDE PENALTIS\n\n(Sin datos)', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        team_data = self.penalties_data[
            self.penalties_data['Team Name'] == team_filter
        ] if team_filter else self.penalties_data
        
        if team_data.empty:
            ax.text(0.5, 0.5, 'RANKING LANZADORES\nDE PENALTIS\n\n(Sin datos del equipo)', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # Contar penaltis por jugador
        penalty_counts = team_data['playerName'].value_counts().head(5)
        photos_data = self.load_player_photos()
        
        ax.set_facecolor('#f8f9fa')
        
        # Título
        ax.text(0.5, 0.99, 'RANKING LANZADORES\nDE PENALTIS', fontsize=12, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes, color='#2c3e50')
        
        position_colors = ['#e74c3c', '#f39c12', '#f1c40f', '#95a5a6', '#bdc3c7']
        
        for i, (player_name, penalty_count) in enumerate(penalty_counts.items()):
            y_pos = 0.80 - (i * 0.15)
            
            # Fondo del jugador
            rect_bg = patches.FancyBboxPatch((0.02, y_pos - 0.06), 0.96, 0.12,
                                           boxstyle="round,pad=0.01", 
                                           facecolor='white',
                                           edgecolor=position_colors[i],
                                           linewidth=2, alpha=0.9)
            ax.add_patch(rect_bg)
            
            # Número de posición
            ax.text(0.08, y_pos, f"#{i+1}", fontsize=14, fontweight='bold', 
                    va='center', ha='center', color=position_colors[i])
            
            # Foto del jugador
            player_photo = self.get_player_photo_without_dorsal(player_name, photos_data)
            if player_photo is not None:
                photo_ax = ax.inset_axes([0.12, y_pos - 0.065, 0.25, 0.16])
                photo_ax.imshow(player_photo, aspect='auto')
                photo_ax.axis('off')
            
            # Nombre del jugador
            line1, line2 = self.format_player_name_multiline(player_name, max_chars_per_line=12)

            if line2 is None:
                ax.text(0.45, y_pos + 0.022, line1, fontsize=12, fontweight='bold', 
                        va='center', ha='left', color='#2c3e50')
            else:
                ax.text(0.45, y_pos + 0.037, line1, fontsize=11, fontweight='bold', 
                        va='center', ha='left', color='#2c3e50')
                ax.text(0.45, y_pos + 0.007, line2, fontsize=11, fontweight='bold', 
                        va='center', ha='left', color='#2c3e50')
            
            # Estadística
            ax.text(0.45, y_pos - 0.02, f'{penalty_count} penaltis', 
                    fontsize=8, va='center', ha='left', color='#7f8c8d')
            
            # Efectividad
            player_penalties = team_data[team_data['playerName'] == player_name]
            goals = len(player_penalties[player_penalties['result_type'] == 'Gol'])
            if penalty_count > 0:
                effectiveness = (goals / penalty_count) * 100
                star_count = min(5, max(1, int(effectiveness / 20)))
                star_text = '★' * star_count
                ax.text(0.45, y_pos - 0.05, f"{effectiveness:.0f}% {star_text}", 
                        fontsize=8, va='center', ha='left', color='#f39c12')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def create_penalties_visualization(self, figsize=(11.69, 8.27), team_filter=None):
        """Crea la visualización principal de penaltis"""
        
        if self.penalties_data.empty:
            pass
            return None
        
        # Filtrar datos del equipo
        team_data = self.penalties_data[
            self.penalties_data['Team Name'] == team_filter
        ] if team_filter else self.penalties_data
        
        if team_data.empty:
            pass
            return None
        
        # Obtener top 3 lanzadores de penaltis
        top_penalty_takers = team_data['playerName'].value_counts().head(3)
        top_3_nombres = top_penalty_takers.index[:3].tolist()
        photos_data = self.load_player_photos()
        
        # Crear figura
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        # Fondo
        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25)
            ax_bg.axis('off')
        
        # Título principal
        fig.suptitle('ANÁLISIS DE PENALTIS', 
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
        
        # Configurar grid de subplots
        gs_main = fig.add_gridspec(2, 9, height_ratios=[2, 1], hspace=0.3,
                left=0.03, right=0.95, top=0.85, bottom=0.05)

        # --- GRÁFICO SUPERIOR ---
        ax_general = fig.add_subplot(gs_main[0, 0:7])
        self.draw_goal_area(ax_general)
        ax_general.invert_xaxis()  # <--- INVERSIÓN PARA EL GRÁFICO SUPERIOR

        ax_general.set_title(f'PENALTIS GENERALES ({len(team_data)} total)', 
                        fontsize=14, fontweight='bold', color='#1e3d59', pad=10, family='serif')
        
        # Plotear todos los penaltis
        for result_type in team_data['result_type'].unique():
            data_tipo = team_data[team_data['result_type'] == result_type]
            if len(data_tipo) > 0:
                ax_general.scatter(data_tipo['goal_y'], data_tipo['goal_z'], 
                                c=self.get_outcome_color(result_type), 
                                marker=self.get_outcome_marker(result_type),
                                s=150, alpha=0.8, edgecolors='white', linewidth=2,
                                label=f'{result_type} ({len(data_tipo)})',
                                zorder=5)
        
        ax_general.set_xticks([])
        ax_general.set_yticks([])
        ax_general.set_aspect('auto')
        ax_general.legend(loc='center left', bbox_to_anchor=(- 0.03, 0.55), 
                        framealpha=0.9, facecolor='white', edgecolor='#1e3d59')

        # --- RANKING ---
        ax_ranking = fig.add_subplot(gs_main[0, 7:9])
        self.create_penalty_ranking(ax_ranking, team_filter)


        # --- SUBPLOTS INFERIORES ---
        for i, jugador in enumerate(top_3_nombres):
            if i >= 3:
                break
                
            col_start = i * 3
            col_end = col_start + 3
            ax = fig.add_subplot(gs_main[1, col_start:col_end])
            self.draw_goal_area(ax)
            ax.invert_xaxis()  # <--- INVERSIÓN PARA CADA GRÁFICO INFERIOR

            data_jugador = team_data[team_data['playerName'] == jugador]
            
            # Textos y fotos
            ax.text(0.5, 1.25, jugador, fontsize=11, fontweight='bold', 
                    color='#1e3d59', ha='center', va='center', transform=ax.transAxes,
                    family='serif')
            total_penalties = len(data_jugador)
            goals = len(data_jugador[data_jugador['result_type'] == 'Gol'])
            effectiveness = (goals / total_penalties * 100) if total_penalties > 0 else 0
            ax.text(0.5, 1.15, f'Penaltis: {total_penalties} | Efectividad: {effectiveness:.0f}%', 
                    fontsize=9, color='#1e3d59', ha='center', va='center', transform=ax.transAxes,
                    family='serif')
            player_photo = self.get_player_photo_without_dorsal(jugador, photos_data)
            if player_photo is not None:
                photo_ax = ax.inset_axes([0.0, 1.05, 0.22, 0.40])
                photo_ax.imshow(player_photo, aspect='auto', alpha=1.0)
                photo_ax.axis('off')
            dorsal = self.get_player_shirt_number_by_name(jugador)
            if dorsal:
                ax.text(0.5, 1.35, f"#{dorsal}", fontsize=18, fontweight='bold', 
                    color='#1e3d59', ha='center', va='center', transform=ax.transAxes)
            if team_filter and (team_logo := self.load_team_logo(team_filter)) is not None:
                logo_ax = ax.inset_axes([0.75, 1.05, 0.25, 0.25])
                logo_ax.imshow(team_logo, alpha=0.6)
                logo_ax.axis('off')
            
            # Plotear penaltis del jugador
            for result_type in data_jugador['result_type'].unique():
                data_result = data_jugador[data_jugador['result_type'] == result_type]
                ax.scatter(data_result['goal_y'], data_result['goal_z'],
                        c=self.get_outcome_color(result_type), 
                        marker=self.get_outcome_marker(result_type),
                        s=60, alpha=0.9, edgecolors='white', linewidth=2,
                        zorder=5)

            # Ejes y logos
            ax.set_aspect('auto')
            ax.set_xticks([])
            ax.set_yticks([])
            if team_filter and (team_logo := self.load_team_logo(team_filter)) is not None:
                logo_ax = ax.inset_axes([1.05, 0.05, 0.25, 0.25])
                logo_ax.imshow(team_logo, alpha=0.4)
                logo_ax.axis('off')

        plt.tight_layout()
        return fig

    def print_summary(self, team_filter=None):
        """Imprime resumen de los datos de penaltis"""
        if self.penalties_data.empty:
            pass
            return
        
        
        if team_filter:
            team_data = self.penalties_data[self.penalties_data['Team Name'] == team_filter]
            if not team_data.empty:
                pass


def seleccionar_equipo_interactivo():
    """Función para seleccionar equipo interactivamente"""
    try:
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/match_events.parquet")
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
        
        analyzer = AnalizadorPenaltis(team_filter=equipo)
        
        # DEBUG: Agregar esta línea
        analyzer.debug_penalty_coordinates(team_filter=equipo)
        
        analyzer.print_summary(team_filter=equipo)
        
        # CAMBIO: Usar create_team_penalties_report que maneja el caso sin datos
        fig = analyzer.create_team_penalties_report(equipo)
        
        if fig:
            plt.show()
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"campogramas_penaltis_{equipo_filename}.pdf"
            analyzer.guardar_sin_espacios(fig, output_path) 
        else:
            print("⚠️  Error crítico al generar la figura")
            
    except Exception as e:
        print(f"⚠️  Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()

def generar_campogramas_personalizado(equipo, mostrar=True, guardar=True):
    """Función para generar campogramas de forma personalizada"""
    try:
        analyzer = AnalizadorPenaltis(team_filter=equipo)
        analyzer.print_summary(team_filter=equipo)
        fig = analyzer.create_penalties_visualization(team_filter=equipo)
        
        if fig:
            if mostrar: plt.show()
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"campogramas_penaltis_{equipo_filename}.pdf"
                fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1, 
                           facecolor='white', dpi=300)
            return fig
        else:
            pass
            return None
            
    except Exception as e:
        pass
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
        pass
    
    if os.path.exists('assets/escudos') and (escudos := [f for f in os.listdir('assets/escudos') if f.endswith('.png')]):
        pass
    else:
        pass

if __name__ == "__main__":
    pass
    try:
        verificar_assets()
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/abp_events.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        if equipos:
            pass
    except Exception as e:
        pass
    
    main()