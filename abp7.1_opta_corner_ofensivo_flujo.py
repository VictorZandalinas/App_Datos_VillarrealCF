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
import unicodedata
from PIL import Image
from mplsoccer import VerticalPitch
from difflib import SequenceMatcher
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Imports para Sankey
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly no disponible. Se usará versión simplificada de Sankey.")

class ReporteFlujoCorners:
    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/abp_events.parquet", team_filter=None):
        self.data_path = data_path
        self.team_filter = team_filter
        self.df = None
        self.corner_data = pd.DataFrame()
        self.team_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/team_stats.parquet")
        self.player_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/player_stats.parquet")

        # Diccionario para nombres de zonas
        self.ZONA_NAMES = {
            'zona_1': 'Z. IZQUIERDA',
            'zona_2': 'BORDE ÁREA',
            'zona_3': 'ÁREA PEQ. IZQ',
            'zona_4': 'Z. PORTERÍA',
            'zona_5': 'ÁREA PEQ. DER',
            'zona_6': 'PENALTI',
            'zona_7': 'Z. DERECHA'
        }
        
        self.load_data(team_filter)
        
        if team_filter:
            self.corner_data = self.df.copy()

    def load_player_photos(self):
        """Carga el JSON con las fotos de jugadores"""
        import json
        try:
            with open('assets/jugadores_optimizados.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ No se encontró el archivo jugadores_optimizados.json")
            return []

    def match_player_name(self, player_name, photos_data):
        """Encuentra el nombre más parecido en los datos de las fotos"""
        def normalize_name(name):
            if not isinstance(name, str):
                return ""
            no_accents = "".join(c for c in unicodedata.normalize('NFD', name) 
                                if unicodedata.category(c) != 'Mn')
            no_accents = no_accents.replace('ø', 'o').replace('Ø', 'O')
            no_accents = no_accents.replace('ł', 'l').replace('Ł', 'L')
            name_lower = no_accents.lower().strip()
            name_clean = re.sub(r'[^\w\s]', '', name_lower)
            return ' '.join(name_clean.split())

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
            
            if score >= 0.8:
                found_matches.append({
                    "entry": photo_entry,
                    "score": score,
                    "reason": reason
                })
        
        if not found_matches:
            return None

        found_matches.sort(key=lambda x: x['score'], reverse=True)
        
        if found_matches[0]['score'] == 1.0:
            return found_matches[0]['entry']
        
        if len(found_matches) > 1 and (found_matches[0]['score'] - found_matches[1]['score']) < 0.1:
            return None
            
        return found_matches[0]['entry']
    
    def get_player_dorsal(self, player_id):
        """Obtiene el dorsal de un jugador por su ID"""
        try:
            if pd.isna(player_id):
                return None
            dorsal_row = self.player_stats[self.player_stats['Player ID'] == player_id]
            if not dorsal_row.empty:
                dorsal = dorsal_row['Shirt Number'].iloc[0]
                return int(dorsal) if pd.notna(dorsal) else None
            return None
        except:
            return None
    
    def calculate_dynamic_sizing(self, num_lanzadores, num_rematadores):
        """Calcula tamaños dinámicos con mejor escalado"""
        max_elementos = max(num_lanzadores, num_rematadores, 1)
        
        # Escalado más suave y progresivo
        if max_elementos <= 2:
            box_size = 1.2
            spacing_factor = 1.0
            box_height_factor = 0.16  # Más alto para pocos elementos
        elif max_elementos <= 4:
            box_size = 1.0
            spacing_factor = 0.95
            box_height_factor = 0.14
        elif max_elementos <= 6:
            box_size = 0.85
            spacing_factor = 0.9
            box_height_factor = 0.13
        elif max_elementos <= 8:
            box_size = 0.7
            spacing_factor = 0.85
            box_height_factor = 0.12
        elif max_elementos <= 12:
            box_size = 0.6
            spacing_factor = 0.8
            box_height_factor = 0.11
        else:
            box_size = 0.5
            spacing_factor = 0.75
            box_height_factor = 0.10  # Más pequeño para muchos elementos
        
        return {
            'box_size': box_size,
            'spacing_factor': spacing_factor,
            'altura_disponible': 0.75,  # Usar más espacio vertical
            'box_height_factor': box_height_factor  # NUEVO: altura específica
        }

    def draw_artistic_player_box(self, ax, x, y, player_name, player_id, photos_data, sizing_params, is_shooter=True, count_value=0):
        """Dibuja elemento cuadrado con escalado dinámico mejorado"""
        box_size = sizing_params['box_size']
        
        # USAR altura dinámica del nuevo parámetro
        box_width = 0.10 * box_size   
        box_height = sizing_params['box_height_factor'] * box_size  # USAR NUEVA altura dinámica
        
        # 1. Recuadro blanco
        main_box = patches.FancyBboxPatch(
            (x - box_width/2, y - box_height/2), 
            box_width, box_height,
            boxstyle="round,pad=0.005", 
            facecolor='white',
            edgecolor='#2C3E50', 
            linewidth=2, 
            zorder=5
        )
        ax.add_patch(main_box)
        
        # OBTENER datos del jugador
        player_photo = self.get_player_photo_without_dorsal(player_name, photos_data)
        dorsal = self.get_player_dorsal(player_id)
        
        # 2. FOTO más alta que ancha - CAMBIAR dimensiones
        if player_photo is not None:
            # ANTES: foto_left/right con 0.3, AHORA más estrecha
            foto_left = x - box_width * 0.33    # ANTES: 0.3, AHORA: 0.25 (más estrecha)
            foto_right = x + box_width * 0.33   # ANTES: 0.3, AHORA: 0.25 (más estrecha)
            
            # ANTES: foto_bottom/top, AHORA más baja y más alta
            foto_bottom = y - box_height * 0.25  # ANTES: y + box_height * 0.05, AHORA: y - box_height * 0.05 (más abajo)
            foto_top = y + box_height * 0.68    # ANTES: y + box_height * 0.4, AHORA: y + box_height * 0.35 (más alta)
            
            ax.imshow(player_photo, 
                    extent=[foto_left, foto_right, foto_bottom, foto_top],
                    aspect='auto', zorder=8)
        else:
            # Ajustar posición de iniciales también
            iniciales = ''.join([word[0] for word in player_name.split()[:2]]).upper()
            ax.text(x, y + box_height * 0.1, iniciales, ha='center', va='center',  # ANTES: 0.2, AHORA: 0.1
                    fontsize=8 * box_size, fontweight='bold', color='#2C3E50', zorder=10)
        
        # 3. NOMBRE completo en 1 o 2 filas
        def format_player_name(name, max_chars_per_line=9):  # REDUCIR de 10 a 9
            """Divide nombre en líneas de forma más inteligente"""
            words = name.split()
            
            # Si es corto, una línea
            if len(name) <= max_chars_per_line:
                return name
            
            # Si son 2 palabras, intentar equilibrar
            if len(words) == 2:
                if len(words[0]) <= max_chars_per_line and len(words[1]) <= max_chars_per_line:
                    return f"{words[0]}\n{words[1]}"
            
            # Si hay más palabras, dividir inteligentemente
            if len(words) >= 2:
                # Intentar primera palabra + resto
                first_line = words[0]
                second_line = ' '.join(words[1:])
                
                # Si la segunda línea es muy larga, cortarla
                if len(second_line) > max_chars_per_line:
                    second_line = second_line[:max_chars_per_line-1] + '.'
                    
                return f"{first_line}\n{second_line}"
            
            # Palabra única muy larga
            return f"{name[:max_chars_per_line-1]}."

        nombre_formateado = format_player_name(player_name, 9)

        ax.text(x, y - box_height * 0.40, nombre_formateado,  # ANTES: 0.32, AHORA: 0.38 (más abajo)
                ha='center', va='center', 
                fontsize=max(6, 8 * box_size), fontweight='bold',  # MÍNIMO de 4 para legibilidad
                color='#2C3E50', zorder=10,
                linespacing=0.8)
        
        # 4. DORSAL en círculo
        if dorsal and dorsal != '':
            circle_x = x + box_width * 0.35
            circle_y = y + box_height * 0.4  # ANTES: 0.32, AHORA: 0.4 (ajustar a nueva altura)
            radio_circulo = max(0.018, 0.022 * box_size)  # MÍNIMO tamaño para visibilidad
    
            # DESPUÉS:
            dorsal_square = patches.Rectangle(
                (circle_x - radio_circulo * 0.6, circle_y - radio_circulo * 0.6),  # (x, y) esquina inferior izquierda
                radio_circulo * 1.2,  # Ancho
                radio_circulo * 1.2,  # Alto
                facecolor='#E74C3C', edgecolor='white', 
                linewidth=1, zorder=15
            )

            ax.add_patch(dorsal_square)

            ax.text(circle_x, circle_y, str(int(dorsal)), 
                    ha='center', va='center', 
                    fontsize=max(6, 8 * box_size), fontweight='black',  # ANTES: 6, AHORA: 8 + weight='black'
                    color='white', zorder=16,
                    family='serif',           # NUEVO: fuente más elegante
                    style='italic')           # NUEVO: cursiva para efecto deportivo

        # 5. CONTADOR en esquina inferior izquierda
        if count_value > 0:
            counter_x = x - box_width * 0.55      # Esquina izquierda
            counter_y = y - box_height * 0.5     # Parte inferior
            counter_radius = max(0.01, 0.018 * box_size)  # Más pequeño que dorsal
            
            # Círculo negro para el contador
            counter_circle = patches.Circle(
                (counter_x, counter_y), counter_radius,
                facecolor='black', edgecolor='white', 
                linewidth=1, zorder=15
            )
            ax.add_patch(counter_circle)

            # Número en blanco, fuente más simple
            ax.text(counter_x, counter_y, str(int(count_value)), 
                    ha='center', va='center', 
                    fontsize=max(7, 9 * box_size), fontweight='normal',  # Normal weight
                    color='white', zorder=16,
                    family='sans-serif')  # Fuente más simple que el dorsal

    def get_player_photo_without_dorsal(self, player_name, photos_data):
        """Obtiene la foto sin fondo blanco pero SIN dorsal"""
        match = self.match_player_name(player_name, photos_data)
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
            print(f"⚠️ Error procesando foto de {player_name}: {e}")
            return None

    def load_data(self, team_filter=None):
        """Carga los datos necesarios"""
        try:
            columns_needed = ['Match ID', 'Team ID', 'Team Name', 'Event Name', 'outcome', 
                            'timeMin', 'timeSec', 'x', 'y', 'Pass End X', 'Pass End Y', 
                            'playerName', 'playerId', 'Corner taken', 'From corner',
                            'In-swinger', 'Out-swinger', 'Straight', 'Left footed', 'Right footed', 'Cross']
            
            try:
                self.df = pd.read_parquet(self.data_path, columns=columns_needed)
            except Exception:
                basic_columns = ['Match ID', 'Team ID', 'Team Name', 'Event Name', 'outcome', 
                            'timeMin', 'timeSec', 'x', 'y', 'Pass End X', 'Pass End Y', 
                            'playerName', 'playerId']
                self.df = pd.read_parquet(self.data_path, columns=basic_columns)
                for col in ['Corner taken', 'From corner', 'In-swinger', 'Out-swinger', 'Straight', 'Cross', 'Left footed', 'Right footed']:
                    if col not in self.df.columns:
                        self.df[col] = 'No'
            
            relevant_events = ['Pass', 'Goal', 'Attempt Saved', 'Miss', 'Post']
            self.df = self.df[self.df['Event Name'].isin(relevant_events)]
            
            if team_filter:
                team_matches = self.team_stats[self.team_stats['Team Name'] == team_filter]['Match ID'].unique()
                self.df = self.df[
                    (self.df['Match ID'].isin(team_matches)) & 
                    (self.df['Team Name'] == team_filter)
                ]
            
            corners_df = self.df[
                (self.df['Corner taken'] == 'Sí') & 
                (self.df['Team Name'] == team_filter if team_filter else True)
            ].copy()
            
            corners_df = corners_df.drop_duplicates(
                subset=['Match ID', 'Team ID', 'playerName', 'timeMin', 'timeSec'], 
                keep='first'
            )
            
            self.corner_data = corners_df.copy()
            
            print(f"✅ Datos cargados: {len(self.df)} eventos totales")
            print(f"✅ Corners únicos del equipo: {len(self.corner_data)}")
            
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")
            import traceback
            traceback.print_exc()

    def get_zona_from_coordinates(self, x, y):
        """Determina la zona según las coordenadas x,y"""
        x, y = float(x), float(y)
        
        if x < 70:
            return None
        
        if x >= 83 and x <= 94.2 and y >= 42 and y <= 58:
            return 'zona_6'
        elif x >= 70 and x <= 100 and y >= 75 and y <= 100:
            return 'zona_1'
        elif x >= 70 and x <= 88.5 and y >= 25 and y <= 75:
            if x >= 83 and y >= 42 and y <= 58:
                return None
            return 'zona_2'
        elif x >= 88.5 and x <= 100 and y >= 58 and y <= 75:
            return 'zona_3'
        elif x >= 94.2 and x <= 100 and y >= 42 and y <= 58:
            return 'zona_4'
        elif x >= 88.5 and x <= 100 and y >= 25 and y <= 42:
            return 'zona_5'
        elif x >= 70 and x <= 100 and y >= 0 and y <= 25:
            return 'zona_7'
        else:
            return None

    def get_tipo_lanzamiento_data_unificado(self):
        """Obtiene tipos de lanzamiento combinando ambos lados"""
        corners_izq = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] > 99) &
            (self.corner_data['Team Name'] == self.team_filter)
        ]
        
        corners_der = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] < 1) &
            (self.corner_data['Team Name'] == self.team_filter)
        ]
        
        corners_combinados = pd.concat([corners_izq, corners_der])
        
        cerrados = len(corners_combinados[corners_combinados['In-swinger'] == 'Sí'])
        abiertos = len(corners_combinados[corners_combinados['Out-swinger'] == 'Sí'])
        planos = len(corners_combinados[corners_combinados['Straight'] == 'Sí'])
        
        return {'Cerrados': cerrados, 'Abiertos': abiertos, 'Planos': planos}

    def get_rematadores_data(self, lado='izquierda'):
        """Obtiene datos de rematadores por lado CON IDs"""
        if lado == 'izquierda':
            corners_lado = self.corner_data[
                (self.corner_data['Corner taken'] == 'Sí') &
                (self.corner_data['y'] > 99) &
                (self.corner_data['Team Name'] == self.team_filter)
            ]
        else:
            corners_lado = self.corner_data[
                (self.corner_data['Corner taken'] == 'Sí') &
                (self.corner_data['y'] < 1) &
                (self.corner_data['Team Name'] == self.team_filter)
            ]
        
        all_events = self.df.sort_values(['Match ID', 'timeMin', 'timeSec']).reset_index(drop=True)
        rematadores_primer_contacto = []
        
        for _, corner in corners_lado.iterrows():
            corner_time = corner['timeMin'] * 60 + corner['timeSec']
            corner_match = corner['Match ID']
            corner_team = corner['Team ID']
            
            eventos_posteriores = all_events[
                (all_events['Match ID'] == corner_match) &
                (all_events['Team ID'] == corner_team) &
                (all_events['timeMin'] * 60 + all_events['timeSec'] > corner_time) &
                (all_events['timeMin'] * 60 + all_events['timeSec'] <= corner_time + 15)
            ].copy()
            
            if eventos_posteriores.empty:
                continue
            
            remates_posteriores = eventos_posteriores[
                eventos_posteriores['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])
            ]
            
            for _, remate in remates_posteriores.iterrows():
                tiempo_desde_corner = (remate['timeMin'] * 60 + remate['timeSec']) - corner_time
                if tiempo_desde_corner <= 4:
                    # CAMBIO: Guardar tupla (nombre, id)
                    rematadores_primer_contacto.append((remate['playerName'], remate['playerId']))
        
        if rematadores_primer_contacto:
            # CAMBIO: Agrupar por tupla (nombre, id) y contar
            rematadores_df = pd.DataFrame(rematadores_primer_contacto, columns=['name', 'id'])
            return rematadores_df.groupby(['name', 'id']).size()
        else:
            return pd.Series([], dtype='int64', name=0).reindex(pd.MultiIndex.from_tuples([], names=['name', 'id']))

    def get_rematadores_data_simple(self, lado='izquierda'):
        """Versión simple para compatibilidad con conexiones"""
        rematadores_with_id = self.get_rematadores_data(lado)
        
        # Convertir de (nombre, id) a solo nombre
        rematadores_simple = {}
        for (nombre, player_id), count in rematadores_with_id.items():
            if nombre in rematadores_simple:
                rematadores_simple[nombre] += count
            else:
                rematadores_simple[nombre] = count
        
        return pd.Series(rematadores_simple)

    def load_team_logo(self, equipo, target_size=(80, 80)):
        """Carga logo del equipo"""
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
        return None

    def load_background(self):
        """Carga imagen de fondo"""
        return plt.imread("assets/fondo_informes.png") if os.path.exists("assets/fondo_informes.png") else None

    def create_sankey_unificado_matplotlib(self, ax):
        """Versión matplotlib del Sankey unificado"""
        ax.clear()
        
        # Obtención de datos
        corners_izq = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] > 99) &
            (self.corner_data['Team Name'] == self.team_filter) 
        ]
        corners_der = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] < 1) &
            (self.corner_data['Team Name'] == self.team_filter)
        ]
        corners_combinados = pd.concat([corners_izq, corners_der])
        
        if corners_combinados.empty:
            ax.text(0.5, 0.5, 'FLUJO DE CÓRNERS UNIFICADO\n(IZQUIERDA + DERECHA)\n\n(Sin datos)', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            ax.axis('off')
            return
        
        photos_data = self.load_player_photos()
        lanzadores_data = corners_combinados['playerName'].value_counts()
        tipos_data = self.get_tipo_lanzamiento_data_unificado()
        
        # Calcular zonas
        zonas_combinadas = {'zona_1': 0, 'zona_2': 0, 'zona_3': 0, 'zona_4': 0, 
                        'zona_5': 0, 'zona_6': 0, 'zona_7': 0}
        
        for _, corner in corners_combinados.iterrows():
            zona = self.get_zona_from_coordinates(corner['Pass End X'], corner['Pass End Y'])
            if zona:
                zonas_combinadas[zona] += 1
        
        rematadores_izq = self.get_rematadores_data_simple('izquierda')
        rematadores_der = self.get_rematadores_data_simple('derecha')
        rematadores_combinados = rematadores_izq.add(rematadores_der, fill_value=0)
        rematadores_data = rematadores_combinados

        # Cálculo de conexiones
        lanzador_tipo_links = {}
        tipo_zona_links = {}
        zona_rematador_links = {}
        lanzador_zona_directa_links = {}

        # Conexión 1: Lanzador -> Tipo (y directa a Zona si sin tipo)
        for _, corner in corners_combinados.iterrows():
            lanzador = corner['playerName']
            tipo = None
            if corner['In-swinger'] == 'Sí': tipo = 'Cerrados'
            elif corner['Out-swinger'] == 'Sí': tipo = 'Abiertos'
            elif corner['Straight'] == 'Sí': tipo = 'Planos'
            
            if tipo:
                if lanzador not in lanzador_tipo_links:
                    lanzador_tipo_links[lanzador] = {}
                lanzador_tipo_links[lanzador][tipo] = lanzador_tipo_links[lanzador].get(tipo, 0) + 1
            else:
                zona = self.get_zona_from_coordinates(corner['Pass End X'], corner['Pass End Y'])
                if zona:
                    if lanzador not in lanzador_zona_directa_links:
                        lanzador_zona_directa_links[lanzador] = {}
                    lanzador_zona_directa_links[lanzador][zona] = lanzador_zona_directa_links[lanzador].get(zona, 0) + 1

        # Conexión 2: Tipo -> Zona
        for _, corner in corners_combinados.iterrows():
            tipo = None
            if corner['In-swinger'] == 'Sí': tipo = 'Cerrados'
            elif corner['Out-swinger'] == 'Sí': tipo = 'Abiertos'
            elif corner['Straight'] == 'Sí': tipo = 'Planos'
            
            if tipo:
                zona = self.get_zona_from_coordinates(corner['Pass End X'], corner['Pass End Y'])
                if zona:
                    if tipo not in tipo_zona_links:
                        tipo_zona_links[tipo] = {}
                    tipo_zona_links[tipo][zona] = tipo_zona_links[tipo].get(zona, 0) + 1

        # Conexión 3: Zona -> Rematador
        remates = self.corner_data[
            (self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])) &
            (self.corner_data['From corner'] == 'Sí') &
            (self.corner_data['Team Name'] == self.team_filter) 
        ]

        for _, remate in remates.iterrows():
            rematador = remate['playerName']
            zona = self.get_zona_from_coordinates(remate['x'], remate['y'])
            if zona:
                if zona not in zona_rematador_links:
                    zona_rematador_links[zona] = {}
                zona_rematador_links[zona][rematador] = zona_rematador_links[zona].get(rematador, 0) + 1
        # DEPURACIÓN: Verificar datos antes de dibujar
        print("=== DEPURACIÓN DATOS ===")
        print(f"Corners combinados shape: {corners_combinados.shape}")
        print(f"Columnas disponibles: {corners_combinados.columns.tolist()}")

        # Verificar que tenemos playerIds
        if 'playerId' not in corners_combinados.columns:
            print("❌ ERROR: No se encuentra columna 'playerId'")
            return

        # Mostrar algunos ejemplos
        print("\n--- LANZADORES ENCONTRADOS ---")
        lanzadores_test = corners_combinados[['playerName', 'playerId']].drop_duplicates()
        for idx, row in lanzadores_test.head().iterrows():
            dorsal = self.get_player_dorsal(row['playerId'])
            print(f"Jugador: {row['playerName']}, ID: {row['playerId']}, Dorsal: {dorsal}")

        print("=== FIN DEPURACIÓN ===\n")

        # Dibujo de nodos CON SISTEMA DINÁMICO
        x_pos = [0.1, 0.35, 0.6, 0.85]

        # Obtener datos con IDs
        lanzadores_data = corners_combinados.groupby(['playerName', 'playerId']).size()
        rematadores_izq = self.get_rematadores_data('izquierda')
        rematadores_der = self.get_rematadores_data('derecha')
        rematadores_combinados = rematadores_izq.add(rematadores_der, fill_value=0)

        # Calcular parámetros dinámicos
        sizing_params = self.calculate_dynamic_sizing(
            len(lanzadores_data), 
            len(rematadores_combinados)
        )

        # Nivel 1: Lanzadores CON RECUADROS - CORREGIDO
        lanzadores_data_with_id = corners_combinados.groupby(['playerName', 'playerId']).size()
        # Obtener solo TOP 5 lanzadores
        lanzadores_top4 = lanzadores_data_with_id.nlargest(4)
        lanzadores_items = list(lanzadores_top4.items())

        # CREAR MAPAS COMPATIBLES para conexiones
        lanzadores_y_map = {}  # Solo con nombres para conexiones
        lanzadores_data = {}   # Solo con nombres para conexiones

        # Crear mapa de colores para cada lanzador (TOP 5)
        colores_jugadores = ['#2C3E50', '#6A1B9A', '#F39C12', '#27AE60', '#8B4513']
        lanzadores_colors = {}
        for i, ((player_name, player_id), count) in enumerate(lanzadores_items):
            lanzadores_colors[player_name] = colores_jugadores[i % len(colores_jugadores)]

        if lanzadores_items:
            # USAR toda la altura disponible de forma inteligente
            altura_total = sizing_params['altura_disponible']
            margen_top = 0.05 * (1 + 1/len(lanzadores_items))  # Margen dinámico
            margen_bottom = 0.05 * (1 + 1/len(lanzadores_items))
    
            inicio_y = 0.9 - margen_top
            fin_y = 0.1 + margen_bottom
            
            lanzadores_y = np.linspace(inicio_y, fin_y, len(lanzadores_items))

            
            for i, ((player_name, player_id), count) in enumerate(lanzadores_items):
                # Dibujar recuadro
                self.draw_artistic_player_box(
                    ax, x_pos[0], lanzadores_y[i], 
                    player_name, player_id, photos_data, 
                    sizing_params, is_shooter=True,
                    count_value=count
                )
                
                # CREAR MAPAS para conexiones (solo nombre)
                lanzadores_y_map[player_name] = lanzadores_y[i]
                lanzadores_data[player_name] = count

        # Nivel 2: Tipos (SIN CAMBIOS en esta parte)
        tipos_items = [item for item in tipos_data.items() if item[1] > 0]
        tipos_y = np.linspace(0.8, 0.2, len(tipos_items)) if tipos_items else []
        tipos_y_map = {item[0]: y for item, y in zip(tipos_items, tipos_y)}
        tipos_colors = {'Cerrados': '#E74C3C', 'Abiertos': '#3498DB', 'Planos': '#F39C12'}

        for i, (tipo, count) in enumerate(tipos_items):
            rad = {'Cerrados': 0.3, 'Abiertos': -0.3, 'Planos': 0.0}.get(tipo)
            grosor = 3 + (count / (sum(tipos_data.values()) or 1)) * 25
            arrow = patches.FancyArrowPatch((x_pos[1] - 0.05, tipos_y[i]), (x_pos[1] + 0.05, tipos_y[i]), connectionstyle=f"arc3,rad={rad}", color=tipos_colors[tipo], linewidth=grosor, arrowstyle='->,head_length=10,head_width=8', zorder=3)
            ax.add_patch(arrow)
            ax.text(x_pos[1], tipos_y[i] + 0.08, f'{tipo}\n({count})', ha='center', va='center', fontsize=12, fontweight='bold', color='#2C3E50')

        # Nivel 3: Zonas (SIN CAMBIOS)
        zonas_con_datos = sorted([item for item in zonas_combinadas.items() if item[1] > 0], key=lambda x: x[1], reverse=True)
        zonas_y = np.linspace(0.85, 0.15, len(zonas_con_datos)) if zonas_con_datos else []
        zonas_y_map = {item[0]: y for item, y in zip(zonas_con_datos, zonas_y)}
        zonas_colors_map = {'zona_1': '#FF6B6B', 'zona_2': '#4ECDC4', 'zona_3': '#45B7D1', 'zona_4': '#96CEB4', 'zona_5': '#FECA57', 'zona_6': '#FF9FF3', 'zona_7': '#54A0FF'}

        for i, (zona_key, count) in enumerate(zonas_con_datos):
            zona_nombre = self.ZONA_NAMES.get(zona_key, zona_key)
            base_radius = 0.04 + (count / (sum(zonas_combinadas.values()) or 1)) * 0.18
            color = zonas_colors_map.get(zona_key, 'gray')
            outer_circle = patches.Circle((x_pos[2], zonas_y[i]), base_radius, facecolor=color, alpha=0.9, zorder=3)
            ax.add_patch(outer_circle)
            middle_circle = patches.Circle((x_pos[2], zonas_y[i]), base_radius * 0.7, facecolor='white', alpha=0.8, zorder=4)
            ax.add_patch(middle_circle)
            inner_circle = patches.Circle((x_pos[2], zonas_y[i]), base_radius * 0.4, facecolor=color, alpha=1.0, zorder=5)
            ax.add_patch(inner_circle)
            text_element = ax.text(x_pos[2], zonas_y[i], f"{zona_nombre}\n({count})", ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=6, path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])

        # Nivel 4: Rematadores CON RECUADROS - CORREGIDO  
        rematadores_izq = self.get_rematadores_data('izquierda')
        rematadores_der = self.get_rematadores_data('derecha')
        rematadores_combinados = rematadores_izq.add(rematadores_der, fill_value=0)
        rematadores_items = list(rematadores_combinados.items())

        # CREAR MAPAS COMPATIBLES para conexiones
        rematadores_y_map = {}  # Solo con nombres para conexiones  
        rematadores_data = {}   # Solo con nombres para conexiones

        if rematadores_items:
            altura_total = sizing_params['altura_disponible'] * sizing_params['spacing_factor']
            rematadores_y = np.linspace(0.85, 0.15, len(rematadores_items))  # USAR RANGO COMPLETO
            
            for i, ((player_name, player_id), count) in enumerate(rematadores_items):
                # Dibujar recuadro
                self.draw_artistic_player_box(
                    ax, x_pos[3], rematadores_y[i], 
                    player_name, player_id, photos_data, 
                    sizing_params, is_shooter=False,
                    count_value=count
                )
                
                # CREAR MAPAS para conexiones (solo nombre)
                rematadores_y_map[player_name] = rematadores_y[i]
                rematadores_data[player_name] = count

        # Dibujo de conexiones
        
        # Conexión 1: Lanzadores -> Tipos
        total_lanzador_tipo_links = sum(sum(d.values()) for d in lanzador_tipo_links.values()) or 1
        for lanzador, tipos in lanzador_tipo_links.items():
            if lanzador in lanzadores_y_map and lanzador in lanzadores_colors:
                for tipo, count in tipos.items():
                    if tipo in tipos_y_map:
                        grosor = 2 + (count / total_lanzador_tipo_links) * 18
                        arrow = patches.FancyArrowPatch((x_pos[0] + 0.02, lanzadores_y_map[lanzador]), (x_pos[1] - 0.06, tipos_y_map[tipo]),
                            connectionstyle="arc3,rad=0.1", color=lanzadores_colors[lanzador], alpha=0.7, linewidth=grosor, zorder=1)
                        ax.add_patch(arrow)

        # Conexión directa: Lanzadores -> Zonas
        total_lanzador_zona_directa = sum(sum(d.values()) for d in lanzador_zona_directa_links.values()) or 1
        for lanzador, zonas in lanzador_zona_directa_links.items():
            if lanzador in lanzadores_y_map and lanzador in lanzadores_colors:
                for zona, count in zonas.items():
                    if zona in zonas_y_map:
                        grosor = 2 + (count / total_lanzador_zona_directa) * 18
                        arrow = patches.FancyArrowPatch(
                            (x_pos[0] + 0.02, lanzadores_y_map[lanzador]), 
                            (x_pos[2] - 0.05, zonas_y_map[zona]),
                            connectionstyle="arc3,rad=0.3", 
                            color=lanzadores_colors[lanzador], alpha=0.7, linewidth=grosor, zorder=1)
                        ax.add_patch(arrow)

        # Conexión 2: Tipos -> Zonas
        total_tipo_zona_links = sum(sum(d.values()) for d in tipo_zona_links.values()) or 1
        for _, corner in corners_combinados.iterrows():
            lanzador = corner['playerName']
            if lanzador not in lanzadores_colors:  # Solo TOP 5
                continue
                
            tipo = None
            if corner['In-swinger'] == 'Sí': tipo = 'Cerrados'
            elif corner['Out-swinger'] == 'Sí': tipo = 'Abiertos'
            elif corner['Straight'] == 'Sí': tipo = 'Planos'
            
            if tipo and tipo in tipos_y_map:
                zona = self.get_zona_from_coordinates(corner['Pass End X'], corner['Pass End Y'])
                if zona and zona in zonas_y_map:
                    rad = {'Cerrados': 0.2, 'Abiertos': -0.2, 'Planos': 0.0}.get(tipo, 0)
                    grosor = 2
                    arrow = patches.FancyArrowPatch((x_pos[1] + 0.06, tipos_y_map[tipo]), (x_pos[2] - 0.05, zonas_y_map[zona]),
                        connectionstyle=f"arc3,rad={rad}", color=lanzadores_colors[lanzador], alpha=0.7, linewidth=grosor, zorder=1)
                    ax.add_patch(arrow)

        # Conexión 3: Zonas -> Rematadores
        total_zona_rematador_links = sum(sum(d.values()) for d in zona_rematador_links.values()) or 1
        # Rastrear zona->rematador por lanzador original
        for _, corner in corners_combinados.iterrows():
            lanzador = corner['playerName']
            if lanzador not in lanzadores_colors:  # Solo TOP 5
                continue
                
            corner_time = corner['timeMin'] * 60 + corner['timeSec']
            corner_match = corner['Match ID']
            corner_team = corner['Team ID']
            
            # Buscar remates de este corner específico
            all_events = self.df.sort_values(['Match ID', 'timeMin', 'timeSec']).reset_index(drop=True)
            eventos_posteriores = all_events[
                (all_events['Match ID'] == corner_match) &
                (all_events['Team ID'] == corner_team) &
                (all_events['timeMin'] * 60 + all_events['timeSec'] > corner_time) &
                (all_events['timeMin'] * 60 + all_events['timeSec'] <= corner_time + 15)
            ]
            
            remates_posteriores = eventos_posteriores[
                eventos_posteriores['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])
            ]
            
            for _, remate in remates_posteriores.iterrows():
                tiempo_desde_corner = (remate['timeMin'] * 60 + remate['timeSec']) - corner_time
                if tiempo_desde_corner <= 4:
                    zona = self.get_zona_from_coordinates(remate['x'], remate['y'])
                    rematador = remate['playerName']
                    
                    if zona in zonas_y_map and rematador in rematadores_y_map:
                        grosor = 2
                        arrow = patches.FancyArrowPatch((x_pos[2] + 0.05, zonas_y_map[zona]), (x_pos[3] - 0.07, rematadores_y_map[rematador]),
                            connectionstyle="arc3,rad=0.1", color=lanzadores_colors[lanzador], alpha=0.7, linewidth=grosor, zorder=1)
                        ax.add_patch(arrow)
                
        # Etiquetas y título
        titulos = ['LANZADORES', 'TIPOS', 'ZONAS', 'REMATADORES']
        colores_modernos = ['#667eea', '#764ba2', '#f093fb', '#f5576c']

        for i, titulo in enumerate(titulos):
            ax.text(x_pos[i] + 0.002, 1.08 - 0.002, titulo, ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='black', alpha=0.3, zorder=8)
            
            ax.text(x_pos[i], 1.08, titulo, ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='white', zorder=10,
                    bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.02", facecolor=colores_modernos[i], 
                            alpha=0.95, edgecolor='white', linewidth=2))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.12)
        ax.set_title('FLUJO CÓRNERS UNIFICADO (IZQUIERDA + DERECHA)', 
                    fontsize=16, fontweight='bold', color='#1ABC9C', pad=20)
        ax.axis('off')

    def create_reporte_flujo(self, figsize=(11.69, 8.27)):    
        """Crea reporte solo con el flujo de córners ocupando toda la página"""
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        # Fondo
        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25)
            ax_bg.axis('off')
        
        # Título principal
        fig.suptitle(f'FLUJO DE CÓRNERS OFENSIVOS', 
                    fontsize=20, fontweight='bold', color='#1e3d59', y=0.95)
        
        # Logo del equipo
        if self.team_filter and (team_logo := self.load_team_logo(self.team_filter)) is not None:
            ax_team = fig.add_axes([0.90, 0.89, 0.07, 0.07]) 
            ax_team.imshow(team_logo, aspect='auto')
            ax_team.axis('off')

        # Sankey ocupando toda la página (con márgenes)
        ax_flujo = fig.add_axes([0.05, 0.05, 0.95, 0.80])  # [left, bottom, width, height]
        self.create_sankey_unificado_matplotlib(ax_flujo)

        return fig

# Funciones auxiliares
def seleccionar_equipo_interactivo():
    """Selección interactiva de equipo"""
    try:
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/abp_events.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        if not equipos: 
            print("No se encontraron equipos.")
            return None
        
        print("\n=== SELECCIÓN DE EQUIPO ===")
        for i, equipo in enumerate(equipos, 1): 
            print(f"{i}. {equipo}")
        
        while True:
            try:
                indice = int(input(f"\nSelecciona un equipo (1-{len(equipos)}): ").strip()) - 1
                if 0 <= indice < len(equipos): 
                    return equipos[indice]
                else: 
                    print(f"Por favor, ingresa un número entre 1 y {len(equipos)}")
            except ValueError: 
                print("Por favor, ingresa un número válido")
    except Exception as e: 
        print(f"Error en la selección: {e}")
        return None

def main():
    """Función principal"""
    try:
        print("=== GENERADOR DE REPORTE FLUJO CÓRNERS ===")
        if (equipo := seleccionar_equipo_interactivo()) is None:
            print("No se pudo completar la selección.")
            return
        
        print(f"\nGenerando reporte de flujo para {equipo}")
        analyzer = ReporteFlujoCorners(team_filter=equipo)
        
        if (fig := analyzer.create_reporte_flujo()):
            plt.show()
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_flujo_corners_{equipo_filename}.pdf"
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1, 
                       facecolor='white', dpi=300, orientation='landscape')
            print(f"✅ Reporte guardado como: {output_path}")
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()

def generar_reporte_personalizado(equipo, mostrar=True, guardar=True):
    """Genera reporte personalizado para un equipo específico"""
    try:
        analyzer = ReporteUnificadoCorners(team_filter=equipo)
        fig = analyzer.create_reporte_unificado()
        
        if fig:
            if mostrar: 
                plt.show()
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"reporte_unificado_corners_{equipo_filename}.pdf"
                # ← CAMBIO: Parámetros de guardado para A4 horizontal
                fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1, 
                           facecolor='white', dpi=300, orientation='landscape')  # ← AGREGAR orientation
                print(f"✅ Reporte guardado como: {output_path}")
            return fig

        else:
            print("❌ No se pudo generar la visualización")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()