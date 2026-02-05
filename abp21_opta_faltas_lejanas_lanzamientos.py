import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import patheffects
import seaborn as sns
import numpy as np
import os
from matplotlib.gridspec import GridSpec
from mplsoccer import VerticalPitch
from difflib import SequenceMatcher
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.stats import gaussian_kde
except ImportError:
    pass

class FreeKicksAnalysis:
    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/abp_events.parquet", team_filter=None):
        self.data_path = data_path
        self.team_filter = team_filter
        self.df = None
        self.free_kicks_data = pd.DataFrame()
        self.team_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/team_stats.parquet")
        self.player_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/player_stats.parquet")
        self.load_data(team_filter)
        
        # Definir zonas de faltas segÃºn las coordenadas proporcionadas
        self.zones = {
            1: [(100, 100), (100, 78.9), (83, 78.9), (83, 100)],
            2: [(100, 78.9), (100, 54.4), (83, 54.4), (83, 78.9)],
            3: [(100, 50), (100, 54.4), (83, 50), (83, 54.4)],
            4: [(100, 45.6), (100, 50), (83, 45.6), (83, 50)],
            5: [(100, 45.6), (100, 21.1), (83, 21.1), (83, 45.6)],
            6: [(100, 21.1), (100, 0), (83, 0), (83, 21.1)],
            7: [(83, 100), (65, 100), (83, 78.9), (65, 78.9)],
            8: [(83, 78.9), (83, 63.2), (65, 78.9), (65, 63.2)],
            9: [(83, 63.2), (83, 36.8), (65, 63.2), (65, 36.8)],
            10: [(65, 36.8), (83, 36.8), (83, 21.1), (65, 21.1)],
            11: [(83, 21.1), (65, 21.1), (83, 0), (65, 0)],
            12: [(65, 100), (50, 100), (50, 78.9), (65, 78.9)],
            13: [(50, 78.9), (65, 78.9), (50, 21.1), (65, 21.1)], 
            14: [(50, 21.1), (65, 21.1), (50, 0), (65, 0)]
        }
        
        if team_filter:
            if self.df is not None:  # â† AÃ‘ADIR ESTA VERIFICACIÃ“N
                self.df = self.df.merge(self.team_stats[['Team ID', 'Team Position']], 
                            on='Team ID', how='left')
                self.extract_free_kicks_data(team_filter)
            else:
                pass

    def guardar_sin_espacios(self, fig, filename):
        """Guarda sin espacios manteniendo landscape A4"""
        fig.set_size_inches(11.69, 8.27)
        
        fig.savefig(
            filename,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0,
            facecolor='white',
            edgecolor='none',
            format='pdf' if filename.endswith('.pdf') else 'png',
            transparent=False,
            orientation='landscape'
        )

    def format_player_name_multiline(self, player_name, max_chars_per_line=12):
        """Divide nombres largos en 2 lÃ­neas de forma inteligente"""
        words = player_name.split()
        
        # Si es una sola palabra
        if len(words) == 1:
            # Solo dividir si es muy larga
            if len(player_name) > max_chars_per_line:
                mid = len(player_name) // 2
                return player_name[:mid], player_name[mid:]
            else:
                return player_name, None
        
        # Si hay mÃºltiples palabras, SIEMPRE dividir en dos lÃ­neas
        # Estrategia: primera palabra en lÃ­nea 1, resto en lÃ­nea 2
        line1 = words[0]
        line2 = ' '.join(words[1:])
        
        # Si la primera palabra es muy larga, truncarla
        if len(line1) > max_chars_per_line:
            line1 = line1[:max_chars_per_line-3] + '...'
        
        # Si la segunda lÃ­nea es muy larga, truncarla
        if len(line2) > max_chars_per_line:
            line2 = line2[:max_chars_per_line-3] + '...'
        
        return line1, line2
    
    def create_team_report(self, team_name, zone_filter='Center', x_threshold=60, figsize=(11.69, 8.27)):
        """
        Método wrapper para generar reporte completo de un equipo.
        Compatible con generador maestro Y ejecución standalone.
        """
        from matplotlib.gridspec import GridSpec
        
        # Filtrar por equipo
        team_matches = self.team_stats[self.team_stats['Team Name'] == team_name]['Match ID'].unique()
        team_events = self.df[self.df['Match ID'].isin(team_matches)]
        
        # Ordenar y resetear índice para análisis de secuencia
        team_events = team_events.sort_values(['Match ID', 'periodId', 'timeStamp']).reset_index(drop=True)

        # Buscar faltas del tipo solicitado
        free_kicks = team_events[
            (team_events['Free kick taken'] == 'Sí') & 
            (team_events['Team Name'] == team_name) &
            (team_events['Zone'] == zone_filter) &
            (team_events['x'] > x_threshold)
        ].copy()
        
        if len(free_kicks) == 0:
            pass
            return None
        
        # Crear figura
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Título
        fig.suptitle(f'ANÁLISIS FALTAS {zone_filter.upper()} - {team_name}', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Análisis por falta
        resultados = []
        for idx, falta in free_kicks.iterrows():
            match_events = team_events[team_events['Match ID'] == falta['Match ID']].reset_index(drop=True)
            resultado = self.analyze_free_kick_sequence(match_events, falta)
            if resultado:
                resultados.append(resultado)
        
        # Gráfico 1: Mapa de faltas
        ax1 = fig.add_subplot(gs[0, :])
        pitch = VerticalPitch(pitch_type='opta', pitch_color='#2d5a27', line_color='white')
        pitch.draw(ax=ax1)
        
        # Plotear ubicaciones
        for falta, res in zip(free_kicks.itertuples(), resultados):
            color = '#FFD700' if res.get('resultado') == 'Gol' else '#95A5A6'
            ax1.scatter(falta.y, falta.x, c=color, s=100, edgecolors='white', linewidth=2)
        
        ax1.set_title(f'Ubicación de {len(free_kicks)} faltas', fontsize=12, pad=10)
        
        # Gráfico 2: Resultados
        ax2 = fig.add_subplot(gs[1, 0])
        resultados_count = pd.Series([r.get('resultado', 'Otro') for r in resultados]).value_counts()
        resultados_count.plot(kind='bar', ax=ax2, color='#3498DB')
        ax2.set_title('Resultados', fontsize=11)
        ax2.set_xlabel('')
        ax2.tick_params(axis='x', rotation=45)
        
        # Gráfico 3: Estadísticas
        ax3 = fig.add_subplot(gs[1, 1])
        stats_text = f"Total faltas: {len(free_kicks)}\n"
        stats_text += f"Analizadas: {len(resultados)}\n"
        stats_text += f"Zona: {zone_filter}\n"
        stats_text += f"x > {x_threshold}"
        ax3.text(0.5, 0.5, stats_text, ha='center', va='center', 
                fontsize=12, transform=ax3.transAxes)
        ax3.axis('off')
        
        return fig

    def load_player_photos(self):
        """Carga el JSON con las fotos de jugadores"""
        import json
        try:
            with open('assets/jugadores_optimizados.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            pass
            return []

    def get_player_photo_with_team_filter(self, player_name, photos_data, team_filter):
        """Obtiene la foto del jugador con filtro de equipo"""
        match = self.match_player_name(player_name, photos_data, team_filter)
        if not match:
            return None
        
        try:
            import base64
            from io import BytesIO
            from PIL import Image
            
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
            
            # Flood fill ITERATIVO para quitar fondo blanco
            def flood_fill_iterative(start_points, threshold=235):
                visited = np.zeros((height, width), dtype=bool)
                background_mask = np.zeros((height, width), dtype=bool)
                
                def is_background_color(y, x):
                    if y < 0 or y >= height or x < 0 or x >= width:
                        return False
                    return (data[y, x, 0] >= threshold and 
                            data[y, x, 1] >= threshold and 
                            data[y, x, 2] >= threshold)
                
                # Usar pila en lugar de recursiÃ³n
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
                        
                        # AÃ±adir vecinos a la pila (4-connected)
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
            
            # Aplicar la mÃ¡scara para hacer transparente el fondo
            data[background_mask] = [0, 0, 0, 0]
            
            return data.astype(np.float32) / 255.0
        
        except Exception as e:
            pass
            return None

    def match_player_name(self, player_name, photos_data, team_filter=None):
        """Encuentra el nombre mÃ¡s parecido en los datos de las fotos"""
        import re
        
        def normalize_name(name):
            """Normaliza el nombre eliminando acentos, puntuaciÃ³n y convirtiendo a minÃºsculas."""
            name = name.lower().strip()
            replacements = {
                'Ã¡': 'a', 'Ã©': 'e', 'Ã­': 'i', 'Ã³': 'o', 'Ãº': 'u',
                'Ã ': 'a', 'Ã¨': 'e', 'Ã¬': 'i', 'Ã²': 'o', 'Ã¹': 'u',
                'Ã¤': 'a', 'Ã«': 'e', 'Ã¯': 'i', 'Ã¶': 'o', 'Ã¼': 'u',
                'Ã¢': 'a', 'Ãª': 'e', 'Ã®': 'i', 'Ã´': 'o', 'Ã»': 'u',
                'Ã±': 'n', 'Ã§': 'c'
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
            """Calcula el score de coincidencia con reglas especÃ­ficas."""
            player_full = player_parts['full']
            photo_full = photo_parts['full']

            # 1. MATCH EXACTO COMPLETO
            if player_full == photo_full:
                return (1.0, "MATCH EXACTO COMPLETO")

            # 2. MATCH DE NOMBRE ÃšNICO / APODO
            if len(player_parts['all_parts']) == 1:
                if player_full in photo_parts['all_parts']:
                    return (0.95, f"MATCH DE NOMBRE ÃšNICO '{player_full}'")

            # 3. MATCH DE INICIAL + APELLIDO
            if (len(player_parts['first_name']) == 1 and
                player_parts['last_name'] == photo_parts['last_name'] and
                photo_parts['first_name'].startswith(player_parts['first_name'])):
                return (0.90, "INICIAL + APELLIDO EXACTO")

            # 4. MATCH DE NOMBRE + INICIAL DE APELLIDO
            if (player_parts['first_name'] == photo_parts['first_name'] and
                len(player_parts['last_name']) == 1 and
                photo_parts['last_name'].startswith(player_parts['last_name'][0])):
                return (0.90, "NOMBRE + INICIAL APELLIDO")

            # 5. MATCH DE APELLIDO EXACTO Y NOMBRE SIMILAR
            if player_parts['last_name'] == photo_parts['last_name']:
                first_name_sim = SequenceMatcher(None, player_parts['first_name'], photo_parts['first_name']).ratio()
                if first_name_sim >= 0.8:
                    return (0.85 + (first_name_sim * 0.05), f"APELLIDO EXACTO + NOMBRE SIMILAR ({first_name_sim:.2f})")
            
            # 6. Fallback a similitud general
            full_sim = SequenceMatcher(None, player_full, photo_full).ratio()
            if full_sim > 0.8:
                return (full_sim, f"SIMILITUD GENERAL ALTA ({full_sim:.2f})")

            return (0.0, "SIN COINCIDENCIA CLARA")

        # LÃ³gica principal
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
                    
        # LÃ³gica de desambiguaciÃ³n
        if len(found_matches) == 1:
            return found_matches[0]['entry']
        elif len(found_matches) > 1:
            return None  # Ambiguo
        else:
            return None  # No encontrado

    def get_faltas_ranking_data(self, team_filter=None):
        """Obtiene datos de ranking de lanzadores de faltas con remates"""
        from collections import defaultdict
        
        if self.free_kicks_data.empty:
            return {}
        
        # Filtrar por equipo Y por zonas de lanzamiento (8, 9, 10)
        team_data = self.free_kicks_data[
            (self.free_kicks_data['Team Name'] == team_filter) &
            (self.free_kicks_data['zone'].isin([12, 13, 14]))  # â† AÃ‘ADIR ESTA LÃNEA
        ] if team_filter else self.free_kicks_data[
            self.free_kicks_data['zone'].isin([12, 13, 14])     # â† AÃ‘ADIR ESTA LÃNEA
        ]
        
        # Diccionario para acumular estadÃ­sticas por jugador
        stats_por_jugador = defaultdict(lambda: {
            'lanzamientos': 0,
            'lanzamientos_exitosos': 0,  # outcome = 1
            'remates': 0,  # result_type en ['Gol', 'Tiro a puerta', 'Tiro fuera', 'Poste']
            'goles': 0
        })
        
        for _, falta in team_data.iterrows():
            player_name = falta['player_name']
            if pd.isna(player_name):
                continue
                
            # Contar lanzamiento
            stats_por_jugador[player_name]['lanzamientos'] += 1
            
            # Contar si fue exitoso (outcome = 1)
            if falta.get('outcome') == 1:
                stats_por_jugador[player_name]['lanzamientos_exitosos'] += 1
            
            # Contar remates
            if falta.get('result_type') in ['Gol', 'Tiro a puerta', 'Tiro fuera', 'Poste']:
                stats_por_jugador[player_name]['remates'] += 1
                
            # Contar goles
            if falta.get('result_type') == 'Gol':
                stats_por_jugador[player_name]['goles'] += 1
        
        # Calcular efectividad y convertir a lista ordenada
        ranking = []
        for player_name, stats in stats_por_jugador.items():
            efectividad = (stats['lanzamientos_exitosos'] / stats['lanzamientos'] * 100) if stats['lanzamientos'] > 0 else 0
            
            ranking.append({
                'player_name': player_name,
                'lanzamientos': stats['lanzamientos'],
                'remates': stats['remates'],
                'goles': stats['goles'],
                'efectividad': efectividad
            })
        
        # Ordenar por nÃºmero de lanzamientos (criterio principal)
        ranking.sort(key=lambda x: x['lanzamientos'], reverse=True)
        
        return ranking

    def create_faltas_ranking(self, ax, team_filter):
        """Crea el ranking visual de jugadores lanzadores de faltas"""
        ranking_data = self.get_faltas_ranking_data(team_filter)
        photos_data = self.load_player_photos()
        
        if not ranking_data:
            ax.text(0.5, 0.5, 'RANKING LANZADORES\nDE FALTAS\n\n(Sin datos)', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        ax.set_facecolor('#f8f9fa')
        top_players = ranking_data[:5]  # Top 5
        
        # TÃ­tulo
        ax.text(0.5, 1.00, 'MEJORES LANZADORES\n FALTAS LEJANAS', fontsize=10, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes, color='#2c3e50')
        
        position_colors = ['#e74c3c', '#f39c12', '#f1c40f', '#95a5a6', '#bdc3c7']
        
        for i, player_data in enumerate(top_players):
            y_pos = 0.80 - (i * 0.18)
            player_name = player_data['player_name']
            lanzamientos = player_data['lanzamientos']
            remates = player_data['remates']
            efectividad = player_data['efectividad']
            
            # Fondo del jugador
            rect_bg = patches.FancyBboxPatch((0.02, y_pos - 0.08), 0.96, 0.16,
                                            boxstyle="round,pad=0.01", 
                                            facecolor='white',
                                            edgecolor=position_colors[i],
                                            linewidth=2, alpha=0.9)
            ax.add_patch(rect_bg)
            
            # NÃºmero de posiciÃ³n
            ax.text(0.08, y_pos, f"#{i+1}", fontsize=12, fontweight='bold', 
                    va='center', ha='center', color=position_colors[i])
            
            # Foto del jugador
            player_photo = self.get_player_photo_with_team_filter(player_name, photos_data, team_filter)
            if player_photo is not None:
                photo_ax = ax.inset_axes([0.10, y_pos - 0.085, 0.25, 0.20])
                photo_ax.imshow(player_photo, aspect='auto')
                photo_ax.axis('off')
            
            # Nombre del jugador (multilinea)
            line1, line2 = self.format_player_name_multiline(player_name, max_chars_per_line=12)

            if line2 is None:
                # Nombre corto - una lÃ­nea centrada
                ax.text(0.35, y_pos + 0.02, line1, fontsize=10, fontweight='bold', 
                        va='center', ha='left', color='#2c3e50')
            else:
                # Nombre largo - dos lÃ­neas
                ax.text(0.35, y_pos + 0.040, line1, fontsize=9, fontweight='bold', 
                        va='center', ha='left', color='#2c3e50')
                ax.text(0.35, y_pos + 0.005, line2, fontsize=9, fontweight='bold', 
                        va='center', ha='left', color='#2c3e50')
            
            # EstadÃ­sticas
            ax.text(0.45, y_pos - 0.04, f'{lanzamientos} lanz. | {remates} rem.', 
                    fontsize=8, va='center', ha='left', color='#7f8c8d')
            ax.text(0.45, y_pos - 0.07, f'Efect: {efectividad:.1f}%', 
                    fontsize=8, va='center', ha='left', color='#27ae60' if efectividad >= 50 else '#e74c3c')
            
            # Sistema de estrellas basado en efectividad
            if efectividad >= 80:
                stars = 5
            elif efectividad >= 60:
                stars = 4
            elif efectividad >= 40:
                stars = 3
            elif efectividad >= 20:
                stars = 2
            else:
                stars = 1
                
            star_text = '★' * stars
            ax.text(0.85, y_pos, star_text, fontsize=12, va='center', ha='center',
                    color='#f39c12')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def load_data(self, team_filter=None):
        """Carga solo los datos necesarios desde el inicio"""
        try:
            # Cargar solo columnas necesarias
            columns_needed = ['Match ID', 'periodId', 'Team ID', 'Team Name', 'Event Name', 'outcome',
                'timeMin', 'timeSec', 'timeStamp', 'x', 'y', 'Pass End X', 'Pass End Y', 
                'playerName', 'playerId', 'Free kick taken', 'Zone']
            
            self.df = pd.read_parquet(self.data_path, columns=columns_needed)

            # Eliminar duplicados exactos
            self.df = self.df.drop_duplicates(subset=['Match ID', 'timeMin', 'timeSec', 'Event Name', 'playerName'], keep='first')

            self.df['timeStamp'] = self.df['timeStamp'].apply(self.normalize_timestamp)

            # Filtrar eventos 'Deleted event' despuÃ©s de la de-duplicaciÃ³n
            self.df = self.df[self.df['Event Name'] != 'Deleted event']
            
            # Si hay filtro de equipo, filtrar matches desde el inicio
            if team_filter:
                team_matches = self.team_stats[self.team_stats['Team Name'] == team_filter]['Match ID'].unique()
                self.df = self.df[self.df['Match ID'].isin(team_matches)]
            

            # *** AÃ‘ADIR ESTE DEBUG ***
            if 'Free kick taken' in self.df.columns:
                pass
                fk_values = self.df['Free kick taken'].value_counts()
            
            if 'Zone' in self.df.columns:
                pass
                zone_values = self.df['Zone'].value_counts()
            
            # Contar potenciales faltas
            potential_fks = self.df[
                (self.df.get('Free kick taken', '') == 'Sí­') &
                (self.df['x'] > 50) &
                (self.df.get('Zone', '') == 'Center')
            ]
            
            if len(potential_fks) == 0:
                pass
                # Probar sin filtro Zone
                potential_fks2 = self.df[
                    (self.df.get('Free kick taken', '') == 'Sí­') &
                    (self.df['x'] > 50)
                ]
                
                if len(potential_fks2) > 0:
                    pass
        except Exception as e:
            pass

    def normalize_timestamp(self, timestamp):
        """Normaliza timestamps quitando la Z final si existe"""
        if pd.isna(timestamp):
            return timestamp
        
        timestamp_str = str(timestamp).strip()
        
        # Quitar la Z final si existe
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1]
        
        try:
            # Convertir a datetime y volver a string para normalizar formato
            dt = pd.to_datetime(timestamp_str)
            return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Mantener 3 decimales
        except:
            return timestamp_str

    def point_in_zone(self, x, y, zone_coords):
        """Determina si un punto estÃ¡ dentro de una zona rectangular"""
        try:
            x, y = float(x), float(y)
            # Obtener coordenadas min y max de la zona
            x_coords = [coord[0] for coord in zone_coords]
            y_coords = [coord[1] for coord in zone_coords]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            return min_x <= x <= max_x and min_y <= y <= max_y
        except:
            return False

    def get_zone_for_coordinates(self, x, y):
        """Determina la zona para unas coordenadas dadas"""
        for zone_num, zone_coords in self.zones.items():
            if self.point_in_zone(x, y, zone_coords):
                return zone_num
        return None

    def extract_free_kicks_data(self, team_filter=None):
        """Extrae datos de faltas ofensivas lejanas desde zonas 12, 13 y 14"""
        if self.df is None:
            pass
            return


        if team_filter:
            team_matches = set(self.team_stats[self.team_stats['Team Name'] == team_filter]['Match ID'].unique())

        df_sorted = self.df.sort_values(['Match ID', 'periodId', 'timeStamp']).reset_index(drop=True)

        # --- CAMBIO 1: CALCULAR LA ZONA DE INICIO PARA TODOS LOS EVENTOS ---
        df_sorted['start_zone'] = df_sorted.apply(
            lambda row: self.get_zone_for_coordinates(row['x'], row['y']),
            axis=1
        )
        
        free_kicks_list = []

        # --- CAMBIO 2: MODIFICAR EL FILTRO PRINCIPAL ---
        # Usamos la columna 'start_zone' para filtrar por las zonas 12, 13 y 14
        free_kicks_directos = df_sorted[
            (df_sorted.get('Free kick taken', '') == 'Sí') &
            (df_sorted['start_zone'].isin([12, 13, 14])) &  # <-- ESTE ES EL NUEVO FILTRO
            (df_sorted['x'].notna()) &
            (df_sorted['y'].notna())
        ]

        if team_filter:
            free_kicks_directos = free_kicks_directos[
                free_kicks_directos['Match ID'].isin(team_matches)
            ]

        # Eliminar duplicados
        free_kicks_directos_unique = free_kicks_directos.drop_duplicates(
            subset=['Match ID', 'timeMin', 'timeSec', 'playerName'], 
            keep='first'
        ).reset_index(drop=True)

        for _, free_kick in free_kicks_directos_unique.iterrows():
            try:
                x, y = float(free_kick['x']), float(free_kick['y'])
                
                # --- CAMBIO 3: USAR LA ZONA YA CALCULADA ---
                zone = free_kick['start_zone']
                
                free_kick_data = {
                    'Match ID': free_kick['Match ID'],
                    'Team ID': free_kick['Team ID'],
                    'Team Name': free_kick['Team Name'],
                    'player_name': free_kick.get('playerName', ''),
                    'player_id': free_kick.get('playerId', ''),
                    'timeMin': free_kick['timeMin'],
                    'timeSec': free_kick['timeSec'],
                    'x': x,
                    'y': y,
                    'Pass End X': free_kick.get('Pass End X', 0),
                    'Pass End Y': free_kick.get('Pass End Y', 0),
                    'zone': zone,
                    'outcome': free_kick['outcome'],
                    'Event Name': free_kick['Event Name']
                }
                
                result_data = self.analyze_free_kick_sequence(df_sorted, free_kick)
                free_kick_data.update(result_data)
                
                free_kicks_list.append(free_kick_data)
                
            except (ValueError, TypeError) as e:
                pass
                continue

        # Crear DataFrame final
        if free_kicks_list:
            self.free_kicks_data = pd.DataFrame(free_kicks_list)

            if 'result_type' in self.free_kicks_data.columns:
                pass
                
            if 'zone' in self.free_kicks_data.columns:
                pass
        else:
            pass

    def analyze_free_kick_sequence(self, match_events, free_kick):
        """Analiza la secuencia posterior a la falta"""
        match_id = free_kick['Match ID']
        start_time_min = free_kick['timeMin']
        start_time_sec = free_kick['timeSec']
        period_id = free_kick['periodId']
        team_id = free_kick['Team ID']
        
        # Buscar el Ã­ndice del evento de la falta
        free_kick_idx = None
        match_data = match_events[match_events['Match ID'] == match_id]
        
        for idx, event in match_data.iterrows():
            if (event['timeMin'] == start_time_min and 
                event['timeSec'] == start_time_sec and
                event['playerName'] == free_kick['playerName'] and
                event.get('Free kick taken', '') == 'Sí­'):
                free_kick_idx = idx
                break
        
        if free_kick_idx is None:
            return {
                'result_type': 'Sin resultado',
                'result_x': float(free_kick.get('Pass End X', 0)),
                'result_y': float(free_kick.get('Pass End Y', 0)),
                'goal_player': None,
                'goal_player_id': None
            }

        events_found = {
            'Goal': None, 'Post': None, 'Attempt Saved': None,
            'Miss': None, 'Card': None, 'Pass': None
        }
        
        # Buscar eventos posteriores en un rango de tiempo limitado
        free_kick_timestamp = pd.to_datetime(free_kick['timeStamp'])
        
        for next_idx in range(free_kick_idx + 1, min(free_kick_idx + 10, len(match_events))):
            next_event = match_events.iloc[next_idx]
            
            # CondiciÃ³n de parada: cambio de perÃ­odo
            if next_event['periodId'] != period_id:
                break
                
            # CondiciÃ³n de parada: tiempo excesivo
            next_timestamp = pd.to_datetime(next_event['timeStamp'])
            time_diff = (next_timestamp - free_kick_timestamp).total_seconds()
            if time_diff > 10:  # MÃ¡ximo 10 segundos
                break
            
            event_name = next_event['Event Name']
            event_team_id = next_event['Team ID']
            
            # Buscar eventos de finalizaciÃ³n del mismo equipo
            if (event_name in ['Goal', 'Post', 'Attempt Saved', 'Miss'] and 
                event_team_id == team_id and 
                events_found[event_name] is None):
                events_found[event_name] = next_event
                if event_name == 'Goal':
                    break

        # Determinar el resultado final
        result_type = 'Sin resultado'
        result_x = float(free_kick.get('Pass End X', 0))
        result_y = float(free_kick.get('Pass End Y', 0))
        goal_player = None
        goal_player_id = None

        if events_found['Goal'] is not None:
            event = events_found['Goal']
            result_type = 'Gol'
            result_x, result_y = float(event.get('x', 0)), float(event.get('y', 0))
            goal_player = event.get('playerName', '')
            goal_player_id = event.get('playerId')
        elif events_found['Post'] is not None:
            event = events_found['Post']
            result_type = 'Poste'
            result_x, result_y = float(event.get('x', 0)), float(event.get('y', 0))
        elif events_found['Attempt Saved'] is not None:
            event = events_found['Attempt Saved']
            result_type = 'Tiro a puerta'
            result_x, result_y = float(event.get('x', 0)), float(event.get('y', 0))
        elif events_found['Miss'] is not None:
            event = events_found['Miss']
            result_type = 'Tiro fuera'
            result_x, result_y = float(event.get('x', 0)), float(event.get('y', 0))

        return {
            'result_type': result_type,
            'result_x': result_x,
            'result_y': result_y,
            'goal_player': goal_player,
            'goal_player_id': goal_player_id
        }

    def get_outcome_marker(self, result_type):
        """Retorna el marcador segÃºn el tipo de resultado"""
        marker_map = {
            'Sin resultado': 'v',
            'Tiro a puerta': 's',
            'Tiro fuera': 'X',
            'Poste': '^',
            'Gol': '*'
        }
        return marker_map.get(result_type, 'o')

    def get_player_shirt_number(self, player_id):
        """Obtiene el nÃºmero de camiseta del jugador"""
        if pd.isna(player_id):
            return None
        
        player_info = self.player_stats[self.player_stats['Player ID'] == player_id]
        if not player_info.empty:
            shirt_number = player_info['Shirt Number'].iloc[0]
            return str(int(shirt_number)) if pd.notna(shirt_number) else None
        return None

    def load_team_logo(self, equipo, target_size=(80, 80)):
        """Carga y redimensiona el logo del equipo a un tamaÃ±o fijo"""
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            pass
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
                # Abrir y redimensionar con PIL
                with Image.open(logo_path) as img:
                    # Convertir a RGBA si no lo estÃ¡
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    
                    # Redimensionar manteniendo aspecto
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    
                    # Crear imagen de tamaÃ±o fijo con fondo transparente
                    final_img = Image.new('RGBA', target_size, (0, 0, 0, 0))
                    
                    # Centrar la imagen redimensionada
                    paste_x = (target_size[0] - img.width) // 2
                    paste_y = (target_size[1] - img.height) // 2
                    final_img.paste(img, (paste_x, paste_y), img)
                    
                    return np.array(final_img) / 255.0
            except Exception as e:
                pass
                return self._load_team_logo_original(equipo)
        
        return None

    def _load_team_logo_original(self, equipo):
        """MÃ©todo original como fallback"""
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

    def load_villarreal_logo(self): return self.load_team_logo('Villarreal CF')
    def load_ball_image(self): return plt.imread("assets/balon.png") if os.path.exists("assets/balon.png") else None
    def load_background(self): return plt.imread("assets/fondo_informes.png") if os.path.exists("assets/fondo_informes.png") else None

    def prepare_free_kicks_data(self, team_filter=None):
        """Prepara datos de faltas para visualizaciÃ³n"""
        if self.free_kicks_data.empty:
            pass
            return {}

        # Estructura para organizar los datos por zona y posiciÃ³n
        zone_data = {}
        for zone_num in self.zones.keys():
            zone_data[zone_num] = {'home': [], 'away': []}

        if not team_filter:
            return zone_data

        team_matches = self.team_stats[self.team_stats['Team Name'] == team_filter]
        match_positions = dict(zip(team_matches['Match ID'], team_matches['Team Position']))

        # Procesar faltas del equipo filtrado
        team_free_kicks = self.free_kicks_data[self.free_kicks_data['Team Name'] == team_filter]
        
        for _, free_kick in team_free_kicks.iterrows():
            if pd.isna(free_kick['zone']) or free_kick['zone'] not in self.zones:
                continue
                
            team_pos = match_positions.get(free_kick['Match ID'])
            if team_pos is None:
                continue
                
            zone = int(free_kick['zone'])
            
            point_data = {
                'x': free_kick['x'],
                'y': free_kick['y'],
                'pass_end_x': float(free_kick.get('Pass End X', 0)),  # â† AÃ‘ADIR
                'pass_end_y': float(free_kick.get('Pass End Y', 0)),  # â† AÃ‘ADIR
                'result_x': free_kick.get('result_x', free_kick['x']),
                'result_y': free_kick.get('result_y', free_kick['y']),
                'result_type': free_kick.get('result_type', 'Sin resultado'),
                'player_name': free_kick.get('player_name', ''),
                'goal_player': free_kick.get('goal_player', ''),
                'goal_player_id': free_kick.get('goal_player_id', ''),
                'Team Name': free_kick.get('Team Name', ''),
            }
            
            zone_data[zone][team_pos].append(point_data)


        for zone, data in zone_data.items():
            home_count, away_count = len(data['home']), len(data['away'])
            if home_count + away_count > 0:
                pass

        return zone_data
    
    def get_caidas_por_zona(self, zona_lanzamiento):
        """Cuenta caÃ­das por zona desde una zona de lanzamiento especÃ­fica"""
        zone_data = self.prepare_free_kicks_data(self.team_filter)
        data = zone_data.get(zona_lanzamiento, {'home': [], 'away': []})
        
        # Definir todas las zonas de caÃ­da con sus coordenadas
        zonas_caida = {
            'zona_1': {'count': 0, 'coords': [(83, 78.9), (100, 100)]},
            'zona_2': {'count': 0, 'coords': [(83, 54.4), (100, 78.9)]},
            'zona_3': {'count': 0, 'coords': [(83, 50), (100, 54.4)]},
            'zona_4': {'count': 0, 'coords': [(83, 45.6), (100, 50)]},
            'zona_5': {'count': 0, 'coords': [(83, 21.1), (100, 45.6)]},
            'zona_6': {'count': 0, 'coords': [(83, 0), (100, 21.1)]},
            'zona_7': {'count': 0, 'coords': [(65, 78.9), (83, 100)]},
            'zona_8': {'count': 0, 'coords': [(65, 63.2), (83, 78.9)]},
            'zona_9': {'count': 0, 'coords': [(65, 36.8), (83, 63.2)]},
            'zona_10': {'count': 0, 'coords': [(65, 21.1), (83, 36.8)]},
            'zona_11': {'count': 0, 'coords': [(65, 0), (83, 21.1)]},
            'zona_12': {'count': 0, 'coords': [(50, 78.9), (65, 100)]},
            'zona_13': {'count': 0, 'coords': [(50, 21.1), (65, 78.9)]},
            'zona_14': {'count': 0, 'coords': [(50, 0), (65, 21.1)]}
        }
        
        # --- CORRECCIÃ“N ---
        # Contar caÃ­das en cada zona usando result_x y result_y para el resultado real.
        for team_pos in ['home', 'away']:
            for point in data[team_pos]:
                # Se utiliza result_x/y que contiene el final de la jugada (tiro, gol, etc.)
                end_x = point.get('result_x', 0) # <-- CORREGIDO
                end_y = point.get('result_y', 0) # <-- CORREGIDO
                
                if end_x != 0 and end_y != 0:
                    for zona_name, zona_info in zonas_caida.items():
                        coords = zona_info['coords']
                        min_x = min(coord[0] for coord in coords)
                        max_x = max(coord[0] for coord in coords)
                        min_y = min(coord[1] for coord in coords)
                        max_y = max(coord[1] for coord in coords)
                        
                        if min_x <= end_x <= max_x and min_y <= end_y <= max_y:
                            zonas_caida[zona_name]['count'] += 1
                            break
        
        return zonas_caida
    
    def get_top_lanzadores_por_zona(self, zona_lanzamiento, team_filter=None):
        """Obtiene el top 3 de jugadores con mÃ¡s lanzamientos en una zona especÃ­fica"""
        
        # Filtrar faltas de la zona de lanzamiento especÃ­fica
        if self.free_kicks_data.empty:
            return []
        
        # Filtrar por equipo si se especifica
        if team_filter:
            faltas_zona = self.free_kicks_data[
                (self.free_kicks_data['zone'] == zona_lanzamiento) &
                (self.free_kicks_data['Team Name'] == team_filter)
            ]
        else:
            faltas_zona = self.free_kicks_data[self.free_kicks_data['zone'] == zona_lanzamiento]
        
        if faltas_zona.empty:
            return []
        
        # Contar lanzamientos por jugador - USAR 'player_id' en lugar de 'playerId'
        ranking = faltas_zona.groupby(['player_id', 'player_name']).size().reset_index(name='lanzamientos')
        
        # Hacer merge con player_stats para obtener dorsales
        player_dorsales = self.player_stats[['Player ID', 'Shirt Number']].copy()
        player_dorsales = player_dorsales.rename(columns={
            'Player ID': 'player_id',  # CAMBIAR: de 'playerId' a 'player_id'
            'Shirt Number': 'dorsal'
        })
        
        # Merge para obtener dorsales - USAR 'player_id'
        ranking = ranking.merge(player_dorsales, on='player_id', how='left')
        
        # Ordenar por nÃºmero de lanzamientos y tomar top 3
        ranking = ranking.sort_values('lanzamientos', ascending=False).head(3)
        
        # Formatear resultado
        top_jugadores = []
        for _, jugador in ranking.iterrows():
            dorsal = int(jugador['dorsal']) if pd.notna(jugador['dorsal']) else '?'
            nombre = jugador['player_name'] if pd.notna(jugador['player_name']) else 'Desconocido'
            lanzamientos = jugador['lanzamientos']
            
            top_jugadores.append({
                'dorsal': dorsal,
                'nombre': nombre,
                'lanzamientos': lanzamientos
            })
        
        return top_jugadores

    
    def create_mapa_calor_zona_lanzamiento(self, ax, zona_target, titulo):
        """Crea mapa de calor para una zona de lanzamiento especÃ­fica con ranking de jugadores"""
        pitch = VerticalPitch(half=True, pitch_type='opta', pitch_color='#2d5a27', 
                            line_color='white', linewidth=2)
        pitch.draw(ax=ax)
        ax.set_aspect('auto')
        ax.set_title(titulo, fontsize=12, fontweight='bold', color='#1e3d59', pad=10)
        
        # Obtener datos de caÃ­das para esta zona de lanzamiento
        zonas_caida = self.get_caidas_por_zona(zona_target)
        
        # Obtener el mÃ¡ximo de caÃ­das para normalizar colores
        max_caidas = max(zona_info['count'] for zona_info in zonas_caida.values()) if zonas_caida else 1
        
        import matplotlib.cm as cm
        colormap = cm.get_cmap('Reds')
        
        # Definir coordenadas de todas las zonas (intercambiar x,y para matplotlib)
        zona_coords_matplotlib = {
            'zona_1': [(78.9, 83), (100, 100)],
            'zona_2': [(54.4, 83), (78.9, 100)],
            'zona_3': [(50, 83), (54.4, 100)],
            'zona_4': [(45.6, 83), (50, 100)],
            'zona_5': [(21.1, 83), (45.6, 100)],
            'zona_6': [(0, 83), (21.1, 100)],
            'zona_7': [(78.9, 65), (100, 83)],
            'zona_8': [(63.2, 65), (78.9, 83)],
            'zona_9': [(36.8, 65), (63.2, 83)],
            'zona_10': [(21.1, 65), (36.8, 83)],
            'zona_11': [(0, 65), (21.1, 83)],
            'zona_12': [(78.9, 50), (100, 65)],
            'zona_13': [(21.1, 50), (78.9, 65)], 
            'zona_14': [(0, 50), (21.1, 65)]
        }
        
        # Orden de dibujo para evitar superposiciones
        orden_dibujo = ['zona_1', 'zona_2', 'zona_3', 'zona_4', 'zona_5', 'zona_6', 
                        'zona_7', 'zona_8', 'zona_9', 'zona_10', 'zona_11', 'zona_12', 'zona_13', 'zona_14']
        
        zona_lanzamiento_center = None
        
        for zona in orden_dibujo:
            if zona in zona_coords_matplotlib:
                coords = zona_coords_matplotlib[zona]
                (y_min, x_min), (y_max, x_max) = coords
                width, height = y_max - y_min, x_max - x_min
                
                count_caidas = zonas_caida.get(zona, {}).get('count', 0)
                
                # Color y transparencia
                if zona == f'zona_{zona_target}':
                    # Zona de lanzamiento en negro
                    color_zona = 'black'
                    alpha_zona = 0.8
                    edge_color = 'red'
                    edge_width = 4
                    # Guardar centro para el ranking
                    zona_lanzamiento_center = (y_min + width/2, x_min + height/2)
                else:
                    # Otras zonas con intensidad segÃºn caÃ­das
                    if max_caidas > 0:
                        intensidad = 0.1 + (count_caidas / max_caidas) * 0.9
                    else:
                        intensidad = 0.1
                    color_zona = colormap(intensidad)
                    alpha_zona = 0.7
                    edge_color = 'black'
                    edge_width = 1
                
                # Dibujar rectÃ¡ngulo de zona
                rect = patches.Rectangle((y_min, x_min), width, height, 
                                    linewidth=edge_width, edgecolor=edge_color,  
                                    facecolor=color_zona, alpha=alpha_zona)
                ax.add_patch(rect)
                
                # Texto con nÃºmero de caÃ­das
                center_y, center_x = y_min + width/2, x_min + height/2
                
                if zona == f'zona_{zona_target}':
                    # En zona de lanzamiento, mostrar balÃ³n
                    if (ball := self.load_ball_image()) is not None:
                        ball_box = OffsetImage(ball, zoom=0.05)
                        ab_ball = AnnotationBbox(ball_box, (center_y, center_x), 
                                            frameon=False, zorder=15)
                        ax.add_artist(ab_ball)
                else:
                    text_color = 'white' if (count_caidas / max_caidas if max_caidas > 0 else 0) > 0.6 else 'black'
                    
                    # Solo mostrar nÃºmero si hay caÃ­das
                    if count_caidas > 0:
                        ax.text(center_y, center_x, str(count_caidas), 
                            fontsize=16, fontweight='bold', ha='center', va='center',
                            color=text_color, zorder=10)


    def create_free_kicks_visualization(self, figsize=(11.69, 8.27), team_filter=None):
        """Crear visualizaciÃ³n de faltas ofensivas frontales con mapas de calor y ranking"""
        plt.rcParams.update({
            'figure.autolayout': False,
            'figure.constrained_layout.use': False,
            'figure.subplot.left': 0,
            'figure.subplot.right': 1, 
            'figure.subplot.top': 1,
            'figure.subplot.bottom': 0,
            'figure.subplot.wspace': 0,
            'figure.subplot.hspace': 0,
            'axes.xmargin': 0,
            'axes.ymargin': 0,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0
        })

        fig = plt.figure(figsize=figsize, facecolor='white')
        
        # Fondo y tÃ­tulo
        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25)
            ax_bg.axis('off')
        
        fig.suptitle('ANÁLISIS DE FALTAS OFENSIVAS LEJANAS', 
                    fontsize=20, fontweight='bold', color='#1e3d59', y=0.98, family='serif')
        
        # Logos
        if (ball := self.load_ball_image()) is not None:
            ax_ball = fig.add_axes([0.05, 0.92, 0.09, 0.09])
            ax_ball.imshow(ball)
            ax_ball.axis('off')
        
        if team_filter and (team_logo := self.load_team_logo(team_filter)) is not None:
            ax_team = fig.add_axes([0.88, 0.90, 0.08, 0.09])
            ax_team.imshow(team_logo, aspect='auto')
            ax_team.axis('off')
            
        if (villarreal_logo := self.load_villarreal_logo()) is not None:
            ax_villarreal = fig.add_axes([0.85, 0.90, 0.08, 0.09])
            ax_villarreal.imshow(villarreal_logo, aspect='auto')
            ax_villarreal.axis('off')

        # CREAR GRID CON 4 COLUMNAS
        gs = GridSpec(2, 4, figure=fig, 
            left=0.05, right=0.95,
            bottom=0.14, top=0.90,
            wspace=0.1, hspace=0.15)

        # FILA SUPERIOR: Zona 9 (columnas 2-3) + Ranking (columna 4)
        ax_zona13 = fig.add_subplot(gs[0, 1:3])   # Columnas centrales (1 y 2)
        ax_ranking = fig.add_subplot(gs[0, 3])   # Ãšltima columna

        # FILA INFERIOR: Zona 8 (columnas 1-2) y Zona 10 (columnas 3-4)
        ax_zona12 = fig.add_subplot(gs[1, 0:2])   # Primeras 2 columnas
        ax_zona14 = fig.add_subplot(gs[1, 2:4])  # Ãšltimas 2 columnas

        # ConfiguraciÃ³n de campos
        campos_config = [
            (ax_zona13, 13, 'LEJANO CENTRO'),
            (ax_zona12, 12, 'LEJANO IZQUIERDA'), 
            (ax_zona14, 14, 'LEJANO DERECHA')
        ]

        # Crear mapa de calor para cada zona de lanzamiento
        for ax, zona_target, titulo in campos_config:
            self.create_mapa_calor_zona_lanzamiento(ax, zona_target, titulo)

        # CREAR RANKING DE LANZADORES
        self.create_faltas_ranking(ax_ranking, team_filter)

        # Crear leyenda simplificada
        from matplotlib.lines import Line2D
        legend_ax = fig.add_axes([0.05, 0.02, 0.90, 0.08])
        legend_ax.axis('off')
        
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=8, 
                markeredgecolor='red', linewidth=2, label='Zona lanzamiento'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=8, 
                markeredgecolor='black', alpha=0.7, label='Zona caída (intensidad)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=6,
                markeredgecolor='black', label='Balón (posición lanzamiento)'),
        ]
        
        legend = legend_ax.legend(handles=legend_elements, loc='center', frameon=True, 
                                fancybox=True, shadow=True, ncol=3, fontsize=10, 
                                columnspacing=2.0, handletextpad=0.8)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(0.95)
        frame.set_edgecolor('#1e3d59')
        frame.set_linewidth(1.5)
        
        return fig

    def print_summary(self, team_filter=None):
        """Imprime resumen de faltas"""
        if self.free_kicks_data.empty:
            pass
            return
        
        
        if team_filter:
            team_free_kicks = self.free_kicks_data[self.free_kicks_data['Team Name'] == team_filter]
            rival_free_kicks = self.free_kicks_data[
                (self.free_kicks_data['Match ID'].isin(
                    self.team_stats[self.team_stats['Team Name'] == team_filter]['Match ID'].unique()
                )) & (self.free_kicks_data['Team Name'] != team_filter)
            ]
            
            if not team_free_kicks.empty and 'result_type' in team_free_kicks.columns:
                pass
        
        if 'zone' in self.free_kicks_data.columns:
            pass

def seleccionar_equipo_interactivo():
    """SelecciÃ³n interactiva de equipo"""
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
    """FunciÃ³n principal"""
    try:
        pass
        if (equipo := seleccionar_equipo_interactivo()) is None:
            pass
            return
        
        analyzer = FreeKicksAnalysis(team_filter=equipo)
        analyzer.print_summary(team_filter=equipo)
        
        if (fig := analyzer.create_free_kicks_visualization(team_filter=equipo)):
            plt.show()
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_faltas_lejanas_{equipo_filename}.pdf"
            analyzer.guardar_sin_espacios(fig, output_path)
        else:
            pass
            
    except Exception as e:
        pass
        import traceback
        traceback.print_exc()

def generar_reporte_personalizado(equipo, mostrar=True, guardar=True):
    """Generar reporte personalizado"""
    try:
        analyzer = FreeKicksAnalysis(team_filter=equipo)
        analyzer.print_summary(team_filter=equipo)
        fig = analyzer.create_free_kicks_visualization(team_filter=equipo)
        
        if fig:
            if mostrar:
                plt.show()
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"reporte_faltas_lejanas_{equipo_filename}.pdf"
                analyzer.guardar_sin_espacios(fig, output_path)
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
    """Verificar assets necesarios"""
    os.makedirs('assets/escudos', exist_ok=True)
    files_to_check = [
        'extraccion_opta/datos_opta_parquet/abp_events.parquet',
        'extraccion_opta/datos_opta_parquet/team_stats.parquet',
        'extraccion_opta/datos_opta_parquet/player_stats.parquet',
        'assets/fondo_informes.png',
        'assets/balon.png'
    ]
    for file_path in files_to_check:
        if os.path.exists(file_path):
            pass
        else:
            pass
    
    if os.path.exists('assets/escudos'):
        escudos = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
        if escudos:
            pass
        else:
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