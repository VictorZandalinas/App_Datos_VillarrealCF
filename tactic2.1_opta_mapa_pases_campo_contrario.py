import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import patheffects
import numpy as np
import os
from mplsoccer import VerticalPitch, Pitch
from difflib import SequenceMatcher
from collections import defaultdict, Counter
import warnings
import json
import base64
import re
import unicodedata  
from io import BytesIO
from PIL import Image
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.cluster import DBSCAN

warnings.filterwarnings('ignore')

class PasesCampoContrario:
    # 🔥 CACHÉ DE DATOS: Compartido entre todas las instancias para evitar cargas repetidas
    _open_play_cache = None
    _team_stats_cache = None
    _player_stats_cache = None
    _match_events_cache = None  # 🔥 NUEVO CACHÉ

    @classmethod
    def _get_open_play_data(cls, columns=None):
        """Carga open_play_events.parquet una sola vez y lo cachea."""
        if cls._open_play_cache is None:
            print("📥 [CACHÉ] Cargando open_play_events.parquet por primera vez...")
            cls._open_play_cache = pd.read_parquet("extraccion_opta/datos_opta_parquet/open_play_events.parquet")
        if columns:
            return cls._open_play_cache[columns].copy()
        return cls._open_play_cache.copy()

    @classmethod
    def _get_match_events_data(cls):
        """Carga match_events.parquet una sola vez y lo cachea."""
        if cls._match_events_cache is None:
            print("📥 [CACHÉ] Cargando match_events.parquet por primera vez...")
            cls._match_events_cache = pd.read_parquet("extraccion_opta/datos_opta_parquet/match_events.parquet")
        return cls._match_events_cache.copy()

    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/open_play_events.parquet", team_filter=None):
        self.data_path = data_path
        self.team_filter = team_filter
        self.df = None
        self.passes_data = pd.DataFrame()

        # Usar caché para team_stats y player_stats
        if PasesCampoContrario._team_stats_cache is None:
            PasesCampoContrario._team_stats_cache = pd.read_parquet("extraccion_opta/datos_opta_parquet/team_stats.parquet")
        if PasesCampoContrario._player_stats_cache is None:
            PasesCampoContrario._player_stats_cache = pd.read_parquet("extraccion_opta/datos_opta_parquet/player_stats.parquet")

        self.team_stats = PasesCampoContrario._team_stats_cache
        self.player_stats = PasesCampoContrario._player_stats_cache
        self.player_stats_df = PasesCampoContrario._player_stats_cache

        self.load_data(team_filter)
        self.events_df = None
        self.load_match_events()
        
        self.photos_data = None
        
        self.formation_mapping = {
            1: "not_in_use", 2: "442", 3: "41212", 4: "433", 5: "451",
            6: "4411", 7: "4141", 8: "4231", 9: "4321", 10: "532",
            11: "541", 12: "352", 13: "343", 15: "4222", 16: "3511",
            17: "3421", 18: "3412", 19: "3142", 20: "343d", 21: "4132",
            22: "4240", 23: "4312", 24: "3241", 25: "3331"
        }

        self.formation_demarcations = {
            2: {1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI', 4: 'MCD', 8: 'MCI', 7: 'ED', 11: 'EI', 10: 'DC', 9: 'SD'},
            3: {1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI', 4: 'MCD', 7: 'ED', 11: 'EI', 8: 'MP', 10: 'SD', 9: 'DC'},
            4: {1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI', 4: 'MCD', 7: 'ED', 8: 'EI', 10: 'EI', 9: 'DC', 11: 'ED'},
            5: {1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI', 4: 'MC', 10: 'MP', 8: 'MCI', 7: 'ED', 11: 'EI', 9: 'DC'},
            6: {1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI', 4: 'MCD', 8: 'MCI', 7: 'ED', 11: 'EI', 10: 'MP', 9: 'DC'},
            7: {1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI', 4: 'MCD', 7: 'ED', 8: 'MCD', 10: 'MCI', 11: 'EI', 9: 'DC'},
            8: {1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI', 4: 'MCD', 8: 'MCI', 7: 'ED', 10: 'MP', 11: 'EI', 9: 'DC'},
            9: {1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI', 4: 'MCD', 7: 'MC', 8: 'MCI', 10: 'SD', 11: 'SD', 9: 'DC'},
            10: {1: 'POR', 2: 'LD', 4: 'CC', 6: 'CD', 5: 'CI', 3: 'LI', 7: 'MCD', 8: 'MCI', 10: 'MP', 11: 'DC', 9: 'SD'},
            11: {1: 'POR', 2: 'LD', 5: 'CC', 6: 'CD', 4: 'CI', 3: 'LI', 7: 'ED', 8: 'MCD', 10: 'MCI', 11: 'EI', 9: 'DC'},
            12: {1: 'POR', 2: 'LD', 5: 'CC', 6: 'CD', 4: 'CI', 3: 'LI', 7: 'MCD', 11: 'MP', 8: 'MCI', 10: 'DC', 9: 'SD'},
            13: {1: 'POR', 2: 'LD', 6: 'CC', 5: 'CD', 4: 'CI', 3: 'LI', 7: 'MCD', 8: 'MCI', 10: 'EI', 9: 'DC', 11: 'ED'},
            15: {1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI', 4: 'MCD', 8: 'MCI', 7: 'ED', 11: 'EI', 10: 'SD', 9: 'DC'},
            16: {1: 'POR', 2: 'LD', 5: 'CC', 6: 'CD', 4: 'CI', 3: 'LI', 7: 'MCD', 11: 'MC', 8: 'MCI', 10: 'MP', 9: 'DC'},
            17: {1: 'POR', 2: 'LD', 5: 'CC', 6: 'CD', 4: 'CI', 3: 'LI', 7: 'MCD', 8: 'MCI', 10: 'SD', 11: 'SD', 9: 'DC'},
            18: {1: 'POR', 2: 'LD', 5: 'CC', 6: 'CD', 4: 'CI', 3: 'LI', 7: 'MCD', 8: 'MCI', 9: 'MP', 10: 'DC', 11: 'SD'},
            19: {1: 'POR', 2: 'LD', 5: 'CC', 4: 'CD', 6: 'CI', 3: 'LI', 8: 'MCD', 7: 'MCD', 11: 'MCI', 9: 'DC', 10: 'SD'},
            20: {1: 'POR', 2: 'LD', 5: 'CC', 6: 'CD', 4: 'CI', 3: 'LI', 8: 'MCD', 7: 'MP', 10: 'EI', 11: 'ED', 9: 'DC'},
            21: {1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI', 4: 'MCD', 7: 'MCD', 10: 'MP', 8: 'MCI', 9: 'DC', 11: 'SD'},
            22: {1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI', 4: 'MCD', 8: 'MCI', 7: 'ED', 11: 'EI', 9: 'DC', 10: 'SD'},
            23: {1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI', 7: 'MCD', 4: 'MC', 8: 'MCI', 9: 'MP', 10: 'DC', 11: 'SD'},
            24: {1: 'POR', 2: 'CD', 3: 'CC', 4: 'CI', 5: 'LD', 6: 'LI', 10: 'ED', 11: 'EI', 7: 'MCD', 8: 'MCI', 9: 'DC'},
            25: {1: 'POR', 2: 'CD', 3: 'CC', 4: 'CI', 5: 'LD', 6: 'LI', 7: 'MCD', 8: 'MCD', 11: 'MCI', 10: 'MP', 9: 'DC'}
        }

        self.demarcation_labels = {
            'POR': 'Portero', 'LD': 'Lateral/Carrilero Der', 'LI': 'Lateral/Carrilero Izq',
            'CD': 'Central Derecho', 'CI': 'Central Izquierdo', 'CC': 'Central Centro',
            'MCD': 'Mediocentro Defensivo', 'MCI': 'Mediocentro Izquierdo',
            'MC': 'Mediocampista Centro', 'MP': 'Mediapunta',
            'EI': 'Extremo Izquierdo', 'ED': 'Extremo Derecho',
            'DC': 'Delantero Centro', 'SD': 'Segundo Delantero'
        }
        if team_filter:
            self.extract_passes(team_filter)
    
    def _draw_player_card(self, ax, x_pos, y_pos, player_data, panel_color='#3498db'):
        """Dibuja una tarjeta de jugador con foto, nombre y dorsal (estilo tactic3)."""
        
        # Cargar fotos si no están cargadas
        if self.photos_data is None:
            self.load_player_photos()

        # 📛 NOMBRE
        player_name = player_data.get('name', 'N/A')
        apellido = player_name.split()[-1].upper() if player_name != 'N/A' else 'N/A'
        ax.text(x_pos, y_pos - 0.10, apellido, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=15,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a252f', 
                          alpha=0.9, edgecolor='white', linewidth=1.5))

        # 📸 FOTO
        player_photo = self.get_player_photo(player_name)
        if player_photo is not None:
            img = OffsetImage(player_photo, zoom=0.25)
            ab = AnnotationBbox(img, (x_pos, y_pos), frameon=False, zorder=13)
            ax.add_artist(ab)
        else:
            # Círculo placeholder si no hay foto
            circle_placeholder = patches.Circle((x_pos, y_pos), 0.15, 
                                            color='grey', alpha=0.5, zorder=12)
            ax.add_patch(circle_placeholder)

        # 🔢 DORSAL
        dorsal = player_data.get('shirt', '?')
        circle_bg = patches.Circle((x_pos + 0.10, y_pos + 0.06), 0.03, 
                                color=panel_color, alpha=0.95, zorder=14)
        ax.add_patch(circle_bg)
        
        circle_border = patches.Circle((x_pos + 0.10, y_pos + 0.06), 0.05, fill=False,
                                    ec='white', linewidth=1.5, zorder=15)
        ax.add_patch(circle_border)
        
        ax.text(x_pos + 0.10, y_pos + 0.06, str(dorsal), ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=16) 

    
    # REEMPLAZA ESTA FUNCIÓN COMPLETA
    def draw_top_assisters_panel(self, ax, ranking_data):
        """Dibuja los 2 mejores pasadores clave (asistencias y pases verticales)."""
        ax.set_facecolor('#f0f0f0')
        ax.axis('off')

        ax.text(0.5, 0.95, 'TOP PASADORES CLAVE', ha='center', va='top', fontsize=9,
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#2c3e50', alpha=0.9))
        
        # 🔥 SUBTÍTULO AÑADIDO
        ax.text(0.5, 0.88, 'Asistencias (A) y Pases Verticales (V)', ha='center', va='top',
                fontsize=7, color='grey', style='italic')

        if not ranking_data:
            ax.text(0.5, 0.5, 'Sin Datos', ha='center', va='center', color='grey')
            return

        x_positions = [0.3, 0.7]
        y_pos = 0.50 # Bajamos un poco la posición general para dar espacio al subtítulo

        for i, (player_id, stats) in enumerate(ranking_data[:2]):
            self._draw_player_card(ax, x_positions[i], y_pos, stats, panel_color='#3498db')
            
            assists = stats.get('assists', 0)
            vertical = stats.get('vertical_passes', 0)
            
            x_center = x_positions[i]
            y_badge = y_pos - 0.15 # 🔥 BAJAMOS LOS BADGES JUNTO CON EL NOMBRE

            # Badge de Asistencias
            ax.text(x_center - 0.08, y_badge, f'A: {assists}', ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#e74c3c',
                              alpha=0.95, edgecolor='white', linewidth=1))
            
            # Badge de Pases Verticales
            ax.text(x_center + 0.08, y_badge, f'V: {vertical}', ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#27ae60',
                              alpha=0.95, edgecolor='white', linewidth=1))

    def draw_cross_finisher_panel(self, ax, sequence_data):
        """Dibuja las 2 conexiones más repetidas de Centro -> Remate."""
        ax.set_facecolor('#f0f0f0')
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'CONEXIÓN: CENTRO → REMATE', ha='center', va='top', fontsize=9,
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#2c3e50', alpha=0.9))
        
        if not sequence_data:
            ax.text(0.5, 0.5, 'Sin Datos', ha='center', va='center', color='grey')
            return
            
        y_positions = [0.65, 0.25]
        for i, seq in enumerate(sequence_data[:2]):
            y = y_positions[i]
            
            # Centrador
            self._draw_player_card(ax, 0.25, y, seq['crosser'], panel_color='#3498db')
            
            # Rematador
            self._draw_player_card(ax, 0.75, y, seq['finisher'], panel_color='#9b59b6')
            
            # Flecha y contador
            arrow = patches.FancyArrowPatch((0.4, y), (0.6, y),
                                            arrowstyle='->,head_width=0.2,head_length=0.15',
                                            color='#e67e22', linewidth=2.5)
            ax.add_patch(arrow)
            
            count = seq.get('count', 0)
            ax.text(0.5, y + 0.08, f'{count}x', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='#e67e22')

    def draw_sequences_to_box_panel(self, ax, sequence_data):
        """Dibuja las 3 secuencias más comunes a área con tarjetas de jugador."""
        ax.set_facecolor('#f0f0f0')
        ax.axis('off')
        
        # 🔥 TÍTULO ACTUALIZADO para reflejar el nuevo rango de pases
        ax.text(0.5, 0.97, 'CONEXIONES CLAVE AL ÁREA', ha='center', va='top', fontsize=9,
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#2c3e50', alpha=0.9))

        if not sequence_data:
            ax.text(0.5, 0.5, 'Sin Patrones de Secuencia', ha='center', va='center', color='grey')
            return

        sequence_keys = [tuple(p['player_id'] for p in seq) for seq in sequence_data]
        counts = Counter(sequence_keys)
        filtered_sequences = [(seq, count) for seq, count in counts.items() if count >= 2]
        most_common = sorted(filtered_sequences, key=lambda x: x[1], reverse=True)[:3]

        if not most_common:
            ax.text(0.5, 0.5, 'Sin Patrones Repetidos (≥2x)', ha='center', va='center', fontsize=8, color='grey')
            return

        y_positions = [0.82, 0.50, 0.18]
        colors = ['#f39c12', '#1abc9c', '#e74c3c']

        for i, (seq_key, count) in enumerate(most_common):
            if i >= len(y_positions): break
            y_pos = y_positions[i]
            
            full_sequence = next(s for s in sequence_data if tuple(p['player_id'] for p in s) == seq_key)
            
            num_players = len(full_sequence)
            
            # --- 🔥 AÑADIDO: Lógica para dibujar secuencias de 2 jugadores (1 pase) ---
            if num_players == 2: # 1 pase
                x_positions = [0.35, 0.65] # Posiciones para 2 jugadores
            elif num_players == 3: # 2 pases
                x_positions = [0.25, 0.5, 0.75]
            elif num_players == 4: # 3 pases
                x_positions = [0.18, 0.40, 0.62, 0.84]
            else:
                continue
            # --- FIN DE LA MODIFICACIÓN ---

            ax.text(0.02, y_pos, f'{count}x', ha='left', va='center', fontsize=10, fontweight='bold',
                    color=colors[i], bbox=dict(boxstyle='circle,pad=0.25', facecolor='white',
                                               edgecolor=colors[i], linewidth=2))

            for j, player in enumerate(full_sequence):
                player_card_data = {'name': player['player_name'], 'shirt': player['shirt_number']}
                self._draw_player_card(ax, x_positions[j], y_pos, player_card_data, panel_color=colors[i])
                
                if j < num_players - 1:
                    arrow_start_x = x_positions[j] + 0.08
                    arrow_end_x = x_positions[j+1] - 0.08
                    arrow = patches.FancyArrowPatch(
                        (arrow_start_x, y_pos), (arrow_end_x, y_pos),
                        arrowstyle='->,head_width=0.15,head_length=0.1',
                        color='#34495e', linewidth=2, zorder=5)
                    ax.add_patch(arrow)
    
    def draw_gradient_arrow(self, ax, start_pos, end_pos, color='yellow',
                            min_width=1, max_width=5, min_alpha=0.3, max_alpha=1.0,
                            n_segments=40, curve_rad=0):
        """
        Dibuja una flecha 100% continua y suave con grosor y transparencia gradual.
        
        Utiliza ax.plot para el cuerpo, garantizando una línea perfecta sin discontinuidades,
        y añade una cabeza de flecha al final.
        """
        import numpy as np
        from matplotlib.patches import FancyArrowPatch

        # Convertir posiciones a arrays de numpy para cálculos vectoriales
        start = np.array(start_pos)
        end = np.array(end_pos)
        
        # Generar puntos intermedios para la línea base (recta)
        x_points = np.linspace(start[0], end[0], n_segments + 1)
        y_points = np.linspace(start[1], end[1], n_segments + 1)

        # Aplicar curvatura si el valor de curve_rad es distinto de cero
        if curve_rad != 0:
            # Calcular el vector director y su longitud
            vec = end - start
            path_length = np.linalg.norm(vec)
            
            if path_length > 0:
                # Calcular el vector perpendicular para la curvatura
                perp_vec = np.array([-vec[1], vec[0]]) / path_length
                
                # Magnitud del desplazamiento en el punto medio de la curva
                offset_magnitude = path_length * curve_rad * -1.5

                # Ajustar cada punto intermedio para formar una parábola suave
                for i in range(1, n_segments):
                    # El factor parabólico es 0 en los extremos y 1 en el centro
                    t = i / n_segments
                    parabolic_factor = 4 * t * (1 - t)
                    
                    # Aplicar el desplazamiento
                    x_points[i] += parabolic_factor * offset_magnitude * perp_vec[0]
                    y_points[i] += parabolic_factor * offset_magnitude * perp_vec[1]

        # Interpolar el grosor y la transparencia a lo largo de los segmentos
        widths = np.linspace(min_width, max_width, n_segments)
        alphas = np.linspace(min_alpha, max_alpha, n_segments)

        # DIBUJAR EL CUERPO DE LA FLECHA segmento por segmento usando ax.plot
        # Esto garantiza una línea perfectamente continua y suave.
        for i in range(n_segments):
            ax.plot(
                [x_points[i], x_points[i+1]],
                [y_points[i], y_points[i+1]],
                color=color,
                linewidth=widths[i],
                alpha=alphas[i],
                solid_capstyle='round', # Clave para uniones suaves e invisibles
                zorder=10
            )
        
        # AÑADIR SOLO LA CABEZA DE LA FLECHA AL FINAL
        # Se usa FancyArrowPatch sin cuerpo, solo para dibujar la punta.
        arrow_head = FancyArrowPatch(
            # Posición desde el penúltimo al último punto para darle la dirección correcta
            (x_points[-2], y_points[-2]),
            (x_points[-1], y_points[-1]),
            # Estilo '-|>' significa: no dibujar línea, solo cabeza al final
            arrowstyle=f'-|>,head_width={max_width*2},head_length={max_width*1.1}',
            color=color,
            linewidth=0, # No dibujar el cuerpo del patch
            alpha=max_alpha,
            zorder=11,
            shrinkA=0, # No encoger el inicio
            shrinkB=0  # No encoger el final
        )
        ax.add_patch(arrow_head)

    def create_crosses_map(self, ax, title="MAPA DE CENTROS AL ÁREA"):
        """
        Mapa de centros con zonas sombreadas y 2 jugadores por lado en la parte inferior.
        """
        if self.df is None:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='white')
            return
        
        from mplsoccer import VerticalPitch
        from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        import matplotlib.cm as cm
        from collections import defaultdict, Counter
        
        pitch = VerticalPitch(pitch_type='opta', line_zorder=2,
                            pitch_color='#22312b', line_color='white',
                            half=True)
        pitch.draw(ax=ax)
        
        team_data = self.df[self.df['Team Name'] == self.team_filter].copy()
        
        for col in ['x', 'y', 'Pass End X', 'Pass End Y']:
            team_data[col] = pd.to_numeric(team_data[col], errors='coerce')
        
        crosses = team_data[team_data.get('Cross') == 'Sí'].copy()
        
        if crosses.empty:
            ax.text(0.5, 0.5, 'Sin centros', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='white')
            return
        
        zones = {
            'left_far': {'x_min': 50, 'x_max': 83, 'y_min': 78.9, 'y_max': 100, 'label': 'IZQ-LEJOS'},
            'left_deep': {'x_min': 83, 'x_max': 100, 'y_min': 78.9, 'y_max': 100, 'label': 'IZQ-FONDO'},
            'right_far': {'x_min': 50, 'x_max': 83, 'y_min': 0, 'y_max': 21.1, 'label': 'DER-LEJOS'},
            'right_deep': {'x_min': 83, 'x_max': 100, 'y_min': 0, 'y_max': 21.1, 'label': 'DER-FONDO'}
        }
        
        curve_info = {
            'concava': {'label': 'C', 'name': 'Cerrado', 'color': '#ff6600'},
            'convexa': {'label': 'A', 'name': 'Abierto', 'color': '#00ff00'},
            'recta':   {'label': 'P', 'name': 'Plano',   'color': '#ffffff'}
        }
        
        # 🔥 ESTRUCTURA PARA DATOS POR JUGADOR EN CADA LADO (no por zona)
        side_player_data = defaultdict(lambda: defaultdict(lambda: {
            'total': 0, 'concava': 0, 'convexa': 0, 'recta': 0,
            'dorsal': None, 'player_id': None, 'player_name': None
        }))
        
        zone_data = defaultdict(lambda: defaultdict(list))
        
        for _, cross in crosses.iterrows():
            x, y = cross['x'], cross['y']
            end_x, end_y = cross['Pass End X'], cross['Pass End Y']
            
            if pd.notna(x) and pd.notna(y) and pd.notna(end_x) and pd.notna(end_y):
                zone_key = next((zk for zk, z_info in zones.items() 
                            if z_info['x_min'] <= x <= z_info['x_max'] 
                            and z_info['y_min'] <= y <= z_info['y_max']), None)
                if not zone_key: 
                    continue
                
                # 🔥 DETERMINAR LADO (izquierda o derecha)
                side = 'left' if 'left' in zone_key else 'right'
                
                # Obtener info del jugador
                player_name = cross.get('playerName', 'N/A')
                player_id = cross.get('playerId')
                
                # Obtener dorsal desde player_stats
                shirt_number = None
                if self.player_stats is not None and pd.notna(player_id):
                    player_info = self.player_stats[self.player_stats['Player ID'] == player_id]
                    if not player_info.empty:
                        shirt_number = int(player_info['Shirt Number'].iloc[0])
                
                # Calcular tipo de centro
                is_left_side = y > 50
                is_right_footed = cross.get('Right footed') == 'Sí'
                is_left_footed = cross.get('Left footed') == 'Sí'
                
                if (is_left_side and is_right_footed) or (not is_left_side and is_left_footed):
                    curve_type = 'concava'
                elif (is_left_side and is_left_footed) or (not is_left_side and is_right_footed):
                    curve_type = 'convexa'
                else:
                    curve_type = 'recta'
                
                curve_rad = 0
                if curve_type == 'concava':
                    curve_rad = 0.2 if is_left_side else -0.2
                elif curve_type == 'convexa':
                    curve_rad = -0.2 if is_left_side else 0.2
                
                # Guardar datos del jugador POR LADO
                side_player_data[side][player_name]['total'] += 1
                side_player_data[side][player_name][curve_type] += 1
                side_player_data[side][player_name]['dorsal'] = shirt_number
                side_player_data[side][player_name]['player_id'] = player_id
                side_player_data[side][player_name]['player_name'] = player_name
                
                # También guardar en zone_data para las flechas
                zone_data[zone_key][curve_type].append({
                    'x': x, 'y': y, 'end_x': end_x, 'end_y': end_y, 'curve_rad': curve_rad
                })
        
        # Dibujar flechas individuales (semitransparentes)
        for _, curve_types in zone_data.items():
            for _, crosses_list in curve_types.items():
                for cross_data in crosses_list:
                    arrow = FancyArrowPatch(
                        (cross_data['y'], cross_data['x']), 
                        (cross_data['end_y'], cross_data['end_x']),
                        connectionstyle=f"arc3,rad={cross_data['curve_rad']}", 
                        arrowstyle='->,head_width=0.3,head_length=0.3',
                        color='yellow', linewidth=1, alpha=0.15, zorder=2
                    )
                    ax.add_patch(arrow)
        
        # PRE-CÁLCULO DEL SCORE MÁXIMO
        max_score = 0
        scores = {}
        for zone_k, c_types in zone_data.items():
            total = sum(len(c_list) for c_list in c_types.values())
            if total > 0:
                for curve_t, c_list in c_types.items():
                    count = len(c_list)
                    percentage_ratio = count / total
                    score = count * percentage_ratio
                    scores[(zone_k, curve_t)] = score
                    if score > max_score:
                        max_score = score
        if max_score == 0: 
            max_score = 1
        
        # DIBUJAR ZONAS SOMBREADAS Y PATRONES PROMEDIO
        max_count_overall = max((sum(len(c) for c in ct.values()) for ct in zone_data.values()), default=1)
        cmap = cm.get_cmap('YlOrRd')
        
        for zone_key, zone_info in zones.items():
            count_in_zone = sum(len(crosses_list) for crosses_list in zone_data[zone_key].values())
            
            # Intensidad del sombreado
            intensity = count_in_zone / max_count_overall if count_in_zone > 0 else 0
            rect = Rectangle(
                (zone_info['y_min'], zone_info['x_min']), 
                zone_info['y_max'] - zone_info['y_min'],
                zone_info['x_max'] - zone_info['x_min'], 
                linewidth=2, edgecolor='yellow', 
                facecolor=cmap(intensity * 0.7), alpha=0.4, zorder=1
            )
            ax.add_patch(rect)
            
            center_x = (zone_info['x_min'] + zone_info['x_max']) / 2
            center_y = (zone_info['y_min'] + zone_info['y_max']) / 2
            
            # Número total de centros en la zona
            pitch.annotate(str(count_in_zone), xy=(center_x, center_y), 
                        c='white', va='center', ha='center',
                        size=18, weight='bold', ax=ax, zorder=20, 
                        path_effects=[patheffects.withStroke(linewidth=4, foreground='black')])
            
            # Flechas promedio
            for curve_type, crosses_list in zone_data[zone_key].items():
                if not crosses_list: 
                    continue
                
                avg_x = np.mean([c['x'] for c in crosses_list])
                avg_y = np.mean([c['y'] for c in crosses_list])
                avg_end_x = np.mean([c['end_x'] for c in crosses_list])
                avg_end_y = np.mean([c['end_y'] for c in crosses_list])
                
                is_left_zone = 'left' in zone_key
                avg_curve_rad = 0
                if curve_type == 'convexa':
                    avg_curve_rad = 0.082 if is_left_zone else -0.082
                elif curve_type == 'concava':
                    avg_curve_rad = -0.082 if is_left_zone else 0.082
                
                # Grosor dinámico
                base_min_thick, base_max_thick = 1.5, 8.0
                current_score = scores.get((zone_key, curve_type), 0)
                normalized_score = current_score / max_score
                dynamic_max_width = base_min_thick + (base_max_thick - base_min_thick) * normalized_score
                dynamic_min_width = max(0.8, dynamic_max_width / 2.5)
                
                self.draw_gradient_arrow(
                    ax, start_pos=(avg_y, avg_x), end_pos=(avg_end_y, avg_end_x),
                    color=curve_info[curve_type]['color'], curve_rad=avg_curve_rad,
                    min_width=dynamic_min_width, max_width=dynamic_max_width
                )
        
        # 🔥 DIBUJAR LOS 2 JUGADORES PRINCIPALES POR LADO (ABAJO DEL CAMPOGRAMA)
        for side in ['left', 'right']:
            if side not in side_player_data or not side_player_data[side]:
                continue
            
            # Obtener top 2 jugadores del lado
            players_sorted = sorted(
                side_player_data[side].items(),
                key=lambda x: x[1]['total'],
                reverse=True
            )[:2]
            
            if not players_sorted:
                continue
            
            # 🔥 POSICIONES ABAJO DEL CAMPOGRAMA
            # X base (abajo del campo)
            base_x = 48
            
            # Y positions según el lado
            if side == 'left':
                # Lado izquierdo: posiciones en y alto (cerca de 100)
                y_positions = [94, 78]  # Jugador 1 más a la izquierda, jugador 2 más al centro
            else:  # right
                # Lado derecho: posiciones en y bajo (cerca de 0)
                y_positions = [22, 6]  # Jugador 1 más al centro, jugador 2 más a la derecha
            
            for idx, (player_name, player_data) in enumerate(players_sorted):
                pos_y = y_positions[idx]
                pos_x = base_x
                
                # 🔢 DORSAL (arriba de todo)
                if player_data['dorsal']:
                    dorsal_x = pos_x - 6.5  # Arriba
                    dorsal_y = pos_y
                    radius = 2.2
                    
                    circle_bg = Circle((dorsal_y, dorsal_x), radius, 
                                    color='yellow', alpha=0.95, zorder=21)
                    ax.add_patch(circle_bg)
                    
                    circle_border = Circle((dorsal_y, dorsal_x), radius, fill=False,
                                        ec='white', linewidth=1.5, zorder=22)
                    ax.add_patch(circle_border)
                    
                    ax.text(dorsal_y, dorsal_x, str(player_data['dorsal']),
                        ha='center', va='center', fontsize=7, 
                        fontweight='bold', color='black', zorder=23)

                # 📸 FOTO DEL JUGADOR (centro)
                self.load_player_photos()
                player_photo = self.get_player_photo(player_name)
                if player_photo is not None:
                    img = OffsetImage(player_photo, zoom=0.18)
                    ab = AnnotationBbox(img, (pos_y, pos_x), frameon=False, zorder=20)
                    ax.add_artist(ab)

                # 📛 NOMBRE (debajo de la foto) - 🔥 FONDO MEJORADO
                apellido = player_name.split()[-1].upper() if player_name != 'N/A' else 'N/A'
                name_x = pos_x + 6  # Debajo de la foto
                name_y = pos_y

                pitch.annotate(apellido, xy=(name_x, name_y),
                            c='white', va='center', ha='center',
                            size=4, weight='bold', ax=ax, zorder=21,  # 🔥 Tamaño 5
                            bbox=dict(boxstyle='round,pad=0.5',  # 🔥 Más padding
                                    facecolor='#1a252f', alpha=1.0,  # 🔥 Fondo más oscuro y opaco
                                    edgecolor='white', linewidth=2))  # 🔥 Borde más grueso

                # 🎯 TIPOS DE LANZAMIENTO (badges más abajo y más grandes)
                types_x = pos_x + 10  # 🔥 MÁS ABAJO (era 10)
                badge_spacing = 3.2  # Espaciado entre badges

                # Centrar verticalmente los badges
                active_badges = sum(1 for ct in ['concava', 'convexa', 'recta'] if player_data[ct] > 0)
                badge_start = pos_y - ((active_badges - 1) * badge_spacing / 2)

                badge_idx = 0
                for curve_t in ['concava', 'convexa', 'recta']:
                    count = player_data[curve_t]
                    if count > 0:
                        badge_y = badge_start + (badge_idx * badge_spacing)
                        color = curve_info[curve_t]['color']
                        label = curve_info[curve_t]['label']
                        
                        pitch.annotate(f'{label}{count}', xy=(types_x, badge_y),
                                    c='white', va='center', ha='center',
                                    size=8, weight='bold', ax=ax, zorder=21,  # 🔥 MÁS GRANDE (era 4)
                                    bbox=dict(boxstyle='round,pad=0.4',  # 🔥 MÁS GRANDE (era 0.2)
                                            facecolor=color, alpha=0.95,
                                            edgecolor='white', linewidth=1.2))  # 🔥 Borde más visible
                        badge_idx += 1
                
                badge_idx = 0
                for curve_t in ['concava', 'convexa', 'recta']:
                    count = player_data[curve_t]
                    if count > 0:
                        badge_y = badge_start + (badge_idx * badge_spacing)
                        color = curve_info[curve_t]['color']
                        label = curve_info[curve_t]['label']
                        
                        pitch.annotate(f'{label}{count}', xy=(types_x, badge_y),
                                    c='white', va='center', ha='center',
                                    size=4.5, weight='bold', ax=ax, zorder=21,
                                    bbox=dict(boxstyle='round,pad=0.25', 
                                            facecolor=color, alpha=0.95,
                                            edgecolor='white', linewidth=1))
                        badge_idx += 1
        
        ax.set_title(title, fontsize=10, fontweight='bold', color='white', 
                    pad=10, bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='#2c3e50', alpha=0.8))

    def get_most_frequent_formation(self, team_name):
        if self.events_df is None: 
            return None
        
        weeks = self.events_df[self.events_df['Team Name'].str.contains(team_name, case=False, na=False)]['Week'].unique()
        formaciones = []
        all_starters_by_position = defaultdict(list)
        
        for week in weeks:
            setup = self.get_team_setup_from_events(team_name, week)
            if setup:
                formaciones.append(setup['formation_name'])
                for pos_idx, dorsal in enumerate(setup['starters']):
                    all_starters_by_position[pos_idx].append(dorsal)
        
        if not formaciones: 
            return None
        
        formacion_ganadora = Counter(formaciones).most_common(1)[0][0]
        top_players_by_position = {}
        for pos_idx in range(11):
            if pos_idx in all_starters_by_position:
                dorsales_en_pos = all_starters_by_position[pos_idx]
                top_2 = Counter(dorsales_en_pos).most_common(2)
                
                if len(top_2) >= 2:
                    top_players_by_position[pos_idx] = {'primary': top_2[0][0], 'secondary': top_2[1][0]}
                elif len(top_2) == 1:
                    top_players_by_position[pos_idx] = {'primary': top_2[0][0], 'secondary': None}
        
        jugadores_primarios = [top_players_by_position[i]['primary'] for i in range(11) if i in top_players_by_position]
        
        
        return {
            'formation': formacion_ganadora, 
            'starters': jugadores_primarios,
            'top_players_by_position': top_players_by_position
        }
    
    def are_teams_equivalent(self, team1, team2):
        if not team1 or not team2: 
            return False
        
        def normalize(text):
            text = unicodedata.normalize('NFD', text)
            text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
            return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).strip()
        
        norm1, norm2 = normalize(team1), normalize(team2)
        if norm1 == norm2: 
            return True
        if ('real' in norm1 and 'atletico' in norm2) or ('atletico' in norm1 and 'real' in norm2): 
            return False
        return SequenceMatcher(None, norm1, norm2).ratio() > 0.85

    def extract_passes(self, team_filter):
        """🔥 FILTRO ESPECIAL: Solo pases en campo contrario (x > 50 y Pass End X > 50)"""
        if self.df is None:
            print("❌ No hay datos cargados")
            return
        titular_info = self.get_most_frequent_formation(team_filter)
        if not titular_info:
            print("❌ No se pudo determinar el once titular")
            return
        titulares_dorsales = set(str(d) for d in titular_info['starters'])
        team_data = self.df[self.df['Team Name'] == team_filter].copy()
        
        for col in ['x', 'y', 'Pass End X', 'Pass End Y']:
            team_data[col] = pd.to_numeric(team_data[col], errors='coerce')
        
        team_data['shirt_number'] = team_data['playerId'].apply(self.get_player_shirt_number)
        
        # 🔥 FILTRO CAMPO CONTRARIO: x > 50 Y Pass End X > 50
        self.passes_data = team_data[
            team_data['shirt_number'].isin(titulares_dorsales) &
            team_data['Pass End X'].notna() &
            team_data['Pass End Y'].notna() &
            (team_data['x'] > 50) &  # 🔥 INICIO en campo contrario
            (team_data['Pass End X'] > 50)  # 🔥 FIN en campo contrario
        ].copy()
        
    
    def load_data(self, team_filter=None):
        try:
            columns_needed = ['Match ID', 'Team ID', 'Team Name', 'Event Name', 'outcome',
                  'x', 'y', 'Pass End X', 'Pass End Y', 'playerName', 'playerId',
                  'timeMin', 'timeSec', 'Cross', 'Right footed', 'Left footed',
                  'Assist', 'Through ball']
            # 🔥 Usar caché en lugar de cargar desde disco
            self.df = self._get_open_play_data(columns=columns_needed)
            self.df = self.df[(self.df['Event Name'] == 'Pass') & (self.df['outcome'] == 1)]
            if team_filter:
                team_matches = self.team_stats[self.team_stats['Team Name'] == team_filter]['Match ID'].unique()
                self.df = self.df[self.df['Match ID'].isin(team_matches)]
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")
    
    def load_match_events(self):
        try:
            # 🔥 Usar caché en lugar de leer desde disco
            self.events_df = self._get_match_events_data()
            if 'Week' in self.events_df.columns:
                self.events_df['Week'] = self.events_df['Week'].astype(str)
            return True
        except Exception as e:
            print(f"⚠️ Error al cargar eventos: {e}")
            self.events_df = None
            return False
    
    def get_team_setup_from_events(self, team_name, week):
        if self.events_df is None: return None
        week_str = str(week)
        setup_events = self.events_df[(self.events_df['Event Name'] == 'Team set up') & (self.events_df['Week'] == week_str)]
        if setup_events.empty: return None
        team_setup = None
        for _, row in setup_events.iterrows():
            opta_team = str(row.get('Team Name', ''))
            if self.are_teams_equivalent(team_name, opta_team):
                team_setup = row
                break
        if team_setup is None: return None
        formation_number = team_setup.get('Team Formation')
        jersey_numbers_str = str(team_setup.get('Jersey Number', ''))
        player_formation_str = str(team_setup.get('Team Player Formation', ''))
        try:
            jersey_list = [int(j.strip()) for j in jersey_numbers_str.split(',') if j.strip().isdigit()]
            formation_slots = [int(s.strip()) for s in player_formation_str.split(',') if s.strip().isdigit()]
            if len(jersey_list) != len(formation_slots): return None
            ordered_starters = [None] * 11 
            for i in range(len(formation_slots)):
                slot_position = formation_slots[i]
                if 1 <= slot_position <= 11:
                    jersey_number = jersey_list[i]
                    ordered_starters[slot_position - 1] = jersey_number
            if None in ordered_starters: return None
            formation_name = self.formation_mapping.get(int(formation_number), f"Unknown_{formation_number}")
            return {'formation_number': int(formation_number), 'formation_name': formation_name, 'starters': ordered_starters}
        except Exception as e:
            return None
    
    def get_player_shirt_number(self, player_id):
        if pd.isna(player_id): return None
        player_info = self.player_stats[self.player_stats['Player ID'] == player_id]
        if not player_info.empty:
            shirt_number = player_info['Shirt Number'].iloc[0]
            return str(int(shirt_number)) if pd.notna(shirt_number) else None
        return None
    
    def get_player_demarcation(self, player_id, equipo):
        if self.events_df is None:
            return None
        
        demarcaciones_encontradas = []
        weeks = self.events_df['Week'].unique()
        
        for week in weeks:
            setup = self.get_team_setup_from_events(equipo, week)
            if not setup:
                continue
            
            player_shirt = self.get_player_shirt_number(player_id)
            if not player_shirt:
                continue
                
            for pos_idx, dorsal in enumerate(setup['starters'], 1):
                if str(dorsal) == str(player_shirt):
                    formation_num = setup['formation_number']
                    if formation_num in self.formation_demarcations:
                        demarcacion = self.formation_demarcations[formation_num].get(pos_idx)
                        if demarcacion:
                            demarcaciones_encontradas.append(demarcacion)
                    break
        
        if demarcaciones_encontradas:
            return Counter(demarcaciones_encontradas).most_common(1)[0][0]
        
        return 'MC'
    
    def extract_pass_to_box_pairs(self):
        """
        🔥 NUEVA LÓGICA: Extrae únicamente la pareja Pasador -> Receptor de los pases que terminan en el área.
        """
        sequences = []

        # 🔥 Usar caché en lugar de cargar desde disco
        all_team_events = self._get_open_play_data()
        all_team_events = all_team_events[all_team_events['Team Name'] == self.team_filter]

        for col in ['x', 'y', 'Pass End X', 'Pass End Y', 'timeMin', 'timeSec']:
            all_team_events[col] = pd.to_numeric(all_team_events[col], errors='coerce')

        all_team_events = all_team_events.sort_values(['Match ID', 'timeMin', 'timeSec']).reset_index()

        # 1. Encontrar todos los pases exitosos que terminan en el área (los "triggers")
        trigger_passes = all_team_events[
            (all_team_events['Event Name'] == 'Pass') &
            (all_team_events['outcome'] == 1) &
            (all_team_events['Pass End X'] >= 83) &
            (all_team_events['Pass End Y'] >= 21.1) & (all_team_events['Pass End Y'] <= 78.9)
        ]
        

        # 2. Para cada pase, crear la secuencia [Pasador, Receptor]
        for idx, trigger_pass in trigger_passes.iterrows():
            
            # El "siguiente evento" en el DataFrame ordenado por tiempo
            if idx + 1 >= len(all_team_events):
                continue # Es el último evento del dataset, no hay receptor

            receiver_event = all_team_events.iloc[idx + 1]

            # Validar que el siguiente evento es del mismo partido y equipo
            if receiver_event['Match ID'] == trigger_pass['Match ID'] and receiver_event['Team Name'] == trigger_pass['Team Name']:
                
                # Evitar "auto-pases" si el siguiente evento es del mismo jugador
                if receiver_event['playerId'] == trigger_pass['playerId']:
                    continue

                player_sequence = [
                    # Jugador 1: El PASADOR
                    {
                        'player_id': trigger_pass['playerId'],
                        'player_name': trigger_pass['playerName'],
                        'shirt_number': self.get_player_shirt_number(trigger_pass['playerId'])
                    },
                    # Jugador 2: El RECEPTOR
                    {
                        'player_id': receiver_event['playerId'],
                        'player_name': receiver_event['playerName'],
                        'shirt_number': self.get_player_shirt_number(receiver_event['playerId'])
                    }
                ]
                sequences.append(player_sequence)

        return sequences
    
    def get_vertical_passers_ranking(self, top_n=3):
        """
        Ranking de jugadores con más pases verticales y asistencias.
        """
        player_stats = defaultdict(lambda: {'vertical_passes': 0, 'assists': 0, 'name': None, 'shirt': None})

        for _, pass_row in self.passes_data.iterrows():
            player_id = pass_row['playerId']

            # Contar pases verticales (progresión en X > 10)
            if pd.notna(pass_row['Pass End X']) and pd.notna(pass_row['x']):
                progression = pass_row['Pass End X'] - pass_row['x']
                if progression > 10:
                    player_stats[player_id]['vertical_passes'] += 1

            # 🔥 SOLUCIÓN DEFINITIVA: Convertir a número antes de comparar
            assist_value = pass_row.get('Assist')
            
            # Intentamos convertir el valor a un número. Si no puede (ej: "No"), se convierte en NaN.
            numeric_assist = pd.to_numeric(assist_value, errors='coerce')
            
            assist_codes = [13, 14, 15, 16]
            
            # Comprobamos si no es NaN y si está en la lista de códigos
            if pd.notna(numeric_assist) and int(numeric_assist) in assist_codes:
                player_stats[player_id]['assists'] += 1
            
            # Guardar nombre y dorsal
            player_stats[player_id]['name'] = pass_row['playerName']
            player_stats[player_id]['shirt'] = pass_row['shirt_number']
    
        # Ordenar por una puntuación combinada
        ranked = sorted(
            player_stats.items(), 
            key=lambda item: (item[1]['assists'] * 3) + item[1]['vertical_passes'],
            reverse=True
        )
    
        return ranked[:top_n]
    
    def extract_cross_finisher_sequences(self):
        """
        Encuentra secuencias Centrador (Cross='Sí') -> Rematador (Miss/Post/Goal/Attempt Saved)
        """
        sequences = []

        # 🔥 Usar caché en lugar de cargar desde disco
        df_all = self._get_open_play_data()
        df_team = df_all[df_all['Team Name'] == self.team_filter]
        
        for idx in range(len(df_team) - 1):
            current = df_team.iloc[idx]
            next_event = df_team.iloc[idx + 1]
            
            # ✅ CENTRADOR: Cross='Sí'
            if current.get('Cross') == 'Sí':
                # ✅ REMATADOR: siguiente evento es remate
                if next_event['Event Name'] in ['Miss', 'Post', 'Goal', 'Attempt Saved']:
                    
                    crosser_id = current['playerId']
                    finisher_id = next_event['playerId']
                    
                    sequences.append({
                        'crosser': {
                            'id': crosser_id,
                            'name': current['playerName'],
                            'shirt': self.get_player_shirt_number(crosser_id)
                        },
                        'finisher': {
                            'id': finisher_id,
                            'name': next_event['playerName'],
                            'shirt': self.get_player_shirt_number(finisher_id)
                        }
                    })
        
        # Contar secuencias repetidas
        from collections import Counter
        sequence_keys = [(s['crosser']['id'], s['finisher']['id']) for s in sequences]
        most_common = Counter(sequence_keys).most_common(2)
        
        result = []
        for (cross_id, finish_id), count in most_common:
            # Buscar info completa
            seq_match = next(s for s in sequences 
                            if s['crosser']['id'] == cross_id and s['finisher']['id'] == finish_id)
            seq_match['count'] = count
            result.append(seq_match)
        
        return result

    def calculate_pass_network_by_demarcation(self, data=None):
        if self.df is None:
            print("❌ No hay datos cargados")
            return {}, {}
        
        # 🔥 USAR passes_data que ya tiene el filtro de campo contrario
        team_data = self.passes_data.copy()
        
        if team_data.empty:
            return {}, {}
        
        
        pass_counts_by_demarcation = defaultdict(int)
        demarcation_positions = defaultdict(lambda: {'x': [], 'y': [], 'count': 0, 'dorsales': set()})
        demarcation_players = defaultdict(set)
        
        titular_info = self.get_most_frequent_formation(self.team_filter)
        player_demarcations_from_formation = {}
        
        if titular_info:
            formation_name = titular_info['formation']
            formation_num = [k for k, v in self.formation_mapping.items() if v == formation_name][0]
            
            if formation_num in self.formation_demarcations:
                for pos_idx, dorsal in enumerate(titular_info['starters'], 1):
                    demarcacion = self.formation_demarcations[formation_num].get(pos_idx, 'MC')
                    player_demarcations_from_formation[str(dorsal)] = demarcacion
        
        data_sorted = team_data.sort_values(['Match ID', 'timeMin', 'timeSec']).reset_index(drop=True)
        player_demarcations_cache = {}
        
        for idx, row in data_sorted.iterrows():
            passer_id = row['playerId']
            passer_name = row['playerName']
            passer_shirt = self.get_player_shirt_number(passer_id)
            if not passer_shirt:
                continue
            
            passer_demarcation = None
            
            if passer_shirt in player_demarcations_from_formation:
                passer_demarcation = player_demarcations_from_formation[passer_shirt]
            elif passer_id in player_demarcations_cache:
                passer_demarcation = player_demarcations_cache[passer_id]
            else:
                passer_demarcation = self.get_player_demarcation(passer_id, self.team_filter)
                player_demarcations_cache[passer_id] = passer_demarcation
            
            if not passer_demarcation:
                passer_demarcation = 'MC'
            
            demarcation_positions[passer_demarcation]['x'].append(row['x'])
            demarcation_positions[passer_demarcation]['y'].append(row['y'])
            demarcation_positions[passer_demarcation]['count'] += 1
            demarcation_positions[passer_demarcation]['dorsales'].add(passer_shirt)
            demarcation_players[passer_demarcation].add(passer_name)
            
            if idx + 1 < len(data_sorted):
                next_pass = data_sorted.iloc[idx + 1]
                
                if (next_pass['Match ID'] == row['Match ID'] and 
                    next_pass['Team Name'] == row['Team Name']):
                    
                    time_diff = (next_pass['timeMin'] * 60 + next_pass['timeSec']) - \
                            (row['timeMin'] * 60 + row['timeSec'])
                    
                    if 0 < time_diff <= 10:
                        receiver_id = next_pass['playerId']
                        
                        if receiver_id != passer_id:
                            receiver_shirt = self.get_player_shirt_number(receiver_id)
                            if not receiver_shirt:
                                continue
                            
                            receiver_demarcation = None
                            
                            if receiver_shirt in player_demarcations_from_formation:
                                receiver_demarcation = player_demarcations_from_formation[receiver_shirt]
                            elif receiver_id in player_demarcations_cache:
                                receiver_demarcation = player_demarcations_cache[receiver_id]
                            else:
                                receiver_demarcation = self.get_player_demarcation(receiver_id, self.team_filter)
                                player_demarcations_cache[receiver_id] = receiver_demarcation
                            
                            if not receiver_demarcation:
                                receiver_demarcation = 'MC'
                            
                            pass_key = tuple(sorted([passer_demarcation, receiver_demarcation]))
                            pass_counts_by_demarcation[pass_key] += 1
        
        all_demarcations = {}
        for demarcacion, data in demarcation_positions.items():
            if data['x']:
                all_demarcations[demarcacion] = {
                    'x': np.mean(data['x']),
                    'y': np.mean(data['y']),
                    'count': data['count'],
                    'players': list(demarcation_players[demarcacion]),
                    'dorsales': sorted([int(d) for d in data['dorsales'] if d.isdigit()])
                }
        
        sorted_demarcations = sorted(
            all_demarcations.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:11]
        
        top_11_demarcations = dict(sorted_demarcations)
        
        for i, (dem, info) in enumerate(sorted_demarcations, 1):
            pass
        
        filtered_passes = {}
        top_11_keys = set(top_11_demarcations.keys())
        
        for pass_key, count in pass_counts_by_demarcation.items():
            dem1, dem2 = pass_key
            if dem1 in top_11_keys and dem2 in top_11_keys:
                filtered_passes[pass_key] = count
        
        
        return filtered_passes, top_11_demarcations
    
    def draw_pass_network_by_demarcation(self, ax, pass_counts, demarcation_positions, 
                                      title="RED DE PASES CAMPO CONTRARIO", min_passes=3):
        if not pass_counts or not demarcation_positions:
            ax.text(0.5, 0.5, 'Sin datos suficientes', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12, fontweight='bold', color='white')
            return

        from mplsoccer import VerticalPitch
        
        pitch = VerticalPitch(pitch_type='opta', pitch_color='#2d5a27', 
                     line_color='white', linewidth=2, label=False, tick=False,
                     half=True)
        pitch.draw(ax=ax)
        
        max_passes = max(pass_counts.values()) if pass_counts else 1
        min_passes_val = min(pass_counts.values()) if pass_counts else 1
        
        def get_color_gradient(count, min_val, max_val):
            if max_val == min_val:
                return '#FFFF00'
            ratio = (count - min_val) / (max_val - min_val)
            r = 255
            g = int(255 * (1 - ratio))
            b = 0
            return f'#{r:02x}{g:02x}{b:02x}'
        
        for (dem1, dem2), count in pass_counts.items():
            if count >= min_passes:
                if dem1 in demarcation_positions and dem2 in demarcation_positions:
                    pos1 = demarcation_positions[dem1]
                    pos2 = demarcation_positions[dem2]
                    
                    width = 1 + (count / max_passes) * 9
                    color = get_color_gradient(count, min_passes_val, max_passes)
                    
                    pitch.lines(pos1['x'], pos1['y'], pos2['x'], pos2['y'],
                            lw=width, color=color, zorder=1, alpha=0.8, ax=ax)
        
        max_count = max(d['count'] for d in demarcation_positions.values())
        
        for demarcacion, info in demarcation_positions.items():
            marker_size = 150 + (info['count'] / max_count) * 350
            
            pitch.scatter(info['x'], info['y'],
                        s=marker_size,
                        color='#e74c3c', edgecolors='white', linewidth=3, 
                        alpha=1, zorder=3, ax=ax)
            
            pitch.annotate(demarcacion, 
                        xy=(info['x'], info['y']),
                        c='white', va='bottom', ha='center',
                        size=7, weight='bold', ax=ax, zorder=4,
                        path_effects=[patheffects.withStroke(linewidth=3, foreground='#2c3e50')])

            dorsales_list = info.get('dorsales', [])
            if dorsales_list:
                dorsales_text = '-'.join(str(d) for d in dorsales_list)
                
                pitch.annotate(dorsales_text, 
                            xy=(info['x'], info['y']),
                            c='yellow', va='top', ha='center',
                            size=5, weight='bold', ax=ax, zorder=4,
                            path_effects=[patheffects.withStroke(linewidth=2.5, foreground='#2c3e50')])
        
        ax.set_title(title, fontsize=10, fontweight='bold', color='white', pad=10, 
                    family='serif', bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='#2c3e50', alpha=0.8))
    
    def create_positional_heatmap(self, ax, title="SECUENCIAS A ÁREA"):
        """Dibuja las 4 secuencias más repetidas que acaben en área o con ocasión"""
        
        if self.passes_data.empty:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='white')
            return
        
        from mplsoccer import VerticalPitch
        
        pitch = VerticalPitch(pitch_type='opta', line_zorder=2,
                            pitch_color='#2d5a27', line_color='white',
                            linewidth=2, label=False, tick=False,
                            half=True)
        pitch.draw(ax=ax)
        
        # Extraer secuencias que acaben en área o con ocasión
        sequences = self.extract_sequences_to_area()
        
        if not sequences:
            ax.text(0.5, 0.5, 'Sin secuencias a área', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#e74c3c', alpha=0.8))
            return
        
        
        # 🔥 SEPARAR SECUENCIAS POR TIPO DE FINALIZACIÓN
        sequences_shot = [s for s in sequences if s[0].get('finish_type') == 'shot']
        sequences_area = [s for s in sequences if s[0].get('finish_type') == 'area']
        
        
        # Encontrar patrones para cada tipo
        patterns_shot = self.find_most_similar_sequences(
            sequences_shot, 
            top_n=2, 
            passes_to_compare=2,
            eps=30,
            min_samples=2
        ) if len(sequences_shot) >= 2 else []
        
        patterns_area = self.find_most_similar_sequences(
            sequences_area, 
            top_n=2, 
            passes_to_compare=2,
            eps=30,
            min_samples=2
        ) if len(sequences_area) >= 2 else []
        
        # Combinar patrones (máximo 4)
        all_patterns = []
        for p in patterns_shot:
            p['finish_type'] = 'shot'
            all_patterns.append(p)
        for p in patterns_area:
            p['finish_type'] = 'area'
            all_patterns.append(p)
        
        # Ordenar por frecuencia y tomar top 4
        all_patterns.sort(key=lambda x: x['count'], reverse=True)
        patterns = all_patterns[:4]
        
        if not patterns:
            ax.text(0.5, 0.5, 'Sin patrones repetidos', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#e67e22', alpha=0.8))
            return
        
        # 🔥 COLORES ÚNICOS PARA CADA PATRÓN
        pattern_colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6']  # Rojo, Azul, Naranja, Morado
        
        # Dibujar cada patrón
        for idx, pattern_data in enumerate(patterns):
            finish_type = pattern_data.get('finish_type', 'area')
            color = pattern_colors[idx % len(pattern_colors)]  # 🔥 Color único por patrón
            
            all_seqs = pattern_data['all_sequences']
            count = pattern_data['count']
            
            
            # Dibujar todas las secuencias del patrón (semitransparentes)
            for sequence in all_seqs:
                for pass_data in sequence:
                    pitch.arrows(
                        pass_data['x'], pass_data['y'],
                        pass_data['end_x'], pass_data['end_y'],
                        color=color, width=1.5, 
                        headwidth=4, headlength=4,
                        alpha=0.3, zorder=5, ax=ax
                    )
            
            # Calcular y dibujar trayectoria promedio (más destacada)
            avg_path = self.calculate_average_path(all_seqs)
            
            if avg_path and len(avg_path) >= 2:
                # Flechas promedio
                for i in range(len(avg_path) - 1):
                    x1, y1 = avg_path[i]
                    x2, y2 = avg_path[i + 1]
                    
                    pitch.arrows(
                        x1, y1, x2, y2,
                        color=color, width=2,
                        headwidth=5, headlength=5,
                        alpha=1.0, zorder=20, ax=ax
                    )
                
                # Punto de inicio (círculo blanco)
                x_start, y_start = avg_path[0]
                pitch.scatter(x_start, y_start, s=150,
                            color='white', edgecolors=color, 
                            linewidth=2, alpha=1.0, zorder=25, ax=ax)
                
                # 🔥 PUNTO FINAL DIFERENTE SEGÚN TIPO
                x_end, y_end = avg_path[-1]
                if finish_type == 'shot':
                    # Estrella para tiros
                    pitch.scatter(x_end, y_end, s=250, marker='*',
                                color=color, edgecolors='white',
                                linewidth=2, alpha=1.0, zorder=25, ax=ax)
                else:
                    # Círculo para pases a área
                    pitch.scatter(x_end, y_end, s=200,
                                color=color, edgecolors='white',
                                linewidth=2, alpha=1.0, zorder=25, ax=ax)
                
                # Etiqueta con número de repeticiones
                pitch.annotate(
                    f'{count}x',
                    xy=(x_end, y_end),
                    c='white', va='center', ha='center',
                    size=8, weight='bold', ax=ax, zorder=26,
                    path_effects=[patheffects.withStroke(linewidth=2.5, foreground='black')]
                )
        
        # Dibujar rectángulo del área (referencia visual)
        from matplotlib.patches import Rectangle
        area_rect = Rectangle((21.1, 83), 57.8, 17, 
                            linewidth=2, edgecolor='yellow', 
                            facecolor='none', linestyle='--',
                            alpha=0.5, zorder=1)
        ax.add_patch(area_rect)
        
        # 🔥 LEYENDA COMPACTA Y PEQUEÑA
        from matplotlib.lines import Line2D
        
        legend_elements = []
        
        for idx, pattern_data in enumerate(patterns):
            finish_type = pattern_data.get('finish_type', 'area')
            color = pattern_colors[idx % len(pattern_colors)]
            count = pattern_data['count']
            
            # Símbolo según tipo
            if finish_type == 'shot':
                marker = '*'
                label = f'P{idx+1} ({count}x) Tiro'
            else:
                marker = 'o'
                label = f'P{idx+1} ({count}x) Área'
            
            legend_elements.append(
                Line2D([0], [0], marker=marker, color='w', 
                    markerfacecolor=color, markersize=6,  # 🔥 Más pequeño
                    label=label, linestyle='None')
            )
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower right', 
                    fontsize=6,  # 🔥 Fuente más pequeña
                    framealpha=0.85,
                    facecolor='#2c3e50', edgecolor='white',
                    labelcolor='white',
                    handletextpad=0.5,  # 🔥 Menos espacio
                    borderpad=0.3)  # 🔥 Menos padding
        
        ax.set_title(title, fontsize=10, fontweight='bold', color='white',
                    pad=10, bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='#2c3e50', alpha=0.8))
        

    def create_pass_flow_map(self, ax, title="MAPA DE CALOR ACCIONES ATAQUE", bins=(6, 4)):
        """Mapa de calor KDE de eventos ofensivos en campo contrario"""
        if self.passes_data.empty:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='white')
            return
        
        from mplsoccer import VerticalPitch
        
        pitch = VerticalPitch(pitch_type='opta', line_zorder=2,
            pitch_color='#2d5a27',  # 🔥 Verde (era '#22312b')
            line_color='white',
            half=True)
        pitch.draw(ax=ax)
        
        # 🔥 CARGAR TODOS LOS EVENTOS (no solo passes_data) - USANDO CACHÉ
        try:
            df_all = self._get_open_play_data()
            
            # Convertir coordenadas
            for col in ['x', 'y']:
                df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
            
            # 🔥 FILTRAR POR EQUIPO Y CAMPO CONTRARIO (x > 50)
            team_data = df_all[
                (df_all['Team Name'] == self.team_filter) &
                (df_all['x'] > 50)
            ].copy()
            
            # 🔥 FILTRAR TIPOS DE EVENTOS
            eventos_ofensivos = [
                'Pass', 'Offside Pass', 'Corner Awarded', 'Take On', 
                'Miss', 'Goal', 'Attempt Saved', 'Post', 'Aerial', 'Chance missed'
            ]
            
            df_eventos = team_data[
                team_data['Event Name'].isin(eventos_ofensivos) &
                team_data['x'].notna() &
                team_data['y'].notna()
            ].copy()
            
            if df_eventos.empty:
                ax.text(0.5, 0.5, 'Sin eventos ofensivos', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='white')
                return
            
            
            # 🔥 CREAR MAPA DE CALOR KDE
            kde = pitch.kdeplot(
                df_eventos.x, 
                df_eventos.y, 
                ax=ax,
                fill=True,           # Rellenar con color
                levels=100,          # Suavidad del gradiente
                thresh=0.1,            # Mostrar todos los valores
                cut=4,               # Extensión del KDE
                cmap='Reds',         # 🔥 Colores rojos
                alpha=0.7,           # Transparencia
                zorder=1
            )
            
        except Exception as e:
            print(f"❌ Error generando mapa de calor: {e}")
            import traceback
            traceback.print_exc()
            ax.text(0.5, 0.5, 'Error al cargar datos', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='white')
            return
        
        ax.set_title(title, fontsize=10, fontweight='bold', color='white',
                    pad=10, bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='#2c3e50', alpha=0.8))
    
    def get_top_pass_sequences(self, sequence_length=2, top_n=2):
        if self.passes_data.empty:
            return []
        
        data_sorted = self.passes_data.sort_values(
            ['Match ID', 'timeMin', 'timeSec']
        ).reset_index(drop=True)
        
        sequences = []
        
        for idx in range(len(data_sorted) - sequence_length):
            current_match = data_sorted.iloc[idx]['Match ID']
            current_team = data_sorted.iloc[idx]['Team Name']
            
            sequence = []
            valid_sequence = True
            first_pass_x = None
            last_pass_end_x = None
            last_pass_outcome = None
            
            for i in range(sequence_length + 1):
                row = data_sorted.iloc[idx + i]
                
                if i == 0:
                    first_pass_x = row.get('x')
                
                if i == sequence_length:
                    last_pass_end_x = row.get('Pass End X')
                    last_pass_outcome = row.get('outcome')
                
                if row['Match ID'] != current_match or row['Team Name'] != current_team:
                    valid_sequence = False
                    break
                
                if i > 0:
                    prev_row = data_sorted.iloc[idx + i - 1]
                    time_diff = (row['timeMin'] * 60 + row['timeSec']) - \
                            (prev_row['timeMin'] * 60 + prev_row['timeSec'])
                    if time_diff > 10 or time_diff <= 0:
                        valid_sequence = False
                        break
                    
                    if row['playerId'] == prev_row['playerId']:
                        valid_sequence = False
                        break
                
                player_id = row['playerId']
                player_name = row['playerName']
                shirt_number = row['shirt_number']
                
                sequence.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'shirt_number': shirt_number
                })
            
            if valid_sequence and len(sequence) == sequence_length + 1:
                player_ids = [p['player_id'] for p in sequence]
                
                progression = 0
                if (first_pass_x is not None and last_pass_end_x is not None and 
                    last_pass_outcome == 1):
                    progression = abs(last_pass_end_x - first_pass_x)
                
                if sequence_length == 1:
                    sequence_key = tuple(sorted(player_ids))
                elif sequence_length == 2:
                    if len(set(player_ids)) != len(player_ids):
                        continue
                    sequence_key = tuple(player_ids)
                elif sequence_length == 3:
                    if len(set(player_ids[:3])) != 3:
                        continue
                    if len(set(player_ids[1:])) != 3:
                        continue
                    sequence_key = tuple(player_ids)
                else:
                    sequence_key = tuple(player_ids)
                
                sequences.append((sequence_key, sequence, progression))
        
        from collections import Counter, defaultdict
        
        sequence_counts = Counter([seq[0] for seq in sequences])
        sequence_max_progression = defaultdict(float)
        sequence_data_map = {}
        
        for seq_key, seq_data, progression in sequences:
            if progression > sequence_max_progression[seq_key]:
                sequence_max_progression[seq_key] = progression
            if seq_key not in sequence_data_map:
                sequence_data_map[seq_key] = seq_data
        
        result = []
        
        if not sequence_counts:
            return result
        
        most_common = sequence_counts.most_common(1)[0]
        seq_tuple_1, count_1 = most_common
        result.append({
            'sequence': sequence_data_map[seq_tuple_1],
            'count': count_1,
            'length': sequence_length,
            'progression': sequence_max_progression[seq_tuple_1]
        })
        
        if top_n >= 2:
            candidates = sequence_counts.most_common(min(10, len(sequence_counts)))
            candidates = [c for c in candidates if c[0] != seq_tuple_1]
            
            if candidates:
                best_candidate = max(
                    candidates,
                    key=lambda x: (sequence_max_progression[x[0]], x[1])
                )
                
                seq_tuple_2, count_2 = best_candidate
                result.append({
                    'sequence': sequence_data_map[seq_tuple_2],
                    'count': count_2,
                    'length': sequence_length,
                    'progression': sequence_max_progression[seq_tuple_2]
                })
                
        
        return result
    
    def load_player_photos(self):
        if self.photos_data is None:
            try:
                with open('assets/jugadores_optimizados.json', 'r', encoding='utf-8') as f:
                    self.photos_data = json.load(f)
            except FileNotFoundError:
                print("⚠️ No se encontró el archivo jugadores_optimizados.json")
                self.photos_data = []
        return self.photos_data
    
    def extract_names_parts(self, name):
        def normalize_name(name):
            if not name:
                return ""
            name = str(name).lower().strip()
            name = unicodedata.normalize('NFD', name)
            name = ''.join(char for char in name if unicodedata.category(char) != 'Mn')
            name = re.sub(r"['\-`´'']", "", name)
            name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
            return ' '.join(name.split())
            
        normalized = normalize_name(name)
        parts = normalized.split()
        
        if not parts:
            return {'full': '', 'first_name': '', 'last_name': '', 'all_parts': []}
            
        first_name = parts[0]
        last_name = parts[-1] if len(parts) > 1 else first_name
        
        return {
            'full': normalized,
            'first_name': first_name,
            'last_name': last_name,
            'all_parts': parts
        }

    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def match_player_name(self, player_name, photos_data, team_filter=None):
        player_parts = self.extract_names_parts(player_name)
        if not player_parts['full']:
            return None

        team_players = []
        if team_filter:
            palabras_ignorar = {'cf', 'fc', 'cd', 'ud', 'rcd', 'rc', 'ca', 'de', 'del', 'la', 'las', 'el', 'los'}
            
            def normalize_word(word):
                word = unicodedata.normalize('NFD', word)
                word = ''.join(char for char in word if unicodedata.category(char) != 'Mn')
                return word.lower().strip()
            
            palabras_equipo = team_filter.split()
            palabras_equipo_norm = []
            
            for palabra in palabras_equipo:
                palabra_norm = normalize_word(palabra)
                if palabra_norm not in palabras_ignorar and len(palabra_norm) > 2:
                    palabras_equipo_norm.append(palabra_norm)
            
            palabras_equipo_ordenadas = sorted(palabras_equipo_norm, key=len, reverse=True)
            
            for photo_entry in photos_data:
                photo_team = photo_entry.get('team_name', '')
                if not photo_team:
                    continue
                
                palabras_photo_team = photo_team.split()
                palabras_photo_norm = [normalize_word(p) for p in palabras_photo_team]
                
                match_encontrado = False
                for palabra_buscar in palabras_equipo_ordenadas:
                    if palabra_buscar in palabras_photo_norm:
                        match_encontrado = True
                        break
                
                if match_encontrado:
                    team_players.append(photo_entry)
            
            if not team_players:
                team_filter_norm = normalize_word(team_filter.replace(' ', ''))
                
                for photo_entry in photos_data:
                    photo_team = photo_entry.get('team_name', '')
                    if not photo_team:
                        continue
                    
                    photo_team_norm = normalize_word(photo_team.replace(' ', ''))
                    similarity = SequenceMatcher(None, team_filter_norm, photo_team_norm).ratio()
                    
                    if similarity > 0.7:
                        team_players.append(photo_entry)
                
                if not team_players:
                    return None
        else:
            team_players = photos_data

        player_words = [w for w in player_parts['all_parts'] if len(w) >= 3]
        player_words_sorted = sorted(player_words, key=len, reverse=True)

        for palabra_buscar in player_words_sorted:
            for photo_entry in team_players:
                photo_name = photo_entry.get('player_name', '')
                photo_parts = self.extract_names_parts(photo_name)
                photo_words = [w for w in photo_parts['all_parts'] if len(w) >= 3]
                
                if palabra_buscar in photo_words:
                    return photo_entry
                
                if len(palabra_buscar) > 5:
                    for ph_word in photo_words:
                        if len(ph_word) > 5:
                            distance = self.levenshtein_distance(palabra_buscar, ph_word)
                            if distance == 1:
                                return photo_entry
        
        candidates = []
        
        for photo_entry in team_players:
            photo_name = photo_entry.get('player_name', '')
            photo_parts = self.extract_names_parts(photo_name)
            photo_words = [w for w in photo_parts['all_parts'] if len(w) >= 3]
            
            matches = []
            for p_word in player_words:
                for ph_word in photo_words:
                    if p_word == ph_word:
                        matches.append(p_word)
                    elif len(p_word) > 5 and len(ph_word) > 5:
                        distance = self.levenshtein_distance(p_word, ph_word)
                        if distance <= 2:
                            matches.append(p_word)
            
            if matches:
                candidates.append({
                    'entry': photo_entry,
                    'matches': matches,
                    'match_count': len(matches)
                })
        
        if len(candidates) == 0:
            return None
        elif len(candidates) == 1:
            return candidates[0]['entry']
        else:
            best_candidates = sorted(candidates, key=lambda x: x['match_count'], reverse=True)
            
            if best_candidates[0]['match_count'] > best_candidates[1]['match_count']:
                return best_candidates[0]['entry']
            
            for candidate in best_candidates:
                photo_parts = self.extract_names_parts(candidate['entry']['player_name'])
                
                for p_word in player_parts['all_parts']:
                    if len(p_word) <= 3:
                        for ph_word in photo_parts['all_parts']:
                            if ph_word.startswith(p_word):
                                return candidate['entry']
            
            return best_candidates[0]['entry']

    def get_player_photo(self, player_name):
        if self.photos_data is None: 
            self.load_player_photos()
        if not self.photos_data: 
            return None
        
        match = self.match_player_name(player_name, self.photos_data, self.team_filter)
        if not match: 
            return None
        
        try:
            img_data = base64.b64decode(match['image_base64'])
            img = Image.open(BytesIO(img_data)).convert('RGBA')
            data = np.array(img)
            
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
    
    def draw_pass_sequence(self, ax, sequence_data, title, panel_color='#3498db', is_second=False):
        ax.set_facecolor('#f0f0f0')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        if not sequence_data or 'sequence' not in sequence_data:
            ax.text(0.5, 0.5, 'Sin datos suficientes', ha='center', va='center',
                    fontsize=8, color='grey')
            return

        sequence = sequence_data['sequence']
        count = sequence_data['count']

        if is_second:
            subtitle_text = f'{count} veces repetida (con mayor progresión)'
        else:
            subtitle_text = f'{count} veces repetida'

        ax.text(0.5, 0.99, subtitle_text, ha='center', va='center',
                fontsize=7, color='grey', style='italic')
        
        n_players = len(sequence)
        
        if n_players == 2:
            x_positions = [0.25, 0.75]
            y_center = 0.5
        elif n_players == 3:
            x_positions = [0.20, 0.50, 0.80]
            y_center = 0.5
        elif n_players == 4:
            x_positions = [0.15, 0.38, 0.62, 0.85]
            y_center = 0.5
        else:
            return
        
        self.load_player_photos()
        
        for i, player in enumerate(sequence):
            x_pos = x_positions[i]
            
            rect_bg = patches.FancyBboxPatch(
                (x_pos - 0.08, y_center - 0.32), 0.16, 0.60,
                boxstyle="round,pad=0.015", 
                facecolor='white',
                edgecolor='lightgrey', 
                linewidth=1.5
            )
            ax.add_patch(rect_bg)
            
            dorsal = str(int(player.get('shirt_number', 0))) if player.get('shirt_number') else '?'
            
            dorsal_x = x_pos + 0.090
            dorsal_y = y_center + 0.40
            
            ax.text(dorsal_x, dorsal_y, dorsal, 
                    ha='right', va='top',
                    fontsize=16,
                    fontweight='heavy',
                    color=panel_color,
                    zorder=15,
                    family='sans-serif',
                    style='italic',
                    path_effects=[
                        patheffects.Stroke(linewidth=4, foreground='white'),
                        patheffects.Normal(),
                        patheffects.SimplePatchShadow(offset=(1, -1), shadow_rgbFace='black', alpha=0.3)
                    ])
            
            player_photo = self.get_player_photo(player.get('player_name', ''))
            if player_photo is not None:
                photo_size = 0.35  
                photo_ax = ax.inset_axes([
                    x_pos - photo_size/2, 
                    y_center - 0.10 - photo_size/2,
                    photo_size, 
                    photo_size
                ])
                photo_ax.imshow(player_photo)
                photo_ax.axis('off')
            
            player_name = player.get('player_name', 'N/A')
            apellido = player_name.split()[-1] if player_name else 'N/A'

            name_badge = patches.FancyBboxPatch(
                (x_pos - 0.055, y_center - 0.30), 0.11, 0.025,
                boxstyle="round,pad=0.003",
                facecolor='white',
                edgecolor=panel_color,
                linewidth=2,
                alpha=0.95,
                zorder=14
            )
            ax.add_patch(name_badge)

            ax.text(x_pos, y_center - 0.288, apellido.upper(), 
                    ha='center', va='center',
                    fontsize=5, fontweight='bold', 
                    color='white', zorder=15)
            
            if i < len(sequence) - 1:
                arrow_start_x = x_pos + 0.09
                arrow_end_x = x_positions[i + 1] - 0.09
                arrow_y = y_center
                
                arrow = patches.FancyArrowPatch(
                    (arrow_start_x, arrow_y), 
                    (arrow_end_x, arrow_y),
                    arrowstyle='->,head_width=0.3,head_length=0.25',
                    color=panel_color,
                    linewidth=2.5,
                    zorder=5,
                    alpha=0.8
                )
                ax.add_patch(arrow)
    
    def extract_sequence_from_goal_kick(self, start_idx, team_name):
        """
        Extrae una secuencia de Pass, Aerial, Take On y otros eventos del mismo equipo.
        La secuencia termina si cambia de equipo o pasan 15 segundos.
        """
        pass_chain = []
        
        # Obtener el evento inicial (saque de puerta)
        start_event = self.df.loc[start_idx]
        match_id = start_event['Match ID']
        start_time = start_event['timeStamp']
        time_limit = start_time + timedelta(seconds=15)
        
        # --- PRIMER PASE (SAQUE DE PUERTA) ---
        first_pass_end_x = float(start_event['Pass End X'])
        first_pass_end_y = float(start_event['Pass End Y'])
        
        pass_chain.append({
            'x': float(start_event['x']),
            'y': float(start_event['y']),
            'end_x': first_pass_end_x,
            'end_y': first_pass_end_y,
            'outcome': start_event['outcome'],
            'team_name': team_name,
            'action_type': 'Pass',
            'value': self.calculate_action_value(start_event),
            'original_index': start_idx
        })
        
        # Guardamos las coordenadas finales para encadenar
        last_end_x = first_pass_end_x
        last_end_y = first_pass_end_y
        
        # Empezar a buscar desde el siguiente evento
        current_idx = start_idx + 1
        
        while current_idx < len(self.df):
            event = self.df.iloc[current_idx]
            
            # --- CONDICIONES DE PARADA ---
            if len(pass_chain) >= 5 or \
            event['Match ID'] != match_id or \
            event['timeStamp'] > time_limit or \
            event['Team Name'] != team_name:
                break
            
            # 🔥 INCLUIR Pass, Aerial, Take On y otros eventos relevantes
            if event['Event Name'] in ['Pass', 'Take On', 'Aerial']:
                
                # Inicio: donde terminó la acción anterior
                current_action_start_x = last_end_x
                current_action_start_y = last_end_y
                
                # Destino según tipo de evento
                if event['Event Name'] == 'Pass':
                    if pd.notna(event.get('Pass End X')) and pd.notna(event.get('Pass End Y')):
                        current_action_end_x = float(event['Pass End X'])
                        current_action_end_y = float(event['Pass End Y'])
                    else:
                        current_idx += 1
                        continue
                
                elif event['Event Name'] == 'Aerial':
                    # Para Aerial, usar la posición del evento
                    if pd.notna(event.get('x')) and pd.notna(event.get('y')):
                        current_action_end_x = float(event['x'])
                        current_action_end_y = float(event['y'])
                    else:
                        current_idx += 1
                        continue
                
                elif event['Event Name'] == 'Take On':
                    # Para Take On, estimar avance
                    if pd.notna(event.get('x')) and pd.notna(event.get('y')):
                        current_action_end_x = float(event['x']) + 2
                        current_action_end_y = float(event['y'])
                    else:
                        current_idx += 1
                        continue
                
                # Añadir acción a la cadena
                pass_chain.append({
                    'x': current_action_start_x,
                    'y': current_action_start_y,
                    'end_x': current_action_end_x,
                    'end_y': current_action_end_y,
                    'outcome': event['outcome'],
                    'team_name': team_name,
                    'action_type': event['Event Name'],
                    'value': self.calculate_action_value(event)
                })
                
                # Actualizar última posición conocida
                last_end_x = current_action_end_x
                last_end_y = current_action_end_y
            
            # Avanzar al siguiente evento
            current_idx += 1
        
        return pass_chain
    
    def find_most_similar_sequences(self, sequences, top_n=3, passes_to_compare=2, eps=25, min_samples=2):
        """
        Encuentra clusters comparando primeros N pases, pero guarda secuencias COMPLETAS
        """
        
        # 1. Filtrar secuencias con al menos N pases
        eligible_sequences = [
            (original_idx, seq) for original_idx, seq in enumerate(sequences) 
            if len(seq) >= passes_to_compare
        ]

        if len(eligible_sequences) < min_samples:
            print(f"⚠️ Solo hay {len(eligible_sequences)} secuencias con {passes_to_compare}+ pases.")
            return []

        original_indices, filtered_seqs = zip(*eligible_sequences)

        # 🔥 2. PRE-AGRUPAR POR DIRECCIÓN Y PROFUNDIDAD
        def get_detailed_zone(first_pass):
            end_x = first_pass['end_x']
            end_y = first_pass['end_y']
            
            if end_y < 33.3:
                lateral = 'L'
            elif end_y < 66.6:
                lateral = 'C'
            else:
                lateral = 'R'
            
            if end_x < 33.3:
                depth = 'S'
            elif end_x < 50.0:
                depth = 'M'
            else:
                depth = 'L'
            
            return f"{lateral}{depth}"
        
        from collections import defaultdict
        sequences_by_zone = defaultdict(list)
        
        for idx, seq in enumerate(filtered_seqs):
            first_pass = seq[0]
            zone = get_detailed_zone(first_pass)
            sequences_by_zone[zone].append((idx, seq))
        
        zone_names = {
            'LS': 'Izquierda-Corto', 'LM': 'Izquierda-Medio', 'LL': 'Izquierda-Largo',
            'CS': 'Centro-Corto', 'CM': 'Centro-Medio', 'CL': 'Centro-Largo',
            'RS': 'Derecha-Corto', 'RM': 'Derecha-Medio', 'RL': 'Derecha-Largo'
        }
        for zone, seqs in sorted(sequences_by_zone.items()):
            zone_name = zone_names.get(zone, zone)
        
        # 🔥 3. CLUSTERING DTW
        all_clusters = []
        
        for zone, zone_sequences in sequences_by_zone.items():
            if len(zone_sequences) < min_samples:
                print(f"   ⚠️ Zona {zone_names.get(zone, zone)}: muy pocas secuencias ({len(zone_sequences)}), omitida")
                continue
            
            zone_indices, zone_seqs = zip(*zone_sequences)
            
            # 🔥 CREAR TRAYECTORIAS PARA CLUSTERING: Solo primeros N pases
            paths_for_clustering = []
            for seq in zone_seqs:
                seq_subset = seq[:passes_to_compare]  # Solo primeros N pases para comparar
                path = [(seq_subset[0]['x'], seq_subset[0]['y'])]
                for pass_data in seq_subset:
                    path.append((pass_data['end_x'], pass_data['end_y']))
                paths_for_clustering.append(np.array(path))
            
            # Matriz de distancias DTW
            num_paths = len(paths_for_clustering)
            distance_matrix = np.zeros((num_paths, num_paths))
            
            for i in range(num_paths):
                for j in range(i, num_paths):
                    distance, _ = fastdtw(paths_for_clustering[i], paths_for_clustering[j], dist=euclidean)
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
            
            # DBSCAN
            eps_zone = eps * 0.8
            db = DBSCAN(eps=eps_zone, min_samples=min_samples, metric='precomputed').fit(distance_matrix)
            labels = db.labels_
            
            valid_labels = [label for label in labels if label != -1]
            
            if valid_labels:
                label_counts = Counter(valid_labels)
                for cluster_label, count in label_counts.items():
                    cluster_indices = [i for i, label in enumerate(labels) if label == cluster_label]
                    # 🔥 GUARDAR SECUENCIAS COMPLETAS (con punto final) para dibujar
                    cluster_sequences = [zone_seqs[i] for i in cluster_indices]
                    
                    all_clusters.append({
                        'zone': zone,
                        'zone_name': zone_names.get(zone, zone),
                        'count': count,
                        'sequences': cluster_sequences  # Completas
                    })
            else:
                print(f"   ⚠️ Zona {zone_names.get(zone, zone)}: sin clusters claros")
        
        # 4. Seleccionar top clusters
        all_clusters.sort(key=lambda x: x['count'], reverse=True)
        top_clusters = all_clusters[:top_n]
        
        # 5. Formatear resultado
        result = []
        for idx, cluster in enumerate(top_clusters):
            result.append({
                'pattern': f"Patrón #{idx+1} ({cluster['zone_name']})",
                'count': cluster['count'],
                'all_sequences': cluster['sequences']  # Completas con punto final
            })
        
        for i, r in enumerate(result, 1):
            pass

        return result

    def calculate_average_path(self, sequences):
        """Calcula la trayectoria promedio de un conjunto de secuencias"""
        if not sequences:
            return None
        
        # Encontrar longitud mínima
        min_len = min(len(seq) for seq in sequences)
        
        avg_path = []
        for i in range(min_len + 1):  # +1 para incluir punto inicial
            x_coords = []
            y_coords = []
            
            for seq in sequences:
                if i == 0:
                    # 🔥 NO INTERCAMBIAR - Punto inicial
                    x_coords.append(seq[0]['x'])
                    y_coords.append(seq[0]['y'])
                else:
                    # 🔥 NO INTERCAMBIAR - Puntos finales de pases
                    x_coords.append(seq[i-1]['end_x'])
                    y_coords.append(seq[i-1]['end_y'])
            
            avg_x = np.mean(x_coords)
            avg_y = np.mean(y_coords)
            avg_path.append((avg_x, avg_y))  # 🔥 (X, Y) en orden correcto
        
        return avg_path
    
    def draw_sequence_panel(self, ax, seq_data, color):
        """Dibuja un panel de secuencia compacto - Distingue Pass y Take On"""
        ax.set_facecolor('#f0f0f0')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        if not seq_data or 'all_sequences' not in seq_data:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                    fontsize=9, color='grey')
            return
        
        count = seq_data['count']
        all_seqs = seq_data['all_sequences']
        
        # Título con contador
        ax.text(0.5, 0.95, f'{count} repeticiones', 
                ha='center', va='top',
                fontsize=8, fontweight='bold', color='#2c3e50')
        
        from matplotlib.patches import Rectangle
        
        # Fondo del campo (mini)
        campo = Rectangle((0.1, 0.1), 0.8, 0.75, 
                        facecolor='#2d5a27', edgecolor='white', linewidth=2)
        ax.add_patch(campo)
        
        # Línea de medio campo
        ax.plot([0.1, 0.9], [0.475, 0.475], 'white', linewidth=1.5)
        
        # Dibujar TODAS las secuencias
        for sequence in all_seqs:
            for i, pass_data in enumerate(sequence):
                # Normalizar coordenadas
                x_start_norm = 0.1 + (pass_data['x'] / 100.0) * 0.8
                y_start_norm = 0.1 + (pass_data['y'] / 100.0) * 0.75
                x_end_norm = 0.1 + (pass_data['end_x'] / 100.0) * 0.8
                y_end_norm = 0.1 + (pass_data['end_y'] / 100.0) * 0.75
                
                # 🔥 DISTINGUIR ENTRE PASS Y TAKE ON
                if pass_data.get('action_type') == 'Take On':
                    # Take On: línea punteada
                    ax.plot([y_start_norm, y_end_norm], 
                        [x_start_norm, x_end_norm],
                        linestyle='--', linewidth=1.5, 
                        color=color, alpha=0.4)
                    # Marcador
                    ax.plot(y_end_norm, x_end_norm, 'o', 
                        markersize=4, color=color, alpha=0.5)
                else:
                    # Pass: flecha normal
                    ax.annotate('', 
                            xy=(y_end_norm, x_start_norm),
                            xytext=(y_start_norm, x_start_norm),
                            arrowprops=dict(arrowstyle='->', 
                                            lw=1.5, color=color, alpha=0.4))
        
        # Trayectoria promedio
        avg_path = self.calculate_average_path(all_seqs)
        if avg_path and len(avg_path) >= 2:
            for i in range(len(avg_path) - 1):
                x1_norm = 0.1 + (avg_path[i][1] / 100.0) * 0.8
                y1_norm = 0.1 + (avg_path[i][0] / 100.0) * 0.75
                x2_norm = 0.1 + (avg_path[i+1][1] / 100.0) * 0.8
                y2_norm = 0.1 + (avg_path[i+1][0] / 100.0) * 0.75
                
                ax.annotate('',
                        xy=(y2_norm, x1_norm),
                        xytext=(y1_norm, x1_norm),
                        arrowprops=dict(arrowstyle='->', 
                                        lw=3, color=color, alpha=1.0))
            
            # Punto de inicio
            x_start = 0.1 + (avg_path[0][1] / 100.0) * 0.8
            y_start = 0.1 + (avg_path[0][0] / 100.0) * 0.75
            ax.plot(y_start, x_start, 'o', 
                    color='white', markersize=8, 
                    markeredgecolor=color, markeredgewidth=2, zorder=10)

    def extract_sequences_to_area(self):
        """Extrae secuencias de 2-3 pases que TERMINAN en área o con ocasión (contando hacia atrás) - CON DEBUG"""
        sequences = []
        
        
        # Cargar TODO el dataframe, no solo passes_data - USANDO CACHÉ
        try:
            df_all = self._get_open_play_data()
            
            # Convertir coordenadas
            for col in ['x', 'y', 'Pass End X', 'Pass End Y']:
                df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
            
            # Convertir timestamp
            if 'timeStamp' not in df_all.columns:
                print("⚠️ No existe columna 'timeStamp', creando artificial...")
                df_all['timeStamp'] = pd.to_datetime(
                    df_all['timeMin'].astype(str) + ':' + df_all['timeSec'].astype(str),
                    format='%M:%S',
                    errors='coerce'
                )
            else:
                df_all['timeStamp'] = pd.to_datetime(df_all['timeStamp'], format='ISO8601')
            
            df_all = df_all.sort_values(['Match ID', 'timeMin', 'timeSec']).reset_index(drop=True)
            
        except Exception as e:
            print(f"❌ Error cargando datos completos: {e}")
            import traceback
            traceback.print_exc()
            return sequences
        
        # Filtrar solo eventos del equipo
        df_team = df_all[df_all['Team Name'] == self.team_filter].copy()
        df_team = df_team.reset_index(drop=True)  # 🔥 RESETEAR ÍNDICE
        
        # Mostrar tipos de eventos disponibles
        event_counts = df_team['Event Name'].value_counts()
        for event, count in event_counts.head(15).items():
            pass
        
        occasion_events = ['Goal', 'Cross', 'Post', 'Attempt Saved']
        
        # Contador de ocasiones encontradas
        occasion_count = 0
        pass_to_area_count = 0
        valid_sequences_count = 0
        
        
        # Buscar todos los eventos de ocasión o pases que acaben en área
        for idx in range(len(df_team)):
            event = df_team.iloc[idx]
            
            # CONDICIÓN 1: Evento de ocasión
            is_occasion = event['Event Name'] in occasion_events
            if is_occasion:
                occasion_count += 1
            
            # CONDICIÓN 2: Pase que acaba en área
            is_pass_to_area = False
            if event['Event Name'] == 'Pass' and event['outcome'] == 1:
                end_x = event.get('Pass End X')
                end_y = event.get('Pass End Y')
                if pd.notna(end_x) and pd.notna(end_y):
                    if 83 <= end_x <= 100 and 21.1 <= end_y <= 78.9:
                        is_pass_to_area = True
                        pass_to_area_count += 1
            
            # Si cumple alguna condición, contar hacia atrás
            if is_occasion or is_pass_to_area:
                sequence = self.build_sequence_backwards(df_team, idx, debug=(valid_sequences_count < 3))
                
                if not sequence or len(sequence) < 2:
                    continue
                
                final_event = event
                last_pass = sequence[-1]
                
                # 🔥 CASO 1: PASE QUE TERMINA EN ÁREA
                if is_pass_to_area:
                    final_x = float(final_event['Pass End X'])
                    final_y = float(final_event['Pass End Y'])
                    finish_type = 'area'
                    
                    # Verificar que realmente esté en área
                    if not (83 <= final_x <= 100 and 21.1 <= final_y <= 78.9):
                        if valid_sequences_count < 3:
                            print(f"     ⚠️ Pase descartado: Pass End fuera de área ({final_x:.1f}, {final_y:.1f})")
                        continue
                    
                    # Si el último pase de la secuencia no es el evento final, agregarlo
                    if last_pass['end_x'] != final_x or last_pass['end_y'] != final_y:
                        sequence.append({
                            'x': last_pass['end_x'],
                            'y': last_pass['end_y'],
                            'end_x': final_x,
                            'end_y': final_y,
                            'outcome': 1,
                            'original_index': final_event.name,
                            'action_type': 'Pass',
                            'finish_type': finish_type
                        })
                
                # 🔥 CASO 2: OCASIÓN (Goal, Attempt Saved, Post, Cross)
                elif is_occasion:
                    # Para ocasiones, necesitamos las coordenadas x, y del evento
                    if not (pd.notna(final_event.get('x')) and pd.notna(final_event.get('y'))):
                        if valid_sequences_count < 3:
                            print(f"     ⚠️ Ocasión descartada: sin coordenadas x,y")
                        continue
                    
                    final_x = float(final_event['x'])
                    final_y = float(final_event['y'])
                    finish_type = 'shot'
                    
                    # Verificar que esté en área
                    if not (83 <= final_x <= 100 and 21.1 <= final_y <= 78.9):
                        if valid_sequences_count < 3:
                            print(f"     ⚠️ Ocasión descartada: x,y fuera de área ({final_x:.1f}, {final_y:.1f})")
                        continue
                    
                    # Agregar el evento de ocasión como punto final
                    sequence.append({
                        'x': last_pass['end_x'],
                        'y': last_pass['end_y'],
                        'end_x': final_x,
                        'end_y': final_y,
                        'outcome': 1,
                        'original_index': final_event.name,
                        'action_type': 'Shot',
                        'finish_type': finish_type
                    })
                
                # 🔥 AGREGAR finish_type A TODOS LOS ELEMENTOS
                for seq_dict in sequence:
                    seq_dict['finish_type'] = finish_type
                
                # ✅ VERIFICACIÓN FINAL
                last_element = sequence[-1]
                if not (83 <= last_element['end_x'] <= 100 and 21.1 <= last_element['end_y'] <= 78.9):
                    if valid_sequences_count < 3:
                        print(f"     ⚠️ Secuencia descartada: punto final fuera de área")
                    continue
                
                valid_sequences_count += 1
                sequences.append(sequence)
                
                if valid_sequences_count <= 3:
                    last = sequence[-1]
        

        # 🔥 AGREGAR CONTEO POR TIPO
        shot_count = sum(1 for s in sequences if s[0].get('finish_type') == 'shot')
        area_count = sum(1 for s in sequences if s[0].get('finish_type') == 'area')
        
        # 🔥 DEBUG: Verificar que las secuencias terminan en área
        for i, seq in enumerate(sequences[:5]):  # Mostrar primeras 5
            last = seq[-1]
            finish = last.get('finish_type', '?')
            in_area = (83 <= last['end_x'] <= 100 and 21.1 <= last['end_y'] <= 78.9)

        return sequences
    
    def build_sequence_backwards(self, df_team, end_idx, debug=False):
        """Construye secuencia HACIA ATRÁS desde un evento final - CON DEBUG"""
        sequence = []
        max_length = 3  # Máximo 3 pases hacia atrás
        time_limit = 15  # segundos hacia atrás
        
        end_event = df_team.iloc[end_idx]
        match_id = end_event['Match ID']
        end_time = end_event.get('timeStamp')
        
        
        if pd.isna(end_time):
            if debug:
                print(f"        ❌ Sin timestamp válido")
            return None
        
        # Buscar pases ANTERIORES del mismo equipo
        previous_passes = df_team[
            (df_team['Match ID'] == match_id) &
            (df_team.index < end_idx) &
            (df_team['Event Name'] == 'Pass') &
            (df_team['outcome'] == 1) &
            (df_team['Team Name'] == end_event['Team Name'])
        ].tail(10)  # Últimos 10 pases antes del evento
        
        
        if previous_passes.empty:
            if debug:
                print(f"        ❌ No hay pases previos")
            return None
        
        # Construir secuencia hacia atrás
        passes_added = 0
        for i, (_, pass_event) in enumerate(previous_passes.iloc[::-1].iterrows()):  # Invertir orden
            pass_time = pass_event.get('timeStamp')
            
            if pd.isna(pass_time):
                continue
            
            # Verificar que esté dentro del límite de tiempo
            time_diff = (end_time - pass_time).total_seconds()
            
            
            if time_diff < 0 or time_diff > time_limit:
                if debug and i < 3:
                    print(f" ❌ (fuera de límite)")
                continue
            
            # Verificar que tenga coordenadas válidas
            has_coords = (pd.notna(pass_event.get('x')) and 
                        pd.notna(pass_event.get('y')) and
                        pd.notna(pass_event.get('Pass End X')) and
                        pd.notna(pass_event.get('Pass End Y')))
            
            if debug and i < 3:
                if has_coords:
                    pass
                else:
                    print(f" ❌ (sin coordenadas)")
            
            if has_coords:
                sequence.insert(0, {  # Insertar al inicio (porque vamos hacia atrás)
                    'x': float(pass_event['x']),
                    'y': float(pass_event['y']),
                    'end_x': float(pass_event['Pass End X']),
                    'end_y': float(pass_event['Pass End Y']),
                    'outcome': pass_event['outcome'],
                    'original_index': pass_event.name,
                    'action_type': 'Pass'
                })
                
                passes_added += 1
                
                # Actualizar end_time para el siguiente pase
                end_time = pass_time
                
                if len(sequence) >= max_length:
                    break
        
        if debug:
            if len(sequence) >= 1:
                last = sequence[-1]

        return sequence if len(sequence) >= 2 else None

    def load_ball_image(self): 
        return plt.imread("assets/balon.png") if os.path.exists("assets/balon.png") else None

    def load_background(self): 
        return plt.imread("assets/fondo_informes.png") if os.path.exists("assets/fondo_informes.png") else None
    
    def load_team_logo(self, equipo, target_size=(80, 80)):
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            return None
        
        if not os.path.exists('assets/escudos'):
            return None
        
        def normalize_word(word):
            word = unicodedata.normalize('NFD', word)
            word = ''.join(char for char in word if unicodedata.category(char) != 'Mn')
            return word.lower().strip()
        
        palabras_ignorar = {'cf', 'fc', 'cd', 'ud', 'rcd', 'rc', 'ca', 'de', 'del', 'la', 'las', 'el', 'los'}
        palabras = equipo.split()
        palabras_normalizadas = []
        
        for palabra in palabras:
            palabra_norm = normalize_word(palabra)
            if palabra_norm not in palabras_ignorar and len(palabra_norm) > 2:
                palabras_normalizadas.append(palabra_norm)
        
        palabras_ordenadas = sorted(palabras_normalizadas, key=len, reverse=True)
        all_files = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
        
        for palabra_buscar in palabras_ordenadas:
            for filename in all_files:
                nombre_archivo = os.path.splitext(filename)[0]
                nombre_archivo_norm = normalize_word(nombre_archivo)
                
                if palabra_buscar == nombre_archivo_norm or palabra_buscar in nombre_archivo_norm:
                    logo_path = f"assets/escudos/{filename}"
                    
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
                        continue
        
        best_match_path = None
        best_score = 0
        equipo_completo_norm = normalize_word(equipo.replace(' ', ''))
        
        for filename in all_files:
            nombre_archivo = os.path.splitext(filename)[0]
            nombre_archivo_norm = normalize_word(nombre_archivo)
            score = SequenceMatcher(None, equipo_completo_norm, nombre_archivo_norm).ratio()
            
            if score > best_score:
                best_score = score
                best_match_path = f"assets/escudos/{filename}"
        
        if best_match_path and best_score > 0.5:
            try:
                with Image.open(best_match_path) as img:
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
        
        return None

    def guardar_sin_espacios(self, fig, filename):
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

    def create_passing_network_visualization(self, figsize=(16.5, 8.27), team_filter=None):
        """🔥 VISUALIZACIÓN REESTRUCTURADA CON RANKINGS MEJORADOS"""
        if self.passes_data.empty:
            print("❌ No hay datos de pases para visualizar.")
            return None
        
        self.photos_data = self.load_player_photos()
        
        fig = plt.figure(figsize=figsize, facecolor='white')

        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25)
            ax_bg.axis('off')
        
        # 🔥 NUEVO LAYOUT: 2 filas x 4 columnas. Las 2 primeras para campogramas, las 2 últimas para rankings.
        gs = fig.add_gridspec(2, 4, 
                            width_ratios=[1.2, 1.2, 0.8, 0.8], # Más espacio para campogramas
                            height_ratios=[1, 1],
                            hspace=0.08, wspace=0.02,
                            left=0.03, right=0.97, top=0.88, bottom=0.03)
        
        fig.suptitle(f'ANÁLISIS PASES CAMPO CONTRARIO - {team_filter.upper()}', 
                    fontsize=16, fontweight='bold', color='#1e3d59', y=0.96, family='serif')

        if (team_logo := self.load_team_logo(team_filter)) is not None:
            ax_team = fig.add_axes([0.90, 0.90, 0.08, 0.08])
            ax_team.imshow(team_logo, aspect='auto')
            ax_team.axis('off')

        # ═══════════════════════════════════════════════════════════
        # COLUMNAS 0 y 1: CAMPOGRAMAS
        # ═══════════════════════════════════════════════════════════
        ax_net = fig.add_subplot(gs[0, 0])
        pass_counts_dem, pos_dem = self.calculate_pass_network_by_demarcation()
        self.draw_pass_network_by_demarcation(ax_net, pass_counts_dem, pos_dem, "RED DE PASES", min_passes=3)
        
        ax_heatmap = fig.add_subplot(gs[0, 1])
        self.create_positional_heatmap(ax_heatmap, "PATRONES DE LLEGADA ÁREA")

        ax_crosses = fig.add_subplot(gs[1, 0])
        self.create_crosses_map(ax_crosses, "CENTROS AL ÁREA")
        
        ax_flow = fig.add_subplot(gs[1, 1])
        self.create_pass_flow_map(ax_flow, "MAPA DE CALOR OFENSIVO")
        
        # ═══════════════════════════════════════════════════════════
        # 🔥 COLUMNAS 2 y 3: NUEVOS RANKINGS
        # ═══════════════════════════════════════════════════════════
        
        # Panel Fila 0, Columna 2: Asistentes
        ax_assists = fig.add_subplot(gs[0, 2])
        ranking_vertical = self.get_vertical_passers_ranking(top_n=2)
        self.draw_top_assisters_panel(ax_assists, ranking_vertical)
        
        # Panel Fila 0, Columna 3: Centro -> Remate
        ax_cross_finish = fig.add_subplot(gs[0, 3])
        cross_finish_seqs = self.extract_cross_finisher_sequences()
        self.draw_cross_finisher_panel(ax_cross_finish, cross_finish_seqs)

        # Panel Fila 1 (completa): Secuencias a área
        ax_seq_box = fig.add_subplot(gs[1, 2:]) # Ocupa las columnas 2 y 3 de la fila 1
        sequences_to_box = self.extract_pass_to_box_pairs()
        self.draw_sequences_to_box_panel(ax_seq_box, sequences_to_box)

        
        return fig

    def print_summary(self, team_filter=None):
        if self.passes_data.empty:
            return
        
        
        top_passers = self.passes_data['playerName'].value_counts().head(5)
        for i, (player, count) in enumerate(top_passers.items(), 1):
            shirt = self.passes_data[self.passes_data['playerName'] == player]['shirt_number'].iloc[0]

def seleccionar_equipo_interactivo():
    try:
        # 🔥 Usar caché en lugar de leer desde disco
        df = PasesCampoContrario._get_open_play_data(columns=['Team Name'])
        equipos = sorted(df['Team Name'].dropna().unique())
        if not equipos: 
            return None
        
        for i, equipo in enumerate(equipos, 1): 
            pass
        
        max_intentos = 3
        for intento in range(max_intentos):
            try:
                indice = int(input(f"\nSelecciona un equipo (1-{len(equipos)}): ").strip()) - 1
                if 0 <= indice < len(equipos):
                    return equipos[indice]
                else:
                    pass
            except ValueError:
                pass
            except EOFError:
                # Input automático agotado - usar primer equipo como fallback
                print("⚠️ Input automático - usando primer equipo disponible")
                return equipos[0] if equipos else None

        # Si se agotan los intentos, usar primer equipo
        print("⚠️ Máximo de intentos alcanzado - usando primer equipo")
        return equipos[0] if equipos else None
    except Exception as e: 
        return None

def main():
    try:
        if (equipo := seleccionar_equipo_interactivo()) is None:
            return
        
        analyzer = PasesCampoContrario(team_filter=equipo)
        analyzer.print_summary(team_filter=equipo)
        
        if (fig := analyzer.create_passing_network_visualization(team_filter=equipo)):
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"pases_campo_contrario_{equipo_filename}.pdf"
            analyzer.guardar_sin_espacios(fig, output_path)
            plt.show()
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()

def generar_analisis_personalizado(equipo, mostrar=True, guardar=True):
    try:
        analyzer = PasesCampoContrario(team_filter=equipo)
        analyzer.print_summary(team_filter=equipo)
        fig = analyzer.create_passing_network_visualization(team_filter=equipo)
        
        if fig:
            if mostrar: plt.show()
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"pases_campo_contrario_{equipo_filename}.pdf"
                analyzer.guardar_sin_espacios(fig, output_path)
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
    try:
        # 🔥 Usar caché para evitar cargar el parquet dos veces
        df = PasesCampoContrario._get_open_play_data()
        equipos = sorted(df['Team Name'].dropna().unique())
        if equipos:
            pass
        main()
    except Exception as e:
        print(f"❌ Error al inicializar: {e}")
    finally:
        # 🧹 Liberar memoria al finalizar
        import gc
        PasesCampoContrario._open_play_cache = None
        PasesCampoContrario._team_stats_cache = None
        PasesCampoContrario._player_stats_cache = None
        gc.collect()
        print("🧹 Memoria liberada al finalizar el script")