import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import numpy as np
from datetime import datetime
import warnings
import os
import re
import unicodedata
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import patheffects

warnings.filterwarnings('ignore')

class AnalizadorSaquesBanda:
    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/open_play_events.parquet"):
        """
        Inicializa el analizador de saques de banda ofensivos
        """
        self.data_path = data_path
        self.df = None
        self.df_complete = None
        self.team_filter = None
        
    def load_data(self, team_filter=None):
        """Carga y filtra los datos de eventos"""
        try:
            print(f"📂 Cargando datos desde {self.data_path}...")
            self.df_complete = pd.read_parquet(self.data_path)
            print(f"✅ Datos cargados: {len(self.df_complete)} eventos totales")
            
            # Filtrar por equipo si se especifica
            if team_filter:
                self.team_filter = team_filter
                self.df = self.df_complete[self.df_complete['Team Name'] == team_filter].copy()
                print(f"🔍 Filtrado por equipo '{team_filter}': {len(self.df)} eventos")
            else:
                self.df = self.df_complete.copy()
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_zone_from_x_coord(self, x):
        """
        Determina la zona según la coordenada X
        Zona 3 (Ofensiva): 66.6 a 100 - Campo rival
        Zona 2 (Media): 33.3 a 66.6 - Zona media
        Zona 1 (Defensiva): 0 a 33.3 - Campo propio
        """
        if pd.isna(x):
            return None
        if 66.6 <= x <= 100:
            return 3  # Zona defensiva
        elif 33.3 <= x < 66.6:
            return 2  # Zona media
        elif 0 <= x < 33.3:
            return 1  # Zona ofensiva
        return None
        
    
    def extract_throw_in_sequences(self, team_name):
        """
        Extrae secuencias de saques de banda.
        - USA el ángulo de la columna 'Angle' y lo normaliza al sistema 0-180°.
        - Lee la distancia desde la columna 'Length'.
        """
        if self.df is None:
            return []
        
        throw_ins = self.df[
            (self.df['Team Name'] == team_name) & 
            (self.df['Event Name'] == 'Pass') &
            (self.df['Throw in'] == 'Sí')
        ].copy()
        
        print(f"🎯 Saques de banda encontrados: {len(throw_ins)}")
        
        if len(throw_ins) == 0:
            return []
        
        sequences = [] 
        
        for idx, throw_in in throw_ins.iterrows():
            # --- LECTURA Y CONVERSIÓN DEL ÁNGULO (LÓGICA FINAL Y CORRECTA) ---
            final_angle = None
            try:
                # 1. Leer el ángulo en radianes y convertir a grados
                angle_rad = float(throw_in.get('Angle'))
                angle_deg = np.degrees(angle_rad)
                
                side = 'Izquierda' if throw_in.get('y', 50) >= 50 else 'Derecha'

                # 2. Aplicar la FÓRMULA DE CONVERSIÓN CORRECTA según el lado
                if side == 'Izquierda':
                    # 0° (atrás) → 0°, 180° (adelante) → 180°, 360° (adelante) → 180°
                    # Normalizar: si está entre 180-360, restar 180
                    final_angle = angle_deg if angle_deg <= 180 else angle_deg - 180
                elif side == 'Derecha':
                    # 0° (adelante) → 180°, 180° (atrás) → 0°
                    final_angle = 180 - angle_deg

            except (ValueError, TypeError):
                final_angle = None

            # --- LECTURA DE LA DISTANCIA ---
            try:
                distance = float(throw_in.get('Length'))
            except (ValueError, TypeError):
                distance = None
            
            # Coordenadas y velocidad
            try:
                end_x = float(throw_in.get('Pass End X'))
                end_y = float(throw_in.get('Pass End Y'))
            except (ValueError, TypeError):
                end_x, end_y = None, None

            speed_category = self.calculate_throw_speed(idx, throw_in.get('timeStamp'))
            
            sequence = {
                'match_id': throw_in.get('Match ID'),
                'period': throw_in.get('Period'),
                'minute': throw_in.get('Minute'),
                'second': throw_in.get('Second'),
                'x': throw_in.get('x'),
                'y': throw_in.get('y'),
                'end_x': end_x,
                'end_y': end_y,
                'outcome': throw_in.get('outcome'),
                'player': throw_in.get('Player Name'),
                'zone_origin': self.get_zone_from_x_coord(throw_in.get('x', 0)),
                'zone_dest': self.get_zone_from_x_coord(end_x),
                'side': side,
                'timestamp': throw_in.get('timeStamp'),
                'angle': final_angle,
                'speed': speed_category,
                'distance': distance
            }
            
            sequences.append(sequence)
        
        return sequences
        
    def calculate_throw_angle(self, x, y, end_x, end_y):
        """
        Calcula el ángulo del lanzamiento en grados
        Retorna None si las coordenadas no son válidas
        """
        if pd.isna(x) or pd.isna(y) or pd.isna(end_x) or pd.isna(end_y):
            return None
        
        # Calcular diferencias
        dx = end_x - x
        dy = end_y - y
        
        # Calcular ángulo en radianes y convertir a grados
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        # Normalizar a rango 0-360
        if angle_deg < 0:
            angle_deg += 360
        
        return round(angle_deg, 1)
    
    def calculate_throw_speed(self, current_idx, current_timestamp):
        """
        Calcula si el saque es rápido (<5s) o lento (>=5s)
        Compara el timestamp con el evento anterior del mismo equipo
        """
        if pd.isna(current_timestamp):
            return 'Desconocido'
        
        # Convertir timestamp a datetime si es string
        if isinstance(current_timestamp, str):
            try:
                current_timestamp = pd.to_datetime(current_timestamp)
            except:
                return 'Desconocido'
        
        # Buscar el evento anterior en el mismo partido
        if current_idx > 0 and current_idx in self.df.index:
            current_match = self.df.loc[current_idx, 'Match ID']
            
            # Buscar hacia atrás hasta encontrar un evento del mismo partido
            for prev_idx in range(current_idx - 1, max(0, current_idx - 20), -1):
                if prev_idx in self.df.index:
                    prev_match = self.df.loc[prev_idx, 'Match ID']
                    prev_timestamp = self.df.loc[prev_idx, 'timeStamp']
                    
                    if prev_match == current_match and pd.notna(prev_timestamp):
                        # Convertir timestamp anterior
                        if isinstance(prev_timestamp, str):
                            try:
                                prev_timestamp = pd.to_datetime(prev_timestamp)
                            except:
                                continue
                        
                        # Calcular diferencia en segundos
                        time_diff = (current_timestamp - prev_timestamp).total_seconds()
                        
                        if time_diff < 5:
                            return 'Rápido'
                        else:
                            return 'Lento'
        
        return 'Desconocido'
    
    def draw_horizontal_bars(self, ax, x_start, y_start, width, height, value1, value2, label1, label2, color1, color2):
        """
        Dibuja dos barras horizontales apiladas
        x_start, y_start: posición inicial (0-1 en coordenadas del eje)
        width, height: tamaño de cada barra
        value1, value2: valores a mostrar (porcentajes)
        """
        # Barra 1
        bar1_width = (value1 / 100) * width
        rect1 = plt.Rectangle((x_start, y_start), bar1_width, height, 
                            facecolor=color1, edgecolor='white', linewidth=1,
                            transform=ax.transAxes, zorder=2)
        ax.add_patch(rect1)
        
        # Barra 2 (resto)
        rect2 = plt.Rectangle((x_start + bar1_width, y_start), width - bar1_width, height,
                            facecolor=color2, edgecolor='white', linewidth=1,
                            transform=ax.transAxes, zorder=2)
        ax.add_patch(rect2)
        
        # Etiquetas dentro de las barras
        if value1 > 15:  # Solo mostrar si hay espacio
            ax.text(x_start + bar1_width/2, y_start + height/2, f'{value1:.0f}%',
                ha='center', va='center', fontsize=6, color='white',
                fontweight='bold', transform=ax.transAxes)
        
        if value2 > 15:
            ax.text(x_start + bar1_width + (width - bar1_width)/2, y_start + height/2, f'{value2:.0f}%',
                ha='center', va='center', fontsize=6, color='white',
                fontweight='bold', transform=ax.transAxes)
    
    def draw_angle_distribution(self, ax, x_center, y_center, angles_list, distances_list, size=0.15, side='left'):
        """
        Dibuja la distribución de ángulos con fondos coloreados según porcentaje.
        - Muestra el % y la distancia media para cada tramo.
        - Los espacios entre líneas discontinuas se colorean según el porcentaje.
        """
        # Ajustar posición según el lado
        if side == 'left':
            x_center = x_center - 0.08  # Mover más a la izquierda
        elif side == 'right':
            x_center = x_center + 0.08  # Mover más a la derecha
        
        # Combinamos y filtramos para mantener la correspondencia entre ángulo y distancia
        valid_pairs = [(a, d) for a, d in zip(angles_list, distances_list) if a is not None and d is not None]

        if not valid_pairs:
            ax.text(x_center, y_center, 'Sin datos', ha='center', va='center',
                fontsize=7, color='#95a5a6', transform=ax.transAxes)
            return

        total_events = len(valid_pairs)
        
        tramos = [
            (144, 180, 'Muy adelante'), (108, 144, 'Adelante'),
            (72, 108, 'Lateral'), (36, 72, 'Atrás'), (0, 36, 'Muy atrás')
        ]
        line_length = size * 1.1
        
        # Dibujar línea vertical central
        ax.plot([x_center, x_center], [y_center - line_length / 1.5, y_center + line_length / 1.5], 
                color='#2d3436', linewidth=2, transform=ax.transAxes, zorder=3)
        
        # Dibujar líneas discontinuas de límites
        boundary_angles_deg = [144, 108, 72, 36]
        for angle_deg in boundary_angles_deg:
            if side == 'right': 
                angle_rad = np.radians(270 - angle_deg)
            else: 
                angle_rad = np.radians(angle_deg - 90)
            x_end = x_center + line_length * np.cos(angle_rad)
            y_end = y_center + line_length * np.sin(angle_rad)
            ax.plot([x_center, x_end], [y_center, y_end],
                    color='#636e72', linestyle=':', linewidth=1.2, transform=ax.transAxes, zorder=4)

        # Dibujar tramos con fondo coloreado
        for angle_min, angle_max, label in tramos:
            # Filtramos los pares que caen en este tramo
            pairs_in_tramo = [(a, d) for a, d in valid_pairs if (angle_max == 180 and angle_min <= a <= angle_max) or (angle_min <= a < angle_max)]
            
            count = len(pairs_in_tramo)
            percentage = (count / total_events * 100) if total_events > 0 else 0

            if count > 0:
                # Calcular color de fondo según porcentaje (verde=alto, rojo=bajo)
                ratio = percentage / 100.0
                r = int(255 * (1 - ratio))
                g = int(255 * ratio)
                bg_color = f'#{r:02x}{g:02x}00'
                
                # Calcular color de texto según porcentaje para mejor legibilidad
                if percentage >= 50: text_color = '#27ae60'
                elif percentage >= 30: text_color = '#2ecc71'
                elif percentage >= 15: text_color = '#f39c12'
                else: text_color = '#e74c3c'
                
                # Calculamos la distancia media del tramo
                distances_in_tramo = [d for a, d in pairs_in_tramo]
                avg_dist = np.mean(distances_in_tramo)
                
                # Dibujar el sector coloreado (fondo)
                angle_start_deg = angle_min
                angle_end_deg = angle_max if angle_max != 180 else 180
                
                # Crear wedge (sector circular) para el fondo
                from matplotlib.patches import Wedge
                if side == 'right':
                    theta1 = 270 - angle_end_deg
                    theta2 = 270 - angle_start_deg
                else:
                    theta1 = angle_start_deg - 90
                    theta2 = angle_end_deg - 90
                
                wedge = Wedge((x_center, y_center), line_length * 0.9, theta1, theta2,
                            facecolor=bg_color, alpha=0.3, transform=ax.transAxes, zorder=1)
                ax.add_patch(wedge)
                
                # Creamos el texto combinado
                display_text = f'{percentage:.0f}%\n{avg_dist:.1f}m'

                angle_text_deg = (angle_min + angle_max) / 2
                text_radius = line_length * 0.6
                if side == 'right': 
                    angle_text_rad = np.radians(270 - angle_text_deg)
                else: 
                    angle_text_rad = np.radians(angle_text_deg - 90)
                x_text = x_center + text_radius * np.cos(angle_text_rad)
                y_text = y_center + text_radius * np.sin(angle_text_rad)
                
                ax.text(x_text, y_text, display_text,
                        ha='center', va='center', fontsize=6, color=text_color,
                        fontweight='bold', transform=ax.transAxes, zorder=10,
                        path_effects=[patheffects.withStroke(linewidth=2.5, foreground='white')],
                        linespacing=0.9)
        
        ax.text(x_center, y_center + line_length/1.5 + 0.015, '180°', ha='center', va='bottom', fontsize=5, color='#2d3436', transform=ax.transAxes)
        ax.text(x_center, y_center - line_length/1.5 - 0.015, '0°', ha='center', va='top', fontsize=5, color='#2d3436', transform=ax.transAxes)
    
    def draw_zone_stats_per_row(self, ax, throw_ins_left, throw_ins_right, zone_number):
        """
        Dibuja las estadísticas de la zona con los nuevos ajustes de diseño.
        - Barras horizontales más gruesas.
        - Gráficos de ángulos más abajo y centrados en sus columnas.
        """
        from collections import Counter

        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        def draw_horizontal_bars_inline(ax, x_start, y_start, width, height, 
                                value1, value2, label1, label2, color1, color2):
            ratio = value1 / 100.0
            r = int(255 * (1 - ratio))
            g = int(255 * ratio)
            color_fondo = f'#{r:02x}{g:02x}00'
            
            rect_fondo = Rectangle((x_start, y_start), width, height, 
                                facecolor=color_fondo, alpha=0.7, 
                                transform=ax.transAxes, zorder=1)
            ax.add_patch(rect_fondo)
            
            width1 = width * (value1 / 100)
            width2 = width * (value2 / 100)
            
            rect1 = Rectangle((x_start, y_start), width1, height,
                            facecolor=color1, edgecolor='white', linewidth=1,
                            transform=ax.transAxes, zorder=2)
            ax.add_patch(rect1)
            
            rect2 = Rectangle((x_start + width1, y_start), width2, height,
                            facecolor=color2, edgecolor='white', linewidth=1,
                            transform=ax.transAxes, zorder=2)
            ax.add_patch(rect2)
            
            luminancia = 0.299 * r + 0.587 * g
            text_color = 'white' if luminancia < 128 else 'black'
            
            fontsize = 6
            if width1 > 0.05:
                ax.text(x_start + width1/2, y_start + height/2, 
                    f'{value1:.0f}%', ha='center', va='center',
                    fontsize=fontsize, color=text_color, fontweight='bold',
                    transform=ax.transAxes, zorder=3)
            
            if width2 > 0.05:
                ax.text(x_start + width1 + width2/2, y_start + height/2,
                    f'{value2:.0f}%', ha='center', va='center', 
                    fontsize=fontsize, color=text_color, fontweight='bold',
                    transform=ax.transAxes, zorder=3)
        
        zone_colors = {1: '#0984e3', 2: '#6c5ce7', 3: '#fd79a8'}
        zone_names = {1: 'Defensiva', 2: 'Media', 3: 'Ofensiva'}
        zone_color = zone_colors.get(zone_number, '#2d3436')
        zone_name = zone_names.get(zone_number, '')
        
        ax.text(0.5, 0.95, f'ZONA {zone_number} ({zone_name})', 
            ha='center', va='top', fontsize=10, fontweight='bold', 
            color=zone_color, transform=ax.transAxes)
        
        ax.plot([0.33, 0.33], [0, 0.85], color='#dfe6e9', lw=1.5, transform=ax.transAxes, zorder=1)
        ax.plot([0.67, 0.67], [0, 0.85], color='#dfe6e9', lw=1.5, transform=ax.transAxes, zorder=1)
        
        # --- IZQUIERDA ---
        total_left = len(throw_ins_left)
        angles_left = [t.get('angle') for t in throw_ins_left]
        distances_left = [t.get('distance') for t in throw_ins_left]
        perc_exito_left = (sum(1 for t in throw_ins_left if t.get('outcome') == 1) / total_left * 100) if total_left > 0 else 0
        perc_rapido_left = (sum(1 for t in throw_ins_left if t.get('speed') == 'Rápido') / total_left * 100) if total_left > 0 else 0
        
        y_pos = 0.85
        ax.text(0.165, y_pos, 'IZQUIERDA', ha='center', va='center', fontsize=8, fontweight='bold', color='#2d3436', transform=ax.transAxes)
        y_pos -= 0.10
        ax.text(0.165, y_pos, f'Total: {total_left}', ha='center', va='center', fontsize=7, color='#2d3436', fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.10
        ax.text(0.165, y_pos + 0.025, 'Éxito', ha='center', va='bottom', fontsize=6, color='#2d3436', transform=ax.transAxes)
        # CAMBIO: Barras más gruesas (height=0.045)
        draw_horizontal_bars_inline(ax, 0.05, y_pos - 0.015, 0.23, 0.045, perc_exito_left, 100 - perc_exito_left, 'Éxito', 'Fallo', '#00b894', '#d63031')
        
        y_pos -= 0.08
        ax.text(0.165, y_pos + 0.025, 'Velocidad', ha='center', va='bottom', fontsize=6, color='#2d3436', transform=ax.transAxes)
        # CAMBIO: Barras más gruesas (height=0.045)
        draw_horizontal_bars_inline(ax, 0.05, y_pos - 0.015, 0.23, 0.045, perc_rapido_left, 100 - perc_rapido_left, 'Rápido', 'Lento', '#0984e3', '#74b9ff')

        y_pos -= 0.08
        ax.text(0.165, y_pos + 0.03, 'Distribución Ángulos', ha='center', va='bottom', fontsize=6, color='#2d3436', transform=ax.transAxes)
        # CAMBIO: Más abajo (y_pos - 0.15) y más a la izquierda (x_center=0.15)
        self.draw_angle_distribution(ax, 0.15, y_pos - 0.15, angles_left, distances_left, size=0.18, side='left')
        
        # --- DERECHA ---
        total_right = len(throw_ins_right)
        angles_right = [t.get('angle') for t in throw_ins_right]
        distances_right = [t.get('distance') for t in throw_ins_right]
        perc_exito_right = (sum(1 for t in throw_ins_right if t.get('outcome') == 1) / total_right * 100) if total_right > 0 else 0
        perc_rapido_right = (sum(1 for t in throw_ins_right if t.get('speed') == 'Rápido') / total_right * 100) if total_right > 0 else 0
        
        y_pos = 0.85
        ax.text(0.835, y_pos, 'DERECHA', ha='center', va='center', fontsize=8, fontweight='bold', color='#2d3436', transform=ax.transAxes)
        y_pos -= 0.10
        ax.text(0.835, y_pos, f'Total: {total_right}', ha='center', va='center', fontsize=7, color='#2d3436', fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.10
        ax.text(0.835, y_pos + 0.025, 'Éxito', ha='center', va='bottom', fontsize=6, color='#2d3436', transform=ax.transAxes)
        # CAMBIO: Barras más gruesas (height=0.045)
        draw_horizontal_bars_inline(ax, 0.72, y_pos - 0.015, 0.23, 0.045, perc_exito_right, 100 - perc_exito_right, 'Éxito', 'Fallo', '#00b894', '#d63031')

        y_pos -= 0.08
        ax.text(0.835, y_pos + 0.025, 'Velocidad', ha='center', va='bottom', fontsize=6, color='#2d3436', transform=ax.transAxes)
        # CAMBIO: Barras más gruesas (height=0.045)
        draw_horizontal_bars_inline(ax, 0.72, y_pos - 0.015, 0.23, 0.045, perc_rapido_right, 100 - perc_rapido_right, 'Rápido', 'Lento', '#0984e3', '#74b9ff')
        
        y_pos -= 0.08
        ax.text(0.835, y_pos + 0.03, 'Distribución Ángulos', ha='center', va='bottom', fontsize=6, color='#2d3436', transform=ax.transAxes)
        # CAMBIO: Más abajo (y_pos - 0.15) y más a la derecha (x_center=0.85)
        self.draw_angle_distribution(ax, 0.85, y_pos - 0.15, angles_right, distances_right, size=0.18, side='right')
        
        # --- CENTRO (RESUMEN) ---
        # (Sin cambios en esta sección)
        total_combined = total_left + total_right
        exitosos_combined = sum(1 for t in throw_ins_left if t.get('outcome') == 1) + sum(1 for t in throw_ins_right if t.get('outcome') == 1)
        perc_exito_combined = (exitosos_combined / total_combined * 100) if total_combined > 0 else 0
        rapidos_combined = sum(1 for t in throw_ins_left if t.get('speed') == 'Rápido') + sum(1 for t in throw_ins_right if t.get('speed') == 'Rápido')
        perc_rapido_combined = (rapidos_combined / total_combined * 100) if total_combined > 0 else 0
        y_pos = 0.82
        ax.text(0.5, y_pos, 'RESUMEN ZONA', ha='center', va='center', fontsize=8, fontweight='bold', color='#1e3d59', transform=ax.transAxes)
        y_pos -= 0.12
        ax.text(0.5, y_pos, f'Total: {total_combined}', ha='center', va='center', fontsize=8, color='#1e3d59', fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.12
        ax.text(0.5, y_pos + 0.03, 'Éxito Total: ' + f'{perc_exito_combined:.0f}%', ha='center', va='bottom', fontsize=7, color='#1e3d59', fontweight='bold', transform=ax.transAxes)
        draw_horizontal_bars_inline(ax, 0.38, y_pos - 0.02, 0.24, 0.05, perc_exito_combined, 100 - perc_exito_combined, 'Éxito', 'Fallo', '#00b894', '#d63031')
        y_pos -= 0.12 # Hacemos más espacio
        ax.text(0.5, y_pos + 0.03, 'Velocidad Total: ' + f'{perc_rapido_combined:.0f}% Rápido', ha='center', va='bottom', fontsize=7, color='#1e3d59', fontweight='bold', transform=ax.transAxes)
        draw_horizontal_bars_inline(ax, 0.38, y_pos - 0.02, 0.24, 0.05, perc_rapido_combined, 100 - perc_rapido_combined, 'Rápido', 'Lento', '#0984e3', '#74b9ff')
        
        all_valid_angles = [a for a in angles_left + angles_right if a is not None]
        all_valid_distances = [d for d in distances_left + distances_right if d is not None]
        
        if all_valid_angles:
            tramos_map = {
                (144, 181): 'Muy Adelante', (108, 144): 'Adelante', (72, 108): 'Lateral',
                (36, 72): 'Atrás', (0, 36): 'Muy Atrás'
            }
            angle_categories = [next((name for r, name in tramos_map.items() if r[0] <= a < r[1]), None) for a in all_valid_angles]
            angle_categories = [cat for cat in angle_categories if cat is not None]
            if angle_categories:
                most_common_angle = Counter(angle_categories).most_common(1)[0]
                angle_text = f"{most_common_angle[0]} ({most_common_angle[1] / len(angle_categories) * 100:.0f}%)"
            else: angle_text = "N/A"
        else: angle_text = "N/A"

        avg_distance = np.mean(all_valid_distances) if all_valid_distances else None
        dist_text = f"{avg_distance:.1f}m" if avg_distance is not None else "N/A"
        
        y_pos -= 0.22
        summary_text = f"Ángulo Frecuente: {angle_text}\nDistancia Media: {dist_text}"
        ax.text(0.5, y_pos, summary_text, ha='center', va='center', fontsize=8, color='#34495e',
                bbox=dict(boxstyle='round,pad=0.5', fc='#ecf0f1', ec='#bdc3c7', lw=1.5),
                linespacing=1.3)
    
    def classify_throw_ins_by_zone_and_side(self, sequences):
        """
        Clasifica los saques de banda por zona y lado del campo
        """
        classification = {
            'Izquierda': {'Zona 1': [], 'Zona 2': [], 'Zona 3': []},
            'Derecha': {'Zona 1': [], 'Zona 2': [], 'Zona 3': []}
        }
        
        for seq in sequences:
            side = seq['side']
            zone = seq['zone_origin']
            
            if zone is not None:
                zone_key = f'Zona {zone}'
                if zone_key in classification[side]:
                    classification[side][zone_key].append(seq)
        
        return classification
    
    def draw_pitch_with_throw_ins(self, ax, throw_ins_list, zone_number, side):
        """
        Dibuja un campograma vertical con los saques de banda
        Estilo: fondo verde oscuro, líneas blancas
        COORDENADAS INTERCAMBIADAS: Y es X, X es Y
        """
        # Configurar el pitch vertical con el estilo correcto
        pitch = VerticalPitch(
            pitch_type='opta',
            pitch_color='#2d5a27',  # Verde oscuro
            line_color='white',      # Líneas blancas
            linewidth=2,
            label=False,
            tick=False
        )
        
        pitch.draw(ax=ax)
        
        # Dibujar las divisiones de zonas con líneas horizontales
        # Zona 1: 66.6-100, Zona 2: 33.3-66.6, Zona 3: 0-33.3
        zone_lines_x = [33.3, 66.6]
        for x_line in zone_lines_x:
            ax.plot([0, 100], [x_line, x_line], color='white',
                linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
        
        # Resaltar la zona actual con un rectángulo
        if zone_number == 1:  # Zona defensiva (campo propio): 0-33.3
            zone_rect = Rectangle((0, 0), 100, 33.3, 
                                facecolor='yellow', alpha=0.12, zorder=1)
        elif zone_number == 2:  # Zona media: 33.3-66.6
            zone_rect = Rectangle((0, 33.3), 100, 33.3,
                                facecolor='yellow', alpha=0.12, zorder=1)
        elif zone_number == 3:  # Zona ofensiva (campo rival): 66.6-100
            zone_rect = Rectangle((0, 66.6), 100, 33.4,
                                facecolor='yellow', alpha=0.12, zorder=1)

        ax.add_patch(zone_rect)
        
        # Dibujar los saques de banda
        for throw_in in throw_ins_list:
            x_start = throw_in.get('x', 0)
            y_start = throw_in.get('y', 0)
            x_end = throw_in.get('end_x', 0)
            y_end = throw_in.get('end_y', 0)
            outcome = throw_in.get('outcome', 0)
            
            # Validar coordenadas
            if pd.isna(x_start) or pd.isna(y_start):
                continue
            
            # Color según resultado
            if outcome == 1:
                color = '#00ff00'  # Verde brillante para exitosos
                alpha = 0.8
                size = 70
            else:
                color = '#ff0000'  # Rojo para fallidos
                alpha = 0.6
                size = 55
            
            # 🔥 INTERCAMBIO: Y es X, X es Y
            ax.scatter(y_start, x_start, s=size, color=color, 
                    alpha=alpha, edgecolor='white', linewidth=1.5, zorder=3)
            
            # Dibujar línea hacia el destino si existe
            if pd.notna(x_end) and pd.notna(y_end):
                # 🔥 INTERCAMBIO: Y es X, X es Y
                ax.annotate('', xy=(y_end, x_end), xytext=(y_start, x_start),
                        arrowprops=dict(arrowstyle='->', color=color, 
                                        lw=1.5, alpha=alpha-0.1, zorder=2))
        
        return ax
    
    def draw_zone_statistics_combined(self, ax, classification):
        """
        Dibuja estadísticas combinadas de ambos lados en las columnas centrales
        """
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Título principal
        ax.text(0.5, 1, 'ESTADÍSTICAS POR ZONA Y LADO', 
               ha='center', va='top', fontsize=11, 
               fontweight='bold', color='#1e3d59',
               transform=ax.transAxes)
        
        # Subtítulos de columnas
        ax.text(0.25, 0.95, 'IZQUIERDA', ha='center', va='top',
               fontsize=10, fontweight='bold', color='#2d3436',
               transform=ax.transAxes)
        
        ax.text(0.75, 0.95, 'DERECHA', ha='center', va='top',
               fontsize=10, fontweight='bold', color='#2d3436',
               transform=ax.transAxes)
        
        # Línea divisoria vertical
        ax.plot([0.5, 0.5], [0.05, 0.88], color='#2d3436', 
               linewidth=2, transform=ax.transAxes, zorder=3)
        
        # Estadísticas por zona (ahora en el orden correcto)
        y_start = 0.78
        zone_height = 0.22
        
        # Zona 3 arriba (ofensiva), Zona 1 abajo (defensiva)
        for i, (zone_name, zone_color, zone_desc) in enumerate([
            ('Zona 3', '#fd79a8', 'Ofensiva'),
            ('Zona 2', '#6c5ce7', 'Media'),
            ('Zona 1', '#0984e3', 'Defensiva')
        ]):
            y_pos = y_start - (i * zone_height)
            
            # Título de la zona (centrado)
            ax.text(0.5, y_pos + 0.02, f'{zone_name} ({zone_desc})', 
                   ha='center', va='top',
                   fontsize=9, fontweight='bold', color=zone_color,
                   transform=ax.transAxes)
            
            # Estadísticas IZQUIERDA
            throw_ins_izq = classification['Izquierda'][zone_name]
            total_izq = len(throw_ins_izq)
            exitosos_izq = sum(1 for t in throw_ins_izq if t.get('outcome', 0) == 1)
            porcentaje_izq = (exitosos_izq / total_izq * 100) if total_izq > 0 else 0
            
            ax.text(0.25, y_pos - 0.05, f'{total_izq} saques', 
                   ha='center', va='center', fontsize=8, color='#2d3436',
                   transform=ax.transAxes)
            ax.text(0.25, y_pos - 0.10, f'{porcentaje_izq:.1f}% éxito', 
                   ha='center', va='center', fontsize=8, 
                   color='#00b894' if porcentaje_izq >= 60 else '#636e72',
                   fontweight='bold', transform=ax.transAxes)
            
            # Estadísticas DERECHA
            throw_ins_der = classification['Derecha'][zone_name]
            total_der = len(throw_ins_der)
            exitosos_der = sum(1 for t in throw_ins_der if t.get('outcome', 0) == 1)
            porcentaje_der = (exitosos_der / total_der * 100) if total_der > 0 else 0
            
            ax.text(0.75, y_pos - 0.05, f'{total_der} saques', 
                   ha='center', va='center', fontsize=8, color='#2d3436',
                   transform=ax.transAxes)
            ax.text(0.75, y_pos - 0.10, f'{porcentaje_der:.1f}% éxito', 
                   ha='center', va='center', fontsize=8,
                   color='#00b894' if porcentaje_der >= 60 else '#636e72',
                   fontweight='bold', transform=ax.transAxes)
            
            # Línea separadora entre zonas
            if i < 2:
                ax.plot([0.05, 0.95], [y_pos - 0.15, y_pos - 0.15], 
                       color='#dfe6e9', linewidth=1, transform=ax.transAxes)
        
        # TOTALES GENERALES en la parte inferior
        y_total = 0.12
        
        ax.text(0.5, y_total + 0.03, 'TOTAL GENERAL', ha='center', va='center',
               fontsize=9, fontweight='bold', color='#1e3d59',
               transform=ax.transAxes)
        
        # Total izquierda
        all_izq = []
        for zone_name in ['Zona 1', 'Zona 2', 'Zona 3']:
            all_izq.extend(classification['Izquierda'][zone_name])
        total_all_izq = len(all_izq)
        exitosos_all_izq = sum(1 for t in all_izq if t.get('outcome', 0) == 1)
        porcentaje_all_izq = (exitosos_all_izq / total_all_izq * 100) if total_all_izq > 0 else 0
        
        ax.text(0.25, y_total - 0.03, f'{total_all_izq} | {porcentaje_all_izq:.1f}%',
               ha='center', va='center', fontsize=9, color='#2d3436',
               fontweight='bold', transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#dfe6e9', 
                        edgecolor='#2d3436', linewidth=1.5))
        
        # Total derecha
        all_der = []
        for zone_name in ['Zona 1', 'Zona 2', 'Zona 3']:
            all_der.extend(classification['Derecha'][zone_name])
        total_all_der = len(all_der)
        exitosos_all_der = sum(1 for t in all_der if t.get('outcome', 0) == 1)
        porcentaje_all_der = (exitosos_all_der / total_all_der * 100) if total_all_der > 0 else 0
        
        ax.text(0.75, y_total - 0.03, f'{total_all_der} | {porcentaje_all_der:.1f}%',
               ha='center', va='center', fontsize=9, color='#2d3436',
               fontweight='bold', transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#dfe6e9', 
                        edgecolor='#2d3436', linewidth=1.5))
    
    def load_background(self):
        """Carga la imagen de fondo si existe"""
        return plt.imread("assets/fondo_informes.png") if os.path.exists("assets/fondo_informes.png") else None

    def load_tactic_logo(self):
        """Carga el logo de Tactic"""
        return plt.imread("assets/tactic_logo.png") if os.path.exists("assets/tactic_logo.png") else None

    def load_team_logo(self, equipo, target_size=(80, 80)):
        """Carga el logo del equipo con búsqueda inteligente"""
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
            from difflib import SequenceMatcher
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
            except:
                pass
        
        return None
    
    def create_visualization(self, team_name):
        """
        Crea la visualización completa del análisis de saques de banda
        Formato A4 horizontal con 5 columnas y estadísticas por zona
        """
        # 1. Extracción de secuencias
        sequences = self.extract_throw_in_sequences(team_name)
        
        if not sequences:
            print("❌ No se encontraron saques de banda para analizar.")
            fig, ax = plt.subplots(figsize=(11.69, 8.27), facecolor='white')
            ax.text(0.5, 0.5, f'No se encontraron saques de banda\npara {team_name}', 
                ha='center', va='center', fontsize=18, color='red')
            ax.axis('off')
            return fig
        
        print(f"✅ Encontrados {len(sequences)} saques de banda")
        
        # 2. Clasificación por zona y lado
        classification = self.classify_throw_ins_by_zone_and_side(sequences)
        
        # 3. Configuración de la figura (A4 horizontal)
        fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
        
        # Fondo si existe
        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.15)
            ax_bg.axis('off')
        
        # Grid: 3 filas x 5 columnas (columnas 1 y 3 estrechas para separar)
        gs = fig.add_gridspec(3, 5, 
                            height_ratios=[1, 1, 1],
                            width_ratios=[1, 0.15, 2.5, 0.15, 1],
                            wspace=0.01, hspace=0.01,
                            left=0.04, right=0.98, top=0.88, bottom=0.05)
        
        # 4. Títulos y logos
        fig.suptitle('Saques de banda Ofensivos',
                    fontsize=16, fontweight='bold', color='#1e3d59', 
                    y=0.96, family='serif')
        
        # Logos
        if (tactic_logo := self.load_tactic_logo()) is not None:
            ax_logo1 = fig.add_axes([0.02, 0.90, 0.08, 0.08], anchor='NW', zorder=10)
            ax_logo1.imshow(tactic_logo)
            ax_logo1.axis('off')
        
        if (team_logo := self.load_team_logo(team_name)) is not None:
            ax_logo2 = fig.add_axes([0.90, 0.90, 0.08, 0.08], anchor='NE', zorder=10)
            ax_logo2.imshow(team_logo)
            ax_logo2.axis('off')
        
        # 5. Título central y etiquetas de lado (MÁS ARRIBA)
        fig.text(0.5, 0.92, 'ESTADÍSTICAS POR ZONA Y LADO', 
                fontsize=11, ha='center', fontweight='bold', 
                color='#1e3d59', family='sans-serif')
        
        fig.text(0.13, 0.90, 'Izquierda', fontsize=10, ha='center', 
                fontweight='bold', color='#2d3436', family='sans-serif')
        fig.text(0.87, 0.90, 'Derecha', fontsize=10, ha='center', 
                fontweight='bold', color='#2d3436', family='sans-serif')
        
        # 6. Dibujar campogramas en columnas 0 (izquierda) y 4 (derecha)
        # FILA 0 - ZONA 3 (ofensiva) - ARRIBA
        ax_izq_z3 = fig.add_subplot(gs[0, 0])
        self.draw_pitch_with_throw_ins(ax_izq_z3, 
                                    classification['Izquierda']['Zona 3'],
                                    3, 'Izquierda')

        ax_der_z3 = fig.add_subplot(gs[0, 4])
        self.draw_pitch_with_throw_ins(ax_der_z3, 
                                    classification['Derecha']['Zona 3'],
                                    3, 'Derecha')
        
        # FILA 1 - ZONA 2 (media)
        ax_izq_z2 = fig.add_subplot(gs[1, 0])
        self.draw_pitch_with_throw_ins(ax_izq_z2, 
                                    classification['Izquierda']['Zona 2'],
                                    2, 'Izquierda')
        
        ax_der_z2 = fig.add_subplot(gs[1, 4])
        self.draw_pitch_with_throw_ins(ax_der_z2, 
                                    classification['Derecha']['Zona 2'],
                                    2, 'Derecha')
        
        # FILA 2 - ZONA 1 (defensiva) - ABAJO
        ax_izq_z1 = fig.add_subplot(gs[2, 0])
        self.draw_pitch_with_throw_ins(ax_izq_z1, 
                                    classification['Izquierda']['Zona 1'],
                                    1, 'Izquierda')

        ax_der_z1 = fig.add_subplot(gs[2, 4])
        self.draw_pitch_with_throw_ins(ax_der_z1, 
                                    classification['Derecha']['Zona 1'],
                                    1, 'Derecha')
        
        # 7. Panel central - Estadísticas por zona (UNA FILA POR ZONA)
        # Zona 3 - ARRIBA
        ax_stats_z3 = fig.add_subplot(gs[0, 2])
        self.draw_zone_stats_per_row(ax_stats_z3, 
            classification['Izquierda']['Zona 3'],
            classification['Derecha']['Zona 3'], 3)

        # Zona 2 - MEDIO
        ax_stats_z2 = fig.add_subplot(gs[1, 2])
        self.draw_zone_stats_per_row(ax_stats_z2,
            classification['Izquierda']['Zona 2'],
            classification['Derecha']['Zona 2'], 2)

        # Zona 1 - ABAJO
        ax_stats_z1 = fig.add_subplot(gs[2, 2])
        self.draw_zone_stats_per_row(ax_stats_z1,
            classification['Izquierda']['Zona 1'],
            classification['Derecha']['Zona 1'], 1)
        
        # 8. Líneas verticales de separación (columnas 1 y 3)
        fig.add_artist(plt.Line2D([0.24, 0.24], [0.05, 0.88], 
                                transform=fig.transFigure, color='#2d3436', linewidth=2.5))
        fig.add_artist(plt.Line2D([0.76, 0.76], [0.05, 0.88], 
                                transform=fig.transFigure, color='#2d3436', linewidth=2.5))
        
        # 9. Leyenda de barras horizontales
        legend_elements = [
            Line2D([0], [0], color='#00b894', linewidth=8, label='% Éxito'),
            Line2D([0], [0], color='#d63031', linewidth=8, label='% Fallo'),
            
            # --- AÑADE ESTAS DOS LÍNEAS ---
            Line2D([0], [0], color='#0984e3', linewidth=8, label='% Rápido'),
            Line2D([0], [0], color='#74b9ff', linewidth=8, label='% Lento'),
            # --- FIN DE LAS LÍNEAS A AÑADIR ---
            
            mpatches.Patch(facecolor='#00ff00', alpha=0.5, label='Alto % (>50%) - Verde'),
            mpatches.Patch(facecolor='#ffff00', alpha=0.5, label='Medio % (30-50%) - Amarillo'),
            mpatches.Patch(facecolor='#ff0000', alpha=0.5, label='Bajo % (<30%) - Rojo')
        ]

        # Puede que necesites ajustar el número de columnas (ncol) para que quepa bien
        fig.legend(handles=legend_elements, loc='lower center', 
                ncol=4, frameon=True, fontsize=8, # <-- Aumenta ncol a 4 o 5 si es necesario
                bbox_to_anchor=(0.5, 0.002))
        
        return fig
    
    def print_summary(self, team_name):
        """Imprime un resumen del análisis"""
        sequences = self.extract_throw_in_sequences(team_name)
        
        print(f"\n{'='*60}")
        print(f"RESUMEN DE SAQUES DE BANDA OFENSIVOS")
        print(f"{'='*60}")
        print(f"Equipo: {team_name}")
        print(f"Total de saques de banda: {len(sequences)}")
        
        if sequences:
            classification = self.classify_throw_ins_by_zone_and_side(sequences)
            
            for side in ['Izquierda', 'Derecha']:
                print(f"\n{side}:")
                # Mostrar en orden: Zona 3 (ofensiva), Zona 2 (media), Zona 1 (defensiva)
                for zone in ['Zona 3', 'Zona 2', 'Zona 1']:
                    throw_ins = classification[side][zone]
                    total = len(throw_ins)
                    exitosos = sum(1 for t in throw_ins if t.get('outcome', 0) == 1)
                    porcentaje = (exitosos / total * 100) if total > 0 else 0
                    zone_desc = {'Zona 3': 'Ofensiva', 'Zona 2': 'Media', 'Zona 1': 'Defensiva'}[zone]
                    print(f"  {zone} ({zone_desc}): {total} saques ({porcentaje:.1f}% éxito)")


def seleccionar_equipo_interactivo():
    """Función para seleccionar un equipo de forma interactiva"""
    try:
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/open_play_events.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        
        if not equipos:
            print("No se encontraron equipos.")
            return None
        
        print("\n" + "="*60)
        print("SELECCIÓN DE EQUIPO")
        print("="*60)
        for i, equipo in enumerate(equipos, 1):
            print(f"{i:2d}. {equipo}")
        
        while True:
            try:
                idx = int(input(f"\nSelecciona un equipo (1-{len(equipos)}): ").strip()) - 1
                if 0 <= idx < len(equipos):
                    return equipos[idx]
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
        print("\n" + "="*60)
        print("ANÁLISIS DE SAQUES DE BANDA OFENSIVOS")
        print("="*60)
        
        equipo = seleccionar_equipo_interactivo()
        if equipo is None:
            print("No se pudo completar la selección.")
            return
        
        print(f"\nAnalizando saques de banda para {equipo}...")
        
        analyzer = AnalizadorSaquesBanda()
        
        if not analyzer.load_data(team_filter=equipo):
            return
        
        # Resumen
        analyzer.print_summary(equipo)
        
        # Visualización
        fig = analyzer.create_visualization(equipo)
        
        if fig:
            equipo_filename = re.sub(r'[\s/]', '_', equipo)
            output_path = f"saques_banda_ofensivos_{equipo_filename}.pdf"
            fig.savefig(output_path, bbox_inches='tight', 
                       pad_inches=0.1, facecolor='white', dpi=300)
            print(f"\n✅ Visualización guardada como: {output_path}")
            plt.show()
        else:
            print("❌ No se pudo generar la visualización")
    
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nINICIALIZANDO ANALIZADOR DE SAQUES DE BANDA")
    try:
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/open_play_events.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        print(f"✅ Sistema listo. Equipos disponibles: {len(equipos)}")
        if equipos:
            print("Para ejecutar el análisis, ejecute: main()")
        main()
    except Exception as e:
        print(f"❌ Error al inicializar: {e}")
        print("\nAsegúrate de que el archivo 'open_play_events.parquet' existe en:")
        print("  extraccion_opta/datos_opta_parquet/open_play_events.parquet")