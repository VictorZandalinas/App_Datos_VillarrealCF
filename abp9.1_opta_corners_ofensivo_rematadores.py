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
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from mplsoccer import VerticalPitch
from difflib import SequenceMatcher
from functools import lru_cache
from collections import Counter
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class ReporteRematadoresCorners:
    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/abp_events.parquet", team_filter=None):
        self.data_path = data_path
        self.team_filter = team_filter
        self.df = None
        self.remates_izquierda = pd.DataFrame()
        self.remates_derecha = pd.DataFrame()
        self.team_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/team_stats.parquet")
        self.player_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/player_stats.parquet")
        self.xg_events = pd.read_parquet("extraccion_opta/datos_opta_parquet/xg_events.parquet")
        
        self.load_data(team_filter)
        
        if team_filter:
            self.extract_remates_corners()

    def load_data(self, team_filter=None):
        """Carga los datos necesarios y convierte timestamps de forma robusta."""
        try:
            columns_needed = ['Match ID', 'periodId', 'Team ID', 'Team Name', 'Event Name', 'outcome', 
                'timeMin', 'timeSec', 'x', 'y', 'Pass End X', 'Pass End Y', 
                'playerName', 'playerId', 'Corner taken', 
                'Throw in', 'Free kick taken', 'timeStamp']
            
            self.df = pd.read_parquet(self.data_path, columns=columns_needed)
            
            # --- CAMBIO CLAVE: Hacemos la conversión de fecha más flexible ---
            # Dejamos que pandas infiera el formato y convertimos los errores en NaT (Not a Time)
            self.df['timeStamp'] = pd.to_datetime(self.df['timeStamp'].str.replace('Z', '', regex=False), errors='coerce')
            # Eliminamos las filas donde la conversión de fecha falló, para asegurar la integridad
            self.df.dropna(subset=['timeStamp'], inplace=True)

            self.df = self.df.drop_duplicates(subset=['Match ID', 'timeMin', 'timeSec', 'Event Name', 'playerName'], keep='first')
            self.df = self.df[self.df['Event Name'] != 'Deleted event']
            
            if team_filter:
                team_matches = self.team_stats[self.team_stats['Team Name'] == team_filter]['Match ID'].unique()
                self.df = self.df[(self.df['Match ID'].isin(team_matches)) & (self.df['Team Name'] == team_filter)]

            
            try:
                xg_data = self.xg_events[['timeStamp', 'playerId', 'Match ID', 'qualifier 321']].copy()
                xg_data = xg_data.rename(columns={'qualifier 321': 'xg_value'})
                xg_data['xg_value'] = pd.to_numeric(xg_data['xg_value'], errors='coerce')
                
                # --- CAMBIO CLAVE (aplicado también aquí por consistencia) ---
                xg_data['timeStamp'] = pd.to_datetime(xg_data['timeStamp'].astype(str).str.replace('Z', '', regex=False), errors='coerce')
                
                # El merge ahora funcionará porque ambas columnas 'timeStamp' son del mismo tipo (datetime)
                self.df = pd.merge(self.df, xg_data, on=['timeStamp', 'playerId', 'Match ID'], how='left')
                print(f"✅ Merge xG exitoso: {self.df['xg_value'].notna().sum()} eventos con xG")
            except Exception as e:
                print(f"⚠️ Error en merge xG: {e}")
                self.df['xg_value'] = None
            
            print(f"✅ Datos filtrados cargados: {len(self.df)} eventos")
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")
            import traceback
            traceback.print_exc()
    
    def get_player_shirt_number_by_name(self, player_name):
        """Obtiene el dorsal del jugador por su nombre en el partido."""
        if pd.isna(player_name):
            return None
        
        # Busca en la tabla de estadísticas de jugadores
        player_info = self.player_stats[self.player_stats['Match Name'] == player_name]
        if not player_info.empty:
            shirt_number = player_info['Shirt Number'].iloc[0]
            # Devuelve el número como string si no es nulo
            return str(int(shirt_number)) if pd.notna(shirt_number) else None
        return None

    def extract_remates_corners(self):
        """
        Extrae remates de corners para ambos lados usando una estructura de análisis por partido.
        """
        if self.df is None:
            print("❌ No hay datos cargados")
            return

        print("🔍 Extrayendo remates de corners con la nueva estructura por partido...")
        
        # --- ESTRUCTURA PRINCIPAL: Ordenar datos una sola vez ---
        df_sorted = self.df.sort_values(['Match ID', 'timeStamp']).reset_index(drop=True)

        remates_izq_list = []
        remates_der_list = []

        # --- BUCLE PRINCIPAL: Procesar por partido (Match ID) ---
        for match_id in df_sorted['Match ID'].unique():
            # Aislar todos los eventos del partido actual para un análisis limpio
            match_events = df_sorted[df_sorted['Match ID'] == match_id].reset_index(drop=True)
            
            # --- IDENTIFICAR CÓRNERS: Buscamos todos los saques de esquina del partido ---
            # Filtramos por el evento 'Corner taken' y las coordenadas que nos interesan.
            # Izquierda (y < 0.5) y Derecha (y > 99)
            all_corners_in_match = match_events[
                (match_events['Corner taken'] == 'Sí') &
                (match_events['x'].notna()) & 
                (match_events['y'].notna()) &
                ((match_events['y'] > 0.5) | (match_events['y'] < 99)) # <-- Filtro para ambos lados
            ]

            # --- ANÁLISIS DE SECUENCIA: Iteramos sobre cada córner encontrado ---
            for corner_idx, corner in all_corners_in_match.iterrows():
                # Llamamos a la función que analiza la ventana de 15 segundos post-córner
                result_data = self.analyze_corner_sequence(match_events, corner_idx, corner)
                
                if result_data:
                    # --- ASIGNAR A LA LISTA CORRECTA ---
                    # Basado en la coordenada 'y' del saque de esquina original.
                    # En coordenadas Opta: y < 50 es el lado IZQUIERDO, y > 50 es el DERECHO.
                    
                    if corner['y'] < 50:  # Córner sacado desde el lado IZQUIERDO (y es 0, 0.1, etc.)
                        remates_izq_list.append(result_data)
                    else:                 # Córner sacado desde el lado DERECHO (y es 100, 99.9, etc.)
                        remates_der_list.append(result_data)

        # --- CREACIÓN DE DATAFRAMES FINALES ---
        # Convertimos las listas a DataFrames, eliminando duplicados si los hubiera.
        if remates_izq_list:
            self.remates_izquierda = pd.DataFrame(remates_izq_list).drop_duplicates()
        else:
            self.remates_izquierda = pd.DataFrame()

        if remates_der_list:
            self.remates_derecha = pd.DataFrame(remates_der_list).drop_duplicates()
        else:
            self.remates_derecha = pd.DataFrame()
        
        print(f"✅ Remates lado izquierdo encontrados: {len(self.remates_izquierda)}")
        print(f"✅ Remates lado derecho encontrados: {len(self.remates_derecha)}")

    def analyze_corner_sequence(self, match_events, corner_idx, corner_pass):
        """
        Analiza la secuencia post-córner durante una ventana de 15 segundos
        desde el momento del saque, usando la columna timeStamp.
        """
        events_found = {'Goal': None, 'Post': None, 'Attempt Saved': None, 'Miss': None, 'Aerial': None}
        result_event_idx = None
        is_second_play = False
        first_pass_end_y = None
        second_pass_end_y = None
        pass_count = 0
        
        corner_timestamp = corner_pass['timeStamp']

        for next_idx in range(corner_idx + 1, len(match_events)):
            next_event = match_events.iloc[next_idx]
            
            time_elapsed = (next_event['timeStamp'] - corner_timestamp).total_seconds()
            
            if time_elapsed > 15:
                break
            
            if next_event['Event Name'] in ['Corner Awarded', 'Foul', 'Offside', 'End Period', 'Out']:
                break
            
            event_name = next_event['Event Name']

            if event_name == 'Pass' and next_event['Team ID'] == corner_pass['Team ID']:
                pass_count += 1
                if pass_count == 1:
                    first_pass_end_y = float(next_event.get('Pass End Y', 0))
                elif pass_count == 2:
                    second_pass_end_y = float(next_event.get('Pass End Y', 0))
            
            if next_event['Team ID'] != corner_pass['Team ID'] and event_name not in ['Ball Recovery', 'Interception']:
                if time_elapsed > 3:
                    break

            if event_name in events_found and events_found[event_name] is None and next_event['Team ID'] == corner_pass['Team ID']:
                events_found[event_name] = next_event
                result_event_idx = next_idx
                if event_name == 'Goal':
                    break
        
        final_event = None
        if events_found['Goal'] is not None: final_event = events_found['Goal']
        elif events_found['Post'] is not None: final_event = events_found['Post']
        elif events_found['Attempt Saved'] is not None: final_event = events_found['Attempt Saved']
        elif events_found['Miss'] is not None: final_event = events_found['Miss']
        elif events_found['Aerial'] is not None: final_event = events_found['Aerial']

        if final_event is None:
            return None

        # --- INICIO DE LA CORRECCIÓN ---
        # Comprobamos si en la secuencia hubo una disputa aérea ganada,
        # independientemente de cuál fue el evento final (Gol, Miss, etc.).
        # Esto asegura que un remate que viene de un duelo se catalogue correctamente.
        secuencia_incluye_aerea_ganada = (events_found['Aerial'] is not None and events_found['Aerial'].get('outcome', 0) == 1)
        # --- FIN DE LA CORRECCIÓN ---

        is_second_play = False
        corner_time_sec = corner_pass['timeMin'] * 60 + corner_pass['timeSec']
        remate_time_sec = final_event['timeMin'] * 60 + final_event['timeSec']
        tiempo_transcurrido = remate_time_sec - corner_time_sec
        
        if tiempo_transcurrido >= 5:
            is_second_play = True

        es_primera = not is_second_play
        
        return {
            'Match ID': final_event['Match ID'], 'Team ID': final_event['Team ID'],
            'Team Name': final_event['Team Name'], 'playerName': final_event['playerName'],
            'playerId': final_event['playerId'], 'Event Name': final_event['Event Name'],
            'outcome': final_event['outcome'], 'x': final_event['x'], 'y': final_event['y'],
            'timeMin': final_event['timeMin'], 'timeSec': final_event['timeSec'],
            'timeStamp': final_event['timeStamp'], 
            'lado': 'izquierda' if corner_pass['y'] < 50 else 'derecha',
            'es_primera': es_primera, 'lanzador_corner': corner_pass['playerName'],
            'xg_value': final_event.get('xg_value', 0),
            # AHORA ESTE VALOR REFLEJA LA REALIDAD DE LA SECUENCIA COMPLETA
            'es_disputa_aerea_ganada': secuencia_incluye_aerea_ganada,
            'es_remate_cabeza': final_event.get('Head', 'No') == 'Sí',
            'first_pass_end_y': first_pass_end_y,
            'second_pass_end_y': second_pass_end_y
        }

    def get_top_rematadores(self, lado, n=8):
        """Obtiene el ranking de rematadores, incluyendo el conteo de segundas jugadas."""
        df_remates = self.remates_izquierda if lado == 'izquierda' else self.remates_derecha
        if df_remates.empty:
            return []

        jugadores_stats = {}
        for _, remate in df_remates.iterrows():
            player = remate['playerName']
            if pd.isna(player):
                continue

            if player not in jugadores_stats:
                jugadores_stats[player] = {
                    'remates': 0, 'xg': 0, 'goles': 0, 'tiros_a_puerta': 0,
                    'disputas_aereas_ganadas': 0, 'remates_cabeza': 0,
                    'remates_2a_jugada': 0  # <--- NUEVA MÉTRICA INICIALIZADA
                }

            stats = jugadores_stats[player]
            stats['remates'] += 1
            xg_val = remate.get('xg_value')
            stats['xg'] += float(0 if pd.isna(xg_val) else xg_val)
            
            if remate['Event Name'] == 'Goal':
                stats['goles'] += 1
            if remate['Event Name'] in ['Goal', 'Attempt Saved']:
                stats['tiros_a_puerta'] += 1
            if remate.get('es_disputa_aerea_ganada', False):
                stats['disputas_aereas_ganadas'] += 1
            if remate.get('es_remate_cabeza', False):
                stats['remates_cabeza'] += 1
            
            # --- LÓGICA PARA CONTAR LA NUEVA MÉTRICA ---
            if not remate.get('es_primera', True):
                stats['remates_2a_jugada'] += 1

        sorted_players = sorted(
            jugadores_stats.items(),
            key=lambda item: (item[1]['remates'], item[1]['xg']),
            reverse=True
        )
        return sorted_players[:n]

    def get_player_color(self, player_name, top_players):
        """
        Asigna un color único y distinto a cada jugador, sin usar verde,
        con una paleta expandida para más de 10 jugadores.
        """
        # --- PALETA DE COLORES AMPLIADA (16 colores) ---
        colors = [
            '#d62728',  # Rojo ladrillo
            '#1f77b4',  # Azul acero
            '#ff7f0e',  # Naranja seguridad
            '#9467bd',  # Púrpura apagado
            '#17becf',  # Cian
            '#e377c2',  # Rosa
            '#8c564b',  # Marrón
            '#7f7f7f',  # Gris medio
            '#bcbd22',  # Oliva (no es verde césped)
            '#d4af37',  # Dorado
            '#a50f15',  # Carmesí
            '#6baed6',  # Azul claro
            '#e6550d',  # Naranja oscuro
            '#c51b7d',  # Magenta
            '#008b8b',  # Cian oscuro / Teal
            '#b39eb5'   # Lavanda
        ]
        
        for i, (name, _) in enumerate(top_players):
            if name == player_name:
                # El módulo (%) asegura que si hay más jugadores que colores,
                # los colores se reciclen de forma segura sin que el programa falle.
                return colors[i % len(colors)]
        
        # Color por defecto para jugadores que no están en el top
        return '#95a5a6'

    def get_outcome_marker(self, result_type):
        """Marcadores por tipo de resultado"""
        marker_map = {
            'Sin remate': 'v',
            'Attempt Saved': 's',
            'Miss': 'X',
            'Post': '^',
            'Goal': 'o',
            'Otro contacto': 'D'
        }
        return marker_map.get(result_type, 'o')

    def get_outcome_color(self, result_type):
        """Colores por tipo de resultado"""
        color_map = {
            'Sin remate': '#95A5A6',
            'Attempt Saved': '#3498DB',
            'Miss': '#E74C3C',
            'Post': '#FF6B35',
            'Goal': '#F1C40F',
            'Otro contacto': '#9B59B6'
        }
        return color_map.get(result_type, '#95A5A6')

    def create_campograma_rematadores(self, ax, lado, top_players_global):
        """
        Crea un campograma SIN TÍTULO sobre el campo, con ajustes de visualización.
        """
        pitch = VerticalPitch(pitch_type='opta', half=True, pitch_color='#2d5016', line_color='white', linewidth=2)
        pitch.draw(ax=ax)
        
        df_remates = self.remates_izquierda if lado == 'izquierda' else self.remates_derecha
        
        if df_remates.empty:
            ax.text(34, 52.5, 'Sin datos', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
            return

        df_remates['xg_value'] = df_remates['xg_value'].fillna(0)
        
        non_goals = df_remates[df_remates['Event Name'] != 'Goal']
        aerials = non_goals[non_goals['es_disputa_aerea_ganada'] == True]
        shots = non_goals[non_goals['es_disputa_aerea_ganada'] == False]

        # --- INICIO DE LAS CORRECCIONES ---

        if not shots.empty:
            # CAMBIO 1: Reducimos el multiplicador de xG de 1500 a 800
            sizes = 50 + (shots['xg_value'] * 600) 
            colors = [self.get_player_color(name, top_players_global) for name in shots['playerName']]
            # CAMBIO 2: Reducimos el grosor del contorno de 1.5 a 0.5
            pitch.scatter(x=shots.x, y=shots.y, ax=ax, s=sizes, c=colors, marker='o', edgecolor='white', linewidth=0.5, alpha=0.8, zorder=10)

        if not aerials.empty:
            # CAMBIO 1: Reducimos el multiplicador de xG de 1500 a 800
            sizes = 50 + (aerials['xg_value'] * 800)
            colors = [self.get_player_color(name, top_players_global) for name in aerials['playerName']]
            # CAMBIO 2: Reducimos el grosor del contorno de 1.5 a 0.5
            pitch.scatter(x=aerials.x, y=aerials.y, ax=ax, s=sizes, c=colors, marker='s', edgecolor='white', linewidth=0.5, alpha=0.8, zorder=10)

        goles = df_remates[df_remates['Event Name'] == 'Goal']
        if not goles.empty:
            # CAMBIO 1: Reducimos el multiplicador de xG de 1500 a 800
            sizes = 100 + (goles['xg_value'] * 800)
            colors = [self.get_player_color(name, top_players_global) for name in goles['playerName']]
            # Mantenemos el contorno del gol más grueso para que destaque
            pitch.scatter(x=goles.x, y=goles.y, ax=ax, s=sizes, c=colors,
                        marker='H', edgecolor='yellow', linewidth=2, alpha=1, zorder=12)

        # --- FIN DE LAS CORRECCIONES ---

        second_plays = df_remates[df_remates['es_primera'] == False]
        for _, row in second_plays.iterrows():
            ax.text(row['y'], row['x'], '2',
                    fontsize=5, fontweight='bold', color='white',
                    ha='center', va='center', zorder=13,
                    path_effects=[patheffects.withStroke(linewidth=1.5, foreground='black')])
        
        if (ball := self.load_ball_image()) is not None:
            ball_box = OffsetImage(ball, zoom=0.06)
            # Lógica de coordenadas corregida
            ball_coords = (0, 100) if lado == 'izquierda' else (100, 100)
            ab_ball = AnnotationBbox(ball_box, ball_coords, frameon=False, zorder=10, xycoords='data')
            ax.add_artist(ab_ball)

    def create_leyenda_completa(self, fig, top_players_global):
        """Crea una leyenda completa con el nuevo símbolo de gol y segunda jugada."""
        ax_legend = fig.add_axes([0.20, 0.465, 0.60, 0.06]) # Ligeramente más ancha
        ax_legend.axis('off')

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Remate', markerfacecolor='#aaa', markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='s', color='w', label='Disputa Aérea Ganada', markerfacecolor='#aaa', markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='H', color='w', label='Gol', markerfacecolor='#F1C40F', markersize=12, markeredgecolor='k'),
            # --- NUEVO ELEMENTO PARA LA LEYENDA ---
            Patch(facecolor='none', edgecolor='none', label='2 = Segunda Jugada')
        ]

        legend = ax_legend.legend(handles=legend_elements, loc='center', ncol=4, # Ahora son 4 columnas
                                fontsize=10, frameon=True, fancybox=True, shadow=True,
                                columnspacing=2.0, handletextpad=0.5)
        
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(0.9)
        frame.set_edgecolor('#1e3d59')

    def create_ranking_rematadores(self, ax, lado, top_players_global):
        """
        Crea un ranking que ahora dibuja el DORSAL AL LADO de la foto.
        """
        top_players = self.get_top_rematadores(lado, n=6)
        photos_data = self.load_player_photos()
        ax.set_facecolor('#f0f2f5')
        ax.axis('off')

        if not top_players:
            ax.text(0.5, 0.5, f'RANKING REMATADORES\nCÓRNER {lado.upper()}\n\n(Sin datos)', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12, fontweight='bold', color='#333')
            return

        titulo_lado = 'IZQUIERDO' if lado == 'izquierda' else 'DERECHO'
        ax.text(0.5, 0.95, f'RANKING REMATADORES - CÓRNER {titulo_lado}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, fontweight='bold', color='#1e3d59')
        
        headers = ['Jugador', 'Rem', 'xG', 'Goles', 'T.P.', 'Aér.G', 'Cab.', '2aJ']
        # Reajustamos posiciones para hacer espacio al dorsal
        col_positions = [0.23, 0.55, 0.62, 0.69, 0.76, 0.83, 0.90, 0.97]
        header_y = 0.82
        
        ax.text(0.05, header_y, '#', ha='center', va='center', transform=ax.transAxes, fontweight='bold', fontsize=9, color='#333')
        for i, header in enumerate(headers):
            ax.text(col_positions[i], header_y, header, ha='center', va='center', transform=ax.transAxes, fontweight='bold', fontsize=9, color='#333')
        ax.axhline(y=header_y - 0.05, xmin=0.02, xmax=0.98, color='#ccc', linewidth=1.5)

        start_y, spacing = 0.70, 0.125
        for i, (player_name, stats) in enumerate(top_players):
            y_pos = start_y - (i * spacing)
            
            ax.text(0.05, y_pos, str(i+1), ha='center', va='center', transform=ax.transAxes, fontweight='bold', fontsize=11, color='#555')
            color = self.get_player_color(player_name, top_players_global)
            circle = plt.Circle((0.12, y_pos), 0.02, color=color, transform=ax.transAxes, ec='black', lw=0.5)
            ax.add_patch(circle)

            line1, line2 = self.format_player_name_multiline(player_name, max_chars_per_line=12)
            if line2 is None:
                ax.text(col_positions[0], y_pos, line1, ha='center', va='center', transform=ax.transAxes, fontweight='bold', fontsize=9, color='#2c3e50')
            else:
                ax.text(col_positions[0], y_pos + 0.015, line1, ha='center', va='center', transform=ax.transAxes, fontweight='bold', fontsize=8, color='#2c3e50')
                ax.text(col_positions[0], y_pos - 0.015, line2, ha='center', va='center', transform=ax.transAxes, fontweight='bold', fontsize=8, color='#2c3e50')
            
            # 1. Obtenemos la foto (ahora sin dorsal)
            player_photo = self.create_circular_player_photo(player_name, photos_data)
            if player_photo is not None:
                photo_ax = ax.inset_axes([0.30, y_pos - 0.057, 0.15, 0.15]) # Mantenemos el tamaño
                photo_ax.imshow(player_photo)
                photo_ax.axis('off')

            # --- INICIO DE LA NUEVA LÓGICA PARA DIBUJAR EL DORSAL ---
            # 2. Obtenemos el dorsal
            dorsal = self.get_player_shirt_number_by_name(player_name)
            if dorsal:
                # Posición a la derecha de la foto
                dorsal_x_pos = 0.47
                # 3. Dibujamos el círculo de fondo
                dorsal_bg = patches.Circle((dorsal_x_pos, y_pos), 0.025, color='black', alpha=0.8, transform=ax.transAxes)
                ax.add_patch(dorsal_bg)
                # 4. Dibujamos el número encima
                ax.text(dorsal_x_pos, y_pos, dorsal, color='white', ha='center', va='center', 
                        fontweight='bold', fontsize=9, transform=ax.transAxes)
            # --- FIN DE LA NUEVA LÓGICA ---

            data_row = [stats.get(k, 0) for k in ['remates', 'xg', 'goles', 'tiros_a_puerta', 'disputas_aereas_ganadas', 'remates_cabeza', 'remates_2a_jugada']]
            data_row[1] = f"{data_row[1]:.2f}"
            
            for j, data in enumerate(data_row):
                ax.text(col_positions[j+1], y_pos, data, ha='center', va='center', transform=ax.transAxes, fontweight='normal', fontsize=10, color='#333')

            if i < len(top_players) - 1:
                ax.axhline(y=y_pos - 0.06, xmin=0.02, xmax=0.98, color='#ddd', linewidth=1, linestyle='--')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def create_leyenda_completa(self, fig, top_players_global):
        """Crea una leyenda completa con el nuevo símbolo de gol y segunda jugada."""
        ax_legend = fig.add_axes([0.20, 0.465, 0.60, 0.06])
        ax_legend.axis('off')

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Remate', markerfacecolor='#aaa', markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='s', color='w', label='Disputa Aérea Ganada', markerfacecolor='#aaa', markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='H', color='w', label='Gol', markerfacecolor='#F1C40F', markersize=12, markeredgecolor='k'),
            # --- ELEMENTO AÑADIDO A LA LEYENDA ---
            Patch(facecolor='none', edgecolor='none', label='2 = Segunda Jugada')
        ]

        legend = ax_legend.legend(handles=legend_elements, loc='center', ncol=4,
                                fontsize=10, frameon=True, fancybox=True, shadow=True,
                                columnspacing=2.0, handletextpad=0.5)
        
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(0.9)
        frame.set_edgecolor('#1e3d59')
    
    def create_circular_player_photo(self, player_name, photos_data, size=(120, 120)):
        """Crea la foto circular del jugador, SIN el dorsal."""
        try:
            player_photo = self._process_player_photo(player_name, photos_data)
            if player_photo is None: return None
            if player_photo.max() <= 1: player_photo = (player_photo * 255).astype(np.uint8)
            
            height, width = size
            pil_photo = Image.fromarray(player_photo).convert('RGBA')
            pil_photo = pil_photo.resize((width, height), Image.Resampling.LANCZOS)
            
            base_img = Image.new('RGBA', (width, height), (255, 255, 255, 255))
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse([0, 0, width, height], fill=255)
            
            circular_photo = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            circular_photo.paste(pil_photo, (0, 0))
            circular_photo.putalpha(mask)
            base_img.paste(circular_photo, (0, 0), circular_photo)
            
            # El bloque que dibujaba el dorsal ha sido eliminado de esta función.
            
            return np.array(base_img) / 255.0
        except Exception as e:
            print(f"⚠️ Error creando foto circular para {player_name}: {e}")
            return None

    def _process_player_photo(self, player_name, photos_data):
        """Procesa la foto eliminando el fondo de forma avanzada."""
        match = self.match_player_name(player_name, photos_data)
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
                    if y < 0 or y >= height or x < 0 or x >= width: return False
                    return (data[y, x, 0] >= threshold and data[y, x, 1] >= threshold and data[y, x, 2] >= threshold)
                for start_y, start_x in start_points:
                    if visited[start_y, start_x] or not is_background_color(start_y, start_x): continue
                    stack = [(start_y, start_x)]
                    while stack:
                        y, x = stack.pop()
                        if (y < 0 or y >= height or x < 0 or x >= width or visited[y, x] or not is_background_color(y, x)): continue
                        visited[y, x] = True
                        background_mask[y, x] = True
                        stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
                return background_mask

            border_points = list(set([(0, x) for x in range(width)] + [(height-1, x) for x in range(width)] + [(y, 0) for y in range(height)] + [(y, width-1) for y in range(height)]))
            background_mask = flood_fill_iterative(border_points, threshold=230)
            
            alpha = np.where(background_mask, 0, 255).astype(np.uint8)
            alpha_suavizado = gaussian_filter(alpha, sigma=1.5)
            data[:, :, 3] = alpha_suavizado
            
            return data
            
        except Exception as e:
            print(f"⚠️ Error procesando foto de {player_name}: {e}")
            return None

    def match_player_name(self, player_name, photos_data):
        """
        Función de matching de nombres definitiva, más flexible y robusta.
        """
        def normalize(name):
            name = name.lower().strip()
            replacements = {'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u', 'ñ':'n', 'ü':'u'}
            for old, new in replacements.items():
                name = name.replace(old, new)
            return re.sub(r'[^\w\s]', '', name)

        if pd.isna(player_name): return None
        
        norm_player_name = normalize(player_name)
        best_match = None
        highest_score = 0.65  # Umbral mínimo de confianza

        for photo in photos_data:
            # Importante: Comprueba tanto 'name' como 'player_name' en el JSON
            photo_name = photo.get('name') or photo.get('player_name')
            if not photo_name: continue
            
            norm_photo_name = normalize(photo_name)
            
            # Estrategia 1: Coincidencia de apellido y similitud del nombre
            player_words = norm_player_name.split()
            photo_words = norm_photo_name.split()
            score = 0
            
            if player_words[-1] == photo_words[-1]: # Apellido coincide
                score = 0.8
                if player_words[0] == photo_words[0]: # Nombre coincide
                    score = 1.0
                elif player_words[0][0] == photo_words[0][0]: # Inicial coincide
                    score = 0.9

            # Estrategia 2: Similitud general si no hay coincidencia de apellido
            if score < 0.8:
                score = SequenceMatcher(None, norm_player_name, norm_photo_name).ratio()

            if score > highest_score:
                highest_score = score
                best_match = photo
        
        return best_match

    def format_player_name_multiline(self, player_name, max_chars_per_line=11):
        """Función de formato de nombres avanzada (copiada de abp2.1)."""
        words = player_name.split()
        if len(words) == 1:
            if len(player_name) > max_chars_per_line:
                mid = len(player_name) // 2
                return player_name[:mid], player_name[mid:]
            return player_name, None
        line1, line2 = words[0], ' '.join(words[1:])
        if len(line1) > max_chars_per_line: line1 = line1[:max_chars_per_line-3] + '...'
        if len(line2) > max_chars_per_line: line2 = line2[:max_chars_per_line-3] + '...'
        return line1, line2

    def load_player_photos(self):
        """Carga fotos desde JSON"""
        try:
            with open('assets/jugadores_optimizados.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ No se encontró el archivo jugadores_optimizados.json")
            return []

    def load_team_logo(self, team_name, target_size=(80, 80)):
        """Carga logo del equipo"""
        try:
            team_row = self.team_stats[self.team_stats['Team Name'] == team_name]
            if not team_row.empty:
                logo_base64 = team_row.iloc[0].get('Team Logo', '')
                if logo_base64 and isinstance(logo_base64, str):
                    img_data = base64.b64decode(logo_base64)
                    img = Image.open(BytesIO(img_data))
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    return np.array(img) / 255.0
        except:
            pass
        
        possible_names = [
            team_name, team_name.replace(' ', '_'), team_name.replace(' ', ''),
            team_name.lower(), team_name.lower().replace(' ', '_')
        ]
        
        for name in possible_names:
            logo_path = f"assets/escudos/{name}.png"
            if os.path.exists(logo_path):
                try:
                    img = Image.open(logo_path)
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    return np.array(img) / 255.0
                except:
                    pass
        
        return None

    def load_ball_image(self):
        if os.path.exists("assets/balon.png"):
            return plt.imread("assets/balon.png")
        return None

    def load_background(self):
        if os.path.exists("assets/fondo_informes.png"):
            return plt.imread("assets/fondo_informes.png")
        return None

    def create_leyenda_conjunta(self, fig, gs):
        """Crea leyenda"""
        ax_legend = fig.add_axes([0.10, 0.47, 0.80, 0.05])
        ax_legend.axis('off')
        
        result_types = ['Attempt Saved', 'Miss', 'Post', 'Goal']
        
        from matplotlib.lines import Line2D
        legend_elements = []
        
        for result_type in result_types:
            if result_type == 'Goal':
                legend_elements.append(Line2D([0], [0], 
                                            marker='H', 
                                            color='w', 
                                            markerfacecolor=self.get_outcome_color(result_type),
                                            markeredgecolor='#E67E22',
                                            markeredgewidth=2,
                                            markersize=10,
                                            label='Gol (Escudo + Dorsal)'))
            else:
                legend_elements.append(Line2D([0], [0], 
                                            marker=self.get_outcome_marker(result_type), 
                                            color='w', 
                                            markerfacecolor=self.get_outcome_color(result_type),
                                            markersize=10,
                                            markeredgecolor='white',
                                            markeredgewidth=1,
                                            label=result_type))
        
        legend = ax_legend.legend(handles=legend_elements, 
                                loc='center',
                                ncol=4,
                                fontsize=9,
                                frameon=True,
                                fancybox=True,
                                shadow=True,
                                columnspacing=1.5)
        
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(0.95)
        frame.set_edgecolor('#1e3d59')
        frame.set_linewidth(1.5)

    def guardar_sin_espacios(self, fig, filename):
        """Guarda en A4 landscape"""
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
        print(f"✅ Archivo guardado: {filename}")

    def create_reporte_rematadores(self, figsize=(11.69, 8.27)):
        """Crea reporte completo"""
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, figure=fig,
                    height_ratios=[1.2, 1],
                    width_ratios=[1, 1],
                    hspace=0.15,
                    wspace=0.10,
                    left=0.04, right=0.96,
                    top=0.88, bottom=0.05)
        
        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25)
            ax_bg.axis('off')
        
        fig.suptitle(f'CÓRNERS OFENSIVOS (REMATADORES) - {self.team_filter}', 
                    fontsize=18, fontweight='bold', color='#1e3d59', y=0.96)
        
        if self.team_filter and (team_logo := self.load_team_logo(self.team_filter)) is not None:
            ax_team = fig.add_axes([0.86, 0.89, 0.08, 0.1])
            ax_team.imshow(team_logo, aspect='auto')
            ax_team.axis('off')

        # --- INICIO DE LA CORRECCIÓN ---
        # 1. Creamos una lista global de jugadores para asegurar colores consistentes
        top_izq = self.get_top_rematadores('izquierda', n=8)
        top_der = self.get_top_rematadores('derecha', n=8)
        
        # Combinamos las listas y obtenemos jugadores únicos, manteniendo el orden
        # de los mejores rematadores en general.
        all_players = top_izq + [p for p in top_der if p[0] not in dict(top_izq)]
        top_players_global = all_players[:8] # Nos quedamos con los 8 mejores en total
        # --- FIN DE LA CORRECCIÓN ---

        ax_campo_izq = fig.add_subplot(gs[0, 0])
        # 2. Pasamos la lista global a la función del campograma
        self.create_campograma_rematadores(ax_campo_izq, 'izquierda', top_players_global)
        ax_campo_izq.set_aspect('auto')
        
        ax_campo_der = fig.add_subplot(gs[0, 1])
        # 2. Pasamos la lista global a la función del campograma
        self.create_campograma_rematadores(ax_campo_der, 'derecha', top_players_global)
        ax_campo_der.set_aspect('auto')
        
        self.create_leyenda_completa(fig, top_players_global)
        
        ax_rank_izq = fig.add_subplot(gs[1, 0])
        self.create_ranking_rematadores(ax_rank_izq, 'izquierda', top_players_global)
        ax_rank_izq.set_aspect('auto')
        
        ax_rank_der = fig.add_subplot(gs[1, 1])
        self.create_ranking_rematadores(ax_rank_der, 'derecha', top_players_global)
        ax_rank_der.set_aspect('auto')
        
        return fig

def seleccionar_equipo_interactivo():
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
    try:
        print("=== GENERADOR DE REPORTE REMATADORES CÓRNERS ===")
        if (equipo := seleccionar_equipo_interactivo()) is None:
            print("No se pudo completar la selección.")
            return
        
        print(f"\nGenerando reporte de rematadores para {equipo}")
        analyzer = ReporteRematadoresCorners(team_filter=equipo)
        
        if (fig := analyzer.create_reporte_rematadores()):
            plt.show()
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_rematadores_corners_{equipo_filename}.pdf"
            analyzer.guardar_sin_espacios(fig, output_path)
            print(f"✅ Reporte guardado como: {output_path}")
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()

def generar_reporte_personalizado(equipo, mostrar=True, guardar=True):
    try:
        analyzer = ReporteRematadoresCorners(team_filter=equipo)
        fig = analyzer.create_reporte_rematadores()
        
        if fig:
            if mostrar:
                plt.show()
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"reporte_rematadores_corners_{equipo_filename}.pdf"
                analyzer.guardar_sin_espacios(fig, output_path)
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