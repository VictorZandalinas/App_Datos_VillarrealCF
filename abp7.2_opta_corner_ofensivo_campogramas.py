import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
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

class ReporteCampogramasCorners:
    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/match_events.parquet", team_filter=None):
        self.data_path = data_path
        self.team_filter = team_filter
        self.df = None
        self.corner_data = pd.DataFrame()
        self.team_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/team_stats.parquet")
        self.player_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/player_stats.parquet")
        self.xg_events = pd.read_parquet("extraccion_opta/datos_opta_parquet/xg_events.parquet") 

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

    def load_data(self, team_filter=None):
        """Carga los datos necesarios"""
        try:
            columns_needed = ['Match ID', 'Team ID', 'Team Name', 'Event Name', 'outcome', 
                            'timeMin', 'timeSec', 'timeStamp', 'x', 'y', 'Pass End X', 'Pass End Y', 
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
            
            # NUEVO: Merge para obtener dorsales del lanzador
            try:
                # Preparar datos de player_stats para merge
                player_dorsales = self.player_stats[['Player ID', 'Shirt Number']].copy()
                player_dorsales = player_dorsales.rename(columns={
                    'Player ID': 'playerId', 
                    'Shirt Number': 'dorsal_lanzador'
                })
                
                # Hacer merge con left join para mantener todos los eventos
                self.df = self.df.merge(
                    player_dorsales,
                    on='playerId', 
                    how='left'
                )
                
                
            except Exception as e:
                print(f"⚠️ Error en merge dorsales: {e}")
                # Añadir columna vacía si falla el merge
                self.df['dorsal_lanzador'] = None
            # Merge con xG events
            try:
                xg_data = self.xg_events[['timeStamp', 'playerId', 'qualifier 321']].copy()
                xg_data = xg_data.rename(columns={'qualifier 321': 'xg_value'})
                
                # AÑADE ESTA LÍNEA PARA CONVERTIR A NÚMERO
                xg_data['xg_value'] = pd.to_numeric(xg_data['xg_value'], errors='coerce')
                
                # Normalizar timestamps: quitar la Z del final en self.df si existe
                self.df['timeStamp_clean'] = self.df['timeStamp'].str.replace('Z', '', regex=False)
                xg_data['timeStamp_clean'] = xg_data['timeStamp']
                
                # Merge
                self.df = self.df.merge(
                    xg_data[['timeStamp_clean', 'playerId', 'xg_value']],
                    on=['timeStamp_clean', 'playerId'],
                    how='left'
                )
                
                
            except Exception as e:
                print(f"⚠️ Error en merge xG: {e}")
                self.df['xg_value'] = None

            corners_df = self.df[
                (self.df['Corner taken'] == 'Sí') & 
                (self.df['Team Name'] == team_filter if team_filter else True)
            ].copy()
            
            corners_df = corners_df.drop_duplicates(
                subset=['Match ID', 'Team ID', 'playerName', 'timeMin', 'timeSec'], 
                keep='first'
            )
            
            self.corner_data = corners_df.copy()
            
            
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")
            import traceback
            traceback.print_exc()

    def extract_lanzamientos_izquierda(self, team_filter=None):
        """Extrae lanzamientos del lado izquierdo"""
        
        df_sorted = self.df.sort_values(['Match ID', 'timeMin', 'timeSec']).reset_index(drop=True)
        lanzamientos_list = []
        
        if team_filter:
            team_matches = set(self.team_stats[
                self.team_stats['Team Name'] == team_filter
            ]['Match ID'].unique())
        
        for match_id in df_sorted['Match ID'].unique():
            if team_filter and match_id not in team_matches:
                continue
                
            match_events = df_sorted[df_sorted['Match ID'] == match_id].reset_index(drop=True)
            
            lanzamientos_izq = match_events[
                (match_events['Event Name'] == 'Pass') & 
                (match_events.get('Corner taken', '') == 'Sí') &
                (match_events['y'] > 99) &
                (match_events['x'].notna()) & 
                (match_events['y'].notna())
            ]
            
            for lanz_idx, lanzamiento in lanzamientos_izq.iterrows():
                result_data = self.analyze_lanzamiento_sequence(match_events, lanz_idx, lanzamiento)
                
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
                    'Pass End Y': lanzamiento['Pass End Y'],
                    'dorsal_lanzador': lanzamiento.get('dorsal_lanzador'),  
                    'tipo_lanzamiento': self.get_tipo_lanzamiento(lanzamiento)  
                }
                
                lanzamiento_data.update(result_data)
                lanzamientos_list.append(lanzamiento_data)
        
        if lanzamientos_list:
            lanzamientos_data = pd.DataFrame(lanzamientos_list)
            lanzamientos_data = lanzamientos_data.drop_duplicates(
                subset=['Match ID', 'timeMin', 'timeSec', 'playerName'], 
                keep='first'
            )
            
            return lanzamientos_data
        else:
            print("❌ No se encontraron lanzamientos del lado izquierdo")
            return pd.DataFrame()

    def extract_lanzamientos_derecha(self, team_filter=None):
        """Extrae lanzamientos del lado derecho"""
        
        df_sorted = self.df.sort_values(['Match ID', 'timeMin', 'timeSec']).reset_index(drop=True)
        lanzamientos_list = []
        
        if team_filter:
            team_matches = set(self.team_stats[
                self.team_stats['Team Name'] == team_filter
            ]['Match ID'].unique())
        
        for match_id in df_sorted['Match ID'].unique():
            if team_filter and match_id not in team_matches:
                continue
                
            match_events = df_sorted[df_sorted['Match ID'] == match_id].reset_index(drop=True)
            
            lanzamientos_der = match_events[
                (match_events['Event Name'] == 'Pass') & 
                (match_events.get('Corner taken', '') == 'Sí') &
                (match_events['y'] < 1) &
                (match_events['x'].notna()) & 
                (match_events['y'].notna())
            ]
            
            for lanz_idx, lanzamiento in lanzamientos_der.iterrows():
                result_data = self.analyze_lanzamiento_sequence(match_events, lanz_idx, lanzamiento)
                
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
                    'Pass End Y': lanzamiento['Pass End Y'],
                    'dorsal_lanzador': lanzamiento.get('dorsal_lanzador'),  # ← FALTA
                    'tipo_lanzamiento': self.get_tipo_lanzamiento(lanzamiento)  # ← FALTA
                }
                
                lanzamiento_data.update(result_data)
                lanzamientos_list.append(lanzamiento_data)
        
        if lanzamientos_list:
            lanzamientos_data = pd.DataFrame(lanzamientos_list)
            lanzamientos_data = lanzamientos_data.drop_duplicates(
                subset=['Match ID', 'timeMin', 'timeSec', 'playerName'], 
                keep='first'
            )
            
            return lanzamientos_data
        else:
            print("❌ No se encontraron lanzamientos del lado derecho")
            return pd.DataFrame()
    
    def get_tipo_lanzamiento(self, lanzamiento_row):
        """Determina si el lanzamiento es cerrado, abierto o plano"""
        if lanzamiento_row.get('In-swinger') == 'Sí':
            return 'cerrado'
        elif lanzamiento_row.get('Out-swinger') == 'Sí':
            return 'abierto'
        elif lanzamiento_row.get('Straight') == 'Sí':
            return 'plano'
        else:
            return 'desconocido'

    def analyze_lanzamiento_sequence(self, match_events, lanzamiento_idx, lanzamiento_pass):
        """
        Analiza la secuencia post-lanzamiento con la LÓGICA MEZCLADA:
        - Detección de fin de secuencia de abp2 (pausas de >5s entre eventos).
        - Clasificación de "primer contacto" de abp7.2 (remate en <=3s desde el córner).
        """
        from datetime import datetime
        
        ### --- LÓGICA DE TIEMPO MEZCLADA --- ###
        # 1. Tiempo total desde el córner (para clasificar 'primer contacto')
        start_timestamp = lanzamiento_pass['timeStamp']
        start_time = datetime.fromisoformat(start_timestamp.replace('Z', '+00:00'))
        
        # 2. Tiempo del evento previo (para detectar pausas en la secuencia)
        prev_event_timestamp = start_time
        
        events_found = {
            'Goal': None, 'Attempt Saved': None, 'Miss': None,
            'Post': None, 'Otro contacto': None
        }
        
        primer_contacto_encontrado = False
        es_primer_contacto = False
        total_time_diff = 0  # Tiempo total desde el córner
        
        # FUNCIÓN AUXILIAR para obtener dorsal del rematador
        def get_dorsal_rematador(player_id):
            if pd.isna(player_id) or player_id is None:
                return None
            try:
                player_info = self.player_stats[self.player_stats['Player ID'] == player_id]
                if not player_info.empty:
                    dorsal = player_info['Shirt Number'].iloc[0]
                    return int(dorsal) if pd.notna(dorsal) else None
            except Exception as e:
                pass
            return None
        
        for next_idx in range(lanzamiento_idx + 1, len(match_events)):
            next_event = match_events.iloc[next_idx]
            
            ### --- CÁLCULO DE AMBOS TIEMPOS --- ###
            next_time = datetime.fromisoformat(next_event['timeStamp'].replace('Z', '+00:00'))
            
            # Tiempo entre eventos consecutivos (lógica de abp2)
            consecutive_time_diff = (next_time - prev_event_timestamp).total_seconds()
            
            # Tiempo total desde el saque de esquina (lógica de abp7.2)
            total_time_diff = (next_time - start_time).total_seconds()
            
            ### --- CONDICIONES DE CORTE (de abp2) --- ###
            # Si hay más de 5 segundos entre un evento y el siguiente, la secuencia termina.
            if consecutive_time_diff > 5:
                break
                
            # Si un evento finaliza la jugada, la secuencia termina.
            if next_event['Event Name'] in ['Corner Awarded', 'Foul', 'Offside', 'End Period']:
                break
            
            event_name = next_event['Event Name']
            event_team_id = next_event['Team ID']
            lanzamiento_team_id = lanzamiento_pass['Team ID']
            
            if event_team_id != lanzamiento_team_id:
                # Actualizamos el tiempo previo incluso si es del rival para detectar pausas
                prev_event_timestamp = next_time
                continue
            
            # Lógica para encontrar el resultado...
            if (event_name == 'Pass' and 
                next_event.get('outcome') == 1 and 
                events_found['Otro contacto'] is None):
                events_found['Otro contacto'] = next_event
            
            elif event_name in ['Goal', 'Attempt Saved', 'Miss', 'Post'] and events_found[event_name] is None:
                events_found[event_name] = next_event
                
                ### --- CLASIFICACIÓN DE CONTACTO (de abp7.2) --- ###
                # Usamos el tiempo TOTAL para decidir si es primer contacto.
                if not primer_contacto_encontrado and total_time_diff <= 5:
                    es_primer_contacto = True
                    primer_contacto_encontrado = True
                
                if event_name == 'Goal':
                    break # Si es gol, terminamos la búsqueda.
            
            ### --- ACTUALIZACIÓN DE TIEMPO (de abp2) --- ###
            # Actualizamos el timestamp del evento previo para la siguiente iteración.
            prev_event_timestamp = next_time
        
        # Procesar resultados (esta parte no cambia)
        if events_found['Goal'] is not None:
            event = events_found['Goal']
            dorsal_rematador = get_dorsal_rematador(event.get('playerId'))
            return {
                'result_type': 'Gol',
                'final_x': float(event.get('x', 0)), 'final_y': float(event.get('y', 0)),
                'goal_player': event.get('playerName', ''), 'goal_player_id': event.get('playerId'),
                'dorsal_rematador': dorsal_rematador,
                'es_primer_contacto': es_primer_contacto,
                'tiempo_desde_corner': total_time_diff,
                'xg_value': event.get('xg_value')  
            }
            
        elif events_found['Attempt Saved'] is not None:
            event = events_found['Attempt Saved']
            dorsal_rematador = get_dorsal_rematador(event.get('playerId'))
            return {
                'result_type': 'Tiro a puerta',
                'final_x': float(event.get('x', 0)), 'final_y': float(event.get('y', 0)),
                'goal_player': event.get('playerName', ''), 'goal_player_id': event.get('playerId'),
                'dorsal_rematador': dorsal_rematador,
                'es_primer_contacto': es_primer_contacto,
                'tiempo_desde_corner': total_time_diff,
                'xg_value': event.get('xg_value')
            }
            
        elif events_found['Miss'] is not None:
            event = events_found['Miss']
            dorsal_rematador = get_dorsal_rematador(event.get('playerId'))
            return {
                'result_type': 'Tiro fuera',
                'final_x': float(event.get('x', 0)), 'final_y': float(event.get('y', 0)),
                'goal_player': event.get('playerName', ''), 'goal_player_id': event.get('playerId'),
                'dorsal_rematador': dorsal_rematador,
                'es_primer_contacto': es_primer_contacto,
                'tiempo_desde_corner': total_time_diff,
                'xg_value': event.get('xg_value')
            }
            
        elif events_found['Post'] is not None:
            event = events_found['Post']
            dorsal_rematador = get_dorsal_rematador(event.get('playerId'))
            return {
                'result_type': 'Tiro al poste',
                'final_x': float(event.get('x', 0)), 'final_y': float(event.get('y', 0)),
                'goal_player': event.get('playerName', ''), 'goal_player_id': event.get('playerId'),
                'dorsal_rematador': dorsal_rematador,
                'es_primer_contacto': es_primer_contacto,
                'tiempo_desde_corner': total_time_diff,
                'xg_value': event.get('xg_value')
            }
            
        elif events_found['Otro contacto'] is not None:
            event = events_found['Otro contacto']
            return {
                'result_type': 'Otro contacto',
                'final_x': float(event.get('Pass End X', 0)), 'final_y': float(event.get('Pass End Y', 0)),
                'goal_player': None, 'goal_player_id': None, 'dorsal_rematador': None,
                'es_primer_contacto': False, # Un 'otro contacto' nunca es el remate final.
                'tiempo_desde_corner': total_time_diff,
                'xg_value': event.get('xg_value')
            }
        else:
            return {
                'result_type': 'Sin remate',
                'final_x': float(lanzamiento_pass.get('Pass End X', 0)), 'final_y': float(lanzamiento_pass.get('Pass End Y', 0)),
                'goal_player': None, 'goal_player_id': None, 'dorsal_rematador': None,
                'es_primer_contacto': False,
                'tiempo_desde_corner': 0,
                'xg_value': None
            }

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

    def get_contactos_por_zona(self, lado='izquierda'):
        """Cuenta primer contacto por zona"""
        if lado == 'izquierda':
            lanzamientos_data = self.extract_lanzamientos_izquierda(self.team_filter)
        else:
            lanzamientos_data = self.extract_lanzamientos_derecha(self.team_filter)
        
        zonas_primer_contacto = {'zona_1': 0, 'zona_2': 0, 'zona_3': 0, 'zona_4': 0, 
                            'zona_5': 0, 'zona_6': 0, 'zona_7': 0}
        zonas_posterior = {'zona_1': 0, 'zona_2': 0, 'zona_3': 0, 'zona_4': 0, 
                        'zona_5': 0, 'zona_6': 0, 'zona_7': 0}
        zonas_total = {'zona_1': 0, 'zona_2': 0, 'zona_3': 0, 'zona_4': 0, 
                    'zona_5': 0, 'zona_6': 0, 'zona_7': 0}
        zonas_xg = {'zona_1': 0.0, 'zona_2': 0.0, 'zona_3': 0.0, 'zona_4': 0.0, 
                    'zona_5': 0.0, 'zona_6': 0.0, 'zona_7': 0.0}

        for _, lanzamiento in lanzamientos_data.iterrows():
            if lanzamiento['result_type'] not in ['Sin remate', 'Otro contacto']:
                zona = self.get_zona_from_coordinates(lanzamiento['final_x'], lanzamiento['final_y'])
                if zona:
                    zonas_total[zona] += 1

                    if pd.notna(lanzamiento.get('xg_value')):
                        zonas_xg[zona] += lanzamiento['xg_value']
                    
                    if lanzamiento.get('es_primer_contacto', False):
                        zonas_primer_contacto[zona] += 1
                    else:
                        zonas_posterior[zona] += 1
        
        return {
            'total': zonas_total,
            'primer_contacto': zonas_primer_contacto,
            'posterior': zonas_posterior,
            'xg': zonas_xg # AÑADE ESTA LÍNEA
        }

    def get_player_shirt_number_by_name(self, player_name):
        """Obtiene el dorsal del jugador por nombre"""
        if pd.isna(player_name):
            return None
        
        player_info = self.player_stats[self.player_stats['Match Name'] == player_name]
        if not player_info.empty:
            shirt_number = player_info['Shirt Number'].iloc[0]
            return str(int(shirt_number)) if pd.notna(shirt_number) else None
        return None

    def get_outcome_marker(self, result_type):
        marker_map = {
            'Sin remate': 'v',
            'Tiro a puerta': 's',
            'Tiro fuera': 'X',
            'Tiro al poste': 'P',
            'Gol': 'o',
            'Otro contacto': 'D'
        }
        return marker_map.get(result_type, 'o')

    def get_outcome_color(self, result_type):
        color_map = {
            'Sin remate': '#95A5A6',
            'Tiro a puerta': '#3498DB', 
            'Tiro fuera': '#E74C3C',
            'Tiro al poste': '#FF6B35',
            'Gol': '#F1C40F',
            'Otro contacto': '#9B59B6'
        }
        return color_map.get(result_type, '#95A5A6')

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

    def load_ball_image(self):
        """Carga imagen del balón"""
        return plt.imread("assets/balon.png") if os.path.exists("assets/balon.png") else None

    def load_background(self):
        """Carga imagen de fondo"""
        return plt.imread("assets/fondo_informes.png") if os.path.exists("assets/fondo_informes.png") else None

    def create_leyenda_conjunta_campogramas(self, fig, gs):
        """Crea leyenda conjunta para ambos campogramas"""
        lanzamientos_izq = self.extract_lanzamientos_izquierda(self.team_filter)
        lanzamientos_der = self.extract_lanzamientos_derecha(self.team_filter)
        lanzamientos_combinados = pd.concat([lanzamientos_izq, lanzamientos_der])
        
        if lanzamientos_combinados.empty:
            return
        
        legend_elements = []
        
        # Agregar goles si existen
        data_goles = lanzamientos_combinados[lanzamientos_combinados['result_type'] == 'Gol']
        if len(data_goles) > 0:
            goles_primer = len(data_goles[data_goles.get('es_primer_contacto', False) == True])
            goles_posterior = len(data_goles[data_goles.get('es_primer_contacto', False) == False])
            
            if goles_primer > 0:
                legend_elements.append(plt.Line2D([0], [0], marker='H', color='w', 
                                                markerfacecolor='#F1C40F', markeredgecolor='#E67E22',
                                                markeredgewidth=2, markersize=14,
                                                label=f'Gol - 1er contacto ({goles_primer})'))
            
            if goles_posterior > 0:
                legend_elements.append(plt.Line2D([0], [0], marker='H', color='w', 
                                                markerfacecolor='#F1C40F', markeredgecolor='#E67E22',
                                                markeredgewidth=1, markersize=11,
                                                label=f'Gol - Posterior ({goles_posterior})'))
        
        # Agregar resto de tipos
        for result_type in ['Tiro a puerta', 'Tiro fuera', 'Tiro al poste', 'Otro contacto', 'Sin remate']:
            data_tipo = lanzamientos_combinados[lanzamientos_combinados['result_type'] == result_type]
            
            if len(data_tipo) > 0:
                primer_contacto_count = len(data_tipo[data_tipo.get('es_primer_contacto', False) == True])
                posterior_count = len(data_tipo[data_tipo.get('es_primer_contacto', False) == False])
                
                if primer_contacto_count > 0:
                    legend_elements.append(plt.Line2D([0], [0], 
                                                    marker=self.get_outcome_marker(result_type), 
                                                    color='w', 
                                                    markerfacecolor=self.get_outcome_color(result_type),
                                                    markeredgewidth=2, markersize=10,
                                                    label=f'{result_type} - 1er contacto ({primer_contacto_count})'))
                
                if posterior_count > 0:
                    legend_elements.append(plt.Line2D([0], [0], 
                                                    marker=self.get_outcome_marker(result_type), 
                                                    color='w', 
                                                    markerfacecolor=self.get_outcome_color(result_type),
                                                    markeredgewidth=1, markersize=8,
                                                    label=f'{result_type} - Posterior ({posterior_count})'))
        
        if legend_elements:
            # Crear leyenda flotante centrada entre los campogramas
            ax_legend = fig.add_axes([0.35, 0.48, 0.3, 0.15])
            ax_legend.axis('off')
            
            legend = ax_legend.legend(handles=legend_elements, 
                                    loc='center', ncol=2,
                                    framealpha=0.95, facecolor='white', 
                                    edgecolor='#1e3d59', fontsize=8,
                                    title='1er contacto: ≤4s | Posterior: >4s',  # Cambiado de 4s a 3s
                                    title_fontproperties={'weight': 'bold', 'size': 9})
            
            if legend.get_title():
                legend.get_title().set_color('#1e3d59')

    def create_campograma_lado_mejorado(self, ax, lado='izquierda'):
        """
        Campograma optimizado con:
        - Flechas de colores según tipo (abierto/rojo, cerrado/amarillo, plano/negro).
        - Dorsales con el mismo color que la flecha y borde de contraste.
        - Flechas completas que no se cortan si se salen del área del gráfico.
        """
        if lado == 'izquierda':
            lanzamientos_data = self.extract_lanzamientos_izquierda(self.team_filter)
            titulo = 'LANZAMIENTOS - LADO IZQUIERDO'
            start_coords = (99.5, 100)
        else:
            lanzamientos_data = self.extract_lanzamientos_derecha(self.team_filter)
            titulo = 'LANZAMIENTOS - LADO DERECHO'
            start_coords = (0.5, 100)
        
        if lanzamientos_data.empty:
            ax.text(0.5, 0.5, f'{titulo}\n\n(Sin datos)', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        pitch = VerticalPitch(half=True, pitch_type='opta', pitch_color='#2d5a27', 
                            line_color='white', linewidth=2)
        pitch.draw(ax=ax)
        
        ax.set_title(titulo, fontsize=11, fontweight='bold', color='#1e3d59', pad=8)
        
        tamaño_primer_contacto = 120
        tamaño_posterior = 20
        
        # Priorizamos dibujar los remates de primer contacto que NO son goles
        # para que las flechas y los goles se superpongan correctamente.
        eventos_prioritarios = ['Tiro a puerta', 'Tiro fuera', 'Tiro al poste', 'Otro contacto', 'Sin remate', 'Gol']
        
        for result_type in sorted(lanzamientos_data['result_type'].unique(), key=lambda x: eventos_prioritarios.index(x)):
            data_tipo = lanzamientos_data[lanzamientos_data['result_type'] == result_type]
            
            if len(data_tipo) > 0:
                # Separamos siempre los datos en primer contacto y posterior
                data_primer_contacto = data_tipo[data_tipo.get('es_primer_contacto', False) == True]
                data_posterior = data_tipo[data_tipo.get('es_primer_contacto', False) == False]

                # --- 1. DIBUJAR FLECHAS Y SÍMBOLOS DE PRIMER CONTACTO ---
                for _, remate in data_primer_contacto.iterrows():
                    
                    # --- LÓGICA DE FLECHAS (AHORA SE APLICA A GOLES TAMBIÉN) ---
                    tipo_lanz = remate.get('tipo_lanzamiento')
                    if tipo_lanz == 'abierto':
                        arrow_color = 'red'
                    elif tipo_lanz == 'cerrado':
                        arrow_color = 'yellow'
                    else:  # 'plano' o 'desconocido'
                        arrow_color = 'black'
                    
                    base_rad = 0.3 if tipo_lanz == 'abierto' else -0.3 if tipo_lanz == 'cerrado' else 0
                    rad = base_rad if lado == 'derecha' else -base_rad
                    stroke_color = 'black' if arrow_color == 'yellow' else 'white'

                    end_coords = (remate['final_y'], remate['final_x'])
                    arrow = FancyArrowPatch(start_coords, end_coords,
                                            connectionstyle=f"arc3,rad={rad}", 
                                            arrowstyle='->', mutation_scale=15,
                                            color=arrow_color, alpha=0.7, linewidth=2, 
                                            zorder=5, clip_on=False)
                    ax.add_patch(arrow)

                    if pd.notna(remate.get('dorsal_lanzador')):
                        mid_y = (start_coords[0] + end_coords[0]) / 2
                        mid_x = (start_coords[1] + end_coords[1]) / 2
                        vec_y, vec_x = end_coords[0] - start_coords[0], end_coords[1] - start_coords[1]
                        perp_vec_y, perp_vec_x = -vec_x, vec_y
                        offset_y, offset_x = (perp_vec_y * rad) / 2, (perp_vec_x * rad) / 2
                        mid_y_adjusted, mid_x_adjusted = mid_y + offset_y, mid_x + offset_x
                        ax.text(mid_y_adjusted, mid_x_adjusted, str(int(remate['dorsal_lanzador'])),
                            fontsize=7, fontweight='bold', color=arrow_color, 
                            ha='center', va='center', zorder=6,
                            path_effects=[patheffects.withStroke(linewidth=3, foreground=stroke_color)])
                    
                    # --- LÓGICA PARA DIBUJAR EL SÍMBOLO (GOL vs OTROS) ---
                    if remate['result_type'] == 'Gol':
                        if remate['goal_player']:
                            dorsal = self.get_player_shirt_number_by_name(remate['goal_player'])
                            if self.team_filter and (team_logo := self.load_team_logo(self.team_filter)) is not None:
                                logo_box = OffsetImage(team_logo, zoom=0.3)
                                ab_logo = AnnotationBbox(logo_box, (remate['final_y'], remate['final_x']), 
                                                    frameon=False, zorder=12)
                                ax.add_artist(ab_logo)
                            if dorsal:
                                ax.text(remate['final_y'], remate['final_x'], dorsal,
                                    fontsize=12, fontweight='bold', color='white',
                                    ha='center', va='center', zorder=15,
                                    path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
                    else:
                        # Para otros tipos, dibujamos el scatter (lo hacemos fuera del bucle para eficiencia)
                        pass

                    # Colocar dorsal del REMATADOR (para no goles)
                    if remate['result_type'] != 'Gol' and pd.notna(remate.get('dorsal_rematador')):
                        ax.text(remate['final_y'], remate['final_x'], str(int(remate['dorsal_rematador'])),
                            fontsize=7, fontweight='bold', color=arrow_color, 
                            ha='center', va='center', zorder=12,
                            path_effects=[patheffects.withStroke(linewidth=3, foreground=stroke_color)])

                # Dibujar todos los puntos de primer contacto (no goles) de una vez
                data_primer_no_goles = data_primer_contacto[data_primer_contacto['result_type'] != 'Gol']
                if not data_primer_no_goles.empty:
                    ax.scatter(data_primer_no_goles['final_y'], data_primer_no_goles['final_x'], 
                                c=self.get_outcome_color(result_type), 
                                marker=self.get_outcome_marker(result_type),
                                s=tamaño_primer_contacto, alpha=0.9,
                                edgecolors='white', linewidth=2, zorder=10)

                # --- 2. DIBUJAR SÍMBOLOS DE CONTACTO POSTERIOR (SIN FLECHAS) ---
                for _, remate in data_posterior.iterrows():
                    if remate['result_type'] == 'Gol':
                         if remate['goal_player']:
                            dorsal = self.get_player_shirt_number_by_name(remate['goal_player'])
                            if self.team_filter and (team_logo := self.load_team_logo(self.team_filter)) is not None:
                                logo_box = OffsetImage(team_logo, zoom=0.25) # Más pequeño
                                ab_logo = AnnotationBbox(logo_box, (remate['final_y'], remate['final_x']), 
                                                    frameon=False, zorder=12)
                                ax.add_artist(ab_logo)
                            if dorsal:
                                ax.text(remate['final_y'], remate['final_x'], dorsal,
                                    fontsize=8, fontweight='bold', color='white', # Más pequeño
                                    ha='center', va='center', zorder=15,
                                    path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
                    else:
                        # Para otros tipos, dibujamos el scatter (lo hacemos fuera del bucle)
                        pass
                
                # Dibujar todos los puntos posteriores (no goles) de una vez
                data_posterior_no_goles = data_posterior[data_posterior['result_type'] != 'Gol']
                if not data_posterior_no_goles.empty:
                    ax.scatter(data_posterior_no_goles['final_y'], data_posterior_no_goles['final_x'], 
                                c=self.get_outcome_color(result_type), 
                                marker=self.get_outcome_marker(result_type),
                                s=tamaño_posterior, alpha=0.6,
                                edgecolors='white', linewidth=1, zorder=4)
                                   
                    
                    for _, remate in data_primer_contacto.iterrows():
                        # --- NUEVA LÓGICA DE COLORES Y FLECHAS ---
                        
                        # 1. Definir el color para la flecha y dorsales según el tipo
                        tipo_lanz = remate.get('tipo_lanzamiento')
                        if tipo_lanz == 'abierto':
                            arrow_color = 'red'
                        elif tipo_lanz == 'cerrado':
                            arrow_color = 'yellow'
                        else:  # 'plano' o 'desconocido'
                            arrow_color = 'black'
                        
                        # 2. Definir curvatura y borde de texto para contraste
                        base_rad = 0.3 if tipo_lanz == 'abierto' else -0.3 if tipo_lanz == 'cerrado' else 0
                        
                        # Invertir la curvatura si es el lado izquierdo
                        rad = base_rad if lado == 'derecha' else -base_rad
                        
                        stroke_color = 'black' if arrow_color == 'yellow' else 'white'


                        # 3. Crear y dibujar la flecha con el color y sin corte
                        end_coords = (remate['final_y'], remate['final_x'])
                        arrow = FancyArrowPatch(start_coords, end_coords,
                                                connectionstyle=f"arc3,rad={rad}", 
                                                arrowstyle='->', mutation_scale=15,
                                                color=arrow_color, alpha=0.7, linewidth=2, 
                                                zorder=5, clip_on=False) # <--- ¡CLAVE PARA EVITAR EL CORTE!
                        ax.add_patch(arrow)

                        # 4. Colocar dorsal del LANZADOR en medio de la flecha (sobre el arco)
                        if pd.notna(remate.get('dorsal_lanzador')):
                            # 1. Calcular el punto medio de la cuerda (línea recta)
                            mid_y = (start_coords[0] + end_coords[0]) / 2
                            mid_x = (start_coords[1] + end_coords[1]) / 2
                            
                            # 2. Calcular el vector de la cuerda
                            vec_y = end_coords[0] - start_coords[0]
                            vec_x = end_coords[1] - start_coords[1]
                            
                            # 3. Calcular el vector perpendicular a la cuerda
                            perp_vec_y = -vec_x
                            perp_vec_x = vec_y
                            
                            # 4. Calcular el desplazamiento desde el punto medio de la cuerda hasta el
                            #    punto medio del arco. La altura del arco en 'arc3' es (rad * longitud_cuerda) / 2.
                            #    Como nuestro vector perpendicular ya tiene la longitud de la cuerda,
                            #    solo necesitamos escalarlo por rad / 2.
                            offset_y = (perp_vec_y * rad) / 2
                            offset_x = (perp_vec_x * rad) / 2
                            
                            # 5. La posición final del texto es el punto medio de la cuerda más el desplazamiento
                            mid_y_adjusted = mid_y + offset_y
                            mid_x_adjusted = mid_x + offset_x

                            # Usar las coordenadas ajustadas para posicionar el texto
                            ax.text(mid_y_adjusted, mid_x_adjusted, str(int(remate['dorsal_lanzador'])),
                                fontsize=7, fontweight='bold', color=arrow_color, 
                                ha='center', va='center', zorder=6,
                                path_effects=[patheffects.withStroke(linewidth=3, foreground=stroke_color)])

                        # 5. Colocar dorsal del REMATADOR dentro del símbolo
                        if pd.notna(remate.get('dorsal_rematador')):
                            ax.text(remate['final_y'], remate['final_x'], str(int(remate['dorsal_rematador'])),
                                fontsize=7, fontweight='bold', color=arrow_color, 
                                ha='center', va='center', zorder=12,
                                path_effects=[patheffects.withStroke(linewidth=3, foreground=stroke_color)])

                    if len(data_posterior) > 0:
                        ax.scatter(data_posterior['final_y'], data_posterior['final_x'], 
                                    c=self.get_outcome_color(result_type), 
                                    marker=self.get_outcome_marker(result_type),
                                    s=tamaño_posterior,
                                    alpha=0.6,
                                    edgecolors='white',
                                    linewidth=1, zorder=4)

        if (ball := self.load_ball_image()) is not None:
            ball_box = OffsetImage(ball, zoom=0.05)
            ab_ball = AnnotationBbox(ball_box, start_coords, frameon=False, zorder=15)
            ax.add_artist(ab_ball)

    def create_mapa_calor_lado(self, ax, lado='izquierda'):
        """Crea mapa de calor para un lado específico"""
        if lado == 'izquierda':
            titulo = '1er CONTACTO - LADO IZQUIERDO'
        else:
            titulo = '1er CONTACTO - LADO DERECHO'
        
        pitch = VerticalPitch(half=True, pitch_type='opta', pitch_color='#2d5a27', 
                                    line_color='white', linewidth=2)
        pitch.draw(ax=ax)
        
        ax.set_xlim(0, 100)
        ax.set_ylim(70, 102)
        ax.set_title(titulo, fontsize=12, fontweight='bold', color='#1e3d59', pad=10)

        zonas_data_completa = self.get_contactos_por_zona(lado)
        zonas_data = zonas_data_completa['primer_contacto']
        zonas_primer_contacto = zonas_data_completa['primer_contacto']
        zonas_xg = zonas_data_completa['xg']
        
        max_remates = max(zonas_data.values()) if zonas_data.values() else 1

        import matplotlib.cm as cm
        colormap = cm.get_cmap('Blues')

        zona_coords = {
            'zona_1': [(70, 0), (100, 25)],
            'zona_2': [(70, 25), (88.5, 75)],    
            'zona_3': [(88.5, 25), (100, 42)],
            'zona_4': [(94.2, 42), (100, 58)],   
            'zona_5': [(88.5, 58), (100, 75)],
            'zona_6': [(83, 42), (94.2, 58)],    
            'zona_7': [(70, 75), (100, 100)]
        }

        orden_dibujo = ['zona_1', 'zona_2', 'zona_3', 'zona_4', 'zona_5', 'zona_7', 'zona_6']

        for zona in orden_dibujo:
            if zona in zona_coords:
                coords = zona_coords[zona]
                (x_min, y_min), (x_max, y_max) = coords
                width, height = x_max - x_min, y_max - y_min
                count_total = zonas_data.get(zona, 0)
                count_primer = zonas_primer_contacto.get(zona, 0)
                
                if max_remates > 0:
                    intensidad = 0.2 + (count_total / max_remates) * 0.8
                else:
                    intensidad = 0.2
                
                color_zona = colormap(intensidad)
                alpha_zona = 1.0 if zona == 'zona_6' else 0.8
                
                rect = patches.Rectangle((y_min, x_min), height, width, 
                                    linewidth=3, edgecolor='black',  
                                    facecolor=color_zona, alpha=alpha_zona)
                ax.add_patch(rect)
                
                center_y, center_x = y_min + height/2, x_min + width/2
                text_color = 'white' if intensidad > 0.6 else 'black'
                
                ax.text(center_y, center_x, str(count_total), 
                        fontsize=20, fontweight='bold', ha='center', va='top',  # Cambia va='center' a va='top'
                        color=text_color)
                
                xg_acumulado = zonas_xg.get(zona, 0.0)
                
                # AÑADIR texto del xG debajo, SOLO si es mayor que 0:
                if xg_acumulado > 0:
                    ax.text(center_y, center_x, f"({xg_acumulado:.2f})", 
                            fontsize=10, fontweight='normal', ha='center', va='bottom',
                            color=text_color)

        
        ax.text(0.02, 0.02, 'Primer contacto - (xG)',  # Cambiado de 4s a 3s
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='bottom')

    def create_reporte_campogramas(self, figsize=(11.69, 8.27)):    
        """Crea reporte con campogramas y mapas de calor en layout 2x2 para A4 HORIZONTAL"""
        # El figsize se cambia aquí a las dimensiones de A4 horizontal
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        # Grid 2x2 ajustado para el nuevo tamaño
        gs = fig.add_gridspec(2, 2, 
                            height_ratios=[1, 1],
                            width_ratios=[1, 1],
                            hspace=0.15,
                            wspace=0.10,
                            # Márgenes ajustados para A4
                            left=0.04, right=0.96,
                            top=0.88, bottom=0.05)
        
        # Fondo (sin cambios)
        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25)
            ax_bg.axis('off')
        
        # Título principal (sin cambios)
        fig.suptitle(f'CÓRNERS OFENSIVOS', 
                    fontsize=20, fontweight='bold', color='#1e3d59', y=0.96)
        
        # Logo del equipo (posición X ajustada para el nuevo ancho)
        if self.team_filter and (team_logo := self.load_team_logo(self.team_filter)) is not None:
            # Se mueve el logo un poco más a la izquierda para que no se salga
            ax_team = fig.add_axes([0.86, 0.89, 0.08, 0.1])
            ax_team.imshow(team_logo, aspect='auto')
            ax_team.axis('off')

        # FILA 1: Campogramas (sin cambios)
        ax_campo_izq = fig.add_subplot(gs[0, 0])
        self.create_campograma_lado_mejorado(ax_campo_izq, 'izquierda')
        ax_campo_izq.set_aspect('auto') 

        ax_campo_der = fig.add_subplot(gs[0, 1])
        self.create_campograma_lado_mejorado(ax_campo_der, 'derecha')
        ax_campo_der.set_aspect('auto') 

        # Leyenda conjunta (sin cambios)
        self.create_leyenda_conjunta_campogramas(fig, gs)
        
        # FILA 2: Mapas de calor (sin cambios)
        ax_calor_izq = fig.add_subplot(gs[1, 0])
        self.create_mapa_calor_lado(ax_calor_izq, 'izquierda')
        ax_calor_izq.set_aspect('auto') 

        ax_calor_der = fig.add_subplot(gs[1, 1])
        self.create_mapa_calor_lado(ax_calor_der, 'derecha')
        ax_calor_der.set_aspect('auto') 

        return fig

# Funciones auxiliares
def seleccionar_equipo_interactivo():
    """Selección interactiva de equipo"""
    try:
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/match_events.parquet", columns=['Team Name'])
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
        
        analyzer = ReporteCampogramasCorners(team_filter=equipo)
        
        if (fig := analyzer.create_reporte_campogramas()):
            plt.show()
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_campogramas_corners_{equipo_filename}.pdf"
            # PARÁMETROS DE GUARDADO AJUSTADOS
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0, 
                       facecolor='white', dpi=300, orientation='landscape')
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()

def generar_reporte_personalizado(equipo, mostrar=True, guardar=True):
    """Genera reporte personalizado para un equipo específico"""
    try:
        analyzer = ReporteCampogramasCorners(team_filter=equipo)
        fig = analyzer.create_reporte_campogramas()
        
        if fig:
            if mostrar: 
                plt.show()
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"reporte_campogramas_corners_{equipo_filename}.pdf"
                # PARÁMETROS DE GUARDADO AJUSTADOS
                fig.savefig(output_path, bbox_inches='tight', pad_inches=0, 
                           facecolor='white', dpi=300, orientation='landscape')
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