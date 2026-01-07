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

@lru_cache(maxsize=128)
def determine_corner_side_cached(self, x, y):
    return self.determine_corner_side(x, y)

class CornersSequenceAnalysis:
    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/abp_events.parquet", team_filter=None):
        self.data_path = data_path
        self.team_filter = team_filter
        self.df = None
        self.corner_sequences = pd.DataFrame()
        self.team_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/team_stats.parquet")
        self.player_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/player_stats.parquet")  # ← AÑADIR ESTA LÍNEA
        self.load_data(team_filter)
        
        if team_filter:
            self.df = self.df.merge(self.team_stats[['Team ID', 'Team Position']], 
                        on='Team ID', how='left')
            self.extract_corner_sequences(team_filter)

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
        print(f"Archivo guardado SIN espacios formato A4: {filename}")

    def load_data(self, team_filter=None):
        """Carga solo los datos necesarios desde el inicio"""
        try:
            # Cargar solo columnas necesarias
            columns_needed = ['Match ID', 'periodId', 'Team ID', 'Team Name', 'Event Name', 'outcome',
                'timeMin', 'timeSec', 'timeStamp', 'x', 'y', 'Pass End X', 'Pass End Y', 'playerName', 'playerId', 'Corner taken']
            
            self.df = pd.read_parquet(self.data_path, columns=columns_needed)

            # <<< AÑADE ESTA LÍNEA PARA LA SOLUCIÓN >>>
            # Elimina las filas que son duplicados exactos en tiempo, evento y jugador.
            self.df = self.df.drop_duplicates(subset=['Match ID', 'timeMin', 'timeSec', 'Event Name', 'playerName'], keep='first')

            self.df['timeStamp'] = self.df['timeStamp'].apply(self.normalize_timestamp)

            # Filtrar eventos 'Deleted event' después de la de-duplicación
            self.df = self.df[self.df['Event Name'] != 'Deleted event']
            
            # Si hay filtro de equipo, filtrar matches desde el inicio
            if team_filter:
                team_matches = self.team_stats[self.team_stats['Team Name'] == team_filter]['Match ID'].unique()
                self.df = self.df[self.df['Match ID'].isin(team_matches)]
            
            print(f"✅ Datos filtrados cargados: {len(self.df)} eventos")
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")

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
        
    # --- REEMPLAZA LA FUNCIÓN ENTERA POR ESTA ---

    def extract_corner_sequences(self, team_filter=None):
        """Extrae secuencias de córners con la lógica directa y simplificada."""
        if self.df is None:
            print("❌ No hay datos cargados")
            return

        print("🔍 Extrayendo secuencias de córners...")

        df_sorted = self.df.sort_values(['Match ID', 'periodId', 'timeStamp']).reset_index(drop=True)

        if team_filter:
            team_matches = set(self.team_stats[self.team_stats['Team Name'] == team_filter]['Match ID'].unique())

        corner_list = []

        for match_id in df_sorted['Match ID'].unique():
            if team_filter and match_id not in team_matches:
                continue

            match_events = df_sorted[df_sorted['Match ID'] == match_id].reset_index(drop=True)

            # ▼▼▼ LÓGICA DE BÚSQUEDA SIMPLIFICADA (IGUAL QUE EN LOS OTROS SCRIPTS) ▼▼▼
            # Buscamos directamente los eventos marcados como saque de esquina.
            corners_directos = match_events[
                (match_events.get('Corner taken', '') == 'Sí') &
                (match_events['x'].notna()) &
                (match_events['y'].notna())
            ]

            # Para cada córner encontrado - ELIMINAR DUPLICADOS PRIMERO
            corners_directos_unique = corners_directos.drop_duplicates(
                subset=['Match ID', 'timeMin', 'timeSec', 'playerName'], 
                keep='first'
            ).reset_index(drop=True)

            for corner_idx, corner_pass in corners_directos_unique.iterrows():
                # Encontrar el índice real en match_events
                real_corner_idx = None
                for idx, event in match_events.iterrows():
                    if (event['timeMin'] == corner_pass['timeMin'] and 
                        event['timeSec'] == corner_pass['timeSec'] and
                        event['playerName'] == corner_pass['playerName'] and
                        event.get('Corner taken', '') == 'Sí'):
                        real_corner_idx = idx
                        break
                
                if real_corner_idx is None:
                    continue
                    
                # Construir datos del córner usando el índice real
                corner_data = {
                    'corner_pass_id': real_corner_idx,
                    'Match ID': match_id,
                    'Team ID': corner_pass['Team ID'],
                    'Team Name': corner_pass['Team Name'],
                    'player_name': corner_pass.get('playerName', ''),
                    'timeMin': corner_pass['timeMin'],
                    'timeSec': corner_pass['timeSec'],
                    'x': corner_pass['x'],
                    'y': corner_pass['y'],
                    'outcome': corner_pass['outcome']
                }

                # Analizar la secuencia usando el índice real
                result_data = self.analyze_corner_sequence(
                    match_events, real_corner_idx, corner_pass, self.team_filter
                )
                corner_data.update(result_data)
                corner_data['corner_side'] = self.determine_corner_side(
                    corner_data['x'], corner_data['y']
                )
                corner_list.append(corner_data)

        # Crear DataFrame final
        if corner_list:
            self.corner_sequences = pd.DataFrame(corner_list)
            # Eliminar duplicados por si acaso
            self.corner_sequences.drop_duplicates(
                subset=['Match ID', 'timeMin', 'timeSec', 'player_name'],
                keep='first', inplace=True
            )
            print(f"✅ Total de córners extraídos: {len(self.corner_sequences)}")

            print("\n📊 Resumen por tipo de resultado:")
            print(self.corner_sequences['result_type'].value_counts())
        else:
            print("❌ No se encontraron córners")
    
    # ### CAMBIO CRUCIAL Y LÓGICA SIMPLIFICADA AQUÍ ###
    def analyze_corner_sequence(self, match_events, corner_pass_idx, corner_pass, team_filter=None):
        """
        Analiza la secuencia post-córner con lógica de tiempo entre eventos consecutivos.
        """
        start_time_min = corner_pass['timeMin']
        start_time_sec = corner_pass['timeSec']
        corner_period_id = corner_pass['periodId']
        
        # --- INICIO: DEBUG DE SECUENCIA ---
        print("\n" + "="*80)
        print(f"--- Córner a analizar ---")
        print(f"Match ID: {corner_pass['Match ID']}, Equipo: {corner_pass['Team Name']}, Jugador: {corner_pass.get('playerName', 'N/A')}, "
            f"Min: {start_time_min}:{start_time_sec:02d}")
        print("--- Secuencia de eventos ---")
        # --- FIN: DEBUG DE SECUENCIA ---

        events_found = {
            'Goal': None, 'Post': None, 'Attempt Saved': None,
            'Miss': None, 'Otro contacto': None
        }
        goal_player_id = None
        is_second_play = False
        previous_event_coords = None
        result_event_idx = None
        pass_count = 0  
        
        # Usar timestamps para tiempo entre eventos consecutivos
        corner_timestamp = pd.to_datetime(corner_pass['timeStamp'])
        prev_event_timestamp = corner_timestamp
        
        # <--- NUEVA LÍNEA --->
        # Guardamos el timestamp del último pase contado para evitar contar duplicados en el mismo instante.
        last_pass_timestamp = corner_timestamp

        for next_idx in range(corner_pass_idx + 1, len(match_events)):
            next_event = match_events.iloc[next_idx]
            event_name = next_event['Event Name']
            
            # --- DEBUG DE EVENTO INDIVIDUAL ---
            print(f"  -> Evento: {next_event['Event Name']:<15} | Equipo: {next_event['Team Name']:<15} | "
                f"Min: {next_event['timeMin']}:{next_event['timeSec']:02d} | Jugador: {next_event.get('playerName', 'N/A')}")
            # --- FIN DEBUG DE EVENTO INDIVIDUAL ---

            # Condición de parada: cambio de período
            if next_event['periodId'] != corner_period_id:
                print("  [STOP] Cambio de periodo.")
                break

            # Condición de parada: evento retrocede demasiado
            event_x_coord = float(next_event.get('x', 0))
            if event_name == 'Pass' and event_x_coord <= 60:
                print(f"  [STOP] Evento retrocede (x={event_x_coord}).")
                break
            
            # Condición de parada: eventos que terminan la jugada
            if event_name in ['Corner Awarded', 'Foul', 'Offside', 'End Period', 'Out']:
                print(f"  [STOP] Evento de corte: {event_name}")
                break
            
            # LÓGICA CORREGIDA: Tiempo entre eventos consecutivos
            next_timestamp = pd.to_datetime(next_event['timeStamp'])
            time_diff = (next_timestamp - prev_event_timestamp).total_seconds()

            if time_diff > 5:
                print(f"  [STOP] Más de 5 segundos entre eventos consecutivos: {time_diff}s")
                break

            event_team_id = next_event['Team ID']
            corner_team_id = corner_pass['Team ID']

            # <--- LÓGICA MODIFICADA --->
            # Contar pases del mismo equipo, asegurando que el timestamp sea diferente.
            if event_name == 'Pass' and event_team_id == corner_team_id:
                # Solo contamos el pase si su timestamp es posterior al del último pase contado.
                if next_timestamp > last_pass_timestamp:
                    pass_count += 1
                    last_pass_timestamp = next_timestamp  # Actualizamos el timestamp del último pase.
                    
                    # Si llegamos a 5 o más pases, verificar la nueva condición
                    if pass_count >= 5:
                        # Contar pases con x < 70 en los últimos eventos
                        passes_back_field = 0
                        for check_idx in range(corner_pass_idx + 1, next_idx + 1):
                            check_event = match_events.iloc[check_idx]
                            if (check_event['Event Name'] == 'Pass' and 
                                check_event['Team ID'] == corner_team_id and
                                float(check_event.get('x', 0)) < 70):
                                passes_back_field += 1
                        
                        # Si hay 2 o más pases con x < 70, cortar la secuencia
                        if passes_back_field >= 2:
                            print(f"  [STOP] Más de 5 pases con 2+ pases en campo defensivo (x<70)")
                            break
            
            # Buscar eventos de finalización (solo del equipo que lanza el córner)
            if (event_name in ['Goal', 'Post', 'Attempt Saved', 'Miss'] and 
                event_team_id == corner_team_id and 
                events_found[event_name] is None):
                events_found[event_name] = next_event
                result_event_idx = next_idx
                if event_name == 'Goal': 
                    break

            # Buscar "otro contacto"
            elif (event_name == 'Pass' and 
                event_team_id == corner_team_id and
                float(next_event.get('x', 0)) > 99 and 
                next_event.get('outcome') == 1 and 
                next_idx + 1 < len(match_events) and 
                match_events.iloc[next_idx + 1]['Event Name'] == 'Pass' and 
                match_events.iloc[next_idx + 1]['Team ID'] == corner_team_id and
                events_found['Otro contacto'] is None):
                events_found['Otro contacto'] = next_event

            # ACTUALIZAR el timestamp previo para la próxima iteración
            prev_event_timestamp = next_timestamp

        # Analizar si es segunda jugada
        if result_event_idx is not None and result_event_idx > corner_pass_idx + 1:
            # Buscar el evento inmediatamente anterior al remate con x > 55 (de CUALQUIER equipo)
            previous_event_with_x_55 = None
            previous_event_idx = None
            
            # Buscar hacia atrás desde el remate
            for idx in range(result_event_idx - 1, corner_pass_idx, -1):
                event = match_events.iloc[idx]
                # CONSIDERAR EVENTOS DE CUALQUIER EQUIPO con x > 55
                # AÑADIR CONDICIÓN: El timestamp del evento previo debe ser diferente al del córner.
                if (float(event.get('x', 0)) > 55 and
                        event['Event Name'] not in ['Deleted event'] and
                        event['timeStamp'] != corner_pass['timeStamp']):
                    previous_event_with_x_55 = event
                    previous_event_idx = idx
                    break
            
            # LÓGICA CAMBIADA: Si NO hay eventos previos con x > 55, NO es segunda jugada
            # Si SÍ hay eventos previos con x > 55, SÍ es segunda jugada
            if previous_event_with_x_55 is None:
                is_second_play = False
                print(f"  [NO ES SEGUNDA JUGADA - Sin eventos intermedios con x > 55]")
            else:
                is_second_play = True
                previous_event_coords = (float(previous_event_with_x_55.get('x', 0)), float(previous_event_with_x_55.get('y', 0)))
                
                # DEBUG DE SEGUNDA JUGADA
                print(f"  [SEGUNDA JUGADA DETECTADA]")
                print(f"    Córner: x={corner_pass['x']}, y={corner_pass['y']}, Event={corner_pass['Event Name']}, TimeStamp={corner_pass['timeStamp']}")
                print(f"    Evento previo: x={previous_event_with_x_55.get('x')}, y={previous_event_with_x_55.get('y')}, Event={previous_event_with_x_55['Event Name']}, TimeStamp={previous_event_with_x_55['timeStamp']}")
                
                # Buscar el evento de remate para completar la secuencia
                remate_event = None
                for event_type in ['Goal', 'Post', 'Attempt Saved', 'Miss']:
                    if events_found[event_type] is not None:
                        remate_event = events_found[event_type]
                        break
                
                if remate_event is not None:
                    print(f"    Remate: x={remate_event.get('x')}, y={remate_event.get('y')}, Event={remate_event['Event Name']}, TimeStamp={remate_event['timeStamp']}")
                print(f"  [FIN DEBUG SEGUNDA JUGADA]")

        # Determinar el resultado final
        result_type, goal_player, event_x, event_y, result_time, result_event = (
            'Sin remate', None, 
            float(corner_pass.get('Pass End X', 0)), 
            float(corner_pass.get('Pass End Y', 0)), 
            "N/A", "None"
        )

        if events_found['Goal'] is not None:
            event = events_found['Goal']
            result_type = 'Gol'
            result_event = 'Goal'
            goal_player_id = event.get('playerId')
        elif events_found['Post'] is not None:
            event = events_found['Post']
            result_type = 'Poste'
            result_event = 'Post'
        elif events_found['Attempt Saved'] is not None:
            event = events_found['Attempt Saved']
            result_type = 'Tiro a puerta'
            result_event = 'Attempt Saved'
        elif events_found['Miss'] is not None:
            event = events_found['Miss']
            result_type = 'Tiro fuera'
            result_event = 'Miss'
        elif events_found['Otro contacto'] is not None:
            event = events_found['Otro contacto']
            result_type = 'Otro contacto'
            result_event = 'Pass (otro contacto)'
            event_x, event_y = float(event.get('Pass End X', 0)), float(event.get('Pass End Y', 0))
            is_second_play = False
            previous_event_coords = None

        # Actualizar coordenadas y datos del jugador si hay evento
        if 'event' in locals():
            goal_player = event.get('playerName', '')
            event_x, event_y = float(event.get('x', event_x)), float(event.get('y', event_y))
            result_time = f"{event.get('timeMin', 0)}:{event.get('timeSec', 0):02d}"

        # --- DEBUG DE RESULTADO ---
        print(f"--- Resultado final: {result_type} ---")
        print("="*80 + "\n")

        return {
            'corner_x': event_x, 'corner_y': event_y, 'result_type': result_type,
            'goal_player': goal_player, 'goal_player_id': goal_player_id,
            'is_second_play': is_second_play, 'previous_event_coords': previous_event_coords,
            'week': corner_pass.get('Week', 'N/A'),
            'debug_info': f"Corner at {start_time_min}:{start_time_sec:02d} -> Result: {result_type} at {result_time}",
            'corner_player': corner_pass.get('playerName', 'N/A'),
            'corner_time_min': start_time_min, 'corner_time_sec': start_time_sec,
            'result_time': result_time
        }

    def _get_team_id(self, team_name):
        """Helper para obtener Team ID desde team_name"""
        team_info = self.team_stats[self.team_stats['Team Name'] == team_name]
        return team_info['Team ID'].iloc[0] if not team_info.empty else None
    
    def _get_default_result(self, corner_pass, corner_debug_info):
        """Helper para resultado por defecto (Sin remate)"""    
        return {
            'corner_x': float(corner_pass.get('Pass End X', 0)),
            'corner_y': float(corner_pass.get('Pass End Y', 0)),
            'result_type': 'Sin remate',
            'goal_player': None,
            'week': corner_debug_info['Week'],
            'debug_info': f"Week {corner_debug_info['Week']} | Match {corner_debug_info['Match_ID']} | Corner {corner_debug_info['Time']} | Result: Sin remate",
            'corner_player': corner_debug_info['Player'],
            'corner_time_min': corner_debug_info['Time'].split(':')[0],
            'corner_time_sec': corner_debug_info['Time'].split(':')[1],
            'result_time': "N/A"
        }

    def get_player_shirt_number(self, player_id):
        """Obtiene el número de camiseta del jugador"""
        if pd.isna(player_id):
            return None
        
        player_info = self.player_stats[self.player_stats['Player ID'] == player_id]
        if not player_info.empty:
            shirt_number = player_info['Shirt Number'].iloc[0]
            return str(int(shirt_number)) if pd.notna(shirt_number) else None
        return None

    def _format_result(self, event, result_type, corner_debug_info):
        """Helper para formatear resultados"""
        return {
            'corner_x': float(event.get('x', 0)),
            'corner_y': float(event.get('y', 0)),
            'result_type': result_type,
            'goal_player': event.get('playerName', ''),
            'week': corner_debug_info['Week'],
            'debug_info': f"Week {corner_debug_info['Week']} | Match {corner_debug_info['Match_ID']} | Corner {corner_debug_info['Time']} | Result: {result_type}",
            'corner_player': corner_debug_info['Player'],
            'corner_time_min': corner_debug_info['Time'].split(':')[0],
            'corner_time_sec': corner_debug_info['Time'].split(':')[1],
            'result_time': f"{event.get('timeMin', 0)}:{event.get('timeSec', 0):02d}"
        }

    def _format_otro_contacto_result(self, event, corner_debug_info):
        """Helper para formatear resultado de otro contacto"""
        return {
            'corner_x': float(event.get('Pass End X', 0)),
            'corner_y': float(event.get('Pass End Y', 0)),
            'result_type': 'Otro contacto',
            'goal_player': None,
            'week': corner_debug_info['Week'],
            'debug_info': f"Week {corner_debug_info['Week']} | Match {corner_debug_info['Match_ID']} | Corner {corner_debug_info['Time']} | Result: Otro contacto",
            'corner_player': corner_debug_info['Player'],
            'corner_time_min': corner_debug_info['Time'].split(':')[0],
            'corner_time_sec': corner_debug_info['Time'].split(':')[1],
            'result_time': f"{event.get('timeMin', 0)}:{event.get('timeSec', 0):02d}"
        }
    
    def determine_corner_side(self, x, y):
        y = float(y)
        return 'izquierda' if y > 50 else 'derecha'
    
    def prepare_corner_data(self, team_filter=None):
        if self.corner_sequences.empty:
            print("❌ No hay datos de córners para preparar")
            return {}
        
        corner_data = {
            'ofensivo_izquierda': {'home': [], 'away': []}, 'ofensivo_derecha': {'home': [], 'away': []},
            'defensivo_izquierda': {'home': [], 'away': []}, 'defensivo_derecha': {'home': [], 'away': []}
        }
        
        if not team_filter:
            return corner_data
        
        team_matches = self.team_stats[self.team_stats['Team Name'] == team_filter]
        match_positions = dict(zip(team_matches['Match ID'], team_matches['Team Position']))
        
        # Córners ofensivos
        team_corners = self.corner_sequences[self.corner_sequences['Team Name'] == team_filter]
        for _, corner in team_corners.iterrows():
            if (team_pos := match_positions.get(corner['Match ID'])) is None: continue
            key = f"ofensivo_{corner.get('corner_side', 'izquierda')}"
            try:
                x, y = float(corner['corner_x']), float(corner['corner_y'])
            except (ValueError, TypeError): continue
            if x == 0 or y == 0: continue
            
            point_data = {
                'x': x, 'y': y, 
                'result_type': corner['result_type'],
                'player_name': corner.get('player_name', ''), 
                'goal_player': corner.get('goal_player', ''),
                'goal_player_id': corner.get('goal_player_id', ''),
                'Team Name': corner.get('Team Name', ''),
                # Nuevos campos para segundo toque
                'is_second_play': corner.get('is_second_play', False),
                'previous_event_coords': corner.get('previous_event_coords', None)
            }
            corner_data[key][team_pos].append(point_data)
        
        # Córners defensivos
        rival_corners = self.corner_sequences[
            (self.corner_sequences['Match ID'].isin(match_positions.keys())) &
            (self.corner_sequences['Team Name'] != team_filter)
        ]
        for _, corner in rival_corners.iterrows():
            if (team_pos := match_positions.get(corner['Match ID'])) is None: continue
            rival_pos = 'away' if team_pos == 'home' else 'home'
            key = f"defensivo_{corner.get('corner_side', 'izquierda')}"
            try:
                x, y = float(corner['corner_x']), float(corner['corner_y'])
            except (ValueError, TypeError): continue
            if x == 0 or y == 0: continue
            
            point_data = {
                'x': x, 'y': y, 'result_type': corner['result_type'],
                'player_name': corner.get('player_name', ''), 
                'goal_player': corner.get('goal_player', ''),
                'goal_player_id': corner.get('goal_player_id'),
                'Team Name': corner.get('Team Name', ''),
                # Nuevos campos para segundo toque
                'is_second_play': corner.get('is_second_play', False),
                'previous_event_coords': corner.get('previous_event_coords', None)
            }
            corner_data[key][rival_pos].append(point_data)
        
        print("\n📊 Resumen de córners preparados:")
        for key, data in corner_data.items():
            home_count, away_count = len(data['home']), len(data['away'])
            if home_count + away_count > 0:
                print(f"  {key}: home={home_count}, away={away_count}")
        
        return corner_data
    
    # ### CAMBIO AQUÍ: ELIMINAMOS "LANZAMIENTO" Y "OTRO CONTACTO" ###
    def get_outcome_marker(self, result_type):
        """Retorna el marcador según el tipo de resultado"""
        marker_map = {
            'Sin remate': 'v',
            'Tiro a puerta': 's',
            'Tiro fuera': 'X',
            'Poste': '^',  # Nuevo marcador para poste (triángulo hacia arriba)
            'Gol': '*',
            'Otro contacto': 'D'
        }
        return marker_map.get(result_type, 'o')

    def export_debug_info(self, filename="debug_corners.csv"):
        """Exporta información de debug de todos los córners procesados"""
        if self.corner_sequences.empty:
            return
    
        debug_df = self.corner_sequences[['Match ID', 'Team Name', 'week', 'timeMin', 
                                          'timeSec', 'result_type', 'corner_x', 
                                          'corner_y', 'debug_info']]
        debug_df.to_csv(filename, index=False)
        print(f"✅ Debug info exportada a: {filename}")
    
    def load_team_logo(self, equipo, target_size=(80, 80)):
        """Carga y redimensiona el logo del equipo a un tamaño fijo"""
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
                # Abrir y redimensionar con PIL
                with Image.open(logo_path) as img:
                    # Convertir a RGBA si no lo está
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    
                    # Redimensionar manteniendo aspecto
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    
                    # Crear imagen de tamaño fijo con fondo transparente
                    final_img = Image.new('RGBA', target_size, (0, 0, 0, 0))
                    
                    # Centrar la imagen redimensionada
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
    
    def load_villarreal_logo(self): return self.load_team_logo('Villarreal CF')
    def load_ball_image(self): return plt.imread("assets/balon.png") if os.path.exists("assets/balon.png") else None
    def load_background(self): return plt.imread("assets/fondo_informes.png") if os.path.exists("assets/fondo_informes.png") else None

    
    def create_corners_visualization(self, figsize=(11.69, 8.27), team_filter=None):
        corner_data = self.prepare_corner_data(team_filter)

        # Configuración agresiva para eliminar espacios
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

        ball_img = self.load_ball_image()  # Cargar una sola vez
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25)
            ax_bg.axis('off')
        
        fig.suptitle('ANÁLISIS DE SECUENCIAS DE CÓRNERS', fontsize=24, fontweight='bold', color='#1e3d59', y=0.98, family='serif')
        
        if (ball := self.load_ball_image()) is not None:
            ax_ball = fig.add_axes([0.05, 0.92, 0.09, 0.09]); ax_ball.imshow(ball); ax_ball.axis('off')
        
        if team_filter and (team_logo := self.load_team_logo(team_filter)) is not None:
            ax_team = fig.add_axes([0.88, 0.90, 0.08, 0.09]); ax_team.imshow(team_logo, aspect='auto'); ax_team.axis('off')
        if (villarreal_logo := self.load_villarreal_logo()) is not None:
            ax_villarreal = fig.add_axes([0.85, 0.90, 0.08, 0.09]); ax_villarreal.imshow(villarreal_logo, aspect='auto'); ax_villarreal.axis('off')

        # Crear una rejilla de 2x2 más directa para un mejor control del espacio
        gs = GridSpec(2, 2, figure=fig, 
            left=0.05, right=0.95,      # Dejamos un margen mínimo en los bordes
            bottom=0.14, top=0.90,
            wspace=0.1,               # Espacio horizontal mínimo entre gráficos
            hspace=0.15)               # Un poco más de espacio vertical para los títulos

        # Crear los 4 ejes (axes) directamente de la rejilla 2x2
        axes = [
            fig.add_subplot(gs[0, 0]),  # Gráfico superior izquierdo
            fig.add_subplot(gs[0, 1]),  # Gráfico superior derecho
            fig.add_subplot(gs[1, 0]),  # Gráfico inferior izquierdo
            fig.add_subplot(gs[1, 1])   # Gráfico inferior derecho
        ]
        
        titles = ['OFENSIVO IZQUIERDA', 'DEFENSIVO IZQUIERDA', 'OFENSIVO DERECHA', 'DEFENSIVO DERECHA']
        keys = ['ofensivo_izquierda', 'defensivo_izquierda', 'ofensivo_derecha', 'defensivo_derecha']

        # Cargar el logo del equipo una sola vez
        team_logo = self.load_team_logo(team_filter) if team_filter else None
        
        lanzadores_stats = {}
        if team_filter and not self.corner_sequences.empty:
            team_corners_ofensivos = self.corner_sequences[
                (self.corner_sequences['Team Name'] == team_filter) & (self.corner_sequences['player_name'].notna())
            ]
            for side in ['izquierda', 'derecha']:
                side_corners = team_corners_ofensivos[team_corners_ofensivos['corner_side'] == side]
                if 'player_name' in side_corners.columns:
                    lanzadores_stats[f'ofensivo_{side}'] = side_corners.drop_duplicates('corner_pass_id')['player_name'].value_counts()
        
        all_result_types = set()
        
        for ax, title, key in zip(axes, titles, keys):
            # Determinar colores según si es defensivo o no
            if 'defensivo' in key:
                # En defensivos: invertir colores (home=rojo, away=azul)
                home_color = 'red'
                home_edge_color = 'darkred'
                away_color = 'blue'
                away_edge_color = 'darkblue'
            else:
                # En ofensivos: colores normales (home=azul, away=rojo)
                home_color = 'blue'
                home_edge_color = 'darkblue'
                away_color = 'red'
                away_edge_color = 'darkred'
            
            pitch = VerticalPitch(half=True, pitch_type='opta', pitch_color='none', 
                                line_color='black', linewidth=2)
            pitch.draw(ax=ax)
            ax.set_title(title, fontsize=14, fontweight='bold', color='#1e3d59', 
                        pad=5, family='serif')
            
            ax.set_aspect('auto')
            
            # Escudo grande en zonas defensivas
            if 'defensivo' in key and team_logo is not None:
                logo_ax = ax.inset_axes([0.31, 0.21, 0.40, 0.40])  # Coordenadas relativas al gráfico (0-1)
                logo_ax.imshow(team_logo, alpha=0.1)
                logo_ax.axis('off')
            
            # Añadir balón en las esquinas
            if 'defensivo' in key and ball_img is not None:
                if 'izquierda' in key:
                    ball_box = OffsetImage(ball_img, zoom=0.05)
                    ball_ab = AnnotationBbox(ball_box, (99.5, 99.5), frameon=False)
                    ax.add_artist(ball_ab)
                else:
                    ball_box = OffsetImage(ball_img, zoom=0.05)
                    ball_ab = AnnotationBbox(ball_box, (0.5, 99.5), frameon=False)
                    ax.add_artist(ball_ab)    

            data = corner_data.get(key, {'home': [], 'away': []})
            
            # Sección de puntos home:
            for point in data['home']:
                result_type = point['result_type']
                all_result_types.add(result_type)
                
                # Dibujar línea si es segundo toque
                if point.get('is_second_play', False) and point.get('previous_event_coords'):
                    prev_x, prev_y = point['previous_event_coords']
                    # Dibujar línea desde evento anterior hasta evento de resultado
                    ax.plot([prev_y, point['y']], [prev_x, point['x']], 
                            color=home_color, linestyle='--', linewidth=2, alpha=0.7)
                
                if result_type == 'Gol':
                    # Intentar cargar el escudo del equipo
                    team_logo_for_goal = self.load_team_logo(point.get('Team Name', team_filter))
                    
                    if team_logo_for_goal is not None:
                        # Usar el escudo en lugar de la estrella
                        imagebox = OffsetImage(team_logo_for_goal, zoom=0.2)
                        ab = AnnotationBbox(imagebox, (point['y'], point['x']), 
                                            frameon=False)  
                        ax.add_artist(ab)
                    else:
                        # Si no hay escudo, usar la estrella como antes
                        ax.scatter(point['y'], point['x'], c=home_color, s=60, alpha=0.8, 
                                marker='*', edgecolors=home_edge_color, linewidth=1.5)
                    
                    # Añadir nombre del jugador
                    if point.get('goal_player'):
                        goal_player_id = point.get('goal_player_id')
                        shirt_number = self.get_player_shirt_number(goal_player_id)
                        
                        if shirt_number:
                            ax.text(point['y'], point['x'] + 1.2, f"{shirt_number}", 
                                fontsize=12, fontweight='bold', ha='center', va='center',
                                color=home_color, family='monospace')
                    
                else:
                    # Para otros tipos de resultado, usar el scatter normal
                    ax.scatter(point['y'], point['x'], c=home_color, s=60, alpha=0.8, 
                            marker=self.get_outcome_marker(result_type), 
                            edgecolors=home_edge_color, linewidth=1.5)

            # Sección de puntos away:
            for point in data['away']:
                result_type = point['result_type']
                all_result_types.add(result_type)
                
                # Dibujar línea si es segundo toque
                if point.get('is_second_play', False) and point.get('previous_event_coords'):
                    prev_x, prev_y = point['previous_event_coords']
                    # Dibujar línea desde evento anterior hasta evento de resultado
                    ax.plot([prev_y, point['y']], [prev_x, point['x']], 
                            color=away_color, linestyle='--', linewidth=2, alpha=0.7)
                
                if result_type == 'Gol':
                    # Intentar cargar el escudo del equipo
                    team_logo_for_goal = self.load_team_logo(point.get('Team Name', team_filter))
                    
                    if team_logo_for_goal is not None:
                        # Usar el escudo en lugar de la estrella
                        imagebox = OffsetImage(team_logo_for_goal, zoom=0.2)
                        ab = AnnotationBbox(imagebox, (point['y'], point['x']), 
                                            frameon=False)
                        ax.add_artist(ab)
                    else:
                        # Si no hay escudo, usar la estrella como antes
                        ax.scatter(point['y'], point['x'], c=away_color, s=60, alpha=0.8, 
                            marker='*', edgecolors=away_edge_color, linewidth=1.5)
                    
                    # Añadir nombre del jugador
                    if point.get('goal_player'):
                        goal_player_id = point.get('goal_player_id')
                        shirt_number = self.get_player_shirt_number(goal_player_id)
                        print(f"[DEBUG-DORSAL] Jugador: {point.get('goal_player')}, PlayerID: {goal_player_id}, Dorsal encontrado: {shirt_number}")
                        
                        if shirt_number:
                            ax.text(point['y'], point['x'] + 1.2, f"{shirt_number}", 
                                fontsize=12, fontweight='bold', ha='center', va='center',
                                color=away_color, family='monospace')
                
                else:
                    # Para otros tipos de resultado, usar el scatter normal
                    ax.scatter(point['y'], point['x'], c=away_color, s=60, alpha=0.8, 
                            marker=self.get_outcome_marker(result_type), 
                            edgecolors=away_edge_color, linewidth=1.5)
                    

            if 'ofensivo' in key and key in lanzadores_stats and not (player_stats := lanzadores_stats[key]).empty:
                bbox = [0.01, 0.85, 0.15, 0.17] if 'izquierda' in key else [0.84, 0.85, 0.15, 0.17]
                table_ax = ax.inset_axes(bbox, zorder=10); table_ax.axis('off')
                if team_filter and team_logo is not None:
                    table_ax.imshow(team_logo, aspect='auto', alpha=0.5, extent=(0, 1, 0, 1))
                table_ax.set_facecolor('none')
                title_text = table_ax.text(0.6, 0.90, "Lanzadores", color='white', fontsize=8, fontweight='bold', ha='center', va='center', fontfamily='sans-serif')
                title_text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='#1e3d59')])
                for i, (player, count) in enumerate(player_stats.head(3).items()):
                    display_name = player[:12] + '.' if len(player) > 12 else player
                    player_text = table_ax.text(0.08, 0.70 - (i * 0.30), display_name, color='white', fontsize=7, ha='left', va='center', fontfamily='sans-serif')
                    count_text = table_ax.text(1.5, 0.70 - (i * 0.30), str(count), color='white', fontsize=8, ha='right', va='center', fontweight='bold', fontfamily='sans-serif')
                    player_text.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='#1e3d59')])
                    count_text.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='#1e3d59')])

        from matplotlib.lines import Line2D
        legend_ax = fig.add_axes([0.05, 0.02, 0.90, 0.10]); legend_ax.axis('off')
        
        # ### CAMBIO AQUÍ: ACTUALIZAMOS LA LEYENDA ###
        ordered_types = ['Sin remate', 'Tiro a puerta', 'Tiro fuera', 'Poste', 'Otro contacto', 'Gol']
        
        existing_types = [t for t in ordered_types if t in all_result_types]
        for t in all_result_types:
            if t not in existing_types: existing_types.append(t)
        
        legend_elements = []
        for result_type in existing_types:
            if result_type == 'Gol':
                legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=8, markeredgecolor='darkred', label='Gol - Escudo (Visitante)'))  # <-- Cambiar a Visitante
                legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, markeredgecolor='darkblue', label='Gol - Escudo (Local)'))      # <-- Cambiar a Local
            else:
                marker = self.get_outcome_marker(result_type)
                legend_elements.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor='red', markersize=10, markeredgecolor='darkred', label=f'{result_type} (Visitante)'))  # <-- Cambiar a Visitante
                legend_elements.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor='blue', markersize=10, markeredgecolor='darkblue', label=f'{result_type} (Local)'))      # <-- Cambiar a Local
        
        if legend_elements:
            legend = legend_ax.legend(handles=legend_elements, loc='center', frameon=True, 
                                    fancybox=True, shadow=True, ncol=4, fontsize=11, 
                                    columnspacing=2.0, handletextpad=0.8)
            frame = legend.get_frame()
            frame.set_facecolor('white'); frame.set_alpha(0.95); 
            frame.set_edgecolor('#1e3d59'); frame.set_linewidth(1.5)
        
        return fig
    
    # ... (resto de funciones main, print_summary, etc., sin cambios) ...
    def print_summary(self, team_filter=None):
        if self.corner_sequences.empty: print("No hay datos de córners para mostrar"); return
        
        print(f"\n=== RESUMEN DE CÓRNERS ===\nTotal de córners: {len(self.corner_sequences)}")
        
        if team_filter:
            team_corners = self.corner_sequences[self.corner_sequences['Team Name'] == team_filter]
            rival_corners = self.corner_sequences[
                (self.corner_sequences['Match ID'].isin(
                    self.team_stats[self.team_stats['Team Name'] == team_filter]['Match ID'].unique()
                )) & (self.corner_sequences['Team Name'] != team_filter)
            ]
            print(f"\nCórners de {team_filter}: {len(team_corners)}\nCórners de rivales: {len(rival_corners)}")
            if not team_corners.empty:
                print(f"\nResultados de córners de {team_filter}:")
                print(team_corners['result_type'].value_counts())
        
        print("\nDistribución por lado:\n", self.corner_sequences['corner_side'].value_counts())

def seleccionar_equipo_interactivo():
    try:
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/abp_events.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        if not equipos: print("No se encontraron equipos."); return None
        
        print("\n=== SELECCIÓN DE EQUIPO ===")
        for i, equipo in enumerate(equipos, 1): print(f"{i}. {equipo}")
        
        while True:
            try:
                indice = int(input(f"\nSelecciona un equipo (1-{len(equipos)}): ").strip()) - 1
                if 0 <= indice < len(equipos): return equipos[indice]
                else: print(f"Por favor, ingresa un número entre 1 y {len(equipos)}")
            except ValueError: print("Por favor, ingresa un número válido")
    except Exception as e: print(f"Error en la selección: {e}"); return None

def main():
    try:
        print("=== GENERADOR DE REPORTES DE ANÁLISIS DE SECUENCIAS DE CÓRNERS ===")
        if (equipo := seleccionar_equipo_interactivo()) is None:
            print("No se pudo completar la selección."); return
        
        print(f"\nGenerando reporte para {equipo}")
        analyzer = CornersSequenceAnalysis(team_filter=equipo)
        analyzer.print_summary(team_filter=equipo)
        
        if (fig := analyzer.create_corners_visualization(team_filter=equipo)):
            plt.show()
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_secuencias_corners_{equipo_filename}.pdf"
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
        analyzer = CornersSequenceAnalysis(team_filter=equipo)
        analyzer.print_summary(team_filter=equipo)
        fig = analyzer.create_corners_visualization(team_filter=equipo)
        
        if fig:
            if mostrar: plt.show()
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"reporte_secuencias_corners_{equipo_filename}.pdf"
                analyzer.guardar_sin_espacios(fig, output_path)
                print(f"✅ Reporte guardado como: {output_path}")
            return fig
        else:
            print("❌ No se pudo generar la visualización"); return None
            
    except Exception as e:
        print(f"❌ Error: {e}"); import traceback; traceback.print_exc(); return None

def verificar_assets():
    print("\n=== VERIFICACIÓN DE ASSETS ===")
    os.makedirs('assets/escudos', exist_ok=True)
    files_to_check = [
        'extraccion_opta/datos_opta_parquet/abp_events.parquet',
        'extraccion_opta/datos_opta_parquet/team_stats.parquet',
        'assets/fondo_informes.png', 'assets/balon.png'
    ]
    for file_path in files_to_check:
        print(f"✅ Encontrado: {file_path}" if os.path.exists(file_path) else f"❌ Faltante: {file_path}")
    if os.path.exists('assets/escudos') and (escudos := [f for f in os.listdir('assets/escudos') if f.endswith('.png')]):
        print(f"✅ Escudos disponibles ({len(escudos)}): {escudos[:5]}...")
    else:
        print("⚠️  No hay escudos en el directorio")

if __name__ == "__main__":
    print("=== INICIALIZANDO GENERADOR DE REPORTES DE SECUENCIAS DE CÓRNERS ===")
    try:
        verificar_assets()
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/abp_events.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        print(f"\n✅ Sistema listo. Equipos disponibles: {len(equipos)}")
        if equipos:
            print("📝 Para generar un reporte ejecuta: main()")
            print("📝 Para uso directo: generar_reporte_personalizado('Nombre_Equipo')")
    except Exception as e:
        print(f"❌ Error al inicializar: {e}")
    
    main()