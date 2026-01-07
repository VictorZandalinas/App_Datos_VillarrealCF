import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import patheffects
import seaborn as sns
import numpy as np
import re
import os
from mplsoccer import VerticalPitch
from difflib import SequenceMatcher
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Imports para Sankey
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from PIL import Image
    import io
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly no disponible. Se usará versión simplificada de Sankey.")

class CornersOffensiveReport:
    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/abp_events.parquet", team_filter=None):
        self.data_path = data_path
        self.team_filter = team_filter
        self.df = None
        self.corner_data = pd.DataFrame()
        self.team_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/team_stats.parquet")
        self.player_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/player_stats.parquet")

        # Diccionario central para los nombres de las zonas. ¡Esta es la clave!
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
            self.extract_corner_data(team_filter)

    def normalize_timestamp(self, timestamp):
        """Normaliza timestamps quitando la Z final si existe (copiado de ABP2)"""
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
    
    def get_zonas_de_caida_data(self):
        """
        NUEVA FUNCIÓN: Cuenta la zona de caída de TODOS los lanzamientos de córner 
        desde la izquierda (y > 99) usando Pass End X/Y.
        """
        # Obtener todos los lanzamientos de córner desde la izquierda
        lanzamientos = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] > 99)
        ]

        # Inicializar el contador para todas las zonas definidas
        zonas_count = {zona_key: 0 for zona_key in self.ZONA_NAMES.keys()}

        # Iterar sobre CADA lanzamiento
        for _, corner in lanzamientos.iterrows():
            pass_end_x = corner['Pass End X']
            pass_end_y = corner['Pass End Y']
            
            # Obtener la zona de destino del PASE
            zona = self.get_zona_from_coordinates(pass_end_x, pass_end_y)
            
            # Si se encuentra una zona válida, se incrementa el contador
            if zona in zonas_count:
                zonas_count[zona] += 1
        
        return zonas_count

    def debug_remates_coordenadas(self):
        """Debug específico para ver remates, coordenadas y zonas"""
        print("\n" + "="*60)
        print("DEBUG REMATES - COORDENADAS Y ZONAS")
        print("="*60)
        
        # Obtener córners desde y < 1 (DERECHA)
        corners_y1 = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] < 1)
        ]
        
        # Obtener todos los remates
        remates = self.corner_data[
            (self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])) &
            (self.corner_data['From corner'] == 'Sí')
        ]
        
        print(f"📍 CÓRNERS DESDE Y < 1: {len(corners_y1)}")
        print(f"🎯 REMATES TOTALES: {len(remates)}")
        
        contador_zona = {}
        remates_validos = 0
        
        for idx, remate in remates.iterrows():
            remate_time = remate['timeMin'] * 60 + remate['timeSec']
            
            # Buscar córner previo
            corner_previo = corners_y1[
                (corners_y1['Match ID'] == remate['Match ID']) &
                (corners_y1['timeMin'] * 60 + corners_y1['timeSec'] >= remate_time - 4) &
                (corners_y1['timeMin'] * 60 + corners_y1['timeSec'] < remate_time)
            ]
            
            if not corner_previo.empty:
                remates_validos += 1
                zona = self.get_zona_from_coordinates(remate['x'], remate['y'])
                zona_nombre = self.ZONA_NAMES.get(zona, 'FUERA DE ZONA')
                
                contador_zona[zona_nombre] = contador_zona.get(zona_nombre, 0) + 1
                
                print(f"\nREMATE {remates_validos}:")
                print(f"  Rematador: {remate['playerName']}")
                print(f"  Coordenadas: ({remate['x']:.1f}, {remate['y']:.1f})")
                print(f"  Zona: {zona_nombre}")
                print(f"  Tipo: {remate['Event Name']}")
                print(f"  Tiempo: {remate['timeMin']}:{remate['timeSec']:02d}")
                print(f"  Match ID: {remate['Match ID']}")
        
        print(f"\n📊 RESUMEN POR ZONAS:")
        for zona, count in sorted(contador_zona.items(), key=lambda x: x[1], reverse=True):
            print(f"  {zona}: {count} remates")
        
        print(f"\n✅ TOTAL REMATES VÁLIDOS: {remates_validos}")
        return contador_zona

    def get_aerial_duels_data(self):
        """Obtiene jugadores que ganan duelos aéreos tras corners desde y > 99"""
        
        # Obtener pases de córner desde y > 99
        corners_y99 = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] > 99)
        ]
        
        # Obtener duelos aéreos ganados
        aerials_won = self.corner_data[
            (self.corner_data['Event Name'] == 'Aerial') &
            (self.corner_data['outcome'] == 1)
        ]
        
        aerial_players = []
        
        for _, corner in corners_y99.iterrows():
            corner_time = corner['timeMin'] * 60 + corner['timeSec']
            
            # Buscar duelos aéreos ganados en los siguientes 15 segundos
            matching_aerials = aerials_won[
                (aerials_won['Team ID'] == corner['Team ID']) &
                (aerials_won['Match ID'] == corner['Match ID']) &
                (aerials_won['timeMin'] * 60 + aerials_won['timeSec'] > corner_time) &
                (aerials_won['timeMin'] * 60 + aerials_won['timeSec'] <= corner_time + 15)
            ]
            
            for _, aerial in matching_aerials.iterrows():
                aerial_players.append(aerial['playerName'])
        
        # Contar duelos aéreos ganados por jugador
        from collections import Counter
        return Counter(aerial_players)
    
    def get_aerial_duels_data_completo(self):
        """Obtiene datos completos de duelos aéreos tras corners desde y > 99 (versión simplificada)"""
        
        # Obtener córners desde y > 99
        corners_y99 = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] > 99)
        ]
        
        if corners_y99.empty:
            return {}
        
        # Obtener TODOS los duelos aéreos
        aerials_all = self.corner_data[self.corner_data['Event Name'] == 'Aerial']
        
        aerial_stats = {}  # {player_name: {'exitos': int, 'total': int}}
        
        print(f"DEBUG: Analizando {len(corners_y99)} córners desde y > 99")
        print(f"DEBUG: Total duelos aéreos disponibles: {len(aerials_all)}")
        
        for _, corner in corners_y99.iterrows():
            corner_time = corner['timeMin'] * 60 + corner['timeSec']
            
            # VENTANA TEMPORAL SIMPLE: 30 segundos después del córner
            matching_aerials = aerials_all[
                (aerials_all['Team ID'] == corner['Team ID']) &
                (aerials_all['Match ID'] == corner['Match ID']) &
                (aerials_all['timeMin'] * 60 + aerials_all['timeSec'] > corner_time) &
                (aerials_all['timeMin'] * 60 + aerials_all['timeSec'] <= corner_time + 30)  # 30 segundos
            ]
            
            print(f"DEBUG: Córner de {corner['playerName']} en {corner['timeMin']}:{corner['timeSec']} -> {len(matching_aerials)} duelos aéreos encontrados")
            
            # Contar duelos aéreos encontrados
            for _, aerial in matching_aerials.iterrows():
                player_name = aerial['playerName']
                
                if player_name not in aerial_stats:
                    aerial_stats[player_name] = {'exitos': 0, 'total': 0}
                
                aerial_stats[player_name]['total'] += 1
                
                if aerial['outcome'] == 1:  # Éxito
                    aerial_stats[player_name]['exitos'] += 1
                    print(f"DEBUG:   - {player_name}: duelo aéreo GANADO en {aerial['timeMin']}:{aerial['timeSec']}")
                else:
                    print(f"DEBUG:   - {player_name}: duelo aéreo perdido en {aerial['timeMin']}:{aerial['timeSec']}")
        
        print(f"DEBUG: Estadísticas finales de duelos aéreos: {aerial_stats}")
        return aerial_stats

    def get_aerial_ranking_scores(self):
        """Calcula ranking combinando remates primer contacto, duelos aéreos ganados y porcentaje"""
        
        # Obtener las 3 métricas
        remates_data = self.get_remates_primer_contacto_data()
        aerial_data = self.get_aerial_duels_data_completo()
        
        # Combinar jugadores de ambas métricas
        all_players = set(remates_data.keys()) | set(aerial_data.keys())
        
        ranking = []
        
        for player_name in all_players:
            # Métrica 1: Remates de primer contacto (AUMENTAR PESO)
            remates_primer = remates_data.get(player_name, 0)
            
            # Métrica 2 y 3: Duelos aéreos (REDUCIR PESO)
            aerial_stats = aerial_data.get(player_name, {'exitos': 0, 'total': 0})
            duelos_ganados = aerial_stats['exitos']
            duelos_total = aerial_stats['total']
            porcentaje_exito = duelos_ganados / duelos_total if duelos_total > 0 else 0
            
            # Solo incluir jugadores con al menos 1 actividad
            if remates_primer > 0 or duelos_total > 0:
                # Fórmula combinada - PRIORIDAD A REMATES DE PRIMERAS:
                peso_remates = 0.6    # ↑ Aumentado de 0.4 a 0.6
                peso_duelos = 0.2     # ↓ Reducido de 0.3 a 0.2  
                peso_porcentaje = 0.2 # ↓ Reducido de 0.3 a 0.2
                
                score = (remates_primer * peso_remates) + \
                    (duelos_ganados * peso_duelos) + \
                    (porcentaje_exito * 10 * peso_porcentaje)
                
                ranking.append({
                    'player_name': player_name,
                    'remates_primer_contacto': remates_primer,
                    'duelos_ganados': duelos_ganados,
                    'duelos_total': duelos_total,
                    'porcentaje_exito': porcentaje_exito,
                    'score': score
                })
        
        # Ordenar por score descendente
        ranking.sort(key=lambda x: x['score'], reverse=True)
        return ranking

    def get_remates_primer_contacto_data(self):
        """Obtiene remates de primer contacto directo (sin pases intermedios usando timeStamp)"""
        
        # Corners desde y > 99
        corners_y99 = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] > 99)
        ]
        
        # Remates
        remates = self.corner_data[
            (self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])) &
            (self.corner_data['From corner'] == 'Sí')
        ]
        
        remates_primer_contacto = {}
        
        for _, remate in remates.iterrows():
            remate_timestamp = pd.to_datetime(remate['timeStamp'])
            
            # Buscar corner previo en la ventana de 6 segundos
            corner_previo = corners_y99[
                (corners_y99['Match ID'] == remate['Match ID']) &
                (pd.to_datetime(corners_y99['timeStamp']) < remate_timestamp) &
                (pd.to_datetime(corners_y99['timeStamp']) >= remate_timestamp - pd.Timedelta(seconds=6))
            ]
            
            if not corner_previo.empty:
                # Tomar el corner más cercano al remate
                corner = corner_previo.iloc[-1]  # El último (más reciente)
                corner_timestamp = pd.to_datetime(corner['timeStamp'])
                
                # Verificar que NO haya otros pases del mismo equipo entre el corner y el remate
                pases_intermedios = self.corner_data[
                    (self.corner_data['Event Name'] == 'Pass') &
                    (self.corner_data['Team ID'] == remate['Team ID']) &
                    (self.corner_data['Match ID'] == remate['Match ID']) &
                    (pd.to_datetime(self.corner_data['timeStamp']) > corner_timestamp) &
                    (pd.to_datetime(self.corner_data['timeStamp']) < remate_timestamp) &
                    (self.corner_data['timeStamp'] != corner['timeStamp'])  # Excluir el corner original
                ]
                
                # Solo contar si NO hay pases intermedios
                if pases_intermedios.empty:
                    player_name = remate['playerName']
                    remates_primer_contacto[player_name] = remates_primer_contacto.get(player_name, 0) + 1
        
        return remates_primer_contacto

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
            
            # Contar pases del mismo equipo
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
                    # Para goles, verificar que no sea el mismo jugador
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
                
        # --- Determinar resultado final ---
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
    
    def create_aerial_ranking(self, ax):
        """Crea el ranking visual moderno de jugadores poderosos en juego aéreo"""
        
        aerial_ranking = self.get_aerial_ranking_scores()
        photos_data = self.load_player_photos()
        
        if not aerial_ranking:
            ax.text(0.5, 0.5, 'RANKING JUEGO AÉREO\n\n(Sin datos suficientes)', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, fontweight='bold')
            ax.axis('off')
            return
        
        # Configuración moderna
        ax.set_facecolor('#f8f9fa')  # Fondo gris claro
        top_players = aerial_ranking[:5]  # Máximo 5 jugadores
        
        # Título
        ax.text(0.5, 1.02, 'RANKING JUEGO AÉREO/1ER CONTACTO', fontsize=8, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes, color='#2c3e50')
        
        # Colores modernos para cada posición
        position_colors = ['#e74c3c', '#f39c12', '#f1c40f', '#95a5a6', '#bdc3c7', '#34495e']
        
        for i, player_data in enumerate(top_players):
            player_name = player_data['player_name']
            tasa_exito = player_data['porcentaje_exito'] # Se mantiene por si se usa en otro lado
            score = player_data['score']
            
            # Posición Y
            y_pos = 0.88 - (i * 0.18)
            
            # Rectángulo de fondo
            rect_bg = patches.FancyBboxPatch((0.01, y_pos - 0.06), 0.97, 0.14,
                                            boxstyle="round,pad=0.01", 
                                            facecolor='white',
                                            edgecolor=position_colors[i],
                                            linewidth=2,
                                            alpha=0.9)
            ax.add_patch(rect_bg)
            
            # Número de posición
            ax.text(0.08, y_pos, f"#{i+1}", fontsize=14, fontweight='bold', 
                    va='center', ha='center', color=position_colors[i])
            
            # Foto del jugador
            player_photo = self.get_player_photo_without_dorsal(player_name, photos_data)
            if player_photo is not None:
                photo_ax = ax.inset_axes([0.11, y_pos - 0.068, 0.22, 0.14])
                photo_ax.imshow(player_photo, aspect='auto')
                photo_ax.axis('off')
            
            # Nombre del jugador
            ax.text(0.32, y_pos + 0.055, player_name, fontsize=7, fontweight='bold', 
                    va='center', ha='left', color='#2c3e50')

            # Métrica 1: Remates primer contacto
            remates_primer = player_data['remates_primer_contacto']
            ax.text(0.32, y_pos + 0.020, f'Remates 1er contacto: {remates_primer}', 
                    fontsize=5, va='center', ha='left', color='#e74c3c')

            # Métrica 2: Duelos aéreos
            duelos_ganados = player_data['duelos_ganados']
            duelos_total = player_data['duelos_total']
            ax.text(0.32, y_pos - 0.005, f'Duelos aéreos: {duelos_ganados}/{duelos_total}', 
                    fontsize=5, va='center', ha='left', color='#3498db')

            # Métrica 3: Porcentaje de éxito
            porcentaje = player_data['porcentaje_exito']
            ax.text(0.32, y_pos - 0.030, f'% Éxito: {porcentaje:.1%}', 
                    fontsize=5, va='center', ha='left', color='#27ae60')

            # Score total
            ax.text(0.32, y_pos - 0.055, f'Score: {score:.1f}', 
                    fontsize=5, va='center', ha='left', color='#2c3e50', fontweight='bold')
            
            # === CAMBIO CLAVE AQUÍ ===
            # Sistema de estrellas basado en la posición en el ranking (índice 'i')
            # El 1º (i=0) tiene 5 estrellas, el 2º (i=1) tiene 4, etc.
            stars = 5 - i
            
            if stars > 0:
                star_text = '★' * stars
                ax.text(0.85, y_pos, star_text, fontsize=10, va='center', ha='center',
                        color='#f39c12')
        
        # Configuración final
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Recuadro
        ax.add_patch(patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.01",
                                        facecolor='none', edgecolor='#bdc3c7', 
                                        linewidth=1, alpha=0.3))

    def match_player_name(self, player_name, photos_data):
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

        print(f"🔍 Buscando foto para: '{player_name}' (Normalizado: '{player_parts['full']}')")
        
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
                print(f"✅ Match potencial: '{photo_name}' (score: {score:.3f}) - Razón: {reason}")
                
        # --- Lógica de desambiguación ---
        if len(found_matches) == 1:
            best_match = found_matches[0]
            print(f"🎯 MATCH ÚNICO Y VÁLIDO ENCONTRADO: '{best_match['entry']['player_name']}' con score {best_match['score']:.3f}")
            return best_match['entry']
        
        elif len(found_matches) > 1:
            # Si hay múltiples matches con score alto, es ambiguo.
            print(f"⚠️  ADVERTENCIA: Se encontraron {len(found_matches)} matches de alta calidad para '{player_name}'. Se descarta por ambigüedad.")
            for match in sorted(found_matches, key=lambda x: x['score'], reverse=True):
                print(f"  - Candidato: '{match['entry']['player_name']}' (Score: {match['score']:.2f}, Razón: {match['reason']})")
            return None
            
        else:
            print(f"❌ NO SE ENCONTRÓ UN MATCH DE ALTA CONFIANZA para '{player_name}'")
            return None

    def get_player_photo(self, player_name, photos_data):
        """Obtiene la foto en base64 de un jugador"""
        import base64
        from io import BytesIO
        from PIL import Image
        import numpy as np
        
        match = self.match_player_name(player_name, photos_data)
        if match:
            try:
                # Decodificar base64 y convertir a array numpy
                img_data = base64.b64decode(match['image_base64'])
                img = Image.open(BytesIO(img_data))
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                return np.array(img) / 255.0
            except Exception as e:
                print(f"⚠️ Error cargando foto de {player_name}: {e}")
        
        return None
    
    def load_data(self, team_filter=None):
        """Carga datos necesarios para el análisis de córneres ofensivos"""
        try:
            # Columnas necesarias incluyendo las nuevas columnas requeridas
            columns_needed = ['Match ID', 'periodId', 'Team ID', 'Team Name', 'Event Name', 'outcome', 
                            'timeMin', 'timeSec', 'x', 'y', 'Pass End X', 'Pass End Y', 
                            'playerName', 'playerId', 'Corner taken', 'From corner',
                            'In-swinger', 'Out-swinger', 'Straight', 'Left footed', 'Right footed', 'Cross',
                            'Throw in', 'Free kick taken', 'timeStamp']  # ← AÑADIR ESTAS COLUMNAS
                        
            # Intentar cargar todas las columnas, si alguna no existe, continuar sin ella
            try:
                self.df = pd.read_parquet(self.data_path, columns=columns_needed)
            except Exception:
                # Si falla, cargar las columnas básicas
                basic_columns = ['Match ID', 'Team ID', 'Team Name', 'Event Name', 'outcome', 
                            'timeMin', 'timeSec', 'x', 'y', 'Pass End X', 'Pass End Y', 
                            'playerName', 'playerId']
                self.df = pd.read_parquet(self.data_path, columns=basic_columns)
                # Añadir columnas faltantes con valores por defecto
                for col in ['Corner taken', 'From corner', 'In-swinger', 'Out-swinger', 'Straight', 'Cross', 'timeStamp', 'periodId', 'Throw in', 'Free kick taken']:
                    if col not in self.df.columns:
                        if col == 'timeStamp':
                            self.df[col] = pd.NaT
                        elif col == 'periodId':
                            self.df[col] = 1
                        else:
                            self.df[col] = 'No'
            
            # ← AÑADIR ESTA LÍNEA: Aplicar la normalización del timestamp
            if 'timeStamp' in self.df.columns:
                self.df['timeStamp'] = self.df['timeStamp'].apply(self.normalize_timestamp)
            
            # Filtrar eventos relevantes
            relevant_events = ['Pass', 'Goal', 'Attempt Saved', 'Miss', 'Post', 'Aerial']
            self.df = self.df[self.df['Event Name'].isin(relevant_events)]
            
            # Si hay filtro de equipo, filtrar matches
            if team_filter:
                team_matches = self.team_stats[self.team_stats['Team Name'] == team_filter]['Match ID'].unique()
                self.df = self.df[self.df['Match ID'].isin(team_matches)]
            
            print(f"✅ Datos cargados: {len(self.df)} eventos totales")
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")
    
    def extract_corner_data(self, team_filter):
        """Extrae datos específicos de córneres ofensivos"""
        if self.df is None:
            print("❌ No hay datos cargados")
            return
        
        print("🔍 Extrayendo datos de córneres ofensivos...")
        
        # Filtrar solo datos del equipo seleccionado
        team_data = self.df[self.df['Team Name'] == team_filter].copy()
        
        # Ordenar por tiempo
        team_data = team_data.sort_values(['Match ID', 'timeMin', 'timeSec']).reset_index(drop=True)
        
        self.corner_data = team_data
        print(f"✅ Datos de córneres extraídos: {len(self.corner_data)} eventos")
    
    def get_lanzadores_data(self):
        """Obtiene datos de lanzadores de córner SOLO desde y > 99"""
        lanzadores = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] > 99) 
        ]
        return lanzadores['playerName'].value_counts()
    
    def get_pierna_lanzador_principal(self):
        lanzadores_data = self.get_lanzadores_data()
        if lanzadores_data.empty:
            return "N/A"
        
        lanzador_principal = lanzadores_data.index[0]
        lanzador_pases = self.corner_data[
            (self.corner_data['playerName'] == lanzador_principal) &
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['Cross'] == 'Sí')
        ]
        
        if lanzador_pases['Right footed'].eq('Sí').any():
            return "DERECHA"
        elif lanzador_pases['Left footed'].eq('Sí').any():
            return "IZQUIERDA"
        else:
            return "N/A"
    
    def get_golpeo_mas_comun(self):
        golpeos = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['Cross'] == 'Sí')
        ]
        
        if golpeos.empty:
            return "N/A"
        
        # Contar cuántas veces cada tipo tiene "Sí"
        cerrados = len(golpeos[golpeos['In-swinger'] == 'Sí'])
        abiertos = len(golpeos[golpeos['Out-swinger'] == 'Sí'])
        planos = len(golpeos[golpeos['Straight'] == 'Sí'])
        
        # Encontrar el máximo y devolver la traducción
        golpeo_counts = {'CERRADO': cerrados, 'ABIERTO': abiertos, 'PLANO': planos}
        
        if max(golpeo_counts.values()) == 0:
            return "N/A"
        
        return max(golpeo_counts, key=golpeo_counts.get)
    
    def get_zona_mas_comun(self):
        zonas_data = self.get_zonas_de_caida_data()
        if not zonas_data:
            return "N/A"
        
        zona_max = max(zonas_data, key=zonas_data.get)
        # Usar el diccionario centralizado en lugar de uno local
        return self.ZONA_NAMES.get(zona_max, "N/A") 
    
    def debug_aerial_complete(self):
        """Debug completo para eventos Aerial en todo el parquet"""
        print("\n" + "="*60)
        print("DEBUG COMPLETO - EVENTOS AERIAL EN TODO EL DATASET")
        print("="*60)
        
        # 1. CARGAR DATOS COMPLETOS (SIN FILTRO DE EQUIPO)
        try:
            df_completo = pd.read_parquet("extraccion_opta/datos_opta_parquet/abp_events.parquet")
            print(f"📊 TOTAL EVENTOS EN PARQUET: {len(df_completo)}")
            
            # Ver todos los Event Name únicos
            print(f"\n📋 TODOS LOS EVENT NAME:")
            event_names = df_completo['Event Name'].value_counts()
            for event, count in event_names.items():
                print(f"  - {event}: {count}")
            
            # 2. BUSCAR EVENTOS AERIAL
            aerials_all = df_completo[df_completo['Event Name'] == 'Aerial']
            print(f"\n⚽ EVENTOS AERIAL TOTALES: {len(aerials_all)}")
            
            if len(aerials_all) > 0:
                print(f"Valores únicos en 'outcome' para Aerial: {aerials_all['outcome'].unique()}")
                print(f"Distribución de outcomes:")
                print(aerials_all['outcome'].value_counts())
                
                # Ver algunos ejemplos
                print(f"\n📋 MUESTRA DE EVENTOS AERIAL:")
                for i, row in aerials_all.head(5).iterrows():
                    print(f"  - {row['playerName']} | Team: {row['Team Name']} | Outcome: {row['outcome']} | Tiempo: {row['timeMin']}:{row['timeSec']:02d}")
            
            # 3. VERIFICAR EVENTOS AERIAL PARA NUESTRO EQUIPO
            aerials_equipo = aerials_all[aerials_all['Team Name'] == self.team_filter]
            print(f"\n🎯 EVENTOS AERIAL PARA {self.team_filter}: {len(aerials_equipo)}")
            
            if len(aerials_equipo) > 0:
                print(f"Distribución de outcomes para {self.team_filter}:")
                print(aerials_equipo['outcome'].value_counts())
                
                print(f"\n📋 EVENTOS AERIAL DE {self.team_filter}:")
                for i, row in aerials_equipo.iterrows():
                    print(f"  - {row['playerName']} | Outcome: {row['outcome']} | Match: {row['Match ID']} | Tiempo: {row['timeMin']}:{row['timeSec']:02d}")
            
            # 4. VERIFICAR SI ESTÁN EN self.corner_data
            aerials_filtrados = self.corner_data[self.corner_data['Event Name'] == 'Aerial']
            print(f"\n🔍 EVENTOS AERIAL EN DATOS FILTRADOS (self.corner_data): {len(aerials_filtrados)}")
            
            return aerials_all, aerials_equipo
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None
    
    def debug_secuencias_completo(self):
        """Debug completo para ver secuencias, tipos de lanzamiento y remates"""
        print("\n" + "="*60)
        print("DEBUG COMPLETO - SECUENCIAS Y REMATES")
        print("="*60)
        
        # 1. CÓRNERS DESDE X > 99
        corners_x99 = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['x'] > 99)
        ]
        
        print(f"\n📍 CÓRNERS DESDE X > 99: {len(corners_x99)}")
        print("-" * 40)
        
        if not corners_x99.empty:
            for idx, corner in corners_x99.iterrows():
                print(f"\nCÓRNER {idx + 1}:")
                print(f"  Lanzador: {corner['playerName']}")
                print(f"  Coordenadas: ({corner['x']:.1f}, {corner['y']:.1f})")
                print(f"  Minuto: {corner['timeMin']}:{corner['timeSec']:02d}")
                
                # Tipo de golpeo
                if corner['In-swinger'] == 'Sí':
                    tipo_golpeo = "CERRADO (In-swinger)"
                elif corner['Out-swinger'] == 'Sí':
                    tipo_golpeo = "ABIERTO (Out-swinger)"
                elif corner['Straight'] == 'Sí':
                    tipo_golpeo = "PLANO (Straight)"
                else:
                    tipo_golpeo = "NO ESPECIFICADO"
                
                print(f"  Tipo de golpeo: {tipo_golpeo}")
                
                # Pierna
                if corner['Right footed'] == 'Sí':
                    pierna = "DERECHA"
                elif corner['Left footed'] == 'Sí':
                    pierna = "IZQUIERDA"
                else:
                    pierna = "NO ESPECIFICADO"
                
                print(f"  Pierna: {pierna}")
                
                # Buscar remates posteriores
                corner_time = corner['timeMin'] * 60 + corner['timeSec']
                remates_posteriores = self.corner_data[
                    (self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])) &
                    (self.corner_data['From corner'] == 'Sí') &
                    (self.corner_data['Match ID'] == corner['Match ID']) &
                    (self.corner_data['timeMin'] * 60 + self.corner_data['timeSec'] > corner_time) &
                    (self.corner_data['timeMin'] * 60 + self.corner_data['timeSec'] <= corner_time + 15)
                ]
                
                if not remates_posteriores.empty:
                    print(f"  🎯 REMATES POSTERIORES: {len(remates_posteriores)}")
                    for r_idx, remate in remates_posteriores.iterrows():
                        zona = self.get_zona_from_coordinates(remate['x'], remate['y'])
                        zona_nombre = {
                            'zona_1': 'ZONA IZQUIERDA',
                            'zona_2': 'BORDE AREA CENTRO', 
                            'zona_3': 'AREA PEQ IZQ',
                            'zona_4': 'PORTERIA-AREA PEQ',
                            'zona_5': 'AREA PEQ DER',
                            'zona_6': 'PENALTI',
                            'zona_7': 'ZONA DERECHA'
                        }.get(zona, 'FUERA DE ZONA')
                        
                        tiempo_diferencia = (remate['timeMin'] * 60 + remate['timeSec']) - corner_time
                        
                        print(f"    - Rematador: {remate['playerName']}")
                        print(f"      Coordenadas: ({remate['x']:.1f}, {remate['y']:.1f})")
                        print(f"      Zona: {zona_nombre}")
                        print(f"      Tipo: {remate['Event Name']}")
                        print(f"      Tiempo después: {tiempo_diferencia:.1f}s")
                        print()
                else:
                    print("  ❌ Sin remates posteriores detectados")
        
        # 2. RESUMEN DE TIPOS DE LANZAMIENTO
        print("\n" + "="*40)
        print("RESUMEN TIPOS DE LANZAMIENTO")
        print("="*40)
        
        tipos_data = self.get_tipo_lanzamiento_data()
        for tipo, cantidad in tipos_data.items():
            print(f"{tipo}: {cantidad}")
        
        # 3. SECUENCIA MÁS REPETIDA
        print("\n" + "="*40)
        print("SECUENCIA MÁS REPETIDA")
        print("="*40)
        
        secuencia = self.get_secuencia_mas_repetida()
        print(f"Lanzador: {secuencia['lanzador']}")
        print(f"Zona: {secuencia['zona']}")
        print(f"Golpeo: {secuencia['golpeo']}")
        print(f"Pierna: {secuencia['pierna']}")
        
        # 4. TODAS LAS SECUENCIAS (CONTEO)
        print("\n" + "="*40)
        print("TODAS LAS SECUENCIAS ENCONTRADAS")
        print("="*40)
        
        # Recrear la lógica para mostrar todas las secuencias
        secuencias = []
        
        for _, corner in corners_x99.iterrows():
            corner_time = corner['timeMin'] * 60 + corner['timeSec']
            
            remates_posteriores = self.corner_data[
                (self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])) &
                (self.corner_data['From corner'] == 'Sí') &
                (self.corner_data['Match ID'] == corner['Match ID']) &
                (self.corner_data['timeMin'] * 60 + self.corner_data['timeSec'] > corner_time) &
                (self.corner_data['timeMin'] * 60 + self.corner_data['timeSec'] <= corner_time + 15)
            ]
            
            if not remates_posteriores.empty:
                primer_remate = remates_posteriores.iloc[0]
                
                # Obtener datos de la secuencia
                lanzador = corner['playerName']
                
                zona = self.get_zona_from_coordinates(primer_remate['x'], primer_remate['y'])
                zona_nombre = {
                    'zona_1': 'ZONA IZQUIERDA',
                    'zona_2': 'BORDE AREA CENTRO', 
                    'zona_3': 'AREA PEQ IZQ',
                    'zona_4': 'PORTERIA-AREA PEQ',
                    'zona_5': 'AREA PEQ DER',
                    'zona_6': 'PENALTI',
                    'zona_7': 'ZONA DERECHA'
                }.get(zona, 'N/A')
                
                if corner['In-swinger'] == 'Sí':
                    golpeo = 'CERRADO'
                elif corner['Out-swinger'] == 'Sí':
                    golpeo = 'ABIERTO'
                elif corner['Straight'] == 'Sí':
                    golpeo = 'PLANO'
                else:
                    golpeo = 'N/A'
                
                if corner['Right footed'] == 'Sí':
                    pierna = 'DERECHA'
                elif corner['Left footed'] == 'Sí':
                    pierna = 'IZQUIERDA'
                else:
                    pierna = 'N/A'
                
                secuencia = (lanzador, zona_nombre, golpeo, pierna)
                secuencias.append(secuencia)
        
        # Contar secuencias y obtener la más repetida
        from collections import Counter
        contador = Counter(secuencias)

        # Obtener el máximo número de repeticiones
        max_count = contador.most_common(1)[0][1]

        # Filtrar todas las secuencias que tienen el máximo número de repeticiones
        secuencias_empatadas = [seq for seq, count in contador.items() if count == max_count]

        # Si hay solo una secuencia ganadora, devolverla normalmente
        if len(secuencias_empatadas) == 1:
            secuencia_ganadora = secuencias_empatadas[0]
            print(f"🏆 SECUENCIA GANADORA: {secuencia_ganadora[0]} ({max_count} repeticiones)")
            
            return {
                'lanzador': secuencia_ganadora[0],
                'zona': secuencia_ganadora[1], 
                'golpeo': secuencia_ganadora[2],
                'pierna': secuencia_ganadora[3]
            }

        # Si hay empate, mostrar todas las opciones
        else:
            print(f"🎯 EMPATE DETECTADO: {len(secuencias_empatadas)} secuencias con {max_count} repeticiones cada una")
            print("🏆 TODAS LAS OPCIONES EMPATADAS:")
            
            for i, seq in enumerate(secuencias_empatadas, 1):
                print(f"  {i}. {seq[0]} → {seq[1]} → {seq[2]} → {seq[3]}")
            
            # Recopilar todas las opciones por categoría
            lanzadores = list(set(seq[0] for seq in secuencias_empatadas))
            zonas = list(set(seq[1] for seq in secuencias_empatadas))
            golpeos = list(set(seq[2] for seq in secuencias_empatadas))
            piernas = list(set(seq[3] for seq in secuencias_empatadas))
            
            # Formatear para mostrar múltiples opciones
            return {
                'lanzador': ' / '.join(sorted(lanzadores)),
                'zona': ' / '.join(sorted(zonas)) if len(zonas) > 1 else zonas[0],
                'golpeo': ' / '.join(sorted(golpeos)) if len(golpeos) > 1 else golpeos[0],
                'pierna': ' / '.join(sorted(piernas)) if len(piernas) > 1 else piernas[0]
            }

    def get_rematadores_data(self):
        """Obtiene datos de rematadores usando LA MISMA LÓGICA QUE EL MAPA DE CALOR"""
        
        # Obtener eventos de remate con From corner = 'Sí'
        remates = self.corner_data[
            (self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])) &
            (self.corner_data['From corner'] == 'Sí')
        ]
        
        # Obtener pases de córner SOLO desde y > 99
        pases_corner = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] > 99)
        ]
        
        rematadores_validos = []
        
        # USAR LA MISMA LÓGICA QUE get_primer_contacto_data()
        for _, remate in remates.iterrows():
            remate_time = remate['timeMin'] * 60 + remate['timeSec']
            
            # Buscar si hay un pase de córner desde y > 99 en los 4 segundos anteriores
            pase_previo = pases_corner[
                (pases_corner['Match ID'] == remate['Match ID']) &
                (pases_corner['timeMin'] * 60 + pases_corner['timeSec'] >= remate_time - 5) &
                (pases_corner['timeMin'] * 60 + pases_corner['timeSec'] < remate_time)
            ]
            
            if not pase_previo.empty:
                rematadores_validos.append(remate['playerName'])
        
        return pd.Series(rematadores_validos).value_counts()
    
    def get_tipo_lanzamiento_data(self):
        """Obtiene datos de tipos de lanzamiento, incluyendo los no especificados."""
        lanzamientos = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] > 99)
        ]

        # Contar los tipos conocidos
        cerrados = len(lanzamientos[lanzamientos['In-swinger'] == 'Sí'])
        abiertos = len(lanzamientos[lanzamientos['Out-swinger'] == 'Sí'])
        planos = len(lanzamientos[lanzamientos['Straight'] == 'Sí'])

        # EL CAMBIO CLAVE: Calcular los no especificados
        total_lanzamientos = len(lanzamientos)
        tipos_conocidos = cerrados + abiertos + planos
        sin_tipo = total_lanzamientos - tipos_conocidos
        
        return {'Cerrados': cerrados, 'Abiertos': abiertos, 'Planos': planos, 'Sin Tipo': sin_tipo}
    
    def get_primer_contacto_data(self):
        """
        LÓGICA SINCRONIZADA CON ABP7.2:
        Busca el resultado principal (prioridad Gol > Tiro) por cada saque de esquina.
        """
        primer_contacto_list = []
        
        # 1. Obtener los saques de esquina del lado correspondiente
        # Para abp8: usar y > 99 | Para abp9: usar y < 1
        lado_condicion = self.corner_data['y'] > 99 if hasattr(self, 'ZONA_NAMES') and "IZQUIERDA" in self.ZONA_NAMES['zona_1'] else self.corner_data['y'] < 1
        
        saques = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') & 
            (self.corner_data['Corner taken'] == 'Sí') & 
            lado_condicion
        ].copy()

        # 2. Por cada saque, buscar la secuencia igual que en abp7.2
        for _, saque in saques.iterrows():
            match_id = saque['Match ID']
            # Obtener eventos del mismo partido posteriores al saque
            secuencia = self.corner_data[
                (self.corner_data['Match ID'] == match_id) & 
                (self.corner_data['timeStamp'] > saque['timeStamp'])
            ].sort_values('timeStamp').head(10) # Miramos los siguientes 10 eventos
            
            saque_dt = pd.to_datetime(saque['timeStamp'])
            
            # Buscamos el mejor resultado en la ventana de 5 segundos (igual que 7.2)
            resultado_encontrado = None
            prioridad = {'Goal': 1, 'Attempt Saved': 2, 'Miss': 3, 'Post': 4}
            mejor_evento = None
            min_prioridad = 5

            for _, evento in secuencia.iterrows():
                ev_dt = pd.to_datetime(evento['timeStamp'])
                diff = (ev_dt - saque_dt).total_seconds()
                
                if diff > 5: break # Límite de abp7.2
                
                # Si el equipo es el mismo y es un remate
                if evento['Team ID'] == saque['Team ID'] and evento['Event Name'] in prioridad:
                    prio_actual = prioridad[evento['Event Name']]
                    if prio_actual < min_prioridad:
                        min_prioridad = prio_actual
                        mejor_evento = evento
            
            if mejor_evento is not None:
                primer_contacto_list.append({
                    'x': mejor_evento['x'],
                    'y': mejor_evento['y'],
                    'playerName': mejor_evento['playerName'],
                    'event_type': mejor_evento['Event Name']
                })
        
        return primer_contacto_list
    
    def get_zona_from_coordinates(self, x, y):
        """Determina la zona según las coordenadas x,y basándose en las 7 zonas nuevas"""
        x, y = float(x), float(y)
        
        # Filtrar remates fuera del área definida
        if x < 70:
            return None
        
        # PRIORIDAD: PENALTI tiene prioridad sobre BORDE AREA CENTRO en superposiciones
        # PENALTI: x: 83 --> 94.2, y: 42 --> 58
        if x >= 83 and x <= 94.2 and y >= 42 and y <= 58:
            return 'zona_6'
        
        # ZONA IZQUIERDA: x: 70 --> 100, y: 75 --> 100
        elif x >= 70 and x <= 100 and y >= 75 and y <= 100:
            return 'zona_1'
        
        # BORDE AREA CENTRO: x: 70 --> 88.5, y: 25 --> 75
        elif x >= 70 and x <= 88.5 and y >= 25 and y <= 75:
            return 'zona_2'
        
        # AREA PEQ IZQ: x: 88.5 --> 100, y: 58 --> 75
        elif x >= 88.5 and x <= 100 and y >= 58 and y <= 75:
            return 'zona_3'
        
        # PORTERIA-AREA PEQ: x: 94.2 --> 100, y: 42 --> 58
        elif x >= 94.2 and x <= 100 and y >= 42 and y <= 58:
            return 'zona_4'
        
        # AREA PEQ DER: x: 88.5 --> 100, y: 25 --> 42
        elif x >= 88.5 and x <= 100 and y >= 25 and y <= 42:
            return 'zona_5'
        
        # ZONA DERECHA: x: 70 --> 100, y: 0 --> 25
        elif x >= 70 and x <= 100 and y >= 0 and y <= 25:
            return 'zona_7'
        
        else:
            return None
    
    def get_contactos_por_zona(self):
        """Cuenta primer contacto por zona"""
        primer_contacto = self.get_primer_contacto_data()
        zonas_count = {'zona_1': 0, 'zona_2': 0, 'zona_3': 0, 'zona_4': 0, 
                    'zona_5': 0, 'zona_6': 0, 'zona_7': 0}
    
        for contacto in primer_contacto:
            zona = self.get_zona_from_coordinates(contacto['x'], contacto['y'])
            if zona:  # Solo contar si la zona no es None
                zonas_count[zona] += 1
        
        return zonas_count
    
    def debug_corner_data(self):
        """Debug para verificar qué datos tenemos"""
        print("=== DEBUG CÓRNERS ===")
        print(f"Total eventos: {len(self.corner_data)}")
        
        # Verificar columnas
        print(f"Columnas disponibles: {list(self.corner_data.columns)}")
        
        # Verificar valores únicos en columnas clave
        if 'Corner taken' in self.corner_data.columns:
            print(f"Valores únicos en 'Corner taken': {self.corner_data['Corner taken'].unique()}")
            print(f"Eventos con Corner taken = 'Sí': {len(self.corner_data[self.corner_data['Corner taken'] == 'Sí'])}")
        
        if 'From corner' in self.corner_data.columns:
            print(f"Valores únicos en 'From corner': {self.corner_data['From corner'].unique()}")
            print(f"Eventos con From corner = 'Sí': {len(self.corner_data[self.corner_data['From corner'] == 'Sí'])}")
        
        # Verificar eventos de pase
        pases = self.corner_data[self.corner_data['Event Name'] == 'Pass']
        print(f"Total pases: {len(pases)}")
        
        # Verificar eventos de remate
        remates = self.corner_data[self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])]
        print(f"Total remates: {len(remates)}")
        
        return True
    
    def get_matriz_lanzador_rematador(self):
        """
        NUEVA LÓGICA: Crea una matriz de lanzador vs actor, donde el "actor" es quien
        realiza una acción clave (remate o duelo aéreo ganado) dentro de los 6 segundos
        posteriores al córner, usando timeStamp y deduplicando acciones en la misma jugada.
        """
        from collections import defaultdict

        # --- 1. PREPARACIÓN DE DATOS ---

        # Obtener los lanzamientos de córner desde la izquierda.
        # Es crucial usar .copy() para evitar advertencias de Pandas.
        lanzadores = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] > 99)
        ].copy()
        # Convertir la columna timeStamp a formato datetime para cálculos precisos.
        lanzadores['timeStamp'] = pd.to_datetime(lanzadores['timeStamp'])

        # Definir las "acciones clave" que queremos contar.
        # a) Todos los remates.
        remates = self.corner_data[
            self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])
        ]
        # b) Solo los duelos aéreos GANADOS (outcome == 1).
        duelos_aereos_ganados = self.corner_data[
            (self.corner_data['Event Name'] == 'Aerial') &
            (self.corner_data['outcome'] == 1)
        ]

        # Combinar ambos tipos de acciones en un único DataFrame.
        acciones_clave = pd.concat([remates, duelos_aereos_ganados]).copy()
        acciones_clave['timeStamp'] = pd.to_datetime(acciones_clave['timeStamp'])
        # Ordenar por tiempo es una buena práctica.
        acciones_clave = acciones_clave.sort_values('timeStamp')

        # --- 2. PROCESAMIENTO Y CONTEO ---

        # Usamos defaultdict para facilitar la creación de la matriz anidada.
        matriz = defaultdict(lambda: defaultdict(int))

        # Iterar sobre cada saque de córner.
        for _, corner in lanzadores.iterrows():
            lanzador = corner['playerName']
            corner_timestamp = corner['timeStamp']

            # Encontrar todas las acciones clave que ocurrieron en la ventana de 6 segundos.
            acciones_en_ventana = acciones_clave[
                (acciones_clave['Match ID'] == corner['Match ID']) &
                (acciones_clave['timeStamp'] > corner_timestamp) &
                (acciones_clave['timeStamp'] <= corner_timestamp + pd.Timedelta(seconds=6))
            ]

            # Si se encontraron acciones en esa ventana...
            if not acciones_en_ventana.empty:
                # --- LA LÓGICA DE DEDUPLICACIÓN ---
                # Obtenemos los nombres de los jugadores que realizaron acciones,
                # pero nos quedamos solo con los nombres ÚNICOS.
                # Si 'Jugador A' tuvo un 'Aerial' y un 'Miss' en la misma jugada,
                # su nombre solo aparecerá una vez aquí.
                actores_unicos_en_secuencia = acciones_en_ventana['playerName'].unique()
                
                # Incrementar el contador solo una vez por cada actor único en la secuencia.
                for actor in actores_unicos_en_secuencia:
                    matriz[lanzador][actor] += 1

        # --- 3. PREPARAR LISTAS PARA LA TABLA VISUAL ---

        # Obtener la lista de todos los lanzadores que tienen al menos una conexión.
        lanzadores_list = list(matriz.keys())

        # Crear una lista de todos los "actores" y ordenarla por el total de acciones
        # para que los más importantes aparezcan primero en la tabla.
        actor_totals = defaultdict(int)
        for lanzador in matriz:
            for actor, count in matriz[lanzador].items():
                actor_totals[actor] += count
        
        actores_list = sorted(actor_totals, key=actor_totals.get, reverse=True)

        return matriz, lanzadores_list, actores_list
    
    def load_player_photos(self):
        """Carga el JSON con las fotos de jugadores"""
        import json
        try:
            with open('assets/jugadores_optimizados.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ No se encontró el archivo jugadores_optimizados.json")
            return []

    def get_zona_caida_con_rematador(self):
        """Obtiene información de zona de caída y rematadores para cada lanzador"""
        
        # Obtener córners desde y > 99
        corners_izq = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] > 99)
        ]
        
        lanzador_info = {}
        
        for _, corner in corners_izq.iterrows():
            lanzador = corner['playerName']
            
            # Zona de caída del balón
            pass_end_x = corner['Pass End X']
            pass_end_y = corner['Pass End Y']
            
            try:
                x_coord = float(pass_end_x) if pd.notna(pass_end_x) else 0.0
                y_coord = float(pass_end_y) if pd.notna(pass_end_y) else 0.0
                coordenadas_text = f"({x_coord:.1f}, {y_coord:.1f})"
            except (ValueError, TypeError):
                coordenadas_text = "(0.0, 0.0)"
                x_coord = y_coord = 0.0

            zona_caida = self.get_zona_from_coordinates(x_coord, y_coord)
            zona_nombre = self.ZONA_NAMES.get(zona_caida, 'FUERA DE ZONA')
            
            # Buscar rematador en esa zona
            corner_time = corner['timeMin'] * 60 + corner['timeSec']
            
            remates_posteriores = self.corner_data[
                (self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])) &
                (self.corner_data['From corner'] == 'Sí') &
                (self.corner_data['Match ID'] == corner['Match ID']) &
                (self.corner_data['timeMin'] * 60 + self.corner_data['timeSec'] > corner_time) &
                (self.corner_data['timeMin'] * 60 + self.corner_data['timeSec'] <= corner_time + 4)
            ]
            
            # NUEVO: Determinar si hubo remate y quién fue
            if not remates_posteriores.empty:
                primer_remate = remates_posteriores.iloc[0]
                rematador = primer_remate['playerName']
                tuvo_remate = True
                tipo_remate = primer_remate['Event Name']  # Goal, Miss, etc.
            else:
                rematador = None
                tuvo_remate = False
                tipo_remate = None
            
            # Guardar información COMPLETA
            if lanzador not in lanzador_info:
                lanzador_info[lanzador] = []
            
            lanzador_info[lanzador].append({
                'zona_caida': zona_nombre,
                'coordenadas': coordenadas_text,
                'rematador': rematador,
                'tuvo_remate': tuvo_remate,          # NUEVO
                'tipo_remate': tipo_remate,          # NUEVO
                'minuto': f"{corner['timeMin']}:{corner['timeSec']:02d}"  # NUEVO
            })
        
        return lanzador_info
    
    def get_secuencia_mas_repetida(self):
        """
        NUEVA LÓGICA: Encuentra al lanzador más frecuente desde la izquierda (y > 99)
        y luego calcula SUS características más comunes (zona, golpeo, pierna).
        """
        # 1. Obtener todos los córners lanzados desde la izquierda.
        corners_izq = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] > 99)
        ]
        
        # Si no hay datos, devolver valores por defecto.
        if corners_izq.empty:
            return {'lanzador': 'N/A', 'zona': 'N/A', 'golpeo': 'N/A', 'pierna': 'N/A'}

        # 2. Encontrar al lanzador principal (el que más lanza desde este lado).
        lanzador_principal = corners_izq['playerName'].value_counts().index[0]
        
        # 3. Filtrar los datos para obtener solo los lanzamientos de ESE jugador.
        lanzador_data = corners_izq[corners_izq['playerName'] == lanzador_principal]

        # 4. Calcular la ZONA más frecuente para este lanzador.
        # Usamos apply para obtener la zona de cada lanzamiento y .mode() para la más común.
        zonas = lanzador_data.apply(
            lambda row: self.get_zona_from_coordinates(row['Pass End X'], row['Pass End Y']),
            axis=1
        ).dropna()
        zona_mas_comun_key = zonas.mode()[0] if not zonas.empty else 'N/A'
        zona_nombre = self.ZONA_NAMES.get(zona_mas_comun_key, 'N/A')

        # 5. Calcular el GOLPEO más frecuente para este lanzador.
        def get_golpeo(row):
            if row['In-swinger'] == 'Sí': return 'CERRADO'
            if row['Out-swinger'] == 'Sí': return 'ABIERTO'
            if row['Straight'] == 'Sí': return 'PLANO'
            return None # Devolvemos None para poder usar dropna()

        golpeos = lanzador_data.apply(get_golpeo, axis=1).dropna()
        golpeo_principal = golpeos.mode()[0] if not golpeos.empty else 'N/A'
        
        # 6. Calcular la PIERNA más frecuente para este lanzador.
        def get_pierna(row):
            if row['Right footed'] == 'Sí': return 'DERECHA'
            if row['Left footed'] == 'Sí': return 'IZQUIERDA'
            return None

        piernas = lanzador_data.apply(get_pierna, axis=1).dropna()
        pierna_principal = piernas.mode()[0] if not piernas.empty else 'N/A'

        # 7. Devolver el diccionario con el perfil del lanzador principal.
        return {
            'lanzador': lanzador_principal,
            'zona': zona_nombre, 
            'golpeo': golpeo_principal,
            'pierna': pierna_principal
        }

    # === NUEVAS FUNCIONES PARA SANKEY ===
    def get_sankey_data(self):
        """Prepara datos para gráfica Sankey: Golpeo → Lanzadores → Zonas → Rematadores"""
        
        # 1. Obtener córners desde y > 99
        corners_y99 = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] > 99)
        ]
        
        # 2. Preparar nodos en el orden correcto
        golpeo_nombres = ['Cerrados', 'Abiertos', 'Planos']  # CAMBIO: de tipos_nombres a golpeo_nombres
        
        # Top lanzadores
        lanzadores_data = self.get_lanzadores_data()
        top_lanzadores = list(lanzadores_data.head(5).index)
        
        # Zonas de caída (usando el diccionario centralizado para asegurar el orden y nombre correctos)
        zonas_nombres = list(self.ZONA_NAMES.values())

        
        # Top rematadores
        rematadores_data = self.get_rematadores_data()
        top_rematadores = list(rematadores_data.head(8).index)
        
        # 3. Labels combinados: Golpeo + Lanzadores + Zonas + Rematadores
        labels = golpeo_nombres + top_lanzadores + zonas_nombres + top_rematadores  # CAMBIO: golpeo_nombres
        
        # 4. Crear flujos
        source = []
        target = []
        value = []
        
        from collections import defaultdict
        
        # FLUJO 1: Golpeo → Lanzadores
        golpeo_lanzador_counts = defaultdict(lambda: defaultdict(int))  # CAMBIO: nombre variable
        
        for _, corner in corners_y99.iterrows():
            lanzador = corner['playerName']
            if lanzador in top_lanzadores:
                lanzador_idx = top_lanzadores.index(lanzador)
                
                if corner['In-swinger'] == 'Sí':
                    golpeo_lanzador_counts[0][lanzador_idx] += 1  # Cerrados
                elif corner['Out-swinger'] == 'Sí':
                    golpeo_lanzador_counts[1][lanzador_idx] += 1  # Abiertos
                elif corner['Straight'] == 'Sí':
                    golpeo_lanzador_counts[2][lanzador_idx] += 1  # Planos
        
        # Agregar flujos Golpeo → Lanzadores
        for golpeo_idx, lanzadores_dict in golpeo_lanzador_counts.items():
            for lanzador_idx, count in lanzadores_dict.items():
                if count > 0:
                    source.append(golpeo_idx)
                    target.append(len(golpeo_nombres) + lanzador_idx)  # CAMBIO: golpeo_nombres
                    value.append(count)
        
        # FLUJO 2: Lanzadores → Zonas (usando Pass End X, Y)
        lanzador_zona_counts = defaultdict(lambda: defaultdict(int))
        
        for _, corner in corners_y99.iterrows():
            lanzador = corner['playerName']
            if lanzador in top_lanzadores:
                lanzador_idx = top_lanzadores.index(lanzador)
                
                # Obtener zona de caída usando Pass End X, Y
                pass_end_x = corner['Pass End X']
                pass_end_y = corner['Pass End Y']
                zona = self.get_zona_from_coordinates(pass_end_x, pass_end_y)
                
                if zona:
                    zona_map = {
                        'zona_1': 0, 'zona_2': 1, 'zona_3': 2, 'zona_4': 3,
                        'zona_5': 4, 'zona_6': 5, 'zona_7': 6
                    }
                    if zona in zona_map:
                        zona_idx = zona_map[zona]
                        lanzador_zona_counts[lanzador_idx][zona_idx] += 1
        
        # Agregar flujos Lanzadores → Zonas
        for lanzador_idx, zonas_dict in lanzador_zona_counts.items():
            for zona_idx, count in zonas_dict.items():
                if count > 0:
                    source.append(len(golpeo_nombres) + lanzador_idx)  # CAMBIO: golpeo_nombres
                    target.append(len(golpeo_nombres) + len(top_lanzadores) + zona_idx)  # CAMBIO: golpeo_nombres
                    value.append(count)
        
        # FLUJO 3: Zonas → Rematadores (USAR MISMA LÓGICA QUE MAPA DE CALOR)
        zona_rematador_counts = defaultdict(lambda: defaultdict(int))

        # Obtener eventos de remate con From corner = 'Sí'
        remates = self.corner_data[
            (self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])) &
            (self.corner_data['From corner'] == 'Sí')
        ]

        for _, remate in remates.iterrows():
            remate_time = remate['timeMin'] * 60 + remate['timeSec']
            
            # Buscar córner en los 4 segundos anteriores        
            corner_previo = corners_y99[
                (corners_y99['Match ID'] == remate['Match ID']) &
                (corners_y99['timeMin'] * 60 + corners_y99['timeSec'] >= remate_time - 5) &
                (corners_y99['timeMin'] * 60 + corners_y99['timeSec'] < remate_time)
            ]
            
            if not corner_previo.empty:
                # Usar el primer córner encontrado
                corner = corner_previo.iloc[0]
                
                # Obtener zona del remate (MISMA LÓGICA QUE MAPA DE CALOR)
                zona = self.get_zona_from_coordinates(remate['x'], remate['y'])
                
                if zona:
                    zona_map = {'zona_1': 0, 'zona_2': 1, 'zona_3': 2, 'zona_4': 3,
                            'zona_5': 4, 'zona_6': 5, 'zona_7': 6}
                    if zona in zona_map:
                        zona_idx = zona_map[zona]
                        rematador = remate['playerName']
                        if rematador in top_rematadores:
                            rematador_idx = top_rematadores.index(rematador)
                            zona_rematador_counts[zona_idx][rematador_idx] += 1
        
        # Agregar flujos Zonas → Rematadores
        for zona_idx, rematadores_dict in zona_rematador_counts.items():
            for rematador_idx, count in rematadores_dict.items():
                if count > 0:
                    source.append(len(golpeo_nombres) + len(top_lanzadores) + zona_idx)  # CAMBIO: golpeo_nombres
                    target.append(len(golpeo_nombres) + len(top_lanzadores) + len(zonas_nombres) + rematador_idx)  # CAMBIO: golpeo_nombres
                    value.append(count)
        
        return {
            'source': source,
            'target': target,
            'value': value,
            'labels': labels
        }

    def get_player_shirt_number(self, player_name):
        """Obtiene el dorsal del jugador desde player_stats"""
        try:
            # Filtrar por el equipo actual
            team_players = self.player_stats[self.player_stats['Team Name'] == self.team_filter]
            
            # Buscar coincidencia exacta primero
            exact_match = team_players[team_players['Match Name'] == player_name]
            if not exact_match.empty:
                return exact_match.iloc[0]['Shirt Number']
            
            # Si no hay coincidencia exacta, buscar similitud
            from difflib import SequenceMatcher
            best_match = None
            best_score = 0.8  # Umbral alto para nombres
            
            for _, player in team_players.iterrows():
                score = SequenceMatcher(None, player_name.lower(), player['Match Name'].lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_match = player
            
            if best_match is not None:
                return best_match['Shirt Number']
            
            return None
        except Exception as e:
            print(f"⚠️ Error obteniendo dorsal de {player_name}: {e}")
            return None

    def get_player_photo_without_dorsal(self, player_name, photos_data):
        """Obtiene la foto sin fondo blanco pero SIN dorsal"""
        import base64
        from io import BytesIO
        from PIL import Image
        import numpy as np
        
        match = self.match_player_name(player_name, photos_data)
        if not match:
            return None
        
        try:
            # Decodificar base64 y convertir a imagen
            img_data = base64.b64decode(match['image_base64'])
            img = Image.open(BytesIO(img_data))
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Remover fondo blanco/claro
            data = np.array(img)
            # Definir rango de colores blancos/claros a hacer transparentes
            white_mask = (data[:,:,0] > 240) & (data[:,:,1] > 240) & (data[:,:,2] > 240)
            data[white_mask] = [0, 0, 0, 0]  # Hacer transparente
            img = Image.fromarray(data, 'RGBA')
            
            # NO agregar dorsal, solo devolver la imagen limpia
            return np.array(img) / 255.0
            
        except Exception as e:
            print(f"⚠️ Error procesando foto de {player_name}: {e}")
            return None

    def calculate_dynamic_sizing(self, num_lanzadores, num_rematadores):
        """Calcula tamaños dinámicos con mejor escalado"""
        max_elementos = max(num_lanzadores, num_rematadores, 1)
        
        if max_elementos <= 2:
            box_size = 1.2
            spacing_factor = 1.0
            box_height_factor = 0.16
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
        else:
            box_size = 0.5
            spacing_factor = 0.75
            box_height_factor = 0.10
        
        return {
            'box_size': box_size,
            'spacing_factor': spacing_factor,
            'altura_disponible': 0.75,
            'box_height_factor': box_height_factor
        }

    def draw_player_box_simple(self, ax, x, y, player_name, player_id, sizing_params, count_value=0, zona_info=None):
        """Dibuja recuadro de jugador con ancho dinámico."""
        box_size = sizing_params['box_size']
        
        # --- CAMBIO: ANCHO DINÁMICO ---
        # 1. Formatear el nombre primero para saber si tendrá una o dos líneas
        def format_player_name(name, max_chars_per_line=9):
            words = name.split()
            if len(name) <= max_chars_per_line: return name
            if len(words) == 2: return f"{words[0]}\n{words[1]}"
            if len(words) > 2:
                first_line = words[0]
                second_line = ' '.join(words[1:])
                if len(second_line) > max_chars_per_line: second_line = second_line[:max_chars_per_line-1] + '.'
                return f"{first_line}\n{second_line}"
            return f"{name[:max_chars_per_line-1]}."

        nombre_formateado = format_player_name(player_name, 9)
        
        # 2. Medir la línea más larga para determinar el ancho necesario
        longest_line = max(nombre_formateado.split('\n'), key=len)
        num_chars = len(longest_line)

        # 3. Calcular el ancho basado en los caracteres (estos valores se pueden ajustar)
        base_width = 0.05 
        char_increment = 0.009
        box_width = (base_width + num_chars * char_increment) * box_size
        box_height = sizing_params['box_height_factor'] * box_size
        
        # --- EL RESTO DEL CÓDIGO AHORA USA EL NUEVO box_width ---
        
        # 1. Recuadro blanco principal
        main_box = patches.FancyBboxPatch(
            (x - box_width/2, y - box_height/2), 
            box_width, box_height,
            boxstyle="round,pad=0.002", 
            facecolor='white',
            edgecolor='#2C3E50', 
            linewidth=2, 
            zorder=5
        )
        ax.add_patch(main_box)
        
        # 2. DORSAL
        dorsal = self.get_player_shirt_number(player_name)
        if dorsal:
            dorsal_rect_width = min(box_width * 0.5, 0.05) # Limitar ancho del dorsal
            dorsal_rect = patches.Rectangle(
                (x - dorsal_rect_width/2, y + box_height*0.15),
                dorsal_rect_width, box_height*0.35,
                facecolor='#E74C3C', edgecolor='white', 
                linewidth=1, zorder=8
            )
            ax.add_patch(dorsal_rect)
            ax.text(x, y + box_height*0.32, str(int(dorsal)), ha='center', va='center', fontsize=max(8, 10 * box_size), fontweight='bold', color='white', zorder=9)
        
        # 3. NOMBRE
        ax.text(x, y - box_height * 0.20, nombre_formateado,
                ha='center', va='center', 
                fontsize=max(4, 6 * box_size), fontweight='bold',
                color='#2C3E50', zorder=8,
                linespacing=0.6)
        
        # 4. CONTADOR
        if count_value > 0:
            ax.text(x, y - box_height * 0.705, str(int(count_value)),
                    ha='center', va='center', fontsize=max(4, 6 * box_size), fontweight='bold',
                    color='white', zorder=11,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='#2C3E50', edgecolor='white', linewidth=1, alpha=1.0))

        
    def create_sankey_plot(self):
        """Crea gráfica Sankey y la convierte a imagen para matplotlib"""
        
        if not PLOTLY_AVAILABLE:
            return None
            
        sankey_data = self.get_sankey_data()
        
        # Obtener datos para valores en etiquetas
        tipos_data = self.get_tipo_lanzamiento_data()  
        lanzadores_data = self.get_lanzadores_data()
        rematadores_data = self.get_rematadores_data()
        
        # CORREGIR ESTA SECCIÓN:
        labels_with_values = []
        
        for i, label in enumerate(sankey_data['labels']):
            if i < 3:  # Primeros 3 son tipos de golpeo
                golpeo_names = ['Cerrados', 'Abiertos', 'Planos']
                value = tipos_data.get(golpeo_names[i], 0)
                labels_with_values.append(f"{golpeo_names[i]}\n{value}")
                
            elif i < 3 + len(lanzadores_data.head(5)):  # Lanzadores (posiciones 3-7)
                lanzador_idx = i - 3  # Restar 3 para obtener índice correcto
                lanzador_name = list(lanzadores_data.head(5).index)[lanzador_idx]
                value = lanzadores_data.get(lanzador_name, 0)
                labels_with_values.append(f"{lanzador_name}\n{value}")
                
            elif i < 3 + len(lanzadores_data.head(5)) + 7:  # Zonas (posiciones 8-14)
                # La variable 'label' ya tiene el nombre correcto de la zona, 
                # tal como se definió en get_sankey_data usando self.ZONA_NAMES.
                # Simplemente la usamos directamente.
                labels_with_values.append(label)
                
            else:  # Rematadores (resto)
                rematador_idx = i - 3 - len(lanzadores_data.head(5)) - 7
                rematador_name = list(rematadores_data.head(8).index)[rematador_idx]
                value = rematadores_data.get(rematador_name, 0)
                labels_with_values.append(f"{rematador_name}\n{value}")

        # Colores para 4 niveles: Tipos + Lanzadores + Zonas + Rematadores
        node_colors = (['#FF6B6B', '#4ECDC4', '#45B7D1'] +  # 3 Tipos
                       ['#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD'] +  # 5 Lanzadores
                       ['#00D2D3', '#FF9F43', '#54A0FF', '#5F27CD', '#FF6B6B', '#10AC84', '#EE5A24'] +  # 7 Zonas
                       ['#0ABDE3', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF'])  # 8 Rematadores
        
        # Links con colores específicos por categoría
        link_colors = []
        top_lanzadores = list(lanzadores_data.head(5).index)
        
        for i, (s, t) in enumerate(zip(sankey_data['source'], sankey_data['target'])):
            if t < len(top_lanzadores) + 3:  # Lanzadores → Tipos
                link_colors.append('rgba(255, 107, 107, 0.6)')
            else:  # Tipos → Rematadores
                link_colors.append('rgba(78, 205, 196, 0.6)')
        
        # Crear figura Sankey
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=25,          # Más espacio entre nodos
                thickness=40,    # Nodos más gruesos (era 25)
                line=dict(color="white", width=3),  # Bordes más gruesos
                label=labels_with_values,
                color=node_colors,
                hovertemplate='%{label}<br>Conexiones: %{value}<extra></extra>'
            ),
            link=dict(
                source=sankey_data['source'],
                target=sankey_data['target'],
                value=sankey_data['value'],
                color=link_colors,  # Colores específicos por categoría
                hovertemplate='Flujo: %{value}<extra></extra>'
            )
        )])
        
        fig.update_layout(
            title_text="",  # Sin título arriba
            font=dict(size=11, family="Arial", color="black"),
            width=900,      # Más ancho
            height=500,     # Más alto
            margin=dict(l=10, r=10, t=10, b=10),  # Márgenes mínimos
            plot_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Convertir a imagen
        img_bytes = pio.to_image(fig, format='png', width=900, height=500)
        img = Image.open(io.BytesIO(img_bytes))
        
        return img

    def create_sankey_izquierda_avanzado(self, ax):
        """Flujo avanzado SOLO desde la izquierda (y > 99) con recuadros simples"""
        ax.clear()

        vertical_offset = -0.08 
        
        corners_izq = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] > 99)
        ]
        
        if corners_izq.empty:
            ax.text(0.5, 0.5, 'FLUJO CÓRNERS IZQUIERDA\n(y > 99)\n\n(Sin datos)', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            ax.axis('off')
            return
        
        # --- Preparación de datos (sin cambios) ---
        lanzadores_data = corners_izq['playerName'].value_counts()
        tipos_data = self.get_tipo_lanzamiento_data()
        zonas_data = self.get_zonas_de_caida_data()
        rematadores_data = self.get_rematadores_data()
        
        sizing_params = self.calculate_dynamic_sizing(len(lanzadores_data), len(rematadores_data))
        x_pos = [0.07, 0.3, 0.65, 0.92]
        
        # --- DIBUJO DE NODOS (sin cambios) ---
        # NIVEL 1: LANZADORES
        lanzadores_items = list(lanzadores_data.head(5).items())
        colores_jugadores = ['#2C3E50', '#6A1B9A', '#F39C12', '#27AE60', '#8B4513']
        lanzadores_colors = {}
        lanzadores_y_map = {}
        if lanzadores_items:
            lanzadores_y = np.linspace(0.90, 0.15, len(lanzadores_items)) + vertical_offset
            zona_info_data = self.get_zona_caida_con_rematador()
            for i, (player_name, count) in enumerate(lanzadores_items):
                player_matches = corners_izq[corners_izq['playerName'] == player_name]['playerId'].dropna()
                player_id = player_matches.iloc[0] if not player_matches.empty else None
                zona_info = zona_info_data.get(player_name, None)
                self.draw_player_box_simple(ax, x_pos[0], lanzadores_y[i], player_name, player_id, sizing_params, count_value=count, zona_info=zona_info)
                lanzadores_colors[player_name] = colores_jugadores[i % len(colores_jugadores)]
                lanzadores_y_map[player_name] = lanzadores_y[i]

        # NIVEL 2: TIPOS
        tipos_items = [item for item in tipos_data.items() if item[1] > 0]
        tipos_y = (np.linspace(0.8, 0.2, len(tipos_items)) + vertical_offset) if tipos_items else []
        tipos_y_map = {item[0]: y for item, y in zip(tipos_items, tipos_y)}
        tipos_colors = {'Cerrados': '#E74C3C', 'Abiertos': '#3498DB', 'Planos': '#F39C12', 'Sin Tipo': '#95a5a6'}
        for i, (tipo, count) in enumerate(tipos_items):
            color = tipos_colors[tipo]
            ax.text(x_pos[1], tipos_y[i], f'{tipo}\n({count})', ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=10, bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.9, edgecolor='white', linewidth=2))

        # NIVEL 3: ZONAS
        zonas_con_datos = sorted([item for item in zonas_data.items() if item[1] > 0], key=lambda x: x[1], reverse=True)
        zonas_y = (np.linspace(0.85, 0.15, len(zonas_con_datos)) + vertical_offset) if zonas_con_datos else []
        zonas_y_map = {item[0]: y for item, y in zip(zonas_con_datos, zonas_y)}
        zonas_colors_map = {'zona_1': '#FF6B6B', 'zona_2': '#4ECDC4', 'zona_3': '#45B7D1', 'zona_4': '#96CEB4', 'zona_5': '#FECA57', 'zona_6': '#FF9FF3', 'zona_7': '#54A0FF', 'zona_corto': '#8e44ad', 'zona_otros': '#7f8c8d'}
        for i, (zona_key, count) in enumerate(zonas_con_datos):
            zona_nombre = self.ZONA_NAMES.get(zona_key, zona_key)
            base_radius = 0.04 + (count / (sum(zonas_data.values()) or 1)) * 0.18
            color = zonas_colors_map.get(zona_key, 'gray')
            outer_circle = patches.Circle((x_pos[2], zonas_y[i]), base_radius, facecolor=color, alpha=0.9, zorder=3)
            ax.add_patch(outer_circle)
            middle_circle = patches.Circle((x_pos[2], zonas_y[i]), base_radius * 0.7, facecolor='white', alpha=0.8, zorder=4)
            ax.add_patch(middle_circle)
            inner_circle = patches.Circle((x_pos[2], zonas_y[i]), base_radius * 0.4, facecolor=color, alpha=1.0, zorder=5)
            ax.add_patch(inner_circle)
            ax.text(x_pos[2], zonas_y[i], f"{zona_nombre}\n({count})", ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=6, path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])

        # NIVEL 4: REMATADORES
        rematadores_items = list(rematadores_data.head(8).items())
        rematadores_y_map = {}
        if rematadores_items:
            rematadores_y = np.linspace(0.85, 0.15, len(rematadores_items)) + vertical_offset
            for i, (player_name, count) in enumerate(rematadores_items):
                remates = self.corner_data[(self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])) & (self.corner_data['From corner'] == 'Sí') & (self.corner_data['playerName'] == player_name)]
                player_id = remates['playerId'].iloc[0] if not remates.empty else None
                self.draw_player_box_simple(ax, x_pos[3], rematadores_y[i], player_name, player_id, sizing_params, count_value=count)
                rematadores_y_map[player_name] = rematadores_y[i]
        
        # ================================================================
        # === NUEVO BUCLE DE CONEXIONES UNIFICADO ===
        # ================================================================
        for _, corner in corners_izq.iterrows():
            lanzador = corner['playerName']
            
            # Solo procesar lanzadores que están visibles en la gráfica
            if lanzador not in lanzadores_y_map:
                continue

            # --- 1. Determinar TIPO y dibujar conexión LANZADOR -> TIPO ---
            if corner['In-swinger'] == 'Sí': tipo = 'Cerrados'
            elif corner['Out-swinger'] == 'Sí': tipo = 'Abiertos'
            elif corner['Straight'] == 'Sí': tipo = 'Planos'
            else: tipo = 'Sin Tipo'
            
            if tipo in tipos_y_map:
                arrow1 = patches.FancyArrowPatch(
                    (x_pos[0] + 0.05, lanzadores_y_map[lanzador]), 
                    (x_pos[1] - 0.06, tipos_y_map[tipo]),
                    connectionstyle="arc3,rad=0.1", 
                    color=lanzadores_colors[lanzador], alpha=0.7, 
                    linewidth=2, zorder=1)
                ax.add_patch(arrow1)

            # --- 2. Determinar ZONA DE CAÍDA y dibujar conexión TIPO -> ZONA ---
            zona_caida = self.get_zona_from_coordinates(corner['Pass End X'], corner['Pass End Y'])
            if zona_caida in zonas_y_map and tipo in tipos_y_map:
                rad = {'Cerrados': 0.2, 'Abiertos': -0.2, 'Planos': 0.0, 'Sin Tipo': 0.1}.get(tipo, 0)
                arrow2 = patches.FancyArrowPatch(
                    (x_pos[1] + 0.06, tipos_y_map[tipo]), 
                    (x_pos[2] - 0.05, zonas_y_map[zona_caida]),
                    connectionstyle=f"arc3,rad={rad}", 
                    color=lanzadores_colors[lanzador], alpha=0.7, 
                    linewidth=2, zorder=1)
                ax.add_patch(arrow2)

            # --- 3. Buscar REMATE y dibujar conexión ZONA -> REMATADOR ---
            corner_time = corner['timeMin'] * 60 + corner['timeSec']
            remates_posteriores = self.corner_data[
                (self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])) &
                (self.corner_data['From corner'] == 'Sí') &
                (self.corner_data['Match ID'] == corner['Match ID']) &
                (self.corner_data['timeMin'] * 60 + self.corner_data['timeSec'] > corner_time) &
                (self.corner_data['timeMin'] * 60 + self.corner_data['timeSec'] <= corner_time + 5) # Solo primer contacto
            ]

            if not remates_posteriores.empty:
                primer_remate = remates_posteriores.iloc[0]
                rematador = primer_remate['playerName']
                zona_remate = self.get_zona_from_coordinates(primer_remate['x'], primer_remate['y'])

                # Conectar la ZONA DE CAÍDA del balón con el REMATADOR
                if zona_caida in zonas_y_map and rematador in rematadores_y_map:
                    arrow3 = patches.FancyArrowPatch(
                        (x_pos[2] + 0.05, zonas_y_map[zona_caida]), 
                        (x_pos[3] - 0.05, rematadores_y_map[rematador]),
                        connectionstyle="arc3,rad=0.1", 
                        color=lanzadores_colors[lanzador], alpha=0.7, 
                        linewidth=2, zorder=1)
                    ax.add_patch(arrow3)

        # --- Títulos y configuración final (sin cambios) ---
        titulos = ['LANZADORES', 'TIPOS', 'ZONAS', 'REMATADORES']
        colores_modernos = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
        for i, titulo in enumerate(titulos):
            ax.text(x_pos[i], 1.02 + vertical_offset, titulo, ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=10, bbox=dict(boxstyle="round,pad=0.5", facecolor=colores_modernos[i], alpha=0.95, edgecolor='white', linewidth=2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0 + vertical_offset, 1.02 + vertical_offset)
        ax.set_title('FLUJO CÓRNERS DESDE LA IZQUIERDA', fontsize=10, fontweight='bold', color='black', pad=20)
        ax.axis('off')

    def load_team_logo(self, equipo, target_size=(80, 80)):
        """Carga logo del equipo con tamaño fijo"""
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

    def create_corners_report(self, figsize=(11.69, 8.27), team_filter=None):
        """Crea el reporte con Sankey y ranking de juego aéreo"""
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        # FILA 1: MODIFICADO - 4 columnas para incluir ranking
        gs_fila1 = fig.add_gridspec(1, 5, hspace=0.3, wspace=0.05, 
                            top=0.9, bottom=0.5,
                            left=0.05, right=0.95,
                            width_ratios=[1.2, 1, 1, 1, 1])

        # FILA 2: Sin cambios - Solo 2 columnas
        gs_fila2 = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.11,
                                top=0.45, bottom=0.05,
                                left=0.05, right=0.95,
                                width_ratios=[3, 2])  # Campo y Sankey
        
        # Fondo
        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25)
            ax_bg.axis('off')
        
        # Título principal
        fig.suptitle('CÓRNERS OFENSIVOS', fontsize=18, fontweight='bold', 
                    color='#1e3d59', y=0.95, family='serif')
        
        # Logo del equipo
        # <-- CORRECCIÓN AQUÍ
        if team_filter and (team_logo := self.load_team_logo(team_filter)) is not None:
            ax_team = fig.add_axes([0.92, 0.90, 0.06, 0.06])
            ax_team.imshow(team_logo, aspect='auto')
            ax_team.axis('off')
        
        # === FILA 1, COLUMNA 1: CAMPO SUPERIOR CON INFORMACIÓN ===
        ax_campo_superior = fig.add_subplot(gs_fila1[0, 0])  # CAMBIO: ahora columna 0
        ax_campo_superior.set_facecolor('none')
        pitch_superior = VerticalPitch(half=True, pitch_type='opta', pitch_color='none', 
                                    line_color='black', linewidth=3)
        pitch_superior.draw(ax=ax_campo_superior)


        # === FILA 1, COLUMNA 2: RANKING JUEGO AÉREO ===
        ax_ranking = fig.add_subplot(gs_fila1[0, 1])  # CAMBIO: ahora columna 1
        self.create_aerial_ranking(ax_ranking)
        
        # Información "MAYOR PROBABILIDAD" 
        secuencia = self.get_secuencia_mas_repetida()
        photos_data = self.load_player_photos()
        player_photo = self.get_player_photo_without_dorsal(secuencia['lanzador'], photos_data)

        if player_photo is not None:
            # Crear un espacio para la foto en la esquina superior derecha del campo
            photo_ax = ax_campo_superior.inset_axes([0.01, 0.10, 0.4, 0.5])  # [x, y, width, height]
            photo_ax.imshow(player_photo, aspect='auto')
            photo_ax.axis('off')

        info_text = f"""MAYOR PROBABILIDAD

        LANZADOR: {secuencia['lanzador']}
        ZONA: {secuencia['zona']}
        GOLPEO: {secuencia['golpeo']}
        PIERNA: {secuencia['pierna']}"""

        ax_campo_superior.text(0.25, 0.7, info_text, fontsize=8, va='top', ha='left',
                            transform=ax_campo_superior.transAxes,
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

        ax_campo_superior.set_title('DESDE LA IZQUIERDA', fontsize=12, fontweight='bold', pad=20)

        # Balón en córner
        if (ball_img := self.load_ball_image()) is not None:
            ball_box = OffsetImage(ball_img, zoom=0.15)
            ball_ab = AnnotationBbox(ball_box, (99.5, 99.5), frameon=False)
            ax_campo_superior.add_artist(ball_ab)
        
        # === FILA 1, COLUMNAS 3-5: TABLA LANZADORES VS REMATADORES (VERSIÓN FINAL) ===
        ax_tabla = fig.add_subplot(gs_fila1[0, 2:])
        ax_tabla.set_facecolor('none')
        ax_tabla.axis('off')

        matriz, lanzadores, rematadores = self.get_matriz_lanzador_rematador()

        if matriz and lanzadores and rematadores:
            
            # --- FUNCIONES DE FORMATO DE TEXTO ---
            def format_lanzador_name(name, max_len=12):
                """Parte nombres largos de lanzadores en dos líneas."""
                if len(name) > max_len:
                    words = name.split()
                    if len(words) > 1:
                        mid = len(words) // 2
                        return ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
                    else:
                        mid = len(name) // 2
                        return name[:mid] + '\n' + name[mid:]
                return name

            def format_vertical_header(name):
                """Formatea cabeceras de rematadores para que sean verticales y cortas."""
                words = name.split()
                return '\n'.join(words)

            # --- PREPARACIÓN DE DATOS ---
            table_data = []

            # NUEVA ESTRUCTURA: Solo una fila de cabecera con nombres de rematadores
            rematadores_limitados = rematadores[:10]
            header_nombres = [''] + [format_vertical_header(rem) for rem in rematadores_limitados]
            table_data.append(header_nombres)

            # Filas de datos: solo nombres de lanzadores + datos
            lanzadores_limitados = lanzadores[:8]
            for lanzador in lanzadores_limitados:
                row = [format_lanzador_name(lanzador)]
                for rematador in rematadores_limitados:
                    value = matriz.get(lanzador, {}).get(rematador, 0)
                    row.append(str(value) if value > 0 else '')
                table_data.append(row)
            
            # --- DIBUJAR LA TABLA ---
            if len(table_data) > 1:
                col_widths = [0.20] + [0.08] * len(rematadores_limitados)  # Más espacio para lanzadores
                
                # Crear la tabla completa
                table = ax_tabla.table(cellText=table_data,
                                    cellLoc='center', loc='center', 
                                    bbox=[0, 0, 1, 1],
                                    colWidths=col_widths)
                
                table.auto_set_font_size(False)
                table.set_fontsize(7)
                table.scale(1, 2.8)  # Más altura

                # --- FORMATO DE CELDAS ---
                all_values = []
                for row in table_data[1:]:  # CAMBIO: de [2:] a [1:] ya que ahora solo hay una fila de cabecera
                    for val in row[1:]:  # CAMBIO: de [2:] a [1:] ya que ahora solo hay una columna de lanzadores
                        if val and val.isdigit():
                            all_values.append(int(val))
                max_value = max(all_values) if all_values else 1
                
                cmap = plt.cm.get_cmap('Reds')

                for i in range(len(table_data)):
                    for j in range(len(table_data[0])):
                        cell = table[(i, j)]
                        cell.set_edgecolor('lightgray')
                        cell.set_linewidth(0.5)

                        if i == 0 and j == 0:
                            # Celda vacía superior izquierda
                            cell.set_facecolor('none')
                            cell.set_edgecolor('none')
                            cell.get_text().set_text('')
                        
                        elif i == 0 and j > 0:
                            # Nombres de rematadores (primera fila)
                            cell.set_facecolor('#2C3E50')  # Color para rematadores
                            cell.get_text().set_rotation(45)  # ← DIAGONAL HACIA LA DERECHA (45 grados)
                            cell.get_text().set_color('white')
                            cell.get_text().set_weight('bold')
                            cell.get_text().set_va('center')
                            cell.get_text().set_ha('center')  # ← AÑADIR ESTO PARA CENTRAR MEJOR
                            cell.get_text().set_fontsize(9)
                        
                        elif i > 0 and j == 0:
                            # Nombres de lanzadores (primera columna)
                            cell.set_facecolor('#E74C3C')  # Color para lanzadores
                            cell.get_text().set_color('white')
                            cell.get_text().set_weight('bold')
                            cell.get_text().set_ha('center')
                            cell.get_text().set_va('center')
                            cell.get_text().set_fontsize(9)
                        
                        elif i > 0 and j > 0:
                            # Celdas de datos
                            try:
                                value = int(table_data[i][j]) if table_data[i][j] else 0
                                if value > 0:
                                    intensity = value / max_value
                                    cell.set_facecolor(cmap(intensity))
                                    cell.get_text().set_weight('bold')
                                    text_color = 'white' if intensity > 0.6 else 'black'
                                    cell.get_text().set_color(text_color)
                                else:
                                    cell.set_facecolor('none')
                            except (ValueError, IndexError):
                                cell.set_facecolor('none')
                    
                    # --- LEYENDA ARRIBA DE LA TABLA ---
                    from matplotlib.patches import Rectangle

                    # Cuadrado rojo para lanzadores
                    rect_lanzadores = Rectangle((0.01, 0.94), 0.03, 0.02, 
                                                facecolor='#E74C3C', edgecolor='black', linewidth=1,
                                                alpha=1.0, transform=ax_tabla.transAxes)
                    ax_tabla.add_patch(rect_lanzadores)
                    ax_tabla.text(0.05, 0.945, 'LANZADORES', transform=ax_tabla.transAxes,
                                ha='left', va='center', fontsize=6, fontweight='bold')

                    # Cuadrado azul para rematadores  
                    rect_rematadores = Rectangle((0.01, 0.91), 0.03, 0.02,
                                                facecolor='#2C3E50', edgecolor='black', linewidth=1,
                                                alpha=1.0, transform=ax_tabla.transAxes)
                    ax_tabla.add_patch(rect_rematadores)
                    ax_tabla.text(0.05, 0.915, 'REMATADORES', transform=ax_tabla.transAxes,
                                ha='left', va='center', fontsize=6, fontweight='bold')


        
        # === FILA 2, COLUMNA 1: CAMPO AMPLIADO CON MAPA DE CALOR ===
        ax_campo_inferior = fig.add_subplot(gs_fila2[0, 0])
        ax_campo_inferior.set_facecolor('none')

        pitch_inferior = VerticalPitch(half=True, pitch_type='opta', pitch_color='none', 
                            line_color='black', linewidth=3)
        pitch_inferior.draw(ax=ax_campo_inferior)

        ax_campo_inferior.set_xlim(0, 100)
        ax_campo_inferior.set_ylim(70, 102)

        ax_campo_inferior.set_title('1er contacto tras el saque - MAPA DE CALOR', fontsize=12, fontweight='bold')

        zonas_data = self.get_contactos_por_zona()
        max_remates = max(zonas_data.values()) if zonas_data.values() else 1

        import matplotlib.cm as cm
        colormap = cm.get_cmap('Blues')

        zona_coords = {
            'zona_1': [(70, 0), (100, 25)],      # ✅ Z. IZQUIERDA ahora abajo  
            'zona_2': [(70, 25), (88.5, 75)],    
            'zona_3': [(88.5, 25), (100, 42)],   # ✅ ÁREA PEQ. IZQ ahora abajo
            'zona_4': [(94.2, 42), (100, 58)],   
            'zona_5': [(88.5, 58), (100, 75)],   # ✅ ÁREA PEQ. DER ahora arriba
            'zona_6': [(83, 42), (94.2, 58)],    
            'zona_7': [(70, 75), (100, 100)]     # ✅ Z. DERECHA ahora arriba
        }

        orden_dibujo = ['zona_1', 'zona_2', 'zona_3', 'zona_4', 'zona_5', 'zona_7', 'zona_6']

        for zona in orden_dibujo:
            if zona in zona_coords:
                coords = zona_coords[zona]
                (x_min, y_min), (x_max, y_max) = coords
                width, height = x_max - x_min, y_max - y_min
                count = zonas_data.get(zona, 0)
                if max_remates > 0:
                    intensidad = 0.2 + (count / max_remates) * 0.8
                else:
                    intensidad = 0.2
                color_zona = colormap(intensidad)
                # Definir la transparencia: 1.0 (opaco) para penalti, 0.8 para el resto.
                alpha_zona = 1.0 if zona == 'zona_6' else 0.8
                rect = patches.Rectangle((y_min, x_min), height, width, 
                                    linewidth=3, edgecolor='black',  
                                    facecolor=color_zona, alpha=alpha_zona) # <-- Usamos la nueva variable aquí
                ax_campo_inferior.add_patch(rect)
                center_y, center_x = y_min + height/2, x_min + width/2
                text_color = 'white' if intensidad > 0.6 else 'black'
                ax_campo_inferior.text(center_y, center_x, str(count), 
                                    fontsize=16, fontweight='bold', ha='center', va='center',
                                    color=text_color)
                
        # === FILA 2, COLUMNA 2: GRÁFICA SANKEY SIMPLE (SIN FOTOS) ===
        ax_sankey = fig.add_subplot(gs_fila2[0, 1])
        ax_sankey.set_facecolor('none')
        ax_sankey.axis('off')

        # LLAMAR DIRECTAMENTE a la versión simple sin fotos
        self.create_sankey_izquierda_avanzado(ax_sankey)
        
        plt.tight_layout()
        return fig

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
        print("=== GENERADOR DE REPORTES DE CÓRNERES OFENSIVOS ===")
        if (equipo := seleccionar_equipo_interactivo()) is None:
            print("No se pudo completar la selección.")
            return
        
        print(f"\nGenerando reporte para {equipo}")
        analyzer = CornersOffensiveReport(team_filter=equipo)
        
        # AÑADIR ESTA LÍNEA PARA EL DEBUG:
        analyzer.debug_secuencias_completo()
        analyzer.debug_aerial_complete()      # ← AÑADIR ESTA LÍNEA
        analyzer.debug_remates_coordenadas()

        
        if (fig := analyzer.create_corners_report(team_filter=equipo)):
            plt.show()
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_corners_ofensivos_{equipo_filename}.pdf"
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1, 
                       facecolor='white', dpi=300)
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
        analyzer = CornersOffensiveReport(team_filter=equipo)
        fig = analyzer.create_corners_report(team_filter=equipo)
        
        if fig:
            if mostrar: 
                plt.show()
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"reporte_corners_ofensivos_{equipo_filename}.pdf"
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

def verificar_assets():
    """Verifica la disponibilidad de assets necesarios"""
    print("\n=== VERIFICACIÓN DE ASSETS ===")
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
        print(f"✅ Escudos disponibles ({len(escudos)}): {escudos[:5]}...")
    else:
        print("⚠️  No hay escudos en el directorio")

if __name__ == "__main__":
    print("=== INICIALIZANDO GENERADOR DE REPORTES DE CÓRNERES OFENSIVOS ===")
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