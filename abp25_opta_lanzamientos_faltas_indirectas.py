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

class ReporteCampogramasFaltas:
    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/abp_events.parquet", 
             team_filter=None, jornadas_filter=None):
        """
        Inicialización mejorada con filtro opcional de jornadas
        
        Args:
            data_path: Ruta al archivo de datos
            team_filter: Nombre del equipo a filtrar (opcional)
            jornadas_filter: Lista de jornadas a incluir (opcional)
        """
        self.data_path = data_path
        self.team_filter = team_filter
        self.jornadas_filter = jornadas_filter  # Guardar filtro sin aplicar
        self.df = None
        self.falta_data = pd.DataFrame()
        self.team_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/team_stats.parquet")
        self.player_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/player_stats.parquet")

        self.zones = {
            1: [(100, 100), (100, 78.9), (83, 78.9), (83, 100)],
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

        # Diccionario para nombres de categorías
        self.CATEGORIA_NAMES = {
            'frontales': 'FALTAS FRONTALES',
            'alejadas': 'FALTAS ALEJADAS', 
            'laterales': 'FALTAS LATERALES'
        }
        
        # Cargar datos sin aplicar filtro de jornadas
        self.load_data(team_filter)
        
        # Obtener jornadas disponibles después de cargar datos
        self.jornadas_disponibles = sorted([int(j) for j in self.df['Week'].dropna().unique()])
        
        # Si se especificó un filtro de jornadas, validarlo
        if self.jornadas_filter:
            self.jornadas_filter = [j for j in self.jornadas_filter if j in self.jornadas_disponibles]
            print(f"✅ Filtro de jornadas aplicado: {self.jornadas_filter}")
    
    def create_team_report(self, team_name, jornadas_filter=None, figsize=(11.69, 8.27)):
        """Método wrapper para generador maestro"""
        # Filtrar por equipo si los datos ya están cargados
        if hasattr(self, 'df_eventos') and not self.df_eventos.empty:
            original_count = len(self.df_eventos)
            self.df_eventos = self.df_eventos[self.df_eventos['Team Name'] == team_name]
            print(f"Datos filtrados para {team_name}: {len(self.df_eventos)}/{original_count}")
        
        # Llamar al método existente
        return self.create_reporte_campogramas(figsize=figsize, jornadas_filter=jornadas_filter)
    
    def create_team_report(self, team_name, jornadas_filter=None, figsize=(11.69, 8.27)):
        """
        Método wrapper para generador maestro.
        Filtra datos y llama a create_reporte_campogramas.
        """
        # Filtrar por equipo antes de generar
        if hasattr(self, 'df_eventos') and not self.df_eventos.empty:
            self.df_eventos = self.df_eventos[self.df_eventos['Team Name'] == team_name]
        
        # Llamar al método existente
        return self.create_reporte_campogramas(figsize=figsize, jornadas_filter=jornadas_filter)
    
    def seleccionar_jornada_con_4_anteriores(self):
        """
        Selección de jornada final (incluye automáticamente las 4 anteriores)
        Total: 5 jornadas acumuladas
        
        Returns:
            list: Lista de 5 jornadas consecutivas
        """
        if not self.jornadas_disponibles:
            print("⚠️ No hay jornadas disponibles")
            return []
        
        print("\n=== SELECCIÓN DE JORNADA FINAL ===")
        print(f"Jornadas disponibles: {self.jornadas_disponibles}")
        print("Se mostrarán 5 jornadas acumuladas (la seleccionada + 4 anteriores)")
        print("\nSelecciona la jornada FINAL del análisis:")
        
        # Mostrar opciones válidas (solo desde jornada 5 en adelante para tener 5 acumuladas)
        min_jornada_valida = 5
        jornadas_validas = [j for j in self.jornadas_disponibles if j >= min_jornada_valida]
        
        if not jornadas_validas:
            print(f"⚠️ No hay suficientes jornadas. Se necesita al menos jornada {min_jornada_valida}")
            # Si hay menos de 5 jornadas, devolver todas las disponibles
            return self.jornadas_disponibles[:5]
        
        for i, jornada in enumerate(jornadas_validas, 1):
            jornada_inicio = max(1, jornada - 4)
            print(f"{i}. Jornada {jornada} (mostrará J{jornada_inicio} a J{jornada})")
        
        while True:
            try:
                indice = int(input(f"\nSelecciona jornada final (1-{len(jornadas_validas)}): ").strip()) - 1
                if 0 <= indice < len(jornadas_validas):
                    jornada_final = jornadas_validas[indice]
                    jornada_inicial = max(min(self.jornadas_disponibles), jornada_final - 4)
                    
                    # Obtener las 5 jornadas consecutivas
                    jornadas_seleccionadas = [j for j in self.jornadas_disponibles 
                                            if jornada_inicial <= j <= jornada_final]
                    
                    # Asegurar que sean exactamente 5 o menos si no hay suficientes
                    jornadas_seleccionadas = jornadas_seleccionadas[-5:]
                    
                    print(f"✅ Seleccionadas jornadas {jornadas_seleccionadas[0]} a {jornadas_seleccionadas[-1]}")
                    print(f"   Total: {len(jornadas_seleccionadas)} jornadas acumuladas")
                    return jornadas_seleccionadas
                else:
                    print(f"Por favor, ingresa un número entre 1 y {len(jornadas_validas)}")
            except ValueError:
                print("Por favor, ingresa un número válido")

    def point_in_zone(self, x, y, zone_coords):
        """Determina si un punto estÃ¡ dentro de una zona rectangular"""
        try:
            x, y = float(x), float(y)
            x_coords = [coord[0] for coord in zone_coords]
            y_coords = [coord[1] for coord in zone_coords]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            return min_x <= x <= max_x and min_y <= y <= max_y
        except:
            return False

    def get_zone_for_coordinates(self, x, y):
        """Determina la zona para unas coordenadas dadas"""
        if pd.isna(x) or pd.isna(y):
            return None
        for zone_num, zone_coords in self.zones.items():
            if self.point_in_zone(x, y, zone_coords):
                return zone_num
        return None

    def normalize_timestamp(self, timestamp):
        """Normaliza timestamps quitando la Z final si existe"""
        if pd.isna(timestamp):
            return timestamp
        
        timestamp_str = str(timestamp).strip()
        
        # Quitar la Z final si existe
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1]
        
        # Asegurar formato consistente
        try:
            # Convertir a datetime y volver a string para normalizar formato
            dt = pd.to_datetime(timestamp_str)
            return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Mantener 3 decimales
        except:
            # Si falla la conversión, devolver el string sin Z
            return timestamp_str

    def load_data(self, team_filter=None):
        """Carga los datos necesarios"""
        try:
            columns_needed = ['Match ID', 'periodId', 'Team ID', 'Team Name', 'Event Name', 'outcome', 
                            'timeMin', 'timeSec', 'timeStamp', 'x', 'y', 'Pass End X', 'Pass End Y', 
                            'playerName', 'playerId', 'Free kick taken', 'Zone', 'Week',
                            'In-swinger', 'Out-swinger', 'Straight', 'Left footed', 'Right footed']
            
            try:
                self.df = pd.read_parquet(self.data_path, columns=columns_needed)
            except Exception:
                basic_columns = ['Match ID', 'Team ID', 'Team Name', 'Event Name', 'outcome', 
                            'timeMin', 'timeSec', 'x', 'y', 'Pass End X', 'Pass End Y', 
                            'playerName', 'playerId', 'timeStamp', 'Week']
                self.df = pd.read_parquet(self.data_path, columns=basic_columns)
                for col in ['Free kick taken', 'Zone', 'In-swinger', 'Out-swinger', 'Straight', 'Left footed', 'Right footed']:
                    if col not in self.df.columns:
                        self.df[col] = 'No'
            
            # Normalizar timestamps
            self.df['timeStamp'] = self.df['timeStamp'].apply(self.normalize_timestamp)
            
            relevant_events = ['Pass', 'Goal', 'Attempt Saved', 'Miss', 'Post']
            self.df = self.df[self.df['Event Name'].isin(relevant_events)]
            
            if team_filter:
                team_matches = self.team_stats[self.team_stats['Team Name'] == team_filter]['Match ID'].unique()
                self.df = self.df[self.df['Match ID'].isin(team_matches)]
            
            # Merge para obtener dorsales del lanzador
            try:
                player_dorsales = self.player_stats[['Player ID', 'Shirt Number']].copy()
                player_dorsales = player_dorsales.rename(columns={
                    'Player ID': 'playerId', 
                    'Shirt Number': 'dorsal_lanzador'
                })
                
                self.df = self.df.merge(
                    player_dorsales,
                    on='playerId', 
                    how='left'
                )
                
                print(f"✅ Merge dorsales exitoso: {len(self.df[self.df['dorsal_lanzador'].notna()])} eventos con dorsal")
                
            except Exception as e:
                print(f"⚠️ Error en merge dorsales: {e}")
                self.df['dorsal_lanzador'] = None
            
            try:
                match_info = self.team_stats[['Match ID', 'Team Name', 'Is Home']].copy()
                self.df = self.df.merge(
                    match_info,
                    on=['Match ID', 'Team Name'],
                    how='left'
                )
                print(f"✅ Merge local/visitante exitoso.")
            except Exception as e:
                print(f"⚠️ Error en merge local/visitante: {e}")
                self.df['Is Home'] = False  # Añadir columna por defecto si falla
            
            print(f"✅ Datos cargados: {len(self.df)} eventos totales")
            
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")
            import traceback
            traceback.print_exc()
    
    def debug_data_loading(self):
        """Debug para verificar qué datos se están cargando"""
        print("\n=== DEBUG CARGA DE DATOS ===")
        print(f"Total eventos cargados: {len(self.df)}")
        print(f"Equipos únicos: {self.df['Team Name'].nunique()}")
        print(f"Eventos únicos: {self.df['Event Name'].value_counts().to_dict()}")
        
        if 'Free kick taken' in self.df.columns:
            print(f"Free kick taken - Sí: {len(self.df[self.df['Free kick taken'] == 'Sí'])}")
            print(f"Free kick taken - No: {len(self.df[self.df['Free kick taken'] == 'No'])}")
        else:
            print("❌ Columna 'Free kick taken' no encontrada")
        
        if 'Zone' in self.df.columns:
            print(f"Zones disponibles: {self.df['Zone'].value_counts().to_dict()}")
        else:
            print("❌ Columna 'Zone' no encontrada")
        
        if 'Week' in self.df.columns:
            print(f"Jornadas disponibles: {sorted(self.df['Week'].dropna().unique())}")
        else:
            print("❌ Columna 'Week' no encontrada")
        
        if self.team_filter:
            team_data = self.df[self.df['Team Name'] == self.team_filter]
            print(f"\nDatos del equipo '{self.team_filter}': {len(team_data)} eventos")
            if len(team_data) > 0:
                print(f"Jornadas del equipo: {sorted(team_data['Week'].dropna().unique())}")
    
    def seleccionar_jornada_inicial(self):
        """Selección interactiva de jornada inicial"""
        jornadas_disponibles = sorted([int(j) for j in self.df['Week'].dropna().unique()])
        if not jornadas_disponibles:
            return None
        
        max_jornada = max(jornadas_disponibles)
        
        print(f"\n=== SELECCIÓN DE JORNADA INICIAL ===")
        print(f"Se mostrarán hasta 5 jornadas consecutivas (disponibles hasta J{max_jornada})")
        for i, jornada in enumerate(jornadas_disponibles, 1):
            posibles_jornadas = min(5, max_jornada - jornada + 1)
            print(f"{i}. Jornada {jornada} (mostraría {posibles_jornadas} jornadas: {jornada}-{min(jornada + 4, max_jornada)})")
        
        while True:
            try:
                indice = int(input(f"\nSelecciona jornada inicial (1-{len(jornadas_disponibles)}): ").strip()) - 1
                if 0 <= indice < len(jornadas_disponibles):
                    return jornadas_disponibles[indice]
            except ValueError:
                print("Por favor, ingresa un número válido")

    def get_freekick_indirect_sequences(self, match_ids=None, jornadas_filter=None):
        """
        Extracción de faltas indirectas con filtro opcional de jornadas
        
        Args:
            match_ids: Lista de IDs de partidos a filtrar (opcional)
            jornadas_filter: Lista de jornadas a incluir (opcional). 
                            Si es None, usa self.jornadas_filter.
                            Si tampoco existe self.jornadas_filter, usa todas.
        
        Returns:
            list: Lista de secuencias de faltas encontradas
        """
        # Hacer copia para no modificar el DataFrame original
        df = self.df.copy()
        
        # Aplicar filtro de jornadas
        if jornadas_filter is not None:
            # Usar el filtro proporcionado como parámetro
            df = df[df['Week'].isin([str(j) for j in jornadas_filter])]
            print(f"🔍 Filtrando por jornadas (parámetro): {jornadas_filter}")
        elif hasattr(self, 'jornadas_filter') and self.jornadas_filter:
            # Usar el filtro guardado en el objeto
            df = df[df['Week'].isin([str(j) for j in self.jornadas_filter])]
            print(f"🔍 Filtrando por jornadas (objeto): {self.jornadas_filter}")
        else:
            print(f"🔍 Usando todas las jornadas disponibles: {self.jornadas_disponibles}")
        
        # Aplicar filtro de partidos si se proporciona
        if match_ids is not None:
            df = df[df['Match ID'].isin(match_ids)]
            print(f"🔍 Filtrando por {len(match_ids)} partidos específicos")
        
        if self.team_filter:
            df = df[df['Team Name'] == self.team_filter]
            print(f"🎯 Filtrando eventos para mostrar únicamente a: {self.team_filter}")

        print(f"📊 Eventos después de filtros: {len(df)}")
        
        # Calcular zona de inicio para cada evento
        print("🎯 Calculando zona de inicio para los eventos...")
        df['start_zone'] = df.apply(
            lambda row: self.get_zone_for_coordinates(row['x'], row['y']),
            axis=1
        )
        
        # Definir zonas de interés
        zonas_interes = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        # Filtrar faltas en zonas de interés
        lanzamientos_faltas = df[
            (df.get('Free kick taken', '') == 'Sí') &
            (df['start_zone'].isin(zonas_interes)) &
            (df['x'] > 60) &
            (df['x'].notna()) & (df['y'].notna())
            # Eliminar los 3 filtros de Pass End X/Y
        ].copy()
        
        print(f"✅ Total de lanzamientos de falta extraídos: {len(lanzamientos_faltas)}")
        
        # Desglose por zona
        if not lanzamientos_faltas.empty:
            zonas_count = lanzamientos_faltas['start_zone'].value_counts().sort_index()
            print("\n📍 Lanzamientos por zona:")
            for zona, count in zonas_count.items():
                print(f"   Zona {zona}: {count} lanzamientos")
        
        # Convertir a lista de diccionarios (formato esperado por el resto del código)
        sequences = [{'freekick_event': row.to_dict()} for index, row in lanzamientos_faltas.iterrows()]
        
        # Si hay filtro de jornadas, añadir info al resultado
        if jornadas_filter or self.jornadas_filter:
            jornadas_usadas = jornadas_filter if jornadas_filter else self.jornadas_filter
            print(f"\n📅 Jornadas incluidas en el análisis: {sorted(jornadas_usadas)}")
        
        return sequences

    def analyze_free_kick_sequence(self, match_events, free_kick):
        """Analiza la secuencia posterior a la falta - COPIADO COMPLETO de abp17/19/21"""
        match_id = free_kick['Match ID']
        start_time_min = free_kick['timeMin']
        start_time_sec = free_kick['timeSec']
        period_id = free_kick.get('periodId', 1)  # Agregar default
        team_id = free_kick['Team ID']
        
        # Buscar el índice del evento de la falta
        free_kick_idx = None
        match_data = match_events[match_events['Match ID'] == match_id]
        
        for idx, event in match_data.iterrows():
            if (event['timeMin'] == start_time_min and 
                event['timeSec'] == start_time_sec and
                event['playerName'] == free_kick['playerName'] and
                event.get('Free kick taken', '') == 'Sí'):
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
        
        # Buscar eventos posteriores
        free_kick_timestamp = pd.to_datetime(free_kick['timeStamp'])
        
        for next_idx in range(free_kick_idx + 1, min(free_kick_idx + 10, len(match_events))):
            next_event = match_events.iloc[next_idx]
            
            if next_event.get('periodId', 1) != period_id:
                break
                
            next_timestamp = pd.to_datetime(next_event['timeStamp'])
            time_diff = (next_timestamp - free_kick_timestamp).total_seconds()
            if time_diff > 10:
                break
            
            event_name = next_event['Event Name']
            event_team_id = next_event['Team ID']
            
            if (event_name in ['Goal', 'Post', 'Attempt Saved', 'Miss'] and 
                event_team_id == team_id and 
                events_found[event_name] is None):
                events_found[event_name] = next_event
                if event_name == 'Goal':
                    break

        # Determinar resultado
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

    def debug_sequences(self):
        """Debug para verificar las secuencias de faltas"""
        print("\n=== DEBUG SECUENCIAS DE FALTAS ===")
        sequences = self.get_freekick_indirect_sequences()
        print(f"Total secuencias encontradas: {len(sequences)}")
        
        if sequences:
            for i, seq in enumerate(sequences[:3]):  # Solo mostrar las primeras 3
                fk = seq['freekick_event']
                print(f"\nSecuencia {i+1}:")
                print(f"  Equipo: {fk.get('Team Name')}")
                print(f"  Jornada: {fk.get('Week')}")
                print(f"  Coordenadas: x={fk.get('x')}, y={fk.get('y')}")
                print(f"  Zone: {fk.get('Zone')}")
                print(f"  Pass End: x={fk.get('Pass End X')}, y={fk.get('Pass End Y')}")
        
        categorized = self.categorizar_faltas(sequences)
        print(f"\nCategorización:")
        for cat, seq_list in categorized.items():
            print(f"  {cat}: {len(seq_list)} faltas")

    def categorizar_faltas(self, sequences):
        """Categoriza las faltas ya filtradas segÃºn su zona de inicio ('start_zone')."""
        frontales = []
        alejadas = []  
        laterales = []
        
        # Zonas para cada categorÃ­a
        zonas_frontales = [8, 9, 10]
        zonas_laterales = [1, 6, 7, 11]
        zonas_alejadas = [12, 13, 14]
        
        for seq in sequences:
            fk_event = seq['freekick_event']
            zona_inicio = fk_event.get('start_zone')
            
            if zona_inicio in zonas_frontales:
                frontales.append(seq)
            elif zona_inicio in zonas_laterales:
                laterales.append(seq)
            elif zona_inicio in zonas_alejadas:
                alejadas.append(seq)
                
        return {
            'frontales': frontales,
            'alejadas': alejadas,
            'laterales': laterales
        }

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

    def is_local_team(self, team_name, match_id):
        """Determina si un equipo es local en un partido específico"""
        # Si tenemos team_filter, consideramos local al team_filter
        if self.team_filter:
            return team_name == self.team_filter
        # Si no hay filtro, usar alguna lógica para determinar local/visitante
        # Por simplicidad, consideramos el primer equipo alfabéticamente como local
        teams_in_match = self.df[self.df['Match ID'] == match_id]['Team Name'].unique()
        if len(teams_in_match) >= 2:
            return team_name == sorted(teams_in_match)[0]
        return True

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

    def create_leyenda_conjunta_campogramas(self, fig, gs, jornadas_filter=None):
        """Leyenda con filtro de jornadas"""
        sequences = self.get_freekick_indirect_sequences(jornadas_filter=jornadas_filter)
        categorized = self.categorizar_faltas(sequences)
        
        # Contar lanzamientos por color (local/visitante)
        total_casa = 0
        total_fuera = 0
        for categoria, seq_list in categorized.items():
            for seq in seq_list:
                fk_event = seq['freekick_event']
                # Contamos basándonos en la columna 'Is Home'
                if fk_event.get('Is Home', False):
                    total_casa += 1
                else:
                    total_fuera += 1
        
        legend_elements = []
        
        if total_casa > 0:
            legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=3,
                                            label=f'Partido en Casa ({total_casa})'))
        
        if total_fuera > 0:
            legend_elements.append(plt.Line2D([0], [0], color='blue', linewidth=3,
                                            label=f'Partido Fuera ({total_fuera})'))
        
        # Información sobre tipos de lanzamiento
        legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=0,
                                        label='Curvatura: Abierto (→) | Cerrado (←) | Plano (—)'))
        
        if legend_elements:
            # Crear leyenda flotante centrada (ajustada para layout 2x4)
            ax_legend = fig.add_axes([0.25, -0.01, 0.5, 0.12])  # Posición entre las filas
            ax_legend.axis('off')
            
            legend = ax_legend.legend(handles=legend_elements, 
                                    loc='center', ncol=2,
                                    framealpha=0.95, facecolor='white', 
                                    edgecolor='#1e3d59', fontsize=9,
                                    title='Faltas Indirectas por Categoría (Pass End X ≥ 50)',
                                    title_fontproperties={'weight': 'bold', 'size': 10})
            
            if legend.get_title():
                legend.get_title().set_color('#1e3d59')

    def create_campograma_categoria(self, ax, categoria='frontales', jornadas_filter=None):
        """Campograma con filtro de jornadas"""
        sequences = self.get_freekick_indirect_sequences(jornadas_filter=jornadas_filter)
        categorized = self.categorizar_faltas(sequences)
        
        seq_list = categorized[categoria]
        titulo = self.CATEGORIA_NAMES[categoria]
        
        if not seq_list:
            ax.text(0.5, 0.5, f'{titulo}\n\n(Sin datos)', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        pitch = VerticalPitch(half=True, pitch_type='opta', pitch_color='#2d5a27', 
                            line_color='white', linewidth=2)
        pitch.draw(ax=ax)
        
        ax.set_title(titulo, fontsize=11, fontweight='bold', color='#1e3d59', pad=8)
        
        # Dibujar cada lanzamiento
        for seq in seq_list:
            fk_event = seq['freekick_event']
            
            # Verificar que las coordenadas existen y son válidas
            try:
                start_x = float(fk_event.get('x', 0))
                start_y = float(fk_event.get('y', 0))
                end_x = float(fk_event.get('Pass End X', 0))
                end_y = float(fk_event.get('Pass End Y', 0))
                
                # Saltar si alguna coordenada es 0 o inválida
                if start_x == 0 or start_y == 0 or end_x == 0 or end_y == 0:
                    continue
                    
            except (ValueError, TypeError):
                continue

            if end_x < 50:
                continue
            
            # Determinar color según local/visitante
            is_home_match = fk_event.get('Is Home', False) # Usamos .get para seguridad
            arrow_color = 'red' if is_home_match else 'blue'
            stroke_color = 'white' if arrow_color in ['red', 'blue'] else 'black'
            
            # Coordenadas de inicio y fin
            start_coords = (start_y, start_x)
            end_coords = (end_y, end_x)
            
            # Determinar curvatura según tipo de lanzamiento
            tipo_lanz = self.get_tipo_lanzamiento(fk_event)
            base_rad = 0 # Valor por defecto para pases rectos o desconocidos

            # El centro del campo en el eje Y de Opta es 50.
            if start_y >= 50:  # Lanzamiento desde el lado DERECHO
                if tipo_lanz == 'abierto':
                    base_rad = -0.3  # Curva a la derecha (out-swinger)
                elif tipo_lanz == 'cerrado':
                    base_rad = 0.3   # Curva a la izquierda (in-swinger)
            
            else:  # Lanzamiento desde el lado IZQUIERDO (start_y < 50)
                if tipo_lanz == 'abierto':
                    base_rad = 0.3   # Curva a la izquierda (out-swinger)
                elif tipo_lanz == 'cerrado':
                    base_rad = -0.3  # Curva a la derecha (in-swinger)

            # Crear y dibujar la flecha curva
            arrow = FancyArrowPatch(start_coords, end_coords,
                                    connectionstyle=f"arc3,rad={base_rad}",
                                    arrowstyle='->', mutation_scale=15,
                                    color=arrow_color, alpha=0.8, linewidth=2.5, 
                                    zorder=5, clip_on=False)
            ax.add_patch(arrow)
            
            # Colocar dorsal del lanzador en medio de la flecha
            if pd.notna(fk_event.get('dorsal_lanzador')):
                # Calcular punto medio de la flecha curva
                mid_y = (start_coords[0] + end_coords[0]) / 2
                mid_x = (start_coords[1] + end_coords[1]) / 2
                
                # Calcular desplazamiento para el arco
                vec_y = end_coords[0] - start_coords[0]
                vec_x = end_coords[1] - start_coords[1]
                perp_vec_y = -vec_x
                perp_vec_x = vec_y
                
                offset_y = (perp_vec_y * base_rad) / 2
                offset_x = (perp_vec_x * base_rad) / 2
                
                mid_y_adjusted = mid_y + offset_y
                mid_x_adjusted = mid_x + offset_x
                
                ax.text(mid_y_adjusted, mid_x_adjusted, str(int(fk_event['dorsal_lanzador'])),
                    fontsize=8, fontweight='bold', color=arrow_color, 
                    ha='center', va='center', zorder=6,
                    path_effects=[patheffects.withStroke(linewidth=3, foreground=stroke_color)])

    def create_reporte_campogramas(self, figsize=(11.69, 8.27), jornadas_filter=None):    
        """Crea reporte con campogramas en layout 2x4: frontales y laterales arriba, alejadas abajo centrado"""
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        # Grid 2x4 para el nuevo layout
        gs = fig.add_gridspec(2, 4, 
                            height_ratios=[1, 1],
                            width_ratios=[1, 1, 1, 1],
                            hspace=0.25,
                            wspace=0.15,
                            left=0.04, right=0.96,
                            top=0.88, bottom=0.08)
        
        # Fondo
        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25)
            ax_bg.axis('off')
        
        # Título principal
        fig.suptitle(f'FALTAS INDIRECTAS - LANZAMIENTOS', 
                    fontsize=20, fontweight='bold', color='#1e3d59', y=0.96)
        
        # Subtítulo con jornadas
        if jornadas_filter:
            jornadas_texto = f"J{min(jornadas_filter)} - J{max(jornadas_filter)}"
            fig.text(0.5, 0.91, f'Últimas 5 jornadas ({jornadas_texto})', 
                    ha='center', fontsize=12, color='#34495e', style='italic')
        
        # Logo del equipo si hay filtro
        if self.team_filter and (team_logo := self.load_team_logo(self.team_filter)) is not None:
            ax_team = fig.add_axes([0.86, 0.89, 0.08, 0.1])
            ax_team.imshow(team_logo, aspect='auto')
            ax_team.axis('off')

        # FILA SUPERIOR: Frontales (izquierda) y Laterales (derecha)
        ax_frontales = fig.add_subplot(gs[0, 0:2])  # Arriba izquierda
        self.create_campograma_categoria(ax_frontales, 'frontales', jornadas_filter)
        ax_frontales.set_aspect('auto') 

        ax_laterales = fig.add_subplot(gs[0, 2:4])  # Arriba derecha
        self.create_campograma_categoria(ax_laterales, 'laterales', jornadas_filter)
        ax_laterales.set_aspect('auto') 

        # FILA INFERIOR: Alejadas (centrado, ocupando 2 columnas del medio)
        ax_alejadas = fig.add_subplot(gs[1, 1:3])  # Abajo centro (columnas 1-2)
        self.create_campograma_categoria(ax_alejadas, 'alejadas', jornadas_filter)
        ax_alejadas.set_aspect('auto') 

        # Leyenda conjunta (ajustar posición para el nuevo layout)
        self.create_leyenda_conjunta_campogramas(fig, gs, jornadas_filter)

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
        print("=== GENERADOR DE CAMPOGRAMAS FALTAS INDIRECTAS ===")
        if (equipo := seleccionar_equipo_interactivo()) is None:
            print("No se pudo completar la selección.")
            return
        
        analyzer = ReporteCampogramasFaltas(team_filter=equipo)
        
        # DEBUG: Verificar datos cargados
        analyzer.debug_data_loading()
        
        # SELECCIÓN AUTOMÁTICA DE LAS 5 JORNADAS MÁS RECIENTES
        jornadas_disponibles = sorted(analyzer.jornadas_disponibles, reverse=True)
        jornadas_filtro = jornadas_disponibles[:5]

        if not jornadas_filtro:
            print("⚠️ No hay jornadas disponibles para generar el reporte.")
            return
        
        # Crear título descriptivo
        jornadas_texto = f"J{min(jornadas_filtro)}-{max(jornadas_filtro)}"
        
        print(f"\nGenerando campogramas para {equipo} - Jornadas: {jornadas_texto}")
        print(f"Total jornadas acumuladas: {len(jornadas_filtro)}")
        
        # Pasar jornadas_filtro como parámetro
        if (fig := analyzer.create_reporte_campogramas(jornadas_filter=jornadas_filtro)):
            plt.show()
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_campogramas_faltas_{equipo_filename}_{jornadas_texto}.pdf"
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0, 
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
        analyzer = ReporteCampogramasFaltas(team_filter=equipo)
        fig = analyzer.create_reporte_campogramas()
        
        if fig:
            if mostrar: 
                plt.show()
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"reporte_campogramas_faltas_{equipo_filename}.pdf"
                fig.savefig(output_path, bbox_inches='tight', pad_inches=0, 
                           facecolor='white', dpi=300, orientation='landscape')
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