import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

# CONFIGURACIÓN GLOBAL PARA ELIMINAR ESPACIOS
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

# Instalar mplsoccer si no está instalado
try:
    from mplsoccer import Pitch
except ImportError:
    pass
    import subprocess
    subprocess.check_call(["pip", "install", "mplsoccer"])
    from mplsoccer import Pitch

class CampoFutbolBarras:
    def __init__(self, data_path="extraccion_mediacoach/data/rendimiento_fisico.parquet"):
        """
        Inicializa la clase para generar informes con gráficos de barras por demarcación
        """
        self.data_path = data_path
        self.df = None
        self.load_data()
        self.clean_team_names()
        self.opta_df = None
        self.load_opta_positions()
        
        # 🎨 COLORES ESPECÍFICOS POR EQUIPO
        self.team_colors = {
            'Athletic Club': {'primary': '#EE2E24', 'secondary': '#FFFFFF', 'text': 'white'},
            'Atlético de Madrid': {'primary': '#CB3524', 'secondary': '#FFFFFF', 'text': 'white'},
            'CA Osasuna': {'primary': '#D2001C', 'secondary': '#001A4B', 'text': 'white'},
            'CD Leganés': {'primary': '#004C9F', 'secondary': '#FFFFFF', 'text': 'white'},
            'Deportivo Alavés': {'primary': '#1F4788', 'secondary': '#FFFFFF', 'text': 'white'},
            'FC Barcelona': {'primary': '#004D98', 'secondary': '#A50044', 'text': 'white'},
            'Getafe CF': {'primary': '#005CA9', 'secondary': '#FFFFFF', 'text': 'white'},
            'Girona FC': {'primary': '#CC0000', 'secondary': '#FFFFFF', 'text': 'white'},
            'RC Celta': {'primary': '#87CEEB', 'secondary': '#FFFFFF', 'text': 'black'},
            'RCD Espanyol': {'primary': '#004C9F', 'secondary': '#FFFFFF', 'text': 'white'},
            'RCD Mallorca': {'primary': '#CC0000', 'secondary': '#FFFF00', 'text': 'white'},
            'Rayo Vallecano': {'primary': '#CC0000', 'secondary': '#FFFFFF', 'text': 'white'},
            'Real Betis': {'primary': '#00954C', 'secondary': '#FFFFFF', 'text': 'white'},
            'Real Madrid': {'primary': '#FFFFFF', 'secondary': '#FFD700', 'text': 'black'},
            'Real Sociedad': {'primary': '#004C9F', 'secondary': '#FFFFFF', 'text': 'white'},
            'Real Valladolid CF': {'primary': '#663399', 'secondary': '#FFFFFF', 'text': 'white'},
            'Sevilla FC': {'primary': '#D2001C', 'secondary': '#FFFFFF', 'text': 'white'},
            'UD Las Palmas': {'primary': '#FFFF00', 'secondary': '#004C9F', 'text': 'black'},
            'Valencia CF': {'primary': '#FF7F00', 'secondary': '#000000', 'text': 'white'},
            'Villarreal CF': {'primary': '#FFD700', 'secondary': '#004C9F', 'text': 'black'},
        }

        # 🎯 DICCIONARIO DE ANULACIONES MANUALES DE POSICIÓN (puede estar vacío)
        self.player_position_overrides = {
            # ('NombreJugador', 'NombreEquipo'): 'POSICION_MANUAL'
        }

        # Colores por defecto para equipos no reconocidos
        self.default_team_colors = {'primary': '#2c3e50', 'secondary': '#FFFFFF', 'text': 'white'}

        # Mapeo exacto basado en las demarcaciones encontradas
        self.demarcacion_to_position = {
            # Portero (queda igual)
            'Portero': 'PORTERO',
            
            # Defensas - Posiciones específicas
            'Defensa - Central Derecho': 'CENTRAL_DERECHO',
            'Defensa - Lateral Derecho': 'LATERAL_DERECHO', 
            'Defensa - Central Izquierdo': 'CENTRAL_IZQUIERDO',
            'Defensa - Lateral Izquierdo': 'LATERAL_IZQUIERDO',
            
            # Mediocampo - Posiciones específicas
            'Centrocampista - MC Box to Box': 'MC_BOX_TO_BOX',
            'Centrocampista - MC Organizador': 'MC_ORGANIZADOR',
            'Centrocampista - MC Posicional': 'MC_POSICIONAL',
            'Centrocampista de ataque - Banda Derecha': 'BANDA_DERECHA',
            'Centrocampista de ataque - Banda Izquierda': 'BANDA_IZQUIERDA',
            'Centrocampista de ataque - Mediapunta': 'MC_BOX_TO_BOX',

            # Delanteros - Dos posiciones diferenciadas
            'Delantero - Delantero Centro': 'DELANTERO_CENTRO',
            'Delantero - Segundo Delantero': 'DELANTERO_CENTRO',
            
            # Jugadores sin posición definida
            'Sin Posición': 'MC_POSICIONAL',
        }
        
        # Coordenadas específicas para cada posición en el campo
        self.coordenadas_graficos = {
            # Villarreal (lado izquierdo)
            'villarreal': {
                'PORTERO': (10, 40),              # Portería
                'LATERAL_DERECHO': (25, 12),      # Lateral derecho (arriba)
                'CENTRAL_DERECHO': (20, 25),      # Central derecho (centro-arriba)
                'CENTRAL_IZQUIERDO': (20, 53),    # Central izquierdo (centro-abajo)
                'LATERAL_IZQUIERDO': (25, 68),    # Lateral izquierdo (abajo)
                'MC_POSICIONAL': (30, 40),        # Mediocampo defensivo (centro)
                'MC_BOX_TO_BOX': (62, 55),        # Box to box (centro-arriba)
                'MC_ORGANIZADOR': (50, 40),       # Organizador (centro-abajo)
                'BANDA_DERECHA': (70, 12),        # Banda derecha (extremo arriba)
                'BANDA_IZQUIERDA': (70, 68),      # Banda izquierda (extremo abajo)
                'DELANTERO_CENTRO': (85, 55),     # Delantero centro (arriba)
                'SEGUNDO_DELANTERO': (85, 25),    # Segundo delantero (abajo)
            },
            # Equipo rival (lado derecho - espejo)
            'rival': {
                'PORTERO': (110, 40),             # Portería
                'LATERAL_DERECHO': (100, 68),      # Lateral derecho (abajo - espejo)
                'CENTRAL_DERECHO': (105, 53),      # Central derecho (centro-abajo - espejo)
                'CENTRAL_IZQUIERDO': (105, 25),    # Central izquierdo (centro-arriba - espejo)
                'LATERAL_IZQUIERDO': (100, 12),    # Lateral izquierdo (arriba - espejo)
                'MC_POSICIONAL': (90, 40),        # Mediocampo defensivo (centro)
                'MC_BOX_TO_BOX': (60, 25),        # Box to box (centro-abajo - espejo)
                'MC_ORGANIZADOR': (70, 40),       # Organizador (centro-arriba - espejo)
                'BANDA_DERECHA': (45, 68),        # Banda derecha (extremo abajo - espejo)
                'BANDA_IZQUIERDA': (45, 12),      # Banda izquierda (extremo arriba - espejo)
                'DELANTERO_CENTRO': (38, 25),     # Delantero centro (abajo - espejo)
                'SEGUNDO_DELANTERO': (38, 53),    # Segundo delantero (arriba - espejo)
            }
        }
        
        # 🔥 MÉTRICAS PARA LOS GRÁFICOS DE BARRAS
        self.metricas_barras = [
            'Distancia Total 14-21 km / h',
            'Distancia Total >21 km / h', 
            'Distancia Total >24 km / h'
        ]
        
        # Métrica para el punto rojo
        self.metrica_punto_rojo = 'Velocidad Máxima Total'
        
        # 🔥 MÉTRICAS AMPLIADAS PARA RESUMEN DE EQUIPOS
        self.metricas_resumen_equipos = [
            'Distancia Total 14-21 km / h',
            'Distancia Total >21 km / h',
            'Distancia Total >24 km / h',
            'Velocidad Máxima Total'
        ]
        
        # Colores para las barras de métricas
        self.colores_barras = [
            '#45B7D1',  # Azul claro para 14-21 km/h
            '#4ECDC4',  # Turquesa para >21 km/h
            '#FF7F50',  # Naranja para >24 km/h
        ]
        
    def load_data(self):
        """Carga los datos del archivo parquet"""
        try:
            self.df = pd.read_parquet(self.data_path)
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")
            
    def similarity(self, a, b):
        """Calcula la similitud entre dos strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def clean_team_names(self):
        """Limpia y agrupa nombres de equipos similares"""
        if self.df is None:
            return
        
        # Limpiar nombres de equipos
        unique_teams = self.df['Equipo'].unique()
        team_mapping = {}
        processed_teams = set()
        
        for team in unique_teams:
            if team in processed_teams:
                continue
                
            # Buscar equipos similares
            similar_teams = [team]
            for other_team in unique_teams:
                if other_team != team and other_team not in processed_teams:
                    if self.similarity(team, other_team) > 0.85:
                        similar_teams.append(other_team)
            
            # Elegir el nombre más largo como representativo
            canonical_name = max(similar_teams, key=len)
            
            # Mapear todos los nombres similares al canónico
            for similar_team in similar_teams:
                team_mapping[similar_team] = canonical_name
                processed_teams.add(similar_team)
        
        # Aplicar el mapeo
        self.df['Equipo'] = self.df['Equipo'].map(team_mapping)
        
        # Normalizar jornadas
        def normalize_jornada(jornada):
            if isinstance(jornada, str) and jornada.startswith('J'):
                try:
                    return int(jornada[1:])
                except ValueError:
                    return jornada
            elif isinstance(jornada, str) and jornada.startswith('j'):
                try:
                    return int(jornada[1:])
                except ValueError:
                    return jornada
            return jornada
        
        self.df['Jornada'] = self.df['Jornada'].apply(normalize_jornada)

    def load_opta_positions(self):
        """Carga las posiciones desde el archivo Opta"""
        try:
            opta_path = "extraccion_opta/datos_opta_parquet/player_stats.parquet"
            self.opta_df = pd.read_parquet(opta_path)
            
            # Verificar columnas necesarias
            required_columns = ['Match Name', 'Team Name', 'Position', 'Position Side']
            optional_columns = ['Shirt Number']
            
            missing_required = [col for col in required_columns if col not in self.opta_df.columns]
            if missing_required:
                print(f"⚠️ Columnas requeridas faltantes en Opta: {missing_required}")
                self.opta_df = None
                return False
            
            available_optional = [col for col in optional_columns if col in self.opta_df.columns]
            if available_optional:
                pass
            
            # Normalizar nombres de equipos en Opta
            if 'Team Name' in self.opta_df.columns:
                self.opta_df['Team Name Normalized'] = self.opta_df['Team Name'].apply(self.normalize_text)
            
            return True
        except Exception as e:
            print(f"⚠️ Error al cargar datos Opta: {e}")
            self.opta_df = None
            return False

    def are_teams_equivalent(self, team1, team2):
        """
        Compara dos nombres de equipo de forma inteligente.
        """
        if not team1 or not team2:
            return False

        # Normalizar
        norm1 = self.normalize_text(team1)
        norm2 = self.normalize_text(team2)
        
        # ✅ MATCH EXACTO PRIMERO
        if norm1 == norm2:
            return True

        # Palabras comunes a filtrar
        common_words = {
            'club', 'cf', 'fc', 'de', 'del', 'la', 'el', 'deportivo',
            'sociedad', 'union', 'racing', 'sporting', 'gimnastic', 'cd', 'ud', 'rcd'
        }
        
        # Filtrar palabras
        words1 = [w for w in norm1.split() if w not in common_words and len(w) > 2]
        words2 = [w for w in norm2.split() if w not in common_words and len(w) > 2]
        
        # Si no quedan palabras, usar original
        if not words1 or not words2:
            words1 = [w for w in norm1.split() if len(w) > 1]
            words2 = [w for w in norm2.split() if len(w) > 1]
        
        words1_set = set(words1)
        words2_set = set(words2)
        
        # ✅ VERIFICAR QUE TODAS LAS PALABRAS COINCIDAN
        if words1_set == words2_set and len(words1_set) > 0:
            return True
        
        # Subset solo si hay MÁS de una palabra en común
        common = words1_set & words2_set
        if len(common) > 1 and (words1_set.issubset(words2_set) or words2_set.issubset(words1_set)):
            return True

        return False
            
    def map_opta_position_to_system(self, position, position_side, team_name=None, week=None):
        """Mapea posición Opta al sistema interno"""
        if pd.isna(position):
            return None
        
        position = str(position).strip()
        position_side = str(position_side).strip() if pd.notna(position_side) else ""
            
        # Mapeo según las reglas especificadas
        if position == "Goalkeeper":
            return "PORTERO"
        
        elif position == "Defender":
            if "Centre/Right" in position_side:
                return "CENTRAL_DERECHO"
            elif "Left/Centre" in position_side:
                return "CENTRAL_IZQUIERDO"
            elif "Left" in position_side:
                return "LATERAL_IZQUIERDO"
            elif "Right" in position_side:
                return "LATERAL_DERECHO"
            else:
                return "CENTRAL_DERECHO"  # Por defecto

        elif position == "Defensive Midfielder":
            if "Centre" in position_side:
                return "MC_POSICIONAL"
            elif "Left" in position_side:
                return "MC_POSICIONAL"
            else:
                return "MC_POSICIONAL" 
        
        elif position == "Attacking Midfielder":
            if "Right" in position_side:
                return "BANDA_DERECHA"
            elif "Left" in position_side:
                return "BANDA_IZQUIERDA"
            elif "Centre" in position_side:
                return "MC_BOX_TO_BOX"
            else:
                return "MC_BOX_TO_BOX"

        elif position == "Midfielder":
            # 1. Casos específicos con tendencia a un lado siguen siendo MC_ORGANIZADOR
            if "Centre/Right" in position_side:
                return "MC_ORGANIZADOR"
            elif "Left/Centre" in position_side:
                return "MC_ORGANIZADOR"
            
            # 2. Jugadores de banda
            elif "Right" in position_side:
                return "BANDA_DERECHA"
            elif "Left" in position_side:
                return "BANDA_IZQUIERDA"

            # 3. CAMBIO CLAVE: El caso puramente central ahora es MC_POSICIONAL
            elif "Centre" in position_side:
                return "MC_POSICIONAL"
            
            # 4. Y si no hay información de lado, asumimos también MC_POSICIONAL
            else:
                return "MC_POSICIONAL"
        
        elif position == "Striker":
            # Verificar si hay Striker Centre en la misma jornada
            has_centre = False
            if team_name and week:
                has_centre = self.has_striker_centre_in_match(team_name, week)
            
            if "Centre" in position_side:
                return "DELANTERO_CENTRO"
            elif "Centre/Right" in position_side:
                return "BANDA_DERECHA" if has_centre else "DELANTERO_CENTRO"
            elif "Left/Centre" in position_side:
                return "BANDA_IZQUIERDA" if has_centre else "DELANTERO_CENTRO"
            else:
                return "DELANTERO_CENTRO"
        
        else:
            return None

    def convert_jornada_to_week(self, jornada):
        """Convierte jornada MediaCoach (j1, j2) a Week Opta (1, 2)"""
        if not jornada:
            return None
        
        try:
            if isinstance(jornada, str):
                if jornada.lower().startswith('j'):
                    return int(jornada[1:])
                else:
                    return int(jornada)
            else:
                return int(jornada)
        except (ValueError, TypeError):
            print(f"   ⚠️ Error convirtiendo jornada: {jornada}")
            return None
    
    def has_striker_centre_in_match(self, team_name, week):
        """Verifica si existe un Striker Centre en el equipo y jornada específica"""
        if self.opta_df is None:
            return False
        
        match_data = self.opta_df[
            (self.opta_df['Team Name'].apply(lambda x: self.are_teams_equivalent(team_name, x))) &
            (self.opta_df['Week'] == week)
        ]
        
        return any(
            (row['Position'] == 'Striker') and ('Centre' in str(row.get('Position Side', '')))
            for _, row in match_data.iterrows()
        )

    def find_improved_position(self, player_alias, team_name, player_dorsal=None, jornada=None):
        """Busca posición en Opta usando SOLO DORSAL + JORNADA + EQUIPO"""
        if self.opta_df is None:
            return None
        
        if player_dorsal is None:
            pass
            return None
        
        opta_week = self.convert_jornada_to_week(jornada)
        if opta_week is None:
            pass
            return None
        
        
        for _, opta_player in self.opta_df.iterrows():
            # 1. VERIFICAR DORSAL
            dorsal_match = False
            if 'Shirt Number' in opta_player:
                try:
                    opta_shirt_number = opta_player['Shirt Number']
                    if pd.notna(opta_shirt_number):
                        player_dorsal_clean = str(int(float(player_dorsal)))
                        opta_dorsal_clean = str(int(float(opta_shirt_number)))
                        if player_dorsal_clean == opta_dorsal_clean:
                            dorsal_match = True
                except (ValueError, TypeError):
                    continue
            
            if not dorsal_match:
                continue
            
            # 2. VERIFICAR JORNADA/WEEK
            week_match = False
            if 'Week' in opta_player:
                try:
                    opta_week_value = opta_player['Week']
                    if pd.notna(opta_week_value) and int(opta_week_value) == opta_week:
                        week_match = True
                except (ValueError, TypeError):
                    continue
            
            if not week_match:
                continue
            
            # 3. VERIFICAR EQUIPO
            team_match = False
            if 'Team Name' in opta_player:
                opta_team = str(opta_player.get('Team Name', ''))
                team_match = self.are_teams_equivalent(team_name, opta_team)

            if not team_match:
                continue
            

            opta_position_raw = opta_player.get('Position')
            opta_position_side_raw = opta_player.get('Position Side')
            
            
            if pd.isna(opta_position_raw) or opta_position_raw == "Substitute" or str(opta_position_raw).strip() == "":
                pass
                return None
                
            opta_position = self.map_opta_position_to_system(
                opta_position_raw, 
                opta_position_side_raw,
                team_name,
                opta_week,
            )

            if opta_position:
                pass
                return opta_position
            else:
                print(f"   ❌ {player_alias}: Posición Opta no mapeada: {opta_position_raw} {opta_position_side_raw}")
                return None
        
        print(f"   ❌ {player_alias}: Sin triple match en Opta")
        return None

    @staticmethod
    def normalize_text(text):
        """Normaliza texto eliminando acentos, espacios extra y caracteres especiales"""
        import re
        import unicodedata
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_available_teams(self):
        """Retorna la lista de equipos disponibles"""
        if self.df is None:
            return []
        return sorted(self.df['Equipo'].unique())
    
    def get_available_jornadas(self, equipo=None):
        """Retorna las jornadas disponibles"""
        if self.df is None:
            return []
        
        if equipo:
            filtered_df = self.df[self.df['Equipo'] == equipo]
            return sorted(filtered_df['Jornada'].unique())
        else:
            return sorted(self.df['Jornada'].unique())
    
    def fill_missing_demarcaciones(self, df):
        """Rellena demarcaciones vacías con la más frecuente para cada jugador"""
        
        # Crear copia para trabajar
        df_work = df.copy()
        
        # Identificar registros con demarcación vacía
        mask_empty = df_work['Demarcacion'].isna() | (df_work['Demarcacion'] == '') | (df_work['Demarcacion'].str.strip() == '')
        empty_count = mask_empty.sum()
        
        if empty_count > 0:
            pass
            
            # Para cada jugador con demarcación vacía, buscar su demarcación más frecuente
            for idx in df_work[mask_empty].index:
                jugador_id = df_work.loc[idx, 'Id Jugador']
                jugador_alias = df_work.loc[idx, 'Alias']
                
                # Buscar todas las demarcaciones de este jugador (no vacías)
                jugador_demarcaciones = self.df[
                    (self.df['Id Jugador'] == jugador_id) & 
                    (self.df['Demarcacion'].notna()) & 
                    (self.df['Demarcacion'] != '') &
                    (self.df['Demarcacion'].str.strip() != '')
                ]['Demarcacion']
                
                if len(jugador_demarcaciones) > 0:
                    # Usar la demarcación más frecuente
                    demarcacion_mas_frecuente = jugador_demarcaciones.value_counts().index[0]
                    df_work.loc[idx, 'Demarcacion'] = demarcacion_mas_frecuente
                else:
                    # Si no hay datos históricos, asignar "Sin Posición"
                    df_work.loc[idx, 'Demarcacion'] = 'Sin Posición'
                    print(f"   ⚠️  {jugador_alias}: Sin posición histórica -> MC Posicional")
        
        return df_work
    
    def filter_and_accumulate_data(self, equipo, jornadas, min_avg_minutes=60):
        """Filtra por promedio de minutos y acumula datos por jugador CON POSICIONES OPTA"""
        if self.df is None:
            return None
        
        normalized_jornadas = []
        for jornada in jornadas:
            if isinstance(jornada, str) and (jornada.startswith('J') or jornada.startswith('j')):
                try:
                    normalized_jornadas.append(int(jornada[1:]))
                except ValueError:
                    normalized_jornadas.append(jornada)
            else:
                normalized_jornadas.append(jornada)
        
        filtered_df = self.df[
            (self.df['Equipo'] == equipo) & 
            (self.df['Jornada'].isin(normalized_jornadas))
        ].copy()
        
        filtered_df = self.fill_missing_demarcaciones(filtered_df)
        
        # PASO 1: Buscar posiciones Opta para cada partido individual
        filtered_df['Opta_Position'] = None
        
        for idx, row in filtered_df.iterrows():
            opta_position = self.find_improved_position(
                row.get('Alias', ''),
                row.get('Equipo', ''),
                row.get('Dorsal'),
                row.get('Jornada')
            )
            if opta_position:
                filtered_df.loc[idx, 'Opta_Position'] = opta_position
        
        if 'Nombre' in filtered_df.columns:
            mask_empty_alias = filtered_df['Alias'].isna() | (filtered_df['Alias'] == '') | (filtered_df['Alias'].str.strip() == '')
            filtered_df.loc[mask_empty_alias, 'Alias'] = filtered_df.loc[mask_empty_alias, 'Nombre']
        
        if 'Minutos jugados' not in filtered_df.columns:
            print("⚠️ Columna 'Minutos jugados' no encontrada.")
            return None
        
        # PASO 2: Acumular datos y determinar la posición final
        accumulated_data = []
        
        for jugador in filtered_df['Alias'].unique():
            jugador_data = filtered_df[filtered_df['Alias'] == jugador]
            jugador_data_filtered = jugador_data[jugador_data['Minutos jugados'] >= min_avg_minutes]
            
            if len(jugador_data_filtered) > 0:
                latest_record = jugador_data_filtered.iloc[-1]

                # LÓGICA DE POSICIÓN FINAL (DE FISICO10)
                opta_positions = jugador_data_filtered['Opta_Position'].dropna()
                if len(opta_positions) > 0:
                    final_position = opta_positions.mode().iloc[0]
                    position_source = "OPTA"
                else:
                    demarcacion = jugador_data_filtered['Demarcacion'].mode().iloc[0] if len(jugador_data_filtered['Demarcacion'].mode()) > 0 else 'Sin Posición'
                    final_position = self.demarcacion_to_position.get(demarcacion, 'MC_BOX_TO_BOX')
                    position_source = "MediaCoach Fallback"
                
                # 🎯 PASO FINAL: APLICAR ANULACIONES MANUALES SI EXISTEN
                player_key = (latest_record['Alias'], latest_record['Equipo'])
                if player_key in self.player_position_overrides:
                    new_position = self.player_position_overrides[player_key]
                    final_position = new_position
                    position_source = "Manual Override"

                accumulated_record = {
                    'Id Jugador': latest_record['Id Jugador'],
                    'Dorsal': latest_record['Dorsal'],
                    'Nombre': latest_record['Nombre'],
                    'Alias': latest_record['Alias'],
                    'Final_Position': final_position,
                    'Position_Source': position_source,
                    'Demarcacion': jugador_data_filtered['Demarcacion'].mode().iloc[0] if len(jugador_data_filtered['Demarcacion'].mode()) > 0 else latest_record['Demarcacion'],
                    'Equipo': latest_record['Equipo'],
                    'Minutos jugados': jugador_data_filtered['Minutos jugados'].mean(),
                    'Distancia Total': jugador_data_filtered['Distancia Total'].sum(),
                    'Distancia Total 14-21 km / h': jugador_data_filtered['Distancia Total 14-21 km / h'].sum(),
                    'Distancia Total >21 km / h': jugador_data_filtered['Distancia Total >21 km / h'].sum(),
                    'Distancia Total >24 km / h': jugador_data_filtered.get('Distancia Total >24 km / h', pd.Series([0])).sum(),
                    'Velocidad Máxima Total': jugador_data_filtered['Velocidad Máxima Total'].max(),
                }
                
                accumulated_data.append(accumulated_record)
        
        if accumulated_data:
            result_df = pd.DataFrame(accumulated_data)
            
            # 🔥 DEDUPLICACIÓN POR DORSAL (mantener el que tenga posición Opta o más minutos)
            duplicated_dorsals = result_df[result_df.duplicated('Dorsal', keep=False)]
            
            if len(duplicated_dorsals) > 0:
                print(f"⚠️ Encontrados {len(duplicated_dorsals)} jugadores con dorsales duplicados. Deduplicando...")
                
                # Para cada dorsal duplicado, mantener solo uno
                to_remove = []
                for dorsal in duplicated_dorsals['Dorsal'].unique():
                    duplicates = result_df[result_df['Dorsal'] == dorsal]
                    
                    # Prioridad: 1. Opta, 2. Más minutos
                    opta_players = duplicates[duplicates['Position_Source'] == 'OPTA']
                    
                    if len(opta_players) > 0:
                        # Mantener el de Opta con más minutos
                        keep_idx = opta_players['Minutos jugados'].idxmax()
                    else:
                        # Mantener el con más minutos
                        keep_idx = duplicates['Minutos jugados'].idxmax()
                    
                    # Marcar los demás para eliminar
                    for idx in duplicates.index:
                        if idx != keep_idx:
                            to_remove.append(idx)
                
                # Eliminar duplicados
                result_df = result_df.drop(to_remove)
            
            return result_df
        else:
            print(f"❌ No hay jugadores con al menos 1 partido de {min_avg_minutes}+ minutos para {equipo}")
            return None
    
    def load_team_logo(self, equipo):
        """
        Carga el escudo del equipo con una potente lógica de búsqueda jerárquica.
        1. Mapeo Manual: Anula todo lo demás para casos conflictivos.
        2. Coincidencia Exacta: Busca una correspondencia perfecta entre nombres normalizados.
        3. Coincidencia de Palabra Larga: Busca si una palabra significativa (>4 letras) coincide.
        4. Similitud: Como último recurso, busca nombres con una alta similitud de texto.
        """
        escudos_dir = "assets/escudos"
        if not os.path.exists(escudos_dir):
            pass
            return None

        # --- Nivel 1: MAPEO MANUAL (Máxima Prioridad) ---
        # Edita esta sección para forzar la correspondencia de equipos problemáticos.
        # ¡Asegúrate de que este diccionario esté igual en todos tus scripts!
        TEAM_LOGO_MAP = {
            self.normalize_text("Athletic Club"): "Athletic",
            self.normalize_text("Atletico de Madrid"): "Atlético",
            # self.normalize_text("Real Betis Balompie"): "betis", # Ejemplo
        }
        
        equipo_norm = self.normalize_text(equipo)
        if equipo_norm in TEAM_LOGO_MAP:
            logo_filename = TEAM_LOGO_MAP[equipo_norm]
            for ext in ['.png', '.jpg', '.jpeg']:
                logo_path = os.path.join(escudos_dir, f"{logo_filename}{ext}")
                if os.path.exists(logo_path):
                    pass
                    try:
                        return plt.imread(logo_path)
                    except Exception as e:
                        pass
            print(f"⚠️ Advertencia: El archivo mapeado '{logo_filename}' no fue encontrado.")

        # --- Búsqueda Automática ---
        available_files = [f for f in os.listdir(escudos_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # --- Nivel 2: COINCIDENCIA EXACTA ---
        for filename in available_files:
            file_base_norm = self.normalize_text(os.path.splitext(filename)[0])
            if file_base_norm == equipo_norm:
                logo_path = os.path.join(escudos_dir, filename)
                try:
                    return plt.imread(logo_path)
                except Exception as e:
                    pass

        # --- Nivel 3: COINCIDENCIA DE PALABRA LARGA ---
        MIN_WORD_LENGTH = 4 # Busca palabras con 5 o más letras
        team_long_words = {word for word in equipo_norm.split() if len(word) > MIN_WORD_LENGTH}

        if team_long_words:
            # Crea un diccionario de búsqueda de archivos normalizados
            normalized_files = {self.normalize_text(os.path.splitext(f)[0]): f for f in available_files}

            for file_norm, original_filename in normalized_files.items():
                file_words = set(file_norm.split())
                
                # Comprueba si alguna palabra larga del equipo está en las palabras del nombre del archivo
                if not team_long_words.isdisjoint(file_words):
                    logo_path = os.path.join(escudos_dir, original_filename)
                    try:
                        return plt.imread(logo_path)
                    except Exception as e:
                        pass

        # --- Nivel 4: BÚSQUEDA POR SIMILITUD (Último Recurso) ---
        best_match_file = None
        best_similarity = 0.88  # Umbral alto para evitar errores

        for filename in available_files:
            file_base_norm = self.normalize_text(os.path.splitext(filename)[0])
            similarity = self.similarity(equipo_norm, file_base_norm)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_file = filename
        
        if best_match_file:
            logo_path = os.path.join(escudos_dir, best_match_file)
            try:
                return plt.imread(logo_path)
            except Exception as e:
                pass

        print(f"❌ No se encontró un escudo definitivo para: {equipo} (normalizado como: {equipo_norm})")
        return None

    def get_team_colors(self, equipo):
        """Obtiene los colores del equipo o devuelve colores por defecto"""
        # Buscar coincidencia exacta primero
        if equipo in self.team_colors:
            return self.team_colors[equipo]
        
        # Buscar coincidencia parcial (por si hay variaciones en el nombre)
        for team_name in self.team_colors.keys():
            if team_name.lower() in equipo.lower() or equipo.lower() in team_name.lower():
                return self.team_colors[team_name]
        
        # Si no encuentra nada, devolver colores por defecto
        print(f"⚠️  Equipo '{equipo}' no reconocido, usando colores por defecto")
        return self.default_team_colors
    
    def group_players_by_final_position(self, filtered_df):
        """Agrupa jugadores por su posición final ya determinada (Opta o fallback)."""
        if filtered_df is None or 'Final_Position' not in filtered_df.columns:
            return {}
            
        
        # Ordenar por Distancia Total para que los gráficos muestren de mayor a menor
        filtered_df_sorted = filtered_df.sort_values('Distancia Total', ascending=False)
        
        grouped_players = {}
        
        # Agrupar por la nueva columna 'Final_Position'
        for position, group in filtered_df_sorted.groupby('Final_Position'):
            grouped_players[position] = group.to_dict('records')
            
        return grouped_players
    
    def create_campo_sin_espacios(self, figsize=(11.69, 8.27)):
        """Crea el campo que ocupe TODA la página sin espacios"""
        
        # Crear pitch sin padding
        pitch = Pitch(
            pitch_color='grass', 
            line_color='white', 
            stripe=True, 
            linewidth=3,
            pad_left=0, pad_right=0, pad_bottom=0, pad_top=0
        )
        
        # Crear figura sin layouts automáticos
        fig, ax = pitch.draw(
            figsize=figsize,
            tight_layout=False,
            constrained_layout=False
        )
        
        # ✅ CONFIGURACIÓN AGRESIVA PARA ELIMINAR TODOS LOS ESPACIOS
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        ax.set_position([0, 0, 1, 1])
        ax.margins(0, 0)
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 80)
        ax.autoscale(enable=False)
        ax.set_aspect('equal')
        fig.patch.set_visible(False)
        ax.set_frame_on(False)
        
        return fig, ax
    
    def get_player_position_history(self, player_id, team_name):
        """Obtiene todas las posiciones donde ha jugado un jugador, ordenadas por frecuencia"""
        player_matches = self.df[
            (self.df['Id Jugador'] == player_id) & 
            (self.df['Equipo'] == team_name) &
            (self.df['Minutos jugados'] >= 45)
        ]
        
        position_counts = {}
        
        for _, match in player_matches.iterrows():
            opta_pos = self.find_improved_position(
                match.get('Alias', ''),
                match.get('Equipo', ''),
                match.get('Dorsal'),
                match.get('Jornada')
            )
            if opta_pos:
                position_counts[opta_pos] = position_counts.get(opta_pos, 0) + 1
            else:
                # Fallback a MediaCoach
                demarcacion = match['Demarcacion']
                mediacoach_pos = self.demarcacion_to_position.get(demarcacion, 'MC_BOX_TO_BOX')
                position_counts[mediacoach_pos] = position_counts.get(mediacoach_pos, 0) + 1
        
        return sorted(position_counts.items(), key=lambda x: x[1], reverse=True)

    def redistribute_overcrowded_positions(self, grouped_players, max_players_per_position=3):
        """
        Redistribuye jugadores según reglas específicas para no exceder el máximo por posición.
        Prioriza mover a jugadores que ya han jugado en la posición de destino.
        Si no, mueve al que menos minutos ha jugado en la posición actual.
        """
        
        redistributed_players = {pos: list(players) for pos, players in grouped_players.items()}

        # Reglas de desbordamiento: de dónde a dónde mover jugadores
        overflow_rules = {
            'MC_POSICIONAL': 'MC_ORGANIZADOR',
            'MC_ORGANIZADOR': 'MC_BOX_TO_BOX',   # <-- ¡AÑADIR ESTA LÍNEA!
            'CENTRAL_DERECHO': 'CENTRAL_IZQUIERDO',
            'CENTRAL_IZQUIERDO': 'CENTRAL_DERECHO'
        }

        made_a_change = True
        while made_a_change:
            made_a_change = False
            
            for position in list(redistributed_players.keys()):
                
                if position in overflow_rules and len(redistributed_players.get(position, [])) > max_players_per_position:
                    
                    target_position = overflow_rules[position]
                    
                    # --- Evitar mover a un destino que ya está lleno ---
                    if len(redistributed_players.get(target_position, [])) >= max_players_per_position:
                        print(f"   ⚠️  Destino {target_position} también lleno. Saltando movimiento desde {position} por ahora.")
                        continue

                    candidates = redistributed_players[position]
                    player_to_move = None
                    
                    eligible_candidates = []
                    for candidate in candidates:
                        position_history = self.get_player_position_history(candidate['Id Jugador'], candidate['Equipo'])
                        historical_positions = [pos[0] for pos in position_history]
                        
                        if target_position in historical_positions:
                            eligible_candidates.append(candidate)
                    
                    if eligible_candidates:
                        eligible_candidates.sort(key=lambda p: p['Minutos jugados'])
                        player_to_move = eligible_candidates[0]

                    if not player_to_move:
                        candidates.sort(key=lambda p: p['Minutos jugados'])
                        player_to_move = candidates[0]

                    if player_to_move:
                        pass
                        
                        redistributed_players[position].remove(player_to_move)
                        
                        if target_position not in redistributed_players:
                            redistributed_players[target_position] = []
                        redistributed_players[target_position].append(player_to_move)
                        
                        made_a_change = True
                        break
            
        for pos, players in redistributed_players.items():
            pass
            
        return redistributed_players
    
    def create_position_graph(self, players_list, demarcacion, x, y, ax, team_colors, team_logo=None):
        """🔥 NUEVA FUNCIÓN: Crea un gráfico de barras para cada demarcación"""
        if not players_list or len(players_list) < 1:
            return
        
        
        if team_colors['primary'] == '#FFD700':
            texto_color = '#001A4B'  # Un azul oscuro
        else:
            texto_color = 'white' # Mantenemos el color blanco para el rival

        # Dimensiones del gráfico
        graph_width = 18
        graph_height = 12
        
        # Fondo del gráfico - AMARILLO PARA VILLARREAL
        if team_colors['primary'] == '#FFD700':  # Color primario de Villarreal
            fondo_color = '#FFD700'  # Amarillo Villarrea
            fondo_alpha = 0.8
        else:
            fondo_color = '#2c3e50'  # Gris oscuro para otros equipos
            fondo_alpha = 0.95

        graph_rect = plt.Rectangle((x - graph_width/2, y - graph_height/2), 
                                graph_width, graph_height,
                                facecolor=fondo_color, alpha=fondo_alpha,
                                edgecolor='white', linewidth=2)
        ax.add_patch(graph_rect)
        
        # Título del gráfico
        if demarcacion == 'MC Box to Box':
            titulo_grafico = 'MC BOX TO BOX + MEDIAPUNTA'
        elif demarcacion == 'Delantero Centro':
            titulo_grafico = 'DELANTERO CENTRO + 2º DELANTERO'
        else:
            titulo_grafico = demarcacion.upper()
        
        # Ajustar tamaño de fuente según longitud
        if len(titulo_grafico) > 20:
            font_size = 7
        elif len(titulo_grafico) > 15:
            font_size = 8
        else:
            font_size = 9

        ax.text(x, y + graph_height/2 - 0.5, titulo_grafico, 
                fontsize=font_size, weight='bold', color='white',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#1e3d59', alpha=0.8))
        
        # 🏆 AÑADIR ESCUDO en la esquina superior izquierda
        if team_logo is not None:
            try:
                logo_x = x - graph_width/2 + 1.0
                logo_y = y + graph_height/2 - 1.5
                zoom_factor = 0.08
                
                imagebox = OffsetImage(team_logo, zoom=zoom_factor)
                ab = AnnotationBbox(imagebox, (logo_x, logo_y), 
                                frameon=False, 
                                boxcoords='data')
                ax.add_artist(ab)
            except Exception as e:
                print(f"⚠️  Error al añadir escudo: {e}")
        
        # Preparar datos para el gráfico
        num_players = len(players_list)
        if num_players < 1:
            return
        
        # Crear posiciones X para los jugadores (ya ordenados por Distancia Total descendente)
        if num_players == 1:
            x_positions = [x]
        else:
            x_positions = np.linspace(x - graph_width/2 + 3, x + graph_width/2 - 3, num_players)
        
        # Calcular rangos para normalización Y de las barras
        y_base = y - graph_height/2 + 3.5
        y_max_bars = y + graph_height/2 - 5
        
        # 🔥 CALCULAR RANGO DE VMAX PARA NORMALIZACIÓN
        vmax_values = [player.get(self.metrica_punto_rojo, 0) for player in players_list]
        min_vmax = min(vmax_values) if vmax_values else 0
        max_vmax = max(vmax_values) if vmax_values else 1
        vmax_range = max_vmax - min_vmax if max_vmax > min_vmax else 1

        # Rango de alturas para puntos rojos
        punto_y_min = y_max_bars + 1
        punto_y_max = y_max_bars + 2
        
        # Recopilar todos los valores para normalización
        all_bar_values = []
        players_bar_data = []
        
        for player in players_list:
            player_values = []
            for metric in self.metricas_barras:
                value = player.get(metric, 0)
                player_values.append(value)
                all_bar_values.append(value)
            players_bar_data.append(player_values)
        
        # Encontrar valor máximo para normalización
        max_bar_value = max(all_bar_values) if all_bar_values else 1
        if max_bar_value == 0:
            max_bar_value = 1
        
        # Ancho de las barras
        bar_width = 1.2
        bar_spacing = 0.3
        total_bar_width = len(self.metricas_barras) * bar_width + (len(self.metricas_barras) - 1) * bar_spacing
        
        # Lista para almacenar posiciones de puntos rojos
        puntos_rojos_x = []
        puntos_rojos_y = []
        
        # 🔥 CREAR BARRAS PARA CADA JUGADOR
        for i, (x_pos, player, player_values) in enumerate(zip(x_positions, players_list, players_bar_data)):
            
            # Posición inicial para las barras de este jugador
            start_x = x_pos - total_bar_width/2
            
            # Crear las 3 barras para cada métrica
            for j, (metric, value, color) in enumerate(zip(self.metricas_barras, player_values, self.colores_barras)):
                bar_x = start_x + j * (bar_width + bar_spacing)
                
                # Altura normalizada de la barra
                bar_height = (value / max_bar_value) * (y_max_bars - y_base)
                
                # Crear la barra
                bar_rect = plt.Rectangle((bar_x, y_base), bar_width, bar_height,
                                    facecolor=color, alpha=0.8, 
                                    edgecolor='white', linewidth=1)
                ax.add_patch(bar_rect)
                
                # Valor encima de cada barra
                ax.text(bar_x + bar_width/2, y_base + bar_height + 0.2, f"{value:.0f}",
                        fontsize=4, color='white', weight='bold',
                        ha='center', va='bottom')
            
            # 🔴 PUNTO ROJO VARIABLE SEGÚN VMAX
            vmax_value = player.get(self.metrica_punto_rojo, 0)
            
            # Altura variable según Vmax
            if vmax_range > 0:
                vmax_normalized = (vmax_value - min_vmax) / vmax_range
            else:
                vmax_normalized = 0.5
            punto_y = punto_y_min + (vmax_normalized * (punto_y_max - punto_y_min))
            
            # Guardar posiciones para líneas discontinuas
            puntos_rojos_x.append(x_pos)
            puntos_rojos_y.append(punto_y)
            
            # Punto rojo
            circle = plt.Circle((x_pos, punto_y), 0.3, color='red', alpha=0.9)
            ax.add_patch(circle)
            
            # Línea roja discontinua conectando
            ax.plot([x_pos, x_pos], [y_max_bars + 0.5, punto_y - 0.3], 
                    color='red', linewidth=2, alpha=0.8, linestyle='--')
            
            # Valor de Vmax en blanco encima del punto
            ax.text(x_pos, punto_y + 0.6, f"{vmax_value:.1f}",
                    fontsize=6, color='white', weight='bold',
                    ha='center', va='bottom')
            
            # --- INICIO: Lógica mejorada para nombres y dorsales ---
            player_name = player.get('Alias', 'N/A')
            dorsal_raw = player.get('Dorsal', '')

            # Formatear dorsal como siempre
            if pd.notna(dorsal_raw) and dorsal_raw != '':
                try:
                    dorsal_text = str(int(float(dorsal_raw)))
                except (ValueError, TypeError):
                    dorsal_text = str(dorsal_raw)
            else:
                dorsal_text = 'S/N'

            # Lógica para dividir nombres largos en dos líneas
            max_chars = 9  # Máximo de caracteres antes de dividir el nombre
            if len(player_name) > max_chars:
                words = player_name.split()
                if len(words) > 1:
                    # Si hay varias palabras, buscar el punto de división más equilibrado
                    mid_point = (len(player_name) // 2) - 2 # Aproximado
                    line1 = ""
                    line2 = ""
                    current_len = 0
                    for i, word in enumerate(words):
                        if current_len + len(word) < mid_point or i == 0:
                            line1 += word + " "
                            current_len += len(word) + 1
                        else:
                            line2 += word + " "
                    display_text = f"{dorsal_text}\n{line1.strip()}\n{line2.strip()}"
                else:
                    # Si es una sola palabra larga, la partimos por la mitad
                    mid = len(player_name) // 2
                    display_text = f"{dorsal_text}\n{player_name[:mid]}-\n{player_name[mid:]}"
            else:
                # Si el nombre es corto, se queda en una sola línea
                display_text = f"{dorsal_text}\n{player_name}"

            # Reposicionar texto y ajustar fuente/espaciado
            ax.text(x_pos, y_base - 0.3, display_text,
                    fontsize=6, color=texto_color, weight='bold',
                    ha='center', va='top', linespacing=0.9)
            # --- FIN: Lógica mejorada para nombres y dorsales ---
        
        # 🔗 LÍNEAS DISCONTINUAS CONECTANDO PUNTOS ROJOS
        if len(players_list) > 1:
            # Dibujar línea discontinua conectando todos los puntos rojos
            ax.plot(puntos_rojos_x, puntos_rojos_y, 
                    color='red', linewidth=2, alpha=0.6, linestyle='--')
    
    def create_global_legend(self, ax):
        """🔥 LEYENDA GLOBAL para todas las gráficas"""
        legend_x = 100  # Posición X en el campo
        legend_y = 78  # MÁS ARRIBA (era 75)

        # Fondo de la leyenda MÁS PEQUEÑO
        legend_bg = plt.Rectangle((legend_x - 0.5, legend_y - 2), 18, 4,  # Era (22, 6)
                                 facecolor='#2c3e50', alpha=0.95,
                                 edgecolor='white', linewidth=2)
        ax.add_patch(legend_bg)
        
        # Título de la leyenda
        ax.text(legend_x + 8.5, legend_y + 1, 'LEYENDA',
                fontsize=8, weight='bold', color='white',
                ha='center', va='center')
        
        # Colores y métricas de barras
        metric_labels = ['14-21 Km/h', '>21 Km/h', '>24 Km/h']
        
        for i, (label, color) in enumerate(zip(metric_labels, self.colores_barras)):
            rect_x = legend_x + (i * 5.5)
            rect_y = legend_y - 0.5
            
            # Rectángulo de color
            legend_rect = plt.Rectangle((rect_x, rect_y - 0.3), 1, 0.6,
                                      facecolor=color, alpha=0.8,
                                      edgecolor='white', linewidth=0.5)
            ax.add_patch(legend_rect)
            
            # Etiqueta
            ax.text(rect_x + 1.2, rect_y, label,
                    fontsize=5, color='white', weight='bold',
                    ha='left', va='center')
        
        # Punto rojo para Vmax
        circle_x = legend_x + 4
        circle_y = legend_y - 1.5
        circle_legend = plt.Circle((circle_x, circle_y), 0.25, color='red', alpha=0.9)
        ax.add_patch(circle_legend)
        
        # Línea discontinua ejemplo
        ax.plot([circle_x, circle_x], [circle_y - 0.4, circle_y - 0.8], 
                color='red', linewidth=2, linestyle='--', alpha=0.8)
        
        ax.text(circle_x + 1, circle_y, 'Velocidad Máxima',
                fontsize=5, color='white', weight='bold',
                ha='left', va='center')
    
    def create_team_summary_table(self, team_data, ax, x_pos, y_pos, team_name, team_colors, team_logo=None):
        """Crea una tabla de resumen del equipo con el logo del equipo como fondo."""
        
        # Calcular estadísticas del equipo
        summary_stats = {}
        for metric in self.metricas_resumen_equipos:
            if metric in team_data.columns:
                summary_stats[metric] = team_data[metric].max() if 'Velocidad Máxima' in metric else team_data[metric].mean()
        
        # Dimensiones de la tabla
        num_metrics = len(summary_stats)
        metric_col_width = 8
        table_width = num_metrics * metric_col_width
        row_height = 1.5
        table_height = row_height * 2
        
        # 1. DIBUJAR EL RECTÁNGULO DE FONDO (será la máscara)
        main_rect = mpatches.Rectangle((x_pos - table_width/2, y_pos - table_height/2), 
                                       table_width, table_height,
                                       facecolor='#2c3e50', alpha=0.85, # Alpha ligeramente ajustado
                                       edgecolor='white', linewidth=2, zorder=1)
        ax.add_patch(main_rect)
        
        # 2. AÑADIR EL LOGO COMO FONDO DE AGUA
        if team_logo is not None:
            logo_extent = [x_pos - table_width/2, x_pos + table_width/2, 
                           y_pos - table_height/2, y_pos + table_height/2]
            img = ax.imshow(team_logo, extent=logo_extent, aspect='auto', alpha=0.1, zorder=0)
            img.set_clip_path(main_rect)

        # Borde superior decorativo
        top_rect = plt.Rectangle((x_pos - table_width/2, y_pos + table_height/2 - 0.3), 
                                table_width, 0.3,
                                facecolor=team_colors['primary'], alpha=0.9,
                                edgecolor='none', zorder=2)
        ax.add_patch(top_rect)
        
        # FILA 1: NOMBRES DE MÉTRICAS
        metrics_y = y_pos + row_height/2
        for i, (metric, value) in enumerate(summary_stats.items()):
            metric_x = x_pos - table_width/2 + (i * metric_col_width) + metric_col_width/2
            metric_rect = plt.Rectangle((metric_x - metric_col_width/2, metrics_y - row_height/2), 
                                    metric_col_width, row_height,
                                    facecolor=team_colors['primary'], alpha=0.6, 
                                    edgecolor='white', linewidth=0.5, zorder=2)
            ax.add_patch(metric_rect)
            
            metric_short = metric.replace('Distancia Total ', '')
            if '14-21' in metric: metric_short = '14-21 Km/h'
            elif '>21' in metric and '>24' not in metric: metric_short = '>21 Km/h'
            elif '>24' in metric: metric_short = '>24 Km/h'
            elif 'Velocidad Máxima' in metric: metric_short = 'Vmax'
                
            ax.text(metric_x, metrics_y, metric_short, 
                    fontsize=6, weight='bold', color='white',
                    ha='center', va='center', zorder=3)
        
        # FILA 2: VALORES DE MÉTRICAS
        values_y = y_pos - row_height/2
        for i, (metric, value) in enumerate(summary_stats.items()):
            metric_x = x_pos - table_width/2 + (i * metric_col_width) + metric_col_width/2
            if i % 2 == 0:
                value_rect = plt.Rectangle((metric_x - metric_col_width/2, values_y - row_height/2), 
                                        metric_col_width, row_height,
                                        facecolor='#3c566e', alpha=0.3, 
                                        edgecolor='none', zorder=2)
                ax.add_patch(value_rect)
            
            formatted_value = f"{value:.1f}" if 'Velocidad' in metric else f"{value:.0f}"
            ax.text(metric_x, values_y, formatted_value, 
                    fontsize=9, weight='bold', color='#FFD700',
                    ha='center', va='center', zorder=3)
        
        # LÍNEAS SEPARADORAS
        ax.plot([x_pos - table_width/2, x_pos + table_width/2], 
                [y_pos, y_pos], 
                color='white', linewidth=1.5, alpha=0.8, zorder=3)
        for i in range(1, num_metrics):
            line_x = x_pos - table_width/2 + (i * metric_col_width)
            ax.plot([line_x, line_x], 
                    [y_pos - table_height/2, y_pos + table_height/2], 
                    color='white', linewidth=0.5, alpha=0.6, zorder=3)
    
    def get_position_for_demarcation(self, demarcacion_display, team_side):
        """Obtiene la posición correcta para una demarcación específica basándose en el mapeo original"""
        
        # 🔥 MAPEO ESPECIAL PARA DEMARCACIONES COMBINADAS
        demarcacion_to_position_map = {
            'MC Box to Box': 'MC_BOX_TO_BOX',
            'Delantero Centro': 'DELANTERO_CENTRO',
            'Portero': 'PORTERO',
            'Central Derecho': 'CENTRAL_DERECHO',
            'Central Izquierdo': 'CENTRAL_IZQUIERDO', 
            'Lateral Derecho': 'LATERAL_DERECHO',
            'Lateral Izquierdo': 'LATERAL_IZQUIERDO',
            'MC Organizador': 'MC_ORGANIZADOR',
            'MC Posicional': 'MC_POSICIONAL',
            'Banda Derecha': 'BANDA_DERECHA',
            'Banda Izquierda': 'BANDA_IZQUIERDA',
        }
        
        position = demarcacion_to_position_map.get(demarcacion_display, 'MC_BOX_TO_BOX')
        
        if position in self.coordenadas_graficos[team_side]:
            return self.coordenadas_graficos[team_side][position]
        else:
            return self.coordenadas_graficos[team_side]['MC_BOX_TO_BOX']
    
    def create_visualization(self, equipo_rival, jornadas, figsize=(11.69, 8.27)):
        """Crea la visualización completa con gráficos de barras por demarcación"""
        
        fig, ax = self.create_campo_sin_espacios(figsize)
        
        ax.text(60, 78, f'DATOS PROMEDIO - ÚLTIMAS {len(jornadas)} JORNADAS | MÍNIMO 60+ MIN', 
                fontsize=14, weight='bold', color='white', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.8", facecolor='#1e3d59', alpha=0.95,
                         edgecolor='white', linewidth=2))
        
        villarreal_data = self.filter_and_accumulate_data('Villarreal CF', jornadas, min_avg_minutes=60)
        rival_data = self.filter_and_accumulate_data(equipo_rival, jornadas, min_avg_minutes=60)
        
        if villarreal_data is None or len(villarreal_data) == 0:
            print("❌ No hay jugadores de Villarreal CF con promedio 60+ minutos")
            return None
            
        if rival_data is None or len(rival_data) == 0:
            print(f"❌ No hay jugadores de {equipo_rival} con promedio 60+ minutos")
            return None
        
        villarreal_logo = self.load_team_logo('Villarreal CF')
        rival_logo = self.load_team_logo(equipo_rival)
        
        if villarreal_logo is not None:
            imagebox = OffsetImage(villarreal_logo, zoom=0.45)
            ab = AnnotationBbox(imagebox, (5, 5), frameon=False)
            ax.add_artist(ab)
        
        if rival_logo is not None:
            imagebox = OffsetImage(rival_logo, zoom=0.45)
            ab = AnnotationBbox(imagebox, (115, 5), frameon=False)
            ax.add_artist(ab)
        
        self.create_global_legend(ax)
        
        villarreal_by_position = self.group_players_by_final_position(villarreal_data)
        rival_by_position = self.group_players_by_final_position(rival_data)

        villarreal_by_position = self.redistribute_overcrowded_positions(villarreal_by_position, max_players_per_position=3)
        rival_by_position = self.redistribute_overcrowded_positions(rival_by_position, max_players_per_position=3)

        villarreal_colors = self.get_team_colors('Villarreal CF')
        rival_colors = self.get_team_colors(equipo_rival)

        for position, players in villarreal_by_position.items():
            if players and position in self.coordenadas_graficos['villarreal']:
                x, y = self.coordenadas_graficos['villarreal'][position]
                self.create_position_graph(players, position, x, y, ax, 
                                         villarreal_colors, villarreal_logo)

        for position, players in rival_by_position.items():
            if players and position in self.coordenadas_graficos['rival']:
                x, y = self.coordenadas_graficos['rival'][position]
                self.create_position_graph(players, position, x, y, ax, 
                                     rival_colors, rival_logo)
        
        self.create_team_summary_table(villarreal_data, ax, 30, 2, 'Villarreal CF', 
                             villarreal_colors, villarreal_logo)
        self.create_team_summary_table(rival_data, ax, 90, 2, equipo_rival, 
                             rival_colors, rival_logo)
        
        return fig
    
    def create_visualization(self, equipo_rival, jornadas, figsize=(11.69, 8.27)):
        """Crea la visualización completa con gráficos de barras por demarcación"""
        
        # Crear campo SIN espacios
        fig, ax = self.create_campo_sin_espacios(figsize)
        
        # Título superpuesto en el campo
        ax.text(60, 78, f'DATOS PROMEDIO - ÚLTIMAS {len(jornadas)} JORNADAS | MÍNIMO 60+ MIN', 
                fontsize=14, weight='bold', color='white', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.8", facecolor='#1e3d59', alpha=0.95,
                         edgecolor='white', linewidth=2))
        
        # Obtener datos acumulados de ambos equipos
        villarreal_data = self.filter_and_accumulate_data('Villarreal CF', jornadas, min_avg_minutes=60)
        rival_data = self.filter_and_accumulate_data(equipo_rival, jornadas, min_avg_minutes=60)
        
        if villarreal_data is None or len(villarreal_data) == 0:
            print("❌ No hay jugadores de Villarreal CF con promedio 60+ minutos")
            return None
            
        if rival_data is None or len(rival_data) == 0:
            print(f"❌ No hay jugadores de {equipo_rival} con promedio 60+ minutos")
            return None
        
        # Cargar escudos
        villarreal_logo = self.load_team_logo('Villarreal CF')
        rival_logo = self.load_team_logo(equipo_rival)
        
        # Posicionar escudos dentro del campo
        if villarreal_logo is not None:
            imagebox = OffsetImage(villarreal_logo, zoom=0.45)
            ab = AnnotationBbox(imagebox, (5, 5), frameon=False)
            ax.add_artist(ab)
        
        if rival_logo is not None:
            imagebox = OffsetImage(rival_logo, zoom=0.45)
            ab = AnnotationBbox(imagebox, (115, 5), frameon=False)
            ax.add_artist(ab)
        
        # 🔥 LEYENDA GLOBAL
        self.create_global_legend(ax)
        
        # 🔥 AGRUPAR POR DEMARCACIÓN REAL (no por posición mappeada)
        villarreal_by_position = self.group_players_by_final_position(villarreal_data)
        rival_by_position = self.group_players_by_final_position(rival_data)

        # 🔥 NUEVO: Redistribuir jugadores sobrepoblados
        villarreal_by_position = self.redistribute_overcrowded_positions(villarreal_by_position, max_players_per_position=3)
        rival_by_position = self.redistribute_overcrowded_positions(rival_by_position, max_players_per_position=3)

        # Obtener colores para cada equipo
        villarreal_colors = self.get_team_colors('Villarreal CF')
        rival_colors = self.get_team_colors(equipo_rival)

        # 🔥 CREAR GRÁFICOS EN LAS POSICIONES CORRECTAS DEL SCRIPT ORIGINAL
        for position, players in villarreal_by_position.items():
            if players and position in self.coordenadas_graficos['villarreal']:
                x, y = self.coordenadas_graficos['villarreal'][position]
                self.create_position_graph(players, position, x, y, ax, 
                                         villarreal_colors, villarreal_logo)

        for position, players in rival_by_position.items():
            if players and position in self.coordenadas_graficos['rival']:
                x, y = self.coordenadas_graficos['rival'][position]
                self.create_position_graph(players, position, x, y, ax, 
                                     rival_colors, rival_logo)
        
        # Resúmenes de equipos con métricas de barras
        self.create_team_summary_table(villarreal_data, ax, 30, 2, 'Villarreal CF', 
                             villarreal_colors, villarreal_logo)
        self.create_team_summary_table(rival_data, ax, 90, 2, equipo_rival, 
                             rival_colors, rival_logo)
        
        return fig
    
    def guardar_sin_espacios(self, fig, filename):
        """Guarda el archivo sin ningún espacio en blanco"""
        fig.savefig(
            filename,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0,
            facecolor='white',
            edgecolor='none',
            format='pdf' if filename.endswith('.pdf') else 'png',
            transparent=False
        )

def seleccionar_equipo_jornadas_barras():
    """Permite al usuario seleccionar un equipo rival y jornadas"""
    try:
        report_generator = CampoFutbolBarras()
        equipos = report_generator.get_available_teams()
        
        # Filtrar Villarreal CF de la lista de oponentes
        equipos_rival = [eq for eq in equipos if 'Villarreal' not in eq]
        
        if len(equipos_rival) == 0:
            print("❌ No se encontraron equipos rivales en los datos.")
            return None, None
        
        for i, equipo in enumerate(equipos_rival, 1):
            pass
        
        while True:
            try:
                seleccion = input(f"\nSelecciona un equipo rival (1-{len(equipos_rival)}): ").strip()
                indice = int(seleccion) - 1
                
                if 0 <= indice < len(equipos_rival):
                    equipo_seleccionado = equipos_rival[indice]
                    break
                else:
                    print(f"❌ Por favor, ingresa un número entre 1 y {len(equipos_rival)}")
            except ValueError:
                print("❌ Por favor, ingresa un número válido")
        
        # Obtener jornadas disponibles
        jornadas_disponibles = report_generator.get_available_jornadas()
        
        # Preguntar cuántas jornadas incluir
        while True:
            try:
                num_jornadas = input(f"¿Cuántas jornadas incluir? (máximo {len(jornadas_disponibles)}): ").strip()
                num_jornadas = int(num_jornadas)
                
                if 1 <= num_jornadas <= len(jornadas_disponibles):
                    jornadas_seleccionadas = sorted(jornadas_disponibles)[-num_jornadas:]
                    break
                else:
                    print(f"❌ Por favor, ingresa un número entre 1 y {len(jornadas_disponibles)}")
            except ValueError:
                print("❌ Por favor, ingresa un número válido")
        
        return equipo_seleccionado, jornadas_seleccionadas
        
    except Exception as e:
        print(f"❌ Error en la selección: {e}")
        return None, None

def main_campo_futbol_barras():
    """Función principal para generar el informe con gráficos de barras por demarcación"""
    try:
        pass
        
        # Selección interactiva
        equipo_rival, jornadas = seleccionar_equipo_jornadas_barras()
        
        if equipo_rival is None or jornadas is None:
            print("❌ No se pudo completar la selección.")
            return
        
        
        # Crear el reporte
        report_generator = CampoFutbolBarras()
        fig = report_generator.create_visualization(equipo_rival, jornadas)
        
        if fig:
            # Mostrar en pantalla
            plt.show()
            
            # Guardar
            equipo_filename = equipo_rival.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_BARRAS_Villarreal_vs_{equipo_filename}.pdf"
            
            report_generator.guardar_sin_espacios(fig, output_path)
            
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()

def generar_reporte_barras_personalizado(equipo_rival, jornadas, mostrar=True, guardar=True):
    """Función para generar un reporte personalizado con gráficos de barras"""
    try:
        report_generator = CampoFutbolBarras()
        fig = report_generator.create_visualization(equipo_rival, jornadas)
        
        if fig:
            if mostrar:
                plt.show()
            
            if guardar:
                equipo_filename = equipo_rival.replace(' ', '_').replace('/', '_')
                output_path = f"reporte_BARRAS_Villarreal_vs_{equipo_filename}.pdf"
                report_generator.guardar_sin_espacios(fig, output_path)
            
            return fig
        else:
            print("❌ No se pudo generar la visualización")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Inicialización
try:
    report_generator = CampoFutbolBarras()
    equipos = report_generator.get_available_teams()
    
    if len(equipos) > 0:
        pass
    
except Exception as e:
    print(f"❌ Error al inicializar: {e}")

if __name__ == "__main__":
    main_campo_futbol_barras()