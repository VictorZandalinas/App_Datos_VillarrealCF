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

class CampoFutbolMaximos:
    def __init__(self, data_path="extraccion_mediacoach/data/rendimiento_fisico.parquet"):
        """
        Inicializa la clase para generar informes con DATOS MÁXIMOS
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
        self.coordenadas_tablas = {
            # Villarreal (lado izquierdo)
            'villarreal': {
                'PORTERO': (10, 40),              # Portería
                'LATERAL_DERECHO': (25, 12),      # Lateral derecho (arriba)
                'CENTRAL_DERECHO': (20, 25),      # Central derecho (centro-arriba)
                'CENTRAL_IZQUIERDO': (20, 53),    # Central izquierdo (centro-abajo)
                'LATERAL_IZQUIERDO': (25, 68),    # Lateral izquierdo (abajo)
                'MC_POSICIONAL': (35, 40),        # Mediocampo defensivo (centro)
                'MC_BOX_TO_BOX': (62, 55),        # Box to box (centro-arriba)
                'MC_ORGANIZADOR': (47, 40),       # Organizador (centro-abajo)
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
                'MC_ORGANIZADOR': (72, 40),       # Organizador (centro-arriba - espejo)
                'BANDA_DERECHA': (45, 68),        # Banda derecha (extremo abajo - espejo)
                'BANDA_IZQUIERDA': (45, 12),      # Banda izquierda (extremo arriba - espejo)
                'DELANTERO_CENTRO': (38, 25),     # Delantero centro (abajo - espejo)
                'SEGUNDO_DELANTERO': (38, 53),    # Segundo delantero (arriba - espejo)
            }
        }
        
        # Métricas principales para mostrar en las tablas
        self.metricas_principales = [
            'Minutos CON posesión',
            'Minutos SIN posesión',
            'Distancia CON posesión',
            'Distancia SIN posesión',
            'Distancia >21 km / h CON posesión',
            'Distancia >21 km / h SIN posesión',
            'Distancia >24 km / h CON posesión',
            'Distancia >24 km / h SIN posesión'
        ]

    def get_contrasting_color(self, background_color):
        """Devuelve blanco o negro según el color de fondo para mejor contraste"""
        # Si el color primario es claro, usar texto oscuro
        light_colors = ['#FFFFFF', '#FFD700', '#FFFF00', '#87CEEB']  # Blanco, dorado, amarillo, celeste
        
        if background_color in light_colors:
            return '#2c3e50'  # Azul oscuro
        else:
            return 'white'

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
        """Compara dos nombres de equipo de forma inteligente"""
        if not team1 or not team2:
            return False

        norm1 = self.normalize_text(team1)
        norm2 = self.normalize_text(team2)

        words1 = set(norm1.split())
        words2 = set(norm2.split())

        if words1.issubset(words2) or words2.issubset(words1):
            return True

        return False
            
    def map_opta_position_to_system(self, position, position_side, team_name=None, week=None):
        """Mapea posición Opta al sistema interno"""
        if pd.isna(position):
            return None
        
        position = str(position).strip()
        position_side = str(position_side).strip() if pd.notna(position_side) else ""
        
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

        elif position == "Midfielder":  # ✅ UN SOLO BLOQUE PARA MIDFIELDER
            # 1. Comprobar primero los casos específicos combinados
            if "Centre/Right" in position_side:
                return "MC_ORGANIZADOR"
            elif "Left/Centre" in position_side:
                return "MC_ORGANIZADOR"
            
            # 2. Ahora, comprobar los casos más generales (bandas)
            elif "Right" in position_side:
                return "BANDA_DERECHA"
            elif "Left" in position_side:
                return "BANDA_IZQUIERDA"

            # 3. Finalmente, el caso puramente central como fallback
            elif "Centre" in position_side:
                return "MC_ORGANIZADOR"
            
            # 4. Si no hay información de lado, se asume central
            else:
                return "MC_ORGANIZADOR"
        
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
            return None
        
        opta_week = self.convert_jornada_to_week(jornada)
        if opta_week is None:
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
                return None
                
            opta_position = self.map_opta_position_to_system(
                opta_position_raw, 
                opta_position_side_raw,
                team_name,
                opta_week
            )

            if opta_position:
                return opta_position
            else:
                return None
        
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
    
    def get_player_position_history(self, jugador_id):
        """Obtiene el historial de posiciones de un jugador"""
        if self.df is None:
            return []
        
        player_positions = self.df[
            (self.df['Id Jugador'] == jugador_id) & 
            (self.df['Demarcacion'].notna()) & 
            (self.df['Demarcacion'] != '') &
            (self.df['Demarcacion'].str.strip() != '')
        ]['Demarcacion'].tolist()
        
        return player_positions
    
    def has_played_position(self, jugador_id, demarcacion):
        """Verifica si un jugador ha jugado en una demarcación específica"""
        history = self.get_player_position_history(jugador_id)
        return demarcacion in history
    
    def filter_and_get_maximum_data(self, equipo, jornadas, min_minutes=70):
        """🔥 NUEVA FUNCIÓN: Filtra por minutos mínimos y obtiene DATOS MÁXIMOS por jugador"""
        if self.df is None:
            return None
        
        # Normalizar jornadas
        normalized_jornadas = []
        for jornada in jornadas:
            if isinstance(jornada, str) and jornada.startswith('J'):
                try:
                    normalized_jornadas.append(int(jornada[1:]))
                except ValueError:
                    normalized_jornadas.append(jornada)
            elif isinstance(jornada, str) and jornada.startswith('j'):
                try:
                    normalized_jornadas.append(int(jornada[1:]))
                except ValueError:
                    normalized_jornadas.append(jornada)
            else:
                normalized_jornadas.append(jornada)
        
        # Filtrar por equipo y jornadas
        filtered_df = self.df[
            (self.df['Equipo'] == equipo) & 
            (self.df['Jornada'].isin(normalized_jornadas))
        ].copy()
        
        # Rellenar demarcaciones vacías
        filtered_df = self.fill_missing_demarcaciones(filtered_df)
        
        # Verificar si Alias está vacío y usar Nombre en su lugar
        if 'Nombre' in filtered_df.columns:
            mask_empty_alias = filtered_df['Alias'].isna() | (filtered_df['Alias'] == '') | (filtered_df['Alias'].str.strip() == '')
            filtered_df.loc[mask_empty_alias, 'Alias'] = filtered_df.loc[mask_empty_alias, 'Nombre']

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

        if 'Minutos jugados' not in filtered_df.columns:
            print("⚠️  Columna 'Minutos jugados' no encontrada.")
            return None
        
        # 🔥 NUEVA LÓGICA: OBTENER DATOS MÁXIMOS por jugador
        
        maximum_data = []
        
        for jugador in filtered_df['Alias'].unique():
            jugador_data = filtered_df[filtered_df['Alias'] == jugador]
    
            # NUEVO: Filtrar solo partidos donde jugó 70+ minutos
            jugador_data_filtered = jugador_data[jugador_data['Minutos jugados'] >= min_minutes]
    
            # Solo incluir jugadores que tengan al menos 1 partido con 70+ minutos
            if len(jugador_data_filtered) > 0:
                # Tomar datos básicos del jugador (usar el registro más reciente)
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

                # 🔥 CREAR REGISTRO CON VALORES MÁXIMOS
                maximum_record = {
                    'Id Jugador': latest_record['Id Jugador'],
                    'Dorsal': latest_record['Dorsal'],
                    'Nombre': latest_record['Nombre'],
                    'Alias': latest_record['Alias'],
                    'Final_Position': final_position,
                    'Position_Source': position_source,
                    'Demarcacion': jugador_data_filtered['Demarcacion'].mode().iloc[0] if len(jugador_data_filtered['Demarcacion'].mode()) > 0 else latest_record['Demarcacion'],
                    'Equipo': latest_record['Equipo'],
                    
                    # Minutos: MÁXIMO jugado SOLO en partidos de 70+
                    'Minutos jugados': jugador_data_filtered['Minutos jugados'].max(),
                }


                # 🔥 CALCULAR MÉTRICAS COMBINADAS CORRECTAMENTE
                # Para cada métrica, sumar 1P + 2P por jornada, luego tomar el máximo

                # Minutos CON y SIN posesión
                maximum_record['Minutos CON posesión'] = pd.to_numeric(
                    jugador_data.get('Minutos CON posesión', pd.Series([0])), errors='coerce'
                ).fillna(0).max()

                maximum_record['Minutos SIN posesión'] = pd.to_numeric(
                    jugador_data.get('Minutos SIN posesión', pd.Series([0])), errors='coerce'
                ).fillna(0).max()

                # Distancia CON posesión = suma por jornada, luego máximo
                dist_con_1p = pd.to_numeric(jugador_data.get('Distancia CON posesión 1P', pd.Series([0])), errors='coerce').fillna(0)
                dist_con_2p = pd.to_numeric(jugador_data.get('Distancia CON posesión 2P', pd.Series([0])), errors='coerce').fillna(0)
                maximum_record['Distancia CON posesión'] = (dist_con_1p + dist_con_2p).max()

                # Distancia SIN posesión = suma por jornada, luego máximo
                dist_sin_1p = pd.to_numeric(jugador_data.get('Distancia SIN posesión 1P', pd.Series([0])), errors='coerce').fillna(0)
                dist_sin_2p = pd.to_numeric(jugador_data.get('Distancia SIN posesión 2P', pd.Series([0])), errors='coerce').fillna(0)
                maximum_record['Distancia SIN posesión'] = (dist_sin_1p + dist_sin_2p).max()

                # Distancia >21 km/h CON posesión = suma por jornada, luego máximo
                dist21_con_1p = pd.to_numeric(jugador_data.get('Distancia >21 km / h CON posesión 1P', pd.Series([0])), errors='coerce').fillna(0)
                dist21_con_2p = pd.to_numeric(jugador_data.get('Distancia >21 km / h CON posesión 2P', pd.Series([0])), errors='coerce').fillna(0)
                maximum_record['Distancia >21 km / h CON posesión'] = (dist21_con_1p + dist21_con_2p).max()

                # Distancia >21 km/h SIN posesión = suma por jornada, luego máximo
                dist21_sin_1p = pd.to_numeric(jugador_data.get('Distancia >21 km / h SIN posesión 1P', pd.Series([0])), errors='coerce').fillna(0)
                dist21_sin_2p = pd.to_numeric(jugador_data.get('Distancia >21 km / h SIN posesión 2P', pd.Series([0])), errors='coerce').fillna(0)
                maximum_record['Distancia >21 km / h SIN posesión'] = (dist21_sin_1p + dist21_sin_2p).max()

                # Distancia >24 km/h CON posesión = suma por jornada, luego máximo
                dist24_con_1p = pd.to_numeric(jugador_data.get('Distancia >24 km / h CON posesión 1P', pd.Series([0])), errors='coerce').fillna(0)
                dist24_con_2p = pd.to_numeric(jugador_data.get('Distancia >24 km / h CON posesión 2P', pd.Series([0])), errors='coerce').fillna(0)
                maximum_record['Distancia >24 km / h CON posesión'] = (dist24_con_1p + dist24_con_2p).max()

                # Distancia >24 km/h SIN posesión = suma por jornada, luego máximo
                dist24_sin_1p = pd.to_numeric(jugador_data.get('Distancia >24 km / h SIN posesión 1P', pd.Series([0])), errors='coerce').fillna(0)
                dist24_sin_2p = pd.to_numeric(jugador_data.get('Distancia >24 km / h SIN posesión 2P', pd.Series([0])), errors='coerce').fillna(0)
                maximum_record['Distancia >24 km / h SIN posesión'] = (dist24_sin_1p + dist24_sin_2p).max()

                maximum_data.append(maximum_record)
        
        # Calcular métricas combinadas CON y SIN posesión - MANEJANDO NaN
        def safe_max_sum(series1, series2):
            """Suma dos series manejando NaN y valores vacíos"""
            s1 = pd.to_numeric(series1, errors='coerce').fillna(0)
            s2 = pd.to_numeric(series2, errors='coerce').fillna(0)
            result = s1.max() + s2.max()
            return result if not pd.isna(result) else 0

        maximum_record['Distancia CON posesión'] = safe_max_sum(
            jugador_data.get('Distancia CON posesión 1P', pd.Series([0])),
            jugador_data.get('Distancia CON posesión 2P', pd.Series([0]))
        )

        maximum_record['Distancia SIN posesión'] = safe_max_sum(
            jugador_data.get('Distancia SIN posesión 1P', pd.Series([0])),
            jugador_data.get('Distancia SIN posesión 2P', pd.Series([0]))
        )

        maximum_record['Distancia >21 km / h CON posesión'] = safe_max_sum(
            jugador_data.get('Distancia >21 km / h CON posesión 1P', pd.Series([0])),
            jugador_data.get('Distancia >21 km / h CON posesión 2P', pd.Series([0]))
        )

        maximum_record['Distancia >21 km / h SIN posesión'] = safe_max_sum(
            jugador_data.get('Distancia >21 km / h SIN posesión 1P', pd.Series([0])),
            jugador_data.get('Distancia >21 km / h SIN posesión 2P', pd.Series([0]))
        )

        maximum_record['Distancia >24 km / h CON posesión'] = safe_max_sum(
            jugador_data.get('Distancia >24 km / h CON posesión 1P', pd.Series([0])),
            jugador_data.get('Distancia >24 km / h CON posesión 2P', pd.Series([0]))
        )

        maximum_record['Distancia >24 km / h SIN posesión'] = safe_max_sum(
            jugador_data.get('Distancia >24 km / h SIN posesión 1P', pd.Series([0])),
            jugador_data.get('Distancia >24 km / h SIN posesión 2P', pd.Series([0]))
        )

        # También agregar las métricas básicas manejando NaN
        maximum_record['Minutos CON posesión'] = pd.to_numeric(
            jugador_data.get('Minutos CON posesión', pd.Series([0])), errors='coerce'
        ).fillna(0).max()

        maximum_record['Minutos SIN posesión'] = pd.to_numeric(
            jugador_data.get('Minutos SIN posesión', pd.Series([0])), errors='coerce'
        ).fillna(0).max()

        # Convertir a DataFrame
        if maximum_data:
            result_df = pd.DataFrame(maximum_data)
            return result_df
        else:
            print(f"❌ No hay jugadores con al menos {min_minutes} minutos en una jornada para {equipo}")
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
    
    
    def group_players_by_specific_position(self, filtered_df):
        """Agrupa jugadores por posiciones específicas con lógica mejorada"""
        # Verificar si Alias está vacío y usar Nombre en su lugar
        if 'Nombre' in filtered_df.columns:
            mask_empty_alias = filtered_df['Alias'].isna() | (filtered_df['Alias'] == '') | (filtered_df['Alias'].str.strip() == '')
            filtered_df.loc[mask_empty_alias, 'Alias'] = filtered_df.loc[mask_empty_alias, 'Nombre']
        
        # Ordenar jugadores por minutos jugados (descendente) - EN DATOS MÁXIMOS es el máximo de minutos
        filtered_df_sorted = filtered_df.sort_values('Minutos jugados', ascending=False)
        
        grouped_players = {
            'PORTERO': [],
            'LATERAL_DERECHO': [],
            'CENTRAL_DERECHO': [],
            'CENTRAL_IZQUIERDO': [],
            'LATERAL_IZQUIERDO': [],
            'MC_POSICIONAL': [],
            'MC_BOX_TO_BOX': [],
            'MC_ORGANIZADOR': [],
            'BANDA_DERECHA': [],
            'BANDA_IZQUIERDA': [],
            'DELANTERO_CENTRO': [],
        }
        
        for _, player in filtered_df_sorted.iterrows():
            demarcacion = player.get('Demarcacion', 'Centrocampista - MC Box to Box')
            position = self.demarcacion_to_position.get(demarcacion, 'MC_BOX_TO_BOX')
            
            # Convertir Series a dict para facilitar el acceso
            player_dict = player.to_dict()
            
            # Agrupar por posiciones específicas
            if position in grouped_players:
                grouped_players[position].append(player_dict)
        
        
        # 🔥 LÓGICA DE DELANTEROS: Dividir delanteros cuando hay más de 1 columna
        delanteros = grouped_players['DELANTERO_CENTRO']

        if len(delanteros) > 1:  # Si hay más de 1 delantero
            # Dividir en dos grupos
            mitad = len(delanteros) // 2 + len(delanteros) % 2  # Redondear hacia arriba
            
            # Primer grupo se queda en DELANTERO_CENTRO
            primer_grupo = delanteros[:mitad]
            # Segundo grupo va a SEGUNDO_DELANTERO
            segundo_grupo = delanteros[mitad:]
            
            # Actualizar los grupos
            grouped_players['DELANTERO_CENTRO'] = primer_grupo
            grouped_players['SEGUNDO_DELANTERO'] = segundo_grupo
            
        else:
            # Si solo hay 1 delantero, crear grupo vacío para segundo delantero
            grouped_players['SEGUNDO_DELANTERO'] = []

        # Limitar jugadores por posición (máximo 3 por posición para evitar tablas muy anchas)
        for posicion in grouped_players:
            grouped_players[posicion] = grouped_players[posicion][:3]
        
        return grouped_players
    
    def group_players_by_final_position(self, filtered_df):
        """Agrupa jugadores por su posición final ya determinada (Opta o fallback)."""
        if filtered_df is None or 'Final_Position' not in filtered_df.columns:
            return self.group_players_by_specific_position(filtered_df)  # Fallback
            
        
        # Ordenar por minutos jugados (máximos)
        filtered_df_sorted = filtered_df.sort_values('Minutos jugados', ascending=False)
        
        grouped_players = {}
        
        # Agrupar por la nueva columna 'Final_Position'
        for position, group in filtered_df_sorted.groupby('Final_Position'):
            grouped_players[position] = group.to_dict('records')
            
        return grouped_players

    def redistribute_and_split_players(self, grouped_players):
        """Balancea centrales y divide delanteros para una distribución visual equitativa."""
        import math

        # PARTE 1: BALANCEO DE CENTRALES (si es necesario)
        centrales_d = grouped_players.get('CENTRAL_DERECHO', [])
        centrales_i = grouped_players.get('CENTRAL_IZQUIERDO', [])
        
        if len(centrales_d) > 2 and len(centrales_i) == 0:
            jugador_a_mover = centrales_d.pop()
            grouped_players['CENTRAL_IZQUIERDO'] = [jugador_a_mover]
        
        if len(centrales_i) > 2 and len(centrales_d) == 0:
            jugador_a_mover = centrales_i.pop()
            grouped_players['CENTRAL_DERECHO'] = [jugador_a_mover]

        # PARTE 2: DIVISIÓN EQUITATIVA DE DELANTEROS
        if 'DELANTERO_CENTRO' in grouped_players:
            delanteros = grouped_players.get('DELANTERO_CENTRO', [])
            
            if len(delanteros) > 1:
                # Calcula el punto de división. math.ceil redondea hacia arriba.
                # Para 4 jugadores, mitad = 2. Para 3, mitad = 2.
                mitad = math.ceil(len(delanteros) / 2)
                
                primer_grupo = delanteros[:mitad]
                segundo_grupo = delanteros[mitad:]
                
                # Reasignar los grupos
                grouped_players['DELANTERO_CENTRO'] = primer_grupo
                grouped_players['SEGUNDO_DELANTERO'] = segundo_grupo # Crea o sobrescribe la posición
                

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
    

    def create_position_table(self, players_list, x, y, ax, team_colors, position_name, team_logo=None):
        """Crea una tabla moderna con demarcación, nombres+dorsales y métricas en filas"""
        if not players_list:
            return
        
        num_players = len(players_list)
        num_metrics = len(self.metricas_principales)
        
        # Dimensiones de la tabla (REDUCIDAS)
        metric_col_width = 6   # Ancho de la columna de métricas
        player_col_width = 4   # Ancho por columna de jugador
        table_width = metric_col_width + (num_players * player_col_width)
        
        header_height = 1.4      # Altura del header de demarcación
        names_height = 2.2       # Altura de la fila de nombres
        metric_row_height = 1.0  # Altura por fila de métrica
        table_height = header_height + names_height + (num_metrics * metric_row_height)
        
        # 🎨 NUEVO FONDO MODERNO - Gradiente simulado con múltiples rectángulos (MÁS COMPACTO)
        # Fondo principal con bordes redondeados simulados
        main_rect = plt.Rectangle((x - table_width/2, y - table_height/2), 
                                table_width, table_height,
                                facecolor='#2c3e50', alpha=0.95, 
                                edgecolor='white', linewidth=2)  # Reducido de 3 a 2
        ax.add_patch(main_rect)
        
        # Efecto de borde superior más claro (MÁS FINO)
        top_rect = plt.Rectangle((x - table_width/2, y + table_height/2 - 0.5), 
                                table_width, 0.5,  # Reducido de 1 a 0.5
                                facecolor=team_colors['primary'], alpha=0.8,
                                edgecolor='none')
        ax.add_patch(top_rect)
        
        # 📍 FILA 1: DEMARCACIÓN CON ESCUDO
        # Verificar si hay jugadores sin posición en esta tabla
        has_sin_posicion = any(player.get('Demarcacion') == 'Sin Posición' for player in players_list)

        if has_sin_posicion and position_name.replace('_', ' ').upper() == 'MC POSICIONAL':
            clean_position_name = 'SIN POSICIÓN'
        else:
            clean_position_name = position_name.replace('_', ' ').replace('Mc ', 'MC ').replace('Delantero Centro', 'DEL. CENTRO').replace('Segundo Delantero', '2º DELANTERO')
        
        # Crear el rectángulo del header
        header_rect = plt.Rectangle((x - table_width/2, y + table_height/2 - header_height), 
                                table_width, header_height,
                                facecolor=team_colors['primary'], alpha=0.8,
                                edgecolor='white', linewidth=1)
        ax.add_patch(header_rect)

        # Añadir escudo si está disponible
        text_x = x

        # Texto de la demarcación
        ax.text(text_x, y + table_height/2 - header_height/2, clean_position_name, 
                fontsize=8, weight='bold', color=team_colors['text'],
                ha='center', va='center')
        
        # 📍 FILA 2: NOMBRES + DORSALES
        names_y = y + table_height/2 - header_height - names_height/2
        
        # Fondo especial para la fila de nombres (BORDE MÁS FINO)
        names_rect = plt.Rectangle((x - table_width/2 + metric_col_width, names_y - names_height/2), 
                                num_players * player_col_width, names_height,
                                facecolor='#34495e', alpha=0.7, 
                                edgecolor='white', linewidth=0.5)  # Reducido de 1 a 0.5
        ax.add_patch(names_rect)
        
        # 🏆 AÑADIR ESCUDO EN LA COLUMNA DE MÉTRICAS, FILA DE NOMBRES
        if team_logo is not None:
            try:
                # Crear rectángulo para recortar el escudo (clip path)
                logo_rect = plt.Rectangle((x - table_width/2, names_y - names_height/2), 
                                        metric_col_width, names_height,
                                        facecolor='none', edgecolor='none')
                
                # Calcular posición central de la celda
                logo_x = x - table_width/2 + metric_col_width/2
                logo_y = names_y
                
                # Calcular el zoom para que llene la celda (ampliado)
                # Usar un zoom alto para que llene bien la celda
                zoom_factor = min(metric_col_width / 100, names_height / 100) * 3.6
                
                # Crear imagen del escudo
                imagebox = OffsetImage(team_logo, zoom=zoom_factor)
                ab = AnnotationBbox(imagebox, (logo_x, logo_y), 
                                frameon=False, 
                                boxcoords='data',
                                clip_on=True)
                
                # Añadir clip path para recortar el escudo a los límites de la celda
                ab.set_clip_path(logo_rect)
                ab.set_clip_on(True)
                
                ax.add_artist(ab)
                
            except Exception as e:
                print(f"⚠️  Error al añadir escudo en celda: {e}")

        # Agregar nombres y dorsales CON FONDOS INDIVIDUALES
        for i, player in enumerate(players_list):
            player_x = x - table_width/2 + metric_col_width + (i * player_col_width) + player_col_width/2
            player_name = player['Alias'] if pd.notna(player['Alias']) else 'N/A'
            
            # Formatear dorsal para quitar .0
            dorsal_raw = player.get('Dorsal', 'N/A')
            if pd.notna(dorsal_raw) and dorsal_raw != 'N/A':
                try:
                    dorsal = str(int(float(dorsal_raw)))  # Convertir a int para quitar decimales
                except (ValueError, TypeError):
                    dorsal = str(dorsal_raw)
            else:
                dorsal = 'S/N'
            
            # FONDO INDIVIDUAL para cada jugador con color del equipo
            player_bg = plt.Rectangle((player_x - player_col_width/2, names_y - names_height/2), 
                                    player_col_width, names_height,
                                    facecolor=team_colors['primary'], alpha=0.9,
                                    edgecolor='white', linewidth=1)
            ax.add_patch(player_bg)
            
            # Color de texto contrastante
            text_color = self.get_contrasting_color(team_colors['primary'])
            
            # Número del dorsal con color contrastante
            ax.text(player_x, names_y + 0.4, str(dorsal), 
                    fontsize=10, weight='bold', color=text_color,
                    ha='center', va='center')
            
            # 🔥 DIVIDIR NOMBRES LARGOS EN DOS LÍNEAS
            max_chars_per_line = 8  # Máximo caracteres por línea
            
            if len(player_name) > max_chars_per_line:
                # Dividir el nombre en dos líneas
                words = player_name.split()
                if len(words) >= 2:
                    # Si hay múltiples palabras, dividir por palabras
                    mid_point = len(words) // 2
                    first_line = " ".join(words[:mid_point])
                    second_line = " ".join(words[mid_point:])
                else:
                    # Si es una sola palabra larga, dividir por caracteres
                    mid_char = len(player_name) // 2
                    first_line = player_name[:mid_char]
                    second_line = player_name[mid_char:]
                
                # Mostrar en dos líneas
                ax.text(player_x, names_y - 0.3, first_line, 
                        fontsize=5, weight='bold', color=text_color,  # Fuente más pequeña
                        ha='center', va='center')
                ax.text(player_x, names_y - 0.9, second_line, 
                        fontsize=5, weight='bold', color=text_color,
                        ha='center', va='center')
            else:
                # Nombre normal en una línea
                ax.text(player_x, names_y - 0.6, player_name, 
                        fontsize=6, weight='bold', color=text_color,
                        ha='center', va='center')
        
        # 📍 FILAS 3+: MÉTRICAS Y VALORES
        for i, metric in enumerate(self.metricas_principales):
            metric_y = names_y - names_height/2 - (i + 1) * metric_row_height + metric_row_height/2
            
            # Fondo alternado para las filas de métricas
            if i % 2 == 0:
                row_rect = plt.Rectangle((x - table_width/2, metric_y - metric_row_height/2), 
                                    table_width, metric_row_height,
                                    facecolor='#3c566e', alpha=0.3, 
                                    edgecolor='none')
                ax.add_patch(row_rect)
            
            # Columna de métrica (nombre) con fondo destacado (BORDE MÁS FINO)
            metric_bg = plt.Rectangle((x - table_width/2, metric_y - metric_row_height/2), 
                                    metric_col_width, metric_row_height,
                                    facecolor=team_colors['primary'], alpha=0.6,
                                    edgecolor='white', linewidth=0.3)  # Reducido de 0.5 a 0.3
            ax.add_patch(metric_bg)
            
            # Nombre de la métrica (ABREVIATURAS MÁS CORTAS)
            metric_map = {
                'Minutos CON posesión': 'Min CON',
                'Minutos SIN posesión': 'Min SIN',
                'Distancia CON posesión': 'Dist CON',
                'Distancia SIN posesión': 'Dist SIN',
                'Distancia >21 km / h CON posesión': '>21kmh CON',
                'Distancia >21 km / h SIN posesión': '>21kmh SIN',
                'Distancia >24 km / h CON posesión': '>24kmh CON',
                'Distancia >24 km / h SIN posesión': '>24kmh SIN'
            }
            metric_name = metric_map.get(metric, metric) # <-- CAMBIO: Lógica de abreviación

            ax.text(x - table_width/2 + metric_col_width/2, metric_y, metric_name, 
                    fontsize=6, weight='bold', color='white',  # <-- CAMBIO: de 7 a 6
                    ha='center', va='center')
            
            # Valores para cada jugador
            for j, player in enumerate(players_list):
                player_x = x - table_width/2 + metric_col_width + (j * player_col_width) + player_col_width/2
                
                # Definir value manejando NaN y valores vacíos
                raw_value = player.get(metric, 0)
                value = pd.to_numeric(raw_value, errors='coerce')

                if pd.isna(value) or value == 0:
                    formatted_value = "0"
                    value = 0
                else:
                    if 'Velocidad' in metric:
                        formatted_value = f"{value:.1f}"
                    elif 'Minutos' in metric or 'Min' in metric:
                        formatted_value = f"{value:.0f}"
                    else:
                        formatted_value = f"{value:.0f}"
                
                # 🔥 DESTACAR VALORES ALTOS CON COLOR DORADO (para datos máximos)
                max_value = max([p.get(metric, 0) for p in players_list])
                text_color = '#FFD700' if value == max_value and value > 0 else 'white'
                
                ax.text(player_x, metric_y, formatted_value, 
                        fontsize=6, weight='bold', color=text_color,  # Dorado para valores máximos
                        ha='center', va='center')
        
        # 🔹 LÍNEAS SEPARADORAS ELEGANTES (MÁS FINAS)
        # Línea horizontal debajo de nombres
        ax.plot([x - table_width/2 + metric_col_width, x + table_width/2], 
                [names_y - names_height/2, names_y - names_height/2], 
                color='white', linewidth=1.5, alpha=0.8)  # Reducido de 2 a 1.5
    
    def create_team_summary_table(self, team_data, ax, x_pos, y_pos, team_name, team_colors, team_logo=None):
        """🔥 NUEVA FUNCIÓN: Crea una tabla de resumen del equipo con VALORES MÁXIMOS"""
        
        # 🔥 CALCULAR ESTADÍSTICAS MÁXIMAS DEL EQUIPO
        summary_stats = {}
        
        for metric in self.metricas_principales:
            if metric in team_data.columns:
                # TODOS LOS VALORES SON MÁXIMOS (ya que cada jugador tiene sus máximos)
                summary_stats[metric] = team_data[metric].max()
        
        # Dimensiones de la tabla (2 FILAS)
        num_metrics = len(summary_stats)
        metric_col_width = 7  # Ancho por cada métrica
        table_width = num_metrics * metric_col_width
        row_height = 1.5  # Altura de cada fila
        table_height = row_height * 2  # 2 filas
        
        # 🎨 FONDO MODERNO
        main_rect = plt.Rectangle((x_pos - table_width/2, y_pos - table_height/2), 
                                table_width, table_height,
                                facecolor='#2c3e50', alpha=0.95, 
                                edgecolor='white', linewidth=2)
        ax.add_patch(main_rect)
        
        # Efecto de borde superior
        top_rect = plt.Rectangle((x_pos - table_width/2, y_pos + table_height/2 - 0.3), 
                                table_width, 0.3,
                                facecolor=team_colors['primary'], alpha=0.9,
                                edgecolor='none')
        ax.add_patch(top_rect)
        
        # 📍 FILA 1: NOMBRES DE MÉTRICAS
        metrics_y = y_pos + row_height/2  # Fila superior
        
        for i, (metric, value) in enumerate(summary_stats.items()):
            metric_x = x_pos - table_width/2 + (i * metric_col_width) + metric_col_width/2
            
            # Fondo para cada métrica en fila 1
            metric_rect = plt.Rectangle((metric_x - metric_col_width/2, metrics_y - row_height/2), 
                                    metric_col_width, row_height,
                                    facecolor=team_colors['primary'], alpha=0.6, 
                                    edgecolor='white', linewidth=0.5)
            ax.add_patch(metric_rect)
            
            # Nombre de la métrica (MÁS COMPLETO)
            if 'Minutos CON posesión' == metric:
                metric_short = 'Min. CON\nPosesión'
            elif 'Minutos SIN posesión' == metric:
                metric_short = 'Min. SIN\nPosesión'
            elif 'Distancia CON posesión' == metric:
                metric_short = 'Dist. CON\nPosesión'
            elif 'Distancia SIN posesión' == metric:
                metric_short = 'Dist. SIN\nPosesión'
            elif 'Distancia >21 km / h CON posesión' == metric:
                metric_short = '>21 CON\nPosesión'
            elif 'Distancia >21 km / h SIN posesión' == metric:
                metric_short = '>21 SIN\nPosesión'
            elif 'Distancia >24 km / h CON posesión' == metric:
                metric_short = '>24 CON\nPosesión'
            elif 'Distancia >24 km / h SIN posesión' == metric:
                metric_short = '>24 SIN\nPosesión'
            else:
                metric_short = metric[:10]  # Fallback: primeros 10 caracteres
            ax.text(metric_x, metrics_y, metric_short, 
                    fontsize=6, weight='bold', color='white',  # Reducido de 8 a 6
                    ha='center', va='center')
        
        # 📍 FILA 2: VALORES MÁXIMOS DE MÉTRICAS
        values_y = y_pos - row_height/2  # Fila inferior
        
        for i, (metric, value) in enumerate(summary_stats.items()):
            metric_x = x_pos - table_width/2 + (i * metric_col_width) + metric_col_width/2
            
            # Fondo alternado para valores en fila 2
            if i % 2 == 0:
                value_rect = plt.Rectangle((metric_x - metric_col_width/2, values_y - row_height/2), 
                                        metric_col_width, row_height,
                                        facecolor='#3c566e', alpha=0.3, 
                                        edgecolor='none')
                ax.add_patch(value_rect)
            
            # Valor MÁXIMO de la métrica
            if 'Velocidad' in metric:
                formatted_value = f"{value:.1f}"
            else:
                formatted_value = f"{value:.0f}"
            
            # 🔥 VALORES MÁXIMOS EN ROJO BRILLANTE
            ax.text(metric_x, values_y, formatted_value, 
                    fontsize=10, weight='bold', color='#FF4444',  # Rojo brillante para máximos
                    ha='center', va='center')
        
        # 🔹 LÍNEA SEPARADORA entre filas
        ax.plot([x_pos - table_width/2, x_pos + table_width/2], 
                [y_pos, y_pos], 
                color='white', linewidth=1.5, alpha=0.8)
        
        # Líneas verticales separando columnas
        for i in range(1, num_metrics):
            line_x = x_pos - table_width/2 + (i * metric_col_width)
            ax.plot([line_x, line_x], 
                    [y_pos - table_height/2, y_pos + table_height/2], 
                    color='white', linewidth=0.5, alpha=0.6)
    
    def create_visualization(self, equipo_rival, jornadas, figsize=(11.69, 8.27)):
        """🔥 NUEVA FUNCIÓN: Crea la visualización completa con DATOS MÁXIMOS"""
        
        # Crear campo SIN espacios
        fig, ax = self.create_campo_sin_espacios(figsize)
        
        # 🔥 TÍTULO ACTUALIZADO PARA DATOS MÁXIMOS
        ax.text(60, 78, f'DATOS MÁXIMOS - ÚLTIMAS {len(jornadas)} JORNADAS | AL MENOS 70\' JUGADOS', 
                fontsize=14, weight='bold', color='white', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.8", facecolor='#1e3d59', alpha=0.95,
                         edgecolor='white', linewidth=2))
        
        # 🔥 OBTENER DATOS MÁXIMOS de ambos equipos (en lugar de acumulados)
        villarreal_data = self.filter_and_get_maximum_data('Villarreal CF', jornadas)
        rival_data = self.filter_and_get_maximum_data(equipo_rival, jornadas)
        
        if villarreal_data is None or len(villarreal_data) == 0:
            print("❌ No hay jugadores de Villarreal CF con al menos 70 minutos en una jornada")
            return None
            
        if rival_data is None or len(rival_data) == 0:
            print(f"❌ No hay jugadores de {equipo_rival} con al menos 70 minutos en una jornada")
            return None
        
        # Cargar escudos
        villarreal_logo = self.load_team_logo('Villarreal CF')
        rival_logo = self.load_team_logo(equipo_rival)
        
        # Posicionar escudos dentro del campo
        if villarreal_logo is not None:
            imagebox = OffsetImage(villarreal_logo, zoom=0.45)
            ab = AnnotationBbox(imagebox, (5, 8), frameon=False)
            ax.add_artist(ab)
        
        if rival_logo is not None:
            imagebox = OffsetImage(rival_logo, zoom=0.45)
            ab = AnnotationBbox(imagebox, (115, 8), frameon=False)
            ax.add_artist(ab)
        
        # Agrupar jugadores por posiciones específicas
        villarreal_grouped = self.group_players_by_final_position(villarreal_data)
        rival_grouped = self.group_players_by_final_position(rival_data)

        # 🔥 APLICAR BALANCEO Y DIVISIÓN EQUITATIVA
        villarreal_grouped = self.redistribute_and_split_players(villarreal_grouped)
        rival_grouped = self.redistribute_and_split_players(rival_grouped)

        # Obtener colores para cada equipo
        villarreal_colors = self.get_team_colors('Villarreal CF')
        rival_colors = self.get_team_colors(equipo_rival)

        # Crear tablas para Villarreal
        for position, players in villarreal_grouped.items():
            if players and position in self.coordenadas_tablas['villarreal']:
                x, y = self.coordenadas_tablas['villarreal'][position]
                position_name = position.replace('_', ' ').title()
                self.create_position_table(players, x, y, ax, villarreal_colors, 
                        position_name, villarreal_logo)

        # ✅ MANEJAR SEGUNDO_DELANTERO SI EXISTE
        if 'SEGUNDO_DELANTERO' in villarreal_grouped and villarreal_grouped['SEGUNDO_DELANTERO']:
            x, y = self.coordenadas_tablas['villarreal']['SEGUNDO_DELANTERO']
            self.create_position_table(villarreal_grouped['SEGUNDO_DELANTERO'], x, y, ax, 
                         villarreal_colors, 'Segundo Delantero', villarreal_logo)

        # Crear tablas para equipo rival
        for position, players in rival_grouped.items():
            if players and position in self.coordenadas_tablas['rival']:
                x, y = self.coordenadas_tablas['rival'][position]
                position_name = position.replace('_', ' ').title()
                self.create_position_table(players, x, y, ax, rival_colors, 
                                        position_name, rival_logo)

        # ✅ MANEJAR SEGUNDO_DELANTERO SI EXISTE
        if 'SEGUNDO_DELANTERO' in rival_grouped and rival_grouped['SEGUNDO_DELANTERO']:
            x, y = self.coordenadas_tablas['rival']['SEGUNDO_DELANTERO']
            self.create_position_table(rival_grouped['SEGUNDO_DELANTERO'], x, y, ax, 
                                     rival_colors, 'Segundo Delantero', rival_logo)
        
        # 🔥 RESÚMENES DE EQUIPOS CON DATOS MÁXIMOS
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

def seleccionar_equipo_jornadas_maximos():
    """Permite al usuario seleccionar un equipo rival y jornadas para DATOS MÁXIMOS"""
    try:
        report_generator = CampoFutbolMaximos()
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

def main_campo_futbol_maximos():
    """🔥 FUNCIÓN PRINCIPAL PARA GENERAR EL INFORME CON DATOS MÁXIMOS"""
    try:
        pass
        
        # Selección interactiva
        equipo_rival, jornadas = seleccionar_equipo_jornadas_maximos()
        
        if equipo_rival is None or jornadas is None:
            print("❌ No se pudo completar la selección.")
            return
        
        
        # Crear el reporte
        report_generator = CampoFutbolMaximos()
        fig = report_generator.create_visualization(equipo_rival, jornadas)
        
        if fig:
            # Mostrar en pantalla
            plt.show()
            
            # Guardar
            equipo_filename = equipo_rival.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_MAXIMOS_Villarreal_vs_{equipo_filename}.pdf"
            
            report_generator.guardar_sin_espacios(fig, output_path)
            
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()

def generar_reporte_campo_maximos(equipo_rival, jornadas, mostrar=True, guardar=True):
    """🔥 FUNCIÓN PARA GENERAR UN REPORTE PERSONALIZADO CON DATOS MÁXIMOS"""
    try:
        report_generator = CampoFutbolMaximos()
        fig = report_generator.create_visualization(equipo_rival, jornadas)
        
        if fig:
            if mostrar:
                plt.show()
            
            if guardar:
                equipo_filename = equipo_rival.replace(' ', '_').replace('/', '_')
                output_path = f"reporte_MAXIMOS_Villarreal_vs_{equipo_filename}.pdf"
                report_generator.guardar_sin_espacios(fig, output_path)
            
            return fig
        else:
            print("❌ No se pudo generar la visualización")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# 🔥 INICIALIZACIÓN PARA DATOS MÁXIMOS
try:
    report_generator = CampoFutbolMaximos()
    equipos = report_generator.get_available_teams()
    
    if len(equipos) > 0:
        pass
    
except Exception as e:
    print(f"❌ Error al inicializar: {e}")

if __name__ == "__main__":
    main_campo_futbol_maximos()