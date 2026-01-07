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
    print("Instalando mplsoccer...")
    import subprocess
    subprocess.check_call(["pip", "install", "mplsoccer"])
    from mplsoccer import Pitch

class CampoFutbolGraficos:
    def __init__(self, data_path="extraccion_mediacoach/data/rendimiento_fisico.parquet"):
        """
        Inicializa la clase para generar informes con gráficos de líneas por demarcación
        """
        self.data_path = data_path
        self.df = None
        self.load_data()
        self.clean_team_names()
        # AÑADIR ESTAS DOS LÍNEAS:
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
        self.coordenadas_graficos = {
            # Villarreal (lado izquierdo)
            'villarreal': {
                'PORTERO': (10, 40),              # Portería
                'LATERAL_DERECHO': (25, 12),      # Lateral derecho (arriba)
                'CENTRAL_DERECHO': (20, 25),      # Central derecho (centro-arriba)
                'CENTRAL_IZQUIERDO': (20, 53),    # Central izquierdo (centro-abajo)
                'LATERAL_IZQUIERDO': (25, 68),    # Lateral izquierdo (abajo)
                'MC_POSICIONAL': (33, 40),        # Mediocampo defensivo (centro)
                'MC_BOX_TO_BOX': (62, 55),        # Box to box (centro-arriba)
                'MC_ORGANIZADOR': (51, 40),       # Organizador (centro-abajo)
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
        
        # 🔥 MÉTRICAS PARA LOS GRÁFICOS DE LÍNEAS
        self.metricas_graficos = [
            'Distancia Total 14-21 km / h',
            'Distancia Total >21 km / h', 
            'Distancia Total'
        ]
        
        # 🔥 MÉTRICAS AMPLIADAS PARA RESUMEN DE EQUIPOS
        self.metricas_resumen_equipos = [
            'Distancia Total 14-21 km / h',
            'Distancia Total >21 km / h',
            'Distancia Total  21-24 km / h',
            'Distancia Total >24 km / h',
            'Velocidad Máxima Total'
        ]
        
        # ✅ NUEVO - Colores más diferenciados
        self.colores_metricas = [
            '#FF6B6B',  # Rojo para 14-21 km/h
            '#00FF7F',  # Verde brillante para >21 km/h  
            '#87CEEB',  # Azul claro para Distancia Total (en vez de amarillo)
        ]
    def load_opta_positions(self):
        """Carga las posiciones desde el archivo Opta"""
        try:
            opta_path = "extraccion_opta/datos_opta_parquet/player_stats.parquet"
            self.opta_df = pd.read_parquet(opta_path)
            print(f"✅ Datos Opta cargados: {self.opta_df.shape[0]} filas")
            
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
                print(f"✅ Columnas opcionales disponibles: {available_optional}")
            
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

        # Normalizar ambos nombres
        norm1 = self.normalize_text(team1)
        norm2 = self.normalize_text(team2)

        # Palabras comunes que NO identifican equipos (sin RC)
        common_words = {
            'club', 'cf', 'fc', 'de', 'del', 'la', 'el', 'deportivo',
            'sociedad', 'union', 'racing', 'sporting', 'gimnastic', 'cd', 'ud', 'rcd'
        }
        
        # Filtrar palabras comunes de ambos equipos
        words1 = [w for w in norm1.split() if w not in common_words and len(w) > 2]
        words2 = [w for w in norm2.split() if w not in common_words and len(w) > 2]
        
        # Si después del filtrado no quedan palabras significativas, usar nombres originales
        if not words1 or not words2:
            words1 = [w for w in norm1.split() if len(w) > 1]  # Cambiar de >2 a >1
            words2 = [w for w in norm2.split() if len(w) > 1]  # Cambiar de >2 a >1
        
        # Convertir a conjuntos
        words1_set = set(words1)
        words2_set = set(words2)
        
        # Verificar subset
        if words1_set.issubset(words2_set) or words2_set.issubset(words1_set):
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
                    return int(jornada[1:])  # j1 → 1, j2 → 2
                else:
                    return int(jornada)  # '1' → 1
            else:
                return int(jornada)  # 1 → 1
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
        
        # Si no tiene dorsal, no puede hacer match
        if player_dorsal is None:
            print(f"   🔍 {player_alias}: Sin dorsal → no buscar en Opta")
            return None
        
        # Convertir jornada de MediaCoach (j1, j2) a Opta (1, 2)
        opta_week = self.convert_jornada_to_week(jornada)
        if opta_week is None:
            print(f"   🔍 {player_alias}: Jornada '{jornada}' no válida → no buscar en Opta")
            return None
        
        print(f"   🔍 {player_alias}: Buscando Dorsal={player_dorsal}, Week={opta_week}, Team≈{team_name}")
        
        # Buscar en todos los jugadores Opta
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
            
            # 3. VERIFICAR EQUIPO (con lógica de subconjunto de palabras)
            team_match = False
            if 'Team Name' in opta_player:
                opta_team = str(opta_player.get('Team Name', ''))
                # Usamos nuestra nueva función precisa
                team_match = self.are_teams_equivalent(team_name, opta_team)

            if not team_match:
                continue
            
            # 🎯 MATCH ENCONTRADO - Ya no importa el nombre
            print(f"     ✅ MATCH TRIPLE: Dorsal={player_dorsal}, Week={opta_week}, Team={opta_team}")

            # Verificar si la posición Opta es válida
            opta_position_raw = opta_player.get('Position')
            opta_position_side_raw = opta_player.get('Position Side')
            
            # ✅ MOSTRAR POSICIÓN ORIGINAL PARA DEBUG
            print(f"     📍 Opta Original: Position='{opta_position_raw}' | Position Side='{opta_position_side_raw}'")
            
            if pd.isna(opta_position_raw) or opta_position_raw == "Substitute" or str(opta_position_raw).strip() == "":
                print(f"   🔄 {player_alias}: Posición Opta inválida ({opta_position_raw}), usando MediaCoach fallback")
                return None  # Esto hará que use el fallback de MediaCoach
                
            # Mapear posición Opta válida
            opta_position = self.map_opta_position_to_system(
                opta_position_raw, 
                opta_position_side_raw,
                team_name,
                opta_week   
            )

            if opta_position:
                print(f"   ✅ {player_alias}: {opta_position} (Opta: {opta_position_raw} + {opta_position_side_raw})")
                return opta_position
            else:
                print(f"   ❌ {player_alias}: Posición Opta no mapeada: {opta_position_raw} {opta_position_side_raw}")
                return None
        
        print(f"   ❌ {player_alias}: Sin triple match en Opta")
        return None

    def load_data(self):
        """Carga los datos del archivo parquet"""
        try:
            self.df = pd.read_parquet(self.data_path)
            print(f"✅ Datos cargados exitosamente: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
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
        print(f"✅ Limpieza completada. Equipos únicos: {len(self.df['Equipo'].unique())}")
        
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
        print("🔄 Rellenando demarcaciones vacías...")
        
        # Crear copia para trabajar
        df_work = df.copy()
        
        # Identificar registros con demarcación vacía
        mask_empty = df_work['Demarcacion'].isna() | (df_work['Demarcacion'] == '') | (df_work['Demarcacion'].str.strip() == '')
        empty_count = mask_empty.sum()
        
        if empty_count > 0:
            print(f"📝 Encontrados {empty_count} registros con demarcación vacía")
            
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
                    print(f"   ✅ {jugador_alias}: {demarcacion_mas_frecuente} (histórico)")
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
        print(f"🎯 Buscando posiciones Opta para {equipo}...")
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
            print("⚠️  Columna 'Minutos jugados' no encontrada.")
            return None
        
        # PASO 2: Acumular datos y determinar la posición final
        print(f"🔄 Procesando datos acumulados por jugador para {equipo}...")
        accumulated_data = []
        
        for dorsal in filtered_df['Dorsal'].unique():
            # Filtrar por dorsal en lugar de alias
            jugador_data = filtered_df[filtered_df['Dorsal'] == dorsal]
            jugador_data_filtered = jugador_data[jugador_data['Minutos jugados'] >= min_avg_minutes]
            
            if len(jugador_data_filtered) > 0:
                # Ordenar por jornada para obtener el registro MÁS RECIENTE
                jugador_data_sorted = jugador_data_filtered.sort_values('Jornada', ascending=True)
                latest_record = jugador_data_sorted.iloc[-1]  # Último partido = nombre más reciente

                # LÓGICA DE POSICIÓN FINAL (DE FISICO9)
                opta_positions = jugador_data_filtered['Opta_Position'].dropna()
                if len(opta_positions) > 0:
                    final_position = opta_positions.mode().iloc[0]
                    position_source = "OPTA"
                else:
                    demarcacion = jugador_data_filtered['Demarcacion'].mode().iloc[0] if len(jugador_data_filtered['Demarcacion'].mode()) > 0 else 'Sin Posición'
                    final_position = self.demarcacion_to_position.get(demarcacion, 'MC_BOX_TO_BOX')
                    position_source = "MediaCoach Fallback"

                accumulated_record = {
                    'Id Jugador': latest_record['Id Jugador'],
                    'Dorsal': latest_record['Dorsal'],
                    'Nombre': latest_record['Nombre'],
                    'Alias': latest_record['Alias'],
                    'Final_Position': final_position,
                    'Position_Source': position_source,
                    'Equipo': latest_record['Equipo'],
                    'Minutos jugados': jugador_data_filtered['Minutos jugados'].mean(),
                    'Distancia Total': jugador_data_filtered['Distancia Total'].sum(),
                    'Distancia Total 14-21 km / h': jugador_data_filtered['Distancia Total 14-21 km / h'].sum(),
                    'Distancia Total >21 km / h': jugador_data_filtered['Distancia Total >21 km / h'].sum(),
                    'Distancia Total  21-24 km / h': jugador_data_filtered.get('Distancia Total  21-24 km / h', pd.Series([0])).sum(),
                    'Distancia Total >24 km / h': jugador_data_filtered.get('Distancia Total >24 km / h', pd.Series([0])).sum(),
                    'Velocidad Máxima Total': jugador_data_filtered['Velocidad Máxima Total'].max(),
                }
                
                print(f"   ✅ {latest_record['Alias']}: {final_position} ({position_source})")
                accumulated_data.append(accumulated_record)
        
        if accumulated_data:
            result_df = pd.DataFrame(accumulated_data)
            print(f"✅ {len(result_df)} jugadores con al menos 1 partido de {min_avg_minutes}+ minutos")
            return result_df
        else:
            print(f"❌ No hay jugadores con al menos 1 partido de {min_avg_minutes}+ minutos para {equipo}")
            return None
    
    def load_team_logo(self, equipo):
        """Carga el escudo del equipo con búsqueda inteligente"""
        
        # Primero, obtener todos los archivos disponibles
        escudos_dir = "assets/escudos"
        if not os.path.exists(escudos_dir):
            return None
        
        available_files = [f for f in os.listdir(escudos_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Crear variaciones del nombre
        equipo_norm = self.normalize_text(equipo)
        possible_names = [
            equipo,
            equipo.lower(),
            equipo.upper(),
            equipo_norm,
            equipo_norm.replace(' ', '_'),
            equipo_norm.replace(' ', '-'),
            equipo_norm.replace(' ', ''),
            # Variaciones sin artículos
            equipo_norm.replace('rc ', '').replace('rcd ', '').replace('cf ', '').replace('fc ', ''),
            equipo_norm.replace('rc', '').replace('rcd', '').replace('cf', '').replace('fc', ''),
            # Palabras individuales
            equipo_norm.split()[0] if ' ' in equipo_norm else equipo_norm,
            equipo_norm.split()[-1] if ' ' in equipo_norm else equipo_norm,
            # *** AGREGAR ESTAS LÍNEAS ESPECÍFICAS ***
            'celta vigo' if 'celta' in equipo_norm else '',
            'celta_vigo' if 'celta' in equipo_norm else '',
            'at madrid' if 'atletico' in equipo_norm else '',
            'athletic bilbao' if 'athletic' in equipo_norm else '',
            # Filtrar nombres vacíos al final
        ]

        # Filtrar nombres vacíos
        possible_names = [name for name in possible_names if name and len(name) > 1]
        
        # Búsqueda exacta
        for name in possible_names:
            for ext in ['.png', '.jpg', '.jpeg']:
                logo_path = f"{escudos_dir}/{name}{ext}"
                if os.path.exists(logo_path):
                    try:
                        return plt.imread(logo_path)
                    except:
                        continue
        
        # Búsqueda por similitud
        best_match = None
        best_similarity = 0
        
        for filename in available_files:
            file_norm = self.normalize_text(filename.split('.')[0])
            for name in possible_names:
                similarity = self.similarity(name, file_norm)
                if similarity > best_similarity and similarity > 0.7:
                    best_similarity = similarity
                    best_match = filename
        
        if best_match:
            logo_path = f"{escudos_dir}/{best_match}"
            try:
                print(f"Escudo encontrado por similitud: {logo_path} (similitud: {best_similarity:.2f})")
                return plt.imread(logo_path)
            except:
                pass
        
        print(f"No se encontró escudo para: {equipo}")
        print(f"Archivos disponibles: {available_files}")
        print(f"📁 Intentando encontrar escudo para: '{equipo}'")
        print(f"   Variaciones probadas: {possible_names[:5]}...")  # Solo mostrar las primeras 5
        return None
    
    @staticmethod
    def normalize_text(text):
        """Normaliza texto eliminando acentos, espacios extra y caracteres especiales"""
        import re
        import unicodedata
        # Eliminar acentos
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        # Convertir a minúsculas y limpiar
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text

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
            
        print("🎯 Agrupando jugadores por posiciones finales para los gráficos...")
        
        # Ordenar por Distancia Total para que los gráficos muestren de mayor a menor
        filtered_df_sorted = filtered_df.sort_values('Distancia Total', ascending=False)
        
        grouped_players = {}
        
        # Agrupar por la nueva columna 'Final_Position'
        for position, group in filtered_df_sorted.groupby('Final_Position'):
            grouped_players[position] = group.to_dict('records')
            print(f"   ✅ Grupo '{position}': {len(group)} jugadores")
            
        return grouped_players
    
    def create_campo_sin_espacios(self, figsize=(11.69, 8.27)):
        """Crea el campo que ocupe TODA la página sin espacios"""
        print("🎯 Creando campo SIN espacios...")
        
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
            # ✅ CORREGIDO: Parámetros completos
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
        """Redistribuye jugadores cuando hay más de max_players_per_position en una posición"""
        
        redistributed = {}
        moved_players = []  # Para tracking
        
        for position, players in grouped_players.items():
            if len(players) <= max_players_per_position:
                redistributed[position] = players
            else:
                print(f"🔄 {position} tiene {len(players)} jugadores, redistribuyendo...")
                
                # Ordenar por minutos jugados (los que más juegan se quedan)
                players_sorted = sorted(players, key=lambda p: p['Minutos jugados'], reverse=True)
                
                # Los primeros 3 se quedan
                redistributed[position] = players_sorted[:max_players_per_position]
                
                # Los demás se redistribuyen
                excess_players = players_sorted[max_players_per_position:]
                
                for player in excess_players:
                    # Obtener historial de posiciones del jugador
                    position_history = self.get_player_position_history(
                        player['Id Jugador'], 
                        player['Equipo']
                    )
                    
                    # Buscar una posición alternativa con espacio
                    relocated = False
                    for alt_position, frequency in position_history:
                        if alt_position != position:  # No la posición actual
                            current_count = len(redistributed.get(alt_position, []))
                            if current_count < max_players_per_position:
                                # Mover a esta posición
                                if alt_position not in redistributed:
                                    redistributed[alt_position] = []
                                redistributed[alt_position].append(player)
                                moved_players.append((player['Alias'], position, alt_position))
                                relocated = True
                                break
                    
                    if not relocated:
                        # ✅ NUEVO - Forzar reubicación en posición menos poblada
                        if redistributed:  # Verificar que hay posiciones disponibles
                            min_populated_position = min(redistributed.keys(), 
                                                    key=lambda pos: len(redistributed[pos]))
                            
                            if len(redistributed[min_populated_position]) < max_players_per_position:
                                redistributed[min_populated_position].append(player)
                                moved_players.append((player['Alias'], position, min_populated_position))
                                print(f"🔄 Forzado: {player['Alias']}: {position} → {min_populated_position}")
                            else:
                                # Último recurso: no agregar el jugador
                                print(f"⚠️ ADVERTENCIA: {player['Alias']} excede límite, no se incluye en gráfico")
        
        # ✅ MOVER AQUÍ - Mostrar movimientos realizados AL FINAL
        for player_name, from_pos, to_pos in moved_players:
            print(f"   ✅ {player_name}: {from_pos} → {to_pos}")
        
        # ✅ AGREGAR RETURN AL FINAL
        return redistributed

    def create_position_graph(self, players_list, demarcacion, x, y, ax, team_colors, team_logo=None):
        """🔥 NUEVA FUNCIÓN: Crea un gráfico de líneas para cada demarcación"""
        if not players_list or len(players_list) < 1:
            return
        
        print(f"🎯 Creando gráfico para {demarcacion} con {len(players_list)} jugadores")
        num_players = len(players_list)

        # Dimensiones del gráfico
        graph_width = 18
        graph_height = 12
        
        # Fondo del gráfico - AMARILLO PARA VILLARREAL
        if team_colors['primary'] == '#FFD700':  # Color primario de Villarreal
            fondo_color = '#FFD700'  # Amarillo Villarreal
            fondo_alpha = 0.8
        else:
            fondo_color = '#2c3e50'  # Gris oscuro para otros equipos
            fondo_alpha = 0.95

        graph_rect = plt.Rectangle((x - graph_width/2, y - graph_height/2), 
                                 graph_width, graph_height,
                                 facecolor=fondo_color, alpha=fondo_alpha,  # Usar colores dinámicos
                                 edgecolor='white', linewidth=2)
        ax.add_patch(graph_rect)
        
        # Título del gráfico (demarcación limpia)
        if demarcacion == 'MC Box to Box':
            titulo_grafico = 'MC BOX TO BOX + MEDIAPUNTA'
        elif demarcacion == 'Delantero Centro':
            titulo_grafico = 'DELANTERO CENTRO + 2º DELANTERO'
        else:
            titulo_grafico = demarcacion.upper()
        # 🔥 AJUSTAR TAMAÑO DE FUENTE SEGÚN LONGITUD
        if len(titulo_grafico) > 20:
            font_size = 7  # Más pequeño para títulos largos
        elif len(titulo_grafico) > 15:
            font_size = 8  # Mediano
        else:
            font_size = 9  # Normal

        ax.text(x, y + graph_height/2 - 0.5, titulo_grafico, 
                fontsize=font_size, weight='bold', color='white',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#1e3d59', alpha=0.8))
        
        # 🏆 AÑADIR ESCUDO en la esquina superior izquierda
        if team_logo is not None:
            try:
                logo_x = x - graph_width/2 + 1.5
                logo_y = y + graph_height/2 - 1.5
                zoom_factor = 0.075
                
                imagebox = OffsetImage(team_logo, zoom=zoom_factor)
                ab = AnnotationBbox(imagebox, (logo_x, logo_y), 
                                frameon=False, 
                                boxcoords='data')
                ax.add_artist(ab)
            except Exception as e:
                print(f"⚠️  Error al añadir escudo: {e}")
        
        # Preparar datos para el gráfico
        # LÓGICA CORREGIDA PARA POSICIONES
        if num_players == 1:
            x_positions = [x]  # Solo una posición X
        else:
            x_positions = np.linspace(x - graph_width/2 + 3, x + graph_width/2 - 3, num_players)
        
        # Calcular rangos para normalización Y
        y_min = y - graph_height/2 + 3
        y_max = y + graph_height/2 - 4
        
        # Crear líneas para cada métrica
        for i, metric in enumerate(self.metricas_graficos):
            metric_values = []
            
            # Recopilar valores de la métrica para todos los jugadores
            for player in players_list:
                value = player.get(metric, 0)
                metric_values.append(value)
            
            if not metric_values or all(v == 0 for v in metric_values):
                continue  # Saltar métricas sin datos
            
            # Normalizar valores para el rango Y
            min_val = min(metric_values)
            max_val = max(metric_values)
            
            if max_val > min_val:
                normalized_values = [(val - min_val) / (max_val - min_val) for val in metric_values]
            else:
                normalized_values = [0.5] * len(metric_values)  # Todos iguales al centro
            
            # Mapear a posiciones Y
            y_positions = [y_min + norm * (y_max - y_min) for norm in normalized_values]
            
            # Dibujar línea de la métrica
            color = self.colores_metricas[i % len(self.colores_metricas)]
            
            # ✅ CÓDIGO CORREGIDO
            if num_players == 1:
                # Para 1 jugador: posicionar cada punto según el valor real de la métrica
                player = players_list[0]
                metric_value = player.get(metric, 0)
                
                # Crear rango basado en los valores del jugador para normalización
                all_player_values = [
                    player.get('Distancia Total 14-21 km / h', 0),
                    player.get('Distancia Total >21 km / h', 0),
                    player.get('Distancia Total', 0)
                ]
                
                # Filtrar valores cero para normalización
                non_zero_values = [v for v in all_player_values if v > 0]
                
                if non_zero_values:
                    min_val = min(non_zero_values)
                    max_val = max(non_zero_values)
                    
                    if max_val > min_val:
                        # Normalizar el valor actual
                        normalized_value = (metric_value - min_val) / (max_val - min_val)
                    else:
                        # Si todos los valores son iguales
                        normalized_value = 0.5
                else:
                    # Si todos los valores son cero
                    normalized_value = 0.5
                
                # Posicionar el punto según el valor normalizado
                y_point = y_min + normalized_value * (y_max - y_min)
                
                ax.plot(x_positions[0], y_point, 
                        color=color, marker='o', markersize=8, alpha=0.9)
            else:
                # Para múltiples jugadores: líneas normales
                ax.plot(x_positions, y_positions, 
                        color=color, linewidth=3, 
                        marker='o', markersize=5, alpha=0.9)
            
            # Etiqueta de la métrica
            if '14-21' in metric:
                label = '14-21 Km/h'
            elif '>21' in metric:
                label = '>21 Km/h'
            elif 'Distancia Total' == metric:
                label = 'Distancia'
            else:
                label = metric[:8]
            
            # Leyenda en la parte superior
            # legend_x = x - graph_width/2 + 3 + (i * 5)
            # ax.text(legend_x, y + graph_height/2 - 3, f"● {label}", 
            #       fontsize=6, color=color, weight='bold',  # 🔥 AUMENTADO de 5 a 6
            #       ha='left', va='center')
        
        # Nombres de jugadores en el eje X (rotados)
        for i, (x_pos, player) in enumerate(zip(x_positions, players_list)):
            player_name = player.get('Alias', 'N/A')
            dorsal_raw = player.get('Dorsal', '')

            # --- INICIO: Lógica de formateo de dorsal ---
            # Formatear el dorsal para que sea un entero y manejar valores vacíos
            if pd.notna(dorsal_raw) and dorsal_raw != '':
                try:
                    # Convertir a float (por si es texto), luego a int (para quitar decimales), y finalmente a string
                    dorsal_text = str(int(float(dorsal_raw)))
                except (ValueError, TypeError):
                    dorsal_text = str(dorsal_raw) # Fallback si no es un número (ej. 'N/A')
            else:
                dorsal_text = 'S/N' # Para "Sin Número" si el dorsal está vacío
            # --- FIN: Lógica de formateo de dorsal ---

            # Mostrar dorsal y nombre usando el texto ya formateado
            max_chars_per_line = 8  # Máximo caracteres por línea

            if len(player_name) > max_chars_per_line:
                # Dividir el nombre en dos líneas
                words = player_name.split()
                if len(words) >= 2:
                    # Si hay múltiples palabras, dividir por palabras
                    mid_point = len(words) // 2
                    first_line = " ".join(words[:mid_point])
                    second_line = " ".join(words[mid_point:])
                    display_text = f"{dorsal_text}\n{first_line}\n{second_line}"
                else:
                    # Si es una sola palabra larga, dividir por caracteres
                    mid_char = len(player_name) // 2
                    first_part = player_name[:mid_char]
                    second_part = player_name[mid_char:]
                    display_text = f"{dorsal_text}\n{first_part}\n{second_part}"
            else:
                # Nombre normal en una línea
                display_text = f"{dorsal_text}\n{player_name}"

            # Color de texto contrastante para fondo amarillo
            if team_colors['primary'] == '#FFD700':  # Villarreal
                text_color = '#1e3d59'  # Azul oscuro para contraste con amarillo
            else:
                text_color = 'white'  # Blanco para fondos oscuros

            ax.text(x_pos, y_min + 0.4, display_text, 
                    fontsize=7, color=text_color, weight='bold',  # Reducido a 9 para que quepa mejor
                    ha='center', va='top', rotation=0)
        
        # Líneas de rejilla horizontales sutiles
        for i in range(1, 4):
            grid_y = y_min + (i/4) * (y_max - y_min)
            ax.plot([x - graph_width/2 + 2, x + graph_width/2 - 2], 
                    [grid_y, grid_y], 
                    color='white', linewidth=0.5, alpha=0.3, linestyle='--')
    
    def create_team_summary_table(self, team_data, ax, x_pos, y_pos, team_name, team_colors, team_logo=None):
        """Crea una tabla de resumen del equipo con métricas ampliadas"""
        
        # Calcular estadísticas del equipo
        summary_stats = {}
        
        for metric in self.metricas_resumen_equipos:
            if metric in team_data.columns:
                if 'Velocidad Máxima' in metric:
                    summary_stats[metric] = team_data[metric].max()
                else:
                    summary_stats[metric] = team_data[metric].mean()
        
        # Dimensiones de la tabla (2 FILAS)
        num_metrics = len(summary_stats)
        metric_col_width = 8  # Ancho por cada métrica
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
            
            # Nombre de la métrica (ABREVIADO)
            if '14-21' in metric:
                metric_short = '14-21 Km/h'
            elif '>21' in metric and '21-24' not in metric:
                metric_short = '>21 Km/h'
            elif '21-24' in metric:
                metric_short = '21-24 Km/h'
            elif '>24' in metric:
                metric_short = '>24 Km/h'
            elif 'Velocidad Máxima' in metric:
                metric_short = 'V.Max'
            else:
                metric_short = metric.replace('Distancia Total ', '').replace('Distancia Total', 'Distancia')
                
            ax.text(metric_x, metrics_y, metric_short, 
                    fontsize=6, weight='bold', color='white',
                    ha='center', va='center')
        
        # 📍 FILA 2: VALORES DE MÉTRICAS
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
            
            # Valor de la métrica
            if 'Velocidad' in metric:
                formatted_value = f"{value:.1f}"
            else:
                formatted_value = f"{value:.0f}"
            
            ax.text(metric_x, values_y, formatted_value, 
                    fontsize=9, weight='bold', color='#FFD700',  # Valores en dorado
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
    
    def get_position_for_demarcation(self, demarcacion_display, team_side):
        """Obtiene la posición correcta para una demarcación específica basándose en el mapeo original"""
        
        # 🔥 MAPEO ESPECIAL PARA DEMARCACIONES COMBINADAS
        demarcacion_to_position_map = {
            # Combinadas
            'MC Box to Box': 'MC_BOX_TO_BOX',        # Para mediapunta + box to box
            'Delantero Centro': 'DELANTERO_CENTRO',  # Para delantero centro + segundo delantero
            
            # Individuales (nombres limpios)
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
        
        # Obtener posición mapeada
        position = demarcacion_to_position_map.get(demarcacion_display, 'MC_BOX_TO_BOX')
        
        # Obtener coordenadas de esa posición
        if position in self.coordenadas_graficos[team_side]:
            return self.coordenadas_graficos[team_side][position]
        else:
            # Si no encuentra la posición, usar una por defecto
            return self.coordenadas_graficos[team_side]['MC_BOX_TO_BOX']
    
    def create_visualization(self, equipo_rival, jornadas, figsize=(11.69, 8.27)):
        """Crea la visualización completa con gráficos de líneas por demarcación"""
        
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

        # *** AGREGAR ESTAS LÍNEAS DE DEBUG ***
        print(f"🔍 Debug escudos:")
        print(f"  - Villarreal logo: {'✅ Cargado' if villarreal_logo is not None else '❌ No encontrado'}")
        print(f"  - {equipo_rival} logo: {'✅ Cargado' if rival_logo is not None else '❌ No encontrado'}")
        
        # Posicionar escudos dentro del campo
        if villarreal_logo is not None:
            imagebox = OffsetImage(villarreal_logo, zoom=0.45)
            ab = AnnotationBbox(imagebox, (5, 5), frameon=False)
            ax.add_artist(ab)
        
        if rival_logo is not None:
            imagebox = OffsetImage(rival_logo, zoom=0.45)
            ab = AnnotationBbox(imagebox, (115, 5), frameon=False)
            ax.add_artist(ab)

        # Agrupar por posición final
        villarreal_by_position = self.group_players_by_final_position(villarreal_data)
        rival_by_position = self.group_players_by_final_position(rival_data)

        # 🔥 NUEVO: Redistribuir jugadores sobrepoblados
        villarreal_by_position = self.redistribute_overcrowded_positions(villarreal_by_position, max_players_per_position=3)
        rival_by_position = self.redistribute_overcrowded_positions(rival_by_position, max_players_per_position=3)

        # Obtener colores para cada equipo
        villarreal_colors = self.get_team_colors('Villarreal CF')
        rival_colors = self.get_team_colors(equipo_rival)

        # --- Bucle para Villarreal ---
        print(f"🔄 Creando {len(villarreal_by_position)} gráficos para Villarreal CF")
        for position, players in villarreal_by_position.items():
            if players and position in self.coordenadas_graficos['villarreal']:
                x, y = self.coordenadas_graficos['villarreal'][position]
                # El título del gráfico ahora es el nombre de la posición
                self.create_position_graph(players, position, x, y, ax, 
                                         villarreal_colors, villarreal_logo)

        # --- Bucle para el Equipo Rival ---
        print(f"🔄 Creando {len(rival_by_position)} gráficos para {equipo_rival}")
        for position, players in rival_by_position.items():
            if players and position in self.coordenadas_graficos['rival']:
                x, y = self.coordenadas_graficos['rival'][position]
                self.create_position_graph(players, position, x, y, ax, 
                                         rival_colors, rival_logo)
        
        # Leyenda común arriba a la derecha
        legend_x = 110
        legend_y = 75
        for i, metric in enumerate(self.metricas_graficos):
            if '14-21' in metric:
                label = '14-21 Km/h'
            elif '>21' in metric:
                label = '>21 Km/h'
            elif 'Distancia Total' == metric:
                label = 'Distancia'
            else:
                label = metric[:8]
            
            color = self.colores_metricas[i % len(self.colores_metricas)]
            ax.text(legend_x, legend_y - (i * 3), f"● {label}", 
                    fontsize=8, color=color, weight='bold',
                    ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))

        # Resúmenes de equipos con métricas ampliadas
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
        print(f"✅ Archivo guardado SIN espacios: {filename}")

def seleccionar_equipo_jornadas_graficos():
    """Permite al usuario seleccionar un equipo rival y jornadas"""
    try:
        report_generator = CampoFutbolGraficos()
        equipos = report_generator.get_available_teams()
        
        # Filtrar Villarreal CF de la lista de oponentes
        equipos_rival = [eq for eq in equipos if 'Villarreal' not in eq]
        
        if len(equipos_rival) == 0:
            print("❌ No se encontraron equipos rivales en los datos.")
            return None, None
        
        print("\n=== SELECCIÓN DE EQUIPO RIVAL - GRÁFICOS DE LÍNEAS POR DEMARCACIÓN ===")
        for i, equipo in enumerate(equipos_rival, 1):
            print(f"{i:2d}. {equipo}")
        
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
        print(f"\nJornadas disponibles: {jornadas_disponibles}")
        
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

def main_campo_futbol_graficos():
    """Función principal para generar el informe con gráficos de líneas por demarcación"""
    try:
        print("🏟️ === GENERADOR DE INFORMES CON GRÁFICOS DE LÍNEAS ===")
        
        # Selección interactiva
        equipo_rival, jornadas = seleccionar_equipo_jornadas_graficos()
        
        if equipo_rival is None or jornadas is None:
            print("❌ No se pudo completar la selección.")
            return
        
        print(f"\n🔄 Generando reporte CON GRÁFICOS DE LÍNEAS para Villarreal CF vs {equipo_rival}")
        print(f"📅 Jornadas: {jornadas}")
        print(f"🔥 Características:")
        print(f"   • Mínimo 60 minutos (en lugar de 70)")
        print(f"   • Gráfico de líneas por cada demarcación")
        print(f"   • Jugadores ordenados por Distancia Total (mayor a menor)")
        print(f"   • 3 líneas por gráfico: 14-21 Km/h, >21 Km/h, Distancia Total")
        print(f"   • Resumen ampliado de equipos")
        
        # Crear el reporte
        report_generator = CampoFutbolGraficos()
        fig = report_generator.create_visualization(equipo_rival, jornadas)
        
        if fig:
            # Mostrar en pantalla
            plt.show()
            
            # Guardar
            equipo_filename = equipo_rival.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_GRAFICOS_Villarreal_vs_{equipo_filename}.pdf"
            
            report_generator.guardar_sin_espacios(fig, output_path)
            
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()

def generar_reporte_graficos_personalizado(equipo_rival, jornadas, mostrar=True, guardar=True):
    """Función para generar un reporte personalizado con gráficos de líneas"""
    try:
        report_generator = CampoFutbolGraficos()
        fig = report_generator.create_visualization(equipo_rival, jornadas)
        
        if fig:
            if mostrar:
                plt.show()
            
            if guardar:
                equipo_filename = equipo_rival.replace(' ', '_').replace('/', '_')
                output_path = f"reporte_GRAFICOS_Villarreal_vs_{equipo_filename}.pdf"
                report_generator.guardar_sin_espacios(fig, output_path)
            
            return fig
        else:
            print("❌ No se pudo generar la visualización")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Inicialización
print("🏟️ === INICIALIZANDO GENERADOR CON GRÁFICOS DE LÍNEAS ===")
try:
    report_generator = CampoFutbolGraficos()
    equipos = report_generator.get_available_teams()
    print(f"\n✅ Sistema CON GRÁFICOS DE LÍNEAS listo. Equipos disponibles: {len(equipos)}")
    
    if len(equipos) > 0:
        print("📝 Para generar un reporte CON GRÁFICOS DE LÍNEAS ejecuta: main_campo_futbol_graficos()")
        print("📝 Para uso directo: generar_reporte_graficos_personalizado('Equipo_Rival', [33,34,35])")
        print("\n🔥 NUEVAS CARACTERÍSTICAS:")
        print("   • Mínimo 60 minutos (en lugar de 70)")
        print("   • GRÁFICO DE LÍNEAS por cada demarcación")
        print("   • Jugadores ordenados por Distancia Total (mayor → menor)")
        print("   • 3 líneas por gráfico: 14-21 Km/h, >21 Km/h, Distancia Total")
        print("   • DEMARCACIONES COMBINADAS:")
        print("     - MC Box to Box + Mediapunta = Mismo gráfico")
        print("     - Delantero Centro + Segundo Delantero = Mismo gráfico")
        print("   • Resumen ampliado: 14-21, >21, 21-24, >24 Km/h + V.Max")
    
except Exception as e:
    print(f"❌ Error al inicializar: {e}")

if __name__ == "__main__":
    main_campo_futbol_graficos()