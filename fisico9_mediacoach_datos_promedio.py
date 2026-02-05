import pandas as pd
from matplotlib import patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import unicodedata
import re
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
    import subprocess
    subprocess.check_call(["pip", "install", "mplsoccer"])
    from mplsoccer import Pitch

class CampoFutbolAcumulado:
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
            self.opta_df = None
            return False
    
    def calculate_table_bounds(self, x, y, num_players):
        """Calcula los límites (bounds) de una tabla"""
        num_metrics = len(self.metricas_principales)
        
        metric_col_width = 4
        player_col_width = 6
        table_width = metric_col_width + (num_players * player_col_width)
        
        header_height = 2
        names_height = 3
        metric_row_height = 1.5
        table_height = header_height + names_height + (num_metrics * metric_row_height)
        
        return {
            'left': x - table_width/2,
            'right': x + table_width/2,
            'bottom': y - table_height/2,
            'top': y + table_height/2,
            'width': table_width,
            'height': table_height
        }

    def tables_overlap(self, bounds1, bounds2, margin=1):
        """Verifica si dos tablas se solapan (con margen de seguridad)"""
        return not (bounds1['right'] + margin < bounds2['left'] or 
                    bounds2['right'] + margin < bounds1['left'] or 
                    bounds1['top'] + margin < bounds2['bottom'] or 
                    bounds2['top'] + margin < bounds1['bottom'])

    def find_nearest_position(self, original_x, original_y, existing_bounds, table_bounds, search_radius=15):
        """Encuentra la posición más cercana sin colisiones"""
        best_x, best_y = original_x, original_y
        min_distance = float('inf')
        
        # Buscar en círculos concéntricos alrededor de la posición original
        for radius in range(1, search_radius):
            for angle in range(0, 360, 15):  # Cada 15 grados
                rad = np.radians(angle)
                test_x = original_x + radius * np.cos(rad)
                test_y = original_y + radius * np.sin(rad)
                
                # Mantener dentro del campo
                if test_x < 5 or test_x > 115 or test_y < 5 or test_y > 75:
                    continue
                
                # Calcular bounds en la nueva posición
                test_bounds = self.calculate_table_bounds(test_x, test_y, 
                                                        table_bounds['width']//10)  # Aproximación
                
                # Verificar si hay colisión
                has_collision = any(self.tables_overlap(test_bounds, existing) 
                                for existing in existing_bounds)
                
                if not has_collision:
                    distance = np.sqrt((test_x - original_x)**2 + (test_y - original_y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        best_x, best_y = test_x, test_y
            
            # Si encontramos una posición en este radio, usarla
            if min_distance < float('inf'):
                break
        
        return best_x, best_y

    def resolve_table_positions(self, team_grouped, team_coords, team_name):
        """Resuelve colisiones entre tablas y devuelve posiciones ajustadas"""
        
        # Recopilar todas las tablas que se van a crear
        tables_to_create = []
        for position, players in team_grouped.items():
            if players and position in team_coords:
                x, y = team_coords[position]
                num_players = len(players)
                bounds = self.calculate_table_bounds(x, y, num_players)
                
                tables_to_create.append({
                    'position': position,
                    'players': players,
                    'original_x': x,
                    'original_y': y,
                    'final_x': x,
                    'final_y': y,
                    'bounds': bounds,
                    'num_players': num_players
                })
        
        # Ordenar por prioridad (portero primero, luego por número de jugadores)
        def get_priority(table):
            if table['position'] == 'PORTERO':
                return 0
            return -table['num_players']  # Más jugadores = mayor prioridad
        
        tables_to_create.sort(key=get_priority)
        
        # Resolver colisiones
        final_positions = {}
        placed_bounds = []
        
        for table in tables_to_create:
            # Verificar si la posición original tiene colisiones
            has_collision = any(self.tables_overlap(table['bounds'], existing) 
                            for existing in placed_bounds)
            
            if has_collision:
                # Buscar nueva posición
                new_x, new_y = self.find_nearest_position(
                    table['original_x'], table['original_y'], 
                    placed_bounds, table['bounds']
                )
                
                # Recalcular bounds con nueva posición
                new_bounds = self.calculate_table_bounds(new_x, new_y, table['num_players'])
                
                
                final_positions[table['position']] = (new_x, new_y)
                placed_bounds.append(new_bounds)
            else:
                # Posición original está bien
                final_positions[table['position']] = (table['original_x'], table['original_y'])
                placed_bounds.append(table['bounds'])
        
        return final_positions

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

        # Palabras a filtrar
        common_words = {
            'club', 'cf', 'fc', 'de', 'del', 'la', 'el', 'deportivo',
            'sociedad', 'union', 'racing', 'sporting', 'gimnastic', 'cd', 'ud', 'rc', 'rcd'
        }
        
        # Filtrar palabras
        words1 = [w for w in norm1.split() if w not in common_words and len(w) > 2]
        words2 = [w for w in norm2.split() if w not in common_words and len(w) > 2]
        
        # Si no quedan palabras, usar original
        if not words1 or not words2:
            words1 = [w for w in norm1.split() if len(w) > 2]
            words2 = [w for w in norm2.split() if len(w) > 2]
        
        words1_set = set(words1)
        words2_set = set(words2)
        
        # ✅ NUEVO: VERIFICAR QUE TODAS LAS PALABRAS COINCIDAN (no solo una)
        if words1_set == words2_set and len(words1_set) > 0:
            return True
        
        # Subset solo si hay MÁS de una palabra en común
        common = words1_set & words2_set
        if len(common) > 1 and (words1_set.issubset(words2_set) or words2_set.issubset(words1_set)):
            return True

        return False

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

    def map_opta_position_to_system(self, position, position_side, has_striker_centre=False):
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
            if "Centre" in position_side:
                return "DELANTERO_CENTRO"
            elif "Centre/Right" in position_side:
                return "BANDA_DERECHA" if has_striker_centre else "DELANTERO_CENTRO"
            elif "Left/Centre" in position_side:
                return "BANDA_IZQUIERDA" if has_striker_centre else "DELANTERO_CENTRO"
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
            return None

    def find_improved_position(self, player_alias, team_name, player_dorsal=None, jornada=None):
        """Busca posición en Opta usando SOLO DORSAL + JORNADA + EQUIPO"""
        if self.opta_df is None:
            return None
        
        # Si no tiene dorsal, no puede hacer match
        if player_dorsal is None:
            return None
        
        # Convertir jornada de MediaCoach (j1, j2) a Opta (1, 2)
        opta_week = self.convert_jornada_to_week(jornada)
        if opta_week is None:
            return None
        
        
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
            
        

            # Verificar si la posición Opta es válida
            opta_position_raw = opta_player.get('Position')
            opta_position_side_raw = opta_player.get('Position Side')
            
            
            if pd.isna(opta_position_raw) or opta_position_raw == "Substitute" or str(opta_position_raw).strip() == "":
                pass
                return None  # Esto hará que use el fallback de MediaCoach

            has_centre = self.has_striker_centre_in_match(team_name, opta_week)
    
            # Mapear posición Opta válida
            opta_position = self.map_opta_position_to_system(
                opta_position_raw, 
                opta_position_side_raw,
                has_centre
            )

            if opta_position:
                pass
                return opta_position
            else:
                print(f"   ❌ {player_alias}: Posición Opta no mapeada: {opta_position_raw} {opta_position_side_raw}")
                return None
        
        print(f"   ❌ {player_alias}: Sin triple match en Opta")
        return None

    def flexible_name_similarity(self, name1, name2):
        """Calcula similitud flexible entre nombres de jugadores"""
        if not name1 or not name2:
            return 0
        
        # Normalizar ambos nombres
        norm1 = self.normalize_text(name1)
        norm2 = self.normalize_text(name2)
        
        # Similitud básica
        basic_similarity = self.similarity(norm1, norm2)
        
        # 🔥 MOVER ESTA SECCIÓN AQUÍ PRIMERO
        # Dividir en palabras
        words1 = norm1.split()
        words2 = norm2.split()

        # Verificar iniciales + apellido
        if len(words1) > 1 and len(words2) > 1:
            # Inicial del nombre + apellido completo
            inicial1_apellido1 = f"{words1[0][0]} {words1[-1]}"
            inicial2_apellido2 = f"{words2[0][0]} {words2[-1]}"
            if inicial1_apellido1 == inicial2_apellido2:
                basic_similarity = max(basic_similarity, 0.8)
        
        # Verificar coincidencias parciales
        if len(words1) > 1 and len(words2) > 1:
            # Verificar si alguna palabra coincide completamente
            common_words = set(words1) & set(words2)
            if common_words:
                basic_similarity = max(basic_similarity, 0.7)
        
        # Verificar apellidos (última palabra)
        if len(words1) > 0 and len(words2) > 0:
            if words1[-1] == words2[-1] and len(words1[-1]) > 3:
                basic_similarity = max(basic_similarity, 0.8)
        
        return basic_similarity

    def enhanced_team_similarity(self, team1, team2):
        """Calcula similitud mejorada entre nombres de equipos"""
        if not team1 or not team2:
            return 0
        
        # Normalizar nombres de equipos
        norm1 = self.normalize_text(team1)
        norm2 = self.normalize_text(team2)
        
        # Similitud básica
        basic_similarity = self.similarity(norm1, norm2)
        
        # 🔥 PALABRAS COMUNES EXPANDIDAS (no identifican equipos únicos)
        common_words = [
            'club', 'cf', 'fc', 'de', 'del', 'la', 'el', 'real', 'atletico', 'deportivo',
            'sociedad', 'union', 'racing', 'sporting', 'gimnastic', 'cd', 'ud', 'rc', 'rcd'
        ]
        
        # Filtrar palabras comunes
        words1 = [w for w in norm1.split() if w not in common_words and len(w) > 2]
        words2 = [w for w in norm2.split() if w not in common_words and len(w) > 2]
        
        if words1 and words2:
            # 🎯 ENCONTRAR PALABRA PRINCIPAL (la más larga que identifica al equipo)
            main_word1 = max(words1, key=len)  # Palabra más larga
            main_word2 = max(words2, key=len)  # Palabra más larga
            
            # 🔥 SI PALABRAS PRINCIPALES COINCIDEN EXACTAMENTE
            if main_word1 == main_word2:
                return 1.0  # Coincidencia perfecta
            
            # 🔥 SI UNA PALABRA PRINCIPAL CONTIENE A LA OTRA
            if main_word1 in main_word2 or main_word2 in main_word1:
                return 0.9  # Alta similitud
            
            # 🔥 SIMILITUD ENTRE PALABRAS PRINCIPALES
            main_similarity = self.similarity(main_word1, main_word2)
            if main_similarity > 0.8:
                return main_similarity
            
            # Comparar todas las palabras filtradas como fallback
            core1 = ' '.join(words1)
            core2 = ' '.join(words2)
            core_similarity = self.similarity(core1, core2)
            basic_similarity = max(basic_similarity, core_similarity)
        
        # 🔥 CASOS ESPECIALES HARDCODEADOS
        special_cases = {
            ('villarreal', 'villareal'): 1.0,
            ('atletico', 'atleti'): 1.0,
            ('barcelona', 'barca'): 1.0,
            ('madrid', 'real'): 0.9,
        }
        
        for (team_a, team_b), similarity in special_cases.items():
            if (team_a in norm1 and team_b in norm2) or (team_b in norm1 and team_a in norm2):
                return similarity
        
        return basic_similarity

    def generate_team_name_variations(self, team_name):
        """Genera variaciones sistemáticas del nombre del equipo"""
        base_name = team_name.strip()
        
        variations = [
            # Nombre original
            base_name,
            base_name.lower(),
            
            # Sin espacios
            base_name.replace(' ', '_'),
            base_name.replace(' ', '_').lower(),
            base_name.replace(' ', ''),
            base_name.replace(' ', '').lower(),
            
            # Sin "CF", "FC", etc.
            base_name.replace(' CF', '').replace(' FC', '').replace(' Club', ''),
            base_name.replace(' CF', '').replace(' FC', '').replace(' Club', '').lower(),
            base_name.replace('CF', '').replace('FC', '').replace('Club', '').strip(),
            base_name.replace('CF', '').replace('FC', '').replace('Club', '').strip().lower(),
            
            # Variaciones con guiones
            base_name.replace(' ', '-'),
            base_name.replace(' ', '-').lower(),
            
            # Solo primera palabra (para casos como "Real Madrid" -> "Real")
            base_name.split()[0] if ' ' in base_name else base_name,
            base_name.split()[0].lower() if ' ' in base_name else base_name.lower(),
            
            # Solo última palabra (para casos como "Real Madrid" -> "Madrid")  
            base_name.split()[-1] if ' ' in base_name else base_name,
            base_name.split()[-1].lower() if ' ' in base_name else base_name.lower(),
            
            # Sin artículos comunes
            base_name.replace('Real ', '').replace('Club ', '').replace('Deportivo ', ''),
            base_name.replace('Real ', '').replace('Club ', '').replace('Deportivo ', '').lower(),
        ]
        
        # Eliminar duplicados manteniendo el orden
        seen = set()
        unique_variations = []
        for variation in variations:
            if variation and variation not in seen:
                unique_variations.append(variation)
                seen.add(variation)
        
        return unique_variations

    def robust_team_matching(self, mediacoach_team, opta_team):
        """Matching robusto entre nombres de equipos usando variaciones sistemáticas"""
        
        # Generar variaciones para ambos equipos
        mc_variations = self.generate_team_name_variations(mediacoach_team)
        opta_variations = self.generate_team_name_variations(opta_team)
        
        # Buscar coincidencias exactas entre variaciones
        for mc_var in mc_variations:
            for opta_var in opta_variations:
                if mc_var.lower() == opta_var.lower():
                    pass
                    return True
        
        # Si no hay coincidencia exacta, usar similitud como fallback
        similarity = self.enhanced_team_similarity(mediacoach_team, opta_team)
        if similarity > 0.7:
            pass
            return True
        
        return False
    
    def find_improved_position_for_player(self, player_alias, team_name, player_dorsal, jornadas_list):
        """Busca TODAS las posiciones Opta para un jugador y devuelve la predominante"""
        if self.opta_df is None or player_dorsal is None:
            return None
        

    def __init__(self, data_path="extraccion_mediacoach/data/rendimiento_fisico.parquet"):
        """
        Inicializa la clase para generar informes con tablas completas
        """
        self.data_path = data_path
        self.df = None
        self.load_data()
        self.clean_team_names()
        # Cargar datos Opta para posiciones mejoradas
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

        # Mapeo de posiciones actualizado (mantener como fallback)
        self.demarcacion_to_position = {
            # Portero
            'Portero': 'PORTERO',
            
            # Defensas
            'Defensa - Central Derecho': 'CENTRAL_DERECHO',
            'Defensa - Lateral Derecho': 'LATERAL_DERECHO', 
            'Defensa - Central Izquierdo': 'CENTRAL_IZQUIERDO',
            'Defensa - Lateral Izquierdo': 'LATERAL_IZQUIERDO',
            
            # Mediocampo
            'Centrocampista - MC Box to Box': 'MC_BOX_TO_BOX',
            'Centrocampista - MC Organizador': 'MC_ORGANIZADOR',
            'Centrocampista - MC Posicional': 'MC_POSICIONAL',
            'Centrocampista de ataque - Banda Derecha': 'BANDA_DERECHA',
            'Centrocampista de ataque - Banda Izquierda': 'BANDA_IZQUIERDA',
            'Centrocampista de ataque - Mediapunta': 'MC_BOX_TO_BOX',

            # Delanteros
            'Delantero - Delantero Centro': 'DELANTERO_CENTRO',
            'Delantero - Segundo Delantero': 'DELANTERO_CENTRO',
            
            # Sin posición
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
                'MC_POSICIONAL': (33, 40),        # Mediocampo defensivo (centro)
                'MC_BOX_TO_BOX': (62, 55),        # Box to box (centro-arriba)
                'MC_ORGANIZADOR': (45, 40),       # Organizador (centro-abajo)
                'BANDA_DERECHA': (80, 12),        # Banda derecha (extremo arriba)
                'BANDA_IZQUIERDA': (80, 68),      # Banda izquierda (extremo abajo)
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
                'MC_POSICIONAL': (88, 40),        # Mediocampo defensivo (centro)
                'MC_BOX_TO_BOX': (60, 25),        # Box to box (centro-abajo - espejo)
                'MC_ORGANIZADOR': (67, 40),       # Organizador (centro-arriba - espejo)
                'BANDA_DERECHA': (45, 68),        # Banda derecha (extremo abajo - espejo)
                'BANDA_IZQUIERDA': (45, 12),      # Banda izquierda (extremo arriba - espejo)
                'DELANTERO_CENTRO': (38, 25),     # Delantero centro (abajo - espejo)
                'SEGUNDO_DELANTERO': (41, 53),    # Segundo delantero (arriba - espejo)
            }
        }
        
        # Métricas principales para mostrar en las tablas
        self.metricas_principales = [
            'Distancia Total',
            'Distancia Total / min',
            'Distancia Total 14-21 km / h',
            'Distancia Total >21 km / h',
            'Velocidad Máxima Total'
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
    
    def filter_and_accumulate_data(self, equipo, jornadas, min_avg_minutes=70):
        """Filtra por promedio de minutos y acumula datos por jugador"""
        if self.df is None:
            return None
        
        # Normalizar jornadas (código existente...)
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
        
        # ✅ NUEVO: HACER MATCHING DE OPTA ANTES DE ACUMULAR
        filtered_df['Opta_Position'] = None  # Nueva columna
        
        for idx, row in filtered_df.iterrows():
            player_alias = row.get('Alias', '')
            team_name = row.get('Equipo', '')
            player_dorsal = row.get('Dorsal')
            jornada = row.get('Jornada')  # ✅ Aquí SÍ tenemos jornada
            
            # Buscar en Opta con jornada específica
            opta_position = self.find_improved_position(player_alias, team_name, player_dorsal, jornada)
            if opta_position:
                filtered_df.loc[idx, 'Opta_Position'] = opta_position
        
        # Verificar si Alias está vacío y usar Nombre en su lugar
        if 'Nombre' in filtered_df.columns:
            mask_empty_alias = filtered_df['Alias'].isna() | (filtered_df['Alias'] == '') | (filtered_df['Alias'].str.strip() == '')
            filtered_df.loc[mask_empty_alias, 'Alias'] = filtered_df.loc[mask_empty_alias, 'Nombre']
        
        if 'Minutos jugados' not in filtered_df.columns:
            print("⚠️  Columna 'Minutos jugados' no encontrada.")
            return None
        
        # Agrupar por jugador y calcular estadísticas acumuladas
        
        accumulated_data = []
        
        for jugador in filtered_df['Alias'].unique():
            jugador_data = filtered_df[filtered_df['Alias'] == jugador]

            # NUEVO: Filtrar solo partidos donde jugó 70+ minutos
            jugador_data_filtered = jugador_data[jugador_data['Minutos jugados'] >= min_avg_minutes]
            
            # Solo incluir jugadores que tengan al menos 1 partido con 70+ minutos
            if len(jugador_data_filtered) > 0:
                # Tomar datos básicos del jugador (usar el registro más reciente)
                latest_record = jugador_data_filtered.iloc[-1]

                # ✅ DETERMINAR POSICIÓN FINAL (Opta primero, MediaCoach fallback)
                opta_positions = jugador_data_filtered['Opta_Position'].dropna()
                if len(opta_positions) > 0:
                    # Usar la posición Opta más frecuente
                    final_position = opta_positions.mode().iloc[0]
                    position_source = "OPTA"
                else:
                    # Fallback a MediaCoach
                    demarcacion = jugador_data_filtered['Demarcacion'].mode().iloc[0] if len(jugador_data_filtered['Demarcacion'].mode()) > 0 else latest_record['Demarcacion']

                    # Intentar mapear directamente
                    if demarcacion and demarcacion.strip() != '':
                        final_position = self.demarcacion_to_position.get(demarcacion, None)
                        
                        if final_position:
                            position_source = "MediaCoach"
                        else:
                            # Si no está mapeado, usar default
                            final_position = 'MC_BOX_TO_BOX'
                            position_source = "Default"
                    else:
                        final_position = 'MC_BOX_TO_BOX'
                        position_source = "Default"

                # Crear registro acumulado
                accumulated_record = {
                    'Id Jugador': latest_record['Id Jugador'],
                    'Dorsal': latest_record['Dorsal'],
                    'Nombre': latest_record['Nombre'],
                    'Alias': latest_record['Alias'],
                    'Demarcacion': latest_record['Demarcacion'],
                    'Final_Position': final_position,  # ✅ Nueva columna
                    'Position_Source': position_source,  # ✅ Para debug
                    'Equipo': latest_record['Equipo'],
                    
                    # ... resto de métricas igual ...
                    'Minutos jugados': jugador_data_filtered['Minutos jugados'].mean(),
                    'Distancia Total': jugador_data_filtered['Distancia Total'].sum(),
                    'Distancia Total 14-21 km / h': jugador_data_filtered['Distancia Total 14-21 km / h'].sum(),
                    'Distancia Total >21 km / h': jugador_data_filtered['Distancia Total >21 km / h'].sum(),
                    'Distancia Total / min': jugador_data_filtered['Distancia Total / min'].mean(),
                    'Distancia Total 14-21 km / h / min': jugador_data_filtered.get('Distancia Total 14-21 km / h / min', pd.Series([0])).mean(),
                    'Distancia Total >21 km / h / min': jugador_data_filtered.get('Distancia Total >21 km / h / min', pd.Series([0])).mean(),
                    'Velocidad Máxima Total': jugador_data_filtered['Velocidad Máxima Total'].max(),
                    'Velocidad Máxima 1P': jugador_data_filtered.get('Velocidad Máxima 1P', pd.Series([0])).max(),
                    'Velocidad Máxima 2P': jugador_data_filtered.get('Velocidad Máxima 2P', pd.Series([0])).max(),
                }
                
                accumulated_data.append(accumulated_record)
        
        # Convertir a DataFrame
        if accumulated_data:
            result_df = pd.DataFrame(accumulated_data)
            return result_df
        else:
            print(f"❌ No hay jugadores con al menos 1 partido de {min_avg_minutes}+ minutos para {equipo}")
            return None
    @staticmethod
    def normalize_text(text):
        """Normaliza texto eliminando acentos, espacios extra y caracteres especiales"""
        # Eliminar acentos
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        # Convertir a minúsculas y limpiar
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text

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
            equipo_norm.replace('club ', '').replace('cf ', '').replace('fc ', ''),
            # Solo la primera palabra
            equipo_norm.split()[0] if ' ' in equipo_norm else equipo_norm,
            # Solo la última palabra
            equipo_norm.split()[-1] if ' ' in equipo_norm else equipo_norm
        ]
        
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
                pass
                return plt.imread(logo_path)
            except:
                pass
        
        # Búsqueda por contención (archivo que contenga parte del nombre)
        for filename in available_files:
            file_norm = self.normalize_text(filename.split('.')[0])
            for name in possible_names:
                if len(name) > 3 and (name in file_norm or file_norm in name):
                    logo_path = f"{escudos_dir}/{filename}"
                    try:
                        pass
                        return plt.imread(logo_path)
                    except:
                        continue
        
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
        """Agrupa jugadores por posiciones ya determinadas"""
        # Verificar si Alias está vacío y usar Nombre en su lugar
        if 'Nombre' in filtered_df.columns:
            mask_empty_alias = filtered_df['Alias'].isna() | (filtered_df['Alias'] == '') | (filtered_df['Alias'].str.strip() == '')
            filtered_df.loc[mask_empty_alias, 'Alias'] = filtered_df.loc[mask_empty_alias, 'Nombre']
        
        # Ordenar jugadores por minutos jugados (descendente)
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
            'SEGUNDO_DELANTERO': [],
        }
        
        
        for _, player in filtered_df_sorted.iterrows():
            player_dict = player.to_dict()
            player_alias = player_dict.get('Alias', '')
            
            # ✅ USAR POSICIÓN YA DETERMINADA
            position = player_dict.get('Final_Position', 'MC_BOX_TO_BOX')
            source = player_dict.get('Position_Source', 'Unknown')
            
            
            # Agrupar por posiciones específicas
            if position in grouped_players:
                grouped_players[position].append(player_dict)
        
        # 🔥 BALANCEO DE CENTRALES MEJORADO
        centrales_derecho = grouped_players['CENTRAL_DERECHO']
        centrales_izquierdo = grouped_players['CENTRAL_IZQUIERDO']
        
        # Si hay desbalance significativo (diferencia > 1)
        if abs(len(centrales_derecho) - len(centrales_izquierdo)) > 1:
            if len(centrales_derecho) > len(centrales_izquierdo) + 1:
                # Mover el que menos minutos tenga a izquierdo
                jugador_a_mover = min(centrales_derecho, key=lambda x: x['Minutos jugados'])
                grouped_players['CENTRAL_IZQUIERDO'].append(jugador_a_mover)
                grouped_players['CENTRAL_DERECHO'].remove(jugador_a_mover)
            
            elif len(centrales_izquierdo) > len(centrales_derecho) + 1:
                # Mover el que menos minutos tenga a derecho
                jugador_a_mover = min(centrales_izquierdo, key=lambda x: x['Minutos jugados'])
                grouped_players['CENTRAL_DERECHO'].append(jugador_a_mover)
                grouped_players['CENTRAL_IZQUIERDO'].remove(jugador_a_mover)
        
        # 🔥 DIVISIÓN DE DELANTEROS MEJORADA
        delanteros = grouped_players['DELANTERO_CENTRO']
        
        if len(delanteros) > 2:  # Si hay más de 2 delanteros
            # Dividir en dos grupos equilibrados
            mitad = len(delanteros) // 2
            
            # Primer grupo (más minutos) se queda en DELANTERO_CENTRO
            primer_grupo = delanteros[:mitad]
            # Segundo grupo va a SEGUNDO_DELANTERO
            segundo_grupo = delanteros[mitad:]
            
            grouped_players['DELANTERO_CENTRO'] = primer_grupo
            grouped_players['SEGUNDO_DELANTERO'] = segundo_grupo
            
        else:
            grouped_players['SEGUNDO_DELANTERO'] = []
        
        # Limitar máximo 3 jugadores por posición
        for posicion in grouped_players:
            grouped_players[posicion] = grouped_players[posicion][:3]
        
        return grouped_players

    def is_dominant_position(self, player_dict, target_position):
        """Verifica si un jugador es dominante en una posición específica"""
        player_id = player_dict.get('Id Jugador')
        
        if not player_id or self.df is None:
            return False
        
        # Buscar todas las demarcaciones del jugador en MediaCoach
        player_positions = self.df[
            (self.df['Id Jugador'] == player_id) & 
            (self.df['Demarcacion'].notna()) & 
            (self.df['Demarcacion'] != '') &
            (self.df['Demarcacion'].str.strip() != '')
        ]
        
        if len(player_positions) == 0:
            return False
        
        # Mapear posiciones MediaCoach a sistema
        target_demarcaciones = []
        for demarcacion, position in self.demarcacion_to_position.items():
            if position == target_position:
                target_demarcaciones.append(demarcacion)
        
        if not target_demarcaciones:
            return False
        
        # Calcular minutos jugados en la posición target
        target_minutes = player_positions[
            player_positions['Demarcacion'].isin(target_demarcaciones)
        ]['Minutos jugados'].sum()
        
        total_minutes = player_positions['Minutos jugados'].sum()
        
        # Es dominante si ha jugado >30% de sus minutos en esa posición
        dominance_ratio = target_minutes / total_minutes if total_minutes > 0 else 0
        
        
        return dominance_ratio > 0.3  # 30% de sus minutos
    
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
        metric_col_width = 4   # Ancho de la columna de métricas
        player_col_width = 6   # Ancho por columna de jugador
        table_width = metric_col_width + (num_players * player_col_width)
        
        header_height = 2      # Altura del header de demarcación
        names_height = 3       # Altura de la fila de nombres
        metric_row_height = 1.5  # Altura por fila de métrica
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
                fontsize=6, weight='bold', color=team_colors['text'],
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
                zoom_factor = min(metric_col_width / 100, names_height / 100) * 3.1
                
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
            dorsal = player.get('Dorsal', 'N/A')
            
            # FONDO INDIVIDUAL para cada jugador con color del equipo
            player_bg = plt.Rectangle((player_x - player_col_width/2, names_y - names_height/2), 
                                    player_col_width, names_height,
                                    facecolor=team_colors['primary'], alpha=0.9,  # Más opaco
                                    edgecolor='white', linewidth=1)
            ax.add_patch(player_bg)
            
            # Color de texto contrastante
            text_color = self.get_contrasting_color(team_colors['primary'])
            
            # Número del dorsal con color contrastante
            if pd.notna(dorsal) and dorsal != 'N/A':
                try:
                    dorsal_text = str(int(float(dorsal)))
                except (ValueError, TypeError):
                    dorsal_text = str(dorsal)
            else:
                dorsal_text = 'S/N'  # Sin número

            ax.text(player_x, names_y + 0.6, dorsal_text, 
                    fontsize=8, weight='bold', color=text_color,
                    ha='center', va='center') 
            
            # Dividir nombres largos en dos líneas
            if len(player_name) > 8:  # Umbral más estricto
                words = player_name.split()
                if len(words) > 1:
                    # Dividir en dos líneas por palabras
                    mid = len(words) // 2
                    line1 = ' '.join(words[:mid])
                    line2 = ' '.join(words[mid:])
                    
                    # Truncar líneas si siguen siendo muy largas
                    if len(line1) > 10:
                        line1 = line1[:10] + "."
                    if len(line2) > 10:
                        line2 = line2[:10] + "."
                else:
                    # Si es una sola palabra muy larga, cortarla
                    line1 = player_name[:8]
                    line2 = player_name[8:16] + ("." if len(player_name) > 16 else "")
                
                # Mostrar en dos líneas con fuente más pequeña
                ax.text(player_x, names_y - 0.3, line1, 
                        fontsize=5, weight='bold', color=text_color,
                        ha='center', va='center')
                ax.text(player_x, names_y - 0.9, line2, 
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
            
            # Nombre de la métrica - PRIMERO crear metric_name
            metric_name = metric.replace('Distancia Total ', '').replace('Velocidad Máxima Total', 'V.Max').replace('Distancia Total', 'Dist.').replace('14-21 km / h', 'H.Int').replace('>21 km / h', 'H.Alt').replace('/ min', '/min')

            # Dividir texto largo en múltiples líneas si es necesario
            if len(metric_name) > 8:
                # Partir por espacios o caracteres especiales
                words = metric_name.replace('/', '/\n').replace(' ', '\n').split('\n')
                if len(words) > 1:
                    metric_name = '\n'.join(words)

            ax.text(x - table_width/2 + metric_col_width/2, metric_y, metric_name, 
                    fontsize=5.5, weight='bold', color='white',
                    ha='center', va='center')
            
            # Valores para cada jugador
            for j, player in enumerate(players_list):
                player_x = x - table_width/2 + metric_col_width + (j * player_col_width) + player_col_width/2
                
                if metric in player:
                    value = player[metric]
                    if 'Velocidad' in metric:
                        formatted_value = f"{value:.1f}"
                    elif 'min' in metric:
                        formatted_value = f"{value:.0f}"
                    else:
                        formatted_value = f"{value:.0f}"
                else:
                    formatted_value = "N/A"
                
                # Destacar valores altos con color diferente
                text_color = 'white'  # Todos los jugadores en blanco
                
                ax.text(player_x, metric_y, formatted_value, 
                        fontsize=7, weight='bold', color=text_color,  # Reducido de 8 a 6
                        ha='center', va='center')
        
        # 🔹 LÍNEAS SEPARADORAS ELEGANTES (MÁS FINAS)
        # Línea horizontal debajo de nombres
        ax.plot([x - table_width/2 + metric_col_width, x + table_width/2], 
                [names_y - names_height/2, names_y - names_height/2], 
                color='white', linewidth=1.5, alpha=0.8)  # Reducido de 2 a 1.5
    
    def create_team_summary_table(self, team_data, ax, x_pos, y_pos, team_name, team_colors, team_logo=None):
        """Crea una tabla de resumen del equipo con métricas en fila 1 y valores en fila 2"""
        
        # Calcular estadísticas del equipo
        summary_stats = {}
        
        for metric in self.metricas_principales:
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
            
            # Nombre de la métrica
            metric_short = metric.replace('Distancia Total ', 'Dist. ').replace('Velocidad Máxima Total', 'V.Max').replace('Distancia Total', 'Distancia')
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
                    fontsize=7, weight='bold', color='#FFD700',  # Valores en dorado
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
        """Crea la visualización completa con tablas por posición y datos acumulados"""
        
        # Crear campo SIN espacios
        fig, ax = self.create_campo_sin_espacios(figsize)
        
        # Título superpuesto en el campo
        ax.text(60, 78, f'DATOS ACUMULADOS - ÚLTIMAS {len(jornadas)} JORNADAS | PROMEDIO 70+ MIN', 
                fontsize=14, weight='bold', color='white', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.8", facecolor='#1e3d59', alpha=0.95,
                         edgecolor='white', linewidth=2))
        
        # Obtener datos acumulados de ambos equipos
        villarreal_data = self.filter_and_accumulate_data('Villarreal CF', jornadas)
        rival_data = self.filter_and_accumulate_data(equipo_rival, jornadas)
        
        if villarreal_data is None or len(villarreal_data) == 0:
            print("❌ No hay jugadores de Villarreal CF con promedio 70+ minutos")
            return None
            
        if rival_data is None or len(rival_data) == 0:
            print(f"❌ No hay jugadores de {equipo_rival} con promedio 70+ minutos")
            return None
        
        # Cargar escudos
        villarreal_logo = self.load_team_logo('Villarreal CF')
        rival_logo = self.load_team_logo(equipo_rival)
        
        # Posicionar escudos dentro del campo
        if villarreal_logo is not None:
            imagebox = OffsetImage(villarreal_logo, zoom=0.4)
            ab = AnnotationBbox(imagebox, (5, 5), frameon=False)
            ax.add_artist(ab)

        if rival_logo is not None:
            imagebox = OffsetImage(rival_logo, zoom=0.4)
            ab = AnnotationBbox(imagebox, (115, 5), frameon=False)
            ax.add_artist(ab)
        
        # Agrupar jugadores por posiciones específicas CON NUEVA LÓGICA
        villarreal_grouped = self.group_players_by_specific_position(villarreal_data)
        rival_grouped = self.group_players_by_specific_position(rival_data)

        

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
        
        # Resúmenes de equipos con colores personalizados
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

def seleccionar_equipo_jornadas_campo():
    """Permite al usuario seleccionar un equipo rival y jornadas"""
    try:
        report_generator = CampoFutbolAcumulado()
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

def main_campo_futbol():
    """Función principal para generar el informe con posiciones mejoradas"""
    try:
        pass
        
        # Selección interactiva
        equipo_rival, jornadas = seleccionar_equipo_jornadas_campo()
        
        if equipo_rival is None or jornadas is None:
            print("❌ No se pudo completar la selección.")
            return
        
        
        # Crear el reporte
        report_generator = CampoFutbolAcumulado()
        fig = report_generator.create_visualization(equipo_rival, jornadas)
        
        if fig:
            # Mostrar en pantalla
            plt.show()
            
            # Guardar
            equipo_filename = equipo_rival.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_MEJORADO_Villarreal_vs_{equipo_filename}.pdf"
            
            report_generator.guardar_sin_espacios(fig, output_path)
            
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()

def generar_reporte_campo_personalizado(equipo_rival, jornadas, mostrar=True, guardar=True):
    """Función para generar un reporte personalizado con posiciones mejoradas"""
    try:
        report_generator = CampoFutbolAcumulado()
        fig = report_generator.create_visualization(equipo_rival, jornadas)
        
        if fig:
            if mostrar:
                plt.show()
            
            if guardar:
                equipo_filename = equipo_rival.replace(' ', '_').replace('/', '_')
                output_path = f"reporte_MEJORADO_Villarreal_vs_{equipo_filename}.pdf"
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
    report_generator = CampoFutbolAcumulado()
    equipos = report_generator.get_available_teams()
    
    if len(equipos) > 0:
        pass
    
except Exception as e:
    print(f"❌ Error al inicializar: {e}")

if __name__ == "__main__":
    main_campo_futbol()