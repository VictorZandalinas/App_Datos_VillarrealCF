import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import patheffects
import numpy as np
import os
from mplsoccer import VerticalPitch, Pitch # Añadido Pitch para el campo horizontal
from difflib import SequenceMatcher
from collections import defaultdict, Counter # Añadido Counter
import warnings
import json
import base64
import re
import unicodedata  
from io import BytesIO
from PIL import Image

warnings.filterwarnings('ignore')

class RedPasesEquipo:
    # 🔥 CACHÉ DE DATOS: Compartido entre todas las instancias para evitar cargas repetidas
    _open_play_cache = None
    _team_stats_cache = None
    _player_stats_cache = None
    _match_events_cache = None  # 🔥 NUEVO CACHÉ

    @classmethod
    def _get_open_play_data(cls, columns=None):
        """Carga open_play_events.parquet una sola vez y lo cachea."""
        if cls._open_play_cache is None:
            print("📥 [CACHÉ] Cargando open_play_events.parquet por primera vez...")
            cls._open_play_cache = pd.read_parquet("extraccion_opta/datos_opta_parquet/open_play_events.parquet")
        if columns:
            return cls._open_play_cache[columns].copy()
        return cls._open_play_cache.copy()

    @classmethod
    def _get_match_events_data(cls, columns=None):
        """Carga match_events.parquet una sola vez y lo cachea."""
        if cls._match_events_cache is None:
            print("📥 [CACHÉ] Cargando match_events.parquet por primera vez...")
            cls._match_events_cache = pd.read_parquet("extraccion_opta/datos_opta_parquet/match_events.parquet")
        if columns:
            return cls._match_events_cache[columns].copy()
        return cls._match_events_cache.copy()

    @classmethod
    def clear_cache(cls):
        """Limpia todos los cachés de clase para liberar memoria."""
        print("🧹 [CACHÉ] Limpiando cachés de RedPasesEquipo...")
        cls._open_play_cache = None
        cls._team_stats_cache = None
        cls._player_stats_cache = None
        cls._match_events_cache = None

    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/open_play_events.parquet", team_filter=None):
        self.data_path = data_path
        self.team_filter = team_filter
        self.df = None
        self.passes_data = pd.DataFrame()

        # Usar caché para team_stats y player_stats (optimizado con columnas necesarias)
        if RedPasesEquipo._team_stats_cache is None:
            RedPasesEquipo._team_stats_cache = pd.read_parquet(
                "extraccion_opta/datos_opta_parquet/team_stats.parquet",
                columns=['Team Name', 'Match ID', 'Team ID', 'Week']
            )
        if RedPasesEquipo._player_stats_cache is None:
            RedPasesEquipo._player_stats_cache = pd.read_parquet(
                "extraccion_opta/datos_opta_parquet/player_stats.parquet",
                columns=['Player ID', 'Player Name', 'Team Name', 'Week']
            )

        self.team_stats = RedPasesEquipo._team_stats_cache
        self.player_stats = RedPasesEquipo._player_stats_cache
        self.player_stats_df = RedPasesEquipo._player_stats_cache

        self.load_data(team_filter)
        self.events_df = None
        self.load_match_events()
        
        self.photos_data = None
        
        self.formation_mapping = {
            1: "not_in_use", 2: "442", 3: "41212", 4: "433", 5: "451",
            6: "4411", 7: "4141", 8: "4231", 9: "4321", 10: "532",
            11: "541", 12: "352", 13: "343", 15: "4222", 16: "3511",
            17: "3421", 18: "3412", 19: "3142", 20: "343d", 21: "4132",
            22: "4240", 23: "4312", 24: "3241", 25: "3331"
        }

        # Mapeo de demarcaciones por formación (posición 1-11 -> demarcación abreviada)
        self.formation_demarcations = {
            2: {  # 442
                1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI',
                4: 'MCD', 8: 'MCI', 7: 'ED', 11: 'EI', 10: 'DC', 9: 'SD'
            },
            3: {  # 41212 (Diamond)
                1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI',
                4: 'MCD', 7: 'ED', 11: 'EI', 8: 'MP', 10: 'SD', 9: 'DC'
            },
            4: {  # 433
                1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI',
                4: 'MCD', 7: 'ED', 8: 'EI', 10: 'EI', 9: 'DC', 11: 'ED'
            },
            5: {  # 451
                1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI',
                4: 'MC', 10: 'MP', 8: 'MCI', 7: 'ED', 11: 'EI', 9: 'DC'
            },
            6: {  # 4411
                1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI',
                4: 'MCD', 8: 'MCI', 7: 'ED', 11: 'EI', 10: 'MP', 9: 'DC'
            },
            7: {  # 4141
                1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI',
                4: 'MCD', 7: 'ED', 8: 'MCD', 10: 'MCI', 11: 'EI', 9: 'DC'
            },
            8: {  # 4231
                1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI',
                4: 'MCD', 8: 'MCI', 7: 'ED', 10: 'MP', 11: 'EI', 9: 'DC'
            },
            9: {  # 4321
                1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI',
                4: 'MCD', 7: 'MC', 8: 'MCI', 10: 'SD', 11: 'SD', 9: 'DC'
            },
            10: {  # 532
                1: 'POR', 2: 'LD', 4: 'CC', 6: 'CD', 5: 'CI', 3: 'LI',
                7: 'MCD', 8: 'MCI', 10: 'MP', 11: 'DC', 9: 'SD'
            },
            11: {  # 541
                1: 'POR', 2: 'LD', 5: 'CC', 6: 'CD', 4: 'CI', 3: 'LI',
                7: 'ED', 8: 'MCD', 10: 'MCI', 11: 'EI', 9: 'DC'
            },
            12: {  # 352
                1: 'POR', 2: 'LD', 5: 'CC', 6: 'CD', 4: 'CI', 3: 'LI',
                7: 'MCD', 11: 'MP', 8: 'MCI', 10: 'DC', 9: 'SD'
            },
            13: {  # 343
                1: 'POR', 2: 'LD', 6: 'CC', 5: 'CD', 4: 'CI', 3: 'LI',
                7: 'MCD', 8: 'MCI', 10: 'EI', 9: 'DC', 11: 'ED'
            },
            15: {  # 4222
                1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI',
                4: 'MCD', 8: 'MCI', 7: 'ED', 11: 'EI', 10: 'SD', 9: 'DC'
            },
            16: {  # 3511
                1: 'POR', 2: 'LD', 5: 'CC', 6: 'CD', 4: 'CI', 3: 'LI',
                7: 'MCD', 11: 'MC', 8: 'MCI', 10: 'MP', 9: 'DC'
            },
            17: {  # 3421
                1: 'POR', 2: 'LD', 5: 'CC', 6: 'CD', 4: 'CI', 3: 'LI',
                7: 'MCD', 8: 'MCI', 10: 'SD', 11: 'SD', 9: 'DC'
            },
            18: {  # 3412
                1: 'POR', 2: 'LD', 5: 'CC', 6: 'CD', 4: 'CI', 3: 'LI',
                7: 'MCD', 8: 'MCI', 9: 'MP', 10: 'DC', 11: 'SD'
            },
            19: {  # 3142
                1: 'POR', 2: 'LD', 5: 'CC', 4: 'CD', 6: 'CI', 3: 'LI',
                8: 'MCD', 7: 'MCD', 11: 'MCI', 9: 'DC', 10: 'SD'
            },
            20: {  # 343d (Diamond)
                1: 'POR', 2: 'LD', 5: 'CC', 6: 'CD', 4: 'CI', 3: 'LI',
                8: 'MCD', 7: 'MP', 10: 'EI', 11: 'ED', 9: 'DC'
            },
            21: {  # 4132
                1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI',
                4: 'MCD', 7: 'MCD', 10: 'MP', 8: 'MCI', 9: 'DC', 11: 'SD'
            },
            22: {  # 4240
                1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI',
                4: 'MCD', 8: 'MCI', 7: 'ED', 11: 'EI', 9: 'DC', 10: 'SD'
            },
            23: {  # 4312
                1: 'POR', 2: 'LD', 5: 'CD', 6: 'CI', 3: 'LI',
                7: 'MCD', 4: 'MC', 8: 'MCI', 9: 'MP', 10: 'DC', 11: 'SD'
            },
            24: {  # 3241
                1: 'POR', 2: 'CD', 3: 'CC', 4: 'CI', 5: 'LD', 6: 'LI',
                10: 'ED', 11: 'EI', 7: 'MCD', 8: 'MCI', 9: 'DC'
            },
            25: {  # 3331
                1: 'POR', 2: 'CD', 3: 'CC', 4: 'CI', 5: 'LD', 6: 'LI',
                7: 'MCD', 8: 'MCD', 11: 'MCI', 10: 'MP', 9: 'DC'
            }
        }

        # Leyenda de abreviaciones
        self.demarcation_labels = {
            'POR': 'Portero',
            'LD': 'Lateral/Carrilero Der',
            'LI': 'Lateral/Carrilero Izq',
            'CD': 'Central Derecho',
            'CI': 'Central Izquierdo',
            'CC': 'Central Centro',
            'MCD': 'Mediocentro Defensivo',
            'MCI': 'Mediocentro Izquierdo',
            'MC': 'Mediocampista Centro',
            'MP': 'Mediapunta',
            'EI': 'Extremo Izquierdo',
            'ED': 'Extremo Derecho',
            'DC': 'Delantero Centro',
            'SD': 'Segundo Delantero'
        }
        if team_filter:
            self.extract_passes(team_filter)

    # --- VERSIÓN CORRECTA Y ÚNICA DE CREATE_RANKING_PANEL ---
    def create_ranking_panel(self, ax, ranking_data, title, total_label, panel_color, equipo, metric_label="por partido"):
        """
        Versión ajustada para el nuevo layout
        """
        ax.set_facecolor('#f0f0f0')
        ax.axis('off')

        # 🔥 Título más pequeño
        ax.text(0.5, 0.98, title.upper(), ha='center', va='top', fontsize=10,
                fontweight='bold', color='#1e3d59')

        if not ranking_data:
            ax.text(0.5, 0.5, 'Datos insuficientes', ha='center', va='center',
                    fontsize=9, color='grey')
            return

        self.load_player_photos()
        y_pos = 0.80  # 🔥 Ajustado
        
        for i, player in enumerate(ranking_data):
            rect_bg = patches.FancyBboxPatch((0.05, y_pos - 0.12), 0.9, 0.24,
                                boxstyle="round,pad=0.02", facecolor='white',
                                edgecolor='lightgrey', linewidth=1)
            ax.add_patch(rect_bg)
            
            dorsal = str(int(player.get('Shirt Number', 0)))
            
            circle = plt.Circle((0.12, y_pos), 0.04, color=panel_color, alpha=0.9, zorder=10)
            ax.add_patch(circle)
            
            circle_border = plt.Circle((0.12, y_pos), 0.04, fill=False, 
                                    edgecolor='white', linewidth=2, zorder=11)
            ax.add_patch(circle_border)
            
            ax.text(0.12, y_pos, dorsal, ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white', zorder=12)
            
            player_photo = self.get_player_photo(player['Match Name'])
            if player_photo is not None:
                photo_ax = ax.inset_axes([0.08, y_pos - 0.11, 0.30, 0.26])
                photo_ax.imshow(player_photo)
                photo_ax.axis('off')

            apellido = player.get('Match Name', 'N/A').split()[-1]
            
            ax.text(0.32, y_pos + 0.02, f"{apellido.upper()}", ha='left', va='center',
                    fontsize=9, fontweight='bold', color='#2c3e50')
            
            avg_str = f"{player['average_per_match']:.1f}"
            
            ax.text(0.65, y_pos + 0.03, avg_str, ha='center', va='center', fontsize=12,
                    fontweight='bold', color=panel_color)
            ax.text(0.65, y_pos - 0.05, metric_label, ha='center', va='center',
                    fontsize=7, color='grey')
            
            stars = 4 - i
            star_text = '★' * stars + '☆' * (4 - stars)
            ax.text(0.88, y_pos, star_text, ha='center', va='center', fontsize=11, color='#f39c12')
            
            y_pos -= 0.26  # 🔥 Espacio entre jugadores

    def get_team_colors(self, equipo):
        team_colors_map = {
            'Real Madrid': {'primary': '#FFD700'}, 
            'FC Barcelona': {'primary': '#004D98'},
            'Villarreal CF': {'primary': '#FFD700'},
        }
        default_colors = {'primary': '#2c3e50'}
        
        for team_name, colors in team_colors_map.items():
            if self.are_teams_equivalent(equipo, team_name):
                return colors
        return default_colors
    
    def load_match_events(self):
        try:
            # 🔥 Usar caché en lugar de leer desde disco
            self.events_df = self._get_match_events_data()
            if 'Week' in self.events_df.columns:
                self.events_df['Week'] = self.events_df['Week'].astype(str)
            return True
        except Exception as e:
            print(f"⚠️ Error al cargar eventos: {e}")
            self.events_df = None
            return False
    
    def get_team_setup_from_events(self, team_name, week):
        if self.events_df is None: return None
        week_str = str(week)
        setup_events = self.events_df[(self.events_df['Event Name'] == 'Team set up') & (self.events_df['Week'] == week_str)]
        if setup_events.empty: return None
        team_setup = None
        for _, row in setup_events.iterrows():
            opta_team = str(row.get('Team Name', ''))
            if self.are_teams_equivalent(team_name, opta_team):
                team_setup = row
                break
        if team_setup is None: return None
        formation_number = team_setup.get('Team Formation')
        jersey_numbers_str = str(team_setup.get('Jersey Number', ''))
        player_formation_str = str(team_setup.get('Team Player Formation', ''))
        try:
            jersey_list = [int(j.strip()) for j in jersey_numbers_str.split(',') if j.strip().isdigit()]
            formation_slots = [int(s.strip()) for s in player_formation_str.split(',') if s.strip().isdigit()]
            if len(jersey_list) != len(formation_slots): return None
            ordered_starters = [None] * 11 
            for i in range(len(formation_slots)):
                slot_position = formation_slots[i]
                if 1 <= slot_position <= 11:
                    jersey_number = jersey_list[i]
                    ordered_starters[slot_position - 1] = jersey_number
            if None in ordered_starters: return None
            formation_name = self.formation_mapping.get(int(formation_number), f"Unknown_{formation_number}")
            return {'formation_number': int(formation_number), 'formation_name': formation_name, 'starters': ordered_starters}
        except Exception as e:
            pass
            return None

    # --- VERSIÓN CORRECTA Y ÚNICA DE GET_PASS_RANKING ---
    def get_pass_ranking_with_efficiency(self, success_col, total_col, min_total_passes=50):
        """
        Ranking que combina VOLUMEN de pases exitosos y EFECTIVIDAD (%).
        Score final = (volumen * efectividad) para valorar ambos factores.
        """
        if self.player_stats is None or self.team_filter is None:
            return []

        team_players_df = self.player_stats[
            self.player_stats['Team Name'] == self.team_filter
        ].copy()

        if team_players_df.empty:
            print(f"⚠️ No se encontraron datos para el equipo '{self.team_filter}'")
            return []

        # Agrupar por jugador
        ranking = team_players_df.groupby(['Player ID', 'Match Name', 'Shirt Number']).agg(
            total_success=(success_col, 'sum'),
            total_attempts=(total_col, 'sum'),
            matches_played=('Week', 'nunique')
        ).reset_index()

        # Filtrar jugadores con mínimo de pases
        ranking = ranking[ranking['total_attempts'] >= min_total_passes]
        if ranking.empty:
            return []

        # Calcular promedios por partido
        ranking['success_per_match'] = (ranking['total_success'] / ranking['matches_played']).fillna(0)
        ranking['attempts_per_match'] = (ranking['total_attempts'] / ranking['matches_played']).fillna(0)
        
        # Calcular efectividad (%)
        ranking['efficiency'] = ((ranking['total_success'] / ranking['total_attempts']) * 100).fillna(0)
        
        # 🔥 SCORE COMBINADO: Volumen normalizado (60%) + Efectividad normalizada (40%)
        # Normalizar valores entre 0-1
        max_success = ranking['success_per_match'].max()
        max_efficiency = ranking['efficiency'].max()
        
        ranking['normalized_volume'] = ranking['success_per_match'] / max_success if max_success > 0 else 0
        ranking['normalized_efficiency'] = ranking['efficiency'] / max_efficiency if max_efficiency > 0 else 0
        
        # Score combinado: 60% volumen + 40% efectividad
        ranking['combined_score'] = (ranking['normalized_volume'] * 0.6) + (ranking['normalized_efficiency'] * 0.4)
        
        # Guardar el promedio por partido para mostrar en el panel
        ranking['average_per_match'] = ranking['success_per_match']
        
        return ranking.sort_values('combined_score', ascending=False).head(4).to_dict('records')

    # 🔥 AGREGAR ESTE MÉTODO AQUÍ:
    def get_pass_ranking_per_match(self, pass_cols_to_sum, result_col_name, min_total_attempts=10):
        """
        Función para obtener rankings de pases basados en el PROMEDIO POR PARTIDO.
        """
        if self.player_stats is None or self.team_filter is None:
            return []

        team_players_df = self.player_stats[
            self.player_stats['Team Name'] == self.team_filter
        ].copy()

        if team_players_df.empty:
            print(f"⚠️ No se encontraron datos para el equipo '{self.team_filter}'")
            return []

        team_players_df[result_col_name] = team_players_df[pass_cols_to_sum].sum(axis=1)

        ranking = team_players_df.groupby(['Player ID', 'Match Name', 'Shirt Number']).agg(
            total_passes=(result_col_name, 'sum'),
            matches_played=('Week', 'nunique')
        ).reset_index()

        ranking = ranking[ranking['total_passes'] >= min_total_attempts]
        if ranking.empty:
            return []

        ranking['average_per_match'] = (ranking['total_passes'] / ranking['matches_played']).fillna(0)
        ranking.rename(columns={'total_passes': result_col_name}, inplace=True)
        
        return ranking.sort_values('average_per_match', ascending=False).head(4).to_dict('records')

    def get_most_frequent_formation(self, team_name):
        """Retorna formación más usada y los 2 jugadores más frecuentes por posición"""
        if self.events_df is None: 
            return None
        
        weeks = self.events_df[self.events_df['Team Name'].str.contains(team_name, case=False, na=False)]['Week'].unique()
        formaciones = []
        all_starters_by_position = defaultdict(list)  # {posicion: [dorsales]}
        
        for week in weeks:
            setup = self.get_team_setup_from_events(team_name, week)
            if setup:
                formaciones.append(setup['formation_name'])
                # Guardar qué jugador jugó en qué posición
                for pos_idx, dorsal in enumerate(setup['starters']):
                    all_starters_by_position[pos_idx].append(dorsal)
        
        if not formaciones: 
            return None
        
        formacion_ganadora = Counter(formaciones).most_common(1)[0][0]
        
        # Para cada posición, obtener los 2 jugadores más frecuentes
        top_players_by_position = {}
        for pos_idx in range(11):
            if pos_idx in all_starters_by_position:
                dorsales_en_pos = all_starters_by_position[pos_idx]
                top_2 = Counter(dorsales_en_pos).most_common(2)
                
                if len(top_2) >= 2:
                    top_players_by_position[pos_idx] = {
                        'primary': top_2[0][0],
                        'secondary': top_2[1][0]
                    }
                elif len(top_2) == 1:
                    top_players_by_position[pos_idx] = {
                        'primary': top_2[0][0],
                        'secondary': None
                    }
        
        jugadores_primarios = [top_players_by_position[i]['primary'] for i in range(11) if i in top_players_by_position]
        
        
        return {
            'formation': formacion_ganadora, 
            'starters': jugadores_primarios,
            'top_players_by_position': top_players_by_position
        }
    
    def are_teams_equivalent(self, team1, team2):
        if not team1 or not team2: 
            return False
        
        def normalize(text):
            import unicodedata
            text = unicodedata.normalize('NFD', text)
            text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
            return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).strip()
        
        norm1, norm2 = normalize(team1), normalize(team2)
        if norm1 == norm2: 
            return True
        if ('real' in norm1 and 'atletico' in norm2) or ('atletico' in norm1 and 'real' in norm2): 
            return False
        return SequenceMatcher(None, norm1, norm2).ratio() > 0.85

    def extract_passes(self, team_filter):
        if self.df is None:
            print("❌ No hay datos cargados")
            return
        titular_info = self.get_most_frequent_formation(team_filter)
        if not titular_info:
            print("❌ No se pudo determinar el once titular")
            return
        titulares_dorsales = set(str(d) for d in titular_info['starters'])
        team_data = self.df[self.df['Team Name'] == team_filter].copy()
        for col in ['x', 'y', 'Pass End X', 'Pass End Y']:
            team_data[col] = pd.to_numeric(team_data[col], errors='coerce')
        team_data['shirt_number'] = team_data['playerId'].apply(self.get_player_shirt_number)
        self.passes_data = team_data[
            team_data['shirt_number'].isin(titulares_dorsales) &
            team_data['Pass End X'].notna() &
            team_data['Pass End Y'].notna()
        ].copy()
    
    def load_data(self, team_filter=None):
        try:
            columns_needed = ['Match ID', 'Team ID', 'Team Name', 'Event Name', 'outcome',
                        'x', 'y', 'Pass End X', 'Pass End Y', 'playerName', 'playerId',
                        'timeMin', 'timeSec']
            # 🔥 Usar caché en lugar de cargar desde disco
            self.df = self._get_open_play_data(columns=columns_needed)
            self.df = self.df[(self.df['Event Name'] == 'Pass') & (self.df['outcome'] == 1)]
            if team_filter:
                team_matches = self.team_stats[self.team_stats['Team Name'] == team_filter]['Match ID'].unique()
                self.df = self.df[self.df['Match ID'].isin(team_matches)]
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")
    
    def get_player_shirt_number(self, player_id):
        if pd.isna(player_id): return None
        player_info = self.player_stats[self.player_stats['Player ID'] == player_id]
        if not player_info.empty:
            shirt_number = player_info['Shirt Number'].iloc[0]
            return str(int(shirt_number)) if pd.notna(shirt_number) else None
        return None
    
    def get_player_demarcation(self, player_id, equipo):
        """Obtiene la demarcación más frecuente de un jugador basándose en su historial"""
        if self.events_df is None:
            return None
        
        demarcaciones_encontradas = []
        weeks = self.events_df['Week'].unique()
        
        for week in weeks:
            setup = self.get_team_setup_from_events(equipo, week)
            if not setup:
                continue
            
            # Buscar si este jugador está en el setup
            player_shirt = self.get_player_shirt_number(player_id)
            if not player_shirt:
                continue
                
            for pos_idx, dorsal in enumerate(setup['starters'], 1):
                if str(dorsal) == str(player_shirt):
                    formation_num = setup['formation_number']
                    if formation_num in self.formation_demarcations:
                        demarcacion = self.formation_demarcations[formation_num].get(pos_idx)
                        if demarcacion:
                            demarcaciones_encontradas.append(demarcacion)
                    break
        
        if demarcaciones_encontradas:
            # Devolver la demarcación más frecuente
            return Counter(demarcaciones_encontradas).most_common(1)[0][0]
        
        return 'MC'  # Demarcación por defecto
    
    def calculate_pass_network(self, data=None):
        """Calcula red de pases simple basada en jugadores individuales"""
        if data is None: 
            data = self.passes_data
        if data.empty: 
            return {}, {}
        
        pass_counts = defaultdict(int)
        player_positions = defaultdict(lambda: {'x': [], 'y': [], 'shirt': None, 'name': None})
        
        data_sorted = data.sort_values(['Match ID', 'timeMin', 'timeSec']).reset_index(drop=True)
        
        for idx, row in data_sorted.iterrows():
            passer_id, passer_shirt = row['playerId'], row['shirt_number']
            
            player_positions[passer_id]['x'].append(row['x'])
            player_positions[passer_id]['y'].append(row['y'])
            player_positions[passer_id]['shirt'] = passer_shirt
            player_positions[passer_id]['name'] = row['playerName']
            
            if idx + 1 < len(data_sorted):
                next_pass = data_sorted.iloc[idx + 1]
                if (next_pass['Match ID'] == row['Match ID'] and next_pass['Team Name'] == row['Team Name']):
                    time_diff = (next_pass['timeMin'] * 60 + next_pass['timeSec']) - (row['timeMin'] * 60 + row['timeSec'])
                    if 0 < time_diff <= 10:
                        receiver_id = next_pass['playerId']
                        if receiver_id != passer_id:
                            pass_key = tuple(sorted([passer_id, receiver_id]))
                            pass_counts[pass_key] += 1
        
        avg_positions = {}
        for player_id, p_data in player_positions.items():
            if p_data['x'] and p_data['shirt']:
                avg_positions[player_id] = {
                    'x': np.mean(p_data['x']), 
                    'y': np.mean(p_data['y']), 
                    'shirt': p_data['shirt'], 
                    'name': p_data['name']
                }
        
        return dict(pass_counts), avg_positions

    def calculate_pass_network_by_demarcation(self, data=None):
        """
        Calcula red de pases AGRUPADA POR DEMARCACIÓN.
        🔥 VERSIÓN CORREGIDA: Primero analiza TODAS las demarcaciones, 
        luego selecciona las 11 más usadas.
        """
        # 🔥 NO usar self.passes_data (que ya está filtrado por titulares)
        # En su lugar, cargar TODOS los pases del equipo
        if self.df is None:
            print("❌ No hay datos cargados")
            return {}, {}
        
        # Filtrar pases del equipo (SIN filtrar por titulares)
        team_data = self.df[self.df['Team Name'] == self.team_filter].copy()
        
        if team_data.empty:
            return {}, {}
        
        
        pass_counts_by_demarcation = defaultdict(int)
        demarcation_positions = defaultdict(lambda: {'x': [], 'y': [], 'count': 0, 'dorsales': set()})
        demarcation_players = defaultdict(set)
        
        # Obtener la formación más usada (para tener referencia de demarcaciones)
        titular_info = self.get_most_frequent_formation(self.team_filter)
        player_demarcations_from_formation = {}
        
        if titular_info:
            formation_name = titular_info['formation']
            formation_num = [k for k, v in self.formation_mapping.items() if v == formation_name][0]
            
            if formation_num in self.formation_demarcations:
                for pos_idx, dorsal in enumerate(titular_info['starters'], 1):
                    demarcacion = self.formation_demarcations[formation_num].get(pos_idx, 'MC')
                    player_demarcations_from_formation[str(dorsal)] = demarcacion
        
        # 🔥 PROCESAR CADA PASE Y ASIGNARLE UNA DEMARCACIÓN
        data_sorted = team_data.sort_values(['Match ID', 'timeMin', 'timeSec']).reset_index(drop=True)
        
        # Cache para demarcaciones (evitar calcular múltiples veces)
        player_demarcations_cache = {}
        
        for idx, row in data_sorted.iterrows():
            passer_id = row['playerId']
            passer_name = row['playerName']
            
            # Obtener dorsal
            passer_shirt = self.get_player_shirt_number(passer_id)
            if not passer_shirt:
                continue
            
            # 🔥 DETERMINAR DEMARCACIÓN DEL PASADOR
            passer_demarcation = None
            
            # 1. Intentar desde formación más usada
            if passer_shirt in player_demarcations_from_formation:
                passer_demarcation = player_demarcations_from_formation[passer_shirt]
            
            # 2. Si no está en la formación, buscar en caché
            elif passer_id in player_demarcations_cache:
                passer_demarcation = player_demarcations_cache[passer_id]
            
            # 3. Si no está en caché, calcular histórico
            else:
                passer_demarcation = self.get_player_demarcation(passer_id, self.team_filter)
                player_demarcations_cache[passer_id] = passer_demarcation
            
            if not passer_demarcation:
                passer_demarcation = 'MC'  # Fallback
            
            # Acumular datos de esta demarcación
            demarcation_positions[passer_demarcation]['x'].append(row['x'])
            demarcation_positions[passer_demarcation]['y'].append(row['y'])
            demarcation_positions[passer_demarcation]['count'] += 1
            demarcation_positions[passer_demarcation]['dorsales'].add(passer_shirt)
            demarcation_players[passer_demarcation].add(passer_name)
            
            # 🔥 DETECTAR PASES ENTRE DEMARCACIONES
            if idx + 1 < len(data_sorted):
                next_pass = data_sorted.iloc[idx + 1]
                
                # Verificar que sea el mismo partido y equipo
                if (next_pass['Match ID'] == row['Match ID'] and 
                    next_pass['Team Name'] == row['Team Name']):
                    
                    time_diff = (next_pass['timeMin'] * 60 + next_pass['timeSec']) - \
                            (row['timeMin'] * 60 + row['timeSec'])
                    
                    # Pase válido (entre 0 y 10 segundos)
                    if 0 < time_diff <= 10:
                        receiver_id = next_pass['playerId']
                        
                        if receiver_id != passer_id:
                            # Obtener dorsal del receptor
                            receiver_shirt = self.get_player_shirt_number(receiver_id)
                            if not receiver_shirt:
                                continue
                            
                            # 🔥 DETERMINAR DEMARCACIÓN DEL RECEPTOR
                            receiver_demarcation = None
                            
                            if receiver_shirt in player_demarcations_from_formation:
                                receiver_demarcation = player_demarcations_from_formation[receiver_shirt]
                            elif receiver_id in player_demarcations_cache:
                                receiver_demarcation = player_demarcations_cache[receiver_id]
                            else:
                                receiver_demarcation = self.get_player_demarcation(receiver_id, self.team_filter)
                                player_demarcations_cache[receiver_id] = receiver_demarcation
                            
                            if not receiver_demarcation:
                                receiver_demarcation = 'MC'
                            
                            # Contar pase entre demarcaciones
                            pass_key = tuple(sorted([passer_demarcation, receiver_demarcation]))
                            pass_counts_by_demarcation[pass_key] += 1
        
        # 🔥 CALCULAR POSICIONES PROMEDIO POR DEMARCACIÓN
        all_demarcations = {}
        for demarcacion, data in demarcation_positions.items():
            if data['x']:
                all_demarcations[demarcacion] = {
                    'x': np.mean(data['x']),
                    'y': np.mean(data['y']),
                    'count': data['count'],
                    'players': list(demarcation_players[demarcacion]),
                    'dorsales': sorted([int(d) for d in data['dorsales'] if d.isdigit()])
                }
        
        # 🔥 SELECCIONAR LAS 11 DEMARCACIONES MÁS USADAS
        sorted_demarcations = sorted(
            all_demarcations.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:11]
        
        top_11_demarcations = dict(sorted_demarcations)
        
        for i, (dem, info) in enumerate(sorted_demarcations, 1):
            pass
        
        # 🔥 FILTRAR SOLO PASES ENTRE LAS TOP 11 DEMARCACIONES
        filtered_passes = {}
        top_11_keys = set(top_11_demarcations.keys())
        
        for pass_key, count in pass_counts_by_demarcation.items():
            dem1, dem2 = pass_key
            if dem1 in top_11_keys and dem2 in top_11_keys:
                filtered_passes[pass_key] = count
        
        
        return filtered_passes, top_11_demarcations
    
    def draw_pass_network_by_demarcation(self, ax, pass_counts, demarcation_positions, 
                                      title="RED DE PASES POR DEMARCACIÓN", min_passes=5):
        """
        Dibuja red de pases agrupada por DEMARCACIÓN.
        """
        if not pass_counts or not demarcation_positions:
            ax.text(0.5, 0.5, 'Sin datos suficientes', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12, fontweight='bold', color='white')
            return

        from mplsoccer import VerticalPitch
        
        pitch = VerticalPitch(pitch_type='opta', pitch_color='#2d5a27', 
                            line_color='white', linewidth=2, label=False, tick=False)
        pitch.draw(ax=ax)
        
        max_passes = max(pass_counts.values()) if pass_counts else 1
        min_passes_val = min(pass_counts.values()) if pass_counts else 1
        
        def get_color_gradient(count, min_val, max_val):
            if max_val == min_val:
                return '#FFFF00'
            ratio = (count - min_val) / (max_val - min_val)
            r = 255
            g = int(255 * (1 - ratio))
            b = 0
            return f'#{r:02x}{g:02x}{b:02x}'
        
        # Dibujar líneas entre demarcaciones
        for (dem1, dem2), count in pass_counts.items():
            if count >= min_passes:
                if dem1 in demarcation_positions and dem2 in demarcation_positions:
                    pos1 = demarcation_positions[dem1]
                    pos2 = demarcation_positions[dem2]
                    
                    width = 1 + (count / max_passes) * 9
                    color = get_color_gradient(count, min_passes_val, max_passes)
                    
                    pitch.lines(pos1['x'], pos1['y'], pos2['x'], pos2['y'],
                            lw=width, color=color, zorder=1, alpha=0.8, ax=ax)
        
        # Dibujar círculos por demarcación
        max_count = max(d['count'] for d in demarcation_positions.values())
        
        for demarcacion, info in demarcation_positions.items():
            marker_size = 300 + (info['count'] / max_count) * 700
            
            pitch.scatter(info['x'], info['y'],
                        s=marker_size,
                        color='#e74c3c', edgecolors='white', linewidth=3, 
                        alpha=1, zorder=3, ax=ax)
            
            # Etiqueta de demarcación (ARRIBA)
            pitch.annotate(demarcacion, 
                        xy=(info['x'], info['y']),  # Centro del círculo
                        c='white', va='bottom', ha='center',  # 🔥 va='bottom' para que el texto quede arriba del punto
                        size=10, weight='bold', ax=ax, zorder=4,
                        path_effects=[patheffects.withStroke(linewidth=3, foreground='#2c3e50')])

            # 🔥 Dorsales de los jugadores (DEBAJO)
            dorsales_list = info.get('dorsales', [])
            if dorsales_list:
                # Formato: "2-4-22"
                dorsales_text = '-'.join(str(d) for d in dorsales_list)
                
                pitch.annotate(dorsales_text, 
                            xy=(info['x'], info['y']),  # Mismo centro
                            c='yellow', va='top', ha='center',  # 🔥 va='top' para que el texto quede debajo del punto
                            size=7, weight='bold', ax=ax, zorder=4,
                            path_effects=[patheffects.withStroke(linewidth=2.5, foreground='#2c3e50')])
        
        ax.set_title(title, fontsize=11, fontweight='bold', color='white', pad=10, 
                    family='serif', bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='#2c3e50', alpha=0.8))

    def load_player_photos(self):
        if self.photos_data is None:
            try:
                with open('assets/jugadores_optimizados.json', 'r', encoding='utf-8') as f:
                    self.photos_data = json.load(f)
            except FileNotFoundError:
                print("⚠️ No se encontró el archivo jugadores_optimizados.json")
                self.photos_data = []
        return self.photos_data
    
    def load_ball_image(self): 
        return plt.imread("assets/balon.png") if os.path.exists("assets/balon.png") else None

    def load_background(self): 
        return plt.imread("assets/fondo_informes.png") if os.path.exists("assets/fondo_informes.png") else None
    
    def load_team_logo(self, equipo, target_size=(80, 80)):
        """Carga y redimensiona el logo del equipo buscando por palabras más largas primero"""
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            return self._load_team_logo_original(equipo)
        
        if not os.path.exists('assets/escudos'):
            print(f"⚠️ No existe la carpeta assets/escudos")
            return None
        
        # 🔥 NUEVA LÓGICA: Ordenar palabras por longitud
        def normalize_word(word):
            """Normaliza una palabra (sin acentos, minúsculas)"""
            word = unicodedata.normalize('NFD', word)
            word = ''.join(char for char in word if unicodedata.category(char) != 'Mn')
            return word.lower().strip()
        
        # Filtrar palabras comunes que no son útiles
        palabras_ignorar = {'cf', 'fc', 'cd', 'ud', 'rcd', 'rc', 'ca', 'de', 'del', 'la', 'las', 'el', 'los'}
        
        # Extraer palabras del nombre del equipo
        palabras = equipo.split()
        palabras_normalizadas = []
        
        for palabra in palabras:
            palabra_norm = normalize_word(palabra)
            # Solo agregar si no está en la lista de ignorar y tiene más de 2 caracteres
            if palabra_norm not in palabras_ignorar and len(palabra_norm) > 2:
                palabras_normalizadas.append(palabra_norm)
        
        # Ordenar por longitud (más larga primero)
        palabras_ordenadas = sorted(palabras_normalizadas, key=len, reverse=True)
        
        
        # Obtener todos los archivos disponibles
        all_files = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
        
        # Buscar por cada palabra en orden de longitud
        for palabra_buscar in palabras_ordenadas:
            pass
            
            for filename in all_files:
                nombre_archivo = os.path.splitext(filename)[0]
                nombre_archivo_norm = normalize_word(nombre_archivo)
                
                # Coincidencia exacta de la palabra en el nombre del archivo
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
                        print(f"   ⚠️ Error procesando {logo_path}: {e}")
                        continue
        
        # Si no encuentra nada, hacer búsqueda por similitud como fallback
        print(f"   ⚠️ No se encontró con palabras exactas, usando similitud...")
        best_match_path = None
        best_score = 0
        
        equipo_completo_norm = normalize_word(equipo.replace(' ', ''))
        
        for filename in all_files:
            nombre_archivo = os.path.splitext(filename)[0]
            nombre_archivo_norm = normalize_word(nombre_archivo)
            score = SequenceMatcher(None, equipo_completo_norm, nombre_archivo_norm).ratio()
            
            if score > best_score:
                best_score = score
                best_match_path = f"assets/escudos/{filename}"
        
        if best_match_path and best_score > 0.5:
            pass
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
            except Exception as e:
                print(f"   ⚠️ Error procesando {best_match_path}: {e}")
        
        print(f"   ❌ No se encontró escudo para: {equipo}")
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

    def extract_names_parts(self, name):
        """Extrae las partes de un nombre normalizado"""
        def normalize_name(name):
            """Normaliza un nombre eliminando acentos y caracteres especiales"""
            if not name:
                return ""
            name = str(name).lower().strip()
            name = unicodedata.normalize('NFD', name)
            name = ''.join(char for char in name if unicodedata.category(char) != 'Mn')
            name = re.sub(r"['\-`´'']", "", name)
            name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
            return ' '.join(name.split())
            
        normalized = normalize_name(name)
        parts = normalized.split()
        
        if not parts:
            return {
                'full': '',
                'first_name': '',
                'last_name': '',
                'all_parts': []
            }
            
        first_name = parts[0]
        last_name = parts[-1] if len(parts) > 1 else first_name
        
        return {
            'full': normalized,
            'first_name': first_name,
            'last_name': last_name,
            'all_parts': parts
        }

    def levenshtein_distance(self, s1, s2):
        """Calcula la distancia de Levenshtein entre dos strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def match_player_name(self, player_name, photos_data, team_filter=None):
        """Encuentra el jugador más parecido: primero filtra por equipo, luego busca por palabras más largas"""
        
        player_parts = self.extract_names_parts(player_name)
        if not player_parts['full']:
            return None

        # 🔥 PASO 1 MEJORADO: Filtrar por equipo usando palabras más largas
        team_players = []
        if team_filter:
            # Normalizar y ordenar palabras del equipo por longitud
            palabras_ignorar = {'cf', 'fc', 'cd', 'ud', 'rcd', 'rc', 'ca', 'de', 'del', 'la', 'las', 'el', 'los'}
            
            def normalize_word(word):
                word = unicodedata.normalize('NFD', word)
                word = ''.join(char for char in word if unicodedata.category(char) != 'Mn')
                return word.lower().strip()
            
            palabras_equipo = team_filter.split()
            palabras_equipo_norm = []
            
            for palabra in palabras_equipo:
                palabra_norm = normalize_word(palabra)
                if palabra_norm not in palabras_ignorar and len(palabra_norm) > 2:
                    palabras_equipo_norm.append(palabra_norm)
            
            # Ordenar por longitud (más larga primero)
            palabras_equipo_ordenadas = sorted(palabras_equipo_norm, key=len, reverse=True)
            
            
            # Buscar fotos que contengan al menos UNA de las palabras del equipo
            for photo_entry in photos_data:
                photo_team = photo_entry.get('team_name', '')
                if not photo_team:
                    continue
                
                # Normalizar el nombre del equipo en la foto
                palabras_photo_team = photo_team.split()
                palabras_photo_norm = [normalize_word(p) for p in palabras_photo_team]
                
                # Verificar si alguna palabra del equipo buscado está en el equipo de la foto
                match_encontrado = False
                for palabra_buscar in palabras_equipo_ordenadas:
                    if palabra_buscar in palabras_photo_norm:
                        match_encontrado = True
                        break
                
                if match_encontrado:
                    team_players.append(photo_entry)
            
            if not team_players:
                print(f"   ⚠️ No hay fotos para el equipo: {team_filter}")
                
                # Fallback: buscar con similitud en el nombre completo
                team_filter_norm = normalize_word(team_filter.replace(' ', ''))
                
                for photo_entry in photos_data:
                    photo_team = photo_entry.get('team_name', '')
                    if not photo_team:
                        continue
                    
                    photo_team_norm = normalize_word(photo_team.replace(' ', ''))
                    similarity = SequenceMatcher(None, team_filter_norm, photo_team_norm).ratio()
                    
                    if similarity > 0.7:  # 70% de similitud
                        team_players.append(photo_entry)
                
                if not team_players:
                    print(f"   ❌ No se encontraron fotos para '{team_filter}'")
                    return None
                else:
                    pass
        else:
            team_players = photos_data


        # 🔥 PASO 2: Ordenar palabras del jugador por longitud (más larga primero)
        # Filtrar palabras muy cortas (< 3 caracteres)
        player_words = [w for w in player_parts['all_parts'] if len(w) >= 3]
        player_words_sorted = sorted(player_words, key=len, reverse=True)
        

        # 🔥 PASO 3: Buscar por cada palabra en orden de longitud
        for palabra_buscar in player_words_sorted:
            pass
            
            for photo_entry in team_players:
                photo_name = photo_entry.get('player_name', '')
                photo_parts = self.extract_names_parts(photo_name)
                photo_words = [w for w in photo_parts['all_parts'] if len(w) >= 3]
                
                # Coincidencia exacta de la palabra
                if palabra_buscar in photo_words:
                    pass
                    return photo_entry
                
                # Tolerancia para palabras largas (1 letra de diferencia)
                if len(palabra_buscar) > 5:
                    for ph_word in photo_words:
                        if len(ph_word) > 5:
                            distance = self.levenshtein_distance(palabra_buscar, ph_word)
                            if distance == 1:
                                pass
                                return photo_entry

        # 🔥 PASO 4: Si no encuentra match exacto, usar sistema de scoring (fallback)
        print(f"   ⚠️ No se encontró match exacto, usando scoring...")
        
        candidates = []
        
        for photo_entry in team_players:
            photo_name = photo_entry.get('player_name', '')
            photo_parts = self.extract_names_parts(photo_name)
            photo_words = [w for w in photo_parts['all_parts'] if len(w) >= 3]
            
            matches = []
            for p_word in player_words:
                for ph_word in photo_words:
                    # Coincidencia exacta
                    if p_word == ph_word:
                        matches.append(p_word)
                    # Tolerancia para palabras largas
                    elif len(p_word) > 5 and len(ph_word) > 5:
                        distance = self.levenshtein_distance(p_word, ph_word)
                        if distance <= 2:  # Hasta 2 letras de diferencia
                            matches.append(p_word)
            
            if matches:
                candidates.append({
                    'entry': photo_entry,
                    'matches': matches,
                    'match_count': len(matches)
                })
        
        # Resolver conflictos por scoring
        if len(candidates) == 0:
            print(f"   ❌ No se encontró ningún candidato para: {player_name}")
            return None
        elif len(candidates) == 1:
            pass
            return candidates[0]['entry']
        else:
            best_candidates = sorted(candidates, key=lambda x: x['match_count'], reverse=True)
            
            if best_candidates[0]['match_count'] > best_candidates[1]['match_count']:
                pass
                return best_candidates[0]['entry']
            
            # Desempate por palabras cortas que coincidan al inicio
            for candidate in best_candidates:
                photo_parts = self.extract_names_parts(candidate['entry']['player_name'])
                
                for p_word in player_parts['all_parts']:
                    if len(p_word) <= 3:
                        for ph_word in photo_parts['all_parts']:
                            if ph_word.startswith(p_word):
                                pass
                                return candidate['entry']
            
            print(f"   ⚠️ Múltiples candidatos, devolviendo el primero: {best_candidates[0]['entry']['player_name']}")
            return best_candidates[0]['entry']

    def get_player_photo(self, player_name):
        """Obtiene la foto del jugador con el sistema robusto"""
        if self.photos_data is None: 
            self.load_player_photos()
        if not self.photos_data: 
            return None
        
        match = self.match_player_name(player_name, self.photos_data, self.team_filter)
        if not match: 
            return None
        
        try:
            img_data = base64.b64decode(match['image_base64'])
            img = Image.open(BytesIO(img_data)).convert('RGBA')
            data = np.array(img)
            
            # Flood fill para eliminar fondo blanco
            height, width = data.shape[:2]
            
            def flood_fill_iterative(start_points, threshold=235):
                visited = np.zeros((height, width), dtype=bool)
                background_mask = np.zeros((height, width), dtype=bool)
                
                def is_background_color(y, x):
                    if y < 0 or y >= height or x < 0 or x >= width:
                        return False
                    return (data[y, x, 0] >= threshold and 
                            data[y, x, 1] >= threshold and 
                            data[y, x, 2] >= threshold)
                
                for start_y, start_x in start_points:
                    if visited[start_y, start_x] or not is_background_color(start_y, start_x):
                        continue
                    
                    stack = [(start_y, start_x)]
                    
                    while stack:
                        y, x = stack.pop()
                        
                        if (y < 0 or y >= height or x < 0 or x >= width or 
                            visited[y, x] or not is_background_color(y, x)):
                            continue
                        
                        visited[y, x] = True
                        background_mask[y, x] = True
                        
                        stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
                
                return background_mask
            
            border_points = [
                (0, 0), (0, width-1), (height-1, 0), (height-1, width-1),
                (0, width//2), (height-1, width//2),
                (height//2, 0), (height//2, width-1)
            ]
            
            background_mask = flood_fill_iterative(border_points, threshold=230)
            data[background_mask] = [0, 0, 0, 0]
            
            return data.astype(np.float32) / 255.0
        
        except Exception as e:
            print(f"⚠️ Error procesando foto de {player_name}: {e}")
            return None

    def draw_pass_network_modern(self, ax, pass_counts, player_positions, title="RED DE PASES", min_passes=3):
        """
        Versión moderna con GRADIENTE ROJO (más pases) -> AMARILLO (menos pases)
        """
        if not pass_counts or not player_positions:
            ax.text(0.5, 0.5, 'Sin datos suficientes', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12, fontweight='bold', color='white')
            return

        passes_between_list = []
        
        max_passes = max(pass_counts.values()) if pass_counts else 1
        min_passes_val = min(pass_counts.values()) if pass_counts else 1
        
        # 🔥 FUNCIÓN PARA GRADIENTE ROJO -> AMARILLO
        def get_color_gradient(count, min_val, max_val):
            """Amarillo (menos pases) -> Rojo (más pases)"""
            if max_val == min_val:
                return '#FFFF00'
            
            ratio = (count - min_val) / (max_val - min_val)
            r = 255
            g = int(255 * (1 - ratio))
            b = 0
            
            return f'#{r:02x}{g:02x}{b:02x}'
        
        # 🔥 PASO 1: AGRUPAR JUGADORES POR DORSAL (eliminar duplicados)
        players_by_shirt = {}
        for player_id, pos in player_positions.items():
            shirt_number = pos['shirt']
            
            # Validar dorsal
            if not shirt_number or pd.isna(shirt_number) or not str(shirt_number).strip():
                continue
            
            shirt_str = str(shirt_number).strip()
            
            # Si ya existe este dorsal, promediar posiciones
            if shirt_str in players_by_shirt:
                players_by_shirt[shirt_str]['x'].append(pos['x'])
                players_by_shirt[shirt_str]['y'].append(pos['y'])
            else:
                players_by_shirt[shirt_str] = {
                    'x': [pos['x']],
                    'y': [pos['y']],
                    'name': pos['name']
                }
        
        # 🔥 PASO 2: CALCULAR POSICIONES PROMEDIO POR DORSAL
        unique_players = {}
        for shirt_str, data in players_by_shirt.items():
            unique_players[shirt_str] = {
                'x': np.mean(data['x']),
                'y': np.mean(data['y']),
                'name': data['name'],
                'shirt': shirt_str
            }
        
        
        # 🔥 PASO 3: CREAR LISTA PARA VISUALIZACIÓN
        average_locs_list = []
        for shirt_str, player_data in unique_players.items():
            average_locs_list.append({
                'shirt': shirt_str,
                'x': player_data['x'],
                'y': player_data['y'],
                'marker_size': 500
            })
        
        # Convertir pass_counts usando dorsales
        for (p1_id, p2_id), count in pass_counts.items():
            if count >= min_passes and p1_id in player_positions and p2_id in player_positions:
                pos1 = player_positions[p1_id]
                pos2 = player_positions[p2_id]
                
                shirt1 = str(pos1['shirt']).strip() if pos1['shirt'] else None
                shirt2 = str(pos2['shirt']).strip() if pos2['shirt'] else None
                
                # Solo dibujar si ambos tienen dorsal válido y están en unique_players
                if shirt1 and shirt2 and shirt1 in unique_players and shirt2 in unique_players:
                    width = 1 + (count / max_passes) * 9
                    color = get_color_gradient(count, min_passes_val, max_passes)
                    
                    passes_between_list.append({
                        'x': unique_players[shirt1]['x'], 
                        'y': unique_players[shirt1]['y'],
                        'x_end': unique_players[shirt2]['x'], 
                        'y_end': unique_players[shirt2]['y'],
                        'width': width, 
                        'count': count, 
                        'color': color
                    })
        
        passes_between = pd.DataFrame(passes_between_list)
        average_locs = pd.DataFrame(average_locs_list)
        
        if average_locs.empty:
            ax.text(0.5, 0.5, 'Sin jugadores válidos', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='white')
            return
        
        from mplsoccer import VerticalPitch
        pitch = VerticalPitch(pitch_type='opta', pitch_color='#2d5a27', 
                            line_color='white', linewidth=2, label=False, tick=False)
        
        # 🔥 DIBUJAR LÍNEAS
        if not passes_between.empty:
            for _, row in passes_between.iterrows():
                pitch.lines(row.x, row.y, row.x_end, row.y_end,
                        lw=row.width, color=row.color, zorder=1, 
                        alpha=0.8, ax=ax)
        
        # 🔥 DIBUJAR CÍRCULOS
        pitch.scatter(average_locs.x, average_locs.y,
                    s=average_locs.marker_size,
                    color='#e74c3c', edgecolors='white', linewidth=3, 
                    alpha=1, zorder=3, ax=ax)
        
        # 🔥 DIBUJAR DORSALES (uno por círculo)
        for _, player in average_locs.iterrows():
            pitch.annotate(player.shirt, 
                        xy=(player.x, player.y), 
                        c='white', va='center', ha='center', 
                        size=16, weight='bold', ax=ax, zorder=4,
                        path_effects=[patheffects.withStroke(linewidth=3, foreground='#2c3e50')])
        
        ax.set_title(title, fontsize=11, fontweight='bold', color='white', pad=10, 
                    family='serif', bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='#2c3e50', alpha=0.8))
    
    def create_positional_heatmap(self, ax, title="MAPA DE CALOR POSICIONAL"):
        """
        Crea un heatmap posicional con los pases exitosos
        """
        if self.passes_data.empty:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='white')
            return
        
        from mplsoccer import VerticalPitch
        
        # Pitch vertical con fondo oscuro
        pitch = VerticalPitch(pitch_type='opta', line_zorder=2,
                            pitch_color='#22312b', line_color='white')
        pitch.draw(ax=ax)
        
        # Filtrar solo pases exitosos
        df_success = self.passes_data[self.passes_data['x'].notna() & 
                                    self.passes_data['y'].notna()].copy()
        
        if df_success.empty:
            return
        
        # Calcular estadística posicional
        bin_statistic = pitch.bin_statistic_positional(
            df_success.x, df_success.y, 
            statistic='count',
            positional='full', 
            normalize=True
        )
        
        # Dibujar heatmap
        pitch.heatmap_positional(bin_statistic, ax=ax, cmap='coolwarm', 
                                edgecolors='#22312b', alpha=0.8)
        
        # Puntos de pases
        pitch.scatter(df_success.x, df_success.y, c='white', s=2, ax=ax, alpha=0.5)
        
        # Etiquetas con porcentajes
        path_eff = [patheffects.Stroke(linewidth=3, foreground='#22312b'),
                    patheffects.Normal()]
        
        pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=14,
                        ax=ax, ha='center', va='center',
                        str_format='{:.0%}', path_effects=path_eff)
        
        ax.set_title(title, fontsize=10, fontweight='bold', color='white', 
                    pad=10, bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='#2c3e50', alpha=0.8))

    def create_pass_flow_map(self, ax, title="MAPA DE FLUJO DE PASES", bins=(6, 4)):
        """
        Crea un mapa de flujo de pases con flechas
        """
        if self.passes_data.empty:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='white')
            return
        
        from mplsoccer import VerticalPitch
        
        # Pitch vertical
        pitch = VerticalPitch(pitch_type='opta', line_zorder=2,
                            pitch_color='#22312b', line_color='white')
        pitch.draw(ax=ax)
        
        # Filtrar pases con coordenadas válidas
        df_pass = self.passes_data[
            self.passes_data['x'].notna() & 
            self.passes_data['y'].notna() &
            self.passes_data['Pass End X'].notna() &
            self.passes_data['Pass End Y'].notna()
        ].copy()
        
        if df_pass.empty:
            return
        
        # Heatmap de origen de pases
        bs_heatmap = pitch.bin_statistic(df_pass.x, df_pass.y, 
                                        statistic='count', bins=bins)
        pitch.heatmap(bs_heatmap, ax=ax, cmap='Greens', alpha=0.6)
        
        # Mapa de flujo con flechas
        pitch.flow(df_pass.x, df_pass.y, 
                df_pass['Pass End X'], df_pass['Pass End Y'],
                color='black', arrow_type='average', 
                bins=bins, ax=ax, alpha=0.8)
        
        ax.set_title(title, fontsize=10, fontweight='bold', color='white',
                    pad=10, bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='#2c3e50', alpha=0.8))

    def get_top_pass_sequences(self, sequence_length=2, top_n=2):
        """
        Encuentra las secuencias de pases más frecuentes de longitud específica.
        
        Reglas especiales:
        - Para 2 pases (1 conexión): A→B y B→A se cuentan como la misma secuencia (ida y vuelta)
        - Para 3 pases (2 conexiones): No puede aparecer el mismo jugador dos veces
        - Para 4 pases (3 conexiones): El jugador puede estar en posición 1 y 4, pero no en el tramo de 3 pases intermedios
        
        Selección:
        - 1ª secuencia: la más repetida
        - 2ª secuencia: la más repetida con mayor progresión (distancia en X)
        """
        if self.passes_data.empty:
            return []
        
        data_sorted = self.passes_data.sort_values(
            ['Match ID', 'timeMin', 'timeSec']
        ).reset_index(drop=True)
        
        sequences = []
        
        for idx in range(len(data_sorted) - sequence_length):
            current_match = data_sorted.iloc[idx]['Match ID']
            current_team = data_sorted.iloc[idx]['Team Name']
            
            sequence = []
            valid_sequence = True
            first_pass_x = None
            last_pass_end_x = None
            last_pass_outcome = None
            
            for i in range(sequence_length + 1):
                row = data_sorted.iloc[idx + i]
                
                # Capturar coordenadas del primer pase
                if i == 0:
                    first_pass_x = row.get('x')
                
                # Capturar coordenadas y outcome del último pase
                if i == sequence_length:
                    last_pass_end_x = row.get('Pass End X')
                    last_pass_outcome = row.get('outcome')
                
                if row['Match ID'] != current_match or row['Team Name'] != current_team:
                    valid_sequence = False
                    break
                
                if i > 0:
                    prev_row = data_sorted.iloc[idx + i - 1]
                    time_diff = (row['timeMin'] * 60 + row['timeSec']) - \
                            (prev_row['timeMin'] * 60 + prev_row['timeSec'])
                    if time_diff > 10 or time_diff <= 0:
                        valid_sequence = False
                        break
                    
                    # Validar que no sea el mismo jugador consecutivo
                    if row['playerId'] == prev_row['playerId']:
                        valid_sequence = False
                        break
                
                player_id = row['playerId']
                player_name = row['playerName']
                shirt_number = row['shirt_number']
                
                sequence.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'shirt_number': shirt_number
                })
            
            if valid_sequence and len(sequence) == sequence_length + 1:
                player_ids = [p['player_id'] for p in sequence]
                
                # 🔥 CALCULAR PROGRESIÓN
                progression = 0
                if (first_pass_x is not None and last_pass_end_x is not None and 
                    last_pass_outcome == 1):
                    progression = abs(last_pass_end_x - first_pass_x)
                
                # 🔥 VALIDACIONES ESPECIALES POR TIPO DE SECUENCIA
                
                # Para 2 pases (sequence_length=1): A→B y B→A son lo mismo (ida y vuelta)
                if sequence_length == 1:
                    sequence_key = tuple(sorted(player_ids))
                
                # Para 3 pases (sequence_length=2): No puede haber jugadores repetidos
                elif sequence_length == 2:
                    if len(set(player_ids)) != len(player_ids):
                        continue
                    sequence_key = tuple(player_ids)
                
                # Para 4 pases (sequence_length=3): Jugador 1 y 4 pueden ser iguales, pero no en tramos de 3
                elif sequence_length == 3:
                    if len(set(player_ids[:3])) != 3:
                        continue
                    if len(set(player_ids[1:])) != 3:
                        continue
                    sequence_key = tuple(player_ids)
                
                else:
                    sequence_key = tuple(player_ids)
                
                sequences.append((sequence_key, sequence, progression))
        
        from collections import Counter, defaultdict
        
        # Contar frecuencias
        sequence_counts = Counter([seq[0] for seq in sequences])
        
        # Calcular progresión máxima por secuencia
        sequence_max_progression = defaultdict(float)
        sequence_data_map = {}
        
        for seq_key, seq_data, progression in sequences:
            if progression > sequence_max_progression[seq_key]:
                sequence_max_progression[seq_key] = progression
            if seq_key not in sequence_data_map:
                sequence_data_map[seq_key] = seq_data
        
        # 🔥 SELECCIÓN INTELIGENTE
        result = []
        
        if not sequence_counts:
            return result
        
        # 1️⃣ Primera secuencia: LA MÁS REPETIDA
        most_common = sequence_counts.most_common(1)[0]
        seq_tuple_1, count_1 = most_common
        result.append({
            'sequence': sequence_data_map[seq_tuple_1],
            'count': count_1,
            'length': sequence_length,
            'progression': sequence_max_progression[seq_tuple_1]
        })
        
        # 2️⃣ Segunda secuencia: LA MÁS REPETIDA CON MAYOR PROGRESIÓN
        if top_n >= 2:
            # Obtener candidatos (las top 10 más repetidas, o todas si hay menos)
            candidates = sequence_counts.most_common(min(10, len(sequence_counts)))
            
            # Filtrar la que ya seleccionamos
            candidates = [c for c in candidates if c[0] != seq_tuple_1]
            
            if candidates:
                # Encontrar la que tenga mayor progresión entre las candidatas
                best_candidate = max(
                    candidates,
                    key=lambda x: (sequence_max_progression[x[0]], x[1])  # Prioridad: progresión, luego frecuencia
                )
                
                seq_tuple_2, count_2 = best_candidate
                result.append({
                    'sequence': sequence_data_map[seq_tuple_2],
                    'count': count_2,
                    'length': sequence_length,
                    'progression': sequence_max_progression[seq_tuple_2]
                })
                
        
        return result

    def draw_pass_sequence(self, ax, sequence_data, title, panel_color='#3498db', is_second=False):
        """
        Dibuja una secuencia de pases con fotos GRANDES, dorsales estilo fútbol en esquina.
        """
        ax.set_facecolor('#f0f0f0')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        if not sequence_data or 'sequence' not in sequence_data:
            ax.text(0.5, 0.5, 'Sin datos suficientes', ha='center', va='center',
                    fontsize=8, color='grey')
            return

        sequence = sequence_data['sequence']
        count = sequence_data['count']

        # Subtítulo según si es primera o segunda secuencia
        if is_second:
            subtitle_text = f'{count} veces repetida (con mayor progresión)'
        else:
            subtitle_text = f'{count} veces repetida'

        ax.text(0.5, 0.99, subtitle_text, ha='center', va='center',
                fontsize=7, color='grey', style='italic')
        
        n_players = len(sequence)
        
        if n_players == 2:
            x_positions = [0.25, 0.75]
            y_center = 0.5
        elif n_players == 3:
            x_positions = [0.20, 0.50, 0.80]
            y_center = 0.5
        elif n_players == 4:
            x_positions = [0.15, 0.38, 0.62, 0.85]
            y_center = 0.5
        else:
            return
        
        self.load_player_photos()
        
        for i, player in enumerate(sequence):
            x_pos = x_positions[i]
            
            # Fondo blanco para cada jugador
            rect_bg = patches.FancyBboxPatch(
                (x_pos - 0.08, y_center - 0.32), 0.16, 0.60,  # 🔥 Más alto para acomodar foto grande
                boxstyle="round,pad=0.015", 
                facecolor='white',
                edgecolor='lightgrey', 
                linewidth=1.5
            )
            ax.add_patch(rect_bg)
            
            # 🔥 DORSAL EN ESQUINA SUPERIOR DERECHA - ESTILO DEPORTIVO
            dorsal = str(int(player.get('shirt_number', 0))) if player.get('shirt_number') else '?'
            
            # Posición en esquina superior derecha del cuadrado
            dorsal_x = x_pos + 0.090  # Esquina derecha
            dorsal_y = y_center + 0.40  # Esquina superior
            
            # Texto con estilo deportivo (con sombra y outline)
            ax.text(dorsal_x, dorsal_y, dorsal, 
                    ha='right', va='top',
                    fontsize=16,  # Más grande
                    fontweight='heavy',  # Extra bold
                    color=panel_color,
                    zorder=15,
                    family='sans-serif',
                    style='italic',  # Cursiva deportiva
                    path_effects=[
                        patheffects.Stroke(linewidth=4, foreground='white'),  # Borde blanco grueso
                        patheffects.Normal(),
                        patheffects.SimplePatchShadow(offset=(1, -1), shadow_rgbFace='black', alpha=0.3)  # Sombra
                    ])
            
            # 🔥 FOTO GRANDE (el doble)
            player_photo = self.get_player_photo(player.get('player_name', ''))
            if player_photo is not None:
                photo_size = 0.60  # 🔥 DOBLE DE TAMAÑO (era 0.10)
                photo_ax = ax.inset_axes([
                    x_pos - photo_size/2, 
                    y_center - 0.05 - photo_size/2,  # Ajustado para centrar mejor
                    photo_size, 
                    photo_size
                ])
                photo_ax.imshow(player_photo)
                photo_ax.axis('off')
            
            # Nombre del jugador con BADGE moderno
            player_name = player.get('player_name', 'N/A')
            apellido = player_name.split()[-1] if player_name else 'N/A'

            # Badge de fondo BLANCO para contraste con azul
            name_badge = patches.FancyBboxPatch(
                (x_pos - 0.055, y_center - 0.30), 0.11, 0.025,
                boxstyle="round,pad=0.003",
                facecolor='white',  # 🔥 FONDO BLANCO
                edgecolor=panel_color,  # 🔥 BORDE del color del panel
                linewidth=2,
                alpha=0.95,
                zorder=14
            )
            ax.add_patch(name_badge)

            # 🔥 Texto en AZUL y NEGRITA
            ax.text(x_pos, y_center - 0.288, apellido.upper(), 
                    ha='center', va='center',
                    fontsize=5, fontweight='bold', 
                    color='white', zorder=15)  # 🔥 AZUL OSCURO (#004C9F)
            
            # Dibujar flecha hacia el siguiente jugador
            if i < len(sequence) - 1:
                arrow_start_x = x_pos + 0.09
                arrow_end_x = x_positions[i + 1] - 0.09
                arrow_y = y_center
                
                arrow = patches.FancyArrowPatch(
                    (arrow_start_x, arrow_y), 
                    (arrow_end_x, arrow_y),
                    arrowstyle='->,head_width=0.3,head_length=0.25',
                    color=panel_color,
                    linewidth=2.5,
                    zorder=5,
                    alpha=0.8
                )
                ax.add_patch(arrow)

    # --- FUNCIÓN DE VISUALIZACIÓN CON NUEVOS RANKINGS POR PARTIDO --
    def create_passing_network_visualization(self, figsize=(16.5, 8.27), team_filter=None):
        """
        Layout con secuencias de pases en columnas 4-5
        """
        if self.passes_data.empty:
            print("❌ No hay datos de pases para visualizar la red.")
            return None
        
        self.photos_data = self.load_player_photos()
        
        fig = plt.figure(figsize=figsize, facecolor='white')

        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25)
            ax_bg.axis('off')

        # 🔥 GRIDSPEC: 6 filas x 5 columnas para secuencias
        gs = fig.add_gridspec(6, 5, 
                            width_ratios=[1, 1, 1, 0.8, 0.8],
                            height_ratios=[1, 1, 1, 1, 1, 1],
                            hspace=0.12, wspace=0.08,
                            left=0.02, right=0.98, top=0.88, bottom=0.03)
        
        fig.suptitle(f'ANÁLISIS COMPLETO DE PASES - {team_filter.upper()}', 
                    fontsize=18, fontweight='bold', color='#1e3d59', y=0.96, family='serif')
        
        if (ball := self.load_ball_image()) is not None:
            ax_ball = fig.add_axes([0.02, 0.91, 0.04, 0.04])
            ax_ball.imshow(ball)
            ax_ball.axis('off')

        if team_filter and (team_logo := self.load_team_logo(team_filter)) is not None:
            ax_team = fig.add_axes([0.92, 0.90, 0.08, 0.08])
            ax_team.imshow(team_logo, aspect='auto')
            ax_team.axis('off')

        # COLUMNAS 1-2: RED DE PASES
        ax_main = fig.add_subplot(gs[:, 0:2])
        pitch_main = VerticalPitch(pitch_type='opta', pitch_color='#2d5a27', 
                                line_color='white', linewidth=2, label=False, tick=False)
        pitch_main.draw(ax=ax_main)
        
        pass_counts_dem, positions_dem = self.calculate_pass_network_by_demarcation(self.passes_data)
        self.draw_pass_network_by_demarcation(ax_main, pass_counts_dem, positions_dem, 
                                     f"RED DE PASES - TOP 11 DEMARCACIONES", min_passes=5)
        
        # COLUMNA 3: HEATMAPS
        ax_heatmap_pos = fig.add_subplot(gs[0:3, 2])
        self.create_positional_heatmap(ax_heatmap_pos, "MAPA POSICIONAL")
        
        ax_flow = fig.add_subplot(gs[3:6, 2])
        self.create_pass_flow_map(ax_flow, "FLUJO DE PASES")
        
        # 🔥 COLUMNAS 4-5: SECUENCIAS DE PASES

        # 🔥 TÍTULO GENERAL PARA LAS SECUENCIAS (similar a los otros títulos)
        ax_titulo_secuencias = fig.add_axes([0.70, 0.89, 0.28, 0.03])  # Posición y tamaño
        ax_titulo_secuencias.axis('off')
        ax_titulo_secuencias.text(0.5, 0.5, 'SECUENCIAS MÁS REPETIDAS', 
                                ha='center', va='center',
                                fontsize=9, fontweight='bold', color='white',
                                bbox=dict(boxstyle='round,pad=0.5', 
                                        facecolor='#2c3e50', alpha=0.9,
                                        edgecolor='white', linewidth=2))

        color_1_pase = '#e67e22'
        color_2_pases = '#3498db'
        color_3_pases = '#27ae60'

        sequences_1 = self.get_top_pass_sequences(sequence_length=1, top_n=2)
        sequences_2 = self.get_top_pass_sequences(sequence_length=2, top_n=2)
        sequences_3 = self.get_top_pass_sequences(sequence_length=3, top_n=2)
        
        # SECUENCIAS DE 1 PASE (filas 0-1)
        if len(sequences_1) > 0:
            ax_seq1_1 = fig.add_subplot(gs[0, 3:5])
            self.draw_pass_sequence(ax_seq1_1, sequences_1[0], "", color_1_pase)

        if len(sequences_1) > 1:
            ax_seq1_2 = fig.add_subplot(gs[1, 3:5])
            self.draw_pass_sequence(ax_seq1_2, sequences_1[1], "", color_1_pase, is_second=True)

        # SECUENCIAS DE 2 PASES (filas 2-3)
        if len(sequences_2) > 0:
            ax_seq2_1 = fig.add_subplot(gs[2, 3:5])
            self.draw_pass_sequence(ax_seq2_1, sequences_2[0], "", color_2_pases)

        if len(sequences_2) > 1:
            ax_seq2_2 = fig.add_subplot(gs[3, 3:5])
            self.draw_pass_sequence(ax_seq2_2, sequences_2[1], "", color_2_pases, is_second=True)

        # SECUENCIAS DE 3 PASES (filas 4-5)
        if len(sequences_3) > 0:
            ax_seq3_1 = fig.add_subplot(gs[4, 3:5])
            self.draw_pass_sequence(ax_seq3_1, sequences_3[0], "", color_3_pases)

        if len(sequences_3) > 1:
            ax_seq3_2 = fig.add_subplot(gs[5, 3:5])
            self.draw_pass_sequence(ax_seq3_2, sequences_3[1], "", color_3_pases, is_second=True)
        
        
        return fig

    def print_summary(self, team_filter=None):
        """Imprime resumen de los datos"""
        if self.passes_data.empty:
            pass
            return
        
        
        top_passers = self.passes_data['playerName'].value_counts().head(5)
        for i, (player, count) in enumerate(top_passers.items(), 1):
            shirt = self.passes_data[self.passes_data['playerName'] == player]['shirt_number'].iloc[0]

def seleccionar_equipo_interactivo():
    """Función para seleccionar equipo interactivamente"""
    try:
        # 🔥 Usar caché con solo la columna necesaria
        df = RedPasesEquipo._get_open_play_data(columns=['Team Name'])
        equipos = sorted(df['Team Name'].dropna().unique())
        if not equipos: 
            pass
            return None
        
        for i, equipo in enumerate(equipos, 1): 
            pass
        
        for _ in range(3):
            try:
                indice = int(input(f"\nSelecciona un equipo (1-{len(equipos)}): ").strip()) - 1
                if 0 <= indice < len(equipos):
                    return equipos[indice]
                else:
                    pass
            except EOFError:
                return equipos[0] if equipos else None
            except ValueError:
                pass
        return equipos[0] if equipos else None
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
        
        analyzer = RedPasesEquipo(team_filter=equipo)
        analyzer.print_summary(team_filter=equipo)
        
        if (fig := analyzer.create_passing_network_visualization(team_filter=equipo)):
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"red_pases_y_secuencias_{equipo_filename}.pdf"
            analyzer.guardar_sin_espacios(fig, output_path)
            plt.show()
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()

def generar_red_pases_personalizado(equipo, mostrar=True, guardar=True):
    """Función para generar red de pases de forma personalizada"""
    try:
        analyzer = RedPasesEquipo(team_filter=equipo)
        analyzer.print_summary(team_filter=equipo)
        fig = analyzer.create_passing_network_visualization(team_filter=equipo)
        
        if fig:
            if mostrar: plt.show()
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"red_pases_y_secuencias_{equipo_filename}.pdf"
                analyzer.guardar_sin_espacios(fig, output_path)
                fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1, 
                           facecolor='white', dpi=300)
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
    pass
    try:
        # 🔥 Usar caché para evitar cargar el parquet dos veces
        df = RedPasesEquipo._get_open_play_data()
        equipos = sorted(df['Team Name'].dropna().unique())
        if equipos:
            pass
        main()
    except Exception as e:
        print(f"❌ Error al inicializar: {e}")
    finally:
        # 🧹 Liberar memoria al finalizar
        import gc
        RedPasesEquipo._open_play_cache = None
        RedPasesEquipo._team_stats_cache = None
        RedPasesEquipo._player_stats_cache = None
        gc.collect()
        print("🧹 Memoria liberada al finalizar el script")