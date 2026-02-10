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

        # Diccionario central para los nombres de las zonas
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
        """Normaliza timestamps quitando la Z final si existe"""
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
        Cuenta la zona de caída de TODOS los lanzamientos de córner 
        desde la DERECHA (y < 1) usando Pass End X/Y.
        """
        # CAMBIO PRINCIPAL: y < 1 en lugar de y > 99
        lanzamientos = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] < 1)  # ← CAMBIO AQUÍ
        ]

        zonas_count = {zona_key: 0 for zona_key in self.ZONA_NAMES.keys()}

        for _, corner in lanzamientos.iterrows():
            pass_end_x = corner['Pass End X']
            pass_end_y = corner['Pass End Y']
            
            zona = self.get_zona_from_coordinates(pass_end_x, pass_end_y)
            
            if zona in zonas_count:
                zonas_count[zona] += 1
        
        return zonas_count

    def get_aerial_duels_data(self):
        """Obtiene jugadores que ganan duelos aéreos tras corners desde y < 1"""
        
        # CAMBIO: y < 1
        corners_y1 = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] < 1)  # ← CAMBIO AQUÍ
        ]
        
        aerials_won = self.corner_data[
            (self.corner_data['Event Name'] == 'Aerial') &
            (self.corner_data['outcome'] == 1)
        ]
        
        aerial_players = []
        
        for _, corner in corners_y1.iterrows():
            corner_time = corner['timeMin'] * 60 + corner['timeSec']
            
            matching_aerials = aerials_won[
                (aerials_won['Team ID'] == corner['Team ID']) &
                (aerials_won['Match ID'] == corner['Match ID']) &
                (aerials_won['timeMin'] * 60 + aerials_won['timeSec'] > corner_time) &
                (aerials_won['timeMin'] * 60 + aerials_won['timeSec'] <= corner_time + 15)
            ]
            
            for _, aerial in matching_aerials.iterrows():
                aerial_players.append(aerial['playerName'])
        
        from collections import Counter
        return Counter(aerial_players)
    
    def get_aerial_duels_data_completo(self):
        """Obtiene datos completos de duelos aéreos tras corners desde y < 1"""
        
        # CAMBIO: y < 1
        corners_y1 = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] < 1)  # ← CAMBIO AQUÍ
        ]
        
        if corners_y1.empty:
            return {}
        
        aerials_all = self.corner_data[self.corner_data['Event Name'] == 'Aerial']
        
        aerial_stats = {}
        
        
        for _, corner in corners_y1.iterrows():
            corner_time = corner['timeMin'] * 60 + corner['timeSec']
            
            matching_aerials = aerials_all[
                (aerials_all['Team ID'] == corner['Team ID']) &
                (aerials_all['Match ID'] == corner['Match ID']) &
                (aerials_all['timeMin'] * 60 + aerials_all['timeSec'] > corner_time) &
                (aerials_all['timeMin'] * 60 + aerials_all['timeSec'] <= corner_time + 30)
            ]
            
            
            for _, aerial in matching_aerials.iterrows():
                player_name = aerial['playerName']
                
                if player_name not in aerial_stats:
                    aerial_stats[player_name] = {'exitos': 0, 'total': 0}
                
                aerial_stats[player_name]['total'] += 1
                
                if aerial['outcome'] == 1:
                    aerial_stats[player_name]['exitos'] += 1
                else:
                    pass
        
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
        """Obtiene remates de primer contacto directo desde y < 1"""
        
        # CAMBIO: y < 1
        corners_y1 = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] < 1)  # ← CAMBIO AQUÍ
        ]
        
        remates = self.corner_data[
            (self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])) &
            (self.corner_data['From corner'] == 'Sí')
        ]
        
        remates_primer_contacto = {}
        
        for _, remate in remates.iterrows():
            remate_timestamp = pd.to_datetime(remate['timeStamp'])
            
            corner_previo = corners_y1[
                (corners_y1['Match ID'] == remate['Match ID']) &
                (pd.to_datetime(corners_y1['timeStamp']) < remate_timestamp) &
                (pd.to_datetime(corners_y1['timeStamp']) >= remate_timestamp - pd.Timedelta(seconds=6))
            ]
            
            if not corner_previo.empty:
                corner = corner_previo.iloc[-1]
                corner_timestamp = pd.to_datetime(corner['timeStamp'])
                
                pases_intermedios = self.corner_data[
                    (self.corner_data['Event Name'] == 'Pass') &
                    (self.corner_data['Team ID'] == remate['Team ID']) &
                    (self.corner_data['Match ID'] == remate['Match ID']) &
                    (pd.to_datetime(self.corner_data['timeStamp']) > corner_timestamp) &
                    (pd.to_datetime(self.corner_data['timeStamp']) < remate_timestamp) &
                    (self.corner_data['timeStamp'] != corner['timeStamp'])
                ]
                
                if pases_intermedios.empty:
                    player_name = remate['playerName']
                    remates_primer_contacto[player_name] = remates_primer_contacto.get(player_name, 0) + 1
        
        return remates_primer_contacto

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
        """Encuentra el nombre más parecido en los datos de las fotos"""
        
        def normalize_name(name):
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
            player_full = player_parts['full']
            photo_full = photo_parts['full']

            if player_full == photo_full:
                return (1.0, "MATCH EXACTO COMPLETO")

            if len(player_parts['all_parts']) == 1:
                if player_full in photo_parts['all_parts']:
                    return (0.95, f"MATCH DE NOMBRE ÚNICO '{player_full}'")

            if (len(player_parts['first_name']) == 1 and
                player_parts['last_name'] == photo_parts['last_name'] and
                photo_parts['first_name'].startswith(player_parts['first_name'])):
                return (0.90, "INICIAL + APELLIDO EXACTO")

            if (player_parts['first_name'] == photo_parts['first_name'] and
                len(player_parts['last_name']) == 1 and
                photo_parts['last_name'].startswith(player_parts['last_name'][0])):
                return (0.90, "NOMBRE + INICIAL APELLIDO")

            if player_parts['last_name'] == photo_parts['last_name']:
                first_name_sim = SequenceMatcher(None, player_parts['first_name'], photo_parts['first_name']).ratio()
                if first_name_sim >= 0.8:
                    return (0.85 + (first_name_sim * 0.05), f"APELLIDO EXACTO + NOMBRE SIMILAR ({first_name_sim:.2f})")
            
            full_sim = SequenceMatcher(None, player_full, photo_full).ratio()
            if full_sim > 0.8:
                return (full_sim, f"SIMILITUD GENERAL ALTA ({full_sim:.2f})")

            return (0.0, "SIN COINCIDENCIA CLARA")

        player_parts = extract_names_parts(player_name)
        if not player_parts['full']:
            return None

        
        found_matches = []
        
        for photo_entry in photos_data:
            photo_name = photo_entry.get('player_name')
            if not photo_name:
                continue
                
            photo_parts = extract_names_parts(photo_name)
            score, reason = calculate_match_score(player_parts, photo_parts)
            
            if score >= 0.90:
                found_matches.append({
                    "entry": photo_entry,
                    "score": score,
                    "reason": reason
                })
                
        if len(found_matches) == 1:
            best_match = found_matches[0]
            return best_match['entry']
        
        elif len(found_matches) > 1:
            print(f"⚠️  ADVERTENCIA: Se encontraron {len(found_matches)} matches de alta calidad para '{player_name}'. Se descarta por ambigüedad.")
            for match in sorted(found_matches, key=lambda x: x['score'], reverse=True):
                pass
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
            columns_needed = ['Match ID', 'periodId', 'Team ID', 'Team Name', 'Event Name', 'outcome', 
                            'timeMin', 'timeSec', 'x', 'y', 'Pass End X', 'Pass End Y', 
                            'playerName', 'playerId', 'Corner taken', 'From corner',
                            'In-swinger', 'Out-swinger', 'Straight', 'Left footed', 'Right footed', 'Cross',
                            'Throw in', 'Free kick taken', 'timeStamp']
                        
            try:
                self.df = pd.read_parquet(self.data_path, columns=columns_needed)
            except Exception:
                basic_columns = ['Match ID', 'Team ID', 'Team Name', 'Event Name', 'outcome', 
                            'timeMin', 'timeSec', 'x', 'y', 'Pass End X', 'Pass End Y', 
                            'playerName', 'playerId']
                self.df = pd.read_parquet(self.data_path, columns=basic_columns)
                for col in ['Corner taken', 'From corner', 'In-swinger', 'Out-swinger', 'Straight', 'Cross', 'timeStamp', 'periodId', 'Throw in', 'Free kick taken']:
                    if col not in self.df.columns:
                        if col == 'timeStamp':
                            self.df[col] = pd.NaT
                        elif col == 'periodId':
                            self.df[col] = 1
                        else:
                            self.df[col] = 'No'
            
            if 'timeStamp' in self.df.columns:
                self.df['timeStamp'] = self.df['timeStamp'].apply(self.normalize_timestamp)
            
            relevant_events = ['Pass', 'Goal', 'Attempt Saved', 'Miss', 'Post', 'Aerial']
            self.df = self.df[self.df['Event Name'].isin(relevant_events)]
            
            if team_filter:
                team_matches = self.team_stats[self.team_stats['Team Name'] == team_filter]['Match ID'].unique()
                self.df = self.df[self.df['Match ID'].isin(team_matches)]
            
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")
    
    def extract_corner_data(self, team_filter):
        """Extrae datos específicos de córneres ofensivos"""
        if self.df is None:
            print("❌ No hay datos cargados")
            return
        
        
        team_data = self.df[self.df['Team Name'] == team_filter].copy()
        team_data = team_data.sort_values(['Match ID', 'timeMin', 'timeSec']).reset_index(drop=True)
        
        self.corner_data = team_data
    
    def get_lanzadores_data(self):
        """Obtiene datos de lanzadores de córner SOLO desde y < 1"""
        # CAMBIO: y < 1
        lanzadores = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] < 1)  # ← CAMBIO AQUÍ
        ]
        return lanzadores['playerName'].value_counts()
    
    def get_rematadores_data(self):
        """Obtiene datos de rematadores usando la misma lógica que el mapa de calor"""
        
        remates = self.corner_data[
            (self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])) &
            (self.corner_data['From corner'] == 'Sí')
        ]
        
        # CAMBIO: y < 1
        pases_corner = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] < 1)  # ← CAMBIO AQUÍ
        ]
        
        rematadores_validos = []
        
        for _, remate in remates.iterrows():
            remate_time = remate['timeMin'] * 60 + remate['timeSec']
            
            pase_previo = pases_corner[
                (pases_corner['Match ID'] == remate['Match ID']) &
                (pases_corner['timeMin'] * 60 + pases_corner['timeSec'] >= remate_time - 5) &
                (pases_corner['timeMin'] * 60 + pases_corner['timeSec'] < remate_time)
            ]
            
            if not pase_previo.empty:
                rematadores_validos.append(remate['playerName'])
        
        return pd.Series(rematadores_validos).value_counts()
    
    def get_tipo_lanzamiento_data(self):
        """Obtiene datos de tipos de lanzamiento"""
        # CAMBIO: y < 1
        lanzamientos = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] < 1)  # ← CAMBIO AQUÍ
        ]

        cerrados = len(lanzamientos[lanzamientos['In-swinger'] == 'Sí'])
        abiertos = len(lanzamientos[lanzamientos['Out-swinger'] == 'Sí'])
        planos = len(lanzamientos[lanzamientos['Straight'] == 'Sí'])

        total_lanzamientos = len(lanzamientos)
        tipos_conocidos = cerrados + abiertos + planos
        sin_tipo = total_lanzamientos - tipos_conocidos
        
        return {'Cerrados': cerrados, 'Abiertos': abiertos, 'Planos': planos, 'Sin Tipo': sin_tipo}
    
    def get_primer_contacto_data(self):
        """
        LÓGICA SINCRONIZADA CON ABP7.2 (LADO DERECHO):
        Busca el resultado principal (prioridad Gol > Tiro) por cada saque de esquina.
        """
        primer_contacto_list = []
        
        # 1. Obtener los saques de esquina SOLO del LADO DERECHO (y < 1)
        saques = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') & 
            (self.corner_data['Corner taken'] == 'Sí') & 
            (self.corner_data['y'] < 1)
        ].copy()

        # 2. Por cada saque, buscar el remate igual que en abp7.2
        for _, saque in saques.iterrows():
            match_id = saque['Match ID']
            saque_dt = pd.to_datetime(saque['timeStamp'])
            
            # Buscamos eventos posteriores al saque en el mismo partido
            secuencia = self.corner_data[
                (self.corner_data['Match ID'] == match_id) & 
                (pd.to_datetime(self.corner_data['timeStamp']) > saque_dt)
            ].sort_values('timeStamp').head(12) 
            
            prioridad = {'Goal': 1, 'Attempt Saved': 2, 'Miss': 3, 'Post': 4}
            mejor_evento = None
            min_prioridad = 5

            for _, evento in secuencia.iterrows():
                ev_dt = pd.to_datetime(evento['timeStamp'])
                diff = (ev_dt - saque_dt).total_seconds()
                
                # Ventana de 5 segundos igual que abp7.2
                if diff > 5: break 
                
                # Si el evento es del mismo equipo y es un remate
                if evento['Team ID'] == saque['Team ID'] and evento['Event Name'] in prioridad:
                    prio_actual = prioridad[evento['Event Name']]
                    # Sistema de prioridad: si hay varios remates, el Gol manda
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
        """Determina la zona según las coordenadas x,y"""
        x, y = float(x), float(y)
        
        if x < 70:
            return None
        
        if x >= 83 and x <= 94.2 and y >= 42 and y <= 58:
            return 'zona_6'
        elif x >= 70 and x <= 100 and y >= 75 and y <= 100:
            return 'zona_1'
        elif x >= 70 and x <= 88.5 and y >= 25 and y <= 75:
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
    
    def get_contactos_por_zona(self):
        """Cuenta primer contacto por zona"""
        primer_contacto = self.get_primer_contacto_data()
        zonas_count = {'zona_1': 0, 'zona_2': 0, 'zona_3': 0, 'zona_4': 0, 
                    'zona_5': 0, 'zona_6': 0, 'zona_7': 0}
    
        for contacto in primer_contacto:
            zona = self.get_zona_from_coordinates(contacto['x'], contacto['y'])
            if zona:
                zonas_count[zona] += 1
        
        return zonas_count
    
    def get_matriz_lanzador_rematador(self):
        """Crea una matriz de lanzador vs actor desde y < 1"""
        from collections import defaultdict

        # CAMBIO: y < 1
        lanzadores = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] < 1)  # ← CAMBIO AQUÍ
        ].copy()
        lanzadores['timeStamp'] = pd.to_datetime(lanzadores['timeStamp'])

        remates = self.corner_data[
            self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])
        ]
        duelos_aereos_ganados = self.corner_data[
            (self.corner_data['Event Name'] == 'Aerial') &
            (self.corner_data['outcome'] == 1)
        ]

        acciones_clave = pd.concat([remates, duelos_aereos_ganados]).copy()
        acciones_clave['timeStamp'] = pd.to_datetime(acciones_clave['timeStamp'])
        acciones_clave = acciones_clave.sort_values('timeStamp')

        matriz = defaultdict(lambda: defaultdict(int))

        for _, corner in lanzadores.iterrows():
            lanzador = corner['playerName']
            corner_timestamp = corner['timeStamp']

            acciones_en_ventana = acciones_clave[
                (acciones_clave['Match ID'] == corner['Match ID']) &
                (acciones_clave['timeStamp'] > corner_timestamp) &
                (acciones_clave['timeStamp'] <= corner_timestamp + pd.Timedelta(seconds=6))
            ]

            if not acciones_en_ventana.empty:
                actores_unicos_en_secuencia = acciones_en_ventana['playerName'].unique()
                
                for actor in actores_unicos_en_secuencia:
                    matriz[lanzador][actor] += 1

        lanzadores_list = list(matriz.keys())

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

    def get_secuencia_mas_repetida(self):
        """Encuentra al lanzador más frecuente desde la derecha (y < 1)"""
        # CAMBIO: y < 1
        corners_der = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] < 1)  # ← CAMBIO AQUÍ
        ]
        
        if corners_der.empty:
            return {'lanzador': 'N/A', 'zona': 'N/A', 'golpeo': 'N/A', 'pierna': 'N/A'}

        lanzador_principal = corners_der['playerName'].value_counts().index[0]
        lanzador_data = corners_der[corners_der['playerName'] == lanzador_principal]

        zonas = lanzador_data.apply(
            lambda row: self.get_zona_from_coordinates(row['Pass End X'], row['Pass End Y']),
            axis=1
        ).dropna()
        zona_mas_comun_key = zonas.mode()[0] if not zonas.empty else 'N/A'
        zona_nombre = self.ZONA_NAMES.get(zona_mas_comun_key, 'N/A')

        def get_golpeo(row):
            if row['In-swinger'] == 'Sí': return 'CERRADO'
            if row['Out-swinger'] == 'Sí': return 'ABIERTO'
            if row['Straight'] == 'Sí': return 'PLANO'
            return None

        golpeos = lanzador_data.apply(get_golpeo, axis=1).dropna()
        golpeo_principal = golpeos.mode()[0] if not golpeos.empty else 'N/A'
        
        def get_pierna(row):
            if row['Right footed'] == 'Sí': return 'DERECHA'
            if row['Left footed'] == 'Sí': return 'IZQUIERDA'
            return None

        piernas = lanzador_data.apply(get_pierna, axis=1).dropna()
        pierna_principal = piernas.mode()[0] if not piernas.empty else 'N/A'

        return {
            'lanzador': lanzador_principal,
            'zona': zona_nombre, 
            'golpeo': golpeo_principal,
            'pierna': pierna_principal
        }

    def get_player_shirt_number(self, player_name):
        """Obtiene el dorsal del jugador"""
        try:
            team_players = self.player_stats[self.player_stats['Team Name'] == self.team_filter]
            
            exact_match = team_players[team_players['Match Name'] == player_name]
            if not exact_match.empty:
                return exact_match.iloc[0]['Shirt Number']
            
            from difflib import SequenceMatcher
            best_match = None
            best_score = 0.8
            
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
            img_data = base64.b64decode(match['image_base64'])
            img = Image.open(BytesIO(img_data))
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            data = np.array(img)
            white_mask = (data[:,:,0] > 240) & (data[:,:,1] > 240) & (data[:,:,2] > 240)
            data[white_mask] = [0, 0, 0, 0]
            img = Image.fromarray(data, 'RGBA')
            
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
        """Dibuja recuadro de jugador con ancho dinámico"""
        box_size = sizing_params['box_size']
        
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
        
        longest_line = max(nombre_formateado.split('\n'), key=len)
        num_chars = len(longest_line)

        base_width = 0.05 
        char_increment = 0.009
        box_width = (base_width + num_chars * char_increment) * box_size
        box_height = sizing_params['box_height_factor'] * box_size
        
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
        
        dorsal = self.get_player_shirt_number(player_name)
        if dorsal:
            dorsal_rect_width = min(box_width * 0.5, 0.05)
            dorsal_rect = patches.Rectangle(
                (x - dorsal_rect_width/2, y + box_height*0.15),
                dorsal_rect_width, box_height*0.35,
                facecolor='#E74C3C', edgecolor='white', 
                linewidth=1, zorder=8
            )
            ax.add_patch(dorsal_rect)
            ax.text(x, y + box_height*0.32, str(int(dorsal)), ha='center', va='center', fontsize=max(8, 10 * box_size), fontweight='bold', color='white', zorder=9)
        
        ax.text(x, y - box_height * 0.20, nombre_formateado,
                ha='center', va='center', 
                fontsize=max(4, 6 * box_size), fontweight='bold',
                color='#2C3E50', zorder=8,
                linespacing=0.6)
        
        if count_value > 0:
            ax.text(x, y - box_height * 0.705, str(int(count_value)),
                    ha='center', va='center', fontsize=max(4, 6 * box_size), fontweight='bold',
                    color='white', zorder=11,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='#2C3E50', edgecolor='white', linewidth=1, alpha=1.0))

    def create_sankey_derecha_avanzado(self, ax):
        """Flujo avanzado DESDE LA DERECHA (y < 1) con flujo invertido"""
        ax.clear()

        vertical_offset = -0.08 
        
        # CAMBIO: y < 1
        corners_der = self.corner_data[
            (self.corner_data['Event Name'] == 'Pass') &
            (self.corner_data['Corner taken'] == 'Sí') &
            (self.corner_data['y'] < 1)  # ← CAMBIO AQUÍ
        ]
        
        if corners_der.empty:
            ax.text(0.5, 0.5, 'FLUJO CÓRNERS DERECHA\n(y < 1)\n\n(Sin datos)', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            ax.axis('off')
            return
        
        lanzadores_data = corners_der['playerName'].value_counts()
        tipos_data = self.get_tipo_lanzamiento_data()
        zonas_data = self.get_zonas_de_caida_data()
        rematadores_data = self.get_rematadores_data()
        
        sizing_params = self.calculate_dynamic_sizing(len(lanzadores_data), len(rematadores_data))
        
        # FLUJO INVERTIDO: De derecha a izquierda
        x_pos = [0.93, 0.70, 0.35, 0.08]  # ← INVERTIDO
        
        # NIVEL 1: LANZADORES (ahora a la DERECHA)
        lanzadores_items = list(lanzadores_data.head(5).items())
        colores_jugadores = ['#2C3E50', '#6A1B9A', '#F39C12', '#27AE60', '#8B4513']
        lanzadores_colors = {}
        lanzadores_y_map = {}
        if lanzadores_items:
            lanzadores_y = np.linspace(0.90, 0.15, len(lanzadores_items)) + vertical_offset
            for i, (player_name, count) in enumerate(lanzadores_items):
                player_matches = corners_der[corners_der['playerName'] == player_name]['playerId'].dropna()
                player_id = player_matches.iloc[0] if not player_matches.empty else None
                self.draw_player_box_simple(ax, x_pos[0], lanzadores_y[i], player_name, player_id, sizing_params, count_value=count)
                lanzadores_colors[player_name] = colores_jugadores[i % len(colores_jugadores)]
                lanzadores_y_map[player_name] = lanzadores_y[i]

        # NIVEL 2: TIPOS
        tipos_items = [item for item in tipos_data.items() if item[1] > 0]
        tipos_y = (np.linspace(0.8, 0.2, len(tipos_items)) + vertical_offset) if tipos_items else []
        tipos_y_map = {item[0]: y for item, y in zip(tipos_items, tipos_y)}
        tipos_colors = {'Cerrados': '#E74C3C', 'Abiertos': '#3498DB', 'Planos': '#F39C12', 'Sin Tipo': '#95a5a6'}
        for i, (tipo, count) in enumerate(tipos_items):
            color = tipos_colors[tipo]
            ax.text(x_pos[1], tipos_y[i], f'{tipo}\n({count})', ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=10, 
                   bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.9, edgecolor='white', linewidth=2))

        # NIVEL 3: ZONAS
        zonas_con_datos = sorted([item for item in zonas_data.items() if item[1] > 0], key=lambda x: x[1], reverse=True)
        zonas_y = (np.linspace(0.85, 0.15, len(zonas_con_datos)) + vertical_offset) if zonas_con_datos else []
        zonas_y_map = {item[0]: y for item, y in zip(zonas_con_datos, zonas_y)}
        zonas_colors_map = {'zona_1': '#FF6B6B', 'zona_2': '#4ECDC4', 'zona_3': '#45B7D1', 'zona_4': '#96CEB4', 
                           'zona_5': '#FECA57', 'zona_6': '#FF9FF3', 'zona_7': '#54A0FF'}
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
            ax.text(x_pos[2], zonas_y[i], f"{zona_nombre}\n({count})", ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=6, 
                   path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])

        # NIVEL 4: REMATADORES (ahora a la IZQUIERDA)
        rematadores_items = list(rematadores_data.head(8).items())
        rematadores_y_map = {}
        if rematadores_items:
            rematadores_y = np.linspace(0.85, 0.15, len(rematadores_items)) + vertical_offset
            for i, (player_name, count) in enumerate(rematadores_items):
                remates = self.corner_data[(self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])) & 
                                          (self.corner_data['From corner'] == 'Sí') & 
                                          (self.corner_data['playerName'] == player_name)]
                player_id = remates['playerId'].iloc[0] if not remates.empty else None
                self.draw_player_box_simple(ax, x_pos[3], rematadores_y[i], player_name, player_id, sizing_params, count_value=count)
                rematadores_y_map[player_name] = rematadores_y[i]
        
        # CONEXIONES INVERTIDAS
        for _, corner in corners_der.iterrows():
            lanzador = corner['playerName']
            
            if lanzador not in lanzadores_y_map:
                continue

            # LANZADOR -> TIPO (ahora de derecha a izquierda)
            if corner['In-swinger'] == 'Sí': tipo = 'Cerrados'
            elif corner['Out-swinger'] == 'Sí': tipo = 'Abiertos'
            elif corner['Straight'] == 'Sí': tipo = 'Planos'
            else: tipo = 'Sin Tipo'
            
            if tipo in tipos_y_map:
                arrow1 = patches.FancyArrowPatch(
                    (x_pos[0] - 0.05, lanzadores_y_map[lanzador]),  # ← INVERTIDO
                    (x_pos[1] + 0.06, tipos_y_map[tipo]),           # ← INVERTIDO
                    connectionstyle="arc3,rad=-0.1",  # ← Rad negativo para curva inversa
                    color=lanzadores_colors[lanzador], alpha=0.7, 
                    linewidth=2, zorder=1)
                ax.add_patch(arrow1)

            # TIPO -> ZONA
            zona_caida = self.get_zona_from_coordinates(corner['Pass End X'], corner['Pass End Y'])
            if zona_caida in zonas_y_map and tipo in tipos_y_map:
                rad = {'Cerrados': -0.2, 'Abiertos': 0.2, 'Planos': 0.0, 'Sin Tipo': -0.1}.get(tipo, 0)  # ← Invertidos
                arrow2 = patches.FancyArrowPatch(
                    (x_pos[1] - 0.06, tipos_y_map[tipo]),     # ← INVERTIDO
                    (x_pos[2] + 0.05, zonas_y_map[zona_caida]), # ← INVERTIDO
                    connectionstyle=f"arc3,rad={rad}", 
                    color=lanzadores_colors[lanzador], alpha=0.7, 
                    linewidth=2, zorder=1)
                ax.add_patch(arrow2)

            # ZONA -> REMATADOR
            corner_time = corner['timeMin'] * 60 + corner['timeSec']
            remates_posteriores = self.corner_data[
                (self.corner_data['Event Name'].isin(['Miss', 'Goal', 'Attempt Saved', 'Post'])) &
                (self.corner_data['From corner'] == 'Sí') &
                (self.corner_data['Match ID'] == corner['Match ID']) &
                (self.corner_data['timeMin'] * 60 + self.corner_data['timeSec'] > corner_time) &
                (self.corner_data['timeMin'] * 60 + self.corner_data['timeSec'] <= corner_time + 5)
            ]

            if not remates_posteriores.empty:
                primer_remate = remates_posteriores.iloc[0]
                rematador = primer_remate['playerName']

                if zona_caida in zonas_y_map and rematador in rematadores_y_map:
                    arrow3 = patches.FancyArrowPatch(
                        (x_pos[2] - 0.05, zonas_y_map[zona_caida]),    # ← INVERTIDO
                        (x_pos[3] + 0.05, rematadores_y_map[rematador]), # ← INVERTIDO
                        connectionstyle="arc3,rad=-0.1",  # ← Rad negativo
                        color=lanzadores_colors[lanzador], alpha=0.7, 
                        linewidth=2, zorder=1)
                    ax.add_patch(arrow3)

        # Títulos (orden invertido)
        titulos = ['LANZADORES', 'TIPOS', 'ZONAS', 'REMATADORES']
        colores_modernos = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
        for i, titulo in enumerate(titulos):
            ax.text(x_pos[i], 1.02 + vertical_offset, titulo, ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=10, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=colores_modernos[i], alpha=0.95, edgecolor='white', linewidth=2))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0 + vertical_offset, 1.02 + vertical_offset)
        ax.set_title('FLUJO CÓRNERS DESDE LA DERECHA', fontsize=10, fontweight='bold', color='black', pad=20)
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
        """Crea el reporte con Sankey y ranking de juego aéreo - NUEVA MAQUETACIÓN"""
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        # FILA 1: Tabla + Ranking + Campo
        gs_fila1 = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.05, 
                            top=0.9, bottom=0.5,
                            left=0.05, right=0.95,
                            width_ratios=[2, 1, 1.2])  # ← ORDEN CAMBIADO

        # FILA 2: Flujo + Mapa de calor
        gs_fila2 = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.11,
                                top=0.45, bottom=0.05,
                                left=0.05, right=0.95,
                                width_ratios=[2, 3])  # ← ORDEN CAMBIADO
        
        # Fondo
        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25)
            ax_bg.axis('off')
        
        # Título principal
        fig.suptitle('CÓRNERS OFENSIVOS', fontsize=18, fontweight='bold', 
                    color='#1e3d59', y=0.95, family='serif')
        
        # Logo del equipo
        if team_filter and (team_logo := self.load_team_logo(team_filter)) is not None:
            ax_team = fig.add_axes([0.92, 0.90, 0.06, 0.06])
            ax_team.imshow(team_logo, aspect='auto')
            ax_team.axis('off')
        
        # === FILA 1, COLUMNA 1: TABLA (AHORA PRIMERA) ===
        ax_tabla = fig.add_subplot(gs_fila1[0, 0])
        ax_tabla.set_facecolor('none')
        ax_tabla.axis('off')

        matriz, lanzadores, rematadores = self.get_matriz_lanzador_rematador()

        if matriz and lanzadores and rematadores:
            
            def format_lanzador_name(name, max_len=12):
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
                words = name.split()
                return '\n'.join(words)

            table_data = []

            rematadores_limitados = rematadores[:10]
            header_nombres = [''] + [format_vertical_header(rem) for rem in rematadores_limitados]
            table_data.append(header_nombres)

            lanzadores_limitados = lanzadores[:8]
            for lanzador in lanzadores_limitados:
                row = [format_lanzador_name(lanzador)]
                for rematador in rematadores_limitados:
                    value = matriz.get(lanzador, {}).get(rematador, 0)
                    row.append(str(value) if value > 0 else '')
                table_data.append(row)
            
            if len(table_data) > 1:
                col_widths = [0.20] + [0.08] * len(rematadores_limitados)
                
                table = ax_tabla.table(cellText=table_data,
                                    cellLoc='center', loc='center', 
                                    bbox=[0, 0, 1, 1],
                                    colWidths=col_widths)
                
                table.auto_set_font_size(False)
                table.set_fontsize(7)
                table.scale(1, 2.8)

                all_values = []
                for row in table_data[1:]:
                    for val in row[1:]:
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
                            cell.set_facecolor('none')
                            cell.set_edgecolor('none')
                            cell.get_text().set_text('')
                        
                        elif i == 0 and j > 0:
                            cell.set_facecolor('#2C3E50')
                            cell.get_text().set_rotation(45)
                            cell.get_text().set_color('white')
                            cell.get_text().set_weight('bold')
                            cell.get_text().set_va('center')
                            cell.get_text().set_ha('center')
                            cell.get_text().set_fontsize(9)
                        
                        elif i > 0 and j == 0:
                            cell.set_facecolor('#E74C3C')
                            cell.get_text().set_color('white')
                            cell.get_text().set_weight('bold')
                            cell.get_text().set_ha('center')
                            cell.get_text().set_va('center')
                            cell.get_text().set_fontsize(9)
                        
                        elif i > 0 and j > 0:
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
                    
                    from matplotlib.patches import Rectangle

                    rect_lanzadores = Rectangle((0.01, 0.94), 0.03, 0.02, 
                                                facecolor='#E74C3C', edgecolor='black', linewidth=1,
                                                alpha=1.0, transform=ax_tabla.transAxes)
                    ax_tabla.add_patch(rect_lanzadores)
                    ax_tabla.text(0.05, 0.945, 'LANZADORES', transform=ax_tabla.transAxes,
                                ha='left', va='center', fontsize=6, fontweight='bold')

                    rect_rematadores = Rectangle((0.01, 0.91), 0.03, 0.02,
                                                facecolor='#2C3E50', edgecolor='black', linewidth=1,
                                                alpha=1.0, transform=ax_tabla.transAxes)
                    ax_tabla.add_patch(rect_rematadores)
                    ax_tabla.text(0.05, 0.915, 'REMATADORES', transform=ax_tabla.transAxes,
                                ha='left', va='center', fontsize=6, fontweight='bold')
        
        # === FILA 1, COLUMNA 2: RANKING (AHORA SEGUNDO) ===
        ax_ranking = fig.add_subplot(gs_fila1[0, 1])
        self.create_aerial_ranking(ax_ranking)
        
        # === FILA 1, COLUMNA 3: CAMPO SUPERIOR (AHORA TERCERO) ===
        ax_campo_superior = fig.add_subplot(gs_fila1[0, 2])
        ax_campo_superior.set_facecolor('none')
        pitch_superior = VerticalPitch(half=True, pitch_type='opta', pitch_color='none', 
                                    line_color='black', linewidth=3)
        pitch_superior.draw(ax=ax_campo_superior)

        secuencia = self.get_secuencia_mas_repetida()
        photos_data = self.load_player_photos()
        player_photo = self.get_player_photo_without_dorsal(secuencia['lanzador'], photos_data)

        if player_photo is not None:
            photo_ax = ax_campo_superior.inset_axes([0.01, 0.10, 0.4, 0.5])
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

        ax_campo_superior.set_title('DESDE LA DERECHA', fontsize=12, fontweight='bold', pad=20)  # ← CAMBIADO

        # Balón en córner DERECHO
        if (ball_img := self.load_ball_image()) is not None:
            ball_box = OffsetImage(ball_img, zoom=0.15)
            ball_ab = AnnotationBbox(ball_box, (0.5, 99.5), frameon=False)  # ← CAMBIADO A ESQUINA DERECHA
            ax_campo_superior.add_artist(ball_ab)
        
        # === FILA 2, COLUMNA 1: FLUJO (AHORA IZQUIERDA) ===
        ax_sankey = fig.add_subplot(gs_fila2[0, 0])
        ax_sankey.set_facecolor('none')
        ax_sankey.axis('off')
        self.create_sankey_derecha_avanzado(ax_sankey)  # ← USANDO FUNCIÓN DERECHA
        
        # === FILA 2, COLUMNA 2: MAPA DE CALOR (AHORA DERECHA) ===
        ax_campo_inferior = fig.add_subplot(gs_fila2[0, 1])
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
                count = zonas_data.get(zona, 0)
                if max_remates > 0:
                    intensidad = 0.2 + (count / max_remates) * 0.8
                else:
                    intensidad = 0.2
                color_zona = colormap(intensidad)
                alpha_zona = 1.0 if zona == 'zona_6' else 0.8
                rect = patches.Rectangle((y_min, x_min), height, width, 
                                    linewidth=3, edgecolor='black',
                                    facecolor=color_zona, alpha=alpha_zona)
                ax_campo_inferior.add_patch(rect)
                center_y, center_x = y_min + height/2, x_min + width/2
                text_color = 'white' if intensidad > 0.6 else 'black'
                ax_campo_inferior.text(center_y, center_x, str(count), 
                                    fontsize=16, fontweight='bold', ha='center', va='center',
                                    color=text_color)
        
        plt.tight_layout()
        return fig

def seleccionar_equipo_interactivo():
    """Selección interactiva de equipo"""
    try:
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/abp_events.parquet")
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
        
        analyzer = CornersOffensiveReport(team_filter=equipo)
        
        if (fig := analyzer.create_corners_report(team_filter=equipo)):
            plt.show()
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_corners_derecha_{equipo_filename}.pdf"
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1, 
                       facecolor='white', dpi=300)
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
                output_path = f"reporte_corners_derecha_{equipo_filename}.pdf"
                fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1, 
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

def verificar_assets():
    """Verifica la disponibilidad de assets necesarios"""
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
        pass
    else:
        print("⚠️  No hay escudos en el directorio")

if __name__ == "__main__":
    pass
    try:
        verificar_assets()
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/abp_events.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        if equipos:
            pass
    except Exception as e:
        print(f"❌ Error al inicializar: {e}")
    
    main()