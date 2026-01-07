import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import numpy as np
from datetime import datetime
import warnings
import os
import re
import unicodedata
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from matplotlib import patheffects
from collections import Counter, defaultdict
from matplotlib.gridspec import GridSpecFromSubplotSpec
import json
import base64
from io import BytesIO
from PIL import Image
from difflib import SequenceMatcher
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN


warnings.filterwarnings('ignore')

class AnalizadorSaquesBanda:
    @staticmethod
    def safe_to_float(value):
        """Intenta convertir a float, si falla, devuelve np.nan"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return np.nan

    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/open_play_events.parquet"):
        """
        Inicializa el analizador de saques de banda ofensivos
        """
        self.data_path = data_path
        self.df = None
        self.df_complete = None
        self.team_filter = None
        self.df_players = None
        self.photos_data = None

    def find_visual_patterns_clustering(self, sequences_list, max_patterns=3):
        """
        Encuentra patrones visuales similares usando clustering.
        VERSIÓN AFINADA: Más estricto para asegurar patrones coherentes.
        """
        if not sequences_list or len(sequences_list) < 3: # Necesitamos al menos 3 para encontrar un patrón
            return []
        
        def sequence_distance(seq1, seq2):
            len_diff = abs(len(seq1) - len(seq2))
            length_penalty = len_diff * 15 # Aumentamos ligeramente la penalización

            min_len = min(len(seq1), len(seq2))
            total_dist = 0
            
            for i in range(min_len):
                p1, p2 = seq1[i], seq2[i]
                coords_start1, coords_start2 = [p1.get('x'), p1.get('y')], [p2.get('x'), p2.get('y')]
                coords_end1, coords_end2 = [p1.get('end_x'), p1.get('end_y')], [p2.get('end_x'), p2.get('end_y')]
                
                dist_start, dist_end = 0, 0
                if not any(pd.isna(c) for c in coords_start1 + coords_start2):
                    dist_start = euclidean(coords_start1, coords_start2)
                if not any(pd.isna(c) for c in coords_end1 + coords_end2):
                    dist_end = euclidean(coords_end1, coords_end2)

                total_dist += (dist_start + dist_end) / 2
            
            avg_geometric_dist = total_dist / min_len if min_len > 0 else 0
            return avg_geometric_dist + length_penalty

        n = len(sequences_list)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = sequence_distance(sequences_list[i], sequences_list[j])
                distance_matrix[i, j] = distance_matrix[j, i] = dist
        
        distance_matrix = np.nan_to_num(distance_matrix, nan=1000.0)

        clustering = DBSCAN(eps=18, min_samples=2, metric='precomputed').fit(distance_matrix)
        
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label != -1:
                clusters[label].append(sequences_list[idx])
        
        sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
        
        patterns = []
        for cluster_seqs in sorted_clusters[:max_patterns]:
            if len(cluster_seqs) >= 3:
                patterns.append({
                    'count': len(cluster_seqs),
                    'all_sequences': cluster_seqs,
                    'average_sequence': self._calculate_average_sequence(cluster_seqs)
                })
        
        return patterns
    
    def _calculate_average_sequence(self, sequences):
        """
        Calcula la secuencia promedio de un grupo.
        VERSIÓN CORREGIDA: Usa nanmean para ignorar NaNs de forma segura.
        """
        if not sequences: return []
        
        most_common_length = Counter(len(s) for s in sequences).most_common(1)[0][0]
        
        avg_seq = []
        for i in range(most_common_length):
            filtered_seqs = [s for s in sequences if len(s) == most_common_length]
            
            if not filtered_seqs: continue # Evitar errores si no hay secuencias de la longitud común

            x_vals = [s[i]['x'] for s in filtered_seqs]
            y_vals = [s[i]['y'] for s in filtered_seqs]
            end_x_vals = [s[i]['end_x'] for s in filtered_seqs]
            end_y_vals = [s[i]['end_y'] for s in filtered_seqs]
            
            avg_seq.append({
                'x': np.nanmean(x_vals), 'y': np.nanmean(y_vals),
                'end_x': np.nanmean(end_x_vals), 'end_y': np.nanmean(end_y_vals),
                'is_aerial': any(s[i].get('is_aerial') for s in filtered_seqs)
            })
        return avg_seq

    def load_data(self, team_filter=None):
        """Carga y filtra los datos de eventos y jugadores"""
        try:
            print(f"📂 Cargando datos desde {self.data_path}...")
            # Cargamos y ORDENAMOS los datos de una vez para asegurar la secuencia
            self.df_complete = pd.read_parquet(self.data_path).sort_values(
                by=['Match ID', 'periodId', 'timeStamp']
            ).reset_index(drop=True)
            print(f"✅ Datos cargados y ordenados: {len(self.df_complete)} eventos totales")

            try:
                players_path = "extraccion_opta/datos_opta_parquet/player_stats.parquet"
                self.df_players = pd.read_parquet(players_path)
                print(f"✅ Datos de jugadores cargados: {len(self.df_players)} registros")
                # Procesamos las posiciones de los jugadores inmediatamente
                self._preprocess_player_positions()
            except Exception as e:
                print(f"⚠️ Aviso: No se pudieron cargar los datos de jugadores. Nombres/posiciones no estarán disponibles. Error: {e}")
                self.df_players = None

            if team_filter:
                self.team_filter = team_filter
                self.df = self.df_complete[self.df_complete['Team Name'] == team_filter].copy()
                print(f"🔍 Filtrado por equipo '{team_filter}': {len(self.df)} eventos")
            else:
                self.df = self.df_complete.copy()

            return True

        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_most_common_positional_sequence(self, pattern_sequences):
        """
        Analiza un cluster de secuencias y devuelve la cadena de demarcaciones más común.
        """
        if not pattern_sequences:
            return "N/A", 0

        positional_chains = []
        for seq in pattern_sequences:
            chain = []
            match_id = seq[0].get('match_id')

            # 1. Posición del lanzador
            thrower_id = seq[0].get('thrower_id')
            _, thrower_pos = self.get_player_info(thrower_id, match_id)
            chain.append(thrower_pos if thrower_pos else 'DESC')

            # 2. Posiciones de los receptores
            for pass_data in seq:
                receiver_id = pass_data.get('receiver_id')
                _, receiver_pos = self.get_player_info(receiver_id, match_id)
                chain.append(receiver_pos if receiver_pos else 'DESC')
            
            positional_chains.append(" → ".join(chain))

        if not positional_chains:
            return "N/A", 0

        # Contamos cuál es la cadena más común
        most_common = Counter(positional_chains).most_common(1)[0]
        return most_common[0], (most_common[1] / len(positional_chains)) * 100

    def _preprocess_player_positions(self):
        """
        Limpia y enriquece los datos de posición de los jugadores.
        - Traduce posiciones de inglés a español.
        - Calcula y asigna la posición más frecuente para jugadores sin posición definida (suplentes/vacío).
        """
        if self.df_players is None:
            return

        print("🔄 Pre-procesando posiciones de jugadores...")
        
        # 1. Diccionario de traducción
        pos_map = {
            'Goalkeeper': 'POR', 'Defender': 'DEF', 'Right Back': 'LD', 'Left Back': 'LI',
            'Centre Back': 'DFC', 'Midfielder': 'MED', 'Defensive Midfielder': 'MCD',
            'Attacking Midfielder': 'MCO', 'Right Wing': 'ED', 'Left Wing': 'EI',
            'Forward': 'DEL', 'Striker': 'DC', 'Substitute': 'Sustituto',
            'Wing Back': 'CAR'  
        }
        self.df_players['Position_ES'] = self.df_players['Position'].map(pos_map).fillna('Desconocido')

        # 2. Calcular la posición más frecuente de cada jugador (ignorando "Sustituto")
        valid_positions = self.df_players[self.df_players['Position_ES'] != 'Sustituto']
        # Usamos mode() para encontrar el valor más repetido. dropna() por si un jugador solo ha sido suplente.
        most_common_pos = valid_positions.groupby('Player ID')['Position_ES'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Desconocido').to_dict()

        # 3. Rellenar posiciones 'Sustituto' o 'Desconocido' con la más frecuente
        def fill_pos(row):
            if row['Position_ES'] in ['Sustituto', 'Desconocido']:
                return most_common_pos.get(row['Player ID'], 'Desconocido')
            return row['Position_ES']
        
        self.df_players['Position_Final'] = self.df_players.apply(fill_pos, axis=1)
        print("✅ Posiciones procesadas.")
        print("\n--- DEBUG: Comprobando el procesamiento de posiciones ---")
        # Mostramos las primeras 20 filas para comparar la columna original, la traducida y la final
        print(self.df_players[['Position', 'Position_ES', 'Position_Final']].head(20))
        
        print("\nConteo de valores en 'Position_Final':")
        print(self.df_players['Position_Final'].value_counts())
        print("------------------------------------------------------\n")


    def get_player_info(self, player_id, match_id):
        """
        Obtiene el nombre y la posición final de un jugador para un partido específico.
        """
        if self.df_players is None or pd.isna(player_id):
            return "Desconocido (sin datos)", ""

        # Búsqueda principal: jugador en el partido específico
        player_data = self.df_players[
            (self.df_players['Player ID'] == player_id) &
            (self.df_players['Match ID'] == match_id)
        ]
        
        if not player_data.empty:
            player_row = player_data.iloc[0]
            name = player_row.get('Match Name', f'ID: {player_id}')
            pos = player_row.get('Position_Final', 'No Encontrada')
            return name.split()[-1], pos

        # Búsqueda de respaldo: si no estaba en ese partido
        player_general_data = self.df_players[self.df_players['Player ID'] == player_id]
        
        if not player_general_data.empty:
            player_row = player_general_data.iloc[0]
            name = player_row.get('Match Name', f'ID: {player_id}')
            pos = player_row.get('Position_Final', 'No Encontrada')
            return name.split()[-1], pos

        # Si no se encuentra en ninguna parte
        return "Desconocido", ""
    
    def load_player_photos(self):
        """Carga las fotos de jugadores desde el JSON"""
        if self.photos_data is None:
            try:
                print("📂 Intentando cargar jugadores_optimizados.json...")
                with open('assets/jugadores_optimizados.json', 'r', encoding='utf-8') as f:
                    self.photos_data = json.load(f)
                print(f"✅ Cargadas {len(self.photos_data)} fotos de jugadores")
            except FileNotFoundError:
                print("⚠️ No se encontró el archivo assets/jugadores_optimizados.json")
                self.photos_data = []
            except Exception as e:
                print(f"❌ Error cargando fotos: {e}")
                self.photos_data = []
        return self.photos_data

    def extract_names_parts(self, name):
        """Extrae y normaliza las partes de un nombre"""
        def normalize_name(name):
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
            return {'full': '', 'first_name': '', 'last_name': '', 'all_parts': []}
            
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
        """Busca la mejor coincidencia de un jugador en los datos de fotos"""
        player_parts = self.extract_names_parts(player_name)
        if not player_parts['full']:
            return None

        team_players = []
        if team_filter:
            palabras_ignorar = {'cf', 'fc', 'cd', 'ud', 'rcd', 'rc', 'ca', 'de', 'del', 'la', 'las', 'el', 'los'}
            
            def normalize_word(word):
                word = unicodedata.normalize('NFD', word)
                word = ''.join(char for char in word if unicodedata.category(char) != 'Mn')
                return word.lower().strip()
            
            palabras_equipo = team_filter.split()
            palabras_equipo_norm = [normalize_word(p) for p in palabras_equipo 
                                    if normalize_word(p) not in palabras_ignorar and len(normalize_word(p)) > 2]
            
            palabras_equipo_ordenadas = sorted(palabras_equipo_norm, key=len, reverse=True)
            
            for photo_entry in photos_data:
                photo_team = photo_entry.get('team_name', '')
                if not photo_team:
                    continue
                
                palabras_photo_team = photo_team.split()
                palabras_photo_norm = [normalize_word(p) for p in palabras_photo_team]
                
                match_encontrado = False
                for palabra_buscar in palabras_equipo_ordenadas:
                    if palabra_buscar in palabras_photo_norm:
                        match_encontrado = True
                        break
                
                if match_encontrado:
                    team_players.append(photo_entry)
            
            if not team_players:
                team_filter_norm = normalize_word(team_filter.replace(' ', ''))
                
                for photo_entry in photos_data:
                    photo_team = photo_entry.get('team_name', '')
                    if not photo_team:
                        continue
                    
                    photo_team_norm = normalize_word(photo_team.replace(' ', ''))
                    from difflib import SequenceMatcher
                    similarity = SequenceMatcher(None, team_filter_norm, photo_team_norm).ratio()
                    
                    if similarity > 0.7:
                        team_players.append(photo_entry)
                
                if not team_players:
                    return None
        else:
            team_players = photos_data

        player_words = [w for w in player_parts['all_parts'] if len(w) >= 3]
        player_words_sorted = sorted(player_words, key=len, reverse=True)

        for palabra_buscar in player_words_sorted:
            for photo_entry in team_players:
                photo_name = photo_entry.get('player_name', '')
                photo_parts = self.extract_names_parts(photo_name)
                photo_words = [w for w in photo_parts['all_parts'] if len(w) >= 3]
                
                if palabra_buscar in photo_words:
                    return photo_entry
                
                if len(palabra_buscar) > 5:
                    for ph_word in photo_words:
                        if len(ph_word) > 5:
                            distance = self.levenshtein_distance(palabra_buscar, ph_word)
                            if distance == 1:
                                return photo_entry
        
        candidates = []
        
        for photo_entry in team_players:
            photo_name = photo_entry.get('player_name', '')
            photo_parts = self.extract_names_parts(photo_name)
            photo_words = [w for w in photo_parts['all_parts'] if len(w) >= 3]
            
            matches = []
            for p_word in player_words:
                for ph_word in photo_words:
                    if p_word == ph_word:
                        matches.append(p_word)
                    elif len(p_word) > 5 and len(ph_word) > 5:
                        distance = self.levenshtein_distance(p_word, ph_word)
                        if distance <= 2:
                            matches.append(p_word)
            
            if matches:
                candidates.append({
                    'entry': photo_entry,
                    'matches': matches,
                    'match_count': len(matches)
                })
        
        if len(candidates) == 0:
            return None
        elif len(candidates) == 1:
            return candidates[0]['entry']
        else:
            best_candidates = sorted(candidates, key=lambda x: x['match_count'], reverse=True)
            
            if best_candidates[0]['match_count'] > best_candidates[1]['match_count']:
                return best_candidates[0]['entry']
            
            for candidate in best_candidates:
                photo_parts = self.extract_names_parts(candidate['entry']['player_name'])
                
                for p_word in player_parts['all_parts']:
                    if len(p_word) <= 3:
                        for ph_word in photo_parts['all_parts']:
                            if ph_word.startswith(p_word):
                                return candidate['entry']
            
            return best_candidates[0]['entry']

    def get_player_photo(self, player_name):
        """Obtiene la foto procesada de un jugador"""
        if self.photos_data is None: 
            self.load_player_photos()
        
        if not self.photos_data: 
            return None
        
        match = self.match_player_name(player_name, self.photos_data, self.team_filter)
        if not match: 
            return None
        
        try:
            import base64
            from PIL import Image
            from io import BytesIO
            
            if 'image_base64' not in match:
                return None
            
            img_data = base64.b64decode(match['image_base64'])
            img = Image.open(BytesIO(img_data)).convert('RGBA')
            data = np.array(img)
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
            
            result = data.astype(np.float32) / 255.0
            return result
        
        except Exception as e:
            return None
    
    def extract_sequences_after_throw_in(self, throw_ins_list):
        """
        Extrae secuencias de pases después de cada saque de banda.
        VERSIÓN MEJORADA: Lógica más robusta y directa para capturar más secuencias.
        """
        sequences_with_players = []
        
        for throw_in in throw_ins_list:
            match_id = throw_in['match_id']
            original_index = throw_in['original_index']
            team_name = throw_in['team_name']
            throw_timestamp = pd.to_datetime(self.df_complete.loc[original_index, 'timeStamp'])

            # Si el saque falla, la secuencia está vacía
            if throw_in['outcome'] == 0:
                sequences_with_players.append([])
                continue

            # Iniciar la secuencia con el propio saque de banda
            current_sequence = [{
                'x': throw_in['x'], 'y': throw_in['y'],
                'end_x': throw_in['end_x'], 'end_y': throw_in['end_y'],
                'passer_id': throw_in['thrower_id'],
                'receiver_id': throw_in.get('receiver_id'), # receiver_id ya viene de extract_throw_in_sequences
                'match_id': match_id,
                'thrower_id': throw_in['thrower_id'],
                'is_aerial': False 
            }]

            last_event_index = original_index
            sequence_ended = False

            # Bucle para encontrar hasta 3 pases más (total 4 eventos en la secuencia)
            for _ in range(3):
                if sequence_ended: break

                # Buscar eventos posteriores al último evento de la secuencia
                subsequent_events = self.df_complete[
                    (self.df_complete['Match ID'] == match_id) &
                    (self.df_complete.index > last_event_index)
                ].head(15) # Miramos los siguientes 15 eventos

                next_pass_found = False
                for idx, event in subsequent_events.iterrows():
                    event_team = event.get('Team Name')
                    event_type = event.get('Event Name')
                    event_outcome = event.get('outcome')
                    event_time = pd.to_datetime(event.get('timeStamp'))

                    # Condiciones para terminar la secuencia
                    if event_team != team_name or (event_time - throw_timestamp).total_seconds() > 10:
                        sequence_ended = True
                        break
                    
                    # Si encontramos un pase fallido, la secuencia termina DESPUÉS de este
                    if event_type == 'Pass' and event_outcome == 0:
                        sequence_ended = True
                        break # No incluimos el pase fallido

                    # Si es un pase válido del mismo equipo, lo procesamos
                    if event_type in ['Pass', 'Aerial']:
                        # Buscamos quién recibe este pase (siguiente evento del equipo)
                        receiver_event = self.df_complete[
                            (self.df_complete['Match ID'] == match_id) &
                            (self.df_complete.index > idx) &
                            (self.df_complete['Team Name'] == team_name)
                        ].head(1)

                        if not receiver_event.empty:
                            receiver = receiver_event.iloc[0]
                            current_sequence.append({
                                'x': self.safe_to_float(event.get('x')),
                                'y': self.safe_to_float(event.get('y')),
                                'end_x': self.safe_to_float(event.get('Pass End X')),
                                'end_y': self.safe_to_float(event.get('Pass End Y')),
                                'passer_id': event.get('playerId'),
                                'receiver_id': receiver.get('playerId'),
                                'match_id': match_id,
                                'thrower_id': throw_in['thrower_id'],
                                'is_aerial': event_type == 'Aerial'
                            })
                            last_event_index = receiver.name # Actualizamos el índice al del receptor
                            next_pass_found = True
                            break # Salimos del bucle interior para buscar el siguiente pase

                if not next_pass_found:
                    break # Si no se encontró un siguiente pase, la secuencia termina
            
            if current_sequence:
                sequences_with_players.append(current_sequence)
        
        return sequences_with_players

    def get_most_common_players_by_position(self, all_seqs):
        """
        Obtiene los jugadores más comunes en cada posición, priorizando:
        1. Frecuencia total del jugador en esa posición
        2. Si hay empate, el que aparece en más secuencias totales
        
        Retorna: Lista de jugadores en orden [Lanzador, Receptor1, Receptor2, ...]
        """
        from collections import Counter, defaultdict
        
        # Estructura: {posicion: {player_id: {count: X, sequences: set()}}}
        position_players = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'sequences': set(), 'info': None}))
        
        for seq_idx, seq in enumerate(all_seqs):
            if len(seq) > 0:
                match_id = seq[0].get('match_id')
                
                # Posición 0: Lanzador
                thrower_id = seq[0].get('thrower_id')
                thrower_shirt = seq[0].get('thrower_shirt', '?')
                thrower_name, thrower_pos = self.get_player_info(thrower_id, match_id)
                
                position_players[0][thrower_id]['count'] += 1
                position_players[0][thrower_id]['sequences'].add(seq_idx)
                position_players[0][thrower_id]['info'] = (thrower_name, thrower_shirt, thrower_pos)
                
                # Posiciones 1+: Receptores
                for pass_idx, pass_data in enumerate(seq):
                    receiver_id = pass_data.get('receiver_id')
                    if receiver_id and not pd.isna(receiver_id):
                        receiver_name, receiver_pos = self.get_player_info(receiver_id, match_id)
                        receiver_shirt = pass_data.get('receiver_shirt', '?')
                        
                        pos = pass_idx + 1  # Posición 1, 2, 3...
                        position_players[pos][receiver_id]['count'] += 1
                        position_players[pos][receiver_id]['sequences'].add(seq_idx)
                        position_players[pos][receiver_id]['info'] = (receiver_name, receiver_shirt, receiver_pos)
        
        # Obtener el jugador más común en cada posición
        sequence_players = []
        last_player_id = None  # 🔥 NUEVO: Para evitar repetidos consecutivos
        
        for pos in sorted(position_players.keys()):
            players_in_pos = position_players[pos]
            
            if players_in_pos:
                # Ordenar por: 1) count en esta posición, 2) número de secuencias totales
                sorted_players = sorted(
                    players_in_pos.items(),
                    key=lambda x: (x[1]['count'], len(x[1]['sequences'])),
                    reverse=True
                )
                
                # 🔥 NUEVO: Buscar el mejor jugador que NO sea el anterior
                best_player_id, best_player_data = None, None
                
                for player_id, player_data in sorted_players:
                    # Si es diferente al anterior, lo usamos
                    if player_id != last_player_id:
                        best_player_id = player_id
                        best_player_data = player_data
                        break
                
                # Si todos son iguales al anterior (caso raro), usar el más común de todos modos
                if best_player_id is None:
                    best_player_id, best_player_data = sorted_players[0]
                
                player_name, player_shirt, player_pos = best_player_data['info']
                
                sequence_players.append({
                    'player_name': player_name,
                    'shirt_number': player_shirt,
                    'position': player_pos,
                    'frequency': best_player_data['count'],
                    'sequences_count': len(best_player_data['sequences'])
                })
                
                # 🔥 NUEVO: Actualizar el último jugador
                last_player_id = best_player_id
        
        return sequence_players
    
    def calculate_throw_speed(self, throw_in_idx, throw_in_timestamp):
        """
        Calcula si el saque es rápido (<5s) o lento (>=5s)
        Compara el timestamp con el evento anterior del mismo partido
        """
        if pd.isna(throw_in_timestamp):
            return 'Desconocido'
        
        # Convertir timestamp a datetime si es string
        if isinstance(throw_in_timestamp, str):
            try:
                throw_in_timestamp = pd.to_datetime(throw_in_timestamp)
            except:
                return 'Desconocido'
        
        # Buscar el evento anterior en el mismo partido
        if throw_in_idx > 0 and throw_in_idx in self.df_complete.index:
            current_match = self.df_complete.loc[throw_in_idx, 'Match ID']
            
            # Buscar hacia atrás hasta encontrar un evento del mismo partido
            for prev_idx in range(throw_in_idx - 1, max(0, throw_in_idx - 20), -1):
                if prev_idx in self.df_complete.index:
                    prev_match = self.df_complete.loc[prev_idx, 'Match ID']
                    prev_timestamp = self.df_complete.loc[prev_idx, 'timeStamp']
                    
                    if prev_match == current_match and pd.notna(prev_timestamp):
                        # Convertir timestamp anterior
                        if isinstance(prev_timestamp, str):
                            try:
                                prev_timestamp = pd.to_datetime(prev_timestamp)
                            except:
                                continue
                        
                        # Calcular diferencia en segundos
                        time_diff = (throw_in_timestamp - prev_timestamp).total_seconds()
                        
                        if time_diff < 5:
                            return 'Rápido'
                        else:
                            return 'Lento'
        
        return 'Desconocido'

    def get_throwers_ranking(self, throw_ins_list, top_n=3):
        """
        Obtiene el ranking de lanzadores con sus estadísticas
        """
        throwers_stats = defaultdict(lambda: {
            'total': 0,
            'successful': 0,
            'fast': 0,
            'player_info': None
        })
        
        for throw_in in throw_ins_list:
            thrower_id = throw_in.get('thrower_id')
            if not thrower_id or pd.isna(thrower_id):
                continue
            
            match_id = throw_in.get('match_id')
            outcome = throw_in.get('outcome', 0)
            
            # Calcular velocidad
            original_idx = throw_in.get('original_index')
            if original_idx in self.df_complete.index:
                timestamp = self.df_complete.loc[original_idx, 'timeStamp']
                speed = self.calculate_throw_speed(original_idx, timestamp)
            else:
                speed = 'Desconocido'
            
            # Actualizar estadísticas
            throwers_stats[thrower_id]['total'] += 1
            if outcome == 1:
                throwers_stats[thrower_id]['successful'] += 1
            if speed == 'Rápido':
                throwers_stats[thrower_id]['fast'] += 1
            
            # Guardar info del jugador (solo la primera vez)
            if throwers_stats[thrower_id]['player_info'] is None:
                player_name, player_pos = self.get_player_info(thrower_id, match_id)
                shirt_number = throw_in.get('thrower_shirt', '?')
                throwers_stats[thrower_id]['player_info'] = {
                    'name': player_name,
                    'position': player_pos,
                    'shirt': shirt_number,
                    'player_id': thrower_id
                }
        
        # Convertir a lista y calcular porcentajes
        ranking = []
        for thrower_id, stats in throwers_stats.items():
            if stats['player_info'] is None:
                continue
            
            total = stats['total']
            success_rate = (stats['successful'] / total * 100) if total > 0 else 0
            fast_rate = (stats['fast'] / total * 100) if total > 0 else 0
            
            ranking.append({
                'player_id': thrower_id,
                'player_name': stats['player_info']['name'],
                'shirt_number': stats['player_info']['shirt'],
                'position': stats['player_info']['position'],
                'total_throws': total,
                'success_rate': success_rate,
                'fast_rate': fast_rate
            })
        
        # Ordenar por total de saques
        ranking.sort(key=lambda x: x['total_throws'], reverse=True)
        
        return ranking[:top_n]
    
    def draw_throwers_ranking(self, ax, throw_ins_list, title="RANKING LANZADORES"):
        """
        Dibuja el ranking de los top 3 lanzadores con un formato de tarjeta
        inspirado en tactic3.1, incluyendo foto, dorsal, nombre y estadísticas.
        """
        ax.axis('off')
        # Usamos un fondo claro para las tarjetas, similar a tactic3.1
        ax.set_facecolor('#f8f9fa')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Título del panel
        ax.text(0.5, 0.98, title, ha='center', va='top',
                fontsize=10, fontweight='bold', color='#2d3436')

        # 1. Obtener los datos del ranking (lógica de tactic4.1)
        ranking = self.get_throwers_ranking(throw_ins_list, top_n=3)

        if not ranking:
            ax.text(0.5, 0.5, "Sin datos de lanzadores",
                    ha='center', va='center', fontsize=8, color='grey')
            return

        # 2. Cargar datos de las fotos
        self.load_player_photos()

        # 3. Definir posiciones y colores para el ranking
        y_positions = [0.78, 0.48, 0.18]
        rank_colors = ['#FFD700', '#C0C0C0', '#CD7F32']  # Oro, Plata, Bronce

        # 4. Iterar y dibujar una tarjeta para cada jugador del ranking
        for rank_idx, player_data in enumerate(ranking):
            y_center = y_positions[rank_idx]
            card_color = rank_colors[rank_idx]
            card_height = 0.25  # Altura de la tarjeta

            # Dibujar la tarjeta de fondo (estilo de tactic3.1)
            rect_bg = patches.FancyBboxPatch(
                (0.02, y_center - card_height/2), 0.96, card_height,
                boxstyle="round,pad=0.01", facecolor='white',
                edgecolor=card_color, linewidth=2.5, alpha=0.95, zorder=10
            )
            ax.add_patch(rect_bg)

            # --- Columna Izquierda de la Tarjeta: Ranking, Foto y Nombre ---

            # Número del ranking (#1, #2, #3)
            ax.text(0.08, y_center, f"#{rank_idx + 1}", fontsize=18, fontweight='bold',
                    va='center', ha='center', color=card_color, zorder=15)

            # Foto del jugador
            player_name = player_data['player_name']
            player_photo = self.get_player_photo(player_name)
            photo_x = 0.23
            photo_size = 0.16

            if player_photo is not None:
                photo_ax = ax.inset_axes([
                    photo_x - photo_size/2, y_center - photo_size/2,
                    photo_size, photo_size
                ], zorder=12)
                photo_ax.imshow(player_photo)
                photo_ax.axis('off')
            else:
                # Si no hay foto, mostrar iniciales
                initials = ''.join([p[0].upper() for p in player_name.split()[:2] if p])
                ax.text(photo_x, y_center, initials, ha='center', va='center',
                        fontsize=16, fontweight='bold', color=card_color,
                        bbox=dict(boxstyle='circle', facecolor='#f0f0f0', edgecolor=card_color, linewidth=2),
                        zorder=12)

            # Dorsal del jugador
            dorsal_text = '?'
            try:
                dorsal = player_data.get('shirt_number', '?')
                if dorsal and str(dorsal).replace('.', '').isdigit():
                    dorsal_text = str(int(float(dorsal)))
            except:
                pass
            
            ax.text(photo_x + photo_size/1.5, y_center + photo_size/2 - 0.015,
                    f"#{dorsal_text}", ha='right', va='top', fontsize=9, fontweight='bold',
                    color='white', zorder=20,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=card_color,
                            edgecolor='white', linewidth=1.5))

            # Nombre y posición
            apellido = player_name.upper() if player_name != 'N/A' else 'DESCONOCIDO'
            posicion = player_data.get('position', '')
            ax.text(0.40, y_center + 0.04, apellido, ha='left', va='center',
                    fontsize=10, fontweight='bold', color='#2d3436', zorder=15)
            ax.text(0.40, y_center - 0.02, f"Pos: {posicion}", ha='left', va='center',
                    fontsize=8, color='#636e72', zorder=15)

            # --- Columna Derecha de la Tarjeta: Estadísticas Clave ---

            # Total de Saques
            total_throws = player_data['total_throws']
            ax.text(0.63, y_center + 0.06, "Total Saques", ha='center', va='center',
                    fontsize=8, color='#636e72', zorder=15)
            ax.text(0.63, y_center, f"{total_throws}", ha='center', va='center',
                    fontsize=18, fontweight='bold', color='#2d3436', zorder=15)

            # Línea separadora vertical
            ax.plot([0.78, 0.78], [y_center - 0.1, y_center + 0.1],
                    color='#dfe6e9', linewidth=1.5, zorder=11)

            # Porcentaje de Éxito
            success_rate = player_data['success_rate']
            ax.text(0.88, y_center + 0.07, f"{success_rate:.0f}%", ha='center', va='center',
                    fontsize=12, fontweight='bold', color='#27ae60', zorder=15)
            ax.text(0.88, y_center + 0.03, "Éxito", ha='center', va='center',
                    fontsize=7, color='#636e72', zorder=15)

            # Porcentaje de Saques Rápidos
            fast_rate = player_data['fast_rate']
            ax.text(0.88, y_center - 0.03, f"{fast_rate:.0f}%", ha='center', va='center',
                    fontsize=12, fontweight='bold', color='#3498db', zorder=15)
            ax.text(0.88, y_center - 0.07, "Rápido", ha='center', va='center',
                    fontsize=7, color='#636e72', zorder=15)
    
    def debug_throw_in_sequences(self, team_name, output_file='debug_throw_ins.txt'):
        """
        Genera un archivo de debug con todas las secuencias de saques de banda
        mostrando las columnas clave para entender qué está pasando
        """
        print("\n" + "="*80)
        print("🔍 INICIANDO DEBUG DE SECUENCIAS DE SAQUES DE BANDA")
        print("="*80)
        
        # Obtener todos los saques de banda
        sequences = self.extract_throw_in_sequences(team_name)
        
        if not sequences:
            print("❌ No se encontraron saques de banda")
            return
        
        # Clasificar por zona y lado
        classification = self.classify_throw_ins_by_zone_and_side(sequences)
        
        def safe_float_format(value):
            """Convierte un valor a float de forma segura para formatear"""
            try:
                if pd.isna(value):
                    return "N/A"
                if isinstance(value, str) and value.lower() in ['no', 'yes', 'sí', 'si']:
                    return "N/A"
                return f"{float(value):.1f}"
            except (ValueError, TypeError):
                return "N/A"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write(f"DEBUG: SAQUES DE BANDA - {team_name}\n")
            f.write("="*100 + "\n\n")
            
            # Para cada lado
            for side in ['Izquierda', 'Derecha']:
                f.write(f"\n{'#'*100}\n")
                f.write(f"{'#'*100}\n")
                f.write(f"LADO: {side}\n")
                f.write(f"{'#'*100}\n")
                f.write(f"{'#'*100}\n\n")
                
                # Para cada zona
                for zone_name in ['Zona 3', 'Zona 2', 'Zona 1']:
                    zone_desc = {
                        'Zona 3': 'OFENSIVA (66.6-100)',
                        'Zona 2': 'MEDIA (33.3-66.6)',
                        'Zona 1': 'DEFENSIVA (0-33.3)'
                    }[zone_name]
                    
                    throw_ins_zone = classification[side][zone_name]
                    
                    f.write(f"\n{'-'*100}\n")
                    f.write(f"{zone_name} - {zone_desc} - {side}\n")
                    f.write(f"Total saques: {len(throw_ins_zone)}\n")
                    f.write(f"{'-'*100}\n\n")
                    
                    if not throw_ins_zone:
                        f.write("  ⚠️ No hay saques en esta zona\n\n")
                        continue
                    
                    # Extraer secuencias después de cada saque
                    sequences_with_players = self.extract_sequences_after_throw_in(throw_ins_zone)
                    
                    for seq_idx, sequence in enumerate(sequences_with_players, 1):
                        f.write(f"\n{'='*100}\n")
                        f.write(f"SECUENCIA #{seq_idx}\n")
                        f.write(f"{'='*100}\n")
                        
                        if not sequence:
                            f.write("  ⚠️ Secuencia vacía\n")
                            continue
                        
                        # Información del saque de banda
                        throw_in = throw_ins_zone[seq_idx - 1]
                        match_id = throw_in['match_id']
                        original_idx = throw_in['original_index']
                        
                        f.write(f"Match ID: {match_id}\n")
                        f.write(f"Original Index: {original_idx}\n")
                        f.write(f"Coordenadas saque: x={throw_in['x']:.1f}, y={throw_in['y']:.1f}\n\n")
                        
                        # Obtener los eventos del DataFrame original
                        f.write(f"{'Idx':<8} | {'x':<6} | {'y':<6} | {'End X':<6} | {'End Y':<6} | {'Event':<20} | {'Out':<4} | {'Throw':<6} | {'Team':<20} | {'Player':<25}\n")
                        f.write("-"*140 + "\n")
                        
                        # Mostrar desde 2 eventos antes hasta 10 eventos después
                        start_idx = max(0, original_idx - 2)
                        end_idx = min(len(self.df_complete), original_idx + 11)
                        
                        events_to_show = self.df_complete.iloc[start_idx:end_idx]
                        
                        for idx, event in events_to_show.iterrows():
                            # Resaltar el saque de banda
                            marker = ">>> " if idx == original_idx else "    "
                            
                            # Extraer valores de forma segura
                            x_str = safe_float_format(event.get('x', ''))
                            y_str = safe_float_format(event.get('y', ''))
                            end_x_str = safe_float_format(event.get('Pass End X', ''))
                            end_y_str = safe_float_format(event.get('Pass End Y', ''))
                            event_name = str(event.get('Event Name', ''))[:20]
                            outcome_val = event.get('outcome', '')
                            outcome_str = str(outcome_val) if pd.notna(outcome_val) else "N/A"
                            thrown_in = str(event.get('Throw in', 'No'))[:6]
                            team_name_event = str(event.get('Team Name', ''))[:20]
                            player_name = str(event.get('playerName', ''))[:25]
                            
                            f.write(f"{marker}{idx:<8} | {x_str:<6} | {y_str:<6} | {end_x_str:<6} | {end_y_str:<6} | {event_name:<20} | {outcome_str:<4} | {thrown_in:<6} | {team_name_event:<20} | {player_name:<25}\n")
                        
                        f.write("\n")
                        
                        # Mostrar la secuencia extraída
                        f.write("SECUENCIA EXTRAÍDA:\n")
                        f.write("-"*100 + "\n")
                        
                        for pass_idx, pass_data in enumerate(sequence):
                            passer_id = pass_data.get('passer_id')
                            receiver_id = pass_data.get('receiver_id')
                            
                            passer_name, _ = self.get_player_info(passer_id, match_id) if passer_id else ("N/A", "")
                            receiver_name, _ = self.get_player_info(receiver_id, match_id) if receiver_id else ("N/A", "")
                            
                            f.write(f"  Pase {pass_idx + 1}:\n")
                            f.write(f"    Pasador: {passer_name} (dorsal {pass_data.get('passer_shirt', '?')})\n")
                            f.write(f"    Receptor: {receiver_name} (dorsal {pass_data.get('receiver_shirt', '?')})\n")
                            f.write(f"    Coords: ({pass_data['x']:.1f}, {pass_data['y']:.1f}) → ({pass_data['end_x']:.1f}, {pass_data['end_y']:.1f})\n")
                            f.write(f"    Outcome: {pass_data.get('outcome')}\n\n")
                        
                        f.write("\n")
        
        print(f"\n✅ Debug guardado en: {output_file}")
        print("="*80 + "\n")
    
    def get_player_shirt_number(self, player_id, match_id):
        """Obtiene el dorsal de un jugador en un partido específico"""
        if self.df_players is None or pd.isna(player_id):
            return '?'
        
        player_data = self.df_players[
            (self.df_players['Player ID'] == player_id) &
            (self.df_players['Match ID'] == match_id)
        ]
        
        if not player_data.empty:
            shirt_number = player_data['Shirt Number'].iloc[0]
            if pd.notna(shirt_number):
                try:
                    return str(int(shirt_number))
                except:
                    return '?'
        
        # Buscar en cualquier partido si no está en este
        player_data_any = self.df_players[self.df_players['Player ID'] == player_id]
        if not player_data_any.empty:
            shirt_number = player_data_any['Shirt Number'].iloc[0]
            if pd.notna(shirt_number):
                try:
                    return str(int(shirt_number))
                except:
                    return '?'
        
        return '?'

    def find_throw_in_patterns(self, sequences, top_n=3, eps=20, min_samples=2):
        """
        Encuentra los patrones más repetidos en las secuencias de saques de banda.
        Utiliza clustering DBSCAN sobre las características del saque y el siguiente pase.
        'eps' es la distancia máxima entre dos muestras para que se consideren vecinas.
        """
        from sklearn.cluster import DBSCAN
        
        valid_sequences = [s for s in sequences if len(s) > 0]
        if len(valid_sequences) < min_samples:
            return []

        feature_vectors = []
        for seq in valid_sequences:
            throw_in = seq[0]
            
            # --- INICIO DE LA CORRECCIÓN ---
            # Nos aseguramos de que cada valor sea numérico, manejando explícitamente los 'None'
            # que pueden venir de la extracción de datos.
            
            # Características del primer pase (saque de banda)
            angle = float(throw_in.get('angle')) if pd.notna(throw_in.get('angle')) else 90.0
            dist = float(throw_in.get('distance')) if pd.notna(throw_in.get('distance')) else 15.0
            end_x1 = float(throw_in.get('end_x')) if pd.notna(throw_in.get('end_x')) else 50.0
            end_y1 = float(throw_in.get('end_y')) if pd.notna(throw_in.get('end_y')) else 50.0

            # Características del segundo pase (si existe)
            if len(seq) > 1:
                pass2 = seq[1]
                # Si el segundo pase no tiene coordenadas finales, usamos las del primero
                end_x2 = float(pass2.get('end_x')) if pd.notna(pass2.get('end_x')) else end_x1
                end_y2 = float(pass2.get('end_y')) if pd.notna(pass2.get('end_y')) else end_y1
            else:
                end_x2, end_y2 = end_x1, end_y1
            # --- FIN DE LA CORRECCIÓN ---

            # Normalizamos el ángulo para que no domine la distancia
            feature_vectors.append([angle / 10, dist, end_x1, end_y1, end_x2, end_y2])

        X = np.array(feature_vectors)
        
        # Aplicar clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
        
        unique_labels = set(labels)
        patterns = []
        
        for k in unique_labels:
            if k == -1:
                continue

            class_member_mask = (labels == k)
            cluster_sequences = [valid_sequences[i] for i, mask_val in enumerate(class_member_mask) if mask_val]
            
            if len(cluster_sequences) >= min_samples:
                patterns.append({
                    'count': len(cluster_sequences),
                    'all_sequences': cluster_sequences
                })
        
        patterns.sort(key=lambda x: x['count'], reverse=True)
        return patterns[:top_n]
        
    def calculate_average_path(self, sequences):
        """
        Calcula la trayectoria promedio de un conjunto de secuencias.
        """
        if not sequences:
            return []
        
        max_len = max(len(s) for s in sequences)
        avg_path = []

        for i in range(max_len):
            points_at_step = [s[i] for s in sequences if len(s) > i]
            
            # Promedio de coordenadas de inicio y fin en este paso
            avg_x = np.mean([p['x'] for p in points_at_step if pd.notna(p.get('x'))])
            avg_y = np.mean([p['y'] for p in points_at_step if pd.notna(p.get('y'))])
            avg_end_x = np.mean([p['end_x'] for p in points_at_step if pd.notna(p.get('end_x'))])
            avg_end_y = np.mean([p['end_y'] for p in points_at_step if pd.notna(p.get('end_y'))])
            
            if i == 0:
                avg_path.append((avg_x, avg_y))
            avg_path.append((avg_end_x, avg_end_y))
            
        return avg_path
    
    def get_zone_from_x_coord(self, x):
        """
        Determina la zona según la coordenada X
        Zona 3 (Ofensiva): 66.6 a 100 - Campo rival
        Zona 2 (Media): 33.3 a 66.6 - Zona media
        Zona 1 (Defensiva): 0 a 33.3 - Campo propio
        """
        if pd.isna(x):
            return None
        if 66.6 <= x <= 100:
            return 3  # Zona defensiva
        elif 33.3 <= x < 66.6:
            return 2  # Zona media
        elif 0 <= x < 33.3:
            return 1  # Zona ofensiva
        return None 
    
    def extract_throw_in_sequences(self, team_name):
        """
        Extrae secuencias de saques de banda, encontrando al receptor en el siguiente evento.
        """
        if self.df is None:
            return []
        
        # El DataFrame ya está ordenado gracias a la nueva función load_data
        throw_ins = self.df[
            (self.df['Team Name'] == team_name) & 
            (self.df['Event Name'] == 'Pass') &
            (self.df['Throw in'] == 'Sí')
        ].copy()
        
        print(f"🎯 Saques de banda encontrados: {len(throw_ins)}")
        if len(throw_ins) == 0:
            return []

        sequences = [] 
        
        # 'idx' aquí es el índice del DataFrame principal (self.df), que está ordenado
        for idx, throw_in in throw_ins.iterrows():
            # ... (código para calcular ángulo, coordenadas, etc. se mantiene igual) ...
            final_angle = None
            angle_rad = AnalizadorSaquesBanda.safe_to_float(throw_in.get('Angle'))
            if not np.isnan(angle_rad):
                angle_deg = np.degrees(angle_rad)
                side = 'Izquierda' if throw_in.get('y', 50) >= 50 else 'Derecha'
                if side == 'Izquierda':
                    final_angle = angle_deg if angle_deg <= 180 else angle_deg - 180
                elif side == 'Derecha':
                    final_angle = 180 - angle_deg
            x_coord = AnalizadorSaquesBanda.safe_to_float(throw_in.get('x'))
            y_coord = AnalizadorSaquesBanda.safe_to_float(throw_in.get('y'))
            end_x_coord = AnalizadorSaquesBanda.safe_to_float(throw_in.get('Pass End X'))

            # --- INICIO DE LA NUEVA LÓGICA PARA ENCONTRAR AL RECEPTOR ---
            receiver_id = None
            # Miramos el siguiente evento en el DataFrame principal
            if idx + 1 < len(self.df):
                next_event = self.df.iloc[idx + 1]
                # El jugador que realiza la siguiente acción para el mismo equipo es el receptor
                if next_event['Team Name'] == team_name:
                    # ¡Usamos 'playerId', el nombre de columna real de tu archivo!
                    receiver_id = next_event.get('playerId')

            match_id = throw_in.get('Match ID')
            thrower_id = throw_in.get('playerId')
            thrower_shirt = self.get_player_shirt_number(thrower_id, match_id)

            sequence = {
                'match_id': match_id,
                'period': throw_in.get('periodId'),
                'minute': throw_in.get('Minute'),
                'second': throw_in.get('Second'),
                'x': x_coord,
                'y': y_coord,
                'end_x': end_x_coord,
                'end_y': AnalizadorSaquesBanda.safe_to_float(throw_in.get('Pass End Y')),
                'outcome': throw_in.get('outcome'),
                'player': throw_in.get('playerName'),
                'zone_origin': self.get_zone_from_x_coord(x_coord),
                'zone_dest': self.get_zone_from_x_coord(end_x_coord),
                'side': 'Izquierda' if y_coord >= 50 else 'Derecha',
                'timestamp': throw_in.get('timeStamp'),
                'angle': final_angle,
                'speed': self.calculate_throw_speed(idx, throw_in.get('timeStamp')),
                'distance': AnalizadorSaquesBanda.safe_to_float(throw_in.get('Length')),
                'thrower_id': thrower_id,
                'thrower_shirt': thrower_shirt,  # 🔥 AÑADIDO
                'receiver_id': receiver_id,
                'original_index': idx,  # 🔥 AÑADIDO - índice en el DataFrame
                'team_name': team_name  # 🔥 AÑADIDO
            }
            
            sequences.append(sequence)
        
        return sequences
        
    def calculate_throw_angle(self, x, y, end_x, end_y):
        """
        Calcula el ángulo del lanzamiento en grados
        Retorna None si las coordenadas no son válidas
        """
        if pd.isna(x) or pd.isna(y) or pd.isna(end_x) or pd.isna(end_y):
            return None
        
        # Calcular diferencias
        dx = end_x - x
        dy = end_y - y
        
        # Calcular ángulo en radianes y convertir a grados
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        # Normalizar a rango 0-360
        if angle_deg < 0:
            angle_deg += 360
        
        return round(angle_deg, 1)
    
    def calculate_throw_speed(self, current_idx, current_timestamp):
        """
        Calcula si el saque es rápido (<5s) o lento (>=5s)
        Compara el timestamp con el evento anterior del mismo equipo
        """
        if pd.isna(current_timestamp):
            return 'Desconocido'
        
        # Convertir timestamp a datetime si es string
        if isinstance(current_timestamp, str):
            try:
                current_timestamp = pd.to_datetime(current_timestamp)
            except:
                return 'Desconocido'
        
        # Buscar el evento anterior en el mismo partido
        if current_idx > 0 and current_idx in self.df.index:
            current_match = self.df.loc[current_idx, 'Match ID']
            
            # Buscar hacia atrás hasta encontrar un evento del mismo partido
            for prev_idx in range(current_idx - 1, max(0, current_idx - 20), -1):
                if prev_idx in self.df.index:
                    prev_match = self.df.loc[prev_idx, 'Match ID']
                    prev_timestamp = self.df.loc[prev_idx, 'timeStamp']
                    
                    if prev_match == current_match and pd.notna(prev_timestamp):
                        # Convertir timestamp anterior
                        if isinstance(prev_timestamp, str):
                            try:
                                prev_timestamp = pd.to_datetime(prev_timestamp)
                            except:
                                continue
                        
                        # Calcular diferencia en segundos
                        time_diff = (current_timestamp - prev_timestamp).total_seconds()
                        
                        if time_diff < 5:
                            return 'Rápido'
                        else:
                            return 'Lento'
        
        return 'Desconocido'
    
    def draw_horizontal_bars(self, ax, x_start, y_start, width, height, value1, value2, label1, label2, color1, color2):
        """
        Dibuja dos barras horizontales apiladas
        x_start, y_start: posición inicial (0-1 en coordenadas del eje)
        width, height: tamaño de cada barra
        value1, value2: valores a mostrar (porcentajes)
        """
        # Barra 1
        bar1_width = (value1 / 100) * width
        rect1 = plt.Rectangle((x_start, y_start), bar1_width, height, 
                            facecolor=color1, edgecolor='white', linewidth=1,
                            transform=ax.transAxes, zorder=2)
        ax.add_patch(rect1)
        
        # Barra 2 (resto)
        rect2 = plt.Rectangle((x_start + bar1_width, y_start), width - bar1_width, height,
                            facecolor=color2, edgecolor='white', linewidth=1,
                            transform=ax.transAxes, zorder=2)
        ax.add_patch(rect2)
        
        # Etiquetas dentro de las barras
        if value1 > 15:  # Solo mostrar si hay espacio
            ax.text(x_start + bar1_width/2, y_start + height/2, f'{value1:.0f}%',
                ha='center', va='center', fontsize=6, color='white',
                fontweight='bold', transform=ax.transAxes)
        
        if value2 > 15:
            ax.text(x_start + bar1_width + (width - bar1_width)/2, y_start + height/2, f'{value2:.0f}%',
                ha='center', va='center', fontsize=6, color='white',
                fontweight='bold', transform=ax.transAxes)
    
    def draw_angle_distribution(self, ax, x_center, y_center, angles_list, distances_list, size=0.15, side='left'):
        """
        Dibuja la distribución de ángulos con fondos coloreados según porcentaje.
        - Muestra el % y la distancia media para cada tramo.
        - Los espacios entre líneas discontinuas se colorean según el porcentaje.
        """
        # Ajustar posición según el lado
        if side == 'left':
            x_center = x_center - 0.08  # Mover más a la izquierda
        elif side == 'right':
            x_center = x_center + 0.08  # Mover más a la derecha
        
        # Combinamos y filtramos para mantener la correspondencia entre ángulo y distancia
        valid_pairs = [(a, d) for a, d in zip(angles_list, distances_list) if a is not None and d is not None]

        if not valid_pairs:
            ax.text(x_center, y_center, 'Sin datos', ha='center', va='center',
                fontsize=7, color='#95a5a6', transform=ax.transAxes)
            return

        total_events = len(valid_pairs)
        
        tramos = [
            (144, 180, 'Muy adelante'), (108, 144, 'Adelante'),
            (72, 108, 'Lateral'), (36, 72, 'Atrás'), (0, 36, 'Muy atrás')
        ]
        line_length = size * 1.1
        
        # Dibujar línea vertical central
        ax.plot([x_center, x_center], [y_center - line_length / 1.5, y_center + line_length / 1.5], 
                color='#2d3436', linewidth=2, transform=ax.transAxes, zorder=3)
        
        # Dibujar líneas discontinuas de límites
        boundary_angles_deg = [144, 108, 72, 36]
        for angle_deg in boundary_angles_deg:
            if side == 'right': 
                angle_rad = np.radians(270 - angle_deg)
            else: 
                angle_rad = np.radians(angle_deg - 90)
            x_end = x_center + line_length * np.cos(angle_rad)
            y_end = y_center + line_length * np.sin(angle_rad)
            ax.plot([x_center, x_end], [y_center, y_end],
                    color='#636e72', linestyle=':', linewidth=1.2, transform=ax.transAxes, zorder=4)

        # Dibujar tramos con fondo coloreado
        for angle_min, angle_max, label in tramos:
            # Filtramos los pares que caen en este tramo
            pairs_in_tramo = [(a, d) for a, d in valid_pairs if (angle_max == 180 and angle_min <= a <= angle_max) or (angle_min <= a < angle_max)]
            
            count = len(pairs_in_tramo)
            percentage = (count / total_events * 100) if total_events > 0 else 0

            if count > 0:
                # Calcular color de fondo según porcentaje (verde=alto, rojo=bajo)
                ratio = percentage / 100.0
                r = int(255 * (1 - ratio))
                g = int(255 * ratio)
                bg_color = f'#{r:02x}{g:02x}00'
                
                # Calcular color de texto según porcentaje para mejor legibilidad
                if percentage >= 50: text_color = '#27ae60'
                elif percentage >= 30: text_color = '#2ecc71'
                elif percentage >= 15: text_color = '#f39c12'
                else: text_color = '#e74c3c'
                
                # Calculamos la distancia media del tramo
                distances_in_tramo = [d for a, d in pairs_in_tramo]
                avg_dist = np.mean(distances_in_tramo)
                
                # Dibujar el sector coloreado (fondo)
                angle_start_deg = angle_min
                angle_end_deg = angle_max if angle_max != 180 else 180
                
                # Crear wedge (sector circular) para el fondo
                from matplotlib.patches import Wedge
                if side == 'right':
                    theta1 = 270 - angle_end_deg
                    theta2 = 270 - angle_start_deg
                else:
                    theta1 = angle_start_deg - 90
                    theta2 = angle_end_deg - 90
                
                wedge = Wedge((x_center, y_center), line_length * 0.9, theta1, theta2,
                            facecolor=bg_color, alpha=0.3, transform=ax.transAxes, zorder=1)
                ax.add_patch(wedge)
                
                # Creamos el texto combinado
                display_text = f'{percentage:.0f}%\n{avg_dist:.1f}m'

                angle_text_deg = (angle_min + angle_max) / 2
                text_radius = line_length * 0.6
                if side == 'right': 
                    angle_text_rad = np.radians(270 - angle_text_deg)
                else: 
                    angle_text_rad = np.radians(angle_text_deg - 90)
                x_text = x_center + text_radius * np.cos(angle_text_rad)
                y_text = y_center + text_radius * np.sin(angle_text_rad)
                
                ax.text(x_text, y_text, display_text,
                        ha='center', va='center', fontsize=6, color=text_color,
                        fontweight='bold', transform=ax.transAxes, zorder=10,
                        path_effects=[patheffects.withStroke(linewidth=2.5, foreground='white')],
                        linespacing=0.9)
        
        ax.text(x_center, y_center + line_length/1.5 + 0.015, '180°', ha='center', va='bottom', fontsize=5, color='#2d3436', transform=ax.transAxes)
        ax.text(x_center, y_center - line_length/1.5 - 0.015, '0°', ha='center', va='top', fontsize=5, color='#2d3436', transform=ax.transAxes)
    
    def draw_zone_stats_per_row(self, ax, throw_ins_left, throw_ins_right, zone_number):
        """
        Dibuja las estadísticas de la zona con los nuevos ajustes de diseño.
        - Barras horizontales más gruesas.
        - Gráficos de ángulos más abajo y centrados en sus columnas.
        """
        from collections import Counter

        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        def draw_horizontal_bars_inline(ax, x_start, y_start, width, height, 
                                value1, value2, label1, label2, color1, color2):
            ratio = value1 / 100.0
            r = int(255 * (1 - ratio))
            g = int(255 * ratio)
            color_fondo = f'#{r:02x}{g:02x}00'
            
            rect_fondo = Rectangle((x_start, y_start), width, height, 
                                facecolor=color_fondo, alpha=0.7, 
                                transform=ax.transAxes, zorder=1)
            ax.add_patch(rect_fondo)
            
            width1 = width * (value1 / 100)
            width2 = width * (value2 / 100)
            
            rect1 = Rectangle((x_start, y_start), width1, height,
                            facecolor=color1, edgecolor='white', linewidth=1,
                            transform=ax.transAxes, zorder=2)
            ax.add_patch(rect1)
            
            rect2 = Rectangle((x_start + width1, y_start), width2, height,
                            facecolor=color2, edgecolor='white', linewidth=1,
                            transform=ax.transAxes, zorder=2)
            ax.add_patch(rect2)
            
            luminancia = 0.299 * r + 0.587 * g
            text_color = 'white' if luminancia < 128 else 'black'
            
            fontsize = 6
            if width1 > 0.05:
                ax.text(x_start + width1/2, y_start + height/2, 
                    f'{value1:.0f}%', ha='center', va='center',
                    fontsize=fontsize, color=text_color, fontweight='bold',
                    transform=ax.transAxes, zorder=3)
            
            if width2 > 0.05:
                ax.text(x_start + width1 + width2/2, y_start + height/2,
                    f'{value2:.0f}%', ha='center', va='center', 
                    fontsize=fontsize, color=text_color, fontweight='bold',
                    transform=ax.transAxes, zorder=3)
        
        zone_colors = {1: '#0984e3', 2: '#6c5ce7', 3: '#fd79a8'}
        zone_names = {1: 'Defensiva', 2: 'Media', 3: 'Ofensiva'}
        zone_color = zone_colors.get(zone_number, '#2d3436')
        zone_name = zone_names.get(zone_number, '')
        
        ax.text(0.5, 0.95, f'ZONA {zone_number} ({zone_name})', 
            ha='center', va='top', fontsize=10, fontweight='bold', 
            color=zone_color, transform=ax.transAxes)
        
        ax.plot([0.33, 0.33], [0, 0.85], color='#dfe6e9', lw=1.5, transform=ax.transAxes, zorder=1)
        ax.plot([0.67, 0.67], [0, 0.85], color='#dfe6e9', lw=1.5, transform=ax.transAxes, zorder=1)
        
        # --- IZQUIERDA ---
        total_left = len(throw_ins_left)
        angles_left = [t.get('angle') for t in throw_ins_left]
        distances_left = [t.get('distance') for t in throw_ins_left]
        perc_exito_left = (sum(1 for t in throw_ins_left if t.get('outcome') == 1) / total_left * 100) if total_left > 0 else 0
        perc_rapido_left = (sum(1 for t in throw_ins_left if t.get('speed') == 'Rápido') / total_left * 100) if total_left > 0 else 0
        
        y_pos = 0.85
        ax.text(0.165, y_pos, 'IZQUIERDA', ha='center', va='center', fontsize=8, fontweight='bold', color='#2d3436', transform=ax.transAxes)
        y_pos -= 0.10
        ax.text(0.165, y_pos, f'Total: {total_left}', ha='center', va='center', fontsize=7, color='#2d3436', fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.10
        ax.text(0.165, y_pos + 0.025, 'Éxito', ha='center', va='bottom', fontsize=6, color='#2d3436', transform=ax.transAxes)
        # CAMBIO: Barras más gruesas (height=0.045)
        draw_horizontal_bars_inline(ax, 0.05, y_pos - 0.015, 0.23, 0.045, perc_exito_left, 100 - perc_exito_left, 'Éxito', 'Fallo', '#00b894', '#d63031')
        
        y_pos -= 0.08
        ax.text(0.165, y_pos + 0.025, 'Velocidad', ha='center', va='bottom', fontsize=6, color='#2d3436', transform=ax.transAxes)
        # CAMBIO: Barras más gruesas (height=0.045)
        draw_horizontal_bars_inline(ax, 0.05, y_pos - 0.015, 0.23, 0.045, perc_rapido_left, 100 - perc_rapido_left, 'Rápido', 'Lento', '#0984e3', '#74b9ff')

        y_pos -= 0.08
        ax.text(0.165, y_pos + 0.03, 'Distribución Ángulos', ha='center', va='bottom', fontsize=6, color='#2d3436', transform=ax.transAxes)
        # CAMBIO: Más abajo (y_pos - 0.15) y más a la izquierda (x_center=0.15)
        self.draw_angle_distribution(ax, 0.15, y_pos - 0.15, angles_left, distances_left, size=0.18, side='left')
        
        # --- DERECHA ---
        total_right = len(throw_ins_right)
        angles_right = [t.get('angle') for t in throw_ins_right]
        distances_right = [t.get('distance') for t in throw_ins_right]
        perc_exito_right = (sum(1 for t in throw_ins_right if t.get('outcome') == 1) / total_right * 100) if total_right > 0 else 0
        perc_rapido_right = (sum(1 for t in throw_ins_right if t.get('speed') == 'Rápido') / total_right * 100) if total_right > 0 else 0
        
        y_pos = 0.85
        ax.text(0.835, y_pos, 'DERECHA', ha='center', va='center', fontsize=8, fontweight='bold', color='#2d3436', transform=ax.transAxes)
        y_pos -= 0.10
        ax.text(0.835, y_pos, f'Total: {total_right}', ha='center', va='center', fontsize=7, color='#2d3436', fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.10
        ax.text(0.835, y_pos + 0.025, 'Éxito', ha='center', va='bottom', fontsize=6, color='#2d3436', transform=ax.transAxes)
        # CAMBIO: Barras más gruesas (height=0.045)
        draw_horizontal_bars_inline(ax, 0.72, y_pos - 0.015, 0.23, 0.045, perc_exito_right, 100 - perc_exito_right, 'Éxito', 'Fallo', '#00b894', '#d63031')

        y_pos -= 0.08
        ax.text(0.835, y_pos + 0.025, 'Velocidad', ha='center', va='bottom', fontsize=6, color='#2d3436', transform=ax.transAxes)
        # CAMBIO: Barras más gruesas (height=0.045)
        draw_horizontal_bars_inline(ax, 0.72, y_pos - 0.015, 0.23, 0.045, perc_rapido_right, 100 - perc_rapido_right, 'Rápido', 'Lento', '#0984e3', '#74b9ff')
        
        y_pos -= 0.08
        ax.text(0.835, y_pos + 0.03, 'Distribución Ángulos', ha='center', va='bottom', fontsize=6, color='#2d3436', transform=ax.transAxes)
        # CAMBIO: Más abajo (y_pos - 0.15) y más a la derecha (x_center=0.85)
        self.draw_angle_distribution(ax, 0.85, y_pos - 0.15, angles_right, distances_right, size=0.18, side='right')
        
        # --- CENTRO (RESUMEN) ---
        # (Sin cambios en esta sección)
        total_combined = total_left + total_right
        exitosos_combined = sum(1 for t in throw_ins_left if t.get('outcome') == 1) + sum(1 for t in throw_ins_right if t.get('outcome') == 1)
        perc_exito_combined = (exitosos_combined / total_combined * 100) if total_combined > 0 else 0
        rapidos_combined = sum(1 for t in throw_ins_left if t.get('speed') == 'Rápido') + sum(1 for t in throw_ins_right if t.get('speed') == 'Rápido')
        perc_rapido_combined = (rapidos_combined / total_combined * 100) if total_combined > 0 else 0
        y_pos = 0.82
        ax.text(0.5, y_pos, 'RESUMEN ZONA', ha='center', va='center', fontsize=8, fontweight='bold', color='#1e3d59', transform=ax.transAxes)
        y_pos -= 0.12
        ax.text(0.5, y_pos, f'Total: {total_combined}', ha='center', va='center', fontsize=8, color='#1e3d59', fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.12
        ax.text(0.5, y_pos + 0.03, 'Éxito Total: ' + f'{perc_exito_combined:.0f}%', ha='center', va='bottom', fontsize=7, color='#1e3d59', fontweight='bold', transform=ax.transAxes)
        draw_horizontal_bars_inline(ax, 0.38, y_pos - 0.02, 0.24, 0.05, perc_exito_combined, 100 - perc_exito_combined, 'Éxito', 'Fallo', '#00b894', '#d63031')
        y_pos -= 0.12 # Hacemos más espacio
        ax.text(0.5, y_pos + 0.03, 'Velocidad Total: ' + f'{perc_rapido_combined:.0f}% Rápido', ha='center', va='bottom', fontsize=7, color='#1e3d59', fontweight='bold', transform=ax.transAxes)
        draw_horizontal_bars_inline(ax, 0.38, y_pos - 0.02, 0.24, 0.05, perc_rapido_combined, 100 - perc_rapido_combined, 'Rápido', 'Lento', '#0984e3', '#74b9ff')
        
        all_valid_angles = [a for a in angles_left + angles_right if a is not None]
        all_valid_distances = [d for d in distances_left + distances_right if d is not None]
        
        if all_valid_angles:
            tramos_map = {
                (144, 181): 'Muy Adelante', (108, 144): 'Adelante', (72, 108): 'Lateral',
                (36, 72): 'Atrás', (0, 36): 'Muy Atrás'
            }
            angle_categories = [next((name for r, name in tramos_map.items() if r[0] <= a < r[1]), None) for a in all_valid_angles]
            angle_categories = [cat for cat in angle_categories if cat is not None]
            if angle_categories:
                most_common_angle = Counter(angle_categories).most_common(1)[0]
                angle_text = f"{most_common_angle[0]} ({most_common_angle[1] / len(angle_categories) * 100:.0f}%)"
            else: angle_text = "N/A"
        else: angle_text = "N/A"

        avg_distance = np.mean(all_valid_distances) if all_valid_distances else None
        dist_text = f"{avg_distance:.1f}m" if avg_distance is not None else "N/A"
        
        y_pos -= 0.22
        summary_text = f"Ángulo Frecuente: {angle_text}\nDistancia Media: {dist_text}"
        ax.text(0.5, y_pos, summary_text, ha='center', va='center', fontsize=8, color='#34495e',
                bbox=dict(boxstyle='round,pad=0.5', fc='#ecf0f1', ec='#bdc3c7', lw=1.5),
                linespacing=1.3)
    
    def classify_throw_ins_by_zone_and_side(self, sequences):
        """
        Clasifica los saques de banda por zona y lado del campo
        """
        classification = {
            'Izquierda': {'Zona 1': [], 'Zona 2': [], 'Zona 3': []},
            'Derecha': {'Zona 1': [], 'Zona 2': [], 'Zona 3': []}
        }
        
        for seq in sequences:
            side = seq['side']
            zone = seq['zone_origin']
            
            if zone is not None:
                zone_key = f'Zona {zone}'
                if zone_key in classification[side]:
                    classification[side][zone_key].append(seq)
        
        return classification
    
    def draw_pitch_with_throw_ins(self, ax, throw_ins_list, zone_number, side):
        """
        Dibuja los patrones visuales con leyendas y círculos de demarcación de tamaño fijo y legible.
        """
        pitch = VerticalPitch(
            pitch_type='opta', pitch_color='#2d5a27', line_color='white',
            linewidth=2, label=False, tick=False
        )
        pitch.draw(ax=ax)
        
        # Resaltar la zona actual
        if zone_number == 1:
            zone_rect = Rectangle((0, 0), 100, 33.3, facecolor='yellow', alpha=0.12, zorder=1)
        elif zone_number == 2:
            zone_rect = Rectangle((0, 33.3), 100, 33.3, facecolor='yellow', alpha=0.12, zorder=1)
        else: # Zona 3
            zone_rect = Rectangle((0, 66.6), 100, 33.4, facecolor='yellow', alpha=0.12, zorder=1)
        ax.add_patch(zone_rect)

        total_throws_in_zone = len(throw_ins_list)

        if total_throws_in_zone == 0:
            ax.text(50, 50, 'Sin Datos', ha='center', va='center', fontsize=12, color='white')
            return ax

        sequences = self.extract_sequences_after_throw_in(throw_ins_list)
        sequences = [seq for seq in sequences if seq]
        
        if not sequences:
            ax.text(50, 50, 'Sin Secuencias', ha='center', va='center', fontsize=10, color='white')
            return ax
        
        patterns = self.find_visual_patterns_clustering(sequences, max_patterns=3)

        if not patterns:
            ax.text(50, 50, 'Sin Patrones\nRepetidos (>=2)', ha='center', va='center', 
                    fontsize=10, color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#e67e22', alpha=0.8))
            return ax

        pattern_colors = ['#e74c3c', '#3498db', '#f1c40f']

        for idx, pattern_data in enumerate(patterns):
            color = pattern_colors[idx % len(pattern_colors)]
            all_seqs = pattern_data['all_sequences']
            count = pattern_data['count']
            
            # Dibujar todas las secuencias del clúster con transparencia
            for seq in all_seqs:
                for pass_data in seq:
                    if all(pd.notna(val) for val in [pass_data.get('x'), pass_data.get('y'), pass_data.get('end_x'), pass_data.get('end_y')]):
                        pitch.arrows(
                            pass_data['x'], pass_data['y'], pass_data['end_x'], pass_data['end_y'],
                            color=color, width=0.8, alpha=0.2, zorder=5, ax=ax
                        )

            avg_sequence = pattern_data.get('average_sequence', [])
            sequence_players = self.get_most_common_players_by_position(all_seqs)

            if not avg_sequence or not sequence_players:
                continue
            
            # --- 🔥 INICIO DEL CÓDIGO ACTUALIZADO PARA CÍRCULOS Y TEXTO 🔥 ---

            # Dibujar demarcación del lanzador
            if sequence_players:
                passer_pos_info = sequence_players[0]
                start_x, start_y = avg_sequence[0]['x'], avg_sequence[0]['y']
                if pd.notna(start_x) and pd.notna(start_y):
                    position_text = passer_pos_info.get('position', '')
                    # 1. Dibujar siempre un círculo de tamaño fijo
                    circle = patches.Circle((start_y, start_x), radius=2.5, facecolor='white', edgecolor=color, linewidth=1.5, zorder=29)
                    ax.add_patch(circle)
                    # 2. Solo si la posición es válida, escribir el texto
                    if position_text and position_text not in ['Desconocido', 'No Encontrada', 'Sustituto']:
                        ax.text(start_y, start_x, position_text,
                                ha='center', va='center', fontsize=6, color='black', weight='bold',
                                zorder=30)

            # Dibujar flechas promedio y demarcaciones de receptores
            for i, pass_data in enumerate(avg_sequence):
                if all(pd.notna(val) for val in [pass_data.get('x'), pass_data.get('y'), pass_data.get('end_x'), pass_data.get('end_y')]):
                    pitch.arrows(
                        pass_data['x'], pass_data['y'], pass_data['end_x'], pass_data['end_y'],
                        color=color, width=1.5, headwidth=3, headlength=3, alpha=1.0, zorder=20, ax=ax
                    )
                    if i + 1 < len(sequence_players):
                        receiver_pos_info = sequence_players[i+1]
                        end_x, end_y = pass_data['end_x'], pass_data['end_y']
                        position_text = receiver_pos_info.get('position', '')
                        # 1. Dibujar siempre un círculo de tamaño fijo
                        circle = patches.Circle((end_y, end_x), radius=2.5, facecolor='white', edgecolor=color, linewidth=1.5, zorder=29)
                        ax.add_patch(circle)
                        # 2. Solo si la posición es válida, escribir el texto
                        if position_text and position_text not in ['Desconocido', 'No Encontrada', 'Sustituto']:
                            ax.text(end_y, end_x, position_text,
                                    ha='center', va='center', fontsize=6, color='black', weight='bold',
                                    zorder=30)
            
            # --- 🔥 FIN DEL CÓDIGO ACTUALIZADO 🔥 ---

            # Lógica de leyenda (sin cambios)
            percentage = (count / total_throws_in_zone * 100) if total_throws_in_zone > 0 else 0
            text_display = f'{count}x\n({percentage:.0f}%)'

            legend_x_positions = [95, 85, 75] 
            
            if side == 'Izquierda':
                legend_y_pos = 95
                ha = 'right'
            else:
                legend_y_pos = 5
                ha = 'left'

            ax.text(legend_y_pos, legend_x_positions[idx], text_display,
                   ha=ha, va='top', 
                   fontsize=8,
                   color='white',
                   weight='bold',
                   linespacing=1.2,
                   bbox=dict(boxstyle="round,pad=0.5",
                             facecolor=color,
                             alpha=0.9,
                             edgecolor='white',
                             lw=1.5),
                   zorder=50)
    
    def _get_most_common_positional_sequence(self, pattern_sequences):
        """
        Analiza un cluster de secuencias y devuelve la cadena de demarcaciones más común
        y su porcentaje de aparición dentro de ese cluster.
        """
        if not pattern_sequences:
            return "N/A", 0

        positional_chains = []
        for seq in pattern_sequences:
            chain = []
            match_id = seq[0].get('match_id')

            # 1. Posición del lanzador
            thrower_id = seq[0].get('thrower_id')
            _, thrower_pos = self.get_player_info(thrower_id, match_id)
            chain.append(thrower_pos if thrower_pos else 'DESC')

            # 2. Posición del primer receptor
            receiver1_id = seq[0].get('receiver_id')
            _, receiver1_pos = self.get_player_info(receiver1_id, match_id)
            chain.append(receiver1_pos if receiver1_pos else 'DESC')

            # 3. Posición del segundo receptor (si existe)
            if len(seq) > 2:
                receiver2_id = seq[1].get('receiver_id')
                _, receiver2_pos = self.get_player_info(receiver2_id, match_id)
                chain.append(receiver2_pos if receiver2_pos else 'DESC')
            
            positional_chains.append(" → ".join(chain))

        if not positional_chains:
            return "N/A", 0

        # Contamos cuál es la cadena más común
        most_common = Counter(positional_chains).most_common(1)[0]
        most_common_chain = most_common[0]
        count = most_common[1]
        percentage = (count / len(positional_chains)) * 100
        
        return most_common_chain, percentage

    def draw_pattern_ranking(self, ax, patterns, total_throw_ins, side):
        """
        Dibuja las 3 secuencias de jugadores más repetidas con fotos (1-3 pases)
        Prioriza jugadores por frecuencia en cada posición específica
        """
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        title_side = 'IZQUIERDA' if side == 'Izquierda' else 'DERECHA'
        
        # Título centrado
        ax.text(0.5, 0.98, f'SECUENCIAS - {title_side}', ha='center', va='top', 
                fontsize=9, fontweight='bold', color='#2d3436')
        ax.text(0.5, 0.91, f"Total Saques: {total_throw_ins}", ha='center', va='center', 
                fontsize=7, color='#34495e')

        if not patterns:
            ax.text(0.5, 0.5, "Sin patrones\nrepetidos", ha='center', va='center', 
                    fontsize=8, color='#e67e22')
            return

        # Colores de patrones (igual que en campograma)
        pattern_colors = ['#e74c3c', '#3498db', '#f1c40f']
        
        # Cargar fotos una sola vez
        self.load_player_photos()
        
        # Posiciones verticales para las 3 secuencias
        y_positions = [0.75, 0.50, 0.25]
        
        # Dibujar cada patrón (máximo 3)
        for pattern_idx, pattern in enumerate(patterns[:3]):
            color = pattern_colors[pattern_idx]
            y_center = y_positions[pattern_idx]
            
            # 🔥 USAR LA NUEVA FUNCIÓN QUE PRIORIZA POR POSICIÓN
            all_seqs = pattern['all_sequences']
            sequence_players = self.get_most_common_players_by_position(all_seqs)
            
            if not sequence_players:
                continue
            
            # Calcular porcentaje
            pattern_percentage = (pattern['count'] / total_throw_ins * 100) if total_throw_ins > 0 else 0

            # 🔥 NUEVO: Obtener y mostrar la secuencia de posiciones más común
            all_seqs = pattern['all_sequences']
            pos_chain, pos_perc = self._get_most_common_positional_sequence(all_seqs)

            ax.text(0.5, y_center - 0.08, f"{pos_chain} ({pos_perc:.0f}%)",
                    ha='center', va='center', fontsize=7, color='#2c3e50',
                    bbox=dict(boxstyle='round,pad=0.3', fc=color, alpha=0.3))

            # Obtener los jugadores más comunes para el ranking visual
            sequence_players = self.get_most_common_players_by_position(all_seqs)

            if not sequence_players:
                continue
            
            # Número de patrón y porcentaje (a la izquierda)
            ax.text(0.08, y_center + 0.01, f"{pattern_idx + 1}.", 
                    ha='left', va='center',
                    fontsize=16, fontweight='bold', color=color)
            
            ax.text(0.08, y_center - 0.02, f"{pattern_percentage:.0f}%", 
                    ha='left', va='center',
                    fontsize=10, fontweight='bold', color=color)
            
            # Dibujar las tarjetas de jugadores
            n_players = len(sequence_players)
            
            # Ajustar posiciones según número de jugadores
            if n_players == 2:
                x_positions = [0.35, 0.65]
                card_width = 0.13
            elif n_players == 3:
                x_positions = [0.25, 0.50, 0.75]
                card_width = 0.11
            elif n_players == 4:
                x_positions = [0.20, 0.40, 0.60, 0.80]
                card_width = 0.09
            else:
                x_positions = [0.5]
                card_width = 0.13
            
            for i, player in enumerate(sequence_players):
                x_pos = x_positions[i]
                
                # Fondo de la tarjeta con borde del color del patrón
                card_height = 0.18
                rect_bg = patches.FancyBboxPatch(
                    (x_pos - card_width/2, y_center - card_height/2), 
                    card_width, card_height,
                    boxstyle="round,pad=0.008", 
                    facecolor='white',
                    edgecolor=color,
                    linewidth=2.5,
                    zorder=10
                )
                ax.add_patch(rect_bg)
                
                # Dorsal con mejor procesamiento
                shirt_num = player.get('shirt_number', '?')
                try:
                    if shirt_num and str(shirt_num) != '?':
                        shirt_clean = str(shirt_num).strip()
                        if shirt_clean.replace('.','').isdigit():
                            dorsal = str(int(float(shirt_clean)))
                        else:
                            dorsal = '?'
                    else:
                        dorsal = '?'
                except:
                    dorsal = '?'
                
                dorsal_x = x_pos + card_width/2 - 0.012
                dorsal_y = y_center + card_height/2 - 0.015
                
                ax.text(dorsal_x, dorsal_y, dorsal, 
                        ha='right', va='top',
                        fontsize=11,
                        fontweight='heavy',
                        color=color,
                        zorder=15,
                        family='sans-serif',
                        style='italic',
                        path_effects=[
                            patheffects.Stroke(linewidth=2.5, foreground='white'),
                            patheffects.Normal()
                        ])
                
                # Foto del jugador (COPIADO EXACTAMENTE DE TACTIC2.1)
                player_name = player.get('player_name', 'N/A')
                player_photo = self.get_player_photo(player_name)
                
                if player_photo is not None:
                    photo_size = 0.095
                    photo_y_offset = 0.015
                    photo_ax = ax.inset_axes([
                        x_pos - photo_size/2, 
                        y_center + photo_y_offset - photo_size/2,
                        photo_size, 
                        photo_size
                    ])
                    photo_ax.imshow(player_photo)
                    photo_ax.axis('off')
                else:
                    # Si no hay foto, mostrar iniciales
                    if player_name and player_name != 'N/A' and player_name != 'Desconocido (sin datos)':
                        parts = player_name.split()
                        initials = ''.join([p[0].upper() for p in parts[:2] if len(p) > 0])
                        ax.text(x_pos, y_center + 0.015, initials, 
                            ha='center', va='center',
                            fontsize=10, fontweight='bold', 
                            color=color, zorder=15,
                            path_effects=[
                                patheffects.Stroke(linewidth=2, foreground='white'),
                                patheffects.Normal()
                            ])
                
                # Nombre (apellido) con mejor contraste
                apellido = player_name.split()[-1].upper() if (player_name and player_name != 'N/A') else 'N/A'

                # Badge con el color del patrón
                badge_height = 0.020
                name_badge = patches.FancyBboxPatch(
                    (x_pos - card_width/2 + 0.005, y_center - card_height/2 + 0.005), 
                    card_width - 0.01, badge_height,
                    boxstyle="round,pad=0.002",
                    facecolor=color,
                    edgecolor='white',
                    linewidth=1.5,
                    alpha=1.0,
                    zorder=14
                )
                ax.add_patch(name_badge)

                # Texto blanco con sombra negra para máximo contraste
                ax.text(x_pos, y_center - card_height/2 + 0.015, 
                        apellido, 
                        ha='center', va='center',
                        fontsize=5.5, fontweight='bold', 
                        color='white', zorder=15,
                        path_effects=[
                            patheffects.Stroke(linewidth=1.5, foreground='black'),
                            patheffects.Normal()
                        ])
                
                # Flecha entre jugadores
                if i < len(sequence_players) - 1:
                    arrow_start_x = x_pos + card_width/2 + 0.01
                    arrow_end_x = x_positions[i + 1] - card_width/2 - 0.01
                    arrow_y = y_center
                    
                    arrow = patches.FancyArrowPatch(
                        (arrow_start_x, arrow_y), 
                        (arrow_end_x, arrow_y),
                        arrowstyle='->,head_width=0.15,head_length=0.15',
                        color=color,
                        linewidth=2,
                        zorder=5,
                        alpha=0.8
                    )
                    ax.add_patch(arrow)
            
            # Texto de frecuencia
            ax.text(0.98, y_center - 0.02, f'{pattern["count"]}x', 
                    ha='right', va='center',
                    fontsize=8, color=color, fontweight='bold')
            
            # Línea separadora entre patrones
            if pattern_idx < len(patterns[:3]) - 1:
                ax.plot([0.05, 0.95], [y_center - 0.12, y_center - 0.12], 
                    color='#dfe6e9', linewidth=1, transform=ax.transData)
    
    def draw_zone_statistics_combined(self, ax, classification):
        """
        Dibuja estadísticas combinadas de ambos lados en las columnas centrales
        """
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Título principal
        ax.text(0.5, 1, 'ESTADÍSTICAS - PATRONES Y LANZADORES', 
               ha='center', va='top', fontsize=11, 
               fontweight='bold', color='#1e3d59',
               transform=ax.transAxes)
        
        # Subtítulos de columnas
        ax.text(0.25, 0.95, 'IZQUIERDA', ha='center', va='top',
               fontsize=10, fontweight='bold', color='#2d3436',
               transform=ax.transAxes)
        
        ax.text(0.75, 0.95, 'DERECHA', ha='center', va='top',
               fontsize=10, fontweight='bold', color='#2d3436',
               transform=ax.transAxes)
        
        # Línea divisoria vertical
        ax.plot([0.5, 0.5], [0.05, 0.88], color='#2d3436', 
               linewidth=2, transform=ax.transAxes, zorder=3)
        
        # Estadísticas por zona (ahora en el orden correcto)
        y_start = 0.78
        zone_height = 0.22
        
        # Zona 3 arriba (ofensiva), Zona 1 abajo (defensiva)
        for i, (zone_name, zone_color, zone_desc) in enumerate([
            ('Zona 3', '#fd79a8', 'Ofensiva'),
            ('Zona 2', '#6c5ce7', 'Media'),
            ('Zona 1', '#0984e3', 'Defensiva')
        ]):
            y_pos = y_start - (i * zone_height)
            
            # Título de la zona (centrado)
            ax.text(0.5, y_pos + 0.02, f'{zone_name} ({zone_desc})', 
                   ha='center', va='top',
                   fontsize=9, fontweight='bold', color=zone_color,
                   transform=ax.transAxes)
            
            # Estadísticas IZQUIERDA
            throw_ins_izq = classification['Izquierda'][zone_name]
            total_izq = len(throw_ins_izq)
            exitosos_izq = sum(1 for t in throw_ins_izq if t.get('outcome', 0) == 1)
            porcentaje_izq = (exitosos_izq / total_izq * 100) if total_izq > 0 else 0
            
            ax.text(0.25, y_pos - 0.05, f'{total_izq} saques', 
                   ha='center', va='center', fontsize=8, color='#2d3436',
                   transform=ax.transAxes)
            ax.text(0.25, y_pos - 0.10, f'{porcentaje_izq:.1f}% éxito', 
                   ha='center', va='center', fontsize=8, 
                   color='#00b894' if porcentaje_izq >= 60 else '#636e72',
                   fontweight='bold', transform=ax.transAxes)
            
            # Estadísticas DERECHA
            throw_ins_der = classification['Derecha'][zone_name]
            total_der = len(throw_ins_der)
            exitosos_der = sum(1 for t in throw_ins_der if t.get('outcome', 0) == 1)
            porcentaje_der = (exitosos_der / total_der * 100) if total_der > 0 else 0
            
            ax.text(0.75, y_pos - 0.05, f'{total_der} saques', 
                   ha='center', va='center', fontsize=8, color='#2d3436',
                   transform=ax.transAxes)
            ax.text(0.75, y_pos - 0.10, f'{porcentaje_der:.1f}% éxito', 
                   ha='center', va='center', fontsize=8,
                   color='#00b894' if porcentaje_der >= 60 else '#636e72',
                   fontweight='bold', transform=ax.transAxes)
            
            # Línea separadora entre zonas
            if i < 2:
                ax.plot([0.05, 0.95], [y_pos - 0.15, y_pos - 0.15], 
                       color='#dfe6e9', linewidth=1, transform=ax.transAxes)
        
        # TOTALES GENERALES en la parte inferior
        y_total = 0.12
        
        ax.text(0.5, y_total + 0.03, 'TOTAL GENERAL', ha='center', va='center',
               fontsize=9, fontweight='bold', color='#1e3d59',
               transform=ax.transAxes)
        
        # Total izquierda
        all_izq = []
        for zone_name in ['Zona 1', 'Zona 2', 'Zona 3']:
            all_izq.extend(classification['Izquierda'][zone_name])
        total_all_izq = len(all_izq)
        exitosos_all_izq = sum(1 for t in all_izq if t.get('outcome', 0) == 1)
        porcentaje_all_izq = (exitosos_all_izq / total_all_izq * 100) if total_all_izq > 0 else 0
        
        ax.text(0.25, y_total - 0.03, f'{total_all_izq} | {porcentaje_all_izq:.1f}%',
               ha='center', va='center', fontsize=9, color='#2d3436',
               fontweight='bold', transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#dfe6e9', 
                        edgecolor='#2d3436', linewidth=1.5))
        
        # Total derecha
        all_der = []
        for zone_name in ['Zona 1', 'Zona 2', 'Zona 3']:
            all_der.extend(classification['Derecha'][zone_name])
        total_all_der = len(all_der)
        exitosos_all_der = sum(1 for t in all_der if t.get('outcome', 0) == 1)
        porcentaje_all_der = (exitosos_all_der / total_all_der * 100) if total_all_der > 0 else 0
        
        ax.text(0.75, y_total - 0.03, f'{total_all_der} | {porcentaje_all_der:.1f}%',
               ha='center', va='center', fontsize=9, color='#2d3436',
               fontweight='bold', transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#dfe6e9', 
                        edgecolor='#2d3436', linewidth=1.5))
    
    def load_background(self):
        """Carga la imagen de fondo si existe"""
        return plt.imread("assets/fondo_informes.png") if os.path.exists("assets/fondo_informes.png") else None

    def load_tactic_logo(self):
        """Carga el logo de Tactic"""
        return plt.imread("assets/tactic_logo.png") if os.path.exists("assets/tactic_logo.png") else None

    def load_team_logo(self, equipo, target_size=(80, 80)):
        """Carga el logo del equipo con búsqueda inteligente"""
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            return None
        
        if not os.path.exists('assets/escudos'):
            return None
        
        def normalize_word(word):
            word = unicodedata.normalize('NFD', word)
            word = ''.join(char for char in word if unicodedata.category(char) != 'Mn')
            return word.lower().strip()
        
        palabras_ignorar = {'cf', 'fc', 'cd', 'ud', 'rcd', 'rc', 'ca', 'de', 'del', 'la', 'las', 'el', 'los'}
        
        palabras = equipo.split()
        palabras_normalizadas = []
        
        for palabra in palabras:
            palabra_norm = normalize_word(palabra)
            if palabra_norm not in palabras_ignorar and len(palabra_norm) > 2:
                palabras_normalizadas.append(palabra_norm)
        
        palabras_ordenadas = sorted(palabras_normalizadas, key=len, reverse=True)
        
        all_files = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
        
        for palabra_buscar in palabras_ordenadas:
            for filename in all_files:
                nombre_archivo = os.path.splitext(filename)[0]
                nombre_archivo_norm = normalize_word(nombre_archivo)
                
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
                        continue
        
        best_match_path = None
        best_score = 0
        equipo_completo_norm = normalize_word(equipo.replace(' ', ''))
        
        for filename in all_files:
            nombre_archivo = os.path.splitext(filename)[0]
            nombre_archivo_norm = normalize_word(nombre_archivo)
            from difflib import SequenceMatcher
            score = SequenceMatcher(None, equipo_completo_norm, nombre_archivo_norm).ratio()
            
            if score > best_score:
                best_score = score
                best_match_path = f"assets/escudos/{filename}"
        
        if best_match_path and best_score > 0.5:
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
            except:
                pass
        
        return None
    
    def create_visualization(self, team_name):
        """
        Crea la visualización completa del análisis de saques de banda
        Formato A4 horizontal con 5 columnas y estadísticas por zona
        """
        # 1. Extracción de secuencias
        sequences = self.extract_throw_in_sequences(team_name)
        
        if not sequences:
            print("❌ No se encontraron saques de banda para analizar.")
            fig, ax = plt.subplots(figsize=(11.69, 8.27), facecolor='white')
            ax.text(0.5, 0.5, f'No se encontraron saques de banda\npara {team_name}', 
                ha='center', va='center', fontsize=18, color='red')
            ax.axis('off')
            return fig
        
        print(f"✅ Encontrados {len(sequences)} saques de banda")
        
        # 2. Clasificación por zona y lado
        classification = self.classify_throw_ins_by_zone_and_side(sequences)
        
        # 3. Configuración de la figura (A4 horizontal)
        fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
        
        # Fondo si existe
        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.15)
            ax_bg.axis('off')
        
        # Grid: 3 filas x 5 columnas (columnas 1 y 3 estrechas para separar)
        gs = fig.add_gridspec(3, 5, 
                        height_ratios=[1, 1, 1],
                        # Cambiamos el width_ratios para dar más espacio al centro
                        width_ratios=[1, 0.05, 2.8, 0.05, 1], 
                        wspace=0.01, hspace=0.01,
                        left=0.04, right=0.98, top=0.88, bottom=0.05)

        
        # 4. Títulos y logos
        fig.suptitle('Saques de banda Ofensivos',
                    fontsize=16, fontweight='bold', color='#1e3d59', 
                    y=0.96, family='serif')
        
        # Logos
        if (tactic_logo := self.load_tactic_logo()) is not None:
            ax_logo1 = fig.add_axes([0.02, 0.90, 0.08, 0.08], anchor='NW', zorder=10)
            ax_logo1.imshow(tactic_logo)
            ax_logo1.axis('off')
        
        if (team_logo := self.load_team_logo(team_name)) is not None:
            ax_logo2 = fig.add_axes([0.90, 0.90, 0.08, 0.08], anchor='NE', zorder=10)
            ax_logo2.imshow(team_logo)
            ax_logo2.axis('off')
        
        # 5. Título central y etiquetas de lado (MÁS ARRIBA)
        fig.text(0.5, 0.92, 'ESTADÍSTICAS POR ZONA Y LADO', 
                fontsize=11, ha='center', fontweight='bold', 
                color='#1e3d59', family='sans-serif')
        
        fig.text(0.13, 0.90, 'Izquierda', fontsize=10, ha='center', 
                fontweight='bold', color='#2d3436', family='sans-serif')
        fig.text(0.87, 0.90, 'Derecha', fontsize=10, ha='center', 
                fontweight='bold', color='#2d3436', family='sans-serif')
        
        # 6. Dibujar campogramas en columnas 0 (izquierda) y 4 (derecha)
        # FILA 0 - ZONA 3 (ofensiva) - ARRIBA
        ax_izq_z3 = fig.add_subplot(gs[0, 0])
        self.draw_pitch_with_throw_ins(ax_izq_z3, 
                                    classification['Izquierda']['Zona 3'],
                                    3, 'Izquierda')

        ax_der_z3 = fig.add_subplot(gs[0, 4])
        self.draw_pitch_with_throw_ins(ax_der_z3, 
                                    classification['Derecha']['Zona 3'],
                                    3, 'Derecha')
        
        # FILA 1 - ZONA 2 (media)
        ax_izq_z2 = fig.add_subplot(gs[1, 0])
        self.draw_pitch_with_throw_ins(ax_izq_z2, 
                                    classification['Izquierda']['Zona 2'],
                                    2, 'Izquierda')
        
        ax_der_z2 = fig.add_subplot(gs[1, 4])
        self.draw_pitch_with_throw_ins(ax_der_z2, 
                                    classification['Derecha']['Zona 2'],
                                    2, 'Derecha')
        
        # FILA 2 - ZONA 1 (defensiva) - ABAJO
        ax_izq_z1 = fig.add_subplot(gs[2, 0])
        self.draw_pitch_with_throw_ins(ax_izq_z1, 
                                    classification['Izquierda']['Zona 1'],
                                    1, 'Izquierda')

        ax_der_z1 = fig.add_subplot(gs[2, 4])
        self.draw_pitch_with_throw_ins(ax_der_z1, 
                                    classification['Derecha']['Zona 1'],
                                    1, 'Derecha')
        
        # 7. Paneles centrales para los RANKINGS DE LANZADORES (con sub-grid para Izquierda/Derecha)
        
        # --- ZONA 3 (Ofensiva) ---
        # Creamos un sub-grid en la celda central de la primera fila
        gs_z3 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 2], wspace=0.1)
        ax_rank_z3_izq = fig.add_subplot(gs_z3[0, 0])
        ax_rank_z3_der = fig.add_subplot(gs_z3[0, 1])

        # 🔥 NUEVO: Ranking de lanzadores en lugar de secuencias
        self.draw_throwers_ranking(ax_rank_z3_izq, classification['Izquierda']['Zona 3'], 
                                   title="ZONA 3 - IZQUIERDA")
        self.draw_throwers_ranking(ax_rank_z3_der, classification['Derecha']['Zona 3'], 
                                   title="ZONA 3 - DERECHA")

        # --- ZONA 2 (Media) ---
        gs_z2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 2], wspace=0.1)
        ax_rank_z2_izq = fig.add_subplot(gs_z2[0, 0])
        ax_rank_z2_der = fig.add_subplot(gs_z2[0, 1])

        # 🔥 NUEVO: Ranking de lanzadores
        self.draw_throwers_ranking(ax_rank_z2_izq, classification['Izquierda']['Zona 2'], 
                                   title="ZONA 2 - IZQUIERDA")
        self.draw_throwers_ranking(ax_rank_z2_der, classification['Derecha']['Zona 2'], 
                                   title="ZONA 2 - DERECHA")
        
        # --- ZONA 1 (Defensiva) ---
        gs_z1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 2], wspace=0.1)
        ax_rank_z1_izq = fig.add_subplot(gs_z1[0, 0])
        ax_rank_z1_der = fig.add_subplot(gs_z1[0, 1])

        # 🔥 NUEVO: Ranking de lanzadores
        self.draw_throwers_ranking(ax_rank_z1_izq, classification['Izquierda']['Zona 1'], 
                                   title="ZONA 1 - IZQUIERDA")
        self.draw_throwers_ranking(ax_rank_z1_der, classification['Derecha']['Zona 1'], 
                                   title="ZONA 1 - DERECHA")
        
        # 8. Líneas verticales de separación (columnas 1 y 3)
        fig.add_artist(plt.Line2D([0.24, 0.24], [0.05, 0.88], 
                                transform=fig.transFigure, color='#2d3436', linewidth=2.5))
        fig.add_artist(plt.Line2D([0.78, 0.78], [0.05, 0.88], 
                                transform=fig.transFigure, color='#2d3436', linewidth=2.5))
        
        
        return fig
    
    def print_summary(self, team_name):
        """Imprime un resumen del análisis"""
        sequences = self.extract_throw_in_sequences(team_name)
        
        print(f"\n{'='*60}")
        print(f"RESUMEN DE SAQUES DE BANDA OFENSIVOS")
        print(f"{'='*60}")
        print(f"Equipo: {team_name}")
        print(f"Total de saques de banda: {len(sequences)}")
        
        if sequences:
            classification = self.classify_throw_ins_by_zone_and_side(sequences)
            
            for side in ['Izquierda', 'Derecha']:
                print(f"\n{side}:")
                # Mostrar en orden: Zona 3 (ofensiva), Zona 2 (media), Zona 1 (defensiva)
                for zone in ['Zona 3', 'Zona 2', 'Zona 1']:
                    throw_ins = classification[side][zone]
                    total = len(throw_ins)
                    exitosos = sum(1 for t in throw_ins if t.get('outcome', 0) == 1)
                    porcentaje = (exitosos / total * 100) if total > 0 else 0
                    zone_desc = {'Zona 3': 'Ofensiva', 'Zona 2': 'Media', 'Zona 1': 'Defensiva'}[zone]
                    print(f"  {zone} ({zone_desc}): {total} saques ({porcentaje:.1f}% éxito)")

def seleccionar_equipo_interactivo():
    """Función para seleccionar un equipo de forma interactiva"""
    try:
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/open_play_events.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        
        if not equipos:
            print("No se encontraron equipos.")
            return None
        
        print("\n" + "="*60)
        print("SELECCIÓN DE EQUIPO")
        print("="*60)
        for i, equipo in enumerate(equipos, 1):
            print(f"{i:2d}. {equipo}")
        
        while True:
            try:
                idx = int(input(f"\nSelecciona un equipo (1-{len(equipos)}): ").strip()) - 1
                if 0 <= idx < len(equipos):
                    return equipos[idx]
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
        print("\n" + "="*60)
        print("ANÁLISIS DE SAQUES DE BANDA OFENSIVOS")
        print("="*60)
        
        equipo = seleccionar_equipo_interactivo()
        if equipo is None:
            print("No se pudo completar la selección.")
            return
        
        print(f"\nAnalizando saques de banda para {equipo}...")
        
        analyzer = AnalizadorSaquesBanda()
        
        if not analyzer.load_data(team_filter=equipo):
            return
        
        # 🔥 AÑADIR DEBUG AQUÍ
        print("\n🔍 Generando archivo de debug...")
        equipo_filename = re.sub(r'[\s/]', '_', equipo)
        debug_file = f"debug_saques_banda_{equipo_filename}.txt"
        analyzer.debug_throw_in_sequences(equipo, output_file=debug_file)
        print(f"✅ Archivo de debug creado: {debug_file}")
        print("⚠️ REVISA EL ARCHIVO ANTES DE CONTINUAR\n")
        
        # Resumen
        analyzer.print_summary(equipo)
        
        # Visualización
        fig = analyzer.create_visualization(equipo)
        
        if fig:
            output_path = f"saques_banda_ofensivos_{equipo_filename}.pdf"
            fig.savefig(output_path, bbox_inches='tight', 
                       pad_inches=0.1, facecolor='white', dpi=300)
            print(f"\n✅ Visualización guardada como: {output_path}")
            plt.show()
        else:
            print("❌ No se pudo generar la visualización")
    
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nINICIALIZANDO ANALIZADOR DE SAQUES DE BANDA")
    try:
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/open_play_events.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        print(f"✅ Sistema listo. Equipos disponibles: {len(equipos)}")
        if equipos:
            print("Para ejecutar el análisis, ejecute: main()")
        main()
    except Exception as e:
        print(f"❌ Error al inicializar: {e}")
        print("\nAsegúrate de que el archivo 'open_play_events.parquet' existe en:")
        print("  extraccion_opta/datos_opta_parquet/open_play_events.parquet")