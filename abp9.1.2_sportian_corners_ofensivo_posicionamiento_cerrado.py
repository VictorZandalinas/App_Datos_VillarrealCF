import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image, ImageDraw
from mplsoccer import VerticalPitch
from sklearn.cluster import KMeans
import textwrap 
import warnings
import unicodedata
import re
from collections import Counter
import json
from difflib import SequenceMatcher
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')

class ReporteOfensivoCornersBilateral:
    def __init__(self, tracking_path="extraccion_sportian/corners_tracking.parquet", team_filter=None):
        self.tracking_path = tracking_path
        self.team_filter = team_filter
        self.df_tracking = None
        self.historial_lanzadores = {} 
        self.team_stats = None
        
        # Almacenes de datos separados
        self.data_left = []
        self.data_right = []
        
        # Mapas de metadatos (Nombre y Altura)
        self.player_map = {}
        self.player_height_map = {} 

        # --- 1. CARGA DE EVENTOS (OPTA) ---
        try:
            self.df_events = pd.read_parquet("extraccion_opta/datos_opta_parquet/abp_events.parquet")
            self.df_events['timeStamp'] = pd.to_datetime(self.df_events['timeStamp'].astype(str).str.replace('Z', ''), errors='coerce')
            self.df_events['Match ID'] = self.df_events['Match ID'].astype(str)
        except:
            self.df_events = pd.DataFrame()

        # --- 2. CARGA DE ALTURAS (METADATOS LIGA) ---
        id_to_height = {}
        try:
            df_meta = pd.read_parquet("extraccion_opta/datos_opta_parquet/jugadores_liga24_25.parquet")
            df_meta['player_id'] = df_meta['player_id'].astype(str).str.replace('.0', '', regex=False)
            id_to_height = dict(zip(df_meta['player_id'], df_meta['height']))
            print(f"✅ Metadatos de altura cargados: {len(id_to_height)} jugadores.")
        except Exception as e:
            print(f"⚠️ No se pudo cargar fichero de alturas (jugadores_liga24_25.parquet): {e}")

        # --- 3. CARGA xG EVENTS ---
        try:
            self.df_xg = pd.read_parquet("extraccion_opta/datos_opta_parquet/xg_events.parquet")
            self.df_xg['timeStamp'] = pd.to_datetime(self.df_xg['timeStamp'].astype(str).str.replace('Z', ''), errors='coerce')
            self.df_xg['Match ID'] = self.df_xg['Match ID'].astype(str)
            print(f"✅ xG Events cargados: {len(self.df_xg)}")
        except Exception as e:
            print(f"⚠️ No se pudo cargar xg_events.parquet: {e}")
            self.df_xg = pd.DataFrame()

        # --- 4. MERGE xG CON EVENTOS (AHORA QUE AMBOS ESTÁN CARGADOS) ---
        if not self.df_events.empty and not self.df_xg.empty:
            try:
                xg_data = self.df_xg[['timeStamp', 'playerId', 'Match ID', 'qualifier 321']].copy()
                xg_data = xg_data.rename(columns={'qualifier 321': 'xg_value'})
                xg_data['xg_value'] = pd.to_numeric(xg_data['xg_value'], errors='coerce')
                self.df_events = pd.merge(self.df_events, xg_data, on=['timeStamp', 'playerId', 'Match ID'], how='left')
                print(f"✅ Merge xG exitoso: {self.df_events['xg_value'].notna().sum()} eventos con xG")
            except Exception as e:
                print(f"⚠️ Error en merge xG: {e}")
                self.df_events['xg_value'] = None

        # --- 5. CARGA DE OPTA STATS Y MAPEO (NOMBRE Y ALTURA) ---
        try:
            self.team_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/team_stats.parquet")
            df_players = pd.read_parquet("extraccion_opta/datos_opta_parquet/player_stats.parquet")
            
            for _, row in df_players.iterrows():
                team_key = str(row.get('Team Name', '')).strip().lower()
                dorsal_val = row.get('Shirt Number', 0)
                try: 
                    dorsal_key = str(int(float(dorsal_val)))
                except: 
                    dorsal_key = str(dorsal_val)
                
                name_val = row.get('Match Name') 
                if pd.isna(name_val): 
                    name_val = row.get('Player Name') 
                
                self.player_map[(team_key, dorsal_key)] = name_val
                
                p_id = str(row.get('Player ID', '')).replace('.0', '')
                if p_id in id_to_height:
                    self.player_height_map[(team_key, dorsal_key)] = id_to_height[p_id]
                
        except Exception as e:
            print(f"⚠️ Warning carga Opta Stats: {e}")
        
        # --- 6. CARGA DEL TRACKING ---
        self.load_tracking_data()
        
        if self.df_tracking is not None:
            self.construir_historial_lanzadores()

        # --- 6.5 CONSTRUIR DICCIONARIO xG POR JUGADOR (OPTA) ---
        self.xg_por_jugador = self.build_xg_by_player() if team_filter else {}
        
        # --- 7. EJECUCIÓN DEL ANÁLISIS ---
        if team_filter:
            print(f"\n🚀 GENERANDO REPORTE BILATERAL (CERRADOS) PARA: {team_filter}")
            print("   ↳ 🔄 Analizando Córners Izquierda...")
            self.data_left = self.extract_data(side_req='izquierda', type_req='Cerrado')
            print("   ↳ 🔄 Analizando Córners Derecha...")
            self.data_right = self.extract_data(side_req='derecha', type_req='Cerrado')
    
    def get_opta_name(self, team, dorsal, tracking_name):
        """Busca el nombre en Opta usando Equipo + Dorsal. Si falla, devuelve el del tracking."""
        t = str(team).strip().lower()
        d = self.clean_dorsal_str(dorsal) 
        
        opta_name = self.player_map.get((t, d))
        
        if opta_name:
            return opta_name
        return tracking_name
    
    def get_opta_event_raw(self, jornada, team_tracking, min_track, sec_track):
        """Recupera la fila del evento de Opta para leer Pass End X/Y."""
        if self.df_events.empty: return None
        
        subset = self.df_events[self.df_events['Week'].astype(str) == str(jornada)]
        if subset.empty: return None

        time_track = min_track * 60 + sec_track
        
        # Margen de tolerancia de tiempo (10 segundos)
        for _, row in subset.iterrows():
            # Filtro Equipo
            t_evt = str(row.get('Team Name', '')).lower().strip()
            t_trk = str(team_tracking).lower().strip()
            if t_evt not in t_trk and t_trk not in t_evt:
                if SequenceMatcher(None, t_evt, t_trk).ratio() < 0.6: continue
            
            try:
                min_evt = float(row.get('timeMin', 0))
                sec_evt = float(row.get('timeSec', 0))
                time_evt = min_evt * 60 + sec_evt
                
                # Si coincide en tiempo, devolvemos la fila entera
                if abs(time_evt - time_track) <= 10:
                    return row
            except: continue
        return None
    
    def build_xg_by_player(self):
        """Construye diccionario de xG acumulado por jugador desde Opta (solo córners cerrados)."""
        xg_por_jugador = {}
        
        if self.df_events.empty:
            return xg_por_jugador
        
        try:
            # Filtrar por equipo
            team_lower = str(self.team_filter).lower().strip()
            df_team = self.df_events[self.df_events['Team Name'].str.lower().str.strip() == team_lower].copy()
            
            if df_team.empty:
                print(f"⚠️ No hay eventos para {self.team_filter}")
                return xg_por_jugador
            
            # Ordenar por partido y tiempo
            df_team = df_team.sort_values(['Match ID', 'timeMin', 'timeSec']).reset_index(drop=True)
            
            # Encontrar córners cerrados (In-swinger = Sí)
            corners = df_team[(df_team['Corner taken'] == 'Sí') & (df_team['In-swinger'] == 'Sí')]
            
            print(f"✅ Córners cerrados (In-swinger) para {self.team_filter}: {len(corners)}")
            
            for _, corner in corners.iterrows():
                match_id = corner['Match ID']
                time_corner = corner['timeMin'] * 60 + corner['timeSec']
                
                # Buscar remate en los 5 segundos siguientes del mismo partido y equipo
                remate_events = ['Goal', 'Miss', 'Attempt Saved', 'Post']
                
                mask = (df_team['Match ID'] == match_id) & \
                    (df_team['Event Name'].isin(remate_events))
                
                remates = df_team[mask]
                
                for _, remate in remates.iterrows():
                    time_remate = remate['timeMin'] * 60 + remate['timeSec']
                    diff = time_remate - time_corner
                    
                    if 0 < diff <= 5:
                        jugador = remate.get('playerName', '')
                        xg_val = remate.get('xg_value', 0)
                        xg_val = float(xg_val) if pd.notna(xg_val) else 0.0
                        
                        if jugador:
                            if jugador not in xg_por_jugador:
                                xg_por_jugador[jugador] = 0.0
                            xg_por_jugador[jugador] += xg_val
                        break  # Solo el primer remate cuenta
            
            print(f"✅ xG por jugador construido: {len(xg_por_jugador)} jugadores con xG")
            for jug, xg in xg_por_jugador.items():
                if xg > 0:
                    print(f"   • {jug}: {xg:.3f}")
            
        except Exception as e:
            print(f"⚠️ Error construyendo xG por jugador: {e}")
        
        return xg_por_jugador

    def detectar_momento_remate(self, df_corner, target_x, target_y, start_time):
        """
        Busca en el tracking el momento (timestamp) donde el balón está 
        MÁS CERCA de las coordenadas de destino de Opta (Pass End X/Y).
        Busca solo DESPUÉS del saque.
        """
        # Filtramos solo frames posteriores al saque
        df_flight = df_corner[df_corner['Segundos_Desde_Saque'] > start_time].copy()
        
        if df_flight.empty: return None, None, None

        # Coordenadas del balón en tracking
        ball_coords = df_flight[['X_Balon', 'Y_Balon']].values
        timestamps = df_flight['Segundos_Desde_Saque'].values
        
        # Calcular distancia Euclídea a las coordenadas destino de Opta
        # dist = sqrt((x - tx)^2 + (y - ty)^2)
        dists = np.hypot(ball_coords[:, 0] - target_x, ball_coords[:, 1] - target_y)
        
        # Encontrar el índice de la distancia mínima
        min_idx = np.argmin(dists)
        
        # Si la distancia mínima es muy grande (ej: > 10m), quizás no llegó o el tracking falló
        # pero devolveremos el mejor candidato de todas formas.
        
        best_time = timestamps[min_idx]
        best_x = ball_coords[min_idx][0]
        best_y = ball_coords[min_idx][1]
        
        return best_time, best_x, best_y
    
    def get_player_height(self, team_tracking, dorsal_tracking):
        """Busca la altura usando la misma lógica que el nombre."""
        t = str(team_tracking).strip().lower()
        d = self.clean_dorsal_str(dorsal_tracking)
        
        return self.player_height_map.get((t, d), None)
    
    def filter_photos_by_team(self, all_photos):
        """Filtra la lista completa de fotos y devuelve solo las del equipo actual."""
        if not self.team_filter: return []
        
        target_team = str(self.team_filter).lower().strip()
        filtered = []
        
        def clean(t): 
            return re.sub(r'[^\w\s]', '', str(t).lower()).replace('cf','').replace('fc','').strip()

        target_clean = clean(target_team)

        for photo in all_photos:
            p_team = str(photo.get('team', '') or photo.get('team_name', '')).lower()
            p_team_clean = clean(p_team)
            
            if target_clean in p_team_clean or p_team_clean in target_clean:
                filtered.append(photo)
                
        print(f"📸 Fotos filtradas para {self.team_filter}: {len(filtered)} encontradas.")
        return filtered

    def get_sequence_xg(self, match_id, time_kick_abs, team_id_opta=None):
        """Busca el xG acumulado en los 5s posteriores al saque."""
        if self.df_xg.empty: return 0.0, None
        
        # Convertir match_id a string para asegurar compatibilidad
        match_id = str(match_id)
        
        # Filtrar por partido
        subset = self.df_xg[self.df_xg['Match ID'] == match_id]
        if subset.empty: return 0.0, None

        # Opta Events tiene timestamps absolutos. 
        # Como time_kick viene del tracking (segundos relativos), necesitamos sincronizar.
        # Asumimos que la sincronización ya se hace en extract_data, 
        # pero aquí buscaremos usando la hora del evento de Opta si es posible,
        # o aproximando por la secuencia de eventos si ya tenemos el link.
        
        # NOTA: Para simplificar la integración en este script híbrido (Sportian/Opta),
        # usaremos el "jornada" y "tiempo de juego" si estuvieran disponibles, 
        # pero lo más robusto aquí es cruzar por la ventana de tiempo del evento de córner de Opta ya localizado.
        
        return 0.0, None # Placeholder si no tenemos el timestamp absoluto del kick.
    
    def detectar_inicio_saque(self, df_corner, lado):
        """
        Detecta el índice y tiempo exacto donde comienza el lanzamiento.
        Lógica: X > 98 y Y cerca de 0/100, y en el siguiente frame se mueve.
        """
        df = df_corner.sort_values('Segundos_Desde_Saque').reset_index(drop=True)
        coords = df[['X_Balon', 'Y_Balon', 'Segundos_Desde_Saque']].values
        
        start_idx = -1
        
        target_y_start = 100 if lado == 'izquierda' else 0
        
        for i in range(len(coords) - 5):
            x_curr, y_curr, t_curr = coords[i]
            x_next, y_next, t_next = coords[i+1]
            
            en_zona_corner = (x_curr > 96) and (abs(y_curr - target_y_start) < 5)
            
            if en_zona_corner:
                delta_y = y_next - y_curr
                distancia = np.hypot(x_next - x_curr, y_next - y_curr)
                
                moving_towards_center = (lado == 'izquierda' and delta_y < -0.1) or \
                                        (lado == 'derecha' and delta_y > 0.1)
                
                if distancia > 0.2 and moving_towards_center:
                    start_idx = i
                    break
        
        if start_idx != -1:
            return start_idx, coords[start_idx][2] 
        return None, None
    
    def get_opta_trajectory(self, jornada, team_tracking, min_track, sec_track):
        """Busca el evento en Opta y devuelve la línea recta."""
        if self.df_events.empty: return None
        
        subset = self.df_events[self.df_events['Week'].astype(str) == str(jornada)]
        if subset.empty: return None

        time_track = min_track * 60 + sec_track
        best_match = None
        
        for _, row in subset.iterrows():
            t_evt = str(row.get('Team Name', '')).lower().strip()
            t_trk = str(team_tracking).lower().strip()
            if t_evt not in t_trk and t_trk not in t_evt:
                if SequenceMatcher(None, t_evt, t_trk).ratio() < 0.6: continue
            
            try:
                min_evt = float(row.get('timeMin', 0))
                sec_evt = float(row.get('timeSec', 0))
            except: continue
            
            time_evt = min_evt * 60 + sec_evt
            if abs(time_evt - time_track) <= 10:
                best_match = row
                break
        
        if best_match is not None:
            try:
                x_start = float(best_match.get('x'))
                y_start = float(best_match.get('y'))
                x_end = float(best_match.get('Pass End X'))
                y_end = float(best_match.get('Pass End Y'))
                
                xs = np.linspace(x_start, x_end, 5)
                ys = np.linspace(y_start, y_end, 5)
                return np.column_stack((xs, ys))
            except (ValueError, TypeError):
                return None
            
        return None
    
    def extract_player_trajectories(self, df_corner, time_kick, attacking_team):
        """
        Captura trayectorias desde (time_kick - 3s) hasta time_kick.
        """
        if time_kick is None: return {}

        time_start = time_kick - 3.0
        time_end = time_kick + 3.0 
        
        mask_time = (df_corner['Segundos_Desde_Saque'] >= time_start) & \
                    (df_corner['Segundos_Desde_Saque'] <= time_end)
        
        df = df_corner[mask_time].copy()
        df = df[df['NombreEquipoJugador_Tracking'] == attacking_team]
        
        trajectories = {}
        for player_name, group in df.groupby('Nombre_Jugador_Tracking'):
            group = group.sort_values('Segundos_Desde_Saque')
            coords = group[['X_Jugador', 'Y_Jugador']].values
            if len(coords) > 2:
                trajectories[player_name] = coords
                
        return trajectories
    
    def load_tracking_data(self):
        try:
            print("📂 Cargando datos de tracking...")
            self.df_tracking = pd.read_parquet(self.tracking_path)
            print(f"✅ Datos cargados: {len(self.df_tracking)} frames")
        except Exception as e:
            print(f"❌ Error cargando tracking: {e}")
    
    def clean_dorsal_str(self, d):
        try:
            return str(int(float(d))) if str(d).replace('.','',1).isdigit() else str(d)
        except:
            return str(d)

    def load_player_photos(self):
        try:
            with open('assets/jugadores_optimizados.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return []

    def match_player_name(self, player_name, photos_data):
        if pd.isna(player_name) or not photos_data: return None
        
        def normalize(name):
            n = unicodedata.normalize('NFD', str(name)).encode('ascii', 'ignore').decode("utf-8")
            return re.sub(r'[^\w\s]', '', n.lower().strip())

        norm_player = normalize(player_name)
        player_tokens = set(norm_player.split())
        
        best_match = None
        highest_score = 0.0

        for photo in photos_data:
            photo_name = photo.get('name') or photo.get('player_name')
            if not photo_name: continue
            
            norm_photo = normalize(photo_name)
            photo_tokens = set(norm_photo.split())
            
            if norm_player == norm_photo:
                return photo
            
            common = player_tokens.intersection(photo_tokens)
            
            score = 0
            if len(common) > 0:
                match_ratio = len(common) / max(len(player_tokens), len(photo_tokens))
                score = 0.7 + (match_ratio * 0.3)
                
                if norm_player.split()[-1] == norm_photo.split()[-1]:
                    score += 0.1

            fuzzy = SequenceMatcher(None, norm_player, norm_photo).ratio()
            if fuzzy > score:
                score = fuzzy

            if score > highest_score and score > 0.55: 
                highest_score = score
                best_match = photo

        return best_match

    def create_circular_player_photo(self, player_name, photos_data, size=(90, 90)):
        match = self.match_player_name(player_name, photos_data)
        if not match: return None
        
        try:
            img_data = base64.b64decode(match['image_base64'])
            img = Image.open(BytesIO(img_data)).convert('RGBA').resize(size, Image.Resampling.LANCZOS)
            mask = Image.new('L', size, 0)
            ImageDraw.Draw(mask).ellipse([0, 0, size[0], size[1]], fill=255)
            output = Image.new('RGBA', size, (0,0,0,0))
            output.paste(img, (0,0))
            output.putalpha(mask)
            return np.array(output) / 255.0
        except: return None

    def load_background(self):
        return plt.imread("assets/fondo_informes.png") if os.path.exists("assets/fondo_informes.png") else None

    def load_team_logo(self, team_name, target_size=(60, 80)):
        try:
            if self.team_stats is not None:
                row = self.team_stats[self.team_stats['Team Name'] == team_name]
                if not row.empty and (b64 := row.iloc[0].get('Team Logo')):
                    img = Image.open(BytesIO(base64.b64decode(b64))).convert('RGBA')
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    return np.array(img) / 255.0
        except: pass

        folder = "assets/escudos/"
        if not os.path.exists(folder): return None
        
        def normalize(s):
            return unicodedata.normalize('NFKD', str(s).lower()).encode('ascii', 'ignore').decode('utf-8')

        target = normalize(team_name)
        best_file = None
        best_score = 0.0

        for f in os.listdir(folder):
            if not f.endswith(('.png', '.jpg', '.jpeg')): continue
            
            f_name_raw = os.path.splitext(f)[0]
            f_name = normalize(f_name_raw)

            score = SequenceMatcher(None, target, f_name).ratio()
            
            if f_name in target:
                score += 0.3
            
            if "real" in f_name and "real" not in target:
                score -= 0.2

            if score > best_score:
                best_score = score
                best_file = f

        if best_file and best_score > 0.4:
            return plt.imread(folder + best_file)
            
        return None

    def construir_historial_lanzadores(self):
        try:
            eventos_unicos = self.df_tracking.drop_duplicates(subset=['ID_Evento_Corner'])
            for _, row in eventos_unicos.iterrows():
                jugador = str(row.get('Nombre_Lanzador', 'Desconocido')).strip()
                tipo_raw = str(row.get('Tipo_Lanzamiento', ''))
                y_balon = row.get('Y_Balon', 50)
                if jugador in ["Desconocido", "None", "nan"]: continue
                lado = 'izquierda' if y_balon > 50 else 'derecha'
                tipo_real = None
                if 'In-swinger' in tipo_raw: tipo_real = 'Cerrado'
                elif 'Out-swinger' in tipo_raw: tipo_real = 'Abierto'
                
                if tipo_real:
                    if jugador not in self.historial_lanzadores:
                        self.historial_lanzadores[jugador] = {'izquierda': [], 'derecha': []}
                    self.historial_lanzadores[jugador][lado].append(tipo_real)
        except: pass

    def inferir_tipo_lanzamiento(self, jugador, lado):
        jugador = str(jugador).strip()
        if jugador in self.historial_lanzadores:
            historial = self.historial_lanzadores[jugador].get(lado, [])
            if historial: return Counter(historial).most_common(1)[0][0]
        return "Neutro"

    def clasificar_zona_remate(self, x, y, lado_origen):
        """Clasifica la zona final según las nuevas coordenadas personalizadas."""
        es_izq = (lado_origen == 'izquierda') 
        
        # 1. ZONA ATRÁS / VIGILANCIA
        if x < 70: 
            return "VIGILANCIA"
        
        # 2. A LA CORTA (Prioridad alta si está muy pegado a banda en zona final)
        if 75 <= x <= 100:
            if (es_izq and y >= 81) or (not es_izq and y <= 25): 
                return "CORTO"
        
        # 3. ÁREA PEQUEÑA (x > 94.2)
        if x > 94.2:
            if 45 <= y <= 55: return "AP CENTRO"
            elif (es_izq and 55 < y <= 70) or (not es_izq and 30 <= y < 45): return "AP 1ER PALO"
            elif (es_izq and 30 <= y < 45) or (not es_izq and 55 < y <= 70): return "AP 2DO PALO"
            else: return "AP OTRA"
        
        # 4. ÁREA GRANDE / PENALTI (83 <= x <= 94.2)
        elif 83 <= x <= 94.2:
            if 45 <= y <= 55: 
                return "CENTRO AREA" if x >= 88 else "PENALTI"
            elif (es_izq and 55 < y <= 70) or (not es_izq and 30 <= y < 45): return "1ER PALO"
            elif (es_izq and 30 <= y < 45) or (not es_izq and 55 < y <= 70): return "2DO PALO"
            else: return "AREA GRANDE"
        
        # 5. RECHACE (70 <= x < 83)
        elif 70 <= x < 83:
            if 36.8 <= y <= 63.2: 
                return "RECHACE CEN"
            else: 
                if es_izq:
                    if 63.2 < y <= 78.9: return "RECHACE CER"
                    elif 21.1 <= y < 36.8: return "RECHACE LEJ"
                else: # Derecha
                    if 21.1 <= y < 36.8: return "RECHACE CER"
                    elif 63.2 < y <= 78.9: return "RECHACE LEJ"
                return "RECHACE"

        return "OTRO"
    
    def determinar_rol(self, x_jug_remate, y_jug_remate, x_balon_remate, y_balon_remate, es_centro_valido):
        """
        Define si es rematador basándose en la proximidad al balón 
        en el momento exacto de llegada (sincronizado con Opta).
        """
        # Si el centro se fue muy pasado o muy corto (fuera de la zona peligrosa Y),
        # NO contabilizamos rematadores para el porcentaje.
        if not es_centro_valido:
            return "OTRO"

        distancia = np.hypot(x_jug_remate - x_balon_remate, y_jug_remate - y_balon_remate)
        
        # Umbral: Si está a menos de 3.5 metros del balón en el momento clave
        if distancia < 5.0:
            return "REMATADOR"
            
        return "OTRO"

    def analizar_disposicion_ofensiva(self, frame_inicio, frame_final, equipo_atacante, lado, 
                                      ball_x_remate, ball_y_remate, es_centro_valido, lanzador_real=None):
        
        atacantes_ini = frame_inicio[frame_inicio['NombreEquipoJugador_Tracking'] == equipo_atacante].copy()
        
        # Mapa de posiciones finales
        mapa_pos_final = {}
        atacantes_fin = frame_final[frame_final['NombreEquipoJugador_Tracking'] == equipo_atacante]
        for _, r in atacantes_fin.iterrows():
            mapa_pos_final[r['Nombre_Jugador_Tracking']] = (r['X_Jugador'], r['Y_Jugador'])

        # --- CONTADORES DETALLADOS POR ZONA TÁCTICA ---
        # Estructura: [Vigilancia, Corto, 1erPalo, Centro/Penalti, 2doPalo, Rechace]
        cnt_vig = 0
        cnt_corto = 0
        cnt_1er = 0
        cnt_cen = 0
        cnt_2do = 0
        cnt_rech = 0
        
        zonas_asignadas = []
        roles_asignados = [] 
        
        es_izq = (lado == 'izquierda') 

        for _, row in atacantes_ini.iterrows():
            x_ini, y_ini = row.get('X_Jugador', 0.0), row.get('Y_Jugador', 0.0)
            nombre = str(row.get('Nombre_Jugador_Tracking', "Desconocido"))
            
            zona = "SIN CLASIFICAR"
            rol = "OTRO"

            # Detectar si es el lanzador (por nombre o posición)
            es_lanzador_pos = (x_ini > 96 and ((es_izq and y_ini > 96) or (not es_izq and y_ini < 4)))
            es_lanzador_nom = (lanzador_real and str(lanzador_real).lower() in nombre.lower())
            
            if es_lanzador_pos or es_lanzador_nom:
                zona = "LANZADOR"
                rol = "LANZADOR"
            else:
                zona = self.clasificar_zona_remate(x_ini, y_ini, lado)
                
                # Clasificación Táctica para el Patrón
                if "VIGILANCIA" in zona: cnt_vig += 1
                elif "CORTO" in zona: cnt_corto += 1
                elif "1ER PALO" in zona: cnt_1er += 1  # Incluye AP 1ER PALO y 1ER PALO
                elif "CENTRO" in zona or "PENALTI" in zona: cnt_cen += 1 # Incluye AP CENTRO, PENALTI, CENTRO AREA
                elif "2DO PALO" in zona: cnt_2do += 1 # Incluye AP 2DO PALO y 2DO PALO
                elif "RECHACE" in zona: cnt_rech += 1
                else: 
                    # Si cae en "OTRO" o "AP OTRA", lo asignamos por cercanía o defecto a Centro
                    cnt_cen += 1

                # Rol Rematador
                if nombre in mapa_pos_final:
                    x_fin, y_fin = mapa_pos_final[nombre]
                    rol = self.determinar_rol(x_fin, y_fin, ball_x_remate, ball_y_remate, es_centro_valido)
            
            zonas_asignadas.append(zona)
            roles_asignados.append(rol)

        atacantes_ini['Zona_Detallada'] = zonas_asignadas
        atacantes_ini['Rol_Calculado'] = roles_asignados 
        
        # DEVOLVEMOS LA TUPLA ESTRUCTURAL: (Vig, Corto, 1er, Cen, 2do, Rech)
        stats_estructurales = (cnt_vig, cnt_corto, cnt_1er, cnt_cen, cnt_2do, cnt_rech)
        
        return stats_estructurales, atacantes_ini

    def extract_data(self, side_req, type_req):
        if self.df_tracking is None: return []
        datos_contextuales = []
        ids_eventos = self.df_tracking[['ID_Evento_Corner', 'ID_Partido']].drop_duplicates().values
        target_team = str(self.team_filter).strip().lower()
        
        def safe_float(val, default):
            try: return float(val)
            except (ValueError, TypeError): return default

        for id_evento, id_partido in ids_eventos:
            df_corner = self.df_tracking[(self.df_tracking['ID_Evento_Corner'] == id_evento) & (self.df_tracking['ID_Partido'] == id_partido)].copy()
            if df_corner.empty: continue
            
            y_balon = df_corner.head(10)['Y_Balon'].mean()
            lado = 'izquierda' if y_balon > 50 else 'derecha'
            if lado != side_req: continue

            idx_kick, time_kick = self.detectar_inicio_saque(df_corner, lado)
            if idx_kick is None: continue 
            
            row = df_corner.iloc[idx_kick]
            equipo_lanz = str(row.get('Equipo_Lanzador', '')).strip()
            if target_team not in equipo_lanz.lower() and equipo_lanz.lower() not in target_team: continue
            
            lanzador = str(row.get('Nombre_Lanzador', 'Desconocido'))
            tipo_raw = str(row.get('Tipo_Lanzamiento', ''))
            perfil = 'Cerrado' if 'In-swinger' in tipo_raw else ('Abierto' if 'Out-swinger' in tipo_raw else self.inferir_tipo_lanzamiento(lanzador, lado))
            if type_req not in perfil: continue

            jornada = row.get('Jornada', 0)
            mn = row.get('Minuto_Corner', 0)
            sc = row.get('Segundo_Corner', 0)
            
            # Obtener nombre normalizado del lanzador para el patrón
            lanzador_name = self.get_opta_name(equipo_lanz, row.get('Dorsal_Lanzador','?'), lanzador)

            opta_row = self.get_opta_event_raw(jornada, equipo_lanz, mn, sc)
            
            target_x, target_y = 100, 50 
            es_centro_valido = False     
            trayectoria_opta = None
            xg_secuencia = 0.0
            
            time_remate = time_kick + 1.5 
            ball_x_remate, ball_y_remate = 100, 50

            if opta_row is not None:
                try:
                    target_x = safe_float(opta_row.get('Pass End X'), 100.0)
                    target_y = safe_float(opta_row.get('Pass End Y'), 50.0)
                    start_x = safe_float(opta_row.get('x'), 100.0)
                    start_y = safe_float(opta_row.get('y'), 50.0)
                    
                    trayectoria_opta = np.column_stack((np.linspace(start_x, target_x, 5), np.linspace(start_y, target_y, 5)))
                    
                    # Rango ampliado para validez (15-85)
                    if 15 <= target_y <= 85:
                        es_centro_valido = True
                    
                    t_best, bx_best, by_best = self.detectar_momento_remate(df_corner, target_x, target_y, time_kick)
                    if t_best is not None:
                        time_remate = t_best
                        ball_x_remate = bx_best
                        ball_y_remate = by_best
                        
                except Exception as e:
                    print(f"Error procesando Opta: {e}")

            trayectorias_jug = self.extract_player_trajectories(df_corner, time_kick, equipo_lanz)
            
            frame_kick = df_corner[df_corner['Segundos_Desde_Saque'] == time_kick].drop_duplicates(subset=['Nombre_Jugador_Tracking'])
            if frame_kick.empty: frame_kick = df_corner.iloc[[idx_kick]]
            frame_kick = frame_kick.copy()
            frame_kick['Dorsal'] = frame_kick.get('Dorsal_Jugador_Tracking', "?")
            
            frame_remate_all = df_corner[
                (df_corner['Segundos_Desde_Saque'] >= time_remate - 0.04) & 
                (df_corner['Segundos_Desde_Saque'] <= time_remate + 0.04)
            ].drop_duplicates(subset=['Nombre_Jugador_Tracking'])
            
            if frame_remate_all.empty:
                frame_remate_all = df_corner.iloc[(df_corner['Segundos_Desde_Saque'] - time_remate).abs().argsort()[:25]]

            # Obtenemos la estructura táctica (Vig, Corto, 1er, Cen, 2do, Rech)
            stats_struct, df_att = self.analizar_disposicion_ofensiva(
                frame_kick, frame_remate_all, equipo_lanz, lado, 
                ball_x_remate, ball_y_remate, es_centro_valido, lanzador_real=lanzador
            )
            
            df_att['Nombre_Jugador_Tracking'] = df_att.apply(lambda r: self.get_opta_name(r['NombreEquipoJugador_Tracking'], r['Dorsal'], r['Nombre_Jugador_Tracking']), axis=1)
            
            # --- CLAVE DEL PATRÓN: (LANZADOR, VIG, CORTO, 1ER, CEN, 2DO, RECH) ---
            clave_patron = (lanzador_name,) + stats_struct

            datos_contextuales.append({
                'ataque_df': df_att, 
                'ataque_stats_generales': clave_patron, # Clave nueva
                'lado': lado,
                'lanzador_nombre': lanzador_name,
                'lanzador_dorsal': row.get('Dorsal_Lanzador','?'),
                'trayectoria': trayectoria_opta, 
                'player_trajectories': trayectorias_jug,
                'id': id_evento,
                'es_centro_valido': es_centro_valido,
                'xg_generado': xg_secuencia
            })
            
        return datos_contextuales

    def get_scenarios(self, data_list):
        grupos = {} 
        for i, dato in enumerate(data_list):
            stats = tuple(dato['ataque_stats_generales']) 
            if stats not in grupos: grupos[stats] = []
            grupos[stats].append(i)
        
        lista = []
        for stats, indices in grupos.items():
            lista.append({'stats': stats, 'indices': indices, 'count': len(indices)})
        lista.sort(key=lambda x: x['count'], reverse=True)
        return lista

    def _ajustar_solapamientos(self, df, x_col, y_col, radio_minimo=3.0):
        coords = df[[x_col, y_col]].values.copy()
        n = len(coords)
        for _ in range(5): 
            for i in range(n):
                for j in range(i + 1, n):
                    x1, y1 = coords[i]
                    x2, y2 = coords[j]
                    dist = np.hypot(x1 - x2, y1 - y2)
                    if dist < radio_minimo:
                        if dist == 0: dx, dy = np.random.uniform(-1, 1), np.random.uniform(-1, 1); dist = 0.1
                        else: dx, dy = (x1 - x2) / dist, (y1 - y2) / dist
                        overlap = (radio_minimo - dist) / 2
                        coords[i][0] += dx * overlap; coords[i][1] += dy * overlap
                        coords[j][0] -= dx * overlap; coords[j][1] -= dy * overlap
                coords[i][0] = max(2, min(98, coords[i][0]))
                coords[i][1] = max(2, min(98, coords[i][1]))
        df[x_col] = coords[:, 0]
        df[y_col] = coords[:, 1]
        return df

    def create_reporte_grid(self):
        if not self.data_left and not self.data_right:
            print("❌ No hay datos suficientes.")
            return None
            
        # 1. Carga y FILTRADO de fotos por equipo
        all_photos = self.load_player_photos()
        team_photos = self.filter_photos_by_team(all_photos)
        
        scenarios_left = self.get_scenarios(self.data_left)
        scenarios_right = self.get_scenarios(self.data_right)

        # 2. Layout
        fig = plt.figure(figsize=(20, 12), facecolor='white')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.1, hspace=0.2)
        
        gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.3], width_ratios=[1, 1, 1.2, 1, 1])
        
        # Fondo y Escudo
        if (bg := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(bg, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25)
            ax_bg.axis('off')
        
        if self.team_filter and (logo := self.load_team_logo(self.team_filter)) is not None:
            ax_team = fig.add_axes([0.92, 0.93, 0.05, 0.07], zorder=1)
            ax_team.imshow(logo, aspect='auto'); ax_team.axis('off')
            
        fig.suptitle(f"ANÁLISIS CÓRNERS OFENSIVOS CERRADOS\n(IZQUIERDA vs DERECHA)", 
                     fontsize=17, fontweight='bold', color='#E74C3C', y=0.97)

        img_balon = plt.imread("assets/balon.png") if os.path.exists("assets/balon.png") else None

        # ==========================================
        # 1. FUNCIÓN INTERNA DE PATRONES
        # ==========================================
        def draw_pattern(ax, scenarios, idx, data_source, side, img_balon):
            if idx >= len(scenarios):
                ax.axis('off'); return
            
            esc = scenarios[idx]
            indices = esc['indices']
            data_cluster = [data_source[i] for i in indices]
            n_corners = len(indices)
            
            # --- PITCH ---
            pitch = VerticalPitch(pitch_type='opta', half=True, pitch_color='#2d5016', line_color='white')
            pitch.draw(ax=ax)
            
            ax.set_aspect('auto')
            ax.set_ylim(60, 105)
            ax.set_xlim(105, -5) 
            
            # Coordenadas Esquina
            if side == 'izquierda':
                xb, yb, xl, yl = 100, 100, 104, 102
            else:
                xb, yb, xl, yl = 100, 0, 104, -2

            # --- BALÓN (ESQUINA) ---
            if img_balon is not None:
                ab = AnnotationBbox(OffsetImage(img_balon, zoom=0.02), (yb, xb), frameon=False, zorder=20)
                ax.add_artist(ab)
            else:
                pitch.scatter(xb, yb, ax=ax, s=60, c='white', edgecolors='black', zorder=20)
            
            # --- LANZADOR ---
            stats_key = esc['stats']
            nombre_lanz_patron = stats_key[0]
            cnt_vig_k, cnt_corto_k, cnt_1er_k, cnt_cen_k, cnt_2do_k, cnt_rech_k = stats_key[1:]
            
            dorsal_lanzador = "?"
            for d in data_cluster:
                if d['lanzador_nombre'] == nombre_lanz_patron:
                    dorsal_lanzador = self.clean_dorsal_str(d['lanzador_dorsal'])
                    break
            
            # --- CONTROL DE DORSALES ÚNICOS ---
            assigned_dorsals = set()
            if dorsal_lanzador not in ["?", "nan"]:
                assigned_dorsals.add(dorsal_lanzador)

            pitch.scatter(xl, yl, ax=ax, s=150, c='#FFD700', marker='s', edgecolors='black', clip_on=False, zorder=20)
            pitch.annotate(dorsal_lanzador, (xl, yl), ax=ax, fontsize=6, ha='center', va='center', fontweight='bold', color='black', zorder=21)
            
            # --- ZONA DE CAÍDA ---
            traj_list = [d['trayectoria'] for d in data_cluster if d['trayectoria'] is not None and len(d['trayectoria']) > 0]
            
            if traj_list:
                destinos = np.array([t[-1] for t in traj_list])
                drawn_kde = False
                if len(destinos) >= 3:
                    try:
                        pitch.kdeplot(x=destinos[:, 0], y=destinos[:, 1], ax=ax, fill=True, levels=100, thresh=0.05, cmap='YlOrRd', alpha=0.6, zorder=3)
                        drawn_kde = True
                    except: drawn_kde = False
                if not drawn_kde:
                    pitch.scatter(destinos[:, 0], destinos[:, 1], ax=ax, s=350, color='#FFD700', alpha=0.7, edgecolors='none', zorder=3)

            # --- JUGADORES (CLUSTERING) - ASIGNACIÓN ÚNICA OPTIMIZADA ---
            try:
                df_all = pd.concat([d['ataque_df'] for d in data_cluster])
                df_all['Dorsal'] = df_all['Dorsal'].apply(self.clean_dorsal_str)
                
                # Definición de zonas y prioridades
                zonas_tacticas = [
                    (cnt_vig_k, ["VIGILANCIA"]),
                    (cnt_corto_k, ["CORTO"]),
                    (cnt_1er_k, ["1ER PALO", "AP 1ER PALO"]),
                    (cnt_cen_k, ["CENTRO", "PENALTI", "AP CENTRO", "AP OTRA", "OTRO", "SIN CLASIFICAR"]),
                    (cnt_2do_k, ["2DO PALO", "AP 2DO PALO"]),
                    (cnt_rech_k, ["RECHACE", "RECHACE CEN", "RECHACE CER", "RECHACE LEJ"])
                ]

                # Estructuras para gestionar la asignación
                zones_objects = {}  # Mapa {zone_id: {datos_zona}}
                global_candidates = [] # Lista [(count, zone_id, dorsal)]
                zone_id_counter = 0

                # 1. CLUSTERING Y GENERACIÓN DE CANDIDATURAS
                for count_required, zona_labels in zonas_tacticas:
                    if count_required <= 0: continue
                    
                    mask_zone = df_all['Zona_Detallada'].apply(lambda z: any(lbl in z for lbl in zona_labels))
                    df_zone = df_all[mask_zone].copy()
                    if df_zone.empty: continue

                    n_puntos = len(df_zone[['X_Jugador', 'Y_Jugador']])
                    k_zone = min(count_required, n_puntos) 
                    if k_zone < 1: continue

                    kmeans = KMeans(n_clusters=k_zone, random_state=42, n_init=10).fit(df_zone[['X_Jugador', 'Y_Jugador']].values)
                    df_zone['Cluster_Zone'] = kmeans.labels_

                    for c_id in range(k_zone):
                        subset = df_zone[df_zone['Cluster_Zone'] == c_id]
                        center = kmeans.cluster_centers_[c_id]
                        counts = subset['Dorsal'].value_counts()
                        subset_players = subset['Nombre_Jugador_Tracking'].unique()
                        
                        # Guardamos objeto zona
                        z_id = zone_id_counter
                        zones_objects[z_id] = {
                            'center': center,
                            'counts': counts,
                            'subset_players': subset_players,
                            'assigned_dorsal': None
                        }
                        
                        # Generamos candidatos para la Fase 1
                        for dorsal, count in counts.items():
                            d_str = str(dorsal).replace(".0", "")
                            if d_str not in ["?", "nan", "None"]:
                                global_candidates.append({'count': count, 'z_id': z_id, 'dorsal': d_str})
                        
                        zone_id_counter += 1

                # 2. FASE 1: ASIGNACIÓN POR MÁXIMA REPETICIÓN GLOBAL
                # Ordenamos candidatos: El que tenga más repeticiones en todo el gráfico va primero.
                global_candidates.sort(key=lambda x: x['count'], reverse=True)
                
                assigned_zones_ids = set()
                # 'assigned_dorsals' ya contiene al lanzador (del código previo)
                
                for cand in global_candidates:
                    z_id = cand['z_id']
                    dorsal = cand['dorsal']
                    
                    # Si la zona ya tiene dueño o el jugador ya está usado, saltamos
                    if z_id in assigned_zones_ids: continue
                    if dorsal in assigned_dorsals: continue
                    
                    # Asignamos
                    zones_objects[z_id]['assigned_dorsal'] = dorsal
                    assigned_zones_ids.add(z_id)
                    assigned_dorsals.add(dorsal)

                # 3. FASE 2: RELLENO DE HUECOS (EL "SEGUNDO MEJOR")
                # Revisamos zonas que se quedaron sin asignar porque su "titular" estaba ocupado
                for z_id, zone_data in zones_objects.items():
                    if zone_data['assigned_dorsal'] is None:
                        # Buscamos en SU lista local el siguiente mejor que esté libre
                        found_unique = False
                        for d, _ in zone_data['counts'].items():
                            d_str = str(d).replace(".0", "")
                            if d_str not in ["?", "nan", "None"] and d_str not in assigned_dorsals:
                                zone_data['assigned_dorsal'] = d_str
                                assigned_dorsals.add(d_str)
                                found_unique = True
                                break
                        
                        # 4. FASE 3: FALLBACK (Solo si no hay nadie más, repetimos para evitar '?')
                        if not found_unique and not zone_data['counts'].empty:
                             top_d = str(zone_data['counts'].index[0]).replace(".0", "")
                             if top_d not in ["?", "nan", "None"]:
                                 zone_data['assigned_dorsal'] = top_d

                # 5. DIBUJAR FINALMENTE
                for z_id, zone_data in zones_objects.items():
                    dorsal_final = zone_data['assigned_dorsal'] if zone_data['assigned_dorsal'] else "?"
                    start_x, start_y = zone_data['center']
                    subset_players = zone_data['subset_players']

                    # --- Visualización Trayectorias (Igual que antes) ---
                    paths_in_cluster = []
                    final_x, final_y = start_x, start_y

                    for d in data_cluster:
                        p_trajs = d['player_trajectories']
                        for p_name in subset_players:
                            if p_name in p_trajs and len(p_trajs[p_name]) > 0:
                                paths_in_cluster.append(p_trajs[p_name])
                    
                    if paths_in_cluster:
                        resampled_paths = []
                        for p in paths_in_cluster:
                            if len(p) < 2: continue
                            indices = np.linspace(0, len(p)-1, 20).astype(int)
                            resampled_paths.append(p[indices])
                        if resampled_paths:
                            mean_path = np.mean(resampled_paths, axis=0)
                            shift_x, shift_y = start_x - mean_path[0][0], start_y - mean_path[0][1]
                            mean_path[:, 0] += shift_x
                            mean_path[:, 1] += shift_y
                            final_x, final_y = mean_path[-1]
                            pitch.plot(mean_path[:, 0], mean_path[:, 1], ax=ax, color='#00FFFF', ls='--', lw=1.2, alpha=0.8, zorder=14)

                    # --- Puntos y Dorsal ---
                    pitch.scatter(start_x, start_y, ax=ax, s=50, color='#E74C3C', edgecolors='none', zorder=15)
                    pitch.annotate(dorsal_final, (start_x, start_y), ax=ax, ha='center', va='center', fontsize=5, fontweight='bold', color='white', zorder=16)
                    
                    if np.hypot(final_x - start_x, final_y - start_y) > 2.0:
                        pitch.annotate(dorsal_final, (final_x, final_y), ax=ax, ha='center', va='center', fontsize=5, fontweight='bold', color='#00FFFF', zorder=16, bbox=dict(boxstyle="square,pad=0.1", fc="#2d5016", ec="none", alpha=0.7))

            except Exception as e: print(f"Error drawing pattern: {e}")
            
            try:
                struct_str_parts = []
                if cnt_corto_k > 0: struct_str_parts.append(f"{cnt_corto_k}Co")
                if cnt_1er_k > 0:  struct_str_parts.append(f"{cnt_1er_k}P1")
                if cnt_cen_k > 0: struct_str_parts.append(f"{cnt_cen_k}Ce")
                if cnt_2do_k > 0:  struct_str_parts.append(f"{cnt_2do_k}P2")
                if cnt_rech_k > 0: struct_str_parts.append(f"{cnt_rech_k}Re")
                if not struct_str_parts: struct_str_parts = ["Disp. Libre"]
                struct_str = "-".join(struct_str_parts)
                lanz_parts = nombre_lanz_patron.split()
                lanz_corto = lanz_parts[-1] if lanz_parts else nombre_lanz_patron[:8]
                if len(lanz_corto) > 9: lanz_corto = lanz_corto[:9] + "."
                full_title = f"{lanz_corto} | {struct_str}\n({n_corners} veces)"
            except: full_title = f"Patrón #{idx+1} ({n_corners} rep.)"
            ax.set_title(full_title, fontsize=8, fontweight='bold', backgroundcolor='#fce4e4')

        # === EJECUCIÓN DE BUCLES PARA PATRONES ===
        for i in range(4):
            row = 0 if i < 2 else 1
            col = 0 if i % 2 == 0 else 1
            ax = fig.add_subplot(gs[row, col])
            draw_pattern(ax, scenarios_left, i, self.data_left, 'izquierda', img_balon)

        for i in range(4):
            row = 0 if i < 2 else 1
            col = 3 if i % 2 == 0 else 4
            ax = fig.add_subplot(gs[row, col])
            draw_pattern(ax, scenarios_right, i, self.data_right, 'derecha', img_balon)

        # ==========================================
        # 2. FUNCIÓN POSICIONAMIENTO MEDIO (CON AJUSTE DE SOLAPAMIENTO)
        # ==========================================
        def draw_average_pos(ax, data_list, side):
            pitch = VerticalPitch(pitch_type='opta', half=True, pitch_color='#1e3c0e', line_color='white')
            pitch.draw(ax=ax)
            ax.set_aspect('auto')
            ax.set_xlim(105, -5)
            ax.set_ylim(60, 105)
            
            if side == 'izquierda': xb, yb, xl, yl = 100, 100, 104, 102
            else: xb, yb, xl, yl = 100, 0, 104, -2
            
            pitch.scatter(xb, yb, ax=ax, s=100, c='white', edgecolors='black', zorder=20)
            
            if not data_list: ax.text(50, 80, "SIN DATOS", ha='center', color='white'); return

            # --- DIBUJO DEL LANZADOR (Igual) ---
            lanzadores_total = [self.clean_dorsal_str(d['lanzador_dorsal']) for d in data_list]
            dorsal_top_lanzador = Counter(lanzadores_total).most_common(1)[0][0] if lanzadores_total else "?"
            pitch.scatter(xl, yl, ax=ax, s=250, c='#FFD700', marker='s', edgecolors='black', clip_on=False, zorder=20)
            pitch.annotate(dorsal_top_lanzador, (xl, yl), ax=ax, fontsize=7, ha='center', va='center', fontweight='bold', color='black', zorder=21)

            # --- MAPA DE CALOR (Igual) ---
            traj_list = [d['trayectoria'] for d in data_list if d['trayectoria'] is not None and len(d['trayectoria']) > 0]
            if traj_list:
                destinos = np.array([t[-1] for t in traj_list])
                ex, ey = destinos[:, 0], destinos[:, 1]
                if len(destinos) >= 2:
                    try: pitch.kdeplot(x=ex, y=ey, ax=ax, fill=True, levels=100, thresh=0.05, cmap='YlOrRd', alpha=0.5, zorder=3)
                    except: pitch.scatter(ex, ey, ax=ax, s=400, color='#FFD700', alpha=0.4, edgecolors='none', zorder=3)
                else: pitch.scatter(ex, ey, ax=ax, s=400, color='#FFD700', alpha=0.4, edgecolors='none', zorder=3)

            # --- 1. CALCULAR RANKING PARA OBTENER LOS TOP 5 ---
            rank_data = []
            for d in data_list:
                df_att = d['ataque_df']
                xg_corner = d.get('xg_generado', 0.0)
                es_valido = d.get('es_centro_valido', False)
                rematador_nom = None
                for _, r in df_att.iterrows():
                    if r.get('Rol_Calculado') == 'REMATADOR':
                        rematador_nom = r['Nombre_Jugador_Tracking']
                        break
                for _, r in df_att.iterrows():
                    if 'LANZADOR' in r['Zona_Detallada']: continue
                    nm = r['Nombre_Jugador_Tracking']
                    ds = self.clean_dorsal_str(r['Dorsal'])
                    role = r.get('Rol_Calculado', 'OTRO')
                    xg_val = xg_corner if (nm == rematador_nom) else 0.0
                    rank_data.append({'Dorsal': ds, 'Role': role, 'Count': 1 if es_valido else 0, 'xG': xg_val, 'Nombre': nm})
            
            top_dorsales = []
            if rank_data:
                df_rank = pd.DataFrame(rank_data)
                stats = df_rank.groupby('Dorsal').agg(
                    Total=('Count', 'sum'),
                    Remates=('Role', lambda x: (x == 'REMATADOR').sum()),
                    Total_xG=('xG', 'sum')
                ).reset_index()
                stats = stats[stats['Total'] > 0].copy()
                stats['Altura'] = stats.apply(lambda r: float(self.get_player_height(self.team_filter, r['Dorsal']) or 0), axis=1)
                for col in ['Altura', 'Remates', 'Total_xG', 'Total']:
                    mn, mx = stats[col].min(), stats[col].max()
                    stats[f'{col}_Score'] = (stats[col] - mn) / (mx - mn) if (mx - mn) != 0 else 0.0
                stats['Rating'] = stats['Altura_Score'] + stats['Remates_Score'] + stats['Total_xG_Score'] + stats['Total_Score']
                top_dorsales = stats.sort_values('Rating', ascending=False).head(5)['Dorsal'].tolist()

            # --- 2. RECOLECTAR COORDENADAS CRUDAS ---
            players_to_draw = []
            
            for dorsal in top_dorsales:
                player_data = []
                for d in data_list:
                    sub_df = d['ataque_df']
                    row_p = sub_df[sub_df['Dorsal'].apply(self.clean_dorsal_str) == dorsal]
                    if not row_p.empty:
                        px = row_p.iloc[0]['X_Jugador']
                        py = row_p.iloc[0]['Y_Jugador']
                        p_name_curr = row_p.iloc[0]['Nombre_Jugador_Tracking']
                        p_traj = d['player_trajectories'].get(p_name_curr, None)
                        player_data.append({'x': px, 'y': py, 'traj': p_traj})
                
                if not player_data: continue

                # Cálculo de posición media (K-Means simple para evitar outliers)
                coords = np.array([[p['x'], p['y']] for p in player_data])
                dominant_indices = list(range(len(player_data))) 
                if len(coords) >= 3:
                    try:
                        km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(coords)
                        l = km.labels_
                        if np.sum(l == 0) >= np.sum(l == 1): dominant_indices = [i for i, label in enumerate(l) if label == 0]
                        else: dominant_indices = [i for i, label in enumerate(l) if label == 1]
                    except: pass
                
                selected_data = [player_data[i] for i in dominant_indices]
                mean_x = np.mean([p['x'] for p in selected_data])
                mean_y = np.mean([p['y'] for p in selected_data])
                
                # Preparamos la trayectoria media (sin shift todavía)
                valid_trajs = [p['traj'] for p in selected_data if p['traj'] is not None and len(p['traj']) > 1]
                mean_traj_raw = None
                if valid_trajs:
                    resampled = []
                    for t in valid_trajs:
                        idx = np.linspace(0, len(t)-1, 25).astype(int)
                        resampled.append(t[idx])
                    mean_traj_raw = np.mean(resampled, axis=0)

                players_to_draw.append({
                    'Dorsal': dorsal,
                    'X': mean_x,
                    'Y': mean_y,
                    'Traj_Raw': mean_traj_raw
                })

            # --- 3. AJUSTAR SOLAPAMIENTOS (Evitar que se monten) ---
            if players_to_draw:
                df_draw = pd.DataFrame(players_to_draw)
                # radio_minimo=4.5 asegura espacio para el círculo s=150
                df_draw = self._ajustar_solapamientos(df_draw, 'X', 'Y', radio_minimo=4.5)

                # --- 4. DIBUJAR FINALMENTE ---
                for _, row in df_draw.iterrows():
                    adj_x, adj_y = row['X'], row['Y']
                    dorsal = row['Dorsal']
                    traj_raw = row['Traj_Raw']
                    
                    final_tx, final_ty = adj_x, adj_y

                    # Ajustar y dibujar trayectoria conectada a la nueva posición ajustada
                    if traj_raw is not None:
                        # Shift trayectoria para que empiece en la posición ajustada
                        shift_x = adj_x - traj_raw[0][0]
                        shift_y = adj_y - traj_raw[0][1]
                        
                        traj_plot = traj_raw.copy()
                        traj_plot[:, 0] += shift_x
                        traj_plot[:, 1] += shift_y
                        
                        pitch.plot(traj_plot[:, 0], traj_plot[:, 1], ax=ax, color='#00FFFF', ls='--', lw=2, alpha=0.7, zorder=14)
                        final_tx, final_ty = traj_plot[-1][0], traj_plot[-1][1]

                    # Dibujar punto inicial ajustado
                    pitch.scatter(adj_x, adj_y, ax=ax, s=150, color='#E74C3C', edgecolors='none', zorder=29)
                    pitch.annotate(dorsal, (adj_x, adj_y), ax=ax, ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=31)
                    
                    # Dibujar punto final si se aleja suficiente
                    if np.hypot(final_tx - adj_x, final_ty - adj_y) > 2.5:
                        pitch.annotate(dorsal, (final_tx, final_ty), ax=ax, ha='center', va='center', fontsize=8, fontweight='bold', color='#00FFFF', zorder=16, bbox=dict(boxstyle="square,pad=0.1", fc="#1e3c0e", ec="none", alpha=0.7))

            s_label = "IZQUIERDA" if side == 'izquierda' else "DERECHA"
            ax.set_title(f"POSICIONAMIENTO MEDIO (TOP 5 - {s_label})", fontsize=11, fontweight='bold', color='white', backgroundcolor='#2980B9')

        ax_avg_izq = fig.add_subplot(gs[2, 0:2])
        draw_average_pos(ax_avg_izq, self.data_left, 'izquierda')
        ax_avg_der = fig.add_subplot(gs[2, 3:5])
        draw_average_pos(ax_avg_der, self.data_right, 'derecha')

        # ==========================================
        # 3. COLUMNA CENTRAL (TABLA RANKING + xG)
        # ==========================================
        ax_table = fig.add_subplot(gs[:, 2])
        ax_table.axis('off')
        ax_table.set_xlim(0, 1); ax_table.set_ylim(0, 1)
        
        full_data_rows = []
        combined_data = self.data_left + self.data_right
        
        for d in combined_data:
            df_att = d['ataque_df']
            trajs = d['player_trajectories']
            lado = d['lado']
            es_valido = d.get('es_centro_valido', False)
            xg_total_corner = d.get('xg_generado', 0.0)

            # --- IDENTIFICAR AL REMATADOR PARA ASIGNARLE EL XG ---
            rematador_nombre = None
            for _, row in df_att.iterrows():
                if row.get('Rol_Calculado') == 'REMATADOR':
                    rematador_nombre = row['Nombre_Jugador_Tracking']
                    break

            for _, row in df_att.iterrows():
                nombre = row['Nombre_Jugador_Tracking']
                dorsal = self.clean_dorsal_str(row['Dorsal'])
                zona_inicio = row['Zona_Detallada']
                
                zona_fin = zona_inicio 
                if nombre in trajs and len(trajs[nombre]) > 0:
                    end_x, end_y = trajs[nombre][-1]
                    zona_fin = self.clasificar_zona_remate(end_x, end_y, lado)
                
                rol = row.get('Rol_Calculado', 'OTRO')
                z_in = zona_inicio.replace("RECHACE", "RECH").replace("A.PEQ", "AP").replace("AREA", "AR").replace("VIGILANCIA", "VIG").replace("PENALTI", "PEN")
                z_out = zona_fin.replace("RECHACE", "RECH").replace("A.PEQ", "AP").replace("AREA", "AR").replace("VIGILANCIA", "VIG").replace("PENALTI", "PEN")
                seq = z_in if z_in == z_out else f"{z_in} → {z_out}"
                
                # ASIGNAR XG SOLO AL REMATADOR
                xg_asignado = self.xg_por_jugador.get(nombre, 0.0)

                full_data_rows.append({
                    'Nombre': nombre,
                    'Dorsal': dorsal,
                    'Sequence': seq,
                    'Role': rol,
                    'CountForStats': 1 if es_valido else 0,
                    'xG': xg_asignado # <--- Dato individual para sumar
                })
        
        if full_data_rows:
            df_seq = pd.DataFrame(full_data_rows)
            
            # 1. AGRUPAR (Incluyendo suma de xG)
            ranking = df_seq.groupby(['Nombre', 'Dorsal']).agg(
                Total=('CountForStats', 'sum'),
                Remates=('Role', lambda x: (x == 'REMATADOR').sum()),
                Total_xG=('xG', 'sum') # <--- NUEVA SUMA
            ).reset_index()

            ranking = ranking[ranking['Total'] > 0].copy()

            # Filtro Vigilancia
            raw_counts = df_seq.groupby(['Nombre', 'Dorsal']).size().reset_index(name='Total_Absoluto')
            vig_df = df_seq[df_seq['Sequence'].str.contains("VIG", na=False)]
            vig_counts = vig_df.groupby(['Nombre', 'Dorsal']).size().reset_index(name='Vig_Count')
            ranking = ranking.merge(vig_counts, on=['Nombre', 'Dorsal'], how='left').fillna(0)
            ranking = ranking.merge(raw_counts, on=['Nombre', 'Dorsal'], how='left')
            ranking['Pct_Vigilancia'] = (ranking['Vig_Count'] / ranking['Total_Absoluto']) * 100
            ranking = ranking[ranking['Pct_Vigilancia'] <= 50].copy()
            
            # Métricas
            ranking['Pct_Rematador'] = (ranking['Remates'] / ranking['Total']) * 100
            def get_h_val(r):
                val = self.get_player_height(self.team_filter, r['Dorsal'])
                try: return float(val)
                except: return 0.0
            ranking['Altura_Val'] = ranking.apply(get_h_val, axis=1)
            
            # 5. ORDENAR POR PUNTUACIÓN COMPUESTA (NORMALIZADA)
            # Normalizamos cada métrica entre 0.0 y 1.0 para poder sumarlas equitativamente.
            # Fórmula: (Valor - Mínimo) / (Máximo - Mínimo)
            
            # Definimos las columnas a puntuar
            cols_to_score = ['Altura_Val', 'Pct_Rematador', 'Total_xG', 'Total']
            
            for col in cols_to_score:
                min_val = ranking[col].min()
                max_val = ranking[col].max()
                col_score_name = f'{col}_Score'
                
                if max_val - min_val != 0:
                    # Normalización relativa (el mejor del equipo tendrá 1.0 en ese atributo)
                    ranking[col_score_name] = (ranking[col] - min_val) / (max_val - min_val)
                else:
                    # Si todos son iguales (ej: nadie tiene xG), se quedan con 0
                    ranking[col_score_name] = 0.0

            # Calculamos el RATING FINAL sumando los parciales
            # Si quieres que algún dato valga más (ej: Altura), multiplícalo aquí (ej: ... * 1.5)
            ranking['Total_Rating'] = (
                ranking['Altura_Val_Score'] + 
                ranking['Pct_Rematador_Score'] + 
                ranking['Total_xG_Score'] + 
                ranking['Total_Score']
            )

            # Ordenamos por la suma total (Rating)
            # IMPORTANTE: Mantengo .head(12) porque si pones 15, la tabla se saldrá del gráfico 
            # y se pintará encima de otros elementos (por el tamaño de fila row_height).
            ranking = ranking.sort_values(by='Total_Rating', ascending=False).head(12)

            
            # --- CABECERA ---
            y_pos = 0.97
            ax_table.add_patch(patches.FancyBboxPatch((0.02, y_pos - 0.025), 0.96, 0.045, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor='#1a1a2e', edgecolor='none'))
            ax_table.text(0.5, y_pos, "POSICIONAMIENTO - ROLES - xG", ha='center', fontsize=11, fontweight='bold', color='white', va='center')
            y_pos -= 0.055
            
            # COLUMNAS (Ajustadas para meter xG)
            ax_table.text(0.08, y_pos, "JUGADOR", fontsize=7, fontweight='bold', color='#95a5a6', ha='center')
            ax_table.text(0.40, y_pos, "MOVIMIENTO", fontsize=7, fontweight='bold', color='#95a5a6', ha='center')
            ax_table.text(0.72, y_pos, "xG", fontsize=7, fontweight='bold', color='#95a5a6', ha='center') # <--- NUEVA COLUMNA
            ax_table.text(0.90, y_pos, "ROL", fontsize=7, fontweight='bold', color='#95a5a6', ha='center')
            
            y_pos -= 0.025
            ax_table.plot([0.02, 0.98], [y_pos, y_pos], color='#ecf0f1', lw=1.5, solid_capstyle='round')
            y_pos -= 0.015
            row_height = 0.07
            
            for idx, (_, row) in enumerate(ranking.iterrows()):
                nombre = row['Nombre']
                dorsal = row['Dorsal']
                total_jug = int(row['Total']) 
                player_data = df_seq[(df_seq['Nombre'] == nombre) & (df_seq['Dorsal'] == dorsal)]
                top_seqs = player_data['Sequence'].value_counts(normalize=True).head(2)
                pct_rematador = int(row['Pct_Rematador'])
                xg_val_jug = row['Total_xG']
                
                # 1. PRE-CALCULAR ESPACIO NECESARIO (Contar líneas de texto)
                lines_count = 0
                for seq_name, _ in top_seqs.items():
                    # Usamos la misma lógica que al escribir para contar líneas
                    if len(seq_name) > 12: 
                        n_lines = len(textwrap.wrap(seq_name, width=11))
                    else: 
                        n_lines = 1
                    lines_count += n_lines
                
                # Añadimos un poco de margen si hay 2 secuencias
                if len(top_seqs) > 1: lines_count += 0.5
                
                # Altura base estándar
                base_h = 0.058
                row_base_step = 0.07
                
                # Si hay más de 3 líneas, expandimos hacia abajo
                extra_h = 0.0
                if lines_count > 3:
                    extra_h = (lines_count - 3) * 0.012

                # El centro "visual" de la parte superior se mantiene
                y_center = y_pos - 0.025
                
                # Ajustamos el rectángulo de fondo:
                # - Bajamos la coordenada Y inferior (y_center - 0.028 - extra_h)
                # - Aumentamos la altura total (base_h + extra_h)
                box_y = y_center - 0.028 - extra_h
                box_h = base_h + extra_h

                # --- FONDO DE FILAS (Alternancia) ---
                if idx % 2 == 0:
                    # Filas Pares: Color original
                    ax_table.add_patch(patches.FancyBboxPatch((0.02, box_y), 0.96, box_h, boxstyle="round,pad=0.005,rounding_size=0.01", facecolor='#f8f9fa', edgecolor='none', alpha=0.6))
                else:
                    # Filas Impares: Gris Semitransparente
                    ax_table.add_patch(patches.FancyBboxPatch((0.02, box_y), 0.96, box_h, boxstyle="round,pad=0.005,rounding_size=0.01", facecolor='#bdc3c7', edgecolor='none', alpha=0.4))
                
                # FOTO
                foto = self.create_circular_player_photo(nombre, team_photos, size=(90, 90))
                if foto is not None:
                    ib = OffsetImage(foto, zoom=0.28)
                    ab = AnnotationBbox(ib, (0.06, y_center), frameon=False, xycoords='data', box_alignment=(0.5, 0.5))
                    ax_table.add_artist(ab)
                    ax_table.add_patch(patches.Circle((0.012, y_center + 0.010), 0.020, facecolor='none', edgecolor='none', lw=1, zorder=10))
                    ax_table.text(0.012, y_center + 0.010, dorsal, fontsize=6, fontweight='bold', color='black', ha='center', va='center', zorder=11)
                else:
                    ax_table.add_patch(patches.Circle((0.06, y_center), 0.025, facecolor='#3498db', edgecolor='white', lw=1.5))
                    ax_table.text(0.06, y_center, dorsal, ha='center', va='center', fontweight='bold', fontsize=9, color='white')
                
                # NOMBRE Y STATS
                nom_parts = nombre.split()
                nom_fmt = f"{nom_parts[0][0]}. {nom_parts[-1]}" if len(nom_parts) > 1 else nombre[:12]
                ax_table.text(0.01, y_center + 0.025, nom_fmt, fontsize=7, fontweight='bold', va='center', color='#2c3e50')
                ax_table.text(0.01, y_center - 0.03, f"{total_jug} apps (val)", fontsize=5, color='#7f8c8d', va='center')
                
                # SECUENCIAS
                seq_y = y_center + 0.01
                blue_color = '#2980b9' 
                for i, (seq_name, pct) in enumerate(top_seqs.items()):
                    pct_val = int(pct * 100)
                    
                    if len(seq_name) > 12: 
                        lines = textwrap.wrap(seq_name, width=11)
                    else: 
                        lines = [seq_name]

                    if i == 0:
                        for j, line in enumerate(lines):
                            prefix = "● " if j == 0 else "  "
                            ax_table.text(0.18, seq_y - (j * 0.012), f"{prefix}{line}", fontsize=7, fontweight='bold', va='center', color=blue_color)
                        
                        ax_table.text(0.55, seq_y, f"{pct_val}%", fontsize=7, fontweight='bold', va='center', color=blue_color)
                        seq_y -= len(lines) * 0.012 + 0.006
                    else:
                        for j, line in enumerate(lines):
                            prefix = "○ " if j == 0 else "  "
                            ax_table.text(0.18, seq_y - (j * 0.010), f"{prefix}{line}", fontsize=5, va='center', color=blue_color)
                        
                        ax_table.text(0.55, seq_y, f"{pct_val}%", fontsize=5, va='center', color=blue_color)
                        seq_y -= len(lines) * 0.010 + 0.004

                # ALTURA
                altura_val = row.get('Altura_Val', 0)
                if altura_val > 0:
                    altura_str = str(int(altura_val))
                    simbolo = "🧍↕" 
                    # seq_y ya ha bajado lo suficiente gracias al bucle anterior, así que la altura siempre quedará debajo
                    ax_table.text(0.28, seq_y - 0.002, f"{simbolo} {altura_str} cm", fontsize=7, fontweight='bold', color='#34495e', va='center')

                # xG
                xg_color = '#e67e22' if xg_val_jug > 0.1 else '#7f8c8d'
                font_w_xg = 'bold' if xg_val_jug > 0.1 else 'normal'
                ax_table.text(0.72, y_center, f"{xg_val_jug:.2f}", fontsize=8, fontweight=font_w_xg, color=xg_color, ha='center', va='center')

                # ROL
                if pct_rematador >= 15:
                    if pct_rematador >= 70: color_bg = '#27ae60' 
                    elif pct_rematador >= 40: color_bg = '#2ecc71' 
                    else: color_bg = '#82e0aa' 
                    ax_table.add_patch(patches.FancyBboxPatch((0.82, y_center - 0.018), 0.16, 0.036, boxstyle="round,pad=0.01,rounding_size=0.015", facecolor=color_bg, edgecolor='none'))
                    ax_table.text(0.90, y_center, f"🔥 {pct_rematador}%", fontsize=8, fontweight='bold', ha='center', va='center', color='white')
                
                # SALTO A LA SIGUIENTE FILA (Dinámico)
                y_pos -= (row_base_step + extra_h)


        return fig

    def guardar(self, fig, filename):
        fig.set_size_inches(14, 9)
        fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
        print(f"✅ Guardado: {filename}")

def seleccionar_equipo():
    try:
        df = pd.read_parquet("extraccion_sportian/corners_tracking.parquet")
        # ... (código de carga de columnas igual que antes) ...
        if 'Equipo_Lanzador' in df.columns:
            equipos = sorted([str(e) for e in df['Equipo_Lanzador'].dropna().unique()])
        else:
            equipos = sorted([str(e) for e in df['NombreEquipoJugador_Tracking'].dropna().unique()])
            
        print("\n=== SELECCIÓN DE EQUIPO ===")
        for i, e in enumerate(equipos, 1): print(f"{i}. {e}")
        
        # --- CAMBIO AQUÍ ---
        sel = input("Número o Nombre exacto: ").strip() # .strip() quita espacios extra
        
        # 1. Si escribe el nombre tal cual (ej: "Real Madrid")
        if sel in equipos:
            return sel
            
        # 2. Si escribe el número (lógica antigua)
        if sel.isdigit() and 0 < int(sel) <= len(equipos):
            return equipos[int(sel)-1]
            
        return None
        # -------------------
    except: return None

if __name__ == "__main__":
    equipo = seleccionar_equipo()
    if equipo:
        rep = ReporteOfensivoCornersBilateral(team_filter=equipo)
        fig = rep.create_reporte_grid()
        if fig:
            rep.guardar(fig, f"reporte_bilateral_cerrado_{equipo.replace(' ','_')}.pdf")
            plt.show()