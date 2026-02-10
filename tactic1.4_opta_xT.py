import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os
import re
import base64
from io import BytesIO
from PIL import Image
from difflib import SequenceMatcher
import warnings
import json

warnings.filterwarnings('ignore')

class XTFlowReport:
    def __init__(self, 
                 match_events_path="extraccion_opta/datos_opta_parquet/match_events.parquet",
                 player_stats_path="extraccion_opta/datos_opta_parquet/player_stats.parquet"):
        
        self.path = match_events_path
        self.player_stats_path = player_stats_path
        
        self.df = None
        self.player_stats = None
        self.photos_data = []
        
        # Matriz xT estándar
        self.xt_grid = np.array([
            [0.00638, 0.0077, 0.0084, 0.0098, 0.011, 0.012, 0.014, 0.017, 0.021, 0.028, 0.040, 0.032],
            [0.00750, 0.0087, 0.0097, 0.0115, 0.013, 0.015, 0.017, 0.021, 0.027, 0.038, 0.063, 0.053],
            [0.00875, 0.0099, 0.0113, 0.0133, 0.015, 0.018, 0.021, 0.026, 0.036, 0.053, 0.108, 0.180],
            [0.00955, 0.0108, 0.0124, 0.0148, 0.017, 0.021, 0.026, 0.034, 0.049, 0.078, 0.254, 0.350],
            [0.00955, 0.0108, 0.0124, 0.0148, 0.017, 0.021, 0.026, 0.034, 0.049, 0.078, 0.254, 0.350],
            [0.00875, 0.0099, 0.0113, 0.0133, 0.015, 0.018, 0.021, 0.026, 0.036, 0.053, 0.108, 0.180],
            [0.00750, 0.0087, 0.0097, 0.0115, 0.013, 0.015, 0.017, 0.021, 0.027, 0.038, 0.063, 0.053],
            [0.00638, 0.0077, 0.0084, 0.0098, 0.011, 0.012, 0.014, 0.017, 0.021, 0.028, 0.040, 0.032]
        ])
        
        self.load_data()

    def load_data(self):
        # 1. Cargar Eventos
        if not os.path.exists(self.path): 
            print(f"❌ Error: No se encuentra {self.path}")
            return
            
        cols = ['Match ID', 'Team Name', 'Event Name', 'typeId', 'outcome', 
                'timeMin', 'timeSec', 'periodId', 'timeStamp', 'x', 'y', 
                'Pass End X', 'Pass End Y', 'playerName', 'Week'] 
        try:
            self.df = pd.read_parquet(self.path, columns=cols)
        except:
            self.df = pd.read_parquet(self.path)

        # Corrección numérica
        for col in ['x', 'y', 'Pass End X', 'Pass End Y']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        if 'timeStamp' in self.df.columns:
            self.df['timeStamp'] = pd.to_datetime(self.df['timeStamp'].astype(str).str.replace('Z', ''), errors='coerce')
        
        self.df = self.df.sort_values(['Match ID', 'periodId', 'timeMin', 'timeSec'])
        
        # 2. Cargar Player Stats (para dorsales)
        if os.path.exists(self.player_stats_path):
            try:
                self.player_stats = pd.read_parquet(self.player_stats_path)
            except:
                print("⚠️ Error cargando player_stats.parquet")
        
        # 3. Cargar JSON de Fotos
        json_path = 'assets/jugadores_optimizados.json'
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.photos_data = json.load(f)
            except:
                print("⚠️ Error cargando JSON de fotos.")
        
        self.calculate_xt()

    def calculate_xt(self):
        """Calcula xT para Pases, Regates y Tiros"""
        mask_pass = (self.df['typeId'] == 1) & (self.df['outcome'] == 1)
        mask_dribble = (self.df['typeId'] == 3) & (self.df['outcome'] == 1)
        mask_shot = self.df['typeId'].isin([13, 14, 15, 16])
        
        self.df['xT_val'] = 0.0

        def get_grid_values(x_series, y_series):
            xi = np.clip(np.floor(x_series / (100/12)).astype(int), 0, 11)
            yi = np.clip(np.floor(y_series / (100/8)).astype(int), 0, 7)
            return self.xt_grid[yi, xi]

        # Pases
        passes = self.df[mask_pass]
        if not passes.empty:
            start_val = get_grid_values(passes['x'], passes['y'])
            end_x = passes['Pass End X'].fillna(passes['x'])
            end_y = passes['Pass End Y'].fillna(passes['y'])
            end_val = get_grid_values(end_x, end_y)
            xt_added = np.maximum(0, end_val - start_val)
            self.df.loc[mask_pass, 'xT_val'] = xt_added

        # Regates
        dribbles = self.df[mask_dribble]
        if not dribbles.empty:
            curr_val = get_grid_values(dribbles['x'], dribbles['y'])
            self.df.loc[mask_dribble, 'xT_val'] = curr_val * 0.20
            
        # Tiros
        shots = self.df[mask_shot]
        if not shots.empty:
            shot_val = get_grid_values(shots['x'], shots['y'])
            self.df.loc[mask_shot, 'xT_val'] = shot_val

    # ==========================================
    # LÓGICA DE FOTOS Y NOMBRES (Traída de AB2.2)
    # ==========================================
    
    def get_player_photo(self, player_name):
        """Obtiene la foto procesada desde el JSON"""
        match = self.match_player_name(player_name)
        if not match:
            return None
        
        try:
            img_data = base64.b64decode(match['image_base64'])
            img = Image.open(BytesIO(img_data))
            
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            data = np.array(img)
            height, width = data.shape[:2]
            
            # Flood fill simple para quitar fondo blanco
            def flood_fill(start_points, threshold=235):
                mask = np.zeros((height, width), dtype=bool)
                visited = np.zeros((height, width), dtype=bool)
                stack = list(start_points)
                
                while stack:
                    y, x = stack.pop()
                    if visited[y, x]: continue
                    visited[y, x] = True
                    
                    if (data[y, x, 0] >= threshold and 
                        data[y, x, 1] >= threshold and 
                        data[y, x, 2] >= threshold):
                        mask[y, x] = True
                        if y > 0: stack.append((y-1, x))
                        if y < height-1: stack.append((y+1, x))
                        if x > 0: stack.append((y, x-1))
                        if x < width-1: stack.append((y, x+1))
                return mask

            border_points = [(0,0), (0, width-1), (height-1, 0), (height-1, width-1)]
            mask = flood_fill(border_points)
            data[mask] = [0, 0, 0, 0] # Transparente
            
            return data.astype(np.float32) / 255.0
            
        except Exception as e:
            return None

    def match_player_name(self, player_name):
        """Busca el nombre en el JSON con lógica difusa"""
        if not self.photos_data: return None
        
        def normalize(n):
            n = n.lower().strip()
            replacements = {'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u', 'ñ':'n'}
            for old, new in replacements.items(): n = n.replace(old, new)
            return re.sub(r'[^\w\s]', '', n)

        target = normalize(player_name)
        target_parts = target.split()
        
        best_match = None
        best_score = 0
        
        for entry in self.photos_data:
            p_name = entry.get('player_name', '')
            normalized_p = normalize(p_name)
            p_parts = normalized_p.split()
            
            score = 0
            # 1. Exacto
            if target == normalized_p: score = 1.0
            # 2. Apellido coincidente + Inicial
            elif len(target_parts)>1 and len(p_parts)>1:
                if target_parts[-1] == p_parts[-1] and target_parts[0][0] == p_parts[0][0]:
                    score = 0.9
            # 3. Similitud general
            else:
                score = SequenceMatcher(None, target, normalized_p).ratio()
            
            if score > best_score and score > 0.6:
                best_score = score
                best_match = entry
                
        return best_match if best_score > 0.6 else None

    def get_player_dorsal(self, player_name):
        """Busca el dorsal en player_stats"""
        if self.player_stats is None: return ""
        # Buscar coincidencia exacta primero
        row = self.player_stats[self.player_stats['Match Name'] == player_name]
        if not row.empty:
            val = row['Shirt Number'].iloc[0]
            return str(int(val)) if pd.notna(val) else ""
        return ""

    # ==========================================
    # LÓGICA DE REPORTE
    # ==========================================

    def get_player_rankings(self, team_name):
        team_events = self.df[self.df['Team Name'] == team_name]
        
        cats = {
            'Pase': team_events[(team_events['typeId'] == 1) & (team_events['outcome'] == 1)],
            'Regate': team_events[(team_events['typeId'] == 3) & (team_events['outcome'] == 1)],
            'Tiro': team_events[team_events['typeId'].isin([13, 14, 15, 16])]
        }
        
        rankings = {}
        for cat_name, df_cat in cats.items():
            if df_cat.empty:
                rankings[cat_name] = pd.DataFrame()
                continue
            
            grouped = df_cat.groupby('playerName')['xT_val'].sum().reset_index()
            matches_per_player = df_cat.groupby('playerName')['Match ID'].nunique()
            
            grouped['matches'] = grouped['playerName'].map(matches_per_player)
            grouped['xt_per_match'] = grouped['xT_val'] / grouped['matches']
            
            # Top 2
            top2 = grouped.sort_values('xT_val', ascending=False).head(2)
            rankings[cat_name] = top2
            
        return rankings

    def find_team_logo(self, equipo):
        if not os.path.exists('assets/escudos'): return None
        files = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
        
        clean = equipo.lower().replace(' ', '').replace('fc','').replace('cf','')
        best, score = None, 0
        
        for f in files:
            f_clean = f.replace('.png','').lower().replace('_','')
            sim = SequenceMatcher(None, clean, f_clean).ratio()
            if sim > score and sim > 0.4:
                score = sim
                best = f
        if best: return plt.imread(f"assets/escudos/{best}")
        return None

    def get_match_context(self, match_id, team_name):
        match_data = self.df[self.df['Match ID'] == match_id]
        if match_data.empty: return "Rival", "Local"
        teams = match_data['Team Name'].unique()
        rival = next((t for t in teams if t != team_name), "Desconocido")
        is_home = (teams[0] == team_name)
        return rival, "Local" if is_home else "Visitante"

    def create_report(self, team_name):
        team_matches_df = self.df[self.df['Team Name'] == team_name]
        if team_matches_df.empty: 
            print(f"❌ No hay datos para: {team_name}")
            return None
        
        matches_sorted = team_matches_df.sort_values('timeStamp')['Match ID'].unique()

        fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
        
        # Fondo
        if os.path.exists("assets/fondo_informes.png"):
            bg_img = plt.imread("assets/fondo_informes.png")
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(bg_img, extent=[0,1,0,1], aspect='auto', alpha=0.25)
            ax_bg.axis('off')

        ball_img = plt.imread("assets/balon.png") if os.path.exists("assets/balon.png") else None

        # HEADER
        ax_header = fig.add_axes([0.05, 0.88, 0.9, 0.08])
        ax_header.axis('off')
        ax_header.text(0.5, 0.7, 'FLUJO DE AMENAZA Y RENDIMIENTO', ha='center', fontsize=22, weight='bold', color='#1e3d59')
        ax_header.text(0.5, 0.2, f"{team_name} | Análisis de Temporada", ha='center', fontsize=12, color='#555')
        
        logo = self.find_team_logo(team_name)
        if logo is not None:
            ab = AnnotationBbox(OffsetImage(logo, zoom=0.40), (0.95, 0.5), frameon=False)
            ax_header.add_artist(ab)

        # DATOS
        last_5_data = []
        all_team_trends, all_rival_trends = [], []
        
        all_goals_team = []  # Goles del equipo (arriba)
        all_goals_rival = [] # Goles del rival (abajo)

        for mid in matches_sorted:
            match_events = self.df[self.df['Match ID'] == mid]
            rival, cond = self.get_match_context(mid, team_name)
            
            my_events = match_events[match_events['Team Name'] == team_name]
            rival_events = match_events[match_events['Team Name'] != team_name]
            
            t_min = my_events.groupby('timeMin')['xT_val'].sum().reindex(range(96), fill_value=0)
            r_min = rival_events.groupby('timeMin')['xT_val'].sum().reindex(range(96), fill_value=0)
            
            t_roll = t_min.rolling(window=5, min_periods=1, center=True).mean()
            r_roll = r_min.rolling(window=5, min_periods=1, center=True).mean()
            
            all_team_trends.append(t_roll)
            all_rival_trends.append(r_roll)
            
            flow = t_roll - r_roll
            
            # Goles Equipo (para gráfica global)
            g_team = my_events[my_events['typeId'] == 16]
            for _, g in g_team.iterrows(): all_goals_team.append(g['timeMin'])
            
            # Goles Rival (para gráfica global)
            g_rival = rival_events[rival_events['typeId'] == 16]
            for _, g in g_rival.iterrows(): all_goals_rival.append(g['timeMin'])

            # --- CAMBIO IMPORTANTE: JUNTAR GOLES DE AMBOS PARA GRÁFICOS INDIVIDUALES ---
            all_match_goals = pd.concat([g_team, g_rival])

            last_5_data.append({'rival': rival, 'cond': cond, 'flow': flow, 'events': all_match_goals})

        # ==========================================
        # 1. GRÁFICO GLOBAL (IZQUIERDA - 60%)
        # ==========================================
        ax_main = fig.add_axes([0.05, 0.42, 0.55, 0.40]) 
        x_axis = range(96)
        
        if all_team_trends:
            avg_team = np.mean(all_team_trends, axis=0)
            avg_rival = np.mean(all_rival_trends, axis=0)
            
            # Pintar
            ax_main.fill_between(x_axis, avg_team, 0, interpolate=True, color='#f39c12', alpha=0.5, label='Generado')
            ax_main.plot(x_axis, avg_team, color='#d35400', lw=2)
            ax_main.fill_between(x_axis, -avg_rival, 0, interpolate=True, color='#34495e', alpha=0.3, label='Concedido')
            ax_main.plot(x_axis, -avg_rival, color='#2c3e50', lw=2)
            ax_main.axhline(0, color='black', lw=1)
            
            max_val = max(np.max(avg_team), np.max(avg_rival)) * 1.3
            if max_val == 0: max_val = 0.1
            ax_main.set_ylim(-max_val, max_val)
            
            # Goles Equipo (Arriba)
            for g_min in all_goals_team:
                if g_min < 96:
                    y_h = avg_team[int(g_min)]
                    if ball_img is not None:
                        imbox = OffsetImage(ball_img, zoom=0.02, alpha=0.9)
                        ab = AnnotationBbox(imbox, (g_min, y_h), frameon=False, pad=0)
                        ax_main.add_artist(ab)
                    else:
                        ax_main.scatter(g_min, y_h, c='gold', s=15, zorder=5)

            # Goles Rival (Abajo - NUEVO)
            for g_min in all_goals_rival:
                if g_min < 96:
                    y_h = -avg_rival[int(g_min)] # En negativo
                    if ball_img is not None:
                        imbox = OffsetImage(ball_img, zoom=0.02, alpha=0.6) # Un poco mas transparente
                        ab = AnnotationBbox(imbox, (g_min, y_h), frameon=False, pad=0)
                        ax_main.add_artist(ab)
                    else:
                        ax_main.scatter(g_min, y_h, c='red', s=15, zorder=5)

        ax_main.set_title("Momentum Medio + Goles (Equipo vs Rival)", fontsize=12, weight='bold', color='#1e3d59')
        ax_main.set_xlim(0, 95)
        ax_main.grid(True, linestyle='--', alpha=0.3)
        ax_main.spines[['top', 'right', 'bottom']].set_visible(False)
        ax_main.set_yticks([])

        # ==========================================
        # 2. RANKING JUGADORES (DERECHA - 40%) - DISEÑO MODERNO
        # ==========================================
        rankings = self.get_player_rankings(team_name)
        start_y_rank = 0.73
        gap_rank = 0.12
        cats_order = ['Pase', 'Regate', 'Tiro']
        
        # Colores para las barras de cada categoría
        cat_colors = {'Pase': '#3498db', 'Regate': '#e67e22', 'Tiro': '#e74c3c'}
        
        for idx, cat in enumerate(cats_order):
            base_y = start_y_rank - (idx * gap_rank)
            
            # Título Categoría con línea decorativa
            fig.text(0.63, base_y + 0.08, f"TOP xT - {cat.upper()}", 
                     fontsize=10, weight='bold', color='#1e3d59')
            # Línea fina debajo del título
            line = plt.Line2D([0.63, 0.93], [base_y + 0.075, base_y + 0.075], 
                             transform=fig.transFigure, color='#1e3d59', linewidth=0.5, alpha=0.5)
            fig.add_artist(line)
            
            if cat in rankings and not rankings[cat].empty:
                top_players = rankings[cat]
                # Obtener el máximo xT de la categoría para escalar las barras
                max_xt_cat = top_players['xT_val'].max() if not top_players.empty else 1.0
                
                for p_idx, (_, player) in enumerate(top_players.iterrows()):
                    # Coordenadas tarjeta jugador
                    p_x = 0.63 + (p_idx * 0.16)
                    p_y = base_y
                    
                    # Crear eje para la tarjeta
                    ax_p = fig.add_axes([p_x, p_y, 0.15, 0.07])
                    # Fondo blanco limpio con borde muy sutil
                    ax_p.set_facecolor('white')
                    for spine in ax_p.spines.values():
                        spine.set_edgecolor('#e0e0e0')
                        spine.set_linewidth(1)
                    ax_p.set_xticks([])
                    ax_p.set_yticks([])
                    
                    # 1. DORSAL (Estilo Marca de Agua Grande a la izquierda)
                    dorsal = self.get_player_dorsal(player['playerName'])
                    if dorsal:
                        ax_p.text(0.05, 0.5, dorsal, transform=ax_p.transAxes, 
                                 fontsize=20, weight='bold', color='#f0f0f0', 
                                 ha='center', va='center', zorder=1, style='italic')
                    
                    # 2. FOTO DEL JUGADOR (Más grande y centrada sobre el dorsal)
                    p_photo = self.get_player_photo(player['playerName'])
                    if p_photo is not None:
                        # ZOOM AUMENTADO (de 0.08 a 0.14)
                        imbox = OffsetImage(p_photo, zoom=0.25) 
                        ab = AnnotationBbox(imbox, (0.20, 0.5), xycoords='axes fraction', frameon=False)
                        ab.set_zorder(2) # Encima del dorsal
                        ax_p.add_artist(ab)
                    else:
                        # Placeholder si no hay foto
                        circle = plt.Circle((0.2, 0.5), 0.18, transform=ax_p.transAxes, color='#eee', zorder=2)
                        ax_p.add_artist(circle)
                    
                    # 3. NOMBRE (Arriba a la derecha)
                    short_name = player['playerName'].split()[-1].upper()
                    ax_p.text(0.42, 0.75, short_name, transform=ax_p.transAxes, 
                             fontsize=8, weight='bold', color='#333', ha='left')
                    
                    # 4. DATOS xT VISUALES (Barra de progreso + Número grande)
                    xt_val = player['xT_val']
                    # Calcular porcentaje de la barra relativo al máximo
                    bar_pct = (xt_val / max_xt_cat) if max_xt_cat > 0 else 0
                    
                    # Número Grande (xT Total)
                    ax_p.text(0.42, 0.45, f"{xt_val:.2f}", transform=ax_p.transAxes,
                             fontsize=14, weight='bold', color=cat_colors.get(cat, '#333'))
                    
                    # Etiqueta pequeña "xT"
                    ax_p.text(0.85, 0.48, "xT", transform=ax_p.transAxes,
                             fontsize=6, color='#777')

                    # Barra de fondo (Gris)
                    rect_bg = plt.Rectangle((0.42, 0.25), 0.50, 0.08, transform=ax_p.transAxes, 
                                          color='#f0f0f0', zorder=3)
                    ax_p.add_artist(rect_bg)
                    
                    # Barra de valor (Color Categoría)
                    rect_val = plt.Rectangle((0.42, 0.25), 0.50 * bar_pct, 0.08, transform=ax_p.transAxes, 
                                           color=cat_colors.get(cat, '#333'), zorder=4)
                    ax_p.add_artist(rect_val)
                    
                    # Dato secundario (xT por partido) debajo de la barra
                    ax_p.text(0.42, 0.10, f"{player['xt_per_match']:.2f} x PARTIDO", transform=ax_p.transAxes,
                             fontsize=6, color='#999', ha='left')

        # ==========================================
        # 3. ÚLTIMOS 5 PARTIDOS (ABAJO)
        # ==========================================
        ax_label = fig.add_axes([0.05, 0.33, 0.2, 0.05])
        ax_label.axis('off')
        ax_label.text(0, 0.5, "ÚLTIMOS 5 PARTIDOS:", fontsize=12, weight='bold', color='#1e3d59')

        recent = last_5_data[-5:]
        start_x = 0.05
        y_pos = 0.08
        width = 0.16
        height = 0.22
        gap = 0.025
        
        for i, match in enumerate(recent):
            ax_sub = fig.add_axes([start_x + i*(width+gap), y_pos, width, height])
            
            flow = match['flow']
            c_team = '#e74c3c' if match['cond'] == 'Local' else '#f39c12'
            c_rival = '#2c3e50'
            
            ax_sub.fill_between(x_axis, flow, 0, where=(flow>=0), interpolate=True, color=c_team, alpha=0.8)
            ax_sub.fill_between(x_axis, flow, 0, where=(flow<0), interpolate=True, color=c_rival, alpha=0.8)
            ax_sub.plot(x_axis, flow, color='#333333', lw=1)
            ax_sub.axhline(0, color='black', lw=0.5)

            y_max_v = max(abs(flow.min()), abs(flow.max())) * 1.4 if not flow.isnull().all() else 0.1
            ax_sub.set_ylim(-y_max_v, y_max_v)

            # Goles (Estilo Etiqueta)
            events = match['events']
            if not events.empty:
                for _, event in events.iterrows():
                    minute = int(event['timeMin'])
                    if minute > 95: continue
                    is_my_team = event['Team Name'] == team_name
                    
                    y_pos_marker = y_max_v * 0.7 if is_my_team else -y_max_v * 0.7
                    y_line_end = y_pos_marker * 0.85
                    
                    ax_sub.plot([minute, minute], [0, y_line_end], color='black', linestyle='-', alpha=0.8, lw=0.8)
                    
                    if ball_img is not None:
                        ab = AnnotationBbox(OffsetImage(ball_img, zoom=0.02), (minute, y_pos_marker), frameon=False, pad=0)
                        ax_sub.add_artist(ab)
                    
                    short_n = event['playerName'].split()[-1]
                    ax_sub.text(minute, y_pos_marker * 1.4, f"{minute}'\n{short_n}", 
                               ha='center', va='center', fontsize=5, weight='bold',
                               bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.1))

            rival = match['rival'] # Nombre completo siempre
            
            # Ajuste dinámico de fuente: si es muy largo (>12 caracteres), letra más pequeña (7), si no, normal (9)
            font_s = 7 if len(rival) > 12 else 9
            
            ax_sub.set_title(f"vs {rival}", fontsize=font_s, weight='bold', pad=12)
            
            r_logo = self.find_team_logo(match['rival'])
            if r_logo is not None:
                 ab = AnnotationBbox(OffsetImage(r_logo, zoom=0.30, alpha=0.3), (0.1, 0.15), xycoords='axes fraction', frameon=False)
                 ax_sub.add_artist(ab)

            ax_sub.set_xlim(0, 95)
            ax_sub.set_xticks([0, 45, 90])
            ax_sub.set_xticklabels(['0', '45', '90'], fontsize=6)
            ax_sub.set_yticks([])
            ax_sub.spines['top'].set_visible(False)
            ax_sub.spines['right'].set_visible(False)
            ax_sub.spines['left'].set_visible(False)

        return fig

def seleccionar_equipo_interactivo(df):
    if df is None or df.empty: return None
    equipos = sorted(df['Team Name'].dropna().unique())
    for i, e in enumerate(equipos, 1): print(f"{i}. {e}")
    for _ in range(3):
        try:
            sel = int(input("\nSelecciona número: ")) - 1
            if 0 <= sel < len(equipos): return equipos[sel]
        except EOFError:
            return equipos[0] if equipos else None
        except: pass
    return equipos[0] if equipos else None

if __name__ == "__main__":
    rep = XTFlowReport()
    if rep.df is not None:
        eq = seleccionar_equipo_interactivo(rep.df)
        if eq:
            pass
            fig = rep.create_report(eq)
            if fig:
                name = f"reporte_xt_final_{eq.replace(' ', '_')}.pdf"
                fig.savefig(name, dpi=300, bbox_inches='tight', orientation='landscape')
                plt.show()