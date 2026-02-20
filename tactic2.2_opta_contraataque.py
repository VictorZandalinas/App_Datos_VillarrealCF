import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.cluster import DBSCAN
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import warnings
import os
import unicodedata
import re
from difflib import SequenceMatcher 
from matplotlib.patches import FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

class AnalizadorContraataques:
    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/match_events.parquet"):
        self.data_path = data_path
        self.df = None
        self.team_filter = None
        
        try:
            self.player_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/player_stats.parquet",
                                                 columns=['Player ID', 'Player Name'])
        except:
            self.player_stats = None

    def load_data(self, team_filter=None):
        try:
            pass
            self.df = pd.read_parquet(self.data_path)
            self.df['timeStamp'] = pd.to_datetime(self.df['timeStamp'], format='ISO8601')
            for col in ['x', 'y', 'Pass End X', 'Pass End Y']:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            self.df = self.df.sort_values(['Match ID', 'timeStamp']).reset_index(drop=True)
            self.team_filter = team_filter

            # --- EXTRACCIÓN MASIVA EN UN SOLO PASO ---
            self.all_league_sequences = self.extract_all_league_sequences()
            
            # --- CÁLCULO DE RANKING ---
            self.ranking_data = Counter([s['team'] for s in self.all_league_sequences])
            
            # --- CÁLCULO DE MEDIAS ---
            total_seqs = len(self.all_league_sequences)
            origin_counts = Counter([s['origin'] for s in self.all_league_sequences])
            self.league_averages = {k: (v / total_seqs) * 100 for k, v in origin_counts.items()}
            
            return True
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            import traceback
            traceback.print_exc()
            return False

    def extract_all_league_sequences(self):
        """Extrae los contraataques de TODOS los equipos en una sola pasada por los datos."""
        all_sequences = []
        
        # Agrupamos por partido una sola vez
        for match_id, match_group in self.df.groupby('Match ID'):
            events = match_group.sort_values('timeStamp').reset_index()
            i = 0
            num_events = len(events)
            
            while i < num_events:
                row = events.iloc[i]
                
                # Si el evento es una recuperación (de cualquier equipo)
                if self.is_recovery(row):
                    team_name = row['Team Name']
                    start_x = row['x']
                    
                    # Filtros de zona y tiempo
                    if start_x < 25: max_dur = 12
                    elif start_x < 50: max_dur = 10
                    elif 50 <= start_x < 70: max_dur = 5
                    else: 
                        i += 1
                        continue
                    
                    start_time = row['timeStamp']
                    current_seq = [row.to_dict()]
                    reached_target = False
                    last_event_time = start_time
                    
                    # Rastrear hacia adelante
                    for j in range(i + 1, num_events):
                        nxt = events.iloc[j]
                        dur = (nxt['timeStamp'] - start_time).total_seconds()
                        gap = (nxt['timeStamp'] - last_event_time).total_seconds()
                        
                        if dur > max_dur: break
                        
                        if nxt['Team Name'] != team_name:
                            # Continuidad si es mismo tiempo o gap < 2s
                            if nxt['timeStamp'] == last_event_time: continue
                            if gap > 2: break
                            continue
                        
                        # Es del mismo equipo
                        current_seq.append(nxt.to_dict())
                        last_event_time = nxt['timeStamp']
                        
                        # Comprobar llegada a zona final
                        ex = nxt.get('Pass End X', nxt['x'])
                        final_x = ex if pd.notna(ex) else nxt['x']
                        
                        if nxt['x'] > 83 or final_x > 83 or nxt['typeId'] in [13, 14, 15, 16]:
                            reached_target = True
                            break
                    
                    if reached_target and len(current_seq) > 1:
                        # Clasificamos contexto (esta función usa self.df, es rápida)
                        origin = self.get_context_origin(row['index'], team_name)
                        all_sequences.append({
                            'team': team_name,
                            'events': current_seq,
                            'origin': origin
                        })
                        i = j # Saltamos eventos procesados
                        continue
                i += 1
        return all_sequences

    def get_sequences_for_team(self, team_name):
        """Filtra las secuencias ya extraídas para el equipo seleccionado."""
        return [s for s in self.all_league_sequences if s['team'] == team_name]

    def get_context_origin(self, start_idx, team_name):
        """Clasifica la acción previa según las reglas de 8 segundos y zonas."""
        start_row = self.df.loc[start_idx]
        start_time = start_row['timeStamp']
        lookback_time = start_time - timedelta(seconds=8)
        
        # Filtrar eventos del rival en los últimos 8 segundos
        prev_events = self.df[
            (self.df['Match ID'] == start_row['Match ID']) &
            (self.df['timeStamp'] >= lookback_time) &
            (self.df['timeStamp'] < start_time) &
            (self.df['Team Name'] != team_name)
        ].copy()

        if not prev_events.empty:
            # 1. Corner
            if 'Corner taken' in prev_events.columns:
                if (prev_events['Corner taken'] == 'Sí').any():
                    return "Tras Corner Rival"
            
            # 2. Falta (Lateral o Central)
            if 'Free kick taken' in prev_events.columns:
                fks = prev_events[prev_events['Free kick taken'] == 'Sí']
                if not fks.empty:
                    last_fk = fks.iloc[-1]
                    y = last_fk['y']
                    if y < 21.1 or y > 78.9:
                        return "Tras Falta Lateral"
                    else:
                        return "Tras Falta Central"

            # 3. Saque de banda
            if 'Throw in' in prev_events.columns:
                if (prev_events['Throw in'] == 'Sí').any():
                    return "Tras Saque de Banda"

        # --- REGLAS DE POSICIÓN PARA ORIGEN ---
        rx = start_row['x']
        # Prioridad a las jugadas de campo rival (Presión Alta 50-70)
        if 50 <= rx < 70:
            return "Presión Alta"
        elif rx < 50:
            return "Repliegue Bajo"

    
    def load_team_logo(self, equipo, target_size=(80, 80)):
        """Carga y redimensiona el logo del equipo buscando por palabras más largas primero"""
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            return None
        
        if not os.path.exists('assets/escudos'):
            print(f"⚠️ No existe la carpeta assets/escudos")
            return None
        
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
            if palabra_norm not in palabras_ignorar and len(palabra_norm) > 2:
                palabras_normalizadas.append(palabra_norm)
        
        # Ordenar por longitud (más larga primero)
        palabras_ordenadas = sorted(palabras_normalizadas, key=len, reverse=True)
        
        # Obtener todos los archivos disponibles
        all_files = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
        
        # Buscar por cada palabra en orden de longitud
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
        
        # Fallback por similitud
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
    
    def load_background(self): 
        return plt.imread("assets/fondo_informes.png") if os.path.exists("assets/fondo_informes.png") else None

    def load_tactic_logo(self):
        logo_path = "assets/tactic_logo.png"
        return plt.imread(logo_path) if os.path.exists(logo_path) else None

    def is_recovery(self, event):
        """IDs Opta de recuperación exitosa."""
        recovery_ids = [7, 8, 44, 49, 52, 54, 59, 74]
        try:
            tid = int(event['typeId'])
            outcome = int(event['outcome'])
            if tid in [7, 8, 49, 52, 54, 74]: return True
            if tid in [44, 59]: return outcome == 1
        except:
            return False
        return False

    def extract_counter_attack_sequences(self, team_name, debug=True):
        sequences = []
        team_df = self.df[self.df['Team Name'] == team_name] if not debug else self.df
        
        for match_id, match_group in self.df.groupby('Match ID'):
            events = match_group.sort_values('timeStamp').reset_index()
            i = 0
            while i < len(events):
                row = events.iloc[i]
                if row['Team Name'] == team_name and self.is_recovery(row):
                    start_x = row['x']
                    # REGLAS DE ZONA Y TIEMPO
                    if start_x < 25: max_dur = 12
                    elif start_x < 50: max_dur = 10
                    elif 50 <= start_x < 70: max_dur = 5
                    else: # x >= 70 se descarta
                        i += 1
                        continue
                    
                    start_time = row['timeStamp']
                    current_seq = [row.to_dict()]
                    reached_target = False
                    
                    for j in range(i + 1, len(events)):
                        nxt = events.iloc[j]
                        dur = (nxt['timeStamp'] - start_time).total_seconds()
                        gap = (nxt['timeStamp'] - pd.to_datetime(current_seq[-1]['timeStamp'])).total_seconds()
                        
                        if dur > max_dur: break
                        if nxt['Team Name'] != team_name:
                            if nxt['timeStamp'] == current_seq[-1]['timeStamp']: continue
                            if gap > 2: break
                            continue
                        
                        current_seq.append(nxt.to_dict())
                        ex = nxt.get('Pass End X', nxt['x'])
                        final_x = ex if pd.notna(ex) else nxt['x']
                        
                        if nxt['x'] > 83 or final_x > 83 or nxt['typeId'] in [13, 14, 15, 16]:
                            reached_target = True
                            break
                            
                    if reached_target and len(current_seq) > 1:
                        origin = self.get_context_origin(row['index'], team_name)
                        sequences.append({'events': current_seq, 'origin': origin})
                        i = j
                        continue
                i += 1
        return sequences

    def get_tactical_arquetype(self, seq_events):
        """Genera una clave táctica simplificada para evitar fragmentación excesiva."""
        first = seq_events[0]
        last = seq_events[-1]
        
        # 1. Zona de inicio (Coherente con las reglas de tiempo)
        rx = first['x']
        if 50 <= rx < 70:
            s_depth = "Presión Alta"
        elif 25 <= rx < 50:
            s_depth = "Zona 25-50m"
        else:
            s_depth = "Zona <25m"
            
        # 2. Carril de inicio (Simplificado)
        s_lane = "Banda" if (first['y'] > 70 or first['y'] < 30) else "Centro"
        
        # 3. Estilo de progresión
        passes = [e for e in seq_events if pd.notna(e.get('Pass End X'))]
        dist = [np.sqrt((p['Pass End X']-p['x'])**2 + (p['Pass End Y']-p['y'])**2) for p in passes]
        estilo = "Directo" if (dist and np.mean(dist) > 30) else "Combinado"
        
        return f"{s_depth} ({s_lane}) -> {estilo}"

    def find_most_similar_patterns(self, sequences, top_n=3):
        """Agrupa secuencias y extrae estadísticas reales de CADA grupo."""
        if not sequences: return []
        
        arquetipos = defaultdict(list)
        for s in sequences:
            key = self.get_tactical_arquetype(s['events'])
            arquetipos[key].append(s)
        
        # Ordenar por los grupos que más se repiten
        sorted_groups = sorted(arquetipos.items(), key=lambda x: len(x[1]), reverse=True)
        
        cluster_data = []
        for key, members in sorted_groups[:top_n]:
            # IMPORTANTE: Sacar el origen real predominante de ESTE grupo específico
            origins = [m['origin'] for m in members]
            main_origin_name, count = Counter(origins).most_common(1)[0]
            pct_origin = (count / len(members)) * 100
            
            # Jugadores del grupo
            starters = [m['events'][0].get('playerName', '???') for m in members]
            finishers = [m['events'][-1].get('playerName', '???') for m in members]
            
            cluster_data.append({
                'label': key,
                'count': len(members),
                'all_sequences': [m['events'] for m in members],
                'starter': Counter(starters).most_common(1)[0][0],
                'finisher': Counter(finishers).most_common(1)[0][0],
                'origin_info': f"{pct_origin:.0f}% {main_origin_name}"
            })
        return cluster_data

    def find_most_similar_patterns(self, sequences, top_n=3):
        """Agrupa secuencias y calcula los jugadores/orígenes clave de cada grupo."""
        if not sequences: return []
        
        arquetipos = defaultdict(list)
        for s in sequences:
            key = self.get_tactical_arquetype(s['events'])
            arquetipos[key].append(s) # Guardamos la secuencia completa con su 'origin'
        
        sorted_arquetipos = sorted(arquetipos.items(), key=lambda x: len(x[1]), reverse=True)
        
        cluster_data = []
        for key, members in sorted_arquetipos[:top_n]:
            # Extraer estadísticas del grupo
            starters = [m['events'][0].get('playerName', 'Desconocido') for m in members]
            finishers = [m['events'][-1].get('playerName', 'Desconocido') for m in members]
            origins = [m['origin'] for m in members]
            
            # Obtener el más común de cada uno
            main_starter = Counter(starters).most_common(1)[0][0]
            main_finisher = Counter(finishers).most_common(1)[0][0]
            main_origin_tuple = Counter(origins).most_common(1)[0]
            pct_origin = (main_origin_tuple[1] / len(members)) * 100
            
            cluster_data.append({
                'label': key,
                'count': len(members),
                'all_sequences': [m['events'] for m in members],
                'starter': main_starter,
                'finisher': main_finisher,
                'origin_info': f"{pct_origin:.0f}% {main_origin_tuple[0]}"
            })
        return cluster_data

    def draw_sequence_pattern(self, ax, all_seqs, color, title):
        pitch = VerticalPitch(pitch_type='opta', pitch_color='#2d5a27', line_color='white', linewidth=2)
        pitch.draw(ax=ax)
        
        def plot_path(events, is_lead=False):
            alpha = 0.9 if is_lead else 0.08
            lw = 2.5 if is_lead else 1.0
            
            # Crear hilo continuo
            full_x, full_y = [], []
            for e in events:
                full_x.append(e['x']); full_y.append(e['y'])
                if pd.notna(e.get('Pass End X')):
                    full_x.append(e['Pass End X']); full_y.append(e['Pass End Y'])
            
            # Dibujar línea base
            ax.plot(full_y, full_x, color=color, lw=lw, alpha=alpha, solid_capstyle='round', zorder=2)
            
            # Cabezas de flecha
            for e in events:
                if pd.notna(e.get('Pass End X')):
                    ax.annotate('', xy=(e['Pass End Y'], e['Pass End X']), xytext=(e['y'], e['x']),
                                arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=alpha, mutation_scale=10))

        for seq in all_seqs: plot_path(seq, is_lead=False)
        if all_seqs: 
            plot_path(all_seqs[0], is_lead=True)
            ax.scatter(all_seqs[0][0]['y'], all_seqs[0][0]['x'], color='white', edgecolor=color, s=70, zorder=10, lw=2)

        ax.set_title(f"{title}\n({len(all_seqs)} veces)", color='white', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', fc='#2c3e50', alpha=0.8))

    def create_visualization(self, team_name, sequences):
        if not sequences: return None
        patterns = self.find_most_similar_patterns(sequences)
        origin_counts = Counter([s['origin'] for s in sequences])
        total_seqs = len(sequences)

        fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
        
        # Fondo y Logos
        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.15)
            ax_bg.axis('off')

        fig.suptitle(f'INFORME ESTRATÉGICO: CONTRAATAQUES - {team_name.upper()}', fontsize=22, fontweight='bold', y=0.96)
        
        # Grid: Fila 1 (3 patrones) | Fila 2 (Barras media, Ranking, Info)
        gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1], hspace=0.6, wspace=0.4, top=0.85, bottom=0.1)

        # --- PATRONES (Top 3) ---
        colors = ['#f1c40f', '#3498db', '#e74c3c']
        for i in range(3):
            ax = fig.add_subplot(gs[0, i])
            if i < len(patterns):
                p = patterns[i]
                self.draw_sequence_pattern(ax, p['all_sequences'], colors[i], f"PATRÓN #{i+1}")
                # Info debajo del campograma
                s_name = str(p['starter']).split()[-1].upper()
                f_name = str(p['finisher']).split()[-1].upper()
                info = f"📍 {p['origin_info']}\n🏃 INICIA: {s_name}\n🎯 ACABA: {f_name}"
                ax.text(0.5, -0.2, info, transform=ax.transAxes, ha='center', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=colors[i], lw=1.5))
            else: ax.axis('off')

        # --- GRÁFICO BARRAS: EQUIPO VS MEDIA ---
        ax_bar = fig.add_subplot(gs[1, :2])
        labels = ["Presión Alta", "Repliegue Bajo", "Tras Corner Rival", "Tras Falta Lateral", "Tras Saque de Banda"]
        labels = [l for l in labels if l in self.league_averages]
        
        y = np.arange(len(labels))
        height = 0.35
        
        team_pcts = [(origin_counts[l]/total_seqs)*100 for l in labels]
        league_pcts = [self.league_averages[l] for l in labels]
        
        ax_bar.barh(y + height/2, team_pcts, height, label=team_name, color='#3498db', edgecolor='black')
        ax_bar.barh(y - height/2, league_pcts, height, label='Media Liga', color='#bdc3c7', alpha=0.6)
        
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels(labels, fontsize=8, fontweight='bold')
        ax_bar.set_title("ORIGEN: EQUIPO VS MEDIA (%)", fontsize=10, fontweight='bold')
        ax_bar.legend(fontsize=8, loc='lower right')
        ax_bar.spines['top'].set_visible(False); ax_bar.spines['right'].set_visible(False)

        # --- RANKING TOTAL CON ESCUDOS ---
        ax_rank = fig.add_subplot(gs[1, 2])
        top_ranking = dict(self.ranking_data.most_common(10))
        y_rank = np.arange(len(top_ranking))
        teams_rank = list(top_ranking.keys())
        counts_rank = list(top_ranking.values())
        
        bars = ax_rank.barh(y_rank, counts_rank, color=['#f1c40f' if t == team_name else '#34495e' for t in teams_rank])
        ax_rank.set_xlim(0, max(counts_rank) * 1.3) # Da un 30% de espacio extra para el logo
        ax_rank.set_yticks(y_rank)
        ax_rank.set_yticklabels(teams_rank, fontsize=7)
        ax_rank.set_title("RANKING TOTAL CONTRAS", fontsize=10, fontweight='bold')
        ax_rank.invert_yaxis()
        
        # Añadir escudos
        for idx, (t, val) in enumerate(zip(teams_rank, counts_rank)):
            logo = self.load_team_logo(t, target_size=(25, 25))
            if logo is not None:
                ab = AnnotationBbox(OffsetImage(logo, zoom=0.4), (val, idx), frameon=False, xybox=(12, 0), boxcoords="offset points")
                ax_rank.add_artist(ab)

        # --- INFO TÉCNICA ---
        ax_info = fig.add_subplot(gs[1, 3])
        ax_info.axis('off')
        resumen = f"TOTAL CONTRAS: {total_seqs}\n\nREGLAS TIEMPO:\nx<25m: 12s\n25-50m: 10s\n50-70m: 5s\n\nCONTINUIDAD:\nRival < 2s"
        ax_info.text(0.5, 0.5, resumen, ha='center', va='center', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=1', fc='#f8f9fa', ec='#bdc3c7'))

        return fig

def seleccionar_equipo():
    df_temp = pd.read_parquet("extraccion_opta/datos_opta_parquet/match_events.parquet", columns=['Team Name'])
    equipos = sorted(df_temp['Team Name'].dropna().unique())
    for i, eq in enumerate(equipos, 1):
        pass
    idx = int(input(f"\nSelecciona equipo (1-{len(equipos)}): ")) - 1
    return equipos[idx]

def main():
    equipo = seleccionar_equipo()
    if not equipo: return
    
    analyzer = AnalizadorContraataques()
    # load_data ahora hace todo el trabajo pesado de la liga
    if analyzer.load_data(team_filter=equipo):
        # En lugar de extraer, simplemente filtramos lo que ya tenemos
        sequences = analyzer.get_sequences_for_team(equipo)
        
        if not sequences:
            pass
            return
            
        fig = analyzer.create_visualization(equipo, sequences)
        if fig:
            filename = f"contraataques_{equipo.replace(' ', '_')}.pdf"
            fig.savefig(filename, bbox_inches='tight', dpi=300)
            plt.show()

if __name__ == "__main__":
    main()