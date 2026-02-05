import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle
from mplsoccer import VerticalPitch
import numpy as np
import warnings
import os
import re
import unicodedata
from matplotlib import rcParams

# Configuración general
warnings.filterwarnings('ignore')
rcParams['font.family'] = 'sans-serif'

# =============================================================================
# FUNCIONES AUXILIARES (LÓGICA DE PERDIDAS.PY)
# =============================================================================

def color_texto_segun_fondo(rgba):
    """Calcula si el texto debe ser blanco o negro por contraste."""
    if isinstance(rgba, str): return 'black' # Fallback
    r, g, b = rgba[:3]
    luminancia = 0.299 * r + 0.587 * g + 0.114 * b
    return 'black' if luminancia > 0.5 else 'white'

def calcular_velocidad_y_zona(df_seq):
    if df_seq.empty: return False, False, 0, False, 0, 0, 0
    eventos_tiro = ['Miss', 'Attempt Saved', 'Goal', 'Post']
    hay_tiro = df_seq['Event Name'].isin(eventos_tiro).any()
    x_vals = df_seq['x'].dropna()
    if x_vals.empty: return False, False, 0, hay_tiro, 0, 0, 0
    x_inicio = x_vals.iloc[0]
    x_fin = x_vals.iloc[-1]
    ultimo_ev = df_seq.iloc[-1]
    if ultimo_ev['Event Name'] == 'Pass' and pd.notna(ultimo_ev.get('Pass End X')):
        x_fin = ultimo_ev['Pass End X']
    outcomes = pd.to_numeric(df_seq['outcome'], errors='coerce').fillna(1)
    en_z4 = df_seq['x'] > 75
    es_pase = df_seq['Event Name'] == 'Pass'
    pass_end_x = pd.to_numeric(df_seq['Pass End X'], errors='coerce').fillna(0)
    pases_a_z4 = es_pase & (pass_end_x > 75)
    candidatos_z4 = en_z4 | pases_a_z4
    exitos_en_z4 = df_seq[candidatos_z4 & (outcomes != 0)]
    llego_z4 = not exitos_en_z4.empty
    if 'timeStamp' in df_seq.columns and pd.api.types.is_datetime64_any_dtype(df_seq['timeStamp']):
         duracion = (df_seq.iloc[-1]['timeStamp'] - df_seq.iloc[0]['timeStamp']).total_seconds()
    else: duracion = 0
    distancia_x = x_fin - x_inicio
    es_rapido = False
    if distancia_x > 0:
        if duracion <= (distancia_x / 10.0) * 2.0: es_rapido = True
    num_pases = len(df_seq[df_seq['Event Name'] == 'Pass'])
    return llego_z4, es_rapido, num_pases, hay_tiro, x_inicio, x_fin, duracion

def obtener_posesiones(df_match):
    mask_seq = (pd.notna(df_match['sequenceId'])) & (df_match['sequenceId'] != 0)
    df_seqs = df_match[mask_seq].copy()
    if df_seqs.empty: return []
    df_seqs['prev_team'] = df_seqs['Team ID'].shift(1)
    df_seqs['cambio_posesion'] = df_seqs['Team ID'] != df_seqs['prev_team']
    df_seqs['possession_group_id'] = df_seqs['cambio_posesion'].cumsum()
    posesiones = []
    for pid, grupo in df_seqs.groupby('possession_group_id'):
        posesiones.append({
            'id_posesion': pid,
            'team_id': str(grupo.iloc[0]['Team ID']).replace('.0', ''),
            't_inicio': grupo.iloc[0]['timeStamp'],
            't_fin': grupo.iloc[-1]['timeStamp'],
            'idx_inicio': grupo.index[0],
            'idx_fin': grupo.index[-1],
            'seq_ids': grupo['sequenceId'].unique().tolist()
        })
    return posesiones

# =============================================================================
# CLASE PRINCIPAL (ESTILO TACTIC)
# =============================================================================

class AnalizadorPerdidas:
    def __init__(self, parquet_file='extraccion_opta/datos_opta_parquet/posesiones.parquet'):
        self.parquet_file = parquet_file
        self.df = None
        self.df_loss = None
        
        # Paletas y estilos Tactic
        self.colors = {
            'text_dark': '#2d3436',
            'text_blue': '#1e3d59',
            'pitch_green': '#2d5a27',
            'lines': 'white',
            'arrow_highlight': '#ffff00', # Amarillo para contraste en verde
            'bg_stats': '#ecf0f1'
        }
        
        self.tipos_buenos = ['Robo', 'Robo_y_Contraataque', 'Robo_y_Ataque_Posicional', 'Robo_y_Posesion']
        self.tipos_malos = ['Fuera', 'Contraataque_Rival', 'Ataque_Posicional_Rival', 'Posesion_Rival', 'Gol_Rival']
        
        self.nombres_cortos = {
            'Robo': 'Robo',
            'Robo_y_Contraataque': 'Robo\nContra',
            'Robo_y_Ataque_Posicional': 'Robo\nAtaque',
            'Robo_y_Posesion': 'Robo\nPosesión',
            'Fuera': 'Fuera',
            'Contraataque_Rival': 'Contra\nRival',
            'Ataque_Posicional_Rival': 'Ataque\nRival',
            'Posesion_Rival': 'Posesión\nRival',
            'Gol_Rival': 'Gol\nRival',
            'Perdida_sin_evento': 'Sin\nEvento'
        }

    def load_data(self):
        try:
            if not os.path.exists(self.parquet_file):
                print(f"❌ No se encuentra el archivo: {self.parquet_file}")
                return False
            
            self.df = pd.read_parquet(self.parquet_file)
            
            # Limpieza básica
            for col in ['x', 'y', 'Pass End X', 'Pass End Y', 'outcome']:
                if col in self.df.columns: self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            if 'Team ID' in self.df.columns:
                self.df['Team ID'] = self.df['Team ID'].astype(str).str.replace(r'\.0$', '', regex=True)
            if 'timeStamp' in self.df.columns:
                self.df['timeStamp'] = pd.to_datetime(self.df['timeStamp'], errors='coerce')
            if 'Match ID' in self.df.columns:
                self.df = self.df.sort_values(['Match ID', 'periodId', 'timeStamp'])
                
            return True
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False

    def process_losses(self, team_name):
        """Ejecuta la lógica de detección de pérdidas para un equipo dado por nombre."""
        if self.df is None: return False

        # Obtener Team ID desde el nombre
        team_row = self.df[self.df['Team Name'] == team_name]
        if team_row.empty:
            print(f"❌ No se encontró el equipo {team_name}")
            return False
        
        target_team_id = str(team_row.iloc[0]['Team ID']).replace('.0', '')
        
        lista_perdidas = []
        match_ids = self.df['Match ID'].unique()
        
        print(f"⚙️ Analizando pérdidas para {team_name} (ID: {target_team_id})...")

        for match_id in match_ids:
            df_match = self.df[self.df['Match ID'] == match_id].sort_values(['periodId', 'timeStamp'])
            posesiones = obtener_posesiones(df_match)
            
            for i in range(len(posesiones) - 1):
                pos_actual = posesiones[i]
                pos_siguiente = posesiones[i+1]
                if pos_actual['team_id'] != target_team_id: continue
                
                # Filtros
                if (pos_actual['t_fin'] - pos_actual['t_inicio']).total_seconds() > 30: continue
                
                df_pos_actual = df_match.loc[pos_actual['idx_inicio']:pos_actual['idx_fin']]
                if df_pos_actual.empty: continue
                
                if df_pos_actual.iloc[-1]['Event Name'] in ['Miss', 'Attempt Saved', 'Goal', 'Post']: continue
                if len(df_pos_actual) == 1 and df_pos_actual.iloc[0]['Event Name'] == 'Aerial': continue

                eventos_hueco = df_match[(df_match['timeStamp'] > pos_actual['t_fin']) & (df_match['timeStamp'] < pos_siguiente['t_inicio'])]
                
                if len(df_pos_actual) == 1:
                    ev = df_pos_actual.iloc[0]
                    if ev['Event Name'] == 'Pass':
                        yv = ev.get('y', 50)
                        if (yv < 0 or yv > 100) and eventos_hueco.empty: continue

                df_pos_rival = df_match.loc[pos_siguiente['idx_inicio']:pos_siguiente['idx_fin']]
                
                es_corte = df_pos_actual.iloc[-1]['Event Name'] in ['Offside Pass', 'Foul'] or (not eventos_hueco.empty and eventos_hueco['Event Name'].isin(['Foul']).any())
                es_fuera = (not eventos_hueco.empty and eventos_hueco['Event Name'].isin(['Out']).any())
                es_out_rival = df_pos_rival['Event Name'].isin(['Out']).any()
                
                z4_riv, rap_riv, pas_riv, tiro_riv, xi_riv, xf_riv, dur_riv = calcular_velocidad_y_zona(df_pos_rival)
                
                recuperacion = False
                dur_recup = 0; z4_recup = False; rap_recup = False; pas_recup = 0; xi_recup = 0
                if i + 2 < len(posesiones):
                    pos_recup = posesiones[i+2]
                    if pos_recup['team_id'] == target_team_id and (pos_recup['t_inicio'] - pos_actual['t_fin']).total_seconds() <= 5:
                        recuperacion = True
                        df_recup = df_match.loc[pos_recup['idx_inicio']:pos_recup['idx_fin']]
                        z4_recup, rap_recup, pas_recup, _, xi_recup, _, dur_recup = calcular_velocidad_y_zona(df_recup)

                es_larga = False
                if len(df_pos_actual) == 1 and df_pos_actual.iloc[0]['Event Name'] == 'Pass':
                    if (df_pos_actual.iloc[0].get('Pass End X', 0) - df_pos_actual.iloc[0].get('x', 0)) > 35: es_larga = True

                tipo = "Desconocido"
                if es_corte: tipo = "Perdida_sin_evento"
                elif es_fuera: tipo = "Fuera"
                elif es_out_rival and pas_riv < 3: tipo = "Fuera"
                elif recuperacion:
                    if dur_recup < 5: tipo = "Robo"
                    else:
                        if z4_recup and rap_recup: tipo = "Robo_y_Contraataque"
                        elif ((z4_recup and not rap_recup) or xi_recup > 75) and pas_recup >= 2: tipo = "Robo_y_Ataque_Posicional"
                        elif not z4_recup: tipo = "Robo" if pas_recup <= 1 else "Robo_y_Posesion"
                        else: tipo = "Robo"
                else:
                    if pas_riv <= 1 and not tiro_riv: tipo = "Perdida_sin_evento"
                    elif z4_riv:
                        min_p = 1 if es_larga else 2
                        tipo = "Contraataque_Rival" if rap_riv and pas_riv >= min_p else "Ataque_Posicional_Rival"
                    elif dur_riv >= 5: tipo = "Posesion_Rival"
                    else: tipo = "Perdida_sin_evento"

                ev_fin = df_pos_actual.iloc[-1]
                orig_x, orig_y = ev_fin.get('x', 0), ev_fin.get('y', 0)
                punto_perdida_x, punto_perdida_y = orig_x, orig_y
                es_un_pase = False

                if ev_fin['Event Name'] == 'Pass':
                    es_un_pase = True
                    punto_perdida_x = ev_fin.get('Pass End X', orig_x)
                    punto_perdida_y = ev_fin.get('Pass End Y', orig_y)
                
                lista_perdidas.append({
                    'x': punto_perdida_x,
                    'y': punto_perdida_y,
                    'orig_x': orig_x,
                    'orig_y': orig_y,
                    'tipo': tipo,
                    'es_pase': es_un_pase
                })

        if not lista_perdidas: return False
        self.df_loss = pd.DataFrame(lista_perdidas).dropna(subset=['x', 'y'])
        return True

    # =========================================================================
    # VISUALIZACIÓN
    # =========================================================================

    def load_background(self):
        return plt.imread("assets/fondo_informes.png") if os.path.exists("assets/fondo_informes.png") else None

    def load_tactic_logo(self):
        return plt.imread("assets/tactic_logo.png") if os.path.exists("assets/tactic_logo.png") else None

    def load_team_logo(self, equipo, target_size=(100, 100)):
        try:
            from PIL import Image
            import numpy as np
        except ImportError: return None
        
        if not os.path.exists('assets/escudos'): return None
        
        def normalize_word(word):
            word = unicodedata.normalize('NFD', word)
            word = ''.join(char for char in word if unicodedata.category(char) != 'Mn')
            return word.lower().strip()
        
        palabras_ignorar = {'cf', 'fc', 'cd', 'ud', 'rcd', 'rc', 'ca', 'de', 'del', 'la', 'las', 'el', 'los'}
        palabras = equipo.split()
        palabras_normalizadas = [normalize_word(p) for p in palabras if normalize_word(p) not in palabras_ignorar and len(normalize_word(p)) > 2]
        palabras_ordenadas = sorted(palabras_normalizadas, key=len, reverse=True)
        all_files = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
        
        best_match_path = None
        
        # Búsqueda exacta parcial
        for palabra_buscar in palabras_ordenadas:
            for filename in all_files:
                nombre_archivo = os.path.splitext(filename)[0]
                nombre_archivo_norm = normalize_word(nombre_archivo)
                if palabra_buscar in nombre_archivo_norm:
                    best_match_path = f"assets/escudos/{filename}"
                    break
            if best_match_path: break
        
        if best_match_path:
            try:
                with Image.open(best_match_path) as img:
                    if img.mode != 'RGBA': img = img.convert('RGBA')
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    final_img = Image.new('RGBA', target_size, (0, 0, 0, 0))
                    paste_x = (target_size[0] - img.width) // 2
                    paste_y = (target_size[1] - img.height) // 2
                    final_img.paste(img, (paste_x, paste_y), img)
                    return np.array(final_img) / 255.0
            except: pass
        return None

    def draw_heatmap_pitch(self, ax):
        """Dibuja el campograma con heatmap y flechas."""
        pitch = VerticalPitch(
            pitch_type='opta',
            pitch_color=self.colors['pitch_green'],
            line_color='white',
            linewidth=2,
            pad_bottom=0.5, pad_top=0.5, pad_left=0.5, pad_right=0.5
        )
        pitch.draw(ax=ax)
        
        x_bins = [0, 25, 50, 75, 100]       
        y_bins = [0, 25, 75, 100]
        total_eventos = len(self.df_loss)
        cmap_pitch = plt.get_cmap('Reds') # Mapa rojo para pérdidas, contraste con verde

        # 1. Heatmap
        for i in range(len(x_bins) - 1):
            xmin, xmax = x_bins[i], x_bins[i+1]
            for j in range(len(y_bins) - 1):
                ymin, ymax = y_bins[j], y_bins[j+1]
                mask = (self.df_loss['x'] >= xmin) & (self.df_loss['x'] < xmax) & (self.df_loss['y'] >= ymin) & (self.df_loss['y'] < ymax)
                count = len(self.df_loss[mask])
                pct = (count / total_eventos * 100) if total_eventos > 0 else 0
                
                norm = min(pct / 15.0, 1.0) 
                color = cmap_pitch(norm)
                # Ajustamos alpha para que se vea bien sobre verde
                alpha = 0.2 + (0.7 * norm) if count > 0 else 0
                
                # Invertir coordenadas para VerticalPitch (x=y, y=x en rect)
                # mplsoccer VerticalPitch invierte ejes visualmente pero coordinate system is opta
                # Rect(xy, width, height). Para vertical, Opta X es vertical, Y horizontal.
                # Rectangle: (left, bottom), width, height
                
                # En vertical pitch Opta:
                # Eje X visual (ancho) corresponde a Y Opta (0-100)
                # Eje Y visual (alto) corresponde a X Opta (0-100)
                
                rect = Rectangle((ymin, xmin), ymax-ymin, xmax-xmin, facecolor=color, alpha=alpha, zorder=1)
                ax.add_patch(rect)
                
                if pct > 1.0:
                    ax.text(ymin + (ymax-ymin)/2, xmin + (xmax-xmin)/2, f"{pct:.1f}%",
                                  ha='center', va='center', color='white', fontweight='bold', fontsize=14,
                                  path_effects=[path_effects.withStroke(linewidth=2, foreground='black')],
                                  zorder=4)

        # 2. Flechas y Puntos (Pérdidas)
        df_contra = self.df_loss[self.df_loss['tipo'] == 'Contraataque_Rival']
        df_resto = self.df_loss[self.df_loss['tipo'] != 'Contraataque_Rival']

        # Resto (Sutil)
        resto_pases = df_resto[df_resto['es_pase'] == True]
        if not resto_pases.empty:
            pitch.arrows(resto_pases.orig_x, resto_pases.orig_y, resto_pases.x, resto_pases.y,
                ax=ax, width=1, headwidth=3, color=self.colors['arrow_highlight'], alpha=0.3, zorder=2)
        
        resto_puntos = df_resto[df_resto['es_pase'] == False]
        if not resto_puntos.empty:
            pitch.scatter(resto_puntos.x, resto_puntos.y, ax=ax, s=15, color=self.colors['arrow_highlight'], alpha=0.3, zorder=2)

        # Contras (Resaltadas)
        contra_pases = df_contra[df_contra['es_pase'] == True]
        if not contra_pases.empty:
            pitch.arrows(contra_pases.orig_x, contra_pases.orig_y, contra_pases.x, contra_pases.y,
                ax=ax, width=3, headwidth=8, color=self.colors['arrow_highlight'], edgecolor='black', alpha=0.9, zorder=3)

        contra_puntos = df_contra[df_contra['es_pase'] == False]
        if not contra_puntos.empty:
            pitch.scatter(contra_puntos.x, contra_puntos.y, ax=ax, s=120, color=self.colors['arrow_highlight'], edgecolors='black', linewidth=1.5, alpha=1.0, zorder=3)

        ax.text(50, 103, "ZONA DE PÉRDIDA", color=self.colors['text_blue'], fontsize=14, fontweight='bold', ha='center')

    def draw_stats_table(self, ax):
        """Dibuja una tabla estilizada con Matplotlib."""
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        zonas_filas = [
            (75, 100, "Z4 - Finalización"), 
            (50, 75, "Z3 - Ataque"), 
            (25, 50, "Z2 - Creación"), 
            (0, 25, "Z1 - Inicio")
        ]
        
        tipos_ordenados = [t for t in self.tipos_buenos if t in self.df_loss['tipo'].unique()] + \
                          [t for t in self.tipos_malos if t in self.df_loss['tipo'].unique()]
        
        col_labels = [self.nombres_cortos.get(t, t) for t in tipos_ordenados]
        total_eventos = len(self.df_loss)
        
        cmap_green = plt.get_cmap('Greens')
        cmap_red = plt.get_cmap('Reds')

        # Configuración Tabla
        x_start = 0.15
        y_start = 0.85
        row_height = 0.18
        col_width = 0.85 / len(col_labels) if col_labels else 0.1
        
        # Título
        ax.text(0.5, 0.96, "CONSECUENCIA TRAS PÉRDIDA", ha='center', va='top', 
                fontsize=14, fontweight='bold', color=self.colors['text_blue'])

        # Headers Columnas
        for i, label in enumerate(col_labels):
            x = x_start + (i * col_width) + (col_width/2)
            ax.text(x, y_start + 0.05, label, ha='center', va='bottom', fontsize=9, fontweight='bold', 
                    color=self.colors['text_dark'], rotation=45 if len(label) > 6 else 0)

        # Filas y Celdas
        for i, (zmin, zmax, zname) in enumerate(zonas_filas):
            y_row = y_start - (i * row_height)
            
            # Etiqueta Fila (Zona)
            ax.text(0.0, y_row - row_height/2, zname, ha='left', va='center', fontsize=10, 
                    fontweight='bold', color=self.colors['text_blue'])
            
            # Línea separadora
            ax.plot([0, 1], [y_row, y_row], color='#bdc3c7', lw=0.5)

            for j, t in enumerate(tipos_ordenados):
                sub = self.df_loss[(self.df_loss['tipo'] == t) & (self.df_loss['x'] >= zmin) & (self.df_loss['x'] < zmax)]
                count = len(sub)
                pct_total = (count / total_eventos * 100) if total_eventos > 0 else 0
                
                x_cell = x_start + (j * col_width)
                
                if count > 0:
                    intensity = min(pct_total / 5.0, 1.0)
                    bg_color = cmap_green(0.2 + 0.6*intensity) if t in self.tipos_buenos else cmap_red(0.2 + 0.6*intensity)
                    
                    # Fondo Celda
                    rect = Rectangle((x_cell + 0.005, y_row - row_height + 0.01), col_width - 0.01, row_height - 0.02, 
                                     facecolor=bg_color, alpha=0.8, transform=ax.transAxes, zorder=2,
                                     edgecolor='white', linewidth=1, joinstyle='round')
                    ax.add_patch(rect)
                    
                    # Texto
                    txt_col = color_texto_segun_fondo(bg_color)
                    ax.text(x_cell + col_width/2, y_row - row_height/2, f"{count}\n({pct_total:.1f}%)", 
                            ha='center', va='center', fontsize=8, fontweight='bold', color=txt_col, transform=ax.transAxes, zorder=3)
                else:
                    ax.text(x_cell + col_width/2, y_row - row_height/2, "-", 
                            ha='center', va='center', fontsize=8, color='#95a5a6', transform=ax.transAxes)

        # Línea final
        ax.plot([0, 1], [y_start - 4*row_height, y_start - 4*row_height], color='#bdc3c7', lw=1)

    def create_visualization(self, team_name):
        if self.df_loss is None or self.df_loss.empty:
            print("❌ No hay datos para graficar.")
            return None

        # Configuración A4 Horizontal
        fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
        
        # Fondo
        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.15)
            ax_bg.axis('off')

        # Grid: Izquierda (Pitch) 40%, Derecha (Stats) 60%
        gs = fig.add_gridspec(1, 2, width_ratios=[0.45, 0.55], left=0.05, right=0.95, top=0.85, bottom=0.05, wspace=0.1)
        
        ax_pitch = fig.add_subplot(gs[0])
        ax_table = fig.add_subplot(gs[1])

        # Logos y Títulos
        fig.suptitle('ANÁLISIS DE PÉRDIDAS Y TRANSICIÓN', fontsize=20, fontweight='bold', color=self.colors['text_blue'], y=0.96, family='sans-serif')
        
        if (tactic_logo := self.load_tactic_logo()) is not None:
            ax_logo1 = fig.add_axes([0.02, 0.90, 0.08, 0.08], anchor='NW', zorder=10)
            ax_logo1.imshow(tactic_logo)
            ax_logo1.axis('off')
        
        if (team_logo := self.load_team_logo(team_name)) is not None:
            ax_logo2 = fig.add_axes([0.90, 0.90, 0.08, 0.08], anchor='NE', zorder=10)
            ax_logo2.imshow(team_logo)
            ax_logo2.axis('off')

        # Dibujar
        self.draw_heatmap_pitch(ax_pitch)
        self.draw_stats_table(ax_table)
        
        return fig

# =============================================================================
# MAIN INTERACTIVO
# =============================================================================
def seleccionar_equipo_interactivo(df):
    try:
        equipos = sorted(df['Team Name'].dropna().unique())
        if not equipos: return None
        
        print("\nEquipos disponibles:")
        for i, equipo in enumerate(equipos, 1):
            if i % 4 == 0: print(f"{i}. {equipo}")
            else: print(f"{i}. {equipo:<25}", end="")
        print("\n")
        
        while True:
            try:
                sel = input(f"Selecciona un equipo (1-{len(equipos)}): ").strip()
                idx = int(sel) - 1
                if 0 <= idx < len(equipos): return equipos[idx]
            except ValueError: pass
            print("Entrada inválida.")
    except Exception: return None

if __name__ == "__main__":
    analyzer = AnalizadorPerdidas()
    
    if analyzer.load_data():
        equipo = seleccionar_equipo_interactivo(analyzer.df)
        
        if equipo:
            if analyzer.process_losses(equipo):
                fig = analyzer.create_visualization(equipo)
                if fig:
                    filename = f"analisis_perdidas_{equipo.replace(' ', '_')}.png"
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"\n✅ Gráfica guardada: {filename}")
                    plt.show()
            else:
                print("No se encontraron pérdidas significativas.")