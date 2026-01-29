import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.cluster import DBSCAN
from collections import Counter
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import warnings
import os
import json  # 🔥 NUEVO
import base64  # 🔥 NUEVO
import re  # 🔥 NUEVO
import unicodedata  # 🔥 NUEVO
from io import BytesIO  # 🔥 NUEVO
from PIL import Image  # 🔥 NUEVO
from sklearn.cluster import DBSCAN
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

try:
    from socceraction.vaep import features as fs
    from socceraction.vaep import labels as lab
    SOCCERACTION_AVAILABLE = True
except ImportError:
    SOCCERACTION_AVAILABLE = False

warnings.filterwarnings('ignore')

class AnalizadorSaquesPorteria:
    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/open_play_events.parquet"):
        self.data_path = data_path
        self.df = None
        self.df_complete = None  # 🔥 NUEVO
        self.sequences = []
        self.photos_data = None
        self.team_filter = None
        
        # 🔥 CARGAR PLAYER_STATS
        try:
            self.player_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/player_stats.parquet")
            print(f"✅ Player stats cargado: {len(self.player_stats)} registros")
        except Exception as e:
            print(f"❌ Error cargando player_stats: {e}")
            self.player_stats = None
    
    def get_opponent_by_week(self, team_name):
        """Obtiene el rival jugado en cada jornada DESDE open_play_events.parquet"""
        if self.df_complete is None:
            return {}
        
        opponents = {}
        
        # 🔥 USAR EL DATAFRAME COMPLETO para encontrar rivales
        team_matches = self.df_complete[self.df_complete['Team Name'] == team_name][['Match ID', 'Week']].drop_duplicates()
        
        print(f"\n🔍 DEBUG get_opponent_by_week:")
        print(f"   Equipo buscado: {team_name}")
        print(f"   Partidos encontrados: {len(team_matches)}")
        
        for _, match_row in team_matches.iterrows():
            match_id = match_row['Match ID']
            week = int(match_row['Week']) if pd.notna(match_row.get('Week')) else 0
            
            # 🔥 Buscar en el dataframe COMPLETO
            match_teams = self.df_complete[self.df_complete['Match ID'] == match_id]['Team Name'].unique()
            
            print(f"   Match ID {match_id}, Week {week}: Equipos = {match_teams}")
            
            for opponent in match_teams:
                if opponent != team_name:
                    opponents[week] = opponent
                    print(f"      ✅ Rival asignado: J{week} vs {opponent}")
                    break
        
        print(f"   Total rivales encontrados: {len(opponents)}")
        
        return opponents

    def get_zone_from_coords(self, x, y):
        """Devuelve el número de zona (1-22) según las coordenadas x, y."""
        # Zonas 1-10 (Coordenadas Y personalizadas)
        if 0 <= x < 11.50:
            if 36.8 <= y < 63.2: return 1
            if 21.1 <= y < 36.8: return 2
            if 63.2 <= y < 78.9: return 3
            if 0 <= y < 21.1: return 4
            if 78.9 <= y <= 100: return 5
        elif 11.50 <= x < 23:
            if 21.1 <= y < 36.8: return 6
            if 63.2 <= y < 78.9: return 7
            if 0 <= y < 21.1: return 8
            if 78.9 <= y <= 100: return 9
            if 36.8 <= y < 63.2: return 10
        
        # --- 🔥 ZONAS 11-22 (NUEVA LÓGICA CON 3 CARRILES) ---
        # Carril Izquierdo (y >= 66.6), Central (33.3 <= y < 66.6), Derecho (y < 33.3)
        elif 23 <= x < 34.50:
            if y >= 66.6: return 11
            if 33.3 <= y < 66.6: return 12
            if y < 33.3: return 13
        elif 34.50 <= x < 50:
            if y >= 66.6: return 14
            if 33.3 <= y < 66.6: return 15
            if y < 33.3: return 16
        elif 50 <= x < 75:
            if y >= 66.6: return 17
            if 33.3 <= y < 66.6: return 18
            if y < 33.3: return 19
        elif 75 <= x <= 100:
            if y >= 66.6: return 20
            if 33.3 <= y < 66.6: return 21
            if y < 33.3: return 22
            
        return None # Si por alguna razón no cae en ninguna zona
    
    def analyze_zone_patterns(self, sequences):
        """
        LÓGICA MEJORADA: Ahora agrupa las secuencias RAW por cada sub-patrón
        para poder calcular promedios de coordenadas más tarde.
        """
        # Usamos un defaultdict para agrupar las secuencias originales por patrón
        from collections import defaultdict
        patterns_data = defaultdict(list)
        
        for seq in sequences:
            successful_seq = [evt for evt in seq if evt['outcome'] == 1]
            if not successful_seq:
                continue

            # Obtenemos la secuencia de zonas
            start_zone = self.get_zone_from_coords(successful_seq[0]['x'], successful_seq[0]['y'])
            if start_zone is None: start_zone = 1
            
            zone_sequence = [start_zone]
            for event in successful_seq:
                dest_zone = self.get_zone_from_coords(event['end_x'], event['end_y'])
                if dest_zone is None: continue
                
                if dest_zone != zone_sequence[-1]:
                    zone_sequence.append(dest_zone)
                elif len(zone_sequence) == 1 and dest_zone == zone_sequence[-1]:
                    zone_sequence.append(dest_zone)

            # 🔥 AQUÍ ESTÁ LA MAGIA: Guardamos la secuencia RAW para cada sub-patrón
            if len(zone_sequence) > 1:
                for i in range(2, len(zone_sequence) + 1):
                    sub_pattern = tuple(zone_sequence[:i])
                    patterns_data[sub_pattern].append(seq) # Guardamos la secuencia completa original

        # Ahora convertimos el defaultdict a la lista de diccionarios final
        final_patterns_list = []
        for pattern, raw_sequences_list in patterns_data.items():
            # La categoría se basa en el primer pase de la primera secuencia del grupo
            first_pass = raw_sequences_list[0][0]
            first_zone_dest = self.get_zone_from_coords(first_pass['end_x'], first_pass['end_y'])
            
            category = 'medios' # Default
            if first_zone_dest is not None:
                if 1 <= first_zone_dest <= 10: category = 'cortos'
                elif 11 <= first_zone_dest <= 16: category = 'medios'
                elif first_zone_dest > 16: category = 'largos'

            final_patterns_list.append({
                'sequence': pattern,
                'count': len(raw_sequences_list),
                'type': category,
                'raw_sequences': raw_sequences_list  # 🔥 Guardamos las secuencias para el dibujo
            })
        
        top_20_global = sorted(final_patterns_list, key=lambda x: x['count'], reverse=True)[:20]
        return top_20_global
    
    def draw_legend_zone_patterns(self, ax, top_5_patterns, total_valid_sequences):
        """Ranking visual de patrones de zona - DIVIDIDO EN 3 SECCIONES (VERSIÓN COMPACTA)"""
        ax.axis('off')
        ax.set_facecolor('#f8f9fa')
        
        ax.text(0.5, 0.99, 'PATRONES POR ZONAS', 
                fontsize=9, fontweight='bold', ha='center', va='top', 
                transform=ax.transAxes, color='#2c3e50', family='sans-serif')
        
        if not top_5_patterns:
            ax.text(0.5, 0.60, 'RANKING PATRONES POR ZONAS\n\n(Sin datos)', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, fontweight='bold', color='#7f8c8d')
            return
        
        seccion_1 = [p for p in top_5_patterns if p.get('seccion') == 1]
        seccion_2 = [p for p in top_5_patterns if p.get('seccion') == 2]
        seccion_3 = [p for p in top_5_patterns if p.get('seccion') == 3]
        
        y_pos = 0.90
        contador_global = 1
        
        # ============ SECCIÓN 1: UNA O DOS ZONAS ============
        if seccion_1:
            ax.text(0.5, y_pos + 0.02, '━━━ 1-2 ZONAS ━━━',
                    fontsize=7, fontweight='bold', ha='center', va='center',
                    color='#34495e', transform=ax.transAxes, style='italic')
            y_pos -= 0.07 # 🔥 AJUSTADO: Menos espacio para el título (antes 0.08)
            
            for pattern in seccion_1:
                self._dibujar_patron_leyenda(ax, pattern, y_pos, contador_global, total_valid_sequences)
                y_pos -= 0.11 # 🔥 AJUSTADO: Menos espacio por cada patrón (antes 0.14)
                contador_global += 1
        
        # ============ SECCIÓN 2: MÁS DE 2 ZONAS ============
        if seccion_2:
            y_pos -= 0.01 # 🔥 AJUSTADO: Menos espacio entre secciones (antes 0.02)
            ax.text(0.5, y_pos + 0.02, '━━━ +2 ZONAS ━━━', 
                    fontsize=7, fontweight='bold', ha='center', va='center',
                    color='#34495e', transform=ax.transAxes, style='italic')
            y_pos -= 0.07 # 🔥 AJUSTADO: Menos espacio para el título (antes 0.08)
            
            for pattern in seccion_2:
                self._dibujar_patron_leyenda(ax, pattern, y_pos, contador_global, total_valid_sequences)
                y_pos -= 0.11 # 🔥 AJUSTADO: Menos espacio por cada patrón (antes 0.14)
                contador_global += 1
        
        # ============ SECCIÓN 3: PROGRESIÓN ALTA ============
        if seccion_3:
            y_pos -= 0.01 # 🔥 AJUSTADO: Menos espacio entre secciones (antes 0.02)
            ax.text(0.5, y_pos + 0.02, '━━━ PROGRESIÓN ZONA ≥13 ━━━', 
                    fontsize=7, fontweight='bold', ha='center', va='center',
                    color='#34495e', transform=ax.transAxes, style='italic')
            y_pos -= 0.07 # 🔥 AJUSTADO: Menos espacio para el título (antes 0.08)
            
            for pattern in seccion_3:
                self._dibujar_patron_leyenda(ax, pattern, y_pos, contador_global, total_valid_sequences)
                y_pos -= 0.11 # 🔥 AJUSTADO: Menos espacio por cada patrón (antes 0.14)
                contador_global += 1
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _dibujar_patron_leyenda(self, ax, pattern, y_pos, numero, total_valid_sequences):
        """Dibuja un patrón individual en la leyenda"""
        position_colors = ['#e74c3c', '#f39c12', '#f1c40f', '#95a5a6', '#bdc3c7', '#7f8c8d']
        color_borde = position_colors[min(numero-1, len(position_colors)-1)]
        
        # Recuadro
        rect_bg = mpatches.FancyBboxPatch((0.02, y_pos - 0.06), 0.96, 0.12,
                            boxstyle="round,pad=0.01", facecolor='white',
                            edgecolor=color_borde, linewidth=2, alpha=0.95)
        ax.add_patch(rect_bg)
        
        # Número
        ax.text(0.08, y_pos, f"#{numero}", fontsize=13, fontweight='bold', 
                va='center', ha='center', color=color_borde, transform=ax.transAxes)
        
        # Color y etiqueta de categoría
        pattern_color = pattern.get('color', '#f9c80e')
        categoria = pattern.get('categoria', 'N/A')
        
        if categoria == 'dos_zonas_directo':
            type_label = 'JD'  # Juego directo
        elif categoria == 'dos_zonas_corto':
            type_label = 'JC'  # Juego corto
        elif categoria == 'mas_dos_zonas':
            type_label = '+2'  # Más de 2 zonas
        elif categoria == 'progresion_alta':
            type_label = 'P+'  # Progresión alta
        else:
            type_label = '?'
        
        ax.text(0.18, y_pos + 0.03, type_label, fontsize=7, fontweight='bold',
                va='center', ha='center', color=pattern_color,
                bbox=dict(boxstyle='circle,pad=0.25', facecolor='white',
                        edgecolor=pattern_color, linewidth=2),
                transform=ax.transAxes)
        
        # Secuencia
        seq_str = " → ".join(map(str, pattern['sequence']))
        if len(seq_str) > 18:
            seq_str = seq_str[:15] + "..."
            
        ax.text(0.30, y_pos + 0.02, seq_str, fontsize=9, fontweight='bold', 
                va='center', ha='left', color='#2c3e50', transform=ax.transAxes)
        
        # Porcentaje
        percentage = (pattern['count'] / total_valid_sequences) * 100 if total_valid_sequences > 0 else 0
        ax.text(0.78, y_pos - 0.03, f"{percentage:.1f}%", fontsize=9, 
                fontweight='bold', va='center', ha='left', color=pattern_color,
                transform=ax.transAxes)

    def draw_legend_first_pass(self, ax, receiver_stats, total_first_passes):
        """
        Ranking visual de receptores principales (VERSIÓN CON COLORES UNIFICADOS).
        El color del ranking ahora se aplica al borde, número y porcentaje.
        """
        ax.axis('off')
        ax.set_facecolor('#f8f9fa')
        
        ax.text(0.5, 0.99, 'RECEPTORES PRIMER PASE', 
                fontsize=9, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes, 
                color='#2c3e50', family='sans-serif')
        
        # 🔥 NUEVA PALETA DE COLORES (debe ser idéntica a la de la otra función)
        position_colors = ['#00BFFF', '#FF1493', '#32CD32', '#FFD700', '#9400D3']
        
        all_receivers = sorted(receiver_stats, key=lambda x: x['count'], reverse=True)[:5]
        
        if not all_receivers:
            ax.text(0.5, 0.60, 'RANKING RECEPTORES\nPRIMER PASE\n\n(Sin datos)', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, fontweight='bold', color='#7f8c8d')
            return
        
        y_pos = 0.80
        
        for i, stat in enumerate(all_receivers):
            # El color del ranking se usa para el borde y el número
            rank_color = position_colors[i % len(position_colors)]
            
            rect_bg = mpatches.FancyBboxPatch((0.01, y_pos - 0.08), 0.98, 0.15,
                                boxstyle="round,pad=0.01", facecolor='white',
                                edgecolor=rank_color, linewidth=2.5, alpha=0.95)
            ax.add_patch(rect_bg)
            
            ax.text(0.08, y_pos, f"#{i+1}", fontsize=16, fontweight='bold', 
                    va='center', ha='center', color=rank_color, transform=ax.transAxes)
            
            player_name = stat.get('nombre_completo', stat['apellido'])
            self.load_player_photos()
            player_photo = self.get_player_photo(player_name)
            
            if player_photo is not None:
                photo_ax = ax.inset_axes([0.12, y_pos - 0.08, 0.20, 0.18])
                photo_ax.imshow(player_photo, aspect='auto')
                photo_ax.axis('off')
            
            # Etiqueta del tipo (C/M/L) en la esquina. El color ahora es gris para no competir.
            tipo_label = stat['tipo'][0]
            ax.text(0.95, y_pos + 0.05, tipo_label,
                    fontsize=9, fontweight='bold', va='center', ha='center',
                    color='#7f8c8d', # 🔥 Color neutro
                    bbox=dict(boxstyle='circle,pad=0.3', facecolor='white',
                            edgecolor='#bdc3c7', linewidth=2),
                    transform=ax.transAxes)
            
            # Nombre del jugador
            nombre = stat['apellido']
            if len(nombre) > 12:
                words = nombre.split()
                if len(words) > 1:
                    ax.text(0.47, y_pos + 0.035, words[0], fontsize=9, fontweight='bold',
                            va='center', ha='left', color='#2c3e50', transform=ax.transAxes)
                    ax.text(0.47, y_pos + 0.01, ' '.join(words[1:]), fontsize=9, fontweight='bold',
                            va='center', ha='left', color='#2c3e50', transform=ax.transAxes)
                else:
                    ax.text(0.47, y_pos + 0.02, nombre[:12], fontsize=9, fontweight='bold',
                            va='center', ha='left', color='#2c3e50', transform=ax.transAxes)
            else:
                ax.text(0.47, y_pos + 0.02, nombre, fontsize=9, fontweight='bold',
                        va='center', ha='left', color='#2c3e50', transform=ax.transAxes)
            
            ax.text(0.29, y_pos + 0.03, f"{stat.get('dorsal', '?')}", 
                    fontsize=12, fontweight='bold',
                    va='center', ha='left', color='black', transform=ax.transAxes)
            
            # 🔥 CAMBIO CLAVE: El color del porcentaje ahora viene de stat['color'], que es el color del ranking
            percentage = stat.get('percentage', 0)
            ax.text(0.44, y_pos - 0.04, f"{percentage:.1f}%",
                    fontsize=11, fontweight='bold', va='center', ha='left', 
                    color=stat['color'], # ¡Ahora es el color del ranking!
                    transform=ax.transAxes)
            
            y_pos -= 0.18
        
        ax.text(0.5, 1, f'TOTAL: {total_first_passes} primeros pases', 
                fontsize=10, fontweight='bold',
                ha='center', va='bottom', color='#2c3e50', transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def draw_legend_zone_flow(self, ax, top_patterns, total_valid_sequences):
        """
        Ranking visual de patrones progresivos con celdas más altas y ajuste de texto automático.
        """
        import textwrap  # 🔥 1. IMPORTAMOS LA LIBRERÍA PARA AJUSTAR TEXTO

        ax.axis('off')
        ax.set_facecolor('#f8f9fa')
        
        ax.text(0.5, 0.99, 'PATRONES PROGRESIVOS', 
                fontsize=9, fontweight='bold', ha='center', va='top', 
                transform=ax.transAxes, color='#2c3e50', family='sans-serif')
        
        position_colors = ['#e74c3c', '#f39c12', '#f1c40f', '#95a5a6', '#bdc3c7']
        pattern_colors = ['#FFD700', '#00BFFF', '#FF69B4', '#95a5a6', '#bdc3c7']
        
        if not top_patterns or len(top_patterns) == 0:
            ax.text(0.5, 0.60, 'RANKING PATRONES\nPROGRESIVOS\n\n(Sin datos)', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, fontweight='bold', color='#7f8c8d')
            return
        
        top_5 = top_patterns[:5]
        
        # 🔥 2. AJUSTAMOS LA ALTURA Y EL ESPACIADO DE LAS CELDAS
        y_pos = 0.90          # Empezamos un poco más arriba
        box_height = 0.20     # Hacemos la caja más alta (antes 0.15)
        y_step = 0.20         # Dejamos más espacio entre cada caja (antes 0.18)

        for i, pattern_data in enumerate(top_5):
            # Usamos las nuevas variables para dibujar la caja de fondo
            rect_bg = mpatches.FancyBboxPatch((0.01, y_pos - box_height), 0.98, box_height,
                                boxstyle="round,pad=0.01", 
                                facecolor='white',
                                edgecolor=position_colors[i],
                                linewidth=2.5, alpha=0.95)
            ax.add_patch(rect_bg)
            
            # Centramos el número de posición en la nueva altura
            ax.text(0.08, y_pos - (box_height / 2), f"#{i+1}", 
                    fontsize=16, fontweight='bold', 
                    va='center', ha='center', 
                    color=position_colors[i],
                    transform=ax.transAxes)
            
            # Patrón legible
            readable = self.pattern_to_readable(pattern_data['pattern'])
            clean_text = readable.strip('()')
            
            # 🔥 3. LÓGICA DE AJUSTE DE TEXTO AUTOMÁTICO
            # `textwrap.wrap` divide el texto en una lista de líneas de longitud máxima `width`
            lines = textwrap.wrap(clean_text, width=28) # Puedes ajustar el '28' si lo ves muy ancho o estrecho
            
            # Dibujamos cada línea, una debajo de la otra
            text_y = y_pos - 0.01  # Posición vertical de la primera línea de texto
            line_spacing = 0.042   # Espacio vertical entre líneas de texto

            for line in lines:
                ax.text(0.20, text_y, line, 
                        fontsize=7, fontweight='bold',
                        va='top', ha='left', color='#2c3e50',
                        transform=ax.transAxes)
                text_y -= line_spacing # Movemos la posición hacia abajo para la siguiente línea

            # 🔥 4. REPOSICIONAMOS EL PORCENTAJE MÁS ABAJO
            percentage = (pattern_data['count'] / total_valid_sequences) * 100 if total_valid_sequences > 0 else 0
            ax.text(0.02, y_pos - box_height + 0.03, # Lo alineamos con la parte inferior de la caja
                    f"{percentage:.1f}%",
                    fontsize=11, fontweight='bold', va='center', ha='left', 
                    color=pattern_colors[i],
                    transform=ax.transAxes)
            
            y_pos -= y_step # Movemos la posición para la siguiente celda
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

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
        
    def draw_most_frequent_first_passes(self, ax, sequences, team_name):
        """
        🔥 LÓGICA FINAL v3: Dibuja flechas de colores. Las flechas para pases largos
        (avg_end_x >= 50) ahora son curvas.
        """
        pitch = VerticalPitch(pitch_type='opta', pitch_color='#2d5a27',
                            line_color='white', linewidth=2, label=False, tick=False)
        pitch.draw(ax=ax)

        if not sequences:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='white')
            return [], 0

        print(f"\n🎯 Analizando primeros pases para {team_name} (Lógica directa con colores unificados)")

        # 1. Recopilar pases (sin cambios)
        first_passes_data = []
        first_pass_indices = [seq[0]['original_index'] for seq in sequences if seq[0]['outcome'] == 1]
        if not first_pass_indices: return [], 0
        first_pass_events = self.df.loc[first_pass_indices]

        for index, pass_event in first_pass_events.iterrows():
            receiver_dorsal, player_name = None, None
            current_team = pass_event['Team Name']
            next_idx = index + 1
            if next_idx in self.df.index:
                next_event = self.df.loc[next_idx]
                if next_event.get('Team Name') == current_team:
                    player_id = next_event.get('playerId')
                    if pd.notna(player_id) and self.player_stats is not None:
                        player_info = self.player_stats[self.player_stats['Player ID'] == player_id]
                        if not player_info.empty:
                            dorsal = player_info.iloc[0].get('Shirt Number')
                            if pd.notna(dorsal):
                                receiver_dorsal = int(dorsal)
                                player_name = player_info.iloc[0].get('Match Name', 'Desconocido')
            
            if receiver_dorsal is not None:
                first_passes_data.append({
                    'start_x': pass_event['x'], 'start_y': pass_event['y'],
                    'end_x': pass_event['Pass End X'], 'end_y': pass_event['Pass End Y'],
                    'receiver_dorsal': receiver_dorsal, 'receiver_name': player_name
                })
        
        total_first_passes = len(first_passes_data)
        if total_first_passes == 0: return [], 0

        # 2. Agrupar y contar (sin cambios)
        from collections import defaultdict
        passes_by_receiver = defaultdict(list)
        for p in first_passes_data:
            passes_by_receiver[p['receiver_dorsal']].append(p)

        receiver_stats = []
        for dorsal, passes in passes_by_receiver.items():
            receiver_stats.append({
                'dorsal': dorsal, 'count': len(passes),
                'percentage': (len(passes) / total_first_passes) * 100,
                'player_name': passes[0]['receiver_name'], 'passes': passes
            })
        
        top_5_receivers = sorted(receiver_stats, key=lambda x: x['count'], reverse=True)[:5]
        
        print(f"\n🏆 TOP 5 RECEPTORES (conteo directo):")
        for recv in top_5_receivers:
            print(f"   #{recv['dorsal']} ({recv['player_name']}): {recv['count']} recepciones ({recv['percentage']:.1f}%)")

        # 3. Paleta de colores (sin cambios)
        position_colors = ['#00BFFF', '#FF1493', '#32CD32', '#FFD700', '#9400D3']
        
        receiver_stats_for_legend = []
        
        # 4. Dibujar flechas con lógica condicional
        for i, recv in enumerate(top_5_receivers):
            passes = recv['passes']
            arrow_color = position_colors[i % len(position_colors)]
            
            avg_start_x = np.mean([p['start_x'] for p in passes])
            avg_start_y = np.mean([p['start_y'] for p in passes])
            avg_end_x = np.mean([p['end_x'] for p in passes])
            avg_end_y = np.mean([p['end_y'] for p in passes])

            # 🔥 --- LÓGICA CONDICIONAL PARA CURVAR LA FLECHA --- 🔥
            if avg_end_x >= 50:
                # PASE LARGO -> Usar FancyArrowPatch para curvar
                # ¡OJO! Hay que invertir las coordenadas para matplotlib puro
                curved_arrow = FancyArrowPatch(
                    (avg_start_y, avg_start_x), 
                    (avg_end_y, avg_end_x),
                    connectionstyle="arc3,rad=0.2",
                    color=arrow_color,
                    linewidth=2.5,
                    arrowstyle='->,head_width=5,head_length=5',
                    alpha=0.9,
                    zorder=15
                )
                ax.add_patch(curved_arrow)
            else:
                # PASE CORTO/MEDIO -> Usar pitch.arrows para flecha recta
                pitch.arrows(avg_start_x, avg_start_y, avg_end_x, avg_end_y,
                            color=arrow_color, width=2.5, headwidth=5, headlength=5,
                            alpha=0.9, zorder=15, ax=ax)
            
            # El resto de la lógica para la leyenda no cambia
            if avg_end_x < 33.3: categoria = 'CORTOS'
            elif avg_end_x < 50.0: categoria = 'MEDIOS'
            else: categoria = 'LARGOS'
            
            if recv['player_name']:
                apellido = recv['player_name'].split()[-1].upper()
                receiver_stats_for_legend.append({
                    'apellido': apellido, 'nombre_completo': recv['player_name'],
                    'tipo': categoria, 'count': recv['count'],
                    'color': arrow_color,
                    'dorsal': recv['dorsal'], 'percentage': recv['percentage']
                })

        ax.set_title(f'RECEPTORES - PRIMER PASE ({total_first_passes} sec.)',
             fontsize=10, fontweight='bold', color='white', pad=8,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#2c3e50', alpha=0.8))


        return receiver_stats_for_legend, total_first_passes

    
    def analyze_pass_patterns(self, sequences):
        """
        Analiza patrones de pases. AHORA recibe secuencias ya pre-filtradas.
        Solo extrae patrones de secuencias 100% exitosas.
        """
        patterns = []
        
        # 'sequences' ya es la lista de secuencias relevantes
        for seq in sequences:
            # 🔥 Mantenemos esta regla para asegurar que el PATRÓN en sí es "perfecto"
            if not all(p.get('outcome') == 1 for p in seq):
                continue

            # Ya no necesitamos más filtros aquí, porque se hicieron antes.
            # Simplemente extraemos el patrón de la secuencia validada.
            pattern = self.extract_pattern(seq)
            if pattern:
                patterns.append(pattern)
        
        # El resto de la función (contar y devolver top 3) no cambia
        from collections import Counter
        pattern_counts = Counter(patterns)
        total = len(patterns)
        
        top_3 = []
        for pattern_str, count in pattern_counts.most_common(3):
            # OJO: el 'percentage' aquí es sobre el total de patrones encontrados,
            # pero el que se muestra en el gráfico usará el denominador correcto.
            percentage = (count / total) * 100 if total > 0 else 0
            top_3.append({
                'pattern': pattern_str,
                'count': count,
                'percentage': percentage
            })
        
        return top_3
    
    def extract_pattern(self, seq):
        """
        🔥 LÓGICA RECONSTRUIDA: Crea una cadena de patrón limpia y espaciada,
        diferenciando claramente los pases intra-zona de las transiciones.
        """
        def get_zone(x):
            if not pd.notna(x): return 0
            if x < 25: return 1
            elif x < 50: return 2
            elif x < 75: return 3
            else: return 4

        if not seq: return ""
        
        pattern_blocks = []
        for i, event in enumerate(seq):
            start_zone = get_zone(event['x'])
            end_x = event.get('end_x') if event.get('action_type') == 'Pass' else event.get('x')
            end_zone = get_zone(end_x)

            # --- Acción DENTRO de la misma zona ---
            if start_zone == end_zone:
                # Si es el primer bloque o el anterior fue una transición, creamos un nuevo contador
                if not pattern_blocks or not pattern_blocks[-1].startswith(f'Z{start_zone}('):
                    pattern_blocks.append(f'Z{start_zone}(1)')
                # Si el último bloque ya era un contador para esta zona, lo incrementamos
                else:
                    count = int(re.search(r'\((\d+)\)', pattern_blocks[-1]).group(1))
                    pattern_blocks[-1] = f'Z{start_zone}({count + 1})'
            
            # --- Acción de TRANSICIÓN entre zonas ---
            else:
                arrow = '=>' if abs(end_zone - start_zone) > 1 else '->'
                aerial_marker = '(A)' if event.get('action_type') == 'Aerial' else ''
                pattern_blocks.append(f'{arrow}Z{end_zone}{aerial_marker}')
        
        return " ".join(pattern_blocks)

    def pattern_to_readable(self, pattern_str):
        """
        🔥 FUNCIÓN REESCRITA: Traduce un patrón a una descripción humana clara,
        usando "+" como separador de acciones.
        """
        import re
        
        if not pattern_str:
            return "()"

        parts = pattern_str.split(' ')
        descriptions = []
        
        # Asumimos que el inicio siempre es desde la Zona 1 (saque de puerta)
        last_zone = 1

        for part in parts:
            desc = ""
            # Caso 1: Pases dentro de una zona (ej: "Z1(2)")
            if part.startswith('Z'):
                zone = int(re.search(r'Z(\d+)', part).group(1))
                n_passes = int(re.search(r'\((\d+)\)', part).group(1))
                plural = "pase" if n_passes == 1 else "pases"
                desc = f"{n_passes} {plural} en zona {zone}"
                last_zone = zone # Actualizamos la última zona conocida

            # Caso 2: Transición a otra zona (ej: "=>Z3(A)")
            elif part.startswith('->') or part.startswith('=>'):
                target_zone = int(re.search(r'Z(\d+)', part).group(1))
                is_aerial = '(A)' in part
                
                prefix = "Pase largo" if '=>' in part else "Pase"
                desc = f"{prefix} de zona {last_zone} a zona {target_zone}"
                
                if is_aerial:
                    desc += " (disputa)"
                
                last_zone = target_zone # Actualizamos la última zona conocida
            
            if desc:
                descriptions.append(desc)
        
        # Unir las descripciones con " + " y capitalizar
        final_text = " + ".join(descriptions)
        if final_text:
            final_text = final_text[0].upper() + final_text[1:]
            
        return f"({final_text})"

    def draw_zone_flow_analysis(self, ax, sequences, team_name, total_valid_sequences=None):
        """Dibuja los 3 patrones más frecuentes de pases por zonas de forma esquemática."""
        pitch = VerticalPitch(pitch_type='opta', pitch_color='#2d5a27',
                            line_color='white', linewidth=2, label=False, tick=False)
        pitch.draw(ax=ax)

        if not sequences:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='white')
            return

        top_patterns = self.analyze_pass_patterns(sequences)
        
        if not top_patterns:
            ax.text(0.5, 0.5, 'Sin patrones detectados', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, color='white')
            return
        
        # 🔥 CALCULAR TOTAL DE SECUENCIAS PROGRESIVAS
        total_progresivas = sum(p['count'] for p in top_patterns)
        
        colors = ['#FFD700', '#00BFFF', '#FF69B4']
        
        for i in [25, 50, 75]:
            ax.plot([0, 100], [i, i], 'white', linewidth=1.5, linestyle='--', alpha=0.5)
        
        zone_labels = ['ZONA 1', 'ZONA 2', 'ZONA 3', 'ZONA 4']
        for i, label in enumerate(zone_labels, 1):
            y_pos = (i * 25) - 12.5
            ax.text(3, y_pos, label, fontsize=7, color='white', 
                fontweight='bold', alpha=0.6, rotation=90, va='center')
        
        start_positions_y = [65, 50, 35]
        
        for idx, pattern_data in enumerate(top_patterns):
            if idx >= len(start_positions_y): break
            color = colors[idx]
            pattern_str = pattern_data['pattern']
            start_y = start_positions_y[idx]
            self.draw_pattern_visualization(ax, pattern_str, start_y, color)
        
        # 🔥 TÍTULO CON NÚMERO DE SECUENCIAS PROGRESIVAS
        ax.set_title(f'PATRONES PROGRESIVOS', 
            fontsize=10, fontweight='bold',
            color='white', pad=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#2c3e50', alpha=0.8))

    
    def draw_pattern_visualization(self, ax, pattern_str, start_y, color):
        """
        🔥 FUNCIÓN REESCRITA: Dibuja un patrón de forma esquemática, SIMULANDO CADA PASE.
        """
        current_x = 6
        current_y = start_y
        
        ax.plot(current_y, current_x, 'o', color='white', markersize=8, 
                markeredgecolor=color, markeredgewidth=2, zorder=20)

        pattern_components = re.findall(r'(Z\d+\(\d+\))|(->Z\d+\(A\)|=>Z\d+\(A\)|->Z\d+|=>Z\d+)', pattern_str)
        
        for component_tuple in pattern_components:
            component = next(item for item in component_tuple if item)
            
            # Caso 1: Pases DENTRO de una zona (ej: "Z1(2)")
            if '(' in component and 'A' not in component:
                zone = int(re.search(r'Z(\d+)', component).group(1))
                num_passes = int(re.search(r'\((\d+)\)', component).group(1))
                
                zone_start_x = (zone - 1) * 25
                zone_end_x = zone * 25
                
                # El punto de partida es donde nos quedamos o el inicio de la zona
                start_x_for_block = max(current_x, zone_start_x)
                
                # Dividir el espacio restante en la zona para dibujar los pases
                available_height = zone_end_x - start_x_for_block
                pass_height = available_height / num_passes
                
                for i in range(num_passes):
                    start_x = start_x_for_block + (i * pass_height)
                    end_x = start_x + pass_height
                    ax.arrow(current_y, start_x, 0, (end_x - start_x) * 0.9, # Flecha un 90% del espacio
                            head_width=2.5, head_length=2.5, fc=color, ec=color, linewidth=1.5, alpha=0.9, zorder=10)
                
                current_x = zone_end_x # Actualizamos la posición al final de la zona

            # Caso 2: TRANSICIÓN a otra zona (ej: "=>Z3(A)")
            elif '->' in component or '=>' in component:
                target_zone = int(re.search(r'Z(\d+)', component).group(1))
                is_aerial = '(A)' in component
                
                # El destino es el INICIO de la nueva zona
                target_x = (target_zone - 1) * 25
                
                # Si el saque es un pase largo directo, que la flecha sea más visible
                if current_x < 10 and '=>' in component:
                    rad_val = 0.3
                    lw = 3
                else:
                    rad_val = 0.2
                    lw = 2.5

                arrow = FancyArrowPatch((current_y, current_x), (current_y, target_x),
                                        connectionstyle=f"arc3,rad={rad_val if '=>' in component else 0}",
                                        color=color, linewidth=lw, 
                                        arrowstyle='->,head_width=5,head_length=5',
                                        linestyle='--' if '=>' in component else '-',
                                        alpha=0.9, zorder=15)
                ax.add_patch(arrow)
                
                if is_aerial:
                    ax.text(current_y + 4, target_x + 5, "DISPUTA", ha='center', va='center',
                            fontsize=7, fontweight='bold', color=color,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, lw=1, alpha=0.8), zorder=20)
                
                current_x = target_x # Actualizamos la posición para el siguiente bloque de acciones
    
    def analyze_goal_kicks_by_week(self):
        """Analiza saques de puerta agrupados por jornada (Week) - INCLUYE OUTCOME 0 y 1"""
        if self.df is None:
            return None
        
        goal_kicks_data = []
        
        print(f"\n🔍 DEBUG analyze_goal_kicks_by_week:")
        print(f"   Total eventos en self.df: {len(self.df)}")
        
        # Buscar usando columna 'Goal Kick'
        for idx in range(len(self.df)):
            current_event = self.df.iloc[idx]
            
            # Verificar si es saque de puerta
            if pd.notna(current_event.get('Goal Kick')) and current_event['Goal Kick'] == 'Sí':
                outcome = int(current_event['outcome']) if pd.notna(current_event['outcome']) else 0
                week = int(current_event['Week']) if pd.notna(current_event.get('Week')) else 0
                
                # Para outcome=0, estimar distancia desde posición inicial
                if outcome == 0 or not pd.notna(current_event.get('Pass End X')):
                    end_x = 20.0
                else:
                    end_x = float(current_event['Pass End X'])
                
                # Clasificar por distancia
                if end_x < 33.3:
                    distance_type = 'CORTO'
                elif end_x < 50.0:
                    distance_type = 'MEDIO'
                else:
                    distance_type = 'LARGO'
                
                goal_kicks_data.append({
                    'Week': week,
                    'Distance': distance_type,
                    'Outcome': outcome,
                    'End_X': end_x
                })
        
        if not goal_kicks_data:
            print("   ⚠️ NO SE ENCONTRARON SAQUES DE PUERTA")
            return None
        
        df_gk = pd.DataFrame(goal_kicks_data)
        
        print(f"\n📊 Total saques de puerta extraídos: {len(df_gk)}")
        print(f"   - Exitosos (outcome=1): {len(df_gk[df_gk['Outcome'] == 1])}")
        print(f"   - Fallidos (outcome=0): {len(df_gk[df_gk['Outcome'] == 0])}")
        
        # 🔥 DEBUG: Ver todas las jornadas encontradas
        all_weeks_found = sorted(df_gk['Week'].unique())
        print(f"\n📅 Jornadas encontradas en TODOS los datos: {all_weeks_found}")
        print(f"   Total jornadas distintas: {len(all_weeks_found)}")
        
        # Contar saques por jornada ANTES de filtrar
        for week in all_weeks_found:
            week_count = len(df_gk[df_gk['Week'] == week])
            print(f"   J{week}: {week_count} saques")
        
        # 🔥 FILTRAR ÚLTIMAS 10 JORNADAS
        all_weeks = sorted(df_gk['Week'].unique())
        if len(all_weeks) > 10:
            last_10_weeks = all_weeks[-10:]
            print(f"\n🔥 FILTRANDO últimas 10 jornadas: {last_10_weeks}")
            df_gk_filtered = df_gk[df_gk['Week'].isin(last_10_weeks)]
            print(f"   Saques DESPUÉS del filtro: {len(df_gk_filtered)}")
            
            # 🔥 Ver cuántos saques quedaron por jornada
            for week in last_10_weeks:
                week_count = len(df_gk_filtered[df_gk_filtered['Week'] == week])
                print(f"   J{week}: {week_count} saques (después de filtrar)")
            
            df_gk = df_gk_filtered
        else:
            print(f"\n✅ Se muestran TODAS las {len(all_weeks)} jornadas (< 10)")
        
        stats_by_week = []
        
        for week in sorted(df_gk['Week'].unique()):
            week_data = df_gk[df_gk['Week'] == week]
            row = {'Week': week}
            
            print(f"\n   Procesando J{week}: {len(week_data)} saques totales")
            
            for dist_type in ['CORTO', 'MEDIO', 'LARGO']:
                dist_data = week_data[week_data['Distance'] == dist_type]
                total = len(dist_data)
                exitosos = len(dist_data[dist_data['Outcome'] == 1])
                pct = (exitosos / total * 100) if total > 0 else 0
                
                row[f'{dist_type}_total'] = total
                row[f'{dist_type}_pct'] = pct
                
                if total > 0:
                    print(f"      {dist_type}: {total} saques ({pct:.0f}% exitosos)")
            
            stats_by_week.append(row)
        
        # Fila de TOTALES (con TODOS los datos, no solo últimas 10)
        df_gk_completo = pd.DataFrame(goal_kicks_data)
        total_row = {'Week': 'TOTAL'}
        for dist_type in ['CORTO', 'MEDIO', 'LARGO']:
            dist_data = df_gk_completo[df_gk_completo['Distance'] == dist_type]
            total = len(dist_data)
            exitosos = len(dist_data[dist_data['Outcome'] == 1])
            pct = (exitosos / total * 100) if total > 0 else 0
            
            total_row[f'{dist_type}_total'] = total
            total_row[f'{dist_type}_pct'] = pct
        
        stats_by_week.append(total_row)
        
        print(f"\n✅ stats_by_week contiene {len(stats_by_week)} filas (incluyendo TOTAL)")
        
        return pd.DataFrame(stats_by_week)

    def draw_goal_kicks_chart(self, ax_bars, ax_pie, stats_df, fig):
        """
        🔥 FUNCIÓN CORREGIDA: Arreglado el error de propiedad 'fw' a 'fontweight'.
        """
        # --- Preparación de datos ---
        if stats_df is None or stats_df.empty or len(stats_df) <= 1:
            for ax in [ax_bars, ax_pie]:
                ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center', fontsize=12, color='gray', transform=ax.transAxes)
                ax.axis('off')
            return

        total_data = stats_df[stats_df['Week'] == 'TOTAL'].iloc[0]
        stats_df_weeks = stats_df[stats_df['Week'] != 'TOTAL'].copy()
        
        opponents = self.get_opponent_by_week(self.team_filter)
        color_corto, color_medio, color_largo = '#3498db', '#9b59b6', '#e67e22'

        # --- GRÁFICO 1: BARRAS HORIZONTALES (Jornadas) ---
        ax_bars.set_facecolor('none')
        ax_bars.patch.set_alpha(0)
        
        weeks, data_cortos, data_medios, data_largos, logos_opponents = [], [], [], [], []
        for idx, row in stats_df_weeks.iterrows():
            week_num = int(row['Week'])
            weeks.append(f"J{week_num}")
            logos_opponents.append(opponents.get(week_num) or None)
            data_cortos.append(int(row['CORTO_total']))
            data_medios.append(int(row['MEDIO_total']))
            data_largos.append(int(row['LARGO_total']))

        totals_per_week = [c + m + l for c, m, l in zip(data_cortos, data_medios, data_largos)]
        if totals_per_week:
            max_total = max(totals_per_week)
            average_total = np.mean(totals_per_week)
            ax_bars.set_xlim(0, max_total * 1.15 if max_total > 0 else 10)
        else:
            average_total = 0
            ax_bars.set_xlim(0, 10)

        y_positions = np.arange(len(weeks))
        ax_bars.set_ylim(-0.4, len(weeks) - 0.4)
        for i, (week, c, m, l) in enumerate(zip(weeks, data_cortos, data_medios, data_largos)):
            y_pos = len(weeks) - 1 - i
            # 🔥 CORRECCIÓN AQUÍ: Cambiado 'fw' por 'fontweight' en las tres líneas
            if c > 0: ax_bars.barh(y_pos, c, 0.7, left=0, color=color_corto, ec='white', lw=1.5); ax_bars.text(c/2, y_pos, f'{c}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
            if m > 0: ax_bars.barh(y_pos, m, 0.7, left=c, color=color_medio, ec='white', lw=1.5); ax_bars.text(c+m/2, y_pos, f'{m}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
            if l > 0: ax_bars.barh(y_pos, l, 0.7, left=c+m, color=color_largo, ec='white', lw=1.5); ax_bars.text(c+m+l/2, y_pos, f'{l}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

        if average_total > 0:
            ax_bars.axvline(x=average_total, color='#c0392b', linestyle='--', linewidth=2, zorder=10)

        ax_bars.set_yticks(y_positions)
        ax_bars.set_yticklabels([weeks[len(weeks)-1-i] for i in range(len(weeks))], fontsize=9, fontweight='bold')
        ax_bars.spines['top'].set_visible(False); ax_bars.spines['right'].set_visible(False)
        ax_bars.tick_params(colors='#2c3e50', labelsize=8)
        ax_bars.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, opponent in enumerate(logos_opponents):
            if opponent:
                logo = self.load_team_logo(opponent, target_size=(35, 35))
                if logo is not None:
                    ab = AnnotationBbox(OffsetImage(logo, zoom=0.5), (-0.19, len(weeks) - 1 - i), xycoords=('axes fraction', 'data'), frameon=False)
                    ax_bars.add_artist(ab)

        # --- GRÁFICO 2: CIRCULAR (Total) ---
        ax_pie.set_facecolor('none')
        ax_pie.patch.set_alpha(0)

        pie_data = [total_data['CORTO_total'], total_data['MEDIO_total'], total_data['LARGO_total']]
        pie_labels = ['Cortos', 'Medios', 'Largos']
        pie_colors = [color_corto, color_medio, color_largo]
        
        non_zero_data = [(data, label) for data, label in zip(pie_data, pie_labels) if data > 0]
        if non_zero_data:
            final_pie_data, final_pie_labels = zip(*non_zero_data)
            # Hacemos el radio un poco más pequeño para que no toque los bordes
            ax_pie.pie(final_pie_data, labels=final_pie_labels, colors=[pie_colors[pie_labels.index(l)] for l in final_pie_labels],
                    autopct='%1.1f%%', startangle=70, radius=0.8, # 🔥 Radio más pequeño
                    textprops={'color': 'black', 'fontweight': 'bold', 'fontsize': 8},
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    
    def load_player_photos(self):
        if self.photos_data is None:
            try:
                with open('assets/jugadores_optimizados.json', 'r', encoding='utf-8') as f:
                    self.photos_data = json.load(f)
            except FileNotFoundError:
                print("⚠️ No se encontró el archivo jugadores_optimizados.json")
                self.photos_data = []
        return self.photos_data

    def extract_names_parts(self, name):
        """Extrae las partes de un nombre normalizado"""
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
        
        return {'full': normalized, 'first_name': first_name, 
                'last_name': last_name, 'all_parts': parts}

    def match_player_name(self, player_name, photos_data, team_filter=None):
        """Encuentra el jugador más parecido en las fotos"""
        
        player_parts = self.extract_names_parts(player_name)
        if not player_parts['full']:
            return None

        # Filtrar por equipo si se proporciona
        team_players = []
        if team_filter:
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
                # Fallback por similitud
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

        # Buscar por palabras más largas primero
        player_words = [w for w in player_parts['all_parts'] if len(w) >= 3]
        player_words_sorted = sorted(player_words, key=len, reverse=True)
        
        for palabra_buscar in player_words_sorted:
            for photo_entry in team_players:
                photo_name = photo_entry.get('player_name', '')
                photo_parts = self.extract_names_parts(photo_name)
                photo_words = [w for w in photo_parts['all_parts'] if len(w) >= 3]
                
                if palabra_buscar in photo_words:
                    return photo_entry
                
                # Tolerancia para palabras largas
                if len(palabra_buscar) > 5:
                    for ph_word in photo_words:
                        if len(ph_word) > 5:
                            distance = self.levenshtein_distance(palabra_buscar, ph_word)
                            if distance == 1:
                                return photo_entry

        # Fallback: scoring
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
        
        if not candidates:
            return None
        elif len(candidates) == 1:
            return candidates[0]['entry']
        else:
            best_candidates = sorted(candidates, key=lambda x: x['match_count'], reverse=True)
            if best_candidates[0]['match_count'] > best_candidates[1]['match_count']:
                return best_candidates[0]['entry']
            return best_candidates[0]['entry']

    def levenshtein_distance(self, s1, s2):
        """Calcula la distancia de Levenshtein"""
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

    def get_player_photo(self, player_name):
        """Obtiene la foto del jugador con el sistema robusto"""
        if self.photos_data is None: 
            self.load_player_photos()
        if not self.photos_data: 
            print("⚠️ No hay datos de fotos cargados")
            return None
        
        print(f"   🔍 Buscando foto para: {player_name}")
        match = self.match_player_name(player_name, self.photos_data, self.team_filter)
        
        if not match:
            print(f"   ❌ No se encontró match para {player_name}")
            return None
        
        print(f"   ✅ Match encontrado: {match.get('player_name', 'N/A')}")
        
        try:
            import base64
            from io import BytesIO
            from PIL import Image
            
            img_data = base64.b64decode(match['image_base64'])
            img = Image.open(BytesIO(img_data)).convert('RGBA')
            data = np.array(img)
            
            # 🔥 FLOOD FILL para eliminar fondo blanco
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
            
            print(f"   ✅ Foto procesada correctamente")
            return data.astype(np.float32) / 255.0
        
        except Exception as e:
            print(f"   ⚠️ Error procesando foto de {player_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_receiver_info_from_pass(self, pass_event_row):
        """
        Encuentra al receptor buscando el SIGUIENTE evento del mismo equipo,
        sin importar el tipo de evento ni su outcome.
        """
        # Detalles del pase original
        match_id = pass_event_row['Match ID']
        pass_time = pass_event_row['timeStamp']
        team_name = pass_event_row['Team Name']
        pass_idx = pass_event_row.name

        # Definir la ventana de búsqueda
        time_limit = pass_time + timedelta(seconds=5)

        # Eventos candidatos que ocurren DESPUÉS del pase
        candidate_events = self.df[
            (self.df['Match ID'] == match_id) &
            (self.df.index > pass_idx) &
            (self.df['timeStamp'] <= time_limit) &
            (self.df['Team Name'] == team_name)  # 🔥 Mismo equipo
        ]

        if candidate_events.empty:
            return None, None

        # 🔥 SIMPLEMENTE EL PRIMER EVENTO DEL MISMO EQUIPO
        next_event = candidate_events.iloc[0]
        player_id = next_event['playerId']
        
        # Usar player_stats para obtener el dorsal
        if self.player_stats is not None:
            player_info = self.player_stats[self.player_stats['Player ID'] == player_id]
            if not player_info.empty:
                shirt_number = player_info['Shirt Number'].iloc[0]
                player_name = player_info['Match Name'].iloc[0]
                if pd.notna(shirt_number):
                    return str(int(shirt_number)), player_name
        
        return None, None

    def get_player_name_by_shirt(self, shirt_number, team_name):
        """
        Obtiene el nombre del jugador dado su dorsal y equipo.
        """
        if self.player_stats is None:
            return None
        
        try:
            shirt_int = int(shirt_number)
            
            # Buscar en player_stats
            player_match = self.player_stats[
                (self.player_stats['Team Name'] == team_name) &
                (self.player_stats['Shirt Number'] == shirt_int)
            ]
            
            if not player_match.empty:
                # Retornar el primer nombre encontrado
                return player_match['Match Name'].iloc[0]
        except Exception as e:
            print(f"⚠️ Error buscando nombre para dorsal {shirt_number}: {e}")
        
        return None
    
    def draw_sequence_panel(self, ax, seq_data, color):
        """Dibuja un panel de secuencia compacto - Distingue Pass y Take On"""
        ax.set_facecolor('#f0f0f0')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        if not seq_data or 'all_sequences' not in seq_data:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                    fontsize=9, color='grey')
            return
        
        count = seq_data['count']
        all_seqs = seq_data['all_sequences']
        
        # Título con contador
        ax.text(0.5, 0.95, f'{count} repeticiones', 
                ha='center', va='top',
                fontsize=8, fontweight='bold', color='#2c3e50')
        
        from matplotlib.patches import Rectangle
        
        # Fondo del campo (mini)
        campo = Rectangle((0.1, 0.1), 0.8, 0.75, 
                        facecolor='#2d5a27', edgecolor='white', linewidth=2)
        ax.add_patch(campo)
        
        # Línea de medio campo
        ax.plot([0.1, 0.9], [0.475, 0.475], 'white', linewidth=1.5)
        
        # Dibujar TODAS las secuencias
        for sequence in all_seqs:
            for i, pass_data in enumerate(sequence):
                # Normalizar coordenadas
                x_start_norm = 0.1 + (pass_data['x'] / 100.0) * 0.8
                y_start_norm = 0.1 + (pass_data['y'] / 100.0) * 0.75
                x_end_norm = 0.1 + (pass_data['end_x'] / 100.0) * 0.8
                y_end_norm = 0.1 + (pass_data['end_y'] / 100.0) * 0.75
                
                # 🔥 DISTINGUIR ENTRE PASS Y TAKE ON
                if pass_data.get('action_type') == 'Take On':
                    # Take On: línea punteada
                    ax.plot([y_start_norm, y_end_norm], 
                        [x_start_norm, x_end_norm],
                        linestyle='--', linewidth=1.5, 
                        color=color, alpha=0.4)
                    # Marcador
                    ax.plot(y_end_norm, x_end_norm, 'o', 
                        markersize=4, color=color, alpha=0.5)
                else:
                    # Pass: flecha normal
                    ax.annotate('', 
                            xy=(y_end_norm, x_start_norm),
                            xytext=(y_start_norm, x_start_norm),
                            arrowprops=dict(arrowstyle='->', 
                                            lw=1.5, color=color, alpha=0.4))
        
        # Trayectoria promedio
        avg_path = self.calculate_average_path(all_seqs)
        if avg_path and len(avg_path) >= 2:
            for i in range(len(avg_path) - 1):
                x1_norm = 0.1 + (avg_path[i][1] / 100.0) * 0.8
                y1_norm = 0.1 + (avg_path[i][0] / 100.0) * 0.75
                x2_norm = 0.1 + (avg_path[i+1][1] / 100.0) * 0.8
                y2_norm = 0.1 + (avg_path[i+1][0] / 100.0) * 0.75
                
                ax.annotate('',
                        xy=(y2_norm, x1_norm),
                        xytext=(y1_norm, x1_norm),
                        arrowprops=dict(arrowstyle='->', 
                                        lw=3, color=color, alpha=1.0))
            
            # Punto de inicio
            x_start = 0.1 + (avg_path[0][1] / 100.0) * 0.8
            y_start = 0.1 + (avg_path[0][0] / 100.0) * 0.75
            ax.plot(y_start, x_start, 'o', 
                    color='white', markersize=8, 
                    markeredgecolor=color, markeredgewidth=2, zorder=10)
    
    def get_player_shirt_from_pass(self, pass_data):
        """Extrae el dorsal de un pase (si está disponible)"""
        # Intentar obtener el dorsal del diccionario
        if 'shirt_number' in pass_data:
            return pass_data.get('shirt_number')
        
        # Si no está, intentar inferirlo (necesitarías player_stats cargado)
        return None
    
    def calculate_action_value(self, event):
        """Calcula un valor estimado para la acción (estilo VAEP simplificado)"""
        value = 0.0
        
        if event['Event Name'] == 'Pass':
            value = 0.02
            # Bonus por pase progresivo
            if pd.notna(event.get('Pass End X')) and pd.notna(event.get('x')):
                try:
                    progression = float(event['Pass End X']) - float(event['x'])  # ✅ CON float()
                    if progression > 20:
                        value += 0.03
                    elif progression > 10:
                        value += 0.01
                except (ValueError, TypeError):
                    pass
        
        elif event['Event Name'] == 'Take On':
            value = 0.05
        
        else:
            value = 0.01
        
        # Penalizar si falla
        try:
            if int(event['outcome']) == 0:
                value *= 0.2
        except (ValueError, TypeError):
            pass
        
        return round(value, 4)
    
    def load_background(self): 
        return plt.imread("assets/fondo_informes.png") if os.path.exists("assets/fondo_informes.png") else None

    # 🔥 NUEVO: Función para cargar el logo de Tactic
    def load_tactic_logo(self):
        """Carga el logo de tactic_logo.png desde la carpeta assets."""
        logo_path = "assets/tactic_logo.png"
        if os.path.exists(logo_path):
            return plt.imread(logo_path)
        else:
            print(f"⚠️  Logo no encontrado en: {logo_path}")
            return None

    def load_ball_image(self): 
        return plt.imread("assets/tactic_logo.png") if os.path.exists("assets/tactic_logo.png") else None


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
        
        print(f"🔍 Buscando escudo para '{equipo}'")
        print(f"   Palabras a buscar (orden): {palabras_ordenadas}")
        
        # Obtener todos los archivos disponibles
        all_files = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
        
        # Buscar por cada palabra en orden de longitud
        for palabra_buscar in palabras_ordenadas:
            print(f"   → Buscando con: '{palabra_buscar}'")
            
            for filename in all_files:
                nombre_archivo = os.path.splitext(filename)[0]
                nombre_archivo_norm = normalize_word(nombre_archivo)
                
                # Coincidencia exacta de la palabra en el nombre del archivo
                if palabra_buscar == nombre_archivo_norm or palabra_buscar in nombre_archivo_norm:
                    logo_path = f"assets/escudos/{filename}"
                    print(f"   ✅ Encontrado: {filename}")
                    
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
            print(f"   ✅ Encontrado por similitud ({best_score:.2f}): {os.path.basename(best_match_path)}")
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
    
    def load_data(self, team_filter=None):
        """Carga los datos y filtra por equipo si es necesario"""
        try:
            # 🔥 CARGAR DATAFRAME COMPLETO PRIMERO
            self.df_complete = pd.read_parquet(self.data_path)
            
            if team_filter:
                # Filtrar para el equipo
                self.df = self.df_complete[self.df_complete['Team Name'] == team_filter].copy()
                self.team_filter = team_filter
            else:
                self.df = self.df_complete.copy()
            
            # Convertir coordenadas a tipo numérico
            for col in ['x', 'y', 'Pass End X', 'Pass End Y']:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df_complete[col] = pd.to_numeric(self.df_complete[col], errors='coerce')
            
            # Convertir timestamp a datetime
            self.df['timeStamp'] = pd.to_datetime(self.df['timeStamp'], format='ISO8601')
            self.df_complete['timeStamp'] = pd.to_datetime(self.df_complete['timeStamp'], format='ISO8601')
            
            # Ordenar por Match ID y timestamp
            self.df = self.df.sort_values(['Match ID', 'timeStamp']).reset_index(drop=True)
            self.df_complete = self.df_complete.sort_values(['Match ID', 'timeStamp']).reset_index(drop=True)
            
            print(f"Datos cargados: {len(self.df)} eventos")
            return True
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            return False
    
    def calculate_average_path(self, sequences):
        """Calcula la trayectoria promedio de un conjunto de secuencias"""
        if not sequences:
            return None
        
        # Encontrar longitud mínima
        min_len = min(len(seq) for seq in sequences)
        
        avg_path = []
        for i in range(min_len + 1):  # +1 para incluir punto inicial
            x_coords = []
            y_coords = []
            
            for seq in sequences:
                if i == 0:
                    # Punto inicial
                    x_coords.append(seq[0]['y'])  # Intercambiado para VerticalPitch
                    y_coords.append(seq[0]['x'])
                else:
                    # Puntos finales de pases
                    x_coords.append(seq[i-1]['end_y'])
                    y_coords.append(seq[i-1]['end_x'])
            
            avg_x = np.mean(x_coords)
            avg_y = np.mean(y_coords)
            avg_path.append((avg_x, avg_y))
        
        return avg_path
    
    def draw_pass_type_analysis(self, ax, sequences, tipo, min_dist, max_dist):
        """Dibuja análisis de pases por distancia"""
        pitch = VerticalPitch(pitch_type='opta', pitch_color='#2d5a27',
                            line_color='white', linewidth=2, label=False, tick=False)
        pitch.draw(ax=ax)
        
        # Filtrar primeros pases según distancia
        first_passes = []
        receivers = []
        
        for seq in sequences:
            first_pass = seq[0]
            end_x = first_pass['end_x']
            
            if min_dist <= end_x < max_dist:
                first_passes.append(first_pass)
                
                # Buscar receptor (segundo pase si existe)
                if len(seq) > 1:
                    receivers.append({
                        'x': seq[1]['y'],  # Intercambiado
                        'y': seq[1]['x'],
                        'dorsal': self.get_player_shirt_from_pass(seq[1])
                    })
        
        # Dibujar pases
        for fp in first_passes:
            x_start, y_start = fp['y'], fp['x']
            x_end, y_end = fp['end_y'], fp['end_x']
            ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                    arrowprops=dict(arrowstyle='->', lw=2, 
                                    color='yellow', alpha=0.4))
        
        # 🔥 TOP 2 RECEPTORES MÁS FRECUENTES
        from collections import Counter
        dorsal_counts = Counter([r['dorsal'] for r in receivers if r['dorsal']])
        top_2 = dorsal_counts.most_common(2)
        
        # Marcar dorsales
        for dorsal, count in top_2:
            # Posición promedio donde recibe
            positions = [(r['x'], r['y']) for r in receivers if r['dorsal'] == dorsal]
            avg_x = np.mean([p[0] for p in positions])
            avg_y = np.mean([p[1] for p in positions])
            
            # Círculo y dorsal
            pitch.scatter(avg_x, avg_y, s=800, color='red', 
                        edgecolors='white', linewidth=3, alpha=0.9, ax=ax, zorder=10)
            pitch.annotate(str(dorsal), xy=(avg_x, avg_y), 
                        c='white', ha='center', va='center',
                        size=14, weight='bold', ax=ax, zorder=11)
        
        ax.set_title(f'PASES {tipo} ({len(first_passes)} total)',
                    fontsize=10, fontweight='bold', color='white', pad=8,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#2c3e50', alpha=0.8))
    
    def draw_sequences_column(self, fig, gs, top_sequences, colors):
        """Dibuja secuencias más repetidas estilo tactic2"""
        # Título de la columna
        ax_titulo = fig.add_axes([0.70, 0.89, 0.27, 0.03])
        ax_titulo.axis('off')
        ax_titulo.text(0.5, 0.5, 'SECUENCIAS MÁS REPETIDAS',
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white', bbox=dict(boxstyle='round,pad=0.5',
                    facecolor='#2c3e50', alpha=0.9))
        
        for i, seq_data in enumerate(top_sequences):
            ax = fig.add_subplot(gs[i, 2])
            self.draw_sequence_panel(ax, seq_data, colors[i])
    
    def guardar_sin_espacios(self, fig, filename):
        """Guarda sin espacios manteniendo landscape A4"""
        fig.set_size_inches(11.69, 8.27)
        fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0,
                   facecolor='white', format='pdf', orientation='landscape')
        print(f"✅ Archivo guardado: {filename}")

    def get_zone_center(self, zone_number):
        """Devuelve las coordenadas (y, x) del centro de una zona para el gráfico."""
        # Mapeo de zona -> (centro_y, centro_x)
        # Estos valores son aproximados, puedes ajustarlos para que se vean mejor
        zone_centers = {
            1: (50, 6), 2: (29, 6), 3: (71, 6), 4: (10.5, 6), 5: (89.5, 6),
            6: (29, 17), 7: (71, 17), 8: (10.5, 17), 9: (89.5, 17), 10: (50, 17),
            11: (89.5, 28.75), 12: (57.5, 28.75), 13: (18.4, 28.75),
            14: (89.5, 42.25), 15: (50, 42.25), 16: (10.5, 42.25),
            17: (89.5, 62.5), 18: (50, 62.5), 19: (10.5, 62.5),
            20: (89.5, 87.5), 21: (50, 87.5), 22: (10.5, 87.5)
        }
        # ¡IMPORTANTE! Devolvemos (y, x) porque VerticalPitch los usa invertidos
        return zone_centers.get(zone_number, (50, 50))
    
    def draw_zone_sequence_patterns(self, ax, top_5_patterns, total_valid_sequences):
        """
        Dibuja el campograma de patrones de zona con flechas finales claramente visibles.
        """
        pitch = VerticalPitch(pitch_type='opta', pitch_color='#2d5a27',
                            line_color='white', linewidth=2, label=False, tick=False)
        pitch.draw(ax=ax)
        
        # Dibujo de las divisiones y los números de zona (sin cambios)
        for y_coord in [21.1, 36.8, 63.2, 78.9]:
            ax.plot([y_coord, y_coord], [0, 23], color='yellow', linestyle='--', linewidth=1, alpha=0.4)
        for y_coord in [33.3, 66.6]:
            ax.plot([y_coord, y_coord], [23, 100], color='yellow', linestyle='--', linewidth=1, alpha=0.4)
        for x_coord in [11.5, 23, 34.5, 50, 75]:
            ax.plot([0, 100], [x_coord, x_coord], color='yellow', linestyle='--', linewidth=1, alpha=0.4)
        for zone_num in range(1, 23):
            center_y, center_x = self.get_zone_center(zone_num)
            ax.text(center_y, center_x, str(zone_num), ha='center', va='center', fontsize=22, 
                    color='white', fontweight='bold', alpha=0.2, zorder=3)

        # 🔥 CALCULAR TOTALES POR TIPO (de los patrones que se están mostrando)
        total_1_2_zonas = 0
        total_mas_2_zonas = 0
        total_progresion = 0
        
        for pattern in top_5_patterns:
            seccion = pattern.get('seccion')
            if seccion == 1:
                # Solo contar una vez por sección (evitar duplicados)
                if total_1_2_zonas == 0:
                    total_1_2_zonas = pattern.get('total_seccion', 0)
            elif seccion == 2:
                if total_mas_2_zonas == 0:
                    total_mas_2_zonas = pattern.get('total_seccion', 0)
            elif seccion == 3:
                if total_progresion == 0:
                    total_progresion = pattern.get('total_seccion', 0)

        max_percentage = 0
        if total_valid_sequences > 0 and top_5_patterns:
            max_percentage = (top_5_patterns[0]['count'] / total_valid_sequences) * 100

        min_width, max_width = 1.5, 4.0
        
        for i, pattern_data in enumerate(top_5_patterns):
            sequence = pattern_data['sequence']
            count = pattern_data['count']
            color = pattern_data.get('color', '#f9c80e')

            percentage = (count / total_valid_sequences) * 100 if total_valid_sequences > 0 else 0
            dynamic_width = min_width + (percentage / max_percentage) * (max_width - min_width) if max_percentage > 0 else min_width
            
            path_x, path_y = [], []
            offset = (i * 4) - 8

            for zone in sequence:
                center_y, center_x = self.get_zone_center(zone)
                path_x.append(center_x)
                path_y.append(center_y + offset)

            # CASO ESPECIAL: Secuencia de una zona a sí misma (ej: (1, 1))
            if len(sequence) == 2 and sequence[0] == sequence[1]:
                raw_sequences = pattern_data['raw_sequences']
                
                avg_start_x = np.mean([seq[0]['x'] for seq in raw_sequences])
                avg_start_y = np.mean([seq[0]['y'] for seq in raw_sequences])
                avg_end_x = np.mean([seq[0]['end_x'] for seq in raw_sequences])
                avg_end_y = np.mean([seq[0]['end_y'] for seq in raw_sequences])
                
                pitch.arrows(avg_start_x, avg_start_y, avg_end_x, avg_end_y,
                            color=color, width=dynamic_width, headwidth=dynamic_width*1, headlength=dynamic_width*2,
                            alpha=0.9, zorder=15 + i, ax=ax)

            # CASO NORMAL: Secuencia esquemática entre zonas diferentes
            elif len(path_x) >= 2:
                ax.plot(path_y[:-1], path_x[:-1], color=color, linewidth=dynamic_width,
                        alpha=0.9, zorder=10 + i, solid_capstyle='round')

                arrow = FancyArrowPatch(
                    (path_y[-2], path_x[-2]), (path_y[-1], path_x[-1]),
                    color=color, linewidth=dynamic_width, mutation_scale=dynamic_width * 6,
                    arrowstyle='->', zorder=11 + i, alpha=0.9
                )
                ax.add_patch(arrow)
            
            elif len(path_x) == 1:
                ax.plot(path_y, path_x, 'o', color=color, markersize=dynamic_width * 3,
                        markeredgecolor='white', markeredgewidth=1.5)

        # 🔥 TÍTULO CON DESGLOSE DE TIPOS
        titulo_parts = []
        if total_1_2_zonas > 0:
            titulo_parts.append(f'1-2Z: {total_1_2_zonas}')
        if total_mas_2_zonas > 0:
            titulo_parts.append(f'+2Z: {total_mas_2_zonas}')
        if total_progresion > 0:
            titulo_parts.append(f'Prog: {total_progresion}')
        
        titulo_completo = 'PATRONES POR ZONAS'
        
        ax.set_title(titulo_completo, 
            fontsize=10, fontweight='bold', color='white', pad=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#2c3e50', alpha=0.8))
        
    def print_sequence_details(self, sequences, n_examples=4):
        """Imprime detalles de N secuencias de ejemplo"""
        print(f"\n{'='*80}")
        print(f"EJEMPLOS DE SECUENCIAS (mostrando {min(n_examples, len(sequences))} de {len(sequences)} totales)")
        print(f"{'-'*80}\n")
        
        for seq_num, sequence in enumerate(sequences[:n_examples], 1):
            print(f"SECUENCIA #{seq_num} ({len(sequence)} pases)")
            print(f"{'-'*80}")
            
            for i, pass_data in enumerate(sequence, 1):
                print(f"  Pase {i}:")
                print(f"    Inicio:    x={pass_data['x']:.2f}, y={pass_data['y']:.2f}")
                print(f"    Destino:   x={pass_data['end_x']:.2f}, y={pass_data['end_y']:.2f}")
                print(f"    Event:     Pass")
                print(f"    Equipo:    {pass_data.get('team_name', 'N/A')}")
                print(f"    Outcome:   {'Exitoso' if pass_data['outcome'] == 1 else 'Fallido'}")
                
                # Calcular distancia
                dist = np.sqrt((pass_data['end_x'] - pass_data['x'])**2 + 
                            (pass_data['end_y'] - pass_data['y'])**2)
                print(f"    Distancia: {dist:.2f} unidades")
                print()
            
            print()
        
    def extract_goal_kick_sequences(self, team_name):
        """Extrae todas las secuencias que comienzan con un saque de puerta"""
        if self.df is None:
            print("No hay datos cargados")
            return []
        
        sequences = []
        
        # Filtrar solo eventos del equipo
        team_df = self.df[self.df['Team Name'] == team_name].copy()
        
        # 🔥 NUEVO: Encontrar saques de puerta usando la columna 'Goal Kick'
        goal_kicks = []
        
        for idx in range(len(team_df)):
            current_event = team_df.iloc[idx]
            
            # Verificar si el evento tiene Goal Kick == 'Sí'
            if pd.notna(current_event.get('Goal Kick')) and current_event['Goal Kick'] == 'Sí':
                goal_kicks.append(team_df.index[idx])
        
        print(f"Encontrados {len(goal_kicks)} saques de puerta (Goal Kick = Sí)")
        
        for gk_idx in goal_kicks:
            sequence = self.extract_sequence_from_goal_kick(gk_idx, team_name)
            if sequence and len(sequence) >= 2:  # Al menos 2 pases
                sequences.append(sequence)
        
        print(f"Extraidas {len(sequences)} secuencias validas")
        return sequences
    
    def extract_sequence_from_goal_kick(self, start_idx, team_name):
        """
        Extrae una secuencia de Pass, Aerial, Take On y otros eventos del mismo equipo.
        La secuencia termina si cambia de equipo o pasan 15 segundos.
        """
        pass_chain = []
        
        # Obtener el evento inicial (saque de puerta)
        start_event = self.df.loc[start_idx]
        match_id = start_event['Match ID']
        start_time = start_event['timeStamp']
        time_limit = start_time + timedelta(seconds=15)
        
        # --- PRIMER PASE (SAQUE DE PUERTA) ---
        first_pass_end_x = float(start_event['Pass End X'])
        first_pass_end_y = float(start_event['Pass End Y'])
        
        pass_chain.append({
            'x': float(start_event['x']),
            'y': float(start_event['y']),
            'end_x': first_pass_end_x,
            'end_y': first_pass_end_y,
            'outcome': start_event['outcome'],
            'team_name': team_name,
            'action_type': 'Pass',
            'value': self.calculate_action_value(start_event),
            'original_index': start_idx
        })
        
        # Guardamos las coordenadas finales para encadenar
        last_end_x = first_pass_end_x
        last_end_y = first_pass_end_y
        
        # Empezar a buscar desde el siguiente evento
        current_idx = start_idx + 1
        
        while current_idx < len(self.df):
            event = self.df.iloc[current_idx]
            
            # --- CONDICIONES DE PARADA ---
            if len(pass_chain) >= 5 or \
            event['Match ID'] != match_id or \
            event['timeStamp'] > time_limit or \
            event['Team Name'] != team_name:
                break
            
            # 🔥 INCLUIR Pass, Aerial, Take On y otros eventos relevantes
            if event['Event Name'] in ['Pass', 'Take On', 'Aerial']:
                
                # Inicio: donde terminó la acción anterior
                current_action_start_x = last_end_x
                current_action_start_y = last_end_y
                
                # Destino según tipo de evento
                if event['Event Name'] == 'Pass':
                    if pd.notna(event.get('Pass End X')) and pd.notna(event.get('Pass End Y')):
                        current_action_end_x = float(event['Pass End X'])
                        current_action_end_y = float(event['Pass End Y'])
                    else:
                        current_idx += 1
                        continue
                
                elif event['Event Name'] == 'Aerial':
                    # Para Aerial, usar la posición del evento
                    if pd.notna(event.get('x')) and pd.notna(event.get('y')):
                        current_action_end_x = float(event['x'])
                        current_action_end_y = float(event['y'])
                    else:
                        current_idx += 1
                        continue
                
                elif event['Event Name'] == 'Take On':
                    # Para Take On, estimar avance
                    if pd.notna(event.get('x')) and pd.notna(event.get('y')):
                        current_action_end_x = float(event['x']) + 2
                        current_action_end_y = float(event['y'])
                    else:
                        current_idx += 1
                        continue
                
                # Añadir acción a la cadena
                pass_chain.append({
                    'x': current_action_start_x,
                    'y': current_action_start_y,
                    'end_x': current_action_end_x,
                    'end_y': current_action_end_y,
                    'outcome': event['outcome'],
                    'team_name': team_name,
                    'action_type': event['Event Name'],
                    'value': self.calculate_action_value(event)
                })
                
                # Actualizar última posición conocida
                last_end_x = current_action_end_x
                last_end_y = current_action_end_y
            
            # Avanzar al siguiente evento
            current_idx += 1
        
        return pass_chain
    
    def clasificar_secuencias_especiales(self, top_patterns, total_sequences):
        """
        Clasifica secuencias en 3 secciones:
        SECCIÓN 1: Secuencias de 1 o 2 zonas
        - 1 de juego directo (zona final ≥14) - solo puede ser 2 zonas
        - 1 de juego corto (zona final <14) - puede ser 1 zona (zona 1) o 2 zonas
        SECCIÓN 2: Secuencias con más de 2 zonas
        - Top 2 con mayor porcentaje
        SECCIÓN 3: Secuencias con progresión a zona ≥13
        - Top 2 con mayor porcentaje (más de 2 pases)
        """
        # Clasificar por tipo
        dos_zonas_directo = []  # 1-2 zonas, zona final ≥14
        dos_zonas_corto = []    # 1-2 zonas, zona final <14
        mas_dos_zonas = []      # Más de 2 zonas
        progresion_alta = []    # Más de 2 pases, zona final ≥13
        
        for pattern in top_patterns:
            sequence = pattern['sequence']
            n_zonas = len(sequence)
            zona_final = sequence[-1]
            
            # SECCIÓN 1: Secuencias de 1 o 2 zonas
            # 🔥 CAMBIO: Ahora acepta n_zonas == 1 (pase en zona 1) o n_zonas == 2
            if n_zonas <= 2 and sequence[0] == 1:
                if zona_final >= 14:
                    dos_zonas_directo.append(pattern)
                else:
                    # Juego corto: incluye zona 1 a zona 1 (n_zonas==1 o ==2)
                    dos_zonas_corto.append(pattern)
            
            # SECCIÓN 2: Más de 2 zonas
            elif n_zonas > 2:
                mas_dos_zonas.append(pattern)
                
                # SECCIÓN 3: Si además supera zona 13, también va a progresión
                if zona_final >= 14:
                    progresion_alta.append(pattern)
        
        # Construir resultado ordenado
        resultado = []
        
        # === SECCIÓN 1: UNA O DOS ZONAS ===
        # Top 1 juego directo (verde)
        if dos_zonas_directo:
            mejor_directo = sorted(dos_zonas_directo, key=lambda x: x['count'], reverse=True)[0]
            resultado.append({
                **mejor_directo, 
                'categoria': 'dos_zonas_directo', 
                'color': '#27ae60',
                'seccion': 1
            })
        
        # Top 1 juego corto (azul)
        if dos_zonas_corto:
            mejor_corto = sorted(dos_zonas_corto, key=lambda x: x['count'], reverse=True)[0]
            resultado.append({
                **mejor_corto, 
                'categoria': 'dos_zonas_corto', 
                'color': '#3498db',
                'seccion': 1
            })
        
        # === SECCIÓN 2: MÁS DE 2 ZONAS ===
        # Top 2 (naranja)
        for p in sorted(mas_dos_zonas, key=lambda x: x['count'], reverse=True)[:2]:
            resultado.append({
                **p, 
                'categoria': 'mas_dos_zonas', 
                'color': '#e67e22',
                'seccion': 2
            })
        
        # === SECCIÓN 3: PROGRESIÓN ALTA (≥zona 13) ===
        # Top 2 (morado)
        for p in sorted(progresion_alta, key=lambda x: x['count'], reverse=True)[:2]:
            resultado.append({
                **p, 
                'categoria': 'progresion_alta', 
                'color': '#8e44ad',
                'seccion': 3
            })
        
        return resultado
    
    def calculate_sequence_pattern(self, sequence):
        """Calcula un patron basado en ZONA del primer pase y direccion general"""
        if not sequence:
            return tuple()
        
        first_pass = sequence[0]
        
        # ZONA donde CAE el primer pase (Pass End X, Pass End Y)
        end_x = first_pass['end_x']
        end_y = first_pass['end_y']
        
        # Dividir campo en 9 zonas (3x3)
        if end_x < 33.3:
            zone_x = 'IZQ'
        elif end_x < 66.6:
            zone_x = 'CEN'
        else:
            zone_x = 'DER'
        
        if end_y < 33.3:
            zone_y = 'DEF'
        elif end_y < 66.6:
            zone_y = 'MED'
        else:
            zone_y = 'ATK'
        
        first_pass_zone = f"{zone_x}_{zone_y}"
        
        # Direccion general de avance (si hay mas pases)
        if len(sequence) > 1:
            # Comparar primer y ultimo pase
            last_pass = sequence[-1]
            dy = last_pass['end_y'] - first_pass['y']
            dx = last_pass['end_x'] - first_pass['x']
            
            # Direccion predominante
            if abs(dy) > abs(dx) * 1.5:  # Mas vertical
                direction = 'VERTICAL'
            elif abs(dx) > abs(dy) * 1.5:  # Mas horizontal
                direction = 'LATERAL'
            else:
                direction = 'DIAGONAL'
            
            # Sentido (hacia adelante o hacia atras)
            if dy > 10:
                advance = 'AVANZA'
            elif dy < -10:
                advance = 'RETROCEDE'
            else:
                advance = 'MANTIENE'
        else:
            direction = 'CORTA'
            advance = 'UNICO'
        
        return (first_pass_zone, direction, advance)
    
    def normalize_sequence(self, sequence, n_passes=3):
        """Normaliza secuencia para detectar patrones independientes de la posición inicial"""
        if len(sequence) < n_passes:
            return None
        
        # Usar los primeros n pases
        seq_subset = sequence[:n_passes]
        
        # Punto de referencia (inicio del primer pase)
        ref_x = seq_subset[0]['x']
        ref_y = seq_subset[0]['y']
        
        # Crear trayectoria normalizada
        normalized = []
        normalized.append((0, 0))  # Inicio siempre en (0,0)
        
        for pass_data in seq_subset:
            norm_end_x = pass_data['end_x'] - ref_x
            norm_end_y = pass_data['end_y'] - ref_y
            normalized.append((norm_end_x, norm_end_y))
        
        return np.array(normalized)
    
    def find_most_similar_sequences(self, sequences, top_n=3, passes_to_compare=3, eps=25, min_samples=2):
        """
        Encuentra clusters considerando:
        1. Dirección lateral del primer pase (izquierda/derecha)
        2. Profundidad del primer pase (corto/medio/largo)
        3. Forma de la trayectoria (DTW)
        """
        
        # 1. Filtrar secuencias con al menos N pases
        eligible_sequences = [
            (original_idx, seq) for original_idx, seq in enumerate(sequences) 
            if len(seq) >= passes_to_compare
        ]

        if len(eligible_sequences) < min_samples:
            print(f"⚠️ Solo hay {len(eligible_sequences)} secuencias con {passes_to_compare}+ pases.")
            return []

        original_indices, filtered_seqs = zip(*eligible_sequences)

        # 🔥 2. PRE-AGRUPAR POR DIRECCIÓN Y PROFUNDIDAD DEL PRIMER PASE
        def get_detailed_zone(first_pass):
            """
            Clasifica el primer pase por:
            - Dirección lateral (L/C/R): izquierda, centro, derecha
            - Profundidad (S/M/L): short, medium, long
            """
            end_x = first_pass['end_x']
            end_y = first_pass['end_y']
            
            # 🎯 DIRECCIÓN LATERAL (según Pass End Y)
            if end_y < 33.3:
                lateral = 'L'  # Izquierda (Left)
            elif end_y < 66.6:
                lateral = 'C'  # Centro
            else:
                lateral = 'R'  # Derecha (Right)
            
            # 🎯 PROFUNDIDAD (según Pass End X)
            if end_x < 33.3:
                depth = 'S'  # Short (Corto)
            elif end_x < 50.0:
                depth = 'M'  # Medium (Medio)
            else:
                depth = 'L'  # Long (Largo)
            
            return f"{lateral}{depth}"  # Ej: "LS" = Izquierda-Corto
        
        # Agrupar secuencias por zona detallada
        from collections import defaultdict
        sequences_by_zone = defaultdict(list)
        
        for idx, seq in enumerate(filtered_seqs):
            first_pass = seq[0]
            zone = get_detailed_zone(first_pass)
            sequences_by_zone[zone].append((idx, seq))
        
        print(f"\n📍 Secuencias agrupadas por dirección y profundidad:")
        zone_names = {
            'LS': 'Izquierda-Corto', 'LM': 'Izquierda-Medio', 'LL': 'Izquierda-Largo',
            'CS': 'Centro-Corto', 'CM': 'Centro-Medio', 'CL': 'Centro-Largo',
            'RS': 'Derecha-Corto', 'RM': 'Derecha-Medio', 'RL': 'Derecha-Largo'
        }
        for zone, seqs in sorted(sequences_by_zone.items()):
            zone_name = zone_names.get(zone, zone)
            print(f"   {zone_name}: {len(seqs)} secuencias")
        
        # 🔥 3. CLUSTERING DTW DENTRO DE CADA ZONA
        all_clusters = []
        
        for zone, zone_sequences in sequences_by_zone.items():
            if len(zone_sequences) < min_samples:
                print(f"   ⚠️ Zona {zone_names.get(zone, zone)}: muy pocas secuencias ({len(zone_sequences)}), omitida")
                continue
            
            zone_indices, zone_seqs = zip(*zone_sequences)
            
            # Crear trayectorias para esta zona
            paths = []
            for seq in zone_seqs:
                seq_subset = seq[:passes_to_compare]
                # Trayectoria: inicio + finales de cada pase
                path = [(seq_subset[0]['x'], seq_subset[0]['y'])]
                for pass_data in seq_subset:
                    path.append((pass_data['end_x'], pass_data['end_y']))
                paths.append(np.array(path))
            
            # Matriz de distancias DTW
            num_paths = len(paths)
            distance_matrix = np.zeros((num_paths, num_paths))
            
            for i in range(num_paths):
                for j in range(i, num_paths):
                    distance, _ = fastdtw(paths[i], paths[j], dist=euclidean)
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
            
            # 🔥 DBSCAN dentro de cada zona
            # Usar eps más pequeño porque ya están pre-agrupadas
            eps_zone = eps * 0.8  # Más estricto dentro de cada zona
            db = DBSCAN(eps=eps_zone, min_samples=min_samples, metric='precomputed').fit(distance_matrix)
            labels = db.labels_
            
            # Recopilar clusters de esta zona
            valid_labels = [label for label in labels if label != -1]
            
            if valid_labels:
                label_counts = Counter(valid_labels)
                for cluster_label, count in label_counts.items():
                    cluster_indices = [i for i, label in enumerate(labels) if label == cluster_label]
                    cluster_sequences = [zone_seqs[i][:passes_to_compare] for i in cluster_indices]
                    
                    all_clusters.append({
                        'zone': zone,
                        'zone_name': zone_names.get(zone, zone),
                        'count': count,
                        'sequences': cluster_sequences
                    })
            else:
                print(f"   ⚠️ Zona {zone_names.get(zone, zone)}: sin clusters claros")
        
        # 🔥 4. SELECCIONAR LOS TOP_N CLUSTERS MÁS FRECUENTES
        all_clusters.sort(key=lambda x: x['count'], reverse=True)
        top_clusters = all_clusters[:top_n]
        
        # 5. Formatear resultado
        result = []
        for idx, cluster in enumerate(top_clusters):
            result.append({
                'pattern': f"Patrón #{idx+1} ({cluster['zone_name']})",
                'count': cluster['count'],
                'all_sequences': cluster['sequences']
            })
        
        print(f"\n✅ Encontrados {len(result)} patrones distintos")
        for i, r in enumerate(result, 1):
            print(f"   Patrón {i}: {r['count']} repeticiones - {r['pattern']}")

        return result
    
    def draw_sequence_on_pitch(self, ax, all_sequences, title, count, color):
        """Dibuja TODAS las secuencias de un patrón en un campograma"""
        pitch = VerticalPitch(
            pitch_type='opta',
            pitch_color='#2d5a27',
            line_color='white',
            linewidth=2,
            label=False,
            tick=False
        )
        pitch.draw(ax=ax)
        
        # Dibujar TODAS las secuencias del patrón
        for sequence in all_sequences:
            for i, pass_data in enumerate(sequence):
                # 🔥 INTERCAMBIAR X e Y para VerticalPitch
                x_start = pass_data['y']  # ← Cambio
                y_start = pass_data['x']  # ← Cambio
                x_end = pass_data['end_y']  # ← Cambio
                y_end = pass_data['end_x']  # ← Cambio
                
                # Flecha del pase
                ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                        arrowprops=dict(arrowstyle='->', lw=2.5, color=color, alpha=0.3))
                
                # Punto inicial
                ax.plot(x_start, y_start, 'o', color='white', markersize=5, 
                    markeredgecolor=color, markeredgewidth=2, zorder=10, alpha=0.4)
            
            # Punto final de cada secuencia
            last_pass = sequence[-1]
            ax.plot(last_pass['end_y'], last_pass['end_x'], 'o', color=color,  # ← Cambio
                markersize=8, markeredgecolor='white', markeredgewidth=2, zorder=10, alpha=0.5)
    
    def create_visualization(self, team_name):
        """Visualización final con diseño 2x4, formato A4 apaisado y leyendas corregidas."""
        sequences = self.extract_goal_kick_sequences(team_name)
        if not sequences:
            print("❌ No se encontraron secuencias válidas para analizar.")
            return None

        # Filtrado de secuencias
        sequences_for_analysis = [
            seq for seq in sequences if seq and seq[0].get('outcome') == 1
        ]
        
        if not sequences_for_analysis:
            print("❌ No se encontraron secuencias con al menos un primer pase exitoso.")
            return None
            
        total_valid_sequences = len(sequences_for_analysis)
        print(f"✅ Encontradas {total_valid_sequences} secuencias con primer pase exitoso para analizar.")

        # Obtenemos los datos para las leyendas usando estas secuencias
        top_zone_patterns = self.analyze_zone_patterns(sequences_for_analysis)
        top_flow_patterns = self.analyze_pass_patterns(sequences_for_analysis)

        # --- 🔥 CONFIGURACIÓN FINAL DE LA FIGURA Y GRIDSPEC ---
        # Tamaño A4 Apaisado en pulgadas (11.69 x 8.27)
        fig = plt.figure(figsize=(11.69, 8.27), facecolor='white') 
    
        if (background := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.15)
            ax_bg.axis('off')
        
        gs = fig.add_gridspec(2, 4, height_ratios=[2.8, 1.3], width_ratios=[1, 1, 1, 0.9],
                                        wspace=0.30, hspace=0.04, left=0.04, right=0.98, top=0.88, bottom=0.02)
                    # 🔥 Cambiado de 0.25 a 0.15

        # --- 🔥 DEFINICIÓN DE EJES CORREGIDA PARA LA COLUMNA 4 ---
        ax_patterns_plot = fig.add_subplot(gs[0, 0])
        ax_firstpass_plot = fig.add_subplot(gs[0, 1])
        ax_flow_plot = fig.add_subplot(gs[0, 2])
        ax_bars = fig.add_subplot(gs[0, 3])  # Las barras van en la fila superior
        
        ax_patterns_legend = fig.add_subplot(gs[1, 0])
        ax_firstpass_legend = fig.add_subplot(gs[1, 1])
        ax_flow_legend = fig.add_subplot(gs[1, 2])
        ax_pie = fig.add_subplot(gs[1, 3])      

        # Título y logos
        fig.suptitle(f'ANÁLISIS DE SAQUES DE PUERTA - {team_name.upper()}',
                    fontsize=18, fontweight='bold', color='#1e3d59', y=0.96, family='serif')
        if (tactic_logo := self.load_tactic_logo()) is not None:
            ax_logo1 = fig.add_axes([0.02, 0.90, 0.08, 0.08], anchor='NW', zorder=10); ax_logo1.imshow(tactic_logo); ax_logo1.axis('off')
        if (team_logo := self.load_team_logo(team_name)) is not None:
            ax_logo2 = fig.add_axes([0.90, 0.90, 0.08, 0.08], anchor='NE', zorder=10); ax_logo2.imshow(team_logo); ax_logo2.axis('off')

        # --- 🔥 LLAMADAS A DIBUJO CON EL ORDEN DEFINITIVO ---
        
        # Fila 1: Gráficos
        secuencias_clasificadas = self.clasificar_secuencias_especiales(top_zone_patterns, total_valid_sequences)
        self.draw_zone_sequence_patterns(ax_patterns_plot, secuencias_clasificadas, total_valid_sequences)
        receiver_stats, total_passes = self.draw_most_frequent_first_passes(ax_firstpass_plot, sequences, team_name)
        self.draw_zone_flow_analysis(ax_flow_plot, sequences_for_analysis, team_name, total_valid_sequences)
        stats_df = self.analyze_goal_kicks_by_week()
        self.draw_goal_kicks_chart(ax_bars, ax_pie, stats_df, fig)

        # Fila 2: Leyendas
        self.draw_legend_zone_patterns(ax_patterns_legend, secuencias_clasificadas, total_valid_sequences)
        self.draw_legend_first_pass(ax_firstpass_legend, receiver_stats, total_passes)
        self.draw_legend_zone_flow(ax_flow_legend, top_flow_patterns, total_valid_sequences)

        return fig
    
    def print_summary(self, team_name):
        """Imprime un resumen de las secuencias encontradas"""
        sequences = self.extract_goal_kick_sequences(team_name)
        
        print(f"\n{'='*60}")
        print(f"RESUMEN DE SECUENCIAS TRAS SAQUE DE PUERTA")
        print(f"{'='*60}")
        print(f"Equipo: {team_name}")
        print(f"Total de secuencias validas: {len(sequences)}")
        
        if sequences:
            lengths = [len(seq) for seq in sequences]
            print(f"Promedio de pases por secuencia: {np.mean(lengths):.1f}")
            print(f"Secuencia mas larga: {max(lengths)} pases")
            print(f"Secuencia mas corta: {min(lengths)} pases")
            
            # 🔥 MISMO PARÁMETRO
            top_sequences = self.find_most_similar_sequences(
                sequences, 
                top_n=3, 
                passes_to_compare=2,  # 2 pases
                eps=20, 
                min_samples=2
            )

            if top_sequences:
                print(f"\nTop 3 patrones de 3 pases más repetidos:")
                for i, seq_data in enumerate(top_sequences, 1):
                    pattern_str = seq_data['pattern']
                    print(f"  {i}. {pattern_str}")
                    print(f"     Repeticiones: {seq_data['count']} veces")
            else:
                print("\n⚠️ No se encontraron patrones claros en los 3 primeros pases.")

def seleccionar_equipo_interactivo():
    """Funcion para seleccionar equipo interactivamente"""
    try:
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/open_play_events.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        
        if not equipos:
            print("No se encontraron equipos.")
            return None
        
        print("\n" + "="*60)
        print("SELECCION DE EQUIPO")
        print("="*60)
        for i, equipo in enumerate(equipos, 1):
            print(f"{i:2d}. {equipo}")
        
        for _ in range(3):
            try:
                indice = int(input(f"\nSelecciona un equipo (1-{len(equipos)}): ").strip()) - 1
                if 0 <= indice < len(equipos):
                    return equipos[indice]
                else:
                    print(f"Por favor, ingresa un numero entre 1 y {len(equipos)}")
            except EOFError:
                return equipos[0] if equipos else None
            except ValueError:
                print("Por favor, ingresa un numero valido")
        return equipos[0] if equipos else None
    except Exception as e:
        print(f"Error en la seleccion: {e}")
        return None

def main():
    """Funcion principal"""
    try:
        print("\n" + "="*60)
        print("ANALISIS DE SECUENCIAS TRAS SAQUE DE PUERTA")
        print("="*60)
        
        # Seleccionar equipo
        equipo = seleccionar_equipo_interactivo()
        if equipo is None:
            print("No se pudo completar la seleccion.")
            return
        
        print(f"\nAnalizando secuencias para {equipo}...")
        
        # Crear analizador
        analyzer = AnalizadorSaquesPorteria()
        
        # Cargar datos
        if not analyzer.load_data(team_filter=equipo):
            return
        
        # Mostrar resumen
        analyzer.print_summary(equipo)
        
        # Crear visualizacion
        fig = analyzer.create_visualization(equipo)
        
        if fig:
            # Guardar
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"secuencias_saque_porteria_{equipo_filename}.pdf"
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1,
                       facecolor='white', dpi=300)
            print(f"\nVisualizacion guardada como: {output_path}")
            
            plt.show()
        else:
            print("No se pudo generar la visualizacion")
    
    except Exception as e:
        print(f"Error en la ejecucion: {e}")
        import traceback
        traceback.print_exc()

def generar_secuencias_personalizado(equipo, mostrar=True, guardar=True):
    """Funcion para generar analisis de forma personalizada"""
    try:
        analyzer = AnalizadorSaquesPorteria()
        
        if not analyzer.load_data(team_filter=equipo):
            return None
        
        analyzer.print_summary(equipo)
        fig = analyzer.create_visualization(equipo)
        
        if fig:
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"secuencias_saque_porteria_{equipo_filename}.pdf"
                fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1,
                           facecolor='white', dpi=300)
                print(f"Visualizacion guardada como: {output_path}")
            
            if mostrar:
                plt.show()
            
            return fig
        else:
            print("No se pudo generar la visualizacion")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("\nINICIALIZANDO ANALIZADOR DE SAQUES DE PUERTA")
    try:
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/open_play_events.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        print(f"Sistema listo. Equipos disponibles: {len(equipos)}")
        if equipos:
            print("Para ejecutar el analisis: main()")
            print("Para uso directo: generar_secuencias_personalizado('Nombre_Equipo')")
        main()
    except Exception as e:
        print(f"Error al inicializar: {e}")