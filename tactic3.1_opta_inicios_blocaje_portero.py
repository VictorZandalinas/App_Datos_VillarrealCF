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
import json
import base64
import re
import unicodedata
from io import BytesIO
from PIL import Image
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

class AnalizadorLanzamientosPortero:
    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/match_events.parquet"): # 🔥 CAMBIO: Usar match_events
        self.data_path = data_path
        self.df = None
        self.df_complete = None
        self.sequences = []
        self.photos_data = None
        self.team_filter = None
        
        try:
            self.player_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/player_stats.parquet")
            print(f"✅ Player stats cargado: {len(self.player_stats)} registros")
        except Exception as e:
            print(f"❌ Error cargando player_stats: {e}")
            self.player_stats = None
    
    def get_opponent_by_week(self, team_name):
        if self.df_complete is None:
            return {}
        
        opponents = {}
        team_matches = self.df_complete[self.df_complete['Team Name'] == team_name][['Match ID', 'Week']].drop_duplicates()
        
        for _, match_row in team_matches.iterrows():
            match_id = match_row['Match ID']
            week = int(match_row['Week']) if pd.notna(match_row.get('Week')) else 0
            
            match_teams = self.df_complete[self.df_complete['Match ID'] == match_id]['Team Name'].unique()
            
            for opponent in match_teams:
                if opponent != team_name:
                    opponents[week] = opponent
                    break
        
        return opponents

    def get_zone_from_coords(self, x, y):
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
            
        return None
    
    def analyze_zone_patterns(self, sequences):
        from collections import defaultdict
        patterns_data = defaultdict(list)
        
        for seq in sequences:
            successful_seq = [evt for evt in seq if evt['outcome'] == 1]
            if not successful_seq:
                continue

            start_zone = self.get_zone_from_coords(successful_seq[0]['x'], successful_seq[0]['y'])
            if start_zone is None: start_zone = 1 # Asumimos que el portero empieza en zona 1
            
            zone_sequence = [start_zone]
            for event in successful_seq:
                dest_zone = self.get_zone_from_coords(event['end_x'], event['end_y'])
                if dest_zone is None: continue
                
                if dest_zone != zone_sequence[-1]:
                    zone_sequence.append(dest_zone)
                elif len(zone_sequence) == 1 and dest_zone == zone_sequence[-1]:
                    zone_sequence.append(dest_zone)

            if len(zone_sequence) > 1:
                for i in range(2, len(zone_sequence) + 1):
                    sub_pattern = tuple(zone_sequence[:i])
                    patterns_data[sub_pattern].append(seq)

        final_patterns_list = []
        for pattern, raw_sequences_list in patterns_data.items():
            first_pass = raw_sequences_list[0][0]
            first_zone_dest = self.get_zone_from_coords(first_pass['end_x'], first_pass['end_y'])
            
            category = 'medios'
            if first_zone_dest is not None:
                if 1 <= first_zone_dest <= 10: category = 'cortos'
                elif 11 <= first_zone_dest <= 16: category = 'medios'
                elif first_zone_dest > 16: category = 'largos'

            final_patterns_list.append({
                'sequence': pattern,
                'count': len(raw_sequences_list),
                'type': category,
                'raw_sequences': raw_sequences_list
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
        
        # 🔥 Calcular porcentajes de cada sección sobre el total global
        total_seccion_1 = seccion_1[0].get('total_seccion', 0) if seccion_1 else 0
        total_seccion_2 = seccion_2[0].get('total_seccion', 0) if seccion_2 else 0
        total_seccion_3 = seccion_3[0].get('total_seccion', 0) if seccion_3 else 0
        
        perc_seccion_1 = (total_seccion_1 / total_valid_sequences) * 100 if total_valid_sequences > 0 else 0
        perc_seccion_2 = (total_seccion_2 / total_valid_sequences) * 100 if total_valid_sequences > 0 else 0
        perc_seccion_3 = (total_seccion_3 / total_valid_sequences) * 100 if total_valid_sequences > 0 else 0
        
        y_pos = 0.90
        contador_global = 1
        
        # ============ SECCIÓN 1: UNA O DOS ZONAS ============
        if seccion_1:
            # 🔥 Título con porcentaje del total global a la derecha
            ax.text(0.45, y_pos + 0.02, '━━━ 1-2 ZONAS ━━━',
                    fontsize=7, fontweight='bold', ha='center', va='center',
                    color='#34495e', transform=ax.transAxes, style='italic')
            y_pos -= 0.07
            
            for pattern in seccion_1:
                self._dibujar_patron_leyenda(ax, pattern, y_pos, contador_global, total_valid_sequences)
                y_pos -= 0.11
                contador_global += 1
        
        # ============ SECCIÓN 2: MÁS DE 2 ZONAS ============
        if seccion_2:
            y_pos -= 0.01
            # 🔥 Título con porcentaje del total global a la derecha
            ax.text(0.45, y_pos + 0.02, '━━━ +2 ZONAS ━━━', 
                    fontsize=7, fontweight='bold', ha='center', va='center',
                    color='#34495e', transform=ax.transAxes, style='italic')
            y_pos -= 0.07
            
            for pattern in seccion_2:
                self._dibujar_patron_leyenda(ax, pattern, y_pos, contador_global, total_valid_sequences)
                y_pos -= 0.11
                contador_global += 1
        
        # ============ SECCIÓN 3: PROGRESIÓN ALTA ============
        if seccion_3:
            y_pos -= 0.01
            # 🔥 Título con porcentaje del total global a la derecha
            ax.text(0.45, y_pos + 0.02, '━━━ PROGR. ZONA ≥14 ━━━', 
                    fontsize=7, fontweight='bold', ha='center', va='center',
                    color='#34495e', transform=ax.transAxes, style='italic')
            y_pos -= 0.07
            
            for pattern in seccion_3:
                self._dibujar_patron_leyenda(ax, pattern, y_pos, contador_global, total_valid_sequences)
                y_pos -= 0.11
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
        
        # 🔥 PORCENTAJE: Sobre el total de todos los lanzamientos
        percentage_global = (pattern['count'] / total_valid_sequences) * 100 if total_valid_sequences > 0 else 0

        ax.text(0.78, y_pos - 0.03, f"{percentage_global:.1f}%", fontsize=9, 
                fontweight='bold', va='center', ha='left', color=pattern_color,
                transform=ax.transAxes)

    def draw_legend_first_pass(self, ax, receiver_stats, total_first_passes):

        # Mapeo de abreviaturas
        tipo_abrev = {
            'MANO': 'Mano', 'MANO LARGO': 'Mano Largo',
            'VOLEA LARGA': 'Volea Larga', 'VOLEA CORTA': 'Volea Corta',
            'SUELO CORTO': 'Suelo Corto', 'SUELO LARGO': 'Suelo Largo'
        }
        speed_abrev = {
            'MUY RÁPIDO': 'Muy rápido', 'RÁPIDO': 'Rápido',
            'LENTO': 'Lento', 'MUY LENTO': 'Muy lento'
        }

        ax.axis('off')
        ax.set_facecolor('#f8f9fa')
        
        ax.text(0.5, 0.99, 'RECEPTORES PRIMER PASE', 
                fontsize=9, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes, 
                color='#2c3e50', family='sans-serif')
        
        position_colors = ['#00BFFF', '#FF1493', '#32CD32', '#FFD700', '#9400D3']
        
        all_receivers = sorted(receiver_stats, key=lambda x: x['count'], reverse=True)[:5]
        
        if not all_receivers:
            ax.text(0.5, 0.60, 'RANKING RECEPTORES\nPRIMER PASE\n\n(Sin datos)', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, fontweight='bold', color='#7f8c8d')
            return
        
        y_pos = 0.80
        
        for i, stat in enumerate(all_receivers):
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
                photo_ax = ax.inset_axes([0.11, y_pos - 0.08, 0.20, 0.18])
                photo_ax.imshow(player_photo, aspect='auto')
                photo_ax.axis('off')
            
            # Símbolo de distancia (arriba)
            tipo_label = stat['tipo'][0]
            ax.text(0.95, y_pos + 0.06, tipo_label,
                    fontsize=9, fontweight='bold', va='center', ha='center',
                    color='#7f8c8d',
                    bbox=dict(boxstyle='circle,pad=0.3', facecolor='white',
                            edgecolor='#bdc3c7', linewidth=2),
                    transform=ax.transAxes)

            # Velocidad (centro)
            speed_text = speed_abrev.get(stat.get('speed_category', ''), '?')
            ax.text(0.95, y_pos, speed_text,
                    fontsize=7, fontweight='bold', va='center', ha='center',
                    color='#2c3e50', transform=ax.transAxes)

            # Tipo de lanzamiento (abajo)
            tipo_text = tipo_abrev.get(stat.get('launch_type', ''), '?')
            ax.text(0.95, y_pos - 0.05, tipo_text,
                    fontsize=7, fontweight='bold', va='center', ha='center',
                    color='#2c3e50', transform=ax.transAxes)

            
            nombre = stat['apellido']
            if len(nombre) > 12:
                words = nombre.split()
                if len(words) > 1:
                    ax.text(0.40, y_pos + 0.035, words[0], fontsize=9, fontweight='bold',
                            va='center', ha='left', color='#2c3e50', transform=ax.transAxes)
                    ax.text(0.40, y_pos + 0.01, ' '.join(words[1:]), fontsize=9, fontweight='bold',
                            va='center', ha='left', color='#2c3e50', transform=ax.transAxes)
                else:
                    ax.text(0.40, y_pos + 0.02, nombre[:12], fontsize=9, fontweight='bold',
                            va='center', ha='left', color='#2c3e50', transform=ax.transAxes)
            else:
                ax.text(0.40, y_pos + 0.02, nombre, fontsize=9, fontweight='bold',
                        va='center', ha='left', color='#2c3e50', transform=ax.transAxes)
            
            ax.text(0.27, y_pos + 0.03, f"{stat.get('dorsal', '?')}", 
                    fontsize=12, fontweight='bold',
                    va='center', ha='left', color='black', transform=ax.transAxes)
            
            percentage = stat.get('percentage', 0)
            ax.text(0.44, y_pos - 0.04, f"{percentage:.1f}%",
                    fontsize=11, fontweight='bold', va='center', ha='left', 
                    color=stat['color'],
                    transform=ax.transAxes)
            
            y_pos -= 0.18
        
        ax.text(0.5, 1, f'TOTAL: {total_first_passes} primeros pases', 
                fontsize=10, fontweight='bold',
                ha='center', va='bottom', color='#2c3e50', transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def draw_legend_zone_flow(self, ax, patterns_stats, total_valid_sequences):
        """
        Dibuja la leyenda para los patrones según origen del blocaje.
        Incluye porcentajes de zona, velocidad y tipo de lanzamiento.
        VERSIÓN OPTIMIZADA PARA ESPACIO VERTICAL.
        """
        ax.axis('off')
        ax.set_facecolor('#f8f9fa')
        
        ax.text(0.5, 0.99, 'TOP 2 PATRONES POR ORIGEN', 
                fontsize=8, fontweight='bold', ha='center', va='top', 
                transform=ax.transAxes, color='#2c3e50', family='sans-serif')
        
        if not patterns_stats or not any(patterns_stats[cat]['count'] > 0 for cat in patterns_stats):
            ax.text(0.5, 0.60, 'RANKING PATRONES\nSEGÚN ORIGEN\n\n(Sin datos)', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, fontweight='bold', color='#7f8c8d')
            return
        
        # Configuración de categorías
        categories_config = [
            {'key': 'ABP_LATERAL', 'color': '#FFD700', 'label': 'ABP LATERAL'},
            {'key': 'ABP_FRONTAL', 'color': '#00BFFF', 'label': 'ABP FRONTAL'},
            {'key': 'JUEGO_REGULAR', 'color': '#FF69B4', 'label': 'JUEGO REGULAR'}
        ]
        
        # Abreviaturas
        tipo_abrev = {
            'MANO': 'M', 'MANO LARGO': 'ML',
            'VOLEA LARGA': 'VL', 'VOLEA CORTA': 'VC',
            'SUELO CORTO': 'SC', 'SUELO LARGO': 'SL'
        }
        
        speed_abrev = {
            'MUY_RAPIDO': 'MR', 'RAPIDO': 'R',
            'LENTO': 'L', 'MUY_LENTO': 'ML'
        }
        
        # 🔥 CALCULAR DINÁMICAMENTE EL ESPACIO DISPONIBLE
        # Contar cuántos sub-patrones hay en total
        total_subpatterns = sum(
            len(patterns_stats.get(config['key'], {}).get('top_sub_patterns', []))
            for config in categories_config
        )

        # Espacio disponible: desde 0.90 hasta 0.08 (para dejar espacio al total)
        available_space = 0.90 - 0.08

        # Altura por sub-patrón (ajustada dinámicamente) - 🔥 MÁS ESTRECHO
        if total_subpatterns > 0:
            box_height = min(0.10, (available_space / total_subpatterns) - 0.03)  # 🔥 Cambios: 0.13→0.10 y 0.02→0.03
        else:
            box_height = 0.10  # 🔥 Cambio: 0.13→0.10
        
        y_pos = 0.90
        contador = 1
        
        for config in categories_config:
            stats = patterns_stats.get(config['key'], {})
            
            if stats.get('count', 0) == 0:
                continue
            
            top_sub_patterns = stats.get('top_sub_patterns', [])
            
            # Título del origen con porcentaje
            total_categoria = stats.get('count', 0)
            perc_categoria = (total_categoria / total_valid_sequences) * 100 if total_valid_sequences > 0 else 0

            ax.text(0.05, y_pos, config['label'], 
                    fontsize=6, fontweight='bold',
                    va='center', ha='left', color=config['color'],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor=config['color'], linewidth=1.5),
                    transform=ax.transAxes)

            # Porcentaje a la derecha
            ax.text(0.95, y_pos, f"{perc_categoria:.1f}%",
                    fontsize=7, fontweight='bold',
                    va='center', ha='right', color=config['color'],
                    transform=ax.transAxes)
            
            y_pos -= 0.05
            
            # Mostrar cada sub-patrón (máximo 2)
            for idx, sub_pattern in enumerate(top_sub_patterns):
                # Fondo de caja (más pequeño)
                rect_bg = mpatches.FancyBboxPatch((0.02, y_pos - box_height), 0.96, box_height,
                                    boxstyle="round,pad=0.01", 
                                    facecolor='white',
                                    edgecolor=config['color'],
                                    linewidth=1.5, alpha=0.9)
                ax.add_patch(rect_bg)
                
                # Número
                ax.text(0.06, y_pos - box_height/2, f"#{contador}", 
                        fontsize=9, fontweight='bold', 
                        va='center', ha='center', color=config['color'],
                        transform=ax.transAxes)
                
                # 🔥 INDICADOR VISUAL DE LÍNEA (continua o discontinua)
                # idx = 0 → línea continua, idx = 1 → línea discontinua
                linestyle = '-' if idx == 0 else '--'
                line_y = y_pos - box_height/2
                
                # Dibujar pequeña línea horizontal
                from matplotlib.lines import Line2D
                if idx == 0:
                    # Línea continua
                    line = Line2D([0.11, 0.15], [line_y, line_y], 
                                 transform=ax.transAxes, 
                                 color=config['color'], linewidth=2.5,
                                 linestyle='-', solid_capstyle='round')
                else:
                    # Línea discontinua
                    line = Line2D([0.11, 0.15], [line_y, line_y], 
                                 transform=ax.transAxes, 
                                 color=config['color'], linewidth=2.5,
                                 linestyle='--', dashes=(3, 2))
                ax.add_line(line)
                
                # Zona (C/M/L) - círculo
                zona_letra = sub_pattern['zone_category'][0]
                ax.text(0.20, y_pos - box_height/2, zona_letra,
                        fontsize=8, fontweight='bold', va='center', ha='center',
                        color='white',
                        bbox=dict(boxstyle='circle,pad=0.2', facecolor=config['color'],
                                edgecolor='white', linewidth=1.5),
                        transform=ax.transAxes)
                
                # Palabras completas de tipo y velocidad
                launch_completo = sub_pattern['launch_type']
                speed_completo = sub_pattern['speed_category'].replace('_', ' ')

                # Texto completo dividido en 2 líneas
                linea1 = launch_completo
                linea2 = speed_completo

                ax.text(0.32, y_pos - box_height/2 + 0.02, linea1, 
                        fontsize=6, fontweight='bold',
                        va='center', ha='left', color='#2c3e50',
                        transform=ax.transAxes)
                ax.text(0.32, y_pos - box_height/2 - 0.02, linea2, 
                        fontsize=6, fontweight='bold',
                        va='center', ha='left', color='#2c3e50',
                        transform=ax.transAxes)
                
                # Porcentaje (más grande y destacado)
                ax.text(0.88, y_pos - box_height/2 + 0.01, f"{sub_pattern['percentage']:.1f}%", 
                        fontsize=9, fontweight='bold',
                        va='center', ha='center', color=config['color'],
                        transform=ax.transAxes)
                
                # Conteo (debajo del porcentaje, más pequeño)
                ax.text(0.88, y_pos - box_height/2 - 0.025, f"({sub_pattern['count']})", 
                        fontsize=5,
                        va='center', ha='center', color='#7f8c8d',
                        transform=ax.transAxes)
                
                y_pos -= (box_height + 0.015)  # 🔥 Separación más pequeña entre cajas
                contador += 1
            
            # Pequeña separación extra entre orígenes
            y_pos -= 0.02
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def load_team_logo(self, equipo, target_size=(80, 80)):
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
        
    def draw_most_frequent_first_passes(self, ax, sequences, team_name):
        pitch = VerticalPitch(pitch_type='opta', pitch_color='#2d5a27',
                            line_color='white', linewidth=2, label=False, tick=False)
        pitch.draw(ax=ax)

        if not sequences:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='white')
            return [], 0

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
            
            # Clasificar tipo de lanzamiento
            launch_type = self._classify_launch_type(index)

            # Obtener velocidad desde el primer pase de la secuencia
            speed_category = None
            for seq in sequences:
                if seq and seq[0]['original_index'] == index:
                    time_diff = seq[0].get('time_diff')
                    if time_diff is not None:
                        if time_diff < 2: speed_category = 'MUY RÁPIDO'
                        elif 2 <= time_diff < 5: speed_category = 'RÁPIDO'
                        elif 5 <= time_diff <= 10: speed_category = 'LENTO'
                        else: speed_category = 'MUY LENTO'
                    break
            
            if receiver_dorsal is not None:
                first_passes_data.append({
                    'start_x': pass_event['x'], 'start_y': pass_event['y'],
                    'end_x': pass_event['Pass End X'], 'end_y': pass_event['Pass End Y'],
                    'receiver_dorsal': receiver_dorsal, 'receiver_name': player_name,
                    'launch_type': launch_type,  # NUEVO
                    'speed_category': speed_category  # NUEVO
                })
        
        total_first_passes = len(first_passes_data)
        if total_first_passes == 0: return [], 0

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
        
        position_colors = ['#00BFFF', '#FF1493', '#32CD32', '#FFD700', '#9400D3']
        receiver_stats_for_legend = []
        
        for i, recv in enumerate(top_5_receivers):
            passes = recv['passes']
            arrow_color = position_colors[i % len(position_colors)]
            
            avg_start_x = np.mean([p['start_x'] for p in passes])
            avg_start_y = np.mean([p['start_y'] for p in passes])
            avg_end_x = np.mean([p['end_x'] for p in passes])
            avg_end_y = np.mean([p['end_y'] for p in passes])

            if avg_end_x >= 50:
                curved_arrow = FancyArrowPatch(
                    (avg_start_y, avg_start_x), 
                    (avg_end_y, avg_end_x),
                    connectionstyle="arc3,rad=0.2",
                    color=arrow_color,
                    linewidth=2.5,
                    arrowstyle='->,head_width=5,head_length=5',
                    alpha=0.9, zorder=15
                )
                ax.add_patch(curved_arrow)
            else:
                pitch.arrows(avg_start_x, avg_start_y, avg_end_x, avg_end_y,
                            color=arrow_color, width=2.5, headwidth=5, headlength=5,
                            alpha=0.9, zorder=15, ax=ax)
            
            if avg_end_x < 33.3: categoria = 'CORTOS'
            elif avg_end_x < 50.0: categoria = 'MEDIOS'
            else: categoria = 'LARGOS'
            
            if recv['player_name']:
                apellido = recv['player_name'].split()[-1].upper()
                # Calcular el tipo y velocidad más frecuentes para cada receptor
                tipo_counts = Counter([p['launch_type'] for p in passes])
                speed_counts = Counter([p['speed_category'] for p in passes])
                most_common_type = tipo_counts.most_common(1)[0][0] if tipo_counts else 'N/A'
                most_common_speed = speed_counts.most_common(1)[0][0] if speed_counts else 'N/A'

                receiver_stats_for_legend.append({
                    'apellido': apellido, 'nombre_completo': recv['player_name'],
                    'tipo': categoria, 'count': recv['count'],
                    'color': arrow_color,
                    'dorsal': recv['dorsal'], 'percentage': recv['percentage'],
                    'launch_type': most_common_type,  # NUEVO
                    'speed_category': most_common_speed  # NUEVO
                })


        ax.set_title('TOP 5 RECEPTORES - 1er PASE',
            fontsize=10, fontweight='bold', color='white', pad=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#2c3e50', alpha=0.8))

        return receiver_stats_for_legend, total_first_passes

    def analyze_pass_patterns(self, sequences):
        patterns = []
        for seq in sequences:
            if not all(p.get('outcome') == 1 for p in seq):
                continue
            pattern = self.extract_pattern(seq)
            if pattern:
                patterns.append(pattern)
        
        from collections import Counter
        pattern_counts = Counter(patterns)
        total = len(patterns)
        
        top_3 = []
        for pattern_str, count in pattern_counts.most_common(3):
            percentage = (count / total) * 100 if total > 0 else 0
            top_3.append({
                'pattern': pattern_str,
                'count': count,
                'percentage': percentage
            })
        
        return top_3
    
    def extract_pattern(self, seq):
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

            if start_zone == end_zone:
                if not pattern_blocks or not pattern_blocks[-1].startswith(f'Z{start_zone}('):
                    pattern_blocks.append(f'Z{start_zone}(1)')
                else:
                    count = int(re.search(r'\((\d+)\)', pattern_blocks[-1]).group(1))
                    pattern_blocks[-1] = f'Z{start_zone}({count + 1})'
            else:
                arrow = '=>' if abs(end_zone - start_zone) > 1 else '->'
                aerial_marker = '(A)' if event.get('action_type') == 'Aerial' else ''
                pattern_blocks.append(f'{arrow}Z{end_zone}{aerial_marker}')
        
        return " ".join(pattern_blocks)
    
    def classify_keeper_event_origin(self, keeper_event_index, team_name):
        """
        Clasifica el origen del evento de portero en 3 categorías:
        - ABP Lateral: Corner taken o Free kick taken lateral del rival (y < 21.1 o y > 78.9)
        - ABP Frontal: Free kick taken frontal del rival (21.1 <= y <= 78.9)
        - Juego Regular: Ninguna de las anteriores
        """
        # 🔥 IMPORTANTE: Usar df original, no el filtrado
        keeper_event = self.df.loc[keeper_event_index]
        keeper_time = keeper_event['timeStamp']
        match_id = keeper_event['Match ID']
        
        # Buscar eventos en los 15 segundos anteriores
        time_limit = keeper_time - timedelta(seconds=15)
        
        # 🔥 CLAVE: Buscar en self.df_complete (todos los equipos), no en self.df
        previous_events = self.df_complete[
            (self.df_complete['Match ID'] == match_id) &
            (self.df_complete['Team Name'] != team_name) &
            (self.df_complete['timeStamp'] >= time_limit) &
            (self.df_complete['timeStamp'] < keeper_time)
        ]
        
        # Buscar Corner taken o Free kick taken del rival
        for idx, event in previous_events.iterrows():
            corner_taken = event.get('Corner taken')
            fk_taken = event.get('Free kick taken')
            y_coord = event.get('y')
            
            # ABP Lateral: Corner taken o Free kick taken lateral (y < 21.1 o y > 78.9)
            if pd.notna(y_coord):
                if (corner_taken == 'Sí' or fk_taken == 'Sí') and (y_coord < 21.1 or y_coord > 78.9):
                    return 'ABP_LATERAL'
                
                # ABP Frontal: Free kick taken frontal (21.1 <= y <= 78.9)
                if fk_taken == 'Sí' and 21.1 <= y_coord <= 78.9:
                    return 'ABP_FRONTAL'
        
        return 'JUEGO_REGULAR'
    
    def analyze_patterns_by_origin(self, sequences, team_name):
        """
        Analiza las secuencias agrupándolas por su origen y luego por sub-patrones
        (combinación de zona de destino y tipo de lanzamiento).
        Retorna los 2 sub-patrones más frecuentes de cada origen.
        """
        patterns_by_origin = {
            'ABP_LATERAL': [],
            'ABP_FRONTAL': [],
            'JUEGO_REGULAR': []
        }
        
        for seq in sequences:
            if not seq or seq[0].get('outcome') != 1:
                continue
            
            original_index = seq[0].get('original_index')
            if original_index is None:
                continue
            
            # Obtener el índice del evento de portero
            keeper_event_index = original_index - 1
            
            # Buscar el evento de portero más cercano
            if keeper_event_index >= 0:
                search_range = range(max(0, keeper_event_index - 5), keeper_event_index + 1)
                keeper_event_index_found = None
                
                for idx in reversed(list(search_range)):
                    if idx in self.df.index:
                        event = self.df.loc[idx]
                        if (event['Event Name'] in ['Keeper pick-up', 'Claim', 'Drop of Ball'] and 
                            event['Team Name'] == team_name and 
                            event.get('outcome') == 1):
                            keeper_event_index_found = idx
                            break
                
                if keeper_event_index_found is not None:
                    origin = self.classify_keeper_event_origin(keeper_event_index_found, team_name)
                    
                    # Calcular estadísticas del primer pase
                    first_pass = seq[0]
                    pass_end_x = first_pass['end_x']
                    
                    # Clasificar zona según Pass End X
                    if pass_end_x < 33.3:
                        zone_category = 'CORTO'
                    elif pass_end_x < 50.0:
                        zone_category = 'MEDIO'
                    else:
                        zone_category = 'LARGO'
                    
                    # Obtener velocidad
                    time_diff = first_pass.get('time_diff')
                    speed_category = 'N/A'
                    if time_diff is not None:
                        if time_diff < 2: speed_category = 'MUY_RAPIDO'
                        elif 2 <= time_diff < 5: speed_category = 'RAPIDO'
                        elif 5 <= time_diff <= 10: speed_category = 'LENTO'
                        else: speed_category = 'MUY_LENTO'
                    
                    # Obtener tipo de lanzamiento
                    launch_type = self._classify_launch_type(original_index)
                    
                    # Crear clave de sub-patrón: zona + tipo
                    sub_pattern_key = f"{zone_category}_{launch_type}"
                    
                    patterns_by_origin[origin].append({
                        'sequence': seq,
                        'zone_category': zone_category,
                        'speed_category': speed_category,
                        'launch_type': launch_type,
                        'sub_pattern_key': sub_pattern_key,
                        'first_pass': first_pass
                    })
        
        # Analizar sub-patrones para cada origen
        stats = {}
        total_sequences = sum(len(patterns_by_origin[cat]) for cat in patterns_by_origin)
        
        for origin, sequences_list in patterns_by_origin.items():
            count = len(sequences_list)
            percentage = (count / total_sequences * 100) if total_sequences > 0 else 0
            
            # Agrupar por sub-patrón
            sub_patterns = defaultdict(list)
            for item in sequences_list:
                sub_patterns[item['sub_pattern_key']].append(item)
            
            # Obtener top 2 sub-patrones
            sorted_sub_patterns = sorted(sub_patterns.items(), key=lambda x: len(x[1]), reverse=True)[:2]
            
            # Calcular estadísticas para cada sub-patrón
            top_sub_patterns = []
            for sub_key, sub_sequences in sorted_sub_patterns:
                sub_count = len(sub_sequences)
                sub_percentage = (sub_count / count * 100) if count > 0 else 0
                
                # Calcular promedios de coordenadas
                avg_start_x = np.mean([s['first_pass']['x'] for s in sub_sequences])
                avg_start_y = np.mean([s['first_pass']['y'] for s in sub_sequences])
                avg_end_x = np.mean([s['first_pass']['end_x'] for s in sub_sequences])
                avg_end_y = np.mean([s['first_pass']['end_y'] for s in sub_sequences])
                
                # Velocidad más frecuente
                speed_counts = Counter([s['speed_category'] for s in sub_sequences if s['speed_category'] != 'N/A'])
                most_common_speed = speed_counts.most_common(1)[0][0] if speed_counts else 'N/A'
                
                # Tipo más frecuente (debería ser el mismo por construcción)
                launch_type = sub_sequences[0]['launch_type']
                zone_category = sub_sequences[0]['zone_category']
                
                top_sub_patterns.append({
                    'sub_pattern_key': sub_key,
                    'count': sub_count,
                    'percentage': sub_percentage,
                    'avg_start_x': avg_start_x,
                    'avg_start_y': avg_start_y,
                    'avg_end_x': avg_end_x,
                    'avg_end_y': avg_end_y,
                    'zone_category': zone_category,
                    'launch_type': launch_type,
                    'speed_category': most_common_speed
                })
            
            stats[origin] = {
                'count': count,
                'percentage': percentage,
                'top_sub_patterns': top_sub_patterns
            }
        
        return stats

    def pattern_to_readable(self, pattern_str):
        import re
        
        if not pattern_str: return "()"
        parts = pattern_str.split(' ')
        descriptions = []
        last_zone = 1

        for part in parts:
            desc = ""
            if part.startswith('Z'):
                zone = int(re.search(r'Z(\d+)', part).group(1))
                n_passes = int(re.search(r'\((\d+)\)', part).group(1))
                plural = "pase" if n_passes == 1 else "pases"
                desc = f"{n_passes} {plural} en zona {zone}"
                last_zone = zone
            elif part.startswith('->') or part.startswith('=>'):
                target_zone = int(re.search(r'Z(\d+)', part).group(1))
                is_aerial = '(A)' in part
                prefix = "Pase largo" if '=>' in part else "Pase"
                desc = f"{prefix} de zona {last_zone} a zona {target_zone}"
                if is_aerial: desc += " (disputa)"
                last_zone = target_zone
            
            if desc: descriptions.append(desc)
        
        final_text = " + ".join(descriptions)
        if final_text:
            final_text = final_text[0].upper() + final_text[1:]
        return f"({final_text})"

    def draw_zone_flow_patterns_on_pitch(self, ax, sequences_for_analysis, team_name, total_valid_sequences):
        """Dibuja patrones visuales de lanzamientos según origen en el campo"""
        from mplsoccer import VerticalPitch
        from matplotlib import patheffects
        
        pitch = VerticalPitch(pitch_type='opta', line_zorder=2,
                            pitch_color='#2d5a27', line_color='white',
                            linewidth=2, label=False, tick=False)
        pitch.draw(ax=ax)
        
        # Clasificar secuencias por origen usando la función correcta
        abp_lateral = []
        abp_frontal = []
        juego_regular = []

        for seq in sequences_for_analysis:
            if not seq or seq[0].get('outcome') != 1:
                continue
            
            original_index = seq[0].get('original_index')
            if original_index is None:
                continue
            
            # Buscar el evento de portero
            keeper_event_index = original_index - 1
            
            if keeper_event_index >= 0:
                search_range = range(max(0, keeper_event_index - 5), keeper_event_index + 1)
                keeper_event_index_found = None
                
                for idx in reversed(list(search_range)):
                    if idx in self.df.index:
                        event = self.df.loc[idx]
                        if (event['Event Name'] in ['Keeper pick-up', 'Claim', 'Drop of Ball'] and 
                            event['Team Name'] == team_name and 
                            event.get('outcome') == 1):
                            keeper_event_index_found = idx
                            break
                
                if keeper_event_index_found is not None:
                    origin = self.classify_keeper_event_origin(keeper_event_index_found, team_name)
                    
                    if origin == 'ABP_LATERAL':
                        abp_lateral.append(seq)
                    elif origin == 'ABP_FRONTAL':
                        abp_frontal.append(seq)
                    else:
                        juego_regular.append(seq)
        
        # Configuración de orígenes
        origins_config = [
            {'sequences': abp_lateral, 'color': '#FFD700', 'label': 'ABP LAT', 'key': 'ABP_LATERAL'},
            {'sequences': abp_frontal, 'color': '#00BFFF', 'label': 'ABP FRO', 'key': 'ABP_FRONTAL'},
            {'sequences': juego_regular, 'color': '#FF69B4', 'label': 'JUEGO REG', 'key': 'JUEGO_REGULAR'}
        ]
        
        all_patterns_drawn = []
        
        for config in origins_config:
            seqs = config['sequences']
            if not seqs:
                continue
            
            # Encontrar los top 2 patrones más repetidos
            patterns = self.find_most_similar_sequences(
                seqs, top_n=2, passes_to_compare=2, eps=30, min_samples=2
            ) if len(seqs) >= 2 else []
            
            # Dibujar cada patrón
            for idx, pattern_data in enumerate(patterns[:2]):  # Máximo 2
                all_seqs = pattern_data['all_sequences']
                count = pattern_data['count']
                color = config['color']
                
                # Dibujar todas las secuencias (semitransparentes)
                for sequence in all_seqs:
                    for pass_data in sequence:
                        pitch.arrows(
                            pass_data['x'], pass_data['y'],
                            pass_data['end_x'], pass_data['end_y'],
                            color=color, width=1.5,
                            headwidth=4, headlength=4,
                            alpha=0.3, zorder=5, ax=ax
                        )
                
                # Calcular y dibujar trayectoria promedio
                avg_path = self.calculate_average_path(all_seqs)
                
                if avg_path and len(avg_path) >= 2:
                    # 🔥 ESTILO DE LÍNEA: Primera continua, segunda discontinua
                    linestyle = '-' if idx == 0 else '--'
                    
                    # Dibujar flechas con matplotlib directamente para soportar linestyle
                    from matplotlib.patches import FancyArrowPatch
                    
                    for i in range(len(avg_path) - 1):
                        x1, y1 = avg_path[i]
                        x2, y2 = avg_path[i + 1]
                        
                        # Usar FancyArrowPatch en lugar de pitch.arrows para soportar linestyle
                        arrow = FancyArrowPatch(
                            (y1, x1), (y2, x2),  # Nota: mplsoccer usa (y, x) en vertical pitch
                            arrowstyle='->,head_width=0.4,head_length=0.5',
                            color=color,
                            linewidth=2.5,
                            linestyle=linestyle,  # 🔥 Aquí se aplica continua o discontinua
                            alpha=1.0,
                            zorder=20
                        )
                        ax.add_patch(arrow)
                    
                    # Punto de inicio (círculo blanco)
                    x_start, y_start = avg_path[0]
                    pitch.scatter(x_start, y_start, s=150,
                                color='white', edgecolors=color,
                                linewidth=2, alpha=1.0, zorder=25, ax=ax)
                    
                    # Punto final
                    x_end, y_end = avg_path[-1]
                    pitch.scatter(x_end, y_end, s=200,
                                color=color, edgecolors='white',
                                linewidth=2, alpha=1.0, zorder=25, ax=ax)
                    
                    # Etiqueta con repeticiones
                    pitch.annotate(
                        f'{count}x',
                        xy=(x_end, y_end),
                        c='white', va='center', ha='center',
                        size=8, weight='bold', ax=ax, zorder=26,
                        path_effects=[patheffects.withStroke(linewidth=2.5, foreground='black')]
                    )
                    
                    all_patterns_drawn.append({
                        'label': f"{config['label']} ({count}x)",
                        'color': color,
                        'linestyle': linestyle  # 🔥 GUARDAR ESTILO
                    })
        
        ax.set_title('PATRONES POR ORIGEN',
                    fontsize=10, fontweight='bold', color='white',
                    pad=10, bbox=dict(boxstyle='round,pad=0.3',
                    facecolor='#2c3e50', alpha=0.8))

        # Preparar datos para la leyenda
        patterns_stats = {
            'ABP_LATERAL': {'count': len(abp_lateral), 'top_sub_patterns': []},
            'ABP_FRONTAL': {'count': len(abp_frontal), 'top_sub_patterns': []},
            'JUEGO_REGULAR': {'count': len(juego_regular), 'top_sub_patterns': []}
        }
        
        # Extraer info de los patrones para la leyenda
        for config in origins_config:
            seqs = config['sequences']
            key = config['key']
            
            if not seqs:
                continue
            
            patterns = self.find_most_similar_sequences(
                seqs, top_n=2, passes_to_compare=2, eps=30, min_samples=2
            ) if len(seqs) >= 2 else []
            
            for pattern_data in patterns[:2]:
                count = pattern_data['count']
                percentage = (count / total_valid_sequences) * 100 if total_valid_sequences > 0 else 0
                
                # Obtener datos del primer pase de la primera secuencia
                first_seq = pattern_data['all_sequences'][0]
                first_pass = first_seq[0]
                original_index = first_pass.get('original_index')
                
                # Clasificar zona por destino del primer pase
                end_zone = self.get_zone_from_coords(first_pass['end_x'], first_pass['end_y'])
                
                if end_zone and end_zone <= 10:
                    zone_category = 'CORTOS'
                elif end_zone and 11 <= end_zone <= 16:
                    zone_category = 'MEDIOS'
                else:
                    zone_category = 'LARGOS'
                
                # 🔥 OBTENER TIPO DE LANZAMIENTO CORRECTAMENTE
                if original_index is not None:
                    launch_type = self._classify_launch_type(original_index)
                else:
                    launch_type = 'DESCONOCIDO'
                
                # Obtener velocidad
                time_diff = first_pass.get('time_diff', 10)
                
                if time_diff < 2:
                    speed_category = 'MUY_RAPIDO'
                elif time_diff < 5:
                    speed_category = 'RAPIDO'
                elif time_diff < 10:
                    speed_category = 'LENTO'
                else:
                    speed_category = 'MUY_LENTO'
                
                patterns_stats[key]['top_sub_patterns'].append({
                    'count': count,
                    'percentage': percentage,
                    'zone_category': zone_category,
                    'launch_type': launch_type,
                    'speed_category': speed_category
                })
        
        return patterns_stats

    def find_most_similar_sequences(self, sequences, top_n=2, passes_to_compare=2, eps=30, min_samples=2):
        """Encuentra patrones de secuencias similares usando DBSCAN"""
        if len(sequences) < min_samples:
            return []
        
        # Extraer coordenadas de los primeros N pases
        coords_list = []
        for seq in sequences:
            coords = []
            for i, pass_data in enumerate(seq[:passes_to_compare]):
                coords.extend([pass_data['x'], pass_data['y'],
                            pass_data['end_x'], pass_data['end_y']])
            coords_list.append(coords)
        
        # Rellenar con ceros si faltan pases
        max_len = max(len(c) for c in coords_list)
        coords_array = np.array([c + [0] * (max_len - len(c)) for c in coords_list])
        
        # Clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_array)
        labels = clustering.labels_
        
        # Agrupar por cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:  # Ruido
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(sequences[idx])
        
        # Ordenar clusters por tamaño
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        
        patterns = []
        for cluster_id, cluster_seqs in sorted_clusters[:top_n]:
            patterns.append({
                'count': len(cluster_seqs),
                'all_sequences': cluster_seqs
            })
        
        return patterns

    def calculate_average_path(self, sequences):
        """Calcula la trayectoria promedio de un conjunto de secuencias"""
        if not sequences:
            return []
        
        max_passes = max(len(seq) for seq in sequences)
        avg_path = []
        
        for i in range(max_passes + 1):
            x_coords = []
            y_coords = []
            
            for seq in sequences:
                if i == 0:  # Punto inicial
                    x_coords.append(seq[0]['x'])
                    y_coords.append(seq[0]['y'])
                elif i <= len(seq):  # Puntos intermedios/finales
                    x_coords.append(seq[i-1]['end_x'])
                    y_coords.append(seq[i-1]['end_y'])
            
            if x_coords and y_coords:
                avg_path.append((np.mean(x_coords), np.mean(y_coords)))
        
        return avg_path


    def draw_pattern_visualization(self, ax, pattern_str, start_y, color):
        current_x = 6
        current_y = start_y
        
        ax.plot(current_y, current_x, 'o', color='white', markersize=8, 
                markeredgecolor=color, markeredgewidth=2, zorder=20)

        pattern_components = re.findall(r'(Z\d+\(\d+\))|(->Z\d+\(A\)|=>Z\d+\(A\)|->Z\d+|=>Z\d+)', pattern_str)
        
        for component_tuple in pattern_components:
            component = next(item for item in component_tuple if item)
            
            if '(' in component and 'A' not in component:
                zone = int(re.search(r'Z(\d+)', component).group(1))
                num_passes = int(re.search(r'\((\d+)\)', component).group(1))
                zone_start_x = (zone - 1) * 25
                zone_end_x = zone * 25
                start_x_for_block = max(current_x, zone_start_x)
                available_height = zone_end_x - start_x_for_block
                pass_height = available_height / num_passes
                
                for i in range(num_passes):
                    start_x = start_x_for_block + (i * pass_height)
                    end_x = start_x + pass_height
                    ax.arrow(current_y, start_x, 0, (end_x - start_x) * 0.9,
                            head_width=2.5, head_length=2.5, fc=color, ec=color, linewidth=1.5, alpha=0.9, zorder=10)
                
                current_x = zone_end_x
            elif '->' in component or '=>' in component:
                target_zone = int(re.search(r'Z(\d+)', component).group(1))
                is_aerial = '(A)' in component
                target_x = (target_zone - 1) * 25
                
                if current_x < 10 and '=>' in component: rad_val, lw = 0.3, 3
                else: rad_val, lw = 0.2, 2.5

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
                
                current_x = target_x
    
    def diagnosticar_lanzamientos_portero(self, team_name, max_samples=10):
        """
        🔍 FUNCIÓN DE DIAGNÓSTICO: Muestra qué columnas y valores tienen los pases del portero
        """
        print("\n" + "="*80)
        print("🔍 DIAGNÓSTICO PROFUNDO DE LANZAMIENTOS DEL PORTERO")
        print("="*80)
        
        # 1. Buscar si EXISTE algún evento con GK kick from hands = Sí en TODO el dataset
        gk_kicks = self.df[self.df['GK kick from hands'] == 'Sí']
        print(f"\n🔎 BÚSQUEDA GLOBAL: ¿Existe 'GK kick from hands = Sí' en el dataset?")
        print(f"   {'✅' if len(gk_kicks) > 0 else '❌'} Encontrados {len(gk_kicks)} eventos con GK kick from hands = Sí")
        
        if len(gk_kicks) > 0:
            print(f"\n   📊 Distribución de estos eventos:")
            print(gk_kicks['Event Name'].value_counts().to_string())
            print(f"\n   🔍 Primeros 5 ejemplos:")
            for idx, evento in gk_kicks.head(5).iterrows():
                print(f"      • Evento #{idx}: {evento['Event Name']} | Team: {evento.get('Team Name', 'N/A')}")
        
        # 2. Buscar eventos de portero
        keeper_events = self.df[
            (self.df['Team Name'] == team_name) &
            (self.df['Event Name'].isin(['Keeper pick-up', 'Claim', 'Drop of Ball'])) &
            (self.df['outcome'] == 1)
        ].head(max_samples)
        
        print(f"\n✅ Encontrados {len(keeper_events)} eventos de portero para analizar")
        
        # Buscar columnas relacionadas con portero y pases
        columnas_relevantes = [col for col in self.df.columns if any(keyword in col.lower() 
                              for keyword in ['keeper', 'throw', 'kick', 'hand', 'long', 'ball', 'goal kick'])]
        
        print(f"\n📋 Columnas relevantes encontradas ({len(columnas_relevantes)}):")
        for col in sorted(columnas_relevantes):
            print(f"   - {col}")
        
        # 3. Analizar TODOS los eventos siguientes (no solo Pass)
        print("\n" + "-"*80)
        print("📊 ANÁLISIS DE EVENTOS DESPUÉS DE KEEPER PICK-UP/CLAIM:")
        print("-"*80)
        
        for idx, keeper_event in keeper_events.iterrows():
            print(f"\n🟢 Evento #{idx} - {keeper_event['Event Name']} (Match ID: {keeper_event['Match ID']})")
            
            # Ver los siguientes 3 eventos
            for offset in range(1, 4):
                siguiente_idx = idx + offset
                if siguiente_idx in self.df.index:
                    siguiente = self.df.loc[siguiente_idx]
                    
                    if siguiente['Team Name'] == team_name:
                        print(f"\n   📍 Evento +{offset}: {siguiente['Event Name']}")
                        if siguiente['Event Name'] == 'Pass':
                            print(f"      Coordenadas: x={siguiente['x']:.1f}, y={siguiente['y']:.1f} → end_x={siguiente.get('Pass End X', 'N/A')}")
                            
                            print(f"      Valores relevantes:")
                            for col in ['Keeper Throw', 'GK kick from hands', 'Long ball', 'Goal Kick']:
                                valor = siguiente.get(col)
                                if pd.notna(valor) and valor != '':
                                    print(f"         • {col}: {valor}")
                        break
                    else:
                        print(f"   ⚠️ Evento +{offset} es de otro equipo: {siguiente['Team Name']}")
        
        # 4. Buscar específicamente Drop of Ball
        print("\n" + "-"*80)
        print("📊 ANÁLISIS DE 'DROP OF BALL':")
        print("-"*80)
        
        drop_events = self.df[
            (self.df['Team Name'] == team_name) &
            (self.df['Event Name'] == 'Drop of Ball')
        ].head(5)
        
        print(f"   Encontrados {len(drop_events)} eventos 'Drop of Ball'")
        
        for idx, drop_event in drop_events.iterrows():
            print(f"\n🔵 Drop of Ball #{idx}")
            siguiente_idx = idx + 1
            if siguiente_idx in self.df.index:
                siguiente = self.df.loc[siguiente_idx]
                if siguiente['Event Name'] == 'Pass' and siguiente['Team Name'] == team_name:
                    print(f"   ✓ Siguiente es Pass")
                    for col in ['Keeper Throw', 'GK kick from hands', 'Long ball', 'Goal Kick']:
                        valor = siguiente.get(col)
                        if pd.notna(valor) and valor != '':
                            print(f"      • {col}: {valor}")
        
        print("\n" + "="*80)
        print("🔍 FIN DEL DIAGNÓSTICO")
        print("="*80 + "\n")
    
    def _classify_launch_type(self, launch_event_index):
        """
        Clasifica un evento de lanzamiento del portero según los hallazgos del diagnóstico.
        
        🔥 LÓGICA CORRECTA (basada en análisis real de datos):
        
        Después de Keeper pick-up/Claim:
        - GK kick from hands = Sí + Long ball = Sí → VOLEA LARGA
        - GK kick from hands = Sí + Long ball = No → VOLEA CORTA (si existe)
        - Keeper Throw = Sí + Long ball = Sí → MANO LARGO
        - Keeper Throw = Sí + Long ball = No → MANO CORTO
        
        Después de Drop of Ball:
        - Long ball = Sí → SUELO LARGO
        - Long ball = No → SUELO CORTO
        """
        # Obtener el evento de pase que inicia la secuencia
        launch_event = self.df.loc[launch_event_index]
        
        # --- VERIFICAR SI VIENE DE 'Drop of Ball' ---
        if launch_event_index > 0:
            previous_event_index = launch_event_index - 1
            previous_event = self.df.loc[previous_event_index]
            
            # Si el evento anterior fue un 'Drop of Ball' del mismo equipo...
            if previous_event['Event Name'] == 'Drop of Ball' and previous_event['Team Name'] == launch_event['Team Name']:
                # Clasificar según distancia
                if launch_event.get('Long ball') == 'Sí':
                    return 'SUELO LARGO'
                else:
                    return 'SUELO CORTO'
        
        # --- DESPUÉS DE Keeper pick-up/Claim ---
        
        # 1. VOLEA (patada desde las manos)
        if launch_event.get('GK kick from hands') == 'Sí':
            if launch_event.get('Long ball') == 'Sí':
                return 'VOLEA LARGA'
            else:
                return 'VOLEA CORTA'  # Por si existe este caso
        
        # 2. MANO (lanzamiento con la mano)
        if launch_event.get('Keeper Throw') == 'Sí':
            if launch_event.get('Long ball') == 'Sí':
                return 'MANO LARGO'
            else:
                return 'MANO'  # Mano corto (el más común)
        
        # 3. Fallback (no debería llegar aquí normalmente)
        return 'OTRO'
    
    def analyze_launches_by_week(self):
        if self.df is None: return None
        
        launches_data = []
        tipos_lanzamiento = ['MANO', 'MANO LARGO', 'VOLEA LARGA', 'VOLEA CORTA', 'SUELO CORTO', 'SUELO LARGO']
        tipos_velocidad = ['MUY RÁPIDO', 'RÁPIDO', 'LENTO', 'MUY LENTO']


        for seq in self.sequences:
            if not seq: continue
            
            first_pass = seq[0]
            original_event = self.df.loc[first_pass['original_index']]
            launch_type = self._classify_launch_type(first_pass['original_index'])
            
            # 🔥 LÓGICA PARA CLASIFICAR VELOCIDAD
            time_diff = first_pass.get('time_diff')
            speed_category = None
            if time_diff is not None:
                if time_diff < 2: speed_category = 'MUY RÁPIDO'
                elif 2 <= time_diff < 5: speed_category = 'RÁPIDO'
                elif 5 <= time_diff <= 10: speed_category = 'LENTO'
                else: speed_category = 'MUY LENTO'

            if launch_type in tipos_lanzamiento:
                launches_data.append({
                    'Week': int(original_event['Week']) if pd.notna(original_event.get('Week')) else 0,
                    'Launch_Type': launch_type,
                    'Outcome': int(first_pass['outcome']) if pd.notna(first_pass['outcome']) else 0,
                    'Speed_Category': speed_category
                })
        
        if not launches_data: return None
        df_launches = pd.DataFrame(launches_data)
        
        # Filtrado de semanas (sin cambios)
        all_weeks = sorted(df_launches['Week'].unique())
        if len(all_weeks) > 10:
            df_launches = df_launches[df_launches['Week'].isin(all_weeks[-10:])]

        stats_by_week = []
        for week in sorted(df_launches['Week'].unique()):
            week_data = df_launches[df_launches['Week'] == week]
            row = {'Week': week}
            
            # Agregar totales por tipo de lanzamiento
            for l_type in tipos_lanzamiento:
                row[f'{l_type}_total'] = len(week_data[week_data['Launch_Type'] == l_type])
            
            # 🔥 Agregar totales por velocidad
            for s_type in tipos_velocidad:
                row[f'{s_type}_total'] = len(week_data[week_data['Speed_Category'] == s_type])
            
            stats_by_week.append(row)
        
        # Fila de TOTALES
        df_launches_completo = pd.DataFrame(launches_data)
        total_row = {'Week': 'TOTAL'}
        for l_type in tipos_lanzamiento: total_row[f'{l_type}_total'] = len(df_launches_completo[df_launches_completo['Launch_Type'] == l_type])
        for s_type in tipos_velocidad: total_row[f'{s_type}_total'] = len(df_launches_completo[df_launches_completo['Speed_Category'] == s_type])
        stats_by_week.append(total_row)
        
        return pd.DataFrame(stats_by_week)

    def draw_launches_chart(self, ax_bars, ax_pie_combined, stats_df, fig):
        """
        Dibuja el gráfico de barras y el de doble anillo (pizza más grande).
        """
        # --- 1. PREPARACIÓN DE DATOS ---
        if stats_df is None or stats_df.empty or len(stats_df) <= 1:
            for ax in [ax_bars, ax_pie_combined]:
                ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center', fontsize=12, color='gray', transform=ax.transAxes)
                ax.axis('off')
            return

        total_data = stats_df[stats_df['Week'] == 'TOTAL'].iloc[0]
        stats_df_weeks = stats_df[stats_df['Week'] != 'TOTAL'].copy()
        
        # --- 2. DEFINICIÓN DE COLORES Y ETIQUETAS ---
        tipos_lanzamiento = ['MANO', 'MANO LARGO', 'VOLEA LARGA', 'VOLEA CORTA', 'SUELO CORTO', 'SUELO LARGO']
        colores_lanzamiento = ['#3498db', '#2980b9', '#e67e22', '#d35400', '#2ecc71', '#27ae60']
        pie_labels_tipos = ['Mano', 'Mano Largo', 'Volea Larga', 'Volea Corta', 'Suelo Corto', 'Suelo Largo']
        tipos_velocidad = ['MUY RÁPIDO', 'RÁPIDO', 'LENTO', 'MUY LENTO']
        colores_velocidad = ['#c0392b', '#e74c3c', '#f1c40f', '#34495e']
        
        # --- 3. GRÁFICO DE BARRAS POR JORNADA ---
        ax_bars.set_facecolor('none')
        ax_bars.patch.set_alpha(0)
        weeks = [f"J{int(row['Week'])}" for idx, row in stats_df_weeks.iterrows()]
        y_positions = np.arange(len(weeks))
        
        for i, (idx, row) in enumerate(stats_df_weeks.iterrows()):
            left_offset = 0
            y_pos = len(weeks) - 1 - i
            for tipo, color in zip(tipos_lanzamiento, colores_lanzamiento):
                valor = row.get(f'{tipo}_total', 0)
                if valor > 0:
                    ax_bars.barh(y_pos, valor, 0.6, left=left_offset, color=color, ec='white', lw=1)
                    if valor > 0:
                        ax_bars.text(left_offset + valor/2, y_pos, f'{valor}', 
                                ha='center', va='center', fontsize=8, color='white', fontweight='bold')
                    left_offset += valor
            
            speed_counts = [row.get(f'{s_type}_total', 0) for s_type in tipos_velocidad]
            total_speed_week = sum(speed_counts)
            if total_speed_week > 0:
                left_offset_speed = 0
                ancho_total_barra_principal = left_offset
                for count, color in zip(speed_counts, colores_velocidad):
                    percentage_width = (count / total_speed_week) * ancho_total_barra_principal
                    ax_bars.barh(y_pos + 0.35, percentage_width, 0.1, left=left_offset_speed, 
                            color=color, edgecolor='white')
                    left_offset_speed += percentage_width
        
        totals_per_week = [sum(row.get(f'{t}_total', 0) for t in tipos_lanzamiento) 
                        for i, row in stats_df_weeks.iterrows()]
        if totals_per_week:
            max_total = max(totals_per_week)
            average_total = np.mean(totals_per_week)
            ax_bars.set_xlim(0, max_total * 1.15 if max_total > 0 else 10)
            if average_total > 0:
                ax_bars.axvline(x=average_total, color='#c0392b', linestyle='--', linewidth=2, zorder=10)
        
        ax_bars.set_yticks(y_positions)
        ax_bars.set_yticklabels(list(reversed(weeks)), fontsize=9, fontweight='bold')
        ax_bars.spines['top'].set_visible(False)
        ax_bars.spines['right'].set_visible(False)
        ax_bars.tick_params(colors='#2c3e50', labelsize=8)
        ax_bars.grid(axis='x', alpha=0.3, linestyle='--')
        
        opponents = self.get_opponent_by_week(self.team_filter)
        logos_opponents = [opponents.get(int(row['Week'])) for i, row in stats_df_weeks.iterrows()]
        for i, opponent in enumerate(logos_opponents):
            if opponent:
                logo = self.load_team_logo(opponent, target_size=(35, 35))
                if logo is not None:
                    ab = AnnotationBbox(OffsetImage(logo, zoom=0.5), 
                                    (-0.19, len(weeks) - 1 - i), 
                                    xycoords=('axes fraction', 'data'), frameon=False)
                    ax_bars.add_artist(ab)

        # --- 4. GRÁFICO DE DOBLE ANILLO (PIZZA MÁS GRANDE) ---
        ax_pie_combined.set_facecolor('none')
        ax_pie_combined.patch.set_alpha(0)
        ax_pie_combined.axis('equal')
        
        # 🔥 ZOOM para hacer la pizza más grande
        ax_pie_combined.set_xlim(-1.2, 1.2)
        ax_pie_combined.set_ylim(-1.2, 1.2)
        
        speed_data = [total_data.get(f'{s}_total', 0) for s in tipos_velocidad]
        pie_data_tipos = [total_data.get(f'{t}_total', 0) for t in tipos_lanzamiento]

        # --- ANILLO EXTERIOR (VELOCIDAD) MÁS GRANDE ---
        if sum(speed_data) > 0:
            # 🔥 FILTRAR datos y colores para eliminar los valores 0
            filtered_speed_data = []
            filtered_speed_colors = []
            for valor, color in zip(speed_data, colores_velocidad):
                if valor > 0:
                    filtered_speed_data.append(valor)
                    filtered_speed_colors.append(color)
            
            if filtered_speed_data:
                wedges_speed, _ = ax_pie_combined.pie(filtered_speed_data, radius=1.3,  # 🔥 Aumentado de 1.0 a 1.3
                                                    colors=filtered_speed_colors,
                                                    wedgeprops=dict(width=0.35, edgecolor='white'),  # 🔥 width de 0.3 a 0.35
                                                    startangle=90)
                
                total_speed = sum(filtered_speed_data)
                for i, wedge in enumerate(wedges_speed):
                    percentage = filtered_speed_data[i] / total_speed * 100
                    if percentage < 4: continue

                    angle = (wedge.theta1 + wedge.theta2) / 2.
                    text_radius = 1.05  # 🔥 Ajustado de 0.85 a 1.05
                    x = text_radius * np.cos(np.deg2rad(angle))
                    y = text_radius * np.sin(np.deg2rad(angle))
                    text = f'{percentage:.1f}%'
                    
                    final_rotation = angle + 90
                    if 90 < angle < 270:
                        final_rotation += 180

                    ax_pie_combined.text(x, y, text, 
                                        ha='center', va='center', 
                                        rotation=final_rotation, rotation_mode='anchor', 
                                        color='white', fontweight='bold', size=8)
        
        # --- ANILLO INTERIOR (TIPO) MÁS GRANDE ---
        if sum(pie_data_tipos) > 0:
            # 🔥 FILTRAR datos y colores para eliminar los valores 0
            filtered_data = []
            filtered_colors = []
            for valor, color in zip(pie_data_tipos, colores_lanzamiento):
                if valor > 0:
                    filtered_data.append(valor)
                    filtered_colors.append(color)
            
            if filtered_data:
                _, _, autotexts_tipos = ax_pie_combined.pie(filtered_data, radius=0.95,  # 🔥 Aumentado de 0.7 a 0.95
                                                            colors=filtered_colors,
                                                            wedgeprops=dict(edgecolor='white'), 
                                                            startangle=90,
                                                            autopct='%1.1f%%', pctdistance=0.6)
                plt.setp(autotexts_tipos, size=7, weight="bold", color="white")
        
        # --- 5. LEYENDA EXTERNA HORIZONTAL (sin cambios) ---
        velocidad_segundos = {
            'MUY RÁPIDO': '<2s',
            'RÁPIDO': '2-5s',
            'LENTO': '5-10s',
            'MUY LENTO': '>10s'
        }

        legend_labels_speed = [f"{s.replace('_', ' ').capitalize()} {velocidad_segundos[s]} ({int(v)})" 
                            for s, v in zip(tipos_velocidad, speed_data) if v > 0]
        legend_patches_speed = [mpatches.Patch(color=c) for c, v in zip(colores_velocidad, speed_data) if v > 0]

        legend_labels_tipos = [f"{s} ({int(v)})" for s, v in zip(pie_labels_tipos, pie_data_tipos) if v > 0]
        legend_patches_tipos = [mpatches.Patch(color=c) for c, v in zip(colores_lanzamiento, pie_data_tipos) if v > 0]

        all_patches = legend_patches_speed + legend_patches_tipos
        all_labels = legend_labels_speed + legend_labels_tipos

        if all_patches:
            ax_pie_combined.legend(all_patches, all_labels, 
                                loc="upper center", 
                                bbox_to_anchor=(-1.15, 1.39), 
                                ncol=3,
                                fontsize=6, 
                                frameon=True,
                                title="Velocidad y Tipo de Lanzamiento",
                                title_fontsize=7)
    
    def load_player_photos(self):
        if self.photos_data is None:
            try:
                with open('assets/jugadores_optimizados.json', 'r', encoding='utf-8') as f:
                    self.photos_data = json.load(f)
            except FileNotFoundError:
                self.photos_data = []
        return self.photos_data

    def extract_names_parts(self, name):
        def normalize_name(name):
            if not name: return ""
            name = str(name).lower().strip()
            name = unicodedata.normalize('NFD', name)
            name = ''.join(char for char in name if unicodedata.category(char) != 'Mn')
            name = re.sub(r"['\-`´'']", "", name)
            name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
            return ' '.join(name.split())
            
        normalized = normalize_name(name)
        parts = normalized.split()
        if not parts: return {'full': '', 'first_name': '', 'last_name': '', 'all_parts': []}
        first_name = parts[0]
        last_name = parts[-1] if len(parts) > 1 else first_name
        return {'full': normalized, 'first_name': first_name, 'last_name': last_name, 'all_parts': parts}

    def match_player_name(self, player_name, photos_data, team_filter=None):
        player_parts = self.extract_names_parts(player_name)
        if not player_parts['full']: return None

        team_players = photos_data
        if team_filter:
            team_players = [p for p in photos_data if self.extract_names_parts(team_filter)['full'] in self.extract_names_parts(p.get('team_name', ''))['full']]
            if not team_players: team_players = photos_data

        player_words_sorted = sorted([w for w in player_parts['all_parts'] if len(w) >= 3], key=len, reverse=True)
        
        for palabra_buscar in player_words_sorted:
            for photo_entry in team_players:
                photo_words = [w for w in self.extract_names_parts(photo_entry.get('player_name', '')).get('all_parts', []) if len(w) >= 3]
                if palabra_buscar in photo_words: return photo_entry

        return None

    def get_player_photo(self, player_name):
        if self.photos_data is None: self.load_player_photos()
        if not self.photos_data: return None
        
        match = self.match_player_name(player_name, self.photos_data, self.team_filter)
        if not match: return None
        
        try:
            img_data = base64.b64decode(match['image_base64'])
            img = Image.open(BytesIO(img_data)).convert('RGBA')
            data = np.array(img)
            height, width = data.shape[:2]
            
            def flood_fill_iterative(start_points, threshold=230):
                visited = np.zeros((height, width), dtype=bool)
                background_mask = np.zeros((height, width), dtype=bool)
                
                for start_y, start_x in start_points:
                    if visited[start_y, start_x] or not (data[start_y, start_x, 0] >= threshold and data[start_y, start_x, 1] >= threshold and data[start_y, start_x, 2] >= threshold):
                        continue
                    
                    stack = [(start_y, start_x)]
                    while stack:
                        y, x = stack.pop()
                        if not (0 <= y < height and 0 <= x < width and not visited[y, x] and (data[y, x, 0] >= threshold and data[y, x, 1] >= threshold and data[y, x, 2] >= threshold)):
                            continue
                        
                        visited[y, x] = True
                        background_mask[y, x] = True
                        stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
                return background_mask

            border_points = [(0, 0), (0, width-1), (height-1, 0), (height-1, width-1),
                             (0, width//2), (height-1, width//2), (height//2, 0), (height//2, width-1)]
            
            background_mask = flood_fill_iterative(border_points)
            data[background_mask] = [0, 0, 0, 0]
            
            return data.astype(np.float32) / 255.0
        except Exception:
            return None
    
    def calculate_action_value(self, event):
        value = 0.0
        if event['Event Name'] == 'Pass':
            value = 0.02
            if pd.notna(event.get('Pass End X')) and pd.notna(event.get('x')):
                try:
                    progression = float(event['Pass End X']) - float(event['x'])
                    if progression > 20: value += 0.03
                    elif progression > 10: value += 0.01
                except (ValueError, TypeError): pass
        elif event['Event Name'] == 'Take On': value = 0.05
        else: value = 0.01
        
        try:
            if int(event['outcome']) == 0: value *= 0.2
        except (ValueError, TypeError): pass
        return round(value, 4)
    
    def load_background(self): 
        return plt.imread("assets/fondo_informes.png") if os.path.exists("assets/fondo_informes.png") else None

    def load_tactic_logo(self):
        return plt.imread("assets/tactic_logo.png") if os.path.exists("assets/tactic_logo.png") else None

    def load_data(self, team_filter=None):
        try:
            self.df_complete = pd.read_parquet(self.data_path)
            
            if team_filter:
                self.df = self.df_complete[self.df_complete['Team Name'] == team_filter].copy()
                self.team_filter = team_filter
            else:
                self.df = self.df_complete.copy()
            
            for col in ['x', 'y', 'Pass End X', 'Pass End Y', 'outcome']:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df_complete[col] = pd.to_numeric(self.df_complete[col], errors='coerce')
            
            self.df['timeStamp'] = pd.to_datetime(self.df['timeStamp'], format='ISO8601', errors='coerce')
            self.df_complete['timeStamp'] = pd.to_datetime(self.df_complete['timeStamp'], format='ISO8601', errors='coerce')
            
            self.df = self.df.sort_values(['Match ID', 'timeStamp']).reset_index(drop=True)
            self.df_complete = self.df_complete.sort_values(['Match ID', 'timeStamp']).reset_index(drop=True)
            
            print(f"Datos cargados: {len(self.df)} eventos para el equipo seleccionado.")
            return True
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            return False

    def get_zone_center(self, zone_number):
        zone_centers = {
            1: (50, 6), 2: (29, 6), 3: (71, 6), 4: (10.5, 6), 5: (89.5, 6),
            6: (29, 17), 7: (71, 17), 8: (10.5, 17), 9: (89.5, 17), 10: (50, 17),
            11: (89.5, 28.75), 12: (57.5, 28.75), 13: (18.4, 28.75),
            14: (89.5, 42.25), 15: (50, 42.25), 16: (10.5, 42.25),
            17: (89.5, 62.5), 18: (50, 62.5), 19: (10.5, 62.5),
            20: (89.5, 87.5), 21: (50, 87.5), 22: (10.5, 87.5)
        }
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

        min_width, max_width = 1.0, 2.5
        
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
        
    # 🔥 ====================================================================
    # 🔥 FUNCIÓN CLAVE MODIFICADA: extract_keeper_launch_sequences
    # 🔥 ====================================================================
    def extract_keeper_launch_sequences(self, team_name):
        if self.df is None:
            print("No hay datos cargados")
            return []
        
        sequences = []
        team_df = self.df[self.df['Team Name'] == team_name].copy().reset_index()
        
        keeper_events = team_df[
            (team_df['Event Name'].isin(['Keeper pick-up', 'Claim', 'Drop of Ball'])) &
            (team_df['outcome'] == 1)
        ]
        
        print(f"Encontrados {len(keeper_events)} eventos de inicio ('Keeper pick-up', 'Claim', 'Drop of Ball').")
        
        for idx, keeper_event in keeper_events.iterrows():
            start_search_index = keeper_event.name + 1
            
            if start_search_index < len(team_df):
                future_events = team_df.iloc[start_search_index:]
                next_pass = future_events[future_events['Event Name'] == 'Pass'].head(1)
                
                if not next_pass.empty:
                    pass_event = next_pass.iloc[0]
                    pass_index_in_original_df = pass_event['index']
                    
                    # 🔥 CÁLCULO DEL TIEMPO DE REACCIÓN
                    time_diff = (pass_event['timeStamp'] - keeper_event['timeStamp']).total_seconds()
                    
                    # Pasamos el tiempo calculado a la siguiente función
                    sequence = self.extract_sequence_from_pass(pass_index_in_original_df, team_name, time_diff)
                    
                    if sequence and len(sequence) >= 1:
                        sequences.append(sequence)
        
        self.sequences = sequences
        print(f"Extraidas {len(sequences)} secuencias válidas de lanzamiento tras blocaje/cesión.")
        return sequences

    def extract_sequence_from_pass(self, start_idx, team_name, time_diff=None):
        pass_chain = []
        start_event = self.df.loc[start_idx]
        match_id, start_time = start_event['Match ID'], start_event['timeStamp']
        time_limit = start_time + timedelta(seconds=15)
        
        if pd.isna(start_event['Pass End X']) or pd.isna(start_event['Pass End Y']):
            return None

        first_pass_end_x, first_pass_end_y = float(start_event['Pass End X']), float(start_event['Pass End Y'])
        
        pass_chain.append({
            'x': float(start_event['x']), 'y': float(start_event['y']),
            'end_x': first_pass_end_x, 'end_y': first_pass_end_y,
            'outcome': start_event['outcome'], 'team_name': team_name,
            'action_type': 'Pass', 'value': self.calculate_action_value(start_event),
            'original_index': start_idx,
            'time_diff': time_diff # Aquí se guarda el tiempo de reacción
        })
        
        last_end_x, last_end_y = first_pass_end_x, first_pass_end_y
        current_idx = start_idx + 1
        
        while current_idx < len(self.df):
            event = self.df.iloc[current_idx]
            if len(pass_chain) >= 5 or event['Match ID'] != match_id or event['timeStamp'] > time_limit or event['Team Name'] != team_name:
                break
            if event['Event Name'] in ['Pass', 'Take On', 'Aerial']:
                current_action_start_x, current_action_start_y = last_end_x, last_end_y
                if event['Event Name'] == 'Pass' and pd.notna(event.get('Pass End X')):
                    current_action_end_x, current_action_end_y = float(event['Pass End X']), float(event['Pass End Y'])
                elif event['Event Name'] == 'Aerial' and pd.notna(event.get('x')):
                    current_action_end_x, current_action_end_y = float(event['x']), float(event['y'])
                elif event['Event Name'] == 'Take On' and pd.notna(event.get('x')):
                    current_action_end_x, current_action_end_y = float(event['x']) + 2, float(event['y'])
                else:
                    current_idx += 1; continue
                pass_chain.append({
                    'x': current_action_start_x, 'y': current_action_start_y,
                    'end_x': current_action_end_x, 'end_y': current_action_end_y,
                    'outcome': event['outcome'], 'team_name': team_name,
                    'action_type': event['Event Name'], 'value': self.calculate_action_value(event)
                })
                last_end_x, last_end_y = current_action_end_x, current_action_end_y
            current_idx += 1
        return pass_chain
    
    def clasificar_secuencias_especiales(self, top_patterns, total_sequences):
        """
        Clasifica secuencias en 3 secciones:
        SECCIÓN 1: Secuencias de 1 o 2 zonas
        - 1 de juego directo (zona final ≥14) - solo puede ser 2 zonas
        - 1 de juego corto (zona final <14) - puede ser 1 zona o 2 zonas (cualquier inicio)
        SECCIÓN 2: Secuencias con más de 2 zonas
        - Top 2 con mayor porcentaje
        SECCIÓN 3: Secuencias con progresión a zona ≥13
        - Top 2 con mayor porcentaje (más de 2 pases)
        """
        # Clasificar por tipo
        dos_zonas_directo = []  # 1-2 zonas, zona final ≥14
        dos_zonas_corto = []    # 1-2 zonas, zona final <14
        mas_dos_zonas = []      # Más de 2 zonas
        progresion_alta = []    # TODAS las que acaban en zona ≥14

        # 🔥 NUEVOS CONTADORES GLOBALES POR TIPO
        total_1_2_zonas = 0  # Total de secuencias con 1-2 zonas
        total_mas_2_zonas = 0  # Total de secuencias con más de 2 zonas
        total_progresion = 0  # Total de secuencias que acaban en zona ≥14

        for pattern in top_patterns:
            sequence = pattern['sequence']
            n_zonas = len(sequence)
            zona_final = sequence[-1]
            
            # SECCIÓN 3: TODAS las secuencias que acaban en zona ≥14
            if zona_final >= 14:
                total_progresion += pattern['count']
                progresion_alta.append(pattern)
            
            # SECCIÓN 1: Secuencias de 1 o 2 zonas (cualquier zona de inicio)
            if n_zonas <= 2:
                total_1_2_zonas += pattern['count']
                if zona_final >= 14:
                    dos_zonas_directo.append(pattern)
                else:
                    dos_zonas_corto.append(pattern)
            
            # SECCIÓN 2: Más de 2 zonas
            elif n_zonas > 2:
                total_mas_2_zonas += pattern['count']
                mas_dos_zonas.append(pattern)
        
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
                'seccion': 1,
                'total_seccion': total_1_2_zonas  # 🔥 Total de 1-2 zonas
            })
        
        # Top 1 juego corto (azul)
        if dos_zonas_corto:
            mejor_corto = sorted(dos_zonas_corto, key=lambda x: x['count'], reverse=True)[0]
            resultado.append({
                **mejor_corto, 
                'categoria': 'dos_zonas_corto', 
                'color': '#3498db',
                'seccion': 1,
                'total_seccion': total_1_2_zonas  # 🔥 Total de 1-2 zonas
            })
        
        # === SECCIÓN 2: MÁS DE 2 ZONAS ===
        # Top 2 (naranja)
        for p in sorted(mas_dos_zonas, key=lambda x: x['count'], reverse=True)[:2]:
            resultado.append({
                **p, 
                'categoria': 'mas_dos_zonas', 
                'color': '#e67e22',
                'seccion': 2,
                'total_seccion': total_mas_2_zonas  # 🔥 Total de +2 zonas
            })
        
        # === SECCIÓN 3: PROGRESIÓN ALTA (≥zona 14) ===
        # Top 2 (morado)
        for p in sorted(progresion_alta, key=lambda x: x['count'], reverse=True)[:2]:
            resultado.append({
                **p, 
                'categoria': 'progresion_alta', 
                'color': '#8e44ad',
                'seccion': 3,
                'total_seccion': total_progresion  # 🔥 Total de progresión alta
            })
        
        return resultado
        
    def create_visualization(self, team_name):
            """
            Crea la visualización final completa, con el nuevo gráfico de doble anillo y leyenda externa.
            """
            # 1. Extracción y validación de datos
            sequences = self.extract_keeper_launch_sequences(team_name)
            if not sequences:
                print("❌ No se encontraron secuencias válidas para analizar.")
                fig, ax = plt.subplots(figsize=(11.69, 8.27), facecolor='white')
                ax.text(0.5, 0.5, f'No se encontraron secuencias de lanzamiento\npara {team_name}', 
                        ha='center', va='center', fontsize=18, color='red')
                ax.axis('off')
                return fig

            sequences_for_analysis = [s for s in sequences if s and s[0].get('outcome') == 1]
            
            if not sequences_for_analysis:
                print("❌ No se encontraron secuencias con al menos un primer pase exitoso.")
                fig, ax = plt.subplots(figsize=(11.69, 8.27), facecolor='white')
                ax.text(0.5, 0.5, f'No se encontraron secuencias con primer pase exitoso\npara {team_name}', 
                        ha='center', va='center', fontsize=18, color='orange')
                ax.axis('off')
                return fig
                
            total_valid_sequences = len(sequences_for_analysis)
            print(f"✅ Encontradas {total_valid_sequences} secuencias con primer pase exitoso para analizar.")

            # 2. Análisis de datos
            top_zone_patterns = self.analyze_zone_patterns(sequences_for_analysis)
            top_flow_patterns = self.analyze_pass_patterns(sequences_for_analysis)
            stats_df = self.analyze_launches_by_week()

            # 3. Configuración de la figura y la maquetación
            fig = plt.figure(figsize=(11.69, 8.27), facecolor='white') 
        
            if (background := self.load_background()) is not None:
                ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
                ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.15)
                ax_bg.axis('off')
            
            # Maquetación con un solo espacio para el gráfico de anillo
            gs = fig.add_gridspec(2, 4, 
                      height_ratios=[2.8, 1.3], 
                      width_ratios=[1, 1, 1, 0.9],
                      wspace=0.30, hspace=0.05, 
                      left=0.04, right=0.98, top=0.88, bottom=0.02)

            # Definición de los ejes
            ax_patterns_plot = fig.add_subplot(gs[0, 0])
            ax_firstpass_plot = fig.add_subplot(gs[0, 1])
            ax_flow_plot = fig.add_subplot(gs[0, 2])
            ax_bars = fig.add_subplot(gs[0, 3])

            ax_patterns_legend = fig.add_subplot(gs[1, 0])
            ax_firstpass_legend = fig.add_subplot(gs[1, 1])
            ax_flow_legend = fig.add_subplot(gs[1, 2])

            # 🔥 ELIMINADO: ax_legend_vertical (no se usa)
            # ax_legend_vertical = fig.add_subplot(gs[1, 3])

            # El gráfico de anillo
            ax_pie_combined = fig.add_subplot(gs[1, 3])


            # 4. Títulos y Logos
            fig.suptitle(f'ANÁLISIS LANZAMIENTOS DE PORTERO (TRAS BLOCAJE)',
                        fontsize=16, fontweight='bold', color='#1e3d59', y=0.96, family='serif')
            if (tactic_logo := self.load_tactic_logo()) is not None:
                ax_logo1 = fig.add_axes([0.02, 0.90, 0.08, 0.08], anchor='NW', zorder=10); ax_logo1.imshow(tactic_logo); ax_logo1.axis('off')
            if (team_logo := self.load_team_logo(team_name)) is not None:
                ax_logo2 = fig.add_axes([0.90, 0.90, 0.08, 0.08], anchor='NE', zorder=10); ax_logo2.imshow(team_logo); ax_logo2.axis('off')

            # 5. Llamadas a las funciones de dibujo
            secuencias_clasificadas = self.clasificar_secuencias_especiales(top_zone_patterns, total_valid_sequences)
            self.draw_zone_sequence_patterns(ax_patterns_plot, secuencias_clasificadas, total_valid_sequences)
            receiver_stats, total_passes = self.draw_most_frequent_first_passes(ax_firstpass_plot, sequences, team_name)
            patterns_by_origin = self.draw_zone_flow_patterns_on_pitch(ax_flow_plot, sequences_for_analysis, team_name, total_valid_sequences)
            
            # Llamada actualizada con el eje único
            self.draw_launches_chart(ax_bars, ax_pie_combined, stats_df, fig)

            self.draw_legend_zone_patterns(ax_patterns_legend, secuencias_clasificadas, total_valid_sequences)
            self.draw_legend_first_pass(ax_firstpass_legend, receiver_stats, total_passes)
            self.draw_legend_zone_flow(ax_flow_legend, patterns_by_origin, total_valid_sequences)

            return fig    
    
    def print_summary(self, team_name):
        # La extracción se hace ahora en create_visualization, así que la llamamos primero
        sequences = self.extract_keeper_launch_sequences(team_name)
        
        print(f"\n{'='*60}")
        print(f"RESUMEN DE SECUENCIAS TRAS BLOCAJE DE PORTERO")
        print(f"{'='*60}")
        print(f"Equipo: {team_name}")
        print(f"Total de secuencias validas: {len(sequences)}")
        
        if sequences:
            lengths = [len(seq) for seq in sequences]
            print(f"Promedio de pases por secuencia: {np.mean(lengths):.1f}")
            print(f"Secuencia mas larga: {max(lengths)} pases")
            print(f"Secuencia mas corta: {min(lengths)} pases")

def seleccionar_equipo_interactivo():
    try:
        # 🔥 Cargar desde match_events para la selección
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/match_events.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        
        if not equipos: print("No se encontraron equipos."); return None
        
        print("\n" + "="*60 + "\nSELECCION DE EQUIPO\n" + "="*60)
        for i, equipo in enumerate(equipos, 1): print(f"{i:2d}. {equipo}")
        
        while True:
            try:
                idx = int(input(f"\nSelecciona un equipo (1-{len(equipos)}): ").strip()) - 1
                if 0 <= idx < len(equipos): return equipos[idx]
                else: print(f"Por favor, ingresa un numero entre 1 y {len(equipos)}")
            except ValueError: print("Por favor, ingresa un numero valido")
    except Exception as e:
        print(f"Error en la seleccion: {e}"); return None

def main():
    try:
        print("\n" + "="*60)
        print("ANALISIS DE SECUENCIAS TRAS BLOCAJE DE PORTERO")
        print("="*60)
        
        equipo = seleccionar_equipo_interactivo()
        if equipo is None: print("No se pudo completar la seleccion."); return
        
        print(f"\nAnalizando secuencias para {equipo}...")
        
        analyzer = AnalizadorLanzamientosPortero()
        
        if not analyzer.load_data(team_filter=equipo): return
        
        # 🔥 COMENTADAS LAS LÍNEAS DE DEBUG
        # print("\n🔍 Ejecutando diagnóstico de orígenes de blocaje...")
        # analyzer.debug_keeper_event_origins(equipo, max_samples=10)
        # continuar = input("\n¿Continuar con la visualización? (s/n): ").strip().lower()
        # if continuar != 's':
        #     print("Diagnóstico completado. Saliendo...")
        #     return
        
        fig = analyzer.create_visualization(equipo)
        
        if fig:
            equipo_filename = re.sub(r'[\s/]', '_', equipo)
            output_path = f"lanzamientos_portero_blocaje_{equipo_filename}.pdf"
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1, facecolor='white', dpi=300)
            print(f"\nVisualizacion guardada como: {output_path}")
            plt.show()
        else:
            print("No se pudo generar la visualizacion")
    
    except Exception as e:
        print(f"Error en la ejecucion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\nINICIALIZANDO ANALIZADOR DE LANZAMIENTOS DE PORTERO")
    try:
        df = pd.read_parquet("extraccion_opta/datos_opta_parquet/match_events.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        print(f"Sistema listo. Equipos disponibles: {len(equipos)}")
        if equipos:
            print("Para ejecutar el analisis, ejecute: main()")
        main()
    except Exception as e:
        print(f"Error al inicializar: {e}")