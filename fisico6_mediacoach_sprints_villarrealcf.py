import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import patheffects
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

class VillarrealSprintsReport:
    def __init__(self, data_path="extraccion_mediacoach/data/rendimiento_fisico.parquet"):
        """
        Inicializa la clase para generar informes de sprints del Villarreal CF
        """
        self.data_path = data_path
        self.equipo = "Villarreal CF"  # Equipo fijo
        self.df = None
        self.load_data()
        self.clean_team_names()
        
    def load_data(self):
        """Carga los datos del archivo parquet"""
        try:
            self.df = pd.read_parquet(self.data_path)
        except Exception as e:
            pass
            
    def similarity(self, a, b):
        """Calcula la similitud entre dos strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def clean_team_names(self):
        """Limpia y agrupa nombres de equipos similares y normaliza jornadas"""
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
                    if self.similarity(team, other_team) > 0.7:  # 70% de similitud
                        similar_teams.append(other_team)
            
            # Elegir el nombre más largo como representativo
            canonical_name = max(similar_teams, key=len)
            
            # Mapear todos los nombres similares al canónico
            for similar_team in similar_teams:
                team_mapping[similar_team] = canonical_name
                processed_teams.add(similar_team)
        
        # Aplicar el mapeo
        self.df['Equipo'] = self.df['Equipo'].map(team_mapping)
        
        # Normalizar jornadas: convertir 'J1', 'J2', etc. a números 1, 2, etc.
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
        
        
    def get_available_jornadas(self):
        """Retorna las jornadas disponibles para el Villarreal CF"""
        if self.df is None:
            return []
        
        # Filtrar solo datos del Villarreal CF
        villarreal_df = self.df[self.df['Equipo'].str.contains('Villarreal', case=False, na=False)]
        if len(villarreal_df) == 0:
            print("⚠️ No se encontraron datos del Villarreal CF. Equipos disponibles:")
            return []
        
        return sorted(villarreal_df['Jornada'].unique())
    
    def filter_data(self, jornadas):
        """Filtra los datos por Villarreal CF y jornadas específicas"""
        if self.df is None:
            return None
        
        # Normalizar jornadas (pueden venir como 'J1', 'J2' o como números)
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
        
        
        # Filtrar por Villarreal CF y jornadas
        filtered_df = self.df[
            (self.df['Equipo'].str.contains('Villarreal', case=False, na=False)) & 
            (self.df['Jornada'].isin(normalized_jornadas))
        ].copy()
        
        return filtered_df
    
    def load_team_logo(self):
        """Carga el escudo del Villarreal CF"""
        # Intentar diferentes variaciones del nombre del Villarreal
        possible_names = [
            "Villarreal",
            "villarreal",
            "Villarreal CF",
            "villarreal_cf",
            "VillarrealCF",
            "villarrealcf"
        ]
        
        for name in possible_names:
            logo_path = f"assets/escudos/{name}.png"
            if os.path.exists(logo_path):
                pass
                try:
                    return plt.imread(logo_path)
                except Exception as e:
                    pass
                    continue
        
        return None
    
    def load_ball_image(self):
        """Carga la imagen del balón"""
        ball_path = "assets/balon.png"
        if os.path.exists(ball_path):
            pass
            try:
                return plt.imread(ball_path)
            except Exception as e:
                pass
                return None
        else:
            pass
            return None
    
    def load_background(self):
        """Carga el fondo del informe"""
        bg_path = "assets/fondo_informes.png"
        if os.path.exists(bg_path):
            pass
            try:
                return plt.imread(bg_path)
            except Exception as e:
                pass
                return None
        else:
            pass
            return None
    
    def create_sprints_data(self, filtered_df, jornadas):
        """Procesa los datos de sprints para los gráficos"""
        # Normalizar jornadas de entrada
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
        
        # Verificar si Alias está vacío y usar Nombre en su lugar
        if 'Nombre' in filtered_df.columns:
            mask_empty_alias = filtered_df['Alias'].isna() | (filtered_df['Alias'] == '') | (filtered_df['Alias'].str.strip() == '')
            filtered_df.loc[mask_empty_alias, 'Alias'] = filtered_df.loc[mask_empty_alias, 'Nombre']

        # Agrupar datos por jugador y jornada para el gráfico apilado
        pivot_data = filtered_df.groupby(['Alias', 'Jornada']).agg({
            'N Total Sprints >21 km / h': 'sum',
            'N Total Sprints 21-24 km / h': 'sum',
            'N Total Sprints >24 km / h': 'sum',
            'N Total Sprints >21 km / h 1P': 'sum',
            'N Total Sprints >21 km / h 2P': 'sum',
            'N Total Sprints >24 km / h 1P': 'sum',
            'N Total Sprints >24 km / h 2P': 'sum',
            'Dorsal': 'first'
        }).reset_index()
        
        # Agrupar datos por jugador para gráficos totales
        total_data = filtered_df.groupby(['Alias']).agg({
            'N Total Sprints >21 km / h': 'sum',
            'N Total Sprints 21-24 km / h': 'sum',
            'N Total Sprints >24 km / h': 'sum',
            'N Total Sprints >21 km / h 1P': 'sum',
            'N Total Sprints >21 km / h 2P': 'sum',
            'N Total Sprints >24 km / h 1P': 'sum',
            'N Total Sprints >24 km / h 2P': 'sum',
            'Dorsal': 'first'
        }).reset_index()
        
        # Procesar datos para gráficos
        jugadores_data = {}
        for jugador in pivot_data['Alias'].unique():
            player_data = pivot_data[pivot_data['Alias'] == jugador]
            total_player = total_data[total_data['Alias'] == jugador].iloc[0]
            dorsal = player_data['Dorsal'].iloc[0] if len(player_data) > 0 else 'N/A'
            
            jugadores_data[jugador] = {
                'dorsal': dorsal,
                'jornadas': {},
                'totales': {
                    'sprints_21': int(total_player['N Total Sprints >21 km / h']),
                    'sprints_21_24': int(total_player['N Total Sprints 21-24 km / h']),
                    'sprints_24': int(total_player['N Total Sprints >24 km / h']),
                    'sprints_21_1p': int(total_player['N Total Sprints >21 km / h 1P']),
                    'sprints_21_2p': int(total_player['N Total Sprints >21 km / h 2P']),
                    'sprints_24_1p': int(total_player['N Total Sprints >24 km / h 1P']),
                    'sprints_24_2p': int(total_player['N Total Sprints >24 km / h 2P'])
                }
            }
            
            for jornada in normalized_jornadas:
                jornada_data = player_data[player_data['Jornada'] == jornada]
                if len(jornada_data) > 0:
                    row = jornada_data.iloc[0]
                    sprints = {
                        'sprints_21': int(row['N Total Sprints >21 km / h'])
                    }
                else:
                    sprints = {'sprints_21': 0}
                
                jugadores_data[jugador]['jornadas'][jornada] = sprints
        
        return jugadores_data, normalized_jornadas
    
    def create_visualization(self, jornadas, figsize=(16, 11)):
        """Crea la visualización completa de sprints para el Villarreal CF"""
        # Filtrar datos
        filtered_df = self.filter_data(jornadas)
        if filtered_df is None or len(filtered_df) == 0:
            pass
            return None
        
        # Crear figura
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        # Cargar y establecer fondo
        background = self.load_background()
        if background is not None:
            try:
                ax_background = fig.add_axes([0, 0, 1, 1], zorder=-1)
                ax_background.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25, zorder=-1)
                ax_background.axis('off')
                ax_background.set_xticks([])
                ax_background.set_yticks([])
                for spine in ax_background.spines.values():
                    spine.set_visible(False)
            except Exception as e:
                pass
        
        # Configurar grid: header + 4 gráficos (2 izq, 1 centro, 2 der apilados)
        gs = fig.add_gridspec(3, 3, 
                             height_ratios=[0.08, 0.46, 0.46], 
                             width_ratios=[1, 1, 1], 
                             hspace=0.30, wspace=0.40,
                             left=0.05, right=0.97, top=0.95, bottom=0.05)
        
        # Área del título (toda la fila superior)
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        # Título principal
        ax_title.text(0.5, 0.8, 'Nº SPRINTS - VILLARREAL CF', 
                     fontsize=24, weight='bold', ha='center', va='center',
                     color='#1e3d59', family='serif')
        ax_title.text(0.5, 0.2, f'ÚLTIMAS {len(jornadas)} JORNADAS', 
                     fontsize=12, ha='center', va='center',
                     color='#2c3e50', weight='bold')
        
        # Balón izquierda
        ball = self.load_ball_image()
        if ball is not None:
            try:
                imagebox = OffsetImage(ball, zoom=0.15)
                ab = AnnotationBbox(imagebox, (0.05, 0.5), frameon=False)
                ax_title.add_artist(ab)
            except Exception as e:
                print(f"❌ Error al aplicar balón: {e}")
        else:
            print("⚠️ No se pudo cargar el balón")
        
        # Escudo derecha (Villarreal CF)
        logo = self.load_team_logo()
        if logo is not None:
            try:
                imagebox = OffsetImage(logo, zoom=0.45)
                ab = AnnotationBbox(imagebox, (0.95, 0.5), frameon=False)
                ax_title.add_artist(ab)
            except Exception as e:
                print(f"❌ Error al aplicar escudo: {e}")
        else:
            print("⚠️ No se pudo cargar el escudo del Villarreal")
        
        # Procesar datos
        jugadores_data, normalized_jornadas = self.create_sprints_data(filtered_df, jornadas)
        
        # Gráfico 1: Nº SPRINTS >21km/h (por jornadas) - ocupa 2 filas izquierda
        ax1 = fig.add_subplot(gs[1:, 0])
        ax1.set_facecolor('none')
        ax1.set_title('Nº SPRINTS >21km/h', fontsize=11, weight='bold', 
                     color='#1e3d59', pad=0.5)
        self.plot_sprints_by_jornadas(ax1, jugadores_data, normalized_jornadas)
        
        # Gráfico 2: Nº SPRINTS 21-24km/h / Nº SPRINTS >24km/h - ocupa 2 filas centro
        ax2 = fig.add_subplot(gs[1:, 1])
        ax2.set_facecolor('none')
        ax2.set_title('Nº SPRINTS 21-24km/h\nNº SPRINTS >24km/h', fontsize=11, weight='bold', 
                     color='#1e3d59', pad=0.5)
        self.plot_sprints_21_24_vs_24(ax2, jugadores_data)
        
        # Gráfico 3: Nº SPRINTS >21km/h (1P vs 2P) - superior derecha
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.set_facecolor('none')
        ax3.set_title('Nº SPRINTS >21km/h', fontsize=11, weight='bold', 
                     color='#1e3d59', pad=0.5)
        self.plot_sprints_1p_vs_2p(ax3, jugadores_data, sprint_type='21')
        
        # Gráfico 4: Nº SPRINTS >24km/h (1P vs 2P) - inferior derecha
        ax4 = fig.add_subplot(gs[2, 2])
        ax4.set_facecolor('none')
        ax4.set_title('Nº SPRINTS >24km/h', fontsize=11, weight='bold', 
                     color='#1e3d59', pad=0.5)
        self.plot_sprints_1p_vs_2p(ax4, jugadores_data, sprint_type='24')
        
        return fig
    
    def plot_sprints_by_jornadas(self, ax, jugadores_data, jornadas):
        """Dibuja barras horizontales apiladas por jornadas"""
        jugadores = list(jugadores_data.keys())
        if not jugadores:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center')
            ax.axis('off')
            return
        
        # Ordenar jugadores por total de sprints >21
        jugadores_ordenados = sorted(jugadores, 
                                   key=lambda x: jugadores_data[x]['totales']['sprints_21'], 
                                   reverse=False)
        
        # Colores para jornadas (tonos verdes)
        colors = ['#a8e6cf', '#7fcdcd', '#81c784', '#66bb6a', '#4caf50', '#43a047', '#388e3c', '#2e7d32']
        
        y_positions = np.arange(len(jugadores_ordenados))
        bar_width = 0.8
        
        # Crear barras apiladas
        left_values = np.zeros(len(jugadores_ordenados))
        
        for j_idx, jornada in enumerate(jornadas):
            sprints_jornada = []
            for jugador in jugadores_ordenados:
                sprints = jugadores_data[jugador]['jornadas'].get(jornada, {'sprints_21': 0})['sprints_21']
                sprints_jornada.append(sprints)
            
            bars = ax.barh(y_positions, sprints_jornada, bar_width, 
                          left=left_values, 
                          label=f'{jornada}', 
                          color=colors[j_idx % len(colors)])
            
            # Añadir valores en segmentos
            for i, (bar, sprints) in enumerate(zip(bars, sprints_jornada)):
                if sprints > 2:  # Solo mostrar si es mayor a 2
                    ax.text(left_values[i] + sprints/2, bar.get_y() + bar.get_height()/2, 
                           f"{sprints}", ha='center', va='center', 
                           fontsize=6, weight='bold', color='white')
            
            left_values += sprints_jornada
        
        # Etiquetas de jugadores y totales
        for i, (total, jugador) in enumerate(zip(left_values, jugadores_ordenados)):
            if total > 0:
                # Total al final
                ax.text(total + total*0.02, i, f"{int(total)}", 
                       va='center', fontsize=7, color='#1a237e', weight='bold')
                
                # Nombre del jugador
                ax.text(-max(left_values)*0.02 if left_values.max() > 0 else -2, i, jugador,
                       va='center', ha='right', fontsize=7, color='#1a237e', weight='bold')
        
        # Configurar ejes
        ax.set_yticks([])
        ax.set_xlabel('Nº Sprints', fontsize=8, color='#2c3e50')
        
        # Ajustar límites
        max_total = max(left_values) if len(left_values) > 0 and max(left_values) > 0 else 50
        ax.set_xlim(-max_total*0.1, max_total * 1.1)
        
        # Leyenda compacta
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), 
                 fontsize=6, ncol=len(jornadas), frameon=True, fancybox=True, shadow=True)
        
        # Estilo
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', alpha=0.2)
        ax.tick_params(axis='both', labelsize=7)
    
    def plot_sprints_21_24_vs_24(self, ax, jugadores_data):
        """Dibuja barras horizontales apiladas para 21-24 vs >24"""
        jugadores = list(jugadores_data.keys())
        if not jugadores:
            ax.text(0.5, 0.5, 'No hay datos', ha='center', va='center')
            ax.axis('off')
            return
        
        # Ordenar por total de sprints de alta velocidad
        jugadores_ordenados = sorted(jugadores, 
                                   key=lambda x: jugadores_data[x]['totales']['sprints_21_24'] + jugadores_data[x]['totales']['sprints_24'], 
                                   reverse=False)
        
        y_positions = np.arange(len(jugadores_ordenados))
        
        # Datos para las zonas
        sprints_21_24 = [jugadores_data[j]['totales']['sprints_21_24'] for j in jugadores_ordenados]
        sprints_24 = [jugadores_data[j]['totales']['sprints_24'] for j in jugadores_ordenados]
        
        # Crear barras apiladas
        bars_21_24 = ax.barh(y_positions, sprints_21_24, 
                           label='Sprints 21-24 km/h', color='#f39c12', alpha=0.8)
        bars_24 = ax.barh(y_positions, sprints_24, left=sprints_21_24,
                        label='Sprints >24 km/h', color='#e74c3c', alpha=0.8)
        
        # Calcular totales
        totales = [s21_24 + s24 for s21_24, s24 in zip(sprints_21_24, sprints_24)]
        
        # Añadir valores
        for i, (s21_24, s24, total, jugador) in enumerate(zip(sprints_21_24, sprints_24, totales, jugadores_ordenados)):
            # Total al final
            if total > 0:
                ax.text(total + total*0.02, i, f"{int(total)}", 
                       ha='left', va='center', fontsize=7, weight='bold', color='#2c3e50')
            
            # Valores en segmentos
            if s21_24 > 2:
                ax.text(s21_24/2, i, f"{int(s21_24)}", 
                       ha='center', va='center', fontsize=6, weight='bold', color='white')
            
            if s24 > 2:
                ax.text(s21_24 + s24/2, i, f"{int(s24)}", 
                       ha='center', va='center', fontsize=6, weight='bold', color='white')
            
            # Nombre del jugador
            ax.text(-max(totales)*0.02 if totales else -2, i, jugador,
                   va='center', ha='right', fontsize=7, color='#1a237e', weight='bold')
        
        # Configurar ejes
        ax.set_yticks([])
        ax.set_xlabel('Nº Sprints', fontsize=8, color='#2c3e50')
        
        # Ajustar límites
        max_total = max(totales) if totales else 30
        ax.set_xlim(-max_total*0.1, max_total * 1.1)
        
        # Leyenda
        ax.legend(loc='lower right', bbox_to_anchor=(0.98, 0.02), 
                 fontsize=7, frameon=True, fancybox=True, shadow=True)
        
        # Estilo
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', alpha=0.2)
        ax.tick_params(axis='both', labelsize=7)
    
    def plot_sprints_1p_vs_2p(self, ax, jugadores_data, sprint_type='21'):
        """Dibuja gráfico de barras verticales para 1P vs 2P"""
        jugadores = list(jugadores_data.keys())
        if not jugadores:
            ax.text(0.5, 0.5, 'No hay datos', ha='center', va='center')
            ax.axis('off')
            return
        
        # Determinar qué datos usar según el tipo de sprint
        if sprint_type == '21':
            sort_key = 'sprints_21'
            data_1p_key = 'sprints_21_1p'
            data_2p_key = 'sprints_21_2p'
            label_1p = 'Sprints >21 km/h 1P'
            label_2p = 'Sprints >21 km/h 2P'
        else:  # sprint_type == '24'
            sort_key = 'sprints_24'
            data_1p_key = 'sprints_24_1p'
            data_2p_key = 'sprints_24_2p'
            label_1p = 'Sprints >24 km/h 1P'
            label_2p = 'Sprints >24 km/h 2P'
        
        # Ordenar por total de sprints del tipo especificado
        jugadores_ordenados = sorted(jugadores, 
                                   key=lambda x: jugadores_data[x]['totales'][sort_key], 
                                   reverse=True)[:12]  # Top 12 para que quepan mejor
        
        x_positions = np.arange(len(jugadores_ordenados))
        bar_width = 0.35
        
        # Datos
        sprints_1p = [jugadores_data[j]['totales'][data_1p_key] for j in jugadores_ordenados]
        sprints_2p = [jugadores_data[j]['totales'][data_2p_key] for j in jugadores_ordenados]
        
        # Crear barras
        bars_1p = ax.bar(x_positions - bar_width/2, sprints_1p, bar_width, 
                        label=label_1p, color='#3498db', alpha=0.8)
        bars_2p = ax.bar(x_positions + bar_width/2, sprints_2p, bar_width, 
                        label=label_2p, color='#2c3e50', alpha=0.8)
        
        # Añadir valores encima de las barras
        for i, (s1p, s2p) in enumerate(zip(sprints_1p, sprints_2p)):
            if s1p > 0:
                ax.text(i - bar_width/2, s1p + 0.3, f"{int(s1p)}", 
                       ha='center', va='bottom', fontsize=6, weight='bold', color='#2c3e50')
            if s2p > 0:
                ax.text(i + bar_width/2, s2p + 0.3, f"{int(s2p)}", 
                       ha='center', va='bottom', fontsize=6, weight='bold', color='#2c3e50')
        
        # Configurar ejes
        ax.set_xticks(x_positions)
        ax.set_xticklabels(jugadores_ordenados, rotation=45, ha='right', fontsize=6)
        ax.set_ylabel('Nº Sprints', fontsize=8, color='#2c3e50')
        
        # Leyenda
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                 fontsize=6, frameon=True, fancybox=True, shadow=True)
        
        # Estilo
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.2)
        ax.tick_params(axis='both', labelsize=6)

# Funciones auxiliares específicas para Villarreal CF
def seleccionar_jornadas_villarreal_sprints():
    """Permite al usuario seleccionar jornadas para el Villarreal CF - Reporte de Sprints"""
    try:
        report_generator = VillarrealSprintsReport()
        jornadas_disponibles = report_generator.get_available_jornadas()
        
        if len(jornadas_disponibles) == 0:
            pass
            return None
        
        
        # Preguntar cuántas jornadas incluir
        while True:
            try:
                num_jornadas = input(f"¿Cuántas jornadas incluir? (máximo {len(jornadas_disponibles)}): ").strip()
                num_jornadas = int(num_jornadas)
                
                if 1 <= num_jornadas <= len(jornadas_disponibles):
                    jornadas_seleccionadas = sorted(jornadas_disponibles)[-num_jornadas:]
                    break
                else:
                    pass
            except ValueError:
                pass
        
        return jornadas_seleccionadas
        
    except Exception as e:
        pass
        return None

def main_villarreal_sprints():
    try:
        pass
        
        # Selección de jornadas para Villarreal CF
        jornadas = seleccionar_jornadas_villarreal_sprints()
        
        if jornadas is None:
            pass
            return
        
        
        # Crear el reporte
        report_generator = VillarrealSprintsReport()
        fig = report_generator.create_visualization(jornadas)
        
        if fig:
            # Mostrar en pantalla
            plt.show()
            
            # Guardar como PDF
            output_path = f"reporte_sprints_Villarreal_CF.pdf"
            
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(output_path) as pdf:
                fig.patch.set_alpha(0.0)
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0, 
                          facecolor='none', edgecolor='none', dpi=300,
                          transparent=True)
            
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()

def generar_reporte_sprints_villarreal_personalizado(jornadas, mostrar=True, guardar=True):
    """Función para generar un reporte personalizado de sprints del Villarreal CF"""
    try:
        report_generator = VillarrealSprintsReport()
        fig = report_generator.create_visualization(jornadas)
        
        if fig:
            if mostrar:
                plt.show()
            
            if guardar:
                output_path = f"reporte_sprints_Villarreal_CF.pdf"
                
                from matplotlib.backends.backend_pdf import PdfPages
                with PdfPages(output_path) as pdf:
                    fig.patch.set_alpha(0.0)
                    pdf.savefig(fig, bbox_inches='tight', pad_inches=0, 
                              facecolor='none', edgecolor='none', dpi=300,
                              transparent=True)
                
            
            return fig
        else:
            print("❌ No se pudo generar la visualización")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Inicialización específica para Villarreal CF - Sprints
try:
    report_generator = VillarrealSprintsReport()
    jornadas_disponibles = report_generator.get_available_jornadas()
    
    if len(jornadas_disponibles) > 0:
        pass
    
except Exception as e:
    print(f"❌ Error al inicializar: {e}")

if __name__ == "__main__":
    main_villarreal_sprints()