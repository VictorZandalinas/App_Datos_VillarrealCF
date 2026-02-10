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

class VillarrealDistanciasReport:
    def __init__(self, data_path="extraccion_mediacoach/data/rendimiento_fisico.parquet"):
        """
        Inicializa la clase para generar informes de distancias recorridas del Villarreal CF
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
            "Villarreal_CF",
            "villarreal_cf",
            "Villarreal CF",
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
    
    def create_distances_data(self, filtered_df, jornadas):
        """Procesa los datos de distancias para los gráficos"""
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

        # Agrupar datos por jugador y jornada
        pivot_data = filtered_df.groupby(['Alias', 'Jornada']).agg({
            'Distancia Total': 'sum',
            'Distancia Total 14-21 km / h': 'sum',
            'Distancia Total  21-24 km / h': 'sum',
            'Distancia Total >24 km / h': 'sum',
            'Dorsal': 'first'
        }).reset_index()
        
        # Procesar datos para gráficos (MANTENER EN METROS)
        jugadores_data = {}
        for jugador in pivot_data['Alias'].unique():
            player_data = pivot_data[pivot_data['Alias'] == jugador]
            dorsal = player_data['Dorsal'].iloc[0] if len(player_data) > 0 else 'N/A'
            
            jugadores_data[jugador] = {
                'dorsal': dorsal,
                'jornadas': {},
                'totales': {
                    'total': 0,
                    '14_21': 0,
                    '21_24': 0,
                    'mas_24': 0
                }
            }
            
            for jornada in normalized_jornadas:
                jornada_data = player_data[player_data['Jornada'] == jornada]
                if len(jornada_data) > 0:
                    row = jornada_data.iloc[0]
                    distancias = {
                        'total': float(row['Distancia Total']),  # Mantener en metros
                        '14_21': float(row['Distancia Total 14-21 km / h']),
                        '21_24': float(row['Distancia Total  21-24 km / h']),
                        'mas_24': float(row['Distancia Total >24 km / h'])
                    }
                else:
                    distancias = {'total': 0, '14_21': 0, '21_24': 0, 'mas_24': 0}
                
                jugadores_data[jugador]['jornadas'][jornada] = distancias
                
                # Acumular totales
                for key in distancias:
                    jugadores_data[jugador]['totales'][key] += distancias[key]
        
        return jugadores_data, normalized_jornadas
    
    def create_visualization(self, jornadas, figsize=(16, 11)):
        """Crea la visualización completa para el Villarreal CF"""
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
        
        # Configurar grid: header + 4 gráficos verticales con barras horizontales
        gs = fig.add_gridspec(2, 4, 
                             height_ratios=[0.08, 1], 
                             width_ratios=[1, 1, 1, 1], 
                             hspace=0.12, wspace=0.12,
                             left=0.03, right=0.97, top=0.95, bottom=0.05)
        
        # Área del título (toda la fila superior)
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        # Título principal
        ax_title.text(0.5, 0.8, 'DISTANCIAS RECORRIDAS - VILLARREAL CF', 
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
                imagebox = OffsetImage(logo, zoom=0.40)
                ab = AnnotationBbox(imagebox, (0.97, 0.5), frameon=False)
                ax_title.add_artist(ab)
            except Exception as e:
                print(f"❌ Error al aplicar escudo: {e}")
        else:
            print("⚠️ No se pudo cargar el escudo del Villarreal")
        
        # Procesar datos
        jugadores_data, normalized_jornadas = self.create_distances_data(filtered_df, jornadas)
        
        # 4 gráficos verticales con barras horizontales, uno al lado del otro
        titles = ['DISTANCIA TOTAL POR PARTIDO', 'DIST. 14-21 km/h', 'DIST. 21-24 km/h', 'DIST. >24 km/h']
        data_keys = ['total', '14_21', '21_24', 'mas_24']
        chart_types = ['stacked', 'simple', 'simple', 'simple']  # El primero es acumulativo
        
        for i, (title, data_key, chart_type) in enumerate(zip(titles, data_keys, chart_types)):
            ax_chart = fig.add_subplot(gs[1, i])
            ax_chart.set_facecolor('none')  # Sin fondo
            ax_chart.set_title(title, fontsize=10, weight='bold', 
                              color='#1e3d59', pad=10)
            
            if chart_type == 'stacked':
                self.plot_horizontal_stacked_bars(ax_chart, jugadores_data, normalized_jornadas)
            else:
                color = '#2ecc71' if data_key == '14_21' else '#f39c12' if data_key == '21_24' else '#e74c3c'
                self.plot_horizontal_simple_bars(ax_chart, jugadores_data, data_key, color)
        
        return fig
    
    def plot_horizontal_stacked_bars(self, ax, jugadores_data, jornadas):
        """Dibuja barras horizontales acumulativas por jornadas"""
        jugadores = list(jugadores_data.keys())
        if not jugadores:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center')
            ax.axis('off')
            return
        
        # Ordenar jugadores por total de distancia
        jugadores_ordenados = sorted(jugadores, 
                                   key=lambda x: jugadores_data[x]['totales']['total'], 
                                   reverse=True)
        jugadores_ordenados.reverse()  # Para mostrar de mayor a menor en el gráfico
        
        # Colores para jornadas (misma paleta)
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
        
        y_positions = np.arange(len(jugadores_ordenados))
        bar_width = 0.8
        
        # Crear barras apiladas
        left_values = np.zeros(len(jugadores_ordenados))
        
        for j_idx, jornada in enumerate(jornadas):
            distancias_jornada = []
            for jugador in jugadores_ordenados:
                distancia = jugadores_data[jugador]['jornadas'].get(jornada, {'total': 0})['total']
                distancias_jornada.append(distancia)
            
            bars = ax.barh(y_positions, distancias_jornada, bar_width, 
                          left=left_values, 
                          label=f'J{jornada}', 
                          color=colors[j_idx % len(colors)])
            
            # Añadir valores en segmentos grandes (mostrar en miles)
            for i, (bar, distancia) in enumerate(zip(bars, distancias_jornada)):
                if distancia > 1000:  # Solo mostrar si es mayor a 1000 metros
                    valor_mostrar = distancia / 1000
                    ax.text(left_values[i] + distancia/2, bar.get_y() + bar.get_height()/2, 
                           f"{valor_mostrar:.1f}mil", ha='center', va='center', 
                           fontsize=6, weight='bold', color='white')
            
            left_values += distancias_jornada
        
        # Etiquetas de jugadores y totales
        for i, (total, jugador) in enumerate(zip(left_values, jugadores_ordenados)):
            if total > 0:
                # Total al final (en miles)
                total_km = total / 1000
                ax.text(total + total*0.02, i, f"{total_km:.1f}mil", 
                       va='center', fontsize=7, color='#1a237e', weight='bold')
                
                # Nombre del jugador - posicionado a la izquierda donde empieza la barra y más arriba
                ax.text(0, i + 0.30, jugador,
                       va='center', ha='left', fontsize=7, color='#1a237e', weight='bold',
                       bbox=dict(facecolor='#ffffff', alpha=0.9, edgecolor='#2c3e50',
                                linewidth=0.5, pad=1, boxstyle='round,pad=0.05'))
        
        # Configurar ejes
        ax.set_yticks([])
        ax.set_xlabel('metros', fontsize=8, color='#2c3e50')
        
        # Ajustar límites
        max_total = max(left_values) if len(left_values) > 0 else 20000
        ax.set_xlim(0, max_total * 1.08)
        
        # Leyenda en la parte inferior
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), 
                 fontsize=6, ncol=len(jornadas), frameon=True, fancybox=True, shadow=True)
        
        # Estilo sin fondo
        ax.set_facecolor('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', alpha=0.2)
        ax.tick_params(axis='both', labelsize=7)
    
    def plot_horizontal_simple_bars(self, ax, jugadores_data, data_key, color):
        """Dibuja barras horizontales simples para un tipo específico de distancia"""
        jugadores = list(jugadores_data.keys())
        if not jugadores:
            ax.text(0.5, 0.5, 'No hay datos', ha='center', va='center')
            ax.axis('off')
            return
        
        # Ordenar por total del tipo específico
        jugadores_ordenados = sorted(jugadores, 
                                   key=lambda x: jugadores_data[x]['totales'][data_key], 
                                   reverse=True)
        jugadores_ordenados.reverse()  # Para mostrar de mayor a menor en el gráfico
        
        # Datos para el gráfico (mantener en metros)
        valores = [jugadores_data[jugador]['totales'][data_key] for jugador in jugadores_ordenados]
        
        # Crear gráfico de barras horizontales
        y_positions = np.arange(len(jugadores_ordenados))
        bars = ax.barh(y_positions, valores, color=color, alpha=0.8, 
                      edgecolor='white', linewidth=0.5, height=0.8)
        
        # Añadir valores al final de las barras (en miles)
        for i, (bar, valor, jugador) in enumerate(zip(bars, valores, jugadores_ordenados)):
            if valor > 0:
                valor_mostrar = valor / 1000
                ax.text(valor + valor*0.02, bar.get_y() + bar.get_height()/2, 
                       f"{valor_mostrar:.1f}mil", ha='left', va='center', 
                       fontsize=6, weight='bold', color='#2c3e50')
                
                # Nombre del jugador - posicionado a la izquierda donde empieza la barra y más arriba
                ax.text(0, i + 0.30, jugador,
                       va='center', ha='left', fontsize=7, color='#1a237e', weight='bold',
                       bbox=dict(facecolor='#ffffff', alpha=0.9, edgecolor='#2c3e50',
                                linewidth=0.5, pad=1, boxstyle='round,pad=0.05'))
        
        # Configurar ejes
        ax.set_yticks([])
        ax.set_xlabel('metros', fontsize=8, color='#2c3e50')
        
        # Ajustar límites
        if valores:
            max_val = max(valores)
            ax.set_xlim(0, max_val * 1.08)
        
        # Leyenda simple (mostrar total en miles)
        total_km = sum(valores) / 1000
        ax.legend([f'Total: {total_km:.1f}mil metros'], 
                 loc='upper right', bbox_to_anchor=(0.98, 0.98),
                 fontsize=6, frameon=True, fancybox=True, shadow=True)
        
        # Estilo sin fondo
        ax.set_facecolor('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', alpha=0.2)
        ax.tick_params(axis='both', labelsize=7)

# Funciones auxiliares específicas para Villarreal CF
def seleccionar_jornadas_villarreal():
    """Permite al usuario seleccionar jornadas para el Villarreal CF"""
    try:
        report_generator = VillarrealDistanciasReport()
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

def main_villarreal():
    try:
        pass
        
        # Selección de jornadas para Villarreal CF
        jornadas = seleccionar_jornadas_villarreal()
        
        if jornadas is None:
            pass
            return
        
        
        # Crear el reporte
        report_generator = VillarrealDistanciasReport()
        fig = report_generator.create_visualization(jornadas)
        
        if fig:
            # Mostrar en pantalla
            plt.show()
            
            # Guardar como PDF
            output_path = f"reporte_distancias_Villarreal_CF.pdf"
            
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

def generar_reporte_villarreal_personalizado(jornadas, mostrar=True, guardar=True):
    """Función para generar un reporte personalizado del Villarreal CF"""
    try:
        report_generator = VillarrealDistanciasReport()
        fig = report_generator.create_visualization(jornadas)
        
        if fig:
            if mostrar:
                plt.show()
            
            if guardar:
                output_path = f"reporte_distancias_Villarreal_CF.pdf"
                
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

# Inicialización específica para Villarreal CF
try:
    report_generator = VillarrealDistanciasReport()
    jornadas_disponibles = report_generator.get_available_jornadas()
    
    if len(jornadas_disponibles) > 0:
        pass
    
except Exception as e:
    print(f"❌ Error al inicializar: {e}")

if __name__ == "__main__":
    main_villarreal()