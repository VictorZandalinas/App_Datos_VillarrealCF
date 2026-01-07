import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import patheffects
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import unicodedata
import re
import os
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

class DistanciaZonasReport:
    def __init__(self, data_path="extraccion_mediacoach/data/rendimiento_fisico.parquet"):
        """
        Inicializa la clase para generar informes de distancia por zonas
        """
        self.data_path = data_path
        self.df = None
        self.load_data()
        self.clean_team_names()
        
    def load_data(self):
        """Carga los datos del archivo parquet"""
        try:
            self.df = pd.read_parquet(self.data_path)
            print(f"Datos cargados exitosamente: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
            print(f"Columnas disponibles: {list(self.df.columns)}")
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            
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
                    if self.similarity(team, other_team) > 0.85:
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
        
        print(f"Limpieza completada. Equipos únicos: {len(self.df['Equipo'].unique())}")
        print(f"Jornadas normalizadas en datos: {sorted(self.df['Jornada'].unique())}")
        
    def get_available_teams(self):
        """Retorna la lista de equipos disponibles"""
        if self.df is None:
            return []
        return sorted(self.df['Equipo'].unique())
    
    def get_available_jornadas(self, equipo=None):
        """Retorna las jornadas disponibles, opcionalmente filtradas por equipo"""
        if self.df is None:
            return []
        
        if equipo:
            filtered_df = self.df[self.df['Equipo'] == equipo]
            return sorted(filtered_df['Jornada'].unique())
        else:
            return sorted(self.df['Jornada'].unique())
    
    def filter_data(self, equipo, jornadas):
        """Filtra los datos por equipo y jornadas específicas"""
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
        
        print(f"Jornadas normalizadas: {normalized_jornadas}")
        print(f"Jornadas únicas en datos: {sorted(self.df['Jornada'].unique())}")
        
        filtered_df = self.df[
            (self.df['Equipo'] == equipo) & 
            (self.df['Jornada'].isin(normalized_jornadas))
        ].copy()
        
        print(f"Datos filtrados: {len(filtered_df)} filas para {equipo}")
        return filtered_df
    
    @staticmethod
    def normalize_text(text):
        """Normaliza texto eliminando acentos, espacios extra y caracteres especiales"""
        # Eliminar acentos
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        # Convertir a minúsculas y limpiar
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_team_logo(self, equipo):
        """
        Carga el escudo del equipo con una potente lógica de búsqueda jerárquica.
        1. Mapeo Manual: Anula todo lo demás para casos conflictivos.
        2. Coincidencia Exacta: Busca una correspondencia perfecta entre nombres normalizados.
        3. Coincidencia de Palabra Larga: Busca si una palabra significativa (>4 letras) coincide.
        4. Similitud: Como último recurso, busca nombres con una alta similitud de texto.
        """
        escudos_dir = "assets/escudos"
        if not os.path.exists(escudos_dir):
            print(f"Directorio de escudos no encontrado: {escudos_dir}")
            return None

        # --- Nivel 1: MAPEO MANUAL (Máxima Prioridad) ---
        # Edita esta sección para forzar la correspondencia de equipos problemáticos.
        TEAM_LOGO_MAP = {
            self.normalize_text("Athletic Club"): "Athletic",
            self.normalize_text("Atletico de Madrid"): "Atlético",
            # self.normalize_text("Real Betis Balompie"): "betis", # Ejemplo
        }
        
        equipo_norm = self.normalize_text(equipo)
        if equipo_norm in TEAM_LOGO_MAP:
            logo_filename = TEAM_LOGO_MAP[equipo_norm]
            for ext in ['.png', '.jpg', '.jpeg']:
                logo_path = os.path.join(escudos_dir, f"{logo_filename}{ext}")
                if os.path.exists(logo_path):
                    print(f"✅ Escudo encontrado por Mapeo Manual: {logo_path}")
                    try:
                        return plt.imread(logo_path)
                    except Exception as e:
                        print(f"Error al cargar escudo mapeado: {e}")
            print(f"⚠️ Advertencia: El archivo mapeado '{logo_filename}' no fue encontrado.")

        # --- Búsqueda Automática ---
        available_files = [f for f in os.listdir(escudos_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # --- Nivel 2: COINCIDENCIA EXACTA ---
        for filename in available_files:
            file_base_norm = self.normalize_text(os.path.splitext(filename)[0])
            if file_base_norm == equipo_norm:
                logo_path = os.path.join(escudos_dir, filename)
                print(f"✅ Escudo encontrado por Coincidencia Exacta: {logo_path}")
                try:
                    return plt.imread(logo_path)
                except Exception as e:
                    print(f"Error al cargar escudo por coincidencia exacta: {e}")

        # --- Nivel 3: COINCIDENCIA DE PALABRA LARGA (Tu idea) ---
        MIN_WORD_LENGTH = 4 # Busca palabras con 5 o más letras
        team_long_words = {word for word in equipo_norm.split() if len(word) > MIN_WORD_LENGTH}

        if team_long_words:
            # Crea un diccionario de búsqueda de archivos normalizados
            normalized_files = {self.normalize_text(os.path.splitext(f)[0]): f for f in available_files}

            for file_norm, original_filename in normalized_files.items():
                file_words = set(file_norm.split())
                
                # Comprueba si alguna palabra larga del equipo está en las palabras del nombre del archivo
                if not team_long_words.isdisjoint(file_words):
                    logo_path = os.path.join(escudos_dir, original_filename)
                    print(f"✅ Escudo encontrado por Palabra Larga Común ({team_long_words.intersection(file_words)}): {logo_path}")
                    try:
                        return plt.imread(logo_path)
                    except Exception as e:
                        print(f"Error al cargar escudo por palabra larga: {e}")

        # --- Nivel 4: BÚSQUEDA POR SIMILITUD (Último Recurso) ---
        best_match_file = None
        best_similarity = 0.88  # Umbral alto para evitar errores

        for filename in available_files:
            file_base_norm = self.normalize_text(os.path.splitext(filename)[0])
            similarity = self.similarity(equipo_norm, file_base_norm)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_file = filename
        
        if best_match_file:
            logo_path = os.path.join(escudos_dir, best_match_file)
            print(f"✅ Escudo encontrado por Similitud (score: {best_similarity:.2f}): {logo_path}")
            try:
                return plt.imread(logo_path)
            except Exception as e:
                print(f"Error al cargar escudo por similitud: {e}")

        print(f"❌ No se encontró un escudo definitivo para: {equipo} (normalizado como: {equipo_norm})")
        return None
        
    def load_ball_image(self):
        """Carga la imagen del balón"""
        ball_path = "assets/balon.png"
        if os.path.exists(ball_path):
            print(f"Balón encontrado: {ball_path}")
            try:
                return plt.imread(ball_path)
            except Exception as e:
                print(f"Error al cargar balón: {e}")
                return None
        else:
            print(f"No se encontró el balón: {ball_path}")
            return None
    
    def load_background(self):
        """Carga el fondo del informe"""
        bg_path = "assets/fondo_informes.png"
        if os.path.exists(bg_path):
            print(f"Fondo encontrado: {bg_path}")
            try:
                return plt.imread(bg_path)
            except Exception as e:
                print(f"Error al cargar fondo: {e}")
                return None
        else:
            print(f"No se encontró el fondo: {bg_path}")
            return None
    
    def create_zones_data(self, filtered_df, jornadas):
        """Procesa los datos de distancias por zonas para los gráficos"""
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

        # Agrupar datos por jugador
        pivot_data = filtered_df.groupby(['Alias']).agg({
            'Distancia Total': 'sum',
            'Distancia Total 14-21 km / h': 'sum',
            'Distancia Total  21-24 km / h': 'sum',
            'Distancia Total >24 km / h': 'sum',
            'Dorsal': 'first'
        }).reset_index()
        
        # Procesar datos para gráficos (MANTENER EN METROS)
        jugadores_data = {}
        for jugador in pivot_data['Alias'].unique():
            player_data = pivot_data[pivot_data['Alias'] == jugador].iloc[0]
            
            jugadores_data[jugador] = {
                'dorsal': player_data['Dorsal'],
                'distancia_total': float(player_data['Distancia Total']),
                'dist_14_21': float(player_data['Distancia Total 14-21 km / h']),
                'dist_21_24': float(player_data['Distancia Total  21-24 km / h']),
                'dist_mas_24': float(player_data['Distancia Total >24 km / h'])
            }
        
        return jugadores_data, normalized_jornadas
    
    def create_visualization(self, equipo, jornadas, figsize=(16, 11)):
        """Crea la visualización completa para distancia por zonas"""
        # Filtrar datos
        filtered_df = self.filter_data(equipo, jornadas)
        if filtered_df is None or len(filtered_df) == 0:
            print("No hay datos para los filtros especificados")
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
                print("Fondo aplicado correctamente")
            except Exception as e:
                print(f"Error al aplicar fondo: {e}")
        
        # Configurar grid: header + 2 gráficos horizontales
        gs = fig.add_gridspec(2, 2, 
                             height_ratios=[0.08, 1], 
                             width_ratios=[1, 1], 
                             hspace=0.12, wspace=0.1,
                             left=0.03, right=0.97, top=0.95, bottom=0.05)
        
        # Área del título (toda la fila superior)
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        # Título principal
        ax_title.text(0.5, 0.8, 'DISTANCIA POR ZONAS', 
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
                print("✅ Balón aplicado correctamente")
            except Exception as e:
                print(f"❌ Error al aplicar balón: {e}")
        else:
            print("⚠️ No se pudo cargar el balón")
        
        # Escudo derecha
        logo = self.load_team_logo(equipo)
        if logo is not None:
            try:
                imagebox = OffsetImage(logo, zoom=0.45)
                ab = AnnotationBbox(imagebox, (0.95, 0.5), frameon=False)
                ax_title.add_artist(ab)
                print("✅ Escudo aplicado correctamente")
            except Exception as e:
                print(f"❌ Error al aplicar escudo: {e}")
        else:
            print("⚠️ No se pudo cargar el escudo")
        
        # Procesar datos
        jugadores_data, normalized_jornadas = self.create_zones_data(filtered_df, jornadas)
        
        # Gráfico izquierdo: DISTANCIA TOTAL / DIST 14-21KM/H
        ax_left = fig.add_subplot(gs[1, 0])
        ax_left.set_facecolor('none')
        ax_left.set_title('DISTANCIA TOTAL / DIST 14-21KM/H', fontsize=12, weight='bold', 
                         color='#1e3d59', pad=15)
        self.plot_total_vs_14_21(ax_left, jugadores_data)
        
        # Gráfico derecho: DISTANCIA POR ZONAS
        ax_right = fig.add_subplot(gs[1, 1])
        ax_right.set_facecolor('none')
        ax_right.set_title('DISTANCIA POR ZONAS', fontsize=12, weight='bold', 
                          color='#1e3d59', pad=15)
        self.plot_zones(ax_right, jugadores_data)
        
        return fig
    
    def plot_total_vs_14_21(self, ax, jugadores_data):
        """Dibuja el gráfico de distancia total vs 14-21 km/h"""
        jugadores = list(jugadores_data.keys())
        if not jugadores:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center')
            ax.axis('off')
            return
        
        # Ordenar jugadores por distancia total
        jugadores_ordenados = sorted(jugadores, 
                                   key=lambda x: jugadores_data[x]['distancia_total'], 
                                   reverse=False)
        
        y_positions = np.arange(len(jugadores_ordenados))
        bar_height = 0.35
        
        # Datos para los gráficos
        distancias_total = [jugadores_data[j]['distancia_total'] for j in jugadores_ordenados]
        distancias_14_21 = [jugadores_data[j]['dist_14_21'] for j in jugadores_ordenados]
        
        # Crear barras
        bars_total = ax.barh(y_positions - bar_height/2, distancias_total, bar_height, 
                           label='Distancia Total', color='#3498db', alpha=0.8)
        bars_14_21 = ax.barh(y_positions + bar_height/2, distancias_14_21, bar_height, 
                           label='14-21 km/h', color='#2ecc71', alpha=0.8)
        
        # Añadir valores al final de las barras
        for i, (total, dist_14_21, jugador) in enumerate(zip(distancias_total, distancias_14_21, jugadores_ordenados)):
            # Valor total
            total_km = total / 1000
            ax.text(total + total*0.01, i - bar_height/2, f"{total_km:.1f}mil", 
                   ha='left', va='center', fontsize=7, weight='bold', color='#2c3e50')
            
            # Valor 14-21
            dist_14_21_km = dist_14_21 / 1000
            ax.text(dist_14_21 + dist_14_21*0.01, i + bar_height/2, f"{dist_14_21_km:.1f}mil", 
                   ha='left', va='center', fontsize=7, weight='bold', color='#2c3e50')
            
            # Nombre del jugador
            ax.text(-max(distancias_total)*0.02, i, jugador,
                   va='center', ha='right', fontsize=8, color='#1a237e', weight='bold')
        
        # Configurar ejes
        ax.set_yticks([])
        ax.set_xlabel('metros', fontsize=10, color='#2c3e50')
        
        # Ajustar límites
        max_val = max(max(distancias_total), max(distancias_14_21)) if distancias_total else 20000
        ax.set_xlim(-max_val*0.15, max_val * 1.08)
        
        # Leyenda
        ax.legend(loc='lower right', bbox_to_anchor=(0.98, 0.02), 
                 fontsize=8, frameon=True, fancybox=True, shadow=True)
        
        # Estilo
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
    
    def plot_zones(self, ax, jugadores_data):
        """Dibuja el gráfico de distancia por zonas"""
        jugadores = list(jugadores_data.keys())
        if not jugadores:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center')
            ax.axis('off')
            return
        
        # Ordenar jugadores por suma de zonas de alta velocidad
        jugadores_ordenados = sorted(jugadores, 
                                   key=lambda x: jugadores_data[x]['dist_21_24'] + jugadores_data[x]['dist_mas_24'], 
                                   reverse=False)
        
        y_positions = np.arange(len(jugadores_ordenados))
        
        # Datos para las zonas
        distancias_21_24 = [jugadores_data[j]['dist_21_24'] for j in jugadores_ordenados]
        distancias_mas_24 = [jugadores_data[j]['dist_mas_24'] for j in jugadores_ordenados]
        
        # Crear barras apiladas
        bars_21_24 = ax.barh(y_positions, distancias_21_24, 
                           label='21-24 km/h', color='#f39c12', alpha=0.8)
        bars_mas_24 = ax.barh(y_positions, distancias_mas_24, left=distancias_21_24,
                            label='>24 km/h', color='#e74c3c', alpha=0.8)
        
        # Calcular totales para etiquetas
        totales = [d21_24 + dmas_24 for d21_24, dmas_24 in zip(distancias_21_24, distancias_mas_24)]
        
        # Añadir valores
        for i, (d21_24, dmas_24, total, jugador) in enumerate(zip(distancias_21_24, distancias_mas_24, totales, jugadores_ordenados)):
            # Total al final
            if total > 0:
                total_km = total / 1000
                ax.text(total + total*0.02, i, f"{total_km:.3f}mil", 
                       ha='left', va='center', fontsize=7, weight='bold', color='#2c3e50')
            
            # Valores en segmentos si son significativos
            if d21_24 > 200:  # Solo mostrar si es mayor a 200 metros
                d21_24_km = d21_24 / 1000
                ax.text(d21_24/2, i, f"{d21_24_km:.3f}mil", 
                       ha='center', va='center', fontsize=6, weight='bold', color='white')
            
            if dmas_24 > 200:  # Solo mostrar si es mayor a 200 metros
                dmas_24_km = dmas_24 / 1000
                ax.text(d21_24 + dmas_24/2, i, f"{dmas_24_km:.3f}mil", 
                       ha='center', va='center', fontsize=6, weight='bold', color='white')
            
            # Nombre del jugador
            ax.text(-max(totales)*0.02 if totales else -100, i, jugador,
                   va='center', ha='right', fontsize=8, color='#1a237e', weight='bold')
        
        # Configurar ejes
        ax.set_yticks([])
        ax.set_xlabel('metros', fontsize=10, color='#2c3e50')
        
        # Ajustar límites
        max_total = max(totales) if totales else 2000
        ax.set_xlim(-max_total*0.15, max_total * 1.08)
        
        # Leyenda
        ax.legend(loc='lower right', bbox_to_anchor=(0.98, 0.02), 
                 fontsize=8, frameon=True, fancybox=True, shadow=True)
        
        # Estilo
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)

# Funciones auxiliares para selección de equipo y jornadas
def seleccionar_equipo_jornadas_zonas():
    """Permite al usuario seleccionar un equipo y jornadas de forma interactiva"""
    try:
        report_generator = DistanciaZonasReport()
        equipos = report_generator.get_available_teams()
        
        if len(equipos) == 0:
            print("No se encontraron equipos en los datos.")
            return None, None
        
        print("\n=== SELECCIÓN DE EQUIPO - DISTANCIA POR ZONAS ===")
        for i, equipo in enumerate(equipos, 1):
            print(f"{i}. {equipo}")
        
        while True:
            try:
                seleccion = input(f"\nSelecciona un equipo (1-{len(equipos)}): ").strip()
                indice = int(seleccion) - 1
                
                if 0 <= indice < len(equipos):
                    equipo_seleccionado = equipos[indice]
                    break
                else:
                    print(f"Por favor, ingresa un número entre 1 y {len(equipos)}")
            except ValueError:
                print("Por favor, ingresa un número válido")
        
        # Obtener jornadas disponibles
        jornadas_disponibles = report_generator.get_available_jornadas(equipo_seleccionado)
        print(f"\nJornadas disponibles para {equipo_seleccionado}: {jornadas_disponibles}")
        
        # Preguntar cuántas jornadas incluir
        while True:
            try:
                num_jornadas = input(f"¿Cuántas jornadas incluir? (máximo {len(jornadas_disponibles)}): ").strip()
                num_jornadas = int(num_jornadas)
                
                if 1 <= num_jornadas <= len(jornadas_disponibles):
                    jornadas_seleccionadas = sorted(jornadas_disponibles)[-num_jornadas:]
                    break
                else:
                    print(f"Por favor, ingresa un número entre 1 y {len(jornadas_disponibles)}")
            except ValueError:
                print("Por favor, ingresa un número válido")
        
        return equipo_seleccionado, jornadas_seleccionadas
        
    except Exception as e:
        print(f"Error en la selección: {e}")
        return None, None

def main_zonas():
    try:
        print("=== GENERADOR DE REPORTES - DISTANCIA POR ZONAS ===")
        
        # Selección interactiva
        equipo, jornadas = seleccionar_equipo_jornadas_zonas()
        
        if equipo is None or jornadas is None:
            print("No se pudo completar la selección.")
            return
        
        print(f"\nGenerando reporte de zonas para {equipo} - Jornadas: {jornadas}")
        
        # Crear el reporte
        report_generator = DistanciaZonasReport()
        fig = report_generator.create_visualization(equipo, jornadas)
        
        if fig:
            # Mostrar en pantalla
            plt.show()
            
            # Guardar como PDF
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_zonas_{equipo_filename}.pdf"
            
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(output_path) as pdf:
                fig.patch.set_alpha(0.0)
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0, 
                          facecolor='none', edgecolor='none', dpi=300,
                          transparent=True)
            
            print(f"✅ Reporte guardado como: {output_path}")
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()

def generar_reporte_zonas_personalizado(equipo, jornadas, mostrar=True, guardar=True):
    """Función para generar un reporte personalizado de zonas"""
    try:
        report_generator = DistanciaZonasReport()
        fig = report_generator.create_visualization(equipo, jornadas)
        
        if fig:
            if mostrar:
                plt.show()
            
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"reporte_zonas_{equipo_filename}.pdf"
                
                from matplotlib.backends.backend_pdf import PdfPages
                with PdfPages(output_path) as pdf:
                    fig.patch.set_alpha(0.0)
                    pdf.savefig(fig, bbox_inches='tight', pad_inches=0, 
                              facecolor='none', edgecolor='none', dpi=300,
                              transparent=True)
                
                print(f"✅ Reporte guardado como: {output_path}")
            
            return fig
        else:
            print("❌ No se pudo generar la visualización")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Inicialización
print("=== INICIALIZANDO GENERADOR DE REPORTES DE ZONAS ===")
try:
    report_generator = DistanciaZonasReport()
    equipos = report_generator.get_available_teams()
    print(f"\n✅ Sistema listo. Equipos disponibles: {len(equipos)}")
    
    if len(equipos) > 0:
        print("📝 Para generar un reporte ejecuta: main_zonas()")
        print("📝 Para uso directo: generar_reporte_zonas_personalizado('Nombre_Equipo', [33,34,35])")
    
except Exception as e:
    print(f"❌ Error al inicializar: {e}")

if __name__ == "__main__":
    main_zonas()