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

class VelocidadesMaximasReport:
    def __init__(self, data_path="extraccion_mediacoach/data/rendimiento_fisico.parquet"):
        """
        Inicializa la clase para generar informes de velocidades máximas
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
    
    @staticmethod
    def normalize_text(text):
        """Normaliza texto eliminando acentos, espacios extra y caracteres especiales"""
        # Eliminar acentos
        import unicodedata
        import re
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        # Convertir a minúsculas y limpiar
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
            
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
    
    def calculate_league_averages(self, jornadas):
        """Calcula las medias de la liga para las jornadas especificadas"""
        if self.df is None:
            return {}
        
        # Normalizar jornadas
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
        
        # Filtrar datos de la liga para las jornadas específicas
        league_data = self.df[self.df['Jornada'].isin(normalized_jornadas)]
        
        if len(league_data) == 0:
            return {}
        
        # Calcular medias de la liga
        averages = {
            'vel_max_total': league_data['Velocidad Máxima Total'].mean(),
            'vel_max_1p': league_data['Velocidad Máxima 1P'].mean(),
            'vel_max_2p': league_data['Velocidad Máxima 2P'].mean()
        }
        
        print(f"Medias de la liga calculadas: {averages}")
        return averages
    
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
        # ¡Asegúrate de que este diccionario esté igual en todos tus scripts!
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

        # --- Nivel 3: COINCIDENCIA DE PALABRA LARGA ---
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
    
    def create_velocities_data(self, filtered_df, jornadas):
        """Procesa los datos de velocidades para los gráficos"""
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

        # Procesar datos por jugador y jornada
        jugadores_data = {}
        jugadores_list = filtered_df['Alias'].unique()
        
        for jugador in jugadores_list:
            jugador_data = filtered_df[filtered_df['Alias'] == jugador]
            jugadores_data[jugador] = {
                'jornadas': {},
                'max_total': 0,
                'max_1p': 0,
                'max_2p': 0,
                'dorsal': jugador_data['Dorsal'].iloc[0] if len(jugador_data) > 0 else 'N/A'
            }
            
            for jornada in normalized_jornadas:
                jornada_data = jugador_data[jugador_data['Jornada'] == jornada]
                if len(jornada_data) > 0:
                    row = jornada_data.iloc[0]
                    velocidades = {
                        'vel_max_total': float(row['Velocidad Máxima Total']),
                        'vel_max_1p': float(row['Velocidad Máxima 1P']),
                        'vel_max_2p': float(row['Velocidad Máxima 2P'])
                    }
                    jugadores_data[jugador]['jornadas'][jornada] = velocidades
                    
                    # Actualizar máximos
                    jugadores_data[jugador]['max_total'] = max(jugadores_data[jugador]['max_total'], velocidades['vel_max_total'])
                    jugadores_data[jugador]['max_1p'] = max(jugadores_data[jugador]['max_1p'], velocidades['vel_max_1p'])
                    jugadores_data[jugador]['max_2p'] = max(jugadores_data[jugador]['max_2p'], velocidades['vel_max_2p'])
                else:
                    jugadores_data[jugador]['jornadas'][jornada] = {
                        'vel_max_total': 0, 'vel_max_1p': 0, 'vel_max_2p': 0
                    }
        
        return jugadores_data, normalized_jornadas
    
    def create_visualization(self, equipo, jornadas, figsize=(16, 11)):
        """Crea la visualización completa de velocidades máximas"""
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
        
        # Configurar grid: header + gráfico grande izq + 2 gráficos pequeños der
        gs = fig.add_gridspec(3, 2, 
                             height_ratios=[0.08, 0.5, 0.42], 
                             width_ratios=[1.2, 0.8], 
                             hspace=0.65, wspace=0.2,
                             left=0.03, right=0.97, top=0.95, bottom=0.05)
        
        # Área del título (toda la fila superior)
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        # Título principal
        ax_title.text(0.5, 0.8, 'VEL. MÁXIMAS', 
                     fontsize=24, weight='bold', ha='center', va='center',
                     color='#1e3d59', family='serif')
        ax_title.text(0.5, 0.2, f'ÚLTIMAS {len(jornadas)} JORNADAS', 
                     fontsize=12, ha='center', va='center',
                     color='#2c3e50', weight='bold')
        
        # Texto "10 JUGADORES MÁS RÁPIDOS" centrado
        ax_title.text(0.5, 0.5, '10 JUGADORES MÁS RÁPIDOS', 
                     fontsize=14, ha='center', va='center',
                     color='#1e3d59', weight='bold')
        
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
        
        # Procesar datos y calcular medias
        jugadores_data, normalized_jornadas = self.create_velocities_data(filtered_df, jornadas)
        league_averages = self.calculate_league_averages(jornadas)
        
        # Gráfico grande izquierda: VELOCIDAD MÁXIMA (ocupa 2 filas)
        ax_main = fig.add_subplot(gs[1:, 0])
        ax_main.set_facecolor('none')
        ax_main.set_title('VELOCIDAD MÁXIMA', fontsize=14, weight='bold', 
                         color='white', pad=15, 
                         bbox=dict(boxstyle="round,pad=0.5", facecolor='#1e3d59', alpha=0.8))
        self.plot_velocidad_maxima_vertical(ax_main, jugadores_data, normalized_jornadas, 
                                          league_averages.get('vel_max_total', 0), 'vel_max_total')
        
        # Gráfico superior derecha: VELOCIDAD MÁXIMA 1er TIEMPO
        ax_sup_der = fig.add_subplot(gs[1, 1])
        ax_sup_der.set_facecolor('none')
        ax_sup_der.set_title('VELOCIDAD MÁXIMA 1er TIEMPO', fontsize=12, weight='bold', 
                            color='white', pad=10,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='#1e3d59', alpha=0.8))
        self.plot_velocidad_maxima_vertical(ax_sup_der, jugadores_data, normalized_jornadas, 
                                          league_averages.get('vel_max_1p', 0), 'vel_max_1p')
        
        # Gráfico inferior derecha: VELOCIDAD MÁXIMA 2o TIEMPO
        ax_inf_der = fig.add_subplot(gs[2, 1])
        ax_inf_der.set_facecolor('none')
        ax_inf_der.set_title('VELOCIDAD MÁXIMA 2o TIEMPO', fontsize=12, weight='bold', 
                            color='white', pad=10,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='#1e3d59', alpha=0.8))
        self.plot_velocidad_maxima_vertical(ax_inf_der, jugadores_data, normalized_jornadas, 
                                          league_averages.get('vel_max_2p', 0), 'vel_max_2p')
        
        return fig
    
    def plot_velocidad_maxima_vertical(self, ax, jugadores_data, jornadas, league_average, metric):
        """Dibuja barras verticales para velocidades máximas con línea de media y etiquetado inteligente."""
        if not jugadores_data:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center')
            ax.axis('off')
            return
        
        # Determinar qué métrica usar
        if metric == 'vel_max_total':
            sort_key, data_key = 'max_total', 'vel_max_total'
        elif metric == 'vel_max_1p':
            sort_key, data_key = 'max_1p', 'vel_max_1p'
        else: # metric == 'vel_max_2p'
            sort_key, data_key = 'max_2p', 'vel_max_2p'
        
        # Ordenar jugadores por velocidad máxima (de mayor a menor) y tomar top 10
        jugadores_ordenados = sorted(jugadores_data.keys(), 
                                   key=lambda x: jugadores_data[x][sort_key], 
                                   reverse=True)[:10]
        
        # --- PASO 1: Pre-calcular la velocidad máxima para cada jugador en la selección ---
        max_velocities_por_jugador = {}
        for jugador in jugadores_ordenados:
            max_vel = 0
            for jornada in jornadas:
                vel = jugadores_data[jugador]['jornadas'].get(jornada, {}).get(data_key, 0)
                if vel > max_vel:
                    max_vel = vel
            max_velocities_por_jugador[jugador] = max_vel
        
        # Colores para jornadas
        colors = ['#9bc53d', '#7fb142', '#659d47', '#4b894c', '#317551', '#3a7ca8', '#1f5f99', '#05428a']
        
        x_positions = np.arange(len(jugadores_ordenados))
        num_jornadas = len(jornadas)
        bar_width = 0.6
        bar_width_individual = bar_width / num_jornadas if num_jornadas > 0 else bar_width
        
        # Crear barras agrupadas por jornada
        for j_idx, jornada in enumerate(jornadas):
            x_offset = x_positions + (j_idx - num_jornadas/2) * bar_width_individual + bar_width_individual/2
            velocidades_jornada = [jugadores_data[jugador]['jornadas'].get(jornada, {}).get(data_key, 0) for jugador in jugadores_ordenados]
            
            bars = ax.bar(x_offset, velocidades_jornada, bar_width_individual, 
                         label=f'J{jornada}', 
                         color=colors[j_idx % len(colors)], alpha=0.8)
            
            # --- PASO 2: Lógica de etiquetado modificada ---
            for i, (bar, velocidad) in enumerate(zip(bars, velocidades_jornada)):
                if velocidad > 0:
                    jugador_actual = jugadores_ordenados[i]
                    max_vel_jugador = max_velocities_por_jugador.get(jugador_actual, 0)

                    # Si la velocidad de esta barra es la máxima del jugador
                    if velocidad >= max_vel_jugador:
                        # Poner el texto ENCIMA de la barra
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                               f"{velocidad:.1f}", ha='center', va='bottom', 
                               fontsize=7, weight='bold', color='#2c3e50')
                    else:
                        # Si NO es la máxima, poner el texto DENTRO y en VERTICAL
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                               f"{velocidad:.1f}", ha='center', va='center', 
                               fontsize=6, weight='bold', color='white', rotation=90)

        # Línea discontinua para la media de la liga
        if league_average > 0:
            ax.axhline(y=league_average, color='#e91e63', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(len(jugadores_ordenados) - 0.2, league_average, f"{league_average:.1f}", 
                   ha='center', va='center', fontsize=8, weight='bold', color='black',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='#e91e63', alpha=0.8, edgecolor='none'))
        
        # Configurar ejes
        ax.set_xticks(x_positions)
        ax.set_xticklabels(jugadores_ordenados, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('km/h', fontsize=10, color='#2c3e50')
        
        # Ajustar límites
        if jugadores_data:
            all_velocities = [max_velocities_por_jugador.get(j, 0) for j in jugadores_ordenados]
            if any(all_velocities):
                max_vel = max(all_velocities)
                ax.set_ylim(0, max_vel * 1.15) # Un poco más de margen para el texto superior
        
        # Leyenda para el gráfico principal
        if metric == 'vel_max_total':
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1), 
                     fontsize=8, ncol=1, frameon=True, fancybox=True, shadow=True)
        
        # Estilo
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)

# Funciones auxiliares
def seleccionar_equipo_jornadas_velocidades():
    """Permite al usuario seleccionar un equipo y jornadas para velocidades"""
    try:
        report_generator = VelocidadesMaximasReport()
        equipos = report_generator.get_available_teams()
        
        if len(equipos) == 0:
            print("No se encontraron equipos en los datos.")
            return None, None
        
        print("\n=== SELECCIÓN DE EQUIPO - VEL. MÁXIMAS ===")
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

def main_velocidades():
    try:
        print("=== GENERADOR DE REPORTES - VEL. MÁXIMAS ===")
        
        # Selección interactiva
        equipo, jornadas = seleccionar_equipo_jornadas_velocidades()
        
        if equipo is None or jornadas is None:
            print("No se pudo completar la selección.")
            return
        
        print(f"\nGenerando reporte de velocidades para {equipo} - Jornadas: {jornadas}")
        
        # Crear el reporte
        report_generator = VelocidadesMaximasReport()
        fig = report_generator.create_visualization(equipo, jornadas)
        
        if fig:
            # Mostrar en pantalla
            plt.show()
            
            # Guardar como PDF
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_velocidades_{equipo_filename}.pdf"
            
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

def generar_reporte_velocidades_personalizado(equipo, jornadas, mostrar=True, guardar=True):
    """Función para generar un reporte personalizado de velocidades"""
    try:
        report_generator = VelocidadesMaximasReport()
        fig = report_generator.create_visualization(equipo, jornadas)
        
        if fig:
            if mostrar:
                plt.show()
            
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"reporte_velocidades_{equipo_filename}.pdf"
                
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
print("=== INICIALIZANDO GENERADOR DE REPORTES DE VELOCIDADES MÁXIMAS ===")
try:
    report_generator = VelocidadesMaximasReport()
    equipos = report_generator.get_available_teams()
    print(f"\n✅ Sistema listo. Equipos disponibles: {len(equipos)}")
    
    if len(equipos) > 0:
        print("📝 Para generar un reporte ejecuta: main_velocidades()")
        print("📝 Para uso directo: generar_reporte_velocidades_personalizado('Nombre_Equipo', [33,34,35])")
    
except Exception as e:
    print(f"❌ Error al inicializar: {e}")

if __name__ == "__main__":
    main_velocidades()