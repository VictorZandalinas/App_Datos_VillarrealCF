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

class ComparativaSprintsReport:
    def __init__(self, data_path="extraccion_mediacoach/data/rendimiento_fisico.parquet"):
        """
        Inicializa la clase para generar informes comparativos de sprints
        Villarreal CF vs Equipo seleccionado
        """
        self.data_path = data_path
        self.equipo_fijo = "Villarreal CF"  # Siempre en la izquierda
        self.df = None
        self.load_data()
        self.clean_team_names()
        
    def load_data(self):
        """Carga los datos del archivo parquet"""
        try:
            self.df = pd.read_parquet(self.data_path)
        except Exception as e:
            pass
    
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
        
        
    def get_available_teams(self):
        """Retorna la lista de equipos disponibles (excluyendo Villarreal)"""
        if self.df is None:
            return []
        
        all_teams = sorted(self.df['Equipo'].unique())
        # Filtrar equipos que no sean Villarreal
        equipos_disponibles = [equipo for equipo in all_teams 
                              if not 'villarreal' in equipo.lower()]
        return equipos_disponibles
    
    def get_available_jornadas(self, equipo=None):
        """Retorna las jornadas disponibles, opcionalmente filtradas por equipo"""
        if self.df is None:
            return []
        
        if equipo:
            filtered_df = self.df[self.df['Equipo'] == equipo]
            return sorted(filtered_df['Jornada'].unique())
        else:
            return sorted(self.df['Jornada'].unique())
    
    def filter_data(self, equipo_rival, jornadas):
        """Filtra los datos por Villarreal CF + equipo rival y jornadas específicas"""
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
        
        
        # Filtrar por ambos equipos y jornadas
        filtered_df = self.df[
            ((self.df['Equipo'].str.contains('Villarreal', case=False, na=False)) |
             (self.df['Equipo'] == equipo_rival)) & 
            (self.df['Jornada'].isin(normalized_jornadas))
        ].copy()
        
        return filtered_df
    
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
            pass
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
                    pass
                    try:
                        return plt.imread(logo_path)
                    except Exception as e:
                        pass
            print(f"⚠️ Advertencia: El archivo mapeado '{logo_filename}' no fue encontrado.")

        # --- Búsqueda Automática ---
        available_files = [f for f in os.listdir(escudos_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # --- Nivel 2: COINCIDENCIA EXACTA ---
        for filename in available_files:
            file_base_norm = self.normalize_text(os.path.splitext(filename)[0])
            if file_base_norm == equipo_norm:
                logo_path = os.path.join(escudos_dir, filename)
                try:
                    return plt.imread(logo_path)
                except Exception as e:
                    pass

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
                    try:
                        return plt.imread(logo_path)
                    except Exception as e:
                        pass

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
            try:
                return plt.imread(logo_path)
            except Exception as e:
                pass

        print(f"❌ No se encontró un escudo definitivo para: {equipo} (normalizado como: {equipo_norm})")
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
    
    def create_comparative_data(self, filtered_df, jornadas):
        """Procesa los datos de sprints para la comparativa"""
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

        # Separar datos por equipo
        villarreal_df = filtered_df[filtered_df['Equipo'].str.contains('Villarreal', case=False, na=False)]
        rival_df = filtered_df[~filtered_df['Equipo'].str.contains('Villarreal', case=False, na=False)]
        
        # Procesar datos para cada equipo
        equipos_data = {}
        
        for equipo_name, equipo_df in [("Villarreal CF", villarreal_df), ("Rival", rival_df)]:
            if len(equipo_df) == 0:
                continue
                
            # Datos por jornada para gráfico superior comparativo
            jornadas_data = {}
            for jornada in normalized_jornadas:
                jornada_data = equipo_df[equipo_df['Jornada'] == jornada]
                if len(jornada_data) > 0:
                    total_sprints = jornada_data['N Total Sprints >21 km / h'].sum()
                    jornadas_data[jornada] = int(total_sprints)
                else:
                    jornadas_data[jornada] = 0
            
            # Datos por jugador y jornada para gráficos superiores apilados
            jugadores_jornadas = {}
            jugadores_list = equipo_df['Alias'].unique()
            
            for jugador in jugadores_list:
                jugador_data = equipo_df[equipo_df['Alias'] == jugador]
                jugadores_jornadas[jugador] = {
                    'jornadas': {},
                    'total': 0,
                    'dorsal': jugador_data['Dorsal'].iloc[0] if len(jugador_data) > 0 else 'N/A'
                }
                
                for jornada in normalized_jornadas:
                    jornada_data = jugador_data[jugador_data['Jornada'] == jornada]
                    if len(jornada_data) > 0:
                        sprints = int(jornada_data['N Total Sprints >21 km / h'].sum())
                        jugadores_jornadas[jugador]['jornadas'][jornada] = sprints
                        jugadores_jornadas[jugador]['total'] += sprints
                    else:
                        jugadores_jornadas[jugador]['jornadas'][jornada] = 0
            
            # Datos por jugador para gráficos inferiores
            jugadores_data = equipo_df.groupby(['Alias']).agg({
                'N Total Sprints 21-24 km / h': 'sum',
                'N Total Sprints >24 km / h': 'sum',
                'Dorsal': 'first'
            }).reset_index()
            
            jugadores_dict = {}
            for _, row in jugadores_data.iterrows():
                jugadores_dict[row['Alias']] = {
                    'sprints_21_24': int(row['N Total Sprints 21-24 km / h']),
                    'sprints_24': int(row['N Total Sprints >24 km / h']),
                    'dorsal': row['Dorsal']
                }
            
            equipos_data[equipo_name] = {
                'jornadas': jornadas_data,
                'jugadores': jugadores_dict,
                'jugadores_jornadas': jugadores_jornadas,
                'equipo_original': equipo_df['Equipo'].iloc[0] if len(equipo_df) > 0 else equipo_name
            }
        
        return equipos_data, normalized_jornadas
    
    def create_visualization(self, equipo_rival, jornadas, figsize=(16, 11)):
        """Crea la visualización completa comparativa"""
        # Filtrar datos
        filtered_df = self.filter_data(equipo_rival, jornadas)
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
        
        # Configurar grid: header + 2 gráficos superiores + 2 gráficos inferiores
        gs = fig.add_gridspec(3, 2, 
                             height_ratios=[0.08, 0.3, 0.62], 
                             width_ratios=[1, 1], 
                             hspace=0.48, wspace=0.2,
                             left=0.05, right=0.97, top=0.95, bottom=0.05)
        
        # Área del título (toda la fila superior)
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        # Título principal
        ax_title.text(0.5, 0.8, 'COMPARATIVA SPRINTS', 
                     fontsize=24, weight='bold', ha='center', va='center',
                     color='#1e3d59', family='serif')
        ax_title.text(0.5, 0.2, f'ÚLTIMAS {len(jornadas)} JORNADAS', 
                     fontsize=12, ha='center', va='center',
                     color='#2c3e50', weight='bold')
        
        # Mostrar jornadas específicas en la esquina superior derecha
        jornadas_text = "Jornadas " + " ".join([f"{j}" for j in jornadas])
        ax_title.text(0.45, - 0.2, jornadas_text, 
                     fontsize=10, ha='left', va='top',
                     color='#2c3e50', weight='bold')
        
        # Texto "Nº SPRINTS >21km/h" en la parte superior
        ax_title.text(0.43, - 0.7, 'Nº SPRINTS >21km/h', 
                     fontsize=11, ha='left', va='center',
                     color='#1e3d59', weight='bold')
        
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
        
        # Escudos en la derecha (Villarreal adelante, rival atrás)
        # Escudo Villarreal CF (adelante)
        logo_villarreal = self.load_team_logo("Villarreal CF")
        if logo_villarreal is not None:
            try:
                imagebox = OffsetImage(logo_villarreal, zoom=0.45)
                ab = AnnotationBbox(imagebox, (0.90, 0.5), frameon=False, zorder=2)
                ax_title.add_artist(ab)
            except Exception as e:
                print(f"❌ Error al aplicar escudo Villarreal: {e}")
        
        # Escudo rival (un poco a la derecha y atrás)
        logo_rival = self.load_team_logo(equipo_rival)
        if logo_rival is not None:
            try:
                imagebox = OffsetImage(logo_rival, zoom=0.45)
                ab = AnnotationBbox(imagebox, (0.95, 0.5), frameon=False, zorder=1)
                ax_title.add_artist(ab)
            except Exception as e:
                print(f"❌ Error al aplicar escudo rival: {e}")
        
        # Procesar datos
        equipos_data, normalized_jornadas = self.create_comparative_data(filtered_df, jornadas)
        
        # Gráfico superior izquierda: Villarreal CF por jugadores y jornadas
        ax_sup_izq = fig.add_subplot(gs[1, 0])
        ax_sup_izq.set_facecolor('none')
        ax_sup_izq.set_title('VILLARREAL CF', fontsize=12, weight='bold', 
                            color='#1e3d59', pad=10)
        if "Villarreal CF" in equipos_data:
            self.plot_team_sprints_by_jornadas_vertical(ax_sup_izq, equipos_data["Villarreal CF"], normalized_jornadas)
        
        # Gráfico superior derecha: Equipo rival por jugadores y jornadas
        ax_sup_der = fig.add_subplot(gs[1, 1])
        ax_sup_der.set_facecolor('none')
        ax_sup_der.set_title(equipo_rival.upper(), fontsize=12, weight='bold', 
                            color='#1e3d59', pad=10)
        if "Rival" in equipos_data:
            self.plot_team_sprints_by_jornadas_vertical(ax_sup_der, equipos_data["Rival"], normalized_jornadas)
        
        # Gráfico inferior izquierda: Villarreal CF (21-24 vs >24)
        ax_inf_izq = fig.add_subplot(gs[2, 0])
        ax_inf_izq.set_facecolor('none')
        ax_inf_izq.set_title('Nº SPRINTS 21-24km/h\nNº SPRINTS >24km/h', fontsize=10, weight='bold', 
                            color='#1e3d59', pad=4)
        if "Villarreal CF" in equipos_data:
            self.plot_team_sprints_horizontal(ax_inf_izq, equipos_data["Villarreal CF"]['jugadores'], invertir=True)
        
        # Gráfico inferior derecha: Equipo rival (21-24 vs >24)
        ax_inf_der = fig.add_subplot(gs[2, 1])
        ax_inf_der.set_facecolor('none')
        ax_inf_der.set_title('Nº SPRINTS 21-24km/h\nNº SPRINTS >24km/h', fontsize=10, weight='bold', 
                            color='#1e3d59', pad=4)
        if "Rival" in equipos_data:
            self.plot_team_sprints_horizontal(ax_inf_der, equipos_data["Rival"]['jugadores'], invertir=False)
        
        return fig
    
    def plot_team_sprints_by_jornadas_vertical(self, ax, equipo_data, jornadas):
        """Dibuja barras verticales apiladas por jugador y jornadas"""
        if 'jugadores_jornadas' not in equipo_data or not equipo_data['jugadores_jornadas']:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center')
            ax.axis('off')
            return
        
        jugadores_data = equipo_data['jugadores_jornadas']
        
        # Ordenar jugadores por total de sprints (de mayor a menor)
        jugadores_ordenados = sorted(jugadores_data.keys(), 
                                   key=lambda x: jugadores_data[x]['total'], 
                                   reverse=True)
        
        # Limitar a top 15 para que quepan en el gráfico
        jugadores_ordenados = jugadores_ordenados[:15]
        
        # Colores para jornadas (tonos azules y verdes como en la imagen)
        colors = ['#2c5f8a', '#3d7ea6', '#4e9dc2', '#5fbcde', '#70dbfa', '#81faf6', '#92f9d2']
        
        x_positions = np.arange(len(jugadores_ordenados))
        bar_width = 0.6
        
        # Crear barras apiladas
        bottom_values = np.zeros(len(jugadores_ordenados))
        
        for j_idx, jornada in enumerate(jornadas):
            sprints_jornada = []
            for jugador in jugadores_ordenados:
                sprints = jugadores_data[jugador]['jornadas'].get(jornada, 0)
                sprints_jornada.append(sprints)
            
            bars = ax.bar(x_positions, sprints_jornada, bar_width, 
                         bottom=bottom_values, 
                         label=f'{jornada}', 
                         color=colors[j_idx % len(colors)])
            
            # Añadir valores en segmentos grandes
            for i, (bar, sprints) in enumerate(zip(bars, sprints_jornada)):
                if sprints > 3:  # Solo mostrar si es mayor a 3
                    ax.text(bar.get_x() + bar.get_width()/2, 
                           bottom_values[i] + sprints/2, 
                           f"{sprints}", ha='center', va='center', 
                           fontsize=6, weight='bold', color='white')
            
            bottom_values += sprints_jornada
        
        # Añadir totales encima de las barras
        for i, (total, jugador) in enumerate(zip(bottom_values, jugadores_ordenados)):
            if total > 0:
                ax.text(i, total + max(bottom_values)*0.01, f"{int(total)}", 
                       ha='center', va='bottom', fontsize=7, color='#1a237e', weight='bold')
        
        # Configurar ejes
        ax.set_xticks(x_positions)
        ax.set_xticklabels(jugadores_ordenados, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Nº Sprints >21 km/h', fontsize=8, color='#2c3e50')
        
        # Ajustar límites
        if len(bottom_values) > 0 and max(bottom_values) > 0:
            ax.set_ylim(0, max(bottom_values) * 1.05)
        
        # Leyenda compacta
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1), 
                 fontsize=6, ncol=1, frameon=True, fancybox=True, shadow=True)
        
        # Estilo
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.2)
        ax.tick_params(axis='both', labelsize=7)

    def plot_team_sprints_horizontal(self, ax, jugadores_data, invertir=False):
        """Dibuja barras horizontales para sprints de un equipo"""
        if not jugadores_data:
            ax.text(0.5, 0.5, 'No hay datos', ha='center', va='center')
            ax.axis('off')
            return
        
        # Ordenar por total de sprints (de mayor a menor)
        jugadores_ordenados = sorted(jugadores_data.keys(), 
                                key=lambda x: jugadores_data[x]['sprints_21_24'] + jugadores_data[x]['sprints_24'], 
                                reverse=False)
        
        y_positions = np.arange(len(jugadores_ordenados))
        
        # Datos para las zonas
        sprints_21_24 = [jugadores_data[j]['sprints_21_24'] for j in jugadores_ordenados]
        sprints_24 = [jugadores_data[j]['sprints_24'] for j in jugadores_ordenados]
        
        if invertir:
            # Para el gráfico invertido (izquierda), usar valores negativos
            sprints_21_24_plot = [-x for x in sprints_21_24]
            sprints_24_plot = [-x for x in sprints_24]
            left_values = sprints_21_24_plot
        else:
            # Para el gráfico normal (derecha)
            sprints_21_24_plot = sprints_21_24
            sprints_24_plot = sprints_24
            left_values = sprints_21_24
        
        # Crear barras apiladas
        bars_21_24 = ax.barh(y_positions, sprints_21_24_plot, 
                        label='Sprints 21-24 km/h', color='#f39c12', alpha=0.8)
        bars_24 = ax.barh(y_positions, sprints_24_plot, left=left_values,
                        label='Sprints >24 km/h', color='#e74c3c', alpha=0.8)
        
        # Calcular totales
        totales = [s21_24 + s24 for s21_24, s24 in zip(sprints_21_24, sprints_24)]
        
        # Añadir valores
        for i, (s21_24, s24, total, jugador) in enumerate(zip(sprints_21_24, sprints_24, totales, jugadores_ordenados)):
            if invertir:
                # Total al principio (lado izquierdo)
                if total > 0:
                    ax.text(-total - total*0.02, i, f"{int(total)}", 
                        ha='right', va='center', fontsize=8, weight='bold', color='#2c3e50')
                
                # Valores en segmentos (posiciones invertidas)
                if s21_24 > 2:
                    ax.text(-s21_24/2, i, f"{int(s21_24)}", 
                        ha='center', va='center', fontsize=7, weight='bold', color='white')
                
                if s24 > 2:
                    ax.text(-s21_24 - s24/2, i, f"{int(s24)}", 
                        ha='center', va='center', fontsize=7, weight='bold', color='white')
                
                # Nombre del jugador a la derecha
                max_total = max(totales) if totales else 30
                ax.text(max_total*0.02, i, jugador,
                    va='center', ha='left', fontsize=8, color='#1a237e', weight='bold')
            else:
                # Total al final (lado derecho)
                if total > 0:
                    ax.text(total + total*0.02, i, f"{int(total)}", 
                        ha='left', va='center', fontsize=8, weight='bold', color='#2c3e50')
                
                # Valores en segmentos
                if s21_24 > 2:
                    ax.text(s21_24/2, i, f"{int(s21_24)}", 
                        ha='center', va='center', fontsize=7, weight='bold', color='white')
                
                if s24 > 2:
                    ax.text(s21_24 + s24/2, i, f"{int(s24)}", 
                        ha='center', va='center', fontsize=7, weight='bold', color='white')
                
                # Nombre del jugador a la izquierda
                ax.text(-max(totales)*0.02 if totales else -2, i, jugador,
                    va='center', ha='right', fontsize=8, color='#1a237e', weight='bold')
        
        # Configurar ejes
        ax.set_yticks([])
        ax.set_xlabel('Nº Sprints', fontsize=9, color='#2c3e50')
        
        # Ajustar límites
        max_total = max(totales) if totales else 30
        if invertir:
            ax.set_xlim(-max_total * 1.1, max_total*0.15)
        else:
            ax.set_xlim(-max_total*0.15, max_total * 1.1)
        
        # Leyenda
        ax.legend(loc='lower right' if not invertir else 'lower left', 
                bbox_to_anchor=(0.98, 0.02) if not invertir else (0.02, 0.02), 
                fontsize=8, frameon=True, fancybox=True, shadow=True)
        
        # Estilo
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', alpha=0.2)
        ax.tick_params(axis='both', labelsize=8)

# Funciones auxiliares
def seleccionar_equipo_jornadas_comparativa():
    """Permite al usuario seleccionar un equipo rival y jornadas para la comparativa"""
    try:
        report_generator = ComparativaSprintsReport()
        equipos = report_generator.get_available_teams()
        
        if len(equipos) == 0:
            pass
            return None, None
        
        for i, equipo in enumerate(equipos, 1):
            pass
        
        while True:
            try:
                seleccion = input(f"\nSelecciona equipo rival (1-{len(equipos)}): ").strip()
                indice = int(seleccion) - 1
                
                if 0 <= indice < len(equipos):
                    equipo_seleccionado = equipos[indice]
                    break
                else:
                    pass
            except ValueError:
                pass
        
        # Obtener jornadas disponibles
        jornadas_disponibles = report_generator.get_available_jornadas()
        
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
        
        return equipo_seleccionado, jornadas_seleccionadas
        
    except Exception as e:
        pass
        return None, None

def main_comparativa_sprints():
    try:
        pass
        
        # Selección interactiva
        equipo_rival, jornadas = seleccionar_equipo_jornadas_comparativa()
        
        if equipo_rival is None or jornadas is None:
            pass
            return
        
        
        # Crear el reporte
        report_generator = ComparativaSprintsReport()
        fig = report_generator.create_visualization(equipo_rival, jornadas)
        
        if fig:
            # Mostrar en pantalla
            plt.show()
            
            # Guardar como PDF
            equipo_filename = equipo_rival.replace(' ', '_').replace('/', '_')
            output_path = f"comparativa_sprints_Villarreal_vs_{equipo_filename}.pdf"
            
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

def generar_comparativa_sprints_personalizada(equipo_rival, jornadas, mostrar=True, guardar=True):
    """Función para generar una comparativa personalizada de sprints"""
    try:
        report_generator = ComparativaSprintsReport()
        fig = report_generator.create_visualization(equipo_rival, jornadas)
        
        if fig:
            if mostrar:
                plt.show()
            
            if guardar:
                equipo_filename = equipo_rival.replace(' ', '_').replace('/', '_')
                output_path = f"comparativa_sprints_Villarreal_vs_{equipo_filename}.pdf"
                
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

# Inicialización
try:
    report_generator = ComparativaSprintsReport()
    equipos = report_generator.get_available_teams()
    
    if len(equipos) > 0:
        pass
    
except Exception as e:
    print(f"❌ Error al inicializar: {e}")

if __name__ == "__main__":
    main_comparativa_sprints()