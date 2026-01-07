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
import unicodedata
import re
import warnings
warnings.filterwarnings('ignore')

class MinutosJugadosReport:
    def __init__(self, data_path="extraccion_mediacoach/data/rendimiento_fisico.parquet"):
        """
        Inicializa la clase para generar informes de minutos jugados
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
                # Extraer el número de la jornada
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
    
    def create_minutes_table(self, filtered_df, jornadas):
        """Crea la tabla de minutos jugados por tiempo"""
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

        # Agrupar por jugador y jornada, INCLUYENDO EL DORSAL
        pivot_data = filtered_df.groupby(['Alias', 'Jornada']).agg({
            'Minutos jugados 1P': 'sum',
            'Minutos jugados 2P': 'sum',
            'Minutos jugados': 'sum',
            'Dorsal': 'first'  # AGREGAR ESTA LÍNEA
        }).reset_index()
        
        print(f"Datos agrupados: {len(pivot_data)} registros")
        print(f"Jugadores únicos: {pivot_data['Alias'].unique()}")

        # Crear tabla pivoteada para cada tipo de minutos
        table_data = {}
        for jugador in pivot_data['Alias'].unique():
            player_data = pivot_data[pivot_data['Alias'] == jugador]
            
            # Obtener el dorsal del jugador (tomar el primer valor disponible)
            dorsal_value = player_data['Dorsal'].iloc[0] if len(player_data) > 0 and 'Dorsal' in player_data.columns else 'N/A'
            dorsal = int(dorsal_value) if dorsal_value != 'N/A' and pd.notna(dorsal_value) else 'N/A'
            
            table_data[jugador] = {
                'dorsal': dorsal,  # AGREGAR ESTA LÍNEA
                'jornadas': {}
            }
            
            for jornada in normalized_jornadas:
                jornada_data = player_data[player_data['Jornada'] == jornada]
                if len(jornada_data) > 0:
                    row = jornada_data.iloc[0]
                    table_data[jugador]['jornadas'][jornada] = {
                        '1er_tiempo': int(row['Minutos jugados 1P']),
                        '2do_tiempo': int(row['Minutos jugados 2P']),
                        'total': int(row['Minutos jugados'])
                    }
                else:
                    table_data[jugador]['jornadas'][jornada] = {
                        '1er_tiempo': 0,
                        '2do_tiempo': 0,
                        'total': 0
                    }
        
        print(f"Tabla creada para {len(table_data)} jugadores")
        return table_data
    
    def create_visualization(self, equipo, jornadas, figsize=(16, 11)):
        """Crea la visualización completa del informe optimizada para PDF"""
        # Filtrar datos
        filtered_df = self.filter_data(equipo, jornadas)
        if filtered_df is None or len(filtered_df) == 0:
            print("No hay datos para los filtros especificados")
            return None
        
        # Crear figura con tamaño A4 landscape
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        # Cargar y establecer fondo que ocupe toda la figura
        background = self.load_background()
        if background is not None:
            try:
                # Usar la figura completa para el fondo
                ax_background = fig.add_axes([0, 0, 1, 1], zorder=-1)
                ax_background.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25, zorder=-1)
                
                # AGREGAR ESTAS LÍNEAS PARA ELIMINAR EJES:
                ax_background.axis('off')
                ax_background.set_xticks([])
                ax_background.set_yticks([])
                for spine in ax_background.spines.values():
                    spine.set_visible(False)
                
                print("Fondo aplicado correctamente")
            except Exception as e:
                print(f"Error al aplicar fondo: {e}")
        
        # Configurar grid: 2 tablas + 1 gráfico
        gs = fig.add_gridspec(2, 3, 
                             height_ratios=[0.08, 1], 
                             width_ratios=[1, 1, 0.8], 
                             hspace=0.12, wspace=0.05, # distancia horizontal
                             left=0.03, right=0.97, top=0.97, bottom=0.05)
        
        # Área del título (toda la fila superior)
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        # Título principal centrado más arriba
        ax_title.text(0.5, 0.8, 'MINUTOS JUGADOS', 
                     fontsize=24, weight='bold', ha='center', va='center',
                     color='#1e3d59', family='serif')
        ax_title.text(0.5, 0.2, f'ÚLTIMAS {len(jornadas)} JORNADAS', 
                     fontsize=12, ha='center', va='center',
                     color='#2c3e50', weight='bold')
        
        # Balón más grande arriba izquierda
        ball = self.load_ball_image()
        if ball is not None:
            try:
                imagebox = OffsetImage(ball, zoom=0.15)  # Más grande
                ab = AnnotationBbox(imagebox, (0.05, 0.5), frameon=False)
                ax_title.add_artist(ab)
                print("Balón aplicado correctamente")
            except Exception as e:
                print(f"Error al aplicar balón: {e}")
        
        # Escudo más pequeño arriba derecha
        logo = self.load_team_logo(equipo)
        if logo is not None:
            try:
                imagebox = OffsetImage(logo, zoom=0.40)  # Más pequeño
                ab = AnnotationBbox(imagebox, (0.97, 0.5), frameon=False)
                ax_title.add_artist(ab)
                print("Logo aplicado correctamente")
            except Exception as e:
                print(f"Error al aplicar logo: {e}")
        
        # Crear tabla de minutos
        table_data = self.create_minutes_table(filtered_df, jornadas)
        
        # Dividir jugadores en dos mitades
        jugadores = list(table_data.keys())
        # Ordenar por total de minutos primero
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
        
        jugadores_con_totales = []
        for jugador in jugadores:
            total = sum(table_data[jugador]['jornadas'][j]['total'] for j in normalized_jornadas)
            jugadores_con_totales.append((jugador, total))
        jugadores_con_totales.sort(key=lambda x: x[1], reverse=True)  # De mayor a menor
        jugadores_ordenados = [item[0] for item in jugadores_con_totales]
        
        # Dividir en dos mitades
        mitad = len(jugadores_ordenados) // 2
        primera_mitad = jugadores_ordenados[:mitad]
        segunda_mitad = jugadores_ordenados[mitad:]
        
        # Tabla izquierda (primera mitad de jugadores)
        ax_table1 = fig.add_subplot(gs[1, 0])
        ax_table1.set_facecolor('white')
        ax_table1.set_title('MINUTOS JUGADOS POR TIEMPO', fontsize=12, weight='bold', 
                           color='#1e3d59', pad=8)
        self.plot_half_table(ax_table1, table_data, jornadas, primera_mitad)
        
        # Tabla derecha (segunda mitad de jugadores)
        ax_table2 = fig.add_subplot(gs[1, 1])
        ax_table2.set_facecolor('white')
        ax_table2.set_title('MINUTOS JUGADOS POR TIEMPO', fontsize=12, weight='bold', 
                           color='#1e3d59', pad=8)
        self.plot_half_table(ax_table2, table_data, jornadas, segunda_mitad)
        
        # Gráfico de barras por jornada (columna derecha)
        ax_bars = fig.add_subplot(gs[1, 2])
        ax_bars.set_facecolor('white')
        ax_bars.set_title('MINUTOS JUGADOS\nPOR JORNADA', fontsize=12, weight='bold', 
                         color='#1e3d59', pad=8)
        self.plot_stacked_bars(ax_bars, table_data, jornadas)
        
        return fig
    
    def plot_half_table(self, ax, table_data, jornadas, jugadores_lista):
        """Dibuja una tabla con la mitad de jugadores especificada con columna separada para 1P/2P/TOT"""
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
        
        if len(jugadores_lista) == 0:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center', fontsize=10)
            ax.axis('off')
            return
        
        # Nueva estructura: Jugador | Tipo | Jornada1 | Jornada2 | ... | Total
        col_widths = [0.3, 0.08]  # Jugador más ancho, Tipo (1P/2P/TOT)
        col_widths.extend([0.06] * len(normalized_jornadas))  # Jornadas más estrechas
        col_widths.append(0.08)  # Total
        
        # Normalizar anchos para que sumen 1
        total_width = sum(col_widths)
        col_widths = [w/total_width * 0.98 for w in col_widths]
        
        # Cada jugador tiene 3 filas + 1 fila de header
        total_rows = len(jugadores_lista) * 3 + 1
        
        # Calcular dimensiones
        available_height = 0.92
        cell_height = available_height / (total_rows - 0.5)  # Filas un poco más altas
        
        # Dibujar header
        headers = ['Jugador', 'Tipo'] + [f'{j}' for j in normalized_jornadas] + ['Total']
        colors = ['#2c3e50', '#34495e'] + ['#3498db'] * len(normalized_jornadas) + ['#8e44ad']
        
        y_header = available_height
        for i, (header, color, width) in enumerate(zip(headers, colors, col_widths)):
            x_pos = sum(col_widths[:i])
            rect = patches.Rectangle((x_pos, y_header), 
                                width, cell_height, 
                                linewidth=1, edgecolor='black', 
                                facecolor=color)
            ax.add_patch(rect)
            ax.text(x_pos + width/2, y_header + cell_height/2, header, 
                ha='center', va='center', fontsize=8, weight='bold', color='white')
        
        # Dibujar datos por jugador
        current_row = 1
        for jugador in jugadores_lista:
            # Obtener dorsal del jugador
            dorsal_value = table_data[jugador].get('dorsal', 'N/A')
            dorsal = int(dorsal_value) if dorsal_value != 'N/A' and isinstance(dorsal_value, (int, float)) and not pd.isna(dorsal_value) else 'N/A'
            
            # Calcular totales del jugador - ACTUALIZAR REFERENCIAS
            total_1p = sum(table_data[jugador]['jornadas'][j]['1er_tiempo'] for j in normalized_jornadas)
            total_2p = sum(table_data[jugador]['jornadas'][j]['2do_tiempo'] for j in normalized_jornadas)
            total_general = sum(table_data[jugador]['jornadas'][j]['total'] for j in normalized_jornadas)
            
            y_jugador = available_height - current_row * cell_height
            
            # Nombre del jugador (3 filas de altura) - fondo diferenciado
            rect = patches.Rectangle((0, y_jugador - 2 * cell_height), 
                                col_widths[0], cell_height * 3,
                                linewidth=2, edgecolor='#2c3e50',
                                facecolor='#ecf0f1')  # Fondo gris claro diferenciado
            ax.add_patch(rect)
            
            # AGREGAR DORSAL ESTILO CAMISETA DE FÚTBOL
            if dorsal != 'N/A':
                # Posición del dorsal (esquina superior izquierda)
                dorsal_x = col_widths[0] * 0.12
                dorsal_y = y_jugador - cell_height * 0.25
                
                # Dorsal con contorno estilo fútbol profesional
                ax.text(dorsal_x, dorsal_y, str(dorsal),
                       ha='center', va='center', fontsize=18, weight='black',
                       color='#ffffff',  # Blanco principal
                       path_effects=[
                           patheffects.Stroke(linewidth=4, foreground='#2c3e50'),  # Contorno oscuro grueso
                           patheffects.Normal()
           ])
            # Nombre del jugador ajustado para no superponerse con el dorsal
            nombre_x_offset = col_widths[0] * 0.08 if dorsal != 'N/A' else 0  # Desplazamiento si hay dorsal
            ax.text(col_widths[0]/2 + nombre_x_offset, y_jugador - cell_height, jugador,
                ha='center', va='center', fontsize=8, weight='bold',
                color='#1a237e')
            
            # Etiquetas y datos para cada fila
            tipos = ['1P', '2P', 'TOT']
            tipo_colors = ['#3498db', '#e74c3c', '#27ae60']
            datos_por_tipo = [
                [table_data[jugador]['jornadas'][j]['1er_tiempo'] for j in normalized_jornadas] + [total_1p],
                [table_data[jugador]['jornadas'][j]['2do_tiempo'] for j in normalized_jornadas] + [total_2p],
                [table_data[jugador]['jornadas'][j]['total'] for j in normalized_jornadas] + [total_general]
            ]
            
            for i, (tipo, tipo_color, datos) in enumerate(zip(tipos, tipo_colors, datos_por_tipo)):
                y_fila = y_jugador - i * cell_height
                
                # Columna de tipo
                rect = patches.Rectangle((col_widths[0], y_fila), 
                                    col_widths[1], cell_height,
                                    linewidth=1, edgecolor='black',
                                    facecolor=tipo_color)
                ax.add_patch(rect)
                ax.text(col_widths[0] + col_widths[1]/2, y_fila + cell_height/2, tipo,
                    ha='center', va='center', fontsize=7, weight='bold', color='white')
                
                # Datos por jornada
                for j_idx, valor in enumerate(datos[:-1]):  # Excluir el total
                    x_pos = sum(col_widths[:j_idx+2])  # +2 porque incluye jugador y tipo
                    
                    # Color según tipo y valor
                    if tipo == 'TOT':
                        if valor >= 90:
                            color = '#27ae60'
                            text_color = 'white'
                        elif valor >= 45:
                            color = '#f39c12'
                            text_color = 'white'
                        elif valor > 0:
                            color = '#e74c3c'
                            text_color = 'white'
                        else:
                            color = '#bdc3c7'
                            text_color = 'black'
                    else:
                        color = '#d5dbdb' if valor > 0 else '#ecf0f1'
                        text_color = 'black'
                    
                    rect = patches.Rectangle((x_pos, y_fila), col_widths[j_idx+2], cell_height,
                                        linewidth=1, edgecolor='black', facecolor=color)
                    ax.add_patch(rect)
                    ax.text(x_pos + col_widths[j_idx+2]/2, y_fila + cell_height/2, str(valor),
                        ha='center', va='center', fontsize=8, color=text_color, weight='bold' if tipo == 'TOT' else 'normal')
                
                # Columna total
                x_total = sum(col_widths[:-1])
                total_val = datos[-1]
                
                if tipo == 'TOT':
                    total_color = '#8e44ad'
                    total_text_color = 'white'
                else:
                    total_color = '#bdc3c7'
                    total_text_color = 'black'
                
                rect = patches.Rectangle((x_total, y_fila), col_widths[-1], cell_height,
                                    linewidth=1, edgecolor='black', facecolor=total_color)
                ax.add_patch(rect)
                ax.text(x_total + col_widths[-1]/2, y_fila + cell_height/2, str(total_val),
                    ha='center', va='center', fontsize=8, color=total_text_color, weight='bold')
            
            current_row += 3
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, available_height + cell_height)
        ax.axis('off')
        ax.patch.set_facecolor('white')
    
    def plot_stacked_bars(self, ax, table_data, jornadas):
        """Dibuja el gráfico de barras apiladas por jornada con nombres resaltados y leyenda"""
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
        
        # Preparar datos para el gráfico
        jugadores = list(table_data.keys())
        if not jugadores:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center')
            ax.axis('off')
            return
        
        # Calcular totales por jugador y ordenar de mayor a menor
        jugadores_con_totales = []
        for jugador in jugadores:
            total = sum(table_data[jugador]['jornadas'][j]['total'] for j in normalized_jornadas)
            jugadores_con_totales.append((jugador, total))
        
        # Ordenar por total de minutos (de mayor a menor) y invertir para mostrar descendente
        jugadores_con_totales.sort(key=lambda x: x[1], reverse=True)
        jugadores_ordenados = [item[0] for item in jugadores_con_totales]
        jugadores_ordenados.reverse()  # Invertir para que sea descendente en el gráfico
        
        # Colores para cada jornada - paleta más vibrante
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
        
        y_positions = np.arange(len(jugadores_ordenados))
        bar_width = 0.7
        
        # Preparar datos para barras apiladas por jornada
        left_values = np.zeros(len(jugadores_ordenados))  # Para apilar las barras
        
        # Crear barras apiladas para cada jornada
        for j_idx, jornada in enumerate(normalized_jornadas):
            minutos_jornada = []
            for jugador in jugadores_ordenados:
                minutos_jornada.append(table_data[jugador]['jornadas'][jornada]['total'])
            
            # Crear barra apilada
            bars = ax.barh(y_positions, minutos_jornada, bar_width, 
                        left=left_values, 
                        label=f'Jornada {jornada}', 
                        color=colors[j_idx % len(colors)])
            
            # Añadir valores en cada segmento (solo si hay suficiente espacio)
            for i, (bar, minutos) in enumerate(zip(bars, minutos_jornada)):
                if minutos > 12:  # Solo mostrar si hay suficiente espacio
                    ax.text(left_values[i] + minutos/2, bar.get_y() + bar.get_height()/2, 
                        str(minutos), ha='center', va='center', fontsize=7, weight='bold',
                        color='white')
            
            # Actualizar left_values para la siguiente jornada
            left_values += minutos_jornada
        
        # Añadir etiquetas con los totales al final de cada barra
        for i, (total, jugador) in enumerate(zip(left_values, jugadores_ordenados)):
            if total > 0:
                # Añadir el total al final de la barra
                ax.text(total + 2, i, str(int(total)), 
                    va='center', fontsize=9, color='#1a237e', weight='bold')
                
                # NOMBRES MÁS RESALTADOS - Mejorado significativamente
                ax.text(2, i + 0.25, jugador,  
                    va='bottom', ha='left', 
                    fontsize=9,  # Aumentado de 9 a 11
                    color='#1a237e',  # Color más oscuro y contrastante
                    weight='bold',  # Texto en negrita
                    bbox=dict(
                        facecolor='#ffffff',  # Fondo blanco sólido
                        alpha=0.75,  # Más opaco
                        edgecolor='#2c3e50',  # Borde visible
                        linewidth=1.0,  # Grosor del borde
                        pad=1.5,  # Más padding
                        boxstyle='round,pad=0.15'  # Bordes más redondeados
                    ),
                    # Efecto de sombra para mayor resalte
                    path_effects=[
                        patheffects.withStroke(linewidth=1, foreground='white')
                    ])
        
        # Configurar ejes
        ax.set_yticks([])  # Eliminar ticks del eje Y
        ax.set_yticklabels([])  # Asegurarse de que no hay etiquetas en Y
        ax.set_xticks([])  # Eliminar ticks del eje X
        ax.set_xticklabels([])  # Asegurarse de que no hay etiquetas en X
        
        # Ajustar límite X según el máximo total
        max_total = max(left_values) if len(left_values) > 0 else 100
        ax.set_xlim(0, max_total * 1.15)  # Añadir margen
        
        # Eliminar todos los bordes
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Desactivar todos los ticks
        ax.tick_params(axis='both', which='both', length=0)
        
        # Eliminar la cuadrícula
        ax.grid(False)
        
        # Asegurarse de que no hay espacio en blanco alrededor
        ax.margins(0)
        ax.set_frame_on(False)
        
        # AÑADIR LEYENDA - Nueva funcionalidad
        # Posicionar la leyenda en la esquina superior derecha
        legend = ax.legend(
            loc='upper right',  # Posición
            bbox_to_anchor=(0.98, 0.18),  # Ajuste fino de posición
            frameon=True,  # Mostrar marco
            fancybox=True,  # Bordes redondeados
            shadow=True,  # Sombra
            ncol=1,  # Una columna
            fontsize=8,  # Tamaño de fuente
            title='JORNADAS',  # Título de la leyenda
            title_fontsize=9,  # Tamaño del título
            markerscale=0.8,  # Tamaño de los marcadores
            columnspacing=0.5,  # Espacio entre columnas
            handletextpad=0.3,  # Espacio entre marcador y texto
            borderpad=0.5,  # Padding interno
            handlelength=1.2  # Longitud del marcador
        )
        
        # Personalizar el marco de la leyenda
        legend.get_frame().set_facecolor('#f8f9fa')  # Fondo gris muy claro
        legend.get_frame().set_edgecolor('#2c3e50')  # Borde oscuro
        legend.get_frame().set_linewidth(1)  # Grosor del borde
        legend.get_frame().set_alpha(0.95)  # Transparencia
        
        # Personalizar el título de la leyenda
        legend.get_title().set_color('#1a237e')
        legend.get_title().set_weight('bold')

# Función para seleccionar equipo interactivamente
def seleccionar_equipo_interactivo():
    """
    Permite al usuario seleccionar un equipo de forma interactiva
    """
    try:
        report_generator = MinutosJugadosReport()
        equipos = report_generator.get_available_teams()
        
        if len(equipos) == 0:
            print("No se encontraron equipos en los datos.")
            return None, None
        
        print("\n=== SELECCIÓN DE EQUIPO ===")
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
        
        # Obtener jornadas disponibles para el equipo seleccionado
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

# Ejemplo de uso
def main():
    try:
        print("=== GENERADOR DE REPORTES DE MINUTOS JUGADOS ===")
        
        # Selección interactiva
        equipo, jornadas = seleccionar_equipo_interactivo()
        
        if equipo is None or jornadas is None:
            print("No se pudo completar la selección.")
            return
        
        print(f"\nGenerando reporte para {equipo} - Jornadas: {jornadas}")
        
        # Crear el reporte
        report_generator = MinutosJugadosReport()
        fig = report_generator.create_visualization(equipo, jornadas)
        
        if fig:
            # Mostrar en pantalla
            plt.show()
            
            # Guardar como PDF
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_minutos_{equipo_filename}.pdf"
            
            # Configuración optimizada para PDF
            plt.rcParams['savefig.transparent'] = True
            plt.rcParams['savefig.facecolor'] = 'none'
            plt.rcParams['savefig.edgecolor'] = 'none'
            plt.rcParams['savefig.bbox'] = 'tight'
            plt.rcParams['savefig.pad_inches'] = 0
            
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(output_path) as pdf:
                # Asegurarse de que la figura no tenga bordes
                fig.patch.set_alpha(0.0)
                # Guardar la figura sin bordes
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0, 
                          facecolor='none', edgecolor='none', dpi=300,
                          transparent=True)
            # En la función main() o donde llames al reporte, agrega:
            print("=== DEBUG JORNADAS ===")
            report_generator = MinutosJugadosReport()
            print(f"Jornadas únicas en datos originales: {sorted(report_generator.df['Jornada'].unique())}")
            print(f"Tipos de jornadas: {[type(j) for j in report_generator.df['Jornada'].unique()]}")

            # Después del filtrado:
            filtered_df = report_generator.filter_data(equipo, jornadas)
            print(f"Jornadas después del filtro: {sorted(filtered_df['Jornada'].unique())}")
            print(f"Filas por jornada: {filtered_df['Jornada'].value_counts()}")
            print(f"✅ Reporte guardado como: {output_path}")
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()

# Función para uso directo con parámetros
def generar_reporte_personalizado(equipo, jornadas, mostrar=True, guardar=True):
    """
    Función para generar un reporte personalizado
    
    Args:
        equipo (str): Nombre del equipo
        jornadas (list): Lista de jornadas a incluir
        mostrar (bool): Si mostrar el gráfico en pantalla
        guardar (bool): Si guardar como PDF
    """
    try:
        report_generator = MinutosJugadosReport()
        fig = report_generator.create_visualization(equipo, jornadas)
        
        if fig:
            if mostrar:
                plt.show()
            
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"reporte_minutos_{equipo_filename}.pdf"
                
                # Configuración optimizada para PDF
                plt.rcParams['savefig.transparent'] = True
                plt.rcParams['savefig.facecolor'] = 'none'
                plt.rcParams['savefig.edgecolor'] = 'none'
                plt.rcParams['savefig.bbox'] = 'tight'
                plt.rcParams['savefig.pad_inches'] = 0
                
                from matplotlib.backends.backend_pdf import PdfPages
                with PdfPages(output_path) as pdf:
                    # Asegurarse de que la figura no tenga bordes
                    fig.patch.set_alpha(0.0)
                    # Guardar la figura sin bordes
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

# Verificar archivos necesarios
def verificar_assets():
    """Verifica que existan los directorios y archivos necesarios"""
    print("\n=== VERIFICACIÓN DE ASSETS ===")
    
    # Verificar directorios
    dirs_to_check = ['assets', 'assets/escudos', 'data']
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"✅ Directorio encontrado: {dir_path}")
        else:
            print(f"❌ Directorio faltante: {dir_path}")
            
    # Verificar archivos clave
    files_to_check = [
        'data/rendimiento_fisico.parquet',
        'assets/fondo_informes.png',
        'assets/balon.png'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ Archivo encontrado: {file_path}")
        else:
            print(f"❌ Archivo faltante: {file_path}")
    
    # Verificar escudos
    if os.path.exists('assets/escudos'):
        escudos = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
        print(f"✅ Escudos disponibles ({len(escudos)}): {escudos}")
    else:
        print("❌ No se encontró el directorio de escudos")

# Crear instancia y verificar todo al importar
print("=== INICIALIZANDO GENERADOR DE REPORTES ===")
try:
    verificar_assets()
    report_generator = MinutosJugadosReport()
    equipos = report_generator.get_available_teams()
    print(f"\n✅ Sistema listo. Equipos disponibles: {len(equipos)}")
    
    if len(equipos) > 0:
        print("📝 Para generar un reporte ejecuta: main()")
        print("📝 Para uso directo: generar_reporte_personalizado('Nombre_Equipo', [33,34,35])")
    
except Exception as e:
    print(f"❌ Error al inicializar: {e}")

if __name__ == "__main__":
    main()