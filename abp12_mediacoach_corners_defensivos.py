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

class CornersDefensivosReport:
    def convert_to_grayscale(self, image):
        """Convierte una imagen a escala de grises"""
        try:
            from PIL import Image
            # Convertir numpy array a PIL Image
            pil_img = Image.fromarray((image * 255).astype('uint8'))
            # Convertir a escala de grises
            gray_img = pil_img.convert('L')
            # Convertir de vuelta a RGB para mantener compatibilidad
            gray_rgb = gray_img.convert('RGB')
            # Convertir de vuelta a numpy array normalizado
            return np.array(gray_rgb) / 255.0
        except Exception as e:
            print(f"Error convirtiendo a escala de grises: {e}")
            return image

    def convert_to_grayscale_no_background(self, image):
        """Convierte una imagen a escala de grises y quita el fondo de manera más efectiva"""
        try:
            from PIL import Image
            # Convertir numpy array a PIL Image
            pil_img = Image.fromarray((image * 255).astype('uint8'))
            
            # Convertir a RGBA para manejar transparencia
            if pil_img.mode != 'RGBA':
                pil_img = pil_img.convert('RGBA')
            
            # Obtener dimensiones
            width, height = pil_img.size
            
            # Detectar el color de fondo analizando las esquinas
            corner_pixels = [
                pil_img.getpixel((0, 0)),
                pil_img.getpixel((width-1, 0)),
                pil_img.getpixel((0, height-1)),
                pil_img.getpixel((width-1, height-1))
            ]
            
            # Encontrar el color más común en las esquinas (probable fondo)
            from collections import Counter
            corner_colors = [pixel[:3] for pixel in corner_pixels]  # Solo RGB, sin alpha
            most_common = Counter(corner_colors).most_common(1)[0][0]
            bg_r, bg_g, bg_b = most_common
            
            print(f"Color de fondo detectado: RGB({bg_r}, {bg_g}, {bg_b})")
            
            # Convertir a escala de grises
            gray_img = pil_img.convert('L')
            gray_rgba = gray_img.convert('RGBA')
            
            # Obtener los datos de píxeles
            data = gray_rgba.getdata()
            new_data = []
            
            # Calcular tolerancia basada en el fondo detectado
            tolerance = 30  # Ajusta este valor si es necesario
            
            for i, item in enumerate(data):
                # Obtener el píxel original para comparar colores
                x = i % width
                y = i // width
                original_pixel = pil_img.getpixel((x, y))
                orig_r, orig_g, orig_b = original_pixel[:3]
                
                # Verificar si es similar al color de fondo
                if (abs(orig_r - bg_r) <= tolerance and 
                    abs(orig_g - bg_g) <= tolerance and 
                    abs(orig_b - bg_b) <= tolerance):
                    # Hacer transparente
                    new_data.append((item[0], item[1], item[2], 0))
                else:
                    # Mantener con transparencia reducida
                    new_data.append((item[0], item[1], item[2], 200))
            
            gray_rgba.putdata(new_data)
            
            # Convertir de vuelta a numpy array normalizado
            return np.array(gray_rgba) / 255.0
            
        except Exception as e:
            print(f"Error convirtiendo a escala de grises sin fondo: {e}")
            # Fallback: simplemente convertir a escala de grises normal
            return self.convert_to_grayscale(image)
        
    def load_villarreal_logo(self):
        """Carga el escudo del Villarreal con búsqueda mejorada"""
        possible_names = [
            'Villarreal CF',
            'villarreal_cf', 
            'Villarreal_CF',
            'villarreal', 
            'Villarreal', 
            'VILLARREAL', 
            'VILLARREAL_CF',
            'VillarrealCF',
            'Villarreal_cf',
            'villarrealcf'
        ]
        
        # Primero búsqueda exacta
        for name in possible_names:
            logo_path = f"assets/escudos/{name}.png"
            if os.path.exists(logo_path):
                print(f"Escudo Villarreal encontrado: {logo_path}")
                try:
                    return plt.imread(logo_path)
                except Exception as e:
                    print(f"Error al cargar escudo Villarreal {logo_path}: {e}")
                    continue
        
        # Si no encuentra nada, buscar por similitud
        if os.path.exists('assets/escudos'):
            escudos_disponibles = [f.replace('.png', '') for f in os.listdir('assets/escudos') if f.endswith('.png')]
            
            best_match = None
            best_similarity = 0
            
            # Buscar por "villarreal"
            for escudo_file in escudos_disponibles:
                if 'villarreal' in escudo_file.lower():
                    similarity = self.similarity('villarreal', escudo_file.lower())
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = escudo_file
            
            if best_match:
                print(f"Escudo Villarreal encontrado por similitud: {best_match}")
                try:
                    return plt.imread(f"assets/escudos/{best_match}.png")
                except:
                    pass
        
        print("No se encontró el escudo del Villarreal")
        return None

    def __init__(self, data_path="extraccion_mediacoach/data/estadisticas_equipo.parquet"):
        """
        Inicializa la clase para generar informes de córners defensivos
        """
        self.data_path = data_path
        self.df = None
        self.load_data()
        self.clean_and_filter_data()
        
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
    
    
    def clean_and_filter_data(self):
        """Limpia y filtra los datos según los criterios especificados para múltiples métricas defensivas"""
        if self.df is None:
            return
        
        print("Datos originales:")
        print(f"- Métricas únicas: {self.df['NOMBRE MÉTRICA'].unique()}")
        print(f"- Períodos únicos: {self.df['PERIODO'].unique()}")
        
        # Definir múltiples métricas defensivas
        defensive_metrics = [
            'B.P Saque de esquina en contra (Nº)',
            'Goles en contra B. P. saque de esquina (Nº)',
            'B.P Acciones totales en contra (Nº)',
            'Goles en contra balón parado (% Total)'
        ]    
        
        # Filtrar por las métricas defensivas especificadas y período
        self.df = self.df[
            (self.df['NOMBRE MÉTRICA'].isin(defensive_metrics)) &
            (self.df['PERIODO'] == 'Total Partido')
        ].copy()
        
        print(f"Datos después del filtro: {self.df.shape[0]} filas")
        print(f"Métricas incluidas: {self.df['NOMBRE MÉTRICA'].unique()}")
        
        if len(self.df) == 0:
            print("⚠️ No se encontraron datos con los filtros aplicados")
            return
        
        # Limpiar nombres de equipos
        unique_teams = self.df['EQUIPO'].unique()
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
        self.df['EQUIPO'] = self.df['EQUIPO'].map(team_mapping)
        
        # AÑADIR ESTA LÍNEA:
        self.clean_numeric_data()
        
        print(f"Limpieza completada. Equipos únicos: {len(self.df['EQUIPO'].unique())}")
        print(f"Jornadas disponibles: {sorted(self.df['jornada'].unique())}")
        
        # Almacenar las métricas utilizadas para uso posterior
        self.defensive_metrics = defensive_metrics

    def clean_numeric_data(self):
        """Limpia y convierte los datos numéricos de la columna VALOR"""
        if self.df is None or len(self.df) == 0:
            return
        
        # Función para limpiar valores numéricos
        def clean_valor(valor):
            if pd.isna(valor):
                return 0.0
            
            # Convertir a string si no lo es
            valor_str = str(valor)
            
            # Reemplazar comas por puntos para decimales
            valor_str = valor_str.replace(',', '.')
            
            try:
                return float(valor_str)
            except (ValueError, TypeError):
                print(f"⚠️ No se pudo convertir el valor: {valor}")
                return 0.0
        
        # Aplicar limpieza a la columna VALOR
        print("Limpiando datos numéricos...")
        self.df['VALOR'] = self.df['VALOR'].apply(clean_valor)
        
        print(f"✅ Datos numéricos limpiados. Tipo de columna VALOR: {self.df['VALOR'].dtype}")
        print(f"Valores VALOR únicos (muestra): {sorted(self.df['VALOR'].unique())[:10]}")
        
    def get_available_teams(self):
        """Retorna la lista de equipos disponibles"""
        if self.df is None or len(self.df) == 0:
            return []
        return sorted(self.df['EQUIPO'].unique())
    
    def get_available_jornadas(self, equipo=None):
        """Retorna las jornadas disponibles, opcionalmente filtradas por equipo"""
        if self.df is None or len(self.df) == 0:
            return []
        
        if equipo:
            filtered_df = self.df[self.df['EQUIPO'] == equipo]
            return sorted(filtered_df['jornada'].unique())
        else:
            return sorted(self.df['jornada'].unique())
    
    def filter_data(self, equipo, jornada_hasta, metrica_especifica='B.P Saque de esquina en contra (Nº)'):
        """Filtra los datos por equipo, jornada y métrica específica"""
        if self.df is None or len(self.df) == 0:
            return None
        
        jornadas_incluir = [f'j{i}' for i in range(1, jornada_hasta + 1)]
        
        filtered_df = self.df[
            (self.df['EQUIPO'] == equipo) & 
            (self.df['jornada'].isin(jornadas_incluir)) &
            (self.df['NOMBRE MÉTRICA'] == metrica_especifica)
        ].copy()
        
        print(f"Datos filtrados: {len(filtered_df)} filas para {equipo} - {metrica_especifica}")
        return filtered_df
    
    def get_all_teams_data(self, jornada_hasta):
        """Obtiene datos de todos los equipos con todas las métricas defensivas"""
        jornadas_incluir = [f'j{i}' for i in range(1, jornada_hasta + 1)]
        
        df_original = pd.read_parquet(self.data_path)
        df_original['VALOR'] = df_original['VALOR'].apply(self.clean_valor_helper)
        
        result_data = []
        equipos_unicos = df_original['EQUIPO'].unique()
        
        for equipo in equipos_unicos:
            team_metrics = {'EQUIPO': equipo}
            
            team_data = df_original[
                (df_original['EQUIPO'] == equipo) &
                (df_original['jornada'].isin(jornadas_incluir)) &
                (df_original['PERIODO'] == 'Total Partido')
            ]
            
            num_partidos = len(team_data['jornada'].unique()) if len(team_data) > 0 else 1
            
            # Obtener cada métrica defensiva
            for metrica in self.defensive_metrics:
                metrica_data = team_data[team_data['NOMBRE MÉTRICA'] == metrica]
                if len(metrica_data) > 0:
                    if '(%)' in metrica:  # Para porcentajes, usar promedio
                        team_metrics[metrica] = metrica_data['VALOR'].mean()
                    else:  # Para números absolutos, usar suma/partidos
                        team_metrics[metrica] = metrica_data['VALOR'].sum() / num_partidos
                else:
                    team_metrics[metrica] = 0
            
            result_data.append(team_metrics)
        
        df_result = pd.DataFrame(result_data)
        
        # Crear columnas simplificadas para compatibilidad
        if 'B.P Saque de esquina en contra (Nº)' in df_result.columns:      # ← CORREGIDO
            df_result['CORNERS'] = df_result['B.P Saque de esquina en contra (Nº)']  # ← CORREGIDO
        if 'B.P Acciones totales en contra (Nº)' in df_result.columns:      # ← CORREGIDO
            df_result['ACCIONES'] = df_result['B.P Acciones totales en contra (Nº)']  # ← CORREGIDO

        return df_result

    def clean_valor_helper(self, valor):
        """Helper function para limpiar valores"""
        if pd.isna(valor):
            return 0.0
        valor_str = str(valor).replace(',', '.')
        try:
            return float(valor_str)
        except (ValueError, TypeError):
            return 0.0
    
    def load_team_logo(self, equipo):
        """Carga el escudo del equipo con búsqueda mejorada"""
        # Primero intenta búsqueda exacta
        possible_names = [
            equipo,
            equipo.replace(' ', '_'),
            equipo.replace(' ', ''),
            equipo.replace(' CF', ''),
            equipo.replace(' FC', ''),
            equipo.replace('Real ', ''),
            equipo.replace('CF ', ''),
            equipo.replace('FC ', ''),
            equipo.lower(),
            equipo.lower().replace(' ', '_'),
            equipo.lower().replace(' ', ''),
            equipo.lower().replace(' cf', ''),
            equipo.lower().replace(' fc', ''),
            equipo.upper(),
            equipo.upper().replace(' ', '_')
        ]
        
        for name in possible_names:
            logo_path = f"assets/escudos/{name}.png"
            if os.path.exists(logo_path):
                print(f"Escudo encontrado: {logo_path}")
                try:
                    return plt.imread(logo_path)
                except Exception as e:
                    print(f"Error al cargar escudo {logo_path}: {e}")
                    continue
        
        # Si no encuentra nada, buscar por similitud (igual que load_any_team_logo)
        if os.path.exists('assets/escudos'):
            escudos_disponibles = [f.replace('.png', '') for f in os.listdir('assets/escudos') if f.endswith('.png')]
            
            best_match = None
            best_similarity = 0
            
            for escudo_file in escudos_disponibles:
                similarity = self.similarity(equipo, escudo_file)
                if similarity > best_similarity and similarity > 0.6:  # 60% mínimo
                    best_similarity = similarity
                    best_match = escudo_file
            
            if best_match:
                print(f"Escudo encontrado por similitud: {best_match} (similitud: {best_similarity:.2f})")
                try:
                    return plt.imread(f"assets/escudos/{best_match}.png")
                except:
                    pass
        
        print(f"No se encontró el escudo para: {equipo}")
        return None
    
    def load_any_team_logo(self, equipo):
        """Intenta cargar el escudo de cualquier equipo con búsqueda mejorada"""
        
        def clean_team_name(name):
            clean = name.lower().strip().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
            return clean
        
        def create_variants(team_name):
            variants = []
            clean_name = clean_team_name(team_name)
            variants.extend([team_name, clean_name, team_name.replace(' ', '_'), team_name.replace(' ', ''), clean_name.replace(' ', '_'), clean_name.replace(' ', '')])
            variants.extend([team_name.upper(), team_name.upper().replace(' ', '_'), team_name.lower(), team_name.lower().replace(' ', '_')])
            
            prefixes_to_remove = ['real ', 'rcd ', 'rc ', 'deportivo ', 'club ', 'cf ', 'fc ', 'ud ', 'sd ', 'ad ']
            suffixes_to_remove = [' cf', ' fc', ' cd', ' ud', ' sd', ' ad']
            
            for prefix in prefixes_to_remove:
                if clean_name.startswith(prefix):
                    core_name = clean_name[len(prefix):]
                    variants.extend([core_name, core_name.replace(' ', '_'), core_name.upper()])
                    break
            for suffix in suffixes_to_remove:
                if clean_name.endswith(suffix):
                    core_name = clean_name[:-len(suffix)]
                    variants.extend([core_name, core_name.replace(' ', '_'), core_name.upper()])
                    break
            
            specific_mappings = {'rc celta': ['celta', 'celta_vigo'], 'deportivo alaves': ['alaves'], 'athletic club': ['athletic', 'athletic_bilbao'], 'real betis': ['betis'], 'real madrid': ['madrid'], 'fc barcelona': ['barcelona', 'barca'], 'atletico madrid': ['atletico'], 'real sociedad': ['sociedad'], 'valencia cf': ['valencia'], 'sevilla fc': ['sevilla'], 'villarreal cf': ['villarreal'], 'getafe cf': ['getafe'], 'rayo vallecano': ['rayo'], 'ca osasuna': ['osasuna'], 'espanyol': ['espanyol', 'rcd_espanyol'], 'mallorca': ['mallorca', 'rcd_mallorca']}
            
            clean_lower = clean_name.lower()
            for key, mappings in specific_mappings.items():
                if key in clean_lower or clean_lower in key:
                    variants.extend(mappings)
            
            seen = set()
            return [v for v in variants if v and not (v in seen or seen.add(v))]

        possible_names = create_variants(equipo)
        
        for name in possible_names:
            logo_path = f"assets/escudos/{name}.png"
            if os.path.exists(logo_path):
                try: return plt.imread(logo_path)
                except: continue
        
        if os.path.exists('assets/escudos'):
            escudos_disponibles = [f.replace('.png', '') for f in os.listdir('assets/escudos') if f.endswith('.png')]
            best_match, best_similarity = None, 0
            for variant in possible_names:
                for escudo_file in escudos_disponibles:
                    similarity = self.similarity(variant, escudo_file)
                    if similarity > best_similarity and similarity > 0.5:
                        best_similarity, best_match = similarity, escudo_file
            if best_match:
                try: return plt.imread(f"assets/escudos/{best_match}.png")
                except: pass
                
        print(f"❌ No se encontró escudo para: {equipo}")
        return None
    
    def load_ball_image(self):
        """Carga la imagen del balón"""
        ball_path = "assets/balon.png"
        if os.path.exists(ball_path):
            try:
                return plt.imread(ball_path)
            except Exception as e:
                print(f"Error al cargar balón: {e}")
                return None
        else:
            print(f"No se encontró el balón: {ball_path}")
            return None
    
    def find_real_team_name(self, extracted_name):
        """Encuentra el nombre real del equipo comparando con la columna EQUIPO"""
        # Obtener todos los equipos únicos de la columna EQUIPO
        df_original = pd.read_parquet(self.data_path)
        unique_teams = df_original['EQUIPO'].unique()
        
        best_match = extracted_name
        best_similarity = 0
        
        for team in unique_teams:
            # Calcular similitud entre el nombre extraído y cada equipo real
            team_clean = team.lower().replace(' ', '').replace('fc', '').replace('cf', '').replace('real', '').replace('rcd', '')
            extracted_clean = extracted_name.lower().replace('fc', '').replace('cf', '').replace('real', '').replace('rcd', '')
            
            similarity = self.similarity(extracted_clean, team_clean)
            
            if similarity > best_similarity and similarity > 0.6:  # Mínimo 60% de similitud
                best_similarity = similarity
                best_match = team
        
        print(f"'{extracted_name}' -> '{best_match}' (similitud: {best_similarity:.2f})")
        return best_match

    def load_background(self):
        """Carga el fondo del informe"""
        bg_path = "assets/fondo_informes.png"
        if os.path.exists(bg_path):
            try:
                return plt.imread(bg_path)
            except Exception as e:
                print(f"Error al cargar fondo: {e}")
                return None
        else:
            print(f"No se encontró el fondo: {bg_path}")
            return None
    
    def create_visualization(self, equipo, jornada_hasta, figsize=(11.69, 8.27)):
        """Crea la visualización completa del informe de córners defensivos"""
        # Filtrar datos del equipo específico SOLO para corners (para el gráfico principal)
        filtered_df = self.filter_data(equipo, jornada_hasta)
        if filtered_df is None or len(filtered_df) == 0:
            print("No hay datos para los filtros especificados")
            return None
        
        # Obtener datos de todos los equipos para comparación (todas las métricas)
        all_teams_data = self.get_all_teams_data(jornada_hasta)
        
        # Crear figura - A4 HORIZONTAL
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        # Cargar y establecer fondo
        background = self.load_background()
        if background is not None:
            try:
                ax_background = fig.add_axes([0, 0, 1, 1], zorder=-1)
                ax_background.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25, zorder=-1)
                ax_background.axis('off')
                print("Fondo aplicado correctamente")
            except Exception as e:
                print(f"Error al aplicar fondo: {e}")
        
        # Configurar grid - IGUAL QUE ABP6
        gs = fig.add_gridspec(3, 4, 
                height_ratios=[0.08, 0.4, 1], 
                width_ratios=[1, 1, 1, 0.6],
                hspace=0.35, wspace=0.4,  # ← CAMBIO CLAVE
                left=0.11, right=0.95, top=0.97, bottom=0.05)  # ← CAMBIO left
        
        # Área del título
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        # Título principal
        ax_title.text(0.5, 0.8, 'CÓRNERS DEFENSIVOS', 
                    fontsize=26, weight='bold', ha='center', va='center',
                    color='#c0392b', family='serif')
        ax_title.text(0.5, 0.18, f'DESDE JORNADA 1 HASTA JORNADA {jornada_hasta}', 
                    fontsize=12, ha='center', va='center',
                    color='#2c3e50', weight='bold')
        
        # Balón arriba izquierda
        ball = self.load_ball_image()
        if ball is not None:
            try:
                imagebox = OffsetImage(ball, zoom=0.12)
                ab = AnnotationBbox(imagebox, (0.05, 0.5), frameon=False)
                ax_title.add_artist(ab)
            except Exception as e:
                print(f"Error al aplicar balón: {e}")
        
        # Escudo del equipo seleccionado arriba derecha (segundo)
        team_logo = self.load_team_logo(equipo)
        if team_logo is not None:
            try:
                imagebox = OffsetImage(team_logo, zoom=0.3)  # ← CAMBIO zoom
                ab = AnnotationBbox(imagebox, (0.95, 0.7), frameon=False, pad=0)  # ← CAMBIO posición
                ax_title.add_artist(ab)
            except Exception as e:
                print(f"Error al aplicar logo del equipo: {e}")

        # Escudo Villarreal arriba derecha (primero)
        villarreal_logo = self.load_villarreal_logo()  # ← CAMBIO función
        if villarreal_logo is not None:
            try:
                imagebox = OffsetImage(villarreal_logo, zoom=0.3)  # ← CAMBIO zoom
                ab = AnnotationBbox(imagebox, (0.92, 0.7), frameon=False, pad=0)  # ← CAMBIO posición
                ax_title.add_artist(ab)
            except Exception as e:
                print(f"Error al aplicar logo Villarreal: {e}")
        
        # Gráfico 1: Evolución por jornadas (fila 2, spanning 3 columnas)
        ax_evolution = fig.add_subplot(gs[1, :3])
        ax_evolution.set_facecolor('white')
        ax_evolution.set_title(f'CÓRNERS EN CONTRA POR PARTIDO - {equipo.upper()}', 
                            fontsize=12, weight='bold', color='#c0392b', pad=10)
        self.plot_evolution_chart(ax_evolution, filtered_df, jornada_hasta)
        
        # Estadísticas generales (fila 2, columna 3)
        ax_stats = fig.add_subplot(gs[1, 3])
        ax_stats.set_facecolor('white')
        ax_stats.set_title('ESTADÍSTICAS GENERALES', fontsize=12, weight='bold', 
                        color='#c0392b', pad=10)
        self.plot_general_stats(ax_stats, filtered_df, all_teams_data, equipo)
        
        # Gráfico 2: Ranking de equipos (fila 3, columna 0)
        ax_ranking = fig.add_subplot(gs[2, 0])
        ax_ranking.set_facecolor('white')
        ax_ranking.set_title('MEDIA DE SAQUES DE ESQUINA EN CONTRA', 
                            fontsize=12, weight='bold', color='#c0392b', pad=10)
        self.plot_team_ranking(ax_ranking, all_teams_data, equipo)
        
        # Gráfico 3: Scatter plot comparativo (fila 3, columnas 1-3)
        ax_scatter = fig.add_subplot(gs[2, 1:])
        ax_scatter.set_facecolor('white')
        ax_scatter.set_title('COMPARACIÓN ENTRE EQUIPOS', fontsize=12, weight='bold', 
                            color='#c0392b', pad=10)
        self.plot_team_scatter(ax_scatter, all_teams_data, equipo)
        
        return fig
    
    def plot_evolution_chart(self, ax, filtered_df, jornada_hasta):
        """Dibuja el gráfico de evolución por jornadas"""
        # Cargar datos originales para obtener información del rival
        df_original = pd.read_parquet(self.data_path)

        # Asegurarse de que tenemos datos para todas las jornadas
        jornadas_completas = list(range(1, jornada_hasta + 1))
        
        # Crear dataframe con todas las jornadas
        data_evolution = []
        for jornada in jornadas_completas:
            jornada_data = filtered_df[filtered_df['jornada'] == f'j{jornada}']
            if len(jornada_data) > 0:
                valor = jornada_data['VALOR'].iloc[0]
            else:
                valor = 0  # Si no hay datos para esa jornada
            data_evolution.append({'jornada': jornada, 'valor': valor})
        
        evolution_df = pd.DataFrame(data_evolution)
        
        # Crear el gráfico de barras rojas (para defensivos)
        bars = ax.bar(evolution_df['jornada'], evolution_df['valor'], 
                    color='#e74c3c', alpha=0.8, edgecolor='#c0392b', linewidth=1)
        
        # Añadir valores arriba de las barras
        for bar, valor in zip(bars, evolution_df['valor']):
            if valor > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{valor:.0f}', ha='center', va='bottom', fontsize=9, weight='bold')
        
        for i, jornada in enumerate(jornadas_completas):
            jornada_str = f'j{jornada}'
            equipo_principal = filtered_df['EQUIPO'].iloc[0] if len(filtered_df) > 0 else ''
            
            # Buscar el partido en los datos originales
            partidos_jornada = df_original[
                (df_original['jornada'] == jornada_str) & 
                (df_original['EQUIPO'] == equipo_principal)
            ]
            
            rival = f"J{jornada}"  # Valor por defecto
            
            if len(partidos_jornada) > 0 and 'partido' in df_original.columns:
                partido_str = partidos_jornada['partido'].iloc[0]
                print(f"Jornada {jornada}: Partido = {partido_str}")
                
                # Parsear el formato "sevillafc0-2gironafc"
                import re
                # Buscar patrón: texto + número + - + número + texto
                match = re.match(r'(.+?)(\d+)-(\d+)(.+)', partido_str)
                
                if match:
                    equipo1 = match.group(1).strip()
                    equipo2 = match.group(4).strip()
                    
                    print(f"Equipos encontrados: '{equipo1}' vs '{equipo2}'")
                    
                    # Determinar cuál es el rival
                    # Comparar con el equipo principal (sin espacios y en minúsculas)
                    equipo_principal_clean = equipo_principal.lower().replace(' ', '').replace('fc', '').replace('cf', '')
                    equipo1_clean = equipo1.lower().replace('fc', '').replace('cf', '')
                    equipo2_clean = equipo2.lower().replace('fc', '').replace('cf', '')
                    
                    if equipo_principal_clean in equipo1_clean:
                        rival = self.find_real_team_name(equipo2)
                    elif equipo_principal_clean in equipo2_clean:
                        rival = self.find_real_team_name(equipo1)
                    else:
                        rival = self.find_real_team_name(equipo1)
                        
                    print(f"Rival identificado: {rival}")
            
            # Intentar cargar escudo del rival
            escudo = self.load_any_team_logo(rival)
            
            # Posición Y: un poco por debajo del eje X
            max_val = max(evolution_df['valor']) if max(evolution_df['valor']) > 0 else 10
            y_pos = max_val * -0.1
            
            if escudo is not None:
                # Colocar escudo
                imagebox = OffsetImage(escudo, zoom=0.08)
                ab = AnnotationBbox(imagebox, (jornada, y_pos), frameon=False)
                ax.add_artist(ab)
                print(f"Escudo colocado para {rival}")
            else:
                # Crear abreviatura de 3 letras
                abrev = rival[:3].upper()
                ax.text(jornada, y_pos, abrev, ha='center', va='center', 
                        fontsize=7, weight='bold', color='#2c3e50')
                    
        # Configurar ejes
        ax.set_xlabel('Jornada', fontsize=10, weight='bold')
        ax.set_ylabel('Nº Córners', fontsize=10, weight='bold')
        ax.set_xticks(jornadas_completas)
        ax.set_xticklabels([f'J{j}' for j in jornadas_completas])
        ax.set_xlim(0.5, jornada_hasta + 0.5)
        max_val = max(evolution_df['valor']) if max(evolution_df['valor']) > 0 else 10
        ax.set_ylim(max_val * -0.15, max_val * 1.2)
        
        # Estilo
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def plot_general_stats(self, ax, filtered_df, all_teams_data, equipo):
        """Dibuja un gráfico de radar con las métricas defensivas correctas y ranking."""
        ax.clear()
        
        df_original = pd.read_parquet(self.data_path)
        
        def clean_valor(valor):
            if pd.isna(valor): return 0.0
            return float(str(valor).replace(',', '.'))
        
        df_original['VALOR'] = df_original['VALOR'].apply(clean_valor)

        def get_team_defensive_metrics(team_name, jornada_hasta, all_teams_df):
            jornadas_incluir = [f'j{i}' for i in range(1, jornada_hasta + 1)]
            team_data = all_teams_df[
                (all_teams_df['EQUIPO'] == team_name) &
                (all_teams_df['jornada'].isin(jornadas_incluir)) &
                (all_teams_df['PERIODO'] == 'Total Partido')
            ]
            
            if len(team_data) == 0: return None
            
            num_partidos = len(team_data['jornada'].unique())
            if num_partidos == 0: return {'equipo': team_name, 'corners_contra': 0, 'goles_corner_contra': 0, 'acciones_contra': 0, 'pct_goles_bp_contra': 0}

            # Extraer las 4 métricas defensivas
            corners_contra = team_data[team_data['NOMBRE MÉTRICA'] == 'B.P Saque de esquina en contra (Nº)']['VALOR'].sum() / num_partidos
            goles_corner_contra = team_data[team_data['NOMBRE MÉTRICA'] == 'Goles en contra B. P. saque de esquina (Nº)']['VALOR'].sum() / num_partidos
            acciones_contra = team_data[team_data['NOMBRE MÉTRICA'] == 'B.P Acciones totales en contra (Nº)']['VALOR'].sum() / num_partidos
            pct_goles_bp_contra = team_data[team_data['NOMBRE MÉTRICA'] == 'Goles en contra balón parado (% Total)']['VALOR'].mean()

            return {
                'equipo': team_name,
                'corners_contra': corners_contra,
                'goles_corner_contra': goles_corner_contra,
                'acciones_contra': acciones_contra,
                'pct_goles_bp_contra': pct_goles_bp_contra
            }

        jornada_hasta = max([int(j.replace('j', '')) for j in df_original['jornada'].unique() if j.startswith('j')], default=1)
        
        all_teams_metrics_list = [m for m in [get_team_defensive_metrics(t, jornada_hasta, df_original) for t in df_original['EQUIPO'].unique()] if m]
        
        # Ranking: Menos goles de córner en contra es mejor
        all_teams_metrics_list.sort(key=lambda x: x['goles_corner_contra'])
        
        metrics_equipo = next((m for m in all_teams_metrics_list if m['equipo'] == equipo), None)
        villarreal_name = next((t for t in df_original['EQUIPO'].unique() if 'villarreal' in t.lower()), None)
        metrics_villarreal = next((m for m in all_teams_metrics_list if m['equipo'] == villarreal_name), None)

        if not metrics_equipo: metrics_equipo = {'equipo': equipo, 'corners_contra': 0, 'goles_corner_contra': 0, 'acciones_contra': 0, 'pct_goles_bp_contra': 0}
        if not metrics_villarreal: metrics_villarreal = {'equipo': villarreal_name, 'corners_contra': 0, 'goles_corner_contra': 0, 'acciones_contra': 0, 'pct_goles_bp_contra': 0}
        
        ranking_equipo = next((i + 1 for i, m in enumerate(all_teams_metrics_list) if m['equipo'] == equipo), len(all_teams_metrics_list))
        ranking_villarreal = next((i + 1 for i, m in enumerate(all_teams_metrics_list) if m['equipo'] == villarreal_name), len(all_teams_metrics_list))

        metrics = ['Córners\nContra', 'Goles Córner\nContra', 'Acciones\nContra', '% Goles BP\nContra', 'Ranking\nSolidez']
        
        max_vals = {
            'corners': max(m['corners_contra'] for m in all_teams_metrics_list),
            'goles': max(m['goles_corner_contra'] for m in all_teams_metrics_list),
            'acciones': max(m['acciones_contra'] for m in all_teams_metrics_list),
            'pct': max(m['pct_goles_bp_contra'] for m in all_teams_metrics_list),
            'ranking': len(all_teams_metrics_list)
        }
        
        # Normalización INVERTIDA: un valor bajo (bueno) da una puntuación alta (cerca de 100)
        def normalize_defensive(value, max_val):
            if max_val == 0: return 100
            return max(0, min(100, (1 - (value / max_val)) * 100))

        values_equipo = [
            normalize_defensive(metrics_equipo['corners_contra'], max_vals['corners']),
            normalize_defensive(metrics_equipo['goles_corner_contra'], max_vals['goles']),
            normalize_defensive(metrics_equipo['acciones_contra'], max_vals['acciones']),
            normalize_defensive(metrics_equipo['pct_goles_bp_contra'], max_vals['pct']),
            normalize_defensive(ranking_equipo - 1, max_vals['ranking'])
        ]
        
        values_villarreal = [
            normalize_defensive(metrics_villarreal['corners_contra'], max_vals['corners']),
            normalize_defensive(metrics_villarreal['goles_corner_contra'], max_vals['goles']),
            normalize_defensive(metrics_villarreal['acciones_contra'], max_vals['acciones']),
            normalize_defensive(metrics_villarreal['pct_goles_bp_contra'], max_vals['pct']),
            normalize_defensive(ranking_villarreal - 1, max_vals['ranking'])
        ]
        
        N = len(metrics)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        values_equipo += values_equipo[:1]
        values_villarreal += values_villarreal[:1]
        
        fig = ax.get_figure()
        pos = ax.get_position()
        ax.remove()
        ax = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height], polar=True)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.3)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=6, weight='bold')
        
        ax.plot(angles, values_villarreal, 'o-', linewidth=2.5, label='Villarreal CF', color='#f1c40f', markersize=6)
        ax.fill(angles, values_villarreal, alpha=0.25, color='#f39c12')
        
        ax.plot(angles, values_equipo, 'o-', linewidth=2.5, label=equipo, color='#e74c3c', markersize=6)
        ax.fill(angles, values_equipo, alpha=0.25, color='#c0392b')
        
        fig.text(pos.x0 + pos.width/2, pos.y0 - 0.05, 'COMPARACIÓN DEFENSIVA', 
                fontsize=8, weight='bold', color='#c0392b', ha='center', va='center')
        
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 1.4), fontsize=7, frameon=True, fancybox=True, shadow=True)
        
        # Guardar los valores reales para mostrarlos
        valores_reales_equipo = [
            metrics_equipo['corners_contra'],
            metrics_equipo['goles_corner_contra'],
            metrics_equipo['acciones_contra'],
            metrics_equipo['pct_goles_bp_contra'],
            ranking_equipo
        ]

        valores_reales_villarreal = [
            metrics_villarreal['corners_contra'],
            metrics_villarreal['goles_corner_contra'],
            metrics_villarreal['acciones_contra'],
            metrics_villarreal['pct_goles_bp_contra'],
            ranking_villarreal
        ]

        # Añadir los valores en recuadros fuera del radar
        for i, (val_villarreal, val_equipo) in enumerate(zip(valores_reales_villarreal, valores_reales_equipo)):
            angle = angles[i]
            radius_base = 130 

            # Formatear el texto de los valores
            if i == 3:  # Porcentaje de Goles BP
                texto_villarreal = f"{val_villarreal:.1f}%"
                texto_equipo = f"{val_equipo:.1f}%"
            elif i == 4:  # Ranking
                texto_villarreal = f"#{val_villarreal:.0f}"
                texto_equipo = f"#{val_equipo:.0f}"
            else:  # El resto de métricas
                texto_villarreal = f"{val_villarreal:.1f}"
                texto_equipo = f"{val_equipo:.1f}"
            
            # Calcular posiciones
            x_center = radius_base * np.cos(angle)
            y_center = radius_base * np.sin(angle)
            offset_horizontal = 25 
            
            x_villarreal = x_center - offset_horizontal
            x_equipo = x_center + offset_horizontal
            
            # Convertir de vuelta a coordenadas polares para el texto
            angle_villarreal = np.arctan2(y_center, x_villarreal)
            radius_villarreal = np.sqrt(x_villarreal**2 + y_center**2)
            
            angle_equipo = np.arctan2(y_center, x_equipo)
            radius_equipo = np.sqrt(x_equipo**2 + y_center**2)
            
            # Dibujar texto de Villarreal (color amarillo/dorado)
            ax.text(angle_villarreal, radius_villarreal, texto_villarreal, 
                    ha='center', va='center', fontsize=6, weight='bold', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#f1c40f', alpha=0.8, edgecolor='none'))
            
            # Dibujar texto del equipo seleccionado (color rojo defensivo)
            ax.text(angle_equipo, radius_equipo, texto_equipo, 
                    ha='center', va='center', fontsize=6, weight='bold', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#e74c3c', alpha=0.8, edgecolor='none'))

        ax.spines['polar'].set_visible(False)
        ax.set_facecolor('#fafafa')
        
        return ax
    
    def plot_team_ranking(self, ax, all_teams_data, selected_equipo):
        """Dibuja el ranking de equipos (para defensivos, menos es mejor)"""
        if all_teams_data is None or len(all_teams_data) == 0:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center')
            return
        
        # Ordenar equipos por promedio de corners (ascendente para defensivos - menos es mejor)
        sorted_data = all_teams_data.sort_values('CORNERS', ascending=True)
        
        # Colores: destacar Villarreal y equipo seleccionado
        colors = []
        for team in sorted_data['EQUIPO']:
            if team == selected_equipo:
                colors.append('#c0392b')
            elif 'villarreal' in team.lower():
                colors.append('#d68910')
            else:
                colors.append('#bdc3c7')
        
        # Crear gráfico horizontal
        bars = ax.barh(range(len(sorted_data)), sorted_data['CORNERS'], 
                    color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=0.5)
        
        ax.invert_yaxis()
        
        # Añadir valores solo para Villarreal y equipo seleccionado
        for i, (bar, valor, team) in enumerate(zip(bars, sorted_data['CORNERS'], sorted_data['EQUIPO'])):
            if team == selected_equipo or 'villarreal' in team.lower():
                ax.text(valor + 0.1, i, f'{valor:.1f}', va='center', fontsize=9, weight='bold')
        
        # Configurar ejes
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data['EQUIPO'], fontsize=8)
        ax.set_xlabel('Media saques de esquina en contra', fontsize=10, weight='bold')
        
        # Línea promedio de la liga
        promedio_liga = sorted_data['CORNERS'].mean()
        ax.axvline(promedio_liga, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(promedio_liga, len(sorted_data) * 0.95, f'Media de Liga: {promedio_liga:.1f}', 
            rotation=90, ha='right', va='top', fontsize=8, color='#e74c3c')
        
        # Estilo
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def plot_team_scatter(self, ax, all_teams_data, selected_equipo):
        """Dibuja el scatter plot con dos métricas, cruz divisoria y escudos"""
        if all_teams_data is None or len(all_teams_data) == 0:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center')
            return
        
        # Extraer datos
        x_data = all_teams_data['CORNERS']    # Córners en contra por partido
        y_data = all_teams_data['ACCIONES']   # Acciones en contra por partido
        equipos = all_teams_data['EQUIPO']
        
        # Calcular promedios para las líneas de la cruz
        x_promedio = x_data.mean()
        y_promedio = y_data.mean()
        
        # Dibujar líneas de la cruz
        ax.axvline(x_promedio, color='#34495e', linestyle='-', linewidth=2, alpha=0.6)
        ax.axhline(y_promedio, color='#34495e', linestyle='-', linewidth=2, alpha=0.6)
        
        # Añadir etiquetas de la cruz
        ax.text(x_promedio + 0.1, max(y_data) * 0.95, f'Media de Liga: {x_promedio:.1f}', 
            rotation=90, ha='left', va='top', fontsize=8, color='#34495e', weight='bold')
        ax.text(max(x_data) * 0.98, y_promedio + 0.2, f'Media acciones en contra: {y_promedio:.1f}', 
            ha='right', va='bottom', fontsize=8, color='#34495e', weight='bold')
        
        # Crear scatter plot con escudos y nombres
        legend_elements = []
        
        for i, (equipo, x_val, y_val) in enumerate(zip(equipos, x_data, y_data)):
            is_selected = equipo == selected_equipo
            is_villarreal = 'villarreal' in equipo.lower()
            is_highlighted = is_selected or is_villarreal
            
            if is_villarreal:
                escudo = self.load_villarreal_logo()
            else:
                escudo = self.load_any_team_logo(equipo) # Ahora usará la nueva y mejorada función
            
            if is_highlighted:
                zoom_size = 0.24
                alpha_val = 1.0
                processed_escudo = escudo # A color
            else:
                zoom_size = 0.12
                alpha_val = 0.8
                processed_escudo = self.convert_to_grayscale_no_background(escudo) if escudo is not None else None

            if processed_escudo is not None:
                try:
                    imagebox = OffsetImage(processed_escudo, zoom=zoom_size, alpha=alpha_val)
                    ab = AnnotationBbox(imagebox, (x_val, y_val), 
                                    frameon=False, pad=0, boxcoords="data")
                    ax.add_artist(ab)
                except Exception as e:
                    print(f"Error al cargar escudo para {equipo}: {e}")
            elif escudo is None:
                color = '#c0392b' if is_selected else '#d68910' if is_villarreal else '#2c3e50'
                ax.text(x_val, y_val, equipo[:3].upper(), ha='center', va='center', fontsize=7, 
                        weight='bold', color='white',
                        bbox=dict(boxstyle='circle,pad=0.3', facecolor=color, alpha=0.9))
        
        # Configurar ejes
        ax.set_xlabel('Media saques de esquina en contra', fontsize=10, weight='bold')
        ax.set_ylabel('Media acciones en contra', fontsize=10, weight='bold')
        
        # Ajustar límites con margen
        x_margin = (max(x_data) - min(x_data)) * 0.1
        y_margin = (max(y_data) - min(y_data)) * 0.1
        ax.set_xlim(min(x_data) - x_margin, max(x_data) + x_margin)
        ax.set_ylim(min(y_data) - y_margin, max(y_data) + y_margin)
        
        # Estilo
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Añadir etiquetas de cuadrantes (para defensivos)
        ax.text(0.02, 0.98, 'Muchas Acciones\nPocos Córners\n(VULNERABLE)', transform=ax.transAxes, 
            fontsize=8, ha='left', va='top', style='italic', color='#e74c3c')
        ax.text(0.98, 0.98, 'Muchas Acciones\nMuchos Córners\n(MUY VULNERABLE)', transform=ax.transAxes, 
            fontsize=8, ha='right', va='top', style='italic', color='#8e44ad')
        ax.text(0.02, 0.02, 'Pocas Acciones\nPocos Córners\n(SÓLIDO)', transform=ax.transAxes, 
            fontsize=8, ha='left', va='bottom', style='italic', color='#27ae60')
        ax.text(0.98, 0.02, 'Pocas Acciones\nMuchos Córners\n(POCO SÓLIDO)', transform=ax.transAxes, 
            fontsize=8, ha='right', va='bottom', style='italic', color='#f39c12')

# Función para seleccionar equipo interactivamente
def seleccionar_equipo_interactivo():
    """Permite al usuario seleccionar un equipo de forma interactiva"""
    try:
        report_generator = CornersDefensivosReport()
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
        
        # Extraer números de las jornadas disponibles
        jornadas_numericas = []
        for j in jornadas_disponibles:
            try:
                numero = int(j.replace('j', ''))
                jornadas_numericas.append(numero)
            except:
                pass

        max_jornada = max(jornadas_numericas) if jornadas_numericas else 5

        # Preguntar hasta qué jornada incluir
        while True:
            try:
                jornada_hasta = input(f"¿Hasta qué jornada incluir? (máximo {max_jornada}): ").strip()
                jornada_hasta = int(jornada_hasta)
                
                if 1 <= jornada_hasta <= max_jornada:
                    break
                else:
                    print(f"Por favor, ingresa un número entre 1 y {max_jornada}")
            except ValueError:
                print("Por favor, ingresa un número válido")
        
        return equipo_seleccionado, jornada_hasta
        
    except Exception as e:
        print(f"Error en la selección: {e}")
        return None, None

# Función principal
def main():
    try:
        print("=== GENERADOR DE REPORTES DE CÓRNERS DEFENSIVOS ===")
        
        # Selección interactiva
        equipo, jornada_hasta = seleccionar_equipo_interactivo()
        
        if equipo is None or jornada_hasta is None:
            print("No se pudo completar la selección.")
            return
        
        print(f"\nGenerando reporte para {equipo} - Desde jornada 1 hasta jornada {jornada_hasta}")
        
        # Crear el reporte
        report_generator = CornersDefensivosReport()
        fig = report_generator.create_visualization(equipo, jornada_hasta)
        
        if fig:
            # Mostrar en pantalla
            plt.show()
            
            # Guardar como PDF
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_corners_defensivos_{equipo_filename}.pdf"
            
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

# Función para uso directo con parámetros
def generar_reporte_personalizado(equipo, jornada_hasta, mostrar=True, guardar=True):
    """
    Función para generar un reporte personalizado
    
    Args:
        equipo (str): Nombre del equipo
        jornada_hasta (int): Hasta qué jornada incluir (desde la 1)
        mostrar (bool): Si mostrar el gráfico en pantalla
        guardar (bool): Si guardar como PDF
    """
    try:
        report_generator = CornersDefensivosReport()
        fig = report_generator.create_visualization(equipo, jornada_hasta)
        
        if fig:
            if mostrar:
                plt.show()
            
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"reporte_corners_defensivos_{equipo_filename}.pdf"
                
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

# Verificar archivos necesarios
def verificar_assets():
    """Verifica que existan los directorios y archivos necesarios"""
    print("\n=== VERIFICACIÓN DE ASSETS ===")
    
    # Verificar directorios
    dirs_to_check = ['assets', 'assets/escudos']
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"✅ Directorio encontrado: {dir_path}")
        else:
            print(f"❌ Directorio faltante: {dir_path}")
            
    files_to_check = [
        'extraccion_mediacoach/data/estadisticas_equipo.parquet',
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

# Inicialización
print("=== INICIALIZANDO GENERADOR DE REPORTES CÓRNERS DEFENSIVOS ===")
try:
    verificar_assets()
    report_generator = CornersDefensivosReport()
    equipos = report_generator.get_available_teams()
    print(f"\n✅ Sistema listo. Equipos disponibles: {len(equipos)}")
    
    if len(equipos) > 0:
        print("📝 Para generar un reporte ejecuta: main()")
        print("📝 Para uso directo: generar_reporte_personalizado('Nombre_Equipo', 15)")
    
except Exception as e:
    print(f"❌ Error al inicializar: {e}")

if __name__ == "__main__":
    main()