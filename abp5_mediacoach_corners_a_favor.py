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

class CornersOfensivosReport:
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
    
    def guardar_sin_espacios(self, fig, filename):
        """Guarda sin espacios manteniendo landscape A4"""
        fig.set_size_inches(11.69, 8.27)
        
        fig.savefig(
            filename,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0,
            facecolor='white',
            edgecolor='none',
            format='pdf' if filename.endswith('.pdf') else 'png',
            transparent=False,
            orientation='landscape'
        )
        print(f"Archivo guardado SIN espacios formato A4: {filename}")

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
        Inicializa la clase para generar informes de córners ofensivos
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
        """Limpia y filtra los datos según los criterios especificados"""
        if self.df is None:
            return
        
        print("Datos originales:")
        print(f"- Métricas únicas: {self.df['NOMBRE MÉTRICA'].unique()}")
        print(f"- Períodos únicos: {self.df['PERIODO'].unique()}")
        
        # Filtrar por la métrica específica y período
        self.df = self.df[
            (self.df['NOMBRE MÉTRICA'] == 'B.P Saque de esquina a favor (Nº)') &
            (self.df['PERIODO'] == 'Total Partido')
        ].copy()
        
        print(f"Datos después del filtro: {self.df.shape[0]} filas")
        
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
    
    def filter_data(self, equipo, jornada_hasta):
        """Filtra los datos por equipo y desde la jornada 1 hasta jornada_hasta"""
        if self.df is None or len(self.df) == 0:
            return None
        
        # Obtener todas las jornadas desde 1 hasta jornada_hasta
        jornadas_incluir = [f'j{i}' for i in range(1, jornada_hasta + 1)]
        
        print(f"Jornadas a incluir: {jornadas_incluir}")
        
        filtered_df = self.df[
            (self.df['EQUIPO'] == equipo) & 
            (self.df['jornada'].isin(jornadas_incluir))
        ].copy()
        
        print(f"Datos filtrados: {len(filtered_df)} filas para {equipo}")
        return filtered_df
    
    def get_all_teams_data(self, jornada_hasta):
        """Obtiene datos de todos los equipos hasta la jornada especificada"""
        if self.df is None or len(self.df) == 0:
            return None
        
        jornadas_incluir = [f'j{i}' for i in range(1, jornada_hasta + 1)]
        
        # Obtener datos originales antes del filtrado
        df_original = pd.read_parquet(self.data_path)
        
        # Limpiar datos numéricos en df_original
        def clean_valor(valor):
            if pd.isna(valor):
                return 0.0
            valor_str = str(valor).replace(',', '.')
            try:
                return float(valor_str)
            except (ValueError, TypeError):
                return 0.0
        
        df_original['VALOR'] = df_original['VALOR'].apply(clean_valor)
        
        # Filtrar por las dos métricas que necesitamos
        df_corners = df_original[
            (df_original['NOMBRE MÉTRICA'] == 'B.P Saque de esquina a favor (Nº)') &
            (df_original['PERIODO'] == 'Total Partido') &
            (df_original['jornada'].isin(jornadas_incluir))
        ].copy()
        
        df_acciones = df_original[
            (df_original['NOMBRE MÉTRICA'] == 'B.P Acciones totales a favor (Nº)') &
            (df_original['PERIODO'] == 'Total Partido') &
            (df_original['jornada'].isin(jornadas_incluir))
        ].copy()
        
        # Limpiar nombres de equipos en ambos dataframes
        for df in [df_corners, df_acciones]:
            unique_teams = df['EQUIPO'].unique()
            team_mapping = {}
            processed_teams = set()
            
            for team in unique_teams:
                if team in processed_teams:
                    continue
                similar_teams = [team]
                for other_team in unique_teams:
                    if other_team != team and other_team not in processed_teams:
                        if self.similarity(team, other_team) > 0.7:
                            similar_teams.append(other_team)
                canonical_name = max(similar_teams, key=len)
                for similar_team in similar_teams:
                    team_mapping[similar_team] = canonical_name
                    processed_teams.add(similar_team)
            df['EQUIPO'] = df['EQUIPO'].map(team_mapping)
        
        # Calcular totales y número de partidos por equipo
        corners_total = df_corners.groupby('EQUIPO').agg({
            'VALOR': 'sum',
            'jornada': 'count'
        }).reset_index()
        corners_total.columns = ['EQUIPO', 'CORNERS_TOTAL', 'PARTIDOS']
        corners_total['CORNERS_POR_PARTIDO'] = corners_total['CORNERS_TOTAL'] / corners_total['PARTIDOS']
        
        acciones_total = df_acciones.groupby('EQUIPO').agg({
            'VALOR': 'sum',
            'jornada': 'count'
        }).reset_index()
        acciones_total.columns = ['EQUIPO', 'ACCIONES_TOTAL', 'PARTIDOS_ACC']
        acciones_total['ACCIONES_POR_PARTIDO'] = acciones_total['ACCIONES_TOTAL'] / acciones_total['PARTIDOS_ACC']
        
        # Combinar ambas métricas
        combined_data = pd.merge(
            corners_total[['EQUIPO', 'CORNERS_POR_PARTIDO']], 
            acciones_total[['EQUIPO', 'ACCIONES_POR_PARTIDO']], 
            on='EQUIPO', 
            how='inner'
        )
        combined_data.columns = ['EQUIPO', 'CORNERS', 'ACCIONES']
        
        return combined_data
    
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
        
        # Función para limpiar nombres
        def clean_team_name(name):
            """Limpia y normaliza nombres de equipos"""
            clean = name.lower().strip()
            # Remover acentos comunes
            clean = clean.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
            clean = clean.replace('ñ', 'n')
            return clean
        
        # Crear variantes del nombre del equipo
        def create_variants(team_name):
            """Crea todas las variantes posibles del nombre"""
            variants = []
            clean_name = clean_team_name(team_name)
            
            # Nombre original y limpio
            variants.extend([team_name, clean_name])
            
            # Variantes con/sin espacios
            variants.extend([
                team_name.replace(' ', '_'),
                team_name.replace(' ', ''),
                clean_name.replace(' ', '_'),
                clean_name.replace(' ', '')
            ])
            
            # Variantes de mayúsculas/minúsculas
            variants.extend([
                team_name.upper(),
                team_name.upper().replace(' ', '_'),
                team_name.upper().replace(' ', ''),
                team_name.lower(),
                team_name.lower().replace(' ', '_'),
                team_name.lower().replace(' ', '')
            ])
            
            # Remover prefijos/sufijos comunes
            prefixes_to_remove = ['real ', 'rcd ', 'rc ', 'deportivo ', 'club ', 'cf ', 'fc ', 'ud ', 'sd ', 'ad ']
            suffixes_to_remove = [' cf', ' fc', ' cd', ' ud', ' sd', ' ad']
            
            for prefix in prefixes_to_remove:
                if clean_name.startswith(prefix):
                    core_name = clean_name[len(prefix):]
                    variants.extend([
                        core_name,
                        core_name.replace(' ', '_'),
                        core_name.replace(' ', ''),
                        core_name.upper(),
                        core_name.upper().replace(' ', '_'),
                        core_name.upper().replace(' ', '')
                    ])
                    break
            
            for suffix in suffixes_to_remove:
                if clean_name.endswith(suffix):
                    core_name = clean_name[:-len(suffix)]
                    variants.extend([
                        core_name,
                        core_name.replace(' ', '_'),
                        core_name.replace(' ', ''),
                        core_name.upper(),
                        core_name.upper().replace(' ', '_'),
                        core_name.upper().replace(' ', '')
                    ])
                    break
            
            # Casos específicos conocidos
            specific_mappings = {
                'rc celta': ['celta', 'celta_vigo', 'celtavigo'],
                'deportivo alaves': ['alaves', 'deportivo_alaves', 'deportivoalaves'],
                'athletic club': ['athletic', 'athletic_bilbao', 'athleticbilbao'],
                'real betis': ['betis', 'real_betis', 'realbetis'],
                'real madrid': ['madrid', 'real_madrid', 'realmadrid'],
                'fc barcelona': ['barcelona', 'barca', 'fc_barcelona', 'fcbarcelona'],
                'atletico madrid': ['atletico', 'atletico_madrid', 'atleticomadrid'],
                'real sociedad': ['sociedad', 'real_sociedad', 'realsociedad'],
                'valencia cf': ['valencia', 'valencia_cf', 'valenciacf'],
                'sevilla fc': ['sevilla', 'sevilla_fc', 'sevillafc'],
                'villarreal cf': ['villarreal', 'villarreal_cf', 'villarrealcf'],
                'getafe cf': ['getafe', 'getafe_cf', 'getafecf'],
                'rayo vallecano': ['rayo', 'rayo_vallecano', 'rayovallecano'],
                'ca osasuna': ['osasuna', 'ca_osasuna', 'caosasuna'],
                'espanyol': ['espanyol', 'rcd_espanyol', 'rcdespaola'],
                'mallorca': ['mallorca', 'rcd_mallorca', 'rcdmallorca'],
                'cadiz cf': ['cadiz', 'cadiz_cf', 'cadizcf'],
                'elche cf': ['elche', 'elche_cf', 'elchecf'],
                'levante ud': ['levante', 'levante_ud', 'levanteud'],
                'granada cf': ['granada', 'granada_cf', 'granadacf']
            }
            
            clean_lower = clean_name.lower()
            for key, mappings in specific_mappings.items():
                if key in clean_lower or clean_lower in key:
                    variants.extend(mappings)
            
            # Remover duplicados manteniendo orden
            seen = set()
            unique_variants = []
            for variant in variants:
                if variant and variant not in seen:
                    seen.add(variant)
                    unique_variants.append(variant)
            
            return unique_variants
        
        # Obtener todas las variantes posibles
        possible_names = create_variants(equipo)
        
        print(f"Buscando escudo para '{equipo}'. Variantes: {possible_names[:10]}...")  # Mostrar solo las primeras 10
        
        # Búsqueda exacta primero
        for name in possible_names:
            logo_path = f"assets/escudos/{name}.png"
            if os.path.exists(logo_path):
                print(f"✅ Escudo encontrado por coincidencia exacta: {logo_path}")
                try:
                    return plt.imread(logo_path)
                except Exception as e:
                    print(f"Error al cargar escudo {logo_path}: {e}")
                    continue
        
        # Si no encuentra nada, buscar por similitud
        if os.path.exists('assets/escudos'):
            escudos_disponibles = [f.replace('.png', '') for f in os.listdir('assets/escudos') if f.endswith('.png')]
            
            print(f"Escudos disponibles para similitud: {escudos_disponibles}")
            
            best_match = None
            best_similarity = 0
            best_variant = None
            
            # Probar similitud con cada variante del nombre
            for variant in possible_names:
                for escudo_file in escudos_disponibles:
                    # Calcular similitud
                    similarity = self.similarity(variant, escudo_file)
                    
                    # También probar similitud con nombres limpios
                    clean_variant = clean_team_name(variant)
                    clean_escudo = clean_team_name(escudo_file)
                    clean_similarity = self.similarity(clean_variant, clean_escudo)
                    
                    max_similarity = max(similarity, clean_similarity)
                    
                    if max_similarity > best_similarity and max_similarity > 0.5:  # Reducir umbral a 50%
                        best_similarity = max_similarity
                        best_match = escudo_file
                        best_variant = variant
            
            if best_match:
                print(f"✅ Escudo encontrado por similitud: '{best_variant}' -> '{best_match}' (similitud: {best_similarity:.2f})")
                try:
                    return plt.imread(f"assets/escudos/{best_match}.png")
                except Exception as e:
                    print(f"Error al cargar escudo por similitud {best_match}: {e}")
        
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
        """Crea la visualización completa del informe de córners ofensivos"""
        # Filtrar datos del equipo específico
        filtered_df = self.filter_data(equipo, jornada_hasta)
        if filtered_df is None or len(filtered_df) == 0:
            print("No hay datos para los filtros especificados")
            return None
        
        # Obtener datos de todos los equipos para comparación
        all_teams_data = self.get_all_teams_data(jornada_hasta)
        
        # Crear figura
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
        
        # Configurar grid
        gs = fig.add_gridspec(3, 4, 
                     height_ratios=[0.08, 0.4, 1], 
                     width_ratios=[1, 1, 1, 0.6],
                     hspace=0.35, wspace=0.4,
                     left=0.11, right=0.95, top=0.97, bottom=0.05)
        
        # Área del título
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        # Título principal
        ax_title.text(0.5, 0.8, 'CÓRNERS OFENSIVOS', 
                     fontsize=26, weight='bold', ha='center', va='center',
                     color='#1e3d59', family='serif')
        ax_title.text(0.5, 0.15, f'DESDE JORNADA 1 HASTA JORNADA {jornada_hasta}', 
                     fontsize=11, ha='center', va='center',
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
                imagebox = OffsetImage(team_logo, zoom=0.3)
                ab = AnnotationBbox(imagebox, (0.95, 0.7), frameon=False, pad=0)
                ax_title.add_artist(ab)
            except Exception as e:
                print(f"Error al aplicar logo del equipo: {e}")
        # Escudo Villarreal arriba derecha (primero)
        villarreal_logo = self.load_any_team_logo('Villarreal CF')
        if villarreal_logo is not None:
            try:
                imagebox = OffsetImage(villarreal_logo, zoom=0.3)
                ab = AnnotationBbox(imagebox, (0.92, 0.7), frameon=False, pad=0)
                ax_title.add_artist(ab)
            except Exception as e:
                print(f"Error al aplicar logo Villarreal: {e}")
        
        # Gráfico 1: Evolución por jornadas (fila 2, spanning 2 columnas)
        ax_evolution = fig.add_subplot(gs[1, :3])
        ax_evolution.set_facecolor('white')
        ax_evolution.set_title(f'CÓRNERS POR PARTIDO - {equipo.upper()}', 
                               fontsize=12, weight='bold', color='#1e3d59', pad=10)
        self.plot_evolution_chart(ax_evolution, filtered_df, jornada_hasta)
        
        # Estadísticas generales (fila 2, columna 3)
        ax_stats = fig.add_subplot(gs[1, 3])
        ax_stats.set_facecolor('white')
        ax_stats.set_title('ESTADÍSTICAS GENERALES', fontsize=12, weight='bold', 
                          color='#1e3d59', pad=10)
        self.plot_general_stats(ax_stats, filtered_df, all_teams_data, equipo)
        
        # Gráfico 2: Ranking de equipos (fila 3, spanning 2 columnas)
        ax_ranking = fig.add_subplot(gs[2, 0])
        ax_ranking.set_facecolor('white')
        ax_ranking.set_title('MEDIA DE SAQUES DE ESQUINA A FAVOR', 
                            fontsize=12, weight='bold', color='#1e3d59', pad=10)
        self.plot_team_ranking(ax_ranking, all_teams_data, equipo)
        
        # Gráfico 3: Scatter plot comparativo (fila 3, columna 3)
        ax_scatter = fig.add_subplot(gs[2, 1:])
        ax_scatter.set_facecolor('white')
        ax_scatter.set_title('COMPARACIÓN ENTRE EQUIPOS', fontsize=12, weight='bold', 
                            color='#1e3d59', pad=10)
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
        
        # Crear el gráfico de barras verdes
        bars = ax.bar(evolution_df['jornada'], evolution_df['valor'], 
                    color='#27ae60', alpha=0.8, edgecolor='#1e8449', linewidth=1)
        
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
        """Dibuja un gráfico de radar comparando el equipo seleccionado con Villarreal"""
        ax.clear()
        
        # Cargar datos originales para obtener todas las métricas
        df_original = pd.read_parquet(self.data_path)
        
        # Función para limpiar valores numéricos
        def clean_valor(valor):
            if pd.isna(valor):
                return 0.0
            valor_str = str(valor).replace(',', '.')
            try:
                return float(valor_str)
            except (ValueError, TypeError):
                return 0.0
        
        df_original['VALOR'] = df_original['VALOR'].apply(clean_valor)
        
        # Función para obtener datos de un equipo específico
        def get_team_metrics(team_name, jornada_hasta):
            # Determinar jornadas a incluir
            if len(filtered_df) > 0:
                jornadas_incluir = [f'j{i}' for i in range(1, jornada_hasta + 1)]
            else:
                jornadas_incluir = [f'j{i}' for i in range(1, 6)]  # Por defecto hasta jornada 5
            
            # Filtrar datos para el equipo
            team_data = df_original[
                (df_original['EQUIPO'] == team_name) &
                (df_original['jornada'].isin(jornadas_incluir)) &
                (df_original['PERIODO'] == 'Total Partido')
            ]
            
            if len(team_data) == 0:
                return None
            
            # Contar número de partidos únicos (jornadas diferentes)
            num_partidos = len(team_data['jornada'].unique())
            
            if num_partidos == 0:
                return None
            
            # Métrica 1: Corners por partido
            corners_data = team_data[team_data['NOMBRE MÉTRICA'] == 'B.P Saque de esquina a favor (Nº)']
            corners_total = corners_data['VALOR'].sum()
            corners_por_partido = corners_total / num_partidos if num_partidos > 0 else 0
            
            # Métrica 2: Goles de corner por partido
            goles_corner_data = team_data[team_data['NOMBRE MÉTRICA'] == 'Goles a favor B. P. saque de esquina (Nº)']
            goles_corner_total = goles_corner_data['VALOR'].sum()
            goles_corner_por_partido = goles_corner_total / num_partidos if num_partidos > 0 else 0
            
            # Métrica 3: % Goles de balón parado del total
            pct_goles_bp_data = team_data[team_data['NOMBRE MÉTRICA'] == 'Goles a favor balón parado (% Total)']
            if len(pct_goles_bp_data) > 0:
                pct_goles_bp = pct_goles_bp_data['VALOR'].mean()
            else:
                pct_goles_bp = 0
            
            # Métrica 4: Acciones a balón parado por partido
            acciones_data = team_data[team_data['NOMBRE MÉTRICA'] == 'B.P Acciones totales a favor (Nº)']
            acciones_total = acciones_data['VALOR'].sum()
            acciones_por_partido = acciones_total / num_partidos if num_partidos > 0 else 0
            
            return {
                'corners_por_partido': corners_por_partido,
                'goles_corner_por_partido': goles_corner_por_partido,
                'pct_goles_bp': pct_goles_bp,
                'acciones_por_partido': acciones_por_partido,
                'num_partidos': num_partidos
            }
        
        # Obtener jornada hasta
        jornada_hasta = len(filtered_df) if len(filtered_df) > 0 else 5
        
        # Obtener métricas del equipo seleccionado
        metrics_equipo = get_team_metrics(equipo, jornada_hasta)
        
        # Buscar Villarreal en los datos
        villarreal_name = None
        equipos_unicos = df_original['EQUIPO'].unique()
        for team in equipos_unicos:
            if 'villarreal' in team.lower():
                villarreal_name = team
                break
        
        # Obtener métricas de Villarreal
        if villarreal_name:
            metrics_villarreal = get_team_metrics(villarreal_name, jornada_hasta)
        else:
            # Valores por defecto si no se encuentra Villarreal
            metrics_villarreal = {
                'corners_por_partido': 5.0,
                'acciones_por_partido': 10.0,
                'exito_promedio': 15.0,
                'xg_por_partido': 0.8,
                'num_partidos': 5
            }
        
        # Si no se pueden obtener métricas del equipo, usar valores por defecto
        if metrics_equipo is None:
            metrics_equipo = {
                'corners_por_partido': 0,
                'acciones_por_partido': 0,
                'exito_promedio': 0,
                'xg_por_partido': 0,
                'num_partidos': 0
            }
        
        # Calcular rankings de corners para todos los equipos
        teams_for_ranking = []
        for team in equipos_unicos:
            team_metrics = get_team_metrics(team, jornada_hasta)
            if team_metrics:
                teams_for_ranking.append((team, team_metrics['corners_por_partido']))
        
        # Ordenar por corners (descendente) para obtener rankings
        teams_for_ranking.sort(key=lambda x: x[1], reverse=True)
        
        # Encontrar el ranking específico del equipo seleccionado y Villarreal
        ranking_equipo = 999  # Valor por defecto
        ranking_villarreal = 999  # Valor por defecto
        
        for rank, (team_name, corners_value) in enumerate(teams_for_ranking):
            if team_name == equipo:
                ranking_equipo = rank + 1
            if team_name == villarreal_name:
                ranking_villarreal = rank + 1
        
        # Asignar los rankings a las métricas
        metrics_equipo['ranking'] = ranking_equipo
        metrics_villarreal['ranking'] = ranking_villarreal
        
        # Definir las métricas y sus valores
        metrics = ['Corners/\nPartido', 'Goles Corner/\nPartido', '% Goles\nBalón Parado', 'Acc. ABP/\nPartido', 'Ranking\nGeneral']
        
        # Obtener rangos de toda la liga para normalización
        all_teams_metrics = []
        for team in equipos_unicos:
            team_metrics = get_team_metrics(team, jornada_hasta)
            if team_metrics:
                all_teams_metrics.append(team_metrics)
        
        if len(all_teams_metrics) > 0:
            max_corners = max([m['corners_por_partido'] for m in all_teams_metrics])
            min_corners = min([m['corners_por_partido'] for m in all_teams_metrics])
            max_goles_corner = max([m['goles_corner_por_partido'] for m in all_teams_metrics])
            min_goles_corner = min([m['goles_corner_por_partido'] for m in all_teams_metrics])
            max_pct_bp = max([m['pct_goles_bp'] for m in all_teams_metrics])
            min_pct_bp = min([m['pct_goles_bp'] for m in all_teams_metrics])
            max_acciones = max([m['acciones_por_partido'] for m in all_teams_metrics])
            min_acciones = min([m['acciones_por_partido'] for m in all_teams_metrics])
            max_ranking = len(teams_for_ranking)  # Número total de equipos
        else:
            # Valores por defecto si no hay datos
            max_corners, min_corners = 10, 0
            max_goles_corner, min_goles_corner = 2, 0
            max_pct_bp, min_pct_bp = 50, 0
            max_acciones, min_acciones = 20, 0
            max_ranking = 20
        
        # Función para normalizar valores a escala 0-100
        def normalize_value(value, min_val, max_val):
            if max_val == min_val:
                return 50
            return max(0, min(100, (value - min_val) / (max_val - min_val) * 100))
        
        # Valores del equipo seleccionado (normalizados)
        values_equipo = [
            normalize_value(metrics_equipo['corners_por_partido'], min_corners, max_corners),
            normalize_value(metrics_equipo['goles_corner_por_partido'], min_goles_corner, max_goles_corner),
            normalize_value(metrics_equipo['pct_goles_bp'], min_pct_bp, max_pct_bp),
            normalize_value(metrics_equipo['acciones_por_partido'], min_acciones, max_acciones),
            # Para el ranking, invertir la normalización (ranking 1 = 100, ranking máximo = 0)
            normalize_value(max_ranking - ranking_equipo + 1, 1, max_ranking)
        ]

        # Valores de Villarreal (normalizados)
        values_villarreal = [
            normalize_value(metrics_villarreal['corners_por_partido'], min_corners, max_corners),
            normalize_value(metrics_villarreal['goles_corner_por_partido'], min_goles_corner, max_goles_corner),
            normalize_value(metrics_villarreal['pct_goles_bp'], min_pct_bp, max_pct_bp),
            normalize_value(metrics_villarreal['acciones_por_partido'], min_acciones, max_acciones),
            # Para el ranking, invertir la normalización (ranking 1 = 100, ranking máximo = 0)
            normalize_value(max_ranking - ranking_villarreal + 1, 1, max_ranking)
        ]

        # Valores reales para mostrar en los rectángulos
        valores_reales_equipo = [
            metrics_equipo['corners_por_partido'],
            metrics_equipo['goles_corner_por_partido'],
            metrics_equipo['pct_goles_bp'],
            metrics_equipo['acciones_por_partido'],
            ranking_equipo
        ]

        valores_reales_villarreal = [
            metrics_villarreal['corners_por_partido'],
            metrics_villarreal['goles_corner_por_partido'],
            metrics_villarreal['pct_goles_bp'],
            metrics_villarreal['acciones_por_partido'],
            ranking_villarreal
        ]
        
        # Número de métricas
        N = len(metrics)
        
        # Ángulos para cada métrica
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Cerrar el círculo
        
        # Cerrar los valores también
        values_equipo += values_equipo[:1]
        values_villarreal += values_villarreal[:1]
        
        # Configurar el gráfico polar
        fig = ax.get_figure()
        pos = ax.get_position()
        ax.remove()  # Eliminar el eje actual
        ax = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height], polar=True)
        
        # Dibujar los ejes y etiquetas
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Dibujar las líneas de la grilla (sin números)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels([])  # Quitar los números de las líneas del radar
        ax.grid(True, alpha=0.3)
        
        # Etiquetas de las métricas (en dos líneas y más pequeñas)
        metrics_two_lines = [
            'Corners/\nPartido', 
            'Goles Corner/\nPartido', 
            '% Goles\nBalón Parado', 
            'Acc. ABP/\nPartido', 
            'Ranking\nGeneral'
        ]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_two_lines, fontsize=6, weight='bold')

        # Dibujar las áreas
        # Villarreal (amarillo)
        ax.plot(angles, values_villarreal, 'o-', linewidth=2.5, label='Villarreal CF', color='#f1c40f', markersize=6)
        ax.fill(angles, values_villarreal, alpha=0.25, color='#f39c12')

        # Equipo seleccionado (azul)
        ax.plot(angles, values_equipo, 'o-', linewidth=2.5, label=equipo, color='#3498db', markersize=6)
        ax.fill(angles, values_equipo, alpha=0.25, color='#2980b9')

        # Título del gráfico abajo
        fig.text(pos.x0 + pos.width/2, pos.y0 - 0.05, 'COMPARACIÓN DE RENDIMIENTO', 
                 fontsize=8, weight='bold', color='#1e3d59', ha='center', va='center')

        # Leyenda
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 1.4), fontsize=7, 
                frameon=True, fancybox=True, shadow=True)

        # Añadir valores organizados: Villarreal (izq) - Métrica (centro) - Equipo (der)
        for i, (val_villarreal, val_equipo) in enumerate(zip(valores_reales_villarreal, valores_reales_equipo)):
            angle = angles[i]
            radius_base = 130  # Mismo radio para todos
            
            # Formatear valores según la métrica
            if i == 2:  # % Éxito
                texto_villarreal = f"{val_villarreal:.1f}%"
                texto_equipo = f"{val_equipo:.1f}%"
            elif i == 3:  # xG
                texto_villarreal = f"{val_villarreal:.2f}"
                texto_equipo = f"{val_equipo:.2f}"
            elif i == 4:  # Ranking
                texto_villarreal = f"#{val_villarreal:.0f}"
                texto_equipo = f"#{val_equipo:.0f}"
            else:  # Corners y Acciones
                texto_villarreal = f"{val_villarreal:.1f}"
                texto_equipo = f"{val_equipo:.1f}"
            
            # Calcular posiciones horizontales
            x_center = radius_base * np.cos(angle)
            y_center = radius_base * np.sin(angle)
            
            # Separación horizontal desde el centro
            offset_horizontal = 25
            
            # Villarreal a la izquierda
            if angle <= np.pi:  # Parte superior del círculo
                x_villarreal = x_center - offset_horizontal
                x_equipo = x_center + offset_horizontal
            else:  # Parte inferior del círculo
                x_villarreal = x_center - offset_horizontal
                x_equipo = x_center + offset_horizontal
            
            y_villarreal = y_center
            y_equipo = y_center
            
            # Convertir coordenadas cartesianas a polares
            angle_villarreal = np.arctan2(y_villarreal, x_villarreal)
            radius_villarreal = np.sqrt(x_villarreal**2 + y_villarreal**2)
            
            angle_equipo = np.arctan2(y_equipo, x_equipo)
            radius_equipo = np.sqrt(x_equipo**2 + y_equipo**2)
            
            # Añadir texto de Villarreal (izquierda)
            ax.text(angle_villarreal, radius_villarreal, texto_villarreal, 
                    ha='center', va='center', fontsize=6, weight='bold', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#f1c40f', alpha=0.8, edgecolor='none'))
            
            # Añadir texto del equipo seleccionado (derecha)
            ax.text(angle_equipo, radius_equipo, texto_equipo, 
                    ha='center', va='center', fontsize=6, weight='bold', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#3498db', alpha=0.8, edgecolor='none'))

        # Estilo adicional
        ax.spines['polar'].set_visible(False)
        ax.set_facecolor('#fafafa')
        
        return ax
    
    def plot_team_ranking(self, ax, all_teams_data, selected_equipo):
        """Dibuja el ranking de equipos"""
        if all_teams_data is None or len(all_teams_data) == 0:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center')
            return
        
        # Ordenar equipos por promedio de corners
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
        
        # Añadir valores solo para Villarreal y equipo seleccionado
        for i, (bar, valor, team) in enumerate(zip(bars, sorted_data['CORNERS'], sorted_data['EQUIPO'])):
            if team == selected_equipo or 'villarreal' in team.lower():
                ax.text(valor + 0.1, i, f'{valor:.1f}', va='center', fontsize=9, weight='bold')
        
        # Configurar ejes
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data['EQUIPO'], fontsize=8)
        ax.set_xlabel('Media saques de esquina a favor', fontsize=10, weight='bold')
        
        # Línea promedio de la liga
        promedio_liga = sorted_data['CORNERS'].mean()
        ax.axvline(promedio_liga, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(promedio_liga, len(sorted_data) * 0.95, f'Media de Liga: {promedio_liga:.1f}', 
            rotation=90, ha='right', va='top', fontsize=8, color='#27ae60')
        
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
        x_data = all_teams_data['CORNERS']    # Córners por partido
        y_data = all_teams_data['ACCIONES']   # Acciones por partido
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
        ax.text(max(x_data) * 0.98, y_promedio + 0.2, f'Media acciones a balón parado: {y_promedio:.1f}', 
            ha='right', va='bottom', fontsize=8, color='#34495e', weight='bold')
        
        # Crear scatter plot con escudos y nombres
        legend_elements = []
        
        for i, (equipo, x_val, y_val) in enumerate(zip(equipos, x_data, y_data)):
            # Definir si es equipo destacado
            is_selected = equipo == selected_equipo
            is_villarreal = 'villarreal' in equipo.lower()
            is_highlighted = is_selected or is_villarreal
            
            # Cargar escudo
            if is_villarreal:
                escudo = self.load_villarreal_logo()
            else:
                escudo = self.load_any_team_logo(equipo)
            
            # Configurar tamaño y procesamiento de imagen
            if is_highlighted:
                zoom_size = 0.24  # Cambiar de 0.12 a 0.24 (doble)
                alpha_val = 1.0
                # Mantener imagen original (a color)
                processed_escudo = escudo
            else:
                zoom_size = 0.12  # Cambiar de 0.04 a 0.08 (doble)
                alpha_val = 0.8
                # Convertir a escala de grises
                processed_escudo = self.convert_to_grayscale_no_background(escudo) if escudo is not None else None

            if processed_escudo is not None:
                try:
                    imagebox = OffsetImage(processed_escudo, zoom=zoom_size, alpha=alpha_val)
                    ab = AnnotationBbox(imagebox, (x_val, y_val), 
                                    frameon=False,
                                    pad=0,
                                    boxcoords="data")
                    ax.add_artist(ab)
                except Exception as e:
                    print(f"Error al cargar escudo para {equipo}: {e}")
                    processed_escudo = None
            
            if escudo is None:
                # Si no hay escudo, mostrar nombre del equipo
                if is_highlighted:
                    fontsize = 9
                    weight = 'bold'
                    color = '#e74c3c' if is_selected else '#f39c12'
                else:
                    fontsize = 7
                    weight = 'normal'
                    color = '#2c3e50'
                
                ax.text(x_val, y_val, equipo, ha='center', va='center', fontsize=fontsize, 
                    weight=weight, color=color, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.8))
        
        # Configurar ejes
        ax.set_xlabel('Media saques de esquina a favor', fontsize=10, weight='bold')
        ax.set_ylabel('Media acciones a balón parado', fontsize=10, weight='bold')
        
        # Ajustar límites con margen
        x_margin = (max(x_data) - min(x_data)) * 0.1
        y_margin = (max(y_data) - min(y_data)) * 0.1
        ax.set_xlim(min(x_data) - x_margin, max(x_data) + x_margin)
        ax.set_ylim(min(y_data) - y_margin, max(y_data) + y_margin)
        
        # Estilo
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Añadir etiquetas de cuadrantes
        ax.text(0.02, 0.98, 'Muchas Acciones\nPocos Córners', transform=ax.transAxes, 
            fontsize=8, ha='left', va='top', style='italic', color='#7f8c8d')
        ax.text(0.98, 0.98, 'Muchas Acciones\nMuchos Córners', transform=ax.transAxes, 
            fontsize=8, ha='right', va='top', style='italic', color='#7f8c8d')
        ax.text(0.02, 0.02, 'Pocas Acciones\nPocos Córners', transform=ax.transAxes, 
            fontsize=8, ha='left', va='bottom', style='italic', color='#7f8c8d')
        ax.text(0.98, 0.02, 'Pocas Acciones\nMuchos Córners', transform=ax.transAxes, 
            fontsize=8, ha='right', va='bottom', style='italic', color='#7f8c8d')

# Función para seleccionar equipo interactivamente
def seleccionar_equipo_interactivo():
    """Permite al usuario seleccionar un equipo de forma interactiva"""
    try:
        report_generator = CornersOfensivosReport()
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
        print("=== GENERADOR DE REPORTES DE CÓRNERS OFENSIVOS ===")
        
        # Selección interactiva
        equipo, jornada_hasta = seleccionar_equipo_interactivo()
        
        if equipo is None or jornada_hasta is None:
            print("No se pudo completar la selección.")
            return
        
        print(f"\nGenerando reporte para {equipo} - Desde jornada 1 hasta jornada {jornada_hasta}")
        
        # Crear el reporte
        report_generator = CornersOfensivosReport()
        fig = report_generator.create_visualization(equipo, jornada_hasta)
        
        if fig:
            # Mostrar en pantalla
            plt.show()
            
            # Guardar como PDF
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_corners_ofensivos_{equipo_filename}.pdf"
            report_generator.guardar_sin_espacios(fig, output_path)
            
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
        report_generator = CornersOfensivosReport()
        fig = report_generator.create_visualization(equipo, jornada_hasta)
        
        if fig:
            if mostrar:
                plt.show()
            
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"reporte_corners_ofensivos_{equipo_filename}.pdf"
                report_generator.guardar_sin_espacios(fig, output_path)
            
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
print("=== INICIALIZANDO GENERADOR DE REPORTES CÓRNERS OFENSIVOS ===")
try:
    verificar_assets()
    report_generator = CornersOfensivosReport()
    equipos = report_generator.get_available_teams()
    print(f"\n✅ Sistema listo. Equipos disponibles: {len(equipos)}")
    
    if len(equipos) > 0:
        print("📝 Para generar un reporte ejecuta: main()")
        print("📝 Para uso directo: generar_reporte_personalizado('Nombre_Equipo', 15)")
    
except Exception as e:
    print(f"❌ Error al inicializar: {e}")

if __name__ == "__main__":
    main()