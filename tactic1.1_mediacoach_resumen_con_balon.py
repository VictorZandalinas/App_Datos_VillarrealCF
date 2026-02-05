import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime
import os
import warnings
from mplsoccer import VerticalPitch, PyPizza
import unicodedata
from PIL import Image
import textwrap

warnings.filterwarnings('ignore')

class KPIConBalonMediaCoach:
    def __init__(self, parquet_path='extraccion_mediacoach/data/estadisticas_equipo.parquet'):
        self.parquet_path = parquet_path
        self.df_stats = None
        # Métricas ajustadas para coincidir exactamente con la imagen PNG
        self.metricas_agrupadas = {
            'VERTICALIDAD': [
                # Primero sin porcentaje (Nº)
                'Pases hacia adelante (Nº)',
                'Pases profundos totales (Nº)',
                'Regates totales (Nº)',
                'Centros totales (Nº)',
                'Pases largos cambio de orientación (Nº)',
                'Pases largos totales (Nº)',
                # Luego con porcentaje (%)
                'Pases hacia adelante (% Éxito)',
                'Pases profundos totales con exito (%)',
                'Regates totales (% Éxito)',
                'Centros totales (% Éxito)',
                'Modo de Canalización (% Pase Largo)',
                'Pases largos cambio de orientación (% Éxito)'
            ],
            'CREACION_PELIGRO': [
                # Primero sin porcentaje (Nº)
                'Remates totales minuto 90 (Nº)',
                'Remates juego dinámico (Nº)',
                'Pases Propiciadores Remate (Nº)',
                'Remates a portería (Nº)',
                'Remates goles (Nº)',
                'Aprovechamiento Ofensivo (Goles a favor)',
                'Peligrosidad Ofensiva (puntos xG)',
                'Eficiencia Ofensiva (Nº Goles a Favor/Remate)',
                # Luego con porcentaje (%)
                'Eficacia Finalización (%)',
                'Pases Propiciadores Remate (% Total)',
                'Remates a portería (% Total)',
                'Remates goles (% Total)'
            ],

            'POSESION': [
                # Primero sin porcentaje (Nº)
                'Ritmo de Circulación (Pases/Min)',
                'Pases totales (Precisos)',
                'Pases totales (Nº)',
                # Luego con porcentaje (%)
                'Posesión del Balón (%)',
                'Posesion de balon en campo propio (%)',
                'Posesion de balon en campo rival (%)',
                'Pases totales (% Éxito)'
            ],
            'DUELOS_JUEGO_AEREO': [
                # Primero sin porcentaje (Nº)
                'Duelos Aéreos Ofensivos (Nº)',
                # Luego con porcentaje (%)
                'Duelos Aéreos Ofensivos (% Éxito)',
                'Duelos Aéreos Ofensivos (% Posesión Ganada)',
                'Modo de Canalización (% Pase Largo)'
            ],
            'POSICIONAMIENTO': [
                'Profundidad Posicional Global (m.)',
                'Anchura Posicional Global (m.)',
                'Centroide Colectivo Ofensivo (m.)',
                'Distancia Línea Defensiva Global (m. Propia Puerta)'
            ]
        }

        # Diccionario para renombrar métricas en visualización
        self.nombres_cortos = {
            # VERTICALIDAD
            'Pases hacia adelante (Nº)': 'Pases adelante',
            'Pases hacia adelante (% Éxito)': 'Pases adelante %',
            'Pases profundos totales (Nº)': 'Pases profundos',
            'Pases profundos totales con exito (%)': 'Pases profundos %',
            'Regates totales (Nº)': 'Regates',
            'Regates totales (% Éxito)': 'Regates %',
            'Centros totales (Nº)': 'Centros',
            'Centros totales (% Éxito)': 'Centros %',
            'Modo de Canalización (% Pase Largo)': 'Pases largos %',
            'Pases largos cambio de orientación (% Éxito)': 'Cambios orient. %',
            'Pases largos cambio de orientación (Nº)': 'Cambios orient.',
            'Pases largos totales (Nº)': 'Pases largos',
            
            # CREACION_PELIGRO
            'Remates totales minuto 90 (Nº)': 'Remates 90´',
            'Peligrosidad Ofensiva (puntos xG)': 'Puntos xG',
            'Aprovechamiento Ofensivo (Goles a favor)': 'Goles favor',
            'Eficacia Finalización (%)': 'Eficacia final. %',
            'Eficiencia Ofensiva (Nº Goles a Favor/Remate)': 'Goles/Remate',
            'Remates juego dinámico (Nº)': 'Remates juego',
            'Pases Propiciadores Remate (Nº)': 'Pases clave',
            'Pases Propiciadores Remate (% Total)': 'Pases clave %',
            'Remates a portería (Nº)': 'Remates portería',
            'Remates a portería (% Total)': 'Remates portería %',
            'Remates goles (Nº)': 'Goles',
            'Remates goles (% Total)': 'Goles %',
            
            # POSESION
            'Posesión del Balón (%)': 'Posesión %',
            'Posesion de balon en campo propio (%)': 'Posesión propio %',
            'Posesion de balon en campo rival (%)': 'Posesión rival %',
            'Ritmo de Circulación (Pases/Min)': 'Pases/min',
            'Pases totales (Precisos)': 'Pases precisos',
            'Pases totales (% Éxito)': 'Pases totales %',
            'Pases totales (Nº)': 'Pases totales',
            
            # DUELOS_JUEGO_AEREO
            'Duelos Aéreos Ofensivos (Nº)': 'Duelos aéreos',
            'Duelos Aéreos Ofensivos (% Éxito)': 'Duelos aéreos %',
            'Duelos Aéreos Ofensivos (% Posesión Ganada)': 'Duelos % posesión',
            
            # POSICIONAMIENTO
            'Profundidad Posicional Global (m.)': 'Profundidad (m)',
            'Anchura Posicional Global (m.)': 'Anchura (m)',
            'Centroide Colectivo Ofensivo (m.)': 'Centroide of. (m)',
            'Distancia Línea Defensiva Global (m. Propia Puerta)': 'Línea def. (m)'
        }


    def _formatear_label(self, metrica):
        """Formatea las etiquetas de las métricas para mejor legibilidad."""
        # Usar nombre corto si existe
        metrica_display = self.analyzer.nombres_cortos.get(metrica, metrica)
        return textwrap.fill(metrica_display, width=12, break_long_words=False)

    
    def calcular_estadisticas_liga(self):
        """Calcula media y máximo de la liga para cada métrica."""
        estadisticas = {}
        datos_partido = self.df_stats[self.df_stats['PERIODO'] == 'Total Partido']
        
        for grupo, lista_metricas in self.metricas_agrupadas.items():
            for metrica in lista_metricas:
                datos_metrica = datos_partido[datos_partido['NOMBRE MÉTRICA'] == metrica]
                if not datos_metrica.empty:
                    # Agrupar por equipo y jornada, luego calcular media por equipo
                    valores_por_equipo = datos_metrica.groupby('EQUIPO')['VALOR'].mean()
                    
                    estadisticas[metrica] = {
                        'media_liga': valores_por_equipo.mean(),
                        'maximo': valores_por_equipo.max(),
                        'equipo_maximo': valores_por_equipo.idxmax(),
                        'minimo': valores_por_equipo.min()
                    }
        return estadisticas

    def normalizar_valores(self, valores, estadisticas_metrica):
        """Normaliza valores basándose en el máximo de la liga."""
        if not estadisticas_metrica:
            return valores
        
        maximo = estadisticas_metrica.get('maximo', 100)
        if maximo == 0:
            return valores
        
        # Normalizar a escala 0-100 basándose en el máximo
        return [(v / maximo) * 100 for v in valores]
    
    def load_team_logo(self, equipo, target_size=(80, 80)):
        """Carga y redimensiona el logo del equipo buscando por palabras más largas primero (estilo Tactic3)."""
        def normalize_word(word):
            word = unicodedata.normalize('NFD', word)
            word = ''.join(char for char in word if unicodedata.category(char) != 'Mn')
            return word.lower().strip()

        if not os.path.exists('assets/escudos'):
            print("⚠️  La carpeta assets/escudos no existe.")
            return None

        palabras_ignorar = {'cf', 'fc', 'cd', 'ud', 'rcd', 'rc', 'ca', 'de', 'del', 'la', 'las', 'el', 'los'}
        palabras = equipo.split()
        palabras_normalizadas = [normalize_word(p) for p in palabras if normalize_word(p) not in palabras_ignorar and len(normalize_word(p)) > 2]
        palabras_ordenadas = sorted(palabras_normalizadas, key=len, reverse=True)
        
        all_files = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
        
        for palabra_buscar in palabras_ordenadas:
            for filename in all_files:
                nombre_archivo_norm = normalize_word(os.path.splitext(filename)[0])
                if palabra_buscar in nombre_archivo_norm:
                    logo_path = f"assets/escudos/{filename}"
                    try:
                        with Image.open(logo_path) as img:
                            img = img.convert('RGBA')
                            img.thumbnail(target_size, Image.Resampling.LANCZOS)
                            return np.array(img)
                    except Exception as e:
                        pass
                        continue
        
        print(f"⚠️  No se encontró logo para {equipo} con búsqueda directa. Intentando por similitud...")
        return None # Opcional: añadir lógica de similitud si se desea

    def cargar_datos(self):
        try:
            self.df_stats = pd.read_parquet(self.parquet_path)
            self.df_stats['VALOR'] = pd.to_numeric(
                self.df_stats['VALOR'].astype(str).str.replace(',', '.'), errors='coerce'
            )
            self.df_stats.dropna(subset=['VALOR'], inplace=True)
            
            # LIMPIAR ESPACIOS EN NOMBRES DE MÉTRICAS
            self.df_stats['NOMBRE MÉTRICA'] = self.df_stats['NOMBRE MÉTRICA'].str.strip()
            
            return self.df_stats
        except Exception as e:
            pass
            return None

    def obtener_equipos_disponibles(self):
        """Obtiene lista de equipos únicos"""
        if self.df_stats is not None:
            return sorted(self.df_stats['EQUIPO'].unique())
        return []

    def filtrar_datos_equipo_jornada(self, equipo, jornada=None):
        """Filtra datos por equipo y jornada, promediando métricas duplicadas."""
        datos = self.df_stats[self.df_stats['EQUIPO'] == equipo].copy()
        if jornada is not None:
            datos = datos[datos['jornada'] == jornada]
        
        return datos.groupby(['jornada', 'NOMBRE MÉTRICA', 'EQUIPO', 'PERIODO']).agg(VALOR=('VALOR', 'mean')).reset_index()

    # REEMPLAZA ESTA FUNCIÓN

    def calcular_promedios_metricas(self, datos):
        """Calcula promedios dividiendo por partidos jugados."""
        metricas = {}
        datos_partido = datos[datos['PERIODO'] == 'Total Partido']
        
        # Contar partidos jugados (jornadas únicas)
        num_partidos = datos_partido['jornada'].nunique()
        if num_partidos == 0:
            num_partidos = 1  # Evitar división por 0
        
        # Métricas posicionales que deben usar MEAN en vez de SUM
        metricas_posicionales = [
            'Profundidad Posicional Global (m.)',
            'Anchura Posicional Global (m.)',
            'Centroide Colectivo Ofensivo (m.)',
            'Distancia Línea Defensiva Global (m. Propia Puerta)'
        ]
        
        # Calcular SUMA para métricas acumulativas
        agregados_suma = datos_partido.groupby('NOMBRE MÉTRICA')['VALOR'].sum()
        # Calcular MEDIA para métricas posicionales
        agregados_media = datos_partido.groupby('NOMBRE MÉTRICA')['VALOR'].mean()
        
        for grupo, lista_metricas in self.metricas_agrupadas.items():
            for metrica in lista_metricas:
                if metrica in metricas_posicionales:
                    # Usar media directamente para métricas posicionales
                    metricas[metrica] = agregados_media.get(metrica, 0)
                else:
                    # Usar suma dividida por partidos para métricas acumulativas
                    valor_total = agregados_suma.get(metrica, 0)
                    metricas[metrica] = valor_total / num_partidos
        
        return metricas

    def calcular_puntuacion_grupo(self, metricas_equipo, grupo):
        """Calcula puntuación 1-10 del equipo en un grupo de métricas basada en estadísticas de liga."""
        estadisticas_liga = self.calcular_estadisticas_liga()
        metricas_grupo = self.metricas_agrupadas.get(grupo, [])
        
        puntuaciones = []
        for metrica in metricas_grupo:
            valor_equipo = metricas_equipo.get(metrica, 0)
            est = estadisticas_liga.get(metrica, {})
            
            maximo = est.get('maximo', 1)
            minimo = est.get('minimo', 0)
            media = est.get('media_liga', (maximo + minimo) / 2)
            
            # Normalizar a escala 1-10
            if maximo != minimo and maximo > 0:
                puntuacion = 1 + ((valor_equipo - minimo) / (maximo - minimo)) * 9
                # Limitar entre 1 y 10
                puntuacion = max(1, min(10, puntuacion))
            else:
                puntuacion = 5
            
            # Para POSICIONAMIENTO: amplificar diferencias respecto a la media
                if grupo == 'POSICIONAMIENTO':
                    # Calcular desviación de la media (normalizada)
                    if media > 0:
                        desviacion = (valor_equipo - media) / media
                        # Amplificar la desviación (factor 2.5, antes 1.5) y añadir a puntuación base
                        puntuacion = 5.5 + (desviacion * 2.5 * 4.5)
                        # Limitar entre 1 y 10
                        puntuacion = max(1, min(10, puntuacion))
            
            puntuaciones.append(puntuacion)
        
        # Promedio de todas las métricas del grupo
        return np.mean(puntuaciones) if puntuaciones else 5

    def obtener_metricas_comparativas(self, equipo1, equipo2, jornada=None):
        """Obtiene métricas comparativas entre dos equipos."""
        if jornada is None:
            datos1 = self.filtrar_datos_equipo_jornada(equipo1)
            datos2 = self.filtrar_datos_equipo_jornada(equipo2)
            metricas1 = self.calcular_promedios_metricas(datos1)
            metricas2 = self.calcular_promedios_metricas(datos2)
        else:
            datos1 = self.filtrar_datos_equipo_jornada(equipo1, jornada)
            datos2 = self.filtrar_datos_equipo_jornada(equipo2, jornada)
            metricas1 = self.extraer_metricas_jornada(datos1)
            metricas2 = self.extraer_metricas_jornada(datos2)
        return metricas1, metricas2


class VisualizadorKPIConBalon:
    def __init__(self, kpi_analyzer):
        self.analyzer = kpi_analyzer
        self.colores_equipo = {
            'equipo1': '#00A2E8',
            'equipo2': '#FEE500'
        }
    
    def _get_text_color_for_bg(self, hex_color):
        """Devuelve 'white' o 'black' para el texto según el color de fondo."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        # Fórmula de luminosidad perceptual
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return 'white' if luminance < 0.5 else 'black'

    def debug_metricas(self, metricas1, metricas2, equipo1, equipo2):
        """Imprime información detallada de todas las métricas para debug."""
        
        estadisticas_liga = self.analyzer.calcular_estadisticas_liga()
        
        for grupo, lista_metricas in self.analyzer.metricas_agrupadas.items():
            pass
            
            for metrica in lista_metricas:
                # Nombre corto si existe
                nombre_corto = self.analyzer.nombres_cortos.get(metrica, metrica)
                
                # Valores originales
                val1 = metricas1.get(metrica, 0)
                val2 = metricas2.get(metrica, 0)
                
                # Estadísticas de la liga
                est = estadisticas_liga.get(metrica, {})
                media_liga = est.get('media_liga', 0)
                maximo_liga = est.get('maximo', 0)
                equipo_max = est.get('equipo_maximo', 'N/A')
                
                # Valores normalizados
                if media_liga > 0:
                    val1_norm = (val1 / media_liga) * 50
                    val2_norm = (val2 / media_liga) * 50
                    max_norm = (maximo_liga / media_liga) * 50
                else:
                    val1_norm = val1
                    val2_norm = val2
                    max_norm = 100
                
                
                # Alertas
                if val1 == 0 and val2 == 0:
                    print(f"     ⚠️  ALERTA: Ambos valores son 0")
                if val1_norm < 1 or val2_norm < 1:
                    print(f"     ⚠️  ALERTA: Valores normalizados muy pequeños (< 1)")
                if val1_norm > 150 or val2_norm > 150:
                    print(f"     ⚠️  ALERTA: Valores normalizados muy grandes (> 150)")
        
    
    def _formatear_label(self, metrica):
        """Formatea las etiquetas de las métricas para mejor legibilidad."""
        # Usar nombre corto si existe
        metrica_display = self.analyzer.nombres_cortos.get(metrica, metrica)
        return textwrap.fill(metrica_display, width=20, break_long_words=False)
    
    def obtener_escudo_equipo(self, equipo):
        """Obtiene la ruta del escudo de un equipo."""
        def normalize_word(word):
            word = unicodedata.normalize('NFD', word)
            word = ''.join(char for char in word if unicodedata.category(char) != 'Mn')
            return word.lower().strip()

        if not os.path.exists('assets/escudos'):
            return None

        palabras_ignorar = {'cf', 'fc', 'cd', 'ud', 'rcd', 'rc', 'ca', 'de', 'del', 'la', 'las', 'el', 'los'}
        palabras = equipo.split()
        palabras_normalizadas = [normalize_word(p) for p in palabras if normalize_word(p) not in palabras_ignorar and len(normalize_word(p)) > 2]
        palabras_ordenadas = sorted(palabras_normalizadas, key=len, reverse=True)
        
        all_files = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
        
        for palabra_buscar in palabras_ordenadas:
            for filename in all_files:
                nombre_archivo_norm = normalize_word(os.path.splitext(filename)[0])
                if palabra_buscar in nombre_archivo_norm:
                    return f"assets/escudos/{filename}"
        
        return None
    
    def _resize_image(self, img, target_size=(40, 40)):
        """Redimensiona una imagen manteniendo su aspecto."""
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        return img

    def crear_panel_barras_horizontal(self, ax, grupo, metricas1, metricas2, equipo1, equipo2, invert_axis=False):
        """Crea un gráfico de barras HORIZONTAL con normalización y referencias."""
        ax.set_title(grupo.replace('_', ' '), fontsize=11, fontweight='bold', pad=15, color='#333333')
        
        # Obtener estadísticas de la liga
        estadisticas_liga = self.analyzer.calcular_estadisticas_liga()
        
        # Preparar datos
        metricas_grupo = self.analyzer.metricas_agrupadas.get(grupo, [])
        labels = [self._formatear_label(m) for m in metricas_grupo]
        
        valores1 = [metricas1.get(m, 0) for m in metricas_grupo]
        valores2 = [metricas2.get(m, 0) for m in metricas_grupo]
        
        # Normalizar valores basándose en estadísticas de liga
        valores1_norm = []
        valores2_norm = []

        for i, metrica in enumerate(metricas_grupo):
            est = estadisticas_liga.get(metrica, None)
            if est and est.get('maximo', 0) > 0:
                media = est['media_liga']
                maximo = est.get('maximo')
                minimo = est.get('minimo', 0)
                
                # Si la media es negativa o muy pequeña, usar normalización min-max
                if media <= 0 or abs(media) < 0.1:
                    # Min-Max scaling: (valor - min) / (max - min) * 100
                    rango = maximo - minimo if maximo != minimo else 1
                    valores1_norm.append(((valores1[i] - minimo) / rango) * 100)
                    valores2_norm.append(((valores2[i] - minimo) / rango) * 100)
                else:
                    # Normalización estándar basada en media
                    valores1_norm.append((valores1[i] / media) * 50)
                    valores2_norm.append((valores2[i] / media) * 50)
            else:
                valores1_norm.append(valores1[i])
                valores2_norm.append(valores2[i])
        
        # Invertir valores si invert_axis=True
        if invert_axis:
            valores1_norm = [-v for v in valores1_norm]
            valores2_norm = [-v for v in valores2_norm]
        
        # Posiciones en el eje Y (ahora vertical)
        y_pos = np.arange(len(metricas_grupo)) * 2.0
        bar_height = 0.7
        
        # Crear barras HORIZONTALES
        bars1 = ax.barh(y_pos - bar_height/2, valores1_norm, bar_height, 
                label=equipo1, color=self.colores_equipo['equipo1'], alpha=0.85, zorder=3)
        bars2 = ax.barh(y_pos + bar_height/2, valores2_norm, bar_height,
                        label=equipo2, color=self.colores_equipo['equipo2'], alpha=0.85, zorder=3)
        
        # Añadir línea de media de la liga
        media_x = -50 if invert_axis else 50
        ax.axvline(x=media_x, color='red', linestyle='--', linewidth=1.5, 
                alpha=0.7, zorder=2)
        
        # Calcular límite dinámico del eje X
        todos_valores = valores1_norm + valores2_norm
        if invert_axis:
            max_valor = abs(min(todos_valores)) if todos_valores else 100
            limite_x_min = -max(max_valor * 1.2, 70)
            limite_x_max = 0
        else:
            max_valor = max(todos_valores) if todos_valores else 100
            limite_x_min = 0
            limite_x_max = max(max_valor * 1.2, 70)
        
        # Añadir marca del máximo y escudos
        maximos_norm = []
        for i, metrica in enumerate(metricas_grupo):
            est = estadisticas_liga.get(metrica, {})
            if est and 'equipo_maximo' in est:
                equipo_max = est['equipo_maximo']
                media_val = est.get('media_liga', 1)
                maximo_val_real = est.get('maximo', 0)
                
                # Normalizar máximo
                if media_val > 0:
                    maximo_norm = (maximo_val_real / media_val) * 50
                else:
                    maximo_norm = 100
                
                # Invertir máximo si invert_axis=True
                if invert_axis:
                    maximo_norm = -maximo_norm
                
                maximos_norm.append(maximo_norm)
                
                # Actualizar límite si es necesario
                if invert_axis:
                    if maximo_norm < limite_x_min + 15:
                        limite_x_min = maximo_norm - 20
                else:
                    if maximo_norm > limite_x_max - 15:
                        limite_x_max = maximo_norm + 20
                
                # Barra negra vertical más gruesa
                ax.barh(y_pos[i], maximo_norm, height=0.20, color='black', alpha=0.8, zorder=8)
                
                # Cargar escudo (MÁS PEQUEÑO)
                logo = self.analyzer.load_team_logo(equipo_max, target_size=(35, 35))
                if logo is not None:
                    # Posición: al lado de la barra negra
                    if invert_axis:
                        im_x = maximo_norm - 1
                        box_align = (1, 0.5)
                        text_x = maximo_norm - 2
                        text_ha = 'right'
                    else:
                        im_x = maximo_norm + 1
                        box_align = (0, 0.5)
                        text_x = maximo_norm + 2
                        text_ha = 'left'
                    
                    im_y = y_pos[i]
                    imagebox = OffsetImage(logo, zoom=0.3)  # MÁS PEQUEÑO
                    ab = AnnotationBbox(imagebox, (im_x, im_y), frameon=False, 
                                        xycoords='data', box_alignment=box_align)
                    ax.add_artist(ab)
                    
                    # Valor del máximo al lado del escudo
                    metrica_actual = metricas_grupo[i]
                    texto_max = f'{maximo_val_real:.1f}%' if '%' in metrica_actual else f'{maximo_val_real:.1f}'
                    offset_x = -6.5 if invert_axis else 6.5
                    ax.text(text_x + offset_x, y_pos[i] + 0.3, texto_max, 
                            ha=text_ha, va='top', fontsize=7, color='black', fontweight='bold')
        
        ax.set_xlim(limite_x_min, limite_x_max)
        
        # Añadir valores en las barras
        for i, bar in enumerate(bars1):
            xval = bar.get_width()
            valor_abs = abs(xval)
            if valor_abs > 1:
                metrica_actual = metricas_grupo[i]
                label_texto = f'{valores1[i]:.1f}%' if '%' in metrica_actual else f'{valores1[i]:.1f}'
                
                # Posición del texto dentro o fuera según el espacio
                if valor_abs < 20:
                    if invert_axis:
                        x_text = xval - 2
                        ha = 'right'
                    else:
                        x_text = xval + 2
                        ha = 'left'
                    color = '#333333'
                else:
                    if invert_axis:
                        x_text = xval + 2
                        ha = 'left'
                    else:
                        x_text = xval - 2
                        ha = 'right'
                    color = 'white'
                
                ax.text(x_text, bar.get_y() + bar.get_height()/2, label_texto,
                    ha=ha, va='center', fontsize=5, color=color, fontweight='bold')
        
        for i, bar in enumerate(bars2):
            xval = bar.get_width()
            valor_abs = abs(xval)
            if valor_abs > 5:
                metrica_actual = metricas_grupo[i]
                label_texto = f'{valores2[i]:.1f}%' if '%' in metrica_actual else f'{valores2[i]:.1f}'
                
                if valor_abs < 20:
                    if invert_axis:
                        x_text = xval - 2
                        ha = 'right'
                    else:
                        x_text = xval + 2
                        ha = 'left'
                    color = '#333333'
                else:
                    if invert_axis:
                        x_text = xval + 2
                        ha = 'left'
                    else:
                        x_text = xval - 2
                        ha = 'right'
                    color = '#333333'
                
                ax.text(x_text, bar.get_y() + bar.get_height()/2, label_texto,
                    ha=ha, va='center', fontsize=5, color=color, fontweight='bold')
        
        # Configurar etiquetas del eje Y (ahora las métricas)
        ax.set_yticks(y_pos)
        if invert_axis:
            ax.yaxis.tick_right()  # Etiquetas a la derecha
            ax.set_yticklabels(labels, fontsize=7, color='#333333')
            # Alternar negritas en las etiquetas
            for i, label in enumerate(ax.get_yticklabels()):
                if i % 2 == 0:  # Posiciones pares en negrita
                    label.set_fontweight('bold')
                else:
                    label.set_fontweight('normal')
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_color('lightgrey')
        else:
            ax.set_yticklabels(labels, fontsize=7, color='#333333')
            # Alternar negritas en las etiquetas
            for i, label in enumerate(ax.get_yticklabels()):
                if i % 2 == 0:  # Posiciones pares en negrita
                    label.set_fontweight('bold')
                else:
                    label.set_fontweight('normal')
        
        # Estilo moderno
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('lightgrey')
        ax.spines['bottom'].set_visible(False)
        ax.grid(axis='x', linestyle=':', alpha=0.5, linewidth=0.5, zorder=0)
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', left=False)
        
        ax.set_facecolor('none')
        
    def crear_reporte_comparativo(self, equipo1, equipo2, jornada=None, figsize=(11.69, 8.27)):
        """Crea reporte comparativo A4 con layout de 8 columnas y barras horizontales/verticales."""
        metricas1, metricas2 = self.analyzer.obtener_metricas_comparativas(equipo1, equipo2, jornada)
        if not metricas1 or not metricas2:
            pass
            return None
        # DEBUG: Imprimir análisis de métricas
        self.debug_metricas(metricas1, metricas2, equipo1, equipo2)

        fig = plt.figure(figsize=figsize, facecolor='white')
        if os.path.exists('assets/fondo_informes.png'):
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(plt.imread('assets/fondo_informes.png'), extent=[0, 1, 0, 1], aspect='auto', alpha=0.15)
            ax_bg.axis('off')
        
        fig.suptitle('Análisis comparativo', fontsize=18, fontweight='bold', y=0.97)
        plt.figtext(0.5, 0.91, 'Con Balón', fontsize=14, ha='center', style='italic')

        logo1 = self.analyzer.load_team_logo(equipo1, target_size=(100, 100))
        logo2 = self.analyzer.load_team_logo(equipo2, target_size=(100, 100))
        
        if logo1 is not None:
            ax_logo1 = fig.add_axes([0.01, 0.90, 0.1, 0.1], anchor='NW', zorder=10)
            ax_logo1.imshow(logo1)
            ax_logo1.axis('off')
        
        if logo2 is not None:
            ax_logo2 = fig.add_axes([0.88, 0.90, 0.1, 0.1], anchor='NE', zorder=10)
            ax_logo2.imshow(logo2)
            ax_logo2.axis('off')

        gs = GridSpec(2, 8, figure=fig, hspace=0.2, wspace=0.2, 
                      left=0.11, right=0.89, bottom=0.08, top=0.85)

        # --- FILA SUPERIOR: BARRAS - RADAR - BARRAS ---
        ax_verticalidad = fig.add_subplot(gs[0, 0:2])
        self.crear_panel_barras_horizontal(ax_verticalidad, 'VERTICALIDAD', metricas1, metricas2, equipo1, equipo2)

        ax_radar = fig.add_subplot(gs[0, 2:6], projection='polar')
        self.crear_grafico_pizza(ax_radar, metricas1, metricas2, equipo1, equipo2, fig)

        ax_peligro = fig.add_subplot(gs[0, 6:8])
        self.crear_panel_barras_horizontal(ax_peligro, 'CREACION_PELIGRO', metricas1, metricas2, equipo1, equipo2, invert_axis=True)

        # --- FILA INFERIOR: BARRAS - CAMPOGRAMAS - BARRAS ---
        ax_posesion = fig.add_subplot(gs[1, 0:2])
        self.crear_panel_barras_vertical(ax_posesion, 'POSESION', metricas1, metricas2, equipo1, equipo2)

        ax_campo1 = fig.add_subplot(gs[1, 2:4])
        self.crear_campograma(ax_campo1, metricas1, equipo1, self.colores_equipo['equipo1'])

        ax_campo2 = fig.add_subplot(gs[1, 4:6])
        self.crear_campograma(ax_campo2, metricas2, equipo2, self.colores_equipo['equipo2'])

        ax_duelos = fig.add_subplot(gs[1, 6:8])
        self.crear_panel_barras_vertical(ax_duelos, 'DUELOS_JUEGO_AEREO', metricas1, metricas2, equipo1, equipo2)

        return fig

    def crear_panel_barras_vertical(self, ax, grupo, metricas1, metricas2, equipo1, equipo2):
        """Crea un gráfico de barras VERTICAL con puntuaciones."""
        ax.set_title(grupo.replace('_', ' '), fontsize=11, fontweight='bold', pad=15, color='#333333')
        
        # --- [Esta parte de carga y normalización de datos no cambia] ---
        estadisticas_liga = self.analyzer.calcular_estadisticas_liga()
        metricas_grupo = self.analyzer.metricas_agrupadas.get(grupo, [])
        labels = [self._formatear_label(m) for m in metricas_grupo]
        valores1 = [metricas1.get(m, 0) for m in metricas_grupo]
        valores2 = [metricas2.get(m, 0) for m in metricas_grupo]
        valores1_norm, valores2_norm, maximos_norm, equipos_max, valores_max_real = [], [], [], [], []
        for i, metrica in enumerate(metricas_grupo):
            est = estadisticas_liga.get(metrica, {})
            media_val = est.get('media_liga', 1)
            if est and media_val is not None and media_val > 0:
                valores1_norm.append((valores1[i] / media_val) * 50)
                valores2_norm.append((valores2[i] / media_val) * 50)
                maximos_norm.append((est.get('maximo', media_val) / media_val) * 50)
            else:
                valores1_norm.append(valores1[i] if valores1[i] is not None else 0)
                valores2_norm.append(valores2[i] if valores2[i] is not None else 0)
                maximos_norm.append(100)
            equipos_max.append(est.get('equipo_maximo', ''))
            valores_max_real.append(est.get('maximo', 0))

        x_pos = np.arange(len(labels)) * 2.0
        bar_width = 0.50
        
        bars1 = ax.bar(x_pos - bar_width/2, valores1_norm, bar_width, 
                    color=self.colores_equipo['equipo1'], alpha=0.9, zorder=10)
        bars2 = ax.bar(x_pos + bar_width/2, valores2_norm, bar_width, 
                    color=self.colores_equipo['equipo2'], alpha=0.9, zorder=10)
        
        ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5, label='Media Liga')
        
        # --- [La parte de los escudos y máximos no cambia] ---
        for i, (maximo, eq_max, val_max_real) in enumerate(zip(maximos_norm, equipos_max, valores_max_real)):
            ax.bar(x_pos[i], maximo, width=0.20, color='black', alpha=0.8, zorder=8)
            if eq_max:
                logo = self.analyzer.load_team_logo(eq_max, target_size=(40, 40))
                if logo is not None:
                    im_x = x_pos[i]
                    im_y = maximo + 8
                    imagebox = OffsetImage(logo, zoom=0.4)
                    ab = AnnotationBbox(imagebox, (im_x, im_y), frameon=False, xycoords='data', box_alignment=(0.5, 0))
                    ax.add_artist(ab)
                    metrica_actual = metricas_grupo[i]
                    texto_max = f'{val_max_real:.1f}%' if '%' in metrica_actual else f'{val_max_real:.1f}'
                    ax.text(x_pos[i], maximo + 2, texto_max, ha='center', va='bottom', fontsize=6, color='black', fontweight='bold')

        max_y = max(max(maximos_norm) + 25, 100)
        ax.set_ylim(0, max_y)
        
        # --- LÓGICA DE TEXTO CORREGIDA (IMITANDO LAS BARRAS HORIZONTALES) ---
        UMBRAL_ALTURA = 30 # Si la barra es más alta que esto, el texto va dentro.

        for i, bar in enumerate(bars1):
            yval = bar.get_height()
            if yval > 1:
                metrica_actual = metricas_grupo[i]
                label_texto = f'{valores1[i]:.1f}%' if '%' in metrica_actual else f'{valores1[i]:.1f}'
                
                if yval < UMBRAL_ALTURA:
                    # TEXTO AFUERA: Un poco por encima de la barra
                    y_text = yval + 3
                    va = 'bottom' # Alinea la base del texto rotado
                    color_text = '#333333'
                else:
                    # TEXTO ADENTRO: Un poco por debajo del tope de la barra
                    y_text = yval - 3
                    va = 'top' # Alinea el tope del texto rotado (así crece hacia abajo)
                    color_text = self._get_text_color_for_bg(self.colores_equipo['equipo1'])
                
                ax.text(bar.get_x() + bar.get_width()/2.0, y_text, label_texto, 
                        ha='center', va=va, fontsize=6, color=color_text, 
                        fontweight='bold', rotation=90, zorder=12)

        for i, bar in enumerate(bars2):
            yval = bar.get_height()
            if yval > 1:
                metrica_actual = metricas_grupo[i]
                label_texto = f'{valores2[i]:.1f}%' if '%' in metrica_actual else f'{valores2[i]:.1f}'
                
                if yval < UMBRAL_ALTURA:
                    # TEXTO AFUERA
                    y_text = yval + 3
                    va = 'bottom'
                    color_text = '#333333'
                else:
                    # TEXTO ADENTRO
                    y_text = yval - 3
                    va = 'top'
                    color_text = self._get_text_color_for_bg(self.colores_equipo['equipo2'])
                
                ax.text(bar.get_x() + bar.get_width()/2.0, y_text, label_texto, 
                        ha='center', va=va, fontsize=6, color=color_text, 
                        fontweight='bold', rotation=90, zorder=12) 
        
        # --- [El código final de configuración de ejes no cambia] ---
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=5, ha='center', color='#333333', rotation=90, rotation_mode='anchor', va='top')
        # Alternar negritas en las etiquetas
        for i, label in enumerate(ax.get_xticklabels()):
            if i % 2 == 0:  # Posiciones pares en negrita
                label.set_fontweight('bold')
            else:
                label.set_fontweight('normal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('lightgrey')
        ax.grid(axis='y', linestyle=':', alpha=0.5, linewidth=0.5, zorder=0)
        ax.tick_params(axis='y', left=False, labelleft=False)
        ax.tick_params(axis='x', bottom=False, pad=25)
        ax.set_facecolor('none')

    def crear_grafico_pizza(self, ax, metricas1, metricas2, equipo1, equipo2, fig):
        """Crea un gráfico de pizza (PyPizza) comparativo entre dos equipos."""
        
        categorias_keys = ['POSESION', 'CREACION_PELIGRO', 'VERTICALIDAD', 'DUELOS_JUEGO_AEREO', 'POSICIONAMIENTO']
        categorias_labels = ['Posesión', 'Peligro', 'Verticalidad', 'Duelos', 'Posición']
        
        # Calcular puntuaciones normalizadas (0-10 scale)
        puntuaciones1 = []
        puntuaciones2 = []
        
        for cat_key in categorias_keys:
            punt1 = self.analyzer.calcular_puntuacion_grupo(metricas1, cat_key)
            punt2 = self.analyzer.calcular_puntuacion_grupo(metricas2, cat_key)
            # Convertir de escala 1-10 a 0-100 para PyPizza
            puntuaciones1.append(round(punt1 * 10, 1))
            puntuaciones2.append(round(punt2 * 10, 1))
        
        # Crear instancia de PyPizza
        baker = PyPizza(
            params=categorias_labels,
            background_color="#FFFFFF",
            straight_line_color="#222222",
            straight_line_lw=1,
            last_circle_lw=1,
            last_circle_color="#222222",
            other_circle_ls="-.",
            other_circle_lw=1
        )
        
        # Dibujar el gráfico de pizza
        baker.make_pizza(
            puntuaciones1,
            compare_values=puntuaciones2,
            ax=ax,
            kwargs_slices=dict(
                facecolor=self.colores_equipo['equipo1'], 
                edgecolor="#222222",
                zorder=2, 
                linewidth=1
            ),
            kwargs_compare=dict(
                facecolor=self.colores_equipo['equipo2'], 
                edgecolor="#222222",
                zorder=2, 
                linewidth=1
            ),
            kwargs_params=dict(
                color="#000000", 
                fontsize=10,
                va="center",
                fontweight='bold'
            ),
            kwargs_values=dict(
                color="#000000", 
                fontsize=8,
                zorder=3,
                bbox=dict(
                    edgecolor="#222222", 
                    facecolor=self.colores_equipo['equipo1'],
                    boxstyle="round,pad=0.2", 
                    lw=1
                )
            ),
            kwargs_compare_values=dict(
                color="#000000", 
                fontsize=8, 
                zorder=3,
                bbox=dict(
                    edgecolor="#222222", 
                    facecolor=self.colores_equipo['equipo2'], 
                    boxstyle="round,pad=0.2", 
                    lw=1
                )
            )
        )

        # Añadir leyenda explicativa para Posición
        ax.text(0.5, -0.08, 
                '*Posición: Mayor si Amplitud, Profundidad, Dist. Línea Def. y Centroide ↑',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=6, style='italic', color='#666666',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor='#cccccc', alpha=0.8))


    def crear_campograma(self, ax, metricas, equipo, color):
        """Crea una visualización en el campo de juego con la lógica de posicionamiento corregida."""
        pitch = VerticalPitch(pitch_type='opta', pitch_color='#2d5a27', line_color='white', linewidth=1)
        pitch.draw(ax=ax)
        ax.set_title(equipo, fontsize=9, color='black', pad=3)
        
        profundidad = metricas.get('Profundidad Posicional Global (m.)', 0)
        anchura = metricas.get('Anchura Posicional Global (m.)', 0)
        linea_def = metricas.get('Distancia Línea Defensiva Global (m. Propia Puerta)', 0)
        centroide_altura = metricas.get('Centroide Colectivo Ofensivo (m.)', 0)
        
        # ESCALAR anchura para que se vea más ancha (x2)
        anchura_escalada = anchura * 2
        
        y_start = linea_def
        x_start = 50 - (anchura_escalada / 2)
        
        rect = patches.Rectangle((x_start, y_start), anchura_escalada, profundidad, 
                                linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        
        pitch.scatter(centroide_altura, 50, s=150, color='grey', edgecolors='white', lw=1.5, zorder=5, ax=ax)
        
        texto_labels = (f"Dist. Línea Def.: {linea_def:.1f} m\n"
                        f"Profundidad: {profundidad:.1f} m\n"
                        f"Anchura: {anchura:.1f} m\n"
                        f"Centroide Of.: {centroide_altura:.1f} m")
        
        ax.text(50, -8, texto_labels, ha='center', va='top', fontsize=6, color='black',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    def guardar_reporte(self, fig, equipo1, equipo2, jornada=None):
        """Guarda el reporte en PDF."""
        if fig is None: 
            return
        equipo1_fn = equipo1.replace(' ', '_')
        equipo2_fn = equipo2.replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if jornada:
            filename = f"analisis_con_balon_{equipo1_fn}_vs_{equipo2_fn}_J{jornada}_{timestamp}.pdf"
        else:
            filename = f"analisis_con_balon_{equipo1_fn}_vs_{equipo2_fn}_promedio_{timestamp}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.2, format='pdf', orientation='landscape')


def main():
    """Función principal interactiva para generar el informe."""
    
    analyzer = KPIConBalonMediaCoach()
    if analyzer.cargar_datos() is None:
        pass
        return
        
    equipos_disponibles = analyzer.obtener_equipos_disponibles()
    
    villarreal_nombre = next((eq for eq in equipos_disponibles if 'villarreal' in eq.lower()), None)
            
    if not villarreal_nombre:
        pass
        return

    
    equipos_comparar = [eq for eq in equipos_disponibles if eq != villarreal_nombre]
    for i, equipo in enumerate(equipos_comparar, 1):
        pass
    
    equipo_oponente = None
    for _ in range(3):
        try:
            seleccion = int(input(f"\nSelecciona un número (1-{len(equipos_comparar)}): "))
            if 1 <= seleccion <= len(equipos_comparar):
                equipo_oponente = equipos_comparar[seleccion - 1]
                break
            else:
                pass
        except EOFError:
            equipo_oponente = equipos_comparar[0] if equipos_comparar else None
            break
        except ValueError:
            pass
    if equipo_oponente is None and equipos_comparar:
        equipo_oponente = equipos_comparar[0]

    # Análisis siempre basado en el promedio de toda la temporada
    jornada_seleccionada = None
    
        
    visualizador = VisualizadorKPIConBalon(analyzer)
    
    fig = visualizador.crear_reporte_comparativo(equipo_oponente, villarreal_nombre, jornada_seleccionada)
    
    if fig:
        pass
        plt.show()
        visualizador.guardar_reporte(fig, equipo_oponente, villarreal_nombre, jornada_seleccionada)
    else:
        pass


if __name__ == "__main__":
    main()