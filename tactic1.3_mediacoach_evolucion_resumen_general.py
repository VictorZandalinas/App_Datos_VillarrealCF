import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from datetime import datetime
import os
import warnings
import unicodedata
from PIL import Image

warnings.filterwarnings('ignore')

class EvolucionRendimientoMediaCoach:
    def __init__(self, parquet_path='extraccion_mediacoach/data/estadisticas_equipo.parquet'):
        self.parquet_path = parquet_path
        self.df_stats = None
        
        # Grupos ofensivos (CON BALÓN)
        self.grupos_ofensivos = {
            'VERTICALIDAD': [
                'Pases hacia adelante (Nº)',
                'Pases profundos totales (Nº)',
                'Regates totales (Nº)',
                'Centros totales (Nº)',
                'Pases largos cambio de orientación (Nº)',
                'Pases largos totales (Nº)',
                'Pases hacia adelante (% Éxito)',
                'Pases profundos totales con exito (%)',
                'Regates totales (% Éxito)',
                'Centros totales (% Éxito)',
                'Modo de Canalización (% Pase Largo)',
                'Pases largos cambio de orientación (% Éxito)'
            ],
            'CREACION_PELIGRO': [
                'Remates totales minuto 90 (Nº)',
                'Remates juego dinámico (Nº)',
                'Pases Propiciadores Remate (Nº)',
                'Remates a portería (Nº)',
                'Remates goles (Nº)',
                'Aprovechamiento Ofensivo (Goles a favor)',
                'Peligrosidad Ofensiva (puntos xG)',
                'Eficiencia Ofensiva (Nº Goles a Favor/Remate)',
                'Eficacia Finalización (%)',
                'Pases Propiciadores Remate (% Total)',
                'Remates a portería (% Total)',
                'Remates goles (% Total)'
            ],
            'POSESION': [
                'Ritmo de Circulación (Pases/Min)',
                'Pases totales (Precisos)',
                'Pases totales (Nº)',
                'Posesión del Balón (%)',
                'Posesion de balon en campo propio (%)',
                'Posesion de balon en campo rival (%)',
                'Pases totales (% Éxito)'
            ]
        }
        
        # Grupos defensivos (SIN BALÓN)
        self.grupos_defensivos = {
            'PTP_VIGILANCIAS': [
                'Goles en contra J. D. contraataque - ataque rápido (Nº)',
                'Recuperaciones Rápidas <5 (Nº)'
            ],
            'FIABILIDAD': [
                'Eficacia Evitación (%)',
                'Eficiencia Ofensiva Rival (Nº Goles en Contra/Remate Rival)',
                'Goles en contra Totales (Nº)',
                'Remates Bloqueados Defensivos (Nº)',
                'Ratio Recuperaciones - Pérdidas (Nº)'
            ],
            'PRESION_ALTA': [
                'Ritmo de Recuperación (Recuperaciones/Min)',
                'Recuperaciones en campo contrario (Nº)',
                'Pases totales (% Éxito)',
            ],
            'DUELOS_JUEGO_AEREO': [
                'Duelos Aéreos Defensivos (Nº)',
                'Duelos Aéreos Defensivos (% Éxito)',
                'Duelos Aéreos Defensivos (% Posesión Ganada)',
                'Duelos Tackle Defensivos (Nº)',
                'Duelos Tackle Defensivos (% Éxito)'
            ],
            'REPLIEGUE': [
                'Goles en contra J. D. ataque (Nº)',
                'Eficacia Contención Defensiva (%)',
                'Recuperaciones totales / Altura media (Nº m.)',
                'Despejes (Nº)',
                'Centros totales (% Éxito)'
            ]
        }
        
        # Métricas invertidas (menos es mejor)
        self.metricas_invertidas = [
            'Goles en contra J. D. contraataque - ataque rápido (Nº)',
            'Goles en contra Totales (Nº)',
            'Goles en contra J. D. ataque (Nº)',
            'Eficiencia Ofensiva Rival (Nº Goles en Contra/Remate Rival)'
        ]
        
        # Nombres cortos para visualización
        self.nombres_cortos = {
            # Ofensivos
            'Pases hacia adelante (Nº)': 'Pases adelante',
            'Pases profundos totales (Nº)': 'Pases profundos',
            'Regates totales (Nº)': 'Regates',
            'Remates totales minuto 90 (Nº)': 'Remates 90´',
            'Aprovechamiento Ofensivo (Goles a favor)': 'Goles favor',
            'Peligrosidad Ofensiva (puntos xG)': 'Puntos xG',
            'Posesión del Balón (%)': 'Posesión %',
            'Eficacia Finalización (%)': 'Eficacia final. %',
            
            # Defensivos
            'Goles en contra J. D. contraataque - ataque rápido (Nº)': 'Goles contraataq.',
            'Recuperaciones Rápidas <5 (Nº)': 'Recup. rápidas',
            'Eficacia Evitación (%)': 'Eficacia evit. %',
            'Goles en contra Totales (Nº)': 'Goles contra',
            'Recuperaciones en campo contrario (Nº)': 'Recup. rival',
            'Duelos Aéreos Defensivos (% Éxito)': 'Duelos aéreos %',
            'Goles en contra J. D. ataque (Nº)': 'Goles ataque',
            'Eficacia Contención Defensiva (%)': 'Contención %'
        }
    
    def load_team_logo(self, equipo, target_size=(80, 80)):
        """Carga el logo del equipo."""
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
                    logo_path = f"assets/escudos/{filename}"
                    try:
                        with Image.open(logo_path) as img:
                            img = img.convert('RGBA')
                            img.thumbnail(target_size, Image.Resampling.LANCZOS)
                            return np.array(img)
                    except Exception as e:
                        continue
        return None
    
    def cargar_datos(self):
        """Carga datos del parquet."""
        try:
            self.df_stats = pd.read_parquet(self.parquet_path)
            self.df_stats['VALOR'] = pd.to_numeric(
                self.df_stats['VALOR'].astype(str).str.replace(',', '.'), errors='coerce'
            )
            self.df_stats.dropna(subset=['VALOR'], inplace=True)
            self.df_stats['NOMBRE MÉTRICA'] = self.df_stats['NOMBRE MÉTRICA'].str.strip()
            return self.df_stats
        except Exception as e:
            pass
            return None
    
    def obtener_equipos_disponibles(self):
        """Obtiene lista de equipos únicos."""
        if self.df_stats is not None:
            return sorted(self.df_stats['EQUIPO'].unique())
        return []
    
    def calcular_estadisticas_liga(self):
        """Calcula estadísticas de la liga para normalización."""
        estadisticas = {}
        datos_partido = self.df_stats[self.df_stats['PERIODO'] == 'Total Partido']
        
        # Combinar todos los grupos
        todas_metricas = {}
        todas_metricas.update(self.grupos_ofensivos)
        todas_metricas.update(self.grupos_defensivos)
        
        for grupo, lista_metricas in todas_metricas.items():
            for metrica in lista_metricas:
                datos_metrica = datos_partido[datos_partido['NOMBRE MÉTRICA'] == metrica]
                if not datos_metrica.empty:
                    valores_por_equipo = datos_metrica.groupby('EQUIPO')['VALOR'].mean()
                    estadisticas[metrica] = {
                        'media_liga': valores_por_equipo.mean(),
                        'maximo': valores_por_equipo.max(),
                        'minimo': valores_por_equipo.min()
                    }
        return estadisticas
    
    def calcular_puntuacion_metrica(self, valor, metrica, estadisticas):
        """Calcula puntuación 1-10 para una métrica."""
        est = estadisticas.get(metrica, {})
        maximo = est.get('maximo', 1)
        minimo = est.get('minimo', 0)
        
        if maximo == minimo or maximo == 0:
            return 5
        
        # Determinar si es invertida
        es_invertida = metrica in self.metricas_invertidas
        
        if es_invertida:
            puntuacion = 1 + ((maximo - valor) / (maximo - minimo)) * 9
        else:
            puntuacion = 1 + ((valor - minimo) / (maximo - minimo)) * 9
        
        return max(1, min(10, puntuacion))
    
    def calcular_rendimiento_jornada(self, equipo, jornada, tipo='ofensivo'):
        """Calcula el rendimiento ofensivo o defensivo de un equipo en una jornada."""
        datos = self.df_stats[
            (self.df_stats['EQUIPO'] == equipo) &
            (self.df_stats['jornada'] == jornada) &
            (self.df_stats['PERIODO'] == 'Total Partido')
        ]
        
        if datos.empty:
            return None
        
        estadisticas = self.calcular_estadisticas_liga()
        grupos = self.grupos_ofensivos if tipo == 'ofensivo' else self.grupos_defensivos
        
        puntuaciones = []
        for grupo, metricas in grupos.items():
            for metrica in metricas:
                valor = datos[datos['NOMBRE MÉTRICA'] == metrica]['VALOR'].values
                if len(valor) > 0:
                    punt = self.calcular_puntuacion_metrica(valor[0], metrica, estadisticas)
                    puntuaciones.append(punt)
        
        return np.mean(puntuaciones) if puntuaciones else 5
    
    def obtener_rival_jornada(self, equipo, jornada):
        """Obtiene el rival de un equipo en una jornada específica."""
        # Buscar el ID PARTIDO del equipo en esa jornada
        datos_equipo = self.df_stats[
            (self.df_stats['EQUIPO'] == equipo) &
            (self.df_stats['jornada'] == jornada)
        ]
        
        if datos_equipo.empty:
            return None
        
        id_partido = datos_equipo['ID PARTIDO'].iloc[0]
        
        # Buscar el otro equipo en ese partido
        equipos_partido = self.df_stats[
            self.df_stats['ID PARTIDO'] == id_partido
        ]['EQUIPO'].unique()
        
        rival = [eq for eq in equipos_partido if eq != equipo]
        return rival[0] if rival else None
    
    def es_partido_local(self, equipo, jornada):
        """
        Determina si un equipo jugó como local en una jornada.
        Lee la columna 'partido' y verifica si el equipo aparece antes del guión (-).
        Ejemplo: 'gironafc0-2sevillafc' -> gironafc es local, sevillafc visitante
        """
        # Buscar el partido del equipo en esa jornada
        datos_equipo = self.df_stats[
            (self.df_stats['EQUIPO'] == equipo) &
            (self.df_stats['jornada'] == jornada)
        ]
        
        if datos_equipo.empty:
            return False
        
        # Obtener el string del partido
        if 'partido' not in datos_equipo.columns:
            # Si no existe la columna, usar método alternativo
            id_partido = datos_equipo['ID PARTIDO'].iloc[0]
            todos_equipos = self.df_stats[
                self.df_stats['ID PARTIDO'] == id_partido
            ]['EQUIPO'].unique()
            return equipo == sorted(todos_equipos)[0]
        
        partido_str = datos_equipo['partido'].iloc[0].lower()
        
        # Normalizar nombre del equipo (quitar espacios, convertir a minúsculas)
        equipo_norm = equipo.lower().replace(' ', '').replace('.', '')
        
        # Separar por el guión (formato: equipolocal0-2equipovisitante)
        # Buscar el guión con números alrededor
        import re
        match = re.search(r'(\d+)-(\d+)', partido_str)
        
        if match:
            # Posición del guión
            pos_guion = match.start()
            # Todo antes del primer número es el equipo local
            parte_antes = partido_str[:pos_guion].rstrip('0123456789')
            
            # Si el equipo normalizado está en la parte antes del guión, es local
            return equipo_norm in parte_antes
        
        # Si no se encuentra el patrón, devolver False
        return False
    
    def calcular_ranking_liga_metrica(self, metrica):
        """Calcula el ranking de todos los equipos para una métrica específica."""
        datos = self.df_stats[
            (self.df_stats['NOMBRE MÉTRICA'] == metrica) &
            (self.df_stats['PERIODO'] == 'Total Partido')
        ]
        
        if datos.empty:
            return {}
        
        # Calcular promedio por equipo
        promedios = datos.groupby('EQUIPO')['VALOR'].mean()
        
        # Determinar si es métrica invertida
        es_invertida = metrica in self.metricas_invertidas
        
        # Ordenar (descendente para normales, ascendente para invertidas)
        promedios_ordenados = promedios.sort_values(ascending=es_invertida)
        
        # Crear ranking (1 = mejor)
        ranking = {}
        for posicion, (equipo, valor) in enumerate(promedios_ordenados.items(), 1):
            ranking[equipo] = posicion
        
        return ranking
    
    def calcular_mejores_metricas(self, equipo, tipo='ofensivo', top_n=3):
        """Identifica las mejores métricas de un equipo con su ranking de liga."""
        datos = self.df_stats[
            (self.df_stats['EQUIPO'] == equipo) &
            (self.df_stats['PERIODO'] == 'Total Partido')
        ]
        
        if datos.empty:
            return []
        
        estadisticas = self.calcular_estadisticas_liga()
        grupos = self.grupos_ofensivos if tipo == 'ofensivo' else self.grupos_defensivos
        
        metricas_puntuaciones = []
        for grupo, metricas in grupos.items():
            for metrica in metricas:
                valores = datos[datos['NOMBRE MÉTRICA'] == metrica]['VALOR'].values
                if len(valores) > 0:
                    valor_promedio = valores.mean()
                    punt = self.calcular_puntuacion_metrica(valor_promedio, metrica, estadisticas)
                    nombre_corto = self.nombres_cortos.get(metrica, metrica)
                    
                    # Calcular ranking en la liga
                    ranking_liga = self.calcular_ranking_liga_metrica(metrica)
                    posicion_liga = ranking_liga.get(equipo, 0)
                    
                    metricas_puntuaciones.append({
                        'metrica': nombre_corto,
                        'metrica_completa': metrica,
                        'valor': valor_promedio,
                        'puntuacion': punt,
                        'posicion_liga': posicion_liga
                    })
        
        # Ordenar por puntuación y tomar las top N
        metricas_puntuaciones.sort(key=lambda x: x['puntuacion'], reverse=True)
        return metricas_puntuaciones[:top_n]


class VisualizadorEvolucion:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.colores = {
            'equipo1': '#d62728',  # Rojo
            'equipo2': '#FEE500'   # Amarillo
        }
    
    def crear_informe_evolucion(self, equipo1, equipo2):
        """Crea informe comparativo de evolución de rendimiento entre dos equipos."""
        # Calcular estadísticas de liga
        self.analyzer.calcular_estadisticas_liga()
        
        # Obtener jornadas disponibles
        def extraer_numero_jornada(x):
            import re
            numeros = re.findall(r'\d+', str(x))
            return int(numeros[0]) if numeros else 0
        
        jornadas = sorted(self.analyzer.df_stats['jornada'].unique(), key=extraer_numero_jornada)
        
        if not jornadas:
            pass
            return None
        
        # Crear figura con dimensiones A3 (como tactic1_2)
        fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')

        # Fondo de imagen (igual que tactic1_2)
        if os.path.exists('assets/fondo_informes.png'):
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(plt.imread('assets/fondo_informes.png'), extent=[0, 1, 0, 1], aspect='auto', alpha=0.15)
            ax_bg.axis('off')

        fig.suptitle(f'Evolución del Rendimiento\n{equipo1} vs {equipo2}',
                    fontsize=18, fontweight='bold', y=0.97)
        
        # Logos
        logo1 = self.analyzer.load_team_logo(equipo1, target_size=(100, 100))
        logo2 = self.analyzer.load_team_logo(equipo2, target_size=(100, 100))
        
        if logo1 is not None:
            ax_logo1 = fig.add_axes([0.01, 0.90, 0.08, 0.08], anchor='NW', zorder=10)
            ax_logo1.imshow(logo1)
            ax_logo1.axis('off')
        
        if logo2 is not None:
            ax_logo2 = fig.add_axes([0.91, 0.90, 0.08, 0.08], anchor='NE', zorder=10)
            ax_logo2.imshow(logo2)
            ax_logo2.axis('off')
        
        # GridSpec: 2 filas, 4 columnas (gráficas, rankings, leyenda)
        gs = GridSpec(2, 4, figure=fig, 
                    height_ratios=[1.6, 1.4],  # Gráficas, rankings, leyenda
                    hspace=0.3, wspace=0.45,
                    left=0.05, right=0.96, bottom=0.02, top=0.88)
        
        # FILA 1: Gráficas de evolución separadas (2 columnas cada una)
        ax_evol_equipo1 = fig.add_subplot(gs[0, 0:2])
        self.crear_grafico_evolucion_dual(ax_evol_equipo1, equipo1, jornadas, self.colores['equipo1'])

        ax_evol_equipo2 = fig.add_subplot(gs[0, 2:4])
        self.crear_grafico_evolucion_dual(ax_evol_equipo2, equipo2, jornadas, self.colores['equipo2'])
        
        # Fondo semitransparente para toda la sección de rankings (Posición Y ajustada)
        ax_fondo = fig.add_axes([0.055, 0.1, 0.89, 0.35], zorder=0) # Posición Y bajada a 0.1
        ax_fondo.add_patch(patches.Rectangle((0, 0), 1, 1, 
                                            facecolor='lightgray', 
                                            alpha=0.15, 
                                            transform=ax_fondo.transAxes,
                                            zorder=0))
        ax_fondo.axis('off')

        # FILA 2: Top métricas (1 columna por panel)
        ax_top1_ataque = fig.add_subplot(gs[1, 0])
        self.crear_panel_top_metricas(ax_top1_ataque, equipo1, 'ofensivo', self.colores['equipo1'])

        ax_top1_defensa = fig.add_subplot(gs[1, 1])
        self.crear_panel_top_metricas(ax_top1_defensa, equipo1, 'defensivo', self.colores['equipo1'])

        ax_top2_ataque = fig.add_subplot(gs[1, 2])
        self.crear_panel_top_metricas(ax_top2_ataque, equipo2, 'ofensivo', self.colores['equipo2'])

        ax_top2_defensa = fig.add_subplot(gs[1, 3])
        self.crear_panel_top_metricas(ax_top2_defensa, equipo2, 'defensivo', self.colores['equipo2'])

        # Líneas divisorias verticales entre equipos y entre ataque/defensa
        # ¡AÑADIMOS LA LÍNEA QUE FALTABA AQUÍ!
        y_range_lineas = [0.1, 0.45] # Define el rango vertical para las líneas

        # Línea vertical central (entre equipos)
        fig.add_artist(plt.Line2D([0.5, 0.5], y_range_lineas, 
                                color='gray', linewidth=2.5, alpha=0.6,
                                transform=fig.transFigure, zorder=10))

        # Líneas verticales entre ataque y defensa de cada equipo
        fig.add_artist(plt.Line2D([0.278, 0.278], y_range_lineas, 
                                color='gray', linewidth=1.5, alpha=0.4,
                                linestyle='--', transform=fig.transFigure, zorder=10))

        fig.add_artist(plt.Line2D([0.722, 0.722], y_range_lineas, 
                                color='gray', linewidth=1.5, alpha=0.4,
                                linestyle='--', transform=fig.transFigure, zorder=10))
        
        return fig
    
    def crear_grafico_evolucion(self, ax, equipo1, equipo2, jornadas, tipo):
        """Crea gráfico de líneas mostrando la evolución del rendimiento."""
        titulo = 'Rendimiento Ofensivo' if tipo == 'ofensivo' else 'Rendimiento Defensivo'
        ax.set_title(titulo, fontsize=14, fontweight='bold', pad=10)
        
        # Calcular puntuaciones por jornada
        puntuaciones1 = []
        puntuaciones2 = []
        jornadas_validas = []
        rivales1 = []
        rivales2 = []
        
        for jornada in jornadas:
            punt1 = self.analyzer.calcular_rendimiento_jornada(equipo1, jornada, tipo)
            punt2 = self.analyzer.calcular_rendimiento_jornada(equipo2, jornada, tipo)
            
            if punt1 is not None and punt2 is not None:
                puntuaciones1.append(punt1)
                puntuaciones2.append(punt2)
                jornadas_validas.append(jornada)
                
                # Obtener rivales
                rival1 = self.analyzer.obtener_rival_jornada(equipo1, jornada)
                rival2 = self.analyzer.obtener_rival_jornada(equipo2, jornada)
                rivales1.append(rival1)
                rivales2.append(rival2)
        
        if not jornadas_validas:
            ax.text(0.5, 0.5, 'Sin datos disponibles', ha='center', va='center',
                   fontsize=12, transform=ax.transAxes)
            return
        
        # Autoajustar tamaños según número de jornadas
        n_jornadas = len(jornadas_validas)
        markersize = max(6, 10 - n_jornadas // 8)  # De 10 a 6
        linewidth = max(1.5, 2.5 - n_jornadas // 15)  # De 2.5 a 1.5
        logo_size = max(20, 30 - n_jornadas // 5)  # De 30 a 20
        
        # Dibujar líneas
        ax.plot(jornadas_validas, puntuaciones1, marker='o', linewidth=linewidth, 
               markersize=markersize, color=self.colores['equipo1'], label=equipo1,
               markeredgewidth=1.5, markeredgecolor='white', zorder=3)
        
        ax.plot(jornadas_validas, puntuaciones2, marker='s', linewidth=2.5,
               markersize=8, color=self.colores['equipo2'], label=equipo2,
               markeredgewidth=1.5, markeredgecolor='white', zorder=3)
        
        # Añadir escudos de rivales (pequeños)
        for i, (jornada, punt1, rival1) in enumerate(zip(jornadas_validas, puntuaciones1, rivales1)):
            if rival1:
                logo = self.analyzer.load_team_logo(rival1, target_size=(25, 25))
                if logo is not None:
                    imagebox = OffsetImage(logo, zoom=0.7)
                    # Colocar debajo del punto
                    ab = AnnotationBbox(imagebox, (jornada, punt1 - 0.6), 
                                       frameon=False, zorder=4)
                    ax.add_artist(ab)
        
        for i, (jornada, punt2, rival2) in enumerate(zip(jornadas_validas, puntuaciones2, rivales2)):
            if rival2:
                logo = self.analyzer.load_team_logo(rival2, target_size=(25, 25))
                if logo is not None:
                    imagebox = OffsetImage(logo, zoom=0.7)
                    # Colocar encima del punto
                    ab = AnnotationBbox(imagebox, (jornada, punt2 + 0.6),
                                       frameon=False, zorder=4)
                    ax.add_artist(ab)
        
        # Línea de media
        ax.axhline(y=5.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Media')
        
        # Configuración
        ax.set_xlabel('Jornada', fontsize=11, fontweight='bold')
        ax.set_ylabel('Puntuación (1-10)', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 11)  # Aumentado para dar espacio a los escudos
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        # Estilo
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#f9f9f9')
    
    def crear_grafico_evolucion_dual(self, ax, equipo, jornadas, color_equipo):
        """
        Crea gráfico con líneas de ataque y defensa para un solo equipo.
        Colorea el área entre líneas según mejor rendimiento.
        Muestra V/L para visitante/local y escudos de rivales en eje X.
        """
        
        # Calcular puntuaciones
        puntuaciones_ataque = []
        puntuaciones_defensa = []
        jornadas_validas = []
        rivales = []
        es_local = []  # True si juega en casa
        
        for jornada in jornadas:
            punt_ataque = self.analyzer.calcular_rendimiento_jornada(equipo, jornada, 'ofensivo')
            punt_defensa = self.analyzer.calcular_rendimiento_jornada(equipo, jornada, 'defensivo')
            
            if punt_ataque is not None and punt_defensa is not None:
                puntuaciones_ataque.append(punt_ataque)
                puntuaciones_defensa.append(punt_defensa)
                jornadas_validas.append(jornada)
                
                # Obtener rival y si es local
                rival = self.analyzer.obtener_rival_jornada(equipo, jornada)
                rivales.append(rival)
                
                # Determinar si es local (puedes ajustar esta lógica según tus datos)
                local = self.analyzer.es_partido_local(equipo, jornada)
                es_local.append(local)
        
        if not jornadas_validas:
            ax.text(0.5, 0.5, 'Sin datos disponibles', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
            return
        
        # Convertir a arrays para manipulación
        x = np.array(jornadas_validas)
        y_ataque = np.array(puntuaciones_ataque)
        y_defensa = np.array(puntuaciones_defensa)
        
        # Autoajustar tamaños según número de jornadas
        n_jornadas = len(jornadas_validas)
        markersize = max(6, 10 - n_jornadas // 8)  # De 10 a 6
        linewidth = max(1.5, 2.5 - n_jornadas // 15)  # De 2.5 a 1.5
        logo_size = max(20, 30 - n_jornadas // 5)  # De 30 a 20
        tam_texto = max(6, 9 - n_jornadas // 10)  # Tamaño texto V/L
        
        # Colorear área entre líneas según mejor rendimiento
        for i in range(len(x) - 1):
            # Si ataque > defensa, color más rojizo; si defensa > ataque, más azulado
            if y_ataque[i] > y_defensa[i]:
                color_fill = '#FF6B6B'  # Rojo suave (mejor ataque)
                alpha_fill = 0.3
            else:
                color_fill = '#4ECDC4'  # Azul/verde suave (mejor defensa)
                alpha_fill = 0.3
            
            # Rellenar área entre líneas
            ax.fill_between([x[i], x[i+1]], 
                            [y_ataque[i], y_ataque[i+1]], 
                            [y_defensa[i], y_defensa[i+1]],
                            color=color_fill, alpha=alpha_fill, zorder=1)
        
        # Dibujar líneas
        ax.plot(x, y_ataque, marker='o', linewidth=linewidth, markersize=markersize,
            color='#E74C3C', label='Ataque', markeredgewidth=2, 
            markeredgecolor='white', zorder=3)
        
        ax.plot(x, y_defensa, marker='s', linewidth=linewidth, markersize=markersize,
            color='#3498DB', label='Defensa', markeredgewidth=2,
            markeredgecolor='white', zorder=3)
        
        # Añadir marcadores V/L debajo del eje X
        for i, (jorn, local) in enumerate(zip(x, es_local)):
            marcador = 'L' if local else 'V'
            
            ax.text(jorn, -1.5, marcador,
                ha='center', va='center',
                fontsize=tam_texto, fontweight='bold',
                color='#333333',
                bbox=dict(boxstyle='circle,pad=0.15',
                            facecolor='white',
                            edgecolor='#666666',
                            linewidth=1),
                zorder=5)
        
        # Configurar eje X con jornadas y escudos
        ax.set_xlabel('Jornada', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'J{str(j).replace("j", "").replace("J", "")}' for j in x], fontsize=9)
        
        # Añadir escudos de rivales en el eje X (debajo de las etiquetas)
        for i, (jorn, rival) in enumerate(zip(x, rivales)):
            if rival:
                logo = self.analyzer.load_team_logo(rival, target_size=(logo_size, logo_size))
                if logo is not None:
                    imagebox = OffsetImage(logo, zoom=0.6)
                    # Posicionar debajo del eje X
                    ab = AnnotationBbox(imagebox, (jorn, -0.4),
                                    frameon=False, 
                                    xycoords='data',
                                    box_alignment=(0.5, 1.0),
                                    zorder=5)
                    ax.add_artist(ab)
        
        # Línea de media
        ax.axhline(y=5.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Media')
        
        # Configuración general
        ax.set_ylabel('Puntuación (1-10)', fontsize=11, fontweight='bold')
        ax.set_ylim(-1.5, 11)  # Espacio extra para escudos
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        # Estilo
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('none')
    
    def crear_panel_top_metricas(self, ax, equipo, tipo, color):
        """Crea panel con las top 3 métricas en estilo tabla."""
        import textwrap
        
        top_metricas = self.analyzer.calcular_mejores_metricas(equipo, tipo, top_n=3)
        
        if not top_metricas:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                fontsize=10, transform=ax.transAxes)
            ax.axis('off')
            return
        
        # Detectar si es Villarreal (MOVER AQUÍ)
        es_villarreal = 'villarreal' in equipo.lower()
        
        # Título con estilo especial para Villarreal
        titulo = f'Top 3 Ataque' if tipo == 'ofensivo' else f'Top 3 Defensa'
        if es_villarreal:
            # Crear un bbox con fondo azul oscuro para el título amarillo
            title_text = ax.text(0.5, 1.075, f'{equipo}\n{titulo}',
                                ha='center', va='center',
                                fontsize=8, fontweight='bold',
                                color='#FFD700',  # Amarillo Villarreal
                                transform=ax.transAxes,
                                bbox=dict(boxstyle='round,pad=0.5',
                                        facecolor='#003D6B',  # Azul oscuro
                                        edgecolor='#FFD700',  # Borde amarillo
                                        linewidth=2),
                                zorder=10)
        else:
            ax.set_title(f'{equipo}\n{titulo}', fontsize=8, fontweight='bold', pad=3,
                        color=color)

        
        top_metricas = self.analyzer.calcular_mejores_metricas(equipo, tipo, top_n=3)
        
        if not top_metricas:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                fontsize=10, transform=ax.transAxes)
            ax.axis('off')
            return
        
        # Detectar si es Villarreal
        es_villarreal = 'villarreal' in equipo.lower()
        
        # Posiciones Y para cada fila (más espaciadas)
        y_positions = [0.90, 0.54, 0.14]

        # LEYENDA - Encabezados de las columnas
        leyenda_y = 1 # Posición arriba de todo
        leyenda_fontsize = 5
        leyenda_color = '#666666'

        # Fondo sutil para la leyenda
        ax.add_patch(patches.Rectangle(
            (0.02, leyenda_y-0.03), 0.96, 0.04,
            facecolor='#f0f0f0',
            edgecolor='#cccccc',
            linewidth=1,
            alpha=0.6,
            zorder=1))

        # Textos de la leyenda alineados con cada columna
        ax.text(0.095, leyenda_y, 'POS', 
                ha='center', va='center',
                fontsize=leyenda_fontsize, fontweight='bold',
                color=leyenda_color, zorder=2)

        ax.text(0.48, leyenda_y, 'MÉTRICA',
                ha='center', va='center',
                fontsize=leyenda_fontsize, fontweight='bold',
                color=leyenda_color, zorder=2)

        ax.text(0.95, leyenda_y, 'PROMEDIO',
                ha='center', va='center',
                fontsize=leyenda_fontsize, fontweight='bold',
                color=leyenda_color, zorder=2)
        
        for i, metrica_data in enumerate(top_metricas):
            y = y_positions[i]

            # Card moderna con colores según tipo (ataque/defensa)
            if tipo == 'ofensivo':
                card_borde = '#4CAF50'  # Verde para ataque
                sombra_color = '#81C784'
            else:
                card_borde = '#F44336'  # Rojo para defensa
                sombra_color = '#E57373'

            # Sombra de la card (efecto 3D)
            ax.add_patch(patches.FancyBboxPatch(
                (0.025, y-0.115), 0.96, 0.18,
                boxstyle="round,pad=0.01",
                facecolor=sombra_color,
                edgecolor='none',
                linewidth=0,
                alpha=0.2,
                zorder=1))

            # Card principal con borde de color
            ax.add_patch(patches.FancyBboxPatch(
                (0.02, y-0.11), 0.96, 0.18,
                boxstyle="round,pad=0.01",
                facecolor='white',
                edgecolor=card_borde,
                linewidth=2.5,
                alpha=1.0,
                zorder=2))
            
            # Logo de fondo RECORTADO dentro de la card (método robusto)
            logo_equipo = self.analyzer.load_team_logo(equipo, target_size=(400, 400))
            if logo_equipo is not None:
                # Posición y tamaño de la card
                card_x, card_y = 0.02, y-0.11
                card_width, card_height = 0.96, 0.18
                
                # Mostrar el logo con extent (posicionamiento exacto)
                im = ax.imshow(logo_equipo, 
                            extent=[card_x, card_x + card_width, card_y, card_y + card_height],
                            aspect='auto',
                            alpha=0.10,  # Transparencia
                            zorder=2.5,
                            interpolation='bilinear')
                
                # Crear el clip_path (rectángulo redondeado exacto)
                clip_patch = patches.FancyBboxPatch(
                    (card_x, card_y), card_width, card_height,
                    boxstyle="round,pad=0.01",
                    transform=ax.transData,
                    facecolor='none',
                    edgecolor='none')
                
                # Aplicar el clip al logo
                im.set_clip_path(clip_patch)
            
            posicion_liga = metrica_data['posicion_liga']
            
            # Determinar color según posición en la liga
            if posicion_liga <= 3:
                color_podium = '#FFD700'
                borde_color = '#FFA500'
                color_texto = 'black'
            else:
                color_podium = '#2C2C2C'
                borde_color = '#1A1A1A'
                color_texto = 'white'
            
            # COLUMNA 1: Cuadrado de ranking del equipo
            square = patches.FancyBboxPatch(
                (-0.02, y-0.04), 0.09, 0.08,
                boxstyle="round,pad=0.01",
                facecolor=color_podium,
                edgecolor=borde_color,
                linewidth=2,
                alpha=0.95,
                zorder=2)
            ax.add_patch(square)
            
            ax.text(0.045, y, str(i+1),
                    ha='center', va='center',
                    fontsize=10, fontweight='bold', 
                    color=color_texto, zorder=3,
                    family='monospace')
            
            # COLUMNA 2: Nombre de métrica (con tamaño dinámico)
            nombre_metrica = metrica_data['metrica']

            # Determinar si necesita wrap y ajustar tamaño de fuente
            if len(nombre_metrica) > 20:
                lineas = textwrap.wrap(nombre_metrica, width=20)
                if len(lineas) > 2:
                    lineas = lineas[:2]
                    lineas[1] = lineas[1][:17] + '...'
                texto_metrica = '\n'.join(lineas)
                # Texto largo → fuente más pequeña
                fontsize_metrica = 8
                linespacing_metrica = 0.9
            else:
                texto_metrica = nombre_metrica
                # Texto corto → fuente normal
                fontsize_metrica = 10.5
                linespacing_metrica = 1.0

            ax.text(0.48, y - 0.02, texto_metrica,
                    ha='center', va='center',
                    fontsize=fontsize_metrica,  # Tamaño dinámico
                    fontweight='bold', 
                    color='#333333',
                    linespacing=linespacing_metrica,  # Line spacing dinámico
                    zorder=2)

            # Badge de ranking debajo del nombre de métrica
            if posicion_liga <= 3:
                badge = patches.FancyBboxPatch(
                    (0.43, y-0.09), 0.1, 0.025,
                    boxstyle="round,pad=0.002",
                    facecolor='#FFD700',
                    edgecolor='#FFA500',
                    linewidth=1, zorder=3)
                ax.add_patch(badge)
                ax.text(0.48, y-0.0775, f'#{posicion_liga} Liga',
                        ha='center', va='center',
                        fontsize=5.5, fontweight='bold',
                        color='black', zorder=4)
            else:
                badge = patches.Rectangle(
                    (0.43, y-0.09), 0.1, 0.025,
                    facecolor='#2C2C2C',
                    edgecolor='#1A1A1A',
                    linewidth=1, zorder=3)
                ax.add_patch(badge)
                ax.text(0.48, y-0.0775, f'#{posicion_liga} Liga',
                        ha='center', va='center',
                        fontsize=5.5, fontweight='bold',
                        color='white', zorder=4)
            
            # COLUMNA 3: Valor por partido (más a la derecha)
            valor_promedio = metrica_data.get('valor_promedio', metrica_data.get('valor', 0))
            nombre_metrica_original = metrica_data.get('metrica_original', nombre_metrica)
            
            if '%' in nombre_metrica_original or 'Eficacia' in nombre_metrica_original or 'Eficiencia' in nombre_metrica_original:
                valor_text = f"{valor_promedio:.1f}%"
            elif valor_promedio >= 10:
                valor_text = f"{valor_promedio:.0f}"
            else:
                valor_text = f"{valor_promedio:.1f}"
            
            # Estilo especial para Villarreal
            if es_villarreal:
                # Sombra del texto (relieve negro)
                ax.text(1.00, y-0.002, valor_text,  # CAMBIADO: x de 0.92 a 0.95
                        ha='center', va='center',
                        fontsize=10, fontweight='bold', 
                        color='black', alpha=0.7, zorder=2)
                
                # Texto principal con fondo amarillo
                ax.text(1.00, y, valor_text,  # CAMBIADO: x de 0.92 a 0.95
                        ha='center', va='center',
                        fontsize=10, fontweight='bold', 
                        color='black',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='#FFD700',
                                edgecolor='#FFA500', 
                                linewidth=1.8),
                        zorder=3)
            else:
                ax.text(1.00, y, valor_text,  # CAMBIADO: x de 0.92 a 0.95
                        ha='center', va='center',
                        fontsize=10, fontweight='bold', 
                        color=color,
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white',
                                edgecolor=color, 
                                linewidth=1.8),
                        zorder=2)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def guardar_reporte(self, fig, equipo1, equipo2):
        """Guarda el reporte en PDF."""
        if fig is None:
            return
        equipo1_fn = equipo1.replace(' ', '_')
        equipo2_fn = equipo2.replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evolucion_rendimiento_{equipo1_fn}_vs_{equipo2_fn}_{timestamp}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.2, format='pdf')


def main():
    """Función principal."""
    
    analyzer = EvolucionRendimientoMediaCoach()
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
    
    
    visualizador = VisualizadorEvolucion(analyzer)
    fig = visualizador.crear_informe_evolucion(equipo_oponente, villarreal_nombre)
    
    if fig:
        pass
        plt.show()
        visualizador.guardar_reporte(fig, equipo_oponente, villarreal_nombre)
    else:
        pass


if __name__ == "__main__":
    main()