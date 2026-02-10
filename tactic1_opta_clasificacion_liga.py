import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
from scipy import ndimage
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ClasificacionFromParquet:
    def __init__(self, parquet_path='extraccion_opta/datos_opta_parquet/team_stats.parquet'):
        self.parquet_path = parquet_path
        self.df_stats = None
        self.clasificaciones = None
        self.equipos = []
        
    def cargar_datos(self):
        """Carga el archivo parquet"""
        try:
            self.df_stats = pd.read_parquet(self.parquet_path)
            return self.df_stats
        except Exception as e:
            pass
            return None
    
    def calcular_resultados_partidos(self):
        """Calcula ganador/empate de cada partido"""
        # El parquet ya tiene Goals por equipo y partido
        resultados = []
        
        # Agrupar por Match ID para obtener ambos equipos del partido
        for match_id, grupo in self.df_stats.groupby('Match ID'):
            week = grupo['Week'].iloc[0]
            
            # Obtener equipos del partido
            equipos_partido = grupo['Team Name'].unique()
            if len(equipos_partido) != 2:
                continue
            
            equipo1, equipo2 = equipos_partido[0], equipos_partido[1]
            
            # Obtener goles
            gf_eq1 = grupo[grupo['Team Name'] == equipo1]['goals'].values
            gf_eq2 = grupo[grupo['Team Name'] == equipo2]['goals'].values
            
            gf_eq1 = int(gf_eq1[0]) if len(gf_eq1) > 0 and pd.notna(gf_eq1[0]) else 0
            gf_eq2 = int(gf_eq2[0]) if len(gf_eq2) > 0 and pd.notna(gf_eq2[0]) else 0
            
            # GC = GF del rival
            gc_eq1 = gf_eq2
            gc_eq2 = gf_eq1
            
            # Calcular puntos
            if gf_eq1 > gc_eq1:
                puntos_eq1, puntos_eq2 = 3, 0
            elif gf_eq1 == gc_eq1:
                puntos_eq1, puntos_eq2 = 1, 1
            else:
                puntos_eq1, puntos_eq2 = 0, 3
            
            # Obtener Team Position para cada equipo
            pos_eq1 = grupo[grupo['Team Name'] == equipo1]['Team Position'].values
            pos_eq2 = grupo[grupo['Team Name'] == equipo2]['Team Position'].values

            pos_eq1 = pos_eq1[0] if len(pos_eq1) > 0 else 'unknown'
            pos_eq2 = pos_eq2[0] if len(pos_eq2) > 0 else 'unknown'
            
            # Añadir resultados
            resultados.append({
                'ID PARTIDO': match_id,
                'jornada': f'j{int(week)}',
                'EQUIPO': equipo1,
                'GF': gf_eq1,
                'GC': gc_eq1,
                'Puntos': puntos_eq1,
                'Team Position': pos_eq1  # AÑADIR ESTA LÍNEA
            })

            resultados.append({
                'ID PARTIDO': match_id,
                'jornada': f'j{int(week)}',
                'EQUIPO': equipo2,
                'GF': gf_eq2,
                'GC': gc_eq2,
                'Puntos': puntos_eq2,
                'Team Position': pos_eq2  # AÑADIR ESTA LÍNEA
            })


        
        return pd.DataFrame(resultados)
    
    def construir_clasificacion_local_visitante(self):
        """Construye clasificaciones separadas de local y visitante"""
        resultados_df = self.calcular_resultados_partidos()
        
        if resultados_df.empty:
            return None, None
        
        # Extraer número de jornada
        resultados_df['num_jornada'] = pd.to_numeric(resultados_df['jornada'].str.extract(r'(\d+)')[0], errors='coerce')
        resultados_df = resultados_df.dropna(subset=['num_jornada'])
        resultados_df['num_jornada'] = resultados_df['num_jornada'].astype(int)
        
        ultima_jornada = resultados_df['num_jornada'].max()
        
        # Filtrar por posición
        datos_local = resultados_df[
            (resultados_df['Team Position'] == 'home') & 
            (resultados_df['num_jornada'] <= ultima_jornada)
        ]
        
        datos_visitante = resultados_df[
            (resultados_df['Team Position'] == 'away') & 
            (resultados_df['num_jornada'] <= ultima_jornada)
        ]
        
        # Clasificación LOCAL
        clasificacion_local = datos_local.groupby('EQUIPO').agg({
            'Puntos': 'sum',
            'GF': 'sum',
            'GC': 'sum'
        }).reset_index()
        clasificacion_local['DG'] = clasificacion_local['GF'] - clasificacion_local['GC']
        clasificacion_local = clasificacion_local.sort_values(
            ['Puntos', 'GF', 'DG'], 
            ascending=[False, False, False]
        ).reset_index(drop=True)
        clasificacion_local['Posicion'] = range(1, len(clasificacion_local) + 1)
        clasificacion_local = clasificacion_local.rename(columns={'EQUIPO': 'Equipo'})
        
        # Clasificación VISITANTE
        clasificacion_visitante = datos_visitante.groupby('EQUIPO').agg({
            'Puntos': 'sum',
            'GF': 'sum',
            'GC': 'sum'
        }).reset_index()
        clasificacion_visitante['DG'] = clasificacion_visitante['GF'] - clasificacion_visitante['GC']
        clasificacion_visitante = clasificacion_visitante.sort_values(
            ['Puntos', 'GF', 'DG'], 
            ascending=[False, False, False]
        ).reset_index(drop=True)
        clasificacion_visitante['Posicion'] = range(1, len(clasificacion_visitante) + 1)
        clasificacion_visitante = clasificacion_visitante.rename(columns={'EQUIPO': 'Equipo'})
        
        return clasificacion_local, clasificacion_visitante
    
    def construir_clasificacion_por_jornada(self):
        """Construye clasificación acumulada jornada a jornada"""
        resultados_df = self.calcular_resultados_partidos()
        
        if resultados_df.empty:
            pass
            return None
        
        # Extraer número de jornada
        resultados_df['num_jornada'] = pd.to_numeric(resultados_df['jornada'].str.extract(r'(\d+)')[0], errors='coerce')
        resultados_df = resultados_df.dropna(subset=['num_jornada'])  # Eliminar filas sin número de jornada
        resultados_df['num_jornada'] = resultados_df['num_jornada'].astype(int)        
        resultados_df = resultados_df.sort_values(['num_jornada', 'EQUIPO'])
        
        clasificaciones = []
        
        # Calcular clasificación acumulada para cada jornada
        for jornada in sorted(resultados_df['num_jornada'].unique()):
            # Datos hasta esta jornada
            datos_hasta_jornada = resultados_df[resultados_df['num_jornada'] <= jornada]
            
            # Acumular por equipo
            stats_equipos = datos_hasta_jornada.groupby('EQUIPO').agg({
                'Puntos': 'sum',
                'GF': 'sum',
                'GC': 'sum'
            }).reset_index()
            
            stats_equipos['DG'] = stats_equipos['GF'] - stats_equipos['GC']
            
            # Ordenar: 1) Puntos DESC, 2) GF DESC, 3) DG DESC
            stats_equipos = stats_equipos.sort_values(
                ['Puntos', 'GF', 'DG'], 
                ascending=[False, False, False]
            ).reset_index(drop=True)
            
            # Asignar posiciones
            stats_equipos['Posicion'] = range(1, len(stats_equipos) + 1)
            stats_equipos['Jornada'] = jornada
            
            clasificaciones.append(stats_equipos[['Jornada', 'Posicion', 'EQUIPO', 'Puntos', 'GF', 'GC', 'DG']])
        
        self.clasificaciones = pd.concat(clasificaciones, ignore_index=True)
        self.clasificaciones = self.clasificaciones.rename(columns={'EQUIPO': 'Equipo'})
        self.equipos = sorted(self.clasificaciones['Equipo'].unique())
        
        return self.clasificaciones
    
    def guardar_datos(self, filename='clasificaciones_laliga.csv'):
        """Guarda los datos en CSV"""
        if self.clasificaciones is not None and not self.clasificaciones.empty:
            self.clasificaciones.to_csv(filename, index=False, encoding='utf-8')


class VisualizadorEvolucionClasificacion:
    def __init__(self, clasificaciones_df, parquet_path=None):
        self.df = clasificaciones_df
        self.equipos = sorted(clasificaciones_df['Equipo'].unique().tolist())
        self.parquet_path = parquet_path
        
    def load_team_logo(self, equipo, target_size=(80, 80)):
        """Carga el logo del equipo"""
        try:
            from PIL import Image
            from difflib import SequenceMatcher
            import unicodedata
        except ImportError:
            return None
        
        def normalizar_nombre(texto):
            """Quita acentos y normaliza texto"""
            # Quitar acentos
            texto = ''.join(
                c for c in unicodedata.normalize('NFD', texto)
                if unicodedata.category(c) != 'Mn'
            )
            return texto.lower()
        
        # Generar variaciones del nombre
        equipo_normalizado = normalizar_nombre(equipo)
        
        possible_names = [
            equipo,  # Nombre original
            equipo_normalizado,  # Sin acentos
            equipo.replace(' ', '_'),
            equipo.replace(' ', ''),
            equipo_normalizado.replace(' ', '_'),
            equipo_normalizado.replace(' ', ''),
            equipo.split()[0] if ' ' in equipo else equipo,  # Primera palabra
            normalizar_nombre(equipo.split()[0]) if ' ' in equipo else equipo_normalizado,
        ]
        
        # Casos especiales
        if 'atletico' in equipo_normalizado:
            possible_names.extend(['Atletico', 'atletico', 'Atletico_de_Madrid'])
        
        logo_path = None
        for name in possible_names:
            path = f"assets/escudos/{name}.png"
            if os.path.exists(path):
                logo_path = path
                break
        
        # Busqueda por similitud si no encuentra exacto
        if not logo_path and os.path.exists('assets/escudos'):
            all_files = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
            best_match, best_score = None, 0
            for filename in all_files:
                name_without_ext = os.path.splitext(filename)[0]
                name_normalized = normalizar_nombre(name_without_ext)
                
                # Comparar con nombre normalizado
                score = SequenceMatcher(None, equipo_normalizado, name_normalized).ratio()
                if score > best_score:
                    best_score, best_match = score, filename
            
            if best_match and best_score > 0.6:
                logo_path = f"assets/escudos/{best_match}"
        
        if logo_path:
            try:
                with Image.open(logo_path) as img:
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    final_img = Image.new('RGBA', target_size, (0, 0, 0, 0))
                    paste_x = (target_size[0] - img.width) // 2
                    paste_y = (target_size[1] - img.height) // 2
                    final_img.paste(img, (paste_x, paste_y), img)
                    return np.array(final_img) / 255.0
            except:
                pass
        return None
    
    def crear_reporte_evolucion(self, equipo_comparar, villarreal_nombre='Villarreal CF', figsize=(11.69, 8.27)):
        """Crea el reporte de evolucion comparando Villarreal con otro equipo"""
        
        # Datos de evolucion
        villarreal_data = self.df[self.df['Equipo'] == villarreal_nombre].sort_values('Jornada')
        equipo_data = self.df[self.df['Equipo'] == equipo_comparar].sort_values('Jornada')
        
        if villarreal_data.empty:
            pass
            return None
        
        if equipo_data.empty:
            pass
            return None
        
        # Crear figura
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        # Fondo
        if os.path.exists('assets/fondo_informes.png'):
            background = plt.imread('assets/fondo_informes.png')
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25)
            ax_bg.axis('off')
        
        # Titulo
        fig.suptitle('EVOLUCION CLASIFICACION LALIGA EA SPORTS', 
                    fontsize=20, fontweight='bold', color='#1e3d59', y=0.96)
        
        # Logos
        logo_villarreal = self.load_team_logo(villarreal_nombre)
        logo_equipo = self.load_team_logo(equipo_comparar)
        
        if logo_villarreal is not None:
            ax_vill = fig.add_axes([0.05, 0.88, 0.08, 0.08])
            ax_vill.imshow(logo_villarreal, aspect='auto')
            ax_vill.axis('off')
        
        if logo_equipo is not None:
            ax_eq = fig.add_axes([0.87, 0.88, 0.08, 0.08])
            ax_eq.imshow(logo_equipo, aspect='auto')
            ax_eq.axis('off')
        
        # Grid: 3 filas, 4 columnas
        gs = fig.add_gridspec(nrows=3, ncols=4, left=0.05, right=0.98, bottom=0.08, top=0.85,
                            wspace=0.20, hspace=0.35, height_ratios=[1.5, 1.5, 0.8])
        
        ultima_jornada = self.df['Jornada'].max()

        # Generar clasificaciones local y visitante
        if self.parquet_path:
            clasificador_temp = ClasificacionFromParquet(self.parquet_path)
            df_temp = clasificador_temp.cargar_datos()
            if df_temp is not None:
                clasificacion_local, clasificacion_visitante = clasificador_temp.construir_clasificacion_local_visitante()
            else:
                clasificacion_local = None
                clasificacion_visitante = None
        
        # ========== FILA 0-1, COLUMNAS 0-1: TABLA CLASIFICACION GENERAL ==========
        ax_tabla = fig.add_subplot(gs[0:2, 0:2])
        ax_tabla.axis('off')
        
        clasificacion_completa = self.df[self.df['Jornada'] == ultima_jornada].sort_values('Posicion')
        
        y_start = 0.95
        y_step = 0.047
        
        ax_tabla.text(0.5, 0.98, f'CLASIFICACION GENERAL J{int(ultima_jornada)}', 
                    ha='center', fontsize=10, fontweight='bold', transform=ax_tabla.transAxes)
        
        # Encabezado
        headers = ['P', 'Equipo', 'Pts', 'PJ', 'GF', 'GC', 'DG']
        x_positions = [0.05, 0.28, 0.52, 0.62, 0.72, 0.82, 0.92]
        for header, x in zip(headers, x_positions):
            ax_tabla.text(x, y_start, header, fontsize=7, fontweight='bold', 
                        transform=ax_tabla.transAxes, ha='center')
        
        # Filas de equipos
        y_pos = y_start - y_step
        for _, row in clasificacion_completa.iterrows():
            equipo_nombre = row['Equipo']
            pj = df_temp[(df_temp['Team Name'] == equipo_nombre) & (df_temp['Week'].astype(int) <= ultima_jornada)]['Match ID'].nunique()
            
            # Fondo segun equipo
            if villarreal_nombre in equipo_nombre:
                rect = patches.Rectangle((0.02, y_pos - 0.015), 0.96, 0.05,
                                        transform=ax_tabla.transAxes, facecolor='#FFF8DC',
                                        edgecolor='black', linewidth=0.5, zorder=1)
                ax_tabla.add_patch(rect)
            elif equipo_comparar in equipo_nombre:
                rect = patches.Rectangle((0.02, y_pos - 0.015), 0.96, 0.05,
                                        transform=ax_tabla.transAxes, facecolor='#E8F4F8',
                                        edgecolor='black', linewidth=0.5, zorder=1)
                ax_tabla.add_patch(rect)
            
            # Datos
            datos = [
                f"{int(row['Posicion'])}",
                int(row['Puntos']),
                pj,
                int(row['GF']),
                int(row['GC']),
                int(row['DG'])
            ]

            # Dibujar datos excepto el nombre
            for i, (dato, x) in enumerate(zip(datos, [x_positions[0]] + x_positions[2:])):
                ax_tabla.text(x, y_pos, str(dato), fontsize=7, 
                            transform=ax_tabla.transAxes, ha='center', zorder=2)

            # Dibujar nombre del equipo con tamaño ajustado
            nombre_display = equipo_nombre
            fontsize_nombre = 7
            if len(equipo_nombre) > 15:
                fontsize_nombre = 5
            elif len(equipo_nombre) > 12:
                fontsize_nombre = 6

            ax_tabla.text(x_positions[1], y_pos, nombre_display, fontsize=fontsize_nombre, 
                        transform=ax_tabla.transAxes, ha='center', zorder=2)
            
            # Escudo pequeño
            logo = self.load_team_logo(equipo_nombre, target_size=(30, 30))
            if logo is not None:
                imagebox = OffsetImage(logo, zoom=0.5)
                ab = AnnotationBbox(imagebox, (0.12, y_pos + 0.008), 
                                frameon=False, transform=ax_tabla.transAxes, 
                                box_alignment=(0.5, 0.5), zorder=3)
                ax_tabla.add_artist(ab)
            
            y_pos -= y_step
            if y_pos < 0.05:
                break
        
        # ========== FILA 0, COLUMNAS 2-3: GRAFICA EVOLUTIVA ==========
        ax_evolucion = fig.add_subplot(gs[0, 2:])
        ax_evolucion.patch.set_alpha(0)

        jornadas_vill = villarreal_data['Jornada'].values
        posiciones_vill = villarreal_data['Posicion'].values
        jornadas_eq = equipo_data['Jornada'].values
        posiciones_eq = equipo_data['Posicion'].values

        # Líneas sin marcadores
        ax_evolucion.plot(jornadas_vill, posiciones_vill, linewidth=2.5, 
                color='#FFD700', label=villarreal_nombre, alpha=0.7, zorder=2)
        ax_evolucion.plot(jornadas_eq, posiciones_eq, linewidth=2.5,
                color='#3498DB', label=equipo_comparar, alpha=0.7, zorder=2)

        # Escudos en cada punto para Villarreal
        logo_vill_mini = self.load_team_logo(villarreal_nombre, target_size=(25, 25))
        if logo_vill_mini is not None:
            for jornada, pos in zip(jornadas_vill, posiciones_vill):
                imagebox = OffsetImage(logo_vill_mini, zoom=0.4)
                ab = AnnotationBbox(imagebox, (jornada, pos), frameon=False, 
                                box_alignment=(0.5, 0.5), zorder=3)
                ax_evolucion.add_artist(ab)

        # Escudos en cada punto para equipo comparado
        logo_eq_mini = self.load_team_logo(equipo_comparar, target_size=(25, 25))
        if logo_eq_mini is not None:
            for jornada, pos in zip(jornadas_eq, posiciones_eq):
                imagebox = OffsetImage(logo_eq_mini, zoom=0.4)
                ab = AnnotationBbox(imagebox, (jornada, pos), frameon=False,
                                box_alignment=(0.5, 0.5), zorder=3)
                ax_evolucion.add_artist(ab)

        ax_evolucion.invert_yaxis()
        ax_evolucion.set_ylim(20.5, 0.5)
        ax_evolucion.axhspan(0.5, 4.5, alpha=0.15, color='green', label='Champions')
        ax_evolucion.axhspan(17.5, 20.5, alpha=0.15, color='red', label='Descenso')

        ax_evolucion.set_xlabel('Jornada', fontsize=10, fontweight='bold')
        ax_evolucion.set_ylabel('Posicion', fontsize=10, fontweight='bold')
        ax_evolucion.set_title('Evolucion de la Clasificacion', fontsize=12, fontweight='bold', pad=10)
        ax_evolucion.grid(True, alpha=0.3, linestyle='--')
        ax_evolucion.legend(bbox_to_anchor=(0.01, 1.15), loc='center left', 
                fontsize=5, framealpha=0.9, borderaxespad=0)

        max_jornada = max(jornadas_vill.max(), jornadas_eq.max())
        ax_evolucion.set_xticks(range(1, int(max_jornada) + 1, 2))
        ax_evolucion.set_yticks(range(1, 21))
        
        # ========== FILA 1, COL 2: CLASIFICACION LOCAL ==========
        ax_local = fig.add_subplot(gs[1, 2])
        ax_local.axis('off')

        ax_local.text(0.5, 0.98, 'CLASIFICACION LOCAL', ha='center', fontsize=8, 
                    fontweight='bold', transform=ax_local.transAxes, color='#2c3e50')

        if clasificacion_local is not None and not clasificacion_local.empty:
            # Encabezado pequeño
            headers_mini = ['P', 'Equipo', 'Pts', 'GF', 'GC']
            x_pos_mini = [0.08, 0.35, 0.60, 0.75, 0.90]
            
            y_start_mini = 0.92  # Empezar un poco más arriba
            y_step_mini = 0.042  # Espaciado vertical reducido para 20 equipos
            
            # Dibujar encabezado
            for h, x in zip(headers_mini, x_pos_mini):
                ax_local.text(x, y_start_mini, h, fontsize=5.5, fontweight='bold',
                            transform=ax_local.transAxes, ha='center')
            
            y_mini = y_start_mini - y_step_mini * 1.2 # Espacio después del header

            # --- LÓGICA MODIFICADA PARA MOSTRAR LOS 20 EQUIPOS ---
            # Se itera sobre toda la clasificación, sin filtrar
            for idx, row in clasificacion_local.iterrows():
                equipo_nombre = row['Equipo']
                
                # Resaltar equipos comparados
                if villarreal_nombre in equipo_nombre or equipo_comparar in equipo_nombre:
                    # Altura del rectángulo reducida
                    rect = patches.Rectangle((0.02, y_mini - 0.011), 0.96, 0.036,
                                            transform=ax_local.transAxes, 
                                            facecolor='#ffffcc' if villarreal_nombre in equipo_nombre else '#e6f3ff',
                                            edgecolor='gray', linewidth=0.5, zorder=1)
                    ax_local.add_patch(rect)
                
                datos_mini = [
                    int(row['Posicion']),
                    int(row['Puntos']),
                    int(row['GF']),
                    int(row['GC'])
                ]
                
                # Tamaño de fuente reducido
                for dato, x in zip(datos_mini, [x_pos_mini[0]] + x_pos_mini[2:]):
                    ax_local.text(x, y_mini, str(dato), fontsize=4,
                                transform=ax_local.transAxes, ha='center', zorder=2)
                
                # Nombre acortado y fuente reducida
                nombre_corto = equipo_nombre[:12] + '...' if len(equipo_nombre) > 15 else equipo_nombre
                ax_local.text(x_pos_mini[1], y_mini, nombre_corto, fontsize=4,
                            transform=ax_local.transAxes, ha='center', zorder=2)
                
                # Moverse a la siguiente fila
                y_mini -= y_step_mini
                if y_mini < 0.05:
                    break
        
        # ========== FILA 1, COL 3: CLASIFICACION VISITANTE ==========
        ax_visitante = fig.add_subplot(gs[1, 3])
        ax_visitante.axis('off')

        ax_visitante.text(0.5, 0.98, 'CLASIFICACION VISITANTE', ha='center', fontsize=8,
                        fontweight='bold', transform=ax_visitante.transAxes, color='#2c3e50')

        if clasificacion_visitante is not None and not clasificacion_visitante.empty:
            # Encabezado pequeño
            headers_mini = ['P', 'Equipo', 'Pts', 'GF', 'GC']
            x_pos_mini = [0.08, 0.35, 0.60, 0.75, 0.90]
            
            y_start_mini = 0.92  # Empezar un poco más arriba
            y_step_mini = 0.042  # Espaciado vertical reducido para 20 equipos
            
            # Dibujar encabezado
            for h, x in zip(headers_mini, x_pos_mini):
                ax_visitante.text(x, y_start_mini, h, fontsize=5.5, fontweight='bold',
                                transform=ax_visitante.transAxes, ha='center')
            
            y_mini = y_start_mini - y_step_mini * 1.2 # Espacio después del header

            # --- LÓGICA MODIFICADA PARA MOSTRAR LOS 20 EQUIPOS ---
            # Se itera sobre toda la clasificación, sin filtrar
            for idx, row in clasificacion_visitante.iterrows():
                equipo_nombre = row['Equipo']
                
                if villarreal_nombre in equipo_nombre or equipo_comparar in equipo_nombre:
                    # Altura del rectángulo reducida
                    rect = patches.Rectangle((0.02, y_mini - 0.011), 0.96, 0.036,
                                            transform=ax_visitante.transAxes,
                                            facecolor='#ffffcc' if villarreal_nombre in equipo_nombre else '#e6f3ff',
                                            edgecolor='gray', linewidth=0.5, zorder=1)
                    ax_visitante.add_patch(rect)
                
                datos_mini = [
                    int(row['Posicion']),
                    int(row['Puntos']),
                    int(row['GF']),
                    int(row['GC'])
                ]
                
                # Tamaño de fuente reducido
                for dato, x in zip(datos_mini, [x_pos_mini[0]] + x_pos_mini[2:]):
                    ax_visitante.text(x, y_mini, str(dato), fontsize=4,
                                    transform=ax_visitante.transAxes, ha='center', zorder=2)
                
                # Nombre acortado y fuente reducida
                nombre_corto = equipo_nombre[:12] + '...' if len(equipo_nombre) > 15 else equipo_nombre
                ax_visitante.text(x_pos_mini[1], y_mini, nombre_corto, fontsize=4,
                                transform=ax_visitante.transAxes, ha='center', zorder=2)
                
                # Moverse a la siguiente fila
                y_mini -= y_step_mini
                if y_mini < 0.05:
                    break
        
        # ========== FILA 2, COL 0: STATS VILLARREAL ==========
        ax_stats_vill = fig.add_subplot(gs[2, 0])
        ax_stats_vill.axis('off')

        mejor_jornada_vill = villarreal_data.loc[villarreal_data['Posicion'].idxmin()]
        pos_actual_vill = int(villarreal_data[villarreal_data['Jornada'] == ultima_jornada]['Posicion'].values[0])
        pts_actual_vill = int(villarreal_data[villarreal_data['Jornada'] == ultima_jornada]['Puntos'].values[0])

        # Tarjeta Villarreal
        rect_vill = patches.FancyBboxPatch((0.05, 0.1), 0.90, 0.80,
                                        boxstyle="round,pad=0.02", 
                                        facecolor='#FFF8DC', edgecolor='#FFD700',
                                        linewidth=2, transform=ax_stats_vill.transAxes)
        ax_stats_vill.add_patch(rect_vill)

        # Escudo de fondo Villarreal
        logo_vill_bg = self.load_team_logo(villarreal_nombre, target_size=(200, 200))
        if logo_vill_bg is not None:
            logo_rotado = ndimage.rotate(logo_vill_bg, 35, reshape=False, order=1)
            logo_rotado[:,:,3] = logo_rotado[:,:,3] * 0.15
            
            imagebox = OffsetImage(logo_rotado, zoom=1.2)
            ab = AnnotationBbox(imagebox, (0.50, 0.50), frameon=False,
                            transform=ax_stats_vill.transAxes, box_alignment=(0.5, 0.5), zorder=1)
            ax_stats_vill.add_artist(ab)

        ax_stats_vill.text(0.50, 0.75, villarreal_nombre, ha='center', va='center',
                    fontsize=10, fontweight='bold', transform=ax_stats_vill.transAxes, zorder=3)
        ax_stats_vill.text(0.50, 0.55, f"Posicion: {pos_actual_vill}°", ha='center',
                    fontsize=9, transform=ax_stats_vill.transAxes, zorder=3)
        ax_stats_vill.text(0.50, 0.40, f"Puntos: {pts_actual_vill}", ha='center',
                    fontsize=9, transform=ax_stats_vill.transAxes, zorder=3)
        ax_stats_vill.text(0.50, 0.25, f"Mejor: {int(mejor_jornada_vill['Posicion'])}° (J{int(mejor_jornada_vill['Jornada'])})", 
                    ha='center', fontsize=8, transform=ax_stats_vill.transAxes, style='italic', zorder=3)
        
        # ========== FILA 2, COL 1: STATS EQUIPO COMPARADO ==========
        ax_stats_eq = fig.add_subplot(gs[2, 1])
        ax_stats_eq.axis('off')

        mejor_jornada_eq = equipo_data.loc[equipo_data['Posicion'].idxmin()]
        pos_actual_eq = int(equipo_data[equipo_data['Jornada'] == ultima_jornada]['Posicion'].values[0])
        pts_actual_eq = int(equipo_data[equipo_data['Jornada'] == ultima_jornada]['Puntos'].values[0])

        # Tarjeta equipo comparado
        rect_eq = patches.FancyBboxPatch((0.05, 0.1), 0.90, 0.80,
                                        boxstyle="round,pad=0.02", 
                                        facecolor='#E8F4F8', edgecolor='#3498DB',
                                        linewidth=2, transform=ax_stats_eq.transAxes)
        ax_stats_eq.add_patch(rect_eq)

        # Escudo de fondo equipo comparado
        logo_eq_bg = self.load_team_logo(equipo_comparar, target_size=(200, 200))
        if logo_eq_bg is not None:
            logo_rotado_eq = ndimage.rotate(logo_eq_bg, 35, reshape=False, order=1)
            logo_rotado_eq[:,:,3] = logo_rotado_eq[:,:,3] * 0.15
            
            imagebox = OffsetImage(logo_rotado_eq, zoom=1.2)
            ab = AnnotationBbox(imagebox, (0.50, 0.50), frameon=False,
                            transform=ax_stats_eq.transAxes, box_alignment=(0.5, 0.5), zorder=1)
            ax_stats_eq.add_artist(ab)

        ax_stats_eq.text(0.50, 0.75, equipo_comparar, ha='center', va='center',
                    fontsize=10, fontweight='bold', transform=ax_stats_eq.transAxes, zorder=3)
        ax_stats_eq.text(0.50, 0.55, f"Posicion: {pos_actual_eq}°", ha='center',
                    fontsize=9, transform=ax_stats_eq.transAxes, zorder=3)
        ax_stats_eq.text(0.50, 0.40, f"Puntos: {pts_actual_eq}", ha='center',
                    fontsize=9, transform=ax_stats_eq.transAxes, zorder=3)
        ax_stats_eq.text(0.50, 0.25, f"Mejor: {int(mejor_jornada_eq['Posicion'])}° (J{int(mejor_jornada_eq['Jornada'])})", 
                    ha='center', fontsize=8, transform=ax_stats_eq.transAxes, style='italic', zorder=3)
        
        # ========== FILA 2, COLUMNAS 2-3: ANALISIS COMPARATIVO ==========
        ax_resumen = fig.add_subplot(gs[2, 2:])
        ax_resumen.axis('off')
        
        # Fondo con efecto de profundidad
        for i in range(5):
            alpha_val = 0.1 - (i * 0.02)
            rect = patches.Rectangle((0.02 + i*0.002, 0.05 + i*0.02), 0.96 - i*0.004, 0.90 - i*0.04,
                                    transform=ax_resumen.transAxes, 
                                    facecolor='#f0f0f0', alpha=alpha_val, zorder=i)
            ax_resumen.add_patch(rect)
        
        # Contenedor principal
        main_rect = patches.FancyBboxPatch((0.02, 0.05), 0.96, 0.90,
                                        boxstyle="round,pad=0.015", 
                                        facecolor='white', edgecolor='#2c3e50',
                                        linewidth=2, alpha=0.95, transform=ax_resumen.transAxes)
        
        ax_resumen.add_patch(main_rect)
        
        # Metricas comparativas
        diff_puestos = abs(pos_actual_vill - pos_actual_eq)
        diff_puntos = abs(pts_actual_vill - pts_actual_eq)
        gf_vill = int(villarreal_data[villarreal_data['Jornada'] == ultima_jornada]['GF'].values[0])
        gf_eq = int(equipo_data[equipo_data['Jornada'] == ultima_jornada]['GF'].values[0])
        gc_vill = int(villarreal_data[villarreal_data['Jornada'] == ultima_jornada]['GC'].values[0])
        gc_eq = int(equipo_data[equipo_data['Jornada'] == ultima_jornada]['GC'].values[0])
        
        # Titulo
        ax_resumen.text(0.50, 0.80, 'ANALISIS COMPARATIVO', ha='center', va='center',
                    fontsize=11, fontweight='bold', transform=ax_resumen.transAxes,
                    color='#2c3e50')
        
        # Columna 1: Diferencias
        ax_resumen.text(0.20, 0.60, 'DIFERENCIAS', ha='center', fontsize=9, 
                    fontweight='bold', transform=ax_resumen.transAxes, color='#34495e')
        ax_resumen.text(0.20, 0.40, f'{diff_puestos} puestos', ha='center', fontsize=10,
                    transform=ax_resumen.transAxes, color='#e74c3c', fontweight='bold')
        ax_resumen.text(0.20, 0.25, f'{diff_puntos} puntos', ha='center', fontsize=10,
                    transform=ax_resumen.transAxes, color='#e67e22', fontweight='bold')
        
        # Columna 2: Goles a favor
        ax_resumen.text(0.50, 0.60, 'GOLES A FAVOR', ha='center', fontsize=9,
                    fontweight='bold', transform=ax_resumen.transAxes, color='#34495e')
        ax_resumen.text(0.40, 0.40, villarreal_nombre, ha='center', fontsize=7,
                    transform=ax_resumen.transAxes)
        ax_resumen.text(0.40, 0.25, f'{gf_vill}', ha='center', fontsize=11,
                    transform=ax_resumen.transAxes, color='#27ae60', fontweight='bold')
        ax_resumen.text(0.60, 0.40, equipo_comparar, ha='center', fontsize=7,
                    transform=ax_resumen.transAxes)
        ax_resumen.text(0.60, 0.25, f'{gf_eq}', ha='center', fontsize=11,
                    transform=ax_resumen.transAxes, color='#2980b9', fontweight='bold')

        # Columna 3: Goles en contra
        ax_resumen.text(0.80, 0.60, 'GOLES EN CONTRA', ha='center', fontsize=9,
                    fontweight='bold', transform=ax_resumen.transAxes, color='#34495e')
        ax_resumen.text(0.70, 0.40, villarreal_nombre, ha='center', fontsize=7,
                    transform=ax_resumen.transAxes)
        ax_resumen.text(0.70, 0.25, f'{gc_vill}', ha='center', fontsize=11,
                    transform=ax_resumen.transAxes, color='#c0392b', fontweight='bold')
        ax_resumen.text(0.90, 0.40, equipo_comparar, ha='center', fontsize=7,
                    transform=ax_resumen.transAxes)
        ax_resumen.text(0.90, 0.25, f'{gc_eq}', ha='center', fontsize=11,
                    transform=ax_resumen.transAxes, color='#8e44ad', fontweight='bold')
        
        # Separadores verticales
        for x in [0.35, 0.65]:
            ax_resumen.plot([x, x], [0.15, 0.85], '--', color='#bdc3c7', linewidth=1,
                        transform=ax_resumen.transAxes, alpha=0.5)
        
        return fig
    
    def guardar_reporte(self, fig, equipo_comparar):
        """Guarda el reporte en PDF"""
        if fig is None:
            return
        
        equipo_filename = equipo_comparar.replace(' ', '_').replace('/', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evolucion_clasificacion_Villarreal_vs_{equipo_filename}_{timestamp}.pdf"
        
        fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1,
                   facecolor='white', format='pdf')
        return filename


def main():
    """Funcion principal"""
    
    # Cargar datos primero para obtener lista de equipos
    clasificador = ClasificacionFromParquet()
    
    if clasificador.cargar_datos() is None:
        return
    
    df = clasificador.construir_clasificacion_por_jornada()
    
    if df is None or df.empty:
        pass
        return
    
    # Obtener lista de equipos
    visualizador = VisualizadorEvolucionClasificacion(df, parquet_path='extraccion_opta/datos_opta_parquet/team_stats.parquet')
    equipos = visualizador.equipos
    
    # Buscar Villarreal (flexible)
    villarreal_nombre = None
    for eq in equipos:
        if 'villarreal' in eq.lower():
            villarreal_nombre = eq
            break
    
    if not villarreal_nombre:
        pass
        return
    
    
    # Mostrar equipos disponibles (excepto Villarreal)
    equipos_comparar = [eq for eq in equipos if eq != villarreal_nombre]
    
    for i, equipo in enumerate(equipos_comparar, 1):
        pass
    
    # Seleccionar equipo
    equipo_encontrado = None
    for _ in range(3):
        try:
            seleccion = int(input(f"\nSelecciona equipo para comparar con {villarreal_nombre} (1-{len(equipos_comparar)}): "))
            if 1 <= seleccion <= len(equipos_comparar):
                equipo_encontrado = equipos_comparar[seleccion - 1]
                break
            else:
                pass
        except EOFError:
            equipo_encontrado = equipos_comparar[0] if equipos_comparar else None
            break
        except ValueError:
            pass
    if equipo_encontrado is None and equipos_comparar:
        equipo_encontrado = equipos_comparar[0]
    
    # Guardar CSV opcional
    clasificador.guardar_datos()
    
    # Generar reporte (usando el nombre exacto encontrado)
    fig = visualizador.crear_reporte_evolucion(equipo_encontrado, villarreal_nombre)
    
    if fig:
        plt.show()
        visualizador.guardar_reporte(fig, equipo_encontrado)
    else:
        pass


if __name__ == "__main__":
    main()