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

class FaltasOfensivasReport:
    def __init__(self, 
                 match_events_path="./extraccion_opta/datos_opta_parquet/abp_events.parquet",
                 xg_events_path="./extraccion_opta/datos_opta_parquet/xg_events.parquet"):
        self.match_events_path = match_events_path
        self.xg_events_path = xg_events_path
        self.match_events_df = None
        self.xg_events_df = None
        self.load_opta_data()  # En lugar de load_data()
    
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
    
    def normalize_timestamp(self, timestamp):
        """Normaliza timestamps quitando la Z final si existe"""
        if pd.isna(timestamp):
            return timestamp
        
        timestamp_str = str(timestamp).strip()
        
        # Quitar la Z final si existe
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1]
        
        # Asegurar formato consistente
        try:
            # Convertir a datetime y volver a string para normalizar formato
            dt = pd.to_datetime(timestamp_str)
            return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Mantener 3 decimales
        except:
            # Si falla la conversión, devolver el string sin Z
            return timestamp_str
    
    def load_villarreal_logo(self):
        """Carga el escudo del Villarreal usando búsqueda por similitud"""
        return self.find_team_logo_by_similarity('Villarreal')

    def find_team_logo_by_similarity(self, equipo):
        """Busca el escudo del equipo por similitud en la carpeta escudos"""
        if not os.path.exists('assets/escudos'):
            return None
        
        # Obtener todos los archivos .png en la carpeta
        escudos_disponibles = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
        
        if not escudos_disponibles:
            return None
        
        # Limpiar nombre del equipo para comparar
        equipo_clean = equipo.lower().replace(' ', '').replace('cf', '').replace('fc', '').replace('real', '').replace('rcd', '')
        
        best_match = None
        best_similarity = 0
        
        # Buscar por similitud
        for escudo_file in escudos_disponibles:
            escudo_name = escudo_file.replace('.png', '').lower().replace('_', '').replace('cf', '').replace('fc', '').replace('real', '').replace('rcd', '')
            
            similarity = self.similarity(equipo_clean, escudo_name)
            
            if similarity > best_similarity and similarity > 0.4:  # Mínimo 40% similitud
                best_similarity = similarity
                best_match = escudo_file
        
        # Cargar el mejor match
        if best_match:
            try:
                logo_path = f"assets/escudos/{best_match}"
                
                # CARGAR Y REDIMENSIONAR A TAMAÑO FIJO
                escudo_original = plt.imread(logo_path)
                escudo_redimensionado = self.resize_image_to_fixed_size(escudo_original, target_size=100)
                
                return escudo_redimensionado
            except Exception as e:
                pass
        
        return None

    def resize_image_to_fixed_size(self, image, target_size=100):
        """Redimensiona imagen a un tamaño fijo manteniendo proporción"""
        try:
            from PIL import Image as PILImage
            import numpy as np
            
            # Convertir de matplotlib a PIL
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Crear imagen PIL
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    pil_image = PILImage.fromarray(image, 'RGBA')
                else:  # RGB
                    pil_image = PILImage.fromarray(image, 'RGB')
            else:  # Grayscale
                pil_image = PILImage.fromarray(image, 'L')
            
            # Redimensionar manteniendo proporción
            pil_image.thumbnail((target_size, target_size), PILImage.Resampling.LANCZOS)
            
            # Crear imagen cuadrada con fondo transparente
            square_image = PILImage.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
            
            # Centrar la imagen redimensionada
            x_offset = (target_size - pil_image.width) // 2
            y_offset = (target_size - pil_image.height) // 2
            square_image.paste(pil_image, (x_offset, y_offset))
            
            # Convertir de vuelta a numpy array
            return np.array(square_image) / 255.0
            
        except Exception as e:
            pass
            return image

    def convert_to_grayscale(self, image):
        """Convierte imagen a escala de grises"""
        try:
            # Si la imagen tiene canal alfa, mantenerlo
            if image.shape[2] == 4:  # RGBA
                gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])  # Solo RGB
                # Crear imagen en escala de grises manteniendo alfa
                gray_image = np.zeros_like(image)
                gray_image[..., 0] = gray  # R
                gray_image[..., 1] = gray  # G  
                gray_image[..., 2] = gray  # B
                gray_image[..., 3] = image[..., 3]  # Alpha
                return gray_image
            else:  # RGB
                gray = np.dot(image, [0.2989, 0.5870, 0.1140])
                return np.stack([gray, gray, gray], axis=2)
        except Exception as e:
            pass
            return image
        
    def load_opta_data(self):
        """Carga todos los datasets necesarios"""
        try:
            # Cargar match events
            if not os.path.exists(self.match_events_path):
                print(f"❌ Error: No se encontró {self.match_events_path}")
                return
            self.match_events_df = pd.read_parquet(self.match_events_path)
            self.match_events_df['timeStamp'] = self.match_events_df['timeStamp'].apply(self.normalize_timestamp)
            
            # Cargar xG events
            if not os.path.exists(self.xg_events_path):
                print(f"❌ Error: No se encontró {self.xg_events_path}")
                return
            self.xg_events_df = pd.read_parquet(self.xg_events_path)
            self.xg_events_df['timeStamp'] = self.xg_events_df['timeStamp'].apply(self.normalize_timestamp)

            # TEMPORAL - para verificar Zone
            if 'Zone' in self.match_events_df.columns:
                pass
            else:
                print("❌ Columna 'Zone' NO encontrada")
                zone_cols = [col for col in self.match_events_df.columns if 'zone' in col.lower()]

            # Añadir al final del método load_data()
            if 'Free kick' in self.match_events_df.columns:
                pass
            else:
                print("❌ Columna 'Free kick' NO encontrada")
                free_cols = [col for col in self.match_events_df.columns if 'free' in col.lower() or 'kick' in col.lower()]
            
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")
    
    def get_team_fault_stats(self, equipo_seleccionado=None):
        if self.match_events_df is None:
            return None
        
        team_stats_list = []
        equipos = self.match_events_df['Team Name'].dropna().unique()
        
        for equipo in equipos:
            # Obtener partidos del equipo
            team_matches = self.match_events_df[self.match_events_df['Team Name'] == equipo]['Match ID'].unique()
            partidos_jugados = len(team_matches)
            
            # Obtener secuencias de faltas indirectas
            fk_sequences = self.get_freekick_indirect_sequences(equipo, team_matches)
            num_secuencias = len(fk_sequences)
            
            # Contar goles y tiros de las secuencias
            total_goles = 0
            total_tiros = 0
            
            for seq in fk_sequences:
                for event in seq['shot_events']:
                    if event['Event Name'] == 'Goal':
                        total_goles += 1
                    if event['Event Name'] in ['Miss', 'Goal', 'Post', 'Attempt Saved']:
                        total_tiros += 1
            
            stats = {
                'EQUIPO': equipo,
                'num_jornadas': partidos_jugados,
                'faltas_indirectas_total': num_secuencias,  # Número de secuencias
                'goles_faltas_indirectas': total_goles,     # Goles de secuencias
                'tiros_falta_total': total_tiros,           # Tiros de secuencias
            }
            
            # Calcular métricas derivadas
            stats['faltas_por_jornada'] = num_secuencias / partidos_jugados
            stats['goles_por_falta'] = total_goles / num_secuencias if num_secuencias > 0 else 0
            stats['tiros_por_falta'] = total_tiros / num_secuencias if num_secuencias > 0 else 0
            
            team_stats_list.append(stats)
        
        team_stats = pd.DataFrame(team_stats_list)
        
        if equipo_seleccionado:
            equipo_data = team_stats[team_stats['EQUIPO'] == equipo_seleccionado]
            if not equipo_data.empty:
                return team_stats, equipo_data.iloc[0]
            else:
                return team_stats, None
        
        return team_stats
    
    def similarity(self, a, b):
        """Calcula la similitud entre dos strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def load_team_logo(self, equipo):
        """Carga el escudo del equipo - mismo método que el código original"""
        possible_names = [
            equipo,
            equipo.replace(' ', '_'),
            equipo.replace(' ', ''),
            equipo.lower(),
            equipo.lower().replace(' ', '_'),
            equipo.lower().replace(' ', '')
        ]
        
        for name in possible_names:
            logo_path = f"assets/escudos/{name}.png"
            if os.path.exists(logo_path):
                try:
                    return plt.imread(logo_path)
                except Exception as e:
                    continue
        
        return None
    
    def load_ball_image(self):
        """Carga la imagen del balón"""
        ball_path = "assets/balon.png"
        if os.path.exists(ball_path):
            try:
                return plt.imread(ball_path)
            except Exception as e:
                pass
                return None
        return None
    
    def load_background(self):
        """Carga el fondo del informe"""
        bg_path = "assets/fondo_informes.png"
        if os.path.exists(bg_path):
            try:
                return plt.imread(bg_path)
            except Exception as e:
                pass
                return None
        return None
    
    def get_freekick_indirect_sequences(self, team_name, match_ids=None):
        """
        Extrae secuencias de faltas indirectas (Free kick taken = Sí y x > 55)
        """
        df = self.match_events_df.copy()
        
        if match_ids is not None:
            df = df[df['Match ID'].isin(match_ids)]
        
        # Filtrar faltas indirectas del equipo (Free kick taken = Sí y x > 55)
        team_freekicks_indirect = df[
            (df['Team Name'] == team_name) & 
            (df['Free kick taken'] == 'Sí') &
            (df['Zone'].isin(['Center', 'Right', 'Left']))
        ].copy()
        
        freekick_sequences = []
        
        for _, fk_event in team_freekicks_indirect.iterrows():
            match_id = fk_event['Match ID']
            match_events = df[df['Match ID'] == match_id].sort_values(['timeMin', 'timeSec']).reset_index()
            
            # Encontrar el índice de la falta indirecta en el partido
            fk_idx = None
            for idx, event in match_events.iterrows():
                if (event['Team Name'] == team_name and 
                    event['timeMin'] == fk_event['timeMin'] and 
                    event['timeSec'] == fk_event['timeSec'] and
                    event['Free kick taken'] == 'Sí'):
                    fk_idx = idx
                    break
            
            if fk_idx is None:
                continue
                
            # Analizar secuencia después de la falta (IGUAL QUE CORNERS)
            sequence_events = []
            current_time = fk_event['timeMin'] * 60 + fk_event['timeSec']
            pass_count = 0  # Contador de pases
            last_pass_timestamp = fk_event['timeStamp'] 
            
            for next_idx in range(fk_idx + 1, len(match_events)):
                next_event = match_events.iloc[next_idx]
                next_time = next_event['timeMin'] * 60 + next_event['timeSec']
                time_diff = next_time - current_time

                
                
                # Más de 5 segundos o cambio de período
                if time_diff > 5 or next_event.get('periodId', 1) != fk_event.get('periodId', 1):
                    break
                
                # Eventos que terminan la secuencia
                if next_event['Event Name'] in ['Corner Awarded', 'Out','Smother','Foul','Save','Offside', 'End Period']:
                    break
                    
                # Pass con x < 55
                if next_event['Event Name'] == 'Pass' and float(next_event.get('x', 100)) < 55:
                    break

                # Contar pases del mismo equipo y verificar límite
                if next_event['Event Name'] == 'Pass' and next_event['Team Name'] == team_name:
                    # Solo contamos el pase si su timestamp es posterior al del último pase contado.
                    if next_event['timeStamp'] > last_pass_timestamp:
                        pass_count += 1
                        last_pass_timestamp = next_event['timeStamp']  # Actualizamos el timestamp.
                        
                        # Si llegamos a 5 o más pases, verificar la nueva condición
                        if pass_count >= 5:
                            # Contar pases con x < 70 en los últimos eventos
                            passes_back_field = 0
                            for check_idx in range(fk_idx + 1, next_idx + 1):
                                check_event = match_events.iloc[check_idx]
                                if (check_event['Event Name'] == 'Pass' and 
                                    check_event['Team Name'] == team_name and
                                    float(check_event.get('x', 0)) < 70):
                                    passes_back_field += 1
                            
                            # Si hay 2 o más pases con x < 70, cortar la secuencia
                            if passes_back_field >= 2:
                                break
                
                sequence_events.append(next_event)
                current_time = next_time
            
            # Buscar eventos de finalización en la secuencia
            shot_events = []
            for event in sequence_events:
                if event['Event Name'] in ['Miss', 'Goal', 'Post', 'Attempt Saved']:
                    shot_events.append(event)
            
            freekick_sequences.append({
                'freekick_event': fk_event,
                'sequence_events': sequence_events,
                'shot_events': shot_events
            })
        
        return freekick_sequences

    def create_visualization(self, equipo_seleccionado, figsize=(11.69, 8.27)):
        """Crea la visualización completa del informe de faltas ofensivas"""
        
        # Obtener datos de faltas
        result = self.get_team_fault_stats(equipo_seleccionado)
        if result is None:
            pass
            return None
        
        team_stats, equipo_data = result
        if equipo_data is None:
            pass
            return None
                
        # Configurar fuentes y eliminar espacios (copiado de ABP1)
        plt.rcParams['font.family'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['font.weight'] = 'normal'

        # Configuración agresiva para eliminar espacios
        plt.rcParams.update({
            'figure.autolayout': False,
            'figure.constrained_layout.use': False,
            'figure.subplot.left': 0,
            'figure.subplot.right': 1, 
            'figure.subplot.top': 1,
            'figure.subplot.bottom': 0,
            'figure.subplot.wspace': 0,
            'figure.subplot.hspace': 0,
            'axes.xmargin': 0,
            'axes.ymargin': 0,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0
        })

        # Crear figura
        fig = plt.figure(figsize=figsize, facecolor='white', constrained_layout=False)
        fig.patch.set_visible(False)
        
        # Cargar y establecer fondo
        background = self.load_background()
        if background is not None:
            try:
                ax_background = fig.add_axes([0, 0, 1, 1], zorder=-1)
                ax_background.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25, zorder=-1)
                ax_background.axis('off')
            except Exception as e:
                pass
        
        # Configurar grid - Layout similar al PNG
        gs = fig.add_gridspec(2, 4, 
                height_ratios=[0.2, 1],  # Más espacio para el título
                width_ratios=[1, 1, 1.4, 0.1],  # Mejor distribución en A4
                hspace=0.25, wspace=0.45,
                left=0.08, right=0.95, top=0.92, bottom=0.08)
        
        # Área del título
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        # Título principal
        ax_title.text(0.5, 0.6, 'FALTAS OFENSIVAS', 
                     fontsize=28, weight='bold', ha='center', va='center',
                     color='#1e3d59', family='serif')
        
        # Logos
        ball = self.load_ball_image()
        if ball is not None:
            try:
                imagebox = OffsetImage(ball, zoom=0.12)
                ab = AnnotationBbox(imagebox, (0.05, 0.5), frameon=False)
                ax_title.add_artist(ab)
            except:
                pass

        # ESCUDOS CON MISMO TAMAÑO GARANTIZADO
        escudo_zoom = 1.2

        # Escudo Villarreal
        villarreal_logo = self.find_team_logo_by_similarity('Villarreal')
        if villarreal_logo is not None:
            try:
                imagebox = OffsetImage(villarreal_logo, zoom=escudo_zoom)
                ab = AnnotationBbox(imagebox, (0.88, 0.5), frameon=False, zorder=2) 
                ax_title.add_artist(ab)
            except Exception as e:
                print(f"❌ Error con escudo Villarreal: {e}")

        # Escudo del equipo seleccionado  
        equipo_logo = self.find_team_logo_by_similarity(equipo_seleccionado)
        if equipo_logo is not None:
            try:
                imagebox = OffsetImage(equipo_logo, zoom=escudo_zoom)  # MISMO ZOOM
                ab = AnnotationBbox(imagebox, (0.92, 0.5), frameon=False, zorder=1)
                ax_title.add_artist(ab)
            except Exception as e:
                print(f"❌ Error con escudo {equipo_seleccionado}: {e}")
        else:
            print(f"❌ No se encontró escudo para {equipo_seleccionado}")
        
        # Gráfico 1: Goles a favor de falta indirecta (izquierda)
        ax_goles = fig.add_subplot(gs[1, 0])
        ax_goles.set_facecolor('none')
        ax_goles.set_title('Goles a favor de falta indirecta', fontsize=14, weight='bold', 
                          color='#1e3d59', pad=15)
        self.plot_goles_falta(ax_goles, team_stats, equipo_seleccionado)
        
        # Gráfico 2: Goles a favor por falta indirecta (centro)
        ax_eficiencia = fig.add_subplot(gs[1, 1])
        ax_eficiencia.set_facecolor('none')
        ax_eficiencia.set_title('Goles a favor por falta indirecta', fontsize=14, weight='bold', 
                               color='#1e3d59', pad=15)
        self.plot_eficiencia_falta(ax_eficiencia, team_stats, equipo_seleccionado)  

        
        # Gráfico 3: Scatter plot (derecha)
        ax_scatter = fig.add_subplot(gs[1, 2])
        ax_scatter.set_facecolor('none')
        ax_scatter.set_title('Suma Tiros a favor / falta indirecta', fontsize=14, weight='bold', 
                            color='#1e3d59', pad=15)
        self.plot_fault_scatter(ax_scatter, team_stats, equipo_seleccionado)
        
        return fig
    
    def plot_goles_falta(self, ax, team_stats, equipo_seleccionado):
        """Dibuja el gráfico de goles a favor de falta indirecta"""
        # Ordenar por goles de faltas indirectas
        sorted_data = team_stats.sort_values('goles_faltas_indirectas', ascending=True)
        
        # Colores: destacar equipo seleccionado y Villarreal
        colors = []
        for team in sorted_data['EQUIPO']:
            if team == equipo_seleccionado:
                colors.append('#e74c3c')  # Rojo para equipo seleccionado
            elif 'villarreal' in team.lower():
                colors.append('#f39c12')  # Naranja para Villarreal
            else:
                colors.append('#95a5a6')  # Gris para otros
        
        # Crear gráfico horizontal
        bars = ax.barh(range(len(sorted_data)), sorted_data['goles_faltas_indirectas'], 
                    height=0.6,  # Barras más estrechas
                    color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=0.5)
        
        # Añadir valores para equipos destacados
        for i, (bar, valor, team) in enumerate(zip(bars, sorted_data['goles_faltas_indirectas'], sorted_data['EQUIPO'])):
            if team == equipo_seleccionado or 'villarreal' in team.lower():
                ax.text((valor) + 0.05, i, f'{int(valor)}', va='center', fontsize=9, weight='bold')
        
        # Configurar ejes
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data['EQUIPO'], fontsize=8)
        ax.set_xlabel('Goles de Falta Indirecta', fontsize=10, weight='bold')
        
        # Línea promedio
        promedio = sorted_data['goles_faltas_indirectas'].mean()
        ax.axvline(promedio, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def plot_eficiencia_falta(self, ax, team_stats, equipo_seleccionado):
        """Dibuja el gráfico de eficiencia (goles por falta indirecta)"""
        sorted_data = team_stats.sort_values('goles_por_falta', ascending=True)
        
        colors = []
        for team in sorted_data['EQUIPO']:
            if team == equipo_seleccionado:
                colors.append('#e74c3c')
            elif 'villarreal' in team.lower():
                colors.append('#f39c12')
            else:
                colors.append('#95a5a6')
        
        bars = ax.barh(range(len(sorted_data)), sorted_data['goles_por_falta'], 
                    color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=0.5)
        
        for i, (bar, valor, team) in enumerate(zip(bars, sorted_data['goles_por_falta'], sorted_data['EQUIPO'])):
            if team == equipo_seleccionado or 'villarreal' in team.lower():
                ax.text((valor) + 0.001, i, f'{float(valor):.3f}', va='center', fontsize=9, weight='bold')
        
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data['EQUIPO'], fontsize=8)
        ax.set_xlabel('Goles por Falta Indirecta', fontsize=10, weight='bold')
        
        promedio = sorted_data['goles_por_falta'].mean()
        ax.axvline(promedio, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def plot_fault_scatter(self, ax, team_stats, equipo_seleccionado):
        """Dibuja el scatter plot: Faltas Indirectas vs Tiros por Falta"""
        x_data = team_stats['faltas_indirectas_total']
        y_data = team_stats['tiros_por_falta']
        equipos = team_stats['EQUIPO']
        
        # Calcular promedios para las líneas de la cruz
        x_promedio = x_data.mean()
        y_promedio = y_data.mean()
        
        # Dibujar líneas de la cruz
        ax.axvline(x_promedio, color='#34495e', linestyle='-', linewidth=2, alpha=0.6)
        ax.axhline(y_promedio, color='#34495e', linestyle='-', linewidth=2, alpha=0.6)
        
        # PROCESAR EN DOS PASADAS: primero equipos normales, después destacados
        
        # PASADA 1: Equipos normales (z-order bajo)
        for i, (equipo, x_val, y_val) in enumerate(zip(equipos, x_data, y_data)):
            is_selected = equipo == equipo_seleccionado
            is_villarreal = 'villarreal' in equipo.lower()
            
            # Solo procesar equipos normales en esta pasada
            if is_selected or is_villarreal:
                continue
                
            
            # Buscar escudo por similitud
            escudo = self.find_team_logo_by_similarity(equipo)
            
            if escudo is not None:
                try:
                    # Convertir a blanco y negro y tamaño normal
                    escudo_bn = self.convert_to_grayscale(escudo)
                    zoom_size = 0.24
                    imagebox = OffsetImage(escudo_bn, zoom=zoom_size, alpha=0.8)
                    ab = AnnotationBbox(imagebox, (x_val, y_val), frameon=False, pad=0, zorder=1)  # z-order bajo
                    ax.add_artist(ab)
                    continue
                except Exception as e:
                    print(f"❌ Error al mostrar escudo para {equipo}: {e}")
            
            # Si no hay escudo, mostrar nombre del equipo
            ax.text(x_val, y_val, equipo, ha='center', va='center', 
                    fontsize=7, weight='normal', color='#2c3e50',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='#2c3e50', alpha=0.8), zorder=1)
        
        # PASADA 2: Equipos destacados (z-order alto) 
        for i, (equipo, x_val, y_val) in enumerate(zip(equipos, x_data, y_data)):
            is_selected = equipo == equipo_seleccionado
            is_villarreal = 'villarreal' in equipo.lower()
            
            # Solo procesar equipos destacados en esta pasada
            if not (is_selected or is_villarreal):
                continue
                
            
            # Buscar escudo por similitud
            escudo = self.find_team_logo_by_similarity(equipo)
            
            if escudo is not None:
                try:
                    # Mantener colores originales y más grandes
                    zoom_size = 0.45  # 33% más grande que 0.24
                    imagebox = OffsetImage(escudo, zoom=zoom_size, alpha=1.0)
                    ab = AnnotationBbox(imagebox, (x_val, y_val), frameon=False, pad=0, zorder=10)  # z-order alto
                    ax.add_artist(ab)
                    continue
                except Exception as e:
                    print(f"❌ Error al mostrar escudo para {equipo}: {e}")
            
            # Si no hay escudo, mostrar nombre del equipo
            color = '#e74c3c' if is_selected else '#f39c12'
            ax.text(x_val, y_val, equipo, ha='center', va='center', 
                    fontsize=8, weight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.8), zorder=10)  # z-order alto
        
        # Resto del método igual...
        ax.set_xlabel('Suma de Faltas Indirectas a Favor', fontsize=12, weight='bold')
        ax.set_ylabel('Tiros por Falta Indirecta', fontsize=12, weight='bold')
        
        # Asegurar que todos los puntos sean visibles
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()

        # Añadir margen del 10%
        x_margin = (x_max - x_min) * 0.1 if x_max > x_min else 1
        y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1

        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)


        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

def main():
    pass
    
    # Verificar que existan los archivos OPTA
    match_events_path = "./extraccion_opta/datos_opta_parquet/abp_events.parquet"
    if not os.path.exists(match_events_path):
        pass
        return
    
    report_generator = FaltasOfensivasReport()
    
    if report_generator.match_events_df is None:
        pass
        return
    
    # Seleccionar equipo de la lista de OPTA
    equipos = sorted(report_generator.match_events_df['Team Name'].dropna().unique())
    
    for i, equipo in enumerate(equipos, 1):
        pass
    
    while True:
        try:
            seleccion = input(f"\nSelecciona un equipo (1-{len(equipos)}): ").strip()
            indice = int(seleccion) - 1
            if 0 <= indice < len(equipos):
                equipo_seleccionado = equipos[indice]
                break
            else:
                pass
        except ValueError:
            pass

    
    fig = report_generator.create_visualization(equipo_seleccionado)
    
    if fig:
        plt.show()
        
        # Guardar como PDF en formato A4
        output_path = f"reporte_faltas_ofensivas_{equipo_seleccionado.replace(' ', '_')}.pdf"
        report_generator.guardar_sin_espacios(fig, output_path)
        
    else:
        print("❌ No se pudo generar la visualización")

if __name__ == "__main__":
    main()