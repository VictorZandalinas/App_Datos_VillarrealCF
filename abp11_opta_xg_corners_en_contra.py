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

class XGCornersDefensivosReport:
    def __init__(self, 
                 match_events_path="./extraccion_opta/datos_opta_parquet/abp_events.parquet",
                 xg_events_path="./extraccion_opta/datos_opta_parquet/xg_events.parquet",
                 team_stats_path="./extraccion_opta/datos_opta_parquet/team_stats.parquet"):
        """
        Inicializa la clase para generar informes de xG de córners defensivos (en contra)
        """
        self.match_events_path = match_events_path
        self.xg_events_path = xg_events_path
        self.team_stats_path = team_stats_path
        self.match_events_df = None
        self.xg_events_df = None
        self.team_stats_df = None
        self.corner_xg_defensivo_data = None
        self.load_data()
        self.calculate_all_teams_xg_defensive_stats()

    def guardar_sin_espacios(self, fig, filename):
        """Guarda sin espacios manteniendo landscape A4"""
        # Aseguramos el tamaño A4 landscape (11.69 x 8.27 pulgadas)
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
    
    def load_data(self):
        """Carga todos los datasets necesarios"""
        try:
            # Cargar match events
            if not os.path.exists(self.match_events_path):
                print(f"❌ Error: No se encontró {self.match_events_path}")
                return
            self.match_events_df = pd.read_parquet(self.match_events_path)
            self.match_events_df['timeStamp'] = self.match_events_df['timeStamp'].apply(self.normalize_timestamp) # <-- AÑADIR
            
            # Cargar xG events
            if not os.path.exists(self.xg_events_path):
                print(f"❌ Error: No se encontró {self.xg_events_path}")
                return
            self.xg_events_df = pd.read_parquet(self.xg_events_path)
            self.xg_events_df['timeStamp'] = self.xg_events_df['timeStamp'].apply(self.normalize_timestamp) # <-- AÑADIR
            
            # Cargar team stats
            if not os.path.exists(self.team_stats_path):
                print(f"❌ Error: No se encontró {self.team_stats_path}")
                return
            self.team_stats_df = pd.read_parquet(self.team_stats_path)
            
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")
    
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

    def get_corner_sequences(self, team_name, match_ids=None):
        """
        Extrae secuencias de corners para un equipo específico
        """
        df = self.match_events_df.copy()
        
        if match_ids is not None:
            df = df[df['Match ID'].isin(match_ids)]
        
        # Filtrar corners del equipo
        team_corners = df[
            (df['Team Name'] == team_name) & 
            (df['Corner taken'] == 'Sí')
        ].copy()
        
        corner_sequences = []
        
        for _, corner_event in team_corners.iterrows():
            match_id = corner_event['Match ID']
            match_events = df[df['Match ID'] == match_id].sort_values(['timeMin', 'timeSec']).reset_index()
            
            # Encontrar el índice del corner en el partido
            corner_idx = None
            for idx, event in match_events.iterrows():
                if (event['Team Name'] == team_name and 
                    event['timeMin'] == corner_event['timeMin'] and 
                    event['timeSec'] == corner_event['timeSec'] and
                    event['Corner taken'] == 'Sí'):
                    corner_idx = idx
                    break
            
            if corner_idx is None:
                continue
                
            # Analizar secuencia después del corner
            sequence_events = []
            current_time = corner_event['timeMin'] * 60 + corner_event['timeSec']
            pass_count = 0  # Contador de pases
            
            # ▼▼▼ AÑADE ESTA LÍNEA ▼▼▼
            # Guarda el timeStamp del último pase contado. Empezamos con el del córner.
            last_pass_timestamp = corner_event['timeStamp']

            for next_idx in range(corner_idx + 1, len(match_events)):
                next_event = match_events.iloc[next_idx]
                next_time = next_event['timeMin'] * 60 + next_event['timeSec']
                time_diff = next_time - current_time
                
                # Más de 5 segundos o cambio de período
                if time_diff > 5 or next_event.get('periodId', 1) != corner_event.get('periodId', 1):
                    break
                
                # Eventos que terminan la secuencia
                if next_event['Event Name'] in ['Corner Awarded', 'Out','Smother','Foul','Save','Offside', 'End Period']:
                    break
                    
                # Pass con x < 55
                if next_event['Event Name'] == 'Pass' and float(next_event.get('x', 100)) < 55:
                    break
                
                # ← NUEVO: Contar pases del mismo equipo y verificar límite
                if next_event['Event Name'] == 'Pass' and next_event['Team Name'] == team_name:
                    # Solo contamos el pase si su timestamp es posterior al del último pase contado.
                    if next_event['timeStamp'] > last_pass_timestamp:
                        pass_count += 1
                        last_pass_timestamp = next_event['timeStamp']  # Actualizamos el timestamp.
                        
                        # Si llegamos a 5 o más pases, verificar la nueva condición
                        if pass_count >= 5:
                            # Contar pases con x < 70 en los últimos eventos
                            passes_back_field = 0
                            for check_idx in range(corner_idx + 1, next_idx + 1):
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
            
            corner_sequences.append({
                'corner_event': corner_event,
                'sequence_events': sequence_events,
                'shot_events': shot_events
            })
        
        return corner_sequences

    def get_xg_for_shot_events(self, shot_events):
        """
        Obtiene xG para eventos de tiro mediante merge por timeStamp
        """
        if not shot_events or self.xg_events_df is None:
            return []
        
        xg_values = []
        
        for event in shot_events:
            # Hacer merge por Match ID, Team ID y timeStamp
            matching_xg = self.xg_events_df[
                (self.xg_events_df['Match ID'] == event['Match ID']) &
                (self.xg_events_df['Team ID'] == event['Team ID']) &
                (self.xg_events_df['timeStamp'] == event['timeStamp'])
            ]
            
            if not matching_xg.empty:
                try:
                    xg_val = float(matching_xg.iloc[0]['qualifier 321'])
                    xg_values.append(xg_val)
                except (ValueError, TypeError, KeyError):
                    xg_values.append(0.0)
            else:
                xg_values.append(0.0)
        
        return xg_values
    
    def calculate_all_teams_xg_defensive_stats(self):
        """
        Calcula las estadísticas de xG en contra desde córners para todos los equipos,
        analizando las secuencias ofensivas de sus rivales en cada partido.
        """
        if self.match_events_df is None or self.xg_events_df is None:
            print("❌ No hay datos para procesar")
            self.corner_xg_defensivo_data = pd.DataFrame()
            return

        
        all_teams = sorted(self.match_events_df['Team Name'].dropna().unique())
        stats_list = []

        for team_name in all_teams:
            pass
            
            team_match_ids = self.match_events_df[self.match_events_df['Team Name'] == team_name]['Match ID'].unique()
            
            total_xg_concedido = 0
            total_shots_concedidos = 0
            total_corners_concedidos = 0
            partidos_jugados = len(team_match_ids)
            
            if partidos_jugados == 0:
                continue
                
            for match_id in team_match_ids:
                # Identificar al rival en este partido
                teams_in_match = self.match_events_df[self.match_events_df['Match ID'] == match_id]['Team Name'].unique()
                opponent_name = next((t for t in teams_in_match if t != team_name), None)
                
                if not opponent_name:
                    continue
                
                # Obtener las secuencias de córner A FAVOR del RIVAL en este partido
                opponent_corner_sequences = self.get_corner_sequences(opponent_name, match_ids=[match_id])
                total_corners_concedidos += len(opponent_corner_sequences)
                
                for seq in opponent_corner_sequences:
                    if seq['shot_events']:
                        # Obtener el xG de los remates del rival
                        xg_values = self.get_xg_for_shot_events(seq['shot_events'])
                        total_xg_concedido += sum(xg_values)
                        total_shots_concedidos += len(seq['shot_events'])
            
            # Calcular métricas finales para el equipo
            stats_list.append({
                'Team Name': team_name,
                'xg_corners_contra_total': total_xg_concedido,
                'shots_contra_from_corners': total_shots_concedidos,
                'corners_concedidos': total_corners_concedidos,
                'num_matches': partidos_jugados,
                'xg_contra_por_partido': total_xg_concedido / partidos_jugados if partidos_jugados > 0 else 0,
                'xg_contra_por_corner': total_xg_concedido / total_corners_concedidos if total_corners_concedidos > 0 else 0
            })

        # Crear el DataFrame final
        self.corner_xg_defensivo_data = pd.DataFrame(stats_list).fillna(0)
    
    # Métodos de diseño visual (copiados y adaptados)
    def find_team_logo_by_similarity(self, equipo):
        """Busca el escudo del equipo por similitud en la carpeta escudos"""
        if not os.path.exists('assets/escudos'):
            return None
        
        escudos_disponibles = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
        if not escudos_disponibles:
            return None
        
        equipo_clean = equipo.lower().replace(' ', '').replace('cf', '').replace('fc', '').replace('real', '').replace('rcd', '')
        
        best_match = None
        best_similarity = 0
        
        for escudo_file in escudos_disponibles:
            escudo_name = escudo_file.replace('.png', '').lower().replace('_', '').replace('cf', '').replace('fc', '').replace('real', '').replace('rcd', '')
            similarity = self.similarity(equipo_clean, escudo_name)
            
            if similarity > best_similarity and similarity > 0.4:
                best_similarity = similarity
                best_match = escudo_file
        
        if best_match:
            try:
                logo_path = f"assets/escudos/{best_match}"
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
            
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    pil_image = PILImage.fromarray(image, 'RGBA')
                else:
                    pil_image = PILImage.fromarray(image, 'RGB')
            else:
                pil_image = PILImage.fromarray(image, 'L')
            
            pil_image.thumbnail((target_size, target_size), PILImage.Resampling.LANCZOS)
            square_image = PILImage.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
            
            x_offset = (target_size - pil_image.width) // 2
            y_offset = (target_size - pil_image.height) // 2
            square_image.paste(pil_image, (x_offset, y_offset))
            
            return np.array(square_image) / 255.0
            
        except Exception as e:
            pass
            return image
    
    def convert_to_grayscale(self, image):
        """Convierte imagen a escala de grises"""
        try:
            if image.shape[2] == 4:
                gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
                gray_image = np.zeros_like(image)
                gray_image[..., 0] = gray
                gray_image[..., 1] = gray  
                gray_image[..., 2] = gray
                gray_image[..., 3] = image[..., 3]
                return gray_image
            else:
                gray = np.dot(image, [0.2989, 0.5870, 0.1140])
                return np.stack([gray, gray, gray], axis=2)
        except Exception as e:
            pass
            return image
    
    def similarity(self, a, b):
        """Calcula la similitud entre dos strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
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
    
    def create_xg_defensivo_visualization(self, equipo_seleccionado, figsize=(11.69, 8.27)):
        """Crea la visualización completa del informe de xG de córners defensivos"""
        
        team_stats = self.corner_xg_defensivo_data
        if team_stats is None or team_stats.empty:
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
            except Exception as e:
                pass
        
        # Configurar grid - Layout similar al PNG
        gs = fig.add_gridspec(2, 4, 
                 height_ratios=[0.15, 1], 
                 width_ratios=[1.2, 1, 1.6, 0.1],
                 hspace=0.3, wspace=0.35,
                 left=0.10, right=0.99, top=0.95, bottom=0.05)
        
        # Área del título
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        # Título principal
        ax_title.text(0.5, 0.6, 'XG CÓRNERS DEFENSIVOS', 
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
        
        # Escudos
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
                imagebox = OffsetImage(equipo_logo, zoom=escudo_zoom)
                ab = AnnotationBbox(imagebox, (0.92, 0.5), frameon=False, zorder=1)
                ax_title.add_artist(ab)
            except Exception as e:
                print(f"❌ Error con escudo {equipo_seleccionado}: {e}")
        
        # Gráfico 1: xG de córner en contra (izquierda)
        ax_xg_contra = fig.add_subplot(gs[1, 0])
        ax_xg_contra.set_facecolor('none')
        ax_xg_contra.set_title('xG de córner en contra', fontsize=14, weight='bold', 
                  color='#1e3d59', pad=15)
        self.plot_xg_contra_total(ax_xg_contra, team_stats, equipo_seleccionado)
        
        # Gráfico 2: xG en contra por córner (centro)
        ax_xg_contra_por_corner = fig.add_subplot(gs[1, 1])
        ax_xg_contra_por_corner.set_facecolor('none')
        ax_xg_contra_por_corner.set_title('xG en contra por córner', fontsize=14, weight='bold', 
                               color='#1e3d59', pad=15)
        self.plot_xg_contra_por_corner(ax_xg_contra_por_corner, team_stats, equipo_seleccionado)
        
        # Gráfico 3: Scatter plot (derecha)
        ax_scatter = fig.add_subplot(gs[1, 2])
        ax_scatter.set_facecolor('none')
        ax_scatter.set_title('Suma de xG en contra de córner', fontsize=14, weight='bold', 
                            color='#1e3d59', pad=15)
        self.plot_xg_contra_scatter(ax_scatter, team_stats, equipo_seleccionado)
        
        return fig
    
    def plot_xg_contra_total(self, ax, team_stats, equipo_seleccionado):
        """Dibuja el gráfico de xG total en contra por córners"""
        sorted_data = team_stats.sort_values('xg_corners_contra_total', ascending=True)
        
        colors = []
        for team in sorted_data['Team Name']:
            if team == equipo_seleccionado:
                colors.append('#e74c3c')
            elif 'villarreal' in team.lower():
                colors.append('#f39c12')
            else:
                colors.append('#95a5a6')
        
        bars = ax.barh(range(len(sorted_data)), sorted_data['xg_corners_contra_total'], 
                    height=0.6, color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=0.5)
        
        # Añadir valores para equipos destacados
        for i, (bar, valor, team) in enumerate(zip(bars, sorted_data['xg_corners_contra_total'], sorted_data['Team Name'])):
            if team == equipo_seleccionado or 'villarreal' in team.lower():
                ax.text(valor + (valor * 0.02), i, f'{float(valor):.3f}', va='center', fontsize=9, weight='bold')
        
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data['Team Name'], fontsize=8)
        ax.set_xlabel('xG Total en Contra de Córner', fontsize=10, weight='bold')
        
        promedio = sorted_data['xg_corners_contra_total'].mean()
        ax.axvline(promedio, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
    
    def plot_xg_contra_por_corner(self, ax, team_stats, equipo_seleccionado):
        """Dibuja el gráfico de xG en contra por córner"""
        sorted_data = team_stats.sort_values('xg_contra_por_corner', ascending=True)
        
        colors = []
        for team in sorted_data['Team Name']:
            if team == equipo_seleccionado:
                colors.append('#e74c3c')
            elif 'villarreal' in team.lower():
                colors.append('#f39c12')
            else:
                colors.append('#95a5a6')
        
        bars = ax.barh(range(len(sorted_data)), sorted_data['xg_contra_por_corner'], 
                    color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=0.5)
        
        for i, (bar, valor, team) in enumerate(zip(bars, sorted_data['xg_contra_por_corner'], sorted_data['Team Name'])):
            if team == equipo_seleccionado or 'villarreal' in team.lower():
                ax.text(valor + (valor * 0.02), i, f'{float(valor):.3f}', va='center', fontsize=9, weight='bold')
        
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data['Team Name'], fontsize=8)
        ax.set_xlabel('xG en Contra por Córner', fontsize=10, weight='bold')
        
        promedio = sorted_data['xg_contra_por_corner'].mean()
        ax.axvline(promedio, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
    
    def plot_xg_contra_scatter(self, ax, team_stats, equipo_seleccionado):
        """Dibuja el scatter plot: xG total en contra vs xG en contra por córner"""
        x_data = team_stats['xg_corners_contra_total']
        y_data = team_stats['xg_contra_por_corner']
        equipos = team_stats['Team Name']
        
        # Calcular promedios para las líneas de la cruz
        x_promedio = x_data.mean()
        y_promedio = y_data.mean()
        
        ax.axvline(x_promedio, color='#34495e', linestyle='-', linewidth=2, alpha=0.6)
        ax.axhline(y_promedio, color='#34495e', linestyle='-', linewidth=2, alpha=0.6)
        
        for i, (equipo, x_val, y_val) in enumerate(zip(equipos, x_data, y_data)):
            is_selected = equipo == equipo_seleccionado
            is_villarreal = 'villarreal' in equipo.lower()
            
            escudo = self.find_team_logo_by_similarity(equipo)
            
            if escudo is not None:
                try:
                    if is_selected or is_villarreal:
                        zoom_size = 0.45
                        imagebox = OffsetImage(escudo, zoom=zoom_size, alpha=1.0)
                    else:
                        escudo_bn = self.convert_to_grayscale(escudo)
                        zoom_size = 0.24
                        imagebox = OffsetImage(escudo_bn, zoom=zoom_size, alpha=0.8)
                    
                    ab = AnnotationBbox(imagebox, (x_val, y_val), frameon=False, pad=0)
                    ax.add_artist(ab)
                    continue
                except Exception as e:
                    print(f"❌ Error al mostrar escudo para {equipo}: {e}")
            
            # Si no hay escudo, mostrar nombre del equipo
            if is_selected or is_villarreal:
                color, fontsize, weight = '#e74c3c' if is_selected else '#f39c12', 8, 'bold'
            else:
                color, fontsize, weight = '#2c3e50', 7, 'normal'
            
            ax.text(x_val, y_val, equipo, ha='center', va='center', 
                    fontsize=fontsize, weight=weight, color=color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.8))
        
        ax.set_xlabel('Suma de xG en Contra de Córner', fontsize=12, weight='bold')
        ax.set_ylabel('xG en Contra por Córner', fontsize=12, weight='bold')
        
        # Asegurar que todos los puntos sean visibles
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()
        
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def verificar_datos_xg_defensivos_disponibles():
    """Función para verificar qué datos están disponibles para xG defensivos"""
    paths_to_check = {
        'match_events': "./extraccion_opta/datos_opta_parquet/abp_events.parquet",
        'xg_events': "./extraccion_opta/datos_opta_parquet/xg_events.parquet", 
        'team_stats': "./extraccion_opta/datos_opta_parquet/team_stats.parquet"
    }
    
    
    available_data = {}
    for name, path in paths_to_check.items():
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                available_data[name] = df
                
                if name == 'xg_events':
                    if 'qualifier 321' in df.columns:
                        pass
                        non_null_xg = df['qualifier 321'].notna().sum()
                    else:
                        print(f"   ❌ Columna 'qualifier 321' NO encontrada")
                        
            except Exception as e:
                print(f"❌ Error al cargar {name}: {e}")
        else:
            print(f"❌ {name}: Archivo no encontrado en {path}")
    
    return available_data

def seleccionar_equipo_interactivo_defensivo():
    """Selección interactiva de equipo para xG defensivo"""
    try:
        df = pd.read_parquet("./extraccion_opta/datos_opta_parquet/abp_events.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        
        if not equipos:
            pass
            return None
        
        for i, equipo in enumerate(equipos, 1):
            pass
        
        while True:
            try:
                seleccion = input(f"\nSelecciona un equipo (1-{len(equipos)}): ").strip()
                indice = int(seleccion) - 1
                if 0 <= indice < len(equipos):
                    return equipos[indice]
                else:
                    pass
            except ValueError:
                pass
                
    except Exception as e:
        pass
        return None

def main():
    """Función principal para ejecutar el reporte de xG córners defensivos"""
    
    # Verificar datos disponibles
    available_data = verificar_datos_xg_defensivos_disponibles()
    if not all(key in available_data for key in ['match_events', 'xg_events', 'team_stats']):
        print("❌ No se pueden generar reportes: faltan archivos necesarios")
        return
    
    # Seleccionar equipo
    equipo_seleccionado = seleccionar_equipo_interactivo_defensivo()
    if not equipo_seleccionado:
        return
    
    
    # Crear reporte
    try:
        report_generator = XGCornersDefensivosReport()
        
        if report_generator.corner_xg_defensivo_data is None or report_generator.corner_xg_defensivo_data.empty:
            print("❌ No se pudieron extraer datos de xG de córners defensivos")
            return
            
        fig = report_generator.create_xg_defensivo_visualization(equipo_seleccionado)
        
        if fig:
            plt.show()
            
            # --- CAMBIO CLAVE ---
            # Guardar como PDF usando la nueva función para formato A4 horizontal
            equipo_filename = equipo_seleccionado.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_xg_corners_defensivos_{equipo_filename}.pdf"
            report_generator.guardar_sin_espacios(fig, output_path)
            
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error al generar el reporte: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()