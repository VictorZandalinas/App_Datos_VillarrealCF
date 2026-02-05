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

class XGFaltasIndirectasReport:
    def __init__(self, 
                 match_events_path="./extraccion_opta/datos_opta_parquet/abp_events.parquet",
                 xg_events_path="./extraccion_opta/datos_opta_parquet/xg_events.parquet",
                 team_stats_path="./extraccion_opta/datos_opta_parquet/team_stats.parquet"):
        """
        Inicializa la clase para generar informes de xG de faltas indirectas
        """
        self.match_events_path = match_events_path
        self.xg_events_path = xg_events_path
        self.team_stats_path = team_stats_path
        self.match_events_df = None
        self.xg_events_df = None
        self.team_stats_df = None
        self.falta_xg_data = None
        self.sequences_data = None
        self.load_data()
        self.extract_falta_xg_sequences()
    
    def export_sequences_to_csv(self, filename="faltas_indirectas_sequences.csv"):
        """Exporta las secuencias completas a CSV"""
        if self.sequences_data is not None and not self.sequences_data.empty:
            self.sequences_data.to_csv(filename, index=False, encoding='utf-8')
        else:
            print("❌ No hay secuencias para exportar")
    
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
        
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1]
        
        try:
            dt = pd.to_datetime(timestamp_str)
            return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        except:
            return timestamp_str
        
    def load_data(self):
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
            
            # Team stats (opcional)
            if os.path.exists(self.team_stats_path):
                self.team_stats_df = pd.read_parquet(self.team_stats_path)
            
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")
    
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
    
    def extract_falta_xg_sequences(self):
        """Extrae secuencias de faltas indirectas usando el método de ABP1 y relaciona con xG"""
        if any(df is None for df in [self.match_events_df, self.xg_events_df]):
            print("❌ No se pueden procesar datos: archivos faltantes")
            return
        
        
        # Obtener todos los equipos
        teams = self.match_events_df['Team Name'].dropna().unique()
        falta_xg_list = []
        sequences_list = []
        
        for team in teams:
            team_matches = self.match_events_df[self.match_events_df['Team Name'] == team]['Match ID'].unique()
            
            # Usar el método de ABP1 para obtener secuencias
            fk_sequences = self.get_freekick_indirect_sequences(team, team_matches)
            
            for seq in fk_sequences:
                match_id = seq['freekick_event']['Match ID']
                team_id = seq['freekick_event']['Team ID']
                
                # Agregar evento de falta como primer evento de la secuencia
                falta_row = seq['freekick_event'].copy()
                falta_row['sequence_id'] = f"{match_id}_{team_id}_{seq['freekick_event']['timeMin']}_{seq['freekick_event']['timeSec']}"
                falta_row['event_order'] = 0
                falta_row['time_from_falta'] = 0
                sequences_list.append(falta_row)
                
                # Agregar eventos de la secuencia
                for order, event in enumerate(seq['sequence_events'], 1):
                    event_copy = event.copy()
                    event_copy['sequence_id'] = falta_row['sequence_id']
                    event_copy['event_order'] = order
                    event_copy['time_from_falta'] = (event['timeMin'] * 60 + event['timeSec']) - (seq['freekick_event']['timeMin'] * 60 + seq['freekick_event']['timeSec'])
                    sequences_list.append(event_copy)
                
                # Calcular xG para eventos de tiro en la secuencia
                total_xg = 0
                num_shots = len(seq['shot_events'])
                shot_details = []
                
                if seq['shot_events']:
                    xg_values = self.get_xg_for_shot_events(seq['shot_events'])
                    total_xg = sum(xg_values)
                    
                    for shot_event, xg_value in zip(seq['shot_events'], xg_values):
                        shot_details.append({
                            'event_type': shot_event['Event Name'],
                            'player': shot_event.get('playerName', ''),
                            'xg': xg_value,
                            'time': f"{shot_event['timeMin']}:{shot_event['timeSec']:02d}"
                        })
                
                # Solo agregar si hay algo relevante
                if total_xg > 0 or num_shots > 0 or len(seq['sequence_events']) > 0:
                    falta_data = {
                        'Match ID': match_id,
                        'Team ID': team_id,
                        'Team Name': team,
                        'falta_time': f"{seq['freekick_event']['timeMin']}:{seq['freekick_event']['timeSec']:02d}",
                        'total_xg': total_xg,
                        'num_shots': num_shots,
                        'shot_details': shot_details,
                        'Week': seq['freekick_event'].get('Week', 'N/A')
                    }
                    falta_xg_list.append(falta_data)
        
        # Crear DataFrames
        if falta_xg_list:
            self.falta_xg_data = pd.DataFrame(falta_xg_list)
        else:
            print("❌ No se encontraron secuencias de faltas indirectas con xG")
            self.falta_xg_data = pd.DataFrame()
        
        if sequences_list:
            self.sequences_data = pd.DataFrame(sequences_list)
        else:
            self.sequences_data = pd.DataFrame()  
    
    def get_team_xg_falta_stats(self, equipo_seleccionado=None):
        """Calcula estadísticas de xG por faltas indirectas por equipo"""
        if self.falta_xg_data is None or self.falta_xg_data.empty:
            print("❌ No hay datos de xG de faltas indirectas para procesar")
            return None, None
        
        # Agrupar por equipo
        team_stats = self.falta_xg_data.groupby('Team Name').agg({
            'total_xg': 'sum',
            'num_shots': 'sum',
            'Match ID': 'nunique'  # Número de jornadas
        }).reset_index()
        
        team_stats.columns = ['Team Name', 'xg_faltas_total', 'shots_from_faltas', 'num_jornadas']
        
        # Calcular métricas derivadas
        team_stats['xg_faltas_por_jornada'] = team_stats['xg_faltas_total'] / team_stats['num_jornadas']
        
        # Contar total de faltas indirectas por equipo desde los datos originales
        if self.match_events_df is not None:
            faltas_count = self.match_events_df[
                (self.match_events_df['Event Name'] == 'Pass') & 
                (self.match_events_df['Free kick taken'] == 'Sí') &
                (self.match_events_df['x'].fillna(0) > 50)  # Solo faltas ofensivas
            ].groupby('Team Name').size().reset_index(name='total_faltas_indirectas')  
            
            team_stats = team_stats.merge(faltas_count, on='Team Name', how='left')
            team_stats['total_faltas_indirectas'] = team_stats['total_faltas_indirectas'].fillna(1)
            team_stats['xg_por_falta'] = team_stats['xg_faltas_total'] / team_stats['total_faltas_indirectas']
        else:
            team_stats['total_faltas_indirectas'] = 1
            team_stats['xg_por_falta'] = team_stats['xg_faltas_total']
        
        # Rellenar valores faltantes
        team_stats = team_stats.fillna(0)
        
        
        if equipo_seleccionado:
            equipo_data = team_stats[team_stats['Team Name'] == equipo_seleccionado]
            if not equipo_data.empty:
                return team_stats, equipo_data.iloc[0]
            else:
                print(f"❌ No se encontraron datos para {equipo_seleccionado}")
                return team_stats, None
        
        return team_stats, None
    
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
    
    def create_xg_visualization(self, equipo_seleccionado, figsize=(11.69, 8.27)):
        """Crea la visualización completa del informe de xG de faltas indirectas"""
        
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
        
        # Obtener datos de xG
        team_stats, equipo_data = self.get_team_xg_falta_stats(equipo_seleccionado)
        if team_stats is None:
            pass
            return None
        
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
        
        # Configurar grid - Ajustado para A4 horizontal
        gs = fig.add_gridspec(2, 4, 
                height_ratios=[0.2, 1],  # Más espacio para el título
                width_ratios=[1, 1, 1.4, 0.1],  # Mejor distribución en A4
                hspace=0.25, wspace=0.45,
                left=0.10, right=0.99, top=0.92, bottom=0.08)
        
        # Área del título
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        # Título principal
        ax_title.text(0.5, 0.6, 'XG FALTAS OFENSIVAS', 
                    fontsize=24, weight='bold', ha='center', va='center',  # Reducido de 28
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
        
        # Gráfico 1: xG de falta por jornada (izquierda)
        ax_xg_jornada = fig.add_subplot(gs[1, 0])
        ax_xg_jornada.set_facecolor('none')
        ax_xg_jornada.set_title('xG a favor de falta indirecta', fontsize=12, weight='bold',  # Reducido de 14
                color='#1e3d59', pad=15)
        self.plot_xg_por_jornada(ax_xg_jornada, team_stats, equipo_seleccionado)
        
        # Gráfico 2: xG por falta (eficiencia) (centro)
        ax_xg_eficiencia = fig.add_subplot(gs[1, 1])
        ax_xg_eficiencia.set_facecolor('none')
        ax_xg_eficiencia.set_title('xG a favor por falta indirecta', fontsize=12, weight='bold',  # Reducido de 14
                            color='#1e3d59', pad=15)
        self.plot_xg_por_falta(ax_xg_eficiencia, team_stats, equipo_seleccionado)
        
        # Gráfico 3: Scatter plot (derecha)
        ax_scatter = fig.add_subplot(gs[1, 2])
        ax_scatter.set_facecolor('none')
        ax_scatter.set_title('Suma de xG a favor de falta indirecta', fontsize=12, weight='bold',  # Reducido de 14
                            color='#1e3d59', pad=15)
        self.plot_xg_scatter(ax_scatter, team_stats, equipo_seleccionado)
        
        return fig
    
    def plot_xg_por_jornada(self, ax, team_stats, equipo_seleccionado):
        """Dibuja el gráfico de suma de xG en faltas indirectas"""
        sorted_data = team_stats.sort_values('xg_faltas_total', ascending=True)
        
        colors = []
        for team in sorted_data['Team Name']:
            if team == equipo_seleccionado:
                colors.append('#e74c3c')
            elif 'villarreal' in team.lower():
                colors.append('#f39c12')
            else:
                colors.append('#95a5a6')
        
        bars = ax.barh(range(len(sorted_data)), sorted_data['xg_faltas_total'], 
                    height=0.6, color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=0.5)
        
        # Añadir valores para equipos destacados
        for i, (bar, valor, team) in enumerate(zip(bars, sorted_data['xg_faltas_total'], sorted_data['Team Name'])):
            if team == equipo_seleccionado or 'villarreal' in team.lower():
                ax.text(valor + (valor * 0.02), i, f'{float(valor):.3f}', va='center', fontsize=9, weight='bold')
        
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data['Team Name'], fontsize=8)
        ax.set_xlabel('Suma de xG en Faltas Indirectas', fontsize=10, weight='bold')
        
        promedio = sorted_data['xg_faltas_total'].mean()
        ax.axvline(promedio, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def plot_xg_por_falta(self, ax, team_stats, equipo_seleccionado):
        """Dibuja el gráfico de xG por falta indirecta"""
        sorted_data = team_stats.sort_values('xg_por_falta', ascending=True)
        
        colors = []
        for team in sorted_data['Team Name']:
            if team == equipo_seleccionado:
                colors.append('#e74c3c')
            elif 'villarreal' in team.lower():
                colors.append('#f39c12')
            else:
                colors.append('#95a5a6')
        
        bars = ax.barh(range(len(sorted_data)), sorted_data['xg_por_falta'], 
                    color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=0.5)
        
        for i, (bar, valor, team) in enumerate(zip(bars, sorted_data['xg_por_falta'], sorted_data['Team Name'])):
            if team == equipo_seleccionado or 'villarreal' in team.lower():
                ax.text(valor + (valor * 0.02), i, f'{float(valor):.3f}', va='center', fontsize=9, weight='bold')
        
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data['Team Name'], fontsize=8)
        ax.set_xlabel('xG por Falta Indirecta', fontsize=10, weight='bold')
        
        promedio = sorted_data['xg_por_falta'].mean()
        ax.axvline(promedio, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def plot_xg_scatter(self, ax, team_stats, equipo_seleccionado):
        """Dibuja el scatter plot: xG total vs xG por falta indirecta"""
        x_data = team_stats['xg_faltas_total']
        y_data = team_stats['xg_por_falta']
        equipos = team_stats['Team Name']
        
        # Calcular promedios para las líneas de la cruz
        x_promedio = x_data.mean()
        y_promedio = y_data.mean()
        
        ax.axvline(x_promedio, color='#34495e', linestyle='-', linewidth=2, alpha=0.6)
        ax.axhline(y_promedio, color='#34495e', linestyle='-', linewidth=2, alpha=0.6)
        
        # PASADA 1: Equipos normales (z-order bajo)
        for i, (equipo, x_val, y_val) in enumerate(zip(equipos, x_data, y_data)):
            is_selected = equipo == equipo_seleccionado
            is_villarreal = 'villarreal' in equipo.lower()
            
            # Solo procesar equipos normales en esta pasada
            if is_selected or is_villarreal:
                continue
                
            
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
        
        ax.set_xlabel('Suma de xG a Favor de Falta Indirecta', fontsize=12, weight='bold')
        ax.set_ylabel('xG por Falta Indirecta', fontsize=12, weight='bold')
        
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


def verificar_datos_xg_faltas_disponibles():
    """Función para verificar qué datos están disponibles para xG de faltas indirectas"""
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
                
                if name == 'match_events':
                    if 'Free kick taken' in df.columns:
                        pass
                        faltas_si = (df['Free kick taken'] == 'Sí').sum()
                    else:
                        print(f"   ❌ Columna 'Free kick taken' NO encontrada")
                        
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

def seleccionar_equipo_interactivo_faltas():
    """Selección interactiva de equipo para xG de faltas indirectas"""
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
    """Función principal para ejecutar el reporte de xG faltas indirectas"""
    
    # Verificar datos disponibles
    available_data = verificar_datos_xg_faltas_disponibles()
    if not all(key in available_data for key in ['match_events', 'xg_events']):
        print("❌ No se pueden generar reportes: faltan archivos necesarios")
        return
    
    # Seleccionar equipo
    equipo_seleccionado = seleccionar_equipo_interactivo_faltas()
    if not equipo_seleccionado:
        return
    
    
    # Crear reporte
    try:
        report_generator = XGFaltasIndirectasReport()
        
        if report_generator.falta_xg_data is None or report_generator.falta_xg_data.empty:
            print("❌ No se pudieron extraer datos de xG de faltas indirectas")
            return
            
        fig = report_generator.create_xg_visualization(equipo_seleccionado)
        
        if fig:
            plt.show()
            
            # Guardar como PDF en formato A4
            equipo_filename = equipo_seleccionado.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_xg_faltas_indirectas_{equipo_filename}.pdf"
            report_generator.guardar_sin_espacios(fig, output_path)
            
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error al generar el reporte: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()