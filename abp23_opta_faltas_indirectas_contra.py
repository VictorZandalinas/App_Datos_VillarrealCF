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

class FaltasDefensivasReport:
    def __init__(self, data_path="extraccion_opta/datos_opta_parquet/abp_events.parquet"):
        self.data_path = data_path
        self.team_stats_path = "extraccion_opta/datos_opta_parquet/team_stats.parquet"
        self.df_events = None  # Para eventos
        self.df_teams = None   # Para equipos
        self.freekick_sequences = None  # Para secuencias defensivas de faltas indirectas
        
        self.load_data()
        if self.df_events is not None:
            self.extract_all_freekick_indirect_sequences_defensivo()
    
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
        
    def load_data(self):
        try:
            self.df_events = pd.read_parquet(self.data_path)
            self.df_teams = pd.read_parquet(self.team_stats_path)
            
            if 'timeStamp' in self.df_events.columns:
                self.df_events['timeStamp'] = pd.to_datetime(self.df_events['timeStamp'].str.replace('Z', ''), errors='coerce')
            
        except Exception as e:
            print(f"❌ Error al cargar los datos de Opta: {e}")
    
    def extract_all_freekick_indirect_sequences_defensivo(self):
        """Extrae faltas indirectas CONTRA todos los equipos"""
        df_sorted = self.df_events.sort_values(['Match ID', 'periodId', 'timeStamp']).reset_index(drop=True)
        lanzamientos_list = []
        
        for match_id, match_events in df_sorted.groupby('Match ID'):
            # Obtener los dos equipos del partido
            equipos_partido = match_events['Team Name'].unique()
            
            # Faltas indirectas de TODOS los equipos en este partido
            all_freekicks_indirect = match_events[
                (match_events.get('Free kick taken', '') == 'Sí') &
                (match_events['Zone'].isin(['Center', 'Right', 'Left'])) &
                (match_events['x'].notna()) & 
                (match_events['y'].notna())
            ].copy()
            
            for _, lanzamiento in all_freekicks_indirect.iterrows():
                # Identificar el equipo que DEFIENDE la falta indirecta (el que NO lanza)
                equipo_lanzador = lanzamiento['Team Name']
                equipo_defensor = [eq for eq in equipos_partido if eq != equipo_lanzador][0]
                
                # Analizar la secuencia (misma función que corners)
                result_data = self.analyze_lanzamiento_sequence(df_sorted, lanzamiento.name, lanzamiento)
                
                # Guardar desde la perspectiva DEFENSIVA
                lanzamiento_data = {
                    'Match ID': match_id,
                    'Team ID': lanzamiento['Team ID'],  # Equipo que lanza
                    'Team Name': equipo_defensor,       # ● CAMBIO CLAVE: Equipo que defiende
                    'playerName': lanzamiento['playerName'],
                    'rival_team': equipo_lanzador       # Equipo rival que lanza
                }
                
                lanzamiento_data.update(result_data)
                lanzamientos_list.append(lanzamiento_data)
        
        if lanzamientos_list:
            self.freekick_sequences = pd.DataFrame(lanzamientos_list)
    
    def analyze_lanzamiento_sequence(self, match_events, lanzamiento_idx, lanzamiento_pass):
        """
        Analiza la secuencia post-lanzamiento con la lógica de tiempo y segunda jugada
        corregida y replicada exactamente desde ABP2.
        """
        start_time_min = lanzamiento_pass['timeMin']
        start_time_sec = lanzamiento_pass['timeSec']
        
        events_found = {
            'Goal': None, 'Post': None, 'Attempt Saved': None,
            'Miss': None, 'Otro contacto': None
        }
        is_second_play = False
        previous_event_coords = None
        result_event_idx = None
        pass_count = 0  # Contador de pases
        
        # ▼▼▼ AÑADE ESTA LÍNEA ▼▼▼
        # Guarda el timeStamp del último pase contado. Empezamos con el del lanzamiento.
        last_pass_timestamp = pd.to_datetime(lanzamiento_pass['timeStamp'])
        
        # Usamos el tiempo del evento previo para calcular la diferencia
        prev_event_time = lanzamiento_pass['timeMin'] * 60 + lanzamiento_pass['timeSec']

        for next_idx in range(lanzamiento_idx + 1, len(match_events)):

            next_event = match_events.iloc[next_idx]
            
            if (next_event['Event Name'] in ['Corner Awarded', 'Foul', 'Offside', 'End Period', 'Out'] or
                next_event.get('Throw in', '') == 'Sí' or
                next_event.get('Free kick taken', '') == 'Sí'):
                break
                
            next_event_time = next_event['timeMin'] * 60 + next_event['timeSec']
            time_diff = next_event_time - prev_event_time
            
            if time_diff > 5:
                break

            prev_event_time = next_event_time
            
            event_name = next_event['Event Name']
            event_team_id = next_event['Team ID']
            lanzamiento_team_id = lanzamiento_pass['Team ID']
            
            # ↑ NUEVO: Contar pases del mismo equipo
            if event_name == 'Pass' and event_team_id == lanzamiento_team_id:
                next_timestamp = pd.to_datetime(next_event['timeStamp'])
                # Solo contamos el pase si su timestamp es posterior al del último pase contado.
                if next_timestamp > last_pass_timestamp:
                    pass_count += 1
                    last_pass_timestamp = next_timestamp  # Actualizamos el timestamp.
                    
                    # Si llegamos a 5 o más pases, verificar la nueva condición
                    if pass_count >= 5:
                        # Contar pases con x < 70 en los últimos eventos
                        passes_back_field = 0
                        for check_idx in range(lanzamiento_idx + 1, next_idx + 1):
                            check_event = match_events.iloc[check_idx]
                            if (check_event['Event Name'] == 'Pass' and 
                                check_event['Team ID'] == lanzamiento_team_id and
                                float(check_event.get('x', 0)) < 70):
                                passes_back_field += 1
                        
                        # Si hay 2 o más pases con x < 70, cortar la secuencia
                        if passes_back_field >= 2:
                            break
            
            if event_name in ['Goal', 'Post', 'Attempt Saved', 'Miss'] and events_found[event_name] is None:
                if event_team_id == lanzamiento_team_id:
                    # ↑ NUEVO: Para goles, verificar que no sea el mismo jugador
                    if event_name == 'Goal' and next_event.get('playerName') == lanzamiento_pass.get('playerName'):
                        continue  # Saltar goles del mismo lanzador
                    
                    events_found[event_name] = next_event
                    result_event_idx = next_idx
                    if event_name == 'Goal':
                        break
            
            elif (event_name == 'Pass' and 
                event_team_id == lanzamiento_team_id and
                next_event.get('outcome') == 1 and
                # Añadimos la lógica estricta: debe haber un segundo pase consecutivo
                next_idx + 1 < len(match_events) and 
                match_events.iloc[next_idx + 1]['Event Name'] == 'Pass' and 
                match_events.iloc[next_idx + 1]['Team ID'] == lanzamiento_team_id and
                events_found['Otro contacto'] is None):
                events_found['Otro contacto'] = next_event

        # --- LÓGICA DE SEGUNDA JUGADA (COPIA EXACTA Y CORREGIDA DE ABP2) ---
        if result_event_idx is not None and result_event_idx > lanzamiento_idx + 1:
            previous_event_found = None
            # Buscar hacia atrás desde el remate hasta justo después de la falta indirecta
            for search_idx in range(result_event_idx - 1, lanzamiento_idx, -1):
                candidate_event = match_events.iloc[search_idx]
                
                # CORRECCIÓN 1: La condición es x > 55, no 60.
                # CORRECCIÓN 2: Se añade la comprobación crucial del timeStamp.
                if (float(candidate_event.get('x', 0)) > 55 and
                        candidate_event['Event Name'] != 'Deleted event' and
                        # Esta línea es la clave: asegura que no es el mismo evento de la falta
                        candidate_event['timeStamp'] != lanzamiento_pass['timeStamp']):
                    
                    previous_event_found = candidate_event
                    break # Encontramos el evento relevante, salimos del bucle.
            
            # CORRECCIÓN 3: La lógica es más explícita. Si se encontró un evento, es segunda jugada.
            if previous_event_found is not None:
                is_second_play = True
                prev_x = float(previous_event_found.get('x', 0))
                prev_y = float(previous_event_found.get('y', 0))
                previous_event_coords = (prev_x, prev_y)
            else:
                is_second_play = False
                
        # --- Determinar resultado final (sin cambios en esta parte) ---
        # (El resto de la función para devolver el diccionario de resultados es idéntico y correcto)
        if events_found['Goal'] is not None:
            event = events_found['Goal']
            return {
                'result_type': 'Gol', 'final_x': float(event.get('x', 0)), 'final_y': float(event.get('y', 0)),
                'goal_player': event.get('playerName', ''), 'goal_player_id': event.get('playerId'),
                'is_second_play': is_second_play, 'previous_event_coords': previous_event_coords
            }
        elif events_found['Attempt Saved'] is not None:
            event = events_found['Attempt Saved']
            return {
                'result_type': 'Tiro a puerta', 'final_x': float(event.get('x', 0)), 'final_y': float(event.get('y', 0)),
                'goal_player': event.get('playerName', ''), 'goal_player_id': event.get('playerId'),
                'is_second_play': is_second_play, 'previous_event_coords': previous_event_coords
            }
        elif events_found['Post'] is not None:
            event = events_found['Post']
            return {
                'result_type': 'Tiro al poste', 'final_x': float(event.get('x', 0)), 'final_y': float(event.get('y', 0)),
                'goal_player': event.get('playerName', ''), 'goal_player_id': event.get('playerId'),
                'is_second_play': is_second_play, 'previous_event_coords': previous_event_coords
            }
        elif events_found['Miss'] is not None:
            event = events_found['Miss']
            return {
                'result_type': 'Tiro fuera', 'final_x': float(event.get('x', 0)), 'final_y': float(event.get('y', 0)),
                'goal_player': event.get('playerName', ''), 'goal_player_id': event.get('playerId'),
                'is_second_play': is_second_play, 'previous_event_coords': previous_event_coords
            }
        elif events_found['Otro contacto'] is not None:
            event = events_found['Otro contacto']
            return {
                'result_type': 'Otro contacto', 'final_x': float(event.get('Pass End X', 0)), 'final_y': float(event.get('Pass End Y', 0)),
                'goal_player': None, 'goal_player_id': None,
                'is_second_play': False, 'previous_event_coords': None
            }
        else:
            return {
                'result_type': 'Sin remate', 'final_x': float(lanzamiento_pass.get('Pass End X', 0)), 'final_y': float(lanzamiento_pass.get('Pass End Y', 0)),
                'goal_player': None, 'goal_player_id': None,
                'is_second_play': False, 'previous_event_coords': None
            }
    
    def get_team_freekick_indirect_stats(self, equipo_seleccionado=None):
        """Estadísticas DEFENSIVAS de faltas indirectas"""
        if self.freekick_sequences is None:
            return None, None
        
        # Filtrar por liga
        liga_seleccionada = self.df_teams[self.df_teams['Team Name'] == equipo_seleccionado]['Competition Name'].iloc[0]
        equipos_liga = self.df_teams[self.df_teams['Competition Name'] == liga_seleccionada]['Team Name'].unique()
        
        df_filtrado = self.freekick_sequences[self.freekick_sequences['Team Name'].isin(equipos_liga)].copy()
        
        # Agrupar por equipo DEFENSOR
        stats_agrupadas = df_filtrado.groupby('Team Name').agg(
            num_faltas_indirectas_contra=('Team Name', 'size'),           
            goles_concedidos_falta_indirecta=('result_type', lambda x: (x == 'Gol').sum()),
            tiros_concedidos_falta_indirecta=('result_type', lambda x: (x.isin(['Gol', 'Tiro a puerta', 'Tiro al poste', 'Tiro fuera'])).sum())
        ).reset_index()
        
        # AÑADIR ESTAS LÍNEAS:
        # Calcular número de jornadas por equipo (aproximación)
        jornadas_por_equipo = df_filtrado.groupby('Team Name')['Match ID'].nunique().reset_index()
        jornadas_por_equipo.columns = ['Team Name', 'num_jornadas']
        
        # Fusionar con stats
        stats_agrupadas = stats_agrupadas.merge(jornadas_por_equipo, on='Team Name')
        
        # Calcular métricas que necesitan los gráficos
        stats_agrupadas['faltas_indirectas_contra_por_jornada'] = stats_agrupadas['num_faltas_indirectas_contra'] / stats_agrupadas['num_jornadas'].replace(0, 1)
        stats_agrupadas['goles_concedidos_por_falta_indirecta'] = stats_agrupadas['goles_concedidos_falta_indirecta'] / stats_agrupadas['num_faltas_indirectas_contra'].replace(0, 1)
        stats_agrupadas['tiros_por_falta_indirecta'] = stats_agrupadas['tiros_concedidos_falta_indirecta'] / stats_agrupadas['num_faltas_indirectas_contra'].replace(0, 1)
        stats_agrupadas['lostFreeKicksIndirect'] = stats_agrupadas['num_faltas_indirectas_contra']  # Alias para compatibilidad
        
        # Renombrar columna
        stats_agrupadas.rename(columns={'Team Name': 'EQUIPO'}, inplace=True)
        
        if equipo_seleccionado:
            equipo_data = stats_agrupadas[stats_agrupadas['EQUIPO'] == equipo_seleccionado]
            if not equipo_data.empty:
                return stats_agrupadas, equipo_data.iloc[0]
        
        return stats_agrupadas, None
    
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
    
    def create_visualization(self, equipo_seleccionado, figsize=(11.69, 8.27)):
        """Crea la visualización completa del informe de faltas indirectas defensivas"""
        
        # Obtener datos de faltas indirectas defensivas
        team_stats, equipo_data = self.get_team_freekick_indirect_stats(equipo_seleccionado)
        if team_stats is None:
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
                width_ratios=[1.0, 1, 1.6, 0.05],  # ↑ VALORES DE ABP3
                hspace=0.35, wspace=0.65,            # ↑ VALORES DE ABP3
                left=0.11, right=0.95, top=0.95, bottom=0.05)
        
        # Área del título
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        # Título principal - CAMBIO PRINCIPAL: DEFENSIVOS
        ax_title.text(0.5, 0.6, 'FALTAS DEFENSIVAS', 
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
        escudo_zoom = 1.0

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
        
        # Gráfico 1: Goles en contra de falta indirecta (izquierda)
        ax_faltas = fig.add_subplot(gs[1, 0])
        ax_faltas.set_facecolor('none')
        ax_faltas.set_title('Goles en contra\nde falta indirecta', fontsize=12, weight='bold', 
                          color='#1e3d59', pad=15)
        self.plot_goles_contra_falta_indirecta(ax_faltas, team_stats, equipo_seleccionado)
        
        # Gráfico 2: Goles concedidos por falta indirecta (centro)
        ax_goles_concedidos = fig.add_subplot(gs[1, 1])
        ax_goles_concedidos.set_facecolor('none')
        ax_goles_concedidos.set_title('Goles en contra\npor falta indirecta', fontsize=12, weight='bold', 
                               color='#1e3d59', pad=15)
        self.plot_goles_concedidos_por_falta_indirecta(ax_goles_concedidos, team_stats, equipo_seleccionado)  

        
        # Gráfico 3: Scatter plot (derecha)
        ax_scatter = fig.add_subplot(gs[1, 2])
        ax_scatter.set_facecolor('none')
        ax_scatter.set_title('Suma de faltas\nindirectas en contra', fontsize=12, weight='bold', 
                            color='#1e3d59', pad=15)
        self.plot_falta_indirecta_scatter_defensivo(ax_scatter, team_stats, equipo_seleccionado)
        
        return fig
    
    def plot_goles_contra_falta_indirecta(self, ax, team_stats, equipo_seleccionado):
        """Dibuja el gráfico de goles en contra de falta indirecta"""
        # Ordenar por goles_concedidos_falta_indirecta (INVERTIDO: menos es mejor)
        sorted_data = team_stats.sort_values('goles_concedidos_falta_indirecta', ascending=False)
        
        # Colores: destacar equipo seleccionado y Villarreal
        colors = []
        for team in sorted_data['EQUIPO']:
            if team == equipo_seleccionado:
                colors.append('#e74c3c')
            elif 'villarreal' in team.lower():
                colors.append('#f39c12')  # Naranja para Villarreal
            else:
                colors.append('#95a5a6')  # Gris para otros
        
        # Crear gráfico horizontal usando goles_concedidos_falta_indirecta
        bars = ax.barh(range(len(sorted_data)), sorted_data['goles_concedidos_falta_indirecta'], 
                    height=0.6,  # Barras más estrechas
                    color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=0.5)
        
        # Añadir valores para equipos destacados
        ax.set_yticklabels(sorted_data['EQUIPO'], fontsize=8)
        for i, (bar, valor, team) in enumerate(zip(bars, sorted_data['goles_concedidos_falta_indirecta'], sorted_data['EQUIPO'])):
            if team == equipo_seleccionado or 'villarreal' in team.lower():
                ax.text((valor) + 0.02, i, f'{int(valor)}', va='center', fontsize=9, weight='bold')
        
        # Configurar ejes
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data['EQUIPO'], fontsize=8)  
        ax.set_xlabel('Goles en contra de falta indirecta', fontsize=10, weight='bold')
        
        # Línea promedio
        promedio = sorted_data['goles_concedidos_falta_indirecta'].mean()
        ax.axvline(promedio, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def plot_goles_concedidos_por_falta_indirecta(self, ax, team_stats, equipo_seleccionado):
        """Dibuja el gráfico de goles concedidos por falta indirecta"""
        # INVERTIDO: menos goles concedidos por falta indirecta es mejor
        sorted_data = team_stats.sort_values('goles_concedidos_por_falta_indirecta', ascending=False)
        
        colors = []
        for team in sorted_data['EQUIPO']:
            if team == equipo_seleccionado:
                colors.append('#e74c3c')
            elif 'villarreal' in team.lower():
                colors.append('#f39c12')
            else:
                colors.append('#95a5a6')
        
        bars = ax.barh(range(len(sorted_data)), sorted_data['goles_concedidos_por_falta_indirecta'], 
                    color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=0.5)
        
        for i, (bar, valor, team) in enumerate(zip(bars, sorted_data['goles_concedidos_por_falta_indirecta'], sorted_data['EQUIPO'])):
            if team == equipo_seleccionado or 'villarreal' in team.lower():
                ax.text((valor) + 0.002, i, f'{float(valor):.2f}', va='center', fontsize=9, weight='bold')
        
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data['EQUIPO'], fontsize=8)
        ax.set_xlabel('Goles en contra por falta indirecta', fontsize=10, weight='bold')
        
        promedio = sorted_data['goles_concedidos_por_falta_indirecta'].mean()
        ax.axvline(promedio, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def plot_falta_indirecta_scatter_defensivo(self, ax, team_stats, equipo_seleccionado):
        """Dibuja el scatter plot: Faltas indirectas en contra vs Tiros concedidos"""
        x_data = team_stats['lostFreeKicksIndirect']  # Suma de faltas indirectas en contra
        y_data = team_stats['tiros_concedidos_falta_indirecta']  # Suma de tiros concedidos por falta indirecta
        equipos = team_stats['EQUIPO']
        
        # Calcular promedios para las líneas de la cruz
        x_promedio = x_data.mean()
        y_promedio = y_data.mean()
        
        # Dibujar líneas de la cruz
        ax.axvline(x_promedio, color='#34495e', linestyle='-', linewidth=2, alpha=0.6)
        ax.axhline(y_promedio, color='#34495e', linestyle='-', linewidth=2, alpha=0.6)
        
        # TAMAÑO UNIFORME PARA TODOS LOS ESCUDOS
        uniform_zoom = 0.24
        
        # Scatter plot con escudos
        for i, (equipo, x_val, y_val) in enumerate(zip(equipos, x_data, y_data)):
            is_selected = equipo == equipo_seleccionado
            is_villarreal = 'villarreal' in equipo.lower()
            
            
            # Buscar escudo por similitud
            escudo = self.find_team_logo_by_similarity(equipo)
            
            if escudo is not None:
                pass
                try:
                    # APLICAR FILTRO DE COLOR Y TAMAÑO
                    if is_selected or is_villarreal:
                        # Mantener colores originales Y MÁS GRANDES
                        zoom_size = 0.45  # 33% más grande que 0.24
                        imagebox = OffsetImage(escudo, zoom=zoom_size, alpha=1.0)
                    else:
                        # Convertir a blanco y negro Y TAMAÑO NORMAL
                        escudo_bn = self.convert_to_grayscale(escudo)
                        zoom_size = 0.24  # Tamaño normal
                        imagebox = OffsetImage(escudo_bn, zoom=zoom_size, alpha=0.8)
                    
                    if is_selected or is_villarreal:
                        zorder_level = 10  # Nivel alto para que estén por delante
                    else:
                        zorder_level = 1   # Nivel bajo para el resto

                    ab = AnnotationBbox(imagebox, (x_val, y_val), frameon=False, pad=0, zorder=zorder_level)
                    ax.add_artist(ab)
                    continue
                except Exception as e:
                    print(f"❌ Error al mostrar escudo para {equipo}: {e}")
            else:
                print(f"❌ No se encontró escudo para {equipo}")
            
            # Si no hay escudo, mostrar nombre del equipo
            if is_selected or is_villarreal:
                color, fontsize, weight = '#e74c3c' if is_selected else '#f39c12', 8, 'bold'
            else:
                color, fontsize, weight = '#2c3e50', 7, 'normal'
            
            ax.text(x_val, y_val, equipo, ha='center', va='center', 
                    fontsize=fontsize, weight=weight, color=color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.8))
        
        ax.set_xlabel('Suma de faltas indirectas en contra', fontsize=12, weight='bold')
        ax.set_ylabel('Suma de tiros en contra por falta indirecta', fontsize=12, weight='bold')
        
        # Asegurar que todos los puntos sean visibles
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()

        # Añadir margen del 10%
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1

        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)


        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def verificar_datos_disponibles(data_path="./extraccion_opta/datos_opta_parquet/team_stats.parquet"):
    """Función para verificar qué datos están disponibles"""
    try:
        if not os.path.exists(data_path):
            print(f"❌ Error: No se encontró el archivo en la ruta: {data_path}")
            
            # Verificar rutas alternativas comunes
            alternative_paths = [
                "team_stats.parquet",
                "datos_opta_parquet/team_stats.parquet", 
                "extraccion_opta/team_stats.parquet",
                "./team_stats.parquet"
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    pass
                    data_path = alt_path
                    break
            else:
                return None
        
        df = pd.read_parquet(data_path)
        for i, col in enumerate(df.columns, 1):
            pass
        
        
        # Buscar columnas relacionadas con faltas indirectas defensivas
        defensive_related = [col for col in df.columns if 'lost' in col.lower() or 'conceded' in col.lower() or 'freekick' in col.lower() or 'indirect' in col.lower()]
        if defensive_related:
            pass
            for col in defensive_related:
                pass
        else:
            print(f"\n❌ No se encontraron columnas relacionadas con estadísticas defensivas")
        
        return df
    except Exception as e:
        print(f"❌ Error al verificar datos: {e}")
        return None

def seleccionar_equipo_interactivo(df):
    equipos = sorted(df['Team Name'].unique())
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

def main():
    """Función principal para ejecutar el reporte de faltas indirectas defensivas"""
    
    df = verificar_datos_disponibles()
    if df is None:
        return
    
    equipo_seleccionado = seleccionar_equipo_interactivo(df)
    
    report_generator = FaltasDefensivasReport()
    fig = report_generator.create_visualization(equipo_seleccionado)
    
    if fig:
        plt.show()
        
        # Guardar como PDF
        output_path = "reporte_faltas_defensivas.pdf"
        from matplotlib.backends.backend_pdf import PdfPages
        fig.set_size_inches(11.69, 8.27)  # ↑ AÑADIR ESTA LÍNEA

        with PdfPages(output_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight', pad_inches=0, dpi=300)
        
    else:
        print("❌ No se pudo generar la visualización defensiva")

if __name__ == "__main__":
    main()