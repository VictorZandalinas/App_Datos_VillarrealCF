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

class BalonParadoReportCustom:
    def __init__(self, 
                stats_path="./extraccion_opta/datos_opta_parquet/estadisticas_abp.parquet"):
        self.stats_path = stats_path
        self.combined_stats = None
        self.load_data()
        self.medias_liga = self.calcular_medias_liga()

    
    def load_data(self):
        """Carga las estadísticas ya procesadas"""
        try:
            if not os.path.exists(self.stats_path):
                print(f"❌ Error: No se encontró {self.stats_path}")
                return
            
            self.combined_stats = pd.read_parquet(self.stats_path)
            
            # ✅ AÑADIR ESTAS LÍNEAS:
            # Convertir Week a numérico para ordenamiento correcto
            self.combined_stats['Week'] = pd.to_numeric(self.combined_stats['Week'], errors='coerce')
            
            
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")
    
    
    def get_xg_by_matchday(self, equipo1, equipo2):
        """Obtiene xG de BP por jornada para ambos equipos desde estadisticas_abp"""
        
        # Filtrar datos de ambos equipos
        team1_data = self.combined_stats[self.combined_stats['Team Name'] == equipo1].copy()
        team2_data = self.combined_stats[self.combined_stats['Team Name'] == equipo2].copy()
        
        # Calcular xG total de BP (corners + faltas directas + faltas indirectas)
        team1_data['xg_bp_total'] = (
            team1_data['xg_corner_a_favor'] + 
            team1_data['xg_falta_a_favor'] + 
            team1_data['xg_falta_indirecta_a_favor']
        )
        
        team2_data['xg_bp_total'] = (
            team2_data['xg_corner_a_favor'] + 
            team2_data['xg_falta_a_favor'] + 
            team2_data['xg_falta_indirecta_a_favor']
        )
        
        # Preparar datos por jornada
        team1_weeks = team1_data[['Week', 'xg_bp_total']].rename(columns={'xg_bp_total': 'team1_xg'})
        team2_weeks = team2_data[['Week', 'xg_bp_total']].rename(columns={'xg_bp_total': 'team2_xg'})
        
        # Hacer merge para tener ambos equipos en la misma tabla
        df_jornadas = pd.merge(team1_weeks, team2_weeks, on='Week', how='outer').fillna(0)
        df_jornadas = df_jornadas.rename(columns={'Week': 'week'})
        df_jornadas = df_jornadas.sort_values('week').reset_index(drop=True)
        
        return df_jornadas
    
    def calcular_medias_liga(self):
        """Calcula las medias de todas las métricas de BP en la liga"""
        if self.combined_stats is None or self.combined_stats.empty:
            print("⚠️ No hay datos para calcular medias")
            return {}
        
        try:
            medias = {
                # ===== ATAQUE =====
                'xg_corner_a_favor': self.combined_stats['xg_corner_a_favor'].mean(),
                'xg_falta_a_favor': self.combined_stats['xg_falta_a_favor'].mean(),
                'xg_falta_indirecta_a_favor': self.combined_stats['xg_falta_indirecta_a_favor'].mean(),
                
                # ===== DEFENSA =====
                'xg_corner_en_contra': self.combined_stats['xg_corner_en_contra'].mean(),
                'xg_falta_en_contra': self.combined_stats['xg_falta_en_contra'].mean(),
                'xg_falta_indirecta_en_contra': self.combined_stats['xg_falta_indirecta_en_contra'].mean(),
                
                # ===== TOTALES =====
                'xg_bp_total_a_favor': (
                    self.combined_stats['xg_corner_a_favor'].mean() +
                    self.combined_stats['xg_falta_a_favor'].mean() +
                    self.combined_stats['xg_falta_indirecta_a_favor'].mean()
                ),
                'xg_bp_total_en_contra': (
                    self.combined_stats['xg_corner_en_contra'].mean() +
                    self.combined_stats['xg_falta_en_contra'].mean() +
                    self.combined_stats['xg_falta_indirecta_en_contra'].mean()
                ),
                # AÑADIR ESTA LÍNEA:
                'xg_bp_neto': (
                    self.combined_stats['xg_corner_a_favor'].mean() +
                    self.combined_stats['xg_falta_a_favor'].mean() +
                    self.combined_stats['xg_falta_indirecta_a_favor'].mean()
                ) - (
                    self.combined_stats['xg_corner_en_contra'].mean() +
                    self.combined_stats['xg_falta_en_contra'].mean() +
                    self.combined_stats['xg_falta_indirecta_en_contra'].mean()
                )
            }
            
            
            return medias
            
        except KeyError as e:
            print(f"⚠️ Error: Columna no encontrada en los datos: {e}")
            return {}
        except Exception as e:
            print(f"❌ Error al calcular medias: {e}")
            return {}
    
    def calcular_puntuaciones(self, equipo):
        """
        Calcula puntuaciones 1-10 para cada categoría considerando múltiples métricas:
        - xG total
        - Eficiencia (xG por corner/falta)
        - Tiros a favor/contra
        - Número de corners/faltas
        """
        if self.combined_stats is None or self.medias_liga is None:
            print(f"⚠️ No hay datos para calcular puntuaciones de {equipo}")
            return None
        
        try:
            # Filtrar datos del equipo
            equipo_data = self.combined_stats[self.combined_stats['Team Name'] == equipo]
            
            if equipo_data.empty:
                print(f"⚠️ No se encontraron datos para {equipo}")
                return None
            
            # Calcular promedios del equipo por jornada
            promedios_equipo = {
                # CORNERS
                'xg_corner_a_favor': equipo_data['xg_corner_a_favor'].mean(),
                'xg_corner_en_contra': equipo_data['xg_corner_en_contra'].mean(),
                'corners_a_favor': equipo_data['corners_a_favor'].mean(),
                'corners_en_contra': equipo_data['corners_en_contra'].mean(),
                'xg_por_corner_favor': equipo_data['xg_por_corner_favor'].mean(),
                'xg_por_corner_contra': equipo_data['xg_por_corner_contra'].mean(),
                'tiros_por_corner_favor': equipo_data['tiros_por_corner_favor'].mean(),
                'tiros_por_corner_contra': equipo_data['tiros_por_corner_contra'].mean(),
                
                # FALTAS DIRECTAS
                'xg_falta_a_favor': equipo_data['xg_falta_a_favor'].mean(),
                'xg_falta_en_contra': equipo_data['xg_falta_en_contra'].mean(),
                'faltas_a_favor': equipo_data['faltas_a_favor'].mean(),
                'faltas_en_contra': equipo_data['faltas_en_contra'].mean(),
                'tiros_por_falta_favor': equipo_data['tiros_por_falta_favor'].mean(),
                'tiros_por_falta_contra': equipo_data['tiros_por_falta_contra'].mean(),
                
                # FALTAS INDIRECTAS
                'xg_falta_indirecta_a_favor': equipo_data['xg_falta_indirecta_a_favor'].mean(),
                'xg_falta_indirecta_en_contra': equipo_data['xg_falta_indirecta_en_contra'].mean(),
                'faltas_indirectas_a_favor': equipo_data['faltas_indirectas_a_favor'].mean(),
                'faltas_indirectas_en_contra': equipo_data['faltas_indirectas_en_contra'].mean(),
                'tiros_por_falta_indirecta_favor': equipo_data['tiros_por_falta_indirecta_favor'].mean(),
                'tiros_por_falta_indirecta_contra': equipo_data['tiros_por_falta_indirecta_contra'].mean(),
            }
            
            # Función auxiliar mejorada para normalizar
            def normalizar_puntuacion(valor_equipo, media_liga, es_defensa=False):
                """
                Normaliza la puntuación a escala 1-10
                - es_defensa=True: valores bajos son mejores
                - es_defensa=False: valores altos son mejores
                """
                if media_liga == 0 or valor_equipo == 0:
                    return 5.0  # Valor neutro
                
                ratio = valor_equipo / media_liga
                
                if es_defensa:
                    # En defensa, MENOS es MEJOR (invertido)
                    if ratio <= 0.5:
                        puntuacion = 10.0
                    elif ratio <= 0.7:
                        puntuacion = 9.0
                    elif ratio <= 0.85:
                        puntuacion = 8.0
                    elif ratio <= 1.0:
                        puntuacion = 7.0
                    elif ratio <= 1.15:
                        puntuacion = 6.0
                    elif ratio <= 1.3:
                        puntuacion = 5.0
                    elif ratio <= 1.5:
                        puntuacion = 4.0
                    elif ratio <= 1.8:
                        puntuacion = 3.0
                    elif ratio <= 2.0:
                        puntuacion = 2.0
                    else:
                        puntuacion = 1.0
                else:
                    # En ataque, MÁS es MEJOR
                    if ratio >= 2.0:
                        puntuacion = 10.0
                    elif ratio >= 1.8:
                        puntuacion = 9.0
                    elif ratio >= 1.5:
                        puntuacion = 8.0
                    elif ratio >= 1.3:
                        puntuacion = 7.0
                    elif ratio >= 1.15:
                        puntuacion = 6.0
                    elif ratio >= 1.0:
                        puntuacion = 5.5
                    elif ratio >= 0.85:
                        puntuacion = 5.0
                    elif ratio >= 0.7:
                        puntuacion = 4.0
                    elif ratio >= 0.5:
                        puntuacion = 3.0
                    else:
                        puntuacion = 2.0
                
                return round(puntuacion, 1)
            
            # Calcular medias de la liga para métricas adicionales
            medias_liga_extra = {
                'corners_a_favor': self.combined_stats['corners_a_favor'].mean(),
                'xg_por_corner_favor': self.combined_stats['xg_por_corner_favor'].mean(),
                'xg_por_corner_contra': self.combined_stats['xg_por_corner_contra'].mean(),
                'tiros_por_corner_favor': self.combined_stats['tiros_por_corner_favor'].mean(),
                'tiros_por_corner_contra': self.combined_stats['tiros_por_corner_contra'].mean(),
                'faltas_a_favor': self.combined_stats['faltas_a_favor'].mean(),
                'tiros_por_falta_favor': self.combined_stats['tiros_por_falta_favor'].mean(),
                'tiros_por_falta_contra': self.combined_stats['tiros_por_falta_contra'].mean(),
                'faltas_indirectas_a_favor': self.combined_stats['faltas_indirectas_a_favor'].mean(),
                'tiros_por_falta_indirecta_favor': self.combined_stats['tiros_por_falta_indirecta_favor'].mean(),
                'tiros_por_falta_indirecta_contra': self.combined_stats['tiros_por_falta_indirecta_contra'].mean(),
            }
            
            # ========== VALORACIÓN CORNERS ==========
            # Ataque: combinar xG, eficiencia y tiros
            corners_ataque_scores = [
                normalizar_puntuacion(promedios_equipo['xg_corner_a_favor'], 
                                    self.medias_liga['xg_corner_a_favor'], False),
                normalizar_puntuacion(promedios_equipo['xg_por_corner_favor'], 
                                    medias_liga_extra['xg_por_corner_favor'], False),
                normalizar_puntuacion(promedios_equipo['tiros_por_corner_favor'], 
                                    medias_liga_extra['tiros_por_corner_favor'], False),
                normalizar_puntuacion(promedios_equipo['corners_a_favor'], 
                                    medias_liga_extra['corners_a_favor'], False)
            ]
            punt_corners_ataque = round(sum(corners_ataque_scores) / len(corners_ataque_scores), 1)
            
            # Defensa: combinar xG en contra, eficiencia defensiva y tiros en contra
            corners_defensa_scores = [
                normalizar_puntuacion(promedios_equipo['xg_corner_en_contra'], 
                                    self.medias_liga['xg_corner_en_contra'], True),
                normalizar_puntuacion(promedios_equipo['xg_por_corner_contra'], 
                                    medias_liga_extra['xg_por_corner_contra'], True),
                normalizar_puntuacion(promedios_equipo['tiros_por_corner_contra'], 
                                    medias_liga_extra['tiros_por_corner_contra'], True)
            ]
            punt_corners_defensa = round(sum(corners_defensa_scores) / len(corners_defensa_scores), 1)
            
            # Puntuación combinada corners
            punt_corners_total = round((punt_corners_ataque + punt_corners_defensa) / 2, 1)
            
            # ========== VALORACIÓN FALTAS DIRECTAS ==========
            faltas_dir_ataque_scores = [
                normalizar_puntuacion(promedios_equipo['xg_falta_a_favor'], 
                                    self.medias_liga['xg_falta_a_favor'], False),
                normalizar_puntuacion(promedios_equipo['tiros_por_falta_favor'], 
                                    medias_liga_extra['tiros_por_falta_favor'], False),
                normalizar_puntuacion(promedios_equipo['faltas_a_favor'], 
                                    medias_liga_extra['faltas_a_favor'], False)
            ]
            punt_faltas_dir_ataque = round(sum(faltas_dir_ataque_scores) / len(faltas_dir_ataque_scores), 1)
            
            faltas_dir_defensa_scores = [
                normalizar_puntuacion(promedios_equipo['xg_falta_en_contra'], 
                                    self.medias_liga['xg_falta_en_contra'], True),
                normalizar_puntuacion(promedios_equipo['tiros_por_falta_contra'], 
                                    medias_liga_extra['tiros_por_falta_contra'], True)
            ]
            punt_faltas_dir_defensa = round(sum(faltas_dir_defensa_scores) / len(faltas_dir_defensa_scores), 1)
            
            punt_faltas_dir_total = round((punt_faltas_dir_ataque + punt_faltas_dir_defensa) / 2, 1)
            
            # ========== VALORACIÓN FALTAS INDIRECTAS ==========
            faltas_ind_ataque_scores = [
                normalizar_puntuacion(promedios_equipo['xg_falta_indirecta_a_favor'], 
                                    self.medias_liga['xg_falta_indirecta_a_favor'], False),
                normalizar_puntuacion(promedios_equipo['tiros_por_falta_indirecta_favor'], 
                                    medias_liga_extra['tiros_por_falta_indirecta_favor'], False),
                normalizar_puntuacion(promedios_equipo['faltas_indirectas_a_favor'], 
                                    medias_liga_extra['faltas_indirectas_a_favor'], False)
            ]
            punt_faltas_ind_ataque = round(sum(faltas_ind_ataque_scores) / len(faltas_ind_ataque_scores), 1)
            
            faltas_ind_defensa_scores = [
                normalizar_puntuacion(promedios_equipo['xg_falta_indirecta_en_contra'], 
                                    self.medias_liga['xg_falta_indirecta_en_contra'], True),
                normalizar_puntuacion(promedios_equipo['tiros_por_falta_indirecta_contra'], 
                                    medias_liga_extra['tiros_por_falta_indirecta_contra'], True)
            ]
            punt_faltas_ind_defensa = round(sum(faltas_ind_defensa_scores) / len(faltas_ind_defensa_scores), 1)
            
            punt_faltas_ind_total = round((punt_faltas_ind_ataque + punt_faltas_ind_defensa) / 2, 1)
            
            # ========== VALORACIÓN xG NETO TOTAL ==========
            xg_neto_equipo = (
                promedios_equipo['xg_corner_a_favor'] +
                promedios_equipo['xg_falta_a_favor'] +
                promedios_equipo['xg_falta_indirecta_a_favor']
            ) - (
                promedios_equipo['xg_corner_en_contra'] +
                promedios_equipo['xg_falta_en_contra'] +
                promedios_equipo['xg_falta_indirecta_en_contra']
            )

            xg_neto_liga = self.medias_liga.get('xg_bp_neto', 0)

            # Normalizar xG neto (más es mejor)
            punt_xg_neto = normalizar_puntuacion(xg_neto_equipo, xg_neto_liga if xg_neto_liga != 0 else 0.1, False)

            # Estructura de puntuaciones
            puntuaciones = {
                'ataque': {
                    'corners': punt_corners_ataque,
                    'faltas_directas': punt_faltas_dir_ataque,
                    'faltas_indirectas': punt_faltas_ind_ataque
                },
                'defensa': {
                    'corners': punt_corners_defensa,
                    'faltas_directas': punt_faltas_dir_defensa,
                    'faltas_indirectas': punt_faltas_ind_defensa
                }
            }
            
            # Calcular promedios
            puntuaciones['promedio_ataque'] = round(
                sum(puntuaciones['ataque'].values()) / len(puntuaciones['ataque']), 1
            )
            puntuaciones['promedio_defensa'] = round(
                sum(puntuaciones['defensa'].values()) / len(puntuaciones['defensa']), 1
            )
            # 60% valoración técnica (ataque+defensa) + 40% xG neto
            promedio_tecnico = (puntuaciones['promedio_ataque'] + puntuaciones['promedio_defensa']) / 2
            puntuaciones['promedio_total'] = round(
                (promedio_tecnico * 0.6) + (punt_xg_neto * 0.4), 1
            )
            
            return puntuaciones
            
        except Exception as e:
            print(f"❌ Error al calcular puntuaciones para {equipo}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generar_resumen_bp(self, puntuaciones):
        """
        Genera un resumen textual basado en las puntuaciones
        """
        if puntuaciones is None:
            return "Sin datos suficientes"
        
        try:
            ataque = puntuaciones['promedio_ataque']
            defensa = puntuaciones['promedio_defensa']
            total = puntuaciones['promedio_total']
            
            # Puntuaciones individuales
            corners_atq = puntuaciones['ataque']['corners']
            faltas_dir_atq = puntuaciones['ataque']['faltas_directas']
            faltas_ind_atq = puntuaciones['ataque']['faltas_indirectas']
            
            # Clasificación principal
            if total >= 9.0:
                clasificacion = "[★] Excelente en BP"  # Cambiar ⭐ por [★]
                emoji = "[★]"
            elif total >= 6.5:
                clasificacion = "[✓] Bien en BP"  # Cambiar ✅ por [✓]
                emoji = "[✓]"
            elif total < 5.0:
                clasificacion = "[-] Correcto en BP"  # Cambiar ➖ por [-]
                emoji = "[-]"
            else:
                clasificacion = "[!] En desarrollo"  # Cambiar ⚠️ por [!]
                emoji = "[!]"
            
            # Análisis detallado
            detalles = []
            
            # Análisis de ataque vs defensa
            if ataque >= 8.0 and defensa >= 8.0:
                detalles.append("Dominio total en BP")
            elif ataque >= 8.0:
                detalles.append("Excelente en Ataque")
            elif defensa >= 8.0:
                detalles.append("Excelente en Defensa")
            elif ataque >= 7.0 and defensa < 6.0:
                detalles.append("Bien en Ataque")
            elif defensa >= 7.0 and ataque < 6.0:
                detalles.append("Bien en Defensa")
            
            # Especialidades específicas
            if corners_atq >= 8.0:
                detalles.append("Especialista en Corners")
            if faltas_dir_atq >= 8.0:
                detalles.append("Peligroso en Faltas Directas")
            if faltas_ind_atq >= 8.0:
                detalles.append("Efectivo en Faltas Indirectas")
            
            # Construir resumen
            if detalles:
                resumen_detallado = f"{clasificacion} - {', '.join(detalles)}"
            else:
                resumen_detallado = clasificacion
            
            return {
                'clasificacion': clasificacion,
                'emoji': emoji,
                'detalle': resumen_detallado,
                'puntuacion_total': total,
                'puntuacion_ataque': ataque,
                'puntuacion_defensa': defensa
            }
            
        except Exception as e:
            print(f"❌ Error al generar resumen: {e}")
            return {
                'clasificacion': "Error",
                'emoji': "❌",
                'detalle': "No se pudo generar resumen",
                'puntuacion_total': 0,
                'puntuacion_ataque': 0,
                'puntuacion_defensa': 0
            }
    
    def formatear_ranking_con_valoracion(self, ax, x, y, categoria, ranking, total_equipos, valoracion, ha='left'):
        """
        Formatea un ranking con su valoración destacando top 5
        categoria: nombre (ej: "Corners")
        ranking: posición (1-20)
        total_equipos: total de equipos
        valoracion: nota 1-10
        """
        # Determinar si es top 5
        es_top5 = ranking <= 5
        
        # Color según valoración
        if valoracion >= 8:
            color_val = '#28a745'  # Verde
        elif valoracion >= 6:
            color_val = '#1e90ff'  # Azuk
        else:
            color_val = '#dc3545'  # Rojo
        
        # Formato del texto
        if es_top5:
            # TOP 5: más grande, negrita, con emoji
            texto = f'[★] {categoria}: {ranking}º/{total_equipos} | {valoracion:.1f}/10'  
            fontsize = 11
            weight = 'bold'
            color_ranking = '#d4af37'  # Dorado para top 5
        else:
            # No top 5: normal
            texto = f'{categoria}: {ranking}º/{total_equipos} | {valoracion:.1f}/10'
            fontsize = 9
            weight = 'medium'
            color_ranking = '#555555'
        
        # Dibujar el texto del ranking
        ax.text(x, y, texto.split('|')[0], 
                fontsize=fontsize, ha=ha, va='top', 
                weight=weight, color=color_ranking, family='sans-serif')
        
        # Dibujar la valoración con su color
        valoracion_offset = 5.2 if not es_top5 else 5.8
        ax.text(x + valoracion_offset, y, f'| {valoracion:.1f}/10', 
                fontsize=fontsize, ha='left', va='top', 
                weight=weight, color=color_val, family='sans-serif')

    def formatear_ranking_completo(self, ax, x, y, categoria, ranking, total_equipos, 
                                punt_ataque, punt_defensa, equipo_color='#555555'):
        """
        Formatea un ranking completo con valoraciones de ATAQUE y DEFENSA separadas (versión compacta)
        """
        es_top5 = ranking <= 5
        
        # Colores según valoración
        def get_color(valoracion):
            if valoracion >= 9:
                return '#28a745'  # Verde
            elif valoracion >= 6:
                return '#1e90ff'  # Azul
            elif valoracion >= 5:
                return '#ff8c00'  # Naranja
            else:
                return '#dc3545'  # Rojo
        
        color_ataque = get_color(punt_ataque)
        color_defensa = get_color(punt_defensa)
        
        # Formato del ranking principal
        if es_top5:
            texto_ranking = f'[★] {categoria}' 
            fontsize_titulo = 10
            weight_titulo = 'bold'
            color_ranking = '#d4af37'  # Dorado
        else:
            texto_ranking = f'{categoria}'
            fontsize_titulo = 9.5
            weight_titulo = 'bold'
            color_ranking = '#555555'
        
        # LÍNEA 1: Título
        ax.text(x + 1.5, y, texto_ranking, 
                fontsize=fontsize_titulo, ha='center', va='top', 
                weight=weight_titulo, color=color_ranking, family='sans-serif')
        
        # LÍNEA 2: Ranking
        y_ranking = y - 0.18
        ax.text(x + 1.5, y_ranking, f'{ranking}º/{total_equipos}', 
                fontsize=8, ha='center', va='top', weight='bold', 
                color='#666666', family='sans-serif')
        
        # LÍNEA 3: Ataque
        y_ataque = y - 0.35
        ax.text(x + 1.5, y_ataque, f'⚔️ {punt_ataque:.1f}', 
                fontsize=8, ha='center', va='top', weight='bold', 
                color=color_ataque, family='sans-serif',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                        edgecolor=color_ataque, linewidth=1, alpha=0.8))
        
        # LÍNEA 4: Defensa (usando escudo más simple)
        y_defensa = y - 0.52
        ax.text(x + 1.5, y_defensa, f'🔰 {punt_defensa:.1f}', 
                fontsize=8, ha='center', va='top', weight='bold', 
                color=color_defensa, family='sans-serif',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                        edgecolor=color_defensa, linewidth=1, alpha=0.8))
                                        
    # Métodos de diseño visual (copiados del script original)
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
    
    def draw_leader_badge(self, ax, x, y, escudo, team_name, position='right'):
        """
        Dibuja el escudo del líder con círculo "1º"
        position: 'right', 'left', o 'center'
        """
        if escudo is None:
            return
        
        try:
            # Ajustar posición según lado
            if position == 'center':
                escudo_x = x
                circulo_x = x
            elif position == 'right':
                escudo_x = x + 0.6
                circulo_x = x + 1.0
            else:  # left
                escudo_x = x - 0.6
                circulo_x = x - 1.0
            
            # Dibujar escudo
            imagebox = OffsetImage(escudo, zoom=0.35, alpha=0.95)
            ab = AnnotationBbox(imagebox, (escudo_x, y), frameon=False, zorder=6)
            ax.add_artist(ab)
            
            # Círculo blanco con "1º"
            circle = patches.Circle((circulo_x, y), 0.22, 
                                facecolor='white', edgecolor='#FFD700', 
                                linewidth=2.5, alpha=0.95, zorder=7)
            ax.add_patch(circle)
            
            ax.text(circulo_x, y, '1º', fontsize=8, weight='bold', 
                ha='center', va='center', color='#d4af37', 
                family='sans-serif', zorder=8)
            
        except Exception as e:
            pass
    
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

    def get_smart_text_position(self, h1, h2, h_prom, y_base, bar_width):
        """Función para decidir posicionamiento inteligente de texto"""
        base_offset = 0.15
        
        # Crear lista de alturas con info
        bars = [
            ('vil', h1, 0), 
            ('riv', h2, bar_width*0.6), 
            ('prom', h_prom, -bar_width*0.8)
        ]
        bars.sort(key=lambda x: x[1], reverse=True)  # Más alta primero
        
        # Asignar posiciones sin colisiones
        used_heights = []
        positions = []
        
        for tipo, height, x_offset in bars:
            y_pos = y_base + height + base_offset
            
            # Verificar colisiones con posiciones usadas
            while any(abs(y_pos - used_y) < 0.25 for used_y in used_heights):
                y_pos += 0.25
                
            used_heights.append(y_pos)
            positions.append((tipo, y_pos, x_offset))
        
        return positions
    
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
    
    def create_balon_parado_visualization(self, equipo_rival="Sevilla", figsize=(11.69, 8.27)):
        """Crea la visualización completa del informe de balón parado"""
        
        if self.combined_stats is None or self.combined_stats.empty:
            pass
            return None
        
        # Configurar fuentes modernas
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

        # Villarreal siempre es el primer equipo
        equipo1 = "Villarreal"
        equipo2 = equipo_rival
        
        # Filtrar estadísticas de los equipos y calcular promedios
        equipo1_data = self.combined_stats[self.combined_stats['Team Name'] == equipo1]
        equipo2_data = self.combined_stats[self.combined_stats['Team Name'] == equipo2]
        
        if equipo1_data.empty:
            pass
            return None
        if equipo2_data.empty:
            pass
            return None

        # ✅ GUARDAR el número de partidos jugados (número de jornadas)
        vil_partidos = len(equipo1_data)
        riv_partidos = len(equipo2_data)

        # Calcular promedios por equipo (excluyendo Week y Team Name)
        numeric_cols = equipo1_data.select_dtypes(include=[np.number]).columns
        equipo1_stats = equipo1_data[numeric_cols].mean().to_dict()
        equipo2_stats = equipo2_data[numeric_cols].mean().to_dict()

        # ========== CALCULAR PUNTUACIONES Y RESÚMENES ==========
        punt_equipo1 = self.calcular_puntuaciones(equipo1)
        punt_equipo2 = self.calcular_puntuaciones(equipo2)

        resumen_equipo1 = self.generar_resumen_bp(punt_equipo1)
        resumen_equipo2 = self.generar_resumen_bp(punt_equipo2)


        # Añadir Team Name y partidos jugados para compatibilidad
        equipo1_stats['Team Name'] = equipo1
        equipo2_stats['Team Name'] = equipo2
        equipo1_stats['partidos_jugados'] = vil_partidos
        equipo2_stats['partidos_jugados'] = riv_partidos
        
        # ✅ AÑADIR ESTO: Recalcular rankings basados en promedios por equipo
        # Calcular promedios para TODOS los equipos
        all_teams_stats = self.combined_stats.groupby('Team Name')[numeric_cols].mean().reset_index()
        
        # Recalcular rankings
        all_teams_stats['ranking_corners_favor'] = all_teams_stats['corners_a_favor'].rank(ascending=False, method='min').astype(int)
        all_teams_stats['ranking_xg_total'] = (
            all_teams_stats['xg_corner_a_favor'] + 
            all_teams_stats['xg_falta_a_favor'] +
            all_teams_stats['xg_falta_indirecta_a_favor']
        ).rank(ascending=False, method='min').astype(int)
        all_teams_stats['ranking_faltas_favor'] = all_teams_stats['faltas_a_favor'].rank(ascending=False, method='min').astype(int)
        all_teams_stats['ranking_faltas_indirectas_favor'] = all_teams_stats['faltas_indirectas_a_favor'].rank(ascending=False, method='min').astype(int)
        
        # Actualizar rankings en equipo1_stats y equipo2_stats
        equipo1_rankings = all_teams_stats[all_teams_stats['Team Name'] == equipo1].iloc[0]
        equipo2_rankings = all_teams_stats[all_teams_stats['Team Name'] == equipo2].iloc[0]
        
        equipo1_stats['ranking_corners_favor'] = equipo1_rankings['ranking_corners_favor']
        equipo1_stats['ranking_xg_total'] = equipo1_rankings['ranking_xg_total']
        equipo1_stats['ranking_faltas_favor'] = equipo1_rankings['ranking_faltas_favor']
        equipo1_stats['ranking_faltas_indirectas_favor'] = equipo1_rankings['ranking_faltas_indirectas_favor']
        
        equipo2_stats['ranking_corners_favor'] = equipo2_rankings['ranking_corners_favor']
        equipo2_stats['ranking_xg_total'] = equipo2_rankings['ranking_xg_total']
        equipo2_stats['ranking_faltas_favor'] = equipo2_rankings['ranking_faltas_favor']
        equipo2_stats['ranking_faltas_indirectas_favor'] = equipo2_rankings['ranking_faltas_indirectas_favor']

        
        
        # ========== IDENTIFICAR LÍDERES EN CADA CATEGORÍA ==========
        lider_corners = all_teams_stats[all_teams_stats['ranking_corners_favor'] == 1]['Team Name'].iloc[0]
        lider_faltas = all_teams_stats[all_teams_stats['ranking_faltas_favor'] == 1]['Team Name'].iloc[0]
        lider_faltas_ind = all_teams_stats[all_teams_stats['ranking_faltas_indirectas_favor'] == 1]['Team Name'].iloc[0]

        # Cargar escudos de los líderes
        escudo_lider_corners = self.find_team_logo_by_similarity(lider_corners)
        escudo_lider_faltas = self.find_team_logo_by_similarity(lider_faltas)
        escudo_lider_faltas_ind = self.find_team_logo_by_similarity(lider_faltas_ind)


        # Crear figura que ocupe toda la hoja
        fig = plt.figure(figsize=figsize, facecolor='white', constrained_layout=False)
        fig.patch.set_visible(False)
        
        # Cargar y establecer fondo
        background = self.load_background()
        if background is not None:
            try:
                ax_background = fig.add_axes([0, 0, 1, 1], zorder=-1)
                ax_background.imshow(background, extent=[0, 1, 0, 1], aspect='auto', alpha=0.15, zorder=-1)
                ax_background.axis('off')
            except Exception as e:
                pass
        
        # Configurar el área principal que ocupe casi toda la hoja
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 28)  # Mantener ancho para acomodar la gráfica evolutiva
        ax.set_ylim(0, 12)  # Mantener altura
        ax.axis('off')
        
        # Título principal con diseño moderno
        center_x = 14  # Centro del canvas
        title_text = ax.text(center_x, 11.2, 'BALÓN PARADO', fontsize=42, weight='bold', ha='center', va='center',
                color='#1a1a1a', family='sans-serif')
        title_text.set_path_effects([patheffects.withStroke(linewidth=4, foreground='white')])
        
        # Subtítulo
        ax.text(center_x, 10.6, 'Análisis Comparativo de Situaciones de Balón Parado', 
                fontsize=18, ha='center', va='center', color='#333333', weight='medium', family='sans-serif')
        
        # Logo y escudos en la parte superior con mejor posicionamiento
        ball = self.load_ball_image()
        if ball is not None:
            try:
                imagebox = OffsetImage(ball, zoom=0.15)
                ab = AnnotationBbox(imagebox, (1.5, 10.9), frameon=False)
                ax.add_artist(ab)
            except:
                pass
        
        # Escudos más pequeños y más a la derecha
        rival_logo = self.find_team_logo_by_similarity(equipo2)
        if rival_logo is not None:
            try:
                imagebox = OffsetImage(rival_logo, zoom=0.8)  # Más pequeño
                ab = AnnotationBbox(imagebox, (27.0, 10.9), frameon=False)  # Más a la derecha
                ax.add_artist(ab)
            except Exception as e:
                pass

        villarreal_logo = self.find_team_logo_by_similarity(equipo1)
        if villarreal_logo is not None:
            try:
                imagebox = OffsetImage(villarreal_logo, zoom=0.8)  # Más pequeño
                ab = AnnotationBbox(imagebox, (25.5, 10.9), frameon=False)  # Más a la derecha
                ax.add_artist(ab)
            except Exception as e:
                pass
        
        # ========== PANEL VILLARREAL (IZQUIERDA) - REDISEÑADO CON 2 COLUMNAS ==========
        panel_vil_x = 0.5
        panel_vil_y = 9.4
        panel_vil_width = 6.5
        panel_vil_height = 5.0  # Reducida para dar más separación

        # Fondo del panel Villarreal con sombra
        shadow_vil = patches.FancyBboxPatch(
            (panel_vil_x + 0.08, panel_vil_y - panel_vil_height - 0.08), 
            panel_vil_width, panel_vil_height,
            boxstyle="round,pad=0.15",
            facecolor='#000000',
            alpha=0.15,
            zorder=0
        )
        ax.add_patch(shadow_vil)

        panel_vil_bg = patches.FancyBboxPatch(
            (panel_vil_x, panel_vil_y - panel_vil_height), 
            panel_vil_width, panel_vil_height,
            boxstyle="round,pad=0.15", 
            facecolor='#fffef7',  # Amarillo muy suave
            edgecolor='#FFD700',
            linewidth=3,
            alpha=0.95,
            zorder=1
        )
        ax.add_patch(panel_vil_bg)

        # Barra superior dorada
        barra_vil = patches.Rectangle(
            (panel_vil_x, panel_vil_y - 0.4), 
            panel_vil_width, 0.4,
            facecolor='#FFD700',
            edgecolor='none',
            alpha=0.3,
            zorder=2
        )
        ax.add_patch(barra_vil)

        # Escudo de fondo (si existe)
        if villarreal_logo is not None:
            imagebox_vil = OffsetImage(villarreal_logo, zoom=2.2, alpha=0.08)
            ab_vil = AnnotationBbox(imagebox_vil, 
                                    (panel_vil_x + panel_vil_width/2, panel_vil_y - panel_vil_height/2), 
                                    frameon=False, zorder=1)
            ax.add_artist(ab_vil)

        # === CONTENIDO PANEL VILLARREAL ===
        # Título
        ax.text(panel_vil_x + panel_vil_width/2, panel_vil_y - 0.2, 
                f'{equipo1.upper()}', 
                fontsize=16, weight='bold', ha='center', va='center', 
                color='#B8860B', family='sans-serif', zorder=3)

        # xG Total (destacado)
        vil_total_xg = (equipo1_stats['xg_corner_a_favor'] + 
                        equipo1_stats['xg_falta_a_favor'] + 
                        equipo1_stats['xg_falta_indirecta_a_favor'])
        vil_xg_contra_total = (equipo1_stats['xg_corner_en_contra'] + 
                            equipo1_stats['xg_falta_en_contra'] + 
                            equipo1_stats['xg_falta_indirecta_en_contra'])

        ax.text(panel_vil_x + panel_vil_width/2, panel_vil_y - 0.65, 
                f'xG BP: {(vil_total_xg - vil_xg_contra_total):.2f}', 
                fontsize=15, weight='bold', ha='center', va='center', 
                color='#2c3e50', family='sans-serif', zorder=3,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='#FFD700', linewidth=2, alpha=0.9))

        # Clasificación
        ax.text(panel_vil_x + panel_vil_width/2, panel_vil_y - 1.05, 
                resumen_equipo1['clasificacion'], 
                fontsize=11, weight='bold', ha='center', va='center', 
                color='#28a745' if resumen_equipo1['puntuacion_total'] >= 6.5 else '#dc3545',
                family='sans-serif', zorder=3)

        # Valoraciones Ataque/Defensa en columnas
        col_vil_izq = panel_vil_x + 0.6
        col_vil_der = panel_vil_x + panel_vil_width/2 + 0.3

        # COLUMNA IZQUIERDA: Ataque
        ax.text(col_vil_izq + 0.3, panel_vil_y - 1.4, 
                'ATAQUE', 
                fontsize=10, weight='bold', ha='left', va='top', 
                color='#2c3e50', family='sans-serif', zorder=3)

        ax.text(col_vil_izq + 0.6, panel_vil_y - 1.63, 
                f'{punt_equipo1["promedio_ataque"]:.1f}', 
                fontsize=15, weight='bold', ha='left', va='top', 
                color='#28a745' if punt_equipo1['promedio_ataque'] >= 7 else '#dc3545',
                family='sans-serif', zorder=3)

        # COLUMNA DERECHA: Defensa
        ax.text(col_vil_der + 0.28, panel_vil_y - 1.4, 
                'DEFENSA', 
                fontsize=10, weight='bold', ha='left', va='top', 
                color='#2c3e50', family='sans-serif', zorder=3)

        ax.text(col_vil_der + 0.6, panel_vil_y - 1.63, 
                f'{punt_equipo1["promedio_defensa"]:.1f}', 
                fontsize=15, weight='bold', ha='left', va='top', 
                color='#28a745' if punt_equipo1['promedio_defensa'] >= 7 else '#dc3545',
                family='sans-serif', zorder=3)

        # Línea separadora
        ax.plot([panel_vil_x + 0.3, panel_vil_x + panel_vil_width - 0.3], 
                [panel_vil_y - 1.93, panel_vil_y - 1.93], 
                color='#ddd', linewidth=1.5, alpha=0.7, zorder=2)

        # Rankings detallados EN FORMA DE TRIÁNGULO
        total_equipos = len(self.combined_stats['Team Name'].unique())
        vil_rank_corners = int(equipo1_stats['ranking_corners_favor'])
        vil_rank_faltas = int(equipo1_stats['ranking_faltas_favor'])

        # Definir geometría del triángulo - AJUSTADO PARA QUEDAR DENTRO DEL PANEL
        triangle_center_x = panel_vil_x + panel_vil_width/2
        triangle_center_y = panel_vil_y - 2.8
        triangle_radius = 1.2  # Radio reducido para que quepa mejor
        
        # VÉRTICE SUPERIOR: Corners
        corner_x = triangle_center_x
        corner_y = triangle_center_y + triangle_radius * 0.7
        
        self.formatear_ranking_completo(
            ax, corner_x - 1.5, corner_y, 
            "Corners", vil_rank_corners, total_equipos,
            punt_equipo1['ataque']['corners'],
            punt_equipo1['defensa']['corners'],
            '#FFD700'
        )

        # VÉRTICE INFERIOR IZQUIERDO: Faltas directas
        falta_dir_x = triangle_center_x - triangle_radius * 1.4
        falta_dir_y = triangle_center_y - triangle_radius * 0.6
        
        self.formatear_ranking_completo(
            ax, falta_dir_x - 1.5, falta_dir_y,
            "F. Directas", vil_rank_faltas, total_equipos,
            punt_equipo1['ataque']['faltas_directas'],
            punt_equipo1['defensa']['faltas_directas'],
            '#FFD700'
        )

        # VÉRTICE INFERIOR DERECHO: Faltas indirectas
        falta_ind_x = triangle_center_x + triangle_radius * 1.4
        falta_ind_y = triangle_center_y - triangle_radius * 0.6
        
        self.formatear_ranking_completo(
            ax, falta_ind_x - 1.5, falta_ind_y,
            "F. Indirectas", int(equipo1_stats["ranking_faltas_indirectas_favor"]), total_equipos,
            punt_equipo1['ataque']['faltas_indirectas'],
            punt_equipo1['defensa']['faltas_indirectas'],
            '#FFD700'
        )

        # AÑADIR ESCUDOS DE LÍDERES EN EL TRIÁNGULO VILLARREAL
        # Corners - CENTRADO
        self.draw_leader_badge(ax, corner_x, corner_y - 1.1, 
                            escudo_lider_corners, lider_corners, 'center')

        # F. Directas - izquierda
        self.draw_leader_badge(ax, falta_dir_x - 0.5, falta_dir_y - 0.55, 
                            escudo_lider_faltas, lider_faltas, 'left')

        # F. Indirectas - derecha
        self.draw_leader_badge(ax, falta_ind_x + 0.5, falta_ind_y - 0.55, 
                            escudo_lider_faltas_ind, lider_faltas_ind, 'right')

        
        # Dibujar líneas del triángulo (más sutiles)
        triangle_line_alpha = 0.2
        ax.plot([corner_x, falta_dir_x], [corner_y - 0.3, falta_dir_y + 0.2], 
               color='#FFD700', linestyle='--', linewidth=1, alpha=triangle_line_alpha, zorder=1)
        ax.plot([corner_x, falta_ind_x], [corner_y - 0.3, falta_ind_y + 0.2], 
               color='#FFD700', linestyle='--', linewidth=1, alpha=triangle_line_alpha, zorder=1)
        ax.plot([falta_dir_x, falta_ind_x], [falta_dir_y, falta_ind_y], 
               color='#FFD700', linestyle='--', linewidth=1, alpha=triangle_line_alpha, zorder=1)
        

        # ========== PANEL RIVAL (DERECHA) - REDISEÑADO CON 2 COLUMNAS ==========
        panel_riv_x = 7.5
        panel_riv_y = 9.4
        panel_riv_width = 6.5
        panel_riv_height = 5.0  # Reducida para dar más separación

        # Fondo del panel Rival con sombra
        shadow_riv = patches.FancyBboxPatch(
            (panel_riv_x + 0.08, panel_riv_y - panel_riv_height - 0.08), 
            panel_riv_width, panel_riv_height,
            boxstyle="round,pad=0.15",
            facecolor='#000000',
            alpha=0.15,
            zorder=0
        )
        ax.add_patch(shadow_riv)

        panel_riv_bg = patches.FancyBboxPatch(
            (panel_riv_x, panel_riv_y - panel_riv_height), 
            panel_riv_width, panel_riv_height,
            boxstyle="round,pad=0.15", 
            facecolor='#fff5f5',  # Rojo muy suave
            edgecolor='#DC143C',
            linewidth=3,
            alpha=0.95,
            zorder=1
        )
        ax.add_patch(panel_riv_bg)

        # Barra superior roja
        barra_riv = patches.Rectangle(
            (panel_riv_x, panel_riv_y - 0.4), 
            panel_riv_width, 0.4,
            facecolor='#DC143C',
            edgecolor='none',
            alpha=0.3,
            zorder=2
        )
        ax.add_patch(barra_riv)

        # Escudo de fondo (si existe)
        if rival_logo is not None:
            imagebox_riv = OffsetImage(rival_logo, zoom=2.2, alpha=0.08)
            ab_riv = AnnotationBbox(imagebox_riv, 
                                    (panel_riv_x + panel_riv_width/2, panel_riv_y - panel_riv_height/2), 
                                    frameon=False, zorder=1)
            ax.add_artist(ab_riv)

        # === CONTENIDO PANEL RIVAL ===
        # Título
        ax.text(panel_riv_x + panel_riv_width/2, panel_riv_y - 0.2, 
                f'{equipo2.upper()}', 
                fontsize=16, weight='bold', ha='center', va='center', 
                color='#8B0000', family='sans-serif', zorder=3)

        # xG Total (destacado)
        riv_total_xg = (equipo2_stats['xg_corner_a_favor'] + 
                        equipo2_stats['xg_falta_a_favor'] + 
                        equipo2_stats['xg_falta_indirecta_a_favor'])
        riv_xg_contra_total = (equipo2_stats['xg_corner_en_contra'] + 
                            equipo2_stats['xg_falta_en_contra'] + 
                            equipo2_stats['xg_falta_indirecta_en_contra'])

        ax.text(panel_riv_x + panel_riv_width/2, panel_riv_y - 0.65, 
                f'xG BP: {(riv_total_xg - riv_xg_contra_total):.2f}', 
                fontsize=15, weight='bold', ha='center', va='center', 
                color='#2c3e50', family='sans-serif', zorder=3,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='#DC143C', linewidth=2, alpha=0.9))

        # Clasificación
        ax.text(panel_riv_x + panel_riv_width/2, panel_riv_y - 1.05, 
                resumen_equipo2['clasificacion'], 
                fontsize=11, weight='bold', ha='center', va='center', 
                color='#28a745' if resumen_equipo2['puntuacion_total'] >= 6.5 else '#dc3545',
                family='sans-serif', zorder=3)

        # Valoraciones Ataque/Defensa en columnas
        col_riv_izq = panel_riv_x + 0.6
        col_riv_der = panel_riv_x + panel_riv_width/2 + 0.3

        # COLUMNA IZQUIERDA: Ataque
        ax.text(col_riv_izq + 0.3, panel_riv_y - 1.4, 
                'ATAQUE', 
                fontsize=10, weight='bold', ha='left', va='top', 
                color='#2c3e50', family='sans-serif', zorder=3)

        ax.text(col_riv_izq + 0.6, panel_riv_y - 1.63, 
                f'{punt_equipo2["promedio_ataque"]:.1f}', 
                fontsize=15, weight='bold', ha='left', va='top', 
                color='#28a745' if punt_equipo2['promedio_ataque'] >= 7 else '#dc3545',
                family='sans-serif', zorder=3)

        # COLUMNA DERECHA: Defensa (usando emoji sin variación)
        ax.text(col_riv_der + 0.28, panel_riv_y - 1.4, 
                'DEFENSA', 
                fontsize=10, weight='bold', ha='left', va='top', 
                color='#2c3e50', family='sans-serif', zorder=3)

        ax.text(col_riv_der + 0.6, panel_riv_y - 1.63, 
                f'{punt_equipo2["promedio_defensa"]:.1f}', 
                fontsize=15, weight='bold', ha='left', va='top', 
                color='#28a745' if punt_equipo2['promedio_defensa'] >= 7 else '#dc3545',
                family='sans-serif', zorder=3)

        # Línea separadora
        ax.plot([panel_riv_x + 0.3, panel_riv_x + panel_riv_width - 0.3], 
                [panel_riv_y - 1.93, panel_riv_y - 1.93], 
                color='#ddd', linewidth=1.5, alpha=0.7, zorder=2)

        # Rankings detallados EN FORMA DE TRIÁNGULO
        riv_rank_corners = int(equipo2_stats['ranking_corners_favor'])
        riv_rank_faltas = int(equipo2_stats['ranking_faltas_favor'])

        # Definir geometría del triángulo - AJUSTADO PARA QUEDAR DENTRO DEL PANEL
        triangle_center_x = panel_riv_x + panel_riv_width/2
        triangle_center_y = panel_riv_y - 2.8
        triangle_radius = 1.2  # Radio reducido para que quepa mejor
        
        # VÉRTICE SUPERIOR: Corners
        corner_x = triangle_center_x
        corner_y = triangle_center_y + triangle_radius * 0.7
        
        self.formatear_ranking_completo(
            ax, corner_x - 1.5, corner_y,
            "Corners", riv_rank_corners, total_equipos,
            punt_equipo2['ataque']['corners'],
            punt_equipo2['defensa']['corners'],
            '#DC143C'
        )

        # VÉRTICE INFERIOR IZQUIERDO: Faltas directas
        falta_dir_x = triangle_center_x - triangle_radius * 1.4
        falta_dir_y = triangle_center_y - triangle_radius * 0.6
        
        self.formatear_ranking_completo(
            ax, falta_dir_x - 1.5, falta_dir_y,
            "F. Directas", riv_rank_faltas, total_equipos,
            punt_equipo2['ataque']['faltas_directas'],
            punt_equipo2['defensa']['faltas_directas'],
            '#DC143C'
        )

        # VÉRTICE INFERIOR DERECHO: Faltas indirectas
        falta_ind_x = triangle_center_x + triangle_radius * 1.4
        falta_ind_y = triangle_center_y - triangle_radius * 0.6
        
        self.formatear_ranking_completo(
            ax, falta_ind_x - 1.5, falta_ind_y,
            "F. Indirectas", int(equipo2_stats["ranking_faltas_indirectas_favor"]), total_equipos,
            punt_equipo2['ataque']['faltas_indirectas'],
            punt_equipo2['defensa']['faltas_indirectas'],
            '#DC143C'
        )
        
        # AÑADIR ESCUDOS DE LÍDERES EN EL TRIÁNGULO RIVAL
        # Corners - CENTRADO
        self.draw_leader_badge(ax, corner_x, corner_y - 1.1, 
                            escudo_lider_corners, lider_corners, 'center')

        # F. Directas - izquierda
        self.draw_leader_badge(ax, falta_dir_x - 0.5, falta_dir_y - 0.55, 
                            escudo_lider_faltas, lider_faltas, 'left')

        # F. Indirectas - derecha
        self.draw_leader_badge(ax, falta_ind_x + 0.5, falta_ind_y - 0.55, 
                            escudo_lider_faltas_ind, lider_faltas_ind, 'right')

        # Dibujar líneas del triángulo (más sutiles)
        triangle_line_alpha = 0.2
        ax.plot([corner_x, falta_dir_x], [corner_y - 0.3, falta_dir_y + 0.2], 
               color='#DC143C', linestyle='--', linewidth=1, alpha=triangle_line_alpha, zorder=1)
        ax.plot([corner_x, falta_ind_x], [corner_y - 0.3, falta_ind_y + 0.2], 
               color='#DC143C', linestyle='--', linewidth=1, alpha=triangle_line_alpha, zorder=1)
        ax.plot([falta_dir_x, falta_ind_x], [falta_dir_y, falta_ind_y], 
               color='#DC143C', linestyle='--', linewidth=1, alpha=triangle_line_alpha, zorder=1)
        
        
        # ===== LEYENDA GENERAL DEBAJO DE AMBOS PANELES =====
        legend_y_general = panel_vil_y - panel_vil_height - 0.4
        legend_x_center = (panel_vil_x + panel_vil_width + panel_riv_x) / 2  # Centro entre ambos paneles

        ax.text(legend_x_center, legend_y_general, 
                '● Valoración (1-10): xG total • Eficiencia (xG/acción) • Tiros generados/concedidos • Comparado vs media liga',
                fontsize=6.5, ha='center', va='top', color='#555555', 
                family='sans-serif', weight='normal', alpha=0.9, zorder=3,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', 
                        edgecolor='#95a5a6', linewidth=1, alpha=0.8))

        # === GRÁFICA DE JORNADAS (A LA DERECHA DE LOS PANELES) ===
        # Obtener datos por jornada
        jornadas_df = self.get_xg_by_matchday(equipo1, equipo2)

        if not jornadas_df.empty:
            # Adaptar ancho según número de jornadas
            num_jornadas = len(jornadas_df)
            
            if num_jornadas <= 15:
                jornadas_width = 11.0  # Gráfica más estrecha para pocas jornadas
            elif num_jornadas <= 25:
                jornadas_width = 12.5  # Ancho medio
            else:
                jornadas_width = 13.0  # Ancho completo para muchas jornadas
            
            # Configuración de la gráfica (A LA DERECHA de los paneles)
            jornadas_x_start = 14.5  # Empieza después del panel derecho
            jornadas_y_base = panel_vil_y - panel_vil_height + 0.3  # Misma altura que los paneles
            jornadas_height = panel_vil_height - 0.6  # Altura similar a los paneles
            
            # Fondo de la gráfica
            jornadas_bg = patches.FancyBboxPatch(
                (jornadas_x_start, jornadas_y_base), 
                jornadas_width, jornadas_height,
                boxstyle="round,pad=0.15", 
                facecolor='#f8f9fa',
                edgecolor='#95a5a6',
                linewidth=2,
                alpha=0.95,
                zorder=1
            )
            ax.add_patch(jornadas_bg)
            
            # Título de la gráfica
            ax.text(jornadas_x_start + jornadas_width/2, jornadas_y_base + jornadas_height - 0.4, 
                    'EVOLUCIÓN xG BP POR JORNADA', 
                    fontsize=13, weight='bold', ha='center', va='center', 
                    color='#1a1a1a', family='sans-serif', zorder=3)
            
            # Área de dibujo de barras
            plot_area_x = jornadas_x_start + 0.8
            plot_area_width = jornadas_width - 1.6
            plot_area_y = jornadas_y_base + 0.6
            plot_area_height = jornadas_height - 1.4
            
            # Calcular ancho de barras adaptativo según número de jornadas
            # El objetivo es que las barras ocupen bien el espacio sin estar ni muy juntas ni muy separadas
            
            if num_jornadas <= 10:
                # Pocas jornadas: barras anchas y bien espaciadas
                single_bar_width = plot_area_width / (num_jornadas * 2.5)
                bar_spacing = plot_area_width / num_jornadas
            elif num_jornadas <= 15:
                # Jornadas medias-bajas: barras medias
                single_bar_width = plot_area_width / (num_jornadas * 3)
                bar_spacing = plot_area_width / num_jornadas
            elif num_jornadas <= 25:
                # Jornadas medias: barras más delgadas
                single_bar_width = plot_area_width / (num_jornadas * 3.5)
                bar_spacing = plot_area_width / num_jornadas
            elif num_jornadas <= 35:
                # Muchas jornadas: barras delgadas
                single_bar_width = plot_area_width / (num_jornadas * 4)
                bar_spacing = plot_area_width / num_jornadas
            else:
                # Muchísimas jornadas: barras muy delgadas
                single_bar_width = plot_area_width / (num_jornadas * 5)
                bar_spacing = plot_area_width / num_jornadas
            
            # Asegurar un ancho mínimo visible
            single_bar_width = max(0.08, min(single_bar_width, 0.5))
            
            # Escalar valores para la visualización
            max_xg_jornada = max(jornadas_df['team1_xg'].max(), jornadas_df['team2_xg'].max()) if not jornadas_df.empty else 1
            if max_xg_jornada > 0:
                scale_factor_jornadas = plot_area_height / max_xg_jornada
            else:
                scale_factor_jornadas = 1
            
            # Líneas de referencia horizontales
            for i in range(1, 4):
                y_ref = plot_area_y + (plot_area_height * i / 4)
                ax.plot([plot_area_x, plot_area_x + plot_area_width], 
                       [y_ref, y_ref], 
                       color='#ddd', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)
            
            # Calcular y dibujar línea de media
            media_xg = jornadas_df[['team1_xg', 'team2_xg']].mean().mean()
            y_media = plot_area_y + (media_xg * scale_factor_jornadas)
            
            ax.plot([plot_area_x, plot_area_x + plot_area_width], 
                   [y_media, y_media], 
                   color='#708090', linestyle='-', linewidth=2.5, alpha=0.8, zorder=2,
                   label='Media')
            
            # Etiqueta de la línea de media
            ax.text(plot_area_x + plot_area_width + 0.3, y_media, 
                   f'Media: {media_xg:.2f}', 
                   ha='left', va='center', fontsize=8, weight='bold', 
                   color='#708090', family='sans-serif',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                           edgecolor='#708090', linewidth=1.5, alpha=0.9), zorder=3)
            
            # Dibujar barras por jornada
            for i, row in jornadas_df.iterrows():
                week = row['week']
                team1_xg = row['team1_xg']
                team2_xg = row['team2_xg']
                
                # Posición X centrada para esta jornada
                x_center = plot_area_x + (i + 0.5) * bar_spacing
                
                # Alturas escaladas
                h1_jornada = team1_xg * scale_factor_jornadas
                h2_jornada = team2_xg * scale_factor_jornadas
                
                # Barra Villarreal (amarilla)
                ax.add_patch(patches.Rectangle(
                    (x_center - single_bar_width, plot_area_y), 
                    single_bar_width, h1_jornada,
                    facecolor='#FFD700', alpha=0.9, 
                    edgecolor='#B8860B', linewidth=1.5, zorder=2
                ))
                
                # Barra rival (roja)
                ax.add_patch(patches.Rectangle(
                    (x_center, plot_area_y), 
                    single_bar_width, h2_jornada,
                    facecolor='#DC143C', alpha=0.9, 
                    edgecolor='#8B0000', linewidth=1.5, zorder=2
                ))
                
                # Valores encima de las barras (solo si hay espacio y son significativos)
                if team1_xg > 0.01 and single_bar_width > 0.08:
                    ax.text(x_center - single_bar_width/2, plot_area_y + h1_jornada + 0.05, 
                            f'{team1_xg:.1f}', ha='center', va='bottom', 
                            fontsize=max(6, min(9, single_bar_width * 40)), weight='bold', color='#B8860B', zorder=3)
                
                if team2_xg > 0.01 and single_bar_width > 0.08:
                    ax.text(x_center + single_bar_width/2, plot_area_y + h2_jornada + 0.05, 
                            f'{team2_xg:.1f}', ha='center', va='bottom', 
                            fontsize=max(6, min(9, single_bar_width * 40)), weight='bold', color='#8B0000', zorder=3)
                
                # Etiqueta de jornada (mostrar según densidad)
                show_label = False
                if num_jornadas <= 15:
                    show_label = True  # Mostrar todas
                elif num_jornadas <= 25:
                    show_label = (i % 2 == 0)  # Mostrar cada 2
                else:
                    show_label = (i % 3 == 0)  # Mostrar cada 3
                
                if show_label:
                    ax.text(x_center - single_bar_width/4, plot_area_y - 0.15, 
                            f'J{int(week)}', ha='center', va='top', 
                            fontsize=max(7, min(10, single_bar_width * 50)), weight='normal', color='#555555', rotation=45, zorder=3)
            
            # Etiqueta del eje Y
            ax.text(plot_area_x - 0.5, plot_area_y + plot_area_height/2, 
                    'xG BP a favor', ha='center', va='center', fontsize=9, 
                    weight='bold', color='#333333', rotation=90, zorder=3)

        # Comparación colocada encima de la leyenda
        vil_punt_total = punt_equipo1['promedio_total']
        riv_punt_total = punt_equipo2['promedio_total']

        if vil_punt_total > riv_punt_total + 0.5:
            texto_comp = f"[★] {equipo1} superior en BP ({vil_punt_total:.1f} vs {riv_punt_total:.1f})"
            color_comp = '#28a745'
        elif riv_punt_total > vil_punt_total + 0.5:
            texto_comp = f"[★] {equipo2} superior en BP ({riv_punt_total:.1f} vs {vil_punt_total:.1f})"
            color_comp = '#dc3545'
        else:
            texto_comp = f"[=] Igualados en BP | {equipo1}: {vil_punt_total:.1f} - {equipo2}: {riv_punt_total:.1f}"
            color_comp = '#ffc107'

        ax.text(18, 10.1, texto_comp, 
                fontsize=11, weight='bold', ha='center', va='center', 
                color=color_comp, family='sans-serif',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor=color_comp, linewidth=2, alpha=0.95), zorder=5)

        # Leyenda de colores ARRIBA Y CENTRADA
        legend_y = 10.1
        legend_center = 7  # Centro entre los dos paneles
        legend_spacing = 2.3

        # Calcular posiciones centradas
        pos1 = legend_center - legend_spacing  # Villarreal a la izquierda
        pos2 = legend_center  # Promedio Liga en el centro  
        pos3 = legend_center + legend_spacing  # Rival a la derecha

        # Villarreal (izquierda)
        ax.add_patch(patches.Rectangle((pos1-1.0, legend_y-0.16), 2.0, 0.32, 
                                    facecolor='#FFD700', alpha=0.9, edgecolor='#B8860B', linewidth=2, zorder=4))
        ax.text(pos1, legend_y, f'{equipo1.upper()}', fontsize=7, weight='bold', 
                ha='center', va='center', color='#1a1a1a', family='sans-serif', zorder=5)

        # Promedio Liga (centro)
        ax.add_patch(patches.Rectangle((pos2-1.3, legend_y-0.16), 2.6, 0.32, 
                                    facecolor='#708090', alpha=0.8, edgecolor='#2F4F4F', linewidth=2, zorder=4))
        ax.text(pos2, legend_y, 'PROMEDIO LIGA', fontsize=7, weight='bold', 
                ha='center', va='center', color='white', family='sans-serif', zorder=5)

        # Rival (derecha)
        ax.add_patch(patches.Rectangle((pos3-1.0, legend_y-0.16), 2.0, 0.32, 
                                    facecolor='#DC143C', alpha=0.9, edgecolor='#8B0000', linewidth=2, zorder=4))
        ax.text(pos3, legend_y, f'{equipo2.upper()}', fontsize=7, weight='bold', 
                ha='center', va='center', color='white', family='sans-serif', zorder=5)
        
        # Definir las métricas
        metrics_groups = [
            {
                'name': 'CORNERS',
                'metrics': [
                    ('corners_a_favor', 'Córners\na favor'),
                    ('corners_en_contra', 'Córners\nen contra'),
                    ('xg_corner_a_favor', 'xG de\ncórner\na favor'),
                    ('xg_corner_en_contra', 'xG de\ncórner\nen contra'),
                    ('tiros_por_corner_favor', 'Tiros a\nfavor /\ncórner'),
                    ('tiros_por_corner_contra', 'Tiros en\ncontra /\ncórner'),
                    ('xg_por_corner_favor', 'xG a\nfavor /\ncórner'),
                    ('xg_por_corner_contra', 'xG en\ncontra /\ncórner')
                ]
            },
            {
                'name': 'FALTAS DIRECTAS',
                'metrics': [
                    ('faltas_a_favor', 'Falta\ndirecta\na favor'),
                    ('faltas_en_contra', 'Falta\ndirecta\nen contra'),
                    ('xg_falta_a_favor', 'xG a\nfavor\nfalta\ndirecta'),
                    ('xg_falta_en_contra', 'xG en\ncontra\nfalta\ndirecta'),
                    ('tiros_por_falta_favor', 'Tiros a\nfavor /\nfalta\ndirecta'),
                    ('tiros_por_falta_contra', 'Tiros en\ncontra /\nfalta\ndirecta')
                ]
            },
            {
                'name': 'FALTAS INDIRECTAS',
                'metrics': [
                    ('faltas_indirectas_a_favor', 'Falta\nindirecta\na favor'),
                    ('faltas_indirectas_en_contra', 'Falta\nindirecta\nen contra'),
                    ('xg_falta_indirecta_a_favor', 'xG a\nfavor\nfalta\nindirecta'),
                    ('xg_falta_indirecta_en_contra', 'xG en\ncontra\nfalta\nindirecta'),
                    ('tiros_por_falta_indirecta_favor', 'Tiros a\nfavor /\nfalta\nindirecta'),
                    ('tiros_por_falta_indirecta_contra', 'Tiros en\ncontra /\nfalta\nindirecta')
                ]
            }
        ]
        
        # Definir variables de dibujo
        bar_width = 0.35
        bar_spacing = 1.2
        x_start = 0.8
        y_base = 0.5  # Bajado mucho más para dar amplia separación
        max_bar_height = 1.3  # Reducido para que quepan bien

        # Dibujar las barras agrupadas con separadores
        current_x = x_start
        group_separator_width = 1.8  # Espacio extra entre grupos

        for group_idx, group in enumerate(metrics_groups):
            # Añadir título del grupo
            group_center_x = current_x + (len(group['metrics']) - 1) * bar_spacing * 0.5
            ax.text(group_center_x, y_base + max_bar_height + 1.8, group['name'], 
                    ha='center', va='center', fontsize=12, weight='bold', 
                    color='#2c3e50', family='sans-serif',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', edgecolor='#bdc3c7', alpha=0.8))
            
            # Dibujar barras del grupo
            for i, (metric_key, metric_label) in enumerate(group['metrics']):
                x_center = current_x + i * bar_spacing
                
                # Obtener valores (código igual que antes)
                val1 = equipo1_stats[metric_key] if metric_key in equipo1_stats else 0
                val2 = equipo2_stats[metric_key] if metric_key in equipo2_stats else 0
                val_promedio = self.combined_stats[metric_key].mean()
                
                # Escalar para visualización (código igual que antes)
                max_val = max(val1, val2, val_promedio) if max(val1, val2, val_promedio) > 0 else 1
                scale_factor = max_bar_height / max_val
                
                h1 = val1 * scale_factor
                h2 = val2 * scale_factor
                h_prom = val_promedio * scale_factor
                
                # Barras con cantos redondos (código igual que antes)
                ax.add_patch(patches.FancyBboxPatch((x_center - bar_width*0.7, y_base), bar_width*1.4, h_prom,
                                                boxstyle="round,pad=0.06", facecolor='#708090', alpha=0.7, 
                                                edgecolor='#2F4F4F', linewidth=1.5, zorder=1))

                ax.add_patch(patches.FancyBboxPatch((x_center - bar_width*0.5, y_base), bar_width, h1,
                                                boxstyle="round,pad=0.06", facecolor='#FFD700', alpha=0.95, 
                                                edgecolor='#B8860B', linewidth=2.5, zorder=3))

                ax.add_patch(patches.FancyBboxPatch((x_center + bar_width*0.1, y_base), bar_width, h2,
                                                boxstyle="round,pad=0.06", facecolor='#DC143C', alpha=0.95, 
                                                edgecolor='#8B0000', linewidth=2.5, zorder=3))
                
                # Gradientes (código igual que antes)
                if h1 > 0.4:
                    ax.add_patch(patches.FancyBboxPatch((x_center - bar_width*0.45, y_base + h1*0.7), 
                                                    bar_width*0.9, h1*0.25,
                                                    boxstyle="round,pad=0.03", facecolor='#FFFF99', 
                                                    edgecolor='none', alpha=0.4, zorder=4))

                if h2 > 0.4:
                    ax.add_patch(patches.FancyBboxPatch((x_center + bar_width*0.15, y_base + h2*0.7), 
                                                    bar_width*0.9, h2*0.25,
                                                    boxstyle="round,pad=0.03", facecolor='#FF6B6B', 
                                                    edgecolor='none', alpha=0.4, zorder=4))

                # Valores en las barras (código igual que antes)
                if h1 > 0.4:
                    ax.text(x_center - bar_width*0.25, y_base + h1*0.5, f'{val1:.2f}', 
                            ha='center', va='center', fontsize=9, weight='bold', 
                            color='white', family='sans-serif',
                            bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.7))
                else:
                    ax.text(x_center - bar_width*0.8, y_base + h1 + 0.1, f'{val1:.2f}', 
                            ha='center', va='bottom', fontsize=8, weight='bold', 
                            color='#B8860B', family='sans-serif')

                if h2 > 0.4:
                    ax.text(x_center + bar_width*0.35, y_base + h2*0.5, f'{val2:.2f}', 
                            ha='center', va='center', fontsize=9, weight='bold', 
                            color='white', family='sans-serif',
                            bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.7))
                else:
                    ax.text(x_center + bar_width*0.8, y_base + h2 + 0.1, f'{val2:.2f}', 
                            ha='center', va='bottom', fontsize=8, weight='bold', 
                            color='#8B0000', family='sans-serif')

                # Promedio
                if h_prom > 0.2:
                    ax.text(x_center, y_base + h_prom + 0.05, f'{val_promedio:.2f}', 
                            ha='center', va='bottom', fontsize=7, weight='normal', 
                            color='#555555', family='sans-serif', alpha=0.8)
                
                # Etiquetas de métricas (alternando arriba y abajo)
                if i % 2 == 0:
                    label_y = y_base - 0.3
                    va_pos = 'top'
                else:
                    label_y = y_base + max_bar_height + 0.4
                    va_pos = 'bottom'

                ax.text(x_center, label_y, metric_label, ha='center', va=va_pos, 
                    fontsize=9, weight='bold', color='#1a1a1a', family='sans-serif')  # Reducido de 10 a 9
            
            # Actualizar posición X para el siguiente grupo
            current_x += len(group['metrics']) * bar_spacing + group_separator_width
            
            # Añadir línea separadora entre grupos (excepto después del último)
            if group_idx < len(metrics_groups) - 1:
                separator_x = current_x - group_separator_width * 0.5
                ax.plot([separator_x, separator_x], [y_base - 0.8, y_base + max_bar_height + 0.9], 
                        color='#95a5a6', linestyle='--', linewidth=2, alpha=0.6, zorder=2)

        plt.tight_layout()
        return fig

def seleccionar_equipo_rival():
    """Selección interactiva del equipo rival (Villarreal es fijo)"""
    try:
        df = pd.read_parquet("./extraccion_opta/datos_opta_parquet/estadisticas_abp.parquet")
        equipos = sorted(df['Team Name'].dropna().unique())
        
        # Quitar Villarreal de la lista
        equipos = [eq for eq in equipos if eq != "Villarreal"]
        
        if not equipos:
            pass
            return None
        
        
        for i, equipo in enumerate(equipos, 1):
            pass
        
        # Seleccionar equipo rival
        while True:
            try:
                seleccion = input(f"\nSelecciona el equipo rival (1-{len(equipos)}): ").strip()
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
    """Función principal para ejecutar el reporte de balón parado"""
    
    # Seleccionar equipo rival
    equipo_rival = seleccionar_equipo_rival()
    if not equipo_rival:
        print("❌ No se pudo seleccionar el equipo rival")
        return
    
    
    # Crear reporte
    try:
        report_generator = BalonParadoReportCustom()
        
        if report_generator.combined_stats is None or report_generator.combined_stats.empty:
            print("❌ No se pudieron calcular las estadísticas")
            return
            
        fig = report_generator.create_balon_parado_visualization(equipo_rival)
        
        if fig:
            plt.show()
            
            # Guardar como PDF
            rival_filename = equipo_rival.replace(' ', '_').replace('/', '_')
            output_path = f"reporte_balon_parado_Villarreal_vs_{rival_filename}.pdf"
            report_generator.guardar_sin_espacios(fig, output_path)
            
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error al generar el reporte: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()