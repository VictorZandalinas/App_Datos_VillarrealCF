import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image
from mplsoccer import VerticalPitch
from sklearn.cluster import KMeans
import textwrap 
from matplotlib.patches import Rectangle
import warnings
import unicodedata
import re
from collections import Counter


warnings.filterwarnings('ignore')

class ReporteDefensivoCorners:
    def __init__(self, tracking_path="extraccion_sportian/corners_tracking.parquet", team_filter=None):
        self.tracking_path = tracking_path
        self.team_filter = team_filter
        self.df_tracking = None
        self.historial_lanzadores = {} # <--- NUEVO: Memoria
        
        self.posicionamiento_izquierda = pd.DataFrame()
        self.posicionamiento_derecha = pd.DataFrame()
        
        try:
            self.team_stats = pd.read_parquet("extraccion_opta/datos_opta_parquet/team_stats.parquet")
        except:
            self.team_stats = None
        
        self.load_tracking_data()
        
        # --- NUEVO: CONSTRUIR MEMORIA ---
        if self.df_tracking is not None:
            self.construir_historial_lanzadores()
        # -------------------------------
        
        if team_filter:
            self.extract_posicionamiento_defensivo()
    
    def diagnostico_equipos(self):
        """
        Analiza una muestra de córners para ver qué equipos detecta el sistema
        y por qué podría estar fallando la lógica de 'Defensor = No Atacante'.
        """
        if self.df_tracking is None:
            print("❌ No hay datos cargados.")
            return

        # Obtenemos una lista de IDs únicos de córner
        ids_unicos = self.df_tracking['ID_Evento_Corner'].unique()
        
        # Analizamos solo los primeros 10 para no saturar, o una muestra aleatoria
        muestra = ids_unicos[:15] 


        conteo_problemas = {
            'ok': 0,
            'solo_uno': 0,
            'nombre_mismatch': 0
        }

        for id_evento in muestra:
            # Filtramos datos de ese corner (todo el evento, no solo un frame)
            df_evento = self.df_tracking[self.df_tracking['ID_Evento_Corner'] == id_evento]
            
            if df_evento.empty: continue

            # 1. Quién dice el evento que lanza
            lanzador_teorico = str(df_evento['Equipo_Lanzador'].iloc[0]).strip()
            
            # 2. Qué equipos hay realmente en las coordenadas
            equipos_tracking = df_evento['NombreEquipoJugador_Tracking'].dropna().unique()
            equipos_tracking = [str(e).strip() for e in equipos_tracking]
            
            # 3. Lógica de deducción
            # Intentamos encontrar al atacante en la lista de tracking
            atacante_detectado = None
            defensor_detectado = None
            
            # Buscamos coincidencia exacta o contenida
            for eq in equipos_tracking:
                if lanzador_teorico == eq:
                    atacante_detectado = eq
                elif lanzador_teorico in eq or eq in lanzador_teorico:
                    # Caso donde "Girona" != "Girona FC" pero son el mismo
                    atacante_detectado = eq 
            
            # El defensor es quien NO es el atacante detectado
            if atacante_detectado:
                posibles_defensores = [e for e in equipos_tracking if e != atacante_detectado]
                if posibles_defensores:
                    defensor_detectado = posibles_defensores[0] # Asumimos que hay 2 equipos
            
            # 4. Diagnóstico
            status = "✅ OK"
            if len(equipos_tracking) < 2:
                status = "⚠️ SOLO 1 EQUIPO"
                conteo_problemas['solo_uno'] += 1
            elif not atacante_detectado:
                status = "❌ NOMBRE NO COINCIDE"
                conteo_problemas['nombre_mismatch'] += 1
            else:
                conteo_problemas['ok'] += 1

            # Imprimir resultado fila
            equipos_str = ", ".join(equipos_tracking)
            if status == "✅ OK":
                pass

        
        if self.team_filter:
            pass
    
    def analizar_tipo_marcaje(self, df_evento, equipo_defensor, equipo_atacante):
        """
        Clasifica a cada defensor en 'Zona' o 'Hombre' basándose en su movimiento
        y correlación con los atacantes.
        """
        # 1. Filtramos ventana de tiempo de movimiento previo
        window = df_evento[
            (df_evento['Segundos_Desde_Saque'] >= -3.0) & 
            (df_evento['Segundos_Desde_Saque'] <= 5.0)
        ]
        
        if window.empty: return {}

        # Obtenemos listas de jugadores
        defensores = window[window['NombreEquipoJugador_Tracking'] == equipo_defensor]['Nombre_Jugador_Tracking'].unique()
        atacantes = window[window['NombreEquipoJugador_Tracking'] == equipo_atacante]['Nombre_Jugador_Tracking'].unique()
        
        resultados = {}

        for defen in defensores:
            df_def = window[window['Nombre_Jugador_Tracking'] == defen].sort_values('Segundos_Desde_Saque')
            if df_def.empty: continue
            
            # A) ¿Se mueve el defensor? (Desplazamiento Euclídeo)
            coords = df_def[['X_Jugador', 'Y_Jugador']].values
            desplazamiento = np.linalg.norm(coords[-1] - coords[0])
            
            # UMBRAL MOVIMIENTO: Si se mueve menos de 2 metros (aprox 2 unidades Opta en X o Y), es ZONA estática
            # Nota: Opta 100x100, 1 unidad ~ 1 metro aprox (depende del campo, pero sirve de ref)
            if desplazamiento < 2.5:
                resultados[defen] = 'Zona'
                continue

            # B) Si se mueve, ¿sigue a alguien?
            mejor_correlacion = -1
            distancia_promedio_minima = 100

            for atac in atacantes:
                df_att = window[window['Nombre_Jugador_Tracking'] == atac].sort_values('Segundos_Desde_Saque')
                
                # Sincronizar frames (inner join por tiempo)
                merged = pd.merge(df_def, df_att, on='Segundos_Desde_Saque', suffixes=('_def', '_att'))
                if len(merged) < 5: continue # Pocos datos

                # Distancia promedio al atacante
                dists = np.sqrt((merged['X_Jugador_def'] - merged['X_Jugador_att'])**2 + 
                                (merged['Y_Jugador_def'] - merged['Y_Jugador_att'])**2)
                avg_dist = dists.mean()

                # Si está lejos (>4m), difícilmente es marca al hombre
                if avg_dist > 5.0: continue

                # Correlación de movimiento (Vectores)
                # Calculamos cambios frame a frame
                vec_def_x = merged['X_Jugador_def'].diff().fillna(0)
                vec_def_y = merged['Y_Jugador_def'].diff().fillna(0)
                vec_att_x = merged['X_Jugador_att'].diff().fillna(0)
                vec_att_y = merged['Y_Jugador_att'].diff().fillna(0)

                # Producto punto simple para ver si van en misma dirección
                dot_prod = (vec_def_x * vec_att_x) + (vec_def_y * vec_att_y)
                score_movimiento = dot_prod.sum() # Si positivo alto, se mueven igual
                
                if score_movimiento > mejor_correlacion:
                    mejor_correlacion = score_movimiento
                    distancia_promedio_minima = avg_dist
            
            # C) Decisión final
            # Si se mueve mucho Y tiene correlación positiva alta con un rival cercano -> HOMBRE
            # Ajustar umbral según datos reales (0.5 es un valor empírico conservador)
            if mejor_correlacion > 0.5 and distancia_promedio_minima < 4.0:
                resultados[defen] = 'Hombre'
            else:
                resultados[defen] = 'Zona' # Se mueve pero no sigue a nadie específico (bascular)
        
        return resultados
    
    def load_custom_icon(self):
        """Carga el logo/anagrama de la izquierda."""
        possible_paths = ["assets/tactic_logo.png", "assets/anagrama.png", "assets/laliga.png"]
        for path in possible_paths:
            if os.path.exists(path):
                return plt.imread(path)
        return None

    def load_tracking_data(self):
        try:
            pass
            self.df_tracking = pd.read_parquet(self.tracking_path)
        except Exception as e:
            print(f"❌ Error cargando tracking: {e}")

    def get_roles_equipos(self, df_frame):
        """
        Determina los nombres de los equipos Atacante y Defensor en un frame específico.
        Lógica estricta:
        - Atacante: El equipo que figura en la columna 'Equipo_Lanzador'.
        - Defensor: El otro equipo presente en el tracking.
        """
        try:
            # 1. Obtener el nombre del equipo lanzador (ATACANTE)
            if 'Equipo_Lanzador' not in df_frame.columns:
                return None, None
            
            # El atacante es explícitamente el que pone en esta columna
            atacante = df_frame['Equipo_Lanzador'].iloc[0]
            
            # 2. Buscar el defensor (el equipo que NO es el atacante)
            if 'NombreEquipoJugador_Tracking' not in df_frame.columns:
                return None, None
                
            equipos_en_frame = df_frame['NombreEquipoJugador_Tracking'].unique()
            
            defensor = None
            for equipo in equipos_en_frame:
                # Comparación de strings segura
                if str(equipo).strip() != str(atacante).strip():
                    defensor = equipo
                    break
            
            return atacante, defensor

        except Exception as e:
            pass
            return None, None

    def determinar_lado_corner(self, y_balon):
        """
        Determina el lado basándose en la Y del balón (Opta coords 0-100).
        Y > 50 es Izquierda, Y < 50 es Derecha.
        """
        return 'izquierda' if y_balon > 50 else 'derecha'
    
    def analizar_disposicion_ofensiva(self, frame_exacto, equipo_atacante, lado, lanzador_real=None):
        """
        Clasifica atacantes en zonas, usando el nombre real del lanzador para identificarlo.
        """
        # Filtramos solo atacantes
        atacantes = frame_exacto[frame_exacto['NombreEquipoJugador_Tracking'] == equipo_atacante].copy()
        
        if atacantes.empty:
            return {}, [], atacantes

        n_remate, n_rechace, n_corto, n_atras = 0, 0, 0, 0
        zonas_asignadas = []
        es_izq = (lado == 'izquierda') 


        for _, row in atacantes.iterrows():
            x = row.get('X_Jugador', 0.0)
            y = row.get('Y_Jugador', 0.0)
            raw_nombre = row.get('Nombre_Jugador_Tracking')
            nombre = str(raw_nombre) if pd.notna(raw_nombre) else "Desconocido"

            if pd.isna(x): x = 0.0
            if pd.isna(y): y = 0.0
            
            zona = "SIN CLASIFICAR"

            # 1. LANZADOR (Prioridad por nombre)
            es_lanzador_por_nombre = False
            if lanzador_real and str(lanzador_real) != "Desconocido":
                if str(lanzador_real).strip().lower() in nombre.strip().lower():
                    es_lanzador_por_nombre = True
            
            if es_lanzador_por_nombre:
                zona = "LANZADOR"
            # Fallback posicional si no coincide nombre pero está en la esquina
            elif x > 96 and ((es_izq and y > 96) or (not es_izq and y < 4)):
                zona = "LANZADOR"

            # 2. A LA CORTA (Lógica Actualizada: X entre 75-100; Y extremos)
            elif zona == "SIN CLASIFICAR" and 75 <= x <= 100:
                if (es_izq and y >= 81) or (not es_izq and y <= 25):
                    zona = "CORTO"
                    n_corto += 1

            if zona != "SIN CLASIFICAR": pass

            # 3. ZONA ATRÁS
            elif x < 70:
                zona = "VIGILANCIA"
                n_atras += 1

            # 4. ZONA REMATE (ÁREA PEQUEÑA: X > 94.2)
            elif x > 94.2:
                if 45 <= y <= 55: zona = "A.PEQ CENTRO"
                elif (es_izq and 55 < y <= 70) or (not es_izq and 30 <= y < 45): zona = "A.PEQ 1ER PALO"
                elif (es_izq and 30 <= y < 45) or (not es_izq and 55 < y <= 70): zona = "A.PEQ 2DO PALO"
                else: zona = "A.PEQ OTRA"
                n_remate += 1

            # 5. ZONA REMATE (ÁREA GRANDE: 83 <= X <= 94.2)
            elif 83 <= x <= 94.2:
                if 45 <= y <= 55:
                    if x >= 88: zona = "CENTRO AREA"
                    else: zona = "PUNTO PENALTI"
                elif (es_izq and 55 < y <= 70) or (not es_izq and 30 <= y < 45): zona = "1ER PALO"
                elif (es_izq and 30 <= y < 45) or (not es_izq and 55 < y <= 70): zona = "2DO PALO"
                else: zona = "AREA GRANDE"
                n_remate += 1

            # 6. RECHACE (70 <= X < 83)
            elif 70 <= x < 83:
                if 36.8 <= y <= 63.2: zona = "RECHACE CENTRADO"
                else:
                    if es_izq:
                        if 63.2 < y <= 78.9: zona = "RECHACE CERCANO"
                        elif 21.1 <= y < 36.8: zona = "RECHACE LEJANO"
                        else: zona = "RECHACE"
                    else:
                        if 21.1 <= y < 36.8: zona = "RECHACE CERCANO"
                        elif 63.2 < y <= 78.9: zona = "RECHACE LEJANO"
                        else: zona = "RECHACE"
                n_rechace += 1
            
            zonas_asignadas.append(zona)

        atacantes['Zona_Detallada'] = zonas_asignadas
        stats_generales = [n_remate, n_rechace, n_corto, n_atras]
        
        return {}, stats_generales, atacantes
    
    def construir_historial_lanzadores(self):
        """
        Aprende qué suele hacer cada jugador en cada lado.
        """
        try:
            # Optimizamos: Solo miramos una fila por córner
            eventos_unicos = self.df_tracking.drop_duplicates(subset=['ID_Evento_Corner'])
            
            for _, row in eventos_unicos.iterrows():
                jugador = str(row.get('Nombre_Lanzador', 'Desconocido')).strip()
                tipo_raw = str(row.get('Tipo_Lanzamiento', ''))
                y_balon = row.get('Y_Balon', 50)
                
                if jugador in ["Desconocido", "None", "nan"]: continue
                
                lado = 'izquierda' if y_balon > 50 else 'derecha'
                
                tipo_real = None
                if 'In-swinger' in tipo_raw: tipo_real = 'Cerrado'
                elif 'Out-swinger' in tipo_raw: tipo_real = 'Abierto'
                
                if tipo_real:
                    if jugador not in self.historial_lanzadores:
                        self.historial_lanzadores[jugador] = {'izquierda': [], 'derecha': []}
                    self.historial_lanzadores[jugador][lado].append(tipo_real)
                    
        except Exception as e:
            print(f"⚠️ Error construyendo historial: {e}")

    def inferir_tipo_lanzamiento(self, jugador, lado):
        """
        Devuelve la MODA (lo más repetido) para ese jugador y lado.
        """
        jugador = str(jugador).strip()
        if jugador in self.historial_lanzadores:
            historial = self.historial_lanzadores[jugador].get(lado, [])
            if historial:
                # Devuelve el más común. Ej: ['Cerrado', 'Cerrado', 'Abierto'] -> 'Cerrado'
                return Counter(historial).most_common(1)[0][0]
        return "Neutro"

    def extract_posicionamiento_defensivo(self):
        if self.df_tracking is None: return

        
        # Contadores para saber qué falla
        stats = {
            'Total_Eventos': 0,
            'Fallos_Tiempo': 0,    # No existe el segundo -3.0
            'Solo_Atacantes': 0,   # No se detectan jugadores del otro equipo
            'No_Es_El_Equipo': 0,  # El equipo que defiende no es el seleccionado
            'Pocos_Defensas': 0,   # Hay menos de 8 defensores trackeados
            'Aceptados': 0
        }

        # Iteramos por cada córner único
        ids_eventos = self.df_tracking[['ID_Evento_Corner', 'ID_Partido']].drop_duplicates().values
        stats['Total_Eventos'] = len(ids_eventos)
        
        self.datos_contextuales = [] 
        
        # Limite de impresiones detalladas para no saturar la consola
        debug_prints = 0
        
        for id_evento, id_partido in ids_eventos:
            # Filtramos filas de este evento (Toda la secuencia temporal)
            df_corner = self.df_tracking[
                (self.df_tracking['ID_Evento_Corner'] == id_evento) & 
                (self.df_tracking['ID_Partido'] == id_partido)
            ]
            
            # --- DEBUG 1: CHEQUEO DE TIEMPO ---
            # Buscamos el frame más cercano a -3.0 (momento estático para la foto)
            OBJETIVO_TIEMPO = -3.0 
            idx_analisis = (df_corner['Segundos_Desde_Saque'] - OBJETIVO_TIEMPO).abs().idxmin()
            tiempo_encontrado = df_corner.loc[idx_analisis, 'Segundos_Desde_Saque']
            
            # Si difiere en más de 0.5s, lo descartamos
            if abs(tiempo_encontrado - OBJETIVO_TIEMPO) > 0.5:
                stats['Fallos_Tiempo'] += 1
                if debug_prints < 5:
                    print(f"❌ [ID {id_evento}] Fallo Tiempo: Buscaba {OBJETIVO_TIEMPO}, encontró {tiempo_encontrado:.2f}")
                    debug_prints += 1
                continue

            # Frame exacto (FOTO)
            frame_analisis = df_corner[df_corner['Segundos_Desde_Saque'] == tiempo_encontrado].copy()
            frame_analisis = frame_analisis.drop_duplicates(subset=['Nombre_Jugador_Tracking'])
            
            if frame_analisis.empty: continue
            
            # --- DEBUG 2: ROLES ---
            row_info = frame_analisis.iloc[0]
            nombre_atacante = str(row_info['Equipo_Lanzador']).strip()
            
            # Buscamos defensor
            _, nombre_defensor = self.get_roles_equipos(frame_analisis)
            
            if not nombre_defensor: 
                stats['Solo_Atacantes'] += 1
                if debug_prints < 10:
                    equipos_vistos = frame_analisis['NombreEquipoJugador_Tracking'].unique()
                    print(f"⚠️ [ID {id_evento}] Solo veo atacantes. Lanzador: {nombre_atacante}. Vistos en tracking: {equipos_vistos}")
                    debug_prints += 1
                continue

            # --- DEBUG 3: FILTRO DE NOMBRE ---
            # Comparamos ignorando mayúsculas y espacios extra
            target = str(self.team_filter).strip().lower()
            actual = str(nombre_defensor).strip().lower()

            if target not in actual and actual not in target:
                stats['No_Es_El_Equipo'] += 1
                continue

            # --- DEBUG 4: CANTIDAD JUGADORES ---
            df_defensores = frame_analisis[
                frame_analisis['NombreEquipoJugador_Tracking'] != nombre_atacante
            ].copy()

            if len(df_defensores) < 8: 
                stats['Pocos_Defensas'] += 1
                if debug_prints < 15:
                    pass
                    debug_prints += 1
                continue

            # --- SI LLEGA AQUI, ES UN EXITO ---
            stats['Aceptados'] += 1
            
            # 1. Metadatos básicos
            y_balon = row_info.get('Y_Balon', 50)
            lado = self.determinar_lado_corner(y_balon)
            tipo_lanz = str(row_info.get('Tipo_Lanzamiento', 'Mixto'))
            nombre_lanzador = str(row_info.get('Nombre_Lanzador', 'Desconocido'))
            dorsal_lanzador = str(row_info.get('Dorsal_Lanzador', '?'))

            if 'In-swinger' in tipo_lanz: perfil = 'Cerrado'
            elif 'Out-swinger' in tipo_lanz: perfil = 'Abierto'
            else: perfil = self.inferir_tipo_lanzamiento(nombre_lanzador, lado)

            df_atacantes = frame_analisis[frame_analisis['NombreEquipoJugador_Tracking'] == nombre_atacante].copy()
            
            # 2. Estandarizar Dorsales
            col_dorsal = 'Dorsal_Jugador_Tracking' if 'Dorsal_Jugador_Tracking' in df_defensores.columns else 'Dorsal'
            if col_dorsal not in df_defensores.columns: df_defensores['Dorsal'] = "?"
            
            # 3. Analizar Ataque (Zonas)
            _, stats_gen, df_att_processed = self.analizar_disposicion_ofensiva(
                df_atacantes, nombre_atacante, lado, lanzador_real=nombre_lanzador
            )
            stats_con_lado = stats_gen + [10 if lado == 'izquierda' else 0]

            # 4. NUEVO: Analizar Tipo de Marcaje (Zona vs Hombre) usando la SECUENCIA (df_corner)
            #    Se usa df_corner (video) y no frame_analisis (foto) para ver movimiento
            mapa_marcajes = self.analizar_tipo_marcaje(df_corner, nombre_defensor, nombre_atacante)

            # 5. Preparar DataFrame Defensivo
            pos_def = df_defensores[['Nombre_Jugador_Tracking', col_dorsal, 'X_Jugador', 'Y_Jugador']].copy()
            pos_def.columns = ['Nombre_Jugador', 'Dorsal', 'X_Jugador', 'Y_Jugador']
            
            # Asignar columna de marcaje (si no se detectó, por defecto 'Zona')
            pos_def['Tipo_Marcaje'] = pos_def['Nombre_Jugador'].map(mapa_marcajes).fillna('Zona')

            # 6. Guardar en acumuladores
            if lado == 'izquierda':
                self.posicionamiento_izquierda = pd.concat([self.posicionamiento_izquierda, pos_def])
            else:
                self.posicionamiento_derecha = pd.concat([self.posicionamiento_derecha, pos_def])

            self.datos_contextuales.append({
                'defensa': pos_def,
                'ataque_df': df_att_processed, 
                'ataque_stats_generales': stats_con_lado, 
                'lado': lado,
                'perfil': perfil,
                'lanzador_nombre': nombre_lanzador, 
                'lanzador_dorsal': dorsal_lanzador,
                'id': id_evento
            })

    def get_scenarios_exactos(self, min_repeticiones=1):
        """
        Agrupa jugadas idénticas. Min_repeticiones=1 asegura que salgan jugadas únicas.
        """
        grupos = {} 
        for i, dato in enumerate(self.datos_contextuales):
            stats = tuple(dato['ataque_stats_generales']) 
            if stats not in grupos: grupos[stats] = []
            grupos[stats].append(i)
            
        lista_escenarios = []
        for stats, indices in grupos.items():
            lista_escenarios.append({
                'stats': stats,
                'indices': indices,
                'count': len(indices)
            })
            
        lista_escenarios.sort(key=lambda x: x['count'], reverse=True)
        filtrados = [e for e in lista_escenarios if e['count'] >= min_repeticiones]
        
        if len(filtrados) == 0: return lista_escenarios[:6]
        return filtrados

    def _ajustar_solapamientos(self, df, x_col, y_col, radio_minimo=2.5):
        coords = df[[x_col, y_col]].values.copy()
        n = len(coords)
        for _ in range(5): 
            for i in range(n):
                for j in range(i + 1, n):
                    x1, y1 = coords[i]
                    x2, y2 = coords[j]
                    dist = np.hypot(x1 - x2, y1 - y2)
                    if dist < radio_minimo:
                        if dist == 0: dx, dy = np.random.uniform(-1, 1), np.random.uniform(-1, 1); dist = 0.1
                        else: dx, dy = (x1 - x2) / dist, (y1 - y2) / dist
                        overlap = (radio_minimo - dist) / 2
                        coords[i][0] += dx * overlap; coords[i][1] += dy * overlap
                        coords[j][0] -= dx * overlap; coords[j][1] -= dy * overlap
                coords[i][0] = max(2, min(98, coords[i][0]))
                coords[i][1] = max(2, min(98, coords[i][1]))
        df_ajustado = df.copy()
        df_ajustado[x_col] = coords[:, 0]
        df_ajustado[y_col] = coords[:, 1]
        return df_ajustado

    def get_defensive_clusters(self, lado='izquierda', n_clusters=11):
        df = self.posicionamiento_izquierda if lado == 'izquierda' else self.posicionamiento_derecha
        if df.empty or len(df) < n_clusters: return pd.DataFrame()

        X = df[['X_Jugador', 'Y_Jugador']].values
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
            df['Cluster_ID'] = kmeans.labels_
            centroids = kmeans.cluster_centers_
        except: return pd.DataFrame()
        
        zonas_info = []
        for i, (cx, cy) in enumerate(centroids):
            datos_zona = df[df['Cluster_ID'] == i]
            if not datos_zona.empty:
                top_jugadores = datos_zona['Nombre_Jugador'].value_counts()
                jugador_p = top_jugadores.index[0]
                count = top_jugadores.iloc[0]
                pct = (count / len(datos_zona)) * 100
                dorsal = datos_zona[datos_zona['Nombre_Jugador'] == jugador_p]['Dorsal'].mode()
                d_str = str(int(dorsal[0])) if not dorsal.empty and pd.notna(dorsal[0]) else "?"
            else:
                jugador_p, d_str, pct = "Desc.", "?", 0

            zonas_info.append({
                'Zona_ID': i + 1, 'x': cx, 'y': cy,
                'Jugador_Principal': jugador_p, 'Dorsal': d_str, 'Ocupacion_Pct': pct
            })
        return pd.DataFrame(zonas_info)

    def create_reporte_contextual_grid(self):
        if not hasattr(self, 'datos_contextuales') or not self.datos_contextuales:
            print("❌ No hay datos para generar el reporte.")
            return None

        escenarios = self.get_scenarios_exactos(min_repeticiones=1)
        if not escenarios: return None
        escenarios_a_dibujar = escenarios[:5]
        
        fig = plt.figure(figsize=(16, 10), facecolor='white')
        
        # --- FONDOS Y LOGOS ---
        if (bg := self.load_background()) is not None:
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(bg, extent=[0, 1, 0, 1], aspect='auto', alpha=0.25)
            ax_bg.axis('off')

        if self.team_filter and (logo := self.load_team_logo(self.team_filter)) is not None:
            ax_team = fig.add_axes([0.88, 0.88, 0.08, 0.10], zorder=1)
            ax_team.imshow(logo, aspect='auto')
            ax_team.axis('off')

        if (icon := self.load_custom_icon()) is not None:
            ax_icon = fig.add_axes([0.04, 0.88, 0.08, 0.10], zorder=1)
            ax_icon.imshow(icon, aspect='auto')
            ax_icon.axis('off')

        gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.15, top=0.82, bottom=0.05, left=0.05, right=0.95)
        
        total_corners = len(self.datos_contextuales)
        fig.suptitle(f"ANÁLISIS POSICIONAMIENTO DEFENSIVO - {self.team_filter}", 
                     fontsize=18, fontweight='bold', color='#1e3d59', y=0.95)

        # --- LEYENDA FLOTANTE ENTRE FILAS ---
        ax_legend_mid = fig.add_axes([0.30, 0.47, 0.40, 0.04], zorder=10)
        ax_legend_mid.axis('off')
        ax_legend_mid.text(0.15, 0.5, "LEYENDA MARCAJES:", ha='right', va='center', fontsize=10, fontweight='bold', color='#1e3d59')
        ax_legend_mid.scatter(0.25, 0.5, s=200, color='#001f3f', edgecolors='none')
        ax_legend_mid.text(0.28, 0.5, "ZONA", ha='left', va='center', fontsize=9, fontweight='bold', color='#333')
        ax_legend_mid.scatter(0.55, 0.5, s=200, color='#5DADE2', edgecolors='none')
        ax_legend_mid.text(0.58, 0.5, "HOMBRE", ha='left', va='center', fontsize=9, fontweight='bold', color='#333')

        img_balon = None
        if os.path.exists("assets/balon.png"):
            img_balon = plt.imread("assets/balon.png")

        # --- BUCLE DE ESCENARIOS (CAMPOGRAMAS) ---
        for i, esc in enumerate(escenarios_a_dibujar):
            row, col = divmod(i, 3)
            ax = fig.add_subplot(gs[row, col])
            indices = esc['indices']
            data_cluster = [self.datos_contextuales[idx] for idx in indices]
            n_corners = len(indices)

            # Contexto
            lado_val = data_cluster[0]['lado']
            perfiles = [d['perfil'] for d in data_cluster]
            from collections import Counter
            perfil_comun = Counter(perfiles).most_common(1)[0][0]

            pitch = VerticalPitch(pitch_type='opta', half=True, pitch_color='#2d5016', line_color='white')
            pitch.draw(ax=ax)

            points_to_plot = []
            zonas_presentes_titulo = {}
            df_all_att = pd.DataFrame() # Inicializar variable para usarla luego

            # 1. BALÓN
            if lado_val == 'izquierda': x_balon, y_balon = 100, 100
            else: x_balon, y_balon = 0, 100
            
            if img_balon is not None:
                imagebox = OffsetImage(img_balon, zoom=0.018) 
                ab = AnnotationBbox(imagebox, (x_balon, y_balon), frameon=False, zorder=20)
                ax.add_artist(ab)
            else:
                pitch.scatter(x_balon, y_balon, ax=ax, s=100, c='white', edgecolors='black', zorder=20)

            # 2. LANZADOR
            y_jugador = 103 if lado_val == 'izquierda' else -3
            if 'Cerrado' in perfil_comun: x_jugador = 103
            elif 'Abierto' in perfil_comun: x_jugador = 97
            else: x_jugador = 100
            pitch.scatter(x_jugador, y_jugador, ax=ax, s=75, c='#FFD700', marker='^', edgecolors='black', lw=1.5, zorder=20)

            # 3. ATACANTES
            try:
                df_all_att = pd.concat([item['ataque_df'] for item in data_cluster])
                conteo_zonas = df_all_att['Zona_Detallada'].value_counts()
                for zona, total_count in conteo_zonas.items():
                    n_avg = int(round(total_count / n_corners))
                    if n_avg > 0:
                        zonas_presentes_titulo[zona] = n_avg
                        if zona == "LANZADOR": continue

                        df_zona = df_all_att[df_all_att['Zona_Detallada'] == zona]
                        X_zona = df_zona[['X_Jugador', 'Y_Jugador']].values
                        if len(df_zona) >= n_avg:
                            km_local = KMeans(n_clusters=n_avg, random_state=42, n_init=10).fit(X_zona)
                            centros = km_local.cluster_centers_
                        else: centros = X_zona 
                        for cx, cy in centros:
                            if cx >= 50:
                                points_to_plot.append({'x': cx, 'y': cy, 'type': 'att', 'label': '', 'zona': zona})
            except: pass

            # 4. DEFENSORES
            try:
                df_def = pd.concat([item['defensa'] for item in data_cluster])
                conteo_jugadores = df_def['Nombre_Jugador'].value_counts()
                top11_nombres = conteo_jugadores.head(11).index.tolist()
                for nombre in top11_nombres:
                    datos_jugador = df_def[df_def['Nombre_Jugador'] == nombre]
                    cx, cy = datos_jugador['X_Jugador'].mean(), datos_jugador['Y_Jugador'].mean()
                    dorsal = datos_jugador['Dorsal'].mode()
                    d_str = str(int(dorsal[0])) if not dorsal.empty and pd.notna(dorsal[0]) else "?"
                    
                    modo_marcaje = datos_jugador['Tipo_Marcaje'].mode()
                    es_hombre = False
                    if not modo_marcaje.empty and modo_marcaje[0] == 'Hombre':
                        es_hombre = True
                    
                    if es_hombre:
                        color_bg = '#5DADE2'  # Azul Claro
                        color_txt = 'white'
                    else:
                        color_bg = '#001f3f'  # Azul Oscuro
                        color_txt = 'white'

                    points_to_plot.append({
                        'x': cx, 'y': cy, 'type': 'def', 'label': d_str, 
                        'zona': 'defensa', 'color': color_bg, 'text_color': color_txt
                    })
            except: pass

            # 5. DIBUJAR PUNTOS
            df_points = pd.DataFrame(points_to_plot)
            if not df_points.empty:
                df_points = self._ajustar_solapamientos(df_points, 'x', 'y', radio_minimo=2.5)
                for _, p in df_points.iterrows():
                    if p['type'] == 'att':
                        pitch.scatter(p['x'], p['y'], ax=ax, s=45, color='#E74C3C', edgecolors='white', alpha=0.9, zorder=12) 
                    else:
                        c_bg = p.get('color', '#3498DB')
                        c_txt = p.get('text_color', 'white')
                        pitch.scatter(p['x'], p['y'], ax=ax, s=60, color=c_bg, edgecolors='none', lw=0, zorder=11)
                        pitch.annotate(p['label'], (p['x'], p['y']), ax=ax, ha='center', va='center', 
                                     fontsize=5.5, fontweight='bold', color=c_txt, zorder=13)

            # 6. TÍTULO
            lado_str = "IZQ" if lado_val == 'izquierda' else "DER"
            subtitulo_perfil = f" - {perfil_comun.upper()}" if perfil_comun != "Desc." else ""
            
            detalles_str = []
            orden_prioridad = ["CORTO", "A.PEQ 1ER PALO", "A.PEQ 2DO PALO", "A.PEQ CENTRO", "1ER PALO", "2DO PALO", "PUNTO PENALTI", "CENTRO AREA", "RECHACE CERCANO", "RECHACE LEJANO"]
            for z in orden_prioridad:
                if z in zonas_presentes_titulo:
                    detalles_str.append(f"{zonas_presentes_titulo[z]} {z.replace('RECHACE ','rech ').replace('A.PEQ ','AP ').lower()}")
            for z, n in zonas_presentes_titulo.items():
                if z not in orden_prioridad and "SIN CLASIFICAR" not in z and "LANZADOR" not in z:
                    detalles_str.append(f"{n} {z.lower()}")
            
            titulo_final = f"SAQUE {lado_str}{subtitulo_perfil} (n={n_corners})\n{textwrap.fill(', '.join(detalles_str), width=32)}"
            ax.set_title(titulo_final, fontsize=8, fontweight='bold', backgroundcolor='#f0f0f0', pad=6)

        # --- 7. TABLA DETALLADA CON ROLES Y PORCENTAJES ---
        try:
            ax_ley = fig.add_subplot(gs[1, 2])
            ax_ley.axis('off')
            ax_ley.set_title("ROL DEFENSIVO Y ZONAS (% TIEMPO)", fontweight='bold', fontsize=10, pad=35, color='#1e3d59')
            
            all_data = []
            for dato in self.datos_contextuales:
                lado_saque = dato['lado']
                es_izq = (lado_saque == 'izquierda') 
                for _, row in dato['defensa'].iterrows():
                    x, y = row['X_Jugador'], row['Y_Jugador']
                    tipo_marcaje = row.get('Tipo_Marcaje', 'Zona')

                    zona = "OTRO"
                    if x < 75: zona = "ARRIBA"
                    elif 75 <= x <= 100 and ((es_izq and y >= 81) or (not es_izq and y <= 25)): 
                        zona = "A LA CORTA"
                    elif x > 94.2:
                        if 45 <= y <= 55: zona = "AP CENTRO"
                        else: zona = "AP 1ER PALO" if ((es_izq and y > 55) or (not es_izq and y < 45)) else "AP 2DO PALO"
                    elif 90 <= x <= 94.2:
                        if 45 <= y <= 55: zona = "PENALTI"
                        else: zona = "1ER PALO" if ((es_izq and y > 55) or (not es_izq and y < 45)) else "2DO PALO"
                    elif 75 <= x < 90:
                        if 36.8 <= y <= 63.2: zona = "RECHACE CEN"
                        else: zona = "RECHACE CER" if ((es_izq and y > 63.2) or (not es_izq and y < 36.8)) else "RECHACE LEJ"
                    
                    all_data.append({
                        'Nombre': row['Nombre_Jugador'], 
                        'Dorsal': row['Dorsal'], 
                        'Zona': zona,
                        'Tipo_Marcaje': tipo_marcaje
                    })
            
            df_total = pd.DataFrame(all_data)
            if not df_total.empty:
                top_p = df_total['Nombre'].value_counts().index
                df_filtered = df_total[df_total['Nombre'].isin(top_p)]
                d_map = df_filtered.groupby('Nombre')['Dorsal'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "?")
                
                pivot_zones = pd.crosstab(df_filtered['Nombre'], df_filtered['Zona'], normalize='index') * 100
                pivot_roles = pd.crosstab(df_filtered['Nombre'], df_filtered['Tipo_Marcaje'], normalize='index') * 100
                
                if 'Zona' not in pivot_roles.columns: pivot_roles['Zona'] = 0
                if 'Hombre' not in pivot_roles.columns: pivot_roles['Hombre'] = 0
                
                pivot = pivot_zones.round(0).astype(int)
                pivot_roles = pivot_roles.round(0).astype(int)
                
                n_rows, n_cols_zones = pivot.shape
                n_cols_totales = n_cols_zones + 2 
                
                ROW_H, F_ROW, F_HEAD = (1.0, 6, 5) if n_rows > 15 else (1.4, 7, 5)
                
                HEAD_H, off_x = 5.8, 13.5
                
                tot_w, tot_h = n_cols_totales + off_x + 1.0, (n_rows * ROW_H) + HEAD_H
                ax_ley.set_xlim(0, tot_w); ax_ley.set_ylim(0, tot_h)
                
                y_base = n_rows * ROW_H

                # Rectángulo Cabecera
                ax_ley.add_patch(Rectangle((0, y_base + 0.2), tot_w, HEAD_H, color='#1e3d59', zorder=0))

                # --- CABECERAS DE COLUMNA ---
                ax_ley.text(0.5, y_base + 1.0, "#", color='white', fontweight='bold', fontsize=F_HEAD+1, ha='center')
                ax_ley.text(2.0, y_base + 1.0, "JUGADOR", color='white', fontweight='bold', fontsize=F_HEAD+1, ha='left')
                
                # Columnas Porcentaje
                ax_ley.text(9.0, y_base + 0.5, "% ZONA", color='#AED6F1', fontweight='bold', fontsize=F_HEAD, ha='center', rotation=50)
                ax_ley.text(11.5, y_base + 0.5, "% HOMB", color='#F9E79F', fontweight='bold', fontsize=F_HEAD, ha='center', rotation=50)

                col_ord = ["ARRIBA", "RECHACE LEJ", "RECHACE CEN", "RECHACE CER", "PENALTI", "A LA CORTA", "1ER PALO", "CENTRO AREA", "2DO PALO", "AP 1ER PALO", "AP CENTRO", "AP 2DO PALO"]
                cols_fin = [c for c in col_ord if c in pivot.columns] + [c for c in pivot.columns if c not in col_ord]
                pivot = pivot[cols_fin].loc[top_p]
                pivot_roles = pivot_roles.loc[top_p]

                for i, col in enumerate(cols_fin):
                    ax_ley.text(i + off_x, y_base + 0.5, col, color='white', fontweight='bold', fontsize=F_HEAD, ha='left', va='bottom', rotation=50)

                # FILAS
                for r_idx, (p_name, r_data) in enumerate(pivot.iloc[::-1].iterrows()):
                    yb = r_idx * ROW_H; yc = yb + (ROW_H/2)
                    if r_idx % 2 == 0: ax_ley.add_patch(Rectangle((0, yb), tot_w, ROW_H, color='#f4f4f4', zorder=-1))
                    
                    # Dorsal
                    d_val = str(int(d_map[p_name])) if str(d_map[p_name]).replace('.','').isdigit() else "?"
                    ax_ley.text(0.5, yc, d_val, ha='center', va='center', fontsize=F_ROW, fontweight='bold', color='#333')
                    
                    # Nombre (Abreviado)
                    parts = p_name.split()
                    if len(parts) > 2: parts = parts[:2]
                    temp_name = " ".join(parts)
                    display_name = temp_name
                    if len(temp_name) > 10 and len(parts) >= 2:
                        display_name = f"{parts[0][0]}. {parts[1]}"
                    
                    ax_ley.text(2.0, yc, textwrap.fill(display_name, width=25), ha='left', va='center', fontsize=F_ROW, fontfamily='monospace', color='#333')
                    
                    # Porcentajes
                    pct_zona = pivot_roles.loc[p_name, 'Zona']
                    pct_hombre = pivot_roles.loc[p_name, 'Hombre']
                    
                    if pct_zona > 0:
                        ax_ley.text(9.0, yc, f"{pct_zona}%", ha='center', va='center', fontsize=F_ROW-1, color='#001f3f', fontweight='bold')
                    else: ax_ley.text(9.0, yc, "-", ha='center', va='center', color='#ccc', fontsize=F_ROW-1)

                    if pct_hombre > 0:
                        ax_ley.text(11.5, yc, f"{pct_hombre}%", ha='center', va='center', fontsize=F_ROW-1, color='#B7950B', fontweight='bold')
                    else: ax_ley.text(11.5, yc, "-", ha='center', va='center', color='#ccc', fontsize=F_ROW-1)

                    # Zonas
                    for c_idx, val in enumerate(r_data):
                        x = c_idx + off_x
                        if val > 0:
                            a = 0.2 + (val/100)*0.8
                            ax_ley.add_patch(Rectangle((x-0.4, yc-0.4), 0.8, 0.8, color='#3498DB', alpha=a, zorder=1))
                            ax_ley.text(x, yc, str(val), ha='center', va='center', fontsize=F_ROW-1, color='white' if a>0.6 else 'black', fontweight='bold', zorder=2)
                        else: ax_ley.text(x, yc, "-", ha='center', va='center', fontsize=F_ROW-1, color='#ccc')
        except Exception as e:
            pass

        return fig

    def create_campograma_defensivo(self, ax, lado='izquierda'):
        pitch = VerticalPitch(pitch_type='opta', half=True, pitch_color='#2d5016', line_color='white', linewidth=2)
        pitch.draw(ax=ax)
        
        df_zonas = self.get_defensive_clusters(lado, n_clusters=11)
        if df_zonas.empty:
            ax.text(50, 50, 'Sin datos', ha='center', color='white')
            return

        df_zonas = df_zonas.sort_values('x')
        cmap = plt.cm.get_cmap('tab20', len(df_zonas))

        for i, (_, zona) in enumerate(df_zonas.iterrows()):
            color_zona = cmap(i)
            pitch.scatter(zona['x'], zona['y'], ax=ax, s=150, color=color_zona, edgecolors='black', linewidth=1, alpha=1, zorder=10)

        ax.set_title(f"ESTRUCTURA DEFENSIVA ({lado.upper()})", fontsize=14, color='#1e3d59', fontweight='bold', pad=10)

    def create_tabla_zonas(self, ax, lado='izquierda'):
        ax.axis('off')
        df_zonas = self.get_defensive_clusters(lado, n_clusters=11)
        if df_zonas.empty: return

        df_zonas = df_zonas.sort_values('x')
        cmap = plt.cm.get_cmap('tab20', len(df_zonas))
        
        ax.text(0.5, 1.0, f'OCUPACIÓN DE ZONAS', ha='center', va='top', fontsize=12, fontweight='bold', color='#1e3d59', transform=ax.transAxes)
        ax.text(0.10, 0.92, 'Color', fontweight='bold', fontsize=9, transform=ax.transAxes, ha='center')
        ax.text(0.25, 0.92, 'Jugador Principal', fontweight='bold', fontsize=9, transform=ax.transAxes, ha='left')
        ax.text(0.85, 0.92, '% Ocup.', fontweight='bold', fontsize=9, transform=ax.transAxes, ha='center')
        ax.axhline(y=0.90, xmin=0.05, xmax=0.95, color='black', linewidth=0.5)

        y_pos = 0.84
        for i, (_, row) in enumerate(df_zonas.iterrows()):
            color_zona = cmap(i)
            ax.add_patch(plt.Circle((0.10, y_pos), 0.020, color=color_zona, ec='black', transform=ax.transAxes))
            ax.text(0.25, y_pos, f"#{row['Dorsal']} {row['Jugador_Principal']}", ha='left', va='center', fontsize=9, transform=ax.transAxes, color='#333')
            ax.text(0.85, y_pos, f"{int(row['Ocupacion_Pct'])}%", ha='center', va='center', fontsize=9, fontweight='bold', color='#555', transform=ax.transAxes)
            if y_pos > 0.1: ax.axhline(y=y_pos - 0.035, xmin=0.1, xmax=0.9, color='#eee', linewidth=1)
            y_pos -= 0.075

    def load_team_logo(self, team_name, target_size=(60, 80)):
        try:
            if self.team_stats is not None:
                row = self.team_stats[self.team_stats['Team Name'] == team_name]
                if not row.empty and (b64 := row.iloc[0].get('Team Logo')):
                    img = Image.open(BytesIO(base64.b64decode(b64))).convert('RGBA')
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    return np.array(img) / 255.0
        except: pass
        
        folder = "assets/escudos/"
        if not os.path.exists(folder): return None

        # Quitamos palabras comunes para que encuentre "Real Sociedad"
        def clean(t): return set(re.sub(r'[^a-z0-9\s]', '', unicodedata.normalize('NFKD', str(t).lower())).split()) - \
                           {'fc','cf','rc','cd','ud','sad','club','de','del','la','el','los','balompie'}
        
        target = clean(team_name)
        best, max_match = None, 0
        for f in os.listdir(folder):
            if not f.endswith(('.png','.jpg')): continue
            curr = clean(os.path.splitext(f)[0])
            common = len(target & curr)
            if common > max_match: max_match, best = common, f
            
        if not best:
            manual = {
                'celta':'RC Celta.png',
                'alaves':'Deportivo Alaves.png',
                'betis':'Real Betis.png',
                'athletic':'Athletic Club.png',
                'real sociedad': 'Real Sociedad.png'
            }
            for k,v in manual.items(): 
                if k in team_name.lower() and os.path.exists(folder+v): best = v; break
        
        return plt.imread(folder+best) if best else None

    def load_background(self):
        # Esta es la función que te faltaba
        return plt.imread("assets/fondo_informes.png") if os.path.exists("assets/fondo_informes.png") else None

    def guardar_sin_espacios(self, fig, filename):
        fig.set_size_inches(11.69, 8.27)
        fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white', transparent=False, orientation='landscape')

def seleccionar_equipo_interactivo():
    try:
        pass
        df = pd.read_parquet("extraccion_sportian/corners_tracking.parquet")
        col = 'NombreEquipoJugador_Tracking'
        
        if col not in df.columns:
            print(f"❌ Error: No existe '{col}'")
            return None
            
        # Aseguramos que sean strings y ordenamos
        equipos = sorted([str(e) for e in df[col].dropna().unique()])
        
        for i, e in enumerate(equipos, 1): print(f"{i}. {e}")
            
        while True:
            # .strip() elimina espacios accidentales al principio o final
            sel = input("Selecciona número o escribe nombre: ").strip()
            
            if not sel: return None
            
            # --- NUEVA LÓGICA ---
            # 1. Si el input coincide exactamente con un nombre de la lista
            if sel in equipos:
                return sel
            
            # 2. Si el input es un número (lógica original)
            try:
                idx = int(sel) - 1
                if 0 <= idx < len(equipos): return equipos[idx]
            except ValueError: 
                pass
            
            print("❌ Selección no válida. Introduce un número de la lista o el nombre exacto.")
            
    except Exception as e: print(f"❌ Error: {e}"); return None

def main():
    pass
    equipo = seleccionar_equipo_interactivo()
    if not equipo: return
    
    analyzer = ReporteDefensivoCorners(team_filter=equipo)
    
    # --- AÑADE ESTA LÍNEA AQUÍ ---
    analyzer.diagnostico_equipos()
    # -----------------------------
    
    fig = analyzer.create_reporte_contextual_grid()
    
    if fig:
        filename = f"reporte_defensivo_contextual_{equipo.replace(' ', '_')}.pdf"
        analyzer.guardar_sin_espacios(fig, filename)
        plt.show()
    else:
        print("❌ No se pudo generar el gráfico.")

if __name__ == "__main__":
    main()