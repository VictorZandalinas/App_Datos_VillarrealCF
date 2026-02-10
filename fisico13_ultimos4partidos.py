import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import re
import unicodedata
import os
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

import matplotlib.patheffects as patheffects

# 🔥 CONFIGURACIÓN GLOBAL AGRESIVA (COPIADA DEL PRIMER SCRIPT)
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

try:
    from mplsoccer import Pitch
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "mplsoccer"])
    from mplsoccer import Pitch

class ReporteTactico4CamposHorizontalesMejorado:
    def __init__(self, data_path="extraccion_mediacoach/data/rendimiento_fisico.parquet"):
        """
        Inicializa la clase para generar reportes tácticos con 4 campos horizontales
        CON COORDENADAS FIJAS (SIN AUTO-MOVIMIENTO)
        """
        self.data_path = data_path
        self.df = None
        self.load_data()
        self.clean_team_names()
        self.opta_df = None
        self.load_match_events()
        
        
        # ✅ MÉTRICAS COMPLETAS
        self.metricas_tabla = [
            'Distancia Total',
            'Distancia Total / min',
            'Distancia Total 14-21 km / h',
            'Distancia Total 14-21 km / h / min',
            'Distancia Total >21 km / h', 
            'Distancia Total >21 km / h / min',
            'Distancia Total >24 km / h',
            'Distancia Total >24 km / h / min',
            'Velocidad Máxima Total'
        ]

        # MAPEO DE FORMACIONES OPTA
        self.formation_mapping = {
            1: "not_in_use", 2: "442", 3: "41212", 4: "433", 5: "451",
            6: "4411", 7: "4141", 8: "4231", 9: "4321", 10: "532",
            11: "541", 12: "352", 13: "343", 15: "4222", 16: "3511",
            17: "3421", 18: "3412", 19: "3142", 20: "343d", 21: "4132",
            22: "4240", 23: "4312", 24: "3241", 25: "3331"
        }

        self.formation_coordinates = {
            2: {  # 442
                1: (10, 40), 2: (40, 15), 3: (40, 65), 5: (25, 30), 6: (25, 50),
                7: (70, 15), 11: (70, 65), 4: (65, 30), 8: (65, 50),
                10: (100, 30), 9: (100, 50)
            },
            3: {  # 41212 (Diamond)
                1: (10, 40), 2: (40, 15), 3: (40, 65), 5: (25, 30), 6: (25, 50),
                4: (50, 40), 7: (70, 15), 11: (70, 65), 8: (85, 40),
                10: (105, 30), 9: (105, 50)
            },
            4: {  # 433
                1: (10, 40), 2: (40, 15), 3: (40, 65), 5: (25, 30), 6: (25, 50),
                4: (55, 40), 7: (70, 20), 8: (70, 60),
                10: (90, 15), 11: (90, 65), 9: (105, 40)
            },
            5: {  # 451
                1: (10, 40), 2: (40, 15), 3: (40, 65), 5: (25, 30), 6: (25, 50),
                7: (65, 10), 11: (65, 70), 4: (62, 28), 10: (45, 40), 8: (62, 52),
                9: (100, 40)
            },
            6: {  # 4411
                1: (10, 40), 2: (40, 15), 3: (40, 65), 5: (25, 30), 6: (25, 50),
                7: (70, 15), 11: (70, 65), 4: (65, 30), 8: (65, 50),
                10: (85, 40), 9: (105, 40)
            },
            7: {  # 4141
                1: (10, 40), 2: (40, 15), 3: (40, 65), 5: (25, 28), 6: (25, 52),
                4: (50, 40), 7: (75, 15), 11: (75, 65), 8: (75, 30), 10: (75, 50),
                9: (100, 40)
            },
            8: {  # 4231
                1: (10, 40), 2: (40, 15), 3: (40, 65), 5: (25, 28), 6: (25, 52),
                4: (55, 30), 8: (55, 50), 10: (75, 40),
                7: (75, 15), 11: (75, 65), 9: (105, 40)
            },
            9: {  # 4321
                1: (10, 40), 2: (40, 15), 3: (40, 65), 5: (25, 28), 6: (25, 52),
                8: (65, 25), 4: (65, 40), 7: (65, 55),
                10: (85, 30), 11: (85, 50), 9: (105, 40)
            },
            10: {  # 532
                1: (10, 40), 2: (45, 10), 3: (45, 70), 4: (30, 55), 5: (30, 25), 6: (30, 40),
                7: (70, 25), 8: (70, 55), 10: (70, 40),
                11: (100, 50), 9: (100, 30)
            },
            11: {  # 541
                1: (10, 40), 2: (45, 10), 3: (45, 70), 4: (30, 55), 5: (30, 25), 6: (30, 40),
                7: (70, 15), 11: (70, 65), 8: (70, 35), 10: (70, 45),
                9: (100, 40)
            },
            12: {  # 352
                1: (10, 40), 2: (45, 10), 3: (45, 70), 4: (30, 55), 5: (30, 25), 6: (30, 40),
                7: (70, 25), 11: (70, 40), 8: (70, 55),
                10: (100, 30), 9: (100, 50)
            },
            13: {  # 343
                1: (10, 40), 2: (45, 10), 3: (45, 70), 4: (30, 55), 5: (30, 25), 6: (30, 40),
                7: (65, 25), 8: (65, 55),
                10: (95, 15), 11: (95, 65), 9: (95, 40)
            },
            15: {  # 4222
                1: (10, 40), 2: (40, 15), 3: (40, 65), 5: (25, 28), 6: (25, 52),
                4: (60, 30), 8: (60, 50),
                7: (80, 20), 11: (80, 60), 10: (100, 25), 9: (100, 55)
            },
            16: {  # 3511
                1: (10, 40), 2: (45, 10), 3: (45, 70), 4: (30, 55), 5: (30, 25), 6: (30, 40),
                7: (65, 25), 11: (65, 40), 8: (65, 55),
                10: (85, 40), 9: (105, 40)
            },
            17: {  # 3421
                1: (10, 40), 2: (45, 10), 3: (45, 70), 4: (30, 55), 5: (30, 25), 6: (30, 40),
                7: (65, 25), 8: (65, 55), 10: (85, 30), 11: (85, 50),
                9: (105, 40)
            },
            18: {  # 3412
                1: (10, 40), 2: (45, 10), 3: (45, 70), 4: (30, 55), 5: (30, 25), 6: (30, 40),
                7: (65, 25), 8: (65, 55), 9: (85, 40),
                10: (100, 30), 11: (100, 50)
            },
            19: {  # 3142
                1: (10, 40), 2: (45, 15), 3: (45, 65), 4: (30, 40), 5: (30, 25), 6: (30, 55),
                8: (50, 40), 7: (70, 30), 11: (70, 50),
                9: (100, 30), 10: (100, 50)
            },
            20: {  # 343d (Diamond)
                1: (10, 40), 2: (45, 15), 3: (45, 65), 4: (30, 55), 5: (30, 25), 6: (30, 40),
                8: (60, 40), 7: (80, 40),
                10: (100, 25), 11: (100, 55)
            },
            21: {  # 4132
                1: (10, 40), 2: (40, 15), 3: (40, 65), 5: (25, 28), 6: (25, 52),
                4: (50, 40), 7: (70, 25), 10: (70, 40), 8: (70, 55),
                9: (100, 30), 11: (100, 50)
            },
            22: {  # 4240
                1: (10, 40), 2: (40, 15), 3: (40, 65), 5: (25, 28), 6: (25, 52),
                4: (55, 35), 8: (55, 45),
                7: (85, 10), 11: (85, 70), 9: (85, 30), 10: (85, 50)
            },
            23: {  # 4312
                1: (10, 40), 2: (40, 15), 3: (40, 65), 5: (25, 28), 6: (25, 52),
                7: (60, 20), 4: (60, 40), 8: (60, 60),
                9: (85, 40), 10: (100, 30), 11: (100, 50)
            },
            24: {  # 3241
                1: (10, 40), 2: (50, 30), 3: (50, 50), 4: (30, 60), 5: (30, 20), 6: (30, 40),
                10: (75, 15), 11: (75, 65), 7: (75, 30), 8: (75, 50),
                9: (100, 40)
            },
            25: {  # 3331
                1: (10, 40), 2: (50, 20), 3: (50, 60), 4: (30, 60), 5: (30, 20), 6: (30, 40),
                7: (50, 40), 8: (75, 20), 11: (75, 60),
                10: (75, 40), 9: (100, 40)
            }
        }

        # Colores por equipo
        self.team_colors = {
            'Athletic Club': {'primary': '#EE2E24', 'secondary': '#FFFFFF', 'text': 'white'},
            'Atlético de Madrid': {'primary': '#CB3524', 'secondary': '#FFFFFF', 'text': 'white'},
            'CA Osasuna': {'primary': '#D2001C', 'secondary': '#001A4B', 'text': 'white'},
            'CD Leganés': {'primary': '#004C9F', 'secondary': '#FFFFFF', 'text': 'white'},
            'Deportivo Alavés': {'primary': '#1F4788', 'secondary': '#FFFFFF', 'text': 'white'},
            'FC Barcelona': {'primary': '#004D98', 'secondary': '#A50044', 'text': 'white'},
            'Getafe CF': {'primary': '#005CA9', 'secondary': '#FFFFFF', 'text': 'white'},
            'Girona FC': {'primary': '#CC0000', 'secondary': '#FFFFFF', 'text': 'white'},
            'RC Celta': {'primary': '#87CEEB', 'secondary': '#FFFFFF', 'text': 'black'},
            'RCD Espanyol': {'primary': '#004C9F', 'secondary': '#FFFFFF', 'text': 'white'},
            'RCD Mallorca': {'primary': '#CC0000', 'secondary': '#FFFF00', 'text': 'white'},
            'Rayo Vallecano': {'primary': '#CC0000', 'secondary': '#FFFFFF', 'text': 'white'},
            'Real Betis': {'primary': '#00954C', 'secondary': '#FFFFFF', 'text': 'white'},
            'Real Madrid': {'primary': '#FFFFFF', 'secondary': '#FFD700', 'text': 'black'},
            'Real Sociedad': {'primary': '#004C9F', 'secondary': '#FFFFFF', 'text': 'white'},
            'Real Valladolid CF': {'primary': '#663399', 'secondary': '#FFFFFF', 'text': 'white'},
            'Sevilla FC': {'primary': '#D2001C', 'secondary': '#FFFFFF', 'text': 'white'},
            'UD Las Palmas': {'primary': '#FFFF00', 'secondary': '#004C9F', 'text': 'black'},
            'Valencia CF': {'primary': '#FF7F00', 'secondary': '#000000', 'text': 'white'},
            'Villarreal CF': {'primary': '#FFD700', 'secondary': '#004C9F', 'text': 'black'},
        }
        self.default_team_colors = {'primary': '#2c3e50', 'secondary': '#FFFFFF', 'text': 'white'}

        # Colores para métricas individuales - MEJORADOS PARA CONTRASTE
        self.colores_metricas = {
            'Distancia Total': '#d32f2f',                   # Rojo Fuerte (se mantiene)
            'Distancia Total / min': '#00796b',              # Verde Azulado Oscuro
            'Distancia Total 14-21 km / h': '#ffc107',       # Ámbar/Amarillo (Claro -> texto negro)
            'Distancia Total 14-21 km / h / min': '#5d4037', # Marrón Oscuro
            'Distancia Total >21 km / h': '#f57c00',        # Naranja (se mantiene)
            'Distancia Total >21 km / h / min': '#0288d1',   # Azul Claro Intenso
            'Distancia Total >24 km / h': '#c2185b',        # Rosa/Magenta Fuerte
            'Distancia Total >24 km / h / min': '#90a4ae',   # Gris Azulado (Claro -> texto negro)
            'Velocidad Máxima Total': '#6a1b9a'               # Púrpura (se mantiene, ahora es único)
        }
    
    def _calculate_dynamic_fontsize(self, name, scale):
        """
        Calcula un tamaño de fuente dinámico para que el texto quepa en el círculo.
        """
        base_fontsize = int(8 * scale)
        min_fontsize = int(4 * scale)
        base_len = 8  # Longitud de nombre "ideal" para el tamaño base

        name_len = len(str(name))
        if name_len <= base_len:
            return base_fontsize

        # Reducir el tamaño de la fuente proporcionalmente para nombres más largos
        # Usamos una escala no lineal (raíz cuadrada) para que no se encoja demasiado rápido
        scale_factor = (base_len / name_len) ** 0.7 
        dynamic_size = base_fontsize * scale_factor
        
        # Asegurarse de no ser más pequeño que el mínimo permitido
        return int(max(dynamic_size, min_fontsize))

    def levenshtein_distance(self, s1, s2):
        """Calcula la distancia de Levenshtein entre dos strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def construir_nombre_completo(self, jugador):
        """Construye el nombre completo concatenando Nombre + Apellido"""
        # 🆕 VERIFICAR SI LAS COLUMNAS EXISTEN
        has_nombre = 'Nombre' in self.df.columns if self.df is not None else False
        has_apellido = 'Apellido' in self.df.columns if self.df is not None else False
        
        if not has_nombre and not has_apellido:
            print("⚠️ Las columnas 'Nombre' y 'Apellido' no existen en el parquet")
            alias = jugador.get('Alias', 'N/A')
            return str(alias) if pd.notna(alias) else 'N/A'
        
        nombre = jugador.get('Nombre', '') if has_nombre else ''
        apellido = jugador.get('Apellido', '') if has_apellido else ''
        
        # Limpiar valores nan o vacíos
        if pd.isna(nombre):
            nombre = ''
        if pd.isna(apellido):
            apellido = ''
        
        nombre = str(nombre).strip()
        apellido = str(apellido).strip()
        
        # Concatenar con espacio si ambos existen
        if nombre and apellido:
            return f"{nombre} {apellido}"
        elif nombre:
            return nombre
        elif apellido:
            return apellido
        else:
            # Fallback a Alias si no hay nombre ni apellido
            alias = jugador.get('Alias', 'N/A')
            return str(alias) if pd.notna(alias) else 'N/A'

    def dibujar_alineacion_desde_formacion(self, jugadores_con_coordenadas, ax, team_colors, escala, team_logo):
        """
        Dibuja cada jugador directamente en sus coordenadas (X, Y)
        proporcionadas por la lógica de formación de Opta.
        """
        if not jugadores_con_coordenadas:
            return

        for jugador in jugadores_con_coordenadas:
            x = jugador.get('Formation_X')
            y = jugador.get('Formation_Y')
            pos_name = f"Pos_{jugador.get('Formation_Position', 'N/A')}" # Nombre para debug o abreviatura

            if x is not None and y is not None:
                # El diccionario 'jugador' ya contiene todas las métricas físicas necesarias.
                self.crear_jugador_circular(
                    jugador, x, y, ax, team_colors,
                    pos_name, escala, team_logo
                )

    def get_jugadores_con_formacion_opta(self, equipo, jornada, jugadores_df):
        """Método principal que usa SOLO formaciones Opta"""
        formation_result = self.get_formation_coordinates_for_match_FIXED(equipo, jornada, jugadores_df)
        
        if not formation_result:
            print("❌ Sin datos de formación Opta válidos")
            return {}
        
        return self.agrupar_jugadores_formacion_opta(formation_result)

    def get_contrast_color_for_metric(self, hex_color):
        """
        Devuelve 'white' o 'black' para el mejor contraste con un color de fondo hexadecimal.
        """
        try:
            # Quitar el '#' si existe
            hex_color = hex_color.lstrip('#')
            # Convertir a RGB
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            # Fórmula de luminancia (simplificada)
            # Si la suma es alta, el color es claro -> texto negro
            # Si la suma es baja, el color es oscuro -> texto blanco
            if (r * 0.299 + g * 0.587 + b * 0.114) > 186:
                return '#2c3e50'  # Negro/Azul oscuro para fondos claros
            else:
                return 'white'  # Blanco para fondos oscuros
        except Exception:
            return 'white' # Valor por defecto

    def load_player_photos(self):
        """Carga el JSON con las fotos de jugadores"""
        import json
        try:
            with open('assets/jugadores_optimizados.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ No se encontró el archivo jugadores_optimizados.json")
            return []

    def load_match_events(self):
        """Carga los datos del archivo abp_events.parquet Y ASEGURA QUE 'Week' SEA STRING."""
        try:
            events_path = "extraccion_opta/datos_opta_parquet/match_events.parquet"
            self.events_df = pd.read_parquet(events_path)
            
            # 🔥 CORRECCIÓN CLAVE: Asegurar que la columna 'Week' sea de tipo string
            # Esto soluciona el problema de comparación entre integers y strings (ej: 4 == '4')
            if 'Week' in self.events_df.columns:
                self.events_df['Week'] = self.events_df['Week'].astype(str)

            return True
        except Exception as e:
            print(f"⚠️ Error al cargar eventos: {e}")
            self.events_df = None
            return False
    
    def dibujar_alineacion_desde_formacion(self, jugadores_con_coordenadas, ax, team_colors, escala, team_logo):
        """
        NUEVO: Dibuja cada jugador directamente en sus coordenadas (X, Y)
        proporcionadas por la lógica de formación de Opta.
        """
        if not jugadores_con_coordenadas:
            return

        for jugador in jugadores_con_coordenadas:
            x = jugador.get('Formation_X')
            y = jugador.get('Formation_Y')
            # El nombre de la posición es solo para la abreviatura, no para posicionar.
            pos_name = jugador.get('Final_Position', 'N/A')

            if x is not None and y is not None:
                # La función crear_jugador_circular ya tiene todo lo que necesita
                # porque el diccionario 'jugador' contiene las métricas físicas.
                self.crear_jugador_circular(
                    jugador, x, y, ax, team_colors,
                    pos_name, escala, team_logo
                )

    def get_team_setup_from_events_FIXED(self, team_name, week):
        """
        🔥 VERSIÓN FINAL CORREGIDA: Utiliza 'Team Player Formation' como máscara para
        extraer y ordenar correctamente a los 11 jugadores titulares.
        """
        if self.events_df is None:
            print("⚠️ Eventos no cargados, imposible buscar formación.")
            return None

        week_str = str(week)
        
        setup_events = self.events_df[
            (self.events_df['Event Name'] == 'Team set up') &
            (self.events_df['Week'] == week_str)
        ]
        
        if setup_events.empty:
            print(f"   -> ❌ No se encontró el evento 'Team set up' para la Week {week_str}.")
            return None
        
        team_setup = None
        for _, row in setup_events.iterrows():
            opta_team = str(row.get('Team Name', ''))
            if self.are_teams_equivalent(team_name, opta_team):
                team_setup = row
                break
        
        if team_setup is None:
            print(f"   -> ❌ No se encontró setup para el equipo '{team_name}' en esta Week.")
            return None
        
        # --- INICIO DE LA NUEVA LÓGICA DE FILTRADO ---
        
        # 1. Extraer los tres datos clave
        formation_number = team_setup.get('Team Formation')
        jersey_numbers_str = str(team_setup.get('Jersey Number', ''))
        player_formation_str = str(team_setup.get('Team Player Formation', '')) # <-- LA MÁSCARA


        # 2. Validar y convertir las cadenas a listas de números
        try:
            if not jersey_numbers_str or jersey_numbers_str.lower() == 'nan': raise ValueError("Jersey Numbers está vacío")
            if not player_formation_str or player_formation_str.lower() == 'nan': raise ValueError("Player Formation está vacío")

            jersey_list = [int(j.strip()) for j in jersey_numbers_str.split(',') if j.strip().isdigit()]
            formation_slots = [int(s.strip()) for s in player_formation_str.split(',') if s.strip().isdigit()]
            
            if len(jersey_list) != len(formation_slots):
                raise ValueError(f"Las listas no coinciden en longitud: {len(jersey_list)} vs {len(formation_slots)}")

        except Exception as e:
            print(f"   -> ❌ Error crítico procesando las listas de alineación: {e}")
            return None

        # 3. Aplicar la máscara para obtener y ordenar a los 11 titulares
        # Creamos una lista de 11 huecos para colocar a los jugadores en su posición correcta
        ordered_starters = [None] * 11 
        
        for i in range(len(formation_slots)):
            slot_position = formation_slots[i]
            
            # Si la posición es entre 1 y 11, es un titular
            if 1 <= slot_position <= 11:
                jersey_number = jersey_list[i]
                # La posición del slot (1-11) nos dice el índice (0-10) en la lista
                ordered_starters[slot_position - 1] = jersey_number

        # 4. Verificar si tenemos una alineación completa
        if None in ordered_starters:
            num_missing = ordered_starters.count(None)
            print(f"   -> ❌ Error: Faltan {num_missing} jugadores en la alineación titular después de filtrar.")
            return None


        # 5. Crear el mapeo final, ahora sí, solo con los 11 titulares
        position_jersey_map = {i + 1: jersey for i, jersey in enumerate(ordered_starters)}
        
        try:
            formation_number = int(formation_number)
        except (ValueError, TypeError):
            print(f"   -> ❌ Error: 'Team Formation' no es un número válido: {formation_number}")
            return None

        formation_name = self.formation_mapping.get(formation_number, f"Unknown_{formation_number}")
        
        
        return {
            'formation_number': formation_number,
            'formation_name': formation_name,
            'position_jersey_map': position_jersey_map,
            'total_players': len(ordered_starters),
            'raw_jerseys': ordered_starters # Devolvemos la lista limpia
        }

    def get_formation_coordinates_for_match_FIXED(self, team_name, week, jugadores_df):
        """
        VERSIÓN CORREGIDA: Obtiene coordenadas usando el sistema corregido
        """
        
        # 1. Obtener setup desde eventos (versión corregida)
        setup_info = self.get_team_setup_from_events_FIXED(team_name, week)
        
        if not setup_info:
            return None
        
        formation_number = setup_info['formation_number']
        position_jersey_map = setup_info['position_jersey_map']  # posición -> dorsal
        
        # 2. Verificar si tenemos coordenadas para esta formación
        if formation_number not in self.formation_coordinates:
            print(f"⚠️ Formación {formation_number} no tiene coordenadas definidas")
            return None
        
        formation_coords = self.formation_coordinates[formation_number]  # posición -> (x,y)
        
        # 3. MAPEAR JUGADORES A COORDENADAS
        jugadores_con_coordenadas = []
        
        for _, jugador in jugadores_df.iterrows():
            jugador_dorsal = jugador.get('Dorsal')
            
            if pd.isna(jugador_dorsal):
                continue
                
            try:
                dorsal_num = int(float(jugador_dorsal))
            except (ValueError, TypeError):
                continue
            
            # Buscar la posición de este dorsal
            found_position = None
            for pos_num, mapped_dorsal in position_jersey_map.items():
                if mapped_dorsal == dorsal_num:
                    found_position = pos_num
                    break
            
            if found_position and found_position in formation_coords:
                x, y = formation_coords[found_position]
                
                jugador_dict = jugador.to_dict()
                jugador_dict['Formation_Position'] = found_position
                jugador_dict['Formation_X'] = x
                jugador_dict['Formation_Y'] = y
                jugador_dict['Final_Position'] = f"FORMATION_POS_{found_position}"
                
                jugadores_con_coordenadas.append(jugador_dict)
            else:
                print(f"   ❌ #{dorsal_num} {jugador.get('Alias', 'N/A')}: No mapeado")
        
        if not jugadores_con_coordenadas:
            print("❌ No se pudieron mapear jugadores a la formación")
            return None
        
        
        return {
            'jugadores': jugadores_con_coordenadas,
            'formation_info': setup_info
        }

    # FUNCIÓN DE DIAGNÓSTICO MEJORADA
    def debug_team_setup_data(self, team_name, week):
        """Diagnóstico específico para la estructura de datos correcta"""
        
        if self.events_df is None:
            print("❌ events_df no está cargado")
            return
        
        
        # 1. Verificar eventos 'Team set up'
        setup_events = self.events_df[
            (self.events_df['Event Name'] == 'Team set up') &
            (self.events_df['Week'] == week)
        ]
        
        
        if setup_events.empty:
            print("❌ No hay eventos 'Team set up' en esta week")
            return
        
        # 2. Mostrar todos los equipos disponibles
        for i, (_, row) in enumerate(setup_events.iterrows(), 1):
            team = row.get('Team Name', 'N/A')
            formation = row.get('Team Formation', 'N/A')
            jerseys_raw = str(row.get('Jersey Number', 'N/A'))
            jerseys_preview = jerseys_raw[:50] + "..." if len(jerseys_raw) > 50 else jerseys_raw
            
        
        # 3. Buscar coincidencia específica
        
        matches = []
        for _, row in setup_events.iterrows():
            opta_team = str(row.get('Team Name', ''))
            similarity = self.similarity(team_name.lower(), opta_team.lower())
            is_equivalent = self.are_teams_equivalent(team_name, opta_team)
            
            matches.append({
                'team': opta_team,
                'similarity': similarity,
                'equivalent': is_equivalent,
                'row': row
            })
            
        
        # 4. Mostrar mejor match
        best_match = max(matches, key=lambda x: x['similarity']) if matches else None
        
        if best_match and (best_match['equivalent'] or best_match['similarity'] > 0.7):
            row = best_match['row']
            
            formation_num = row.get('Team Formation')
            jersey_str = str(row.get('Jersey Number', ''))
            
            
            # Procesar dorsales
            if jersey_str and jersey_str.lower() not in ['nan', 'none']:
                try:
                    jerseys = [int(j.strip()) for j in jersey_str.split(',') if j.strip().isdigit()]
                except:
                    pass
        else:
            print("❌ No se encontró coincidencia válida")

    
    def agrupar_jugadores_formacion_opta(self, formation_result):
        """Agrupa jugadores usando coordenadas exactas de formación Opta"""
        jugadores_data = formation_result['jugadores']
        formation_info = formation_result['formation_info']
        
        # Agrupar por coordenadas exactas
        jugadores_agrupados = {}
        
        for jugador in jugadores_data:
            # Usar coordenadas como clave única
            x = jugador['Formation_X']
            y = jugador['Formation_Y']
            pos = jugador['Formation_Position']
            
            # Crear clave única basada en posición en formación
            position_key = f"FORMATION_POS_{pos}"
            
            if position_key not in jugadores_agrupados:
                jugadores_agrupados[position_key] = []
            
            # Añadir datos adicionales para compatibilidad
            jugador['Final_Position'] = position_key
            jugador['Coordenada_X'] = x
            jugador['Coordenada_Y'] = y
            
            jugadores_agrupados[position_key].append(jugador)
        
        return jugadores_agrupados

    def get_player_photo_without_dorsal(self, player_name, photos_data, equipo=None):
        """Obtiene la foto sin fondo blanco pero SIN dorsal"""
        match = self.match_player_name(player_name, photos_data, equipo)
        if not match:
            return None
        
        try:
            import base64
            from io import BytesIO
            from PIL import Image
            
            # Decodificar base64 y convertir a imagen
            img_data = base64.b64decode(match['image_base64'])
            img = Image.open(BytesIO(img_data))
            
            # Asegurar que sea RGBA
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            data = np.array(img)
            
            # Verificar dimensiones
            if len(data.shape) != 3 or data.shape[2] != 4:
                return None
            
            height, width = data.shape[:2]
            
            # Flood fill ITERATIVO para quitar fondo blanco
            def flood_fill_iterative(start_points, threshold=235):
                visited = np.zeros((height, width), dtype=bool)
                background_mask = np.zeros((height, width), dtype=bool)
                
                def is_background_color(y, x):
                    if y < 0 or y >= height or x < 0 or x >= width:
                        return False
                    return (data[y, x, 0] >= threshold and 
                            data[y, x, 1] >= threshold and 
                            data[y, x, 2] >= threshold)
                
                for start_y, start_x in start_points:
                    if visited[start_y, start_x] or not is_background_color(start_y, start_x):
                        continue
                    
                    stack = [(start_y, start_x)]
                    
                    while stack:
                        y, x = stack.pop()
                        
                        if (y < 0 or y >= height or x < 0 or x >= width or 
                            visited[y, x] or not is_background_color(y, x)):
                            continue
                        
                        visited[y, x] = True
                        background_mask[y, x] = True
                        
                        stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
                
                return background_mask
            
            # Puntos de inicio para flood fill
            border_points = [
                (0, 0), (0, width-1), (height-1, 0), (height-1, width-1),  # Esquinas
                (0, width//2), (height-1, width//2),  # Centro superior e inferior
                (height//2, 0), (height//2, width-1),  # Centro izquierda y derecha
                (0, width//4), (0, 3*width//4),  # Más puntos en bordes
                (height-1, width//4), (height-1, 3*width//4),
                (height//4, 0), (3*height//4, 0),
                (height//4, width-1), (3*height//4, width-1)
            ]
            
            # Aplicar flood fill
            background_mask = flood_fill_iterative(border_points, threshold=220)  # ← MÁS AGRESIVO
            
            # Hacer transparente el fondo
            data[background_mask] = [0, 0, 0, 0]
            
            return data.astype(np.float32) / 255.0
        
        except Exception as e:
            print(f"⚠️ Error procesando foto de {player_name}: {e}")
            return None

    def extract_names_parts(self, name):
        """Extrae las partes de un nombre normalizado - VERSIÓN MEJORADA"""
        def normalize_name(name):
            """Normaliza un nombre eliminando acentos y caracteres especiales"""
            if not name:
                return ""
            name = str(name).lower().strip()
            
            # Normalización más agresiva para caracteres especiales
            replacements = {
                'ñ': 'n', 'ç': 'c', 'ü': 'u', 'ï': 'i',
                'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
                'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u'
            }
            
            for orig, repl in replacements.items():
                name = name.replace(orig, repl)
            
            # Normalización NFD estándar
            name = unicodedata.normalize('NFD', name)
            name = ''.join(char for char in name if unicodedata.category(char) != 'Mn')
            name = re.sub(r"['\-`´'']", "", name)
            name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
            return ' '.join(name.split())
            
        normalized = normalize_name(name)
        parts = normalized.split()
        
        if not parts:
            return {
                'full': '',
                'first_name': '',
                'last_name': '',
                'all_parts': []
            }
            
        first_name = parts[0]
        last_name = parts[-1] if len(parts) > 1 else first_name
        
        return {
            'full': normalized,
            'first_name': first_name,
            'last_name': last_name,
            'all_parts': parts
        }

    def match_player_name(self, player_name, photos_data, team_filter=None):
        """Encuentra el jugador más parecido filtrando primero por equipo"""
        
        player_parts = self.extract_names_parts(player_name)
        if not player_parts['full']:
            return None

        # PASO 1: Filtrar por equipo
        team_players = []
        if team_filter:
            for photo_entry in photos_data:
                photo_team = photo_entry.get('team_name')
                if photo_team and self.are_teams_equivalent(team_filter, photo_team):
                    team_players.append(photo_entry)
            
            if not team_players:
                return None
        else:
            return None

        # PASO 2: Buscar coincidencias exactas y tolerantes
        candidates = []
        player_words = [w for w in player_parts['all_parts'] if len(w) > 3]
        
        for photo_entry in team_players:
            photo_name = photo_entry.get('player_name', '')
            photo_parts = self.extract_names_parts(photo_name)
            photo_words = [w for w in photo_parts['all_parts'] if len(w) > 3]
            
            matches = []
            for p_word in player_words:
                for ph_word in photo_words:
                    # Coincidencia exacta
                    if p_word == ph_word:
                        matches.append(p_word)
                    # 🆕 NUEVA LÓGICA: Tolerancia para palabras largas
                    elif len(p_word) > 5 and len(ph_word) > 5:
                        distance = self.levenshtein_distance(p_word, ph_word)  # ← Ahora con self.
                        if distance == 1:  # Solo 1 letra de diferencia
                            matches.append(p_word)
            
            if matches:
                candidates.append({
                    'entry': photo_entry,
                    'matches': matches,
                    'match_count': len(matches)
                })
        
        # PASO 3: Resolver conflictos (sin cambios)
        if len(candidates) == 0:
            return None
        elif len(candidates) == 1:
            return candidates[0]['entry']
        else:
            best_candidates = sorted(candidates, key=lambda x: x['match_count'], reverse=True)
            
            if best_candidates[0]['match_count'] > best_candidates[1]['match_count']:
                return best_candidates[0]['entry']
            
            for candidate in best_candidates:
                photo_parts = self.extract_names_parts(candidate['entry']['player_name'])
                
                for p_word in player_parts['all_parts']:
                    if len(p_word) <= 3:
                        for ph_word in photo_parts['all_parts']:
                            if ph_word.startswith(p_word):
                                return candidate['entry']
            
            return None

    def get_contrasting_color(self, background_color):
        """Devuelve blanco o negro según el color de fondo para mejor contraste"""
        # Si el color primario es claro, usar texto oscuro
        light_colors = ['#FFFFFF', '#FFD700', '#FFFF00', '#87CEEB']  # Blanco, dorado, amarillo, celeste
        
        if background_color in light_colors:
            return '#2c3e50'  # Azul oscuro
        else:
            return 'white'
    
    def load_data(self):
        """Carga los datos del archivo parquet"""
        try:
            self.df = pd.read_parquet(self.data_path)
        except Exception as e:
            print(f"❌ Error al cargar datos: {e}")
            
    def similarity(self, a, b):
        """Calcula la similitud entre dos strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def clean_team_names(self):
        """Limpia y agrupa nombres de equipos similares"""
        if self.df is None:
            return
        
        # Limpiar nombres de equipos
        unique_teams = self.df['Equipo'].unique()
        team_mapping = {}
        processed_teams = set()
        
        for team in unique_teams:
            if team in processed_teams:
                continue
                
            similar_teams = [team]
            for other_team in unique_teams:
                if other_team != team and other_team not in processed_teams:
                    if self.similarity(team, other_team) > 0.85:
                        similar_teams.append(other_team)
            
            canonical_name = max(similar_teams, key=len)
            for similar_team in similar_teams:
                team_mapping[similar_team] = canonical_name
                processed_teams.add(similar_team)
        
        self.df['Equipo'] = self.df['Equipo'].map(team_mapping)
        
        # Normalizar jornadas
        def normalize_jornada(jornada):
            if isinstance(jornada, str) and jornada.startswith(('J', 'j')):
                try:
                    return int(jornada[1:])
                except ValueError:
                    return jornada
            return jornada
        
        self.df['Jornada'] = self.df['Jornada'].apply(normalize_jornada)

    def are_teams_equivalent(self, team1, team2):
        """
        VERSIÓN CORREGIDA Y REFORZADA: Compara dos nombres de equipo con una lógica
        más estricta para evitar falsos positivos, especialmente entre equipos de la misma ciudad.
        """
        if not team1 or not team2:
            return False

        # Normalizar nombres para la comparación (eliminar acentos, mayúsculas, etc.)
        norm1 = self.normalize_text(team1)
        norm2 = self.normalize_text(team2)

        # 1. Coincidencia exacta (la más fiable y rápida)
        if norm1 == norm2:
            return True

        # 2. REGLA DE EXCLUSIÓN CRÍTICA: Evitar falsos positivos (ej. Real Madrid vs Atlético)
        # Si uno contiene "real" y el otro "atletico", NUNCA pueden ser equivalentes.
        is_norm1_real = 'real' in norm1 and 'atletico' not in norm1
        is_norm2_real = 'real' in norm2 and 'atletico' not in norm2
        is_norm1_atletico = 'atletico' in norm1
        is_norm2_atletico = 'atletico' in norm2

        if (is_norm1_real and is_norm2_atletico) or (is_norm1_atletico and is_norm2_real):
            return False # ¡Caso explícito del derbi de Madrid, no son el mismo equipo!

        # 3. Lógica de palabras significativas (mejorada)
        # Extrae palabras de más de 3 letras, ignorando prefijos comunes.
        def extract_meaningful_words(team_name):
            words = team_name.split()
            prefixes_to_ignore = {'fc', 'cf', 'cd', 'ud', 'rcd', 'rc', 'ca', 'real', 'deportivo'}
            # Se mantiene 'atletico' como palabra significativa
            return {word for word in words if len(word) > 3 and word not in prefixes_to_ignore}

        words1 = extract_meaningful_words(norm1)
        words2 = extract_meaningful_words(norm2)

        # Si ambos conjuntos de palabras existen y uno es subconjunto del otro, es un match.
        # (Ej: "betis" es subconjunto de "real betis")
        if words1 and words2:
            if words1.issubset(words2) or words2.issubset(words1):
                return True

        # 4. Mapeos específicos para casos conocidos (como fallback)
        team_mappings = {
            'atletico de madrid': ['atletico madrid', 'atletico'],
            'ca osasuna': ['osasuna'],
            'real sociedad': ['sociedad'],
            'athletic club': ['athletic bilbao', 'athletic'],
            'rc celta': ['celta vigo', 'celta de vigo', 'celta'],
            'real betis': ['betis'],
            'deportivo alaves': ['alaves'],
            'rayo vallecano': ['rayo'],
            'ud las palmas': ['las palmas']
        }

        for canonical, variations in team_mappings.items():
            is_team1 = norm1 == canonical or norm1 in variations
            is_team2 = norm2 == canonical or norm2 in variations
            if is_team1 and is_team2:
                return True

        # 5. Fallback final por similitud alta (ahora es menos probable que se necesite)
        if SequenceMatcher(None, norm1, norm2).ratio() > 0.9:
            return True
            
        return False
            

    def convert_jornada_to_week(self, jornada):
        """Convierte jornada MediaCoach (j1, j2) a Week Opta (1, 2)"""
        if not jornada:
            return None
        
        try:
            if isinstance(jornada, str):
                if jornada.lower().startswith('j'):
                    resultado = int(jornada[1:])
                else:
                    resultado = int(jornada)
            else:
                resultado = int(jornada)
            
            return resultado
        except (ValueError, TypeError):
            return None

    @staticmethod
    def normalize_text(text):
        """Normaliza texto eliminando acentos, espacios extra y caracteres especiales"""
        import re
        import unicodedata
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text    
    
    def get_available_teams(self):
        """Retorna equipos disponibles"""
        if self.df is None:
            return []
        return sorted(self.df['Equipo'].unique())
    
    def get_available_jornadas(self):
        """Retorna jornadas disponibles"""
        if self.df is None:
            return []
        return sorted(self.df['Jornada'].unique())
    
    def determinar_local_visitante(self, partido, equipo):
        """Determina si un partido es local o visitante para un equipo"""
        if '-' not in partido:
            return 'desconocido'
        
        partes = partido.split('-')
        if len(partes) != 2:
            return 'desconocido'
        
        equipo_local_partido = partes[0].strip()
        equipo_visitante_partido = partes[1].strip()
        
        sim_local = self.similarity(equipo, equipo_local_partido)
        sim_visitante = self.similarity(equipo, equipo_visitante_partido)
        
        if sim_local > sim_visitante:
            return 'local'
        elif sim_visitante > sim_local:
            return 'visitante'
        else:
            return 'desconocido'

    def extraer_rival(self, partido, equipo):
        """Extrae el nombre del equipo rival del partido"""
        if '-' not in partido:
            return 'Rival'
        
        partes = partido.split('-')
        if len(partes) != 2:
            return 'Rival'
        
        equipo_local = partes[0].strip()
        equipo_visitante = partes[1].strip()
        
        sim_local = self.similarity(equipo, equipo_local)
        sim_visitante = self.similarity(equipo, equipo_visitante)
        
        if sim_local > sim_visitante:
            return equipo_visitante
        else:
            return equipo_local
    
    def get_last_5_jornadas(self, equipo, jornada_referencia):
        """Obtiene las últimas 5 jornadas incluyendo la de referencia"""
        jornadas_disponibles = self.get_available_jornadas()
        
        # Normalizar jornada de referencia
        if isinstance(jornada_referencia, str) and jornada_referencia.startswith('J'):
            try:
                jornada_referencia = int(jornada_referencia[1:])
            except ValueError:
                pass
        elif isinstance(jornada_referencia, str) and jornada_referencia.startswith('j'):
            try:
                jornada_referencia = int(jornada_referencia[1:])
            except ValueError:
                pass
        
        # Filtrar jornadas menores o iguales a la de referencia
        jornadas_validas = [j for j in jornadas_disponibles if j <= jornada_referencia]
        
        # Tomar las últimas 5
        if len(jornadas_validas) >= 5:
            return sorted(jornadas_validas)[-5:]
        else:
            return sorted(jornadas_validas)
    
    def get_available_jornadas(self, equipo=None):
        """Retorna las jornadas disponibles para un equipo específico o todas"""
        if self.df is None:
            return []
        
        if equipo:
            filtered_df = self.df[self.df['Equipo'] == equipo]
            return sorted(filtered_df['Jornada'].unique())
        else:
            return sorted(self.df['Jornada'].unique())

    
    def parsear_partido_completo(self, partido, equipo):
        """Parsea un partido completo manteniendo el orden original"""
        if '-' not in partido:
            return equipo, 'Rival', 'N', 'N', 'desconocido'
        
        partes = partido.split('-')
        if len(partes) != 2:
            return equipo, 'Rival', 'N', 'N', 'desconocido'
        
        parte_local = partes[0].strip()
        parte_visitante = partes[1].strip()
        
        import re
        
        # Para la parte local
        match_local = re.match(r'(.+?)(\d+)$', parte_local)
        if match_local:
            equipo_local_raw = match_local.group(1)
            goles_local = match_local.group(2)
        else:
            equipo_local_raw = parte_local
            goles_local = 'N'
        
        # Para la parte visitante
        match_visitante = re.match(r'(\d+)(.+)$', parte_visitante)
        if match_visitante:
            goles_visitante = match_visitante.group(1)
            equipo_visitante_raw = match_visitante.group(2)
        else:
            goles_visitante = 'N'
            equipo_visitante_raw = parte_visitante
        
        equipo_local_limpio = self.limpiar_nombre_equipo(equipo_local_raw)
        equipo_visitante_limpio = self.limpiar_nombre_equipo(equipo_visitante_raw)
        
        sim_local = self.similarity(equipo, equipo_local_limpio)
        sim_visitante = self.similarity(equipo, equipo_visitante_limpio)
        
        if sim_local > sim_visitante:
            return equipo_local_limpio, equipo_visitante_limpio, goles_local, goles_visitante, 'local'
        else:
            return equipo_local_limpio, equipo_visitante_limpio, goles_local, goles_visitante, 'visitante'

    def limpiar_nombre_equipo(self, nombre_raw):
        """Limpia nombres de equipos"""
        equipos_conocidos = {
            'sevillafc': 'Sevilla FC',
            'getafecf': 'Getafe CF', 
            'gironafc': 'Girona FC',
            'villarrealcf': 'Villarreal CF',
            'realmadrid': 'Real Madrid',
            'fcbarcelona': 'FC Barcelona',
            'athleticclub': 'Athletic Club',
            'atleticodemadrid': 'Atlético de Madrid',
            'realbetis': 'Real Betis',
            'realsociedad': 'Real Sociedad',
            'valenciacf': 'Valencia CF',
            'rcelta': 'RC Celta',
            'caosasuna': 'CA Osasuna',
            'rayovallecano': 'Rayo Vallecano',
            'udlaspalmas': 'UD Las Palmas',
            'rcdespanyol': 'RCD Espanyol',
            'deportivoalaves': 'Deportivo Alavés',
            'cdleganes': 'CD Leganés',
            'realvalladolidcf': 'Real Valladolid CF',
            'rcdmallorca': 'RCD Mallorca'
        }
        
        nombre_lower = nombre_raw.lower().strip()
        
        if nombre_lower in equipos_conocidos:
            return equipos_conocidos[nombre_lower]
        
        for key, value in equipos_conocidos.items():
            if key in nombre_lower or nombre_lower in key:
                return value
        
        return nombre_raw.replace('fc', ' FC').replace('cf', ' CF').title()
    
    def get_ultimos_4_partidos(self, equipo, jornada_maxima, tipo_partido_filter=None):
        """
        Obtiene los últimos 4 partidos del equipo, incluyendo SOLO jugadores
        que hayan jugado MÁS DE 70 MINUTOS en cada partido.
        """
        if self.df is None:
            return []
        
        if isinstance(jornada_maxima, str) and jornada_maxima.startswith(('J', 'j')):
            try:
                jornada_maxima = int(jornada_maxima[1:])
            except ValueError:
                pass
        
        tipo_display = tipo_partido_filter.upper() if tipo_partido_filter else "TODOS"
        
        filtrado = self.df[
            (self.df['Equipo'] == equipo) & 
            (self.df['Jornada'] <= jornada_maxima)
        ].copy()
        
        if len(filtrado) == 0:
            return []
        
        partidos_info = filtrado[['Partido', 'Jornada']].drop_duplicates()
        
        if tipo_partido_filter:
            partidos_filtrados = []
            for _, partido_info in partidos_info.iterrows():
                partido = partido_info['Partido']
                tipo = self.determinar_local_visitante(partido, equipo)
                if tipo == tipo_partido_filter:
                    partidos_filtrados.append(partido_info)
            
            if partidos_filtrados:
                partidos_info = pd.DataFrame(partidos_filtrados)
            else:
                print(f"❌ No hay partidos {tipo_partido_filter.upper()} para {equipo}")
                return []
        
        partidos_info = partidos_info.sort_values('Jornada', ascending=False)
        
        ultimos_partidos = partidos_info.head(4)
        
        resultados = []
        for _, partido_info in ultimos_partidos.iterrows():
            partido = partido_info['Partido']
            jornada = partido_info['Jornada']
            
            if 'Nombre' in filtrado.columns:
                mask_empty_alias = filtrado['Alias'].isna() | (filtrado['Alias'] == '') | (filtrado['Alias'].str.strip() == '')
                filtrado.loc[mask_empty_alias, 'Alias'] = filtrado.loc[mask_empty_alias, 'Nombre']
            
            datos_partido = filtrado[filtrado['Partido'] == partido].copy()
            
            # 🔥 CAMBIO CLAVE AQUÍ 🔥
            if 'Minutos jugados' in datos_partido.columns:
                jugadores_antes = len(datos_partido)
                # Se filtra para que solo queden los jugadores con MÁS de 70 minutos
                datos_partido = datos_partido[datos_partido['Minutos jugados'] > 70]
                jugadores_despues = len(datos_partido)

            
            if len(datos_partido) > 0:
                tipo_partido = self.determinar_local_visitante(partido, equipo)
                rival = self.extraer_rival(partido, equipo)
                
                resultados.append({
                    'partido': partido,
                    'jornada': jornada,
                    'tipo': tipo_partido,
                    'rival': rival,
                    'datos': datos_partido
                })
        
        return resultados

    def calcular_dimensiones_tabla(self, jugadores_list, scale=0.9):
        """Calcula dimensiones simples y compactas"""
        if not jugadores_list:
            return 0, 0
        
        # Cargar fotos
        photos_data = self.load_player_photos()

        # Dimensiones para la foto del jugador
        photo_width = 15.0 * scale
        photo_height = names_height * 14.5  # Más alta que la fila de nombres
        
        num_players = len(jugadores_list)
        num_metrics = len(self.metricas_tabla)
        
        # Dimensiones BASE más pequeñas
        base_metric_width = 3.5 * scale
        base_player_width = 2.0 * scale
        base_row_height = 0.8 * scale
        
        # 🔧 ANCHO MANUAL - USA EL MISMO VALOR QUE ARRIBA
        player_col_width = 5.0 * scale     # ← EL MISMO VALOR que en crear_tabla_posicion
        
        if num_players > 3:
            player_col_width *= 0.9
        
        table_width = base_metric_width + (num_players * player_col_width)
        table_height = (base_row_height * 1.5) + (base_row_height * 1.8) + (num_metrics * base_row_height)
        
        return table_width, table_height
    
    def debug_player_photos(self, equipo, player_names_to_check):
        """Método para diagnosticar problemas con las fotos"""
        photos_data = self.load_player_photos()
        
        
        # 🆕 DIAGNÓSTICO DEL PARQUET
        if self.df is not None:
            columnas = list(self.df.columns)
            
            # Mostrar una muestra de datos de un jugador
            sample_player = self.df.head(1).iloc[0]
            for col in ['Nombre', 'Apellido', 'Alias', 'Equipo']:
                if col in self.df.columns:
                    valor = sample_player.get(col, 'N/A')
        
        # Mostrar jugadores del equipo en el JSON
        team_players_in_json = []
        for photo_entry in photos_data:
            photo_team = photo_entry.get('team_name')
            if photo_team and self.are_teams_equivalent(equipo, photo_team):
                team_players_in_json.append(photo_entry.get('player_name', 'N/A'))
        
        
        for player_name in player_names_to_check:
            
            # 🆕 MOSTRAR CÓMO SE CONSTRUYE EL NOMBRE COMPLETO
            # Crear un jugador de ejemplo para probar
            sample_jugador = {'Alias': player_name}
            if 'Nombre' in self.df.columns:
                sample_jugador['Nombre'] = 'NOMBRE_EJEMPLO'
            if 'Apellido' in self.df.columns:
                sample_jugador['Apellido'] = 'APELLIDO_EJEMPLO'
                
            nombre_construido = self.construir_nombre_completo(sample_jugador)
            
            # Mostrar versión normalizada
            player_parts = self.extract_names_parts(player_name)
            
            # Buscar coincidencias
            match = self.match_player_name(player_name, photos_data, equipo)
            if match:
                pass
            else:
                print(f"   ❌ NO ENCONTRADO")
                
                # Mostrar similitudes con los nombres del equipo
                for json_name in team_players_in_json:
                    similarity = self.similarity(player_parts['full'], self.extract_names_parts(json_name)['full'])

    def get_team_colors(self, equipo):
        """Obtiene colores del equipo"""
        if equipo in self.team_colors:
            return self.team_colors[equipo]
        
        for team_name in self.team_colors.keys():
            if team_name.lower() in equipo.lower() or equipo.lower() in team_name.lower():
                return self.team_colors[team_name]
        
        return self.default_team_colors
    
    def load_team_logo(self, equipo):
        """Carga el escudo del equipo con lógica mejorada SIN diccionario"""
        import re
        import glob
        
        # Crear múltiples variaciones inteligentes del nombre
        possible_names = []
        
        # Limpiar el nombre base
        equipo_limpio = re.sub(r'[^a-zA-Z0-9\s]', '', equipo.lower().strip())
        
        # Variaciones básicas
        possible_names.extend([
            equipo,
            equipo_limpio,
            equipo.replace(' ', ''),
            equipo.replace(' ', '_'),
            equipo.lower(),
            equipo.lower().replace(' ', ''),
            equipo.lower().replace(' ', '_'),
        ])
        
        # 🔥 LÓGICA INTELIGENTE PARA EQUIPOS ESPAÑOLES
        # Eliminar prefijos/sufijos comunes automáticamente
        prefixes_suffixes = [
            'fc ', 'cf ', 'cd ', 'ud ', 'rcd ', 'rc ', 'ca ', 'real ', 'deportivo ',
            ' fc', ' cf', ' cd', ' ud', ' rcd', ' rc', ' ca'
        ]
        
        for prefix in prefixes_suffixes:
            if equipo_limpio.startswith(prefix.strip()):
                equipo_sin_prefijo = equipo_limpio[len(prefix):].strip()
                possible_names.extend([
                    equipo_sin_prefijo,
                    equipo_sin_prefijo.replace(' ', ''),
                    equipo_sin_prefijo.replace(' ', '_')
                ])
            
            if equipo_limpio.endswith(prefix.strip()):
                equipo_sin_sufijo = equipo_limpio[:-len(prefix)].strip()
                possible_names.extend([
                    equipo_sin_sufijo,
                    equipo_sin_sufijo.replace(' ', ''),
                    equipo_sin_sufijo.replace(' ', '_')
                ])
        
        # Variaciones adicionales para casos específicos
        # CASO ESPECIAL: Atlético de Madrid -> buscar "Atlético" con tilde
        if 'atlético' in equipo.lower() or 'atletico' in equipo_limpio:
            possible_names.insert(0, 'Atlético')  # Prioritario con tilde
            possible_names.extend(['atletico', 'atleticomadrid', 'atletico_madrid'])
        
        if 'madrid' in equipo_limpio and 'atletico' not in equipo_limpio:
            possible_names.extend(['madrid', 'realmadrid', 'real_madrid'])
        
        if 'barcelona' in equipo_limpio:
            possible_names.extend(['barcelona', 'barca', 'fcbarcelona', 'fcb'])
        
        if 'athletic' in equipo_limpio:
            possible_names.extend(['athletic', 'bilbao', 'athleticclub'])
        
        if 'betis' in equipo_limpio:
            possible_names.extend(['betis', 'realbetis', 'real_betis'])
        
        if 'sociedad' in equipo_limpio:
            possible_names.extend(['sociedad', 'realsociedad', 'real_sociedad'])
        
        if 'celta' in equipo_limpio:
            possible_names.extend(['celta', 'vigo', 'celta_vigo', 'rcelta'])
        
        if 'espanyol' in equipo_limpio:
            possible_names.extend(['espanyol', 'espanol', 'rcdespanyol'])
        
        if 'las palmas' in equipo_limpio:
            possible_names.extend(['laspalmas', 'las_palmas', 'palmas', 'udlaspalmas'])
        
        if 'rayo' in equipo_limpio:
            possible_names.extend(['rayo', 'vallecano', 'rayo_vallecano', 'rayovallecano'])
        
        # Eliminar duplicados manteniendo orden
        seen = set()
        unique_names = []
        for name in possible_names:
            if name and name not in seen:
                seen.add(name)
                unique_names.append(name)
        
        # 🔍 FASE 1: Búsqueda exacta
        for name in unique_names:
            logo_path = f"assets/escudos/{name}.png"
            if os.path.exists(logo_path):
                try:
                    return plt.imread(logo_path)
                except Exception as e:
                    print(f"⚠️ Error al cargar {logo_path}: {e}")
                    continue
        
        # 🔍 FASE 2: Búsqueda por similitud en archivos existentes - MEJORADA
        try:
            escudos_disponibles = glob.glob("assets/escudos/*.png")
            
            mejor_match = None
            mejor_similitud = 0
            
            for escudo_path in escudos_disponibles:
                nombre_archivo = os.path.basename(escudo_path).lower().replace('.png', '')
                
                # IMPORTANTE: Evitar falsos positivos entre Atlético y Real Madrid
                # Si estamos buscando Atlético, no considerar Real Madrid como candidato
                if 'atletico' in equipo_limpio and 'real' in nombre_archivo and 'atletico' not in nombre_archivo:
                    continue
                # Si estamos buscando Real Madrid, no considerar Atlético como candidato  
                if 'real' in equipo_limpio and 'atletico' in nombre_archivo:
                    continue
                
                # Calcular similitud con cada variación
                for variacion in unique_names:
                    similitud = self.similarity(variacion.lower(), nombre_archivo)
                    
                    if similitud > mejor_similitud:
                        mejor_similitud = similitud
                        mejor_match = escudo_path
            
            # Si encontramos una buena similitud (>60% para ser más estrictos)
            if mejor_match and mejor_similitud > 0.6:
                try:
                    return plt.imread(mejor_match)
                except Exception as e:
                    print(f"⚠️ Error al cargar {mejor_match}: {e}")
            
            # Mostrar archivos disponibles para debug
            
        except Exception as e:
            print(f"⚠️ Error en búsqueda por similitud: {e}")
        
        print(f"❌ No se encontró escudo para: {equipo}")
        return None

    def load_background_image(self):
        """Carga la imagen de fondo"""
        background_path = "assets/fondo_informes.png"
        if os.path.exists(background_path):
            try:
                return plt.imread(background_path)
            except Exception:
                print(f"⚠️ No se pudo cargar la imagen de fondo: {background_path}")
                return None
        return None
    
    def crear_campo_sin_espacios(self, ax):
        """🔥 MÉTODO MEJORADO: Crea campo horizontal SIN ESPACIOS como el primer script"""
        
        # Crear pitch sin padding
        pitch = Pitch(
            pitch_color='grass', 
            line_color='white', 
            stripe=True, 
            linewidth=2,
            pad_left=0, pad_right=0, pad_bottom=0, pad_top=0
        )
        
        # Dibujar en el ax proporcionado
        pitch.draw(ax=ax)
        
        # 🔥 CONFIGURACIÓN AGRESIVA PARA ELIMINAR ESPACIOS (COPIADA DEL PRIMER SCRIPT)
        ax.set_position(ax.get_position())
        ax.margins(0, 0)
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 80)
        ax.autoscale(enable=False)
        ax.set_aspect('equal')
        ax.set_frame_on(False)
        
        return pitch

    def crear_jugador_circular(self, jugador, x, y, ax, team_colors, posicion_name, scale=0.8, team_logo=None):
        """
        🔥 VERSIÓN FINAL: Dibuja el nombre del jugador en una sola línea con fuente dinámica
        y fondo sólido para máxima legibilidad.
        """
        abbreviaciones = {
            'PORTERO': 'PO', 'LATERAL_DERECHO': 'LD', 'LATERAL_IZQUIERDO': 'LI',
            'CENTRAL_DERECHO': 'CD', 'CENTRAL_IZQUIERDO': 'CI','CENTRAL_CENTRO': 'CC', 'MC_POSICIONAL': 'MCD',
            'MC_ORGANIZADOR': 'MC', 'MC_BOX_TO_BOX': 'MP', 'BANDA_DERECHA': 'MD',
            'BANDA_IZQUIERDA': 'MI', 'DELANTERO_CENTRO': 'DC', 'SEGUNDO_DELANTERO': 'SD'
        }
        if not isinstance(jugador, dict):
            return

        photos_data = self.load_player_photos()
        
        radio_circulo = 6 * scale
        radio_metricas = 7 * scale
        
        # Lógica de búsqueda de fotos (compatible con ambos scripts)
        equipo = jugador.get('Equipo', None)
        full_name = self.construir_nombre_completo(jugador)
        player_photo = self.get_player_photo_without_dorsal(full_name, photos_data, equipo)
        if player_photo is None:
            alias = jugador.get('Alias')
            if pd.notna(alias) and alias.strip() and alias != full_name:
                player_photo = self.get_player_photo_without_dorsal(alias, photos_data, equipo)

        circle_bg = plt.Circle((x, y), radio_circulo, facecolor='white', alpha=0.7, edgecolor='gray', linewidth=2, zorder=5)
        ax.add_patch(circle_bg)

        if player_photo is not None:
            img_extent = [x - radio_circulo * 0.8, x + radio_circulo * 0.8, y - radio_circulo * 0.8, y + radio_circulo * 0.8]
            ax.imshow(player_photo, extent=img_extent, aspect='auto', clip_on=True, zorder=6)
        elif team_logo is not None:
            logo_extent = [x - radio_circulo * 0.7, x + radio_circulo * 0.7, y - radio_circulo * 0.7, y + radio_circulo * 0.7]
            ax.imshow(team_logo, extent=logo_extent, aspect='auto', clip_on=True, zorder=6, alpha=0.8)
        
        # --- DIBUJO DE MÉTRICAS CON DESPLAZAMIENTO ---
        metric_offsets = {
            'Velocidad Máxima Total': 0.5,
            'Distancia Total / min': 0.5
        }

        num_metricas = len(self.metricas_tabla)
        angulo_step = 2 * np.pi / num_metricas
        for i, metrica in enumerate(self.metricas_tabla):
            angulo = i * angulo_step - np.pi / 2
            base_rect_x = x + radio_metricas * np.cos(angulo)
            base_rect_y = y + radio_metricas * np.sin(angulo)
            y_offset = metric_offsets.get(metrica, 0)
            rect_x = base_rect_x
            rect_y = base_rect_y + y_offset

            valor = jugador.get(metrica)
            if pd.notna(valor):
                if 'Velocidad' in metrica or '/ min' in metrica: valor_text = f"{valor:.1f}"
                else: valor_text = f"{valor:.0f}"
            else: valor_text = "N/A"
            color_metrica = self.colores_metricas.get(metrica, '#2c3e50')
            color_texto_contraste = self.get_contrast_color_for_metric(color_metrica)
            ax.text(rect_x, rect_y, valor_text, fontsize=int(6*scale), weight='bold', color=color_texto_contraste,
                    ha='center', va='center', zorder=15,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=color_metrica, alpha=0.95, edgecolor='white', linewidth=0.5))
        
        # --- DORSAL Y POSICIÓN ---
        dorsal = jugador.get('Dorsal', 'N/A')
        dorsal_str = str(int(float(dorsal))) if pd.notna(dorsal) and dorsal != 'N/A' else ''
        if dorsal_str:
            dorsal_x = x - radio_circulo * 0.60
            dorsal_y = y + radio_circulo * 0.18
            ax.text(dorsal_x, dorsal_y, dorsal_str, 
                    fontsize=int(9*scale), weight='900', color='black',
                    ha='center', va='center', zorder=10,
                    path_effects=[patheffects.withStroke(linewidth=2.5, foreground='white')])

        posicion_base_nombre = re.sub(r'_\d+$', '', posicion_name)
        demarcacion_abreviada = abbreviaciones.get(posicion_base_nombre, '')
        if demarcacion_abreviada:
            dem_x = x + radio_circulo * 0.60
            dem_y = y + radio_circulo * 0.20
            ax.text(dem_x, dem_y, demarcacion_abreviada, 
                    fontsize=int(6.5*scale), weight='bold', color='#2c3e50',
                    ha='center', va='center', zorder=10,
                    path_effects=[patheffects.withStroke(linewidth=2.5, foreground='white')])

        # 🔥 --- INICIO DE LOS CAMBIOS PARA EL NOMBRE --- 🔥
        # 1. Obtener el nombre para mostrar en una sola línea
        player_name = jugador.get('Alias', 'N/A')
        
        # 2. Calcular el tamaño de fuente dinámico para que se ajuste
        fontsize = self._calculate_dynamic_fontsize(player_name, scale)
        
        # 3. Posición vertical, ligeramente en la parte inferior del círculo
        y_pos = y - radio_circulo * 0.50
        
        # 4. Dibujar el texto con fondo blanco sólido (alpha=1.0)
        ax.text(x, y_pos, player_name,
                fontsize=fontsize,
                weight='bold', 
                color='black', 
                ha='center', va='center', zorder=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=1.0, edgecolor='none'))
       
    
    def calcular_fontsize_para_circulo(self, texto, radio, scale, factor=0.2):
        """Calcula el tamaño de fuente para que el texto quepa en el círculo"""
        if not texto:
            return int(4 * scale)
        
        # Estimar ancho del texto (aproximación)
        ancho_disponible = radio * 2 * factor  # Factor determina qué porción del círculo usar
        chars_length = len(texto)
        
        # Fórmula empírica para calcular fontsize
        fontsize_estimado = max(int(ancho_disponible / chars_length * 2.5 * scale), int(3 * scale))
        fontsize_max = int(8 * scale)  # Límite máximo
        fontsize_min = int(3 * scale)  # Límite mínimo
        
        return min(max(fontsize_estimado, fontsize_min), fontsize_max)

    def acortar_nombre_para_circulo(self, nombre, radio, scale):
        """Acorta el nombre inteligentemente para que quepa en el círculo"""
        if not nombre or pd.isna(nombre):
            return "N/A"
        
        nombre_str = str(nombre).strip()
        
        # Si es corto, devolverlo tal como está
        if len(nombre_str) <= 8:
            return nombre_str
        
        # Estrategias de acortamiento
        partes = nombre_str.split(' ')
        
        if len(partes) > 1:
            # Primera letra + apellido
            primer_nombre = partes[0]
            apellido = partes[-1]
            
            if len(primer_nombre) + len(apellido) + 2 <= 10:  # Con punto y espacio
                return f"{primer_nombre[0]}. {apellido}"
            elif len(apellido) <= 8:
                return apellido
            else:
                return apellido[:8]
        else:
            # Una sola palabra, truncar
            return nombre_str[:8]

    def crear_leyenda_general(self, fig):
        """
        🔥 VERSIÓN FINAL: Leyenda que muestra el estilo real (fondo+texto),
        es dinámica y se centra perfectamente en la figura.
        """
        # 1. Definir estilo y elementos
        y_centro = 0.499  # Posición vertical (un poco por encima del centro)
        alto_leyenda = 0.012
        padding_horizontal = 0.015
        espacio_entre_elementos = 0.01

        elementos_info = []
        
        # 2. PRIMERA PASADA: Medir todos los elementos para calcular el ancho total
        ancho_total = 0
        for metrica in self.metricas_tabla:
            metrica_corta = metrica.replace('Distancia Total ', 'D').replace(' km / h', '').replace(' / min', '/m').replace('Velocidad Máxima Total', 'Vel. Máx.')
            
            # Crear un texto temporal para medir su ancho real
            temp_text = fig.text(0, 0, metrica_corta, fontsize=7, weight='bold', ha='center')
            bbox = temp_text.get_window_extent(fig.canvas.get_renderer())
            ancho_texto_fig = bbox.width / fig.dpi / fig.get_size_inches()[0]
            temp_text.remove() # Eliminar el texto temporal

            # Ancho del elemento = ancho del texto + un poco de padding
            ancho_elemento = ancho_texto_fig + 0.01 
            elementos_info.append({
                'texto': metrica_corta,
                'color': self.colores_metricas.get(metrica, '#2c3e50'),
                'ancho': ancho_elemento
            })
            ancho_total += ancho_elemento

        ancho_total += espacio_entre_elementos * (len(elementos_info) - 1)
        
        # 3. Calcular la posición de inicio para centrar todo el bloque
        x_inicio = (1 - (ancho_total + padding_horizontal * 2)) / 2

        # 4. Dibujar el fondo general de la leyenda (opcional, pero da un buen acabado)
        fondo = patches.FancyBboxPatch(
            (x_inicio, y_centro - alto_leyenda / 2),
            ancho_total + padding_horizontal * 2, alto_leyenda,
            boxstyle="round,pad=0.005,rounding_size=0.01",
            facecolor='#f0f0f0', alpha=0.9, edgecolor='black',
            linewidth=0.5, zorder=30, transform=fig.transFigure
        )
        fig.patches.append(fondo)

        # 5. SEGUNDA PASADA: Dibujar cada elemento de la leyenda
        x_offset = x_inicio + padding_horizontal
        for elemento in elementos_info:
            color_fondo = elemento['color']
            texto_str = elemento['texto']
            ancho = elemento['ancho']
            
            # El centro de este elemento
            centro_elemento_x = x_offset + ancho / 2
            
            # Determinar el color del texto para el contraste
            color_texto = self.get_contrast_color_for_metric(color_fondo)

            # Dibujar el texto con su fondo de color
            fig.text(centro_elemento_x, y_centro, texto_str,
                    ha='center', va='center', # <-- CENTRADO
                    fontsize=7, weight='bold',
                    color=color_texto,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color_fondo, edgecolor='white', linewidth=0.5),
                    transform=fig.transFigure, zorder=32
            )
            
            # Actualizar el offset para el siguiente elemento
            x_offset += ancho + espacio_entre_elementos

    def ajustar_texto_columna(self, texto, ancho_columna, scale):
        """Ajusta el texto para que encaje perfectamente en la columna"""
        if not texto or pd.isna(texto):
            return "N/A"
        
        texto_str = str(texto).strip()
        
        # Cálculo más preciso del espacio disponible
        chars_disponibles = int(ancho_columna / (0.25 * scale))
        
        if len(texto_str) <= chars_disponibles:
            return texto_str
        
        # Si es muy largo, usar estrategias de truncado inteligente
        partes = texto_str.split(' ')
        
        if len(partes) > 1:
            # Para nombres con espacios: Primera letra + apellido
            primer_nombre = partes[0]
            ultimo_apellido = partes[-1]
            
            if len(primer_nombre) + len(ultimo_apellido) + 2 <= chars_disponibles:
                return f"{primer_nombre[0]}.{ultimo_apellido}"
            else:
                return f"{primer_nombre[0]}.{ultimo_apellido[:chars_disponibles-3]}"
        else:
            # Para una sola palabra: truncar con puntos
            return texto_str[:chars_disponibles-1] + "."
    
    def crear_titulo_elegante_con_escudos(self, ax, titulo_texto, equipo_local, equipo_visitante, 
                              team_colors, y_position=0.95):
        """Método COMPLETO con título y escudos abajo con imshow"""
        
        # 📍 TÍTULO EN LA PARTE DE ABAJO CON FONDO AJUSTADO AL TEXTO
        bbox = ax.get_position()
        fig_x = bbox.x0 + bbox.width/2
        fig_y = bbox.y0 + 0.005  # ← Abajo del campo

        # 🔧 CALCULAR ANCHO REAL DEL TEXTO
        import matplotlib.pyplot as plt
        temp_fig = plt.figure(figsize=(1, 1))
        temp_ax = temp_fig.add_subplot(111)

        # Crear texto temporal para medir dimensiones
        temp_text = temp_ax.text(0, 0, titulo_texto, fontsize=10, weight='bold')
        temp_fig.canvas.draw()

        # Obtener dimensiones reales del texto
        bbox_texto = temp_text.get_window_extent()
        ancho_texto_pixels = bbox_texto.width
        alto_texto_pixels = bbox_texto.height

        # Convertir a coordenadas de figura
        ancho_texto_fig = ancho_texto_pixels / ax.figure.dpi / ax.figure.get_size_inches()[0]
        alto_texto_fig = alto_texto_pixels / ax.figure.dpi / ax.figure.get_size_inches()[1]

        # Cerrar figura temporal
        plt.close(temp_fig)

        # 🔧 DIMENSIONES AJUSTADAS CON PADDING
        padding_horizontal = ancho_texto_fig * 0.2  # 20% de padding horizontal
        padding_vertical = alto_texto_fig * 0.3     # 30% de padding vertical

        titulo_width = ancho_texto_fig + padding_horizontal
        titulo_height = alto_texto_fig + padding_vertical

        # Fondo del título AJUSTADO
        fondo_rect = plt.Rectangle(
            (fig_x - titulo_width/2, fig_y - titulo_height/2), 
            titulo_width, titulo_height,
            facecolor=team_colors['primary'], 
            alpha=0.9,
            edgecolor='white', 
            linewidth=0.5,
            transform=ax.figure.transFigure,
            zorder=10
        )
        ax.figure.patches.append(fondo_rect)

        # Texto del título
        ax.figure.text(
            fig_x, fig_y, titulo_texto,
            fontsize=10, weight='bold', color=team_colors['text'],
            ha='center', va='center',
            transform=ax.figure.transFigure,
            zorder=12
        )
        
        # 🏆 ESCUDOS CON SOMBRA
        escudo_local = self.load_team_logo(equipo_local)
        escudo_visitante = self.load_team_logo(equipo_visitante)

        # Escudo LOCAL CON SOMBRA
        if escudo_local is not None:
            try:
                # 1. SOMBRA (desplazada hacia abajo-derecha)
                ax.imshow(escudo_local, 
                        extent=[2.5, 21.5, -0.5, 12.5],  # ← SOMBRA desplazada
                        aspect='auto', zorder=99, alpha=0.4)  # ← Semi-transparente y atrás
                
                # 2. ESCUDO PRINCIPAL (encima de la sombra)
                ax.imshow(escudo_local, 
                        extent=[1, 20, 1, 14],  # ← ESCUDO ORIGINAL
                        aspect='auto', zorder=100)  # ← Adelante
                
            except Exception as e:
                print(f"⚠️ Error escudo local: {e}")

        # Escudo VISITANTE CON SOMBRA
        if escudo_visitante is not None:
            try:
                # 1. SOMBRA (desplazada hacia abajo-izquierda)
                ax.imshow(escudo_visitante, 
                        extent=[98.5, 117.5, -0.5, 12.5],  # ← SOMBRA desplazada
                        aspect='auto', zorder=99, alpha=0.4)  # ← Semi-transparente y atrás
                
                # 2. ESCUDO PRINCIPAL (encima de la sombra)
                ax.imshow(escudo_visitante, 
                        extent=[100, 119, 1, 14],  # ← ESCUDO ORIGINAL
                        aspect='auto', zorder=100)  # ← Adelante
                
            except Exception as e:
                print(f"⚠️ Error escudo visitante: {e}")

    def calcular_area_jugador(self, scale=0.8):
        """Calcula el área que ocupa un jugador circular"""
        radio_metricas = 8 * scale  # ← AUMENTADO: era 7, ahora 8
        margen = 4 * scale          # ← AUMENTADO: era 3, ahora 4
        
        diametro_total = (radio_metricas + margen) * 2
        return diametro_total, diametro_total

    def verificar_colision(self, x1, y1, width1, height1, x2, y2, width2, height2, margen=1.8): # <-- AUMENTAMOS EL MARGEN
        """Verifica si dos tablas se superponen con un margen de seguridad más grande"""
        # Expandir áreas con margen
        left1, right1 = x1 - width1/2 - margen, x1 + width1/2 + margen
        bottom1, top1 = y1 - height1/2 - margen, y1 + height1/2 + margen
        
        left2, right2 = x2 - width2/2 - margen, x2 + width2/2 + margen
        bottom2, top2 = y2 - height2/2 - margen, y2 + height2/2 + margen
        
        # Verificar superposición
        return not (right1 < left2 or right2 < left1 or top1 < bottom2 or top2 < bottom1)

    def encontrar_posicion_cercana_libre(self, posicion_ideal, posiciones_ocupadas, width, height, coordenadas_reservadas=None):
        """
        Busca posición libre evitando tanto ocupadas como coordenadas originales del sistema
        """
        if coordenadas_reservadas is None:
            coordenadas_reservadas = []
        
        # 1. Probar la posición ideal primero
        es_libre = self.verificar_posicion_libre(posicion_ideal, width, height, posiciones_ocupadas, coordenadas_reservadas)
        if es_libre:
            return posicion_ideal

        # 2. Probar desplazamientos
        desplazamientos = [
            (0, 15), (0, -15), (15, 0), (-15, 0),
            (12, 12), (-12, -12), (12, -12), (-12, 12),
            (0, 25), (25, 0), (0, -25), (-25, 0),
            (18, 18), (-18, -18), (18, -18), (-18, 18)
        ]

        for dx, dy in desplazamientos:
            x_test, y_test = posicion_ideal[0] + dx, posicion_ideal[1] + dy
            
            if self.verificar_posicion_libre((x_test, y_test), width, height, posiciones_ocupadas, coordenadas_reservadas):
                return (x_test, y_test)

        # 3. Búsqueda en grilla si no encuentra
        return self.buscar_posicion_en_grilla_mejorada(posicion_ideal, posiciones_ocupadas, width, height, coordenadas_reservadas)

    def buscar_posicion_en_grilla_mejorada(self, posicion_ideal, posiciones_ocupadas, width, height, coordenadas_reservadas):
        """Búsqueda sistemática evitando coordenadas reservadas"""
        x_ideal, y_ideal = posicion_ideal
        
        for radio in [20, 30, 40, 50]:
            for angulo in range(0, 360, 30):
                x_test = x_ideal + radio * np.cos(np.radians(angulo))
                y_test = y_ideal + radio * np.sin(np.radians(angulo))
                
                if self.verificar_posicion_libre((x_test, y_test), width, height, posiciones_ocupadas, coordenadas_reservadas):
                    return x_test, y_test
        
        print(f"   ⚠️ No se encontró posición libre - usando ideal con riesgo de superposición")
        return posicion_ideal

    def verificar_posicion_libre(self, posicion, width, height, posiciones_ocupadas, coordenadas_reservadas):
        """
        Verifica si una posición está libre de ocupadas Y de coordenadas reservadas
        """
        x, y = posicion
        
        # Verificar límites del campo
        if not (10 <= x <= 110 and 10 <= y <= 70):
            return False
        
        # Verificar colisión con posiciones ya ocupadas
        for pos in posiciones_ocupadas:
            if self.verificar_colision(x, y, width, height, pos['x'], pos['y'], pos['width'], pos['height']):
                return False
        
        # NUEVO: Verificar colisión con coordenadas reservadas (originales del sistema)
        for coord_reservada in coordenadas_reservadas:
            if self.verificar_colision(x, y, width, height, coord_reservada[0], coord_reservada[1], width, height):
                return False
        
        return True
    
    def ajustar_posicion_a_limites(self, x, y, escala):
        """
        Asegura que las coordenadas del centro de un jugador no se salgan de los 
        límites visibles del campo, dejando un margen para el radio del círculo.
        """
        # Calculamos el área total que ocupa un jugador para saber su radio
        width, height = self.calcular_area_jugador(scale=escala)
        radio = width / 2  # El radio es la mitad del diámetro

        # Límites del campo (0-120 para X, 0-80 para Y) con un margen igual al radio
        limite_x_min = radio
        limite_x_max = 120 - radio
        limite_y_min = radio
        limite_y_max = 80 - radio

        # Forzamos los valores a estar dentro de los límites seguros
        x_ajustado = max(limite_x_min, min(x, limite_x_max))
        y_ajustado = max(limite_y_min, min(y, limite_y_max))
        
        # Si hemos tenido que ajustar la posición, lo indicamos en la consola
        if x_ajustado != x or y_ajustado != y:
            print(f"   -> ⚠️ Posición ({x:.1f}, {y:.1f}) ajustada a los límites del campo -> ({x_ajustado:.1f}, {y_ajustado:.1f})")

        return x_ajustado, y_ajustado

        
    def guardar_sin_espacios(self, fig, filename):
        """🔥 MÉTODO ORIGINAL: Guarda sin espacios manteniendo landscape 16:9"""
        # Ajustar tamaño para 16:9 antes de guardar
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

    def crear_4_partidos_campos_horizontales(self, equipo, jornada_maxima, tipo_partido_filter=None, figsize=(11.69, 8.27)):
        """
        🔥 VERSIÓN FINAL: Crea la visualización 2x2 utilizando exclusivamente la formación
        táctica extraída de los eventos de Opta para posicionar a los jugadores.
        """
        tipo_display = tipo_partido_filter.upper() if tipo_partido_filter else "TODOS"
        
        # 1. OBTENER LOS DATOS DE LOS ÚLTIMOS 4 PARTIDOS
        partidos = self.get_ultimos_4_partidos(equipo, jornada_maxima, tipo_partido_filter)
        
        if not partidos:
            print(f"❌ No se encontraron partidos de tipo '{tipo_display}' para {equipo} hasta la jornada {jornada_maxima}.")
            return None
        
        # 2. CONFIGURACIÓN INICIAL DE LA FIGURA Y EJES
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        fig.subplots_adjust(left=0, right=1, top=0.90, bottom=0, wspace=0.0, hspace=0.0)
        fig.patch.set_facecolor('#f0f0f0') # Un color de fondo suave por si acaso

        # Crear los 4 ejes (subplots) para los campos
        axes = []
        subplot_positions = [
            [0.0, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 0.5, 0.5]
        ]
        for pos in subplot_positions:
            ax = fig.add_axes(pos)
            ax.set_aspect('equal')
            axes.append(ax)

        # Añadir imagen de fondo general a toda la figura
        background_img = self.load_background_image()
        if background_img is not None:
            ax_background = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_background.imshow(background_img, extent=[0, 1, 0, 1], aspect='auto', alpha=0.15)
            ax_background.axis('off')

        # Cargar recursos del equipo
        team_colors = self.get_team_colors(equipo)
        team_logo = self.load_team_logo(equipo)
        
        # 3. BUCLE PRINCIPAL PARA DIBUJAR CADA UNO DE LOS 4 CAMPOS
        for i in range(4):
            ax = axes[i]
            
            if i < len(partidos):
                # --- Hay un partido para dibujar en este eje ---
                partido_info = partidos[i]
                
                # Dibujar el terreno de juego
                self.crear_campo_sin_espacios(ax)
                
                # 🔥 LÓGICA CENTRAL: Obtener la alineación y coordenadas desde Opta
                formation_result = self.get_formation_coordinates_for_match_FIXED(
                    equipo, 
                    partido_info['jornada'], 
                    partido_info['datos'] # El DataFrame con los datos físicos de este partido
                )

                # Dibujar los jugadores SI se encontró la formación
                if formation_result and formation_result.get('jugadores'):
                    escala = 0.9
                    # Llama a la nueva función de dibujado que no tiene anti-colisiones
                    self.dibujar_alineacion_desde_formacion(
                        formation_result['jugadores'], ax, team_colors, escala, team_logo
                    )
                else:
                    # Si no se encontraron datos de formación, mostrar un mensaje claro
                    print(f"   ❌ No se encontró formación en Opta para J{partido_info['jornada']}.")
                    ax.text(60, 40, 'Formación Táctica\nno disponible en Opta', 
                        ha='center', va='center', fontsize=12, color='white', weight='bold',
                        bbox=dict(facecolor='red', alpha=0.7, boxstyle='round,pad=0.5'))

                # Añadir título con resultado y escudos al campo
                equipo_local, equipo_visitante, gl, gv, tipo = self.parsear_partido_completo(partido_info['partido'], equipo)
                titulo = f"J{partido_info['jornada']} - {equipo_local} {gl}-{gv} {equipo_visitante}"
                self.crear_titulo_elegante_con_escudos(ax, titulo, equipo_local, equipo_visitante, team_colors)
                
            else:
                # --- No hay más partidos para dibujar, rellenar con un campo vacío ---
                ax.set_xlim(0, 120); ax.set_ylim(0, 80)
                ax.text(60, 40, f'Sin partido {tipo_display.lower()}\ndisponible', 
                    ha='center', va='center', fontsize=12, color='gray', style='italic')
                ax.set_facecolor('lightgray'); ax.set_alpha(0.3)
                ax.axis('off')
        
        # 4. TÍTULOS FINALES Y LEYENDA
        fig.suptitle(f'{equipo.upper()} - ÚLTIMOS 4 PARTIDOS {tipo_display} (hasta J{jornada_maxima})',
            fontsize=10, weight='bold', color='white', y=0.98,
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#1e3d59', alpha=0.95, edgecolor='white'))
        
        self.crear_leyenda_general(fig)
        
        return fig


def main_4_campos_horizontales_coordenadas_fijas():
    """Función principal con coordenadas fijas"""
    try:
        report_gen = ReporteTactico4CamposHorizontalesMejorado()
        if hasattr(report_gen, 'events_df') and report_gen.events_df is not None:
            pass
        else:
            pass

        equipos = report_gen.get_available_teams()
        
        if not equipos:
            print("❌ No hay equipos disponibles")
            return
        
        
        while True:
            try:
                seleccion = input(f"\nSelecciona equipo (1-{len(equipos)}): ").strip()
                indice = int(seleccion) - 1
                if 0 <= indice < len(equipos):
                    equipo_seleccionado = equipos[indice]
                    break
                else:
                    print(f"❌ Número entre 1 y {len(equipos)}")
            except ValueError:
                print("❌ Ingresa un número válido")
        
        # Jornada
        jornadas = report_gen.get_available_jornadas()
        
        while True:
            try:
                jornada_input = input("Jornada máxima a considerar: ").strip()
                if jornada_input.startswith(('J', 'j')):
                    jornada = int(jornada_input[1:])
                else:
                    jornada = int(jornada_input)
                
                if jornada in jornadas:
                    break
                else:
                    print(f"❌ Jornada no disponible")
            except ValueError:
                print("❌ Formato de jornada inválido")
        
        # Tipo de partido
        
        while True:
            try:
                tipo_seleccion = input("Selecciona opción (1-3): ").strip()
                if tipo_seleccion == "1":
                    tipo_partido_filter = "local"
                    break
                elif tipo_seleccion == "2":
                    tipo_partido_filter = "visitante"
                    break
                elif tipo_seleccion == "3":
                    tipo_partido_filter = None
                    break
                else:
                    print("❌ Selecciona 1, 2 o 3")
            except ValueError:
                print("❌ Ingresa 1, 2 o 3")
        
        # Generar
        fig = report_gen.crear_4_partidos_campos_horizontales(equipo_seleccionado, jornada, tipo_partido_filter)
        
        if fig:
            plt.show()
            
            # Guardar con método mejorado
            tipo_filename = f"_{tipo_partido_filter}" if tipo_partido_filter else "_todos"
            filename = f"reporte_4_campos_FIJAS_{equipo_seleccionado.replace(' ', '_')}_hasta_J{jornada}{tipo_filename}.pdf"
            report_gen.guardar_sin_espacios(fig, filename)
        else:
            print("❌ No se pudo generar el reporte")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def generar_4_campos_coordenadas_fijas(equipo, jornada_maxima, tipo_partido_filter=None, mostrar=True, guardar=True):
    """Función para uso directo con coordenadas fijas"""
    try:
        report_gen = ReporteTactico4CamposHorizontalesMejorado()
        fig = report_gen.crear_4_partidos_campos_horizontales(equipo, jornada_maxima, tipo_partido_filter)
        
        if fig:
            if mostrar:
                plt.show()
            if guardar:
                tipo_filename = f"_{tipo_partido_filter}" if tipo_partido_filter else "_todos"
                filename = f"reporte_4_campos_FIJAS_{equipo.replace(' ', '_')}_hasta_J{jornada_maxima}{tipo_filename}.pdf"
                report_gen.guardar_sin_espacios(fig, filename)
            return fig
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Funciones rápidas con coordenadas fijas
def rapido_horizontal_fijas(equipo, jornada=35, tipo_partido=None):
    """Genera 4 campos horizontales CON COORDENADAS FIJAS rápidamente"""
    tipo_display = tipo_partido.upper() if tipo_partido else "TODOS"
    return generar_4_campos_coordenadas_fijas(equipo, jornada, tipo_partido)

def sevilla_horizontal_fijas(tipo_partido=None):
    """Sevilla FC con campos horizontales COORDENADAS FIJAS"""
    return rapido_horizontal_fijas("Sevilla FC", 35, tipo_partido)

def sevilla_horizontal_local_fijas():
    """Sevilla FC solo locales horizontal COORDENADAS FIJAS"""
    return sevilla_horizontal_fijas("local")

def sevilla_horizontal_visitante_fijas():
    """Sevilla FC solo visitantes horizontal COORDENADAS FIJAS"""
    return sevilla_horizontal_fijas("visitante")

# Inicialización
try:
    report_gen = ReporteTactico4CamposHorizontalesMejorado()
    equipos = report_gen.get_available_teams()
    jornadas = report_gen.get_available_jornadas()
except Exception as e:
    print(f"❌ Error inicialización: {e}")

if __name__ == "__main__":
    main_4_campos_horizontales_coordenadas_fijas()

