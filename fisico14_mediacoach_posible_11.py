import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import matplotlib.patheffects as patheffects
import re
import unicodedata
import json
import glob
import os
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

# CONFIGURACIÓN GLOBAL PARA ELIMINAR ESPACIOS
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

# Instalar mplsoccer si no está instalado
try:
    from mplsoccer import Pitch
except ImportError:
    print("Instalando mplsoccer...")
    import subprocess
    subprocess.check_call(["pip", "install", "mplsoccer"])
    from mplsoccer import Pitch

class Posible11Inicial:
    def __init__(self, data_path="extraccion_mediacoach/data/rendimiento_fisico.parquet"):
        """
        Inicializa la clase para generar el posible 11 inicial
        """
        self.data_path = data_path
        self.df = None
        self.load_data()
        self.clean_team_names()
        self.events_df = None
        self.load_match_events()
        

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
                7: (70, 15), 11: (70, 65), 4: (62, 30), 8: (62, 50),
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
                7: (65, 10), 11: (65, 70), 4: (60, 30), 10: (60, 40), 8: (60, 50),
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
                1: (10, 40), 2: (48, 15), 3: (48, 65), 4: (28, 60), 5: (28, 20), 6: (28, 40),
                7: (70, 20), 8: (70, 60), 10: (70, 40),
                11: (100, 50), 9: (100, 30)
            },
            11: {  # 541
                1: (10, 40), 2: (45, 12), 3: (45, 70), 4: (30, 55), 5: (30, 25), 6: (30, 40),
                7: (70, 15), 11: (70, 65), 8: (70, 35), 10: (70, 45),
                9: (100, 40)
            },
            12: {  # 352
                1: (10, 40), 2: (45, 12), 3: (45, 70), 4: (30, 55), 5: (30, 25), 6: (30, 40),
                7: (70, 25), 11: (70, 40), 8: (70, 55),
                10: (100, 30), 9: (100, 50)
            },
            13: {  # 343
                1: (10, 40), 2: (45, 12), 3: (45, 70), 4: (30, 55), 5: (30, 25), 6: (30, 40),
                7: (65, 25), 8: (65, 55),
                10: (95, 15), 11: (95, 65), 9: (95, 40)
            },
            15: {  # 4222
                1: (10, 40), 2: (40, 15), 3: (40, 65), 5: (25, 28), 6: (25, 52),
                4: (60, 30), 8: (60, 50),
                7: (80, 20), 11: (80, 60), 10: (100, 25), 9: (100, 55)
            },
            16: {  # 3511
                1: (10, 40), 2: (45, 12), 3: (45, 70), 4: (30, 55), 5: (30, 25), 6: (30, 40),
                7: (65, 25), 11: (65, 40), 8: (65, 55),
                10: (85, 40), 9: (105, 40)
            },
            17: {  # 3421
                1: (10, 40), 2: (45, 12), 3: (45, 70), 4: (30, 55), 5: (30, 25), 6: (30, 40),
                7: (65, 25), 8: (65, 55), 10: (85, 30), 11: (85, 50),
                9: (105, 40)
            },
            18: {  # 3412
                1: (10, 40), 2: (45, 12), 3: (45, 70), 4: (30, 55), 5: (30, 25), 6: (30, 40),
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

        # 🎨 COLORES ESPECÍFICOS POR EQUIPO
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

        # Colores por defecto para equipos no reconocidos
        self.default_team_colors = {'primary': '#2c3e50', 'secondary': '#FFFFFF', 'text': 'white'}

        # MAPEO EXPLÍCITO DE NOMBRES DE EQUIPOS
        self.team_name_mapping = {
            'RC Celta': ['Celta de Vigo', 'Celta Vigo', 'Celta'],
            'Athletic Club': ['Athletic Bilbao'],
            'Atlético de Madrid': ['Atletico Madrid', 'Atletico'],  # <- CORREGIDO
            'Real Betis': ['Real Betis Balompie'],
            'Real Sociedad': ['Real Sociedad de Futbol'],
        }
        
        # Mapeo de demarcaciones a posiciones específicas
        self.demarcacion_to_position = {
            'Portero': 'PORTERO',
            'Defensa - Central Derecho': 'CENTRAL_DERECHO',
            'Defensa - Central Izquierdo': 'CENTRAL_IZQUIERDO',
            'Defensa - Central': 'CENTRAL_DERECHO',
            'Central': 'CENTRAL_DERECHO',
            'Defensa - Lateral Derecho': 'LATERAL_DERECHO',
            'Defensa - Lateral Izquierdo': 'LATERAL_IZQUIERDO',
            'Lateral Derecho': 'LATERAL_DERECHO',
            'Lateral Izquierdo': 'LATERAL_IZQUIERDO',
            'Centrocampista - MC Box to Box': 'MC_BOX_TO_BOX',
            'Centrocampista - MC Organizador': 'MC_ORGANIZADOR',
            'Centrocampista - MC Posicional': 'MC_POSICIONAL',
            'Pivote': 'MC_POSICIONAL',
            'Mediocentro': 'MC_ORGANIZADOR',
            'Interior': 'MC_BOX_TO_BOX',
            'Centrocampista de ataque - Banda Derecha': 'BANDA_DERECHA',
            'Centrocampista de ataque - Banda Izquierda': 'BANDA_IZQUIERDA',
            'Centrocampista de ataque - Mediapunta': 'MEDIAPUNTA',
            'Mediapunta': 'MEDIAPUNTA',
            'Delantero - Delantero Centro': 'DELANTERO_CENTRO',
            'Delantero - Segundo Delantero': 'SEGUNDO_DELANTERO',
            'Delantero': 'DELANTERO_CENTRO',
            'Extremo Derecho': 'BANDA_DERECHA',
            'Extremo Izquierdo': 'BANDA_IZQUIERDA',
            'Sin Posición': 'MC_POSICIONAL',
        }
        
        self.metricas_tabla = [
            'Distancia Total', 'Distancia Total / min', 'Distancia Total 14-21 km / h',
            'Distancia Total 14-21 km / h / min', 'Distancia Total >21 km / h', 
            'Distancia Total >21 km / h / min', 'Distancia Total >24 km / h',
            'Distancia Total >24 km / h / min', 'Velocidad Máxima Total'
        ]

        self.colores_metricas = {
            'Distancia Total': '#d32f2f', 'Distancia Total / min': '#00796b',
            'Distancia Total 14-21 km / h': '#ffc107', 'Distancia Total 14-21 km / h / min': '#5d4037',
            'Distancia Total >21 km / h': '#f57c00', 'Distancia Total >21 km / h / min': '#0288d1',
            'Distancia Total >24 km / h': '#c2185b', 'Distancia Total >24 km / h / min': '#90a4ae',
            'Velocidad Máxima Total': '#6a1b9a'
        }

        self.coordenadas_posiciones = {
            'PORTERO': (10, 40), 'LATERAL_DERECHO': (50, 15), 'CENTRAL_DERECHO': (30, 20),
            'CENTRAL_CENTRO': (30, 40), 'CENTRAL_IZQUIERDO': (30, 60), 'LATERAL_IZQUIERDO': (50, 65),
            'MC_POSICIONAL': (47, 40), 'MC_BOX_TO_BOX': (71, 25), 'MC_ORGANIZADOR': (68, 55),
            'MEDIAPUNTA': (90, 40), 'BANDA_DERECHA': (90, 15), 'BANDA_IZQUIERDA': (90, 65),
            'DELANTERO_CENTRO': (108, 45), 'SEGUNDO_DELANTERO': (107, 20),
        }

        # 🔥 DICCIONARIO DE POSICIONES AMIGAS AÑADIDO AQUÍ 🔥
        self.friendly_positions_map = {
            'MC_ORGANIZADOR': ['MC_BOX_TO_BOX', 'MC_POSICIONAL', 'MEDIAPUNTA'],
            'MC_BOX_TO_BOX': ['MC_ORGANIZADOR', 'MEDIAPUNTA', 'MC_POSICIONAL'],
            'MC_POSICIONAL': ['MC_ORGANIZADOR', 'MC_BOX_TO_BOX'],
            'MEDIAPUNTA': ['MC_BOX_TO_BOX', 'SEGUNDO_DELANTERO', 'MC_ORGANIZADOR'],
            'LATERAL_IZQUIERDO': ['BANDA_IZQUIERDA'],
            'BANDA_IZQUIERDA': ['LATERAL_IZQUIERDO'],
            'LATERAL_DERECHO': ['BANDA_DERECHA'],
            'BANDA_DERECHA': ['LATERAL_DERECHO'],
            'DELANTERO_CENTRO': ['SEGUNDO_DELANTERO'],
            'SEGUNDO_DELANTERO': ['DELANTERO_CENTRO'],
            'CENTRAL_DERECHO': ['CENTRAL_IZQUIERDO', 'CENTRAL_CENTRO', 'MC_POSICIONAL'],
            'CENTRAL_IZQUIERDO': ['CENTRAL_DERECHO', 'CENTRAL_CENTRO', 'MC_POSICIONAL'],
            'CENTRAL_CENTRO': ['CENTRAL_DERECHO', 'CENTRAL_IZQUIERDO', 'MC_POSICIONAL'],
        }

        # Mapeo de demarcaciones a posiciones específicas
        self.demarcacion_to_position = {
            # Portero
            'Portero': 'PORTERO',
            
            # Defensas
            'Defensa - Central Derecho': 'CENTRAL_DERECHO',
            'Defensa - Central Izquierdo': 'CENTRAL_IZQUIERDO',
            'Defensa - Central': 'CENTRAL_DERECHO', # Genérico
            'Central': 'CENTRAL_DERECHO', # Genérico
            'Defensa - Lateral Derecho': 'LATERAL_DERECHO',
            'Defensa - Lateral Izquierdo': 'LATERAL_IZQUIERDO',
            'Lateral Derecho': 'LATERAL_DERECHO',
            'Lateral Izquierdo': 'LATERAL_IZQUIERDO',
            
            # Mediocampo
            'Centrocampista - MC Box to Box': 'MC_BOX_TO_BOX',
            'Centrocampista - MC Organizador': 'MC_ORGANIZADOR',
            'Centrocampista - MC Posicional': 'MC_POSICIONAL',
            'Pivote': 'MC_POSICIONAL',
            'Mediocentro': 'MC_ORGANIZADOR',
            'Interior': 'MC_BOX_TO_BOX',
            'Centrocampista de ataque - Banda Derecha': 'BANDA_DERECHA',
            'Centrocampista de ataque - Banda Izquierda': 'BANDA_IZQUIERDA',
            'Centrocampista de ataque - Mediapunta': 'MEDIAPUNTA',
            'Mediapunta': 'MEDIAPUNTA',

            # Delanteros
            'Delantero - Delantero Centro': 'DELANTERO_CENTRO',
            'Delantero - Segundo Delantero': 'SEGUNDO_DELANTERO',
            'Delantero': 'DELANTERO_CENTRO', # Genérico
            'Extremo Derecho': 'BANDA_DERECHA',
            'Extremo Izquierdo': 'BANDA_IZQUIERDA',
            
            # Sin posición
            'Sin Posición': 'MC_POSICIONAL',
        }
        
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

        self.colores_metricas = {
            'Distancia Total': '#d32f2f',
            'Distancia Total / min': '#00796b',
            'Distancia Total 14-21 km / h': '#ffc107',
            'Distancia Total 14-21 km / h / min': '#5d4037',
            'Distancia Total >21 km / h': '#f57c00',
            'Distancia Total >21 km / h / min': '#0288d1',
            'Distancia Total >24 km / h': '#c2185b',
            'Distancia Total >24 km / h / min': '#90a4ae',
            'Velocidad Máxima Total': '#6a1b9a'
        }
        # Coordenadas para posicionar las tablas en el campo (formación 4-3-3)
        self.coordenadas_posiciones = {
            'PORTERO': (10, 40),
            'LATERAL_DERECHO': (50, 15),
            'CENTRAL_DERECHO': (30, 20),
            'CENTRAL_CENTRO': (30, 40),
            'CENTRAL_IZQUIERDO': (30, 60),
            'LATERAL_IZQUIERDO': (50, 65),
            'MC_POSICIONAL': (47, 40),
            'MC_BOX_TO_BOX': (71, 25),
            'MC_ORGANIZADOR': (68, 55),
            'MEDIAPUNTA': (90, 40),
            'BANDA_DERECHA': (90, 15),
            'BANDA_IZQUIERDA': (90, 65),
            'DELANTERO_CENTRO': (108, 45),
            'SEGUNDO_DELANTERO': (107, 20),
        }

    @staticmethod
    def normalize_text(text):
        """Normaliza texto eliminando acentos, espacios extra y caracteres especiales."""
        import re
        import unicodedata
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_team_logo_as_array(self, equipo):
        """Convierte el escudo del equipo a array compatible con las fotos"""
        team_logo = self.load_team_logo(equipo)
        if team_logo is not None:
            # Convertir a formato RGBA si no lo está
            if len(team_logo.shape) == 3 and team_logo.shape[2] == 3:
                # RGB -> RGBA
                alpha_channel = np.ones((team_logo.shape[0], team_logo.shape[1], 1))
                team_logo = np.concatenate([team_logo, alpha_channel], axis=2)
            
            return team_logo.astype(np.float32) / 255.0 if team_logo.dtype != np.float32 else team_logo
        return None

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

    def get_team_setup_from_events_FIXED(self, team_name, week):
        """
        🔥 VERSIÓN FINAL CORREGIDA: Utiliza 'Team Player Formation' como máscara para
        extraer y ordenar correctamente a los 11 jugadores titulares.
        """
        if self.events_df is None:
            print("⚠️ Eventos no cargados, imposible buscar formación.")
            return None

        week_str = str(week)
        print(f"🔍 Buscando 'Team set up' para '{team_name}' - Week '{week_str}'")
        
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
                print(f"   -> ✅ Match de equipo encontrado: '{team_name}' ≈ '{opta_team}'")
                break
        
        if team_setup is None:
            print(f"   -> ❌ No se encontró setup para el equipo '{team_name}' en esta Week.")
            return None
        
        # --- INICIO DE LA NUEVA LÓGICA DE FILTRADO ---
        
        # 1. Extraer los tres datos clave
        formation_number = team_setup.get('Team Formation')
        jersey_numbers_str = str(team_setup.get('Jersey Number', ''))
        player_formation_str = str(team_setup.get('Team Player Formation', '')) # <-- LA MÁSCARA

        print("\n   --- Procesando Alineación ---")
        print(f"   - Jersey Numbers (raw): {jersey_numbers_str[:70]}...")
        print(f"   - Player Formation (máscara): {player_formation_str[:70]}...")

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
            print(f"      Alineación resultante: {ordered_starters}")
            return None

        print(f"   -> ✅ Alineación titular filtrada y ordenada: {ordered_starters}")

        # 5. Crear el mapeo final, ahora sí, solo con los 11 titulares
        position_jersey_map = {i + 1: jersey for i, jersey in enumerate(ordered_starters)}
        
        try:
            formation_number = int(formation_number)
        except (ValueError, TypeError):
            print(f"   -> ❌ Error: 'Team Formation' no es un número válido: {formation_number}")
            return None

        formation_name = self.formation_mapping.get(formation_number, f"Unknown_{formation_number}")
        
        print(f"   -> 🎯 ÉXITO: Formación {formation_name} con {len(ordered_starters)} titulares encontrada.")
        
        return {
            'formation_number': formation_number,
            'formation_name': formation_name,
            'position_jersey_map': position_jersey_map,
            'total_players': len(ordered_starters),
            'raw_jerseys': ordered_starters # Devolvemos la lista limpia
        }

    def buscar_posicion_opta_por_partido(self, dorsal, equipo, jornada):
        """Busca la posición de un jugador en Opta para una jornada específica."""
        if self.opta_df is None or pd.isna(dorsal):
            return None

        week = self.convert_jornada_to_week(jornada)
        if week is None:
            return None

        try:
            dorsal_str = str(int(float(dorsal)))
        except (ValueError, TypeError):
            return None

        # Filtrar Opta por jornada y dorsal para eficiencia
        match_df = self.opta_df[(self.opta_df['Week'] == str(week))]  # FIX: convertir a string
        
        for _, row in match_df.iterrows():
            opta_dorsal_raw = row.get('Shirt Number')
            opta_team = row.get('Team Name', '')

            if pd.notna(opta_dorsal_raw):
                try:
                    opta_dorsal = str(int(float(opta_dorsal_raw)))
                    # Comprobar si dorsal y equipo coinciden
                    if dorsal_str == opta_dorsal and self.are_teams_equivalent(equipo, opta_team):
                        position = row.get('Position')
                        # Filtrar aquí las posiciones no deseadas
                        if pd.notna(position) and str(position).strip() not in ["Substitute", "Not Used", "", "nan"]:
                            position_side = row.get('Position Side')
                            return self.mapear_posicion_opta_a_coordenadas(position, position_side)
                except (ValueError, TypeError):
                    continue
        return None

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

    def get_player_photo_without_dorsal(self, player_data, photos_data, equipo=None):
        """
        🔥 VERSIÓN FINAL: Busca la foto del jugador con una estrategia de dos pasos:
        1. Intenta encontrarlo por su nombre completo (Nombre + Apellido).
        2. Si falla, intenta encontrarlo por su Alias.
        Esto soluciona el problema de nombres legales vs. nombres profesionales (ej: Pedri).
        """
        match = None
        if not isinstance(player_data, dict):
            # Fallback por si llega un string en lugar de un diccionario
            match = self.match_player_name(str(player_data), photos_data, equipo)
        else:
            # --- Estrategia de búsqueda en dos pasos ---
            
            # 1. Primer intento: Usar el nombre completo y legal
            full_name = self.construir_nombre_completo(player_data)
            match = self.match_player_name(full_name, photos_data, equipo)

            # 2. Segundo intento (Fallback): Si el primero falla, usar el Alias
            if not match:
                alias = player_data.get('Alias')
                # Comprobar que el alias es válido y no es el mismo que ya hemos buscado
                if pd.notna(alias) and alias.strip() and alias != full_name:
                    match = self.match_player_name(alias, photos_data, equipo)
        
        # Si después de ambos intentos no hay coincidencia, usar el logo del equipo
        if not match:
            return self.load_team_logo_as_array(equipo)
        
        # --- Procesamiento de la imagen (sin cambios) ---
        try:
            import base64
            from io import BytesIO
            from PIL import Image
            
            img_data = base64.b64decode(match['image_base64'])
            img = Image.open(BytesIO(img_data))
            
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            data = np.array(img)
            
            if len(data.shape) != 3 or data.shape[2] != 4:
                return None
            
            height, width = data.shape[:2]
            
            def flood_fill_iterative(start_points, threshold=235):
                visited = np.zeros((height, width), dtype=bool)
                background_mask = np.zeros((height, width), dtype=bool)
                
                def is_background_color(y, x):
                    if y < 0 or y >= height or x < 0 or x >= width: return False
                    return (data[y, x, 0] >= threshold and 
                            data[y, x, 1] >= threshold and 
                            data[y, x, 2] >= threshold)
                
                for start_y, start_x in start_points:
                    if visited[start_y, start_x] or not is_background_color(start_y, start_x): continue
                    
                    stack = [(start_y, start_x)]
                    while stack:
                        y, x = stack.pop()
                        if (y < 0 or y >= height or x < 0 or x >= width or 
                            visited[y, x] or not is_background_color(y, x)): continue
                        
                        visited[y, x] = True
                        background_mask[y, x] = True
                        stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
                return background_mask
            
            border_points = [
                (0, 0), (0, width-1), (height-1, 0), (height-1, width-1),
                (0, width//2), (height-1, width//2), (height//2, 0), (height//2, width-1),
                (0, width//4), (0, 3*width//4), (height-1, width//4), (height-1, 3*width//4),
                (height//4, 0), (3*height//4, 0), (height//4, width-1), (3*height//4, width-1)
            ]
            
            background_mask = flood_fill_iterative(border_points, threshold=220)
            data[background_mask] = [0, 0, 0, 0]
            
            return data.astype(np.float32) / 255.0
        
        except Exception as e:
            player_name_for_error = player_data.get('Alias', 'Desconocido')
            print(f"⚠️ Error procesando foto de {player_name_for_error}: {e}")
            return None

    def extract_names_parts(self, name):
        """Extrae las partes de un nombre normalizado"""
        def normalize_name(name):
            """Normaliza un nombre eliminando acentos y caracteres especiales"""
            if not name:
                return ""
            name = str(name).lower().strip()
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

    def calculate_match_score(self, player_parts, photo_parts):
        """
        Calcula el puntaje de coincidencia con lógica MEJORADA para apodos y nombres compuestos.
        """
        player_full = player_parts['full']
        photo_full = photo_parts['full']
        
        # ✨ NUEVA REGLA 0: MATCH EXACTO EN APODOS (La más prioritaria)
        # Comprueba si el nombre del jugador (ej: "vini") está en la lista de alias de la foto
        photo_aliases = [alias.lower() for alias in photo_parts.get('entry', {}).get('aliases', [])]
        if player_full in photo_aliases:
            return (1.0, f"MATCH EXACTO DE APODO '{player_full}'")

        # Regla 1: Match exacto del nombre completo
        if player_full == photo_full:
            return (1.0, "MATCH EXACTO COMPLETO")

        # ✨ NUEVA REGLA 1.5: EL APODO ES EL INICIO DE UNA PALABRA (ej. "Vini" en "Vinicius")
        # Esto funciona si no has definido el alias en el JSON
        player_words = player_parts['all_parts']
        photo_words = photo_parts['all_parts']
        for p_word in player_words:
            for ph_word in photo_words:
                if ph_word.startswith(p_word):
                    return (0.99, f"MATCH DE APODO POR COMIENZO DE PALABRA ('{p_word}' en '{ph_word}')")

        # Regla 2: Apellido único (ej. "Lato" vs "Antonio Lato")
        if len(player_parts['all_parts']) == 1 and player_parts['full'] == photo_parts['last_name']:
            return (0.98, f"MATCH APELLIDO ÚNICO '{player_parts['full']}'")

        # Regla 2.5: Si todas las palabras buscadas están en el nombre de la foto (ej. "El Hilali" en "Omar El Hilali")
        player_words_set = set(player_parts['all_parts'])
        photo_words_set = set(photo_parts['all_parts'])
        if player_words_set.issubset(photo_words_set):
            return (0.97, f"MATCH DE SUBCONJUNTO DE PALABRAS")

        # Regla 3: Nombre único / apodo simple
        if len(player_parts['all_parts']) == 1 and player_full in photo_parts['all_parts']:
            return (0.95, f"MATCH DE NOMBRE ÚNICO '{player_full}'")

        # Regla 4: Inicial + Apellido
        if (len(player_parts['first_name']) == 1 and
            player_parts['last_name'] == photo_parts['last_name'] and
            photo_parts['first_name'].startswith(player_parts['first_name'])):
            return (0.90, "INICIAL + APELLIDO EXACTO")
        
        # Reglas restantes (sin cambios)
        if (player_parts['first_name'] == photo_parts['first_name'] and
            len(player_parts['last_name']) == 1 and
            photo_parts['last_name'].startswith(player_parts['last_name'][0])):
            return (0.90, "NOMBRE + INICIAL APELLIDO")

        if player_parts['last_name'] == photo_parts['last_name']:
            first_name_sim = SequenceMatcher(None, player_parts['first_name'], photo_parts['first_name']).ratio()
            if first_name_sim >= 0.8:
                return (0.85 + (first_name_sim * 0.05), f"APELLIDO EXACTO + NOMBRE SIMILAR ({first_name_sim:.2f})")
        
        full_sim = SequenceMatcher(None, player_full, photo_full).ratio()
        if full_sim > 0.8:
            return (full_sim, f"SIMILITUD GENERAL ALTA ({full_sim:.2f})")

        return (0.0, "SIN COINCIDENCIA CLARA")

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
                        distance = self.levenshtein_distance(p_word, ph_word)  # ← Con self.
                        if distance == 1:  # Solo 1 letra de diferencia
                            matches.append(p_word)
                            print(f"   ✅ COINCIDENCIA TOLERANTE: '{p_word}' ≈ '{ph_word}' (distancia: {distance})")
            
            if matches:
                candidates.append({
                    'entry': photo_entry,
                    'matches': matches,
                    'match_count': len(matches)
                })
        
        # PASO 3: Resolver conflictos
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

    def debug_player_photos(self, equipo, player_names_to_check):
        """
        🔥 FUNCIÓN DE DIAGNÓSTICO: Muestra un informe detallado de por qué
        las fotos de ciertos jugadores no se están encontrando.
        """
        photos_data = self.load_player_photos()
        
        print(f"\n\n=======================================================")
        print(f"🔍 INICIANDO DIAGNÓSTICO DE FOTOS PARA: {equipo.upper()}")
        print(f"=======================================================")
        
        # 1. DIAGNÓSTICO DEL PARQUET: ¿Qué datos de nombres tenemos?
        print("\n--- 1. Análisis de los Datos de Origen (Parquet) ---")
        if self.df is not None:
            columnas = [col for col in ['Nombre', 'Apellido', 'Alias', 'Equipo'] if col in self.df.columns]
            print(f"✅ Columnas de nombres disponibles en el Parquet: {columnas}")
            
            # Mostrar una muestra de datos de un jugador del equipo
            sample_player_df = self.df[self.df['Equipo'] == equipo]
            if not sample_player_df.empty:
                sample_player = sample_player_df.head(1).iloc[0].to_dict()
                print("\n📋 Muestra de datos de un jugador de este equipo:")
                for col in columnas:
                    valor = sample_player.get(col, 'N/A')
                    print(f"   - {col}: '{valor}'")
            else:
                print(f"⚠️ No se encontraron datos para '{equipo}' en el Parquet.")
        else:
            print("❌ El DataFrame principal (self.df) no está cargado.")
        
        # 2. DIAGNÓSTICO DEL JSON: ¿Qué fotos tenemos para este equipo?
        print("\n\n--- 2. Análisis de la Base de Datos de Fotos (JSON) ---")
        team_players_in_json = []
        for photo_entry in photos_data:
            photo_team = photo_entry.get('team_name')
            if photo_team and self.are_teams_equivalent(equipo, photo_team):
                team_players_in_json.append(photo_entry.get('player_name', 'N/A'))
        
        if team_players_in_json:
            print(f"✅ Encontrados {len(team_players_in_json)} jugadores en el JSON para '{equipo}':")
            # Muestra los primeros 10 para no saturar la consola
            for i, name in enumerate(sorted(team_players_in_json)[:10], 1):
                print(f"   {i:2d}. '{name}'")
            if len(team_players_in_json) > 10:
                print(f"   ... y {len(team_players_in_json) - 10} más.")
        else:
            print(f"❌ ¡ALERTA! No se encontró NINGÚN jugador para '{equipo}' en el archivo JSON. Revisa los nombres de equipo.")

        # 3. TRAZA DE BÚSQUEDA: ¿Cómo se busca a cada jugador?
        print("\n\n--- 3. Traza de Búsqueda Detallada ---")
        for player_name in player_names_to_check:
            print(f"\n➤ Buscando a: '{player_name}'")
            print("-" * 30)
            
            # Simular la creación de un diccionario de jugador como lo haría el script
            # Esta simulación es clave para replicar el error
            player_dict_simulado = {'Alias': player_name, 'Nombre': '', 'Apellido': ''}
            # Si el nombre es compuesto, intentamos dividirlo
            name_parts = player_name.split()
            if len(name_parts) > 1:
                player_dict_simulado['Nombre'] = name_parts[0]
                player_dict_simulado['Apellido'] = " ".join(name_parts[1:])
            
            # Paso A: ¿Cómo se construye el nombre completo?
            nombre_buscado = self.construir_nombre_completo(player_dict_simulado)
            print(f"   A. Nombre construido para la búsqueda: '{nombre_buscado}'")

            # Paso B: ¿Cómo se normaliza ese nombre?
            player_parts_norm = self.extract_names_parts(nombre_buscado)
            print(f"   B. Nombre normalizado: '{player_parts_norm['full']}'")

            # Paso C: Ejecutar la búsqueda real
            print("   C. Ejecutando match_player_name...")
            match = self.match_player_name(nombre_buscado, photos_data, equipo)
            
            # Paso D: Mostrar el resultado y las pistas
            if match:
                print(f"   D. ✅ ¡ÉXITO! Coincidencia encontrada: '{match.get('player_name')}'")
            else:
                print(f"   D. ❌ FALLO. No se encontró una coincidencia clara.")
                print(f"      Pistas:")
                # Calcular similitudes con todos los jugadores del equipo en el JSON
                best_json_match = None
                highest_similarity = 0
                for json_name in team_players_in_json:
                    json_name_norm = self.extract_names_parts(json_name)['full']
                    similarity = SequenceMatcher(None, player_parts_norm['full'], json_name_norm).ratio()
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_json_match = json_name
                
                if best_json_match:
                    print(f"         - La coincidencia más cercana en el JSON fue '{best_json_match}'")
                    print(f"         - Similitud del nombre completo: {highest_similarity:.2f}")
                    if highest_similarity < 0.8:
                         print("           (Esta similitud es probablemente demasiado baja para ser considerada un match)")
                else:
                    print("         - No se encontró ninguna coincidencia remotamente similar en el JSON.")

        print(f"\n=======================================================")
        print(f"🔍 DIAGNÓSTICO FINALIZADO")
        print(f"=======================================================")

    def check_collision(self, x1, y1, width1, height1, x2, y2, width2, height2, margin=2):
        """Verifica si dos rectángulos se solapan con un margen de separación"""
        return not (x1 + width1/2 + margin < x2 - width2/2 or 
                    x1 - width1/2 - margin > x2 + width2/2 or 
                    y1 + height1/2 + margin < y2 - height2/2 or 
                    y1 - height1/2 - margin > y2 + height2/2)

    def get_fixed_areas(self):
        """Define las áreas ocupadas por título y escudo (inamovibles)"""
        return [
            {'x': 60, 'y': 75, 'width': 40, 'height': 4, 'name': 'titulo_principal'},
            {'x': 60, 'y': 72, 'width': 35, 'height': 3, 'name': 'titulo_secundario'},
            {'x': 20, 'y': 70, 'width': 8, 'height': 8, 'name': 'escudo'},  # Ajustar según dónde pongas el escudo
            {'x': 60, 'y': 2, 'width': 50, 'height': 4, 'name': 'tabla_resumen'}  # Tabla resumen
        ]

    def reposition_tables(self, posible_11, table_width=16, table_height=20):
        """Reposiciona las tablas para evitar solapamientos"""
        fixed_areas = self.get_fixed_areas()
        positioned_tables = []
        new_positions = {}
        
        for posicion, player_data in posible_11.items():
            if posicion in self.coordenadas_posiciones:
                original_x, original_y = self.coordenadas_posiciones[posicion]
                
                # Buscar la mejor posición cerca de la original
                best_x, best_y = self.find_best_position(
                    original_x, original_y, table_width, table_height,
                    fixed_areas + positioned_tables
                )
                
                new_positions[posicion] = (best_x, best_y)
                positioned_tables.append({
                    'x': best_x, 'y': best_y, 
                    'width': table_width, 'height': table_height,
                    'name': posicion
                })
        
        return new_positions
    
    def find_best_position(self, original_x, original_y, width, height, occupied_areas):
        """Encuentra la mejor posición cerca del punto original"""
        max_distance = 15  # Máxima distancia de búsqueda
        step = 1  # Paso de búsqueda
        
        for distance in range(0, max_distance, step):
            # Buscar en círculos concéntricos alrededor del punto original
            for angle in range(0, 360, 30):  # Cada 30 grados
                x = original_x + distance * np.cos(np.radians(angle))
                y = original_y + distance * np.sin(np.radians(angle))
                
                # Verificar límites del campo
                if not (5 <= x <= 115 and 5 <= y <= 75):
                    continue
                
                # Verificar colisiones
                collision = False
                for area in occupied_areas:
                    if self.check_collision(x, y, width, height, 
                                        area['x'], area['y'], area['width'], area['height']):
                        collision = True
                        break
                
                if not collision:
                    return x, y
        
        # Si no encuentra posición, devolver la original
        return original_x, original_y

    def load_data(self):
        """Carga los datos del archivo parquet"""
        try:
            self.df = pd.read_parquet(self.data_path)
            print(f"✅ Datos cargados exitosamente: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
        except Exception as e:
            print(f"❌ Error al cargar los datos: {e}")
            
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
                
            # Buscar equipos similares
            similar_teams = [team]
            for other_team in unique_teams:
                if other_team != team and other_team not in processed_teams:
                    if self.similarity(team, other_team) > 0.85:
                        similar_teams.append(other_team)
            
            # Elegir el nombre más largo como representativo
            canonical_name = max(similar_teams, key=len)
            
            # Mapear todos los nombres similares al canónico
            for similar_team in similar_teams:
                team_mapping[similar_team] = canonical_name
                processed_teams.add(similar_team)
        
        # Aplicar el mapeo
        self.df['Equipo'] = self.df['Equipo'].map(team_mapping)
        
        # Normalizar jornadas
        def normalize_jornada(jornada):
            if isinstance(jornada, str) and jornada.startswith('J'):
                try:
                    return int(jornada[1:])
                except ValueError:
                    return jornada
            elif isinstance(jornada, str) and jornada.startswith('j'):
                try:
                    return int(jornada[1:])
                except ValueError:
                    return jornada
            return jornada
        
        self.df['Jornada'] = self.df['Jornada'].apply(normalize_jornada)
        print(f"✅ Limpieza completada. Equipos únicos: {len(self.df['Equipo'].unique())}")

    def load_match_events(self):
        """Carga los datos del archivo abp_events.parquet Y ASEGURA QUE 'Week' SEA STRING."""
        try:
            events_path = "extraccion_opta/datos_opta_parquet/match_events.parquet"
            # Cambia self.opta_df por self.events_df para mayor claridad
            self.events_df = pd.read_parquet(events_path)
            
            if 'Week' in self.events_df.columns:
                self.events_df['Week'] = self.events_df['Week'].astype(str)
                print("✅ Columna 'Week' en events_df convertida a string.")

            print(f"✅ Eventos de partido cargados: {self.events_df.shape[0]} filas")
            return True
        except Exception as e:
            print(f"⚠️ Error al cargar eventos de partido: {e}")
            self.events_df = None
            return False

    def convert_jornada_to_week(self, jornada):
        """Convierte jornada MediaCoach (j1, j2) a Week Opta (1, 2)"""
        if not jornada:
            return None
        
        try:
            if isinstance(jornada, str):
                if jornada.lower().startswith('j'):
                    return int(jornada[1:])
                else:
                    return int(jornada)
            else:
                return int(jornada)
        except (ValueError, TypeError):
            return None

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
    
    def generate_team_variations(self, team_name):
        """Genera variaciones del nombre como en load_team_logo"""
        variations = [team_name]
        
        # Quitar prefijos/sufijos comunes
        prefixes_suffixes = ['fc', 'cf', 'cd', 'ud', 'rcd', 'rc', 'ca', 'real', 'deportivo']
        
        for prefix in prefixes_suffixes:
            if team_name.startswith(prefix + ' '):
                clean_name = team_name[len(prefix + ' '):]
                variations.append(clean_name)
            if team_name.endswith(' ' + prefix):
                clean_name = team_name[:-len(' ' + prefix)]
                variations.append(clean_name)
        
        # Palabras clave específicas que mencionaste
        key_words = {
            'celta': ['celta', 'vigo', 'rc celta', 'celta de vigo'],
            'girona': ['girona'],
            'atletico': ['atletico', 'madrid', 'at', 'atletico madrid'],
            'madrid': ['madrid', 'real madrid', 'r madrid'],
            'barcelona': ['barcelona', 'barca', 'fcb'],
            'villarreal': ['villarreal'],
            'valencia': ['valencia'],
            'elche': ['elche'],
            'levante': ['levante'],
            'athletic': ['athletic', 'ath', 'bilbao'],
            'sociedad': ['sociedad', 'real sociedad'],
            'valladolid': ['valladolid', 'real valladolid'],
            'las palmas': ['las palmas', 'palmas', 'ud las palmas'],
            'getafe': ['getafe'],
            'espanyol': ['espanyol', 'espanol', 'rcd espanyol'],
            'osasuna': ['osasuna', 'ca osasuna'],
            'sevilla': ['sevilla'],
            'betis': ['betis', 'real betis'],
            'rayo': ['rayo', 'vallecano', 'rayo vallecano'],
            'mallorca': ['mallorca', 'rcd mallorca']
        }
        
        for key, words in key_words.items():
            if key in team_name:
                variations.extend(words)
        
        return list(set(variations))  # Eliminar duplicados

    def obtener_posiciones_equipo_opta(self, equipo):
        """NUEVO: Obtiene TODAS las posiciones de un equipo desde Opta"""
        if self.opta_df is None:
            print(f"   ❌ No hay datos Opta cargados")
            return {}
        
        print(f"🔍 Obteniendo todas las posiciones de {equipo} desde Opta...")
        
        # Buscar registros del equipo en Opta
        registros_equipo = []
        
        for _, opta_row in self.opta_df.iterrows():
            opta_team = str(opta_row.get('Team Name', ''))
            if self.are_teams_equivalent(equipo, opta_team):
                registros_equipo.append(opta_row)
        
        print(f"   📊 Encontrados {len(registros_equipo)} registros en Opta para {equipo}")
        
        if not registros_equipo:
            return {}
        
        # Agrupar por Shirt Number y obtener posición más frecuente
        posiciones_por_dorsal = {}
        
        for registro in registros_equipo:
            shirt_number = registro.get('Shirt Number')
            position = registro.get('Position')
            position_side = registro.get('Position Side')
            
            # Filtrar registros válidos
            if (pd.notna(shirt_number) and 
                pd.notna(position) and 
                str(position).strip() not in ["Substitute", "Not Used", "", "nan"]):
                
                try:
                    dorsal = str(int(float(shirt_number)))
                    
                    if dorsal not in posiciones_por_dorsal:
                        posiciones_por_dorsal[dorsal] = []
                    
                    # Mapear la posición
                    posicion_mapeada = self.mapear_posicion_opta_a_coordenadas(position, position_side)
                    if posicion_mapeada:
                        posiciones_por_dorsal[dorsal].append(posicion_mapeada)
                        
                except (ValueError, TypeError):
                    continue
        
        # Obtener posición más frecuente por dorsal
        posiciones_finales = {}
        for dorsal, posiciones in posiciones_por_dorsal.items():
            if posiciones:
                from collections import Counter
                posicion_mas_frecuente = Counter(posiciones).most_common(1)[0][0]
                frecuencia = Counter(posiciones)[posicion_mas_frecuente]
                posiciones_finales[dorsal] = {
                    'posicion': posicion_mas_frecuente,
                    'frecuencia': frecuencia,
                    'total_registros': len(posiciones)
                }
                print(f"   ✅ Dorsal {dorsal}: {posicion_mas_frecuente} ({frecuencia}/{len(posiciones)} veces)")
        
        print(f"   🎯 Total dorsales con posición válida: {len(posiciones_finales)}")
        return posiciones_finales

    def buscar_posicion_jugador_por_dorsal(self, dorsal, equipo, posiciones_opta):
        """NUEVO: Busca la posición de un jugador por dorsal en los datos Opta procesados"""
        if not posiciones_opta:
            return None
        
        try:
            dorsal_str = str(int(float(dorsal)))
            if dorsal_str in posiciones_opta:
                return posiciones_opta[dorsal_str]['posicion']
        except (ValueError, TypeError):
            pass
        
        return None

    def mapear_posicion_opta_a_coordenadas(self, position, position_side, partido_sin_dc_puro=False):
        """Mapea Position + Position Side de Opta a nuestras coordenadas fijas"""
        
        if pd.isna(position):
            return None
        
        position = str(position).strip()
        position_side = str(position_side).strip() if pd.notna(position_side) else ""
        
        # Mapeo según las reglas especificadas
        if position == "Goalkeeper":
            return "PORTERO"
        
        elif position == "Defender":
            if "Centre/Right" in position_side:
                return "CENTRAL_DERECHO"
            elif "Left/Centre" in position_side:
                return "CENTRAL_IZQUIERDO"
            elif "Left" in position_side:
                return "LATERAL_IZQUIERDO"
            elif "Right" in position_side:
                return "LATERAL_DERECHO"
            elif "Centre" in position_side:  # <- ESTA LÍNEA
                return "CENTRAL_CENTRO"      # <- CAMBIAR A ESTO
            else:
                return "CENTRAL_DERECHO"  # Por defecto

        elif position == "Defensive Midfielder":
            if "Centre" in position_side:
                return "MC_POSICIONAL"
            elif "Left" in position_side:
                return "MC_POSICIONAL"
            else:
                return "MC_POSICIONAL" 
        
        elif position == "Attacking Midfielder":
            if "Right" in position_side:
                return "BANDA_DERECHA"
            elif "Left" in position_side:
                return "BANDA_IZQUIERDA"
            elif "Centre" in position_side:
                return "MC_BOX_TO_BOX"
            else:
                return "MC_BOX_TO_BOX"

        elif position == "Midfielder":  # ✅ UN SOLO BLOQUE PARA MIDFIELDER
            # 1. Comprobar primero los casos específicos combinados
            if "Centre/Right" in position_side:
                return "MC_ORGANIZADOR"
            elif "Left/Centre" in position_side:
                return "MC_ORGANIZADOR"
            
            # 2. Ahora, comprobar los casos más generales (bandas)
            elif "Right" in position_side:
                return "BANDA_DERECHA"
            elif "Left" in position_side:
                return "BANDA_IZQUIERDA"

            # 3. Finalmente, el caso puramente central como fallback
            elif "Centre" in position_side:
                return "MC_ORGANIZADOR"
            
            # 4. Si no hay información de lado, se asume central
            else:
                return "MC_ORGANIZADOR"
        
        elif position == "Striker":
            # 🔥 LÓGICA CONDICIONAL APLICADA AQUÍ 🔥
            # Si el flag del partido indica que no hay un DC puro, cualquier Striker es DELANTERO_CENTRO
            if partido_sin_dc_puro:
                return "DELANTERO_CENTRO"
            
            # De lo contrario (si hay un DC puro), se usa la lógica detallada para diferenciar roles
            else:
                if "Centre/Right" in position_side:
                    return "BANDA_DERECHA"
                elif "Left/Centre" in position_side:
                    return "BANDA_IZQUIERDA"
                elif "Centre" in position_side:
                    return "DELANTERO_CENTRO"
                else:
                    # Un Striker sin lado especificado o con lado no claro también es DC
                    return "DELANTERO_CENTRO" 
        
        else:
            return None

    def buscar_posiciones_jugador_opta(self, jugador_alias, equipo, dorsal):
        """Busca todas las posiciones de un jugador en Opta para determinar la más frecuente"""
        
        if self.opta_df is None:
            print(f"   ❌ {jugador_alias}: No hay datos Opta cargados")
            return None
        
        print(f"   🔍 Buscando {jugador_alias} - Dorsal: {dorsal} - Equipo: {equipo}")
        
        posiciones_encontradas = []
        matches_encontrados = 0
        
        # Buscar por dorsal y equipo (más confiable)
        if pd.notna(dorsal) and dorsal != 'N/A':
            try:
                dorsal_buscar = str(int(float(dorsal)))
            except (ValueError, TypeError):
                print(f"   ❌ Dorsal inválido: {dorsal}")
                return None
            
            for _, opta_row in self.opta_df.iterrows():
                # Verificar dorsal
                dorsal_match = False
                if 'Shirt Number' in opta_row and pd.notna(opta_row['Shirt Number']):
                    try:
                        opta_dorsal = str(int(float(opta_row['Shirt Number'])))
                        if opta_dorsal == dorsal_buscar:
                            dorsal_match = True
                    except (ValueError, TypeError):
                        continue
                
                # Verificar equipo
                team_match = False
                if 'Team Name' in opta_row:
                    opta_team = str(opta_row.get('Team Name', ''))
                    team_match = self.are_teams_equivalent(equipo, opta_team)
                    if dorsal_match:  # Solo debug si dorsal coincide
                        print(f"      🏟️ Equipo Opta: '{opta_team}' vs '{equipo}' → Match: {team_match}")
                
                # Si coinciden dorsal y equipo
                if dorsal_match and team_match:
                    matches_encontrados += 1
                    position = opta_row.get('Position')
                    position_side = opta_row.get('Position Side')
                    
                    print(f"      ✅ Match #{matches_encontrados}: Position='{position}', Side='{position_side}'")
                    
                    if pd.notna(position) and str(position).strip() not in ["Substitute", "", "Not Used"]:
                        posicion_mapeada = self.mapear_posicion_opta_a_coordenadas(position, position_side)
                        if posicion_mapeada:
                            posiciones_encontradas.append(posicion_mapeada)
                            print(f"         → Mapeada a: {posicion_mapeada}")
                    else:
                        print(f"         → Posición descartada: '{position}'")
        else:
            print(f"   ❌ {jugador_alias}: Dorsal faltante o inválido")
        
        print(f"   📊 Total matches: {matches_encontrados}, Posiciones válidas: {len(posiciones_encontradas)}")
        return posiciones_encontradas

    def get_posicion_mas_frecuente_opta(self, jugador_alias, equipo, dorsal):
        """Obtiene la posición más frecuente de un jugador según Opta"""
        
        posiciones = self.buscar_posiciones_jugador_opta(jugador_alias, equipo, dorsal)
        
        if not posiciones:
            return None
        
        # Contar frecuencias y devolver la más común
        from collections import Counter
        contador_posiciones = Counter(posiciones)
        posicion_mas_frecuente = contador_posiciones.most_common(1)[0][0]
        
        print(f"   🎯 {jugador_alias}: {posicion_mas_frecuente} (Opta - {contador_posiciones[posicion_mas_frecuente]} veces)")
        return posicion_mas_frecuente

    def get_available_teams(self):
        """Retorna la lista de equipos disponibles"""
        if self.df is None:
            return []
        return sorted(self.df['Equipo'].unique())
    
    def get_available_jornadas(self, equipo=None):
        """Retorna las jornadas disponibles"""
        if self.df is None:
            return []
        
        if equipo:
            filtered_df = self.df[self.df['Equipo'] == equipo]
            return sorted(filtered_df['Jornada'].unique())
        else:
            return sorted(self.df['Jornada'].unique())

    def get_last_5_jornadas(self, equipo, jornada_referencia):
        """Obtiene las últimas 5 jornadas incluyendo la de referencia"""
        jornadas_disponibles = self.get_available_jornadas(equipo)
        
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

    def fill_missing_demarcaciones(self, df):
        """Rellena demarcaciones vacías con la más frecuente para cada jugador"""
        print("🔄 Rellenando demarcaciones vacías...")
        
        # Crear copia para trabajar
        df_work = df.copy()
        
        # Identificar registros con demarcación vacía
        mask_empty = df_work['Demarcacion'].isna() | (df_work['Demarcacion'] == '') | (df_work['Demarcacion'].str.strip() == '')
        empty_count = mask_empty.sum()
        
        if empty_count > 0:
            print(f"📝 Encontrados {empty_count} registros con demarcación vacía")
            
            # Para cada jugador con demarcación vacía, buscar su demarcación más frecuente
            for idx in df_work[mask_empty].index:
                jugador_id = df_work.loc[idx, 'Id Jugador']
                jugador_alias = df_work.loc[idx, 'Alias']
                
                # Buscar todas las demarcaciones de este jugador (no vacías)
                jugador_demarcaciones = self.df[
                    (self.df['Id Jugador'] == jugador_id) & 
                    (self.df['Demarcacion'].notna()) & 
                    (self.df['Demarcacion'] != '') &
                    (self.df['Demarcacion'].str.strip() != '')
                ]['Demarcacion']
                
                if len(jugador_demarcaciones) > 0:
                    # Usar la demarcación más frecuente
                    demarcacion_mas_frecuente = jugador_demarcaciones.value_counts().index[0]
                    df_work.loc[idx, 'Demarcacion'] = demarcacion_mas_frecuente
                    print(f"   ✅ {jugador_alias}: {demarcacion_mas_frecuente} (histórico)")
                else:
                    # Si no hay datos históricos, asignar "Sin Posición"
                    df_work.loc[idx, 'Demarcacion'] = 'Sin Posición'
                    print(f"   ⚠️  {jugador_alias}: Sin posición histórica -> MC Posicional")
        
        return df_work

    def get_posiciones_mas_utilizadas_equipo(self, equipo, jornadas_analizar):
        """Obtiene las 11 posiciones más utilizadas por el equipo en Opta"""
        print(f"\nDEBUG get_posiciones_mas_utilizadas_equipo:")
        print(f"   equipo: {equipo}")
        print(f"   jornadas_analizar: {jornadas_analizar}")
        
        if self.opta_df is None:
            print(f"   ERROR: opta_df is None")
            return {}
        
        posiciones_utilizadas = {}
        registros_totales = 0
        registros_equipo = 0
        registros_validos = 0
        
        for jornada in jornadas_analizar:
            week = self.convert_jornada_to_week(jornada)
            print(f"   Procesando jornada {jornada} -> week {week}")
            if week is None:
                continue
                
            # FIX CRÍTICO: Week se almacena como string en el DataFrame
            match_df = self.opta_df[self.opta_df['Week'] == str(week)]
            print(f"      Registros para week {week}: {len(match_df)}")
            
            for _, row in match_df.iterrows():
                registros_totales += 1
                opta_team = row.get('Team Name', '')
                
                # DEBUG: Mostrar nombres de equipos en Opta
                if registros_totales <= 10:  # Solo mostrar los primeros 10
                    print(f"      🔍 DEBUG Opta: Team='{opta_team}'")
                
                if self.are_teams_equivalent(equipo, opta_team):
                    registros_equipo += 1
                    if registros_equipo <= 3:  # Solo mostrar los primeros 3
                        print(f"      MATCH equipo: '{opta_team}' == '{equipo}'")
                    
                    position = row.get('Position')
                    position_side = row.get('Position Side', '')
                    
                    # Filtrar posiciones válidas
                    if (pd.notna(position) and 
                        str(position).strip() not in ["Substitute", "Not Used", "", "nan"]):
                        
                        registros_validos += 1
                        posicion_key = f"{position}|{position_side}"
                        
                        if posicion_key not in posiciones_utilizadas:
                            posiciones_utilizadas[posicion_key] = 0
                        posiciones_utilizadas[posicion_key] += 1
                        
                        if registros_validos <= 5:  # Solo mostrar las primeras 5
                            print(f"         Posición registrada: {posicion_key}")
                    else:
                        if registros_equipo <= 3:
                            print(f"         Posición inválida: '{position}'")
                else:
                    if registros_totales <= 3:  # Solo mostrar los primeros 3 para no spam
                        print(f"      NO MATCH equipo: '{opta_team}' != '{equipo}'")
        
        print(f"   Resumen:")
        print(f"      - Registros totales procesados: {registros_totales}")
        print(f"      - Registros del equipo: {registros_equipo}")
        print(f"      - Registros válidos: {registros_validos}")
        print(f"      - Posiciones encontradas: {len(posiciones_utilizadas)}")
        
        # Obtener las 11 más utilizadas
        sorted_positions = sorted(posiciones_utilizadas.items(), 
                                key=lambda x: x[1], reverse=True)[:11]
        
        print(f"   Top posiciones: {dict(sorted_positions)}")
        return dict(sorted_positions)
    
    def contar_apariciones_en_posicion(self, dorsal, posicion_objetivo):
        """
        NUEVO: Cuenta cuántas veces un jugador (por dorsal) ha jugado en una posición específica según Opta.
        Es robusta contra datos faltantes o incorrectos.
        """
        if self.opta_df is None:
            return 0
        
        try:
            # Asegurar que el dorsal a buscar sea un string de un entero, ej: '5'
            dorsal_str = str(int(float(dorsal)))
            contador = 0
            
            # Crear una máscara booleana para encontrar todas las filas del dorsal
            # Esto es mucho más rápido que iterar fila por fila
            mask = self.opta_df['Shirt Number'].astype(str).str.replace(r'\.0$', '', regex=True) == dorsal_str
            
            # Filtrar el DataFrame de Opta una sola vez
            jugador_df = self.opta_df[mask]
            
            # Iterar solo sobre los registros de ese jugador
            for _, row in jugador_df.iterrows():
                position = row.get('Position')
                position_side = row.get('Position Side')
                
                # Mapear la posición de Opta a nuestro sistema
                posicion_mapeada = self.mapear_posicion_opta_a_coordenadas(position, position_side)
                
                if posicion_mapeada == posicion_objetivo:
                    contador += 1
            
            return contador
            
        except (ValueError, TypeError):
            # Si el dorsal no es un número válido, no se puede buscar.
            return 0

    def get_dorsal_temporada_reciente(self, jugador_id, equipo):
        """Obtiene el dorsal del jugador de la temporada más reciente disponible"""
        if self.df is None:
            return None
        
        # Filtrar por jugador y equipo
        jugador_data = self.df[
            (self.df['Id Jugador'] == jugador_id) & 
            (self.df['Equipo'] == equipo) &
            (self.df['Dorsal'].notna())
        ]
        
        if jugador_data.empty:
            return None
        
        # Usar 'Temporada' en lugar de 'Season'
        if 'Temporada' in jugador_data.columns:
            # Convertir temporadas a valores numéricos para comparar
            jugador_data = jugador_data.copy()
            jugador_data['season_numeric'] = jugador_data['Temporada'].apply(self.season_to_numeric)
            
            # Obtener el dorsal de la temporada más reciente
            temporada_reciente = jugador_data.loc[jugador_data['season_numeric'].idxmax()]
            return temporada_reciente['Dorsal']
        else:
            # Si no hay columna Temporada, usar el dorsal más frecuente
            return jugador_data['Dorsal'].mode().iloc[0] if len(jugador_data['Dorsal'].mode()) > 0 else None
    
    def season_to_numeric(self, season_str):
        if pd.isna(season_str):
            return 0
        
        season_str = str(season_str).strip()
        
        try:
            # Formato: "24/25", "2024/2025"
            if '/' in season_str:
                return int(season_str.split('/')[-1])
            
            # Formato: "2024-2025", "24-25"
            elif '-' in season_str:
                return int(season_str.split('-')[-1])
            
            # Formato: año único "2025", "25"
            else:
                year = int(season_str)
                # Si es año de 2 dígitos, asumimos que 24 = 2024, 25 = 2025, etc.
                if year < 100:
                    return year
                else:
                    return year % 100  # 2025 -> 25
        except:
            return 0

    def get_max_season(self, equipo):
        """Obtiene el valor numérico de la temporada más reciente para un equipo"""
        if self.df is None:
            return None
        
        # Determinar qué columna usar
        season_column = None
        if 'Temporada' in self.df.columns:
            season_column = 'Temporada'
        elif 'Season' in self.df.columns:
            season_column = 'Season'
        else:
            print("⚠️ No se encontró columna de temporada")
            return None
        
        equipo_data = self.df[self.df['Equipo'] == equipo]
        if equipo_data.empty:
            return None
        
        # Convertir temporadas a numérico y obtener la máxima
        season_values = equipo_data[season_column].apply(self.season_to_numeric)
        return season_values.max()
    
    def calcular_minutos_totales(self, dorsal, equipo, jornadas):
        """Calcula los minutos totales jugados por un jugador en las jornadas especificadas"""
        try:
            dorsal_str = str(int(float(dorsal)))
            dorsal_mask = self.df['Dorsal'].astype(str).str.replace(r'\.0$', '', regex=True) == dorsal_str
            
            jugador_df = self.df[
                (self.df['Equipo'] == equipo) & 
                (self.df['Jornada'].isin(jornadas)) &
                dorsal_mask
            ]
            
            if 'Minutos jugados' in jugador_df.columns:
                minutos_totales = jugador_df['Minutos jugados'].sum()
                return minutos_totales if pd.notna(minutos_totales) else 0
            else:
                return 0
        except:
            return 0
    
    def get_posible_11(self, equipo, jornada):
        """
        🔥 VERSIÓN CON DESEMPATE POR MINUTOS:
        1. Elige al jugador más frecuente en cada posición
        2. Si hay empate, elige al que más minutos jugó en las últimas 5 jornadas
        3. Asigna puestos con menos candidatos primero (para evitar huecos)
        """
        from collections import Counter

        jornadas_analizar = self.get_last_5_jornadas(equipo, jornada)
        print(f"Analizando las jornadas: {jornadas_analizar}")

        if self.events_df is None:
            print("⚠️ No hay datos de eventos de partido disponibles. Usando fallback.")
            return None, None

        # --- PASO 1 y 2: Obtener formación más usada ---
        partidos_analizados = []
        print("\n--- Analizando formaciones de partidos recientes ---")
        for j in jornadas_analizar:
            setup = self.get_team_setup_from_events_FIXED(equipo, j)
            if setup:
                partidos_analizados.append(setup)

        if not partidos_analizados:
            print("❌ No se encontró ninguna formación válida en los datos de eventos. Usando fallback.")
            return None, None

        contador_formaciones = Counter(p['formation_name'] for p in partidos_analizados)
        if not contador_formaciones:
            print("❌ No se pudo determinar ninguna formación. Usando fallback.")
            return None, None
            
        formacion_ganadora_nombre, _ = contador_formaciones.most_common(1)[0]
        formacion_ganadora_num = [k for k, v in self.formation_mapping.items() if v == formacion_ganadora_nombre][0]
        print(f"\n🏆 Formación más utilizada: {formacion_ganadora_nombre.upper()} ({formacion_ganadora_num})")

        # --- PASO 3: Contar apariciones por puesto ---
        partidos_filtrados = [p for p in partidos_analizados if p['formation_name'] == formacion_ganadora_nombre]
        apariciones_por_puesto = {i: Counter() for i in range(1, 12)}
        for partido in partidos_filtrados:
            for puesto, dorsal in partido['position_jersey_map'].items():
                apariciones_por_puesto[puesto][dorsal] += 1
        
        # --- PASO 4: Ordenar puestos por escasez de candidatos ---
        def calcular_prioridad_puesto(puesto):
            candidatos = apariciones_por_puesto.get(puesto, Counter())
            num_candidatos = len(candidatos)
            total_apariciones = sum(candidatos.values())
            return (num_candidatos, -total_apariciones)
        
        puestos_ordenados = sorted(range(1, 12), key=calcular_prioridad_puesto)
        
        print(f"\n📋 Orden de asignación por escasez de candidatos:")
        for puesto in puestos_ordenados:
            num_candidatos = len(apariciones_por_puesto.get(puesto, {}))
            print(f"   - Puesto {puesto}: {num_candidatos} candidato(s)")
        
        # --- PASO 5: Construir el 11 final con desempate por minutos ---
        posible_11 = {}
        jugadores_ya_elegidos = set()

        print("\n👥 Asignando jugadores al 'Once de Gala' (con desempate por minutos):")

        for puesto in puestos_ordenados:
            contador_dorsales = apariciones_por_puesto.get(puesto)
            
            if not contador_dorsales:
                print(f"   - ⚠️ Puesto {puesto}: No hay candidatos.")
                continue

            # 🔥 NUEVO: Ordenar candidatos por apariciones Y minutos (como desempate)
            candidatos_con_minutos = []
            for dorsal, num_apariciones in contador_dorsales.items():
                minutos_totales = self.calcular_minutos_totales(dorsal, equipo, jornadas_analizar)
                candidatos_con_minutos.append((dorsal, num_apariciones, minutos_totales))
            
            # Ordenar por: 1) Más apariciones, 2) Más minutos (desempate)
            candidatos_ordenados = sorted(
                candidatos_con_minutos, 
                key=lambda x: (-x[1], -x[2])  # Negativo para orden descendente
            )
            
            jugador_asignado = False
            for dorsal_candidato, num_apariciones, minutos_totales in candidatos_ordenados:
                
                if dorsal_candidato not in jugadores_ya_elegidos:
                    # Jugador disponible, lo asignamos
                    jugadores_ya_elegidos.add(dorsal_candidato)
                    
                    # Obtener datos completos del jugador
                    dorsal_str = str(int(float(dorsal_candidato)))
                    dorsal_mask = self.df['Dorsal'].astype(str).str.replace(r'\.0$', '', regex=True) == dorsal_str
                    jugador_df = self.df[(self.df['Equipo'] == equipo) & dorsal_mask]

                    if not jugador_df.empty:
                        info_jugador_completa = jugador_df.sort_values(by='Jornada', ascending=False).iloc[0]
                        datos_completos = info_jugador_completa.to_dict()
                        stats_acumuladas = self.get_jugador_complete_data(dorsal_candidato, equipo, jornadas_analizar)['stats']
                        datos_completos['stats'] = stats_acumuladas
                        
                        posicion_key = f"PUESTO_{puesto}"
                        posible_11[posicion_key] = datos_completos

                        nombre_display = datos_completos.get('Alias', 'N/A')
                        print(f"   - Puesto {puesto}: ✅ {nombre_display} (Dorsal {dorsal_candidato}) - {num_apariciones} vez/veces, {minutos_totales:.0f} min")
                    
                    jugador_asignado = True
                    break
                else:
                    print(f"   - Puesto {puesto}: ⏭️  Saltando dorsal {dorsal_candidato} (ya usado)")

            if not jugador_asignado:
                print(f"   - Puesto {puesto}: ❌ No se pudo asignar (todos los candidatos ya usados)")
        
        # --- PASO 6: FALLBACK - Rellenar puestos vacíos con suplentes ---
        if len(posible_11) < 11:
            print(f"\n🆘 Rellenando {11 - len(posible_11)} puesto(s) vacío(s) con suplentes...")
            
            equipo_df = self.df[
                (self.df['Equipo'] == equipo) & 
                (self.df['Jornada'].isin(jornadas_analizar))
            ].copy()
            
            if 'Minutos jugados' in equipo_df.columns:
                equipo_df = equipo_df.sort_values('Minutos jugados', ascending=False)
            
            for puesto in range(1, 12):
                if f"PUESTO_{puesto}" not in posible_11:
                    for _, jugador in equipo_df.iterrows():
                        dorsal = jugador['Dorsal']
                        if dorsal not in jugadores_ya_elegidos:
                            jugadores_ya_elegidos.add(dorsal)
                            
                            stats_acumuladas = self.get_jugador_complete_data(dorsal, equipo, jornadas_analizar)['stats']
                            datos_completos = jugador.to_dict()
                            datos_completos['stats'] = stats_acumuladas
                            
                            posible_11[f"PUESTO_{puesto}"] = datos_completos
                            print(f"   - Puesto {puesto}: 🔄 {jugador['Alias']} (suplente - Dorsal {dorsal})")
                            break

        return posible_11, formacion_ganadora_num

    def contar_apariciones_en_posicion(self, dorsal, posicion_objetivo):
        """Cuenta cuántas veces un jugador ha jugado en una posición específica"""
        if self.opta_df is None:
            return 0
        
        try:
            dorsal_str = str(int(float(dorsal)))
            contador = 0
            
            # Buscar en todos los partidos de Opta
            for _, row in self.opta_df.iterrows():
                opta_dorsal_raw = row.get('Shirt Number')
                if pd.notna(opta_dorsal_raw):
                    try:
                        opta_dorsal = str(int(float(opta_dorsal_raw)))
                        if dorsal_str == opta_dorsal:
                            position = row.get('Position')
                            position_side = row.get('Position Side')
                            posicion_mapeada = self.mapear_posicion_opta_a_coordenadas(position, position_side)
                            
                            if posicion_mapeada == posicion_objetivo:
                                contador += 1
                    except (ValueError, TypeError):
                        continue
            
            return contador
            
        except (ValueError, TypeError):
            return 0
    
    def get_posible_11_fallback(self, equipo, jornada):
        """Lógica estricta que respeta las posiciones naturales de los jugadores"""
        
        jornadas_analizar = self.get_last_5_jornadas(equipo, jornada)
        print(f"📄 Analizando jornadas (Fallback ESTRICTO): {jornadas_analizar}")
        
        filtered_df = self.df[
            (self.df['Equipo'] == equipo) & 
            (self.df['Jornada'].isin(jornadas_analizar))
        ].copy()

        max_season = self.get_max_season(equipo)
        if max_season is not None:
            season_column = 'Temporada' if 'Temporada' in filtered_df.columns else 'Season'
            if season_column in filtered_df.columns:
                # Filtrar solo por la temporada más reciente
                filtered_df['season_numeric'] = filtered_df[season_column].apply(self.season_to_numeric)
                filtered_df = filtered_df[filtered_df['season_numeric'] == max_season]
                print(f"🔍 Filtrando por temporada más reciente: {max_season}")
        
        if filtered_df.empty:
            print(f"❌ No hay datos para {equipo} en las jornadas {jornadas_analizar}")
            return None
        
        filtered_df = self.fill_missing_demarcaciones(filtered_df)
        
        # Calcular minutos POR JUGADOR Y POR POSICIÓN
        jugador_posicion_minutos = {}
        
        for _, row in filtered_df.iterrows():
            jugador_id = row['Id Jugador']
            
            # ✅ PRIORIZAR ALIAS, FALLBACK A NOMBRE
            alias = row.get('Alias')
            if pd.isna(alias) or not str(alias).strip() or str(alias).strip().lower() == 'nan':
                alias = row.get('Nombre', 'N/A')
            
            dorsal = row.get('Dorsal', 'N/A')
            demarcacion = row.get('Demarcacion', 'Sin Posición')
            minutos = row.get('Minutos jugados', 0)
            
            # ✅ FILTRAR SOLO SI AMBOS ESTÁN VACÍOS
            if pd.isna(alias) or not str(alias).strip() or str(alias).strip().lower() == 'nan':
                continue
            
            # Obtener posición Opta o usar demarcación MediaCoach
            pos_opta = self.buscar_posicion_opta_por_partido(
                dorsal, equipo, row.get('Jornada')
            )
            
            posicion_final = pos_opta if pos_opta else self.demarcacion_to_position.get(demarcacion, 'MC_POSICIONAL')
            
            # Clave única: jugador + posición
            key = f"{jugador_id}_{posicion_final}"
            
            if key not in jugador_posicion_minutos:
                jugador_posicion_minutos[key] = {
                    'jugador_id': jugador_id,  # ✅ GUARDAR ID
                    'Alias': alias,
                    'Dorsal': dorsal,
                    'Posicion_Final': posicion_final,
                    'minutos_total': 0,
                    'apariciones': 0,
                    'stats': {}
                }
            
            jugador_posicion_minutos[key]['minutos_total'] += minutos
            jugador_posicion_minutos[key]['apariciones'] += 1
            
            # Acumular estadísticas
            for metric_full in self.metricas_tabla:
                if metric_full in row and pd.notna(row[metric_full]):
                    if metric_full not in jugador_posicion_minutos[key]['stats']:
                        jugador_posicion_minutos[key]['stats'][metric_full] = []
                    jugador_posicion_minutos[key]['stats'][metric_full].append(row[metric_full])
        
        # Calcular promedios
        for key in jugador_posicion_minutos:
            for metric_full in self.metricas_tabla:
                if metric_full in jugador_posicion_minutos[key]['stats']:
                    values = jugador_posicion_minutos[key]['stats'][metric_full]
                    if 'Velocidad Máxima' in metric_full or '/ min' in metric_full:
                        jugador_posicion_minutos[key]['stats'][metric_full] = np.mean(values)
                    else:
                        jugador_posicion_minutos[key]['stats'][metric_full] = np.sum(values)
                else:
                    jugador_posicion_minutos[key]['stats'][metric_full] = 0
        
        # Agrupar por posición
        jugadores_por_posicion = {}
        for key, data in jugador_posicion_minutos.items():
            posicion = data['Posicion_Final']
            if posicion not in jugadores_por_posicion:
                jugadores_por_posicion[posicion] = []
            jugadores_por_posicion[posicion].append(data)
        
        # Ordenar por minutos en cada posición
        for pos in jugadores_por_posicion:
            jugadores_por_posicion[pos].sort(key=lambda x: (x['apariciones'], x['minutos_total']), reverse=True)
        
        # Selección ESTRICTA del once
        posible_11 = {}
        jugadores_ya_elegidos = set()
        
        formacion_prioritaria = [
            'PORTERO', 'LATERAL_DERECHO', 'CENTRAL_DERECHO', 'CENTRAL_IZQUIERDO', 'LATERAL_IZQUIERDO',
            'MC_POSICIONAL', 'MC_ORGANIZADOR', 'MC_BOX_TO_BOX',
            'BANDA_DERECHA', 'BANDA_IZQUIERDA', 'DELANTERO_CENTRO'
        ]
        
        print(f"🔍 DEBUG: Jugadores disponibles por posición:")
        for pos, jugadores in jugadores_por_posicion.items():
            print(f"   {pos}: {len(jugadores)} jugadores")
            for i, jug in enumerate(jugadores[:3]):  # Mostrar los 3 primeros
                print(f"      {i+1}. {jug['Alias']} - {jug['minutos_total']} min")
        
        # FASE 1: Selección estricta (solo jugadores en su posición natural)
        for puesto in formacion_prioritaria:
            if puesto in jugadores_por_posicion:
                for candidato in jugadores_por_posicion[puesto]:
                    if candidato['Alias'] not in jugadores_ya_elegidos:
                        posible_11[puesto] = candidato
                        jugadores_ya_elegidos.add(candidato['Alias'])
                        print(f"✅ {puesto}: {candidato['Alias']} ({candidato['minutos_total']} min)")
                        break
            else:
                print(f"⚠️ No hay jugadores para {puesto}")
        
        # FASE 2: Solo si faltan jugadores críticos, usar alternativas MUY limitadas
        if len(posible_11) < 8:  # Si faltan más de 3 jugadores, algo está mal
            print(f"⚠️ Solo {len(posible_11)} jugadores asignados. Revisando alternativas limitadas...")
            
            alternativas_criticas = {
                'MC_BOX_TO_BOX': ['MC_ORGANIZADOR', 'MEDIAPUNTA'],
                'MC_ORGANIZADOR': ['MC_BOX_TO_BOX'],
                'BANDA_DERECHA': ['MEDIAPUNTA', 'LATERAL_DERECHO'],  # ✅ AÑADIR LATERAL_DERECHO
                'BANDA_IZQUIERDA': ['MEDIAPUNTA', 'LATERAL_IZQUIERDO']  # ✅ AÑADIR LATERAL_IZQUIERDO
            }
            
            for puesto in formacion_prioritaria:
                if puesto not in posible_11 and puesto in alternativas_criticas:
                    for alt_pos in alternativas_criticas[puesto]:
                        if alt_pos in jugadores_por_posicion:
                            for candidato in jugadores_por_posicion[alt_pos]:
                                if candidato['Alias'] not in jugadores_ya_elegidos:
                                    # Cambiar la posición del candidato
                                    candidato_copia = candidato.copy()
                                    candidato_copia['Posicion_Final'] = puesto
                                    candidato_copia['Posicion_Original'] = alt_pos
                                    posible_11[puesto] = candidato_copia
                                    jugadores_ya_elegidos.add(candidato['Alias'])
                                    print(f"🔄 {puesto}: {candidato['Alias']} (reubicado desde {alt_pos})")
                                    break
                        if puesto in posible_11:
                            break

        # FASE 3: Completar hasta 11 jugadores si faltan
        jugadores_disponibles = []
        for pos, jugadores_list in jugadores_por_posicion.items():
            for candidato in jugadores_list:
                if candidato['Alias'] not in jugadores_ya_elegidos:
                    jugadores_disponibles.append((candidato, pos))

        # Ordenar por minutos y llenar posiciones faltantes
        jugadores_disponibles.sort(key=lambda x: x[0]['minutos_total'], reverse=True)

        posiciones_faltantes = [pos for pos in formacion_prioritaria if pos not in posible_11]
        for pos_faltante in posiciones_faltantes:
            if jugadores_disponibles:
                mejor_candidato, pos_original = jugadores_disponibles.pop(0)
                mejor_candidato = mejor_candidato.copy()
                mejor_candidato['Posicion_Final'] = pos_faltante
                mejor_candidato['Posicion_Original'] = pos_original
                posible_11[pos_faltante] = mejor_candidato
                jugadores_ya_elegidos.add(mejor_candidato['Alias'])
                print(f"🔄 {pos_faltante}: {mejor_candidato['Alias']} (completando desde {pos_original})")

        # Si aún faltan, usar cualquier jugador disponible con minutos > 0
        while len(posible_11) < 11 and jugadores_disponibles:
            # Buscar posición libre
            pos_libre = None
            for pos in formacion_prioritaria:
                if pos not in posible_11:
                    pos_libre = pos
                    break
            
            if pos_libre and jugadores_disponibles:
                candidato_extra, _ = jugadores_disponibles.pop(0)
                candidato_extra = candidato_extra.copy()
                candidato_extra['Posicion_Final'] = pos_libre
                posible_11[pos_libre] = candidato_extra
                jugadores_ya_elegidos.add(candidato_extra['Alias'])
                print(f"➕ {pos_libre}: {candidato_extra['Alias']} (relleno)")
        
        # ✅ PASO FINAL: Actualizar dorsales con temporada más reciente
        print("\n🔄 Actualizando dorsales con temporada más reciente...")
        
        for posicion, player_data in posible_11.items():
            if 'jugador_id' in player_data and player_data['jugador_id']:
                dorsal_actualizado = self.get_dorsal_temporada_reciente(
                    player_data['jugador_id'], equipo
                )
                if dorsal_actualizado and str(dorsal_actualizado) != str(player_data['Dorsal']):
                    print(f"   📝 {player_data['Alias']}: Dorsal {player_data['Dorsal']} → {dorsal_actualizado}")
                    player_data['Dorsal'] = dorsal_actualizado
        
        print(f"\n📊 Resultado final: {len(posible_11)} jugadores asignados")
        return posible_11

    def get_minutos_jugador_jornada(self, dorsal, equipo, jornada):
        """Obtiene los minutos de un jugador en una jornada específica"""
        try:
            dorsal_str = str(int(float(dorsal)))
            
            # Crear máscara de dorsales más robusta
            dorsal_mask = self.df['Dorsal'].fillna(0).astype(str).str.replace('.0', '', regex=False) == dorsal_str
            
            filtered_df = self.df[
                (self.df['Equipo'] == equipo) & 
                (self.df['Jornada'] == jornada) &
                dorsal_mask
            ]
            
            if not filtered_df.empty:
                return filtered_df['Minutos jugados'].iloc[0]
            return 0
        except Exception as e:
            print(f"Error buscando minutos para dorsal {dorsal}: {e}")
            return 0

    def get_jugador_complete_data(self, dorsal, equipo, jornadas_analizar):
        """
        Obtiene los datos completos de un jugador (Alias, stats acumuladas)
        basado en su dorsal en las jornadas analizadas.
        """
        try:
            dorsal_str = str(int(float(dorsal)))
            dorsal_mask = self.df['Dorsal'].astype(str).str.replace(r'\.0$', '', regex=True) == dorsal_str
            
            filtered_df = self.df[
                (self.df['Equipo'] == equipo) &
                (self.df['Jornada'].isin(jornadas_analizar)) &
                dorsal_mask
            ]
            
            # Si no hay datos en esas jornadas, buscar en todo el historial solo para el nombre
            if filtered_df.empty:
                all_player_data = self.df[(self.df['Equipo'] == equipo) & dorsal_mask]
                if not all_player_data.empty:
                    alias = all_player_data.iloc[0].get('Alias', f'Jugador {dorsal}')
                else:
                    alias = f'Jugador {dorsal}'
                return {'Alias': alias, 'stats': {metric: 0 for metric in self.metricas_tabla}}

            # Datos del jugador
            info_jugador = filtered_df.iloc[0]
            datos = {
                'Alias': info_jugador.get('Alias', f'Jugador {dorsal}'),
                'stats': {metric: 0 for metric in self.metricas_tabla} # Inicializar con ceros
            }

            # Calcular métricas físicas
            for metric in self.metricas_tabla:
                if metric in filtered_df.columns and filtered_df[metric].notna().any():
                    values = filtered_df[metric].dropna()
                    if 'Velocidad Máxima' in metric or '/ min' in metric:
                        datos['stats'][metric] = values.mean()
                    else:
                        datos['stats'][metric] = values.sum()
            return datos

        except Exception as e:
            print(f"   -> Error obteniendo datos para dorsal {dorsal}: {e}")
            return {'Alias': f'Jugador {dorsal}', 'stats': {metric: 0 for metric in self.metricas_tabla}}
        

    def load_team_logo(self, equipo):
        """Carga el escudo del equipo con lógica mejorada y normalización robusta."""
        import re
        import glob
        
        # PASO 1: Generar variaciones del nombre del equipo (esto ya estaba bien)
        possible_names = []
        equipo_limpio_base = re.sub(r'[^a-zA-Z0-9\s]', '', equipo.lower().strip())
        possible_names.extend([
            equipo, equipo_limpio_base, equipo.replace(' ', ''), equipo.replace(' ', '_'),
            equipo.lower(), equipo.lower().replace(' ', ''), equipo.lower().replace(' ', '_'),
        ])
        prefixes_suffixes = [
            'fc ', 'cf ', 'cd ', 'ud ', 'rcd ', 'rc ', 'ca ', 'real ', 'deportivo ',
            ' fc', ' cf', ' cd', ' ud', ' rcd', ' rc', ' ca'
        ]
        for prefix in prefixes_suffixes:
            if equipo_limpio_base.startswith(prefix.strip()):
                possible_names.append(equipo_limpio_base[len(prefix):].strip())
            if equipo_limpio_base.endswith(prefix.strip()):
                possible_names.append(equipo_limpio_base[:-len(prefix)].strip())
        if 'atletico' in equipo_limpio_base:
            possible_names.extend(['atletico', 'atleticomadrid', 'atletico_madrid'])
        if 'madrid' in equipo_limpio_base and 'atletico' not in equipo_limpio_base:
            possible_names.extend(['madrid', 'realmadrid', 'real_madrid'])
        
        # Normalizar todas las variaciones y eliminar duplicados
        seen = set()
        unique_names = []
        for name in possible_names:
            if name:
                normalized_name = self.normalize_text(name)
                if normalized_name and normalized_name not in seen:
                    seen.add(normalized_name)
                    unique_names.append(normalized_name)

        # PASO 2: Obtener y normalizar todos los escudos disponibles en la carpeta
        try:
            escudos_disponibles_paths = glob.glob("assets/escudos/*.png")
            if not escudos_disponibles_paths:
                print("⚠️ No se encontraron archivos de escudo en assets/escudos/")
                return None
        except Exception as e:
            print(f"⚠️ Error al leer la carpeta de escudos: {e}")
            return None

        # Crear un diccionario: {nombre_normalizado: ruta_completa}
        # Ejemplo: {"atletico": "assets/escudos/Atlético.png"}
        normalized_escudos_map = {
            self.normalize_text(os.path.basename(p).rsplit('.', 1)[0]): p
            for p in escudos_disponibles_paths
        }

        # PASO 3: Búsqueda por coincidencia exacta (ahora funcionará)
        for name in unique_names:
            if name in normalized_escudos_map:
                logo_path = normalized_escudos_map[name]
                try:
                    print(f"✅ Escudo encontrado (coincidencia normalizada exacta): {logo_path}")
                    return plt.imread(logo_path)
                except Exception as e:
                    print(f"⚠️ Error al cargar {logo_path}: {e}")

        # PASO 4: Si todo falla, usar búsqueda por similitud con lógica de exclusión mejorada
        mejor_match_path = None
        mejor_similitud = 0
        equipo_limpio = self.normalize_text(equipo)

        for nombre_normalizado, escudo_path in normalized_escudos_map.items():
            
            # 🔥 Lógica de exclusión más robusta para los equipos de Madrid 🔥
            buscando_atletico = 'atletico' in equipo_limpio
            buscando_real = 'real' in equipo_limpio and not buscando_atletico

            if buscando_atletico and 'real madrid' in nombre_normalizado:
                continue  # Si busco Atleti, ignoro el del Real Madrid
            if buscando_real and 'atletico' in nombre_normalizado:
                continue  # Si busco Real Madrid, ignoro el del Atleti

            # Calcular la mejor similitud para este archivo
            similitud_actual = max(self.similarity(variacion, nombre_normalizado) for variacion in unique_names)
            
            if similitud_actual > mejor_similitud:
                mejor_similitud = similitud_actual
                mejor_match_path = escudo_path
        
        if mejor_match_path and mejor_similitud > 0.6:
            try:
                print(f"✅ Escudo encontrado por similitud ({mejor_similitud:.2f}): {mejor_match_path}")
                return plt.imread(mejor_match_path)
            except Exception as e:
                print(f"⚠️ Error al cargar {mejor_match_path}: {e}")

        print(f"❌ No se encontró escudo para: {equipo}")
        return None

    def get_team_colors(self, equipo):
        """Obtiene los colores del equipo o devuelve colores por defecto"""
        # Buscar coincidencia exacta primero
        if equipo in self.team_colors:
            return self.team_colors[equipo]
        
        # Buscar coincidencia parcial
        for team_name in self.team_colors.keys():
            if team_name.lower() in equipo.lower() or equipo.lower() in team_name.lower():
                return self.team_colors[team_name]
        
        # Si no encuentra nada, devolver colores por defecto
        print(f"⚠️  Equipo '{equipo}' no reconocido, usando colores por defecto")
        return self.default_team_colors

    def create_campo_sin_espacios(self, figsize=(20, 14)):
        """Crea el campo que ocupe TODA la página sin espacios"""
        print("🎯 Creando campo SIN espacios...")
        
        # Crear pitch sin padding
        pitch = Pitch(
            pitch_color='grass', 
            line_color='white', 
            stripe=True, 
            linewidth=3,
            pad_left=0, pad_right=0, pad_bottom=0, pad_top=0
        )
        
        # Crear figura sin layouts automáticos
        fig, ax = pitch.draw(
            figsize=figsize,
            tight_layout=False,
            constrained_layout=False
        )
        
        # ✅ CONFIGURACIÓN AGRESIVA PARA ELIMINAR TODOS LOS ESPACIOS
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        ax.set_position([0, 0, 1, 1])
        ax.margins(0, 0)
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 80)
        ax.autoscale(enable=False)
        ax.set_aspect('equal')
        fig.patch.set_visible(False)
        ax.set_frame_on(False)
        
        return fig, ax
    
    def split_long_name(self, name, max_chars=12):
        """Divide el nombre en dos líneas si es muy largo"""
        if len(name) <= max_chars:
            return name, ""
        
        # Intentar dividir por palabras
        words = name.split()
        if len(words) > 1:
            mid = len(words) // 2
            line1 = " ".join(words[:mid])
            line2 = " ".join(words[mid:])
            return line1, line2
        else:
            # Si es una sola palabra muy larga, dividir por la mitad
            mid = len(name) // 2
            return name[:mid], name[mid:]

    def get_best_contrast_colors(self, team_color_primary):
        """Determina los mejores colores de fondo y texto para máximo contraste"""
        
        # Convertir color hex a RGB si es necesario
        if isinstance(team_color_primary, str) and team_color_primary.startswith('#'):
            hex_color = team_color_primary.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        else:
            # Si no es hex, usar valores por defecto
            r, g, b = 44, 62, 80  # Color por defecto
        
        # Calcular luminosidad (fórmula estándar)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        
        # Si el color primario es claro (luminosidad > 0.5), usar fondo oscuro
        if luminance > 0.5:
            return {
                'background': '#2c3e50',  # Fondo oscuro
                'dorsal_color': team_color_primary,  # Mantener color del equipo
                'name_color': 'white'  # Texto blanco
            }
        else:
            return {
                'background': 'white',  # Fondo claro
                'dorsal_color': team_color_primary,  # Mantener color del equipo  
                'name_color': 'black'  # Texto negro
            }

    def create_player_table(self, player_data, x, y, ax, team_colors, position_name, team_logo=None):
        """Crea una tabla individual para un jugador con sus estadísticas"""
        
        # Dimensiones de la tabla compacta
        table_width = 16
        header_height = 2.5
        name_height = 3.0
        metric_row_height = 1.8
        table_height = header_height + name_height + (len(self.metricas_mostrar) * metric_row_height)
        
        # 🎨 FONDO MODERNO
        main_rect = plt.Rectangle((x - table_width/2, y - table_height/2), 
                                table_width, table_height,
                                facecolor='#2c3e50', alpha=0.95, 
                                edgecolor='white', linewidth=2)
        ax.add_patch(main_rect)
        
        # Efecto de borde superior
        top_rect = plt.Rectangle((x - table_width/2, y + table_height/2 - 0.4), 
                                table_width, 0.4,
                                facecolor=team_colors['primary'], alpha=0.9,
                                edgecolor='none')
        ax.add_patch(top_rect)
        
        # 📍 FILA 1: POSICIÓN
        if 'Posicion_Final' in player_data:
            position_key = player_data['Posicion_Final']
            # Convertir clave a nombre legible
            nombres_posiciones = {
                'PORTERO': 'Portero',
                'LATERAL_DERECHO': 'Lateral Derecho',
                'CENTRAL_DERECHO': 'Central Derecho', 
                'CENTRAL_IZQUIERDO': 'Central Izquierdo',
                'LATERAL_IZQUIERDO': 'Lateral Izquierdo',
                'MC_POSICIONAL': 'MC Posicional',
                'MC_BOX_TO_BOX': 'MC Box to Box',
                'MC_ORGANIZADOR': 'MC Organizador',
                'BANDA_DERECHA': 'Banda Derecha',
                'BANDA_IZQUIERDA': 'Banda Izquierda',
                'MEDIAPUNTA': 'Mediapunta',
                'DELANTERO_CENTRO': 'Delantero Centro',
                'SEGUNDO_DELANTERO': 'Segundo Delantero'
            }
            clean_position_name = nombres_posiciones.get(position_key, position_key.replace('_', ' ').title())
        else:
            # Fallback a demarcación MediaCoach
            clean_position_name = player_data['Demarcacion']
            if ' - ' in clean_position_name:
                clean_position_name = clean_position_name.split(' - ', 1)[1]
        
        header_rect = plt.Rectangle((x - table_width/2, y + table_height/2 - header_height), 
                                table_width, header_height,
                                facecolor=team_colors['primary'], alpha=0.8,
                                edgecolor='white', linewidth=1)
        ax.add_patch(header_rect)

        ax.text(x, y + table_height/2 - header_height/2, clean_position_name, 
                fontsize=10, weight='bold', color=team_colors['text'],
                ha='center', va='center')
        
        # 📍 FILA 2: NOMBRE + DORSAL
        names_y = y + table_height/2 - header_height - name_height/2

        # Obtener colores optimizados para contraste
        contrast_colors = self.get_best_contrast_colors(team_colors['primary'])

        names_rect = plt.Rectangle((x - table_width/2, names_y - name_height/2), 
                                table_width, name_height,
                                facecolor=contrast_colors['background'], alpha=1.0,
                                edgecolor='white', linewidth=0.5)
        ax.add_patch(names_rect)

        # Dorsal a la IZQUIERDA
        dorsal_raw = player_data['Dorsal']
        if pd.notna(dorsal_raw) and dorsal_raw != 'N/A':
            try:
                dorsal_display = str(int(float(dorsal_raw)))
            except (ValueError, TypeError):
                dorsal_display = str(dorsal_raw)
        else:
            dorsal_display = 'N/A'

        # Dorsal a la IZQUIERDA
        ax.text(x - table_width/3, names_y, dorsal_display, 
                fontsize=18, weight='bold', color=contrast_colors['dorsal_color'],
                ha='center', va='center')

        # Nombre a la DERECHA (dividido en dos líneas si es necesario)
        line1, line2 = self.split_long_name(player_data['Alias'], max_chars=12)

        if line2:  # Si hay segunda línea
            ax.text(x + table_width/6, names_y + 0.5, line1, 
                    fontsize=10, weight='bold', color=contrast_colors['name_color'],
                    ha='center', va='center')
            ax.text(x + table_width/6, names_y - 0.5, line2, 
                    fontsize=10, weight='bold', color=contrast_colors['name_color'],
                    ha='center', va='center')
        else:  # Si cabe en una línea
            ax.text(x + table_width/6, names_y, line1, 
                    fontsize=12, weight='bold', color=contrast_colors['name_color'],
                    ha='center', va='center')
        
        # 📍 FILAS 3+: MÉTRICAS Y VALORES
        for i, metric_short in enumerate(self.metricas_mostrar):
            metric_y = names_y - name_height/2 - (i + 1) * metric_row_height + metric_row_height/2
            
            # Fondo alternado para las filas de métricas
            if i % 2 == 0:
                row_rect = plt.Rectangle((x - table_width/2, metric_y - metric_row_height/2), 
                                    table_width, metric_row_height,
                                    facecolor='#3c566e', alpha=0.3, 
                                    edgecolor='none')
                ax.add_patch(row_rect)
            
            # Valor de la métrica
            value = player_data['stats'].get(metric_short, 0)
            
            if 'V.Max' in metric_short or '/min' in metric_short:
                formatted_value = f"{value:.1f}"
            else:
                formatted_value = f"{value:.0f}"
            
            # Métrica y valor en la misma fila
            ax.text(x - table_width/4, metric_y, metric_short, 
                    fontsize=12, weight='bold', color='white',
                    ha='center', va='center')
            
            ax.text(x + table_width/4, metric_y, formatted_value, 
                    fontsize=12, weight='bold', color='#FFD700',
                    ha='center', va='center')

    def create_team_summary_table(self, posible_11, ax, x_pos, y_pos, team_name, team_colors, team_logo=None):
        """Crea una tabla de resumen del equipo con promedios del posible 11"""
        
        # Calcular estadísticas promedio del posible 11
        summary_stats = {}
        
        for metric_short in self.metricas_mostrar:
            values = []
            for posicion, player in posible_11.items():
                if metric_short in player['stats']:
                    values.append(player['stats'][metric_short])
            
            if values:
                if metric_short in ['V.Max']:  # Velocidad máxima: máximo
                    summary_stats[metric_short] = max(values)
                else:  # Resto: promedio
                    summary_stats[metric_short] = np.mean(values)
            else:
                summary_stats[metric_short] = 0
        
        # Dimensiones de la tabla (2 FILAS)
        num_metrics = len(summary_stats)
        metric_col_width = 7
        table_width = num_metrics * metric_col_width
        row_height = 1.2
        table_height = row_height * 2
        
        # 🎨 FONDO MODERNO
        main_rect = plt.Rectangle((x_pos - table_width/2, y_pos - table_height/2), 
                                table_width, table_height,
                                facecolor='#2c3e50', alpha=0.95, 
                                edgecolor='white', linewidth=2)
        ax.add_patch(main_rect)
        
        # Efecto de borde superior
        top_rect = plt.Rectangle((x_pos - table_width/2, y_pos + table_height/2 - 0.3), 
                                table_width, 0.3,
                                facecolor=team_colors['primary'], alpha=0.9,
                                edgecolor='none')
        ax.add_patch(top_rect)
        
        # 📍 FILA 1: NOMBRES DE MÉTRICAS
        metrics_y = y_pos + row_height/2
        
        for i, (metric_short, value) in enumerate(summary_stats.items()):
            metric_x = x_pos - table_width/2 + (i * metric_col_width) + metric_col_width/2
            
            # Fondo para cada métrica en fila 1
            metric_rect = plt.Rectangle((metric_x - metric_col_width/2, metrics_y - row_height/2), 
                                    metric_col_width, row_height,
                                    facecolor=team_colors['primary'], alpha=0.6, 
                                    edgecolor='white', linewidth=0.5)
            ax.add_patch(metric_rect)
            
            # Nombre de la métrica
            ax.text(metric_x, metrics_y, metric_short, 
                    fontsize=8, weight='bold', color='white',
                    ha='center', va='center')
        
        # 📍 FILA 2: VALORES DE MÉTRICAS
        values_y = y_pos - row_height/2
        
        for i, (metric_short, value) in enumerate(summary_stats.items()):
            metric_x = x_pos - table_width/2 + (i * metric_col_width) + metric_col_width/2
            
            # Fondo alternado para valores en fila 2
            if i % 2 == 0:
                value_rect = plt.Rectangle((metric_x - metric_col_width/2, values_y - row_height/2), 
                                        metric_col_width, row_height,
                                        facecolor='#3c566e', alpha=0.3, 
                                        edgecolor='none')
                ax.add_patch(value_rect)
            
            # Valor de la métrica
            if 'V.Max' in metric_short or '/min' in metric_short:
                formatted_value = f"{value:.1f}"
            else:
                formatted_value = f"{value:.0f}"
            
            ax.text(metric_x, values_y, formatted_value, 
                    fontsize=10, weight='bold', color='#FFD700',
                    ha='center', va='center')

    def create_visualization(self, equipo, jornada, figsize=(11.69, 8.27)):
        """
        Crea la visualización completa del posible 11 inicial, ahora usando la nueva
        lógica de "formación más frecuente".
        """
        # 1. Obtener el 11 y la formación más usada con la nueva lógica
        posible_11, formacion_num = self.get_posible_11(equipo, jornada)
        
        if not posible_11:
            print("❌ No se pudo generar el posible 11 inicial. No hay datos suficientes.")
            # Opcional: podrías generar una imagen vacía con un mensaje de error aquí
            return None
        
        # 2. Configurar el campo, títulos y logos (esta parte no cambia)
        fig, ax = self.create_campo_sin_espacios(figsize)
        jornadas_analizadas = self.get_last_5_jornadas(equipo, jornada)
        
        ax.text(60, 79, f'POSIBLE 11 INICIAL - {equipo.upper()}',
                fontsize=16, weight='bold', color='white', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.8", facecolor='#1e3d59', alpha=1.0, edgecolor='white', linewidth=2))
        ax.text(60, 75, f'Basado en la formación y jugadores más frecuentes | Jornadas: {min(jornadas_analizadas)}-{max(jornadas_analizadas)}',
                fontsize=10, weight='bold', color='white', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#34495e', alpha=0.98, edgecolor='white', linewidth=1))
        team_logo = self.load_team_logo(equipo)
        if team_logo is not None:
            imagebox = OffsetImage(team_logo, zoom=0.45)
            ab = AnnotationBbox(imagebox, (110, 70), frameon=False)
            ax.add_artist(ab)
        
        team_colors = self.get_team_colors(equipo)

        # 3. Bucle de Dibujado
        print("\n🎨 Dibujando la alineación en el campo...")
        for posicion_unica, player_data in posible_11.items():
            # Extraer el número del puesto (ej: de "PUESTO_9" obtenemos 9)
            puesto_num = int(posicion_unica.split('_')[1])
            
            # Usar el número de la formación y el número del puesto para obtener las coordenadas
            if formacion_num in self.formation_coordinates and puesto_num in self.formation_coordinates[formacion_num]:
                x, y = self.formation_coordinates[formacion_num][puesto_num]

                # Preparar el diccionario del jugador para la función de dibujado
                jugador_compatible = {**player_data, **player_data.get('stats', {})}
                jugador_compatible['Equipo'] = equipo

                self.crear_jugador_circular(
                    jugador_compatible, x, y, ax, team_colors, posicion_unica,
                    scale=1.1,
                    team_logo=team_logo
                )
            else:
                print(f"   ⚠️ No se encontraron coordenadas para el puesto {puesto_num} en la formación {formacion_num}.")
                
        self.crear_leyenda_general(fig)
        return fig
    
    def crear_jugador_circular(self, jugador, x, y, ax, team_colors, posicion_name, scale=0.8, team_logo=None):
        """
        🔥 VERSIÓN FINAL: Dibuja un jugador como un círculo con sus métricas alrededor.
        """
        abbreviaciones = {
            'PORTERO': 'PO', 'LATERAL_DERECHO': 'LD', 'LATERAL_IZQUIERDO': 'LI',
            'CENTRAL_DERECHO': 'CD', 'CENTRAL_IZQUIERDO': 'CI','CENTRAL_CENTRO': 'CC', 'MC_POSICIONAL': 'MCD',
            'MC_ORGANIZADOR': 'MC', 'MC_BOX_TO_BOX': 'MP', 'BANDA_DERECHA': 'MD',
            'BANDA_IZQUIERDA': 'MI', 'DELANTERO_CENTRO': 'DC', 'SEGUNDO_DELANTERO': 'SD'
        }
        if not jugador:
            return

        photos_data = self.load_player_photos()
        
        radio_circulo = 6 * scale
        radio_metricas = 7 * scale
        
        # 🔥 AQUÍ NO HAY CAMBIOS, PERO ES IMPORTANTE ENTENDER QUÉ PASA AHORA:
        # 'jugador' es ahora el diccionario completo con 'Nombre' y 'Apellido'.
        # Por lo tanto, 'get_player_photo_without_dorsal' llamará a 'construir_nombre_completo'
        # y esta podrá crear "Iñaki Williams" o "Nico Williams", permitiendo una búsqueda precisa.
        equipo = jugador.get('Equipo', None)
        player_photo = self.get_player_photo_without_dorsal(jugador, photos_data, equipo)


        circle_bg = plt.Circle((x, y), radio_circulo, facecolor='white', alpha=0.7, edgecolor='gray', linewidth=2, zorder=5)
        ax.add_patch(circle_bg)

        if player_photo is not None:
            img_extent = [x - radio_circulo * 0.8, x + radio_circulo * 0.8, y - radio_circulo * 0.8, y + radio_circulo * 0.8]
            ax.imshow(player_photo, extent=img_extent, aspect='auto', clip_on=True, zorder=6)
        elif team_logo is not None:
            logo_extent = [x - radio_circulo * 0.7, x + radio_circulo * 0.7, y - radio_circulo * 0.7, y + radio_circulo * 0.7]
            ax.imshow(team_logo, extent=logo_extent, aspect='auto', clip_on=True, zorder=6, alpha=0.8)
        
        # --- DIBUJO DE MÉTRICAS ---
        num_metricas = len(self.metricas_tabla)
        angulo_step = 2 * np.pi / num_metricas
        for i, metrica in enumerate(self.metricas_tabla):
            angulo = i * angulo_step - np.pi / 2
            rect_x = x + radio_metricas * np.cos(angulo)
            rect_y = y + radio_metricas * np.sin(angulo)
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
        
        # --- DORSAL ---
        dorsal = jugador.get('Dorsal', 'N/A')
        dorsal_str = str(int(float(dorsal))) if pd.notna(dorsal) and dorsal != 'N/A' else ''
        if dorsal_str:
            dorsal_x = x - radio_circulo * 0.60
            dorsal_y = y + radio_circulo * 0.18
            ax.text(dorsal_x, dorsal_y, dorsal_str, 
                    fontsize=int(9*scale),
                    weight='900', 
                    color='black',
                    ha='center', va='center', zorder=10,
                    path_effects=[patheffects.withStroke(linewidth=2.5, foreground='white')])

        # --- ETIQUETA DE POSICIÓN ---
        posicion_base_nombre = re.sub(r'_\d+$', '', posicion_name)
        demarcacion_abreviada = abbreviaciones.get(posicion_base_nombre, '')
        if demarcacion_abreviada:
            dem_x = x + radio_circulo * 0.60
            dem_y = y + radio_circulo * 0.20
            ax.text(dem_x, dem_y, demarcacion_abreviada, 
                    fontsize=int(6.5*scale),
                    weight='bold', 
                    color='#2c3e50',
                    ha='center', va='center', zorder=10,
                    path_effects=[patheffects.withStroke(linewidth=2.5, foreground='white')])

        # --- NOMBRE COMPLETO ---
        player_name = jugador.get('Alias', 'N/A')
        nombre_parts = str(player_name).split() if player_name else ['N/A']
        nombre_linea1 = nombre_parts[0] if nombre_parts else 'N/A'
        nombre_linea2 = ' '.join(nombre_parts[1:]) if len(nombre_parts) > 1 else ''
        
        ax.text(x, y - radio_circulo * 0.3, nombre_linea1,
                fontsize=int(5.5*scale), 
                weight='bold', 
                color='black', 
                ha='center', va='center', zorder=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.85, edgecolor='none'))

        if nombre_linea2:
            ax.text(x, y - radio_circulo * 0.65, nombre_linea2,
                    fontsize=int(5.5*scale), 
                    weight='bold', 
                    color='black', 
                    ha='center', va='center', zorder=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.85, edgecolor='none'))

    def crear_leyenda_general(self, fig):
        """
        🔥 VERSIÓN FINAL: Leyenda dinámica que se posiciona arriba a la derecha.
        """
        import matplotlib.patches as patches
        y_centro = 0.02
        padding_horizontal = 0.015
        espacio_entre_elementos = 0.01
        alto_leyenda = 0.012

        elementos_info = []
        ancho_total = 0
        for metrica in self.metricas_tabla:
            metrica_corta = metrica.replace('Distancia Total ', 'D').replace(' km / h', '').replace(' / min', '/m').replace('Velocidad Máxima Total', 'Vel. Máx.')
            temp_text = fig.text(0, 0, metrica_corta, fontsize=7, weight='bold', ha='center')
            bbox = temp_text.get_window_extent(fig.canvas.get_renderer())
            ancho_texto_fig = bbox.width / fig.dpi / fig.get_size_inches()[0]
            temp_text.remove()
            ancho_elemento = ancho_texto_fig + 0.01
            elementos_info.append({'texto': metrica_corta, 'color': self.colores_metricas.get(metrica, '#2c3e50'), 'ancho': ancho_elemento})
            ancho_total += ancho_elemento

        ancho_total += espacio_entre_elementos * (len(elementos_info) - 1)
        
        # Alineación a la derecha
        x_inicio = (1 - (ancho_total + padding_horizontal * 2)) / 2

        fondo = patches.FancyBboxPatch(
            (x_inicio, y_centro - alto_leyenda / 2),
            ancho_total + padding_horizontal * 2, alto_leyenda,
            boxstyle="round,pad=0.005,rounding_size=0.01",
            facecolor='#f0f0f0', alpha=0.9, edgecolor='black',
            linewidth=0.5, zorder=30, transform=fig.transFigure
        )
        fig.patches.append(fondo)

        x_offset = x_inicio + padding_horizontal
        for elemento in elementos_info:
            color_fondo = elemento['color']
            texto_str = elemento['texto']
            ancho = elemento['ancho']
            centro_elemento_x = x_offset + ancho / 2
            color_texto = self.get_contrast_color_for_metric(color_fondo)
            fig.text(centro_elemento_x, y_centro, texto_str,
                    ha='center', va='center',
                    fontsize=7, weight='bold',
                    color=color_texto,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color_fondo, edgecolor='white', linewidth=0.5),
                    transform=fig.transFigure, zorder=32
            )
            x_offset += ancho + espacio_entre_elementos
    
    def guardar_sin_espacios(self, fig, filename):
        """Guarda el archivo sin ningún espacio en blanco en formato A4 Horizontal"""
        fig.set_size_inches(11.69, 8.27) # Asegura el tamaño A4 apaisado
        fig.savefig(
            filename,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0,
            facecolor='white',
            edgecolor='none',
            format='pdf' if filename.endswith('.pdf') else 'png',
            transparent=False,
            orientation='landscape' # <-- AÑADE ESTA LÍNEA
        )
        print(f"✅ Archivo guardado SIN espacios: {filename}")

def seleccionar_equipo_jornada():
    """Permite al usuario seleccionar un equipo y jornada"""
    try:
        report_generator = Posible11Inicial()
        equipos = report_generator.get_available_teams()
        
        if len(equipos) == 0:
            print("❌ No se encontraron equipos en los datos.")
            return None, None
        
        print("\n=== SELECCIÓN DE EQUIPO - POSIBLE 11 INICIAL ===")
        for i, equipo in enumerate(equipos, 1):
            print(f"{i:2d}. {equipo}")
        
        while True:
            try:
                seleccion = input(f"\nSelecciona un equipo (1-{len(equipos)}): ").strip()
                indice = int(seleccion) - 1
                
                if 0 <= indice < len(equipos):
                    equipo_seleccionado = equipos[indice]
                    break
                else:
                    print(f"❌ Por favor, ingresa un número entre 1 y {len(equipos)}")
            except ValueError:
                print("❌ Por favor, ingresa un número válido")
        
        # Obtener jornadas disponibles para el equipo
        jornadas_disponibles = report_generator.get_available_jornadas(equipo_seleccionado)
        print(f"\nJornadas disponibles para {equipo_seleccionado}: {jornadas_disponibles}")
        
        # Seleccionar jornada de referencia
        while True:
            try:
                jornada_input = input(f"Selecciona la jornada de referencia (ej: {max(jornadas_disponibles)}): ").strip()
                
                # Intentar convertir a entero
                if jornada_input.startswith('J') or jornada_input.startswith('j'):
                    jornada_seleccionada = int(jornada_input[1:])
                else:
                    jornada_seleccionada = int(jornada_input)
                
                if jornada_seleccionada in jornadas_disponibles:
                    break
                else:
                    print(f"❌ Jornada {jornada_seleccionada} no disponible. Disponibles: {jornadas_disponibles}")
            except ValueError:
                print("❌ Por favor, ingresa una jornada válida")
        
        return equipo_seleccionado, jornada_seleccionada
        
    except Exception as e:
        print(f"❌ Error en la selección: {e}")
        return None, None

def main_posible_11():
    """Función principal para generar el posible 11 inicial"""
    try:
        print("⚽ === GENERADOR POSIBLE 11 INICIAL ===")
        
        # Selección interactiva (sin cambios)
        equipo, jornada = seleccionar_equipo_jornada()
        
        if equipo is None or jornada is None:
            print("❌ No se pudo completar la selección.")
            return
        
        print(f"\n🔄 Generando posible 11 inicial para {equipo}")
        print(f"📅 Jornada de referencia: {jornada}")
        
        # Crear el reporte (ya no necesita el parámetro debug)
        report_generator = Posible11Inicial()
        fig = report_generator.create_visualization(equipo, jornada)
        
        if fig:
            plt.show()
            
            equipo_filename = equipo.replace(' ', '_').replace('/', '_')
            output_path = f"posible_11_inicial_{equipo_filename}_J{jornada}.pdf"
            
            report_generator.guardar_sin_espacios(fig, output_path)
            
        else:
            print("❌ No se pudo generar la visualización")
            
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()

def generar_posible_11_personalizado(equipo, jornada, mostrar=True, guardar=True):
    """Función para generar un posible 11 personalizado"""
    try:
        # Ya no necesita el parámetro debug
        report_generator = Posible11Inicial()
        fig = report_generator.create_visualization(equipo, jornada)
        
        if fig:
            if mostrar:
                plt.show()
            
            if guardar:
                equipo_filename = equipo.replace(' ', '_').replace('/', '_')
                output_path = f"posible_11_inicial_{equipo_filename}_J{jornada}.pdf"
                report_generator.guardar_sin_espacios(fig, output_path)
            
            return fig
        else:
            print("❌ No se pudo generar la visualización")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Inicialización
print("⚽ === INICIALIZANDO GENERADOR POSIBLE 11 INICIAL ===")
try:
    report_generator = Posible11Inicial()
    equipos = report_generator.get_available_teams()
    print(f"\n✅ Sistema POSIBLE 11 INICIAL listo. Equipos disponibles: {len(equipos)}")
    
    if len(equipos) > 0:
        print("📝 Para generar un reporte ejecuta: main_posible_11()")
        print("📝 Para uso directo: generar_posible_11_personalizado('Equipo', jornada)")
        print("\n🔥 CARACTERÍSTICAS:")
        print("   • Selecciona al jugador con más minutos por posición")
        print("   • Analiza las últimas 5 jornadas")
        print("   • Métricas físicas completas por jugador")
        print("   • Formación 4-3-3 visual")
        print("   • Colores personalizados por equipo")
    
except Exception as e:
    print(f"❌ Error al inicializar: {e}")

if __name__ == "__main__":
    main_posible_11()