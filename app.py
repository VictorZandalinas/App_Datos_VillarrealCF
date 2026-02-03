import dash
import re
from dash import dcc, html, Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import os
from pathlib import Path
import base64
import threading
from datetime import datetime
from functools import lru_cache # Para optimización
import actualizar_datos
import subprocess
import shutil
import glob
from flask import send_from_directory, Response, request, abort



# --- En app.py, debajo de las rutas ---
EQUIPOS_REPORTE = [
    "1. Alavés", "2. Athletic Club", "3. Atlético de Madrid", "4. Barcelona", "5. Celta de Vigo",
    "6. Elche", "7. Espanyol", "8. Getafe", "9. Girona", "10. Levante",
    "11. Mallorca", "12. Osasuna", "13. Rayo Vallecano", "14. Real Betis", "15. Real Madrid",
    "16. Real Oviedo", "17. Real Sociedad", "18. Sevilla", "19. Valencia", "20. Villarreal"
]

# Mapeo de nombres de equipos a sus archivos de escudo
ESCUDOS_MAPPING = {
    'Alavés': 'Alaves.png', 'Deportivo Alavés': 'Alaves.png',
    'Athletic Club': 'Athletic.png', 'Athletic': 'Athletic.png',
    'Atlético de Madrid': 'Atlético.png', 'Atlético': 'Atlético.png',
    'Barcelona': 'FC Barcelona.png', 'FC Barcelona': 'FC Barcelona.png',
    'Celta de Vigo': 'Celta Vigo.png', 'RC Celta': 'Celta Vigo.png', 'Celta': 'Celta Vigo.png',
    'Elche': 'Elche.png', 'Elche CF': 'Elche.png',
    'Espanyol': 'Espanyol.png', 'RCD Espanyol': 'Espanyol.png',
    'Getafe': 'Getafe.png', 'Getafe CF': 'Getafe.png',
    'Girona': 'Girona.png', 'Girona FC': 'Girona.png',
    'Levante': 'Levante.png', 'Levante UD': 'Levante.png',
    'Mallorca': 'Mallorca.png', 'RCD Mallorca': 'Mallorca.png',
    'Osasuna': 'Osasuna.png', 'CA Osasuna': 'Osasuna.png',
    'Rayo Vallecano': 'Rayo Vallecano.png',
    'Real Betis': 'Betis.png', 'Betis': 'Betis.png',
    'Real Madrid': 'Real Madrid.png',
    'Real Oviedo': 'Real Oviedo.png',
    'Real Sociedad': 'Real Sociedad.png',
    'Sevilla': 'Sevilla FC.png', 'Sevilla FC': 'Sevilla FC.png',
    'Valencia': 'Valencia.png', 'Valencia CF': 'Valencia.png',
    'Villarreal': 'Villarreal CF.png', 'Villarreal CF': 'Villarreal CF.png',
    'Las Palmas': 'Las Palmas.png', 'UD Las Palmas': 'Las Palmas.png',
    'Leganés': 'Leganes.png', 'CD Leganés': 'Leganes.png',
    'Valladolid': 'Valladolid.png', 'Real Valladolid': 'Valladolid.png',
}

def get_escudo_base64(equipo_nombre):
    """Obtiene el escudo de un equipo en base64 para mostrar en el dropdown"""
    escudo_file = ESCUDOS_MAPPING.get(equipo_nombre)
    if escudo_file:
        escudo_path = ASSETS_PATH / 'escudos' / escudo_file
        if escudo_path.exists():
            try:
                with open(escudo_path, 'rb') as f:
                    return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
            except:
                pass
    return None

# --- CONFIGURACIÓN DE RUTAS ---
BASE_PATH = Path(__file__).parent
ASSETS_PATH = BASE_PATH / 'assets'

FILE_MEDIACOACH = BASE_PATH / 'extraccion_mediacoach/data/estadisticas_equipo.parquet'
FILE_OPTA = BASE_PATH / 'extraccion_opta/datos_opta_parquet/abp_events.parquet'
FILE_SPORTIAN = BASE_PATH / 'extraccion_sportian/corners_tracking.parquet'

progress_data = {'active': False, 'progress': 0, 'status': 'Esperando...', 'messages': []}
report_progress = {'active': False, 'progress': 0, 'status': '', 'final_path': ''}

# Sistema de cola de jobs para informes
import json
import uuid
from datetime import datetime as dt

JOBS_DIR = BASE_PATH / "jobs"
JOBS_PENDING = JOBS_DIR / "pending"
JOBS_PROCESSING = JOBS_DIR / "processing"
JOBS_COMPLETED = JOBS_DIR / "completed"
JOBS_FAILED = JOBS_DIR / "failed"

# Crear directorios de jobs
for d in [JOBS_PENDING, JOBS_PROCESSING, JOBS_COMPLETED, JOBS_FAILED]:
    d.mkdir(parents=True, exist_ok=True)

# Job actual en seguimiento (por sesión simple - en producción usar session/cookie)
current_job_id = None

def crear_job(script, equipo, j_inicio, j_fin):
    """Crea un nuevo job de generación de informe"""
    job_id = str(uuid.uuid4())[:8]
    job = {
        'id': job_id,
        'script': script,
        'equipo': equipo,
        'j_inicio': j_inicio,
        'j_fin': j_fin,
        'status': 'pending',
        'progress': 0,
        'message': 'En cola...',
        'created_at': dt.now().isoformat(),
        'updated_at': dt.now().isoformat()
    }

    job_path = JOBS_PENDING / f"{job_id}.json"
    with open(job_path, 'w') as f:
        json.dump(job, f, indent=2)

    return job_id

def obtener_estado_job(job_id):
    """Obtiene el estado actual de un job buscando en todas las carpetas"""
    if not job_id:
        return None

    for carpeta in [JOBS_PROCESSING, JOBS_COMPLETED, JOBS_FAILED, JOBS_PENDING]:
        job_path = carpeta / f"{job_id}.json"
        if job_path.exists():
            try:
                with open(job_path, 'r') as f:
                    return json.load(f)
            except:
                pass
    return None

# CONFIGURACIÓN DE BLOQUES DE INFORMES
BLOQUES_CONFIG = {
    'ABP': {
        'parquet': 'extraccion_opta/datos_opta_parquet/abp_events.parquet',
        'col_equipo': 'Team Name',  # ⚠️ NO es 'Team', es 'Team Name'
        'col_jornada': 'Week',
        'script': 'abp_informe_todo.py'
    },
    'FISICO': {
        'parquet': 'extraccion_mediacoach/data/rendimiento_fisico.parquet',
        'col_equipo': 'Equipo',
        'col_jornada': 'Jornada',
        'script': 'fisico_completo_pdf.py'
    },
    'TACTICO': {
        'parquet': 'extraccion_opta/datos_opta_parquet/abp_events.parquet',
        'col_equipo': 'Team Name',  # ⚠️ NO es 'Team', es 'Team Name'
        'col_jornada': 'Week',
        'script': 'tactic_informe_todo.py'
    }
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Departamento Datos - Villarreal CF</title>
    <link rel="icon" href="/assets/favicon.ico">
    {%metas%}{%css%}
</head>
<body>{%app_entry%}{%config%}{%scripts%}{%renderer%}</body>
</html>
'''

app.config.suppress_callback_exceptions = True
server = app.server

USERNAME = os.environ.get('APP_USERNAME', 'admin')
PASSWORD = os.environ.get('APP_PASSWORD', 'password123')

# --- FUNCIONES DE NORMALIZACIÓN ---

def get_logo_base64(filename):
    try:
        with open(ASSETS_PATH / filename, 'rb') as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    except: return ""

def normalizar_liga(nombre):
    if not nombre: return "La Liga"
    n = str(nombre).lower().replace('_', ' ').strip() # Convierte "La_Liga" en "la liga"
    variantes_laliga = ['la liga', 'laliga', 'primera division', 'primera división', 'spanish la liga', 'laliga ea sports']
    if any(x in n for x in variantes_laliga):
        return "La Liga"
    return nombre.replace('_', ' ').title()

def normalizar_jornada(j):
    if pd.isna(j): return None
    try:
        s = str(j).lower().replace('j', '').strip()
        if s == '' or s == 'nan': return None
        val = int(float(s))
        return val if val > 0 else None
    except: return None

def temp_mediacoach(t):
    if not t: return ""
    # Convierte "Season_2025_2026" -> "2025-2026"
    return str(t).replace('Season_', '').replace('_', '-')

def temp_opta(t):
    if not t or '/' not in str(t): return str(t)
    p = str(t).split('/')
    return f"20{p[0]}-20{p[1]}"

def temp_sportian(fecha_str):
    try:
        dt = datetime.fromisoformat(str(fecha_str).replace('Z', ''))
        return f"{dt.year}-{dt.year + 1}" if dt.month >= 8 else f"{dt.year - 1}-{dt.year}"
    except: return "Desconocida"

# --- LÓGICA DE DATOS OPTIMIZADA (CON CACHÉ) ---
def obtener_huella_archivos():
    """Calcula una huella única basada en el nombre y fecha de modificación de los archivos."""
    archivos = [FILE_MEDIACOACH, FILE_OPTA, FILE_SPORTIAN]
    huella_parts = []
    for f in archivos:
        if f.exists():
            # Tomamos el nombre y la fecha de última modificación (st_mtime)
            huella_parts.append(f"{f.name}_{f.stat().st_mtime}")
        else:
            huella_parts.append(f"{f.name}_missing")
    return "|".join(huella_parts)

@lru_cache(maxsize=1)
def obtener_resumen_datos_cached(huella):
    """
    Lee los datos de todas las fuentes, normaliza y cuenta partidos únicos.
    Solo se ejecuta si la 'huella' cambia.
    """
    print(f"🔄 [CACHE] Recargando datos desde el disco. Nueva huella detectada.")
    data_list = []
    
    # 1. MediaCoach
    if FILE_MEDIACOACH.exists():
        try:
            df = pd.read_parquet(FILE_MEDIACOACH, columns=['liga', 'temporada', 'jornada', 'partido'])
            df['j_norm'] = df['jornada'].apply(normalizar_jornada)
            df['liga_norm'] = df['liga'].apply(normalizar_liga)
            df['temp_norm'] = df['temporada'].apply(temp_mediacoach)
            df = df.dropna(subset=['j_norm'])
            df['partido'] = df['partido'].fillna('sin_id') 
            
            res = df.groupby(['liga_norm', 'temp_norm', 'j_norm'])['partido'].nunique().reset_index()
            for _, row in res.iterrows():
                data_list.append({
                    'liga': row['liga_norm'],
                    'temporada': row['temp_norm'],
                    'jornada': int(row['j_norm']),
                    'fuente': 'mediacoach',
                    'n_partidos': int(row['partido'])
                })
        except Exception as e: 
            print(f"❌ Error procesando MediaCoach: {e}")

    # 2. Opta
    if FILE_OPTA.exists():
        try:
            df = pd.read_parquet(FILE_OPTA, columns=['Competition Name', 'Season', 'Week', 'Match ID'])
            df['j_norm'] = df['Week'].apply(normalizar_jornada)
            df = df.dropna(subset=['j_norm'])
            res = df.groupby(['Competition Name', 'Season', 'j_norm'])['Match ID'].nunique().reset_index()
            for _, row in res.iterrows():
                data_list.append({
                    'liga': normalizar_liga(row['Competition Name']),
                    'temporada': temp_opta(row['Season']),
                    'jornada': int(row['j_norm']),
                    'fuente': 'opta',
                    'n_partidos': int(row['Match ID'])
                })
        except Exception as e: 
            print(f"❌ Error procesando Opta: {e}")

    # 3. Sportian
    if FILE_SPORTIAN.exists():
        try:
            df = pd.read_parquet(FILE_SPORTIAN, columns=['Competicion', 'Fecha_Partido', 'Jornada', 'ID_Partido'])
            df['j_norm'] = df['Jornada'].apply(normalizar_jornada)
            df = df.dropna(subset=['j_norm'])
            res = df.groupby(['Competicion', 'Fecha_Partido', 'j_norm'])['ID_Partido'].nunique().reset_index()
            for _, row in res.iterrows():
                data_list.append({
                    'liga': normalizar_liga(row['Competicion']),
                    'temporada': temp_sportian(row['Fecha_Partido']),
                    'jornada': int(row['j_norm']),
                    'fuente': 'sportian',
                    'n_partidos': int(row['ID_Partido'])
                })
        except Exception as e: 
            print(f"❌ Error procesando Sportian: {e}")

    print(f"✅ Datos cargados exitosamente: {len(data_list)} registros de jornadas.")
    return pd.DataFrame(data_list)

def obtener_resumen_datos():
    """Función de acceso público que utiliza la huella para el caché."""
    huella = obtener_huella_archivos()
    return obtener_resumen_datos_cached(huella)

# --- LAYOUTS ---

login_layout = html.Div([
    html.Div(className="login-background"), # Capa de la foto que se mueve
    html.Div(className="login-bg-gradient"), # Capa oscura para contraste
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Img(src=get_logo_base64("logodatos-villarrealcf.png"), height="300px", className="mb-4 d-block mx-auto"),
                            html.H2("Villarreal CF", className="text-center mb-4 login-title"),
                        ]),
                        dbc.Input(id="username", placeholder="Usuario", type="text", className="mb-3 modern-input"),
                        dbc.Input(id="password", placeholder="Contraseña", type="password", className="mb-4 modern-input"),
                        dbc.Button("INICIAR SESIÓN", id="login-button", className="w-100 modern-button"),
                        html.Div(id="login-error", className="text-danger mt-3 text-center")
                    ])
                ], className="login-card")
            ], width=12, md=6, lg=4)
        ], justify="center", className="vh-100 align-items-center")
    ], fluid=True),
], className="login-container")

def crear_layout_principal():
    df = obtener_resumen_datos()
    ligas = sorted(df['liga'].unique()) if not df.empty else []
    temporadas = sorted(df['temporada'].unique(), reverse=True) if not df.empty else []
    
    return dbc.Container([
        # --- NAVBAR ---
        dbc.Navbar([
            dbc.Container([
                dbc.NavbarBrand([
                    html.Img(src=get_logo_base64("logodatos-villarrealcf.png"), height="80px", className="me-2"),
                    "Departamento de Datos"
                ], className="ms-2 fw-bold d-flex align-items-center"),
                dbc.Nav([
                    dbc.NavItem(
                        html.A("🔄 Actualizar Datos", href="/actualizar", className="nav-link nav-link-custom", style={'cursor': 'pointer'})
                    ),
                    dbc.NavItem(dbc.NavLink("🚪 Cerrar Sesión", id="logout-btn", href="#", className="nav-link-custom"))
                ], className="ms-auto", navbar=True)
            ])
        ], color="dark", dark=True, className="mb-4 shadow"),
        
        # --- BLOQUE: GENERADOR DE INFORMES ---
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("📊 Generador de Informes", className="text-white mb-0"), className="bg-dark"),
                    
                    dbc.CardBody([
                        # --- LOS 3 BOTONES PRINCIPALES ---
                        dbc.Row([
                            dbc.Col([
                                dbc.Button([
                                    html.Div([
                                        html.Img(src="/assets/abp_icono.png", height="150px", className="mb-2"),
                                        html.Span("Informe ABP", className="fw-bold")
                                    ], className="d-flex flex-column align-items-center justify-content-center")
                                ], id="btn-rep-abp", n_clicks=0, color="danger", outline=True, className="w-100 py-3 shadow-sm")
                            ], width=4),
                            
                            dbc.Col([
                                dbc.Button([
                                    html.Div([
                                        html.Img(src="/assets/fisico_icono.png", height="150px", className="mb-2"),
                                        html.Span("Informe Físico", className="fw-bold")
                                    ], className="d-flex flex-column align-items-center justify-content-center")
                                ], id="btn-rep-fisico", n_clicks=0, color="success", outline=True, className="w-100 py-3 shadow-sm")
                            ], width=4),
                            
                            dbc.Col([
                                dbc.Button([
                                    html.Div([
                                        html.Img(src="/assets/tactica_icono.png", height="150px", className="mb-2"),
                                        html.Span("Informe Táctico", className="fw-bold")
                                    ], className="d-flex flex-column align-items-center justify-content-center")
                                ], id="btn-rep-tactic", n_clicks=0, color="primary", outline=True, className="w-100 py-3 shadow-sm")
                            ], width=4),
                        ], className="g-3 mb-4"),

                        # --- CONTENEDOR DINÁMICO DE FILTROS (vacío inicialmente) ---
                        html.Div(id='report-selectors-container', children=[], style={'display': 'none'}),
                        
                        # --- ALMACÉN DEL BLOQUE SELECCIONADO ---
                        dcc.Store(id='selected-report-block', data=None),
                        # --- INTERVALO PARA ACTUALIZAR PROGRESO ---
                        dcc.Interval(id='report-interval', interval=800, disabled=True),

                        # --- STORE PARA NOMBRE DEL EQUIPO ---
                        dcc.Store(id='report-team-name', data=None),

                        # --- MODAL DE PROGRESO ---
                        dbc.Modal([
                            dbc.ModalHeader(
                                dbc.ModalTitle("Generando Informe", id="report-modal-title"),
                                close_button=False
                            ),
                            dbc.ModalBody(id="report-modal-body", children=[
                                html.Div([
                                    dbc.Spinner(color="primary", size="lg"),
                                    html.P("Preparando...", className="text-muted mt-3 text-center")
                                ], className="text-center py-4")
                            ]),
                        ], id="report-modal", centered=True, backdrop="static", size="lg", is_open=False),

                        dcc.Download(id="download-pdf"),
                    ])
                ], className="shadow mb-5")
            ], width=10, className="mx-auto")
        ]),

        html.Hr(className="my-5"),

        # --- BLOQUE: VISUALIZACIÓN DE DATOS ---
        dbc.Row([
            dbc.Col([
                html.H4("📈 Visualización de Datos Cargados", className="text-center mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Visualizar Liga:", className="fw-bold mb-2"),
                                dcc.Dropdown(id='liga-dropdown', options=[{'label': l, 'value': l} for l in ligas],
                                             value=ligas[0] if ligas else None, clearable=False)
                            ], width=6),
                            dbc.Col([
                                html.Label("Visualizar Temporada:", className="fw-bold mb-2"),
                                dcc.Dropdown(id='temporada-dropdown', options=[{'label': t, 'value': t} for t in temporadas],
                                             value=temporadas[0] if temporadas else None, clearable=False)
                            ], width=6),
                        ])
                    ])
                ], className="shadow-sm mb-4 bg-light")
            ], width=10, className="mx-auto")
        ]),
        
        # --- GRID DE JORNADAS ---
        html.Div(id='jornadas-container'),
        
    ], fluid=True)

def run_report_process(script_name, equipo_nombre, j_inicio, j_fin, destination_folder):
    global report_progress
    report_progress = {'active': True, 'progress': 5, 'status': 'Iniciando generador...', 'final_path': ''}

    import sys
    import shutil
    import subprocess
    import select
    import time

    # Timeout global de 20 minutos para todo el proceso
    TIMEOUT_GLOBAL = 1200  # 20 minutos
    # Timeout de inactividad: si no hay output en 3 minutos, asumir que está colgado
    TIMEOUT_INACTIVIDAD = 180  # 3 minutos

    process = None

    try:
        # Limpiar PDFs antiguos
        ahora = time.time()
        for archivo in glob.glob(os.path.join(destination_folder, "*.pdf")):
            if os.path.getmtime(archivo) < ahora - 86400:  # 24 horas
                os.remove(archivo)
                print(f"🗑️ Limpiado: {archivo}")
    except Exception as e:
        print(f"⚠️ Error limpiando: {e}")

    try:
        cmd = [sys.executable, "-u", script_name, equipo_nombre, str(j_inicio), str(j_fin)]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        tiempo_inicio = time.time()
        ultimo_output = time.time()

        # Lectura con timeout usando select (no bloqueante)
        while True:
            # Verificar timeout global
            if time.time() - tiempo_inicio > TIMEOUT_GLOBAL:
                report_progress['status'] = f"⏰ Timeout: el informe tardó más de {TIMEOUT_GLOBAL//60} minutos"
                print(f"⏰ TIMEOUT GLOBAL alcanzado ({TIMEOUT_GLOBAL}s)")
                break

            # Verificar timeout de inactividad
            if time.time() - ultimo_output > TIMEOUT_INACTIVIDAD:
                report_progress['status'] = f"⏰ Sin respuesta del generador en {TIMEOUT_INACTIVIDAD//60} minutos"
                print(f"⏰ TIMEOUT INACTIVIDAD alcanzado ({TIMEOUT_INACTIVIDAD}s sin output)")
                break

            # Verificar si el proceso terminó
            if process.poll() is not None:
                # Leer cualquier output restante
                remaining = process.stdout.read()
                if remaining:
                    for line in remaining.split('\n'):
                        if line.strip():
                            print(f"DEBUG SCRIPT: {line.strip()}")
                break

            # Usar select para esperar output con timeout (1 segundo)
            ready, _, _ = select.select([process.stdout], [], [], 1.0)

            if ready:
                line = process.stdout.readline()
                if not line:
                    # EOF - proceso terminó
                    break

                ultimo_output = time.time()
                status_msg = line.strip()
                if not status_msg:
                    continue

                print(f"DEBUG SCRIPT: {status_msg}")

                match = re.search(r"\[(\d+)/(\d+)\] Ejecutando: (.*) ---", status_msg)

                if match:
                    pag_actual = int(match.group(1))
                    pag_total = int(match.group(2))
                    nombre_pag = match.group(3)
                    porcentaje = int((pag_actual / pag_total) * 95)
                    report_progress['progress'] = porcentaje
                    report_progress['status'] = f"Generando página {pag_actual} de {pag_total}: {nombre_pag}"

                elif "Uniendo todos los reportes" in status_msg:
                    report_progress['progress'] = 96
                    report_progress['status'] = "Uniendo páginas en el PDF final..."

        # Matar proceso si sigue vivo (por timeout)
        if process.poll() is None:
            print("🛑 Matando proceso por timeout...")
            process.kill()
            process.wait(timeout=5)

        # Buscar el PDF generado y moverlo
        lista_pdfs = glob.glob("*.pdf")
        if lista_pdfs:
            archivo_reciente = max(lista_pdfs, key=os.path.getctime)
            if os.path.exists(destination_folder):
                final_dest = os.path.join(destination_folder, archivo_reciente)
                try:
                    shutil.move(archivo_reciente, final_dest)
                    report_progress['final_path'] = os.path.abspath(final_dest)
                    report_progress['status'] = f"✅ Informe listo"
                    report_progress['progress'] = 100
                except Exception as e:
                    report_progress['status'] = f"⚠️ Error al mover: {e}"
                    report_progress['progress'] = 100
            else:
                report_progress['final_path'] = os.path.abspath(archivo_reciente)
                report_progress['progress'] = 100
        elif "Timeout" not in report_progress.get('status', ''):
            report_progress['status'] = "⚠️ No se generó ningún PDF"
            report_progress['progress'] = 100

    except Exception as e:
        report_progress['status'] = f"❌ Error: {str(e)}"
        report_progress['progress'] = 100
        print(f"❌ Error en run_report_process: {e}")

    finally:
        # Asegurar que el proceso está muerto
        if process and process.poll() is None:
            try:
                process.kill()
            except:
                pass
        report_progress['active'] = False

def run_report_process_with_restore(script_name, equipo_nombre, j_inicio, j_fin, destination_folder, parquet_original, parquet_backup):
    """Ejecuta el script y SIEMPRE restaura el parquet original al final"""
    global report_progress

    import sys
    import subprocess
    import select
    import time

    # Timeout global de 20 minutos para todo el proceso
    TIMEOUT_GLOBAL = 1200  # 20 minutos
    # Timeout de inactividad: si no hay output en 3 minutos, asumir que está colgado
    TIMEOUT_INACTIVIDAD = 180  # 3 minutos

    process = None

    def restaurar_backup():
        """Función auxiliar para restaurar el backup de parquet"""
        try:
            if parquet_backup.exists():
                shutil.copy2(parquet_backup, parquet_original)
                parquet_backup.unlink()
                print(f"✅ Datos restaurados correctamente: {parquet_original.name}")
                obtener_resumen_datos.cache_clear()
                print(f"🧹 Caché limpiado")
            else:
                print("⚠️ No se encontró backup para restaurar")
        except Exception as e:
            print(f"⚠️ Error restaurando backup: {e}")

    try:
        report_progress = {'active': True, 'progress': 5, 'status': 'Iniciando generador...', 'final_path': ''}

        cmd = [sys.executable, "-u", script_name, equipo_nombre, str(j_inicio), str(j_fin)]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        tiempo_inicio = time.time()
        ultimo_output = time.time()
        timeout_reached = False

        # Lectura con timeout usando select (no bloqueante)
        while True:
            # Verificar timeout global
            if time.time() - tiempo_inicio > TIMEOUT_GLOBAL:
                report_progress['status'] = f"⏰ Timeout: el informe tardó más de {TIMEOUT_GLOBAL//60} minutos"
                print(f"⏰ TIMEOUT GLOBAL alcanzado ({TIMEOUT_GLOBAL}s)")
                timeout_reached = True
                break

            # Verificar timeout de inactividad
            if time.time() - ultimo_output > TIMEOUT_INACTIVIDAD:
                report_progress['status'] = f"⏰ Sin respuesta del generador en {TIMEOUT_INACTIVIDAD//60} minutos"
                print(f"⏰ TIMEOUT INACTIVIDAD alcanzado ({TIMEOUT_INACTIVIDAD}s sin output)")
                timeout_reached = True
                break

            # Verificar si el proceso terminó
            if process.poll() is not None:
                remaining = process.stdout.read()
                if remaining:
                    for line in remaining.split('\n'):
                        if line.strip():
                            print(f"DEBUG SCRIPT: {line.strip()}")
                break

            # Usar select para esperar output con timeout (1 segundo)
            ready, _, _ = select.select([process.stdout], [], [], 1.0)

            if ready:
                line = process.stdout.readline()
                if not line:
                    break

                ultimo_output = time.time()
                status_msg = line.strip()
                if not status_msg:
                    continue

                print(f"DEBUG SCRIPT: {status_msg}")

                match = re.search(r"\[(\d+)/(\d+)\] Ejecutando: (.*) ---", status_msg)

                if match:
                    pag_actual = int(match.group(1))
                    pag_total = int(match.group(2))
                    nombre_pag = match.group(3)
                    porcentaje = int((pag_actual / pag_total) * 95)
                    report_progress['progress'] = porcentaje
                    report_progress['status'] = f"Generando página {pag_actual} de {pag_total}: {nombre_pag}"

                elif "Uniendo todos los reportes" in status_msg:
                    report_progress['progress'] = 96
                    report_progress['status'] = "Uniendo páginas en el PDF final..."

        # Matar proceso si sigue vivo (por timeout)
        if process.poll() is None:
            print("🛑 Matando proceso por timeout...")
            process.kill()
            process.wait(timeout=5)

        # RESTAURAR BACKUP después de terminar el script
        print("🔄 Restaurando datos originales...")
        report_progress['status'] = '🔄 Restaurando datos originales...'
        restaurar_backup()

        # Buscar el PDF generado y moverlo
        report_progress['progress'] = 98
        report_progress['status'] = 'Buscando PDF generado...'

        lista_pdfs = glob.glob("*.pdf")
        if lista_pdfs:
            archivo_reciente = max(lista_pdfs, key=os.path.getctime)
            if os.path.exists(destination_folder):
                final_dest = os.path.join(destination_folder, archivo_reciente)
                try:
                    shutil.move(archivo_reciente, final_dest)
                    report_progress['final_path'] = final_dest
                    if timeout_reached:
                        report_progress['status'] = f"⚠️ Informe parcial (timeout) guardado"
                    else:
                        report_progress['status'] = f"✅ Informe guardado con éxito (datos restaurados)"
                except Exception as e:
                    report_progress['status'] = f"⚠️ PDF generado pero no se pudo mover: {archivo_reciente}"
            else:
                report_progress['status'] = f"✅ Informe generado en carpeta del proyecto: {archivo_reciente}"
        elif not timeout_reached:
            report_progress['status'] = "⚠️ No se encontró PDF generado"

        report_progress['progress'] = 100

    except Exception as e:
        report_progress['status'] = f'❌ Error: {str(e)}'
        report_progress['progress'] = 100
        print(f"❌ Error en run_report_process_with_restore: {e}")

        # RESTAURAR BACKUP incluso si hay error
        print("🔄 Restaurando datos tras error...")
        restaurar_backup()

    finally:
        # Asegurar que el proceso está muerto
        if process and process.poll() is None:
            try:
                process.kill()
            except:
                pass

        # Doble verificación: asegurar que el backup se restaura
        try:
            if parquet_backup.exists():
                shutil.copy2(parquet_backup, parquet_original)
                parquet_backup.unlink()
                print(f"✅ Limpieza final: backup restaurado y eliminado")
        except:
            pass

        report_progress['active'] = False
        obtener_resumen_datos.cache_clear()


def crear_pagina_actualizacion():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("🔄 Panel de Actualización de Datos", className="text-center my-4"),
                dcc.Link(dbc.Button("⬅️ Volver al Inicio", color="secondary", outline=True), href="/", className="mb-4 d-inline-block")
            ], width=12)
        ]),
        
        dbc.Row([
            # --- COLUMNA 1: OPTA ---
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4([
                        html.Img(src="/assets/opta_logo.png", style={"height": "24px", "marginRight": "8px", "verticalAlign": "middle"}),
                        "OPTA"
                    ], className="mb-0 text-white"), style={"background": "#1E90FF"}),
                    dbc.CardBody([
                        html.Label("Competición:"),
                        dcc.Dropdown(id='opta-liga-dropdown', placeholder="Seleccionar liga...", className="mb-3"),
                        html.Label("Temporada:"),
                        dcc.Dropdown(id='opta-temporada-dropdown', placeholder="Seleccionar temp...", className="mb-3"),
                        dbc.Row([
                            dbc.Col([html.Label("J. Inicial:"), dbc.Input(id="opta-jornada-inicial", type="number", value=1)], width=6),
                            dbc.Col([html.Label("J. Final:"), dbc.Input(id="opta-jornada-final", type="number", value=1)], width=6),
                        ], className="mb-3"),
                        dbc.Button("Descargar Opta", id="btn-update-opta", color="primary", className="w-100", disabled=True),
                    ])
                ], className="shadow mb-4")
            ], width=12, lg=4),

            # --- COLUMNA 2: MEDIACOACH (Sustituir en crear_pagina_actualizacion) ---
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4([
                        html.Img(src="/assets/mediacoach_logo.png", style={"height": "24px", "marginRight": "8px", "verticalAlign": "middle"}),
                        "MEDIACOACH"
                    ], className="mb-0 text-white"), style={"background": "#DC143C"}),
                    dbc.CardBody([
                        html.Label("Temporada:"),
                        dcc.Dropdown(id='mediacoach-temporada-dropdown', placeholder="Seleccionar temp...", className="mb-3"),

                        html.Label("Competición:"),
                        dcc.Dropdown(id='mediacoach-liga-dropdown', placeholder="Esperando temporada...", className="mb-3"),
                        dbc.Row([
                            dbc.Col([html.Label("J. Inicial:"), dbc.Input(id="mediacoach-jornada-inicial", type="number", value=1)], width=6),
                            dbc.Col([html.Label("J. Final:"), dbc.Input(id="mediacoach-jornada-final", type="number", value=1)], width=6),
                        ], className="mb-3"),
                        dbc.Button("Descargar MediaCoach", id="btn-update-mediacoach", color="danger", className="w-100", disabled=True),
                    ])
                ], className="shadow mb-4")
            ], width=12, lg=4),

            # --- COLUMNA 3: SPORTIAN ---
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4([
                        html.Img(src="/assets/sportian_logo.png", style={"height": "24px", "marginRight": "8px", "verticalAlign": "middle"}),
                        "SPORTIAN"
                    ], className="mb-0 text-white"), style={"background": "#FFD700"}),
                    dbc.CardBody([
                        html.P("Arrastra un CSV de corners o faltas para procesarlo."),
                        dcc.Upload(
                            id='sportian-upload',
                            children=html.Div([
                                html.I(className="fas fa-cloud-upload-alt", style={"fontSize": "48px", "color": "#FFD700"}),
                                html.P("Arrastra un archivo CSV aquí", className="mt-2 mb-1"),
                                html.P("o haz clic para seleccionar", className="text-muted small")
                            ]),
                            style={
                                "height": "160px", "border": "2px dashed #FFD700", "borderRadius": "10px",
                                "display": "flex", "alignItems": "center", "justifyContent": "center",
                                "cursor": "pointer", "backgroundColor": "#fffef5"
                            },
                            multiple=False,
                            accept='.csv'
                        ),
                        html.Div(id='sportian-upload-status', className="mt-2 text-center"),
                        dbc.Button("Procesar CSV", id="btn-update-sportian", color="warning", className="w-100 mt-3", disabled=True),
                    ])
                ], className="shadow mb-4")
            ], width=12, lg=4),
        ]),

        # --- SECCIÓN DE PROGRESO (Común para todos) ---
        dbc.Card([
            dbc.CardBody([
                html.H5("Estado de la descarga:"),
                dbc.Progress(id="progress-bar", value=0, striped=True, animated=True, style={"height": "25px"}, className="mb-3"),
                html.Div(id="update-progress", style={
                    "height": "150px", "overflow-y": "auto", "background": "#222", 
                    "color": "#00FF00", "padding": "10px", "font-family": "monospace", "fontSize": "12px"
                })
            ])
        ], className="shadow"),
        
        dcc.Interval(id='progress-interval', interval=1000, disabled=True)
    ], fluid=True)

# --- CALLBACKS ---

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='login-status', storage_type='session'),
    dcc.Store(id='mediacoach-seasons-data', storage_type='memory'),  # ← AQUÍ
    
    html.Div(id='login-wrapper', children=login_layout, style={'display': 'none'}),
    html.Div(id='app-wrapper', style={'display': 'none'})
])

@app.callback(
    [Output('login-wrapper', 'style'),
     Output('app-wrapper', 'style')],
    [Input('login-status', 'data')]
)
def control_acceso(logged):
    # Si es True, mostrar app
    if logged is True:
        return {'display': 'none'}, {'display': 'block'}
    
    # Si es None o False, mostrar login
    return {'display': 'block'}, {'display': 'none'}

@app.callback(
    Output('app-wrapper', 'children'),
    [Input('url', 'pathname')]
)
def navegar_internamente(path): 
    
    if path == '/actualizar':
        return crear_pagina_actualizacion()
    return crear_layout_principal()

@app.callback(
    Output('filtros-informe-container', 'children'),
    Input({'type': 'btn-informe', 'bloque': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def mostrar_filtros_informe(clicks):
    # Detectar qué botón fue clickeado
    ctx = dash.callback_context
    if not ctx.triggered or not any(clicks):
        raise dash.exceptions.PreventUpdate
    
    # Obtener el bloque clickeado (ABP, FISICO, TACTICO)
    trigger_id = ctx.triggered_id
    bloque = trigger_id['bloque']
    
    # Leer configuración del bloque
    config = BLOQUES_CONFIG[bloque]
    parquet_path = BASE_PATH / config['parquet']
    
    # Leer datos para obtener equipos y jornadas
    try:
        df = pd.read_parquet(parquet_path, columns=[config['col_equipo'], config['col_jornada']])
        
        # Equipos únicos
        equipos = sorted(df[config['col_equipo']].dropna().unique())
        opciones_equipos = [{'label': eq, 'value': eq} for eq in equipos]
        
        # Rango de jornadas
        jornadas = df[config['col_jornada']].dropna()
        jornada_min = int(jornadas.min())
        jornada_max = int(jornadas.max())
        
    except Exception as e:
        return dbc.Alert(f"Error cargando datos: {str(e)}", color="danger")
    
    # Crear el formulario de filtros
    return dbc.Card([
        dbc.CardHeader(html.H5(f"Filtros para informe {bloque}", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Equipo"),
                    dcc.Dropdown(
                        id='informe-equipo',
                        options=opciones_equipos,
                        placeholder="Selecciona un equipo"
                    )
                ], width=12, className="mb-3"),
                
                dbc.Col([
                    dbc.Label("Jornada Inicial"),
                    dbc.Input(
                        id='informe-jornada-ini',
                        type="number",
                        min=jornada_min,
                        max=jornada_max,
                        value=jornada_min
                    )
                ], width=6, className="mb-3"),
                
                dbc.Col([
                    dbc.Label("Jornada Final"),
                    dbc.Input(
                        id='informe-jornada-fin',
                        type="number",
                        min=jornada_min,
                        max=jornada_max,
                        value=jornada_max
                    )
                ], width=6, className="mb-3"),
            ]),
            
            dbc.Button(
                "Generar Informe", 
                id='btn-generar-informe',
                color="success",
                size="lg",
                className="w-100"
            ),
            
            # Guardamos el bloque actual en un Store
            dcc.Store(id='bloque-actual', data=bloque)
        ])
    ], className="mt-3")

@app.callback(
    Output('descarga-informe-container', 'children'),
    Input('btn-generar-informe', 'n_clicks'),
    [State('informe-equipo', 'value'),
     State('informe-jornada-ini', 'value'),
     State('informe-jornada-fin', 'value'),
     State('bloque-actual', 'data')],
    prevent_initial_call=True
)
def generar_informe(n_clicks, equipo, j_ini, j_fin, bloque):
    if not n_clicks or not equipo:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Obtener script del bloque
        config = BLOQUES_CONFIG[bloque]
        script_path = BASE_PATH / config['script']
        
        # Crear carpeta de informes si no existe
        carpeta_informes = BASE_PATH / 'informes_generados'
        carpeta_informes.mkdir(exist_ok=True)
        
        # Ejecutar script
        resultado = subprocess.run(
            ['python', str(script_path), 
             '--equipo', equipo,
             '--jornada_inicial', str(j_ini),
             '--jornada_final', str(j_fin)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutos máximo
        )
        
        if resultado.returncode == 0:
            # Buscar el PDF generado más reciente
            pdfs = list(carpeta_informes.glob(f"*{bloque}*.pdf"))
            if pdfs:
                pdf_mas_reciente = max(pdfs, key=lambda p: p.stat().st_mtime)
                nombre_archivo = pdf_mas_reciente.name
                
                return dbc.Alert([
                    html.H5("✅ Informe generado correctamente", className="alert-heading"),
                    html.Hr(),
                    dbc.Button(
                        "📥 Descargar Informe",
                        href=f"/descargar/{nombre_archivo}",
                        color="success",
                        size="lg",
                        external_link=True
                    )
                ], color="success")
            else:
                return dbc.Alert("⚠️ Informe generado pero no se encontró el PDF", color="warning")
        else:
            return dbc.Alert(
                f"❌ Error generando informe: {resultado.stderr}", 
                color="danger"
            )
            
    except Exception as e:
        return dbc.Alert(f"❌ Error: {str(e)}", color="danger")

# CALLBACK 1: Mostrar selectores al hacer clic en botón
@app.callback(
    [Output('report-selectors-container', 'children'),
     Output('report-selectors-container', 'style'),
     Output('selected-report-block', 'data')],
    [Input('btn-rep-abp', 'n_clicks'),
     Input('btn-rep-fisico', 'n_clicks'),
     Input('btn-rep-tactic', 'n_clicks')],
    prevent_initial_call=True
)
def mostrar_selectores(n_abp, n_fisico, n_tactic):
    ctx = dash.callback_context
    if not ctx.triggered:
        return [], {'display': 'none'}, None
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    button_id_base = button_id.replace('btn-rep-', '').upper()
    # Mapear nombres de botones a claves de configuración
    bloque = 'TACTICO' if button_id_base == 'TACTIC' else button_id_base
        
    config = BLOQUES_CONFIG[bloque]
    
    # Cargar equipos y jornadas del parquet
    try:
        df = pd.read_parquet(config['parquet'])
        equipos = sorted(df[config['col_equipo']].unique())
        
        # Normalizar jornadas y filtrar J0
        if 'Jornada_num' not in df.columns:
            df['Jornada_num'] = df[config['col_jornada']].apply(
                lambda x: int(str(x).replace('J', '').replace('j', '').strip()) if pd.notna(x) and str(x).strip() else 0
            )
        # Excluir jornada 0 (datos inválidos)
        jornadas = sorted([j for j in df['Jornada_num'].unique() if j > 0])
        
    except Exception as e:
        print(f"Error cargando datos: {e}")
        equipos, jornadas = [], []
    
    # Crear opciones de equipos con escudos (HTML en label)
    equipo_options = []
    for e in equipos:
        escudo_b64 = get_escudo_base64(e)
        if escudo_b64:
            # Opción con escudo como imagen inline
            equipo_options.append({
                'label': html.Div([
                    html.Img(src=escudo_b64, style={'height': '24px', 'marginRight': '10px', 'verticalAlign': 'middle'}),
                    html.Span(e, style={'verticalAlign': 'middle', 'fontWeight': '500'})
                ], style={'display': 'flex', 'alignItems': 'center'}),
                'value': e
            })
        else:
            equipo_options.append({'label': e, 'value': e})

    contenido = dbc.Card([
        dbc.CardBody([
            html.H5(f"📊 Configuración Informe {bloque}",
                   className="text-center mb-4 fw-bold",
                   style={'color': '#1e3d59', 'borderBottom': '2px solid #ffc107', 'paddingBottom': '10px'}),

            # Selector de equipo con escudos - ancho completo
            html.Div([
                html.Label("🏟️ Selecciona Equipo:", className="fw-bold mb-2", style={'fontSize': '14px'}),
                dcc.Dropdown(
                    id='report-team-selector',
                    options=equipo_options,
                    placeholder="Buscar equipo...",
                    style={'fontSize': '14px'},
                    className="mb-3"
                )
            ], className="mb-3"),

            # Selectores de jornada en fila
            dbc.Row([
                dbc.Col([
                    html.Label("📅 Desde Jornada:", className="fw-bold mb-2", style={'fontSize': '13px'}),
                    dcc.Dropdown(
                        id='report-jornada-inicio',
                        options=[{'label': f"Jornada {j}", 'value': j} for j in jornadas],
                        placeholder="Inicio...",
                        style={'fontSize': '13px'}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("📅 Hasta Jornada:", className="fw-bold mb-2", style={'fontSize': '13px'}),
                    dcc.Dropdown(
                        id='report-jornada-fin',
                        options=[{'label': f"Jornada {j}", 'value': j} for j in jornadas],
                        placeholder="Fin...",
                        style={'fontSize': '13px'}
                    )
                ], width=6),
            ], className="mb-4"),

            dbc.Button("🚀 Generar Informe PDF", id="btn-generate-report",
                      color="warning", className="w-100 fw-bold", disabled=True,
                      style={'fontSize': '16px', 'padding': '12px', 'borderRadius': '8px'})
        ], style={'padding': '20px'})
    ], className="border-0 shadow-sm mt-3", style={'borderRadius': '12px', 'backgroundColor': '#f8f9fa'})
    
    return contenido, {'display': 'block'}, bloque

# CALLBACK 2: Habilitar botón generar cuando hay equipo y jornada
@app.callback(
    Output('btn-generate-report', 'disabled'),
    [Input('report-team-selector', 'value'),
     Input('report-jornada-inicio', 'value'),
     Input('report-jornada-fin', 'value')]
)
def habilitar_generar(equipo, j_inicio, j_fin):
    return not (equipo and j_inicio and j_fin)

# CALLBACK 3: Ejecutar generación directamente en hilo
@app.callback(
    [Output('report-modal', 'is_open'),
     Output('report-interval', 'disabled'),
     Output('report-team-name', 'data')],
    Input('btn-generate-report', 'n_clicks'),
    [State('selected-report-block', 'data'),
     State('report-team-selector', 'value'),
     State('report-jornada-inicio', 'value'),
     State('report-jornada-fin', 'value')],
    prevent_initial_call=True
)
def ejecutar_generacion(n_clicks, bloque, equipo, j_inicio, j_fin):
    """Lanza la generación del informe en un hilo de fondo"""
    print(f">>> GENERANDO INFORME: Bloque={bloque}, Equipo={equipo}, Inicio={j_inicio}, Fin={j_fin}")

    if not all([bloque, equipo, j_inicio, j_fin]):
        raise dash.exceptions.PreventUpdate

    config = BLOQUES_CONFIG[bloque]
    script = config['script']

    # Crear carpeta de destino
    dest_folder = str(BASE_PATH / 'informes_generados')
    os.makedirs(dest_folder, exist_ok=True)

    # Lanzar en hilo de fondo
    threading.Thread(
        target=run_report_process,
        args=(script, equipo, str(j_inicio), str(j_fin), dest_folder),
        daemon=True
    ).start()

    print(f">>> HILO LANZADO para {script}")
    return True, False, equipo

# CALLBACK 4: Actualizar la barra de progreso (lee estado del archivo JSON)
@app.callback(
    [Output("report-modal-body", "children"),
     Output("report-modal-title", "children"),
     Output("report-interval", "disabled", allow_duplicate=True)],
    [Input("report-interval", "n_intervals")],
    [State("report-team-name", "data")],
    prevent_initial_call=True
)
def update_report_ui(n, team_name):
    """Lee el estado del report_progress global y actualiza el modal"""
    global report_progress

    progress = report_progress.get('progress', 0)
    status_msg = report_progress.get('status', 'Iniciando...')
    is_active = report_progress.get('active', False)
    final_path = report_progress.get('final_path', '')

    # --- INICIANDO (progreso bajo, activo) ---
    if is_active and progress <= 5:
        body = html.Div([
            html.Div([
                dbc.Spinner(color="warning", size="lg", spinner_style={"width": "3rem", "height": "3rem"}),
            ], className="text-center mb-3"),
            html.H5(status_msg, className="text-center text-muted"),
            html.P("El informe se está preparando...",
                   className="text-center text-muted small mt-2"),
            dbc.Progress(value=100, striped=True, animated=True, color="warning",
                        style={"height": "6px"}, className="mt-3"),
        ], className="py-4")
        return body, "Iniciando...", False

    # --- EN PROGRESO ---
    if is_active and progress < 100:
        body = html.Div([
            html.Div([
                html.H2(f"{progress}%", className="text-center fw-bold mb-0",
                        style={"color": "#0d6efd", "fontSize": "2.5rem"}),
            ], className="mb-3"),
            dbc.Progress(
                value=progress, striped=True, animated=True, color="primary",
                style={"height": "20px", "borderRadius": "10px"},
                className="mb-3"
            ),
            html.P(status_msg, className="text-center fw-bold", style={"fontSize": "14px"}),
        ], className="py-3")
        return body, "Generando Informe...", False

    # --- TERMINADO (activo=False, progreso=100) ---
    if not is_active and progress >= 100:
        has_error = '❌' in status_msg or 'Error' in status_msg or '⚠️' in status_msg

        # --- ERROR ---
        if has_error and not final_path:
            body = html.Div([
                html.Div([
                    html.I(className="bi bi-exclamation-triangle-fill",
                           style={"fontSize": "60px", "color": "#dc3545"})
                ], className="text-center mb-3"),
                html.H5("Error al generar el informe", className="text-center text-danger mb-2"),
                html.P(status_msg, className="text-center text-muted small mb-4",
                       style={"wordBreak": "break-word"}),
                dbc.Button(
                    "Reintentar",
                    id="btn-descargar-pdf",
                    color="danger",
                    className="w-100",
                    style={"borderRadius": "10px"}
                )
            ], className="py-3")
            return body, "Error", True

        # --- COMPLETADO CON PDF ---
        if final_path and os.path.exists(final_path):
            pdf_name = os.path.basename(final_path)

            # Obtener escudo del equipo
            escudo_src = get_escudo_base64(team_name) if team_name else None
            escudo_element = html.Img(
                src=escudo_src, style={"height": "120px", "objectFit": "contain"}
            ) if escudo_src else html.I(
                className="bi bi-check-circle-fill",
                style={"fontSize": "80px", "color": "#198754"}
            )

            # Calcular tamaño
            file_size_bytes = os.path.getsize(final_path)
            if file_size_bytes >= 1024 * 1024:
                file_size_str = f" ({file_size_bytes / (1024 * 1024):.1f} MB)"
            else:
                file_size_str = f" ({file_size_bytes / 1024:.0f} KB)"

            body = html.Div([
                html.Div([escudo_element], className="text-center mb-3"),
                html.H4("Informe completado", className="text-center fw-bold text-success mb-1"),
                html.P(team_name or "", className="text-center text-muted mb-4", style={"fontSize": "18px"}),
                html.A(
                    [html.I(className="bi bi-download me-2"), f"Descargar PDF{file_size_str}"],
                    href=f"/descargar/{pdf_name}",
                    download=pdf_name,
                    className="btn btn-success btn-lg w-100 mb-2",
                    style={"textDecoration": "none", "display": "block", "textAlign": "center",
                           "borderRadius": "10px", "padding": "14px", "fontSize": "16px"}
                ),
                dbc.Button(
                    "Generar otro informe",
                    id="btn-descargar-pdf",
                    color="outline-secondary",
                    className="w-100 mt-1",
                    style={"borderRadius": "10px"}
                )
            ], className="py-3")
            return body, "Informe Completado", True

        # --- COMPLETADO SIN PDF (warning) ---
        body = html.Div([
            html.Div([
                html.I(className="bi bi-exclamation-circle-fill",
                       style={"fontSize": "60px", "color": "#ffc107"})
            ], className="text-center mb-3"),
            html.H5(status_msg, className="text-center text-warning mb-2"),
            dbc.Button(
                "Reintentar",
                id="btn-descargar-pdf",
                color="warning",
                className="w-100",
                style={"borderRadius": "10px"}
            )
        ], className="py-3")
        return body, "Aviso", True

    # --- DEFAULT: spinner ---
    body = html.Div([
        dbc.Spinner(color="primary", size="lg"),
        html.P(status_msg, className="text-muted mt-3 text-center")
    ], className="text-center py-4")
    return body, "Procesando...", False

@app.callback(
    [Output('report-selectors-container', 'style', allow_duplicate=True),
     Output('report-selectors-container', 'children', allow_duplicate=True),
     Output('selected-report-block', 'data', allow_duplicate=True),
     Output('report-modal', 'is_open', allow_duplicate=True),
     Output('report-team-name', 'data', allow_duplicate=True)],
    Input('btn-descargar-pdf', 'n_clicks'),
    prevent_initial_call=True
)
def resetear_formulario_informe(n_clicks):
    """Resetea el formulario para generar otro informe"""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    return (
        {'display': 'none'},          # Ocultar filtros
        [],                           # Limpiar filtros
        None,                         # Resetear bloque
        False,                        # Cerrar modal
        None                          # Limpiar team name
    )
    
@app.callback(
    [Output('report-selectors-container', 'style', allow_duplicate=True),
     Output('report-selectors-container', 'children', allow_duplicate=True),
     Output('selected-report-block', 'data', allow_duplicate=True),
     Output('report-modal', 'is_open', allow_duplicate=True),
     Output('report-team-name', 'data', allow_duplicate=True)],
    Input('btn-reset-informe', 'n_clicks'),
    prevent_initial_call=True
)
def reset_informe_form(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    return {'display': 'none'}, [], None, False, None
    
@app.callback(
    Output('btn-update-mediacoach', 'disabled'),
    [Input('mediacoach-liga-dropdown', 'value'),
     Input('mediacoach-temporada-dropdown', 'value')]
)
def enable_btn_mediacoach(l, t):
    return not (l and t)

# --- CALLBACKS SPORTIAN ---
@app.callback(
    [Output('btn-update-sportian', 'disabled'),
     Output('sportian-upload-status', 'children')],
    Input('sportian-upload', 'contents'),
    State('sportian-upload', 'filename'),
    prevent_initial_call=True
)
def sportian_file_uploaded(contents, filename):
    if contents is None:
        return True, ""

    # Verificar que sea CSV de corners o faltas
    if filename and filename.lower().endswith('.csv'):
        nombre_lower = filename.lower()
        if 'corner' in nombre_lower or 'falta' in nombre_lower:
            return False, html.Div([
                html.I(className="fas fa-check-circle text-success me-2"),
                html.Span(f"Archivo listo: {filename}", className="text-success")
            ])
        else:
            return True, html.Div([
                html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                html.Span("El archivo debe contener 'corners' o 'faltas' en el nombre", className="text-warning")
            ])

    return True, html.Div([
        html.I(className="fas fa-times-circle text-danger me-2"),
        html.Span("Solo se aceptan archivos CSV", className="text-danger")
    ])

@app.callback(
    [Output('progress-interval', 'disabled', allow_duplicate=True),
     Output('btn-update-sportian', 'disabled', allow_duplicate=True),
     Output('sportian-upload-status', 'children', allow_duplicate=True)],
    Input('btn-update-sportian', 'n_clicks'),
    [State('sportian-upload', 'contents'),
     State('sportian-upload', 'filename')],
    prevent_initial_call=True
)
def process_sportian_csv(n_clicks, contents, filename):
    if not n_clicks or not contents:
        raise dash.exceptions.PreventUpdate

    global progress_data
    progress_data = {'active': True, 'progress': 0, 'status': 'Procesando CSV Sportian...', 'messages': []}

    def cb(p, s, msgs):
        global progress_data
        progress_data.update({'progress': p, 'status': s, 'messages': msgs})
        if p >= 100: progress_data['active'] = False

    # Lanzar proceso en hilo separado
    threading.Thread(
        target=actualizar_datos.process_sportian_csv_upload,
        args=(contents, filename, cb),
        daemon=True
    ).start()

    return False, True, html.Div([
        html.I(className="fas fa-spinner fa-spin text-warning me-2"),
        html.Span("Procesando...", className="text-warning")
    ])

@app.callback(
    [Output('sportian-upload', 'contents', allow_duplicate=True),
     Output('sportian-upload-status', 'children', allow_duplicate=True)],
    Input('progress-interval', 'n_intervals'),
    State('progress-interval', 'disabled'),
    prevent_initial_call=True
)
def clear_sportian_after_complete(n, disabled):
    global progress_data
    if disabled or progress_data.get('active', True):
        raise dash.exceptions.PreventUpdate

    # Si el proceso terminó y hay mensaje de éxito, limpiar el upload
    if progress_data.get('progress', 0) >= 100:
        return None, html.Div([
            html.I(className="fas fa-check-circle text-success me-2"),
            html.Span("¡Procesado correctamente!", className="text-success")
        ])

    raise dash.exceptions.PreventUpdate

@app.callback(
    [Output('login-status', 'data'), Output('login-error', 'children')],
    Input('login-button', 'n_clicks'),
    [State('username', 'value'), State('password', 'value')],
    prevent_initial_call=True
)
def login_callback(n, u, p):
    if u == USERNAME and p == PASSWORD: return True, ""
    return False, "Credenciales incorrectas"

@app.callback(
    [Output('progress-interval', 'disabled', allow_duplicate=True), 
     Output('btn-update-mediacoach', 'disabled', allow_duplicate=True)],
    Input('btn-update-mediacoach', 'n_clicks'),
    [State('mediacoach-liga-dropdown', 'value'), 
     State('mediacoach-temporada-dropdown', 'value'), # Aquí va el ID de la temporada
     State('mediacoach-temporada-dropdown', 'options'), # Para sacar el nombre de la temporada
     State('mediacoach-jornada-inicial', 'value'), 
     State('mediacoach-jornada-final', 'value')],
    prevent_initial_call=True
)
def start_mediacoach_update(n, liga, season_id, season_options, ji, jf):
    if not n: raise dash.exceptions.PreventUpdate
    
    # Obtener el nombre de la temporada (ej: "Temporada 24-25") a partir del ID
    season_name = next((opt['label'] for opt in season_options if opt['value'] == season_id), season_id)
    
    global progress_data
    progress_data = {'active': True, 'progress': 0, 'status': 'Preparando MediaCoach...', 'messages': []}
    
    def cb(p, s, msgs):
        global progress_data
        progress_data.update({'progress': p, 'status': s, 'messages': msgs})
        if p >= 100: progress_data['active'] = False

    # Lanzamos el proceso en un hilo separado para que la web no se congele
    threading.Thread(
        target=actualizar_datos.update_mediacoach_data_web, 
        args=(liga, season_name, ji, jf, cb), 
        daemon=True
    ).start()
    
    return False, True

# Callback 1: Solo actualiza el Store (siempre existe)
@app.callback(
    Output('mediacoach-seasons-data', 'data'),
    Input('url', 'pathname'),
    prevent_initial_call=True  # ✅ No se ejecuta al cargar
)

def load_mediacoach_seasons_data(path):
    """Solo carga datos cuando se navega a /actualizar"""
    if path != '/actualizar': 
        raise dash.exceptions.PreventUpdate
    print("Consultando API MediaCoach para temporadas...")
    datos_completos = actualizar_datos.get_mediacoach_seasons_api()
    return datos_completos

# Callback 1b: Actualiza el dropdown de temporadas desde el Store
@app.callback(
    Output('mediacoach-temporada-dropdown', 'options'),
    Input('mediacoach-seasons-data', 'data'),
    prevent_initial_call=True
)
def update_mediacoach_temporada_dropdown(seasons_data):
    """Actualiza dropdown solo cuando hay datos en el Store"""
    if not seasons_data:
        raise dash.exceptions.PreventUpdate
    opciones_dropdown = [{'label': t['label'], 'value': t['value']} for t in seasons_data]
    return opciones_dropdown

# Callback 2: Carga las ligas al elegir temporada
@app.callback(
    Output('mediacoach-liga-dropdown', 'options'),
    [Input('mediacoach-temporada-dropdown', 'value'),
     Input('mediacoach-seasons-data', 'data')],
    prevent_initial_call=True
)
def load_mediacoach_ligas(season_id, seasons_data):
    if not season_id or not seasons_data:
        return []
    
    # Buscar la temporada seleccionada
    for temp in seasons_data:
        if temp['value'] == season_id:
            competiciones = temp.get('competitions', [])
            opciones = []
            for c in competiciones:
                nombre = c.get('name') or c.get('Name') or "Liga desconocida"
                # Mapear nombres a los que usa tu script
                if 'la liga' in nombre.lower() and 'segunda' not in nombre.lower():
                    nombre_final = 'La Liga'
                elif 'segunda' in nombre.lower():
                    nombre_final = 'La Liga 2'
                else:
                    nombre_final = nombre
                opciones.append({'label': nombre_final, 'value': nombre_final})
            return opciones
    
    return [{'label': 'La Liga (Backup)', 'value': 'La Liga'}]

@app.callback(
    Output('page-content', 'children'), 
    [Input('login-status', 'data'), Input('url', 'pathname')]
)
def display_page(logged, path):
    # 1. Si el estado es None, Dash está cargando la sesión. No hagas nada.
    if logged is None:
        raise dash.exceptions.PreventUpdate

    # 2. Si el estado es explícitamente False, ve al login.
    if logged is False:
        return login_layout
    
    # 3. Si es True, permite navegar.
    if path == '/actualizar': 
        return crear_pagina_actualizacion()
    
    return crear_layout_principal()

@app.callback(
    Output('jornadas-container', 'children'),
    [Input('liga-dropdown', 'value'), Input('temporada-dropdown', 'value')]
)
def update_jornadas(liga, temp):
    df = obtener_resumen_datos()
    if df.empty: return html.Div("No hay datos cargados.", className="text-center mt-5")
    
    df_f = df[(df['liga'] == liga) & (df['temporada'] == temp)]
    cards = []
    
    jornadas_unicas = sorted(df_f['jornada'].unique(), reverse=True)
    
    for j in jornadas_unicas:
        try: j_int = int(j)
        except: continue

        df_j = df_f[df_f['jornada'] == j]
        
        m_count = df_j[df_j['fuente'] == 'mediacoach']['n_partidos'].sum()
        o_count = df_j[df_j['fuente'] == 'opta']['n_partidos'].sum()
        s_count = df_j[df_j['fuente'] == 'sportian']['n_partidos'].sum()

        # Gráfico (Sportian en blanco)
        fig = go.Figure(data=[go.Pie(
            labels=['MediaCoach', 'Opta', 'Sportian'],
            values=[1 if m_count > 0 else 0, 1 if o_count > 0 else 0, 1 if s_count > 0 else 0],
            hole=.7, 
            marker=dict(colors=['#DC143C', '#1E90FF', '#FFFFFF'], line=dict(color='#444', width=0.5)),
            textinfo='none'
        )])
        fig.update_layout(showlegend=False, height=75, margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')

        cards.append(dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6(f"J {j_int}", className="text-center fw-bold mb-1"), 
                    dcc.Graph(figure=fig, config={'displayModeBar': False}),
                    html.Div([
                        html.Div([
                            html.Img(src=get_logo_base64("mediacoach_logo.png")), 
                            html.Span(f"{int(m_count)} part." if m_count > 0 else "--", className="ms-auto")
                        ], className="d-flex align-items-center mb-1"),
                        html.Div([
                            html.Img(src=get_logo_base64("opta_logo.png")), 
                            html.Span(f"{int(o_count)} part." if o_count > 0 else "--", className="ms-auto")
                        ], className="d-flex align-items-center mb-1"),
                        html.Div([
                            html.Img(src=get_logo_base64("sportian_logo.png")), 
                            html.Span(f"{int(s_count)} part." if s_count > 0 else "--", className="ms-auto")
                        ], className="d-flex align-items-center"),
                    ], className="fuente-stats mt-1")
                ])
            ], className="shadow-sm jornada-card")
        ], width=4, md=2, lg=1, className="mb-2 px-1")) 
        
    return dbc.Row(cards, className="g-1 justify-content-start")

# --- CALLBACKS DE ACTUALIZACIÓN ---
@app.callback(
    Output('opta-liga-dropdown', 'options'), 
    Input('url', 'pathname') # Quitamos prevent_initial_call
)
def load_ligas(path):
    if path != '/actualizar': 
        raise dash.exceptions.PreventUpdate
    try:
        # Añade un print aquí para ver en tu terminal si la API responde
        print("Consultando ligas Opta...")
        c = actualizar_datos.get_all_competitions_and_stages()
        if not c:
            print("La API de Opta devolvió un diccionario vacío")
            return []
        return [{'label': v['name'], 'value': k} for k, v in c.items()]
    except Exception as e:
        print(f"Error cargando ligas: {e}")
        return []

@app.callback(
    Output('opta-temporada-dropdown', 'options'), 
    Input('opta-liga-dropdown', 'value')
)
def load_temps(liga):
    if not liga: 
        return [] # Retornar vacío en lugar de PreventUpdate para limpiar el dropdown

    try:
        c = actualizar_datos.get_all_competitions_and_stages()
        if liga in c:
            # En tu archivo actualizar_datos.py, la clave es 'season'
            return [{'label': v['season'], 'value': k} for k, v in c[liga]['stages'].items()]
    except Exception as e:
        print(f"Error cargando temporadas: {e}")
    return []

@app.callback(Output('btn-update-opta', 'disabled'), [Input('opta-liga-dropdown', 'value'), Input('opta-temporada-dropdown', 'value')])
def enable_btn(l, t): return not (l and t)

@app.callback(
    [Output('progress-interval', 'disabled', allow_duplicate=True), 
     # AÑADIMOS allow_duplicate=True AQUÍ ABAJO:
     Output('btn-update-opta', 'disabled', allow_duplicate=True)], 
    Input('btn-update-opta', 'n_clicks'),
    [State('opta-liga-dropdown', 'value'), 
     State('opta-temporada-dropdown', 'value'),
     State('opta-jornada-inicial', 'value'), 
     State('opta-jornada-final', 'value')],
    prevent_initial_call=True
)
def start_opta_update(n, comp_id, stage_id, ji, jf):
    if not n: raise dash.exceptions.PreventUpdate
    
    global progress_data
    progress_data = {'active': True, 'progress': 0, 'status': 'Iniciando Opta...', 'messages': [], 'error': False}
    
    def cb(p, s, msgs):
        global progress_data
        progress_data.update({'progress': p, 'status': s, 'messages': msgs})
        if p >= 100:
            progress_data['active'] = False
            # Detectar si el status indica error
            if s and ('error' in s.lower() or 'fatal' in s.lower()):
                progress_data['error'] = True

    def safe_opta_update():
        """Wrapper para capturar errores no controlados del hilo"""
        try:
            actualizar_datos.update_opta_data_web(comp_id, stage_id, ji, jf, cb)
        except Exception as e:
            import logging
            import traceback
            logging.error(f"Error fatal en hilo de actualización Opta: {e}", exc_info=True)
            error_tb = traceback.format_exc()
            timestamp = datetime.now().strftime("%H:%M:%S")
            global progress_data
            # Añadir el error como mensajes visibles en la consola
            error_msgs = progress_data.get('messages', [])
            error_msgs.append({'timestamp': timestamp, 'message': '=' * 50, 'type': 'error'})
            error_msgs.append({'timestamp': timestamp, 'message': f'💥 ERROR FATAL: {e}', 'type': 'error'})
            for line in error_tb.strip().split('\n'):
                error_msgs.append({'timestamp': timestamp, 'message': line, 'type': 'error'})
            error_msgs.append({'timestamp': timestamp, 'message': '=' * 50, 'type': 'error'})
            progress_data.update({
                'progress': 100,
                'status': f'Error fatal: {e}',
                'messages': error_msgs,
                'error': True,
                'active': False
            })

    threading.Thread(
        target=safe_opta_update,
        daemon=True
    ).start()
    
    return False, True

def _parse_range(range_header, file_size):
    """Parsea la cabecera Range del navegador. Devuelve (start, end) o None."""
    if not range_header or not range_header.startswith('bytes='):
        return None
    try:
        range_spec = range_header[6:]  # quitar "bytes="
        start_str, end_str = range_spec.split('-', 1)
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        if start < 0 or end >= file_size or start > end:
            return None
        return (start, end)
    except (ValueError, IndexError):
        return None


@server.route('/descargar/<path:filename>')
def descargar_informe(filename):
    """Endpoint para descargar informes generados (streaming con soporte Range)"""
    directorio = os.path.realpath(os.path.join(os.getcwd(), "informes_generados"))
    filepath = os.path.realpath(os.path.join(directorio, filename))

    # Protección contra path traversal
    if not filepath.startswith(directorio + os.sep):
        abort(403)

    if not os.path.isfile(filepath):
        abort(404)

    file_size = os.path.getsize(filepath)
    chunk_size = 64 * 1024  # 64 KB

    range_header = request.headers.get('Range')
    byte_range = _parse_range(range_header, file_size)

    if byte_range:
        # Respuesta parcial (206) para descargas reanudables
        start, end = byte_range
        length = end - start + 1

        def generate_range():
            with open(filepath, 'rb') as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    read_size = min(chunk_size, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        response = Response(
            generate_range(),
            status=206,
            mimetype='application/pdf',
            direct_passthrough=True
        )
        response.headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
        response.headers['Content-Length'] = length
    else:
        # Respuesta completa (200) con streaming
        def generate_full():
            with open(filepath, 'rb') as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    yield data

        response = Response(
            generate_full(),
            status=200,
            mimetype='application/pdf',
            direct_passthrough=True
        )
        response.headers['Content-Length'] = file_size

    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    return response


@app.callback(
    [Output('progress-bar', 'value'), 
     Output('update-progress', 'children'), 
     Output('progress-interval', 'disabled', allow_duplicate=True), 
     Output('btn-update-opta', 'disabled', allow_duplicate=True)],
    Input('progress-interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_ui(n):
    global progress_data

    if not progress_data['active'] and progress_data['progress'] >= 100:
        obtener_resumen_datos_cached.cache_clear()

    has_error = progress_data.get('error', False)

    # Convertimos los mensajes en párrafos HTML - errores en rojo
    msgs = []
    for m in progress_data.get('messages', []):
        msg_type = m.get('type', 'info')
        if msg_type == 'error':
            msgs.append(html.P(
                f"[{m['timestamp']}] {m['message']}",
                style={'color': '#FF4444', 'fontWeight': 'bold'}
            ))
        elif msg_type == 'warning':
            msgs.append(html.P(
                f"[{m['timestamp']}] {m['message']}",
                style={'color': '#FFAA00'}
            ))
        else:
            msgs.append(html.P(f"[{m['timestamp']}] {m['message']}"))

    # Si hay error, añadir banner visible al final
    if has_error and not progress_data['active']:
        msgs.append(html.Hr(style={'borderColor': '#FF4444'}))
        msgs.append(html.P(
            f"La descarga se ha detenido por un error. "
            f"Revisa los mensajes anteriores para más detalle.",
            style={
                'color': '#FF4444', 'fontWeight': 'bold', 'fontSize': '14px',
                'padding': '8px', 'background': '#330000', 'borderRadius': '4px'
            }
        ))

    # Retornamos: progreso, mensajes, si el intervalo se apaga, si el botón se habilita
    return progress_data['progress'], msgs, not progress_data['active'], progress_data['active']

@app.callback(
    Output('login-status', 'data', allow_duplicate=True), 
    Input('logout-btn', 'n_clicks'), 
    prevent_initial_call=True
)
def logout(n):
    if n is None or n == 0:
        raise dash.exceptions.PreventUpdate
    return False

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=True)