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
import glob

# --- En app.py, debajo de las rutas ---
EQUIPOS_REPORTE = [
    "1. Alavés", "2. Athletic Club", "3. Atlético de Madrid", "4. Barcelona", "5. Celta de Vigo",
    "6. Elche", "7. Espanyol", "8. Getafe", "9. Girona", "10. Levante",
    "11. Mallorca", "12. Osasuna", "13. Rayo Vallecano", "14. Real Betis", "15. Real Madrid",
    "16. Real Oviedo", "17. Real Sociedad", "18. Sevilla", "19. Valencia", "20. Villarreal"
]

# --- CONFIGURACIÓN DE RUTAS ---
BASE_PATH = Path(__file__).parent
ASSETS_PATH = BASE_PATH / 'assets'

FILE_MEDIACOACH = BASE_PATH / 'extraccion_mediacoach/data/estadisticas_equipo.parquet'
FILE_OPTA = BASE_PATH / 'extraccion_opta/datos_opta_parquet/abp_events.parquet'
FILE_SPORTIAN = BASE_PATH / 'extraccion_sportian/corners_tracking.parquet'

progress_data = {'active': False, 'progress': 0, 'status': 'Esperando...', 'messages': []}
report_progress = {'active': False, 'progress': 0, 'status': '', 'final_path': ''}


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

@lru_cache(maxsize=1)
def obtener_resumen_datos():
    """Lee los datos de todas las fuentes, normaliza y cuenta partidos únicos."""
    data_list = []
    
    # 1. MediaCoach (Ajustado a minúsculas según tu error)
    if FILE_MEDIACOACH.exists():
        try:
            # Columnas exactas del archivo: liga, temporada, jornada, partido
            df = pd.read_parquet(FILE_MEDIACOACH, columns=['liga', 'temporada', 'jornada', 'partido'])
            
            # Normalizamos los datos del archivo
            df['j_norm'] = df['jornada'].apply(normalizar_jornada)
            df['liga_norm'] = df['liga'].apply(normalizar_liga)
            df['temp_norm'] = df['temporada'].apply(temp_mediacoach)
            
            df = df.dropna(subset=['j_norm'])
            # Aseguramos que 'partido' no sea nulo antes de contar
            df['partido'] = df['partido'].fillna('sin_id') 
            
            # Agrupamos por los campos normalizados y contamos partidos únicos
            res = df.groupby(['liga_norm', 'temp_norm', 'j_norm'])['partido'].nunique().reset_index()
            
            for _, row in res.iterrows():
                data_list.append({
                    'liga': row['liga_norm'],
                    'temporada': row['temp_norm'],
                    'jornada': int(row['j_norm']),
                    'fuente': 'mediacoach',
                    'n_partidos': int(row['partido'])
                })
            print(f"✅ MediaCoach: {len(res)} jornadas procesadas.")
        except Exception as e: 
            print(f"❌ Error MediaCoach: {e}")

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
        except Exception as e: print(f"❌ Error Opta: {e}")

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
        except Exception as e: print(f"❌ Error Sportian: {e}")

    return pd.DataFrame(data_list)

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
        dbc.Navbar([
            dbc.Container([
                dbc.NavbarBrand([
                    html.Img(src=get_logo_base64("logodatos-villarrealcf.png"), height="80px", className="me-2"),
                    "Departamento de Datos"
                ], className="ms-2 fw-bold d-flex align-items-center"),
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("📄 Descargar PDF", id="download-pdf", href="#", className="nav-link-custom")),
                    dbc.NavItem(
                        html.A("🔄 Actualizar Datos", href="/actualizar", className="nav-link nav-link-custom", style={'cursor': 'pointer'})
                    ),
                    dbc.NavItem(dbc.NavLink("🚪 Cerrar Sesión", id="logout-btn", href="#", className="nav-link-custom"))
                ], className="ms-auto", navbar=True)
            ])
        ], color="dark", dark=True, className="mb-4 shadow"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Liga:", className="fw-bold mb-2"),
                                dcc.Dropdown(id='liga-dropdown', options=[{'label': l, 'value': l} for l in ligas],
                                             value=ligas[0] if ligas else None, clearable=False)
                            ], width=6),
                            dbc.Col([
                                html.Label("Temporada:", className="fw-bold mb-2"),
                                dcc.Dropdown(id='temporada-dropdown', options=[{'label': t, 'value': t} for t in temporadas],
                                             value=temporadas[0] if temporadas else None, clearable=False)
                            ], width=6),
                        ])
                    ])
                ], className="shadow-sm mb-4")
            ], width=8, className="mx-auto")
        ]),
        # --- Dentro de crear_layout_principal() en app.py ---
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    # 1. TÍTULO EN NEGRO
                    dbc.CardHeader(html.H5("📑 Generador de Informes PDF", className="text-dark fw-bold mb-0")),
                    
                    dbc.CardBody([
                        # 2. SELECTORES DE EQUIPO Y JORNADA
                        dbc.Row([
                            dbc.Col([
                                html.Label("Equipo:", className="fw-bold small"),
                                dcc.Dropdown(
                                    id='report-team-idx',
                                    options=[{'label': e, 'value': i+1} for i, e in enumerate(EQUIPOS_REPORTE)],
                                    placeholder="Seleccionar equipo..."
                                )
                            ], width=8),
                            dbc.Col([
                                html.Label("Hasta Jornada:", className="fw-bold small"),
                                dcc.Input(id='report-matchday', type='number', value=1, className="form-control")
                            ], width=4),
                        ], className="mb-3"),

                        # 3. RUTA DE GUARDADO LOCAL
                        html.Div([
                            html.Label("Ruta de guardado en el disco local:", className="fw-bold small"),
                            dbc.Input(id='report-save-path', placeholder="Ej: /Users/imac/Desktop/Informes", 
                                     value=os.path.expanduser("~/Downloads"), type="text"),
                            html.Small("Se guardará en esta carpeta de tu ordenador.", className="text-muted")
                        ], className="mb-4"),

                        # 4. BOTONES CON ICONOS 4 VECES MÁS GRANDES (100px)
                        dbc.Row([
                            dbc.Col([
                                dbc.Button([
                                    html.Div([
                                        html.Img(src="/assets/abp_icono.png", height="100px", className="mb-2"),
                                        html.Span("Informe ABP", className="fw-bold")
                                    ], className="d-flex flex-column align-items-center justify-content-center")
                                ], id="btn-rep-abp", color="danger", outline=True, className="w-100 py-3 shadow-sm")
                            ], width=4),
                            
                            dbc.Col([
                                dbc.Button([
                                    html.Div([
                                        html.Img(src="/assets/fisico_icono.png", height="100px", className="mb-2"),
                                        html.Span("Informe Físico", className="fw-bold")
                                    ], className="d-flex flex-column align-items-center justify-content-center")
                                ], id="btn-rep-fisico", color="success", outline=True, className="w-100 py-3 shadow-sm")
                            ], width=4),
                            
                            dbc.Col([
                                dbc.Button([
                                    html.Div([
                                        html.Img(src="/assets/tactica_icono.png", height="100px", className="mb-2"),
                                        html.Span("Informe Táctico", className="fw-bold")
                                    ], className="d-flex flex-column align-items-center justify-content-center")
                                ], id="btn-rep-tactic", color="primary", outline=True, className="w-100 py-3 shadow-sm")
                            ], width=4),
                        ], className="g-3 mb-4"),

                        # 5. BARRA DE PROGRESO Y ESTADO
                        html.Div(id="report-progress-container", children=[
                            html.P(id="report-status-text", className="small mb-1 fw-bold text-muted", children="Esperando selección..."),
                            dbc.Progress(id="report-progress-bar", value=0, striped=True, animated=True, color="info", style={"height": "15px"}),
                        ], style={"display": "none"}),
                        
                        dcc.Interval(id='report-interval', interval=800, disabled=True),
                        dcc.Download(id="download-pdf-report")
                    ])
                ], className="shadow mb-4 border-dark")
            ], width=10, className="mx-auto")
        ]),
        html.Div(id='jornadas-container'),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Información")),
            dbc.ModalBody("Esta función se implementará próximamente."),
            dbc.ModalFooter(dbc.Button("Cerrar", id="close-modal"))
        ], id="modal", is_open=False)
    ], fluid=True)

def run_report_process(script_name, equipo_idx, jornada, destination_folder):
    global report_progress
    report_progress = {'active': True, 'progress': 5, 'status': 'Iniciando generador...', 'final_path': ''}
    
    import sys
    import shutil
    import subprocess

    # Usamos -u para que el print sea unbuffered (tiempo real)
    cmd = [sys.executable, "-u", script_name, str(equipo_idx), str(jornada)]
    
    # bufsize=1 y universal_newlines=True para leer línea a línea
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1, 
        universal_newlines=True
    )
    
    for line in process.stdout:
        status_msg = line.strip()
        if not status_msg: continue
        
        # Imprimir en la terminal del Mac para debug
        print(f"DEBUG SCRIPT: {status_msg}")

        # Buscamos el patrón: --- [1/12] Ejecutando: abp1.py ---
        # Esta expresión regular extrae el número de página y el total
        match = re.search(r"\[(\d+)/(\d+)\] Ejecutando: (.*) ---", status_msg)
        
        if match:
            pag_actual = int(match.group(1))
            pag_total = int(match.group(2))
            nombre_pag = match.group(3)
            
            # Calculamos porcentaje (hasta el 95%)
            porcentaje = int((pag_actual / pag_total) * 95)
            
            report_progress['progress'] = porcentaje
            report_progress['status'] = f"Generando página {pag_actual} de {pag_total}: {nombre_pag}"
        
        elif "Uniendo todos los reportes" in status_msg:
            report_progress['progress'] = 96
            report_progress['status'] = "Uniendo páginas en el PDF final..."

    process.wait()
    
    # Buscar el PDF generado y moverlo a la ruta deseada
    lista_pdfs = glob.glob("*.pdf")
    if lista_pdfs:
        archivo_reciente = max(lista_pdfs, key=os.path.getctime)
        if os.path.exists(destination_folder):
            final_dest = os.path.join(destination_folder, archivo_reciente)
            try:
                shutil.move(archivo_reciente, final_dest)
                report_progress['final_path'] = final_dest
                report_progress['status'] = f"✅ Informe guardado con éxito."
            except Exception as e:
                report_progress['status'] = f"⚠️ PDF generado pero no se pudo mover: {archivo_reciente}"
        else:
            report_progress['status'] = f"✅ Informe generado en carpeta del proyecto: {archivo_reciente}"
    
    report_progress['progress'] = 100
    report_progress['active'] = False

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
                        html.P("Módulo de descarga de datos de tracking y eventos Sportian."),
                        html.Div(style={"height": "195px", "border": "1px dashed #ccc", "borderRadius": "5px"}, className="d-flex align-items-center justify-content-center", 
                                 children=[html.Span("Próximamente", className="text-muted")]),
                        dbc.Button("Descargar Sportian", id="btn-update-sportian", color="warning", className="w-100 mt-3", disabled=True),
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

# 1. Callback para iniciar el proceso
@app.callback(
    Output('report-interval', 'disabled'),
    [Input("btn-rep-abp", "n_clicks"),
     Input("btn-rep-fisico", "n_clicks"),
     Input("btn-rep-tactic", "n_clicks")],
    [State("report-team-idx", "value"),
     State("report-matchday", "value"),
     State("report-save-path", "value")],
    prevent_initial_call=True
)
def start_report(n1, n2, n3, eq_idx, jor, path):
    ctx = dash.callback_context
    if not eq_idx: return True
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    scripts = {
        "btn-rep-abp": "abp_informe_todo.py",
        "btn-rep-fisico": "fisico_completo_pdf.py",
        "btn-rep-tactic": "tactic_informe_todo.py"
    }
    
    threading.Thread(target=run_report_process, args=(scripts[button_id], eq_idx, jor, path), daemon=True).start()
    return False

# 2. Callback para actualizar la barra de progreso
@app.callback(
    [Output("report-progress-bar", "value"),
     Output("report-status-text", "children"),
     Output("report-progress-container", "style")],
    [Input("report-interval", "n_intervals")]
)
def update_report_ui(n):
    global report_progress
    if not report_progress['active'] and report_progress['progress'] == 0:
        return 0, "", {"display": "none"}
    
    display = {"display": "block"}
    return report_progress['progress'], report_progress['status'], display

@app.callback(
    Output('btn-update-mediacoach', 'disabled'),
    [Input('mediacoach-liga-dropdown', 'value'), 
     Input('mediacoach-temporada-dropdown', 'value')]
)
def enable_btn_mediacoach(l, t):
    return not (l and t)

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
        obtener_resumen_datos.cache_clear()

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
    progress_data = {'active': True, 'progress': 0, 'status': 'Iniciando Opta...', 'messages': []}
    
    def cb(p, s, msgs):
        global progress_data
        progress_data.update({'progress': p, 'status': s, 'messages': msgs})
        if p >= 100: 
            progress_data['active'] = False
            obtener_resumen_datos.cache_clear()

    threading.Thread(
        target=actualizar_datos.update_opta_data_web, 
        args=(comp_id, stage_id, ji, jf, cb), 
        daemon=True
    ).start()
    
    return False, True

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
    # Convertimos los mensajes en párrafos HTML para la consola verde
    msgs = [html.P(f"[{m['timestamp']}] {m['message']}") for m in progress_data['messages']]
    
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

@app.callback(Output('modal', 'is_open'), [Input('download-pdf', 'n_clicks'), Input('close-modal', 'n_clicks')], [State('modal', 'is_open')], prevent_initial_call=True)
def toggle_modal(n1, n2, is_open): return not is_open if (n1 or n2) else is_open

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)