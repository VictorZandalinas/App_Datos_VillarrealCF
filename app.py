import dash
from dash import dcc, html, Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import os
from pathlib import Path
import base64
import subprocess
import threading
from datetime import datetime
import actualizar_datos

# Variable global para el progreso
progress_data = {
    'active': False,
    'progress': 0,
    'status': 'Esperando...',
    'messages': []
}

# Inicializar la app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Departamento Datos - Villarreal CF")
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Departamento Datos - Villarreal CF</title>
        <link rel="icon" type="image/png" href="/assets/favicon.ico">
        
        <!-- Open Graph / Facebook / WhatsApp -->
        <meta property="og:type" content="website">
        <meta property="og:url" content="http://154.56.153.161:8050">
        <meta property="og:title" content="Departamento Datos - Villarreal CF">
        <meta property="og:description" content="Análisis de datos oficial del Villarreal CF">
        <meta property="og:image" content="http://154.56.153.161:8050/assets/logodatos-villarrealcf.png">
        
        <!-- Twitter -->
        <meta property="twitter:card" content="summary_large_image">
        <meta property="twitter:url" content="http://154.56.153.161:8050">
        <meta property="twitter:title" content="Departamento Datos - Villarreal CF">
        <meta property="twitter:description" content="Análisis de datos oficial del Villarreal CF">
        <meta property="twitter:image" content="http://154.56.153.161:8050/assets/logodatos-villarrealcf.png">
        
        {%metas%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.config.suppress_callback_exceptions = True  # <-- Añade esta línea
server = app.server

# Credenciales de login (en producción usar variables de entorno)
USERNAME = os.environ.get('APP_USERNAME', 'admin')
PASSWORD = os.environ.get('APP_PASSWORD', 'password123')

# Rutas de las carpetas
MEDIACOACH_PATH = Path('datos_mediacoach_parquet')
OPTA_PATH = Path('datos_opta_parquet')
ASSETS_PATH = Path('assets')


def get_logo_base64(filename):
    """Convierte una imagen a base64"""
    try:
        with open(ASSETS_PATH / filename, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{encoded}"
    except:
        return ""

def mapear_valores_filtros():
    """Mapea valores entre MediaCoach y Opta"""
    return {
        'liga_mapping': {
            'La Liga': 'Primera Division',  # MediaCoach -> Opta
            'Primera Division': 'La Liga'   # Opta -> MediaCoach
        },
        'temporada_mapping': {
            '24_25': '4xu8dwf3cotp5qu0ddi50wkyc',  # MediaCoach -> Opta
            '4xu8dwf3cotp5qu0ddi50wkyc': '24_25'   # Opta -> MediaCoach
        }
    }

def normalizar_valores_filtros():
    """Normaliza valores entre MediaCoach y Opta para mostrar valores unificados"""
    return {
        'liga_normalize': {
            'La Liga': 'La Liga',  # MediaCoach
            'Primera División': 'La Liga'  # Opta -> normalizado (nota la tilde!)
        },
        'temporada_normalize': {
            '24_25': 'Temporada 24/25',  # MediaCoach
            '4xu8dwf3cotp5qu0ddi50wkyc': 'Temporada 24/25'  # Opta -> normalizado
        }
    }

def contar_archivos_jornada(jornada_num):
    """Cuenta archivos por jornada en cada carpeta"""
    mediacoach_count = 0
    opta_count = 0
    
    # Contar archivos MediaCoach
    if MEDIACOACH_PATH.exists():
        for file in MEDIACOACH_PATH.glob('*.parquet'):
            try:
                df = pd.read_parquet(file)
                # Buscar columna de jornada
                jornada_col = None
                for col in ['week', 'jornada', 'Jornada']:
                    if col in df.columns:
                        jornada_col = col
                        break
                
                if jornada_col and jornada_num in df[jornada_col].values:
                    mediacoach_count += 1
            except:
                pass
    
    # Contar archivos Opta
    if OPTA_PATH.exists():
        for file in OPTA_PATH.glob('*.parquet'):
            try:
                df = pd.read_parquet(file)
                # Para Opta usar 'Week' con mayúscula
                if 'Week' in df.columns and jornada_num in df['Week'].values:
                    opta_count += 1
            except:
                pass
    
    total = mediacoach_count + opta_count
    mediacoach_percent = (mediacoach_count / total * 100) if total > 0 else 0
    opta_percent = (opta_count / total * 100) if total > 0 else 0
    
    return mediacoach_percent, opta_percent

def crear_pagina_actualizacion():
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2("🔄 Actualización de Datos", className="text-center mb-4"),
                html.Hr()
            ])
        ]),
        
        # Formulario de configuración para Opta
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("📈 Configuración de Descarga - Opta", className="card-title mb-4"),
                        
                        # Dropdown para Liga
                        dbc.Row([
                            dbc.Col([
                                html.Label("Liga:", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='opta-liga-dropdown',
                                    placeholder="Selecciona una liga...",
                                    className="mb-3"
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Temporada:", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='opta-temporada-dropdown',
                                    placeholder="Selecciona una temporada...",
                                    className="mb-3"
                                )
                            ], width=6)
                        ]),
                        
                        # Input para jornadas inicial y final
                        dbc.Row([
                            dbc.Col([
                                html.Label("Jornada inicial:", className="fw-bold mb-2"),
                                dbc.Input(
                                    id="opta-jornada-inicial",
                                    type="number",
                                    placeholder="Ej: 1",
                                    min=1,
                                    max=38,
                                    value=1,
                                    className="mb-3"
                                )
                            ], width=3),
                            dbc.Col([
                                html.Label("Jornada final:", className="fw-bold mb-2"),
                                dbc.Input(
                                    id="opta-jornada-final", 
                                    type="number",
                                    placeholder="Ej: 10",
                                    min=1,
                                    max=38,
                                    value=1,
                                    className="mb-3"
                                )
                            ], width=3)
                        ])
                    ])
                ], className="shadow-sm mb-4")
            ])
        ]),
        
        # Botones de acción
        dbc.Row([
            dbc.Col([
                dbc.Button("📈 Actualizar Opta", id="btn-update-opta", color="primary", className="me-2", disabled=True),
                dbc.Button("🔄 Volver", id="btn-back-main", color="secondary")
            ], className="text-center")
        ]),
        
        # NUEVA: Barra de progreso
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.H5("📊 Progreso de Actualización"),
                html.Div(id="progress-info", className="mb-2"),
                dbc.Progress(
                    id="progress-bar",
                    value=0,
                    style={"height": "25px"},
                    className="mb-3"
                ),
                html.Div(id="update-progress", children=[
                    html.P("Configura los parámetros y presiona 'Actualizar Opta' para comenzar...", className="text-muted")
                ], style={
                    "height": "300px",
                    "overflow-y": "auto",
                    "border": "1px solid #dee2e6",
                    "border-radius": "0.375rem",
                    "padding": "1rem",
                    "background-color": "#f8f9fa"
                })
            ])
        ], className="mt-4"),
        
        # Interval para actualizar progreso
        dcc.Interval(
            id='progress-interval',
            interval=1000,  # 1 segundo
            n_intervals=0,
            disabled=True
        ),
        
        # Store para datos de progreso
        dcc.Store(id='progress-store', data={'active': False, 'progress': 0, 'status': '', 'messages': []})
    ], fluid=True)

def crear_grafico_circular(mediacoach_percent, opta_percent):
    """Crea gráfico circular para mostrar porcentajes"""
    fig = go.Figure(data=[go.Pie(
        labels=['MediaCoach', 'Opta'],
        values=[mediacoach_percent, opta_percent],
        hole=.6,  # Donut moderno
        marker=dict(
            colors=['#DC143C', '#1E90FF'],  # Rojo y azul
            line=dict(color='#FFFFFF', width=2)
        ),
        textfont=dict(size=12),
        textposition='outside',
        textinfo='percent',
        hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        showlegend=False,
        height=120,  # Más pequeño
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[
            dict(
                text=f'{int(mediacoach_percent + opta_percent)}%',
                x=0.5, y=0.5,
                font_size=16,
                font_color='#2c3e50',
                showarrow=False
            )
        ]
    )
    
    return fig


# Layout de login
login_layout = html.Div([
    # Fondo animado
    html.Div(className="login-background"),
    html.Div(className="login-bg-gradient"),
    
    # Contenedor del formulario
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Img(
                                src=get_logo_base64("logodatos-villarrealcf.png"), 
                                height="200px", 
                                className="mb-4 d-block mx-auto"
                            ),
                            html.H2("Villarreal CF", className="text-center mb-2 login-title"),
                            html.P("Ingrese sus credenciales", className="text-center text-muted mb-4"),
                        ]),
                        dbc.Input(
                            id="username", 
                            placeholder="Usuario", 
                            type="text", 
                            className="mb-3 modern-input",
                            style={"padding": "12px"}
                        ),
                        dbc.Input(
                            id="password", 
                            placeholder="Contraseña", 
                            type="password", 
                            className="mb-4 modern-input",
                            style={"padding": "12px"}
                        ),
                        dbc.Button(
                            "INICIAR SESIÓN", 
                            id="login-button", 
                            color="primary", 
                            className="w-100 modern-button",
                            style={"padding": "12px", "fontWeight": "bold"}
                        ),
                        html.Div(id="login-error", className="text-danger mt-3 text-center")
                    ])
                ], className="login-card shadow-lg")
            ], width=12, md=6, lg=4)
        ], justify="center", className="vh-100 align-items-center")
    ], fluid=True),
], className="login-container")


def obtener_jornadas_disponibles(liga_filtro=None, temporada_filtro=None, messages=None):
    """Obtiene jornadas filtradas por liga y temporada usando valores normalizados"""
    def add_message(msg):
        if messages is not None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            messages.append({
                'timestamp': timestamp,
                'message': msg,
                'type': 'info'
            })
        print(msg)  # Mantener también el print para terminal
    
    # Reemplaza todos los print() con add_message()
    add_message(f"🔍 Filtrando por: Liga='{liga_filtro}', Temporada='{temporada_filtro}'")

    jornadas_data = {}
    normalize_map = normalizar_valores_filtros()
    
    def normalizar_jornada(jornada):
        if isinstance(jornada, str):
            try:
                cleaned = ''.join(filter(str.isdigit, jornada))
                return int(cleaned) if cleaned else jornada
            except:
                return jornada
        return jornada
    
    def formatear_nombre_jornada(jornada):
        if isinstance(jornada, str):
            cleaned = jornada.lower().replace('j', '').strip()
            try:
                return str(int(cleaned))
            except:
                return cleaned
        return str(jornada)
    
    print(f"🔍 Filtrando por: Liga='{liga_filtro}', Temporada='{temporada_filtro}'")  # DEBUG
    
    # Procesar archivos MediaCoach
    if MEDIACOACH_PATH.exists():
        for file in MEDIACOACH_PATH.glob('*.parquet'):
            try:
                df = pd.read_parquet(file)
                
                # Filtrar por liga - SOLO si hay filtro
                if liga_filtro:
                    if 'Competicion' in df.columns:
                        # Verificar si algún valor normalizado coincide
                        mask = df['Competicion'].apply(
                            lambda x: normalize_map['liga_normalize'].get(x, x) == liga_filtro
                        )
                        df = df[mask]
                        print(f"📄 MediaCoach {file.name}: {len(df)} filas después del filtro de liga")
                
                # Filtrar por temporada - SOLO si hay filtro
                if temporada_filtro:
                    if 'Temporada' in df.columns:
                        mask = df['Temporada'].apply(
                            lambda x: normalize_map['temporada_normalize'].get(x, x) == temporada_filtro
                        )
                        df = df[mask]
                        print(f"📄 MediaCoach {file.name}: {len(df)} filas después del filtro de temporada")
                
                if df.empty:
                    continue
                
                jornada_col = next((col for col in ['week', 'jornada', 'Jornada'] if col in df.columns), None)
                if jornada_col:
                    for jornada in df[jornada_col].unique():
                        if pd.notna(jornada):
                            jornada_norm = normalizar_jornada(jornada)
                            nombre_mostrar = formatear_nombre_jornada(jornada)
                            if jornada_norm not in jornadas_data:
                                jornadas_data[jornada_norm] = {
                                    'mediacoach': 0,
                                    'opta': 0,
                                    'nombre_original': nombre_mostrar
                                }
                            jornadas_data[jornada_norm]['mediacoach'] += len(df[df[jornada_col] == jornada])
            except Exception as e:
                print(f"❌ Error procesando MediaCoach {file}: {e}")
    
    # Procesar archivos Opta
    if OPTA_PATH.exists():
        for file in OPTA_PATH.glob('*.parquet'):
            try:
                df = pd.read_parquet(file)
                
                # Filtrar por liga - SOLO si hay filtro
                if liga_filtro:
                    if 'Competition Name' in df.columns:
                        mask = df['Competition Name'].apply(
                            lambda x: normalize_map['liga_normalize'].get(x, x) == liga_filtro
                        )
                        df = df[mask]
                        print(f"📄 Opta {file.name}: {len(df)} filas después del filtro de liga")
                
                # Filtrar por temporada - SOLO si hay filtro
                if temporada_filtro:
                    if 'Stage ID' in df.columns:
                        mask = df['Stage ID'].apply(
                            lambda x: normalize_map['temporada_normalize'].get(x, x) == temporada_filtro
                        )
                        df = df[mask]
                        print(f"📄 Opta {file.name}: {len(df)} filas después del filtro de temporada")
                
                if df.empty:
                    continue
                
                if 'Week' in df.columns:
                    for jornada in df['Week'].unique():
                        if pd.notna(jornada):
                            jornada_norm = normalizar_jornada(jornada)
                            nombre_mostrar = formatear_nombre_jornada(jornada)
                            if jornada_norm not in jornadas_data:
                                jornadas_data[jornada_norm] = {
                                    'mediacoach': 0,
                                    'opta': 0,
                                    'nombre_original': nombre_mostrar
                                }
                            jornadas_data[jornada_norm]['opta'] += len(df[df['Week'] == jornada])
            except Exception as e:
                print(f"❌ Error procesando Opta {file}: {e}")
    
    print(f"✅ Jornadas encontradas: {list(jornadas_data.keys())}")  # DEBUG
    return jornadas_data

def obtener_valores_filtros():
    """Obtiene valores únicos NORMALIZADOS para los filtros"""
    ligas_raw = set()
    temporadas_raw = set()
    
    # MediaCoach
    if MEDIACOACH_PATH.exists():
        for file in MEDIACOACH_PATH.glob('*.parquet'):
            try:
                df = pd.read_parquet(file)
                if 'Competicion' in df.columns:
                    ligas_raw.update(df['Competicion'].dropna().unique())
                if 'Temporada' in df.columns:
                    temporadas_raw.update(df['Temporada'].dropna().unique())
            except:
                pass
    
    # Opta
    if OPTA_PATH.exists():
        for file in OPTA_PATH.glob('*.parquet'):
            try:
                df = pd.read_parquet(file)
                if 'Competition Name' in df.columns:
                    ligas_raw.update(df['Competition Name'].dropna().unique())
                if 'Stage ID' in df.columns:
                    temporadas_raw.update(df['Stage ID'].dropna().unique())
            except:
                pass
    
    # Normalizar valores
    normalize_map = normalizar_valores_filtros()
    
    ligas_normalizadas = set()
    for liga in ligas_raw:
        if liga in normalize_map['liga_normalize']:
            ligas_normalizadas.add(normalize_map['liga_normalize'][liga])
    
    temporadas_normalizadas = set()
    for temp in temporadas_raw:
        if temp in normalize_map['temporada_normalize']:
            temporadas_normalizadas.add(normalize_map['temporada_normalize'][temp])
    
    return list(ligas_normalizadas), list(temporadas_normalizadas)

# Layout principal
def crear_layout_principal():
    # Obtener valores para filtros
    ligas, temporadas = obtener_valores_filtros()
    
    return dbc.Container([
        # Navbar
        dbc.Navbar([
            dbc.Container([
                dbc.NavbarBrand([
                    html.Img(src=get_logo_base64("logodatos-villarrealcf.png"), height="80px", className="me-2"),
                    "Departamento de Datos"
                ], className="ms-2 fw-bold d-flex align-items-center"),
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("📄 Descargar PDF", id="download-pdf", href="#", className="nav-link-custom")),
                    dbc.NavItem(dbc.NavLink("🔄 Actualizar Datos", id="update-data", href="#", className="nav-link-custom")),
                    dbc.NavItem(dbc.NavLink("🚪 Cerrar Sesión", id="logout-btn", href="#", className="nav-link-custom"))
                ], className="ms-auto", navbar=True)
            ])
        ], color="dark", dark=True, className="mb-4 shadow"),
        
        # NUEVO: Selectores de filtros
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Liga:", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='liga-dropdown',
                                    options=[{'label': liga, 'value': liga} for liga in ligas],
                                    value=ligas[0] if ligas else None,
                                    clearable=False,
                                    className="mb-2"
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Temporada:", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='temporada-dropdown',
                                    options=[{'label': temp, 'value': temp} for temp in temporadas],
                                    value=temporadas[0] if temporadas else None,
                                    clearable=False,
                                    className="mb-2"
                                )
                            ], width=6)
                        ])
                    ])
                ], className="shadow-sm mb-4")
            ], width=8, className="mx-auto")
        ]),
        
        # Grid de jornadas (ahora se actualiza dinámicamente)
        html.Div(id='jornadas-container'),
        
        # Modal para mensajes
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Información")),
            dbc.ModalBody("Esta función se implementará próximamente.", id="modal-body"),
            dbc.ModalFooter(
                dbc.Button("Cerrar", id="close-modal", className="ms-auto", n_clicks=0)
            )
        ], id="modal", is_open=False)
    ], fluid=True)

# Layout de la app
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='login-status', storage_type='session'),
    html.Div(id='page-content')
])

# Callbacks
@app.callback(
    Output('login-status', 'data'),
    Output('login-error', 'children'),
    Input('login-button', 'n_clicks'),
    State('username', 'value'),
    State('password', 'value'),
    prevent_initial_call=True
)
def login(n_clicks, username, password):
    if n_clicks:
        print(f"Intento de login - Usuario: {username}, Pass: {password}")  # <-- Añade esto
        if username == USERNAME and password == PASSWORD:
            return True, ""
        else:
            return False, "Usuario o contraseña incorrectos"
    return dash.no_update, dash.no_update

@app.callback(
    Output('page-content', 'children'),
    Input('login-status', 'data'),
    Input('url', 'pathname')
)
def display_page(login_status, pathname):
    if login_status:
        if pathname == '/actualizar':
            return crear_pagina_actualizacion()
        else:
            return crear_layout_principal()
    else:
        return login_layout

# NUEVO: Callback para actualizar las jornadas cuando cambien los filtros
@app.callback(
    Output('jornadas-container', 'children'),
    [Input('liga-dropdown', 'value'),
     Input('temporada-dropdown', 'value')]
)
def actualizar_jornadas(liga_seleccionada, temporada_seleccionada):
    # Obtener datos filtrados
    jornadas_data = obtener_jornadas_disponibles(liga_seleccionada, temporada_seleccionada)
    
    # Si no hay datos, mostrar una jornada vacía
    if not jornadas_data:
        jornadas_data = {1: {'mediacoach': 0, 'opta': 0, 'nombre_original': '1'}}
    
    jornadas_cards = []
    
    # Ordenar jornadas numéricamente
    for jornada_num in sorted(jornadas_data.keys(), key=lambda x: int(x) if str(x).isdigit() else x, reverse=True):
        datos = jornadas_data[jornada_num]
        total = datos['mediacoach'] + datos['opta']
        
        # Calcular porcentajes
        mediacoach_pct = (datos['mediacoach'] / total * 100) if total > 0 else 0
        opta_pct = (datos['opta'] / total * 100) if total > 0 else 0
        
        # Crear tarjeta para la jornada
        card = dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5(f"Jornada {datos['nombre_original']}", className="text-center mb-3 fw-bold"),
                    dcc.Graph(
                        figure=crear_grafico_circular(mediacoach_pct, opta_pct),
                        config={'displayModeBar': False}
                    ),
                    html.Div([
                        html.Div([
                            html.Img(src=get_logo_base64("mediacoach_logo.png"), height="20px", className="me-2"),
                            html.Span(f"{mediacoach_pct:.1f}%", className="fw-bold", style={"color": "#DC143C"})
                        ], className="d-flex align-items-center justify-content-center mb-2"),
                        html.Div([
                            html.Img(src=get_logo_base64("opta_logo.png"), height="20px", className="me-2"),
                            html.Span(f"{opta_pct:.1f}%", className="fw-bold", style={"color": "#1E90FF"})
                        ], className="d-flex align-items-center justify-content-center")
                    ])
                ])
            ], className="shadow-sm h-100 jornada-card")
        ], width=6, md=3, lg=2, className="mb-4")
        
        jornadas_cards.append(card)
    
    return dbc.Row(jornadas_cards)

@app.callback(
    Output('url', 'pathname'),
    Input({'type': 'dynamic-logout', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def logout(n_clicks):
    if any(n_clicks):
        return '/'
    return dash.no_update

@app.callback(
    Output('modal', 'is_open'),
    Output('modal-body', 'children'),
    Input({'type': 'dynamic-action', 'index': ALL}, 'n_clicks'),
    Input('close-modal', 'n_clicks'),
    State('modal', 'is_open'),
    prevent_initial_call=True
)
def toggle_modal(action_clicks, close_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, ""
    
    trigger_id = ctx.triggered[0]['prop_id']
    
    if 'close-modal' in trigger_id:
        return False, ""
    
    # Determinar qué botón se presionó
    if any(action_clicks):
        return True, "Esta función se implementará próximamente."
    
    return is_open, ""

# Callback para manejar clicks del navbar después de que se carga
app.clientside_callback(
    """
    function(pdf_clicks, update_clicks) {
        if (pdf_clicks) {
            document.getElementById('modal').click();
        }
        if (update_clicks) {
            window.location.href = '/actualizar';
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('modal', 'is_open', allow_duplicate=True),
    Input('download-pdf', 'n_clicks'),
    Input('update-data', 'n_clicks'),
    prevent_initial_call=True
)
app.clientside_callback(
    """
    function(pdf_clicks, update_clicks) {
        if (pdf_clicks) {
            document.getElementById('modal').click();
            return window.dash_clientside.no_update;
        }
        if (update_clicks) {
            window.location.href = '/actualizar';
            return '/actualizar';
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('url', 'pathname', allow_duplicate=True),
    Input('download-pdf', 'n_clicks'),
    Input('update-data', 'n_clicks'),
    prevent_initial_call=True
)

# Callback para el botón "Volver"
@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    Input('btn-back-main', 'n_clicks'),
    prevent_initial_call=True
)
def volver_principal(n_clicks):
    if n_clicks:
        return '/'
    return dash.no_update


# Callback para cargar las opciones de liga y temporada
@app.callback(
    Output('opta-liga-dropdown', 'options'),
    Input('url', 'pathname')
)
def cargar_ligas_opta(pathname):
    if pathname == '/actualizar':
        try:
            competitions = actualizar_datos.get_all_competitions_and_stages()
            liga_options = [
                {'label': comp_info['name'], 'value': comp_id} 
                for comp_id, comp_info in competitions.items()
            ]
            return liga_options
        except Exception as e:
            print(f"Error: {e}")
            return []
    return []

# NUEVO callback para temporadas
@app.callback(
    Output('opta-temporada-dropdown', 'options'),
    Input('opta-liga-dropdown', 'value')
)
def cargar_temporadas_opta(liga_seleccionada):
    if liga_seleccionada:
        try:
            competitions = actualizar_datos.get_all_competitions_and_stages()
            if liga_seleccionada in competitions:
                stages = competitions[liga_seleccionada]['stages']
                temporada_options = [
                    {'label': f"{stage_info['season']} - {stage_info['name']}", 'value': stage_id}
                    for stage_id, stage_info in stages.items()
                ]
                return temporada_options
        except Exception as e:
            print(f"Error: {e}")
            return []
    return []

# Callback para habilitar/deshabilitar botón
@app.callback(
    Output('btn-update-opta', 'disabled'),
    [Input('opta-liga-dropdown', 'value'),
     Input('opta-temporada-dropdown', 'value'),
     Input('opta-jornada-inicial', 'value'),
     Input('opta-jornada-final', 'value')]
)
def habilitar_boton_opta(liga, temporada, jornada_inicial, jornada_final):
    if liga and temporada and jornada_inicial and jornada_final and jornada_inicial <= jornada_final:
        return False
    return True

@app.callback(
    Output('login-status', 'data', allow_duplicate=True),
    Input('logout-btn', 'n_clicks'),
    prevent_initial_call=True
)
def logout_user(n_clicks):
    if n_clicks:
        return False  # Esto limpiará la sesión
    return dash.no_update

@app.callback(
    [Output('progress-interval', 'disabled'),
     Output('btn-update-opta', 'disabled', allow_duplicate=True)],
    Input('btn-update-opta', 'n_clicks'),
    [State('opta-liga-dropdown', 'value'),
     State('opta-temporada-dropdown', 'value'),
     State('opta-jornada-inicial', 'value'),
     State('opta-jornada-final', 'value')],
    prevent_initial_call=True
)
def iniciar_actualizacion_opta(n_clicks, liga_seleccionada, temporada_seleccionada, jornada_inicial, jornada_final):
    if n_clicks:
        global progress_data
        
        # Resetear progreso
        progress_data = {
            'active': True,
            'progress': 0,
            'status': 'Iniciando...',
            'messages': []
        }
        
        # Función para actualizar progreso
        def progress_callback(progress, status, messages):
            global progress_data
            progress_data['progress'] = progress
            progress_data['status'] = status
            progress_data['messages'] = messages
        
        # Iniciar hilo
        thread = threading.Thread(target=lambda: actualizar_datos.update_opta_data_web_ranges(
            competition_id=liga_seleccionada,
            stage_id=temporada_seleccionada,
            start_week=jornada_inicial,      # ← Ahora usa jornada_inicial
            end_week=jornada_final,         # ← Ahora usa jornada_final
            progress_callback=progress_callback
        ))
        thread.start()  # ← Faltaba iniciar el hilo
        
        # Habilitar el interval y deshabilitar el botón
        return False, True
    
    return dash.no_update, dash.no_update

@app.callback(
    [Output('progress-bar', 'value'),
     Output('progress-info', 'children'),
     Output('update-progress', 'children'),
     Output('progress-bar', 'color'),
     Output('progress-interval', 'disabled', allow_duplicate=True),
     Output('btn-update-opta', 'disabled', allow_duplicate=True)],
    Input('progress-interval', 'n_intervals'),
    prevent_initial_call=True
)
def actualizar_progreso(n_intervals):
    global progress_data
    
    # Convertir mensajes a elementos HTML
    progress_elements = []
    for msg in progress_data['messages']:
        css_class = "text-info"
        if msg['type'] == 'error':
            css_class = "text-danger"
        elif msg['type'] == 'success':
            css_class = "text-success"
        
        progress_elements.append(
            html.P(f"[{msg['timestamp']}] {msg['message']}", className=css_class)
        )
    
    # Determinar color de la barra
    if progress_data['progress'] == 100:
        bar_color = "success"
    elif progress_data['progress'] > 0:
        bar_color = "info"
    else:
        bar_color = "secondary"
    
    # Info del progreso
    info_text = f"Estado: {progress_data['status']} ({progress_data['progress']}%)"
    
    # Si el proceso terminó, deshabilitar interval y habilitar botón
    if not progress_data['active']:
        return (
            progress_data['progress'],
            html.P(info_text, className="fw-bold"),
            progress_elements,
            bar_color,
            True,  # Deshabilitar interval
            False  # Habilitar botón
        )
    
    # Proceso activo
    return (
        progress_data['progress'],
        html.P(info_text, className="fw-bold"),
        progress_elements,
        bar_color,
        False,  # Mantener interval activo
        True   # Mantener botón deshabilitado
    )

if __name__ == '__main__':
    app.run_server( host='0.0.0.0', port=8050, debug=True)
