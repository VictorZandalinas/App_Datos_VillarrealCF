# Dashboard de Análisis de Datos - MediaCoach y Opta

## Estructura del Proyecto

```
├── app.py                      # Aplicación principal
├── requirements.txt            # Dependencias
├── .gitignore                 # Archivos ignorados por Git
├── assets/                    # Archivos estáticos
│   ├── styles.css            # Estilos CSS
│   ├── mediacoach_logo.png  # Logo MediaCoach
│   ├── opta_logo.png        # Logo Opta
│   └── [otros_recursos].png  # Escudos y fondos
├── datos_mediacoach_parquet/  # Datos MediaCoach
│   └── *.parquet
└── datos_opta_parquet/        # Datos Opta
    └── *.parquet
```

## Instalación Local

1. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Crear estructura de carpetas:
```bash
mkdir assets datos_mediacoach_parquet datos_opta_parquet
```

4. Ejecutar la aplicación:
```bash
python app.py
```

## Configuración para Render

1. Crear archivo `render.yaml` en la raíz del proyecto:

```yaml
services:
  - type: web
    name: dashboard-analisis
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "gunicorn app:server"
    envVars:
      - key: APP_USERNAME
        generateValue: false
      - key: APP_PASSWORD
        generateValue: false
```

2. Variables de entorno en Render:
   - `APP_USERNAME`: Usuario para login
   - `APP_PASSWORD`: Contraseña para login

## Actualización de Datos

Los archivos parquet pueden actualizarse de dos formas:

1. **Manual**: Subir nuevos archivos a las carpetas correspondientes
2. **Automática**: (Pendiente de implementar) Botón "Actualizar Datos" en la interfaz

## Notas Importantes

- Los archivos parquet deben contener una columna llamada `week`, `jornada` o `Jornada`
- Los logos deben estar en formato PNG en la carpeta `assets/`
- La aplicación cuenta automáticamente los archivos por jornada
- El login usa sesiones, así que se mantiene activo durante la navegación
- El login cuenta con un fondo animado con gradientes en tonos amarillos y azules

## Próximas Funcionalidades

- [ ] Descarga de informes PDF
- [ ] Actualización automática de datos
- [ ] Filtros por equipo
- [ ] Visualizaciones detalladas por jornada