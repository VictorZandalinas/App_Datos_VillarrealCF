#!/bin/bash
###############################################################################
# AUTO-COMMIT Y PUSH DE DATOS ACTUALIZADOS
# Este script se ejecuta automáticamente después de actualizar datos en el servidor
###############################################################################

# Configuración
REPO_DIR="/root/App_Datos_VillarrealCF"
LOG_FILE="$REPO_DIR/auto_commit.log"
MAX_LOG_SIZE=10485760  # 10MB

# Función de logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Rotar log si es muy grande
if [ -f "$LOG_FILE" ] && [ $(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE") -gt $MAX_LOG_SIZE ]; then
    mv "$LOG_FILE" "$LOG_FILE.old"
    log "Log rotado"
fi

log "========================================="
log "Iniciando auto-commit de datos"
log "========================================="

# Cambiar al directorio del repositorio
cd "$REPO_DIR" || {
    log "ERROR: No se pudo acceder a $REPO_DIR"
    exit 1
}

# Configurar git si no está configurado
git config user.email "servidor@villarreal.com" 2>/dev/null || true
git config user.name "Servidor Villarreal" 2>/dev/null || true

# Verificar si hay cambios
if git diff --quiet && git diff --cached --quiet; then
    log "No hay cambios para commitear"
    exit 0
fi

# Mostrar archivos modificados
log "Archivos modificados:"
git status --short | tee -a "$LOG_FILE"

# Añadir solo archivos de datos (no código)
log "Añadiendo archivos de datos..."
git add extraccion_opta/datos_opta_parquet/*.parquet 2>/dev/null || true
git add extraccion_mediacoach/datos_mediacoach_parquet/*.parquet 2>/dev/null || true
git add extraccion_mediacoach/data/*.parquet 2>/dev/null || true
git add extraccion_sportian/*.parquet 2>/dev/null || true
git add extraccion_sportian/datos_sportian_parquet/*.parquet 2>/dev/null || true
git add data_update.log 2>/dev/null || true
git add informes_generados/*.pdf 2>/dev/null || true

# Verificar si hay algo en staging
if git diff --cached --quiet; then
    log "No hay cambios en archivos de datos"
    exit 0
fi

# Crear mensaje de commit descriptivo
TIMESTAMP=$(date '+%a %d %b %Y %H:%M:%S %Z')
CHANGED_FILES=$(git diff --cached --name-only | wc -l | tr -d ' ')

COMMIT_MSG="Actualización automática $TIMESTAMP

Archivos actualizados: $CHANGED_FILES
- Datos Opta
- Datos MediaCoach
- Datos Sportian

[skip ci]"

# Hacer commit
log "Creando commit..."
if git commit -m "$COMMIT_MSG"; then
    log "✅ Commit creado exitosamente"

    # Push a GitHub
    log "Pushing a GitHub..."
    if git push origin main; then
        log "✅ Push completado exitosamente"
        log "Cambios subidos a GitHub"
    else
        log "❌ ERROR: Push falló"
        log "Intentando pull + rebase..."

        # Intentar resolver conflictos
        git pull --rebase origin main
        if [ $? -eq 0 ]; then
            log "Rebase exitoso, reintentando push..."
            if git push origin main; then
                log "✅ Push completado tras rebase"
            else
                log "❌ ERROR: Push falló después de rebase"
                exit 1
            fi
        else
            log "❌ ERROR: Rebase falló - revisar manualmente"
            exit 1
        fi
    fi
else
    log "❌ ERROR: Commit falló"
    exit 1
fi

log "========================================="
log "Auto-commit completado"
log "========================================="
