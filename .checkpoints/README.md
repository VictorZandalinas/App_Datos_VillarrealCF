# Sistema de Checkpoints para Recuperación de Actualizaciones

## ¿Qué es?

El sistema de checkpoints permite **reanudar automáticamente** las actualizaciones de datos (Opta, MediaCoach, Sportian) si el proceso se interrumpe inesperadamente (corte de luz, reinicio del servicio, timeout, etc.).

## ¿Cómo funciona?

1. **Al iniciar** una actualización, se crea un archivo de checkpoint en `.checkpoints/`
2. **Después de cada fase crítica**, el checkpoint se actualiza en disco inmediatamente
3. **Si el proceso se interrumpe**, al reiniciar se detecta el checkpoint pendiente
4. **El sistema reanuda** exactamente desde la fase donde se quedó

## Archivos de Checkpoint

```
.checkpoints/
├── opta_checkpoint.json       # Checkpoint de actualización de Opta
├── mediacoach_checkpoint.json # Checkpoint de actualización de MediaCoach
└── sportian_checkpoint.json   # Checkpoint de actualización de Sportian
```

## Fases Protegidas por Checkpoint

### Opta

| Fase | Descripción | Punto de Guardado |
|------|-------------|-------------------|
| `fase_1_competicion` | Detección del tipo de competición | Tras analizar competición |
| `fase_2_partidos` | Obtención de lista de partidos | Tras descargar partidos |
| `fase_3_verificacion` | Verificación de datos existentes | Tras verificar qué falta |
| `fase_4_descarga` | Descarga granular de datos | **Después de CADA partido** |
| `fase_5_guardado_abp` | Guardado de parquet + ABP | Tras guardar datos |
| `fase_6_carpetas` | Actualizar carpetas de equipos | Tras actualizar carpetas |
| `fase_7_git` | Sincronización con GitHub | Al finalizar todo |

### MediaCoach

| Fase | Descripción | Punto de Guardado |
|------|-------------|-------------------|
| `fase_1_descarga` | Ejecución de descarga_completa.py | Tras completar script |
| `fase_2_carpetas` | Cálculo delta + carpetas equipos | Tras actualizar carpetas |
| `fase_3_git` | Sincronización con GitHub | Al finalizar todo |

### Sportian

| Fase | Descripción | Punto de Guardado |
|------|-------------|-------------------|
| `fase_1_csv` | Decodificar y guardar CSV | Tras guardar archivo |
| `fase_2_procesar` | Procesar dataset CSV → Parquet | Tras crear parquet |
| `fase_3_carpetas` | Cálculo delta + carpetas equipos | Tras actualizar carpetas |
| `fase_4_git` | Sincronización con GitHub | Al finalizar todo |

## Comportamiento Automático

### Detección de Interrupción

Cuando inicias una actualización y existe un checkpoint incompleto:

```
📍 SE DETECTÓ UNA INTERRUPCIÓN PREVIA DE OPTA
   📁 Fuente: Opta
   🕐 Iniciado: 2026-05-06T16:25:00
   👉 Reanudando desde el último checkpoint...
```

### Reanudación Inteligente

El sistema **NO repite las fases ya completadas**. Por ejemplo, si se interrumpió en mitad de la descarga de partidos:

- ✅ Fases 1-3: Se saltan (ya completadas)
- 🔄 Fase 4: Reanuda desde el primer partido NO completado
- ⏳ Fases 5-7: Se ejecutan normalmente al finalizar

### Limpieza Automática

Cuando una actualización se completa con éxito:
- El checkpoint se elimina automáticamente
- El siguiente inicio será "limpio"

## Comandos de Gestión

### Ver Checkpoints Pendientes

En el menú principal, si hay checkpoints pendientes, se muestra automáticamente:

```
⚠️  SE DETECTARON ACTUALIZACIONES INTERRUMPIDAS:
   • Opta - Iniciado: 2026-05-06T16:25:00
     Fases completadas: fase_1_competicion, fase_2_partidos
```

### Limpiar Checkpoints

**Opción 5 del menú principal:**

```
🗑️  Opciones de limpieza:
   1. Limpiar checkpoint de Opta
   2. Limpiar checkpoint de MediaCoach
   3. Limpiar checkpoint de Sportian
   4. Limpiar TODOS los checkpoints
```

### Forzar Reinicio desde la Web

La interfaz web acepta un parámetro `force_restart=True` para ignorar checkpoints:

```python
update_opta_data_web(comp_id, stage_id, 1, 34, force_restart=True)
```

## Escenarios de Uso

### Escenario 1: Corte de Luz a Mitad de Descarga

**Situación:** Descargando Jornada 34, se va la luz en el partido 7 de 10.

**Recuperación:**
1. Reiniciar el servidor
2. Ejecutar actualización desde web o CLI
3. El sistema detecta checkpoint incompleto
4. **Reanuda desde el partido 7** (los partidos 1-6 ya están guardados)

### Escenario 2: Servicio Detenido Durante ABP

**Situación:** El servicio systemd se detiene justo después de guardar `match_events.parquet` pero antes de procesar ABP.

**Recuperación:**
1. El servicio se reinicia automáticamente
2. El checkpoint indica que `fase_4_descarga` está completa
3. **Reanuda desde `fase_5_guardado_abp`**
4. Se procesa `update_abp_events_standalone()` sin redescargar nada

### Escenario 3: Error de API a Mitad de Lote

**Situación:** La API de Opta devuelve error en el partido 5, pero el script continúa.

**Comportamiento:**
- El checkpoint guarda el estado tras CADA partido completado
- Los partidos con error se saltan (se registran en el log)
- La siguiente ejecución reanuda desde donde se quedó

## Consideraciones Técnicas

### Persistencia Inmediata

Cada llamada a `checkpoint.mark_phase()` escribe **inmediatamente** en disco:

```python
def _save(self):
    self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(self.data, f, indent=2, default=str)
```

### Atomicidad

El checkpoint se guarda **después** de completar la operación, no antes. Esto significa:
- ✅ Si falla a mitad de una fase, la fase se repite completa
- ⚠️ No hay recuperación "a mitad de una operación"

### Concurrencia

Los checkpoints son **por fuente de datos** (Opta, MediaCoach, Sportian):
- Puedes actualizar Opta y MediaCoach simultáneamente
- Cada una tiene su propio archivo de checkpoint

## Archivos de Log

Los checkpoints se complementan con `data_update.log`:

```bash
tail -f data_update.log  # Ver logs en tiempo real
grep "Checkpoint" data_update.log  # Ver actividad de checkpoints
```

## Troubleshooting

### "Checkpoint corrupto"

Si el JSON del checkpoint se corrompe:
```bash
rm .checkpoints/opta_checkpoint.json  # Eliminar checkpoint corrupto
```

### "Reanuda pero quiero empezar desde cero"

Usa la opción 5 del menú para limpiar, o llama con `force_restart=True`:
```python
update_opta_data_web(..., force_restart=True)
```

### "No detecta interrupción"

Verifica que:
1. El archivo de checkpoint existe en `.checkpoints/`
2. El campo `"completed": false` está presente
3. Los logs muestran el checkpoint cargado

## Ejemplo de JSON de Checkpoint

```json
{
  "started_at": "2026-05-06T16:25:00.123456",
  "source": "Opta",
  "phases": {
    "fase_1_competicion": {
      "completed_at": "2026-05-06T16:25:15.789012",
      "metadata": {
        "is_cup_format": false,
        "season": "2025/2026",
        "comp_name": "primera división"
      }
    },
    "fase_2_partidos": {
      "completed_at": "2026-05-06T16:26:00.123456",
      "metadata": {
        "match_ids": ["match1", "match2", ...]
      }
    },
    "fase_4_descarga": {
      "completed_at": "2026-05-06T16:45:00.123456",
      "metadata": {
        "processed_match_ids": ["match1", "match2", "match3"]
      }
    }
  },
  "completed": false
}
```

---

**Implementado:** Mayo 2026  
**Versión:** 1.0  
**Archivos modificados:** `actualizar_datos.py`
