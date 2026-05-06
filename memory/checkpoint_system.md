---
name: Sistema de Checkpoints para Recuperación de Actualizaciones
description: Implementación de checkpoint manager para reanudar actualizaciones de Opta, MediaCoach y Sportian tras interrupciones
type: project
---

**Decisión:** Implementar sistema de checkpoints persistente para las tres fuentes de datos (Opta, MediaCoach, Sportian).

**Why:** El servicio se detuvo durante la descarga de la Jornada 34 (SIGTERM a las 16:30:40), justo después de guardar `match_events.parquet` pero ANTES de ejecutar `update_abp_events_standalone()`. Resultado: `abp_events.parquet` no tiene los datos de la Jornada 34 aunque están en `match_events.parquet`.

**How to apply:**
- Cada actualización (Opta/MediaCoach/Sportian) tiene su propio archivo de checkpoint en `.checkpoints/`
- Checkpoints se guardan inmediatamente tras cada fase crítica (JSON en disco)
- Al detectar interrupción, el sistema reanuda desde la última fase completada
- Fases Opta: 1-competicion, 2-partidos, 3-verificacion, 4-descarga (por partido), 5-guardado_abp, 6-carpetas, 7-git
- Opción en menú CLI para limpiar checkpoints manuales
- Web: parámetro `force_restart=True` para ignorar checkpoint y empezar desde cero
- Documentación completa en `.checkpoints/README.md`
