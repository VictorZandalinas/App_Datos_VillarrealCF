import json
import pandas as pd
import os
import glob
from datetime import datetime
from collections import defaultdict

class InspectorOptaArchivos:
    def __init__(self, carpeta="."):
        self.carpeta = carpeta
        self.archivos_json = []
        self.archivos_parquet = []
        self.tipos_json = {}
        self.estructuras_parquet = {}
        self.mapeo_json_parquet = defaultdict(list)
        
    def escanear_archivos(self):
        """Escanea la carpeta y encuentra todos los archivos JSON y Parquet"""
        print(f"🔍 Escaneando archivos en: {self.carpeta}")
        
        # Buscar archivos JSON
        self.archivos_json = glob.glob(os.path.join(self.carpeta, "*.json"))
        print(f"   📄 Archivos JSON encontrados: {len(self.archivos_json)}")
        
        # Buscar archivos Parquet
        self.archivos_parquet = glob.glob(os.path.join(self.carpeta, "*.parquet"))
        print(f"   💾 Archivos Parquet encontrados: {len(self.archivos_parquet)}")
        
        if not self.archivos_json and not self.archivos_parquet:
            print("❌ No se encontraron archivos JSON ni Parquet")
            return False
        
        return True
    
    def detectar_tipo_json(self, archivo_json):
        """Detecta si un JSON es MA2, MA3, MA12 o desconocido - MEJORADO"""
        try:
            with open(archivo_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            live_data = data.get('liveData', {})
            match_info = data.get('matchInfo', {})
            
            # Verificar que es un archivo válido de Opta
            if not match_info.get('id'):
                return 'INVALID'
            
            # TIPO 1: MA2 Completo (lineUp + posibles eventos)
            if 'lineUp' in live_data:
                # Verificar si tiene eventos xG (MA12)
                if 'event' in live_data:
                    eventos = live_data.get('event', [])
                    tiene_xg = any(
                        any(q.get('qualifierId') in [321, 322] for q in evento.get('qualifier', []))
                        for evento in eventos
                    )
                    if tiene_xg:
                        return 'MA12'  # Match xG (tiene lineUp + eventos con xG)
                return 'MA2_COMPLETO'  # Match Stats completo (tiene lineUp)
            
            # TIPO 2: MA3 (solo eventos detallados)
            elif 'event' in live_data:
                eventos = live_data.get('event', [])
                # Verificar si tiene xG
                tiene_xg = any(
                    any(q.get('qualifierId') in [321, 322] for q in evento.get('qualifier', []))
                    for evento in eventos
                )
                if tiene_xg:
                    return 'MA12_EVENTOS'  # Solo eventos xG
                return 'MA3'  # Match Events normales
            
            # TIPO 3: MA2 Simplificado (goal, card, substitute pero sin lineUp)
            elif any(key in live_data for key in ['goal', 'card', 'substitute']):
                return 'MA2_SIMPLE'  # Match events básicos (goals, cards, subs)
            
            # TIPO 4: Archivo con solo matchDetails
            elif 'matchDetails' in live_data:
                return 'MA1_DETAILS'  # Solo detalles del partido
            
            else:
                # Debug: mostrar qué contiene liveData
                claves = list(live_data.keys()) if live_data else []
                return f'UNKNOWN_({",".join(claves[:3])})'
                
        except Exception as e:
            print(f"   ❌ Error leyendo {archivo_json}: {e}")
            return 'ERROR'
    
    def analizar_estructura_json(self, archivo_json, tipo_json):
        """Analiza la estructura detallada de un archivo JSON"""
        try:
            with open(archivo_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            match_info = data.get('matchInfo', {})
            live_data = data.get('liveData', {})
            
            estructura = {
                'archivo': os.path.basename(archivo_json),
                'tipo': tipo_json,
                'match_id': match_info.get('id', 'N/A'),
                'competition': match_info.get('competition', {}).get('name', 'N/A'),
                'stage': match_info.get('stage', {}).get('name', 'N/A'),
                'week': match_info.get('week', 'N/A'),
                'tiene_lineUp': 'lineUp' in live_data,
                'tiene_events': 'event' in live_data,
                'claves_liveData': list(live_data.keys())
            }
            
            # Analizar según tipo
            if tipo_json in ['MA2_COMPLETO', 'MA12']:
                line_ups = live_data.get('lineUp', [])
                if line_ups:
                    primer_lineup = line_ups[0]
                    estructura.update({
                        'num_teams': len(line_ups),
                        'tiene_players': 'player' in primer_lineup,
                        'tiene_team_stats': 'stat' in primer_lineup,
                        'tiene_team_officials': 'teamOfficial' in primer_lineup,
                        'num_players_primer_equipo': len(primer_lineup.get('player', [])),
                        'stats_disponibles': []
                    })
                    
                    # Obtener tipos de stats disponibles
                    for player in primer_lineup.get('player', [])[:3]:  # Solo primeros 3 jugadores
                        for stat in player.get('stat', []):
                            if stat.get('type') not in estructura['stats_disponibles']:
                                estructura['stats_disponibles'].append(stat.get('type'))
                    
                    estructura['num_stats_tipos'] = len(estructura['stats_disponibles'])
            
            # Analizar MA2_SIMPLE (goals, cards, substitutes)
            if tipo_json == 'MA2_SIMPLE':
                goals = live_data.get('goal', [])
                cards = live_data.get('card', [])
                substitutes = live_data.get('substitute', [])
                
                estructura.update({
                    'num_goals': len(goals),
                    'num_cards': len(cards),
                    'num_substitutes': len(substitutes),
                    'total_eventos_simples': len(goals) + len(cards) + len(substitutes),
                    'tipos_eventos_simples': []
                })
                
                if goals:
                    estructura['tipos_eventos_simples'].append(f"goals({len(goals)})")
                if cards:
                    estructura['tipos_eventos_simples'].append(f"cards({len(cards)})")
                if substitutes:
                    estructura['tipos_eventos_simples'].append(f"subs({len(substitutes)})")
            
            if tipo_json in ['MA3', 'MA12', 'MA12_EVENTOS']:
                events = live_data.get('event', [])
                if events:
                    # Analizar tipos de eventos
                    tipos_eventos = set()
                    qualifier_ids = set()
                    
                    for event in events[:50]:  # Solo primeros 50 eventos para análisis
                        if 'typeId' in event:
                            tipos_eventos.add(event['typeId'])
                        for qualifier in event.get('qualifier', []):
                            qualifier_ids.add(qualifier.get('qualifierId'))
                    
                    estructura.update({
                        'num_events': len(events),
                        'tipos_eventos_unicos': sorted(list(tipos_eventos)),
                        'num_tipos_eventos': len(tipos_eventos),
                        'qualifier_ids_unicos': sorted(list(qualifier_ids)),
                        'num_qualifiers': len(qualifier_ids)
                    })
                    
                    # Verificar si tiene xG
                    tiene_xg = 321 in qualifier_ids or 322 in qualifier_ids
                    estructura['tiene_xG_qualifiers'] = tiene_xg
            
            return estructura
            
        except Exception as e:
            print(f"   ❌ Error analizando estructura de {archivo_json}: {e}")
            return {'archivo': os.path.basename(archivo_json), 'tipo': 'ERROR', 'error': str(e)}
    
    def analizar_estructura_parquet(self, archivo_parquet):
        """Analiza la estructura de un archivo Parquet"""
        try:
            df = pd.read_parquet(archivo_parquet)
            
            estructura = {
                'archivo': os.path.basename(archivo_parquet),
                'num_filas': len(df),
                'num_columnas': len(df.columns),
                'columnas': list(df.columns),
                'tipos_datos': dict(df.dtypes.astype(str)),
                'tiene_match_id': 'Match ID' in df.columns,
                'matches_unicos': df['Match ID'].nunique() if 'Match ID' in df.columns else 'N/A',
                'memoria_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            # Identificar tipo de parquet por contenido
            nombre_archivo = os.path.basename(archivo_parquet).lower()
            if 'player_stats' in nombre_archivo:
                estructura['tipo_parquet'] = 'player_stats'
                estructura['players_unicos'] = df['Player ID'].nunique() if 'Player ID' in df.columns else 'N/A'
            elif 'team_stats' in nombre_archivo:
                estructura['tipo_parquet'] = 'team_stats'
                estructura['teams_unicos'] = df['Team ID'].nunique() if 'Team ID' in df.columns else 'N/A'
            elif 'xg_events' in nombre_archivo:
                estructura['tipo_parquet'] = 'xg_events'
                estructura['events_unicos'] = df['EventId'].nunique() if 'EventId' in df.columns else 'N/A'
            elif 'match_events' in nombre_archivo:
                estructura['tipo_parquet'] = 'match_events'
                estructura['events_unicos'] = df['EventId'].nunique() if 'EventId' in df.columns else 'N/A'
            elif 'team_officials' in nombre_archivo:
                estructura['tipo_parquet'] = 'team_officials'
                estructura['officials_unicos'] = df['Official ID'].nunique() if 'Official ID' in df.columns else 'N/A'
            elif 'player_xg' in nombre_archivo:
                estructura['tipo_parquet'] = 'player_xg_stats'
                estructura['players_unicos'] = df['Player ID'].nunique() if 'Player ID' in df.columns else 'N/A'
            else:
                estructura['tipo_parquet'] = 'desconocido'
            
            # Buscar columnas específicas
            estructura['columnas_qualifier'] = [col for col in df.columns if col.startswith('qualifier')]
            estructura['num_qualifiers'] = len(estructura['columnas_qualifier'])
            
            # Si es un DataFrame vacío
            if len(df) == 0:
                estructura['estado'] = 'vacio'
            else:
                estructura['estado'] = 'con_datos'
                # Muestra de datos
                estructura['sample_match_ids'] = df['Match ID'].head(3).tolist() if 'Match ID' in df.columns else []
            
            return estructura
            
        except Exception as e:
            print(f"   ❌ Error analizando {archivo_parquet}: {e}")
            return {'archivo': os.path.basename(archivo_parquet), 'error': str(e)}
    
    def crear_mapeo_json_parquet(self):
        """Crea el mapeo de qué JSON debe ir a qué Parquet - MEJORADO"""
        print("\n🗺️ CREANDO MAPEO JSON → PARQUET:")
        
        mapeo = {
            'MA2_COMPLETO': ['player_stats.parquet', 'team_stats.parquet', 'team_officials.parquet'],
            'MA2_SIMPLE': ['abp_events.parquet'],  # Solo eventos básicos (goals, cards, subs)
            'MA3': ['abp_events.parquet'],
            'MA12': ['player_xg_stats.parquet', 'xg_events.parquet', 'team_officials.parquet'],
            'MA12_EVENTOS': ['xg_events.parquet'],
            'MA1_DETAILS': []  # Solo detalles, no genera parquets
        }
        
        for json_file in self.archivos_json:
            tipo = self.tipos_json.get(json_file, 'UNKNOWN')
            nombre_json = os.path.basename(json_file)
            
            print(f"\n📄 {nombre_json} ({tipo}):")
            
            if tipo in mapeo:
                parquets_destino = mapeo[tipo]
                for parquet in parquets_destino:
                    parquet_path = os.path.join(self.carpeta, parquet)
                    existe = os.path.exists(parquet_path)
                    estado = "✅ existe" if existe else "❌ no existe"
                    print(f"   → {parquet} {estado}")
                    
                    self.mapeo_json_parquet[json_file].append({
                        'parquet_destino': parquet,
                        'existe': existe,
                        'path_completo': parquet_path
                    })
            else:
                print(f"   ⚠️ Tipo {tipo} no reconocido - no se puede mapear")
    
    def generar_informe_completo(self):
        """Genera un informe completo de todos los archivos"""
        print(f"\n{'='*80}")
        print(f"📊 INFORME COMPLETO DE ARCHIVOS OPTA")
        print(f"📅 Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Carpeta: {os.path.abspath(self.carpeta)}")
        print(f"{'='*80}")
        
        # Resumen de archivos JSON
        if self.archivos_json:
            print(f"\n📄 ARCHIVOS JSON ({len(self.archivos_json)}):")
            tipos_count = defaultdict(int)
            
            for json_file in self.archivos_json:
                tipo = self.tipos_json.get(json_file, 'UNKNOWN')
                tipos_count[tipo] += 1
                estructura = self.analizar_estructura_json(json_file, tipo)
                
                print(f"\n   🔸 {estructura['archivo']} ({estructura['tipo']})")
                print(f"      Match ID: {estructura['match_id']}")
                print(f"      Competición: {estructura['competition']}")
                print(f"      Stage: {estructura['stage']}")
                print(f"      Jornada: {estructura['week']}")
                
                if estructura['tipo'] in ['MA2', 'MA12']:
                    print(f"      Teams: {estructura.get('num_teams', 'N/A')}")
                    print(f"      Players (primer equipo): {estructura.get('num_players_primer_equipo', 'N/A')}")
                    print(f"      Tipos de stats: {estructura.get('num_stats_tipos', 'N/A')}")
                
                if estructura['tipo'] in ['MA3', 'MA12']:
                    print(f"      Eventos: {estructura.get('num_events', 'N/A')}")
                    print(f"      Tipos de eventos: {estructura.get('num_tipos_eventos', 'N/A')}")
                    print(f"      Qualifiers: {estructura.get('num_qualifiers', 'N/A')}")
                    if estructura.get('tiene_xG_qualifiers'):
                        print(f"      🎯 Contiene datos xG")
            
            print(f"\n📊 RESUMEN POR TIPO:")
            for tipo, count in tipos_count.items():
                print(f"      {tipo}: {count} archivos")
        
        # Resumen de archivos Parquet
        if self.archivos_parquet:
            print(f"\n💾 ARCHIVOS PARQUET ({len(self.archivos_parquet)}):")
            
            for parquet_file in self.archivos_parquet:
                estructura = self.analizar_estructura_parquet(parquet_file)
                
                print(f"\n   🔸 {estructura['archivo']} ({estructura.get('tipo_parquet', 'desconocido')})")
                print(f"      Estado: {estructura.get('estado', 'desconocido')}")
                print(f"      Filas: {estructura['num_filas']:,}")
                print(f"      Columnas: {estructura['num_columnas']}")
                print(f"      Matches únicos: {estructura.get('matches_unicos', 'N/A')}")
                print(f"      Memoria: {estructura.get('memoria_mb', 0):.2f} MB")
                
                if estructura.get('num_qualifiers', 0) > 0:
                    print(f"      Qualifiers: {estructura['num_qualifiers']}")
                
                if estructura.get('players_unicos'):
                    print(f"      Players únicos: {estructura['players_unicos']}")
                if estructura.get('teams_unicos'):
                    print(f"      Teams únicos: {estructura['teams_unicos']}")
                if estructura.get('events_unicos'):
                    print(f"      Events únicos: {estructura['events_unicos']}")
        
        # Mapeo y compatibilidad
        print(f"\n🗺️ MAPEO JSON → PARQUET:")
        self.crear_mapeo_json_parquet()
        
        # Análisis de compatibilidad
        print(f"\n🔍 ANÁLISIS DE COMPATIBILIDAD:")
        self.analizar_compatibilidad()
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        self.generar_recomendaciones()
    
    def analizar_compatibilidad(self):
        """Analiza la compatibilidad entre JSON y Parquet existentes"""
        print("\n   🔄 Verificando compatibilidad de estructuras...")
        
        # Verificar Match IDs duplicados
        match_ids_json = set()
        match_ids_parquet = set()
        
        # Obtener Match IDs de JSON
        for json_file in self.archivos_json:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                match_id = data.get('matchInfo', {}).get('id')
                if match_id:
                    match_ids_json.add(match_id)
            except:
                continue
        
        # Obtener Match IDs de Parquet
        for parquet_file in self.archivos_parquet:
            try:
                df = pd.read_parquet(parquet_file)
                if 'Match ID' in df.columns:
                    match_ids_parquet.update(df['Match ID'].unique())
            except:
                continue
        
        print(f"      Match IDs en JSON: {len(match_ids_json)}")
        print(f"      Match IDs en Parquet: {len(match_ids_parquet)}")
        
        duplicados = match_ids_json.intersection(match_ids_parquet)
        nuevos = match_ids_json - match_ids_parquet
        
        if duplicados:
            print(f"      ⚠️ Match IDs duplicados: {len(duplicados)}")
            print(f"         (Se sobrescribirán en la conversión)")
        
        if nuevos:
            print(f"      ✅ Match IDs nuevos: {len(nuevos)}")
        
        if not match_ids_json:
            print(f"      ❌ No se encontraron Match IDs válidos en JSON")
    
    def generar_recomendaciones(self):
        """Genera recomendaciones basadas en el análisis - MEJORADO"""
        recomendaciones = []
        
        # Contar tipos de archivos
        tipos_count = defaultdict(int)
        for archivo, tipo in self.tipos_json.items():
            tipos_count[tipo] += 1
        
        # Verificar si hay JSON sin mapear
        tipos_problematicos = [tipo for tipo in tipos_count.keys() 
                              if tipo.startswith('UNKNOWN') or tipo == 'ERROR' or tipo == 'INVALID']
        if tipos_problematicos:
            recomendaciones.append(
                f"❌ Hay {sum(tipos_count[t] for t in tipos_problematicos)} archivos JSON con problemas"
            )
            for tipo in tipos_problematicos:
                recomendaciones.append(f"   - {tipo}: {tipos_count[tipo]} archivos")
        
        # Mostrar distribución de tipos válidos
        tipos_validos = [t for t in tipos_count.keys() if not (t.startswith('UNKNOWN') or t in ['ERROR', 'INVALID'])]
        if tipos_validos:
            recomendaciones.append("✅ Archivos JSON válidos encontrados:")
            for tipo in tipos_validos:
                destinos = {
                    'MA2_COMPLETO': 'player_stats, team_stats, team_officials',
                    'MA2_SIMPLE': 'match_events (goals, cards, subs)',
                    'MA3': 'match_events (eventos detallados)',
                    'MA12': 'player_xg_stats, xg_events, team_officials',
                    'MA12_EVENTOS': 'xg_events',
                    'MA1_DETAILS': 'solo metadatos (no genera parquets)'
                }
                destino = destinos.get(tipo, 'destino desconocido')
                recomendaciones.append(f"   - {tipo}: {tipos_count[tipo]} → {destino}")
        
        # Verificar archivos Parquet faltantes
        parquets_esperados = ['player_stats.parquet', 'team_stats.parquet', 
                             'abp_events.parquet', 'team_officials.parquet']
        parquets_existentes = [os.path.basename(p) for p in self.archivos_parquet]
        faltantes = set(parquets_esperados) - set(parquets_existentes)
        
        if faltantes:
            recomendaciones.append(
                f"📄 Archivos Parquet faltantes: {', '.join(faltantes)}"
            )
            recomendaciones.append(
                f"   Estos se crearán automáticamente durante la conversión"
            )
        
        # Verificar compatibilidad de columnas
        if self.archivos_parquet:
            recomendaciones.append(
                f"✅ Se detectaron archivos Parquet existentes - se realizará merge incremental"
            )
        else:
            recomendaciones.append(
                f"🆕 No hay archivos Parquet existentes - se crearán nuevos"
            )
        
        # Recomendaciones específicas basadas en tipos encontrados
        if tipos_count.get('MA2_SIMPLE', 0) > 0:
            recomendaciones.append(
                f"💡 Se encontraron {tipos_count['MA2_SIMPLE']} archivos MA2_SIMPLE"
            )
            recomendaciones.append(
                f"   Estos contienen solo goals/cards/substitutes y se procesarán como eventos"
            )
        
        # Mostrar recomendaciones
        for i, rec in enumerate(recomendaciones, 1):
            print(f"      {i}. {rec}")
        
        if not recomendaciones:
            print(f"      ✅ Todo parece estar en orden para la conversión")
    
    def inspeccionar_todo(self):
        """Función principal que ejecuta toda la inspección"""
        if not self.escanear_archivos():
            return False
        
        # Detectar tipos de JSON
        print(f"\n🔍 DETECTANDO TIPOS DE JSON:")
        for json_file in self.archivos_json:
            tipo = self.detectar_tipo_json(json_file)
            self.tipos_json[json_file] = tipo
            print(f"   📄 {os.path.basename(json_file)} → {tipo}")
        
        # Generar informe completo
        self.generar_informe_completo()
        
        return True

def main():
    """Función principal"""
    print("🔍 INSPECTOR DE ARCHIVOS OPTA")
    print("=" * 50)
    
    # Permitir especificar carpeta
    carpeta = input("📁 Carpeta a inspeccionar (Enter para carpeta actual): ").strip()
    if not carpeta:
        carpeta = "."
    
    if not os.path.exists(carpeta):
        print(f"❌ La carpeta {carpeta} no existe")
        return
    
    inspector = InspectorOptaArchivos(carpeta)
    
    if inspector.inspeccionar_todo():
        print(f"\n🎉 Inspección completada")
        print(f"📋 Usa este análisis para ajustar tu script convertir_json_parquet.py")
    else:
        print(f"❌ No se pudo completar la inspección")

if __name__ == "__main__":
    main()