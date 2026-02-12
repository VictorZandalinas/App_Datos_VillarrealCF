#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SISTEMA DE ACTUALIZACIÓN DE DATOS
Opta API + MediaCoach Data Updater
MODIFICADO para incluir procesamiento de secuencias de Balón Parado (ABP)
"""

import json
import numpy as np
import pandas as pd
import hashlib
import requests
import time
import os
from datetime import datetime
import shutil
import pyarrow.parquet as pq
from pathlib import Path
import sys
import tempfile
import threading
import unicodedata
import subprocess
import logging

# Configurar logging al inicio del script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_update.log'),
        logging.StreamHandler()
    ]
)
# ====================================
# CONFIGURACIÓN
# ====================================

# Opta API Credentials
OPTA_API_KEY = '10lthl3y5chwn1m0fa4mfg3bqy'
OPTA_SECRET_KEY = '1u3x3eovxa0vh1lwmutbygq8xn'
DELAY_SECONDS = 60
API_TIMEOUT = 60  # Timeout para todas las llamadas a la API (segundos)

# Token cache para evitar pedir un token OAuth en cada request
_token_cache = {
    'token': None,
    'expires_at': 0
}
_token_cache_lock = threading.Lock()

# Sesión HTTP reutilizable (connection pooling)
_opta_session = None
_opta_session_lock = threading.Lock()

def _get_opta_session():
    """Obtiene o crea una sesión HTTP reutilizable para Opta API"""
    global _opta_session
    with _opta_session_lock:
        if _opta_session is None:
            _opta_session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                max_retries=requests.adapters.Retry(
                    total=3,
                    backoff_factor=2,
                    status_forcelist=[429, 500, 502, 503, 504]
                ),
                pool_connections=5,
                pool_maxsize=5
            )
            _opta_session.mount('https://', adapter)
        return _opta_session

# Paths
BASE_PATH = Path(__file__).parent
OPTA_PATH = Path('extraccion_opta/datos_opta_parquet')
MEDIACOACH_PATH = Path('extraccion_mediacoach/datos_mediacoach_parquet') # Preparado para el futuro
SPORTIAN_PATH = Path('extraccion_sportian/datos_sportian_parquet')       # Preparado para el futuro


# Event Types Mapping (sin cambios)
EVENT_TYPE_MAPPING = {
    1: "Pass", 2: "Offside Pass", 3: "Take On", 4: "Foul", 5: "Out",
    6: "Corner Awarded", 7: "Tackle", 8: "Interception", 10: "Save",
    11: "Claim", 12: "Clearance", 13: "Miss", 14: "Post", 15: "Attempt Saved",
    16: "Goal", 17: "Card", 18: "Player off", 19: "Player on", 20: "Player retired",
    21: "Player returns", 22: "Player becomes goalkeeper", 23: "Goalkeeper becomes player",
    24: "Condition change", 25: "Official change", 27: "Start delay", 28: "End delay",
    29: "Temporary stop", 30: "End", 32: "Start", 34: "Team set up",
    36: "Player changed Jersey number", 37: "Collection End", 38: "Temp Goal",
    39: "Temp Attempt", 40: "Formation change", 41: "Punch", 42: "Good skill",
    43: "Deleted event", 44: "Aerial", 45: "Challenge", 47: "Rescinded card",
    49: "Ball recovery", 50: "Dispossessed", 51: "Error", 52: "Keeper pick-up",
    53: "Cross not claimed", 54: "Smother", 55: "Offside provoked", 56: "Shield ball opp",
    57: "Foul throw-in", 58: "Penalty faced", 59: "Keeper Sweeper", 60: "Chance missed",
    61: "Ball touch", 63: "Temp Save", 64: "Resume", 65: "Contentious referee decision",
    67: "50/50", 68: "Referee Drop Ball", 69: "Failed to Block", 70: "Injury Time Announcement",
    71: "Coach Setup", 72: "Caught Offside", 73: "Other Ball Contact", 74: "Blocked Pass",
    75: "Delayed start", 76: "Early end", 79: "Coverage interruption", 80: "Drop of Ball",
    81: "Obstacle", 82: "Control", 83: "Attempted tackle", 84: "Deleted After Review"
}

# Qualifier Types Mapping (sin cambios)
QUALIFIER_MAPPING = {
    1: "Long ball", 2: "Cross", 3: "Head pass", 4: "Through ball", 5: "Free kick taken",
    6: "Corner taken", 7: "Players caught offside", 8: "Goal disallowed", 9: "Penalty",
    10: "Handball", 11: "6-seconds violation", 12: "Dangerous play", 13: "Foul",
    14: "Last line", 15: "Head", 16: "Small box - Centre", 17: "Box - Centre",
    18: "Out of box - Centre", 19: "35+ Centre", 20: "Right footed", 21: "Other body part",
    22: "Regular play", 23: "Fast break", 24: "Set piece", 25: "From corner",
    26: "Free kick", 28: "Own Goal", 29: "Assisted", 30: "Involved", 31: "Yellow Card",
    32: "Second yellow", 33: "Red Card", 34: "Referee abuse", 35: "Argument",
    36: "Violent Conduct", 37: "Time wasting", 38: "Excessive celebration",
    39: "Crowd interaction", 40: "Other reason", 41: "Injury", 42: "Tactical",
    43: "Deleted event", 44: "Player position", 45: "Temperature", 46: "Conditions",
    47: "Field Pitch", 48: "Lightings", 49: "Attendance figure", 50: "Official position",
    51: "Official ID", 52: "Possession time", 53: "Injured player ID", 54: "End cause",
    55: "Related event ID", 56: "Zone", 57: "End type", 58: "Temp stop status",
    59: "Jersey Number", 60: "Small box - Right", 61: "Small box - Left",
    62: "Box - Deep Right", 63: "Box - Right", 64: "Box - Left", 65: "Box - Deep Left",
    66: "Out of box - Deep Right", 67: "Out of box - Right", 68: "Out of box - Left",
    69: "Out of box - Deep Left", 70: "35+ Right", 71: "35+ Left", 72: "Left footed",
    73: "Left", 74: "High", 75: "Right", 76: "Low Left", 77: "High Left",
    78: "Low Centre", 79: "High Centre", 80: "Low Right", 81: "High Right",
    82: "Blocked", 83: "Close Left", 84: "Close Right", 85: "Close High",
    86: "Close Left and High", 87: "Close Right and High", 88: "High claim",
    89: "1 on 1", 90: "Deflected save", 91: "Dive and deflect", 92: "Catch",
    93: "Dive and catch", 94: "Def block", 95: "Back pass", 96: "Corner situation",
    97: "Direct free", 98: "Pitch X Coordinate", 99: "Pitch Y Coordinate",
    100: "Six Yard Blocked", 101: "Saved Off Line", 102: "Goalmouth Y Coordinate",
    103: "Goalmouth Z Coordinate", 104: "Attempt Position X Coordinate",
    105: "Attempt Position Y Coordinate", 106: "Attacking Pass", 107: "Throw in",
    108: "Volley", 109: "Overhead", 110: "Half Volley", 111: "Diving Header",
    112: "Scramble", 113: "Strong", 114: "Weak", 115: "Rising", 116: "Dipping",
    117: "Lob", 118: "One Bounce", 119: "Few Bounces", 120: "Swerve Left",
    121: "Swerve Right", 122: "Swerve Moving", 123: "Keeper Throw", 124: "Goal Kick",
    125: "Free Kick Position X Coordinate", 126: "Free Kick Position Y Coordinate",
    127: "Direction of Play", 128: "Punch", 129: "Ten Minute Possession",
    130: "Team Formation", 131: "Team Player Formation", 132: "Simulation",
    133: "Deflection", 134: "Far Wide Left", 135: "Far Wide Right", 136: "Keeper Touched",
    137: "Keeper Saved", 138: "Hit Woodwork", 139: "Own Player", 140: "Pass End X",
    141: "Pass End Y", 142: "Flag to Checker", 143: "Star Rating",
    144: "Deleted Event Type", 145: "Formation slot", 146: "Blocked X Coordinate",
    147: "Blocked Y Coordinate", 148: "Danger", 149: "Inside", 150: "Outside",
    151: "Short", 152: "Direct", 153: "Not past goal line", 154: "Intentional Assist",
    155: "Chipped", 156: "Lay-off", 157: "Launch", 158: "Persistent Infringement",
    159: "Foul and Abusive Language", 160: "Throw-in set piece", 161: "Encroachment",
    162: "Leaving field", 163: "Entering field", 164: "Spitting",
    165: "Professional Foul Last Man", 166: "Professional Foul Handball",
    167: "Out of play", 168: "Flick-on", 169: "Leading to attempt",
    170: "Leading to goal", 171: "Rescinded Card", 173: "Parried safe",
    174: "Parried danger", 175: "Fingertip", 176: "Caught", 177: "Collected",
    178: "Standing", 179: "Diving", 180: "Stooping", 181: "Reaching", 182: "Hands",
    183: "Feet", 184: "Dissent", 185: "Blocked cross", 186: "Scored", 187: "Saved",
    188: "Missed", 189: "Not visible", 190: "From shot off target",
    191: "Off the ball foul", 192: "Block by hand", 193: "Goal measure",
    194: "Captain", 195: "Pull back", 196: "Switch of play", 197: "Team kit",
    198: "GK hoof", 199: "GK kick from hands", 200: "Referee stop",
    201: "Referee delay", 202: "Weather problem", 203: "Crowd trouble", 204: "Fire",
    205: "Object thrown on pitch", 206: "Spectator on pitch",
    207: "Awaiting official's decision", 208: "Referee injury", 209: "Game end",
    210: "Assist", 211: "Overrun", 212: "Length", 213: "Angle", 214: "Big chance",
    215: "Individual play", 216: "2nd related event ID", 217: "2nd assisted",
    218: "2nd assist", 219: "Players on both posts", 220: "Player on near post",
    221: "Player on far post", 222: "No players on posts", 223: "In-swinger",
    224: "Out-swinger", 225: "Straight", 226: "Suspended", 227: "Resume",
    228: "Own shot blocked", 229: "Post match complete", 230: "GK X Coordinate",
    231: "GK Y Coordinate", 232: "Unchallenged", 233: "Opposite related event ID",
    234: "Home Team Possession", 235: "Away Team Possession", 236: "Blocked pass",
    237: "Low", 238: "Fair Play", 239: "By Wall", 240: "GK Start", 241: "Indirect",
    242: "Obstruction", 243: "Unsporting behaviour", 244: "Not Retreating",
    245: "Serious Foul", 246: "Drinks Break", 247: "Offside", 248: "Goal line",
    249: "Temp Shot On", 250: "Temp Blocked", 251: "Temp Post", 252: "Temp Missed",
    253: "Temp Miss Not Passed Goal Line", 254: "Follows a Dribble",
    255: "Open Roof", 256: "Air Humidity", 257: "Air Pressure", 258: "Sold Out",
    259: "Celsius degrees", 260: "Floodlight", 261: "1 on 1 chip", 262: "Back heel",
    263: "Direct corner", 264: "Aerial Foul", 265: "Attempted Tackle",
    266: "Put Through", 267: "Right Arm", 268: "Left Arm", 269: "Both Arms",
    270: "Right Leg", 271: "Left Leg", 272: "Both Legs", 273: "Hit Right Post",
    274: "Hit Left Post", 275: "Hit Bar", 276: "Out on sideline", 277: "Minutes",
    278: "Tap", 279: "Kick Off", 280: "Fantasy Assist Type",
    281: "Fantasy Assisted By", 282: "Fantasy Assist Team", 283: "Coach ID",
    284: "Duel", 285: "Defensive", 286: "Offensive", 287: "Over-arm",
    288: "Out of Play Secs", 289: "Denied goal-scoring opp", 290: "Coach types",
    291: "Other Ball Contact Type", 292: "Detailed Position ID",
    293: "Position Side ID", 294: "Shove/Push", 295: "Shirt Pull/Holding",
    296: "Elbow/Violent Conduct", 297: "Follows Shot Rebound",
    298: "Follows Shot Blocked", 299: "Clock Affecting", 300: "Solo Run",
    301: "Shot from cross", 302: "Checks complete/Live collection checks complete",
    303: "Floodlight failure", 304: "Ball In Play", 305: "Ball Out of Play",
    306: "Kit change", 307: "Phase of possession ID", 308: "Goes to Extra Time",
    309: "Goes to Penalties", 310: "Player goes out", 311: "Player comes back",
    312: "Phase of possession start", 313: "Illegal Restart", 314: "End of Offside",
    315: "Related Event Player ID", 316: "Passed Penalty", 317: "Penalty Set Piece",
    319: "Captain change", 323: "Follows a Rebound", 324: "Follows a Take On",
    325: "Abandonment To Follow", 328: "First Touch", 329: "VAR - Goal Awarded",
    330: "VAR - Penalty Awarded", 331: "VAR - Penalty Not Awarded",
    332: "VAR - (Red) Card Upgrade", 333: "VAR - Mistaken Identity",
    334: "VAR - Other", 335: "Referee Decision Confirmed",
    336: "Referee Decision Cancelled", 338: "Follows a Rebound Event ID",
    341: "VAR - Goal Not Awarded", 342: "VAR - Red Card Given", 343: "Review",
    344: "Video coverage lost", 345: "Overhit cross", 346: "Next event Goal-Kick",
    347: "Next event Throw-In", 348: "Penalty taker ID",
    349: "Goalkeeper punch outcome", 353: "Second (2nd) opposite related event ID",
    354: "Ball hits referee", 355: "Entering referee review area",
    356: "Excessive usage of review signal", 357: "Entering video operations room",
    358: "Official body: Reviewed and confirmed",
    359: "Official body: Reviewed and changed", 361: "Incorrect out of play decision",
    362: "Viral", 363: "Away attendance", 364: "VAR Delay", 365: "Reviewed event ID",
    374: "Goal shot timestamp", 375: "Goal shot game clock",
    376: "Low GK intervention", 377: "Medium GK intervention",
    378: "High GK intervention", 380: "Other obstacle", 381: "Fumble",
    383: "Touch type control", 384: "Touch type pass", 385: "Touch type clearance",
    386: "Driven cross", 387: "Floated cross", 388: "Jumping", 389: "Sliding",
    390: "Causing player", 391: "Mis-hit", 392: "Reckless offence",
    393: "Tactical Foul", 394: "Corner not taken", 395: "GK x coordinate time of goal",
    396: "GK y coordinate time of goal", 397: "Blocked clearance",
    398: "GK Challenge", 399: "Intended tackle target", 406: "Collection complete",
    436: "Pre-Review Event Type", 458: "Not assisted", 459: "Event type review",
    464: "Take on space", 465: "Take on overtake", 467: "Defensive 1 v 1",
    468: "Related error 1 ID", 472: "Fantasy assist ID", 474: "Related error 2 ID",
    476: "New start time", 478: "Officially announced", 479: "Estimated",
    484: "Dubious scorer", 485: "Advantage played", 486: "Concussion",
    487: "Panenka", 488: "8-Second Violation"
}

# ====================================
# GIT AUTO-SYNC FUNCTION
# ====================================

def git_auto_sync(source_name):
    """
    Realiza git add, commit y push automáticamente después de una descarga.

    Args:
        source_name: Nombre de la fuente de datos (Opta, MediaCoach, Sportian)
    """
    try:
        # Obtener el directorio del script
        script_dir = Path(__file__).parent

        print(f"\n🔄 Sincronizando cambios con GitHub ({source_name})...")

        # Git add
        result_add = subprocess.run(
            ['git', 'add', '.'],
            cwd=script_dir,
            capture_output=True,
            text=True
        )

        if result_add.returncode != 0:
            print(f"⚠️ Error en git add: {result_add.stderr}")
            return False

        # Verificar si hay cambios para commitear
        result_status = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=script_dir,
            capture_output=True,
            text=True
        )

        if not result_status.stdout.strip():
            print("ℹ️ No hay cambios nuevos para sincronizar")
            return True

        # Git commit (con [skip ci] para evitar loops infinitos)
        timestamp = datetime.now().strftime('%a %d %b %Y %H:%M:%S %Z')
        commit_message = f"Actualización automática {timestamp}\n\nDatos actualizados: {source_name}\n\n[skip ci]"
        result_commit = subprocess.run(
            ['git', 'commit', '-m', commit_message],
            cwd=script_dir,
            capture_output=True,
            text=True
        )

        if result_commit.returncode != 0:
            print(f"⚠️ Error en git commit: {result_commit.stderr}")
            return False

        print(f"✅ Commit realizado: {commit_message}")

        # Git push
        result_push = subprocess.run(
            ['git', 'push'],
            cwd=script_dir,
            capture_output=True,
            text=True
        )

        if result_push.returncode != 0:
            print(f"⚠️ Error en git push: {result_push.stderr}")
            return False

        print("✅ Push realizado correctamente a GitHub")
        logging.info(f"Git sync completado para {source_name}")
        return True

    except Exception as e:
        print(f"❌ Error en sincronización git: {e}")
        logging.error(f"Error en git_auto_sync: {e}")
        return False

# ====================================
# ABP SEQUENCE PROCESSING FUNCTIONS
# ====================================

def _get_corner_sequences(match_events_df, team_name):
    """
    Extrae secuencias de corners para un equipo específico en un partido.
    Lógica adaptada de abp1_opta_estadisticas_abp.py
    """
    # Filtrar corners del equipo
    team_corners = match_events_df[
        (match_events_df['Team Name'] == team_name) &
        (match_events_df.get('Corner taken') == 'Sí')
    ].copy()

    corner_sequences = []
    
    for _, corner_event in team_corners.iterrows():
        corner_idx = corner_event.name
        
        sequence_events = []
        current_time = corner_event['timeMin'] * 60 + corner_event['timeSec']
        pass_count = 0
        last_pass_timestamp = corner_event['timeStamp']

        for next_idx in range(corner_idx + 1, len(match_events_df)):
            next_event = match_events_df.iloc[next_idx]
            next_time = next_event['timeMin'] * 60 + next_event['timeSec']
            time_diff = next_time - current_time

            if time_diff > 5 or next_event.get('periodId', 1) != corner_event.get('periodId', 1):
                break
            if next_event['Event Name'] in ['Corner Awarded', 'Out', 'Smother', 'Foul', 'Offside', 'End Period']:
                break
            if next_event['Event Name'] == 'Pass' and float(next_event.get('x', 100)) < 55:
                break
            
            if next_event['Event Name'] == 'Pass' and next_event['Team Name'] == team_name:
                if next_event['timeStamp'] > last_pass_timestamp:
                    pass_count += 1
                    last_pass_timestamp = next_event['timeStamp']
                    
                    if pass_count >= 5:
                        passes_back_field = 0
                        for check_idx in range(corner_idx + 1, next_idx + 1):
                            check_event = match_events_df.iloc[check_idx]
                            if (check_event['Event Name'] == 'Pass' and
                                check_event['Team Name'] == team_name and
                                float(check_event.get('x', 0)) < 70):
                                passes_back_field += 1
                        
                        if passes_back_field >= 2:
                            break
            
            sequence_events.append(next_event)
            current_time = next_time
        
        # Guardar la secuencia completa (evento inicial + eventos siguientes)
        full_sequence = [corner_event.to_dict()] + [e.to_dict() for e in sequence_events]
        corner_sequences.extend(full_sequence)
        
    return corner_sequences

def _get_freekick_indirect_sequences(match_events_df, team_name):
    """
    Extrae secuencias de faltas indirectas para un equipo.
    Lógica adaptada de abp1_opta_estadisticas_abp.py
    """
    team_freekicks_indirect = match_events_df[
        (match_events_df['Team Name'] == team_name) &
        (match_events_df.get('Free kick taken') == 'Sí') &
        (match_events_df.get('Zone', '').isin(['Center', 'Right', 'Left']))
    ].copy()

    freekick_sequences = []
    
    for _, fk_event in team_freekicks_indirect.iterrows():
        fk_idx = fk_event.name
        
        sequence_events = []
        current_time = fk_event['timeMin'] * 60 + fk_event['timeSec']
        pass_count = 0
        last_pass_timestamp = fk_event['timeStamp']

        for next_idx in range(fk_idx + 1, len(match_events_df)):
            next_event = match_events_df.iloc[next_idx]
            next_time = next_event['timeMin'] * 60 + next_event['timeSec']
            time_diff = next_time - current_time

            if time_diff > 5 or next_event.get('periodId', 1) != fk_event.get('periodId', 1):
                break
            if next_event['Event Name'] in ['Corner Awarded', 'Out', 'Smother', 'Foul', 'Offside', 'End Period']:
                break
            if next_event['Event Name'] == 'Pass' and float(next_event.get('x', 100)) < 55:
                break
            
            if next_event['Event Name'] == 'Pass' and next_event['Team Name'] == team_name:
                if next_event['timeStamp'] > last_pass_timestamp:
                    pass_count += 1
                    last_pass_timestamp = next_event['timeStamp']
                    
                    if pass_count >= 5:
                        passes_back_field = 0
                        for check_idx in range(fk_idx + 1, next_idx + 1):
                            check_event = match_events_df.iloc[check_idx]
                            if (check_event['Event Name'] == 'Pass' and
                                check_event['Team Name'] == team_name and
                                float(check_event.get('x', 0)) < 70):
                                passes_back_field += 1
                        
                        if passes_back_field >= 2:
                            break
            
            sequence_events.append(next_event)
            current_time = next_time
        
        full_sequence = [fk_event.to_dict()] + [e.to_dict() for e in sequence_events]
        freekick_sequences.extend(full_sequence)
        
    return freekick_sequences

def _get_freekick_direct_sequences(match_events_df, team_name):
    """
    Extrae eventos de falta directa (la secuencia es el propio evento).
    """
    team_freekicks = match_events_df[
        (match_events_df['Team Name'] == team_name) &
        (match_events_df.get('Free kick') == 'Sí')
    ]
    return team_freekicks.to_dict('records')


def update_abp_events_standalone():
    """
    Función robusta para crear o actualizar `abp_events.parquet`.
    Compara `match_events.parquet` con `abp_events.parquet` y procesa solo los partidos que falten.
    También genera `open_play_events.parquet` con todos los eventos que NO son ABP.
    """
    logging.info("=============================================")
    logging.info("✅ INICIANDO ACTUALIZACIÓN DE EVENTOS DE ABP")
    logging.info("=============================================")

    match_events_path = OPTA_PATH / 'match_events.parquet'
    abp_events_path = OPTA_PATH / 'abp_events.parquet'
    unique_keys = ['Match ID', 'EventId', 'timeStamp', 'playerId']

    if not match_events_path.exists():
        logging.error("❌ No se encontró 'match_events.parquet'. No se puede procesar ABP.")
        return

    # Cargar todos los eventos de partido existentes
    all_match_events_df = pd.read_parquet(match_events_path)
    all_match_ids = set(all_match_events_df['Match ID'].unique())
    logging.info(f"🔍 Encontrados {len(all_match_ids)} partidos en 'match_events.parquet'.")

    # Obtener IDs de partidos ya procesados en ABP
    processed_match_ids = set()
    if abp_events_path.exists():
        try:
            processed_match_ids = set(pd.read_parquet(abp_events_path)['Match ID'].unique())
            logging.info(f"🔍 Encontrados {len(processed_match_ids)} partidos ya procesados en 'abp_events.parquet'.")
        except Exception as e:
            logging.warning(f"⚠️ No se pudo leer 'abp_events.parquet' existente: {e}. Se reconstruirá.")

    # Determinar qué partidos necesitan ser procesados
    match_ids_to_process = all_match_ids - processed_match_ids

    if not match_ids_to_process:
        logging.info("🎉 ¡Excelente! 'abp_events.parquet' ya está completamente actualizado.")
        
        # Aún así, regenerar open_play por si acaso
        logging.info("🏃 Regenerando 'open_play_events.parquet'...")
        try:
            existing_abp_df = pd.read_parquet(abp_events_path)
            
            all_match_events_df['event_key'] = (
                all_match_events_df['Match ID'].astype(str) + '_' +
                all_match_events_df['EventId'].astype(str) + '_' +
                all_match_events_df['timeStamp'].astype(str)
            )
            
            existing_abp_df['event_key'] = (
                existing_abp_df['Match ID'].astype(str) + '_' +
                existing_abp_df['EventId'].astype(str) + '_' +
                existing_abp_df['timeStamp'].astype(str)
            )
            
            open_play_df = all_match_events_df[
                ~all_match_events_df['event_key'].isin(existing_abp_df['event_key'])
            ].copy()
            
            open_play_df = open_play_df.drop(columns=['event_key'])
            open_play_path = OPTA_PATH / 'open_play_events.parquet'
            open_play_df.to_parquet(open_play_path, index=False)
            logging.info(f"✅ 'open_play_events.parquet' actualizado con {len(open_play_df)} eventos.")
            
        except Exception as e:
            logging.error(f"❌ Error al actualizar 'open_play_events.parquet': {e}")
        
        return
    
    logging.info(f"📊 Se procesarán {len(match_ids_to_process)} partidos nuevos/faltantes para ABP.")
    
    # Filtrar solo los eventos de los partidos a procesar
    events_to_process_df = all_match_events_df[all_match_events_df['Match ID'].isin(match_ids_to_process)].copy()

    # Asegurarse que las columnas requeridas existen
    required_cols = ['Team Name', 'Corner taken', 'Free kick taken', 'Free kick', 'Zone', 'x', 'timeStamp', 'timeMin', 'timeSec', 'periodId', 'Event Name']
    for col in required_cols:
        if col not in events_to_process_df.columns:
            logging.warning(f"ABP: La columna requerida '{col}' no existe. Se creará vacía.")
            events_to_process_df[col] = pd.NA

    # Procesar secuencias
    all_new_abp_rows = []
    for match_id in match_ids_to_process:
        match_df = events_to_process_df[events_to_process_df['Match ID'] == match_id]
        match_df = match_df.sort_values(['periodId', 'timeMin', 'timeSec']).reset_index(drop=True)
        
        teams = match_df['Team Name'].dropna().unique()

        for team_name in teams:
            all_new_abp_rows.extend(_get_corner_sequences(match_df, team_name))
            all_new_abp_rows.extend(_get_freekick_indirect_sequences(match_df, team_name))
            all_new_abp_rows.extend(_get_freekick_direct_sequences(match_df, team_name))

    if not all_new_abp_rows:
        logging.info("ABP: No se encontraron secuencias de balón parado en los partidos analizados.")
        return

    # Crear DataFrame y eliminar duplicados internos
    new_abp_df = pd.DataFrame(all_new_abp_rows)
    new_abp_df = new_abp_df.drop_duplicates(subset=unique_keys)
    
    logging.info(f"ABP: Se encontraron {len(new_abp_df)} eventos únicos en secuencias de ABP.")

    # Guardado incremental y seguro
    try:
        if abp_events_path.exists():
            # Cargar el archivo existente y simplemente añadir lo nuevo
            existing_abp_df = pd.read_parquet(abp_events_path)
            final_df = pd.concat([existing_abp_df, new_abp_df], ignore_index=True)
            # Asegurarse de que no haya duplicados después de concatenar
            final_df = final_df.drop_duplicates(subset=unique_keys, keep='last')
            logging.info(f"ABP: Añadidas {len(final_df) - len(existing_abp_df)} nuevas filas a 'abp_events.parquet'.")
        else:
            final_df = new_abp_df
            logging.info(f"ABP: Creando nuevo archivo 'abp_events.parquet' con {len(final_df)} filas.")

        # Asegurarse que el DataFrame final tenga todas las columnas del original
        final_df = final_df.reindex(columns=all_match_events_df.columns)

        final_df.to_parquet(abp_events_path, index=False)
        logging.info(f"✅ 'abp_events.parquet' actualizado. Total de filas: {len(final_df)}.")

    except Exception as e:
        logging.error(f"❌ Error al guardar 'abp_events.parquet': {e}", exc_info=True)
        return

    # ========================================
    # PROCESAR OPEN PLAY EVENTS (todos los eventos que NO son ABP)
    # ========================================
    logging.info("🏃 PROCESANDO EVENTOS DE JUEGO ABIERTO (Open Play)...")
    
    open_play_path = OPTA_PATH / 'open_play_events.parquet'
    
    try:
        # Crear identificadores únicos para comparación
        all_match_events_df['event_key'] = (
            all_match_events_df['Match ID'].astype(str) + '_' +
            all_match_events_df['EventId'].astype(str) + '_' +
            all_match_events_df['timeStamp'].astype(str)
        )
        
        final_df['event_key'] = (
            final_df['Match ID'].astype(str) + '_' +
            final_df['EventId'].astype(str) + '_' +
            final_df['timeStamp'].astype(str)
        )
        
        # Filtrar: Open Play = todos los eventos que NO están en ABP
        open_play_df = all_match_events_df[
            ~all_match_events_df['event_key'].isin(final_df['event_key'])
        ].copy()
        
        # Eliminar columna auxiliar
        open_play_df = open_play_df.drop(columns=['event_key'])
        
        # Guardar (sobrescribir completo)
        open_play_df.to_parquet(open_play_path, index=False)
        logging.info(f"✅ 'open_play_events.parquet' creado con {len(open_play_df)} eventos de juego abierto.")
        
    except Exception as e:
        logging.error(f"❌ Error al crear 'open_play_events.parquet': {e}", exc_info=True)

def calculate_and_save_abp_statistics():
    """
    Calcula estadísticas de ABP desde abp_events y las guarda en estadisticas_abp.parquet
    Procesa solo las combinaciones (Team Name, Week) nuevas de forma incremental.
    """
    logging.info("🔢 CALCULANDO ESTADÍSTICAS DE ABP...")
    
    abp_events_path = OPTA_PATH / 'abp_events.parquet'
    xg_events_path = OPTA_PATH / 'xg_events.parquet'
    stats_output_path = OPTA_PATH / 'estadisticas_abp.parquet'
    
    if not abp_events_path.exists():
        logging.error("❌ No se encontró abp_events.parquet")
        return
    
    if not xg_events_path.exists():
        logging.error("❌ No se encontró xg_events.parquet")
        return
    
    # Cargar datos
    abp_events_df = pd.read_parquet(abp_events_path)
    xg_events_df = pd.read_parquet(xg_events_path)
    
    # Normalizar timestamps
    def normalize_timestamp(timestamp):
        if pd.isna(timestamp):
            return timestamp
        timestamp_str = str(timestamp).strip()
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1]
        try:
            dt = pd.to_datetime(timestamp_str)
            return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        except:
            return timestamp_str
    
    abp_events_df['timeStamp'] = abp_events_df['timeStamp'].apply(normalize_timestamp)
    xg_events_df['timeStamp'] = xg_events_df['timeStamp'].apply(normalize_timestamp)
    
    # Verificar qué combinaciones (Team, Week) ya están procesadas
    if stats_output_path.exists():
        existing_stats = pd.read_parquet(stats_output_path)
        existing_combinations = set(zip(existing_stats['Team Name'], existing_stats['Week']))
        logging.info(f"📊 Encontradas {len(existing_combinations)} combinaciones (Team, Week) ya procesadas.")
    else:
        existing_combinations = set()
        logging.info("📝 No existe archivo previo. Se procesarán todas las combinaciones.")
    
    # Obtener combinaciones únicas de (Team, Week)
    all_combinations = abp_events_df[['Team Name', 'Week']].drop_duplicates()
    
    # Filtrar solo las combinaciones nuevas
    new_combinations = []
    for _, row in all_combinations.iterrows():
        if (row['Team Name'], row['Week']) not in existing_combinations:
            new_combinations.append(row)
    
    if not new_combinations:
        logging.info("🎉 ¡Estadísticas ABP ya están actualizadas! No hay nuevas combinaciones (Team, Week).")
        return
    
    team_week_combinations = pd.DataFrame(new_combinations)
    logging.info(f"🆕 Se procesarán {len(team_week_combinations)} nuevas combinaciones (Team, Week).")
    
    stats_list = []

    for _, row in team_week_combinations.iterrows():
        team = row['Team Name']
        week = row['Week']
        
        # Filtrar eventos de ese equipo en esa jornada
        team_week_events = abp_events_df[
            (abp_events_df['Team Name'] == team) & 
            (abp_events_df['Week'] == week)
        ]
        
        team_stats = {
            'Team Name': team, 
            'Week': week
        }
        # Obtener el Match ID único de esta jornada
        match_id = team_week_events['Match ID'].iloc[0]

        # === CORNERS ===
        # Corners a favor
        corners_favor_count = len(team_week_events[
            team_week_events['Corner taken'] == 'Sí'
        ])

        team_stats['corners_a_favor'] = corners_favor_count
        
        # Corners en contra
        corners_contra_count = len(abp_events_df[
            (abp_events_df['Match ID'] == match_id) &
            (abp_events_df['Team Name'] != team) &
            (abp_events_df['Corner taken'] == 'Sí')
        ])
        team_stats['corners_en_contra'] = corners_contra_count
        
        # xG de corner a favor
        corner_sequences = _abp_get_corner_sequences(abp_events_df, team, [match_id])
        total_corner_xg = 0
        total_corner_shots = 0
        
        for seq in corner_sequences:
            if seq['shot_events']:
                xg_values = _abp_get_xg_for_shot_events(seq['shot_events'], xg_events_df)
                total_corner_xg += sum(xg_values)
                total_corner_shots += len(seq['shot_events'])
        
        team_stats['xg_corner_a_favor'] = total_corner_xg
        
        # xG de corner en contra
        corner_xg_contra = 0
        corner_shots_contra = 0
        
        rival_teams = abp_events_df[
            (abp_events_df['Match ID'] == match_id) & 
            (abp_events_df['Team Name'] != team)
        ]['Team Name'].unique()

        for rival_team in rival_teams:
            rival_sequences = _abp_get_corner_sequences(abp_events_df, rival_team, [match_id])
            for seq in rival_sequences:
                if seq['shot_events']:
                    xg_values = _abp_get_xg_for_shot_events(seq['shot_events'], xg_events_df)
                    corner_xg_contra += sum(xg_values)
                    corner_shots_contra += len(seq['shot_events'])
        
        team_stats['xg_corner_en_contra'] = corner_xg_contra
        
        # Tiros por corner
        team_stats['tiros_por_corner_favor'] = (total_corner_shots / corners_favor_count) if corners_favor_count > 0 else 0
        team_stats['tiros_por_corner_contra'] = (corner_shots_contra / corners_contra_count) if corners_contra_count > 0 else 0
        
        # xG por corner
        team_stats['xg_por_corner_favor'] = (total_corner_xg / corners_favor_count) if corners_favor_count > 0 else 0
        team_stats['xg_por_corner_contra'] = (corner_xg_contra / corners_contra_count) if corners_contra_count > 0 else 0
        
        # === FALTAS DIRECTAS ===
        # Faltas a favor
        faltas_favor_count = len(team_week_events[
            team_week_events['Free kick'] == 'Sí'
        ])
        team_stats['faltas_a_favor'] = faltas_favor_count
        
        # Faltas en contra
        faltas_contra_count = len(abp_events_df[
            (abp_events_df['Match ID'] == match_id) &
            (abp_events_df['Team Name'] != team) &
            (abp_events_df['Free kick'] == 'Sí')
        ])
        team_stats['faltas_en_contra'] = faltas_contra_count
        
        # xG de falta a favor
        fk_sequences = _abp_get_freekick_sequences(abp_events_df, team, [match_id])
        total_fk_xg = 0
        total_fk_shots = 0
        
        for seq in fk_sequences:
            if seq['shot_events']:
                xg_values = _abp_get_xg_for_shot_events(seq['shot_events'], xg_events_df)
                total_fk_xg += sum(xg_values)
                total_fk_shots += len(seq['shot_events'])
        
        team_stats['xg_falta_a_favor'] = total_fk_xg
        
        # xG de falta en contra
        fk_xg_contra = 0
        fk_shots_contra = 0
        
        rival_teams = abp_events_df[
            (abp_events_df['Match ID'] == match_id) & 
            (abp_events_df['Team Name'] != team)
        ]['Team Name'].unique()

        for rival_team in rival_teams:
            rival_fk_sequences = _abp_get_freekick_sequences(abp_events_df, rival_team, [match_id])
            for seq in rival_fk_sequences:
                if seq['shot_events']:
                    xg_values = _abp_get_xg_for_shot_events(seq['shot_events'], xg_events_df)
                    fk_xg_contra += sum(xg_values)
                    fk_shots_contra += len(seq['shot_events'])
        
        team_stats['xg_falta_en_contra'] = fk_xg_contra
        
        # Tiros por falta
        team_stats['tiros_por_falta_favor'] = (total_fk_shots / faltas_favor_count) if faltas_favor_count > 0 else 0
        team_stats['tiros_por_falta_contra'] = (fk_shots_contra / faltas_contra_count) if faltas_contra_count > 0 else 0
        
        # xG por falta
        team_stats['xg_por_falta_favor'] = (total_fk_xg / faltas_favor_count) if faltas_favor_count > 0 else 0
        team_stats['xg_por_falta_contra'] = (fk_xg_contra / faltas_contra_count) if faltas_contra_count > 0 else 0
        
        # === FALTAS INDIRECTAS ===
        # Faltas indirectas a favor
        faltas_indirectas_favor_count = len(team_week_events[
            (team_week_events['Free kick taken'] == 'Sí') &
            (team_week_events['Zone'].isin(['Center', 'Right', 'Left']))
        ])
        team_stats['faltas_indirectas_a_favor'] = faltas_indirectas_favor_count

        # Faltas indirectas en contra
        faltas_indirectas_contra_count = len(abp_events_df[
            (abp_events_df['Match ID'] == match_id) &
            (abp_events_df['Team Name'] != team) &
            (abp_events_df['Free kick taken'] == 'Sí') &
            (abp_events_df['Zone'].isin(['Center', 'Right', 'Left']))
        ])
        team_stats['faltas_indirectas_en_contra'] = faltas_indirectas_contra_count

        # xG de falta indirecta a favor
        fk_indirect_sequences = _abp_get_freekick_indirect_sequences(abp_events_df, team, [match_id])
        total_fk_indirect_xg = 0
        total_fk_indirect_shots = 0

        for seq in fk_indirect_sequences:
            if seq['shot_events']:
                xg_values = _abp_get_xg_for_shot_events(seq['shot_events'], xg_events_df)
                total_fk_indirect_xg += sum(xg_values)
                total_fk_indirect_shots += len(seq['shot_events'])

        team_stats['xg_falta_indirecta_a_favor'] = total_fk_indirect_xg

        # xG de falta indirecta en contra
        fk_indirect_xg_contra = 0
        fk_indirect_shots_contra = 0

        rival_teams = abp_events_df[
            (abp_events_df['Match ID'] == match_id) & 
            (abp_events_df['Team Name'] != team)
        ]['Team Name'].unique()

        for rival_team in rival_teams:
            rival_fk_indirect_sequences = _abp_get_freekick_indirect_sequences(abp_events_df, rival_team, [match_id])
            for seq in rival_fk_indirect_sequences:
                if seq['shot_events']:
                    xg_values = _abp_get_xg_for_shot_events(seq['shot_events'], xg_events_df)
                    fk_indirect_xg_contra += sum(xg_values)
                    fk_indirect_shots_contra += len(seq['shot_events'])

        team_stats['xg_falta_indirecta_en_contra'] = fk_indirect_xg_contra

        # Tiros por falta indirecta
        team_stats['tiros_por_falta_indirecta_favor'] = (total_fk_indirect_shots / faltas_indirectas_favor_count) if faltas_indirectas_favor_count > 0 else 0
        team_stats['tiros_por_falta_indirecta_contra'] = (fk_indirect_shots_contra / faltas_indirectas_contra_count) if faltas_indirectas_contra_count > 0 else 0

        # xG por falta indirecta
        team_stats['xg_por_falta_indirecta_favor'] = (total_fk_indirect_xg / faltas_indirectas_favor_count) if faltas_indirectas_favor_count > 0 else 0
        team_stats['xg_por_falta_indirecta_contra'] = (fk_indirect_xg_contra / faltas_indirectas_contra_count) if faltas_indirectas_contra_count > 0 else 0

        stats_list.append(team_stats)

    # Crear DataFrame final
    combined_stats = pd.DataFrame(stats_list)
    combined_stats = combined_stats.fillna(0)
    
    # Calcular rankings
    combined_stats['ranking_corners_favor'] = combined_stats['corners_a_favor'].rank(ascending=False, method='min').astype(int)
    combined_stats['ranking_xg_total'] = (
        combined_stats['xg_corner_a_favor'] + 
        combined_stats['xg_falta_a_favor'] +
        combined_stats['xg_falta_indirecta_a_favor']
    ).rank(ascending=False, method='min').astype(int)
    combined_stats['ranking_faltas_favor'] = combined_stats['faltas_a_favor'].rank(ascending=False, method='min').astype(int)
    combined_stats['ranking_faltas_indirectas_favor'] = combined_stats['faltas_indirectas_a_favor'].rank(ascending=False, method='min').astype(int)
    
    # Guardar con append incremental
    if stats_output_path.exists():
        existing_stats = pd.read_parquet(stats_output_path)
        final_stats = pd.concat([existing_stats, combined_stats], ignore_index=True)
        logging.info(f"✅ Añadidas {len(combined_stats)} nuevas filas (Team, Week)")
        logging.info(f"📊 Total: {len(existing_stats)} → {len(final_stats)} filas")
    else:
        final_stats = combined_stats
        logging.info(f"✅ Archivo creado con {len(final_stats)} filas")

    final_stats.to_parquet(stats_output_path, index=False)
    logging.info(f"💾 'estadisticas_abp.parquet' guardado correctamente.")


def _abp_get_corner_sequences(abp_events_df, team_name, match_ids=None):
    """Extrae secuencias de corners para análisis de estadísticas ABP"""
    df = abp_events_df.copy()
    
    if match_ids is not None:
        df = df[df['Match ID'].isin(match_ids)]
    
    # Filtrar corners del equipo
    team_corners = df[
        (df['Team Name'] == team_name) & 
        (df['Corner taken'] == 'Sí')
    ].copy()
    
    corner_sequences = []
    
    for _, corner_event in team_corners.iterrows():
        match_id = corner_event['Match ID']
        match_events = df[df['Match ID'] == match_id].sort_values(['timeMin', 'timeSec']).reset_index()
        
        # Encontrar el índice del corner
        corner_idx = None
        for idx, event in match_events.iterrows():
            if (event['Team Name'] == team_name and 
                event['timeMin'] == corner_event['timeMin'] and 
                event['timeSec'] == corner_event['timeSec'] and
                event['Corner taken'] == 'Sí'):
                corner_idx = idx
                break
        
        if corner_idx is None:
            continue
            
        # Analizar secuencia después del corner
        sequence_events = []
        current_time = corner_event['timeMin'] * 60 + corner_event['timeSec']
        pass_count = 0
        last_pass_timestamp = corner_event['timeStamp']

        for next_idx in range(corner_idx + 1, len(match_events)):
            next_event = match_events.iloc[next_idx]
            next_time = next_event['timeMin'] * 60 + next_event['timeSec']
            time_diff = next_time - current_time
            
            # Más de 5 segundos o cambio de período
            if time_diff > 5 or next_event.get('periodId', 1) != corner_event.get('periodId', 1):
                break
            
            # Eventos que terminan la secuencia
            if next_event['Event Name'] in ['Corner Awarded', 'Out','Smother','Foul','Offside', 'End Period']:
                break
                
            # Pass con x < 55
            if next_event['Event Name'] == 'Pass' and float(next_event.get('x', 100)) < 55:
                break
            
            # Contar pases
            if next_event['Event Name'] == 'Pass' and next_event['Team Name'] == team_name:
                if next_event['timeStamp'] > last_pass_timestamp:
                    pass_count += 1
                    last_pass_timestamp = next_event['timeStamp']
                    
                    if pass_count >= 5:
                        passes_back_field = 0
                        for check_idx in range(corner_idx + 1, next_idx + 1):
                            check_event = match_events.iloc[check_idx]
                            if (check_event['Event Name'] == 'Pass' and 
                                check_event['Team Name'] == team_name and
                                float(check_event.get('x', 0)) < 70):
                                passes_back_field += 1
                        
                        if passes_back_field >= 2:
                            break
            
            sequence_events.append(next_event)
            current_time = next_time
        
        # Buscar eventos de finalización
        shot_events = []
        for event in sequence_events:
            if event['Event Name'] in ['Miss', 'Goal', 'Post', 'Attempt Saved']:
                shot_events.append(event)
        
        corner_sequences.append({
            'corner_event': corner_event,
            'sequence_events': sequence_events,
            'shot_events': shot_events
        })
    
    return corner_sequences


def _abp_get_freekick_indirect_sequences(abp_events_df, team_name, match_ids=None):
    """Extrae secuencias de faltas indirectas para análisis de estadísticas ABP"""
    df = abp_events_df.copy()
    
    if match_ids is not None:
        df = df[df['Match ID'].isin(match_ids)]
    
    # Filtrar faltas indirectas
    team_freekicks_indirect = df[
        (df['Team Name'] == team_name) & 
        (df['Free kick taken'] == 'Sí') &
        (df['Zone'].isin(['Center', 'Right', 'Left']))
    ].copy()
    
    freekick_sequences = []
    
    for _, fk_event in team_freekicks_indirect.iterrows():
        match_id = fk_event['Match ID']
        match_events = df[df['Match ID'] == match_id].sort_values(['timeMin', 'timeSec']).reset_index()
        
        # Encontrar el índice de la falta
        fk_idx = None
        for idx, event in match_events.iterrows():
            if (event['Team Name'] == team_name and 
                event['timeMin'] == fk_event['timeMin'] and 
                event['timeSec'] == fk_event['timeSec'] and
                event['Free kick taken'] == 'Sí'):
                fk_idx = idx
                break
        
        if fk_idx is None:
            continue
            
        # Analizar secuencia (igual que corners)
        sequence_events = []
        current_time = fk_event['timeMin'] * 60 + fk_event['timeSec']
        pass_count = 0
        last_pass_timestamp = fk_event['timeStamp']
        
        for next_idx in range(fk_idx + 1, len(match_events)):
            next_event = match_events.iloc[next_idx]
            next_time = next_event['timeMin'] * 60 + next_event['timeSec']
            time_diff = next_time - current_time
            
            if time_diff > 5 or next_event.get('periodId', 1) != fk_event.get('periodId', 1):
                break
            
            if next_event['Event Name'] in ['Corner Awarded', 'Out','Smother','Foul','Offside', 'End Period']:
                break
                
            if next_event['Event Name'] == 'Pass' and float(next_event.get('x', 100)) < 55:
                break

            if next_event['Event Name'] == 'Pass' and next_event['Team Name'] == team_name:
                if next_event['timeStamp'] > last_pass_timestamp:
                    pass_count += 1
                    last_pass_timestamp = next_event['timeStamp']
                    
                    if pass_count >= 5:
                        passes_back_field = 0
                        for check_idx in range(fk_idx + 1, next_idx + 1):
                            check_event = match_events.iloc[check_idx]
                            if (check_event['Event Name'] == 'Pass' and 
                                check_event['Team Name'] == team_name and
                                float(check_event.get('x', 0)) < 70):
                                passes_back_field += 1
                        
                        if passes_back_field >= 2:
                            break
            
            sequence_events.append(next_event)
            current_time = next_time
        
        # Buscar eventos de finalización
        shot_events = []
        for event in sequence_events:
            if event['Event Name'] in ['Miss', 'Goal', 'Post', 'Attempt Saved']:
                shot_events.append(event)
        
        freekick_sequences.append({
            'freekick_event': fk_event,
            'sequence_events': sequence_events,
            'shot_events': shot_events
        })
    
    return freekick_sequences


def _abp_get_freekick_sequences(abp_events_df, team_name, match_ids=None):
    """Extrae secuencias de faltas directas para análisis de estadísticas ABP"""
    df = abp_events_df.copy()
    
    if match_ids is not None:
        df = df[df['Match ID'].isin(match_ids)]
    
    # Filtrar faltas directas
    team_freekicks = df[
        (df['Team Name'] == team_name) & 
        (df['Free kick'] == 'Sí')
    ].copy()
    
    freekick_sequences = []
    
    for _, fk_event in team_freekicks.iterrows():
        # Buscar eventos de finalización directamente en la falta
        shot_events = []
        if fk_event['Event Name'] in ['Goal', 'Attempt Saved']:
            shot_events.append(fk_event)
        
        freekick_sequences.append({
            'freekick_event': fk_event,
            'shot_events': shot_events
        })
    
    return freekick_sequences


def _abp_get_xg_for_shot_events(shot_events, xg_events_df):
    """Obtiene xG para eventos de tiro mediante merge por timeStamp"""
    if not shot_events or xg_events_df is None or xg_events_df.empty:
        return []
    
    xg_values = []
    
    for event in shot_events:
        # Hacer merge por Match ID, Team ID y timeStamp
        matching_xg = xg_events_df[
            (xg_events_df['Match ID'] == event['Match ID']) &
            (xg_events_df['Team ID'] == event['Team ID']) &
            (xg_events_df['timeStamp'] == event['timeStamp'])
        ]
        
        if not matching_xg.empty:
            try:
                xg_val = float(matching_xg.iloc[0]['qualifier 321'])
                xg_values.append(xg_val)
            except (ValueError, TypeError, KeyError):
                xg_values.append(0.0)
        else:
            xg_values.append(0.0)
    
    return xg_values

# ====================================
# OPTA API FUNCTIONS (sin cambios)
# ====================================

def request_headers():
    """OAuth authentication for Opta API - con caché de token"""
    global _token_cache

    with _token_cache_lock:
        # Reutilizar token si aún es válido (margen de 30s)
        if _token_cache['token'] and time.time() < (_token_cache['expires_at'] - 30):
            return {'Authorization': f'Bearer {_token_cache["token"]}'}

    timestamp = int(round(time.time() * 1000))
    post_url = f'https://oauth.performgroup.com/oauth/token/{OPTA_API_KEY}?_fmt=json&_rt=b'

    # Generate unique hash
    key = str.encode(OPTA_API_KEY + str(timestamp) + OPTA_SECRET_KEY)
    unique_hash = hashlib.sha512(key).hexdigest()

    # OAuth headers
    oauth_headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f'Basic {unique_hash}',
        'Timestamp': str(timestamp)
    }

    body = {
        'grant_type': 'client_credentials',
        'scope': 'b2b-feeds-auth'
    }

    try:
        session = _get_opta_session()
        response = session.post(post_url, data=body, headers=oauth_headers, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        access_token = data['access_token']

        # Cachear token (duración típica: 5-10 min, usamos 4 min por seguridad)
        with _token_cache_lock:
            _token_cache['token'] = access_token
            _token_cache['expires_at'] = time.time() + 240  # 4 minutos

        return {'Authorization': f'Bearer {access_token}'}
    except Exception as e:
        logging.error(f"❌ Error obteniendo token OAuth: {e}")
        # Si hay un token antiguo, intentar usarlo
        if _token_cache['token']:
            logging.warning("⚠️ Usando token OAuth anterior como fallback")
            return {'Authorization': f'Bearer {_token_cache["token"]}'}
        raise

def get_available_stages():
    """Get available stages/seasons from Opta API"""
    request_parameters = {
        "_fmt": "json",
        "_pgSz": "1000",
        "_pgNm": "1",
        "_rt": "b"
    }
    
    sdapi_get_url = f'https://api.performfeeds.com/soccerdata/match/{OPTA_API_KEY}/'
    
    try:
        response = _get_opta_session().get(sdapi_get_url, headers=request_headers(), params=request_parameters, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        matches = data.get('match', [])
        
        stages = {}
        for match in matches:
            match_info = match.get('matchInfo', {})
            competition = match_info.get('competition', {})
            stage = match_info.get('stage', {})
            stage_id = stage.get('id')
            stage_name = stage.get('name', 'N/A')
            comp_name = competition.get('name', 'N/A')
            comp_id = competition.get('id', 'N/A')
            
            if stage_id and stage_id not in stages:
                stages[stage_id] = {
                    'name': stage_name,
                    'competition': comp_name,
                    'competition_id': comp_id,
                    'start_date': stage.get('startDate', 'N/A'),
                    'end_date': stage.get('endDate', 'N/A')
                }
        
        return stages
    except Exception as e:
        print(f"❌ Error obteniendo stages: {e}")
        return {}

def get_all_tournaments():
    """
    Obtiene TODAS las competiciones/torneos disponibles en la API usando el feed OT2 (Tournament Calendar).
    Esto lista todas las ligas a las que tienes acceso con tu suscripción.
    """
    request_parameters = {
        "_fmt": "json",
        "_rt": "b",
        "_pgSz": "1000"  # Tamaño de página grande para obtener todo
    }
    
    sdapi_get_url = f'https://api.performfeeds.com/soccerdata/tournamentcalendar/{OPTA_API_KEY}/'
    
    try:
        response = _get_opta_session().get(sdapi_get_url, headers=request_headers(), params=request_parameters, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        competitions = {}
        
        # El feed devuelve una estructura con competiciones y sus tournament calendars
        for competition in data.get('competition', []):
            comp_id = competition.get('competitionId')
            comp_name = competition.get('competitionName', 'N/A')
            comp_code = competition.get('competitionCode', 'N/A')
            area_name = competition.get('areaName', 'N/A')
            
            # Cada competición tiene tournament calendars (temporadas/fases)
            for tc in competition.get('tournamentCalendar', []):
                tc_id = tc.get('id')
                tc_name = tc.get('name', 'N/A')
                start_date = tc.get('startDate', 'N/A')
                end_date = tc.get('endDate', 'N/A')
                
                if tc_id:
                    competitions[tc_id] = {
                        'tournament_calendar_id': tc_id,
                        'tournament_name': tc_name,
                        'competition_id': comp_id,
                        'competition_name': comp_name,
                        'competition_code': comp_code,
                        'area': area_name,
                        'start_date': start_date,
                        'end_date': end_date
                    }
        
        return competitions
        
    except Exception as e:
        print(f"❌ Error obteniendo tournament calendars: {e}")
        return {}


def get_granular_data_status(match_ids, messages=None):
    """
    Verifica qué datos existen para cada Match ID en cada parquet.

    Retorna: {
        'match_id_123': {
            'player_stats': True,
            'team_stats': True,
            'match_events': True,
            'posesiones': False,  # Este falta
            ...
        }
    }
    """
    def add_message(msg, msg_type="info"):
        if messages is not None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            messages.append({
                'timestamp': timestamp,
                'message': msg,
                'type': msg_type
            })
        print(msg)

    # Mapeo de tipo de dato a archivo parquet
    parquet_config = {
        'player_stats': 'player_stats.parquet',
        'team_stats': 'team_stats.parquet',
        'player_xg_stats': 'player_xg_stats.parquet',
        'xg_events': 'xg_events.parquet',
        'match_events': 'match_events.parquet',
        'team_officials': 'team_officials.parquet',
        'posesiones': 'posesiones.parquet'
    }

    # Inicializar resultado: cada match_id con todos los tipos en False
    result = {}
    for match_id in match_ids:
        result[match_id] = {data_type: False for data_type in parquet_config.keys()}

    add_message("🔍 Verificación granular de datos existentes...")

    # Para cada parquet, verificar qué match_ids ya existen (solo cargamos columna Match ID)
    for data_type, filename in parquet_config.items():
        filepath = OPTA_PATH / filename
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath, columns=['Match ID'])
                if not df.empty:
                    existing_ids = set(df['Match ID'].unique())
                    found_count = 0
                    for match_id in match_ids:
                        if match_id in existing_ids:
                            result[match_id][data_type] = True
                            found_count += 1
                    add_message(f"   📄 {filename}: {found_count}/{len(match_ids)} partidos con datos")
                    del df, existing_ids  # Liberar memoria
            except Exception as e:
                add_message(f"   ❌ Error leyendo {filename}: {e}", "warning")
        else:
            add_message(f"   📄 {filename}: no existe (se creará)")

    # Resumen de partidos con datos incompletos
    incomplete_count = sum(1 for mid in match_ids if not all(result[mid].values()))
    complete_count = len(match_ids) - incomplete_count

    add_message(f"   ✅ Partidos completos: {complete_count}")
    add_message(f"   ⚠️ Partidos con datos faltantes: {incomplete_count}")

    return result


def get_existing_match_ids(messages=None):
    """Get existing match IDs from parquet files"""
    def add_message(msg):
        if messages is not None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            messages.append({
                'timestamp': timestamp,
                'message': msg,
                'type': 'info'
            })
        print(msg)
    
    existing_match_ids = set()
    
    parquet_files = [
        'player_stats.parquet',
        'team_stats.parquet',
        'player_xg_stats.parquet',
        'xg_events.parquet',
        'match_events.parquet',
        'team_officials.parquet',
        'posesiones.parquet'
    ]
    
    add_message("🔍 Revisando archivos existentes...")
    
    for filename in parquet_files:
        filepath = OPTA_PATH / filename
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath, columns=['Match ID'])
                if not df.empty:
                    file_match_ids = set(df['Match ID'].unique())
                    existing_match_ids.update(file_match_ids)
                    add_message(f"   📄 {filename}: {len(file_match_ids)} partidos únicos")
                    del df, file_match_ids
            except Exception as e:
                add_message(f"   ❌ Error leyendo {filename}: {e}")
        else:
            add_message(f"   📄 {filename}: no existe")
    
    if existing_match_ids:
        add_message(f"   ✅ Total Match IDs existentes: {len(existing_match_ids)}")
    else:
        add_message(f"   📁 No se encontraron datos previos - descarga completa")
    
    # Al final, antes de return:
    if existing_match_ids:
        add_message(f"   ✅ Total Match IDs únicos: {len(existing_match_ids)}")
        # Verificar integridad
        match_ids_list = list(existing_match_ids)
        if None in match_ids_list or 'N/A' in match_ids_list:
            add_message(f"   ⚠️ Detectados Match IDs inválidos", "warning")
    
    return existing_match_ids

def get_matches_by_weeks_range(stage_id, start_week, end_week, messages=None, progress_callback=None):
    """Get matches by range - versión web con progreso CORREGIDA"""
    # --- CORRECCIÓN AQUÍ: Añadido msg_type="info" ---
    def add_message(msg, msg_type="info"):
        if messages is not None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            messages.append({
                'timestamp': timestamp,
                'message': msg,
                'type': msg_type  # Usamos el tipo pasado
            })
        print(msg)
    
    all_matches = []
    add_message(f"🔍 Buscando partidos para Stage ID: {stage_id}")
    total_weeks = end_week - start_week + 1
    
    for week in range(start_week, end_week + 1):
        # Evitar división por cero si start_week == end_week
        week_progress = 20
        if total_weeks > 0:
            week_progress = 20 + ((week - start_week) / total_weeks) * 10
            
        if progress_callback:
            progress_callback(week_progress, f"Buscando jornada {week}/{end_week}", messages)
        
        add_message(f"📅 Jornada {week}...")
        matches_df = get_match_ids_advanced(
            max_matches=50,
            specific_week=str(week),
            stage_id=stage_id,
            messages=messages
        )
        
        if not matches_df.empty:
            add_message(f"   ✅ Encontrados {len(matches_df)} partidos en jornada {week}")
            all_matches.append(matches_df)
        else:
            add_message(f"   ⚠️ No se encontraron partidos en jornada {week}", "warning")
        time.sleep(2)
    
    if all_matches:
        result_df = pd.concat(all_matches, ignore_index=True)
        # Eliminar duplicados por Match ID (puede ocurrir si el fallback sin week devolvió todo)
        before_dedup = len(result_df)
        result_df = result_df.drop_duplicates(subset=['Match ID'], keep='first')
        if len(result_df) < before_dedup:
            add_message(f"   🔄 Eliminados {before_dedup - len(result_df)} partidos duplicados")
        add_message(f"📊 Total partidos encontrados: {len(result_df)}")
        return result_df
    else:
        add_message("❌ No se encontraron partidos", "error")
        return pd.DataFrame()

def get_max_existing_week(existing_match_ids):
    """Obtiene la jornada máxima que ya existe en los datos"""
    max_week = 0
    
    parquet_files = ['player_stats.parquet', 'team_stats.parquet']
    
    for filename in parquet_files:
        filepath = OPTA_PATH / filename
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath)
                if not df.empty and 'Week' in df.columns:
                    file_max_week = df['Week'].max()
                    if pd.notna(file_max_week):
                        max_week = max(max_week, int(file_max_week))
            except Exception as e:
                print(f"   ❌ Error verificando {filename}: {e}")
    
    return max_week

def get_match_ids_advanced(max_matches=50, specific_week=None, stage_id=None, messages=None):
    """Get match IDs - versión robusta anti-404"""
    def add_message(msg):
        if messages is not None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            messages.append({
                'timestamp': timestamp,
                'message': msg,
                'type': 'info'
            })
        print(msg)

    # Configuración base
    base_params = {
        "_fmt": "json",
        "_pgSz": str(max_matches),
        "_pgNm": "1",
        "live": "yes",
        "_rt": "b"
    }
    
    if stage_id:
        base_params["stg"] = str(stage_id)

    # URL base
    sdapi_get_url = f'https://api.performfeeds.com/soccerdata/match/{OPTA_API_KEY}/'
    
    # Intento 1: Con filtro de semana (si existe)
    try:
        current_params = base_params.copy()
        if specific_week:
            current_params["week"] = str(specific_week)
            
        response = _get_opta_session().get(sdapi_get_url, headers=request_headers(), params=current_params, timeout=API_TIMEOUT)
        response.raise_for_status()
        
    except requests.exceptions.HTTPError as e:
        # Si da error 404 y estamos filtrando por semana, es probable que la competición
        # no soporte jornadas. Intentamos buscar SIN jornada.
        if response.status_code == 404 and specific_week:
            print(f"   ⚠️ Error 404 con jornada {specific_week}. Probando descarga general de la fase...")
            try:
                current_params = base_params.copy()
                # NO agregamos "week" esta vez
                response = _get_opta_session().get(sdapi_get_url, headers=request_headers(), params=current_params, timeout=API_TIMEOUT)
                response.raise_for_status()
            except Exception as e2:
                print(f"   ❌ Error también en búsqueda general: {e2}")
                return pd.DataFrame()
        else:
            print(f"❌ Error obteniendo partidos: {e}")
            return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return pd.DataFrame()

    # Procesamiento de la respuesta (común para ambos intentos)
    try:
        data = response.json()
        matches = data.get('match', [])
        
        match_list = []
        for match in matches:
            match_info = match.get('matchInfo', {})
            match_id = match_info.get('id')
            
            # Verificar si coincide la semana si se pidió una específica
            # Esto es necesario porque el segundo intento descarga todo
            match_week = match_info.get('week', 'N/A')
            
            # Si pedimos una semana específica y el fallback descargó todo,
            # filtramos manualmente para quedarnos solo con la jornada solicitada
            if specific_week and str(match_week) != str(specific_week):
                 continue

            if match_id:
                competition = match_info.get('competition', {})
                stage = match_info.get('stage', {})
                contestants = match_info.get('contestant', [])
                teams = [{'id': c.get('id'), 'name': c.get('name'), 'code': c.get('code')} for c in contestants]
                
                live_data = match.get('liveData', {})
                match_details = live_data.get('matchDetails', {})
                match_status = match_details.get('matchStatus', '')
                
                if match_status.lower() in ['played', 'finished', 'ft', 'final']:
                    match_data = {
                        'Match ID': match_id,
                        'Competition': competition.get('name', 'N/A'),
                        'Competition ID': competition.get('id', 'N/A'),
                        'Stage ID': stage.get('id', 'N/A'),
                        'Stage Name': stage.get('name', 'N/A'),
                        'Date': match_info.get('date', 'N/A'),
                        'Week': match_week,
                        'Match Status': match_status,
                        'Teams': teams
                    }
                    match_list.append(match_data)
        
        return pd.DataFrame(match_list)
    except Exception as e:
        print(f"❌ Error procesando JSON de partidos: {e}")
        return pd.DataFrame()

def process_match_player_stats(match_id, season):
    """Process MA2 Player Stats + Team Officials - CORREGIDA"""
    request_parameters = {
        "_fmt": "json",
        "detailed": "yes",
        "fx": match_id,
        "_rt": "b"
    }
    
    sdapi_get_url = f'https://api.performfeeds.com/soccerdata/matchstats/{OPTA_API_KEY}/'
    response = _get_opta_session().get(sdapi_get_url, headers=request_headers(), params=request_parameters, timeout=API_TIMEOUT)
    
    if response.status_code != 200:
        print(f"❌ Error MA2 Player Stats: {response.status_code}")
        return pd.DataFrame(), pd.DataFrame()
    
    data = response.json()
    match_info = data.get('matchInfo', {})
    competition_info = match_info.get('competition', {})
    stage_info = match_info.get('stage', {})
    live_data = data.get('liveData', {})
    home_away_info = get_home_away_info(match_info, live_data)
    line_ups = live_data.get('lineUp', [])
    
    player_stats_data = []
    team_officials_data = []
    
    for line_up in line_ups:
        team_id = line_up.get('contestantId')
        team_name = home_away_info['team_mapping'].get(team_id, {}).get('name', 'N/A')
        team_position = home_away_info['team_mapping'].get(team_id, {}).get('position', 'N/A')
        is_home = team_id == home_away_info['home_team_id']
        is_away = team_id == home_away_info['away_team_id']
        
        # Process players
        for player in line_up.get('player', []):
            player_entry = {
                'Match ID': match_info.get('id', 'N/A'),
                'Competition ID': competition_info.get('id', 'N/A'),
                'Competition Name': competition_info.get('name', 'N/A'),
                'Season': season,
                'Week': match_info.get('week', 'N/A'),
                'Stage ID': stage_info.get('id', 'N/A'),
                'Stage Name': stage_info.get('name', 'N/A'),
                'Team ID': team_id,
                'Team Name': team_name,
                'Team Position': team_position,
                'Is Home': is_home,
                'Is Away': is_away,
                'HT Home Score': home_away_info['ht_home'],
                'HT Away Score': home_away_info['ht_away'],
                'FT Home Score': home_away_info['ft_home'],
                'FT Away Score': home_away_info['ft_away'],
                'Player ID': player.get('playerId', 'N/A'),
                'First Name': player.get('firstName', 'N/A'),
                'Last Name': player.get('lastName', 'N/A'),
                'Match Name': player.get('matchName', 'N/A'),
                'Shirt Number': player.get('shirtNumber', 'N/A'),
                'Position': player.get('position', 'N/A'),
                'Position Side': player.get('positionSide', 'N/A'),
                'Formation Place': player.get('formationPlace', 'N/A'),
            }
            
            for stat in player.get('stat', []):
                stat_type = stat.get('type', '')
                stat_value = stat.get('value', 0)
                player_entry[stat_type] = stat_value
            
            player_stats_data.append(player_entry)
        
        # Process team officials
        for official in line_up.get('teamOfficial', []):
            official_entry = {
                'Match ID': match_info.get('id', 'N/A'),
                'Competition ID': competition_info.get('id', 'N/A'),
                'Competition Name': competition_info.get('name', 'N/A'),
                'Season': season,
                'Week': match_info.get('week', 'N/A'),
                'Stage ID': stage_info.get('id', 'N/A'),
                'Stage Name': stage_info.get('name', 'N/A'),
                'Team ID': team_id,
                'Team Name': team_name,
                'Official ID': official.get('id', 'N/A'),
                'First Name': official.get('firstName', 'N/A'),
                'Last Name': official.get('lastName', 'N/A'),
                'Short First Name': official.get('shortFirstName', 'N/A'),
                'Short Last Name': official.get('shortlastName', 'N/A'),
                'Known Name': official.get('knownName', 'N/A'),
                'Type': official.get('type', 'N/A')
            }
            team_officials_data.append(official_entry)
    
    df_player_stats = pd.DataFrame(player_stats_data).fillna(0)
    df_team_officials = pd.DataFrame(team_officials_data).fillna('N/A')
    
    # Convert stat columns to numeric
    non_stat_cols = ['Match ID', 'Competition ID', 'Competition Name', 'Season', 'Week', 'Stage ID', 'Stage Name', 
                 'Team ID', 'Team Name', 'Team Position', 'Is Home', 'Is Away', 
                 'HT Home Score', 'HT Away Score', 'FT Home Score', 'FT Away Score',
                 'Player ID', 'First Name', 'Last Name', 'Match Name',
                 'Shirt Number', 'Position', 'Position Side', 'Formation Place']
    
    if not df_player_stats.empty:
        stat_cols = [col for col in df_player_stats.columns if col not in non_stat_cols]
        for col in stat_cols:
            df_player_stats[col] = pd.to_numeric(df_player_stats[col], errors='coerce').fillna(0).astype(float)
    
    return df_player_stats, df_team_officials

def process_match_team_stats(match_id, season):
    """Process MA2 Team Stats - CORREGIDA"""
    request_parameters = {
        "_fmt": "json",
        "detailed": "yes",
        "fx": match_id,
        "_rt": "b"
    }
    
    sdapi_get_url = f'https://api.performfeeds.com/soccerdata/matchstats/{OPTA_API_KEY}/'
    response = _get_opta_session().get(sdapi_get_url, headers=request_headers(), params=request_parameters, timeout=API_TIMEOUT)
    
    if response.status_code != 200:
        print(f"❌ Error MA2 Team Stats: {response.status_code}")
        return pd.DataFrame()
    
    data = response.json()
    match_info = data.get('matchInfo', {})
    competition_info = match_info.get('competition', {})
    stage_info = match_info.get('stage', {})
    live_data = data.get('liveData', {})
    home_away_info = get_home_away_info(match_info, live_data)
    line_ups = live_data.get('lineUp', [])
    
    team_stats_data = []
    
    for team_stats in line_ups:
        team_id = team_stats['contestantId']
        team_name = home_away_info['team_mapping'].get(team_id, {}).get('name', 'N/A')
        team_position = home_away_info['team_mapping'].get(team_id, {}).get('position', 'N/A')
        is_home = team_id == home_away_info['home_team_id']
        is_away = team_id == home_away_info['away_team_id']
        
        for stat in team_stats['stat']:
            stat_info = {
                'Match ID': match_info.get('id', 'N/A'),
                'Competition ID': competition_info.get('id', 'N/A'),
                'Competition Name': competition_info.get('name', 'N/A'),
                'Season': season,
                'Week': match_info.get('week', 'N/A'),
                'Stage ID': stage_info.get('id', 'N/A'),
                'Stage Name': stage_info.get('name', 'N/A'),
                'Team ID': team_id,
                'Team Name': team_name,
                'Team Position': team_position,
                'Is Home': is_home,
                'Is Away': is_away,
                'HT Home Score': home_away_info['ht_home'],
                'HT Away Score': home_away_info['ht_away'],
                'FT Home Score': home_away_info['ft_home'],
                'FT Away Score': home_away_info['ft_away'],
                'Stat Type': stat.get('type', 'N/A'),
                'Total': stat.get('value', 0)
            }
            team_stats_data.append(stat_info)
    
    df_team_stats = pd.DataFrame(team_stats_data)
    if not df_team_stats.empty:
        df_team_stats = df_team_stats.pivot(
            index=['Team ID', 'Team Name', 'Team Position', 'Is Home', 'Is Away', 
                   'HT Home Score', 'HT Away Score', 'FT Home Score', 'FT Away Score',
                   'Match ID', 'Competition ID', 'Competition Name', 'Season', 'Week', 'Stage ID', 'Stage Name'],
            columns='Stat Type', 
            values='Total'
        ).reset_index()
        df_team_stats = df_team_stats.fillna(0)
        
        # Convert stat columns to float
        metadata_cols = ['Team ID', 'Team Name', 'Team Position', 'Is Home', 'Is Away', 
                        'HT Home Score', 'HT Away Score', 'FT Home Score', 'FT Away Score',
                        'Match ID', 'Competition ID', 'Competition Name', 'Season', 'Week', 'Stage ID', 'Stage Name']
        stat_cols = [col for col in df_team_stats.columns if col not in metadata_cols]
        for col in stat_cols:
            df_team_stats[col] = pd.to_numeric(df_team_stats[col], errors='coerce').fillna(0).astype(float)
    
    return df_team_stats

def get_home_away_info(match_info, live_data):
    """Extrae información de home/away del partido"""
    contestants = match_info.get('contestant', [])
    match_details = live_data.get('matchDetails', {})
    
    # Crear mapeo de equipos
    team_mapping = {}
    home_team_id = None
    away_team_id = None
    
    for contestant in contestants:
        team_id = contestant.get('id')
        team_name = contestant.get('name', 'N/A')
        position = contestant.get('position', '')
        
        team_mapping[team_id] = {
            'name': team_name,
            'position': position
        }
        
        if position.lower() == 'home':
            home_team_id = team_id
        elif position.lower() == 'away':
            away_team_id = team_id
    
    # Obtener marcadores si están disponibles
    scores = match_details.get('scores', {})
    ht_scores = scores.get('ht', {})
    ft_scores = scores.get('ft', {})
    
    return {
        'home_team_id': home_team_id,
        'away_team_id': away_team_id,
        'team_mapping': team_mapping,
        'ht_home': ht_scores.get('home', None),
        'ht_away': ht_scores.get('away', None),
        'ft_home': ft_scores.get('home', None),
        'ft_away': ft_scores.get('away', None)
    }

def get_all_competitions_and_stages():
    """Obtiene TODAS las competiciones y temporadas disponibles"""
    print("🔄 Obteniendo todas las competiciones disponibles...")
    
    # Primero probamos sin filtros para obtener todo
    requestParameters = {
        "_fmt": "json",
        "_pgSz": "500",
        "_pgNm": "1",
        "live": "yes",
        "_rt": "b"
    }
    
    sdapi_get_url = f'https://api.performfeeds.com/soccerdata/match/{OPTA_API_KEY}/'  

    try:
        response = _get_opta_session().get(sdapi_get_url, headers=request_headers(), params=requestParameters, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        matches = data.get('match', [])
        
        competitions = {}
        for match in matches:
            match_info = match.get('matchInfo', {})
            competition = match_info.get('competition', {})
            stage = match_info.get('stage', {})
            
            comp_id = competition.get('id')
            comp_name = competition.get('name', 'N/A')
            stage_id = stage.get('id')
            stage_name = stage.get('name', 'N/A')
            start_date = stage.get('startDate', '')
            end_date = stage.get('endDate', '')
            
            if comp_id and stage_id:
                # Convertir fechas a formato temporada
                season = convert_dates_to_season(start_date, end_date)
                
                if comp_id not in competitions:
                    competitions[comp_id] = {
                        'name': comp_name,
                        'stages': {}
                    }
                
                if stage_id not in competitions[comp_id]['stages']:
                    competitions[comp_id]['stages'][stage_id] = {
                        'name': stage_name,
                        'start_date': start_date,
                        'end_date': end_date,
                        'season': season
                    }
        
        return competitions
        
    except Exception as e:
        print(f"   ❌ Error obteniendo competiciones: {e}")
        return {}

def convert_dates_to_season(start_date, end_date):
    """Convierte fechas start/end a formato temporada (24/25, 23/24, etc.)"""
    try:
        if not start_date or not end_date:
            return 'N/A'
            
        # Limpiar las fechas (quitar Z si existe)
        start_clean = start_date.replace('Z', '') if 'Z' in start_date else start_date
        end_clean = end_date.replace('Z', '') if 'Z' in end_date else end_date
        
        # Extraer años de las fechas
        start_year = int(start_clean[:4])
        end_year = int(end_clean[:4])
        
        # Si el año final es mayor, usar formato temporada
        if end_year > start_year:
            return f"{str(start_year)[2:]}/{str(end_year)[2:]}"
        else:
            # Si es el mismo año, usar solo ese año
            return f"{str(start_year)[2:]}"
            
    except Exception as e:
        print(f"   ⚠️ Error convirtiendo fechas {start_date} - {end_date}: {e}")
        return 'N/A'


def process_match_events(match_id, season):
    """Process MA3 Match Events - VERSIÓN ROBUSTA ANTI-404"""
    import logging
    
    logging.info(f"🚀 Iniciando procesamiento de eventos para partido {match_id}")
    logging.info(f"📊 Season detectada: {season}")
    
    # Parámetros estándar para MA3
    request_parameters = {
        "_fmt": "json",
        "fx": match_id,
        "_rt": "b"
    }
    
    sdapi_get_url = f'https://api.performfeeds.com/soccerdata/matchevent/{OPTA_API_KEY}/'
    
    try:
        # Obtener headers
        headers = request_headers()
        
        # Hacer la request
        start_time = time.time()
        response = _get_opta_session().get(sdapi_get_url, headers=headers, params=request_parameters, timeout=API_TIMEOUT)
        response_time = time.time() - start_time
        
        logging.info(f"⏱️ Tiempo de respuesta: {response_time:.2f} segundos")
        
        # ==========================================
        # MANEJO DE ERRORES (400, 404, etc.)
        # ==========================================
        
        if response.status_code == 404:
            # Caso típico en Copa del Rey: El partido existe pero no tiene feed de eventos (MA3)
            try:
                error_data = response.json()
                error_code = error_data.get('errorCode', 'N/A')
                
                if error_code == "10400":
                    logging.warning(f"⚠️ El partido {match_id} no tiene datos de eventos (MA3).")
                    logging.warning(f"   ℹ️ Esto es normal en rondas previas de Copa o partidos no televisados.")
                    return pd.DataFrame() # Devolvemos vacío y continuamos
            except:
                pass
            
            logging.error(f"❌ ERROR 404 no controlado en partido {match_id}")
            logging.error(f"📄 Response: {response.text}")
            return pd.DataFrame()

        elif response.status_code == 400:
            logging.error(f"❌ ERROR 400 - Verificando causa...")
            try:
                error_data = response.json()
                error_code = error_data.get('errorCode', 'N/A')
                
                if error_code == "10217":
                    logging.error("🔑 ERROR 10217: Feed MA3 no autorizado en tu suscripción")
                    return pd.DataFrame()
                elif error_code == "10203":
                    logging.error("🔧 ERROR 10203: Parámetros inválidos")
                    return pd.DataFrame()
                else:
                    logging.error(f"❓ Error desconocido: {error_code}")
            except:
                logging.error(f"📄 Respuesta no JSON: {response.text}")
            return pd.DataFrame()
        
        elif response.status_code != 200:
            logging.error(f"❌ ERROR {response.status_code}")
            logging.error(f"📄 Response: {response.text}")
            return pd.DataFrame()
        
        # ==========================================
        # PROCESAMIENTO DE DATOS (200 OK)
        # ==========================================
        
        try:
            data = response.json()
            match_info = data.get('matchInfo', {})
            live_data = data.get('liveData', {})
            
            # Obtener eventos
            events = live_data.get('event', [])
            
            if not events:
                logging.warning(f"⚠️ El partido {match_id} devolvió una lista vacía de eventos.")
                return pd.DataFrame()
            
            # Procesar información del partido
            competition_info = match_info.get('competition', {})
            stage_info = match_info.get('stage', {})
            home_away_info = get_home_away_info(match_info, live_data)
            
            logging.info(f"📊 Eventos encontrados: {len(events)}")

            # Encontrar qualifiers únicos para crear columnas
            qualifier_ids = set()
            for event in events:
                for q in event.get('qualifier', []):
                    qid = q.get('qualifierId', '')
                    if qid and str(qid).isdigit():
                        qualifier_ids.add(str(qid))
            
            # Inicializar nombres de columnas
            qualifier_names = []
            for qid in qualifier_ids:
                try:
                    qualifier_name = QUALIFIER_MAPPING.get(int(qid), f'qualifier {qid}')
                    qualifier_names.append(qualifier_name)
                except ValueError:
                    continue
            
            columns = [
                'Match ID', 'Competition ID', 'Competition Name', 'Season', 'Week', 'Stage ID', 'Stage Name',
                'EventId', 'typeId', 'Event Name', 'timeStamp', 'contestantId', 'Team ID', 'Team Name', 
                'Team Position', 'Is Home', 'Is Away', 'HT Home Score', 'HT Away Score', 
                'FT Home Score', 'FT Away Score', 'periodId', 'timeMin', 'timeSec', 
                'playerId', 'playerName', 'outcome', 'x', 'y'
            ] + qualifier_names
            
            # Procesar eventos fila por fila
            events_data = []
            
            for event in events:
                try:
                    contestant_id = event.get('contestantId', None)
                    team_name = home_away_info['team_mapping'].get(contestant_id, {}).get('name', 'N/A') if contestant_id else 'N/A'
                    team_position = home_away_info['team_mapping'].get(contestant_id, {}).get('position', 'N/A') if contestant_id else 'N/A'
                    is_home = contestant_id == home_away_info['home_team_id']
                    is_away = contestant_id == home_away_info['away_team_id']
                    type_id = event.get('typeId', None)
                    event_name = EVENT_TYPE_MAPPING.get(type_id, 'Unknown Event') if type_id else 'Unknown Event'
                    
                    event_info = {
                        'Match ID': match_info.get('id', 'N/A'),
                        'Competition ID': competition_info.get('id', 'N/A'),
                        'Competition Name': competition_info.get('name', 'N/A'),
                        'Season': season,
                        'Week': match_info.get('week', 'N/A'),
                        'Stage ID': stage_info.get('id', 'N/A'),
                        'Stage Name': stage_info.get('name', 'N/A'),
                        'EventId': event.get('eventId', None),
                        'typeId': type_id,
                        'Event Name': event_name,
                        'timeStamp': event.get('timeStamp', None),
                        'contestantId': contestant_id,
                        'Team ID': contestant_id,
                        'Team Name': team_name,
                        'Team Position': team_position,
                        'Is Home': is_home,
                        'Is Away': is_away,
                        'HT Home Score': home_away_info['ht_home'],
                        'HT Away Score': home_away_info['ht_away'],
                        'FT Home Score': home_away_info['ft_home'],
                        'FT Away Score': home_away_info['ft_away'],
                        'periodId': event.get('periodId', None),
                        'timeMin': event.get('timeMin', None),
                        'timeSec': event.get('timeSec', None),
                        'playerId': event.get('playerId', None),
                        'playerName': event.get('playerName', None),
                        'outcome': event.get('outcome', None),
                        'x': event.get('x', None),
                        'y': event.get('y', None),
                    }
                    
                    # 1. Inicializar todos los qualifiers de este partido a "No"
                    for qname in qualifier_names:
                        event_info[qname] = "No"

                    # 2. Rellenar con los valores reales del evento actual
                    for q in event.get('qualifier', []):
                        try:
                            qualifier_id = int(q["qualifierId"])
                            qualifier_name = QUALIFIER_MAPPING.get(qualifier_id, f'qualifier {qualifier_id}')
                            
                            if 'value' in q and q['value'] is not None:
                                event_info[qualifier_name] = str(q['value'])
                            else:
                                event_info[qualifier_name] = "Sí"
                        except (ValueError, KeyError):
                            continue
                    
                    events_data.append(event_info)
                    
                except Exception as e:
                    logging.warning(f"⚠️ Error procesando un evento individual: {e}")
                    continue
            
            # Crear DataFrame
            df = pd.DataFrame(events_data, columns=columns)
            logging.info(f"✅ DataFrame creado: {len(df)} filas")
            return df
                
        except json.JSONDecodeError as e:
            logging.error(f"❌ Error parseando JSON: {e}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"❌ Error procesando estructura de datos: {e}")
            return pd.DataFrame()
            
    except requests.Timeout:
        logging.error(f"⏰ Timeout en request para partido {match_id}")
        return pd.DataFrame()
    except requests.RequestException as e:
        logging.error(f"📡 Error de conexión para partido {match_id}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"💥 Error inesperado en partido {match_id}: {e}", exc_info=True)
        return pd.DataFrame()


def process_possession_events(match_id, season):
    """Process MA13 Possession Events - Similar a MA3 pero con campos de posesión"""
    import logging

    logging.info(f"🚀 Iniciando procesamiento de posesiones para partido {match_id}")

    # Parámetros estándar para MA13
    request_parameters = {
        "_fmt": "json",
        "fx": match_id,
        "_rt": "b"
    }

    sdapi_get_url = f'https://api.performfeeds.com/soccerdata/possessionevent/{OPTA_API_KEY}/'

    try:
        headers = request_headers()
        start_time = time.time()
        response = _get_opta_session().get(sdapi_get_url, headers=headers, params=request_parameters, timeout=API_TIMEOUT)
        response_time = time.time() - start_time

        logging.info(f"⏱️ Tiempo de respuesta MA13: {response_time:.2f} segundos")

        # ==========================================
        # MANEJO DE ERRORES (400, 404, etc.)
        # ==========================================

        if response.status_code == 404:
            try:
                error_data = response.json()
                error_code = error_data.get('errorCode', 'N/A')

                if error_code == "10400":
                    logging.warning(f"⚠️ El partido {match_id} no tiene datos de posesiones (MA13).")
                    return pd.DataFrame()
            except:
                pass

            logging.error(f"❌ ERROR 404 no controlado en partido {match_id} (MA13)")
            return pd.DataFrame()

        elif response.status_code == 400:
            logging.error(f"❌ ERROR 400 en MA13 - Verificando causa...")
            try:
                error_data = response.json()
                error_code = error_data.get('errorCode', 'N/A')

                if error_code == "10217":
                    logging.error("🔑 ERROR 10217: Feed MA13 no autorizado en tu suscripción")
                    return pd.DataFrame()
                elif error_code == "10203":
                    logging.error("🔧 ERROR 10203: Parámetros inválidos")
                    return pd.DataFrame()
                else:
                    logging.error(f"❓ Error desconocido: {error_code}")
            except:
                logging.error(f"📄 Respuesta no JSON: {response.text}")
            return pd.DataFrame()

        elif response.status_code != 200:
            logging.error(f"❌ ERROR {response.status_code} en MA13")
            return pd.DataFrame()

        # ==========================================
        # PROCESAMIENTO DE DATOS (200 OK)
        # ==========================================

        try:
            data = response.json()
            match_info = data.get('matchInfo', {})
            live_data = data.get('liveData', {})

            events = live_data.get('event', [])

            if not events:
                logging.warning(f"⚠️ El partido {match_id} devolvió una lista vacía de posesiones.")
                return pd.DataFrame()

            competition_info = match_info.get('competition', {})
            stage_info = match_info.get('stage', {})
            home_away_info = get_home_away_info(match_info, live_data)

            logging.info(f"📊 Eventos de posesión encontrados: {len(events)}")

            # Encontrar qualifiers únicos para crear columnas
            qualifier_ids = set()
            for event in events:
                for q in event.get('qualifier', []):
                    qid = q.get('qualifierId', '')
                    if qid and str(qid).isdigit():
                        qualifier_ids.add(str(qid))

            qualifier_names = []
            for qid in qualifier_ids:
                try:
                    qualifier_name = QUALIFIER_MAPPING.get(int(qid), f'qualifier {qid}')
                    qualifier_names.append(qualifier_name)
                except ValueError:
                    continue

            # Columnas base + campos específicos de MA13
            columns = [
                'Match ID', 'Competition ID', 'Competition Name', 'Season', 'Week', 'Stage ID', 'Stage Name',
                'EventId', 'typeId', 'Event Name', 'timeStamp', 'contestantId', 'Team ID', 'Team Name',
                'Team Position', 'Is Home', 'Is Away', 'HT Home Score', 'HT Away Score',
                'FT Home Score', 'FT Away Score', 'periodId', 'timeMin', 'timeSec',
                'playerId', 'playerName', 'outcome', 'x', 'y',
                'sequenceId', 'possessionNumber'  # Campos específicos MA13
            ] + qualifier_names

            events_data = []

            for event in events:
                try:
                    contestant_id = event.get('contestantId', None)
                    team_name = home_away_info['team_mapping'].get(contestant_id, {}).get('name', 'N/A') if contestant_id else 'N/A'
                    team_position = home_away_info['team_mapping'].get(contestant_id, {}).get('position', 'N/A') if contestant_id else 'N/A'
                    is_home = contestant_id == home_away_info['home_team_id']
                    is_away = contestant_id == home_away_info['away_team_id']
                    type_id = event.get('typeId', None)
                    event_name = EVENT_TYPE_MAPPING.get(type_id, 'Unknown Event') if type_id else 'Unknown Event'

                    event_info = {
                        'Match ID': match_info.get('id', 'N/A'),
                        'Competition ID': competition_info.get('id', 'N/A'),
                        'Competition Name': competition_info.get('name', 'N/A'),
                        'Season': season,
                        'Week': match_info.get('week', 'N/A'),
                        'Stage ID': stage_info.get('id', 'N/A'),
                        'Stage Name': stage_info.get('name', 'N/A'),
                        'EventId': event.get('eventId', None),
                        'typeId': type_id,
                        'Event Name': event_name,
                        'timeStamp': event.get('timeStamp', None),
                        'contestantId': contestant_id,
                        'Team ID': contestant_id,
                        'Team Name': team_name,
                        'Team Position': team_position,
                        'Is Home': is_home,
                        'Is Away': is_away,
                        'HT Home Score': home_away_info['ht_home'],
                        'HT Away Score': home_away_info['ht_away'],
                        'FT Home Score': home_away_info['ft_home'],
                        'FT Away Score': home_away_info['ft_away'],
                        'periodId': event.get('periodId', None),
                        'timeMin': event.get('timeMin', None),
                        'timeSec': event.get('timeSec', None),
                        'playerId': event.get('playerId', None),
                        'playerName': event.get('playerName', None),
                        'outcome': event.get('outcome', None),
                        'x': event.get('x', None),
                        'y': event.get('y', None),
                        'sequenceId': event.get('sequenceId', None),
                        'possessionNumber': event.get('possessionNumber', None),
                    }

                    # Inicializar todos los qualifiers a "No"
                    for qname in qualifier_names:
                        event_info[qname] = "No"

                    # Rellenar con los valores reales del evento actual
                    for q in event.get('qualifier', []):
                        try:
                            qualifier_id = int(q["qualifierId"])
                            qualifier_name = QUALIFIER_MAPPING.get(qualifier_id, f'qualifier {qualifier_id}')

                            if 'value' in q and q['value'] is not None:
                                event_info[qualifier_name] = str(q['value'])
                            else:
                                event_info[qualifier_name] = "Sí"
                        except (ValueError, KeyError):
                            continue

                    events_data.append(event_info)

                except Exception as e:
                    logging.warning(f"⚠️ Error procesando evento de posesión: {e}")
                    continue

            df = pd.DataFrame(events_data, columns=columns)
            logging.info(f"✅ DataFrame posesiones creado: {len(df)} filas")
            return df

        except json.JSONDecodeError as e:
            logging.error(f"❌ Error parseando JSON MA13: {e}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"❌ Error procesando estructura MA13: {e}")
            return pd.DataFrame()

    except requests.Timeout:
        logging.error(f"⏰ Timeout en request MA13 para partido {match_id}")
        return pd.DataFrame()
    except requests.RequestException as e:
        logging.error(f"📡 Error de conexión MA13 para partido {match_id}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"💥 Error inesperado en MA13 partido {match_id}: {e}", exc_info=True)
        return pd.DataFrame()


# 3. FUNCIÓN DE DIAGNÓSTICO RÁPIDO
def diagnostico_rapido_eventos(match_id_sample):
    """Diagnóstico rápido para un partido específico"""
    logging.info(f"🔍 DIAGNÓSTICO RÁPIDO - Match ID: {match_id_sample}")
    
    # Probar diferentes combinaciones de parámetros
    parameter_combinations = [
        {"_fmt": "json", "fx": match_id_sample, "_rt": "b"},
        {"_fmt": "json", "fx": match_id_sample, "_rt": "b", "detailed": "yes"},
        {"_fmt": "json", "fx": match_id_sample, "_rt": "b", "detailed": "yes", "live": "yes"},
        {"_fmt": "json", "fx": match_id_sample, "_rt": "b", "detailed": "yes", "_pgSz": "all"},
    ]
    
    for i, params in enumerate(parameter_combinations, 1):
        try:
            response = _get_opta_session().get(
                f'https://api.performfeeds.com/soccerdata/matchevent/{OPTA_API_KEY}/',
                headers=request_headers(),
                params=params,
                timeout=API_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                live_data = data.get('liveData', {})
                main_events = live_data.get('event', [])
                
                logging.info(f"   Combinación {i}: {len(main_events)} eventos")
                logging.info(f"   Parámetros: {params}")
                
                # Guardar la mejor respuesta
                if len(main_events) > 0:
                    with open(f'best_response_{match_id_sample}.json', 'w') as f:
                        json.dump(data, f, indent=2)
                    break
            else:
                logging.error(f"   Combinación {i}: Error {response.status_code}")
                
            time.sleep(1)  # Evitar rate limiting
            
        except Exception as e:
            logging.error(f"   Combinación {i}: Exception {e}")

# 4. FUNCIÓN PARA EJECUTAR AL FINAL DEL SCRIPT
def ejecutar_diagnosticos():
    """Ejecutar diagnósticos automáticos"""
    try:
        # Obtener un match ID existente
        df_sample = pd.read_parquet(OPTA_PATH / 'player_stats.parquet')
        match_id_sample = df_sample['Match ID'].iloc[0]
        
        logging.info("🔧 EJECUTANDO DIAGNÓSTICOS AUTOMÁTICOS")
        diagnostico_rapido_eventos(match_id_sample)
        verificar_completitud_eventos([match_id_sample])
        
    except Exception as e:
        logging.error(f"Error en diagnósticos: {e}")
        
# ✅ FUNCIÓN ADICIONAL: Verificar completitud de datos
def verificar_completitud_eventos(match_ids_sample=None):
    """Verifica la completitud de los datos de eventos"""
    if not match_ids_sample:
        # Obtener algunos match IDs existentes
        try:
            df_sample = pd.read_parquet(OPTA_PATH / 'player_stats.parquet')
            match_ids_sample = df_sample['Match ID'].unique()[:5]  # Solo 5 para prueba
        except:
            logging.error("No se pudieron obtener match IDs para verificación")
            return
    
    logging.info("🔍 VERIFICACIÓN DE COMPLETITUD DE EVENTOS")
    logging.info("=" * 50)
    
    for match_id in match_ids_sample:
        logging.info(f"\n🔎 Verificando partido: {match_id}")
        
        # Obtener eventos directamente de API
        request_parameters = {
            "_fmt": "json",
            "fx": match_id,
            "_rt": "b"
        }
        
        try:
            response = _get_opta_session().get(
                f'https://api.performfeeds.com/soccerdata/matchevent/{OPTA_API_KEY}/',
                headers=request_headers(),
                params=request_parameters,
                timeout=API_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                api_events = data.get('liveData', {}).get('event', [])
                
                # Verificar en archivo parquet
                try:
                    df_stored = pd.read_parquet(OPTA_PATH / 'match_events.parquet')
                    stored_events = df_stored[df_stored['Match ID'] == match_id]
                    
                    logging.info(f"   📊 API: {len(api_events)} eventos")
                    logging.info(f"   💾 Archivo: {len(stored_events)} eventos")
                    
                    if len(api_events) != len(stored_events):
                        logging.warning(f"   ⚠️ DISCREPANCIA: Faltan {len(api_events) - len(stored_events)} eventos")
                    else:
                        logging.info(f"   ✅ Completitud verificada")
                        
                except Exception as e:
                    logging.error(f"   ❌ Error leyendo archivo: {e}")
            else:
                logging.error(f"   ❌ Error API: {response.status_code}")
                
        except Exception as e:
            logging.error(f"   ❌ Error verificando partido: {e}")
        
        time.sleep(2)  # Evitar rate limiting

def update_mediacoach_data_web(liga, temporada, j_inicio, j_fin, progress_callback=None):
    """Ejecuta el script descarga_completa.py y captura su progreso"""
    import subprocess
    import sys
    import os

    # 1. Construir la ruta al script orquestador
    # Asumimos que app.py está en la raíz
    base_path = os.getcwd()
    script_path = os.path.join(base_path, 'extraccion_mediacoach', 'descarga_completa.py')
    
    # 2. Comando exacto con los argumentos que espera tu nuevo descarga_completa.py
    cmd = [
        sys.executable, script_path,
        '--liga', str(liga),
        '--temporada', str(temporada),
        '--j_inicio', str(j_inicio),
        '--j_fin', str(j_fin)
    ]

    print(f"🚀 Lanzando orquestador: {' '.join(cmd)}")

    # 3. Ejecutar y leer la salida en tiempo real
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    messages = []
    
    # Leemos línea a línea lo que imprime tu script (los PROGRESS:XX que pusimos)
    for line in process.stdout:
        clean_line = line.strip()
        if not clean_line: continue
        
        print(f"STDOUT: {clean_line}") # Para ver en la terminal del Mac
        
        # Detectar el porcentaje para la barra de progreso
        progreso_actual = 0
        if "PROGRESS:" in clean_line:
            try:
                # Extrae el número entre PROGRESS: y -
                progreso_actual = int(clean_line.split("PROGRESS:")[1].split("-")[0])
            except: pass
        
        # Enviar a la web
        if progress_callback:
            messages.append({'timestamp': datetime.now().strftime("%H:%M:%S"), 'message': clean_line})
            # Limitamos a los últimos 50 mensajes para no saturar la web
            progress_callback(progreso_actual, clean_line, messages[-50:])

    process.wait()

    # Auto-commit y push a GitHub
    print("📤 Subiendo cambios de MediaCoach a GitHub...")
    git_auto_sync("MediaCoach")

    if progress_callback:
        progress_callback(100, "✅ Proceso MediaCoach finalizado con éxito.", messages)

def update_opta_data_web(competition_id, stage_id, start_week, end_week, progress_callback=None):
    """Versión web INTELIGENTE: Detecta si es Copa para ignorar las jornadas"""
    messages = []

    def add_message(msg, msg_type="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        messages.append({
            'timestamp': timestamp,
            'message': msg,
            'type': msg_type
        })
        print(msg)

    def update_progress(progress, status=""):
        if progress_callback:
            progress_callback(progress, status, messages)

    try:
        return _update_opta_data_web_inner(competition_id, stage_id, start_week, end_week, add_message, update_progress, messages, progress_callback)
    except Exception as e:
        add_message(f"💥 Error crítico en la actualización: {e}", "error")
        update_progress(100, f"Error crítico: {e}")
        return messages

def _update_opta_data_web_inner(competition_id, stage_id, start_week, end_week, add_message, update_progress, messages, progress_callback=None):
    """Lógica interna de update_opta_data_web"""
    add_message("🎯 ACTUALIZACIÓN DE DATOS OPTA (WEB)")
    add_message("=" * 50)
    update_progress(5, "Iniciando proceso...")
    
    # 1. Obtener información de la competición para saber si es LIGA o COPA
    # Necesitamos el nombre para saber cómo comportarnos
    add_message("🔍 Analizando tipo de competición...")
    is_cup_format = False
    season = "N/A"
    
    try:
        # Obtenemos las competiciones para sacar el nombre y la temporada
        competitions = get_all_competitions_and_stages()
        comp_name = "Desconocida"
        
        # Buscamos la info correspondiente a los IDs recibidos de la web
        for c_id, c_info in competitions.items():
            if c_id == competition_id:
                if stage_id in c_info['stages']:
                    season = c_info['stages'][stage_id]['season']
                    comp_name = c_info.get('name', '').lower()
                    break
        
        # LÓGICA DE DETECCIÓN
        # Si NO contiene palabras de liga regular, asumimos que es formato Copa/Torneo
        # Normalizamos quitando acentos para evitar fallos (ej: "división" vs "division")
        comp_name_normalized = ''.join(
            c for c in unicodedata.normalize('NFD', comp_name) if unicodedata.category(c) != 'Mn'
        )
        keywords_liga = ['liga', 'division', 'premier', 'serie a', 'bundesliga', 'regular season', 'primera']
        is_cup_format = not any(k in comp_name_normalized for k in keywords_liga)
        
        add_message(f"🏆 Competición detectada: {comp_name.upper()}")
        add_message(f"🗓️ Temporada: {season}")
        
        if is_cup_format:
            add_message(f"ℹ️ Formato detectado: COPA/ELIMINATORIA")
            add_message(f"   👉 Se ignorará la selección de jornadas {start_week}-{end_week}.")
            add_message(f"   👉 Se descargarán TODOS los partidos de esta fase.")
        else:
            add_message(f"ℹ️ Formato detectado: LIGA REGULAR")
            add_message(f"   👉 Descargando jornadas: {start_week} a {end_week}")

    except Exception as e:
        add_message(f"⚠️ No se pudo determinar el tipo de competición: {e}", "warning")
        # Si falla la detección, seguimos con la lógica por defecto (Liga)

    add_message("=" * 50)

    # 2. Get matches (Lógica diferenciada)
    add_message("🔄 Obteniendo partidos...")
    update_progress(20, "Conectando con API...")
    
    all_matches_df = pd.DataFrame()
    
    if is_cup_format:
        # SI ES COPA: Llamamos directamente a advanced SIN semana específica
        # Esto evita el error 404 de "Week 1"
        matches_df = get_match_ids_advanced(
            max_matches=500, # Pedimos muchos para traer toda la fase
            specific_week=None, # <--- CLAVE: Ignoramos la semana que viene de la web
            stage_id=stage_id,
            messages=messages
        )
        if not matches_df.empty:
            all_matches_df = matches_df
            add_message(f"✅ Descarga completa de la fase: {len(all_matches_df)} partidos.")
    else:
        # SI ES LIGA: Usamos el rango de semanas seleccionado en la web
        all_matches_df = get_matches_by_weeks_range(
            stage_id, start_week, end_week, 
            messages=messages, progress_callback=progress_callback
        )
    
    if all_matches_df.empty:
        add_message("❌ No se encontraron partidos", "error")
        update_progress(100, "Error: No se encontraron partidos")
        return messages

    # 4. Verificación granular de datos existentes
    update_progress(30, "Verificando datos existentes por partido...")
    match_ids = all_matches_df['Match ID'].tolist()
    data_status = get_granular_data_status(match_ids, messages)

    # Determinar qué partidos necesitan alguna descarga
    matches_needing_data = []
    for match_id in match_ids:
        status = data_status.get(match_id, {})
        if not all(status.values()):
            matches_needing_data.append(match_id)

    if not matches_needing_data:
        add_message("🎉 ¡No hay datos nuevos que descargar!", "success")
        update_progress(95, "Verificando ABP...")
        update_abp_events_standalone()
        calculate_and_save_abp_statistics()
        update_progress(100, "Completado: Todos los datos están actualizados.")
        return messages

    add_message(f"📊 {len(matches_needing_data)} partidos necesitan datos")

    # 5. Process data loop - DESCARGA GRANULAR
    all_data = {
        'player_stats': [], 'team_stats': [], 'player_xg_stats': [],
        'xg_events': [], 'match_events': [], 'team_officials': [],
        'posesiones': []
    }

    progress_increment = 55 / len(matches_needing_data)

    for i, match_id in enumerate(matches_needing_data):
        current_progress = 30 + (i * progress_increment)
        update_progress(current_progress, f"Procesando partido {i+1}/{len(matches_needing_data)}")

        status = data_status.get(match_id, {})
        missing_feeds = [k for k, v in status.items() if not v]
        add_message(f"⚽ Partido {i+1}/{len(matches_needing_data)}: {match_id}")
        add_message(f"   📥 Feeds faltantes: {', '.join(missing_feeds)}")

        try:
            # MA2 - Player Stats y Team Officials (se descargan juntos)
            if not status.get('player_stats') or not status.get('team_officials'):
                p_stats, t_off = process_match_player_stats(match_id, season)
                if not p_stats.empty:
                    all_data['player_stats'].append(p_stats)
                if not t_off.empty:
                    all_data['team_officials'].append(t_off)

            # MA2 - Team Stats
            if not status.get('team_stats'):
                t_stats = process_match_team_stats(match_id, season)
                if not t_stats.empty:
                    all_data['team_stats'].append(t_stats)

            # MA3 - Match Events
            if not status.get('match_events'):
                m_events = process_match_events(match_id, season)
                if not m_events.empty:
                    all_data['match_events'].append(m_events)

            # MA12 - xG Player Stats
            if not status.get('player_xg_stats'):
                p_xg = process_xg_player_stats(match_id, season)
                if not p_xg.empty:
                    all_data['player_xg_stats'].append(p_xg)

            # MA12 - xG Events
            if not status.get('xg_events'):
                xg_ev = process_xg_events(match_id, season)
                if not xg_ev.empty:
                    all_data['xg_events'].append(xg_ev)

            # MA13 - Possession Events (NUEVO)
            if not status.get('posesiones'):
                pos_events = process_possession_events(match_id, season)
                if not pos_events.empty:
                    all_data['posesiones'].append(pos_events)

        except Exception as e:
            add_message(f"❌ Error en partido {match_id}: {e}", "error")
            continue

        if i < len(matches_needing_data) - 1:
            time.sleep(DELAY_SECONDS)
    
    # 6. Save Data
    update_progress(85, "Guardando datos...")
    new_data = {}
    for data_type, df_list in all_data.items():
        if df_list:
            new_data[data_type] = pd.concat(df_list, ignore_index=True)
        else:
            new_data[data_type] = pd.DataFrame()
    
    save_opta_data(new_data, messages)
    
    # 7. ABP Processing
    update_progress(95, "Procesando ABP...")
    try:
        update_abp_events_standalone()
        calculate_and_save_abp_statistics()
    except Exception as e:
        add_message(f"⚠️ Error procesando ABP: {e}", "warning")
    
    add_message("🎉 ¡Actualización completada!", "success")
    update_progress(100, "¡Actualización completada!")

    # Auto-commit y push a GitHub
    add_message("📤 Subiendo cambios a GitHub...")
    git_auto_sync("Opta")

    return messages

def process_xg_player_stats(match_id, season):
    """Process MA12 Player xG Stats - FUNCIÓN QUE FALTABA"""
    request_parameters = {
        "_fmt": "json",
        "fx": match_id,
        "_rt": "b"
    }
    
    sdapi_get_url = f'https://api.performfeeds.com/soccerdata/matchexpectedgoals/{OPTA_API_KEY}/'
    response = _get_opta_session().get(sdapi_get_url, headers=request_headers(), params=request_parameters, timeout=API_TIMEOUT)
    
    if response.status_code != 200:
        print(f"❌ Error MA12 xG Player Stats: {response.status_code}")
        return pd.DataFrame()
    
    data = response.json()
    match_info = data.get('matchInfo', {})
    competition_info = match_info.get('competition', {})
    stage_info = match_info.get('stage', {})
    live_data = data.get('liveData', {})
    home_away_info = get_home_away_info(match_info, live_data)
    line_ups = live_data.get('lineUp', [])
    
    player_xg_data = []
    
    for line_up in line_ups:
        team_id = line_up.get('contestantId')
        team_name = home_away_info['team_mapping'].get(team_id, {}).get('name', 'N/A')
        team_position = home_away_info['team_mapping'].get(team_id, {}).get('position', 'N/A')
        is_home = team_id == home_away_info['home_team_id']
        is_away = team_id == home_away_info['away_team_id']
        
        for player in line_up.get('player', []):
            player_entry = {
                'Match ID': match_info.get('id', 'N/A'),
                'Competition ID': competition_info.get('id', 'N/A'),
                'Competition Name': competition_info.get('name', 'N/A'),
                'Season': season,
                'Week': match_info.get('week', 'N/A'),
                'Stage ID': stage_info.get('id', 'N/A'),
                'Stage Name': stage_info.get('name', 'N/A'),
                'Team ID': team_id,
                'Team Name': team_name,
                'Team Position': team_position,
                'Is Home': is_home,
                'Is Away': is_away,
                'HT Home Score': home_away_info['ht_home'],
                'HT Away Score': home_away_info['ht_away'],
                'FT Home Score': home_away_info['ft_home'],
                'FT Away Score': home_away_info['ft_away'],
                'Player ID': player.get('playerId', 'N/A'),
                'First Name': player.get('firstName', 'N/A'),
                'Last Name': player.get('lastName', 'N/A'),
                'Match Name': player.get('matchName', 'N/A'),
                'Shirt Number': player.get('shirtNumber', 'N/A'),
                'Position': player.get('position', 'N/A'),
                'Position Side': player.get('positionSide', 'N/A'),
                'Formation Place': player.get('formationPlace', 'N/A'),
            }
            
            for stat in player.get('stat', []):
                stat_type = stat.get('type', '')
                stat_value = stat.get('value', 0)
                player_entry[stat_type] = stat_value
            
            player_xg_data.append(player_entry)
    
    df_player_xg = pd.DataFrame(player_xg_data).fillna(0)
    
    # Convert stat columns to numeric
    non_stat_cols = ['Match ID', 'Competition ID', 'Competition Name', 'Season', 'Week', 'Stage ID', 'Stage Name', 
                 'Team ID', 'Team Name', 'Team Position', 'Is Home', 'Is Away', 
                 'HT Home Score', 'HT Away Score', 'FT Home Score', 'FT Away Score',
                 'Player ID', 'First Name', 'Last Name', 'Match Name',
                 'Shirt Number', 'Position', 'Position Side', 'Formation Place']
    
    if not df_player_xg.empty:
        stat_cols = [col for col in df_player_xg.columns if col not in non_stat_cols]
        for col in stat_cols:
            df_player_xg[col] = pd.to_numeric(df_player_xg[col], errors='coerce').fillna(0).astype(float)
    
    return df_player_xg

def process_xg_events(match_id, season):
    """Process MA12 xG Events - VERSIÓN CORREGIDA"""
    request_parameters = {
        "_rt": "b",
        "_fmt": "json",
        "fx": match_id
    }
    
    sdapi_get_url = f'https://api.performfeeds.com/soccerdata/matchexpectedgoals/{OPTA_API_KEY}/'
    response = _get_opta_session().get(sdapi_get_url, headers=request_headers(), params=request_parameters, timeout=API_TIMEOUT)
    
    if response.status_code != 200:
        print(f"❌ Error MA12 xG Events: {response.status_code}")
        return pd.DataFrame()
    
    data = response.json()
    match_info = data.get('matchInfo', {})
    competition_info = match_info.get('competition', {})
    stage_info = match_info.get('stage', {})
    live_data = data.get('liveData', {})
    home_away_info = get_home_away_info(match_info, live_data)
    xg_events = live_data.get('event', [])
    
    qualifier_ids = {'321', '322'}  # xG specific qualifiers
    
    columns = [
        'Match ID', 'Competition ID', 'Competition Name', 'Season', 'Week', 'Stage ID', 'Stage Name', 'EventId', 'timeStamp', 
        'contestantId', 'Team ID', 'Team Name', 'Team Position', 'Is Home', 'Is Away',
        'HT Home Score', 'HT Away Score', 'FT Home Score', 'FT Away Score',
        'periodId', 'timeMin', 'timeSec', 'playerId', 'playerName', 'typeId', 'Event Name', 'outcome', 'x', 'y'
    ] + [f'qualifier {qid}' for qid in qualifier_ids]
    
    xg_events_data = []
    
    for event in xg_events:
        contestant_id = event.get('contestantId', None)
        team_name = home_away_info['team_mapping'].get(contestant_id, {}).get('name', 'N/A') if contestant_id else 'N/A'
        team_position = home_away_info['team_mapping'].get(contestant_id, {}).get('position', 'N/A') if contestant_id else 'N/A'
        is_home = contestant_id == home_away_info['home_team_id']
        is_away = contestant_id == home_away_info['away_team_id']
        type_id = event.get('typeId', None)
        event_name = EVENT_TYPE_MAPPING.get(type_id, 'Unknown Event') if type_id else 'Unknown Event'
        
        xg_event_info = {
            'Match ID': match_info.get('id', 'N/A'),
            'Competition ID': competition_info.get('id', 'N/A'),
            'Competition Name': competition_info.get('name', 'N/A'),
            'Season': season,
            'Week': match_info.get('week', 'N/A'),
            'Stage ID': stage_info.get('id', 'N/A'),
            'Stage Name': stage_info.get('name', 'N/A'),
            'EventId': event.get('eventId', None),
            'typeId': type_id,
            'Event Name': event_name,
            'periodId': event.get('periodId', None),
            'timeMin': event.get('timeMin', None),
            'timeSec': event.get('timeSec', None),
            'contestantId': contestant_id,
            'Team ID': contestant_id,
            'Team Name': team_name,
            'Team Position': team_position,
            'Is Home': is_home,
            'Is Away': is_away,
            'HT Home Score': home_away_info['ht_home'],
            'HT Away Score': home_away_info['ht_away'],
            'FT Home Score': home_away_info['ft_home'],
            'FT Away Score': home_away_info['ft_away'],
            'playerId': event.get('playerId', None),
            'playerName': event.get('playerName', None),
            'outcome': event.get('outcome', None),
            'x': event.get('x', None),
            'y': event.get('y', None),
            'timeStamp': event.get('timeStamp', None),
        }
        
        # ✅ CORRECCIÓN: Initialize qualifiers to "No" 
        for qid in qualifier_ids:
            xg_event_info[f'qualifier {qid}'] = "No"
        
        # ✅ CORRECCIÓN: Update with actual qualifier values
        for q in event.get('qualifier', []):
            qualifier_id = str(q["qualifierId"])
            if qualifier_id in qualifier_ids:
                if 'value' in q and q['value'] is not None:
                    xg_event_info[f'qualifier {qualifier_id}'] = str(q['value'])
                else:
                    xg_event_info[f'qualifier {qualifier_id}'] = "Sí"
        
        xg_events_data.append(xg_event_info)
    
    return pd.DataFrame(xg_events_data, columns=columns)

def diagnosticar_duplicados(data_dict):
    """Diagnostica problemas de duplicados antes de guardar"""
    logging.info("🔍 DIAGNÓSTICO DE DUPLICADOS")
    
    file_config = {
        'player_stats': ['Match ID', 'Player ID'],
        'team_stats': ['Match ID', 'Team ID'],
        'player_xg_stats': ['Match ID', 'Player ID'],
        'xg_events': ['Match ID', 'EventId', 'timeStamp'],
        'match_events': ['Match ID', 'EventId', 'timeStamp'],
        'team_officials': ['Match ID', 'Official ID'],
        'posesiones': ['Match ID', 'EventId', 'timeStamp']
    }
    
    for data_type, new_df in data_dict.items():
        if new_df.empty or data_type not in file_config:
            continue
            
        keys = file_config[data_type]
        
        # Verificar NaNs en claves
        for key in keys:
            if key in new_df.columns:
                nan_count = new_df[key].isna().sum()
                if nan_count > 0:
                    logging.warning(f"⚠️ {data_type}: {nan_count} filas con {key} = NaN")
        
        # Verificar duplicados internos
        duplicates = new_df.duplicated(subset=keys, keep=False)
        if duplicates.any():
            logging.warning(f"⚠️ {data_type}: {duplicates.sum()} filas duplicadas internamente")
            
        # Mostrar sample de claves
        if not new_df.empty:
            sample_keys = new_df[keys].head(3)
            logging.info(f"📋 {data_type} - Muestra de claves:\n{sample_keys}")

def save_opta_data(data_dict, messages=None):
    """Versión mejorada con append seguro"""
    logging.info("Iniciando guardado incremental seguro")
    
    def add_message(msg, msg_type="info"):
        if messages is not None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            messages.append({'timestamp': timestamp, 'message': msg, 'type': msg_type})
        print(msg)

    try:
        OPTA_PATH.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        add_message(f"⌚ Error crítico: {str(e)}", "error")
        return

    # Primero diagnosticar
    diagnosticar_duplicados(data_dict)

    file_config = {
        'player_stats': {'filename': 'player_stats.parquet', 'keys': ['Match ID', 'Player ID']},
        'team_stats': {'filename': 'team_stats.parquet', 'keys': ['Match ID', 'Team ID']},
        'player_xg_stats': {'filename': 'player_xg_stats.parquet', 'keys': ['Match ID', 'Player ID']},
        'xg_events': {'filename': 'xg_events.parquet', 'keys': ['Match ID', 'EventId', 'timeStamp']},
        'match_events': {'filename': 'match_events.parquet', 'keys': ['Match ID', 'EventId', 'timeStamp']},
        'team_officials': {'filename': 'team_officials.parquet', 'keys': ['Match ID', 'Official ID']},
        'posesiones': {'filename': 'posesiones.parquet', 'keys': ['Match ID', 'EventId', 'timeStamp']}
    }

    for data_type, new_df in data_dict.items():
        if new_df.empty or data_type not in file_config:
            continue

        config = file_config[data_type]
        filename = OPTA_PATH / config['filename']
        unique_keys = config['keys']
        
        add_message(f"💾 Procesando: {filename.name}")

        try:
            # Limpiar NaNs en las claves del nuevo DataFrame
            for key in unique_keys:
                if key in new_df.columns:
                    new_df = new_df.dropna(subset=[key])
            
            if filename.exists():
                existing_df = pd.read_parquet(filename)
                initial_existing = len(existing_df)

                # Filtrar filas nuevas usando merge (mucho más rápido que iterrows)
                merged = new_df[unique_keys].merge(
                    existing_df[unique_keys].drop_duplicates(),
                    on=unique_keys,
                    how='left',
                    indicator=True
                )
                mask_new = merged['_merge'] == 'left_only'
                truly_new_df = new_df[mask_new.values]
                
                if len(truly_new_df) > 0:
                    # Solo append las filas nuevas
                    final_df = pd.concat([existing_df, truly_new_df], ignore_index=True)
                    add_message(f"   ✅ Añadidas {len(truly_new_df)} filas nuevas")
                else:
                    final_df = existing_df
                    add_message(f"   ℹ️ No hay filas nuevas para añadir")
                
                add_message(f"   📊 Total: {initial_existing} → {len(final_df)} filas")
                final_df.to_parquet(filename, index=False)
                
            else:
                # Archivo nuevo
                new_df.to_parquet(filename, index=False)
                add_message(f"   ✨ Archivo creado con {len(new_df)} filas")

        except Exception as e:
            add_message(f"❌ Error procesando {filename.name}: {e}", "error")
            logging.error(f"Error: {e}", exc_info=True)

def get_matches_by_weeks(stage_id, max_week, messages=None):
    """Get matches by weeks - versión web corregida"""
    def add_message(msg):
        if messages is not None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            messages.append({
                'timestamp': timestamp,
                'message': msg,
                'type': 'info'
            })
        print(msg)
    
    all_matches = []
    add_message(f"🔍 Buscando partidos para Stage ID: {stage_id}")
    
    for week in range(1, max_week + 1):
        add_message(f"📅 Jornada {week}...")
        matches_df = get_match_ids_advanced(
            max_matches=50,
            specific_week=str(week),
            stage_id=stage_id,
            messages=messages
        )
        
        if not matches_df.empty:
            add_message(f"   ✅ Encontrados {len(matches_df)} partidos en jornada {week}")
            all_matches.append(matches_df)
        else:
            add_message(f"   ⚠️ No se encontraron partidos en jornada {week}")
        time.sleep(2)
    
    if not all_matches:
        add_message("❌ No se encontraron partidos", "error")
        return pd.DataFrame()  # CORREGIDO: era return messages
    
    all_matches_df = pd.concat(all_matches, ignore_index=True)
    add_message(f"📊 Total partidos encontrados: {len(all_matches_df)}")
    
    # CORREGIDO: Obtener existing_match_ids aquí
    existing_match_ids = get_existing_match_ids(messages)
    
    # Filter new matches
    add_message("🔍 Filtrando partidos nuevos...")
    new_matches_df = filter_new_matches(all_matches_df, existing_match_ids, messages)
    
    if new_matches_df.empty:
        add_message("🎉 ¡No hay partidos nuevos que procesar!", "success")
        return pd.DataFrame()  # CORREGIDO: era return messages
    
    return new_matches_df  # CORREGIDO: retornar el DataFrame filtrado

def filter_new_matches(matches_df, existing_match_ids, messages=None):
    """Filter DataFrame to keep only new matches"""
    def add_message(msg):
        if messages is not None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            messages.append({
                'timestamp': timestamp,
                'message': msg,
                'type': 'info'
            })
        print(msg)
    
    if matches_df.empty:
        return matches_df
    
    if not existing_match_ids:
        print(f"🆕 Todos los partidos son nuevos: {len(matches_df)}")
        return matches_df
    
    new_matches = matches_df[~matches_df['Match ID'].isin(existing_match_ids)].copy()
    
    total_matches = len(matches_df)
    new_matches_count = len(new_matches)
    existing_matches_count = total_matches - new_matches_count
    
    print(f"📊 Total partidos encontrados: {total_matches}")
    print(f"✅ Ya procesados: {existing_matches_count}")
    print(f"🆕 Nuevos por procesar: {new_matches_count}")
    
    if new_matches_count == 0:
        print(f"🎉 ¡Todos los partidos ya están procesados!")
    
    return new_matches

def update_opta_data():
    """Main function to update Opta data - CORREGIDA"""
    print("🎯 ACTUALIZACIÓN DE DATOS OPTA")
    print("=" * 50)
    
    # Get ALL available tournaments/leagues
    print("📄 Obteniendo TODAS las ligas disponibles...")
    tournaments = get_all_tournaments()
    
    if not tournaments:
        print("⚠️ No se encontraron torneos con OT2, intentando con método alternativo...")
        stages = get_available_stages()
    else:
        # Convertir formato para compatibilidad
        stages = {}
        for tc_id, tc_info in tournaments.items():
            stages[tc_id] = {
                'name': tc_info['tournament_name'],
                'competition': tc_info['competition_name'],
                'competition_id': tc_info['competition_id'],
                'area': tc_info['area'],
                'start_date': tc_info['start_date'],
                'end_date': tc_info['end_date']
            }
    
    if not stages:
        print("❌ No se pudieron obtener las temporadas disponibles")
        return
    
    # Filter La Liga stages
    #la_liga_stages = {}
    #for stage_id, stage_info in stages.items():
        comp_name = stage_info.get('competition', '').lower()
        if 'primera' in comp_name or 'la liga' in comp_name:
            la_liga_stages[stage_id] = stage_info
    
    # Usamos directamente 'stages' (todas las encontradas)
    stages_to_show = stages 

    if not stages_to_show:
        print("❌ No se encontraron temporadas disponibles")
        return
    
    # Show available seasons
    print("\n🌎 TODAS LAS COMPETICIONES DISPONIBLES:")
    season_options = {}
    # Iteramos sobre 'stages_to_show' en lugar de 'la_liga_stages'
    for i, (stage_id, stage_info) in enumerate(stages_to_show.items(), 1):
        season_options[str(i)] = {
            'stage_id': stage_id,
            'stage_info': stage_info
        }
        print(f"  {i}: {stage_info['name']} ({stage_info['competition']})")
    
    # Get user selection
    while True:
        try:
            choice = input(f"\n🗓️ Selecciona temporada (1-{len(season_options)}): ").strip()
            if choice in season_options:
                selected = season_options[choice]
                break
            else:
                print("❌ Selección no válida")
        except KeyboardInterrupt:
            print("\n👋 Cancelado por el usuario")
            return
    
    # Get max week
    while True:
        try:
            max_week = int(input("📅 ¿Hasta qué jornada descargar? (ej: 10): ").strip())
            if max_week > 0:
                break
            else:
                print("❌ Debe ser un número mayor a 0")
        except (ValueError, KeyboardInterrupt):
            print("❌ Entrada no válida")
            return
    
    stage_id = selected['stage_id']
    stage_info = selected['stage_info']
    
    print(f"\n✅ Configuración:")
    print(f"   🏆 Competición: {stage_info['competition']}")
    print(f"   🗓️ Temporada: {stage_info['name']}")
    print(f"   📅 Jornadas: 1-{max_week}")
    print(f"   📊 Feeds: MA2 (Stats), MA3 (Events), MA12 (xG)")
    
    # Get existing match IDs
    existing_match_ids = get_existing_match_ids()
    
    # Get matches - CORREGIDO: usar la función simple
    print("\n📄 Obteniendo partidos...")
    all_matches_df = get_matches_by_weeks(stage_id, max_week)
    
    # Filter new matches
    if not all_matches_df.empty:
        print("\n🔍 Filtrando partidos nuevos...")
        new_matches_df = filter_new_matches(all_matches_df, existing_match_ids)
    else:
        new_matches_df = pd.DataFrame()
    
    if new_matches_df.empty:
        print("🎉 ¡No hay partidos nuevos que descargar!")
        # AUNQUE NO HAYA PARTIDOS, ACTUALIZAMOS ABP POR SI ACASO
        update_abp_events_standalone()
        calculate_and_save_abp_statistics()
        # Sincronizar con GitHub automáticamente
        git_auto_sync("Opta")
        return
    
    print(f"\n📊 Procesando {len(new_matches_df)} partidos nuevos...")
    match_ids = new_matches_df['Match ID'].tolist()

    # Get season from competitions data
    try:
        competitions = get_all_competitions_and_stages()
        season = "N/A"
        for comp_id, comp_info in competitions.items():
            if stage_id in comp_info['stages']:
                season = comp_info['stages'][stage_id]['season']
                break
        print(f"🗓️ Temporada detectada: {season}")
    except Exception as e:
        season = "N/A"
        print(f"⚠️ No se pudo detectar la temporada: {e}")
    
    # Process data
    all_data = {
        'player_stats': [],
        'team_stats': [],
        'player_xg_stats': [],
        'xg_events': [],
        'match_events': [],
        'team_officials': []
    }
    
    for i, match_id in enumerate(match_ids):
        print(f"⚽ Partido {i+1}/{len(match_ids)}: {match_id}")
        
        # ... (código de procesamiento de MA2, MA3, MA12 sin cambios) ...
        # MA2 - Player Stats + Team Officials
        player_stats_df, team_officials_df = process_match_player_stats(match_id, season)
        if not player_stats_df.empty:
            all_data['player_stats'].append(player_stats_df)
        if not team_officials_df.empty:
            all_data['team_officials'].append(team_officials_df)
        
        # MA2 - Team Stats
        team_stats_df = process_match_team_stats(match_id, season)
        if not team_stats_df.empty:
            all_data['team_stats'].append(team_stats_df)
        
        # MA3 - Match Events
        match_events_df = process_match_events(match_id, season)
        if not match_events_df.empty:
            all_data['match_events'].append(match_events_df)
        
        # MA12 - Player xG Stats
        player_xg_df = process_xg_player_stats(match_id, season)
        if not player_xg_df.empty:
            all_data['player_xg_stats'].append(player_xg_df)
        
        # MA12 - xG Events
        xg_events_df = process_xg_events(match_id, season)
        if not xg_events_df.empty:
            all_data['xg_events'].append(xg_events_df)
        
        if i < len(match_ids) - 1:
            time.sleep(DELAY_SECONDS)
    
    # Combine DataFrames
    final_data = {}
    for data_type, df_list in all_data.items():
        if df_list:
            final_data[data_type] = pd.concat(df_list, ignore_index=True)
            print(f"✅ {data_type}: {len(final_data[data_type])} filas nuevas")
        else:
            final_data[data_type] = pd.DataFrame()
    
    # Save data
    print(f"\n💾 Guardando datos...")
    save_opta_data(final_data)
    
    # >>>>> MODIFICACIÓN: Llamar a la función robusta de ABP <<<<<
    update_abp_events_standalone()
    calculate_and_save_abp_statistics()
    
    print(f"\n🎉 ¡Actualización de Opta completada!")
    print(f"📊 {len(new_matches_df)} partidos procesados")

    # Sincronizar con GitHub automáticamente
    git_auto_sync("Opta")


# ====================================
# SPORTIAN DATA UPDATER
# ====================================

def process_sportian_csv_upload(contents, filename, progress_callback=None):
    """Procesa un archivo CSV de Sportian subido via web (corners o faltas)"""
    import base64
    import tempfile
    import sys
    import os

    messages = []

    def add_message(msg, msg_type="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        messages.append({'timestamp': timestamp, 'message': msg, 'type': msg_type})
        print(msg)

    def update_progress(progress, status=""):
        if progress_callback:
            progress_callback(progress, status, messages)

    add_message(f"📤 PROCESANDO CSV SPORTIAN: {filename}")
    add_message("=" * 50)
    update_progress(10, "Decodificando archivo...")

    try:
        # 1. Decodificar el contenido base64
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        # 2. Guardar temporalmente en la carpeta de Sportian
        sportian_dir = os.path.join(os.getcwd(), 'extraccion_sportian')
        temp_csv_path = os.path.join(sportian_dir, filename)

        with open(temp_csv_path, 'wb') as f:
            f.write(decoded)

        add_message(f"✅ Archivo guardado: {temp_csv_path}")
        update_progress(30, "Archivo guardado, procesando...")

        # 3. Importar y ejecutar la función del script csv_a_parquet
        sys.path.insert(0, sportian_dir)
        from csv_a_parquet import procesar_dataset

        # Determinar tipo de archivo y procesar
        nombre_lower = filename.lower()

        if 'corner' in nombre_lower:
            parquet_dest = os.path.join(sportian_dir, 'corners_tracking.parquet')
            id_col = 'ID_Evento_Corner'
            tipo = 'corners'
        elif 'falta' in nombre_lower:
            parquet_dest = os.path.join(sportian_dir, 'faltas_tracking.parquet')
            id_col = 'ID_Evento_Falta'
            tipo = 'faltas'
        else:
            add_message(f"❌ El archivo debe contener 'corners' o 'faltas' en el nombre", "error")
            update_progress(100, "Error: nombre de archivo inválido")
            return

        add_message(f"🔍 Tipo detectado: {tipo}")
        update_progress(50, f"Procesando {tipo}...")

        # 4. Procesar el dataset
        procesar_dataset(temp_csv_path, parquet_dest, id_col)

        add_message(f"✅ Datos de {tipo} actualizados en {parquet_dest}")
        update_progress(80, "Sincronizando con Git...")

        # 5. Sincronizar con Git
        git_auto_sync("Sportian")

        add_message("✅ Proceso completado exitosamente")
        update_progress(100, "✅ Proceso completado")

    except Exception as e:
        add_message(f"❌ Error: {str(e)}", "error")
        update_progress(100, f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

# ====================================
# MEDIACOACH DATA UPDATER
# ====================================

def update_mediacoach_data():
    """Update MediaCoach data - TO BE IMPLEMENTED"""
    print("🎯 ACTUALIZACIÓN DE DATOS MEDIACOACH")
    print("=" * 50)
    print("⏳ Funcionalidad pendiente de implementar...")
    print("📧 Contacta con el administrador para más información")

    # TODO: Implement MediaCoach data update logic
    # Cuando se implemente, añadir al final:
    git_auto_sync("MediaCoach")

# ====================================
# MAIN INTERFACE
# ====================================

def main():
    """Main interface for data updates"""
    print("🔄 SISTEMA DE ACTUALIZACIÓN DE DATOS")
    print("=" * 50)
    print("📊 Villarreal CF - Departamento de Datos")
    print("")
    
    while True:
        print("\n📋 OPCIONES DISPONIBLES:")
        print("  1. 📈 Actualizar datos de Opta (incluye ABP)")
        print("  2. ⚽ Actualizar/Crear Parquet de Eventos ABP solamente")
        print("  3. 🎮 Actualizar datos de MediaCoach")
        print("  4. 🔄 Actualizar ambos (Opta y MediaCoach)")
        print("  5. 🚪 Salir")
        
        try:
            choice = input("\n🎯 Selecciona una opción (1-5): ").strip()
            
            if choice == '1':
                update_opta_data()
            elif choice == '2':
                update_abp_events_standalone()
                calculate_and_save_abp_statistics()
                git_auto_sync("ABP")
            elif choice == '3':
                update_mediacoach_data()
            elif choice == '4':
                print("🔄 Actualizando ambos sistemas...")
                update_opta_data()
                print("\n" + "="*50)
                update_mediacoach_data()
            elif choice == '5':
                print("👋 ¡Hasta luego!")
                break
            else:
                print("❌ Opción no válida. Intenta de nuevo.")
                
        except KeyboardInterrupt:
            print("\n👋 Cancelado por el usuario")
            break
        except Exception as e:
            print(f"❌ Error inesperado: {e}")

# Credenciales exactas de tu archivo clave
SUBSCRIPTION_KEY = '729f9154234d4ff3bb0a692c6a0510c4'
API_URL_BASE = "https://club-api.mediacoach.es"

def ejecutar_curl_comando_vcf(comando):
    """Lógica idéntica a tu script: usa curl del sistema"""
    try:
        process = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            return None
        return json.loads(stdout)
    except:
        return None

def get_mediacoach_token():
    """Obtiene el token usando requests (esta parte sí funcionaba)"""
    url = 'https://id.mediacoach.es/connect/token'
    data = {
        'client_id': '58191b89-cee4-11ed-a09d-ee50c5eb4bb5',
        'scope': 'b2bapiclub-api',
        'grant_type': 'password',
        'username': 'b2bvillarealcf@mediacoach.es',
        'password': 'r728-FHj3RE!'
    }
    try:
        r = requests.post(url, data=data, timeout=10)
        return r.json().get('access_token')
    except: return None

def get_mediacoach_seasons_api():
    """Prueba endpoints usando CURL y mapea los IDs correctamente"""
    token = get_mediacoach_token()
    if not token: return []
    
    credenciales = f"--header 'Ocp-Apim-Subscription-Key: {SUBSCRIPTION_KEY}' --header 'Authorization: Bearer {token}'"
    endpoints = ["/Championships", "/Championships/seasons", "/seasons", "/api/seasons"]
    
    for ep in endpoints:
        print(f"🔍 Probando endpoint MediaCoach: {ep}")
        comando = f"curl -s --location '{API_URL_BASE}{ep}' {credenciales}"
        data = ejecutar_curl_comando_vcf(comando)
        
        if data and isinstance(data, list):
            print(f"✅ Éxito en endpoint: {ep}. Procesando {len(data)} elementos.")
            
            opciones = []
            for i, t in enumerate(data):
                # Buscamos el nombre en varias posibles llaves
                label = t.get('name') or t.get('Name') or t.get('seasonName') or f"Temporada {i+1}"
                
                # Buscamos el ID en varias posibles llaves (ESTO ES LO QUE FALLABA)
                value = t.get('id') or t.get('Id') or t.get('seasonId')
                
                # Si no hay ID, usamos el índice como último recurso para que no sea null
                if value is None:
                    value = str(i)

                competiciones = t.get('competitions', [])

                opciones.append({
                    'label': label, 
                    'value': value,
                    'competitions': competiciones  # ← AÑADIR ESTO
                })
            
            return opciones
    
    return [{'label': 'Temporada 24-25 (Backup)', 'value': '3a134240-833f-41dd-c6b0-3d6b87479c15'}]

def get_mediacoach_competitions_api(season_id):
    """Obtiene ligas desde las temporadas ya cargadas"""
    if not season_id: 
        return []
    
    # Obtener temporadas de nuevo para extraer las competiciones
    temporadas = get_mediacoach_seasons_api()
    
    for temp in temporadas:
        if temp['value'] == season_id:
            competiciones = temp.get('competitions', [])
            if competiciones:
                opciones = []
                for c in competiciones:
                    nombre = c.get('name') or c.get('Name') or "Liga desconocida"
                    opciones.append({'label': nombre, 'value': nombre})
                return opciones
    
    # Backup si no encuentra
    return [
        {'label': 'La Liga', 'value': 'La Liga'},
        {'label': 'La Liga 2', 'value': 'La Liga 2'}
    ]
    
if __name__ == "__main__":
    main()