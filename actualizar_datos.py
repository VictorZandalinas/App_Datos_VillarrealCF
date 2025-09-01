#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SISTEMA DE ACTUALIZACIÓN DE DATOS
Opta API + MediaCoach Data Updater
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

# Paths
OPTA_PATH = Path('datos_opta_parquet')
MEDIACOACH_PATH = Path('datos_mediacoach_parquet')

# Event Types Mapping
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

# Qualifier Types Mapping (basado en la documentación)
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
# OPTA API FUNCTIONS
# ====================================

def request_headers():
    """OAuth authentication for Opta API"""
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
    
    response = requests.post(post_url, data=body, headers=oauth_headers)
    access_token = response.json()['access_token']
    return {'Authorization': f'Bearer {access_token}'}

def get_available_stages():
    """Get available stages/seasons from Opta API"""
    request_parameters = {
        "_fmt": "json",
        "_pgSz": "100",
        "_pgNm": "1",
        "_rt": "b"
    }
    
    sdapi_get_url = f'https://api.performfeeds.com/soccerdata/match/{OPTA_API_KEY}/'
    
    try:
        response = requests.get(sdapi_get_url, headers=request_headers(), params=request_parameters)
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
        'team_officials.parquet'
    ]
    
    add_message("🔍 Revisando archivos existentes...")
    
    for filename in parquet_files:
        filepath = OPTA_PATH / filename
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath)
                if not df.empty and 'Match ID' in df.columns:
                    file_match_ids = set(df['Match ID'].unique())
                    existing_match_ids.update(file_match_ids)
                    add_message(f"   📄 {filename}: {len(file_match_ids)} partidos únicos")
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
    """Get matches by range - versión web con progreso"""
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
    total_weeks = end_week - start_week + 1
    
    for week in range(start_week, end_week + 1):
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
            add_message(f"   ⚠️ No se encontraron partidos en jornada {week}")
        time.sleep(2)
    
    if all_matches:
        result_df = pd.concat(all_matches, ignore_index=True)
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
    """Get match IDs - versión de la notebook que funciona"""
    def add_message(msg):
        if messages is not None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            messages.append({
                'timestamp': timestamp,
                'message': msg,
                'type': 'info'
            })
        print(msg)

    request_parameters = {
        "_fmt": "json",
        "_pgSz": str(max_matches),
        "_pgNm": "1",
        "live": "yes",
        "_rt": "b"
    }
    
    if specific_week:
        request_parameters["week"] = str(specific_week)
    
    if stage_id:
        request_parameters["stg"] = str(stage_id)

    sdapi_get_url = f'https://api.performfeeds.com/soccerdata/match/{OPTA_API_KEY}/'
    
    try:
        response = requests.get(sdapi_get_url, headers=request_headers(), params=request_parameters)
        response.raise_for_status()
        data = response.json()
        matches = data.get('match', [])
        
        match_list = []
        for match in matches:
            match_info = match.get('matchInfo', {})
            match_id = match_info.get('id')
            if match_id:
                competition = match_info.get('competition', {})
                stage = match_info.get('stage', {})
                contestants = match_info.get('contestant', [])
                teams = [{'id': c.get('id'), 'name': c.get('name'), 'code': c.get('code')} for c in contestants]
                
                # Verificar que el partido esté finalizado
                live_data = match.get('liveData', {})
                match_details = live_data.get('matchDetails', {})
                match_status = match_details.get('matchStatus', '')
                
                # Solo incluir partidos finalizados
                if match_status.lower() in ['played', 'finished', 'ft', 'final']:
                    match_data = {
                        'Match ID': match_id,
                        'Competition': competition.get('name', 'N/A'),
                        'Competition ID': competition.get('id', 'N/A'),
                        'Stage ID': stage.get('id', 'N/A'),
                        'Stage Name': stage.get('name', 'N/A'),
                        'Date': match_info.get('date', 'N/A'),
                        'Week': match_info.get('week', 'N/A'),
                        'Match Status': match_status,
                        'Teams': teams
                    }
                    match_list.append(match_data)
        
        return pd.DataFrame(match_list)
    except Exception as e:
        print(f"❌ Error obteniendo partidos: {e}")
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
    response = requests.get(sdapi_get_url, headers=request_headers(), params=request_parameters)
    
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
    response = requests.get(sdapi_get_url, headers=request_headers(), params=request_parameters)
    
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
        response = requests.get(sdapi_get_url, headers=request_headers(), params=requestParameters)
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
    """Process MA3 Match Events - VERSIÓN CORREGIDA CON PARÁMETROS DOCUMENTADOS"""
    import logging
    
    logging.info(f"🚀 Iniciando procesamiento de eventos para partido {match_id}")
    logging.info(f"📊 Season detectada: {season}")
    
    # ✅ SOLO PARÁMETROS DOCUMENTADOS SEGÚN LA DOCUMENTACIÓN OFICIAL
    request_parameters = {
        "_fmt": "json",
        "fx": match_id,
        "_rt": "b"
        # ❌ ELIMINADOS: detailed, _pgSz, live (NO documentados para MA3)
    }
    
    logging.info(f"🔧 Parámetros de request (SOLO DOCUMENTADOS):")
    for key, value in request_parameters.items():
        logging.info(f"   - {key}: {value}")
    
    sdapi_get_url = f'https://api.performfeeds.com/soccerdata/matchevent/{OPTA_API_KEY}/'
    logging.info(f"🌐 URL completa: {sdapi_get_url}")
    
    try:
        # Obtener headers con logging
        logging.info("🔑 Obteniendo headers de autenticación...")
        headers = request_headers()
        logging.info(f"✅ Headers obtenidos exitosamente")
        
        # Log del token (solo primeros/últimos caracteres por seguridad)
        auth_header = headers.get('Authorization', '')
        if len(auth_header) > 20:
            token_preview = f"{auth_header[:20]}...{auth_header[-10:]}"
            logging.info(f"🎫 Token preview: {token_preview}")
        
        # Hacer la request con parámetros documentados
        logging.info(f"📡 Enviando request a Opta API con parámetros documentados...")
        logging.info(f"📡 Match ID: {match_id}")
        logging.info(f"📡 URL: {sdapi_get_url}")
        
        start_time = time.time()
        response = requests.get(sdapi_get_url, headers=headers, params=request_parameters, timeout=30)
        response_time = time.time() - start_time
        
        logging.info(f"⏱️ Tiempo de respuesta: {response_time:.2f} segundos")
        logging.info(f"📡 Status Code: {response.status_code}")
        logging.info(f"📡 Response Headers: {dict(response.headers)}")
        
        # MANEJO ESPECÍFICO DE ERRORES
        if response.status_code == 400:
            logging.error(f"❌ ERROR 400 - Verificando causa...")
            try:
                error_data = response.json()
                error_code = error_data.get('errorCode', 'N/A')
                logging.error(f"🔍 Error Code: {error_code}")
                
                if error_code == "10217":
                    logging.error("🔑 ERROR 10217: Feed MA3 no autorizado en tu suscripción")
                    logging.error("   Opciones:")
                    logging.error("   1. Contactar Opta para activar MA3")
                    logging.error("   2. Continuar solo con MA2 y MA12")
                    return pd.DataFrame()
                elif error_code == "10203":
                    logging.error("🔧 ERROR 10203: Parámetros inválidos/ambiguos")
                    logging.error("   Los parámetros enviados no son válidos para este endpoint")
                    return pd.DataFrame()
                else:
                    logging.error(f"❓ Error desconocido: {error_code}")
                    
            except json.JSONDecodeError:
                logging.error(f"📄 Respuesta no JSON: {response.text}")
            
            return pd.DataFrame()
        
        elif response.status_code != 200:
            logging.error(f"❌ ERROR {response.status_code}")
            logging.error(f"📄 Response: {response.text}")
            return pd.DataFrame()
        
        # ✅ RESPUESTA EXITOSA
        logging.info(f"✅ Respuesta API exitosa para partido {match_id}")
        
        try:
            data = response.json()
            logging.info(f"📋 Estructura de respuesta:")
            logging.info(f"   - Keys principales: {list(data.keys())}")
            
            # Verificar estructura de datos
            match_info = data.get('matchInfo', {})
            live_data = data.get('liveData', {})
            
            logging.info(f"📋 matchInfo keys: {list(match_info.keys())}")
            logging.info(f"📋 liveData keys: {list(live_data.keys())}")
            
            # Obtener eventos de la ubicación estándar
            events = live_data.get('event', [])
            logging.info(f"📊 Eventos encontrados: {len(events)}")
            
            if not events:
                logging.warning(f"⚠️ NO SE ENCONTRARON EVENTOS para partido {match_id}")
                logging.warning(f"📋 Estructura liveData completa: {list(live_data.keys())}")
                
                # Guardar respuesta para debug
                debug_file = f"debug_no_events_{match_id}.json"
                with open(debug_file, 'w') as f:
                    json.dump(data, f, indent=2)
                logging.warning(f"💾 Respuesta guardada en: {debug_file}")
                
                return pd.DataFrame()
            
            # Procesar información del partido
            competition_info = match_info.get('competition', {})
            stage_info = match_info.get('stage', {})
            home_away_info = get_home_away_info(match_info, live_data)
            
            logging.info(f"🏟️ Información del partido:")
            logging.info(f"   - Competition: {competition_info.get('name', 'N/A')}")
            logging.info(f"   - Stage: {stage_info.get('name', 'N/A')}")
            logging.info(f"   - Home Team: {home_away_info.get('home_team_id', 'N/A')}")
            logging.info(f"   - Away Team: {home_away_info.get('away_team_id', 'N/A')}")
            
            # Análisis de tipos de eventos
            event_types = {}
            events_without_type = 0
            events_without_contestant = 0
            
            for event in events:
                type_id = event.get('typeId')
                if type_id:
                    event_types[type_id] = event_types.get(type_id, 0) + 1
                else:
                    events_without_type += 1
                    
                if not event.get('contestantId'):
                    events_without_contestant += 1
            
            logging.info(f"📊 Análisis de tipos de eventos:")
            for type_id, count in sorted(event_types.items()):
                event_name = EVENT_TYPE_MAPPING.get(type_id, f'Unknown ({type_id})')
                logging.info(f"   - {event_name}: {count}")
            
            if events_without_type > 0:
                logging.warning(f"⚠️ {events_without_type} eventos sin typeId")
            if events_without_contestant > 0:
                logging.info(f"ℹ️ {events_without_contestant} eventos sin contestantId")
            
            # Encontrar qualifiers únicos
            qualifier_ids = set()
            invalid_qualifiers = 0
            
            for event in events:
                for q in event.get('qualifier', []):
                    qid = q.get('qualifierId', '')
                    if qid and str(qid).isdigit():
                        qualifier_ids.add(str(qid))
                    else:
                        invalid_qualifiers += 1
                        if invalid_qualifiers <= 5:
                            logging.warning(f"⚠️ Qualifier ID inválido: {qid}")
            
            logging.info(f"🔍 Qualifiers únicos encontrados: {len(qualifier_ids)}")
            if invalid_qualifiers > 0:
                logging.warning(f"⚠️ {invalid_qualifiers} qualifiers inválidos")
            
            # Inicializar columnas con nombres descriptivos
            qualifier_names = []
            for qid in qualifier_ids:
                try:
                    qualifier_name = QUALIFIER_MAPPING.get(int(qid), f'qualifier {qid}')
                    qualifier_names.append(qualifier_name)
                except ValueError:
                    logging.error(f"❌ Error convirtiendo qualifier ID: {qid}")
                    continue
            
            columns = [
                'Match ID', 'Competition ID', 'Competition Name', 'Season', 'Week', 'Stage ID', 'Stage Name',
                'EventId', 'typeId', 'Event Name', 'timeStamp', 'contestantId', 'Team ID', 'Team Name', 
                'Team Position', 'Is Home', 'Is Away', 'HT Home Score', 'HT Away Score', 
                'FT Home Score', 'FT Away Score', 'periodId', 'timeMin', 'timeSec', 
                'playerId', 'playerName', 'outcome', 'x', 'y'
            ] + qualifier_names
            
            logging.info(f"📋 DataFrame final: {len(columns)} columnas")
            
            # Procesar eventos
            events_data = []
            processed_events = 0
            skipped_events = 0
            
            for i, event in enumerate(events):
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
                    
                    # Inicializar todos los qualifiers a "No"
                    for qid in qualifier_ids:
                        try:
                            qualifier_name = QUALIFIER_MAPPING.get(int(qid), f'qualifier {qid}')
                            event_info[qualifier_name] = "No"
                        except ValueError:
                            continue

                    # Actualizar con valores reales de qualifiers
                    for q in event.get('qualifier', []):
                        try:
                            qualifier_id = int(q["qualifierId"])
                            qualifier_name = QUALIFIER_MAPPING.get(qualifier_id, f'qualifier {qualifier_id}')
                            
                            if 'value' in q and q['value'] is not None:
                                event_info[qualifier_name] = str(q['value'])
                            else:
                                event_info[qualifier_name] = "Sí"
                        except (ValueError, KeyError) as e:
                            logging.warning(f"⚠️ Error procesando qualifier: {e}")
                            continue
                    
                    events_data.append(event_info)
                    processed_events += 1
                    
                except Exception as e:
                    logging.error(f"❌ Error procesando evento {i}: {e}")
                    skipped_events += 1
                    continue
            
            logging.info(f"✅ Procesamiento completado:")
            logging.info(f"   - Eventos procesados: {processed_events}")
            logging.info(f"   - Eventos omitidos: {skipped_events}")
            logging.info(f"   - Total en respuesta: {len(events)}")
            
            # Crear DataFrame
            try:
                df = pd.DataFrame(events_data, columns=columns)
                logging.info(f"✅ DataFrame creado: {len(df)} filas, {len(df.columns)} columnas")
                
                # Verificar datos críticos
                critical_columns = ['EventId', 'typeId', 'periodId']
                for col in critical_columns:
                    if col in df.columns:
                        null_count = df[col].isnull().sum()
                        if null_count > 0:
                            logging.warning(f"⚠️ Columna '{col}': {null_count} valores nulos")
                
                return df
                
            except Exception as e:
                logging.error(f"❌ Error creando DataFrame: {e}")
                return pd.DataFrame()
        
        except json.JSONDecodeError as e:
            logging.error(f"❌ Error parseando JSON: {e}")
            logging.error(f"📄 Respuesta cruda: {response.text[:500]}...")
            return pd.DataFrame()
            
    except requests.Timeout as e:
        logging.error(f"⏰ Timeout en request para partido {match_id}: {e}")
        return pd.DataFrame()
        
    except requests.ConnectionError as e:
        logging.error(f"🌐 Error de conexión para partido {match_id}: {e}")
        return pd.DataFrame()
        
    except requests.RequestException as e:
        logging.error(f"📡 Error de request para partido {match_id}: {e}")
        return pd.DataFrame()
        
    except Exception as e:
        logging.error(f"💥 Error inesperado en partido {match_id}: {e}")
        logging.error(f"💥 Traceback completo:", exc_info=True)
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
            response = requests.get(
                f'https://api.performfeeds.com/soccerdata/matchevent/{OPTA_API_KEY}/',
                headers=request_headers(),
                params=params
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
            response = requests.get(
                f'https://api.performfeeds.com/soccerdata/matchevent/{OPTA_API_KEY}/',
                headers=request_headers(),
                params=request_parameters
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

def update_opta_data_web(competition_id, stage_id, start_week, end_week, progress_callback=None):
    """Versión web que devuelve mensajes progresivos con barra de progreso"""
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
    
    add_message("🎯 ACTUALIZACIÓN DE DATOS OPTA (WEB - RANGOS)")
    add_message("=" * 50)
    update_progress(5, "Iniciando proceso...")
    
    add_message(f"✅ Configuración:")
    add_message(f"   🏆 Competición ID: {competition_id}")
    add_message(f"   🗓️ Stage ID: {stage_id}")
    add_message(f"   📅 Jornadas: {start_week}-{end_week}")
    add_message(f"   📊 Feeds: MA2 (Stats), MA3 (Events), MA12 (xG)")
    
    # Get existing match IDs
    add_message("🔍 Revisando archivos existentes...")
    update_progress(10, "Revisando archivos existentes...")
    existing_match_ids = get_existing_match_ids(messages)
    
    
    # Get matches by range
    add_message("🔄 Obteniendo partidos...")
    update_progress(20, "Obteniendo partidos...")
    all_matches_df = get_matches_by_weeks_range(stage_id, start_week, end_week, messages=messages, progress_callback=progress_callback)
    
    if all_matches_df.empty:
        add_message("❌ No se encontraron partidos", "error")
        update_progress(100, "Error: No se encontraron partidos")
        return messages
    
    # Filter new matches
    add_message("🔍 Filtrando partidos nuevos...")
    update_progress(30, "Filtrando partidos nuevos...")
    new_matches_df = filter_new_matches(all_matches_df, existing_match_ids, messages)
    
    if new_matches_df.empty:
        add_message("🎉 ¡No hay partidos nuevos que procesar!", "success")
        update_progress(100, "Completado: No hay partidos nuevos")
        return messages
    
    add_message(f"📊 Procesando {len(new_matches_df)} partidos nuevos...")
    match_ids = new_matches_df['Match ID'].tolist()
    
    # Get season from competitions data
    try:
        competitions = get_all_competitions_and_stages()
        season = "N/A"
        for comp_id, comp_info in competitions.items():
            if comp_id == competition_id:
                if stage_id in comp_info['stages']:
                    season = comp_info['stages'][stage_id]['season']
                    break
        add_message(f"🗓️ Temporada detectada: {season}")
    except Exception as e:
        season = "N/A"
        add_message(f"⚠️ No se pudo detectar la temporada: {e}")
    
    # Process data
    all_data = {
        'player_stats': [],
        'team_stats': [],
        'player_xg_stats': [],
        'xg_events': [],
        'match_events': [],
        'team_officials': []
    }
    
    # Progreso de procesamiento de partidos (30% a 85%)
    progress_increment = 55 / len(match_ids)
    
    for i, match_id in enumerate(match_ids):
        current_progress = 30 + (i * progress_increment)
        update_progress(current_progress, f"Procesando partido {i+1}/{len(match_ids)}")
        add_message(f"⚽ Partido {i+1}/{len(match_ids)}: {match_id}")
        
        try:
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
                
        except Exception as e:
            add_message(f"❌ Error en partido {match_id}: {e}")
            continue
        
        if i < len(match_ids) - 1:
            time.sleep(DELAY_SECONDS)
    
    # Combine DataFrames
    update_progress(85, "Combinando datos...")
    new_data = {}
    for data_type, df_list in all_data.items():
        if df_list:
            new_data[data_type] = pd.concat(df_list, ignore_index=True)
            add_message(f"✅ {data_type} (nuevos): {len(new_data[data_type])} filas")
        else:
            new_data[data_type] = pd.DataFrame()
    
    # Save data
    add_message("💾 Guardando datos nuevos y combinando con existentes...")
    update_progress(95, "Guardando datos...")
    save_opta_data(new_data)
    
    add_message("🎉 ¡Actualización completada!", "success")
    add_message(f"📊 {len(new_matches_df)} partidos nuevos procesados")
    update_progress(100, "¡Actualización completada!")
    
    return messages

def process_xg_player_stats(match_id, season):
    """Process MA12 Player xG Stats - FUNCIÓN QUE FALTABA"""
    request_parameters = {
        "_fmt": "json",
        "fx": match_id,
        "_rt": "b"
    }
    
    sdapi_get_url = f'https://api.performfeeds.com/soccerdata/matchexpectedgoals/{OPTA_API_KEY}/'
    response = requests.get(sdapi_get_url, headers=request_headers(), params=request_parameters)
    
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
    response = requests.get(sdapi_get_url, headers=request_headers(), params=request_parameters)
    
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
        'xg_events': ['Match ID', 'EventId', 'timeStamp'],  # <-- Añadir timeStamp
        'match_events': ['Match ID', 'EventId', 'timeStamp'],  # <-- Añadir timeStamp
        'team_officials': ['Match ID', 'Official ID']
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
        'xg_events': {'filename': 'xg_events.parquet', 'keys': ['Match ID', 'EventId', 'timeStamp']},  # <-- Añadir
        'match_events': {'filename': 'match_events.parquet', 'keys': ['Match ID', 'EventId', 'timeStamp']},  # <-- Añadir
        'team_officials': {'filename': 'team_officials.parquet', 'keys': ['Match ID', 'Official ID']}
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
                
                # Crear conjunto de claves existentes (como tuplas)
                existing_keys = set()
                for _, row in existing_df[unique_keys].iterrows():
                    existing_keys.add(tuple(row.values))
                
                # Filtrar solo filas realmente nuevas
                mask_new = ~new_df.apply(
                    lambda row: tuple(row[unique_keys].values) in existing_keys, 
                    axis=1
                )
                truly_new_df = new_df[mask_new]
                
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

def get_matches_by_weeks_range(stage_id, start_week, end_week, messages=None, progress_callback=None):
    """Get matches by range - versión web con progreso"""
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
    total_weeks = end_week - start_week + 1
    
    for week in range(start_week, end_week + 1):
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
            add_message(f"   ⚠️ No se encontraron partidos en jornada {week}")
        time.sleep(2)
    
    if all_matches:
        result_df = pd.concat(all_matches, ignore_index=True)
        add_message(f"📊 Total partidos encontrados: {len(result_df)}")
        return result_df
    else:
        add_message("❌ No se encontraron partidos", "error")
        return pd.DataFrame()

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
    
    # Get available stages
    print("📄 Obteniendo temporadas disponibles...")
    stages = get_available_stages()
    
    if not stages:
        print("❌ No se pudieron obtener las temporadas disponibles")
        return
    
    # Filter La Liga stages
    la_liga_stages = {}
    for stage_id, stage_info in stages.items():
        comp_name = stage_info.get('competition', '').lower()
        if 'primera' in comp_name or 'la liga' in comp_name:
            la_liga_stages[stage_id] = stage_info
    
    if not la_liga_stages:
        print("❌ No se encontraron temporadas de La Liga")
        return
    
    # Show available seasons
    print("\n🇪🇸 TEMPORADAS DE LA LIGA DISPONIBLES:")
    season_options = {}
    for i, (stage_id, stage_info) in enumerate(la_liga_stages.items(), 1):
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
    
    if all_matches_df.empty:
        print("❌ No se encontraron partidos")
        return
    
    # Filter new matches
    print("\n🔍 Filtrando partidos nuevos...")
    new_matches_df = filter_new_matches(all_matches_df, existing_match_ids)
    
    if new_matches_df.empty:
        print("🎉 ¡No hay partidos nuevos que procesar!")
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
        print(f"🗓️ Temporada detectada: {season}")  # CORREGIDO: era add_message
    except Exception as e:
        season = "N/A"
        print(f"⚠️ No se pudo detectar la temporada: {e}")  # CORREGIDO: era add_message
    
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
    
    print(f"\n🎉 ¡Actualización de Opta completada!")
    print(f"📊 {len(new_matches_df)} partidos procesados")

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
    # This will be implemented when MediaCoach notebook is provided

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
        print("  1. 📈 Actualizar datos de Opta")
        print("  2. 🎮 Actualizar datos de MediaCoach")
        print("  3. 🔄 Actualizar ambos")
        print("  4. 🚪 Salir")
        
        try:
            choice = input("\n🎯 Selecciona una opción (1-4): ").strip()
            
            if choice == '1':
                update_opta_data()
            elif choice == '2':
                update_mediacoach_data()
            elif choice == '3':
                print("🔄 Actualizando ambos sistemas...")
                update_opta_data()
                print("\n" + "="*50)
                update_mediacoach_data()
            elif choice == '4':
                print("👋 ¡Hasta luego!")
                break
            else:
                print("❌ Opción no válida. Intenta de nuevo.")
                
        except KeyboardInterrupt:
            print("\n👋 Cancelado por el usuario")
            break
        except Exception as e:
            print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    main()