import json
import pandas as pd
import os
from datetime import datetime

EVENT_TYPE_MAPPING = {
    1: "Pass",
    2: "Offside Pass", 
    3: "Take On",
    4: "Foul",
    5: "Out",
    6: "Corner Awarded",
    7: "Tackle",
    8: "Interception",
    10: "Save",
    11: "Claim",
    12: "Clearance",
    13: "Miss",
    14: "Post",
    15: "Attempt Saved",
    16: "Goal",
    17: "Card",
    18: "Player off",
    19: "Player on",
    20: "Player retired",
    21: "Player returns",
    22: "Player becomes goalkeeper",
    23: "Goalkeeper becomes player",
    24: "Condition change",
    25: "Official change",
    27: "Start delay",
    28: "End delay",
    29: "Temporary stop",
    30: "End",
    32: "Start",
    34: "Team set up",
    36: "Player changed Jersey number",
    37: "Collection End",
    38: "Temp Goal",
    39: "Temp Attempt",
    40: "Formation change",
    41: "Punch",
    42: "Good skill",
    43: "Deleted event",
    44: "Aerial",
    45: "Challenge",
    47: "Rescinded card",
    49: "Ball recovery",
    50: "Dispossessed",
    51: "Error",
    52: "Keeper pick-up",
    53: "Cross not claimed",
    54: "Smother",
    55: "Offside provoked",
    56: "Shield ball opp",
    57: "Foul throw-in",
    58: "Penalty faced",
    59: "Keeper Sweeper",
    60: "Chance missed",
    61: "Ball touch",
    63: "Temp Save",
    64: "Resume",
    65: "Contentious referee decision",
    67: "50/50",
    68: "Referee Drop Ball",
    69: "Failed to Block",
    70: "Injury Time Announcement",
    71: "Coach Setup",
    72: "Caught Offside",
    73: "Other Ball Contact",
    74: "Blocked Pass",
    75: "Delayed start",
    76: "Early end",
    79: "Coverage interruption",
    80: "Drop of Ball",
    81: "Obstacle",
    82: "Control",
    83: "Attempted tackle",
    84: "Deleted After Review"
}

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

def get_home_away_info_from_json(match_info, live_data):
    """Extrae información de home/away del partido"""
    contestants = match_info.get('contestant', [])
    match_details = live_data.get('matchDetails', {})
    
    # Crear mapeo de equipos
    team_mapping = {}
    home_team_id = None
    away_team_id = None
    
    for contestant in contestants:
        team_id = contestant.get('id')
        team_name = contestant.get('name', 'N/A')  # ✅ ESTA LÍNEA ES CORRECTA
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

def detect_json_type(json_data):
    """
    Detecta si el JSON es de tipo MA2 (match stats) o MA3 (match events)
    """
    live_data = json_data.get('liveData', {})
    
    # Si tiene lineUp, es MA2 (match stats)
    if 'lineUp' in live_data:
        return 'MA2'
    
    # Si tiene event (y no lineUp), es MA3 (match events)
    elif 'event' in live_data:
        return 'MA3'
    
    # Por defecto, intentar detectar por contenido
    elif 'goal' in live_data or 'card' in live_data or 'substitute' in live_data:
        return 'MA2'
    
    else:
        return 'UNKNOWN'

def process_ma3_events_to_dataframe(json_data):
    """
    Procesa un JSON MA3 (match events) y extrae todos los eventos detallados
    """
    # Extraer información básica
    match_info = json_data.get('matchInfo', {})
    competition_info = match_info.get('competition', {})
    stage_info = match_info.get('stage', {})
    live_data = json_data.get('liveData', {})
    events = live_data.get('event', [])
    
    # Obtener información home/away
    home_away_info = get_home_away_info_from_json(match_info, live_data)
    
    # 🔍 DEBUG: Función para analizar qualifiers
    def debug_qualifiers(events, target_qualifier_name="Corner taken"):
        print(f"\n🔍 DEBUG: Analizando qualifier '{target_qualifier_name}'")
        
        # Contar en JSON crudo
        manual_count = 0
        found_qualifiers = set()
        
        for event in events:
            for q in event.get('qualifier', []):
                qualifier_id = q.get('qualifierId')
                if qualifier_id is not None:
                    qualifier_name = QUALIFIER_MAPPING.get(qualifier_id, f"qualifier {qualifier_id}")
                    found_qualifiers.add(f"{qualifier_id}: {qualifier_name}")
                    
                    if qualifier_name == target_qualifier_name:
                        manual_count += 1
                        print(f"   ✅ Encontrado en evento {event.get('eventId')}: qualifierId={qualifier_id}")
        
        print(f"📊 Total '{target_qualifier_name}' en JSON: {manual_count}")
        print(f"📋 Todos los qualifiers encontrados ({len(found_qualifiers)}):")
        for q in sorted(found_qualifiers):
            print(f"     {q}")
        
        return manual_count

    # Encontrar todos los qualifier IDs únicos
    qualifier_ids = set()
    for event in events:
        for q in event.get('qualifier', []):
            qualifier_id = q.get('qualifierId')  # ✅ Sin valor por defecto
            if qualifier_id is not None:  # ✅ Verificar que existe
                qualifier_name = QUALIFIER_MAPPING.get(qualifier_id, f"qualifier {qualifier_id}")
                qualifier_ids.add(qualifier_name)

    # Llamar el debug
    manual_count = debug_qualifiers(events, "Corner taken")
    
    # Crear columnas del DataFrame
    columns = [
        'Match ID', 'Event ID', 'Competition ID', 'Competition Name', 'Week', 'Stage ID', 'Stage Name',
        'EventId', 'timeStamp', 'contestantId', 'Team ID', 'Team Name', 'Team Position',
        'Is Home', 'Is Away', 'HT Home Score', 'HT Away Score', 'FT Home Score', 'FT Away Score',
        'periodId', 'timeMin', 'timeSec', 'playerId', 'playerName', 'typeId', 'Event Name', 
        'outcome', 'x', 'y'
    ] + list(qualifier_ids)
    
    # Procesar eventos
    events_data = []
    
    # 🔍 DEBUG: Verificar eventos específicos con Corner taken
    corner_event_ids = []
    for event in events:
        for q in event.get('qualifier', []):
            if q.get('qualifierId') == 6:  # Corner taken
                corner_event_ids.append(event.get('id'))
                break

    print(f"🔍 IDs de eventos con Corner taken: {corner_event_ids}")

    
    for event in events:
        contestant_id = event.get('contestantId', None)
        team_name = home_away_info['team_mapping'].get(contestant_id, {}).get('name', 'N/A') if contestant_id else 'N/A'
        team_position = home_away_info['team_mapping'].get(contestant_id, {}).get('position', 'N/A') if contestant_id else 'N/A'
        
        # Determinar si es home o away
        is_home = contestant_id == home_away_info['home_team_id']
        is_away = contestant_id == home_away_info['away_team_id']
        
        type_id = event.get('typeId', None)
        event_name = EVENT_TYPE_MAPPING.get(type_id, 'Unknown Event') if type_id else 'Unknown Event'
        
        event_info = {
            'Match ID': match_info.get('id', 'N/A'),
            'Event ID': event.get('id', None),  # ✅ Agregar el ID único
            'EventId': event.get('eventId', None),
            'Competition ID': competition_info.get('id', 'N/A'),
            'Competition Name': competition_info.get('name', 'N/A'),
            'Week': match_info.get('week', 'N/A'),
            'Stage ID': stage_info.get('id', 'N/A'),
            'Stage Name': stage_info.get('name', 'N/A'),
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
        
        # Inicializar todos los qualifiers a "No"
        for qualifier_name in qualifier_ids:
            event_info[qualifier_name] = "No"
        
        # Actualizar con valor o "Sí" para qualifiers presentes
        for q in event.get('qualifier', []):
            qualifier_id = q.get('qualifierId')
            if qualifier_id is not None:
                qualifier_name = QUALIFIER_MAPPING.get(qualifier_id, f"qualifier {qualifier_id}")
                value = q.get('value', None)
                event_info[qualifier_name] = value if value is not None else "Sí"

        events_data.append(event_info) 
    
    events_df = pd.DataFrame(events_data, columns=columns)
    # 🔍 DEBUG: Verificar resultado final
    if 'Corner taken' in events_df.columns:
        parquet_count = (events_df['Corner taken'] == 'Sí').sum()
        print(f"📊 Total 'Corner taken' en parquet: {parquet_count}")
        print(f"🔍 Diferencia: {manual_count - parquet_count}")
        
        if manual_count != parquet_count:
            print("❌ HAY INCONSISTENCIA!")
            # Mostrar algunos ejemplos
            corner_events = events_df[events_df['Corner taken'] == 'Sí']
            print(f"📋 Primeros eventos con Corner taken en parquet:")
            print(corner_events[['EventId', 'Event Name', 'Corner taken']].head())
    else:
        print("❌ ERROR: Columna 'Corner taken' no existe en el DataFrame!")
    return events_df

def process_json_to_dataframes(json_data):
    """
    Procesa un JSON y extrae los datos según su tipo (MA2 o MA3)
    """
    json_type = detect_json_type(json_data)
    
    if json_type == 'MA3':
        print("📊 Detectado: JSON de eventos (MA3)")
        events_df = process_ma3_events_to_dataframe(json_data)
        return {'match_events': events_df}
    
    elif json_type == 'MA2':
        print("📊 Detectado: JSON de match stats (MA2)")
        return process_ma2_stats_to_dataframes(json_data)
    
    else:
        print("❌ Tipo de JSON no reconocido")
        return {}

def process_ma2_stats_to_dataframes(json_data):
    """
    Procesa un JSON MA2 (match stats) y extrae los datos para cada parquet
    """
    # Extraer información básica
    match_info = json_data.get('matchInfo', {})
    competition_info = match_info.get('competition', {})
    stage_info = match_info.get('stage', {})
    live_data = json_data.get('liveData', {})
    line_ups = live_data.get('lineUp', [])
    
    # Obtener información home/away
    home_away_info = get_home_away_info_from_json(match_info, live_data)
    
    # Inicializar listas para cada tipo de datos
    player_stats_data = []
    team_stats_data = []
    team_officials_data = []
    match_events_data = []
    
    # 1. PROCESAR PLAYER STATS
    for line_up in line_ups:
        team_id = line_up.get('contestantId')
        team_name = home_away_info['team_mapping'].get(team_id, {}).get('name', 'N/A')
        team_position = home_away_info['team_mapping'].get(team_id, {}).get('position', 'N/A')
        
        # Determinar si es home o away
        is_home = team_id == home_away_info['home_team_id']
        is_away = team_id == home_away_info['away_team_id']
        
        for player in line_up.get('player', []):
            player_entry = {
                'Match ID': match_info.get('id', 'N/A'),
                'Competition ID': competition_info.get('id', 'N/A'),
                'Competition Name': competition_info.get('name', 'N/A'),
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
            
            # Añadir stats del jugador
            for stat in player.get('stat', []):
                stat_type = stat.get('type', '')
                stat_value = stat.get('value', 0)
                player_entry[stat_type] = stat_value
            
            player_stats_data.append(player_entry)
    
    # 2. PROCESAR TEAM STATS
    for line_up in line_ups:
        team_id = line_up.get('contestantId')
        team_name = home_away_info['team_mapping'].get(team_id, {}).get('name', 'N/A')
        team_position = home_away_info['team_mapping'].get(team_id, {}).get('position', 'N/A')
        
        is_home = team_id == home_away_info['home_team_id']
        is_away = team_id == home_away_info['away_team_id']
        
        for stat in line_up.get('stat', []):
            stat_info = {
                'Match ID': match_info.get('id', 'N/A'),
                'Competition ID': competition_info.get('id', 'N/A'),
                'Competition Name': competition_info.get('name', 'N/A'),
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
    
    # 3. PROCESAR TEAM OFFICIALS
    for line_up in line_ups:
        team_id = line_up.get('contestantId')
        team_name = home_away_info['team_mapping'].get(team_id, {}).get('name', 'N/A')
        
        for official in line_up.get('teamOfficial', []):
            official_entry = {
                'Match ID': match_info.get('id', 'N/A'),
                'Competition ID': competition_info.get('id', 'N/A'),
                'Competition Name': competition_info.get('name', 'N/A'),
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
    
    # 4. PROCESAR MATCH EVENTS (goals, cards, substitutions)
    # Detectar todos los qualifier names únicos
    qualifier_names = set()
    for goal in live_data.get('goal', []):
        for q in goal.get('qualifier', []):
            qualifier_id = q.get('qualifierId', '')
            qualifier_name = QUALIFIER_MAPPING.get(qualifier_id, f"qualifier {qualifier_id}")
            qualifier_names.add(qualifier_name)
    for card in live_data.get('card', []):
        for q in card.get('qualifier', []):
            qualifier_id = q.get('qualifierId', '')
            qualifier_name = QUALIFIER_MAPPING.get(qualifier_id, f"qualifier {qualifier_id}")
            qualifier_names.add(qualifier_name)
    for sub in live_data.get('substitute', []):
        for q in sub.get('qualifier', []):
            qualifier_id = q.get('qualifierId', '')
            qualifier_name = QUALIFIER_MAPPING.get(qualifier_id, f"qualifier {qualifier_id}")
            qualifier_names.add(qualifier_name)
    # Goals
    for goal in live_data.get('goal', []):
        event_entry = {
            'Match ID': match_info.get('id', 'N/A'),
            'Competition ID': competition_info.get('id', 'N/A'),
            'Competition Name': competition_info.get('name', 'N/A'),
            'Week': match_info.get('week', 'N/A'),
            'Stage ID': stage_info.get('id', 'N/A'),
            'Stage Name': stage_info.get('name', 'N/A'),
            'EventId': goal.get('optaEventId', 'N/A'),
            'typeId': 16,  # Goal event type
            'Event Name': 'Goal',
            'periodId': goal.get('periodId', None),
            'timeMin': goal.get('timeMin', None),
            'timeSec': goal.get('timeMinSec', None),
            'contestantId': goal.get('contestantId', None),
            'Team ID': goal.get('contestantId', None),
            'Team Name': home_away_info['team_mapping'].get(goal.get('contestantId'), {}).get('name', 'N/A'),
            'Team Position': home_away_info['team_mapping'].get(goal.get('contestantId'), {}).get('position', 'N/A'),
            'Is Home': goal.get('contestantId') == home_away_info['home_team_id'],
            'Is Away': goal.get('contestantId') == home_away_info['away_team_id'],
            'HT Home Score': home_away_info['ht_home'],
            'HT Away Score': home_away_info['ht_away'],
            'FT Home Score': home_away_info['ft_home'],
            'FT Away Score': home_away_info['ft_away'],
            'playerId': goal.get('scorerId', None),
            'playerName': goal.get('scorerName', None),
            'outcome': None,
            'x': None,
            'y': None,
            'timeStamp': goal.get('timestamp', None),
        }
        # Añadir qualifiers para goals
        for qualifier_name in qualifier_names:
            event_entry[qualifier_name] = "No"
        for q in goal.get('qualifier', []):
            qualifier_id = q["qualifierId"]
            qualifier_name = QUALIFIER_MAPPING.get(qualifier_id, f"qualifier {qualifier_id}")
            value = q.get('value', None)
            event_entry[qualifier_name] = value if value is not None else "Sí"
        
        match_events_data.append(event_entry) 
    
    # Cards
    for card in live_data.get('card', []):
        event_entry = {
            'Match ID': match_info.get('id', 'N/A'),
            'Competition ID': competition_info.get('id', 'N/A'),
            'Competition Name': competition_info.get('name', 'N/A'),
            'Week': match_info.get('week', 'N/A'),
            'Stage ID': stage_info.get('id', 'N/A'),
            'Stage Name': stage_info.get('name', 'N/A'),
            'EventId': card.get('optaEventId', 'N/A'),
            'typeId': 17,  # Card event type
            'Event Name': 'Card',
            'periodId': card.get('periodId', None),
            'timeMin': card.get('timeMin', None),
            'timeSec': card.get('timeMinSec', None),
            'contestantId': card.get('contestantId', None),
            'Team ID': card.get('contestantId', None),
            'Team Name': home_away_info['team_mapping'].get(card.get('contestantId'), {}).get('name', 'N/A'),
            'Team Position': home_away_info['team_mapping'].get(card.get('contestantId'), {}).get('position', 'N/A'),
            'Is Home': card.get('contestantId') == home_away_info['home_team_id'],
            'Is Away': card.get('contestantId') == home_away_info['away_team_id'],
            'HT Home Score': home_away_info['ht_home'],
            'HT Away Score': home_away_info['ht_away'],
            'FT Home Score': home_away_info['ft_home'],
            'FT Away Score': home_away_info['ft_away'],
            'playerId': card.get('playerId', None),
            'playerName': card.get('playerName', None),
            'outcome': None,
            'x': None,
            'y': None,
            'timeStamp': card.get('timestamp', None),
        }
        # Añadir qualifiers para cards
        for qualifier_name in qualifier_names:
            event_entry[qualifier_name] = "No"
        for q in card.get('qualifier', []):
            qualifier_id = q["qualifierId"]
            qualifier_name = QUALIFIER_MAPPING.get(qualifier_id, f"qualifier {qualifier_id}")
            value = q.get('value', None)
            event_entry[qualifier_name] = value if value is not None else "Sí"
            
        match_events_data.append(event_entry)
    
    # Substitutions
    for sub in live_data.get('substitute', []):
        event_entry = {
            'Match ID': match_info.get('id', 'N/A'),
            'Competition ID': competition_info.get('id', 'N/A'),
            'Competition Name': competition_info.get('name', 'N/A'),
            'Week': match_info.get('week', 'N/A'),
            'Stage ID': stage_info.get('id', 'N/A'),
            'Stage Name': stage_info.get('name', 'N/A'),
            'EventId': sub.get('optaEventId', 'N/A'),
            'typeId': 19,  # Player on event type
            'Event Name': 'Player on',
            'periodId': sub.get('periodId', None),
            'timeMin': sub.get('timeMin', None),
            'timeSec': sub.get('timeMinSec', None),
            'contestantId': sub.get('contestantId', None),
            'Team ID': sub.get('contestantId', None),
            'Team Name': home_away_info['team_mapping'].get(sub.get('contestantId'), {}).get('name', 'N/A'),
            'Team Position': home_away_info['team_mapping'].get(sub.get('contestantId'), {}).get('position', 'N/A'),
            'Is Home': sub.get('contestantId') == home_away_info['home_team_id'],
            'Is Away': sub.get('contestantId') == home_away_info['away_team_id'],
            'HT Home Score': home_away_info['ht_home'],
            'HT Away Score': home_away_info['ht_away'],
            'FT Home Score': home_away_info['ft_home'],
            'FT Away Score': home_away_info['ft_away'],
            'playerId': sub.get('playerOnId', None),
            'playerName': sub.get('playerOnName', None),
            'outcome': None,
            'x': None,
            'y': None,
            'timeStamp': sub.get('timestamp', None),
        }
        
        # Añadir qualifiers para substitutions
        for qualifier_name in qualifier_names:
            event_entry[qualifier_name] = "No"
        for q in sub.get('qualifier', []):
            qualifier_id = q["qualifierId"]
            qualifier_name = QUALIFIER_MAPPING.get(qualifier_id, f"qualifier {qualifier_id}")
            value = q.get('value', None)
            event_entry[qualifier_name] = value if value is not None else "Sí"
            
        match_events_data.append(event_entry)
    
    # Crear DataFrames
    dataframes = {
        'player_stats': pd.DataFrame(player_stats_data),
        'team_stats': pd.DataFrame(team_stats_data),
        'team_officials': pd.DataFrame(team_officials_data),
        'match_events': pd.DataFrame(match_events_data)
    }
    
    # Procesar team_stats para pivotear
    if not dataframes['team_stats'].empty:
        metadata_cols = ['Team ID', 'Team Name', 'Team Position', 'Is Home', 'Is Away', 
                'HT Home Score', 'HT Away Score', 'FT Home Score', 'FT Away Score',
                'Match ID', 'Competition ID', 'Competition Name', 'Week', 'Stage ID', 'Stage Name']
        
        dataframes['team_stats'] = dataframes['team_stats'].pivot(
            index=['Team ID', 'Team Name', 'Team Position', 'Is Home', 'Is Away', 
                   'HT Home Score', 'HT Away Score', 'FT Home Score', 'FT Away Score',
                   'Match ID', 'Competition ID', 'Competition Name', 'Week', 'Stage ID', 'Stage Name'],
            columns='Stat Type', 
            values='Total'
        ).reset_index()
        dataframes['team_stats'] = dataframes['team_stats'].fillna(0)
    
    # Convertir columnas numéricas problemáticas a string
    for col in dataframes['team_stats'].columns:
        if col not in metadata_cols:
            dataframes['team_stats'][col] = dataframes['team_stats'][col].astype(str)

    return dataframes

def append_to_existing_parquets(new_dataframes, folder="."):
    """
    Añade los nuevos datos a los parquets existentes
    """
    file_config = {
        'player_stats': {
            'filename': 'player_stats.parquet',
            'duplicate_keys': ['Match ID', 'Player ID']
        },
        'team_stats': {
            'filename': 'team_stats.parquet', 
            'duplicate_keys': ['Match ID', 'Team ID']
        },
        'team_officials': {
            'filename': 'team_officials.parquet',
            'duplicate_keys': ['Match ID', 'Team ID', 'Official ID']
        },
        'match_events': {
            'filename': 'abp_events.parquet',
            'duplicate_keys': []  # ✅ Sin deduplicación para eventos únicos
        }
    }
    
    for data_type, new_df in new_dataframes.items():
        if not new_df.empty and data_type in file_config:
            config = file_config[data_type]
            filename = f"{folder}/{config['filename']}"
            duplicate_keys = config['duplicate_keys']
            
            print(f"🔄 Procesando {data_type}...")
            
            try:
                if os.path.exists(filename):
                    existing_df = pd.read_parquet(filename)
                    
                    # NUEVO: Adaptar nuevos datos al formato existente
                    for col in new_df.columns:
                        if col in existing_df.columns:
                            existing_dtype = existing_df[col].dtype
                            
                            # Si es numérico, convertir a numérico
                            if pd.api.types.is_numeric_dtype(existing_dtype):
                                new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                                if existing_dtype == 'int64':
                                    new_df[col] = new_df[col].fillna(0).astype('int64')
                                elif existing_dtype == 'float64':
                                    new_df[col] = new_df[col].astype('float64')
                            
                            # Si es string, mantener como string
                            elif existing_dtype == 'object':
                                new_df[col] = new_df[col].astype(str)
                    
                    # Limpiar tipos de datos de qualifiers
                    for col in new_df.columns:
                        if col in QUALIFIER_MAPPING.values() or col.startswith('qualifier'):
                            # Procesar qualifiers manteniendo valores o convirtiendo a Sí/No
                            new_df[col] = new_df[col].astype(str).replace({'0': 'No', '1': 'Sí', 'nan': 'No', 'None': 'No'})
                            if col in existing_df.columns:
                                existing_df[col] = existing_df[col].astype(str).replace({'0': 'No', '1': 'Sí', 'nan': 'No', 'None': 'No'})
                    
                    # Verificar claves de duplicación
                    available_keys = [key for key in duplicate_keys if key in new_df.columns and key in existing_df.columns]
                    
                    if available_keys:
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        combined_df = combined_df.drop_duplicates(subset=available_keys, keep='last')
                        print(f"💾 Actualizado: {filename} ({len(existing_df)} → {len(combined_df)} filas)")
                        combined_df.to_parquet(filename, index=False)
                    else:
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        print(f"💾 Actualizado (sin deduplicación): {filename} ({len(existing_df)} → {len(combined_df)} filas)")
                        combined_df.to_parquet(filename, index=False)
                else:
                    print(f"💾 Creado: {filename} ({len(new_df)} filas)")
                    new_df.to_parquet(filename, index=False)
                    
                print(f"   ✅ {data_type} guardado exitosamente")
                    
            except Exception as e:
                print(f"❌ Error con {data_type}: {e}")

# Función principal para procesar un archivo JSON
def process_json_file(json_file_path):
    """
    Procesa un archivo JSON y lo añade a los parquets existentes
    """
    print(f"📄 Procesando archivo: {json_file_path}")
    
    # Leer el JSON
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Procesar y extraer datos
    dataframes = process_json_to_dataframes(json_data)
    
    if dataframes:
        # Añadir a parquets existentes
        append_to_existing_parquets(dataframes)
        print(f"✅ Archivo {json_file_path} procesado correctamente")
    else:
        print(f"❌ No se pudo procesar {json_file_path}")

def process_json_folder(folder_path, file_extension=".json"):
    """
    Procesa todos los archivos JSON de una carpeta
    """
    import glob
    
    json_files = glob.glob(os.path.join(folder_path, f"*{file_extension}"))
    
    if not json_files:
        print(f"❌ No se encontraron archivos {file_extension} en {folder_path}")
        return
    
    print(f"📁 Encontrados {len(json_files)} archivos JSON en {folder_path}")
    
    for json_file in json_files:
        try:
            process_json_file(json_file)
        except Exception as e:
            print(f"❌ Error procesando {json_file}: {e}")
            continue
    
    print(f"✅ Procesamiento de carpeta completado")

# Funciones de utilidad
def show_parquet_summary(folder="datos_opta_parquet"):
    """
    Muestra un resumen de los datos en los parquets
    """
    file_config = {
        'player_stats': 'player_stats.parquet',
        'team_stats': 'team_stats.parquet', 
        'player_xg_stats': 'player_xg_stats.parquet',
        'xg_events': 'xg_events.parquet',
        'match_events': 'abp_events.parquet',
        'team_officials': 'team_officials.parquet'
    }
    
    print("📊 RESUMEN DE PARQUETS:")
    for data_type, filename in file_config.items():
        filepath = f"./{filename}"
        if os.path.exists(filepath):
            try:
                df = pd.read_parquet(filepath)
                unique_matches = df['Match ID'].nunique() if 'Match ID' in df.columns else 'N/A'
                print(f"   ✅ {data_type}: {len(df):,} filas | {unique_matches} partidos únicos")
            except Exception as e:
                print(f"   ❌ {data_type}: Error leyendo archivo - {e}")
        else:
            print(f"   📄 {data_type}: archivo no existe")

def backup_parquets(source_folder="datos_opta_parquet", backup_folder=None):
    """
    Crea una copia de seguridad de los parquets
    """
    import shutil
    from datetime import datetime
    
    if backup_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder = f"{source_folder}_backup_{timestamp}"
    
    if os.path.exists(source_folder):
        shutil.copytree(source_folder, backup_folder)
        print(f"💾 Backup creado: {backup_folder}")
    else:
        print(f"❌ Carpeta {source_folder} no existe")

# ====================================
# EJEMPLOS DE USO
# ====================================

# Ejemplo 1: Procesar un archivo individual
# process_json_file("oviedo_vs_mirandes_match_stats.json")

# Ejemplo 2: Procesar una carpeta completa
process_json_folder(".")

# Ejemplo 3: Crear backup antes de procesar
# backup_parquets()
# process_json_folder("./nuevos_jsons/")

# Ejemplo 4: Ver resumen de datos
show_parquet_summary()

print("🎯 PROCESADOR JSON A PARQUET CARGADO")
print("📋 Funciones disponibles:")
print("   • process_json_file(archivo.json) - Procesa un archivo individual")
print("   • process_json_folder(carpeta/) - Procesa todos los JSON de una carpeta") 
print("   • show_parquet_summary() - Muestra resumen de datos existentes")
print("   • backup_parquets() - Crea backup de los parquets")