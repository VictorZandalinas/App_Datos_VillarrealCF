import requests
import hashlib
import time
import pandas as pd

# Usa TUS credenciales que funcionan
outletApiKey = '10lthl3y5chwn1m0fa4mfg3bqy'
secretKey = '1c9prnxo6sjwj155dgf0qu2ep3'

def requestHeaders():
    """Función OAuth idéntica a tu notebook"""
    timestamp = int(round(time.time() * 1000))
    post_url = f'https://oauth.performgroup.com/oauth/token/{outletApiKey}?_fmt=json&_rt=b'
    
    key = str.encode(outletApiKey + str(timestamp) + secretKey)
    unique_hash = hashlib.sha512(key).hexdigest()
    
    oauthHeaders = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f'Basic {unique_hash}',
        'Timestamp': str(timestamp)
    }
    
    BODY = {
        'grant_type': 'client_credentials',
        'scope': 'b2b-feeds-auth'
    }
    
    response = requests.post(post_url, data=BODY, headers=oauthHeaders)
    access_token = response.json()['access_token']
    return {'Authorization': f'Bearer {access_token}'}

def get_competitions():
    """Obtiene competiciones usando MA1 básico como en tu notebook"""
    params = {
    "_fmt": "json",
        "_pgSz": "500",
        "_pgNm": "1",
        "cvlv": "15",  # Lo que funcionaba antes
        "live": "yes",
        "_rt": "b"
    }
    
    url = f'https://api.performfeeds.com/soccerdata/match/{outletApiKey}/'
    response = requests.get(url, headers=requestHeaders(), params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return pd.DataFrame()
    
    data = response.json()
    matches = data.get('match', [])
    
    stages = {}
    for match in matches:
        match_info = match.get('matchInfo', {})
        competition = match_info.get('competition', {})
        stage = match_info.get('stage', {})
        
        stage_id = stage.get('id')
        if stage_id and stage_id not in stages:
            stages[stage_id] = {
                'Competition': competition.get('name', 'N/A'),
                'Competition ID': competition.get('id', 'N/A'),
                'Stage': stage.get('name', 'N/A'),
                'Stage ID': stage_id,
                'Start Date': stage.get('startDate', 'N/A'),
                'End Date': stage.get('endDate', 'N/A')
            }
    
    return pd.DataFrame(list(stages.values()))

if __name__ == "__main__":
    df = get_competitions()
    print(f"Competiciones encontradas: {len(df)}")
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("No se encontraron competiciones")