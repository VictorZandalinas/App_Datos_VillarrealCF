import requests
import json

# --- CONFIGURACIÓN ---
CLIENT_ID = '58191b89-cee4-11ed-a09d-ee50c5eb4bb5'
SUBSCRIPTION_KEY = '729f9154234d4ff3bb0a692c6a0510c4'
USERNAME = 'b2bvillarealcf@mediacoach.es'
PASSWORD = 'r728-FHj3RE!'

def buscar_endpoint_valido():
    print("🚀 BUSCADOR DE ENDPOINTS MEDIACOACH")
    print("-" * 50)

    # 1. Obtener Token
    token_url = 'https://id.mediacoach.es/connect/token'
    payload = {'client_id': CLIENT_ID, 'scope': 'b2bapiclub-api', 'grant_type': 'password', 'username': USERNAME, 'password': PASSWORD}
    
    try:
        r_token = requests.post(token_url, data=payload, timeout=10)
        token = r_token.json().get('access_token')
        print("✅ Token obtenido.")
    except:
        print("❌ Error obteniendo token.")
        return

    # 2. Probar los 4 endpoints posibles
    base_url = "https://club-api.mediacoach.es"
    endpoints = [
        "/Championships/seasons",
        "/seasons",
        "/Championships",
        "/api/seasons"
    ]

    headers = {
        'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY,
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }

    print("\n🔍 Probando rutas disponibles...")
    for path in endpoints:
        url = f"{base_url}{path}"
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                print(f"   ✨ [EXITO 200] -> {path}")
                print(f"      Muestra de datos: {r.text[:100]}...")
                # Si este funciona, paramos y lo marcamos como el bueno
                return path
            else:
                print(f"   ❌ [ERROR {r.status_code}] -> {path}")
        except Exception as e:
            print(f"   ⚠️ [FALLO CONEXIÓN] -> {path}: {e}")

    print("\n🛑 Ningún endpoint respondió con éxito.")
    return None

if __name__ == "__main__":
    path_valido = buscar_endpoint_valido()
    if path_valido:
        print(f"\n✅ USA ESTE PATH EN TU CÓDIGO: {path_valido}")