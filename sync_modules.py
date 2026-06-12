import os
import re
import json
import base64
import urllib.request
import urllib.error

BASE_URL = "https://update.lemoe.link"
LICENSE_PATH = "config/licencia.asc"
CONFIG_PATH = "config/config.json"

def main():
    if not os.path.exists(LICENSE_PATH):
        # Modo estándar: Si no hay clave de acceso, salir silenciosamente
        return

    print("==================================================")
    print("  LeMoE - Sincronizando módulos extendidos...")
    print("==================================================\n")

    with open(LICENSE_PATH, 'r', encoding='utf-8') as f:
        license_text = f.read()

    match = re.search(r'\{.*?\}', license_text, re.DOTALL)
    if not match:
        print(" Error: Clave de acceso corrupta o inválida.")
        return
        
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        print(" Error: No se pudo verificar la estructura de la clave.")
        return

    allowed_plugins = payload.get("allowed_plugins", [])
    if not allowed_plugins:
        print(" Info: Clave válida, pero no hay módulos adicionales asignados.")
        return

    channel = "release"
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                channel = cfg.get("updater", {}).get("channel", "release")
        except:
            pass

    base64_license = base64.b64encode(license_text.encode('utf-8')).decode('utf-8')
    headers = {"X-Lemoe-License": base64_license}

    os.makedirs("plugins/bis", exist_ok=True)
    os.makedirs("tools/bis", exist_ok=True)

    for plugin in allowed_plugins:
        print(f"Sincronizando '{plugin}'...")
        success = False
        
        url_plugin = f"{BASE_URL}/plugins/{plugin}/{channel}/{plugin}.py"
        try:
            req = urllib.request.Request(url_plugin, headers=headers)
            with urllib.request.urlopen(req) as response:
                content = response.read()
                with open(f"plugins/bis/{plugin}.py", "wb") as out_file:
                    out_file.write(content)
                print(f"   Instalado en plugins/bis/{plugin}.py")
                success = True
        except urllib.error.HTTPError as e:
            if e.code == 403:
                print(f"   Acceso Denegado: Clave revocada o expirada.")
                continue
            elif e.code != 404:
                print(f"   Error remoto ({e.code}): {e.reason}")
                
        if success:
            continue
            
        url_tool = f"{BASE_URL}/tools/{plugin}/{channel}/{plugin}.py"
        try:
            req = urllib.request.Request(url_tool, headers=headers)
            with urllib.request.urlopen(req) as response:
                content = response.read()
                with open(f"tools/bis/{plugin}.py", "wb") as out_file:
                    out_file.write(content)
                print(f"   Instalado en tools/bis/{plugin}.py")
                success = True
        except urllib.error.HTTPError as e:
            if e.code == 403:
                print(f"   Acceso Denegado: Clave revocada o expirada.")
            elif e.code == 404:
                print(f"   No encontrado en el repositorio remoto.")
            else:
                print(f"   Error HTTP ({e.code}): {e.reason}")
            
        if not success:
            print(f"   Fallo en la sincronización del módulo '{plugin}'.")

if __name__ == "__main__":
    main()
