"""
Orquestador del pipeline completo de recomendación.

Ejecuta en secuencia:
  1. Entrenamiento de modelos (data_clean + feature_engineering + train)
  2. Exportación de datos para Power BI
  3. Levantamiento de API (FastAPI/uvicorn) y frontend (Streamlit)

Uso:
  python run_pipeline.py          # pipeline completo + servers
  python run_pipeline.py --no-servers  # solo pipeline, sin levantar servers
"""

import subprocess
import sys
import os
import time
import signal
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent

API_HOST = "127.0.0.1"
API_PORT = 8000
FRONTEND_PORT = 8501


def detectar_python():
    for name in ("venv", ".venv"):
        python = ROOT / name / "Scripts" / "python.exe"
        if python.exists():
            return str(python)
    return sys.executable


def liberar_puerto(puerto):
    if os.name != "nt":
        return
    resultado = subprocess.run(
        f'netstat -ano | findstr ":{puerto} " | findstr "LISTENING"',
        capture_output=True, text=True, shell=True,
    )
    for linea in resultado.stdout.strip().splitlines():
        pid = linea.split()[-1]
        try:
            subprocess.run(f"taskkill /F /PID {pid}", shell=True,
                           capture_output=True)
            print(f"  Proceso anterior en puerto {puerto} finalizado (PID {pid})")
        except Exception:
            pass


def banner():
    print("=" * 50)
    print("  Orquestador - Sistema de Recomendacion")
    print("=" * 50)
    print()


def paso(num, total, descripcion):
    print(f"[{num}/{total}] {descripcion}")


def ejecutar_paso(comando, descripcion_paso, num, total):
    paso(num, total, descripcion_paso)
    resultado = subprocess.run(
        comando,
        cwd=str(ROOT),
        shell=(os.name == "nt"),
    )
    if resultado.returncode != 0:
        print(f"\n  ERROR: '{descripcion_paso}' fallo (codigo {resultado.returncode})")
        sys.exit(1)
    print(f"  OK {descripcion_paso}\n")
    return True


def verificar_archivos_modelos():
    warm = ROOT / "Models" / "warm_start_lightgbm.joblib"
    cold = ROOT / "Models" / "cold_start_content_based.joblib"
    if not warm.exists() or not cold.exists():
        print("  ERROR: No se encontraron los archivos de modelo en Models/")
        sys.exit(1)


def limpiar_procesos(procesos):
    print("\nDeteniendo servidores...")
    for proc in procesos:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    print("Servidores detenidos correctamente.")


def main():
    banner()

    python = detectar_python()
    no_servers = "--no-servers" in sys.argv
    total = 3 if no_servers else 5

    # --- Paso 1: Entrenar modelos ---
    ejecutar_paso(
        [python, "SRC/pipeline_modelos.py"],
        "Entrenando modelos...",
        1, total,
    )

    # --- Paso 2: Exportar para Power BI ---
    ejecutar_paso(
        [python, "SRC/export_powerbi.py"],
        "Exportando datos para PowerBI...",
        2, total,
    )

    if no_servers:
        print("=" * 50)
        print("  Pipeline completado (sin servers).")
        print("=" * 50)
        return

    # --- Verificar modelos ---
    verificar_archivos_modelos()

    # --- Paso 3: Levantar API ---
    paso(3, total, "Levantando API (FastAPI)...")
    liberar_puerto(API_PORT)
    api_proc = subprocess.Popen(
        [python, "-m", "uvicorn", "SRC.appi:app",
         "--host", API_HOST, "--port", str(API_PORT)],
        cwd=str(ROOT),
        shell=(os.name == "nt"),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    print(f"  OK API en http://{API_HOST}:{API_PORT}\n")

    time.sleep(2)

    # --- Paso 4: Levantar Frontend ---
    paso(4, total, "Levantando Frontend (Streamlit)...")
    front_proc = subprocess.Popen(
        [python, "-m", "streamlit", "run", "SRC/app_front.py",
         "--server.headless", "true"],
        cwd=str(ROOT),
        shell=(os.name == "nt"),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    frontend_url = f"http://localhost:{FRONTEND_PORT}"
    print(f"  OK Frontend en {frontend_url}\n")

    webbrowser.open(frontend_url)

    # --- Paso 5: Esperar ---
    paso(5, total, "Listo!")
    print("Presiona Ctrl+C para detener los servidores.\n")

    procesos = [api_proc, front_proc]

    def signal_handler(sig, frame):
        limpiar_procesos(procesos)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            time.sleep(1)
            for proc in procesos:
                if proc.poll() is not None:
                    print(f"\n  AVISO: Un servidor se detuvo inesperadamente (codigo {proc.returncode})")
                    limpiar_procesos(procesos)
                    sys.exit(1)
    except KeyboardInterrupt:
        limpiar_procesos(procesos)


if __name__ == "__main__":
    main()
