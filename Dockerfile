# Usamos una imagen base ligera de Python
FROM python:3.11-slim

# Instalamos la dependencia del sistema necesaria para LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiamos primero el archivo de requerimientos
COPY requirements.txt .

# Instalamos las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto de los archivos
COPY SRC/ ./SRC/
COPY Models/ ./Models/

# Exponemos el puerto
EXPOSE 8000

# Comando para ejecutar la API
CMD ["uvicorn", "SRC.appi:app", "--host", "0.0.0.0", "--port", "8000"]