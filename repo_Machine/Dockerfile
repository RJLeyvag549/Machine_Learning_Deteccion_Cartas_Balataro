FROM ultralytics/ultralytics:latest-python

WORKDIR /app

# Instalar dependencias para visualización y OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-tk

COPY . .

# Comando por defecto
CMD ["python3", "detect.py"]