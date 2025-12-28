import cv2
from ultralytics import YOLO
import time
import os

# Forzamos que no intente usar la GPU para evitar el error de librerías
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = YOLO('best.pt')
rtmp_url = "rtmp://rtmp-server:1935/live/balatro"

def run_detection():
    print("--- Iniciando Sistema de Detección Balatro (Modo CPU) ---")
    cap = cv2.VideoCapture(rtmp_url)
    
    while not cap.isOpened():
        print("Esperando señal de OBS en rtmp://localhost:1935/live/balatro...")
        time.sleep(3)
        cap = cv2.VideoCapture(rtmp_url)

    print("¡Señal recibida! Empezando procesamiento...")
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Se perdió la señal de video.")
            break

        # Inferencia en CPU (device='cpu')
        results = model.predict(frame, conf=0.5, device='cpu', verbose=False)

        for r in results:
            if len(r.boxes) > 0:
                # Si detecta algo, nos avisa en la terminal
                n = len(r.boxes)
                clases = [model.names[int(c)] for c in r.boxes.cls]
                print(f"Detectadas {n} cartas: {', '.join(clases)}")
                
                # Guarda una imagen del resultado en tu carpeta del escritorio
                r.save(filename='ultima_deteccion.jpg')

    cap.release()

if __name__ == "__main__":
    run_detection()