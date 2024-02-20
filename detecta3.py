import torch
from PIL import Image
import csv
from datetime import datetime
import matplotlib.pyplot as plt

# Cargar el modelo
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Cargar la imagen
img = Image.open('imagen.jpg')  # Cambia esto a la ruta de tu imagen

# Realizar la inferencia
results = model(img)

# Filtrar los resultados para obtener solo las detecciones de 'persona'
person_detections = [detection for detection in results.xyxy[0] if detection[-1] == 0]

# Contar el número de personas
num_people = len(person_detections)

# Obtener la fecha y hora actual
fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Crear el nombre del archivo CSV
nombre_archivo = f"conteo_{fecha_hora}.csv"

# Escribir el número de personas detectadas en el archivo CSV
with open(nombre_archivo, 'w', newline='') as archivo_csv:
    writer = csv.writer(archivo_csv)
    writer.writerow(['Fecha y Hora', 'Número de Personas'])
    writer.writerow([fecha_hora, num_people])

print(f'Número de personas detectadas: {num_people}')
print(f'El conteo se ha guardado en el archivo: {nombre_archivo}')

# Mostrar la imagen con las detecciones
results.show()
plt.show()
