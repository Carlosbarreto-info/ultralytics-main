## Departamento informatico de Cencosu 
## Proyecto Cuenta Personas.

import torch
from PIL import Image
import csv
from datetime import datetime
import configparser

# Leer el archivo de configuración
config = configparser.ConfigParser()
config.read('conf.ini')

# Obtener el valor de pventa del archivo de configuración
pventa = config.get('Configuracion', 'pventa')

# Cargar el modelo
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Cargar la imagen
img = Image.open('imagen.jpg')  # Cambia esto a la ruta de tu imagen

# Realizar la inferencia
results = model(img)

# Filtrar los resultados para obtener solo las detecciones de 'persona'
persona_deteccion= [detection for detection in results.xyxy[0] if detection[-1] == 0]


# Contar el número de personas
num_personas = len(persona_deteccion)
personas=num_personas

# Obtener la fecha y hora actual
fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Crear el nombre del archivo CSV
nombre_archivo = "registro.csv"

# Escribir el número de personas detectadas en el archivo CSV junto con el valor de pventa
with open(nombre_archivo, 'a', newline='') as archivo_csv:
    writer = csv.writer(archivo_csv)
    # Solo escribir encabezados si el archivo está vacío
    if archivo_csv.tell() == 0:
        writer.writerow(['Fecha y Hora','Número de Personas', 'pventa'])
    writer.writerow([fecha_hora, personas, pventa])

print(f'Número de personas detectadas: {personas}')
print(f'El conteo se ha agregado al archivo: {nombre_archivo}')

# Mostrar la imagen con las detecciones
results.show()
