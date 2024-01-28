from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLOv8m model. M medium
model = YOLO('yolov5s.pt')

# Run inference on 'imagen.jpg'
results = model('imagen.jpg')  # results list

# Save bounding box coordinates to a text file
with open('resultado.txt', 'w') as f:
    for i, pred in enumerate(results[0]['labels']):  # Assuming you're interested in the first set of predictions
        class_id = pred[0]
        confidence = pred[1]
        bbox = pred[2:]
        f.write(f"Prediction {i+1}: Class {class_id}, Confidence {confidence}\n")
        f.write(f"Bounding Box: {bbox}\n")

# Show the results
for r in results:
    im_array = r['img']  # Get image array from result
    im = Image.fromarray(im_array)  # Create PIL image from numpy array
    im.show()  # Show image

