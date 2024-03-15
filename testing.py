from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

model.train(data='./final/data.yaml', epochs=10)