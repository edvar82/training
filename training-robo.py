from ultralytics import YOLO

model = YOLO('yolov8n.pt')  

model.train(data='./datasets/roboflow-data/data.yaml', epochs=30) 

model.save('roboflow.pt')  # save final model to roboflow.pt