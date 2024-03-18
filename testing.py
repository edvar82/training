from ultralytics import YOLO


if __name__ == '__main__':
    # Load model
    model = YOLO('yolov8n.pt')

    model.train(data='./roboflow/data.yaml', epochs=15)
