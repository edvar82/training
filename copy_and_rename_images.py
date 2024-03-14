from shutil import copy2
from tqdm import tqdm
from glob import glob
import json
import os
from ultralytics import YOLO

def adapt_label_by_dataset(label_path, classes_dict, dataset):
    labels = []
    with open(label_path, "r") as file:
        for row in file.readlines():
            row = row.strip().split(sep=" ")
            c = int(row[0])
            class_name = classes_dict[dataset][str(c)]

            if class_name not in classes_dict["final"]:
                continue

            x, y, w, h = [float(i) for i in row[1:]]

            c = classes_dict["final"][class_name]
            labels.append([c, x, y, w, h])

    return "\n".join([f"{c} {x} {y} {w} {h}" for c, x, y, w, h in labels])

def load_model(model_path):
    """ Ler o modelo de inferência

    :param model_path: Caminho para o modelo
    :type model_path: str
    :return: Modelo de inferência
    :rtype: YoloV8
    """
    return YOLO(model_path)

def predict(model, image):
    """ Fazer a inferência

    :param model: Modelo de inferência
    :type model: YoloV8
    :param image: Imagem
    :type image: np.ndarray
    :return: Labels
    :rtype: List[List[int, float, float, float, float]]
    """
    return model(image)


if __name__ == "__main__":
    MAIN_FOLDERS = ["roboflow", "kaggle"]
    TARGET_FOLDERS = ["train", "test", "valid"]
    DEST_MAIN_FOLDER = "final"
    CSV_FILE_PATH = "final.csv"
    CLASS_DICT_PATH = os.path.join("final", "class_dict.json")

    class_dict_roboflow = json.load(open(os.path.join("roboflow", "classes.json"), "r"))
    class_dict_kaggle = json.load(open(os.path.join("kaggle", "classes.json"), "r"))
    final_class_dict = json.load(open(CLASS_DICT_PATH, "r"))

    classes_dict = {
        "roboflow": class_dict_roboflow,
        "kaggle": class_dict_kaggle,
        "final": final_class_dict
    }

    file_pairs = []

    current_image_index = 0
    for dataset_folder in MAIN_FOLDERS:
        if dataset_folder == "roboflow":
            model = load_model("./models/kaggle.pt")
        else:
            model = load_model("./models/roboflow.pt")
        for target_folder in tqdm(TARGET_FOLDERS):
            FOLDER_PATH = os.path.join(dataset_folder, target_folder)
            IMAGES_PATH = os.path.join(FOLDER_PATH, "images")
            LABELS_PATH = os.path.join(FOLDER_PATH, "labels")
            DEST_PATH = os.path.join(DEST_MAIN_FOLDER, target_folder)

            os.makedirs(os.path.join(DEST_PATH, "images"), exist_ok=True)
            os.makedirs(os.path.join(DEST_PATH, "labels"), exist_ok=True)

            images = glob(os.path.join(IMAGES_PATH, "*.jpg"))

            for image_path in images:
                label_path = image_path.replace(".jpg", ".txt").replace("images", "labels")

                if not os.path.exists(label_path) or not os.path.exists(image_path):
                    print(image_path, label_path)
                    continue

                adapted_labels = adapt_label_by_dataset(label_path, classes_dict, dataset_folder)

                ## Quando estiver no dataset do Kaggle, executar o modelo do Roboflow
                ## Quando estiver no dataset do Roboflow, executar o modelo do Kaggle
                ## Inferência do modelo
                ## Adiciona as labels ao adapted_labels
                predictions = predict(model, image_path)
                for prediction in predictions:
                    boxes = prediction.boxes.xyxy
                    confidences = prediction.boxes.conf
                    classes = prediction.boxes.cls
                    for box, conf, cls in zip(boxes, confidences, classes):
                        x1, y1, x2, y2 = box.tolist()
                        confidence = conf.item()
                        class_id = cls.item()
                        class_name = prediction.names[class_id]
                        if class_name not in classes_dict["final"]:
                            continue    
                        c = classes_dict["final"][class_name]
                        w = x2 - x1
                        h = y2 - y1
                        adapted_labels += f"\n{c} {x1} {y1} {w} {h}"

                dest_image_path = os.path.join(DEST_PATH, "images", f"{current_image_index:07d}.jpg")
                dest_label_path = os.path.join(DEST_PATH, "labels", f"{current_image_index:07d}.txt")

                copy2(image_path, dest_image_path)

                with open(dest_label_path, "w") as file:
                    file.write(adapted_labels)

                file_pairs.append((dest_image_path, dest_label_path))

                current_image_index += 1

    with open(CSV_FILE_PATH, "w") as file:
        for image_path, label_path in file_pairs:
            file.write(f"{image_path};{label_path}\n")