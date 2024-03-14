from functools import lru_cache
from random import choice
from glob import glob
import cv2
import os

def read_image_and_label(image_path, label_path):
    image = cv2.imread(image_path) # BGR

    labels = []
    with open(label_path, "r") as file:
        for row in file.readlines():
            row = row.strip().split(sep=" ")
            c = int(row[0])
            x, y, w, h = [float(i) for i in row[1:]]

            labels.append([c, x, y, w, h])
    return image, labels

@lru_cache(maxsize=None)
def random_color(class_id):
    return (choice(range(256)), choice(range(256)), choice(range(256)))

if __name__ == "__main__":
    FOLDER_PATH = os.path.join("kaggle", "train")
    IMAGES_PATH = os.path.join(FOLDER_PATH, "images")
    LABELS_PATH = os.path.join(FOLDER_PATH, "labels")

    images = glob(os.path.join(IMAGES_PATH, "*.jpg"))

    for image_path in images:
        label_path = image_path.replace(".jpg", ".txt").replace("images", "labels")

        image, label = read_image_and_label(image_path, label_path)

        img_h, img_w, _ = image.shape

        for c, xc, yc, w, h in label:
            color = random_color(c)
            pt1 = (int((xc - w/2) * img_w), int((yc - h/2) * img_h))
            pt2 = (int((xc + w/2) * img_w), int((yc + h/2) * img_h))

            cv2.rectangle(image, pt1, pt2, color, 2)

        cv2.imshow("image", image)
        if cv2.waitKey(0) == ord("q"):
            break
    cv2.destroyAllWindows()