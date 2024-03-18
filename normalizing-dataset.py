from tqdm import tqdm
from glob import glob
import os

import os
import struct

class UnknownImageFormat(Exception):
    pass

def get_image_size(file_path):
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct modules from core
    """
    size = os.path.getsize(file_path)

    with open(file_path, 'rb') as input:
        height = -1
        width = -1
        data = input.read(25)

        if (size >= 10) and data[:6] in (b'GIF87a', b'GIF89a'):
            # GIFs
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif ((size >= 24) and data.startswith(b'\211PNG\r\n\032\n')
              and (data[12:16] == b'IHDR')):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 16) and data.startswith(b'\211PNG\r\n\032\n'):
            # older PNGs?
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith(b'\377\330'):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while (b and b != b'\xDA'):
                    while (b != b'\xFF'): b = input.read(1)
                    while (b == b'\xFF'): b = input.read(1)
                    if (b >= b'\xC0' and b <= b'\xC3'):
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(int(struct.unpack(">H", input.read(2))[0])-2)
                    b = input.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
        else:
            raise UnknownImageFormat(
                "Sorry, don't know how to get information from this file."
            )

    return width, height

if __name__ == "__main__":
    FOLDER = "final"
    TARGET_FOLDERS = ["train", "test", "valid"]

    for target_folder in TARGET_FOLDERS:
        IMAGES_PATH = os.path.join(FOLDER, target_folder, "images")
        LABELS_PATH = os.path.join(FOLDER, target_folder, "labels")

        images = glob(os.path.join(IMAGES_PATH, "*.jpg"))

        print(f"Normalizing {target_folder} images")
        for image_path in tqdm(images):
            label_path = image_path.replace(".jpg", ".txt").replace("images", "labels")

            if not os.path.exists(label_path) or not os.path.exists(image_path):
                continue

            width, height = get_image_size(image_path)

            with open(label_path, "r") as f:
                lines = f.readlines()

            with open(label_path, "w") as f:
                for line in lines:
                    line = line.strip().split(" ")
                    if len(line) >= 5:  # Verifica se a linha possui pelo menos 5 elementos
                        line[1] = str(float(line[1]) / width)
                        line[2] = str(float(line[2]) / height)
                        line[3] = str(float(line[3]) / width)
                        line[4] = str(float(line[4]) / height)
                        f.write(" ".join(line))
                        f.write("\n")
                    else:
                        print(f"A linha não tem informações suficientes: {line}")

            print(f"Normalized {image_path}")