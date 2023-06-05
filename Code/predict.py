import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from PIL import Image
import tensorflow as tf
from PIL import ImageFilter
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mean_squared_error


def predict(job_dir="..", file_name="/UTKFace/48_0_0_20170120134009260.jpg.chip.jpg"):
    if not os.path.exists(job_dir + "/checkpoint/generator/"):
        print("Downloading model")
        os.system("gdown --fuzzy \"https://drive.google.com/drive/folders/1WbHOMngvKUR2iOntC77oSijkUlMxL6Tw?usp=sharing\"")
    gen = load_model(job_dir + "/checkpoint/generator/")

    if not os.path.exists(job_dir + file_name):
        print("Downloading the dataset")
        os.system("gdown --fuzzy \"https://drive.google.com/file/d/0BxYys69jI14kYVM3aVhKS1VhRUk/view?usp=share_link&resourcekey=0-dabpv_3J0C0cditpiAfhAw\"")
        os.system("tar -xf UTKFace.tar.gz")
        os.system("rm UTKFace.tar.gz")
        os.system("mv UTKFace ../")
    img = np.array(Image.open(job_dir + file_name).resize((200, 200)))
    source_age = int(file_name.split("/")[2].split("_")[0])
    print("Source age = " + str(source_age))
    target_age = int(input("Enter the target age: "))

    if target_age <= 5:
        target_age_group = 0
    elif target_age <= 10:
        target_age_group = 1
    elif target_age <= 15:
        target_age_group = 2
    elif target_age <= 20:
        target_age_group = 3
    elif target_age <= 25:
        target_age_group = 4
    elif target_age <= 30:
        target_age_group = 5
    elif target_age <= 40:
        target_age_group = 6
    elif target_age <= 50:
        target_age_group = 7
    elif target_age <= 60:
        target_age_group = 8
    elif target_age <= 70:
        target_age_group = 9
    elif target_age <= 80:
        target_age_group = 10
    elif target_age <= 90:
        target_age_group = 11
    elif target_age_group <= 100:
        target_age_group = 12
    else:
        target_age_group = 13

    img = img[None, :, :, :]
    target_age_group = np.array([target_age_group])[None, :]

    _, output_image = gen([img, target_age_group])
    output_image = np.rint(output_image.numpy()[0]).astype(int)
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    im = Image.fromarray(output_image).filter(ImageFilter.SMOOTH_MORE)
    im.save(job_dir + "/Result.jpg")


if __name__ == "__main__":
    predict()
