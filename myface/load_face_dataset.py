import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

images = []
labels = []


def ReadPath(path_name):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if os.path.isdir(full_path):
            ReadPath(full_path)
        else:
            if dir_item.endswith(".jpg"):
                image = cv2.imread(full_path)
                h, w, _ = image.shape
                top, bottom, left, right = (0, 0, 0, 0)
                constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                image = cv2.resize(constant, (64, 64))
                images.append(image)
                labels.append(path_name)
    return images, labels


def LoadPictures(path_name):
    images, labels = ReadPath(path_name)
    images = np.array(images)
    for i, label in enumerate(labels):
        if label.endswith("me"):
            labels[i] = 1
        elif label.endswith("you"):
            labels[i] = 2
        elif label.endswith("he"):
            labels[i] = 3
        else:
            labels[i] = 0
    return images, labels


if __name__ == '__main__':
    images, labels = ReadPath("pic")

    train_images, train_labels, test_images, test_labels = train_test_split(images, labels, test_size=0.3, random_state=1, stratify=labels)
