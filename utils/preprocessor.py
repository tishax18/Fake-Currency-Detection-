import cv2
import numpy as np


def preprocess_roi(img):
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    return img
