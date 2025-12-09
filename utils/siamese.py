import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers


# Custom L1Dist layer
class L1Dist(layers.Layer):
    def call(self, inputs):
        emb_a, emb_b = inputs
        return tf.abs(emb_a - emb_b)


class SiameseComparator:
    def __init__(self, model_path):
        self.model = load_model(
            model_path,
            custom_objects={"L1Dist": L1Dist},
            compile=False
        )
        self.encoder = self.model.layers[2]  # shared encoder

    def embed(self, img):
        img = np.expand_dims(img, axis=0)
        return self.encoder.predict(img)

    def compare(self, img1, img2):
        emb1 = self.embed(img1)
        emb2 = self.embed(img2)
        dist = np.abs(emb1 - emb2)
        return float(np.exp(-np.mean(dist)))
