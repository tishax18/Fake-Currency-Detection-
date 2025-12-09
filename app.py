import os
import cv2
from flask import Flask, render_template, request
from ultralytics import YOLO
from utils.preprocessor import preprocess_roi
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

# -----------------------------
# FLASK APP SETUP
# -----------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class L1Dist(layers.Layer):
    def call(self, inputs):
        emb_a, emb_b = inputs
        return tf.abs(emb_a - emb_b)

class SiameseComparator:
    def __init__(self, model_path):
        self.model = load_model(model_path, custom_objects={"L1Dist": L1Dist}, compile=False)
        self.encoder = self.model.layers[2]  # shared encoder layer

    def embed(self, img):
        img = np.expand_dims(img, axis=0)
        return self.encoder.predict(img)

    def compare(self, img1, img2):
        emb1 = self.embed(img1)
        emb2 = self.embed(img2)
        dist = np.abs(emb1 - emb2)
        return float(np.exp(-np.mean(dist)))
# -----------------------------
# LOAD MODELS
# -----------------------------
note_model = YOLO("models/note.pt")
feature_model = YOLO("models/feature.pt")

print("Loading Siamese model...")
siamese = SiameseComparator("models/siamese_model.keras")

# -----------------------------
# REFERENCE FEATURES
# -----------------------------
REFERENCE_FEATURES = {
    "watermark_window": "models/reference_features/watermark_window.png",
    "security_thread": "models/reference_features/security_thread.png",
    "number_panel": "models/reference_features/number_panel.png"
}

# -----------------------------
# UTILITY: Crop Region
# -----------------------------
def crop_region(img, box):
    x1, y1, x2, y2 = map(int, box)
    return img[y1:y2, x1:x2]


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------
# DETECT API
# -----------------------------
@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)

    # --------------------------------------------------
    # 1. DETECT NOTE
    # --------------------------------------------------
    results_note = note_model(img)

    if len(results_note[0].boxes) == 0:
        return render_template("index.html", error="No note detected!")

    box = results_note[0].boxes[0].xyxy[0].tolist()
    note = crop_region(img, box)

    # --------------------------------------------------
    # 2. DETECT FEATURES
    # --------------------------------------------------
    results_feat = feature_model(note)
    feature_boxed_path = os.path.join(UPLOAD_FOLDER, "features_" + file.filename)
    results_feat[0].save(feature_boxed_path)

    feature_results = []

    for box in results_feat[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = feature_model.names[int(box.cls)]

        roi = crop_region(note, (x1, y1, x2, y2))
        roi_prep = preprocess_roi(roi)

        ref_path = REFERENCE_FEATURES[label]
        ref_img = preprocess_roi(cv2.imread(ref_path))

        score = siamese.compare(roi_prep, ref_img)
        status = "REAL" if score > 0.4 else "FAKE"

        feature_results.append({
            "feature": label,
            "score": round(score, 4),
            "status": status
        })
    final_note_status = "FAKE NOTE" if any(r["status"] == "FAKE" for r in feature_results) else "REAL NOTE"

    return render_template(
        "index.html",
        results=feature_results,
        final_status=final_note_status,
        uploaded=file.filename,
        boxed_image="boxed_" + file.filename,
        feature_boxed="features_" + file.filename
    )


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
