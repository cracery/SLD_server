# main.py

import os
import sys
import numpy as np
import cv2
from flask import Flask, request, jsonify
from deepface import DeepFace
from deepface.basemodels import VGGFace
import joblib
import tensorflow as tf

# Краще DEEPFACE_HOME встановити на /tmp (без додаткового вкладення .deepface)
os.environ["DEEPFACE_HOME"] = "/tmp"
os.makedirs("/tmp/.deepface/weights", exist_ok=True)

app = Flask(__name__)
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "stress_svm_model.pkl")

try:
    clf, scaler = joblib.load(model_path)
    print("SVM model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    clf, scaler = None, None

print("Loading VGGFace model...")

# Завантаження моделі з локального шляху
weights_path = os.path.join(base_dir, "weights", "vgg_face_weights.h5")
vgg_model = VGGFace.loadModel(weights_path)

print("VGGFace loaded.")

@app.route("/")
def index():
    return "Server for stress detection is running."

@app.route("/predict", methods=["POST"])
def predict_stress():
    if clf is None or scaler is None:
        return jsonify({"error": "SVM model not loaded on server."}), 500
    if "image" not in request.files:
        return jsonify({"error": "No file 'image' in the request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"error": "File is empty"}), 400

    np_arr = np.frombuffer(file_bytes, np.uint8)
    color_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if color_frame is None:
        return jsonify({"error": "Invalid image or cannot read image"}), 400

    try:
        emb_list = DeepFace.represent(
            img_path=color_frame,
            model_name="VGG-Face",
            model=vgg_model,
            enforce_detection=False
        )
        if not emb_list:
            return jsonify({"error": "No face embedding found."}), 400

        embedding = np.array(emb_list[0]['embedding']).reshape(1, -1)
        embedding_scaled = scaler.transform(embedding)
        predicted_label = clf.predict(embedding_scaled)[0]
        return jsonify({"stress_level": predicted_label}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
