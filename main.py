import os
import sys
import shutil
import numpy as np
import cv2
from flask import Flask, request, jsonify
from deepface import DeepFace
from deepface.basemodels import VGGFace
import joblib
import tensorflow as tf
import traceback

# Встановлюємо стабільну кеш-директорію
os.environ["DEEPFACE_HOME"] = "/tmp/.deepface"
os.makedirs(os.path.join(os.environ["DEEPFACE_HOME"], "weights"), exist_ok=True)

app = Flask(__name__)
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "stress_svm_model.pkl")

# Завантаження SVM-моделі
try:
    clf, scaler = joblib.load(model_path)
    print("✅ SVM model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    clf, scaler = None, None

# Копіюємо ваги у DeepFace кеш, якщо потрібно
local_weights_path = os.path.join(base_dir, "weights", "vgg_face_weights.h5")
deepface_weights_path = os.path.join(os.environ["DEEPFACE_HOME"], "weights", "vgg_face_weights.h5")
if not os.path.exists(deepface_weights_path):
    print("📦 Copying VGGFace weights to DeepFace cache...")
    shutil.copy(local_weights_path, deepface_weights_path)

# Завантаження VGGFace моделі
print("📥 Loading VGGFace model...")
vgg_model = VGGFace.loadModel()
print("✅ VGGFace model loaded.")

@app.route("/")
def index():
    return "✅ Server for stress detection is running."

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
        return jsonify({"stress_level": int(predicted_label)}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Exception: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
