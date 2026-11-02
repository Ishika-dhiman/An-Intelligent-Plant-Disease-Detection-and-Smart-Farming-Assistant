import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
LABELS_PATH = MODEL_DIR / "class_names.json"
DB_PATH = BASE_DIR / "database" / "history.db"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMAGE_SIZE = (160,160)
MAX_FILE_SIZE = 8 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_model = None
_class_names = None

# General educational guidance only; not a substitute for local agricultural advice.
ADVICE = {
    "healthy": {
        "title": "Plant appears healthy",
        "tips": [
            "Continue regular monitoring of leaves and stems.",
            "Water near the soil rather than repeatedly wetting leaves.",
            "Use balanced nutrients based on your soil and crop needs."
        ]
    },
    "early blight": {
        "title": "Possible Early Blight",
        "tips": [
            "Remove badly affected leaves and dispose of them away from the crop.",
            "Avoid overhead watering and improve airflow between plants.",
            "Rotate crops and avoid planting tomatoes or potatoes in the same spot every season.",
            "Ask a local agricultural expert before using any crop-protection product."
        ]
    },
    "late blight": {
        "title": "Possible Late Blight",
        "tips": [
            "Isolate and remove severely affected plant material where appropriate.",
            "Avoid working among wet plants; reduce leaf wetness and improve airflow.",
            "Seek prompt local agricultural guidance because late blight can spread quickly.",
            "Use only locally approved treatments and follow the product label."
        ]
    },
    "leaf mold": {
        "title": "Possible Leaf Mold",
        "tips": [
            "Improve ventilation and reduce humidity around plants.",
            "Avoid wetting foliage during irrigation.",
            "Remove heavily affected leaves and clean tools between plants."
        ]
    },
    "bacterial spot": {
        "title": "Possible Bacterial Spot",
        "tips": [
            "Avoid handling plants when foliage is wet.",
            "Remove badly affected plant debris and keep tools clean.",
            "Use disease-free seeds/transplants and rotate crops where practical."
        ]
    },
    "powdery mildew": {
        "title": "Possible Powdery Mildew",
        "tips": [
            "Improve airflow and avoid overcrowding.",
            "Remove heavily affected leaves when appropriate.",
            "Confirm the diagnosis locally before selecting a treatment."
        ]
    }
}

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

def load_model_once():
    global _model, _class_names
    if _model is not None:
        return _model, _class_names
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, None
    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
        _class_names = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        return _model, _class_names
    except Exception as exc:
        app.logger.exception("Could not load the trained model: %s", exc)
        return None, None

def get_advice(label):
    low = label.lower().replace("_", " ")
    for key, value in ADVICE.items():
        if key in low:
            return value
    return {
        "title": f"Detected class: {label}",
        "tips": [
            "Compare this result with a trusted crop-disease guide.",
            "Check several leaves and observe whether symptoms are spreading.",
            "Consult a local agriculture extension officer before treatment."
        ]
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/history")
def history():
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT disease, confidence, created_at FROM predictions ORDER BY id DESC LIMIT 8"
        ).fetchall()
    return jsonify([dict(row) for row in rows])

@app.route("/api/predict", methods=["POST"])
def predict():
    if "leaf_image" not in request.files:
        return jsonify(error="Please choose a leaf image first."), 400

    file = request.files["leaf_image"]
    if not file or not file.filename:
        return jsonify(error="Please choose an image file."), 400

    safe_name = secure_filename(file.filename)
    if "." not in safe_name or safe_name.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS:
        return jsonify(error="Use a JPG, JPEG, PNG, or WEBP image."), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        image.thumbnail((1600, 1600))
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{safe_name}"
        saved_path = UPLOAD_DIR / filename
        image.save(saved_path, format="JPEG", quality=90)
        image = image.resize(IMAGE_SIZE)
        image_array = np.asarray(image, dtype=np.float32)

        model, class_names = load_model_once()
        if model is None:
            return jsonify(
                error="The AI model has not been trained yet. Follow README.md to add the PlantVillage dataset and run train_model.py.",
                model_ready=False
            ), 503

        prediction = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
        index = int(np.argmax(prediction))
        label = str(class_names[index])
        confidence = float(prediction[index]) * 100.0
        advice = get_advice(label)

        with sqlite3.connect(DB_PATH) as con:
            con.execute(
                "INSERT INTO predictions (filename, disease, confidence, created_at) VALUES (?, ?, ?, ?)",
                (filename, label, confidence, datetime.now().strftime("%d %b %Y, %I:%M %p"))
            )

        return jsonify(
            disease=label.replace("_", " "),
            confidence=round(confidence, 1),
            advice=advice,
            image_url=f"/static/uploads/{filename}",
            model_ready=True
        )
    except UnidentifiedImageError:
        return jsonify(error="This file could not be read as an image."), 400
    except Exception as exc:
        app.logger.exception("Prediction failed: %s", exc)
        return jsonify(error="Something went wrong while analysing the image."), 500

@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="Image is too large. Please upload an image under 8 MB."), 413

init_db()

if __name__ == "__main__":
    app.run(debug=True)
