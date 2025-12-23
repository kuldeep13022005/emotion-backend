import os

# 🔹 Force TensorFlow to use CPU only (avoid CUDA init noise)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image

app = Flask(__name__)

# 🔹 CORS for all routes (localhost + future deployed frontend)
CORS(app, resources={r"/*": {"origins": "*"}})
# If you want stricter later:
# CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://your-frontend-domain.com"]}})

# 🔹 Emotion labels
CLASS_NAMES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 🔹 Load face cascade once
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 🔹 Lazy model: only load when first needed
_model = None

def get_model():
    global _model
    if _model is None:
        app.logger.info("Loading emotion_detection_model.h5...")
        _model = load_model('emotion_detection_model.h5')
        app.logger.info("Model loaded.")
    return _model

def process_image(image_data: str):
    # Expect full data URL: "data:image/jpeg;base64,...."
    if ',' in image_data:
        image_data = image_data.split(',', 1)[1]

    # Decode base64 to image bytes
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Convert to OpenCV BGR
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Grayscale
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)

    results = []
    model = get_model()

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (48, 48))

        roi = face_roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # This is where TF compute happens
        prediction = model.predict(roi, verbose=0)[0]
        emotion_index = int(np.argmax(prediction))
        emotion = CLASS_NAMES[emotion_index]
        confidence = float(prediction[emotion_index] * 100.0)

        results.append({
            "emotion": emotion,
            "confidence": confidence,
            "face_location": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
            },
        })

    return results

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"success": True, "message": "Backend is active"}), 200

@app.route("/detect-emotion", methods=["POST", "OPTIONS"])
def detect_emotion():
    try:
        # For CORS preflight, Flask-CORS handles OPTIONS, but we keep method list explicit
        if request.method == "OPTIONS":
            return jsonify({"success": True}), 200

        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"success": False, "error": "No image data provided"}), 400

        app.logger.info("Received /detect-emotion request")
        results = process_image(data["image"])
        app.logger.info("Processed image, faces=%d", len(results))

        return jsonify({"success": True, "results": results}), 200

    except Exception as e:
        # Log full error server-side
        app.logger.exception("Error in /detect-emotion: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    # Local dev only
    app.run(host="0.0.0.0", port=5000, debug=True)