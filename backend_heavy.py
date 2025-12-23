from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image

app = Flask(__name__)
# 👇 IMPORTANT: enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})
# or more strict:
# CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://your-frontend.vercel.app"]}})
# CORS(app)

# Load the model and class names
model = load_model('emotion_detection_model.h5')
class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def process_image(image_data):
    # Convert base64 to image
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to opencv format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    results = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))

        roi = face_roi.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi)[0]
        emotion_index = np.argmax(prediction)
        emotion = class_names[emotion_index]
        confidence = float(prediction[emotion_index] * 100)

        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'face_location': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            }
        })

    return results

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        results = process_image(data['image'])

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'success': True, 'message': 'Backend is active'}), 200

if __name__ == '__main__':
    # Local development
    app.run(host='0.0.0.0', port=5000, debug=True)