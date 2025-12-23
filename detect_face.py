import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained emotion detection model
model = load_model('emotion_detection_model_73.h5')

# Define emotion classes
class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Process each face
    for (x, y, w, h) in faces:
        # Increase frame size by 25%
        increase = 0.25
        new_x = int(x - (w * increase/2))
        new_y = int(y - (h * increase/2))
        new_w = int(w * (1 + increase))
        new_h = int(h * (1 + increase))
        
        # Ensure coordinates don't go outside frame bounds
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(frame.shape[1] - new_x, new_w)
        new_h = min(frame.shape[0] - new_y, new_h)
        
        # Extract face ROI with larger area
        face_roi = gray[new_y:new_y+new_h, new_x:new_x+new_w]
        
        # Resize to 48x48
        face_roi = cv2.resize(face_roi, (48, 48))
        
        # Prepare for model
        roi = face_roi.astype('float')/255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        
        # Make prediction
        prediction = model.predict(roi)[0]
        emotion_index = np.argmax(prediction)
        emotion = class_names[emotion_index]
        confidence = prediction[emotion_index] * 100
        
        # Draw rectangle around face with new coordinates
        cv2.rectangle(frame, (new_x, new_y), (new_x+new_w, new_y+new_h), (255, 0, 0), 2)
        
        # Adjust text position for new coordinates
        text = f'{emotion}: {confidence:.2f}%'
        cv2.putText(frame, text, (new_x, new_y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                    (0, 0, 255), 2)  # Changed color to bright red (BGR format)
    
    # Display the frame
    cv2.imshow('Emotion Detection', frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()