import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load model
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")

print("Model loaded successfully")

# Emotion labels (common order - may vary slightly)
emotion_labels = ["Angry", "Disgust", "Fear",
                  "Happy", "Neutral", "Sad", "Surprise"]

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        roi = roi_gray / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))

        prediction = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw rectangle + text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
