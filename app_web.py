from flask import Flask, request, jsonify
import numpy as np
import cv2
from keras.models import model_from_json

app = Flask(__name__)

model = load_model("facialemotionmodel.h5")

emotion_labels = ["Angry", "Disgust", "Fear",
                  "Happy", "Neutral", "Sad", "Surprise"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


@app.route("/")
def home():
    return "Emotion API is running"


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))

        pred = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(pred)]

        results.append(label)

    return jsonify({"emotions": results})


if __name__ == "__main__":
    app.run()
