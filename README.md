# Face Emotion Recognition (Real-Time)

This project is a real-time facial emotion recognition system that uses a webcam to detect and classify human emotions.

The system captures live video, detects faces, and predicts emotions such as Happy, Sad, Angry, etc., using a trained deep learning model.

---

## Features

* Real-time emotion detection using webcam
* Pre-trained deep learning model (.h5 + .json)
* Face detection using OpenCV
* Fast and responsive predictions

---

## Tech Stack

* Python
* OpenCV
* NumPy
* TensorFlow / Keras

---

## Project Structure

```
FaceEmotion-api/
│── realtime_emotion.py      # Main script (Run this)
│── facialemotionmodel.h5    # Model weights
│── facialemotionmodel.json  # Model architecture
│── requirements.txt
│── README.md
```

---

## How to Run (Local Setup)

### 1. Clone the repository

```
git clone https://github.com/sivap365/FaceEmotion-api.git
cd FaceEmotion-api
```

---

### 2. Create virtual environment (recommended)

```
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

### 4. Run the application

```
python realtime_emotion.py
```

---

## Output

* Opens webcam
* Detects face in real-time
* Displays predicted emotion on screen

---

## Future Improvements

* Web-based frontend with live camera support
* API integration for deployment
* Improved model accuracy
