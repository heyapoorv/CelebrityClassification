# 🎭 Celebrity Image Classification Web App

A full-stack AI-powered web application that detects faces in an image and classifies them into known celebrity categories using Machine Learning.

🔗 Live Demo: https://celebrityclassification.onrender.com

---

## 🚀 Features

- 📸 Upload image for instant prediction
- 🧠 Face detection using MTCNN
- 🧩 Feature extraction using Wavelet Transform
- 🤖 Celebrity classification using trained ML model
- 🌐 Deployed on cloud (Render)
- ⚡ Fast and interactive UI with real-time feedback
- 🔍 Handles multiple faces in an image

---

## 🏗️ Tech Stack

### 🔹 Frontend
- HTML
- CSS
- JavaScript

### 🔹 Backend
- Flask (Python)

### 🔹 Machine Learning
- OpenCV
- MTCNN (Face Detection)
- Wavelet Transform
- Scikit-learn (Model Training)

### 🔹 Deployment
- Render (Cloud Hosting)
- Gunicorn (Production Server)

---

## 📁 Project Structure
CelebrityClassification/
│
├── server/
│ ├── app.py
│ ├── util.py
│ ├── wavelet.py
│ ├── templates/
│ │ └── index.html
│ ├── static/
│ │ ├── script.js
│ │ ├── style.css
│
├── model/
│ ├── saved_model.pkl
│ ├── class_dictionary.json
│
├── requirements.txt
└── README.md

---

## ⚙️ How It Works

1. User uploads an image via UI
2. Image is sent to Flask backend
3. MTCNN detects faces
4. Faces are cropped and preprocessed
5. Wavelet transform extracts features
6. Features are passed to trained ML model
7. Model predicts celebrity class + confidence
8. Results are displayed on UI

---

## 🧠 Model Details

- Input: Cropped face image (64x64)
- Feature Engineering:
  - Raw pixels
  - Wavelet transformed image
- Model Used:
  - Support Vector Machine (SVM) / Logistic Regression (based on training)
- Output:
  - Celebrity name
  - Confidence score

---

## ⚡ Setup Instructions (Local)

### 1. Clone repo
git clone https://github.com/heyapoorv/CelebrityClassification.git
cd CelebrityClassification

###2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

###3. Install dependencies
pip install -r requirements.txt

###4. Run app
python server/app.py

###5. Open in browser
http://127.0.0.1:5000

🚀 Future Scope

This project can be significantly improved with modern AI and system design enhancements:

🧠 Deep Learning Integration
Replace traditional ML model with FaceNet / DeepFace
Use CNN-based architectures for better feature extraction
Target accuracy improvement to 90%+

👁️ Improved Face Detection
Replace MTCNN with RetinaFace
Handle:
Partial faces (one-eye visibility)
Side profiles
Occlusions

🎥 Real-time Detection
Integrate webcam-based face recognition
Enable real-time predictions with bounding boxes

📊 Enhanced Predictions
Show Top-K predictions (Top 3–5)
Display confidence distribution for better interpretability

🎨 UI/UX Improvements
Modern UI (Glassmorphism / Minimal UI)
Smooth animations and transitions
Better loading indicators and feedback
⚡ Backend Optimization
Migrate from Flask to FastAPI (async support)
Add caching (Redis) for faster inference
Optimize model loading and response time

📱 Mobile & Accessibility
Convert into Progressive Web App (PWA)
Improve responsiveness for mobile users

🔮 Planned Improvements (Next Steps)

In upcoming versions, I plan to:

✅ Upgrade model to deep learning-based embeddings (FaceNet)
✅ Improve accuracy for challenging inputs (low-light, partial faces)
✅ Add webcam-based real-time recognition
✅ Enhance UI for a more premium user experience
✅ Optimize backend for faster and scalable inference

💡 Final Note
This project demonstrates a complete end-to-end machine learning pipeline — from data processing and model building to deployment and user interaction.
It serves as a strong foundation for building scalable, real-world AI applications, and will continue to evolve with more advanced features and improvements.
