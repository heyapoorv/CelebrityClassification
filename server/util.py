import json
import joblib
import cv2
import numpy as np
import base64
from pathlib import Path
from wavelet import w2d

# =====================
# Globals
# =====================
MODEL = None
CLASS_NAME_TO_NUMBER = {}
CLASS_NUMBER_TO_NAME = {}

SERVER_DIR = Path(__file__).resolve().parent
ROOT_DIR = SERVER_DIR.parent

MODEL_PATH = ROOT_DIR / "best_clf_model.pkl"
CLASS_DICT_PATH = ROOT_DIR / "class_dictionary.json"
OPENCV_DIR = ROOT_DIR / "opencv"

# ---------------------
# Load artifacts
# ---------------------
def load_saved_artifacts():
    global MODEL, CLASS_NAME_TO_NUMBER, CLASS_NUMBER_TO_NAME
    if MODEL is not None:
        return

    print("Loading saved artifacts...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not CLASS_DICT_PATH.exists():
        raise FileNotFoundError(f"Class dictionary not found: {CLASS_DICT_PATH}")

    with open(CLASS_DICT_PATH, "r") as f:
        CLASS_NAME_TO_NUMBER = {k: int(v) for k, v in json.load(f).items()}
        CLASS_NUMBER_TO_NAME = {v: k for k, v in CLASS_NAME_TO_NUMBER.items()}

    MODEL = joblib.load(MODEL_PATH)
    print("Artifacts loaded successfully.")

# ---------------------
# API callable
# ---------------------
def classify_image(image_base64_data):
    img = _decode_base64_image(image_base64_data)

    if img is None:
        return {"error": "Invalid image data"}

    return classify_image_from_array(img)

# ---------------------
# Core inference
# ---------------------
def classify_image_from_array(img):
    faces = _get_faces(img)
    if len(faces) == 0:
        return [{
        "class": "No face detected",
        "confidence": 0,
        "probabilities": [],
        "class_dictionary": CLASS_NAME_TO_NUMBER
    }]

    results = []
    for face in faces:
        try:
            features = _prepare_image(face)
            if features is None or features.shape[0] == 0:
                continue
            pred = MODEL.predict(features)[0]
            probs = MODEL.predict_proba(features)[0]
        
            class_name = CLASS_NUMBER_TO_NAME.get(int(pred), "Unknown")
            results.append({
            "class": class_name,
            "confidence": float(round(np.max(probs) * 100, 2)),
            "probabilities": [float(p) for p in np.round(probs * 100, 2)],
           "class_dictionary": CLASS_NAME_TO_NUMBER
        })
        except Exception as e:
            print("Prediction error:", e)
        continue

    return results

# ---------------------
# Helpers
# ---------------------
def _prepare_image(img):
    raw = cv2.resize(img, (32, 32))
    wave = w2d(img, "db1", 5)
    wave = cv2.resize(wave, (32, 32))
    combined = np.vstack((raw.reshape(32*32*3,1), wave.reshape(32*32,1)))
    return combined.reshape(1, -1).astype(float)

# def _get_faces(img):
#     face_cascade = cv2.CascadeClassifier(str(OPENCV_DIR / "haarcascade_frontalface_default.xml"))
#     eye_cascade = cv2.CascadeClassifier(str(OPENCV_DIR / "haarcascade_eye.xml"))
#     if face_cascade.empty():
#         raise Exception("Face cascade not loaded properly")

#     if eye_cascade.empty():
#         raise Exception("Eye cascade not loaded properly")

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
#     cropped = []
#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         # if len(eyes) >= 1:
#         cropped.append(roi_color)
#     return cropped



from mtcnn import MTCNN
import cv2

detector = MTCNN()

# Fallback Haar Cascade
face_cascade = cv2.CascadeClassifier(str(OPENCV_DIR / "haarcascade_frontalface_default.xml"))

def _get_faces(img):
    results = detector.detect_faces(img)
    cropped = []

    h_img, w_img, _ = img.shape

    # -------------------
    # 🔥 MTCNN detection
    # -------------------
    for res in results:
        x, y, w, h = res['box']

        # Fix bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)

        # Skip tiny faces
        if w < 40 or h < 40:
            continue

        # 🔥 Add padding (IMPROVES ACCURACY)
        pad = int(0.15 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)

        face = img[y1:y2, x1:x2]

        if face.size == 0:
            continue

        cropped.append(face)

    # -------------------
    # 🔥 FALLBACK (CRITICAL)
    # -------------------
    if len(cropped) == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_cv = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces_cv:
            if w < 40 or h < 40:
                continue

            pad = int(0.15 * w)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)

            face = img[y1:y2, x1:x2]

            if face.size == 0:
                continue

            cropped.append(face)

    return cropped

def _decode_base64_image(b64_string):
    try:
        if "," in b64_string:
            encoded = b64_string.split(",")[1]
        else:
            encoded = b64_string  # fallback if already clean

        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Image decoding failed")

        return img

    except Exception as e:
        print("Decode error:", e)
        raise
