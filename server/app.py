

# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# from server import util
# import logging
# from collections import Counter
# import os
# app = Flask(__name__)
# CORS(app)
# app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# # ---------------------
# # Logging setup
# # ---------------------
# logging.basicConfig(
#     filename="app.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# # ---------------------
# # Load model at startup
# # ---------------------
# util.load_saved_artifacts()

# # ---------------------
# # Analytics storage
# # ---------------------
# prediction_counter = Counter()

# # ---------------------
# # Routes
# # ---------------------

# # Home page
# @app.route("/")
# def home():
#     return render_template("index.html")


# # API endpoint
# @app.route("/api/classify", methods=["POST"])
# def classify():
#     image_data = request.form.get("image_data")

#     if not image_data:
#         return jsonify({"error": "No image_data provided"}), 400

#     try:
#         print("Image received, length:", len(image_data))

#         result = util.classify_image(image_data)

#         # 🔥 Handle error from util
#         if isinstance(result, dict) and "error" in result:
#             logging.warning(f"No face detected / invalid input")
#             return jsonify(result), 200

#         # 🔥 Always take top prediction
#         result = result[0]

#         # 🔥 Confidence threshold
#         if result["confidence"] < 60:
#             result = {
#                 "class": "Unknown",
#                 "confidence": result["confidence"]
#             }

#         # 🔥 Analytics tracking
#         prediction_counter[result["class"]] += 1

#         # 🔥 Logging
#         logging.info(f"Prediction: {result}")

#         print("Final Result:", result)

#         return jsonify(result)

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         logging.error(str(e))
#         return jsonify({"error": str(e)}), 500


# # ---------------------
# # Analytics endpoint
# # ---------------------
# @app.route("/analytics")
# def analytics():
#     return jsonify(dict(prediction_counter))


# # ---------------------
# # Run server
# # ---------------------

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     app.run(host="0.0.0.0", port=port)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from server import util
import logging
from collections import Counter
import os
import base64
import cv2
import numpy as np
import time

# ---------------------
# App Setup
# ---------------------
app = Flask(__name__)
CORS(app)

# Limit upload size (5MB)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# Reduce TensorFlow logs (for MTCNN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ---------------------
# Logging setup
# ---------------------
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------
# Load model at startup (IMPORTANT)
# ---------------------
util.load_saved_artifacts()

# ---------------------
# Analytics storage
# ---------------------
prediction_counter = Counter()

# ---------------------
# Routes
# ---------------------

# Home page
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------
# API endpoint
# ---------------------
@app.route("/api/classify", methods=["POST"])
def classify():
    image_data = request.form.get("image_data")

    if not image_data:
        return jsonify({"error": "No image_data provided"}), 400

    try:
        start_time = time.time()

        # ---------------------
        # Decode Base64 Image
        # ---------------------
        encoded_data = image_data.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        # ---------------------
        # Resize Large Images (CRITICAL FIX)
        # ---------------------
        if img.shape[0] > 600:
            scale = 600 / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * scale), 600))

        # ---------------------
        # Model Prediction
        # ---------------------
        result = util.classify_image(img)  # now expects image, not base64

        # ---------------------
        # Handle No Face / Error
        # ---------------------
        if isinstance(result, dict) and "error" in result:
            logging.warning("No face detected / invalid input")
            return jsonify(result), 200

        if not result:
            return jsonify({"error": "No prediction"}), 200

        # ---------------------
        # Take Top Prediction
        # ---------------------
        result = result[0]

        # ---------------------
        # Confidence Threshold
        # ---------------------
        if result["confidence"] < 60:
            result = {
                "class": "Unknown",
                "confidence": result["confidence"]
            }

        # ---------------------
        # Analytics Tracking
        # ---------------------
        prediction_counter[result["class"]] += 1

        # ---------------------
        # Logging
        # ---------------------
        processing_time = time.time() - start_time
        logging.info(f"Prediction: {result} | Time: {processing_time:.2f}s")

        print("Final Result:", result)
        print("Processing time:", processing_time)

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(str(e))
        return jsonify({"error": str(e)}), 500


# ---------------------
# Analytics endpoint
# ---------------------
@app.route("/analytics")
def analytics():
    return jsonify(dict(prediction_counter))


# ---------------------
# Run server (for local only)
# ---------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)