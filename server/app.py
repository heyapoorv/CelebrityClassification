# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import util

# app = Flask(__name__)
# CORS(app)

# # Load model at startup
# util.load_saved_artifacts()


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

#         print("Result:", result)

#         return jsonify(result)

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500
    
# if __name__ == "__main__":
#     print("Starting Flask Server...")
#     app.run(debug=True, port=5000)


from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import util
import logging
from collections import Counter
import os
app = Flask(__name__, 
            template_folder="../server/templates",
            static_folder="../static")
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

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
# Load model at startup
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


# API endpoint
@app.route("/api/classify", methods=["POST"])
def classify():
    image_data = request.form.get("image_data")

    if not image_data:
        return jsonify({"error": "No image_data provided"}), 400

    try:
        print("Image received, length:", len(image_data))

        result = util.classify_image(image_data)

        # 🔥 Handle error from util
        if isinstance(result, dict) and "error" in result:
            logging.warning(f"No face detected / invalid input")
            return jsonify(result), 200

        # 🔥 Always take top prediction
        result = result[0]

        # 🔥 Confidence threshold
        if result["confidence"] < 60:
            result = {
                "class": "Unknown",
                "confidence": result["confidence"]
            }

        # 🔥 Analytics tracking
        prediction_counter[result["class"]] += 1

        # 🔥 Logging
        logging.info(f"Prediction: {result}")

        print("Final Result:", result)

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
# Run server
# ---------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)