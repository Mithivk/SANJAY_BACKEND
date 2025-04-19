import os
import time
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import pytesseract
from PIL import Image
import io
import base64
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)
CORS(app)

# Initialize Roboflow API client for currency detection
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="W9sCQe7BAeAN2vJ9pC7q"
)

# Folder to store registered face images
FACE_DB_FOLDER = "face_db"
os.makedirs(FACE_DB_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")

# -------------------------- FACE REGISTRATION -------------------------- #
@app.route("/api/register", methods=['POST'])
def register_face():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    name = request.form.get('name')
    if not name:
        return jsonify({"error": "Name is required"}), 400

    image_path = os.path.join(FACE_DB_FOLDER, f"{name}.jpg")
    file.save(image_path)
    return jsonify({"message": f"Face of {name} registered successfully."}), 200

# -------------------------- FACE RECOGNITION -------------------------- #
def match_face(unknown_img):
    orb = cv2.ORB_create()
    unknown_gray = cv2.cvtColor(unknown_img, cv2.COLOR_BGR2GRAY)
    keypoints_unknown, descriptors_unknown = orb.detectAndCompute(unknown_gray, None)
    
    best_match = None
    best_score = 0
    for filename in os.listdir(FACE_DB_FOLDER):
        registered_img = cv2.imread(os.path.join(FACE_DB_FOLDER, filename))
        if registered_img is None:
            continue
        registered_gray = cv2.cvtColor(registered_img, cv2.COLOR_BGR2GRAY)
        keypoints_registered, descriptors_registered = orb.detectAndCompute(registered_gray, None)
        
        if descriptors_unknown is None or descriptors_registered is None:
            continue
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_unknown, descriptors_registered)
        score = len(matches)

        if score > best_score:
            best_score = score
            best_match = filename.split(".")[0]

    return best_match if best_score > 20 else None

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image_path = f"uploaded_{int(time.time())}.jpg"
    file.save(image_path)
    
    img = cv2.imread(image_path)
    recognized_name = match_face(img)
    
    os.remove(image_path)
    return jsonify({"message": f"Recognized: {recognized_name}" if recognized_name else "Face not recognized"}), 200

# -------------------------- OBJECT DETECTION -------------------------- #
@app.route('/api/object-detection', methods=['POST'])
def object_detection():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image_path = f"uploaded_{int(time.time())}.jpg"
    file.save(image_path)

    results = model(image_path)
    detected_objects = [model.names[int(box.cls[0])] for result in results for box in result.boxes]

    os.remove(image_path)
    return jsonify({"message": "Objects detected: " + ", ".join(detected_objects) if detected_objects else "No objects detected"}), 200

# -------------------------- OCR PROCESSING -------------------------- #
def preprocess_image(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    pil_image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(pil_image)

    raw_text = pytesseract.image_to_string(processed_image, config='--oem 3 --psm 6')
    clean_text = ' '.join(raw_text.split())

    return jsonify({"message": clean_text if clean_text else "No readable text found"}), 200

# -------------------------- CURRENCY DETECTION (ROBOFLOW) -------------------------- #
@app.route('/api/currency-detection', methods=['POST'])
def currency_detection():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image_bytes = file.read()
    
    try:
        # Convert image to Base64
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        response = CLIENT.infer(encoded_image, model_id="fy23/1")

        currency_count = {}
        confidence_threshold = 0.5
        for detection in response.get("predictions", []):
            if detection['confidence'] >= confidence_threshold:
                label = detection['class']
                currency_count[label] = currency_count.get(label, 0) + 1

        detected_notes = [f"{count} {note} rupee note(s)" for note, count in currency_count.items()]
        return jsonify({"message": "Currency detected: " + ", ".join(detected_notes) if detected_notes else "No currency detected"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------- RUN FLASK SERVER -------------------------- #
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
