# app.py
import pytesseract
from PIL import Image
import io
import cv2
import numpy as np
import re
from atexit import register
from flask import Flask, jsonify, request
import urllib.request
import re
from insightface.app import FaceAnalysis
import pickle

# Initialize the Flask application
app = Flask(__name__)

face_app = FaceAnalysis(name='buffalo_sc')  # 'buffalo_l' is a pretrained model
face_app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for CPU, -1 for GPU

# In-memory storage
known_face_embeddings = []
known_face_names = []


def load_image_from_url(url):
    """Helper function to load image/frame from URL with error handling"""
    try:
        # Try as image first
        resp = urllib.request.urlopen(url)
        image_data = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if image is not None:
            return image
        
        # If image loading failed, try as video stream
        video_capture = cv2.VideoCapture(url)
        if not video_capture.isOpened():
            raise ValueError("Failed to open URL as either image or video stream")
            
        ret, frame = video_capture.read()
        video_capture.release()
        
        if not ret:
            raise ValueError("Failed to read frame from video stream")
            
        return frame
        
    except Exception as e:
        raise ValueError(f"Error loading media from URL: {str(e)}")
    
@app.route("/api/register", methods=['POST'])
def register_face():
    data = request.get_json()  # Changed to get_json() for better error handling
    if not data or 'name' not in data or 'url' not in data:
        return jsonify({"error": "Missing name or URL in request"}), 400

    name = data['name']
    url = data['url']

    try:
        image = load_image_from_url(url)
        # Detect faces
        faces = face_app.get(image)
        if len(faces) == 0:
            return jsonify({"error": "No faces found in the image"}), 400

        # Store the first face's embedding
        face_embedding = faces[0].normed_embedding
        known_face_embeddings.append(face_embedding)
        known_face_names.append(name)
        
        with open("registered_faces/face_encodings.pkl", "wb") as f:
            pickle.dump((known_face_embeddings, known_face_names), f)
        return jsonify({
            "message": f"Face registered successfully for {name}",
            "faces_detected": len(faces)
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"message": "Please provide a URL"}), 400

    url = data['url']

    try:
        with open("registered_faces/face_encodings.pkl", "rb") as f:
            known_face_embeddings, known_face_names = pickle.load(f)
        image = load_image_from_url(url)
        faces = face_app.get(image)

        if not faces:
            return jsonify({"message": "No faces found in the image"})

        recognized_names = []
        for face in faces:
            if not known_face_embeddings:
                recognized_names.append("Unknown")
                continue

            current_embedding = face.normed_embedding
            scores = np.dot(known_face_embeddings, current_embedding)
            best_match_idx = np.argmax(scores)
            best_score = scores[best_match_idx]

            # Adjust threshold as needed (0.5 is typical)
            recognized_names.append(
                known_face_names[best_match_idx] if best_score > 0.5 else "Unknown"
            )

        return jsonify({
            "recognized_faces": recognized_names,
            "confidence_scores": scores.tolist() if faces else []
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

def clean_text(text):
    """
    Remove escape sequences and clean up OCR output
    Returns single-line text with proper spacing
    """
    # Replace all whitespace characters (including newlines, tabs) with single space
    text = ' '.join(text.split())
    
    # Remove other non-printable characters (except basic punctuation)
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    
    # Clean up any remaining issues
    text = text.strip()  # Remove leading/trailing spaces
    text = re.sub(r' +', ' ', text)  # Collapse multiple spaces
    
    return text

def preprocess_image(image):
    """Enhance image for better OCR results"""
    # Convert to numpy array
    img = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Optional: Add denoising
    # processed = cv2.fastNlMeansDenoising(processed, h=10)
    
    return processed

@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
    """
    Endpoint that accepts an image and returns clean OCR text
    Example usage:
    curl -X POST -F "image=@document.jpg" http://localhost:5000/ocr
    """
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    
    # Verify file was selected
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read image directly into memory
        image_bytes = image_file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image
        processed_image = preprocess_image(pil_image)
        
        # Perform OCR with optimized settings
        custom_config = r'--oem 3 --psm 6'  # LSTM OCR Engine, assume uniform text block
        raw_text = pytesseract.image_to_string(processed_image, config=custom_config)
        
        # Clean the text output
        clean_result = clean_text(raw_text)
        
        if not clean_result:
            return jsonify({'error': 'No readable text found in image'}), 400
        
        return jsonify({
            'text': clean_result,
            'original_length': len(raw_text),
            'clean_length': len(clean_result)
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': 'OCR processing failed',
            'details': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)