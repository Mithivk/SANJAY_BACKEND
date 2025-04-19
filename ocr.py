from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import io
import cv2
import numpy as np
import re

app = Flask(__name__)

# Configure Tesseract path (Windows users may need this)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

@app.route('/ocr', methods=['POST'])
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
    app.run(host='0.0.0.0', port=5000)