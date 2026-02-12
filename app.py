import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import required modules
try:
    import google.generativeai as genai
    logger.info("Successfully imported google.generativeai")
except ImportError as e:
    logger.error(f"Failed to import google.generativeai: {e}")
    logger.error("Please install it using: pip install google-generativeai")
    sys.exit(1)

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from dotenv import load_dotenv
import io
import base64
import traceback

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables")
    logger.error("Please add your API key to the .env file")
    sys.exit(1)

try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Successfully configured Gemini API")
    
    # List available models for debugging
    models = genai.list_models()
    available_models = []
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            available_models.append(model.name)
            logger.info(f"Available model: {model.name}")
    
    logger.info(f"Available models for content generation: {available_models}")
    
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    sys.exit(1)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_available_model():
    """Get the first available Gemini model for content generation"""
    try:
        models = genai.list_models()
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                # Prefer gemini-pro-vision for image analysis
                if 'vision' in model.name:
                    return model.name
        # Fallback to first available model that supports generateContent
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                return model.name
    except Exception as e:
        logger.error(f"Error listing models: {e}")
    
    # Hardcoded fallback models that typically work
    fallback_models = [
        'models/gemini-pro-vision',
        'models/gemini-1.0-pro-vision',
        'models/gemini-1.5-pro-vision',
        'models/gemini-pro',
    ]
    
    return fallback_models[0]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def analyze_plant_disease(image_path):
    """Analyze plant disease using Gemini API"""
    try:
        # Get available model
        model_name = get_available_model()
        logger.info(f"Using model: {model_name}")
        
        # Initialize Gemini model
        model = genai.GenerativeModel(model_name)
        
        # Open and prepare image
        img = Image.open(image_path)
        
        # Resize image if too large (Gemini has limits)
        max_size = (1024, 1024)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Create comprehensive prompt for disease detection
        prompt = """You are an expert plant pathologist and agricultural specialist. Analyze this plant leaf image and provide a detailed report.

IMPORTANT: Format your response with clear emoji headers and bullet points.

🌿 DISEASE IDENTIFICATION
• Disease Name: [Name or "Healthy Plant"]
• Confidence: [High/Medium/Low]
• Plant Type: [If identifiable]

⚠️ SEVERITY
• Level: [Mild/Moderate/Severe]
• Spread: [Estimated percentage]
• Urgency: [Immediate/Soon/Monitor]

💊 TREATMENT

Immediate Actions:
• [Action 1]
• [Action 2]
• [Action 3]

Organic Options:
• [Option 1]
• [Option 2]

Chemical Options (if needed):
• [Option 1 with precautions]

🌱 PREVENTION
• Cultural: [Practice 1], [Practice 2]
• Environmental: [Adjustment 1], [Adjustment 2]
• Long-term: [Strategy 1], [Strategy 2]

📋 SUMMARY
[A simple 2-sentence summary for farmers]

If no disease is detected, clearly state the plant appears healthy and provide maintenance tips."""
        
        # Get response from Gemini
        logger.info(f"Sending request to Gemini API...")
        response = model.generate_content([prompt, img])
        
        if not response or not response.text:
            return "Unable to analyze the image. Please try again with a clearer photo."
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error in analyze_plant_disease: {e}")
        logger.error(traceback.format_exc())
        return f"""❌ ANALYSIS ERROR

We encountered an error: {str(e)}

Please try:
1. Using a clearer image of the plant leaf
2. Ensuring the leaf is well-lit and in focus
3. Trying a different image format (JPG or PNG)
4. If problem persists, try again later

Error details: {type(e).__name__}"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image file (JPG, PNG, GIF, WebP).'}), 400
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"File saved: {filepath}")
        
        # Analyze the image
        analysis_result = analyze_plant_disease(filepath)
        
        # Read image for display
        with open(filepath, 'rb') as f:
            img_data = f.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        return render_template('result.html', 
                             filename=filename,
                             image=img_base64,
                             analysis=analysis_result)
    
    except Exception as e:
        logger.error(f"Error in upload_file: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_webcam():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = data['image'].split(',')[1]
        
        # Convert base64 to image
        img_data = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_data))
        
        # Resize image
        max_size = (1024, 1024)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam_capture.jpg')
        img.save(temp_path, 'JPEG', quality=85)
        
        # Analyze
        analysis_result = analyze_plant_disease(temp_path)
        
        # Convert image back to base64 for display
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'analysis': analysis_result,
            'image': f'data:image/jpeg;base64,{img_base64}'
        })
    
    except Exception as e:
        logger.error(f"Error in analyze_webcam: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/debug')
def debug():
    """Debug endpoint to check available models"""
    try:
        models = genai.list_models()
        model_info = []
        for model in models:
            model_info.append({
                'name': model.name,
                'methods': list(model.supported_generation_methods)
            })
        return jsonify({
            'api_configured': True,
            'available_models': model_info,
            'api_key_present': bool(GEMINI_API_KEY)
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'api_configured': False,
            'api_key_present': bool(GEMINI_API_KEY)
        })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File is too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error. Please try again later.'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌿 Plant Disease Detection System")
    print("="*60)
    print(f"✅ Flask application starting...")
    print(f"✅ Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"✅ Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024)}MB")
    print(f"✅ Allowed formats: {', '.join(app.config['ALLOWED_EXTENSIONS'])}")
    print(f"✅ Gemini API Key: {'✓ Present' if GEMINI_API_KEY else '✗ Missing'}")
    
    # Test API connection
    try:
        model_name = get_available_model()
        print(f"✅ Using Gemini model: {model_name}")
    except Exception as e:
        print(f"⚠️  Warning: Could not get available models: {e}")
    
    print("\n🚀 Server running at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
