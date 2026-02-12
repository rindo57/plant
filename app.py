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
    
    # Use Gemini 2.0 Flash Lite model (confirmed available from your list)
    MODEL_NAME = 'models/gemini-2.0-flash-lite'
    logger.info(f"Using model: {MODEL_NAME}")
    
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    sys.exit(1)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def analyze_plant_disease(image_path):
    """Analyze plant disease using Gemini API"""
    try:
        # Initialize Gemini model with confirmed working model
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Open and prepare image
        img = Image.open(image_path)
        
        # Resize image if too large (optimize for API)
        max_size = (800, 800)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Create comprehensive prompt for disease detection
        prompt = """You are an expert plant pathologist and agricultural specialist. Analyze this plant leaf image and provide a detailed report.

IMPORTANT: Format your response with clear emoji headers and bullet points. Be specific and practical for farmers.

🌿 DISEASE IDENTIFICATION
• Disease Name: [Name of the disease or "Healthy Plant" if no disease]
• Confidence Level: [High/Medium/Low]
• Plant Type: [Identify the plant if possible]

⚠️ SEVERITY ASSESSMENT
• Severity Level: [Mild/Moderate/Severe]
• Estimated Spread: [Percentage of leaf affected]
• Urgency Level: [Immediate/This Week/Monitor]

💊 TREATMENT RECOMMENDATIONS

Immediate Actions (24-48 hours):
• [List 2-3 specific actions farmer can take today]

Organic/Biological Treatments:
• [List 2-3 organic treatment options with application methods]

Chemical Treatments (if necessary):
• [List specific chemicals with proper names and safety precautions]

🌱 MANAGEMENT STRATEGIES

Cultural Practices:
• [Watering, spacing, pruning, sanitation recommendations]

Environmental Adjustments:
• [Light, humidity, air circulation, temperature recommendations]

Preventive Measures:
• [Long-term prevention strategies]

📊 ADDITIONAL INFORMATION
• Expected Recovery Time: [With proper treatment]
• Similar Diseases: [Diseases with similar symptoms to watch for]
• When to Consult Expert: [Specific conditions requiring professional help]

💡 FARMER-FRIENDLY SUMMARY
[A simple, easy-to-understand summary in 2-3 sentences in simple language]

If the plant appears healthy, clearly state "HEALTHY PLANT" and provide maintenance tips for keeping it healthy."""
        
        # Get response from Gemini
        logger.info(f"Sending request to Gemini API with model: {MODEL_NAME}")
        response = model.generate_content([prompt, img])
        
        if not response or not response.text:
            return "Unable to analyze the image. Please try again with a clearer photo of the plant leaf."
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error in analyze_plant_disease: {e}")
        logger.error(traceback.format_exc())
        return f"""❌ ANALYSIS ERROR

We encountered an error while analyzing your image:

Error: {str(e)}

Please try:
1. Take a clearer photo with good lighting
2. Ensure the leaf fills most of the frame
3. Use a plain background
4. Try a different image format (JPG or PNG recommended)

If the problem persists, please try again later."""

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
        max_size = (800, 800)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam_capture.jpg')
        img.save(temp_path, 'JPEG', quality=90)
        
        # Analyze
        analysis_result = analyze_plant_disease(temp_path)
        
        # Convert image back to base64 for display
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=90)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'analysis': analysis_result,
            'image': f'data:image/jpeg;base64,{img_base64}'
        })
    
    except Exception as e:
        logger.error(f"Error in analyze_webcam: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/test')
def test_api():
    """Test endpoint to verify Gemini API is working"""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Simple test prompt
        response = model.generate_content("Say 'Plant disease detection system is working!' if you can read this.")
        
        return jsonify({
            'status': 'success',
            'model': MODEL_NAME,
            'response': response.text,
            'api_key_present': bool(GEMINI_API_KEY)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'model': MODEL_NAME,
            'error': str(e),
            'api_key_present': bool(GEMINI_API_KEY)
        })

@app.route('/debug')
def debug():
    """Debug endpoint to check configuration"""
    return jsonify({
        'model_configured': MODEL_NAME,
        'api_key_present': bool(GEMINI_API_KEY),
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'allowed_extensions': list(app.config['ALLOWED_EXTENSIONS']),
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
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
    print(f"✅ Using model: {MODEL_NAME}")
    
    # Test API connection
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        test_response = model.generate_content("Test connection")
        print(f"✅ API connection successful!")
    except Exception as e:
        print(f"⚠️  Warning: API test failed: {e}")
        print(f"   Please check your API key and internet connection")
    
    print("\n🚀 Server running at: http://localhost:5000")
    print("📝 Test API at: http://localhost:5000/test")
    print("🔍 Debug info at: http://localhost:5000/debug")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
