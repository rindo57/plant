import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from dotenv import load_dotenv
import io
import base64

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def analyze_plant_disease(image_path):
    """Analyze plant disease using Gemini API"""
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Open and prepare image
        img = Image.open(image_path)
        
        # Create prompt for disease detection
        prompt = """
        You are an expert plant pathologist. Analyze this plant leaf image and provide:
        
        1. DISEASE IDENTIFICATION:
           - Name of the disease (if any)
           - Confidence level
           - Affected plant type (if identifiable)
        
        2. SEVERITY ASSESSMENT:
           - Mild/Moderate/Severe
           - Estimated spread percentage
        
        3. TREATMENT RECOMMENDATIONS:
           - Immediate actions
           - Organic treatments
           - Chemical treatments (with precautions)
           - Preventive measures
        
        4. MANAGEMENT STRATEGIES:
           - Cultural practices
           - Environmental adjustments
           - Long-term prevention
        
        5. ADDITIONAL INFORMATION:
           - Similar diseases to watch for
           - Expected recovery time
           - When to seek expert help
        
        Format the response in clear sections with bullet points. If no disease is detected, state that the plant appears healthy.
        """
        
        # Get response from Gemini
        response = model.generate_content([prompt, img])
        
        return response.text
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
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
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze', methods=['POST'])
def analyze_webcam():
    """Handle webcam captured images"""
    data = request.json
    image_data = data['image'].split(',')[1]
    
    # Convert base64 to image
    img_data = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_data))
    
    # Save temporarily
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam_capture.jpg')
    img.save(temp_path)
    
    # Analyze
    analysis_result = analyze_plant_disease(temp_path)
    
    return jsonify({
        'analysis': analysis_result,
        'image': data['image']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
