import os
from flask import Flask, render_template, request, redirect, url_for
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Configure Gemini
genai.configure(api_key="AIzaSyDF7DAX4sepDhh-PXSGyMlcczzhxtEHixE")
model = genai.GenerativeModel('gemini-2.5-flash')
def get_gemini_response(image_path):
    img = Image.open(image_path)
    prompt = """
    Analyze this plant leaf image. Provide:
    1. Disease Name (or 'Healthy')
    2. Confidence Level
    3. Symptoms observed
    4. Treatment Recommendations for farmers
    5. Prevention Tips
    Format the output clearly.
    """
    response = model.generate_content([prompt, img])
    return response.text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Get diagnosis from Gemini
        analysis = get_gemini_response(filepath)
        
        return render_template('result.html', 
                               analysis=analysis, 
                               image_path=filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
