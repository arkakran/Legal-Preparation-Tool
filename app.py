import os
import tempfile
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from utils.processor import HybridProcessor

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  #file sie
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {'pdf'}
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', groq_configured=bool(GROQ_API_KEY))

@app.route('/upload', methods=['POST'])
def upload_file():
    if not GROQ_API_KEY:
        flash('GROQ_API_KEY is not configured. Please check your environment variables.', 'error')
        return redirect(url_for('index'))
    
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                file.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            # Process the document
            processor = HybridProcessor(GROQ_API_KEY)
            results = processor.process(tmp_path, filename)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            if results:
                return render_template('results.html', results=results)
            else:
                flash('Failed to process the document. Please try again.', 'error')
                return redirect(url_for('index'))
                
        except Exception as e:
            flash(f'Error processing document: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload a PDF file.', 'error')
        return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)