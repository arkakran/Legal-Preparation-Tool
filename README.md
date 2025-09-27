# Legal Document Analysis Tool

A Flask-based web application that analyzes legal documents using hybrid AI techniques to extract key legal points, arguments, and precedents. The tool combines rule-based analysis, statistical methods, and large language model (LLM) processing to provide comprehensive document insights.

## Features

- **PDF Document Processing**: Extract and analyze text from legal PDF documents
- **Hybrid Analysis Engine**: Combines three analysis methods:
  - Rule-based pattern matching for legal citations and terms
  - Statistical analysis using TF-IDF vectorization
  - LLM-powered content analysis via Groq API
- **Stance Classification**: Automatically categorizes points as "for", "against", or "neutral"
- **Balanced Results**: Ensures equal representation of different perspectives
- **Interactive Web Interface**: Clean, responsive UI for document upload and results viewing

## Technology Stack

- **Backend**: Flask, Python
- **PDF Processing**: pdfplumber for text extraction
- **Machine Learning**: scikit-learn for statistical analysis
- **AI Integration**: Groq API for LLM analysis
- **Deployment**: Render-ready with gunicorn

## Project Structure

```
legal-preparation-tool/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── utils/
│   ├── __init__.py
│   ├── models.py         # Data models and structures
│   ├── processor.py      # Main hybrid processing engine
│   ├── pdf_processor.py  # PDF text extraction logic
│   ├── analyzers.py      # Rule-based and statistical analyzers
│   └── llm_client.py     # Groq API client
└── templates/
│   ├── index.html        # Upload interface
│   └── results.html      # Analysis results display
└── static/
    ├── style.css        # Css for styling
```

## Setup and Installation

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/arkakran/Legal-Preparation-Tool.git
   cd Legal-Preparation-Tool
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**:
   ```bash
   export GROQ_API_KEY="your-groq-api-key-here"
   export SECRET_KEY="your-secret-key-here"
   ```

5. **Run the application**:
   ```bash
   python app.py
   ```

Visit `http://localhost:5000` to access the application.

### Deployment on Render
https://legal-preparation-tool.onrender.com


## API Requirements

This application requires a [Groq API key](https://console.groq.com/) for LLM analysis functionality. The Groq API provides access to fast inference using models like Llama 3.3.

## Usage

1. **Upload Document**: Select a PDF legal document using the web interface
2. **Processing**: The system will:
   - Extract text from the PDF
   - Apply rule-based legal pattern matching
   - Perform statistical analysis to identify important sections
   - Send key content to Groq LLM for detailed analysis
   - Combine results using confidence scoring
3. **View Results**: Review the top 10 balanced key points with:
   - Point summaries and full content
   - Stance classification (for/against/neutral)
   - Page and line references
   - Confidence scores
   - Analysis method attribution

## Analysis Methods

### Rule-Based Analysis
Identifies legal patterns including:
- Case citations (e.g., "Brown v. Board")
- Statutory references (e.g., "18 U.S.C. § 1461")
- Legal terminology and argument markers

### Statistical Analysis
Uses TF-IDF vectorization to:
- Calculate document importance scores
- Apply positional weighting (earlier content prioritized)
- Normalize scores across paragraphs

### LLM Analysis
Leverages Groq's Llama models to:
- Extract semantic meaning from legal text
- Identify arguments, precedents, and evidence
- Provide structured analysis with importance ratings

## File Size Limits

- Maximum upload size: 16MB
- Supported format: PDF only

## Thankyou.
