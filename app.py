import os
import io
import uuid
import requests
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "monochrome_agent_secret_key_1337")

# =====================================================================
# AGENT INSTRUCTIONS (Customize your contract review rules below)
# =====================================================================
AGENT_INSTRUCTIONS = """
You are the Watsonx Granite Contract & Policy Review Agent, an elite AI assistant specialized in legal compliance, risk mitigation, and document optimization. 

OPERATIONAL PROTOCOLS:
1. Conduct a rigorous review of any provided company policy, contract, or corporate document.
2. Identify potential legal risks, operational bottlenecks, non-compliance issues, and language ambiguities.
3. Suggest clear, actionable redrafts or amendments for problematic clauses.
4. Explain the rationale behind every modification, focusing on protecting corporate interests, liability reduction, and compliance alignment.
5. Structure your output clearly using Markdown elements: Use bold section headings, list items for clarity, and tables for comparative "Current Clause vs. Proposed Change" suggestions.
6. When answering user follow-up questions, always stay objective, context-aware, and highly practical.
7. Standard Disclaimer: Conclude your initial document analysis with a subtle, professional note: "Disclaimer: This analysis is powered by AI and is meant for administrative assistance. It does not constitute formal legal representation or binding legal advice."
8. Maintain a professional, balanced, and humble tone. Never claim 100% legal coverage or make overconfident assertions.
"""
# =====================================================================

# Fetch Watsonx Credentials
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").rstrip('/')
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct")

def get_watsonx_token():
    """Generates an IAM access token using the provided IBM Cloud API Key."""
    if not WATSONX_APIKEY:
        return None
    
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": WATSONX_APIKEY
    }
    try:
        response = requests.post(url, headers=headers, data=data, timeout=15)
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as e:
        app.logger.error(f"Failed to fetch IAM token: {str(e)}")
        return None

def query_watsonx(prompt):
    """Sends inference requests to Watsonx.ai text generation endpoint using a Granite model."""
    token = get_watsonx_token()
    if not token:
        return "Error: Watsonx API authentication failed. Please verify your WATSONX_APIKEY in the .env configuration."

    if not WATSONX_PROJECT_ID:
        return "Error: WATSONX_PROJECT_ID is missing. Please configure it inside your .env file."

    endpoint = f"{WATSONX_URL}/ml/v1/text/generation?version=2023-05-29"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    # Text Generation Payload for Watsonx
    payload = {
        "model_id": WATSONX_MODEL_ID,
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 1500,
            "min_new_tokens": 1,
            "repetition_penalty": 1.05
        },
        "project_id": WATSONX_PROJECT_ID
    }

    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        res_data = response.json()
        
        results = res_data.get("results", [])
        if results:
            return results[0].get("generated_text", "").strip()
        return "Error: Watsonx returned an empty response body."
    except requests.exceptions.RequestException as re:
        app.logger.error(f"Inference request error: {str(re)}")
        return f"Error contacting Watsonx endpoint: {str(re)}"

def extract_text(file_stream, filename):
    """Extracts text content from TXT, PDF, or DOCX formats."""
    ext = filename.split('.')[-1].lower()
    
    if ext in ['txt', 'md', 'markdown']:
        return file_stream.read().decode('utf-8', errors='ignore')
    
    elif ext == 'pdf':
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_stream.read()))
            text_slices = []
            for page in pdf_reader.pages:
                text_slices.append(page.extract_text() or "")
            return "\n".join(text_slices)
        except Exception as e:
            return f"[PDF Parsing Error: {str(e)}]"
            
    elif ext == 'docx':
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_stream.read()))
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            return f"[Word document Parsing Error: {str(e)}]"
            
    else:
        # Fallback to direct string conversion
        try:
            return file_stream.read().decode('utf-8', errors='ignore')
        except Exception:
            return "[Error: Unsupported file format]"

@app.route('/')
def index():
    # Initialize a clean session ID if missing
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session["document_text"] = ""
        session["chat_history"] = []
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Handles parsing from files or custom text pastes."""
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        text = extract_text(file, file.filename)
        session["document_text"] = text
        session["chat_history"] = []  # reset history on new document
        session.modified = True
        return jsonify({"status": "success", "text": text, "filename": file.filename})
    
    elif request.is_json:
        data = request.get_json()
        text = data.get("text", "").strip()
        if text:
            session["document_text"] = text
            session["chat_history"] = []  # reset history on new text paste
            session.modified = True
            return jsonify({"status": "success", "text": text, "filename": "Pasted Clipboard Text"})
            
    return jsonify({"status": "error", "message": "No valid document text received."})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handles user prompts, updating contextual memory and formatting Granite prompts."""
    data = request.get_json() or {}
    user_msg = data.get("message", "").strip()
    is_initial_review = data.get("initial_review", False)
    
    if not user_msg and not is_initial_review:
        return jsonify({"status": "error", "message": "Message content is empty."})
    
    doc_text = session.get("document_text", "")
    history = session.get("chat_history", [])
    
    # Setup prompt based on whether it is a fresh review trigger or a standard message interaction
    if is_initial_review:
        if not doc_text:
            return jsonify({"status": "error", "message": "Please upload or paste a document first."})
        user_msg = "Please perform a comprehensive initial contract review, highlighting any potential issues, legal vulnerabilities, or unclear clauses, and offer concrete revision recommendations."
        # Reset history on initial review to prevent compounding previous iterations
        history = []
    
    # Construct formatting matching Granite's instruction template
    prompt_builder = []
    prompt_builder.append(f"<|system|>\n{AGENT_INSTRUCTIONS}\n")
    
    if doc_text:
        prompt_builder.append(f"Context document under review:\n---\n{doc_text}\n---\n")
        
    for item in history:
        prompt_builder.append(f"<|user|>\n{item['content']}\n")
        prompt_builder.append(f"<|assistant|>\n{item['response']}\n")
        
    prompt_builder.append(f"<|user|>\n{user_msg}\n<|assistant|>\n")
    full_prompt = "".join(prompt_builder)
    
    ai_response = query_watsonx(full_prompt)
    
    # Save step to current session's memory
    history.append({
        "content": user_msg,
        "response": ai_response
    })
    session["chat_history"] = history
    session.modified = True
    
    return jsonify({
        "status": "success",
        "response": ai_response,
        "history": history
    })

@app.route('/api/clear', methods=['POST'])
def clear_session():
    """Wipes active workspace states."""
    session["document_text"] = ""
    session["chat_history"] = []
    session.modified = True
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)