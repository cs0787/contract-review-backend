import os
import io
import uuid
import math
import requests
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "monochrome_agent_secret_key_1337")

# =====================================================================
# BROADENED AGENT INSTRUCTIONS
# =====================================================================
AGENT_INSTRUCTIONS = """
You are the Watsonx Granite Enterprise Compliance & Corporate Advisory Agent, an elite AI assistant specialized in regulatory standards, corporate governance, legal risk mitigation, and document optimization.

OPERATIONAL PROTOCOLS:
1. Conduct a rigorous, multi-perspective review of corporate documents, guidelines, agreements, or general corporate legal queries.
2. Cross-reference provided local documents against external legal, regulatory, or market-standard frameworks when relevant context is provided.
3. Identify operational bottlenecks, compliance conflicts, language ambiguities, and liability exposure.
4. Suggest clear, actionable redrafts or compliance amendments. Always ground your revisions in the provided factual context (both local documentation and verified external web search data).
5. If live search context is available, prioritize its real-world facts to answer current regulatory or industrial questions, explicitly citing the source title/URL provided.
6. Structure your output clearly using Markdown: bold section headings, logical list items, and clean comparative tables for "Current State vs. Proposed Standard/Change".
7. Conclude your analysis with a professional note: "Disclaimer: This analysis is powered by AI and live-retrieved data for administrative assistance. It does not constitute formal legal representation or binding legal advice."
8. Maintain an objective, balanced, and humble tone. Never make overconfident assertions, claim perfect legal coverage, or synthesize/hallucinate legal statutes without source evidence.

RESULT FORMAT:
The final review report includes:
1.Executive Summary 
2.Compliance Status 
3.Clause-by-Clause Analysis 
4.Risk Level 
5.Suggested Corrections 
6.Missing Clauses 
7.Overall Contract Score 

"""

# Fetch Watsonx Credentials
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").rstrip('/')
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct")
WATSONX_EMBEDDING_MODEL_ID = os.getenv("WATSONX_EMBEDDING_MODEL_ID", "ibm/slate-125m-english-rtrvr")

# Real-World Search API Config
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Server-side in-memory store for session embeddings
# Map: session_id -> { "chunks": [str, ...], "embeddings": [[float, ...], ...] }
RAG_STORE = {}

# =====================================================================
# PURE PYTHON RAG UTILITIES (Math & Chunking)
# =====================================================================

def chunk_text(text, chunk_size=300, chunk_overlap=50):
    """Splits text into sliding window chunks with overlap."""
    if not text:
        return []
    words = text.split()
    word_count = len(words)
    
    if word_count == 0:
        return []
    
    # If the text is shorter than the desired chunk size, return it as one chunk
    if word_count <= chunk_size:
        return [text]
        
    chunks = []
    i = 0
    while i < word_count:
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - chunk_overlap
    return chunks

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def magnitude(v):
    return math.sqrt(sum(x * x for x in v))

def cosine_similarity(v1, v2):
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    if not mag1 or not mag2:
        return 0.0
    return dot_product(v1, v2) / (mag1 * mag2)

# =====================================================================
# LIVE WEB RETRIEVAL (To mitigate hallucinations with real-world data)
# =====================================================================

def search_web(query, max_results=3):
    """Fetches real-world search results from Tavily API to ground the model."""
    if not TAVILY_API_KEY:
        app.logger.warning("TAVILY_API_KEY not found in environment. Web search skipped.")
        return ""
    
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results
    }
    try:
        response = requests.post(url, json=payload, timeout=12)
        response.raise_for_status()
        results = response.json().get("results", [])
        
        snippets = []
        for index, item in enumerate(results, 1):
            title = item.get("title", "Reference")
            source_url = item.get("url", "#")
            content = item.get("content", "")
            snippets.append(f"[{index}] Source: {title} ({source_url})\nFact: {content}")
        return "\n\n".join(snippets)
    except Exception as e:
        app.logger.error(f"Web search failed: {str(e)}")
        return ""

# =====================================================================
# WATSONX API UTILITIES
# =====================================================================

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

def get_watsonx_embeddings(texts):
    """Fetches text embeddings from Watsonx REST API for a list of strings."""
    token = get_watsonx_token()
    if not token or not WATSONX_PROJECT_ID:
        return None

    endpoint = f"{WATSONX_URL}/ml/v1/text/embeddings?version=2024-05-02"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    payload = {
        "model_id": WATSONX_EMBEDDING_MODEL_ID,
        "inputs": texts,
        "project_id": WATSONX_PROJECT_ID
    }

    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        res_data = response.json()
        
        results = res_data.get("results", [])
        return [item["embedding"] for item in results if "embedding" in item]
    except Exception as e:
        app.logger.error(f"Embedding API Request Failed: {str(e)}")
        return None

def get_watsonx_embeddings_batched(texts, batch_size=20):
    """Splits input into batches to accommodate upstream embedding constraints."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = get_watsonx_embeddings(batch)
        if batch_embeddings is None:
            return None
        embeddings.extend(batch_embeddings)
    return embeddings

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
        try:
            return file_stream.read().decode('utf-8', errors='ignore')
        except Exception:
            return "[Error: Unsupported file format]"

def retrieve_relevant_chunks(session_id, query, top_k=5):
    """Finds top_k most relevant chunks using Cosine Similarity."""
    store = RAG_STORE.get(session_id)
    if not store or not store.get("chunks") or not store.get("embeddings"):
        return []
        
    query_embeddings = get_watsonx_embeddings([query])
    if not query_embeddings:
        return []
    query_vector = query_embeddings[0]
    
    scored_chunks = []
    for chunk, chunk_vector in zip(store["chunks"], store["embeddings"]):
        similarity = cosine_similarity(query_vector, chunk_vector)
        scored_chunks.append((chunk, similarity))
        
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:top_k]]

# =====================================================================
# FLASK ROUTING
# =====================================================================

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
    """Handles parsing from files or custom text pastes and builds vector indexes."""
    session_id = session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id

    # Clean old index if updating
    if session_id in RAG_STORE:
        del RAG_STORE[session_id]

    text = ""
    filename = ""

    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        text = extract_text(file, file.filename)
        filename = file.filename
    
    elif request.is_json:
        data = request.get_json()
        text = data.get("text", "").strip()
        filename = "Pasted Clipboard Text"
            
    if text:
        session["document_text"] = text
        session["chat_history"] = []
        session.modified = True

        # Process chunks and vector index
        chunks = chunk_text(text)
        if chunks:
            embeddings = get_watsonx_embeddings_batched(chunks)
            if embeddings:
                RAG_STORE[session_id] = {
                    "chunks": chunks,
                    "embeddings": embeddings
                }
        
        return jsonify({"status": "success", "text": text, "filename": filename})

    return jsonify({"status": "error", "message": "No valid document text received."})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handles user prompts, retrieving target context from local RAG and live web search."""
    data = request.get_json() or {}
    user_msg = data.get("message", "").strip()
    is_initial_review = data.get("initial_review", False)
    
    if not user_msg and not is_initial_review:
        return jsonify({"status": "error", "message": "Message content is empty."})
    
    doc_text = session.get("document_text", "")
    session_id = session.get("session_id")
    history = session.get("chat_history", [])
    
    if is_initial_review:
        if not doc_text:
            return jsonify({"status": "error", "message": "Please upload or paste a document first."})
        user_msg = "Please perform an initial compliance and risk review of this document, and provide concrete recommendations based on best practices and regulations."
        history = []

    # 1. LOCAL DOCUMENT CONTEXT (Via Vector Search)
    context_text = ""
    if doc_text and session_id:
        if is_initial_review:
            # Anchor queries for initial analysis around common compliance friction points
            search_query = "liability limitation, compliance breaches, governing law, termination clauses, indemnification terms"
            retrieved_chunks = retrieve_relevant_chunks(session_id, search_query, top_k=5)
        else:
            retrieved_chunks = retrieve_relevant_chunks(session_id, user_msg, top_k=4)
            
        if retrieved_chunks:
            context_text = "\n\n---\n\n".join(retrieved_chunks)
        else:
            # Fallback to simple context truncation if search failed
            context_text = doc_text[:3000]

    # 2. REAL-WORLD DATA CONTEXT (Via Live Web Search)
    web_context = ""
    # Decide if a web search is needed based on query terms or missing document context
    search_triggers = ["update", "news", "regulation", "law", "statute", "latest", "rule", "standard", "gdpr", "sec", "compliance", "policy"]
    needs_web_search = not doc_text or any(kw in user_msg.lower() for kw in search_triggers)
    
    if needs_web_search and TAVILY_API_KEY:
        # Formulate search query focusing on regulatory facts or standards
        search_query = f"site:gov OR legal OR regulatory standards {user_msg}" if not doc_text else f"standard industry compliance rule for {user_msg}"
        web_context = search_web(search_query, max_results=3)

    # 3. PROMPT COMPOSITION
    prompt_builder = []
    prompt_builder.append(f"<|system|>\n{AGENT_INSTRUCTIONS}\n")
    
    if context_text:
        prompt_builder.append(f"Context from Uploaded Document:\n---\n{context_text}\n---\n")
    if web_context:
        prompt_builder.append(f"Live External Reference Context (Verified Web Facts):\n---\n{web_context}\n---\n")
        
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
    """Wipes active workspace states and deletes the in-memory vectors."""
    session_id = session.get("session_id")
    if session_id in RAG_STORE:
        del RAG_STORE[session_id]
        
    session["document_text"] = ""
    session["chat_history"] = []
    session.modified = True
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
