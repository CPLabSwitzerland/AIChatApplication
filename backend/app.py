from flask import Flask, render_template, request, jsonify, stream_with_context, Response, session
from llm.llm_mock import MockLLM
from llm.llm_rag import RagLLM
from llm.llm_tinyllama import TinyLlamaLLM
from llm.llm_llama3_1_8b import Llama3_1_8bLLM
from utils.logger_setup import getlogger
import uuid

# Initialize logger for the Flask app
logger = getlogger("app")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "e4b7a6f9c3d2e1f0b5a697c4d8e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9"

# --- In-memory session-specific chat histories ---
CHAT_SESSIONS = {}  # { session_id: [ {role, content}, ... ] }

# Default LLM mode
LLM_MODE = "mock"

# ------------------------------------------------------
# Assign a unique session_id to each user automatically
# ------------------------------------------------------
@app.before_request
def ensure_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

def get_user_messages():
    sid = session["session_id"]
    if sid not in CHAT_SESSIONS:
        CHAT_SESSIONS[sid] = []
    return CHAT_SESSIONS[sid]

# --- Helper function to select LLM based on mode ---
def get_llm(mode):
    if mode == "rag":
        return RagLLM()           # Retrieval-augmented generation
    elif mode == "tinyllama":
        return TinyLlamaLLM()     # TinyLlama streaming model
    elif mode == "llama3_1_8b":
        return Llama3_1_8bLLM()   # Llama3.1.8b streaming model
    else:
        return MockLLM()          # Mock LLM for testing

# --- Routes ---
@app.route("/")
def index():
    """
    Main page route.
    Renders the chat interface with the current messages and LLM mode.
    """
    user_messages = get_user_messages()
    return render_template(
        "app.html",
        messages=user_messages,
        llm_mode=LLM_MODE
    )

@app.route("/set_mode", methods=["POST"])
def set_mode():
    """
    Endpoint to change the active LLM mode.
    Expects JSON: {"mode": "mock"|"rag"|"tinyllama"}
    Logs the user IP and the session_id automatically (via SessionFilter)
    """
    global LLM_MODE
    user_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    mode = request.json.get("mode")
    if mode in ["mock", "rag", "tinyllama", "llama3_1_8b"]:
        LLM_MODE = mode
        logger.info(f"User {user_ip} set LLM mode to {LLM_MODE}")
    return jsonify({"status": "ok", "mode": LLM_MODE})

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    """
    Endpoint to clear the chat history for this session only.
    """
    sid = session["session_id"]
    CHAT_SESSIONS[sid] = []

    user_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    logger.info(f"User {user_ip} cleared their chat history")
    return jsonify({"status": "cleared"})

@app.route("/send_message", methods=["POST"])
def send_message():
    """
    Endpoint to send a user message to the selected LLM.
    Streams the assistant response chunk by chunk to the frontend.
    Updates the session's chat history after streaming completes.
    """
    data = request.json
    prompt = data.get("prompt", "").strip()
    user_messages = get_user_messages()

    if not prompt:
        return jsonify({"status": "empty"})

    user_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    user_messages.append({"role": "user", "content": prompt})
    logger.info(f"User message from {user_ip}: {prompt}")

    llm = get_llm(LLM_MODE)

    def generate():
        """
        Generator to stream the assistant response.
        Each chunk is yielded immediately to the client.
        After streaming, the full response is stored in the session's messages and logged.
        """
        response_text = ""
        for chunk in llm.stream(prompt):
            response_text += chunk
            yield chunk
        user_messages.append({"role": "assistant", "content": response_text})
        logger.info(f"Assistant response: {response_text}")

    return Response(stream_with_context(generate()), mimetype="text/plain")


if __name__ == "__main__":
    # Run Flask app in debug mode on port 5010
    app.run(host="0.0.0.0", port=5010)
