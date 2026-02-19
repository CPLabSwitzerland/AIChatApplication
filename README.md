# AIChatApplication

A simple web frontend for interacting with AI APIs using FLASK.

---

## 📂 Project Structure

```
prettyAIChat/

├── backend/               
│   ├── app.py            # Flask UI app
│   │
│   ├── templates/
│   │   ├── app.html      # HTML structure
│   │
│   ├── static/
│       ├─ style.css      # All CSS styles
│       └─ app.js         # All JS functions
│
├── llm/
│   ├── llm_rag.py         # Calls RAG Api at ai-rag-01
│   ├── llm_tinylama.py    # Calls TinyLlama
│   ├── llm_llama3_1_8b.py # Calls Llama3.1.8b
│   ├── llm_mock.py        # Local mock mode locally
│
├── utils/
│   ├── logger_setup.py    # Logger setup
│
├── logs/
    ├── prettyAIChat.log
    ├── prettyAIChat.log.1
    ├── prettyAIChat.log.2
    ├── prettyAIChat.log.3
    ├── prettyAIChat.log.4
    └── prettyAIChat.log.5```

---

## ⚑ Setup Instructions

1. **Clone the repository:**

```bash
git clone git@github.com:CPLabSwitzerland/AIChatApplication
cd prettyAIChat
```

2. **Create and activate a virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Create a `.env` file in the project root** with your credentials:

```
FLASK_SECRET_KEY=your_flask_secret_here
```

> ⚠️ Do **not** commit your `.env` file. It is excluded in `.gitignore` for security.

5. **Run the Flask app with unicorn gevent**

gunicorn -b 0.0.0.0:5000 backend.app:app -k gevent --timeout 60


## 📌 Usage

- Access the web interface in your browser at `http://localhost:5000`
- Each chat session is stored in memory (`CHAT_SESSIONS`) for the current server run
- LLM responses are handled via `llm/llm.py`
- Chat logic is in `backend/app.py`
- Logs are written to `logs/prettyAIChat.log`

---

## 🛠️ Project Notes

- Secrets are loaded from `.env` using `python-dotenv`
- Logging is set up in `utils/logger_setup.py`
- Static files (JS/CSS) are in `backend/static/`
- HTML templates are in `backend/templates/`

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
