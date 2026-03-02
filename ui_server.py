"""
ui_server.py — SAIYAARA UI Bridge Server v4

HOW TO RUN:
    python ui_server.py

DO NOT run main.py at the same time — that causes double TTS.
This server does everything: serves the HTML + handles all API calls.

Browser opens automatically at http://127.0.0.1:5500
"""

import os
import sys
import time
import threading
import webbrowser
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from groq import Groq

# ── Import ONLY utility functions from brain.py — NOT think() ──
from backend.brain import (
    clean_text_for_speech,
    format_history_for_prompt,
    build_groq_messages,
    get_realtime_info,
    MODELS,
)
from backend.stt import listen
from backend.tts import start_tts_generation, play_pregenerated
from backend.router import route
from backend.chat_history import (
    load_long_term_memory,
    load_recent_chats,
    save_chat_history,
    add_to_memory,
    build_memory_prompt,
    REMEMBER_TRIGGERS,
)

load_dotenv()

# ── BASE DIR (where ui_server.py lives) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Auto-detect where saiyaara_ui.html actually is
_candidates = [
    os.path.join(BASE_DIR, 'frontend'),    # Project/frontend/saiyaara_ui.html
    os.path.join(BASE_DIR, 'Front-end'),   # Project/Front-end/saiyaara_ui.html
    os.path.join(BASE_DIR, 'frontend_ui'), # another common name
    BASE_DIR,                               # same folder as ui_server.py
]
FRONTEND_DIR = BASE_DIR  # fallback default
for _c in _candidates:
    if os.path.isfile(os.path.join(_c, 'saiyaara_ui.html')):
        FRONTEND_DIR = _c
        break

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

groq_client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
GEMINI_KEY    = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

conversation_history = []
current_model_index  = 0


# ═══════════════════════════════════════════════════
# SERVE HTML — Flask serves the UI directly
# Visit http://127.0.0.1:5500 to open the UI
# ═══════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve the main UI page."""
    return send_from_directory(FRONTEND_DIR, 'saiyaara_ui.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve any static files from frontend folder."""
    return send_from_directory(FRONTEND_DIR, filename)


# ═══════════════════════════════════════════════════
# think_ui — Pure Groq call. NO TTS. NO display.
# ═══════════════════════════════════════════════════
def think_ui(user_input):
    global conversation_history, current_model_index

    memory_section   = build_memory_prompt()
    realtime_section = get_realtime_info()

    system_prompt = (
        "You are SAIYAARA, a friendly AI assistant and companion.\n\n"
        "PERSONALITY:\n"
        "- Talk like a supportive, caring friend\n"
        "- Be warm, casual, and conversational\n"
        "- Never mention you're an AI, LLM, or language model\n"
        "- ALWAYS address the user as 'sir' — never their actual name out loud\n"
        "- Occasionally use 1-2 relevant emojis to add warmth\n"
        "- Keep responses concise and natural (2-4 sentences)\n"
        "- If user says 'remember this/that', 'don't forget', 'keep in mind' — confirm warmly\n"
        f"{memory_section}{realtime_section}"
    )

    conversation_history.append({"role": "user", "parts": [user_input]})
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]

    wait_times = [10, 20, 40]

    while current_model_index < len(MODELS):
        model_name = MODELS[current_model_index]
        for attempt in range(3):
            try:
                messages = build_groq_messages(conversation_history, system_prompt)
                stream   = groq_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.8,
                    stream=True,
                )
                full_response = ""
                for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        full_response += token

                clean = clean_text_for_speech(full_response)
                conversation_history.append({"role": "model", "parts": [clean]})
                print(f"✅ Response ready ({len(clean.split())} words)")
                return clean

            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    if attempt < 2:
                        print(f"⏳ Rate limit, waiting {wait_times[attempt]}s...")
                        time.sleep(wait_times[attempt])
                    else:
                        current_model_index += 1
                        break
                else:
                    if conversation_history and conversation_history[-1]["role"] == "user":
                        conversation_history.pop()
                    return "Sorry sir, I'm having some trouble. Can you try again?"

    if conversation_history and conversation_history[-1]["role"] == "user":
        conversation_history.pop()
    return "I've hit my daily quota, sir. Let's continue tomorrow!"


# ═══════════════════════════════════════════════════
# TTS — fires in background thread.
# Called with a small delay so Flask returns the
# HTTP response to browser BEFORE audio starts.
# Browser typewriter and TTS begin at the same time.
# ═══════════════════════════════════════════════════
def _play_tts(text):
    try:
        tts_file, tts_ready = start_tts_generation(text)
        tts_ready.wait(timeout=15)
        play_pregenerated(tts_file, tts_ready)
    except Exception as e:
        print(f"⚠️  TTS error: {e}")

def fire_tts(text, delay=0.05):
    """Schedule TTS `delay` seconds from now (after Flask returns response)."""
    t = threading.Timer(delay, _play_tts, args=[text])
    t.daemon = True
    t.start()


# ═══════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════

@app.route('/ping')
def ping():
    return jsonify({"status": "online", "version": "4.2"})


@app.route('/chat', methods=['POST'])
def chat():
    data      = request.get_json()
    user_text = data.get('message', '').strip()
    if not user_text:
        return jsonify({"error": "empty"}), 400

    print(f"\n👤 User: {user_text}")

    # Memory trigger
    for trigger in REMEMBER_TRIGGERS:
        if trigger in user_text.lower():
            add_to_memory(user_text, conversation_history, gemini_client, clean_text_for_speech)
            break

    # Route
    tasks    = route(user_text)
    response = None

    for task in tasks:
        if task.startswith("general"):
            response = think_ui(user_text)
        else:
            response = f"That feature ({task.split()[0]}) is coming soon, sir!"

    print(f"🤖 SAIYAARA: {response}")

    # ── Generate TTS file NOW, before sending HTTP response ──
    # Browser gets text + TTS is already ready → play starts within ~50ms of text appearing
    tts_file  = None
    tts_ready = None
    if response:
        try:
            tts_file, tts_ready = start_tts_generation(response)
            # Wait up to 8s for TTS file to be ready
            tts_ready.wait(timeout=8)
        except Exception as e:
            print(f"⚠️  Pre-gen TTS error: {e}")

    # ── Return text to browser ──
    resp = jsonify({"response": response})

    # ── Play TTS in background (file already generated, near-instant) ──
    if tts_file and tts_ready:
        t = threading.Thread(target=play_pregenerated, args=(tts_file, tts_ready), daemon=True)
        t.start()

    return resp


@app.route('/voice', methods=['POST'])
def voice():
    try:
        text = listen()
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e), "text": None})


@app.route('/chats')
def chats():
    try:
        recent = load_recent_chats(limit=10)
        return jsonify([
            {
                "title":    c.get("title", "Untitled"),
                "date":     c.get("date", ""),
                "messages": c.get("messages", []),
                "path":     c.get("path", ""),
            }
            for c in recent
        ])
    except:
        return jsonify([])


@app.route('/memory')
def memory():
    try:
        facts = load_long_term_memory()
        return jsonify({"count": len(facts), "facts": facts})
    except:
        return jsonify({"count": 0, "facts": []})


@app.route('/new_chat', methods=['POST'])
def new_chat():
    global conversation_history
    if conversation_history:
        save_chat_history(conversation_history, format_history_for_prompt, gemini_client)
    conversation_history = []
    return jsonify({"status": "reset"})


@app.route('/delete_chat', methods=['POST'])
def delete_chat():
    try:
        data      = request.get_json()
        file_path = data.get('path', '').strip()
        title     = data.get('title', '').strip()

        print(f"\n🗑  Delete request — path={file_path!r}  title={title!r}")

        # Normalise path separators (browser may send forward slashes on Windows)
        if file_path:
            file_path = os.path.normpath(file_path)

        # Try 1: exact path
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"✅ Deleted by path: {file_path}")
            return jsonify({"status": "deleted"})

        # Try 2: path relative to BASE_DIR
        if file_path:
            rel = os.path.join(BASE_DIR, file_path)
            rel = os.path.normpath(rel)
            if os.path.exists(rel):
                os.remove(rel)
                print(f"✅ Deleted by relative path: {rel}")
                return jsonify({"status": "deleted"})

        # Try 3: search all chats by title
        all_chats = load_recent_chats(limit=50)
        print(f"   Searching {len(all_chats)} chats for title match...")
        for c in all_chats:
            c_title = c.get('title','').strip()
            c_path  = c.get('path','')
            print(f"   → {c_title!r}  {c_path!r}")
            if c_title.lower() == title.lower() and c_path and os.path.exists(c_path):
                os.remove(c_path)
                print(f"✅ Deleted by title: {c_path}")
                return jsonify({"status": "deleted"})

        # Try 4: filename contains title slug
        if title:
            slug = title.lower().replace(' ', '-')
            for c in all_chats:
                fname = os.path.basename(c.get('path','')).lower()
                if slug in fname:
                    p = c.get('path','')
                    if p and os.path.exists(p):
                        os.remove(p)
                        print(f"✅ Deleted by slug: {p}")
                        return jsonify({"status": "deleted"})

        print(f"❌ Delete: no match found for path={file_path!r} title={title!r}")
        return jsonify({"status": "not_found", "tried_path": file_path, "tried_title": title}), 404
    except Exception as e:
        print(f"⚠️  Delete error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/status')
def status():
    return jsonify({"status": "idle"})


# ═══════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════
if __name__ == '__main__':
    HOST = '127.0.0.1'
    PORT = 5500
    URL  = f'http://{HOST}:{PORT}'

    print("\n" + "═" * 55)
    print("   🤖  SAIYAARA — UI Server")
    print("═" * 55)
    print(f"\n   ✅  Running at  →  {URL}")
    print(f"   📁  HTML at    →  {os.path.join(FRONTEND_DIR, "saiyaara_ui.html")}")
    print(f"\n   ⚠️   DO NOT run main.py at the same time!")
    print(f"       That causes double TTS (voice playing twice).")
    print("\n   Opening browser in 1.5 seconds...")
    print("   Ctrl+C to stop.\n")
    print("═" * 55 + "\n")

    # Auto-open Chrome first, fall back to system default
    def open_chrome(url):
        import subprocess, shutil
        chrome_paths = [
            os.path.join('C:\\', 'Program Files', 'Google', 'Chrome', 'Application', 'chrome.exe'),
            os.path.join('C:\\', 'Program Files (x86)', 'Google', 'Chrome', 'Application', 'chrome.exe'),
            os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Google', 'Chrome', 'Application', 'chrome.exe'),
        ]
        opened = False
        for path in chrome_paths:
            if os.path.exists(path):
                subprocess.Popen([path, url])
                opened = True
                break
        if not opened:
            webbrowser.open(url)  # fallback to system default

    threading.Timer(1.5, open_chrome, args=[URL]).start()

    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )