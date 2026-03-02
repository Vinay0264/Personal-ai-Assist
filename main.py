import os
import random
import threading
import time
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from groq import Groq
from rich import print
import keyboard
from backend.tts import speak, start_tts_generation, play_pregenerated
from backend.stt import listen
from backend.brain import think, clean_text_for_speech, format_history_for_prompt, MODELS, current_model_index
from backend.chat_history import (
    REMEMBER_TRIGGERS,
    CHAT_HISTORY_TRIGGERS,
    add_to_memory,
    show_recent_chats_on_demand,
    save_chat_history
)
from backend.router import route
from backend.brain import slow_display

# ===== LOAD ENVIRONMENT VARIABLES =====
load_dotenv()

# ===== GROQ CLIENT (conversation brain — 14,400 RPD) =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ ERROR: Groq API key not found!")
    exit()
groq_client = Groq(api_key=GROQ_API_KEY)

# ===== GEMINI CLIENT (title generation only — ~1 call per session) =====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

conversation_history = []
MAX_HISTORY = 20

# ===== AUTOMATION TASK PREFIXES =====
AUTOMATION_PREFIXES = ("open", "close", "play", "system", "media", "google search", "youtube search")

# ===== MEDIA KEYWORDS — detected directly from user sentence =====
# Router can't reliably classify pause/next/prev as automation, so we detect them here
MEDIA_KEYWORDS = {
    "pause":        "pause",
    "resume":       "play",
    "unpause":      "play",
    "next song":    "next",
    "next track":   "next",
    "skip":         "next",
    "previous song":"previous",
    "prev song":    "previous",
    "go back":      "previous",
    "stop playing": "stop",
}

# ===== EXIT TRIGGERS =====
EXIT_TRIGGERS = [
    "you can quit now",
    "you can quit",
    "you can exit now",
    "you can exit",
    "you can stop",
    "you can go to sleep",
    "you can sleep now",
    "stop saiyaara",
    "exit saiyaara",
    "saiyaara exit",
    "saiyaara stop",
    "saiyaara sleep",
]


def greet_on_startup():
    """Greet user based on time of day"""
    hour = datetime.now().hour

    if 5 <= hour < 12:
        time_of_day = "morning"
    elif 12 <= hour < 17:
        time_of_day = "afternoon"
    elif 17 <= hour < 21:
        time_of_day = "evening"
    else:
        time_of_day = "night"

    greetings = {
        "morning": [
            "Good morning sir! Hope you slept well. What are we working on today?",
            "Good morning Vinay! Fresh start to a new day. How can I help?",
            "Morning sir! Ready when you are.",
        ],
        "afternoon": [
            "Good afternoon sir! How's the day going so far?",
            "Afternoon Vinay! What can I do for you?",
            "Good afternoon sir! What do you need?",
        ],
        "evening": [
            "Good evening sir! Long day? I'm here if you need anything.",
            "Good evening Vinay! What's on your mind?",
            "Evening sir! How can I help you tonight?",
        ],
        "night": [
            "Still up, sir? I'm here. What do you need?",
            "Good night Vinay! Working late? Let's get it done.",
            "Night sir! What are we doing?",
        ],
    }

    greeting = random.choice(greetings[time_of_day])

    tts_file, tts_ready = start_tts_generation(greeting)
    tts_ready.wait(timeout=10)

    print("\n🤖 SAIYAARA:")

    audio_thread = threading.Thread(
        target=play_pregenerated,
        args=(tts_file, tts_ready),
        daemon=True
    )
    audio_thread.start()
    slow_display(greeting)
    audio_thread.join()


def get_exit_message(user_lower):
    """Return contextual farewell based on what user said"""
    if "sleep" in user_lower:
        messages = [
            "Going to sleep now. Goodnight sir!",
            "Sleep mode on. Rest well too Vinay!",
            "Okay, lights out. Goodnight!",
        ]
    elif "stop" in user_lower:
        messages = [
            "Stopping now. Take care sir!",
            "Alright, stopping. See you soon Vinay!",
            "Stopped. Come back anytime sir!",
        ]
    elif "quit" in user_lower:
        messages = [
            "Quitting now. Bye sir!",
            "Alright, quitting. Take care Vinay!",
            "See you next time sir!",
        ]
    elif "exit" in user_lower:
        messages = [
            "Exiting now. See you soon sir!",
            "Okay, exiting. Bye Vinay!",
            "Exited. Come back anytime sir!",
        ]
    else:
        messages = [
            "Going offline now. Take care sir!",
            "Signing off. See you soon Vinay!",
            "Okay, bye for now sir!",
        ]
    return random.choice(messages)


def handle_coming_soon(task):
    """Friendly response for features not yet built"""
    if task.startswith("open"):
        message = "App opening is coming soon! I'll be able to open apps for you."
    elif task.startswith("close"):
        message = "App closing is coming soon! I'll be able to close apps for you."
    elif task.startswith("play"):
        message = "Music and video control is coming soon! I'll be able to play things for you."
    elif task.startswith("generate"):
        message = "Image generation is coming soon! I'll be able to create images for you."
    elif task.startswith("reminder"):
        message = "Reminders are coming soon! I'll be able to set alarms and reminders for you."
    elif task.startswith("system"):
        message = "System controls are coming soon! I'll be able to control volume and settings for you."
    elif task.startswith("content"):
        message = "Content writing is coming soon! I'll be able to write emails and documents for you."
    elif task.startswith("google search"):
        message = "Google search is coming soon! I'll be able to search the web for you."
    elif task.startswith("youtube search"):
        message = "YouTube search is coming soon! I'll be able to search YouTube for you."
    else:
        message = "That feature is coming soon!"

    speak(message, display=True)


def do_save_and_exit(user_lower):
    """Save chat and return True to signal exit"""
    speak(get_exit_message(user_lower), display=True)
    save_chat_history(
        conversation_history,
        format_history_for_prompt,
        gemini_client
    )
    return True


def process_input(user_text):
    """Process any input — text or voice — through the same pipeline"""
    global conversation_history, current_model_index

    if not user_text:
        return False

    user_lower = user_text.lower().strip()

    # ── EXIT TRIGGER CHECK ──
    for trigger in EXIT_TRIGGERS:
        if trigger in user_lower:
            return do_save_and_exit(user_lower)

    # ── CHAT HISTORY TRIGGER CHECK ──
    for trigger in CHAT_HISTORY_TRIGGERS:
        if trigger in user_lower:
            loaded = show_recent_chats_on_demand(conversation_history, lambda t: speak(t, display=True))
            if loaded is not None:
                conversation_history = loaded
            return False

    # ── MEMORY TRIGGER CHECK ──
    for trigger in REMEMBER_TRIGGERS:
        if trigger in user_lower:
            add_to_memory(user_text, conversation_history, gemini_client, clean_text_for_speech)
            break

    # ── ROUTE THE QUERY → returns a list ──
    tasks = route(user_text)

    # ── DETECT MEDIA COMMANDS DIRECTLY FROM USER SENTENCE ──
    # Router can't reliably classify pause/next/prev as automation
    media_task = None
    for kw, action in MEDIA_KEYWORDS.items():
        if kw in user_lower:
            media_task = f"media {action}"
            break

    # ── SEPARATE: collect automation tasks, run all together in parallel ──
    automation_tasks = []

    for task in tasks:

        if task == "exit":
            return do_save_and_exit(user_lower)

        elif task.startswith("general"):
            ai_response, conversation_history = think(
                user_text, conversation_history, groq_client
            )

        elif task.startswith("realtime"):
            from backend.search import realtime_search
            search_query = task[len("realtime"):].strip() or user_text
            realtime_search(
                query=search_query,
                groq_client=groq_client,
                clean_text_fn=clean_text_for_speech,
                speak_fn=speak,
                slow_display_fn=slow_display,
                start_tts_fn=start_tts_generation,
                play_pregenerated_fn=play_pregenerated,
            )

        elif task.startswith(AUTOMATION_PREFIXES):
            automation_tasks.append(task)

        elif task.startswith(("generate", "reminder", "content")):
            handle_coming_soon(task)

        else:
            ai_response, conversation_history = think(
                user_text, conversation_history, groq_client
            )


def main():
    global conversation_history, current_model_index

    print("\n" + "=" * 60)
    print("[bold blue]🤖 SAIYAARA - Your Personal AI Assistant")
    print("=" * 60)
    print("[bold green]✅ Status: Active and Ready!")
    print("=" * 60)
    print("💡 Type your message and press Enter anytime")
    print("💡 Press F2 to start voice input, F2 again to stop")
    print("💡 Say/Type: 'remember this — [fact]' to save to memory")
    print("💡 Say/Type: 'show my chats' to browse history")
    print("💡 Say/Type: 'you can exit now' / 'you can sleep' to exit")
    print("💡 Press Ctrl+C for emergency exit")
    print("=" * 60)

    # ===== STARTUP GREETING =====
    greet_on_startup()

    # ===== MAIN LOOP =====
    while True:
        try:
            msg_count = len(conversation_history)
            if msg_count > 0:
                print(f"\n[🧠 {msg_count} messages in memory | F2 for voice]")
            else:
                print("\n[F2 for voice input]")

            # ── WAIT FOR TEXT INPUT OR F2 ──
            voice_triggered = threading.Event()

            def on_f2():
                voice_triggered.set()

            keyboard.add_hotkey('f2', on_f2)

            print("You: ", end="", flush=True)

            input_done = threading.Event()
            text_result = [None]

            def get_text():
                try:
                    text_result[0] = input("")
                    input_done.set()
                except:
                    input_done.set()

            text_thread = threading.Thread(target=get_text, daemon=True)
            text_thread.start()

            import time
            while not input_done.is_set() and not voice_triggered.is_set():
                time.sleep(0.05)

            keyboard.remove_hotkey('f2')

            if voice_triggered.is_set() and not input_done.is_set():
                print("\n")
                user_text = listen()
            else:
                user_text = text_result[0].strip() if text_result[0] else None

            if not user_text:
                continue

            should_exit = process_input(user_text)
            if should_exit:
                break

        except KeyboardInterrupt:
            print("\n\n👋 Emergency exit. Saving chat...")
            save_chat_history(
                conversation_history,
                format_history_for_prompt,
                gemini_client
            )
            break

        except Exception as e:
            print(f"❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()