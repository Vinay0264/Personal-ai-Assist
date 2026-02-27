import os
from dotenv import load_dotenv
from google import genai

from backend.tts import speak
from backend.stt import listen
from backend.brain import think, clean_text_for_speech, format_history_for_prompt, MODELS, current_model_index
from backend.memory import REMEMBER_TRIGGERS, add_to_memory
from backend.chat_history import (
    CHAT_HISTORY_TRIGGERS,
    show_recent_chats_on_demand,
    save_chat_history
)
from backend.router import route

# ===== LOAD ENVIRONMENT VARIABLES =====
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ ERROR: Gemini API key not found!")
    exit()

client = genai.Client(api_key=GEMINI_API_KEY)

conversation_history = []
MAX_HISTORY = 20


def get_input_choice():
    """Ask user how they want to input (voice or text)"""
    print("\n" + "=" * 60)
    print("📝 How do you want to communicate?")
    print("=" * 60)
    print("1. 🎤 Voice Input (speak)")
    print("2. ⌨️  Text Input (type)")
    print("=" * 60)

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ['1', '2']:
            return choice
        print("❌ Invalid choice. Please enter 1 or 2.")


def type_input():
    """Get text input from keyboard"""
    msg_count = len(conversation_history)
    if msg_count > 0:
        print(f"[🧠 Memory: {msg_count} messages]")
    print("\n⌨️  Type your message:")
    text = input("You: ").strip()
    if text:
        return text
    else:
        print("❌ Empty input. Please type something.")
        return None


def handle_coming_soon(category):
    """Friendly response for features not yet built"""
    messages = {
        "Realtime": "Real-time search is coming soon! I'll be able to look things up on the internet for you.",
        "Open":     "App opening is coming soon! I'll be able to open apps for you.",
        "Close":    "App closing is coming soon! I'll be able to close apps for you.",
        "Play":     "Music and video control is coming soon! I'll be able to play things for you.",
        "Generate": "Image generation is coming soon! I'll be able to create images for you.",
    }
    message = messages.get(category, "That feature is coming soon!")
    print(f"\n🚧 [{category}] {message}")
    speak(message)

from datetime import datetime

def greet_on_startup():
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
    
    import random
    greeting = random.choice(greetings[time_of_day])
    speak(greeting)

def main():
    global conversation_history, current_model_index

    print("\n" + "=" * 60)
    print("🤖 SAIYAARA - Your Personal AI Assistant")
    print("=" * 60)
    print("✅ Status: Active and Ready!")
    print("=" * 60)
    print("💡 Say/Type: 'quit', 'bye', 'goodbye', 'stop' to switch modes")
    print("💡 Say/Type: 'remember this — [fact]' to save to long-term memory")
    print("💡 Say/Type: 'show my chats' or 'previous chats' to browse history")
    print("💡 Press Ctrl+C to exit completely (chat will be saved!)")
    print("=" * 60)

    # ===== STARTUP GREETING =====
    greet_on_startup()

    while True:
        try:
            choice = get_input_choice()

            print(f"\n✅ {'Voice' if choice == '1' else 'Text'} mode activated!")
            if choice == '1':
                print("💡 Press SPACE BAR to start/stop recording")
                print("💡 Say 'bye', 'goodbye', 'quit', or 'stop' to switch modes")
                print("💡 Say 'remember this — [fact]' to save to long-term memory")
            else:
                print("💡 Type 'quit', 'bye', 'goodbye', or 'exit' to switch modes")
                print("💡 Type 'remember this — [fact]' to save to long-term memory")
            print(f"💡 Currently in: {'🎤 VOICE MODE' if choice == '1' else '⌨️ TEXT MODE'}\n")

            failed_attempts = 0
            max_failed_attempts = 3

            while True:
                try:
                    # ── GET INPUT ──
                    if choice == '1':
                        user_text = listen()

                        if user_text is None:
                            failed_attempts += 1
                            print(f"[Attempt {failed_attempts}/{max_failed_attempts}]")
                            if failed_attempts >= max_failed_attempts:
                                print("\n⚠️ Too many failed attempts. Returning to menu...\n")
                                break
                            continue
                        else:
                            failed_attempts = 0
                    else:
                        user_text = type_input()

                    if not user_text:
                        continue

                    user_lower = user_text.lower().strip()

                    # ── CHAT HISTORY TRIGGER CHECK ──
                    # Check before routing (these are meta-commands, not queries)
                    chat_history_triggered = False
                    for trigger in CHAT_HISTORY_TRIGGERS:
                        if trigger in user_lower:
                            chat_history_triggered = True
                            loaded = show_recent_chats_on_demand(conversation_history, speak)
                            if loaded is not None:
                                conversation_history = loaded
                            break

                    if chat_history_triggered:
                        continue

                    # ── MEMORY TRIGGER CHECK ──
                    # Check before routing (these are meta-commands, not queries)
                    for trigger in REMEMBER_TRIGGERS:
                        if trigger in user_lower:
                            add_to_memory(
                                user_text, conversation_history,
                                client, current_model_index,
                                MODELS, clean_text_for_speech
                            )
                            break

                    # ── ROUTE THE QUERY ──
                    decision = route(user_text)

                    # ── HANDLE BASED ON DECISION ──

                    if decision == "exit":
                        speak("Goodbye! Take care. See you next time!")
                        save_chat_history(
                            conversation_history, client,
                            current_model_index, MODELS,
                            format_history_for_prompt
                        )
                        conversation_history.clear()
                        print("\n🔄 Returning to input selection...\n")
                        break

                    elif decision.startswith("general"):
                        ai_response, conversation_history = think(
                            user_text, conversation_history, client
                        )
                        speak(ai_response)

                    elif decision.startswith(("realtime", "open", "close", "play", "generate", "reminder", "system", "content", "google search", "youtube search")):

                        handle_coming_soon(decision)

                    else:
                        # Fallback — treat as General
                        ai_response, conversation_history = think(
                            user_text, conversation_history, client
                        )
                        speak(ai_response)

                except KeyboardInterrupt:
                    print("\n\n👋 Ctrl+C pressed. Saving chat and exiting...")
                    save_chat_history(
                        conversation_history, client,
                        current_model_index, MODELS,
                        format_history_for_prompt
                    )
                    return

                except Exception as e:
                    print(f"❌ Error in inner loop: {e}")
                    continue

        except KeyboardInterrupt:
            print("\n\n👋 Ctrl+C pressed. Saving chat and exiting...")
            save_chat_history(
                conversation_history, client,
                current_model_index, MODELS,
                format_history_for_prompt
            )
            break

        except Exception as e:
            print(f"❌ Error in outer loop: {e}")
            break


if __name__ == "__main__":
    main()