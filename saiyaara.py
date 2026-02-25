import speech_recognition as sr
from google import genai
import os
import re
import asyncio
import edge_tts
import pygame
from dotenv import load_dotenv
import tempfile
import time
import keyboard  # For detecting key press/release
import threading  # For handling async key detection
import json  # For saving chat history
from datetime import datetime  # For timestamps

# ===== LOAD ENVIRONMENT VARIABLES =====
load_dotenv()

# ===== CONFIGURATION =====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ ERROR: API key not found!")
    exit()

client = genai.Client(api_key=GEMINI_API_KEY)
recognizer = sr.Recognizer()

conversation_history = []  # Stores current conversation
MAX_HISTORY = 20  # Keep last 20 messages (10 exchanges)

# Fallback model chain — if one runs out of quota, auto-switch to next
MODELS = [
    "gemini-2.5-flash-lite",   # Primary    — 1000 req/day, 15 RPM
    "gemini-2.5-flash",        # Fallback 1 —  250 req/day, 10 RPM
    "gemini-2.5-pro",          # Fallback 2 —  100 req/day,  5 RPM
]
current_model_index = 0  # Start with flash-lite

print("✅ Using Edge-TTS (Microsoft Edge Text-to-Speech) - Most reliable on Windows!")

# ===== HELPER FUNCTIONS =====

def clean_text_for_speech(text):
    """Remove markdown and formatting symbols"""
    text = text.replace('**', '')
    text = re.sub(r'(?<!\*)\*(?!\*)', '', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = text.replace('_', '')
    text = text.replace('`', '')
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = ' '.join(text.split())
    # Remove emojis for speech
    text = re.sub(r'[^\w\s,.!?-]', '', text)
    return text

def format_history_for_prompt(history):
    """Convert conversation history list to a readable string for AI prompt"""
    formatted = ""
    for msg in history:
        speaker = "User" if msg["role"] == "user" else "Saiyaara"
        formatted += f"{speaker}: {msg['parts'][0]}\n"
    return formatted

# ===== INPUT FUNCTIONS =====

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

def listen():
    """Listen using push-to-talk (Press Space to start/stop)"""
    print("\n" + "=" * 60)
    print("🎤 PUSH-TO-TALK MODE")
    print("=" * 60)
    print("👉 Press SPACE BAR once to START speaking")
    print("👉 Press SPACE BAR again to STOP")
    print("=" * 60)

    with sr.Microphone() as source:
        # FIX 1: Calibrate BEFORE space press so no delay when you start speaking
        print("\n🔧 Calibrating microphone...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        recognizer.energy_threshold = 200
        recognizer.dynamic_energy_threshold = True
        print("✅ Ready!")

        print("\n⏸️  Press SPACE BAR to start recording...")
        keyboard.wait('space')
        time.sleep(0.1)

        print("\n🔴 RECORDING... Press SPACE BAR again when done!")
        print("=" * 60)
        print("🎙️  Speak now...")

        stop_recording = threading.Event()

        def wait_for_space():
            keyboard.wait('space')
            stop_recording.set()

        space_thread = threading.Thread(target=wait_for_space, daemon=True)
        space_thread.start()

        audio_data = []

        try:
            start_time = time.time()
            max_duration = 60

            while not stop_recording.is_set():
                if time.time() - start_time > max_duration:
                    print("\n⏱️  Maximum recording time reached (60    seconds)")
                    break

                try:
                    audio = recognizer.listen(source, timeout=0.5, phrase_time_limit=1)
                    audio_data.append(audio.get_raw_data())
                except sr.WaitTimeoutError:
                    continue

            # FIX 2: After stop pressed, wait a tiny bit to capture tail end of speech
            time.sleep(0.5)
            try:
                audio = recognizer.listen(source, timeout=0.5, phrase_time_limit=1)
                audio_data.append(audio.get_raw_data())
            except:
                pass  # If nothing left to capture, that's fine

            print("⏹️  Recording stopped!")

            if not audio_data:
                print("❌ No audio recorded. Please try again.")
                return None

            print("🔄 Processing speech...")

            sample_rate = source.SAMPLE_RATE
            sample_width = source.SAMPLE_WIDTH
            combined_data = b''.join(audio_data)
            audio_full = sr.AudioData(combined_data, sample_rate, sample_width)

            text = recognizer.recognize_google(audio_full)
            print(f"✅ You said: {text}\n")
            return text

        except sr.UnknownValueError:
            print("❌ Couldn't understand. Please speak more clearly.")
            return None

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

def type_input():
    """Get text input from keyboard"""
    # Show memory count for awareness
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

# ===== AI PROCESSING =====

def think(user_input):
    """Send to Gemini AI and get response with friendly personality, memory, auto-retry, and model fallback"""
    global conversation_history, current_model_index

    # System prompt
    system_prompt = """You are SAIYAARA, a friendly AI assistant and companion. 

PERSONALITY:
- Talk like a supportive, caring friend
- Be warm, casual, and conversational
- Show empathy and understanding
- Use natural language, not robotic responses
- Never mention you're an AI, LLM, or language model
- Respond as if you're a real person having a genuine conversation

RULES:
- Keep responses concise and natural (2-4 sentences usually)
- Be helpful but not overly formal
- Show personality and warmth
- Act like you genuinely care about the user
- Be encouraging and positive

Remember: You're not just an assistant, you're a friend."""

    # Add user message to history BEFORE sliding window trim
    conversation_history.append({
        "role": "user",
        "parts": [user_input]
    })

    # Implement sliding window AFTER adding user message
    if len(conversation_history) > MAX_HISTORY:
        conversation_history = conversation_history[-MAX_HISTORY:]

    # Format history as readable text string
    history_text = format_history_for_prompt(conversation_history)

    # Build full prompt
    prompt = system_prompt + "\n\nConversation so far:\n" + history_text + "\nSaiyaara:"

    # ===== AUTO-RETRY WITH BACKOFF + MODEL FALLBACK =====
    max_retries = 3
    wait_times = [15, 30, 60]  # seconds to wait before retrying same model

    # Try each model in the chain
    while current_model_index < len(MODELS):
        model_name = MODELS[current_model_index]

        for attempt in range(max_retries):
            try:
                label = f"🧠 Thinking (model: {model_name})..." if attempt == 0 else f"🧠 Retrying (attempt {attempt + 1}/{max_retries}, model: {model_name})..."
                print(label)

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                clean_response = clean_text_for_speech(response.text.strip())

                # Add assistant response to history only on success
                conversation_history.append({
                    "role": "model",
                    "parts": [clean_response]
                })

                return clean_response

            except Exception as e:
                error_msg = str(e)

                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    if attempt < max_retries - 1:
                        # Still have retries left for this model — wait and retry
                        wait = wait_times[attempt]
                        print(f"\n⚠️  Rate limit hit on {model_name}! Waiting {wait}s then retrying...")
                        for remaining in range(wait, 0, -5):
                            print(f"   ⏳ Retrying in {remaining}s...", end="\r")
                            time.sleep(5)
                        print(" " * 40, end="\r")
                    else:
                        # All retries for this model exhausted — switch to next model
                        current_model_index += 1
                        if current_model_index < len(MODELS):
                            next_model = MODELS[current_model_index]
                            print(f"\n🔄 {model_name} quota exhausted! Switching to {next_model}...")
                        break  # Break retry loop, outer while will try next model
                else:
                    print(f"⚠️ AI Error: {error_msg}")
                    conversation_history.pop()
                    return "Sorry, I'm having some trouble right now. Can you try again?"

    # All models exhausted
    print("⚠️  All models exhausted. Daily quota finished across all models.")
    conversation_history.pop()
    return "I've used up all available models for today. Quota resets at midnight Pacific Time — let's continue then!"

# ===== SPEECH OUTPUT =====

async def _generate_speech(text, output_file):
    """Generate speech using edge-tts"""
    voice = "en-US-JennyNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

def speak(text):
    """Display text word-by-word AND speak using Edge-TTS"""
    temp_filename = None
    sound = None

    try:
        # Generate speech first
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_filename = temp_file.name
        temp_file.close()

        asyncio.run(_generate_speech(text, temp_filename))

        # Initialize pygame mixer
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        pygame.mixer.music.load(temp_filename)

        # Load sound to get duration
        sound = pygame.mixer.Sound(temp_filename)
        audio_duration = sound.get_length()

        # Split text into words
        words = text.split()
        total_words = len(words)

        time_per_word = audio_duration / total_words if total_words > 0 else 0

        # Start playing audio
        pygame.mixer.music.play()

        print("\n" + "=" * 60)
        print("🤖 SAIYAARA:")

        # Word-wrap aware progressive printing
        # Each line stays within 56 chars so words never break mid-word
        LINE_WIDTH = 100 # Adjusted for wider console output #56
        current_line = ""
        start_time = time.time()

        for i, word in enumerate(words):
            target_time = start_time + (i * time_per_word)
            current_time = time.time()

            if current_time < target_time:
                time.sleep(target_time - current_time)

            # Check if adding this word exceeds line width
            test_line = current_line + word + " "
            if len(test_line) > LINE_WIDTH and current_line:
                # Print current line and start new one
                print(current_line)
                current_line = word + " "
            else:
                current_line = test_line
                # Print updated line in place (overwrite current line)
                print(f"  {current_line}", end="\r", flush=True)

        # Print final line
        if current_line:
            print(f"  {current_line}")
        print("=" * 60)

        # Wait for audio to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        # FIX: Properly stop and release sound before unlinking
        pygame.mixer.music.unload()
        sound.stop()
        sound = None
        time.sleep(0.1)  # Small buffer to release file handle on Windows
        os.unlink(temp_filename)

        print("✅ Speech complete!\n")

    except KeyboardInterrupt:
        print("\n⏹️ Speech stopped by user!")
        pygame.mixer.music.stop()
        if sound:
            sound.stop()
        if temp_filename and os.path.exists(temp_filename):
            try:
                time.sleep(0.1)
                os.unlink(temp_filename)
            except:
                pass

    except Exception as e:
        print(f"❌ Speech error: {e}")
        if sound:
            sound.stop()
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.unlink(temp_filename)
            except:
                pass

# ===== CHAT HISTORY =====

def generate_fallback_title():
    """Generate a timestamp-based title when AI title generation fails"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return f"chat_{timestamp}"

def save_chat_history():
    """Save conversation to JSON file with AI-generated title (fallback to timestamp if quota exceeded)"""
    global conversation_history

    if len(conversation_history) == 0:
        print("💭 No conversation to save.")
        return

    try:
        print("\n💾 Saving conversation...")

        chat_title = None

        # Try AI title generation
        try:
            # FIX: Format history as readable text, not raw Python list
            history_preview = format_history_for_prompt(conversation_history[:6])

            title_prompt = f"""Based on this conversation, create a SHORT 3-4 word title (like "Cooking Tips Chat" or "Python Help Session"). 

Conversation:
{history_preview}

Reply with ONLY the title, nothing else."""

            title_response = client.models.generate_content(
                model=MODELS[min(current_model_index, len(MODELS) - 1)],
                contents=title_prompt
            )

            raw_title = title_response.text.strip()
            chat_title = raw_title.replace(' ', '-').lower()
            chat_title = re.sub(r'[^\w-]', '', chat_title)  # Remove special chars
            print(f"🏷️  Chat title: {raw_title}")

        except Exception as title_error:
            # FIX: If quota exceeded or any error, fall back to timestamp title
            print(f"⚠️  Could not generate AI title ({type(title_error).__name__}). Using timestamp instead.")
            chat_title = generate_fallback_title()

        # Create filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = f"{chat_title}_{timestamp}.json" if not chat_title.startswith("chat_") else f"{chat_title}.json"

        # Create chat_history folder if it doesn't exist
        chat_dir = "chat_history"
        if not os.path.exists(chat_dir):
            os.makedirs(chat_dir)

        # Prepare chat data
        readable_title = chat_title.replace('-', ' ').title()
        chat_data = {
            "title": readable_title,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "messages": []
        }

        # Convert conversation history to paired exchange format
        # Group messages into {me: ..., saiyaara: ...} pairs
        messages = conversation_history
        i = 0
        while i < len(messages):
            pair = {}
            if i < len(messages) and messages[i]["role"] == "user":
                pair["me"] = messages[i]["parts"][0]
                i += 1
            if i < len(messages) and messages[i]["role"] == "model":
                pair["saiyaara"] = messages[i]["parts"][0]
                i += 1
            if pair:
                chat_data["messages"].append(pair)

        # Save to file
        filepath = os.path.join(chat_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Chat saved as: {readable_title}")
        print(f"📁 Location: {filepath}\n")

    except Exception as e:
        print(f"⚠️ Error saving chat: {e}\n")

# ===== MAIN LOOP =====

def main():
    print("\n" + "=" * 60)
    print("🤖 SAIYAARA - Your Personal AI Assistant")
    print("=" * 60)
    print("✅ Status: Active and Ready!")
    print("💡 Say/Type: 'quit', 'bye', 'goodbye', 'stop' to switch modes")
    print("💡 Press Ctrl+C to exit completely (chat will be saved!)")
    print("=" * 60)

    while True:
        try:
            choice = get_input_choice()

            print(f"\n✅ {'Voice' if choice == '1' else 'Text'} mode activated!")
            if choice == '1':
                print(f"💡 Press SPACE BAR to start/stop recording")
                print(f"💡 Say 'bye', 'goodbye', 'quit', or 'stop' to switch modes")
            else:
                print(f"💡 Type 'quit', 'bye', 'goodbye', or 'exit' to switch modes")
            print(f"💡 Currently in: {'🎤 VOICE MODE' if choice == '1' else '⌨️ TEXT MODE'}\n")

            failed_attempts = 0
            max_failed_attempts = 3

            while True:
                try:
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

                    # Exit check
                    exit_words = ['quit', 'exit', 'bye', 'goodbye', 'stop', 'close', 'end']
                    user_lower = user_text.lower().strip()

                    should_quit = False

                    if choice == '1':  # Voice mode: fuzzy matching
                        for word in exit_words:
                            if word in user_lower:
                                should_quit = True
                                break
                    else:  # Text mode: exact or word match
                        for word in exit_words:
                            if user_lower == word or word in user_lower.split():
                                should_quit = True
                                break

                    if should_quit:
                        speak("Nice talking to you. Take care! Switching modes.")
                        save_chat_history()       # Save before clearing
                        conversation_history.clear()  # Clear memory for next session
                        print("\n🔄 Returning to input selection...\n")
                        break

                    ai_response = think(user_text)
                    speak(ai_response)

                except KeyboardInterrupt:
                    print("\n\n👋 Ctrl+C pressed. Saving chat and exiting...")
                    # FIX: Save chat on Ctrl+C too, so nothing is lost
                    save_chat_history()
                    return

                except Exception as e:
                    print(f"❌ Error in inner loop: {e}")
                    continue

        except KeyboardInterrupt:
            print("\n\n👋 Ctrl+C pressed. Saving chat and exiting...")
            # FIX: Save chat on outer Ctrl+C too
            save_chat_history()
            break

        except Exception as e:
            print(f"❌ Error in outer loop: {e}")
            break

if __name__ == "__main__":
    main()