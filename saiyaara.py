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

# ===== LOAD ENVIRONMENT VARIABLES =====
load_dotenv()

# ===== CONFIGURATION =====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ ERROR: API key not found!")
    exit()

client = genai.Client(api_key=GEMINI_API_KEY)
recognizer = sr.Recognizer()

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
    
    print("\n⏸️  Press SPACE BAR to start recording...")
    keyboard.wait('space')
    time.sleep(0.1)
    
    with sr.Microphone() as source:
        print("\n🔴 RECORDING... Press SPACE BAR again when done!")
        print("=" * 60)
        
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        recognizer.energy_threshold = 200
        recognizer.dynamic_energy_threshold = True
        
        stop_recording = threading.Event()
        
        def wait_for_space():
            keyboard.wait('space')
            stop_recording.set()
        
        space_thread = threading.Thread(target=wait_for_space, daemon=True)
        space_thread.start()
        
        audio_data = []
        
        try:
            print("🎙️  Speak now...")
            start_time = time.time()
            max_duration = 30
            
            while not stop_recording.is_set():
                if time.time() - start_time > max_duration:
                    print("\n⏱️  Maximum recording time reached (30 seconds)")
                    break
                
                try:
                    audio = recognizer.listen(source, timeout=0.5, phrase_time_limit=1)
                    audio_data.append(audio.get_raw_data())
                except sr.WaitTimeoutError:
                    continue
            
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
    print("\n⌨️  Type your message:")
    text = input("You: ").strip()
    if text:
        return text
    else:
        print("❌ Empty input. Please type something.")
        return None

# ===== AI PROCESSING =====

def think(user_input):
    """Send to Gemini AI and get response with friendly personality"""
    try:
        print("🧠 Thinking...")
        
        # System prompt to make SAIYAARA talk like a friend
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

        # Combine system prompt with user input
        full_prompt = f"{system_prompt}\n\nUser: {user_input}\n\nSAIYAARA:"
        
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=full_prompt
        )
        clean_response = clean_text_for_speech(response.text)
        return clean_response
        
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ AI Error: {error_msg}")
        
        # Check if it's a rate limit error
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return "Oops! I've hit my daily request limit. Can we try again in about a minute? Sorry about that!"
        else:
            return "Sorry, I'm having trouble right now."

async def _generate_speech(text, output_file):
    """Generate speech using edge-tts"""
    voice = "en-US-JennyNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

def speak(text):
    """Display text word-by-word AND speak using Edge-TTS"""
    temp_filename = None
    
    try:
        # Generate speech first
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_filename = temp_file.name
        temp_file.close()
        
        asyncio.run(_generate_speech(text, temp_filename))
        
        # Initialize pygame mixer
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        # Load the audio to get duration
        pygame.mixer.music.load(temp_filename)
        
        # Get audio duration using pygame
        sound = pygame.mixer.Sound(temp_filename)
        audio_duration = sound.get_length()
        
        # Split text into words
        words = text.split()
        total_words = len(words)
        
        # Calculate time per word
        time_per_word = audio_duration / total_words if total_words > 0 else 0
        
        # Start playing audio
        pygame.mixer.music.play()
        
        # Display header
        print("\n" + "=" * 60)
        print("🤖 SAIYAARA:", end=" ", flush=True)
        
        # Display words progressively while audio plays
        start_time = time.time()
        
        for i, word in enumerate(words):
            # Wait until it's time to show this word
            target_time = start_time + (i * time_per_word)
            current_time = time.time()
            
            if current_time < target_time:
                time.sleep(target_time - current_time)
            
            # Print word
            print(word, end=" ", flush=True)
        
        print()  # New line after all words
        print("=" * 60)
        
        # Wait for audio to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        # Clean up
        pygame.mixer.music.unload()
        sound = None
        os.unlink(temp_filename)
        
        print("✅ Speech complete!\n")
        
    except KeyboardInterrupt:
        print("\n⏹️ Speech stopped by user!")
        pygame.mixer.music.stop()
        if temp_filename and os.path.exists(temp_filename):
            os.unlink(temp_filename)
    except Exception as e:
        print(f"❌ Speech error: {e}")
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.unlink(temp_filename)
            except:
                pass

# ===== MAIN LOOP =====

def main():
    print("\n" + "=" * 60)
    print("🤖 SAIYAARA - Your Personal AI Assistant")
    print("=" * 60)
    print("✅ Status: Active and Ready!")
    print("💡 Say/Type: 'quit', 'bye', 'goodbye', 'stop' to change mode")
    print("💡 Press Ctrl+C to exit completely")
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
                        speak("Nice Talking to you. Take Care.... Switching modes!")
                        print("\n🔄 Returning to input selection...\n")
                        break
                    
                    ai_response = think(user_text)
                    speak(ai_response)
                
                except KeyboardInterrupt:
                    print("\n\n👋 Ctrl+C pressed. Exiting completely...")
                    return
                except Exception as e:
                    print(f"❌ Error in inner loop: {e}")
                    continue
        
        except KeyboardInterrupt:
            print("\n\n👋 Ctrl+C pressed. Exiting...")
            break
        except Exception as e:
            print(f"❌ Error in outer loop: {e}")
            break

if __name__ == "__main__":
    main()