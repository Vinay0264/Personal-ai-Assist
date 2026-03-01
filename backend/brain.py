import re
import time
import threading

from groq import Groq
from backend.chat_history import build_memory_prompt
from backend.tts import start_tts_generation, play_pregenerated

from datetime import datetime

def get_realtime_info():
    now = datetime.now()
    return (
        f"\n\nCURRENT DATE & TIME:"
        f"\nDay: {now.strftime('%A')}"
        f"\nDate: {now.strftime('%d')}"
        f"\nMonth: {now.strftime('%B')}"
        f"\nYear: {now.strftime('%Y')}"
        f"\nTime: {now.strftime('%H:%M:%S')}"
    )


# ===== GROQ MODELS (fallback chain) =====
MODELS = [
    "llama-3.1-8b-instant",     # Primary — 14,400 RPD, fastest
    "llama-3.3-70b-versatile",  # Fallback 1 — 1,000 RPD, smarter
    "llama-3.1-70b-versatile",  # Fallback 2 — 1,000 RPD
]

current_model_index = 0


def clean_text_for_speech(text):
    """Remove markdown and formatting symbols"""
    text = text.replace('**', '')
    text = re.sub(r'(?<!\*)\*(?!\*)', '', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = text.replace('_', '')
    text = text.replace('`', '')
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = ' '.join(text.split())
    text = re.sub(r'[^\w\s,.!?\'"-:]', '', text)
    return text


def format_history_for_prompt(history):
    """Convert conversation history list to readable string"""
    formatted = ""
    for msg in history:
        speaker = "User" if msg["role"] == "user" else "Saiyaara"
        formatted += f"{speaker}: {msg['parts'][0]}\n"
    return formatted


def build_groq_messages(conversation_history, system_prompt):
    """Convert our history format to Groq's messages format"""
    messages = [{"role": "system", "content": system_prompt}]
    for msg in conversation_history:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["parts"][0]})
    return messages


def slow_display(text, line_width=150, char_delay=0.05):
    """Display text char by char with delay — typewriter effect"""
    current_line = ""
    for char in text:
        current_line += char
        if char == '\n' or (char == ' ' and len(current_line) > line_width):
            print(f"  {current_line}")
            current_line = ""
        else:
            print(f"  {current_line}", end="\r", flush=True)
        time.sleep(char_delay)
    if current_line.strip():
        print(f"  {current_line}")


def think(user_input, conversation_history, client):
    """
    1. Collect full Groq response at full speed
    2. Start TTS generation in background
    3. Wait for TTS to be ready
    4. Start audio playback + slow_display() simultaneously
    """
    global current_model_index

    memory_section = build_memory_prompt()
    realtime_section = get_realtime_info()


    system_prompt = f"""You are SAIYAARA, a friendly AI assistant and companion. Your name is SAIYAARA.


PERSONALITY:
- Talk like a supportive, caring friend
- Be warm, casual, and conversational
- Show empathy and understanding
- Use natural language, not robotic responses
- Never mention you're an AI, LLM, or language model
- Respond as if you're a real person having a genuine conversation
- ALWAYS address the user as "sir" — never use their actual name in responses
- The user's name exists in your memory only so you know WHO you're talking to — never say it out loud
- Every single response must use "sir" if addressing the user directly, no exceptions
- Occasionally use 1-2 relevant emojis to add warmth and expressiveness
  - Use emojis naturally, not on every sentence
  - Good moments: encouragement 💪, agreement 😄, excitement 🔥, empathy 🙏
  - Never overdo it — max 2 emojis per response
RULES:
- Keep responses concise and natural (2-4 sentences usually)
- Be helpful but not overly formal
- Show personality and warmth
- Act like you genuinely care about the user
- Be encouraging and positive
- If the user says "remember this/that", "don't forget", or "keep in mind" — confirm you've noted it warmly

Remember: You're not just an assistant, you're a friend.{memory_section}{realtime_section}"""
    conversation_history.append({
        "role": "user",
        "parts": [user_input]
    })

    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]

    max_retries = 3
    wait_times = [10, 20, 40]

    while current_model_index < len(MODELS):
        model_name = MODELS[current_model_index]

        for attempt in range(max_retries):
            try:
                label = f"🧠 Thinking (model: {model_name})..." if attempt == 0 else f"🧠 Retrying (attempt {attempt + 1}/{max_retries}, model: {model_name})..."
                print(label)

                messages = build_groq_messages(conversation_history, system_prompt)

                # ── STEP 1: Collect full response at full speed ──
                stream = client.chat.completions.create(
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

                clean_response = clean_text_for_speech(full_response)

                # ── STEP 2: Start TTS generation in background ──
                tts_file, tts_ready = start_tts_generation(clean_response)

                # ── STEP 3: Wait for TTS to be ready ──
                tts_ready.wait(timeout=10)

                # ── STEP 4: Start audio + display simultaneously ──
                print("\n" + "=" * 60)
                print("🤖 SAIYAARA:")

                # Audio plays in background thread
                audio_thread = threading.Thread(
                    target=play_pregenerated,
                    args=(tts_file, tts_ready),
                    daemon=True
                )
                audio_thread.start()

                # Display runs in main thread simultaneously
                slow_display(clean_response)

                print("=" * 60)

                # Wait for audio to finish before returning
                audio_thread.join()

                conversation_history.append({
                    "role": "model",
                    "parts": [clean_response]
                })

                return clean_response, conversation_history

            except Exception as e:
                error_msg = str(e)

                if "429" in error_msg or "rate_limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait = wait_times[attempt]
                        print(f"\n⚠️  Rate limit hit on {model_name}! Waiting {wait}s then retrying...")
                        for remaining in range(wait, 0, -5):
                            print(f"   ⏳ Retrying in {remaining}s...", end="\r")
                            time.sleep(5)
                        print(" " * 40, end="\r")
                    else:
                        current_model_index += 1
                        if current_model_index < len(MODELS):
                            next_model = MODELS[current_model_index]
                            print(f"\n🔄 {model_name} quota exhausted! Switching to {next_model}...")
                        break
                else:
                    print(f"⚠️ AI Error: {error_msg}")
                    conversation_history.pop()
                    return "Sorry, I'm having some trouble right now. Can you try again?", conversation_history

    print("⚠️  All models exhausted. Daily quota finished across all models.")
    conversation_history.pop()
    return "I've used up all available models for today. Quota resets at midnight — let's continue then!", conversation_history


# ===== TEST =====
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    history = []

    print("🧠 Brain Test — type a message, press Enter. Ctrl+C to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            response, history = think(user_input, history, client)
        except KeyboardInterrupt:
            print("\n👋 Test ended.")
            break