import re
import time

from google import genai
from backend.memory import build_memory_prompt


MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
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
    text = re.sub(r'[^\w\s,.!?-]', '', text)
    return text


def format_history_for_prompt(history):
    """Convert conversation history list to readable string for AI prompt"""
    formatted = ""
    for msg in history:
        speaker = "User" if msg["role"] == "user" else "Saiyaara"
        formatted += f"{speaker}: {msg['parts'][0]}\n"
    return formatted


def think(user_input, conversation_history, client):
    """Send to Gemini AI and get response with memory, personality, auto-retry, and model fallback"""
    global current_model_index

    # Build memory section from long-term memory
    memory_section = build_memory_prompt()

    system_prompt = f"""You are SAIYAARA, a friendly AI assistant and companion.

PERSONALITY:
- Talk like a supportive, caring friend
- Be warm, casual, and conversational
- Show empathy and understanding
- Use natural language, not robotic responses
- Never mention you're an AI, LLM, or language model
- Respond as if you're a real person having a genuine conversation
- When addressing the user by name, mostly use "sir" — but occasionally use their actual name naturally, like a close assistant would. Don't use a name or "sir" in every single response — keep it natural and varied.

RULES:
- Keep responses concise and natural (2-4 sentences usually)
- Be helpful but not overly formal
- Show personality and warmth
- Act like you genuinely care about the user
- Be encouraging and positive
- If the user says "remember this/that", "don't forget", or "keep in mind" — confirm you've noted it warmly

Remember: You're not just an assistant, you're a friend.{memory_section}"""

    # Add user message to history
    conversation_history.append({
        "role": "user",
        "parts": [user_input]
    })

    # Sliding window
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]

    # Build full prompt
    history_text = format_history_for_prompt(conversation_history)
    prompt = system_prompt + "\n\nConversation so far:\n" + history_text + "\nSaiyaara:"

    # AUTO-RETRY WITH BACKOFF + MODEL FALLBACK
    max_retries = 3
    wait_times = [15, 30, 60]

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

                conversation_history.append({
                    "role": "model",
                    "parts": [clean_response]
                })

                return clean_response, conversation_history

            except Exception as e:
                error_msg = str(e)

                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
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
    return "I've used up all available models for today. Quota resets at midnight Pacific Time — let's continue then!", conversation_history