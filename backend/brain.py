import re
import time

from groq import Groq
from backend.memory import build_memory_prompt


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
    text = re.sub(r'[^\w\s,.!?-]', '', text)
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


def think(user_input, conversation_history, client):
    """Send to Groq AI and get response with memory, personality, auto-retry, and model fallback"""
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
- ALWAYS address the user as "sir" — never use their actual name in responses
- The user's name exists in your memory only so you know WHO you're talking to — never say it out loud
- Every single response must use "sir" if addressing the user directly, no exceptions
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

    # AUTO-RETRY WITH BACKOFF + MODEL FALLBACK
    max_retries = 3
    wait_times = [10, 20, 40]

    while current_model_index < len(MODELS):
        model_name = MODELS[current_model_index]

        for attempt in range(max_retries):
            try:
                label = f"🧠 Thinking (model: {model_name})..." if attempt == 0 else f"🧠 Retrying (attempt {attempt + 1}/{max_retries}, model: {model_name})..."
                print(label)

                # Build Groq messages format
                messages = build_groq_messages(conversation_history, system_prompt)

                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.8,
                )

                raw_response = response.choices[0].message.content.strip()
                clean_response = clean_text_for_speech(raw_response)

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