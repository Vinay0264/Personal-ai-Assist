import os
import re
import json
from google import genai


MEMORY_FILE = "data/memory.json"

# Phrases that trigger memory saving
REMEMBER_TRIGGERS = ["remember this", "remember that", "don't forget", "dont forget", "keep in mind"]


def load_long_term_memory():
    """Load saved facts from memory.json"""
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("facts", [])
    except Exception:
        return []


def save_long_term_memory(facts):
    """Save facts list to memory.json"""
    try:
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({"facts": facts}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Could not save memory: {e}")


def build_memory_prompt():
    """Build a string of long-term memory facts to inject into system prompt"""
    facts = load_long_term_memory()
    if not facts:
        return ""
    facts_text = "\n".join(f"- {f}" for f in facts)
    return f"\n\nTHINGS YOU KNOW ABOUT THE USER (long-term memory):\n{facts_text}"


def get_known_name():
    """Get the user's name from existing memory if available"""
    existing_facts = load_long_term_memory()
    for f in existing_facts:
        name_match = re.search(r"master'?s?\s+name\s+is\s+(\w+)", f.lower())
        if name_match:
            return name_match.group(1).capitalize()
    return None


def convert_to_third_person(fact_raw, gemini_client, clean_text_fn):
    """
    Use Gemini ONLY to convert first-person fact to third-person.
    e.g. "my name is Vinay" → "Master's name is Vinay"
    e.g. "I upgraded you from Gemini to Groq" → "Vinay upgraded SAIYAARA from Gemini to Groq"
    """
    known_name = get_known_name()
    reference = known_name if known_name else "Master"

    try:
        prompt = f"""Convert this first-person statement to third-person. Refer to the person as "{reference}".

Statement: "{fact_raw}"

Rules:
- Just convert first-person words (I, my, me, we) to third-person using "{reference}"
- Keep everything else exactly the same
- One sentence only
- Reply with ONLY the converted sentence, nothing else"""

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return clean_text_fn(response.text.strip())

    except Exception as e:
        print(f"⚠️ Gemini conversion failed: {e}")
        # Simple fallback — just prepend reference
        return f"{reference}: {fact_raw[:80]}"


def add_to_memory(user_input, conversation_history, gemini_client, clean_text_fn):
    """
    Called when user says 'remember this / remember that / don't forget / keep in mind'.

    Two cases:
    1. Inline fact:  "remember that I upgraded you to Groq" → convert to third person and save
    2. No fact:      "remember that" → look back at last SAIYAARA response, convert and save
    """
    user_lower = user_input.lower().strip()

    triggered = None
    for trigger in REMEMBER_TRIGGERS:
        if trigger in user_lower:
            triggered = trigger
            break

    if not triggered:
        return None

    # Extract everything after the trigger phrase
    idx = user_lower.find(triggered)
    fact_raw = user_input[idx + len(triggered):].strip()
    fact_raw = re.sub(r'^[\s\-–—:,]+', '', fact_raw).strip()

    if fact_raw and len(fact_raw.split()) > 2:
        # ── CASE 1: Inline fact ──
        # User said "remember that I upgraded you from Gemini to Groq"
        print(f"\n🧠 Remembering: \"{fact_raw[:60]}\"")
        fact_to_save = convert_to_third_person(fact_raw, gemini_client, clean_text_fn)

    else:
        # ── CASE 2: Look-back ──
        # User said just "remember that" — referring to what SAIYAARA just said
        saiyaara_responses = [
            msg["parts"][0]
            for msg in conversation_history
            if msg["role"] == "model"
        ]

        if not saiyaara_responses:
            print("⚠️ Nothing in conversation to remember yet.")
            return None

        last_response = saiyaara_responses[-1]
        preview = last_response[:60] + "..." if len(last_response) > 60 else last_response
        print(f"\n🧠 Remembering from last exchange: \"{preview}\"")

        # Last SAIYAARA response is already third-person friendly — save directly
        fact_to_save = clean_text_fn(last_response)

    if not fact_to_save:
        print("⚠️ Could not extract a fact. Memory not saved.")
        return None

    # Avoid duplicates
    existing_facts = load_long_term_memory()
    if fact_to_save.lower() not in [f.lower() for f in existing_facts]:
        existing_facts.append(fact_to_save)
        save_long_term_memory(existing_facts)
        print(f"\n🧠 Saved to long-term memory: \"{fact_to_save}\"")
        return fact_to_save
    else:
        print(f"\n🧠 Already in memory: \"{fact_to_save}\"")
        return fact_to_save