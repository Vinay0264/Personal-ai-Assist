import os
import re
import json


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


def extract_fact_with_ai(message, client, current_model_index, models, clean_text_fn):
    """
    Use Gemini to extract a clean, properly phrased fact from a message.
    First fact uses "Master's name is X" style.
    Later facts use the person's actual name once known.
    """
    existing_facts = load_long_term_memory()
    known_name = None
    for f in existing_facts:
        name_match = re.search(r"master'?s?\s+name\s+is\s+(\w+)", f.lower())
        if name_match:
            known_name = name_match.group(1).capitalize()
            break

    if not existing_facts or not known_name:
        reference = "Master"
        example = "Master's name is Vinay"
    else:
        reference = known_name
        example = f"{known_name} is 21 years old"

    try:
        extract_prompt = f"""Extract the key fact from this message and phrase it as a short memory note.

Message: "{message}"

Rules:
- Refer to the person as "{reference}" (e.g. "{example}")
- Keep it short — one sentence max
- No extra explanation, just the fact
- Reply with ONLY the fact sentence, nothing else"""

        response = client.models.generate_content(
            model=models[min(current_model_index, len(models) - 1)],
            contents=extract_prompt
        )
        return clean_text_fn(response.text.strip())

    except Exception:
        if known_name:
            return f"{known_name}: {message[:80]}"
        return f"Master: {message[:80]}"


def add_to_memory(user_input, conversation_history, client, current_model_index, models, clean_text_fn):
    """
    Called when user says 'remember this / remember that / don't forget / keep in mind'.

    Two cases:
    1. Fact inline:  "remember that my name is Vinay" -> extracts and saves
    2. No fact:      "remember that" -> looks back at last conversation message
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

    existing_facts = load_long_term_memory()
    is_first_fact = len(existing_facts) == 0

    if fact_raw:
        fact_to_save = extract_fact_with_ai(fact_raw, client, current_model_index, models, clean_text_fn)
    else:
        # Look back at last user message in conversation
        prior_messages = [
            msg["parts"][0]
            for msg in conversation_history
            if msg["role"] == "user"
            and not any(t in msg["parts"][0].lower() for t in REMEMBER_TRIGGERS)
        ]

        if not prior_messages:
            print("⚠️ Nothing in conversation to remember yet.")
            return None

        last_message = prior_messages[-1]
        preview = last_message[:60] + "..." if len(last_message) > 60 else last_message
        print(f"\n🧠 Remembering from last message: \"{preview}\"")
        fact_to_save = extract_fact_with_ai(last_message, client, current_model_index, models, clean_text_fn)

    if not fact_to_save:
        print("⚠️ Could not extract a fact. Memory not saved.")
        return None

    # Avoid duplicates
    if fact_to_save.lower() not in [f.lower() for f in existing_facts]:
        existing_facts.append(fact_to_save)
        save_long_term_memory(existing_facts)
        print(f"\n🧠 Saved to long-term memory: \"{fact_to_save}\"")
        return fact_to_save
    else:
        print(f"\n🧠 Already in memory: \"{fact_to_save}\"")
        return fact_to_save