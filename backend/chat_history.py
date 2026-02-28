import os
import re
import json
from datetime import datetime


CHAT_DIR = "data/chat_history"
MAX_HISTORY = 20

# Phrases that trigger showing chat history
CHAT_HISTORY_TRIGGERS = ["show my chats", "previous chats"]


def load_recent_chats(limit=5):
    """Load the most recent saved chat files, sorted by date (newest first)"""
    if not os.path.exists(CHAT_DIR):
        return []

    files = []
    for fname in os.listdir(CHAT_DIR):
        if fname.endswith('.json'):
            fpath = os.path.join(CHAT_DIR, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                files.append({
                    "path": fpath,
                    "title": data.get("title", fname),
                    "date": data.get("date", "Unknown date"),
                    "messages": data.get("messages", []),
                    "modified": os.path.getmtime(fpath)
                })
            except Exception:
                continue

    # Sort by most recently saved first
    files.sort(key=lambda x: x["date"], reverse=True)
    return files[:limit]


def show_recent_chats_on_demand(conversation_history, speak_fn):
    """
    Called when user says 'show my chats' or 'previous chats'.
    Displays last 5 saved chats and asks if they want to continue one.
    Returns loaded conversation_history if user picks a chat, else None.
    """
    recent = load_recent_chats(limit=5)

    if not recent:
        print("💭 No previous conversations found yet.\n")
        speak_fn("You don't have any saved conversations yet.")
        return None

    print("\n" + "=" * 60)
    print("📂 RECENT CONVERSATIONS")
    print("=" * 60)
    for i, chat in enumerate(recent, 1):
        print(f"  {i}. {chat['title']} — {chat['date']}")
    print("=" * 60)
    print("  N. Keep current conversation")
    print("=" * 60)

    while True:
        choice = input("Continue a chat? Enter number (1-5) or N to cancel: ").strip().lower()

        if choice == 'n' or choice == '':
            print("\n✅ Keeping current conversation.\n")
            return None

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(recent):
                selected = recent[idx]
                print(f"\n✅ Switching to: {selected['title']} ({selected['date']})\n")

                # Convert saved format back to conversation_history format
                loaded_history = []
                for pair in selected["messages"]:
                    if "me" in pair:
                        loaded_history.append({"role": "user", "parts": [pair["me"]]})
                    if "saiyaara" in pair:
                        loaded_history.append({"role": "model", "parts": [pair["saiyaara"]]})

                # Apply sliding window
                if len(loaded_history) > MAX_HISTORY:
                    loaded_history = loaded_history[-MAX_HISTORY:]

                # Show recap of last exchange
                print("=" * 60)
                print("📜 Last exchange from this chat:")
                print("=" * 60)
                recap_pairs = selected["messages"][-1:]
                for pair in recap_pairs:
                    if "me" in pair:
                        print(f"  You:      {pair['me'][:100]}{'...' if len(pair['me']) > 100 else ''}")
                    if "saiyaara" in pair:
                        print(f"  SAIYAARA: {pair['saiyaara'][:100]}{'...' if len(pair['saiyaara']) > 100 else ''}")
                print("=" * 60)
                print(f"🧠 Loaded {len(loaded_history)} messages into memory.\n")

                speak_fn(f"Okay! I've loaded the chat titled {selected['title']}. Let's continue from where we left off.")
                return loaded_history

        print("❌ Invalid choice. Enter a number from the list or N.")


def generate_fallback_title(conversation_history):
    """Generate a meaningful title from conversation keywords"""
    stopwords = {
        "i", "me", "my", "we", "you", "your", "he", "she", "it", "they", "them",
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "can", "may", "might", "shall", "must", "am",
        "and", "or", "but", "so", "if", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "into", "through",
        "what", "how", "why", "when", "where", "who", "which", "that",
        "this", "these", "those", "there", "here", "not", "no", "yes",
        "just", "like", "also", "then", "than", "more", "some", "any",
        "hey", "hi", "hello", "okay", "ok", "please", "thanks", "thank",
        "tell", "know", "think", "want", "need", "get", "got", "go", "going",
        "saiyaara", "user"
    }

    user_messages = [
        msg["parts"][0]
        for msg in conversation_history[:6]
        if msg["role"] == "user"
    ][:3]

    keyword_counts = {}
    for msg in user_messages:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', msg.lower())
        for word in words:
            if word not in stopwords:
                keyword_counts[word] = keyword_counts.get(word, 0) + 1

    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: (-x[1], x[0]))
    top_words = [w.capitalize() for w, _ in sorted_keywords[:3]]

    if top_words:
        return "-".join(top_words).lower()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return f"chat_{timestamp}"


def save_chat_history(conversation_history, format_history_fn, gemini_client=None):
    """Save conversation to JSON file with AI-generated title (Gemini) or keyword fallback"""
    if len(conversation_history) == 0:
        print("💭 No conversation to save.")
        return

    try:
        print("\n💾 Saving conversation...")

        chat_title = None

        # ── TRY GEMINI TITLE GENERATION (1 call per session — very low quota usage) ──
        if gemini_client:
            try:
                history_preview = format_history_fn(conversation_history[:6])

                title_prompt = f"""Based on this conversation, create a SHORT 3-4 word title (like "Cooking Tips Chat" or "Python Help Session").

Conversation:
{history_preview}

Reply with ONLY the title, nothing else."""

                title_response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=title_prompt
                )

                raw_title = title_response.text.strip()
                chat_title = raw_title.replace(' ', '-').lower()
                chat_title = re.sub(r'[^\w-]', '', chat_title)
                print(f"🏷️  Chat title (AI): {raw_title}")

            except Exception:
                print("⚠️  AI title failed. Using keyword title instead.")
                chat_title = generate_fallback_title(conversation_history)
        else:
            # ── KEYWORD FALLBACK (no API needed) ──
            chat_title = generate_fallback_title(conversation_history)
            print(f"🏷️  Chat title (keywords): {chat_title}")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = f"{chat_title}_{timestamp}.json" if not chat_title.startswith("chat_") else f"{chat_title}.json"

        os.makedirs(CHAT_DIR, exist_ok=True)

        readable_title = chat_title.replace('-', ' ').title()
        chat_data = {
            "title": readable_title,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "messages": []
        }

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

        filepath = os.path.join(CHAT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Chat saved as: {readable_title}")
        print(f"📁 Location: {filepath}\n")

    except Exception as e:
        print(f"⚠️ Error saving chat: {e}\n")