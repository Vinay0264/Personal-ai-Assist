import threading
from ddgs import DDGS
from groq import Groq


# ===== SEARCH ENGINE =====

def web_search(query, max_results=5):
    """
    Search the web using DuckDuckGo (free, no API key needed).
    Returns a list of result dicts: {title, href, body}
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        print(f"⚠️ Search error: {e}")
        return []


def format_search_results(results):
    """
    Format raw search results into a clean context string for Groq.
    """
    if not results:
        return "No search results found."

    formatted = ""
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body", "")
        formatted += f"[Result {i}] {title}\n{body}\n\n"

    return formatted.strip()


def realtime_search(query, groq_client, clean_text_fn, speak_fn, slow_display_fn,
                    start_tts_fn, play_pregenerated_fn):
    """
    Full real-time search pipeline:
    1. Search DuckDuckGo for live results
    2. Feed results + query to Groq for summarization
    3. Deliver answer using SAIYAARA's concurrent speech+display system

    Parameters:
        query           — the user's original query (after "realtime " prefix stripped)
        groq_client     — active Groq client instance
        clean_text_fn   — brain.clean_text_for_speech()
        speak_fn        — tts.speak()
        slow_display_fn — brain.slow_display()
        start_tts_fn    — tts.start_tts_generation()
        play_pregenerated_fn — tts.play_pregenerated()
    """

    print(f"\n🌐 Searching the web for: \"{query}\"")

    # ── STEP 1: Search ──
    results = web_search(query)

    if not results:
        message = "I searched the web but couldn't find anything useful right now. Try again in a moment?"
        speak_fn(message, display=True)
        return message

    # ── STEP 2: Format results as context ──
    context = format_search_results(results)

    # ── STEP 3: Ask Groq to summarize ──
    prompt = f"""You are SAIYAARA, a friendly AI assistant. The user asked: "{query}"

Here are live web search results:

{context}

Based ONLY on the search results above, give a clear, conversational answer in 2-4 sentences.
- Use natural language, like you're talking to a friend
- Include specific facts, numbers, or names from the results
- If results are unclear or conflicting, say so honestly
- Do NOT say "according to the search results" — just answer naturally
- Address the user as "sir"
- Keep it concise"""

    try:
        print("🧠 Summarizing results...")

        stream = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4,
            stream=True,
        )

        full_response = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                full_response += token

        clean_response = clean_text_fn(full_response)

        # ── STEP 4: Concurrent speech + display (same as brain.py) ──
        tts_file, tts_ready = start_tts_fn(clean_response)
        tts_ready.wait(timeout=10)

        print("\n" + "=" * 60)
        print("🤖 SAIYAARA:")

        audio_thread = threading.Thread(
            target=play_pregenerated_fn,
            args=(tts_file, tts_ready),
            daemon=True
        )
        audio_thread.start()

        slow_display_fn(clean_response)

        print("=" * 60)
        audio_thread.join()

        return clean_response

    except Exception as e:
        print(f"⚠️ Groq summarization error: {e}")
        message = "I found some results but had trouble summarizing them. Please try again."
        speak_fn(message, display=True)
        return message


# ===== TEST =====
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv()

    # Minimal stubs for testing without full SAIYAARA stack
    def fake_clean(text):
        return text

    def fake_speak(text, display=True):
        print(f"[SPEAK] {text}")

    def fake_slow_display(text, **kwargs):
        print(f"  {text}")

    def fake_start_tts(text):
        import threading
        e = threading.Event()
        e.set()
        return None, e

    def fake_play(file, event):
        pass

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print("🌐 Real-Time Search Test — type a query, Ctrl+C to quit.\n")
    while True:
        try:
            query = input("Query: ").strip()
            if not query:
                continue
            realtime_search(
                query, client,
                fake_clean, fake_speak, fake_slow_display,
                fake_start_tts, fake_play
            )
            print()
        except KeyboardInterrupt:
            print("\n👋 Test ended.")
            break