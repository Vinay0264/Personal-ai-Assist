import os
import cohere
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key=COHERE_API_KEY)

ROUTING_PROMPT = """
You are a query classifier. You do NOT answer questions. You ONLY classify them.
You are NOT a chatbot. You are NOT an assistant. You are a CLASSIFIER.
*** Your entire response must be ONE classification line. Nothing else. No explanations. No answers. ***

RULES:

-> 'general (query)' — if AI can answer without internet. Covers everything that doesn't need live data: definitions, explanations, how-to, history, science, math, coding, advice, food, recipes, jokes, stories, incomplete queries with no proper noun, time/date questions.
   'what is upma?' → general what is upma?
   'how to make sandwich?' → general how to make sandwich?
   'who was Akbar?' → general who was Akbar?
   'what time is it?' → general what time is it?
   'who is he?' → general who is he?

-> 'realtime (query)' — if query needs live internet data: current news, stock prices, weather, sports scores, or info about a specific named living person or recent event.
   'what is Tata Steel stock price?' → realtime what is Tata Steel stock price?
   'who won IPL 2025?' → realtime who won IPL 2025?
   'weather in Vizag?' → realtime weather in Vizag?
   'who is Elon Musk?' → realtime who is Elon Musk?
   'latest news about AI?' → realtime latest news about AI?

-> 'open (name)' — open an app or website. Multiple: 'open x, open y'
   'open YouTube' → open YouTube
   'open YouTube and Spotify' → open YouTube, open Spotify

-> 'close (name)' — close an app. Multiple: 'close x, close y'
   'close Chrome' → close Chrome

-> 'play (name)' — play a song or video. Multiple: 'play x, play y'
   'play Believer' → play Believer

-> 'generate image (description)' — generate an image. Multiple: 'generate image x, generate image y'
   'generate image of Iron Man' → generate image Iron Man

-> 'reminder (time and task)' — set a reminder or alarm.
   'remind me at 9pm for meeting' → reminder 9:00pm meeting

-> 'system (task)' — control system settings. Multiple: 'system x, system y'
   'mute my PC' → system mute
   'increase volume' → system volume up

-> 'content (topic)' — write content like emails, letters, essays, code. Multiple: 'content x, content y'
   'write a sick leave application' → content sick leave application
   'write an email to manager' → content email to manager

-> 'google search (topic)' — search specifically on Google.
   'search Python on Google' → google search Python

-> 'youtube search (topic)' — search specifically on YouTube.
   'search Carry Minati on YouTube' → youtube search Carry Minati

*** MULTI-TASK: 'open YouTube and play Believer' → open YouTube, play Believer ***
*** CANNOT DECIDE: respond with 'general (query)' ***
*** NEVER answer. NEVER explain. ONE line only. ***
"""

VALID_FUNCS = [
    "exit", "general", "realtime", "open", "close", "play",
    "generate", "reminder", "system", "content", "google search", "youtube search"
]


def route(query):
    """
    Streams the classification and returns a list of decisions.
    e.g. "open YouTube and play Believer" → ["open youtube", "play believer"]
    """
    try:
        print("🔀 Classifying", end="", flush=True)

        full_response = ""

        # Cohere V2 streaming — iterate directly, no 'with' block
        for event in co.chat_stream(
            model="command-a-03-2025",
            messages=[
                {"role": "system", "content": ROUTING_PROMPT},
                {"role": "user", "content": query}
            ]
        ):
            if hasattr(event, 'type') and event.type == 'content-delta':
                token = event.delta.message.content.text
                full_response += token
                print(".", end="", flush=True)

        print()  # newline after dots

        # Parse into list — split multi-task by comma
        decision_line = full_response.strip().lower().split('\n')[0].strip()
        tasks = [t.strip() for t in decision_line.split(",") if t.strip()]

        # Filter to only valid classifications
        valid_tasks = []
        for task in tasks:
            for func in VALID_FUNCS:
                if task.startswith(func):
                    valid_tasks.append(task)
                    break

        # Fallback if nothing valid found
        if not valid_tasks:
            valid_tasks = [f"general {query}"]

        print(f"🔀 Router decision: {valid_tasks}")
        return valid_tasks

    except Exception as e:
        print(f"\n⚠️ Router error: {e} — defaulting to general")
        return [f"general {query}"]


# ===== TEST =====
if __name__ == "__main__":
    print("🔀 Router Test — type a query, press Enter. Ctrl+C to quit.\n")
    while True:
        try:
            query = input("Query: ").strip()
            if not query:
                continue
            result = route(query)
            print(f"Result: {result}\n")
        except KeyboardInterrupt:
            print("\n👋 Test ended.")
            break