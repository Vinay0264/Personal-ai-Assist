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

-> 'exit' — user says goodbye or wants to end.
   'bye', 'goodbye', 'ok bye', 'see you' → exit

*** MULTI-TASK: 'open YouTube and play Believer' → open YouTube, play Believer ***
*** CANNOT DECIDE: respond with 'general (query)' ***
*** NEVER answer. NEVER explain. ONE line only. ***
"""


def route(query):
    """
    Takes user query and returns classification label + query.
    Supports multi-task queries.
    """
    try:
        response = co.chat(
            model="command-a-03-2025",
            messages=[
                {"role": "system", "content": ROUTING_PROMPT},
                {"role": "user", "content": query}
            ]
        )

        decision = response.message.content[0].text.strip().lower()

        # Take only first line in case model adds extra text
        decision = decision.split('\n')[0].strip()

        print(f"🔀 Router decision: {decision}")
        return decision

    except Exception as e:
        print(f"⚠️ Router error: {e} — defaulting to general")
        return f"general {query}"


if __name__ == "__main__":
    print("=" * 60)
    print("🔀 SAIYAARA Router Test")
    print("=" * 60)
    print("Type a query to test routing. Type 'exit' to quit.\n")

    while True:
        query = input("Enter a query: ").strip()
        if query.lower() == "exit":
            break
        if query:
            result = route(query)
            print(f"Result: {result}\n")