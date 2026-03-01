import os
import asyncio
import tempfile
import time
import threading

import edge_tts
import pygame


async def _generate_speech(text, output_file):
    """Generate speech using edge-tts and save to file"""
    voice = "en-US-JennyNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)


def generate_speech_background(text, output_file, ready_event):
    """Run TTS generation in background thread, set event when done"""
    try:
        asyncio.run(_generate_speech(text, output_file))
        ready_event.set()
    except Exception as e:
        print(f"❌ TTS generation error: {e}")
        ready_event.set()  # set anyway so we don't hang


def speak(text, display=True):
    """
    Speak text using Edge-TTS + pygame.
    display=True → print text to terminal (for greetings/coming-soon messages)
    display=False → text already streamed by brain.py, just play audio
    """
    temp_filename = None

    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_filename = temp_file.name
        temp_file.close()

        if display:
            print(f"\n{'=' * 60}")
            print(f"🤖 SAIYAARA: {text}")
            print(f"{'=' * 60}")

        # Generate and play
        asyncio.run(_generate_speech(text, temp_filename))

        if not pygame.mixer.get_init():
            pygame.mixer.init()

        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.music.unload()
        time.sleep(0.1)
        os.unlink(temp_filename)

    except KeyboardInterrupt:
        print("\n⏹️ Speech stopped by user!")
        pygame.mixer.music.stop()
        if temp_filename and os.path.exists(temp_filename):
            try:
                time.sleep(0.1)
                os.unlink(temp_filename)
            except:
                pass

    except Exception as e:
        print(f"❌ Speech error: {e}")
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.unlink(temp_filename)
            except:
                pass


def speak_streamed(text):
    """
    Play audio for text that was already streamed to terminal by brain.py.
    Assumes TTS file was pre-generated during streaming via start_tts_generation().
    Falls back to normal speak() if pre-generation failed.
    """
    speak(text, display=False)


def start_tts_generation(text):
    """
    Start TTS generation in background immediately.
    Returns (temp_filename, ready_event) — call play_pregenerated() when ready to play.
    """
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_filename = temp_file.name
        temp_file.close()

        ready_event = threading.Event()
        thread = threading.Thread(
            target=generate_speech_background,
            args=(text, temp_filename, ready_event),
            daemon=True
        )
        thread.start()

        return temp_filename, ready_event

    except Exception as e:
        print(f"❌ Could not start TTS generation: {e}")
        return None, None


def play_pregenerated(temp_filename, ready_event):
    """
    Wait for pre-generated TTS file and play it.
    Called after streaming text is fully displayed.
    """
    if not temp_filename or not ready_event:
        return

    try:
        # Wait for generation to finish (should be nearly done by now)
        ready_event.wait(timeout=10)

        if not os.path.exists(temp_filename):
            return

        if not pygame.mixer.get_init():
            pygame.mixer.init()

        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.music.unload()
        time.sleep(0.1)
        os.unlink(temp_filename)

    except KeyboardInterrupt:
        print("\n⏹️ Speech stopped by user!")
        pygame.mixer.music.stop()
        if os.path.exists(temp_filename):
            try:
                os.unlink(temp_filename)
            except:
                pass

    except Exception as e:
        print(f"❌ Playback error: {e}")
        if os.path.exists(temp_filename):
            try:
                os.unlink(temp_filename)
            except:
                pass


# ===== TEST =====
if __name__ == "__main__":
    print("🔊 TTS Test — type text to speak, press Enter. Ctrl+C to quit.\n")
    while True:
        try:
            text = input("Text to speak: ").strip()
            if not text:
                continue
            print("🔊 Speaking...")
            speak(text, display=True)
            print("✅ Done.\n")
        except KeyboardInterrupt:
            print("\n👋 Test ended.")
            break