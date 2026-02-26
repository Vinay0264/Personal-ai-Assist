import os
import asyncio
import tempfile
import time

import edge_tts
import pygame


async def _generate_speech(text, output_file):
    """Generate speech using edge-tts and save to file"""
    voice = "en-US-JennyNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)


def speak(text):
    """Speak text using Edge-TTS + pygame, with word-by-word display"""
    temp_filename = None
    sound = None

    try:
        # Generate speech file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_filename = temp_file.name
        temp_file.close()

        asyncio.run(_generate_speech(text, temp_filename))

        # Initialize pygame mixer
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        pygame.mixer.music.load(temp_filename)

        sound = pygame.mixer.Sound(temp_filename)
        audio_duration = sound.get_length()

        # Calculate word timing
        words = text.split()
        total_words = len(words)
        time_per_word = audio_duration / total_words if total_words > 0 else 0

        # Start playing audio
        pygame.mixer.music.play()

        print("\n" + "=" * 60)
        print("🤖 SAIYAARA:")

        LINE_WIDTH = 150
        current_line = ""
        start_time = time.time()

        for i, word in enumerate(words):
            target_time = start_time + (i * time_per_word)
            current_time = time.time()

            if current_time < target_time:
                time.sleep(target_time - current_time)

            test_line = current_line + word + " "
            if len(test_line) > LINE_WIDTH and current_line:
                print(current_line)
                current_line = word + " "
            else:
                current_line = test_line
                print(f"  {current_line}", end="\r", flush=True)

        if current_line:
            print(f"  {current_line}")
        print("=" * 60)

        # Wait for audio to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.music.unload()
        sound.stop()
        sound = None
        time.sleep(0.1)
        os.unlink(temp_filename)

        print("✅ Speech complete!\n")

    except KeyboardInterrupt:
        print("\n⏹️ Speech stopped by user!")
        pygame.mixer.music.stop()
        if sound:
            sound.stop()
        if temp_filename and os.path.exists(temp_filename):
            try:
                time.sleep(0.1)
                os.unlink(temp_filename)
            except:
                pass

    except Exception as e:
        print(f"❌ Speech error: {e}")
        if sound:
            sound.stop()
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.unlink(temp_filename)
            except:
                pass