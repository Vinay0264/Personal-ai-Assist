import time
import threading

import speech_recognition as sr
import keyboard


recognizer = sr.Recognizer()


def listen():
    """Listen using F2 toggle — F2 to start, F2 again to stop"""

    print("\n🎤 Mic is ON — speak now! Press F2 again to stop.")
    print("=" * 60)

    with sr.Microphone() as source:
        # Quick calibration
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        recognizer.energy_threshold = 200
        recognizer.dynamic_energy_threshold = True

        stop_recording = threading.Event()

        def wait_for_f2():
            keyboard.wait('f2')
            stop_recording.set()

        f2_thread = threading.Thread(target=wait_for_f2, daemon=True)
        f2_thread.start()

        audio_data = []

        try:
            start_time = time.time()
            max_duration = 60

            while not stop_recording.is_set():
                if time.time() - start_time > max_duration:
                    print("\n⏱️  Maximum recording time reached (60 seconds)")
                    break
                try:
                    audio = recognizer.listen(source, timeout=0.5, phrase_time_limit=1)
                    audio_data.append(audio.get_raw_data())
                except sr.WaitTimeoutError:
                    continue

            # Capture any final audio after F2 press
            time.sleep(0.3)
            try:
                audio = recognizer.listen(source, timeout=0.3, phrase_time_limit=1)
                audio_data.append(audio.get_raw_data())
            except:
                pass

            print("⏹️  Mic OFF — processing...")

            if not audio_data:
                print("❌ No audio recorded. Please try again.")
                return None

            sample_rate = source.SAMPLE_RATE
            sample_width = source.SAMPLE_WIDTH
            combined_data = b''.join(audio_data)
            audio_full = sr.AudioData(combined_data, sample_rate, sample_width)

            text = recognizer.recognize_google(audio_full)
            print(f"✅ You said: {text}\n")
            return text

        except sr.UnknownValueError:
            print("❌ Couldn't understand. Please speak more clearly.")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None


# ===== TEST =====
if __name__ == "__main__":
    print("🎤 STT Test — press F2 to start, F2 again to stop. Ctrl+C to quit.\n")
    while True:
        try:
            result = listen()
            if result:
                print(f"📝 Transcribed: {result}\n")
            else:
                print("❌ Nothing captured.\n")
        except KeyboardInterrupt:
            print("\n👋 Test ended.")
            break