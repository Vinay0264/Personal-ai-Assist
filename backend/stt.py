import time
import threading

import speech_recognition as sr
import keyboard


recognizer = sr.Recognizer()


def listen():
    """Listen using push-to-talk (Press Space to start/stop)"""
    print("\n" + "=" * 60)
    print("🎤 PUSH-TO-TALK MODE")
    print("=" * 60)
    print("👉 Press SPACE BAR once to START speaking")
    print("👉 Press SPACE BAR again to STOP")
    print("=" * 60)

    with sr.Microphone() as source:
        print("\n🔧 Calibrating microphone...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        recognizer.energy_threshold = 200
        recognizer.dynamic_energy_threshold = True
        print("✅ Ready!")

        print("\n⏸️  Press SPACE BAR to start recording...")
        keyboard.wait('space')
        time.sleep(0.1)

        print("\n🔴 RECORDING... Press SPACE BAR again when done!")
        print("=" * 60)
        print("🎙️  Speak now...")

        stop_recording = threading.Event()

        def wait_for_space():
            keyboard.wait('space')
            stop_recording.set()

        space_thread = threading.Thread(target=wait_for_space, daemon=True)
        space_thread.start()

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

            time.sleep(0.5)
            try:
                audio = recognizer.listen(source, timeout=0.5, phrase_time_limit=1)
                audio_data.append(audio.get_raw_data())
            except:
                pass

            print("⏹️  Recording stopped!")

            if not audio_data:
                print("❌ No audio recorded. Please try again.")
                return None

            print("🔄 Processing speech...")

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