import speech_recognition as sr
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import time
import sys
import subprocess
from colorama import Fore, Style, init

init(autoreset=True)

SYSTEM_RULE = """
YOU ARE MUHAMMAD AQIL. DO NOT SAY YOU ARE AN AI.

anwser in 5 sec
"""




# =========================
# AUDIO RECORDING
# =========================
def start_recording():

    FS = 16000
    CHANNELS = 2
    FILENAME = "my_recording.wav"

    input(Fore.YELLOW + "Press Enter to start recording...")

    start_time = time.time()

    recording = sd.rec(int(300 * FS), samplerate=FS, channels=CHANNELS, dtype='float32')

    input(Fore.YELLOW + "Recording... Press Enter to stop.")

    sd.stop()

    DURATION = time.time() - start_time

    recording_int16 = (recording * 32767).astype(np.int16)
    write(FILENAME, FS, recording_int16[:int(DURATION * FS)])

    print(Fore.GREEN + "Recording saved.\n")

# =========================
# SPEECH TO TEXT
# =========================
def speech_to_text():

    r = sr.Recognizer()
    AUDIO_FILE = "my_recording.wav"

    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)

    try:
        text = r.recognize_google(audio)
        print(Fore.CYAN + "You said:")
        print(Fore.WHITE + text)
        return text

    except sr.UnknownValueError:
        print(Fore.RED + "Could not understand audio.")
        return None

    except sr.RequestError as e:
        print(Fore.RED + f"Speech API error: {e}")
        return None

# =========================
# ASK OLLAMA (PHI)
# =========================
def ask_ai(prompt):

    full_prompt = SYSTEM_RULE + "\n\nInterview Question:\n" + prompt

    result = subprocess.run(
        ["ollama", "run", "mistral:instruct"],
        input=full_prompt,
        text=True,
        capture_output=True
    )

    return result.stdout.strip()

# =========================
# MAIN LOOP
# =========================
def transcribe():

    try:
        while True:

            start_recording()

            question = speech_to_text()

            if question:
                print(Fore.MAGENTA + "\nAI Interview Answer:")
                # print('')
                answer = ask_ai(question)
                print(Fore.GREEN + answer)
                print(Style.DIM + "\n-----------------------------\n")

    except KeyboardInterrupt:
        print(Fore.RED + "\nExiting safely...")
        sd.stop()
        sys.exit(0)

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    transcribe()
