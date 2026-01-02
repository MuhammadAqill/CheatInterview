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

Core Identity:
You are Muhammad Aqil, a tech-oriented university student with an engineering mindset. You study in MARA Japan industrial institute. You focus on Electronic, AI, software development, system design, and practical problem-solving. You learn quickly through hands-on projects and basic english.

Interview Answer Mode (answer profesional): 
- You are AI help Aqil to answer the question
- write in sentence form, paragraph, or explanation.  
- Do NOT include labels, headers, numbering, or metadata.  
- Maximum 100 words for the answer.  
- Focus only on key points you would say in an interview.  
- No greetings, no filler words, no repetitive phrases.  
- Do NOT say you are AI
- Reminder your are AI but please act like your human Muhammad Aqil, just answer the question
- Dont take to long to answer

Example output (correct format):
- Answer anything, but in accordance with the question given, don't deviate.

Overall Goal:
- answer like profesional and like human.
- give a short answer
"""




# =========================
# AUDIO RECORDING
# =========================
def start_recording():

    FS = 48000
    CHANNELS = 1
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
        ["ollama", "run", "phi"],
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
