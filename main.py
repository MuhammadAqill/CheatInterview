import speech_recognition as sr
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import time
import sys
import os
from colorama import Fore, init

from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()
init(autoreset=True)

# =========================
# SYSTEM RULE (FIXED)
# =========================
SYSTEM_RULE = "You are Muhammad Aqil not AI assist. Answer briefly. your job is answer the interview question because your are interviewing"

# =========================
# CLIENTS
# =========================
router_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("DEEPSEEK"),
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI")
)

# =========================
# AUDIO RECORDING
# =========================
def start_recording():
    FS = 16000
    CHANNELS = 1
    FILENAME = "my_recording.wav"

    input(Fore.YELLOW + "Press Enter to start recording...")
    start_time = time.time()

    recording = sd.rec(
        int(30 * FS),  # max 30s (lebih selamat)
        samplerate=FS,
        channels=CHANNELS,
        dtype="float32"
    )

    input(Fore.YELLOW + "Recording... Press Enter to stop.")
    sd.stop()

    duration = time.time() - start_time
    recording_int16 = (recording * 32767).astype(np.int16)
    write(FILENAME, FS, recording_int16[:int(duration * FS)])

    print(Fore.GREEN + "Recording saved.\n")

# =========================
# SPEECH TO TEXT
# =========================
def speech_to_text():
    r = sr.Recognizer()

    with sr.AudioFile("my_recording.wav") as source:
        audio = r.record(source)

    try:
        text = r.recognize_google(audio)
        print(Fore.CYAN + "Interview:")
        print(Fore.WHITE + text)
        return text

    except sr.UnknownValueError:
        print(Fore.RED + "Could not understand audio.")
        return None

    except sr.RequestError as e:
        print(Fore.RED + f"Speech API error: {e}")
        return None

# =========================
# ASK LLM
# =========================
def ask_ai(prompt):
    start_request = time.time()
    first_token_time = None

    try:
        stream = router_client.chat.completions.create(
            model="nex-agi/deepseek-v3.1-nex-n1:free",
            messages=[
                {"role": "system", "content": SYSTEM_RULE},
                {"role": "user", "content": prompt}
            ],
            stream=True,
        )

        for chunk in stream:
            now = time.time()
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    if first_token_time is None:
                        first_token_time = now
                        ttft = first_token_time - start_request
                        print(Fore.YELLOW + f"\n[TTFT: {ttft:.2f}s]\n")
                    print(Fore.GREEN + delta.content, end="", flush=True)

    except RateLimitError:
        print(Fore.RED + "\n[ERROR] Free API limit reached.")
        print(Fore.YELLOW + "→ Switching to OpenAI paid model.\n")

        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_RULE},
                {"role": "user", "content": prompt}
            ]
        )

        print(Fore.GREEN + response.choices[0].message.content)

    except APITimeoutError:
        print(Fore.RED + "\n[ERROR] AI response timeout.\n")

    except APIError as e:
        print(Fore.RED + f"\n[ERROR] API failure: {e}\n")

    except KeyboardInterrupt:
        print("\n[Stopped]")

    except Exception as e:
        print(Fore.RED + f"\n[UNEXPECTED ERROR] {e}\n")

# =========================
# MAIN LOOP
# =========================
def transcribe():
    try:
        while True:
            start_recording()
            question = speech_to_text()

            if question:
                print(Fore.MAGENTA + "\nInterviewing:")
                ask_ai(question)
                print("\n")

    except KeyboardInterrupt:
        print(Fore.RED + "\nExiting safely...")
        sd.stop()
        sys.exit(0)

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    transcribe()
