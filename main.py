import sounddevice as sd
from scipy.io.wavfile import write
import keyboard
import time

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def record_audio():

    print("Press Enter to start recording...")
    keyboard.wait('enter') # Wait for the first Enter key press

    start_time = time.time()

    fs = 44100 # Sample rate
    channels = 2# Number of channels

    # Start recording with an arbitrary large buffer
    recording = sd.rec(int(fs * 300), samplerate=fs, channels=channels)

    keyboard.wait('enter') # wait for the second Enter key press to stop
    print('Stopping...')
    print()

    sd.stop()

    # Calculate the actual duration
    duration = time.time() - start_time

    # Save only teh recorded portion
    write("output.wav", fs, recording[:int(duration * fs)])

def speech_to_text():

    audio_file= open("output", "rb")

    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe", 
        file=audio_file
    )

    return transcription.text

def transcribe():

    while True:
        record_audio()

        output = speech_to_text()

        print()
        print('Transcription')
        print(output)
        print()

if __name__ == "__main__":
    transcribe()
