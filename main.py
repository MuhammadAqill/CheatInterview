import sounddevice as sd
from scipy.io.wavfile import write
import keyboard
import time

def record_audio():
    print("Press Enter to start recording...")
    keyboard.wait('enter') # Wait for the first Enter key press

    start_time = time.time()

    fs = 44100 # Sample rate
    channels = 2 # Number of channels

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

if __name__ == "__main__":
    record_audio()
