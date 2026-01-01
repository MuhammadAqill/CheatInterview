import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

# --- Configuration ---
FS = 44100  # Sample rate (common values are 44100 or 48000 Hz)
DURATION = 5  # Duration of recording in seconds
CHANNELS = 1  # Number of audio channels (1 for mono, 2 for stereo)
FILENAME = "my_recording.wav" # Output filename

print(f"Recording started for {DURATION} seconds...")

# Record audio from the microphone to a NumPy array
# The dtype='float32' is a common format for sounddevice
recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=CHANNELS, dtype='float32')

# Wait until recording is finished
sd.wait()

print(f"Recording complete. Saving to {FILENAME}...")

# Convert the float32 recording to 16-bit integers (standard for Microsoft PCM WAV)
# Scipy's write function handles the WAV header creation, ensuring PCM format
recording_int16 = (recording * 32767).astype(np.int16)

# Save the recorded NumPy array as a WAV file
write(FILENAME, FS, recording_int16) #

print(f"File '{FILENAME}' saved successfully.")
