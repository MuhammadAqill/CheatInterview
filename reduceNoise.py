from scipy.io import wavfile
import noisereduce as nr
import soundfile as sf
import numpy as np

# Read the audio file (replace 'noisy_audio.wav' with your file path)
# Ensure it's a mono channel for simplicity, or handle multi-channel data
try:
    rate, data = wavfile.read('my_recording.wav')
except ValueError:
    # Fallback for formats not supported by scipy.io.wavfile
    data, rate = sf.read('my_recording.wav')
    # Convert to numpy array of the correct data type for processing
    data = (data * 32767).astype(np.int16)

# If the audio has multiple channels, select one (e.g., the first channel)
if len(data.shape) > 1:
    data = data[:, 0]
    # Reduce the noise
    # The 'prop_decrease' parameter controls the amount of noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8)

    # Write the noise-reduced audio to a new file
    wavfile.write('enhanced_audio_nr.wav', rate, reduced_noise)
    print("Background noise removed and saved as 'enhanced_audio_nr.wav'")
