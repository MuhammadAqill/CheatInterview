import speech_recognition as sr

# Initialize the recognizer
r = sr.Recognizer()

# Specify the path to your audio file
AUDIO_FILE = "my_recording.wav"

# Use the audio file as the audio source
with sr.AudioFile(AUDIO_FILE) as source:
    # Record the audio data from the file
    audio = r.record(source)
    print("Audio file loaded, processing...")

try:
    # Convert the audio data to text
    text = r.recognize_google(audio)
    print("The audio file says: " + text)
except sr.UnknownValueError:
    print("Sorry, could not understand audio in the file.")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")
