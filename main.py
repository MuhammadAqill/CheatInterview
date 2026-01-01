import speech_recognition as sr
import time

OUTPUT_FILE_NAME = 'output.txt'

recognizer = sr.Recognizer()

def main():
    RECORDING_TIME_IN_SECONDS = 30
    print('Starting to record audio...')
    start_time = time.time()
    with sr.Microphone() as microphone:
        recognizer.adjust_for_ambient_noise(microphone, duration=1)
        print('Adjusted for ambient noise, Listening...')
        while time.time() - start_time < RECORDING_TIME_IN_SECONDS:
            remaining = RECORDING_TIME_IN_SECONDS = (time.time() - start_time)
            print(f"Remaining time: {int(remaining)} seconds", end='\r')
            try:
                audio = recognizer.listen(microphone, timeout=5, phrase_time_limit=10)
                text = recognize_speech(audio)
                save_text(text)
            except sr.WaitTimeoutError:
                print('Listening timeout while waiting for a phrase to start.')
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.5)
    print('\nFinished recording! Transcription saved to: ', OUTPUT_FILE_NAME)

def recognize_speech(audio):
    try:
        text = recognizer.recognize_google(audio)
        timestap = time.strftime("[%H:%M:%S]", time.localtime())
        print("Recognized Speech: {text}")
        return timestap + text
    except sr.UnknownValueError:
        print("")
    except sr.RequestError as re:
        print(f"Speech recognition service error: {re}")
    return None

def save_text(text):
    if text:
        with open(OUTPUT_FILE_NAME, 'a') as file:
            file.write(text + '\n')

if __name__ == '__main__':
    main()