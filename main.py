import speech_recognition as sr
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import time
import sys
import os
from colorama import Fore, init
import threading
from queue import Queue
from typing import Optional
from dotenv import load_dotenv

from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from google.genai.types import GenerateContentConfig
from google import genai

import ollama


load_dotenv()
init(autoreset=True)

# =========================
# CONSTANTS (UPPERCASE)
# =========================
SYSTEM_RULE_BASIC = "You are [name] not AI assist. Answer briefly. your job is answer the interview question because your are interviewing"
SYSTEM_RULE_LOCAL = """
You are Muhammad Aqil, a fresh engineering student from MARA Japan Industrial Institute (MJII).

CORE VIBE:
- You are NOT a corporate robot. You are a real student.
- You do not use big, fancy words. You use simple, everyday English.
- It is okay to sound a bit raw or "fresh." You are humble.
- You are "clueless" about corporate buzzwords, but you know your technical stuff (Electronics & AI).

STRICT "NO SCRIPT" RULES:
1. NO BUZZWORDS: Banned words: "leverage," "seamlessly," "strive," "utilize," "pivotal," "showcase," "meticulous."
2. KEEP IT SIMPLE: Use "I used" instead of "I utilized." Use "I made" instead of "I constructed."
3. BE HONEST: Don't over-promise. Speak like you are thinking on the spot.
4. NO FLOWERY INTROS: Don't say "That is an excellent question." Just answer.
5. SENTENCE STYLE: Write like you talk. Short sentences. Straight to the point.

FORMAT:
- Max 60 words.
- 1 Paragraph only.
- No headers/bullets.

GOAL:
- Sound like a smart student who is eager to learn, not a CEO giving a speech.
- If asked technical things, explain it simply like you are showing a friend.

Start the answer immediately.
"""
FS = 16000
CHANNELS = 1
FILENAME = "my_recording.wav"
MAX_RECORDING_SECONDS = 30
AUDIO_BUFFER_SIZE = 1024  # Chunk size for streaming

# =========================
# CLIENTS (Pre-initialized)
# =========================
router_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("DEEPSEEK"),
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI")
)

gemini_client = genai.Client(
        api_key=os.environ.get("GEMINI")
)

# =========================
# RECOGNIZER (Pre-initialized)
# =========================
recognizer = sr.Recognizer()

# =========================
# STREAMING AUDIO RECORDING
# =========================
class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.audio_queue = Queue()
        self.recording_thread = None
        self.start_time = 0
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for real-time audio streaming"""
        if self.is_recording:
            self.audio_queue.put(indata.copy())
    
    def start_streaming(self):
        """Start recording with streaming callback"""
        self.is_recording = True
        self.start_time = time.time()
        self.audio_queue = Queue()
        
        # Start recording in a separate thread
        self.stream = sd.InputStream(
            samplerate=FS,
            channels=CHANNELS,
            dtype="float32",
            blocksize=AUDIO_BUFFER_SIZE,
            callback=self._audio_callback
        )
        self.stream.start()
        
        print(Fore.YELLOW + "Recording... Press Enter to stop.")
    
    def stop_and_save(self):
        """Stop recording and save to file"""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        
        duration = time.time() - self.start_time
        print(Fore.CYAN + f"Recording duration: {duration:.2f}s")
        
        # Collect all audio data
        audio_chunks = []
        while not self.audio_queue.empty():
            audio_chunks.append(self.audio_queue.get())
        
        if not audio_chunks:
            return None
            
        # Combine chunks
        recording = np.vstack(audio_chunks)
        recording_int16 = (recording * 32767).astype(np.int16)
        
        # Save to file
        write(FILENAME, FS, recording_int16)
        print(Fore.GREEN + f"Recording saved ({len(recording_int16)/FS:.2f}s).\n")
        
        return FILENAME

# =========================
# OPTIMIZED SPEECH TO TEXT
# =========================
def speech_to_text_optimized(filename: str) -> Optional[str]:
    """Optimized speech recognition with batch processing"""
    try:
        with sr.AudioFile(filename) as source:
            # Adjust for ambient noise once
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
        
        # Use Google's API with timeout and phrase hints
        text = recognizer.recognize_google(
            audio, 
            language="en-US",
            show_all=False  # Faster, only returns best result
        )
        
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
# USE LOCAL AI
# =========================
class LocalAI:
    def __init__(self, choose):
        self.choose = choose

    def start_chose():
        if self.choose == "y":
            return "y"
        
        else:
            return None

# =========================
# ASYNC LLM QUERY WITH CACHING
# =========================
class LLMProcessor:
    def __init__(self):
        self.response_cache = {}  # Simple cache to avoid duplicate processing
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Prevent rapid successive calls
        self.start_time = 0
    
    def ask_ai(self, prompt: str) -> None:
        """Optimized AI query with rate limiting and caching"""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_request_time < self.min_request_interval:
            time.sleep(self.min_request_interval)
        
        self.last_request_time = current_time
        
        # Check cache (simple hash)
        prompt_hash = hash(prompt)
        if prompt_hash in self.response_cache:
            print(Fore.YELLOW + "\n[CACHED RESPONSE]\n")
            print(Fore.GREEN + self.response_cache[prompt_hash])
            return
        
        start_request = time.time()
        first_token_time = None

        localAI = LocalAI(option)

        # Decision to if user local AI or NOT
        if localAI.choose == "y":
            # Use Local AI model first
            print(Fore.YELLOW + "→ Using Local AI model.\n")

            messages = [
                {'role': 'system', 'content': SYSTEM_RULE_LOCAL},
                {'role': 'user', 'content': prompt},
            ]

            # The 'stream=True' parameter enables streaming
            stream = ollama.chat(
                model='gemma:2b',
                messages=messages,
                stream=True,
            )

            # Iterate over the chunks to print the response as it arrives
            for chunk in stream:
                # 'chunk['message']['content']' holds the current portion of the response
                print(chunk['message']['content'], end='', flush=True)

        
        else:
            
            try:
                # Use DeepSeed model first
                print(Fore.YELLOW + "→ Using DeepSeek model.\n")
                response_stream = router_client.chat.completions.create(
                    model="nex-agi/deepseek-v3.1-nex-n1:free",
                    messages=[
                        {"role": "system", "content": SYSTEM_RULE_BASIC},
                        {"role": "user", "content": prompt}
                    ],
                    stream=True,
                    # max_tokens=20,  # Limit response size
                    temperature=0.7,
                )

                # duration = time.time() - self.start_time()

                # print(f'respone time: {duration:.1f}\n')
                
                response_text = ""
                for chunk in response_stream:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            if first_token_time is None:
                                first_token_time = time.time()
                                ttft = first_token_time - start_request
                                print(Fore.YELLOW + f"\n[TTFT: {ttft:.2f}s]\n")
                            
                            content = delta.content

                            print(Fore.GREEN + content, end="", flush=True)
                            response_text += content
                            
                
                # Cache the response
                if response_text:
                    self.response_cache[prompt_hash] = response_text
                
            except RateLimitError:
                print(Fore.RED + "\n[ERROR] Free API limit reached.")
                print(Fore.YELLOW + "→ Switching to GEMINI.\n")

                try :
                    response_stream = gemini_client.models.generate_content_stream(
                        model = "gemini-2.5-flash",
                        contents = prompt,
                        config=GenerateContentConfig(
                            system_instruction=[
                                SYSTEM_RULE_LOCAL
                            ]
                        ),
                    )

                    for chunk in response_stream:
                        print(Fore.GREEN + chunk.text)
                
                except RateLimitError:
                    print(Fore.RED + "\n[ERROR] Free API limit reached.")
                    print(Fore.YELLOW + "→ Switching to OpenAI paid model.\n")

                    # Use OpenAI as fallback
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",  # Updated to newer, faster model
                        messages=[
                            {"role": "system", "content": SYSTEM_RULE_BASIC},
                            {"role": "user", "content": prompt}
                        ],
                        # max_tokens=20,
                        temperature=0.7,
                    )


                
                response_text = response.choices[0].message.content
                print(Fore.GREEN + response_text)
                self.response_cache[prompt_hash] = response_text
                
            except APITimeoutError:
                print(Fore.RED + "\n[ERROR] AI response timeout.\n")
                
            except APIError as e:
                print(Fore.RED + f"\n[ERROR] API failure: {e}\n")
                
            except KeyboardInterrupt:
                print("\n[Stopped]")
                
            except Exception as e:
                print(Fore.RED + f"\n[UNEXPECTED ERROR] {e}\n")

# =========================
# MAIN LOOP WITH PARALLEL PROCESSING
# =========================
def transcribe_optimized():
    """Optimized main transcription loop"""
    recorder = AudioRecorder()
    llm_processor = LLMProcessor()
    
    try:
        while True:
            # Recording phase
            input(Fore.YELLOW + "Press Enter to start recording...")
            recorder.start_streaming()
            
            # Wait for stop signal
            input()  # Just wait for Enter, no prompt
            audio_file = recorder.stop_and_save()
            
            if not audio_file:
                print(Fore.RED + "No audio recorded. Try again.\n")
                continue
            
            # Speech recognition phase
            question = speech_to_text_optimized(audio_file)
            
            if question:
                print(Fore.MAGENTA + "\nInterviewing:")
                # Process LLM response
                llm_processor.ask_ai(question)
                print("\n" + "="*50 + "\n")  # Separator for clarity
                
    except KeyboardInterrupt:
        print(Fore.RED + "\nExiting safely...")
        if recorder.is_recording:
            recorder.stop_and_save()
        sys.exit(0)

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":

    # Pre-warm components
    print(Fore.BLUE + "Initializing speech recognition system...")
    
    # Test audio system
    try:
        sd.check_input_settings(device=None, channels=CHANNELS, samplerate=FS)
        print(Fore.GREEN + "Audio system ready.\n")
    except Exception as e:
        print(Fore.RED + f"Audio system error: {e}")
        sys.exit(1)

    # Choose to use local AI
    print("Use local AI? (Y/N): ")
    option = input()

    if option == "y":
        LocalAI("y")
    
    # Run optimized transcription
    transcribe_optimized()
