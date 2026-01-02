from gpt4all import GPT4All

# Path ke model GGUF terbaru
model_path = "/home/akira/Programming/CheatInterview/models/ggml-gpt4all-j.gguf"

# Load model offline
gptj = GPT4All(model_name="/home/akira/Programming/CheatInterview/models/ggml-gpt4all-j.gguf", allow_download=False)

system_prompt = """
You are Muhammad Aqil. Answer only in bullet points, no full sentences, no greetings, no extra text. Max 4 bullets.
"""

prompt = "Introduce yourself"

response = gptj.generate(system_prompt + "\n\n" + prompt, streaming=False)
bullets = [line.strip() for line in response.splitlines() if line.strip()]

for b in bullets:
    print("-", b)
