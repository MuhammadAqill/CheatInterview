from openai import OpenAI, RateLimitError, APIError, APITimeoutError
import os
from dotenv import load_dotenv
import sys
from colorama import Fore, Style, init
import subprocess

load_dotenv()

prompt="introduce yourself in short way?"

def ask_ai(prompt):

    try:

        client = OpenAI(
            api_key=os.getenv("OPENAI"),
        )

        stream = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True,
        )

        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    print(delta.content, end="", flush=True)

    except KeyboardInterrupt:
        print("\n[Stopped]")

    except RateLimitError:
        print(Fore.RED + "\n[ERROR] Daily API limit reached.")
        print(Fore.YELLOW + "→ Try again tomorrow or add credits.\n")

        result = subprocess.run(
            ["ollama", "run", "phi"],
            input=prompt,
            text=True,
            capture_output=True
        )

        return result.stdout.strip()

if __name__ == "__main__":
    print(ask_ai(prompt))
