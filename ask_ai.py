import requests

def answer_question(question):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi",
            "prompt": question,
            "stream": False
        }
    )

    return response.json()["response"]


if __name__ == "__main__":
    while True:
        try:
            q = input("Soalan interview (Ctrl+C untuk keluar): ")
            ans = answer_question(q)
            print("\nJawapan AI:")
            print(ans)
            print("-" * 30)

        except KeyboardInterrupt:
            print("\nExit.")
            break