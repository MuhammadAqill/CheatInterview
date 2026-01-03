from openai import OpenAI
import datetime

client = OpenAI(api_key="OPENAI")

# Ambil usage dari 1 Jan 2026 hingga sekarang
start_date = "2026-01-01"
end_date = datetime.date.today().isoformat()

usage = client.usage.list(
    start_date=start_date,
    end_date=end_date
)

print(usage)
