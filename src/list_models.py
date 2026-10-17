import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for m in client.models.list():
    name = m.name.replace("models/", "")
    if "flash" in name and "generateContent" in (m.supported_actions or []):
        print(name)