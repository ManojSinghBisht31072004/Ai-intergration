# check_keys.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

keys = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
]

for i, key in enumerate(keys, start=1):
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content("say hi")
        print(f"KEY {i} ✅ WORKING")
    except Exception as e:
        print(f"KEY {i} ❌ FAILED — {str(e)[:80]}")