# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file automatically


GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# API_KEYS = [
#     os.getenv("GEMINI_API_KEY_1"),
#     os.getenv("GEMINI_API_KEY_2"),
#     os.getenv("GEMINI_API_KEY_3"),
# ]

# MODEL_NAME = "gemini-2.0-flash"
DB_FILE = "tickets.db"

_key_index = 0

# def get_next_key():
#     global _key_index
#     key = API_KEYS[_key_index % len(API_KEYS)]
#     _key_index += 1
#     return key