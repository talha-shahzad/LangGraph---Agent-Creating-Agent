import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("GEMINI_API_KEY not found in .env")
else:
    genai.configure(api_key=api_key)
    try:
        print("Listing available embedding models...")
        for m in genai.list_models():
            if 'embedContent' in m.supported_generation_methods:
                print(f"Name: {m.name}, Display Name: {m.display_name}")
    except Exception as e:
        print(f"Error listing models: {e}")
