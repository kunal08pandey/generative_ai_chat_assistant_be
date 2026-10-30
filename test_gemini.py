import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

def list_models():
    env_path = Path(__file__).resolve().parent / '.env'
    load_dotenv(env_path, override=True)
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        print(f"ERROR: GEMINI_API_KEY environment variable is not set. Checked: {env_path}")
        return

    client = genai.Client(api_key=api_key)
    
    print("Available Models for your API Key:")
    print("-" * 30)
    found = False
    for m in client.models.list():
        print(f"- {m.name}")
        found = True
            
    if not found:
        print("No generation models found for this API key.")

if __name__ == "__main__":
    list_models()
