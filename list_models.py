import os
import sys
from google import genai

def list_models():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment.")
        return

    print(f"🔑 Check API Key: {api_key[:5]}...{api_key[-5:]}")
    
    try:
        client = genai.Client(api_key=api_key)
        print("\n🔎 Listing available models...")
        
        # Pagination handling if needed, but usually fits in one page for models
        for m in client.models.list():
            # In google-genai SDK, we might just trust list() returns valid models
            print(f" - {m.name}")
                
    except Exception as e:
        print(f"❌ Failed to list models: {e}")

if __name__ == "__main__":
    list_models()
