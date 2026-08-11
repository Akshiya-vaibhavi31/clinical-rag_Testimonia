"""
test_gemini.py

A tiny standalone script with ONE job: confirm your Gemini API key actually
works before we wire it into the full RAG pipeline. If this fails, it's much
easier to debug in isolation than inside a larger script.

Usage:
    python -m src.test_gemini
"""

from google import genai
from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME


def run():
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found in your .env file.")
        return

    print("Sending a test request to Gemini...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents="Reply with exactly one sentence confirming you received this message.",
    )

    print("\nSUCCESS. Gemini responded:")
    print(response.text)


if __name__ == "__main__":
    run()
