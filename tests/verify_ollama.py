"""Verification script for the local Ollama LLM deployment.

Demonstrates a successful "Hello World" style call against the configured
Ollama server. Used to validate Part 1 of the assignment (Model Serving
& Deployment).

Usage:
    python tests/verify_ollama.py
"""

import sys
import time

from app.llm.errors import LLMClientError
from app.llm.factory import get_llm_client


def main() -> int:
    """Run the verification flow.

    Returns:
        0 on success, 1 on failure.
    """
    print("=" * 60)
    print("Ollama deployment verification")
    print("=" * 60)

    client = get_llm_client()
    print(f"Ollama host:  {client.host}")
    print(f"Ollama model: {client.model}\n")

    print("Step 1: Health check...")
    if not client.health_check():
        print("[FAIL] Ollama server is not reachable.")
        print("       Make sure Ollama is running: 'ollama serve'")
        return 1
    print("[OK] Backend is healthy.\n")

    print("Step 2: Sending 'Hello World' prompt...")
    prompt = "Say hello in exactly one short sentence."

    try:
        start = time.perf_counter()
        response = client.generate(prompt)
        elapsed = time.perf_counter() - start
    except LLMClientError as exc:
        print(f"[FAIL] Generation failed: {exc}")
        return 1

    print(f"[OK] Response received in {elapsed:.2f}s\n")
    print("-" * 60)
    print(f"PROMPT:   {prompt}")
    print(f"RESPONSE: {response.strip()}")
    print("-" * 60)
    print("\nVerification complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
