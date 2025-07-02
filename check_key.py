# /JANUS-CORE/check_key.py

import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("--- Running Key Diagnostic Probe ---")

# Use os.environ.get() which safely returns None if the key is not found
api_key = os.environ.get("GOOGLE_API_KEY")

if api_key:
    print(f"SUCCESS: Python has found the API key.")
    # For security, we'll only print the first few and last few characters
    print(f"Key Found: {api_key[:5]}...{api_key[-4:]}")
else:
    print("FAILURE: Python could NOT find the GOOGLE_API_KEY environment variable.")
    print("The 'export' command did not work correctly for this session.")

print("--- Probe Complete ---")