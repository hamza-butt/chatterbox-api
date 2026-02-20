import requests
import sys

URL = "http://localhost:5001/generate"

def main():
    payload = {
        "text": "Hallo, wie geht es dir heute? Dies is ein Test der deutschen Sprache.",
        "audio_prompt_path": "reference.mp3",
        "exaggeration": 0.5,
        "cfg_weight": 0.5
    }

    print(f"Sending request to {URL}...")
    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status()
        
        output_file = "test_output.wav"
        with open(output_file, "wb") as f:
            f.write(response.content)
            
        print(f"Success! Audio saved to {output_file}")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Is app.py currently running?")
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        if response.text:
            print(f"Details: {response.text}")

if __name__ == "__main__":
    main()
