import os
import uuid
import warnings
os.environ["SDPA_ATTENTION"] = "0"
os.environ["ATTN_IMPLEMENTATION"] = "eager"
from flask import Flask, request, jsonify, send_file
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

app = Flask(__name__)

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
print(f"Loading Chatterbox model on {device}...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Native SDPA is now enabled by default and supported because we patched t3.py
    model = ChatterboxTTS.from_pretrained(device=device)
    
print("Model loaded successfully.")

# Directory to temporarily store generated audio outputs
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import io

@app.route('/generate', methods=['POST'])
def generate_audio():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field in request body'}), 400

    text = data['text']
    audio_prompt_path = data.get('audio_prompt_path')
    exaggeration = data.get('exaggeration', 2.0)
    cfg_weight = data.get('cfg_weight', 0.5)

    if audio_prompt_path and not os.path.exists(audio_prompt_path):
        return jsonify({'error': f'Audio prompt file not found: {audio_prompt_path}'}), 400

    try:
        # Generate audio using Chatterbox
        print(f"Generating audio for text: {text}")
        wav = model.generate(
            text, 
            audio_prompt_path=audio_prompt_path if audio_prompt_path else None,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )
        
        # Save output to an in-memory byte buffer instead of a file
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        
        # Return the generated audio directly from memory
        return send_file(buffer, mimetype='audio/wav', as_attachment=True, download_name=f"speech_{uuid.uuid4().hex}.wav")
    
    except Exception as e:
        print(f"Error during audio generation: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app locally on port 5000
    app.run(host='0.0.0.0', port=5001, debug=False)
