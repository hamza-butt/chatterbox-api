import os
import uuid
from flask import Flask, request, jsonify, send_file
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

app = Flask(__name__)

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

print(f"Loading Chatterbox model on {device}...")
model = ChatterboxTTS.from_pretrained(device=device)
print("Model loaded successfully.")

# Directory to temporarily store generated audio outputs
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    output_filename = os.path.join(OUTPUT_DIR, f"speech_{uuid.uuid4().hex}.wav")

    try:
        # Generate audio using Chatterbox
        print(f"Generating audio for text: {text}")
        wav = model.generate(
            text, 
            audio_prompt_path=audio_prompt_path if audio_prompt_path else None,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )
        
        # Save output
        ta.save(output_filename, wav, model.sr)
        
        # Return the generated audio file
        return send_file(output_filename, mimetype='audio/wav', as_attachment=True)
    
    except Exception as e:
        print(f"Error during audio generation: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app locally on port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)
