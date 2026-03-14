import os
import uuid
import warnings
os.environ["SDPA_ATTENTION"] = "0"
os.environ["ATTN_IMPLEMENTATION"] = "eager"
from flask import Flask, request, jsonify, send_file
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

app = Flask(__name__)

# Detect device (CUDA for RunPod/GPU, MPS for Mac M1-M4)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

print(f"Loading Chatterbox Multilingual model on {device}...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Native SDPA is now enabled by default and supported because we patched t3.py
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    
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
    # Ensure smooth ending by converting trailing colons to periods.
    # The model internally converts colons to commas, which results in an "unfinished" tone.
    if text.strip().endswith(':'):
        text = text.strip()[:-1] + "."
        
    language = data.get('language', 'de')
    audio_prompt_path = data.get('audio_prompt_path')
    exaggeration = data.get('exaggeration', 0.5)
    cfg_weight = data.get('cfg_weight', 0.5)
    temperature = data.get('temperature', 0.8)
    repetition_penalty = data.get('repetition_penalty', 2.0)
    speed = data.get('speed', 1.0)

    if audio_prompt_path and not os.path.exists(audio_prompt_path):
        return jsonify({'error': f'Audio prompt file not found: {audio_prompt_path}'}), 400

    try:
        # Generate audio using Chatterbox Multilingual
        print(f"Generating audio for text (lang={language}): {text[:50]}...")
        wav = model.generate(
            text, 
            language_id=language,
            audio_prompt_path=audio_prompt_path if audio_prompt_path else None,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty
        )
        
        import subprocess
        
        output_filename = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}_raw.wav")
        ta.save(output_filename, wav, model.sr)
        
        if speed != 1.0:
            final_output = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}_speed.wav")
            # Use FFmpeg's atempo filter to change speed while preserving pitch
            subprocess.run([
                "ffmpeg", "-y", "-i", output_filename,
                "-filter:a", f"atempo={speed}",
                final_output
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            final_output = output_filename
            
        with open(final_output, "rb") as f:
            buffer = io.BytesIO(f.read())
            
        # cleanup temp files
        if os.path.exists(output_filename): os.remove(output_filename)
        if final_output != output_filename and os.path.exists(final_output): os.remove(final_output)
        
        buffer.seek(0)
        
        # Return the generated audio directly from memory
        return send_file(buffer, mimetype='audio/wav', as_attachment=True, download_name=f"speech_{uuid.uuid4().hex}.wav")
    
    except Exception as e:
        print(f"Error during audio generation: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app locally on port 5001
    app.run(host='0.0.0.0', port=5001, debug=False)
