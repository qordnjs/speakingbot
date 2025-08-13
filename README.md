# Speaking Bot — Online & Offline (Gradio 4.x)

Features:
- Gradio web UI with microphone input
- Two modes: Online (OpenAI APIs) and Offline (local Whisper + phoneme feedback)
- Phoneme-level feedback (g2p_en)
- Debug log visible on UI
- GPU-ready (uses CUDA if available, works well on NVIDIA A2000)

## Quick start (Windows PowerShell)

1. Create & activate venv:
```powershell
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
$env:WHISPER_MODEL="base"  # or "small"
$env:OPENAI_API_KEY="sk-..."
python app.py
