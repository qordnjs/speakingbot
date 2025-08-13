# app.py
"""
Speaking Bot (Online + Offline) - Gradio 4.x compatible
Features:
 - Two tabs: Online (calls OpenAI APIs) and Offline (Whisper local)
 - Microphone input: gr.Audio(microphone=True, type="filepath")
 - Phoneme-level feedback (g2p_en)
 - Debug log panel (shows errors & progress)
 - GPU-ready for Whisper (detects CUDA / uses environment WHISPER_MODEL)
"""
import os, re, json, time, uuid, traceback, random, shutil, subprocess
from typing import Tuple, Optional, Dict, Any

import numpy as np
import soundfile as sf
import gradio as gr

# Optional imports (handle gracefully)
try:
    import torch
except Exception:
    torch = None

try:
    import whisper
except Exception:
    whisper = None

try:
    import openai
except Exception:
    openai = None

# Phoneme converter
from g2p_en import G2p
from jiwer import wer
from rapidfuzz import fuzz

# TTS (Coqui)
try:
    from TTS.api import TTS
except Exception:
    TTS = None

# ---------------------------
# Config & directories
# ---------------------------
APP_TITLE = "🗣️ Speaking Bot — Online & Offline (Gradio 4.x)"
DATA_DIR = "data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")  # base/small/medium/large
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # optional

# Levels
LEVELS = {
    "Easy": {"speed": 0.85, "wpm": (70, 95), "prompts": [
        "Introduce yourself in a few short sentences.",
        "What's your favorite food and why?",
        "Describe your morning routine."
    ]},
    "Medium": {"speed": 1.0, "wpm": (95, 120), "prompts": [
        "Tell me about your last holiday.",
        "Describe a useful app you use and why you like it.",
        "What are your goals for this year?"
    ]},
    "Hard": {"speed": 1.15, "wpm": (120, 150), "prompts": [
        "Discuss whether schools should adopt AI tutors — pros and cons.",
        "Describe a challenge you faced and how you solved it.",
        "What is your view on climate change impacts in cities?"
    ]},
}

# ---------------------------
# Helpers: logging, audio
# ---------------------------
def _ts(): return time.strftime("%H:%M:%S")
def add_log(log: Optional[str], msg: str) -> str:
    if log is None: log = ""
    return log + f"[{_ts()}] {msg}\n"

def write_wav(np_audio: np.ndarray, sr: int, path: str):
    sf.write(path, np_audio, sr)

def duration_sec(path: str) -> float:
    a, sr = sf.read(path)
    if a.ndim == 2:
        a = a.mean(axis=1)
    return len(a) / sr

def ensure_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False

def change_speed(in_wav: str, out_wav: str, atempo: float):
    """Use ffmpeg atempo if available; fallback to naive resample (pitch changes)."""
    if abs(atempo - 1.0) < 1e-3:
        shutil.copyfile(in_wav, out_wav)
        return
    if ensure_ffmpeg():
        subprocess.run(["ffmpeg", "-y", "-i", in_wav, "-filter:a", f"atempo={atempo}", out_wav],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return
    # fallback: very naive
    audio, sr = sf.read(in_wav)
    new_len = int(len(audio) / atempo)
    idx = np.linspace(0, len(audio)-1, new_len).astype(np.int64)
    audio2 = audio[idx]
    sf.write(out_wav, audio2, sr)

# ---------------------------
# TTS (Coqui) - robust
# ---------------------------
_TTS = None
if TTS is not None:
    try:
        _TTS = TTS("tts_models/en/ljspeech/tacotron2-DDC")
    except Exception:
        _TTS = None

def synth_tts(text: str, out_wav: str, speed: float = 1.0) -> str:
    try:
        if _TTS is None:
            # fallback: short silence so UI doesn't break
            silence = np.zeros(int(16000 * max(1.0, len(text) / 12)))
            sf.write(out_wav, silence, 16000)
            return out_wav
        tmp = out_wav.replace(".wav", ".raw.wav")
        _TTS.tts_to_file(text=text, file_path=tmp)
        change_speed(tmp, out_wav, speed)
        try: os.remove(tmp)
        except: pass
        return out_wav
    except Exception:
        silence = np.zeros(int(16000 * 1.0))
        sf.write(out_wav, silence, 16000)
        return out_wav

# ---------------------------
# Phoneme tools
# ---------------------------
g2p = G2p()

def normalize_text(t: str) -> str:
    t = (t or "").lower().strip()
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def phonemize(text: str):
    if not text:
        return []
    phones = [p for p in g2p(text) if isinstance(p, str) and p.strip()]
    # to uppercase, strip punctuation
    phones = [re.sub(r"[^A-Z0-9]+", "", p.upper()) for p in phones if p.strip()]
    return phones

def levenshtein_tokens(a: list, b: list) -> int:
    if not a: return len(b)
    if not b: return len(a)
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0]=i
    for j in range(n+1): dp[0][j]=j
    for i in range(1,m+1):
        for j in range(1,n+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[m][n]

def phoneme_feedback(expected_text: str, recognized_text: str, max_items: int = 6) -> Dict[str, Any]:
    e = normalize_text(expected_text)
    r = normalize_text(recognized_text)
    ep = phonemize(e)
    rp = phonemize(r)
    if not ep:
        return {"per": 0.0, "mismatches": [], "tips": [], "summary": "No reference phonemes."}
    dist = levenshtein_tokens(ep, rp)
    per = dist / max(1, len(ep))
    mismatches = []
    i = j = 0
    while i < len(ep) and j < len(rp) and len(mismatches) < max_items:
        if ep[i] == rp[j]:
            i+=1; j+=1; continue
        mismatches.append((ep[i], rp[j] if j < len(rp) else ""))
        i+=1; j+=1
    while i < len(ep) and len(mismatches) < max_items:
        mismatches.append((ep[i], ""))
        i+=1
    tips = []
    for ref, hyp in mismatches:
        tip = "Listen to the target and practice minimal pairs."
        if ref.startswith("TH"): tip = "Make /θ/ by placing the tongue between the teeth and pushing air."
        elif ref.startswith("V"): tip = "Pronounce /v/ by placing upper teeth on lower lip and voicing."
        elif ref.startswith("R"): tip = "R: curl or bunch your tongue slightly; contrast with L."
        elif ref.startswith("L"): tip = "L: tip of the tongue touches alveolar ridge; breath out sides."
        elif ref.startswith("Z"): tip = "Z is voiced; feel throat vibration."
        tips.append({"ref": ref, "hyp": hyp, "tip": tip})
    summary = f"PER: {per:.3f} — {len(mismatches)} mismatches."
    return {"per": round(per,3), "mismatches": mismatches, "tips": tips, "summary": summary}

# ---------------------------
# ASR loading & transcribe (offline whisper)
# ---------------------------
ASR_MODEL = None
def load_asr(log: Optional[str]) -> Tuple[Optional[object], str]:
    global ASR_MODEL
    try:
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        log = add_log(log, f"Loading Whisper '{WHISPER_MODEL}' on {device}...")
        if whisper is None:
            log = add_log(log, "[ERR] whisper package not found. Install openai-whisper.")
            return None, log
        ASR_MODEL = whisper.load_model(WHISPER_MODEL, device=device)
        if device == "cuda" and torch is not None:
            try:
                name = torch.cuda.get_device_name(0)
                log = add_log(log, f"CUDA available: {name}")
            except Exception:
                pass
        log = add_log(log, "Whisper loaded.")
        return ASR_MODEL, log
    except Exception as e:
        log = add_log(log, f"[ERR] load_asr: {e}\n{traceback.format_exc()}")
        return None, log

def transcribe_offline(wav_path: str, log: Optional[str]) -> Tuple[str, str]:
    global ASR_MODEL
    if not wav_path or not os.path.exists(wav_path):
        log = add_log(log, "[ERR] transcribe_offline: no audio file")
        return "", log
    try:
        if ASR_MODEL is None:
            ASR_MODEL, log = load_asr(log)
            if ASR_MODEL is None: return "", log
        result = ASR_MODEL.transcribe(wav_path, language="en")
        text = (result.get("text") or "").strip()
        log = add_log(log, f"ASR (offline) => {text}")
        return text, log
    except Exception as e:
        log = add_log(log, f"[ERR] transcribe_offline: {e}\n{traceback.format_exc()}")
        return "", log

# ---------------------------
# Online helpers using OpenAI (optional)
# ---------------------------
def online_synthesize_and_prompt(level: str, log: Optional[str]) -> Tuple[str, Optional[str], str]:
    """
    Uses OpenAI Chat to generate a short prompt + uses gTTS (fallback) to produce a wav.
    If OPENAI_API_KEY is not set or openai lib missing, returns instruction message.
    """
    try:
        if openai is None or not OPENAI_API_KEY:
            msg = "OpenAI key/lib not configured. Set OPENAI_API_KEY to use online mode."
            log = add_log(log, msg)
            return msg, None, log
        openai.api_key = OPENAI_API_KEY
        # Create a short prompt via ChatCompletion
        system = "You are an English tutor. Create one short speaking prompt appropriate for the level."
        user = f"Level: {level}. Provide one prompt sentence for speaking practice (not too long)."
        resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ], max_tokens=64, temperature=0.7)
        bot_text = resp.choices[0].message.content.strip()
        # Synthesize via gTTS (available in requirements)
        out_wav = os.path.join(AUDIO_DIR, f"online_prompt_{uuid.uuid4().hex[:8]}.wav")
        try:
            from gtts import gTTS
            mp3 = out_wav.replace(".wav", ".mp3")
            gTTS(bot_text, lang="en").save(mp3)
            # convert mp3 -> wav using pydub if ffmpeg available
            try:
                from pydub import AudioSegment
                seg = AudioSegment.from_file(mp3)
                seg.export(out_wav, format="wav")
                os.remove(mp3)
            except Exception:
                # fallback: write silence so UI doesn't break
                silence = np.zeros(int(16000 * 1.0))
                sf.write(out_wav, silence, 16000)
        except Exception:
            out_wav = None
        log = add_log(log, f"Online prompt: {bot_text}")
        return bot_text, out_wav, log
    except Exception as e:
        log = add_log(log, f"[ERR] online_synthesize_and_prompt: {e}\n{traceback.format_exc()}")
        return "Online mode failed", None, log

def online_evaluate_reply(level: str, user_wav: str, log: Optional[str]) -> Tuple[str, str]:
    """
    Transcribe with OpenAI Whisper API and ask ChatGPT for feedback. Returns (feedback_md, log)
    """
    try:
        if openai is None or not OPENAI_API_KEY:
            msg = "OpenAI key/lib not configured. Set OPENAI_API_KEY to use online mode."
            log = add_log(log, msg)
            return msg, log
        openai.api_key = OPENAI_API_KEY
        # upload audio file to Whisper (OpenAI) - uses openai.Audio.transcribe
        with open(user_wav, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)
        user_text = transcript.text.strip() if hasattr(transcript, "text") else transcript.get("text","")
        log = add_log(log, f"Online ASR => {user_text}")
        # Get feedback from ChatGPT
        system = "You are a concise pronunciation coach. Give 3 short points about pronunciation and one follow-up question."
        user_msg = f"Student said: {user_text}\nLevel: {level}"
        resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
            {"role":"system","content":system},
            {"role":"user","content":user_msg}
        ], max_tokens=180, temperature=0.6)
        feedback = resp.choices[0].message.content.strip()
        fb_md = f"**You said (online ASR):** {user_text}\n\n**Feedback:**\n{feedback}"
        return fb_md, log
    except Exception as e:
        log = add_log(log, f"[ERR] online_evaluate_reply: {e}\n{traceback.format_exc()}")
        return f"Online eval failed: {e}", log

# ---------------------------
# Scoring util (offline)
# ---------------------------
def words_per_min(text: str, audio_path: str) -> float:
    n = max(1, len(normalize_text(text).split()))
    d = max(0.01, duration_sec(audio_path))
    return 60.0 * n / d

def offline_evaluate(level: str, prompt_text: str, user_wav: str, log: Optional[str]) -> Tuple[str, str]:
    try:
        hyp, log = transcribe_offline(user_wav, log)
        if not hyp:
            return "Please try recording again.", log
        # WER
        w_err = wer(normalize_text(prompt_text), normalize_text(hyp))
        # phoneme feedback
        pf = phoneme_feedback(prompt_text, hyp)
        # WPM
        wpm = words_per_min(hyp, user_wav)
        lo, hi = LEVELS[level]["wpm"]
        pace_msg = "Good pace!"
        if wpm < lo: pace_msg = f"Try speaking faster to reach {lo}-{hi} wpm."
        elif wpm > hi: pace_msg = f"Try slowing down to {lo}-{hi} wpm."
        # Compose markdown feedback
        md = f"**You said (offline ASR):** {hyp}\n\n**WER:** {w_err:.3f}  |  **PER:** {pf['per']}\n**WPM:** {wpm:.1f} — {pace_msg}\n\n**Phoneme tips:**\n"
        for t in pf['tips'][:5]:
            md += f"- Ref `{t['ref']}` vs You `{t['hyp']}` → {t['tip']}\n"
        log = add_log(log, f"Eval offline: WER={w_err:.3f}, PER={pf['per']}, WPM={wpm:.1f}")
        return md, log
    except Exception as e:
        log = add_log(log, f"[ERR] offline_evaluate: {e}\n{traceback.format_exc()}")
        return f"Offline evaluation failed: {e}", log

# ---------------------------
# UI (Gradio Blocks) - two tabs
# ---------------------------
with gr.Blocks(title=APP_TITLE, analytics_enabled=False) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    with gr.Tabs():
        with gr.TabItem("Online (API)"):
            with gr.Row():
                online_level = gr.Dropdown(list(LEVELS.keys()), value="Easy", label="Level (Online)")
                online_new_btn = gr.Button("🎲 New Prompt (Online)")
            online_prompt = gr.Textbox(label="Prompt (Online)", interactive=False)
            online_bot_audio = gr.Audio(label="Bot Voice (Online)", type="filepath")
            online_user_audio = gr.Audio(microphone=True, type="filepath", label="Your Answer (Online)")
            online_eval_btn = gr.Button("✅ Evaluate (Online)")
            online_feedback = gr.Markdown()

        with gr.TabItem("Offline (Local)"):
            with gr.Row():
                offline_level = gr.Dropdown(list(LEVELS.keys()), value="Easy", label="Level (Offline)")
                offline_init_btn = gr.Button("⚙️ Init models (Offline)")
                offline_new_btn = gr.Button("🎲 New Prompt (Offline)")
