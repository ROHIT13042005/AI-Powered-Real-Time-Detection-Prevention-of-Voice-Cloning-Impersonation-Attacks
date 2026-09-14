# ============================================================
# VoiceShield AI
# AI-Powered Voice Cloning / Deepfake Voice Detection
# SIH26 Internal Hackathon Prototype
# ============================================================

import os
import re
import json
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import librosa
import soundfile as sf
import streamlit as st
import pandas as pd

from transformers import (
    AutoProcessor,
    AutoModelForAudioClassification,
    pipeline
)

import imageio_ffmpeg
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"
LANGUAGE_MODEL_NAME = "facebook/mms-lid-126"
ASR_MODEL_NAME = "openai/whisper-tiny"

# Weight given to the voice-clone score vs. the scam-script score
# when computing the combined "Fake Call Risk".
VOICE_RISK_WEIGHT = 0.6
SCAM_TEXT_RISK_WEIGHT = 0.4

# Replay / re-recording (speaker-to-mic) detection tuning.
REPLAY_HIGH_FREQ_HZ = 7000        # frequencies above this are checked
REPLAY_SNR_THRESHOLD_DB = 15      # below this, audio looks "re-captured"

# How much the replay heuristic's score contributes to the combined
# fake-probability. 1.0 = use it as-is (recommended). Raise this
# (e.g. 1.5-2.0) if you want the tool to flag replayed audio more
# aggressively for your demo/dataset — but higher values increase the
# chance of flagging genuinely low-quality (but real) mic recordings.
REPLAY_INFLUENCE = 1.0

# Purely for the on-screen "⚠️ replay signs detected" label — this
# does NOT gate whether the score gets adjusted (see combine function
# below), it only controls when the warning banner is shown.
REPLAY_FLAG_DISPLAY_THRESHOLD = 0.45
DETECTION_WINDOW_SEC = 4.0
DETECTION_HOP_SEC = 2.0
REPLAY_HIGH_RISK_THRESHOLD = 0.55
REPLAY_MEDIUM_THRESHOLD = 0.40

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "incidents.json")

os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VoiceShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    .risk-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 10px;
    }

    .high-risk {
        background-color: #ffdddd;
        border: 2px solid #ff4b4b;
    }

    .medium-risk {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }

    .low-risk {
        background-color: #ddffdd;
        border: 2px solid #28a745;
    }

    .feature-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# AUDIO INPUT / FORMAT HANDLING
# ============================================================

def convert_to_16k_mono_wav(uploaded_file):
    """
    Convert any FFmpeg-supported audio file to 16 kHz mono WAV.
    This makes MP3, M4A, AAC, OGG, FLAC, WMA, WEBM, OPUS, etc.
    much easier to process consistently.
    """
    original_suffix = Path(uploaded_file.name).suffix or ".bin"

    src_path = None
    wav_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=original_suffix
        ) as src_file:
            src_file.write(uploaded_file.getvalue())
            src_path = src_file.name

        wav_path = src_path + "_converted.wav"

        command = [
            FFMPEG_PATH,
            "-y",
            "-i", src_path,
            "-ac", "1",
            "-ar", "16000",
            "-vn",
            wav_path
        ]

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if process.returncode != 0:
            raise RuntimeError(
                "FFmpeg could not decode this file. "
                "Please upload a valid audio file."
            )

        audio, sample_rate = sf.read(
            wav_path,
            dtype="float32"
        )

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        duration = len(audio) / sample_rate if sample_rate else 0

        return audio, sample_rate, duration

    finally:
        for path in (src_path, wav_path):
            if path and os.path.exists(path):
                os.remove(path)


def analyze_uploaded_audio(uploaded_file, detector, source="upload"):
    """
    Convert uploaded/recorded audio, save a temporary WAV,
    run the detector and return the result.
    """
    audio, sample_rate, duration = convert_to_16k_mono_wav(uploaded_file)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as wav_file:
        sf.write(
            wav_file.name,
            audio,
            sample_rate,
            subtype="PCM_16"
        )
        temp_wav_path = wav_file.name

    try:
        result = detector.predict(temp_wav_path)

        label = result["label"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        model_fake_probability = get_fake_probability(probabilities)

        # --- Replay / re-recording (speaker-to-mic) detection ---
        replay_analysis = analyze_recording_channel(audio, sample_rate)

        fake_probability = combine_fake_probability_with_replay(
            model_fake_probability,
            replay_analysis["replay_score"],
            source=source
        )

        risk = calculate_risk(fake_probability)

        analysis_result = {
            "timestamp": datetime.now().isoformat(),
            "filename": uploaded_file.name,
            "source": source,
            "duration_seconds": round(duration, 2),
            "sample_rate": sample_rate,
            "label": label,
            "confidence": confidence,
            "model_fake_probability": model_fake_probability,
            "fake_probability": fake_probability,
            "chunks_analyzed": result.get("chunks_analyzed", 1),
            "chunk_scores": result.get("chunk_scores", [model_fake_probability]),
            "replay_score": replay_analysis["replay_score"],
            "replay_snr_db": replay_analysis["snr_estimate_db"],
            "replay_rolloff_hz": replay_analysis["spectral_rolloff_hz"],
            "replay_flag": (
                replay_analysis["replay_score"]
                >= REPLAY_FLAG_DISPLAY_THRESHOLD
            ),
            "risk": risk["risk"],
            "action": risk["action"]
        }

        return analysis_result, probabilities

    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


# ============================================================
# MODEL CLASS
# ============================================================

class VoiceDeepfakeDetector:

    def __init__(self):

        print("Loading AI model...")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Feature extractor
        self.feature_extractor = (
        AutoProcessor.from_pretrained(
            MODEL_NAME
            )
        )
        

        # AI model
        self.model = (
            AutoModelForAudioClassification
            .from_pretrained(MODEL_NAME)
        )

        # Move model to GPU/CPU
        self.model.to(self.device)

        # Evaluation mode
        self.model.eval()

        print("Model loaded successfully!")

        print(
            "Device:",
            self.device
        )

        print(
            "Labels:",
            self.model.config.id2label
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def _predict_waveform(self, audio):
        """Run one inference on a single 16 kHz mono waveform."""
        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        probabilities = probs.cpu().tolist()
        prediction_id = int(torch.argmax(probs).item())
        confidence = float(probs[prediction_id].cpu())
        label = self.model.config.id2label[prediction_id]

        class_probabilities = {}
        for i, probability in enumerate(probabilities):
            class_probabilities[self.model.config.id2label[i]] = probability

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": class_probabilities
        }

    def predict(self, audio_path):
        """
        Robust inference using overlapping short speech windows.
        This reduces the chance that one noisy microphone segment controls
        the complete decision.
        """
        audio, sample_rate = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

        window = int(DETECTION_WINDOW_SEC * 16000)
        hop = int(DETECTION_HOP_SEC * 16000)

        if len(audio) <= window:
            chunks = [audio]
        else:
            chunks = []
            start = 0
            while start < len(audio):
                chunk = audio[start:start + window]
                if len(chunk) < int(1.5 * 16000):
                    break
                chunks.append(chunk)
                start += hop

        chunk_results = [
            self._predict_waveform(chunk)
            for chunk in chunks
        ]

        chunk_fake_scores = [
            get_fake_probability(item["probabilities"])
            for item in chunk_results
        ]

        scores = np.asarray(chunk_fake_scores, dtype=np.float32)

        if len(scores) == 1:
            fake_probability = float(scores[0])
        else:
            median_score = float(np.median(scores))
            upper_half = scores[scores >= median_score]
            upper_mean = float(np.mean(upper_half))
            fake_probability = float(
                0.60 * upper_mean + 0.40 * median_score
            )

        label = "fake" if fake_probability >= 0.50 else "real"
        confidence = max(fake_probability, 1.0 - fake_probability)

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {
                "real": 1.0 - fake_probability,
                "fake": fake_probability
            },
            "chunk_scores": [float(x) for x in scores],
            "chunks_analyzed": len(scores)
        }

# ============================================================
# LANGUAGE IDENTIFICATION
# ============================================================

LANGUAGE_NAMES = {
    "eng": "English",
    "hin": "Hindi",
    "mar": "Marathi",
    "guj": "Gujarati",
    "ben": "Bengali",
    "tam": "Tamil",
    "tel": "Telugu",
    "kan": "Kannada",
    "mal": "Malayalam",
    "pan": "Punjabi",
    "urd": "Urdu",
    "ori": "Odia",
    "asm": "Assamese",
    "nep": "Nepali",
    "san": "Sanskrit",
}

class VoiceLanguageDetector:

    def __init__(self):
        from transformers import AutoModelForAudioClassification

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            LANGUAGE_MODEL_NAME
        )

        self.model = AutoModelForAudioClassification.from_pretrained(
            LANGUAGE_MODEL_NAME
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

    def detect(self, audio, sample_rate=16000):

        inputs = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]

        prediction_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[prediction_id].cpu())

        raw_label = self.model.config.id2label[prediction_id]
        code = str(raw_label).lower().strip()

        language_name = LANGUAGE_NAMES.get(
            code,
            code.upper()
        )

        return {
            "language": language_name,
            "code": code,
            "confidence": confidence
        }


@st.cache_resource
def load_language_detector():
    return VoiceLanguageDetector()


# ============================================================
# FAKE / SCAM CALL DETECTION (SPEECH-TO-TEXT + KEYWORDS)
# ============================================================
#
# This complements the voice-clone detector above. A call can be
# risky either because the VOICE itself is synthetic, or because
# the SCRIPT being spoken matches known scam / vishing patterns
# (OTP fraud, fake KYC, fake bank verification, etc.), even if the
# voice is a real human. Combining both gives a much more reliable
# "Fake Call Risk" than either signal alone.

SCAM_KEYWORDS = {
    "OTP / Verification Fraud": [
        "otp", "one time password", "verification code", "cvv",
        "card number", "expiry date", "pin number", "atm pin",
        "ओटीपी", "व्हेरिफिकेशन कोड", "सीव्हीव्ही"
    ],
    "Account / KYC Threat": [
        "kyc", "account block", "account suspend", "account frozen",
        "account will be blocked", "update your kyc", "aadhar link",
        "pan card link", "केवायसी", "खाते बंद", "अकाउंट ब्लॉक"
    ],
    "Urgency / Pressure": [
        "urgent action", "act immediately", "last warning",
        "within 24 hours", "immediately or", "final notice",
        "त्वरित", "अर्जंट", "लगेच करा"
    ],
    "Financial Request": [
        "transfer money", "send money", "processing fee",
        "refund pending", "pay now", "gift card", "wire transfer",
        "पैसे पाठवा", "फी भरा"
    ],
    "Impersonation": [
        "this is your bank", "calling from rbi", "income tax department",
        "customs department", "police department", "courier department",
        "बँकेतून बोलतोय", "पोलीस स्टेशन"
    ]
}


def detect_scam_keywords(transcript):
    """
    Scan a transcript for known scam / vishing phrases.
    Returns matched categories, matched phrases and a 0-1 scam score
    based on how many distinct risk categories were triggered.
    """
    if not transcript:
        return {
            "matched_categories": [],
            "matched_phrases": [],
            "scam_score": 0.0
        }

    text_lower = transcript.lower()

    matched_categories = []
    matched_phrases = []

    for category, phrases in SCAM_KEYWORDS.items():
        for phrase in phrases:
            if phrase.lower() in text_lower:
                matched_phrases.append(phrase)
                if category not in matched_categories:
                    matched_categories.append(category)
                break

    # Score scales with number of distinct scam categories triggered,
    # capped at 1.0. Hitting 2+ categories (e.g. "OTP" + "urgency")
    # is a much stronger signal than a single stray keyword.
    scam_score = min(len(matched_categories) / 3, 1.0)

    return {
        "matched_categories": matched_categories,
        "matched_phrases": matched_phrases,
        "scam_score": scam_score
    }


def calculate_combined_call_risk(voice_fake_probability, scam_score):
    """
    Combine the voice-clone probability with the scam-script score
    into a single "Fake Call Risk" percentage and verdict.
    """
    combined = (
        (voice_fake_probability * VOICE_RISK_WEIGHT) +
        (scam_score * SCAM_TEXT_RISK_WEIGHT)
    )

    combined_percentage = combined * 100

    if combined_percentage >= 70:
        verdict = "HIGH"
        action = (
            "Terminate the call and do not share any personal, "
            "banking or OTP information. Report the number."
        )
    elif combined_percentage >= 40:
        verdict = "MEDIUM"
        action = (
            "Treat with caution. Verify independently through the "
            "official helpline before taking any action."
        )
    else:
        verdict = "LOW"
        action = (
            "No strong scam indicators found. Still avoid sharing "
            "sensitive information over an unverified call."
        )

    return {
        "combined_score": combined,
        "combined_percentage": combined_percentage,
        "verdict": verdict,
        "action": action
    }


@st.cache_resource
def load_asr_pipeline():
    """
    Lightweight multilingual speech-to-text pipeline used only to
    transcribe the call so we can scan it for scam phrases. This is
    separate from the deepfake voice classifier.
    """
    device_index = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL_NAME,
        device=device_index
    )


def transcribe_audio(audio, sample_rate, asr_pipeline):
    """
    Run speech-to-text on the already-converted (16kHz mono) audio
    and return the plain transcript text. Fails gracefully.
    """
    try:
        result = asr_pipeline(
            {"array": audio, "sampling_rate": sample_rate}
        )
        return result.get("text", "").strip()

    except Exception:
        return ""


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_detector():

    return VoiceDeepfakeDetector()


# ============================================================
# REPLAY / RE-RECORDING (SPEAKER-TO-MIC) ATTACK DETECTION
# ============================================================
#
# PROBLEM THIS SOLVES:
# The deepfake classifier above detects fine-grained digital synthesis
# artifacts (vocoder periodicity, phase discontinuities). If an
# AI-generated clip is played through a phone speaker and re-captured
# through a microphone ("replay attack"), the speaker/mic frequency
# response, room reverberation and added noise floor wash out those
# exact artifacts — so the same fake audio can come back as "Real".
# This is a well-known bypass technique in real vishing/fraud calls.
#
# This module does NOT try to re-detect the original synthesis.
# Instead it looks for tell-tale signs of the *recording channel*
# itself: narrow frequency bandwidth, low high-frequency energy and
# a raised noise floor, all of which are typical of speaker-to-mic
# playback and are used to keep the overall risk score honest even
# when the base classifier is fooled.

def analyze_recording_channel(audio, sample_rate):
    """
    Secondary acoustic check for speaker -> room -> microphone
    re-recording. It is not treated as proof of AI generation.
    """
    if audio is None or len(audio) == 0:
        return {
            "high_freq_ratio": 0.0,
            "spectral_rolloff_hz": 0.0,
            "snr_estimate_db": 0.0,
            "replay_score": 0.0
        }

    audio = np.asarray(audio, dtype=np.float32)
    audio = audio - float(np.mean(audio))

    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    power = np.abs(stft) ** 2
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)

    total_power = float(np.sum(power)) + 1e-12
    high_freq_ratio = float(
        np.sum(power[freqs >= REPLAY_HIGH_FREQ_HZ]) / total_power
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate,
        roll_percent=0.95
    )
    avg_rolloff_hz = float(np.median(rolloff))

    rms = librosa.feature.rms(
        y=audio,
        frame_length=2048,
        hop_length=512
    )[0]
    rms = rms[rms > 1e-5]

    if len(rms) == 0:
        snr_estimate_db = 0.0
    else:
        noise_floor = float(np.percentile(rms, 10)) + 1e-9
        signal_level = float(np.percentile(rms, 90)) + 1e-9
        snr_estimate_db = float(
            20 * np.log10(signal_level / noise_floor)
        )

    # More conservative than the old formula: the old version gave many
    # normal microphone recordings a non-trivial replay score by default.
    bandwidth_score = float(np.clip(
        (4500.0 - avg_rolloff_hz) / 2500.0,
        0.0,
        1.0
    ))

    if snr_estimate_db < 12:
        noise_score = 1.0
    elif snr_estimate_db < 20:
        noise_score = 0.5
    else:
        noise_score = 0.0

    high_freq_score = float(np.clip(
        (0.035 - high_freq_ratio) / 0.035,
        0.0,
        1.0
    ))

    replay_score = float(np.clip(
        0.55 * bandwidth_score +
        0.25 * noise_score +
        0.20 * high_freq_score,
        0.0,
        1.0
    ))

    return {
        "high_freq_ratio": high_freq_ratio,
        "spectral_rolloff_hz": avg_rolloff_hz,
        "snr_estimate_db": snr_estimate_db,
        "replay_score": replay_score
    }


def combine_fake_probability_with_replay(
    model_fake_probability,
    replay_score,
    source="upload"
):
    """
    Combine model evidence with replay evidence.

    For microphone recordings, strong replay evidence becomes a spoof
    security signal. We never lower the model's score.
    """
    model_fake_probability = float(
        np.clip(model_fake_probability, 0.0, 1.0)
    )
    replay_score = float(np.clip(replay_score, 0.0, 1.0))

    if source == "microphone":
        if replay_score >= REPLAY_HIGH_RISK_THRESHOLD:
            combined = max(
                model_fake_probability,
                0.72 + 0.18 * (
                    replay_score - REPLAY_HIGH_RISK_THRESHOLD
                )
            )
        elif replay_score >= REPLAY_MEDIUM_THRESHOLD:
            combined = max(
                model_fake_probability,
                0.45 + 0.20 * (
                    replay_score - REPLAY_MEDIUM_THRESHOLD
                )
            )
        else:
            combined = model_fake_probability
    else:
        combined = model_fake_probability

    return float(np.clip(combined, 0.0, 1.0))


# ============================================================
# RISK ENGINE
# ============================================================

def calculate_risk(fake_probability):

    percentage = fake_probability * 100

    if percentage >= 70:

        return {
            "risk": "HIGH",
            "message": (
                "The audio shows strong indicators "
                "of synthetic or manipulated speech."
            ),
            "action": (
                "Block voice authentication and "
                "request additional verification."
            )
        }

    elif percentage >= 40:

        return {
            "risk": "MEDIUM",
            "message": (
                "The audio contains suspicious "
                "characteristics and should be verified."
            ),
            "action": (
                "Request additional verification "
                "before accepting the voice."
            )
        }

    else:

        return {
            "risk": "LOW",
            "message": (
                "The audio shows relatively low "
                "probability of being AI-generated."
            ),
            "action": (
                "Allow only according to the "
                "application's security policy."
            )
        }


# ============================================================
# FIND FAKE PROBABILITY
# ============================================================

def get_fake_probability(probabilities):

    fake_probability = 0.0

    for label, probability in probabilities.items():

        label_lower = label.lower()

        if (
            "fake" in label_lower
            or "spoof" in label_lower
            or "synthetic" in label_lower
            or "deepfake" in label_lower
        ):

            fake_probability += probability

    return fake_probability


# ============================================================
# SAVE INCIDENT
# ============================================================

def save_incident(result):

    try:

        with open(
            LOG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            logs = json.load(file)

    except (
        FileNotFoundError,
        json.JSONDecodeError
    ):

        logs = []

    logs.append(result)

    with open(
        LOG_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            logs,
            file,
            indent=4
        )


# ============================================================
# LOAD INCIDENT HISTORY
# ============================================================

def load_incidents():

    try:

        with open(
            LOG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (
        FileNotFoundError,
        json.JSONDecodeError
    ):

        return []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ VoiceShield AI")

    st.markdown("---")

    st.subheader("System Status")

    if torch.cuda.is_available():

        st.success(
            "🟢 GPU / CUDA Available"
        )

        st.write(
            f"Device: `{torch.cuda.get_device_name(0)}`"
        )

    else:

        st.info(
            "🔵 CPU Mode"
        )

    st.markdown("---")

    st.subheader("Detection Pipeline")

    st.write(
        "🎙️ Audio Input"
    )

    st.write(
        "↓"
    )

    st.write(
        "🧠 Wav2Vec2 AI Model"
    )

    st.write(
        "↓"
    )

    st.write(
        "📊 Probability Analysis"
    )

    st.write(
        "↓"
    )

    st.write(
        "🚨 Risk Assessment"
    )

    st.write(
        "↓"
    )

    st.write(
        "🛡️ Security Action"
    )

    st.markdown("---")

    st.caption(
        "SIH26 Internal Hackathon Prototype"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ VoiceShield AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Voice Cloning &amp; Fake Call Detection'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TOP INFORMATION CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "🤖 AI Engine",
        "Wav2Vec2"
    )

with col2:

    st.metric(
        "🎙️ Input",
        "Voice Audio"
    )

with col3:

    st.metric(
        "⚡ Processing",
        "AI Analysis"
    )

with col4:

    st.metric(
        "📞 Scam Detection",
        "Voice + Script"
    )


st.markdown("---")


# ============================================================
# LOAD DETECTOR
# ============================================================

with st.spinner(
    "Loading AI detection model..."
):

    detector = load_detector()

with st.spinner(
    "Loading language identification model..."
):

    language_detector = load_language_detector()

with st.spinner(
    "Loading speech-to-text model for scam detection..."
):

    asr_pipeline = load_asr_pipeline()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎙️ Voice Analysis",
        "📡 Live Voice",
        "📊 Incident Dashboard",
        "ℹ️ About System"
    ]
)


# ============================================================
# TAB 1 — VOICE ANALYSIS
# ============================================================

with tab1:

    st.header("🎙️ Analyze Voice Recording")

    st.write(
        "Upload an audio recording and VoiceShield AI will "
        "analyze whether the voice appears genuine or AI-generated, "
        "and scan what was said for known scam / vishing patterns."
    )

    st.info(
        "Supported through FFmpeg: WAV, MP3, M4A, FLAC, OGG, AAC, "
        "WMA, WEBM, OPUS, AIFF and other formats that FFmpeg can decode."
    )

    uploaded_file = st.file_uploader(
        "Upload Voice Recording",
        type=None,
        accept_multiple_files=False
    )

    if uploaded_file is not None:

        st.success(
            f"File uploaded: `{uploaded_file.name}`"
        )

        st.audio(uploaded_file)

        file_size = len(uploaded_file.getvalue()) / 1024

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("File Name", uploaded_file.name)

        with col2:
            st.metric("File Size", f"{file_size:.2f} KB")

        with col3:
            st.metric("Input Type", "File Upload")

        st.markdown("---")

        analyze = st.button(
            "🔍 ANALYZE VOICE",
            type="primary",
            use_container_width=True,
            key="upload_analyze"
        )

if analyze:
    with st.spinner("🧠 AI is analyzing the voice..."):
        try:

            # ====================================================
            # FAST PRIMARY DETECTION
            # ====================================================

            analysis_result, probabilities = (
                analyze_uploaded_audio(
                    uploaded_file,
                    detector,
                    source="file_upload"
                )
            )

            # ====================================================
            # FAST DEMO MODE
            # Language + Whisper ASR are skipped here
            # to reduce CPU usage and response time.
            # ====================================================

            analysis_result["language"] = "Not analyzed"
            analysis_result["language_code"] = "N/A"
            analysis_result["language_confidence"] = 0.0

            analysis_result["transcript"] = ""
            analysis_result["scam_categories"] = []
            analysis_result["scam_phrases"] = []
            analysis_result["scam_score"] = 0.0

            # Fake Call Risk is based primarily on
            # the voice-clone detection score.
            combined_risk = calculate_combined_call_risk(
                analysis_result["fake_probability"],
                0.0
            )

            analysis_result["combined_risk"] = (
                combined_risk["verdict"]
            )

            analysis_result["combined_percentage"] = (
                combined_risk["combined_percentage"]
            )

            analysis_result["combined_action"] = (
                combined_risk["action"]
            )

            # ====================================================
            # SAVE RESULT
            # ====================================================

            st.session_state["analysis_result"] = (
                analysis_result
            )

            st.session_state["probabilities"] = (
                probabilities
            )

            save_incident(analysis_result)

            st.success(
                "✅ Voice analysis completed successfully."
            )

        except FileNotFoundError:
            st.error(
                "❌ FFmpeg was not found. "
                "Please check the FFmpeg configuration."
            )

        except Exception as error:
            st.error(
                f"❌ Could not analyze this file: {error}"
            )      
# ============================================================
# DISPLAY RESULT
# ============================================================

if "analysis_result" in st.session_state:

    result = st.session_state["analysis_result"]

    probabilities = st.session_state["probabilities"]

    st.markdown("---")

    st.header("📊 AI Analysis Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🤖 AI Prediction",
            result["label"]
        )

    with col2:
        st.metric(
            "🎯 Confidence",
            f"{result['confidence'] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "🚨 Fake Probability",
            f"{result['fake_probability'] * 100:.2f}%"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "⏱️ Duration",
            f"{result.get('duration_seconds', 0):.2f} sec"
        )

    with col2:
        st.metric(
            "🎚️ Sample Rate",
            f"{result.get('sample_rate', 16000)} Hz"
        )

    with col3:
        st.metric(
            "📡 Source",
            result.get("source", "unknown")
        )

    st.markdown("---")

    language_col1, language_col2 = st.columns(2)

    with language_col1:
        st.metric(
            "🌍 Detected Language",
            result.get("language", "Unknown")
        )

    with language_col2:
        st.metric(
            "🗣️ Language Confidence",
            f"{result.get('language_confidence', 0) * 100:.2f}%"
        )

    st.markdown("---")

    st.header("🧩 Chunk-Based Detection")
    st.metric(
        "Speech Chunks Analyzed",
        result.get("chunks_analyzed", 1)
    )
    if result.get("chunks_analyzed", 1) > 1:
        st.caption(
            "Overlapping short speech windows are analyzed so a single "
            "noisy microphone segment does not decide the entire result."
        )

    st.header("📻 Recording Channel Analysis (Replay Attack Check)")

    st.write(
        "AI-generated audio played through a speaker and re-recorded "
        "with a microphone can lose the fine digital artifacts the "
        "voice model relies on, making a fake clip look 'real'. This "
        "check looks at the recording channel itself for signs of "
        "that speaker-to-mic playback."
    )

    replay_score_value = result.get("replay_score", 0.0)
    replay_flag_value = result.get("replay_flag", False)

    replay_col1, replay_col2, replay_col3 = st.columns(3)

    with replay_col1:
        st.metric(
            "📊 Replay Likelihood",
            f"{replay_score_value * 100:.1f}%"
        )

    with replay_col2:
        st.metric(
            "🔊 Est. Bandwidth",
            f"{result.get('replay_rolloff_hz', 0):.0f} Hz"
        )

    with replay_col3:
        st.metric(
            "🎚️ Est. SNR",
            f"{result.get('replay_snr_db', 0):.1f} dB"
        )

    if replay_flag_value:
        st.warning(
            "⚠️ This recording shows meaningful signs of speaker-to-microphone "
            "through a speaker and re-captured by a microphone "
            "(reduced bandwidth / raised noise floor). Even a modest "
            "amount of this, combined with the voice model's own "
            "score, is enough to raise the overall risk — since "
            "replaying a cloned voice this way is a common way to "
            "dodge detection."
        )
    else:
        st.info(
            "No meaningful replay / re-recording signs detected in "
            "this recording's channel characteristics."
        )

    model_score_value = result.get("model_fake_probability", 0)
    final_score_value = result.get("fake_probability", 0)

    if abs(final_score_value - model_score_value) > 0.005:
        st.caption(
            f"Voice model alone said "
            f"{model_score_value * 100:.2f}% fake → combined with "
            f"the channel analysis, the final score used for risk "
            f"is {final_score_value * 100:.2f}%."
        )
    else:
        st.caption(
            f"Channel analysis found no meaningful replay signs, so "
            f"the final score matches the voice model's own "
            f"{model_score_value * 100:.2f}%."
        )

    st.markdown("---")

    st.header("🚨 Security Assessment")

    if result["risk"] == "HIGH":

        st.error("🔴 HIGH RISK")

        st.write(
            "The voice has a high probability of being "
            "AI-generated or manipulated."
        )

        st.error(
            "🚫 Recommended Action: "
            "BLOCK / REQUEST STRONG VERIFICATION"
        )

    elif result["risk"] == "MEDIUM":

        st.warning("🟡 MEDIUM RISK")

        st.write(
            "The voice contains suspicious characteristics."
        )

        st.warning(
            "⚠️ Recommended Action: "
            "REQUEST ADDITIONAL VERIFICATION"
        )

    else:

        st.success("🟢 LOW RISK")

        st.write(
            "The voice has relatively low probability "
            "of being AI-generated."
        )

        st.success(
            "✅ Recommended Action: "
            "ALLOW SUBJECT TO SECURITY POLICY"
        )

    if (
        result.get("source") == "microphone"
        and result.get("replay_score", 0.0) >= REPLAY_HIGH_RISK_THRESHOLD
    ):
        st.error(
            "🛡️ PRESENTATION-ATTACK ALERT: Strong speaker-to-microphone "
            "replay characteristics detected. Treat this as "
            "SPOOF / REPLAY-SUSPECTED and require independent verification."
        )

    st.markdown("---")

    st.header("📞 Fake Call Detection (Voice + Script)")

    st.write(
        "This score combines the AI voice-clone probability above "
        "with a scan of what was actually *said*, so a scam call can "
        "be flagged even when the voice itself sounds like a real human."
    )

    combined_risk_value = result.get("combined_risk", "LOW")
    combined_percentage_value = result.get("combined_percentage", 0)

    call_col1, call_col2 = st.columns(2)

    with call_col1:
        st.metric(
            "🚨 Fake Call Risk",
            combined_risk_value
        )

    with call_col2:
        st.metric(
            "📊 Combined Risk Score",
            f"{combined_percentage_value:.2f}%"
        )

    scam_categories = result.get("scam_categories", [])

    if scam_categories:

        risk_css_class = (
            "high-risk" if combined_risk_value == "HIGH"
            else "medium-risk" if combined_risk_value == "MEDIUM"
            else "low-risk"
        )

        st.markdown(
            f'<div class="risk-box {risk_css_class}">'
            f'⚠️ Scam-pattern categories detected: '
            f'<b>{", ".join(scam_categories)}</b>'
            f'</div>',
            unsafe_allow_html=True
        )

    else:
        st.info(
            "No known scam-script patterns were detected in the "
            "transcribed speech."
        )

    st.write(
        f"**Recommended Action:** {result.get('combined_action', '')}"
    )

    with st.expander("📝 View Call Transcript & Matched Phrases"):

        transcript_text = result.get("transcript", "")

        st.write("**Transcript (speech-to-text):**")

        st.write(
            transcript_text if transcript_text
            else "_No speech could be transcribed from this audio._"
        )

        matched_phrases = result.get("scam_phrases", [])

        if matched_phrases:

            st.write("**Matched scam phrases:**")

            for phrase in matched_phrases:
                st.write(f"• {phrase}")

    st.markdown("---")

    st.header("📈 AI Probability Distribution")

    chart_data = pd.DataFrame(
        {
            "Class": list(probabilities.keys()),
            "Probability": [
                value * 100
                for value in probabilities.values()
            ]
        }
    )

    st.bar_chart(
        chart_data.set_index("Class")
    )

    with st.expander("🔎 View Detailed AI Probabilities"):

        for label_name, probability in probabilities.items():

            st.write(f"**{label_name}**")

            st.progress(probability)

            st.caption(
                f"{probability * 100:.2f}%"
            )

    st.markdown("---")

    st.header("📋 Security Report")

    report_text = json.dumps(
        result,
        indent=4
    )

    st.download_button(
        label="📥 Download Analysis Report",
        data=report_text,
        file_name="voiceshield_report.json",
        mime="application/json"
    )

    if st.button(
        "🧹 Clear Current Analysis",
        key="clear_analysis"
    ):

        st.session_state.pop(
            "analysis_result",
            None
        )

        st.session_state.pop(
            "probabilities",
            None
        )

        st.rerun()


# ============================================================
# TAB 2 — LIVE MICROPHONE VOICE
# ============================================================

with tab2:

    st.header("📡 Live Voice Analysis")

    st.write(
        "Record a voice sample directly from your microphone. "
        "After you stop recording, VoiceShield AI analyzes it immediately."
    )

    st.info(
        "Hackathon note: this is near-real-time recording → analysis. "
        "Continuous chunk-by-chunk streaming can be added later with WebRTC."
    )

    live_audio = st.audio_input(
        "🎙️ Start a microphone recording"           # ithe change kela ahe
    )

    if live_audio is not None:

        st.success("✅ Voice recording captured.")

        st.audio(live_audio)

        if st.button(
            "🧠 ANALYZE LIVE VOICE",
            type="primary",
            use_container_width=True,
            key="live_analyze"
        ):

            with st.spinner(
                "🔴 Analyzing microphone recording..."
            ):

                try:

                    analysis_result, probabilities = (
                        analyze_uploaded_audio(
                            live_audio,
                            detector,
                            source="microphone"
                        )
                    )

                    converted_audio, converted_sr, _ = (
                        convert_to_16k_mono_wav(live_audio)
                    )

                    language_result = language_detector.detect(
                        converted_audio,
                        converted_sr
                    )

                    analysis_result["language"] = (
                        language_result["language"]
                    )
                    analysis_result["language_code"] = (
                        language_result["code"]
                    )
                    analysis_result["language_confidence"] = (
                        language_result["confidence"]
                    )

                    # --- Fake / scam call detection (speech-to-text) ---
                    transcript = transcribe_audio(
                        converted_audio,
                        converted_sr,
                        asr_pipeline
                    )

                    scam_result = detect_scam_keywords(transcript)

                    combined_risk = calculate_combined_call_risk(
                        analysis_result["fake_probability"],
                        scam_result["scam_score"]
                    )

                    analysis_result["transcript"] = transcript
                    analysis_result["scam_categories"] = (
                        scam_result["matched_categories"]
                    )
                    analysis_result["scam_phrases"] = (
                        scam_result["matched_phrases"]
                    )
                    analysis_result["scam_score"] = (
                        scam_result["scam_score"]
                    )
                    analysis_result["combined_risk"] = (
                        combined_risk["verdict"]
                    )
                    analysis_result["combined_percentage"] = (
                        combined_risk["combined_percentage"]
                    )
                    analysis_result["combined_action"] = (
                        combined_risk["action"]
                    )

                    st.session_state["analysis_result"] = (
                        analysis_result
                    )

                    st.session_state["probabilities"] = (
                        probabilities
                    )

                    save_incident(analysis_result)

                    st.success(
                        "✅ Live voice analysis completed."
                    )

                    st.metric(
                        "Risk Level",
                        analysis_result["risk"]
                    )

                    st.metric(
                        "Fake Probability",
                        f"{analysis_result['fake_probability'] * 100:.2f}%"
                    )

                    st.metric(
                        "📞 Fake Call Risk",
                        analysis_result["combined_risk"]
                    )

                except FileNotFoundError:
                    st.error(
                        "❌ FFmpeg was not found. "
                        "Install FFmpeg and restart the terminal."
                    )

                except Exception as error:
                    st.error(
                        f"❌ Could not analyze microphone audio: {error}"
                    )


# ============================================================
# TAB 3 — INCIDENT DASHBOARD
# ============================================================

with tab3:

    st.header(
        "📊 Incident Dashboard"
    )

    incidents = load_incidents()


    if len(incidents) == 0:

        st.info(
            "No voice analysis incidents recorded yet."
        )

    else:

        total = len(
            incidents
        )

        high_risk = sum(
            1
            for item in incidents
            if item["risk"] == "HIGH"
        )

        medium_risk = sum(
            1
            for item in incidents
            if item["risk"] == "MEDIUM"
        )

        low_risk = sum(
            1
            for item in incidents
            if item["risk"] == "LOW"
        )

        fake_call_high_risk = sum(
            1
            for item in incidents
            if item.get("combined_risk") == "HIGH"
        )


        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:

            st.metric(
                "Total Analyses",
                total
            )

        with col2:

            st.metric(
                "🔴 High Risk",
                high_risk
            )

        with col3:

            st.metric(
                "🟡 Medium Risk",
                medium_risk
            )

        with col4:

            st.metric(
                "🟢 Low Risk",
                low_risk
            )

        with col5:

            st.metric(
                "📞 High-Risk Calls",
                fake_call_high_risk
            )


        st.markdown("---")


        # Incident table

        table_data = []

        for incident in reversed(
            incidents
        ):

            table_data.append(
                {
                    "Time":
                        incident.get(
                            "timestamp",
                            ""
                        ),

                    "File":
                        incident.get(
                            "filename",
                            ""
                        ),

                    "Prediction":
                        incident.get(
                            "label",
                            ""
                        ),

                    "Language":
                        incident.get(
                            "language",
                            "Unknown"
                        ),

                    "Language Confidence":
                        f"{incident.get('language_confidence', 0) * 100:.2f}%",

                    "Fake Probability":
                        f"{incident.get('fake_probability', 0) * 100:.2f}%",

                    "Risk":
                        incident.get(
                            "risk",
                            ""
                        ),

                    "Fake Call Risk":
                        incident.get(
                            "combined_risk",
                            "N/A"
                        ),

                    "Replay Detected":
                        "⚠️ Yes" if incident.get("replay_flag") else "No"
                }
            )


        df = pd.DataFrame(
            table_data
        )


        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


        # Clear history

        if st.button(
            "🗑️ Clear Incident History"
        ):

            with open(
                LOG_FILE,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    [],
                    file
                )

            st.success(
                "Incident history cleared."
            )

            st.rerun()


# ============================================================
# TAB 4 — ABOUT
# ============================================================

with tab4:

    st.header(
        "ℹ️ About VoiceShield AI"
    )

    st.write(
        """
        **VoiceShield AI** is a prototype designed
        to detect potentially AI-generated or cloned
        voice recordings.

        The system uses a Wav2Vec2-based audio
        classification model to analyze the uploaded
        speech signal.
        """
    )


    st.subheader(
        "🧠 Detection Pipeline"
    )

    pipeline_steps = [
        "1️⃣ User uploads or records a voice call",
        "2️⃣ Audio is converted to 16 kHz mono",
        "3️⃣ Spoken language is automatically identified",
        "4️⃣ Wav2Vec2 feature extractor processes the audio",
        "5️⃣ Deep-learning model predicts audio class",
        "6️⃣ Class probabilities are calculated",
        "7️⃣ Fake (voice-clone) probability is estimated",
        "8️⃣ Recording channel is checked for speaker-to-mic replay",
        "9️⃣ Fake probability is adjusted if replay signs are found",
        "🔟 Speech is transcribed with a speech-to-text model",
        "1️⃣1️⃣ Transcript is scanned for scam / vishing phrases",
        "1️⃣2️⃣ Voice score + scam-script score are combined into a Fake Call Risk",
        "1️⃣3️⃣ Security risk & recommended action are generated",
        "1️⃣4️⃣ Incident is stored in the dashboard"
    ]


    for step in pipeline_steps:

        st.write(
            step
        )


    st.subheader(
        "🛡️ Possible Applications"
    )

    applications = [
        "Banking voice authentication",
        "Customer support fraud detection",
        "Call-center security",
        "Executive impersonation detection",
        "Voice phishing (vishing) & fake call detection",
        "Digital identity protection"
    ]


    for application in applications:

        st.write(
            f"• {application}"
        )


    st.warning(
        "⚠️ This is a hackathon prototype. "
        "AI audio detection can produce false positives "
        "and false negatives, so it should not be treated "
        "as the sole authentication mechanism."
    )

    st.markdown("---")

    st.subheader("🚀 Realistic Future Roadmap")

    future_features = [
        "📡 Continuous chunk-based live detection (real streaming calls)",
        "📊 Rolling risk score across multiple voice segments",
        "🔔 Configurable security alerts for high-risk events (SMS/email)",
        "🧾 Incident IDs, timestamps and audit history",
        "🎚️ Model confidence calibration and threshold tuning",
        "🧠 Multi-model ensemble for stronger voice detection",
        "🎙️ ML-based replay/anti-spoofing model (CQCC/RawNet2) to replace the current heuristic",
        "🔐 Privacy controls: retention and deletion of recordings",
        "👨‍💼 Human-review workflow for suspicious calls",
        "📈 False-positive / false-negative monitoring",
        "📇 Caller-ID / number spoofing checks (telecom integration)",
        "🌐 Deployment as an API for banking or call-center systems"
    ]

    for feature in future_features:
        st.write(f"• {feature}")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🛡️ VoiceShield AI | SIH26 Internal Hackathon Prototype by Team Matrix"
)

st.caption(   
    "AI-powered voice cloning impersonation detection"
)
