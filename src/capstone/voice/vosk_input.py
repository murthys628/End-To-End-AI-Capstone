from __future__ import annotations

import json
import os
from pathlib import Path


def listen_once() -> str:
    """Transcribe one utterance locally; no audio is sent to a remote service."""
    try:
        import sounddevice as sd
        from vosk import KaldiRecognizer, Model
    except ImportError as exc:
        raise RuntimeError(
            "Voice input needs the optional dependencies. Install with "
            "'pip install -e .[voice]'."
        ) from exc

    model_path = os.environ.get("CAPSTONE_VOSK_MODEL")
    if not model_path or not Path(model_path).is_dir():
        raise RuntimeError(
            "Set CAPSTONE_VOSK_MODEL to a downloaded Vosk model directory "
            "to enable offline voice input."
        )

    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)
    print("Listening (offline). Speak a command; press Ctrl+C to cancel.")
    audio = sd.rec(
        int(5 * 16000),
        samplerate=16000,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    recognizer.AcceptWaveform(audio.tobytes())
    result = json.loads(recognizer.Result())
    return result.get("text", "").strip()
