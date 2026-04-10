"""
Voice coach: plays spoken corrections via Gradium TTS.

Architecture (mirrors faceguard)
---------------------------------
- A background worker thread serialises all speech so clips never overlap.
- Per-issue cooldowns prevent the same note from repeating too quickly.
- The queue is capped at 1: if a clip is already queued, new alerts are dropped
  rather than stacking up.
- Gradium's async `tts()` is called via `asyncio.run()` inside the worker thread.
- The resulting WAV bytes are saved to a temp file and played with `afplay`.
- Requires GRADIUM_API_KEY to be set; raises RuntimeError on startup otherwise.
"""

import asyncio
import os
import queue
import subprocess
import tempfile
import threading
import time

import gradium.client

from config import GRADIUM_VOICE_ID


class VoiceCoach:
    """
    Thread-safe voice alert manager backed exclusively by Gradium TTS.

    Requires GRADIUM_API_KEY to be set in the environment; raises RuntimeError
    at construction time if the key is absent or the client cannot be created.

    Usage
    -----
    coach = VoiceCoach()
    coach.alert("hip_low", "Tighten your core!", cooldown=4.0)
    ...
    coach.stop()
    """

    def __init__(self) -> None:
        self._gradium_client = self._init_gradium()
        self._voice_id = GRADIUM_VOICE_ID or None

        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._cooldowns: dict[str, float] = {}
        self._lock = threading.Lock()

        self._worker = threading.Thread(
            target=self._run, daemon=True, name="voice-worker"
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _init_gradium() -> gradium.client.GradiumClient:
        """
        Create and return a Gradium client.  Raises RuntimeError if the API key
        is missing or the client cannot be initialised.
        """
        api_key = os.environ.get("GRADIUM_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "[VoiceCoach] GRADIUM_API_KEY is not set. "
                "Export GRADIUM_API_KEY=<your-key> before running."
            )
        try:
            client = gradium.client.GradiumClient(api_key=api_key)
            print("[VoiceCoach] Gradium TTS ready.")
            return client
        except Exception as exc:
            raise RuntimeError(f"[VoiceCoach] Gradium init failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def alert(self, issue_key: str, message: str, cooldown: float = 4.0) -> bool:
        """
        Attempt to speak *message* if *issue_key* is not in cooldown.

        Returns True if the message was queued, False if suppressed.
        """
        now = time.monotonic()
        with self._lock:
            if now - self._cooldowns.get(issue_key, 0.0) < cooldown:
                return False
            self._cooldowns[issue_key] = now
        try:
            self._queue.put_nowait(message)
            return True
        except queue.Full:
            return False

    def stop(self) -> None:
        """Signal the worker to stop after the current clip finishes."""
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            self._speak(item)

    def _speak(self, text: str) -> None:
        try:
            self._speak_gradium(text)
        except Exception as exc:
            print(f"[VoiceCoach] Gradium error (utterance skipped): {exc}")

    # ------------------------------------------------------------------
    # Gradium backend
    # ------------------------------------------------------------------

    def _speak_gradium(self, text: str) -> None:
        """Call Gradium TTS, write the WAV to a temp file, play with afplay."""
        setup = {
            "model_name": "default",
            "output_format": "wav",
        }
        if self._voice_id:
            setup["voice_id"] = self._voice_id

        result = asyncio.run(self._gradium_client.tts(setup=setup, text=text))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(result.raw_data)
            tmp_path = tmp.name

        try:
            subprocess.run(
                ["afplay", tmp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        finally:
            os.unlink(tmp_path)

