"""
Thread-safe, asynchronous voice coaching helper using pyttsx3 offline TTS engine.
Designed specifically for Windows compatibility by managing COM threading safely.
"""

import pyttsx3
import threading
import queue
import time


class VoiceCoach:
    """Delivers speech coaching feedback on a daemon thread to prevent UI freezing."""

    def __init__(self) -> None:
        self.voice_queue = queue.Queue()
        self.last_alert_time = {}
        self.running = True

        # Start the voice coaching daemon worker
        self.worker_thread = threading.Thread(
            target=self._voice_worker,
            daemon=True
        )
        self.worker_thread.start()

    def _voice_worker(self) -> None:
        """
        Worker loop that listens for speech messages in the queue.
        Initializes the pyttsx3 engine inside this thread to resolve Windows COM threading issues.
        """
        # Initialize engine inside the worker thread to bind it to this thread context (COM safety on Windows)
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 155)  # Slightly slower for optimal clarity
            # Attempt to select a female voice if available, otherwise defaults
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                # Often index 1 is a female voice (e.g. Zira on Windows)
                engine.setProperty('voice', voices[1].id)
        except Exception as e:
            print(f"[VOICE CORE] Error initializing pyttsx3: {e}")
            engine = None

        while self.running:
            try:
                # Wait for next alert message (blocking with timeout to allow graceful stop checks)
                message = self.voice_queue.get(timeout=1.0)
                if engine and message:
                    engine.say(message)
                    engine.runAndWait()
                self.voice_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VOICE CORE] Error in speech output: {e}")

    def alert(self, key: str, message: str, cooldown: float = 6.0) -> bool:
        """
        Enqueues a voice feedback suggestion if the cooldown period has expired for this key.
        Prevents repeating the same correction too frequently.
        """
        current_time = time.time()

        if key not in self.last_alert_time:
            self.last_alert_time[key] = 0.0

        if current_time - self.last_alert_time[key] > cooldown:
            print(f"[VOICE ALERT] {message}")
            self.voice_queue.put(message)
            self.last_alert_time[key] = current_time
            return True

        return False

    def stop(self) -> None:
        """Stops the speech queue gracefully."""
        self.running = False