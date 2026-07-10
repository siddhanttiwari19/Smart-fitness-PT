from __future__ import annotations

import logging
import platform
import queue
import shutil
import subprocess
import threading
import time
from typing import Callable, Optional

from config import ISSUE_MESSAGES

log = logging.getLogger(__name__)


class VoiceCoach:
    """
    Thread-safe voice alerts backed by the OS TTS binary.

    Usage
    -----
        coach = VoiceCoach()
        coach.alert("Go deeper", cooldown=4.0)
        ...
        coach.stop()
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._build_cmd: Optional[Callable[[str], list[str]]] = self._detect_backend()
        self.available = self._build_cmd is not None

        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=1)
        self._cooldowns: dict[str, float] = {}
        self._msg_cycle: dict[str, int]   = {k: 0 for k in ISSUE_MESSAGES}
        self._lock = threading.Lock()
        self._current_proc: Optional[subprocess.Popen] = None
        self._stop_flag = False

        if self.available:
            self._worker = threading.Thread(
                target=self._run, name="voice-worker", daemon=True
            )
            self._worker.start()
            log.info("VoiceCoach ready (%s)", platform.system())
            print(f"[VoiceCoach] Ready ({platform.system()} TTS).")
        else:
            log.warning("No OS TTS backend found — voice disabled")
            print("[VoiceCoach] No TTS backend detected — voice disabled.")

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_backend() -> Optional[Callable[[str], list[str]]]:
        system = platform.system()
        if system == "Darwin" and shutil.which("say"):
            return lambda t: ["say", "-r", "185", t]
        if system == "Linux":
            for binary in ("espeak-ng", "espeak", "spd-say"):
                if shutil.which(binary):
                    return lambda t, b=binary: [b, t]
        if system == "Windows":
            def build(t: str) -> list[str]:
                safe = t.replace('"', "'").replace("\n", " ")
                ps = (
                    "Add-Type -AssemblyName System.Speech; "
                    f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{safe}")'
                )
                return ["powershell", "-NoProfile", "-Command", ps]
            return build
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def alert(self, issue_key: str, cooldown: float = 4.0) -> bool:
        """
        Queue a spoken alert for *issue_key* if not in cooldown.
        Returns True when queued, False when suppressed.
        """
        if not self._enabled or not self.available:
            return False

        now = time.monotonic()
        with self._lock:
            if now - self._cooldowns.get(issue_key, 0.0) < cooldown:
                return False
            self._cooldowns[issue_key] = now

        message = self._next_message(issue_key)
        try:
            # Drop the slot's old message so we always speak the freshest cue
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put_nowait(message)
            return True
        except queue.Full:
            return False

    def stop(self) -> None:
        """Graceful shutdown — kill any in-flight utterance then end worker."""
        self._stop_flag = True
        self._kill_current()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _next_message(self, key: str) -> str:
        msgs = ISSUE_MESSAGES.get(key)
        if not msgs:
            return key
        idx = self._msg_cycle.get(key, 0) % len(msgs)
        self._msg_cycle[key] = idx + 1
        return msgs[idx]

    def _kill_current(self) -> None:
        proc = self._current_proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=0.3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass
        self._current_proc = None

    def _run(self) -> None:
        while not self._stop_flag:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            self._kill_current()

            try:
                cmd = self._build_cmd(item)   # type: ignore[misc]
                self._current_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
                self._current_proc.wait()   # blocks worker only, never caller
            except FileNotFoundError:
                log.warning("TTS binary vanished — disabling voice")
                self.available = False
                break
            except Exception as exc:
                log.warning("TTS error: %s", exc)
            finally:
                self._current_proc = None
