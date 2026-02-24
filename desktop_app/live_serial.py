import re
import time
import threading
import queue
from typing import Optional, Deque, Tuple
from collections import deque

import numpy as np
import serial

from desktop_app.io_sources import BeatSample, ensure_187


class SerialECGSource:
    
    def __init__(
        self,
        port: str,
        baud: int = 115200,

        # IMPORTANT: effective sample rate reaching Python
        # FS=150 and PRINT_EVERY_N=1 => 150
        sample_rate: int = 150,

        beat_level: float = 400.0,
        refractory_ms: int = 300,
        max_buf: int = 6000,
    ):
        self.port = port
        self.baud = baud
        self.sample_rate = int(sample_rate)
        self.beat_level = float(beat_level)
        self.refractory_ms = int(refractory_ms)
        self.max_buf = int(max_buf)

        # fixed beat window length = 187
        self.PRE = 60
        self.POST = 127  # PRE + POST = 187

        self.q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=60)
        self.stop_flag = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.ser: Optional[serial.Serial] = None

        # Supports lines with BPM or without BPM
        self.pattern_full = re.compile(
            r"DET:\s*([-+]?\d*\.?\d+)\s*,\s*DIS:\s*([-+]?\d*\.?\d+)\s*,\s*BPM:\s*([-+]?\d*\.?\d+)\s*,\s*BEAT:\s*([-+]?\d*\.?\d+)",
            re.IGNORECASE,
        )
        self.pattern_no_bpm = re.compile(
            r"DET:\s*([-+]?\d*\.?\d+)\s*,\s*DIS:\s*([-+]?\d*\.?\d+)\s*,\s*BEAT:\s*([-+]?\d*\.?\d+)",
            re.IGNORECASE,
        )

        self.sig_buf: list[float] = []
        self.base_idx = 0
        self.abs_i = 0

        self.pending_markers: Deque[int] = deque()
        self._prev_marker = False
        self._last_marker_i = -10**9

        self.last_bpm: float = 0.0
        self._prev_true_marker_i: Optional[int] = None

        self.index = 0
        self._lock = threading.Lock()

    def start(self):
        if self.thread:
            return
        self.stop_flag.clear()
        self.ser = serial.Serial(self.port, self.baud, timeout=1)
        time.sleep(1.0)
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        self.thread = None
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

    def _trim_buf(self):
        if len(self.sig_buf) <= self.max_buf:
            return
        keep = self.max_buf // 2
        drop = len(self.sig_buf) - keep
        if drop > 0:
            self.sig_buf = self.sig_buf[drop:]
            self.base_idx += drop

    def _parse_line(self, line: str) -> Optional[Tuple[float, float, Optional[float]]]:
        m = self.pattern_full.search(line)
        if m:
            det = float(m.group(1))
            bpm = float(m.group(3))
            beat = float(m.group(4))
            return det, beat, bpm

        m2 = self.pattern_no_bpm.search(line)
        if m2:
            det = float(m2.group(1))
            beat = float(m2.group(3))
            return det, beat, None

        return None

    def _put_beat(self, beat: np.ndarray):
        beat = ensure_187(beat)
        try:
            self.q.put_nowait(beat)
        except queue.Full:
            try:
                _ = self.q.get_nowait()
                self.q.put_nowait(beat)
            except Exception:
                pass

    def _update_bpm_from_arduino(self, bpm: Optional[float]):
        if bpm is None:
            return
        if 30.0 <= bpm <= 220.0:
            if self.last_bpm <= 0:
                self.last_bpm = float(bpm)
            else:
                # smoothing to reduce jitter
                self.last_bpm = float(0.85 * self.last_bpm + 0.15 * bpm)

    def _update_bpm_from_markers(self, marker_i: int):
        if self._prev_true_marker_i is None:
            self._prev_true_marker_i = marker_i
            return
        di = marker_i - self._prev_true_marker_i
        self._prev_true_marker_i = marker_i
        if di <= 0:
            return

        sr = max(self.sample_rate, 1)
        dt = di / float(sr)
        if dt <= 0:
            return

        bpm = 60.0 / dt
        if 30.0 <= bpm <= 220.0:
            if self.last_bpm <= 0:
                self.last_bpm = float(bpm)
            else:
                self.last_bpm = float(0.85 * self.last_bpm + 0.15 * bpm)

    def _try_emit_pending(self):
        while self.pending_markers:
            m = self.pending_markers[0]

            start = m - self.PRE
            end = m + self.POST  # exclusive

            if start < self.base_idx:
                self.pending_markers.popleft()
                continue

            if end > (self.base_idx + len(self.sig_buf)):
                break

            s = start - self.base_idx
            e = end - self.base_idx

            with self._lock:
                seg = np.asarray(self.sig_buf[s:e], dtype=np.float32)

            self._put_beat(seg)
            self.pending_markers.popleft()

    def _read_loop(self):
        while not self.stop_flag.is_set():
            try:
                line = self.ser.readline().decode(errors="ignore").strip()  # type: ignore
            except Exception:
                continue

            parsed = self._parse_line(line)
            if parsed is None:
                continue

            det, beat, bpm = parsed

            cur_i = self.abs_i
            self.abs_i += 1

            with self._lock:
                self.sig_buf.append(float(det))
                self._trim_buf()

            # update bpm (prefer Arduino bpm if exists)
            self._update_bpm_from_arduino(bpm)

            # marker detection (rising edge)
            is_marker = (beat >= self.beat_level)
            if is_marker and (not self._prev_marker):
                ref_samples = max(1, int((self.refractory_ms / 1000.0) * max(self.sample_rate, 1)))
                if (cur_i - self._last_marker_i) >= ref_samples:
                    self.pending_markers.append(cur_i)
                    self._last_marker_i = cur_i

                    # fallback bpm if Arduino bpm missing
                    if bpm is None:
                        self._update_bpm_from_markers(cur_i)

            self._prev_marker = is_marker
            self._try_emit_pending()

    def next(self) -> BeatSample:
        if not self.thread:
            self.start()

        try:
            beat = self.q.get(timeout=6.0)
        except queue.Empty:
            raise RuntimeError("No BEAT received yet. Check electrodes / marker output / sample_rate.")

        s = BeatSample(beat=beat, true_label=None, index=self.index, source="SERIAL")
        self.index += 1
        return s

    def try_next(self) -> Optional[BeatSample]:
        if not self.thread:
            self.start()

        try:
            beat = self.q.get_nowait()
        except queue.Empty:
            return None

        s = BeatSample(beat=beat, true_label=None, index=self.index, source="SERIAL")
        self.index += 1
        return s

    def get_recent_signal(self, seconds: float = 5.0) -> np.ndarray:
        sr = max(self.sample_rate, 1)
        n = int(max(200, sr * float(seconds)))
        with self._lock:
            if len(self.sig_buf) == 0:
                return np.zeros(0, dtype=np.float32)
            seg = np.asarray(self.sig_buf[-min(n, len(self.sig_buf)):], dtype=np.float32)
        return seg