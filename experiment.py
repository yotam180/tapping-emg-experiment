#!/usr/bin/env python3
"""
Predictability Experiment

Visual cues (white background, filled black):
  ▲ = Predictable High (PH, 988 Hz)
  ▼ = Predictable Low  (PL, 220 Hz)
  ● = Mixed            (MIX, 50/50 pseudo-random)

LSL streams (same schema as duino_bridge_minimal.py):
  FSR_force  – float32, ~830 Hz continuous
  Markers    – string,  irregular rate

Usage:
    python experiment.py block [options]

Options:
    --condition {PL,PH,MIX}   Auditory condition          (default: MIX)
    --port PORT                Serial port                 (default: COM3)
    --trials N                 Trials per block            (default: 60)
    --threshold F              Press-detection threshold   (default: 40)
    --participant N            Participant ID in LSL IDs   (default: 0)
    --data-dir DIR             Root directory for CSV output
                               (default: <script dir>/data)

Quit any time with Q or Escape.
"""

import argparse
import csv
import random
import threading
import time
import tkinter as tk
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import serial
from pylsl import IRREGULAR_RATE, StreamInfo, StreamOutlet, local_clock

# PsychoPy audio – set backend before importing sound; fall back to sounddevice
try:
    from psychopy import prefs as _pp_prefs
    _pp_prefs.hardware['audioLib'] = ['sounddevice']
    from psychopy import sound as _pp_sound
    _USE_PSYCHOPY = True
except Exception as _exc:
    print(f"[warn] PsychoPy unavailable ({_exc}); using sounddevice fallback")
    _USE_PSYCHOPY = False
    import sounddevice as _sd

# ── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE        = 830        # FSR nominal rate pushed to LSL (Hz)
BAUD_RATE          = 115200
DEFAULT_PORT       = "COM3"
DEFAULT_TRIALS     = 60
DEFAULT_THRESHOLD  = 40.0

CUE_DURATION_S     = 0.300      # Symbol display time
AUDIO_DELAY_S      = 0.020      # Auditory feedback delay after press
RESPONSE_TIMEOUT_S = 4.0        # Max wait for press after cue onset
ITI_MIN_S          = 2.5        # Fixation (inter-trial interval)
ITI_MAX_S          = 4.5

TONE_FREQ_LOW      = 220        # Hz – PL
TONE_FREQ_HIGH     = 988        # Hz – PH
TONE_DURATION_S    = 0.050      # 50 ms
AUDIO_FS           = 44100      # Fallback sample rate

DEFAULT_DATA_DIR   = Path(__file__).parent / "data"

_CSV_FIELDS = [
    "trial", "condition", "tone",
    "cue_t_lsl", "press_t_lsl", "force", "response_time_s", "outcome",
]


class Condition(Enum):
    PL  = "PL"
    PH  = "PH"
    MIX = "MIX"


# ── LSL ──────────────────────────────────────────────────────────────────────

def create_outlets(participant_id: int = 0):
    uid = f"p{participant_id}" if participant_id else "exp"
    fsr = StreamOutlet(
        StreamInfo("FSR_force", "FSR", 1, SAMPLE_RATE, "float32", f"fsr_arduino_{uid}")
    )
    markers = StreamOutlet(
        StreamInfo("Markers", "Markers", 1, IRREGULAR_RATE, "string", f"arduino_bridge_{uid}")
    )
    return fsr, markers


# ── Audio ─────────────────────────────────────────────────────────────────────

def _make_tone(freq: float, duration_s: float, fs: int = AUDIO_FS) -> np.ndarray:
    t = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)
    buf = np.sin(2 * np.pi * freq * t).astype(np.float32)
    fade = int(fs * 0.005)
    buf[:fade]  *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    buf[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return buf


class AudioEngine:
    """Pre-loads both tones; plays with a configurable delay (non-blocking)."""

    def __init__(self):
        if _USE_PSYCHOPY:
            self._snd_low  = _pp_sound.Sound(TONE_FREQ_LOW,  secs=TONE_DURATION_S,
                                              stereo=True, hamming=True)
            self._snd_high = _pp_sound.Sound(TONE_FREQ_HIGH, secs=TONE_DURATION_S,
                                              stereo=True, hamming=True)
        else:
            self._buf_low  = _make_tone(TONE_FREQ_LOW,  TONE_DURATION_S)
            self._buf_high = _make_tone(TONE_FREQ_HIGH, TONE_DURATION_S)

    def _play_now(self, tone: str, markers: StreamOutlet):
        marker = "sound_PL" if tone == "low" else "sound_PH"
        if _USE_PSYCHOPY:
            snd = self._snd_low if tone == "low" else self._snd_high
            snd.play()
        else:
            buf = self._buf_low if tone == "low" else self._buf_high
            _sd.play(buf, AUDIO_FS)
        markers.push_sample([marker], local_clock())

    def play(self, tone: str, markers: StreamOutlet, delay_s: float = AUDIO_DELAY_S):
        """Schedule tone playback after delay_s seconds (returns immediately)."""
        t = threading.Timer(delay_s, self._play_now, args=(tone, markers))
        t.daemon = True
        t.start()


# ── Display ───────────────────────────────────────────────────────────────────

class ExperimentDisplay:
    """Fullscreen white window; draws fixation cross or filled black symbol."""

    _SZ = 80   # symbol half-size (px)

    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="white")
        self.canvas = tk.Canvas(self.root, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._items: list[int] = []
        self._key_cbs: dict[str, Callable] = {}
        self.root.bind("<KeyPress>", self._on_key)
        self.root.update_idletasks()
        self.show_fixation()

    def _cx_cy(self) -> tuple[int, int]:
        return self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2

    def _on_key(self, event):
        cb = self._key_cbs.get(event.keysym.lower())
        if cb:
            cb()

    def _clear(self):
        for item in self._items:
            self.canvas.delete(item)
        self._items.clear()

    def bind_key(self, key: str, cb: Callable):
        self._key_cbs[key.lower()] = cb

    def show_fixation(self):
        self._clear()
        cx, cy = self._cx_cy()
        self._items.append(
            self.canvas.create_text(cx, cy, text="+", font=("Helvetica", 48), fill="black")
        )

    def show_cue(self, condition: Condition):
        self._clear()
        cx, cy = self._cx_cy()
        s = self._SZ
        if condition == Condition.PH:          # ▲  up triangle
            pts = [cx, cy - s, cx - s, cy + s, cx + s, cy + s]
            self._items.append(self.canvas.create_polygon(pts, fill="black"))
        elif condition == Condition.PL:        # ▼  down triangle
            pts = [cx, cy + s, cx - s, cy - s, cx + s, cy - s]
            self._items.append(self.canvas.create_polygon(pts, fill="black"))
        else:                                  # ●  circle
            self._items.append(
                self.canvas.create_oval(cx - s, cy - s, cx + s, cy + s,
                                        fill="black", outline="black")
            )

    def update(self):
        self.root.update()

    def destroy(self):
        self.root.destroy()


# ── Serial reader ─────────────────────────────────────────────────────────────

class SerialReader(threading.Thread):
    """
    Continuously reads Arduino serial data.
    - Every P packet is pushed to the FSR LSL stream.
    - Rising-edge press events trigger the registered callback (same logic
      as duino_bridge_minimal.py: armed/disarmed with 200 ms debounce).
    - A press_start marker is always sent to LSL regardless of callback state.
    """

    def __init__(
        self,
        port: str,
        threshold: float,
        fsr_outlet: StreamOutlet,
        marker_outlet: StreamOutlet,
    ):
        super().__init__(daemon=True, name="SerialReader")
        self.port = port
        self.threshold = threshold
        self.fsr_outlet = fsr_outlet
        self.marker_outlet = marker_outlet

        self._stop = threading.Event()
        self._armed = True
        self._min_time_to_below = 0.0
        self._min_time_to_rest  = 0.0

        self._cb_lock = threading.Lock()
        self._press_cb: Callable[[float, float], None] | None = None

    def set_press_callback(self, cb: Callable[[float, float], None] | None):
        with self._cb_lock:
            self._press_cb = cb

    def stop(self):
        self._stop.set()

    def run(self):
        with serial.Serial(self.port, BAUD_RATE, timeout=1) as ser:
            while not self._stop.is_set():
                line = ser.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 3:
                    continue
                packet_type, _, value_str = parts
                try:
                    value = float(value_str)
                except ValueError:
                    continue

                t = local_clock()
                if packet_type == "P":
                    self.fsr_outlet.push_sample([value], t)
                    self._detect_press(value, t)
                else:
                    self.marker_outlet.push_sample(["sound_" + packet_type], t)

    def _detect_press(self, value: float, t: float):
        now = time.monotonic()
        if self._armed and value >= self.threshold and now - self._min_time_to_rest > 0.2:
            self.marker_outlet.push_sample([f"press_start {value:.1f}"], t)
            with self._cb_lock:
                cb = self._press_cb
            if cb:
                cb(t, value)
            self._armed = False
            self._min_time_to_below = now
        elif not self._armed and value < self.threshold and now - self._min_time_to_below > 0.2:
            self._armed = True
            self._min_time_to_rest = now


# ── Data logging ─────────────────────────────────────────────────────────────

def _open_data_file(data_dir: Path, participant_id: int, condition: Condition) -> tuple:
    """Create participant sub-directory and open a timestamped CSV. Returns (file, writer, path)."""
    p_dir = data_dir / f"participant_{participant_id}"
    p_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = p_dir / f"block_{condition.value}_{stamp}.csv"
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    f.flush()
    return f, writer, path


# ── Block runner ──────────────────────────────────────────────────────────────

def run_block(
    condition: Condition,
    n_trials: int,
    reader: SerialReader,
    audio: AudioEngine,
    display: ExperimentDisplay,
    markers: StreamOutlet,
    quit_check: Callable[[], bool],
    data_dir: Path = DEFAULT_DATA_DIR,
    participant_id: int = 0,
):
    """Run a single experimental block of n_trials trials."""

    # Pseudo-random tone sequence: MIX is balanced 50/50, others are fixed
    if condition == Condition.MIX:
        half = n_trials // 2
        tone_seq = ["low"] * half + ["high"] * (n_trials - half)
        random.shuffle(tone_seq)
    else:
        tone_seq = ["low" if condition == Condition.PL else "high"] * n_trials

    csv_f, csv_writer, csv_path = _open_data_file(data_dir, participant_id, condition)
    print(f"  Logging to {csv_path}")

    try:
        markers.push_sample(["block_start"], local_clock())

        for i, tone in enumerate(tone_seq):
            if quit_check():
                break

            tone_label = "PL" if tone == "low" else "PH"

            # ── ITI: fixation cross ───────────────────────────────────────────
            display.show_fixation()
            markers.push_sample([f"trial_start {i + 1}"], local_clock())
            display.update()

            iti_end = time.perf_counter() + random.uniform(ITI_MIN_S, ITI_MAX_S)
            while time.perf_counter() < iti_end:
                if quit_check():
                    return
                display.update()
                time.sleep(0.001)

            # ── Cue (300 ms) ──────────────────────────────────────────────────
            display.show_cue(condition)
            cue_t = local_clock()
            # Stream both the block condition and the trial's assigned tone so
            # downstream analysis always knows which sound will play (critical for MIX)
            markers.push_sample([f"cue_{condition.value}"], cue_t)
            markers.push_sample([f"trial_tone_{tone_label}"], cue_t)
            display.update()

            cue_end = time.perf_counter() + CUE_DURATION_S
            while time.perf_counter() < cue_end:
                if quit_check():
                    return
                display.update()
                time.sleep(0.001)

            # ── Cue offset ────────────────────────────────────────────────────
            display.show_fixation()
            markers.push_sample(["cue_offset"], local_clock())
            display.update()

            # ── Wait for press ────────────────────────────────────────────────
            pressed = threading.Event()
            press_info: dict = {}

            def _on_press(t_lsl: float, force: float, _ev=pressed, _info=press_info):
                if not _ev.is_set():
                    _info["t"]     = t_lsl
                    _info["force"] = force
                    _ev.set()

            reader.set_press_callback(_on_press)

            timeout_end = time.perf_counter() + RESPONSE_TIMEOUT_S
            while not pressed.is_set() and time.perf_counter() < timeout_end:
                if quit_check():
                    reader.set_press_callback(None)
                    return
                display.update()
                time.sleep(0.001)

            reader.set_press_callback(None)

            if not pressed.is_set():
                markers.push_sample(["trial_timeout"], local_clock())
                csv_writer.writerow({
                    "trial": i + 1, "condition": condition.value, "tone": tone_label,
                    "cue_t_lsl": f"{cue_t:.6f}", "press_t_lsl": "", "force": "",
                    "response_time_s": "", "outcome": "timeout",
                })
                csv_f.flush()
                print(f"  trial {i + 1:3d}/{n_trials}: timeout")
                continue

            # ── Tone after 20 ms ──────────────────────────────────────────────
            audio.play(tone, markers)
            rt = press_info["t"] - cue_t
            csv_writer.writerow({
                "trial": i + 1, "condition": condition.value, "tone": tone_label,
                "cue_t_lsl": f"{cue_t:.6f}",
                "press_t_lsl": f"{press_info['t']:.6f}",
                "force": f"{press_info['force']:.1f}",
                "response_time_s": f"{rt:.4f}",
                "outcome": "pressed",
            })
            csv_f.flush()
            print(f"  trial {i + 1:3d}/{n_trials}: {condition.value}/{tone_label}  "
                  f"rt={rt:.3f}s  force={press_info['force']:.1f}")

        markers.push_sample(["block_end"], local_clock())
        print("Block complete.")

    finally:
        csv_f.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    blk = sub.add_parser("block", help="Run a single experimental block")
    blk.add_argument(
        "--condition", choices=["PL", "PH", "MIX"], default="MIX",
        help="Auditory outcome condition (default: MIX)",
    )
    blk.add_argument(
        "--port", default=DEFAULT_PORT,
        help=f"Serial port (default: {DEFAULT_PORT})",
    )
    blk.add_argument(
        "--trials", type=int, default=DEFAULT_TRIALS,
        help=f"Trials per block (default: {DEFAULT_TRIALS})",
    )
    blk.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Force threshold for press detection (default: {DEFAULT_THRESHOLD})",
    )
    blk.add_argument(
        "--participant", type=int, default=0,
        help="Participant ID used in LSL source IDs and data directory (default: 0)",
    )
    blk.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Root directory for CSV output (default: {DEFAULT_DATA_DIR})",
    )

    args = parser.parse_args()

    if args.command != "block":
        parser.print_help()
        return 1

    condition = Condition(args.condition)
    print(f"Starting block: condition={condition.value}, port={args.port}, "
          f"trials={args.trials}, threshold={args.threshold}, participant={args.participant}, "
          f"data_dir={args.data_dir}")

    fsr_outlet, marker_outlet = create_outlets(args.participant)
    audio   = AudioEngine()
    display = ExperimentDisplay()

    quit_flag = [False]

    def _quit():
        quit_flag[0] = True

    display.bind_key("q",      _quit)
    display.bind_key("escape", _quit)

    reader = SerialReader(args.port, args.threshold, fsr_outlet, marker_outlet)
    reader.start()
    time.sleep(0.5)  # let serial port settle

    try:
        run_block(
            condition=condition,
            n_trials=args.trials,
            reader=reader,
            audio=audio,
            display=display,
            markers=marker_outlet,
            quit_check=lambda: quit_flag[0],
            data_dir=args.data_dir,
            participant_id=args.participant,
        )
        # Brief post-block fixation before closing
        display.show_fixation()
        display.update()
        time.sleep(2)
    finally:
        reader.stop()
        display.destroy()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
