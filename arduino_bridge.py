import argparse
import threading
import time

import numpy as np
import serial
import sounddevice as sd
from pylsl import StreamInfo, StreamOutlet

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 905.8
DEFAULT_THRESHOLD = 50
RELEASE_RATIO = 0.5

AUDIO_SAMPLE_RATE = 44100
TONE_FREQ = 440
TONE_DURATION_S = 0.050
FADE_DURATION_S = 0.005

# ── Pre-generate tone buffer ──────────────────────────────────────────────────
_n = int(AUDIO_SAMPLE_RATE * TONE_DURATION_S)
_nf = int(AUDIO_SAMPLE_RATE * FADE_DURATION_S)
_t = np.linspace(0, TONE_DURATION_S, _n, endpoint=False)
TONE_BUFFER = np.sin(2 * np.pi * TONE_FREQ * _t).astype(np.float32)
TONE_BUFFER[:_nf] *= np.linspace(0.0, 1.0, _nf, dtype=np.float32)
TONE_BUFFER[-_nf:] *= np.linspace(1.0, 0.0, _nf, dtype=np.float32)

# ── Audio: persistent stream, callback-driven ─────────────────────────────────
# _tone_pos == -1  → idle (output silence)
# _tone_pos >= 0   → playing from that sample index
_tone_pos = -1
_tone_lock = threading.Lock()


def _audio_callback(outdata, frames, time_info, status):
    global _tone_pos
    with _tone_lock:
        if _tone_pos < 0:
            outdata.fill(0)
            return
        end = _tone_pos + frames
        chunk = min(end, len(TONE_BUFFER)) - _tone_pos
        outdata[:chunk, 0] = TONE_BUFFER[_tone_pos : _tone_pos + chunk]
        if chunk < frames:
            outdata[chunk:] = 0
            _tone_pos = -1
        else:
            _tone_pos = end


_audio_stream = sd.OutputStream(
    samplerate=AUDIO_SAMPLE_RATE,
    channels=1,
    dtype="float32",
    latency="low",
    callback=_audio_callback,
)
_audio_stream.start()


def trigger_tone():
    """Called from serial thread — must never block."""
    global _tone_pos
    with _tone_lock:
        _tone_pos = 0


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("port")
parser.add_argument("--baud", type=int, default=115200)
parser.add_argument(
    "--rate", type=float, default=SAMPLE_RATE, help="Nominal sample rate Hz"
)
parser.add_argument(
    "--threshold",
    type=int,
    default=DEFAULT_THRESHOLD,
    help="FSR press threshold (0–1023)",
)
args = parser.parse_args()

release_threshold = args.threshold * RELEASE_RATIO

# ── LSL outlets ───────────────────────────────────────────────────────────────
fsr_outlet = StreamOutlet(
    StreamInfo("FSR_force", "FSR", 1, args.rate, "float32", "fsr_arduino")
)
marker_outlet = StreamOutlet(
    StreamInfo("Markers", "Markers", 1, 0, "string", "arduino_bridge")
)

print("Audio stream open. LSL streams open.")
print(
    f"Threshold: {args.threshold}  Release: {release_threshold:.0f}  Rate: {args.rate} Hz"
)
print(f"Listening on {args.port} (Ctrl+C to stop)...\n")

# ── Latency measurement ───────────────────────────────────────────────────────
_press_wall_t = None  # perf_counter when trigger_tone() was called
_press_arduino_ts = None  # Arduino micros of the triggering P packet

# ── Serial reader ─────────────────────────────────────────────────────────────
armed = True


def read_serial():
    global armed, _press_wall_t, _press_arduino_ts
    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        while True:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            parts = line.split(",")
            packet_type = parts[0]

            if packet_type == "P" and len(parts) == 3:
                try:
                    arduino_ts = int(parts[1])
                    value = int(parts[2])
                except ValueError:
                    continue

                fsr_outlet.push_sample([float(value)])

                if armed and value >= args.threshold:
                    trigger_tone()  # ← non-blocking
                    marker_outlet.push_sample(["press_start"])
                    _press_wall_t = time.perf_counter()
                    _press_arduino_ts = arduino_ts
                    print(f"PRESS  t={arduino_ts}  value={value}")
                    armed = False
                elif not armed and value < release_threshold:
                    armed = True

            elif packet_type in ("L", "R", "B"):
                marker_outlet.push_sample([f"audio_{packet_type}"])
                if _press_wall_t is not None:
                    delay_ms = (time.perf_counter() - _press_wall_t) * 1000
                    print(f"AUDIO  {packet_type}  delay={delay_ms:.1f} ms")
                    _press_wall_t = None
                else:
                    print(f"AUDIO  {packet_type}  raw={line!r}")

            else:
                print(f"[unknown] {line!r}")


serial_thread = threading.Thread(target=read_serial, daemon=True)
serial_thread.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    _audio_stream.stop()
    _audio_stream.close()
    print("\nStopped.")
