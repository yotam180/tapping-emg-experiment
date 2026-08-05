import argparse
import threading
import time

import numpy as np
import serial
import sounddevice as sd
from pylsl import StreamInfo, StreamOutlet

SAMPLE_RATE = 905
DEFAULT_THRESHOLD = 60
RELEASE_RATIO = 0.5

AUDIO_SAMPLE_RATE = 44100
TONE_FREQ = 440
TONE_DURATION_S = 0.050
FADE_DURATION_S = 0.005

BAUD_RATE = 115200

# Pre-generate tone buffer
_n = int(AUDIO_SAMPLE_RATE * TONE_DURATION_S)
_nf = int(AUDIO_SAMPLE_RATE * FADE_DURATION_S)
_t = np.linspace(0, TONE_DURATION_S, _n, endpoint=False)
TONE_BUFFER = np.sin(2 * np.pi * TONE_FREQ * _t).astype(np.float32)
TONE_BUFFER[:_nf] *= np.linspace(0.0, 1.0, _nf, dtype=np.float32)
TONE_BUFFER[-_nf:] *= np.linspace(1.0, 0.0, _nf, dtype=np.float32)


fsr_outlet = StreamOutlet(
    StreamInfo("FSR_force", "FSR", 1, SAMPLE_RATE, "float32", "fsr_arduino")
)
marker_outlet = StreamOutlet(
    StreamInfo("Markers", "Markers", 1, 0, "string", "arduino_bridge")
)


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("port")
args = parser.parse_args()


# Audio: persistent stream, callback-driven
# _tone_pos == -1  → idle (output silence)
# _tone_pos >= 0   → playing from that sample index
_tone_pos = -1
_tone_lock = threading.Lock()

_tone_started = False


def _audio_callback(outdata, frames, time_info, status):
    global _tone_pos, _tone_started
    with _tone_lock:
        if _tone_pos < 0:
            outdata.fill(0)
            return

        if not _tone_started:
            marker_outlet.push_sample(["tone_start"])
            _tone_started = True

        end = _tone_pos + frames
        chunk = min(end, len(TONE_BUFFER)) - _tone_pos
        outdata[:chunk, 0] = TONE_BUFFER[_tone_pos : _tone_pos + chunk]
        if chunk < frames:
            outdata[chunk:] = 0
            _tone_pos = -1
            _tone_started = False
            marker_outlet.push_sample(["tone_ended"])
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


armed = False


def read_serial():
    global armed

    with serial.Serial(args.port, BAUD_RATE, timeout=1) as ser:
        while True:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) != 3:
                continue

            try:
                packet_type, _, value_str = parts
                # artuino_ts = int(arduino_ts_str)
                value = float(value_str)
            except ValueError:
                continue

            if packet_type == "P":
                fsr_outlet.push_sample([value])
            else:
                marker_outlet.push_sample(["sound_" + packet_type])

            if armed and value >= DEFAULT_THRESHOLD:
                marker_outlet.push_sample("press_start")
                trigger_tone()
                marker_outlet.push_sample("triggered_tone")
                armed = False

            if not armed and value < DEFAULT_THRESHOLD:
                armed = True


serial_thread = threading.Thread(target=read_serial, daemon=True)
serial_thread.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    _audio_stream.stop()
    _audio_stream.close()
    print("\nStopped.")
