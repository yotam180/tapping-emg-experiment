import argparse
import threading
import time

import serial
from psychopy import prefs

prefs.hardware["audioLib"] = ["ptb"]
from psychopy import sound
from pylsl import IRREGULAR_RATE, StreamInfo, StreamOutlet, local_clock

SAMPLE_RATE = 830
DEFAULT_THRESHOLD = 40
RELEASE_RATIO = 0.5

TONE_FREQ = 440
TONE_DURATION_S = 0.050

BAUD_RATE = 115200

stimulus_sound = sound.Sound(
    value=TONE_FREQ, secs=TONE_DURATION_S, stereo=True, hamming=True, name="stim"
)


def _synced_info(name, stype, n_channels, srate, fmt, source_id):
    """StreamInfo declaring that every sample carries an explicit timestamp.

    Without this block MNELAB's read_native_xdf() routes the stream through its
    "legacy robust measured-clock segments" recovery, which rewrote our FSR
    timeline by up to 150 ms while leaving the marker stream on raw timestamps —
    making markers appear a few samples behind the force trace. Declaring the
    same v2 metadata the Xtrodes outlets use marks our per-sample local_clock()
    timestamps as authoritative, so no correction is applied.
    """
    info = StreamInfo(name, stype, n_channels, srate, fmt, source_id)
    sync = info.desc().append_child("synchronization")
    sync.append_child_value("timestamp_model_version", "2")
    sync.append_child_value("timestamp_semantics", "explicit_per_sample")
    sync.append_child_value(
        "timestamp_interpolation", "uniform_between_buffer_endpoints"
    )
    return info


fsr_outlet = StreamOutlet(
    _synced_info("FSR_force", "FSR", 1, IRREGULAR_RATE, "float32", "fsr_arduino")
)
marker_outlet = StreamOutlet(
    _synced_info("Markers", "Markers", 1, IRREGULAR_RATE, "string", "arduino_bridge")
)


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("port")
args = parser.parse_args()


armed = True
min_time_to_below = 0
min_time_to_rest = 0


def read_serial():
    global armed, min_time_to_rest, min_time_to_below

    with serial.Serial(args.port, BAUD_RATE, timeout=1) as ser:
        while True:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 3:
                continue

            # try:
            packet_type, _, value_str = parts
            # artuino_ts = int(arduino_ts_str)
            # except ValueError:
            #     continue
            value = 0.0
            # print(value)
            if packet_type == "P":
                value = float(value_str)
                t_shared = local_clock()
                fsr_outlet.push_sample([value], t_shared)
                if armed and value >= DEFAULT_THRESHOLD:
                    marker_outlet.push_sample(["press_start "+str(value)], t_shared)
                    trigger_tone()
                    armed = False

                if not armed and value < DEFAULT_THRESHOLD:
                    armed = True
            else:
                marker_outlet.push_sample(["sound_" + packet_type], t_shared)



serial_thread = threading.Thread(target=read_serial, daemon=True)
serial_thread.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopped.")
