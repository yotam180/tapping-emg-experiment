"""
Audio latency benchmark.

Measures time from trigger_tone() call to first non-silent callback invocation,
plus reports the stream's reported output latency for each configuration.

Run with:  python latency_bench.py
"""

import time
import threading
import numpy as np
import sounddevice as sd

AUDIO_SAMPLE_RATE = 44100
TONE_FREQ = 440
TONE_DURATION_S = 0.200
N_TRIALS = 10
INTER_TRIAL_S = 0.5

_n = int(AUDIO_SAMPLE_RATE * TONE_DURATION_S)
_t = np.linspace(0, TONE_DURATION_S, _n, endpoint=False)
TONE_BUFFER = np.sin(2 * np.pi * TONE_FREQ * _t).astype(np.float32)

# ── List available host APIs and devices ─────────────────────────────────────
print("=== Host APIs ===")
for i, api in enumerate(sd.query_hostapis()):
    print(f"  [{i}] {api['name']}  (default_output_device={api['default_output_device']})")

print("\n=== Output Devices ===")
for dev in sd.query_devices():
    if dev['max_output_channels'] > 0:
        print(f"  [{dev['index']:2d}] {dev['name']!r}  "
              f"hostapi={sd.query_hostapis(dev['hostapi'])['name']}  "
              f"default_low_out={dev['default_low_output_latency']*1000:.1f}ms  "
              f"default_high_out={dev['default_high_output_latency']*1000:.1f}ms")
print()

# ── Benchmark helpers ─────────────────────────────────────────────────────────

def run_bench(label, stream_kwargs, n_trials=N_TRIALS):
    results_trigger_to_cb = []
    results_reported_latency = []

    _tone_pos = [-1]
    _trigger_time = [None]
    _first_cb_time = [None]
    _dac_time = [None]
    _lock = threading.Lock()

    def callback(outdata, frames, time_info, status):
        with _lock:
            pos = _tone_pos[0]

        if pos < 0:
            outdata.fill(0)
            return

        if _first_cb_time[0] is None:
            _first_cb_time[0] = time.perf_counter()
            _dac_time[0] = time_info.outputBufferDacTime

        end = pos + frames
        chunk = min(end, len(TONE_BUFFER)) - pos
        outdata[:chunk, 0] = TONE_BUFFER[pos:pos + chunk]
        if chunk < frames:
            outdata[chunk:] = 0
            with _lock:
                _tone_pos[0] = -1
        else:
            with _lock:
                _tone_pos[0] = end

    try:
        stream = sd.OutputStream(callback=callback, **stream_kwargs)
        stream.start()
        lat = stream.latency
        # sounddevice <0.4 returns float; >=0.4 returns (in_lat, out_lat)
        reported_out_lat = (lat[1] if isinstance(lat, tuple) else lat) * 1000
        results_reported_latency.append(reported_out_lat)
    except Exception as e:
        print(f"  {label}: FAILED to open stream: {e}")
        return

    time.sleep(0.15)  # warm-up

    for trial in range(n_trials):
        _tone_pos[0] = -1
        _first_cb_time[0] = None
        _dac_time[0] = None
        _trigger_time[0] = None

        time.sleep(0.05)  # let silence settle

        _trigger_time[0] = time.perf_counter()
        with _lock:
            _tone_pos[0] = 0

        deadline = time.perf_counter() + 0.5
        while _first_cb_time[0] is None and time.perf_counter() < deadline:
            time.sleep(0.0005)

        if _first_cb_time[0] is None:
            print(f"  {label} trial {trial}: callback never fired!")
            continue

        trigger_to_cb = (_first_cb_time[0] - _trigger_time[0]) * 1000
        results_trigger_to_cb.append(trigger_to_cb)

        time.sleep(INTER_TRIAL_S)

    stream.stop()
    stream.close()

    if results_trigger_to_cb:
        arr = np.array(results_trigger_to_cb)
        print(f"  {label}")
        print(f"    trigger->first-callback  min={arr.min():.2f}  median={np.median(arr):.2f}  max={arr.max():.2f}  ms")
        print(f"    reported output latency = {reported_out_lat:.2f} ms")
        total_min = arr.min() + reported_out_lat
        total_med = np.median(arr) + reported_out_lat
        print(f"    estimated total latency  min={total_min:.2f}  median={total_med:.2f}  ms")
    print()


base_kwargs = dict(
    samplerate=AUDIO_SAMPLE_RATE,
    channels=1,
    dtype="float32",
)

# ── Test 1: default blocksize, latency="low" (current code) ──────────────────
print("=== Benchmarks ===\n")

run_bench(
    "sounddevice default blocksize, latency='low'",
    {**base_kwargs, "latency": "low"},
)

# ── Test 2: explicit small blocksizes ─────────────────────────────────────────
for bs in [64, 128, 256, 512]:
    run_bench(
        f"sounddevice blocksize={bs}, latency='low'",
        {**base_kwargs, "latency": "low", "blocksize": bs},
    )

# ── Test 3: WASAPI exclusive mode (Windows only) ──────────────────────────────
wasapi_api = None
for i, api in enumerate(sd.query_hostapis()):
    if "WASAPI" in api["name"]:
        wasapi_api = i
        break

if wasapi_api is not None:
    # Find the default WASAPI output device
    default_out = sd.query_hostapis(wasapi_api)["default_output_device"]
    if default_out >= 0:
        dev_info = sd.query_devices(default_out)
        print(f"WASAPI exclusive mode: device={dev_info['name']!r}")
        try:
            extra = sd.WasapiSettings(exclusive=True)
            run_bench(
                "WASAPI exclusive, blocksize=64",
                {**base_kwargs, "device": default_out, "latency": "low",
                 "blocksize": 64, "extra_settings": extra},
            )
            run_bench(
                "WASAPI exclusive, blocksize=128",
                {**base_kwargs, "device": default_out, "latency": "low",
                 "blocksize": 128, "extra_settings": extra},
            )
        except Exception as e:
            print(f"  WASAPI exclusive failed: {e}\n")
    else:
        print("  No default WASAPI output device found.\n")
else:
    print("  WASAPI host API not found on this system.\n")

# ── Test 4: PyAudio (if available) ───────────────────────────────────────────
try:
    import pyaudio
    print("=== PyAudio test ===\n")

    def run_pyaudio_bench(label, chunk_size, n_trials=N_TRIALS):
        results = []
        pa = pyaudio.PyAudio()
        trigger_time = [None]
        first_cb_time = [None]
        active = [False]
        pos = [0]

        def pa_callback(in_data, frame_count, time_info, status):
            if not active[0]:
                return (b'\x00' * frame_count * 4, pyaudio.paContinue)
            if first_cb_time[0] is None:
                first_cb_time[0] = time.perf_counter()
            end = pos[0] + frame_count
            chunk = min(end, len(TONE_BUFFER)) - pos[0]
            out = TONE_BUFFER[pos[0]:pos[0] + chunk]
            if chunk < frame_count:
                out = np.concatenate([out, np.zeros(frame_count - chunk, dtype=np.float32)])
                active[0] = False
            pos[0] = end
            return (out.tobytes(), pyaudio.paContinue)

        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=AUDIO_SAMPLE_RATE,
            output=True,
            frames_per_buffer=chunk_size,
            stream_callback=pa_callback,
        )
        stream.start_stream()
        time.sleep(0.15)

        for trial in range(n_trials):
            active[0] = False
            first_cb_time[0] = None
            pos[0] = 0
            time.sleep(0.05)

            trigger_time[0] = time.perf_counter()
            active[0] = True

            deadline = time.perf_counter() + 0.5
            while first_cb_time[0] is None and time.perf_counter() < deadline:
                time.sleep(0.0005)

            if first_cb_time[0] is not None:
                results.append((first_cb_time[0] - trigger_time[0]) * 1000)

            time.sleep(INTER_TRIAL_S)

        stream.stop_stream()
        stream.close()
        pa.terminate()

        if results:
            arr = np.array(results)
            print(f"  {label}")
            print(f"    trigger→first-callback  min={arr.min():.2f}  median={np.median(arr):.2f}  max={arr.max():.2f}  ms\n")

    for bs in [64, 128, 256]:
        run_pyaudio_bench(f"PyAudio chunk={bs}", bs)

except ImportError:
    print("PyAudio not installed, skipping.\n")

print("Done.")
