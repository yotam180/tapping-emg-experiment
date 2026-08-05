import math
import time

from pylsl import StreamInfo, StreamOutlet

SAMPLE_RATE = 100  # Hz - samples per second
SINE_FREQ = 1.0  # Hz - frequency of the sine wave
AMPLITUDE = 1.0

info = StreamInfo(
    name="HelloWorld",
    type="EEG",
    channel_count=1,
    nominal_srate=SAMPLE_RATE,
    channel_format="float32",
    source_id="helloworld_sine",
)

outlet = StreamOutlet(info)
print(f"Streaming '{info.name()}' - {SINE_FREQ} Hz sine, amplitude {AMPLITUDE}")
print("Press Ctrl+C to stop.\n")

sample_index = 0
interval = 1.0 / SAMPLE_RATE
window_start = time.perf_counter()
window_count = 0
next_deadline = time.perf_counter()

try:
    while True:
        t = sample_index / SAMPLE_RATE
        value = AMPLITUDE * math.sin(2 * math.pi * SINE_FREQ * t)
        outlet.push_sample([value])
        sample_index += 1
        window_count += 1

        now = time.perf_counter()
        if now - window_start >= 1.0:
            print(f"TX rate: {window_count / (now - window_start):.2f} Hz")
            window_start = now
            window_count = 0

        next_deadline += interval
        remaining = next_deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
except KeyboardInterrupt:
    print("\nTransmitter stopped.")
