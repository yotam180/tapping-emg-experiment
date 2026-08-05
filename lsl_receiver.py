import time

from pylsl import StreamInlet, resolve_byprop

print("Looking for a 'HelloWorld' stream on the network...")
streams = resolve_byprop("name", "HelloWorld")
inlet = StreamInlet(streams[0])

info = inlet.info()
print(
    f"Connected to '{info.name()}' — {info.nominal_srate()} Hz, {info.channel_count()} channel(s)"
)
print("Receiving (Ctrl+C to stop):\n")

window_start = time.perf_counter()
window_count = 0

try:
    while True:
        # pull_chunk drains all buffered samples at once so we never fall behind
        samples, timestamps = inlet.pull_chunk(timeout=1.0)
        window_count += len(samples)

        now = time.perf_counter()
        if now - window_start >= 1.0:
            print(f"RX rate: {window_count / (now - window_start):.2f} Hz")
            window_start = now
            window_count = 0
except KeyboardInterrupt:
    print("\nReceiver stopped.")
