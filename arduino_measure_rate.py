import argparse
import statistics

import serial

DURATION_SECONDS = 5
GAP_THRESHOLD = 1.5  # intervals > this multiple of median are flagged as gaps

parser = argparse.ArgumentParser()
parser.add_argument("port")
parser.add_argument("--baud", type=int, default=115200)
args = parser.parse_args()

print(f"Measuring for {DURATION_SECONDS}s on {args.port}...")

timestamps = []

with serial.Serial(args.port, args.baud, timeout=1) as ser:
    import time
    deadline = time.perf_counter() + DURATION_SECONDS
    while time.perf_counter() < deadline:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) == 3 and parts[0] == "P":
            try:
                timestamps.append(int(parts[1]))
            except ValueError:
                continue

n = len(timestamps)
if n < 2:
    print("Not enough samples received — check port, baud rate, and Arduino sketch.")
    raise SystemExit(1)

intervals = [timestamps[i+1] - timestamps[i] for i in range(n - 1)]
median_us = statistics.median(intervals)
mean_us = statistics.mean(intervals)
stdev_us = statistics.stdev(intervals)
sample_rate = 1_000_000 / median_us

gaps = [iv for iv in intervals if iv > GAP_THRESHOLD * median_us]
gap_count = len(gaps)

print(f"\n{'Samples received':<28} {n}")
print(f"{'Median interval':<28} {median_us:.1f} µs")
print(f"{'Mean interval':<28} {mean_us:.1f} µs")
print(f"{'Std deviation':<28} {stdev_us:.1f} µs  ({100*stdev_us/median_us:.2f}% jitter)")
print(f"{'Estimated sample rate':<28} {sample_rate:.1f} Hz")
gap_label = f"Gaps detected (>{GAP_THRESHOLD:.1f}x median)"
print(f"{gap_label:<28} {gap_count}  ({100*gap_count/len(intervals):.2f}% of intervals)")

if gap_count > 0:
    print(f"\nWARNING: {gap_count} gap(s) found — Python may be reading too slowly or Arduino skipped samples.")
    print(f"  Largest gap: {max(gaps):.0f} µs  (~{max(gaps)/median_us:.1f}x expected interval)")
else:
    print("\nOK: no gaps detected — reading is keeping up with Arduino.")
