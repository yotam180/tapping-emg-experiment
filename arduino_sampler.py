import argparse

import serial

parser = argparse.ArgumentParser()
parser.add_argument("port", help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
parser.add_argument("--baud", type=int, default=115200)
args = parser.parse_args()

with serial.Serial(args.port, args.baud, timeout=1) as ser:
    print(f"Listening on {args.port} at {args.baud} baud (Ctrl+C to stop):\n")
    print(f"{'packet_type':<12} {'arduino_micros':>16} {'fsr_value':>10}")
    print("-" * 42)
    try:
        while True:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 3:
                print(f"[unexpected] {line!r}")
                continue
            packet_type, timestamp, value = parts
            print(f"{packet_type:<12} {int(timestamp):>16} {int(value):>10}")
    except KeyboardInterrupt:
        print("\nStopped.")
