#!/usr/bin/env python3
"""
Tapping / sEMG predictability experiment.

Participants perform cue-guided, self-initiated index-finger presses against a
force-sensitive plate. A visual cue indicates the expected auditory outcome, and
the press triggers a short tone. See docs/experiment_design.md for the full
rationale.

Visual cues are three abstract, equal-size shapes on white:
    blue diamond, purple star, green hexagon.
A Condition->Shape mapping (--cue-map) assigns one shape to each of PL, PH and
MIX; in the full experiment this is randomised per participant and stored. A
plus sign is shown as fixation when no cue is present.

Within a single block of N trials the conditions are intermixed:
    N/4 PL, N/4 PH, and N/2 MIX (split N/4 MIX->low, N/4 MIX->high).
N must therefore be divisible by 4.

Each block modulates a single stimulus dimension (--modality):
    identity  -> low/high differ in frequency   (220 Hz vs 988 Hz)   [supported]
    intensity -> low/high differ in volume                           [not yet]

LSL streams (same schema as duino_bridge_minimal.py):
    FSR_force  float32  ~830 Hz continuous force
    Markers    string   irregular-rate event markers

Audio uses the PsychoPy PTB backend, matching duino_bridge_minimal.py, and is
played directly from the serial-reader thread the instant the press threshold is
crossed to keep press->sound latency minimal.

Usage:
    python main.py block [--port COM3] [--trials 60] [--modality identity]
                         [--threshold 40]

Quit any time with Q or Escape.
"""

import argparse
import math
import random
import threading
import time
from dataclasses import dataclass
from enum import Enum

import serial
from psychopy import prefs

prefs.hardware["audioLib"] = ["ptb"]
import pygame  # noqa: E402
from psychopy import sound  # noqa: E402  (must follow the prefs assignment)
from pylsl import IRREGULAR_RATE, StreamInfo, StreamOutlet, local_clock  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

SAMPLE_RATE = 830  # Nominal FSR rate pushed to LSL (Hz)
BAUD_RATE = 115200
DEFAULT_PORT = "COM3"
DEFAULT_TRIALS = 60
DEFAULT_THRESHOLD = 40.0

CUE_DURATION_S = 0.300  # Cue symbol display time
ITI_MIN_S = 2.5  # Inter-trial interval (fixation) bounds
ITI_MAX_S = 4.5
RESPONSE_TIMEOUT_S = 4.0  # Max wait for a press after cue onset

# Participants are asked to wait 1-2 s after the cue before pressing; responses
# outside this window are flagged (but, for now, still kept).
RESPONSE_MIN_S = 1.0
RESPONSE_MAX_S = 2.0

# Identity-modality tones.
TONE_FREQ_LOW = 220
TONE_FREQ_HIGH = 988
TONE_DURATION_S = 0.050

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SYMBOL_HALF = 80  # Circumradius of cue symbols (px) — matched across shapes
FIX_HALF = 20  # Half-size of the fixation cross (px)
STAR_INNER_RATIO = 0.6  # Star inner/outer radius: higher -> chubbier, less pointy


class Condition(Enum):
    PL = "PL"
    PH = "PH"
    MIX = "MIX"


class Tone(Enum):
    LOW = "low"
    HIGH = "high"


class Modality(Enum):
    IDENTITY = "identity"
    INTENSITY = "intensity"


class Shape(Enum):
    """Abstract cue shapes. Each has an intrinsic colour (see SHAPE_COLORS)."""

    DIAMOND = "diamond"  # blue
    STAR = "star"  # purple
    HEXAGON = "hexagon"  # green


SHAPE_COLORS: dict["Shape", tuple[int, int, int]] = {
    Shape.DIAMOND: (40, 90, 230),  # blue
    Shape.STAR: (150, 40, 200),  # purple
    Shape.HEXAGON: (30, 160, 60),  # green
}

# Default Condition->Shape mapping. In the full experiment this is randomised
# per participant and stored in their data files; here it is just a sensible
# fixed default, overridable with --cue-map.
DEFAULT_CUE_MAP = "PL=diamond,PH=star,MIX=hexagon"


@dataclass(frozen=True)
class Trial:
    condition: Condition
    tone: Tone  # The outcome that will actually be played on this trial.


@dataclass
class PressInfo:
    t: float  # LSL timestamp of the threshold crossing.
    force: float


# ── Trial sequence ───────────────────────────────────────────────────────────


def build_block(n_trials: int) -> list[Trial]:
    """Build a shuffled block: N/4 PL, N/4 PH, N/4 MIX-low, N/4 MIX-high."""
    if n_trials % 4 != 0:
        raise ValueError(f"trials must be divisible by 4, got {n_trials}")
    quarter = n_trials // 4
    trials = (
        [Trial(Condition.PL, Tone.LOW)] * quarter
        + [Trial(Condition.PH, Tone.HIGH)] * quarter
        + [Trial(Condition.MIX, Tone.LOW)] * quarter
        + [Trial(Condition.MIX, Tone.HIGH)] * quarter
    )
    random.shuffle(trials)
    return trials


def parse_cue_map(spec: str) -> dict[Condition, Shape]:
    """Parse 'PL=diamond,PH=star,MIX=hexagon' (or 'random') into a mapping.

    Validates that all three conditions are present and mapped to three
    distinct shapes.
    """
    if spec.strip().lower() == "random":
        shapes = list(Shape)
        random.shuffle(shapes)
        return dict(zip([Condition.PL, Condition.PH, Condition.MIX], shapes))

    mapping: dict[Condition, Shape] = {}
    for pair in spec.split(","):
        cond_str, sep, shape_str = pair.partition("=")
        if not sep:
            raise ValueError(f"bad cue-map entry {pair!r}, expected COND=shape")
        try:
            cond = Condition(cond_str.strip().upper())
            shape = Shape(shape_str.strip().lower())
        except ValueError as exc:
            raise ValueError(f"bad cue-map entry {pair!r}: {exc}") from exc
        if cond in mapping:
            raise ValueError(f"condition {cond.value} mapped more than once")
        mapping[cond] = shape

    if set(mapping) != set(Condition):
        raise ValueError("cue-map must cover PL, PH and MIX")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("cue-map shapes must be distinct")
    return mapping


def format_cue_map(cue_map: dict[Condition, Shape]) -> str:
    """Inverse of parse_cue_map, in a stable condition order."""
    return ",".join(
        f"{c.value}={cue_map[c].value}" for c in (Condition.PL, Condition.PH, Condition.MIX)
    )


# ── Audio ──────────────────────────────────────────────────────────────────────


def list_audio_devices() -> int:
    """Print the PTB output devices so a name can be passed to --audio-device."""
    try:
        import psychtoolbox.audio as ptb_audio

        devices = [d for d in ptb_audio.get_devices() if d["NrOutputChannels"] > 0]
    except Exception as exc:  # Fall back to PsychoPy's own enumeration.
        print(f"(psychtoolbox enumeration failed: {exc}; using PsychoPy)")
        from psychopy.sound.backend_ptb import getDevices

        devices = list(getDevices(kind="output").values())

    for d in devices:
        print(
            f"  {d['DeviceName']!r}  "
            f"({int(d['NrOutputChannels'])} out, {d.get('HostAudioAPIName', '?')})"
        )
    return 0


class AudioEngine:
    """Pre-loads the low/high tones for a modality and plays them immediately."""

    def __init__(self, modality: Modality, device: str | None = None):
        if modality != Modality.IDENTITY:
            raise NotImplementedError(
                f"modality {modality.value!r} not supported yet (only 'identity')"
            )
        # Select the output device before the first Sound opens the PTB stream.
        # PsychoPy matches the device name exactly (see 'audio-devices' for the
        # exact strings); duplicate names across host APIs are resolved by PTB's
        # latency preference, normally favouring the low-latency WASAPI entry.
        if device:
            prefs.hardware["audioDevice"] = device
        self._tones = {
            Tone.LOW: sound.Sound(
                value=TONE_FREQ_LOW,
                secs=TONE_DURATION_S,
                stereo=True,
                hamming=True,
                name="low",
            ),
            Tone.HIGH: sound.Sound(
                value=TONE_FREQ_HIGH,
                secs=TONE_DURATION_S,
                stereo=True,
                hamming=True,
                name="high",
            ),
        }

    def play(self, tone: Tone) -> None:
        """Play a tone. Called from the serial thread; must not block."""
        self._tones[tone].play()


# ── Serial reader ────────────────────────────────────────────────────────────


class SerialReader(threading.Thread):
    """Streams FSR samples to LSL and triggers the armed tone on a press.

    The main loop arms the reader with the current trial's tone; the next rising
    edge across ``threshold`` plays that tone, records the press and fires
    ``press_event``. Playing the sound here (rather than from the main loop)
    keeps the press->sound path as short as possible.
    """

    def __init__(
        self,
        port: str,
        threshold: float,
        audio: AudioEngine,
        fsr_outlet: StreamOutlet,
        marker_outlet: StreamOutlet,
    ):
        super().__init__(daemon=True)
        self.port = port
        self.threshold = threshold
        self.audio = audio
        self.fsr_outlet = fsr_outlet
        self.marker_outlet = marker_outlet

        self.press_event = threading.Event()
        self.press_info: PressInfo | None = None
        self.error: Exception | None = None

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._armed_tone: Tone | None = None
        self._below = True  # Whether the last sample was below threshold.

    def arm(self, tone: Tone) -> None:
        """Begin accepting a press for the current trial."""
        with self._lock:
            self._armed_tone = tone
        self.press_info = None
        self.press_event.clear()

    def disarm(self) -> None:
        with self._lock:
            self._armed_tone = None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
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
        except Exception as exc:  # Surface serial errors to the main thread.
            self.error = exc

    def _detect_press(self, value: float, t: float) -> None:
        with self._lock:
            armed = self._armed_tone
            rising_edge = self._below and value >= self.threshold
            if armed is not None and rising_edge:
                self.audio.play(armed)  # Fire sound first — latency critical.
                self.marker_outlet.push_sample([f"press_start {value:.1f}"], t)
                self.marker_outlet.push_sample([f"audio {armed.value}"], t)
                self.press_info = PressInfo(t=t, force=value)
                self._armed_tone = None
                self.press_event.set()
            self._below = value < self.threshold


# ── Display ──────────────────────────────────────────────────────────────────


def _polygon_points(
    cx: float, cy: float, radius: float, n: int, rotation_deg: float
) -> list[tuple[float, float]]:
    """Vertices of a regular ``n``-gon on a circle of ``radius`` (screen coords)."""
    rot = math.radians(rotation_deg)
    return [
        (cx + radius * math.cos(rot + 2 * math.pi * i / n),
         cy - radius * math.sin(rot + 2 * math.pi * i / n))
        for i in range(n)
    ]


def _star_points(
    cx: float, cy: float, r_out: float, r_in: float, points: int = 5,
    rotation_deg: float = 90.0,
) -> list[tuple[float, float]]:
    """Vertices of a ``points``-pointed star alternating outer/inner radii."""
    rot = math.radians(rotation_deg)
    verts = []
    for i in range(points * 2):
        r = r_out if i % 2 == 0 else r_in
        a = rot + math.pi * i / points
        verts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
    return verts


def _shape_points(
    shape: Shape, cx: float, cy: float, s: float
) -> list[tuple[float, float]]:
    if shape == Shape.DIAMOND:  # square rotated 45° -> point up/right/down/left
        return _polygon_points(cx, cy, s, 4, rotation_deg=90)
    if shape == Shape.HEXAGON:  # flat top and bottom
        return _polygon_points(cx, cy, s, 6, rotation_deg=0)
    return _star_points(cx, cy, s, s * STAR_INNER_RATIO)  # STAR, point up


class Display:
    """Fullscreen Pygame window drawing fixation / cue symbols."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)
        self.w, self.h = self.screen.get_size()
        self.quit = False
        self.show_fixation()

    @property
    def _center(self) -> tuple[int, int]:
        return self.w // 2, self.h // 2

    def pump(self) -> None:
        """Process the event queue; sets ``quit`` on Q/Escape/window close."""
        for event in pygame.event.get():
            if (
                event.type == pygame.QUIT
                or event.type == pygame.KEYDOWN
                and event.key
                in (
                    pygame.K_q,
                    pygame.K_ESCAPE,
                )
            ):
                self.quit = True

    def show_fixation(self) -> None:
        cx, cy = self._center
        self.screen.fill(WHITE)
        pygame.draw.line(
            self.screen, BLACK, (cx - FIX_HALF, cy), (cx + FIX_HALF, cy), 4
        )
        pygame.draw.line(
            self.screen, BLACK, (cx, cy - FIX_HALF), (cx, cy + FIX_HALF), 4
        )
        pygame.display.flip()

    def show_cue(self, shape: Shape) -> None:
        cx, cy = self._center
        self.screen.fill(WHITE)
        points = _shape_points(shape, cx, cy, SYMBOL_HALF)
        pygame.draw.polygon(self.screen, SHAPE_COLORS[shape], points)
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()


# ── Waiting helpers ───────────────────────────────────────────────────────────


def wait(duration_s: float, display: Display) -> bool:
    """Idle for ``duration_s`` while pumping events. False if the user quit."""
    end = time.perf_counter() + duration_s
    while time.perf_counter() < end:
        display.pump()
        if display.quit:
            return False
        time.sleep(0.001)
    return True


def wait_for_press(
    reader: SerialReader, timeout_s: float, display: Display
) -> PressInfo | None:
    """Wait for a press (up to ``timeout_s``). None on timeout or quit."""
    end = time.perf_counter() + timeout_s
    while time.perf_counter() < end:
        display.pump()
        if display.quit:
            return None
        if reader.press_event.is_set():
            return reader.press_info
        time.sleep(0.001)
    return None


# ── Block runner ──────────────────────────────────────────────────────────────


def classify_response(rt: float) -> str:
    if rt < RESPONSE_MIN_S:
        return "early"
    if rt > RESPONSE_MAX_S:
        return "late"
    return "valid"


def run_block(
    trials: list[Trial],
    cue_map: dict[Condition, Shape],
    reader: SerialReader,
    display: Display,
    markers: StreamOutlet,
) -> None:
    n = len(trials)
    markers.push_sample(["block_start"], local_clock())
    # Record the mapping so the LSL recording is self-describing.
    markers.push_sample([f"cue_map {format_cue_map(cue_map)}"], local_clock())

    for i, trial in enumerate(trials, start=1):
        # ── Inter-trial interval (fixation) ───────────────────────────────────
        display.show_fixation()
        markers.push_sample([f"trial_start {i}"], local_clock())
        if not wait(random.uniform(ITI_MIN_S, ITI_MAX_S), display):
            break

        # ── Cue (300 ms) ──────────────────────────────────────────────────────
        shape = cue_map[trial.condition]
        display.show_cue(shape)
        cue_t = local_clock()
        markers.push_sample([f"cue {trial.condition.value}"], cue_t)
        # Record the planned outcome so MIX trials are decodable downstream.
        markers.push_sample([f"outcome {trial.tone.value}"], cue_t)
        if not wait(CUE_DURATION_S, display):
            break

        # ── Fixation + wait for press ─────────────────────────────────────────
        display.show_fixation()
        markers.push_sample(["cue_offset"], local_clock())

        reader.arm(trial.tone)
        press = wait_for_press(reader, RESPONSE_TIMEOUT_S, display)
        reader.disarm()

        if display.quit:
            break

        if press is None:
            markers.push_sample([f"trial_timeout {i}"], local_clock())
            print(
                f"  trial {i:3d}/{n}: {trial.condition.value}/{trial.tone.value}  timeout"
            )
            continue

        rt = press.t - cue_t
        outcome = classify_response(rt)
        print(
            f"  trial {i:3d}/{n}: {trial.condition.value}/{trial.tone.value}  "
            f"rt={rt:.3f}s force={press.force:.1f} [{outcome}]"
        )

    markers.push_sample(["block_end"], local_clock())
    print("Block complete.")


# ── LSL ────────────────────────────────────────────────────────────────────────


def create_outlets() -> tuple[StreamOutlet, StreamOutlet]:
    fsr = StreamOutlet(
        StreamInfo("FSR_force", "FSR", 1, SAMPLE_RATE, "float32", "fsr_arduino")
    )
    markers = StreamOutlet(
        StreamInfo("Markers", "Markers", 1, IRREGULAR_RATE, "string", "arduino_bridge")
    )
    return fsr, markers


# ── CLI ────────────────────────────────────────────────────────────────────────


def cmd_block(args: argparse.Namespace) -> int:
    modality = Modality(args.modality)
    trials = build_block(args.trials)
    cue_map = parse_cue_map(args.cue_map)
    print(
        f"Starting block: modality={modality.value}, port={args.port}, "
        f"trials={args.trials}, threshold={args.threshold}, "
        f"audio_device={args.audio_device or '(system default)'}, "
        f"cue_map={format_cue_map(cue_map)}"
    )

    fsr_outlet, marker_outlet = create_outlets()
    audio = AudioEngine(modality, device=args.audio_device)
    display = Display()
    reader = SerialReader(args.port, args.threshold, audio, fsr_outlet, marker_outlet)
    reader.start()
    time.sleep(0.5)  # Let the serial port settle.

    if reader.error is not None:
        display.close()
        print(f"Serial error: {reader.error}")
        return 1

    try:
        run_block(trials, cue_map, reader, display, marker_outlet)
        display.show_fixation()
        wait(1.5, display)  # Brief tail fixation before closing.
    finally:
        reader.stop()
        display.close()
    return 0


def cmd_test_audio(args: argparse.Namespace) -> int:
    """Play the low then high tone so the chosen output device can be verified."""
    audio = AudioEngine(Modality(args.modality), device=args.audio_device)
    print(f"Playing on {args.audio_device or '(system default)'}")
    for tone in (Tone.LOW, Tone.HIGH):
        print(f"  {tone.value}")
        audio.play(tone)
        time.sleep(0.8)
    return 0


def cmd_preview_cues(args: argparse.Namespace) -> int:
    """Cycle through the three cue shapes so their appearance can be checked."""
    display = Display()
    try:
        for shape in Shape:
            display.show_cue(shape)
            if not wait(args.seconds, display):
                break
    finally:
        display.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    blk = sub.add_parser("block", help="Run a single intermixed block")
    blk.add_argument(
        "--port", default=DEFAULT_PORT, help=f"Serial port (default {DEFAULT_PORT})"
    )
    blk.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help=f"Trials per block, divisible by 4 (default {DEFAULT_TRIALS})",
    )
    blk.add_argument(
        "--modality",
        choices=[m.value for m in Modality],
        default=Modality.IDENTITY.value,
        help="Stimulus dimension to modulate (default identity)",
    )
    blk.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"FSR press-detection threshold (default {DEFAULT_THRESHOLD})",
    )
    blk.add_argument(
        "--audio-device",
        default=None,
        help="Exact output device name for the tones; omit for system default. "
        "See 'audio-devices' for names.",
    )
    blk.add_argument(
        "--cue-map",
        default=DEFAULT_CUE_MAP,
        help="Condition->shape mapping, e.g. 'PL=diamond,PH=star,MIX=hexagon', "
        f"or 'random' (default '{DEFAULT_CUE_MAP}'). Shapes: "
        f"{', '.join(s.value for s in Shape)}.",
    )
    blk.set_defaults(func=cmd_block)

    sub.add_parser("audio-devices", help="List available audio output devices")

    test = sub.add_parser("test-audio", help="Play the low/high tones to verify output")
    test.add_argument(
        "--modality",
        choices=[m.value for m in Modality],
        default=Modality.IDENTITY.value,
    )
    test.add_argument(
        "--audio-device",
        default=None,
        help="Exact output device name (see 'audio-devices').",
    )
    test.set_defaults(func=cmd_test_audio)

    preview = sub.add_parser("preview-cues", help="Cycle through the cue shapes")
    preview.add_argument(
        "--seconds", type=float, default=1.5, help="Seconds per shape (default 1.5)"
    )
    preview.set_defaults(func=cmd_preview_cues)

    args = parser.parse_args()
    if args.command == "audio-devices":
        return list_audio_devices()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
