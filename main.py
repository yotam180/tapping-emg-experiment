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
    python main.py experiment 3 [--port COM3] [--audio-device NAME]
        Run the full experiment for participant 3 (data/003/exp.json). Creates
        the participant (randomising the cue map) on first launch, resumes from
        where it left off otherwise, and gates between runs behind a 3 s FSR
        hold. trials-per-run and runs-per-task default to 60 and 5.

    python main.py block [--port COM3] [--trials 60] [--modality identity]
        Run a single standalone block (no data files).

    python main.py audio-devices | test-audio | preview-cues
        Utilities for checking the audio output and cue shapes.

Quit any time with Q or Escape.
"""

import argparse
import json
import math
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import traceback
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

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
PROGRESS_COLOR = (30, 160, 60)  # Fill colour of the hold-to-continue wedge
SYMBOL_HALF = 80  # Circumradius of cue symbols (px) — matched across shapes
FIX_HALF = 20  # Half-size of the fixation cross (px)
STAR_INNER_RATIO = 0.6  # Star inner/outer radius: higher -> chubbier, less pointy
HOLD_RADIUS = 55  # Radius of the hold-to-continue fill circle (px)

# ── Experiment / persistence ────────────────────────────────────────────────

SCHEMA_VERSION = 1
SOFTWARE_VERSION = "0.1"
DATA_ROOT = Path(__file__).parent / "data"
EXP_FILENAME = "exp.json"

DEFAULT_RUNS_PER_TASK = 5
DEFAULT_TRIALS_PER_RUN = 60
TASK_ORDER = ["identity", "intensity"]  # TODO: counterbalance once intensity lands

HOLD_SECONDS = 1.5  # Continuous FSR press needed to advance (hold-to-continue)
HOLD_PROMPT = "Press and hold the surface to continue."
END_OF_RUN_PAUSE_S = 1.0  # Blank pause after a run, before the break screen
BETWEEN_RUN_LINES = ["Take a short break."]

# LSL source ids — stored per session so recordings can be matched to sessions.
SOURCE_ID_FSR = "fsr_arduino"
SOURCE_ID_MARKERS = "arduino_bridge"


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


def cue_map_to_dict(cue_map: dict[Condition, Shape]) -> dict[str, str]:
    """{'PL': 'diamond', ...} for JSON storage."""
    return {c.value: cue_map[c].value for c in (Condition.PL, Condition.PH, Condition.MIX)}


def cue_map_from_dict(data: dict[str, str]) -> dict[Condition, Shape]:
    return {Condition(c): Shape(s) for c, s in data.items()}


# ── Audio ──────────────────────────────────────────────────────────────────────


def _output_devices() -> list[dict]:
    """PTB output devices (with >0 output channels), empty list if unavailable."""
    try:
        import psychtoolbox.audio as ptb_audio

        return [d for d in ptb_audio.get_devices() if d["NrOutputChannels"] > 0]
    except Exception as exc:
        print(f"(psychtoolbox enumeration failed: {exc}; using PsychoPy)")
        try:
            from psychopy.sound.backend_ptb import getDevices

            return list(getDevices(kind="output").values())
        except Exception:
            return []


def list_audio_devices() -> int:
    """Print the PTB output devices so a name can be passed to --audio-device."""
    for d in _output_devices():
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
        # PsychoPy matches the device name exactly. If the requested name isn't
        # in the enumeration, don't set it (an unmatched name yields silence) —
        # warn and fall back to the system default instead.
        if device:
            names = [d["DeviceName"] for d in _output_devices()]
            if names and device not in names:
                print(
                    f"[audio] WARNING: device {device!r} not found; using system "
                    f"default. Run 'audio-devices' for the exact names."
                )
            else:
                prefs.hardware["audioDevice"] = device
        selected = prefs.hardware.get("audioDevice") or "(system default)"
        print(f"[audio] output device: {selected}")
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
        self.current_force = 0.0  # Latest FSR sample, for the hold-to-continue gate

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
                        self.current_force = value
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


def _pie_points(
    cx: float, cy: float, r: float, progress: float
) -> list[tuple[float, float]]:
    """Polygon of a wedge filling clockwise from 12 o'clock by ``progress`` 0..1."""
    sweep = 360.0 * max(0.0, min(1.0, progress))
    steps = max(2, int(sweep // 3) + 1)
    pts = [(cx, cy)]
    for i in range(steps + 1):
        a = math.radians(sweep * i / steps)
        pts.append((cx + r * math.sin(a), cy - r * math.cos(a)))
    return pts


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
        self.font = pygame.font.SysFont(None, 42)
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

    def _blit_block(self, lines: list[str], cx: int, center_y: int) -> None:
        """Blit a block of centred text lines, vertically centred on ``center_y``."""
        line_h = self.font.get_linesize()
        top = center_y - (len(lines) - 1) * line_h // 2
        for i, text in enumerate(lines):
            if not text:
                continue
            surf = self.font.render(text, True, BLACK)
            rect = surf.get_rect(center=(cx, top + i * line_h))
            self.screen.blit(surf, rect)

    def show_message(self, lines: list[str]) -> None:
        cx, cy = self._center
        self.screen.fill(WHITE)
        self._blit_block(lines, cx, cy)
        pygame.display.flip()

    def show_hold_progress(
        self, progress: float, lines: list[str], shape: Shape | None = None
    ) -> None:
        """Optional cue + message, a pie-filling wedge, and the hold prompt.

        Layout top-to-bottom: cue shape (if any), message lines, the wedge that
        fills clockwise with ``progress`` 0..1 (nothing shown until the press
        begins), and HOLD_PROMPT beneath it.
        """
        cx, cy = self._center
        self.screen.fill(WHITE)
        if shape is not None:
            pygame.draw.polygon(
                self.screen, SHAPE_COLORS[shape],
                _shape_points(shape, cx, cy - 240, SYMBOL_HALF),
            )
        self._blit_block(lines, cx, cy - 70)

        circle_cy = cy + 130
        if progress > 0:
            pygame.draw.polygon(
                self.screen, PROGRESS_COLOR, _pie_points(cx, circle_cy, HOLD_RADIUS, progress)
            )

        prompt = self.font.render(HOLD_PROMPT, True, BLACK)
        self.screen.blit(prompt, prompt.get_rect(center=(cx, circle_cy + HOLD_RADIUS + 40)))
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


def wait_for_hold(
    reader: SerialReader,
    display: Display,
    lines: list[str],
    shape: Shape | None = None,
    hold_s: float = HOLD_SECONDS,
) -> bool:
    """Block until the FSR is held above threshold continuously for ``hold_s``.

    Drives the pie-filling animation: the wedge fills with the elapsed hold time
    and resets to empty the moment the press is released. A press carried over
    from the previous screen does not count — the finger must first be seen up
    (below threshold) so filling only ever starts on a fresh press. Returns
    False if the user quit instead.
    """
    hold_start: float | None = None
    armed = False  # Becomes True once the finger has been released at least once.
    while True:
        display.pump()
        if display.quit:
            return False

        now = time.perf_counter()
        pressing = reader.current_force >= reader.threshold
        if not armed:
            armed = not pressing  # Wait for the carried-over press to be lifted.
            hold_start = None
            progress = 0.0
        elif pressing:
            if hold_start is None:
                hold_start = now
            held = now - hold_start
            if held >= hold_s:
                return True
            progress = held / hold_s
        else:
            hold_start = None
            progress = 0.0

        display.show_hold_progress(progress, lines, shape)
        time.sleep(0.005)


# ── Block runner ──────────────────────────────────────────────────────────────


def classify_response(rt: float) -> str:
    if rt < RESPONSE_MIN_S:
        return "early"
    if rt > RESPONSE_MAX_S:
        return "late"
    return "valid"


@dataclass
class TrialResult:
    """What the participant actually saw and did on one trial."""

    index: int
    condition: Condition
    tone: Tone  # Outcome played (low/high) — keeps MIX-low vs MIX-high distinct.
    shape: Shape
    cue_t: float
    press_t: float | None
    force: float | None
    rt_s: float | None
    result: str  # valid | early | late | timeout


def trial_result_to_dict(tr: TrialResult) -> dict:
    return {
        "i": tr.index,
        "condition": tr.condition.value,
        "outcome": tr.tone.value,
        "shape": tr.shape.value,
        "cue_t_lsl": round(tr.cue_t, 6),
        "press_t_lsl": round(tr.press_t, 6) if tr.press_t is not None else None,
        "force": round(tr.force, 1) if tr.force is not None else None,
        "rt_s": round(tr.rt_s, 4) if tr.rt_s is not None else None,
        "result": tr.result,
    }


def present_trial(
    index: int,
    trial: Trial,
    shape: Shape,
    reader: SerialReader,
    display: Display,
    markers: StreamOutlet,
) -> TrialResult | None:
    """Run one trial (ITI -> cue -> wait for press). None if the user quit."""
    # ── Inter-trial interval (fixation) ───────────────────────────────────────
    display.show_fixation()
    markers.push_sample([f"trial_start {index}"], local_clock())
    if not wait(random.uniform(ITI_MIN_S, ITI_MAX_S), display):
        return None

    # ── Cue (300 ms) ──────────────────────────────────────────────────────────
    display.show_cue(shape)
    cue_t = local_clock()
    markers.push_sample([f"cue {trial.condition.value}"], cue_t)
    # Record the planned outcome so MIX trials are decodable downstream.
    markers.push_sample([f"outcome {trial.tone.value}"], cue_t)
    if not wait(CUE_DURATION_S, display):
        return None

    # ── Fixation + wait for press ─────────────────────────────────────────────
    display.show_fixation()
    markers.push_sample(["cue_offset"], local_clock())
    reader.arm(trial.tone)
    press = wait_for_press(reader, RESPONSE_TIMEOUT_S, display)
    reader.disarm()

    if display.quit:
        return None

    if press is None:
        markers.push_sample([f"trial_timeout {index}"], local_clock())
        return TrialResult(
            index, trial.condition, trial.tone, shape, cue_t, None, None, None, "timeout"
        )

    rt = press.t - cue_t
    return TrialResult(
        index, trial.condition, trial.tone, shape, cue_t, press.t, press.force, rt,
        classify_response(rt),
    )


def print_trial(tr: TrialResult, n: int) -> None:
    if tr.result == "timeout":
        print(f"  trial {tr.index:3d}/{n}: {tr.condition.value}/{tr.tone.value}  timeout")
    else:
        print(
            f"  trial {tr.index:3d}/{n}: {tr.condition.value}/{tr.tone.value}  "
            f"rt={tr.rt_s:.3f}s force={tr.force:.1f} [{tr.result}]"
        )


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
        tr = present_trial(i, trial, cue_map[trial.condition], reader, display, markers)
        if tr is None:  # user quit
            break
        print_trial(tr, n)

    markers.push_sample(["block_end"], local_clock())
    print("Block complete.")


# ── LSL ────────────────────────────────────────────────────────────────────────


def create_outlets() -> tuple[StreamOutlet, StreamOutlet]:
    fsr = StreamOutlet(
        StreamInfo("FSR_force", "FSR", 1, SAMPLE_RATE, "float32", SOURCE_ID_FSR)
    )
    markers = StreamOutlet(
        StreamInfo("Markers", "Markers", 1, IRREGULAR_RATE, "string", SOURCE_ID_MARKERS)
    )
    return fsr, markers


# ── Persistence (exp.json) ─────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def config_snapshot(threshold: float) -> dict:
    """Timing/tone/threshold values actually in effect, for provenance."""
    return {
        "cue_duration_s": CUE_DURATION_S,
        "iti_s": [ITI_MIN_S, ITI_MAX_S],
        "response_window_s": [RESPONSE_MIN_S, RESPONSE_MAX_S],
        "response_timeout_s": RESPONSE_TIMEOUT_S,
        "tone": {
            "low_hz": TONE_FREQ_LOW,
            "high_hz": TONE_FREQ_HIGH,
            "duration_s": TONE_DURATION_S,
            "delay_s": 0.0,
        },
        "press_threshold": threshold,
        "fsr_sample_rate": SAMPLE_RATE,
    }


class ExperimentStore:
    """Read/modify/atomically-save a participant's exp.json.

    Holds the JSON as a plain dict (mirrors the file exactly) so live appends —
    e.g. one trial at a time — stay trivial and serialization can't drift.
    """

    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data
        self._next_session = 1 + max((s["id"] for s in data["sessions"]), default=0)
        self._next_run = 1 + max((r["id"] for r in data["runs"]), default=0)

    # ── create / load ─────────────────────────────────────────────────────────

    @classmethod
    def open_or_create(
        cls, participant_id: str, threshold: float, hardware: dict,
        runs_per_task: int, trials_per_run: int,
    ) -> "ExperimentStore":
        path = DATA_ROOT / participant_id / EXP_FILENAME
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                return cls(path, json.load(fh))

        # New participant: randomise the cue map once, now.
        cue_map = parse_cue_map("random")
        data = {
            "schema_version": SCHEMA_VERSION,
            "participant_id": participant_id,
            "created_at": _now_iso(),
            "design": {
                "task_order": list(TASK_ORDER),
                "cue_map": cue_map_to_dict(cue_map),
                "runs_per_task": runs_per_task,
                "trials_per_run": trials_per_run,
            },
            "subject": {"age": None, "sex": None, "dominant_hand": None, "notes": ""},
            "defaults": config_snapshot(threshold),
            "hardware": hardware,
            "calibrations": [],
            "sessions": [],
            "runs": [],
        }
        store = cls(path, data)
        store.save()
        return store

    def save(self) -> None:
        """Atomic write: tmp file + os.replace (atomic on Windows and POSIX)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    # ── sessions ──────────────────────────────────────────────────────────────

    def recover_crashes(self) -> int:
        """Mark prior sessions with no end_time as crashed; abort their runs.

        Called before the current session is registered, so every open-ended
        session found here belongs to a previous process launch.
        """
        crashed = {
            s["id"] for s in self.data["sessions"] if s["end_time"] is None
        }
        for s in self.data["sessions"]:
            if s["id"] in crashed:
                s["clean_exit"] = False
        for r in self.data["runs"]:
            if r["status"] == "in_progress" and r["session_id"] in crashed:
                r["status"] = "aborted"
                r["abort_reason"] = "process_exited"
        return len(crashed)

    def start_session(self, experimenter: str | None) -> int:
        session_id = self._next_session
        self._next_session += 1
        self.data["sessions"].append({
            "id": session_id,
            "start_time": _now_iso(),
            "end_time": None,
            "clean_exit": False,
            "experimenter": experimenter or "",
            "software": {"git_commit": _git_commit(), "version": SOFTWARE_VERSION},
            "machine": {
                "hostname": socket.gethostname(),
                "os": sys.platform,
                "python": platform.python_version(),
            },
            "lsl_source_ids": {"fsr": SOURCE_ID_FSR, "markers": SOURCE_ID_MARKERS},
        })
        return session_id

    def end_session(self, session_id: int, clean_exit: bool) -> None:
        for s in self.data["sessions"]:
            if s["id"] == session_id:
                s["end_time"] = _now_iso()
                s["clean_exit"] = clean_exit
                return

    # ── runs ──────────────────────────────────────────────────────────────────

    def completed_count(self, task: str) -> int:
        return sum(
            1 for r in self.data["runs"]
            if r["task"] == task and r["status"] == "completed"
        )

    def next_task(self) -> str | None:
        """First task in order still short of its run target, else None."""
        target = self.data["design"]["runs_per_task"]
        for task in self.data["design"]["task_order"]:
            if self.completed_count(task) < target:
                return task
        return None

    def start_run(
        self, session_id: int, task: str, cue_map: dict[Condition, Shape],
        threshold: float, seed: int, recording_file: str,
    ) -> int:
        run_id = self._next_run
        self._next_run += 1
        self.data["runs"].append({
            "id": run_id,
            "session_id": session_id,
            "task": task,
            "status": "in_progress",
            "start_time": _now_iso(),
            "end_time": None,
            "cue_map": cue_map_to_dict(cue_map),
            "config": config_snapshot(threshold),
            "rng_seed": seed,
            "recording_file": recording_file,
            "abort_reason": None,
            "trials": [],
        })
        return run_id

    def _run(self, run_id: int) -> dict:
        return next(r for r in self.data["runs"] if r["id"] == run_id)

    def append_trial(self, run_id: int, trial: dict) -> None:
        self._run(run_id)["trials"].append(trial)

    def finish_run(self, run_id: int, status: str, abort_reason: str | None) -> None:
        run = self._run(run_id)
        run["status"] = status
        run["end_time"] = _now_iso()
        run["abort_reason"] = abort_reason


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


def suggested_recording_path(participant_id: str, task: str, ordinal: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(DATA_ROOT / participant_id / f"{task}_run{ordinal}_{stamp}.xdf")


def run_one(
    store: ExperimentStore,
    session_id: int,
    task: str,
    cue_map: dict[Condition, Shape],
    reader: SerialReader,
    display: Display,
    markers: StreamOutlet,
    threshold: float,
) -> str:
    """Run one block, persisting each trial live. Returns the final run status."""
    design = store.data["design"]
    trials_per_run = design["trials_per_run"]
    ordinal = store.completed_count(task) + 1

    seed = random.randrange(2**31)
    random.seed(seed)  # Make the trial order (and ITIs) reproducible from the seed.
    trials = build_block(trials_per_run)

    recording_file = suggested_recording_path(store.data["participant_id"], task, ordinal)
    run_id = store.start_run(session_id, task, cue_map, threshold, seed, recording_file)
    store.save()

    print(f"\n=== {task} run {ordinal}/{design['runs_per_task']} (run id {run_id}) ===")
    markers.push_sample([f"run_start id={run_id} task={task} ordinal={ordinal}"], local_clock())
    markers.push_sample([f"cue_map {format_cue_map(cue_map)}"], local_clock())

    status, reason = "completed", None
    try:
        for i, trial in enumerate(trials, start=1):
            tr = present_trial(i, trial, cue_map[trial.condition], reader, display, markers)
            if tr is None:  # user quit mid-run
                status, reason = "aborted", "user_quit"
                break
            store.append_trial(run_id, trial_result_to_dict(tr))
            store.save()
            print_trial(tr, len(trials))
    finally:
        markers.push_sample([f"run_end id={run_id} status={status}"], local_clock())
        store.finish_run(run_id, status, reason)
        store.save()
    return status


def run_experiment_loop(
    store: ExperimentStore,
    session_id: int,
    cue_map: dict[Condition, Shape],
    reader: SerialReader,
    display: Display,
    markers: StreamOutlet,
    threshold: float,
) -> None:
    first = True
    while not display.quit:
        task = store.next_task()
        if task is None:
            display.show_message(["Experiment complete.", "", "Thank you!"])
            wait(3.0, display)
            break

        if Modality(task) != Modality.IDENTITY:  # TODO: intensity support
            print(f"Next task '{task}' is not implemented yet; stopping.")
            display.show_message([f"The '{task}' task is not ready yet.", "", "Stopping here."])
            wait(3.0, display)
            break

        # Go straight into the first run; gate subsequent runs behind a hold.
        if not first and not wait_for_hold(reader, display, BETWEEN_RUN_LINES):
            break
        first = False

        if run_one(store, session_id, task, cue_map, reader, display, markers, threshold) == "aborted":
            break

        # Brief blank pause after the block before the break / completion screen.
        display.show_fixation()
        if not wait(END_OF_RUN_PAUSE_S, display):
            break


def validate_study_params(trials_per_run: int, runs_per_task: int) -> str | None:
    """Return an error message if the study-size parameters are invalid."""
    if trials_per_run <= 0 or trials_per_run % 4 != 0:
        return f"--trials-per-run must be a positive multiple of 4 (got {trials_per_run})"
    if runs_per_task <= 0:
        return f"--runs-per-task must be positive (got {runs_per_task})"
    return None


def has_completed_runs(store: ExperimentStore) -> bool:
    return any(r["status"] == "completed" for r in store.data["runs"])


def cmd_experiment(args: argparse.Namespace) -> int:
    participant_id = f"{args.participant:03d}"

    # Validate the study size BEFORE touching disk, so a bad value can never be
    # written into (and get stuck in) a participant folder.
    msg = validate_study_params(args.trials_per_run, args.runs_per_task)
    if msg:
        print(msg)
        return 1

    participant_path = DATA_ROOT / participant_id
    was_new = not participant_path.exists()

    hardware = {
        "emg": {"array": None, "placement": None, "sample_rate_hz": None},
        "audio_device": args.audio_device,
        "earplugs": "",
        "serial_port": args.port,
    }

    store: ExperimentStore | None = None
    reader: SerialReader | None = None
    display: Display | None = None
    session_id: int | None = None
    ok = False
    try:
        store = ExperimentStore.open_or_create(
            participant_id, args.threshold, hardware,
            args.runs_per_task, args.trials_per_run,
        )
        # Guard against an older/hand-edited folder with an unusable size.
        stored_trials = store.data["design"]["trials_per_run"]
        if stored_trials % 4 != 0:
            print(
                f"Participant {participant_id} has an invalid stored "
                f"trials_per_run={stored_trials}; its folder is unusable. "
                f"Delete {participant_path} to recreate it."
            )
            return 1

        if store.recover_crashes():
            print("Recovered crashed session(s); their open runs marked aborted.")
        store.save()

        cue_map = cue_map_from_dict(store.data["design"]["cue_map"])
        target = store.data["design"]["runs_per_task"]
        print(f"Participant {participant_id} | cue_map={format_cue_map(cue_map)}")
        for task in store.data["design"]["task_order"]:
            print(f"  {task}: {store.completed_count(task)}/{target} runs completed")

        # Bring up audio + serial before the fullscreen window, so a failure
        # never leaves a black screen hanging around.
        fsr_outlet, marker_outlet = create_outlets()
        audio = AudioEngine(Modality.IDENTITY, device=args.audio_device)
        reader = SerialReader(args.port, args.threshold, audio, fsr_outlet, marker_outlet)
        reader.start()
        time.sleep(0.5)  # Let the serial port settle.
        if reader.error is not None:
            print(f"Serial error: {reader.error}")
        else:
            display = Display()
            session_id = store.start_session(args.experimenter)
            store.save()
            run_experiment_loop(
                store, session_id, cue_map, reader, display, marker_outlet, args.threshold
            )
            ok = True
    except Exception:
        traceback.print_exc()
    finally:
        if reader is not None:
            reader.stop()
        if session_id is not None and store is not None:
            store.end_session(session_id, clean_exit=ok)
        # On failure, discard a freshly created participant that produced no
        # completed run, so a bad first attempt leaves nothing to clean up.
        rolled_back = False
        if not ok and store is not None and was_new and not has_completed_runs(store):
            shutil.rmtree(participant_path, ignore_errors=True)
            rolled_back = True
            print(f"No runs completed — removed freshly created {participant_path}.")
        if store is not None and not rolled_back:
            store.save()
        if display is not None:
            display.close()
    return 0 if ok else 1


# ── Training ───────────────────────────────────────────────────────────────

WELCOME_LINES = [
    "Welcome, and thank you for taking part!",
    "",
    "This short training will walk you through the task",
    "before the real experiment begins.",
]

DESCRIPTION_LINES = [
    "On each trial a symbol appears in the centre of the screen.",
    "Wait about one second after it appears,",
    "then tap the sensor once with your finger.",
    "",
    "Each tap produces a short tone.",
    "Keep the rest of your hand still and relaxed.",
]

# Per symbol: the condition, a description of its outcome, and the fixed outcome
# sequence to practise. PH/PL have one outcome; MIX demonstrates both.
TRAINING_SYMBOLS = [
    (Condition.PH, "a HIGH tone", [Tone.HIGH, Tone.HIGH, Tone.HIGH]),
    (Condition.PL, "a LOW tone", [Tone.LOW, Tone.LOW, Tone.LOW]),
    (Condition.MIX, "either a HIGH or a LOW tone", [Tone.HIGH, Tone.LOW, Tone.LOW]),
]

TRAINING_SUCCESS = ("valid", "late")  # Misses to retry: "early" and "timeout".


def practice_symbol(
    condition: Condition,
    outcomes: list[Tone],
    cue_map: dict[Condition, Shape],
    reader: SerialReader,
    display: Display,
    markers: StreamOutlet,
) -> bool:
    """Practise one symbol: repeat each outcome until a good tap. False if quit."""
    shape = cue_map[condition]
    markers.push_sample([f"training_practice {condition.value}"], local_clock())
    attempt = 0
    for target in outcomes:
        while True:
            attempt += 1
            tr = present_trial(attempt, Trial(condition, target), shape, reader, display, markers)
            if tr is None:
                return False
            if tr.result in TRAINING_SUCCESS:
                print(f"  practice {condition.value}/{target.value}: {tr.result} OK")
                break
            print(f"  practice {condition.value}/{target.value}: {tr.result} — retry")
            if tr.result == "early":
                feedback = ["Too early!", "Wait about a second after the",
                            "symbol appears, then tap once."]
            else:  # timeout
                feedback = ["No tap detected.", "Tap the sensor once when",
                            "you see the symbol."]
            display.show_message(feedback)
            if not wait(2.0, display):
                return False
    return True


def run_training(
    cue_map: dict[Condition, Shape],
    reader: SerialReader,
    display: Display,
    markers: StreamOutlet,
) -> None:
    markers.push_sample(["training_start"], local_clock())

    if not wait_for_hold(reader, display, WELCOME_LINES):
        return
    if not wait_for_hold(reader, display, DESCRIPTION_LINES):
        return

    for condition, outcome_desc, outcomes in TRAINING_SYMBOLS:
        intro = [
            "When you see this symbol,",
            f"your tap will produce {outcome_desc}.",
            "",
            "We will now practise this symbol.",
        ]
        if not wait_for_hold(reader, display, intro, shape=cue_map[condition]):
            return
        if not practice_symbol(condition, outcomes, cue_map, reader, display, markers):
            return

    markers.push_sample(["training_end"], local_clock())
    display.show_message(["Training complete.", "", "Well done!"])
    wait(3.0, display)


def cmd_train(args: argparse.Namespace) -> int:
    participant_id = f"{args.participant:03d}"
    hardware = {
        "emg": {"array": None, "placement": None, "sample_rate_hz": None},
        "audio_device": args.audio_device,
        "earplugs": "",
        "serial_port": args.port,
    }
    # Use the participant's stored cue map (creating them — and fixing the map —
    # on first contact) so training matches the real experiment exactly.
    store = ExperimentStore.open_or_create(
        participant_id, args.threshold, hardware,
        DEFAULT_RUNS_PER_TASK, DEFAULT_TRIALS_PER_RUN,
    )
    cue_map = cue_map_from_dict(store.data["design"]["cue_map"])
    print(f"Training participant {participant_id} | cue_map={format_cue_map(cue_map)}")

    fsr_outlet, marker_outlet = create_outlets()
    audio = AudioEngine(Modality.IDENTITY, device=args.audio_device)
    reader = SerialReader(args.port, args.threshold, audio, fsr_outlet, marker_outlet)
    reader.start()
    time.sleep(0.5)  # Let the serial port settle.
    if reader.error is not None:
        print(f"Serial error: {reader.error}")
        return 1

    display = Display()
    try:
        run_training(cue_map, reader, display, marker_outlet)
    finally:
        reader.stop()
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

    exp = sub.add_parser("experiment", help="Run the experiment for a participant")
    exp.add_argument("participant", type=int, help="Participant number (-> data/NNN)")
    exp.add_argument(
        "--port", default=DEFAULT_PORT, help=f"Serial port (default {DEFAULT_PORT})"
    )
    exp.add_argument(
        "--audio-device",
        default=None,
        help="Exact output device name for the tones; see 'audio-devices'.",
    )
    exp.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"FSR press-detection threshold (default {DEFAULT_THRESHOLD})",
    )
    exp.add_argument("--experimenter", default=None, help="Experimenter name (logged)")
    # Study structure — hard-coded defaults, overridable but rarely needed. Only
    # used when a participant is first created; afterwards the stored design wins.
    exp.add_argument(
        "--runs-per-task", type=int, default=DEFAULT_RUNS_PER_TASK,
        help=f"Runs per task for a NEW participant (default {DEFAULT_RUNS_PER_TASK})",
    )
    exp.add_argument(
        "--trials-per-run", type=int, default=DEFAULT_TRIALS_PER_RUN,
        help=f"Trials per run for a NEW participant (default {DEFAULT_TRIALS_PER_RUN})",
    )
    exp.set_defaults(func=cmd_experiment)

    train = sub.add_parser("train", help="Run the training sequence for a participant")
    train.add_argument("participant", type=int, help="Participant number (-> data/NNN)")
    train.add_argument(
        "--port", default=DEFAULT_PORT, help=f"Serial port (default {DEFAULT_PORT})"
    )
    train.add_argument(
        "--audio-device",
        default=None,
        help="Exact output device name for the tones; see 'audio-devices'.",
    )
    train.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"FSR press-detection threshold (default {DEFAULT_THRESHOLD})",
    )
    train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    if args.command == "audio-devices":
        return list_audio_devices()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
