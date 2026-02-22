from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tempfile

from praatio import textgrid
from praatio.utilities.errors import TextgridStateError


@dataclass
class WordTiming:
    word: str
    start: float
    end: float


def load_word_timings(textgrid_path: Path, tier_name: str | None = None) -> list[WordTiming]:
    try:
        tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)
    except TextgridStateError as original_error:
        fixed_content = _repair_textgrid_interval_boundaries(textgrid_path)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".TextGrid",
                delete=False,
                encoding="utf-8",
                newline="\n",
            ) as tmp:
                tmp.write(fixed_content)
                tmp_path = tmp.name
            tg = textgrid.openTextgrid(tmp_path, includeEmptyIntervals=False)
        except Exception:
            msg = f"{original_error} [file: {textgrid_path}]"
            raise TextgridStateError(msg) from original_error
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)

    names = tg.tierNames
    chosen_name = tier_name or _pick_word_tier(names)
    tier = tg.getTier(chosen_name)
    entries = getattr(tier, "entries", None)
    if entries is None:
        entries = getattr(tier, "entryList", None)
    if entries is None:
        raise RuntimeError(f"Unsupported praatio tier format for {textgrid_path}")

    words: list[WordTiming] = []
    for start, end, token in entries:
        word = token.strip()
        if not word:
            continue
        words.append(WordTiming(word=word, start=float(start), end=float(end)))
    return words


def _pick_word_tier(tier_names: list[str]) -> str:
    lowered = {name.lower(): name for name in tier_names}
    for candidate in ("words", "word", "transcript"):
        if candidate in lowered:
            return lowered[candidate]
    return tier_names[0]


def _repair_textgrid_interval_boundaries(textgrid_path: Path) -> str:
    """
    Normalize TextGrid numeric precision and snap interval starts to previous
    interval ends to fix tiny floating-point overlaps between adjacent intervals.
    """
    lines = textgrid_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []

    number_line = re.compile(r"^(\s*)(xmin|xmax)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$")
    interval_header = re.compile(r"^\s*intervals\s*\[\d+\]\s*:\s*$")
    item_header = re.compile(r"^\s*item\s*\[\d+\]\s*:\s*$")

    in_interval = False
    prev_interval_end: float | None = None

    for line in lines:
        if item_header.match(line):
            in_interval = False
            prev_interval_end = None
            out.append(line)
            continue

        if interval_header.match(line):
            in_interval = True
            out.append(line)
            continue

        m = number_line.match(line)
        if not m:
            out.append(line)
            continue

        indent, key, raw_val = m.groups()
        value = round(float(raw_val), 6)

        if in_interval and key == "xmin":
            if prev_interval_end is not None:
                value = prev_interval_end
        elif in_interval and key == "xmax":
            prev_interval_end = value

        out.append(f"{indent}{key} = {value:.6f}")

    return "\n".join(out) + "\n"
