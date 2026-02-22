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
    # Detect Praat chronological format, which praatio cannot parse.
    try:
        first_bytes = textgrid_path.read_text(encoding="utf-8-sig", errors="replace")[:80]
    except Exception:
        first_bytes = ""
    if "chronological" in first_bytes.lower():
        return _load_chronological_textgrid(textgrid_path, tier_name)

    try:
        tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)
    except (TextgridStateError, ValueError) as original_error:
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
    lines = textgrid_path.read_text(encoding="utf-8-sig").splitlines()
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


def _load_chronological_textgrid(textgrid_path: Path, tier_name: str | None = None) -> list[WordTiming]:
    """
    Parse Praat chronological TextGrid format:
      Line 0: "Praat chronological TextGrid text file"
      Line 1: xmin xmax  ! Time domain.
      Line 2: n_tiers ! Number of tiers.
      Lines 3..3+n_tiers-1: "interval"/"point" "tier_name" xmin xmax
      Remaining: tier_num xmin xmax "text"  (interval) or tier_num time "text" (point)
    """
    content = textgrid_path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    lines = [l.rstrip() for l in content.split("\n")]

    # Parse number of tiers from line 2: "2 ! Number of tiers."
    n_tiers = int(lines[2].split()[0])

    # Parse tier headers to get ordered tier names (lines 3 .. 3+n_tiers-1)
    tier_names: list[str] = []
    for i in range(n_tiers):
        header = lines[3 + i]
        quoted = re.findall(r'"([^"]*)"', header)
        # quoted[0] = type ("interval"/"point"), quoted[1] = tier name
        tier_names.append(quoted[1] if len(quoted) > 1 else quoted[0])

    chosen_name = tier_name or _pick_word_tier(tier_names)
    if chosen_name not in tier_names:
        raise RuntimeError(f"Tier '{chosen_name}' not found in {textgrid_path}; available: {tier_names}")
    target_idx = tier_names.index(chosen_name) + 1  # entries use 1-based tier index

    # Parse interval entries: tier_num xmin xmax "text"
    entry_re = re.compile(r'^(\d+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+"(.*)"$')
    words: list[WordTiming] = []
    for line in lines[3 + n_tiers:]:
        line = line.strip()
        if not line:
            continue
        m = entry_re.match(line)
        if not m:
            continue
        if int(m.group(1)) != target_idx:
            continue
        text = m.group(4).strip()
        if text:
            words.append(WordTiming(word=text, start=float(m.group(2)), end=float(m.group(3))))
    return words
