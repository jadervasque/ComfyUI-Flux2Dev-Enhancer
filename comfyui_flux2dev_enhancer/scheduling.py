"""Schedule parsing and architecture-aware strength helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


class ScheduleParseError(ValueError):
    """Raised when a custom transformer-block schedule is invalid."""


@dataclass(frozen=True)
class IdentityPreset:
    name: str
    similarity_floor: float
    temperature: float
    mask_threshold: float
    total_strength: float
    double_schedule: dict[int, float]
    single_schedule: dict[int, float]


def parse_block_schedule(
    text: str | None,
    max_block: int,
    *,
    strict: bool = True,
    allowed_keys: Sequence[str] = ("mid", "mid_img", "strength"),
) -> dict[int, float]:
    """Parse a schedule such as ``0-3:mid_img=0.25; 5:0.4``."""

    if max_block < 0:
        return {}
    result: dict[int, float] = {}
    allowed = {item.lower() for item in allowed_keys}
    for raw_row in str(text or "").split(";"):
        row = raw_row.strip()
        if not row:
            continue
        if ":" not in row:
            if strict:
                raise ScheduleParseError(f"Missing ':' in schedule entry {row!r}.")
            continue
        block_part, value_part = (part.strip() for part in row.split(":", 1))
        if "=" in value_part:
            key, value_part = (part.strip() for part in value_part.split("=", 1))
            if key.lower() not in allowed:
                if strict:
                    raise ScheduleParseError(f"Unsupported schedule key {key!r} in {row!r}.")
                continue
        try:
            strength = float(value_part)
        except ValueError as exc:
            if strict:
                raise ScheduleParseError(f"Invalid strength in schedule entry {row!r}.") from exc
            continue
        if not math.isfinite(strength) or strength < 0.0:
            raise ScheduleParseError(f"Strength must be finite and non-negative in {row!r}.")

        try:
            if "-" in block_part:
                lo_text, hi_text = block_part.split("-", 1)
                lo, hi = int(lo_text.strip()), int(hi_text.strip())
            else:
                lo = hi = int(block_part)
        except ValueError as exc:
            if strict:
                raise ScheduleParseError(f"Invalid block range in schedule entry {row!r}.") from exc
            continue
        if lo > hi:
            lo, hi = hi, lo
        if strict and (lo < 0 or hi > max_block):
            raise ScheduleParseError(
                f"Block range {lo}-{hi} exceeds the valid range 0-{max_block}."
            )
        lo, hi = max(0, lo), min(max_block, hi)
        for block_index in range(lo, hi + 1):
            result[block_index] = strength
    return result


def format_block_schedule(schedule: Mapping[int, float]) -> str:
    return "; ".join(
        f"{idx}:mid_img={value:.6g}" for idx, value in sorted(schedule.items())
    )


def normalized_per_application(total_strength: float, active_applications: int) -> float:
    """Approximate a per-application blend that composes to ``total_strength``."""

    strength = min(max(float(total_strength), 0.0), 1.0)
    if active_applications <= 0 or strength <= 0.0:
        return 0.0
    if strength >= 1.0:
        return 1.0
    return 1.0 - (1.0 - strength) ** (1.0 / active_applications)


def parse_reference_indices(text: str | None, count: int, fallback: int = 0) -> list[int]:
    if count <= 0:
        return []
    value = str(text or "all").strip().lower()
    if value in {"", "all", "*"}:
        return list(range(count))
    selected: set[int] = set()
    for part in re.split(r"[;,\s]+", value):
        if not part:
            continue
        try:
            if "-" in part:
                lo_text, hi_text = part.split("-", 1)
                lo, hi = int(lo_text), int(hi_text)
                if lo > hi:
                    lo, hi = hi, lo
                selected.update(idx for idx in range(lo, hi + 1) if 0 <= idx < count)
            else:
                idx = int(part)
                if 0 <= idx < count:
                    selected.add(idx)
        except ValueError:
            continue
    if selected:
        return sorted(selected)
    return [min(max(int(fallback), 0), count - 1)]


def relative_sparse_schedule(
    depth: int, points: Iterable[tuple[float, float]]
) -> dict[int, float]:
    if depth <= 0:
        return {}
    result: dict[int, float] = {}
    for position, strength in points:
        index = round(min(max(float(position), 0.0), 1.0) * (depth - 1))
        result[index] = max(result.get(index, 0.0), float(strength))
    return result


def auto_identity_preset(name: str, double_depth: int, single_depth: int) -> IdentityPreset:
    key = str(name).upper()
    if key == "AUTO_SOFT":
        double = relative_sparse_schedule(
            double_depth, [(0.25, 0.30), (0.55, 0.35), (0.80, 0.25)]
        )
        single = relative_sparse_schedule(
            single_depth, [(0.35, 0.22), (0.55, 0.25), (0.75, 0.22)]
        )
        return IdentityPreset(key, 0.50, 0.08, 0.85, 0.45, double, single)
    if key == "AUTO_STRONG":
        double = relative_sparse_schedule(
            double_depth,
            [(x / max(double_depth - 1, 1), 0.55) for x in range(double_depth)],
        )
        single = relative_sparse_schedule(
            single_depth,
            [(0.05, 0.28), (0.15, 0.30), (0.30, 0.30), (0.50, 0.28), (0.72, 0.25)],
        )
        return IdentityPreset(key, 0.04, 0.025, 1.0, 0.85, double, single)
    double = relative_sparse_schedule(
        double_depth,
        [(0.15, 0.42), (0.40, 0.50), (0.65, 0.45), (0.85, 0.35)],
    )
    single = relative_sparse_schedule(
        single_depth,
        [(0.20, 0.24), (0.38, 0.27), (0.58, 0.26), (0.78, 0.22)],
    )
    return IdentityPreset("AUTO_BALANCED", 0.20, 0.07, 0.95, 0.65, double, single)


__all__ = [
    "IdentityPreset",
    "ScheduleParseError",
    "auto_identity_preset",
    "format_block_schedule",
    "normalized_per_application",
    "parse_block_schedule",
    "parse_reference_indices",
    "relative_sparse_schedule",
]
