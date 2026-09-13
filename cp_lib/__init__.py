"""Contact point and swing timing reconstruction.

See docs/superpowers/specs/2026-09-13-contact-point-design.md.
"""
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SEASONS = (2024, 2025, 2026)
FIT_SEASON = 2025
