"""Minimal stand-in for the host app's privacy filter, used only to redact Kobox's local
rejection log (emails and phone numbers become [EMAIL] / [PHONE])."""
from __future__ import annotations

import re

_RX = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "EMAIL"), (re.compile(r"\+?\d[\d -]{7,}\d"), "PHONE")]


def find(text: str) -> list[tuple[int, int, str]]:
    spans = []
    for rx, kind in _RX:
        spans += [(m.start(), m.end(), kind) for m in rx.finditer(text or "")]
    return sorted(spans)
