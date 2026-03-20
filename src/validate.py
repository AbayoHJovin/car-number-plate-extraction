# src/validate.py
"""
Plate text validation using regex patterns.
Supports Rwandan and generic international formats.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_PATTERNS = [
    # Rwanda: RAA 000A  (e.g. RAB 123C)
    re.compile(r"^R[A-Z]{2}\d{3}[A-Z]$"),
    # Generic: 2-3 letters, 2-4 digits, optional trailing 0-2 letters
    re.compile(r"^[A-Z]{2,3}\d{2,4}[A-Z]{0,2}$"),
    # All digits: 4-8 digits
    re.compile(r"^\d{4,8}$"),
]

# Characters that look like digits but get read as letters by Tesseract
_LETTER_TO_DIGIT = {'I': '1', 'L': '1', 'O': '0', 'D': '0', 'S': '5', 'G': '6', 'B': '8', 'Z': '2'}

# Characters that look like letters but get read as digits by Tesseract
_DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z', '6': 'G'}


def clean_text(raw: str) -> str:
    """Strip everything except A-Z and 0-9, upper-case."""
    return re.sub(r"[^A-Z0-9]", "", raw.upper()).strip()


def fix_ocr_errors(text: str) -> str:
    """
    Position-aware correction for the Rwandan plate format: RXX 000 X
    (positions 0-2 = letters, 3-5 = digits, 6 = letter).

    The original function applied corrections unconditionally — if position 4
    happened to be a digit already (e.g. '1'), the correction map left it
    alone, but if it was 'I' it would correctly fix it.  The bug was that it
    also tried to fix positions 0-2 and position 6 (the letter slots) by
    applying the letter→digit map, turning valid letters into digits.

    This version only corrects a character when the character TYPE is wrong
    for its position, not simply because it appears in the correction map.
    """
    cleaned = clean_text(text)
    if len(cleaned) < 5:
        return cleaned

    chars = list(cleaned)

    # Only attempt the Rwanda-specific fix when the string looks like a
    # Rwanda plate (starts with R and is 7 characters long).
    if chars[0] == 'R' and len(chars) == 7:
        # Positions 0-2 should be letters
        for i in range(0, 3):
            if chars[i].isdigit():
                chars[i] = _DIGIT_TO_LETTER.get(chars[i], chars[i])

        # Positions 3-5 should be digits
        for i in range(3, 6):
            if chars[i].isalpha():
                chars[i] = _LETTER_TO_DIGIT.get(chars[i], chars[i])

        # Position 6 should be a letter
        if chars[6].isdigit():
            chars[6] = _DIGIT_TO_LETTER.get(chars[6], chars[6])

    return "".join(chars)


def is_valid_plate(text: str) -> bool:
    """Return True if text matches any known plate pattern."""
    cleaned = clean_text(text)
    if any(p.match(cleaned) for p in _PATTERNS):
        return True
    # Also try with position-aware correction applied
    fixed = fix_ocr_errors(cleaned)
    return any(p.match(fixed) for p in _PATTERNS)


def normalise(text: str) -> str:
    """Return a normalised and corrected plate string."""
    return fix_ocr_errors(text)


# ---------------------------------------------------------------------------
# Fuzzy matching helper (used by the confirmation buffer in main.py)
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Compute edit distance between two strings."""
    if len(a) < len(b):
        return levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def fuzzy_match(a: str, b: str, max_distance: int = 1) -> bool:
    """
    Return True if two plate strings are within max_distance edits of each
    other.  This lets the confirmation buffer accumulate observations even
    when Tesseract flips a single character between frames.
    """
    return levenshtein(clean_text(a), clean_text(b)) <= max_distance