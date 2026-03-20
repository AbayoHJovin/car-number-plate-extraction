# src/validate.py
"""
Plate text validation using regex patterns.
Supports common East-African and generic international formats.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Regex patterns (add more patterns for your country as needed)
# ---------------------------------------------------------------------------

_PATTERNS = [
    # Rwanda: RAA 000A
    re.compile(r"^R[A-Z]{2}\s?\d{3}[A-Z]$"),
    # Generic: 2-3 letters, 2-4 digits, optional trailing letters
    re.compile(r"^[A-Z]{2,3}\s?\d{2,4}[A-Z]{0,2}$"),
    # All digits (some plates): 4-8 digits
    re.compile(r"^\d{4,8}$"),
]


# Common OCR misreads for number plates
_CORRECTIONS = {
    'I': '1', 'L': '1',
    'O': '0', 'D': '0',
    'S': '5', 'G': '6',
    'B': '8', 'Z': '2',
}

def clean_text(raw: str) -> str:
    """Strip whitespace/newlines and upper-case for matching."""
    return re.sub(r"[^A-Z0-9]", "", raw.upper()).strip()

def fix_ocr_errors(text: str) -> str:
    """
    Intelligent correction based on Rwandan plate format (RAA 123A).
    RAA (letters) 123 (digits) A (letter).
    """
    cleaned = clean_text(text)
    if len(cleaned) < 5:
        return cleaned
        
    chars = list(cleaned)
    # Rwandan format: 
    # Index 0,1,2: Should be letters (Rxx)
    # Index 3,4,5: Should be digits (000)
    # Index 6: Should be a letter (A)
    
    # Only apply if it looks kind of like a Rwanda plate
    if chars[0] == 'R':
        # Fix digits that were read as letters
        for i in range(3, min(6, len(chars))):
            if chars[i] in _CORRECTIONS:
                chars[i] = _CORRECTIONS[chars[i]]
        # Fix letters that were read as digits (simple swap)
        # (Though less common for R and A)
    
    return "".join(chars)

def is_valid_plate(text: str) -> bool:
    """Return True if text matches any known plate pattern."""
    cleaned = clean_text(text)
    if any(p.match(cleaned) for p in _PATTERNS):
        return True
    
    # Try with corrections
    fixed = fix_ocr_errors(cleaned)
    return any(p.match(fixed) for p in _PATTERNS)

def normalise(text: str) -> str:
    """Return a normalised and corrected plate string."""
    return fix_ocr_errors(text)
