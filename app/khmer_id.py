"""Helpers for locating fields on a Cambodian National ID card."""
from dataclasses import dataclass

import pytesseract
from PIL import Image

# The Khmer name label on Cambodian IDs is "គោត្តនាម និង នាម:" — always contains "នាម".
NAME_LABEL_KEYWORD = "នាម"
COLON_CHARS = (":", "៖", "ៈ")


def extract_after_colon(text: str) -> str:
    """Return everything after the first colon on the first line.
    Tesseract's full-line OCR often gets the name perfectly right but with the
    label still attached ("...នាមៈជាន សុវត្ថិ"). Splitting at the colon recovers it."""
    if not text:
        return ""
    first_line = text.split("\n", 1)[0]
    for c in COLON_CHARS:
        if c in first_line:
            return first_line.split(c, 1)[1].strip()
    return first_line.strip()


@dataclass
class KhmerIDNameLocation:
    line_bbox: tuple[int, int, int, int]   # full label + name (left, top, right, bottom)
    name_bbox: tuple[int, int, int, int]   # name-only sub-region (right of the colon)
    label_text: str                         # raw Tesseract reading of the line


def pad_bbox(bbox: tuple[int, int, int, int], max_size: tuple[int, int], pad: int = 20):
    x0, y0, x1, y1 = bbox
    W, H = max_size
    return (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))


def locate_name_line(image: Image.Image) -> KhmerIDNameLocation | None:
    """Run Tesseract image_to_data on the full ID and isolate the Khmer name line.

    Strategy:
      1. Run Tesseract psm=6 with khm to get word-level bboxes for the whole image.
      2. Group words by Tesseract's line_num.
      3. Pick the topmost line that contains "នាម".
      4. Within that line, find the word ending in a colon — that's the label terminator.
      5. The name bbox is the region from (label_end_x, line_top) to (line_right, line_bottom).
    """
    data = pytesseract.image_to_data(
        image,
        lang="khm",
        config="--oem 1 --psm 6",
        output_type=pytesseract.Output.DICT,
    )

    lines: dict[tuple[int, int, int], list[int]] = {}
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(i)

    candidate = None
    for idxs in lines.values():
        line_text = " ".join(data["text"][i] for i in idxs)
        if NAME_LABEL_KEYWORD not in line_text:
            continue
        top_y = min(data["top"][i] for i in idxs)
        if candidate is None or top_y < candidate["top_y"]:
            candidate = {"top_y": top_y, "idxs": idxs, "text": line_text}

    if candidate is None:
        return None

    idxs = sorted(candidate["idxs"], key=lambda i: data["left"][i])

    label_end_x: int | None = None
    label_word_height: int | None = None
    for i in idxs:
        text = data["text"][i]
        colon_pos = next((j for j, c in enumerate(text) if c in COLON_CHARS), None)
        if colon_pos is None:
            continue
        word_left = data["left"][i]
        word_width = data["width"][i]
        label_word_height = data["height"][i]
        # If the word ends at the colon ("...នាម:"), use the bbox right edge.
        # If the word continues past the colon ("...នាម:ជាន" — Tesseract sometimes
        # merges label + first name into one bbox), proportionally estimate the
        # colon's x position inside the bbox so we don't crop past the first name.
        if colon_pos == len(text) - 1:
            label_end_x = word_left + word_width
        else:
            label_end_x = word_left + int(word_width * (colon_pos + 1) / len(text))
        break

    lefts = [data["left"][i] for i in idxs]
    tops = [data["top"][i] for i in idxs]
    rights = [data["left"][i] + data["width"][i] for i in idxs]
    bottoms = [data["top"][i] + data["height"][i] for i in idxs]

    # Clamp bottom: name-side bboxes can include long subscripts (្រ, ្ត) that overlap
    # with the English transliteration line below. Cap line height at ~1.5× the label height.
    line_top = min(tops)
    if label_word_height:
        line_bottom = min(max(bottoms), line_top + int(label_word_height * 1.5))
    else:
        line_bottom = max(bottoms)

    line_bbox = (min(lefts), line_top, max(rights), line_bottom)

    if label_end_x is not None and label_end_x < line_bbox[2]:
        name_bbox = (label_end_x, line_top, line_bbox[2], line_bottom)
    else:
        # Fallback: take the right 60% of the line (rough heuristic when no colon found).
        x0, y0, x1, y1 = line_bbox
        name_bbox = (x0 + int((x1 - x0) * 0.4), y0, x1, y1)

    return KhmerIDNameLocation(
        line_bbox=line_bbox,
        name_bbox=name_bbox,
        label_text=candidate["text"],
    )
