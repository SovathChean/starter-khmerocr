"""Helpers for locating fields on a Cambodian National ID card."""
from dataclasses import dataclass

import pytesseract
from PIL import Image

# The Khmer name label on Cambodian IDs is "គោត្តនាម និង នាម:" — always contains "នាម".
NAME_LABEL_KEYWORD = "នាម"
COLON_CHARS = (":", "៖", "ៈ")

# Tesseract khm misreads `ម↔ទ`, `ន↔ខ↔គ` more often when the Khmer line is below
# ~40 px tall. Pre-upscale small images so the keyword "នាម" survives recognition.
UPSCALE_TARGET_WIDTH = 1800


def _filter_line_to_name_neighbours(
    idxs: list[int],
    data: dict,
    *,
    min_width_ratio: float = 0.1,
    min_height_ratio: float = 0.3,
    max_gap_ratio: float = 0.3,
    soft_gap_ratio: float = 0.18,
    soft_conf: int = 30,
) -> list[int]:
    """Drop noise words from the name line.

    Tesseract often groups the photo edge, the ID number "(01)", and the watermark
    into the same logical line as the Khmer name. Using the colon-word as an anchor,
    walk outward and stop on the first 'noise' word. All thresholds are RELATIVE
    to the label (colon-containing) word's width / height — that scales with image
    resolution so the same filter works on a 286-px screenshot and a 3508-px scan.

    A word is noise when:
      • gap > max_gap_ratio × label_width (definitely far away), OR
      • bbox is tiny — width < min_width_ratio × label_width, or
        height < min_height_ratio × label_height, OR
      • gap > soft_gap_ratio × label_width AND Tesseract conf < soft_conf
        (medium-far AND uncertain — usually digits / punctuation from the ID number).
    Real Khmer name words can have conf=0 yet still be valid, so we never drop
    purely on confidence — there must also be a spatial signal.
    """
    colon_idx = next(
        (i for i in idxs if any(c in data["text"][i] for c in COLON_CHARS)),
        None,
    )
    if colon_idx is None:
        return idxs

    pos = idxs.index(colon_idx)
    kept = [colon_idx]
    label_width = data["width"][colon_idx]
    label_height = data["height"][colon_idx]
    min_word_width = max(10, int(label_width * min_width_ratio))
    min_word_height = max(10, int(label_height * min_height_ratio))
    max_gap_px = max(40, int(label_width * max_gap_ratio))
    soft_gap_px = max(25, int(label_width * soft_gap_ratio))

    def _conf(i: int) -> int:
        c = data["conf"][i]
        try:
            return int(c)
        except (TypeError, ValueError):
            return -1

    def _is_noise(gap: int, i: int) -> bool:
        if gap > max_gap_px:
            return True
        if data["width"][i] < min_word_width or data["height"][i] < min_word_height:
            return True
        if gap > soft_gap_px and _conf(i) < soft_conf:
            return True
        return False

    # Walk LEFT from the colon word (drop noise from photo / left margin).
    last_left = data["left"][colon_idx]
    for i in reversed(idxs[:pos]):
        right = data["left"][i] + data["width"][i]
        gap = last_left - right
        if _is_noise(gap, i):
            break
        kept.append(i)
        last_left = data["left"][i]

    # Walk RIGHT from the colon word (drop noise from ID number / "(01)" stamp).
    last_right = data["left"][colon_idx] + data["width"][colon_idx]
    for i in idxs[pos + 1:]:
        left = data["left"][i]
        gap = left - last_right
        if _is_noise(gap, i):
            break
        kept.append(i)
        last_right = data["left"][i] + data["width"][i]

    return sorted(kept, key=lambda i: data["left"][i])


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


def _upscaled_for_ocr(image: Image.Image) -> tuple[Image.Image, float]:
    """Return (image_to_pass_to_OCR, factor_to_divide_bboxes_by_back_to_original)."""
    if image.width >= UPSCALE_TARGET_WIDTH:
        return image, 1.0
    factor = UPSCALE_TARGET_WIDTH / image.width
    new_size = (int(image.width * factor), int(image.height * factor))
    return image.resize(new_size, Image.LANCZOS), factor


def _scale_bbox(bbox: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    return (int(bbox[0] * scale), int(bbox[1] * scale), int(bbox[2] * scale), int(bbox[3] * scale))


def _run_image_to_data(image: Image.Image) -> dict:
    return pytesseract.image_to_data(
        image,
        lang="khm",
        config="--oem 1 --psm 6",
        output_type=pytesseract.Output.DICT,
    )


def _group_words_by_line(data: dict) -> dict[tuple[int, int, int], list[int]]:
    lines: dict[tuple[int, int, int], list[int]] = {}
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(i)
    return lines


def _topmost_keyword_line(lines: dict, data: dict) -> dict | None:
    candidate = None
    for idxs in lines.values():
        line_text = " ".join(data["text"][i] for i in idxs)
        if NAME_LABEL_KEYWORD not in line_text:
            continue
        top_y = min(data["top"][i] for i in idxs)
        if candidate is None or top_y < candidate["top_y"]:
            candidate = {"top_y": top_y, "idxs": idxs, "text": line_text}
    return candidate


def _topmost_colon_line(lines: dict, data: dict, image_height: int) -> dict | None:
    """Structural fallback: pick the topmost line in the upper 50% of the image
    that contains a colon and at least one substantial word (≥ 50 px wide).
    On Cambodian IDs the name line is structurally always the first labeled line."""
    upper_half = image_height * 0.5
    candidate = None
    for idxs in lines.values():
        line_text = " ".join(data["text"][i] for i in idxs)
        if not any(c in line_text for c in COLON_CHARS):
            continue
        top_y = min(data["top"][i] for i in idxs)
        if top_y >= upper_half:
            continue
        if not any(data["width"][i] >= 50 for i in idxs):
            continue
        if candidate is None or top_y < candidate["top_y"]:
            candidate = {"top_y": top_y, "idxs": idxs, "text": line_text}
    return candidate


def locate_name_line(image: Image.Image) -> KhmerIDNameLocation | None:
    """Run Tesseract image_to_data on the full ID and isolate the Khmer name line.

    Strategy (lazy upscaling):
      1. Run image_to_data on the original image and look for a line containing "នាម".
      2. If not found AND the image is small, upscale and try again (rescues
         keyword from misreads on borderline-resolution screenshots).
      3. If still not found, fall back to the topmost line in the upper half
         that contains a colon (Cambodian IDs always have the name line first).
      4. Translate bboxes back to the original image's coordinate space.
    """
    # Pass 1: original image, keyword search.
    data = _run_image_to_data(image)
    lines = _group_words_by_line(data)
    candidate = _topmost_keyword_line(lines, data)
    factor = 1.0
    ocr_image = image

    # Pass 2: if the keyword wasn't found AND the image is small, retry upscaled.
    if candidate is None and image.width < UPSCALE_TARGET_WIDTH:
        ocr_image, factor = _upscaled_for_ocr(image)
        data = _run_image_to_data(ocr_image)
        lines = _group_words_by_line(data)
        candidate = _topmost_keyword_line(lines, data)

    # Pass 3: fallback heuristic (topmost colon line in upper half of the image).
    if candidate is None:
        candidate = _topmost_colon_line(lines, data, ocr_image.height)

    if candidate is None:
        return None

    idxs = sorted(candidate["idxs"], key=lambda i: data["left"][i])
    idxs = _filter_line_to_name_neighbours(idxs, data)

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

    # Translate bboxes back to the ORIGINAL image's coordinate space — the caller
    # crops the original (the upscaled image is only used for OCR detection).
    if factor != 1.0:
        scale = 1.0 / factor
        line_bbox = _scale_bbox(line_bbox, scale)
        name_bbox = _scale_bbox(name_bbox, scale)

    return KhmerIDNameLocation(
        line_bbox=line_bbox,
        name_bbox=name_bbox,
        label_text=candidate["text"],
    )
