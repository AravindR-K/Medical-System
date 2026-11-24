import sys
import pymupdf  # PyMuPDF
import easyocr
import base64

from system_one_models import slave_decision
from typing import Callable, Optional

#REQUIREMENTS:


#Extract all text and image content from the pdf

from __future__ import annotations

import json
import re
import uuid
import logging
from pathlib import Path
from collections import Counter
from typing import Any

import fitz  # PyMuPDF
import numpy as np
import pandas as pd

from paddleocr import PPStructureV3


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR_NAME = "pdf_extracted"
IMAGE_DIR_NAME = "images"

# Semantic types used in the output
VALID_TYPES = {
    "document_title",
    "section_title",
    "paragraph_title",
    "paragraph",
    "list_item",
    "table_title",
    "table",
    "image",
    "figure_caption",
    "header",
    "footer",
    "page_number",
    "equation",
    "footnote",
    "unknown",
}

# Labels commonly returned by document layout models.
# Paddle versions/model configurations may differ, so we normalize
# their names rather than relying on one exact label vocabulary.
LAYOUT_LABEL_MAP = {
    "title": "document_title",
    "doc_title": "document_title",
    "document_title": "document_title",

    "paragraph_title": "paragraph_title",
    "section": "section_title",
    "section_title": "section_title",
    "heading": "section_title",

    "text": "paragraph",
    "paragraph": "paragraph",
    "plain_text": "paragraph",

    "list": "list_item",
    "list_item": "list_item",

    "table": "table",
    "figure": "image",
    "image": "image",
    "chart": "image",

    "caption": "figure_caption",
    "figure_caption": "figure_caption",

    "header": "header",
    "footer": "footer",
    "footnote": "footnote",

    "formula": "equation",
    "equation": "equation",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def _safe_text(value: Any) -> str:
    """Convert arbitrary OCR values into normalized strings."""
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        return " ".join(_safe_text(v) for v in value).strip()

    return re.sub(r"\s+", " ", str(value)).strip()

def _normalize_label(label: Any) -> str:
    """Normalize model labels to a common vocabulary."""
    label = _safe_text(label).lower()
    label = label.replace("-", "_").replace(" ", "_")

    return LAYOUT_LABEL_MAP.get(label, "unknown")

def _normalize_bbox(bbox: Any) -> list[float] | None:
    """
    Normalize a bounding box into [x1, y1, x2, y2].

    Supports:
        [x1, y1, x2, y2]
        [[x1, y1], [x2, y2], ...] polygons
    """
    if bbox is None:
        return None

    try:
        arr = np.asarray(bbox, dtype=float)

        if arr.ndim == 1 and arr.size >= 4:
            x1, y1, x2, y2 = arr[:4]
            return [
                float(min(x1, x2)),
                float(min(y1, y2)),
                float(max(x1, x2)),
                float(max(y1, y2)),
            ]

        if arr.ndim == 2 and arr.shape[1] >= 2:
            xs = arr[:, 0]
            ys = arr[:, 1]

            return [
                float(xs.min()),
                float(ys.min()),
                float(xs.max()),
                float(ys.max()),
            ]

    except (ValueError, TypeError):
        return None

    return None

def _bbox_area(bbox: list[float] | None) -> float:
    if bbox is None:
        return 0.0

    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (
        (bbox[0] + bbox[2]) / 2,
        (bbox[1] + bbox[3]) / 2,
    )

def _vertical_overlap(a: list[float], b: list[float]) -> float:
    overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    denom = max(1.0, min(a[3] - a[1], b[3] - b[1]))
    return overlap / denom

def _horizontal_overlap(a: list[float], b: list[float]) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    denom = max(1.0, min(a[2] - a[0], b[2] - b[0]))
    return overlap / denom

def _sort_reading_order(elements: list[dict]) -> list[dict]:
    """
    Approximate reading order.

    Sorts by vertical position, then horizontal position.
    A later column-aware enhancement can replace this.
    """
    return sorted(
        elements,
        key=lambda e: (
            e.get("bbox", [0, 0, 0, 0])[1],
            e.get("bbox", [0, 0, 0, 0])[0],
        ),
    )


# ============================================================
# PDF RENDERING
# ============================================================

def _render_pdf_pages(
        pdf_path: str,
        dpi: int = 180,
) -> tuple[fitz.Document, list[np.ndarray]]:
    """Open a PDF and render its pages to RGB NumPy arrays."""
    doc = fitz.open(pdf_path)

    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)

    rendered_pages = []

    for page in doc:
        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        image = np.frombuffer(
            pix.samples,
            dtype=np.uint8,
        ).reshape(
            pix.height,
            pix.width,
            pix.n,
        )

        if pix.n == 4:
            image = image[:, :, :3]

        rendered_pages.append(image)

    return doc, rendered_pages


# ============================================================
# OCR ENGINE
# ============================================================

def _initialize_ocr():
    """
    Initialize PP-Structure V3.

    Depending on PaddleOCR/PaddlePaddle versions, model download or
    runtime initialization may occur on first use.
    """
    return PPStructureV3()

def _run_ocr(ocr_engine, page_image: np.ndarray) -> dict:
    """Run PP-Structure analysis on one page."""
    results = ocr_engine.predict(input=page_image)

    if not results:
        return {}

    first = results[0]

    # PP-StructureV3 results commonly expose a result object that
    # can be converted to a dictionary.
    if isinstance(first, dict):
        return first

    if hasattr(first, "json"):
        try:
            value = first.json

            if callable(value):
                value = value()

            if isinstance(value, dict):
                return value
        except Exception:
            pass

    if hasattr(first, "to_dict"):
        try:
            value = first.to_dict()

            if isinstance(value, dict):
                return value
        except Exception:
            pass

    # Some result objects provide their own serialized representation.
    if hasattr(first, "res"):
        value = first.res

        if isinstance(value, dict):
            return value

    return {"raw_result": first}


# ============================================================
# RESULT NORMALIZATION
# ============================================================

def _recursive_find(data: Any, candidate_keys: set[str]) -> list[Any]:
    """
    Recursively find values associated with candidate keys.

    This helps handle output differences between OCR versions.
    """
    found = []

    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in candidate_keys:
                found.append(value)

            found.extend(
                _recursive_find(value, candidate_keys)
            )

    elif isinstance(data, list):
        for item in data:
            found.extend(
                _recursive_find(item, candidate_keys)
            )

    return found

def _extract_layout_regions(raw_result: dict) -> list[dict]:
    """
    Extract layout regions from PP-Structure output.

    Supports several common output forms:
        layout_det_res
        layout_res
        layout
        parsing_res_list
    """
    regions = []

    candidate_keys = {
        "layout_det_res",
        "layout_res",
        "layout",
        "parsing_res_list",
    }

    candidates = _recursive_find(raw_result, candidate_keys)

    for candidate in candidates:
        if not isinstance(candidate, list):
            continue

        for item in candidate:
            if not isinstance(item, dict):
                continue

            bbox = _normalize_bbox(
                item.get("bbox")
                or item.get("coordinate")
                or item.get("polygon")
            )

            if bbox is None:
                continue

            label = (
                    item.get("label")
                    or item.get("type")
                    or item.get("category")
                    or item.get("block_type")
            )

            score = item.get("score")

            regions.append({
                "type": _normalize_label(label),
                "raw_label": _safe_text(label),
                "bbox": bbox,
                "confidence": (
                    float(score)
                    if score is not None
                    else None
                ),
                "text": _safe_text(
                    item.get("text")
                    or item.get("content")
                ),
                "raw": item,
            })

    # Remove exact duplicates that can arise from recursive traversal.
    unique = []
    seen = set()

    for region in regions:
        key = (
            region["type"],
            tuple(round(x, 1) for x in region["bbox"]),
        )

        if key not in seen:
            seen.add(key)
            unique.append(region)

    return unique

def _extract_ocr_lines(raw_result: dict) -> list[dict]:
    """
    Extract OCR text and polygons from common PP-Structure/PaddleOCR
    output structures.
    """
    lines = []

    candidate_keys = {
        "overall_ocr_res",
        "ocr_res",
        "text_res",
        "rec_res",
    }

    candidates = _recursive_find(raw_result, candidate_keys)

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        texts = (
                candidate.get("rec_texts")
                or candidate.get("texts")
                or candidate.get("text")
        )

        scores = (
                candidate.get("rec_scores")
                or candidate.get("scores")
        )

        polys = (
                candidate.get("rec_polys")
                or candidate.get("rec_boxes")
                or candidate.get("dt_polys")
                or candidate.get("boxes")
        )

        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(texts, list):
            continue

        if polys is None:
            polys = [None] * len(texts)

        if not isinstance(polys, (list, tuple)):
            polys = [polys] * len(texts)

        if scores is None:
            scores = [None] * len(texts)

        if not isinstance(scores, (list, tuple)):
            scores = [scores] * len(texts)

        for i, text in enumerate(texts):
            text = _safe_text(text)

            if not text:
                continue

            bbox = (
                _normalize_bbox(polys[i])
                if i < len(polys)
                else None
            )

            confidence = (
                scores[i]
                if i < len(scores)
                else None
            )

            lines.append({
                "text": text,
                "bbox": bbox,
                "confidence": (
                    float(confidence)
                    if confidence is not None
                    else None
                ),
            })

    # De-duplicate repeated OCR records.
    unique = []
    seen = set()

    for line in lines:
        key = (
            line["text"],
            tuple(
                round(x, 1)
                for x in line["bbox"]
            ) if line["bbox"] else None,
        )

        if key not in seen:
            seen.add(key)
            unique.append(line)

    return unique


# ============================================================
# COORDINATE NORMALIZATION
# ============================================================

def _convert_bbox_to_page_pixels(
        bbox: list[float],
        image_width: int,
        image_height: int,
) -> list[int]:
    """
    Normalize model coordinates to the rendered page's pixel coordinates.

    If coordinates already appear to be pixel coordinates, they are
    retained. If they appear to be normalized or page-point coordinates,
    scaling is estimated.
    """
    x1, y1, x2, y2 = bbox

    max_x = max(x1, x2)
    max_y = max(y1, y2)

    # Coordinates in a small range may be normalized (0..1).
    if max_x <= 1.5 and max_y <= 1.5:
        x1 *= image_width
        x2 *= image_width
        y1 *= image_height
        y2 *= image_height

    # If the model coordinates are not in pixel space, infer scale.
    elif max_x > image_width * 1.5 or max_y > image_height * 1.5:
        scale_x = image_width / max(max_x, 1)
        scale_y = image_height / max(max_y, 1)

        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

    return [
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(image_width, int(round(x2))),
        min(image_height, int(round(y2))),
    ]

def _crop_bbox(
        image: np.ndarray,
        bbox: list[float],
) -> np.ndarray | None:
    h, w = image.shape[:2]

    x1, y1, x2, y2 = _convert_bbox_to_page_pixels(
        bbox,
        w,
        h,
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2].copy()


# ============================================================
# TEXT CLASSIFICATION
# ============================================================

def _is_page_number(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:page\s*)?[\divxlcdm]+(?:\s*/\s*\d+)?",
            text.strip(),
            flags=re.IGNORECASE,
        )
    )

def _looks_like_list_item(text: str) -> bool:
    return bool(
        re.match(
            r"^\s*(?:[-*•▪]|\d+[.)]|[a-zA-Z][.)])\s+",
            text,
        )
    )
def _looks_like_heading(text: str) -> bool:
    text = text.strip()

    if not text:
        return False

    if len(text) > 160:
        return False

    # Short text with no sentence-ending punctuation is more likely
    # to be a heading than a paragraph.
    if text.endswith((".", "!", "?", ";")):
        return False

    words = text.split()

    return len(words) <= 16
def _looks_like_caption(text: str) -> bool:
    return bool(
        re.match(
            r"^(figure|fig\.|image|diagram|chart|photo|table)\s*[\d.: -]*",
            text.strip(),
            flags=re.IGNORECASE,
        )
    )
def _classify_text_region(
        text: str,
        bbox: list[float],
        page_width: int,
        page_height: int,
        layout_type: str = "unknown",
) -> str:
    """
    Classify text regions using text and geometric features.
    """
    text = text.strip()

    if not text:
        return "unknown"

    if _is_page_number(text):
        return "page_number"

    if _looks_like_list_item(text):
        return "list_item"

    if _looks_like_caption(text):
        return "figure_caption"

    x1, y1, x2, y2 = bbox
    region_height = y2 - y1

    top_ratio = y1 / max(page_height, 1)
    bottom_ratio = y2 / max(page_height, 1)

    normalized_layout_type = layout_type

    if normalized_layout_type in {
        "document_title",
        "section_title",
        "paragraph_title",
        "paragraph",
        "header",
        "footer",
        "footnote",
        "equation",
        "list_item",
    }:
        return normalized_layout_type

    # Top-of-page text can be a header.
    if top_ratio < 0.08 and len(text) < 120:
        return "header"

    # Bottom-of-page text is a likely footer.
    if bottom_ratio > 0.92 and len(text) < 160:
        return "footer"

    # A very small text region near the bottom can be a footnote.
    if bottom_ratio > 0.75 and len(text) < 220:
        return "footnote"

    # Text near the top, spanning a large width, is a title candidate.
    if (
            top_ratio < 0.25
            and (x2 - x1) > page_width * 0.25
            and len(text) < 140
            and _looks_like_heading(text)
    ):
        return "document_title"

    # Short standalone text is a heading candidate.
    if _looks_like_heading(text):
        return "paragraph_title"

    return "paragraph"


# ============================================================
# TEXT-LAYOUT ASSOCIATION
# ============================================================

def _associate_ocr_with_layout(
        ocr_lines: list[dict],
        layout_regions: list[dict],
) -> list[dict]:
    """
    Associate OCR lines with layout regions based on bounding-box overlap.

    OCR text and layout predictions can sometimes have separate
    coordinates or slightly different boxes.
    """
    elements = []

    for line in ocr_lines:
        bbox = line["bbox"]

        if bbox is None:
            elements.append({
                **line,
                "type": "unknown",
            })
            continue

        best_region = None
        best_score = 0.0

        for region in layout_regions:
            score = _vertical_overlap(bbox, region["bbox"])
            score *= _horizontal_overlap(bbox, region["bbox"])

            if score > best_score:
                best_score = score
                best_region = region

        elements.append({
            **line,
            "type": (
                best_region["type"]
                if best_region and best_score > 0.15
                else "unknown"
            ),
            "layout_label": (
                best_region["raw_label"]
                if best_region and best_score > 0.15
                else None
            ),
        })

    return elements


# ============================================================
# HEADER / FOOTER RECOGNITION ACROSS PAGES
# ============================================================

def _normalize_repeated_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\d+", "#", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _find_repeated_header_footer(
        pages: list[list[dict]],
        page_heights: list[int],
) -> dict[str, set[str]]:
    """
    Detect text repeated near page tops or bottoms.

    This is useful for documents that lack explicit header/footer
    layout labels.
    """
    top_texts = []
    bottom_texts = []

    for page_idx, elements in enumerate(pages):
        page_height = page_heights[page_idx]

        for element in elements:
            bbox = element.get("bbox")

            if bbox is None:
                continue

            text = element.get("text", "").strip()

            if not text or len(text) > 180:
                continue

            normalized = _normalize_repeated_text(text)

            if bbox[1] < page_height * 0.10:
                top_texts.append(normalized)

            if bbox[3] > page_height * 0.90:
                bottom_texts.append(normalized)

    top_counter = Counter(top_texts)
    bottom_counter = Counter(bottom_texts)

    threshold = max(2, int(len(pages) * 0.5))

    return {
        "header": {
            text
            for text, count in top_counter.items()
            if count >= threshold
        },
        "footer": {
            text
            for text, count in bottom_counter.items()
            if count >= threshold
        },
    }


def _apply_repeated_header_footer(
        pages: list[list[dict]],
        repeated: dict[str, set[str]],
) -> None:
    for elements in pages:
        for element in elements:
            normalized = _normalize_repeated_text(
                element.get("text", "")
            )

            if normalized in repeated["header"]:
                element["type"] = "header"

            elif normalized in repeated["footer"]:
                element["type"] = "footer"


# ============================================================
# TITLE CLASSIFICATION
# ============================================================

def _classify_document_titles(
        pages: list[list[dict]],
        page_widths: list[int],
        page_heights: list[int],
) -> str | None:
    """
    Promote the best candidate on the first page to document_title.
    """
    if not pages:
        return None

    candidates = []

    for element in pages[0]:
        text = element.get("text", "").strip()
        bbox = element.get("bbox")

        if not text or bbox is None:
            continue

        if element.get("type") in {"header", "footer", "page_number"}:
            continue

        y_position = bbox[1] / max(page_heights[0], 1)
        width_ratio = (bbox[2] - bbox[0]) / max(page_widths[0], 1)

        if len(text) > 200:
            continue

        score = 0.0

        # Prefer text toward the top of the first page.
        score += max(0.0, 1.0 - y_position) * 2.0

        # Prefer text occupying substantial horizontal space.
        score += width_ratio

        # Short standalone headings are title candidates.
        if _looks_like_heading(text):
            score += 1.5

        if element.get("type") in {
            "document_title",
            "section_title",
            "paragraph_title",
        }:
            score += 2.0

        candidates.append((score, element))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)

    best = candidates[0][1]
    best["type"] = "document_title"

    return best["text"]


# ============================================================
# TABLE EXTRACTION
# ============================================================

def _extract_table_objects(
        raw_result: dict,
        page_number: int,
        page_elements: list[dict],
) -> list[dict]:
    """
    Extract table results from PP-Structure output.

    This implementation attempts to parse HTML tables from PP-Structure
    output. Table output formats vary across PaddleOCR versions.
    """
    table_candidates = _recursive_find(
        raw_result,
        {
            "table_res",
            "tables",
            "table_html",
            "html",
            "parsing_res_list",
        },
    )

    extracted_tables = []
    table_index = 1

    for candidate in table_candidates:
        html_strings = []

        if isinstance(candidate, str):
            if "<table" in candidate.lower():
                html_strings.append(candidate)

        elif isinstance(candidate, dict):
            for key in ("html", "table_html", "html_text"):
                value = candidate.get(key)

                if isinstance(value, str) and "<table" in value.lower():
                    html_strings.append(value)

        elif isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, str) and "<table" in item.lower():
                    html_strings.append(item)

                elif isinstance(item, dict):
                    for key in ("html", "table_html", "html_text"):
                        value = item.get(key)

                        if (
                                isinstance(value, str)
                                and "<table" in value.lower()
                        ):
                            html_strings.append(value)

        for html in html_strings:
            try:
                dataframes = pd.read_html(html)

                if not dataframes:
                    continue

                df = dataframes[0]
                df = df.dropna(how="all").reset_index(drop=True)

                if df.empty:
                    continue

                title = _find_table_title(
                    df,
                    page_elements,
                )

                extracted_tables.append({
                    "table_id": f"page_{page_number}_table_{table_index}",
                    "page_number": page_number,
                    "title": title,
                    "columns": [
                        str(col)
                        for col in df.columns
                    ],
                    "data": df,
                })

                table_index += 1

            except Exception as exc:
                logger.warning(
                    "Failed to parse table HTML: %s",
                    exc,
                )

    return extracted_tables

def _find_table_title(
        dataframe: pd.DataFrame,
        page_elements: list[dict],
) -> str | None:
    """
    Fallback title association.

    PP-Structure may provide a table bbox. If that bbox is available,
    use the nearest short text element above the table. When the bbox
    is unavailable, this falls back to a conservative heuristic.
    """
    for element in page_elements:
        if element.get("type") == "table_title":
            return element.get("text")

    # Conservative fallback: do not assign an arbitrary paragraph
    # as the title of a table.
    return None


# ============================================================
# IMAGE / FIGURE EXTRACTION
# ============================================================

def _extract_images_from_page(
        page_image: np.ndarray,
        layout_regions: list[dict],
        output_dir: Path,
        page_number: int,
) -> list[dict]:
    """
    Crop detected image/figure/chart regions from the rendered PDF.

    Vector figures and image-backed content are both captured because
    crops are taken from the rendered page.
    """
    extracted_images = []

    image_index = 1

    for region in layout_regions:
        if region["type"] != "image":
            continue

        bbox = region["bbox"]

        crop = _crop_bbox(page_image, bbox)

        if crop is None or crop.size == 0:
            continue

        image_name = (
            f"page_{page_number}_image_{image_index}.png"
        )

        image_path = output_dir / image_name

        import cv2

        success = cv2.imwrite(
            str(image_path),
            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
        )

        if not success:
            logger.warning(
                "Failed to save crop: %s",
                image_path,
            )
            continue

        extracted_images.append({
            "type": "image",
            "image_id": f"page_{page_number}_image_{image_index}",
            "page_number": page_number,
            "bbox": bbox,
            "path": str(image_path),
            "confidence": region.get("confidence"),
        })

        image_index += 1

    return extracted_images


# ============================================================
# ELEMENT RECONSTRUCTION
# ============================================================

def _build_page_elements(
        raw_result: dict,
        page_image: np.ndarray,
        page_number: int,
) -> tuple[list[dict], list[dict]]:
    height, width = page_image.shape[:2]

    layout_regions = _extract_layout_regions(raw_result)
    ocr_lines = _extract_ocr_lines(raw_result)

    elements = _associate_ocr_with_layout(
        ocr_lines,
        layout_regions,
    )

    reconstructed = []

    for element in elements:
        bbox = element.get("bbox")

        if bbox is None:
            reconstructed.append({
                "type": "unknown",
                "text": element["text"],
                "page_number": page_number,
                "bbox": None,
                "confidence": element.get("confidence"),
            })
            continue

        # Convert text region to page pixel coordinates.
        bbox_pixels = _convert_bbox_to_page_pixels(
            bbox,
            width,
            height,
        )

        classification = _classify_text_region(
            text=element["text"],
            bbox=bbox_pixels,
            page_width=width,
            page_height=height,
            layout_type=element.get("type", "unknown"),
        )

        reconstructed.append({
            "type": classification,
            "text": element["text"],
            "page_number": page_number,
            "bbox": bbox_pixels,
            "confidence": element.get("confidence"),
        })

    # Add detected tables as region elements, even if their text
    # is not reconstructed as normal OCR text.
    for region in layout_regions:
        if region["type"] == "table":
            bbox_pixels = _convert_bbox_to_page_pixels(
                region["bbox"],
                width,
                height,
            )

            reconstructed.append({
                "type": "table",
                "text": "",
                "page_number": page_number,
                "bbox": bbox_pixels,
                "confidence": region.get("confidence"),
            })

    # Remove duplicate text elements.
    unique = []
    seen = set()

    for element in reconstructed:
        key = (
            element.get("type"),
            element.get("text"),
            tuple(element["bbox"]) if element.get("bbox") else None,
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(element)

    unique = _sort_reading_order(unique)

    return unique, layout_regions


# ============================================================
# JSON SERIALIZATION
# ============================================================

def _serialize_dataframe(df: pd.DataFrame) -> dict:
    """Convert a DataFrame into a JSON-friendly representation."""
    clean = df.copy()

    # Avoid JSON NaN, Infinity and similar non-standard values.
    clean = clean.astype(object).where(pd.notna(clean), None)

    return {
        "columns": [
            str(column)
            for column in clean.columns
        ],
        "rows": clean.values.tolist(),
    }


def document_to_json(
        document: dict,
        output_path: str | Path | None = None,
) -> str:
    """
    Serialize the document structure.

    DataFrames are converted into {columns, rows}.
    """
    def convert(value):
        if isinstance(value, pd.DataFrame):
            return _serialize_dataframe(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, Path):
            return str(value)

        raise TypeError(
            f"Unsupported JSON type: {type(value)}"
        )

    json_string = json.dumps(
        document,
        default=convert,
        ensure_ascii=False,
        indent=2,
    )

    if output_path is not None:
        Path(output_path).write_text(
            json_string,
            encoding="utf-8",
        )

    return json_string


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def extract_pdf_content(pdf_path: str) -> dict:
    """
    Extract and classify content from a PDF.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.

    Returns
    -------
    dict
        {
            "document_title": str | None,
            "source_pdf": str,
            "pages": list[dict],
            "tables": list[dict],
            "image_directory": str,
            "json_path": str
        }

    Tables contain actual Pandas DataFrames.

    Images/figures are saved as separate files.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            "Input file must be a PDF."
        )

    output_dir = (
            pdf_path.parent
            / f"{pdf_path.stem}_{OUTPUT_DIR_NAME}"
    )

    image_dir = output_dir / IMAGE_DIR_NAME

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Rendering PDF: %s", pdf_path)

    pdf_document, rendered_pages = _render_pdf_pages(
        str(pdf_path),
    )

    ocr_engine = _initialize_ocr()

    pages = []
    page_widths = []
    page_heights = []
    all_tables = []

    try:
        for page_index, page_image in enumerate(rendered_pages):
            page_number = page_index + 1

            height, width = page_image.shape[:2]

            page_widths.append(width)
            page_heights.append(height)

            logger.info(
                "Processing page %s/%s",
                page_number,
                len(rendered_pages),
            )

            raw_result = _run_ocr(
                ocr_engine,
                page_image,
            )

            elements, layout_regions = _build_page_elements(
                raw_result=raw_result,
                page_image=page_image,
                page_number=page_number,
            )

            # Crop detected figures/images.
            images = _extract_images_from_page(
                page_image=page_image,
                layout_regions=layout_regions,
                output_dir=image_dir,
                page_number=page_number,
            )

            # Add images to page elements.
            elements.extend(images)

            # Table extraction.
            page_tables = _extract_table_objects(
                raw_result=raw_result,
                page_number=page_number,
                page_elements=elements,
            )

            all_tables.extend(page_tables)

            pages.append({
                "page_number": page_number,
                "width": width,
                "height": height,
                "elements": _sort_reading_order(elements),
            })

    finally:
        pdf_document.close()

    # --------------------------------------------------------
    # Detect repeated headers and footers across pages.
    # --------------------------------------------------------

    repeated = _find_repeated_header_footer(
        [page["elements"] for page in pages],
        page_heights,
    )

    _apply_repeated_header_footer(
        [page["elements"] for page in pages],
        repeated,
    )

    # --------------------------------------------------------
    # Identify document title.
    # --------------------------------------------------------

    document_title = _classify_document_titles(
        [page["elements"] for page in pages],
        page_widths,
        page_heights,
    )

    # --------------------------------------------------------
    # Assemble result.
    # --------------------------------------------------------

    document = {
        "document_title": document_title,
        "source_pdf": str(pdf_path),
        "pages": pages,
        "tables": all_tables,
        "image_directory": str(image_dir),
    }

    json_path = output_dir / "document.json"

    document_to_json(
        document,
        output_path=json_path,
    )

    document["json_path"] = str(json_path)

    return document

'''
Returns
-------
dict
{
    "document_title": str | None,
    "source_pdf": str,
    "pages": list[dict],
    "tables": list[dict],
    "image_directory": str,
    "json_path": str
}
'''

# ============================================================
# OPTIONAL DISPLAY HELPERS
# ============================================================

def print_document_json(document: dict) -> None:
    """
    Print the document as JSON, with DataFrames converted to rows.
    """
    print(document_to_json(document))


def get_all_tables(document: dict) -> list[pd.DataFrame]:
    """Return all extracted DataFrames."""
    return [
        table["data"]
        for table in document.get("tables", [])
        if isinstance(table.get("data"), pd.DataFrame)
    ]


def get_page_elements(
        document: dict,
        page_number: int,
        element_type: str | None = None,
) -> list[dict]:
    """Retrieve elements from a specific page."""
    pages = document.get("pages", [])

    page = next(
        (
            p for p in pages
            if p["page_number"] == page_number
        ),
        None,
    )

    if page is None:
        return []

    elements = page["elements"]

    if element_type is None:
        return elements

    return [
        element
        for element in elements
        if element.get("type") == element_type
    ]






