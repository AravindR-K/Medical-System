import sys
import pymupdf  # PyMuPDF
import easyocr
import base64

from system_one_models import slave_decision
from typing import Callable, Optional

#REQUIREMENTS:

#return the tables as dataframe like objects


#Extract all text and image content from the pdf
def extract_pdf_content(pdf_path: str) -> dict:
    reader = easyocr.Reader(["en"], gpu=False)
    document = pymupdf.open(pdf_path)

    extracted_document = {
        "source": pdf_path,
        "pages": []
    }

    try:
        for page_number, page in enumerate(document, start=1):

            # Render page to image for OCR
            scale = 2
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale),
                alpha=False
            )

            image_bytes = pix.tobytes("png")

            # OCR
            results = reader.readtext(image_bytes, detail=1)

            text_regions = []

            for bbox, text, confidence in results:

                # Convert OCR bbox to simple rectangular coordinates
                xs = [point[0] for point in bbox]
                ys = [point[1] for point in bbox]

                x_min = min(xs)
                y_min = min(ys)
                x_max = max(xs)
                y_max = max(ys)

                text_regions.append({
                    "text": text,
                    "confidence": float(confidence),

                    "bbox": {
                        "x_min": float(x_min),
                        "y_min": float(y_min),
                        "x_max": float(x_max),
                        "y_max": float(y_max)
                    },

                    "center": {
                        "x": float((x_min + x_max) / 2),
                        "y": float((y_min + y_max) / 2)
                    },

                    "width": float(x_max - x_min),
                    "height": float(y_max - y_min),

                    # Coordinates relative to rendered image
                    "coordinate_space": "rendered_pixels"
                })

            # Extract embedded images
            extracted_images = []

            for image_info in page.get_images(full=True):
                xref = image_info[0]

                image_data = document.extract_image(xref)

                image_bytes = image_data["image"]

                # Find image placement on page
                image_rects = page.get_image_rects(xref)

                for rect in image_rects:
                    extracted_images.append({
                        "xref": xref,
                        "extension": image_data["ext"],
                        "width": image_data["width"],
                        "height": image_data["height"],

                        "bbox": {
                            "x_min": float(rect.x0),
                            "y_min": float(rect.y0),
                            "x_max": float(rect.x1),
                            "y_max": float(rect.y1)
                        },

                        "coordinate_space": "pdf_points",

                        "image_base64": base64.b64encode(
                            image_bytes
                        ).decode("utf-8")
                    })

            extracted_document["pages"].append({
                "page_number": page_number,

                # PDF page dimensions, in points
                "width": float(page.rect.width),
                "height": float(page.rect.height),

                "rendered_width": pix.width,
                "rendered_height": pix.height,
                "render_scale": scale,

                "text_regions": text_regions,
                "images": extracted_images
            })

    finally:
        document.close()

    return extracted_document


#cluster the various document regions together based on spatial location
def cluster_document_regions(
        extracted_document: dict,
        threshold_x: float = 100,
        threshold_y: float = 50
) -> dict:

    clustered_document = {
        "source": extracted_document["source"],
        "pages": []
    }

    for page in extracted_document["pages"]:

        entities = []

        # Convert OCR text regions into PDF-point coordinates
        scale = page["render_scale"]

        for text_region in page["text_regions"]:
            bbox = text_region["bbox"]

            x_min = bbox["x_min"] / scale
            y_min = bbox["y_min"] / scale
            x_max = bbox["x_max"] / scale
            y_max = bbox["y_max"] / scale

            entities.append({
                "type": "text",
                "text": text_region["text"],
                "confidence": text_region["confidence"],
                "bbox": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                },
                "center": {
                    "x": (x_min + x_max) / 2,
                    "y": (y_min + y_max) / 2
                }
            })

        # Images are already in PDF-point coordinates
        for image in page["images"]:
            bbox = image["bbox"]

            x_min = bbox["x_min"]
            y_min = bbox["y_min"]
            x_max = bbox["x_max"]
            y_max = bbox["y_max"]

            entities.append({
                "type": "image",
                "image_base64": image["image_base64"],
                "extension": image["extension"],
                "bbox": bbox,
                "center": {
                    "x": (x_min + x_max) / 2,
                    "y": (y_min + y_max) / 2
                }
            })

        # Initialize Union-Find
        parent = list(range(len(entities)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            root_i = find(i)
            root_j = find(j)

            if root_i != root_j:
                parent[root_j] = root_i

        # Spatial proximity comparison
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):

                center_i = entities[i]["center"]
                center_j = entities[j]["center"]

                dx = abs(center_i["x"] - center_j["x"])
                dy = abs(center_i["y"] - center_j["y"])

                if dx <= threshold_x and dy <= threshold_y:
                    union(i, j)

        # Group connected entities
        groups = {}

        for i, entity in enumerate(entities):
            root = find(i)

            if root not in groups:
                groups[root] = []

            groups[root].append(entity)

        # Build final regions
        regions = []

        for group_id, group_entities in enumerate(groups.values(), start=1):

            region_x_min = min(
                e["bbox"]["x_min"] for e in group_entities
            )
            region_y_min = min(
                e["bbox"]["y_min"] for e in group_entities
            )
            region_x_max = max(
                e["bbox"]["x_max"] for e in group_entities
            )
            region_y_max = max(
                e["bbox"]["y_max"] for e in group_entities
            )

            regions.append({
                "region_id": group_id,
                "bbox": {
                    "x_min": region_x_min,
                    "y_min": region_y_min,
                    "x_max": region_x_max,
                    "y_max": region_y_max
                },
                "entities": group_entities
            })

        clustered_document["pages"].append({
            "page_number": page["page_number"],
            "regions": regions
        })

    return clustered_document


CONTENT_CATEGORIES = [
    "Patient Biodata",
    "Medical Data",
    "Analysis Report",
    "Conclusion",
]

#separate the content as bio-data, medical data, analysis and conclusions
def classify_content(extracted_document: dict) -> dict:
    question = (
        "Classify the given text into exactly one of these categories: "
        "Patient Biodata, Medical Data, Analysis Report, or Conclusion. "
        "Patient Biodata refers to patient demographics and personal details. "
        "Medical Data refers to clinical measurements, observations, "
        "diagnoses, and test results. "
        "Analysis Report refers to interpretation, reasoning, evaluation, "
        "and discussion of medical findings. "
        "Conclusion refers to final conclusions, summaries, or recommendations. "
        "Choose the category that best represents the content. "
        "If the text is insufficient to determine its category, abstain."
    )

    classified_document = {
        "source": extracted_document["source"],
        "pages": [],
    }

    for page in extracted_document["pages"]:
        classified_regions = []

        for region in page["regions"]:
            text_entities = [
                entity
                for entity in region["entities"]
                if entity["type"] == "text"
            ]

            image_entities = [
                entity
                for entity in region["entities"]
                if entity["type"] == "image"
            ]

            combined_text = "\n".join(
                entity["text"] for entity in text_entities
            ).strip()

            # Image-only regions remain unchanged.
            if not combined_text:
                classified_regions.append({
                    **region,
                    "classification": None,
                })
                continue

            decision = slave_decision(
                question=question,
                input_text=combined_text,
                options=CONTENT_CATEGORIES,
            )

            category = (
                "Unclassified"
                if decision["abstain"] or decision["unbound"]
                else decision["selected_option"]
            )

            classified_regions.append({
                **region,
                "classification": {
                    "category": category,
                    "decision": decision,
                },
            })

        classified_document["pages"].append({
            "page_number": page["page_number"],
            "regions": classified_regions,
        })

    return classified_document

#normalize the table sub-column titles and associate each table with title

#The output is a much more compacted JSON for representing tables, etc
def structure_medical_document(
        classified_document: dict,
        medical_data_extractor: Callable[[str], list[dict]]
) -> dict:

    structured_document = {
        "source": classified_document["source"],
        "patient_biodata": {},
        "medical_data": {
            "tables": []
        },
        "analysis_report": [],
        "conclusion": []
    }

    for page in classified_document["pages"]:

        for region in page["regions"]:

            classification = region.get("classification")

            # Skip regions that were not classified
            if not classification:
                continue

            category = classification.get("category")

            text_entities = [
                entity
                for entity in region["entities"]
                if entity["type"] == "text"
            ]

            # Arrange text in approximate reading order
            text_entities.sort(
                key=lambda entity: (
                    entity["bbox"]["y_min"],
                    entity["bbox"]["x_min"]
                )
            )

            region_text = "\n".join(
                entity["text"]
                for entity in text_entities
            ).strip()

            if not region_text:
                continue

            # --------------------------------------------------
            # 1. PATIENT BIODATA
            # --------------------------------------------------
            if category == "Patient Biodata":

                structured_document["patient_biodata"].setdefault(
                    "source_regions", []
                )

                structured_document["patient_biodata"]["source_regions"].append({
                    "page_number": page["page_number"],
                    "region_id": region["region_id"],
                    "text": region_text
                })

            # --------------------------------------------------
            # 2. MEDICAL DATA -> STRUCTURED TABLES
            # --------------------------------------------------
            elif category == "Medical Data":

                tables = medical_data_extractor(region_text)

                for table in tables:

                    structured_document["medical_data"]["tables"].append({
                        "page_number": page["page_number"],
                        "region_id": region["region_id"],
                        **table
                    })

            # --------------------------------------------------
            # 3. ANALYSIS REPORT -> PRESERVE NARRATIVE
            # --------------------------------------------------
            elif category == "Analysis Report":

                structured_document["analysis_report"].append({
                    "page_number": page["page_number"],
                    "region_id": region["region_id"],
                    "text": region_text
                })

            # --------------------------------------------------
            # 4. CONCLUSION -> PRESERVE NARRATIVE
            # --------------------------------------------------
            elif category == "Conclusion":

                structured_document["conclusion"].append({
                    "page_number": page["page_number"],
                    "region_id": region["region_id"],
                    "text": region_text
                })

    return structured_document




