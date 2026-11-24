#PROCESSING OF IMAGE USING MEDGEMMA

from ..models.conversational_agent import MedGemmaClient

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


# Batch failure: This implementation treats a failed batch as a failed batch.
# If you need partial recovery, implement chunked batches with individual retries.
#
# Batch size: If a PDF contains hundreds of images, don't submit everything in one batch.
# Process chunks (e.g. 8–16 images), depending on your inference server's capacity and memory.

#Take OCR output, find the images via their image urls, pass to medgemma via batching
#and resolve to text output


from pathlib import Path

def process_images(ocr_result: dict) -> dict:
    medgemma = MedGemmaClient()

    """
    Processes extracted medical images using MedGemma batch inference.

    Returns
    -------
    dict
    {
        "document_title": str | None,
        "source_pdf": str,
        "total_images": int,
        "processed_images": int,
        "findings": list[dict],
        "errors": list[dict]
    }
    """

    image_directory = Path(ocr_result["image_directory"])

    if not image_directory.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_directory}"
        )

    supported_extensions = {
        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"
    }

    image_paths = sorted(
        path for path in image_directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )

    document_title = ocr_result.get("document_title")
    source_pdf = ocr_result.get("source_pdf", "")

    findings = []
    errors = []

    prompt_template = """
You are a medical imaging analysis assistant powered by MedGemma.

Carefully examine the provided image and produce a detailed,
structured radiological observation report.

DOCUMENT CONTEXT:
Document title: {document_title}
Source document: {source_pdf}

ANALYSIS INSTRUCTIONS:

1. IMAGE IDENTIFICATION
   - Identify the imaging modality, if recognizable.
   - Identify the anatomical region and structures visible.
   - Describe orientation and image quality.

2. ANATOMICAL ASSESSMENT
   - Examine visible anatomical structures systematically.
   - Describe morphology, contours, dimensions where measurable,
     relative positioning and structural characteristics.
   - Identify visible asymmetries or structural deviations.

3. ABNORMALITY IDENTIFICATION
   - Identify visible abnormalities or suspicious findings.
   - Examine for lesions, masses, opacities, calcifications,
     fluid collections and other relevant imaging features.
   - Describe location, extent, morphology and appearance.
   - Distinguish definite observations from uncertain possibilities.

4. NORMAL FINDINGS
   - Mention relevant structures that appear unremarkable.
   - Do not invent normal findings for structures that
     cannot be adequately visualized.

5. LIMITATIONS
   - Identify limitations due to image quality, cropping,
     resolution, missing views or artifacts.
   - Do not infer findings outside the visible image.
   - Do not fabricate measurements or clinical history.

OUTPUT FORMAT:

Image type:
[Modality and anatomical region]

Image quality:
[Assessment]

Anatomical observations:
[Detailed observations]

Potential abnormalities:
[Findings, location and characteristics]

Unremarkable structures:
[Relevant normal observations]

Limitations:
[Interpretation limitations]

Overall findings:
[Concise but sufficiently detailed synthesis]

IMPORTANT:
- Be precise and evidence-based.
- Do not assume that an abnormality exists.
- Do not confuse artifacts with anatomical abnormalities.
- Explicitly communicate uncertainty.
- Do not provide a definitive diagnosis based solely on this image.
- If the image is not medical, identify its actual content.
"""

    # Build batch requests
    requests = []

    for image_path in image_paths:
        prompt = prompt_template.format(
            document_title=document_title or "Not provided",
            source_pdf=source_pdf or "Not provided"
        )

        requests.append({
            "image_url": str(image_path),
            "text": prompt
        })

    if not requests:
        return {
            "document_title": document_title,
            "source_pdf": source_pdf,
            "total_images": 0,
            "processed_images": 0,
            "findings": [],
            "errors": []
        }

    # Submit all images together
    try:
        responses = medgemma.converse_batch(requests)

        if len(responses) != len(image_paths):
            raise RuntimeError(
                "Mismatch between submitted requests and responses."
            )

        for image_path, response in zip(image_paths, responses):

            findings.append({
                "image_name": image_path.name,
                "image_path": str(image_path),
                "findings": response
            })

    except Exception as exc:
        errors.append({
            "error": str(exc),
            "stage": "batch_inference"
        })

    return {
        "document_title": document_title,
        "source_pdf": source_pdf,
        "total_images": len(image_paths),
        "processed_images": len(findings),
        "findings": findings,
        "errors": errors
    }





