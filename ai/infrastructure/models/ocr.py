import sys
import pymupdf  # PyMuPDF
import easyocr

#REQUIREMENTS:

def extract_pdf_content(pdf_path: str) -> str:
    reader = easyocr.Reader(["en"], gpu=False)
    document = pymupdf.open(pdf_path)

    extracted_content = []

    for page_number, page in enumerate(document, start=1):
        # Convert PDF page to image
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        image = pix.tobytes("png")

        # OCR the page
        results = reader.readtext(image, detail=1)

        extracted_content.append(f"\n--- Page {page_number} ---\n")

        # Extract recognized text
        for bbox, text, confidence in results:
            extracted_content.append(text)

    document.close()

    return "\n".join(extracted_content)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ocr.py <pdf_path>", file=sys.stderr)
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        content = extract_pdf_content(pdf_path)
        print(content)

    except Exception as e:
        print(f"OCR Error: {e}", file=sys.stderr)
        sys.exit(1)