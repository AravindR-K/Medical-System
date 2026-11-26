from pathlib import Path
import pydicom
from pydicom.datadict import dictionary_description



#Assumes that the pdf is within an extractable dim folder

def extract_dicom_metadata(file_path: str) -> list[dict]:
    """
    Extract DICOM metadata and resolve tag descriptions
    using the DICOM data dictionary.
    """

    path = Path(file_path)

    try:
        dataset = pydicom.dcmread(
            path,
            stop_before_pixels=True
        )
    except Exception as exc:
        raise ValueError(
            "File is not a readable DICOM object. "
            "A standalone PDF does not contain DICOM metadata."
        ) from exc

    metadata = []

    for element in dataset:
        metadata.append({
            "tag": str(element.tag),
            "name": dictionary_description(element.tag),
            "keyword": element.keyword,
            "vr": element.VR,
            "value": str(element.value)
            if element.VR != "SQ"
            else "[Sequence]"
        })

    return metadata
