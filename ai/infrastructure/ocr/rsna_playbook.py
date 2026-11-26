#USE RSNA's RadLex PLAYBOOK for standardising DICOM header

#LOINC data

# Particularly important: Section 10
# If you incorporate Playbook data into your own application
# or online terminology service, you need to include this notice:
# This material contains content from LOINC (http://loinc.org/).
# LOINC is copyright © Regenstrief Institute, Inc. and the
# Logical Observation Identifiers Names and Codes (LOINC) Committee
# and is available at no cost under the license at http://loinc.org/license.
# LOINC® is a registered United States trademark of Regenstrief Institute,
# Inc.


# I have downloaded the "Simplified RPID to LOINC codes.xlsx" as a csv file-in the
# "download LOINC-RSNA playbook" section on the official website. give me the code for
# consulting that csv(assumed to be in an external database, which you can reference
# via querying to an open-ended endpoint) for converting a piece of text to the
# RSNA-compatible name for the radiology exam conducted


# Function Capabilities:
#
# 1. Receive the extracted examination description.
# 2. Query your external database through an HTTP endpoint.
# 3. Retrieve matching LOINC–RPID mappings.
# 4. Return the corresponding RSNA Playbook name, RPID and LOINC code.



import requests
from typing import Optional


def standardize_radiology_exam(
        text: str,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout: int = 10
) -> dict:
    """
    Find the corresponding RSNA-compatible radiology
    examination name using an external LOINC-RPID database.
    """

    if not text or not text.strip():
        raise ValueError("Examination description cannot be empty")

    headers = {}

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    params = {
        "q": text.strip()
    }

    response = requests.get(
        endpoint,
        params=params,
        headers=headers,
        timeout=timeout
    )

    response.raise_for_status()

    data = response.json()

    # Expected response:
    # {
    #   "results": [
    #       {
    #           "rpid": "...",
    #           "loinc_code": "...",
    #           "rsna_name": "...",
    #           "score": 0.98
    #       }
    #   ]
    # }

    results = data.get("results", [])

    if not results:
        return {
            "original_term": text,
            "standardized_name": None,
            "rpid": None,
            "loinc_code": None,
            "confidence": 0.0,
            "status": "no_match"
        }

    # Highest-ranked candidate
    best_match = max(
        results,
        key=lambda item: item.get("score", 0.0)
    )

    return {
        "original_term": text,
        "standardized_name": best_match.get("rsna_name"),
        "rpid": best_match.get("rpid"),
        "loinc_code": best_match.get("loinc_code"),
        "confidence": best_match.get("score", 0.0),
        "status": "matched"
    }
