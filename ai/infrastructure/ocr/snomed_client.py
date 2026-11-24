import requests
from typing import Any


#USE SNOWMED for standardizing table sub-headers

#CHECK FOR ENABLE CONCURRENT BATCHING OF STANDARDIZING TERMS


#search_concepts() : Searches SNOMED CT for matching medical concepts.
#get_concept() : Retrieves the complete details of a particular SNOMED CT concept using its ID.
#close() : Closes the HTTP session.

class SnomedClient:

    BASE_URL = (
        "https://snowstorm.snomedtools.org/"
        "snowstorm/snomed-ct"
    )

    BRANCH = "MAIN"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()

        self.session.headers.update({
            "Accept": "application/json",
            "Accept-Language": "en"
        })


    #------------------------------------------

    def search_concepts(
            self,
            medical_term: str,
            limit: int = 10
    ) -> dict[str, Any]:

        endpoint = (
            f"{self.BASE_URL}/{self.BRANCH}/concepts"
        )

        params = {
            "term": medical_term,
            "activeFilter": "true",
            "termActive": "true",
            "limit": limit,
            "offset": 0
        }

        response = self.session.get(
            endpoint,
            params=params,
            timeout=self.timeout
        )

        response.raise_for_status()

        return response.json()

    #Example expected output:
    # {
    #     "concept_id": "80394007",
    #     "standardized_term": "Hyperglycemia",
    #     "fully_specified_name": "Hyperglycemia (disorder)"
    # }

    #------------------------------------------


    #------------------------------------------

    def get_concept(
            self,
            concept_id: str
    ) -> dict[str, Any]:

        endpoint = (
            f"{self.BASE_URL}/{self.BRANCH}/concepts/"
            f"{concept_id}"
        )

        response = self.session.get(
            endpoint,
            timeout=self.timeout
        )

        response.raise_for_status()

        return response.json()

    #OPTIONAL. Retrieve complete details of a selected concept

    #------------------------------------------

    def close(self):
        self.session.close()
