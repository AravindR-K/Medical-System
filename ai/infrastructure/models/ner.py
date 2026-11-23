

SEARCH_SPACES = {
    "cardiology": {
        "procedure_codes": ["CARD001", "CARD002"],
        "dicom_codes": ["CARDIAC"],
        "files": ["cardiology_knowledge.json", "cardiac_cases.json"]
    },
    "gynaecology": {
        "procedure_codes": ["GYN001", "GYN002"],
        "dicom_codes": ["GYNECOLOGY"],
        "files": ["gynaecology_knowledge.json"]
    },
    "neurology": {
        "procedure_codes": ["NEU001", "NEU002"],
        "dicom_codes": ["NEURO"],
        "files": ["neurology_knowledge.json"]
    }
}



#Takes the structured output from OCR pipeline
#Metadata pre-filtering → determines which medical search space
# (cardiology,gynaecology, neurology, etc.) should be searched.

#This implementation uses spaCy + medspaCy ConText for medical entity
# recognition and assertion detection.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import spacy
import medspacy


@dataclass(frozen=True)
class OCRTextSegment:
    text: str
    confidence: float | None
    page_number: int | None
    element_id: str | None
    bbox: list | None


class MedicalMentionAnalyzer:
    """
    Infrastructure adapter for:
        1. Medical NER
        2. Assertion / negation detection
        3. OCR-confidence-aware extraction

    Original OCR content is not modified.
    """

    def __init__(
            self,
            model_name: str = "en_core_sci_sm",
            minimum_ocr_confidence: float = 0.0,
    ):
        self.minimum_ocr_confidence = minimum_ocr_confidence

        # Load biomedical NLP model
        self.nlp = medspacy.load(model_name)

        # ConText assertion detection
        if "medspacy_context" not in self.nlp.pipe_names:
            self.nlp.add_pipe("medspacy_context")

    def analyze(self, extracted_document: dict) -> dict:
        segments = self._extract_segments(extracted_document)

        mentions = []
        affirmed_sentences = []
        negated_sentences = []
        uncertain_sentences = []
        historical_sentences = []

        for segment in segments:

            # OCR quality filtering
            if (
                    segment.confidence is not None
                    and segment.confidence < self.minimum_ocr_confidence
            ):
                continue

            doc = self.nlp(segment.text)

            sentence_mentions = []

            for entity in doc.ents:

                assertion = self._get_assertion(entity)

                mention = {
                    "text": entity.text,
                    "entity_type": entity.label_,
                    "sentence": entity.sent.text.strip(),
                    "assertion": assertion,
                    "ocr_confidence": segment.confidence,
                    "page_number": segment.page_number,
                    "element_id": segment.element_id,
                    "bbox": segment.bbox,
                }

                mentions.append(mention)
                sentence_mentions.append(mention)

            # Keep the original sentence in the amplified output.
            # Assertion filtering is applied to the entity context,
            # not by deleting the original OCR sentence.

            assertions_in_sentence = {
                mention["assertion"] for mention in sentence_mentions
            }

            if "AFFIRMED" in assertions_in_sentence:
                affirmed_sentences.append(segment.text)

            if "NEGATED" in assertions_in_sentence:
                negated_sentences.append(segment.text)

            if "UNCERTAIN" in assertions_in_sentence:
                uncertain_sentences.append(segment.text)

            if "HISTORICAL" in assertions_in_sentence:
                historical_sentences.append(segment.text)

        return {
            "mentions": mentions,
            "affirmed_sentences": self._unique(affirmed_sentences),
            "negated_sentences": self._unique(negated_sentences),
            "uncertain_sentences": self._unique(uncertain_sentences),
            "historical_sentences": self._unique(historical_sentences),
        }

    @staticmethod
    def _get_assertion(entity) -> str:

        # medspaCy ConText assertion attributes
        if entity._.is_negated:
            return "NEGATED"

        if entity._.is_uncertain:
            return "UNCERTAIN"

        if entity._.is_historical:
            return "HISTORICAL"

        if entity._.is_hypothetical:
            return "HYPOTHETICAL"

        return "AFFIRMED"

    def _extract_segments(
            self,
            extracted_document: dict
    ) -> list[OCRTextSegment]:

        segments = []

        for page in extracted_document.get("pages", []):
            page_number = page.get("page_number")

            for index, element in enumerate(page.get("elements", [])):

                text = element.get("text")

                if not isinstance(text, str) or not text.strip():
                    continue

                # Ignore non-text regions
                element_type = element.get("type", "")

                if element_type in {"image", "table"}:
                    continue

                confidence = element.get("confidence")

                try:
                    confidence = (
                        float(confidence)
                        if confidence is not None
                        else None
                    )
                except (TypeError, ValueError):
                    confidence = None

                segments.append(
                    OCRTextSegment(
                        text=text.strip(),
                        confidence=confidence,
                        page_number=page_number,
                        element_id=element.get("element_id", str(index)),
                        bbox=element.get("bbox"),
                    )
                )

        return segments

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        return list(dict.fromkeys(
            value.strip()
            for value in values
            if value and value.strip()
        ))



#Medical NER + assertion filtering → extracts relevant medical entities

#This adapter adds a new section for Positive entries without deleting or rewriting any existing OCR fields.

from copy import deepcopy


class OCRResultEnricher:

    def __init__(self, medical_mention_analyzer):
        self.medical_mention_analyzer = medical_mention_analyzer

    def enrich(self, extracted_document: dict) -> dict:

        # Preserve the original document.
        enriched_document = deepcopy(extracted_document)

        medical_analysis = self.medical_mention_analyzer.analyze(
            enriched_document
        )

        enriched_document["medical_analysis"] = medical_analysis

        return enriched_document