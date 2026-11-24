from transformers import AutoProcessor, AutoModelForMultimodalLM
import torch

#using a class so that the model and processor are initialized only once and reused across multiple batches.

class MedGemmaClient:

    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            "google/medgemma-4b-it"
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-4b-it",
            device_map="auto"
        )

        # Left padding is important for batched decoder-only generation.
        self.processor.tokenizer.padding_side = "left"

    def converse_batch(
            self,
            requests: list[dict[str, str | None]]
    ) -> list[str]:

        conversations = []

        for request in requests:

            image_url = request.get("image_url")
            text = request.get("text")

            content = []

            if image_url:
                content.append({
                    "type": "image",
                    "url": image_url
                })

            if text:
                content.append({
                    "type": "text",
                    "text": text
                })

            if not content:
                raise ValueError(
                    "Each request must contain an image or text."
                )

            conversations.append([
                {
                    "role": "user",
                    "content": content
                }
            ])

        # Tokenize all conversations together.
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        # Single batched inference call.
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256
        )

        # Remove the padded input portion.
        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]

        # Decode each response independently.
        responses = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return responses


