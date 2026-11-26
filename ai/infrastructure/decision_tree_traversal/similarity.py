
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class SapBERTEmbedder:

    def __init__(self):
        self.model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            device_map="auto"
        )

        self.model.eval()

    def embed_batch(
            self,
            strings: list[str],
            batch_size: int = 32,
            max_length: int = 128
    ) -> dict[str, torch.Tensor]:

        embeddings = {}

        device = self.model.device

        with torch.no_grad():

            for i in range(0, len(strings), batch_size):

                batch = strings[i:i + batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(device)

                outputs = self.model(**inputs)

                # Mean pooling, excluding padding tokens
                token_embeddings = outputs.last_hidden_state

                attention_mask = inputs["attention_mask"].unsqueeze(-1)

                sum_embeddings = torch.sum(
                    token_embeddings * attention_mask,
                    dim=1
                )

                sum_mask = torch.clamp(
                    attention_mask.sum(dim=1),
                    min=1e-9
                )

                batch_embeddings = sum_embeddings / sum_mask

                # L2 normalization
                batch_embeddings = F.normalize(
                    batch_embeddings,
                    p=2,
                    dim=1
                )

                for string, embedding in zip(batch, batch_embeddings):
                    embeddings[string] = embedding.cpu()

        return embeddings




from rapidfuzz.fuzz import ratio


def levenshtein_similarity(
        words: list[str],
        master_word: str
) -> dict[str, float]:

    return {
        word: ratio(word, master_word) / 100.0
        for word in words
    }


import torch
import torch.nn.functional as F

def cosine_similarity_batch(
        embedding_dict: dict[str, torch.Tensor],
        master_word: str,
        embedder: SapBERTEmbedder
) -> dict[str, float]:

    # Calculate master embedding only once
    master_embedding = embedder.embed_batch(
        [master_word]
    )[master_word]

    master_embedding = F.normalize(
        master_embedding.float(),
        p=2,
        dim=0
    )

    similarities = {}

    for word, embedding in embedding_dict.items():

        embedding = F.normalize(
            embedding.float(),
            p=2,
            dim=0
        )

        similarity = F.cosine_similarity(
            embedding.unsqueeze(0),
            master_embedding.unsqueeze(0)
        )

        similarities[word] = similarity.item()

    return similarities


def fuzzy_prune(
        levenshtein_scores: dict[str, float],
        cosine_scores: dict[str, float],
        threshold: float,
        levenshtein_weight: float = 0.3,
        cosine_weight: float = 0.7
) -> dict[str, float]:

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    if levenshtein_weight < 0 or cosine_weight < 0:
        raise ValueError("Weights cannot be negative")

    if not abs(levenshtein_weight + cosine_weight - 1.0) < 1e-9:
        raise ValueError("Weights must sum to 1")

    results = {}

    for word in levenshtein_scores.keys():

        if word not in cosine_scores:
            continue

        overall_similarity = (
                levenshtein_weight * levenshtein_scores[word]
                + cosine_weight * cosine_scores[word]
        )

        if overall_similarity >= threshold:
            results[word] = overall_similarity

    return dict(
        sorted(
            results.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )


#MAIN FUNCTION

def match_medical_terms(
        words: list[str],
        master_word: str,
        embedder: SapBERTEmbedder,
        threshold: float = 0.75,
        levenshtein_weight: float = 0.3,
        cosine_weight: float = 0.7
) -> dict[str, float]:

    # Step 1: Lexical similarity
    levenshtein_scores = levenshtein_similarity(
        words,
        master_word
    )

    # Step 2: Generate embeddings for all candidate words
    embedding_dict = embedder.embed_batch(words)

    # Step 3: Semantic similarity
    cosine_scores = cosine_similarity_batch(
        embedding_dict,
        master_word,
        embedder
    )

    # Step 4: Aggregate scores and prune candidates
    filtered_words = fuzzy_prune(
        levenshtein_scores,
        cosine_scores,
        threshold=threshold,
        levenshtein_weight=levenshtein_weight,
        cosine_weight=cosine_weight
    )

    return filtered_words

