"""Embedding generation and comparison utilities.

Demonstrates how to generate text embeddings using OpenAI's API, compare
vectors with cosine similarity, and find semantically similar texts.

See: docs/03-memory-and-retrieval/02-embeddings-and-vectors.md
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EmbeddingGenerator
# ---------------------------------------------------------------------------


class EmbeddingGenerator:
    """Generate embeddings from text using configurable OpenAI models.

    Args:
        model: The OpenAI embedding model to use.
        dimensions: Optional dimension reduction (only supported by
            text-embedding-3-* models). ``None`` uses the model default.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Core embedding methods
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            openai.OpenAIError: On API failure.
        """
        response = self._client.embeddings.create(
            model=self.model,
            input=text,
            **self._extra_kwargs(),
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single API call.

        Batching is significantly more efficient than calling :meth:`embed`
        in a loop — the API accepts up to 2,048 inputs per request.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of embedding vectors, in the same order as *texts*.

        Raises:
            openai.OpenAIError: On API failure.
        """
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self.model,
            input=texts,
            **self._extra_kwargs(),
        )
        # The API guarantees order matches the input, but sort by index for safety.
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]

    def embed_with_retry(
        self,
        text: str,
        max_retries: int = 3,
        backoff_base: float = 1.0,
    ) -> list[float]:
        """Embed a text with automatic exponential-backoff retry on transient failures.

        Args:
            text: The text to embed.
            max_retries: Maximum number of retry attempts.
            backoff_base: Base sleep time in seconds. Doubles each attempt.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            Exception: Re-raises the last exception after all retries are exhausted.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                return self.embed(text)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < max_retries:
                    sleep_secs = backoff_base * (2**attempt)
                    logger.warning(
                        "Embedding attempt %d/%d failed: %s. Retrying in %.1fs…",
                        attempt + 1,
                        max_retries,
                        exc,
                        sleep_secs,
                    )
                    time.sleep(sleep_secs)
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extra_kwargs(self) -> dict:
        """Build optional keyword arguments for the embeddings API call."""
        kwargs: dict = {}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        return kwargs


# ---------------------------------------------------------------------------
# EmbeddingComparator
# ---------------------------------------------------------------------------


class EmbeddingComparator:
    """Compare embeddings and find semantically similar texts."""

    # ------------------------------------------------------------------
    # Distance / similarity metrics
    # ------------------------------------------------------------------

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two embedding vectors.

        Returns a value in [-1, 1]. Values closer to 1 indicate semantically
        similar texts. For normalised vectors, this equals the dot product.

        Args:
            a: First embedding vector.
            b: Second embedding vector.

        Returns:
            Cosine similarity score.
        """
        a_arr = np.array(a, dtype=np.float64)
        b_arr = np.array(b, dtype=np.float64)
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if denom == 0.0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)

    def euclidean_distance(self, a: list[float], b: list[float]) -> float:
        """Calculate Euclidean distance between two embedding vectors.

        Smaller values indicate more similar texts. For text embeddings,
        prefer :meth:`cosine_similarity` unless you have a specific reason
        to care about vector magnitude.

        Args:
            a: First embedding vector.
            b: Second embedding vector.

        Returns:
            Euclidean distance (>= 0).
        """
        a_arr = np.array(a, dtype=np.float64)
        b_arr = np.array(b, dtype=np.float64)
        return float(np.linalg.norm(a_arr - b_arr))

    # ------------------------------------------------------------------
    # Higher-level utilities
    # ------------------------------------------------------------------

    def find_most_similar(
        self,
        query: str,
        candidates: list[str],
        generator: EmbeddingGenerator,
    ) -> list[dict]:
        """Rank candidate texts by semantic similarity to a query.

        Embeds the query and all candidates in a single batch call for
        efficiency, then ranks by cosine similarity.

        Args:
            query: The question or search string.
            candidates: Texts to compare against the query.
            generator: :class:`EmbeddingGenerator` to use for embedding.

        Returns:
            List of ``{"text": str, "score": float}`` dicts, sorted
            highest-similarity first.
        """
        all_texts = [query] + candidates
        all_embeddings = generator.embed_batch(all_texts)
        query_embedding = all_embeddings[0]
        candidate_embeddings = all_embeddings[1:]

        results = [
            {"text": text, "score": self.cosine_similarity(query_embedding, emb)}
            for text, emb in zip(candidates, candidate_embeddings)
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def visualize_similarity(
        self, texts: list[str], generator: EmbeddingGenerator
    ) -> str:
        """Generate a similarity matrix formatted as ASCII art.

        Example::

                        Text1  Text2  Text3
            Text1        1.00   0.85   0.12
            Text2        0.85   1.00   0.15
            Text3        0.12   0.15   1.00

        Args:
            texts: List of texts to compare pairwise.
            generator: :class:`EmbeddingGenerator` used to embed the texts.

        Returns:
            A multi-line string ready to print.
        """
        embeddings = generator.embed_batch(texts)
        n = len(texts)
        labels = [f"Text{i + 1}" for i in range(n)]
        col_w = 7  # Width of each score column
        label_w = max(len(lbl) for lbl in labels) + 2

        # Header row
        header = " " * label_w + "".join(f"{lbl:>{col_w}}" for lbl in labels)
        rows = [header]
        for i in range(n):
            row = f"{labels[i]:<{label_w}}"
            for j in range(n):
                score = self.cosine_similarity(embeddings[i], embeddings[j])
                row += f"{score:>{col_w}.2f}"
            rows.append(row)
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate embedding generation and semantic similarity comparison."""
    generator = EmbeddingGenerator(model="text-embedding-3-small")
    comparator = EmbeddingComparator()

    # --- 10 texts across 3 semantic categories ---
    weather_texts = [
        "It's raining heavily outside right now.",
        "The forecast calls for sunny skies all week.",
        "A blizzard is expected to hit the northeast tonight.",
    ]
    stock_texts = [
        "The S&P 500 rose 1.2% on strong earnings reports.",
        "Investors are worried about rising interest rates.",
        "Tech stocks led the market rally this afternoon.",
        "The Federal Reserve signalled a rate pause today.",
    ]
    sports_texts = [
        "The home team scored three goals in the second half.",
        "The marathon world record was broken yesterday.",
        "Championship play-offs begin next weekend.",
    ]
    all_texts = weather_texts + stock_texts + sports_texts
    categories = (
        ["weather"] * len(weather_texts)
        + ["stocks"] * len(stock_texts)
        + ["sports"] * len(sports_texts)
    )

    print("Generating embeddings for 10 texts across 3 categories…")
    embeddings = generator.embed_batch(all_texts)
    print(f"  Each embedding has {len(embeddings[0])} dimensions.\n")

    # --- Within-category vs. cross-category similarity ---
    print("=== Within-category similarity (expect > 0.7) ===")
    for i in range(len(all_texts)):
        for j in range(i + 1, len(all_texts)):
            if categories[i] == categories[j]:
                sim = comparator.cosine_similarity(embeddings[i], embeddings[j])
                print(f"  [{categories[i]}]  {sim:.3f}")

    print("\n=== Cross-category similarity (expect < 0.3) ===")
    for i in range(len(all_texts)):
        for j in range(i + 1, len(all_texts)):
            if categories[i] != categories[j]:
                sim = comparator.cosine_similarity(embeddings[i], embeddings[j])
                label = f"{categories[i]} vs {categories[j]}"
                print(f"  [{label}]  {sim:.3f}")

    # --- Similarity matrix (first 4 texts only for readability) ---
    print("\n=== Similarity matrix (first 4 texts) ===")
    matrix = comparator.visualize_similarity(all_texts[:4], generator)
    print(matrix)

    # --- Cosine vs. Euclidean on the same pair ---
    print("\n=== Cosine similarity vs. Euclidean distance ===")
    a, b, c = embeddings[0], embeddings[1], embeddings[4]
    print(f"  Weather[0] vs Weather[1]  — cosine: {comparator.cosine_similarity(a, b):.3f},"
          f" euclidean: {comparator.euclidean_distance(a, b):.3f}")
    print(f"  Weather[0] vs Stocks[0]  — cosine: {comparator.cosine_similarity(a, c):.3f},"
          f" euclidean: {comparator.euclidean_distance(a, c):.3f}")

    # --- Semantic query ---
    print("\n=== Query: 'What's the market doing today?' ===")
    results = comparator.find_most_similar(
        "What's the market doing today?", all_texts, generator
    )
    for rank, r in enumerate(results[:5], 1):
        print(f"  #{rank}  score={r['score']:.3f}  {r['text'][:60]}")


if __name__ == "__main__":
    main()
