"""Embedding visualization utilities for building semantic intuition.

Provides text similarity matrices, outlier detection, and K-means clustering
over embedding vectors — all without requiring matplotlib or a display.

See: docs/03-memory-and-retrieval/02-embeddings-and-vectors.md
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from embedding_generator import EmbeddingGenerator, EmbeddingComparator


class EmbeddingVisualizer:
    """Visualize embeddings to understand semantic relationships.

    Uses the provided :class:`EmbeddingGenerator` for all embedding calls.
    All output is plain text so it works in any terminal or log stream.

    Args:
        generator: Pre-configured :class:`EmbeddingGenerator` to use.
    """

    def __init__(self, generator: EmbeddingGenerator) -> None:
        self.generator = generator
        self._comparator = EmbeddingComparator()

    # ------------------------------------------------------------------
    # Similarity matrix
    # ------------------------------------------------------------------

    def compare_texts(
        self,
        texts: list[str],
        labels: Optional[list[str]] = None,
    ) -> str:
        """Generate a similarity matrix as a formatted Unicode table.

        Args:
            texts: Texts to compare pairwise.
            labels: Optional display labels. Defaults to "Text 1", "Text 2", …

        Returns:
            Multi-line string ready to print.

        Example output::

            ┌──────────────┬────────┬─────────┬────────┐
            │              │ Text 1 │  Text 2 │ Text 3 │
            ├──────────────┼────────┼─────────┼────────┤
            │ Text 1       │  1.00  │   0.85  │  0.12  │
            │ Text 2       │  0.85  │   1.00  │  0.15  │
            │ Text 3       │  0.12  │   0.15  │  1.00  │
            └──────────────┴────────┴─────────┴────────┘
        """
        if labels is None:
            labels = [f"Text {i + 1}" for i in range(len(texts))]
        if len(labels) != len(texts):
            raise ValueError("len(labels) must equal len(texts)")

        embeddings = self.generator.embed_batch(texts)
        n = len(texts)
        scores = [
            [
                self._comparator.cosine_similarity(embeddings[i], embeddings[j])
                for j in range(n)
            ]
            for i in range(n)
        ]

        col_w = 8
        label_w = max(len(lbl) for lbl in labels) + 2
        h_line = "─" * label_w
        col_lines = ["─" * col_w for _ in labels]

        top = "┌" + h_line + "┬" + "┬".join(col_lines) + "┐"
        header_cells = "│" + " " * label_w + "│" + "│".join(
            f" {lbl:^{col_w - 1}}" for lbl in labels
        ) + "│"
        divider = "├" + h_line + "┼" + "┼".join(col_lines) + "┤"
        bottom = "└" + h_line + "┴" + "┴".join(col_lines) + "┘"

        data_rows = []
        for i in range(n):
            cells = "│".join(f" {scores[i][j]:^{col_w - 1}.2f}" for j in range(n))
            data_rows.append("│" + f" {labels[i]:<{label_w - 1}}" + "│" + cells + "│")

        return "\n".join([top, header_cells, divider] + data_rows + [bottom])

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------

    def find_outliers(
        self,
        texts: list[str],
        labels: Optional[list[str]] = None,
        threshold_percentile: float = 25.0,
    ) -> list[str]:
        """Find texts that are semantically different from all others.

        A text is considered an outlier if its average similarity to all
        other texts falls below the *threshold_percentile* of the pairwise
        distribution.

        Args:
            texts: Texts to analyse.
            labels: Display labels (defaults to the texts themselves).
            threshold_percentile: Percentile cutoff for the average similarity.

        Returns:
            List of labels (or texts, if no labels are provided) identified
            as outliers.
        """
        if labels is None:
            labels = texts

        embeddings = self.generator.embed_batch(texts)
        n = len(texts)
        avg_sims = []
        for i in range(n):
            sims = [
                self._comparator.cosine_similarity(embeddings[i], embeddings[j])
                for j in range(n)
                if j != i
            ]
            avg_sims.append(sum(sims) / len(sims) if sims else 1.0)

        cutoff = float(np.percentile(avg_sims, threshold_percentile))
        return [labels[i] for i in range(n) if avg_sims[i] < cutoff]

    # ------------------------------------------------------------------
    # K-means clustering
    # ------------------------------------------------------------------

    def cluster(
        self,
        texts: list[str],
        n_clusters: int = 3,
        max_iterations: int = 100,
        random_seed: int = 42,
    ) -> dict[int, list[str]]:
        """Group texts into semantic clusters using K-means on their embeddings.

        This is a plain NumPy implementation — no scikit-learn required.

        Args:
            texts: Texts to cluster.
            n_clusters: Number of clusters to produce.
            max_iterations: Maximum K-means iterations.
            random_seed: Seed for reproducibility.

        Returns:
            Dict mapping cluster index → list of texts in that cluster.
        """
        embeddings = self.generator.embed_batch(texts)
        matrix = np.array(embeddings, dtype=np.float64)  # shape (n, dim)
        n = len(texts)

        rng = np.random.default_rng(random_seed)
        centroid_indices = rng.choice(n, size=n_clusters, replace=False)
        centroids = matrix[centroid_indices].copy()

        assignments = np.zeros(n, dtype=int)
        for _ in range(max_iterations):
            # Assign each text to the nearest centroid.
            new_assignments = np.array(
                [_nearest_centroid(matrix[i], centroids) for i in range(n)],
                dtype=int,
            )
            if np.array_equal(new_assignments, assignments):
                break
            assignments = new_assignments

            # Recompute centroids.
            for k in range(n_clusters):
                members = matrix[assignments == k]
                if len(members) > 0:
                    centroids[k] = members.mean(axis=0)

        result: dict[int, list[str]] = {k: [] for k in range(n_clusters)}
        for i, text in enumerate(texts):
            result[assignments[i]].append(text)
        return result

    # ------------------------------------------------------------------
    # Built-in demo
    # ------------------------------------------------------------------

    def demo_semantic_categories(self) -> None:
        """Show clear semantic clustering across weather, technology, and food.

        Demonstrates:
        - Within-category similarity > 0.65
        - Cross-category similarity < 0.30
        - K-means clustering correctly groups all 9 texts into 3 clusters
        """
        categories = {
            "weather": [
                "It's raining outside",
                "Sunny with a high of 75",
                "Snow expected tomorrow",
            ],
            "technology": [
                "Python 3.12 released",
                "New GPU architecture announced",
                "API deprecation notice",
            ],
            "food": [
                "Best pizza in New York",
                "How to make sourdough bread",
                "Restaurant review: French cuisine",
            ],
        }

        all_texts = [t for texts in categories.values() for t in texts]
        all_labels = list(categories)
        label_per_text = [
            lbl for lbl, texts in categories.items() for _ in texts
        ]

        print("=== Similarity matrix (9 texts × 9 texts) ===")
        short_labels = [
            f"{lbl[:4]}#{i + 1}"
            for lbl, texts in categories.items()
            for i in range(len(texts))
        ]
        print(self.compare_texts(all_texts, labels=short_labels))

        embeddings = self.generator.embed_batch(all_texts)

        print("\n=== Within-category similarities (expect > 0.65) ===")
        for cat_name, texts in categories.items():
            cat_indices = [i for i, lbl in enumerate(label_per_text) if lbl == cat_name]
            pairs = [
                (i, j)
                for i in cat_indices
                for j in cat_indices
                if i < j
            ]
            for i, j in pairs:
                sim = self._comparator.cosine_similarity(embeddings[i], embeddings[j])
                status = "✓" if sim > 0.65 else "✗"
                print(f"  {status} [{cat_name}]  {sim:.3f}")

        print("\n=== Cross-category similarities (expect < 0.30) ===")
        cat_names = list(categories)
        for ci in range(len(cat_names)):
            for cj in range(ci + 1, len(cat_names)):
                cat_a, cat_b = cat_names[ci], cat_names[cj]
                indices_a = [i for i, lbl in enumerate(label_per_text) if lbl == cat_a]
                indices_b = [i for i, lbl in enumerate(label_per_text) if lbl == cat_b]
                sims = [
                    self._comparator.cosine_similarity(embeddings[ia], embeddings[ib])
                    for ia in indices_a
                    for ib in indices_b
                ]
                avg = sum(sims) / len(sims)
                status = "✓" if avg < 0.30 else "✗"
                print(f"  {status} [{cat_a} vs {cat_b}]  avg={avg:.3f}")

        print("\n=== K-means clustering (k=3) ===")
        clusters = self.cluster(all_texts, n_clusters=3)
        for cluster_id, members in clusters.items():
            print(f"  Cluster {cluster_id}:")
            for text in members:
                print(f"    - {text}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nearest_centroid(vector: np.ndarray, centroids: np.ndarray) -> int:
    """Return the index of the centroid closest to *vector* (cosine distance)."""
    sims = centroids @ vector / (
        np.linalg.norm(centroids, axis=1) * np.linalg.norm(vector) + 1e-10
    )
    return int(np.argmax(sims))


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the built-in semantic categories demo."""
    gen = EmbeddingGenerator(model="text-embedding-3-small")
    viz = EmbeddingVisualizer(gen)
    viz.demo_semantic_categories()

    # --- Outlier detection ---
    texts = [
        "The weather is sunny today.",
        "It will rain tomorrow afternoon.",
        "Clear skies expected all week.",
        "The quarterly earnings beat expectations.",  # outlier in weather context
    ]
    print("\n=== Outlier detection ===")
    outliers = viz.find_outliers(texts)
    print(f"  Outliers: {outliers}")


if __name__ == "__main__":
    main()
