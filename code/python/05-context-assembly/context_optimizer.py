"""Context Optimizer — structure and prioritize context for LLM attention.

Implements research-backed techniques for maximising the model's ability to
find and use information within a context window:

* Structured document layout with table-of-contents and section markers.
* Position-aware re-ordering so critical facts land in the 20-60% "golden
  middle" zone where models recall information best.
* Chunk deduplication to remove near-identical passages (character 3-gram
  Jaccard similarity).
* Attention-score estimation based on position and structural salience.

See: docs/04-context-engineering/01-the-context-window-as-a-resource.md
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Attention zone boundaries (fractions of total context character length)
# ---------------------------------------------------------------------------

_PRIMACY_END    = 0.10   # 0 – 10%  : first impression, often under-attended
_OPTIMAL_START  = 0.20   # 20 – 60% : best recall zone
_OPTIMAL_END    = 0.60
_RECENCY_START  = 0.80   # 80 – 100%: recency bias helps, but crowded at the end


@dataclass
class AttentionZones:
    """Context string split into expected-attention quality regions.

    Attributes:
        primacy_zone:  0–10% — first content the model reads.
        transition_12: 10–20% — transition from primacy to optimal.
        optimal_zone:  20–60% — best recall; place critical facts here.
        transition_23: 60–80% — transition from optimal to recency.
        recency_zone:  80–100% — last content the model reads.
    """

    primacy_zone:  str
    transition_12: str
    optimal_zone:  str
    transition_23: str
    recency_zone:  str


class ContextOptimizer:
    """Optimize context structure and content for LLM comprehension.

    Example::

        optimizer  = ContextOptimizer(model="gpt-4o")
        structured = optimizer.structure_for_retrieval(documents)
        optimized  = optimizer.optimize_for_query(structured, query)
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def structure_for_retrieval(self, documents: list[dict]) -> str:
        """Build a context string optimised for needle-in-a-haystack retrieval.

        Applies five techniques:

        1. **Table of contents** at the top so the model builds a mental map.
        2. **Numbered section markers** (``## [N] Title``) for clear hierarchy.
        3. ``[End Section N]`` footers so the model knows where each source ends.
        4. ``[IMPORTANT]`` tags on documents whose ``metadata["important"]``
           flag is ``True``.
        5. Visual separator lines between sections.

        Args:
            documents: List of dicts with ``text`` and ``metadata`` keys.
                       Optionally ``metadata["important"]`` (bool) marks a
                       document as critical.

        Returns:
            Structured multi-section context string.
        """
        parts: list[str] = []

        # --- Table of contents ---
        parts.append("## Context Overview\n")
        for i, doc in enumerate(documents, start=1):
            source = doc.get("metadata", {}).get("source", f"Document {i}")
            important = doc.get("metadata", {}).get("important", False)
            tag = " [IMPORTANT]" if important else ""
            parts.append(f"- Section {i}: {source}{tag}")
        parts.append("\n---\n")

        # --- Document sections ---
        for i, doc in enumerate(documents, start=1):
            source = doc.get("metadata", {}).get("source", f"Document {i}")
            important = doc.get("metadata", {}).get("important", False)
            importance_tag = (
                "\n> **[IMPORTANT — Key information]**" if important else ""
            )
            text = doc.get("text", "")
            parts.append(f"## [{i}] {source}{importance_tag}\n")
            parts.append(text)
            parts.append(f"\n[End Section {i}]\n")
            parts.append("---\n")

        return "\n".join(parts)

    def prioritize_information(self, context: str, query: str) -> str:
        """Re-order context so the most query-relevant section lands in the
        optimal attention zone (20–60% of the total context).

        Sections are identified by ``## [N]`` headings produced by
        :meth:`structure_for_retrieval`.  If fewer than three sections are
        present the original context is returned unchanged.

        Args:
            context: Structured context string (from :meth:`structure_for_retrieval`).
            query:   The user's question or search query.

        Returns:
            Re-ordered context string.
        """
        sections = self._split_into_sections(context)
        if len(sections) <= 2:
            return context

        query_terms = set(re.sub(r"[^\w\s]", "", query.lower()).split())
        scored: list[tuple[float, dict]] = []
        for section in sections:
            text_lower = section["text"].lower()
            hits = sum(1 for t in query_terms if t in text_lower)
            score = hits / max(len(query_terms), 1)
            scored.append((score, section))

        scored.sort(key=lambda x: x[0], reverse=True)

        best_section = scored[0][1]
        others = [s for _, s in scored[1:]]

        # Place the best section ~35% of the way through the remaining list
        # so it ends up in the 20-60% golden zone of the final context.
        n_before = math.ceil(len(others) * 0.35)
        before = others[:n_before]
        after  = others[n_before:]
        reordered = before + [best_section] + after

        return self._reassemble_sections(reordered)

    def chunk_by_attention_zones(self, context: str) -> AttentionZones:
        """Split *context* into attention zones by character position.

        Args:
            context: Any context string.

        Returns:
            :class:`AttentionZones` with text portions for each zone.
        """
        n = len(context)
        if n == 0:
            return AttentionZones("", "", "", "", "")

        return AttentionZones(
            primacy_zone   = context[:int(n * _PRIMACY_END)],
            transition_12  = context[int(n * _PRIMACY_END): int(n * _OPTIMAL_START)],
            optimal_zone   = context[int(n * _OPTIMAL_START): int(n * _OPTIMAL_END)],
            transition_23  = context[int(n * _OPTIMAL_END): int(n * _RECENCY_START)],
            recency_zone   = context[int(n * _RECENCY_START):],
        )

    def deduplicate_context(self, documents: list[dict]) -> list[dict]:
        """Remove near-duplicate documents.

        Two documents are duplicates when their character 3-gram Jaccard
        similarity exceeds 0.90.  When a duplicate is detected the document
        with more characters is kept.

        Args:
            documents: List of document dicts with a ``text`` key.

        Returns:
            Deduplicated list, preserving the original order of unique docs.
        """
        unique: list[dict] = []
        for doc in documents:
            duplicate_index = -1
            for j, u in enumerate(unique):
                if self._jaccard_similarity(doc["text"], u["text"]) > 0.90:
                    duplicate_index = j
                    break
            if duplicate_index == -1:
                unique.append(doc)
            elif len(doc["text"]) > len(unique[duplicate_index]["text"]):
                unique[duplicate_index] = doc
        return unique

    def estimate_attention_score(self, context: str, fact: str) -> float:
        """Estimate how likely the model is to find and use *fact* in *context*.

        Scoring components:

        * **Position** (0.40–1.00): facts in the 20–60% zone score highest.
        * **Repetition** bonus: +0.05 per additional occurrence, up to +0.15.
        * **Structure** bonus: +0.10 if the fact appears near a heading or
          bullet point.

        Args:
            context: The full context string.
            fact:    The exact text of the fact to locate.

        Returns:
            Float in [0, 1]; higher means more likely to be used by the model.
        """
        if not fact or fact not in context:
            return 0.0

        n = len(context)
        if n == 0:
            return 0.0

        occurrences = [m.start() for m in re.finditer(re.escape(fact), context)]
        if not occurrences:
            return 0.0

        pos = occurrences[0] / n
        position_score = self._position_score(pos)

        rep_bonus = min(0.15, (len(occurrences) - 1) * 0.05)

        structure_bonus = 0.0
        for occ in occurrences:
            snippet = context[max(0, occ - 120): occ]
            if re.search(r"(^|\n)(#{1,3}\s|\*\s|-\s|\d+\.\s)", snippet):
                structure_bonus = 0.10
                break

        return min(1.0, position_score + rep_bonus + structure_bonus)

    def optimize_for_query(self, context: str, query: str) -> str:
        """Full optimisation pipeline.

        Steps:

        1. Split the context into sections (if structured).
        2. Score sections by lexical relevance to *query*; exclude score-0 sections
           only when other sections scored higher.
        3. Deduplicate near-identical sections.
        4. Re-structure with a table of contents and clear section markers.
        5. Place the highest-scoring section in the optimal attention zone.

        Args:
            context: Raw or structured context string.
            query:   The user's question.

        Returns:
            Optimised, structured context string.
        """
        sections = self._split_into_sections(context)
        if not sections:
            sections = [{"text": context, "metadata": {"source": "context"}}]

        query_terms = set(re.sub(r"[^\w\s]", "", query.lower()).split())
        scored: list[tuple[float, dict]] = []
        for sec in sections:
            text_lower = sec["text"].lower()
            score = sum(1 for t in query_terms if t in text_lower) / max(len(query_terms), 1)
            scored.append((score, sec))

        # Keep all sections that have any relevance, or all if none matched
        relevant = [s for _, s in scored if _ > 0]
        if not relevant:
            relevant = sections

        deduplicated = self.deduplicate_context(relevant)
        structured   = self.structure_for_retrieval(deduplicated)
        return self.prioritize_information(structured, query)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _position_score(self, relative_pos: float) -> float:
        """Map a relative position [0, 1] to an attention quality score.

        Quadratic curve centred at 0.40 (centre of the optimal zone),
        with score 1.0 at the peak and ~0.40 at the boundaries.
        """
        centre = 0.40
        width  = 0.30
        raw = 1.0 - ((relative_pos - centre) / width) ** 2
        return max(0.40, min(1.0, raw))

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """Character 3-gram Jaccard similarity between strings *a* and *b*."""
        def trigrams(s: str) -> set[str]:
            return {s[i: i + 3] for i in range(len(s) - 2)}

        ta, tb = trigrams(a), trigrams(b)
        if not ta and not tb:
            return 1.0
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    @staticmethod
    def _split_into_sections(context: str) -> list[dict]:
        """Split a structured context (from :meth:`structure_for_retrieval`)
        back into component document dicts.

        Falls back to a single-section list if no ``## [N]`` markers exist.
        """
        pattern = re.compile(r"## \[(\d+)\] (.+?)(?=\n## \[\d+\]|\Z)", re.DOTALL)
        sections: list[dict] = []
        for match in pattern.finditer(context):
            title = match.group(2).split("\n")[0].strip()
            # Strip the heading line and the [End Section N] footer
            text = re.sub(r"^## \[\d+\].*\n", "", match.group(0)).strip()
            text = re.sub(r"\[End Section \d+\]\s*", "", text).strip()
            sections.append({
                "text": text,
                "metadata": {
                    "source": title,
                    "section_index": int(match.group(1)),
                },
            })
        return sections

    @staticmethod
    def _reassemble_sections(sections: list[dict]) -> str:
        """Reassemble section dicts into a structured context string with
        updated sequential section numbers."""
        parts: list[str] = ["## Context Overview\n"]
        for i, sec in enumerate(sections, start=1):
            source = sec.get("metadata", {}).get("source", f"Section {i}")
            parts.append(f"- Section {i}: {source}")
        parts.append("\n---\n")
        for i, sec in enumerate(sections, start=1):
            source = sec.get("metadata", {}).get("source", f"Section {i}")
            text   = sec.get("text", "")
            parts.append(f"## [{i}] {source}\n")
            parts.append(text)
            parts.append(f"\n[End Section {i}]\n")
            parts.append("---\n")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Context Optimizer Demo ===\n")

    topics = [
        ("Shipping Policy",       "Orders ship within 2 business days via standard mail."),
        ("Return Policy",         "Items may be returned within 30 days for a full refund."),
        ("Payment Methods",       "We accept Visa, Mastercard, PayPal, and Apple Pay."),
        ("Account Creation",      "Create an account to track orders and manage preferences."),
        ("Password Reset",        "Use the 'Forgot Password' link on the login page."),
        ("Product Warranty",      "All electronics carry a 1-year manufacturer warranty."),
        ("Gift Cards",            "Gift cards are available in $10–$500 denominations."),
        ("Loyalty Program",       "Earn 1 point per dollar; 100 points = $1 discount."),
        ("Store Locations",       "We have 50+ retail locations across the United States."),
        ("Customer Support",      "Reach us at support@example.com or 1-800-555-0100."),
        ("Order Tracking",        "Use your order number at track.example.com."),
        ("Bulk Orders",           "Contact sales@example.com for orders above $5,000."),
        ("Newsletter",            "Subscribe for 10% off your first order."),
        ("Privacy Policy",        "We do not sell your personal data to third parties."),
        ("Cookie Policy",         "We use cookies to improve your browsing experience."),
        ("Accessibility",         "Our website meets WCAG 2.1 AA standards."),
        ("International Shipping","We ship to 35 countries; duties are the buyer's responsibility."),
        ("Packaging",             "All orders ship in eco-friendly, recyclable packaging."),
        ("App Download",          "Download our iOS or Android app for mobile-exclusive deals."),
        ("Flash Sales",           "Flash sales happen every Friday at noon — check the homepage."),
    ]

    documents = [
        {"text": f"{title}: {detail}", "metadata": {"source": title}}
        for title, detail in topics
    ]

    # Plant the needle at position 0 (primacy zone — worst possible position)
    needle = "Returns are free; use the pre-paid label included in your package."
    documents[0]["text"] = needle
    documents[0]["metadata"]["source"] = "Return Policy (NEEDLE)"

    query = "Can I return a product and is there a return label?"
    optimizer = ContextOptimizer()

    raw_context = optimizer.structure_for_retrieval(documents)
    n = len(raw_context)

    needle_pos_before = raw_context.find(needle)
    pct_before = needle_pos_before / n * 100 if needle_pos_before >= 0 else -1.0

    print(f"Needle position BEFORE optimization: {pct_before:.1f}% into context")
    score_before = optimizer.estimate_attention_score(raw_context, needle)
    print(f"Estimated attention score BEFORE   : {score_before:.2f}")

    optimized = optimizer.optimize_for_query(raw_context, query)
    n_opt = len(optimized)

    needle_pos_after = optimized.find(needle)
    pct_after = needle_pos_after / n_opt * 100 if needle_pos_after >= 0 else -1.0

    print(f"\nNeedle position AFTER optimization : {pct_after:.1f}% into context")
    score_after = optimizer.estimate_attention_score(optimized, needle)
    print(f"Estimated attention score AFTER    : {score_after:.2f}")

    # Deduplication demo
    dup_docs = [
        {"text": "The quick brown fox jumps over the lazy dog", "metadata": {"source": "A"}},
        {"text": "The quick brown fox jumps over the lazy dog.", "metadata": {"source": "B"}},
        {"text": "Completely different content about AI agents.", "metadata": {"source": "C"}},
    ]
    deduped = optimizer.deduplicate_context(dup_docs)
    print(f"\nDeduplication: {len(dup_docs)} docs → {len(deduped)} unique docs")

    # Attention zone breakdown
    zones = optimizer.chunk_by_attention_zones(raw_context)
    print("\nAttention zones (characters):")
    print(f"  primacy  (0–10%)   : {len(zones.primacy_zone):5,d} chars")
    print(f"  optimal  (20–60%)  : {len(zones.optimal_zone):5,d} chars  ← golden zone")
    print(f"  recency  (80–100%) : {len(zones.recency_zone):5,d} chars")
