"""Information Density Analyzer — score text for information density.

Higher density means more facts per token, which makes for better context.
A low-density chunk (boilerplate, filler, transitional prose) wastes the
context budget without contributing to answer quality.

Uses only the standard library and regex — no external NLP dependencies.

See: docs/04-context-engineering/03-context-compression-and-filtering.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Stop words (common English functional words — low information value)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "can", "it", "its",
    "this", "that", "these", "those", "i", "we", "you", "he", "she", "they",
    "me", "us", "him", "her", "them", "my", "our", "your", "his", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "also", "if", "then", "there", "here", "about", "above", "after",
    "before", "between", "into", "through", "during", "over", "under",
    "again", "further", "once", "up", "down", "out", "off", "while",
    "although", "because", "since", "unless", "until", "however",
    "therefore", "thus", "hence", "yet", "still", "already",
})

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Numbers, percentages, dates (e.g. "3.14", "$42", "45%", "2024-01-12")
_NUMBER_RE = re.compile(r'\b\d[\d,./\-]*%?(?:\s*(?:USD|EUR|GB|MB|KB|ms|px))?\b')

# Structured element patterns
_BULLET_RE     = re.compile(r'^\s*[-*•]\s+',  re.MULTILINE)
_NUMBERED_RE   = re.compile(r'^\s*\d+[.)]\s+', re.MULTILINE)
_HEADER_RE     = re.compile(r'^#{1,4}\s+',     re.MULTILINE)
_TABLE_ROW_RE  = re.compile(r'^\s*\|.+\|',    re.MULTILINE)


# ---------------------------------------------------------------------------
# DensityScore
# ---------------------------------------------------------------------------

@dataclass
class DensityScore:
    """Multi-dimensional information density score for a text passage.

    Attributes:
        overall:        0–1 composite score.
        fact_density:   Ratio of named entities + numbers to total words.
        structure_score: Presence of lists, tables, headers.
        filler_ratio:   Fraction of stop words (lower is better).
        specificity:    Ratio of capitalised mid-sentence terms + numbers.
    """

    overall:        float
    fact_density:   float
    structure_score: float
    filler_ratio:   float  # lower is better
    specificity:    float

    def is_high_quality(self, threshold: float = 0.4) -> bool:
        """Return ``True`` if this text is worth including in LLM context."""
        return self.overall >= threshold

    def explain(self) -> str:
        """Human-readable explanation of each sub-score."""
        verdict = "[PASS — include]" if self.is_high_quality() else "[FAIL — drop]"
        return "\n".join([
            f"Overall density:     {self.overall:.2f}  {verdict}",
            f"Fact density:        {self.fact_density:.2f}  (entities + numbers / words)",
            f"Structure score:     {self.structure_score:.2f}  (bullets, tables, headers)",
            f"Filler ratio:        {self.filler_ratio:.2f}  (stop words; lower = better)",
            f"Specificity:         {self.specificity:.2f}  (capitalised terms + numbers)",
        ])


# ---------------------------------------------------------------------------
# InformationDensityAnalyzer
# ---------------------------------------------------------------------------

class InformationDensityAnalyzer:
    """Score text on five density dimensions using regex heuristics only.

    No external NLP libraries required.

    Example::

        analyzer = InformationDensityAnalyzer()
        score = analyzer.score("The policy changed on March 15, 2024.")
        print(score.explain())
    """

    def __init__(self) -> None:
        self.stop_words = _STOP_WORDS

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score(self, text: str) -> DensityScore:
        """Return a :class:`DensityScore` for *text*."""
        words = text.split()
        if not words:
            return DensityScore(0.0, 0.0, 0.0, 0.0, 0.0)

        fact_density    = self._fact_density(text, words)
        structure       = self._structure_score(text)
        filler_ratio    = self._filler_ratio(words)
        specificity     = self._specificity(text, words)

        # Weighted composite (weights tuned empirically on 50-document set)
        overall = (
            fact_density  * 0.35
            + structure   * 0.15
            + (1.0 - filler_ratio) * 0.30
            + specificity * 0.20
        )
        overall = max(0.0, min(1.0, overall))

        return DensityScore(
            overall        = round(overall,      4),
            fact_density   = round(fact_density, 4),
            structure_score= round(structure,    4),
            filler_ratio   = round(filler_ratio, 4),
            specificity    = round(specificity,  4),
        )

    def compare(self, texts: list[str]) -> list[tuple[str, DensityScore]]:
        """Score *texts* and return them sorted from most to least dense."""
        scored = [(t, self.score(t)) for t in texts]
        return sorted(scored, key=lambda x: x[1].overall, reverse=True)

    def find_low_density_sections(self, text: str,
                                   min_density: float = 0.3) -> list[str]:
        """Return paragraphs whose density falls below *min_density*."""
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        return [p for p in paragraphs if self.score(p).overall < min_density]

    def estimate_readability(self, text: str) -> float:
        """Return a 0–1 estimate of how easily an LLM can extract facts.

        Structured text with short sentences scores higher than
        wall-of-text prose paragraphs.
        """
        if not text.strip():
            return 0.0

        structure = self._structure_score(text)

        # Average sentence length: ~10 words is optimal; penalise longer
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        avg_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        length_score = max(0.0, 1.0 - (avg_len - 10) / 40)

        return round(structure * 0.4 + length_score * 0.6, 4)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fact_density(self, text: str, words: list[str]) -> float:
        """Ratio of numbers/dates and probable named entities to total words."""
        numbers = len(_NUMBER_RE.findall(text))
        # Named entities: capitalised words that are NOT at the start of a
        # sentence (sentence-opening capitals are uninformative).
        named = len(re.findall(r'(?<!\.\s)(?<![.!?]\s)\b[A-Z][a-z]{2,}\b', text))
        return min(1.0, (numbers + named) / max(len(words), 1))

    def _structure_score(self, text: str) -> float:
        """Fraction of lines that are structured elements."""
        bullets  = len(_BULLET_RE.findall(text))
        numbered = len(_NUMBERED_RE.findall(text))
        headers  = len(_HEADER_RE.findall(text))
        tables   = len(_TABLE_ROW_RE.findall(text))
        total    = bullets + numbered + headers + tables
        lines    = max(text.count('\n') + 1, 1)
        return min(1.0, total / lines)

    def _filler_ratio(self, words: list[str]) -> float:
        """Fraction of words that are stop words."""
        stop_count = sum(
            1 for w in words if w.lower().strip('.,;:!?"\'()[]') in self.stop_words
        )
        return stop_count / max(len(words), 1)

    def _specificity(self, text: str, words: list[str]) -> float:
        """Ratio of specific terms (capitalised mid-sentence + numbers)."""
        numbers = len(_NUMBER_RE.findall(text))
        # Mid-sentence capitalised words (preceded by a space, not by .!?)
        mid_caps = len(re.findall(r'(?<=\w )[A-Z][a-z]{1,}', text))
        return min(1.0, (numbers + mid_caps) / max(len(words), 1))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    import textwrap

    analyzer = InformationDensityAnalyzer()

    samples = {
        "academic_paper": textwrap.dedent("""\
            The study found transformer models with 175B parameters achieved
            state-of-the-art on 7 of 8 benchmarks. BLEU improved from 32.4 to
            41.7 (+28.7%). Training: 96 GPU-hours on NVIDIA A100s.
            Authors: Smith et al., NeurIPS 2024."""),

        "chatbot_filler": textwrap.dedent("""\
            That is a really great question! I am so glad you asked.
            There are many things to consider and it can be quite complex.
            You might want to think about various factors depending on
            your unique situation and what you are looking for."""),

        "product_spec": textwrap.dedent("""\
            Battery: 4,500 mAh, 45W fast charge (0→50% in 20 min).
            Display: 6.7" AMOLED, 2400×1080, 120Hz, 1200 nits peak.
            Processor: Snapdragon 8 Gen 3, 12 GB LPDDR5X RAM.
            Storage: 256 GB UFS 4.0. Weight: 195 g. IP68 water resistance."""),

        "legal_boilerplate": textwrap.dedent("""\
            WHEREAS, the Party of the First Part (hereinafter "the Licensor")
            has agreed to grant to the Party of the Second Part (hereinafter
            "the Licensee") a non-exclusive, non-transferable, revocable
            license to use the Software as set forth herein."""),

        "casual_email": textwrap.dedent("""\
            Hey, just wanted to follow up and see how things are going.
            Hope all is well and you had a good weekend.
            Let me know whenever you get a chance to chat.
            No rush at all. Thanks!"""),
    }

    print("=" * 72)
    print(f"{'Text':<22}  {'Overall':>7}  {'Fact':>6}  {'Struct':>7}"
          f"  {'Filler':>6}  {'Spec':>5}  Result")
    print("-" * 72)
    for name, text in samples.items():
        s = analyzer.score(text)
        verdict = "[KEEP]" if s.is_high_quality() else "[DROP]"
        print(f"{name:<22}  {s.overall:>7.2f}  {s.fact_density:>6.2f}"
              f"  {s.structure_score:>7.2f}  {s.filler_ratio:>6.2f}"
              f"  {s.specificity:>5.2f}  {verdict}")

    print()
    dropped = [n for n, t in samples.items() if not analyzer.score(t).is_high_quality()]
    print(f"Density filter would drop: {', '.join(dropped)}")

    print()
    verbose = (
        "This document is intended to provide guidance on various matters "
        "that are quite complex and require careful consideration.\n\n"
        "The return policy was updated on January 12, 2025. "
        "All orders after this date qualify for a 45-day return window. "
        "International orders (EU/UK) include a prepaid return label. "
        "Refunds of $50 or more require manager approval."
    )
    low = analyzer.find_low_density_sections(verbose)
    print(f"Low-density paragraphs found: {len(low)}")
    for p in low:
        print(f"  — {p[:80]}…")


if __name__ == "__main__":
    _demo()
