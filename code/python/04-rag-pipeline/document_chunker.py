"""Document chunking strategies for embedding pipelines.

Splits documents into smaller pieces suitable for embedding and vector search.
Supports fixed-size, semantic, hierarchical, and sentence-boundary chunking.

See: docs/03-memory-and-retrieval/02-embeddings-and-vectors.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import tiktoken

# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single embeddable piece of a document.

    Attributes:
        text: The chunk's text content.
        index: Zero-based position of this chunk in the original document.
        token_count: Number of tokens in *text*.
        metadata: Arbitrary key-value pairs attached by the caller (source,
            page number, section heading, etc.).
        parent_chunk: For hierarchical chunking — the larger section this
            chunk belongs to.
        children: For hierarchical chunking — the smaller detail chunks
            contained within this chunk.
    """

    text: str
    index: int
    token_count: int
    metadata: Optional[dict] = None
    parent_chunk: Optional["Chunk"] = None
    children: list["Chunk"] = field(default_factory=list)

    def __repr__(self) -> str:
        snippet = self.text[:60].replace("\n", "↵")
        return f"Chunk(index={self.index}, tokens={self.token_count}, text={snippet!r})"


# ---------------------------------------------------------------------------
# DocumentChunker
# ---------------------------------------------------------------------------


class DocumentChunker:
    """Split documents into chunks suitable for embedding.

    Args:
        chunk_size: Target chunk size in tokens.
        overlap: Number of tokens to overlap between adjacent chunks.
        strategy: One of ``"fixed"``, ``"semantic"``, ``"hierarchical"``,
            or ``"sentence"``.
        encoding_name: tiktoken encoding to use for token counting.
    """

    _SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: int = 50,
        strategy: str = "semantic",
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        self._enc = tiktoken.get_encoding(encoding_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> list[Chunk]:
        """Split *text* into chunks using the configured strategy.

        Strategies:
        - ``"fixed"``: Fixed token count with overlap.
        - ``"semantic"``: Split at paragraph/section/sentence boundaries.
        - ``"hierarchical"``: Maintain parent–child relationships.
        - ``"sentence"``: Split at sentence boundaries, group to chunk_size.

        Args:
            text: Raw document text to split.

        Returns:
            Ordered list of :class:`Chunk` objects.
        """
        dispatch = {
            "fixed": self.chunk_fixed,
            "semantic": self.chunk_semantic,
            "hierarchical": self.chunk_hierarchical,
            "sentence": self.chunk_sentence,
        }
        if self.strategy not in dispatch:
            raise ValueError(
                f"Unknown strategy {self.strategy!r}. "
                f"Choose from: {list(dispatch)}"
            )
        return dispatch[self.strategy](text)

    def chunk_fixed(self, text: str) -> list[Chunk]:
        """Fixed-size chunks with token overlap.

        Tokenizes the entire document and slides a window of *chunk_size*
        tokens, stepping by ``chunk_size - overlap`` each time.

        Args:
            text: Document text.

        Returns:
            List of :class:`Chunk` objects.
        """
        tokens = self._enc.encode(text)
        step = max(1, self.chunk_size - self.overlap)
        chunks: list[Chunk] = []
        idx = 0
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self._enc.decode(chunk_tokens)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=idx,
                    token_count=len(chunk_tokens),
                )
            )
            start += step
            idx += 1
        return chunks

    def chunk_semantic(self, text: str) -> list[Chunk]:
        """Split at natural boundaries: paragraphs, lines, then sentences.

        Priority order:
        1. Double-newline paragraph breaks.
        2. Single-newline line breaks.
        3. Sentence boundaries (as detected by :attr:`_SENTENCE_END`).
        4. Falls back to :meth:`chunk_fixed` for oversize segments.

        Args:
            text: Document text.

        Returns:
            List of :class:`Chunk` objects.
        """
        # Split by paragraphs first; fall back progressively.
        raw_segments = self._split_semantic(text)
        return self._segments_to_chunks(raw_segments)

    def chunk_hierarchical(self, text: str) -> list[Chunk]:
        """Create parent–child chunk relationships.

        Parents represent complete sections (split at headings or double
        newlines). Each parent is then split into child chunks using the
        semantic strategy. Child chunks hold a reference to their parent
        and the parent holds references to its children.

        On retrieval you might surface the matching child to find a precise
        passage, then return the parent's text for richer context.

        Args:
            text: Document text.

        Returns:
            Flat list of all chunks (both parents and children). Parents
            have ``chunk.children`` populated; children have
            ``chunk.parent_chunk`` set.
        """
        # Use headings (Markdown # / ##) or double-newlines as section breaks.
        section_pattern = re.compile(r"(?m)(?=^#{1,3}\s)|(?<=\n)\n(?=\S)")
        raw_sections = section_pattern.split(text)
        raw_sections = [s.strip() for s in raw_sections if s.strip()]

        all_chunks: list[Chunk] = []
        parent_idx = 0

        for section_text in raw_sections:
            parent_token_count = self._count_tokens(section_text)
            parent = Chunk(
                text=section_text,
                index=parent_idx,
                token_count=parent_token_count,
            )
            parent_idx += 1

            # Build child chunks from this section using semantic strategy.
            child_chunker = DocumentChunker(
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                strategy="semantic",
            )
            child_raw = child_chunker.chunk(section_text)
            for child in child_raw:
                child.index = parent_idx
                child.parent_chunk = parent
                parent.children.append(child)
                parent_idx += 1

            all_chunks.append(parent)
            all_chunks.extend(parent.children)

        return all_chunks

    def chunk_sentence(self, text: str) -> list[Chunk]:
        """Split at sentence boundaries and group into chunk_size buckets.

        Each chunk accumulates complete sentences until adding the next
        sentence would exceed *chunk_size* tokens.

        Args:
            text: Document text.

        Returns:
            List of :class:`Chunk` objects.
        """
        sentences = self._SENTENCE_END.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return self._segments_to_chunks(sentences)

    def chunk_with_metadata(self, text: str, metadata: dict) -> list[Chunk]:
        """Chunk text and attach *metadata* to every resulting chunk.

        Args:
            text: Document text.
            metadata: Key-value pairs to attach (e.g., ``{"source": "faq.pdf",
                "page": 3}``).

        Returns:
            List of :class:`Chunk` objects, each with ``chunk.metadata`` set.
        """
        chunks = self.chunk(text)
        for chunk in chunks:
            chunk.metadata = dict(metadata)
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _split_semantic(self, text: str) -> list[str]:
        """Return a flat list of the smallest natural segments in *text*."""
        segments: list[str] = []
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if self._count_tokens(para) <= self.chunk_size:
                segments.append(para)
            else:
                # Try single-line splits first.
                for line in para.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    if self._count_tokens(line) <= self.chunk_size:
                        segments.append(line)
                    else:
                        # Split long lines at sentence boundaries.
                        for sent in self._SENTENCE_END.split(line):
                            sent = sent.strip()
                            if sent:
                                segments.append(sent)
        return segments

    def _segments_to_chunks(self, segments: list[str]) -> list[Chunk]:
        """Greedily pack *segments* into chunks of at most *chunk_size* tokens.

        Consecutive segments are merged until the next segment would push the
        accumulated token count past *chunk_size*. Overlap is applied by
        re-including the tail tokens of the previous chunk at the start of
        the next.
        """
        chunks: list[Chunk] = []
        current_text = ""
        current_tokens: list[int] = []
        idx = 0

        def _flush() -> None:
            nonlocal current_text, current_tokens, idx
            if current_text:
                chunks.append(
                    Chunk(
                        text=current_text.strip(),
                        index=idx,
                        token_count=len(current_tokens),
                    )
                )
                idx += 1

        for seg in segments:
            seg_tokens = self._enc.encode(seg)
            if len(current_tokens) + len(seg_tokens) + 1 > self.chunk_size:
                _flush()
                # Seed the next chunk with the overlap tail of the previous one.
                if current_tokens and self.overlap > 0:
                    overlap_tokens = current_tokens[-self.overlap:]
                    current_text = self._enc.decode(overlap_tokens) + " " + seg
                    current_tokens = overlap_tokens + [32] + seg_tokens  # 32 = space
                else:
                    current_text = seg
                    current_tokens = seg_tokens
            else:
                joiner = " " if current_text else ""
                current_text = current_text + joiner + seg
                current_tokens = current_tokens + ([32] if current_text else []) + seg_tokens

        _flush()
        return chunks


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

_SAMPLE_DOCUMENT = """\
# Return Policy

## Damaged Items

If you receive a damaged or defective item, please contact our support team
within 30 days of delivery. We will arrange a free return and send a
replacement or issue a full refund — your choice.

To report a damaged item, go to the Orders page, select the item, and click
"Report a Problem." Attach a photo if possible. Our team typically responds
within 24 hours.

## Refund Timeline

Once we receive your returned item, refunds are processed within 3–5 business
days. You will receive a confirmation email when the refund is initiated. Funds
may take an additional 2–3 days to appear in your account depending on your bank.

For digital purchases, refunds are issued immediately upon approval.

## Exchanges

We do not currently support direct exchanges. To exchange an item, return the
original and place a new order. You will receive the refund before the new
charge appears.

## Contact Support

Our support team is available Monday through Friday, 9 AM – 6 PM Eastern Time.
You can reach us via the in-app chat, email at support@example.com, or by
calling 1-800-555-0199.
"""


def main() -> None:
    """Demonstrate the three chunking strategies on a sample support document."""
    print("=" * 60)
    print("Sample document word count:", len(_SAMPLE_DOCUMENT.split()))
    print("=" * 60)

    for strategy in ("fixed", "semantic", "hierarchical"):
        chunker = DocumentChunker(chunk_size=100, overlap=20, strategy=strategy)
        chunks = chunker.chunk(_SAMPLE_DOCUMENT)

        # Only leaf / non-parent chunks for stats (hierarchical includes both)
        leaf_chunks = [c for c in chunks if not c.children]
        sizes = [c.token_count for c in leaf_chunks]
        avg = sum(sizes) / len(sizes) if sizes else 0

        print(f"\n--- Strategy: {strategy.upper()} ---")
        print(f"  Total chunks: {len(chunks)}  (leaf: {len(leaf_chunks)})")
        print(f"  Avg leaf tokens: {avg:.1f}  min: {min(sizes)}  max: {max(sizes)}")
        for i, chunk in enumerate(chunks[:3]):
            label = "(parent)" if chunk.children else ""
            print(f"  [{i}]{label} {chunk!r}")
        if len(chunks) > 3:
            print(f"  … and {len(chunks) - 3} more")

    # --- Show overlap in action ---
    print("\n--- Overlap demonstration (fixed, chunk_size=40, overlap=10) ---")
    chunker = DocumentChunker(chunk_size=40, overlap=10, strategy="fixed")
    chunks = chunker.chunk(_SAMPLE_DOCUMENT)
    if len(chunks) >= 2:
        tail = chunks[0].text[-60:]
        head = chunks[1].text[:60:]
        print(f"  End of chunk 0: …{tail!r}")
        print(f"  Start of chunk 1: {head!r}…")

    # --- Hierarchical parent-child ---
    print("\n--- Hierarchical: parent-child ---")
    chunker = DocumentChunker(chunk_size=100, overlap=20, strategy="hierarchical")
    chunks = chunker.chunk(_SAMPLE_DOCUMENT)
    parents = [c for c in chunks if c.children]
    if parents:
        p = parents[0]
        print(f"  Parent ({p.token_count} tokens): {p.text[:80]!r}…")
        for child in p.children[:2]:
            print(f"    Child ({child.token_count} tokens): {child.text[:60]!r}…")

    # --- Metadata attachment ---
    print("\n--- Metadata ---")
    chunker = DocumentChunker(chunk_size=100, overlap=20, strategy="semantic")
    chunks = chunker.chunk_with_metadata(
        _SAMPLE_DOCUMENT,
        {"source": "return_policy.md", "version": "2024-Q1"},
    )
    print(f"  First chunk metadata: {chunks[0].metadata}")


if __name__ == "__main__":
    main()
