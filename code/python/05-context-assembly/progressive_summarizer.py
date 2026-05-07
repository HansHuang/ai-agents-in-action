"""Progressive Summarizer — incremental, layered conversation summarization.

Older turns are compressed more aggressively than recent turns.  The last
few turns are always preserved verbatim so the agent can follow the
immediate flow of conversation.  Deeper layers hold increasingly compressed
summaries of older turns.

Layers::

    Layer 0 (verbatim):  Last ``verbatim_turns`` turns, exact wording
    Layer 1 (detailed):  Turns outside verbatim window, summarised in detail
    Layer 2 (compressed): Layer-1 overflow, key facts only (~50 % of L1)
    Layer 3 (archival):  Layer-2 overflow, high-compression essence (~50 % of L2)

When a layer's token count exceeds ``layer_token_limit``, its content
cascades into the next deeper layer and the shallower layer is cleared.

LLM calls are made via the OpenAI client when available; a deterministic
fallback (keyword extraction) is used automatically during testing or when
the API key is absent.

See: docs/04-context-engineering/04-multi-turn-context-management.md
"""

from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass, field

from context_budget import count_tokens


# ---------------------------------------------------------------------------
# LLM helpers (optional dependency)
# ---------------------------------------------------------------------------

def _llm_available() -> bool:
    """Return True when the openai package and API key are present."""
    try:
        import openai  # noqa: F401
        return bool(os.environ.get("OPENAI_API_KEY"))
    except ImportError:
        return False


def _llm_summarize(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call the LLM to produce a summary.  Falls back gracefully."""
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Deterministic fallback summarizer (keyword extraction)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "i me my we our you your he she it its they them the a an and or "
    "but in on at to for of with is was are were be been have has had "
    "do does did will would could should may might shall this that these "
    "those not no so just very really also then when where how what who "
    "user assistant".split()
)


def _extract_key_sentences(text: str, max_sentences: int = 6) -> str:
    """Return the most information-dense sentences from *text*.

    Scores each sentence by the density of non-stop, longer words.
    Preserves the original sentence order for readability.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return text

    def _score(s: str) -> float:
        words = re.findall(r"\b\w{4,}\b", s.lower())
        content = [w for w in words if w not in _STOP_WORDS]
        return len(content) / max(len(words), 1)

    scored = sorted(enumerate(sentences), key=lambda t: _score(t[1]), reverse=True)
    keep_indices = sorted(idx for idx, _ in scored[:max_sentences])
    return " ".join(sentences[i] for i in keep_indices)


def _fallback_update_summary(existing: str, new_turn: str) -> str:
    """Combine existing summary with a new turn without an LLM call."""
    combined = f"{existing}\n{new_turn}".strip()
    return _extract_key_sentences(combined, max_sentences=8)


def _fallback_compress(content: str) -> str:
    """Compress a layer to ~50 % without an LLM call."""
    return _extract_key_sentences(content, max_sentences=5)


# ---------------------------------------------------------------------------
# Summarization prompts
# ---------------------------------------------------------------------------

_LAYER_UPDATE_PROMPT = """\
Update this conversation summary with new information.
Preserve: goals, decisions made, specific data (numbers, dates, names),
user preferences, agent recommendations, pending tasks.

Discard: small talk, repeated information, exact wording of resolved questions.

Existing summary: {existing}
New information: {new_turn}

Updated summary (keep approximately the same length):"""

_LAYER_COMPRESSION_PROMPT = """\
Compress this detailed conversation summary into a shorter version.
Keep only: the main goal, key decisions, critical facts, and unresolved items.
The compressed version should be about half the length.

Detailed summary: {layer_content}

Compressed summary:"""


# ---------------------------------------------------------------------------
# ProgressiveSummarizer
# ---------------------------------------------------------------------------


class ProgressiveSummarizer:
    """Summarize conversations incrementally across multiple layers.

    Layers:

    - **Layer 0 (verbatim)**: Last ``verbatim_turns`` turns, preserved exactly.
    - **Layer 1 (detailed)**: Turns 6–15, summarised with moderate detail.
    - **Layer 2 (compressed)**: Turns 16–30, key facts only.
    - **Layer 3 (archival)**: Turns 31+, highly compressed essence.

    Example::

        s = ProgressiveSummarizer(verbatim_turns=5, layer_size=10)
        for user_msg, assistant_msg in conversation:
            s.add_turn(user_msg, assistant_msg)
        context = s.get_context()

    Args:
        verbatim_turns: Number of most-recent turns to keep verbatim.
        layer_size:     Approximate number of turns each summary layer covers
                        before cascading.  Controls the ``layer_token_limit``.
        model:          OpenAI model used for LLM-based summarisation.
        layer_token_limit: Maximum token count per summary layer before
                           cascading to the next deeper layer.
    """

    NUM_LAYERS: int = 3  # Layers 1, 2, 3 (layer 0 = verbatim)

    def __init__(
        self,
        verbatim_turns: int = 5,
        layer_size: int = 10,
        model: str = "gpt-4o-mini",
        layer_token_limit: int = 1500,
    ) -> None:
        self.verbatim_turns = verbatim_turns
        self.layer_size = layer_size
        self.model = model
        self.layer_token_limit = layer_token_limit
        self.use_llm = _llm_available()

        # Layer 0 — verbatim ring buffer
        self.verbatim: list[tuple[str, str]] = []  # (user_msg, assistant_msg)

        # Layers 1–3 — progressively compressed summaries
        self.layers: list[str] = [""] * self.NUM_LAYERS

        # Stats
        self._total_turns: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Add a turn and rebalance layers if needed.

        Older information cascades into deeper (more compressed) layers.

        Args:
            user_msg:      The user's message.
            assistant_msg: The agent's response.
        """
        self._total_turns += 1
        self.verbatim.append((user_msg, assistant_msg))

        if len(self.verbatim) > self.verbatim_turns:
            oldest = self.verbatim.pop(0)
            turn_text = f"User: {oldest[0]}\nAssistant: {oldest[1]}"
            self._incorporate_into_layer(0, turn_text)

    def get_context(self) -> str:
        """Return the full context string for prompt injection.

        Orders from most-compressed (deepest layer) to most-detailed
        (verbatim), so the model sees older context first and recent
        context last.
        """
        parts: list[str] = []

        for layer_idx in range(self.NUM_LAYERS - 1, -1, -1):  # 2, 1, 0
            content = self.layers[layer_idx].strip()
            if content:
                label = ["Early conversation", "Earlier conversation", "Recent conversation"][layer_idx]
                parts.append(f"[{label}:\n{content}]")

        if self.verbatim:
            lines: list[str] = []
            for user_msg, assistant_msg in self.verbatim:
                lines.append(f"User: {user_msg}")
                lines.append(f"Assistant: {assistant_msg}")
            parts.append("[Most recent turns:\n" + "\n".join(lines) + "]")

        return "\n\n".join(parts)

    def get_stats(self) -> dict:
        """Return statistics about the summarizer's current state.

        Returns:
            A dict with keys ``total_turns_processed``, ``verbatim_turns``,
            ``layer_1_tokens``, ``layer_2_tokens``, ``layer_3_tokens``, and
            ``total_context_tokens``.
        """
        layer_tokens = [count_tokens(layer) for layer in self.layers]
        verbatim_text = " ".join(
            f"{u} {a}" for u, a in self.verbatim
        )
        verbatim_tokens = count_tokens(verbatim_text)
        return {
            "total_turns_processed": self._total_turns,
            "verbatim_turns":        len(self.verbatim),
            "layer_1_tokens":        layer_tokens[0],
            "layer_2_tokens":        layer_tokens[1],
            "layer_3_tokens":        layer_tokens[2],
            "total_context_tokens":  sum(layer_tokens) + verbatim_tokens,
        }

    # ------------------------------------------------------------------
    # Internal layer management
    # ------------------------------------------------------------------

    def _incorporate_into_layer(self, layer_index: int, turn_text: str) -> None:
        """Add *turn_text* to ``layers[layer_index]``, cascading if needed."""
        if layer_index >= self.NUM_LAYERS:
            # Already at the deepest layer — just compress in-place
            self.layers[-1] = self._compress_content(
                self.layers[-1], turn_text, depth=self.NUM_LAYERS
            )
            return

        existing = self.layers[layer_index]
        self.layers[layer_index] = self._update_summary(existing, turn_text)

        # Check if the layer has overflowed
        if count_tokens(self.layers[layer_index]) > self.layer_token_limit:
            self._cascade_overflow(layer_index)

    def _cascade_overflow(self, from_layer: int) -> None:
        """Cascade *from_layer*'s content to the next deeper layer.

        The overflowing layer is cleared after its content is absorbed.

        Args:
            from_layer: Zero-based index into ``self.layers`` (0 = Layer 1).
        """
        to_layer = from_layer + 1
        if to_layer >= self.NUM_LAYERS:
            # At deepest layer — compress in-place and truncate
            self.layers[from_layer] = self._compress_content(
                self.layers[from_layer], "", depth=from_layer + 1
            )
            return

        # Merge current layer content into the next layer
        self.layers[to_layer] = self._compress_content(
            self.layers[to_layer],
            self.layers[from_layer],
            depth=to_layer + 1,
        )
        self.layers[from_layer] = ""

        # Recursively cascade if the target layer also overflowed
        if count_tokens(self.layers[to_layer]) > self.layer_token_limit:
            self._cascade_overflow(to_layer)

    def _update_summary(self, existing: str, new_turn: str) -> str:
        """Merge *new_turn* into *existing* summary."""
        if not existing:
            return new_turn

        if self.use_llm:
            try:
                prompt = _LAYER_UPDATE_PROMPT.format(
                    existing=existing or "(none)",
                    new_turn=new_turn,
                )
                return _llm_summarize(prompt, self.model)
            except Exception:
                pass  # Fall through to offline fallback

        return _fallback_update_summary(existing, new_turn)

    def _compress_content(
        self, older: str, newer: str, depth: int
    ) -> str:
        """Compress two pieces of content into one at *depth* compression."""
        combined = f"{older}\n{newer}".strip() if newer else older
        if not combined:
            return ""

        if self.use_llm:
            try:
                prompt = _LAYER_COMPRESSION_PROMPT.format(layer_content=combined)
                return _llm_summarize(prompt, self.model)
            except Exception:
                pass

        # Offline: extract key sentences — fewer at deeper layers
        max_sentences = max(3, 8 - depth * 2)
        return _extract_key_sentences(combined, max_sentences=max_sentences)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:  # pragma: no cover
    """Simulate a 40-turn conversation and demonstrate progressive summarization."""
    import random

    print("=" * 70)
    print("PROGRESSIVE SUMMARIZER DEMO")
    print("=" * 70)

    summarizer = ProgressiveSummarizer(verbatim_turns=5, layer_size=10)

    # Simulate 40 turns of conversation about booking a flight
    turns: list[tuple[str, str]] = [
        ("I need to book a round-trip flight to London.", "I'd be happy to help with that. When are you travelling?"),
        ("I'm flying June 15 and returning June 22.", "Got it. Do you prefer window or aisle seats?"),
        ("Window seat please. My budget is $800.", "Window it is. I'll search for flights within your $800 budget."),
        ("My name is Alice Johnson.", "Thank you Alice. One moment while I search…"),
        ("Can I use miles for an upgrade?", "Yes, you can use miles for a cabin upgrade if available."),
        ("How many miles would it cost?", "Typically 15,000–25,000 miles for Business class."),
        ("I have 30,000 miles available.", "Great — you have enough for an upgrade on most routes."),
        ("What airlines fly direct from JFK to LHR?", "British Airways, Virgin Atlantic, and American fly direct."),
        ("I prefer British Airways.", "Noted — searching BA flights for June 15."),
        ("What's the baggage allowance?", "BA allows one 23 kg checked bag in Economy."),
        ("Found BA 178 at $720. Shall I book it?", "Yes please, go ahead with BA 178."),
        ("Your seat 22A (window) is confirmed.", "Perfect! Can I get the confirmation email sent to alice@example.com?"),
        ("Confirmation sent to alice@example.com.", "Thank you. Do I need travel insurance?"),
        ("Travel insurance is optional but recommended.", "I'd like to add it. How much does it cost?"),
        ("Basic travel insurance is $45.", "Add it please."),
        ("Insurance added. Total: $765.", "Is my visa sorted for the UK?"),
        ("US citizens don't need a visa for UK stays under 6 months.", "Brilliant. What about the hotel?"),
        ("Do you want me to search hotels in London?", "Yes, near the British Museum please."),
        ("Found Kimpton Fitzroy at £150/night.", "That's a bit steep. Any alternatives?"),
        ("Premier Inn Euston from £89/night.", "Book the Premier Inn for June 15–22 please."),
        ("Hotel booked. Confirmation: HOT-4421.", "Wonderful! Do I need a rail pass?"),
        ("An Oyster card is cheaper for London transport.", "How do I get one?"),
        ("You can get an Oyster card at any Tube station.", "Can I pre-load it online?"),
        ("Yes, via the TfL website.", "What's the minimum top-up amount?"),
        ("Minimum top-up is £5.", "I'll load £40. What sights should I not miss?"),
        ("The British Museum, Tower of London, and Hyde Park.", "What about day trips?"),
        ("Oxford and Bath are popular day trips from London.", "How far is Bath?"),
        ("Bath is about 1.5 hours by train.", "Is there a direct train?"),
        ("Yes, GWR runs direct trains from Paddington.", "Perfect. I think I'm all set!"),
        ("Is there anything else you need help with?", "Just double-check — flight BA178, hotel Premier Inn, insurance included."),
        ("Everything confirmed: flight BA178 June 15, return June 22, seat 22A, Premier Inn Jun 15–22, travel insurance.", "Brilliant, thank you so much!"),
        ("You're all set, Alice. Have a wonderful trip!", "One last thing — can you recommend a good restaurant near the hotel?"),
        ("The Ivy Bloomsbury is excellent and close to Premier Inn.", "Is it expensive?"),
        ("Expect to spend around £40–£60 per person.", "I'll book a table for two."),
        ("Shall I make a reservation?", "Yes, for June 16 at 7 PM for 2 people."),
        ("Reservation at The Ivy Bloomsbury confirmed for June 16 at 7 PM.", "You're amazing, thank you!"),
        ("Anything else I can help with?", "No, that's everything. You've been incredibly helpful."),
        ("Safe travels, Alice!", "Thank you, I can't wait!"),
        ("Is there a dress code at The Ivy?", "Smart casual is the recommendation at The Ivy."),
        ("Perfect. Goodbye!", "Goodbye Alice, enjoy London!"),
    ]

    for turn_num, (user_msg, assistant_msg) in enumerate(turns, start=1):
        summarizer.add_turn(user_msg, assistant_msg)

        if turn_num in {10, 20, 30, 40}:
            stats = summarizer.get_stats()
            print(f"\n{'─'*60}")
            print(f"STATS AT TURN {turn_num}")
            print(f"{'─'*60}")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    print(f"\n{'='*60}")
    print("FULL CONTEXT AT TURN 40")
    print(f"{'='*60}")
    context = summarizer.get_context()
    # Show truncated view for readability
    lines = context.split("\n")
    for line in lines[:40]:
        print(line)
    if len(lines) > 40:
        print(f"  … ({len(lines) - 40} more lines) …")

    print(f"\n{'='*60}")
    print("VERBATIM TURNS VERIFICATION")
    print(f"{'='*60}")
    print("Last 5 turns (verbatim):")
    for user_msg, assistant_msg in summarizer.verbatim:
        print(f"  User: {user_msg[:60]}")
        print(f"  Asst: {assistant_msg[:60]}")
        print()

    # Verify verbatim turns are the last 5
    expected_last = turns[-5:]
    for i, ((exp_u, exp_a), (got_u, got_a)) in enumerate(zip(expected_last, summarizer.verbatim)):
        assert exp_u == got_u, f"Verbatim mismatch at position {i}: {exp_u!r} != {got_u!r}"
    print("✓ Verbatim turns are exactly the last 5 turns")

    print("\nDemo complete.")


if __name__ == "__main__":
    _demo()
