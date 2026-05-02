"""Branch manager — parallel conversation contexts.

Allows an agent to explore hypothetical paths or handle sub-tasks without
polluting the main conversation history. Each branch is an independent
MemoryManager instance.

Typical use case: "What would happen if I invested in X instead?" — create
a branch, explore, summarize, then inject only the summary back to main.

See: docs/03-memory-and-retrieval/01-short-term-memory.md
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional, Type

from memory_manager import MemoryManager
from conversation_summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)


class BranchManager:
    """Manages parallel conversation branches for hypothetical exploration.

    Each branch is a separate MemoryManager with its own message history.
    Branches can be summarized and their key context injected into other
    branches.

    Args:
        system_prompt:         Shared system prompt injected into each branch.
        memory_manager_class:  MemoryManager class to instantiate per branch
                               (injectable for testing).
        model:                 LLM model for token counting in branches.
        max_tokens:            Max token budget per branch.
        client:                Optional OpenAI client (shared across branches).
    """

    def __init__(
        self,
        system_prompt: str = "",
        memory_manager_class: Type[MemoryManager] = MemoryManager,
        model: str = "gpt-4o",
        max_tokens: int = 100_000,
        client: Optional[object] = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._manager_class = memory_manager_class
        self._model = model
        self._max_tokens = max_tokens
        self._client = client
        self._branches: dict[str, MemoryManager] = {}
        self._summarizer = ConversationSummarizer(client=client)

    # ------------------------------------------------------------------
    # Branch lifecycle
    # ------------------------------------------------------------------

    def create_branch(
        self,
        name: str,
        user_query: str = "",
        context_from: Optional[str] = None,
    ) -> str:
        """Create a new branch and return its ID.

        Args:
            name:         Human-readable name for this branch.
            user_query:   Opening user message for the branch (optional).
            context_from: Branch ID to copy context from via summary injection.

        Returns:
            Unique branch ID (used for all future operations on this branch).

        Raises:
            KeyError: If context_from points to an unknown branch.
        """
        branch_id = f"{name}-{uuid.uuid4().hex[:8]}"

        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system_prompt": self._system_prompt,
        }
        if self._client:
            kwargs["client"] = self._client

        mem = self._manager_class(**kwargs)
        self._branches[branch_id] = mem

        # Inject context summary from a parent branch
        if context_from:
            parent = self.get_branch(context_from)  # raises KeyError if missing
            parent_msgs = parent.messages[1:]  # skip system prompt
            if parent_msgs:
                summary = self._summarizer.summarize(parent_msgs)
                if summary:
                    mem.add_message(
                        {
                            "role": "user",
                            "content": f"[Context from branch '{context_from}': {summary}]",
                        }
                    )

        if user_query:
            mem.add_user_message(user_query)

        logger.info("Created branch '%s' (id=%s)", name, branch_id)
        return branch_id

    def get_branch(self, branch_id: str) -> MemoryManager:
        """Return the MemoryManager for a branch.

        Raises:
            KeyError: If branch_id is not known.
        """
        if branch_id not in self._branches:
            raise KeyError(f"Unknown branch: '{branch_id}'")
        return self._branches[branch_id]

    def add_to_branch(self, branch_id: str, message: dict) -> None:
        """Append a raw message dict to a branch.

        Raises:
            KeyError: If branch_id is unknown.
        """
        self.get_branch(branch_id).add_message(message)

    def summarize_branch(self, branch_id: str) -> str:
        """Return a text summary of a branch's conversation.

        Args:
            branch_id: Target branch ID.

        Returns:
            Summary string (empty string if the branch has no messages).

        Raises:
            KeyError: If branch_id is unknown.
        """
        mem = self.get_branch(branch_id)
        msgs = mem.messages[1:]  # skip system prompt
        if not msgs:
            return ""
        return self._summarizer.summarize(msgs)

    def close_branch(self, branch_id: str) -> str:
        """Summarize and remove a branch, returning its summary.

        Args:
            branch_id: Branch to close.

        Returns:
            Summary of the closed branch.

        Raises:
            KeyError: If branch_id is unknown.
        """
        summary = self.summarize_branch(branch_id)
        del self._branches[branch_id]
        logger.info("Closed branch '%s'", branch_id)
        return summary

    def get_active_branches(self) -> list[str]:
        """Return the IDs of all currently open branches."""
        return list(self._branches.keys())

    def merge_context(
        self,
        target_branch: str,
        source_branches: list[str],
    ) -> None:
        """Inject summaries of source branches into the target branch.

        Each source branch's summary is added as a user message in the
        target branch, prefixed with the branch ID so the LLM knows where
        the information came from.

        Args:
            target_branch:   Branch to receive context.
            source_branches: Branch IDs whose summaries to inject.

        Raises:
            KeyError: If any branch ID is unknown.
        """
        target = self.get_branch(target_branch)
        for source_id in source_branches:
            summary = self.summarize_branch(source_id)
            if summary:
                target.add_message(
                    {
                        "role": "user",
                        "content": f"[Summary from branch '{source_id}': {summary}]",
                    }
                )
                logger.info(
                    "Merged context from '%s' into '%s'", source_id, target_branch
                )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Use a mock summarizer so the demo works without an API key
    class MockSummarizer(ConversationSummarizer):
        def summarize(self, messages):
            return f"[Summary of {len(messages)} messages]"

    bm = BranchManager(system_prompt="You are a research assistant.")
    bm._summarizer = MockSummarizer()

    # Create a main branch
    main_id = bm.create_branch("main", user_query="Research the electric vehicle market.")
    bm.add_to_branch(main_id, {"role": "assistant", "content": "The EV market is growing rapidly."})

    # Explore a sub-branch without polluting main
    sub_id = bm.create_branch("ev-competitors", user_query="Compare Tesla vs Rivian.", context_from=main_id)
    bm.add_to_branch(sub_id, {"role": "assistant", "content": "Tesla leads in volume; Rivian focuses on adventure trucks."})

    print("Active branches:", bm.get_active_branches())

    # Merge findings back to main
    bm.merge_context(main_id, [sub_id])
    main_mem = bm.get_branch(main_id)
    print(f"Main branch messages after merge: {len(main_mem.messages)}")

    # Close the sub-branch
    sub_summary = bm.close_branch(sub_id)
    print(f"Sub-branch summary: {sub_summary!r}")
    print("Active branches:", bm.get_active_branches())


if __name__ == "__main__":
    main()
