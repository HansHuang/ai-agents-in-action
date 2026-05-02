# Short-Term Memory: Managing the Message List

## What You'll Learn
- Why the message list is the agent's working memory — and its biggest liability
- How the context window fills up and what to do when it overflows
- Truncation, summarization, and sliding windows: the three strategies
- Token-aware message management: counting before you crash
- Conversation branching: when one user session spawns multiple agent threads

## Prerequisites
- [Anatomy of an AI Agent](../02-the-agent-loop/01-anatomy-of-an-agent.md) — the message list in the agent loop
- [How LLMs Actually Work](../01-foundations/01-how-llms-work.md) — tokens and context windows
- [Prompt Engineering](../01-foundations/02-prompt-engineering.md) — system prompts are part of memory

---

## The Message List Is Memory

Every turn in an agent conversation appends to a list:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": None, "tool_calls": [...]},
    {"role": "tool", "content": "{...}", "tool_call_id": "call_1"},
    {"role": "assistant", "content": "Tokyo is 22°C and sunny."},
    {"role": "user", "content": "How about London?"},
    {"role": "assistant", "content": None, "tool_calls": [...]},
    # ... this grows forever
]
```

This list **is** the agent's short-term memory. Everything the agent "knows" about the current conversation is in this list. There is no other memory. The model has no hidden state between API calls.

This means two things:
1. **Every detail must be in the messages.** If it's not in the list, the agent doesn't know it.
2. **The list always grows.** Every tool call adds at least 2 messages. Every user turn adds at least 1. Long conversations create long lists.

---

## The Context Window Problem

You have a fixed budget. GPT-4o gives you 128K tokens. Claude gives you 200K. That sounds like a lot — until you do the math.

### Where Your Tokens Go

```
┌─────────────────────────────────────────┐
│         CONTEXT WINDOW (128K)           │
│                                         │
│  ┌────────────────────────────────────┐ │
│  │ SYSTEM PROMPT         (2,000 tok)  │ │  ← Fixed cost, paid every call
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │ TOOL DEFINITIONS       (3,000 tok) │ │  ← Fixed cost, paid every call
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │ CONVERSATION HISTORY  (variable)   │ │  ← Grows with every turn
│  │  Turn 1: 200 tokens                │ │
│  │  Turn 2: 350 tokens                │ │
│  │  Turn 3: 500 tokens                │ │
│  │  ...                               │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │ CURRENT USER MESSAGE   (100 tok)   │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │ RESERVED FOR OUTPUT  (4,096 tok)   │ │  ← Must leave room for response
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

A 2,000-token system prompt + 3,000 tokens of tool definitions = **5,000 tokens gone before the user says a word.** Every single API call pays this tax.

### Realistic Capacity

With 5K fixed overhead and 4K reserved for output, you have ~119K tokens for conversation history. At ~500 tokens per average turn (user + assistant + tool calls), that's roughly **238 turns** before overflow.

That sounds fine for a chat. But an agent loop can burn through 5-10 turns per user query. Tools, retries, plan steps — each consumes tokens. A complex agent task can use 50+ turns.

---

## Strategy 1: Truncation (Drop Old Messages)

The simplest approach: when you're about to exceed the window, remove the oldest messages.

### Naive Truncation (Don't Do This)
```python
while count_tokens(messages) > MAX_TOKENS:
    messages.pop(1)  # Remove oldest non-system message
```
**Problem:** You might remove a tool result that a later message references. The conversation breaks.

### Smart Truncation: Preserve the System Prompt and Last N Turns
```python
def truncate_messages(messages: list[dict], max_tokens: int) -> list[dict]:
    """
    Keep the system prompt and the most recent complete turns.
    A 'turn' is: user → assistant (with possible tool calls) → tool results.
    """
    if count_tokens(messages) <= max_tokens:
        return messages
    
    system_msg = messages[0]  # Always preserve system prompt
    conversation = messages[1:]
    
    # Group messages into turns
    turns = []
    current_turn = []
    for msg in conversation:
        current_turn.append(msg)
        if msg["role"] == "assistant" and msg.get("content"):
            # Assistant gave a final answer — turn complete
            turns.append(current_turn)
            current_turn = []
    
    # Keep the most recent turns that fit
    kept_turns = []
    remaining = max_tokens - count_tokens([system_msg])
    
    for turn in reversed(turns):
        turn_tokens = count_tokens(turn)
        if turn_tokens <= remaining:
            kept_turns.insert(0, turn)
            remaining -= turn_tokens
        else:
            break
    
    return [system_msg] + [msg for turn in kept_turns for msg in turn]
```

> **Note:** The turn detection heuristic (`msg["role"] == "assistant" and msg.get("content")`) treats only messages with plain-text content as turn boundaries. An assistant message with only `tool_calls` and no `content` does not close a turn — the tool results that follow will be grouped with it. This is the correct behavior, but be aware that a half-complete turn at the end of the list is kept as-is.

**When to use:** Simple chatbots, internal tools, any situation where old context truly doesn't matter.

**When not to use:** Customer support (losing context frustrates users), complex multi-step tasks (the agent forgets what it was doing).

---

## Strategy 2: Summarization (Compress Old Messages)

Instead of dropping old messages, compress them into a summary. The agent loses detail but keeps context.

### Implementation
```python
def summarize_conversation(messages: list[dict], keep_recent: int = 5) -> list[dict]:
    """
    Summarize older messages, keep recent messages in full.
    """
    system_msg = messages[0]
    recent = messages[-keep_recent:]  # Last N messages stay verbatim
    to_summarize = messages[1:-keep_recent]  # Everything else gets summarized
    
    if not to_summarize:
        return messages  # Nothing to summarize
    
    # call_llm() here is shorthand for client.chat.completions.create():
    #   response = client.chat.completions.create(
    #       model="gpt-4o-mini",  # Use a cheaper model for summarization
    #       messages=[{"role": "system", "content": system_prompt},
    #                 {"role": "user", "content": formatted_history}]
    #   )
    #   summary = response.choices[0].message.content
    summary = call_llm(
        system="Summarize this conversation. Include: key facts, decisions "
               "made, tools called and their results, and any unresolved "
               "questions. Write in third person past tense.",
        user=format_messages_for_summary(to_summarize)
    )
    
    # Replace old messages with a summary message
    return [
        system_msg,
        {"role": "user", "content": f"[Previous conversation summary: {summary}]"},
        *recent
    ]
```

### The Summarization Prompt
```markdown
Summarize the following conversation between a user and an AI assistant.

Include:
- The user's original goal and any changes to it
- Key facts and data obtained from tools
- Decisions the assistant made
- Actions taken and their results
- Any unresolved questions or pending tasks

Format as a concise paragraph. Write in third person past tense.
Example: "The user asked for weather in Tokyo. The assistant called the
weather tool, which reported 22°C and sunny. The user then asked for
London weather, which is pending."

DO NOT include: small talk, greetings, or redundant information.
```

### When to Summarize
- Long conversations where the user might reference earlier details
- Customer support with history across multiple sessions
- Tasks where the overall goal matters more than exact wording

### The Risk of Summarization
The summary is a lossy compression. Critical details can be lost. If the user said "I need the report by Friday at 5pm EST," and the summary says "The user needs a report by Friday," the timezone is gone. Always keep the most recent messages verbatim.

---

## Strategy 3: Sliding Window with Overlap

A compromise between truncation and summarization: keep a window of recent messages in full, plus a rolling summary of everything before it.

```python
class SlidingWindowMemory:
    def __init__(self, max_tokens: int = 100000, recent_count: int = 10):
        self.max_tokens = max_tokens
        self.recent_count = recent_count
        self.full_history = []  # Complete history (for reference)
        self.summary = ""       # Compressed older history
        self._last_summarized = None  # Sentinel: None means "not yet summarized"
    
    def add_message(self, message: dict) -> None:
        self.full_history.append(message)
    
    def get_messages(self) -> list[dict]:
        """Get the messages to send to the LLM."""
        if count_tokens(self.full_history) <= self.max_tokens:
            return self.full_history
        
        # Split: recent messages stay verbatim, older gets summarized
        recent = self.full_history[-self.recent_count:]
        older = self.full_history[:-self.recent_count]
        
        # Update rolling summary if older messages changed.
        # Note: comparison by value (list equality) — works for dicts but
        # can be slow for very large histories. In production, compare by
        # length or a hash of the last message.
        if older != self._last_summarized:
            self.summary = self._summarize(older)
            self._last_summarized = older
        
        # Build the context
        result = [self.full_history[0]]  # System prompt
        if self.summary:
            result.append({
                "role": "user",
                "content": f"[Conversation so far: {self.summary}]"
            })
        result.extend(recent)
        return result
    
    def _summarize(self, messages: list[dict]) -> str:
        # Same summarization logic as Strategy 2
        ...
```

This is the production pattern. Recent context stays sharp. Old context stays accessible.

---

## Token Counting: Measure Before You Crash

The context window limit is enforced by the API — and it returns an error. You must count tokens *before* sending.

```python
import tiktoken

def count_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    """Count tokens in a messages array, including message formatting overhead."""
    encoding = tiktoken.encoding_for_model(model)
    
    total = 0
    for message in messages:
        # Every message has a formatting overhead (~4 tokens)
        total += 4
        for key, value in message.items():
            if value is None:
                continue
            if isinstance(value, str):
                total += len(encoding.encode(value))
            elif isinstance(value, list):
                # Tool calls are serialized as JSON
                total += len(encoding.encode(json.dumps(value)))
    
    total += 2  # Assistant message priming
    return total

# Always check before calling the API
def safe_api_call(messages, max_input_tokens=120000):
    token_count = count_tokens(messages)
    if token_count > max_input_tokens:
        messages = truncate_messages(messages, max_input_tokens)
        logging.warning(f"Truncated messages from {token_count} to "
                       f"{count_tokens(messages)} tokens")
    return call_llm(messages)
```

> **Code Reference:** [`memory_manager.py`](../../code/python/04-rag-pipeline/memory_manager.py) · [`conversation_summarizer.py`](../../code/python/04-rag-pipeline/conversation_summarizer.py)  
> `MemoryManager.get_messages(strategy, recent_count)` applies the selected strategy. `ConversationSummarizer` handles the LLM call for compress strategies (uses `gpt-4o-mini` by default). Node.js: [`memory_manager.ts`](../../code/nodejs/04-rag-pipeline/memory_manager.ts) · Go: [`memory_manager.go`](../../code/go/04-rag-pipeline/memory_manager.go)

---

## Conversation Branching

A user session might spawn multiple independent agent runs. Don't share message lists between them.

```
User session:
  ├── "Research Apple stock" → Agent run 1 (own message list)
  │     └── Messages: system + research task + tool calls + result
  │
  ├── "Draft an email about Q3 results" → Agent run 2 (own message list)
  │     └── Messages: system + writing task + result
  │
  └── "What's the weather?" → Agent run 3 (own message list)
        └── Messages: system + weather task + tool calls + result
```

Each branch starts fresh with the system prompt. If the user references a previous branch, you need to explicitly inject that context:

> **Code Reference:** [`branch_manager.py`](../../code/python/04-rag-pipeline/branch_manager.py) — `BranchManager.create_branch(name, user_query, context_from)` creates independent memory contexts. `merge_context(target, sources)` injects source branch summaries into the target.

```python
def create_branch(original_messages: list[dict], 
                  user_query: str,
                  context_from_branches: list[str] = None) -> list[dict]:
    """Start a new conversation branch."""
    messages = [original_messages[0]]  # System prompt only
    
    # Inject context from other branches if needed
    if context_from_branches:
        context_str = "\n".join(context_from_branches)
        messages.append({
            "role": "user",
            "content": f"[Context from previous tasks: {context_str}]"
        })
    
    messages.append({"role": "user", "content": user_query})
    return messages
```

---

## Memory Budgeting in Practice

For a production agent, budget your tokens intentionally:

| Component | Allocation | Notes |
|:---|:---|:---|
| System prompt | 2,000 tokens | Be ruthless. Every word must earn its place. |
| Tool definitions | 1,500 tokens | Use short descriptions. Group related tools. |
| Conversation history | 100,000 tokens | Rotate with sliding window |
| Reserved for output | 16,000 tokens | Depends on expected response length |
| **Buffer** | **8,500 tokens** | For token counting errors and overhead |

If your system prompt is 8,000 tokens, you've already lost 6% of your budget before the user types a character. A verbose system prompt is a tax on every single API call.

> **Code Reference:** [`memory_manager.py`](../../code/python/04-rag-pipeline/memory_manager.py) implements `estimated_cost()` which shows the current input token count and projected cost before the next API call. [`token_tracker.py`](../../code/python/04-rag-pipeline/token_tracker.py) accumulates usage across all calls and can enforce a budget cap with a warning at 80%.

---

## Common Pitfalls

- **"I never clean up the message list"**: Your agent works fine in testing (5 turns) and crashes in production (200 turns). Concretely: at ~500 tokens per turn, 200 turns = 100K tokens — right at GPT-4o's usable limit, and you haven't added tool definitions or system prompt yet. At turn 201, your API call returns a context-length error and the whole run fails. Implement truncation or summarization *before* you need it, ideally at turn 1.
- **"My summarizer loses critical details"**: Summarization needs a detailed prompt. Tell the summarizer exactly what to preserve: decisions, numbers, dates, unresolved items. Test summaries against the original conversation.
- **"I truncate by message count, not token count"**: Not all messages are equal. A tool result with 5,000 tokens of data is not the same as a 10-token user message. Always count tokens, not messages.
- **"I keep the entire history for every branch"**: Branch early, branch often. Each independent task should have its own message list. Shared history creates coupling.
- **"I forgot about tool definitions"**: Tool definitions are sent on every API call. 10 tools with verbose descriptions = thousands of tokens per call. Audit your tool descriptions regularly.

## What's Next

Short-term memory is the conversation history. Long-term memory is external knowledge stored in vector databases. Next: how embeddings turn text into searchable meaning.
→ [Embeddings and Vectors](02-embeddings-and-vectors.md)