# Multi-Turn Context Management

## What You'll Learn
- Why long conversations break agents: context drift, forgotten goals, repeated questions
- Conversation state tracking: maintaining a record of what's been done and what's pending
- Progressive summarization: compressing old turns while preserving key decisions
- Context inheritance: how new conversation branches carry forward relevant history
- Recovery patterns: when the agent loses the thread, how to get it back
- Session design: when to persist, when to reset, and when to branch

## Prerequisites
- [Short-Term Memory](../03-memory-and-retrieval/01-short-term-memory.md) — the message list and truncation strategies
- [Dynamic Prompt Assembly](02-dynamic-prompt-assembly.md) — building prompts from multiple sources
- [Context Compression and Filtering](03-context-compression-and-filtering.md) — compressing context without losing signal

---

## The Long Conversation Problem

Short conversations are easy. The user asks a question, the agent answers, done.

Long conversations are hard. After 30 turns, the agent faces problems no short-lived agent ever sees:

| Problem | Example | Why It Happens |
|:---|:---|:---|
| **Context drift** | User asked about Q3 report, agent is now discussing Q2 | The original goal was in turn 1, long since truncated |
| **Repetition** | Agent asks "What's your order number?" for the third time | The agent forgot it already asked (and the user already answered) |
| **Contradiction** | Agent recommends product A in turn 5, product B in turn 25 | The agent doesn't remember its own recommendations |
| **Goal abandonment** | User changed the subject in turn 12, original task never completed | No task tracking: things just fall off the context |
| **Identity loss** | Agent switches from "support mode" to "general assistant" | System prompt is at the top but behavior drifts over long contexts |

These aren't bugs. They're consequences of the architecture. The agent's only memory is the message list, and the message list can't hold everything.

---

## The Solution: State Tracking

Don't rely on the message list alone. Maintain an explicit state object that survives truncation.

```python
@dataclass
class ConversationState:
    """Explicit state that persists across message list truncations."""
    
    # Task tracking
    current_goal: str = None          # What is the user trying to accomplish?
    subtasks_completed: list[str] = None  # What's been done so far?
    subtasks_pending: list[str] = None    # What's still outstanding?
    
    # User context
    user_name: str = None
    user_preferences: dict = None
    user_provided_info: dict = None   # Info the user has already shared
    
    # Agent context
    agent_recommendations: list[str] = None  # What has the agent suggested?
    agent_questions_asked: list[str] = None  # What has the agent already asked?
    agent_mode: str = "general"        # Current operating mode
    
    # Conversation health
    turns_since_goal_mentioned: int = 0
    user_frustration_signals: int = 0
    topic_changes: list[str] = None
    
    def to_prompt_context(self) -> str:
        """Generate a compact state summary for injection into the prompt."""
        parts = []
        
        if self.current_goal:
            parts.append(f"Current goal: {self.current_goal}")
        
        if self.subtasks_completed:
            parts.append(f"Completed: {', '.join(self.subtasks_completed)}")
        
        if self.subtasks_pending:
            parts.append(f"Still need to: {', '.join(self.subtasks_pending)}")
        
        if self.agent_recommendations:
            parts.append(f"Previous recommendations: {', '.join(self.agent_recommendations)}")
        
        if self.agent_questions_asked:
            parts.append(f"Already asked about: {', '.join(self.agent_questions_asked)}")
        
        if self.user_provided_info:
            info_str = ", ".join(f"{k}={v}" for k, v in self.user_provided_info.items())
            parts.append(f"User provided: {info_str}")
        
        return "\n".join(parts)
```

### Injecting State into Every Turn

The state summary is included in every prompt, even when the message list is truncated:

```python
def build_messages_with_state(messages: list[dict], 
                              state: ConversationState) -> list[dict]:
    """Build messages array with state injected into the system prompt."""
    
    system_msg = messages[0]
    
    # Inject state after the system prompt
    state_context = state.to_prompt_context()
    if state_context:
        augmented_system = (
            f"{system_msg['content']}\n\n"
            f"## Conversation State (maintained across turns)\n"
            f"{state_context}\n\n"
            f"Use this state to maintain continuity. Do not repeat questions "
            f"already asked. Do not contradict previous recommendations."
        )
    else:
        augmented_system = system_msg["content"]
    
    return [
        {"role": "system", "content": augmented_system},
        *messages[1:]  # Rest of the conversation
    ]

# The state persists even when the message list is truncated:
def manage_long_conversation(messages, state, max_tokens=50000):
    """Truncate messages but preserve state."""
    # Truncate old messages
    messages = sliding_window_truncate(messages, max_tokens)
    
    # But state survives independently
    messages = build_messages_with_state(messages, state)
    
    return messages
```

---

## State Transitions

The state should evolve with every turn. After each agent response, update the state:

```python
class StateManager:
    """Manages conversation state across turns."""
    
    def __init__(self):
        self.state = ConversationState()
    
    def process_user_turn(self, user_message: str) -> None:
        """Update state based on what the user said."""
        # Detect if the user changed the subject
        if self.state.current_goal:
            relevance = check_goal_relevance(user_message, self.state.current_goal)
            if relevance < 0.5:
                self.state.topic_changes.append(user_message[:100])
                self.state.turns_since_goal_mentioned += 1
        
        # Extract any information the user provided
        extracted = extract_user_info(user_message)
        self.state.user_provided_info.update(extracted)
    
    def process_agent_turn(self, agent_response: str, 
                           tool_calls: list = None) -> None:
        """Update state based on what the agent did."""
        # Track recommendations
        if "I recommend" in agent_response or "I suggest" in agent_response:
            recommendations = extract_recommendations(agent_response)
            self.state.agent_recommendations.extend(recommendations)
        
        # Track questions asked
        if "?" in agent_response:
            questions = extract_questions(agent_response)
            self.state.agent_questions_asked.extend(questions)
        
        # Track task progress
        if tool_calls:
            for call in tool_calls:
                self.state.subtasks_completed.append(
                    f"{call['function']['name']}({call['function']['arguments']})"
                )
    
    def set_goal(self, goal: str, subtasks: list[str] = None) -> None:
        """Set or update the current conversation goal."""
        self.state.current_goal = goal
        self.state.subtasks_pending = subtasks or []
        self.state.subtasks_completed = []
        self.state.turns_since_goal_mentioned = 0
    
    def build_system_prompt_with_state(self, system_prompt: str) -> str:
        """Augment the system prompt with current state context."""
        ctx = self.state.to_prompt_context()
        if not ctx:
            return system_prompt
        return (
            f"{system_prompt}\n\n"
            "## Conversation State (maintained across turns)\n"
            f"{ctx}\n\n"
            "Use this state to maintain continuity. Do not repeat questions "
            "already asked. Do not contradict previous recommendations."
        )
```

---

## Goal Detection and Tracking

The agent needs to know what the user is trying to accomplish. Detect goals explicitly:

```python
def detect_goal(user_message: str, conversation_history: list[dict]) -> dict:
    """
    Detect the user's goal from their message and conversation context.
    
    Returns: {
        "goal": "...",
        "is_new_goal": True/False,
        "supersedes_previous": True/False,
        "subtasks": ["...", "..."]
    }
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": """Analyze the user's message and determine their goal.

Output JSON:
{
    "goal": "Clear one-sentence description of what the user wants",
    "is_new_goal": true/false,
    "supersedes_previous": true/false,
    "subtasks": ["step1", "step2"],
    "priority": "high/medium/low",
    "expected_turns": 3
}

Rules:
- is_new_goal: true if this is a new topic, false if continuing previous
- supersedes_previous: true if this goal replaces the previous one
- subtasks: break complex goals into 2-5 concrete steps
- expected_turns: estimate how many agent turns this goal requires"""
        }, {
            "role": "user",
            "content": f"Previous conversation summary: {summarize(conversation_history)}\n\nUser message: {user_message}"
        }],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)
```

---

## Progressive Summarization

Don't summarize the entire conversation at once. Summarize incrementally as the conversation grows.

```python
class ProgressiveSummarizer:
    """
    Summarize conversations incrementally.
    Older turns get compressed more aggressively than recent turns.
    """
    
    def __init__(self):
        self.layer_1_summary = ""  # Summary of turns 1-10
        self.layer_2_summary = ""  # Summary of turns 11-20 (when layer 1 fills)
        self.layer_3_summary = ""  # Summary of turns 21+ (when layer 2 fills)
        self.recent_turns = []     # Last 5 turns, always verbatim
    
    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Add a turn and rebalance summaries if needed."""
        self.recent_turns.append((user_msg, assistant_msg))
        
        # Keep only last 5 turns verbatim
        if len(self.recent_turns) > 5:
            oldest_turn = self.recent_turns.pop(0)
            self._incorporate_into_summaries(oldest_turn)
    
    def _incorporate_into_summaries(self, turn: tuple) -> None:
        """Add a turn to the progressive summary layers."""
        turn_text = f"User: {turn[0]}\nAssistant: {turn[1]}"
        
        # If layer 1 is getting long, compress it into layer 2
        if count_tokens(self.layer_1_summary) > 2000:
            self.layer_2_summary = self._compress_layer(
                self.layer_2_summary, self.layer_1_summary
            )
            self.layer_1_summary = ""
        
        # Add turn to layer 1
        self.layer_1_summary = self._summarize_turns(
            self.layer_1_summary, turn_text
        )
    
    def _summarize_turns(self, existing_summary: str, 
                         new_turn: str) -> str:
        """Summarize existing summary + new turn into an updated summary."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Update this conversation summary with the new turn.
Preserve: goals, decisions, key facts, user preferences, agent recommendations.
The updated summary should be about the same length as the original.

Existing summary:
{existing_summary if existing_summary else "(no previous summary)"}

New turn:
{new_turn}

Updated summary:"""
            }]
        )
        return response.choices[0].message.content
    
    def _compress_layer(self, older_summary: str, 
                        newer_summary: str) -> str:
        """Compress two summary layers into one."""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Combine these two conversation summaries into one.
Keep the most important information from both. Discard redundant details.

Earlier conversation:
{older_summary if older_summary else "(none)"}

Later conversation:
{newer_summary}

Combined summary:"""
            }]
        )
        return response.choices[0].message.content
    
    def get_context(self) -> str:
        """Get the full context for prompt injection."""
        parts = []
        
        if self.layer_3_summary:
            parts.append(f"[Early conversation: {self.layer_3_summary}]")
        if self.layer_2_summary:
            parts.append(f"[Earlier conversation: {self.layer_2_summary}]")
        if self.layer_1_summary:
            parts.append(f"[Recent conversation: {self.layer_1_summary}]")
        if self.recent_turns:
            recent_text = "\n".join(
                f"User: {u}\nAssistant: {a}" 
                for u, a in self.recent_turns
            )
            parts.append(f"[Most recent:\n{recent_text}]")
        
        return "\n\n".join(parts)
```

The key insight: **older information gets compressed more.** Turns 1-10 might become a 200-word summary. Turns 11-20 become a 150-word summary. Turns 21+ become a 100-word summary. The last 5 turns are always verbatim.

All summarisation calls include a deterministic fallback (keyword-sentence scoring) for cases where the LLM is unavailable — for example, during automated tests or in environments without an API key.

---

## Recovery Patterns

Even with state tracking, agents sometimes lose the thread. Design recovery mechanisms:

### Pattern 1: Goal Reminder

When the agent hasn't mentioned the goal in several turns, remind it:

```python
def check_goal_drift(messages: list[dict], state: ConversationState) -> bool:
    """
    Check if the conversation has drifted from the original goal.
    If the goal hasn't been mentioned in 5+ turns, it's drift.
    """
    if state.turns_since_goal_mentioned >= 5:
        # Inject a goal reminder
        reminder = (
            f"\n\n[REMINDER: Your current goal is: {state.current_goal}. "
            f"Subtasks pending: {', '.join(state.subtasks_pending)}. "
            f"Stay focused on this goal or ask the user if they want to change it.]"
        )
        messages.append({"role": "system", "content": reminder})
        state.turns_since_goal_mentioned = 0
        return True
    return False
```

### Pattern 2: Explicit Checkpoint

Periodically ask the agent to summarize progress:

```python
def inject_checkpoint(messages: list[dict], state: ConversationState) -> None:
    """
    Every 10 turns, ask the agent to confirm it knows where it is.
    """
    checkpoint_prompt = (
        f"\n\n[CHECKPOINT: Summarize your progress on the goal: {state.current_goal}. "
        f"What have you completed? What remains? Are there any blockers?]"
    )
    messages.append({"role": "system", "content": checkpoint_prompt})
```

### Pattern 3: User-Initiated Reset

Give users a way to reset the conversation:

```python
def detect_reset_intent(user_message: str) -> bool:
    """Detect if the user wants to start over."""
    reset_phrases = [
        "start over", "forget everything", "new topic",
        "let's begin again", "reset", "clear the conversation"
    ]
    return any(phrase in user_message.lower() for phrase in reset_phrases)

def handle_reset(state: ConversationState) -> tuple[list[dict], ConversationState]:
    """Reset the conversation but preserve user identity."""
    new_state = ConversationState()
    # Preserve user identity
    new_state.user_name = state.user_name
    new_state.user_preferences = state.user_preferences
    
    messages = [
        {"role": "system", "content": "Conversation has been reset. Starting fresh."},
        {"role": "assistant", "content": "I've cleared our conversation history. How can I help you?"}
    ]
    
    return messages, new_state
```

---

## Session Design: When to Persist, Reset, or Branch

Not every turn belongs in the same conversation:

| Decision | When | How |
|:---|:---|:---|
| **Continue** | User is working on the same goal | Append to current session |
| **Branch** | User starts a new but related task | New conversation, inherit user context + goal summary |
| **Reset** | User explicitly asks to start over | New conversation, keep only user identity |
| **Expire** | Session is old, user is new | Fresh conversation, no context inherited |

```python
class SessionManager:
    """Manage conversation sessions across multiple interactions."""
    
    def __init__(self, ttl_minutes: int = 60):
        self.sessions: dict[str, Session] = {}
        self.ttl_minutes = ttl_minutes
    
    def get_session(self, user_id: str) -> Session:
        """Get existing session or create a new one."""
        session = self.sessions.get(user_id)
        
        # Expire old sessions
        if session and session.is_expired(self.ttl_minutes):
            session = None
        
        if not session:
            session = Session(user_id=user_id)
            self.sessions[user_id] = session
        
        return session
    
    def branch_session(self, user_id: str, 
                       inherit_context: list[str] = None) -> Session:
        """Create a new session that inherits context from the previous one."""
        parent = self.sessions.get(user_id)
        
        new_session = Session(user_id=user_id)
        
        if parent and inherit_context:
            # Carry forward specific context
            new_session.state.user_name = parent.state.user_name
            new_session.state.user_preferences = parent.state.user_preferences
            
            # Include goal summary if requested
            if "goal" in inherit_context and parent.state.current_goal:
                new_session.inherited_context = (
                    f"Previous conversation goal was: {parent.state.current_goal}. "
                    f"Completed: {', '.join(parent.state.subtasks_completed)}."
                )
        
        self.sessions[user_id] = new_session
        return new_session
    
    def reset_session(self, user_id: str) -> Session:
        """Full reset — only preserve user identity."""
        session = Session(user_id=user_id)
        self.sessions[user_id] = session
        return session

class Session:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.created_at = time.time()
        self.last_activity = time.time()
        self.state = ConversationState()
        self.messages: list[dict] = []
        self.summarizer = ProgressiveSummarizer()
        self.inherited_context: str = None
    
    def is_expired(self, ttl_minutes: int) -> bool:
        return (time.time() - self.last_activity) > (ttl_minutes * 60)
```

> **Code Reference:** [Python](../../code/python/05-context-assembly/) · [Node.js](../../code/nodejs/05-context-assembly/) · [Go](../../code/go/05-context-assembly/)  
> The context assembly implementations include `ConversationState`, `StateManager`, `ProgressiveSummarizer`, `SessionManager`, and `RecoveryManager` (with `GoalDetector`). All LLM-dependent methods include a deterministic keyword-extraction fallback so the code runs offline during testing.

---

## The Complete Multi-Turn Architecture

```
User Message
    │
    ▼
┌─────────────────────┐
│ 1. SESSION LOOKUP   │  Get or create session for user
│    (SessionManager) │  Check TTL, handle expiry
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 2. INTENT DETECTION │  New goal? Continue? Reset? Branch?
│    (StateManager /   │  GoalDetector runs via RecoveryManager,
│     GoalDetector)    │  not inline in the request path
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 3. STATE UPDATE     │  Update ConversationState based on user input
│    (StateManager)    │  Extract info, detect drift, track progress
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 4. CONTEXT ASSEMBLY │  Build messages with state injection
│    (PromptAssembler) │  Include progressive summaries
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 5. DRIFT CHECK      │  If drifting, inject goal reminder
│    (Recovery)        │  If checkpoint needed, inject checkpoint
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 6. LLM CALL         │  Execute with assembled context
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 7. STATE UPDATE     │  Update state based on agent response
│    (StateManager)    │  Track recommendations, questions, progress
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 8. PROGRESSIVE      │  Add turn to summarizer
│    SUMMARIZATION     │  Rebalance layers if needed
└──────────┬──────────┘
           ▼
      Response to User
```

---

## Common Pitfalls

- **"I rely entirely on the message list for state"**: The message list gets truncated. State doesn't. Maintain an explicit state object that survives truncation.
- **"My agent repeats questions because it forgot the answers"**: The state should track `user_provided_info`. Before the agent asks for information, check if it's already in the state.
- **"I summarize the entire conversation from scratch every turn"**: This is expensive and slow. Use progressive summarization — update incrementally, not from scratch.
- **"My session never expires"**: A session from last week shouldn't influence today's conversation. Set a TTL. Expire sessions after 30-60 minutes of inactivity.
- **"I don't give users a way to reset"**: Sometimes the user just wants to start over. Detect reset intent and handle it gracefully.
- **"My agent doesn't know when the goal is complete"**: The state should track `subtasks_pending`. When all subtasks are done, the agent should confirm completion with the user.

## What's Next

You've completed the context engineering section. You can now manage conversations across any number of turns. Next: the broader tool ecosystem — model providers, vector databases, observability, and the MCP protocol.
→ [Model Providers](../05-the-tool-ecosystem/01-model-providers.md)