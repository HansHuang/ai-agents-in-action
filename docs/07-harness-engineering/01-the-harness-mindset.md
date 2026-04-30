# The Harness Mindset

## LLMs Are Components, Not Applications
- They're more like a database than a backend
- You don't trust a database query—you validate, retry, and fallback

## The Harness as a State Machine
- Every LLM interaction is a state transition with defined failure modes
- Diagram: Happy Path → Retry → Fallback → Graceful Degradation → Error

## Core Principles
1. Never trust the model's output without validation
2. Every LLM call needs a timeout
3. Every LLM call needs a fallback path
4. The harness is deterministic; the model is probabilistic
5. Observability is not optional