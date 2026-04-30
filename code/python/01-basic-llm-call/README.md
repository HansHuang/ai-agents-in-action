# 01 — Basic LLM Call (Python)

Demonstrates token counting for OpenAI-compatible models using `tiktoken`.

## Files

| File | Description |
|---|---|
| `main.py` | Counts tokens for a plain string and a messages array |
| `requirements.txt` | Python dependencies |

## Prerequisites

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Expected Output

```
Text  : 'The quick brown fox jumps over the lazy dog.'
Tokens: 9

Messages array token count: 27
```

## Related Docs

→ [How LLMs Actually Work](../../../docs/01-foundations/01-how-llms-work.md)
