# 01 — Basic LLM Call (Node.js)

Demonstrates token counting for OpenAI-compatible models using `js-tiktoken`.

## Files

| File | Description |
|---|---|
| `index.js` | Counts tokens for a plain string and a messages array |
| `package.json` | Node.js dependencies |

## Prerequisites

```bash
npm install
```

## Run

```bash
node index.js
```

## Expected Output

```
Text  : "The quick brown fox jumps over the lazy dog."
Tokens: 9

Messages array token count: 27
```

## Related Docs

→ [How LLMs Actually Work](../../../docs/01-foundations/01-how-llms-work.md)
