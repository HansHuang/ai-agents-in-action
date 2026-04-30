# 02 — Prompt Engineering Techniques (Node.js)

Demonstrates prompt templates with variable substitution, token counting, and LLM calls.

## Files

| File | Description |
|---|---|
| `prompt_template.js` | Template substitution, token counting, LLM call |
| `prompt_template.test.js` | `node:test` suite (no API key required) |
| `index.js` | Entry point — runs the prompt template example |
| `package.json` | Node.js dependencies |

## Prerequisites

```bash
export OPENAI_API_KEY=your_key_here
npm install
```

## Run

```bash
node index.js                           # template + LLM call
node --test prompt_template.test.js     # tests (no API key needed)
```

## Expected Output — `index.js`

```
Token count before sending: 94

Response:
• ...
• ...
• ...

Actual tokens used — prompt: 94, completion: 62
```

## Related Docs

→ [Prompt Engineering](../../../docs/01-foundations/02-prompt-engineering.md)
