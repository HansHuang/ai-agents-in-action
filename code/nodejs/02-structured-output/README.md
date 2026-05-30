# 02 — Prompt Engineering + Structured Output (Node.js)

This folder contains both the chapter-02 prompt-engineering demos and the chapter-03 structured-output demos.

## Files

| File | Description |
|---|---|
| `index.js` | Chapter-02 entry point: `template`, `few-shot`, or `chain-of-thought` |
| `prompt_template.js` | Template substitution, token counting, and LLM call |
| `few_shot_comparison.ts` | Zero-shot vs few-shot with estimated and actual prompt-token counts |
| `chain_of_thought.ts` | Same problem with and without CoT, including output-token cost |
| `zod_extraction.ts` | Chapter 03: schema-first extraction with Zod |
| `instructor_extraction.ts` | Chapter 03: Instructor-style extraction |
| `function_calling_vs_structured.ts` | Chapter 03: compare API paths |
| `retry_handler.ts` | Chapter 03: parse-validate-retry helper |
| `prompt_template.test.js` | `node:test` suite (no API key required) |
| `package.json` | Node.js dependencies |

## Prerequisites

```bash
export OPENAI_API_KEY=your_key_here
npm install
```

## Run

```bash
npm run start:template                  # prompt templates
npm run start:few-shot                  # zero-shot vs few-shot
npm run start:chain-of-thought          # reasoning with and without CoT
npm run start:zod                       # chapter 03: schema-first extraction
npm run start:instructor                # chapter 03: Instructor-style extraction
npm run start:compare                   # chapter 03: compare API paths
npm test                                # chapter 02 tests (no API key needed)
```

## Prompt Engineering Output — `npm run start:few-shot`

```
Input: "The new update is fine, I guess. Not bad, but nothing to get excited about."

Approach     Result        Format    Est. prompt   API prompt
----------------------------------------------------------------
Zero-shot    Neutral       ok                 ...          ...
Few-shot     Neutral       ok                 ...          ...

Few-shot overhead: +... estimated prompt tokens per request
```

## Prompt Template Output — `npm run start:template`

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
→ [Structured Output](../../../docs/01-foundations/03-structured-output.md)
