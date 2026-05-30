# 02 — Prompt Engineering + Structured Output (Go)

This folder contains both the chapter-02 prompt-engineering demos and the chapter-03 structured-output demos.

## Files

| File | Description |
|---|---|
| `main.go` | Entry point: `-mode=template|few-shot|chain-of-thought|structured` |
| `prompt_template.go` | Template substitution, token counting, and LLM call |
| `few_shot_comparison.go` | Zero-shot vs few-shot with estimated and actual prompt-token counts |
| `chain_of_thought.go` | Same problem with and without CoT, including output-token cost |
| `structured_extraction.go` | Chapter 03: schema-validated extraction |
| `instructor_extraction.go` | Chapter 03: Instructor-style extraction |
| `function_calling_vs_structured.go` | Chapter 03: compare API paths |
| `retry_handler.go` | Chapter 03: parse-validate-retry helper |
| `prompt_template_test.go` | Go test suite (no API key required) |
| `go.mod` | Go module definition |

## Prerequisites

```bash
export OPENAI_API_KEY=your_key_here
go mod download
```

## Run

```bash
go run . -mode=template           # prompt templates
go run . -mode=few-shot           # zero-shot vs few-shot
go run . -mode=chain-of-thought   # reasoning with and without CoT
go run . -mode=structured         # chapter 03: schema-validated extraction
go test ./... -v                  # tests (no API key needed)
```

## Prompt Engineering Output — `go run . -mode=few-shot`

```
Input: "The new update is fine, I guess. Not bad, but nothing to get excited about."

Approach     Result        Format    Est. prompt   API prompt
----------------------------------------------------------------
Zero-shot    Neutral       ok                 ...          ...
Few-shot     Neutral       ok                 ...          ...

Few-shot overhead: +... estimated prompt tokens per request
```

## Prompt Template Output — `go run . -mode=template`

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
