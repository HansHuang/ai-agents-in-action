# 02 — Prompt Engineering Techniques (Go)

Demonstrates prompt templates with variable substitution, token counting, and LLM calls.

## Files

| File | Description |
|---|---|
| `prompt_template.go` | Template substitution, token counting, LLM call |
| `prompt_template_test.go` | Go test suite (no API key required) |
| `main.go` | Entry point — calls `runPromptTemplate()` |
| `go.mod` | Go module definition |

## Prerequisites

```bash
export OPENAI_API_KEY=your_key_here
go mod download
```

## Run

```bash
go run .                  # template + LLM call
go test ./... -v          # tests (no API key needed)
```

## Expected Output — `go run .`

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
