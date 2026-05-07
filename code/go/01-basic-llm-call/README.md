# 01 — Basic LLM Call (Go)

Demonstrates token counting for OpenAI-compatible models using `tiktoken-go`.

## Files

| File | Description |
|---|---|
| `main.go` | Counts tokens for a plain string and a messages slice |
| `go.mod` | Go module definition |

## Prerequisites

```bash
go mod tidy
```

## Run

```bash
go run main.go
```

## Expected Output

```
Text  : "The quick brown fox jumps over the lazy dog."
Tokens: 9

Messages array token count: 27
```

## Related Docs

→ [How LLMs Actually Work](../../../docs/01-foundations/01-how-llms-work.md)
