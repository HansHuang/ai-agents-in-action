# 02 — Structured Output (Python)

Demonstrates structured output extraction: function calling, JSON schema, Instructor + Pydantic, the parse-validate-retry pattern, and a side-by-side method comparison.

## Files

| File | Description |
|---|---|
| `prompt_template.py` | Prompt templates, token counting, and LLM call (from [02-prompt-engineering](../../../docs/01-foundations/02-prompt-engineering.md)) |
| `few_shot_comparison.py` | Zero-shot vs few-shot side-by-side with token counts |
| `chain_of_thought.py` | Same problem with and without CoT at `temperature=0` |
| `instructor_extraction.py` | Pydantic model + `instructor.from_openai()` with automatic retry |
| `retry_handler.py` | Generic `extract_with_retry()` using `json_schema` response_format |
| `function_calling_vs_structured.py` | Side-by-side: function calling vs structured output across 5 texts |
| `test_prompt_template.py` | pytest: prompt templates (no API key required) |
| `test_extraction.py` | pytest: extraction + retry logic (no API key required) |
| `requirements.txt` | Python dependencies |

## Prerequisites

```bash
export OPENAI_API_KEY=your_key_here
pip install -r requirements.txt
```

## Run

```bash
python instructor_extraction.py        # Pydantic + Instructor demo
python function_calling_vs_structured.py  # compare both API paths
python retry_handler.py                # reusable handler (import only)
pytest test_extraction.py -v           # tests (no API key needed)
```

## Expected Output — `instructor_extraction.py`

```
Text       : 'I absolutely love this, it changed my life!'
Sentiment  : positive
Confidence : 0.97
Key Phrases: ['absolutely love', 'changed my life']

Text       : "It's fine I guess, nothing special."
Sentiment  : neutral
Confidence : 0.82
Key Phrases: ['fine', 'nothing special']

Text       : 'Terrible product, broke after one day.'
Sentiment  : negative
Confidence : 0.95
Key Phrases: ['terrible', 'broke after one day']
```

All three implementations (Python, [Node.js](../../nodejs/02-structured-output/), [Go](../../go/02-structured-output/)) produce the same output for the same input.

## Related Docs

→ [Structured Output](../../../docs/01-foundations/03-structured-output.md)
