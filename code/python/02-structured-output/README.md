# 02 — Prompt Engineering + Structured Output (Python)

This folder contains both the chapter-02 prompt-engineering demos and the chapter-03 structured-output demos.

## Files

| File | Description |
|---|---|
| `main.py` | Chapter-02 entry point: `--mode template|few-shot|chain-of-thought` |
| `prompt_template.py` | Prompt templates, token counting, and LLM call |
| `few_shot_comparison.py` | Zero-shot vs few-shot with estimated and actual prompt-token counts |
| `chain_of_thought.py` | Same problem with and without CoT, including output-token cost |
| `instructor_extraction.py` | Pydantic model + `instructor.from_openai()` with automatic retry |
| `retry_handler.py` | Generic `extract_with_retry()` using `json_schema` response_format |
| `function_calling_vs_structured.py` | Side-by-side: function calling vs structured output across 5 texts |
| `test_prompt_template.py` | pytest: prompt-template logic (no API key required) |
| `test_extraction.py` | pytest: extraction + retry logic (no API key required) |
| `requirements.txt` | Python dependencies |

## Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
export OPENAI_API_KEY=your_key_here
pip install -r requirements.txt
```

## Run

```bash
python main.py --mode template           # prompt templates
python main.py --mode few-shot           # zero-shot vs few-shot
python main.py --mode chain-of-thought   # reasoning with and without CoT
python instructor_extraction.py          # chapter 03: Instructor-style extraction
python function_calling_vs_structured.py # chapter 03: compare both API paths
pytest test_prompt_template.py -v        # chapter 02 tests (no API key needed)
pytest test_extraction.py -v             # chapter 03 tests (no API key needed)
```

## Prompt Engineering Output — `python main.py --mode few-shot`

```
Input: "The new update is fine, I guess. Not bad, but nothing to get excited about."

Approach     Result        Format    Est. prompt   API prompt
----------------------------------------------------------------
Zero-shot    Neutral       ok                 ...          ...
Few-shot     Neutral       ok                 ...          ...

Few-shot overhead: +... estimated prompt tokens per request
```

## Structured Output Output — `instructor_extraction.py`

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

## Related Docs

→ [Prompt Engineering](../../../docs/01-foundations/02-prompt-engineering.md)
→ [Structured Output](../../../docs/01-foundations/03-structured-output.md)
