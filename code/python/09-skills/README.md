# 09-skills — The Skill Abstraction

A **Skill** is a tool wrapped with everything production needs: input validation,
output normalisation, a fallback, a prompt fragment, and test cases.

> **Skills are testable without an LLM.** This is the boundary between
> AI engineering and software engineering.

See: [docs/02-the-agent-loop/05-skills-composing-capabilities.md](../../../docs/02-the-agent-loop/05-skills-composing-capabilities.md)

---

## Files

| File | What it demonstrates |
|:-----|:---------------------|
| `skill_base.py` | `Skill`, `SkillRegistry`, `SkillResult`, `SkillInputError`, `SkillTest` |
| `skills/weather_tools.py` | Mock weather tool + validator + normaliser + fallback |
| `skills/weather_skill.py` | `create_weather_skill()` — all six skill components |
| `skills/stock_tools.py` | Shared stock tool functions (used by both stock skills) |
| `skills/stock_price_skill.py` | `create_stock_price_skill()` — price + 52-week range |
| `skills/stock_analysis_skill.py` | `create_stock_analysis_skill(registry)` — **depends on stock_price** |
| `skilled_agent.py` | `SkilledAgent` — ReAct agent that loads skills instead of raw tools |
| `skill_test_runner.py` | `SkillTestRunner` — run skill tests, no LLM required |
| `test_skills.py` | 15 pytest tests (all passing, no API key needed) |

---

## Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```python
from skill_base import SkillRegistry
from skills.weather_skill import create_weather_skill
from skill_test_runner import SkillTestRunner

# 1. Register the skill
registry = SkillRegistry()
registry.register(create_weather_skill())

# 2. Run its tests — no agent, no LLM
runner = SkillTestRunner(registry)
for report in runner.run_all():
    print(report)  # [PASS] weather_reporting: 3/3 passed

# 3. Use it in an agent
from skilled_agent import SkilledAgent
agent = SkilledAgent(registry)
agent.load_skills(["weather_reporting"])
result = agent.run("What's the weather in Tokyo?")
print(result.answer)
```

## Skill Dependency Graph

```
stock_analysis  ──depends on──▶  stock_price
```

`stock_analysis_skill` calls `stock_price` via the registry closure.
The registry validates this graph at registration time and rejects cycles.

## Run the Tests

```bash
python3.14 -m pytest test_skills.py -v
```

## Cross-Language

- **Node.js:** [code/nodejs/09-skills/skill_base.ts](../../nodejs/09-skills/skill_base.ts)
- **Go:** [code/go/09-skills/skill_base.go](../../go/09-skills/skill_base.go)
