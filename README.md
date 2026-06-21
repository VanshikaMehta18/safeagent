# SafeAgent

This repository contains two projects:

## [constitution-guard/](constitution-guard/) — pip-installable library

Constitutional AI middleware for any agent:

```bash
pip install -e "constitution-guard[classifiers]"
```

```python
from constitution_guard import Guard, GuardConfig, GeminiCritic

guard = Guard(GuardConfig(jailbreak=True, pii=True, toxicity=True))

@guard.wrap
async def my_agent(query: str) -> str:
    ...
```

See [constitution-guard/README.md](constitution-guard/README.md).

## [apps/safeagent/](apps/safeagent/) — demo application

Full multi-agent LangGraph pipeline (planner → researcher → synthesizer → safety → escalation) that **consumes constitution-guard** as its safety layer.

```bash
cd apps/safeagent
pip install -r requirements.txt
cp .env.example .env   # set GEMINI_API_KEY for optional Tier 2 critics
uvicorn main:app --reload --port 8080
```

See [apps/safeagent/README.md](apps/safeagent/README.md).

## Architecture

```
constitution-guard (library)
    ├── Tier 1: local guardrails (jailbreak, PII, toxicity, crisis) → 4 principles
    ├── Tier 2: optional PrincipleCritic (GeminiCritic reference impl) → 3 principles
    └── guard.wrap(your_agent)

apps/safeagent (demo)
    └── LangGraph pipeline using constitution-guard
```

Seven principles total: 4 scored locally, 3 via injectable LLM critics. Provider-agnostic by design — Gemini is the reference implementation.
