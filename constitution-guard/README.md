# Constitution Guard

> Most agentic systems treat safety as a system prompt. Constitution Guard treats it as middleware.

**pip-installable Constitutional AI middleware** — wrap any agent with local guardrails and optional LLM principle critics.

```bash
pip install constitution-guard[classifiers]   # Tier 1 local guardrails only
pip install constitution-guard[all]             # classifiers + Gemini critic
```

## Design

Seven constitutional principles, hybrid scoring:

| Principle | Tier | Scoring |
|-----------|------|---------|
| non_maleficence | 1 (local) | toxicity + crisis classifiers |
| privacy | 1 (local) | PII classifier |
| legality | 1 (local) | jailbreak classifier |
| scope_appropriateness | 1 (local) | keyword heuristics |
| honesty | 2 (LLM) | `PrincipleCritic` |
| fairness | 2 (LLM) | `PrincipleCritic` |
| autonomy_preservation | 2 (LLM) | `PrincipleCritic` |

Still 7 principles — refined from the original set for hybrid routing: **reversibility** dropped (overlaps non-maleficence), **consent_alignment** renamed to **fairness**, **legality** promoted from an implicit jailbreak gate to a named Tier 1 principle.

- **Tier 1 — guardrails (always local, no API key):** score 4 principles directly. No LLM re-scoring on these.
- **Tier 2 — principle critics (optional, LLM-backed):** honesty, fairness, and autonomy require semantic judgment. One structured LLM call per principle via a pluggable `PrincipleCritic`.

> *4 of 7 principles are fully local and free; the 3 that need semantic judgment go through a pluggable critic.*

When no critic is attached, Tier 2 principles are **`None` / unscored** — not fake-perfect 1.0. `SafetyScore.scored_principles` and `SafetyScore.unscored_principles` make it auditable which principles counted toward `overall_score` and `verdict`. Silently assigning a perfect honesty score to something never evaluated would be exactly the failure mode this library exists to prevent.

Constitutional scoring is **provider-agnostic by design**. `GeminiCritic` is the reference implementation; Anthropic, OpenAI, or Ollama backends are a few lines each.

## Quickstart

```python
from constitution_guard import Guard, GuardConfig, GeminiCritic

guard = Guard(GuardConfig(
    jailbreak=True,
    pii=True,
    toxicity=True,
    crisis=True,
    principle_critic=None,  # local-only, no API key
))

@guard.wrap
async def my_agent(user_message: str):
    return await call_my_llm(user_message)

result = await my_agent("What are cold symptoms?")
# GuardResult with .content, .verdict, .input_checks, .output_checks, .constitutional_score
```

## Tier 1 — local guardrails

| Classifier | Model | Principle |
|------------|-------|-----------|
| Jailbreak | `protectai/deberta-v3-base-prompt-injection-v2` | legality |
| Toxicity | `unitary/toxic-bert` | non-maleficence |
| PII | Regex + optional NER | privacy |
| Crisis | Keyword heuristics | non-maleficence (self-harm → 988) |
| Scope heuristics | Keyword rules | scope_appropriateness |

## Tier 2 — principle critics (optional)

```python
from constitution_guard import GeminiCritic

critic = GeminiCritic(api_key="your-key", model="gemini-2.5-flash")
guard = Guard(GuardConfig(principle_critic=critic))
```

Or inject any object satisfying `PrincipleCritic`:

```python
class MyCritic:
    async def score(self, principle, query, draft, context) -> PrincipleScore: ...
    async def rewrite(self, query, draft, flagged, reasoning_trace) -> RewriteResponse: ...

guard = Guard(GuardConfig(principle_critic=MyCritic()))
```

The deprecated `constitutional_backend="gemini"` string still works as a shim.

## LangGraph integration

```python
graph.add_node("planner", guard.wrap_node(run_planner))
```

## Configuration

See `GuardConfig` for thresholds, model IDs, `max_retries`, and `return_content=True` to return plain strings instead of `GuardResult`.

## License

MIT
