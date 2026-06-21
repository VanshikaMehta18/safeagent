# SafeAgent Demo

Multi-agent LangGraph pipeline that consumes **[constitution-guard](../../constitution-guard/)** as its safety layer.

> Most agentic systems treat safety as a system prompt. SafeAgent treats it as middleware.

## Setup

```bash
cd apps/safeagent
pip install -r requirements.txt
cp .env.example .env   # set GEMINI_API_KEY for optional Tier 2 principle critics
uvicorn main:app --reload --port 8080
```

Open http://127.0.0.1:8080/docs to run queries.

## Architecture

```
Intent Classifier (Constitution Guard Tier 1 local guardrails)
    → Planner → Researcher → Synthesizer
    → Safety Arbitration (hybrid: 4 local + 3 LLM critics via GeminiCritic)
    → Escalation → Final Response
```

**Tier 1** (always local): jailbreak, PII, toxicity, crisis classifiers score non-maleficence, privacy, legality, and scope.

**Tier 2** (optional, when `GEMINI_API_KEY` set): honesty, fairness, and autonomy scored via injectable `PrincipleCritic` (`GeminiCritic` is the reference implementation).

Short-circuits: harmful/jailbreak → BLOCK, crisis → WARN + 988.

## API

| Endpoint | Description |
|----------|-------------|
| `POST /query` | Run full pipeline |
| `GET /safety-log` | Recent safety decisions |
| `GET /safety-report` | Aggregated stats |
| `GET /health` | Health check |
| `POST /eval/run` | Adversarial eval suite |

## Tests

```bash
pytest tests/ -v
```

## Docker

```bash
docker-compose up --build
```
