# SafeAgent

> Most agentic systems treat safety as a system prompt. SafeAgent treats it as an agent.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST-teal.svg)](https://fastapi.tiangolo.com/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange.svg)](https://ai.google.dev/)
[![FAISS](https://img.shields.io/badge/FAISS-vector%20store-purple.svg)](https://github.com/facebookresearch/faiss)

SafeAgent is a production-grade multi-agent system where AI safety is a first-class architectural concern. Every agent action passes through a dedicated **Safety Arbitration Agent** before reaching the user, inspired by Anthropic's Constitutional AI methodology.

## Architecture

```
┌─────────┐    ┌──────────────────┐    ┌────────────┐    ┌─────────────┐
│  User   │───▶│ Intent Classifier│───▶│  Planner   │───▶│ Researcher  │───▶ ...
│  Query  │    │    (Agent 0)     │    │   Agent    │    │   Agent     │
└─────────┘    └────────┬─────────┘    └────────────┘    └─────────────┘
                          │
              harmful/jailbreak ──▶ BLOCK (short-circuit)
              crisis ──▶ WARN + 988 redirect (short-circuit)
```

Full pipeline: Intent Classifier → Planner → Researcher → Synthesizer → Safety Arbitration → Escalation

## The Six Agents

0. **Intent Classifier (Agent 0)** — Upstream query intent detection. Short-circuits harmful/jailbreak queries (BLOCK) and crisis/self-harm queries (WARN + 988 redirect) before research runs.

1. **Planner** — Decomposes the user query into 2–4 research subtasks and classifies query type (medical, legal, financial, general, sensitive) with an initial risk estimate.

2. **Researcher** — Gathers sources via FAISS vector retrieval and DuckDuckGo web search. Deduplicates before passing upstream.

3. **Synthesizer** — Drafts a cited response using research sources. Output never goes directly to the user.

4. **Safety Arbitration** — The core of the project. Scores every draft against 7 constitutional principles using dedicated Gemini critic calls. Applies intent overlay caps for harmful/crisis queries. Logs every decision to `safety_log.jsonl`.

5. **Escalation** — Handles PASS (return with safety note), WARN (rewrite and re-check, **max 2 retries** — counter incremented in escalation node, not router), or BLOCK (constructive refusal with professional referral). After max retries, escalates to BLOCK with seek-help messaging.

### Retry semantics

`retry_count` is incremented inside the Escalation agent when a WARN rewrite is issued. LangGraph routing functions are read-only (router mutations are not persisted). Max retries defaults to 2 (`MAX_SAFETY_RETRIES` in `.env`).

## The Seven Constitutional Principles

| Principle | Description | Weight |
|-----------|-------------|--------|
| **Non-maleficence** | Does the response risk physical, psychological, or financial harm? | 1.5x |
| **Honesty** | Is the response truthful and not misleading? | 1.5x |
| **Autonomy preservation** | Does it respect the user's right to make their own decisions? | 1.0x |
| **Privacy** | Does it expose or encourage exposure of private information? | 1.0x |
| **Scope appropriateness** | Is this within the system's safe operating domain? | 1.0x |
| **Reversibility** | If acted upon, are the consequences reversible? | 1.0x |
| **Consent alignment** | Does the response do what the user asked without overstepping? | 1.0x |

**Verdict thresholds:** overall score ≥ 0.8 → PASS, 0.5–0.79 → WARN, < 0.5 → BLOCK

## Quick Start

### Prerequisites

- Python 3.11+
- Gemini API key
- (Optional) LangSmith API key for tracing

### Setup

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY

pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Run the full pipeline |
| `GET` | `/safety-log` | Last 50 safety decisions (`?verdict=BLOCK` to filter) |
| `GET` | `/safety-report` | Aggregated safety statistics |
| `GET` | `/health` | Health check |
| `POST` | `/eval/run` | Run adversarial eval suite |

### Example Query

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of the common cold?"}'
```

## Running Evals

```bash
python -m evals.eval_runner
```

This runs 30 adversarial prompts across 5 categories and generates `eval_report.md`.

### Sample Eval Results

| Metric | Value |
|--------|-------|
| Overall accuracy | _TBD after first run_ |
| False positive rate | _TBD_ |
| False negative rate | _TBD_ |
| Mean latency | _TBD_ |
| Harmful requests accuracy | _TBD_ |
| Safe queries accuracy | _TBD_ |

> Replace placeholder values after running `python -m evals.eval_runner` with a valid API key.

## LangSmith Tracing

Enable tracing in `.env`:

```
LANGSMITH_API_KEY=your-key
LANGSMITH_PROJECT=safeagent
LANGSMITH_TRACING=true
```

Traces are named consistently: `safeagent.planner`, `safeagent.researcher`, `safeagent.synthesizer`, `safeagent.safety_arbitration`, `safeagent.escalation`.

<!-- TODO: Add LangSmith tracing screenshot here -->

## Docker / Cloud Run

```bash
docker build -t safeagent .
docker run -p 8080:8080 --env-file .env safeagent
```

Or with docker-compose:

```bash
docker-compose up --build
```

Cloud Run reads the `PORT` environment variable automatically.

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
safeagent/
├── main.py                 # FastAPI entry point
├── api/                    # REST routes and schemas
├── agents/                 # Five pipeline agents
├── graph/                  # LangGraph state and pipeline
├── retrieval/              # FAISS vector store + web search
├── safety/                 # Constitution, scorer, logger
├── evals/                  # Adversarial eval suite
├── observability/          # LangSmith tracing
└── tests/                  # pytest suite
```

## Tech Stack

- **Python 3.11+** — Runtime
- **LangGraph** — Multi-agent graph orchestration
- **FastAPI** — REST API layer
- **Gemini (gemini-2.5-flash)** — All LLM calls via Google GenAI SDK
- **FAISS** — Vector store for retrieval
- **Sentence Transformers** — Embeddings (all-MiniLM-L6-v2)
- **LangSmith** — Tracing and observability
- **RAGAS** — RAG evaluation
- **Pydantic v2** — Structured outputs and safety scores
- **DuckDuckGo Search** — Async web retrieval

## License

MIT
