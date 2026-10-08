# AI Resume Screening & Ranking System

A production-minded CLI pipeline that ingests a folder of resumes, applies a
**deterministic hard eligibility filter** (Python evidence **and** AI/agentic
evidence), scores the survivors on a transparent 100-point model, enriches them
with public GitHub activity (and an optional LLM pass), and writes a ranked,
fully explainable `output/results.json`.

- 50 resumes in, 50 resumes accounted for: parsed / failed / duplicate counts
  are reported, never silently dropped.
- 39 candidates eligible, 11 rejected with explicit reasons on the supplied
  resume set.
- Every point is traceable to a line of the resume (`score_breakdown` +
  `evidence`), so a reviewer can audit any rank.

## Installation

```bash
git clone https://github.com/devarasasank31/Kasparro-Assessment.git
cd Kasparro-Assessment

pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # runtime + pytest
```

Runtime dependencies are intentionally tiny: `pdfplumber` (PDF + DOCX/TXT text),
`pydantic` (schemas/validation) and `requests` (LLM + GitHub HTTP).

## Running the Application

```bash
python main.py --input ./resumes --output ./output/results.json
```

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--input DIR` | Folder of resumes (`.pdf` required, `.txt`/`.docx` supported) |
| `--output PATH` | Where to write `results.json` |
| `--report --top 10` | Print a terminal table of the top candidates |
| `--limit N` | Process only the first N files (fast smoke test) |
| `--no-llm` | Skip the model pass entirely |
| `--no-github` | Skip GitHub enrichment (offline / rate-limited runs) |
| `--no-cache` | Ignore and do not write `.cache/` (forces fresh API calls) |
| `--verbose` | Debug logging |
| `--version` | Print version |

Exit codes: `0` success, `2` input directory missing / bad arguments, `1`
unexpected failure.

Example with a report:

```bash
python main.py --input ./resumes --output ./output/results.json --report --top 10
```

## Environment Variables

Copy `.env.example` to `.env`. **Every variable is optional** — the pipeline
runs fully offline without them.

| Variable | Meaning |
| --- | --- |
| `LLM_API_KEY` | Enables the LLM pass. Empty = deterministic-only (recorded honestly in the output). |
| `LLM_PROVIDER` | `openai` \| `anthropic` \| `ollama` \| `echo` (echo = local echo model for tests) |
| `LLM_MODEL` | e.g. `gpt-4o-mini` |
| `LLM_BASE_URL` | Custom gateway / local server |
| `GITHUB_TOKEN` | Raises the GitHub rate limit from 60 to 5000 req/h. Never hard-coded. |
| `GITHUB_MAX_WORKERS` | Bounded enrichment pool (default 4) |
| `USE_CACHE` | `0` to disable the run cache |
| `LOG_LEVEL` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |

## Pipeline Stages

```
ingest -> parse -> extract -> hard filter -> LLM (optional) -> score
       -> GitHub enrichment -> final score -> rank -> results.json
```

Each stage is isolated: a malformed PDF, a failed model call or a rate-limited
GitHub request is recorded in `failures` / `parse_issues` and the batch keeps
going. Stage wall-clocks land in `summary.stages_ms`.

## Hard Filter (Eligibility)

Both conditions must hold, decided by deterministic rules — never by the model:

1. **Python evidence** — Python appears as a genuine skill, project technology,
   work technology or implementation language.
2. **AI/agentic evidence** — at least one meaningful AI/LLM/RAG/agentic
   implementation. Terms are tiered:

   | Tier | Meaning | Example |
   | --- | --- | --- |
   | strong | The term itself proves an implementation | LangChain, LangGraph, RAG pipeline, embeddings, tool-calling, eval harness |
   | moderate | Needs project/experience context or an implementation verb | OpenAI/LLM providers, scikit-learn, PyTorch, NLP, CV |
   | weak | Needs an implementation verb nearby ("built…") | "AI-powered", "chatbot", "generative AI" |

   Mentions in skills/certification sections alone are **not** evidence.

JavaScript/Java/React-only profiles are rejected; JS/React **plus** Python+AI is
accepted. Rejections always carry reasons, e.g.:

```json
{"candidate": "Meera Shah", "eligible": false,
 "rejection_reasons": ["No evidence of Python stack", "No AI/agentic project evidence"],
 "matched_skills": ["Java", "React", "Spring Boot"]}
```

## Scoring Model (100 points)

| Category | Weight | What earns it |
| --- | ---: | --- |
| AI / Agentic / RAG project depth | 40 | Agents, RAG, tool calling, state, orchestration, evaluation, real business logic |
| Python & backend engineering | 30 | Python, FastAPI/Django/Flask, async, PostgreSQL/Redis, API craft, typed & tested code |
| Cloud / deployment / full stack | 15 | GCP/AWS/Azure, Docker/K8s, CI/CD, end-to-end systems |
| GitHub activity | 10 | Recent public engineering activity (capped at 10) |
| Engineering depth signals | 5 | Testing, observability, queues, concurrency, failure handling |

Rules that keep it honest:

- **Evidence weighting** — a match inside a project/internship description scores
  the full value; the same match inside a skills list scores **half**.
- **Project-quality penalties** — 5–15 point deduction when an "AI project" is a
  thin wrapper around an LLM/API call, or a tutorial listed with no
  implementation detail (`SHALLOW_PROJECT_PENALTY = 10`, cap
  `MAX_PROJECT_PENALTY = 15`).
- **Fit tiers** — totals map onto `strong_fit` (≥80), `good_fit` (≥65),
  `possible_fit` (≥50), `weak_fit` (<50) for quick triage.
- Weights live in `config.SCORE_WEIGHTS` and are echoed into the run metadata.

## GitHub Enrichment

- Username is parsed from the resume (reserved paths like `/features` are
  ignored); no username → `github_status: "no_profile"`, no request made.
- Two lightweight calls per profile: `GET /users/{name}` and
  `GET /users/{name}/repos` (max 30 repos inspected).
- **Activity (0–5)**: recency of the newest push (≤30d = 3, ≤90d = 2, ≤180d = 1)
  + volume of repos pushed in the last 90 days (≥5 = 2, ≥2 = 1.5, 1 = 1).
- **Repositories (0–5)**: non-fork/non-archived ownership (≥10 = 2, ≥5 = 1.5,
  ≥3 = 1, else 0.5) + Python/AI relevance (≥3 = 2, ≥1 = 1) + maintenance in the
  last 180 days (≥2 = 1, ≥1 = 0.5).
- Failures (`404`, rate limit, network) are **statuses, not exceptions**: the
  batch continues, `failures.github` records the reason, and the candidate keeps
  screening with `github: 0`.
- Enrichment runs over a bounded worker pool (`GITHUB_MAX_WORKERS`, default 4).
  Once the rate limit is hit, remaining work short-circuits instead of hammering
  the API.
- Successful responses are cached in `.cache/github.json` (TTL 6 h) so re-runs
  are free.

## LLM Usage (optional)

- Provider-specific HTTP lives behind one adapter (`llm.HTTPChatClient`) with a
  `LLMClient` protocol; swapping providers is a one-line change.
- Responses are validated against Pydantic schemas (`LLMAnalysis`,
  `LLMProjectSummary`) — structured output only, malformed JSON = failure.
- The model may nudge the AI-depth score and replace the project summary, but it
  **never** decides eligibility, and its adjustment is bounded and reported as
  `llm_adjustment`.
- One failed call only affects one resume: it lands in `failures.llm` with the
  reason.
- Successful responses are cached in `.cache/llm.json` (TTL 24 h).

> The submitted `results.json` was produced **without** an LLM key in the
> environment, so it records `"llm_used": false` and scores are fully
> deterministic. Setting `LLM_API_KEY` is all that is needed to enable it.

## Output

`output/results.json` shape:

```json
{
  "run": {"tool_version": "1.0.0", "python": "3.11.1", "started_at": "...",
           "duration_ms": 1820.5,
           "settings": {"use_llm": false, "use_github": true, "weights": {"ai_project_depth": 40, "...": "..."}}},
  "summary": {"total_resumes": 50, "parsed": 50, "failed": 0, "duplicates": 0,
               "eligible": 39, "rejected": 11, "scored": 39,
               "llm_used": false, "github_enriched": 18, "github_failures": 1,
               "score_stats": {"mean": 64.64, "median": 68.0, "strong_fit": 9, "...": "..."},
               "stages_ms": {"ingest": 1450.0, "github": 820.3, "...": "..."},
               "status": "complete"},
  "parse_issues": [], "duplicates": [],
  "failures": {"extraction": [], "llm": [], "github": [], "scoring": []},
  "candidates": [
    {"rank": 1, "fit_tier": "strong_fit", "candidate_name": "...", "eligible": true,
     "total_score": 98.5,
     "score_breakdown": {"ai_project_depth": 40, "python_backend": 30, "cloud_fullstack": 15,
                          "github": 10, "engineering_depth": 3.5, "penalties": 0},
     "matched_skills": ["Python", "FastAPI", "LangGraph"],
     "project_summary": "...", "github_status": "ok", "github_summary": "...",
     "strengths": ["..."], "concerns": ["..."], "evidence": {"AI / agentic": [{"label": "...", "points": 8, "detail": "..."}]},
     "rejection_reasons": []},
    {"rank": null, "candidate_name": "Meera Shah", "eligible": false,
     "rejection_reasons": ["No evidence of Python stack", "No AI/agentic project evidence"]}
  ]
}
```

Ranked candidates come first, then unscored, then rejected — all 50 records are
always present.

## Caching & Reliability

- `.cache/github.json` / `.cache/llm.json` — file cache with per-namespace TTL,
  survives restarts, corrupt entries are treated as misses, never fatal.
- Per-resume isolation: a bad PDF, a bad extraction, a bad score all become
  recorded failures instead of exceptions.
- Duplicate files (identical SHA-256) are skipped and counted.
- HTTP calls use bounded retries with backoff for transient errors only.
- Offline by design: `--no-github` / `--no-llm` / `--no-cache` produce a fully
  local run.

## Testing

```bash
pytest                     # 95 tests
pytest tests/test_eligibility.py -v
```

Coverage includes eligibility edge cases, scoring rules and penalties,
extraction, PDF/TXT ingestion with a malformed file, GitHub failure/rate-limit
paths, cache semantics, concurrency bounds, CLI behaviour and an end-to-end
batch over synthetic resumes (`tests/fixtures/resumes/`).

## Project Structure

```
main.py                       entry point
src/resume_screening/
  config.py                   weights, thresholds, env/CLI settings
  parser.py                   discovery, dedupe, PDF/TXT/DOCX extraction
  extractor.py                name/email/skills/projects/experience
  skills.py                   vocabulary + alias matching
  eligibility.py              hard filter (deterministic)
  scoring.py                  100-point model with per-point evidence
  llm.py                      provider adapter + Pydantic schemas
  github.py                   public API client, scoring, worker pool
  cache.py                    TTL file cache
  ranking.py                  deterministic ordering, tiers, stats
  output.py                   JSON record shaping
  report.py                   terminal report
  pipeline.py                 stage orchestration + run metadata
  cli.py                      argparse interface
tests/                        116 unit/integration tests + fixtures
output/results.json           generated result for the supplied resume set
```

## Design Decisions

- **Hard filter outside the LLM.** Eligibility is a correctness requirement, not
  a judgement call, so it is regex/section based with tiered evidence and always
  returns the snippets that justified the decision. This makes rejections
  auditable and the run reproducible without credentials.
- **Deterministic scoring first, LLM second.** The 100-point model is a rule set
  over normalised resume corpora (projects + experience weighted higher than
  skills lists). The optional LLM pass can only nudge the AI-depth score within
  a bound and replace the project summary; its adjustment is printed in the
  output so a reviewer can see exactly what the model changed.
- **Explainability beats cleverness.** Every scored signal is a named rule with
  points and a human-readable `detail`, grouped into `evidence` per category.
  Penalties are simple, capped deductions rather than a learned weight — easy to
  argue with, easy to tune in `config.py`.
- **Failures are data, not exceptions.** Rate limits, 404s, malformed PDFs and
  model errors become statuses in the JSON with reasons. The batch always
  completes, which is what an interviewer's folder of 50 real resumes needs.
- **Cache the network, not the decision.** GitHub and LLM responses are cached
  (with TTLs) but scores are recomputed each run, so tuning weights never serves
  stale output.
- **Bounded concurrency.** GitHub lookups fan out over a small worker pool
  (default 4) with one session per thread and a shared stop-event for rate
  limits — enough parallelism to cut wall-clock, not enough to look like a
  scraper.
- **Tiny dependency surface.** `pdfplumber` + `pydantic` + `requests` keeps the
  system easy to install and review inside a short time box.

## If I Had More Time

1. **Semantic matching instead of keyword corpora** — embed project bullets and
   score against a curated "SDE intern with AI" reference profile to cut
   false positives from resume keyword stuffing.
2. **Calibration pass** — label 10–15 of these resumes by hand and fit the rule
   weights/thresholds to that gold set (plus a small eval harness that reports
   ranking agreement).
3. **Bounded async enrichment** — move GitHub/LLM calls to `asyncio` with a
   token-bucket rate limiter and a retry queue, so a 500-resume batch stays
   inside provider quotas with measurable throughput gains.
4. **FastAPI surface** — `POST /screen` (async job) and `GET /results` around
   the same pipeline, with the JSON contract already in place.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `github_status: "rate_limited"` | Unauthenticated quota (60/h) is spent; set `GITHUB_TOKEN` or wait for reset — `.cache/` keeps whatever succeeded. |
| `llm_used: false` | No `LLM_API_KEY` in the environment; the run is deterministic by design. |
| A resume shows up in `parse_issues` | File is corrupt/unsupported; it is counted, not fatal. |
| Scores changed after an edit | Expected — weights live in `config.SCORE_WEIGHTS`; `run.settings.weights` records what was used. |
