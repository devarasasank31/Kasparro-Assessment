# AI Resume Screening & Ranking System

CLI pipeline that ingests a folder of resume PDFs, applies a **deterministic hard
eligibility filter** (Python evidence **and** AI/agentic evidence), scores eligible
candidates on a transparent 100-point model, optionally enriches them with an LLM
and public GitHub activity, and writes a ranked, explainable `results.json`.

> Status: work in progress — sections are completed phase by phase.

## Installation

```bash
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # runtime + tests
```

## Running the Application

```bash
python main.py --input ./resumes --output ./output/results.json
```

## Environment Variables

Copy `.env.example` to `.env`. Every variable is optional — see `.env.example`.

## Testing

```bash
pytest
```
