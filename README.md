# 🔍 VC Analyst

A **7-agent AI pipeline** that converts a startup URL or pitch description into a structured investment memo in ~45 seconds. Built to replace the manual first-pass triage that consumes most of a seed analyst's day — making the process reproducible, auditable, and scalable.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AS230924/vc-analyst/blob/main/VC_Analyst_Colab.ipynb)
![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![xAI Grok](https://img.shields.io/badge/LLM-xAI%20Grok-black?logo=x&logoColor=white)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![Evals](https://img.shields.io/badge/Evals-20%20golden%20cases-green)

<!-- Add a screenshot: save to docs/screenshot.png and uncomment the line below -->
<!-- ![VC Analyst UI](docs/screenshot.png) -->

---

## The Problem

Seed analysts review 200–500 startups per year. Most of that time is first-pass triage:

- **Subjective** — two analysts score the same deck differently with no shared framework
- **No audit trail** — decisions can't be reviewed, compared, or improved over time
- **Unscalable** — adding headcount doesn't fix an unstructured process

VC Analyst replaces that pass with a deterministic 7-step pipeline, a 12-point scoring formula, and a ranked comparison table — so every decision is explainable and repeatable.

---

## Pipeline

```
Input: URL  ──or──  one-paragraph description
        │
        ▼
[1] Research        httpx scrape + BeautifulSoup extraction
                    ↳ optional: Playwright browser + DuckDuckGo (USE_BROWSER_RESEARCH=1)
        │
        ▼
[2] Classify        map to 1 of 9 AI stack layers  (e.g. "AI Developer Platforms")
        │
        ▼
[3] Evaluate        score 12 binary criteria — each is 0 or 1, never fractional
        │
        ▼
[4] Wrapper Risk    detect LOW / MEDIUM / HIGH API-wrapper risk → score penalty
        │
        ▼
[5] Score           deterministic formula — no LLM in this step
                    Final Score = Base (0–12) + Layer Adjustment + Wrapper Penalty
        │
        ▼
[6] Verdict         Ignore / Weak Signal / Watch / Strong Opportunity + Key Insight
        │
        ▼
[7] Nuance          TAM estimate · top risks · moat analysis · competitive landscape
                    + investment memo  ← only generated for Watch & Strong Opportunity
```

Each step is an isolated agent in `vc_analyst/agents/`. Step 5 has **no LLM** — the same inputs always produce the same score, making regressions catchable by the eval suite.

---

## 12-Point Scoring Framework

Every startup is evaluated on 12 binary criteria. Each scores **0 or 1** — binary scoring forces a thesis on every dimension and prevents grade inflation.

| # | Criterion | What earns a 1 |
|---|-----------|----------------|
| 1 | **Market Size** | TAM > $1B with identifiable customers and real spending power |
| 2 | **Market Growth** | >20% YoY growth, or market newly created by AI / regulatory tailwind |
| 3 | **Problem Severity** | Measurable pain: financial loss, regulatory risk, or ops inefficiency |
| 4 | **Clear Wedge** | Specific narrow use case to dominate before expanding |
| 5 | **Unique Insight** | Non-obvious view; something incumbents are structurally missing |
| 6 | **Data Moat** | Proprietary data accumulates and compounds as the product scales |
| 7 | **Workflow Lock-in** | Deeply embedded in customer workflows; high switching cost |
| 8 | **Distribution Advantage** | Existing community, partnership, PLG motion, or captive user base |
| 9 | **Network Effects** | Value increases with more users, data, or participants |
| 10 | **Platform Potential** | Credible path to SDK, marketplace, or ecosystem |
| 11 | **Competition Intensity** | Fragmented market, early stage, or clearly defensible niche |
| 12 | **Founder Advantage** | Prior exits, rare domain access, or proprietary technical edge |

### Scoring formula

```
Final Score = Base (0–12) + Layer Adjustment + Wrapper Penalty
```

**Layer adjustments** encode a VC prior — infrastructure compounds; apps churn:

| Layer | Adj | Layer | Adj |
|-------|-----|-------|-----|
| Foundation Models | +2 | AI Developer Platforms | +1 |
| Compute Infrastructure | +2 | AI Data Platforms | +1 |
| Model Infrastructure | +2 | AI Security / Governance | +1 |
| Vertical AI | 0 | AI Applications | −1 |
| | | AI Agents / Automation Platforms | −1 |

**Wrapper penalty:** LOW → 0 · MEDIUM → −1 · HIGH → −2

**Verdicts:** 0–4 Ignore · 5–7 Weak Signal · 8–9 Watch · 10+ Strong Opportunity

---

## Evaluations

Most AI tools ship without evals. VC Analyst has a structured evaluation harness with a **20-case golden dataset** across 8 categories and 3 difficulty levels (easy / medium / hard).

### Two evaluation phases

| Phase | Method | What it catches |
|-------|--------|----------------|
| **Deterministic** | Exact / range match | Wrong layer, out-of-range score, wrong verdict, pipeline crash |
| **Hallucination check** | Field + math validation | Empty outputs, non-binary criterion scores, broken score arithmetic |
| **LLM-as-Judge** | Grok 1–5 rubric | Shallow rationale, fabricated TAM, vague insight, thin investment memo |

The two phases are complementary: deterministic checks catch structural regressions; LLM-as-Judge catches quality regressions.

### Run the eval suite

```bash
python run_evals.py                              # all 20 cases
python run_evals.py --max 5                      # quick smoke test
python run_evals.py --category "Vertical AI" --difficulty hard
python run_evals.py --no-quality                 # skip LLM judge (faster, cheaper)
python run_evals.py --list                       # print all case IDs and exit
python run_evals.py --output report.txt          # save report to file
```

Exit code `2` if overall pass rate drops below 70% — safe to wire into CI.

---

## Observability

Every LLM call is traceable. Two options:

| | Phoenix by Arize | Logfire by Pydantic |
|-|-----------------|---------------------|
| **Best for** | Local development | Colab / cloud / CI |
| **UI** | `localhost:6006` (self-hosted) | [logfire.pydantic.dev](https://logfire.pydantic.dev) |
| **Setup** | `PHOENIX_ENABLED=1` in `.env` | Run Cell 5 in the notebook |
| **Free tier** | Unlimited (self-hosted) | 10M spans/month |

Both capture per-call: prompt · response · token counts · latency · model name.

---

## Quick Start

### Local

```bash
git clone https://github.com/AS230924/vc-analyst.git
cd vc-analyst

pip install -r requirements.txt

cp .env.example .env
# Edit .env — add your XAI_API_KEY

python app.py
# → http://localhost:7860
```

### Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AS230924/vc-analyst/blob/main/VC_Analyst_Colab.ipynb)

1. Add `XAI_API_KEY` to Colab Secrets (🔑 icon in the left sidebar)
2. **Runtime → Run all**
3. Click the Gradio public link that appears after the last cell

---

## Configuration

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `XAI_API_KEY` | ✅ | — | xAI Grok — primary LLM ([console.x.ai](https://console.x.ai)) |
| `GROK_MODEL` | | `grok-4-1-fast-reasoning` | Model name override |
| `ANTHROPIC_API_KEY` | | — | Claude fallback if Grok fails |
| `USE_BROWSER_RESEARCH` | | `0` | Set `1` for Playwright + DuckDuckGo research |
| `PHOENIX_ENABLED` | | `0` | Set `1` to launch Phoenix at `localhost:6006` |
| `LOGFIRE_TOKEN` | | — | Cloud traces ([logfire.pydantic.dev](https://logfire.pydantic.dev)) |

---

## Project Structure

```
vc-analyst/
├── app.py                         # Entry point (Gradio)
├── run_evals.py                   # Eval CLI
├── VC_Analyst_Colab.ipynb         # 7-cell Google Colab notebook
├── requirements.txt
├── .env.example
└── vc_analyst/
    ├── agents/
    │   ├── researcher.py          # Step 1 — httpx + BeautifulSoup
    │   ├── browser_researcher.py  # Step 1 alt — Playwright + DuckDuckGo
    │   ├── classifier.py          # Step 2 — 9-layer AI stack classification
    │   ├── evaluator.py           # Step 3 — 12-point binary scoring
    │   ├── wrapper_detector.py    # Step 4 — wrapper risk detection
    │   ├── scorer.py              # Step 5 — deterministic formula (no LLM)
    │   ├── verdict.py             # Step 6 — verdict + key insight
    │   └── nuance.py              # Step 7 — TAM · risks · moat · investment memo
    ├── core/
    │   ├── pipeline.py            # 7-step orchestrator with OpenTelemetry spans
    │   ├── llm_client.py          # Grok primary, Claude fallback
    │   └── tracer.py              # Phoenix / OTel setup
    ├── evals/
    │   ├── runner.py              # Eval orchestrator
    │   ├── evaluators.py          # Deterministic + hallucination checks
    │   ├── judge.py               # LLM-as-Judge (4 quality dimensions, 1–5 rubric)
    │   └── golden_dataset.json    # 20 hand-labelled test cases
    ├── config/
    │   └── frameworks.py          # 12 criteria · 9 layers · scoring constants
    └── models/
        └── schemas.py             # Pydantic v2 output schemas
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Primary LLM | xAI Grok (`grok-4-1-fast-reasoning`) |
| Fallback LLM | Anthropic Claude (`claude-haiku-4-5`) |
| Web UI | Gradio 4+ with custom CSS |
| Data validation | Pydantic v2 |
| HTTP scraping | httpx + BeautifulSoup4 |
| Browser research | Playwright + DuckDuckGo *(optional)* |
| Tracing — local | Phoenix by Arize + OpenTelemetry |
| Tracing — cloud | Logfire by Pydantic |
