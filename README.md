# Job Scout Agent

A resume-grounded job search agent that finds jobs, explains why they matter, and helps
you improve your odds of landing interviews. It parses your resume, searches multiple job
providers, and scores each posting against your background with structured evidence
(matched skills, missing hard requirements, and blocker risks).

Built on [Google ADK](https://google.github.io/adk-docs/) with a FastAPI backend and a
Next.js frontend.

## Features

- **Resume parsing** — extracts a structured profile from PDF/DOCX/text resumes, with an
  optional LLM parser (NVIDIA NIM, Groq, OpenAI, or Gemini via LiteLLM).
- **Job search** — pluggable providers including Apify (Indeed), Adzuna, and BrowserAct.
- **Resume-aware scoring** — ranks jobs with a fit verdict, reasons to apply, matched
  evidence, and blocker detection.
- **Fallback LLM support** — explicit model selection with automatic fallback.
- **Web UI** — a Next.js chat frontend for interacting with the agent.

## Project structure

```
main.py            FastAPI server entry point (Google ADK app)
job_scout/         The agent package
  agent.py           Root LlmAgent definition and tools wiring
  tools.py           Agent tools (search, score, resume handling)
  scoring.py         Resume-to-job match scoring
  domain_models.py   Typed models (JobPosting, ResumeProfile, JobMatchScore, ...)
  resume_*.py        Resume parsing and support
  search_*.py        Job search providers and support
  model_config.py    Model resolution and LiteLLM configuration
  fallback_llm.py    Fallback LLM wrapper
frontend/          Next.js 15 + React 19 chat UI
tests/             Test suite
ROADMAP.md         Product roadmap
```

## Requirements

- Python 3.10+
- Node.js 18+ (for the frontend)

## Setup

1. **Clone and install Python dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment**

   Copy the example env file and fill in the keys you want to use:

   ```bash
   cp .env.example .env
   ```

   At minimum, configure one model provider (Groq, NVIDIA NIM, or Gemini) and, for job
   search, a provider such as `APIFY_TOKEN`, Adzuna, or BrowserAct. See `.env.example`
   for all supported options.

## Running

**Backend** (FastAPI on port 8080 by default):

```bash
python main.py
```

**Frontend** (Next.js dev server):

```bash
cd frontend
npm install
npm run dev
```

## Testing

```bash
pytest
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features across resume tailoring, application
asset generation, skill-gap guidance, multi-source aggregation, and workflow tracking.
