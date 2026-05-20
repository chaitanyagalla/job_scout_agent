# Job Scout Agent — Complete Beginner's Learning Guide

> **Who this is for:** Someone who built this project with AI assistance and now wants to understand *every* line, *every* decision, and *every* concept from scratch.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The Big Picture — How Everything Fits Together](#2-the-big-picture)
3. [Core Concepts You Need to Know](#3-core-concepts)
4. [Project File Map](#4-project-file-map)
5. [main.py — The Server Explained](#5-mainpy)
6. [agent.py — The Brain Explained](#6-agentpy)
7. [domain_models.py — The Data Shapes](#7-domain_modelspy)
8. [model_config.py — Choosing the AI Model](#8-model_configpy)
9. [tools.py — The Tool Hub](#9-toolspy)
10. [search_tools.py — Finding Jobs](#10-search_toolspy)
11. [scoring.py — Matching Resume to Job](#11-scoringpy)
12. [litellm_compat.py — Making Different AIs Work Together](#12-litellm_compatpy)
13. [state_keys.py — Memory Between Messages](#13-state_keyspy)
14. [The Scoring Algorithm — Deep Dive](#14-scoring-algorithm-deep-dive)
15. [Why Not the Other Way? — Every Major Design Decision](#15-why-not-the-other-way)
16. [How a Real Conversation Flows](#16-how-a-real-conversation-flows)
17. [Environment Variables — What They All Mean](#17-environment-variables)
18. [Glossary of All Technical Terms](#18-glossary)

---

## 1. What Is This Project?

Job Scout is an **AI agent** — a program that can think, use tools, and hold a conversation — that helps people find jobs.

It can do two things:

```
WITHOUT a resume (Prompt-only mode)
  User: "Find me Python jobs in Hyderabad"
  Agent: [searches] → shows 5 jobs

WITH a resume (Resume-aware mode)
  User: [uploads resume] "Find me matching jobs"
  Agent: [reads resume] → [searches] → [scores each job] → shows ranked results
```

The key difference from a normal Google search for jobs is that this agent:
- Understands your **resume as structured data** (not just a PDF file)
- **Scores** every job against your resume (0–100 match score)
- **Explains** WHY a job is a good or bad match
- Remembers your resume across the entire conversation

---

## 2. The Big Picture

Here is how all the pieces connect:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER'S BROWSER / CLIENT                          │
│                  (sends messages, uploads resume PDF)                   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │  HTTP requests
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         main.py  (FastAPI Server)                       │
│              Listens on port 8080, routes messages to ADK               │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │  hands off to
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Google ADK (Agent Development Kit)                   │
│           Framework that manages: conversation loop, tool calls,        │
│           session state, artifact (file) storage                        │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │  runs
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       agent.py  (root_agent)                            │
│    The LLM agent with instructions + 9 tools registered                 │
│    Before each LLM call: add_runtime_hints() injects dynamic context    │
└──────────┬────────────────────────────────────────────┬─────────────────┘
           │ calls                                      │ calls
           ▼                                            ▼
┌─────────────────────────┐               ┌─────────────────────────────┐
│   RESUME TOOLS          │               │   JOB SEARCH TOOLS          │
│ ─────────────────────── │               │ ─────────────────────────── │
│ extract_resume_profile  │               │ search_jobs                 │
│ get_resume_status       │               │ filter_saved_jobs_by_exp    │
│ save_resume_profile     │               │ fetch_job_details           │
│ clear_resume_profile    │               │ score_job_match             │
│ score_job_match         │               │                             │
└──────────┬──────────────┘               └────────────┬────────────────┘
           │                                           │
           ▼                                           ▼
┌─────────────────────────┐               ┌─────────────────────────────┐
│   resume_support.py     │               │   search_support.py         │
│ ─────────────────────── │               │ ─────────────────────────── │
│ PDF/DOCX/text parsing   │               │ Apify (Indeed scraper)      │
│ LLM-based extraction    │               │ Adzuna REST API             │
│ Resume normalization    │               │ BrowserAct automation       │
└─────────────────────────┘               │ Demo data (no credentials)  │
                                          └─────────────────────────────┘
           │                                           │
           └──────────────────┬────────────────────────┘
                              ▼
                   ┌─────────────────────┐
                   │     scoring.py      │
                   │ ─────────────────── │
                   │ SKILL_ONTOLOGY      │
                   │ Weighted coverage   │
                   │ Title alignment     │
                   │ Experience check    │
                   │ Blocker detection   │
                   └─────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   Session State     │
                   │ ─────────────────── │
                   │ resume_profile      │
                   │ last_search_results │
                   │ extraction_method   │
                   └─────────────────────┘
```

---

## 3. Core Concepts

Before reading the code, you need to understand these ideas. Each one is explained as simply as possible.

### 3.1 What Is an AI Agent?

A normal AI chatbot just answers questions.

An **AI agent** can also *use tools* — it can call external functions (like "search jobs", "parse resume") in the middle of a conversation, get results back, and then continue the conversation with those results.

```
Normal chatbot:
  User → LLM → Answer

AI Agent:
  User → LLM → "I need to call search_jobs" → search_jobs() runs → LLM reads result → Answer
                      ↑                                                      ↓
                      └──────────── Tool call loop ──────────────────────────┘
```

### 3.2 What Is Google ADK?

ADK (Agent Development Kit) is Google's framework for building AI agents. Think of it as the "plumbing" — it handles:
- Managing the conversation history
- Routing tool calls to the right Python functions
- Storing files (artifacts) like uploaded resumes
- Persisting session state (memory between messages)
- Exposing the agent as a web API

You define the agent (the `root_agent` object), and ADK handles everything else.

### 3.3 What Is LiteLLM?

Different AI providers (Google Gemini, Groq, NVIDIA NIM, Anthropic Claude, OpenAI) all have slightly different APIs. **LiteLLM** is a library that gives them all the same interface — so your code doesn't need to change when you switch AI providers.

Think of it like a universal power adapter: different countries have different outlet shapes, but the adapter makes one plug fit all of them.

### 3.4 What Is a Pydantic Model?

Pydantic is a Python library that enforces data types and validates data. When you define a Pydantic model, you're saying "this data MUST look exactly like this shape."

```python
# Without Pydantic — anything can go in, anything can come out
job = {"title": "Engineer", "salary": "unknown"}  # no validation

# With Pydantic — it validates and rejects bad data
class JobPosting(BaseModel):
    title: str = ""
    salary: float = 0.0   # must be a number, not a string
```

If you try to put a string where a float is expected, Pydantic raises an error immediately instead of letting the bug hide and cause problems later.

### 3.5 What Is Session State?

When you have a conversation with this agent:
- Turn 1: "I'm a React developer"
- Turn 2: "Find me jobs"

The agent needs to REMEMBER what you said in Turn 1 when it processes Turn 2. That "memory" is called **session state** — a dictionary that persists throughout the conversation.

```
Session State (like a dictionary that persists across messages):
{
  "resume_profile": { "name": "Alice", "skills": ["React", "Node.js"] },
  "last_search_results": { "role": "Frontend Developer", "jobs": [...] }
}
```

### 3.6 What Is a Tool in Agent Context?

A tool is just a Python function that the agent can call. The agent decides WHEN to call it based on what the user asks.

```python
def search_jobs(role, location, max_results=5):
    # ... does the search ...
    return {"jobs": [...]}

# The agent's instructions tell it:
# "When user asks to find jobs, call search_jobs()"
```

The agent reads the function's **docstring** (the text description inside the function) to understand what the tool does and when to use it.

### 3.7 What Is a Callback?

A callback is a function that runs automatically at a specific point — like a hook. In this project, `add_runtime_hints` is a **before-model callback**. It runs automatically right before every message is sent to the AI model.

```
User sends message
        ↓
add_runtime_hints() runs  ← CALLBACK (automatic)
        ↓
Message + hints sent to LLM
        ↓
LLM responds
```

### 3.8 What Is Deterministic Scoring?

The word "deterministic" means "given the same input, always produces the same output." No randomness.

The scoring in this project does NOT use machine learning or AI to score jobs. Instead it uses hand-crafted rules:
- Count how many required skills you have: +score
- Title matches: +score
- Missing required skills: -score

This is **deterministic** — if your resume has React and the job requires React, the score goes up by exactly the same amount every time. You can predict, debug, and test it precisely.

**Why not use AI for scoring?** Explained in Section 15.

---

## 4. Project File Map

```
job-sout-agent/
│
├── main.py                 ← Entry point. Starts the web server.
├── requirements.txt        ← List of libraries this project needs.
├── .env                    ← Your secret API keys (never commit this!)
├── .env.example            ← Template showing what keys are needed.
│
└── job_scout/              ← The main Python package (the actual agent code)
    ├── __init__.py         ← Makes this folder a Python package.
    ├── agent.py            ← THE BRAIN. Defines the root_agent.
    ├── state_keys.py       ← Constants for session state dictionary keys.
    ├── domain_models.py    ← Pydantic data shapes (resume, job, score).
    ├── tools.py            ← Central hub that exports all tools.
    ├── model_config.py     ← Logic for picking which AI model to use.
    ├── litellm_compat.py   ← Patches to make Groq/NVIDIA work with ADK.
    ├── search_tools.py     ← search_jobs() and filter tools (public API).
    ├── search_support.py   ← Actual job search implementations (private).
    ├── resume_tools.py     ← Resume tool functions (public API).
    ├── resume_support.py   ← Actual resume parsing logic (private).
    ├── scoring.py          ← Deterministic resume-vs-job scoring math.
    └── evaluation.py       ← Test framework for scoring quality.
```

The pattern of `search_tools.py` + `search_support.py` (and similarly `resume_tools.py` + `resume_support.py`) is intentional:
- The **tools** file has the public functions the agent calls. It's the "clean interface."
- The **support** file has all the messy implementation details, hidden from the agent.

---

## 5. main.py — The Server Explained

```python
from google.adk.cli.fast_api import get_fast_api_app
```

**What this does:** Imports a function from Google ADK that creates a complete FastAPI web application — routes, WebSocket handling, artifact storage, session management — all in one call.

**Why FastAPI?** FastAPI is a modern Python web framework that is fast, automatically validates inputs, and generates API documentation. The alternative would be Flask or Django — FastAPI was chosen because ADK's tooling integrates with it directly.

```python
try:
    import google.cloud.logging
    google.cloud.logging.Client().setup_logging()
except Exception:
    pass
```

**What this does:** Tries to connect to Google Cloud's logging service. If it fails (because you're running locally without Google Cloud credentials), it silently continues. The `try/except` makes this optional — it only works in production on Google Cloud.

**Why `pass` instead of logging the error?** Because at startup, the logging system itself might not be ready yet. Logging a logging error would be circular.

```python
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
```

**What this does:** Gets the absolute path of the directory that contains `main.py`. This is important because ADK scans this directory to find the agent package (the `job_scout/` folder).

**Why `abspath(__file__)`?** `__file__` is the path to the current Python file. But it might be a relative path like `./main.py`. `abspath()` converts it to a full absolute path like `C:/Users/Dell/.../main.py`. This makes the path reliable regardless of where you run the script from.

```python
app = get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_service_uri=os.getenv("SESSION_SERVICE_URI"),
    artifact_service_uri=os.getenv("ARTIFACT_SERVICE_URI"),
    allow_origins=["*"],
    web=False,
    use_local_storage=True,
)
```

**Breaking down each parameter:**

| Parameter | What it does | When it matters |
|-----------|-------------|-----------------|
| `agents_dir` | Where ADK scans for agents | Always — this is how ADK finds `job_scout/` |
| `session_service_uri` | Where to store session data | Production (Cloud Firestore). Locally: `None` uses in-memory storage |
| `artifact_service_uri` | Where to store uploaded files | Production (Cloud Storage). Locally: uses local disk |
| `allow_origins=["*"]` | CORS — who can send requests | `"*"` means any website can talk to this API. In production you'd restrict this |
| `web=False` | Should ADK serve a built-in web UI? | `False` because you're building your own frontend |
| `use_local_storage=True` | Use local files when no URIs given | For local development without cloud setup |

```python
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

**What `if __name__ == "__main__"` means:** Python sets `__name__` to `"__main__"` only when the file is run directly (like `python main.py`). If another file imports `main.py`, this block is skipped. This pattern prevents the server from starting accidentally when the file is imported.

**What is uvicorn?** Uvicorn is an ASGI (Asynchronous Server Gateway Interface) server — the thing that actually listens on a network port and handles incoming HTTP connections. FastAPI apps need an ASGI server to run.

**Why `host="0.0.0.0"`?** This makes the server listen on ALL network interfaces, not just `localhost`. This is required for deployment where the server needs to be accessible from outside the machine.

---

## 6. agent.py — The Brain Explained

This is the most important file. Everything starts here.

### 6.1 Imports

```python
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.tools import load_artifacts
from google.genai import types
```

| Import | What it is |
|--------|-----------|
| `LlmAgent` | The main agent class — wraps an LLM + tools + instructions |
| `Gemini` | Native Google Gemini model integration |
| `LiteLlm` | LiteLLM-based integration for Groq, NVIDIA, etc. |
| `LlmRequest` | Object representing a single request to the LLM |
| `load_artifacts` | Built-in ADK tool to load uploaded files (resumes) |
| `types` | Google GenAI types — for `Part`, `GenerateContentConfig`, etc. |

### 6.2 `load_dotenv()` and `patch_litellm_tool_call_id_repair()`

```python
load_dotenv()
patch_litellm_tool_call_id_repair()
```

`load_dotenv()` reads your `.env` file and puts every `KEY=VALUE` into the environment (accessible via `os.getenv()`). This runs at import time — before anything else — so API keys are available immediately.

`patch_litellm_tool_call_id_repair()` applies monkey-patches (runtime modifications) to fix bugs in how LiteLLM formats messages. Explained in Section 12.

### 6.3 `_attachment_hint_from_part()` Function

```python
def _attachment_hint_from_part(part: types.Part, fallback_index: int) -> str:
    if part.file_data:
        return (
            part.file_data.display_name
            or part.file_data.file_uri
            or part.file_data.mime_type
            or f"attachment_{fallback_index}"
        )
    if part.inline_data:
        return (
            part.inline_data.display_name
            or part.inline_data.mime_type
            or f"attachment_{fallback_index}"
        )
    return f"attachment_{fallback_index}"
```

**What it does:** Given an LLM "Part" (a piece of a message — could be text, file, or inline data), this extracts a human-readable name for it.

**Why the chain of `or` operators?** Because different file types have different metadata. A Google Drive file might have a `file_uri`. A directly uploaded file might only have a `mime_type`. The `or` chain tries each option in order and takes the first one that isn't empty. `fallback_index` is the last resort ("attachment_0", "attachment_1", etc.).

This function is a helper — it's only used by `_sanitize_unsupported_file_parts()`.

### 6.4 `_sanitize_unsupported_file_parts()` Function

```python
def _sanitize_unsupported_file_parts(llm_request: LlmRequest) -> None:
    model_name = (getattr(llm_request, "model", None) or resolve_model_name() or "").strip()
    if not model_name.startswith("nvidia_nim/"):
        return
```

**What it does:** NVIDIA NIM models can't handle file attachments (PDFs, images). This function converts any file attachment into a text placeholder like `[Attached file: resume.pdf]` when the model is NVIDIA NIM.

**Why only NVIDIA NIM?** Gemini can natively read PDFs and images. NVIDIA NIM's API doesn't support binary file content. So when a user uploads a resume and we're using NVIDIA NIM, we can't send the file — we replace it with a text hint, and resume parsing falls back to a different method.

```python
    for content in contents:
        for part in parts:
            if part.inline_data or part.file_data:
                hint = _attachment_hint_from_part(part, attachment_index)
                rewritten_parts.append(types.Part.from_text(text=f"[Attached file: {hint}]"))
                changed = True
                continue
            rewritten_parts.append(part)
```

**The pattern here:** Loop through every "content" (message), loop through every "part" (piece of that message). If a part is binary file data, replace it with a text description. If it's already text, keep it as-is.

### 6.5 `add_runtime_hints()` — The Dynamic Prompt Injector

This is one of the most clever parts of the project.

```python
async def add_runtime_hints(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
```

**Why `async`?** Because listing artifacts (`await callback_context.list_artifacts()`) is an async operation — it might need to check remote storage. Python's `async/await` allows this to happen without blocking the server.

```python
    profile = callback_context.state.get(RESUME_PROFILE_STATE_KEY)
    last_search = callback_context.state.get(LAST_SEARCH_RESULTS_STATE_KEY)
    artifact_names: list[str] = []
    try:
        artifact_names = await callback_context.list_artifacts()
    except Exception:
        artifact_names = []
```

**What this reads:**
- `profile`: Is there already a parsed resume in memory?
- `last_search`: Was there a previous job search?
- `artifact_names`: Are there any uploaded files?

**Why `try/except` around `list_artifacts()`?** Because in some configurations (especially local dev), artifact listing might not work. Rather than crashing the agent, we fall back to an empty list.

```python
    if profile:
        llm_request.append_instructions([
            "A normalized resume profile already exists in session state.",
            f"Resume role titles: {', '.join(profile.get('role_titles', []))}.",
            ...
        ])
```

**The key insight:** The base instructions in `root_agent` are static (same every time). But `add_runtime_hints()` ADDS more instructions right before each LLM call, based on the current state. This is called **dynamic prompting**.

**Why not just put everything in the static instructions?** Because the static instructions can't know whether a resume is loaded or not — that's session-specific information. Dynamic hints inject this context at the right moment.

**Visualization of what happens:**

```
Static Instructions (always present):
  "You are Job Scout. You support two modes..."
         +
Dynamic Hints (added by add_runtime_hints each time):
  "A resume is loaded. Skills: React, Node.js. Titles: Full Stack Developer."
         =
Final prompt sent to LLM (combination of both)
```

### 6.6 `_build_model()` and `GEMINI_MODEL`

```python
def _build_model():
    model_name = resolve_model_name()
    if uses_litellm(model_name):
        logger.info("Using LiteLLM provider model: %s", model_name)
        return LiteLlm(model=model_name)

    logger.info("Using native Gemini model: %s", model_name)
    return Gemini(model=model_name)

GEMINI_MODEL = _build_model()
```

**What this does:** Decides which AI model object to create based on environment variables. If the model name starts with `groq/` or `nvidia_nim/`, it uses LiteLLM as the bridge. Otherwise it uses Google's native Gemini integration.

**Why called `GEMINI_MODEL` even when it's not Gemini?** This is a naming artifact — the variable was originally always Gemini. When multi-provider support was added, the variable name wasn't changed. It now holds whichever model is configured.

**Why build it at module level (not inside a function)?** Because the agent is instantiated once when the server starts. If you built the model inside `root_agent`'s constructor call every time a request came in, you'd be doing expensive setup repeatedly.

### 6.7 `root_agent` — The Agent Definition

```python
root_agent = LlmAgent(
    name="job_scout",
    model=GEMINI_MODEL,
    description="An AI job-search assistant...",
    instruction="""...""",
    before_model_callback=add_runtime_hints,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
    ),
    tools=[
        load_artifacts,
        get_resume_status,
        extract_resume_profile_from_artifact,
        save_resume_profile,
        clear_resume_profile,
        search_jobs,
        fetch_job_details,
        filter_saved_jobs_by_experience,
        score_job_match,
    ],
)
```

**Breaking down each parameter:**

| Parameter | What it does |
|-----------|-------------|
| `name` | The agent's identifier in ADK. Must match the package folder name (`job_scout/`). |
| `model` | The LLM that powers this agent |
| `description` | Short summary shown in API discovery |
| `instruction` | The system prompt — the agent's "personality" and rules |
| `before_model_callback` | Function that runs before every LLM call (our `add_runtime_hints`) |
| `generate_content_config` | Settings for generation. `temperature=0.2` means very deterministic output |
| `tools` | List of Python functions the agent can call |

**Why `temperature=0.2`?** Temperature controls how "creative" (random) the LLM is. For a job assistant:
- High temperature (0.8–1.0): Might invent job listings, make up scores
- Low temperature (0.2): Stays factual, follows instructions precisely

This agent needs to be reliable, not creative. Lower is better.

### 6.8 The System Prompt (Instructions) — Line by Line

The `instruction` string is 323 lines. Here are the most important parts explained:

```
You support two modes:
1. Resume-aware mode — use resume for personalized matching
2. Prompt-only mode — search without a resume
```
**Why two modes?** Not everyone has a resume to upload. The agent should be useful either way.

```
Treat short follow-ups like "yes", "yeah", "go ahead" as confirmation
of the last clear action you offered.
```
**Why this instruction?** Without it, the LLM might ask "Do you want me to search for jobs?" after the user already said "yes". This prevents unnecessary back-and-forth.

```
Never call extract_resume_profile_from_artifact, get_resume_status,
and search_jobs as one batched step.
```
**Why this instruction?** Some LLMs try to call all tools at once in parallel to be faster. But resume extraction MUST complete before job search — you can't search based on a resume you haven't parsed yet. This enforces the correct order.

```
Treat "entry level", "fresher", "fresh graduate" as min_years=0, max_years=1
```
**Why?** These phrases mean the same thing but aren't numbers. The agent translates natural language into the parameters the search tool needs.

---

## 7. domain_models.py — The Data Shapes

This file defines all the data structures used throughout the project.

### Why Pydantic?

```python
# Without Pydantic — easy to make mistakes
job = {"titlE": "Engineer", "companY": "Google"}  # typos, no validation

# With Pydantic — strict, validated
class JobPosting(BaseModel):
    title: str = ""  # must be a string
    company: str = ""
```

Any code that creates a `JobPosting` must follow this schema. If it doesn't, Pydantic raises an error immediately — you catch bugs early.

### ResumeProfile

```python
class ResumeProfile(BaseModel):
    candidate_name: str = ""
    professional_summary: str = ""
    role_titles: list[str] = Field(default_factory=list)
    core_skills: list[str] = Field(default_factory=list)
    additional_skills: list[str] = Field(default_factory=list)
    years_experience: Optional[float] = Field(default=None, ge=0)
    preferred_locations: list[str] = Field(default_factory=list)
    work_preferences: list[str] = Field(default_factory=list)
    notable_projects: list[str] = Field(default_factory=list)
    resume_source: str = "uploaded_resume"
```

**Key decisions:**

- `Field(default_factory=list)` instead of `= []`: In Python, using `= []` as a default is a bug — all instances share the SAME list object. `default_factory=list` creates a NEW empty list for each instance.
- `Optional[float] = Field(default=None, ge=0)`: `years_experience` can be `None` (unknown) OR a non-negative float. `ge=0` means "greater than or equal to 0" — Pydantic validates this.
- `resume_source`: Tracks where the resume came from (uploaded file, manual save, etc.) for debugging.

### JobPosting

```python
class JobPosting(BaseModel):
    id: Optional[str] = None
    title: str = ""
    company: str = ""
    location: str = ""
    snippet: str = ""
    url: Optional[str] = None
    matched_role: Optional[str] = None
    experience_min_years: Optional[float] = None
    experience_max_years: Optional[float] = None
    experience_evidence: Optional[str] = None
    description: Optional[str] = None
    fit_verdict: Optional[str] = None
    reason_to_apply: Optional[str] = None
    blocker_risk: Optional[str] = None
```

`Optional[str] = None` means the field can be absent. Real job listings don't always have all fields — some have no URL, some have no experience requirement. Optional fields allow partial data.

### JobMatchScore

```python
class JobMatchScore(BaseModel):
    score: int                              # 0-100
    signals: JobMatchSignals                # breakdown of score components
    matched_skills: list[str]               # skills you have that job wants
    missing_skills: list[str]               # skills job wants that you lack
    explanation: str = ""                   # human-readable explanation
    evidence: JobMatchEvidence              # detailed per-category evidence
    blockers: JobMatchBlockers              # things that might get you rejected
    fit_verdict: str = ""                   # "Strong apply", "Good match", etc.
    reason_to_apply: str = ""              # one-line reason to apply
    blocker_risk: str = ""                 # "Low", "Medium", "High"
```

This is the richest model — it captures not just the score but WHY the score is what it is. Every number is explained.

---

## 8. model_config.py — Choosing the AI Model

This file answers: "Which AI model should we use right now?"

### The Priority Chain

```
Has JOB_SCOUT_MODEL or GEMINI_MODEL set?
    YES → Use that model (user's explicit choice wins)
    NO ↓
Has GOOGLE_API_KEY?
    YES → Use gemini-2.0-flash (Google's native Gemini)
    NO ↓
Has GROQ_API_KEY?
    YES → Use groq/llama-3.3-70b-versatile
    NO ↓
Has NVIDIA_NIM_API_KEY?
    YES → Use nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5
    NO ↓
Default → gemini-2.0-flash (will fail without a key, but that's the fallback)
```

**Why this specific order?** Google Gemini is the "native" provider for ADK, so it's tried first. Groq and NVIDIA are secondary options for people who want cheaper or faster models.

### Reasoning Model Safety

```python
REASONING_MODEL_MARKERS = ("reasoning",)

def is_reasoning_model(model_name: str) -> bool:
    normalized = (model_name or "").strip().lower()
    return bool(normalized) and any(marker in normalized for marker in REASONING_MODEL_MARKERS)
```

Some models (like NVIDIA's `nemotron-super` with the `reasoning` variant) produce "thinking" output before their final answer. This internal reasoning content can confuse the agent framework — it might include long chains of thought that break tool call formatting.

By default, if a reasoning model is detected, it's downgraded to a non-reasoning variant. You can opt back in with `JOB_SCOUT_ALLOW_REASONING_MODEL=true`.

### Three Different Model Roles

```python
resolve_model_name()              # Main chat model
resolve_resume_parser_model()     # For structuring extracted resume text
resolve_resume_attachment_model() # For parsing attached files (always native Gemini)
```

**Why three different models for one agent?**

Different tasks need different capabilities:
- Main chat: Any capable LLM works
- Resume text structuring: NVIDIA/Groq preferred (fast, cheap structured extraction)
- File attachment parsing: MUST be native Gemini (only Gemini can read PDFs/images natively)

This separation allows you to use a cheap model for bulk processing and a capable model for the main conversation.

---

## 9. tools.py — The Tool Hub

```python
from .resume_tools import (
    extract_resume_profile_from_artifact,
    get_resume_status,
    save_resume_profile,
    clear_resume_profile,
    score_job_match,
)
from .search_tools import (
    search_jobs,
    filter_saved_jobs_by_experience,
)
from .search_support import fetch_job_details
```

This file is just a **re-export hub** — it imports tools from their respective modules and makes them all available from one place.

**Why have this file at all?** Without it, `agent.py` would need to import from 3+ different files. With it, `agent.py` imports from one place. It's about organizing your imports cleanly.

---

## 10. search_tools.py — Finding Jobs

### `_split_requested_roles()`

```python
_ROLE_ALIAS_PATTERNS = (
    (re.compile(r"\bfull[\s-]*stack\b"), "Full Stack Developer"),
    (re.compile(r"\bback[\s-]*end\b"), "Backend Developer"),
    (re.compile(r"\bfront[\s-]*end\b"), "Frontend Developer"),
)

def _split_requested_roles(role: str) -> list[str]:
    # First: detect known role patterns
    for pattern, canonical_role in _ROLE_ALIAS_PATTERNS:
        if pattern.search(normalized_lower):
            detected_roles.append(canonical_role)
    
    if detected_roles:
        return _dedupe(detected_roles)
    
    # Second: split by comma, slash, ampersand, "and", "or"
    split_candidates = re.split(r"\s*(?:,|/|&|\band\b|\bor\b)\s*", ...)
```

**What this does:** Converts a user's informal role string into a clean list of role titles.

**Examples:**
```
"full stack"           → ["Full Stack Developer"]
"frontend and backend" → ["Frontend Developer", "Backend Developer"]
"React, Node.js"       → ["React", "Node.js"]
"Python developer"     → ["Python developer"]  (no pattern match, no split)
```

**Why regex instead of simple string matching?** Because users write roles inconsistently:
- "full-stack" vs "full stack" vs "fullstack" — the regex `full[\s-]*stack` handles all three
- "back-end" vs "backend" vs "back end" — similarly handled

The `\b` markers mean "word boundary" — so "backend" won't match "backends" or "my-backend-service."

### `search_jobs()` — The Main Search Tool

```python
def search_jobs(
    role: str,
    location: str,
    max_results: int = 5,
    country: Optional[str] = None,
    min_years: Optional[float] = None,
    max_years: Optional[float] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
```

**The `tool_context` parameter:** This is special — ADK automatically injects the `ToolContext` object when the agent calls this function. The agent doesn't pass it explicitly; ADK does. This gives the tool access to session state.

**The demo mode:**
```python
if provider == "demo":
    demo_jobs = []
    for index in range(1, max_results + 1):
        demo_jobs.append(JobPosting(...fake data...))
    return {"status": "demo_mode", "jobs": demo_jobs}
```

When no API credentials are configured, instead of crashing with an error, the tool returns synthetic fake job data. This lets you test the entire agent flow without paying for API access. Important for development.

**The multi-role search:**
```python
per_role_limit = max(1, min(5, (max_results + len(requested_roles) - 1) // max(1, len(requested_roles))))
search_batches = [
    _search_jobs_once(role=requested_role, ...) 
    for requested_role in requested_roles
]
selected_jobs = _merge_job_lists([batch["jobs"] for batch in search_batches], max_results=max_results)
```

**What this does:** If the user asks for "Frontend and Backend" jobs, it searches separately for each and combines the results. The `per_role_limit` calculation ensures the total doesn't exceed `max_results`.

**Example:** If `max_results=5` and there are 2 roles:
```
per_role_limit = max(1, min(5, ceil(5/2))) = max(1, min(5, 3)) = 3
Search Frontend → up to 3 jobs
Search Backend  → up to 3 jobs
Merge → pick best 5 overall
```

**Saving to state:**
```python
if tool_context:
    tool_context.state[LAST_SEARCH_RESULTS_STATE_KEY] = SearchContext.model_validate({...}).model_dump()
```

After every search, the results are saved to session state. This allows the `filter_saved_jobs_by_experience` tool to work on previous results without re-searching.

### `filter_saved_jobs_by_experience()` — The Follow-up Filter

```python
def filter_saved_jobs_by_experience(
    min_years: float,
    max_years: float,
    tool_context: ToolContext,
) -> dict:
```

This tool only works on results already saved by `search_jobs()`. The flow:

```
1. Read saved jobs from state (LAST_SEARCH_RESULTS_STATE_KEY)
2. For each job:
   a. Check title and snippet for experience hints ("2-3 years", "senior", "junior")
   b. If no hints found AND there's a URL → fetch the full job page and check that
   c. Classify: matched / unknown
3. Return matched + unknown lists
```

**Why "unknown" as a separate category?** Some job postings don't mention experience at all. Instead of discarding them (they might be good for you), they're returned separately so the user can review them manually.

---

## 11. scoring.py — Matching Resume to Job

This is the most mathematically complex file. Let's break it down completely.

### 11.1 SKILL_ONTOLOGY

```python
SKILL_ONTOLOGY = {
    "python": {"weight": 1.4, "aliases": []},
    "react": {"weight": 1.3, "aliases": ["react.js", "reactjs"]},
    "node.js": {"weight": 1.3, "aliases": ["nodejs", "node js", "node"]},
    "html": {"weight": 0.7, "aliases": []},
    "css": {"weight": 0.7, "aliases": []},
    "llm": {"weight": 1.4, "aliases": ["large language model", "llms"]},
    ...
}
```

This is a hand-curated dictionary of 33 skills with:
- **Canonical name**: The "official" name (`"node.js"`)
- **Weight**: How valuable is this skill in the job market (1.0 = average, 1.4 = very valuable)
- **Aliases**: Other ways to write the same skill (`"nodejs"`, `"node js"`, `"node"`)

**Why weights?** If a job requires React (weight 1.3) and CSS (weight 0.7), matching React should count more. Without weights, every skill would count equally — which doesn't reflect the job market.

**Why a hand-coded ontology instead of NLP/AI?** See Section 15.

### 11.2 Text Normalization

```python
def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)       # remove HTML tags
    text = text.replace("&", " and ")
    for source, target in TITLE_SYNONYMS.items():
        text = re.sub(rf"(?<!\w){re.escape(source)}(?!\w)", target, text)
    text = _WS.sub(" ", text).strip()
    return text
```

Before any comparison, all text is normalized:
1. `unicodedata.normalize("NFKC")`: Converts fancy Unicode characters to standard equivalents (`ﬁ` → `fi`, `©` → `©`)
2. `.lower()`: Makes everything lowercase so "React" == "react"
3. Strip HTML tags: Job descriptions often contain `<b>Required</b>` — we remove the tags
4. `& → and`: Standardizes `JavaScript & TypeScript` to `javascript and typescript`
5. Synonym substitution: `sde → software engineer`, `fullstack → full stack`

**Why do all this?** Because job descriptions and resumes come from inconsistent sources. Without normalization, "React.js" ≠ "reactjs" ≠ "React" — even though they mean the same thing.

### 11.3 `_extract_skill_mentions()`

```python
def _extract_skill_mentions(text: str) -> set[str]:
    normalized = _normalize(text)
    found: set[str] = set()
    for canonical, meta in SKILL_ONTOLOGY.items():
        forms = [canonical] + meta["aliases"]
        if any(_count_phrase(normalized, form) > 0 for form in forms):
            found.add(canonical)
    return found
```

**What this does:** Goes through every skill in the ontology and checks if any form of that skill appears in the text.

**Returns a set:** Because we only care whether a skill is mentioned, not how many times. A resume that says "React" 10 times isn't 10x better at React than one that says it once.

**Why `_count_phrase` instead of `in`?** Because `"react"` might appear inside `"interactive"` (it-`react`-ive). `_count_phrase` uses `(?<!\w)react(?!\w)` — the `\w` anchors ensure "react" is not surrounded by other word characters.

### 11.4 `_extract_job_requirements()`

```python
def _extract_job_requirements(job_description: str) -> dict:
    # Split JD into lines/sentences
    # For each line, detect skills AND classify the line as "required", "preferred", or "general"
    return {
        "required": [...],   # "Must have React experience"
        "preferred": [...],  # "Nice to have: Docker"
        "general": [...]     # "We use Python for backend"
    }
```

**Classification logic:**
```python
REQUIRED_HINTS = ("required", "must", "need to", "minimum", "mandatory", "proficient in", ...)
PREFERRED_HINTS = ("preferred", "nice to have", "good to have", "plus", "bonus", "desired")

def _classify_requirement_segment(segment: str) -> str:
    if any(hint in segment for hint in PREFERRED_HINTS):
        return "preferred"
    if any(hint in segment for hint in REQUIRED_HINTS):
        return "required"
    return "general"
```

**Example:**
```
"Must have: React, Node.js (required)"        → required: [react, node.js]
"Nice to have: Docker, Kubernetes"            → preferred: [docker, kubernetes]
"We primarily use Python for data pipelines"  → general: [python]
```

**Why split by line?** Because a JD might say "Required skills: Python" on one line and "Bonus: Docker" on another. You need to classify each line separately.

### 11.5 The Scoring Formula

```python
final_score = (
    0.45 * required_score       # How many required skills you have
    + 0.15 * preferred_score    # How many preferred skills you have
    + 0.10 * general_score      # General skill overlap
    + 0.12 * title_score        # Does your title match the job title?
    + 0.10 * experience_score   # Does your experience match requirements?
    + 0.08 * overlap_score      # General vocabulary overlap
)
```

**Why these weights?**

```
Required skills = 45%   → Most important. Can't do the job without required skills.
Preferred skills = 15%  → Nice to have, but not blockers.
General skills = 10%    → Background context, lower weight.
Title alignment = 12%   → "Frontend Developer" applying to "Frontend Role" is a signal.
Experience = 10%        → Years of experience matter but less than actual skills.
Keyword overlap = 8%    → Vocabulary overlap as a soft signal.
```

The weights sum to 1.0 (100%), so the final score is always 0–1, then multiplied by 100 to get a percentage.

### 11.6 Penalty Multipliers

```python
required_count = len(job_requirements["required"])
if required_count:
    missing_required_ratio = len(missing_required) / required_count
    if missing_required_ratio >= 0.5:
        final_score *= 0.72   # Missing 50%+ required skills → harsh penalty
    elif missing_required_ratio >= 0.3:
        final_score *= 0.85   # Missing 30-49% required skills → moderate penalty

if required_years is not None and experience_score < 0.55:
    final_score *= 0.9        # Experience too low → 10% penalty
```

**Why penalties?** The base formula is additive — good things add up. But if you're missing most of the required skills, the base score might still be surprisingly high (because you have preferred skills, good title alignment, etc.). Penalties ensure a fundamentally bad match is correctly scored low.

**Example:**
```
Job requires: Python, Django, PostgreSQL, Redis
Your skills: Python only (1/4 = 25% coverage)

Without penalty: base_score might be 0.45 * 0.25 + other signals = ~0.40 → 40/100
With 0.72 penalty: 0.40 * 0.72 = 28.8 → 29/100 (correct — you're missing most requirements)
```

### 11.7 Experience Alignment

```python
def _experience_alignment(required_years: Optional[float], actual_years: Optional[float]) -> float:
    if required_years is None:
        return 0.7     # JD doesn't mention years → neutral score
    if actual_years is None:
        return 0.35    # We don't know your experience → low score
    if actual_years >= required_years:
        return 1.0     # You meet or exceed requirement → perfect
    gap = required_years - actual_years
    if gap <= 0.5:  return 0.9   # 6 months short → still good
    if gap <= 1:    return 0.75  # 1 year short → passable
    if gap <= 2:    return 0.55  # 2 years short → stretch
    return 0.2                   # 2+ years short → significant gap
```

**Why these specific numbers?** Job market reality:
- Being 6 months short of a 3-year requirement rarely matters (0.9 score)
- Being 1 year short is common and often overlooked (0.75 score)
- Being 2 years short is a real concern but applications still worth sending (0.55)
- More than 2 years short is a significant risk (0.2 score)

### 11.8 Title Alignment

```python
def _title_alignment(job_title: str, role_titles: list[str]) -> float:
    job_tokens = _tokenize(job_title)
    for role_title in role_titles:
        role_tokens = _tokenize(role_title)
        precision = len(job_tokens & role_tokens) / max(1, len(job_tokens))
        recall = len(job_tokens & role_tokens) / max(1, len(role_tokens))
        best = max(best, 0.7 * precision + 0.3 * recall)
    return min(1.0, best)
```

**Precision and recall:**
- **Precision**: "Of the words in the JOB title, how many match your titles?" (avoids credit for having more words)
- **Recall**: "Of the words in YOUR titles, how many appear in the job title?" (avoids penalizing for extra words)

**Example:**
```
Job title: "Senior Full Stack Developer"    → tokens: {senior, full, stack, developer}
Your title: "Full Stack Developer"          → tokens: {full, stack, developer}

Intersection: {full, stack, developer} = 3

Precision = 3/4 = 0.75  (3 of 4 job words matched)
Recall    = 3/3 = 1.0   (all your words matched)

Score = 0.7 * 0.75 + 0.3 * 1.0 = 0.525 + 0.3 = 0.825
```

**Why 70% precision + 30% recall?** Precision matters more — if the job says "Senior" and you don't have that, it's a signal you might not match. Recall matters less — having extra words in your title that the JD doesn't use is fine.

### 11.9 Verdict Generation

```python
def _summarize_fit_verdict(score: int, has_blockers: bool) -> str:
    if score >= 80 and not has_blockers:
        return "Strong apply"
    if score >= 65:
        return "Good match"
    if score >= 45:
        return "Possible stretch"
    return "Low match"
```

**Why `not has_blockers` for "Strong apply"?** You could score 85 but still be missing a critical required skill. The score measures overall fit — blockers measure dealbreakers. Even a high score with a blocker should be called out.

---

## 12. litellm_compat.py — Making Different AIs Work Together

This file is about fixing incompatibilities between ADK and third-party AI providers.

### The Problem

ADK was built for Google Gemini. When you use Groq or NVIDIA NIM via LiteLLM, there are small differences in how messages are formatted that can cause errors.

### Problem 1: Missing Tool Call IDs (Groq)

When the agent calls a tool, the conversation history looks like:
```
assistant: [tool_call: search_jobs with id="call_abc123"]
tool:      [result of search_jobs, references id="call_abc123"]
```

Groq's API requires every `tool` role message to reference a valid `tool_call_id`. If ADK passes a conversation where the ID is missing, Groq rejects it.

The fix patches the request to fill in missing IDs:
```python
def _repair_missing_tool_call_ids(payload: dict) -> None:
    # For every "tool" role message without a tool_call_id,
    # find the corresponding tool_call in the previous "assistant" message
    # and fill in the ID
```

### Problem 2: Content Block Types (Groq, NVIDIA)

Some AI providers only accept text content in messages. ADK might include "reasoning" content blocks (internal thinking) in the message history. Providers like Groq don't know what to do with these and return errors.

The fix strips or converts unsupported content types to plain text.

### Why Not Fix This in ADK Itself?

ADK is Google's framework and evolves independently. These are workarounds for known version incompatibilities. The `patch_litellm_tool_call_id_repair()` function "monkey-patches" the LiteLLM library at runtime — it replaces a function inside LiteLLM with a fixed version without changing the library's source code.

---

## 13. state_keys.py — Memory Between Messages

```python
RESUME_PROFILE_STATE_KEY = "resume_profile"
LAST_SEARCH_RESULTS_STATE_KEY = "last_search_results"
RESUME_PROFILE_EXTRACTION_METHOD_STATE_KEY = "resume_profile_extraction_method"
```

These are just string constants — the dictionary keys used to store data in session state.

**Why constants instead of raw strings?**

```python
# Bad approach — raw strings everywhere
state["resume_profiIe"] = profile  # typo! "profiIe" not "profile"
state.get("Resume_Profile")        # different capitalization — never finds it!

# Good approach — constants
state[RESUME_PROFILE_STATE_KEY] = profile    # consistent
state.get(RESUME_PROFILE_STATE_KEY)          # always the same key
```

If you ever need to rename the key, you change it in one place, and every file that imports the constant is automatically updated.

---

## 14. Scoring Algorithm Deep Dive

Let's trace through a complete scoring example.

**Resume:** Alice, React Developer, 2 years experience, skills: React, JavaScript, CSS, Git

**Job:** "Junior Frontend Developer at TechCorp"

```
Required: React (listed as "must have React")
Preferred: TypeScript (listed as "nice to have TypeScript")
General: JavaScript, CSS, HTML (mentioned in description)
Required experience: "1-2 years" → 1.0 years (takes the minimum of the range)
```

**Step 1: Extract Alice's skills from resume text**
```
Profile text: "alice react developer react javascript css git"
Skills found by ontology matching: {react, javascript, css, git}
```

**Step 2: Score required skills**
```
Required: [react]
Alice has: react ✓
matched = [react], missing = []
weight(react) = 1.3
required_score = 1.3 / 1.3 = 1.0 (100%)
```

**Step 3: Score preferred skills**
```
Preferred: [typescript]
Alice has: nothing
preferred_score = 0.0
But no preferred skills matched → falls back to 0.6 (neutral)
```

**Step 4: Score general skills**
```
General: [javascript, css]
Alice has: javascript ✓, css ✓
general_score = (1.3 + 0.7) / (1.3 + 0.7) = 2.0/2.0 = 1.0
```

**Step 5: Title alignment**
```
Job: "junior frontend developer" → tokens: {junior, frontend, developer}
Alice: "react developer" → tokens: {react, developer}

Intersection: {developer} = 1 word
Precision = 1/3 = 0.33
Recall    = 1/2 = 0.50
Score = 0.7 * 0.33 + 0.3 * 0.50 = 0.231 + 0.15 = 0.381
```

**Step 6: Experience alignment**
```
Required: 1.0 years
Alice has: 2.0 years
2.0 >= 1.0 → experience_score = 1.0 (perfect)
```

**Step 7: General keyword overlap**
```
[various words from resume vs JD vocabulary]
overlap_score ≈ 0.4 (rough estimate)
```

**Step 8: Combine**
```
final_score = 0.45 * 1.0    (required)
            + 0.15 * 0.6    (preferred — neutral fallback)
            + 0.10 * 1.0    (general)
            + 0.12 * 0.381  (title — lower because "React" ≠ "Frontend")
            + 0.10 * 1.0    (experience — perfect)
            + 0.08 * 0.4    (keyword overlap)

= 0.45 + 0.09 + 0.10 + 0.046 + 0.10 + 0.032
= 0.818

score = round(0.818 * 100) = 82
```

**Step 9: Blockers check**
```
missing_required = []      → no blocker
insufficient_experience? 2.0 >= 1.0 → no blocker
title_mismatch? 0.381 >= 0.35 → no blocker (barely passes threshold)
has_blockers = False
```

**Step 10: Verdict**
```
score=82, no blockers → "Strong apply"
```

**Final result:**
```
Score: 82/100
Verdict: Strong apply
Matched: react, javascript, css
Missing: typescript (preferred only)
Reason: Strongest alignment comes from react, javascript, css.
```

---

## 15. Why Not the Other Way? — Every Major Design Decision

### 15.1 Why hand-coded SKILL_ONTOLOGY instead of AI/embeddings?

**Alternative:** Use sentence embeddings (ML vectors) to measure semantic similarity between resume text and JD text. Libraries like `sentence-transformers` are commented out in `requirements.txt` — they were considered.

**Why hand-coded ontology was chosen:**
1. **Predictability**: If you know a user has React, the score is always exactly the same. Embeddings can give slightly different results each time.
2. **Debuggability**: You can trace exactly why the score changed. With embeddings, it's a black box.
3. **No GPU required**: Embedding models are large (hundreds of MB). The hand-coded approach runs on any machine instantly.
4. **Aliases are explicit**: `"nodejs"` and `"node.js"` are explicitly declared as the same thing. Embeddings might accidentally treat them as different or might conflate unrelated things.
5. **Testable**: Deterministic scoring can have automated tests with exact expected outputs (see `evaluation_cases.json`).

**The tradeoff**: The ontology is limited to 33 pre-defined skills. It won't recognize a niche skill like "Tauri" or "Bun" unless explicitly added.

### 15.2 Why LiteLLM instead of direct API calls to each provider?

**Alternative:** Write direct HTTP requests to each provider's API separately.

**Why LiteLLM:**
1. **One interface**: Add a new provider by just changing the model name string (`groq/llama-3.3...` vs `nvidia_nim/...`)
2. **Maintained**: LiteLLM handles API changes, rate limits, retries
3. **Well-known**: Documentation and community support

**The tradeoff**: LiteLLM has its own bugs (hence `litellm_compat.py`). Direct API calls would be more predictable but much more code to maintain.

### 15.3 Why ADK instead of building the agent loop manually?

**Alternative:** Use LangChain, or write a custom `while True` loop that handles tool calls.

**Why ADK:**
1. **Designed for agents**: ADK handles the tool-call → response → tool-call loop correctly
2. **Session state built-in**: No need to implement persistence yourself
3. **Artifact management**: File upload handling is built in
4. **FastAPI integration**: `get_fast_api_app()` gives you a production server instantly

**The tradeoff**: ADK is opinionated and less flexible than a custom approach. You're locked into Google's abstractions.

### 15.4 Why three resume parsing tiers?

```
Tier 1: Local parsing (regex + text extraction)
Tier 2: Text model (LLM-based structuring)
Tier 3: Gemini attachment (native vision)
```

**Why not just always use the best (Tier 3)?**
1. **Cost**: Gemini API calls cost money. Local parsing is free.
2. **Speed**: Local parsing is instant. API calls add latency.
3. **Reliability**: Local parsing has no external dependency. It works offline.

**Why not just use local parsing always?**
Because local parsing fails on some PDFs (encrypted, image-based, complex layouts). The tiered approach uses the cheapest method first and escalates only when needed.

### 15.5 Why save search results to session state?

**Alternative:** Re-run the search every time the user asks a follow-up question.

**Why save:**
1. **API costs**: Job search APIs charge per request. Re-running wastes money.
2. **Speed**: Search APIs take 2-10 seconds. Using cached results is instant.
3. **Consistency**: If you search again, you might get different results. Caching ensures the user sees consistent results within a conversation.

### 15.6 Why static instructions + dynamic hints instead of all-static?

**Alternative:** Put all instructions in one big static string.

**Why split:**
The static instructions say "if a resume exists, do X." But the static instructions don't know whether a resume exists — that's session-specific. You'd need to include instructions for BOTH cases in the static prompt:
```
"If resume exists, use resume-aware mode.
 If resume doesn't exist, use prompt-only mode.
 If resume exists AND search results exist, also do Y.
 If no resume AND no search results..."
```

This gets complex and confusing. Dynamic hints inject exactly the right context at the right time:
- "Resume is loaded. Skills: React, Node.js" → tells LLM exactly what's in state
- "Previous search found 5 jobs" → reminds LLM to use cached results

### 15.7 Why `temperature=0.2`?

**Alternative:** Higher temperature (0.5, 0.8, 1.0).

| Temperature | Effect | Problem |
|-------------|--------|---------|
| 0.0 | Fully deterministic, always picks most likely token | Can be repetitive, might miss valid paths |
| 0.2 | Very reliable, follows instructions, rare variation | **This project's choice** |
| 0.7 | More creative, but less reliable | Might invent job listings |
| 1.0+ | Very creative, often unpredictable | Would make up scores, invent companies |

For a job assistant that makes real decisions, reliability beats creativity.

### 15.8 Why not use a database for session state?

ADK supports connecting to Firestore or other databases via `session_service_uri`. Locally, it uses in-memory storage. The code handles both transparently via environment variables — no code change needed to switch.

### 15.9 Why FastAPI instead of Flask?

1. **Async native**: FastAPI supports `async/await` natively. ADK uses async extensively.
2. **Type hints**: FastAPI uses Python type hints for automatic validation.
3. **Auto docs**: FastAPI generates interactive API documentation automatically.
4. **ADK integration**: ADK's `get_fast_api_app()` produces a FastAPI app — this choice was made by ADK.

---

## 16. How a Real Conversation Flows

Let's trace exactly what happens when a user says "Find me Python jobs in Bangalore":

```
USER: "Find me Python jobs in Bangalore"
         │
         ▼
   FastAPI receives POST /run
         │
         ▼
   ADK session lookup (or create new session)
         │
         ▼
   add_runtime_hints() runs:
     - state.get(resume_profile) → None
     - state.get(last_search) → None
     - list_artifacts() → []
     - Appends: "No resume profile exists yet. Use prompt-only mode."
         │
         ▼
   LLM receives:
     [system: You are Job Scout...
              No resume profile exists. Use prompt-only mode.]
     [user: "Find me Python jobs in Bangalore"]
         │
         ▼
   LLM thinks and outputs:
     "I'll search for Python Developer jobs in Bangalore."
     [tool_call: search_jobs(role="Python Developer", location="Bangalore")]
         │
         ▼
   ADK intercepts tool call, runs search_jobs()
         │
         ▼
   search_jobs():
     - _split_requested_roles("Python Developer") → ["Python Developer"]
     - _normalize_country("Bangalore", None) → "IN"
     - provider = "adzuna" (or "apify", or "demo")
     - _search_jobs_once(...) → hits Adzuna API
     - Returns: {status: "ok", jobs: [...5 jobs...]}
     - Saves to state: last_search_results = {role: "Python Developer", jobs: [...]}
         │
         ▼
   ADK returns tool result to LLM
         │
         ▼
   add_runtime_hints() runs AGAIN (before next LLM call):
     - state.get(resume_profile) → None
     - state.get(last_search) → {role: "Python Developer", jobs: [...]}
     - Appends: "A previous job search is stored in state."
         │
         ▼
   LLM sees tool result + updated hints, responds:
     "Here are 5 Python Developer jobs in Bangalore:
      1. Python Backend Developer at Infosys...
      2. Python Data Engineer at TCS...
      ..."
         │
         ▼
   Response returned to user
```

Now the user says "Show me only entry level":

```
USER: "Show me only entry level"
         │
         ▼
   add_runtime_hints():
     - last_search found → appends "Previous search available for follow-up"
         │
         ▼
   LLM: "I'll filter the saved jobs for entry level (0-1 years)"
     [tool_call: filter_saved_jobs_by_experience(min_years=0, max_years=1)]
         │
         ▼
   filter_saved_jobs_by_experience():
     - Reads from state: last_search_results.jobs
     - For each job: analyze title/snippet for experience hints
     - Returns: {jobs: [2 matching], unknown_experience_jobs: [1 unknown]}
         │
         ▼
   LLM presents filtered results
```

Key insight: The LLM did NOT re-search. It used the saved results from state.

---

## 17. Environment Variables — What They All Mean

Your `.env` file controls everything about the agent's behavior.

### AI Model Configuration

| Variable | What it does | Example |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Key for Google Gemini | `AIza...` |
| `GEMINI_MODEL` | Which Gemini/LLM to use | `gemini-2.0-flash` |
| `GEMINI_FALLBACK_MODEL` | Backup if primary fails | `gemini-2.0-flash` |
| `GROQ_API_KEY` | Key for Groq (free tier available) | `gsk_...` |
| `NVIDIA_NIM_API_KEY` | Key for NVIDIA NIM models | `nvapi-...` |
| `NVIDIA_NIM_API_BASE` | NVIDIA API URL | `https://integrate.api.nvidia.com/v1` |
| `JOB_SCOUT_MODEL` | Explicit model override | Overrides all auto-detection |
| `JOB_SCOUT_RESUME_PARSER_MODEL` | Model specifically for resume parsing | |
| `JOB_SCOUT_ALLOW_REASONING_MODEL` | Allow reasoning models | `true` or `false` |
| `JOB_SCOUT_INCLUDE_REASONING_PARTS` | Include reasoning in responses | `true` or `false` |

### Job Search Configuration

| Variable | What it does | Example |
|----------|-------------|---------|
| `JOB_SCOUT_JOB_SEARCH_PROVIDER` | Which job search API to use | `adzuna`, `apify`, `browseract` |
| `ADZUNA_APP_ID` | Adzuna API key ID | From adzuna.com developer account |
| `ADZUNA_APP_KEY` | Adzuna API secret key | |
| `ADZUNA_COUNTRY` | Default country for Adzuna | `in` (India), `us`, `gb` |
| `APIFY_TOKEN` | Apify API token (for Indeed scraper) | From apify.com |
| `BROWSERACT_API_KEY` | BrowserAct API key (for custom workflows) | |
| `BROWSERACT_WORKFLOW_ID` | Which BrowserAct workflow to run | |

### Deployment Configuration

| Variable | What it does |
|----------|-------------|
| `SESSION_SERVICE_URI` | Where to store sessions (Firestore URI for production) |
| `ARTIFACT_SERVICE_URI` | Where to store uploaded files (Cloud Storage URI) |
| `PORT` | Which port the server listens on (default: 8080) |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID for Cloud Logging and deployment |
| `GOOGLE_CLOUD_LOCATION` | GCP region for deployment |

---

## 18. Glossary of All Technical Terms

| Term | Simple Definition |
|------|-----------------|
| **ADK** | Agent Development Kit — Google's framework for building AI agents |
| **Agent** | An AI system that can have conversations AND use tools to take actions |
| **Alias** | Alternative name for something (`"reactjs"` is an alias for `"react"`) |
| **ASGI** | Async Server Gateway Interface — how Python web apps receive HTTP requests asynchronously |
| **Artifact** | A file that was uploaded to the agent (like a resume PDF) |
| **Async/Await** | Python keywords for non-blocking operations — other code runs while waiting for slow things |
| **Blockers** | Reasons why a job application might be rejected despite a decent score |
| **Callback** | A function that automatically runs at a specific moment (like before every LLM call) |
| **CORS** | Cross-Origin Resource Sharing — security rule about which websites can call your API |
| **Deterministic** | Same inputs always produce the same outputs — no randomness |
| **Docstring** | Text description inside a Python function — the agent reads this to understand what the tool does |
| **FastAPI** | Python web framework used to build the agent's HTTP API |
| **Field(default_factory=list)** | Pydantic way to give every model instance its own empty list (avoids shared state bug) |
| **LiteLLM** | Library that gives a unified interface to different AI providers |
| **LLM** | Large Language Model — the AI (Gemini, Groq, etc.) that powers the agent |
| **Monkey-patching** | Replacing a function in a library at runtime without modifying the library's source |
| **Ontology** | A structured vocabulary of concepts and their relationships |
| **Optional[X]** | Python type hint meaning "this can be X or None" |
| **Pydantic** | Python library for defining data shapes with automatic validation |
| **Precision** | Of the things I claimed match, how many actually match |
| **Recall** | Of all the things that should match, how many did I find |
| **Regex (re)** | Regular expressions — patterns for searching text |
| **Session State** | A dictionary that persists throughout a user's conversation |
| **System Prompt** | Instructions given to the AI before the user's first message — defines its behavior |
| **Temperature** | Controls randomness in LLM output. Lower = more predictable |
| **ToolContext** | ADK object passed to tools that gives access to session state |
| **Type hint** | Python annotation like `name: str` that tells other code what type a variable should be |
| **Uvicorn** | The ASGI server that actually runs the FastAPI app |
| **Weight** | A number that says how much something should count in a calculation |
| **NFKC** | A Unicode normalization form that converts equivalent characters to a standard form |
| **STOPWORDS** | Common words (the, and, for) that are excluded from meaningful text analysis |

---

## Summary Visualization: The Full System at a Glance

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        JOB SCOUT AGENT                                   │
│                                                                          │
│  INPUT TYPES                    OUTPUT TYPES                             │
│  ───────────                    ────────────                             │
│  • Text messages                • Job listings with URLs                 │
│  • Resume PDF/DOCX              • Match scores (0-100)                   │
│  • Filters (location,           • Matched/missing skills                 │
│    experience, role)            • Fit verdicts                           │
│                                 • Blocker warnings                       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  AGENT LOOP  (manages the conversation)                          │    │
│  │                                                                  │    │
│  │  User message → add_runtime_hints() → LLM → tool call?          │    │
│  │                                         ↓           ↓           │    │
│  │                                    Final answer   Run tool       │    │
│  │                                                      ↓          │    │
│  │                                                 Tool result      │    │
│  │                                                      ↓          │    │
│  │                                              Back to LLM         │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  TOOLS AVAILABLE                                                         │
│  ───────────────                                                         │
│  Resume Tools              Search Tools              Scoring             │
│  ────────────              ────────────              ───────             │
│  extract_resume  ──────→  resume_support.py          scoring.py         │
│  get_status                                           SKILL_ONTOLOGY    │
│  save_profile              search_jobs  ──────────→  search_support.py  │
│  clear_profile             filter_by_exp              Adzuna API         │
│  score_job_match           fetch_details              Apify/Indeed       │
│                                                       BrowserAct         │
│                                                       Demo mode          │
│                                                                          │
│  MEMORY (Session State)                                                  │
│  ──────────────────────                                                  │
│  resume_profile            → Who are you? What are your skills?          │
│  last_search_results       → What jobs did we find last time?            │
│  extraction_method         → How was the resume parsed?                  │
│                                                                          │
│  SCORE FORMULA (scoring.py)                                              │
│  ─────────────────────────                                               │
│  45% Required skills  +  15% Preferred  +  10% General                  │
│  + 12% Title match  +  10% Experience  +  8% Keywords                   │
│  × Penalty multiplier (if missing many required skills)                  │
│  = Final score (0-100)                                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

*This document was written specifically to help you understand every decision made in this codebase. As you read the code now, come back to the relevant section here whenever something is unclear.*
