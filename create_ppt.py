from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)

# ── Color palette ──────────────────────────────────────────────
DARK_BG   = RGBColor(0x0D, 0x1B, 0x2A)   # deep navy
ACCENT1   = RGBColor(0x00, 0xB4, 0xD8)   # cyan
ACCENT2   = RGBColor(0x90, 0xE0, 0xEF)   # light cyan
ACCENT3   = RGBColor(0xFF, 0xB7, 0x03)   # amber
ACCENT4   = RGBColor(0x06, 0xD6, 0xA0)   # green
ACCENT5   = RGBColor(0xEF, 0x47, 0x6F)   # red-pink
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY= RGBColor(0xB0, 0xC4, 0xDE)
CARD_BG   = RGBColor(0x14, 0x2D, 0x4C)   # slightly lighter navy
CODE_BG   = RGBColor(0x06, 0x10, 0x1E)   # very dark for code

blank_layout = prs.slide_layouts[6]   # completely blank

# ══════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════

def add_slide():
    return prs.slides.add_slide(blank_layout)

def bg(slide, color=DARK_BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color

def box(slide, l, t, w, h, fill_color=CARD_BG, alpha=None):
    shape = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    return shape

def txt(slide, text, l, t, w, h, size=18, bold=False, color=WHITE,
        align=PP_ALIGN.LEFT, wrap=True, italic=False):
    txb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    txb.word_wrap = wrap
    tf = txb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txb

def heading(slide, text, subtitle=None):
    """Slide heading bar at top."""
    box(slide, 0, 0, 13.33, 1.1, ACCENT1)
    txt(slide, text, 0.3, 0.1, 12, 0.7, size=28, bold=True, color=DARK_BG, align=PP_ALIGN.LEFT)
    if subtitle:
        txt(slide, subtitle, 0.3, 0.75, 12, 0.4, size=13, color=DARK_BG, align=PP_ALIGN.LEFT, italic=True)

def divider(slide, t, color=ACCENT1, thickness=2):
    line = slide.shapes.add_shape(1, Inches(0.4), Inches(t), Inches(12.5), Inches(0.04))
    line.fill.solid()
    line.fill.fore_color.rgb = color
    line.line.fill.background()

def bullet_card(slide, l, t, w, h, title, bullets, title_color=ACCENT3, bullet_color=ACCENT2):
    box(slide, l, t, w, h, CARD_BG)
    txt(slide, title, l+0.1, t+0.08, w-0.2, 0.38, size=14, bold=True, color=title_color)
    divider_shape = slide.shapes.add_shape(1, Inches(l+0.1), Inches(t+0.46), Inches(w-0.2), Inches(0.03))
    divider_shape.fill.solid(); divider_shape.fill.fore_color.rgb = title_color
    divider_shape.line.fill.background()
    y = t + 0.55
    for b in bullets:
        txt(slide, f"▸  {b}", l+0.15, y, w-0.3, 0.32, size=11, color=WHITE)
        y += 0.33
    return y

def code_box(slide, l, t, w, h, code_text, title=None):
    box(slide, l, t, w, h, CODE_BG)
    inner = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(0.28))
    inner.fill.solid(); inner.fill.fore_color.rgb = RGBColor(0x1A,0x3A,0x5C)
    inner.line.fill.background()
    if title:
        txt(slide, title, l+0.1, t+0.04, w-0.2, 0.22, size=10, bold=True, color=ACCENT2)
    txt(slide, code_text, l+0.15, t+0.32, w-0.3, h-0.4, size=9.5, color=RGBColor(0xA8,0xFF,0xC2), wrap=True)

def arrow(slide, x1, y1, x2, y2, color=ACCENT1):
    """Draw a simple line arrow between two points (inches)."""
    from pptx.util import Inches
    connector = slide.shapes.add_shape(1,
        Inches(min(x1,x2)), Inches(min(y1,y2)),
        Inches(abs(x2-x1)+0.01), Inches(abs(y2-y1)+0.01))
    connector.fill.background()
    connector.line.color.rgb = color
    connector.line.width = Pt(1.5)

def flow_box(slide, l, t, w, h, label, fill=CARD_BG, text_color=WHITE, size=12):
    b = box(slide, l, t, w, h, fill)
    txt(slide, label, l+0.05, t + h/2 - 0.18, w-0.1, 0.36, size=size, bold=True,
        color=text_color, align=PP_ALIGN.CENTER)
    return b

def tag(slide, label, l, t, color=ACCENT4):
    b = box(slide, l, t, len(label)*0.085+0.2, 0.28, color)
    txt(slide, label, l+0.08, t+0.04, len(label)*0.085+0.1, 0.22, size=9, bold=True,
        color=DARK_BG, align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
# big gradient rectangle
big = s.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.33), Inches(7.5))
big.fill.solid(); big.fill.fore_color.rgb = DARK_BG; big.line.fill.background()

# decorative circle accents
for cx, cy, r, col in [(11,1,1.8,ACCENT1),(1.5,6.5,1.2,ACCENT4),(0.5,0.8,0.6,ACCENT3)]:
    c = s.shapes.add_shape(9, Inches(cx-r/2), Inches(cy-r/2), Inches(r), Inches(r))
    c.fill.solid(); c.fill.fore_color.rgb = col
    c.line.fill.background()
    c.fill.fore_color.theme_color  # just to commit
    # make semi-transparent by using a light version trick — just do tinted box
    # (pptx doesn't support alpha easily; use lighter shade)

txt(s, "JOB SCOUT AGENT", 1, 1.8, 11, 1.2, size=52, bold=True, color=ACCENT1, align=PP_ALIGN.CENTER)
txt(s, "A Complete Beginner's Guide — From A to Z", 1, 3.1, 11, 0.6, size=22, color=ACCENT2, align=PP_ALIGN.CENTER)
divider(s, 3.9, ACCENT3)
txt(s, "Agent Architecture  ·  LLMs & Tools  ·  Resume Parsing  ·  Job Scoring  ·  Multi-Provider Search",
    1, 4.1, 11, 0.5, size=14, color=LIGHT_GRAY, align=PP_ALIGN.CENTER, italic=True)
txt(s, "Built with Google ADK · Gemini · FastAPI · Python", 1, 5.0, 11, 0.4, size=13, color=ACCENT3, align=PP_ALIGN.CENTER)
txt(s, "30 Slides", 1, 6.8, 11, 0.4, size=11, color=LIGHT_GRAY, align=PP_ALIGN.CENTER, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 2 — What are we building?
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "What Are We Building?", "Big picture before we dive into code")

txt(s, "Job Scout Agent is an AI-powered assistant that helps you find jobs that match YOUR skills.",
    0.4, 1.3, 12.5, 0.5, size=16, color=ACCENT2)

cols = [
    ("👤 You", "Upload your resume or describe your skills", ACCENT4),
    ("🤖 Agent", "Understands you, searches multiple job boards, scores every job", ACCENT1),
    ("📊 Result", "Ranked list of jobs with match score & missing skills", ACCENT3),
]
for i,(icon, desc, col) in enumerate(cols):
    x = 0.5 + i*4.2
    box(s, x, 2.1, 3.8, 2.2, CARD_BG)
    txt(s, icon, x+0.1, 2.2, 3.6, 0.6, size=28, align=PP_ALIGN.CENTER)
    txt(s, desc, x+0.1, 2.9, 3.6, 1.2, size=13, color=WHITE, align=PP_ALIGN.CENTER)
    b = s.shapes.add_shape(1, Inches(x+0.1), Inches(4.35), Inches(3.6), Inches(0.06))
    b.fill.solid(); b.fill.fore_color.rgb = col; b.line.fill.background()

# arrows between boxes
txt(s, "→", 4.4, 2.8, 0.5, 0.5, size=30, color=ACCENT1, align=PP_ALIGN.CENTER)
txt(s, "→", 8.6, 2.8, 0.5, 0.5, size=30, color=ACCENT1, align=PP_ALIGN.CENTER)

txt(s, "Why is this interesting to build?", 0.4, 4.7, 12, 0.35, size=14, bold=True, color=ACCENT3)
points = [
    "It combines multiple AI concepts: LLMs, tools, agents, state management, multi-provider APIs",
    "It solves a REAL problem — manually checking 10 job boards and comparing them to your resume is painful",
    "It shows when to use AI (conversation, parsing) vs deterministic code (scoring, filtering)",
]
for i,p in enumerate(points):
    txt(s, f"  {i+1}.  {p}", 0.5, 5.1+i*0.42, 12.2, 0.4, size=12, color=WHITE)

# ══════════════════════════════════════════════════════════════
# SLIDE 3 — What is an AI Agent?
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Core Concept: What is an AI Agent?", "The single most important idea in this project")

txt(s, "A regular program follows fixed instructions. An AI Agent decides WHAT to do next based on context.",
    0.4, 1.25, 12.5, 0.45, size=15, color=ACCENT2)

# comparison table
headers = ["Regular Program", "AI Agent"]
col_colors = [ACCENT5, ACCENT4]
for i, (h, col) in enumerate(zip(headers, col_colors)):
    x = 0.5 + i * 6.2
    box(s, x, 1.85, 6.0, 0.5, col)
    txt(s, h, x, 1.88, 6.0, 0.44, size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    rows = [
        ("Follows fixed IF/ELSE rules", "Decides actions by reasoning"),
        ("You hard-code every scenario", "It handles unseen situations"),
        ("Can't understand natural language", "Understands 'Find React jobs near me'"),
        ("One function = one fixed output", "Can chain multiple tools dynamically"),
        ("Deterministic by design", "Adaptive — responds to context"),
    ]
    for j, (left, right) in enumerate(rows):
        cell_text = left if i==0 else right
        bg_c = CARD_BG if j%2==0 else DARK_BG
        box(s, x, 2.38+j*0.55, 6.0, 0.53, bg_c)
        txt(s, cell_text, x+0.15, 2.44+j*0.55, 5.7, 0.43, size=12, color=WHITE)

txt(s, "The Loop at the heart of every agent:", 0.4, 5.45, 9, 0.35, size=13, bold=True, color=ACCENT3)
steps = ["1. Think\n(LLM reads context)", "2. Act\n(call a Tool)", "3. Observe\n(read tool result)", "4. Repeat\n(until done)"]
for i, step in enumerate(steps):
    col = [ACCENT1, ACCENT3, ACCENT4, ACCENT2][i]
    flow_box(s, 0.5+i*3.1, 5.9, 2.8, 1.1, step, fill=col, text_color=DARK_BG, size=12)
    if i < 3:
        txt(s, "→", 3.25+i*3.1, 6.2, 0.4, 0.5, size=22, color=WHITE, align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════
# SLIDE 4 — Project Architecture (visual diagram)
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Project Architecture — The 30,000 ft View")

# User box
flow_box(s, 0.2, 1.3, 1.8, 0.7, "👤 User\n(Browser)", ACCENT4, DARK_BG, 11)
# arrow right
txt(s, "→", 2.05, 1.4, 0.4, 0.5, size=20, color=ACCENT2)
# FastAPI
flow_box(s, 2.5, 1.3, 2.2, 0.7, "FastAPI\nmain.py", ACCENT1, DARK_BG, 11)
txt(s, "→", 4.75, 1.4, 0.4, 0.5, size=20, color=ACCENT2)
# ADK
flow_box(s, 5.2, 1.3, 2.5, 0.7, "Google ADK\nFramework", RGBColor(0x42,0x85,0xF4), WHITE, 11)
txt(s, "→", 7.75, 1.4, 0.4, 0.5, size=20, color=ACCENT2)
# Agent
flow_box(s, 8.2, 1.3, 2.5, 0.7, "root_agent\nagent.py", ACCENT3, DARK_BG, 11)
txt(s, "→", 10.75, 1.4, 0.4, 0.5, size=20, color=ACCENT2)
# LLM
flow_box(s, 11.2, 1.3, 1.9, 0.7, "Gemini\nLLM", ACCENT5, WHITE, 11)

# Arrow down from agent
txt(s, "↓", 9.3, 2.1, 0.5, 0.4, size=20, color=ACCENT2)

txt(s, "9 Tools registered with the Agent", 0.4, 2.55, 12.5, 0.35, size=13, bold=True, color=ACCENT3)

tools = [
    ("extract_resume\n_profile", ACCENT4), ("get_resume\n_status", ACCENT4),
    ("save_resume\n_profile", ACCENT4), ("clear_resume\n_profile", ACCENT4),
    ("score_job\n_match", ACCENT1), ("search_jobs", ACCENT1),
    ("filter_saved\n_jobs", ACCENT1), ("fetch_job\n_details", ACCENT1),
    ("score_job\n_match", ACCENT3),
]
tools2 = [
    ("extract_resume_profile", ACCENT4), ("get_resume_status", ACCENT4),
    ("save_resume_profile", ACCENT4), ("clear_resume_profile", ACCENT4),
    ("score_job_match", ACCENT1), ("search_jobs", ACCENT1),
    ("filter_saved_jobs_by_experience", ACCENT1), ("fetch_job_details", ACCENT1),
    ("score_job_match (scoring)", ACCENT3),
]

for i, (name, col) in enumerate(tools2[:8]):
    x = 0.25 + i * 1.62
    box(s, x, 3.0, 1.5, 0.75, col)
    txt(s, name, x+0.05, 3.05, 1.4, 0.65, size=9, bold=True, color=DARK_BG, align=PP_ALIGN.CENTER)

txt(s, "Support Modules (called by tools):", 0.4, 3.95, 12, 0.3, size=12, bold=True, color=LIGHT_GRAY)
mods = [
    ("resume_support.py\nPDF parsing, LLM extraction", CARD_BG),
    ("search_support.py\nAdzuna, Apify, BrowserAct, Demo", CARD_BG),
    ("scoring.py\nDeterministic skill matching", CARD_BG),
    ("model_config.py\nModel priority chain", CARD_BG),
    ("litellm_compat.py\nProvider patches", CARD_BG),
    ("domain_models.py\nPydantic schemas", CARD_BG),
]
for i, (name, col) in enumerate(mods):
    x = 0.25 + i * 2.18
    box(s, x, 4.3, 2.0, 0.9, CARD_BG)
    txt(s, name, x+0.08, 4.35, 1.85, 0.8, size=9.5, color=ACCENT2, align=PP_ALIGN.CENTER)

txt(s, "External APIs: Gemini · Groq · NVIDIA NIM · Adzuna · Apify · BrowserAct",
    0.4, 5.45, 12.5, 0.35, size=11, color=LIGHT_GRAY, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 5 — File Structure
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Project File Structure — What Every File Does")

files = [
    ("agent.py", "Brain of the agent — defines the LLM, 9 tools, and instructions", ACCENT3),
    ("main.py", "FastAPI web server — entry point, handles HTTP requests from browser", ACCENT1),
    ("domain_models.py", "Pydantic data schemas — ResumeProfile, JobPosting, JobMatchScore etc.", ACCENT4),
    ("scoring.py", "Deterministic scoring algorithm — matches resume skills to job requirements", ACCENT5),
    ("resume_support.py", "Resume parser — extracts text from PDF/DOCX, calls LLM to structure it", ACCENT2),
    ("resume_tools.py", "4 resume-related tools (extract, save, get status, clear)", ACCENT3),
    ("search_support.py", "Job search providers — Adzuna API, Apify scraper, BrowserAct, Demo mode", ACCENT1),
    ("search_tools.py", "3 search-related tools (search, filter by exp, fetch details)", ACCENT4),
    ("model_config.py", "Model selector — picks Gemini/Groq/NVIDIA based on env vars", ACCENT2),
    ("litellm_compat.py", "Compatibility patches for non-Gemini LLM providers", ACCENT5),
    ("test_tools.py", "1039-line test suite for all tools", LIGHT_GRAY),
    (".env", "Secret API keys — never committed to git", ACCENT3),
]

for i, (fname, desc, col) in enumerate(files):
    col_idx = i % 2
    x = 0.3 if col_idx == 0 else 6.9
    y = 1.25 + (i // 2) * 0.62
    box(s, x, y, 6.3, 0.56, CARD_BG)
    txt(s, fname, x+0.1, y+0.06, 1.8, 0.44, size=11, bold=True, color=col)
    txt(s, desc, x+1.95, y+0.1, 4.2, 0.38, size=10, color=WHITE)

# ══════════════════════════════════════════════════════════════
# SLIDE 6 — Concept: How LLMs Work
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Core Concept: How Large Language Models (LLMs) Work", "You must understand this before understanding the agent")

txt(s, "An LLM is a neural network trained on billions of text documents. It predicts the next token.",
    0.4, 1.25, 12.5, 0.4, size=14, color=ACCENT2)

# Token prediction diagram
box(s, 0.4, 1.8, 12.4, 1.1, CARD_BG)
tokens = ["Find", "me", "Python", "jobs", "in", "Hyderabad", "→", "[NEXT?]"]
colors = [ACCENT2]*6 + [ACCENT1, ACCENT3]
for i, (tok, col) in enumerate(zip(tokens, colors)):
    bx = s.shapes.add_shape(1, Inches(0.6+i*1.55), Inches(1.92), Inches(1.3), Inches(0.5))
    bx.fill.solid(); bx.fill.fore_color.rgb = col; bx.line.fill.background()
    txt(s, tok, 0.6+i*1.55, 1.98, 1.3, 0.38, size=12, bold=True, color=DARK_BG, align=PP_ALIGN.CENTER)

txt(s, "Key Properties of LLMs used in this project:", 0.4, 3.05, 12, 0.35, size=13, bold=True, color=ACCENT3)

concepts = [
    ("Context Window", "The LLM reads ALL previous messages + tool results to decide what to do next. This is how it 'remembers' the conversation.", ACCENT1),
    ("Temperature", "Controls randomness. temp=0.2 (used here) = mostly deterministic. temp=1.0 = very creative. Low temp = consistent answers.", ACCENT4),
    ("System Prompt", "Instructions given to the LLM at the start. In agent.py the long instruction string tells the agent HOW to behave.", ACCENT3),
    ("Function Calling\n(Tool Use)", "LLMs can output structured JSON asking to call a function (tool). The framework executes the tool and sends the result back to the LLM.", ACCENT5),
]

for i, (name, desc, col) in enumerate(concepts):
    x = 0.4 + (i%2)*6.4
    y = 3.5 + (i//2)*1.35
    box(s, x, y, 6.2, 1.2, CARD_BG)
    txt(s, name, x+0.1, y+0.08, 2.2, 0.5, size=12, bold=True, color=col)
    txt(s, desc, x+2.3, y+0.08, 3.8, 1.05, size=10.5, color=WHITE)

# ══════════════════════════════════════════════════════════════
# SLIDE 7 — Concept: Tools in Agents
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Core Concept: Tools — How the Agent Takes Actions", "Tools transform a chatbot into an agent")

txt(s, "Without tools: LLM can only generate text.  With tools: LLM can search the web, read files, call APIs, run code.",
    0.4, 1.25, 12.5, 0.4, size=14, color=ACCENT2)

# Tool call flow
steps = [
    ("User says:\n'Find Python jobs'", ACCENT4, "1"),
    ("LLM outputs:\nTool call JSON\n{search_jobs,\nrole:'Python'}", ACCENT1, "2"),
    ("Framework\nexecutes\nsearch_jobs()", ACCENT3, "3"),
    ("Tool returns:\nList of 5 jobs", ACCENT4, "4"),
    ("LLM reads result,\nformats & replies\nto user", ACCENT2, "5"),
]
for i, (label, col, num) in enumerate(steps):
    flow_box(s, 0.3+i*2.55, 1.85, 2.3, 1.5, label, fill=col, text_color=DARK_BG, size=11)
    if i < 4:
        txt(s, "→", 2.65+i*2.55, 2.35, 0.35, 0.5, size=20, color=WHITE)

txt(s, "How a Tool is defined in Python (from search_tools.py):", 0.4, 3.55, 10, 0.35, size=13, bold=True, color=ACCENT3)
code = '''def search_jobs(
    role: str,              # ← what job to search for
    location: str = "",     # ← where (optional)
    max_results: int = 5,   # ← how many jobs
    tool_context: ToolContext = None   # ← ADK injects this automatically
) -> dict:
    """Search for jobs matching the given role and location."""
    # ... implementation ...
    return {"jobs": [...], "total": 5}'''
code_box(s, 0.4, 3.98, 7.8, 2.1, code, "Tool Definition — Python function with type hints")

# why section
box(s, 8.4, 3.98, 4.7, 2.1, CARD_BG)
txt(s, "Why this design?", 8.5, 4.05, 4.5, 0.35, size=12, bold=True, color=ACCENT3)
whys = [
    "Type hints → ADK auto-generates JSON schema for the LLM",
    "Docstring → LLM reads this to decide WHEN to call the tool",
    "ToolContext → gives access to session state (memory)",
    "Returns dict → easy for LLM to read and reason about",
]
for i,w in enumerate(whys):
    txt(s, f"▸ {w}", 8.5, 4.48+i*0.37, 4.5, 0.34, size=10.5, color=WHITE)

txt(s, "Key insight: The LLM never runs code directly — it ASKS for a tool call. The framework runs it and returns the result.",
    0.4, 6.2, 12.5, 0.4, size=12, color=ACCENT3, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 8 — Concept: Pydantic & Data Validation
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Core Concept: Pydantic — Enforcing Data Structure", "domain_models.py — Why we need strict schemas")

txt(s, "AI systems deal with unpredictable data (PDFs, scraped web pages, LLM outputs). Pydantic catches errors early.",
    0.4, 1.25, 12.5, 0.4, size=14, color=ACCENT2)

# Without pydantic vs with pydantic
box(s, 0.4, 1.75, 5.8, 3.8, CARD_BG)
txt(s, "❌ Without Pydantic", 0.5, 1.82, 5.6, 0.38, size=13, bold=True, color=ACCENT5)
code1 = '''# Raw dict — anything goes
resume = {
    "name": "Alice",
    "skills": "Python",  # should be list!
    "years_exp": "three",# should be int!
}
# Later in code... crashes!
for skill in resume["skills"]:
    ...  # iterates chars, not skills!'''
code_box(s, 0.5, 2.28, 5.6, 2.2, code1)

box(s, 6.4, 1.75, 6.6, 3.8, CARD_BG)
txt(s, "✅ With Pydantic (domain_models.py)", 6.5, 1.82, 6.4, 0.38, size=13, bold=True, color=ACCENT4)
code2 = '''class ResumeProfile(BaseModel):
    candidate_name: str
    skills: list[str]     # MUST be list
    years_experience: int # MUST be int

# Pydantic validates at creation:
profile = ResumeProfile(
    candidate_name="Alice",
    skills=["Python","React"],
    years_experience=3
)
# ✓ Safe to use everywhere!'''
code_box(s, 6.5, 2.28, 6.3, 2.2, code2)

txt(s, "Models defined in domain_models.py:", 0.4, 5.75, 12, 0.32, size=13, bold=True, color=ACCENT3)
models = ["ResumeProfile — candidate name, skills, years exp, current role",
          "JobPosting — title, company, location, description, salary",
          "JobMatchScore — score 0-100, fit verdict, matched skills, blockers",
          "SearchContext — what was searched, when, results count"]
for i, m in enumerate(models):
    x = 0.4 + (i%2)*6.5
    y = 6.15 + (i//2)*0.42
    txt(s, f"▸  {m}", x, y, 6.3, 0.38, size=11, color=WHITE)

# ══════════════════════════════════════════════════════════════
# SLIDE 9 — Google ADK Framework
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Google ADK — The Framework Doing the Heavy Lifting", "agent.py uses this — understanding it is key")

txt(s, "ADK (Agent Development Kit) is Google's open-source framework for building LLM agents. It handles the loop.",
    0.4, 1.25, 12.5, 0.4, size=14, color=ACCENT2)

# what ADK handles
box(s, 0.4, 1.75, 5.8, 4.6, CARD_BG)
txt(s, "What ADK handles FOR you:", 0.5, 1.82, 5.6, 0.35, size=13, bold=True, color=ACCENT4)
adk_handles = [
    "Sending messages to the LLM (Gemini or LiteLLM)",
    "Detecting when LLM wants to call a tool",
    "Executing the tool function with the right arguments",
    "Sending tool result back to LLM",
    "Repeating until LLM gives a final response",
    "Managing conversation history",
    "Handling file/artifact attachments",
    "Session state persistence",
]
for i, h in enumerate(adk_handles):
    txt(s, f"✓  {h}", 0.55, 2.28+i*0.48, 5.5, 0.44, size=11, color=WHITE)

box(s, 6.4, 1.75, 6.6, 4.6, CARD_BG)
txt(s, "What YOU write (agent.py):", 6.5, 1.82, 6.4, 0.35, size=13, bold=True, color=ACCENT3)
code3 = '''root_agent = Agent(
  model=model_config.chat_model,
  name="job_scout",
  instruction="""
    You are a job search assistant.
    When user asks for jobs, call
    search_jobs() first...
    [long instructions]
  """,
  tools=[
    search_jobs,
    score_job_match,
    extract_resume_profile_from_artifact,
    # ... 6 more tools
  ],
  before_model_callback=add_runtime_hints,
)'''
code_box(s, 6.5, 2.28, 6.3, 4.0, code3, "agent.py — all you write")

txt(s, "Why ADK instead of building from scratch? → Building the tool-calling loop yourself takes 500+ lines. ADK does it reliably.",
    0.4, 6.55, 12.5, 0.4, size=12, color=ACCENT3, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 10 — agent.py Deep Dive
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "agent.py — The Agent Brain (339 lines)", "Every important section explained")

parts = [
    ("Imports & Model Setup\n(lines 1-40)",
     "Imports ADK, LiteLLM compat patches, model_config.\nPatches run at import time — ensures provider fixes apply before anything else.",
     ACCENT1, "Why import order matters: litellm_compat must patch BEFORE ADK uses LiteLLM."),

    ("add_runtime_hints()\ncallback (lines 42-110)",
     "Called BEFORE every LLM call.\nInjects resume profile summary + current search context into the system prompt dynamically.",
     ACCENT4, "Why dynamic hints? The LLM system prompt is static. This adds live state without changing the base prompt."),

    ("_sanitize_for_nvidia()\n(lines 112-160)",
     "NVIDIA NIM doesn't support binary file attachments.\nThis function replaces file parts with a text placeholder before sending to NVIDIA.",
     ACCENT3, "Why not just remove files? Removing silently breaks functionality. Replacing with text placeholder tells LLM what was there."),

    ("root_agent = Agent(...)\n(lines 162-339)",
     "The actual agent object. Registers all 9 tools, sets model, temperature=0.2, long instruction string, and the before_model_callback.",
     ACCENT5, "Why temperature=0.2? We want consistent, predictable answers — not creative storytelling. Low temp = reliable tool calls."),
]

for i, (title, body, col, insight) in enumerate(parts):
    x = 0.4 + (i%2)*6.5
    y = 1.3 + (i//2)*2.85
    box(s, x, y, 6.2, 2.65, CARD_BG)
    txt(s, title, x+0.1, y+0.08, 6.0, 0.55, size=12, bold=True, color=col)
    txt(s, body, x+0.1, y+0.65, 6.0, 0.95, size=10.5, color=WHITE)
    box(s, x+0.1, y+1.65, 6.0, 0.85, CODE_BG)
    txt(s, f"💡 {insight}", x+0.2, y+1.72, 5.8, 0.72, size=10, color=ACCENT2, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 11 — Resume Processing Overview
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Resume Processing — The 3-Layer Fallback Chain", "resume_support.py + resume_tools.py")

txt(s, "Problem: Resumes come as PDFs with complex layouts. We need structured data (name, skills, years). How?",
    0.4, 1.25, 12.5, 0.4, size=14, color=ACCENT2)

# Fallback chain visual
layers = [
    ("Layer 1: LOCAL TEXT EXTRACTION", "Try to extract text directly from the PDF binary.\nNo API call needed — fast and free.", ACCENT4, "✓ Works for 80% of PDFs\n✗ Fails on scanned/image PDFs"),
    ("Layer 2: LLM TEXT-MODEL PARSE", "Send extracted text to a cheap LLM (Groq/Gemini).\nAsk it to return structured JSON with name, skills, years.", ACCENT3, "✓ Works when text was extracted\n✗ Still fails on image-based PDFs"),
    ("Layer 3: GEMINI ATTACHMENT PARSE", "Send the actual PDF file to Gemini (vision model).\nGemini can 'see' and read image-based PDFs.", ACCENT1, "✓ Works on all PDFs\n✗ Most expensive, requires Gemini API"),
]

y_start = 1.85
for i, (title, desc, col, outcome) in enumerate(layers):
    box(s, 0.4, y_start+i*1.65, 12.4, 1.5, CARD_BG)
    # Number badge
    badge = s.shapes.add_shape(9, Inches(0.5), Inches(y_start+i*1.65+0.35), Inches(0.55), Inches(0.55))
    badge.fill.solid(); badge.fill.fore_color.rgb = col; badge.line.fill.background()
    txt(s, str(i+1), 0.5, y_start+i*1.65+0.35, 0.55, 0.55, size=14, bold=True, color=DARK_BG, align=PP_ALIGN.CENTER)
    txt(s, title, 1.2, y_start+i*1.65+0.1, 6.0, 0.4, size=12, bold=True, color=col)
    txt(s, desc, 1.2, y_start+i*1.65+0.55, 6.3, 0.65, size=11, color=WHITE)
    box(s, 7.6, y_start+i*1.65+0.08, 5.0, 1.32, CODE_BG)
    txt(s, outcome, 7.75, y_start+i*1.65+0.25, 4.7, 1.0, size=11, color=ACCENT2)
    if i < 2:
        txt(s, "↓  if fails, try next layer", 0.5, y_start+i*1.65+1.55, 5, 0.2, size=10, color=ACCENT5, italic=True)

txt(s, "Why 3 layers? Different resumes need different approaches. Trying fast local first saves API costs. Falling back ensures reliability.",
    0.4, 6.8, 12.5, 0.38, size=11.5, color=ACCENT3, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 12 — resume_support.py Deep Dive
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "resume_support.py — How PDF Text Extraction Works", "935 lines — the most complex support module")

txt(s, "PDFs are NOT plain text files. They contain binary streams with compressed, encoded text. We must decode them.",
    0.4, 1.25, 12.5, 0.4, size=14, color=ACCENT2)

box(s, 0.4, 1.75, 12.4, 1.7, CARD_BG)
txt(s, "PDF Internal Structure (what we actually parse):", 0.5, 1.82, 12, 0.32, size=12, bold=True, color=ACCENT3)
code4 = "BT  /F1 12 Tf  72 700 Td  (Alice Johnson) Tj  0 -20 Td  <50797468 6F6E> Tj  ET"
txt(s, f"Raw PDF stream:   {code4}", 0.55, 2.2, 12.0, 0.32, size=9.5, color=RGBColor(0xA8,0xFF,0xC2))
txt(s, "After decoding:   Alice Johnson    Python", 0.55, 2.55, 12.0, 0.32, size=9.5, color=ACCENT2)
txt(s, "BT=Begin Text, Tf=Font, Td=Move position, Tj=Show literal text, <>Hex=hex-encoded text, ET=End Text",
    0.55, 2.92, 12.0, 0.32, size=9, color=LIGHT_GRAY, italic=True)

txt(s, "Key functions in resume_support.py:", 0.4, 3.58, 8, 0.32, size=13, bold=True, color=ACCENT3)
funcs = [
    ("_extract_pdf_text()", "Reads raw PDF bytes. Finds text streams (BT...ET). Decodes hex-encoded strings. Handles both (literal) and <hex> text formats.", ACCENT1),
    ("_structure_with_llm()", "Takes raw extracted text, builds a prompt asking LLM to return JSON with name/skills/years/role. Validates with Pydantic.", ACCENT4),
    ("_detect_role_from_text()", "Regex patterns match phrases like 'Full Stack Developer', 'Frontend Engineer' to classify candidate role.", ACCENT3),
    ("_extract_years_from_text()", "Regex finds patterns like '3 years of experience', '1-3 years', '2+ years' and returns an integer.", ACCENT2),
    ("extract_and_structure_resume()", "Orchestrates all layers. Tries local→LLM text→LLM attachment. Returns ResumeProfile or error dict.", ACCENT5),
    ("_select_resume_artifact()", "Given list of uploaded files, picks the one that looks like a resume (checks MIME type + extension).", ACCENT4),
]
for i, (name, desc, col) in enumerate(funcs):
    x = 0.4 + (i%2)*6.5
    y = 3.98 + (i//2)*0.87
    box(s, x, y, 6.2, 0.8, CARD_BG)
    txt(s, name, x+0.1, y+0.06, 2.5, 0.32, size=11, bold=True, color=col)
    txt(s, desc, x+2.65, y+0.06, 3.45, 0.68, size=9.5, color=WHITE)

# ══════════════════════════════════════════════════════════════
# SLIDE 13 — Scoring Algorithm Overview
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "The Scoring Algorithm — How Jobs Get Ranked", "scoring.py — 505 lines, the most important logic")

txt(s, "Every job gets a 0-100 score. Higher score = better match. Score drives the ranking you see.",
    0.4, 1.25, 12.5, 0.4, size=14, color=ACCENT2)

# Score formula visual
txt(s, "Final Score Formula:", 0.4, 1.78, 6, 0.35, size=14, bold=True, color=ACCENT3)

weights = [
    ("Required Skills\nCoverage", 45, ACCENT5),
    ("Preferred Skills", 15, ACCENT1),
    ("General Skills", 10, ACCENT4),
    ("Title Alignment", 12, ACCENT3),
    ("Experience\nAlignment", 10, ACCENT2),
    ("Keyword Overlap", 8, LIGHT_GRAY),
]
bar_x = 0.4
txt(s, "Score = sum of weighted components:", 0.4, 2.15, 8, 0.32, size=11, color=LIGHT_GRAY, italic=True)
for i, (label, pct, col) in enumerate(weights):
    y = 2.55 + i*0.68
    # bar background
    bar_bg = s.shapes.add_shape(1, Inches(2.8), Inches(y+0.1), Inches(7.0), Inches(0.4))
    bar_bg.fill.solid(); bar_bg.fill.fore_color.rgb = CODE_BG; bar_bg.line.fill.background()
    # bar fill
    bar_fill = s.shapes.add_shape(1, Inches(2.8), Inches(y+0.1), Inches(7.0*pct/100), Inches(0.4))
    bar_fill.fill.solid(); bar_fill.fill.fore_color.rgb = col; bar_fill.line.fill.background()
    txt(s, label, 0.4, y+0.05, 2.3, 0.55, size=10, bold=True, color=col, align=PP_ALIGN.RIGHT)
    txt(s, f"{pct}%", 9.9, y+0.1, 0.7, 0.4, size=12, bold=True, color=col)

# Verdict thresholds
box(s, 9.8, 2.15, 3.3, 4.3, CARD_BG)
txt(s, "Score → Verdict", 9.9, 2.22, 3.1, 0.35, size=12, bold=True, color=ACCENT3)
verdicts = [("80-100", "Strong Match", ACCENT4), ("60-79", "Good Match", ACCENT1),
            ("40-59", "Partial Match", ACCENT3), ("0-39", "Weak Match", ACCENT5)]
for i, (rng, label, col) in enumerate(verdicts):
    box(s, 9.85, 2.65+i*0.88, 3.2, 0.78, col if i==0 else CARD_BG)
    txt(s, rng, 9.95, 2.72+i*0.88, 1.0, 0.35, size=11, bold=True, color=col if i>0 else DARK_BG)
    txt(s, label, 11.0, 2.72+i*0.88, 2.0, 0.35, size=11, color=WHITE if i>0 else DARK_BG, bold=(i==0))

txt(s, "Penalty applied when >50% required skills are missing: score × 0.72", 0.4, 6.58, 9, 0.35, size=11.5, color=ACCENT5, italic=True)
txt(s, "Blockers: Skills/experience missing that would make hiring impossible → shown separately from score",
    0.4, 6.98, 12, 0.35, size=11, color=LIGHT_GRAY, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 14 — Why Deterministic Scoring?
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Design Decision: Why Deterministic Scoring, Not AI?", "The most important architectural choice in this project")

txt(s, "We COULD have asked the LLM: 'Rate this job match 0-100'. Why didn't we?",
    0.4, 1.25, 12.5, 0.4, size=15, color=ACCENT2)

# comparison
headers2 = ["LLM-based Scoring (NOT used)", "Deterministic Scoring (USED)"]
col_colors2 = [ACCENT5, ACCENT4]
for i, (h, col) in enumerate(zip(headers2, col_colors2)):
    x = 0.4 + i*6.4
    box(s, x, 1.78, 6.2, 0.48, col)
    txt(s, h, x, 1.82, 6.2, 0.4, size=13, bold=True, color=DARK_BG if col==ACCENT4 else WHITE, align=PP_ALIGN.CENTER)

rows2 = [
    ("Non-deterministic — same job scores differently each run",     "Always same score for same inputs"),
    ("Can 'hallucinate' matching skills that don't exist",           "Only matches skills explicitly found"),
    ("Slow — extra LLM call per job (expensive at scale)",           "Fast — no API call, runs in microseconds"),
    ("Hard to explain 'why this score'",                             "Shows exact matched/missing skills"),
    ("Can't run automated tests to check correctness",               "Full test suite with fixtures"),
    ("Score drifts if model updates",                                "Score is stable and version-controlled"),
]

for j, (bad, good) in enumerate(rows2):
    bg_c = CARD_BG if j%2==0 else DARK_BG
    for i, cell in enumerate([bad, good]):
        x = 0.4 + i*6.4
        box(s, x, 2.3+j*0.55, 6.2, 0.52, bg_c)
        icon = "✗  " if i==0 else "✓  "
        col = ACCENT5 if i==0 else ACCENT4
        txt(s, icon+cell, x+0.12, 2.35+j*0.55, 5.9, 0.44, size=11, color=col)

txt(s, "When IS LLM used in scoring-adjacent tasks?", 0.4, 5.7, 8, 0.35, size=13, bold=True, color=ACCENT3)
txt(s, "Resume parsing (extracting skills from PDF text) — LLM is better than regex for understanding unstructured text.\nJob description pre-processing (skill classification: required vs preferred) — LLM reads context better than keyword matching.",
    0.4, 6.1, 12.5, 0.65, size=12, color=WHITE)

# ══════════════════════════════════════════════════════════════
# SLIDE 15 — Skill Ontology
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "scoring.py — The Skill Ontology", "Hand-curated skill weights that reflect job market value")

txt(s, "An ontology is a structured knowledge map. Here it maps 33 skills to weights reflecting their market importance.",
    0.4, 1.25, 12.5, 0.4, size=14, color=ACCENT2)

txt(s, "Why weights? Not all skills are equal. Python expertise is harder to replace than CSS knowledge.",
    0.4, 1.72, 12.5, 0.35, size=12, color=LIGHT_GRAY, italic=True)

# Skill weight bars (top 12 skills)
skills_data = [
    ("Python", 1.4, ACCENT4), ("Machine Learning", 1.35, ACCENT4), ("React", 1.3, ACCENT1),
    ("Node.js", 1.25, ACCENT1), ("SQL", 1.2, ACCENT3), ("Docker", 1.2, ACCENT3),
    ("TypeScript", 1.15, ACCENT2), ("AWS", 1.15, ACCENT2), ("FastAPI", 1.1, ACCENT5),
    ("Git", 1.0, LIGHT_GRAY), ("HTML", 0.9, LIGHT_GRAY), ("CSS", 0.7, LIGHT_GRAY),
]

txt(s, "Skill Weight Chart (higher = more valuable in job descriptions):", 0.4, 2.12, 9, 0.32, size=12, bold=True, color=ACCENT3)
bar_max = 1.4
for i, (skill, weight, col) in enumerate(skills_data):
    x = 0.4 + (i%6)*2.18
    y = 2.52 + (i//6)*1.85
    box(s, x, y, 2.0, 1.7, CARD_BG)
    # bar
    bar_h = 0.9 * (weight / bar_max)
    bar_y = y + 0.2 + (0.9 - bar_h)
    bar = s.shapes.add_shape(1, Inches(x+0.35), Inches(bar_y), Inches(1.3), Inches(bar_h))
    bar.fill.solid(); bar.fill.fore_color.rgb = col; bar.line.fill.background()
    txt(s, f"{weight}", x+0.35, bar_y-0.28, 1.3, 0.26, size=10, bold=True, color=col, align=PP_ALIGN.CENTER)
    txt(s, skill, x+0.05, y+1.35, 1.9, 0.3, size=10, color=WHITE, align=PP_ALIGN.CENTER)

box(s, 0.4, 6.3, 12.4, 0.95, CARD_BG)
txt(s, "Why not use NLP embeddings to compare skills automatically?", 0.5, 6.37, 8, 0.32, size=12, bold=True, color=ACCENT3)
txt(s, "Embeddings require training data, are opaque, and can drift. Hand-curated weights are transparent, explainable, and easy to update.",
    0.5, 6.72, 12.0, 0.45, size=11, color=WHITE)

# ══════════════════════════════════════════════════════════════
# SLIDE 16 — Skill Classification (required vs preferred)
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "scoring.py — Skill Classification from Job Descriptions", "How the algorithm reads 'Required' vs 'Preferred'")

txt(s, "Job descriptions mix required and nice-to-have skills. Classification affects scoring weights heavily (45% vs 15%).",
    0.4, 1.25, 12.5, 0.4, size=14, color=ACCENT2)

code5 = '''# From scoring.py — classify_jd_skills()
REQUIRED_MARKERS = [
    "required", "must have", "must-have", "essential",
    "you must", "we require", "mandatory", "minimum requirement"
]
PREFERRED_MARKERS = [
    "preferred", "nice to have", "bonus", "plus",
    "desired", "ideally", "an advantage", "a plus"
]

# How it works:
# 1. Split job description into paragraphs
# 2. Find paragraph containing skill keyword
# 3. Check if REQUIRED_MARKERS appear nearby (within 200 chars)
# 4. Check if PREFERRED_MARKERS appear nearby
# 5. If neither: classify as "general" (middle weight)'''
code_box(s, 0.4, 1.78, 7.5, 3.2, code5, "Skill classification logic")

# Example job description
box(s, 8.1, 1.78, 5.0, 3.2, CARD_BG)
txt(s, "Example Job Description:", 8.2, 1.85, 4.8, 0.32, size=12, bold=True, color=ACCENT3)
jd_text = [
    ("Requirements (REQUIRED):", ACCENT5),
    ("• Python — 3+ years", ACCENT5),
    ("• PostgreSQL experience", ACCENT5),
    ("", WHITE),
    ("Nice to Have (PREFERRED):", ACCENT1),
    ("• React knowledge", ACCENT1),
    ("• Docker experience", ACCENT1),
    ("", WHITE),
    ("About the role (GENERAL):", ACCENT3),
    ("• Team collaboration", ACCENT3),
    ("• Agile methodology", ACCENT3),
]
for i, (line, col) in enumerate(jd_text):
    txt(s, line, 8.2, 2.28+i*0.25, 4.7, 0.23, size=9.5, color=col)

txt(s, "Effect on scoring:", 0.4, 5.15, 5, 0.32, size=13, bold=True, color=ACCENT3)
effect_rows = [
    ("Required Skills", "45%", "Missing required → score × 0.72 penalty", ACCENT5),
    ("Preferred Skills", "15%", "Missing preferred → small deduction only", ACCENT1),
    ("General Skills", "10%", "Missing general → minimal impact", ACCENT3),
]
for i, (cat, wt, effect, col) in enumerate(effect_rows):
    box(s, 0.4, 5.55+i*0.58, 12.4, 0.52, CARD_BG)
    txt(s, cat, 0.5, 5.6+i*0.58, 2.4, 0.42, size=12, bold=True, color=col)
    box(s, 2.95, 5.58+i*0.58, 0.8, 0.38, col)
    txt(s, wt, 2.95, 5.6+i*0.58, 0.8, 0.34, size=12, bold=True, color=DARK_BG, align=PP_ALIGN.CENTER)
    txt(s, effect, 3.85, 5.6+i*0.58, 9.0, 0.42, size=11, color=WHITE)

# ══════════════════════════════════════════════════════════════
# SLIDE 17 — Job Search System
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Job Search System — 4 Providers, 1 Interface", "search_support.py — 849 lines")

txt(s, "Different API credentials = different providers. Same function interface hides the complexity.",
    0.4, 1.25, 12.5, 0.4, size=14, color=ACCENT2)

providers = [
    ("1. Adzuna API", ACCENT4, [
        "REST API — structured JSON responses",
        "Supports: role, location, salary, remote filter",
        "Country codes: gb (UK), us, in (India), ca, au",
        "Needs: ADZUNA_APP_ID + ADZUNA_APP_KEY",
        "Best for: UK/US jobs with salary data",
    ]),
    ("2. Apify (Indeed Scraper)", ACCENT1, [
        "Scrapes Indeed job listings",
        "Apify runs a headless browser in cloud",
        "Returns: title, company, description, URL",
        "Needs: APIFY_TOKEN",
        "Best for: Large volume, India jobs",
    ]),
    ("3. BrowserAct", ACCENT3, [
        "Automated browser workflow tool",
        "Runs predefined workflow in cloud browser",
        "Polls until workflow completes (async)",
        "Needs: BROWSERACT_API_KEY + WORKFLOW_ID",
        "Best for: Sites that block regular scraping",
    ]),
    ("4. Demo Mode", ACCENT2, [
        "No credentials needed",
        "Returns synthetic (fake) job data",
        "Used for testing without API costs",
        "Auto-activates when no credentials found",
        "Jobs are realistic but not real",
    ]),
]
for i, (name, col, bullets) in enumerate(providers):
    x = 0.4 + (i%2)*6.4
    y = 1.78 + (i//2)*2.7
    box(s, x, y, 6.2, 2.55, CARD_BG)
    txt(s, name, x+0.1, y+0.08, 6.0, 0.38, size=13, bold=True, color=col)
    for j, b in enumerate(bullets):
        txt(s, f"▸  {b}", x+0.2, y+0.55+j*0.38, 5.8, 0.35, size=10.5, color=WHITE)

txt(s, "Why 4 providers? Redundancy + coverage. Adzuna has salary data; Apify covers Indeed; BrowserAct bypasses anti-bot measures; Demo allows dev without spending money.",
    0.4, 7.05, 12.5, 0.38, size=11, color=ACCENT3, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 18 — search_tools.py Deep Dive
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "search_tools.py — The 3 Search Tools", "How the agent searches, filters, and fetches jobs")

tools_detail = [
    ("search_jobs()", ACCENT4, "Main search tool. Accepts role, location, max_results, experience_level.",
     ["Handles multi-role: 'Python or React' → splits and searches both",
      "Calls whichever provider has credentials (Adzuna→Apify→BrowserAct→Demo)",
      "Merges and deduplicates results across roles",
      "Saves results to session state (for follow-up questions)",
      "Returns: list of JobPosting dicts"],
     "search_jobs(role='Python Developer', location='Hyderabad', max_results=5)"),

    ("filter_saved_jobs_by_experience()", ACCENT1, "Filters ALREADY searched jobs by experience level. No new search.",
     ["User says: 'Show only entry level jobs' → calls this tool",
      "Reads previous results from session state",
      "Parses experience requirements from job description text",
      "Classifies: entry (0-2 yrs), mid (2-5 yrs), senior (5+ yrs)",
      "Returns filtered subset"],
     "filter_saved_jobs_by_experience(experience_level='entry')"),

    ("fetch_job_details()", ACCENT3, "Fetches full job description for a specific job URL.",
     ["Used when user wants more detail about a specific job",
      "Makes HTTP request to job URL",
      "Extracts full description (not just snippet)",
      "Returns complete job posting data",
      "Note: Not all providers support detail fetch"],
     "fetch_job_details(job_url='https://adzuna.com/jobs/123')"),
]

for i, (name, col, desc, bullets, example) in enumerate(tools_detail):
    y = 1.3 + i*1.95
    box(s, 0.4, y, 12.4, 1.82, CARD_BG)
    txt(s, name, 0.5, y+0.06, 3.5, 0.38, size=13, bold=True, color=col)
    txt(s, desc, 4.1, y+0.06, 8.5, 0.38, size=11.5, color=ACCENT2)
    for j, b in enumerate(bullets[:3]):
        txt(s, f"▸  {b}", 0.55, y+0.52+j*0.32, 7.8, 0.3, size=10, color=WHITE)
    code_box(s, 8.5, y+0.35, 4.2, 0.85, example, "Example call")

txt(s, "Session State is key: search_jobs saves results → filter_saved_jobs reads them. No need to search again!",
    0.4, 7.02, 12.5, 0.38, size=12, color=ACCENT3, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 19 — model_config.py
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "model_config.py — Multi-Model Architecture", "Why 3 different LLMs? Cost, capability, availability.")

txt(s, "Different tasks need different models. Using the right model saves cost without sacrificing quality.",
    0.4, 1.25, 12.5, 0.4, size=14, color=ACCENT2)

# Three model slots
slots = [
    ("chat_model", "Main conversation + tool calling",
     "Gemini 2.0 Flash → Groq Llama-3.3 → NVIDIA Nemotron",
     "Needs: tool calling, fast responses, long context",
     ACCENT3, "Priority: GOOGLE_API_KEY → GROQ_API_KEY → NVIDIA_NIM_API_KEY"),

    ("resume_text_model", "Structures resume text into JSON",
     "Same chain as chat_model (can use cheaper model)",
     "Needs: JSON output, instruction following",
     ACCENT1, "Can set JOB_SCOUT_RESUME_MODEL env var to override"),

    ("resume_attachment_model", "Reads PDF files visually",
     "ALWAYS Gemini (hard requirement)",
     "Needs: vision capability to 'see' PDFs",
     ACCENT4, "Why Gemini only? Other providers don't support PDF attachments"),
]

for i, (slot, purpose, model_chain, needs, col, note) in enumerate(slots):
    x = 0.3 + i*4.3
    box(s, x, 1.78, 4.1, 4.5, CARD_BG)
    txt(s, slot, x+0.1, 1.86, 3.9, 0.4, size=13, bold=True, color=col)
    txt(s, purpose, x+0.1, 2.32, 3.9, 0.4, size=11, color=ACCENT2)
    divider(s, 2.8, LIGHT_GRAY)
    txt(s, "Model Chain:", x+0.1, 2.88, 3.9, 0.28, size=10, bold=True, color=LIGHT_GRAY)
    txt(s, model_chain, x+0.1, 3.18, 3.9, 0.55, size=10, color=WHITE)
    txt(s, "Needs:", x+0.1, 3.8, 3.9, 0.28, size=10, bold=True, color=LIGHT_GRAY)
    txt(s, needs, x+0.1, 4.1, 3.9, 0.4, size=10, color=WHITE)
    box(s, x+0.05, 5.5, 4.0, 0.65, CODE_BG)
    txt(s, note, x+0.12, 5.55, 3.85, 0.55, size=9.5, color=ACCENT2, italic=True)

# Priority chain diagram
txt(s, "Model Priority Chain (how the code chooses):", 0.4, 6.35, 12, 0.32, size=12, bold=True, color=ACCENT3)
chain = ["JOB_SCOUT_MODEL\n(env var override)", "GOOGLE_API_KEY\n→ Gemini 2.0 Flash", "GROQ_API_KEY\n→ Llama-3.3-70b", "NVIDIA_NIM_API_KEY\n→ Nemotron-Super", "ERROR: no model\nconfigured"]
chain_colors = [ACCENT3, RGBColor(0x42,0x85,0xF4), ACCENT5, ACCENT4, ACCENT5]
for i, (c, col) in enumerate(zip(chain, chain_colors)):
    flow_box(s, 0.3+i*2.58, 6.75, 2.3, 0.65, c, fill=col, text_color=DARK_BG if col!=ACCENT5 else WHITE, size=9)
    if i < 4:
        txt(s, "→", 2.65+i*2.58, 6.88, 0.28, 0.4, size=14, color=LIGHT_GRAY)

# ══════════════════════════════════════════════════════════════
# SLIDE 20 — litellm_compat.py
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "litellm_compat.py — Making Different LLMs Work", "288 lines of compatibility glue")

txt(s, "Problem: Groq and NVIDIA don't follow OpenAI spec perfectly. This file patches ADK to handle the differences.",
    0.4, 1.25, 12.5, 0.4, size=14, color=ACCENT2)

patches = [
    ("_repair_missing_tool_call_ids()", ACCENT4,
     "What it fixes: Groq sometimes returns tool calls without tool_call_id (a required field).",
     "Why it matters: ADK checks for tool_call_id. Without it, the tool result can't be linked back to the tool call → conversation breaks.",
     "Fix: Walks the message history. If a tool call is missing an ID, generates one and back-fills it."),

    ("_normalize_groq_content_blocks()", ACCENT1,
     "What it fixes: Groq rejects content blocks with unsupported types (like 'image' or 'file').",
     "Why it matters: ADK builds messages with all content types. Groq errors out on unknown types.",
     "Fix: Before sending to Groq, rewrites content blocks to only include types Groq supports (text only)."),

    ("_provider_rejects_file_parts()", ACCENT3,
     "What it detects: Whether the current provider (NVIDIA) rejects file/binary content in messages.",
     "Why it matters: If resume PDF is in context and we switch to NVIDIA, it would crash.",
     "Fix: In agent.py _sanitize_for_nvidia() uses this to replace file content with '[Resume attached]' placeholder."),
]

for i, (name, col, what, why, fix) in enumerate(patches):
    box(s, 0.4, 1.78+i*1.75, 12.4, 1.65, CARD_BG)
    txt(s, name, 0.5, 1.85+i*1.75, 5.5, 0.38, size=12, bold=True, color=col)
    txt(s, what, 0.5, 2.28+i*1.75, 12.0, 0.32, size=11, color=ACCENT2)
    txt(s, why, 0.5, 2.62+i*1.75, 12.0, 0.32, size=10.5, color=LIGHT_GRAY)
    txt(s, f"✓ {fix}", 0.5, 2.96+i*1.75, 12.0, 0.32, size=10.5, color=ACCENT4)

txt(s, "Why monkey-patch instead of subclassing? ADK's LiteLLM bridge is third-party code we can't modify. Patching at runtime is the only clean option.",
    0.4, 7.08, 12.5, 0.38, size=11, color=ACCENT3, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 21 — Session State Concept
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Core Concept: Session State — Agent Memory", "How the agent remembers things across conversation turns")

txt(s, "Without session state, every message starts fresh. The agent would forget your resume and search results each turn.",
    0.4, 1.25, 12.5, 0.4, size=14, color=ACCENT2)

# Timeline of a conversation with state
txt(s, "Conversation Timeline with Session State:", 0.4, 1.78, 8, 0.32, size=13, bold=True, color=ACCENT3)

turns = [
    ("Turn 1", "You: 'Here is my resume'\nAgent: extract_resume_profile_from_artifact()\n→ Saves ResumeProfile to session state", ACCENT4),
    ("Turn 2", "You: 'Find Python jobs in Hyderabad'\nAgent: search_jobs() → finds 5 jobs\n→ Saves SearchContext to session state", ACCENT1),
    ("Turn 3", "You: 'Show only entry level'\nAgent: filter_saved_jobs_by_experience()\n→ Reads saved jobs from state (no new search!)", ACCENT3),
    ("Turn 4", "You: 'Score the top job for me'\nAgent: score_job_match()\n→ Reads ResumeProfile AND job from state", ACCENT2),
]
for i, (turn, desc, col) in enumerate(turns):
    box(s, 0.4+i*3.2, 2.18, 3.05, 2.3, CARD_BG)
    box(s, 0.4+i*3.2, 2.18, 3.05, 0.4, col)
    txt(s, turn, 0.45+i*3.2, 2.22, 2.95, 0.32, size=11, bold=True, color=DARK_BG, align=PP_ALIGN.CENTER)
    txt(s, desc, 0.5+i*3.2, 2.65, 2.9, 1.7, size=10, color=WHITE)

# State contents diagram
txt(s, "What lives in Session State:", 0.4, 4.65, 8, 0.32, size=13, bold=True, color=ACCENT3)
box(s, 0.4, 5.05, 12.4, 1.85, CODE_BG)
state_keys = [
    ("resume_profile", "ResumeProfile object — skills, years exp, name, role", ACCENT4),
    ("search_context", "SearchContext — what was searched, how many results", ACCENT3),
    ("saved_jobs", "List[JobPosting] — the actual job listings from last search", ACCENT1),
    ("last_search_role", "str — e.g. 'Python Developer' (for follow-up filters)", ACCENT2),
]
for i, (key, desc, col) in enumerate(state_keys):
    txt(s, key, 0.6, 5.15+i*0.42, 2.8, 0.38, size=11, bold=True, color=col)
    txt(s, f"→  {desc}", 3.5, 5.15+i*0.42, 9.1, 0.38, size=11, color=WHITE)

txt(s, "Stored in: tool_context.state (dict). ADK persists this across turns in the same session.",
    0.4, 7.05, 12.5, 0.35, size=11, color=LIGHT_GRAY, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 22 — Complete Data Flow (with resume)
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Complete Data Flow — Resume-Aware Job Search", "End-to-end: what happens when you upload a resume")

steps_flow = [
    ("User uploads\nresume.pdf\n+ 'Find React jobs'", ACCENT4, "Browser"),
    ("FastAPI receives\nmultipart request\n→ creates session", ACCENT1, "main.py"),
    ("ADK routes to\nroot_agent with\nfile attachment", RGBColor(0x42,0x85,0xF4), "ADK"),
    ("LLM sees file,\ncalls extract_resume\n_profile_from_artifact()", ACCENT3, "Gemini"),
    ("PDF parsed:\nlocal→LLM text\n→Gemini vision", ACCENT5, "resume\n_support.py"),
    ("ResumeProfile\nsaved to\nsession state", ACCENT4, "state"),
    ("LLM calls\nsearch_jobs(\nrole='React')", ACCENT3, "Gemini"),
    ("Adzuna/Apify\nreturns 5 jobs\n→ saved to state", ACCENT1, "search\n_support.py"),
    ("LLM calls\nscore_job_match\nfor each job", ACCENT3, "Gemini"),
    ("score_resume_vs_jd\nruns. Returns\n0-100 + blockers", ACCENT2, "scoring.py"),
    ("LLM formats\nranked results\n→ responds to user", ACCENT4, "Gemini"),
]

for i, (label, col, module) in enumerate(steps_flow):
    x = 0.2 + (i%6)*2.18
    y = 1.35 if i < 6 else 4.1
    flow_box(s, x, y, 2.0, 1.4, label, fill=col, text_color=DARK_BG, size=9)
    txt(s, module, x, y+1.45, 2.0, 0.28, size=8, color=LIGHT_GRAY, align=PP_ALIGN.CENTER, italic=True)
    # arrow
    if i < 5:
        txt(s, "→", x+2.05, y+0.5, 0.2, 0.4, size=14, color=WHITE)
    elif i == 5:
        txt(s, "↓  continues below  ↓", 5.5, 2.95, 3, 0.4, size=11, color=ACCENT2, italic=True)
    elif i > 5 and i < 10:
        txt(s, "→", x+2.05, y+0.5, 0.2, 0.4, size=14, color=WHITE)

txt(s, "Result: Ranked job list with match scores, matched skills, missing skills, and blockers specific to YOUR resume.",
    0.4, 6.8, 12.5, 0.4, size=13, bold=True, color=ACCENT4)

# ══════════════════════════════════════════════════════════════
# SLIDE 23 — Complete Data Flow (no resume)
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Complete Data Flow — Prompt-Only Mode (No Resume)", "What happens when you just describe your skills")

txt(s, "You don't HAVE to upload a resume. You can describe your skills in plain text. The agent adapts.",
    0.4, 1.25, 12.5, 0.4, size=14, color=ACCENT2)

box(s, 0.4, 1.75, 12.4, 1.65, CARD_BG)
txt(s, "Example user message (no resume):", 0.5, 1.82, 10, 0.32, size=12, bold=True, color=ACCENT3)
txt(s, '"I have 2 years of Python experience, know FastAPI and PostgreSQL, currently a backend developer. Find me senior Python roles in Bangalore."',
    0.5, 2.2, 12.0, 0.8, size=13, color=WHITE, italic=True)

txt(s, "What the agent does differently:", 0.4, 3.58, 10, 0.32, size=13, bold=True, color=ACCENT3)
steps_no_resume = [
    ("LLM extracts intent", "Parses: role=Python, location=Bangalore, level=senior from the message", ACCENT4),
    ("Calls search_jobs()", "Searches with extracted parameters. No resume in state.", ACCENT1),
    ("Scoring without resume", "score_job_match() called but ResumeProfile is minimal (only what user described)", ACCENT3),
    ("Basic fit verdict", "Score based on: title overlap + keyword match only (less precise than with resume)", ACCENT2),
    ("Agent suggests", "Agent recommends: 'Upload your resume for more accurate job matching'", ACCENT5),
]
for i, (step, desc, col) in enumerate(steps_no_resume):
    box(s, 0.4, 4.0+i*0.58, 12.4, 0.52, CARD_BG if i%2==0 else DARK_BG)
    txt(s, step, 0.5, 4.06+i*0.58, 2.8, 0.42, size=11, bold=True, color=col)
    txt(s, desc, 3.4, 4.06+i*0.58, 9.3, 0.42, size=11, color=WHITE)

box(s, 0.4, 6.98, 12.4, 0.42, CARD_BG)
txt(s, "Key design: The agent works both ways. Resume mode = precise scoring. Prompt mode = quick discovery.",
    0.5, 7.05, 12.0, 0.32, size=12, color=ACCENT3, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 24 — Testing Strategy
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Testing Strategy — Why and How We Test", "test_tools.py (1039 lines) + test_edge_cases.py (517 lines)")

txt(s, "Tests catch bugs before users do. In AI systems, they also validate that deterministic logic stays correct when models change.",
    0.4, 1.25, 12.5, 0.4, size=14, color=ACCENT2)

test_types = [
    ("Unit Tests\n(test_tools.py)", ACCENT4, [
        "Test each tool function in isolation",
        "Mock external APIs (don't actually call Adzuna)",
        "Verify tool call ID repair (litellm_compat)",
        "Verify Groq content normalization",
        "Verify resume placeholder rejection",
        "Fast: runs in milliseconds",
    ]),
    ("Edge Case Tests\n(test_edge_cases.py)", ACCENT1, [
        "'in' as preposition vs India country code",
        "Experience '0-1 years' vs '0 years'",
        "Negative years (invalid input handling)",
        "Multi-role search deduplication",
        "Score consistency across identical inputs",
        "PDF with hex-encoded text only",
    ]),
    ("Evaluation Tests\n(evaluation.py)", ACCENT3, [
        "Scoring quality validation with fixtures",
        "Verify score is within expected range",
        "Verify correct job ranked #1 in set",
        "Verify experience filter works correctly",
        "JSON fixtures = easy to add new test cases",
        "Slow: measures algorithmic correctness",
    ]),
    ("Model Config Tests\n(test_model_config.py)", ACCENT2, [
        "Verify model priority chain order",
        "Verify reasoning model downgrading",
        "Test with each env var combination",
        "No actual API calls made",
        "Ensures model selection is predictable",
        "",
    ]),
]
for i, (title, col, bullets) in enumerate(test_types):
    x = 0.4 + (i%2)*6.4
    y = 1.8 + (i//2)*2.65
    box(s, x, y, 6.2, 2.5, CARD_BG)
    txt(s, title, x+0.1, y+0.08, 6.0, 0.55, size=13, bold=True, color=col)
    for j, b in enumerate([b for b in bullets if b]):
        txt(s, f"▸  {b}", x+0.2, y+0.7+j*0.34, 5.8, 0.32, size=10.5, color=WHITE)

txt(s, "Run all tests: python -m pytest tests/ -v", 0.4, 7.1, 12.5, 0.3, size=12, color=ACCENT4)

# ══════════════════════════════════════════════════════════════
# SLIDE 25 — Environment Variables & Config
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Environment Variables — Configuring the Agent", ".env file — secrets and feature toggles")

txt(s, "Environment variables separate secrets from code. Never hardcode API keys — they'd be exposed in version control.",
    0.4, 1.25, 12.5, 0.4, size=14, color=ACCENT2)

env_vars = [
    # (name, example_value, required, description, category)
    ("GOOGLE_API_KEY", "AIzaSy...", True, "Enables Gemini models + resume attachment parsing", "LLM"),
    ("GROQ_API_KEY", "gsk_...", False, "Enables Groq Llama models (free tier available)", "LLM"),
    ("NVIDIA_NIM_API_KEY", "nvapi-...", False, "Enables NVIDIA Nemotron models", "LLM"),
    ("ADZUNA_APP_ID", "abc123", False, "Adzuna job search API application ID", "Jobs"),
    ("ADZUNA_APP_KEY", "xyz789", False, "Adzuna job search API secret key", "Jobs"),
    ("APIFY_TOKEN", "apfy_...", False, "Apify token for Indeed scraper actor", "Jobs"),
    ("BROWSERACT_API_KEY", "bact_...", False, "BrowserAct API key for browser automation", "Jobs"),
    ("BROWSERACT_WORKFLOW_ID", "wf_123", False, "Which BrowserAct workflow to run", "Jobs"),
    ("JOB_SCOUT_MODEL", "gemini-2.0-flash", False, "Override model selection (skips priority chain)", "Config"),
    ("ADZUNA_COUNTRY", "in", False, "Default country for Adzuna search (default: gb)", "Config"),
]

headers_env = ["Variable", "Example", "Required?", "Description", "Category"]
col_widths = [3.2, 1.8, 1.0, 5.8, 1.1]
col_xs = [0.3, 3.55, 5.4, 6.45, 12.3]

# Header row
for j, (h, x, w) in enumerate(zip(headers_env, col_xs, col_widths)):
    box(s, x, 1.78, w, 0.38, ACCENT1)
    txt(s, h, x+0.05, 1.82, w-0.1, 0.3, size=10, bold=True, color=DARK_BG)

for i, (name, ex, req, desc, cat) in enumerate(env_vars):
    bg_c = CARD_BG if i%2==0 else DARK_BG
    row_y = 2.2 + i*0.46
    box(s, 0.3, row_y, 12.5, 0.44, bg_c)
    req_col = ACCENT4 if req else LIGHT_GRAY
    req_txt = "YES" if req else "no"
    cat_col = ACCENT3 if cat=="LLM" else ACCENT1
    for j, (val, x, w, col) in enumerate(zip(
        [name, ex, req_txt, desc, cat],
        col_xs, col_widths,
        [ACCENT3, WHITE, req_col, WHITE, cat_col]
    )):
        txt(s, val, x+0.05, row_y+0.06, w-0.1, 0.34, size=9.5, color=col, bold=(j==0))

txt(s, "No job API keys? → Demo mode activates automatically with synthetic data. Great for learning without spending money.",
    0.4, 6.95, 12.5, 0.38, size=12, color=ACCENT4, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 26 — Key Design Decisions Summary
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Key Design Decisions — Why Not the Other Way?", "Every architectural choice has a reason")

decisions = [
    ("Google ADK\nnot LangChain", ACCENT1,
     "ADK has native Gemini support, better file handling, and simpler tool registration. LangChain has more community resources but adds abstraction overhead."),
    ("Pydantic\nnot raw dicts", ACCENT4,
     "Pydantic catches type errors at the boundary (parse time) instead of at use time. With AI outputs, validation is critical — LLMs can return unexpected structures."),
    ("FastAPI\nnot Flask", ACCENT3,
     "FastAPI has native async support (needed for concurrent tool calls) and automatic Pydantic validation. Flask is simpler but synchronous by default."),
    ("Local PDF\nextraction first", ACCENT2,
     "API calls cost money and latency. 80% of PDFs can be parsed locally in milliseconds. Fallback to LLM only when needed saves both cost and time."),
    ("Session state\nnot re-search", ACCENT5,
     "Re-searching on every follow-up question wastes API quota and slows responses. Storing results once and filtering locally is faster and cheaper."),
    ("Demo mode\nfor dev/test", ACCENT4,
     "Requiring real API credentials to run any code creates a high barrier to learning. Demo mode lets you explore the full flow without spending money."),
]

for i, (title, col, reason) in enumerate(decisions):
    x = 0.4 + (i%3)*4.3
    y = 1.4 + (i//3)*2.65
    box(s, x, y, 4.1, 2.45, CARD_BG)
    box(s, x, y, 4.1, 0.7, col)
    txt(s, title, x+0.1, y+0.1, 3.9, 0.55, size=13, bold=True, color=DARK_BG, align=PP_ALIGN.CENTER)
    txt(s, reason, x+0.12, y+0.82, 3.85, 1.5, size=10.5, color=WHITE)

txt(s, "Rule of thumb: every design choice in this project favors debuggability and cost-efficiency over novelty.",
    0.4, 7.08, 12.5, 0.38, size=12, color=ACCENT3, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 27 — Concepts Summary (what you learned)
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Concepts You Learned in This Project", "A checklist of your new knowledge")

concepts_learned = [
    ("AI / LLM Concepts", ACCENT1, [
        "What LLMs are and how they generate text (token prediction)",
        "Temperature — controlling randomness in LLM outputs",
        "System prompts — giving instructions to an LLM",
        "Tool calling (function calling) — LLM requesting code execution",
        "Context window — how agents 'remember' conversation history",
    ]),
    ("Agent Architecture", ACCENT4, [
        "Think → Act → Observe loop",
        "What a framework (ADK) does for you vs what you write",
        "Session state — persistent memory across conversation turns",
        "before_model_callback — dynamic prompt injection",
        "Multi-provider LLM routing (Gemini, Groq, NVIDIA)",
    ]),
    ("Python Engineering", ACCENT3, [
        "Pydantic for data validation and schema enforcement",
        "Type hints and docstrings for LLM-readable tool definitions",
        "Fallback chains (try A → B → C for reliability)",
        "Monkey-patching for third-party library compatibility",
        "REST APIs — calling Adzuna, Apify with HTTP requests",
    ]),
    ("System Design", ACCENT2, [
        "When to use AI vs deterministic algorithms (scoring!)",
        "Environment variables for secrets and config",
        "Demo mode for development without real API costs",
        "Multi-provider design for redundancy and cost control",
        "Skill ontology — domain knowledge encoded as weights",
    ]),
]

for i, (cat, col, items) in enumerate(concepts_learned):
    x = 0.4 + (i%2)*6.4
    y = 1.4 + (i//2)*2.7
    box(s, x, y, 6.2, 2.55, CARD_BG)
    box(s, x, y, 6.2, 0.42, col)
    txt(s, cat, x+0.1, y+0.06, 6.0, 0.32, size=13, bold=True, color=DARK_BG, align=PP_ALIGN.CENTER)
    for j, item in enumerate(items):
        txt(s, f"✓  {item}", x+0.15, y+0.52+j*0.38, 5.9, 0.35, size=10.5, color=WHITE)

txt(s, "You built something production-grade. Most developers with 5+ years haven't built a multi-provider AI agent.",
    0.4, 7.1, 12.5, 0.35, size=13, bold=True, color=ACCENT4, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 28 — How to Run the Project
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "How to Run the Project", "Step-by-step setup guide")

steps_run = [
    ("Step 1: Install Dependencies", "pip install -r requirements.txt", ACCENT4),
    ("Step 2: Create .env file", 'GOOGLE_API_KEY=your_key_here\nADZUNA_APP_ID=your_id\nADZUNA_APP_KEY=your_key', ACCENT1),
    ("Step 3: Run the server", "python main.py", ACCENT3),
    ("Step 4: Open browser", "http://localhost:8000  (or the URL shown in terminal)", ACCENT2),
    ("Step 5: Run tests", "python -m pytest tests/ -v", ACCENT5),
]
for i, (title, cmd, col) in enumerate(steps_run):
    box(s, 0.4, 1.3+i*1.13, 12.4, 1.0, CARD_BG)
    badge2 = s.shapes.add_shape(9, Inches(0.5), Inches(1.4+i*1.13), Inches(0.52), Inches(0.52))
    badge2.fill.solid(); badge2.fill.fore_color.rgb = col; badge2.line.fill.background()
    txt(s, str(i+1), 0.5, 1.42+i*1.13, 0.52, 0.38, size=14, bold=True, color=DARK_BG, align=PP_ALIGN.CENTER)
    txt(s, title, 1.2, 1.38+i*1.13, 4.5, 0.35, size=12, bold=True, color=col)
    code_box(s, 5.8, 1.35+i*1.13, 7.2, 0.88, cmd)

txt(s, "No job API keys? The agent runs in Demo Mode automatically — you'll see synthetic jobs. Great for testing the full flow!",
    0.4, 7.02, 12.5, 0.38, size=12, color=ACCENT4, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 29 — Next Steps for Learning
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)
heading(s, "Next Steps — What to Build/Learn Next", "Your learning roadmap from here")

roadmap = [
    ("Beginner Next Steps", ACCENT4, [
        "Read LEARN.md in the project — 1,592 lines of detailed explanations",
        "Add a new tool: e.g. save_job_to_wishlist() that saves interesting jobs",
        "Modify the scoring weights in scoring.py and see how scores change",
        "Add a new job provider to search_support.py (e.g., LinkedIn scraper)",
        "Write 5 new test cases in test_tools.py for edge cases you discover",
    ]),
    ("Intermediate Projects", ACCENT1, [
        "Build a cover letter generator agent that uses your resume + job description",
        "Add email alerts: agent emails you when a new matching job appears",
        "Add a database (SQLite) to persist jobs and scores across sessions",
        "Build a simple React frontend instead of the default ADK UI",
        "Deploy to Google Cloud Run or Railway.app (real production deploy)",
    ]),
    ("Advanced Concepts to Explore", ACCENT3, [
        "RAG (Retrieval-Augmented Generation) — searching a vector DB for similar jobs",
        "Multi-agent systems — orchestrator agent + specialist sub-agents",
        "Streaming responses — show LLM output word-by-word like ChatGPT",
        "Agent evaluation frameworks — systematically measuring agent quality",
        "Fine-tuning: train your own model on job-matching data",
    ]),
]

for i, (title, col, items) in enumerate(roadmap):
    box(s, 0.4+i*4.3, 1.4, 4.1, 5.5, CARD_BG)
    box(s, 0.4+i*4.3, 1.4, 4.1, 0.45, col)
    txt(s, title, 0.5+i*4.3, 1.47, 3.9, 0.35, size=12, bold=True, color=DARK_BG, align=PP_ALIGN.CENTER)
    for j, item in enumerate(items):
        txt(s, f"▸  {item}", 0.55+i*4.3, 1.98+j*0.87, 3.85, 0.82, size=10.5, color=WHITE)

txt(s, "Resources: Google ADK docs · Pydantic docs · FastAPI docs · Gemini API docs · LEARN.md in this project",
    0.4, 7.05, 12.5, 0.38, size=11.5, color=LIGHT_GRAY, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 30 — Final Summary / Thank You
# ══════════════════════════════════════════════════════════════
s = add_slide(); bg(s)

big2 = s.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.33), Inches(7.5))
big2.fill.solid(); big2.fill.fore_color.rgb = DARK_BG; big2.line.fill.background()

txt(s, "You Built This.", 1, 0.8, 11, 1.0, size=52, bold=True, color=ACCENT4, align=PP_ALIGN.CENTER)
txt(s, "A production-grade, multi-provider AI agent with:", 1, 1.9, 11, 0.5, size=18, color=ACCENT2, align=PP_ALIGN.CENTER)
divider(s, 2.55, ACCENT3)

summary_items = [
    "LLM-powered conversation (Google Gemini / Groq / NVIDIA)", ACCENT1,
    "9 registered tools with automatic tool-calling loop", ACCENT4,
    "3-layer resume parsing (local PDF → LLM text → Gemini vision)", ACCENT3,
    "4 job search providers (Adzuna, Apify, BrowserAct, Demo)", ACCENT2,
    "Deterministic skill-based scoring algorithm (0-100)", ACCENT5,
    "Session state for multi-turn conversations", ACCENT1,
    "Full test suite (1500+ lines of tests)", ACCENT4,
]

for i in range(0, len(summary_items), 2):
    item = summary_items[i]
    col = summary_items[i+1]
    txt(s, f"✓  {item}", 1.2, 2.75+(i//2)*0.58, 11, 0.5, size=14, color=col, align=PP_ALIGN.CENTER)

divider(s, 6.45, ACCENT3)
txt(s, "Keep building. Every expert was once a beginner who refused to stop.",
    1, 6.6, 11, 0.5, size=16, color=WHITE, italic=True, align=PP_ALIGN.CENTER)
txt(s, "Job Scout Agent  ·  Built with Google ADK + Gemini  ·  30 Slides",
    1, 7.1, 11, 0.35, size=11, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════
out_path = r"c:\Users\Dell\OneDrive\Desktop\agents\job-sout-agent\JobScoutAgent_Complete_Guide.pptx"
prs.save(out_path)
print(f"Saved: {out_path}")
print(f"Total slides: {len(prs.slides)}")
