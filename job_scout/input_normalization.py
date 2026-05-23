from __future__ import annotations

import difflib
import re

_ROLE_CANONICAL_ALIASES = {
    "ai developer": "GenAI Engineer",
    "ai engineer": "GenAI Engineer",
    "backend": "Backend Developer",
    "backend developer": "Backend Developer",
    "backend engineer": "Backend Developer",
    "back end": "Backend Developer",
    "back end developer": "Backend Developer",
    "back end engineer": "Backend Developer",
    "front end": "Frontend Developer",
    "front end developer": "Frontend Developer",
    "front end engineer": "Frontend Developer",
    "frontend": "Frontend Developer",
    "frontend developer": "Frontend Developer",
    "frontend engineer": "Frontend Developer",
    "full stack": "Full Stack Developer",
    "full stack developer": "Full Stack Developer",
    "full stack engineer": "Full Stack Developer",
    "fullstack": "Full Stack Developer",
    "fullstack developer": "Full Stack Developer",
    "fullstack engineer": "Full Stack Developer",
    "gen ai": "GenAI Engineer",
    "genai": "GenAI Engineer",
    "genai developer": "GenAI Engineer",
    "genai engineer": "GenAI Engineer",
    "llm developer": "GenAI Engineer",
    "llm engineer": "GenAI Engineer",
    "mern": "MERN Stack Developer",
    "mern developer": "MERN Stack Developer",
    "mern stack": "MERN Stack Developer",
    "mern stack developer": "MERN Stack Developer",
    "node developer": "Node.js Developer",
    "node engineer": "Node.js Developer",
    "node js developer": "Node.js Developer",
    "node js engineer": "Node.js Developer",
    "nodejs developer": "Node.js Developer",
    "nodejs engineer": "Node.js Developer",
    "react": "React Developer",
    "react developer": "React Developer",
    "react engineer": "React Developer",
}
_ROLE_VOCABULARY = {
    "ai",
    "back",
    "backend",
    "data",
    "developer",
    "devops",
    "engineer",
    "end",
    "front",
    "frontend",
    "full",
    "fullstack",
    "genai",
    "java",
    "javascript",
    "js",
    "llm",
    "mern",
    "mobile",
    "node",
    "nodejs",
    "python",
    "react",
    "software",
    "stack",
    "typescript",
    "web",
}
_ROLE_NOISE_TERMS = {
    "job",
    "jobs",
    "opening",
    "openings",
    "position",
    "positions",
    "role",
    "roles",
}
_ROLE_VOCAB_BY_KEY = {
    re.sub(r"[^a-z0-9]+", "", term.lower()): term for term in _ROLE_VOCABULARY
}
_ROLE_VOCAB_KEYS = tuple(_ROLE_VOCAB_BY_KEY.keys())
_ROLE_ALIAS_KEYS = tuple(_ROLE_CANONICAL_ALIASES.keys())


def _simplify_text(text: str) -> str:
    normalized = re.sub(r"node\s*\.\s*js", "node js", text, flags=re.IGNORECASE)
    normalized = re.sub(r"\bgen\s+ai\b", "genai", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("&", " and ").replace("/", " / ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized.lower())
    return " ".join(normalized.split())


def _smart_title(text: str) -> str:
    titled = " ".join(word.capitalize() for word in _simplify_text(text).split())
    return (
        titled.replace("Javascript", "JavaScript")
        .replace("Typescript", "TypeScript")
        .replace("Devops", "DevOps")
        .replace("Node Js", "Node.js")
    )


def _repair_role_words(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        token_key = re.sub(r"[^a-z0-9]+", "", token.lower())
        if len(token_key) < 4:
            return token.lower()

        if token_key in _ROLE_VOCAB_BY_KEY:
            return _ROLE_VOCAB_BY_KEY[token_key]

        close_matches = difflib.get_close_matches(token_key, _ROLE_VOCAB_KEYS, n=1, cutoff=0.84)
        if close_matches:
            return _ROLE_VOCAB_BY_KEY[close_matches[0]]
        return token.lower()

    return re.sub(r"[A-Za-z][A-Za-z.+-]*", replace, text or "")


def normalize_role_title(role: str) -> str:
    cleaned = " ".join((role or "").split())
    if not cleaned:
        return ""

    repaired = _repair_role_words(cleaned)
    simplified = _simplify_text(repaired)
    if not simplified:
        return ""

    filtered_tokens = [token for token in simplified.split() if token not in _ROLE_NOISE_TERMS]
    simplified = " ".join(filtered_tokens)
    if not simplified:
        return ""

    canonical = _ROLE_CANONICAL_ALIASES.get(simplified)
    if canonical:
        return canonical

    close_matches = difflib.get_close_matches(simplified, _ROLE_ALIAS_KEYS, n=1, cutoff=0.86)
    if close_matches:
        return _ROLE_CANONICAL_ALIASES[close_matches[0]]

    return _smart_title(repaired)


def normalize_role_titles(roles: list[str]) -> list[str]:
    normalized_roles = []
    seen = set()

    for role in roles:
        normalized = normalize_role_title(role)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized_roles.append(normalized)

    return normalized_roles
