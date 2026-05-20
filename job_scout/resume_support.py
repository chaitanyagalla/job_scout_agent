from __future__ import annotations

import base64
import json
import os
import re
import zlib
from typing import Optional

from google.adk.artifacts import artifact_util
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError

from job_scout.domain_models import ResumeProfile
from job_scout.model_config import resolve_resume_attachment_model
from job_scout.model_config import resolve_resume_parser_model
from job_scout.model_config import uses_litellm
from job_scout.scoring import SKILL_ONTOLOGY

try:
    from litellm import acompletion as litellm_acompletion
except Exception:  # pragma: no cover - exercised only when LiteLLM is unavailable.
    litellm_acompletion = None

_PLACEHOLDER_VALUES = {
    "",
    "unknown",
    "n/a",
    "na",
    "none",
    "null",
    "not specified",
    "not provided",
}
_RESUME_FILENAME_PATTERNS = (
    re.compile(r"\.pdf$", re.IGNORECASE),
    re.compile(r"\.docx?$", re.IGNORECASE),
    re.compile(r"\.(txt|md|rtf|html?)$", re.IGNORECASE),
)
_FRONTEND_ROLE_SKILLS = {
    "react", "react.js", "next.js", "javascript", "typescript", "html", "css",
    "tailwind css", "tailwind", "shadcn ui", "responsive design",
}
_BACKEND_ROLE_SKILLS = {
    "node.js", "node", "express", "express.js", "rest api", "rest apis",
    "postgresql", "postgres", "mongodb", "redis", "aws", "docker", "nginx",
    "prisma orm", "socket.io", "websockets",
}
_ROLE_PRIORITY = (
    "Full Stack Developer",
    "Frontend Developer",
    "Backend Developer",
    "React Developer",
    "Node.js Developer",
)
_PDF_STREAM_PATTERN = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
_PDF_LITERAL_TEXT_PATTERN = re.compile(r"\((?:\\.|[^\\)])*\)\s*(?:Tj|')", re.DOTALL)
_PDF_ARRAY_TEXT_PATTERN = re.compile(r"\[(.*?)\]\s*TJ", re.DOTALL)
_PDF_HEX_TEXT_PATTERN = re.compile(r"<([0-9A-Fa-f\s]+)>")
_SECTION_HEADING_ALIASES = {
    "professional summary": "professional_summary",
    "summary": "professional_summary",
    "technical skills": "technical_skills",
    "skills": "technical_skills",
    "experience": "experience",
    "projects": "projects",
}
_ROLE_TEXT_PATTERNS = (
    (re.compile(r"\bfull[\s-]*stack developer\b", re.IGNORECASE), "Full Stack Developer"),
    (re.compile(r"\bfront[\s-]*end developer\b", re.IGNORECASE), "Frontend Developer"),
    (re.compile(r"\bback[\s-]*end developer\b", re.IGNORECASE), "Backend Developer"),
    (re.compile(r"\breact developer\b", re.IGNORECASE), "React Developer"),
    (re.compile(r"\bnode(?:\.js)? developer\b", re.IGNORECASE), "Node.js Developer"),
)
_DISPLAY_SKILL_PATTERNS = (
    ("React", ("react", "react.js", "reactjs")),
    ("Next.js", ("next.js", "nextjs", "next js")),
    ("JavaScript", ("javascript", "js")),
    ("TypeScript", ("typescript", "ts")),
    ("Tailwind CSS", ("tailwind css", "tailwindcss", "tailwind")),
    ("HTML5", ("html5", "html")),
    ("CSS3", ("css3", "css")),
    ("Node.js", ("node.js", "nodejs", "node js")),
    ("Express.js", ("express.js", "expressjs", "express")),
    ("REST API", ("rest api", "rest apis", "restful api")),
    ("PostgreSQL", ("postgresql", "postgres")),
    ("MongoDB", ("mongodb", "mongo db")),
    ("Redis", ("redis",)),
    ("SQLite", ("sqlite",)),
    ("Prisma ORM", ("prisma orm", "prisma")),
    ("Socket.io", ("socket.io", "socketio")),
    ("WebSockets", ("websockets", "web sockets")),
    ("JWT", ("jwt",)),
    ("AWS", ("aws", "amazon web services")),
    ("Docker", ("docker",)),
    ("CI/CD", ("ci/cd", "cicd", "ci cd")),
    ("GitHub Actions", ("github actions",)),
    ("Linux", ("linux",)),
    ("Nginx", ("nginx",)),
    ("FFmpeg", ("ffmpeg",)),
    ("Zod", ("zod",)),
    ("ShadCN UI", ("shadcn ui", "shadcn")),
)
_SUPPORTED_RESUME_MIME_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "text/rtf",
    "text/html",
}


class ResumeProfileExtraction(BaseModel):
    candidate_name: str = ""
    professional_summary: str = ""
    role_titles: list[str] = Field(default_factory=list)
    core_skills: list[str] = Field(default_factory=list)
    additional_skills: list[str] = Field(default_factory=list)
    years_experience: Optional[float] = None
    preferred_locations: list[str] = Field(default_factory=list)
    work_preferences: list[str] = Field(default_factory=list)
    notable_projects: list[str] = Field(default_factory=list)


_RESUME_PARSER_PROMPT = (
        "Extract a structured resume profile from the provided resume text. "
        "Use only information explicitly supported by the text. "
        "Do not invent names, skills, roles, projects, locations, or work preferences. "
        "For missing fields, return empty strings, empty lists, or null. "
        "Prefer precise web-development roles over broad generic labels. "
    "If the resume supports them, include roles like Full Stack Developer, "
    "Frontend Developer, Backend Developer, React Developer, and Node.js Developer. "
        "Estimate years_experience conservatively from the documented work history only. "
        "Return valid JSON only with exactly these keys: "
        "candidate_name, professional_summary, role_titles, core_skills, "
        "additional_skills, years_experience, preferred_locations, "
        "work_preferences, notable_projects. "
        "The notable_projects field must be an array of plain strings, not objects."
)


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        value = item.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _clean_resume_scalar(value: object) -> str:
    normalized = str(value or "").strip()
    if normalized.lower() in _PLACEHOLDER_VALUES:
        return ""
    return normalized


def _clean_resume_items(items: list[str]) -> list[str]:
    if not isinstance(items, list):
        items = [items] if items else []
    cleaned = []
    for item in items:
        value = _clean_resume_scalar(item)
        if value:
            cleaned.append(value)
    return _dedupe(cleaned)


def _project_payload_to_string(value: object) -> str:
    if isinstance(value, str):
        return _clean_resume_scalar(value)
    if not isinstance(value, dict):
        return ""

    title = _clean_resume_scalar(str(value.get("title") or value.get("name") or ""))
    summary = _clean_resume_scalar(str(value.get("summary") or value.get("description") or ""))
    stack = value.get("tech_stack") or value.get("technologies") or value.get("skills") or []
    if isinstance(stack, list):
        stack_text = ", ".join(
            _clean_resume_scalar(str(item))
            for item in stack
            if _clean_resume_scalar(str(item))
        )
    else:
        stack_text = _clean_resume_scalar(str(stack or ""))

    if title and stack_text:
        return f"{title} | {stack_text}"
    if title and summary:
        return f"{title} - {summary}"
    return title or summary or stack_text


def _clean_notable_projects(items: list[object]) -> list[str]:
    if not isinstance(items, list):
        items = [items] if items else []
    cleaned = []
    for item in items:
        value = _project_payload_to_string(item)
        if value:
            cleaned.append(value)
    return _dedupe(cleaned)


def _normalize_years_experience(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None

    years = float(value)
    if years < 0:
        raise ValueError("Invalid years_experience: must be greater than or equal to 0.")
    return years


def _safe_decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def _artifact_part_to_bytes(artifact_part: genai_types.Part) -> tuple[Optional[bytes], Optional[str]]:
    file_data = getattr(artifact_part, "file_data", None)
    if file_data is not None:
        file_bytes = getattr(file_data, "data", None)
        mime_type = (getattr(file_data, "mime_type", None) or "").split(";", 1)[0].strip().lower() or None
        if isinstance(file_bytes, bytes):
            return file_bytes, mime_type
        if isinstance(file_bytes, str):
            try:
                return base64.b64decode(file_bytes, validate=True), mime_type
            except Exception:
                return file_bytes.encode("utf-8"), mime_type

    if getattr(artifact_part, "inline_data", None) is None:
        text = getattr(artifact_part, "text", None)
        if text:
            return text.encode("utf-8"), "text/plain"
        return None, None

    inline_data = artifact_part.inline_data
    data = inline_data.data
    mime_type = (inline_data.mime_type or "").split(";", 1)[0].strip().lower() or None
    if data is None:
        return None, mime_type
    if isinstance(data, bytes):
        return data, mime_type
    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=True), mime_type
        except Exception:
            return data.encode("utf-8"), mime_type
    return None, mime_type


def _decode_pdf_literal_string(token: str) -> str:
    body = token[1:token.rfind(")")]
    output = []
    index = 0
    while index < len(body):
        char = body[index]
        if char != "\\":
            output.append(char)
            index += 1
            continue

        index += 1
        if index >= len(body):
            break
        escaped = body[index]
        replacements = {
            "n": "\n",
            "r": "\r",
            "t": "\t",
            "b": "\b",
            "f": "\f",
            "\\": "\\",
            "(": "(",
            ")": ")",
        }
        if escaped in replacements:
            output.append(replacements[escaped])
            index += 1
            continue
        if escaped in "\r\n":
            while index < len(body) and body[index] in "\r\n":
                index += 1
            continue
        if escaped in "01234567":
            octal_digits = escaped
            index += 1
            for _ in range(2):
                if index < len(body) and body[index] in "01234567":
                    octal_digits += body[index]
                    index += 1
                else:
                    break
            output.append(chr(int(octal_digits, 8)))
            continue
        if escaped.isdigit():
            output.append(escaped)
            index += 1
            continue

        output.append(escaped)
        index += 1

    return "".join(output)


def _decode_pdf_hex_string(raw_hex: str) -> str:
    cleaned = re.sub(r"\s+", "", raw_hex)
    if len(cleaned) % 2 == 1:
        cleaned += "0"
    try:
        decoded = bytes.fromhex(cleaned)
    except ValueError:
        return ""
    try:
        return decoded.decode("utf-16-be")
    except UnicodeDecodeError:
        return _safe_decode_text(decoded)


def _extract_text_from_pdf_stream_text(stream_text: str) -> list[str]:
    fragments: list[str] = []
    for match in _PDF_LITERAL_TEXT_PATTERN.finditer(stream_text):
        text = _decode_pdf_literal_string(match.group(0))
        if text.strip():
            fragments.append(text.strip())

    for match in _PDF_ARRAY_TEXT_PATTERN.finditer(stream_text):
        array_body = match.group(1)
        pieces = []
        for literal_match in re.finditer(r"\((?:\\.|[^\\)])*\)", array_body, re.DOTALL):
            decoded = _decode_pdf_literal_string(literal_match.group(0))
            if decoded:
                pieces.append(decoded)
        for hex_match in _PDF_HEX_TEXT_PATTERN.finditer(array_body):
            decoded = _decode_pdf_hex_string(hex_match.group(1))
            if decoded:
                pieces.append(decoded)
        joined = " ".join(piece.strip() for piece in pieces if piece.strip())
        if joined:
            fragments.append(joined)

    return fragments


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    fragments: list[str] = []
    for match in _PDF_STREAM_PATTERN.finditer(pdf_bytes):
        stream_data = match.group(1)
        candidate_streams = [stream_data]
        try:
            candidate_streams.insert(0, zlib.decompress(stream_data))
        except Exception:
            pass

        for candidate in candidate_streams:
            stream_text = _safe_decode_text(candidate)
            fragments.extend(_extract_text_from_pdf_stream_text(stream_text))

    if not fragments:
        raw_text = _safe_decode_text(pdf_bytes)
        fragments.extend(_extract_text_from_pdf_stream_text(raw_text))

    cleaned_lines = []
    seen = set()
    for fragment in fragments:
        normalized = " ".join(fragment.replace("\x00", " ").split())
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned_lines.append(normalized)

    return "\n".join(cleaned_lines)


def _looks_like_resume_text(text: str) -> bool:
    normalized = " ".join((text or "").split())
    if len(normalized) < 200:
        return False

    tokens = re.findall(r"[A-Za-z]{3,}", normalized)
    if len(tokens) < 30:
        return False

    keyword_hits = sum(
        1
        for keyword in ("experience", "skills", "projects", "developer", "summary")
        if keyword in normalized.lower()
    )
    return keyword_hits >= 2


def _profile_has_meaningful_resume_signal(profile: dict) -> bool:
    return bool(
        profile.get("professional_summary")
        or profile.get("role_titles")
        or profile.get("core_skills")
        or profile.get("notable_projects")
    )


def _normalize_resume_lines(text: str) -> list[str]:
    lines = []
    for raw_line in re.split(r"[\r\n]+", text or ""):
        line = " ".join(raw_line.replace("\x00", " ").split())
        if not line:
            continue
        lines.append(line)
    return lines


def _split_resume_sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"intro": []}
    current_section = "intro"
    for line in lines:
        normalized = line.strip().lower().rstrip(":")
        matched_section = _SECTION_HEADING_ALIASES.get(normalized)
        if matched_section:
            current_section = matched_section
            sections.setdefault(current_section, [])
            continue
        sections.setdefault(current_section, []).append(line)
    return sections


def _extract_candidate_name(lines: list[str]) -> str:
    for line in lines[:5]:
        if any(char.isdigit() for char in line):
            continue
        if "@" in line or "linkedin" in line.lower() or "github" in line.lower():
            continue
        words = [word for word in re.split(r"\s+", line) if word]
        if 2 <= len(words) <= 5:
            return line.title()
    return ""


def _sort_role_titles(role_titles: list[str]) -> list[str]:
    priority_map = {role: index for index, role in enumerate(_ROLE_PRIORITY)}
    return sorted(
        _dedupe(role_titles),
        key=lambda role: (priority_map.get(role, len(priority_map)), role.lower()),
    )


def _extract_roles_from_text(text: str) -> list[str]:
    roles = []
    for pattern, role in _ROLE_TEXT_PATTERNS:
        if pattern.search(text):
            roles.append(role)
    return _sort_role_titles(roles)


def _extract_resume_skills(text: str) -> list[str]:
    normalized = text.lower()
    found = []
    seen = set()

    for display_name, variants in _DISPLAY_SKILL_PATTERNS:
        if any(re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", normalized) for variant in variants):
            key = display_name.lower()
            if key not in seen:
                seen.add(key)
                found.append(display_name)

    for skill_name, meta in SKILL_ONTOLOGY.items():
        variants = [skill_name] + list(meta["aliases"])
        if any(re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", normalized) for variant in variants):
            display_name = next(
                (name for name, names in _DISPLAY_SKILL_PATTERNS if skill_name in names or skill_name == name.lower()),
                skill_name.title(),
            )
            key = display_name.lower()
            if key not in seen:
                seen.add(key)
                found.append(display_name)

    return found


def _extract_years_experience(text: str) -> Optional[float]:
    patterns = (
        re.compile(r"(\d+(?:\.\d+)?)\+?\s+year[s]?\s+of\s+experience", re.IGNORECASE),
        re.compile(r"(\d+(?:\.\d+)?)\+?\s+year[s]?\s+experience", re.IGNORECASE),
        re.compile(r"with\s+(\d+(?:\.\d+)?)\+?\s+year[s]?\s+of\s+experience", re.IGNORECASE),
        re.compile(r"with\s+(\d+(?:\.\d+)?)\+?\s+year[s]?\s+experience", re.IGNORECASE),
        re.compile(r"(\d+(?:\.\d+)?)\s*-\s*year[s]?\s+experience", re.IGNORECASE),
    )
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def _extract_preferred_locations(intro_lines: list[str]) -> list[str]:
    locations = []
    for line in intro_lines[:5]:
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*,\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", line):
            city = match.group(1).strip()
            if city.lower() in {"professional summary", "technical skills", "work experience"}:
                continue
            locations.append(city)
    return _dedupe(locations)


def _extract_work_preferences(text: str) -> list[str]:
    preferences = []
    for preference in ("remote", "hybrid", "onsite", "on-site"):
        if re.search(rf"(?<!\w){re.escape(preference)}(?!\w)", text, re.IGNORECASE):
            preferences.append(preference.replace("-", " ").title())
    return _dedupe(preferences)


def _extract_notable_projects(project_lines: list[str]) -> list[str]:
    projects = []
    for line in project_lines:
        clean = line.lstrip("-*â€¢ ").strip()
        if "|" in clean:
            clean = clean.split("|", 1)[0].strip()
        if len(clean.split()) >= 2 and len(clean) <= 120:
            projects.append(clean)
    return _dedupe(projects[:6])


def _parse_resume_text_locally(text: str) -> dict:
    lines = _normalize_resume_lines(text)
    sections = _split_resume_sections(lines)
    intro_lines = sections.get("intro", [])
    summary_lines = sections.get("professional_summary", [])
    skills_lines = sections.get("technical_skills", [])
    experience_lines = sections.get("experience", [])
    project_lines = sections.get("projects", [])

    intro_text = "\n".join(intro_lines)
    summary_text = " ".join(summary_lines).strip()
    skills_text = "\n".join(skills_lines)
    full_text = "\n".join(lines)

    core_skills = _extract_resume_skills(skills_text or summary_text or full_text)
    additional_skills = [
        skill for skill in _extract_resume_skills(full_text)
        if skill not in {item for item in core_skills}
    ]
    role_titles = _extract_roles_from_text("\n".join(intro_lines + summary_lines + experience_lines + project_lines))

    return {
        "candidate_name": _extract_candidate_name(lines),
        "professional_summary": summary_text,
        "role_titles": role_titles,
        "core_skills": core_skills[:15],
        "additional_skills": additional_skills[:15],
        "years_experience": _extract_years_experience(full_text),
        "preferred_locations": _extract_preferred_locations(intro_lines),
        "work_preferences": _extract_work_preferences(full_text),
        "notable_projects": _extract_notable_projects(project_lines),
        "resume_text": full_text,
        "intro_text": intro_text,
    }


def _choose_resume_artifact_name(
    artifact_names: list[str],
    requested_name: Optional[str] = None,
) -> Optional[str]:
    if not artifact_names:
        return None

    if requested_name:
        requested_key = requested_name.strip().lower()
        for artifact_name in artifact_names:
            if artifact_name.lower() == requested_key:
                return artifact_name

    for pattern in _RESUME_FILENAME_PATTERNS:
        for artifact_name in artifact_names:
            if pattern.search(artifact_name):
                return artifact_name

    return artifact_names[0]


def _resume_part_display_name(part: object, fallback_index: int = 0) -> str:
    inline_data = getattr(part, "inline_data", None)
    if inline_data is not None:
        display_name = getattr(inline_data, "display_name", None)
        if display_name:
            return str(display_name)
        mime_type = getattr(inline_data, "mime_type", None)
        if mime_type == "application/pdf":
            return f"uploaded_resume_{fallback_index}.pdf"

    file_data = getattr(part, "file_data", None)
    if file_data is not None:
        display_name = getattr(file_data, "display_name", None)
        if display_name:
            return str(display_name)
        file_uri = getattr(file_data, "file_uri", None)
        if file_uri:
            parsed_uri = artifact_util.parse_artifact_uri(str(file_uri))
            if parsed_uri:
                return parsed_uri.filename
            if str(file_uri).startswith("artifact://"):
                return str(file_uri).rsplit("/", 1)[-1]

    return f"uploaded_resume_{fallback_index}"


def _part_looks_like_resume_upload(part: object, fallback_index: int = 0) -> bool:
    inline_data = getattr(part, "inline_data", None)
    if inline_data is not None:
        mime_type = (getattr(inline_data, "mime_type", None) or "").split(";", 1)[0].strip().lower()
        if mime_type in _SUPPORTED_RESUME_MIME_TYPES:
            return True

    file_data = getattr(part, "file_data", None)
    if file_data is not None:
        mime_type = (getattr(file_data, "mime_type", None) or "").split(";", 1)[0].strip().lower()
        if mime_type in _SUPPORTED_RESUME_MIME_TYPES:
            return True
        file_uri = getattr(file_data, "file_uri", None)
        if file_uri and artifact_util.parse_artifact_uri(str(file_uri)):
            filename = _resume_part_display_name(part, fallback_index=fallback_index)
            return bool(_choose_resume_artifact_name([filename]))

    filename = _resume_part_display_name(part, fallback_index=fallback_index)
    return bool(_choose_resume_artifact_name([filename]))


def _extract_resume_parts_from_user_content(
    user_content: object,
) -> list[tuple[str, object]]:
    parts = getattr(user_content, "parts", None)
    if not isinstance(parts, list):
        return []

    candidates: list[tuple[str, object]] = []
    for index, part in enumerate(parts):
        if not _part_looks_like_resume_upload(part, fallback_index=index):
            continue
        candidates.append((_resume_part_display_name(part, fallback_index=index), part))
    return candidates


def _enrich_role_titles_from_skills(role_titles: list[str], skills: list[str]) -> list[str]:
    enriched_roles = list(role_titles)
    normalized_skills = {skill.strip().lower() for skill in skills if skill.strip()}

    has_frontend = bool(normalized_skills & _FRONTEND_ROLE_SKILLS)
    has_backend = bool(normalized_skills & _BACKEND_ROLE_SKILLS)

    if has_frontend and has_backend:
        enriched_roles.append("Full Stack Developer")
    if has_frontend:
        enriched_roles.append("Frontend Developer")
        if any(skill in normalized_skills for skill in ("react", "react.js", "next.js")):
            enriched_roles.append("React Developer")
    if has_backend:
        enriched_roles.append("Backend Developer")
        if any(skill in normalized_skills for skill in ("node.js", "node", "express", "express.js")):
            enriched_roles.append("Node.js Developer")

    return _sort_role_titles(enriched_roles)


def _normalize_resume_profile_payload(payload: dict, *, resume_source: str) -> dict:
    core_skills = _clean_resume_items(payload.get("core_skills", []))
    additional_skills = _clean_resume_items(payload.get("additional_skills", []))

    try:
        profile = ResumeProfile.model_validate({
            "candidate_name": _clean_resume_scalar(payload.get("candidate_name")),
            "professional_summary": _clean_resume_scalar(payload.get("professional_summary")),
            "role_titles": _enrich_role_titles_from_skills(
                _clean_resume_items(payload.get("role_titles", [])),
                core_skills + additional_skills,
            ),
            "core_skills": core_skills,
            "additional_skills": additional_skills,
            "years_experience": _normalize_years_experience(payload.get("years_experience")),
            "preferred_locations": _clean_resume_items(payload.get("preferred_locations", [])),
            "work_preferences": _clean_resume_items(payload.get("work_preferences", [])),
            "notable_projects": _clean_notable_projects(payload.get("notable_projects", [])),
            "resume_source": _clean_resume_scalar(resume_source) or "uploaded_resume",
        })
    except ValidationError as exc:
        first_error = exc.errors()[0]
        field_name = ".".join(str(part) for part in first_error.get("loc", ())) or "resume profile"
        raise ValueError(f"Invalid {field_name}: {first_error.get('msg', 'could not validate value')}.") from exc
    return profile.model_dump()


def _profile_has_strong_local_signal(profile: dict) -> bool:
    return bool(
        profile.get("candidate_name")
        and (
            profile.get("core_skills")
            or profile.get("role_titles")
            or profile.get("professional_summary")
        )
    )


def _extract_resume_profile_locally(
    artifact_name: str,
    artifact_part: genai_types.Part,
) -> dict:
    artifact_bytes, mime_type = _artifact_part_to_bytes(artifact_part)
    if not artifact_bytes:
        return {
            "status": "error",
            "message": f'Artifact "{artifact_name}" does not contain readable bytes.',
        }

    if mime_type == "application/pdf":
        extracted_text = _extract_text_from_pdf_bytes(artifact_bytes)
    else:
        extracted_text = _safe_decode_text(artifact_bytes)

    if not _looks_like_resume_text(extracted_text):
        return {
            "status": "error",
            "message": "Local resume text extraction was too weak to trust.",
            "resume_text": extracted_text,
        }

    parsed = _parse_resume_text_locally(extracted_text)
    try:
        profile = _normalize_resume_profile_payload(parsed, resume_source=artifact_name)
    except ValueError as exc:
        return {
            "status": "error",
            "message": str(exc),
            "resume_text": extracted_text,
        }
    return {
        "status": "ok" if _profile_has_strong_local_signal(profile) else "error",
        "profile": profile,
        "resume_text": extracted_text,
        "message": (
            "Local extraction produced a usable resume profile."
            if _profile_has_strong_local_signal(profile)
            else "Local extraction produced text, but not a strong enough profile."
        ),
    }


def _truncate_resume_text_for_llm(text: str, *, max_chars: int = 24000) -> str:
    normalized = (text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rsplit("\n", 1)[0].strip() or normalized[:max_chars].strip()


def _extract_json_object_from_text(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("The resume parser did not return valid JSON.")

    candidate = cleaned[start:end + 1]
    json.loads(candidate)
    return candidate


def _llm_response_text(response: object) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces = []
            for part in content:
                if isinstance(part, str):
                    pieces.append(part)
                    continue
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    pieces.append(part["text"])
                    continue
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    pieces.append(part_text)
            return "\n".join(piece for piece in pieces if piece).strip()
    return str(getattr(response, "text", "") or "").strip()


def _coerce_resume_parser_payload(payload: object) -> dict:
    if isinstance(payload, ResumeProfileExtraction):
        return payload.model_dump()
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("The resume parser did not return a valid JSON object.")


def _sanitize_part_for_gemini(artifact_part: genai_types.Part) -> genai_types.Part:
    inline_data = getattr(artifact_part, "inline_data", None)
    if inline_data is not None:
        return genai_types.Part(
            inline_data=genai_types.Blob(
                data=getattr(inline_data, "data", None),
                mime_type=getattr(inline_data, "mime_type", None),
            )
        )

    file_data = getattr(artifact_part, "file_data", None)
    if file_data is not None:
        return genai_types.Part(
            file_data=genai_types.FileData(
                file_uri=getattr(file_data, "file_uri", None),
                mime_type=getattr(file_data, "mime_type", None),
            )
        )

    text = getattr(artifact_part, "text", None)
    if text is not None:
        return genai_types.Part(text=text)

    return artifact_part


async def _parse_resume_with_text_model(artifact_name: str, resume_text: str) -> dict:
    prepared_text = _truncate_resume_text_for_llm(resume_text)
    if len(prepared_text) < 80:
        raise ValueError("Resume text extraction did not produce enough readable text for model parsing.")

    parser_model = resolve_resume_parser_model()
    prompt = (
        f"{_RESUME_PARSER_PROMPT}\n\n"
        f"Resume filename: {artifact_name}\n\n"
        "Resume text:\n"
        f"{prepared_text}"
    )

    if uses_litellm(parser_model):
        if litellm_acompletion is None:
            raise ValueError("LiteLLM is not installed for provider-based resume parsing.")

        response = await litellm_acompletion(
            model=parser_model,
            messages=[
                {"role": "system", "content": _RESUME_PARSER_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        response_text = _llm_response_text(response)
        return _coerce_resume_parser_payload(
            _extract_json_object_from_text(response_text)
        )

    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required for native Gemini resume parsing.")

    client = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(
        model=parser_model,
        contents=[prompt],
        config=genai_types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=ResumeProfileExtraction,
        ),
    )
    parsed = response.parsed
    if isinstance(parsed, ResumeProfileExtraction):
        return parsed.model_dump()
    if parsed is not None:
        return ResumeProfileExtraction.model_validate(parsed).model_dump()
    return ResumeProfileExtraction.model_validate_json(response.text).model_dump()


async def _parse_resume_with_gemini(artifact_name: str, artifact_part: genai_types.Part) -> dict:
    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is required for attachment-based resume extraction."
        )

    client = genai.Client(api_key=api_key)
    prompt = _RESUME_PARSER_PROMPT
    response = await client.aio.models.generate_content(
        model=resolve_resume_attachment_model(),
        contents=[prompt, _sanitize_part_for_gemini(artifact_part)],
        config=genai_types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=ResumeProfileExtraction,
        ),
    )
    parsed = response.parsed
    if isinstance(parsed, ResumeProfileExtraction):
        return parsed.model_dump()
    if parsed is not None:
        return ResumeProfileExtraction.model_validate(parsed).model_dump()
    return ResumeProfileExtraction.model_validate_json(response.text).model_dump()
