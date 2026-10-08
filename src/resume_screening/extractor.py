"""Deterministic resume information extraction.

Layout-independent by design: resumes are split into logical sections using a
synonym table, then each section is parsed with format-tolerant heuristics.
Nothing here calls the network or an LLM.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import Candidate, Experience, Project
from .parser import ParsedResume
from .skills import group_skills, is_bullet, scan_skills, strip_bullet
from .utils import dedupe_preserve_order, get_logger

log = get_logger("extractor")

# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------
SECTION_ALIASES: dict[str, set[str]] = {
    "summary": {
        "summary", "professional summary", "profile summary", "career summary",
        "resume summary", "objective", "career objective", "about me", "profile",
        "overview", "career aspiration", "career aspirations", "professional profile",
        "about", "career objective summary",
    },
    "skills": {
        "skills", "technical skills", "tech skills", "key skills", "core skills",
        "skills & tools", "skills and tools", "skills & technologies",
        "skills and technologies", "technical expertise", "expertise",
        "technologies", "technical proficiencies", "core competencies",
        "programming skills", "skills section", "technology stack", "tech stack",
        "core technical expertise", "areas of expertise", "technical skills & tools",
    },
    "experience": {
        "experience", "work experience", "professional experience", "employment history",
        "work history", "internship", "internships", "internship experience",
        "internships & projects", "work experience & internships",
        "experience & internships", "industrial experience", "industry experience",
        "profile experience", "internship(s)", "work experience(s)",
        "professional experience & projects", "internship / training",
        "internship & training", "freelancing", "freelance experience",
        "hackathon experience", "work experience and internships",
    },
    "projects": {
        "projects", "project", "academic projects", "personal projects",
        "key projects", "technical projects", "capstone projects",
        "projects & achievements", "project work", "selected projects",
        "major projects", "project experience", "open source",
        "open source projects", "projects & internships", "side projects",
    },
    "education": {
        "education", "educational qualification", "educational qualifications",
        "educational background", "qualification", "qualifications", "academics",
        "academic background", "education & qualifications", "academic details",
    },
    "certifications": {
        "certifications", "certificates", "courses", "online courses",
        "certifications & achievements", "certification", "training",
    },
    "achievements": {
        "achievements", "achievement", "awards", "achievements & awards",
        "awards & achievements", "honors", "accomplishments", "extra-curricular",
        "extracurricular activities", "activities", "records", "position of responsibility",
    },
    "strengths": {
        "strengths", "soft skills", "personal skills", "core strengths",
        "strengths & skills", "personal details",
    },
}

_ALIAS_TO_SECTION: dict[str, str] = {
    alias: key for key, aliases in SECTION_ALIASES.items() for alias in aliases
}

#: When a merged heading (e.g. "WORK EXPERIENCE PROFESSIONAL SUMMARY") matches
#: several aliases, the section closest to the top of the pipeline wins.
_SECTION_PRIORITY = {
    "experience": 0,
    "projects": 1,
    "skills": 2,
    "education": 3,
    "certifications": 4,
    "achievements": 5,
    "summary": 6,
    "strengths": 7,
}

_HEADING_MAX_LEN = 45
_HEADING_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 &/()+,'-]*$")

# ---------------------------------------------------------------------------
# Contact patterns
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,16}\d)(?!\d)")
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9][A-Za-z0-9-]{0,38})", re.I)
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/([A-Za-z0-9\-_%./]+)", re.I)

GITHUB_RESERVED = {
    "features", "topics", "pricing", "about", "settings", "notifications", "marketplace",
    "sponsors", "login", "join", "search", "site", "orgs", "apps", "collections",
    "trending", "enterprise", "teams", "pulls", "issues", "codespaces", "account",
    "explore", "dashboard", "readme", "docs", "blog", "careers", "resources",
}

_NAME_BLOCKLIST = {
    "resume", "curriculum", "vitae", "developer", "engineer", "student", "aspirant",
    "intern", "summary", "profile", "objective", "experience", "skills", "projects",
    "education", "contact", "details", "fresher", "graduate", "undergraduate",
    "bachelor", "master", "team", "member", "technology", "technologies",
}

_DATE_RANGE_RE = re.compile(
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{2,4}"
    r"(?:\s*[-–—/]\s*(?:present|current|now|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{2,4}))?"
    r"|\d{1,2}/\d{4}\s*[-–—]\s*(?:\d{1,2}/\d{4}|present|current)",
    re.I,
)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _is_heading(line: str) -> str | None:
    """Return the section key when ``line`` looks like a resume heading.

    Handles merged headings ("WORK EXPERIENCE PROFESSIONAL SUMMARY") by falling
    back to word-boundary substring matching and picking the highest-priority
    section, then the longest matched alias.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _HEADING_MAX_LEN or is_bullet(stripped):
        return None
    if stripped.endswith(".") or stripped.count(" ") > 6:
        return None
    if not _HEADING_RE.match(stripped):
        return None
    key = stripped.rstrip(":").rstrip(".").lower()
    key = re.sub(r"\s+", " ", key)
    if ":" in key:
        return None
    exact = _ALIAS_TO_SECTION.get(key)
    if exact:
        return exact

    best: tuple[int, int, str] | None = None
    for alias, section_key in _ALIAS_TO_SECTION.items():
        if len(alias) < 6:
            continue
        if not re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", key):
            continue
        rank = (_SECTION_PRIORITY[section_key], -len(alias))
        if best is None or rank < best[:2]:
            best = (*rank, section_key)
    return best[2] if best else None


def split_sections(text: str) -> dict[str, str]:
    """Split a resume into named sections (text before the first heading = header)."""
    sections: dict[str, str] = {"header": ""}
    current = "header"
    for line in text.splitlines():
        section_key = _is_heading(line)
        if section_key:
            current = section_key
            sections.setdefault(current, "")
            continue
        sections[current] = (sections[current] + "\n" + line).rstrip("\n")
    return {key: value.strip() for key, value in sections.items() if value.strip()}


# ---------------------------------------------------------------------------
# Contact / identity
# ---------------------------------------------------------------------------
def _clean_contact_line(line: str) -> str:
    line = EMAIL_RE.sub(" ", line)
    line = re.sub(r"(?:https?://)?(?:www\.)?(?:github|linkedin|leetcode|leetcode\.com)[^\s|,;]*", " ", line, flags=re.I)
    line = PHONE_RE.sub(" ", line)
    return re.sub(r"\s+", " ", line).strip(" |,;-–—")


def _looks_like_name(candidate: str) -> bool:
    candidate = candidate.strip()
    if not (2 <= len(candidate) <= 45):
        return False
    if any(ch.isdigit() for ch in candidate):
        return False
    if "@" in candidate or "http" in candidate.lower():
        return False
    words = [w for w in re.split(r"[\s]+", candidate) if w]
    if not (1 <= len(words) <= 6):
        return False
    alpha = sum(ch.isalpha() for ch in candidate)
    if alpha < 3:
        return False
    if alpha / max(len(candidate), 1) < 0.6:
        return False
    lowered = candidate.lower()
    if any(blocked in lowered for blocked in _NAME_BLOCKLIST):
        return False
    if not candidate[0].isupper():
        return False
    return True


def extract_name(header: str, source_file: str) -> tuple[str, list[str]]:
    """Best-effort candidate name from the header block."""
    warnings: list[str] = []
    for raw in header.splitlines()[:12]:
        cleaned = _clean_contact_line(raw)
        if not cleaned:
            continue
        for part in re.split(r"\s*[|•·,]\s*|\s+[-–—]\s+", cleaned):
            if _looks_like_name(part):
                return part.strip(), warnings
    warnings.append("Name not confidently detected; fell back to file name")
    return Path(source_file).stem, warnings


def extract_github(text: str) -> str | None:
    for match in GITHUB_RE.finditer(text):
        username = match.group(1)
        if username.lower().removesuffix(".git") in GITHUB_RESERVED:
            continue
        return f"https://github.com/{username}"
    return None


def extract_email(text: str) -> str | None:
    match = EMAIL_RE.search(text)
    return match.group(0) if match else None


def extract_phone(text: str) -> str | None:
    for match in PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if 7 <= len(digits) <= 15:
            return match.group(0).strip()
    return None


def extract_linkedin(text: str) -> str | None:
    match = LINKEDIN_RE.search(text)
    if not match:
        return None
    handle = match.group(1).rstrip("/")
    return f"https://www.linkedin.com/in/{handle}"


# ---------------------------------------------------------------------------
# Skills section
# ---------------------------------------------------------------------------
_ITEM_SPLIT_RE = re.compile(r"\s*[,;|·••/]\s*|\s+&\s+|\s+and\s+")


def parse_skills_section(section: str) -> dict[str, list[str]]:
    """Parse a skills section into ``{category: [items]}``."""
    groups: dict[str, list[str]] = {}
    current = "General"
    for raw in section.splitlines():
        line = strip_bullet(raw).strip()
        if not line:
            continue
        if ":" in line:
            prefix, _, remainder = line.partition(":")
            if 0 < len(prefix) <= 40 and not prefix.strip().isdigit():
                current = prefix.strip().rstrip(".")
                line = remainder.strip()
                if not line:
                    groups.setdefault(current, [])
                    continue
        items = [item.strip(" .") for item in _ITEM_SPLIT_RE.split(line)]
        clean = [item for item in items if item and len(item) <= 45]
        if clean:
            groups.setdefault(current, []).extend(clean)
    return {k: dedupe_preserve_order(v) for k, v in groups.items() if v}


# ---------------------------------------------------------------------------
# Projects / experience blocks
# ---------------------------------------------------------------------------
_DESC_START_RE = re.compile(
    r"^(built|implemented|developed|designed|integrated|created|used|worked|automated|"
    r"deployed|optimized|optimised|reduced|enhanced|engineered|added|configured|managed|"
    r"tested|collaborated|responsible|the |this |a |an |our |its |for |with |by |based )",
    re.I,
)


def _starts_new_block(line: str, previous: str | None, has_current: bool) -> bool:
    if is_bullet(line):
        return not has_current
    if not has_current:
        return True
    if previous is not None and is_bullet(previous):
        return True
    if "|" in line or "—" in line or " – " in line:
        return True
    if len(line) <= 90 and not _DESC_START_RE.match(line):
        return True
    if previous is not None and previous.strip() == "":
        return True
    return False


def _split_blocks(section: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    previous: str | None = None
    for raw in section.splitlines():
        line = raw.rstrip()
        if not line.strip():
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            previous = ""
            continue
        if _starts_new_block(line.strip(), previous, bool(current)):
            if current:
                blocks.append("\n".join(current).strip())
            current = [line.strip()]
        else:
            current.append(line.strip())
        previous = line.strip()
    if current:
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


def _title_and_techs(block: str, first_line: str) -> tuple[str, list[str]]:
    """Split a project header line into a name and its technologies."""
    line = strip_bullet(first_line).strip()
    if "|" in line:
        head, _, tail = line.partition("|")
        techs = [part.strip() for part in re.split(r"[,;/]| and ", tail) if part.strip()]
        return head.strip(" .|-–—"), [t for t in techs if len(t) <= 40]
    for separator in (" — ", " – ", " - "):
        if separator in line:
            head, _, tail = line.partition(separator)
            if tail and len(head) < len(line) * 0.7:
                techs = [part.strip() for part in re.split(r"[,;/]| and ", tail) if part.strip()]
                if techs and all(len(t) <= 40 for t in techs):
                    return head.strip(), techs
    return line.strip(" .|-–—"), []


def parse_projects(section: str) -> list[Project]:
    projects: list[Project] = []
    for block in _split_blocks(section):
        lines = block.splitlines()
        first = strip_bullet(lines[0]).strip()
        name, techs = _title_and_techs(block, first)
        if len(name) > 90:
            name = name[:90].rstrip()
        bullets = [strip_bullet(line) for line in lines[1:] if is_bullet(line)]
        body_lines = [strip_bullet(line) for line in lines[1:]]
        summary = " ".join(body_lines).strip()
        if techs:
            derived = []
        else:
            derived = scan_skills(block)
        projects.append(
            Project(
                name=name or "Untitled project",
                technologies=techs,
                summary=summary[:700],
                bullets=bullets[:8],
                raw=block,
            )
        )
        if derived and not techs:
            projects[-1].technologies = derived[:12]
    return projects


def parse_experiences(section: str) -> list[Experience]:
    experiences: list[Experience] = []
    for block in _split_blocks(section):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        first = strip_bullet(lines[0]).strip()
        period_match = _DATE_RANGE_RE.search(block)
        period = period_match.group(0).strip() if period_match else ""
        title = first
        organization = ""
        if len(lines) > 1 and not is_bullet(lines[1]):
            candidate_org = strip_bullet(lines[1]).strip()
            if 0 < len(candidate_org) <= 70 and not _DATE_RANGE_RE.fullmatch(candidate_org):
                organization = candidate_org
                title = first
        title = re.sub(r"\s*[\|–—-]\s*" + re.escape(period), "", title).strip() if period else title
        bullets = [strip_bullet(line) for line in lines if is_bullet(line)]
        experiences.append(
            Experience(
                title=title[:120],
                organization=organization,
                period=period,
                bullets=bullets[:10],
                raw=block,
            )
        )
    return experiences


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_candidate(parsed: ParsedResume) -> Candidate:
    """Extract a :class:`Candidate` from a parsed resume. Never raises."""
    text = parsed.text
    warnings: list[str] = []

    if not text:
        return Candidate(
            source_file=parsed.filename,
            name=Path(parsed.filename).stem,
            extraction_warnings=["No text available for extraction"],
        )

    sections = split_sections(text)
    header = sections.get("header", "")
    name, name_warnings = extract_name(header, parsed.filename)
    warnings.extend(name_warnings)

    email = extract_email(text)
    phone = extract_phone(text)
    github_url = extract_github(text)
    linkedin_url = extract_linkedin(text)

    if not email:
        warnings.append("No email address found")
    if not github_url:
        warnings.append("No GitHub profile found")

    skill_groups = parse_skills_section(sections.get("skills", ""))
    section_skills = [item for items in skill_groups.values() for item in items]
    vocab_skills = scan_skills(text)
    skills = dedupe_preserve_order(vocab_skills + section_skills)

    projects = parse_projects(sections.get("projects", ""))[:20]
    experiences = parse_experiences(sections.get("experience", ""))[:15]
    education = [
        strip_bullet(line).strip()
        for line in sections.get("education", "").splitlines()
        if line.strip()
    ][:8]

    summary = sections.get("summary", "").strip()
    if not summary:
        summary = " ".join(header.splitlines()[:3]).strip()
    summary = summary[:700]

    if not projects:
        warnings.append("No projects section detected")
    if not experiences:
        warnings.append("No experience section detected")

    candidate = Candidate(
        source_file=parsed.filename,
        name=name,
        email=email,
        phone=phone,
        github_url=github_url,
        linkedin_url=linkedin_url,
        summary=summary,
        skills=skills[:60],
        skill_groups=skill_groups,
        projects=projects,
        experiences=experiences,
        education=education,
        sections=sections,
        full_text=text,
        extraction_warnings=warnings,
    )
    log.debug(
        "Extracted %s: %d skills, %d projects, %d experiences",
        candidate.name,
        len(candidate.skills),
        len(candidate.projects),
        len(candidate.experiences),
    )
    return candidate
