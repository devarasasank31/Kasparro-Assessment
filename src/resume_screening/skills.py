"""Technology lexicon: canonical skill names, aliases and text matching.

Keeping the vocabulary in one place lets eligibility, scoring and output share
the exact same definition of "Python", "FastAPI", "LangGraph", ... which is what
makes the final ranking explainable.
"""

from __future__ import annotations

import re
from typing import Iterable

# canonical name -> aliases (matched case-insensitively with boundaries)
VOCAB: dict[str, list[str]] = {
    # --- languages -------------------------------------------------------
    "Python": ["python", "python3", "python 3"],
    "Java": ["java", "java 8", "java 17"],
    "JavaScript": ["javascript", "js", "es6", "ecmascript"],
    "TypeScript": ["typescript", "ts"],
    "C++": ["c++", "cpp"],
    "C": ["c language"],
    "C#": ["c#", "csharp"],
    "Go": ["golang"],
    "Rust": ["rust"],
    "Kotlin": ["kotlin"],
    "Swift": ["swift"],
    "R": [" r language"],
    "SQL": ["sql", "mysql", "plsql", "pl/sql"],
    "Bash": ["bash", "shell scripting"],
    "Scala": ["scala"],
    # --- backend / web ---------------------------------------------------
    "FastAPI": ["fastapi", "fast api"],
    "Django": ["django", "django rest"],
    "Flask": ["flask"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Express.js": ["express.js", "expressjs", "express js"],
    "Spring Boot": ["spring boot", "springboot"],
    "REST APIs": ["rest api", "restful", "rest apis"],
    "GraphQL": ["graphql", "graph ql"],
    "gRPC": ["grpc"],
    "AsyncIO": ["asyncio", "async python", "async/await", "asynchronous"],
    "Celery": ["celery"],
    "RabbitMQ": ["rabbitmq", "rabbit mq"],
    "Kafka": ["kafka"],
    "Microservices": ["microservice", "micro-services"],
    # --- frontend --------------------------------------------------------
    "React": ["react", "react.js", "reactjs", "react native"],
    "Next.js": ["next.js", "nextjs", "next js"],
    "Angular": ["angular"],
    "Vue.js": ["vue.js", "vuejs"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3", "tailwind"],
    "Redux": ["redux", "zustand"],
    # --- data stores -----------------------------------------------------
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "SQLite": ["sqlite"],
    "Elasticsearch": ["elasticsearch", "elastic search"],
    "Pinecone": ["pinecone"],
    "Qdrant": ["qdrant"],
    "ChromaDB": ["chromadb", "chroma db", "chroma"],
    "Weaviate": ["weaviate"],
    "pgvector": ["pgvector", "pg vector"],
    "Vector DB": ["vector db", "vector database", "vector store"],
    # --- AI / ML ---------------------------------------------------------
    "LangChain": ["langchain", "lang chain"],
    "LangGraph": ["langgraph", "lang graph"],
    "LlamaIndex": ["llamaindex", "llama index", "llama-index"],
    "Google ADK": ["google adk", "adk", "agent development kit"],
    "Haystack": ["haystack"],
    "AutoGen": ["autogen"],
    "CrewAI": ["crewai", "crew ai"],
    "OpenAI": ["openai", "gpt-4", "gpt-3", "gpt4", "chatgpt"],
    "Anthropic": ["anthropic", "claude"],
    "Gemini API": ["gemini api", "google gemini", "bard"],
    "Hugging Face": ["hugging face", "huggingface", "hf transformers"],
    "Transformers": ["transformers", "transformer model"],
    "RAG": ["rag", "retrieval augmented", "retrieval-augmented"],
    "Embeddings": ["embedding", "embeddings", "sentence-transformers", "text-embedding"],
    "Vector Search": ["vector search", "similarity search", "cosine similarity", "semantic search", "semantic chunking"],
    "LLM": ["llm", "large language model", "language model"],
    "AI Agents": ["ai agent", "agentic", "tool-calling agent", "tool calling", "multi-agent", "multi agent", "agent workflow"],
    "Prompt Engineering": ["prompt engineering", "prompt design"],
    "Fine-tuning": ["fine-tuning", "fine tuning", "finetuning", "lora", "peft"],
    "Machine Learning": ["machine learning", "supervised learning", "unsupervised learning"],
    "Deep Learning": ["deep learning", "neural network", "cnn", "rnn", "lstm"],
    "NLP": ["nlp", "natural language processing", "text classification", "named entity"],
    "Computer Vision": ["computer vision", "opencv", "object detection", "image segmentation"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow", "tensor flow", "keras"],
    "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Speech Recognition": ["speech recognition", "stt", "text-to-speech", "tts"],
    "Evaluation Pipeline": ["evaluation pipeline", "eval pipeline", "llm evaluation", "ragas", "benchmark"],
    # --- cloud / devops --------------------------------------------------
    "Docker": ["docker", "dockerfile"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "GCP": ["gcp", "google cloud", "google cloud platform"],
    "Azure": ["azure"],
    "CI/CD": ["ci/cd", "cicd", "github actions", "jenkins", "pipeline automation"],
    "Terraform": ["terraform"],
    "Nginx": ["nginx"],
    "Linux": ["linux", "ubuntu"],
    # --- data / other ----------------------------------------------------
    "ETL": ["etl", "data pipeline", "data pipelines"],
    "Streamlit": ["streamlit"],
    "Power BI": ["power bi"],
    "Tableau": ["tableau"],
    "Selenium": ["selenium"],
    "BeautifulSoup": ["beautifulsoup", "beautiful soup"],
    "Scrapy": ["scrapy"],
    "Git": ["git", "github", "gitlab"],
    "Testing": ["pytest", "unit test", "unit testing", "test coverage", "jest", "junit", "tdd"],
}

#: Skill -> coarse category used for grouping in the output.
SKILL_CATEGORY: dict[str, str] = {
    **{k: "language" for k in ["Python", "Java", "JavaScript", "TypeScript", "C++", "C", "C#", "Go", "Rust", "Kotlin", "Swift", "R", "SQL", "Bash", "Scala"]},
    **{k: "ai_ml" for k in ["LangChain", "LangGraph", "LlamaIndex", "Google ADK", "Haystack", "AutoGen", "CrewAI", "OpenAI", "Anthropic", "Gemini API", "Hugging Face", "Transformers", "RAG", "Embeddings", "Vector Search", "LLM", "AI Agents", "Prompt Engineering", "Fine-tuning", "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "PyTorch", "TensorFlow", "Scikit-learn", "Pandas", "NumPy", "Speech Recognition", "Evaluation Pipeline"]},
    **{k: "cloud_devops" for k in ["Docker", "Kubernetes", "AWS", "GCP", "Azure", "CI/CD", "Terraform", "Nginx", "Linux"]},
    **{k: "backend" for k in ["FastAPI", "Django", "Flask", "Node.js", "Express.js", "Spring Boot", "REST APIs", "GraphQL", "gRPC", "AsyncIO", "Celery", "RabbitMQ", "Kafka", "Microservices"]},
    **{k: "frontend" for k in ["React", "Next.js", "Angular", "Vue.js", "HTML", "CSS", "Redux"]},
    **{k: "database" for k in ["PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "Elasticsearch", "Pinecone", "Qdrant", "ChromaDB", "Weaviate", "pgvector", "Vector DB"]},
}

DEFAULT_CATEGORY = "tool"

_BULLET_RE = re.compile(r"^\s*[•▪●–\-*·‣►]+")


def _compile(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias.lower())
    return re.compile(r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])")


_PATTERNS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    canonical: [(alias, _compile(alias)) for alias in aliases]
    for canonical, aliases in VOCAB.items()
    if aliases
}


def find_aliases(text: str, canonical: str) -> list[str]:
    """Return the aliases of ``canonical`` present in ``text``."""
    if not text:
        return []
    lowered = text.lower()
    compact = re.sub(r"[^a-z0-9+#.]", "", lowered)
    hits: list[str] = []
    for alias, pattern in _PATTERNS.get(canonical, []):
        if pattern.search(lowered):
            hits.append(alias)
        elif len(re.sub(r"[^a-z0-9]", "", alias)) >= 6 and re.sub(r"[^a-z0-9+#.]", "", alias) in compact:
            # PDF extraction occasionally glues words together; long distinctive
            # aliases are still safe to look for in the space-stripped text.
            hits.append(alias)
    return hits


def scan_skills(text: str) -> list[str]:
    """All canonical skills present in ``text``, in vocabulary order."""
    if not text:
        return []
    return [name for name in VOCAB if find_aliases(text, name)]


def scan_skills_with_aliases(text: str) -> dict[str, list[str]]:
    """Canonical skill -> aliases actually matched (used as score evidence)."""
    result: dict[str, list[str]] = {}
    if not text:
        return result
    for name in VOCAB:
        hits = find_aliases(text, name)
        if hits:
            result[name] = hits
    return result


def group_skills(skills: Iterable[str]) -> dict[str, list[str]]:
    """Group canonical skills into coarse categories."""
    groups: dict[str, list[str]] = {}
    for skill in skills:
        groups.setdefault(SKILL_CATEGORY.get(skill, DEFAULT_CATEGORY), []).append(skill)
    return groups


def is_bullet(line: str) -> bool:
    return bool(_BULLET_RE.match(line))


def strip_bullet(line: str) -> str:
    return _BULLET_RE.sub("", line).strip()
