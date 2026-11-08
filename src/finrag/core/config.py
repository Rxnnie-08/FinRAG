"""Runtime settings, read from environment variables so no secret is ever stored in code."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    home: Path = field(default_factory=lambda: Path("finrag_data"))
    # SEC requires a descriptive User-Agent with contact details, e.g. "Jane Doe jane@uni.edu".
    sec_user_agent: str | None = None
    # "lsa" (offline baseline) | "st:<sentence-transformers model>" | "ollama:<model>"
    embedder: str = "lsa"
    # "none" (extractive answers) | "anthropic:<model>" | "openai:<model>@<base_url>" (Ollama, vLLM, ...)
    llm: str = "none"
    top_k: int = 6
    candidates: int = 50
    rrf_k: int = 60

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            home=Path(os.environ.get("FINRAG_HOME", "finrag_data")),
            sec_user_agent=os.environ.get("SEC_USER_AGENT") or None,
            embedder=os.environ.get("FINRAG_EMBEDDER", "lsa"),
            llm=os.environ.get("FINRAG_LLM", "none"),
            top_k=int(os.environ.get("FINRAG_TOP_K", "6")),
        )

    @property
    def index_dir(self) -> Path:
        return self.home / "index"

    @property
    def filings_dir(self) -> Path:
        return self.home / "filings"

    @property
    def manifest_path(self) -> Path:
        return self.home / "manifest.json"
