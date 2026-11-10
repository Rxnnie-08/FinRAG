"""Copy the monolith into four part folders that keep final repo-relative paths (so they merge by simple copy)."""
import shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = {
    "part-a-ingestion": ["src/finrag/__init__.py", "src/finrag/core", "src/finrag/ingestion", "sample_data", "pyproject.toml", "README.md",
                         ".gitignore", "scripts", "docs/PART_A_ingestion.md", "tests/conftest.py", "tests/a_core_ingestion"],
    "part-b-retrieval": ["src/finrag/retrieval", "docs/PART_B_retrieval.md", "tests/b_retrieval"],
    "part-c-query-generation": ["src/finrag/query", "src/finrag/generation", "src/finrag/optimization", "docs/PART_C_query_generation.md", "tests/c_query_generation"],
    "part-d-evaluation-app": ["src/finrag/pipeline.py", "src/finrag/cli.py", "src/finrag/evaluation", "src/finrag/server",
                              "docs/PART_D_evaluation_app.md", "tests/d_eval_app"],
}
out = Path(sys.argv[1])
for name, items in PARTS.items():
    for it in items:
        src, dst = ROOT / it, out / name / it
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
print("wrote", ", ".join(PARTS), "to", out)
