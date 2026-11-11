# How to push the four parts

Each folder holds files at their final repo paths, so merging is a copy. Use one shared repo with four branches.

1. Part A owner: create the GitHub repo, then
   `cd part-a-ingestion && git init -b main && git add . && git commit -m "Part A: core + ingestion" && git remote add origin <url> && git push -u origin main`
2. Parts B, C, D owners: clone the repo, `git checkout -b part-b-retrieval`, copy the contents of your folder over the clone
   (`cp -r part-b-retrieval/. <clone>/`), commit, push the branch, open a pull request into `main`.
3. Merge order: A, then B and C, then D. Each part's tests need the earlier parts present:
   A `pytest tests/a_core_ingestion`; B adds `tests/b_retrieval`; C adds `tests/c_query_generation`; D runs the whole suite (91 tests).
4. After all merges: `pip install -e ".[dev]" && pytest && finrag demo && finrag serve`.

Do not edit `core/schema.py` outside Part A's branch without telling the others; every part imports it.
