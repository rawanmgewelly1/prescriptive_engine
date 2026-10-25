# GitHub publication checklist

This ZIP is a *local full-data research delivery*, not a safe one-click public Git repository.

- Review source data/model/notebook redistribution rights and remove any confidential data before public release.
- `.gitignore` excludes `data/*.csv`, model `.pkl` files, source notebooks and generated outputs, so ordinary Git commits remain small.
- If you legally want to publish artifacts, use Git LFS with tracked patterns for `.pkl` and the **319 MB** Wind CSV; check GitHub/LFS size, quota and bandwidth limits. Do not blindly force-add data without the user's approval.
- Publish `run_pipeline.py`, `prescriptive_engine/pipeline/`, `configs/scenarios/`, `tests/`, `requirements.txt`, README and `docs/`.
- Do not claim pretrained models/entire pipeline are externally scientifically certified.
- Add a license only after rights are determined.
- Fresh clone without models is expected to report missing artifacts, not fabricate results.

## Recommended commands (after review)

```bash
git init
git add .
git status
git commit -m "Organize research prescriptive engine with explicit scenario contracts"
```
