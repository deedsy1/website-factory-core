# Website Factory Core (Golden Engine)

This repo is imported by site repos via **Hugo Modules** for layouts/assets, and is checked out by GitHub Actions for generator scripts.

## Hugo Modules (layouts/assets)
Site repos should include in `hugo.yaml`:

```yaml
module:
  imports:
    - path: github.com/YOURUSER/website-factory-core
```

## GitHub Actions (generator)
Site repos run:

```bash
python _core/scripts/generate_pages.py
python _core/scripts/quality_gates.py
```

…from the **site repo root** (so generated files are written + committed to the site repo).
