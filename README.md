# Website Factory Core (Golden Engine)

This repo contains the **shared engine** for all sites in your Website Factory portfolio:

- Hugo layouts, partials, and theme packs
- Generator scripts (bootstrap, generate, quality gates)
- The stable content contract + compliance rules

## How sites use this core

### 1) Hugo Modules (layouts + static)
In each *site repo* `hugo.yaml`:

```yaml
module:
  imports:
    - path: github.com/deedsy1/website-factory-core
```

Cloudflare Pages builds the site repo normally; Hugo fetches this module during build.

### 2) GitHub Actions (generator scripts)
In each *site repo* GitHub Action:

- checkout the site repo
- checkout this core repo into `_core/`
- run:
  - `python _core/scripts/bootstrap_site.py`
  - `python _core/scripts/generate_pages.py`
  - `python _core/scripts/quality_gates.py`

Run those commands **from the site repo root** so content is written + committed to the site repo.

## What does NOT live here
Site-specific assets belong in each site repo:
- `content/`
- `data/site.yaml`
- `data/plan.yaml`
- `scripts/titles_pool.txt`
- `scripts/manifest.json`

See `CORE_USAGE.md` for quick copy/paste.
