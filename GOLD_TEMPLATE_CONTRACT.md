# Gold Template Contract (Do Not Break)

This repo is the canonical evergreen factory template. Future sites should be created by using this repo as a **GitHub Template**.

## Must remain true
- Hugo builds on Cloudflare Pages without extra tooling
- Generated pages include frontmatter keys:
  - title, slug, summary, description, date, hub, page_type
- Generator outputs a fixed H2 outline + an FAQ section
- Quality gates delete failures safely (no broken deploys)
- Internal links are only to existing pages (no hallucinated slugs)
- Dark mode toggle uses CSS variables + localStorage
- Minimal JS only

## Allowed to change per site
- Brand + home copy
- hub taxonomy labels/ids
- CSS variables (fonts, spacing, colors)
- titles pool + prompts

## Risky changes should be done on a branch first
If it can break builds or deployments, do it on a branch and merge only after Cloudflare + Actions are green.
