import os, zipfile, re
from datetime import date

OUTPUT_ROOT = "content/pages"

def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s[:90].strip("-")

def ensure_frontmatter(md: str, fallback_title: str) -> str:
    """Ensure markdown has valid YAML frontmatter.

    If the file starts with '---' but is missing a closing delimiter, treat it as malformed and
    replace with a fresh frontmatter block (preserving the body as best we can).
    """
    text = (md or "").lstrip("\ufeff")  # strip BOM if present
    if text.strip().startswith("---"):
        parts = text.split("---", 2)
        # Expected: ['', '\n<yaml>\n', '<body>...']
        if len(parts) >= 3:
            fm = parts[1].strip()
            body = parts[2].lstrip("\n")
            if "title:" not in fm:
                fm = f"title: {yaml_quote(fallback_title)}\n" + fm
            return "---\n" + fm.rstrip() + "\n---\n\n" + body
        # malformed frontmatter (no closing ---); fall through and rebuild
        text = text.lstrip("-").lstrip()

    # no frontmatter (or malformed) -> create it
    fm = f"title: {yaml_quote(fallback_title)}\n"
    body = text.lstrip()
    return "---\n" + fm + "---\n\n" + body + ("\n" if not body.endswith("\n") else "")


def main(zip_path: str):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.startswith("pages/") and n.endswith(".md")]
        if not names:
            raise SystemExit("No pages/*.md files found in zip.")
        for n in names:
            raw = z.read(n).decode("utf-8", errors="ignore")
            fallback_title = os.path.splitext(os.path.basename(n))[0].replace("-", " ").title()
            md = ensure_frontmatter(raw, fallback_title)
            # determine slug folder from url if present
            m = re.search(r'^\s*url:\s*["\']?(/[^"\']+)["\']?\s*$', md, flags=re.M)
            if m:
                url = m.group(1).strip("/")
                slug = url.split("/")[-1] if url else slugify(fallback_title)
            else:
                slug = slugify(fallback_title)
            out_dir = os.path.join(OUTPUT_ROOT, slug)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "index.md"), "w", encoding="utf-8") as f:
                f.write(md.strip()+"\n")
    print(f"Imported {len(names)} pages into {OUTPUT_ROOT}/")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/import_kimi_zip.py <zip_path>")
    main(sys.argv[1])
