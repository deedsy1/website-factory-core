import os, zipfile, re
from datetime import date

OUTPUT_ROOT = "content/pages"

def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s[:90].strip("-")

def ensure_frontmatter(md: str, fallback_title: str):
    if md.lstrip().startswith("---"):
        # ensure required keys exist
        head, rest = md.split("---", 2)[1], md.split("---", 2)[2]
        fm = head
        def has(k): return re.search(rf"^\s*{re.escape(k)}\s*:", fm, flags=re.M) is not None
        add=[]
        if not has("slug"): add.append(f"slug: \"{slugify(fallback_title)}\"")
        if not has("date"): add.append(f"date: {date.today().isoformat()}")
        if not has("hub"): add.append("hub: work-career")
        if not has("page_type"): add.append("page_type: explainer")
        if add:
            fm = fm.rstrip()+"\n" + "\n".join(add) + "\n"
        return "---\n"+fm+"---\n"+rest.lstrip("\n")
    else:
        slug = slugify(fallback_title)
        return f"""---\ntitle: \"{fallback_title}\"\nslug: \"{slug}\"\ndescription: \"\"\ndate: {date.today().isoformat()}\nhub: work-career\npage_type: explainer\n---\n\n{md.strip()}\n"""

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
