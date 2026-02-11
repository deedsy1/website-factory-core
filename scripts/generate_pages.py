import os
import sys
import argparse
import json
import time
import re
import random
import hashlib
import requests
from datetime import date
from pathlib import Path
import yaml

START_TIME = time.time()


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "site"


def _apply_site_root_early():
    """Allow running core scripts from inside thin-repo.

    Priority:
      1) --site-root <path>
      2) --site-slug <slug>  -> chdir sites/<slug>
      3) SITE_SLUG env (if set) -> chdir sites/<SITE_SLUG>

    If slug is blank, we derive it from BOOTSTRAP_NICHE/NICHE.
    """
    # 1) explicit --site-root
    if "--site-root" in sys.argv:
        i = sys.argv.index("--site-root")
        if i + 1 < len(sys.argv) and sys.argv[i + 1]:
            os.chdir(sys.argv[i + 1])
            return

    # 2) --site-slug
    slug = ""
    if "--site-slug" in sys.argv:
        i = sys.argv.index("--site-slug")
        if i + 1 < len(sys.argv):
            slug = (sys.argv[i + 1] or "").strip()

    # 3) env
    if not slug:
        slug = (os.getenv("SITE_SLUG", "") or "").strip()

    if not slug:
        niche = (os.getenv("BOOTSTRAP_NICHE", "") or os.getenv("NICHE", "")).strip()
        slug = _slugify(niche)

    if slug:
        root = Path("sites") / slug
        root.mkdir(parents=True, exist_ok=True)
        os.chdir(root)


_apply_site_root_early()


def _parse_site_root() -> Path:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--site-root", default=".")
    p.add_argument("--site-slug", default="")
    # We only care about site-root here; everything else is handled later.
    args, _ = p.parse_known_args()
    return Path(args.site_root).resolve()

SITE_ROOT = Path(_parse_site_root()).resolve()

def _sr(rel: str) -> Path:
    return (SITE_ROOT / rel).resolve()


BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1").rstrip("/")
API_KEY = os.environ["MOONSHOT_API_KEY"]
MODEL = os.getenv("KIMI_MODEL", "kimi-k2.5")

# Reliability knobs (env)
# Keep these shared across bootstrap/factory so a single tuning works everywhere.
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))
CONNECT_TIMEOUT = int(os.getenv("CONNECT_TIMEOUT", "20"))
HTTP_MAX_TRIES = int(os.getenv("KIMI_HTTP_MAX_TRIES", "6"))
BACKOFF_BASE = float(os.getenv("KIMI_BACKOFF_BASE", "1.7"))
FAIL_STOP = int(os.getenv("FAIL_STOP", "6"))  # stop after N consecutive title failures

PAGES_PER_RUN = int(os.getenv("PAGES_PER_RUN", "10"))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "25"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1600"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "1"))
SLEEP_SECONDS = float(os.getenv("SLEEP_SECONDS", "0.3"))

CONTENT_ROOT = _sr("content/pages")
MANIFEST_PATH = _sr("scripts/manifest.json")
TITLES_POOL_PATH = _sr("scripts/titles_pool.txt")
RETRY_TITLES_PATH = _sr("scripts/retry_titles.txt")
FAILED_TITLES_PATH = _sr("scripts/failed_titles.txt")
CACHE_DIR = _sr("scripts/cache")
SITE_CONFIG_PATH = str(_sr(os.getenv("SITE_CONFIG", "data/site.yaml")))

# Queue / plan
PLAN_PATH = str(_sr(os.getenv("PLAN_PATH", "data/plan.yaml")))

# Factory modes
FACTORY_MODE = os.getenv("FACTORY_MODE", "generate").strip().lower()  # generate|regen
REGEN_RULE = os.getenv("REGEN_RULE", "").strip()  # e.g. "version_lt:2" or "contract_mismatch"
REGEN_HUB = os.getenv("REGEN_HUB", "").strip()
REGEN_SLUGS = os.getenv("REGEN_SLUGS", "").strip()  # comma-separated
GEN_VERSION = int(os.getenv("GEN_VERSION", "2"))
BACKFILL_METADATA = os.getenv("BACKFILL_METADATA", "1").strip() == "1"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

def resolve_site_config_path() -> str:
    """Prefer the single contract at data/site.yaml.
    Backward compatible: fall back to scripts/site_config.yaml if needed.
    """
    p = SITE_CONFIG_PATH
    if os.path.isfile(p):
        return p
    fallback = "scripts/site_config.yaml"
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(f"Site config not found: {p} (and no {fallback} fallback)")

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def ensure_manifest_shape(m: dict) -> dict:
    if not isinstance(m, dict):
        return {"used_titles": [], "generated_this_run": []}
    m.setdefault("used_titles", [])
    m.setdefault("generated_this_run", [])
    return m

def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return {"used_titles": [], "generated_this_run": []}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return ensure_manifest_shape(json.load(f))

def save_manifest(m):
    m = ensure_manifest_shape(m)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

def slugify(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s[:80].strip("-")

def load_titles():
    with open(TITLES_POOL_PATH, "r", encoding="utf-8") as f:
        return [t.strip() for t in f if t.strip()]

def load_plan(path: str) -> dict:
    if not os.path.isfile(path):
        return {"items": []}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"items": []}

def save_plan(path: str, plan: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(plan or {"items": []}, f, sort_keys=False, allow_unicode=True)

def parse_json_strict_or_extract(raw: str) -> dict:
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    raw2 = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw2 = re.sub(r"\s*```$", "", raw2)
    try:
        return json.loads(raw2)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise json.JSONDecodeError("No JSON object found", raw, 0)
    return json.loads(m.group(0))

def call_kimi(system: str, prompt: str):
    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }

    last_err = None
    for attempt in range(HTTP_MAX_TRIES):
        try:
            r = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=HEADERS,
                json=payload,
                timeout=(CONNECT_TIMEOUT, REQUEST_TIMEOUT),
            )
        except requests.RequestException as e:
            last_err = str(e)
            # treat network errors like retryable failures
            sleep_s = min(60.0, (BACKOFF_BASE ** attempt)) + random.uniform(0, 0.25)
            time.sleep(sleep_s)
            continue
        if r.status_code < 400:
            return r.json()["choices"][0]["message"]["content"]

        if r.status_code in (408, 429, 500, 502, 503, 504):
            sleep_s = min(60.0, (BACKOFF_BASE ** attempt)) + random.uniform(0, 0.25)
            time.sleep(sleep_s)
            continue

        last_err = r.text
        break

    raise RuntimeError(last_err or "API retries exhausted")


def build_internal_link_hints(content_root: str = "content/pages", limit: int = 40) -> str:
    """
    Build a curated list of existing internal links for the model to use.
    Format: - [Title](/pages/slug/)
    """
    root = Path(content_root)
    items = []
    for md in root.glob("*/index.md"):
        try:
            raw = md.read_text(encoding="utf-8")
        except Exception:
            continue
        fm, _ = read_frontmatter(raw)
        slug = fm.get("slug") or md.parent.name
        title = fm.get("title") or slug.replace("-", " ").title()
        items.append((str(title).strip(), str(slug).strip()))
    # Stable shuffle
    items = sorted(items, key=lambda x: x[1])
    # Take last N (newer slugs tend to be later alphabetically? doesn't matter)
    items = items[:limit]
    return "\n".join([f"- [{t}](/pages/{s}/)" for t, s in items if t and s])


def build_prompts(cfg: dict):
    # data/site.yaml is the single contract.
    site = cfg.get("site", {}) if isinstance(cfg, dict) else {}
    taxonomy = cfg.get("taxonomy", {}) if isinstance(cfg, dict) else {}
    generation = cfg.get("generation", {}) if isinstance(cfg, dict) else {}

    brand = site.get("brand") or site.get("title") or "Reality Checks"
    hubs = [h.get("id") for h in (taxonomy.get("hubs") or []) if isinstance(h, dict) and h.get("id")] or [
        "work-career", "money-stress", "burnout-load", "milestones", "social-norms"
    ]
    page_types = generation.get("page_types") or [
        "is-it-normal", "checklist", "red-flags", "myth-vs-reality", "explainer"
    ]

    wc = generation.get("wordcount") or {}
    wc_min = int(wc.get("min") or 900)
    wc_ideal_min = int(wc.get("ideal_min") or 1100)
    wc_ideal_max = int(wc.get("ideal_max") or 1600)
    wc_max = int(wc.get("max") or 1900)

    forbidden = generation.get("forbidden_words") or []
    forbidden_str = ", ".join(forbidden) if forbidden else "diagnose, diagnosis, prescribed, guaranteed, sue"

    outline = generation.get("outline_h2") or [
        "What this feeling usually means",
        "Common reasons",
        "What makes it worse",
        "What helps (non-advice)",
        "When it might signal a bigger issue",
        "FAQs",
    ]
    outline_md = "\n".join([f"## {h}" for h in outline])

    closing_templates = generation.get("closing_reassurance_templates") or []
    closing_hint = ""
    if closing_templates:
        closing_hint = "Choose ONE closing reassurance line in a similar style to these:\n- " + "\n- ".join(closing_templates[:3])

    system = f"""You write calm, reassuring evergreen content for the site "{brand}".
NO medical, legal, or financial advice. Avoid diagnosing. Avoid giving instructions like a professional.
Forbidden words/phrases: {forbidden_str}.
Return JSON only. Do not wrap in markdown fences.
"""

    page_prompt = f"""Return ONLY JSON with:
title
summary (one sentence reassurance; also used as meta description)
description (<= 160 chars, no quotes)
hub (one of: { " | ".join(hubs) })
page_type (one of: { " | ".join(page_types) })
closing_reassurance (one short, gentle line; NOT advice)
body_md (markdown only; must include the exact H2 headings below)

Use these H2 sections exactly:
{outline_md}

Rules:
- Neutral, encyclopedic tone (beginner-friendly). No hype, no fear framing.
- No medical, legal, or financial advice.
- No dates or time-sensitive language (no years, “recent”, “currently”, “this year”, “today”, “now”).
- No prices, costs, or financial claims.
- No guarantees/promises (“always”, “never”, “100%”, “will definitely”, “guarantee”).
- No first-person language (“I”, “we”, “our”, “my”).
- No calls-to-action or directive language (“you should”, “try this”, “make sure to”, “sign up”, “buy”, “download”).
- No affiliate/product review language (affiliate, sponsored, review, coupon, discount).
- Comparisons must be neutral (avoid superlatives like “best”, “worst”, “better than”).
- Short paragraphs: 2–3 sentences max.
- Use ONLY H2 (##) and H3 (###) headings. No H1, no H4+.
- Include at least 3 contextually relevant internal links using ONLY relative URLs like /pages/<slug>/ (no external links).
- Wordcount: minimum {wc_min} words, target {wc_ideal_min}–{wc_ideal_max}, maximum {wc_max}.

Return ONLY JSON with:
title
summary (one sentence reassurance; also used as meta description)
description (<= 160 chars, no quotes)
hub (one of: { " | ".join(hubs) })
page_type (one of: { " | ".join(page_types) })
closing_reassurance (one short, gentle line; NOT advice)
body_md (markdown only; must include the exact H2 headings below)

Use these H2 sections exactly:
{outline_md}

Rules:
- Keep tone grounded and human, not clinical.
- No "diagnose/diagnosis/prescribed/guaranteed/sue".
- FAQs: 4-6 Q&As (short).
- Do not include the closing reassurance inside body_md; put it in closing_reassurance.
{closing_hint}
"""
    return system, page_prompt

def choose_close(data: dict, cfg: dict) -> str:
    close = (data.get("closing_reassurance") or "").strip()
    if close:
        return close

    templates = (cfg.get("generation", {}) or {}).get("closing_reassurance_templates") or []
    if templates:
        return random.choice(templates).strip()
    return "If this hit close to home, you’re not alone — and you’re not failing."

def compute_contract_hash(site_config_path: str) -> str:
    try:
        with open(site_config_path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()
    except Exception:
        return "unknown"

def read_markdown_frontmatter(md_text: str):
    """
    Returns (frontmatter_dict, body_text_without_frontmatter)
    """
    if not md_text.startswith("---"):
        return {}, md_text
    parts = md_text.split("\n---\n", 2)
    if len(parts) < 3:
        return {}, md_text
    fm_raw = parts[1]
    body = parts[2]
    try:
        fm = yaml.safe_load(fm_raw) or {}
        if not isinstance(fm, dict):
            fm = {}
    except Exception:
        fm = {}
    return fm, body

def write_markdown_with_frontmatter(front: dict, body: str) -> str:
    fm_txt = yaml.safe_dump(front or {}, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{fm_txt}\n---\n\n{body.lstrip() if body else ''}"

def iter_content_pages(root_dir: str) -> list[str]:
    pages = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "index.md" in filenames:
            pages.append(os.path.join(dirpath, "index.md"))
    return pages

def backfill_page_metadata(content_root: str, contract_hash: str) -> int:
    updated = 0
    for path in iter_content_pages(content_root):
        try:
            raw = open(path, "r", encoding="utf-8").read()
            fm, body = read_markdown_frontmatter(raw)
            if not fm:
                continue
            changed = False
            if "gen_version" not in fm:
                fm["gen_version"] = GEN_VERSION
                changed = True
            if "contract_hash" not in fm or str(fm.get("contract_hash")) != str(contract_hash):
                fm["contract_hash"] = contract_hash
                changed = True
            if "prompt_hash" not in fm:
                # We only set a placeholder here; new writes will set a real prompt_hash
                fm["prompt_hash"] = "backfilled"
                changed = True
            if changed:
                open(path, "w", encoding="utf-8").write(write_markdown_with_frontmatter(fm, body))
                updated += 1
        except Exception:
            continue
    return updated

def parse_regen_rule(rule: str) -> dict:
    rule = (rule or "").strip()
    if not rule:
        return {}
    if ":" in rule:
        k, v = rule.split(":", 1)
        return {"type": k.strip(), "value": v.strip()}
    return {"type": rule, "value": ""}

def select_pages_for_regen(content_root: str, contract_hash: str) -> list[dict]:
    """
    Returns list of dicts: {path, fm}
    """
    targets = []
    slugs_set = set([s.strip() for s in REGEN_SLUGS.split(",") if s.strip()]) if REGEN_SLUGS else set()
    rule = parse_regen_rule(REGEN_RULE)
    for path in iter_content_pages(content_root):
        try:
            raw = open(path, "r", encoding="utf-8").read()
            fm, _ = read_markdown_frontmatter(raw)
            if not fm:
                continue
            slug = str(fm.get("slug") or "").strip()
            hub = str(fm.get("hub") or "").strip()
            gv = fm.get("gen_version", 0)
            try:
                gv = int(gv)
            except Exception:
                gv = 0

            # Explicit selection wins
            if slugs_set:
                if slug and slug in slugs_set:
                    targets.append({"path": path, "fm": fm})
                continue
            if REGEN_HUB:
                if hub.lower() == REGEN_HUB.lower():
                    targets.append({"path": path, "fm": fm})
                continue

            # Rule-based selection
            if not rule:
                continue
            rtype = rule.get("type")
            rval = rule.get("value")
            if rtype == "version_lt":
                try:
                    n = int(rval)
                except Exception:
                    n = 0
                if gv < n:
                    targets.append({"path": path, "fm": fm})
            elif rtype == "contract_mismatch":
                if str(fm.get("contract_hash", "")) != str(contract_hash):
                    targets.append({"path": path, "fm": fm})
        except Exception:
            continue

    return targets

def generate_one_page(title: str, system: str, page_prompt: str, cfg: dict, pinned_hub: str = "", pinned_page_type: str = ""):
    """
    Returns (ok, data_dict). data_dict should include title, summary, description, hub, page_type, body_md.
    """
    extra = ""
    if pinned_hub:
        extra += f"\nHub (must use exactly): {pinned_hub}"
    if pinned_page_type:
        extra += f"\nPage type (must use exactly): {pinned_page_type}"

    try:
        raw = call_kimi(system, f"{page_prompt}\n\nTitle: {title}{extra}")
        data = parse_json_strict_or_extract(raw)
    except Exception:
        return False, {}

    body = (data.get("body_md") or "").strip()
    required_h2 = (cfg.get("generation", {}) or {}).get("outline_h2", [])
    if required_h2:
        missing = [h for h in required_h2 if f"## {h}" not in body]
        if missing:
            return False, {}
    else:
        if body.count("## ") < 6:
            return False, {}

    required = ["title", "summary", "description", "hub", "page_type"]
    if any((k not in data or not str(data[k]).strip()) for k in required):
        return False, {}

    data["body_md"] = body
    return True, data

def write_page(slug: str, data: dict, close: str, contract_hash: str, prompt_hash: str) -> None:
    page_dir = os.path.join(CONTENT_ROOT, slug)
    os.makedirs(page_dir, exist_ok=True)

def _remove_one(path: Path, title: str) -> None:
    if not path.exists():
        return
    lines = [l.rstrip("\n") for l in path.read_text(encoding="utf-8").splitlines()]
    out = []
    removed = False
    for l in lines:
        if (not removed) and l.strip() == title:
            removed = True
            continue
        out.append(l)
    path.write_text("\n".join(out).strip() + ("\n" if out else ""), encoding="utf-8")


def mark_failed(title: str, reason: str) -> None:
    FAILED_TITLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FAILED_TITLES_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{title}\t{reason}\n")
    # move to retry list (de-duped)
    existing = set()
    if RETRY_TITLES_PATH.exists():
        existing = {l.strip() for l in RETRY_TITLES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()}
    if title not in existing:
        with RETRY_TITLES_PATH.open("a", encoding="utf-8") as f:
            f.write(title + "\n")
    # remove from pool so we don't spin
    _remove_one(TITLES_POOL_PATH, title)


def mark_done(title: str) -> None:
    # remove from both retry + pool
    _remove_one(RETRY_TITLES_PATH, title)
    _remove_one(TITLES_POOL_PATH, title)

    def esc(s: str) -> str:
        return str(s).replace('"', r'\\"').strip()

    body = (data.get("body_md") or "").strip()

    md = f"""---
title: "{esc(data['title'])}"
slug: "{slug}"
summary: "{esc(data['summary'])}"
description: "{esc(data['description'])}"
date: "{date.today().isoformat()}"
hub: "{esc(data['hub'])}"
page_type: "{esc(data['page_type'])}"
gen_version: "{GEN_VERSION}"
contract_hash: "{contract_hash}"
prompt_hash: "{prompt_hash}"
---

**{esc(data['summary'])}**

{body}

---

*{esc(close)}*
"""
    with open(os.path.join(page_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write(md)

def main():
    site_cfg_path = resolve_site_config_path()
    cfg = load_yaml(site_cfg_path)
    system, page_prompt = build_prompts(cfg)


    # Performance budgets: stop generating if we've hit the cap.
    max_pages = int(os.getenv("PERF_MAX_PAGES", str(cfg.get("gates", {}).get("max_pages", 1000))))
    existing_pages = len(list(CONTENT_ROOT.glob("*/index.md")))
    if existing_pages >= max_pages:
        print(f"✅ Page cap reached ({existing_pages}/{max_pages}). Nothing to do.")
        return

    # Provide internal link candidates so the model can reliably include them
    link_hints = build_internal_link_hints(CONTENT_ROOT, limit=40)
    if link_hints:
        page_prompt = page_prompt + "\n\nInternal links you MAY use (choose at least 3; do not invent links; no external links):\n" + link_hints + "\n"

    os.makedirs(CONTENT_ROOT, exist_ok=True)
    contract_hash = compute_contract_hash(site_cfg_path)
    manifest = load_manifest()

    if BACKFILL_METADATA:
        backfilled = backfill_page_metadata(CONTENT_ROOT, contract_hash)
        if backfilled:
            print(f"[metadata] backfilled gen_version/contract_hash/prompt_hash on {backfilled} pages")

    prompt_hash = hashlib.sha1((system + "\n" + page_prompt).encode("utf-8")).hexdigest()

    # Regen mode: rewrite existing pages deterministically by rule/slug/hub.
    if FACTORY_MODE == "regen":
        targets = select_pages_for_regen(CONTENT_ROOT, contract_hash)
        if not targets:
            print("[regen] no pages matched the regeneration criteria")
            return

        print(f"[regen] matched {len(targets)} pages; regenerating up to {PAGES_PER_RUN}")

        regen_count = 0
        attempts = 0

        for t in targets:
            if regen_count >= PAGES_PER_RUN or attempts >= MAX_ATTEMPTS:
                break
            fm = t["fm"]
            title = str(fm.get("title") or "").strip()
            slug = str(fm.get("slug") or "").strip()
            hub = str(fm.get("hub") or "").strip()
            page_type = str(fm.get("page_type") or "").strip()
            if not title or not slug:
                continue

            attempts += 1
            print(f"[regen] {slug}: {title}")

            ok, data = generate_one_page(
                title=title,
                system=system,
                page_prompt=page_prompt,
                cfg=cfg,
                pinned_hub=hub,
                pinned_page_type=page_type,
            )
            if not ok:
                continue

            close = choose_close(data, cfg)
            write_page(slug=slug, data=data, close=close, contract_hash=contract_hash, prompt_hash=prompt_hash)

            regen_count += 1
            manifest.setdefault("generated_this_run", []).append(slug)
            time.sleep(SLEEP_SECONDS)

        save_manifest(manifest)
        return

    # Generate mode: consume plan todos first, else fall back to titles_pool (legacy).
    plan = load_plan(PLAN_PATH)
    plan_items = plan.get("items", []) if isinstance(plan, dict) else []
    todo_items = [it for it in plan_items if isinstance(it, dict) and str(it.get("status", "todo")).lower() == "todo"]

    titles = []
    if todo_items:
        titles = [it.get("title", "").strip() for it in todo_items if it.get("title")]
    else:
        titles = load_titles()
        random.shuffle(titles)

    produced = 0
    attempts = 0
    retries = 0
    deletes = 0
    consec_fail = 0

    manifest["generated_this_run"] = []
    used = set(manifest.get("used_titles", []))

    per_title_fail = {}
    PER_TITLE_CAP = int(os.getenv("PER_TITLE_CAP", "2"))

    for title in titles:
        if produced >= PAGES_PER_RUN or attempts >= MAX_ATTEMPTS:
            break

        # If the plan provides an explicit slug, respect it.
        plan_item = None
        if todo_items:
            for it in todo_items:
                if it.get("title", "").strip() == title and str(it.get("status", "todo")).lower() == "todo":
                    plan_item = it
                    break

        slug = (plan_item.get("slug") if isinstance(plan_item, dict) and plan_item.get("slug") else None) or slugify(title)

        if slug in used:
            continue

        if per_title_fail.get(slug, 0) >= PER_TITLE_CAP:
            continue

        attempts += 1

        pinned_hub = ""
        pinned_type = ""
        if isinstance(plan_item, dict):
            pinned_hub = str(plan_item.get("hub") or "").strip()
            pinned_type = str(plan_item.get("page_type") or "").strip()

        ok, data = generate_one_page(
            title=title,
            system=system,
            page_prompt=page_prompt,
            cfg=cfg,
            pinned_hub=pinned_hub,
            pinned_page_type=pinned_type,
        )
        if not ok:
            deletes += 1
            consec_fail += 1
            per_title_fail[slug] = per_title_fail.get(slug, 0) + 1
            if consec_fail >= FAIL_STOP:
                print(f"\nStop early: hit {consec_fail} consecutive failures (FAIL_STOP={FAIL_STOP}).")
                break
            continue

        close = choose_close(data, cfg)
        write_page(slug=slug, data=data, close=close, contract_hash=contract_hash, prompt_hash=prompt_hash)

        # Mark plan item done (idempotent queue), if used.
        if isinstance(plan_item, dict):
            plan_item["slug"] = slug
            plan_item["status"] = "done"
            plan_item["generated_date"] = date.today().isoformat()

        produced += 1
        consec_fail = 0
        used.add(slug)
        manifest.setdefault("used_titles", []).append(slug)
        manifest.setdefault("generated_this_run", []).append(slug)
        time.sleep(SLEEP_SECONDS)

    save_manifest(manifest)

    # Persist plan progress.
    if todo_items:
        save_plan(PLAN_PATH, plan)

    duration = int(time.time() - START_TIME)
    print("\n===== FACTORY SUMMARY =====")
    print(f"Pages attempted: {attempts}")
    print(f"Pages produced: {produced}")
    print(f"Retries: {retries}")
    print(f"Deletes: {deletes}")
    print(f"Duration: {duration // 60}m {duration % 60}s")
    print("===========================\n")

if __name__ == "__main__":
    main()
