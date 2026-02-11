import os
import sys
import argparse
import re
import json
import time
import hashlib
import random
from pathlib import Path

import requests
import yaml

# --- site-root support (early) --------------------------------------------
# Some paths are defined at import time, so we apply --site-root before that.
def _apply_site_root_early():
    """Allow running core scripts from inside thin-repo.

    Priority:
      1) --site-root <path>
      2) --site-slug <slug>  -> chdir sites/<slug>
      3) SITE_SLUG env       -> chdir sites/<slug>
      4) slugify(BOOTSTRAP_NICHE or NICHE env) -> chdir sites/<slug>

    If a slug is chosen, sites/<slug> is created if missing.
    """

    def _slugify(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "-", s)
        s = re.sub(r"-+", "-", s).strip("-")
        return s

    # 1) explicit --site-root
    if "--site-root" in sys.argv:
        i = sys.argv.index("--site-root")
        if i + 1 < len(sys.argv) and sys.argv[i + 1]:
            os.chdir(sys.argv[i + 1])
            return

    # 2) explicit --site-slug
    slug = ""
    if "--site-slug" in sys.argv:
        i = sys.argv.index("--site-slug")
        if i + 1 < len(sys.argv):
            slug = (sys.argv[i + 1] or "").strip()

    # 3) env SITE_SLUG
    if not slug:
        slug = (os.getenv("SITE_SLUG") or "").strip()

    # 4) derive from niche envs
    if not slug:
        slug = _slugify(os.getenv("BOOTSTRAP_NICHE") or os.getenv("NICHE") or "")

    if slug:
        root = Path("sites") / slug
        root.mkdir(parents=True, exist_ok=True)
        os.chdir(root)
        return

_apply_site_root_early()

def _sr(rel: str) -> Path:
    """Resolve a path relative to the current working directory."""
    return (Path.cwd() / rel).resolve()

# ---------------------------------------------------------------------------

BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1").rstrip("/")
API_KEY = os.environ.get("MOONSHOT_API_KEY", "")
MODEL = os.getenv("KIMI_MODEL", "kimi-k2.5")

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

SITE_PATH = Path(os.getenv("SITE_CONFIG", "data/site.yaml"))
HUGO_PATH = Path("hugo.yaml")
TITLES_POOL_PATH = _sr("scripts/titles_pool.txt")
MANIFEST_PATH = Path("scripts/manifest.json")

TITLE_COUNT = int(os.getenv("TITLE_COUNT", "300"))
PAGES_NOW = int(os.getenv("PAGES_NOW", "0"))  # optional: for bootstrap workflow convenience

# Inputs from Actions workflow
NICHE = (os.getenv("BOOTSTRAP_NICHE") or "").strip()
TONE = (os.getenv("BOOTSTRAP_TONE") or "").strip()

THEME_PACKS = [
  "calm-paper","charcoal-gold","clinic-clean","earthy-trail","editorial","forest-hush",
  "lavender-dusk","maker","matcha-cream","midnight-plum","minimal-mono","modern-sans",
  "night-ink","ocean-mist","playful-soft","ruby-graphite","sandstone","steel-blue",
  "sunset-clay","warm-sunrise"
]

# Contract defaults (kept stable unless you deliberately change them)
DEFAULT_OUTLINE_H2 = [
  "Intro",
  "Definitions and key terms",
  "Why this topic exists",
  "How people usually experience this",
  "How it typically works",
  "When this topic tends to come up",
  "Clarifying examples",
  "Common misconceptions",
  "Why this topic gets misunderstood online",
  "Related situations that feel similar",
  "Related topics and deeper reading",
  "Neutral summary",
  "FAQs",
]

def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s[:60].strip("-") or "site"

def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def save_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

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
        raise json.JSONDecodeError("No JSON object found in model output", raw, 0)
    return json.loads(m.group(0))

def kimi_json(system: str, user: str, temperature: float = 1.0, max_tokens: int = 1400) -> dict:
    if not API_KEY:
        raise RuntimeError("MOONSHOT_API_KEY is not set")

    payload = {
        "model": MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
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
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt == HTTP_MAX_TRIES - 1:
                raise
            sleep = min(60.0, (BACKOFF_BASE ** attempt) + random.random())
            print(f"HTTP error: {type(e).__name__} — retrying in {sleep:.1f}s")
            time.sleep(sleep)
            continue

        # Success
        if r.status_code < 400:
            content = r.json()["choices"][0]["message"]["content"]
            return parse_json_strict_or_extract(content)

        # Retryable server/rate-limit errors
        if r.status_code in (408, 429, 500, 502, 503, 504):
            last_err = f"HTTP {r.status_code}: {r.text[:4000]}"
            if attempt == HTTP_MAX_TRIES - 1:
                break
            sleep = min(60.0, (BACKOFF_BASE ** attempt) + random.random())
            print(f"HTTP {r.status_code} — retrying in {sleep:.1f}s")
            time.sleep(sleep)
            continue

        # Non-retryable
        last_err = f"HTTP {r.status_code}: {r.text[:4000]}"
        break

    raise RuntimeError(last_err or "Moonshot API retries exhausted")

def ensure_manifest_reset():
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps({"used_titles": [], "generated_this_run": []}, indent=2), encoding="utf-8")

def write_titles_pool(titles: list[str]):
    TITLES_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    uniq = []
    seen = set()
    for t in titles:
        t = (t or "").strip()
        if not t:
            continue
        s = slugify(t)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(t)
    TITLES_POOL_PATH.write_text("\n".join(uniq) + "\n", encoding="utf-8")

def patch_hugo_yaml(site_cfg: dict):
    """Keep hugo.yaml minimal but aligned to site identity for Cloudflare Pages."""
    if not HUGO_PATH.exists():
        return
    cfg = yaml.safe_load(HUGO_PATH.read_text(encoding="utf-8")) or {}

    site = site_cfg.get("site", {}) if isinstance(site_cfg, dict) else {}
    brand = site.get("brand") or site.get("title") or cfg.get("title") or "Site"
    base_url = site.get("base_url") or cfg.get("baseURL") or "https://YOUR-SITE.pages.dev/"
    lang = site.get("language_code") or cfg.get("languageCode") or "en-us"

    cfg["baseURL"] = str(base_url)
    cfg["languageCode"] = str(lang)
    cfg["title"] = str(site.get("title") or brand)

    params = cfg.get("params") or {}
    factory = params.get("factory") or {}
    factory["brand"] = str(brand)
    params["factory"] = factory

    # Ads defaults remain controlled via data/site.yaml + templates
    cfg["params"] = params

    HUGO_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

def main():
    if not NICHE:
        raise SystemExit("BOOTSTRAP_NICHE is required (e.g. 'work anxiety', 'caravan towing safety', etc).")

    existing = load_yaml(SITE_PATH)

    # Bootstrap contract constants
    system = (
        "You are a careful site-bootstrapper for an evergreen informational website.\n"
        "Hard rules:\n"
        "- No dates/years or time-sensitive words (recent/currently/this year/today/now).\n"
        "- No prices/cost claims, no statistics, no 'studies show', no numbers-as-facts.\n"
        "- No medical/legal/financial advice. No guarantees. No first-person.\n"
        "- Output JSON only.\n"
    )

    user = {
        "task": "Create site identity + theme choice + titles pool for an evergreen website.",
        "inputs": {
            "niche": NICHE,
            "tone": TONE or "neutral, calm, beginner-friendly",
            "title_count": TITLE_COUNT,
        },
        "allowed_theme_packs": THEME_PACKS,
        "required_json": {
            "site_title": "string (2-4 words, no punctuation)",
            "brand": "string (same as title or shorter)",
            "tagline": "string (8-14 words, no hype, no promises)",
            "default_meta_description": "string (<= 155 chars, neutral)",
            "theme_pack": "one of allowed_theme_packs",
            "hubs": [
                {"id": "work-career|money-stress|burnout-load|milestones|social-norms", "label": "string"}
            ],
            "titles_pool": ["list of unique page titles, question-style, evergreen, global-friendly"],
        },
        "notes": [
            "Titles must avoid dates/years, prices, stats, brand names, and advice framing.",
            "Prefer novice-friendly, definitional and comparison topics.",
            "Keep titles short and specific; no clickbait."
        ],
    }

    out = kimi_json(system=system, user=json.dumps(user, ensure_ascii=False), temperature=1.0, max_tokens=MAX_OUTPUT_TOKENS)

    theme_pack = out.get("theme_pack")
    if theme_pack not in THEME_PACKS:
        # deterministic fallback
        theme_pack = "modern-sans"

    site_title = (out.get("site_title") or "Evergreen Site").strip()
    brand = (out.get("brand") or site_title).strip()
    tagline = (out.get("tagline") or "Calm, practical explanations — not advice.").strip()
    meta = (out.get("default_meta_description") or tagline).strip()

    # Keep existing base_url if user already deployed a site; otherwise leave placeholder
    base_url = (existing.get("site", {}) or {}).get("base_url") if isinstance(existing, dict) else None
    if not base_url:
        # allow passing from workflow input (optional)
        base_url = (os.getenv("BOOTSTRAP_BASE_URL") or "https://YOUR-SITE.pages.dev/").strip()

    hubs = out.get("hubs") or (existing.get("taxonomy", {}) or {}).get("hubs") or [
        {"id": "work-career", "label": "Work & Career"},
        {"id": "money-stress", "label": "Money & Stress"},
        {"id": "burnout-load", "label": "Burnout & Load"},
        {"id": "milestones", "label": "Milestones"},
        {"id": "social-norms", "label": "Social Norms"},
    ]

    # Apply the locked wordcount (quality strict, not padding strict)
    wc_min = 900
    wc_max = 1900
    ideal_min = 1100
    ideal_max = 1600

    site_cfg = existing if isinstance(existing, dict) else {}
    site_cfg.setdefault("site", {})
    site_cfg.setdefault("theme", {})
    site_cfg.setdefault("taxonomy", {})
    site_cfg.setdefault("generation", {})
    site_cfg.setdefault("internal_linking", {})
    site_cfg.setdefault("ads", {})
    site_cfg.setdefault("gates", {})

    site_cfg["site"].update({
        "title": site_title,
        "brand": brand,
        "language_code": site_cfg["site"].get("language_code") or "en-us",
        "base_url": base_url,
        "default_meta_description": meta,
        "tagline": tagline,
        "niche": NICHE,
    })

    site_cfg["theme"].update({
        "pack": theme_pack,
        # keep existing fonts/spacing defaults unless changed elsewhere
        "font_sans": site_cfg["theme"].get("font_sans") or "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, Arial, sans-serif",
        "font_serif": site_cfg["theme"].get("font_serif") or "ui-serif, Georgia, Cambria, 'Times New Roman', Times, serif",
        "content_max": site_cfg["theme"].get("content_max") or "74ch",
        "radius": site_cfg["theme"].get("radius") or "16px",
    })

    site_cfg["taxonomy"]["hubs"] = hubs

    gen = site_cfg["generation"]
    gen.setdefault("forbidden_words", [])
    # keep existing forbidden words; ensure core set
    core_forbidden = [
        "diagnose", "diagnosis", "prescribed", "guaranteed", "sue",
        "treatment", "treat", "cure", "therapist", "lawyer", "accountant",
    ]
    merged = list(dict.fromkeys((gen.get("forbidden_words") or []) + core_forbidden))
    gen["forbidden_words"] = merged
    gen["page_types"] = gen.get("page_types") or ["is-it-normal", "checklist", "red-flags", "myth-vs-reality", "explainer"]
    gen["outline_h2"] = DEFAULT_OUTLINE_H2
    gen["wordcount"] = {"min": wc_min, "ideal_min": ideal_min, "ideal_max": ideal_max, "max": wc_max}

    # Keep internal linking strict
    il = site_cfg["internal_linking"]
    il.setdefault("enabled", True)
    il["min_links"] = max(int(il.get("min_links") or 3), 3)
    il["forbid_external"] = True

    # Gates mirror wordcount + strictness
    gates = site_cfg["gates"]
    gates["wordcount_min"] = wc_min
    gates["wordcount_max"] = wc_max
    gates["min_internal_links"] = 3
    gates["forbid_external_links"] = True

    save_yaml(SITE_PATH, site_cfg)
    patch_hugo_yaml(site_cfg)

    titles = out.get("titles_pool") or []
    write_titles_pool(titles)

    ensure_manifest_reset()

    # Write a small bootstrap receipt for debugging/panel consumption later
    receipt = {
        "niche": NICHE,
        "tone": TONE,
        "site_title": site_title,
        "theme_pack": theme_pack,
        "title_count_written": len(titles),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract_hash": hashlib.sha256(("|".join(DEFAULT_OUTLINE_H2) + f"|{wc_min}-{wc_max}").encode("utf-8")).hexdigest()[:16],
    }
    rc = Path("scripts/bootstrap_receipt.json")
    if not rc.exists():
        rc.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    print("\n===== BOOTSTRAP SUMMARY =====")
    print(f"Niche: {NICHE}")
    print(f"Site title: {site_title}")
    print(f"Theme pack: {theme_pack}")
    print(f"Titles written: {len(titles)} (target {TITLE_COUNT})")
    print("Reset: manifest.json")
    print("=============================\n")

if __name__ == "__main__":
    main()
