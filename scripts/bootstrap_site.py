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

# Network + retry knobs (override via workflow env vars)
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))
CONNECT_TIMEOUT = int(os.getenv("CONNECT_TIMEOUT", "20"))

# Max tokens we allow Kimi/Moonshot to return for bootstrap artifacts.
# (Workflows set this via env; default keeps responses reasonably small.)
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1600"))

# Some Moonshot/Kimi models only accept temperature=1 (integer).
# We still read TEMPERATURE for compatibility, but clamp to a safe value.
try:
    _t_raw = os.getenv("TEMPERATURE", "1").strip()
    TEMPERATURE = int(float(_t_raw))
except Exception:
    TEMPERATURE = 1

# Hard safety clamp: if a model rejects non-1 values, keep the workflow moving.
if TEMPERATURE != 1:
    TEMPERATURE = 1
HTTP_MAX_TRIES = int(os.getenv("KIMI_HTTP_MAX_TRIES", "6"))
BACKOFF_BASE = float(os.getenv("KIMI_BACKOFF_BASE", "1.7"))

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
    """Parse a JSON object from model output.

    Accepts:
      - A raw JSON object string
      - JSON wrapped in code fences
      - Prose + embedded JSON object
      - Tool/function-call argument payloads (string or dict)

    Raises json.JSONDecodeError on failure so callers can retry safely.
    """
    if raw is None:
        raw = ""

    # Some providers return dict/objects for tool-call arguments.
    if isinstance(raw, (dict, list)):
        try:
            return raw if isinstance(raw, dict) else (raw[0] if raw and isinstance(raw[0], dict) else json.loads(json.dumps(raw)))
        except Exception:
            raw = json.dumps(raw)

    raw = str(raw)

    # Normalize weird whitespace / nulls.
    raw = raw.replace("\ufeff", "").replace("\x00", "").strip()

    # Provider hiccups can occasionally yield an empty string. Treat this as
    # a parse failure so the caller can retry the request.
    if not raw:
        raise json.JSONDecodeError("Empty model output (expected JSON object)", raw, 0)

    def _strip_code_fences(s: str) -> str:
        s2 = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.I)
        s2 = re.sub(r"\s*```\s*$", "", s2.strip())
        return s2.strip()

    cand = _strip_code_fences(raw)

    # 1) strict parse
    try:
        obj = json.loads(cand)
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            return obj[0]
        if not isinstance(obj, dict):
            raise json.JSONDecodeError("Top-level JSON must be an object", cand, 0)
        return obj
    except json.JSONDecodeError:
        pass

    # 2) Extract the first balanced JSON object from the text.
    # This is safer than a greedy regex when braces appear in prose.
    s = cand
    start = s.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found in model output", cand, 0)

    depth = 0
    in_str = False
    esc = False
    end = None
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

    if end is None:
        raise json.JSONDecodeError("JSON object appears truncated (missing closing brace)", cand, 0)

    snippet = s[start:end].strip()
    snippet = _strip_code_fences(snippet)

    try:
        obj = json.loads(snippet)
        if not isinstance(obj, dict):
            raise json.JSONDecodeError("Top-level JSON must be an object", snippet, 0)
        return obj
    except json.JSONDecodeError as e:
        # Last resort: try a smaller match if the extracted snippet still fails.
        raise e


from typing import Optional, Dict, Any


def _safe_write_kimi_dump(kind: str, attempt: int, *, content: str = "", envelope: Optional[Dict[str, Any]] = None, http_status: Optional[int] = None, http_text: Optional[str] = None, payload: Optional[Dict[str, Any]] = None):
    """Write a debug dump to help diagnose provider hiccups.

    Never includes API keys. Keeps content size bounded.
    """
    try:
        log_dir = Path("scripts") / "_logs" / "kimi"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        rid = hashlib.sha1(f"{ts}-{kind}-{attempt}-{random.random()}".encode("utf-8")).hexdigest()[:10]
        p = log_dir / f"{ts}-{kind}-a{attempt+1}-{rid}.json"

        def _trim(s: str, n: int) -> str:
            s = "" if s is None else str(s)
            s = s.replace("\x00", "")
            return s[:n]

        safe_payload = None
        if isinstance(payload, dict):
            # shallow copy and redact any auth-like fields
            safe_payload = {}
            for k, v in payload.items():
                if k.lower() in ("api_key", "authorization"):
                    safe_payload[k] = "[REDACTED]"
                else:
                    safe_payload[k] = v

        dump = {
            "kind": kind,
            "attempt": attempt + 1,
            "model": MODEL,
            "temperature": (payload or {}).get("temperature"),
            "max_tokens": (payload or {}).get("max_tokens"),
            "http_status": http_status,
            "content_preview": _trim(content, 4000),
            "http_text_preview": _trim(http_text, 4000),
            "payload": safe_payload,
            "envelope": envelope,
        }
        p.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # Never let logging break the pipeline.
        return


def kimi_json(system: str, user: str, temperature: float = 1.0, max_tokens: int = 1400) -> dict:
    if not API_KEY:
        raise RuntimeError("MOONSHOT_API_KEY is not set")

    temp = temperature
    if "kimi" in (MODEL or "").lower():
        temp = 1

    payload = {
        "model": MODEL,
        "temperature": temp,
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

        # Success (HTTP), but we may still get non-JSON content on rare provider hiccups.
        if r.status_code < 400:
            try:
                data = r.json()
                msg = (data.get("choices", [{}])[0].get("message") or {})

                content = msg.get("content")

                # Newer/OpenAI-style envelopes may return content as a list of parts.
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict):
                            # Common shapes: {type:'text', text:'...'} or {text:'...'}
                            if "text" in part and isinstance(part.get("text"), str):
                                parts.append(part.get("text"))
                            elif "content" in part and isinstance(part.get("content"), str):
                                parts.append(part.get("content"))
                        elif isinstance(part, str):
                            parts.append(part)
                    content = "".join(parts).strip()

                if content is None:
                    content = ""

                # Some providers/models return tool/function calls with JSON in "arguments"
                # instead of plain text in "content".
                if not str(content).strip():
                    tool_calls = msg.get("tool_calls") or []
                    if isinstance(tool_calls, list) and tool_calls:
                        fn = (tool_calls[0].get("function") or {})
                        content = fn.get("arguments") or ""

                if not str(content).strip():
                    fn_call = (msg.get("function_call") or {})
                    content = fn_call.get("arguments") or ""

                # Some providers put the raw JSON under a different key (rare).
                if not str(content).strip() and isinstance(msg.get("json"), (dict, str)):
                    content = msg.get("json")

                content = content if content is not None else ""
                content = str(content)

            except Exception as e:
                last_err = f"Bad JSON response envelope: {type(e).__name__}: {e}"
                if attempt == HTTP_MAX_TRIES - 1:
                    raise
                sleep = min(60.0, (BACKOFF_BASE ** attempt) + random.random())
                print(f"Response parse error — retrying in {sleep:.1f}s")
                time.sleep(sleep)
                continue

            try:
                return parse_json_strict_or_extract(content)
            except json.JSONDecodeError as e:
                # Tighten prompt + reduce temperature and retry.
                last_err = f"Model JSON decode error: {e}"
                preview = (content or "").strip().replace("\n", " ")[:240]
                print(f"Model returned non-JSON content (attempt {attempt+1}/{HTTP_MAX_TRIES}): {preview}")
                _safe_write_kimi_dump("json_decode", attempt, content=content, envelope=data, http_status=r.status_code, http_text=r.text, payload=payload)

                # Strengthen system instruction once; keep idempotent.
                payload["messages"][0]["content"] = (
                    system
                    + "\n\nCRITICAL: Output MUST be a single valid JSON object. "
                      "No markdown. No code fences. No commentary."
                )
                # Some Moonshot/Kimi models only accept temperature=1.
                if "kimi" in (MODEL or "").lower():
                    payload["temperature"] = 1
                else:
                    payload["temperature"] = min(payload.get("temperature", 1.0), 0.2)
                if attempt == HTTP_MAX_TRIES - 1:
                    raise
                sleep = min(60.0, (BACKOFF_BASE ** attempt) + random.random())
                print(f"Retrying after non-JSON output in {sleep:.1f}s")
                time.sleep(sleep)
                continue

        # Retryable server/rate-limit errors
        if r.status_code in (408, 429, 500, 502, 503, 504):
            last_err = f"HTTP {r.status_code}: {r.text[:4000]}"
            if attempt == HTTP_MAX_TRIES - 1:
                break
            sleep = min(60.0, (BACKOFF_BASE ** attempt) + random.random())
            print(f"HTTP {r.status_code} — retrying in {sleep:.1f}s")
            time.sleep(sleep)
            continue

        # Some models only accept temperature=1; if we accidentally sent another value, fix and retry.
        if r.status_code == 400 and 'invalid temperature' in (r.text or '').lower():
            payload['temperature'] = 1
            last_err = f"HTTP 400 (temperature): {r.text}"
            time.sleep(0.5)
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

def main(site_slug: str = "", force_reset: bool = False):
    """Bootstrap a new site (or re-run safely).

    If `site_slug` is provided, we operate under sites/<site_slug>/... to keep
    multiple sites isolated.
    """

    # Work within the site folder if a slug is provided (keeps runs isolated)
    if site_slug:
        site_root = Path("sites") / site_slug
        site_root.mkdir(parents=True, exist_ok=True)
        os.chdir(site_root)

    # Ensure expected folders exist in the *current* site directory
    Path("scripts").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    if not NICHE:
        raise SystemExit("BOOTSTRAP_NICHE is required (e.g. 'work anxiety', 'caravan towing safety', etc).")

    # Optional destructive reset
    if force_reset:
        for p in (Path(SITE_PATH), Path(TITLES_POOL_PATH), Path(MANIFEST_PATH)):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    # If this site is already bootstrapped, do nothing (idempotent + cheap)
    if Path(SITE_PATH).exists() and Path(TITLES_POOL_PATH).exists() and not force_reset:
        print(f"[bootstrap] Existing bootstrap detected for '{site_slug or Path.cwd().name}'. Skipping LLM calls.")
        return

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

    out = kimi_json(
        system=system,
        user=json.dumps(user, ensure_ascii=False),
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

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
    print("Receipt: scripts/bootstrap_receipt.json")
    print("=============================\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-slug", default="", help="Folder under sites/, e.g. home-espresso-basics")
    ap.add_argument(
        "--force-reset",
        action="store_true",
        help="Wipe existing site.yaml / titles pool / bootstrap receipt for this site before bootstrapping",
    )
    args = ap.parse_args()
    main(site_slug=args.site_slug.strip(), force_reset=bool(args.force_reset))
